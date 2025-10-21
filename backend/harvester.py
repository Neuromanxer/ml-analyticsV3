from __future__ import annotations

# ============================== ml_intake.py ===============================
# Drop-in, pasteable module for intake & normalization with gap priors
# ==========================================================================
import warnings
# stdlib
import csv
import json
import math
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

# 3rd-party
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from pandas.api.types import is_datetime64_any_dtype, is_timedelta64_dtype

# ---- Optional deps (fail-soft)
try:
    import chardet  # type: ignore
except Exception:
    chardet = None  # not required

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class DialectInfo:
    encoding: str
    delimiter: str
    quotechar: str
    has_header: bool
    decimal: str
    thousands: Optional[str]
    confidence: Optional[float] = None


@dataclass
class TypeInfo:
    semantic: str            # id, email, phone, url, ip, json, boolean, timestamp, numeric, money, percent, categorical, text
    pandas_dtype: str        # df[col].dtype.str
    unique_ratio: float
    valid_rate: float        # share of non-nulls that validate the semantic
    notes: Optional[str] = None

@dataclass
class IntakeMeta:
    dialect: DialectInfo
    header_map: Dict[str, ColumnMeta]
    inferred_types: Dict[str, TypeInfo]
    timezone_inference: Optional[str]
    number_format: Dict[str, Optional[str]]    # {'decimal':'.', 'thousands':','}
    currency_symbols: Dict[str, int]           # counts per symbol
    json_columns: Dict[str, List[str]]         # original json col -> expanded subcolumns (if parsed)
    nested_csv_columns: List[str]
    anomalies: List[str]
    data_gaps: Dict[str, Any] = field(default_factory=dict)
    assumptions: Dict[str, Any] = field(default_factory=dict)
    priors: Dict[str, Any] = field(default_factory=dict)
    prior_provenance: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IntakeResult:
    df_raw: pd.DataFrame
    df_normalized: pd.DataFrame
    meta: IntakeMeta

# -----------------------------
# Priors & proxy evidence
# -----------------------------
# role -> (alpha, beta) i.e., Beta(alpha, beta)
INDUSTRY_PRIORS: Dict[str, Tuple[float, float]] = {
    "unsubscribe": (2.0, 98.0),   # mean ~2%
    "discount":    (3.0, 97.0),   # mean ~3%
    "returns":     (2.0, 198.0),  # mean ~1%
    "referral":    (3.0, 97.0),   # mean ~3% of revenue/users
}

def _beta_mean(a: float, b: float) -> float:
    return a / (a + b) if (a + b) > 0 else 0.0

def _beta_ci(a: float, b: float, conf: float = 0.95) -> Tuple[float, float]:
    # Wilson-ish approximation if scipy not installed
    n = a + b
    p = _beta_mean(a, b)
    if n <= 2:
        return (max(0.0, p - 0.2), min(1.0, p + 0.2))
    z = 1.96 if conf >= 0.95 else 1.645
    denom = 1 + z**2 / n
    center = (p + z**2/(2*n)) / denom
    half = z * math.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
    lo, hi = max(0.0, center - half), min(1.0, center + half)
    return (lo, hi)

def _proxy_counts_unsubscribe(df: pd.DataFrame) -> Tuple[int, int]:
    cand = [c for c in df.columns if any(k in c.lower() for k in ["unsub", "opt_out", "optout", "dnc"])]
    for c in cand:
        s = df[c].astype(str).str.lower()
        pos = s.isin(["1","true","t","yes","y"]).sum()
        neg = s.isin(["0","false","f","no","n"]).sum()
        if pos + neg >= 20:
            return int(pos), int(neg)
    return 0, 0

def _proxy_counts_returns(df: pd.DataFrame) -> Tuple[int, int]:
    cand = [c for c in df.columns if any(k in c.lower() for k in ["return", "refund", "chargeback"])]
    for c in cand:
        s = df[c].astype(str).str.lower()
        pos = s.isin(["1","true","t","yes","y"]).sum()
        neg = s.isin(["0","false","f","no","n"]).sum()
        if pos + neg >= 20:
            return int(pos), int(neg)
    amt_cols = [c for c in df.columns if c.endswith("_num") or c.endswith("_usd")]
    if amt_cols:
        x = pd.to_numeric(df[amt_cols[0]], errors="coerce")
        pos = int((x < 0).sum())
        neg = int((x >= 0).sum())
        if pos + neg >= 100:
            return max(1, pos), max(1, neg)
    return 0, 0

def _proxy_counts_discount(df: pd.DataFrame) -> Tuple[int, int]:
    cand = [c for c in df.columns if "discount" in c.lower() and (c.endswith("_num") or c.endswith("_rate"))]
    for c in cand:
        x = pd.to_numeric(df[c], errors="coerce")
        pos = int((x.fillna(0) > 0).sum())
        neg = int((x.fillna(0) <= 0).sum())
        if pos + neg >= 20:
            return pos, neg
    return 0, 0

def _proxy_counts_referral(df: pd.DataFrame) -> Tuple[int, int]:
    cand = [c for c in df.columns if any(k in c.lower() for k in ["referral", "referred_by", "invite", "promo_code"])]
    for c in cand:
        s = df[c].astype(str)
        pos = int(s.notna().sum() - (s == "").sum())
        neg = int(len(s) - pos)
        if pos + neg >= 20:
            return pos, neg
    return 0, 0

PROXY_MAP: Dict[str, Callable[[pd.DataFrame], Tuple[int, int]]] = {
    "unsubscribe": _proxy_counts_unsubscribe,
    "returns":     _proxy_counts_returns,
    "discount":    _proxy_counts_discount,
    "referral":    _proxy_counts_referral,
}

def build_gap_priors(
    df_norm: pd.DataFrame,
    meta: IntakeMeta,
    *,
    industry_priors: Dict[str, Tuple[float, float]] = INDUSTRY_PRIORS,
    tenant_overrides: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """
    Produces Beta(a,b) for each gap role with mean & 95% CI, plus provenance.
    Hierarchy:
      1) tenant_overrides (if provided)
      2) industry_priors
      3) update with observed proxies -> posterior
    """
    tenant_overrides = tenant_overrides or {}
    results: Dict[str, Any] = {}
    provenance: Dict[str, Any] = {}

    gaps = (getattr(meta, "data_gaps", {}) or {}).keys()
    roles = set(list(gaps) + list(industry_priors.keys()))

    for role in roles:
        a0, b0 = tenant_overrides.get(role, industry_priors.get(role, (1.0, 99.0)))
        prov: Dict[str, Any] = {"base": "tenant" if role in tenant_overrides else "industry", "a0": a0, "b0": b0}

        pos, neg = (0, 0)
        if role in PROXY_MAP:
            try:
                pos, neg = PROXY_MAP[role](df_norm)
            except Exception:
                pos, neg = (0, 0)

        weight = {"unsubscribe":1.0, "returns":0.8, "discount":0.6, "referral":0.5}.get(role, 0.5)
        a = a0 + weight * pos
        b = b0 + weight * neg

        mean = _beta_mean(a, b)
        lo, hi = _beta_ci(a, b)

        results[role] = {
            "alpha": float(a), "beta": float(b),
            "mean": float(mean), "ci95": [float(lo), float(hi)],
            "n_proxy_pos": int(pos), "n_proxy_neg": int(neg),
            "proxy_weight": float(weight),
        }
        prov["proxy_counts"] = {"pos": pos, "neg": neg}
        prov["weight"] = weight
        provenance[role] = prov

    meta.priors = results
    meta.prior_provenance = provenance
    return results

def get_rate(meta: IntakeMeta, role: str) -> Dict[str, Any]:
    """
    Returns a dict with final point estimate + CI and provenance for UI/receipts.
    Priority: user override (assumptions) > posterior mean from priors+proxies.
    """
    override_key_map = {
        "unsubscribe": "assume_unsubscribed_rate",
        "discount":    "assume_discount_rate",
        "returns":     "assume_return_loss_rate",
        "referral":    "assume_referral_revenue_share",
    }
    override_key = override_key_map.get(role)
    override_val = None
    if override_key and isinstance(meta.assumptions, dict) and override_key in meta.assumptions:
        try:
            override_val = float(meta.assumptions[override_key])
        except Exception:
            override_val = None

    pr = (meta.priors or {}).get(role, {"mean": 0.0, "ci95": [0.0, 0.0], "alpha": 1.0, "beta": 99.0})
    val = float(override_val) if override_val is not None else float(pr.get("mean", 0.0))
    lo, hi = pr.get("ci95", [val, val])

    return {
        "value": val,
        "ci95": [float(lo), float(hi)],
        "source": "user_override" if override_val is not None else (meta.prior_provenance.get(role, {}).get("base", "industry")),
        "posterior": {"alpha": pr.get("alpha", 1.0), "beta": pr.get("beta", 1.0)},
    }

def apply_rate_uncertainty(point_estimate: float, ci95: List[float], value: float) -> Tuple[float, float]:
    """
    Example: widen an outcome by the relative width of the rate's CI.
    """
    lo, hi = ci95
    if point_estimate <= 0:
        return (value * (1 - hi), value * (1 - lo))
    rel_minus = max(0.0, (point_estimate - lo)) / max(1e-9, point_estimate)
    rel_plus  = max(0.0, (hi - point_estimate)) / max(1e-9, point_estimate)
    return (value * (1 - rel_plus), value * (1 + rel_minus))

from typing import List, Optional, Dict, Any, Tuple
import re
import math

_EMAIL_RX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_DATE_HINTS = ("created_at","order_date","purchase_date","date","timestamp","datetime","time")
_MONEY_HINTS = ("amount","total","subtotal","grand_total","total_price","price","spend","cost")

def _has_any(haystack: set, needles: List[str]) -> bool:
    return any(n in haystack for n in needles)

def _fraction_true(seq) -> float:
    n = len(seq)
    return (sum(1 for x in seq if x) / n) if n else 0.0

def _sample_series(df, col, max_n=200):
    try:
        s = df[col].dropna()
        if s.shape[0] > max_n:
            return s.sample(max_n, random_state=17)
        return s
    except Exception:
        return None

def _light_content_checks(df, norm_set: set) -> Dict[str, float]:
    """
    Cheap signals to confirm kind:
    - email_rate: fraction of valid-looking emails if an email column exists
    - date_parse_rate: fraction of parsable timestamps across any date-ish column
    - id_uniqueness: max uniqueness rate across *_id-ish columns
    """
    out = {"email_rate": 0.0, "date_parse_rate": 0.0, "id_uniqueness": 0.0}
    if df is None or df.empty:
        return out

    # Email
    email_cols = [c for c in df.columns if _slugify(c) in ("email","email_address")]
    email_rates = []
    for c in email_cols:
        s = _sample_series(df, c)
        if s is not None and s.shape[0]:
            email_rates.append(_fraction_true(_EMAIL_RX.match(str(v or "")) is not None for v in s))
    out["email_rate"] = max(email_rates) if email_rates else 0.0

    # Date-ish
    date_cols = [c for c in df.columns if any(h in _slugify(c) for h in _DATE_HINTS)]
    date_rates = []
    for c in date_cols:
        s = _sample_series(df, c)
        if s is not None and s.shape[0]:
            ok = 0
            tot = 0
            for v in s:
                tot += 1
                try:
                    # very lightweight parse test—don’t import heavy libs
                    sv = str(v).strip()
                    # heuristic: digits + separator or ISO-like patterns
                    ok += bool(re.search(r"\d{4}-\d{1,2}-\d{1,2}", sv) or re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", sv))
                except Exception:
                    pass
            date_rates.append(ok / tot if tot else 0.0)
    out["date_parse_rate"] = max(date_rates) if date_rates else 0.0

    # ID uniqueness
    id_cols = [c for c in df.columns if _slugify(c).endswith("_id") or _slugify(c) in ("id","order_id","customer_id","user_id","client_id")]
    uniq_rates = []
    for c in id_cols:
        try:
            s = df[c].dropna()
            if s.shape[0]:
                uniq_rates.append(s.nunique() / max(1, s.shape[0]))
        except Exception:
            pass
    out["id_uniqueness"] = max(uniq_rates) if uniq_rates else 0.0

    return out

def _infer_dataset_context_scored(raw_columns: List[str], df_sample=None) -> Tuple[Optional[str], Dict[str, float]]:
    """
    Improved kind inference with synonym groups + role triads + optional content checks.
    Returns (primary_kind, scores_dict). Use scores>=0.6 to keep multi-kind labels if desired.
    """
    norm = [_slugify(c) for c in raw_columns]
    s = set(norm)

    # Synonyms / signals
    ORD_ID = ["order_id","id","order_no","order_number"]
    ORD_DATE = ["created_at","order_date","purchase_date","timestamp","date","datetime"]
    ORD_MONEY = ["amount","total","subtotal","grand_total","total_price","price"]
    ORD_LINK = ["customer_id","user_id","client_id"]
    ORD_MISC = ["financial_status","fulfillment_status","line_items","currency"]

    CUST_ID = ["customer_id","user_id","client_id","id"]
    CUST_WHO = ["email","email_address","first_name","last_name","name"]
    CUST_META = ["orders_count","total_spent","lifetime_value","ltv","created_at","signup_date","timestamp"]

    MKT_ID = ["campaign_id","adset_id","ad_group_id","creative_id","id"]
    MKT_CHAN = ["channel","medium","source","platform"]
    MKT_MONEY = ["spend","cost","amount","budget"]
    MKT_PERF = ["impressions","clicks","ctr","cpc","cpm","roas","conversions","installs","leads","revenue"]

    def score_group(group: List[str]) -> float:
        return sum(1.0 for g in group if g in s)

    # Base structural scores
    orders_score = (
        1.4 * (score_group(ORD_ID) > 0) +
        1.2 * (score_group(ORD_LINK) > 0) +
        1.2 * (score_group(ORD_DATE) > 0) +
        1.2 * (score_group(ORD_MONEY) > 0) +
        0.6 * (score_group(ORD_MISC) > 0)
    )

    customers_score = (
        1.4 * (score_group(CUST_ID) > 0) +
        1.3 * (score_group(CUST_WHO) > 0) +
        1.0 * (score_group(CUST_META) > 0)
    )

    marketing_score = (
        1.4 * (score_group(MKT_ID) > 0) +
        1.2 * (score_group(MKT_CHAN) > 0) +
        1.2 * (score_group(MKT_MONEY) > 0) +
        0.6 * (score_group(MKT_PERF) > 0)
    )

    # Role triads boost (e.g., for orders: id + money + date)
    if (score_group(ORD_ID) > 0) and (score_group(ORD_MONEY) > 0) and (score_group(ORD_DATE) > 0):
        orders_score += 1.2
    if (score_group(CUST_ID) > 0) and (score_group(CUST_WHO) > 0):
        customers_score += 0.8
    if (score_group(MKT_ID) > 0) and (score_group(MKT_CHAN) > 0) and (score_group(MKT_MONEY) > 0):
        marketing_score += 1.0

    # Optional light content checks (use small df sample for speed)
    if df_sample is not None:
        checks = _light_content_checks(df_sample, s)
        # Email rate should favor customers; not orders/marketing
        customers_score += 0.8 * min(1.0, checks["email_rate"])
        # ID uniqueness favors orders/customers
        orders_score += 0.5 * checks["id_uniqueness"]
        customers_score += 0.5 * checks["id_uniqueness"]
        # Date parse favors orders/marketing when they have date columns
        if _has_any(s, list(ORD_DATE)):
            orders_score += 0.4 * checks["date_parse_rate"]
        if _has_any(s, list(MKT_CHAN)) or _has_any(s, list(MKT_MONEY)):
            marketing_score += 0.4 * checks["date_parse_rate"]

    scores = {"orders": orders_score, "customers": customers_score, "marketing": marketing_score}
    # Normalize to 0..1-ish for easy thresholds
    max_raw = max(scores.values()) if scores else 1.0
    norm_scores = {k: (v / max_raw) if max_raw > 0 else 0.0 for k, v in scores.items()}
    primary = max(norm_scores, key=norm_scores.get) if norm_scores else None
    # If the winner is too weak (<0.5), return None (uncertain)
    if primary and norm_scores[primary] < 0.5:
        primary = None
    return primary, norm_scores

def _infer_dataset_context(raw_columns: List[str]) -> Optional[str]:
    # Use normalized tokens for robust checks
    norm = [ _slugify(c) for c in raw_columns ]
    s = set(norm)

    orders_signals = {"order_id","order_number","total","total_price","subtotal","financial_status","fulfillment_status","line_items"}
    customers_signals = {"customer_id","email","first_name","last_name","orders_count","total_spent","created_at"}

    if len(s & orders_signals) >= 2:
        return "orders"
    if len(s & customers_signals) >= 2:
        return "customers"
    return None
# --- helpers for intake-level PII guardrail (no cfg needed) ---
import re, hashlib, os

_EMAIL_RE  = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_URL_RE    = re.compile(r"^(?:https?://|www\.)\S+$", re.I)
_IPV4_RE   = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
_BOOL_STR  = {"true","false","yes","no","y","n","t","f","1","0"}
_ISO_RE    = re.compile(r"\b(?:USD|EUR|GBP|CAD|AUD|JPY|INR|MXN|BRL|CHF|CNY|HKD|SEK|NOK|DKK|ZAR|NZD)\b", re.I)
_CURRENCY_SYMBOLS = ["$", "€", "£", "¥", "₹", "₽", "₩", "₺", "₴", "₫", "R$", "C$", "A$"]
_PHONE_RE = re.compile(r"""(?x)
    ^(?:\+?\d{1,3}[\s\-\.]?)?          # country
      (?:\(?\d{2,4}\)?[\s\-\.]?)?      # area
      \d{3,4}[\s\-\.]?\d{3,4}$         # local
""")

def _pii_scan(df: pd.DataFrame, sample_rows: int = 1000) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    idx = df.index[: min(sample_rows, len(df))]
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            continue
        try:
            sample = s.loc[idx].astype(str)
        except Exception:
            continue
        def rate(pred) -> float:
            try:
                if sample.empty: return 0.0
                return float(sample.map(lambda x: bool(pred(x))).mean())
            except Exception:
                return 0.0
        out[c] = {
            "email": rate(lambda x: bool(_EMAIL_RE.match(x))),
            "phone": rate(lambda x: bool(_PHONE_RE.match(x))),
            "ipv4":  rate(lambda x: bool(_IPv4_RE.match(x))) if ( _IPv4_RE := _IPV4_RE ) else 0.0,  # alias for style
        }
    return out
from typing import Dict, Optional

def canonical_by_normalized(header_map: Dict[str, "ColumnMeta"]) -> Dict[str, str]:
    """
    Build a mapping {normalized_name -> canonical_hint} from a header_map
    of {original_name -> ColumnMeta}. If multiple originals collapse to the same
    normalized name, we pick the canonical_hint with highest hint_confidence.
    If a ColumnMeta lacks a canonical_hint, we attempt a name-based fallback.

    Expected ColumnMeta attributes:
      - normalized_name: str
      - canonical_hint: Optional[str]
      - hint_confidence: Optional[float]
    """
    best: Dict[str, str] = {}
    best_conf: Dict[str, float] = {}

    for orig, cm in (header_map or {}).items():
        norm = getattr(cm, "normalized_name", None)
        if not norm:
            continue

        hint = getattr(cm, "canonical_hint", None)
        conf = float(getattr(cm, "hint_confidence", 0.0) or 0.0)

        # Fallback to a heuristic if no explicit hint
        if not hint:
            fallback = _canonical_fallback_from_normalized(norm)
            if fallback:
                hint = fallback
                # Give a modest confidence so explicit hints still win
                conf = max(conf, 0.65)

        if not hint:
            # still nothing—skip this normalized name
            continue

        prev = best_conf.get(norm, -1.0)
        if conf > prev:
            best[norm] = hint
            best_conf[norm] = conf

    return best


def _canonical_fallback_from_normalized(name: str) -> Optional[str]:
    """
    Heuristic mapping from a normalized column name to a canonical hint.
    Tailor this to your domain. Keeps things conservative to avoid bad auto-tags.
    """
    n = (name or "").lower()

    # Direct synonyms / common normalizations first
    direct = {
        "id": "id",
        "order_id": "order_id",
        "customer_id": "customer_id",
        "user_id": "customer_id",     # often equivalent in SMB datasets
        "amount": "amount",
        "total": "amount",
        "total_amount": "amount",
        "subtotal": "amount",
        "gross_amount": "amount",
        "net_amount": "amount",
        "created_at": "created_at",
        "updated_at": "updated_at",
        "order_date": "created_at",
        "date": "created_at",         # soft default; override if you track both
        "timestamp": "created_at",
        "email": "email",
        "phone": "phone",
        "phone_number": "phone",
        "ipv4": "ipv4",
        "ip": "ipv4",
        "campaign_id": "campaign_id",
        "adset_id": "adset_id",
        "ad_group_id": "ad_group_id",
        "channel": "channel",
        "source": "channel",
        "medium": "channel",
        "spend": "spend",
        "cost": "spend",
        "revenue": "amount",          # can diverge; map to amount unless you split later
    }
    if n in direct:
        return direct[n]

    # Pattern heuristics
    if n.endswith("_id"):
        # Prefer more specific IDs if known above; otherwise generic
        return "id"
    if "email" in n:
        return "email"
    if "phone" in n or "tel" in n:
        return "phone"
    if any(k in n for k in ("amount", "total", "revenue", "price", "subtotal", "gross", "net")):
        return "amount"
    if any(k in n for k in ("created_at", "order_date", "timestamp", "datetime", "date")):
        return "created_at"
    if any(k in n for k in ("campaign", "adset", "ad_group", "adgroup")):
        # Disambiguate if the normalized name already ends in _id
        return "campaign_id" if n.endswith("_id") else "campaign"
    if "channel" in n:
        return "channel"
    if "spend" in n or "cost" in n or "cpc" in n or "cpm" in n:
        return "spend"
    if "ipv4" in n or "ip" in n:
        return "ipv4"

    return None
from typing import Dict, Optional
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

@dataclass
class CanonPick:
    normalized: str
    hint: Optional[str]
    confidence: float
    reason: str
    originals: List[str] = field(default_factory=list)

@dataclass
class CanonDiag:
    choices: List[CanonPick] = field(default_factory=list)   # per normalized name
    dropped: List[CanonPick] = field(default_factory=list)   # below threshold / lost ties

def _alias_rank(alias_lexicon: Dict[str, List[str]]) -> Dict[str, int]:
    """
    Lower is better. Canonicals seen earlier in the lexicon (or more aliases) get slight priority.
    Helps tie-break equally confident hints.
    """
    rank: Dict[str, int] = {}
    for i, canon in enumerate(alias_lexicon.keys()):
        rank[canon] = min(rank.get(canon, i), i)
    # unseen canonicals default to mid rank
    return rank

_SPECIFICITY = {
    # prefer more specific IDs & money/date canonicals over generic "id" or "date"
    "order_id": 3, "customer_id": 3, "campaign_id": 3,
    "adset_id": 3, "ad_group_id": 3, "email": 3,
    "amount": 2, "created_at": 2, "spend": 2, "channel": 2,
    "id": 1, "date": 1, "timestamp": 1,
}
def _specificity_score(hint: Optional[str]) -> int:
    return _SPECIFICITY.get(hint or "", 0)

_CONTEXT_BOOST = {
    "orders":     {"order_id": 0.03, "customer_id": 0.02, "amount": 0.02, "created_at": 0.02},
    "customers":  {"customer_id": 0.03, "email": 0.04, "created_at": 0.02},
    "marketing":  {"campaign_id": 0.04, "channel": 0.03, "spend": 0.03},
}
def _ctx_boost(hint: Optional[str], context: Optional[str]) -> float:
    if not hint or not context: return 0.0
    return _CONTEXT_BOOST.get(context, {}).get(hint, 0.0)

def canonical_by_normalized(
    header_map: Dict[str, "ColumnMeta"],
    *,
    alias_lexicon: Optional[Dict[str, List[str]]] = None,
    context: Optional[str] = None,
    min_confidence: float = 0.70,             # drop very weak / noisy hints
    allow_fallback: bool = True,
    allow_multi: bool = False,                # if True, keep ties as list
    return_diagnostics: bool = False,
) -> Dict[str, str] | Tuple[Dict[str, str], CanonDiag]:
    """
    Build {normalized_name -> canonical_hint} with smart tie-breaking:
      1) higher confidence (after small context boost)
      2) higher specificity (order_id > id; created_at > date)
      3) alias rank (stable preference by lexicon order/size)
      4) deterministic tiebreak on original column name

    Options:
      - min_confidence: floor after boosts & fallback; else 'unknown' (skipped)
      - allow_fallback: use name-based fallback when explicit hint missing
      - allow_multi: keep multiple hints on perfect ties (rare; for debugging/UX)
      - return_diagnostics: also return CanonDiag with choices/dropped info
    """
    alias_rank = _alias_rank(alias_lexicon or {})

    # Group originals by final normalized names (already deduped by normalize_headers)
    by_norm: Dict[str, List["ColumnMeta"]] = {}
    for orig, cm in (header_map or {}).items():
        norm = getattr(cm, "normalized_name", None)
        if norm:
            by_norm.setdefault(norm, []).append(cm)

    mapping: Dict[str, str] = {}
    diag = CanonDiag()

    for norm, group in by_norm.items():
        # Build candidate list (explicit hint first; fallback if needed)
        cands: List[CanonPick] = []
        for cm in group:
            hint = getattr(cm, "canonical_hint", None)
            conf = float(getattr(cm, "hint_confidence", 0.0) or 0.0)

            reason = "explicit"
            if (not hint) and allow_fallback:
                fb = _canonical_fallback_from_normalized(norm)
                if fb:
                    hint = fb
                    # keep explicit > fallback: at least 0.65, but do not exceed explicit 0.99 tier
                    conf = max(conf, 0.65)
                    reason = "fallback"

            # apply tiny, capped context boost (post-fallback)
            conf = min(1.0, conf + _ctx_boost(hint, context))

            if not hint:
                # no hint at all; skip (but record in diagnostics)
                diag.dropped.append(CanonPick(norm, None, conf, "no_hint", [cm.original_name]))
                continue

            cands.append(CanonPick(
                normalized=norm, hint=hint, confidence=conf,
                reason=reason, originals=[cm.original_name]
            ))

        if not cands:
            continue

        # Merge duplicates per (norm,hint) to keep best confidence / aggregate originals
        merged: Dict[str, CanonPick] = {}
        for p in cands:
            k = p.hint
            if k not in merged or p.confidence > merged[k].confidence:
                merged[k] = CanonPick(p.normalized, p.hint, p.confidence, p.reason, list(p.originals))
            else:
                merged[k].originals.extend(p.originals)
        cands = list(merged.values())

        # Drop below min_confidence
        kept = [p for p in cands if p.confidence >= min_confidence]
        for p in cands:
            if p not in kept:
                diag.dropped.append(CanonPick(p.normalized, p.hint, p.confidence, "below_min_conf", p.originals))
        if not kept:
            continue

        # Sort by (confidence desc, specificity desc, alias rank asc, first original asc)
        def sort_key(p: CanonPick) -> Tuple[Any, ...]:
            return (
                -p.confidence,
                -_specificity_score(p.hint),
                alias_rank.get(p.hint or "", 10_000),
                sorted(p.originals)[0].lower() if p.originals else "",
            )
        kept.sort(key=sort_key)

        # Multi-keep (rare, useful for UI/ambiguity)
        if allow_multi:
            mapping[norm] = ",".join([p.hint for p in kept])
            diag.choices.extend(kept)
            continue

        # Pick the best; note other contenders
        best = kept[0]
        mapping[norm] = best.hint  # type: ignore
        diag.choices.append(best)
        for p in kept[1:]:
            diag.dropped.append(CanonPick(p.normalized, p.hint, p.confidence, "lost_tiebreak", p.originals))

    return (mapping, diag) if return_diagnostics else mapping

def _canonical_fallback_from_normalized(name: str) -> Optional[str]:
    """
    Heuristic mapping from a normalized column name to a canonical hint.
    Tailor this to your domain. Keeps things conservative to avoid bad auto-tags.
    """
    n = (name or "").lower()

    # Direct synonyms / common normalizations first
    direct = {
        "id": "id",
        "order_id": "order_id",
        "customer_id": "customer_id",
        "user_id": "customer_id",     # often equivalent in SMB datasets
        "amount": "amount",
        "total": "amount",
        "total_amount": "amount",
        "subtotal": "amount",
        "gross_amount": "amount",
        "net_amount": "amount",
        "created_at": "created_at",
        "updated_at": "updated_at",
        "order_date": "created_at",
        "date": "created_at",         # soft default; override if you track both
        "timestamp": "created_at",
        "email": "email",
        "phone": "phone",
        "phone_number": "phone",
        "ipv4": "ipv4",
        "ip": "ipv4",
        "campaign_id": "campaign_id",
        "adset_id": "adset_id",
        "ad_group_id": "ad_group_id",
        "channel": "channel",
        "source": "channel",
        "medium": "channel",
        "spend": "spend",
        "cost": "spend",
        "revenue": "amount",          # can diverge; map to amount unless you split later
    }
    if n in direct:
        return direct[n]

    # Pattern heuristics
    if n.endswith("_id"):
        # Prefer more specific IDs if known above; otherwise generic
        return "id"
    if "email" in n:
        return "email"
    if "phone" in n or "tel" in n:
        return "phone"
    if any(k in n for k in ("amount", "total", "revenue", "price", "subtotal", "gross", "net")):
        return "amount"
    if any(k in n for k in ("created_at", "order_date", "timestamp", "datetime", "date")):
        return "created_at"
    if any(k in n for k in ("campaign", "adset", "ad_group", "adgroup")):
        # Disambiguate if the normalized name already ends in _id
        return "campaign_id" if n.endswith("_id") else "campaign"
    if "channel" in n:
        return "channel"
    if "spend" in n or "cost" in n or "cpc" in n or "cpm" in n:
        return "spend"
    if "ipv4" in n or "ip" in n:
        return "ipv4"

    return None

def _safe_hash_series(s: pd.Series, *, salt: Optional[str] = None) -> pd.Series:
    salt = salt or os.environ.get("PII_HASH_SALT", "")
    def h(x: Any) -> str:
        b = (str(x) + salt).encode("utf-8", errors="ignore")
        return hashlib.sha256(b).hexdigest()
    return s.astype(str).map(h)

# ---- Kind detection helpers ----
_KINDS = {
    "orders": {
        "required_any": [["order_id"], ["amount"], ["created_at"]],
        "nice_to_have": [["customer_id"], ["currency"]],
        "join_keys": [{"to": "customers", "left": "customer_id", "right": "customer_id"}],
    },
    "customers": {
        "required_any": [["customer_id"], ["email"], ["created_at"]],
        "nice_to_have": [["phone"], ["country"]],
        "join_keys": [{"to": "orders", "left": "customer_id", "right": "customer_id"}],
    },
    "marketing": {
        "required_any": [["campaign_id"], ["channel"], ["spend"]],
        "nice_to_have": [["impressions"], ["clicks"], ["created_at"]],
        "join_keys": [{"to": "orders", "left": "customer_id", "right": "customer_id"}],  # if present
    },
}

def _has_any(canon_by_norm: Dict[str, str], names: list[str]) -> bool:
    # names are canonical hints (e.g., "order_id", "email")
    have = set(canon_by_norm.values())
    return any(n in have for n in names)

def _infer_present_kinds(canon_by_norm: Dict[str, str]) -> list[str]:
    present = []
    for kind, spec in _KINDS.items():
        req_groups = spec["required_any"]  # OR across groups; each group can be 1 col
        ok = all(_has_any(canon_by_norm, grp) for grp in req_groups)
        if ok:
            present.append(kind)
    return present

def _choose_primary_context(present_kinds: list[str]) -> str:
    # Simple priority; tweak as you learn
    priority = ["orders", "customers", "marketing"]
    for k in priority:
        if k in present_kinds:
            return k
    return present_kinds[0] if present_kinds else "unknown"
from typing import Optional, Any, Dict, List, Tuple
import pandas as pd
import numpy as np

def guess_positive_label(
    df: pd.DataFrame,
    target_cls_col: Optional[str],
) -> Optional[Any]:
    """
    Choose a 'positive' label for classification metrics.
    - Prefer '1' for truly binary 0/1 or booleanish columns.
    - Else map common truthy/positive strings.
    - Else pick the minority class (stable tiebreak via sorted order).
    Returns the label **as it appears in the column** (original dtype if possible).
    """
    if not target_cls_col or target_cls_col not in df.columns:
        return None

    s = df[target_cls_col].dropna()
    if s.empty:
        return None

    # normalize strings for detection, but keep original for return
    s_norm = s.astype(str).str.strip().str.casefold()
    # common mappings
    truthy = {"1", "true", "yes", "y", "t", "purchase", "purchased", "won", "success", "churn", "fraud", "refund", "returned"}
    falsy  = {"0", "false", "no", "n", "f", "lost", "nonchurn", "clean", "no_refund", "not_returned"}

    uniq_norm = set(s_norm.unique())

    # Strict binary (0/1 or bool-ish)
    if uniq_norm <= {"0","1"} or uniq_norm <= {"false","true"}:
        # return the original representation of the positive if possible
        # prefer whatever literal maps to "1"/"true"
        # try exact "1" match first
        mask_pos = s_norm == "1"
        if mask_pos.any():
            return s.loc[mask_pos].iloc[0]
        mask_pos = s_norm == "true"
        if mask_pos.any():
            return s.loc[mask_pos].iloc[0]
        # fallback
        return s.iloc[0]

    # If any known truthy token appears, return that token's original
    for tok in truthy:
        mask = s_norm == tok
        if mask.any():
            return s.loc[mask].iloc[0]

    # Minority class fallback (keeps original labels)
    return _minority_label(s)


def _minority_label(series: pd.Series) -> Optional[Any]:
    if series.empty:
        return None
    counts = series.value_counts(dropna=True)
    # choose smallest count; stable tiebreak on stringified label
    min_count = counts.min()
    cands = sorted(counts[counts == min_count].index, key=lambda x: str(x))
    return cands[0] if cands else None
from typing import Dict, Optional

def _canonical_fallback_from_normalized(name: str) -> Optional[str]:
    """
    Heuristic mapping from a normalized column name to a canonical hint.
    Tailor this to your domain. Keeps things conservative to avoid bad auto-tags.
    """
    n = (name or "").lower()

    # Direct synonyms / common normalizations first
    direct = {
        "id": "id",
        "order_id": "order_id",
        "customer_id": "customer_id",
        "user_id": "customer_id",     # often equivalent in SMB datasets
        "amount": "amount",
        "total": "amount",
        "total_amount": "amount",
        "subtotal": "amount",
        "gross_amount": "amount",
        "net_amount": "amount",
        "created_at": "created_at",
        "updated_at": "updated_at",
        "order_date": "created_at",
        "date": "created_at",         # soft default; override if you track both
        "timestamp": "created_at",
        "email": "email",
        "phone": "phone",
        "phone_number": "phone",
        "ipv4": "ipv4",
        "ip": "ipv4",
        "campaign_id": "campaign_id",
        "adset_id": "adset_id",
        "ad_group_id": "ad_group_id",
        "channel": "channel",
        "source": "channel",
        "medium": "channel",
        "spend": "spend",
        "cost": "spend",
        "revenue": "amount",          # can diverge; map to amount unless you split later
    }
    if n in direct:
        return direct[n]

    # Pattern heuristics
    if n.endswith("_id"):
        # Prefer more specific IDs if known above; otherwise generic
        return "id"
    if "email" in n:
        return "email"
    if "phone" in n or "tel" in n:
        return "phone"
    if any(k in n for k in ("amount", "total", "revenue", "price", "subtotal", "gross", "net")):
        return "amount"
    if any(k in n for k in ("created_at", "order_date", "timestamp", "datetime", "date")):
        return "created_at"
    if any(k in n for k in ("campaign", "adset", "ad_group", "adgroup")):
        # Disambiguate if the normalized name already ends in _id
        return "campaign_id" if n.endswith("_id") else "campaign"
    if "channel" in n:
        return "channel"
    if "spend" in n or "cost" in n or "cpc" in n or "cpm" in n:
        return "spend"
    if "ipv4" in n or "ip" in n:
        return "ipv4"

    return None
def score_kinds(df: pd.DataFrame, canon_by_norm: Dict[str, str]) -> Dict[str, float]:
    """
    Returns per-kind scores based on required+optional signals.
    """
    # Define kind schemas with synonyms
    REQS = {
        "orders": [
            {"any_of": ["order_id"]},
            {"any_of": ["customer_id","user_id","client_id"]},
            {"any_of": ["amount","total","subtotal","grand_total"]},
            {"any_of": ["created_at","order_date","purchase_date","timestamp"]},
        ],
        "customers": [
            {"any_of": ["customer_id","user_id","client_id"]},
            {"any_of": ["email","email_address"]},
            {"any_of": ["created_at","signup_date","timestamp"]},
        ],
        "marketing": [
            {"any_of": ["campaign_id","adset_id","ad_group_id","creative_id"]},
            {"any_of": ["channel","medium","source","platform"]},
            {"any_of": ["spend","cost","amount"]},
            # optional date/impressions/clicks boost
        ],
    }

    # Map normalized -> canonical hint
    hints = set(canon_by_norm.values())
    scores = {}
    for kind, reqs in REQS.items():
        base = 0.0
        for req in reqs:
            if any(tok in hints for tok in req["any_of"]):
                base += 1.0
        # Soft boosts from data content
        if "email" in hints:
            if kind == "customers": base += 0.5
        if "amount" in hints:
            if kind == "orders": base += 0.25
            if kind == "marketing": base += 0.25
        # normalize by number of reqs
        scores[kind] = base / (len(reqs) + 1.0)
    return scores
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import re

_EMAIL_RX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

@dataclass
class ReqGroup:
    any_of: List[str]
    label: Optional[str] = None          # e.g., "Order ID"
    hint: Optional[str] = None           # UI hint: “look for ‘order_number’ or ‘oid’”

@dataclass
class ReqSchema:
    hard: List[ReqGroup] = field(default_factory=list)   # must satisfy all groups (each group = any_of)
    soft: List[ReqGroup] = field(default_factory=list)   # nice to have; improves quality
    min_cover: float = 1.0                               # fraction of hard groups that must pass (1.0 = all)

def _canon_set(header_map) -> Tuple[set, set, Dict[str, List[str]]]:
    """Return (normalized_names, canonical_hints, reverse_index) from header_map."""
    norm_names = {cm.normalized_name for cm in header_map.values()}
    canon_hints = {cm.canonical_hint for cm in header_map.values() if cm.canonical_hint}
    reverse_index: Dict[str, List[str]] = {}
    for orig, cm in header_map.items():
        reverse_index.setdefault(cm.normalized_name, []).append(orig)
    return norm_names, canon_hints, reverse_index

def _any_present(norm_names: set, canon_hints: set, tokens: List[str]) -> bool:
    return any((t in norm_names) or (t in canon_hints) for t in tokens)

def _light_content_quality(df, canon_by_norm: Dict[str,str]) -> Dict[str, float]:
    """Tiny content checks to backstop header presence."""
    out = {"email_rate": 0.0, "date_rate": 0.0, "id_uniqueness": 0.0}
    # Email rate
    email_norms = [n for n, h in canon_by_norm.items() if h == "email"]
    rates = []
    for n in email_norms:
        if n in df.columns:
            s = df[n].dropna().astype(str)
            if not s.empty:
                rates.append(sum(bool(_EMAIL_RX.match(v)) for v in s.head(300)) / min(300, s.shape[0]))
    out["email_rate"] = max(rates) if rates else 0.0

    # Date rate (simple patterns)
    date_norms = [n for n, h in canon_by_norm.items() if h in ("created_at","timestamp")]
    dr = []
    for n in date_norms:
        if n in df.columns:
            s = df[n].dropna().astype(str).str.strip()
            if not s.empty:
                m = s.head(300).str.contains(r"\d{4}-\d{1,2}-\d{1,2}") | s.head(300).str.contains(r"\d{1,2}/\d{1,2}/\d{2,4}")
                dr.append(float(m.mean()))
    out["date_rate"] = max(dr) if dr else 0.0

    # ID uniqueness
    id_norms = [n for n, h in canon_by_norm.items() if h in ("order_id","customer_id","campaign_id","id")]
    ur = []
    for n in id_norms:
        if n in df.columns:
            s = df[n].dropna()
            if s.shape[0]:
                ur.append(s.nunique() / s.shape[0])
    out["id_uniqueness"] = max(ur) if ur else 0.0
    return out

def _schema_for_kind(kind: str) -> ReqSchema:
    """Define hard/soft groups per kind (synonym-aware)."""
    if kind == "orders":
        return ReqSchema(
            hard=[
                ReqGroup(["order_id", "order_number", "oid"], "Order ID", "Try ‘order_no’, ‘order_number’, ‘oid’"),
                ReqGroup(["customer_id","user_id","client_id"], "Customer Link", "Try ‘user_id’, ‘client_id’"),
                ReqGroup(["amount","total","subtotal","grand_total","total_price","price"], "Amount", "Try ‘total_price’"),
                ReqGroup(["created_at","order_date","purchase_date","timestamp","date"], "Order Date", "Try ‘order_date’"),
            ],
            soft=[
                ReqGroup(["currency","currency_code"], "Currency"),
            ],
            min_cover=1.0,
        )
    if kind == "customers":
        return ReqSchema(
            hard=[
                ReqGroup(["customer_id","user_id","client_id","id"], "Customer ID"),
                ReqGroup(["email","email_address"], "Email"),
                ReqGroup(["created_at","signup_date","timestamp","date"], "Created Date"),
            ],
            soft=[
                ReqGroup(["first_name","last_name","name"], "Name"),
                ReqGroup(["phone","phone_number","tel","mobile"], "Phone"),
            ],
            min_cover=1.0,
        )
    if kind == "marketing":
        return ReqSchema(
            hard=[
                ReqGroup(["campaign_id","adset_id","ad_group_id","creative_id","id"], "Campaign/Entity ID"),
                ReqGroup(["channel","medium","source","platform"], "Channel"),
                ReqGroup(["spend","cost","amount","budget"], "Spend"),
            ],
            soft=[
                ReqGroup(["impressions"], "Impressions"),
                ReqGroup(["clicks"], "Clicks"),
                ReqGroup(["created_at","date","timestamp"], "Date"),
            ],
            min_cover=1.0,
        )
    return ReqSchema()

def validate_kind_schema(
    kind: str,
    header_map: Dict[str,"ColumnMeta"],
    df,                         # normalized df
    canon_by_norm: Dict[str,str],
) -> Dict[str, Any]:
    """
    Returns {
      'kind': kind,
      'coverage': 0..1 (hard groups satisfied / total),
      'missing_hard': [ {label, any_of, hint} ],
      'missing_soft': [ {label, any_of, hint} ],
      'severity': 'ok'|'warn'|'error',
      'suggestions': [str],
      'quality': { email_rate, date_rate, id_uniqueness }
    }
    """
    norm_names, canon_hints, reverse_index = _canon_set(header_map)
    spec = _schema_for_kind(kind)

    missing_hard, missing_soft = [], []
    hard_hits = 0
    for g in spec.hard:
        if _any_present(norm_names, canon_hints, g.any_of):
            hard_hits += 1
        else:
            missing_hard.append({"label": g.label or "required", "any_of": g.any_of, "hint": g.hint})

    for g in spec.soft:
        if not _any_present(norm_names, canon_hints, g.any_of):
            missing_soft.append({"label": g.label or "optional", "any_of": g.any_of, "hint": g.hint})

    # Content quality signals
    q = _light_content_quality(df, canon_by_norm)

    # Coverage & severity
    total_hard = max(1, len(spec.hard))
    coverage = hard_hits / total_hard
    passes = coverage >= spec.min_cover

    severity = "ok" if passes and not missing_soft else ("warn" if passes else "error")

    # Suggestions
    suggestions: List[str] = []
    for m in missing_hard + missing_soft:
        tokens = ", ".join(m["any_of"])
        suggestions.append(f"Missing {m['label']}: any of [{tokens}]. {m.get('hint') or ''}".strip())

    # Content warnings upgrade severity if labels look wrong
    if kind == "customers" and q["email_rate"] < 0.5 and _any_present(norm_names, canon_hints, ["email","email_address"]):
        severity = "warn" if severity == "ok" else severity
        suggestions.append("Column labeled as email has low valid-email rate (<50%). Check mapping.")

    if kind in ("orders","customers") and q["id_uniqueness"] < 0.2:
        severity = "warn" if severity == "ok" else severity
        suggestions.append("ID column shows low uniqueness (<20%). Ensure correct ID is used.")

    if kind in ("orders","marketing") and _any_present(norm_names, canon_hints, ["created_at","date","timestamp"]) and q["date_rate"] < 0.5:
        severity = "warn" if severity == "ok" else severity
        suggestions.append("Date column has low parsability (<50%). Check date formats or mapping.")

    return {
        "kind": kind,
        "coverage": round(coverage, 3),
        "missing_hard": missing_hard,
        "missing_soft": missing_soft,
        "severity": severity,
        "suggestions": suggestions,
        "quality": q,
    }
#7
def apply_pii_guardrails(
    df,
    *,
    primary_context: str,
    canon_by_norm: Dict[str, str],
    pii_scan_func,
    pii_hash_salt: str = None,
) -> Tuple[Any, Dict[str, str], Dict[str, str], Dict[str, Dict[str, Any]]]:
    """
    Detects and mitigates PII columns via masking, hashing, or dropping.

    Args:
        df: pandas DataFrame (post-header normalization)
        primary_context: inferred dataset type ("orders", "customers", etc.)
        canon_by_norm: normalized_name -> canonical_hint mapping
        pii_scan_func: function to scan PII likelihoods (signature like _pii_scan(df))
        pii_hash_salt: optional secret salt for deterministic hashing (fallback to env)

    Returns:
        df: transformed DataFrame
        pii_actions: {col -> action}
        hashed_mapping: {raw -> hashed_col}
        pii_metrics: diagnostic info (scores, affected counts)
    """
    import os, re, hmac, hashlib
    from typing import Any, Dict, List

    _DEFAULT_PII_POLICY = {"email": "hash", "phone": "hash", "ipv4": "mask"}
    _CONTEXT_PII_POLICY = {
        "customers": {"email": "hash", "phone": "hash", "ipv4": "drop"},
        "orders":    {"email": "hash", "phone": "hash", "ipv4": "mask"},
        "marketing": {"email": "hash", "phone": "hash", "ipv4": "drop"},
    }
    _PII_THRESHOLDS = {"email": 0.65, "phone": 0.65, "ipv4": 0.70}
    _HEADER_PRIOR = {"email": 0.92, "phone": 0.90, "ipv4": 0.88}
    _EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")
    _PHONE_RE = re.compile(r"^\+?[\d\-\s().]{7,}$")
    _IPV4_RE  = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
    _PII_HASH_SALT = pii_hash_salt or os.environ.get("PII_HASH_SALT", "dev-salt-change-me")

    def _unique_new_name(existing: List[str], base: str) -> str:
        k = 1
        new = base
        while new in existing:
            k += 1
            new = f"{base}_{k}"
        return new

    def _hmac_sha256(value: Any) -> str:
        b = str(value).encode("utf-8", errors="ignore")
        key = _PII_HASH_SALT.encode("utf-8", errors="ignore")
        return hmac.new(key, b, hashlib.sha256).hexdigest()

    def _mask_email(v: str) -> str:
        try:
            local, domain = v.split("@", 1)
            return (local[:2] + "***@" + domain) if len(local) > 2 else "*" * len(local) + "@" + domain
        except Exception:
            return v

    def _mask_phone(v: str) -> str:
        digits = re.sub(r"\D+", "", v)
        return "***-***-" + digits[-4:] if len(digits) >= 4 else "***"

    def _mask_ipv4(v: str) -> str:
        try:
            parts = v.split(".")
            return ".".join(parts[:3] + ["*"]) if len(parts) == 4 else v
        except Exception:
            return v

    def _cell_matches(kind: str, v: Any) -> bool:
        if v is None:
            return False
        s = str(v).strip()
        if not s:
            return False
        if kind == "email":
            return bool(_EMAIL_RE.match(s))
        if kind == "phone":
            return bool(_PHONE_RE.match(s)) and len(re.sub(r"\D+", "", s)) >= 7
        if kind == "ipv4":
            if not _IPV4_RE.match(s):
                return False
            try:
                return all(0 <= int(x) <= 255 for x in s.split("."))
            except Exception:
                return False
        return False

    def _apply_action(kind: str, s, action: str):
        if action == "hash":
            return s.map(lambda x: _hmac_sha256(x) if _cell_matches(kind, x) else x)
        if action == "mask":
            if kind == "email":
                return s.map(lambda x: _mask_email(x) if _cell_matches("email", x) else x)
            if kind == "phone":
                return s.map(lambda x: _mask_phone(x) if _cell_matches("phone", x) else x)
            if kind == "ipv4":
                return s.map(lambda x: _mask_ipv4(x) if _cell_matches("ipv4", x) else x)
        return s

    def _combine_scores(content_score: float, header_prior: float, alpha: float = 0.65):
        return alpha * header_prior + (1 - alpha) * content_score

    # pick policy
    pii_policy = dict(_DEFAULT_PII_POLICY)
    if primary_context in _CONTEXT_PII_POLICY:
        pii_policy.update(_CONTEXT_PII_POLICY[primary_context])

    name_boosts = {
        "email": {n for n, canon in canon_by_norm.items() if canon == "email"},
        "phone": {n for n, canon in canon_by_norm.items() if canon == "phone"},
        "ipv4":  {n for n, canon in canon_by_norm.items() if canon == "ipv4"},
    }

    pii_rates = pii_scan_func(df)
    pii_actions, hashed_mapping, pii_metrics = {}, {}, {}

    for col, rates in pii_rates.items():
        if col.endswith(("_hashed", "_masked")):
            continue
        scores = {k: float(rates.get(k, 0.0)) for k in ("email", "phone", "ipv4")}
        combined = {
            k: _combine_scores(scores[k], _HEADER_PRIOR[k] if col in name_boosts[k] else 0.5)
            for k in scores
        }
        kind, score = max(combined.items(), key=lambda kv: kv[1])
        if score < _PII_THRESHOLDS[kind]:
            continue

        action = pii_policy.get(kind, "hash").lower()
        s = df[col]

        if action == "drop":
            df.drop(columns=[col], inplace=True, errors="ignore")
            pii_actions[col] = "drop"
            pii_metrics[col] = {"kind": kind, "score": score, "action": "drop"}
            continue

        if action in ("hash", "mask"):
            new_name = _unique_new_name(df.columns.tolist(), f"{col}_{action}ed")
            try:
                df[new_name] = _apply_action(kind, s, action)
                if action == "hash":
                    hashed_mapping[col] = new_name
                pii_actions[col] = action
                pii_metrics[col] = {"kind": kind, "score": score, "action": action, "new_col": new_name}
            except Exception:
                pii_actions[col] = "ignore"
        else:
            pii_actions[col] = "ignore"

    return df, pii_actions, hashed_mapping, pii_metrics


# -----------------------------
# Core entry (with PII guardrail)
# -----------------------------
def intake_and_normalize(
    file_path: str,
    *,
    base_currency: str = "USD",
    country_hint: str = "US",
    sample_bytes: int = 200_000,
    parse_json_cells: bool = True,
) -> IntakeResult:
    # 1) sniff + read + strip
    dialect = _sniff_dialect_and_encoding(file_path, sample_bytes=sample_bytes)
    df_raw = _read_dataframe(file_path, dialect)
    df_raw = _strip_bom_and_trim(df_raw)

    # ---- Initialize anomalies BEFORE auto-requirements (important) ----
    anomalies: Dict[str, Any] = {}

    # Helper: synonym-aware requirement (any-of within each group)
    def _require_anyof(header_map, groups: List[List[str]]) -> List[Dict[str, Any]]:
        norm_names = {cm.normalized_name for cm in header_map.values()}
        canon_hints = {cm.canonical_hint for cm in header_map.values() if cm.canonical_hint}
        missing = []
        for group in groups:
            if not any((tok in norm_names) or (tok in canon_hints) for tok in group):
                missing.append({"any_of": group})
        return missing

    # 2) PASS 1 (neutral) — avoid context bias
    header_map1, reverse_index1 = normalize_headers(
        df_raw.columns.tolist(),
        alias_lexicon=_ALIAS_LEXICON,
        unit_suffixes=_UNIT_SUFFIXES,
        symbol_tags=_SYMBOL_TAGS,
        context=None,
        keep_percent_in_slug=False,
    )
    canon_by_norm1 = canonical_by_normalized(header_map1)

    # 3) Score kinds (prefer your content-aware scorer if available)
    try:
        primary_context, scored = _infer_dataset_context_scored(
            df_raw.columns.tolist(), df_sample=df_raw.head(500)
        )
        kind_scores = scored or {}
    except NameError:
        kind_scores = score_kinds(df_raw, canon_by_norm1)
        primary_context = max(kind_scores, key=kind_scores.get) if kind_scores else None

    PRESENT_THRESH = 0.60
    present_kinds = [k for k, v in kind_scores.items() if v >= PRESENT_THRESH] \
                    or ([primary_context] if primary_context else [])

    # 4) PASS 2 (refined) — use chosen context to sharpen hints
    header_map, reverse_index = normalize_headers(
        df_raw.columns.tolist(),
        alias_lexicon=_ALIAS_LEXICON,
        unit_suffixes=_UNIT_SUFFIXES,
        symbol_tags=_SYMBOL_TAGS,
        context=primary_context,
        keep_percent_in_slug=False,
    )

    # Rename columns to normalized
    df = df_raw.rename(columns={c: header_map[c].normalized_name for c in df_raw.columns})
    canon_by_norm = canonical_by_normalized(header_map)  # normalized -> canonical_hint

    # 5) Auto-requirements (synonym-aware, graded, with content checks)
    schema_reports = []
    for kind in present_kinds:
        report = validate_kind_schema(kind, header_map, df, canon_by_norm)
        schema_reports.append(report)
        if report["severity"] in ("warn", "error"):
            anomalies.setdefault("schema", []).append({
                "kind": kind,
                "severity": report["severity"],
                "coverage": report["coverage"],
                "missing_required": report["missing_hard"],
                "missing_optional": report["missing_soft"],
                "quality": report["quality"],
                "suggestions": report["suggestions"],
            })

    # (optional) expose for your debug drawer
    setattr(meta, "schema_reports", schema_reports)

    # 6) JSON expansion (returns a dict mapping)
    json_expansions: Dict[str, List[str]] = {}
    nested_csv_cols: List[str] = []
    if parse_json_cells:
        df, json_expansions = _expand_json_columns(df, min_json_frac=0.2, sample_rows=100)

    # 7) 🛡️ PII guardrails (INTAKE-SIDE)
    df, pii_actions, hashed_mapping, pii_metrics = apply_pii_guardrails(
        df,
        primary_context=primary_context,
        canon_by_norm=canon_by_norm,
        pii_scan_func=_pii_scan,
    )


    # 8) type inference (+ merge anomalies)
    inferred, tz_guess, currency_counts, inferred_anomalies = _infer_types_and_anomalies(df, header_map, dialect)
    for k, v in (inferred_anomalies or {}).items():
        anomalies.setdefault(k, []).extend(v)

    # 9) standardize values
    df_norm = df.copy(deep=True)
    df_norm = _standardize_values(
        df_norm,
        inferred_types=inferred,
        header_map=header_map,
        base_currency=base_currency,
        decimal=dialect.decimal,
        thousands=dialect.thousands,
        country_hint=country_hint,
        anomalies=anomalies,
    )

    # 10) winsorize guard rails
    df_norm = _winsorize_numeric(df_norm, inferred, lower_q=0.01, upper_q=0.99)

    # 11) build meta
    meta = IntakeMeta(
        dialect=dialect,
        header_map=header_map,
        inferred_types=inferred,
        timezone_inference=tz_guess,
        number_format={"decimal": dialect.decimal, "thousands": dialect.thousands},
        currency_symbols=currency_counts,
        json_columns=json_expansions,
        nested_csv_columns=nested_csv_cols,
        anomalies=anomalies,
    )

    # Telemetry
    setattr(meta, "canonical_by_normalized", canon_by_norm)
    setattr(meta, "normalized_to_originals", reverse_index)
    setattr(meta, "json_expansion_counts", {k: len(v) for k, v in json_expansions.items()})
    setattr(meta, "present_kinds", present_kinds)
    setattr(meta, "dataset_identity", {
        "primary": primary_context,
        "present": present_kinds,
        "join_suggestions": {
            k: _KINDS[k].get("join_keys", []) for k in present_kinds if k in _KINDS
        }
    })
    setattr(meta, "pii", {"actions": pii_actions, "hashed_mapping": hashed_mapping})

    # Confidence buckets — compute FIRST, set ONCE
    low, med, high = [], [], []
    for orig, cm in header_map.items():
        c = cm.hint_confidence or 0.0
        rec = {"original": orig, "normalized": cm.normalized_name,
               "hint": cm.canonical_hint, "confidence": round(c, 3)}
        if cm.canonical_hint:
            if c >= 0.90: high.append(rec)
            elif c >= 0.80: med.append(rec)
            else: low.append(rec)
    setattr(meta, "canonical_hint_confidence", {
        "context_inferred": primary_context,
        "high": high, "medium": med, "low": low,
    })

    # Target detection (best-effort)
    try:
        try:
            meta_json = _meta_to_json(meta)
        except NameError:
            meta_json = {
                "canonical_by_normalized": canonical_by_normalized(header_map),
                "context_inferred": primary_context,
                "present_kinds": present_kinds,
                "pii": {"actions": pii_actions},
            }
        det = detect_columns(df_norm, meta=meta_json, context=primary_context)
    except Exception:
        det = {}

    tgt = det.get("target_cls_col")
    pos = guess_positive_label(df_norm, tgt) if tgt else None
    setattr(meta, "suggested", {
        "target_cls_col": tgt,
        "target_cls_candidates": det.get("target_cls_candidates", []),
        "positive_label": pos,
        "why": det.get("why"),
    })

    # Stats
    setattr(meta, "stats", {
        "raw_rows": int(df_raw.shape[0]),
        "raw_cols": int(df_raw.shape[1]),
        "normalized_rows": int(df_norm.shape[0]),
        "normalized_cols": int(df_norm.shape[1]),
    })

    return IntakeResult(df_raw=df_raw, df_normalized=df_norm, meta=meta)


# -----------------------------
# 1) Universal sniffing
# -----------------------------
def _sniff_dialect_and_encoding(file_path: str, *, sample_bytes: int = 200_000) -> DialectInfo:
    raw = Path(file_path).read_bytes()[:sample_bytes]

    # --- encoding ---
    enc = "utf-8"
    if raw.startswith(b"\xef\xbb\xbf"):
        enc = "utf-8-sig"
    elif chardet is not None:
        try:
            guess = chardet.detect(raw) or {}
            if guess.get("encoding"):
                enc = guess["encoding"]
        except Exception:
            pass

    text = raw.decode(enc, errors="replace")

    # --- candidate delimiters ---
    candidates = [",", ";", "\t", "|", "^"]

    # try csv.Sniffer first
    sniffer = csv.Sniffer()
    delim_guess = None
    quotechar = '"'
    try:
        d = sniffer.sniff(text, delimiters=candidates)
        delim_guess = d.delimiter
        quotechar = d.quotechar or '"'
    except Exception:
        pass

    # score each delimiter by consistency of column counts across first ~200 lines
    def _score_delim(delim: str) -> Tuple[float, int]:
        lines = [ln for ln in text.splitlines() if ln.strip()][:200]
        if not lines:
            return (0.0, 0)
        counts = [len(ln.split(delim)) for ln in lines]
        mean_cols = sum(counts)/len(counts)
        var = sum((c-mean_cols)**2 for c in counts)/len(counts)
        score = (mean_cols) / (1.0 + var)   # more columns + lower variance is better
        return (score, int(round(mean_cols)))

    # start with sniffer’s guess, then compare to others
    scored = []
    first = [delim_guess] if delim_guess in candidates else []
    for delim in first + [c for c in candidates if c != delim_guess]:
        s, _ = _score_delim(delim)
        scored.append((s, delim))
    scored.sort(reverse=True)  # best score first

    best_score, best_delim = scored[0] if scored else (0.0, ",")
    delimiter = best_delim

    # header present?
    try:
        has_header = sniffer.has_header(text)
    except Exception:
        has_header = True

    # number format
    decimal, thousands = _infer_number_format_from_text(text)

    return DialectInfo(
        encoding=enc,
        delimiter=delimiter,
        quotechar=quotechar,
        has_header=bool(has_header),
        decimal=decimal,
        thousands=thousands,
        confidence=round(float(best_score), 4),
    )

def _is_phone_like(x: str) -> bool:
    if not x:
        return False
    s = str(x).strip()
    # exclude date-like strings quickly
    if re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$", s):  # YYYY-MM-DD or YYYY/M/D
        return False
    if re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$", s):  # M/D/YYYY etc.
        return False
    digits = len(re.sub(r"\D", "", s))
    if digits < 10:
        return False
    if re.search(r"[A-Za-z]", s):
        return False
    if digits >= 20:
        return False
    return True

def _infer_number_format_from_text(text: str) -> Tuple[str, Optional[str]]:
    sample = re.findall(r"[\d\.,]{3,}", text)[:500]
    dot_as_decimal = 0
    comma_as_decimal = 0
    for token in sample:
        if re.search(r"\d+,\d{2}\b", token) and token.count(",") == 1 and "." not in token:
            comma_as_decimal += 1
        if re.search(r"\d+\.\d{2}\b", token) and token.count(".") == 1 and "," not in token:
            dot_as_decimal += 1
        if re.search(r"\d{1,3}(?:,\d{3})+\.\d{2}\b", token):
            dot_as_decimal += 2
        if re.search(r"\d{1,3}(?:\.\d{3})+,\d{2}\b", token):
            comma_as_decimal += 2
    if comma_as_decimal > dot_as_decimal:
        return ",", "."
    if not sample:
        return ".", ","  # Default to US
    if comma_as_decimal == dot_as_decimal == 0:
        return ".", ","  # No clear signal

    return ".", ","

def _read_dataframe(file_path: str, d: DialectInfo) -> pd.DataFrame:
    suffix = Path(file_path).suffix.lower()
    if suffix in (".xls", ".xlsx", ".xlsm"):
        try:
            return pd.read_excel(file_path)
        except Exception:
            pass
    try:
        return pd.read_csv(
            file_path,
            encoding=d.encoding,
            sep=d.delimiter,
            quotechar=d.quotechar,
            dtype=str,                 # read everything as string first (safer for inference)
            engine="python",
        )
    except Exception:
        return pd.read_csv(file_path, dtype=str, encoding_errors="replace")

import re

_ZW_AND_BOM = r"[\u200B-\u200D\uFEFF]"  # ZWSP, ZWNJ, ZWJ, BOM
_MISSING_RE = r"^(?:nan|NaN|NAN|none|None|null|NULL|NaT)?$"  # empty string matches too

def _strip_bom_and_trim(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 1) Clean column names
    out.columns = (
        pd.Index(out.columns)
        .map(lambda c: re.sub(_ZW_AND_BOM, "", "" if c is None else str(c)).strip())
    )

    # 2) Clean text-like columns only (object or pandas "string")
    text_cols = out.select_dtypes(include=["object", "string"]).columns
    for c in text_cols:
        # Use pandas' nullable StringDtype to preserve <NA>
        s = out[c].astype("string")
        # Remove zero-width & BOM in values, then trim
        s = s.str.replace(_ZW_AND_BOM, "", regex=True).str.strip()
        # Normalize common missing tokens (including empty) to <NA>
        s = s.replace(_MISSING_RE, pd.NA, regex=True)
        out[c] = s

    # 3) Final dtype normalization (ints -> Int64, bool -> boolean, etc.)
    return out.convert_dtypes()


# -----------------------------
# 2) Header normalization

def _norm_tokens(s: str) -> List[str]:
    """Normalized tokens (lowercase a-z0-9) split by non-alnum."""
    s = _ZW_AND_BOM.sub("", s or "")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return [t for t in s.split("_") if t]
def _norm_str(s: str) -> str:
    """Normalized snake_case string (a-z0-9 only)."""
    return "_".join(_norm_tokens(s))
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple, Iterable, Any

# -------------------
# Utilities
# -------------------
_ZW_AND_BOM_RE = r"[\u200B-\u200D\uFEFF]"
_ZW_AND_BOM = re.compile(_ZW_AND_BOM_RE)
_ID_WORD = re.compile(r"(^|_)id($|_)")

def _norm_str(s: str) -> str:
    s = (s or "").strip().lower()
    s = _ZW_AND_BOM.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _slugify(name: str, *, allow_percent: bool = False) -> str:
    """Deterministic identifier for storage / schema."""
    s = (name or "").strip().lower()
    s = _ZW_AND_BOM.sub("", s)
    pattern = r"[^a-z0-9%]+" if allow_percent else r"[^a-z0-9]+"
    s = re.sub(pattern, "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "col"

def _has_word(s: str, word: str) -> bool:
    return re.search(rf"(^|_){re.escape(word)}(_|$)", s) is not None

def _simple_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm_str(a), _norm_str(b)).ratio()

def _best_alias_match(name: str, alias_lexicon: Dict[str, List[str]]) -> Tuple[Optional[str], float]:
    """
    Fuzzy fallback: find the canonical whose alias list best matches `name`.
    Returns (best_canonical, score 0..1). Uses max over aliases per canonical.
    """
    best_canon, best_score = None, 0.0
    for canon, aliases in alias_lexicon.items():
        # score is max similarity vs any alias (including the canon itself as a last alias)
        cand_score = max([_simple_ratio(name, a) for a in (aliases or [])] + [_simple_ratio(name, canon)])
        if cand_score > best_score:
            best_canon, best_score = canon, cand_score
    # small floor to avoid noisy very-low matches
    return (best_canon, best_score) if best_score >= 0.70 else (None, 0.0)

# -------------------
# Confidence (safer)
# -------------------
def _canonical_confidence(
    name: str,
    *,
    alias_lexicon: Dict[str, List[str]],
    context: Optional[str] = None
) -> Optional[float]:
    s_norm = _norm_str(name)

    # Exact "id"
    if s_norm == "id":
        if context in ("orders", "customers"):
            return 1.00
        return None

    # *_id patterns with word boundaries
    if s_norm.endswith("_id") or _ID_WORD.search(s_norm):
        if _has_word(s_norm, "order"):
            return 0.98
        if any(_has_word(s_norm, w) for w in ("customer", "user", "client")):
            return 0.95

    # Exact alias equality (priority)
    for canon, aliases in alias_lexicon.items():
        for alias in aliases:
            if _norm_str(alias) == s_norm:
                return 0.99

    # Fuzzy fallback
    _best, score = _best_alias_match(name, alias_lexicon)
    return float(score) if _best else None

# -------------------
# Tags from header
# -------------------
def _tags_from_header(name: str,
                      unit_suffixes: Iterable[str],
                      symbol_tags: Dict[str, str]) -> List[str]:
    tags: List[str] = []
    s = (name or "")
    s_lower = s.lower()

    for suf in unit_suffixes:
        if s_lower.endswith(suf.lower()):
            tags.append(suf.lower().strip("_"))

    for sym, tag in symbol_tags.items():
        if sym.lower() in s_lower:
            tags.append(tag.lower())

    return sorted(set(tags))

# -------------------
# Data model
# -------------------
@dataclass
class ColumnMeta:
    original_name: str
    normalized_name: str
    canonical_hint: Optional[str]
    tags: List[str] = field(default_factory=list)
    hint_confidence: Optional[float] = None

# -------------------
# Context boost map
# -------------------
_CONTEXT_BOOST: Dict[str, Dict[str, float]] = {
    # canonical_hint -> bonus confidence (capped at 1.0)
    "orders": {
        "order_id": 0.05, "customer_id": 0.03, "amount": 0.03,
        "created_at": 0.03, "currency": 0.02,
    },
    "customers": {
        "customer_id": 0.05, "email": 0.05, "created_at": 0.03,
        "first_name": 0.02, "last_name": 0.02,
    },
    "marketing": {
        "campaign_id": 0.05, "channel": 0.04, "spend": 0.04,
        "impressions": 0.02, "clicks": 0.02,
    },
}

def _apply_context_boost(canon: Optional[str], conf: Optional[float], context: Optional[str]) -> Optional[float]:
    if canon is None or conf is None or context is None:
        return conf
    boost = _CONTEXT_BOOST.get(context, {}).get(canon, 0.0)
    return min(1.0, conf + boost)
import re
from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable
_ZW_AND_BOM_RE = r"[\u200B-\u200D\uFEFF]"
_ZW_AND_BOM = re.compile(_ZW_AND_BOM_RE)
_ALIAS_LEXICON: Dict[str, List[str]] = {
    "amount": ["amount", "price", "total", "subtotal", "grand_total", "net_total", "gross_total", "amt"],
    "created_at": ["created", "create_date", "created_at", "order_date", "timestamp", "processed_at", "date", "txn_date"],
    "updated_at": ["updated", "updated_at", "modified", "last_modified"],
    "order_id": ["order_id", "order_number", "order_no", "oid"],
    "customer_id": ["customer_id", "cust_id", "client_id", "user_id", "uid", "shopper_id"],
    "email": ["email", "email_address", "e-mail"],
    "phone": ["phone", "phone_number", "mobile", "cell", "telephone", "tel"],
    "qty": ["qty", "quantity", "units", "count", "item_count"],
    "discount": ["discount", "promo", "coupon", "markdown"],
    "tax": ["tax", "vat", "gst", "sales_tax"],
    "currency": ["currency", "currency_code", "curr", "ccy"],
    "country": ["country", "country_code"],
    "state": ["state", "province", "region"],
    "postal_code": ["zip", "zipcode", "postal", "postal_code", "postcode"],
    "sku": ["sku", "product_id", "item_id", "variant_id", "asin"],
    "cogs": ["cogs", "unit_cost", "cost", "cost_price", "cost_of_goods"],
    "refund_id": ["refund_id", "return_id", "rma_id"],
    "appointment_id": ["appointment_id", "appt_id", "booking_id", "schedule_id"],
}
_UNIT_SUFFIXES = ["_usd", "_eur", "_gbp", "_cad", "_aud", "_jpy", "_inr", "_qty", "_count", "_pct", "_percentage", "_percent"]
_SYMBOL_TAGS = {"%": "percent", "$": "usd_symbol", "€": "eur_symbol", "£": "gbp_symbol", "¥": "jpy_symbol"}

# -------------------
# Improved normalize_headers
# -------------------
def normalize_headers(columns: List[str],
                      *,
                      alias_lexicon: Dict[str, List[str]],
                      unit_suffixes: Iterable[str],
                      symbol_tags: Dict[str, str],
                      context: Optional[str] = None,
                      keep_percent_in_slug: bool = False,
) -> Tuple[Dict[str, ColumnMeta], Dict[str, List[str]]]:
    """
    Improvements:
    - Uses context-aware confidence boost (capped at 1.0).
    - Collision resolution prefers higher-confidence columns to keep the base normalized name.
    - Deterministic & stable ordering (tie-break by canonical presence, then original name).
    - Reverse index groups original names by the final normalized name.
    - Honors keep_percent_in_slug deterministically.
    """
    metas_by_original: Dict[str, ColumnMeta] = {}

    # 1) First pass: compute metas with hint + confidence (+ context boost)
    for col in columns:
        original = col
        norm = _slugify(original, allow_percent=keep_percent_in_slug)
        tags = _tags_from_header(original, unit_suffixes, symbol_tags)

        canonical = _canonical_hint(original, alias_lexicon=alias_lexicon, context=context)  # <-- your existing function
        conf_raw = _canonical_confidence(original, alias_lexicon=alias_lexicon, context=context)
        conf = _apply_context_boost(canonical, conf_raw, context)

        metas_by_original[original] = ColumnMeta(
            original_name=original,
            normalized_name=norm,
            canonical_hint=canonical,
            tags=tags,
            hint_confidence=conf,
        )

    # 2) Collision handling: prefer higher-confidence to keep the base slug
    grouped: Dict[str, List[ColumnMeta]] = {}
    for meta in metas_by_original.values():
        grouped.setdefault(meta.normalized_name, []).append(meta)

    for norm_name, group in grouped.items():
        if len(group) == 1:
            continue

        # Sort so the "winner" (keeps base name) is first:
        #  - higher confidence first
        #  - if tie: one that HAS a canonical hint beats None
        #  - then lexicographic on original name for determinism
        group.sort(
            key=lambda m: (
                -(m.hint_confidence or 0.0),
                0 if m.canonical_hint else 1,
                _norm_str(m.original_name),
            )
        )

        # First keeps the base normalized name; suffix others
        for idx, meta in enumerate(group, start=1):
            if idx == 1:
                meta.normalized_name = norm_name
            else:
                # find the next free suffix deterministically
                k = idx
                candidate = f"{norm_name}_{k}"
                while candidate in grouped:  # avoid clashing with an existing base elsewhere
                    k += 1
                    candidate = f"{norm_name}_{k}"
                meta.normalized_name = candidate

    # 3) Rebuild reverse index after final names are assigned
    reverse_index: Dict[str, List[str]] = {}
    for orig, meta in metas_by_original.items():
        reverse_index.setdefault(meta.normalized_name, []).append(orig)

    # Ensure deterministic ordering in reverse index (optional nicety)
    for k in reverse_index:
        reverse_index[k].sort(key=_norm_str)

    return metas_by_original, reverse_index


from functools import lru_cache

@lru_cache(maxsize=4096)
def _cached_norm_str(s: str) -> str:
    return _norm_str(s)

@lru_cache(maxsize=4096)
def _cached_norm_tokens_tuple(s: str) -> Tuple[str, ...]:
    return tuple(_norm_tokens(s))

def _best_alias_match(s: str,
                      alias_lexicon: Dict[str, List[str]]) -> Tuple[Optional[str], float]:
    s_norm = _cached_norm_str(s)
    s_tokens = set(_cached_norm_tokens_tuple(s))

    best_canon, best_score, best_alias_norm_len = None, 0.0, -1
    for canon, aliases in alias_lexicon.items():
        for alias in aliases:
            a_norm = _cached_norm_str(alias)
            a_tokens = set(_cached_norm_tokens_tuple(alias))
            base = SequenceMatcher(a=s_norm, b=a_norm).ratio()

            inter = len(s_tokens & a_tokens)
            union = len(s_tokens | a_tokens) or 1
            jaccard = inter / union
            score = 0.8 * base + 0.2 * jaccard

            if (score > best_score) or (abs(score - best_score) < 1e-9 and len(a_norm) > best_alias_norm_len):
                best_canon, best_score, best_alias_norm_len = canon, score, len(a_norm)
    return best_canon, best_score


def _canonical_hint(name: str,
                    *,
                    alias_lexicon: Dict[str, List[str]],
                    context: Optional[str] = None,
                    fuzzy_threshold: float = 0.80) -> Optional[str]:
    """
    Canonical inference:
      1) shortcut rules (context-aware)
      2) exact alias equality (normalized)
      3) fuzzy fallback with token boost
    """
    s_norm = _norm_str(name)

    # 1) Shortcut rules (context-aware "id")
    if s_norm == "id":
        if context == "orders":
            return "order_id"
        if context == "customers":
            return "customer_id"
        # If unknown, return None (safer than opinionated default)
        return None

    if s_norm.endswith("_id"):
        if "order" in s_norm:
            return "order_id"
        if any(tok in s_norm for tok in ("customer", "user", "client")):
            return "customer_id"

    # 2) Exact alias equality (normalized)
    for canon, aliases in alias_lexicon.items():
        for alias in aliases:
            if s_norm == _norm_str(alias):
                return canon

    # 3) Fuzzy fallback
    best, score = _best_alias_match(name, alias_lexicon)
    return best if score >= fuzzy_threshold else None


def _ensure_series(x) -> pd.Series:
    """Return a Series no matter what.
    - DataFrame: returns the first column (deterministically).
    - Scalar/iterable: wraps in Series.
    """
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0] if x.shape[1] else pd.Series([], dtype="object")
    try:
        return pd.Series(x)
    except Exception:
        return pd.Series([], dtype="object")


def _unique_ratio_scalar(s: pd.Series) -> float:
    """Unique/non-null ratio as a float, never a Series."""
    s = _ensure_series(s)
    s = s.dropna()
    n = int(len(s))
    if n == 0:
        return 0.0
    try:
        u = int(s.nunique(dropna=True))
    except Exception:
        # last-ditch: via numpy
        try:
            u = int(np.unique(s.astype(str)).size)
        except Exception:
            u = 0
    return float(u) / float(n)
def _expand_json_columns(df: pd.DataFrame,
                         min_json_frac: float = 0.2,
                         sample_rows: int = 100) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    import json
    json_map: Dict[str, List[str]] = {}
    n = len(df)
    probe_idx = df.index[: min(sample_rows, n)]

    json_like = []
    for col in df.columns:
        s = df[col]
        if not pd.api.types.is_object_dtype(s):
            continue
        try:
            ss = s.loc[probe_idx].astype(str)
            mask = ss.str.startswith("{") | ss.str.startswith("[")
            gate = float(mask.mean())
        except Exception:
            gate = 0.0
        if gate >= min_json_frac:
            json_like.append(col)

    def _safe_parse(x: str):
        try:
            return json.loads(x) if isinstance(x, str) and (x.startswith("{") or x.startswith("[")) else {}
        except Exception:
            return {}

    out = df
    for col in json_like:
        parsed = out[col].astype(str).apply(_safe_parse)
        expanded = pd.json_normalize(parsed)

        # prefix and avoid collisions
        prefix = f"{col}."
        newcols = []
        for c in expanded.columns:
            base = prefix + str(c)
            new_name = base
            k = 2
            while new_name in out.columns or new_name in newcols:
                new_name = f"{base}_{k}"
                k += 1
            newcols.append(new_name)
        expanded.columns = newcols

        out = pd.concat([out.drop(columns=[col]), expanded], axis=1)
        json_map[col] = newcols

    return out, json_map


# -----------------------------
# 4) Type inference + anomalies
# -----------------------------

# Common regexes/helpers your code references

import re
from typing import Dict, Tuple, Iterable, Optional
import pandas as pd
import numpy as np

# --- currency symbol/config ---

# Common symbols you’ll likely see in web/exported CSVs.
_CURRENCY_SYMBOLS: tuple[str, ...] = (
    "$", "€", "£", "¥", "₹", "₩", "₱", "₪", "₫", "₦", "₴", "₡", "₽", "R$", "A$", "C$", "HK$", "NT$", "kr", "Kč"
)

# Optional ISO codes (prefix/suffix). Keep short; false positives are possible.
_CURRENCY_ISO_CODES: tuple[str, ...] = (
    "USD","EUR","GBP","JPY","INR","CNY","CAD","AUD","NZD","HKD","SGD","SEK","NOK","DKK","CHF","RUB","ZAR","MXN","BRL"
)

_ISO_RE = re.compile(r"\b(" + "|".join(_CURRENCY_ISO_CODES) + r")\b", re.IGNORECASE)

def _to_series(x) -> pd.Series:
    if isinstance(x, pd.Series): return x
    if isinstance(x, pd.DataFrame): return x.iloc[:,0] if x.shape[1] else pd.Series([], dtype="object")
    try: return pd.Series(x)
    except Exception: return pd.Series([], dtype="object")


# -------------------------------------------------------------------
# 1) Count currency “hits” in a sample column
# -------------------------------------------------------------------
def _count_currency_symbols(sample: pd.Series | Iterable | object) -> Dict[str, int]:
    s = _to_series(sample).dropna()
    if s.empty: return {}
    s = s.astype(str)

    hits: Dict[str, int] = {}

    # Symbols
    if _CURRENCY_SYMBOLS:
        patt = "|".join(re.escape(sym) for sym in _CURRENCY_SYMBOLS if sym)
        cnts = s.str.contains(patt, regex=True)
        # We want per-symbol counts, not just "any symbol"
        for sym in _CURRENCY_SYMBOLS:
            hits[sym] = int(s.str.contains(re.escape(sym)).sum())

    # ISO codes (word-bounded)
    iso_counts: Dict[str, int] = {}
    for cell in s:
        for m in _ISO_RE.finditer(cell):
            iso = m.group(0).upper()
            iso_counts[iso] = iso_counts.get(iso, 0) + 1

    hits.update({k: hits.get(k, 0) + v for k, v in iso_counts.items()})
    # drop zeroes
    return {k: v for k, v in hits.items() if v > 0}



# -------------------------------------------------------------------
# 2) Numeric coercion rate (handles thousands/decimal/%, currency)
# -------------------------------------------------------------------
def _coerce_numeric_rate(
    sample: pd.Series | Iterable | object,
    decimal: str = ".",
    thousands: Optional[str] = ",",
) -> Tuple[float, pd.Series]:
    s0 = _to_series(sample)
    present_mask = ~s0.isna()
    s = s0[present_mask]
    if s.empty:
        return 0.0, pd.Series(np.nan, index=s0.index, dtype="float64")

    s = s.astype(str).str.strip()

    # Remove JSON-like
    jsonish = s.str.startswith("{") | s.str.startswith("[")
    s = s[~jsonish]

    # Remove currency symbols
    if _CURRENCY_SYMBOLS:
        patt = "|".join(re.escape(c) for c in _CURRENCY_SYMBOLS if c)
        s = s.str.replace(patt, "", regex=True)
    # Remove ISO codes
    s = s.str.replace(_ISO_RE, "", regex=True)

    # Percent handling
    pct_mask = s.str.endswith("%")
    s = s.str.replace(r"%+$", "", regex=True)

    # Remove spaces inside numbers, e.g., "1 234,56"
    s = s.str.replace(r"(?<=\d)\s+(?=\d)", "", regex=True)

    # Thousands → remove
    if thousands:
        s = s.str.replace(re.escape(thousands), "", regex=True)

    # Decimal → normalize to '.'
    if decimal and decimal != ".":
        s = s.str.replace(re.escape(decimal), ".", regex=True)

    # Parens negatives
    s = s.str.replace(r"^\(([^)]+)\)$", r"-\1", regex=True)

    # Strip quotes
    s = s.str.strip().str.strip('"').str.strip("'")

    coerced = pd.to_numeric(s, errors="coerce")

    if pct_mask.any():
        pct_mask = pct_mask.reindex(coerced.index).fillna(False)
        coerced.loc[pct_mask] = coerced.loc[pct_mask] / 100.0

    out = pd.Series(np.nan, index=s0.index, dtype="float64")
    out.loc[coerced.index] = coerced.values

    denom = float(present_mask.sum()) if present_mask.sum() else 1.0
    rate = float(pd.notna(out[present_mask]).mean()) if denom else 0.0
    return rate, out


def _rate(s: pd.Series, pred) -> float:
    """Fraction of values for which pred(value) is True (non-null only)."""
    s = _to_series(s).dropna().astype(str)
    if s.empty: return 0.0
    try:
        return float(s.map(pred).mean())
    except Exception:
        return 0.0

def _infer_types_and_anomalies(
    df: pd.DataFrame,
    header_map: Dict[str, "ColumnMeta"],
    dialect: "DialectInfo",
) -> Tuple[Dict[str, "TypeInfo"], Optional[str], Dict[str, int], Dict[str, List[str]]]:
    inferred: Dict[str, "TypeInfo"] = {}
    anomalies: Dict[str, List[str]] = {}
    tz_guess: Optional[str] = None
    currency_counts: Dict[str, int] = {}

    for col in df.columns:
        raw = df[col]
        s = _ensure_series(raw)
        nonnull = s.dropna()
        unique_ratio_f = _unique_ratio_scalar(s)

        # Uniform sample (avoid head bias)
        sample = nonnull.sample(min(800, len(nonnull)), random_state=42).astype(str) if len(nonnull) else pd.Series([], dtype="object")

        # Rates
        email_f = _rate(sample, lambda x: bool(_EMAIL_RE.match(x)))
        phone_f = _rate(sample, _is_phone_like) if '_is_phone_like' in globals() else 0.0
        url_f   = _rate(sample, lambda x: bool(_URL_RE.match(x)))
        ipv4_f  = _rate(sample, lambda x: bool(_IPV4_RE.match(x)))
        bool_f  = _rate(sample, lambda x: x.strip().lower() in _BOOL_STR)
        json_f  = _rate(sample, _seems_json)

        ts_rate, tz_found = _timestamp_parse_rate(sample)
        ts_f = float(ts_rate)
        tz_guess = tz_guess or tz_found

        money_symbol_hits = _count_currency_symbols(sample)
        for sym, cnt in money_symbol_hits.items():
            currency_counts[sym] = currency_counts.get(sym, 0) + cnt

        num_f, as_numeric = _coerce_numeric_rate(sample, dialect.decimal, dialect.thousands)

        col_str = str(col).lower()
        percent_hint = any(k in col_str for k in ("%", "_pct", "percent", "rate"))

        # zero/one boolean signal
        zero_one_f = _rate(sample, lambda x: x.strip() in {"0","1"})

        # Scalar helper (kept for completeness)
        def _as_ratio(x) -> float:
            try:
                v = float(x)
                return 0.0 if np.isnan(v) else v
            except Exception:
                return 0.0

        try:
            idlen_rate = float(pd.Series(sample).map(lambda x: len(x) <= 64).mean()) if len(sample) else 0.0
        except Exception:
            idlen_rate = 0.0

        # Looser numeric-ID acceptance (int-like + high uniqueness)
        looks_integer = _rate(sample, lambda x: x.isdigit()) > 0.9 if len(sample) else 0.0

        id_like = (
            unique_ratio_f > 0.90
            and email_f < 0.20
            and phone_f < 0.20
            and ts_f    < 0.20
            and idlen_rate > 0.70
            and (num_f < 0.30 or looks_integer > 0.90)
        )

        # Amount hint from header_map (e.g., "amount", "total")
        is_amount_hint = False
        try:
            cm = header_map.get(col)
            is_amount_hint = bool(cm and cm.canonical_hint == "amount")
        except Exception:
            pass

        # --- Classification (ordered) ---
        if email_f > 0.60:
            sem, vr = "email", email_f
        elif ts_f > 0.50:
            sem, vr = "timestamp", ts_f
        elif phone_f > 0.60:
            sem, vr = "phone", phone_f
        elif url_f > 0.60:
            sem, vr = "url", url_f
        elif ipv4_f > 0.60:
            sem, vr = "ip", ipv4_f
        elif (bool_f > 0.80) or (zero_one_f > 0.95):
            sem, vr = "boolean", max(bool_f, zero_one_f)
        elif json_f > 0.60:
            sem, vr = "json", json_f
        elif id_like:
            sem, vr = "id", unique_ratio_f
        elif (money_symbol_hits.get("$",0)+money_symbol_hits.get("€",0)+money_symbol_hits.get("£",0)+money_symbol_hits.get("¥",0)) > 0.10*max(1,len(sample)):
            sem, vr = "money", (sum(money_symbol_hits.values())/max(1,len(sample)))
        elif is_amount_hint and num_f > 0.85:
            sem, vr = "money", num_f
        elif percent_hint and num_f > 0.50:
            # sanity on ranges
            vals = pd.to_numeric(as_numeric, errors="coerce")
            in_0_1   = float(vals.between(0,1).mean())   if len(vals) else 0.0
            in_0_100 = float(vals.between(0,100).mean()) if len(vals) else 0.0
            if max(in_0_1, in_0_100) > 0.70:
                sem, vr = "percent", num_f
            else:
                sem, vr = "numeric", num_f
        elif num_f > 0.85:
            sem, vr = "numeric", num_f
        else:
            avg_len = float(pd.Series(sample).map(len).mean()) if len(sample) else 0.0
            if unique_ratio_f <= 0.05 and nonnull.nunique(dropna=True) <= 50:
                sem, vr = "categorical", 1 - unique_ratio_f
            elif avg_len > 40:
                sem, vr = "text", 0.60
            else:
                sem, vr = "categorical", 0.50

        inferred[col] = TypeInfo(
            semantic=sem,
            pandas_dtype=str(s.dtype),
            unique_ratio=float(unique_ratio_f),
            valid_rate=float(vr),
            notes=None,
        )

        # --- Anomalies ---
        if sem == "percent":
            try:
                vals = pd.to_numeric(as_numeric, errors="coerce")
                over_1 = float((vals > 1.0).mean()) if len(vals) else 0.0
                if over_1 > 0.50 and "%" not in col_str:
                    anomalies.setdefault("typing", []).append(
                        f"{col}: percent-like but majority values > 1 (maybe scale by 100 or mark as '%')."
                    )
            except Exception:
                pass

    return inferred, tz_guess, currency_counts, anomalies

import numpy as np
import pandas as pd
from typing import Any, Optional, Tuple

def _as_ratio(x: Any) -> float:
    """
    Coerce x to a scalar ratio in [0,1].
    - bool/number → float
    - Series/ndarray[bool] → mean(True)
    - tuple (rate, …) → first element
    - None/NaN → 0.0
    """
    # tuple like (rate, tz) or (rate, series)
    if isinstance(x, tuple) and len(x) >= 1:
        return _as_ratio(x[0])

    if isinstance(x, (pd.Series, np.ndarray)):
        try:
            # boolean-like series/array → mean of True
            if getattr(x, "dtype", None) in (bool, np.bool_):
                return float(np.nanmean(x.astype("float64")))
            # numeric-like → fraction of notna
            return float(pd.to_numeric(pd.Series(x), errors="coerce").notna().mean())
        except Exception:
            return 0.0

    try:
        v = float(x)
        return 0.0 if np.isnan(v) else v
    except Exception:
        return 0.0


def _seems_json(x: str) -> bool:
    if not x:
        return False
    if not (x.lstrip().startswith("{") or x.lstrip().startswith("[")):
        return False
    try:
        json.loads(x)
        return True
    except Exception:
        return False

from typing import Tuple
import pandas as pd
import numpy as np
def _timestamp_parse_rate(cand) -> Tuple[float, Optional[str]]:
    """
    Return (parse_rate, tz_guess). Robust & quiet on warnings.
    """
    if isinstance(cand, pd.DataFrame):
        best_rate, best_tz = 0.0, None
        for name in list(cand.columns):
            if isinstance(name, (list, tuple, dict)):  # skip weird labels
                continue
            try:
                s = cand[name]
            except Exception:
                continue
            if not isinstance(s, pd.Series):
                continue
            rate, tz = _timestamp_parse_rate(s)
            if rate > best_rate:
                best_rate, best_tz = rate, tz
        return best_rate, best_tz

    if not isinstance(cand, pd.Series):
        try: cand = pd.Series(cand)
        except Exception: return 0.0, None

    s = cand.dropna()
    if s.empty: return 0.0, None

    try:
        s = s.astype(str).str.strip()
        s = s[~(s.str.startswith("{") | s.str.startswith("["))]  # drop JSONish
    except Exception:
        return 0.0, None
    if s.empty: return 0.0, None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            parsed = pd.to_datetime(s, errors="coerce", utc=False)
        except Exception:
            return 0.0, None

    rate = float(parsed.notna().mean())
    return rate, None

import re
import pandas as pd

def _flatten_and_dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make all column labels plain, unique strings. Handles MultiIndex & duplicates."""
    if isinstance(df.columns, pd.MultiIndex):
        flat = [
            "__".join([str(x) for x in tup if x is not None]).strip()
            for tup in df.columns.to_list()
        ]
    else:
        flat = [str(c).strip() for c in df.columns]

    # normalize whitespace
    flat = [re.sub(r"\s+", "_", c) for c in flat]

    # dedupe
    seen = {}
    new_cols = []
    for c in flat:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}__{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)

    df = df.copy()
    df.columns = new_cols
    return df

# -----------------------------
# 5) Value standardization
# -----------------------------
def _standardize_values(
    df: pd.DataFrame,
    *,
    inferred_types: Dict[str, TypeInfo],
    header_map: Dict[str, ColumnMeta],
    base_currency: str,
    decimal: str,
    thousands: Optional[str],
    country_hint: str,
    anomalies: List[str],
) -> pd.DataFrame:
    out = _flatten_and_dedupe_columns(df)

    for c in list(out.columns):
        s = out[c]
        # If a DataFrame still slipped in, split and process each subcolumn independently
        if isinstance(s, pd.DataFrame):
            for sub in s.columns:
                ss = s[sub]
                if ss.dtype == "O":
                    # ... your object/string processing ...
                    s[sub] = ss  # assign back
            out[c] = s  # keep as block if you really want, but safer if you already flattened
            continue

        # Normal Series path
        if s.dtype == "O":
            # ... your object/string processing ...
            out[c] = s

    for col, info in inferred_types.items():
        sem = info.semantic

        if sem == "email":
            out[col] = out[col].str.lower().str.strip()

        if sem == "phone":
            out[col] = out[col].apply(lambda x: _to_e164_like(str(x), country_hint=country_hint))

        if sem == "boolean":
            out[col] = out[col].map(lambda x: str(x).strip().lower() if pd.notna(x) else x)\
                               .map({"true":True,"t":True,"yes":True,"y":True,"1":True,
                                     "false":False,"f":False,"no":False,"n":False,"0":False})

        if sem == "timestamp":
            out[col] = pd.to_datetime(out[col], errors="coerce")

        if sem in ("money","numeric","percent"):
            s = out[col].astype(str)
            if thousands:
                s = s.str.replace(re.escape(thousands), "", regex=True)
            if decimal != ".":
                s = s.str.replace(decimal, ".", regex=False)
            s = s.str.replace(r"[$€£¥]", "", regex=True)
            vals = pd.to_numeric(s.str.replace("%", "", regex=False), errors="coerce")

            if sem == "percent" and vals.notna().any():
                if (vals > 1).mean() > 0.5:
                    out[col + "_rate"] = vals / 100.0
                else:
                    out[col + "_rate"] = vals
            else:
                out[col + "_num"] = vals

        if sem == "categorical":
            out[col + "_slug"] = out[col].astype(str).map(_slug_for_value)

    _maybe_add_base_money_columns(out, inferred_types, base_currency, anomalies)
    _canonicalize_units_from_headers(out)
    return out

def _to_e164_like(s: str, *, country_hint: str = "US") -> Optional[str]:
    if not s or s.lower() == "nan":
        return None
    digits = re.sub(r"\D", "", s)
    if not digits:
        return None
    if country_hint.upper() in ("US","CA") and len(digits) == 10:
        return "+1" + digits
    if s.strip().startswith("+"):
        return "+" + digits
    return "+" + digits

def _slug_for_value(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or None

def _maybe_add_base_money_columns(df: pd.DataFrame, inferred_types: Dict[str, TypeInfo], base_currency: str, anomalies: List[str]) -> None:
    sym_for = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
    sym = sym_for.get(base_currency.upper())
    for col, ti in inferred_types.items():
        if ti.semantic != "money":
            continue
        s = df[col].astype(str)
        has_sym = s.str.contains(re.escape(sym)) if sym else pd.Series([False]*len(s))
        header_lower = col.lower()
        tagged = any(tag in header_lower for tag in [f"_{base_currency.lower()}", base_currency.lower()])
        if (has_sym.mean() > 0.6) or tagged:
            nums = s.str.replace(r"[$€£¥]", "", regex=True).str.replace(",", "", regex=False)
            try:
                nums = pd.to_numeric(nums, errors="coerce")
                df[col + f"_{base_currency.lower()}"] = nums
            except Exception:
                anomalies.append(f"{col}: failed to create base-currency numeric column")

def _canonicalize_units_from_headers(df: pd.DataFrame) -> None:
    unit_map = [
        (r"_oz$", "g", lambda x: x * 28.3495),
        (r"_lb$", "g", lambda x: x * 453.592),
        (r"_kg$", "g", lambda x: x * 1000.0),
        (r"_g$",  "g", lambda x: x * 1.0),
        (r"_ml$", "l", lambda x: x / 1000.0),
        (r"_l$",  "l", lambda x: x * 1.0),
        (r"_floz$","l", lambda x: x * 0.0295735),
    ]
    for col in list(df.columns):
        if not col.lower().endswith(("_oz","_lb","_kg","_g","_ml","_l","_floz")):
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        for pat, canon, fn in unit_map:
            if re.search(pat, col.lower()):
                new_col = re.sub(pat, f"_{canon}", col, flags=re.IGNORECASE)
                df[new_col] = s.map(lambda v: fn(v) if pd.notna(v) else v)
                break

# -----------------------------
# 6) Winsorize guard (non-destructive)
# -----------------------------
def _winsorize_numeric(df: pd.DataFrame, inferred: Dict[str, TypeInfo], lower_q=0.01, upper_q=0.99) -> pd.DataFrame:
    out = df.copy()
    for col, ti in inferred.items():
        if ti.semantic in ("numeric","money","percent"):
            # include base + derived numeric columns
            candidates = [col] + [c for c in out.columns if c.startswith(col + "_")]
            targets = [c for c in candidates if out[c].dtype != object]
            for t in targets:
                series = pd.to_numeric(out[t], errors="coerce")
                if series.notna().sum() < 50:
                    continue
                lo = series.quantile(lower_q)
                hi = series.quantile(upper_q)
                out[t + "_capped"] = series.clip(lower=lo, upper=hi)
    return out


# -----------------------------
# Helpers for preview & meta json
# -----------------------------
def _preview_table(df: pd.DataFrame, n: int = 50) -> Dict[str, Any]:
    """
    Returns {"columns": [...], "rows": [...]}
    - limits to first n rows
    - converts datetimes to ISO strings
    - converts timedeltas to ISO 8601 durations
    - replaces NaN/Inf/-Inf/NaT with None
    - converts numpy scalars to native Python types
    """
    if df is None or df.empty:
        return {"columns": [], "rows": []}

    sample = df.head(n).copy().reset_index(drop=True)

    # Normalize datetime columns -> ISO strings
    for c in sample.columns:
        if is_datetime64_any_dtype(sample[c]):
            try:
                if getattr(sample[c].dt, "tz", None) is not None:
                    sample[c] = sample[c].dt.tz_convert("UTC")
                else:
                    sample[c] = sample[c].dt.tz_localize(None)
            except Exception:
                sample[c] = pd.to_datetime(sample[c], errors="coerce").dt.tz_localize(None)
            sample[c] = sample[c].dt.strftime("%Y-%m-%dT%H:%M:%S")
        elif is_timedelta64_dtype(sample[c]):
            sample[c] = sample[c].apply(lambda td: _td_to_iso(td) if pd.notna(td) else None)

    num_cols = sample.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        col = sample[c]
        sample[c] = col.apply(lambda x: x if (isinstance(x, (int, float)) and math.isfinite(x)) else (None if pd.isna(x) or isinstance(x, float) else x))

    def _py_simplify(v):
        if isinstance(v, (np.generic,)):
            v = v.item()
        if isinstance(v, float):
            return v if math.isfinite(v) else None
        if pd.isna(v):
            return None
        return v

    sample = sample.applymap(_py_simplify)

    return {"columns": [str(c) for c in sample.columns], "rows": sample.to_dict(orient="records")}

def _td_to_iso(td: pd.Timedelta) -> str:
    if pd.isna(td):
        return None
    total_seconds = int(td.total_seconds())
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    s = "P"
    if days:
        s += f"{days}D"
    if hours or minutes or seconds:
        s += "T"
        if hours:
            s += f"{hours}H"
        if minutes:
            s += f"{minutes}M"
        if seconds or (not hours and not minutes):
            s += f"{seconds}S"
    return s

def _meta_to_json(meta: IntakeMeta) -> Dict[str, Any]:
    data = asdict(meta)
    def _clean(obj):
        if isinstance(obj, float):
            return None if pd.isna(obj) else float(obj)
        if isinstance(obj, dict):
            return {str(k): _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(x) for x in obj]
        return obj
    return _clean(data)

# ----------------------------------------
# Data gap detection & impact heuristics
# ----------------------------------------
EXPECTED_SIGNALS = {
    # ===== Profit & revenue fidelity =====
    "revenue": {
        "aliases": ["revenue","amount","sales","sales_usd","gmv","order_amount","total_amount"],
        "criticality": "high",
        "impact_model": "blocks_profit_calc",
        "bias_pct_range": (0.10, 0.30),
        "default": None,
        "requires": {"mode": "any"},
        "dtype": "numeric",
        "min_coverage": 0.5,
    },
    "cogs": {
        "aliases": ["cogs","cost","unit_cost","cost_price","cost_of_goods_sold"],
        "criticality": "high",
        "impact_model": "overestimates_profit",
        "bias_pct_range": (0.05, 0.20),
        "default": {"cogs_pct_of_revenue": 0.55},
        "requires": {"mode": "any"},
        "dtype": "numeric",
        "min_coverage": 0.25,
    },
    "discounts": {
        "aliases": ["discount","coupon","promo","promo_amount","discount_amount","discount_pct"],
        "criticality": "medium",
        "impact_model": "overestimates_profit",
        "bias_pct_range": (0.02, 0.10),
        "default": {"discount_pct_of_revenue": 0.03},
        "requires": {"mode": "any"},
        "dtype": "numeric",
        "min_coverage": 0.25,
    },
    "tax": {
        "aliases": ["tax","vat","gst","sales_tax","tax_amount"],
        "criticality": "medium",
        "impact_model": "underestimates_revenue",
        "bias_pct_range": (0.01, 0.08),
        "default": {"tax_pct_of_revenue": 0.05},
        "requires": {"mode": "any"},
        "dtype": "numeric",
        "min_coverage": 0.25,
    },
    "returns": {
        "aliases": ["return","refund","chargeback","refunded","returned","chargebacked","return_id","refund_id"],
        "criticality": "medium",
        "impact_model": "underestimates_revenue",
        "bias_pct_range": (0.02, 0.15),
        "default": {"return_pct_of_revenue": 0.04},
        "requires": {"mode": "any"},
        "dtype": "flag_or_numeric",
        "min_coverage": 0.2,
    },

    # ===== Reach/attribution fidelity =====
    "unsubscribed": {
        "aliases": ["unsubscribed","opt_out","dnc","do_not_contact","suppressed"],
        "criticality": "medium",
        "impact_model": "overestimates_reach",
        "bias_pct_range": (0.05, 0.20),
        "default": {"unsub_rate": 0.12},
        "requires": {"mode": "any"},
        "dtype": "boolean",
        "min_coverage": 0.2,
    },
    "referral": {
        "aliases": ["referral","referred_by","ref_source","promo_code","affiliate_id","utm_source"],
        "criticality": "low",
        "impact_model": "underestimates_revenue",
        "bias_pct_range": (0.00, 0.05),
        "default": {"ref_share_pct": 0.0},
        "requires": {"mode": "any"},
        "dtype": "string_or_flag",
        "min_coverage": 0.1,
    },
}

@dataclass
class GapRecord:
    role: str
    status: str
    matched_cols: List[str]
    assumption: Any
    criticality: str
    likely_bias: str
    est_bias_pct_range: Tuple[float, float]
    message: str
    recommendation: str
    ui_override_key: Optional[str] = None
    coverage: float = 0.0
    severity: str = "low"

def _present_cols_for_role(
    role_cfg: Dict[str, Any],
    df_norm: "pd.DataFrame",
    header_map: Dict[str, Any],
) -> List[str]:
    """
    Return normalized column names detected for this role.
    - Matches on normalized names via aliases
    - Also matches canonical hints found in header_map -> normalized_name
    """
    aliases_raw = role_cfg.get("aliases") or []
    aliases = {str(a).strip().lower() for a in aliases_raw if a is not None}

    present_norm = [str(c).strip().lower() for c in getattr(df_norm, "columns", [])]
    present_set = set(present_norm)

    # direct alias matches (normalized)
    hits: List[str] = [a for a in aliases if a in present_set]

    # canonical hints -> normalized names (from header_map)
    try:
        norm_by_canon: Dict[str, List[str]] = {}
        for _orig, meta in (header_map or {}).items():
            canon = (
                (meta.get("canonical_hint") if isinstance(meta, dict) else getattr(meta, "canonical_hint", None))
                or ""
            ).strip().lower()
            norm = (
                (meta.get("normalized_name") if isinstance(meta, dict) else getattr(meta, "normalized_name", None))
                or ""
            ).strip().lower()
            if canon and norm:
                norm_by_canon.setdefault(canon, []).append(norm)

        for alias in aliases:
            for nn in norm_by_canon.get(alias, []):
                if nn:
                    hits.append(nn)
    except Exception:
        pass

    # uniquify while preserving first occurrence
    seen = set()
    uniq: List[str] = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            uniq.append(h)
    return uniq

def _human_bias_label(impact_model: str) -> str:
    return {
        "overestimates_reach": "May overestimate reachable audience",
        "underestimates_revenue": "May underestimate revenue",
        "overestimates_profit": "May overestimate net profit",
        "blocks_profit_calc": "Cannot compute profit reliably",
        "blocks_forecast": "Cannot run forecasting",
        "limits_personalization": "Limited user-level actions/caps",
    }.get(impact_model, "Unknown effect")

def _severity(criticality: str, est_range: Tuple[float, float]) -> str:
    _, hi = est_range
    if criticality == "high":
        return "high"
    if hi >= 0.10:
        return "high"
    if hi >= 0.05:
        return "medium"
    return "low"

def _recommendation(role: str, status: str, likely_bias: str) -> str:
    if status == "present":
        return "No action needed."
    recs = {
        "unsubscribe": "Provide an unsubscribed/opt-out column or confirm an assumed unsubscribed rate.",
        "referral": "Supply a referral/referred_by column or confirm an assumed referral revenue share.",
        "discount": "Provide discount/coupon amounts or confirm an assumed discount rate.",
        "returns": "Provide returns/chargeback fields or confirm an assumed return loss rate.",
        "revenue": "Map a revenue/sales/amount column; without it, profit cannot be estimated.",
        "date": "Map a date/timestamp column to enable time-based analysis & forecasts.",
        "user_id": "Provide a user or customer identifier to enable per-user caps and deduplication.",
    }
    base = recs.get(role, "Consider adding the missing field or confirm an assumption.")
    return f"{base} ({likely_bias})"
def detect_columns(df: pd.DataFrame) -> dict:
    hints = {}
    cols = [c for c in df.columns if isinstance(c, str)]
    lower = {c: c.lower() for c in cols}

    # Heuristics for classification target
    target_aliases = [
        "target","label","class","y","outcome","purchased","converted","conversion",
        "churn","is_churn","fraud","is_fraud","refund","refunded","is_refund",
        "subscribed","is_subscribed","active","is_active","cancelled","is_cancelled",
        "returned","is_returned","success","won","lost","is_new_customer","new_customer",
    ]
    # prefer boolean/low-cardinality columns that match aliases
    cand = [c for c in cols if any(a in lower[c] for a in target_aliases)]
    if not cand:
        # fallback: any boolean or 0/1 column
        boolish = []
        for c in cols:
            s = df[c].dropna()
            uniq = set(map(str, s.unique()))
            if uniq <= {"0","1"} or uniq <= {"true","false","True","False"} or s.dtype == "boolean":
                boolish.append(c)
        cand = boolish

    hints["target_cls_col"] = cand[0] if cand else None
    return hints

def _detect_data_gaps(df_norm, header_map, *, priors=None) -> Dict[str, GapRecord]:
    priors = priors or {}
    results: Dict[str, GapRecord] = {}

    for role, cfg in EXPECTED_SIGNALS.items():
        hits = _present_cols_for_role(cfg, df_norm, header_map)
        status = "present" if hits else "missing"

        est_range = priors.get(role, cfg.get("bias_pct_range", (0.0, 0.0)))
        likely_bias = _human_bias_label(cfg.get("impact_model", ""))
        assumption = cfg.get("default", None)

        nonnull_rates: List[float] = []
        if status == "present":
            for h in hits:
                try:
                    nonnull = 1.0 - float(pd.isna(df_norm[h]).mean())
                    nonnull_rates.append(nonnull)
                except Exception:
                    pass
            avg_nonnull = (sum(nonnull_rates) / len(nonnull_rates)) if nonnull_rates else 0.0
            if nonnull_rates and avg_nonnull < 0.2:
                status = "missing"
                hits = []
            else:
                assumption = None

        msg = (
            f"No {role} field detected; using default {assumption}"
            if status == "missing" and assumption is not None
            else (
                f"No {role} field detected; required for downstream tasks"
                if status == "missing"
                else f"{role} present: {', '.join(hits[:3])}" + ("…" if len(hits) > 3 else "")
            )
        )
        coverage = (sum(nonnull_rates) / len(nonnull_rates)) if nonnull_rates else 0.0
        sev = _severity(cfg.get("criticality", "low"), est_range if status == "missing" else (0.0, 0.0))

        results[role] = GapRecord(
            role=role,
            status=status,
            matched_cols=hits,
            assumption=assumption,
            criticality=cfg.get("criticality", "low"),
            likely_bias=likely_bias if status == "missing" else "None",
            est_bias_pct_range=est_range if status == "missing" else (0.0, 0.0),
            message=msg,
            recommendation=_recommendation(role, status, likely_bias),
            ui_override_key=cfg.get("ui_override_key"),
            coverage=coverage,
            severity=sev,
        )
    return results

# ============================== END OF FILE ===============================
