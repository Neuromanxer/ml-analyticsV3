from __future__ import annotations

# ============================== ml_intake.py ===============================
# Drop-in, pasteable module for intake & normalization with gap priors
# ==========================================================================

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
class ColumnMeta:
    original_name: str
    normalized_name: str
    canonical_hint: Optional[str]
    tags: List[str] = field(default_factory=list)

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

# -----------------------------
# Core entry
# -----------------------------
def intake_and_normalize(
    file_path: str,
    *,
    base_currency: str = "USD",
    country_hint: str = "US",
    sample_bytes: int = 200_000,
    parse_json_cells: bool = True,
) -> IntakeResult:
    dialect = _sniff_dialect_and_encoding(file_path, sample_bytes=sample_bytes)
    df_raw = _read_dataframe(file_path, dialect)
    df_raw = _strip_bom_and_trim(df_raw)

    header_map = _normalize_headers(df_raw.columns.tolist())
    df = df_raw.rename(columns={c: header_map[c].normalized_name for c in df_raw.columns})

    # Detect JSON-in-cell
    json_expansions: Dict[str, List[str]] = {}
    nested_csv_cols: List[str] = []
    if parse_json_cells:
        df, json_expansions = _expand_json_columns(df)

    # Type inference
    inferred, tz_guess, currency_counts, anomalies = _infer_types_and_anomalies(df, header_map, dialect)

    # Value standardization into a NEW normalized frame (do not mutate raw)
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

    # Winsorize guard rails (light-touch, separate from raw)
    df_norm = _winsorize_numeric(df_norm, inferred, lower_q=0.01, upper_q=0.99)

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

    # Detect gaps
    gap_report = _detect_data_gaps(df_norm, header_map)
    meta.data_gaps = {k: asdict(v) for k, v in gap_report.items()}

    # Collect defaults only for missing keys with a real default
    meta.assumptions = {
        k: v.assumption
        for k, v in gap_report.items()
        if v.status == "missing" and v.assumption is not None
    }

    # Build priors (posterior = prior + proxies)
    build_gap_priors(df_norm, meta, industry_priors=INDUSTRY_PRIORS)

    # attach lightweight stats
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

def _strip_bom_and_trim(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [re.sub(r"[\u200B-\u200D\uFEFF]", "", (c or "")).strip() for c in df2.columns]
    for c in df2.columns:
        if df2[c].dtype == object:
            df2[c] = df2[c].astype(str).str.strip()
            df2[c] = df2[c].replace({"nan": None, "NaN": None})
    return df2

# -----------------------------
# 2) Header normalization
# -----------------------------
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

def _normalize_headers(columns: List[str]) -> Dict[str, ColumnMeta]:
    out: Dict[str, ColumnMeta] = {}
    for col in columns:
        original = col
        norm = _slugify(original)
        tags = _tags_from_header(original)
        canonical = _canonical_hint(original)
        out[original] = ColumnMeta(original_name=original, normalized_name=norm, canonical_hint=canonical, tags=tags)
    # ensure uniqueness
    seen: Dict[str, int] = {}
    for meta in out.values():
        name = meta.normalized_name
        if name not in seen:
            seen[name] = 1
        else:
            seen[name] += 1
            meta.normalized_name = f"{name}_{seen[name]}"
    return out

def _slugify(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    s = re.sub(r"[^a-z0-9%]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "col"

def _tags_from_header(name: str) -> List[str]:
    tags: List[str] = []
    s = (name or "").lower()
    for suf in _UNIT_SUFFIXES:
        if s.endswith(suf):
            tags.append(suf.strip("_"))
    for sym, tag in _SYMBOL_TAGS.items():
        if sym in name:
            tags.append(tag)
    return sorted(set(tags))

def _canonical_hint(name: str) -> Optional[str]:
    s = (name or "").strip().lower()
    s_norm = re.sub(r"[^a-z0-9]+", "_", s).strip("_")

    # 1) exact / tokenized shortcuts
    if s_norm == "id":
        return "customer_id"  # safer default for marketing/customer datasets
    if s_norm.endswith("_id"):
        if "order" in s_norm:
            return "order_id"
        if "customer" in s_norm or "user" in s_norm or "client" in s_norm:
            return "customer_id"

    # 2) exact alias equality
    for canon, aliases in _ALIAS_LEXICON.items():
        for alias in aliases:
            if s_norm == re.sub(r"[^a-z0-9]+", "_", alias).strip("_"):
                return canon

    # 3) fuzzy fallback
    best, best_score = None, 0.0
    for canon, aliases in _ALIAS_LEXICON.items():
        for alias in aliases:
            score = SequenceMatcher(a=s, b=alias).ratio()
            if score > best_score:
                best, best_score = canon, score
    return best if best_score >= 0.80 else None

# -----------------------------
# 3) JSON-in-cell expansion
# -----------------------------
def _expand_json_columns(df: pd.DataFrame, min_parse_rate: float = 0.3) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    expansions: Dict[str, List[str]] = {}
    df2 = df.copy()
    for col in list(df2.columns):
        series = df2[col].dropna().astype(str)
        if series.empty:
            continue
        gate = (series.str.startswith("{") | series.str.startswith("[")).mean()
        if gate < min_parse_rate:
            continue
        sample = series.sample(min(100, len(series)), random_state=0)
        ok = 0
        parsed_objects: List[Dict[str, Any]] = []
        for v in sample:
            try:
                pj = json.loads(v)
                if isinstance(pj, dict):
                    parsed_objects.append(pj); ok += 1
            except Exception:
                pass
        if ok / max(1, len(sample)) < min_parse_rate or not parsed_objects:
            continue
        keys = set().union(*(obj.keys() for obj in parsed_objects if isinstance(obj, dict)))
        new_cols: List[str] = []
        for k in sorted(keys):
            new_name = _unique_name(df2, f"{col}__{_slugify(str(k))}")
            df2[new_name] = df2[col].map(lambda x: _safe_json_get(x, k))
            new_cols.append(new_name)
        expansions[col] = new_cols
    return df2, expansions

def _unique_name(df: pd.DataFrame, base: str) -> str:
    name = base; i = 2
    while name in df.columns:
        name = f"{base}_{i}"; i += 1
    return name

def _safe_json_get(raw: Any, k: str) -> Any:
    try:
        obj = raw if isinstance(raw, (dict, list)) else json.loads(raw)
        return obj.get(k) if isinstance(obj, dict) else None
    except Exception:
        return None

# -----------------------------
# 4) Type inference + anomalies
# -----------------------------
_EMAIL_RE = re.compile(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$", re.I)
_URL_RE   = re.compile(r"^https?://", re.I)
_IPV4_RE  = re.compile(r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)$")
_BOOL_STR = {"true","false","yes","no","y","n","1","0","t","f"}

def _infer_types_and_anomalies(
    df: pd.DataFrame,
    header_map: Dict[str, ColumnMeta],
    dialect: DialectInfo,
) -> Tuple[Dict[str, TypeInfo], Optional[str], Dict[str, int], List[str]]:
    inferred: Dict[str, TypeInfo] = {}
    anomalies: List[str] = []
    tz_guess: Optional[str] = None
    currency_counts: Dict[str, int] = {}

    for col in df.columns:
        s = df[col]
        nonnull = s.dropna()
        unique_ratio = (nonnull.nunique(dropna=True) / max(1, len(s))) if len(s) else 0.0
        sample = nonnull.astype(str).head(800)

        email_rate = _rate(sample, lambda x: bool(_EMAIL_RE.match(x)))
        phone_rate = _rate(sample, _is_phone_like)
        url_rate   = _rate(sample, lambda x: bool(_URL_RE.match(x)))
        ipv4_rate  = _rate(sample, lambda x: bool(_IPV4_RE.match(x)))
        bool_rate  = _rate(sample, lambda x: x.strip().lower() in _BOOL_STR)
        json_rate  = _rate(sample, _seems_json)

        ts_rate, tz_found = _timestamp_parse_rate(sample)
        tz_guess = tz_guess or tz_found

        money_symbol_hits = _count_currency_symbols(sample)
        for sym, cnt in money_symbol_hits.items():
            currency_counts[sym] = currency_counts.get(sym, 0) + cnt

        numeric_rate, as_numeric = _coerce_numeric_rate(sample, dialect.decimal, dialect.thousands)
        percent_hint = ("%" in col) or ("_pct" in col) or ("percent" in col) or ("rate" in col)

        id_like = unique_ratio > 0.9 and email_rate < 0.2 and phone_rate < 0.2 and ts_rate < 0.2 and numeric_rate < 0.3 \
                  and s.map(lambda x: isinstance(x, str) and len(str(x)) <= 64).mean() > 0.7

        if email_rate > 0.6:
            sem, vr = "email", email_rate
        elif ts_rate > 0.5:  # tolerate messy/partial timestamps
            sem, vr = "timestamp", ts_rate
        elif phone_rate > 0.6:
            sem, vr = "phone", phone_rate
        elif url_rate > 0.6:
            sem, vr = "url", url_rate
        elif ipv4_rate > 0.6:
            sem, vr = "ip", ipv4_rate
        elif bool_rate > 0.8:
            sem, vr = "boolean", bool_rate
        elif json_rate > 0.6:
            sem, vr = "json", json_rate
        elif id_like:
            sem, vr = "id", unique_ratio
        elif (money_symbol_hits.get("$",0)+money_symbol_hits.get("€",0)+money_symbol_hits.get("£",0)+money_symbol_hits.get("¥",0)) > 0.1*len(sample):
            sem, vr = "money", (sum(money_symbol_hits.values())/max(1,len(sample)))
        elif percent_hint and numeric_rate > 0.5:
            sem, vr = "percent", numeric_rate
        elif numeric_rate > 0.85:
            sem, vr = "numeric", numeric_rate
        else:
            avg_len = sample.map(len).mean() if len(sample) else 0
            if unique_ratio <= 0.05 and nonnull.nunique(dropna=True) <= 50:
                sem, vr = "categorical", 1 - unique_ratio
            elif avg_len > 40:
                sem, vr = "text", 0.6
            else:
                sem, vr = "categorical", 0.5

        inferred[col] = TypeInfo(
            semantic=sem,
            pandas_dtype=str(df[col].dtype),
            unique_ratio=float(unique_ratio),
            valid_rate=float(vr),
            notes=None,
        )

        if sem in ("numeric","money","percent"):
            try:
                vals = pd.to_numeric(as_numeric, errors="coerce")
                if sem == "percent":
                    over_1 = (vals > 1.0).mean()
                    if over_1 > 0.5 and "%" not in col.lower():
                        anomalies.append(f"{col}: percent-like but majority values >1 (consider scaling by 100 or mark as %)")
            except Exception:
                pass

    return inferred, tz_guess, currency_counts, anomalies

def _rate(series: pd.Series, fn) -> float:
    if len(series) == 0:
        return 0.0
    ok = 0
    for v in series:
        try:
            if fn(v):
                ok += 1
        except Exception:
            pass
    return ok / len(series)

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

def _timestamp_parse_rate(series: pd.Series) -> Tuple[float, Optional[str]]:
    if len(series) == 0:
        return 0.0, None
    parsed = pd.to_datetime(series, errors="coerce", utc=False, infer_datetime_format=True)
    rate = parsed.notna().mean()
    tz_guess = None
    for v in series.head(200):
        m = re.search(r"(Z|[+-]\d{2}:\d{2})$", str(v).strip())
        if m:
            tz_guess = "offset_in_values"
            break
    return float(rate), tz_guess

def _coerce_numeric_rate(series: pd.Series, decimal: str, thousands: Optional[str]) -> Tuple[float, pd.Series]:
    s = series.astype(str)
    if thousands:
        s = s.str.replace(re.escape(thousands), "", regex=False)
    if decimal != ".":
        s = s.str.replace(decimal, ".", regex=False)
    s = s.str.replace(r"[$€£¥]", "", regex=True).str.replace("%", "", regex=False)
    s2 = pd.to_numeric(s, errors="coerce")
    return float(s2.notna().mean()), s2

def _count_currency_symbols(series: pd.Series) -> Dict[str, int]:
    counts = {"$":0, "€":0, "£":0, "¥":0}
    for v in series:
        s = str(v)
        for sym in counts:
            if sym in s:
                counts[sym] += 1
    return counts

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
    out = df.copy()

    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().replace({"nan": None, "NaN": None})

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
            out[col] = pd.to_datetime(out[col], errors="coerce", infer_datetime_format=True)

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
            targets = [c for c in out.columns if c.startswith(col + "_") and out[c].dtype != object]
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
