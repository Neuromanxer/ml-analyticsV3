"""
Intake & Normalization (Part 1) — Universal sniffing, header normalization, type inference, value standardization.

Drop-in usage:
    result = intake_and_normalize(file_path, base_currency="USD", country_hint="US")
    df_raw         = result.df_raw
    df_normalized  = result.df_normalized
    meta           = result.meta   # dialect, header_map, types, tags, currency/units info, anomalies, etc.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from difflib import SequenceMatcher

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
    thousands: Optional[str]  # None if not detected

@dataclass
class ColumnMeta:
    original_name: str
    normalized_name: str
    canonical_hint: Optional[str]
    tags: List[str] = field(default_factory=list)

@dataclass
class TypeInfo:
    semantic: str            # one of: id, email, phone, url, ip, json, boolean, timestamp, numeric, money, percent, categorical, text
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

@dataclass
class IntakeResult:
    df_raw: pd.DataFrame
    df_normalized: pd.DataFrame
    meta: IntakeMeta

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

    header_map = _normalize_headers(df_raw.columns)
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
    return IntakeResult(df_raw=df_raw, df_normalized=df_norm, meta=meta)

# -----------------------------
# 1) Universal sniffing
# -----------------------------
def _sniff_dialect_and_encoding(file_path: str, *, sample_bytes: int = 200_000) -> DialectInfo:
    raw = Path(file_path).read_bytes()[:sample_bytes]

    # Encoding
    enc = "utf-8"
    if raw.startswith(b"\xef\xbb\xbf"):
        enc = "utf-8-sig"
    elif chardet:
        try:
            enc_guess = chardet.detect(raw) or {}
            if enc_guess.get("encoding"):
                enc = enc_guess["encoding"]
        except Exception:
            pass

    # Delimiter & header
    text = raw.decode(enc, errors="replace")
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(text, delimiters=[",", ";", "\t", "|", "^"])
        delimiter = dialect.delimiter
        quotechar = dialect.quotechar or '"'
    except Exception:
        # Fallback
        delimiter, quotechar = _fallback_delimiter(text), '"'

    try:
        has_header = sniffer.has_header(text)
    except Exception:
        has_header = True

    # Decimal/thousands inference
    decimal, thousands = _infer_number_format_from_text(text)

    return DialectInfo(
        encoding=enc,
        delimiter=delimiter,
        quotechar=quotechar,
        has_header=bool(has_header),
        decimal=decimal,
        thousands=thousands,
    )

def _fallback_delimiter(text: str) -> str:
    # Count candidate delimiters on first non-empty lines
    lines = [ln for ln in text.splitlines() if ln.strip()][:10]
    counts = {",": 0, ";": 0, "\t": 0, "|": 0, "^": 0}
    for ln in lines:
        for d in counts:
            counts[d] += ln.count(d)
    return max(counts, key=counts.get) if counts else ","

def _infer_number_format_from_text(text: str) -> Tuple[str, Optional[str]]:
    # Look for numbers with . or , near digits
    # Heuristic: if we see patterns like "1.234,56" -> decimal ',' thousands '.', or "1,234.56" -> decimal '.' thousands ','
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
    # Support CSV/TSV; if Excel, fall back to read_excel
    suffix = Path(file_path).suffix.lower()
    if suffix in (".xls", ".xlsx", ".xlsm"):
        try:
            return pd.read_excel(file_path)
        except Exception:
            pass  # fallback to CSV attempt below

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
        # Minimal fallback
        return pd.read_csv(file_path, dtype=str, encoding_errors="replace")

def _strip_bom_and_trim(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    # Trim column names and drop zero-width spaces
    df2.columns = [re.sub(r"[\u200B-\u200D\uFEFF]", "", (c or "")).strip() for c in df2.columns]
    # Trim whitespace in string cells
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
    seen = {}
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
    s = (name or "").lower()
    best = None
    best_score = 0.0
    for canon, aliases in _ALIAS_LEXICON.items():
        for alias in aliases:
            score = SequenceMatcher(a=s, b=alias).ratio()
            if score > best_score:
                best_score = score
                best = canon
    return best if best_score >= 0.76 else None

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
        # quick gate
        gate = (series.str.startswith("{") | series.str.startswith("[")).mean()
        if gate < min_parse_rate:
            continue
        # attempt parse on a sample
        sample = series.sample(min(100, len(series)), random_state=0)
        ok = 0
        parsed_objects: List[Dict[str, Any]] = []
        for v in sample:
            try:
                pj = json.loads(v)
                if isinstance(pj, dict):
                    parsed_objects.append(pj)
                    ok += 1
            except Exception:
                pass
        if ok / max(1, len(sample)) < min_parse_rate or not parsed_objects:
            continue
        # collect keys and expand best-effort
        keys = set().union(*(obj.keys() for obj in parsed_objects if isinstance(obj, dict)))
        new_cols: List[str] = []
        for k in sorted(keys):
            new_name = _unique_name(df2, f"{col}__{_slugify(str(k))}")
            df2[new_name] = df2[col].map(lambda x: _safe_json_get(x, k))
            new_cols.append(new_name)
        expansions[col] = new_cols
    return df2, expansions

def _unique_name(df: pd.DataFrame, base: str) -> str:
    name = base
    i = 2
    while name in df.columns:
        name = f"{base}_{i}"
        i += 1
    return name

def _safe_json_get(raw: Any, k: str) -> Any:
    try:
        if isinstance(raw, (dict, list)):
            obj = raw
        else:
            obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj.get(k)
        return None
    except Exception:
        return None

# -----------------------------
# 4) Type inference + anomalies
# -----------------------------
_EMAIL_RE = re.compile(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$", re.I)
_PHONE_RE = re.compile(r"^\+?\d[\d\s().\-]{6,}$")
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
        n = len(nonnull)
        unique_ratio = (nonnull.nunique(dropna=True) / max(1, len(s))) if len(s) else 0.0
        sample = nonnull.astype(str).head(800)

        # detectors
        email_rate = _rate(sample, lambda x: bool(_EMAIL_RE.match(x)))
        phone_rate = _rate(sample, lambda x: bool(_PHONE_RE.match(x)))
        url_rate   = _rate(sample, lambda x: bool(_URL_RE.match(x)))
        ipv4_rate  = _rate(sample, lambda x: bool(_IPV4_RE.match(x)))
        bool_rate  = _rate(sample, lambda x: x.strip().lower() in _BOOL_STR)
        json_rate  = _rate(sample, _seems_json)

        # timestamp
        ts_rate, tz_found = _timestamp_parse_rate(sample)
        tz_guess = tz_guess or tz_found

        # numeric / money / percent
        money_symbol_hits = _count_currency_symbols(sample)
        for sym, cnt in money_symbol_hits.items():
            currency_counts[sym] = currency_counts.get(sym, 0) + cnt

        numeric_rate, as_numeric = _coerce_numeric_rate(sample, dialect.decimal, dialect.thousands)
        percent_hint = ("%" in col) or ("_pct" in col) or ("percent" in col) or ("rate" in col)

        # id-like heuristic
        id_like = unique_ratio > 0.9 and email_rate < 0.2 and phone_rate < 0.2 and ts_rate < 0.2 and numeric_rate < 0.3 and s.map(lambda x: isinstance(x, str) and len(str(x)) <= 64).mean() > 0.7

        # Decide semantic
        if email_rate > 0.6:
            sem = "email"; vr = email_rate
        elif phone_rate > 0.6:
            sem = "phone"; vr = phone_rate
        elif url_rate > 0.6:
            sem = "url"; vr = url_rate
        elif ipv4_rate > 0.6:
            sem = "ip"; vr = ipv4_rate
        elif bool_rate > 0.8:
            sem = "boolean"; vr = bool_rate
        elif json_rate > 0.6:
            sem = "json"; vr = json_rate
        elif ts_rate > 0.7:
            sem = "timestamp"; vr = ts_rate
        elif id_like:
            sem = "id"; vr = unique_ratio
        elif (money_symbol_hits.get("$",0)+money_symbol_hits.get("€",0)+money_symbol_hits.get("£",0)+money_symbol_hits.get("¥",0)) > 0.1*len(sample):
            sem = "money"; vr = (sum(money_symbol_hits.values())/max(1,len(sample)))
        elif percent_hint and numeric_rate > 0.5:
            sem = "percent"; vr = numeric_rate
        elif numeric_rate > 0.85:
            sem = "numeric"; vr = numeric_rate
        else:
            # categorical vs text
            avg_len = sample.map(len).mean() if len(sample) else 0
            if unique_ratio <= 0.05 and nonnull.nunique(dropna=True) <= 50:
                sem = "categorical"; vr = 1 - unique_ratio
            elif avg_len > 40:
                sem = "text"; vr = 0.6
            else:
                sem = "categorical"; vr = 0.5

        inferred[col] = TypeInfo(
            semantic=sem,
            pandas_dtype=str(df[col].dtype),
            unique_ratio=float(unique_ratio),
            valid_rate=float(vr),
            notes=None,
        )

        # Anomalies (basic)
        if sem in ("numeric","money","percent"):
            # look for impossible values
            try:
                vals = pd.to_numeric(as_numeric, errors="coerce")
                if sem == "percent":
                    over_1 = (vals > 1.0).mean()
                    if over_1 > 0.5 and "%" not in col.lower():
                        anomalies.append(f"{col}: percent-like but majority values >1 (consider scaling by 100 or mark as %) ")
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
    if len(series) == 0: return 0.0, None
    parsed = pd.to_datetime(series, errors="coerce", utc=False, infer_datetime_format=True)
    rate = parsed.notna().mean()
    # timezone inference (very light-touch): if explicit tz suffix observed
    tz_guess = None
    for v in series.head(200):
        m = re.search(r"(Z|[+-]\d{2}:\d{2})$", str(v).strip())
        if m:
            tz_guess = "offset_in_values"
            break
    return float(rate), tz_guess

def _coerce_numeric_rate(series: pd.Series, decimal: str, thousands: Optional[str]) -> Tuple[float, pd.Series]:
    s = series.astype(str)
    # remove thousands separators, normalize decimal to '.'
    if thousands:
        s = s.str.replace(re.escape(thousands), "", regex=True)
    if decimal != ".":
        s = s.str.replace(decimal, ".", regex=False)
    # remove currency symbols/spaces/%
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

    # General whitespace de-dup
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(str).str.replace(r"\s+", " ", regex=True).str.strip().replace({"nan": None, "NaN": None})

    for col, info in inferred_types.items():
        sem = info.semantic
        # Emails
        if sem == "email":
            out[col] = out[col].str.lower().str.strip()

        # Phones → pseudo E.164 (best-effort, no external deps)
        if sem == "phone":
            out[col] = out[col].apply(lambda x: _to_e164_like(str(x), country_hint=country_hint))

        # Booleans
        if sem == "boolean":
            out[col] = out[col].map(lambda x: str(x).strip().lower() if pd.notna(x) else x)\
                               .map({"true":True,"t":True,"yes":True,"y":True,"1":True,
                                     "false":False,"f":False,"no":False,"n":False,"0":False})

        # Timestamp → pandas datetime (naive; do not localize)
        if sem == "timestamp":
            out[col] = pd.to_datetime(out[col], errors="coerce", infer_datetime_format=True)

        # Money / Numeric / Percent
        if sem in ("money","numeric","percent"):
            s = out[col].astype(str)
            if thousands:
                s = s.str.replace(re.escape(thousands), "", regex=True)
            if decimal != ".":
                s = s.str.replace(decimal, ".", regex=False)
            # strip symbols and percent
            s = s.str.replace(r"[$€£¥]", "", regex=True)
            perc = False
            if sem == "percent" or "%" in col.lower() or "pct" in col.lower() or "percent" in col.lower():
                perc = True
            vals = pd.to_numeric(s.str.replace("%", "", regex=False), errors="coerce")

            if sem == "percent" and vals.notna().any():
                # If majority >1, assume already 0-100 and scale to 0-1
                if (vals > 1).mean() > 0.5:
                    out[col + "_rate"] = vals / 100.0
                else:
                    out[col + "_rate"] = vals
            else:
                out[col + "_num"] = vals

        # Categorical → slug
        if sem == "categorical":
            out[col + "_slug"] = out[col].astype(str).map(_slug_for_value)

        # JSON stays as-is (already expanded earlier if configured)

    # Currency-to-base (only if clearly uniform)
    _maybe_add_base_money_columns(out, inferred_types, base_currency, anomalies)

    # Units canonicalization from header hints (e.g., *_oz → grams)
    _canonicalize_units_from_headers(out)

    return out

def _to_e164_like(s: str, *, country_hint: str = "US") -> Optional[str]:
    if not s or s.lower() == "nan": return None
    digits = re.sub(r"\D", "", s)
    if not digits:
        return None
    # simple US default
    if country_hint.upper() in ("US","CA") and len(digits) == 10:
        return "+1" + digits
    if s.strip().startswith("+"):
        return "+" + digits
    # fallback: return digits with plus
    return "+" + digits

def _slug_for_value(x: Any) -> Optional[str]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or None

def _maybe_add_base_money_columns(df: pd.DataFrame, inferred_types: Dict[str, TypeInfo], base_currency: str, anomalies: List[str]) -> None:
    # If a column is money and contains a uniform currency symbol or header tag matching base_currency, produce *_base
    sym_for = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
    sym = sym_for.get(base_currency.upper())
    for col, ti in inferred_types.items():
        if ti.semantic != "money":
            continue
        s = df[col].astype(str)
        has_sym = s.str.contains(re.escape(sym)) if sym else pd.Series([False]*len(s))
        # If majority rows include base symbol OR header tag *_usd etc
        header_lower = col.lower()
        tagged = any(tag in header_lower for tag in [f"_{base_currency.lower()}", base_currency.lower()])
        if (has_sym.mean() > 0.6) or tagged:
            # make numeric base column
            nums = s.str.replace(r"[$€£¥]", "", regex=True)
            nums = nums.str.replace(",", "", regex=False)
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
        if not col.lower().endswith(tuple(["_oz","_lb","_kg","_g","_ml","_l","_floz"])):
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
            # look for numerical companions we created: *_num or *_rate or *_usd
            targets = [c for c in out.columns if c.startswith(col + "_") and out[c].dtype != object]
            for t in targets:
                series = pd.to_numeric(out[t], errors="coerce")
                if series.notna().sum() < 50:
                    continue
                lo = series.quantile(lower_q)
                hi = series.quantile(upper_q)
                out[t + "_capped"] = series.clip(lower=lo, upper=hi)
    return out
# app/routers/datasets_intake.py
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Any, Dict
import json, os, tempfile, shutil
import pandas as pd
from dataclasses import asdict

# --- your project imports (adjust paths as needed)
from auth import get_current_active_user, get_user_db, Dataset
from storage import upload_file_to_supabase
from datasets import resolve_and_cache_dataset_csv   # if you have this helper

router = APIRouter(prefix="/datasets", tags=["datasets"])

@router.post("/{dataset_id}/intake")
def run_intake_and_normalization(
    dataset_id: int,
    base_currency: str = Query("USD"),
    country_hint: str = Query("US"),
    preview_rows: int = Query(50, ge=1, le=500),
    current_user = Depends(get_current_active_user),
):
    """
    Reads the dataset CSV for `dataset_id`, runs Intake & Normalization,
    stores artifacts, and returns a JSON-friendly summary + table previews.
    """
    user_id = getattr(current_user, "id", getattr(current_user, "user_id", None))
    if user_id is None:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    # 1) Resolve the *local* CSV path for this dataset
    # Prefer your existing helper if available:
    src_csv_path = None
    try:
        with get_user_db(current_user) as db:
            src_csv_path = resolve_and_cache_dataset_csv(db=db, dataset_id=dataset_id, user_id=user_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to resolve dataset CSV: {e}")

    # Fallback example (if you don't have resolve_and_cache_dataset_csv here):
    # with get_user_db(current_user) as db:
    #     ds = db.query(Dataset).filter(Dataset.id == int(dataset_id)).first()
    #     if not ds:
    #         raise HTTPException(status_code=404, detail="Dataset not found")
    #     # Download from Supabase to a temp file
    #     src_csv_path = _download_dataset_csv_to_temp(ds.file_path, user_id=user_id, dataset_id=dataset_id)

    if not src_csv_path or not os.path.exists(src_csv_path):
        raise HTTPException(status_code=404, detail="Local CSV for dataset not found")

    # 2) Run Intake & Normalization (from prior code)
    try:
        result = intake_and_normalize(
            src_csv_path,
            base_currency=base_currency,
            country_hint=country_hint,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Intake & normalization failed: {e}")

    # 3) Persist artifacts (normalized CSV + meta JSON) to Supabase
    artifacts: Dict[str, Any] = {}
    temp_dir = tempfile.mkdtemp(prefix=f"d{dataset_id}_intake_")
    try:
        # Save normalized CSV (full)
        norm_csv = os.path.join(temp_dir, f"dataset_{dataset_id}_normalized.csv")
        result.df_normalized.to_csv(norm_csv, index=False)

        # Save meta JSON (JSON-friendly)
        meta_json_path = os.path.join(temp_dir, f"dataset_{dataset_id}_intake_meta.json")
        meta_dict = _meta_to_json(result.meta)
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, ensure_ascii=False, indent=2)

        # Upload artifacts to Supabase
        supa_norm = upload_file_to_supabase(
            user_id=str(user_id),
            file_path=norm_csv,
            filename=f"{dataset_id}/intake/normalized.csv",
        )
        supa_meta = upload_file_to_supabase(
            user_id=str(user_id),
            file_path=meta_json_path,
            filename=f"{dataset_id}/intake/intake_meta.json",
        )
        artifacts["normalized_csv"] = supa_norm
        artifacts["intake_meta_json"] = supa_meta

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to persist artifacts: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # 4) Build lightweight previews (do NOT return full DataFrames)
    preview_raw = _preview_table(result.df_raw, n=preview_rows)
    preview_norm = _preview_table(result.df_normalized, n=preview_rows)

    # 5) Response
    return {
        "ok": True,
        "dataset_id": dataset_id,
        "artifacts": artifacts,
        "meta": meta_dict,
        "preview": {
            "raw": preview_raw,
            "normalized": preview_norm,
        },
        "stats": {
            "raw_rows": int(result.df_raw.shape[0]),
            "raw_cols": int(result.df_raw.shape[1]),
            "normalized_rows": int(result.df_normalized.shape[0]),
            "normalized_cols": int(result.df_normalized.shape[1]),
        },
    }


# ---------------------------
# Helpers
# ---------------------------

def _preview_table(df: pd.DataFrame, n: int = 50):
    # Convert an n-row sample to JSON records; keep types simple
    if df is None or df.empty:
        return {"columns": [], "rows": []}
    sample = df.head(n).copy()
    # Ensure datetimes are ISO strings; numpy types → Python
    for c in sample.columns:
        if str(sample[c].dtype).startswith("datetime"):
            sample[c] = sample[c].astype("datetime64[ns]").dt.strftime("%Y-%m-%dT%H:%M:%S")
    return {
        "columns": list(sample.columns),
        "rows": sample.where(pd.notnull(sample), None).to_dict(orient="records"),
    }

def _meta_to_json(meta) -> Dict[str, Any]:
    """Turn IntakeMeta into JSON-safe dict."""
    # If meta is a dataclass (IntakeMeta), asdict() will recurse.
    data = asdict(meta)

    # Ensure all floats/NaNs are JSON-safe
    def _clean(obj):
        if isinstance(obj, float):
            if pd.isna(obj):
                return None
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(x) for x in obj]
        return obj

    return _clean(data)


# Example fallback downloader (if you need it)
# def _download_dataset_csv_to_temp(object_key: str, *, user_id: int, dataset_id: int) -> str:
#     """
#     Copy/download object to a temp CSV and return its path (local).
#     - If object_key is already a local path, copy to a tmp file and return it.
#     - Else download from Supabase using your storage client.
#     """
#     import tempfile, os, shutil
#     from app.storage import supabase, SUPABASE_BUCKET, _strip_bucket_prefix
#
#     # Local path?
#     if object_key and os.path.exists(object_key):
#         fd, temp_path = tempfile.mkstemp(suffix=".csv"); os.close(fd)
#         shutil.copy2(object_key, temp_path)
#         return temp_path
#
#     # Supabase download
#     key = _strip_bucket_prefix(object_key)
#     data = supabase.storage.from_(SUPABASE_BUCKET).download(key)
#     fd, temp_path = tempfile.mkstemp(suffix=".csv"); os.close(fd)
#     with open(temp_path, "wb") as f:
#         f.write(data)
#     return temp_path
