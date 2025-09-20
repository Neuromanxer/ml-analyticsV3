# planner_router.py
from __future__ import annotations
import json
from typing import Optional, Any, Dict
import pandas as pd
from fastapi import APIRouter, Depends, File, UploadFile, Request, HTTPException
import math
import time 

# --- Mapping helpers ----------------------------------------------------------
import re
from typing import Dict, List, Optional
import pandas as pd
import io, os, re, mimetypes, tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import uuid
from auth import get_current_active_user, User
from storage import upload_file_to_supabase
from planner import compile_plan, signals_from_meta, derive_signals_for_compile, SignalsIn, merge_signals
from datasets import load_intake_meta, load_or_backfill_intake_meta
from storage import sb_download_to_path
#from .planner import compile_plan  # import your function
# from .auth import get_current_active_user
# from .storage import upload_file_to_supabase
router = APIRouter(prefix="/api/plan", tags=["planner"])
CSV_PREVIEW_ROWS = 10

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# Simple in-memory caches keyed by dataset_id
SIGNALS_STORE: Dict[int, SignalsIn] = {}
PREVIEW_STORE: Dict[int, list[dict]] = {}

def _norm(s: str) -> str:
    return re.sub(r"(^_+|_+$)", "",
           re.sub(r"[^\w]+","_", s.strip().lower()))

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_norm(c) for c in out.columns]
    return out
# ---- helpers / fixes ---------------------------------------------------------

from typing import Optional, Union, List, Dict, Any
import pandas as pd

def _as_float(x: Any, default: Optional[float] = 0.0) -> Optional[float]:
    """
    Robust float parsing. Returns `default` (which can be None) if not parseable or non-finite.
    """
    try:
        v = float(x)
        if not (v == v) or v in (float("inf"), float("-inf")):  # NaN/Inf
            return default
        return v
    except Exception:
        return default

def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default

def apply_common_transforms(df: pd.DataFrame, key_field: Optional[str]=None, user_tz: Optional[str]=None):
    """
    Safer TZ handling: localize naive → user_tz; convert aware → user_tz; else ISO UTC.
    """
    transforms = []
    if df is None or df.empty:
        return df, transforms

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        transforms.append(f"trim_whitespace:{col}")
        transforms.append(f"collapse_spaces:{col}")

    if "email" in df.columns:
        df["email"] = df["email"].astype(str).str.lower()
        transforms.append("lowercase_email")

    if "phone" in df.columns:
        df["phone"] = clean_phone(df["phone"])
        transforms.append("normalize_phone")

    date_cols = [c for c in df.columns if ("date" in c) or ("created_at" in c)]
    for col in date_cols:
        try:
            dt = pd.to_datetime(df[col], errors="coerce", utc=False)
            if user_tz:
                # if tz-naive → localize; if tz-aware → convert
                if getattr(dt.dt, "tz", None) is None:
                    dt = dt.dt.tz_localize(user_tz)
                else:
                    dt = dt.dt.tz_convert(user_tz)
                df[col] = dt.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
            else:
                # canonical UTC 'Z'
                dt = dt.dt.tz_localize("UTC") if getattr(dt.dt, "tz", None) is None else dt.dt.tz_convert("UTC")
                df[col] = dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            transforms.append(f"parse_date:{col}")
        except Exception:
            # leave as-is if parsing fails
            pass

    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        transforms.append("drop_exact_duplicates")

    if key_field and key_field in df.columns:
        before = len(df)
        df = df.drop_duplicates(key_field, keep="last")
        if len(df) < before:
            transforms.append(f"dedupe_on_key:{key_field}")

    return df, transforms

def _auto_map(df: pd.DataFrame, synonyms: Dict[str, List[str]]) -> Dict[str, Optional[str]]:
    """
    Return {canonical_field: original_column_name or None}.
    Tries exact normalized match, then 'contains' fallback.
    """
    if df is None or df.empty: 
        return {k: None for k in synonyms.keys()}
    # map normalized->original
    norm2orig = {_norm(c): c for c in df.columns}
    present_norm = list(norm2orig.keys())

    resolved: Dict[str, Optional[str]] = {}
    for field, cands in synonyms.items():
        hit = None
        # 1) exact normalized match
        for cand in cands:
            if _norm(cand) in norm2orig:
                hit = norm2orig[_norm(cand)]
                break
        # 2) contains fallback
        if not hit:
            for cand in cands:
                cn = _norm(cand)
                for coln in present_norm:
                    if cn in coln or coln in cn:
                        hit = norm2orig[coln]
                        break
                if hit: break
        resolved[field] = hit
    return resolved

# --- Synonym dictionaries (tweak as you like) --------------------------------
CUSTOMER_SYNS = {
  "customer_id": ["customer_id","cust_id","cid","user_id","id"],
  "email":       ["email","customer_email","email_address"],
  "phone":       ["phone","phone_number","mobile","cell","tel"],
  "created_at":  ["created_at","created_date","signup_date","joined","first_seen"]
}
ORDER_SYNS = {
  "order_id":    ["order_id","oid","transaction_id","id"],
  "customer_id": ["customer_id","cust_id","cid","user_id","id"],
  "order_date":  ["order_date","created_at","date","timestamp","order_created_at","processed_at"],
  "order_total": ["order_total","total","amount","revenue","price","subtotal_price","total_price"],
  "status":      ["status","order_status"],
  "product_cat": ["product_category","category","product_type","collection","collection_title"]
}
MKT_SYNS = {
  "campaign_id": ["campaign_id","campaign","message_id","id"],
  "customer_id": ["customer_id","cust_id","cid","user_id","id"],
  "channel":     ["channel","medium","source"],
  "sent_date":   ["sent_date","send_date","timestamp","created_at","sent_at","delivered_at"],
  "opened":      ["opened","open","is_open"],
  "clicked":     ["clicked","click","is_click"],
  "converted":   ["converted","purchase","is_purchase","order"],
  "cost":        ["cost","campaign_cost","spend"]
}
# Put near your other constants
SCHEMA_CONTRACT = {
    "customers": {
        "required":    ["customer_id"],
        "recommended": ["email", "phone", "created_at"],
        "optional":    ["country", "marketing_opt_in"],
        "date_fields": ["created_at"],
        "key_field":   "customer_id",
    },
    "orders": {
        "required":    ["order_id", "customer_id", "order_date", "amount"],
        "recommended": ["status", "product_cat"],
        "optional":    [],
        "date_fields": ["order_date"],
        "key_field":   "order_id",
    },
    "marketing": {
        "required":    ["campaign_id", "customer_id", "channel", "sent_date"],
        "recommended": ["opened", "clicked", "converted", "cost"],
        "optional":    [],
        "date_fields": ["sent_date"],
        "key_field":   None,
    },
}

def finalize_mapping(
    df: pd.DataFrame,
    proposed: Optional[Dict[str, Optional[str]]],
    synonyms: Dict[str, List[str]],
) -> Dict[str, Optional[str]]:
    """
    Prefer the frontend-proposed mapping if it exists and the column is present;
    otherwise auto-map using synonyms.
    Returns a dict of canonical_field -> original column name (or None).
    """
    if df is None:
        return {k: None for k in synonyms}
    df_normcols = set([_norm(c) for c in df.columns])
    proposed = proposed or {}
    resolved = {}
    # try proposed first
    for field in synonyms.keys():
        col = proposed.get(field)
        if col and _norm(col) in df_normcols:
            resolved[field] = col
        else:
            # fallback to auto
            auto = _auto_map(df, {field: synonyms[field]})
            resolved[field] = auto[field]
    return resolved
def to_dt(s: Optional[pd.Series]) -> Optional[pd.Series]:
    return pd.to_datetime(s, errors="coerce") if s is not None else None

def to_num_currency(s: Optional[pd.Series]) -> Optional[pd.Series]:
    if s is None: return None
    s = s.astype(str).str.replace(r"[\$,]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def clean_email(s: Optional[pd.Series]) -> Optional[pd.Series]:
    if s is None: return None
    s = s.astype(str).str.strip().str.lower()
    return s.where(s.map(lambda x: bool(EMAIL_RE.match(x))), pd.NA)

def clean_phone(s: Optional[pd.Series]) -> Optional[pd.Series]:
    if s is None: return None
    digits = s.astype(str).str.replace(r"\D", "", regex=True)
    return digits.where(digits.str.len() >= 10, pd.NA)

def to_bool01(s: Optional[pd.Series]) -> Optional[pd.Series]:
    if s is None: return None
    m = {"true":1,"yes":1,"y":1,"1":1,"false":0,"no":0,"n":0,"0":0}
    out = s.astype(str).str.strip().str.lower().map(m)
    return out.astype("Int64")  # keeps NA
import numpy as np
import pandas as pd
from typing import Dict, Optional, Sequence, Union

def coerce_customers(df: Optional[pd.DataFrame], m: Dict[str, object]) -> pd.DataFrame:
    """
    Normalize a Customers table to:
      - customer_id (string, key)
      - email (string, cleaned + validated)
      - phone (string, cleaned to digits; invalid -> NA)
      - created_at (datetime64[ns])
    Mapping values may be str or list[str]; we collapse to a single Series.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["customer_id", "email", "phone", "created_at"])

    # 1) normalize + collapse dupes (prevents df["col"] yielding a DataFrame)
    df = normalize_headers(df)
    df = _collapse_duplicate_columns(df)

    out = pd.DataFrame(index=df.index)

    # 2) resolve mappings (support str or list)
    cid_s   = _pick_or_collapse(df, m.get("customer_id") or m.get("cid") or m.get("user_id") or m.get("id"))
    email_s = _pick_or_collapse(df, m.get("email")       or m.get("customer_email") or m.get("email_address"))
    phone_s = _pick_or_collapse(df, m.get("phone")       or m.get("phone_number")   or m.get("mobile") or m.get("cell") or m.get("tel"))
    crt_s   = _pick_or_collapse(df, m.get("created_at")  or m.get("created_date")   or m.get("signup_date") or m.get("joined") or m.get("first_seen"))

    # 3) assign with safe dtypes
    out["customer_id"] = (cid_s.astype("string") if cid_s is not None
                          else pd.Series([pd.NA]*len(df), dtype="string"))

    if email_s is not None:
        out["email"] = clean_email(email_s)  # should return nullable series
        # ensure dtype string
        out["email"] = out["email"].astype("string")
    else:
        out["email"] = pd.Series([pd.NA]*len(df), dtype="string")

    if phone_s is not None:
        out["phone"] = clean_phone(phone_s)  # should return nullable series
        out["phone"] = out["phone"].astype("string")
    else:
        out["phone"] = pd.Series([pd.NA]*len(df), dtype="string")

    out["created_at"] = (pd.to_datetime(crt_s, errors="coerce") if crt_s is not None
                         else pd.Series([pd.NaT]*len(df), dtype="datetime64[ns]"))

    # 4) key cleanup: require customer_id; dedupe on it
    out = out.dropna(subset=["customer_id"]).drop_duplicates("customer_id", keep="last")

    return out

def _pick_or_collapse(df: pd.DataFrame, spec: Union[str, Sequence[str], None]) -> Optional[pd.Series]:
    """
    From a mapping spec (str or list of strs), return a SINGLE Series:
      - if str and present -> that column
      - if list/tuple/Index/ndarray:
          - use first present col if only one
          - if multiple present -> first-non-null across them (bfill) → one Series
      - else -> None
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        return df[spec] if spec in df.columns else None

    # list/tuple/Index/ndarray
    if isinstance(spec, (list, tuple, pd.Index, np.ndarray)):
        present = [c for c in map(str, spec) if c in df.columns]
        if not present:
            return None
        if len(present) == 1:
            return df[present[0]]
        # collapse: first non-null left→right
        return df[present].bfill(axis=1).iloc[:, 0]

    return None
def _collapse_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If columns are duplicated, collapse each group into a single Series
    by taking the first non-null value left→right.
    """
    if not df.columns.duplicated().any():
        return df

    out = {}
    # Preserve left-to-right order of first appearances
    seen = set()
    ordered_unique = []
    for c in df.columns:
        if c not in seen:
            seen.add(c)
            ordered_unique.append(c)

    for name in ordered_unique:
        dupes = df.loc[:, df.columns == name]
        if dupes.shape[1] == 1:
            out[name] = dupes.iloc[:, 0]
        else:
            out[name] = dupes.bfill(axis=1).iloc[:, 0]

    return pd.DataFrame(out, index=df.index)

def coerce_orders(df: Optional[pd.DataFrame], m: Dict[str, object]) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["order_id","customer_id","order_date","amount"])

    df = normalize_headers(df)
    out = pd.DataFrame(index=df.index)
    df = _collapse_duplicate_columns(df)  # must happen before selecting columns
    oid_s = _pick_or_collapse(df, m.get("order_id"))
    cid_s = _pick_or_collapse(df, m.get("customer_id"))
    dt_s  = _pick_or_collapse(df, m.get("order_date") or m.get("order_timestamp") or m.get("timestamp"))
    amt_s = _pick_or_collapse(df, m.get("order_total")  or m.get("amount")         or m.get("total"))

    out["order_id"]    = (oid_s.astype("string") if oid_s is not None
                          else pd.Series([pd.NA]*len(df), dtype="string"))
    out["customer_id"] = (cid_s.astype("string") if cid_s is not None
                          else pd.Series([pd.NA]*len(df), dtype="string"))
    out["order_date"]  = (pd.to_datetime(dt_s, errors="coerce") if dt_s is not None
                          else pd.Series([pd.NaT]*len(df), dtype="datetime64[ns]"))
    out["amount"]      = (pd.to_numeric(amt_s, errors="coerce").astype("Float64") if amt_s is not None
                          else pd.Series([pd.NA]*len(df), dtype="Float64"))

    out = out.dropna(subset=["order_id"]).drop_duplicates("order_id", keep="last")
    out = out[out["amount"].fillna(0) > 0]
    return out

def coerce_marketing(df: Optional[pd.DataFrame], m: Dict[str, object]) -> pd.DataFrame:
    """
    Normalize a Marketing table to:
      - campaign_id (string, key)
      - customer_id (string)
      - channel (string, lowercased/trimmed)
      - sent_date (datetime64[ns])
      - opened, clicked, converted (Int64: 0/1/NA)
      - cost (Float64, default 0.0)
    Mapping values may be str or list[str]; we collapse to a single Series.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=[
            "campaign_id","customer_id","channel","sent_date","opened","clicked","converted","cost"
        ])

    # 1) normalize + collapse dupes (prevents df["col"] returning a DataFrame)
    df = normalize_headers(df)
    df = _collapse_duplicate_columns(df)

    out = pd.DataFrame(index=df.index)

    # 2) resolve mappings (support str or list)
    camp_s = _pick_or_collapse(df, m.get("campaign_id"))
    cust_s = _pick_or_collapse(df, m.get("customer_id"))
    chan_s = _pick_or_collapse(df, m.get("channel"))
    sent_s = _pick_or_collapse(df, m.get("sent_date") or m.get("send_date") or m.get("timestamp") or m.get("created_at") or m.get("sent_at") or m.get("delivered_at"))
    open_s = _pick_or_collapse(df, m.get("opened")    or m.get("open")      or m.get("is_open"))
    click_s= _pick_or_collapse(df, m.get("clicked")   or m.get("click")     or m.get("is_click"))
    conv_s = _pick_or_collapse(df, m.get("converted") or m.get("purchase")  or m.get("is_purchase") or m.get("order"))
    cost_s = _pick_or_collapse(df, m.get("cost")      or m.get("campaign_cost") or m.get("spend"))

    # 3) assign with safe dtypes
    out["campaign_id"] = (camp_s.astype("string") if camp_s is not None
                          else pd.Series([pd.NA]*len(df), dtype="string"))
    out["customer_id"] = (cust_s.astype("string") if cust_s is not None
                          else pd.Series([pd.NA]*len(df), dtype="string"))

    if chan_s is not None:
        out["channel"] = chan_s.astype("string").str.strip().str.lower()
    else:
        out["channel"] = pd.Series([pd.NA]*len(df), dtype="string")

    out["sent_date"] = (pd.to_datetime(sent_s, errors="coerce") if sent_s is not None
                        else pd.Series([pd.NaT]*len(df), dtype="datetime64[ns]"))

    # booleans as 0/1 (nullable Int64)
    out["opened"]    = (to_bool01(open_s)  if open_s  is not None else pd.Series([pd.NA]*len(df), dtype="Int64"))
    out["clicked"]   = (to_bool01(click_s) if click_s is not None else pd.Series([pd.NA]*len(df), dtype="Int64"))
    out["converted"] = (to_bool01(conv_s)  if conv_s  is not None else pd.Series([pd.NA]*len(df), dtype="Int64"))

    # cost as Float64; keep 0.0 default when column missing
    if cost_s is not None:
        # to_num_currency already strips $ and commas then pd.to_numeric
        out["cost"] = to_num_currency(cost_s).astype("Float64").fillna(0)
    else:
        out["cost"] = pd.Series([0.0]*len(df), dtype="Float64")

    # 4) key cleanup: require campaign_id; dedupe on it
    out = out.dropna(subset=["campaign_id"]).drop_duplicates("campaign_id", keep="last")

    return out


# --------- helpers ---------
async def _parse_json_field(form_value: Any, field_name: str) -> Dict:
    """
    Accepts either a plain text field or an UploadFile (Blob) named `field_name`
    from multipart/form-data and returns parsed JSON dict.
    """
    if form_value is None:
        return {}
    try:
        if isinstance(form_value, UploadFile):
            raw = await form_value.read()
            return json.loads(raw.decode("utf-8") or "{}")
        # else it's a simple string
        return json.loads(str(form_value) or "{}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in '{field_name}': {e}")
CSV_PREVIEW_ROWS = 10

def _file_health(df: Optional[pd.DataFrame], kind: str) -> Dict[str, Any]:
    spec = SCHEMA_CONTRACT[kind]
    present = set(df.columns) if df is not None else set()

    # Missing columns
    missing_required    = [c for c in spec["required"]    if c not in present]
    missing_recommended = [c for c in spec["recommended"] if c not in present]

    # Row count
    row_count = int(len(df)) if df is not None else 0

    # Null rates for req+rec (only if present)
    null_rates: Dict[str, float] = {}
    if df is not None:
        for c in spec["required"] + spec["recommended"]:
            if c in df.columns:
                null_rates[c] = float(df[c].isna().mean())

    # Duplicate keys
    duplicate_keys = 0
    key_field = spec.get("key_field")
    if df is not None and key_field and key_field in df.columns:
        duplicate_keys = int(df[key_field].duplicated().sum())

    # Date ranges
    min_max_dates: Dict[str, Dict[str, Optional[str]]] = {}
    if df is not None:
        for c in spec["date_fields"]:
            if c in df.columns:
                s = pd.to_datetime(df[c], errors="coerce", utc=True)
                if s.notna().any():
                    min_dt = s.min(); max_dt = s.max()
                    min_max_dates[c] = {
                        "min": min_dt.isoformat(),
                        "max": max_dt.isoformat(),
                    }

    # Preview (header + up to 10 rows)
    preview: List[List[Any]] = []
    if df is not None and row_count:
        head = df.head(CSV_PREVIEW_ROWS).astype(object).where(pd.notna(df.head(CSV_PREVIEW_ROWS)), "")
        preview = [list(df.columns)] + head.values.tolist()

    # Status + issues
    issues: List[Dict[str, Any]] = []
    if missing_required:
        issues.append({"severity": "warn", "message": f"Missing required columns: {missing_required}"})
    if missing_recommended:
        issues.append({"severity": "info", "message": f"Missing recommended columns: {missing_recommended}"})
    if duplicate_keys:
        issues.append({"severity": "warn", "message": f"Duplicate {key_field}: {duplicate_keys}"})
    for col, rate in null_rates.items():
        if rate >= 0.2:
            issues.append({"severity": "info", "message": f"{col}: {int(rate*100)}% null"})

    status = "ready" if not missing_required else "degraded"

    return {
        "status": status,
        "row_count": row_count,
        "missing_required": missing_required,
        "missing_recommended": missing_recommended,
        "null_rates": null_rates,           # {col: 0..1}
        "duplicate_keys": duplicate_keys,
        "min_max_dates": min_max_dates,     # {date_col: {min, max}}
        "preview": preview,                 # [headers, ...rows]
        "issues": issues,                   # [{severity, message}]
        "levels": {                         # handy for UI badges
            "required":    spec["required"],
            "recommended": spec["recommended"],
            "optional":    spec["optional"],
        },
    }

# If you have these already, keep yours and remove mine
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "uploads")
# ----------------- small helpers -----------------

# --- small utils reused below -------------------------------------------------
def _now_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def _slug_filename(name: str) -> str:
    base = Path(name).stem
    ext  = Path(name).suffix or ".csv"
    base = re.sub(r"[^a-zA-Z0-9._-]+", "-", base).strip("-._")
    return f"{base}{ext}"

def _now_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

async def _buffer_upload(file: Optional[UploadFile]) -> Tuple[Optional[bytes], Optional[str]]:
    if not file:
        return None, None
    data = await file.read()
    return (data if data else None), (file.filename or "upload.csv")

def _read_csv_from_bytes(data: Optional[bytes]) -> Optional[pd.DataFrame]:
    if not data:
        return None
    try:
        return pd.read_csv(io.BytesIO(data))
    except Exception:
        return None

def _resolve_user_id(request: Request, current_user) -> str:
    return (
        getattr(current_user, "id", None)
        or getattr(current_user, "user_id", None)
        or getattr(current_user, "sub", None)
        or getattr(current_user, "email", None)
        or getattr(getattr(request.state, "user", None), "id", None)
        or request.headers.get("X-User-Id")
        or "anon"
    )

def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default

def _as_int(x: Any, default: int = 0) -> int:
    try:
        v = int(float(x))
        return v
    except Exception:
        return default

def _normalize_actions(actions_in: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Accepts a list of action dicts, returns normalized list + warnings.
    Expected keys (case-insensitive/tolerant): name, channel, unit_cost|cost, cooldown_days|cooldown, daily_cap|dailyCap, provider
    """
    warnings: List[str] = []
    out: List[Dict[str, Any]] = []
    if not isinstance(actions_in, list):
        return [], warnings

    for i, raw in enumerate(actions_in):
        if not isinstance(raw, dict):
            warnings.append(f"Action[{i}] ignored (not an object).")
            continue

        name = str(raw.get("name", "")).strip()
        channel = str(raw.get("channel", "custom")).strip() or "custom"
        unit_cost = _as_float(raw.get("unit_cost", raw.get("cost", 0.0)), 0.0)
        cooldown_days = max(0, _as_int(raw.get("cooldown_days", raw.get("cooldown", 0)), 0))
        daily_cap = max(0, _as_int(raw.get("daily_cap", raw.get("dailyCap", 0)), 0))
        provider = str(raw.get("provider", "")).strip() or None

        if not name:
            warnings.append(f"Action[{i}] missing name; skipped.")
            continue
        if unit_cost < 0:
            warnings.append(f"Action[{i}] '{name}' unit_cost < 0; clamped to 0.")
            unit_cost = 0.0

        out.append({
            "name": name,
            "channel": channel,
            "unit_cost": unit_cost,
            "cooldown_days": cooldown_days,
            "daily_cap": daily_cap,
            "provider": provider
        })

    return out, warnings

SIGNALS_KEY_NAME = "signals.json"

def _signals_key(user_id: str|int, dataset_id: int) -> str:
    return f"{user_id}/{dataset_id}/plan/signals.json"

def save_signals(*, user_id: str|int, dataset_id: int, signals: dict) -> str:
    bucket = os.environ.get("SUPABASE_BUCKET","user-uploads")
    key = _signals_key(user_id, dataset_id)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp_path = tmp.name
    try:
        with open(tmp_path,"w",encoding="utf-8") as f:
            json.dump(signals, f, ensure_ascii=False)
        # reuse your existing uploader; or sb_upload_from_path if you have it
        return upload_file_to_supabase(str(user_id), tmp_path, key)
    finally:
        try: os.remove(tmp_path)
        except Exception: pass
def load_signals(*, user_id: str|int, dataset_id: int) -> dict | None:
    key = _signals_key(user_id, dataset_id)
    try:
        # reuse your existing downloader
        import tempfile, json
        with tempfile.NamedTemporaryFile(delete=True, suffix=".json") as tmp:
            sb_download_to_path(os.environ.get("SUPABASE_BUCKET","user-uploads"), key, tmp.name)
            with open(tmp.name,"r",encoding="utf-8") as f:
                data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None

def _present_from_bytes(cust: Optional[bytes], ords: Optional[bytes], mkt: Optional[bytes]) -> List[str]:
    kinds = []
    if cust: kinds.append("customers")
    if ords: kinds.append("orders")
    if mkt:  kinds.append("marketing")
    return kinds
import base64, hashlib

INLINE_B64_MAX = 1_500_000  # ~1.5 MB; skip inline for larger files
CSV_PREVIEW_ROWS = 10

def _df_to_csv_bytes(df: Optional[pd.DataFrame]) -> Optional[bytes]:
    if df is None:
        return None
    try:
        return df.to_csv(index=False).encode("utf-8")
    except Exception:
        return None

def _csv_meta(bytes_or_none: Optional[bytes]) -> Dict[str, any]:
    if not bytes_or_none:
        return {"size_bytes": 0, "sha256": ""}
    size = len(bytes_or_none)
    sha = hashlib.sha256(bytes_or_none).hexdigest()
    return {"size_bytes": size, "sha256": sha}

async def _upload_csv_bytes(user_id: str, kind: str, df: Optional[pd.DataFrame]) -> Dict[str, any]:
    """
    Create CSV bytes from df, upload via upload_file_to_supabase(), return metadata.
    """
    out: Dict[str, any] = {
        "kind": kind,
        "filename": "",
        "path": "",
        "public_url": "",      # left blank; your uploader returns only the path
        "size_bytes": 0,
        "columns": [],
        "inline_b64": None,
        "preview": [],
    }
    if df is None:
        return out

    csv_bytes = _df_to_csv_bytes(df)
    meta = _csv_meta(csv_bytes)
    out["size_bytes"] = meta["size_bytes"]
    out["columns"] = list(df.columns)

    # Build a consistent stamped filename (under your per-user prefix in the uploader)
    safe_original = _slug_filename(f"{kind}.csv")
    stamped_name = f"derive/{_now_tag()}-{kind}-{safe_original}"

    # Write bytes to a temp file and hand off to the uploader
    suffix = Path(stamped_name).suffix or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        if csv_bytes:
            tmp.write(csv_bytes)
        tmp_path = tmp.name

    try:
        storage_path = upload_file_to_supabase(user_id, tmp_path, filename=stamped_name)
        out["filename"] = stamped_name
        out["path"] = storage_path
        # NOTE: out["public_url"] intentionally left "" (uploader returns only the path)

        # Inline b64 only for small files
        if csv_bytes and len(csv_bytes) <= INLINE_B64_MAX:
            out["inline_b64"] = base64.b64encode(csv_bytes).decode("ascii")

        # Preview rows
        try:
            preview_df = df.head(CSV_PREVIEW_ROWS)
            out["preview"] = [list(preview_df.columns)] + preview_df.astype(object).where(
                pd.notna(preview_df), ""
            ).values.tolist()
        except Exception:
            out["preview"] = []

        return out
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# ---- Preflight (non-blocking) + safe compute utilities ----
from typing import Set, List, Dict, Any, Optional, Tuple

def _cols(df, cols: List[str]) -> Set[str]:
    if df is None: return set(cols)
    return {c for c in cols if c not in df.columns}

def _gap(gaps: List[Dict[str, Any]], calc_key: str, table: str, missing: List[str], fallback: Any, note: str):
    gaps.append({
        "calc": calc_key,                 # e.g., "unit_economics.aov"
        "table": table,                   # "orders"
        "missing": missing,               # ["amount"]
        "fallback": fallback,             # 0.0 (what we used)
        "message": note,                  # user-friendly explanation
        "severity": "warn"                # could be "info" | "warn"
    })
def apply_common_transforms(df: pd.DataFrame, key_field: Optional[str]=None, user_tz: Optional[str]=None) -> Tuple[pd.DataFrame, List[str]]:
    transforms = []
    if df is None or df.empty:
        return df, transforms

    # Trim whitespace and collapse multiple spaces for all string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        transforms.append(f"trim_whitespace:{col}")
        transforms.append(f"collapse_spaces:{col}")

    # Lowercase emails
    if "email" in df.columns:
        df["email"] = df["email"].astype(str).str.lower()
        transforms.append("lowercase_email")

    # E.164 phone normalization (already done in clean_phone)
    if "phone" in df.columns:
        df["phone"] = clean_phone(df["phone"])
        transforms.append("normalize_phone")

    # Parse dates with timezone and unify to ISO-8601
    date_cols = [c for c in df.columns if "date" in c or "created_at" in c]
    for col in date_cols:
        try:
            dt = pd.to_datetime(df[col], errors="coerce")
            if user_tz:
                dt = dt.dt.tz_convert(user_tz)  # user tz view, NO 'Z' suffix
                df[col] = dt.dt.strftime("%Y-%m-%dT%H:%M:%S%z")  # e.g. 2025-03-01T10:00:00-0800
            else:
                # canonical UTC with Z
                df[col] = dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            transforms.append(f"parse_date:{col}")
        except Exception:
            pass

    # Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        transforms.append("drop_exact_duplicates")

    # Dedupe on key field
    if key_field and key_field in df.columns:
        before = len(df)
        df = df.drop_duplicates(key_field, keep="last")
        if len(df) < before:
            transforms.append(f"dedupe_on_key:{key_field}")

    return df, transforms
def _preflight_and_compute(C, O, M, options) -> Tuple[Dict[str, Any], List[Dict[str, Any]], float]:
    """
    Returns (derived_partial, calc_gaps, coverage_pct). We only fill the parts this block owns.
    Your endpoint can then merge with actions, experiment, etc.
    """
    gaps: List[Dict[str, Any]] = []
    total_calcs = 0
    ok_calcs = 0

    # --- Eligible population ---
    total_calcs += 1
    if C is not None:
        eligible_total = int(len(C))
        ok_calcs += 1
    else:
        eligible_total = 0
        _gap(gaps, "eligible_population.total", "customers", ["*"], 0, "No customers table; assuming 0 eligible.")

    # emailable %
    total_calcs += 1
    miss = _cols(C, ["email"])
    if not miss and eligible_total:
        emailable_pct = float(C["email"].notna().mean())
        ok_calcs += 1
    else:
        emailable_pct = 0.0
        if "email" in miss:
            _gap(gaps, "eligible_population.emailable_pct", "customers", ["email"], 0.0,
                 "Missing customer email; email reach assumed 0%.")
        elif not eligible_total:
            _gap(gaps, "eligible_population.emailable_pct", "customers", ["customer_id"], 0.0,
                 "No eligible customers; email reach 0%.")

    # sms %
    total_calcs += 1
    miss = _cols(C, ["phone"])
    if not miss and eligible_total:
        sms_opt_in_pct = float(C["phone"].notna().mean())
        ok_calcs += 1
    else:
        sms_opt_in_pct = 0.0
        if "phone" in miss:
            _gap(gaps, "eligible_population.sms_opt_in_pct", "customers", ["phone"], 0.0,
                 "Missing customer phone; SMS reach assumed 0%.")
        elif not eligible_total:
            _gap(gaps, "eligible_population.sms_opt_in_pct", "customers", ["customer_id"], 0.0,
                 "No eligible customers; SMS reach 0%.")

    # --- Unit economics ---
    margin_pct = float(options.get("margin_pct", 0.40))
    return_pct = float(options.get("return_pct", 0.08))

    # AOV
    total_calcs += 1
    miss = _cols(O, ["amount"])  # you normalized orders total as 'amount'
    if not miss and O is not None and len(O):
        aov = float(O["amount"].mean())
        ok_calcs += 1
    else:
        aov = 0.0
        _gap(gaps, "unit_economics.aov", "orders", ["amount"], 0.0,
             "Missing orders 'amount' (or empty orders); AOV assumed 0.")

    # Profit per touchpoint
    total_calcs += 1
    if aov > 0:
        profit_per_tp = aov * margin_pct * (1 - return_pct)
        ok_calcs += 1
    else:
        profit_per_tp = 0.0
        _gap(gaps, "unit_economics.profit_per_tp", "orders", ["amount"], 0.0,
             "Cannot compute profit_per_tp without AOV; set to 0.")

    # --- Volume & expected value ---
    contactable_pct = max(emailable_pct, sms_opt_in_pct)

    # Volume
    total_calcs += 1
    if eligible_total > 0 and contactable_pct > 0:
        volume = int(eligible_total * contactable_pct * 0.5)
        ok_calcs += 1
    else:
        volume = 0
        miss_cols = []
        if eligible_total == 0: miss_cols.append("customer_id")
        if contactable_pct == 0: miss_cols.extend(["email|phone"])
        _gap(gaps, "threshold_volume.volume", "customers", miss_cols or ["email|phone"], 0,
             "No contactable customers; volume assumed 0.")

    # Expected net
    total_calcs += 1
    if profit_per_tp > 0 and volume > 0:
        expected_net = round(volume * profit_per_tp * 0.1, 2)
        ok_calcs += 1
    else:
        expected_net = 0.0
        missing = []
        if profit_per_tp == 0: missing.append("aov/margin/return")
        if volume == 0: missing.append("contactable_customers")
        _gap(gaps, "threshold_volume.expected_net", "orders|customers", missing or ["*"], 0.0,
             "Expected net requires positive volume and profit_per_tp; set to 0.")

    coverage_pct = ok_calcs / max(total_calcs, 1)
    partial = {
        "eligible_population": {
            "total": eligible_total,
            "emailable_pct": emailable_pct,
            "sms_opt_in_pct": sms_opt_in_pct,
        },
        "unit_economics": {
            "aov": aov,
            "margin_pct": margin_pct,
            "return_pct": return_pct,
            "profit_per_tp": profit_per_tp,
        },
        "threshold_volume": {
            "volume": volume,
            "expected_net": expected_net,
        },
        "meta": {"calc_coverage_pct": coverage_pct},
    }
    return partial, gaps, coverage_pct
# add to imports
import io, gzip, re
from typing import Optional, Tuple, List
import pandas as pd

# --- format sniffers ---
def _sniff_format(name: Optional[str], blob: Optional[bytes]) -> str:
    """
    Returns one of: 'csv', 'csv.gz', 'xlsx', 'parquet'.
    Prefers magic bytes; falls back to filename extension.
    """
    if not blob:
        # fall back to extension
        ext = (name or "").lower()
        if ext.endswith(".csv.gz"): return "csv.gz"
        if ext.endswith(".xlsx"):   return "xlsx"
        if ext.endswith(".parquet"):return "parquet"
        return "csv"

    head = blob[:8]
    # parquet magic: PAR1
    if head.startswith(b"PAR1"):
        return "parquet"
    # gzip magic: 1f 8b
    if head.startswith(b"\x1f\x8b"):
        return "csv.gz"
    # xlsx is a zip; primary signature PK\x03\x04, rely on extension to distinguish
    if head.startswith(b"PK\x03\x04") and (name or "").lower().endswith(".xlsx"):
        return "xlsx"
    # default
    ext = (name or "").lower()
    if ext.endswith(".csv.gz"): return "csv.gz"
    if ext.endswith(".xlsx"):   return "xlsx"
    if ext.endswith(".parquet"):return "parquet"
    return "csv"

def _pick_excel_sheet(xls: pd.ExcelFile, expected_kind: Optional[str]) -> str:
    """
    Choose the best sheet:
      1) exact match to expected_kind
      2) contains expected_kind as substring
      3) first sheet
    """
    sheets = [s for s in xls.sheet_names if isinstance(s, str)]
    if not sheets:
        return 0  # pandas also accepts index
    if expected_kind:
        ek = expected_kind.lower()
        for s in sheets:
            if s.strip().lower() == ek:
                return s
        for s in sheets:
            if ek in s.strip().lower():
                return s
    return sheets[0]
def read_table_bytes(data: bytes | None, filename: str | None) -> Tuple[Optional[pd.DataFrame], List[str]]:
    warns: List[str] = []
    if not data:
        return None, warns

    head = data[:8]

    # gzip?
    if head.startswith(b"\x1f\x8b"):
        try:
            unz = gzip.decompress(data)
            txt = unz.decode("utf-8-sig", errors="replace")
            return pd.read_csv(io.StringIO(txt), sep=None, engine="python",
                               on_bad_lines="skip", low_memory=False), warns
        except Exception as e:
            warns.append(f"gz csv parse error: {e}")

    # CSV with delimiter sniff
    try:
        txt = data.decode("utf-8-sig", errors="replace")
        sample = "\n".join(txt.splitlines()[:50])
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            sep = dialect.delimiter
        except Exception:
            sep = None
        return pd.read_csv(io.StringIO(txt), sep=sep, engine="python",
                           on_bad_lines="skip", low_memory=False), warns
    except Exception as e:
        warns.append(f"text csv parse error: {e}")

    # xlsx?
    try:
        if _looks_xlsx(filename, head):
            return pd.read_excel(io.BytesIO(data)), warns
    except Exception as e:
        warns.append(f"xlsx parse error: {e}")

    # parquet?
    try:
        return pd.read_parquet(io.BytesIO(data)), warns
    except Exception as e:
        warns.append(f"parquet parse error: {e}")

    # final csv fallback
    try:
        return pd.read_csv(io.BytesIO(data), sep=None, engine="python",
                           on_bad_lines="skip", low_memory=False), warns
    except Exception as e:
        warns.append(f"final csv parse error: {e}")
        return None, warns
# --- fallbacks for getting the file ------------------------------------------
def _download_by_processed_key(processed_key: str) -> Optional[bytes]:
    bucket = os.environ.get("SUPABASE_BUCKET", "user-uploads")
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(processed_key).suffix or ".csv") as tmp:
        tmp_path = tmp.name
    try:
        sb_download_to_path(bucket, processed_key, tmp_path)  # must exist in your storage module
        with open(tmp_path, "rb") as f:
            return f.read()
    except Exception:
        return None
    finally:
        try: os.remove(tmp_path)
        except Exception: pass

def _download_by_url(data_url: str) -> Optional[bytes]:
    try:
        import httpx
        with httpx.Client(follow_redirects=True, timeout=30.0) as client:
            r = client.get(data_url)
            if not r.is_success: 
                return None
            ctype = (r.headers.get("content-type") or "").lower()
            if "html" in ctype:   # expired signed URL returns HTML
                return None
            return r.content if r.content else None
    except Exception:
        return None
# --- very small generic mapper so no missing deps ----------------------------
def _auto_pick(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    low  = [c.lower() for c in cols]
    for cand in candidates:
        c = cand.lower()
        # exact
        if c in low:
            return cols[low.index(c)]
        # contains
        for i, name in enumerate(low):
            if c in name or name in c:
                return cols[i]
    return None
from planner import map_target_to_signals
# ----------------- endpoint (single-file, works for ANY file) -----------------
from fastapi import APIRouter, Depends, File, UploadFile, Request, HTTPException
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import tempfile, uuid, os
import pandas as pd

# ---- NEW: broad synonyms for a generic events-like dataset
GENERIC_SYNS: Dict[str, List[str]] = {
    # canonical -> synonyms
    "id":        ["id","record_id","uuid","row_id","user_id","customer_id","order_id","lead_id","account_id"],
    "timestamp": ["timestamp","ts","time","event_time","date","datetime","created_at","updated_at","order_date","sent_at"],
    "amount":    ["amount","revenue","price","value","total","charge","cost","net","gross","amt","order_amount"],
    "email":     ["email","email_address","e-mail"],
    "phone":     ["phone","phone_number","mobile","cell","tel"],
    "channel":   ["channel","utm_source","source","utm_medium","medium","campaign","utm_campaign"],
    "target":    ["target","label","converted","purchase","purchased","response","responded","clicked","opened","churned","y","outcome"],
}

def _pick_first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None

def _canonicalize(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    # rename only mapped columns to their canonical names; keep others
    ren = {v: k for k, v in mapping.items() if v and v in df.columns and k not in df.columns}
    return df.rename(columns=ren)

def _best_match(df: pd.DataFrame, syns: Dict[str, List[str]]) -> Dict[str, Optional[str]]:
    found: Dict[str, Optional[str]] = {}
    cols_lower = {c.lower(): c for c in df.columns}
    for canon, cand in syns.items():
        # try exact (case-insensitive) first
        for s in cand:
            if s.lower() in cols_lower:
                found[canon] = cols_lower[s.lower()]
                break
        else:
            # try relaxed: underscores/spaces stripped
            norm_cols = {c.lower().replace("_","").replace(" ",""): c for c in df.columns}
            for s in cand:
                key = s.lower().replace("_","").replace(" ","")
                if key in norm_cols:
                    found[canon] = norm_cols[key]
                    break
            else:
                found[canon] = None
    return found

def coerce_generic(df: pd.DataFrame, ui_map: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Normalize to canonical columns:
      id, timestamp, amount, email, phone, channel, target
    Returns (df, warnings)
    """
    warnings: List[str] = []
    df = df.copy()

    # 1) Build column mapping: UI map wins, then auto from GENERIC_SYNS
    auto_map = _best_match(df, GENERIC_SYNS)
    ui_map = ui_map or {}
    # ui_map could be: {"id":"user_uuid", "timestamp":"created", "amount":"total_price", ...}
    merged = {k: (ui_map.get(k) or auto_map.get(k)) for k in GENERIC_SYNS.keys()}

    # 2) Rename to canonical
    df = _canonicalize(df, merged)

    # 3) Ensure an id
    if "id" not in df.columns or df["id"].isnull().all():
        df["id"] = df.index.astype(str)
        warnings.append("No ID column detected; synthesized 'id' from row index.")

    # 4) Parse timestamp if present; else synthesize monotonic timestamps (optional)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
        if df["timestamp"].isnull().all():
            warnings.append("Could not parse 'timestamp'; values were all NaT.")
    else:
        # optional synth: comment out if you prefer to leave missing
        # df["timestamp"] = pd.to_datetime(pd.Series(range(len(df))), unit="s")
        warnings.append("No timestamp column detected.")

    # 5) Parse amount to numeric
    if "amount" in df.columns:
        # strip currency symbols and commas
        df["amount"] = (
            pd.to_numeric(
                df["amount"]
                .astype(str)
                .str.replace(r"[^0-9\.\-]", "", regex=True),
                errors="coerce"
            )
        )
        if df["amount"].isnull().all():
            warnings.append("Could not parse 'amount' to numeric.")
    # If missing, it's fine—downstream can use counts-only unit economics.

    # 6) Normalize target (binary-ish strings to 0/1 where obvious)
    if "target" in df.columns:
        # common truthy/falsy mapping
        truthy = {"1","true","t","yes","y","success","converted","purchase","purchased","responded","clicked","open","opened"}
        falsy  = {"0","false","f","no","n","fail","failed","did_not_convert","did not convert","not_clicked","not opened","churned=0"}
        def _to01(x):
            if pd.isna(x): return pd.NA
            s = str(x).strip().lower()
            if s in truthy: return 1
            if s in falsy:  return 0
            return x  # leave as-is; downstream will infer numeric/multiclass if not binary
        df["target"] = df["target"].map(_to01)

    return df, warnings
def _coerce_generic_as_orders(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Minimal generic normalization into orders-like table:
      order_id, customer_id, order_date, amount
    """
    warns: List[str] = []
    out = pd.DataFrame(index=df.index)

    oid = _auto_pick(df, ["order_id","oid","transaction_id","id"])
    cid = _auto_pick(df, ["customer_id","cust_id","user_id","uid","email"])
    dtc = _auto_pick(df, ["order_date","timestamp","date","created_at","processed_at"])
    amt = _auto_pick(df, ["amount","order_total","revenue","price","total","subtotal_price","total_price"])

    if not oid:
        # fabricate an index-based id if missing
        out["order_id"] = (df.reset_index().index.astype(str)).astype("string")
        warns.append("order_id missing; synthesized sequential ids.")
    else:
        out["order_id"] = df[oid].astype("string")

    out["customer_id"] = (df[cid].astype("string") if cid else pd.Series([pd.NA]*len(df), dtype="string"))
    out["order_date"]  = (pd.to_datetime(df[dtc], errors="coerce") if dtc else pd.Series([pd.NaT]*len(df), dtype="datetime64[ns]"))

    if amt:
        out["amount"] = (
            pd.to_numeric(df[amt].astype(str).str.replace(r"[\$,]", "", regex=True), errors="coerce")
            .astype("Float64")
        )
    else:
        out["amount"] = pd.Series([pd.NA]*len(df), dtype="Float64")
        warns.append("amount missing; set to NA.")

    # keep only positive/valid amounts if any exist
    if out["amount"].notna().any():
        out = out[out["amount"].fillna(0) >= 0]

    # dedupe on order_id
    out = out.dropna(subset=["order_id"]).drop_duplicates("order_id", keep="last")
    return out, warns

# --- references to your existing helpers (assumed present in your file) ------
# _file_health, apply_common_transforms, _preflight_and_compute,
# _normalize_actions, TargetSpecModel, map_target_to_signals,
# _upload_csv_bytes, PREVIEW_STORE, SCHEMA_CONTRACT
import io, gzip, csv
import pandas as pd

def _looks_xlsx(name: str | None, head: bytes) -> bool:
    # xlsx zip signature is PK\x03\x04; rely on extension to reduce false positives
    return head.startswith(b"PK\x03\x04") and (name or "").lower().endswith(".xlsx")

def _read_table_bytes(data: bytes | None, filename: str | None) -> tuple[Optional[pd.DataFrame], list[str]]:
    """
    Returns (df, warnings). Tries in this order:
    1) gzip → CSV
    2) CSV (python engine, sep sniff, bad-line skip, UTF-8-BOM)
    3) Excel (xlsx) when the signature & extension match
    4) Parquet
    5) Final CSV fallbacks
    """
    warns: list[str] = []
    if not data:
        return None, warns

    head = data[:8]

    # 1) If gzip magic, decompress and treat as CSV
    if head.startswith(b"\x1f\x8b"):
        try:
            unz = gzip.decompress(data)
            # utf-8-sig handles BOM; engine='python' allows sep=None sniff
            return (
                pd.read_csv(
                    io.StringIO(unz.decode("utf-8-sig", errors="replace")),
                    sep=None, engine="python", on_bad_lines="skip", low_memory=False
                ),
                warns,
            )
        except Exception as e:
            warns.append(f"gz csv parse error: {e}")

    # 2) CSV with delimiter sniff (handles ',', '\t', ';', '|')
    try:
        text = data.decode("utf-8-sig", errors="replace")
        sample = "\n".join(text.splitlines()[:50])  # sniffer needs a sample
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
            sep = dialect.delimiter
        except Exception:
            sep = None  # let pandas sniff with engine='python'
        return (
            pd.read_csv(
                io.StringIO(text),
                sep=sep, engine="python", on_bad_lines="skip", low_memory=False
            ),
            warns,
        )
    except Exception as e:
        warns.append(f"text csv parse error: {e}")

    # 3) Excel (xlsx) if it looks like one
    try:
        if _looks_xlsx(filename, head):
            return (pd.read_excel(io.BytesIO(data)), warns)
    except Exception as e:
        warns.append(f"xlsx parse error: {e}")

    # 4) Parquet
    try:
        return (pd.read_parquet(io.BytesIO(data)), warns)
    except Exception as e:
        warns.append(f"parquet parse error: {e}")

    # 5) Final CSV fallbacks (binary read)
    try:
        return (
            pd.read_csv(io.BytesIO(data), sep=None, engine="python", on_bad_lines="skip", low_memory=False),
            warns,
        )
    except Exception as e:
        warns.append(f"final csv parse error: {e}")
        return None, warns
# --- bytes loader for derive --------------------------------------------------
from fastapi import UploadFile
import tempfile, os, io
import httpx

async def _load_df_from_any(
    *,
    data_file: UploadFile | None,
    processed_key: str | None,
    data_url: str | None,
    expected_kind: str | None,
) -> tuple[Optional[pd.DataFrame], list[str], str]:
    """
    Returns (df, warnings, source_used)
    Tries: data_file -> processed_key (Supabase) -> data_url (server-side HTTP)
    """
    warns: list[str] = []

    # 1) Direct bytes from client
    if data_file is not None:
        try:
            raw = await data_file.read()
            if raw and len(raw) > 0:
                df, w = _read_table_bytes(raw, getattr(data_file, "filename", None), expected_kind=expected_kind)
                return df, warns + w, "data_file"
            warns.append("data_file present but empty.")
        except Exception as e:
            warns.append(f"data_file read failed: {e}")

    # 2) Supabase key
    if processed_key:
        try:
            bucket = os.environ.get("SUPABASE_BUCKET", "user-uploads")
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp_path = tmp.name
            try:
                sb_download_to_path(bucket, _strip_bucket_prefix(processed_key), tmp_path)
                with open(tmp_path, "rb") as fh:
                    raw = fh.read()
                df, w = _read_table_bytes(raw, os.path.basename(processed_key), expected_kind=expected_kind)
                return df, warns + w, "processed_key"
            finally:
                try: os.remove(tmp_path)
                except Exception: pass
        except Exception as e:
            warns.append(f"download via processed_key failed: {e}")

    # 3) Signed URL (server side; avoids CORS)
    if data_url:
        try:
            # short, safe timeout
            with httpx.Client(timeout=15.0, follow_redirects=True) as client:
                r = client.get(data_url, headers={"Accept": "*/*"})
                r.raise_for_status()
                raw = r.content
            # name hint from URL path
            name_hint = data_url.split("?")[0].rsplit("/", 1)[-1]
            df, w = _read_table_bytes(raw, name_hint, expected_kind=expected_kind)
            return df, warns + w, "data_url"
        except Exception as e:
            warns.append(f"server fetch via data_url failed: {e}")

    return None, warns, "none"
from fastapi import APIRouter, Depends, File, UploadFile, Request, HTTPException, Form
@router.post("/derive")
async def derive_columns(
    request: Request,
    data_file: UploadFile | None = File(None),
    processed_key: str | None = Form(None),   # 👈 accept durable key
    data_url: str | None = Form(None),        # 👈 accept signed URL
    current_user: "User" = Depends(get_current_active_user),
):
    form = await request.form()

    # JSON fields as before
    mappings = await _parse_json_field(form.get("mappings"), "mappings")
    options  = await _parse_json_field(form.get("options"),  "options")
    actions  = await _parse_json_field(form.get("actions"),  "actions")
    target   = await _parse_json_field(form.get("target"),   "target")
    options = options or {}
    dataset_id_raw = form.get("dataset_id")
    dataset_id: int | None = int(dataset_id_raw) if dataset_id_raw else None

    # pick expected kind from hint if you support it
    kind_hint = (form.get("kind") or options.get("kind") or "").strip().lower() or None
    VALID_KINDS = {"customers", "orders", "marketing"}
    expected_kind = kind_hint if (kind_hint in VALID_KINDS) else None

    # Normalize actions etc. (unchanged) ...
    # user_id = ...
    # uploads dict ...

    # 📥 Read the table from ANY of the provided sources
    df, read_warnings, source_used = await _load_df_from_any(
        data_file=data_file,
        processed_key=processed_key,
        data_url=data_url,
        expected_kind=expected_kind,
    )

    warnings = [] + read_warnings
    if df is None or (hasattr(df, "__len__") and len(df) == 0):
        # You can 200/“skipped” for smooth UX, or raise 422. Your FE already handles `skipped`.
        return {"skipped": True, "reason": "Uploaded file could not be parsed or is empty.", "source": source_used}

    # continue with your existing flow:
    df = normalize_headers(df)

    # ---------- light kind inference; treat unknown as generic orders ----------
    cols = set(df.columns)
    if {"campaign_id"} & cols or {"channel","sent_date"} & cols:
        kind_specific = "marketing"
    elif {"order_id"} & cols or {"order_date","amount"} & cols:
        kind_specific = "orders"
    elif {"customer_id"} & cols or {"email"} & cols:
        kind_specific = "customers"
    else:
        kind_specific = "orders"  # generic slot

    # ---------- coerce to canonical shape ----------
    if kind_specific == "customers":
        used_mapping = {"customers": finalize_mapping(df, mappings.get("customers") if isinstance(mappings.get("customers"), dict) else None, CUSTOMER_SYNS),
                        "orders": None, "marketing": None}
        C = coerce_customers(df, used_mapping["customers"])
        O = None; M = None; key_field = "customer_id"
    elif kind_specific == "marketing":
        used_mapping = {"marketing": finalize_mapping(df, mappings.get("marketing") if isinstance(mappings.get("marketing"), dict) else None, MKT_SYNS),
                        "customers": None, "orders": None}
        M = coerce_marketing(df, used_mapping["marketing"])
        C = None; O = None; key_field = "campaign_id"
    else:
        # orders or generic
        if isinstance(mappings, dict) and "orders" in mappings and isinstance(mappings["orders"], dict):
            used_mapping = {"orders": finalize_mapping(df, mappings["orders"], ORDER_SYNS),
                            "customers": None, "marketing": None}
            O = coerce_orders(df, used_mapping["orders"])
        else:
            used_mapping = {"orders": None, "customers": None, "marketing": None}
            O, gen_w = _coerce_generic_as_orders(df)
            warnings.extend(gen_w)
        C = None; M = None; key_field = "order_id"

    # ---------- file health / transforms ----------
    file_health = {"customers": None, "orders": None, "marketing": None}
    D = C or O or M
    file_health[kind_specific] = _file_health(D if D is not None and len(D) else None, kind_specific)
    D, _ = apply_common_transforms(D, key_field=key_field, user_tz=options.get("user_tz"))

    # ---------- uploads of normalized csv ----------
    uploads = {"customers": None, "orders": None, "marketing": None}
    uploads[kind_specific] = await _upload_csv_bytes(user_id, kind_specific, D if (D is not None and len(D)) else None)

    # ---------- preflight / compute ----------
    partial, calc_gaps, coverage_pct = _preflight_and_compute(C, O, M, options)

    # ---------- preview cache ----------
    try:
        if dataset_id is not None and D is not None and len(D):
            PREVIEW_STORE[dataset_id] = D.head(50).to_dict("records")
    except Exception:
        pass

    # ---------- planner signals (and persist) ----------
    planner_signals = None
    try:
        preview_rows = D.head(50).to_dict("records") if (D is not None and len(D)) else None
        if target:
            t_model = TargetSpecModel(**target)
            sig_obj = map_target_to_signals(t_model, preview_rows)
            planner_signals = sig_obj.model_dump() if hasattr(sig_obj, "model_dump") else dict(sig_obj or {})
            if planner_signals and dataset_id is not None:
                SIGNALS_STORE[dataset_id] = planner_signals
                try:
                    # optional persist for compile() to load
                    save_signals(user_id=user_id, dataset_id=dataset_id, signals=planner_signals)
                except Exception:
                    pass
    except Exception:
        planner_signals = None

    # ---------- horizons / risk / actions capacity (unchanged) ----------
    tested = options.get("horizons", [7, 14, 30]) or [14]
    try:
        tested = [int(x) for x in tested]
    except Exception:
        tested = [7, 14, 30]
    chosen = tested[len(tested)//2] if tested else 14

    risk_preset = (options.get("risk_preset") or "balanced").lower().strip()
    cap_mult = {"conservative": 0.5, "balanced": 0.8, "aggressive": 1.0}.get(risk_preset, 0.8)

    def _as_float(x, default):
        try:
            v = float(x); 
            if pd.isna(v) or v in (float("inf"), float("-inf")): return default
            return v
        except Exception:
            return default

    ece       = _as_float(options.get("ece", 0.03), 0.03)
    coverage  = _as_float(options.get("coverage", 0.92), 0.92)
    alpha     = _as_float(options.get("alpha", 0.10), 0.10)
    threshold = _as_float(options.get("threshold", 0.71), 0.71)
    ops_cost_per_call = _as_float(options.get("ops_cost_per_call", 1.8), 1.8)

    actions_capacity = []
    max_spend_over_horizon = 0.0
    ops_calls_per_day = 0
    for a in norm_actions:
        daily_cap = int(a["daily_cap"])
        unit_cost = float(a["unit_cost"])
        channel   = a["channel"]
        horizon_cap = int(daily_cap * chosen * cap_mult)
        spend_cap  = float(unit_cost * horizon_cap)
        max_spend_over_horizon += spend_cap
        if channel.lower() == "calls":
            ops_calls_per_day += daily_cap
        actions_capacity.append({
            "name": a["name"],
            "channel": channel,
            "provider": a["provider"],
            "daily_cap": daily_cap,
            "cooldown_days": a["cooldown_days"],
            "unit_cost": unit_cost,
            "horizon_cap": horizon_cap,
            "horizon_spend_cap": round(spend_cap, 2),
        })
    ops_cost_estimate = round(ops_calls_per_day * ops_cost_per_call * chosen * cap_mult, 2)

    budget_cap = options.get("budgetCap")
    budget_cap = _as_float(budget_cap, None) if budget_cap is not None else None
    budget_notes = []
    if budget_cap is not None and max_spend_over_horizon > budget_cap:
        budget_notes.append(
            f"Max action spend over {chosen}d ({max_spend_over_horizon:.2f}) exceeds budget_cap ({budget_cap:.2f}). Will clip by risk preset."
        )

    derived = {
        **partial,
        "target_horizon": {"chosen": chosen, "tested": tested, "rationale": "midpoint (placeholder)"},
        "channel_reach": {
            "email": partial["eligible_population"]["emailable_pct"],
            "sms": partial["eligible_population"]["sms_opt_in_pct"],
            "calls": 0.12,
            "ops_cost_per_call": ops_cost_per_call,
        },
        "baseline": {
            "cvr_7d": _as_float(options.get("cvr_7d", 0.012), 0.012),
            "weekly_pattern": options.get("weekly_pattern", "Mon high"),
            "seasonality": options.get("seasonality", "Nov peak"),
        },
        "modeling": {"ece": ece, "coverage": coverage, "alpha": alpha, "threshold": threshold},
        "actions_capacity": actions_capacity,
        "max_spend_over_horizon": round(max_spend_over_horizon, 2),
        "ops_cost_estimate": ops_cost_estimate,
        "risk_preset": risk_preset,
        "budget": {"budget_cap": budget_cap, "notes": budget_notes},
    }

    return {
        "user": {"id": user_id},
        "present_kinds": [kind_specific],
        "used_mapping": used_mapping,
        "uploads": uploads,
        "options_sanitized": {**options, "risk_preset": risk_preset, "horizons": tested},
        "actions": norm_actions,
        "schema_contract": SCHEMA_CONTRACT,
        "file_health": file_health,
        "derived": derived,
        "warnings": warnings,
        "preflight": {"calc_gaps": calc_gaps, "coverage_pct": coverage_pct},
        "modified_csvs": {"customers": None, "orders": None, "marketing": None} | {kind_specific: uploads[kind_specific]},
        "planner_signals": planner_signals,
    }

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, conint, confloat
from typing import Any, Dict, List, Optional

# --------- IO Schemas (mirror your dataclasses) ----------
class IntentIn(BaseModel):
    goal: str = Field(
        "predict",
        pattern="^(predict|uplift|forecast|segment|survival|what_if|ab_test)$"  # ✅ v2 compatible
    )
    mode: str = Field("train", pattern="^(train|predict|analyze)$")

    risk_preset: str = Field("balanced", pattern="^(conservative|balanced|aggressive)$")
    constraints: Dict[str, Any] = {}

class ArtifactsIn(BaseModel):
    classification: bool = False
    regression: bool = False
    forecast: bool = False
    survival: bool = False
    clustering: bool = False
from typing import Optional, Union, Literal

class TargetSpecModel(BaseModel):
    task: Literal["classification","regression","forecast","survival","segment","auto"] = "auto"
    target_column: Optional[str] = None
    positive_class: Optional[Union[str, int, bool]] = None
    ts_column: Optional[str] = None
    id_column: Optional[str] = None
    horizon_days: Optional[int] = None
def _as_target_model(obj) -> TargetSpecModel:
    # Accept dict or pydantic model; default to auto if malformed
    try:
        if isinstance(obj, TargetSpecModel):
            return obj
        if hasattr(obj, "model_dump"):
            return TargetSpecModel(**obj.model_dump())
        if isinstance(obj, dict):
            return TargetSpecModel(**obj)
    except Exception:
        pass
    return TargetSpecModel()  # task="auto", everything else None
class CompileRequest(BaseModel):
    intent: IntentIn
    signals: Optional[SignalsIn] = None
    artifacts: ArtifactsIn
    top_k: int = 1
    dataset_id: Optional[int] = None
    target: Optional[TargetSpecModel] = None 
# --------- Response Schemas (shape returned by compile_plan) ----------
class PlanStepOut(BaseModel):
    idx: int
    name: str
    endpoint: str
    depends_on: List[str]
    produces: List[str]
    consumes: List[str]
    est_cost: float
    est_latency_ms: int
    est_risk: float
    idempotent: bool
    once_per_dataset: bool
    family: str
    headers: Dict[str, Any]

class PlanSummaryOut(BaseModel):
    score: float
    value: float
    total_cost: float
    total_latency_ms: int
    total_risk: float
    families: List[str]
    chosen_core: Optional[str] = None

class CompileResponse(BaseModel):
    steps: List[PlanStepOut]
    summary: PlanSummaryOut
    # add what you’re already returning:
    budget_context: Optional[Dict[str, Any]] = None
    meta_used: Optional[Dict[str, Any]] = None
    signals: Optional[Dict[str, Any]] = None


import pandas as pd
from typing import Any, Dict, List, Optional, Union
def _pick_processed_key(ds: "Dataset") -> Optional[str]:
    """
    Choose the best object key for the processed dataset.
    """
    for attr in ("processed_file_path", "file_path"):
        k = getattr(ds, attr, None)
        if k:
            return k
    return None

def _read_table_local(path: str) -> Optional[pd.DataFrame]:
    """
    Read CSV/XLSX/Parquet best-effort from a local temp path.
    """
    p = Path(path)
    ext = p.suffix.lower()
    try:
        if ext in (".csv", ".txt", ".tsv"):
            # naive sniff for TSV
            sep = "\t" if ext == ".tsv" or p.name.endswith(".tsv") else ","
            return pd.read_csv(path, sep=sep, low_memory=False)
        if ext in (".xlsx", ".xls", ".xlsm"):
            return pd.read_excel(path)
        if ext in (".parquet",):
            return pd.read_parquet(path)  # requires pyarrow/fastparquet
        # default: try CSV anyway
        return pd.read_csv(path, low_memory=False)
    except Exception:
        # final fallback
        try:
            return pd.read_csv(path, low_memory=False)
        except Exception:
            return None
from auth import get_user_db, Dataset
def _download_processed_to_temp(dataset_id: int, current_user) -> Optional[str]:
    """
    Find the dataset row, resolve the processed key, and download it to a temp file.
    """
    bucket = os.environ.get("SUPABASE_BUCKET", "user-uploads")
    with get_user_db(current_user) as db:
        ds = db.query(Dataset).filter(Dataset.id == int(dataset_id)).first()
        if not ds:
            return None
        key = _pick_processed_key(ds)
        if not key:
            return None

    # download
    suffix = Path(str(key)).suffix or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
    try:
        sb_download_to_path(bucket, key, tmp_path)
        return tmp_path
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return None

def _load_processed_dataframe(dataset_id: int, current_user) -> Optional[pd.DataFrame]:
    """
    Convenience: download and read the processed table to a DataFrame.
    """
    tmp = _download_processed_to_temp(dataset_id, current_user)
    if not tmp:
        return None
    try:
        df = _read_table_local(tmp)
        return df
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


# --- signals construction fallback -------------------------------------------

def _resolve_target_column(df: pd.DataFrame, preferred: str | None) -> str:
    # Try caller’s preferred
    if preferred and preferred in df.columns:
        return preferred
    # Your pipeline writes helpers; fall back to them
    for alt in ["__target_cls", "__target_reg"]:
        if alt in df.columns:
            return alt
    raise ValueError(f"Target column '{preferred}' not found in processed data")
def build_signals_from_processed(df: pd.DataFrame, *, preferred_target: str | None) -> dict:
    cols = [str(c) for c in df.columns]
    low = [c.lower() for c in cols]
    id_col = None
    for i, c in enumerate(low):
        if c in {"id", "customer_id", "user_id", "uid"} or c.endswith("id"):
            id_col = cols[i]
            break
    ts_col = None
    for i, c in enumerate(low):
        if c in {"created_at", "order_date", "timestamp", "date", "processed_at"}:
            ts_col = cols[i]
            break

    target_col = _resolve_target_column(df, preferred_target)
    has_label = target_col in df

    label_type = None
    if has_label:
        nunique = int(df[target_col].nunique(dropna=True))
        if str(df[target_col].dtype) == "object":
            label_type = "binary" if nunique <= 2 else "multiclass"
        else:
            vals = pd.to_numeric(df[target_col], errors="coerce").dropna()
            uniq = set(vals.unique().tolist()[:5])
            label_type = "binary" if uniq.issubset({0, 1}) else ("multiclass" if nunique > 2 else "binary")

    return {
        "hasLabel": bool(has_label),
        "labelType": label_type,          # "binary" | "multiclass" | None (regression implied by planner)
        "hasTime": bool(ts_col),
        "tsColumn": ts_col,
        "idColumn": id_col,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "isCensored": False,
        "horizonDays": 30,                # sensible default; upstream can override
    }




def load_preview_for_dataset(dataset_id: Optional[int]) -> Optional[List[Dict[str, Any]]]:
    if dataset_id is None:
        return None
    return PREVIEW_STORE.get(dataset_id)

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from planner import _infer_label_type_from_preview
@dataclass
class PlanHint:
    goal: Optional[str] = None
    horizon_days: Optional[int] = None
    label_type: Optional[str] = None   # "binary" | "multiclass" | "numeric"
    time_column: Optional[str] = None
    id_column: Optional[str] = None
    preferred_families: Optional[List[str]] = None

def llm_infer_plan_hint(headers: List[str], preview_rows: List[Dict[str, Any]]) -> PlanHint:
    """
    Very lightweight heuristic replacement for an LLM.
    Looks at column names & first few rows to infer intent/signals.
    """
    headers_l = [h.lower() for h in headers]
    hint = PlanHint(preferred_families=[])

    # ---- Goal heuristics ----
    if any("churn" in h or "response" in h for h in headers_l):
        hint.goal = "predict"
    elif any("date" in h or "time" in h for h in headers_l):
        hint.goal = "forecast"
    else:
        hint.goal = "predict"

    # ---- Label type ----
    if "response" in headers_l or "churn" in headers_l:
        # check unique values in preview
        vals = {row.get("Response") for row in preview_rows if "Response" in row}
        if vals and len(vals) <= 2:
            hint.label_type = "binary"
        elif vals and len(vals) > 2:
            hint.label_type = "multiclass"
    elif any("amount" in h or "price" in h or "revenue" in h for h in headers_l):
        hint.label_type = "numeric"

    # ---- Time column ----
    for h in headers_l:
        if "date" in h or "time" in h or "timestamp" in h:
            hint.time_column = headers[headers_l.index(h)]
            hint.horizon_days = 14  # safe default horizon
            break

    # ---- ID column ----
    for h in headers_l:
        if h in ("id", "customer_id", "user_id", "account_id", "order_id"):
            hint.id_column = headers[headers_l.index(h)]
            break

    # ---- Family nudges ----
    if hint.label_type in ("binary", "multiclass"):
        hint.preferred_families.append("classification")
    elif hint.label_type == "numeric":
        hint.preferred_families.append("regression")
    if hint.time_column and not hint.label_type:
        hint.preferred_families.append("forecast")

    return hint
from dataclasses import asdict

FILL_FIELDS = ["hasLabel","labelType","hasTime","isCensored","horizonDays",
               "eventRate","rows","cols","tsColumn","idColumn"]

@router.post("/compile", response_model=CompileResponse)
def compile_route(payload: CompileRequest, current_user=Depends(get_current_active_user)):
    try:
        SAFETY_RESERVE = 0.25  # leave a little runway in the user's token balance

        # ---- identity ----
        user_id = getattr(current_user, "id", None) or getattr(current_user, "user_id", None)
        if user_id is None:
            raise HTTPException(status_code=401, detail="Unauthenticated")

        # ---- helpers ----
        def to_dict(x):
            if x is None: return {}
            if hasattr(x, "model_dump"): return x.model_dump()
            if hasattr(x, "dict"): return x.dict()
            if hasattr(x, "__dict__"): return dict(getattr(x, "__dict__"))
            if isinstance(x, dict): return dict(x)
            return {}

        # ---- 1) Signals: load saved → else build from processed → else preview map → finally payload overrides ----
        signals_dict: Dict[str, Any] = {}
        meta_used_auto: Dict[str, Any] = {}

        dataset_id = getattr(payload, "dataset_id", None)
        preferred_target = None
        if getattr(payload, "target", None):
            try:
                preferred_target = (payload.target.target_column
                                    if hasattr(payload.target, "target_column")
                                    else (payload.target or {}).get("target_column"))
            except Exception:
                preferred_target = None

        if dataset_id is not None:
            # 1a) Try saved signals first
            saved = load_signals(user_id=user_id, dataset_id=int(dataset_id))
            if isinstance(saved, dict) and saved:
                signals_dict.update(saved)
                meta_used_auto = {
                    "rows": saved.get("rows"),
                    "cols": saved.get("cols"),
                    "tsColumn": saved.get("tsColumn"),
                    "idColumn": saved.get("idColumn"),
                    "hasLabel": saved.get("hasLabel"),
                    "labelType": saved.get("labelType"),
                    "hasTime": saved.get("hasTime"),
                    "horizonDays": saved.get("horizonDays"),
                }
            else:
                # 1b) No saved signals → attempt to build from processed dataset
                df_proc = _load_processed_dataframe(int(dataset_id), current_user)
                if df_proc is not None and len(df_proc):
                    built = build_signals_from_processed(df_proc, preferred_target=preferred_target)
                    signals_dict.update(built or {})
                    meta_used_auto = {
                        "rows": built.get("rows"),
                        "cols": built.get("cols"),
                        "tsColumn": built.get("tsColumn"),
                        "idColumn": built.get("idColumn"),
                        "hasLabel": built.get("hasLabel"),
                        "labelType": built.get("labelType"),
                        "hasTime": built.get("hasTime"),
                        "horizonDays": built.get("horizonDays"),
                    }

        # 1c) If caller provided a target + we have a preview cache, nudge missing pieces
        preview_rows = PREVIEW_STORE.get(int(dataset_id)) if dataset_id is not None else None
        if getattr(payload, "target", None) and (preview_rows or []):
            try:
                t_model = _as_target_model(payload.target)
                from pydantic import BaseModel
                mapped = map_target_to_signals(t_model, preview_rows)
                mapped_dict = mapped.model_dump() if hasattr(mapped, "model_dump") else dict(mapped or {})
                # fill only missing
                for k in FILL_FIELDS:
                    if signals_dict.get(k) in (None, "", [], {}):
                        v = mapped_dict.get(k)
                        if v is not None:
                            signals_dict[k] = v
            except Exception:
                pass

        # 1d) Finally, explicit payload signals override everything
        if getattr(payload, "signals", None):
            signals_dict.update(to_dict(payload.signals))

        # Nudge: if time present but unlabeled, set a sane default horizon
        if signals_dict.get("hasTime") and not signals_dict.get("hasLabel") and not signals_dict.get("horizonDays"):
            signals_dict["horizonDays"] = 14

        # ---- 2) Budget enforcement ----
        intent_dict: Dict[str, Any] = to_dict(payload.intent) or {}
        cons: Dict[str, Any] = dict(intent_dict.get("constraints") or {})

        ignore_caps = bool(cons.get("ignoreBudgetCaps", False))
        requested_cap = cons.get("budgetCap", None)

        balance = float(getattr(current_user, "tokens", 0.0) or 0.0)
        cap_from_tokens = max(0.0, balance - SAFETY_RESERVE)

        if not ignore_caps:
            if requested_cap is None:
                effective_cap = cap_from_tokens
            else:
                try:
                    requested_cap = float(requested_cap)
                except Exception:
                    requested_cap = 0.0
                effective_cap = max(0.0, min(requested_cap, cap_from_tokens))
            cons["budgetCap"] = round(effective_cap, 2)
            intent_dict["constraints"] = cons

        # ---- 3) Compile plan ----
        artifacts_dict = to_dict(payload.artifacts)
        plan = compile_plan(
            intent=intent_dict,
            signals=signals_dict,
            artifacts=artifacts_dict,
            top_k=getattr(payload, "top_k", 1) or 1,
        )

        # ---- 4) Materialize response ----
        summary = {
            "score": plan.summary.score,
            "value": plan.summary.value,
            "total_cost": plan.summary.total_cost,
            "total_latency_ms": plan.summary.total_latency_ms,
            "total_risk": plan.summary.total_risk,
            "families": list(plan.summary.families),
            "chosen_core": plan.summary.chosen_core,
        }

        steps_out: List[Dict[str, Any]] = []
        for st in plan.steps:
            s = st if isinstance(st, dict) else {
                "idx": getattr(st, "idx", 0),
                "name": getattr(st, "name", ""),
                "endpoint": getattr(st, "endpoint", ""),
                "depends_on": list(getattr(st, "depends_on", []) or []),
                "produces": list(getattr(st, "produces", []) or []),
                "consumes": list(getattr(st, "consumes", []) or []),
                "est_cost": getattr(st, "est_cost", 0.0),
                "est_latency_ms": getattr(st, "est_latency_ms", 0),
                "est_risk": getattr(st, "est_risk", 0.0),
                "idempotent": getattr(st, "idempotent", True),
                "once_per_dataset": getattr(st, "once_per_dataset", False),
                "family": getattr(st, "family", "other"),
                "headers": getattr(st, "headers", {}) or {},
            }
            ep = s.get("endpoint", "") or ""
            if not ep.startswith("/api/"):
                ep = ("/api" + ep) if ep.startswith("/") else ("/api/" + ep)
            steps_out.append({
                "idx": int(s.get("idx", 0)),
                "name": s.get("name", ""),
                "endpoint": ep,
                "depends_on": list(s.get("depends_on", []) or []),
                "produces": list(s.get("produces", []) or []),
                "consumes": list(s.get("consumes", []) or []),
                "est_cost": float(s.get("est_cost", 0.0)),
                "est_latency_ms": int(s.get("est_latency_ms", 0)),
                "est_risk": float(s.get("est_risk", 0.0)),
                "idempotent": bool(s.get("idempotent", True)),
                "once_per_dataset": bool(s.get("once_per_dataset", False)),
                "family": s.get("family", "other"),
                "headers": s.get("headers", {}) or {},
            })

        est_total_cost = round(sum(s["est_cost"] for s in steps_out), 2)
        remaining = None
        if not ignore_caps and cons.get("budgetCap") is not None:
            try:
                remaining = round(float(cons["budgetCap"]) - float(est_total_cost), 2)
            except Exception:
                remaining = None

        budget_context = {
            "balance": round(balance, 2),
            "requested": None if requested_cap is None else float(requested_cap),
            "cap_used": None if ignore_caps else float(cons.get("budgetCap", 0.0)),
            "est_total_cost": float(est_total_cost),
            "remaining_after_plan": remaining,
            "ignore_caps": bool(ignore_caps),
        }

        meta_used = {
            **{
                "rows": None,
                "cols": None,
                "tsColumn": None,
                "idColumn": None,
                "hasLabel": None,
                "labelType": None,
                "hasTime": None,
                "horizonDays": None,
            },
            **(meta_used_auto or {}),
        }

        return {
            "steps": steps_out,
            "summary": summary,
            "budget_context": budget_context,
            "meta_used": meta_used,
            "signals": signals_dict,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))