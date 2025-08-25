# planner_router.py
from __future__ import annotations
import json
from typing import Optional, Any, Dict
import pandas as pd
from fastapi import APIRouter, Depends, File, UploadFile, Request, HTTPException
import math
# --- Mapping helpers ----------------------------------------------------------
import re
from typing import Dict, List, Optional
import pandas as pd
import io, os, re, mimetypes, tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from auth import get_current_active_user, User
from storage import upload_file_to_supabase
# from .auth import get_current_active_user
# from .storage import upload_file_to_supabase
router = APIRouter(prefix="/api/plan", tags=["planner"])
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def _norm(s: str) -> str:
    return re.sub(r"(^_+|_+$)", "",
           re.sub(r"[^\w]+","_", s.strip().lower()))

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_norm(c) for c in out.columns]
    return out

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

def coerce_customers(df: Optional[pd.DataFrame], m: Dict[str, Optional[str]]) -> pd.DataFrame:
    if df is None: return pd.DataFrame(columns=["customer_id","email","phone","created_at"])
    df = normalize_headers(df)
    cid = m.get("customer_id"); em = m.get("email"); ph = m.get("phone")
    created = m.get("created_at") or m.get("created_date")
    out = pd.DataFrame()
    out["customer_id"] = df[cid].astype(str) if cid and cid in df.columns else pd.NA
    out["email"]      = clean_email(df[em]) if em and em in df.columns else pd.NA
    out["phone"]      = clean_phone(df[ph]) if ph and ph in df.columns else pd.NA
    out["created_at"] = to_dt(df[created]) if created and created in df.columns else pd.NaT
    out = out.dropna(subset=["customer_id"]).drop_duplicates("customer_id", keep="last")
    return out


def coerce_orders(df: Optional[pd.DataFrame], m: Dict[str, Optional[str]]) -> pd.DataFrame:
    if df is None: return pd.DataFrame(columns=["order_id","customer_id","order_date","amount"])
    df = normalize_headers(df)
    oid = m.get("order_id"); cid = m.get("customer_id")
    dt  = m.get("order_date"); amt = m.get("order_total")
    out = pd.DataFrame()
    out["order_id"]    = df[oid].astype(str) if oid and oid in df.columns else pd.NA
    out["customer_id"] = df[cid].astype(str) if cid and cid in df.columns else pd.NA
    out["order_date"]  = to_dt(df[dt]) if dt and dt in df.columns else pd.NaT
    out["amount"]      = to_num_currency(df[amt]) if amt and amt in df.columns else pd.NA
    out = out.dropna(subset=["order_id"]).drop_duplicates("order_id", keep="last")
    out = out[out["amount"].fillna(0) > 0]  # keep positive orders for AOV
    return out
def coerce_marketing(df: Optional[pd.DataFrame], m: Dict[str, Optional[str]]) -> pd.DataFrame:
    if df is None: return pd.DataFrame(columns=[
        "campaign_id","customer_id","channel","sent_date","opened","clicked","converted","cost"
    ])
    df = normalize_headers(df)
    cid = m.get("campaign_id"); cust = m.get("customer_id")
    ch  = m.get("channel"); sent = m.get("sent_date")
    opened = m.get("opened"); clicked = m.get("clicked"); converted = m.get("converted")
    cost = m.get("cost") or m.get("campaign_cost")
    out = pd.DataFrame()
    out["campaign_id"] = df[cid].astype(str) if cid and cid in df.columns else pd.NA
    out["customer_id"] = df[cust].astype(str) if cust and cust in df.columns else pd.NA
    out["channel"]     = df[ch].astype(str).str.strip().str.lower() if ch and ch in df.columns else pd.NA
    out["sent_date"]   = to_dt(df[sent]) if sent and sent in df.columns else pd.NaT
    out["opened"]      = to_bool01(df[opened]) if opened and opened in df.columns else pd.NA
    out["clicked"]     = to_bool01(df[clicked]) if clicked and clicked in df.columns else pd.NA
    out["converted"]   = to_bool01(df[converted]) if converted and converted in df.columns else pd.NA
    out["cost"]        = to_num_currency(df[cost]).fillna(0) if cost and cost in df.columns else 0.0
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
    Create CSV bytes from df, upload to Supabase using your helper, return metadata + public url.
    """
    out: Dict[str, any] = {"kind": kind, "filename": "", "path": "", "public_url": "", "size_bytes": 0, "columns": [], "inline_b64": None, "preview": []}
    if df is None:
        return out

    csv_bytes = _df_to_csv_bytes(df)
    meta = _csv_meta(csv_bytes)
    out["size_bytes"] = meta["size_bytes"]
    out["columns"] = list(df.columns)

    # Build a consistent stamped filename
    safe_original = _slug_filename(f"{kind}.csv")
    stamped_name  = f"derive/{_now_tag()}-{kind}-{safe_original}"

    # Write bytes to temp file to reuse upload_file_to_supabase(user_id, path, filename)
    suffix = Path(stamped_name).suffix or ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        if csv_bytes:
            tmp.write(csv_bytes)
        tmp_path = tmp.name

    try:
        storage_path = upload_file_to_supabase(user_id, tmp_path, stamped_name)
        out["filename"] = stamped_name
        out["path"] = storage_path
        try:
            public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(storage_path)  # type: ignore[attr-defined]
        except Exception:
            public_url = ""
        out["public_url"] = public_url or ""

        # Inline b64 only for small files
        if csv_bytes and len(csv_bytes) <= INLINE_B64_MAX:
            out["inline_b64"] = base64.b64encode(csv_bytes).decode("ascii")

        # Preview rows
        try:
            preview_df = df.head(CSV_PREVIEW_ROWS)
            out["preview"] = [list(preview_df.columns)] + preview_df.astype(object).where(pd.notna(preview_df), "").values.tolist()
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
                dt = dt.dt.tz_localize("UTC").dt.tz_convert(user_tz)
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

    coverage_pct = round(100.0 * ok_calcs / max(total_calcs, 1), 1)

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
        "meta": {
            "calc_coverage_pct": coverage_pct,
        },
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

def _read_table_bytes(data: Optional[bytes], filename: Optional[str], expected_kind: Optional[str]=None) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Read tabular data from bytes across csv, csv.gz, xlsx, parquet.
    Returns (df or None, warnings[]).
    """
    warns: List[str] = []
    if not data:
        return None, warns

    fmt = _sniff_format(filename, data)
    try:
        if fmt == "csv":
            return pd.read_csv(io.BytesIO(data)), warns
        if fmt == "csv.gz":
            # pandas handles compression arg; this avoids double IO
            return pd.read_csv(io.BytesIO(data), compression="gzip"), warns
        if fmt == "xlsx":
            try:
                xls = pd.ExcelFile(io.BytesIO(data))
            except Exception as e:
                warns.append(f"xlsx open failed: {e}")
                return None, warns
            sheet = _pick_excel_sheet(xls, expected_kind)
            try:
                df = pd.read_excel(xls, sheet_name=sheet)
                return df, warns
            except Exception as e:
                warns.append(f"xlsx read failed (sheet {sheet}): {e}")
                return None, warns
        if fmt == "parquet":
            try:
                # requires pyarrow or fastparquet
                return pd.read_parquet(io.BytesIO(data)), warns
            except Exception as e:
                warns.append(f"parquet read failed (install pyarrow or fastparquet): {e}")
                return None, warns
    except Exception as e:
        warns.append(f"{fmt} parse error: {e}")
        return None, warns

    # fallback
    try:
        return pd.read_csv(io.BytesIO(data)), warns
    except Exception as e:
        warns.append(f"fallback csv parse error: {e}")
        return None, warns

# ----------------- endpoint -----------------
@router.post("/derive")
async def derive_columns(
    request: Request,
    customers_file: UploadFile | None = File(None),
    orders_file:    UploadFile | None = File(None),
    marketing_file: UploadFile | None = File(None),
    current_user:   "User" = Depends(get_current_active_user),
):
    form = await request.form()

    # JSON fields (string preferred; your _parse_json_field should also accept UploadFile if present)
    mappings = await _parse_json_field(form.get("mappings"), "mappings")
    options  = await _parse_json_field(form.get("options"),  "options")
    actions  = await _parse_json_field(form.get("actions"),  "actions")  # optional separate field

    # Merge actions: explicit 'actions' field takes precedence, else options.allowed_actions
    if not actions:
        actions = (options or {}).get("allowed_actions", [])
    norm_actions, warnings = _normalize_actions(actions)


    # Buffer files exactly once
    cust_bytes, cust_name = await _buffer_upload(customers_file)
    ord_bytes,  ord_name  = await _buffer_upload(orders_file)
    mkt_bytes,  mkt_name  = await _buffer_upload(marketing_file)

    # Present kinds: from client hint or inferred
    present_kinds = await _parse_json_field(form.get("present_kinds"), "present_kinds") or _present_from_bytes(cust_bytes, ord_bytes, mkt_bytes)

    # Resolve user id from auth
    user_id = str(_resolve_user_id(request, current_user))

    # Upload to Supabase (optional; only if present)
    uploads: Dict[str, Optional[Dict[str, str]]] = {"customers": None, "orders": None, "marketing": None}

    async def _upload(kind: str, data: Optional[bytes], original_name: Optional[str]) -> Optional[Dict[str, str]]:
        if not data:
            return None
        safe_original = _slug_filename(original_name or f"{kind}.csv")
        # Optional: group uploads under a run id to avoid filename collisions
        run_id = uuid.uuid4().hex[:12]
        stamped_name = f"derive/{_now_tag()}-{run_id}-{kind}-{safe_original}"

        suffix = Path(stamped_name).suffix or ".csv"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            storage_path = upload_file_to_supabase(user_id, tmp_path, stamped_name)
            try:
                public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(storage_path)  # type: ignore[attr-defined]
            except Exception:
                public_url = ""
            return {"path": storage_path, "filename": stamped_name, "public_url": public_url}
        finally:
            try: os.remove(tmp_path)
            except Exception: pass

    uploads["customers"] = await _upload("customers", cust_bytes, cust_name)
    uploads["orders"]    = await _upload("orders",    ord_bytes,  ord_name)
    uploads["marketing"] = await _upload("marketing", mkt_bytes,  mkt_name)

    # DataFrames
    read_warnings: List[str] = []

    df_customers, w1 = _read_table_bytes(cust_bytes, cust_name, expected_kind="customers")
    df_orders,    w2 = _read_table_bytes(ord_bytes,  ord_name,  expected_kind="orders")
    df_mkt,       w3 = _read_table_bytes(mkt_bytes,  mkt_name,  expected_kind="marketing")
    read_warnings.extend(w1 + w2 + w3)

    warnings.extend(read_warnings)
    # Normalize headers
    if df_customers is not None: df_customers = normalize_headers(df_customers)
    if df_orders    is not None: df_orders    = normalize_headers(df_orders)
    if df_mkt       is not None: df_mkt       = normalize_headers(df_mkt)

    # Mapping (UI first, then auto)
    ui_cust = mappings.get("customers") if isinstance(mappings.get("customers"), dict) else None
    ui_ord  = mappings.get("orders")    if isinstance(mappings.get("orders"),    dict) else None
    ui_mkt  = mappings.get("marketing") if isinstance(mappings.get("marketing"), dict) else None

    cust_map  = finalize_mapping(df_customers, ui_cust, CUSTOMER_SYNS)
    order_map = finalize_mapping(df_orders,    ui_ord,  ORDER_SYNS)
    mkt_map   = finalize_mapping(df_mkt,       ui_mkt,  MKT_SYNS)

    # Coerce (safe if df is None)
    C = coerce_customers(df_customers, cust_map)
    O = coerce_orders(df_orders,       order_map)
    M = coerce_marketing(df_mkt,       mkt_map)
    file_health = {
        "customers": _file_health(C if C is not None and len(C) else None, "customers"),
        "orders":    _file_health(O if O is not None and len(O) else None, "orders"),
        "marketing": _file_health(M if M is not None and len(M) else None, "marketing"),
    }
    C, c_transforms = apply_common_transforms(C, key_field="customer_id", user_tz=options.get("user_tz"))
    O, o_transforms = apply_common_transforms(O, key_field="order_id", user_tz=options.get("user_tz"))
    M, m_transforms = apply_common_transforms(M, key_field="campaign_id", user_tz=options.get("user_tz"))

    modified_csvs = {
        "customers": await _upload_csv_bytes(user_id, "customers", C if len(C) else None),
        "orders":    await _upload_csv_bytes(user_id, "orders",    O if len(O) else None),
        "marketing": await _upload_csv_bytes(user_id, "marketing", M if len(M) else None),
    }
    # --- Preflight + safe compute (non-blocking)
    partial, calc_gaps, coverage_pct = _preflight_and_compute(C, O, M, options)

    # Horizon selection (unchanged)
    tested = options.get("horizons", [7, 14, 30]) or [14]
    try:
        tested = [int(x) for x in tested]
    except Exception:
        tested = [7, 14, 30]
    chosen = tested[len(tested)//2] if tested else 14

    # Risk preset → capacity scaler
    risk_preset = (options.get("risk_preset") or "balanced").lower().strip()
    risk_knobs = {
        "conservative": {"cap_multiplier": 0.5},
        "balanced":     {"cap_multiplier": 0.8},
        "aggressive":   {"cap_multiplier": 1.0},
    }.get(risk_preset, {"cap_multiplier": 0.8})
    cap_mult = _as_float(risk_knobs["cap_multiplier"], 0.8)

    # Modeling placeholders (keep your knobs)
    ece       = _as_float(options.get("ece", 0.03), 0.03)
    coverage  = _as_float(options.get("coverage", 0.92), 0.92)
    alpha     = _as_float(options.get("alpha", 0.10), 0.10)
    threshold = _as_float(options.get("threshold", 0.71), 0.71)
    chance_to_goal  = _as_float(options.get("chance_to_goal", 0.78), 0.78)
    channel_mix       = options.get("channel_mix") or {"email": 0.7, "sms": 0.25, "calls": 0.05}
    ops_cost_per_call = _as_float(options.get("ops_cost_per_call", 1.8), 1.8)

    # --- Actions capacity / budget (uses norm_actions/action_warnings computed earlier)
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

    budget_cap = options.get("budget_cap")
    budget_cap = _as_float(budget_cap, None) if budget_cap is not None else None
    budget_notes = []
    if budget_cap is not None and max_spend_over_horizon > budget_cap:
        budget_notes.append(
            f"Max action spend over {chosen}d ({max_spend_over_horizon:.2f}) exceeds budget_cap ({budget_cap:.2f}). Will clip by risk preset."
        )

    # Build derived by merging preflight results with the rest
    derived = {
        **partial,  # eligible_population, unit_economics, threshold_volume, meta.calc_coverage_pct
        "target_horizon": {"chosen": chosen, "tested": tested, "rationale": "midpoint (placeholder)"},
        "channel_reach": {
            "email": partial["eligible_population"]["emailable_pct"],
            "sms": partial["eligible_population"]["sms_opt_in_pct"],
            "calls": 0.12,  # placeholder
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
        "present_kinds": present_kinds,
        "used_mapping": {"customers": cust_map, "orders": order_map, "marketing": mkt_map},
        "uploads": uploads,
        "options_sanitized": {
            **options, "risk_preset": risk_preset, "horizons": tested,
            # "budget_cap": budget_cap,
        },
        "actions": norm_actions,
        "schema_contract": SCHEMA_CONTRACT,  # lets the UI label fields
        "file_health": file_health,          # stats + preview + issues
        "preflight": {
            "calc_gaps": calc_gaps,
            "coverage_pct": coverage_pct
        },

        "derived": derived,
        "warnings": warnings,          # your existing general warnings
        "preflight": {
            "calc_gaps": calc_gaps,    # ← per-calc missing columns & fallbacks
            "coverage_pct": coverage_pct
        },
        "modified_csvs": modified_csvs,  # if you added CSV export earlier
    }
