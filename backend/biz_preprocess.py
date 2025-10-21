# biz_preprocess.py
# Generalized preprocessing for business data, preserving your original pattern.
# - Optional data_dictionary.csv (variable,type) with types in:
#   {Categorical, Numerical, Date, Text, ID, Ignore, Group, Target:cls, Target:reg}
# - If no dictionary provided, types are inferred.
# - Numeric -> categorical via quantile bins (_trans2cat) unless opted out or in keep list.
# - OneHotEncoder persisted to disk with its feature list.
# - Returns train/test bundles compatible with your downstream CatBoost / LightGBM steps.

import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Optional (only if you'll train here; kept for parity with your code’s imports)
from catboost import CatBoostClassifier  # noqa: F401
from lightgbm import LGBMRegressor       # noqa: F401
from sklearn.model_selection import StratifiedKFold  # noqa: F401

import logging
from storage import upload_file_to_supabase

# from .storage import upload_file_to_supabase

logger = logging.getLogger(__name__)

@dataclass
class PreprocessConfig:
    data_dir: str = "./data/"
    encoder_info_dir: str = "./artifacts/encoders/"
    data_process_dir: str = "./artifacts/processed/"
    model_dir: Optional[str] = None  # not used by preprocess; ok to keep/omit

    # Column roles
    id_cols: List[str] = field(default_factory=lambda: ["ID"])
    target_cls_col: Optional[str] = None
    positive_label: Optional[str] = None
    target_reg_col: Optional[str] = None
    group_cols: List[str] = field(default_factory=list)

    # Optional manual type nudges (kept for quick fixes)
    treat_as_categorical: List[str] = field(default_factory=list)
    treat_as_numerical: List[str] = field(default_factory=list)
    ignore_cols: List[str] = field(default_factory=list)
    text_cols: List[str] = field(default_factory=list)

    # Dates: optional + auto-detected
    date_cols: Optional[List[str]] = None          # None => auto-detect
    auto_detect_dates: bool = True
    date_detect_sample: int = 200
    date_detect_success_ratio: float = 0.6
    date_name_hints: List[str] = field(default_factory=lambda:
        ["date", "time", "created", "updated", "timestamp", "ordered", "processed"]
    )

    # Numeric→categorical policy
    numeric_to_cat: bool = True
    numeric_bins: int = 10

    # Keep some numerics continuous: optional + auto
    numeric_keep_continuous: Optional[List[str]] = None   # None => rely on auto rules
    numeric_keep_mode: str = "auto+list"                  # {'auto','list','auto+list','none'}
    numeric_keep_min_unique: int = 30
    numeric_keep_min_unique_ratio: float = 0.05
    numeric_keep_name_hints: List[str] = field(default_factory=lambda:
        ["age","tenure","amount","price","cost","revenue","qty","quantity",
         "score","margin","days","duration","time","rate"]
    )

    # One-hot policy
    max_onehot_cardinality: int = 80
    one_hot_filename: str = "one_hot_encoder.pkl"

    # Dates expansion + NA policy
    add_date_parts: bool = True
    fillna_categorical: str = "-1"
     # Column name policies (regex strings, case-insensitive)
    allow_cols_patterns: List[str] = field(default_factory=list)  # force-keep (wins over deny)
    deny_cols_patterns:  List[str] = field(default_factory=list)  # force-ignore unless allowed

    # Train/test split strategy
    split_strategy: str = "random"         # {"random","time","group"}
    split_time_col: Optional[str] = None   # used if split_strategy=="time"
    split_time_cutoff: Optional[str] = None  # e.g., "2024-01-01"
    split_group_col: Optional[str] = None  # used if split_strategy=="group"
    test_size: float = 0.2
    random_seed: int = 42

    # High-cardinality categorical handling
    high_card_threshold: int = 1000
    high_card_action: str = "hash"         # {"hash","ignore","onehot"} (onehot = keep as-is)
    rare_min_count: int = 10
    rare_min_ratio: float = 0.001    # either threshold triggers bucketing
    rare_token: str = "__RARE__"
    text_hash_features: bool = True
    text_hash_buckets: int = 1024
    text_min_len: int = 5
    leak_patterns: List[str] = field(default_factory=lambda: [r"label", r"target", r"outcome", r"refund", r"chargeback", r"settled", r".*_next"])
    # Export safety
    excel_safe_export: bool = True
    verbose: bool = True

# -----------------------------
# Utilities
# -----------------------------
def _apply_rare_bucket(cats_df: pd.DataFrame, *, min_count: int, min_ratio: float, rare_token: str) -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
    """Return (cats_df_bucketted, mapping_per_col {col: {rare_value->rare_token}})"""
    if cats_df.empty:
        return cats_df, {}
    n = len(cats_df)
    out = cats_df.copy()
    mapping: Dict[str, Dict[str, str]] = {}
    for c in out.columns:
        try:
            vc = out[c].value_counts(dropna=False)
            cutoff = max(min_count, int(np.ceil(n * min_ratio)))
            rares = set(vc[vc <= cutoff].index.astype(str))
            if rares:
                out[c] = out[c].astype(str).where(~out[c].astype(str).isin(rares), rare_token)
                mapping[c] = {v: rare_token for v in rares}
        except Exception:
            continue
    return out, mapping

def _ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)
import re

def _compile_patterns(pats: List[str]) -> List[re.Pattern]:
    out = []
    for p in (pats or []):
        try:
            out.append(re.compile(p, re.I))
        except re.error:
            pass
    return out

def _matches_any(name: str, patterns: List[re.Pattern]) -> bool:
    n = str(name or "")
    for rx in patterns:
        if rx.search(n):
            return True
    return False

def _excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Prefix dangerous cells to avoid CSV formula injection when opening in Excel."""
    if df.empty:
        return df
    danger = ("=", "+", "-", "@", "\t", "\r")
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            s = out[c].astype(str)
            mask = s.str.startswith(danger)
            if mask.any():
                out.loc[mask, c] = "'" + s.loc[mask]
    return out

def safe_to_csv(df: pd.DataFrame, path: str, excel_safe: bool = True, **kwargs):
    df2 = _excel_safe(df) if excel_safe else df
    df2.to_csv(path, index=False, **kwargs)

def _as_list(x):
    if x is None: return []
    if isinstance(x, (list, tuple)): return list(x)
    if isinstance(x, pd.Series):     return x.tolist()
    # allow numpy arrays too
    if hasattr(x, "tolist"):         return list(x.tolist())
    return [x]

def _is_valid_colname(x, columns) -> bool:
    # Accept only hashable scalars that are actually in df.columns
    if isinstance(x, (pd.Series, pd.DataFrame)):  # reject non-scalars
        return False
    try:
        hash(x)
    except TypeError:
        return False
    return x in columns
import re
# simple, local slug — mirrors your intake style enough for matching
def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)   # zero-width + BOM
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

def _case_or_slug_lookup(name: str, columns: List[str]) -> Optional[str]:
    """Find a column by exact, case-insensitive, or slug-equal match."""
    if name in columns:
        return name
    lower_map = {c.lower(): c for c in columns}
    slug_map  = {_slug(c): c for c in columns}
    if name.lower() in lower_map:
        return lower_map[name.lower()]
    if _slug(name) in slug_map:
        return slug_map[_slug(name)]
    return None
def _first_valid_colname(x, columns):
    """Return a single valid column name or None (case/slug tolerant)."""
    if isinstance(x, (list, tuple, pd.Series, np.ndarray)):
        for v in x:
            # try exact/CI/slug match
            cand = _case_or_slug_lookup(v, list(columns))
            if cand is not None:
                return cand
        return None
    # single value
    cand = _case_or_slug_lookup(x, list(columns))
    return cand
def _infer_types(df: pd.DataFrame, cfg: PreprocessConfig) -> Dict[str, str]:
    """
    Return {col: type_str} among:
    Categorical, Numerical, Date, Text, ID, Ignore, Group, Target:cls, Target:reg
    """
    types: Dict[str, str] = {}

    cols = list(df.columns)
    # ----- NEW: allow/deny lists -----
    _allow = _compile_patterns(getattr(cfg, "allow_cols_patterns", []))
    _deny  = _compile_patterns(getattr(cfg, "deny_cols_patterns", []))

    forced_ignore = set()
    for col in cols:
        if _matches_any(col, _deny) and not _matches_any(col, _allow):
            forced_ignore.add(col)

    # pre-seed ignores so nothing else touches them
    for col in forced_ignore:
        # don't override explicit targets/ids/groups if user set them; allow wins later
        # we'll still respect allow list (checked again below)
        pass  # just mark; we’ll set types after seeding explicit roles


    # ---------- Seed fixed roles (GUARDED) ----------
    # IDs: resolve tolerant of case/slug mismatches
    resolved_ids: List[str] = []
    for c in _as_list(cfg.id_cols):
        cand = _first_valid_colname(c, cols)
        if cand is not None:
            types[cand] = "ID"
            resolved_ids.append(cand)

    # Groups
    for c in _as_list(cfg.group_cols):
        cand = _first_valid_colname(c, cols)
        if cand is not None:
            types[cand] = "Group"

    # Targets
    tc = _first_valid_colname(cfg.target_cls_col, cols)
    if tc is not None:
        types[tc] = "Target:cls"

    tr = _first_valid_colname(cfg.target_reg_col, cols)
    if tr is not None:
        types[tr] = "Target:reg"
    # Force ignores (deny unless explicitly allowed)
    for col in forced_ignore:
        if col not in types and not _matches_any(col, _allow):
            types[col] = "Ignore"

    # Explicit ignore
    for c in _as_list(cfg.ignore_cols):
        cand = _first_valid_colname(c, cols)
        if cand is not None:
            types[cand] = "Ignore"

    # Explicit user nudges
    for c in _as_list(cfg.treat_as_categorical):
        cand = _first_valid_colname(c, cols)
        if cand is not None:
            types[cand] = "Categorical"
    for c in _as_list(cfg.treat_as_numerical):
        cand = _first_valid_colname(c, cols)
        if cand is not None:
            types[cand] = "Numerical"
    for c in _as_list(cfg.text_cols):
        cand = _first_valid_colname(c, cols)
        if cand is not None:
            types[cand] = "Text"

    # Explicit dates (if provided)
    explicit_dates = set()
    for c in _as_list(cfg.date_cols):
        cand = _first_valid_colname(c, cols)
        if cand is not None:
            types[cand] = "Date"
            explicit_dates.add(cand)

    # ---------- Helper: detect dates (better) ----------
    rng = np.random.default_rng(42)

    def looks_like_date(series: pd.Series, name: str) -> bool:
        if not cfg.auto_detect_dates:
            return False

        # name hints
        name_hint = any(h in str(name).lower() for h in cfg.date_name_hints)

        s = series.dropna()
        if s.empty:
            return False

        # numeric epoch hint (10 or 13 digits typical for s/ms)
        if pd.api.types.is_numeric_dtype(s):
            try:
                sample = s.dropna()
                n = min(cfg.date_detect_sample, len(sample))
                if n == 0:
                    return False
                idx = rng.choice(sample.index.to_numpy(), size=n, replace=False)
                v = pd.to_numeric(sample.loc[idx], errors="coerce")
                # check plausible epoch seconds/millis
                is_epoch_s = ((v >= 1_000_000_000) & (v <= 10_000_000_000)).mean()
                is_epoch_ms = ((v >= 1_000_000_000_000) & (v <= 10_000_000_000_000)).mean()
                if max(is_epoch_s, is_epoch_ms) >= 0.6:
                    return True
            except Exception:
                pass

        # string parse-rate
        try:
            sample = s.astype(str)
            n = min(cfg.date_detect_sample, len(sample))
            if n == 0:
                return False
            idx = rng.choice(sample.index.to_numpy(), size=n, replace=False)
            parsed = pd.to_datetime(sample.loc[idx], errors="coerce", infer_datetime_format=True)
            ratio = float(parsed.notna().mean())
            thresh = cfg.date_detect_success_ratio - (0.15 if name_hint else 0.0)
            return ratio >= max(0.2, thresh)
        except Exception:
            return False

    # ---------- Auto-ignore junk (before inferring others) ----------
    # If the user *explicitly* typed a column, we do not override it here.
    for col in cols:
        if col in types:  # already assigned
            continue
        s = df[col]
        try:
            # --- skip ignore if column looks ID-ish or highly unique
            name_idish = (str(col).lower() == "id") or str(col).lower().endswith("_id")
            nonnull = s.dropna()
            uniq = nonnull.nunique(dropna=True) if len(nonnull) else 0
            ratio = (uniq / max(1, len(nonnull))) if len(nonnull) else 0.0
            if name_idish or ratio >= 0.90:
                continue  # let ID logic decide

            if s.isna().mean() >= 0.98 or s.nunique(dropna=True) <= 1:
                types[col] = "Ignore"
        except Exception:
            pass

    # ---------- Infer remaining columns ----------
    for col in cols:
        if col in types:
            continue
        s = df[col]

        # Date?
        if (pd.api.types.is_datetime64_any_dtype(s)) or (col not in explicit_dates and looks_like_date(s, col)):
            types[col] = "Date"
            continue

        # Try ID auto-promotion: high uniqueness, short tokens, not obviously date/URL/email/etc.
        try:
            nonnull = s.dropna()
            if len(nonnull) == 0:
                raise Exception()

            vals_str = nonnull.astype(str)

            # hard negatives for contact-like fields
            looks_email = vals_str.str.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$").mean() > 0.3
            looks_url   = vals_str.str.match(r"^(?:https?://|www\.)\S+$", flags=re.I).mean() > 0.3
            looks_ipv4  = vals_str.str.match(r"^(?:\d{1,3}\.){3}\d{1,3}$").mean() > 0.3
            if looks_email or looks_url or looks_ipv4:
                raise Exception()  # do not consider as ID

            uniq = nonnull.nunique(dropna=True)
            ratio = uniq / max(1, len(nonnull))
            avg_len = vals_str.map(len).mean()

            # UUID / hex / hyphenated token hints
            uuidish    = vals_str.str.match(r"^[0-9a-fA-F\-]{16,}$").mean() > 0.5  # generous
            hyphenated = vals_str.str.contains(r"-").mean() > 0.3
            name_idish = (str(col).lower() == "id") or str(col).lower().endswith("_id")

            # dynamic thresholds
            need_unique = 0.85 if (name_idish or uuidish or hyphenated) else 0.95
            max_len = 80 if (uuidish or hyphenated) else 64

            # avoid numbers with decimals (true numeric measures)
            floaty = pd.api.types.is_float_dtype(s) and (vals_str.str.contains(r"\.").mean() > 0.2)

            if (ratio >= need_unique) and (avg_len <= max_len) and not floaty:
                types[col] = "ID"
                resolved_ids.append(col)
                continue
        except Exception:
            pass

        # Numeric vs cat vs text
        if pd.api.types.is_numeric_dtype(s):
            # Respect numeric_keep_* to decide categoricalization
            try:
                u = s.nunique(dropna=True)
                r = u / max(1, len(s))
            except Exception:
                u, r = 0, 0.0

            keep_list = set(_as_list(cfg.numeric_keep_continuous)) if cfg.numeric_keep_continuous else set()
            hinted_keep = any(h in col for h in cfg.numeric_keep_name_hints)
            auto_keep = (u >= cfg.numeric_keep_min_unique) or (r >= cfg.numeric_keep_min_unique_ratio) or hinted_keep

            if cfg.numeric_keep_mode in ("auto", "auto+list"):
                keep_numeric = auto_keep or (cfg.numeric_keep_mode == "auto+list" and col in keep_list)
            elif cfg.numeric_keep_mode == "list":
                keep_numeric = (col in keep_list)
            elif cfg.numeric_keep_mode == "none":
                keep_numeric = False
            else:
                keep_numeric = auto_keep  # fallback

            # If not keeping numeric and we’re allowed to bin → categorical
            if cfg.numeric_to_cat and not keep_numeric:
                types[col] = "Categorical"
            else:
                types[col] = "Numerical"
        else:
            # Non-numeric → choose between Categorical vs Text by cardinality & length
            try:
                uniq = s.nunique(dropna=True)
                ratio = uniq / max(1, len(s))
            except Exception:
                uniq, ratio = 0, 0.0

            if uniq == 0:
                types[col] = "Categorical"
            elif uniq <= 30 or ratio <= 0.05:
                types[col] = "Categorical"
            else:
                # check average token length to promote real text
                try:
                    avg_len = s.dropna().astype(str).map(len).mean()
                except Exception:
                    avg_len = 0.0
                types[col] = "Text" if avg_len >= 20 else "Categorical"

    # --- stabilize multiple IDs deterministically (name bonus, uniqueness, nulls)
    if resolved_ids:
        scores = []
        for c in resolved_ids:
            s = df[c]
            try:
                nonnull = s.dropna()
                uniq = nonnull.nunique(dropna=True)
                ratio = uniq / max(1, len(nonnull)) if len(nonnull) else 0.0
                null_rate = s.isna().mean()
            except Exception:
                ratio, null_rate = 0.0, 1.0

            name_bonus = 0
            cl = str(c).lower()
            if cl == "order_id":    name_bonus += 3
            if cl == "customer_id": name_bonus += 2
            if cl.endswith("_id"):  name_bonus += 1
            if cl == "id":          name_bonus += 1

            scores.append((c, ratio, -null_rate, name_bonus))

        # sort by: name_bonus desc, ratio desc, null_rate asc, then name asc
        scores.sort(key=lambda t: (-t[3], -t[1], t[2], t[0]))
        # we keep all as "ID"; the first entry in scores is the most "primary" if needed elsewhere

    return types



from typing import Tuple
import re
_token_re = re.compile(r"[A-Za-z0-9]{2,}")

def _simple_tokens(s: str) -> List[str]:
    return _token_re.findall((s or "").lower())

def _text_hash_block(series: pd.Series, buckets: int, key: Optional[str] = None, prefix: str = "txt") -> pd.DataFrame:
    idx = series.index
    mat = np.zeros((len(idx), buckets), dtype=np.int16)
    for i, val in enumerate(series.fillna("").astype(str).tolist()):
        toks = _simple_tokens(val)
        if not toks:
            continue
        for tok in toks:
            b = _hash_series_to_bucket(pd.Series([tok]), buckets=buckets, key=key).iloc[0]
            mat[i, b] += 1
    cols = [f"{prefix}_{j}" for j in range(buckets)]
    return pd.DataFrame(mat, index=idx, columns=cols)

def _build_type_lists(df: pd.DataFrame, cfg: PreprocessConfig
    ) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str], List[str]]:
    """
    Returns: (cat_cols, num_cols, date_cols, text_cols, id_cols, ignore_cols, group_cols)
    """
    types = _infer_types(df, cfg)

    cat_cols  = [c for c,t in types.items() if t == "Categorical"]
    num_cols  = [c for c,t in types.items() if t == "Numerical"]
    date_cols = [c for c,t in types.items() if t == "Date"]
    text_cols = [c for c,t in types.items() if t == "Text"]
    text_blocks = []
    text_cols_use = [c for c in (text_cols or []) if df[c].astype(str).str.len().mean() >= getattr(cfg, "text_min_len", 5)]
    if getattr(cfg, "text_hash_features", True) and text_cols_use:
        tb = int(getattr(cfg, "text_hash_buckets", 1024))
        for c in text_cols_use:
            block = _text_hash_block(df[c], buckets=tb, key=getattr(cfg, "hash_key", None), prefix=f"{c}__h")
            text_blocks.append(block)
    text_df = pd.concat(text_blocks, axis=1) if text_blocks else pd.DataFrame(index=df.index)

    id_cols   = [c for c,t in types.items() if t == "ID"]
    ignore    = [c for c,t in types.items() if t == "Ignore"]
    groups    = [c for c,t in types.items() if t == "Group"]

    # Remove targets from feature lists (GUARDED)
    t_cls = _first_valid_colname(cfg.target_cls_col, df.columns)
    t_reg = _first_valid_colname(cfg.target_reg_col, df.columns)
    for target in [t_cls, t_reg]:
        if target is None:
            continue
        for L in (cat_cols, num_cols, date_cols, text_cols):
            # safe remove
            if target in L:
                L.remove(target)


    # Ensure id/ignore/group don’t leak into features
    for col in id_cols + ignore + groups:
        for L in (cat_cols, num_cols, date_cols, text_cols):
            if col in L: L.remove(col)

    return cat_cols, num_cols, date_cols, text_cols, id_cols, ignore, groups, text_df

def _add_date_flags(df: pd.DataFrame, date_cols: List[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    added_cat, added_num = [], []
    ref = None
    # choose a primary date as reference for recency (max)
    for dc in date_cols:
        if dc in df.columns:
            try:
                s = pd.to_datetime(df[dc], errors="coerce")
                if s.notna().any():
                    ref = s.max()
                    break
            except Exception:
                continue
    for dc in date_cols:
        if dc not in df.columns:
            continue
        try:
            s = pd.to_datetime(df[dc], errors="coerce")
            wkend = s.dt.dayofweek.isin([5,6]).astype("int8")
            ms = s.dt.is_month_start.astype("int8")
            me = s.dt.is_month_end.astype("int8")
            df[f"{dc}__is_weekend"] = wkend; added_cat.append(f"{dc}__is_weekend")
            df[f"{dc}__is_month_start"] = ms; added_cat.append(f"{dc}__is_month_start")
            df[f"{dc}__is_month_end"] = me; added_cat.append(f"{dc}__is_month_end")
            if ref is not None:
                rec = (ref - s).dt.days.astype("float32")
                df[f"{dc}__recency_days"] = rec
                added_num.append(f"{dc}__recency_days")
        except Exception:
            continue
    return df, added_cat, added_num

def _add_date_parts(df: pd.DataFrame, date_cols: List[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Add year/month/dow; return (df, added_cat_cols, added_num_cols)."""
    added_cat, added_num = [], []
    for col in date_cols:
        # Robust parse
        parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
        y = f"{col}_year"
        m = f"{col}_month"
        dow = f"{col}_dow"
        df[y] = parsed.dt.year
        df[m] = parsed.dt.month
        df[dow] = parsed.dt.dayofweek
        # Treat month & dow as categorical; year often useful as numeric
        added_num.append(y)
        added_cat.extend([m, dow])
    return df, added_cat, added_num
from pandas.api.types import is_numeric_dtype

def _add_numeric_bins(df: pd.DataFrame, num_cols: list[str], cfg) -> tuple[pd.DataFrame, list[str]]:
    """
    Return (df_with_bins, added_cat_cols). Avoids column-by-column insert to prevent fragmentation.
    """
    added = {}
    added_names: list[str] = []
    for c in (num_cols or []):
        # guard: only bin numeric with enough distinct values
        s = pd.to_numeric(df[c], errors="coerce") if is_numeric_dtype(df[c]) else pd.to_numeric(df[c], errors="coerce")
        uniq = s.dropna().unique()
        if uniq.size < 2:
            continue
        try:
            binned = pd.qcut(s, q=cfg.numeric_bins, duplicates="drop").astype(str)
        except Exception:
            # fall back to pd.cut with auto bins if qcut fails on skew
            try:
                binned = pd.cut(s, bins=min(cfg.numeric_bins, max(2, len(uniq))), include_lowest=True).astype(str)
            except Exception:
                continue
        newc = f"{c}__bin"
        added[newc] = binned
        added_names.append(newc)

    if added:
        # single, defragmenting concat
        df = pd.concat([df, pd.DataFrame(added, index=df.index)], axis=1)
        # optional: defragment copy to silence any residual warnings
        df = df.copy()

    return df, added_names

from sklearn.preprocessing import OneHotEncoder

def make_onehot_encoder(handle_unknown: str = "ignore", dtype="float32"):
    """
    Create OHE that works on sklearn 0.24 … 1.6+.
    Newer: sparse_output; Older: sparse.
    """
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False, dtype=dtype)
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(handle_unknown=handle_unknown, sparse=False, dtype=dtype)


def _fit_or_load_onehot(X_cat_num: pd.DataFrame, cfg, ohe_features: list[str], fit: bool):
    enc_path = os.path.join(cfg.encoder_info_dir, "one_hot_encoder.pkl")
    if fit or not os.path.exists(enc_path):
        ohe = make_onehot_encoder()
        ohe.fit(X_cat_num[ohe_features] if ohe_features else X_cat_num.iloc[:, :0])
        os.makedirs(cfg.encoder_info_dir, exist_ok=True)
        with open(enc_path, "wb") as f:
            pickle.dump(ohe, f)
    else:
        with open(enc_path, "rb") as f:
            ohe = pickle.load(f)
    return {"encoder": ohe, "features": ohe_features}
def _transform_onehot(X_cat_num: pd.DataFrame, info: dict[str, object]) -> tuple[pd.DataFrame, list[str]]:
    ohe = info["encoder"]
    feats = info["features"] or []
    if feats:
        arr = ohe.transform(X_cat_num[feats]) if feats else np.empty((len(X_cat_num), 0))
    else:
        # empty selection → empty matrix
        from scipy import sparse
        arr = sparse.csr_matrix((len(X_cat_num), 0))

    # Ensure dense ndarray
    try:
        dense = arr.toarray() if hasattr(arr, "toarray") else np.asarray(arr)
    except AttributeError:
        dense = arr  # already ndarray

    try:
        cols = list(ohe.get_feature_names_out(feats))
    except AttributeError:
        cols = list(ohe.get_feature_names(feats))  # older sklearn

    onehot_df = pd.DataFrame(dense, columns=cols, index=X_cat_num.index)
    return onehot_df, cols
import hashlib

def _hash_series_to_bucket(s: pd.Series, buckets: int = 256, key: Optional[str] = None) -> pd.Series:
    def _h(x: str) -> int:
        if key:
            h = hashlib.blake2b(digest_size=8, key=key.encode("utf-8", "ignore"))
        else:
            h = hashlib.blake2b(digest_size=8)
        h.update(x.encode("utf-8", errors="ignore"))
        return int.from_bytes(h.digest(), byteorder="big") % buckets
    return s.fillna("").astype(str).map(_h).astype("int32")

# -----------------------------
# Core: feature_engineering (generalized) — upgraded, same name/signature
# -----------------------------
def feature_engineering_general(data_file: str, cfg: PreprocessConfig, train: bool):
    _ensure_dirs(cfg.encoder_info_dir, cfg.data_process_dir)
    df = pd.read_csv(data_file)

    # Try to unpack text_df if your helper returns it; otherwise create empty
    try:
        cat_cols, num_cols, date_cols, text_cols, id_cols, ignore_cols, group_cols, text_df = _build_type_lists(df, cfg)
    except ValueError:
        cat_cols, num_cols, date_cols, text_cols, id_cols, ignore_cols, group_cols = _build_type_lists(df, cfg)
        text_df = pd.DataFrame(index=df.index)

    # --- NEVER encode these as features ---
    protected_cols = set((id_cols or [])) | set(ignore_cols or []) \
        | {c for c in [cfg.target_cls_col, cfg.target_reg_col] if _is_valid_colname(c, df.columns)}

    # 1) Dates → parts + flags
    added_cat_from_dates: List[str] = []
    added_num_from_dates: List[str] = []
    if cfg.add_date_parts and len(date_cols) > 0:
        df, added_cat_from_dates, added_num_from_dates = _add_date_parts(df, date_cols)

    # Extra date flags/recency (safe if already parsed in _add_date_parts)
    try:
        df, add_cat2, add_num2 = _add_date_flags(df, date_cols)
        added_cat_from_dates += add_cat2
        added_num_from_dates += add_num2
    except NameError:
        # if you didn’t add _add_date_flags yet, just skip
        pass

    # 2) Numeric → categorical bins (if enabled)
    df, added_cat_from_bins = _add_numeric_bins(df, num_cols, cfg)

    # Build candidate lists (pre-filters)
    leak_rx = _compile_patterns(getattr(cfg, "leak_patterns", []))
    def _safe_cols(cols: List[str]) -> List[str]:
        return [c for c in cols if (c not in protected_cols and not _matches_any(c, leak_rx))]

    feature_cat = _safe_cols(cat_cols + added_cat_from_dates + added_cat_from_bins)
    feature_value = _safe_cols(num_cols + added_num_from_dates)

    # ----- Categoricals: hygiene + rare bucketing -----
    cats_df = df[feature_cat].copy() if feature_cat else pd.DataFrame(index=df.index)
    for c in cats_df.columns:
        cats_df[c] = cats_df[c].astype(str).fillna(cfg.fillna_categorical)

    try:
        cats_df, rare_map = _apply_rare_bucket(
            cats_df,
            min_count=getattr(cfg, "rare_min_count", 10),
            min_ratio=getattr(cfg, "rare_min_ratio", 0.001),
            rare_token=getattr(cfg, "rare_token", "__RARE__"),
        )
    except NameError:
        rare_map = {}

    # ----- Numerics: add missingness indicators -----
    nums_df = df[feature_value].copy() if feature_value else pd.DataFrame(index=df.index)
    miss_ind_cols = []
    if not nums_df.empty:
        for c in list(nums_df.columns):
            mname = f"{c}__is_missing"
            miss = nums_df[c].isna().astype("int8").rename(mname)
            miss_ind_cols.append(mname)
            nums_df[mname] = miss

    # ---------- High-cardinality plan ----------
    def _plan_categorical_encodings(cats: pd.DataFrame) -> Dict[str, str]:
        plan: Dict[str, str] = {}
        hi = int(getattr(cfg, "high_card_threshold", 1000))
        action = getattr(cfg, "high_card_action", "hash").lower()
        max_ohe = int(getattr(cfg, "max_onehot_cardinality", 80))
        for col in cats.columns:
            try:
                k = int(cats[col].nunique(dropna=False))
            except Exception:
                k = 0
            if k == 0:
                plan[col] = "ignore"
            elif k <= max_ohe:
                plan[col] = "onehot"
            elif k > hi:
                plan[col] = action if action in ("hash", "ignore", "onehot") else "hash"
            else:
                plan[col] = action if action in ("hash", "onehot") else "hash"
        return plan

    cat_plan = _plan_categorical_encodings(cats_df)
    plan_ignore = [c for c, a in cat_plan.items() if a == "ignore"]
    plan_ohe    = [c for c, a in cat_plan.items() if a == "onehot"]
    plan_hash   = [c for c, a in cat_plan.items() if a == "hash"]

    if plan_ignore:
        cats_df = cats_df.drop(columns=plan_ignore, errors="ignore")

    # 3) One-hot for small/medium cats
    encoder_input = pd.concat([cats_df, nums_df], axis=1)
    ohe_features = [c for c in plan_ohe if c in cats_df.columns]
    encoder_info = _fit_or_load_onehot(
        encoder_input,
        cfg,
        ohe_features=ohe_features,
        fit=train,
    )
    onehot_df, onehot_cols = _transform_onehot(encoder_input, encoder_info)
    if not onehot_df.empty:
        for c in onehot_df.columns:
            if not pd.api.types.is_sparse(onehot_df[c].dtype):
                onehot_df[c] = onehot_df[c].astype(pd.SparseDtype("int8", fill_value=0))

    # 3b) Hashing for high-card cats (keyed BLAKE2b, configurable buckets)
    import hashlib
    def _hash_series_to_bucket(s: pd.Series, *, buckets: int, key: Optional[str]) -> pd.Series:
        def _h(x: str) -> int:
            h = hashlib.blake2b(digest_size=8, key=(key.encode("utf-8") if key else b""))
            h.update(x.encode("utf-8", errors="ignore"))
            return int.from_bytes(h.digest(), "big") % buckets
        return s.fillna("").astype(str).map(_h).astype("int32")

    hash_buckets = int(getattr(cfg, "hash_buckets", 512))
    hash_key = getattr(cfg, "hash_key", None)
    hashed_blocks, hashed_names = [], []
    for c in plan_hash:
        if c not in cats_df.columns:
            continue
        hname = f"{c}__hash{hash_buckets}"
        hashed_blocks.append(_hash_series_to_bucket(cats_df[c], buckets=hash_buckets, key=hash_key).rename(hname))
        hashed_names.append(hname)
    hashed_df = pd.concat(hashed_blocks, axis=1) if hashed_blocks else pd.DataFrame(index=df.index)

    # Text hashed features (if your upstream didn’t already build text_df)
    if text_df is None or text_df.empty:
        if getattr(cfg, "text_hash_features", True) and (text_cols or []):
            try:
                tb = int(getattr(cfg, "text_hash_buckets", 1024))
                key = getattr(cfg, "hash_key", None)
                def _simple_tokens(s: str) -> List[str]:
                    return re.findall(r"[A-Za-z0-9]{2,}", (s or "").lower())
                def _text_hash_block(series: pd.Series, buckets: int, key: Optional[str], prefix: str) -> pd.DataFrame:
                    idx = series.index
                    mat = np.zeros((len(idx), buckets), dtype=np.int16)
                    for i, val in enumerate(series.fillna("").astype(str).tolist()):
                        toks = _simple_tokens(val)
                        for tok in toks:
                            b = _hash_series_to_bucket(pd.Series([tok]), buckets=buckets, key=key).iloc[0]
                            mat[i, b] += 1
                    cols = [f"{prefix}_{j}" for j in range(buckets)]
                    return pd.DataFrame(mat, index=idx, columns=cols)
                blocks = []
                for c in text_cols:
                    if df[c].astype(str).str.len().mean() >= getattr(cfg, "text_min_len", 5):
                        blocks.append(_text_hash_block(df[c], buckets=tb, key=key, prefix=f"{c}__h"))
                text_df = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=df.index)
            except Exception:
                text_df = pd.DataFrame(index=df.index)

    # Build feature matrix
    features = pd.concat([cats_df, nums_df, onehot_df, hashed_df, text_df], axis=1)

    # Fallback if nothing survived
    if features.shape[1] == 0:
        raw = df.copy()
        num_cols_fb = list(raw.select_dtypes(include=["number"]).columns)[:20]
        other_cols = [c for c in raw.columns if c not in num_cols_fb][:10]
        keep = [c for c in num_cols_fb + other_cols if c not in protected_cols and not _matches_any(c, leak_rx)]
        if keep:
            for c in other_cols:
                raw[c] = raw[c].astype(str)
            features = raw[keep]

    # Belt & suspenders: drop any leakage-pattern columns that slipped through
    for c in list(features.columns):
        if _matches_any(c, leak_rx):
            features.drop(columns=[c], inplace=True, errors="ignore")

    # Deterministic ordering
    if not features.empty:
        features = features.reindex(sorted(features.columns), axis=1)
    if not onehot_df.empty:
        onehot_df = onehot_df.reindex(sorted(onehot_df.columns), axis=1)

    # Prepare outputs (keep your keys)
    feature_onehot = list(onehot_cols)
    feature_cat_final = list(cats_df.columns)
    feature_value_final = list(nums_df.columns) + hashed_names

    bundle: Dict[str, pd.Series | pd.DataFrame | List[str]] = {
        "X": features,
        "feature_cat": feature_cat_final,
        "feature_value": feature_value_final,
        "feature_onehot": feature_onehot,
        "X_onehot": onehot_df,
        "onehot_out_names": onehot_cols,
        "text_hash_cols": list(text_df.columns) if text_df is not None else [],
        "categorical_plan": {c: a for c, a in cat_plan.items() if c in feature_cat},
        "categorical_cardinality": {c: int(cats_df[c].nunique(dropna=False)) for c in cats_df.columns} if not cats_df.empty else {},
        "missing_indicators": miss_ind_cols,
        "rare_bucket_map": rare_map,
    }

    # IDs — pass through only, never encoded
    if id_cols:
        primary_id = id_cols[0]
        if _is_valid_colname(primary_id, df.columns):
            bundle["ID"] = df[primary_id]

    # Groups — pass through
    if group_cols:
        if _is_valid_colname(group_cols[0], df.columns):
            bundle["group_primary"] = df[group_cols[0]]
        for gc in group_cols:
            if _is_valid_colname(gc, df.columns):
                bundle[f"group__{gc}"] = df[gc]
        bundle["group_cols"] = group_cols

    # Targets — never leak into features
    if _is_valid_colname(cfg.target_cls_col, df.columns):
        y = df[cfg.target_cls_col]
        if y.dtype == "O" and cfg.positive_label is not None:
            bundle["y_cls"] = (y == cfg.positive_label).astype(int)
        elif pd.api.types.is_bool_dtype(y):
            bundle["y_cls"] = y.astype(int)
        else:
            bundle["y_cls"] = (pd.to_numeric(y, errors="coerce").fillna(0) > 0).astype(int)
        bundle["target_cls_col"] = cfg.target_cls_col

    if _is_valid_colname(cfg.target_reg_col, df.columns):
        bundle["y_reg"] = pd.to_numeric(df[cfg.target_reg_col], errors="coerce")
        bundle["target_reg_col"] = cfg.target_reg_col

    return bundle



# ---------- Helpers for null handling & safety ----------
_STANDARD_NULL_MARKERS = ["", "na", "n/a", "none", "null", "nan", "nil", "-", "--"]

def _std_null_marker_list():
    out = set()
    for s in _STANDARD_NULL_MARKERS:
        out.update([s, s.upper(), s.title()])
    return list(out)

def standardize_missing_markers(df: pd.DataFrame) -> pd.DataFrame:
    """Trim strings & coerce common placeholders to NaN (object columns only)."""
    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols) == 0:
        return df
    df[obj_cols] = df[obj_cols].apply(lambda s: s.astype(str).str.strip())
    df[obj_cols] = df[obj_cols].replace(_std_null_marker_list(), np.nan)
    return df

def compute_null_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    total = df.shape[0] * df.shape[1]
    if total == 0:
        return 0.0
    return float(df.isna().sum().sum()) / float(total)

def _safe_list(x, default=None):
    if x is None: return (default or [])
    return list(x) if isinstance(x, (list, tuple)) else [x]
# -----------------------------
# Convenience: train/test save (same as your workflow)
# -----------------------------
# Convenience: train/test save (TEMP DIR + UPLOAD)
# -----------------------------
def build_and_save_train_test(cfg: PreprocessConfig,
                              train_csv: str = "train.csv",
                              test_csv: str = "test.csv",
                              *,
                              user_id: str):
    """
    Fit OHE on train, transform test, write CSVs under a temp user_outputs/, upload to Supabase.
    Returns a dict of remote paths.
    """
    import os, tempfile, pickle
    from pathlib import Path as PathL
    from dataclasses import replace

    train_path = os.path.join(cfg.data_dir, train_csv)
    test_path  = os.path.join(cfg.data_dir, test_csv)

    with tempfile.TemporaryDirectory() as temp_dir:
        user_dir = PathL(temp_dir) / "user_outputs"
        enc_dir  = user_dir / "encoders"
        proc_dir = user_dir / "processed"
        enc_dir.mkdir(parents=True, exist_ok=True)
        proc_dir.mkdir(parents=True, exist_ok=True)

        # Route artifacts into the temp dirs
        cfg_tmp = replace(cfg, encoder_info_dir=str(enc_dir), data_process_dir=str(proc_dir))

        # 1) Fit on train (also fits & saves the OHE in enc_dir)
        data_train = feature_engineering_general(train_path, cfg_tmp, train=True)
        # 2) Transform test with the saved OHE
        data_test  = feature_engineering_general(test_path,  cfg_tmp, train=False)

        # -> Build CSVs (features + optional targets/ids/group)
        def _bundle_to_df(bundle):
            df_out = bundle["X"].copy()
            if "y_cls" in bundle:        df_out["__target_cls"] = bundle["y_cls"]
            if "y_reg" in bundle:        df_out["__target_reg"] = bundle["y_reg"]
            if "ID" in bundle:           df_out["__ID"] = bundle["ID"]
            if "group_primary" in bundle: df_out["__group_primary"] = bundle["group_primary"]
            return df_out

        train_df = _bundle_to_df(data_train)
        test_df  = _bundle_to_df(data_test)

        # Write to user_outputs root (as you requested)
        train_csv_path = user_dir / "data_train.csv"
        test_csv_path  = user_dir / "data_test.csv"
        train_df.to_csv(train_csv_path, index=False)
        test_df.to_csv(test_csv_path, index=False)

        # Upload processed datasets
        train_remote = upload_file_to_supabase(user_id, str(train_csv_path), train_csv_path.name)
        test_remote  = upload_file_to_supabase(user_id, str(test_csv_path),  test_csv_path.name)

        # (Optional) upload the fitted encoder for serving
        enc_pkl = enc_dir / "one_hot_encoder.pkl"
        enc_remote = upload_file_to_supabase(user_id, str(enc_pkl), enc_pkl.name) if enc_pkl.exists() else None

        # Print feature counts (same logs as before)
        print("Categorical Feature count:", len(data_train["feature_cat"]))
        print("Numerical  Feature count:", len(data_train["feature_value"]))
        print("One-hot    Feature count:", len(data_train["feature_onehot"]))

        return {
            "train_csv": train_remote,
            "test_csv": test_remote,
            "encoder_pkl": enc_remote,
        }

# -----------------------------
# Model-ready “second process” helpers (generalized)
# -----------------------------
def cat_cls_second_process_generic(data_pkl_file: str, drop_columns: Optional[List[str]] = None):
    """
    Prepare a saved bundle for CatBoostClassifier (categorical dtypes).
    Returns (data, categories_dict) and saves categories for reproducibility.
    """
    with open(data_pkl_file, "rb") as f:
        data = pickle.load(f)

    drop_columns = drop_columns or []
    # Cast categoricals
    for col in data["feature_cat"]:
        if col in data["X"].columns:
            data["X"][col] = data["X"][col].astype("category")

    # Select column order: cats + nums + onehots (minus drops)
    input_columns = [c for c in (data["feature_cat"] + data["feature_value"] + data["feature_onehot"]) if c not in drop_columns]
    data["X"] = data["X"][input_columns]

    # Capture category levels for reproducibility across folds/serving
    categories = {}
    for col in data["X"].columns:
        if str(data["X"][col].dtype) == "category":
            categories[col] = data["X"][col].cat.categories.tolist()

    # Persist categories (optional)
    with open(os.path.join(os.path.dirname(data_pkl_file), "../encoders/categories_cat_cls.pkl"), "wb") as f:
        pickle.dump(categories, f)

    return data, categories


def lgb_reg_second_process_generic(data_pkl_file: str, drop_columns: Optional[List[str]] = None):
    """
    Prepare a saved bundle for LightGBM Regressor (handles categorical as pd.Categorical).
    Returns (data, categories_dict).
    """
    with open(data_pkl_file, "rb") as f:
        data = pickle.load(f)

    drop_columns = drop_columns or []
    # Cat + onehot as category dtypes (LightGBM handles this)
    for col in data["feature_cat"] + data["feature_onehot"]:
        if col in data["X"].columns:
            data["X"][col] = data["X"][col].astype("category")

    input_columns = [c for c in (data["feature_cat"] + data["feature_value"] + data["feature_onehot"]) if c not in drop_columns]
    data["X"] = data["X"][input_columns]

    categories = {}
    for col in data["X"].columns:
        if str(data["X"][col].dtype) == "category":
            categories[col] = data["X"][col].cat.categories.tolist()

    with open(os.path.join(os.path.dirname(data_pkl_file), "../encoders/categories_lgb_reg.pkl"), "wb") as f:
        pickle.dump(categories, f)

    return data, categories
# ---- Add to biz_preprocess.py (below your existing code) ----
import numpy as np
from sklearn.model_selection import train_test_split

def feature_engineering_general_df(df: pd.DataFrame, cfg: PreprocessConfig, train: bool):
    """
    DataFrame version of feature_engineering_general (no CSV read).
    Fits encoders if train=True; otherwise loads encoders and only transforms.
    """
    _ensure_dirs(cfg.encoder_info_dir, cfg.data_process_dir)

    cat_cols, num_cols, date_cols, text_cols, id_cols, ignore_cols, group_cols = _build_type_lists(df, cfg)
    # Dates
    added_cat_from_dates, added_num_from_dates = [], []
    if cfg.add_date_parts and len(date_cols) > 0:
        df, added_cat_from_dates, added_num_from_dates = _add_date_parts(df.copy(), date_cols)

    # Numeric -> categorical bins
    df, added_cat_from_bins = _add_numeric_bins(df, num_cols, cfg)

    feature_cat = cat_cols + added_cat_from_dates + added_cat_from_bins
    feature_value = num_cols + added_num_from_dates

    cats_df = df[feature_cat].copy() if feature_cat else pd.DataFrame(index=df.index)
    for c in cats_df.columns:
        cats_df[c] = cats_df[c].astype(str).fillna(cfg.fillna_categorical)
    nums_df = df[feature_value].copy() if feature_value else pd.DataFrame(index=df.index)

    # cardinality filter for OHE
    card = {c: cats_df[c].nunique(dropna=False) for c in cats_df.columns}
    ohe_features = [c for c in cats_df.columns if card[c] <= cfg.max_onehot_cardinality]

    encoder_info = _fit_or_load_onehot(pd.concat([cats_df, nums_df], axis=1), cfg, ohe_features, fit=train)
    onehot_df, onehot_cols = _transform_onehot(pd.concat([cats_df, nums_df], axis=1), encoder_info)
    # After computing onehot_df / onehot_cols and building `features`
    features = pd.concat([cats_df, nums_df, onehot_df], axis=1)
    feature_onehot = onehot_cols
    features = features.reindex(sorted(features.columns), axis=1)
    onehot_df = onehot_df.reindex(sorted(onehot_df.columns), axis=1) if not onehot_df.empty else onehot_df

    bundle: Dict[str, pd.Series | pd.DataFrame | List[str]] = {
        "X": features,
        "feature_cat": feature_cat,
        "feature_value": feature_value,
        "feature_onehot": feature_onehot,
        # expose the raw one-hot block + its column names
        "X_onehot": onehot_df,
        "onehot_out_names": onehot_cols,
    }

    # IDs & groups
    if cfg.id_cols and cfg.id_cols[0] in df.columns:
        bundle["ID"] = df[cfg.id_cols[0]]
    if cfg.group_cols:
        bundle["group_cols"] = cfg.group_cols
        bundle["group_primary"] = df[cfg.group_cols[0]]
        for gc in cfg.group_cols:
            bundle[f"group__{gc}"] = df[gc]

    # Targets (optional)
    if cfg.target_cls_col and cfg.target_cls_col in df.columns:
        y = df[cfg.target_cls_col]
        if y.dtype == "O" and cfg.positive_label is not None:
            bundle["y_cls"] = (y == cfg.positive_label).astype(int)
        elif pd.api.types.is_bool_dtype(y):
            bundle["y_cls"] = y.astype(int)
        else:
            bundle["y_cls"] = (pd.to_numeric(y, errors="coerce").fillna(0) > 0).astype(int)
        bundle["target_cls_col"] = cfg.target_cls_col

    if cfg.target_reg_col and cfg.target_reg_col in df.columns:
        bundle["y_reg"] = pd.to_numeric(df[cfg.target_reg_col], errors="coerce")
        bundle["target_reg_col"] = cfg.target_reg_col

    return bundle


def build_and_save_from_dfs(cfg: PreprocessConfig, df_train: pd.DataFrame, df_test: pd.DataFrame,
                            save_prefix: str = "data"):
    """
    Fit encoders on df_train, transform both splits, and save pkl bundles.
    """
    _ensure_dirs(cfg.encoder_info_dir, cfg.data_process_dir)

    data_train = feature_engineering_general_df(df_train, cfg, train=True)
    with open(os.path.join(cfg.data_process_dir, f"{save_prefix}_train.pkl"), "wb") as f:
        pickle.dump(data_train, f)

    data_test = feature_engineering_general_df(df_test, cfg, train=False)
    with open(os.path.join(cfg.data_process_dir, f"{save_prefix}_test.pkl"), "wb") as f:
        pickle.dump(data_test, f)

    # Persist split indices (optional but useful)
    try:
        idx_train = getattr(df_train, "index", pd.RangeIndex(len(df_train)))
        idx_test = getattr(df_test, "index", pd.RangeIndex(len(df_test)))
        np.savez(os.path.join(cfg.data_process_dir, f"{save_prefix}_split_indices.npz"),
                 train_index=np.array(idx_train), test_index=np.array(idx_test))
    except Exception:
        pass

    print("Categorical Feature count:", len(data_train["feature_cat"]))
    print("Numerical Feature count:", len(data_train["feature_value"]))
    print("One-hot Feature count:", len(data_train["feature_onehot"]))

    return data_train, data_test


def split_fit_transform_single_csv(cfg: PreprocessConfig, csv_path: str,
                                   test_size: float = 0.2, random_state: int = 42,
                                   stratify_col: str | None = None,
                                   save_prefix: str = "data"):
    """
    Read one CSV, split to train/test, fit on train, transform test, save bundles.
    """
    df = pd.read_csv(csv_path)
    stratify_vals = df[stratify_col] if stratify_col and stratify_col in df.columns else None
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=test_size, random_state=random_state, stratify=stratify_vals
    )
    df_train, df_test = df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    return build_and_save_from_dfs(cfg, df_train, df_test, save_prefix=save_prefix)

from pandas.api.types import is_numeric_dtype

def safe_minimal_bundle(df: pd.DataFrame, cfg) -> dict:
    """
    Ultra-robust preprocessing that avoids any Series-in-boolean evaluations.
    - Numeric: to_numeric -> median impute
    - Categorical: cast to str, normalize NA, fill, bounded get_dummies
    - No target/date detection; purely structural and safe
    Returns a bundle with keys compatible with your pipeline.
    """
    df = df.copy()

    # Normalize common missing markers if available
    try:
        df = standardize_missing_markers(df)
    except Exception:
        pass

    # Split by dtype (strictly)
    num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    cats_cols = [c for c in df.columns if c not in num_cols]

    # Numerics: coerce + median impute
    if num_cols:
        nums_df = df[num_cols].apply(pd.to_numeric, errors="coerce")
        med = nums_df.median(numeric_only=True)
        nums_df[num_cols] = nums_df[num_cols].fillna(med)
    else:
        nums_df = pd.DataFrame(index=df.index)

    # Categoricals: stringify + NA fill
    if cats_cols:
        cats_df = df[cats_cols].astype(str)
        try:
            cats_df = cats_df.replace(_std_null_marker_list(), np.nan)
        except Exception:
            pass
        cats_df = cats_df.fillna(getattr(cfg, "fillna_categorical", "-1"))
    else:
        cats_df = pd.DataFrame(index=df.index)

    # Bound one-hot to reasonable cardinality
    max_card = getattr(cfg, "max_onehot_cardinality", 80)
    small_cats = []
    if not cats_df.empty:
        card = {c: cats_df[c].nunique(dropna=False) for c in cats_df.columns}
        small_cats = [c for c in cats_df.columns if card.get(c, 0) <= max_card]

    if small_cats:
        onehot_df = pd.get_dummies(cats_df[small_cats], dtype=np.float32)
        onehot_cols = list(onehot_df.columns)
    else:
        onehot_df = pd.DataFrame(index=df.index)
        onehot_cols = []

    # Assemble features
    X = pd.concat([cats_df, nums_df, onehot_df], axis=1)

    bundle = {
        "X": X,
        "feature_cat": cats_cols,
        "feature_value": num_cols,
        "feature_onehot": onehot_cols,
    }

    # Best-effort passthroughs (do NOT rely on detectors)
    # ID guess: first column that looks like an id
    try:
        id_guess = next((c for c in df.columns if str(c).lower() in ("id",) or str(c).lower().endswith("id")), None)
        if id_guess is not None:
            bundle["ID"] = df[id_guess]
    except Exception:
        pass

    return bundle
def _count_csv_columns_on_disk(csv_path: str) -> int:
    # Robust header-only count (no full read)
    import csv
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        return len(header or [])

def preprocess_full_dataset(cfg: PreprocessConfig,
                            csv_path: str,
                            *, user_id: str,
                            filename: str = "data.csv",
                            fit_encoders: bool = True):
    """
    Preprocess entire dataset, write a single CSV under a temp user_outputs/, upload to Supabase.
    Fallback order:
      1) feature_engineering_general (CSV path)
      2) feature_engineering_general_df (DF path)
      3) _super_safe_minimal_bundle (truth-value-proof)
    Returns remote paths + simple quality metrics.
    """
    import tempfile
    from pathlib import Path as PathL
    from dataclasses import replace

    with tempfile.TemporaryDirectory() as temp_dir:
        user_dir = PathL(temp_dir) / "user_outputs"
        enc_dir  = user_dir / "encoders"
        proc_dir = user_dir / "processed"
        enc_dir.mkdir(parents=True, exist_ok=True)
        proc_dir.mkdir(parents=True, exist_ok=True)

        cfg_tmp = replace(cfg, encoder_info_dir=str(enc_dir), data_process_dir=str(proc_dir))

        # -------- Try normal CSV FE --------
        bundle = None
        try:
            bundle = feature_engineering_general(csv_path, cfg_tmp, train=fit_encoders)
        except Exception as e1:
            msg1 = str(e1).lower()
            logger.warning(f"feature_engineering_general failed ({msg1}); trying DF path")

            # -------- Try DF FE --------
            try:
                df = pd.read_csv(csv_path, low_memory=False)
                try:
                    df = standardize_missing_markers(df)
                except Exception:
                    pass
                bundle = feature_engineering_general_df(df, cfg_tmp, train=fit_encoders)
            except Exception as e2:
                msg2 = str(e2).lower()
                logger.warning(f"feature_engineering_general_df failed ({msg2}); using super-safe minimal path")

                # -------- Final fallback: minimal, truth-value-free --------
                if 'df' not in locals():
                    df = pd.read_csv(csv_path, low_memory=False)
                bundle = safe_minimal_bundle(df, cfg_tmp)
                # Note: no encoder pickle in this path

                # Materialize final output
        df_out = bundle["X"].copy()
        if "y_cls" in bundle:         df_out["__target_cls"] = bundle["y_cls"]
        if "y_reg" in bundle:         df_out["__target_reg"] = bundle["y_reg"]
        if "ID" in bundle:            df_out["__ID"] = bundle["ID"]
        if "group_primary" in bundle: df_out["__group_primary"] = bundle["group_primary"]

        rows_processed = int(df_out.shape[0])
        cols_processed = int(df_out.shape[1])  # ← used by endpoint check
        # Last-ditch guard – never allow 0 columns
        if df_out.shape[1] == 0:
            try:
                raw = pd.read_csv(csv_path, low_memory=False)
                num_cols = list(raw.select_dtypes(include=["number"]).columns)[:20]
                other_cols = [c for c in raw.columns if c not in num_cols][:10]
                keep = num_cols + other_cols
                if keep:
                    for c in other_cols:
                        raw[c] = raw[c].astype(str)
                    df_out = raw[keep]
                else:
                    # absolute minimum: a synthetic index column
                    df_out = pd.DataFrame({"__row_index": np.arange(rows_processed)})
            except Exception:
                df_out = pd.DataFrame({"__row_index": np.arange(rows_processed)})

        # Recompute after the guard
        rows_processed = int(df_out.shape[0])
        cols_processed = int(df_out.shape[1])
        null_rate_after_full = compute_null_rate(df_out)

        data_path = user_dir / filename
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(data_path, index=False)
        # authoritative counts from the file we uploaded
        cols_on_disk = _count_csv_columns_on_disk(str(data_path))
        rows_on_disk = rows_processed  # rows are already accurate for our df_out

        # If there is a discrepancy, trust the file we just wrote
        cols_processed = int(cols_on_disk)
        rows_processed = int(rows_on_disk)

        # Optional: if somehow only 0–1 columns survived, treat as failure to force fallback
        if cols_processed <= 1:
            raise RuntimeError(f"Post-write sanity: only {cols_processed} column(s) in processed CSV")
        # Upload processed dataset
        data_remote = upload_file_to_supabase(user_id, str(data_path), data_path.name)

        # Upload encoder only if it exists (skip in minimal path)
        enc_pkl = enc_dir / "one_hot_encoder.pkl"
        enc_remote = upload_file_to_supabase(user_id, str(enc_pkl), enc_pkl.name) if (fit_encoders and enc_pkl.exists()) else None

        return {
            "data_csv": data_remote,
            "encoder_pkl": enc_remote,
            "rows_processed": rows_processed,
            "cols_processed": cols_processed,
            "null_rate_after_full": null_rate_after_full,
        }
