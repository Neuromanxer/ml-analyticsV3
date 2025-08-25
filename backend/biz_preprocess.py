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

    verbose: bool = True

# -----------------------------
# Utilities
# -----------------------------
def _ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)
def _as_list(x):
    if x is None: return []
    if isinstance(x, (list, tuple)): return list(x)
    if isinstance(x, pd.Series):     return x.tolist()
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
def _first_valid_colname(x, columns):
    """Return a single valid column name or None."""
    if isinstance(x, (list, tuple, pd.Series, np.ndarray)):
        for v in x:
            if _is_valid_colname(v, columns):
                return v
        return None
    return x if _is_valid_colname(x, columns) else None


def _infer_types(df: pd.DataFrame, cfg: PreprocessConfig) -> Dict[str, str]:
    """
    Return {col: type_str} among:
    Categorical, Numerical, Date, Text, ID, Ignore, Group, Target:cls, Target:reg
    """
    types: Dict[str, str] = {}

    # Seed fixed roles (GUARDED)
    for c in _as_list(cfg.id_cols):
        if _is_valid_colname(c, df.columns): types[c] = "ID"

    for c in _as_list(cfg.group_cols):
        if _is_valid_colname(c, df.columns): types[c] = "Group"

    if _is_valid_colname(cfg.target_cls_col, df.columns):
        types[cfg.target_cls_col] = "Target:cls"

    if _is_valid_colname(cfg.target_reg_col, df.columns):
        types[cfg.target_reg_col] = "Target:reg"

    for c in _as_list(cfg.ignore_cols):
        if _is_valid_colname(c, df.columns): types[c] = "Ignore"

    # Explicit user nudges
    for c in _as_list(cfg.treat_as_categorical):
        if _is_valid_colname(c, df.columns): types[c] = "Categorical"
    for c in _as_list(cfg.treat_as_numerical):
        if _is_valid_colname(c, df.columns): types[c] = "Numerical"
    for c in _as_list(cfg.text_cols):
        if _is_valid_colname(c, df.columns): types[c] = "Text"

    # Explicit dates (if provided)
    explicit_dates = set(_as_list(cfg.date_cols))
    for c in explicit_dates:
        if _is_valid_colname(c, df.columns): types[c] = "Date"


    # Helper: detect dates
    def looks_like_date(series: pd.Series, name: str) -> bool:
        if not cfg.auto_detect_dates:
            return False
        name_hint = any(h in str(name).lower() for h in cfg.date_name_hints)
        sample = series.dropna().astype(str).head(cfg.date_detect_sample)
        if sample.empty:
            return False
        try:
            parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
            ratio = parsed.notna().mean()
            thresh = cfg.date_detect_success_ratio - (0.15 if name_hint else 0.0)
            return ratio >= max(0.2, thresh)
        except Exception:
            return False

    # Infer remaining columns
    for col in df.columns:
        if col in types:
            continue
        s = df[col]
        # Date?
        if pd.api.types.is_datetime64_any_dtype(s) or (col not in explicit_dates and looks_like_date(s, col)):
            types[col] = "Date"
            continue
        # Numeric vs cat vs text
        if pd.api.types.is_numeric_dtype(s):
            types[col] = "Numerical"
        else:
            uniq = s.nunique(dropna=True)
            ratio = uniq / max(1, len(s))
            if uniq == 0:
                types[col] = "Categorical"
            elif uniq <= 30 or ratio <= 0.05:
                types[col] = "Categorical"
            else:
                types[col] = "Text"
    return types

from typing import Tuple

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

    return cat_cols, num_cols, date_cols, text_cols, id_cols, ignore, groups


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

def _add_numeric_bins(df: pd.DataFrame, num_cols: List[str], cfg: PreprocessConfig) -> Tuple[pd.DataFrame, List[str]]:
    """Quantile-bin numerics into categorical *_trans2cat columns, with optional auto 'keep continuous' rules."""
    added_cat_cols: List[str] = []
    if not cfg.numeric_to_cat or not num_cols:
        return df, added_cat_cols

    # Build keep set according to config
    keep_set = set()
    if cfg.numeric_keep_mode in ("list", "auto+list"):
        keep_set |= set(cfg.numeric_keep_continuous or [])
    if cfg.numeric_keep_mode in ("auto", "auto+list"):
        for col in num_cols:
            s = df[col]
            # heuristics for "continuous-like"
            uniq = s.nunique(dropna=True)
            uniq_ratio = uniq / max(1, len(s))
            name_hint = any(h in str(col).lower() for h in cfg.numeric_keep_name_hints)
            continuous_like = (
                (uniq >= cfg.numeric_keep_min_unique) or
                (uniq_ratio >= cfg.numeric_keep_min_unique_ratio) or
                name_hint
            )
            if continuous_like:
                keep_set.add(col)

    for col in num_cols:
        if col in keep_set:
            continue
        newc = f"{col}_trans2cat"
        s = df[col]
        if s.notna().sum() == 0 or s.nunique(dropna=True) <= 1:
            df[newc] = s.astype(str)
        else:
            # Prefer quantile bins; fall back to uniform bins if needed
            try:
                df[newc] = pd.qcut(s, q=cfg.numeric_bins, duplicates="drop").astype(str)
            except Exception:
                try:
                    df[newc] = pd.cut(s, bins=min(cfg.numeric_bins, max(2, s.nunique())), duplicates="drop").astype(str)
                except Exception:
                    df[newc] = s.astype(str)
        added_cat_cols.append(newc)

    return df, added_cat_cols


def _fit_or_load_onehot(X: pd.DataFrame,
                        cfg: PreprocessConfig,
                        ohe_features: list[str],
                        fit: bool):
    """
    Fit or load a OneHotEncoder on the given categorical features.
    No boolean evaluation of Series/DataFrames anywhere.
    """
    enc_path = os.path.join(cfg.encoder_info_dir, getattr(cfg, "one_hot_filename", "one_hot_encoder.pkl"))

    # If there are no OHE features, return a stub
    if not ohe_features or len(ohe_features) == 0:
        return {
            "encoder": None,
            "features": [],
            "out_names": [],
            "path": enc_path,
        }

    # Work on a copy with safe dtype/NA handling
    Z = X[ohe_features].copy()
    for c in ohe_features:
        # ensure string dtype and fill NAs consistently
        Z[c] = Z[c].astype(str).fillna(getattr(cfg, "fillna_categorical", "-1"))

    if fit:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)
        ohe.fit(Z)
        out_names = list(ohe.get_feature_names_out(ohe_features))
        os.makedirs(os.path.dirname(enc_path), exist_ok=True)
        with open(enc_path, "wb") as f:
            pickle.dump({"encoder": ohe, "features": ohe_features, "out_names": out_names}, f)
        return {"encoder": ohe, "features": ohe_features, "out_names": out_names, "path": enc_path}

    # Load prefit
    if not os.path.exists(enc_path):
        raise RuntimeError(f"One-hot encoder file not found at {enc_path}")
    with open(enc_path, "rb") as f:
        payload = pickle.load(f)

    # Align columns (any missing columns are filled with cfg.fillna_categorical)
    miss = [c for c in payload["features"] if c not in Z.columns]
    for c in miss:
        Z[c] = getattr(cfg, "fillna_categorical", "-1")

    # Reorder to match trained features
    Z = Z[payload["features"]].astype(str).fillna(getattr(cfg, "fillna_categorical", "-1"))

    return {"encoder": payload["encoder"],
            "features": payload["features"],
            "out_names": payload["out_names"],
            "path": enc_path}

def _transform_onehot(X: pd.DataFrame, encoder_info: dict) -> tuple[pd.DataFrame, list[str]]:
    """
    Transform using the fitted encoder. Returns (onehot_df, out_cols).
    """
    ohe = encoder_info.get("encoder")
    feats = encoder_info.get("features", [])
    out_names = encoder_info.get("out_names", [])

    if ohe is None or len(feats) == 0:
        # nothing to add
        return pd.DataFrame(index=X.index), []

    Z = X[feats].copy()
    for c in feats:
        Z[c] = Z[c].astype(str).fillna("-1")

    arr = ohe.transform(Z)  # shape: (n_samples, n_out)
    onehot_df = pd.DataFrame(arr, index=X.index, columns=out_names)
    return onehot_df, out_names

# -----------------------------
# Core: feature_engineering (generalized)
# -----------------------------
def feature_engineering_general(data_file: str, cfg: PreprocessConfig, train: bool):
    _ensure_dirs(cfg.encoder_info_dir, cfg.data_process_dir)
    df = pd.read_csv(data_file)
    cat_cols, num_cols, date_cols, text_cols, id_cols, ignore_cols, group_cols = _build_type_lists(df, cfg)

    # 1) Dates → (optional) add parts, then we can treat month/dow as categorical, year as numeric
    added_cat_from_dates: List[str] = []
    added_num_from_dates: List[str] = []
    if cfg.add_date_parts and len(date_cols) > 0:
        df, added_cat_from_dates, added_num_from_dates = _add_date_parts(df, date_cols)

    # 2) Numeric → categorical bins (trans2cat)
    df, added_cat_from_bins = _add_numeric_bins(df, num_cols, cfg)

    # Collect feature lists
    feature_cat = cat_cols + added_cat_from_dates + added_cat_from_bins
    feature_value = num_cols + added_num_from_dates

    # Basic NA + dtype hygiene for categoricals
    cats_df = df[feature_cat].copy() if feature_cat else pd.DataFrame(index=df.index)
    for c in cats_df.columns:
        cats_df[c] = cats_df[c].astype(str).fillna(cfg.fillna_categorical)

    nums_df = df[feature_value].copy() if feature_value else pd.DataFrame(index=df.index)

    # 3) One-hot: only for “reasonable” cardinality to avoid explosion
    card = {c: cats_df[c].nunique(dropna=False) for c in cats_df.columns}
    ohe_features = [c for c in cats_df.columns if card[c] <= cfg.max_onehot_cardinality]

    encoder_info = _fit_or_load_onehot(
        pd.concat([cats_df, nums_df], axis=1),
        cfg,
        ohe_features=ohe_features,
        fit=train,
    )

    onehot_df, onehot_cols = _transform_onehot(pd.concat([cats_df, nums_df], axis=1), encoder_info)
    # After computing onehot_df / onehot_cols and building `features`
    features = pd.concat([cats_df, nums_df, onehot_df], axis=1)
    feature_onehot = onehot_cols

    bundle: Dict[str, pd.Series | pd.DataFrame | List[str]] = {
        "X": features,
        "feature_cat": feature_cat,
        "feature_value": feature_value,
        "feature_onehot": feature_onehot,
        # expose the raw one-hot block + its column names
        "X_onehot": onehot_df,
        "onehot_out_names": onehot_cols,
    }
    # IDs
    if len(id_cols) > 0 and id_cols[0] in df.columns:
        bundle["ID"] = df[id_cols[0]]
    # Groups (business-segmentation analog to race_group)
    if len(group_cols) > 0:
        # store the first as a convenience (mirrors your 'race_group') and all as 'group_cols'
        bundle["group_primary"] = df[group_cols[0]]
        for gc in group_cols:
            bundle[f"group__{gc}"] = df[gc]
        bundle["group_cols"] = group_cols

   # Targets
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
        cols_processed = int(df_out.shape[1])
        null_rate_after_full = compute_null_rate(df_out)

        data_path = user_dir / filename
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(data_path, index=False)

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



# -----------------------------
# Example usage (adjust paths/targets as needed)
# -----------------------------
if __name__ == "__main__":
    cfg = PreprocessConfig(
        data_dir="./data/",
        encoder_info_dir="./artifacts/encoders/",
        data_process_dir="./artifacts/processed/",

        # Typical business roles
        id_cols=["ID"],                         # or ["customer_id"]
        target_cls_col=None,                    # e.g., "churned"
        positive_label=None,                    # e.g., "yes"; or None to auto
        target_reg_col=None,                    # e.g., "revenue_30d"
        group_cols=["segment"],                 # e.g., ["segment","region"]

        # If you know dates/texts up front (optional)
        date_cols=["created_at", "order_date"], # will also be inferred if formatted like dates
        text_cols=[],                           # keep empty unless you plan to vectorize text
        ignore_cols=[],

        # Policy knobs
        numeric_to_cat=True,
        numeric_bins=10,
        numeric_keep_continuous=["age", "tenure_days"],  # example: keep these continuous
        max_onehot_cardinality=80,
        add_date_parts=True,
        verbose=True,
    )

    build_and_save_train_test(cfg, train_csv="train.csv", test_csv="test.csv")
#for train test paths = build_and_save_train_test(cfg, train_csv="train.csv", test_csv="test.csv", user_id=user_id)
#for full data paths = preprocess_full_dataset(cfg, csv_path=os.path.join(cfg.data_dir, "train.csv"), user_id=user_id, filename="data.csv", fit_encoders=True)

