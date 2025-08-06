import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def preprocess_data(df, RMV=[]):
    print("🚀 ENTERED preprocess_data!")  
    """
    Cleans and scales dataset with automatic handling of missing values 
    and encodes all object/categorical fields for ML compatibility.
    """
    # ── 1) Standardize and ensure ID ──
    df = df.copy()
    df.columns = df.columns.str.replace(r'[^\w]', '_', regex=True)
    if "id" in df.columns:
        df.rename(columns={"id": "ID"}, inplace=True)
    if "ID" not in df.columns:
        df.insert(0, "ID", range(1, len(df) + 1))
    
    # ── 2) Compute missing‐value stats ──
    FEATURES = [c for c in df.columns if c not in RMV]
    missing_pct = df[FEATURES].isnull().mean() * 100
    high_missing   = missing_pct[missing_pct  > 50].index.tolist()
    medium_missing = missing_pct[(missing_pct >= 10) & (missing_pct <= 50)].index.tolist()
    low_missing    = missing_pct[missing_pct  < 10].index.tolist()

    # Drop high‐missing
    to_drop = [c for c in high_missing if c not in RMV]
    if to_drop:
        RMV.extend(to_drop)
        FEATURES = [c for c in FEATURES if c not in to_drop]

    # ── 3) Handle medium‐missing: impute or flag ──
    for col in medium_missing:
        if col not in FEATURES:
            continue
        # check if missingness correlates with numeric patterns
        mask = df[col].isnull()
        numeric_cols = [
            c for c in FEATURES 
            if c != col 
            and pd.api.types.is_numeric_dtype(df[c]) 
            and df[c].nunique() > 1
        ]
        pattern = False
        for nc in numeric_cols[:3]:
            if abs(df.loc[mask, nc].mean() - df.loc[~mask, nc].mean()) \
               > 0.1 * df[nc].std():
                pattern = True
                break

        if pattern:
            # add indicator and then impute
            df[f"{col}_missing"] = mask.astype(int)
            df[col] = df[col].fillna(df[col].median())
        else:
            # straight median imputation
            df[col] = df[col].fillna(df[col].median())

    # ── 4) Recompute FEATURES after dropping ──
    FEATURES = [c for c in df.columns if c not in RMV]

    # ── 5) Separate categorical vs. numeric ──
    CATS = [c for c in FEATURES if df[c].dtype in ["object", "category"]]
    NUMS = [c for c in FEATURES if c not in CATS]

    # ── 6) Global imputation & encoding ──
    # 6a) Numeric: fill any remaining NaNs (should be only low_missing) with median
    num_imputer = SimpleImputer(strategy="median")
    df[NUMS] = num_imputer.fit_transform(df[NUMS])
    
    # 6b) Categorical: fill NaN→"MISSING", then factorize
    for c in CATS:
        df[c] = df[c].fillna("MISSING").astype(str)
        df[c], _ = df[c].factorize()
        df[c] = (df[c] - df[c].min()).astype("int32")

    # ── 7) Final dtype cleanup ──
    for c in NUMS:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
        elif df[c].dtype == "int64":
            df[c] = df[c].astype("int32")

    # Safety check
    leftover = df[FEATURES].isnull().any()
    if leftover.any():
        cols = list(leftover[leftover].index)
        raise ValueError(f"Still found NaNs in columns: {cols}")

    return df, CATS, NUMS
