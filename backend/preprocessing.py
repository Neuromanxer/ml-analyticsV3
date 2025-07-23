import pandas as pd
import numpy as np

def preprocess_data(df, RMV=[]):
    """
    Cleans and scales dataset with automatic handling of missing values 
    and encodes all object/categorical fields for ML compatibility.
    """
    # ✅ Standardize column names
    df.columns = df.columns.str.replace(r'[^\w]', '_', regex=True)
    
    # ✅ Ensure ID column exists and is properly named
    if "id" in df.columns:
        df.rename(columns={"id": "ID"}, inplace=True)
    if "ID" not in df.columns:
        df.insert(0, "ID", range(1, len(df) + 1))
    
    # ✅ Initial features list
    FEATURES = [c for c in df.columns if c not in RMV]
    
    # ✅ Calculate missing %
    missing_percent = df[FEATURES].isnull().mean() * 100
    high_missing = missing_percent[missing_percent > 50].index.tolist()
    low_missing = missing_percent[missing_percent < 10].index.tolist()
    medium_missing = missing_percent[(missing_percent >= 10) & (missing_percent <= 50)].index.tolist()

    print(f"Columns with >50% missing: {high_missing}")
    print(f"Columns with 10-50% missing: {medium_missing}")
    print(f"Columns with <10% missing: {low_missing}")

    # ✅ Drop high-missing columns unless protected
    additional_rmv = [col for col in high_missing if col not in RMV]
    if additional_rmv:
        print(f"Automatically dropping columns with >50% missing: {additional_rmv}")
        RMV.extend(additional_rmv)
    
    # ✅ Handle medium-missing columns with pattern detection
    for col in medium_missing:
        missing_mask = df[col].isnull()
        other_cols = [c for c in FEATURES if c != col and c not in high_missing]
        numeric_cols = [c for c in other_cols if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1]

        if numeric_cols:
            has_pattern = False
            for nc in numeric_cols[:3]:
                if abs(df[missing_mask][nc].mean() - df[~missing_mask][nc].mean()) > 0.1 * df[nc].std():
                    has_pattern = True
                    break
            
            if has_pattern:
                print(f"Column {col} shows pattern in missing values - adding indicator column")
                df[f"{col}_missing"] = df[col].isnull().astype(int)
            else:
                print(f"Column {col} has random missing pattern - using imputation")

    # ✅ Update features after removing dropped columns
    FEATURES = [c for c in df.columns if c not in RMV]
    print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")
    
    # ✅ Detect categorical (object or category) and numeric
    CATS = [c for c in FEATURES if df[c].dtype in ["object", "category"]]
    NUMS = [c for c in FEATURES if c not in CATS]

    print("We LABEL ENCODE the CATEGORICAL FEATURES: ", end="")

    # ✅ Process all features
    for c in FEATURES:
        if c in CATS:
            print(f"{c}, ", end="")
            if df[c].isnull().sum() > 0:
                df[c] = df[c].fillna("MISSING")  # ✅ fill missing categorical with "MISSING"
            df[c], _ = df[c].astype("str").factorize()
            df[c] = df[c] - df[c].min()
            df[c] = df[c].astype("int32")
        else:
            missing_pct = df[c].isnull().mean() * 100
            if missing_pct > 0:
                df[c] = df[c].fillna(df[c].median())  # ✅ median imputation for numerics
            if df[c].dtype == "float64":
                df[c] = df[c].astype("float32")
            elif df[c].dtype == "int64":
                df[c] = df[c].astype("int32")

    print("\nPreprocessing complete.")

    # ✅ Final safety check: only allow int, float, or bool
    invalid_cols = df[FEATURES].select_dtypes(exclude=["int", "float", "bool"]).columns
    if not invalid_cols.empty:
        raise ValueError(f"Columns with unsupported dtypes after preprocessing: {list(invalid_cols)}")

    return df, CATS, NUMS
