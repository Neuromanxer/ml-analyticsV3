import pandas as pd
import numpy as np

def preprocess_data(df, RMV=[]):
    """
    Cleans and scales dataset with automatic handling of missing values 
    based on percentage thresholds
    """
    # Standardize column names
    df.columns = df.columns.str.replace(r'[^\w]', '_', regex=True)
    
    # Ensure ID column exists and is properly named
    if "id" in df.columns:
        df.rename(columns={"id": "ID"}, inplace=True)
    if "ID" not in df.columns:
        df.insert(0, "ID", range(1, len(df) + 1))  # Create unique ID column if missing
    
    # Initial features list
    FEATURES = [c for c in df.columns if c not in RMV]
    
    # Calculate percent missing for each column
    missing_percent = df[FEATURES].isnull().mean() * 100
    
    # Create lists based on missing percentages
    high_missing = missing_percent[missing_percent > 50].index.tolist()
    low_missing = missing_percent[missing_percent < 10].index.tolist()
    medium_missing = missing_percent[(missing_percent >= 10) & (missing_percent <= 50)].index.tolist()
    
    print(f"Columns with >50% missing: {high_missing}")
    print(f"Columns with 10-50% missing: {medium_missing}")
    print(f"Columns with <10% missing: {low_missing}")
    
    # Automatically drop columns with >50% missing unless they're explicitly kept
    additional_rmv = [col for col in high_missing if col not in RMV]
    if additional_rmv:
        print(f"Automatically dropping columns with >50% missing: {additional_rmv}")
        RMV.extend(additional_rmv)
    
    # For medium missing (10-50%), check if pattern is random 
    for col in medium_missing:
        # Check if missing values correlate with other columns
        missing_mask = df[col].isnull()
        other_cols = [c for c in FEATURES if c != col and c not in high_missing]
        
        if len(other_cols) > 0:
            # Simple pattern check: see if missing values are correlated with other columns
            numeric_cols = [c for c in other_cols if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1]
            
            if numeric_cols:
                # Check correlation between missingness and other numeric columns
                has_pattern = False
                for nc in numeric_cols[:3]:  # Limit to first 3 columns for efficiency
                    if abs(df[missing_mask][nc].mean() - df[~missing_mask][nc].mean()) > 0.1 * df[nc].std():
                        has_pattern = True
                        break
                
                if has_pattern:
                    print(f"Column {col} shows pattern in missing values - adding indicator column")
                    # Add indicator column for missingness
                    df[f"{col}_missing"] = df[col].isnull().astype(int)
                else:
                    print(f"Column {col} has random missing pattern - using imputation")
    
    # Update features list after potential removals
    FEATURES = [c for c in df.columns if c not in RMV]
    print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")
    
    # Separate features into categorical and numerical
    CATS = [c for c in FEATURES if df[c].dtype == "object"]
    NUMS = [c for c in FEATURES if c not in CATS]
    
    print("We LABEL ENCODE the CATEGORICAL FEATURES: ", end="")
    
    # Process all features
    for c in FEATURES:
        # LABEL ENCODE CATEGORICAL AND CONVERT TO INT32 CATEGORY
        if c in CATS:
            print(f"{c}, ", end="")
            # Handle missing values in categorical features
            if df[c].isnull().sum() > 0:
                df[c] = df[c].fillna("MISSING")
            
            df[c], _ = df[c].factorize()
            df[c] -= df[c].min()
            df[c] = df[c].astype("int32")
        
        # HANDLE NUMERICAL FEATURES
        else:
            # Handle missing values in numerical features based on percentage
            missing_pct = df[c].isnull().mean() * 100
            
            if missing_pct > 0:
                if missing_pct < 10:
                    # For low missing percentages, use median imputation
                    df[c] = df[c].fillna(df[c].median())
                elif missing_pct <= 50:
                    # For medium missing, more careful imputation if we're keeping the column
                    if c in medium_missing:
                        # Could use more sophisticated imputation here
                        df[c] = df[c].fillna(df[c].median())
            
            # Reduce memory usage
            if df[c].dtype == "float64":
                df[c] = df[c].astype("float32")
            elif df[c].dtype == "int64":
                df[c] = df[c].astype("int32")
    
    print("\nPreprocessing complete.")
    return df, CATS, NUMS