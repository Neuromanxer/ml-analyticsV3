import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Tuple
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

logger = logging.getLogger(__name__)

class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        high_missing_thresh: float = 0.5,
        med_missing_thresh: Tuple[float, float] = (0.1, 0.5),
        verbose: bool = False
    ):
        self.high_missing_thresh = high_missing_thresh
        self.med_missing_thresh = med_missing_thresh
        self.verbose = verbose

        # internal state
        self.to_drop_: List[str] = []
        self.med_flag_: List[str] = []
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.num_imputer_: Optional[SimpleImputer] = None
        self.cat_imputer_: Optional[SimpleImputer] = None
        self.cat_encoder_: Optional[OrdinalEncoder] = None

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        # standardize names & ensure ID
        df.columns = df.columns.str.replace(r'[^\w]', '_', regex=True)
        if 'id' in df.columns:
            df.rename(columns={'id': 'ID'}, inplace=True)
        if 'ID' not in df.columns:
            df.insert(0, 'ID', range(1, len(df) + 1))

        # compute missing fractions
        miss_frac = df.isna().mean()
        self.to_drop_ = list(miss_frac[miss_frac > self.high_missing_thresh].index)
        low, high = self.med_missing_thresh
        med_mask = miss_frac.between(low, high, inclusive='both')
        med_cols = list(miss_frac[med_mask].index)

        tmp = df.drop(columns=self.to_drop_, errors='ignore')
        numeric = tmp.select_dtypes(include=[np.number]).columns.tolist()
        for col in med_cols:
            if col not in numeric:
                continue
            mask = tmp[col].isna()
            peers = [c for c in numeric if c != col][:3]
            for p in peers:
                if abs(tmp.loc[mask, p].mean() - tmp.loc[~mask, p].mean()) > 0.1 * tmp[p].std():
                    self.med_flag_.append(col)
                    break

        # record final cols
        self.numeric_cols_ = [c for c in tmp.columns if pd.api.types.is_numeric_dtype(tmp[c])]
        self.categorical_cols_ = [c for c in tmp.columns if c not in self.numeric_cols_]

        if self.verbose:
            logger.info(f"Dropping: {self.to_drop_}")
            logger.info(f"Flagging missing: {self.med_flag_}")
            logger.info(f"Numeric cols: {self.numeric_cols_}")
            logger.info(f"Categorical cols: {self.categorical_cols_}")

        # fit imputers
        self.num_imputer_ = SimpleImputer(strategy='median')
        self.num_imputer_.fit(tmp[self.numeric_cols_])
        self.cat_imputer_ = SimpleImputer(strategy='constant', fill_value='MISSING')
        self.cat_imputer_.fit(tmp[self.categorical_cols_])
        cat_filled = pd.DataFrame(
            self.cat_imputer_.transform(tmp[self.categorical_cols_]),
            columns=self.categorical_cols_
        )
        self.cat_encoder_ = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.cat_encoder_.fit(cat_filled)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df.columns = df.columns.str.replace(r'[^\w]', '_', regex=True)
        if 'id' in df.columns:
            df.rename(columns={'id': 'ID'}, inplace=True)
        if 'ID' not in df.columns:
            df.insert(0, 'ID', range(1, len(df) + 1))

        # drop high missing
        df.drop(columns=self.to_drop_, errors='ignore', inplace=True)
        # flag medium-missing
        for col in self.med_flag_:
            df[f"{col}_was_missing"] = df[col].isna().astype(int)

        # impute numeric
        num_cols = [c for c in self.numeric_cols_ if c in df.columns]
        df[num_cols] = self.num_imputer_.transform(df[num_cols])

        # impute and encode categorical
        cat_cols = [c for c in self.categorical_cols_ if c in df.columns]
        cat_filled = pd.DataFrame(
            self.cat_imputer_.transform(df[cat_cols]),
            columns=cat_cols, index=df.index
        )
        df[cat_cols] = self.cat_encoder_.transform(cat_filled).astype(int)

        # cast types
        for c in num_cols:
            if df[c].dtype == 'float64':
                df[c] = df[c].astype('float32')
            elif df[c].dtype == 'int64':
                df[c] = df[c].astype('int32')

        # final NaN check
        if df.isna().any().any():
            bad = df.columns[df.isna().any()].tolist()
            raise ValueError(f"NaNs remain in columns: {bad}")
        return df

# Wrapper function

def preprocess_data(df: pd.DataFrame, RMV: Optional[List[str]] = None, verbose: bool = False):
    """
    One-stop data cleaning, imputation, and encoding.
    Returns: (clean_df, categorical_columns, numeric_columns)
    """
    if RMV is None:
        RMV = []
    pre = DataPreprocessor(verbose=verbose)
    clean_df = pre.fit_transform(df)
    return clean_df, pre.categorical_cols_, pre.numeric_cols_
