import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, log_loss
import lightgbm as lgb
import xgboost as xgb
from lightgbm import early_stopping, log_evaluation
import catboost as cb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .preprocessing import preprocess_data

from queue import Queue
from threading import Thread
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import root_mean_squared_error
# XGBoost Parameters for Classification
xgb_params_c = {
    "objective": "binary:logistic",  # For binary classification
    "n_estimators": 2000,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 10,
    "random_state": 42,
    "use_label_encoder": False
}

# LightGBM Parameters for Classification
lgb_params_c = {
    "objective": "binary",  # For binary classification
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": -1,  # No limit on tree depth
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": 500 
}

# CatBoost Parameters for Classification
cat_params_c = {
    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "bagging_temperature": 1.0,
    "border_count": 254,
    "loss_function": "Logloss",  # For binary classification
    "random_state": 42,
    "verbose": 500
}
class ModelClassifyingTrainer:
    def __init__(self, data: pd.DataFrame, n_splits: int = 5):
        """
        data: a DataFrame containing ALL training rows, including the target column.
        n_splits: number of CV folds.
        """
        self.raw = data.copy()
        self.n_splits = n_splits

    def train_model(self, params: dict, target: str, title: str):
        """
        Runs Stratified K-Fold CV on the passed data.

        Returns:
          - models: list of fitted models, one per fold
          - oof_preds: numpy array of out-of-fold predictions (length = n_samples)
        """
        df = self.raw
        X = df.drop(columns=['ID', target], errors='ignore')
        y = df[target]

        oof_preds = np.zeros(len(X))
        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        models = []
        for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            # Instantiate the right classifier with common structure
            if title.startswith('LightGBM'):
                model = LGBMClassifier(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric='binary_logloss',
                    verbose=500
                )

            elif title.startswith('CatBoost'):
                model = CatBoostClassifier(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=500,
                    use_best_model=False  # Prevents error if you’re not using early stopping
                )

            else:  # XGBoost
                model = XGBClassifier(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric='logloss',
                    verbose=500
                )


            # fit on fold
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
            models.append(model)

            # store OOF predictions
            oof_preds[val_idx] = model.predict(X_val)

        return models, oof_preds

