import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb

# Model parameters
cat_params = {"iterations": 2000, "learning_rate": 0.05, "depth": 6, "loss_function": "Logloss", "random_state": 42, "verbose": 0}
lgb_params = {"objective": "binary", "n_estimators": 500, "learning_rate": 0.05, "random_state": 42}

def train_best_anomaly_detection(df, target_column, drop_columns=[], n_splits=5):
    """
    Trains multiple anomaly detection models using K-Fold cross-validation and selects the best one.
    
    Parameters:
        df (DataFrame): Input dataset.
        target_column (str): Name of the target variable.
        drop_columns (list): Additional columns to drop before training.
        n_splits (int): Number of folds for cross-validation.
    
    Returns:
        best_model: The best-performing model.
        results (dict): Cross-validation scores for each model.
    """
    X = df.drop(columns=[target_column] + drop_columns + ['ID'])
    y = df[target_column]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    models = {
        "IsolationForest": IsolationForest(contamination=0.1, random_state=42),
        "LocalOutlierFactor": LocalOutlierFactor(contamination=0.1, novelty=True),
        "OneClassSVM": OneClassSVM(nu=0.1),
        "XGBoost": XGBClassifier(objective="binary:logistic", n_estimators=500, random_state=42),
        "CatBoost": CatBoostClassifier(**cat_params),
        "LightGBM": lgb.LGBMClassifier(**lgb_params)
    }

    best_model = None
    best_score = float("-inf")
    results = {}

    for name, model in models.items():
        print(f"Training {name} with {n_splits}-Fold Cross-Validation...")
        
        f1_scores, precision_scores, recall_scores, roc_auc_scores = [], [], [], []

        for train_idx, val_idx in kf.split(X, y):
            X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]

            if name in ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"]:
                model.fit(X_train)
                y_pred = model.predict(X_valid)
                y_pred = np.where(y_pred == -1, 1, 0)  # Convert -1 (outlier) to 1, 1 (inlier) to 0
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_valid)

            # Compute metrics
            precision = precision_score(y_valid, y_pred)
            recall = recall_score(y_valid, y_pred)
            f1 = f1_score(y_valid, y_pred)
            roc_auc = roc_auc_score(y_valid, y_pred)

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            roc_auc_scores.append(roc_auc)

        avg_f1 = np.mean(f1_scores)
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        avg_roc_auc = np.mean(roc_auc_scores)

        results[name] = {
            "Precision": avg_precision,
            "Recall": avg_recall,
            "F1 Score": avg_f1,
            "ROC-AUC": avg_roc_auc
        }

        # Update best model
        if avg_f1 > best_score:
            best_score = avg_f1
            best_model = model

        print(f"{name} - F1 Score: {avg_f1:.4f} | Precision: {avg_precision:.4f} | Recall: {avg_recall:.4f} | ROC-AUC: {avg_roc_auc:.4f}")

    # Save the best model
    joblib.dump(best_model, "best_anomaly_detection_model.pkl")
    print(f"Best Model: {best_model.__class__.__name__} (F1 Score: {best_score:.4f}) saved as 'best_anomaly_detection_model.pkl'")

    return best_model, results
