import os
import time
import uuid
import json
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path as PathL
from celery import Celery
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.base import clone
import traceback
import json
import numpy as np
import io
import sys
import shap
from typing import Union, List, Any, Dict
import tempfile
# -- replace these imports with your actual module paths --
# service.py
from typing import Optional
import subprocess
from scipy.stats import (
    ttest_rel,
    wilcoxon,
    fisher_exact,
    sem,
    t as student_t
)
from statsmodels.stats.proportion import proportions_ztest

import joblib
from datetime import datetime
from pathlib import Path as PathL
from celery import Celery
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# -- replace these imports with your actual module paths --

from queue import Queue
from threading import Thread

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import logging
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report



from .anomaly_detection import train_best_anomaly_detection
from .preprocessing import preprocess_data
from .classification import ModelClassifyingTrainer, lgb_params_c, cat_params_c, xgb_params_c
from .auth import master_db_cm
from .storage import upload_file_to_supabase, download_file_from_supabase, handle_file_upload, download_file_from_supabase, list_user_files, delete_file_from_supabase, get_file_url
from .target import generate_customer_summary
from .auth import _append_limited_metadata, _append_metadata, _load_metadata, _save_metadata, _get_meta_path
from .regression import ModelTrainer, lgb_params, cat_params, xgb_params, DataPreprocessor, train_regression_models, generate_visualizations_improved
from .preprocessing import preprocess_data
from .classification import ModelClassifyingTrainer, lgb_params_c, cat_params_c, xgb_params_c, multiclass_brier, reliability_bins, decision_curve_binary
from .classification import plot_reliability_diagram, plot_decision_curve, plot_segment_heatmap, plot_segment_heatmap, plot_kpi_abstentions, plot_trend
from .classification import compute_reliability_bins_multiclass, ece_from_bins, decision_curve_binary, compute_segments_table
from .feature_importance import safe_generate_feature_importance
from .clustering import run_kmeans, find_optimal_k, label_clusters_general
from .timeSeries import ScenarioManager, ARIMAModel, ExponentialSmoothingModel, LSTMModel, RandomForestModel, generate_scenario_visualizations, SARIMAModel
from .regression import (
 get_confidence_interval,
 get_percentile_summary,
 get_risk_shift_summary,
 get_class_distribution_change
)
from .counterfactual import clean_data_for_json, safe_float_conversion, compute_summary_metrics


# from counterfactual import clean_data_for_json, safe_float_conversion, compute_summary_metrics
# from timeSeries import ScenarioManager, ARIMAModel, ExponentialSmoothingModel, LSTMModel, RandomForestModel, generate_scenario_visualizations, SARIMAModel
# from anomaly_detection import train_best_anomaly_detection
# from preprocessing import preprocess_data
# from classification import to_b64, plot_reliability_diagram, plot_decision_curve, plot_segment_heatmap
# from classification import ModelClassifyingTrainer, lgb_params_c, cat_params_c, xgb_params_c, multiclass_brier, reliability_bins, decision_curve_binary
# from auth import master_db_cm
# from storage import upload_file_to_supabase, download_file_from_supabase, handle_file_upload, download_file_from_supabase, list_user_files, delete_file_from_supabase, get_file_url
# from target import generate_customer_summary
# from auth import _append_limited_metadata, _append_metadata, _load_metadata, _save_metadata, _get_meta_path
# from regression import ModelTrainer, lgb_params, cat_params, xgb_params, DataPreprocessor, train_regression_models, generate_visualizations_improved
# from preprocessing import preprocess_data
# from feature_importance import safe_generate_feature_importance
# from clustering import run_kmeans, find_optimal_k, label_clusters_general
# from regression import (
# get_confidence_interval,
# get_percentile_summary,
# get_risk_shift_summary,
# get_class_distribution_change
# )
# from classification import plot_reliability_diagram, plot_decision_curve, plot_segment_heatmap, plot_segment_heatmap, plot_kpi_abstentions, plot_trend
# from classification import compute_reliability_bins_multiclass, ece_from_bins, decision_curve_binary, compute_segments_table
# schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

class DeliveryConfig(BaseModel):
    mode: Literal["webhook","file","none"] = "none"
    endpoint: Optional[str] = None
    auth: Optional[str] = None
    path: Optional[str] = None  # for file mode

class ChannelConfig(BaseModel):
    name: str
    daily_cap: int = 0
    cooldown_days: int = 7
    unit_cost: Dict[str, float] = Field(default_factory=lambda: {"contact": 0.0, "ops": 0.0})
    provider: Optional[str] = None
    delivery: DeliveryConfig = DeliveryConfig()
    quiet_hours: Optional[List[str]] = None

class PlanOptions(BaseModel):
    user_tz: str = "UTC"
    horizons: List[int] = [7, 14, 30]
    risk_preset: Literal["conservative","balanced","aggressive"] = "balanced"
    alpha: float = 0.10
    coverage: float = 0.90
    ece: float = 0.03
    threshold: float = 0.71

    # Decision Card (v2) extras
    goal_net: Optional[float] = None
    goal_window_days: Optional[int] = None
    budget_cap: Optional[float] = None
    run_window: Optional[Dict[str, Any]] = None   # {days:[], start:"HH:MM", end:"HH:MM", tz:"..."}
    chosen_horizon: Optional[str] = None
    top_drivers: Optional[List[str]] = None
    binding_constraints: Optional[List[str]] = None

class PlanConfig(BaseModel):
    options: PlanOptions
    actions: List[ChannelConfig] = []
    plan_type: Literal["Balanced","High-Precision","Budget-Limited"] = "Balanced"
    # Optional handle to reuse previous derive
    plan_id: Optional[str] = None

    # ----------------------------
    # Imports (kept local as in your style)
    # ----------------------------
    import json
    import tempfile
    import os
    import uuid
    from datetime import datetime
    from pathlib import Path as PathL
    from collections import Counter

    import numpy as np
    import pandas as pd
    import joblib

    from sklearn.model_selection import train_test_split
    from sklearn.base import clone
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix, log_loss
    )
def ensure_json_serializable(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='ignore')
    elif isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    return obj


    # NOTE: The following helpers/classes are assumed to exist elsewhere in your codebase:
    # - download_file_from_supabase(path) -> bytes
    # - upload_file_to_supabase(user_id, file_path, filename) -> storage_path
    # - get_file_url(storage_path, expires_in=...) -> signed_url
    # - preprocess_data(X) -> (X_processed, encoders, meta)
    # - ModelClassifyingTrainer, lgb_params_c, cat_params_c, xgb_params_c
    # - multiclass_brier(y_true, probas, class_order)
    # - compute_reliability_bins_multiclass(probas, y_true, class_order, n_bins)
    # - ece_from_bins(bins)
    # - decision_curve_binary(probas_pos, y_true)
    # - plot_reliability_diagram(bins) -> base64
    # - plot_decision_curve(data, recommended_t=...) -> base64
    # - plot_segment_heatmap(segments_table) -> base64
    # - plot_kpi_abstentions(count, rate) -> base64
    # - plot_trend(values, title) -> base64
    # - compute_segments_table(...)
    # - conformal_utils.fit_thresholds / batch_metrics / prediction_set
    # - master_db_cm(), _append_limited_metadata(user_id, entry, db, max_entries)
    # - ensure_json_serializable(obj)

    # ----------------------------
    # Validate upload mode
    # ----------------------------
def do_classification(
    user_id: str,
    file_path: str = None,
    current_user: dict = None,
    train_path: str = None,
    test_path: str = None,
    target_column: str = "target",
    drop_columns: str = "",
    *,
    dataset_id: Optional[int] = None,
    plan_id: Optional[str] = None,
    plan_config: Optional[PlanConfig] = None
) -> dict:
# ----------------------------
# Imports (kept local as in your style)
# ----------------------------
    import json
    import tempfile
    import os
    import uuid
    from datetime import datetime
    from pathlib import Path as PathL
    from collections import Counter

    import numpy as np
    import pandas as pd
    import joblib

    from sklearn.model_selection import train_test_split
    from sklearn.base import clone
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix, log_loss
    )
    if file_path and (train_path or test_path):
        raise ValueError("Provide either file_path or both train_path+test_path, not both.")
    if (train_path and not test_path) or (test_path and not train_path):
        raise ValueError("Both train_path and test_path must be provided together.")
    if not file_path and not (train_path and test_path):
        raise ValueError("Provide either a full dataset (file_path) or both train_path+test_path.")

    # ----------------------------
    # Initialize locals
    # ----------------------------
    local_file_path = None
    local_train_path = None
    local_test_path = None
    cfg = plan_config
    if cfg is None and plan_id:
        cfg = _load_plan_config_from_db(user_id=user_id, plan_id=plan_id)  # you implement
    if cfg is None and dataset_id:
        cfg = _load_latest_plan_config_for_dataset(user_id=user_id, dataset_id=dataset_id)  # you implement
    if cfg is None:
        cfg = PlanConfig(
            options=PlanOptions(),  # default risk / alpha / horizons
            actions=[],             # you can inject a default email channel if you prefer
            plan_type="Balanced",
        )

    # risk preset harmonization (same logic as your frontend)
    risk = cfg.options.risk_preset
    if risk == "conservative":
        cfg.options.alpha, cfg.options.threshold, cfg.options.coverage, cfg.options.ece = 0.05, 0.80, 0.95, 0.02
    elif risk == "aggressive":
        cfg.options.alpha, cfg.options.threshold, cfg.options.coverage, cfg.options.ece = 0.20, 0.60, 0.85, 0.05
    else:
        cfg.options.alpha, cfg.options.threshold, cfg.options.coverage, cfg.options.ece = 0.10, 0.71, 0.92, 0.03

    # Use cfg.options.alpha as ALPHA everywhere below
    ALPHA = float(cfg.options.alpha)
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # ----------------------------
            # Download files from Supabase to temporary locations
            # ----------------------------
            if file_path:
                file_bytes = download_file_from_supabase(file_path)
                local_file_path = os.path.join(temp_dir, os.path.basename(file_path))
                with open(local_file_path, "wb") as f:
                    f.write(file_bytes)

            if train_path and test_path:
                train_bytes = download_file_from_supabase(train_path)
                test_bytes = download_file_from_supabase(test_path)

                local_train_path = os.path.join(temp_dir, os.path.basename(train_path))
                local_test_path = os.path.join(temp_dir, os.path.basename(test_path))

                with open(local_train_path, "wb") as f:
                    f.write(train_bytes)
                with open(local_test_path, "wb") as f:
                    f.write(test_bytes)

            # ----------------------------
            # Prepare model/output directories
            # ----------------------------
            model_dir = os.path.join(temp_dir, "models")
            os.makedirs(model_dir, exist_ok=True)

            user_dir = PathL(temp_dir) / "user_outputs"
            user_dir.mkdir(exist_ok=True)

            # ----------------------------
            # Process data: single file or train/test split
            # ----------------------------
            if local_file_path:
                df = pd.read_csv(local_file_path)
                if drop_columns:
                    drops = [c.strip() for c in drop_columns.split(",") if c.strip() and c in df.columns]
                    if drops:
                        df.drop(columns=drops, inplace=True)

                if target_column not in df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in dataset.")

                df = df.dropna(subset=[target_column])
                y = df[target_column]
                X = df.drop(columns=["ID", target_column], errors="ignore")

                # Encode categorical target if needed
                if y.dtype == "object" or y.dtype.name == "category":
                    y, _ = y.astype(str).factorize()
                    y = pd.Series(y, name=target_column).astype("int32")  # keep name for later

                X, _, _ = preprocess_data(X)
                print("→ Post-preprocess X NaNs:", X.isna().sum().sum())

                # numeric/bool only to be safe for tree libs defaults
                X = X.select_dtypes(include=["int", "float", "bool"])

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                X_train = X_train.reset_index(drop=True).select_dtypes(include=["int", "float", "bool"])
                X_test = X_test.reset_index(drop=True).select_dtypes(include=["int", "float", "bool"])
                y_train = y_train.reset_index(drop=True)
                y_test = y_test.reset_index(drop=True)

                print("→ X_train NaNs:", X_train.isna().sum().sum())
                print("→ y_train NaNs:", y_train.isna().sum())

                train_df = pd.concat([X_train, y_train.rename(target_column)], axis=1)
                test_df = pd.concat([X_test, y_test.rename(target_column)], axis=1)
                dataset_name = os.path.basename(file_path)
            else:
                train_df = pd.read_csv(local_train_path)
                test_df = pd.read_csv(local_test_path)

                train_df = train_df.dropna(subset=[target_column])
                test_df = test_df.dropna(subset=[target_column])

                if drop_columns:
                    drops = [c.strip() for c in drop_columns.split(",") if c.strip()]
                    if drops:
                        train_df.drop(columns=[c for c in drops if c in train_df.columns], inplace=True, errors="ignore")
                        test_df.drop(columns=[c for c in drops if c in test_df.columns], inplace=True, errors="ignore")

                if target_column not in train_df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in training dataset.")
                if target_column not in test_df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in test dataset.")

                # Extract targets
                y_train = train_df[target_column]
                y_test = test_df[target_column]

                # Encode targets if object/categorical
                for name in ("y_train", "y_test"):
                    y_var = y_train if name == "y_train" else y_test
                    if y_var.dtype == "object" or y_var.dtype.name == "category":
                        encoded, _ = y_var.astype(str).factorize()
                        if name == "y_train":
                            y_train = pd.Series(encoded, name=target_column).astype("int32")
                        else:
                            y_test = pd.Series(encoded, name=target_column).astype("int32")

                # Drop ID and target from features
                X_train = train_df.drop(columns=[target_column], errors="ignore")
                X_test = test_df.drop(columns=[target_column], errors="ignore")

                X_train.drop(columns=["ID"], inplace=True, errors="ignore")
                X_test.drop(columns=["ID"], inplace=True, errors="ignore")

                # Preprocess
                X_train, _, _ = preprocess_data(X_train)
                X_test, _, _ = preprocess_data(X_test)

                print("→ X_train NaNs:", X_train.isna().sum().sum())
                print("→ y_train NaNs:", y_train.isna().sum())

                X_train = X_train.reset_index(drop=True)
                X_test = X_test.reset_index(drop=True)
                y_train = y_train.reset_index(drop=True)
                y_test = y_test.reset_index(drop=True)

                # Align columns
                common_columns = list(set(X_train.columns) & set(X_test.columns))
                X_train = X_train[common_columns]
                X_test = X_test[common_columns]

                train_df = pd.concat([X_train, y_train.rename(target_column)], axis=1).reset_index(drop=True)
                test_df = pd.concat([X_test, y_test.rename(target_column)], axis=1)
                dataset_name = f"{os.path.basename(train_path)}+{os.path.basename(test_path)}"

            # ----------------------------
            # Train models (CV via custom trainer that returns OOF preds)
            # ----------------------------
            trainer = ModelClassifyingTrainer(train_df, n_splits=5)
            lgb_models, lgb_oof = trainer.train_model(lgb_params_c, target_column, title="LightGBM")
            ctb_models, ctb_oof = trainer.train_model(cat_params_c, target_column, title="CatBoost")
            xgb_models, xgb_oof = trainer.train_model(xgb_params_c, target_column, title="XGBoost")

            # CV Performance
            cv_scores = {
                name: {
                    "accuracy": float(accuracy_score(y_train, preds)),
                    "f1": float(f1_score(y_train, preds, average="weighted")),
                    "precision": float(precision_score(y_train, preds, average="weighted")),
                    "recall": float(recall_score(y_train, preds, average="weighted")),
                }
                for name, preds in zip(
                    ["LightGBM", "CatBoost", "XGBoost"], [lgb_oof, ctb_oof, xgb_oof]
                )
            }

            # Save training columns for SHAP alignment
            training_columns = list(X_train.columns)
            training_columns_path = user_dir / "training_columns.json"
            with open(training_columns_path, "w") as f:
                json.dump(training_columns, f)

            # Upload training columns to Supabase for prediction alignment
            upload_file_to_supabase(
                user_id=user_id,
                file_path=str(training_columns_path),
                filename="training_columns.json"
            )

            # Align X_test early (BEFORE any predictions!)
            missing_cols = set(X_train.columns) - set(X_test.columns)
            extra_cols = set(X_test.columns) - set(X_train.columns)
            for col in missing_cols:
                X_test[col] = 0
            X_test.drop(columns=list(extra_cols), inplace=True, errors="ignore")
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

            # --- Pick best base model ---
            best_name = max(cv_scores, key=lambda m: cv_scores[m]["f1"])
            best_models = {"LightGBM": lgb_models, "CatBoost": ctb_models, "XGBoost": xgb_models}[best_name]
            base_model = clone(best_models[0])

            # === (A) Train CALIBRATION model on TRAIN only ===
            calib_model = clone(base_model)
            calib_model.fit(X_train, y_train)
            class_order = calib_model.classes_

            if not hasattr(calib_model, "predict_proba"):
                raise RuntimeError("Model does not support predict_proba; required for conformal sets.")
            calib_probs = calib_model.predict_proba(X_test)

            # Safe label→index mapping
            index_of = {c: i for i, c in enumerate(class_order)}
            calib_true_idx = np.array([index_of[yy] for yy in y_test])

            # === (B) Fit conformal thresholds on calibration split ===
            from conformal_utils import fit_thresholds, batch_metrics, prediction_set
            ALPHA = 0.10
            MONDRIAN = False

            # ensure artifacts dir exists BEFORE saving thresholds
            artifacts_dir = PathL("artifacts") / str(user_id)
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            latest_path = artifacts_dir / "latest_version.txt"

            thr = fit_thresholds(
                calib_probs=calib_probs,
                calib_true_idx=calib_true_idx,
                class_order=class_order,
                alpha=ALPHA,
                mondrian=MONDRIAN
            )

            report_sets = batch_metrics(calib_probs, calib_true_idx, thr, max_set_size=2)
            print("[Conformal] τ:", thr.global_tau if not thr.mondrian else thr.per_class_tau)
            print("[Conformal] calib metrics:", report_sets)

            # Versioning
            model_version = f"{best_name}_v{pd.Timestamp.utcnow():%Y%m%d_%H%M%S}"

            thr_path = artifacts_dir / f"{model_version}_conformal_thr_90.json"
            thr.save(str(thr_path))

            # === (C) Fit FINAL model on FULL data ===
            if local_file_path:
                full_df = train_df.copy()
            else:
                full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

            X_full = full_df.drop(columns=[target_column], errors="ignore").reindex(columns=X_train.columns, fill_value=0)
            y_full = full_df[target_column]

            final_model = clone(base_model)
            final_model.fit(X_full, y_full)

            data_path = artifacts_dir / f"{model_version}_train_full.parquet"
            full_df.to_parquet(data_path, index=False)

            # Save ONE canonical, versioned model
            model_path = artifacts_dir / f"{model_version}.pkl"
            joblib.dump(final_model, model_path)

            latest_path.write_text(str(model_version))

            meta = {
                "model_version": model_version,
                "class_order": list(class_order),
                "training_columns": list(X_train.columns),
                "coverage_target": 1.0 - ALPHA,
            }
            (artifacts_dir / f"{model_version}_meta.json").write_text(json.dumps(meta))

            # --- “shadow” evaluation on X_test with final_model (rename, not 'test') ---
            preds_shadow = final_model.predict(X_test)
            test_probs = final_model.predict_proba(X_test) if hasattr(final_model, "predict_proba") else None

            # Honest test metrics (calibration split, from calib_model)
            test_preds = calib_model.predict(X_test)  # honest
            test_scores = {
                "accuracy": accuracy_score(y_test, test_preds),
                "f1": f1_score(y_test, test_preds, average="weighted"),
                "precision": precision_score(y_test, test_preds, average="weighted"),
                "recall": recall_score(y_test, test_preds, average="weighted")
            }
            if len(set(y_test)) == 2:
                tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
                test_scores["conversion_rate"] = round((tp / (tp + fp)) * 100, 2) if (tp + fp) else 0.0

            # Confusion matrix/class distribution (use honest test)
            cm = confusion_matrix(y_test, test_preds).tolist()
            class_distribution = dict(Counter(test_preds))

            # Core metrics (honest)
            metrics = {
                "accuracy": float(test_scores["accuracy"]),
                "f1_weighted": float(test_scores["f1"]),
                "precision_weighted": float(test_scores["precision"]),
                "recall_weighted": float(test_scores["recall"]),
                "confusion_matrix": cm,
                "class_distribution": {str(k): int(v) for k, v in class_distribution.items()},
            }

            # Calibration metrics (on calib split)
            metrics["logloss"] = float(log_loss(y_test, calib_probs, labels=class_order))
            metrics["brier"] = multiclass_brier(y_test, calib_probs, class_order)

            # Conformal summary (calib split)
            set_sizes = [len(prediction_set(calib_probs[i], thr, max_set_size=2)[0]) for i in range(calib_probs.shape[0])]
            set_size_hist = dict(Counter(set_sizes))
            conformal = {
                "coverage_target": 1.0 - ALPHA,
                "coverage_actual": float(report_sets.get("coverage")),
                "avg_set_size": float(report_sets.get("avg_set_size")),
                "abstention_rate": float(report_sets.get("abstention_rate")),
                "set_size_histogram": {str(k): int(v) for k, v in set_size_hist.items()},
            }

            # Reliability bins (calib split)
            reliability_bins = compute_reliability_bins_multiclass(calib_probs, y_test, class_order, n_bins=10)
            metrics["ece_top1"] = float(ece_from_bins(reliability_bins)) if reliability_bins else None

            decision_curve_data = []
            recommended_t = None

            if len(class_order) == 2:
                # identify positive-class index
                pos_label = 1 if 1 in set(class_order) else class_order[1]
                pos_idx = list(class_order).index(pos_label)

                # curve across thresholds
                decision_curve_data = decision_curve_binary(calib_probs[:, pos_idx], y_test)

                # optional unit economics (override from config/DB)
                PROFIT_PER_TP = 100.0  # revenue/benefit of a true positive
                COST_PER_FP = 10.0     # cost of contacting a false positive, etc.
                OPS_COST_PER_CASE = 0.0

                # fill expected_profit if not present
                for p in decision_curve_data:
                    if p["precision"] is None:
                        p["expected_profit"] = None
                        continue
                    tp_est = p["precision"] * p["volume"]
                    fp_est = p["volume"] - tp_est
                    exp_profit = tp_est * PROFIT_PER_TP - fp_est * COST_PER_FP - p["volume"] * OPS_COST_PER_CASE
                    p["expected_profit"] = float(exp_profit)

                # choose threshold: prefer max expected_profit; fallback to max F1
                def score(p):
                    if p["expected_profit"] is not None:
                        return p["expected_profit"]
                    if not p["precision"] or not p["recall"]:
                        return -1e18
                    return 2 * p["precision"] * p["recall"] / (p["precision"] + p["recall"])

                if decision_curve_data:
                    recommended_t = float(max(decision_curve_data, key=score)["threshold"])

            # Abstentions count on calibration probs
            abstentions_count = 0
            MAX_SET_SIZE = 2
            for i in range(calib_probs.shape[0]):
                _, abst = prediction_set(calib_probs[i], thr, max_set_size=MAX_SET_SIZE)
                if abst:
                    abstentions_count += 1

            # Trends over time (ECE & Coverage)
            ece_by_week = []
            coverage_by_week = []
            week_starts = []

            TS_CANDIDATES = ["timestamp", "ts", "created_at", "event_time", "date"]
            ts_col = next((c for c in TS_CANDIDATES if c in test_df.columns), None)

            if ts_col and test_probs is not None:
                class_to_idx = {c: i for i, c in enumerate(class_order)}
                # robust conversion for y_test values
                y_idx_all = np.array([class_to_idx[v] for v in y_test])

                df_tmp = (
                    pd.DataFrame({
                        "row_id": np.arange(len(test_df)),
                        "ts": pd.to_datetime(test_df[ts_col], errors="coerce"),
                        "y_idx": y_idx_all,
                    })
                    .dropna(subset=["ts"])
                    .sort_values("ts")
                    .reset_index(drop=True)
                )

                for week_start, g in df_tmp.groupby(pd.Grouper(key="ts", freq="W-MON")):
                    if len(g) == 0:
                        continue
                    ridx = g["row_id"].to_numpy()

                    # reliability bins for this window
                    bins_w = compute_reliability_bins_multiclass(
                        calib_probs[ridx],
                        [class_order[i] for i in y_idx_all[ridx]],
                        class_order,
                        n_bins=10
                    )
                    ece_by_week.append(float(ece_from_bins(bins_w)) if bins_w else None)

                    # coverage via conformal sets (compute once per window)
                    covered = 0
                    for r in ridx:
                        idxs, _ = prediction_set(test_probs[r], thr, max_set_size=MAX_SET_SIZE)
                        if y_idx_all[r] in idxs:
                            covered += 1
                    coverage_by_week.append(covered / len(ridx))

                    week_starts.append(week_start.normalize().isoformat())
            else:
                # Fallback: single snapshot
                ece_now = ece_from_bins(reliability_bins)
                ece_by_week = [float(ece_now) if ece_now is not None else None]
                cov_now = conformal.get("coverage_actual", None)
                coverage_by_week = [float(cov_now) if cov_now is not None else None]
                if ts_col and test_df[ts_col].notna().any():
                    week_starts = [pd.to_datetime(test_df[ts_col], errors="coerce").dropna().max().normalize().isoformat()]
                else:
                    week_starts = ["current"]

            TARGET_COVERAGE = 1.0 - ALPHA
            SEGMENT_COLS = [c for c in ["channel", "cohort", "region"] if c in test_df.columns]  # customize

            if test_probs is not None:
                segments_table = compute_segments_table(
                    test_df=test_df,
                    y_test=y_test,
                    test_probs=calib_probs,           # use calibration probs for honesty
                    class_order=class_order,
                    thr=thr,
                    segment_cols=SEGMENT_COLS,
                    min_n=200,
                    max_set_size=MAX_SET_SIZE,
                    coverage_target=TARGET_COVERAGE,
                    ece_alarm=0.08,
                    coverage_tolerance=0.02,
                    max_categories=30,
                )
            else:
                segments_table = []  # no probs → no segments

            visualizations_data = {"reliability_bins": reliability_bins}
            visualizations_data["segments"] = segments_table
            visualizations_data.update({
                "decision_curve": decision_curve_data,
                "recommended_threshold": recommended_t,
                "abstentions_count": int(abstentions_count),
                "ece_by_week": ece_by_week,
                "coverage_by_week": coverage_by_week,
                "trend_weeks": week_starts,
            })

            # Ensure you have a dict (preserves your guard)
            visualizations = {}  # fresh dict

            # 1) Reliability diagram
            try:
                reliability_b64 = plot_reliability_diagram(reliability_bins)
                if reliability_b64:
                    visualizations["reliability_diagram"] = f"data:image/png;base64,{reliability_b64}"
            except Exception as e:
                print(f"[⚠️] Reliability diagram failed: {e}")

            # 2) Decision / ROI curve
            try:
                decision_b64 = plot_decision_curve(decision_curve_data, recommended_t=recommended_t)
                if decision_b64:
                    visualizations["decision_curve"] = f"data:image/png;base64,{decision_b64}"
            except Exception as e:
                print(f"[⚠️] Decision curve failed: {e}")

            # 3) Segment heatmap
            try:
                seg_b64 = plot_segment_heatmap(segments_table)
                if seg_b64:
                    visualizations["segment_heatmap"] = f"data:image/png;base64,{seg_b64}"
            except Exception as e:
                print(f"[⚠️] Segment heatmap failed: {e}")

            # 4) Abstentions KPI
            try:
                rate = float(conformal.get("abstention_rate") or 0.0)
                abst_b64 = plot_kpi_abstentions(int(abstentions_count), rate)
                if abst_b64:
                    visualizations["abstentions_summary"] = f"data:image/png;base64,{abst_b64}"
            except Exception as e:
                print(f"[⚠️] Abstentions KPI failed: {e}")

            # 5) Trends
            try:
                ece_b64 = plot_trend(ece_by_week, "ECE over time")
                if ece_b64:
                    visualizations["trend_ece"] = f"data:image/png;base64,{ece_b64}"
            except Exception as e:
                print(f"[⚠️] ECE trend failed: {e}")

            try:
                cov_b64 = plot_trend(coverage_by_week, "Coverage over time")
                if cov_b64:
                    visualizations["trend_coverage"] = f"data:image/png;base64,{cov_b64}"
            except Exception as e:
                print(f"[⚠️] Coverage trend failed: {e}")

            # ----------------------------
            # Upload artifacts (model + thresholds + data)
            # ----------------------------
            artifact_paths = {}
            try:
                model_filename = PathL(model_path).name
                model_supabase_path = upload_file_to_supabase(user_id, str(model_path), model_filename)
                artifact_paths["model_upload_path"] = model_supabase_path

                thr_filename = PathL(thr_path).name
                thr_supabase_path = upload_file_to_supabase(user_id, str(thr_path), thr_filename)
                artifact_paths["conformal_thresholds_upload_path"] = thr_supabase_path

                data_filename = PathL(data_path).name
                data_supabase_path = upload_file_to_supabase(user_id, str(data_path), data_filename)
                artifact_paths["data_upload_path"] = data_supabase_path

                # Signed URLs (optional — UI-only)
                artifact_paths["model_url"] = get_file_url(model_supabase_path, expires_in=3600)
                artifact_paths["data_url"] = get_file_url(data_supabase_path, expires_in=3600)
                artifact_paths["conformal_thresholds_url"] = get_file_url(thr_supabase_path, expires_in=3600)
            except Exception as upload_error:
                print(f"[⚠️] Upload error: {upload_error}")

            # ----------------------------
            # Slim metadata entry (DB)
            # ----------------------------
            try:
                entry = {
                    "id": str(uuid.uuid4()),
                    "created_at": datetime.utcnow().isoformat(),
                    "type": "classification_train",
                    "dataset": dataset_name,
                    "target_col": target_column,
                    "model_version": model_version,
                    "cv_scores": [[m, s["accuracy"], s["f1"], s["precision"], s["recall"]] for m, s in cv_scores.items()],
                    "visualizations": visualizations,
                    "visualizations_data": visualizations_data,
                    "metrics": metrics,
                    "conformal": conformal,
                    "artifacts": artifact_paths,
                    "debug": {
                        "feature_alignment": {
                            "training_features": int(len(training_columns)),
                            "prediction_features": int(len(X_test.columns)),
                            "missing_features_added": list(missing_cols),
                            "extra_features_removed": list(extra_cols),
                        },
                        "parameters": {"drop_columns": drop_columns},
                        "training_columns": training_columns,
                    },
                }
                entry = ensure_json_serializable(entry)
                with master_db_cm() as db:
                    _append_limited_metadata(user_id, entry, db=db, max_entries=5)
            except Exception as meta_error:
                print(f"[⚠️] Metadata save error: {meta_error}")

            # ----------------------------
            # Final API response (lean + numeric viz data)
            # ----------------------------
            response_data = {
                "status": "success",
                "id": str(uuid.uuid4()),
                "created_at": datetime.utcnow().isoformat(),
                "type": "classification",
                "dataset": dataset_name,
                "target_col": target_column,
                "best_model": best_name,
                "model_version": model_version,
                "total_rows": int(len(X_full)),
                "metrics": metrics,                     # includes confusion_matrix, class_distribution, logloss/brier if available
                "conformal": conformal,                 # coverage, set-size stats, histogram
                "visualizations_data": visualizations_data,  # reliability_bins (+ decision_curve if binary)
                "visualizations": visualizations,
                "cv_scores": cv_scores,
                "class_order": list(class_order),       # <-- FIXED (was '=' instead of ':')
                "artifacts": artifact_paths,            # URLs/paths for UI to fetch downloads
                "debug": {                              # optional; hide via flag if desired
                    "feature_alignment": {
                        "training_features": int(len(training_columns)),
                        "prediction_features": int(len(X_test.columns)),
                        "missing_features_added": list(missing_cols),
                        "extra_features_removed": list(extra_cols),
                    },
                    "parameters": {"drop_columns": drop_columns},
                },
            }

            return response_data

    except Exception as e:
        print(f"[⚠️] Error in do_classification: {e}")
        raise e
