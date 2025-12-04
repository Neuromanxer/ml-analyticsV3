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






# from .anomaly_detection import train_best_anomaly_detection
# from .preprocessing import preprocess_data
# from .classification import ModelClassifyingTrainer, lgb_params_c, cat_params_c, xgb_params_c
# from .auth import master_db_cm
# from .storage import upload_file_to_supabase, download_file_from_supabase, handle_file_upload, download_file_from_supabase, list_user_files, delete_file_from_supabase, get_file_url
# from .target import generate_customer_summary
# from .auth import _append_limited_metadata, _append_metadata, _load_metadata, _save_metadata, _get_meta_path
# from .regression import ModelTrainer, lgb_params, cat_params, xgb_params, DataPreprocessor, train_regression_models, generate_visualizations_improved
# from .preprocessing import preprocess_data
# from .classification import ModelClassifyingTrainer, lgb_params_c, cat_params_c, xgb_params_c
# from .feature_importance import safe_generate_feature_importance
# from .clustering import run_kmeans, find_optimal_k, label_clusters_general
# from .timeSeries import ScenarioManager, ARIMAModel, ExponentialSmoothingModel, LSTMModel, RandomForestModel, generate_scenario_visualizations,  SARIMAModel
# from .regression import (
#     get_confidence_interval,
#     get_percentile_summary,
#     get_risk_shift_summary,
#     get_class_distribution_change
# )
# from .counterfactual import clean_data_for_json, safe_float_conversion, compute_summary_metrics


from counterfactual import clean_data_for_json, safe_float_conversion, compute_summary_metrics
from timeSeries import ScenarioManager, ARIMAModel, ExponentialSmoothingModel, LSTMModel, RandomForestModel, generate_scenario_visualizations, SARIMAModel
from anomaly_detection import train_best_anomaly_detection
from preprocessing import preprocess_data
from classification import ModelClassifyingTrainer, lgb_params_c, cat_params_c, xgb_params_c
from auth import master_db_cm
from storage import upload_file_to_supabase, download_file_from_supabase, handle_file_upload, download_file_from_supabase, list_user_files, delete_file_from_supabase, get_file_url
from target import generate_customer_summary
from auth import _append_limited_metadata, _append_metadata, _load_metadata, _save_metadata, _get_meta_path
from regression import ModelTrainer, lgb_params, cat_params, xgb_params, DataPreprocessor, train_regression_models, generate_visualizations_improved
from preprocessing import preprocess_data
from feature_importance import safe_generate_feature_importance
from clustering import run_kmeans, find_optimal_k, label_clusters_general
from regression import (
    get_confidence_interval,
    get_percentile_summary,
    get_risk_shift_summary,
    get_class_distribution_change
)




# OAuth2 scheme
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Save logs to a file
        logging.StreamHandler()  # Print logs to console
    ],
)
logger = logging.getLogger(__name__)

# celery_app = Celery(
#     "tasks",
#     broker="redis://red-d1n270gdl3ps73fqo7fg:6379/0",
#     backend="redis://red-d1n270gdl3ps73fqo7fg:6379/1"
# )

# BROKER_URL = "redis://localhost:6379/0"
# RESULT_BACKEND = "redis://localhost:6379/1"

# celery_app = Celery(
#     "worker",
#     broker=BROKER_URL,
#     backend=RESULT_BACKEND
# )

celery_app = Celery(
    "worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1"
)

import numpy as np

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

import base64
from io import BytesIO
import seaborn as sns  # Add this import since it was missing but needed
def plot_to_base64(fig):
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    fig.tight_layout()
    buf = BytesIO()
    
    # Use a proper canvas to render the figure
    FigureCanvas(fig).print_png(buf)  # ✅ ensures it's rendered correctly

    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)  # ✅ Close the figure properly
    return img_base64

from fastapi.responses import FileResponse, StreamingResponse
# Assuming you have these imports for your auth system
# from your_auth_module import get_current_active_user, User

def do_classification(
    user_id: str,
    file_path: str = None,
    current_user: dict = None,
    train_path: str = None,
    test_path: str = None,
    target_column: str = "target",
    drop_columns: str = ""
) -> dict:
    import json
    import subprocess
    import tempfile
    import os
    import pandas as pd
    from pathlib import Path as PathL

    # Validate upload mode
    if file_path and (train_path or test_path):
        raise ValueError("Provide either file_path or both train_path+test_path, not both.")
    if (train_path and not test_path) or (test_path and not train_path):
        raise ValueError("Both train_path and test_path must be provided together.")
    if not file_path and not (train_path and test_path):
        raise ValueError("Provide either a full dataset (file_path) or both train_path+test_path.")
    
    # Handle file downloads from Supabase
    # Note: file_path, train_path, test_path are now Supabase storage paths (e.g., "user123/dataset.csv")
    local_file_path = None
    local_train_path = None
    local_test_path = None
    temp_files = []
    
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download files from Supabase to temporary locations
            if file_path:
                # Download single file from Supabase using the storage path
                file_bytes = download_file_from_supabase(file_path)
                local_file_path = os.path.join(temp_dir, os.path.basename(file_path))
                with open(local_file_path, 'wb') as f:
                    f.write(file_bytes)
                temp_files.append(local_file_path)
            
            if train_path and test_path:
                # Download train and test files from Supabase using the storage paths
                train_bytes = download_file_from_supabase(train_path)
                test_bytes = download_file_from_supabase(test_path)
                
                local_train_path = os.path.join(temp_dir, os.path.basename(train_path))
                local_test_path = os.path.join(temp_dir, os.path.basename(test_path))
                
                with open(local_train_path, 'wb') as f:
                    f.write(train_bytes)
                with open(local_test_path, 'wb') as f:
                    f.write(test_bytes)
                    
                temp_files.extend([local_train_path, local_test_path])
            
            # Prepare model directory
            model_dir = os.path.join(temp_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            
            # Create user directory for outputs
            user_dir = PathL(temp_dir) / "user_outputs"
            user_dir.mkdir(exist_ok=True)
            
            # Process data based on single file or train/test split
            if local_file_path:
                df = pd.read_csv(local_file_path)
                if drop_columns:
                    drops = [c.strip() for c in drop_columns.split(",") if c.strip() and c in df.columns]
                    df.drop(columns=drops, inplace=True)

                if target_column not in df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in dataset.")

                df = df.dropna(subset=[target_column])
                y = df[target_column]
                X = df.drop(columns=['ID', target_column], errors='ignore')
                # Encode categorical target if needed
                if y.dtype == 'object' or y.dtype.name == 'category':
                    y, _ = y.astype(str).factorize()
                    y = pd.Series(y, name=target_column).astype("int32")  # keep name for later
                X, _, _ = preprocess_data(X)
                print("→ Post-preprocess X NaNs:", X.isna().sum().sum())

                X = X.select_dtypes(include=["int", "float", "bool"])

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                X_train = X_train.reset_index(drop=True)
                X_test  = X_test.reset_index(drop=True)
                y_train = y_train.reset_index(drop=True)
                y_test  = y_test.reset_index(drop=True)
                
                X_train = X_train.select_dtypes(include=["int", "float", "bool"])
                X_test = X_test.select_dtypes(include=["int", "float", "bool"])
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
                    train_df.drop(columns=[c for c in drops if c in train_df.columns], inplace=True)
                    test_df.drop(columns=[c for c in drops if c in test_df.columns], inplace=True)

                if target_column not in train_df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in training dataset.")
                if target_column not in test_df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in test dataset.")

                # Extract targets
                y_train = train_df[target_column]
                y_test = test_df[target_column]
                # Encode target if object type
                for y_var, name in [(y_train, "y_train"), (y_test, "y_test")]:
                    if y_var.dtype == 'object' or y_var.dtype.name == 'category':
                        encoded, _ = y_var.astype(str).factorize()
                        if name == "y_train":
                            y_train = pd.Series(encoded, name=target_column).astype("int32")
                        else:
                            y_test = pd.Series(encoded, name=target_column).astype("int32")


                # Drop ID and target from features
                X_train = train_df.drop(columns=[target_column], errors='ignore')
                X_test = test_df.drop(columns=[target_column], errors='ignore')

                X_train.drop(columns=["ID"], inplace=True, errors='ignore')
                X_test.drop(columns=["ID"], inplace=True, errors='ignore')

                # Now preprocess
                X_train, _, _ = preprocess_data(X_train)
                X_test, _, _ = preprocess_data(X_test)
                print("→ X_train NaNs:", X_train.isna().sum().sum())
                print("→ y_train NaNs:", y_train.isna().sum())

                X_train = X_train.reset_index(drop=True)
                X_test  = X_test.reset_index(drop=True)
                y_train = y_train.reset_index(drop=True)
                y_test  = y_test.reset_index(drop=True)

                common_columns = list(set(X_train.columns) & set(X_test.columns))
                X_train = X_train[common_columns]
                X_test = X_test[common_columns]

                train_df = pd.concat([X_train, y_train.rename(target_column)], axis=1)
                train_df = train_df.reset_index(drop=True)
                test_df = pd.concat([X_test, y_test.rename(target_column)], axis=1)
                dataset_name = f"{os.path.basename(train_path)}+{os.path.basename(test_path)}"
            
            # Train models
            trainer = ModelClassifyingTrainer(train_df, n_splits=5)
            lgb_models, lgb_oof = trainer.train_model(lgb_params_c, target_column, title="LightGBM")
            ctb_models, ctb_oof = trainer.train_model(cat_params_c, target_column, title="CatBoost")
            xgb_models, xgb_oof = trainer.train_model(xgb_params_c, target_column, title="XGBoost")

            # CV Performance
            cv_scores = {
                name: {
                    "accuracy": accuracy_score(y_train, preds),
                    "f1": f1_score(y_train, preds, average="weighted"),
                    "precision": precision_score(y_train, preds, average="weighted"),
                    "recall": recall_score(y_train, preds, average="weighted")
                } for name, preds in zip(["LightGBM", "CatBoost", "XGBoost"], [lgb_oof, ctb_oof, xgb_oof])
            }
            
            # Save training columns for SHAP alignment
            training_columns = list(X_train.columns)
            training_columns_path = user_dir / "training_columns.json"
            with open(training_columns_path, "w") as f:
                json.dump(training_columns, f)

            # ✅ Upload to Supabase so prediction can access it
            upload_file_to_supabase(
                user_id=user_id,
                file_path=str(training_columns_path),
                filename="training_columns.json"
            )


            # Pick Best Model
            best_name = max(cv_scores, key=lambda m: cv_scores[m]["f1"])
            best_models = {
                "LightGBM": lgb_models,
                "CatBoost": ctb_models,
                "XGBoost": xgb_models
            }[best_name]
            final_model = clone(best_models[0])

            # Align X_test to match training columns
            missing_cols = set(X_train.columns) - set(X_test.columns)
            extra_cols = set(X_test.columns) - set(X_train.columns)
            for col in missing_cols:
                X_test[col] = 0
            X_test.drop(columns=list(extra_cols), inplace=True, errors='ignore')
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

            if local_file_path:
                full_df = train_df.copy()
            else:
                full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

            X_full = full_df.drop(columns=['ID', target_column], errors='ignore')
            y_full = full_df[target_column]

            # Ensure full_df matches training columns
            X_full = X_full.reindex(columns=X_train.columns, fill_value=0)
            final_model.fit(X_full, y_full)
            
            # Evaluate on test set
            preds = final_model.predict(X_test)
            test_scores = {
                "accuracy": accuracy_score(y_test, preds),
                "f1": f1_score(y_test, preds, average="weighted"),
                "precision": precision_score(y_test, preds, average="weighted"),
                "recall": recall_score(y_test, preds, average="weighted")
            }
            
            # Initialize conversion_rate for later use
            conversion_rate = 0.0
            
            # Add conversion rate if binary classification
            if len(set(y_test)) == 2:
                tp = ((y_test == 1) & (preds == 1)).sum()
                predicted_positive = (preds == 1).sum()
                conversion_rate = tp / predicted_positive if predicted_positive else 0.0
                test_scores["conversion_rate"] = round(conversion_rate * 100, 2)
            
            # Class distribution (predicted)
            class_distribution = dict(Counter(preds))

            # Confusion matrix
            cm = confusion_matrix(y_test, preds).tolist()
            
            # Classification report
            report = classification_report(y_test, preds, output_dict=True)
            
            # Generate visualizations
            # 1. Class distribution plot
            try:
                fig_dist = plt.figure()
                sns.barplot(x=list(class_distribution.keys()), y=list(class_distribution.values()))
                plt.title("Class Distribution")
                plt.xlabel("Predicted Class")
                plt.ylabel("Count")
                class_dist_base64 = plot_to_base64(fig_dist)
                plt.close(fig_dist)
            except Exception as e:
                print(f"[⚠️] Class distribution plot failed: {e}")
                class_dist_base64 = ""

            # 2. Confusion matrix plot
            try:
                fig_cm = plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
                plt.title("Confusion Matrix")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                confusion_matrix_base64 = plot_to_base64(fig_cm)
                plt.close(fig_cm)
            except Exception as e:
                print(f"[⚠️] Confusion matrix plot failed: {e}")
                confusion_matrix_base64 = ""

            # 3. Classification report table as image
            try:
                fig_report = plt.figure(figsize=(8, 4))
                report_df = pd.DataFrame(report).transpose()
                sns.heatmap(report_df.iloc[:-1, :-1], annot=True, fmt=".2f", cmap="YlGnBu")
                plt.title("Classification Report")
                classification_report_base64 = plot_to_base64(fig_report)
                plt.close(fig_report)
            except Exception as e:
                print(f"[⚠️] Classification report plot failed: {e}")
                classification_report_base64 = ""

            model_path = PathL(model_dir) / f"{user_id}_best_classifier.pkl"
            joblib.dump(final_model, model_path)

            # Save the data used for SHAP (full processed dataset)
            data_path = user_dir / "data.csv"
            full_df_model_data = pd.concat([X_full, y_full.rename(target_column)], axis=1)
            full_df_model_data.to_csv(data_path, index=False)
            # Attempt to generate customer-level summary stats
            try:
                required_cols = {"clv", "recency", "total_spent", "upsell_score"}
                if required_cols.issubset(full_df_model_data.columns):
                    summary = generate_customer_summary(full_df_model_data)
                else:
                    summary = {
                        "note": "Detailed summary statistics require 'clv', 'recency', 'total_spent', and 'upsell_score'.",
                        "tip": "Visit datasets.html to define or create these columns for deeper customer insights."
                    }
            except Exception as e:
                summary = {
                    "error": f"Failed to generate summary stats: {str(e)}",
                    "tip": "If you're looking for customer insights, visit datasets.html to compute columns like CLV, recency, etc."
                }

            # Create request JSON for SHAP runner
            request_json = user_dir / "request.json"
            with open(request_json, "w") as f:
                json.dump({
                    "user_id": str(user_id),
                    "model_path": str(model_path.resolve()),
                    "data_path": str(data_path.resolve()),
                    "output_dir": str(user_dir.resolve()),
                    "model_type": "classifier",
                    "target_column": target_column,
                    "save_filename": f"{user_id}_feature_importance.png",
                    "training_columns": training_columns
                }, f)
            
            # Run SHAP Visualizations subprocess
            try:
                import sys
                import subprocess
                import time
                import os

                # Set up paths
                current_dir = PathL(__file__).parent
                shap_runner_path = current_dir / "shap_runner.py"

                print(f"[SHAP DEBUG] Current directory: {current_dir}")
                print(f"[SHAP DEBUG] SHAP runner path: {shap_runner_path}")
                print(f"[SHAP DEBUG] SHAP runner exists: {shap_runner_path.exists()}")

                if not shap_runner_path.exists():
                    raise FileNotFoundError(f"shap_runner.py not found at {shap_runner_path}")

                # Create request JSON
                request_json = user_dir / "request.json"
                with open(request_json, "w") as f:
                    json.dump({
                        "user_id": str(user_id),
                        "model_path": str(model_path.resolve()),
                        "data_path": str(data_path.resolve()),
                        "output_dir": str(user_dir.resolve()),
                        "model_type": "classification",
                        "target_column": target_column,
                        "save_filename": f"{user_id}_feature_importance.png",
                        "training_columns" : training_columns
                    }, f, indent=2)

                print(f"[SHAP DEBUG] Request JSON created at: {request_json}")
                print(f"[SHAP DEBUG] JSON contents: {request_json.read_text()}")

                # Prepare subprocess command
                cmd = [sys.executable, str(shap_runner_path.resolve()), str(request_json.resolve())]
                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'
                env['PYTHONIOENCODING'] = 'utf-8'

                print(f"[SHAP DEBUG] Launching subprocess with command: {' '.join(cmd)}")

                process = subprocess.Popen(
                    cmd,
                    cwd=str(current_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                    universal_newlines=True
                )
                # Read and print every line from the child process
                for line in process.stdout:
                    print("SHAP RUNNER:", line.rstrip())

                process.wait()
                if process.returncode != 0:
                    raise RuntimeError(f"SHAP runner failed with exit code {process.returncode}")
                # Stream output
                output_lines = []
                timeout_seconds = 300
                start_time = time.time()

                while True:
                    if process.poll() is not None:
                        remaining_output = process.stdout.read()
                        if remaining_output:
                            output_lines.append(remaining_output)
                            print(f"[SHAP SUBPROCESS] {remaining_output.strip()}")
                        break

                    if time.time() - start_time > timeout_seconds:
                        print(f"[SHAP ERROR] Subprocess timed out after {timeout_seconds} seconds")
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise subprocess.TimeoutExpired(cmd, timeout_seconds)

                    line = process.stdout.readline()
                    if line:
                        output_lines.append(line)
                        print(f"[SHAP SUBPROCESS] {line.strip()}")
                    else:
                        time.sleep(0.1)

                return_code = process.returncode
                full_output = ''.join(output_lines)

                print(f"[SHAP DEBUG] Process finished with code {return_code}")
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, cmd, output=full_output)

                # Load SHAP result
                result_json_path = user_dir / "result.json"
                print(f"[SHAP DEBUG] Checking result: {result_json_path.exists()}")
                if result_json_path.exists():
                    try:
                        with open(result_json_path) as f:
                            shap_result = json.load(f)
                        print(f"[SHAP DEBUG] SHAP result loaded")

                        fi_shap_bar = shap_result.get("shap_bar")
                        fi_shap_dot = shap_result.get("shap_dot")
                        imp_df_data = shap_result.get("imp_df", [])
                        imp_df = pd.DataFrame(imp_df_data) if imp_df_data else pd.DataFrame()

                    except Exception as e:
                        print(f"[SHAP ERROR] Failed to load SHAP result: {e}")
                        fi_shap_bar, fi_shap_dot, imp_df = None, None, pd.DataFrame()
                else:
                    print(f"[SHAP ERROR] result.json not found")
                    fi_shap_bar, fi_shap_dot, imp_df = None, None, pd.DataFrame()

            except subprocess.TimeoutExpired as e:
                print(f"[⚠️] SHAP subprocess timed out: {e}")
                fi_shap_bar, fi_shap_dot, imp_df = None, None, pd.DataFrame()

            except subprocess.CalledProcessError as e:
                print(f"[⚠️] SHAP subprocess failed: {e}")
                print(f"[⚠️] Output: {e.output}")
                fi_shap_bar, fi_shap_dot, imp_df = None, None, pd.DataFrame()

            except FileNotFoundError as e:
                print(f"[⚠️] SHAP runner not found: {e}")
                fi_shap_bar, fi_shap_dot, imp_df = None, None, pd.DataFrame()

            except Exception as viz_error:
                print(f"[⚠️] SHAP visualization error: {viz_error}")
                import traceback
                traceback.print_exc()
                fi_shap_bar, fi_shap_dot, imp_df = None, None, pd.DataFrame()

            response_data = {}
            try:
                # Upload trained model
                model_filename = PathL(model_path).name
                model_supabase_path = upload_file_to_supabase(user_id, str(model_path), model_filename)

                # Upload processed dataset
                
                data_filename = PathL(data_path).name
                data_supabase_path = upload_file_to_supabase(user_id, str(data_path), data_filename)

                # Generate signed URLs for access (e.g., for frontend or gallery)
                model_url = get_file_url(model_supabase_path, expires_in=3600)
                data_url = get_file_url(data_supabase_path, expires_in=3600)

                print(f"[✅] Uploaded model: {model_url}")
                print(f"[✅] Uploaded data: {data_url}")

                # Optionally return/store these URLs in your pipeline response
                response_data.update({
                    "model_url": model_url,
                    "data_url": data_url
                })

            except Exception as upload_error:
                print(f"[⚠️] Failed to upload to Supabase: {upload_error}")
            # Save Metadata for Gallery
            from tabulate import tabulate

            # Flatten cv_scores into a list of rows
            cv_table = [
                [model, scores["accuracy"], scores["f1"], scores["precision"], scores["recall"]]
                for model, scores in cv_scores.items()
            ]

            headers = ["Model", "Accuracy", "F1", "Precision", "Recall"]
            impact_metrics = {}
            # Create entry for metadata
            if len(set(y_test)) == 2:
                tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
                impact_metrics = {
                    "true_positives": int(tp),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    "true_negatives": int(tn),
                    "n_samples": len(y_test),
                    "predicted_positive": int(sum(preds)),
                    "actual_positive": int(sum(y_test)),
                    "conversion_rate": round((tp / (tp + fp)) * 100, 2) if (tp + fp) else 0.0,
                }
            try:
                entry = {
                    "id": str(uuid.uuid4()),
                    "created_at": datetime.utcnow().isoformat(),
                    "type": "classification",
                    "dataset": dataset_name,
                    "parameters": {"drop_columns": drop_columns},
                    "test_scores": test_scores,
                    "cv_scores": cv_table,
                    "target_column": target_column,
                    "visualizations": {
                        "shap_bar": f"data:image/png;base64,{fi_shap_bar}" if fi_shap_bar else "",
                        "shap_dot": f"data:image/png;base64,{fi_shap_dot}" if fi_shap_dot else "",
                    },
                    "thumbnailData": f"data:image/png;base64,{fi_shap_bar or fi_shap_dot or ''}",
                    "imageData": f"data:image/png;base64,{fi_shap_dot or fi_shap_bar or ''}",
                    "top_features": imp_df.head(10).to_dict("records") if not imp_df.empty else [],
                    "conversion_rate": conversion_rate,
                    "impact_metrics": impact_metrics,
                    "training_columns": training_columns
                }
                
                # Add additional visualizations
                if class_dist_base64:
                    entry["visualizations"]["class_distribution"] = f"data:image/png;base64,{class_dist_base64}"
                if confusion_matrix_base64:
                    entry["visualizations"]["confusion_matrix"] = f"data:image/png;base64,{confusion_matrix_base64}"
                if classification_report_base64:
                    entry["visualizations"]["classification_report"] = f"data:image/png;base64,{classification_report_base64}"

                entry = ensure_json_serializable(entry)
                with master_db_cm() as db:
                    _append_limited_metadata(user_id, entry, db=db, max_entries=5)
            except Exception as meta_error:
                print(f"[⚠️] Metadata save error: {meta_error}")

            # Final API Response
            # Final API Response
            response_data = {
                "status": "success",
                "user_id": user_id,
                "id": str(uuid.uuid4()),
                "created_at": datetime.utcnow().isoformat(),
                "type": "classification",
                "best_model": best_name,
                "test_scores": test_scores,
                "cv_scores": cv_scores,
                "visualizations": {},
                "insights": "",
                "conversion_rate": round(conversion_rate * 100, 2),
                "target_column": target_column,
                "dataset": dataset_name,
                "parameters": {"drop_columns": drop_columns}
            }


            if fi_shap_bar:
                response_data["visualizations"]["feature_importance"] = f"data:image/png;base64,{fi_shap_bar}"
            if fi_shap_dot:
                response_data["visualizations"]["feature_importance_detailed"] = f"data:image/png;base64,{fi_shap_dot}"
            if class_dist_base64:
                response_data["visualizations"]["class_distribution"] = f"data:image/png;base64,{class_dist_base64}"
            if confusion_matrix_base64:
                response_data["visualizations"]["confusion_matrix"] = f"data:image/png;base64,{confusion_matrix_base64}"
            if classification_report_base64:
                response_data["visualizations"]["classification_report"] = f"data:image/png;base64,{classification_report_base64}"

            return response_data
            
    except Exception as e:
        print(f"[⚠️] Error in do_classification: {e}")
        raise e
    
def do_classification_predict_score(
    user_id: str,
    current_user: dict,
    file_path: str,
    drop_columns: str = ""
) -> dict:
    """
    Classification endpoint that provides probability scores and risk tiers,
    while keeping original preprocessing, feature alignment, Supabase upload,
    visualizations, and metadata logging.
    """
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    try:
        # ─────────── Load model & training columns ───────────
        model_path = f"{user_id}/{user_id}_best_classifier.pkl"
        model_bytes = download_file_from_supabase(model_path)
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            f.write(model_bytes)
            model_file_path = f.name
        model = joblib.load(model_file_path)

        training_columns_data = download_file_from_supabase(f"{user_id}/training_columns.json")
        training_columns = json.loads(training_columns_data.decode('utf-8'))

        # ─────────── Load prediction CSV ───────────
        prediction_data = download_file_from_supabase(file_path)
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as f:
            f.write(prediction_data)
            temp_csv_path = f.name

        # ─────────── Preprocessing ───────────
        pred_df = pd.read_csv(temp_csv_path)
        original_df = pred_df.copy()
        original_shape = pred_df.shape

        # Identify target column if present
        known_targets = ['target', 'label', 'class', 'y', 'Target', 'Label', 'Class']
        target_column_actual = next((c for c in known_targets if c in pred_df.columns), None)
        true_labels = pred_df[target_column_actual].copy() if target_column_actual else None

        # Drop ID/target/drop_columns
        drop_features = ['ID'] + ([target_column_actual] if target_column_actual else [])
        if drop_columns:
            drops = [c.strip() for c in drop_columns.split(",") if c.strip() and c in pred_df.columns]
            pred_df.drop(columns=drops, inplace=True)
        pred_df.drop(columns=drop_features, inplace=True, errors='ignore')

        # Preprocess and align columns
        X_pred, _, _ = preprocess_data(pred_df)
        X_pred = X_pred.reset_index(drop=True)
        missing_cols = set(training_columns) - set(X_pred.columns)
        extra_cols = set(X_pred.columns) - set(training_columns)
        for col in missing_cols: X_pred[col] = 0
        X_pred.drop(columns=list(extra_cols), inplace=True, errors='ignore')
        X_pred = X_pred.reindex(columns=training_columns, fill_value=0)

        # ─────────── Predict ───────────
        predictions = model.predict(X_pred)
        if hasattr(model, 'predict_proba'):
            prediction_probs = model.predict_proba(X_pred)
            positive_probs = prediction_probs[:, 1]  # probability of positive class
        else:
            positive_probs = predictions
            prediction_probs = None

        # Assign risk tiers
        def assign_risk(prob):
            if prob >= 0.8: return "Extreme Risk"
            if prob >= 0.5: return "High Risk"
            if prob >= 0.2: return "Medium Risk"
            return "Low Risk"
        risk_tiers = [assign_risk(p) for p in positive_probs]

        # Conversion rate calculation
        conversion_rate = None
        if true_labels is not None and hasattr(model, "classes_") and len(model.classes_) == 2:
            predicted_positive = (predictions == 1).sum()
            tp = ((true_labels == 1) & (predictions == 1)).sum()
            conversion_rate = float((tp / predicted_positive) * 100) if predicted_positive else 0.0

        # ─────────── Create output df ───────────
        output_df = original_df.copy()
        output_df['prediction'] = predictions
        output_df['confidence'] = positive_probs
        output_df['risk_tier'] = risk_tiers

        if prediction_probs is not None:
            classes = model.classes_
            for i, class_name in enumerate(classes):
                output_df[f'prob_class_{class_name}'] = prediction_probs[:, i]

        # ─────────── Visualizations ───────────
        visualizations = {}
        try:
            fig_dist = plt.figure(figsize=(8,6))
            sns.barplot(x=list(Counter(predictions).keys()), y=list(Counter(predictions).values()))
            plt.title("Prediction Distribution")
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            visualizations["prediction_distribution"] = f"data:image/png;base64,{plot_to_base64(fig_dist)}"
            plt.close(fig_dist)
        except: pass

        try:
            fig_conf = plt.figure(figsize=(8,6))
            plt.hist(positive_probs, bins=20, alpha=0.7, edgecolor='black')
            plt.axvline(np.mean(positive_probs), color='red', linestyle='--', label=f'Mean: {np.mean(positive_probs):.3f}')
            plt.title("Confidence Distribution")
            plt.xlabel("Confidence Score")
            plt.ylabel("Frequency")
            plt.legend()
            visualizations["confidence_distribution"] = f"data:image/png;base64,{plot_to_base64(fig_conf)}"
            plt.close(fig_conf)
        except: pass

        # ─────────── Save & upload predictions ───────────
        output_filename = f"predictions_{PathL(file_path).stem}.csv"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as temp_output:
            output_df.to_csv(temp_output.name, index=False)
            temp_output_path = temp_output.name

        supabase_output_path = upload_file_to_supabase(
            user_id=str(user_id),
            file_path=temp_output_path,
            filename=output_filename
        )
        signed_url = get_file_url(supabase_output_path, expires_in=3600)
        os.unlink(temp_output_path)
        os.unlink(model_file_path)
        os.unlink(temp_csv_path)

        # ─────────── Stats ───────────
        pred_stats = {
            "total_predictions": int(len(predictions)),
            "unique_classes": int(len(set(predictions))),
            "class_distribution": {str(k): int(v) for k, v in Counter(predictions).items()},
            "mean_confidence": float(np.mean(positive_probs)),
            "min_confidence": float(np.min(positive_probs)),
            "max_confidence": float(np.max(positive_probs))
        }

        response_data = {
            "status": "success",
            "user_id": user_id,
            "dataset": Path(file_path).name,
            "output_file": supabase_output_path,
            "signed_url": signed_url,
            "metrics": pred_stats,
            "data_preview": output_df.head(10).to_dict('records'),
            "conversion_rate": conversion_rate
        }
        return convert_numpy_types(response_data)

    except Exception as e:
        return {"status": "error", "user_id": user_id, "error_message": str(e), "error_type": type(e).__name__}

def do_classification_predict(
    user_id: str,
    current_user: dict,
    file_path: str,
    drop_columns: str = "",
    output_predictions: bool = True
) -> dict:

    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    try:
        # Setup paths - models are still stored locally, but data files are in Supabase
        
        model_supabase_path = f"{user_id}/{user_id}_best_classifier.pkl"

        try:
            print(f"[📦] Downloading model from Supabase: {model_supabase_path}")
            model_bytes = download_file_from_supabase(model_supabase_path)
            
            # Save model to a temporary file
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as model_file:
                model_file.write(model_bytes)
                model_file_path = model_file.name
            
            # Load model
            model = joblib.load(model_file_path)

        except Exception as e:
            raise FileNotFoundError(f"Failed to load model from Supabase for user {user_id}: {str(e)}")

        # For training columns, we'll also store this in Supabase
        training_columns_supabase_path = f"{user_id}/training_columns.json"
        # Download training columns from Supabase
        try:
            training_columns_data = download_file_from_supabase(training_columns_supabase_path)
            training_columns = json.loads(training_columns_data.decode('utf-8'))
        except Exception as e:
            raise FileNotFoundError(f"Training columns metadata not found for user {user_id} in Supabase: {str(e)}")
        
        # Download prediction data from Supabase
        try:
            prediction_data = download_file_from_supabase(file_path)
            dataset_name = os.path.basename(file_path)
        except Exception as e:
            raise FileNotFoundError(f"Prediction file not found in Supabase: {file_path}. Error: {str(e)}")

        # Create temporary file to load CSV data
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as temp_file:
            temp_file.write(prediction_data)
            temp_csv_path = temp_file.name
        
        try:
            # Load and preprocess prediction data
            print(f"[📊] Loading prediction data from Supabase: {file_path}")
            pred_df = pd.read_csv(temp_csv_path)
            original_shape = pred_df.shape
            
            # Store original dataframe for final output
            original_df = pred_df.copy()
            
            # Store original ID column if present
            id_column = None
            if 'ID' in pred_df.columns:
                id_column = pred_df['ID'].copy()

            # Save original ID column if available
            id_column = pred_df['ID'].copy() if 'ID' in pred_df.columns else None

            # Save target column (if known or common name exists)
            known_targets = ['target', 'label', 'class', 'y', 'Target', 'Label', 'Class']
            target_column_actual = None
            for col in known_targets:
                if col in pred_df.columns:
                    target_column_actual = col
                    break

            # Save target values if found (for conversion rate calc)
            true_labels = pred_df[target_column_actual].copy() if target_column_actual else None

            # Drop ID and target ONLY from features going into model
            drop_features = ['ID']
            if target_column_actual:
                drop_features.append(target_column_actual)

            # Drop specified columns
            if drop_columns:
                drops = [c.strip() for c in drop_columns.split(",") if c.strip() and c in pred_df.columns]
                pred_df.drop(columns=drops, inplace=True)
                print(f"[🗑️] Dropped columns: {drops}")
            
            pred_df.drop(columns=drop_features, inplace=True, errors='ignore')

            # Preprocess the data using the same function as training
            print("[🔄] Preprocessing prediction data...")
            X_pred, _, _ = preprocess_data(pred_df)
            X_pred = X_pred.reset_index(drop=True)

            # Align columns with training data
            print("[🔧] Aligning features with training data...")
            missing_cols = set(training_columns) - set(X_pred.columns)
            extra_cols = set(X_pred.columns) - set(training_columns)
            
            # Add missing columns with default values
            for col in missing_cols:
                X_pred[col] = 0
                
            # Remove extra columns
            X_pred.drop(columns=list(extra_cols), inplace=True, errors='ignore')
            
            # Reorder columns to match training
            X_pred = X_pred.reindex(columns=training_columns, fill_value=0)
            
            print(f"[✅] Feature alignment complete: {len(training_columns)} features")
            predictions = model.predict(X_pred)
            
            # Conversion Rate Calculation (only if target exists and binary)
            conversion_rate = None
            if true_labels is not None and hasattr(model, "classes_") and len(model.classes_) == 2:
                predicted_positive = (predictions == 1).sum()
                tp = ((true_labels == 1) & (predictions == 1)).sum()
                conversion_rate = float((tp / predicted_positive) * 100) if predicted_positive else 0.0

            prediction_probs = None
            if hasattr(model, 'predict_proba'):
                try:
                    prediction_probs = model.predict_proba(X_pred)
                    max_probs = np.max(prediction_probs, axis=1)
                except:
                    prediction_probs = None
                    max_probs = None
            else:
                max_probs = None
            
            # Create output dataframe by adding predictions to original data
            output_df = original_df.copy()
            output_df['prediction'] = predictions
            
            if max_probs is not None:
                output_df['confidence'] = max_probs
            # ───── Customer Summary Stats (Optional) ───── #


            # Add probability columns for each class if available
            if prediction_probs is not None:
                classes = model.classes_ if hasattr(model, 'classes_') else np.unique(predictions)
                for i, class_name in enumerate(classes):
                    output_df[f'prob_class_{class_name}'] = prediction_probs[:, i]
            
            # Generate visualizations
            visualizations = {}
            
            # 1. Prediction distribution
            try:
                pred_counts = Counter(predictions)
                fig_dist = plt.figure(figsize=(8, 6))
                sns.barplot(x=list(pred_counts.keys()), y=list(pred_counts.values()))
                plt.title("Prediction Distribution")
                plt.xlabel("Predicted Class")
                plt.ylabel("Count")
                plt.xticks(rotation=45)
                visualizations["prediction_distribution"] = f"data:image/png;base64,{plot_to_base64(fig_dist)}"
                plt.close(fig_dist)
            except Exception as e:
                print(f"[⚠️] Prediction distribution plot failed: {e}")
            
            # 2. Confidence distribution (if available)
            if max_probs is not None:
                try:
                    fig_conf = plt.figure(figsize=(8, 6))
                    plt.hist(max_probs, bins=20, alpha=0.7, edgecolor='black')
                    plt.title("Prediction Confidence Distribution")
                    plt.xlabel("Confidence Score")
                    plt.ylabel("Frequency")
                    plt.axvline(np.mean(max_probs), color='red', linestyle='--', 
                               label=f'Mean: {np.mean(max_probs):.3f}')
                    plt.legend()
                    visualizations["confidence_distribution"] = f"data:image/png;base64,{plot_to_base64(fig_conf)}"
                    plt.close(fig_conf)
                except Exception as e:
                    print(f"[⚠️] Confidence distribution plot failed: {e}")
            
            # Save predictions to temporary file, then upload to Supabase
            output_filename = f"predictions_{PathL(file_path).stem}.csv"
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as temp_output:
                output_df.to_csv(temp_output.name, index=False)
                temp_output_path = temp_output.name
            
            try:
                # Upload predictions to Supabase
                supabase_output_path = upload_file_to_supabase(
                    user_id=str(user_id),
                    file_path=temp_output_path,
                    filename=output_filename
                )

                # Get signed URL as a string
                signed_url = get_file_url(supabase_output_path, expires_in=3600)

                print(f"[💾] Predictions uploaded to Supabase: {supabase_output_path}")

            finally:
                # Clean up temporary output file
                os.unlink(temp_output_path)
                if os.path.exists(model_file_path):
                    os.unlink(model_file_path)

            
            # Calculate prediction statistics - CONVERT NUMPY TYPES HERE
            pred_stats = {
                "total_predictions": int(len(predictions)),
                "unique_classes": int(len(set(predictions))),
                "class_distribution": {str(k): int(v) for k, v in Counter(predictions).items()},
                "most_common_class": str(Counter(predictions).most_common(1)[0][0]),
                "original_data_shape": [int(x) for x in original_shape],
                "processed_data_shape": [int(x) for x in X_pred.shape]
            }
            
            if max_probs is not None:
                pred_stats.update({
                    "mean_confidence": float(np.mean(max_probs)),
                    "min_confidence": float(np.min(max_probs)),
                    "max_confidence": float(np.max(max_probs)),
                    "low_confidence_count": int(np.sum(max_probs < 0.7)),
                    "high_confidence_count": int(np.sum(max_probs > 0.9))
                })
            try:
                entry = {
                    "id": str(uuid.uuid4()),
                    "created_at": datetime.utcnow().isoformat(),
                    "type": "classification_prediction",
                    "target_col": target_column_actual,
                    "dataset": dataset_name,
                    "conversion_rate": conversion_rate if true_labels is not None else None,
                    "metrics": pred_stats,
                    "output_file": supabase_output_path,
                    "signed_url": signed_url,
                    "visualizations": visualizations,
                    "thumbnailData": visualizations.get("prediction_distribution"),
                    "imageData": visualizations.get("confidence_distribution"),
                    "feature_alignment": {
                        "training_features": int(len(training_columns)),
                        "prediction_features": int(len(X_pred.columns)),
                        "missing_features_added": list(missing_cols),
                        "extra_features_removed": list(extra_cols)
                    }
                }


                entry = ensure_json_serializable(entry)
                with master_db_cm() as db:
                    _append_limited_metadata(user_id, entry, db=db, max_entries=5)
            except Exception as meta_error:
                print(f"[⚠️] Metadata save error: {meta_error}")


            # Prepare response - CONVERT ALL NUMPY TYPES
            response_data = {
                "status": "success",
                "user_id": user_id,
                "created_at": datetime.utcnow().isoformat(),
                "type": "classification",
                "dataset": dataset_name,
                "parameters": {"drop_columns": drop_columns},
                "file": file_path,
                "output_file": supabase_output_path,
                "output_file_name": output_filename,
                "signed_url": signed_url,
                "metrics": pred_stats,
                "data_preview": output_df.head(10).to_dict('records'),  # Show first 10 rows as preview
                "total_rows": len(output_df),
                "columns_added": ["prediction"] + ([f"prob_class_{c}" for c in model.classes_] if hasattr(model, 'classes_') and prediction_probs is not None else []) + (["confidence"] if max_probs is not None else []),
                "visualizations": visualizations,
                "feature_alignment": {
                    "training_features": int(len(training_columns)),
                    "prediction_features": int(len(X_pred.columns)),
                    "missing_features_added": list(missing_cols),
                    "extra_features_removed": list(extra_cols)
                },
                "conversion_rate": conversion_rate if true_labels is not None else None
            }
            
            # Convert all numpy types in the response
            response_data = convert_numpy_types(response_data)
            
            print(f"[🎉] Prediction completed successfully!")
            print(f"    • Total predictions: {len(predictions)}")
            print(f"    • Unique classes: {len(set(predictions))}")
            if max_probs is not None:
                print(f"    • Mean confidence: {np.mean(max_probs):.3f}")
            print(f"    • Output file: {supabase_output_path}")
            print(f"    • Signed URL: {signed_url}")
            
            return response_data
            
        finally:
            # Clean up temporary CSV file
            os.unlink(temp_csv_path)
        
    except Exception as e:
        print(f"[❌] Prediction failed: {str(e)}")
        return {
            "status": "error",
            "user_id": user_id,
            "error_message": str(e),
            "error_type": type(e).__name__
        }
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def do_clustering(
    user_id: str,
    current_user: dict = None,
    file_path: str = None,
    train_path: str = None,
    test_path: str = None,
    target_column: str = None,
    time_column: str = None,
    drop_columns: str = ""
):
    import json
    import subprocess
    import tempfile
    import os
    import pandas as pd
    from pathlib import Path as PathL
    import joblib
    import uuid
    from datetime import datetime
    import logging
    
    try:
        # Validate upload mode
        if file_path and (train_path or test_path):
            raise ValueError("Provide either file_path or both train_path+test_path, not both.")
        if (train_path and not test_path) or (test_path and not train_path):
            raise ValueError("Both train_path and test_path must be provided together.")
        if not file_path and not (train_path and test_path):
            raise ValueError("Provide either a full dataset (file_path) or both train_path+test_path.")
        
        # Handle file downloads from Supabase
        # Note: file_path, train_path, test_path are now Supabase storage paths (e.g., "user123/dataset.csv")
        local_file_path = None
        local_train_path = None
        local_test_path = None
        temp_files = []
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download files from Supabase to temporary locations
            if file_path:
                # Download single file from Supabase using the storage path
                file_bytes = download_file_from_supabase(file_path)
                local_file_path = os.path.join(temp_dir, os.path.basename(file_path))
                with open(local_file_path, 'wb') as f:
                    f.write(file_bytes)
                temp_files.append(local_file_path)
            
            if train_path and test_path:
                # Download train and test files from Supabase using the storage paths
                train_bytes = download_file_from_supabase(train_path)
                test_bytes = download_file_from_supabase(test_path)
                
                local_train_path = os.path.join(temp_dir, os.path.basename(train_path))
                local_test_path = os.path.join(temp_dir, os.path.basename(test_path))
                
                with open(local_train_path, 'wb') as f:
                    f.write(train_bytes)
                with open(local_test_path, 'wb') as f:
                    f.write(test_bytes)
                    
                temp_files.extend([local_train_path, local_test_path])
            
            # Prepare model directory
            model_dir = os.path.join(temp_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            
            # Create user directory for outputs
            user_dir = PathL(temp_dir) / "user_outputs"
            user_dir.mkdir(exist_ok=True)
            
            # Process data based on single file or train/test split
            if local_file_path:
                # Single file mode
                df = pd.read_csv(local_file_path)
                if drop_columns:
                    drops = [c.strip() for c in drop_columns.split(",") if c.strip() and c in df.columns]
                    df.drop(columns=drops, inplace=True)
                
                if target_column and target_column in df.columns:
                    df.drop(columns=[target_column], inplace=True)
                
                dataset_name = os.path.basename(file_path)
            else:
                # Train/test pair mode
                train_df = pd.read_csv(local_train_path)
                test_df = pd.read_csv(local_test_path)
                df = pd.concat([train_df, test_df], ignore_index=True)
                
                if drop_columns:
                    drops = [c.strip() for c in drop_columns.split(",") if c.strip()]
                    df.drop(columns=[c for c in drops if c in df.columns], inplace=True)
                
                if target_column and target_column in df.columns:
                    df.drop(columns=[target_column], inplace=True)
                
                dataset_name = f"{os.path.basename(train_path)} + {os.path.basename(test_path)}"
            
            # Drop ID column if present
            df.drop(columns=["ID"], inplace=True, errors='ignore')
            
            # Validate numeric columns
            numeric_df = df.select_dtypes(include=["number"]).fillna(0)
            if numeric_df.empty:
                return {
                    "status": "error",
                    "message": "No numeric columns available for clustering after preprocessing."
                }
            
            # Scale the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)
            
            # Find optimal k
            max_k = min(10, len(df) - 1)
            best_k, distortions, silhouette_scores = find_optimal_k(scaled_data, max_k)
            
            # Run clustering
            clusters, kmeans = run_kmeans(scaled_data, best_k)
            df["cluster"] = clusters
            
            # Metrics
            clustering_metrics = {
                "silhouette_score": float(silhouette_score(scaled_data, clusters)),
                "calinski_harabasz_score": float(calinski_harabasz_score(scaled_data, clusters)),
                "davies_bouldin_score": float(davies_bouldin_score(scaled_data, clusters)),
                "inertia": float(kmeans.inertia_)
            }
            
            # Identify numeric columns (excluding the cluster column)
            numeric_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != "cluster"]
            
            # Cluster sizes
            cluster_counts = df["cluster"].value_counts().sort_index()
            cluster_sizes = cluster_counts.to_dict()
            
            # Ensure we only use numeric columns that are still present in the DataFrame
            available_numeric_cols = df.select_dtypes(include=["number"]).columns.drop("cluster", errors="ignore").tolist()
            
            # Run aggregation safely
            cluster_stats = {}
            for stat_func in ["mean", "median", "std", "min", "max"]:
                try:
                    stat_df = df[available_numeric_cols + ["cluster"]].groupby("cluster").agg(stat_func)
                    for cluster_id, row in stat_df.iterrows():
                        for feature, value in row.items():
                            cluster_stats.setdefault(cluster_id, {})[f"{feature}_{stat_func}"] = round(value, 2)
                except KeyError as e:
                    logging.warning(f"Skipped aggregation '{stat_func}' due to missing columns: {e}")
            
            # Global means
            global_means = df[numeric_cols].mean()
            
            # Cluster feature differences
            cluster_feature_diffs = (
                df.groupby("cluster")[numeric_cols].mean() - global_means
            ).abs()
            
            cluster_feature_diffs_dict = cluster_feature_diffs.to_dict(orient="index")
            
            # Insights per cluster
            cluster_insights = {}
            for cluster_id, diffs in cluster_feature_diffs.iterrows():
                top_feats = diffs.sort_values(ascending=False).head(3)
                cluster_insights[str(cluster_id)] = [
                    f"{feat} is {round(df[df['cluster'] == cluster_id][feat].mean() - global_means[feat], 2)} higher than average"
                    for feat in top_feats.index
                ]
            # ───── Optional: Generate Customer-Level Summary Stats ───── #

            # Visualization
            cluster_viz_base64 = ""
            if numeric_df.shape[1] > 1:
                try:
                    pca = PCA(n_components=2)
                    reduced_features = pca.fit_transform(scaled_data)
                    
                    cluster_fig = plt.figure(figsize=(10, 6))
                    for i in range(best_k):
                        plt.scatter(
                            reduced_features[clusters == i, 0],
                            reduced_features[clusters == i, 1],
                            label=f"Cluster {i}"
                        )
                    plt.legend()
                    plt.title(f"Customer Clusters (K={best_k})")
                    plt.xlabel("Principal Component 1")
                    plt.ylabel("Principal Component 2")
                    cluster_viz_base64 = plot_to_base64(cluster_fig)
                    plt.close(cluster_fig)
                except Exception as e:
                    print(f"[⚠️] Cluster visualization failed: {e}")
                    cluster_viz_base64 = ""
            else:
                try:
                    cluster_fig = plt.figure(figsize=(10, 6))
                    plt.hist(df['cluster'], bins=best_k)
                    plt.title(f'Cluster Distribution (K={best_k})')
                    plt.xlabel('Cluster')
                    plt.ylabel('Count')
                    cluster_viz_base64 = plot_to_base64(cluster_fig)
                    plt.close(cluster_fig)
                except Exception as e:
                    print(f"[⚠️] Cluster distribution plot failed: {e}")
                    cluster_viz_base64 = ""
            
            # Elbow method visualization
            elbow_base64 = ""
            try:
                elbow_fig = plt.figure(figsize=(10, 6))
                plt.plot(range(2, max_k + 1), distortions, 'bx-')
                plt.xlabel("Number of clusters (k)")
                plt.ylabel("Distortion")
                plt.title("Elbow Method For Optimal k")
                elbow_base64 = plot_to_base64(elbow_fig)
                plt.close(elbow_fig)
            except Exception as e:
                print(f"[⚠️] Elbow method plot failed: {e}")
                elbow_base64 = ""
                        
                        # ─────── Save clustered data and model locally ───────
            clustered_data_path = PathL(model_dir) / f"{user_id}_clustered_data.csv"
            df.to_csv(clustered_data_path, index=False)

            model_path = PathL(model_dir) / f"{user_id}_kmeans_model.pkl"
            joblib.dump(kmeans, model_path)

            # ─────── Upload model + data to Supabase ───────
            model_filename = model_path.name
            data_filename = clustered_data_path.name

            model_upload_path = upload_file_to_supabase(
                user_id=user_id,
                file_path=str(model_path),
                filename=model_filename
            )

            data_upload_path = upload_file_to_supabase(
                user_id=user_id,
                file_path=str(clustered_data_path),
                filename=data_filename
            )

            # Optional: Signed URLs
            model_url = get_file_url(model_upload_path)
            data_url = get_file_url(data_upload_path)
                        
            # Create request JSON for additional processing if needed
            request_json = user_dir / "request.json"
            with open(request_json, "w") as f:
                json.dump({
                    "user_id": str(user_id),
                    "model_path": model_upload_path,       # Supabase storage path
                    "model_url": model_url,                # Signed URL for download
                    "data_path": data_upload_path,         # Supabase storage path
                    "data_url": data_url,                  # Signed URL for download
                    "output_dir": str(user_dir.resolve()), # Local output dir for logs/debug
                    "model_type": "clustering",
                    "target_column": target_column,
                    "drop_columns": drop_columns,
                    "optimal_k": best_k
                }, f)

            logger.info(f"✅ Clustering completed for user_id: {user_id} | Clusters: {best_k}")
            
            response_data = {
                "status": "success",
                "user_id": user_id,
                "id": str(uuid.uuid4()),
                "created_at": datetime.utcnow().isoformat(),
                "type": "clustering",
                "dataset": dataset_name,
                "parameters": {
                    "target_column": target_column,
                    "drop_columns": drop_columns,
                    "optimal_k": best_k
                },
                "message": f"Clustering completed with {best_k} clusters",
                "cluster_counts": {str(i): int(count) for i, count in cluster_counts.items()},
                "visualizations": {},
                "insights": "",
                "metrics": clustering_metrics,
                "cluster_stats": cluster_stats,
                "cluster_feature_differences": cluster_feature_diffs_dict,
                "cluster_insights": cluster_insights,
                # ✅ Add model and data file references
                "model_url": model_url,
                "data_url": data_url,
                "model_path": model_upload_path,
                "data_path": data_upload_path
            }

                    
            # Add visualizations if they exist
            if cluster_viz_base64:
                response_data["visualizations"]["cluster_viz"] = f"data:image/png;base64,{cluster_viz_base64}"
            if elbow_base64:
                response_data["visualizations"]["elbow_method"] = f"data:image/png;base64,{elbow_base64}"
            try:
                entry = {
                    "id": response_data["id"],  # Use the same one
                    "created_at": datetime.utcnow().isoformat(),
                    "type": "segmentation",
                    "dataset": dataset_name,
                    "parameters": {
                        "target_column": target_column,
                        "drop_columns": drop_columns,
                        "optimal_k": best_k
                    },
                    "cluster_counts": {str(i): int(count) for i, count in cluster_counts.items()},
                    "thumbnailData": f"data:image/png;base64,{cluster_viz_base64}" if cluster_viz_base64 else "",
                    "imageData": f"data:image/png;base64,{cluster_viz_base64 or elbow_base64}",
                    "visualizations": {
                        "cluster_viz": f"data:image/png;base64,{cluster_viz_base64}" if cluster_viz_base64 else "",
                        "elbow_method": f"data:image/png;base64,{elbow_base64}" if elbow_base64 else ""
                    },
                    "segments_summary": [{"cluster": int(i), "count": int(count)} for i, count in cluster_counts.items()],
                    "metrics": clustering_metrics,
                    "cluster_stats": cluster_stats,
                    "cluster_feature_differences": cluster_feature_diffs_dict,
                    # ✅ Add uploaded file references
                    "model_url": model_url,
                    "data_url": data_url,
                    "model_path": model_upload_path,
                    "data_path": data_upload_path
                }

                entry = ensure_json_serializable(entry)
                with master_db_cm() as db:
                    _append_limited_metadata(user_id, entry, db=db, max_entries=5)
            except Exception as meta_error:
                print(f"[⚠️] Metadata save error: {meta_error}")

            
            return response_data
            
    except Exception as e:
        print(f"[⚠️] Error in do_clustering: {e}")
        raise e
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import json
import tempfile
import os
import pandas as pd
from pathlib import Path as PathL

def do_segment_analysis(
    user_id: str,
    current_user: dict,
    file_path: str = None,
    train_path: str = None,
    test_path: str = None,
    target_column: str = None,
    drop_columns: str = ""
) -> dict:
    """Main segmentation analysis function with Supabase integration"""
    
    import json
    import subprocess
    import tempfile
    import os
    import pandas as pd
    from pathlib import Path as PathL
    import joblib
    import uuid
    from datetime import datetime
    import logging
    
    try:
        # ───────────── Validate upload mode (same as classification) ─────────────
        if file_path and (train_path or test_path):
            raise ValueError("Provide either file_path or both train_path+test_path, not both.")
        if (train_path and not test_path) or (test_path and not train_path):
            raise ValueError("Both train_path and test_path must be provided together.")
        if not file_path and not (train_path and test_path):
            raise ValueError("Provide either a full dataset (file_path) or both train_path+test_path.")

        # Handle file downloads from Supabase
        # Note: file_path, train_path, test_path are now Supabase storage paths (e.g., "user123/dataset.csv")
        local_file_path = None
        local_train_path = None
        local_test_path = None
        temp_files = []
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download files from Supabase to temporary locations
            if file_path:
                # Download single file from Supabase using the storage path
                file_bytes = download_file_from_supabase(file_path)
                local_file_path = os.path.join(temp_dir, os.path.basename(file_path))
                with open(local_file_path, 'wb') as f:
                    f.write(file_bytes)
                temp_files.append(local_file_path)
            
            if train_path and test_path:
                # Download train and test files from Supabase using the storage paths
                train_bytes = download_file_from_supabase(train_path)
                test_bytes = download_file_from_supabase(test_path)
                
                local_train_path = os.path.join(temp_dir, os.path.basename(train_path))
                local_test_path = os.path.join(temp_dir, os.path.basename(test_path))
                
                with open(local_train_path, 'wb') as f:
                    f.write(train_bytes)
                with open(local_test_path, 'wb') as f:
                    f.write(test_bytes)
                    
                temp_files.extend([local_train_path, local_test_path])
            
            # Prepare model directory
            model_dir = os.path.join(temp_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            
            # Create user directory for outputs
            user_dir = PathL(temp_dir) / "user_outputs"
            user_dir.mkdir(exist_ok=True)

            # ───────────── Load data (match classification logic) ─────────────
            if local_file_path:
                # Single file mode
                df = pd.read_csv(local_file_path)
                dataset_name = os.path.basename(file_path)
            else:
                # Train/test pair mode
                train_df = pd.read_csv(local_train_path)
                test_df = pd.read_csv(local_test_path)
                df = pd.concat([train_df, test_df], ignore_index=True)
                dataset_name = f"{os.path.basename(train_path)} + {os.path.basename(test_path)}"

            # ───────────── Prepare data for clustering ─────────────
            if drop_columns:
                drops = [c.strip() for c in drop_columns.split(",") if c.strip() and c in df.columns]
                df.drop(columns=drops, inplace=True)

            if target_column and target_column in df.columns:
                df.drop(columns=[target_column], inplace=True)

            # Drop ID column if present
            df.drop(columns=['ID'], inplace=True, errors='ignore')

            # Select numeric columns only (same as clustering)
            numeric_df = df.select_dtypes(include=["number"]).fillna(0)
            if numeric_df.empty:
                return {
                    "status": "error",
                    "message": "No numeric columns available for clustering after preprocessing."
                }

            # ───────────── Scale data ─────────────
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)

            # ───────────── Find optimal k ─────────────
            max_k = min(10, len(df) - 1)
            best_k, distortions, silhouette_scores = find_optimal_k(scaled_data, max_k)

            # ───────────── Run KMeans ─────────────
            clusters, kmeans = run_kmeans(scaled_data, best_k)
            df["cluster"] = clusters

            # ───────────── Calculate metrics (same as clustering) ─────────────
            clustering_metrics = {
                "silhouette_score": float(silhouette_score(scaled_data, clusters)),
                "calinski_harabasz_score": float(calinski_harabasz_score(scaled_data, clusters)),
                "davies_bouldin_score": float(davies_bouldin_score(scaled_data, clusters)),
                "inertia": float(kmeans.inertia_)
            }

            # ───────────── Visualizations (FIXED) ─────────────
            cluster_viz_base64 = ""
            
            # Fix: Use the same logic as clustering
            if numeric_df.shape[1] > 1:
                try:
                    pca = PCA(n_components=2)
                    reduced_features = pca.fit_transform(scaled_data)

                    cluster_fig = plt.figure(figsize=(10, 6))
                    for i in range(best_k):
                        plt.scatter(
                            reduced_features[clusters == i, 0],
                            reduced_features[clusters == i, 1],
                            label=f"Segment {i}"
                        )
                    plt.legend()
                    plt.title(f"Customer Segments (K={best_k})")
                    plt.xlabel("Principal Component 1")
                    plt.ylabel("Principal Component 2")
                    cluster_viz_base64 = plot_to_base64(cluster_fig)
                    plt.close(cluster_fig)
                except Exception as e:
                    print(f"[⚠️] Segment visualization failed: {e}")
                    cluster_viz_base64 = ""
            else:
                try:
                    # Single feature case - use histogram like clustering
                    cluster_fig = plt.figure(figsize=(10, 6))
                    plt.hist(df['cluster'], bins=best_k)
                    plt.title(f'Segment Distribution (K={best_k})')
                    plt.xlabel('Segment')
                    plt.ylabel('Count')
                    cluster_viz_base64 = plot_to_base64(cluster_fig)
                    plt.close(cluster_fig)
                except Exception as e:
                    print(f"[⚠️] Segment distribution plot failed: {e}")
                    cluster_viz_base64 = ""

            # Elbow method plot
            elbow_base64 = ""
            try:
                elbow_fig = plt.figure(figsize=(10, 6))
                plt.plot(range(2, max_k + 1), distortions, 'bx-')
                plt.xlabel('Number of clusters (k)')
                plt.ylabel('Distortion')
                plt.title('Elbow Method For Optimal k')
                elbow_base64 = plot_to_base64(elbow_fig)
                plt.close(elbow_fig)
            except Exception as e:
                print(f"[⚠️] Elbow method plot failed: {e}")
                elbow_base64 = ""

            # ───────────── Segment analysis (same as clustering) ─────────────
            cluster_counts = df["cluster"].value_counts().sort_index()
            cluster_sizes = cluster_counts.to_dict()

            # Identify numeric columns (excluding the cluster column)
            numeric_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != "cluster"]

            # Ensure we only use numeric columns that are still present in the DataFrame
            available_numeric_cols = df.select_dtypes(include=["number"]).columns.drop("cluster", errors="ignore").tolist()
            
            # Calculate cluster statistics
            cluster_stats = {}
            for stat_func in ["mean", "median", "std", "min", "max"]:
                try:
                    stat_df = df[available_numeric_cols + ["cluster"]].groupby("cluster").agg(stat_func)
                    for cluster_id, row in stat_df.iterrows():
                        for feature, value in row.items():
                            cluster_stats.setdefault(cluster_id, {})[f"{feature}_{stat_func}"] = round(value, 2)
                except KeyError as e:
                    logging.warning(f"Skipped aggregation '{stat_func}' due to missing columns: {e}")

            # Global means and feature differences
            global_means = df[numeric_cols].mean()
            cluster_feature_diffs = (
                df.groupby("cluster")[numeric_cols].mean() - global_means
            ).abs()
            cluster_feature_diffs_dict = cluster_feature_diffs.to_dict(orient="index")

            # Insights per cluster
            cluster_insights = {}
            for cluster_id, diffs in cluster_feature_diffs.iterrows():
                top_feats = diffs.sort_values(ascending=False).head(3)
                cluster_insights[str(cluster_id)] = [
                    f"{feat} is {round(df[df['cluster'] == cluster_id][feat].mean() - global_means[feat], 2)} higher than average"
                    for feat in top_feats.index
                ]

            # ───────────── Save data and model + Upload to Supabase ─────────────
            segmented_data_path = PathL(model_dir) / f"{user_id}_segmented_data.csv"
            df.to_csv(segmented_data_path, index=False)

            model_path = PathL(model_dir) / f"{user_id}_kmeans_model.pkl"
            scaler_path = PathL(model_dir) / f"{user_id}_scaler.pkl"
            
            joblib.dump(kmeans, model_path)
            joblib.dump(scaler, scaler_path)

            # ─────── Upload model, scaler, and data to Supabase ───────
            model_filename = model_path.name
            scaler_filename = scaler_path.name
            data_filename = segmented_data_path.name

            model_upload_path = upload_file_to_supabase(
                user_id=user_id,
                file_path=str(model_path),
                filename=model_filename
            )

            scaler_upload_path = upload_file_to_supabase(
                user_id=user_id,
                file_path=str(scaler_path),
                filename=scaler_filename
            )

            data_upload_path = upload_file_to_supabase(
                user_id=user_id,
                file_path=str(segmented_data_path),
                filename=data_filename
            )

            # Optional: Get signed URLs
            model_url = get_file_url(model_upload_path)
            scaler_url = get_file_url(scaler_upload_path)
            data_url = get_file_url(data_upload_path)

            # Create request JSON for additional processing if needed
            request_json = user_dir / "request.json"
            with open(request_json, "w") as f:
                json.dump({
                    "user_id": str(user_id),
                    "model_path": str(model_path.resolve()),
                    "scaler_path": str(scaler_path.resolve()),
                    "data_path": str(segmented_data_path.resolve()),
                    "output_dir": str(user_dir.resolve()),
                    "model_type": "segmentation",
                    "target_column": target_column,
                    "drop_columns": drop_columns,
                    "optimal_k": best_k
                }, f)

            logger.info(f"✅ Segmentation completed for user_id: {user_id} | Segments: {best_k}")

            # ───────────── Build response (match clustering structure) ─────────────
            response_data = {
                "status": "success",
                "user_id": user_id,
                "id": str(uuid.uuid4()),
                "created_at": datetime.utcnow().isoformat(),
                "type": "segmentation",
                "dataset": dataset_name,
                "parameters": {
                    "target_column": target_column,
                    "drop_columns": drop_columns,
                    "optimal_k": best_k
                },
                "message": f"Segmentation completed with {best_k} segments",
                "cluster_counts": {str(i): int(count) for i, count in cluster_counts.items()},
                "visualizations": {},
                "insights": "",
                "metrics": clustering_metrics,
                "cluster_stats": cluster_stats,
                "cluster_feature_differences": cluster_feature_diffs_dict,
                "cluster_insights": cluster_insights,
                "optimal_k": best_k,
                "total_records": len(df),
                "features_used": list(numeric_cols),
                # ✅ Add uploaded file references
                "model_url": model_url,
                "scaler_url": scaler_url,
                "data_url": data_url,
                "model_path": model_upload_path,
                "scaler_path": scaler_upload_path,
                "data_path": data_upload_path
            }

            # Add visualizations if they exist
            if cluster_viz_base64:
                response_data["visualizations"]["cluster_viz"] = f"data:image/png;base64,{cluster_viz_base64}"
            if elbow_base64:
                response_data["visualizations"]["elbow_method"] = f"data:image/png;base64,{elbow_base64}"
            try:
                # ───────────── Save metadata for gallery ─────────────
                entry = {
                    "id": str(uuid.uuid4()),
                    "created_at": datetime.utcnow().isoformat(),
                    "type": "segmentation",
                    "dataset": dataset_name,
                    "parameters": {
                        "target_column": target_column,
                        "drop_columns": drop_columns,
                        "optimal_k": best_k
                    },
                    "cluster_counts": {str(i): int(count) for i, count in cluster_counts.items()},
                    "thumbnailData": f"data:image/png;base64,{cluster_viz_base64}" if cluster_viz_base64 else "",
                    "imageData": f"data:image/png;base64,{cluster_viz_base64 or elbow_base64}",
                    "visualizations": {
                        "cluster_visualization": f"data:image/png;base64,{cluster_viz_base64}" if cluster_viz_base64 else "",
                        "elbow_method": f"data:image/png;base64,{elbow_base64}" if elbow_base64 else ""
                    },
                    "segments_summary": [{"cluster": int(i), "count": int(count)} for i, count in cluster_counts.items()],
                    "metrics": clustering_metrics,
                    "cluster_stats": cluster_stats,
                    "cluster_feature_differences": cluster_feature_diffs_dict,
                    # ✅ Add uploaded file references to metadata
                    "model_url": model_url,
                    "scaler_url": scaler_url,
                    "data_url": data_url,
                    "model_path": model_upload_path,
                    "scaler_path": scaler_upload_path,
                    "data_path": data_upload_path
                }

                entry = ensure_json_serializable(entry)
                with master_db_cm() as db:
                    _append_limited_metadata(user_id, entry, db=db, max_entries=5)
            except Exception as meta_error:
                print(f"[⚠️] Metadata save error: {meta_error}")


            return response_data

    except Exception as e:
        logger.error(f"❌ Error in do_segment_analysis for user_id {user_id}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}
def generate_segment_insights(stats: dict, revenue_per_customer: float = 1200) -> dict:
    insights = {}
    for segment, metrics in stats.items():
        summary = []
        if not metrics:
            continue

        # Top standout features for the segment (mean-based)
        sorted_feats = sorted(
            [(k.replace("_mean", ""), v) for k, v in metrics.items() if "_mean" in k],
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]  # Top 3

        for feat, val in sorted_feats:
            if val > 0:
                summary.append(f"{feat.replace('_', ' ').title()} is higher than average by {val}")
            elif val < 0:
                summary.append(f"{feat.replace('_', ' ').title()} is lower than average by {abs(val)}")

        # Optional financial insight
        est_value = revenue_per_customer * 0.1  # heuristic
        summary.append(f"🎯 Targeting this segment could generate up to ${round(est_value, 2)} in additional revenue per customer.")

        insights[segment] = summary
    return insights
def do_label_clusters(
    user_id: str,
    current_user: dict = None,
    file_path: str = None,
    train_path: str = None,
    test_path: str = None,
    feature_columns: str = ""
) -> dict:
    """Main cluster labeling function with Supabase integration"""
    
    import json
    import tempfile
    import os
    import pandas as pd
    from pathlib import Path as PathL
    import uuid
    from datetime import datetime
    import logging

    try:
        # ───────────── Validate upload mode (same as segmentation) ─────────────
        if file_path and (train_path or test_path):
            raise ValueError("Provide either file_path or both train_path+test_path, not both.")
        if (train_path and not test_path) or (test_path and not train_path):
            raise ValueError("Both train_path and test_path must be provided together.")
        if not file_path and not (train_path and test_path):
            raise ValueError("Provide either a full dataset (file_path) or both train_path+test_path.")

        # Handle file downloads from Supabase
        # Note: file_path, train_path, test_path are now Supabase storage paths (e.g., "user123/dataset.csv")
        local_file_path = None
        local_train_path = None
        local_test_path = None
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create user directory for outputs
            user_dir = PathL(temp_dir) / "user_outputs"
            user_dir.mkdir(exist_ok=True)
            
            # Create models directory
            model_dir = os.path.join(temp_dir, "models")
            os.makedirs(model_dir, exist_ok=True)

            # ───────────── Load clustered data from Supabase ─────────────
            # First, try to find existing clustered data in Supabase
            clustered_data_path = None
            segmented_data_path = None
            
            # Try to download clustered data
            try:
                clustered_storage_path = f"{user_id}/{user_id}_clustered_data.csv"
                clustered_data_bytes = download_file_from_supabase(clustered_storage_path)
                clustered_data_path = os.path.join(temp_dir, f"{user_id}_clustered_data.csv")
                with open(clustered_data_path, 'wb') as f:
                    f.write(clustered_data_bytes)
            except Exception:
                # If clustered data not found, try segmented data
                try:
                    segmented_storage_path = f"{user_id}/{user_id}_segmented_data.csv"
                    segmented_data_bytes = download_file_from_supabase(segmented_storage_path)
                    segmented_data_path = os.path.join(temp_dir, f"{user_id}_segmented_data.csv")
                    with open(segmented_data_path, 'wb') as f:
                        f.write(segmented_data_bytes)
                except Exception:
                    return {"status": "error", "message": f"No clustering data found for user ID: {user_id}"}

            # Load the clustered data
            if clustered_data_path and os.path.exists(clustered_data_path):
                df = pd.read_csv(clustered_data_path)
            elif segmented_data_path and os.path.exists(segmented_data_path):
                df = pd.read_csv(segmented_data_path)
            else:
                return {"status": "error", "message": f"No clustering data found for user ID: {user_id}"}

            # Determine dataset name
            if file_path:
                dataset_name = os.path.basename(file_path)
            else:
                dataset_name = f"{os.path.basename(train_path)} + {os.path.basename(test_path)}"

            # ───────────── Validate cluster column exists ─────────────
            if 'cluster' not in df.columns:
                return {"status": "error", "message": "No cluster column found in data. Run clustering first."}

            # ───────────── Parse feature columns ─────────────
            feature_list = feature_columns.split(",") if feature_columns else None
            if feature_list:
                available_features = [col for col in feature_list if col in df.columns]
                if not available_features:
                    available_features = df.select_dtypes(include=["number"]).columns[:2].tolist()
            else:
                available_features = df.select_dtypes(include=["number"]).columns[:2].tolist()

            # ───────────── Apply cluster labeling logic ─────────────
            df = label_clusters_general(df, cluster_col="cluster", feature_columns=available_features)

            # ───────────── Save labeled data to temp directory ─────────────
            labeled_data_path = PathL(temp_dir) / f"{user_id}_labeled_data.csv"
            df.to_csv(labeled_data_path, index=False)

            # ───────────── Upload labeled data to Supabase ─────────────
            labeled_data_filename = labeled_data_path.name
            labeled_data_upload_path = upload_file_to_supabase(
                user_id=user_id,
                file_path=str(labeled_data_path),
                filename=labeled_data_filename
            )

            # Get signed URL for the uploaded file
            labeled_data_url = get_file_url(labeled_data_upload_path)

            # ───────────── Segment summary ─────────────
            segments_df = df.groupby(["cluster", "segment_name"]).size().reset_index().rename(columns={0: "count"})
            segments_summary = segments_df.to_dict("records")
            
            # Calculate segment statistics
            segment_stats = {}
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if "cluster" in numeric_cols: 
                numeric_cols.remove("cluster")

            desc = df.groupby("segment_name")[numeric_cols].agg(["mean", "median", "std", "min", "max"])

            for segment in desc.index:
                segment_stats[segment] = {
                    f"{col}_{stat}": round(desc.loc[segment, (col, stat)], 2)
                    for col in numeric_cols
                    for stat in ["mean", "median", "std", "min", "max"]
                }

            # ───────────── Create scatter visualization ─────────────
            cluster_viz_base64 = None
            if len(available_features) >= 2:
                try:
                    feature_1, feature_2 = available_features[:2]
                    cluster_fig = plt.figure(figsize=(12, 8))
                    sns.scatterplot(
                        data=df,
                        x=feature_1,
                        y=feature_2,
                        hue="segment_name",
                        palette="viridis",
                        s=100,
                        alpha=0.7
                    )
                    plt.xlabel(feature_1.replace("_", " ").title())
                    plt.ylabel(feature_2.replace("_", " ").title())
                    plt.title("Cluster Segments")
                    plt.legend(title="Segment", bbox_to_anchor=(1.05, 1), loc="upper left")
                    plt.tight_layout()
                    cluster_viz_base64 = plot_to_base64(cluster_fig)
                    plt.close(cluster_fig)
                except Exception as e:
                    logger.warning(f"Scatter plot generation failed: {e}")

            # ───────────── Create pie distribution visualization ─────────────
            distribution_viz_base64 = None
            try:
                dist_fig = plt.figure(figsize=(10, 6))
                segment_counts = df["segment_name"].value_counts()
                plt.pie(segment_counts.values, labels=segment_counts.index, autopct="%1.1f%%", startangle=90)
                plt.title("Segment Distribution")
                plt.axis("equal")
                distribution_viz_base64 = plot_to_base64(dist_fig)
                plt.close(dist_fig)
            except Exception as viz_err:
                logger.warning(f"Pie chart generation failed: {viz_err}")

            # ───────────── Generate segment insights ─────────────
            segment_insights = generate_segment_insights(segment_stats)

            # ───────────── Build response (match segmentation structure) ─────────────
            response_data = {
                "status": "success",
                "user_id": user_id,
                "id": str(uuid.uuid4()),
                "created_at": datetime.utcnow().isoformat(),
                "type": "label_clusters",
                "dataset": dataset_name,
                "parameters": {
                    "feature_columns": available_features
                },
                "message": "Cluster Labels Generated",
                "segments_summary": segments_summary,
                "segment_stats": segment_stats,
                "segment_insights": segment_insights,
                "visualizations": {},
                "insights": "",
                "total_records": len(df),
                "features_used": available_features,
                # ✅ Add uploaded file references
                "labeled_data_url": labeled_data_url,
                "labeled_data_path": labeled_data_upload_path
            }

            # Add visualizations if they exist
            if cluster_viz_base64:
                response_data["visualizations"]["scatter"] = f"data:image/png;base64,{cluster_viz_base64}"
            if distribution_viz_base64:
                response_data["visualizations"]["distribution"] = f"data:image/png;base64,{distribution_viz_base64}"
            try:
                # ───────────── Save metadata for gallery ─────────────
                entry = {
                    "id": str(uuid.uuid4()),
                    "created_at": datetime.utcnow().isoformat(),
                    "type": "label_clusters",
                    "dataset": dataset_name,
                    "parameters": {
                        "feature_columns": available_features
                    },
                    "thumbnailData": f"data:image/png;base64,{cluster_viz_base64}" if cluster_viz_base64 else "",
                    "imageData": f"data:image/png;base64,{cluster_viz_base64 or distribution_viz_base64}",
                    "visualizations": {
                        "scatter": f"data:image/png;base64,{cluster_viz_base64}" if cluster_viz_base64 else "",
                        "distribution": f"data:image/png;base64,{distribution_viz_base64}" if distribution_viz_base64 else ""
                    },
                    "segments_summary": segments_summary,
                    "segment_stats": segment_stats,
                    "segment_insights": segment_insights,
                    # ✅ Add uploaded file references to metadata
                    "labeled_data_url": labeled_data_url,
                    "labeled_data_path": labeled_data_upload_path
                }

                entry = ensure_json_serializable(entry)
                with master_db_cm() as db:
                    _append_limited_metadata(user_id, entry, db=db, max_entries=5)
            except Exception as meta_error:
                print(f"[⚠️] Metadata save error: {meta_error}")


            return response_data

    except Exception as e:
        logger.error(f"❌ Error in do_label_clusters for user_id {user_id}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}
def generate_regression_insights(pred_stats, top_features, financial_inputs=None):
    insights = []

    # 1. Prediction Range
    insights.append(
        f"🔍 Predicted values range from {pred_stats['min']:.2f} to {pred_stats['max']:.2f}, "
        f"with a median of {pred_stats['median']:.2f}."
    )

    # 2. Standard deviation / variability
    if pred_stats['std'] > pred_stats['mean'] * 0.3:
        insights.append("⚠️ High variability detected in predicted values — consider grouping entities by predicted value ranges.")
    else:
        insights.append("✅ Predicted values are consistent — suggesting a stable pattern across the data.")

    # 3. Top contributing features
    if top_features:
        key_features = [
            f['feature'] if 'feature' in f else list(f.values())[0]
            for f in top_features[:3]
        ]
        insights.append(f"📊 Most influential variables: {', '.join(key_features)}")

    # 4. Estimated magnitude (neutral language)
    if financial_inputs and "average_order_value" in financial_inputs:
        estimated_value = round(pred_stats["mean"] * financial_inputs["average_order_value"], 2)
        insights.append(f"📈 When scaled by the average unit value, the estimated mean impact is approximately {estimated_value:.2f}.")

    return insights
def do_regression(
    user_id: str,
    current_user: dict = None,
    file_path: str = None,
    train_path: str = None,
    test_path: str = None,
    target_column: str = "target",
    drop_columns: str = ""
) -> dict:

    import json
    import subprocess
    import tempfile
    import os
    import pandas as pd
    from pathlib import Path as PathL
    import numpy as np
    import joblib
    import uuid
    from datetime import datetime

    # Validate upload mode
    if file_path and (train_path or test_path):
        raise ValueError("Provide either file_path or both train_path+test_path, not both.")
    if (train_path and not test_path) or (test_path and not train_path):
        raise ValueError("Both train_path and test_path must be provided together.")
    if not file_path and not (train_path and test_path):
        raise ValueError("Provide either a full dataset (file_path) or both train_path+test_path.")

    # Handle file downloads from Supabase
    # Note: file_path, train_path, test_path are now Supabase storage paths (e.g., "user123/dataset.csv")
    local_file_path = None
    local_train_path = None
    local_test_path = None
    temp_files = []

    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download files from Supabase to temporary locations
            if file_path:
                # Download single file from Supabase using the storage path
                file_bytes = download_file_from_supabase(file_path)
                local_file_path = os.path.join(temp_dir, os.path.basename(file_path))
                with open(local_file_path, 'wb') as f:
                    f.write(file_bytes)
                temp_files.append(local_file_path)
            
            if train_path and test_path:
                # Download train and test files from Supabase using the storage paths
                train_bytes = download_file_from_supabase(train_path)
                test_bytes = download_file_from_supabase(test_path)
                
                local_train_path = os.path.join(temp_dir, os.path.basename(train_path))
                local_test_path = os.path.join(temp_dir, os.path.basename(test_path))
                
                with open(local_train_path, 'wb') as f:
                    f.write(train_bytes)
                with open(local_test_path, 'wb') as f:
                    f.write(test_bytes)
                    
                temp_files.extend([local_train_path, local_test_path])
            
            # Prepare model directory
            model_dir = os.path.join(temp_dir, "models")
            os.makedirs(model_dir, exist_ok=True)

            # Create user directory for outputs
            user_dir = PathL(temp_dir) / "user_outputs"
            user_dir.mkdir(exist_ok=True)
            processed_data_path = user_dir / "processed_full_data.csv"
            
            # ───────────── File processing ─────────────
            train_df = None
            test_df = None
            dataset_name = ""

            if local_file_path:
                # Single file mode
                df = pd.read_csv(local_file_path)
                df = df.dropna(subset=[target_column])
                dataset_name = os.path.basename(file_path)

                if drop_columns:
                    drops = [c.strip() for c in drop_columns.split(",") if c.strip() and c in df.columns]
                    if drops:
                        df.drop(columns=drops, inplace=True)

                df, cats, nums = preprocess_data(df)

                if target_column not in df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in dataset.")

                from sklearn.model_selection import train_test_split
                X = df.drop(columns=[target_column])
                y = df[target_column]

                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                X_train = X_train.reset_index(drop=True)
                X_test  = X_test.reset_index(drop=True)
                y_train = y_train.reset_index(drop=True)
                y_test  = y_test.reset_index(drop=True)
                # Filter to numerical and boolean dtypes
                X_train = X_train.select_dtypes(include=["int", "float", "bool"])
                X_test = X_test.select_dtypes(include=["int", "float", "bool"])

                # Reconstruct train/test DataFrames for downstream use
                train_df = pd.concat([X_train, y_train.rename(target_column)], axis=1)
                test_df  = pd.concat([X_test,  y_test.rename(target_column)], axis=1)
            else:
                train_df = pd.read_csv(local_train_path)
                test_df = pd.read_csv(local_test_path)
                train_df = train_df.dropna(subset=[target_column])
                test_df = test_df.dropna(subset=[target_column])
                dataset_name = f"{os.path.basename(train_path)} + {os.path.basename(test_path)}"

                if drop_columns:
                    drops = [c.strip() for c in drop_columns.split(",") if c.strip()]
                    train_df.drop(columns=[c for c in drops if c in train_df.columns], inplace=True)
                    test_df.drop(columns=[c for c in drops if c in test_df.columns], inplace=True)

                train_df, train_cats, train_nums = preprocess_data(train_df)
                test_df, test_cats, test_nums = preprocess_data(test_df)
                train_df = train_df.reset_index(drop=True)

                test_df = test_df.dropna(subset=[target_column])
                test_df, test_cats, test_nums = preprocess_data(test_df)
                test_df = test_df.reset_index(drop=True)

            if target_column not in train_df.columns:
                raise ValueError(f"Target column '{target_column}' not found in training dataset.")

            print(f"Train dataset shape: {train_df.shape}")
            print(f"Test dataset shape: {test_df.shape if test_df is not None else 'None'}")
            print(f"Target column: {target_column}")

            # ───────────── Model Training ─────────────
            print("Starting model training...")
            results = train_regression_models(
                train_df=train_df,
                test_df=test_df,
                target_column=target_column,
            )
            print("Model training completed!")

            # ───────────── Save Model & Preprocessor ─────────────
            model_path = PathL(model_dir) / f"{user_id}_best_regressor.pkl"
            preprocessor_path = PathL(model_dir) / f"{user_id}_preprocessor.pkl"

            joblib.dump(results["final_model"], model_path)
            joblib.dump(results["preprocessor"], preprocessor_path)
            print(f"[💾] Models saved to: {model_path} and {preprocessor_path}")

            # train_df has shape (n_samples, n_features+1) with the last column = target
            training_features = [
                c for c in train_df.columns
                if c != target_column
            ]

            # ───────────── Generate Visualizations & Process Data ─────────────
            try:
                # Prepare full dataset for processing
                if local_file_path:
                    full_df = pd.read_csv(local_file_path)
                    full_df = full_df.dropna(subset=[target_column])
                    if drop_columns:
                        drops = [c.strip() for c in drop_columns.split(",") if c.strip() and c in full_df.columns]
                        if drops:
                            full_df.drop(columns=drops, inplace=True)
                else:
                    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)


                X_full_raw = full_df.reindex(columns=training_features, fill_value=0)
                X_full_raw = full_df.drop(columns=[target_column], errors="ignore")
                X_full_processed, _, _ = preprocess_data(X_full_raw)

                X_full_df = pd.DataFrame(X_full_processed)
                
                # ───────────── Predictions on Full Data ─────────────
                preds_full             = results["final_model"].predict(X_full_processed)

                # ───────────── Compute Prediction Stats ─────────────
                pred_stats = {
                    "count": len(preds_full),
                    "mean": float(np.mean(preds_full)),
                    "std": float(np.std(preds_full)),
                    "min": float(np.min(preds_full)),
                    "max": float(np.max(preds_full)),
                    "median": float(np.median(preds_full))
                }

                # ───────────── Load Financial Assumptions (if saved) ─────────────
                # You can pass these as a parameter too
                financial_path = PathL("data") / "financial_inputs" / f"{user_id}.json"
                if financial_path.exists():
                    with open(financial_path) as f:
                        financial_inputs = json.load(f)
                else:
                    financial_inputs = None
                    
                # Save training columns for SHAP alignment
                training_columns_path = user_dir / "training_columns.json"
                with open(training_columns_path, "w") as f:
                    json.dump(list(X_full_df.columns), f)
                training_columns = list(X_full_df.columns)
                upload_file_to_supabase(
                    user_id=user_id,
                    file_path=str(training_columns_path),
                    filename="training_columns.json"
                )
                # Save processed data for SHAP - CREATE FILE BEFORE UPLOAD
                X_full_df.to_csv(processed_data_path, index=False)

                # ───────────── Upload to Supabase (AFTER creating processed data) ─────────────
                try:
                    model_filename = model_path.name
                    preprocessor_filename = preprocessor_path.name
                    data_filename = processed_data_path.name

                    # Upload model, preprocessor, and processed dataset
                    model_supabase_path = upload_file_to_supabase(user_id, str(model_path), model_filename)
                    preprocessor_supabase_path = upload_file_to_supabase(user_id, str(preprocessor_path), preprocessor_filename)
                    data_supabase_path = upload_file_to_supabase(user_id, str(processed_data_path), data_filename)

                    # Get signed URLs for frontend access (1-hour expiration)
                    model_url = get_file_url(model_supabase_path, expires_in=3600)
                    preprocessor_url = get_file_url(preprocessor_supabase_path, expires_in=3600)
                    data_url = get_file_url(data_supabase_path, expires_in=3600)

                    print(f"[✅] Uploaded model: {model_url}")
                    print(f"[✅] Uploaded preprocessor: {preprocessor_url}")
                    print(f"[✅] Uploaded processed data: {data_url}")

                except Exception as upload_error:
                    print(f"[⚠️] Failed to upload regression artifacts to Supabase: {upload_error}")
                    # Set default values if upload fails
                    model_url = ""
                    preprocessor_url = ""
                    data_url = ""

               # Create request JSON for SHAP
                request_json = user_dir / "request.json"
                with open(request_json, "w") as f:
                    json.dump({
                        "user_id": str(user_id),
                        "model_path": str(model_path.resolve()),
                        "data_path": str(processed_data_path.resolve()),
                        "output_dir": str(user_dir.resolve()),
                        "model_type": "regression",
                        "target_column": target_column,
                        "save_filename": f"{user_id}_feature_importance.png",
                        "training_columns": training_columns
                    }, f)

                # Save training columns needed by SHAP
                try:
                    training_columns_path = user_dir / "training_columns.json"
                    with open(training_columns_path, "w") as f:
                        json.dump(list(X_full_df.columns), f)
                    print(f"[✅] Saved training columns to {training_columns_path}")
                except Exception as col_err:
                    print(f"[⚠️] Failed to save training columns: {col_err}")

                # ───────────── SHAP Analysis ─────────────
                fi_shap_bar, fi_shap_dot, imp_df = None, None, pd.DataFrame()
                try:
                    import sys
                    import subprocess
                    import time
                    
                    # Get the current script directory (where do_regression is located)
                    current_dir = PathL(__file__).parent
                    shap_runner_path = current_dir / "shap_runner.py"

                    if not shap_runner_path.exists():
                        raise FileNotFoundError(f"shap_runner.py not found at {shap_runner_path}")
                    
                    # Create request JSON for SHAP
                    request_json = user_dir / "request.json"
                    with open(request_json, "w") as f:
                        json.dump({
                            "user_id": str(user_id),
                            "model_path": str(model_path.resolve()),
                            "data_path": str(processed_data_path.resolve()),
                            "output_dir": str(user_dir.resolve()),
                            "model_type": "regression",
                            "target_column": target_column,
                            "save_filename": f"{user_id}_feature_importance.png",
                            "training_columns": training_columns
                        }, f, indent=2)

                    # Verify request JSON content
                    with open(request_json, "r") as f:
                        request_content = json.load(f)
                    print(f"[SHAP DEBUG] Request content: {request_content}")

                    # Set up the subprocess command
                    cmd = [sys.executable, str(shap_runner_path.resolve()), str(request_json.resolve())]

                    # Add environment variables for better error reporting
                    env = os.environ.copy()
                    env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output
                    env['PYTHONIOENCODING'] = 'utf-8'  # Ensure proper encoding

                    # Run SHAP subprocess with real-time output
                    process = subprocess.Popen(
                        cmd,
                        cwd=str(current_dir),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,  # Merge stderr into stdout
                        text=True,
                        bufsize=1,  # Line buffered
                        env=env,
                        universal_newlines=True
                    )
                    
                    # Read output in real-time
                    output_lines = []
                    timeout_seconds = 300  # 5 minutes
                    start_time = time.time()
                    
                    while True:
                        # Check if process has finished
                        if process.poll() is not None:
                            # Process finished, read any remaining output
                            remaining_output = process.stdout.read()
                            if remaining_output:
                                output_lines.append(remaining_output)
                                print(f"[SHAP SUBPROCESS] {remaining_output.strip()}")
                            break
                        
                        # Check for timeout
                        if time.time() - start_time > timeout_seconds:
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                            raise subprocess.TimeoutExpired(cmd, timeout_seconds)
                        
                        # Read output line by line
                        line = process.stdout.readline()
                        if line:
                            output_lines.append(line)
                        else:
                            time.sleep(0.1)  # Small delay to prevent busy waiting
                    
                    # Get the return code
                    return_code = process.returncode
                    full_output = ''.join(output_lines)

                    if return_code != 0:
                        raise subprocess.CalledProcessError(return_code, cmd, output=full_output)

                    # Load results
                    result_json_path = user_dir / "result.json"

                    
                    if result_json_path.exists():
                        try:
                            with open(result_json_path) as f:
                                shap_result = json.load(f)
                            print(f"[SHAP DEBUG] Result JSON loaded successfully")
                            print(f"[SHAP DEBUG] Result keys: {list(shap_result.keys())}")

                            fi_shap_bar = shap_result.get("shap_bar")
                            fi_shap_dot = shap_result.get("shap_dot")
                            imp_df_data = shap_result.get("imp_df", [])
                            imp_df = pd.DataFrame(imp_df_data) if imp_df_data else pd.DataFrame()
                            
                            print(f"[SHAP DEBUG] Results extracted - bar: {fi_shap_bar is not None}, dot: {fi_shap_dot is not None}, df shape: {imp_df.shape}")
                            
                            # Check for errors in the result
                            if "error" in shap_result:
                                print(f"[SHAP WARNING] SHAP process reported error: {shap_result['error']}")
                                
                        except Exception as e:
                            print(f"[SHAP ERROR] Failed to parse result JSON: {e}")
                            print(f"[SHAP DEBUG] Raw result file content:")
                            try:
                                with open(result_json_path, 'r') as f:
                                    print(f.read())
                            except:
                                print("[SHAP ERROR] Could not read result file")
                            fi_shap_bar, fi_shap_dot, imp_df = None, None, pd.DataFrame()
                    else:
                        print(f"[SHAP ERROR] Result file not found: {result_json_path}")
                        print(f"[SHAP DEBUG] Files in output directory: {list(user_dir.iterdir())}")
                        fi_shap_bar, fi_shap_dot, imp_df = None, None, pd.DataFrame()

                except subprocess.TimeoutExpired as e:
                    print(f"[⚠️] SHAP subprocess timed out after {timeout_seconds} seconds")
                    print(f"[⚠️] Command: {' '.join(e.cmd)}")
                    fi_shap_bar, fi_shap_dot, imp_df = None, None, pd.DataFrame()
                    
                except subprocess.CalledProcessError as e:
                    print(f"[⚠️] SHAP subprocess failed with return code {e.returncode}")
                    print(f"[⚠️] Command: {' '.join(e.cmd)}")
                    print(f"[⚠️] Output: {e.output}")
                    fi_shap_bar, fi_shap_dot, imp_df = None, None, pd.DataFrame()
                    
                except FileNotFoundError as e:
                    print(f"[⚠️] SHAP runner file not found: {e}")
                    fi_shap_bar, fi_shap_dot, imp_df = None, None, pd.DataFrame()
                    
                except Exception as viz_error:
                    print(f"[⚠️] SHAP visualization error: {viz_error}")
                    import traceback
                    traceback.print_exc()
                    fi_shap_bar, fi_shap_dot, imp_df = None, None, pd.DataFrame()


                # ───────────── Generate Insights ─────────────
                insights = generate_regression_insights(
                    pred_stats=pred_stats,
                    top_features=imp_df.head(10).to_dict("records") if not imp_df.empty else [],
                    financial_inputs=financial_inputs
                )


                # ───────────── Save Metadata for Gallery ─────────────
                try:
                    entry = {
                        "id": str(uuid.uuid4()),
                        "created_at": datetime.utcnow().isoformat(),
                        "type": "regression",
                        "dataset": dataset_name,
                        "parameters": {
                            "target_column": target_column,
                            "drop_columns": drop_columns
                        },
                        "metrics": results.get("test_scores", {}),
                        "thumbnailData": f"data:image/png;base64,{fi_shap_bar}" if fi_shap_bar else "",
                        "imageData": f"data:image/png;base64,{fi_shap_dot or fi_shap_bar or ''}",
                        "top_features": imp_df.head(10).to_dict("records") if not imp_df.empty else [],
                        "visualizations": {
                            "shap_bar": f"data:image/png;base64,{fi_shap_bar}" if fi_shap_bar else "",
                            "shap_dot": f"data:image/png;base64,{fi_shap_dot}" if fi_shap_dot else ""
                        },
                        "pred_stats": pred_stats,
                        "insights": insights,
                        "model_url": model_url,
                        "preprocessor_url": preprocessor_url,
                        "data_url": data_url
                    }
                    entry = ensure_json_serializable(entry)
                    with master_db_cm() as db:
                        _append_limited_metadata(user_id, entry, db=db, max_entries=5)
                except Exception as meta_error:
                    print(f"[⚠️] Metadata save error: {meta_error}")

                # ───────────── Final API Response ─────────────
                response_data = {
                    "status": "success",
                    "user_id": user_id,
                    "id": str(uuid.uuid4()),
                    "created_at": datetime.utcnow().isoformat(),
                    "type": "regression",
                    "dataset": dataset_name,
                    "parameters": {
                        "target_column": target_column,
                        "drop_columns": drop_columns
                    },
                    "metrics": results.get("test_scores", {}),
                    "message": "Regression model training completed",
                    "pred_stats": pred_stats,
                    "top_features": imp_df.head(10).to_dict("records") if not imp_df.empty else [],
                    "insights": insights,
                    "visualizations": {
                        "shap_bar": f"data:image/png;base64,{fi_shap_bar}" if fi_shap_bar else "",
                        "shap_dot": f"data:image/png;base64,{fi_shap_dot}" if fi_shap_dot else ""
                    },
                    "model_url": model_url,
                    "preprocessor_url": preprocessor_url,
                    "data_url": data_url
                }

                return response_data

            except Exception as viz_global_error:
                print(f"[] Visualization pipeline error: {viz_global_error}")
                return {"status": "failed", "error": str(viz_global_error)}

    except Exception as e:
        print(f"[] Regression error: {e}")
        return {"status": "failed", "error": str(e)}
def do_regression_predict(
    user_id: str,
    current_user: dict = None,
    file_path: str = None,
    train_path: str = None,
    test_path: str = None,
    drop_columns: str = ""
) -> dict:

    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    try:
        if file_path is None:
            raise ValueError("You must provide a file_path for prediction input.")

        # Supabase paths
        model_supabase_path = f"{user_id}/{user_id}_best_regressor.pkl"
        preprocessor_supabase_path = f"{user_id}/{user_id}_preprocessor.pkl"
        training_columns_supabase_path = f"{user_id}/training_columns.json"

        # ───── Download training columns ─────
        try:
            training_columns_data = download_file_from_supabase(training_columns_supabase_path)
            training_columns = json.loads(training_columns_data.decode('utf-8'))
        except Exception as e:
            raise FileNotFoundError(f"Training columns metadata not found in Supabase: {str(e)}")

        # ───── Download model ─────
        try:
            print(f"[📦] Downloading model from Supabase: {model_supabase_path}")
            model_data = download_file_from_supabase(model_supabase_path)
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
                tmp.write(model_data)
                model = joblib.load(tmp.name)
        except Exception as e:
            raise FileNotFoundError(f"Failed to download/load model: {e}")

        try:
            print(f"[📦] Downloading preprocessor from Supabase: {preprocessor_supabase_path}")
            preprocessor_data = download_file_from_supabase(preprocessor_supabase_path)

            print(f"[DEBUG] Preprocessor data type: {type(preprocessor_data)}")
            print(f"[DEBUG] Preprocessor data length: {len(preprocessor_data)} bytes")

            if not preprocessor_data:
                raise ValueError("Downloaded preprocessor file is empty")

            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False, mode="wb") as tmp:
                tmp.write(preprocessor_data)
                tmp.flush()
                print(f"[DEBUG] Written preprocessor to temp file: {tmp.name}")
                preprocessor = joblib.load(tmp.name)
                print(f"[✅] Preprocessor loaded successfully")

        except Exception as e:
            raise FileNotFoundError(f"Failed to download/load preprocessor: {e}")

        # ───── Download prediction CSV ─────
        try:
            prediction_data = download_file_from_supabase(file_path)
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as temp_file:
                temp_file.write(prediction_data)
                temp_csv_path = temp_file.name
        except Exception as e:
            raise FileNotFoundError(f"Failed to download prediction file: {e}")

        # ───── Load and preprocess prediction data ─────
        print(f"[📊] Loading prediction data from: {file_path}")
        df = pd.read_csv(temp_csv_path)
        original_df = df.copy()
        original_shape = df.shape
        dataset_name = PathL(file_path).stem

        # Store original ID column if present
        id_column = None
        if 'ID' in df.columns:
            id_column = df['ID'].copy()

        # Save target column (if known or common name exists)
        known_targets = ['target', 'label', 'y', 'Target', 'Label', 'Y']
        target_column_actual = None
        for col in known_targets:
            if col in df.columns:
                target_column_actual = col
                break

        # Save target values if found (for evaluation metrics)
        true_labels = df[target_column_actual].copy() if target_column_actual else None

        # Drop ID and target ONLY from features going into model
        drop_features = ['ID']
        if target_column_actual:
            drop_features.append(target_column_actual)

        # Drop specified columns
        if drop_columns:
            drops = [c.strip() for c in drop_columns.split(",") if c.strip() and c in df.columns]
            df.drop(columns=drops, inplace=True)
            print(f"[🗑️] Dropped columns: {drops}")
        
        df.drop(columns=drop_features, inplace=True, errors='ignore')

        # Preprocess the data using the same function as training
        print("[🔄] Preprocessing prediction data...")
        df_processed, cats, nums = preprocess_data(df)
        
        # Align columns with training data
        print("[🔧] Aligning features with training data...")
        missing_cols = set(training_columns) - set(df_processed.columns)
        extra_cols = set(df_processed.columns) - set(training_columns)
        
        # Add missing columns with default values
        for col in missing_cols:
            df_processed[col] = 0
            
        # Remove extra columns
        df_processed.drop(columns=list(extra_cols), inplace=True, errors='ignore')
        
        # Reorder columns to match training
        df_processed = df_processed.reindex(columns=training_columns, fill_value=0)
        
        print(f"[✅] Feature alignment complete: {len(training_columns)} features")
        
        # Transform and predict
        X_transformed = preprocessor.transform(df_processed)
        predictions = model.predict(X_transformed)
        
        # Create output dataframe by adding predictions to original data
        output_df = original_df.copy()
        output_df['prediction'] = predictions


        # Calculate evaluation metrics if true labels are available
        evaluation_metrics = None
        if true_labels is not None:
            try:
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                evaluation_metrics = {
                    "mae": float(mean_absolute_error(true_labels, predictions)),
                    "mse": float(mean_squared_error(true_labels, predictions)),
                    "rmse": float(np.sqrt(mean_squared_error(true_labels, predictions))),
                    "r2": float(r2_score(true_labels, predictions))
                }
            except Exception as e:
                print(f"[⚠️] Could not calculate evaluation metrics: {e}")
        
        # Generate visualizations
        visualizations = {}
        
        # 1. Prediction distribution
        try:
            fig_dist = plt.figure(figsize=(8, 6))
            plt.hist(predictions, bins=30, alpha=0.7, edgecolor='black')
            plt.title("Prediction Distribution")
            plt.xlabel("Predicted Value")
            plt.ylabel("Frequency")
            plt.axvline(np.mean(predictions), color='red', linestyle='--', 
                        label=f'Mean: {np.mean(predictions):.3f}')
            plt.legend()
            visualizations["prediction_distribution"] = f"data:image/png;base64,{plot_to_base64(fig_dist)}"
            plt.close(fig_dist)
        except Exception as e:
            print(f"[⚠️] Prediction distribution plot failed: {e}")
        
        # 2. Actual vs Predicted scatter plot (if true labels available)
        if true_labels is not None:
            try:
                fig_scatter = plt.figure(figsize=(8, 6))
                plt.scatter(true_labels, predictions, alpha=0.6)
                plt.plot([true_labels.min(), true_labels.max()], 
                        [true_labels.min(), true_labels.max()], 'r--', lw=2)
                plt.xlabel("Actual Values")
                plt.ylabel("Predicted Values")
                plt.title("Actual vs Predicted Values")
                if evaluation_metrics:
                    plt.text(0.05, 0.95, f"R² = {evaluation_metrics['r2']:.3f}", 
                            transform=plt.gca().transAxes, verticalalignment='top')
                visualizations["actual_vs_predicted"] = f"data:image/png;base64,{plot_to_base64(fig_scatter)}"
                plt.close(fig_scatter)
            except Exception as e:
                print(f"[⚠️] Actual vs predicted plot failed: {e}")
        
        # 3. Residuals plot (if true labels available)
        if true_labels is not None:
            try:
                residuals = true_labels - predictions
                fig_residuals = plt.figure(figsize=(8, 6))
                plt.scatter(predictions, residuals, alpha=0.6)
                plt.axhline(y=0, color='r', linestyle='--')
                plt.xlabel("Predicted Values")
                plt.ylabel("Residuals")
                plt.title("Residuals Plot")
                visualizations["residuals"] = f"data:image/png;base64,{plot_to_base64(fig_residuals)}"
                plt.close(fig_residuals)
            except Exception as e:
                print(f"[⚠️] Residuals plot failed: {e}")
        
        # Save predictions to temporary file, then upload to Supabase
        output_filename = f"predictions_{PathL(file_path).stem}.csv"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as temp_output:
            output_df.to_csv(temp_output.name, index=False)
            temp_output_path = temp_output.name
        
        try:
            # Upload predictions to Supabase
            supabase_output_path = upload_file_to_supabase(
                user_id=str(user_id),
                file_path=temp_output_path,
                filename=output_filename
            )

            # Get signed URL as a string (not a dict)
            signed_url = get_file_url(supabase_output_path, expires_in=3600)

            print(f"[💾] Predictions uploaded to Supabase: {supabase_output_path}")
            print(f"[🔗] Signed URL: {signed_url}")

        finally:
            # Clean up temporary output file
            if os.path.exists(temp_output_path):
                os.unlink(temp_output_path)

        # Calculate prediction statistics - CONVERT NUMPY TYPES HERE
        pred_stats = {
            'count': int(len(predictions)),
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'median': float(np.median(predictions)),
            'original_data_shape': [int(x) for x in original_shape],
            'processed_data_shape': [int(x) for x in df_processed.shape]
        }
        
        # Add evaluation metrics if available
        if evaluation_metrics:
            pred_stats.update(evaluation_metrics)
        
        # Generate insights
        insights = generate_regression_insights(
            pred_stats=pred_stats,
            top_features=None,
            financial_inputs=None
        )
        
        try:
            # Create metadata entry
            entry = {
                "id": str(uuid.uuid4()),
                "created_at": datetime.utcnow().isoformat(),
                "type": "regression_prediction",
                "target_col": target_column_actual,
                "dataset": dataset_name,
                "parameters": {"drop_columns": drop_columns},
                "metrics": pred_stats,
                "output_file": supabase_output_path,
                "signed_url": signed_url,
                "visualizations": visualizations,
                "thumbnailData": visualizations.get("prediction_distribution"),
                "imageData": visualizations.get("actual_vs_predicted"),
                "feature_alignment": {
                    "training_features": int(len(training_columns)),
                    "prediction_features": int(len(df_processed.columns)),
                    "missing_features_added": list(missing_cols),
                    "extra_features_removed": list(extra_cols)
                },
                "evaluation_metrics": evaluation_metrics,
                "insights": insights
            }

            entry = ensure_json_serializable(entry)
            with master_db_cm() as db:
                _append_limited_metadata(user_id, entry, db=db, max_entries=5)
        except Exception as meta_error:
            print(f"[⚠️] Metadata save error: {meta_error}")

        # Prepare response - CONVERT ALL NUMPY TYPES
        response_data = {
            "status": "success",
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "type": "regression",
            "dataset": dataset_name,
            "file": file_path,
            "output_file": supabase_output_path,
            "output_file_name": output_filename,
            "signed_url": signed_url,
            "parameters": {"drop_columns": drop_columns},
            "predictions_count": int(len(predictions)),
            "data_preview": output_df.head(10).to_dict('records'),  # Show first 10 rows as preview
            "total_rows": int(len(output_df)),
            "columns_added": ["prediction"],
            "prediction_statistics": pred_stats,
            "model_info": {
                "model_type": type(model).__name__,
                "feature_count": int(X_transformed.shape[1]) if hasattr(X_transformed, 'shape') else None
            },
            "metrics": pred_stats,
            "visualizations": visualizations,
            "insights": insights,
            "feature_alignment": {
                "training_features": int(len(training_columns)),
                "prediction_features": int(len(df_processed.columns)),
                "missing_features_added": list(missing_cols),
                "extra_features_removed": list(extra_cols)
            },
            "evaluation_metrics": evaluation_metrics
        }
        
        # Convert all numpy types in the response
        response_data = convert_numpy_types(response_data)
        
        print(f"[🎉] Regression prediction completed successfully!")
        print(f"    • Total predictions: {len(predictions)}")
        print(f"    • Prediction range: {pred_stats['min']:.3f} to {pred_stats['max']:.3f}")
        print(f"    • Mean prediction: {pred_stats['mean']:.3f}")
        print(f"    • Output file: {supabase_output_path}")
        print(f"    • Signed URL: {signed_url}")
        if evaluation_metrics:
            print(f"    • R² Score: {evaluation_metrics['r2']:.3f}")
        
        return response_data
        
    except Exception as e:
        print(f"[❌] Prediction task error: {e}")
        return {
            "status": "error",
            "user_id": user_id,
            "error_message": str(e),
            "error_type": type(e).__name__
        }
    finally:
        # Clean up temporary CSV file
        if 'temp_csv_path' in locals() and os.path.exists(temp_csv_path):
            os.unlink(temp_csv_path)
def do_visualization(
    user_id: str,
    current_user: dict = None,
    file_path: str = None,
    train_path: str = None,
    test_path: str = None,
    target_column: str = None,
    feature_column: str = None
) -> dict:

    import numpy as np
    import os
    import uuid
    import joblib
    import io
    import base64
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tempfile
    import json
    from pathlib import Path as PathL
    from sklearn.inspection import partial_dependence
    from datetime import datetime

    # ───────────── Validate upload mode ─────────────
    if file_path and (train_path or test_path):
        raise ValueError("Provide either file_path or both train_path+test_path, not both.")
    if (train_path and not test_path) or (test_path and not train_path):
        raise ValueError("Both train_path and test_path must be provided together.")
    if not file_path and not (train_path and test_path):
        raise ValueError("Provide either a full dataset (file_path) or both train_path+test_path.")

    # Handle file downloads from Supabase
    # Note: file_path, train_path, test_path are now Supabase storage paths (e.g., "user123/dataset.csv")
    local_file_path = None
    local_train_path = None
    local_test_path = None
    temp_files = []
    
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download files from Supabase to temporary locations
            if file_path:
                # Download single file from Supabase using the storage path
                file_bytes = download_file_from_supabase(file_path)
                local_file_path = os.path.join(temp_dir, os.path.basename(file_path))
                with open(local_file_path, 'wb') as f:
                    f.write(file_bytes)
                temp_files.append(local_file_path)
            
            if train_path and test_path:
                # Download train and test files from Supabase using the storage paths
                train_bytes = download_file_from_supabase(train_path)
                test_bytes = download_file_from_supabase(test_path)
                
                local_train_path = os.path.join(temp_dir, os.path.basename(train_path))
                local_test_path = os.path.join(temp_dir, os.path.basename(test_path))
                
                with open(local_train_path, 'wb') as f:
                    f.write(train_bytes)
                with open(local_test_path, 'wb') as f:
                    f.write(test_bytes)
                    
                temp_files.extend([local_train_path, local_test_path])
            
            # Create temporary directories for models and outputs
            model_dir = os.path.join(temp_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            
            user_dir = PathL(temp_dir) / "user_outputs"
            user_dir.mkdir(exist_ok=True)

            # ───────────── Load data ─────────────
            if local_file_path:
                df = pd.read_csv(local_file_path)
                dataset_name = os.path.basename(file_path)
            else:
                train_df = pd.read_csv(local_train_path)
                test_df = pd.read_csv(local_test_path)
                df = pd.concat([train_df, test_df], ignore_index=True)
                dataset_name = f"{os.path.basename(train_path)}+{os.path.basename(test_path)}"

            # ───────────── Preprocess data ─────────────
            remove_cols = ["ID", target_column] if "ID" in df.columns else [target_column]
            df, CATS, NUMS = preprocess_data(df, RMV=remove_cols)
            # ───────────── Optional: Generate Customer-Level Summary Stats ─────────────

            df = df.reset_index(drop=True)
            if target_column not in df.columns or feature_column not in df.columns:
                raise ValueError("Specified target or feature column not found in dataset.")

            # ───────────── Generate scatter/box plot ─────────────
            fig1, ax = plt.subplots(figsize=(8, 5))
            if feature_column in CATS:
                sns.boxplot(x=df[feature_column], y=df[target_column], ax=ax)
            else:
                sns.scatterplot(x=df[feature_column], y=df[target_column], ax=ax)
            ax.set_title(f"{feature_column} vs {target_column}")
            ax.set_xlabel(feature_column)
            ax.set_ylabel(target_column)

            scatter_b64 = plot_to_base64(fig1)
            plt.close(fig1)
            
            # ───────────── Generate PDP plot if applicable ─────────────
            pdp_b64 = None
            if feature_column in NUMS:
                try:
                    # Try to load existing model from temp directory or download from Supabase
                    model_path = None
                    classifier_path = PathL(model_dir) / f"{user_id}_best_classifier.pkl"
                    regressor_path = PathL(model_dir) / f"{user_id}_best_regressor.pkl"
                    
                    # Try to download model from Supabase if it exists
                    try:
                        if not classifier_path.exists():
                            # Try to download classifier model from Supabase
                            model_bytes = download_file_from_supabase(f"models/{user_id}_best_classifier.pkl")
                            with open(classifier_path, 'wb') as f:
                                f.write(model_bytes)
                            model_path = classifier_path
                    except:
                        pass
                    
                    try:
                        if not model_path and not regressor_path.exists():
                            # Try to download regressor model from Supabase
                            model_bytes = download_file_from_supabase(f"models/{user_id}_best_regressor.pkl")
                            with open(regressor_path, 'wb') as f:
                                f.write(model_bytes)
                            model_path = regressor_path
                    except:
                        pass
                    
                    # Use whichever model exists
                    if not model_path:
                        if classifier_path.exists():
                            model_path = classifier_path
                        elif regressor_path.exists():
                            model_path = regressor_path
                    
                    if model_path and model_path.exists():
                        model = joblib.load(model_path)
                        estimator = model 
                        FEATURES = [
                            c for c in df.columns if c not in remove_cols
                        ]
                        idx = FEATURES.index(feature_column)
                        pdp_res = partial_dependence(estimator, df[FEATURES], features=[idx])
                        vals, avgs = pdp_res["values"][0], pdp_res["average"][0]

                        fig2 = plt.figure(figsize=(8, 5))
                        plt.plot(vals, avgs, lw=2)
                        plt.title(f"PDP of {feature_column}")
                        plt.xlabel(feature_column)
                        plt.ylabel("Predicted")
                        pdp_b64 = plot_to_base64(fig2)
                        plt.close(fig2)
                except Exception as e:
                    print(f"[⚠️] PDP generation failed: {e}")
                    pdp_b64 = None

            # ───────────── Save metadata to Supabase (similar to classification) ─────────────
            try:
                # Create entry for metadata
                entry = {
                    "id": str(uuid.uuid4()),
                    "created_at": datetime.utcnow().isoformat(),
                    "type": "visualization",
                    "dataset": dataset_name,
                    "parameters": {
                        "target_column": target_column,
                        "feature_column": feature_column
                    },
                    "thumbnailData": f"data:image/png;base64,{scatter_b64}" if scatter_b64 else "",
                    "imageData": f"data:image/png;base64,{scatter_b64}" if scatter_b64 else "",
                    "visualizations": {},
                }

                if scatter_b64:
                    entry["visualizations"]["scatter_plot"] = f"data:image/png;base64,{scatter_b64}"
                
                if pdp_b64:
                    entry["visualizations"]["pdp_plot"] = f"data:image/png;base64,{pdp_b64}"

                entry = ensure_json_serializable(entry)
                with master_db_cm() as db:
                    _append_limited_metadata(user_id, entry, db=db, max_entries=5)
            except Exception as meta_error:
                print(f"[⚠️] Metadata save error: {meta_error}")


            # ───────────── Build response ─────────────
            response_data = {
                "status": "success",
                "user_id": user_id,
                "id": str(uuid.uuid4()),
                "created_at": datetime.utcnow().isoformat(),
                "type": "visualization",
                "dataset": dataset_name,
                "parameters": {
                    "target_column": target_column,
                    "feature_column": feature_column
                },
                "visualizations": {},
                "insights": {}
            }

            if scatter_b64:
                response_data["visualizations"]["scatter_plot"] = f"data:image/png;base64,{scatter_b64}"

            if pdp_b64:
                response_data["visualizations"]["pdp_plot"] = f"data:image/png;base64,{pdp_b64}"
            return response_data  
                
    except Exception as e:
        print(f"[⚠️] Error in do_visualization: {e}")
        raise e        



def do_counterfactual(
    user_id: str,
    current_user: dict = None,
    file_path: str = None,
    train_path: str = None,
    test_path: str = None,
    target_column: str = "target",
    drop_columns: str = "",
    sample_id: int = None,
    sample_strategy: str = "lowest_confidence",
    num_samples: int = 10,
    desired_outcome: Union[int, float, str] = 1,
    editable_features: List[str] = None,
    max_changes: int = 10,
    proximity_metric: str = "euclidean"
) -> dict:
    import pandas as pd
    import numpy as np
    import os
    import uuid
    import joblib
    import io
    import base64
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tempfile
    import json
    from pathlib import Path as PathL
    from datetime import datetime
    from sklearn.model_selection import train_test_split

    # ───────────── Validate upload mode ─────────────
    if file_path and (train_path or test_path):
        raise ValueError("Provide either file_path or both train_path+test_path, not both.")
    if (train_path and not test_path) or (test_path and not train_path):
        raise ValueError("Both train_path and test_path must be provided together.")
    if not file_path and not (train_path and test_path):
        raise ValueError("Provide either a full dataset (file_path) or both train_path+test_path.")

    # Handle file downloads from Supabase
    # Note: file_path, train_path, test_path are now Supabase storage paths (e.g., "user123/dataset.csv")
    local_file_path = None
    local_train_path = None
    local_test_path = None
    temp_files = []
    
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download files from Supabase to temporary locations
            if file_path:
                # Download single file from Supabase using the storage path
                file_bytes = download_file_from_supabase(file_path)
                local_file_path = os.path.join(temp_dir, os.path.basename(file_path))
                with open(local_file_path, 'wb') as f:
                    f.write(file_bytes)
                temp_files.append(local_file_path)
            
            if train_path and test_path:
                # Download train and test files from Supabase using the storage paths
                train_bytes = download_file_from_supabase(train_path)
                test_bytes = download_file_from_supabase(test_path)
                
                local_train_path = os.path.join(temp_dir, os.path.basename(train_path))
                local_test_path = os.path.join(temp_dir, os.path.basename(test_path))
                
                with open(local_train_path, 'wb') as f:
                    f.write(train_bytes)
                with open(local_test_path, 'wb') as f:
                    f.write(test_bytes)
                    
                temp_files.extend([local_train_path, local_test_path])
            
            # Create temporary directories for models and outputs
            model_dir = os.path.join(temp_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            
            user_dir = PathL(temp_dir) / "user_outputs"
            user_dir.mkdir(exist_ok=True)

            # ──────── Load & preprocess data ────────
            if local_file_path:
                df = pd.read_csv(local_file_path)
                df = df.dropna(subset=[target_column])
                dataset_name = os.path.basename(file_path)

                if drop_columns:
                    drops = [c.strip() for c in drop_columns.split(",") if c.strip() and c in df.columns]
                    df.drop(columns=drops, inplace=True)

                if target_column not in df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in dataset.")

                # Separate target
                y = df[target_column]
                X = df.drop(columns=['ID', target_column], errors='ignore')

                # Preprocess input features only (returns new df with same shape)
                X, _, _ = preprocess_data(X)

                # Ensure only valid types remain
                X = X.select_dtypes(include=["int", "float", "bool"])

                # Split into train and test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                X_train = X_train.reset_index(drop=True)
                X_test  = X_test.reset_index(drop=True)
                y_train = y_train.reset_index(drop=True)
                y_test  = y_test.reset_index(drop=True)

                # Reattach target column
                train_df = pd.concat([X_train, y_train.rename(target_column)], axis=1)
                test_df  = pd.concat([X_test,  y_test.rename(target_column)], axis=1)

                # Optional: Reconstruct full dataset (if needed for export or model-wide stats)
                full_df = pd.concat([train_df, test_df], ignore_index=True)


            else:
                train_df = pd.read_csv(local_train_path)
                test_df = pd.read_csv(local_test_path)
                train_df = train_df.dropna(subset=[target_column])
                test_df = test_df.dropna(subset=[target_column])
                dataset_name = f"{os.path.basename(train_path)}+{os.path.basename(test_path)}"

                if drop_columns:
                    drops = [c.strip() for c in drop_columns.split(",") if c.strip()]
                    train_df.drop(columns=[c for c in drops if c in train_df.columns], inplace=True)
                    test_df.drop(columns=[c for c in drops if c in test_df.columns], inplace=True)

                if target_column not in train_df.columns or target_column not in test_df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in training/test dataset.")

                y_train = train_df[target_column]
                y_test  = test_df[target_column]

                X_train = train_df.drop(columns=[target_column, 'ID'], errors='ignore')
                X_test  = test_df.drop(columns=[target_column, 'ID'], errors='ignore')

                X_train, _, _ = preprocess_data(X_train)
                X_test, _, _ = preprocess_data(X_test)

                X_train = X_train.reset_index(drop=True)
                X_test  = X_test.reset_index(drop=True)
                y_train = y_train.reset_index(drop=True)
                y_test  = y_test.reset_index(drop=True)

                # Align columns between train and test after preprocessing
                common_columns = list(set(X_train.columns) & set(X_test.columns))
                X_train = X_train[common_columns]
                X_test  = X_test[common_columns]

                train_df = pd.concat([X_train, y_train.rename(target_column)], axis=1)
                test_df = pd.concat([X_test, y_test.rename(target_column)], axis=1)
                full_df = pd.concat([train_df, test_df], ignore_index=True)

            # Clean data - remove NaN and infinite values
            train_df = train_df.replace([np.inf, -np.inf], np.nan).fillna(0)
            test_df = test_df.replace([np.inf, -np.inf], np.nan).fillna(0)
            full_df = full_df.replace([np.inf, -np.inf], np.nan).fillna(0)
            # ───────────── Optional: Generate Customer-Level Summary Stats ─────────────

                    # Determine target type
            target_is_continuous = False
            if target_column and target_column in train_df.columns:
                target_series = train_df[target_column]
                # Heuristic: continuous if numeric and > N unique values
                if pd.api.types.is_numeric_dtype(target_series):
                    unique_vals = target_series.dropna().unique()
                    if len(unique_vals) > 10:
                        target_is_continuous = True



            # Supabase paths (bucket keys)
            regressor_filename = f"{user_id}/{user_id}_best_regressor.pkl"
            classifier_filename = f"{user_id}/{user_id}_best_classifier.pkl"

            # Local save filenames (no folder nesting)
            regressor_local = PathL(model_dir) / f"{user_id}_best_regressor.pkl"
            classifier_local = PathL(model_dir) / f"{user_id}_best_classifier.pkl"

            model_path = None

            try:
                if target_is_continuous:
                    model_bytes = download_file_from_supabase(regressor_filename)
                    with open(regressor_local, "wb") as f:
                        f.write(model_bytes)
                    model_path = regressor_local
                    logger.info(f"✅ Downloaded regressor model: {regressor_filename}")
                else:
                    model_bytes = download_file_from_supabase(classifier_filename)
                    with open(classifier_local, "wb") as f:
                        f.write(model_bytes)
                    model_path = classifier_local
                    logger.info(f"✅ Downloaded classifier model: {classifier_filename}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to download model: {e}")
                raise ValueError("No trained model found for this user")

            if model_path is None:
                raise ValueError("No trained model found for this user")

            # Load the model
            model = joblib.load(model_path)
            estimator = model



            # Store expected training feature names
            if hasattr(estimator, "feature_names_in_"):
                expected_features = list(estimator.feature_names_in_)
            elif hasattr(estimator, "booster_") and hasattr(estimator.booster_, "feature_name"):
                expected_features = list(estimator.booster_.feature_name())
            else:
                # Fallback: use features from training data (excluding target and ID)
                expected_features = [col for col in train_df.columns if col not in [target_column, 'ID']]

            print(f"Expected model features: {expected_features}")
            print(f"Number of expected features: {len(expected_features)}")
            def safe_float(x):
                try:
                    v = float(x)
                    if np.isfinite(v):
                        return round(v, 6)
                except Exception:
                    pass
                return None

            def summarize_sample_for_ui(sample: pd.DataFrame):
                # Numeric table: feature | mean | median | std | min | max
                num_cols = sample.select_dtypes(include=[np.number]).columns.tolist()
                if num_cols:
                    num_df = (sample[num_cols]
                            .agg(['mean','median','std','min','max'])
                            .T.reset_index()
                            .rename(columns={'index':'feature'}))
                    for col in ['mean','median','std','min','max']:
                        num_df[col] = num_df[col].map(safe_float)
                    numeric_rows = num_df.to_dict(orient='records')
                else:
                    numeric_rows = []

                # Categorical summary: feature | top | top_count | nunique | missing
                cat_cols = [c for c in sample.columns if c not in num_cols]
                cat_rows = []
                for c in cat_cols:
                    s = sample[c]
                    vc = s.value_counts(dropna=True)
                    top = None if vc.empty else str(vc.index[0])
                    top_count = 0 if vc.empty else int(vc.iloc[0])
                    cat_rows.append({
                        "feature": str(c),
                        "top": top,
                        "top_count": top_count,
                        "nunique": int(s.nunique(dropna=True)),
                        "missing": int(s.isna().sum())
                    })

                return {
                    "numeric_table": numeric_rows,      # list[ {feature, mean, median, std, min, max} ]
                    "categorical_table": cat_rows       # list[ {feature, top, top_count, nunique, missing} ]
                }

            # ──────── Helper function for consistent feature alignment ────────
            def prepare_features_for_model(df, expected_features, target_column):
                """
                Prepare features to match model expectations exactly
                """
                # Remove ID and target columns first
                feature_df = df.drop(columns=['ID', target_column], errors='ignore')
                
                # Ensure we have all expected features
                for feature in expected_features:
                    if feature not in feature_df.columns:
                        print(f"Warning: Adding missing feature '{feature}' with default value 0")
                        feature_df[feature] = 0
                
                # Keep only expected features in the correct order
                feature_df = feature_df[expected_features]
                
                # Clean data
                feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0)
                
                return feature_df

            # ──────── Ensure consistent feature alignment ────────
            # Make sure all dataframes have the same features as the model expects
            for df_name, df in [("train_df", train_df), ("test_df", test_df), ("full_df", full_df)]:
                # Keep only model features + target column
                available_features = [col for col in expected_features if col in df.columns]
                missing_features = [col for col in expected_features if col not in df.columns]
                
                if missing_features:
                    print(f"Warning: {df_name} missing features: {missing_features}")
                    # Add missing features with default values
                    for feature in missing_features:
                        df[feature] = 0
                
                # Reorder columns to match expected features + target
                df = df[available_features + [target_column]]
                
                # Update the dataframe reference
                if df_name == "train_df":
                    train_df = df
                elif df_name == "test_df":
                    test_df = df
                elif df_name == "full_df":
                    full_df = df

            # ──────── Sample selection with proper feature alignment ────────
            # Get the base features from full dataset
            base_features = prepare_features_for_model(full_df, expected_features, target_column)
            
            if sample_id is not None and 'ID' in full_df.columns:
                sample_mask = full_df['ID'] == sample_id
                if not sample_mask.any():
                    raise ValueError(f"Sample ID {sample_id} not found")
                sample = full_df[sample_mask]
            else:
                if sample_strategy == "random":
                    sample = full_df.sample(min(num_samples, len(full_df)))
                else:
                    try:
                        # Use properly aligned features for prediction
                        if hasattr(estimator, "predict_proba"):
                            proba = estimator.predict_proba(base_features)
                            confidence = np.max(proba, axis=1)
                        else:
                            preds = estimator.predict(base_features)
                            confidence = np.abs(preds)

                        if sample_strategy == "highest_confidence":
                            idx = np.argsort(-confidence)[:num_samples]
                        elif sample_strategy == "lowest_confidence":
                            idx = np.argsort(confidence)[:num_samples]
                        else:
                            idx = np.random.choice(len(full_df), size=min(num_samples, len(full_df)), replace=False)
                        sample = full_df.iloc[idx]
                    except Exception as e:
                        print(f"Error in sample selection: {e}")
                        sample = full_df.sample(min(num_samples, len(full_df)))
            
            # ──────── Calculate sample metrics ────────
            numeric_cols = sample.select_dtypes(include=["number"]).columns.tolist()
            if target_column in numeric_cols:
                numeric_cols.remove(target_column)

           # NEW
            summary_stats = summarize_sample_for_ui(sample)

            # If your UI wants a single table, you can also render markdown/HTML:
            numeric_md = ""
            if summary_stats["numeric_table"]:
                import pandas as pd
                numeric_md = pd.DataFrame(summary_stats["numeric_table"]).to_markdown(index=False)

            categorical_md = ""
            if summary_stats["categorical_table"]:
                import pandas as pd
                categorical_md = pd.DataFrame(summary_stats["categorical_table"]).to_markdown(index=False)



            # ──────── DiCE-ML Setup with proper feature handling ────────
            try:
                import dice_ml
                from dice_ml.utils import helpers

                # CRITICAL: Prepare training data WITHOUT ID column
                training_data = train_df.copy()
                if 'ID' in training_data.columns:
                    training_data = training_data.drop(columns=['ID'])
                
                # Verify training data has expected features
                missing_in_training = [f for f in expected_features if f not in training_data.columns]
                if missing_in_training:
                    print(f"ERROR: Training data missing expected features: {missing_in_training}")
                    # Add missing features with default values
                    for feature in missing_in_training:
                        training_data[feature] = 0
                
                # Reorder columns to match expected features + target
                training_data = training_data[expected_features + [target_column]]
                
                # Identify continuous features (excluding target)
                continuous_features = [col for col in expected_features 
                                     if training_data[col].dtype in ['int64', 'float64']]
                
                print(f"Continuous features for DiCE: {continuous_features}")
                print(f"Target column: {target_column}")
                print(f"Training data shape: {training_data.shape}")
                print(f"Training data columns: {list(training_data.columns)}")
                
                # Create DiCE data object
                dice_data = dice_ml.Data(
                    dataframe=training_data, 
                    continuous_features=continuous_features, 
                    outcome_name=target_column
                )
                
                # Create DiCE model object
                dice_model = dice_ml.Model(model=estimator, backend="sklearn")
                
                # Initialize DiCE with fallback methods
                methods_to_try = ["random", "genetic"]
                dice_exp = None
                
                for method in methods_to_try:
                    try:
                        dice_exp = dice_ml.Dice(dice_data, dice_model, method=method)
                        print(f"Successfully initialized DiCE with method: {method}")
                        break
                    except Exception as method_error:
                        print(f"Failed to initialize DiCE with method {method}: {method_error}")
                        continue
                
                if dice_exp is None:
                    try:
                        dice_exp = dice_ml.Dice(dice_data, dice_model)
                        print("Successfully initialized DiCE with default method")
                    except Exception as final_error:
                        print(f"Failed to initialize DiCE with default method: {final_error}")
                        raise ValueError("Could not initialize DiCE with any method")
                        
            except ImportError:
                raise ImportError("DiCE-ML not installed. Please install with: pip install dice-ml")
            except Exception as e:
                print(f"Error initializing DiCE: {e}")
                raise
            
            # ──────── Process each sample with proper feature alignment ────────
            visualizations = []
            sample_ids = []
            counterfactuals_data = []
            
            for i, (idx, row) in enumerate(sample.iterrows()):
                try:
                    # CRITICAL FIX: Prepare sample features properly
                    sample_features_df = prepare_features_for_model(
                        pd.DataFrame([row]), expected_features, target_column
                    )
                    for col in expected_features:
                        if col in training_data.columns and col in sample_features_df.columns:
                            train_min = training_data[col].min()
                            train_max = training_data[col].max()
                            sample_features_df[col] = sample_features_df[col].clip(lower=train_min, upper=train_max)
                    print(f"Sample {i} features shape: {sample_features_df.shape}")
                    print(f"Sample {i} features columns: {list(sample_features_df.columns)}")
                    
                    # Verify features match exactly
                    if list(sample_features_df.columns) != expected_features:
                        print(f"ERROR: Feature mismatch!")
                        print(f"Sample features: {list(sample_features_df.columns)}")
                        print(f"Expected features: {expected_features}")
                        continue
                    
                    # Get current prediction
                    current_pred = estimator.predict(sample_features_df)[0]
                    current_pred = safe_float_conversion(current_pred)
                    
                    # Set desired outcome
                    if desired_outcome is None:
                        if hasattr(estimator, "predict_proba"):
                            unique_classes = np.unique(training_data[target_column])
                            if len(unique_classes) == 2:
                                desired_outcome_local = 1 - int(current_pred)
                            else:
                                desired_outcome_local = unique_classes[unique_classes != current_pred][0]
                        else:
                            desired_outcome_local = current_pred * 1.1 if current_pred > 0 else current_pred * 0.9
                    else:
                        desired_outcome_local = desired_outcome
                    
                    # Set editable features
                    if editable_features is None:
                        editable_features_local = expected_features
                    else:
                        editable_features_local = [f for f in editable_features if f in expected_features]
                    
                    # Generate counterfactuals
                    cf_examples = None
                    if hasattr(estimator, "predict_proba"):
                        # Classification
                        try:
                            cf_examples = dice_exp.generate_counterfactuals(
                                sample_features_df,
                                total_CFs=5,
                                desired_class=desired_outcome_local,
                                features_to_vary=editable_features_local,
                                verbose=False
                            )
                        except Exception as cf_error:
                            print(f"Classification CF attempt failed: {cf_error}")
                            try:
                                cf_examples = dice_exp.generate_counterfactuals(
                                    sample_features_df,
                                    total_CFs=2,
                                    desired_class=desired_outcome_local,
                                    features_to_vary=editable_features_local[:10],
                                    verbose=False
                                )
                            except Exception as cf_error2:
                                print(f"Second classification CF attempt failed: {cf_error2}")
                                cf_examples = None
                    else:
                        # Regression
                        try:
                            target_range = [
                                float(desired_outcome_local - abs(desired_outcome_local) * 0.2),
                                float(desired_outcome_local + abs(desired_outcome_local) * 0.2)
                            ]
                            cf_examples = dice_exp.generate_counterfactuals(
                                sample_features_df,
                                total_CFs=5,
                                desired_range=target_range,
                                features_to_vary=editable_features_local,
                                verbose=False
                            )
                        except Exception as cf_error:
                            print(f"Regression CF attempt failed: {cf_error}")
                            try:
                                cf_examples = dice_exp.generate_counterfactuals(
                                    sample_features_df,
                                    total_CFs=2,
                                    desired_range=target_range,
                                    features_to_vary=editable_features_local[:10],
                                    verbose=False
                                )
                            except Exception as cf_error2:
                                print(f"Second regression CF attempt failed: {cf_error2}")
                                cf_examples = None
                    
                    # Process results
                    if cf_examples and len(cf_examples.cf_examples_list) > 0 and len(cf_examples.cf_examples_list[0].final_cfs_df) > 0:
                        cf_df = cf_examples.cf_examples_list[0].final_cfs_df
                        
                        # Create comparison visualization
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
                        # Limit features for visualization readability
                        top_features = expected_features[:10]
                        
                        # Original sample
                        original_vals = sample_features_df.iloc[0][top_features]
                        ax1.barh(range(len(top_features)), original_vals.values)
                        ax1.set_yticks(range(len(top_features)))
                        ax1.set_yticklabels(top_features)
                        ax1.set_title(f'Original Sample (Prediction: {current_pred:.3f})')
                        ax1.set_xlabel('Feature Values')
                        
                        # Counterfactual
                        cf_vals = cf_df.iloc[0][top_features]
                        cf_pred = estimator.predict(cf_df.iloc[[0]])[0]
                        cf_pred = safe_float_conversion(cf_pred)
                        
                        ax2.barh(range(len(top_features)), cf_vals.values)
                        ax2.set_yticks(range(len(top_features)))
                        ax2.set_yticklabels(top_features)
                        ax2.set_title(f'Counterfactual (Prediction: {cf_pred:.3f})')
                        ax2.set_xlabel('Feature Values')
                        
                        plt.tight_layout()
                        fig_base64 = plot_to_base64(fig)
                        visualizations.append(fig_base64)
                        sample_ids.append(int(idx))
                        plt.close(fig)
                        
                        # Store counterfactual data
                        changes = {}
                        for feature in top_features:
                            orig_val = safe_float_conversion(original_vals[feature])
                            cf_val = safe_float_conversion(cf_vals[feature])
                            if abs(orig_val - cf_val) > 1e-6:
                                changes[feature] = {
                                    'original': orig_val,
                                    'counterfactual': cf_val,
                                    'change': cf_val - orig_val
                                }
                        
                        counterfactuals_data.append({
                            'sample_id': int(idx),
                            'original_prediction': current_pred,
                            'counterfactual_prediction': cf_pred,
                            'changes': changes,
                            'counterfactual_features': {k: safe_float_conversion(v) for k, v in cf_vals.to_dict().items()}
                        })
                        
                    else:
                        # No counterfactuals found - create informative visualization
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Show the original sample data
                        top_features = expected_features[:10]
                        vals = sample_features_df.iloc[0][top_features]
                        ax.barh(range(len(top_features)), vals.values)
                        ax.set_yticks(range(len(top_features)))
                        ax.set_yticklabels(top_features)
                        ax.set_title(f'Sample {idx} - No Counterfactuals Found\n(Current Prediction: {current_pred:.3f})')
                        ax.set_xlabel('Feature Values')
                        
                        # Add text with suggestions
                        ax.text(0.02, 0.98, 'Suggestions:\n• Try different editable features\n• Adjust desired outcome\n• Use wider parameter ranges', 
                               transform=ax.transAxes, verticalalignment='top', 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
                        
                        fig_base64 = plot_to_base64(fig)
                        visualizations.append(fig_base64)
                        sample_ids.append(int(idx))
                        plt.close(fig)
                        
                except Exception as e:
                    print(f"Error generating counterfactual for sample {idx}: {e}")
                    # Create error visualization
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.text(0.5, 0.5, f'Error generating counterfactual:\n{str(e)}\n\nTry adjusting parameters or preprocessing data', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.7))
                    ax.set_title(f'Sample {idx} - Counterfactual Generation Failed')
                    ax.axis('off')
                    fig_base64 = plot_to_base64(fig)
                    visualizations.append(fig_base64)
                    sample_ids.append(int(idx))
                    plt.close(fig)
            # Build baseline (original) features once
            X_orig = prepare_features_for_model(sample, expected_features, target_column)

            # Baseline predictions
            original_preds = estimator.predict(X_orig)

            # Baseline probabilities (handle binary + multiclass without assuming [:,1])
            proba_base = None
            try:
                if hasattr(estimator, "predict_proba"):
                    proba_base_raw = estimator.predict_proba(X_orig)  # (n, k) or (n,)
                    # Use helper to flatten to 1D positive-class vector (handles k==2 or multiclass)
                    proba_base = _to_1d_proba(proba_base_raw, desired_outcome=desired_outcome)
            except Exception:
                proba_base = None

            # ---- Map one CF per sample row (e.g., first CF) ----
            # Expect each item to carry which sample it belongs to; if you don't store it yet, add row_index earlier
            best_cf_by_row = {}  # row_idx -> features_modified dict
            for cf in counterfactuals_data:
                row_idx = cf.get("row_index")
                if row_idx is None:
                    # fallback: assume counterfactuals_data order matches sample order
                    row_idx = len(best_cf_by_row)
                if row_idx not in best_cf_by_row and "features_modified" in cf:
                    best_cf_by_row[row_idx] = cf["features_modified"]

            # Build X_mod in the same row order as 'sample'; rows without CF stay None
            mod_rows = [best_cf_by_row.get(i) for i in range(len(sample))]
            valid_idx = [i for i, r in enumerate(mod_rows) if r is not None]

            modified_preds = np.array([])
            proba_mod = None

            if valid_idx:
                X_mod = prepare_features_for_model(pd.DataFrame([mod_rows[i] for i in valid_idx]),
                                                expected_features, target_column)

                # Modified predictions (only for rows that have CFs)
                modified_preds = estimator.predict(X_mod)

                # Modified probabilities (flatten safely)
                try:
                    if hasattr(estimator, "predict_proba"):
                        proba_mod_raw = estimator.predict_proba(X_mod)
                        proba_mod = _to_1d_proba(proba_mod_raw, desired_outcome=desired_outcome)
                except Exception:
                    proba_mod = None
            else:
                # No CFs generated; keep arrays empty so metrics show baseline only
                modified_preds = np.array([])
                proba_mod = None

            # ---- Compute metrics (function aligns lengths automatically) ----
            sample_metrics = compute_summary_metrics(
                sample=sample,
                target_column=target_column,
                applied_changes=changes if 'changes' in locals() and isinstance(changes, dict) else {},
                original_preds=original_preds,
                modified_preds=modified_preds,      # may be shorter; helper aligns
                original_proba=proba_base,
                modified_proba=proba_mod,
                desired_outcome=desired_outcome
            )

            # ──────── Optional Summary Plot ────────
            summary_b64 = None
            if len(counterfactuals_data) > 1:
                try:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    all_changes = {}
                    for cf_data in counterfactuals_data:
                        for feature, change_info in cf_data['changes'].items():
                            if feature not in all_changes:
                                all_changes[feature] = []
                            all_changes[feature].append(change_info['change'])
                    
                    # Plot average changes per feature
                    features = list(all_changes.keys())
                    avg_changes = [np.mean(all_changes[f]) for f in features]
                    
                    ax.barh(range(len(features)), avg_changes)
                    ax.set_yticks(range(len(features)))
                    ax.set_yticklabels(features)
                    ax.set_title('Average Feature Changes in Counterfactuals')
                    ax.set_xlabel('Average Change Value')
                    
                    plt.tight_layout()
                    summary_b64 = plot_to_base64(fig)
                    plt.close(fig)
                except Exception as e:
                    print(f"Error creating summary plot: {e}")

            # ──────── Save metadata to Supabase (similar to visualization) ────────
            try:
                # Create entry for metadata
                entry = {
                    "id": str(uuid.uuid4()),
                    "created_at": datetime.utcnow().isoformat(),
                    "type": "counterfactual_dice",
                    "dataset": dataset_name,
                    "parameters": {
                        "target_column": target_column,
                        "sample_strategy": sample_strategy,
                        "num_samples": num_samples,
                        "drop_columns": drop_columns,
                        "desired_outcome": desired_outcome,
                        "editable_features": editable_features,
                        "max_changes": max_changes,
                        "proximity_metric": proximity_metric
                    },
                    "thumbnailData": f"data:image/png;base64,{visualizations[0]}" if visualizations else "",
                    "imageData": f"data:image/png;base64,{visualizations[0]}" if visualizations else "",
                    "visualizations": {},
                    "counterfactuals": counterfactuals_data,
                    "metrics": sample_metrics,
                }
            
                # Add visualizations to entry
                for i, viz_b64 in enumerate(visualizations):
                    entry["visualizations"][f"counterfactual_{i+1}"] = f"data:image/png;base64,{viz_b64}"

                if summary_b64:
                    entry["visualizations"]["counterfactual_summary"] = f"data:image/png;base64,{summary_b64}"

                # Clean entry for JSON serialization
                entry = clean_data_for_json(entry)

                try:
                    entry = ensure_json_serializable(entry)
                    with master_db_cm() as db:
                        _append_limited_metadata(user_id, entry, db=db, max_entries=5)
                except Exception as meta_error:
                    print(f"[⚠️] Metadata save error: {meta_error}")

            except Exception as meta_error:
                print(f"[⚠️] Metadata save error: {meta_error}")

            # ──────── Build response ────────
            response_data = {
                "status": "success",
                "user_id": user_id,
                "message": "Counterfactual Analysis Generated using DiCE-ML",
                "samples_analyzed": len(sample),
                "counterfactuals_generated": len(counterfactuals_data),
                "visualizations": {},
                "counterfactuals": counterfactuals_data,
                "parameters": {
                    "desired_outcome": desired_outcome,
                    "editable_features": editable_features,
                    "max_changes": max_changes,
                    "proximity_metric": proximity_metric
                },
                "metrics": sample_metrics
            }
            # Add visualizations to response
            for i, viz_b64 in enumerate(visualizations):
                response_data["visualizations"][f"counterfactual_{i+1}"] = f"data:image/png;base64,{viz_b64}"

            if summary_b64:
                response_data["visualizations"]["counterfactual_summary"] = f"data:image/png;base64,{summary_b64}"

            # Clean response data for JSON serialization
            response_data = clean_data_for_json(response_data)
            return response_data
    except Exception as e:
        print(f"[⚠️] Error in do_counterfactual: {e}")
        raise e

def generate_counterfactuals_dice(X_train, model, factual_sample, desired_class, editable_features):
    """
    Helper function to generate counterfactuals using DiCE-ML with improved parameters
    """
    import dice_ml
    from dice_ml.utils import helpers
    
    # Create a temporary dataframe with target column for DiCE
    temp_df = X_train.copy()
    temp_df["target"] = model.predict(X_train)
    
    # Identify continuous features
    continuous_features = X_train.select_dtypes('number').columns.tolist()
    
    # Create DiCE objects
    data = dice_ml.Data(
        dataframe=temp_df, 
        continuous_features=continuous_features, 
        outcome_name="target"
    )
    m = dice_ml.Model(model=model, backend="sklearn")
    exp = dice_ml.Dice(data, m, method="random")
    
    # Generate counterfactuals with improved parameters
    cf = exp.generate_counterfactuals(
        factual_sample, 
        total_CFs=10,  # Generate more CFs
        desired_class=desired_class, 
        features_to_vary=editable_features,
        proximity_weight=0.2,  # Reduce proximity weight
        diversity_weight=0.1,   # Reduce diversity weight
        verbose=False
    )
    
    return cf.cf_examples_list[0].final_cfs_df
import numpy as np

def get_retention_rate(months: float, baseline_survival: pd.DataFrame) -> float:
    time_index = baseline_survival.index.values.astype(float)
    idx = np.abs(time_index - months).argmin()
    return float(baseline_survival.iloc[idx].values[0])

def do_survival(
    user_id: str,
    current_user: dict = None,
    file_path: str = None,
    train_path: str = None,
    test_path: str = None,
    time_col: str = None,
    event_col: str = None,
    drop_cols: str = ""
) -> dict:

    import json
    import os
    import uuid
    import joblib
    import matplotlib.pyplot as plt
    import base64
    import io
    import tempfile
    from datetime import datetime
    from lifelines import CoxPHFitter
    from pathlib import Path as PathL

    # ───────────── Validate upload mode ─────────────
    if file_path and (train_path or test_path):
        raise ValueError("Provide either file_path or both train_path+test_path, not both.")
    if (train_path and not test_path) or (test_path and not train_path):
        raise ValueError("Both train_path and test_path must be provided together.")
    if not file_path and not (train_path and test_path):
        raise ValueError("Provide either a full dataset (file_path) or both train_path+test_path.")

    # Handle file downloads from Supabase
    # Note: file_path, train_path, test_path are now Supabase storage paths (e.g., "user123/dataset.csv")
    local_file_path = None
    local_train_path = None
    local_test_path = None
    temp_files = []
    
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download files from Supabase to temporary locations
            if file_path:
                # Download single file from Supabase using the storage path
                file_bytes = download_file_from_supabase(file_path)
                local_file_path = os.path.join(temp_dir, os.path.basename(file_path))
                with open(local_file_path, 'wb') as f:
                    f.write(file_bytes)
                temp_files.append(local_file_path)
            
            if train_path and test_path:
                # Download train and test files from Supabase using the storage paths
                train_bytes = download_file_from_supabase(train_path)
                test_bytes = download_file_from_supabase(test_path)
                
                local_train_path = os.path.join(temp_dir, os.path.basename(train_path))
                local_test_path = os.path.join(temp_dir, os.path.basename(test_path))
                
                with open(local_train_path, 'wb') as f:
                    f.write(train_bytes)
                with open(local_test_path, 'wb') as f:
                    f.write(test_bytes)
                    
                temp_files.extend([local_train_path, local_test_path])
            
            # Create temporary directories for models and outputs
            model_dir = os.path.join(temp_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            
            user_dir = PathL(temp_dir) / "user_outputs"
            user_dir.mkdir(exist_ok=True)

            # ───────────── Load data ─────────────
            if local_file_path:
                df = pd.read_csv(local_file_path)
                if event_col in df.columns:
                    df = df.dropna(subset=[event_col])
                dataset_name = os.path.basename(file_path)

                if drop_cols:
                    drops = [c.strip() for c in drop_cols.split(",") if c.strip() in df.columns]
                    df.drop(columns=drops, inplace=True)

                if time_col not in df.columns or event_col not in df.columns:
                    raise ValueError("time_col or event_col missing from uploaded file.")
                    
                # Parse time_col if it contains two comma-separated columns
                if time_col and "," in time_col:
                    signup_col, last_active_col = [x.strip() for x in time_col.split(",")]

                    if signup_col not in df.columns or last_active_col not in df.columns:
                        raise ValueError("One or both time columns not found in the dataset.")

                    # Convert to datetime
                    df[signup_col] = pd.to_datetime(df[signup_col], errors='coerce')
                    df[last_active_col] = pd.to_datetime(df[last_active_col], errors='coerce')

                    # Compute duration in months with decimals (approximate: 30.44 days per month)
                    df["__duration_months__"] = (
                        (df[last_active_col] - df[signup_col]).dt.total_seconds() / (60 * 60 * 24 * 30.44)
                    )
                    df["__duration_months__"] = df["__duration_months__"].clip(lower=0)

                    # Overwrite time_col to use computed duration
                    time_col = "__duration_months__"

                df, cats, nums = preprocess_data(df, RMV=[time_col, event_col])
                df = df.reset_index(drop=True)
                from sklearn.model_selection import train_test_split
                train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

            else:
                train_df = pd.read_csv(local_train_path)
                test_df = pd.read_csv(local_test_path)
                if event_col in train_df.columns:
                    train_df = train_df.dropna(subset=[event_col])

                # Only drop in test if it has the event_col
                if event_col in test_df.columns:
                    test_df = test_df.dropna(subset=[event_col])
                dataset_name = f"{os.path.basename(train_path)}+{os.path.basename(test_path)}"
                
                if drop_cols:
                    drops = [c.strip() for c in drop_cols.split(",") if c.strip() in train_df.columns]
                    train_df.drop(columns=drops, inplace=True)
                    test_drops = [c for c in drops if c in test_df.columns]
                    if test_drops:
                        test_df.drop(columns=test_drops, inplace=True)

                if time_col not in train_df.columns or event_col not in train_df.columns:
                    raise ValueError("time_col or event_col missing from training file.")
                    
                # Parse time_col if it contains two comma-separated columns
                if time_col and "," in time_col:
                    signup_col, last_active_col = [x.strip() for x in time_col.split(",")]

                    if signup_col not in train_df.columns or last_active_col not in train_df.columns:
                        raise ValueError("One or both time columns not found in the dataset.")

                    # Convert to datetime for both train and test
                    train_df[signup_col] = pd.to_datetime(train_df[signup_col], errors='coerce')
                    train_df[last_active_col] = pd.to_datetime(train_df[last_active_col], errors='coerce')

                    # Compute duration in months with decimals (approximate: 30.44 days per month)
                    train_df["__duration_months__"] = (
                        (train_df[last_active_col] - train_df[signup_col]).dt.total_seconds() / (60 * 60 * 24 * 30.44)
                    )
                    train_df["__duration_months__"] = train_df["__duration_months__"].clip(lower=0)

                    # Handle test set if it has the same columns
                    if signup_col in test_df.columns and last_active_col in test_df.columns:
                        test_df[signup_col] = pd.to_datetime(test_df[signup_col], errors='coerce')
                        test_df[last_active_col] = pd.to_datetime(test_df[last_active_col], errors='coerce')
                        test_df["__duration_months__"] = (
                            (test_df[last_active_col] - test_df[signup_col]).dt.total_seconds() / (60 * 60 * 24 * 30.44)
                        )
                        test_df["__duration_months__"] = test_df["__duration_months__"].clip(lower=0)

                    # Overwrite time_col to use computed duration
                    time_col = "__duration_months__"

                train_df, train_cats, train_nums = preprocess_data(train_df, RMV=[time_col, event_col])
                has_test_targets = time_col in test_df.columns and event_col in test_df.columns
                if has_test_targets:
                    test_df, test_cats, test_nums = preprocess_data(test_df, RMV=[time_col, event_col])
                else:
                    test_df, test_cats, test_nums = preprocess_data(test_df, RMV=[])
            train_df = train_df.reset_index(drop=True)
            test_df  = test_df.reset_index(drop=True)
            # ───────────── Clean DataFrame ─────────────
            def clean_dataframe_for_cox(df, time_col=None, event_col=None):
                df_clean = df.copy()

                # Clean feature columns (exclude time_col and event_col if provided)
                feature_cols = [c for c in df_clean.columns if c not in [time_col, event_col] and c is not None]

                for col in feature_cols:
                    if df_clean[col].dtype == 'object' or df_clean[col].dtype.name == 'category':
                        # Try to convert to numeric first
                        numeric_converted = pd.to_numeric(df_clean[col], errors='coerce')
                        if numeric_converted.notna().sum() == 0:
                            # Entire column failed conversion - treat as categorical
                            df_clean[col] = df_clean[col].astype(str).fillna("MISSING")
                            df_clean[col], _ = pd.factorize(df_clean[col])
                            df_clean[col] = df_clean[col].astype('float32')
                        else:
                            # Some values converted successfully - use median imputation
                            median_val = numeric_converted.median()
                            if pd.isna(median_val):
                                median_val = 0.0  # Fallback if all values are NaN
                            df_clean[col] = numeric_converted.fillna(median_val).astype('float32')
                    else:
                        # Numeric column - handle NaN values
                        if df_clean[col].isna().all():
                            # All values are NaN - fill with 0
                            df_clean[col] = 0.0
                        else:
                            # Use median imputation
                            median_val = df_clean[col].median()
                            if pd.isna(median_val):
                                median_val = 0.0  # Fallback
                            df_clean[col] = df_clean[col].fillna(median_val)
                        df_clean[col] = df_clean[col].astype('float32')

                # Handle time_col if provided
                if time_col and time_col in df_clean.columns:
                    # Convert to numeric and handle NaN
                    df_clean[time_col] = pd.to_numeric(df_clean[time_col], errors='coerce')
                    if df_clean[time_col].isna().all():
                        raise ValueError(f"All values in time column '{time_col}' are invalid/missing")
                    
                    median_time = df_clean[time_col].median()
                    if pd.isna(median_time):
                        median_time = 1.0  # Fallback value for time
                    
                    df_clean[time_col] = df_clean[time_col].fillna(median_time)
                    # Ensure positive values for survival time
                    df_clean[time_col] = df_clean[time_col].apply(lambda x: max(x, 0.001))

                # Handle event_col if provided
                if event_col and event_col in df_clean.columns:
                    if not pd.api.types.is_numeric_dtype(df_clean[event_col]):
                        # Convert categorical/string event column to binary
                        event_mapping = {
                            'positive': 1, 'negative': 0, 'yes': 1, 'no': 0, 'true': 1, 'false': 0,
                            '1': 1, '0': 0, 'event': 1, 'censored': 0, 'dead': 1, 'alive': 0
                        }
                        event_series = df_clean[event_col].astype(str).str.lower()
                        for key, value in event_mapping.items():
                            event_series = event_series.replace(key, value)
                        df_clean[event_col] = pd.to_numeric(event_series, errors='coerce')
                    
                    # Fill any remaining NaN values in event column with 0 (censored)
                    df_clean[event_col] = df_clean[event_col].fillna(0)
                    # Ensure binary values (0 or 1)
                    df_clean[event_col] = df_clean[event_col].apply(lambda x: 1 if x > 0.5 else 0)

                # Final check: drop any rows that still have NaN values
                initial_rows = len(df_clean)
                df_clean = df_clean.dropna(axis=0, how='any').reset_index(drop=True)
                final_rows = len(df_clean)
                
                if initial_rows != final_rows:
                    print(f"Warning: Dropped {initial_rows - final_rows} rows with remaining NaN values")
                
                # Verify no NaN values remain
                nan_cols = df_clean.columns[df_clean.isna().any()].tolist()
                if nan_cols:
                    print(f"Warning: NaN values still present in columns: {nan_cols}")
                    # Force fill any remaining NaN values
                    for col in nan_cols:
                        if col in [time_col, event_col]:
                            continue  # These should have been handled above
                        if df_clean[col].dtype in ['float32', 'float64', 'int32', 'int64']:
                            df_clean[col] = df_clean[col].fillna(0)
                        else:
                            df_clean[col] = df_clean[col].fillna('MISSING')
                            df_clean[col], _ = pd.factorize(df_clean[col])
                            df_clean[col] = df_clean[col].astype('float32')
                
                return df_clean
            def prune_for_cox(df: pd.DataFrame, time_col: str, event_col: str,
                            low_var_thresh: float = 1e-12,
                            high_corr: float = 0.98) -> pd.DataFrame:
                X = df.copy()

                # 0) Remove exact-duplicate columns (perfect multicollinearity)
                X = X.T.drop_duplicates().T

                # 1) Drop near-constant features (zero/near-zero std → causes divide-by-zero)
                feat_cols = [c for c in X.columns if c not in [time_col, event_col]]
                if feat_cols:
                    var = X[feat_cols].var(numeric_only=True)
                    drop_low_var = var[var <= low_var_thresh].index.tolist()
                    if drop_low_var:
                        X = X.drop(columns=drop_low_var, errors="ignore")

                # 2) Drop features highly correlated with the duration column (complete separation risk)
                feat_cols = [c for c in X.columns if c not in [time_col, event_col]]
                if time_col in X.columns and feat_cols:
                    with np.errstate(all="ignore"):
                        corr_to_time = X[feat_cols].corrwith(X[time_col]).abs()
                    drop_vs_time = corr_to_time[(corr_to_time >= high_corr)].index.tolist()
                    if drop_vs_time:
                        X = X.drop(columns=drop_vs_time, errors="ignore")

                # 3) Drop one of any pair of highly inter-correlated features
                feat_cols = [c for c in X.columns if c not in [time_col, event_col]]
                if len(feat_cols) > 1:
                    num = X[feat_cols].select_dtypes(include=["float32","float64","int32","int64"])
                    if num.shape[1] > 1:
                        corr = num.corr().abs()
                        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                        to_drop = [col for col in upper.columns if any(upper[col] >= high_corr)]
                        if to_drop:
                            X = X.drop(columns=to_drop, errors="ignore")

                # 4) Final safety: drop columns with <=1 unique value
                feat_cols = [c for c in X.columns if c not in [time_col, event_col]]
                drop_single = [c for c in feat_cols if X[c].nunique(dropna=True) <= 1]
                if drop_single:
                    X = X.drop(columns=drop_single, errors="ignore")

                return X

                    # ---------- Fast path: clean → (optional) stratified sample → prune → fit ----------
            def stratified_sample_for_survival(df, time_col, event_col, max_rows=10000, random_state=42):
                """Downsample large frames while keeping the event rate roughly intact."""
                n = len(df)
                if n <= max_rows:
                    return df
                if event_col not in df.columns:
                    return df.sample(max_rows, random_state=random_state)

                pos = df[df[event_col] == 1]
                neg = df[df[event_col] == 0]
                # keep the same proportion of events
                ratio = len(pos) / max(1, n)
                pos_keep = int(round(max_rows * ratio))
                pos_keep = min(pos_keep, len(pos))
                neg_keep = max_rows - pos_keep
                neg_keep = min(neg_keep, len(neg))

                out = pd.concat([
                    pos.sample(pos_keep, random_state=random_state) if pos_keep > 0 else pos.head(0),
                    neg.sample(neg_keep, random_state=random_state) if neg_keep > 0 else neg.head(0),
                ], axis=0)
                return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

            # Clean
            train_df = clean_dataframe_for_cox(train_df, time_col, event_col)
            has_test_targets = time_col in test_df.columns and event_col in test_df.columns
            if has_test_targets:
                test_df = clean_dataframe_for_cox(test_df, time_col, event_col)
            else:
                test_df = clean_dataframe_for_cox(test_df, None, None)

            print(f"Train set shape: {train_df.shape}")
            print(f"Train set NaN check: {train_df.isna().sum().sum()} total NaN values")
            print(f"Test set shape: {test_df.shape}")
            print(f"Test set NaN check: {test_df.isna().sum().sum()} total NaN values")

            # If very large, take a stratified sample BEFORE pruning/fitting (speeds up a lot)
            BIG = len(train_df) > 10000
            if BIG:
                train_df = stratified_sample_for_survival(train_df, time_col, event_col, max_rows=10000)
                print(f"Train stratified sample shape: {train_df.shape}")

            # Prune (low-var, high-corr, duplicates, single-unique)
            train_df_pruned = prune_for_cox(train_df, time_col, event_col)
            print(f"Train pruned shape: {train_df_pruned.shape}")

            test_df_pruned = test_df.copy()
            common_cols = [c for c in train_df_pruned.columns if c in test_df_pruned.columns]
            if len(common_cols) >= 2:
                test_df_pruned = test_df_pruned[common_cols].copy()
            print(f"Test pruned shape: {test_df_pruned.shape}")

            # ⛔️ Skip univariate Cox entirely. We rely on pruning + penalization instead.

            # Fit on PRUNED train only (bounded steps + ridge penalization to stabilize/accelerate)
            from lifelines import CoxPHFitter
            FIT_KW = dict(duration_col=time_col, event_col=event_col, show_progress=False)
            cph = CoxPHFitter(penalizer=2.0, l1_ratio=0.0)
            cph.fit(train_df_pruned, **FIT_KW)  # robust=False for speed; turn on if you really need robust SEs

            # Build PRUNED full dataset (align to pruned train columns)
            if has_test_targets:
                common = set(train_df_pruned.columns).intersection(set(test_df_pruned.columns))
                if time_col not in common or event_col not in common:
                    print("Warning: target cols missing in pruned test set; using pruned train only for full model.")
                    full_df_pruned = train_df_pruned.copy()
                else:
                    ordered = [c for c in train_df_pruned.columns if c in common]
                    train_subset = train_df_pruned[ordered].copy()
                    test_subset  = test_df_pruned[ordered].copy()
                    # align dtypes
                    for col in ordered:
                        if train_subset[col].dtype != test_subset[col].dtype:
                            test_subset[col] = test_subset[col].astype(train_subset[col].dtype)
                    full_df_pruned = pd.concat([train_subset, test_subset], ignore_index=True)
            else:
                print("Test set has no target columns. Using pruned train only for full model.")
                full_df_pruned = train_df_pruned.copy()

            # Final safety on full_df_pruned
            full_df_pruned = full_df_pruned.dropna(axis=0, how="any")
            bad_cols = [c for c in full_df_pruned.columns if c not in [time_col, event_col] and full_df_pruned[c].nunique(dropna=True) <= 1]
            if bad_cols:
                full_df_pruned = full_df_pruned.drop(columns=bad_cols, errors="ignore")

            # For huge full sets, cap size again to keep final fit snappy
            if len(full_df_pruned) > 20000:
                full_df_pruned = stratified_sample_for_survival(full_df_pruned, time_col, event_col, max_rows=20000)
                print(f"Full stratified sample shape: {full_df_pruned.shape}")

            # Fit final model on PRUNED full data (same bounded settings)
            cph_full = CoxPHFitter(penalizer=2.0, l1_ratio=0.0)
            cph_full.fit(full_df_pruned, **FIT_KW)

            # Metrics
            c_index = cph_full.concordance_index_
            log_likelihood = cph_full.log_likelihood_
            k_params = len(cph_full.params_)
            n_obs = full_df_pruned.shape[0]
            aic = float(-2 * log_likelihood + 2 * k_params)
            bic = float(-2 * log_likelihood + k_params * np.log(n_obs))
            significant_predictors = int((cph_full.summary["p"] < 0.05).sum())

            survival_metrics = {
                "concordance_index": round(float(c_index), 4),
                "log_likelihood": round(float(log_likelihood), 4),
                "AIC": round(aic, 4),
                "BIC": round(bic, 4),
                "significant_predictors": significant_predictors
            }

            # Save model to temp and upload (with filename arg)
            temp_model_path = PathL(model_dir) / f"{user_id}_survival_cox.pkl"
            joblib.dump(cph_full, temp_model_path)

            try:
                with open(temp_model_path, 'rb') as f:
                    model_bytes = f.read()
                supabase_model_path = f"models/{user_id}_survival_cox.pkl"
                upload_file_to_supabase(
                    model_bytes,
                    supabase_model_path,
                    f"{user_id}_survival_cox.pkl"  # filename arg
                )
                print(f"Model uploaded to Supabase: {supabase_model_path}")
            except Exception as upload_error:
                print(f"Warning: Failed to upload model to Supabase: {upload_error}")

            # Example: baseline survival & retention (unchanged)
            baseline_survival = cph_full.baseline_survival_
            retention_rates = {
                "1_month": round(get_retention_rate(1, baseline_survival) * 100, 2),
                "3_months": round(get_retention_rate(3, baseline_survival) * 100, 2),
                "6_months": round(get_retention_rate(6, baseline_survival) * 100, 2),
            }

            
            # Baseline survival curve
            fig, ax = plt.subplots(figsize=(10, 6))
            cph_full.baseline_survival_.plot(ax=ax)
            ax.set_xlabel('Time')
            ax.set_ylabel('Survival Probability')
            ax.set_title('Baseline Survival Curve')
            plt.grid(True)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches="tight")
            buf.seek(0)
            baseline_b64 = base64.b64encode(buf.read()).decode()
            plt.close()

            # Cox coefficients plot
            coef_df = cph_full.summary.sort_values('coef')
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            y_pos = range(len(coef_df))
            ax2.errorbar(coef_df['coef'], y_pos, 
                         xerr=[coef_df['coef'] - coef_df['coef lower 95%'], 
                               coef_df['coef upper 95%'] - coef_df['coef']], 
                         fmt='o', capsize=5)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(coef_df.index)
            ax2.set_xlabel('Log Hazard Ratio')
            ax2.set_title('Cox Model Coefficients with 95% Confidence Intervals')
            ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            plt.grid(True)
            buf2 = io.BytesIO()
            plt.savefig(buf2, format='png', bbox_inches="tight")
            buf2.seek(0)
            hazard_b64 = base64.b64encode(buf2.read()).decode()
            plt.close()

            # ───────────── Generate predictions for risk analysis ─────────────
            try:
                # Generate risk scores for the training set
                risk_scores = cph_full.predict_partial_hazard(train_df.drop(columns=[time_col, event_col]))
                
                # Calculate risk probabilities (normalize to 0-1 range)
                risk_prob = (risk_scores - risk_scores.min()) / (risk_scores.max() - risk_scores.min())
                
                # Add risk scores to a copy of the training data for analysis
                risk_df = train_df.copy()
                risk_df["risk_prob"] = risk_prob
                
                high_risk_threshold = 0.7
                high_risk_pct = (risk_prob > high_risk_threshold).mean() * 100
                
                # Create risk distribution plot
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                ax3.hist(risk_prob, bins=30, alpha=0.7, edgecolor='black')
                ax3.axvline(x=high_risk_threshold, color='red', linestyle='--', label=f'High Risk Threshold ({high_risk_threshold})')
                ax3.set_xlabel('Risk Probability')
                ax3.set_ylabel('Frequency')
                ax3.set_title('Distribution of Risk Scores')
                ax3.legend()
                plt.grid(True, alpha=0.3)
                buf3 = io.BytesIO()
                plt.savefig(buf3, format='png', bbox_inches="tight")
                buf3.seek(0)
                risk_fig_base64 = base64.b64encode(buf3.read()).decode()
                plt.close()
                
            except Exception as e:
                print(f"Warning: Could not generate risk analysis: {e}")
                high_risk_pct = 0
                risk_df = train_df.copy()
                risk_df["risk_prob"] = 0
                risk_fig_base64 = None

            # ───────────── Save Metadata to Supabase ─────────────
            try:
                # Create entry for metadata
                entry = {
                    "id": str(uuid.uuid4()),
                    "created_at": datetime.utcnow().isoformat(),
                    "type": "survival",
                    "dataset": dataset_name,
                    "time_column": time_col,
                    "event_column": event_col,
                    "parameters": {"drop_cols": drop_cols},
                    "thumbnailUrl": f"data:image/png;base64,{baseline_b64}",
                    "imageUrl": f"data:image/png;base64,{baseline_b64}",
                    "visualizations": {
                        "baseline_curve": f"data:image/png;base64,{baseline_b64}",
                        "cox_coefficients": f"data:image/png;base64,{hazard_b64}"
                    },
                    "metrics": survival_metrics,
                    "retention_rates": retention_rates,
                }

                if risk_fig_base64:
                    entry["visualizations"]["risk_distribution"] = f"data:image/png;base64,{risk_fig_base64}"

                try:
                    entry = ensure_json_serializable(entry)
                    with master_db_cm() as db:
                        _append_limited_metadata(user_id, entry, db=db, max_entries=5)
                except Exception as meta_error:
                    print(f"[⚠️] Metadata save error: {meta_error}")

            except Exception as meta_error:
                print(f"[⚠️] Metadata save error: {meta_error}")

            # ───────────── Response ─────────────
            response_data = {
                "status": "success",
                "user_id": user_id,
                "dataset": dataset_name,
                "message": "Survival Analysis Complete",
                "high_risk_percentage": high_risk_pct,
                "mean_risk_score": float(risk_df["risk_prob"].mean()),
                "visualizations": {
                    "baseline_curve": f"data:image/png;base64,{baseline_b64}",
                    "cox_coefficients": f"data:image/png;base64,{hazard_b64}"
                },
                "metrics": survival_metrics,
                "retention_rates": retention_rates,
            }

            if risk_fig_base64:
                response_data["visualizations"]["risk_distribution"] = f"data:image/png;base64,{risk_fig_base64}"

            return response_data

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise ValueError(f"Survival analysis failed: {str(e)}")
def do_what_if(
    user_id: str,
    current_user: dict = None,
    file_path: str = None,
    train_path: str = None,
    test_path: str = None,
    target_column: str = "target",
    drop_columns: str = "",
    feature_changes: str = "",
    sample_id: int = None,
    bulk_changes: dict = None,
    analysis_config: dict = None
) -> dict:
    """
    Enhanced what-if analysis for mass scenario simulation.
    
    Purpose: Allow users to simulate mass changes across all records,
    e.g., "What if mntMeatProducts for all orders went up by 20%?"
    
    Args:
        user_id: User identifier
        current_user: User context (for permissions/logging)
        file_path: Path to single dataset file
        train_path: Path to training dataset
        test_path: Path to test dataset
        target_column: Target variable column name
        drop_columns: Comma-separated columns to drop
        feature_changes: Individual feature changes (JSON string)
        sample_id: ID of specific sample to analyze (optional)
        bulk_changes: Bulk changes dictionary
        analysis_config: Configuration for analysis parameters
    
    Returns:
        dict: Analysis results with metrics, visualizations, and insights
    """
    
    import numpy as np
    import pandas as pd
    import os
    import uuid
    import joblib
    import io
    import base64
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tempfile
    import json
    import logging
    from pathlib import Path as PathL
    from datetime import datetime
    from typing import Dict, List, Any, Optional, Tuple
    from dataclasses import dataclass
    
    # Configure matplotlib for better plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    @dataclass
    class AnalysisConfig:
        """Configuration for what-if analysis parameters"""
        revenue_per_conversion: float = 20.0
        confidence_level: float = 0.95
        n_clusters: int = 3
        top_features_count: int = 5
        significance_threshold: float = 0.05
        sample_size_limit: int = 100000
        min_count_for_ztest: int = 5

    @dataclass
    class ValidationResult:
        """Result of input validation"""
        is_valid: bool
        errors: List[str]
        warnings: List[str]
    
    def validate_inputs() -> ValidationResult:
        """Comprehensive input validation"""
        errors = []
        warnings = []

        # Validate file paths - match risk_analysis pattern
        if file_path and (train_path or test_path):
            errors.append("Provide either file_path or both train_path+test_path, not both")
        elif (train_path and not test_path) or (test_path and not train_path):
            errors.append("Both train_path and test_path must be provided together")
        elif not file_path and not (train_path and test_path):
            errors.append("Provide either a full dataset (file_path) or both train_path+test_path")
            
        # Validate target column
        if not target_column or not isinstance(target_column, str):
            errors.append("Valid target_column is required")
            
        # Validate changes
        if not bulk_changes and not feature_changes:
            warnings.append("No changes specified - analysis will show baseline predictions")
            
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    def safe_download_and_load_data() -> Tuple[pd.DataFrame, str]:
        """Safely download and load data with proper error handling"""
        try:
            if file_path:
                # Download single file from Supabase
                file_bytes = download_file_from_supabase(file_path)
                local_file_path = os.path.join(temp_dir, f"data_{uuid.uuid4().hex[:8]}.csv")
                with open(local_file_path, 'wb') as f:
                    f.write(file_bytes)
                
                df = pd.read_csv(local_file_path)
                dataset_name = os.path.basename(file_path)
                
            else:
                # Download train and test files from Supabase
                train_bytes = download_file_from_supabase(train_path)
                test_bytes = download_file_from_supabase(test_path)
                
                local_train_path = os.path.join(temp_dir, f"train_{uuid.uuid4().hex[:8]}.csv")
                local_test_path = os.path.join(temp_dir, f"test_{uuid.uuid4().hex[:8]}.csv")
                
                with open(local_train_path, 'wb') as f:
                    f.write(train_bytes)
                with open(local_test_path, 'wb') as f:
                    f.write(test_bytes)
                
                train_df = pd.read_csv(local_train_path)
                test_df = pd.read_csv(local_test_path)
                df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
                dataset_name = f"{os.path.basename(train_path)}+{os.path.basename(test_path)}"
            
            # Remove rows with missing target values
            initial_rows = len(df)
            df = df.dropna(subset=[target_column])
            final_rows = len(df)
            
            if final_rows == 0:
                raise ValueError(f"No valid rows remaining after removing missing values in '{target_column}'")
            
            if initial_rows != final_rows:
                logging.warning(f"Removed {initial_rows - final_rows} rows with missing target values")
            
            # Drop specified columns
            if drop_columns:
                drops = [c.strip() for c in drop_columns.split(",") if c.strip() and c in df.columns]
                if drops:
                    df.drop(columns=drops, inplace=True)
                    logging.info(f"Dropped columns: {drops}")
            
            # Validate data quality
            if df.empty:
                raise ValueError("Dataset is empty after processing")
                
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset. Available columns: {list(df.columns)}")
            
            # Limit dataset size for performance
            if len(df) > config.sample_size_limit:
                df = df.sample(n=config.sample_size_limit, random_state=42)
                logging.warning(f"Dataset limited to {config.sample_size_limit} rows for performance")
            
            return df, dataset_name
            
        except Exception as e:
            raise ValueError(f"Failed to load data: {str(e)}")
    def detect_task_type(y: pd.Series) -> str:
        y_nonnull = y.dropna()
        nunique = y_nonnull.nunique()

        # clear binary cases
        unique_vals = set(pd.unique(y_nonnull))
        if pd.api.types.is_bool_dtype(y_nonnull) or unique_vals <= {0, 1}:
            return "classification"

        # small-cardinality heuristic (works on small datasets too)
        thresh = min(20, max(2, int(0.1 * len(y_nonnull))))
        if nunique <= thresh:
            return "classification"

        return "regression"

    def load_model(df: pd.DataFrame, target_column: str) -> Any:
        """Load classifier or regressor based on the target column's distribution"""
        model_dir = os.path.join(temp_dir, "models")
        os.makedirs(model_dir, exist_ok=True)

        # Determine whether the task is classification or regression
        target_values = df[target_column].dropna()
        num_unique = target_values.nunique()
        total = len(target_values)

        target_values = df[target_column].dropna()
        selected_model_type = "classifier" if detect_task_type(target_values) == "classification" else "regressor"

        logging.info(f"🔍 Target column '{target_column}' detected as: {selected_model_type.upper()}")
        task_type = detect_task_type(target_values)
        # Choose the correct model path
        model_paths = {
            "classifier": (
                PathL(model_dir) / f"{user_id}_best_classifier.pkl",
                f"{user_id}/{user_id}_best_classifier.pkl"
            ),
            "regressor": (
                PathL(model_dir) / f"{user_id}_best_regressor.pkl",
                f"{user_id}/{user_id}_best_regressor.pkl"
            )
        }

        local_path, remote_path = model_paths[selected_model_type]

        try:
            if not local_path.exists():
                model_bytes = download_file_from_supabase(remote_path)
                with open(local_path, 'wb') as f:
                    f.write(model_bytes)

            model = joblib.load(local_path)
            logging.info(f"✅ Successfully loaded {selected_model_type} model from Supabase: {remote_path}")
            return model, task_type

        except Exception as e:
            raise ValueError(f"❌ Failed to load {selected_model_type} model: {str(e)}")
    def sanitize_for_json(obj):
        """Recursively sanitize data for JSON serialization."""
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(sanitize_for_json(v) for v in obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return sanitize_for_json(obj.tolist())  # New: handle arrays
        elif isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        return obj
    import logging
    from typing import Dict, Any, Tuple, Optional
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    def apply_feature_changes(
        df: pd.DataFrame,
        changes: Dict[str, Any],
        encoders: Optional[Dict[str, LabelEncoder]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply bulk feature changes even if some features are label-encoded or one-hot–encoded.

        - encoders: optional mapping from column → LabelEncoder
        - One-hot dummies detected by prefix "<feature>_<category>"
        """
        modified_df = df.copy(deep=True)
        applied_changes: Dict[str, Any] = {}

        numeric_cols = set(modified_df.select_dtypes(include=[np.number]).columns)

        def safe_to_num(s: pd.Series) -> pd.Series:
            return pd.to_numeric(s, errors="coerce")

        for feature, operation in changes.items():
            # ——— 1) Label-encoded? ———
            if encoders and feature in encoders:
                le = encoders[feature]
                codes = modified_df.loc[:, feature].astype(int)
                orig_lbl = pd.Series(le.inverse_transform(codes), index=modified_df.index)

                # Recurse on the string labels
                tmp_df = pd.DataFrame({feature: orig_lbl})
                new_df, tmp_changes = apply_feature_changes(tmp_df, {feature: operation}, encoders=None)
                new_lbl = new_df[feature].astype(str)

                # Re-encode
                new_codes = pd.Series(le.transform(new_lbl), index=modified_df.index)
                modified_df.loc[:, feature] = new_codes
                applied_changes[feature] = {**tmp_changes[feature], "encoded_with": "label"}
                continue

            # ——— 2) One-hot–encoded? ———
            dummy_cols = [c for c in modified_df.columns if c.startswith(feature + "_")]
            if dummy_cols:
                dummies = modified_df[dummy_cols].fillna(0).astype(int)
                orig_lbl = dummies.idxmax(axis=1).str[len(feature) + 1:]
                tmp_df = pd.DataFrame({feature: orig_lbl})
                new_df, tmp_changes = apply_feature_changes(tmp_df, {feature: operation}, encoders=None)
                new_lbl = new_df[feature].astype(str)

                # Rebuild dummies
                new_d = pd.get_dummies(new_lbl, prefix=feature)
                for col in dummy_cols:
                    modified_df.loc[:, col] = new_d.get(col, 0).astype(int)
                # Add any brand-new category columns
                for col in new_d.columns:
                    if col not in dummy_cols:
                        modified_df.loc[:, col] = new_d[col].astype(int)

                applied_changes[feature] = {
                    **tmp_changes[feature],
                    "encoded_with": "one-hot",
                    "dummy_columns": dummy_cols
                }
                continue

            # ——— 3) Regular numeric vs categorical ———
            try:
                col_series = modified_df.loc[:, feature].copy()
                is_numeric = feature in numeric_cols

                if isinstance(operation, dict) and "type" in operation:
                    op = operation["type"]
                    raw = operation.get("value", 0)

                    # —— Numeric ops ——
                    if is_numeric and op in {
                        "percentage", "percent_increase", "percent_decrease",
                        "additive", "multiplicative", "set_value"
                    }:
                        pct = float(raw)
                        if op == "percentage":
                            with np.errstate(divide='ignore', invalid='ignore'):
                                factor = 1 + pct/100
                                col_series = safe_to_num(col_series) * factor
                            desc = f"{'Increased' if pct>=0 else 'Decreased'} {feature} by {abs(pct)}%"
                        elif op == "percent_increase":
                            with np.errstate(divide='ignore', invalid='ignore'):
                                col_series = safe_to_num(col_series) * (1 + pct/100)
                            desc = f"Increased {feature} by {pct}%"
                        elif op == "percent_decrease":
                            with np.errstate(divide='ignore', invalid='ignore'):
                                col_series = safe_to_num(col_series) * (1 - pct/100)
                            desc = f"Decreased {feature} by {pct}%"
                        elif op == "additive":
                            col_series = safe_to_num(col_series) + pct
                            desc = f"Added {pct} to {feature}"
                        elif op == "multiplicative":
                            col_series = safe_to_num(col_series) * pct
                            desc = f"Multiplied {feature} by {pct}"
                        else:  # set_value
                            col_series = raw
                            desc = f"Set {feature} to {raw}"

                        # clean up any NaNs/Infs
                        col_series = col_series.fillna(0)

                    # —— Categorical ops ——
                    elif not is_numeric and op in {
                        "set_value", "categorical_replace", "categorical_boost"
                    }:
                        if op == "set_value":
                            col_series = raw
                            desc = f"Set {feature} to {raw}"

                        elif op == "categorical_replace":
                            old = operation.get("old_value", "all")
                            new = raw
                            if old == "all":
                                col_series = new
                                desc = f"Replaced all {feature} with {new}"
                            else:
                                mask = col_series == old
                                cnt = mask.sum()
                                col_series.loc[mask] = new
                                desc = f"Replaced {cnt} rows where {feature}=={old} with {new}"

                        else:  # categorical_boost
                            target = raw
                            prop = min(max(operation.get("proportion", 0.1), 0), 1)
                            non_mask = col_series != target
                            non_idx = col_series.index[non_mask]
                            n = int(len(non_idx) * prop)
                            col_series = col_series.copy()
                            if n:
                                chosen = np.random.choice(non_idx, size=n, replace=False)
                                col_series.loc[chosen] = target
                            desc = (
                                f"Boosted {prop*100:.1f}% of non-{target} "
                                f"records ({n} rows) of {feature} to {target}"
                            )
                    else:
                        logging.warning(f"Op '{op}' not valid for '{feature}'")
                        continue

                    modified_df.loc[:, feature] = col_series
                    applied_changes[feature] = {"type": op, "value": raw, "description": desc}

                else:
                    # literal assignment
                    modified_df.loc[:, feature] = operation
                    applied_changes[feature] = {
                        "type": "set_value",
                        "value": operation,
                        "description": f"Set {feature} to {operation}"
                    }

            except Exception as e:
                logging.error(f"Failed to apply change to {feature}: {e}")

        return modified_df, applied_changes
        
    def calculate_advanced_metrics(
        original_preds: np.ndarray,
        modified_preds: np.ndarray,
        task_type: str = "regression",
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Compute paired metrics for regression or classification,
        using z-test/Fisher for binary classification and paired t-test for regression.
        """
        # 1) Clean inputs
        orig = np.nan_to_num(original_preds, nan=0.0, posinf=0.0, neginf=0.0)
        mod  = np.nan_to_num(modified_preds, nan=0.0, posinf=0.0, neginf=0.0)

        metrics: Dict[str, Any] = {}

        if task_type == "classification":
            # Binarize if necessary
            if orig.dtype.kind == "f":
                orig_cls = (orig >= 0.5).astype(int)
                mod_cls  = (mod  >= 0.5).astype(int)
            else:
                orig_cls = orig.astype(int)
                mod_cls  = mod.astype(int)

            # Rates & percent change
            orig_rate = orig_cls.mean()
            mod_rate  = mod_cls.mean()
            with np.errstate(divide='ignore', invalid='ignore'):
                pct_chg = ((mod_rate - orig_rate)/orig_rate*100) if orig_rate else 0.0

            # Choose z-test vs Fisher
            count = np.array([mod_cls.sum(), orig_cls.sum()])
            nobs  = np.array([len(mod_cls),    len(orig_cls)])
            min_n = 5
            if all(count >= min_n) and all(nobs - count >= min_n):
                stat, pval = proportions_ztest(count, nobs)
                test_name = "z-test"
            else:
                a, c = count
                b, d = nobs - count
                _, pval = fisher_exact([[a, b], [c, d]])
                test_name = "fisher"

            metrics.update({
                "original_rate": round(orig_rate, 4),
                "modified_rate": round(mod_rate, 4),
                "percent_change": round(pct_chg, 2),
                "test": test_name,
                "p_value": float(pval),
                "is_significant": bool(pval < (1 - confidence_level)),
                "confidence_level": confidence_level
            })

        else:
            # Regression (paired)
            orig_mean = float(orig.mean())
            mod_mean  = float(mod.mean())
            orig_std  = float(orig.std())
            mod_std   = float(mod.std())

            with np.errstate(divide='ignore', invalid='ignore'):
                pct_chg = ((mod_mean - orig_mean)/orig_mean*100) if orig_mean else 0.0

            # Paired t-test
            try:
                stat, pval = ttest_rel(mod, orig, nan_policy="omit")
            except Exception as e:
                logging.warning(f"Paired t-test failed: {e}")
                stat, pval = None, None

            # 95% CI on difference
            diff = mod - orig
            try:
                ci_low, ci_high = t.interval(
                    confidence_level, len(diff)-1,
                    loc=diff.mean(), scale=sem(diff)
                )
                ci = [float(ci_low), float(ci_high)]
            except Exception:
                ci = [None, None]

            # Cohen's d
            pooled = np.sqrt((orig_std**2 + mod_std**2)/2) if (orig_std or mod_std) else 0
            d = ((mod_mean - orig_mean)/pooled) if pooled else 0

            metrics.update({
                "original_mean": round(orig_mean, 4),
                "modified_mean": round(mod_mean, 4),
                "percent_change": round(pct_chg, 2),
                "t_statistic": stat,
                "p_value": pval,
                "is_significant": bool(pval < (1 - confidence_level)) if pval is not None else None,
                "confidence_interval": ci,
                "effect_size": round(d, 4),
                "confidence_level": confidence_level
            })

        # Shared metrics
        var_chg = round(float(mod.std() - orig.std()), 4)
        metrics["variance_change"] = var_chg
        metrics["risk_assessment"] = (
            "higher_risk" if var_chg > 0 else
            "lower_risk"  if var_chg < 0 else
            "similar_risk"
        )
        # Revenue impact (example $20/unit)
        rev_imp = (metrics.get("modified_mean", 0) - metrics.get("original_mean", 0)) \
                * len(orig) * 20.0
        metrics["estimated_revenue_impact"] = round(rev_imp, 2)

        return metrics
    def create_enhanced_visualizations(
        original_preds: np.ndarray,
        modified_preds: np.ndarray,
        df: pd.DataFrame,
        modified_df: pd.DataFrame,
        feature_cols: List[str],
        applied_changes: Dict[str, Any],
        metrics: Dict[str, Any],
        task_type: str
    ) -> Dict[str, str]:
        """
        Create comprehensive visualizations for what-if analysis.

        Parameters:
        - original_preds, modified_preds: arrays of model outputs
        - df, modified_df: before/after feature DataFrames
        - feature_cols: list of feature names considered
        - applied_changes: record of what was changed
        - metrics: computed metrics (rate or mean, p_value, etc.)
        - task_type: either "classification" or "regression"
        """
        import io, base64
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy import stats

        visualizations: Dict[str, str] = {}

        # ─── 1. MAIN DISTRIBUTION ANALYSIS ───
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        # distribution
        sns.histplot(original_preds, alpha=0.7, label="Original",
                    ax=axes[0,0], stat="density")
        sns.histplot(modified_preds, alpha=0.7, label="Modified",
                    ax=axes[0,0], stat="density")
        axes[0,0].set_title("Prediction Distributions")
        axes[0,0].legend()

        # boxplot
        combined = np.concatenate([original_preds, modified_preds])
        labels = ['Original'] * len(original_preds) + ['Modified'] * len(modified_preds)
        box_df = pd.DataFrame({'Predictions': combined, 'Type': labels})
        sns.boxplot(data=box_df, x='Type', y='Predictions', ax=axes[0,1])
        axes[0,1].set_title("Prediction Box Plot Comparison")

        # scatter
        axes[1,0].scatter(original_preds, modified_preds, alpha=0.6)
        mmin, mmax = min(original_preds), max(original_preds)
        axes[1,0].plot([mmin, mmax], [mmin, mmax], 'r--', alpha=0.8)
        axes[1,0].set_title("Original vs Modified Predictions")

        # diff histogram
        diffs = modified_preds - original_preds
        axes[1,1].hist(diffs, bins=30, alpha=0.7, edgecolor='black')
        axes[1,1].axvline(np.mean(diffs), color='red', linestyle='--',
                        label=f"Mean: {np.mean(diffs):.3f}")
        axes[1,1].set_title("Distribution of Prediction Changes")
        axes[1,1].legend()

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        visualizations["distribution_analysis"] = (
            "data:image/png;base64," +
            base64.b64encode(buf.read()).decode()
        )
        plt.close(fig)

        # ─── 2. FEATURE IMPACT ANALYSIS ───
        if applied_changes:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # magnitude bar
            names, vals, types = [], [], []
            for feat, ch in applied_changes.items():
                names.append(feat)
                t = ch['type']
                if t in ('percent_increase', 'percent_decrease'):
                    vals.append(abs(ch['value']))
                elif t == 'additive':
                    vals.append(abs(ch['value']))
                elif t == 'multiplicative':
                    vals.append(abs(ch['value'] - 1) * 100)
                else:
                    vals.append(1)
                types.append(t.replace('_',' ').title())
            bars = axes[0,0].bar(range(len(names)), vals,
                                color=plt.cm.Set3(np.linspace(0,1,len(names))))
            axes[0,0].set_xticks(range(len(names)))
            axes[0,0].set_xticklabels(names, rotation=45, ha='right')
            axes[0,0].set_title("Magnitude of Feature Changes")
            for b,v in zip(bars, vals):
                axes[0,0].text(b.get_x()+b.get_width()/2, v+max(vals)*0.01,
                            f"{v:.1f}", ha='center', va='bottom', fontsize=9)

            # correlation
            if len(applied_changes) > 1:
                corr_data = []
                for feat in applied_changes:
                    if feat in df and feat in modified_df:
                        orig_vals = df[feat].fillna(0)
                        mod_vals = modified_df[feat].fillna(0)
                        change = mod_vals - orig_vals
                        if len(change)==len(diffs):
                            c = np.corrcoef(change, diffs)[0,1]
                            if not np.isnan(c):
                                corr_data.append((feat, c))
                if corr_data:
                    feats, corrs = zip(*corr_data)
                    cols = ['red' if c<0 else 'green' for c in corrs]
                    axes[0,1].barh(range(len(feats)), corrs, color=cols, alpha=0.7)
                    axes[0,1].set_yticks(range(len(feats)))
                    axes[0,1].set_yticklabels(feats)
                    axes[0,1].set_title("Feature–Prediction Change Correlation")
                else:
                    axes[0,1].text(0.5,0.5,"No valid correlations",
                                ha='center',va='center',fontsize=12)
            else:
                axes[0,1].text(0.5,0.5,"Multiple features needed\nfor correlation",
                            ha='center',va='center',fontsize=12)
                axes[0,1].set_title("Correlation Analysis")

            # impact ranking
            impacts = []
            for feat in applied_changes:
                if feat in df and feat in modified_df:
                    o = pd.to_numeric(df[feat],errors='coerce').fillna(0)
                    m = pd.to_numeric(modified_df[feat],errors='coerce').fillna(0)
                    impacts.append((feat, abs(m.mean()-o.mean())))
            if impacts:
                impacts.sort(key=lambda x: x[1], reverse=True)
                top_feats, top_vals = zip(*impacts[:5])
                bars = axes[1,0].bar(range(len(top_feats)), top_vals,
                                    color=plt.cm.viridis(np.linspace(0,1,len(top_feats))))
                axes[1,0].set_xticks(range(len(top_feats)))
                axes[1,0].set_xticklabels(top_feats,rotation=45,ha='right')
                axes[1,0].set_title("Top 5 Feature Impact")
                for b,v in zip(bars, top_vals):
                    axes[1,0].text(b.get_x()+b.get_width()/2, v+max(top_vals)*0.01,
                                f"{v:.3f}",ha='center',va='bottom',fontsize=9)
            else:
                axes[1,0].text(0.5,0.5,"No numerical features",ha='center',va='center',fontsize=12)
                axes[1,0].set_title("Impact Ranking")

            # change type distribution
            counts = {}
            for ch in applied_changes.values():
                counts[ch['type']] = counts.get(ch['type'],0) + 1
            if counts:
                types, ct = zip(*counts.items())
                axes[1,1].pie(ct, labels=types, autopct='%1.1f%%', startangle=90)
                axes[1,1].set_title("Change Types Distribution")
            else:
                axes[1,1].text(0.5,0.5,"No change types to display",
                            ha='center',va='center',fontsize=12)
                axes[1,1].set_title("Change Types")

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            visualizations["feature_impact_analysis"] = (
                "data:image/png;base64," +
                base64.b64encode(buf.read()).decode()
            )
            plt.close(fig)

        # ─── 3. STATISTICAL SIGNIFICANCE VISUALIZATION ───
        diffs = modified_preds - original_preds
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        orig_r = metrics.get("original_rate",
                 float((original_preds >= 0.5).mean()))
        mod_r  = metrics.get("modified_rate",
                 float((modified_preds  >= 0.5).mean()))
        test_name = metrics.get("test", "n/a")
        p_val     = metrics.get("p_value", None)

        axes[0,0].bar([0,1], [orig_r, mod_r], alpha=0.7)
        axes[0,0].set_xticks([0,1])
        axes[0,0].set_xticklabels(["Original rate","Modified rate"])
        axes[0,0].set_ylabel("Proportion")
        if p_val is not None:
            axes[0,0].set_title(
                f"Positive‐Rate Change ({test_name} p={p_val:.3f})"
            )
        else:
            axes[0,0].set_title("Positive‐Rate Change")
            # CI for rate difference
            n = len(original_preds)
            with np.errstate(divide='ignore', invalid='ignore'):
                # approximate se
                se = np.sqrt((orig_r*(1-orig_r) + mod_r*(1-mod_r)) / n)
                z = stats.norm.ppf(1 - (1-metrics.get("confidence_level",0.95))/2)
                margin = se * z
            diff = mod_r - orig_r
            axes[0,1].barh([0], [diff], xerr=[margin], alpha=0.7)
            axes[0,1].axvline(0, color='black', linestyle='--')
            axes[0,1].set_yticks([0])
            axes[0,1].set_yticklabels([f"{int(metrics.get('confidence_level',0.95)*100)}% CI"])
            axes[0,1].set_title("Proportion Difference CI")

            # blank out regression‐only plots
            axes[1,0].axis('off')
            axes[1,1].axis('off')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        visualizations["statistical_analysis"] = (
            "data:image/png;base64," +
            base64.b64encode(buf.read()).decode()
        )
        plt.close(fig)

       # ===== 4. PREDICTION IMPACT SUMMARY =====
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3,
                            height_ratios=[1, 1, 1],
                            width_ratios=[1, 1, 1])

        if task_type == "regression":
            # 4.1 Percentile comparison
            ax1 = fig.add_subplot(gs[0, :2])
            pctiles = [5, 10, 25, 50, 75, 90, 95]
            orig_pct = np.percentile(original_preds, pctiles)
            mod_pct  = np.percentile(modified_preds, pctiles)
            x = np.arange(len(pctiles))
            w = 0.35
            ax1.bar(x - w/2, orig_pct, w, label="Original", alpha=0.7)
            ax1.bar(x + w/2, mod_pct, w, label="Modified", alpha=0.7)
            ax1.set_xticks(x)
            ax1.set_xticklabels([f"{p}th" for p in pctiles])
            ax1.set_title("Prediction Percentiles Comparison")
            ax1.set_ylabel("Prediction Value")
            ax1.legend()

            # 4.2 Variance gauge
            ax2 = fig.add_subplot(gs[0, 2])
            var_ch = np.var(modified_preds) - np.var(original_preds)
            base_var = np.var(original_preds)
            score = (var_ch / base_var * 100) if base_var > 0 else 0
            # define risk levels
            if abs(score) < 5:
                lvl, col = "Low Risk", "#66c2a5"
            elif abs(score) < 15:
                lvl, col = "Medium Risk", "#fc8d62"
            else:
                lvl, col = "High Risk", "#e78ac3"
            ax2.barh([0], [abs(score)], color=col, alpha=0.8)
            ax2.set_xlim(0, max(abs(score)*1.2, 5))
            ax2.set_yticks([0]); ax2.set_yticklabels([lvl])
            ax2.set_xlabel("% Change in Variance")
            ax2.set_title("Variance Change Risk")

            # 4.3 Direction of change
            ax3 = fig.add_subplot(gs[1, :])
            diffs = modified_preds - original_preds
            counts = {
                "Increase": np.sum(diffs > 0),
                "Decrease": np.sum(diffs < 0),
                "No Change": np.sum(diffs == 0)
            }
            colors = ["#66c2a5", "#fc8d62", "#8da0cb"]
            ax3.bar(counts.keys(), counts.values(), color=colors, alpha=0.8)
            ax3.set_title("Direction of Prediction Changes")
            for i,(k,v) in enumerate(counts.items()):
                pct = v / len(diffs) * 100
                ax3.text(i, v + len(diffs)*0.01, f"{v}\n({pct:.1f}%)",
                        ha="center", va="bottom")

            # 4.4 Summary table
            ax4 = fig.add_subplot(gs[2, :])
            ax4.axis("off")
            summary = [
                ["Metric", "Original", "Modified", "Δ"],
                ["Mean",
                f"{original_preds.mean():.4f}",
                f"{modified_preds.mean():.4f}",
                f"{modified_preds.mean() - original_preds.mean():.4f}"],
                ["Std Dev",
                f"{original_preds.std():.4f}",
                f"{modified_preds.std():.4f}",
                f"{modified_preds.std() - original_preds.std():.4f}"],
                ["Min",
                f"{original_preds.min():.4f}",
                f"{modified_preds.min():.4f}",
                f"{modified_preds.min() - original_preds.min():.4f}"],
                ["Max",
                f"{original_preds.max():.4f}",
                f"{modified_preds.max():.4f}",
                f"{modified_preds.max() - original_preds.max():.4f}"]
            ]
            table = ax4.table(cellText=summary[1:],
                            colLabels=summary[0],
                            cellLoc="center", loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 1.5)

        else:
            # classification flip pie + counts
            ax1 = fig.add_subplot(gs[0, :2])
            oc = (original_preds >= 0.5).astype(int)
            mc = (modified_preds  >= 0.5).astype(int)
            same  = np.sum(oc == mc)
            flips = len(oc) - same
            labels = [f"Same\n{same} ({same/len(oc)*100:.1f}%)",
                    f"Flipped\n{flips} ({flips/len(oc)*100:.1f}%)"]
            colors = ["#8da0cb", "#fc8d62"]
            wedges, texts = ax1.pie([same, flips],
                                    labels=labels,
                                    colors=colors,
                                    startangle=90,
                                    textprops={"va":"center"},
                                    wedgeprops={"linewidth":1, "edgecolor":"white"})
            ax1.set_title("Label Persistence vs Flip", pad=20)

            # collapse remaining subplots
            for pos in [(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]:
                fig.add_subplot(gs[pos]).axis("off")

        # finalize
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        visualizations["prediction_impact_summary"] = (
            "data:image/png;base64," + base64.b64encode(buf.read()).decode()
        )
        plt.close(fig)


        return visualizations

    from scipy import stats
    import numpy as np


    import logging
    
    def generate_insights(
        metrics: Dict[str, Any],
        applied_changes: Dict[str, Any],
        df: pd.DataFrame,
        target_column: str,
        original_preds: Optional[np.ndarray] = None,
        modified_preds: Optional[np.ndarray] = None
    ) -> List[str]:
        """Generate business-facing insights from scenario simulation metrics."""
        insights = []

        # --- Auto-detect task type ---
        task_type = detect_task_type(df[target_column])
        num_unique = df[target_column].dropna().nunique()

        if task_type == "classification":
            task_subtype = "binary" if num_unique == 2 else "multiclass"
        else:
            task_subtype = "continuous"

        logging.info(f"Auto-detected task: {task_type.upper()} ({task_subtype})")


        # --- Statistical significance ---
        if "is_significant" in metrics:
            significance = metrics["is_significant"]
            p_value = metrics.get("p_value", "N/A")
            if significance:
                insights.append(f"📊 **Statistically Significant Impact** (p = {p_value:.4f})")
            else:
                insights.append(f"⚠️ **Not Statistically Significant** (p = {p_value:.4f})")

        # --- Confidence interval ---
        ci = metrics.get("confidence_interval", None)
        if ci is None and original_preds is not None and modified_preds is not None:
            ci = get_confidence_interval(original_preds, modified_preds)

        if ci and len(ci) == 2 and ci[0] is not None and ci[1] is not None:
            insights.append(f"📉 **95% Confidence Interval**: Prediction change lies between {ci[0]:.4f} and {ci[1]:.4f}")

        # --- Effect size ---
        effect_size = metrics.get("effect_size", 0)
        if effect_size != 0:
            if abs(effect_size) < 0.2:
                insights.append("**Small Effect**: Minimal practical impact")
            elif abs(effect_size) < 0.5:
                insights.append("**Medium Effect**: Moderate practical impact")
            else:
                insights.append("**Large Effect**: Substantial impact on predictions")

        # --- Classification-specific ---
        if task_type == "classification":
            orig = metrics.get("original_conversion_rate")
            mod = metrics.get("modified_conversion_rate")
            if orig is not None and mod is not None:
                delta = mod - orig
                if abs(delta) > 0.001:
                    symbol = "🔼" if delta > 0 else "🔽"
                    direction = "increase" if delta > 0 else "decrease"
                    insights.append(f"{symbol} **{direction.title()} in Conversion Rate**: {orig:.2%} → {mod:.2%} ({delta:+.2%})")

            # Check class distribution changes
            class_dist = metrics.get("class_distribution_change", {})
            if not class_dist and original_preds is not None and modified_preds is not None:
                class_dist = get_class_distribution_change(original_preds, modified_preds)

            if class_dist:
                insights.append("📊 **Class Distribution Shift:**")
                for label, pct in class_dist.items():
                    symbol = "🔼" if pct > 0 else "🔽"
                    insights.append(f"  - {symbol} Class '{label}': {pct:+.2f}%")

            # Check risk shift summary
            risk_summary = metrics.get("risk_shift_summary", {})
            if not risk_summary and original_preds is not None and modified_preds is not None:
                risk_summary = get_risk_shift_summary(original_preds, modified_preds)

            if risk_summary:
                insights.append("⚠️ **Risk Profile Change:**")
                for group, stats in risk_summary.items():
                    orig = stats.get("original_pct", 0)
                    mod = stats.get("modified_pct", 0)
                    shift = mod - orig
                    symbol = "🔼" if shift > 0 else "🔽"
                    insights.append(f"  - {symbol} {group}: {orig:.1%} → {mod:.1%} ({shift:+.1%})")

        # --- Regression-specific ---
        if task_type == "regression":
            if "original_mean" in metrics and "modified_mean" in metrics:
                delta = metrics["modified_mean"] - metrics["original_mean"]
                symbol = "🔼" if delta > 0 else "🔽"
                direction = "increase" if delta > 0 else "decrease"
                insights.append(f"{symbol} **{direction.title()} in Mean Prediction**: {metrics['original_mean']:.4f} → {metrics['modified_mean']:.4f} ({delta:+.4f})")

            # Percentile changes
            p = metrics.get("percentile_summary", {})
            if not p and original_preds is not None and modified_preds is not None:
                p = get_percentile_summary(original_preds, modified_preds)

            if p:
                insights.append("📐 **Percentile Changes:**")
                insights.append(f"  - 25th percentile: {p['orig_25']:.2f} → {p['mod_25']:.2f}")
                insights.append(f"  - Median: {p['orig_50']:.2f} → {p['mod_50']:.2f}")
                insights.append(f"  - 75th percentile: {p['orig_75']:.2f} → {p['mod_75']:.2f}")

        # --- Direction summary ---
        direction = metrics.get("direction")
        percent_change = metrics.get("percent_change")
        if direction and percent_change is not None:
            symbol = "📈" if direction == "increase" else "📉"
            label = "Strong" if abs(percent_change) > 10 else "Modest"
            insights.append(f"{symbol} **{label} {direction.title()}**: {percent_change:.1f}% change in average prediction")

        # --- Feature insights ---
        for feature, change in applied_changes.items():
            description = change.get("description", "")
            insights.append(f"🔧 **{feature.title()}**: {description}")

        # --- Revenue impact ---
        revenue_impact = metrics.get("estimated_revenue_impact", 0)
        if abs(revenue_impact) > 1000:
            insights.append(f"💰 **Estimated Revenue {'Gain' if revenue_impact > 0 else 'Loss'}**: ${revenue_impact:,.2f}")

        if not insights:
            insights.append("ℹ️ No measurable business impact detected from the applied changes.")

        return insights

    # Main execution starts here
    try:
        # Initialize configuration
        config = AnalysisConfig(**(analysis_config or {}))
        
        # Validate inputs
        validation = validate_inputs()
        if not validation.is_valid:
            return {
                "status": "error",
                "errors": validation.errors,
                "warnings": validation.warnings
            }
        
        # Process changes - prioritize feature_changes from the UI
        if isinstance(feature_changes, str) and feature_changes.strip():
            try:
                changes = json.loads(feature_changes)
            except json.JSONDecodeError as e:
                return {"status": "error", "errors": [f"Invalid JSON in feature_changes: {e}"]}
        elif isinstance(bulk_changes, dict):
            changes = bulk_changes
        elif isinstance(bulk_changes, str):
            try:
                changes = json.loads(bulk_changes)
            except json.JSONDecodeError as e:
                return {"status": "error", "errors": [f"Invalid JSON in bulk_changes: {e}"]}
        else:
            changes = {}
        
        if not changes:
            return {"status": "error", "errors": ["No feature changes specified for what-if analysis"]}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # ───── Load data, model, and training‐column metadata ─────
            df, dataset_name = safe_download_and_load_data()
            model, task_type = load_model(df, target_column)

            raw = download_file_from_supabase(f"{user_id}/training_columns.json")
            training_columns = json.loads(raw.decode("utf-8"))

            # ───── Remove the target column from the expected features ─────
            training_columns = [c for c in training_columns if c != target_column]
            # ───── Apply any bulk feature changes ─────
            modified_df, applied_changes = apply_feature_changes(df, changes)

            # ───── Pick only the features that both exist in the DF and were in training_columns ─────
            feature_cols = [c for c in training_columns if c in df.columns]
            if not feature_cols:
                return {"status":"error", "errors":["No feature columns found for prediction"]}

                # ───── Drop ONLY the target (and any other unwanted cols) ─────
            raw_orig = df.drop(columns=[target_column], errors="ignore")
            raw_mod  = modified_df.drop(columns=[target_column], errors="ignore")

            # ───────── Preprocess ORIGINAL and MODIFIED ─────────
            try:
                X_orig, _, _ = preprocess_data(raw_orig)
                X_mod , _, _ = preprocess_data(raw_mod)
            except Exception as e:
                return {"status":"error", "errors":[f"Preprocessing failed: {e}"]}

            # sanity‐check sizes
            logging.info(f"Post-preprocess shapes: X_orig={X_orig.shape}, X_mod={X_mod.shape}")

            # ───────── Now align to exactly the saved training_columns ─────────
            # (you already removed target from the list earlier)
            missing = set(training_columns) - set(X_orig.columns)
            extra   = set(X_orig.columns)   - set(training_columns)

            for col in missing:
                X_orig[col] = X_mod[col] = 0

            X_orig.drop(columns=list(extra), inplace=True, errors="ignore")
            X_mod .drop(columns=list(extra), inplace=True, errors="ignore")

            X_orig = X_orig.reindex(columns=training_columns, fill_value=0)
            X_mod  = X_mod .reindex(columns=training_columns, fill_value=0)

            # ───────── Run predictions ─────────
            try:
                predictor = model.predict if hasattr(model, "predict") else model[0]
                # (you can keep your NaN/Inf checks here…)

                original_preds = predictor(X_orig)
                modified_preds = predictor(X_mod)
            except Exception as e:
                return {"status":"error", "errors":[f"Prediction failed: {e}"]}

            # Calculate metrics
            metrics = calculate_advanced_metrics(original_preds, modified_preds)
            
            # Create visualizations
            visualizations = create_enhanced_visualizations(
                original_preds,
                modified_preds,
                df,
                modified_df,
                feature_cols,
                applied_changes,
                metrics,
                task_type
            )
            # ───────────── Save metadata to Supabase ─────────────
            try:
                metadata_entry = {
                    "id": str(uuid.uuid4()),
                    "created_at": datetime.utcnow().isoformat(),
                    "type": "mass_scenario_what_if_analysis",
                    "dataset": dataset_name,
                    "parameters": {
                        "target_column": target_column,
                        "applied_changes": applied_changes,
                        "sample_id": sample_id,
                        "dropped_columns": drop_columns.split(",") if drop_columns else [],
                        "configuration": {
                            "revenue_per_conversion": config.revenue_per_conversion,
                            "confidence_level": config.confidence_level,
                            "significance_threshold": config.significance_threshold
                        }
                    },
                    "metrics": metrics,
                    "visualizations": visualizations,            # ← include the full dict here
                    "insights": insights,
                    "validation": {
                        "warnings": validation.warnings,
                        "changes_applied": len(applied_changes),
                        "records_affected": len(df)
                    }
                }

                metadata_entry = ensure_json_serializable(metadata_entry)
                with master_db_cm() as db:
                    _append_limited_metadata(user_id, metadata_entry, db=db, max_entries=5)

            except Exception as meta_error:
                logging.warning(f"[⚠️] What-if metadata save error: {meta_error}")
            # Generate insights
            insights = generate_insights(
                metrics,
                applied_changes,
                df=df,
                target_column=target_column,
                original_preds=original_preds,
                modified_preds=modified_preds
            )


            if not insights:
                insights.append("ℹ️ No measurable impact detected from the applied changes.")
            # Prepare response
            response_data = {
                "status": "success",
                "user_id": user_id,
                "analysis_id": str(uuid.uuid4()),
                "created_at": datetime.utcnow().isoformat(),
                "type": "mass_scenario_what_if_analysis",
                "dataset": {
                    "name": dataset_name,
                    "rows": len(df),
                    "features": len(feature_cols),
                    "target_column": target_column
                },
                "parameters": {
                    "target_column": target_column,
                    "applied_changes": applied_changes,
                    "sample_id": sample_id,
                    "dropped_columns": drop_columns.split(",") if drop_columns else [],
                    "configuration": {
                        "revenue_per_conversion": config.revenue_per_conversion,
                        "confidence_level": config.confidence_level,
                        "significance_threshold": config.significance_threshold
                    }
                },
                "metrics": metrics,
                "visualizations": visualizations,
                "insights": insights,
                "validation": {
                    "warnings": validation.warnings,
                    "changes_applied": len(applied_changes),
                    "records_affected": len(df)
                }
            }


            return sanitize_for_json(response_data)
    except Exception as e:
        logging.error(f"What-if analysis failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error_id": str(uuid.uuid4()),
            "message": f"Analysis failed: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }

def do_risk_analysis(
    user_id: str,
    current_user: dict = None,
    file_path: str = None,
    train_path: str = None,
    test_path: str = None,
    target_column: str = "target",
    drop_columns: str = ""
) -> dict:

    import numpy as np
    import os
    import uuid
    import joblib
    import io
    import base64
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tempfile
    import json
    from pathlib import Path as PathL
    from datetime import datetime

    # ───────────── Validate upload mode ─────────────
    if file_path and (train_path or test_path):
        raise ValueError("Provide either file_path or both train_path+test_path, not both.")
    if (train_path and not test_path) or (test_path and not train_path):
        raise ValueError("Both train_path and test_path must be provided together.")
    if not file_path and not (train_path and test_path):
        raise ValueError("Provide either a full dataset (file_path) or both train_path+test_path.")

    # Handle file downloads from Supabase
    # Note: file_path, train_path, test_path are now Supabase storage paths (e.g., "user123/dataset.csv")
    local_file_path = None
    local_train_path = None
    local_test_path = None
    temp_files = []
    
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download files from Supabase to temporary locations
            if file_path:
                # Download single file from Supabase using the storage path
                file_bytes = download_file_from_supabase(file_path)
                local_file_path = os.path.join(temp_dir, os.path.basename(file_path))
                with open(local_file_path, 'wb') as f:
                    f.write(file_bytes)
                temp_files.append(local_file_path)
            
            if train_path and test_path:
                # Download train and test files from Supabase using the storage paths
                train_bytes = download_file_from_supabase(train_path)
                test_bytes = download_file_from_supabase(test_path)
                
                local_train_path = os.path.join(temp_dir, os.path.basename(train_path))
                local_test_path = os.path.join(temp_dir, os.path.basename(test_path))
                
                with open(local_train_path, 'wb') as f:
                    f.write(train_bytes)
                with open(local_test_path, 'wb') as f:
                    f.write(test_bytes)
                    
                temp_files.extend([local_train_path, local_test_path])
            
            # Create temporary directories for models and outputs
            model_dir = os.path.join(temp_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            
            user_dir = PathL(temp_dir) / "user_outputs"
            user_dir.mkdir(exist_ok=True)

            # ───────────── Load data ─────────────
            if local_file_path:
                df = pd.read_csv(local_file_path)
                df = df.dropna(subset=[target_column])
                dataset_name = os.path.basename(file_path)
            else:
                train_df = pd.read_csv(local_train_path)
                test_df = pd.read_csv(local_test_path)
                train_df = train_df.dropna(subset=[target_column])
                test_df = test_df.dropna(subset=[target_column])
                df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
                dataset_name = f"{os.path.basename(train_path)}+{os.path.basename(test_path)}"

            # Drop columns if provided
            if drop_columns:
                drops = [c.strip() for c in drop_columns.split(",") if c.strip() and c in df.columns]
                if drops:
                    df.drop(columns=drops, inplace=True)

            # ───────────── Load user model from Supabase ─────────────
            model_path = None
            classifier_path = PathL(model_dir) / f"{user_id}_best_classifier.pkl"
            regressor_path = PathL(model_dir) / f"{user_id}_best_regressor.pkl"
            
            # Try to download model from Supabase if it exists
            try:
                if not classifier_path.exists():
                    # Try to download classifier model from Supabase
                    model_bytes = download_file_from_supabase(f"models/{user_id}_best_classifier.pkl")
                    with open(classifier_path, 'wb') as f:
                        f.write(model_bytes)
                    model_path = classifier_path
            except:
                pass
            
            try:
                if not model_path and not regressor_path.exists():
                    # Try to download regressor model from Supabase
                    model_bytes = download_file_from_supabase(f"models/{user_id}_best_regressor.pkl")
                    with open(regressor_path, 'wb') as f:
                        f.write(model_bytes)
                    model_path = regressor_path
            except:
                pass
            
            # Use whichever model exists
            if not model_path:
                if classifier_path.exists():
                    model_path = classifier_path
                elif regressor_path.exists():
                    model_path = regressor_path
            
            if not model_path or not model_path.exists():
                raise ValueError("No trained model found. Run classification first.")

            model = joblib.load(model_path)

            # Prepare features
            features = df.drop(columns=[target_column], errors='ignore') if target_column in df.columns else df.copy()

            # Get prediction probabilities (assuming binary classification)
            if hasattr(model[0], "predict_proba"):
                probs = model[0].predict_proba(features)
                if len(probs.shape) > 1 and probs.shape[1] > 1:
                    df["risk_prob"] = probs[:, 1]
                else:
                    df["risk_prob"] = probs
            else:
                df["risk_prob"] = model[0].predict(features)

            # Generate histogram of risk probabilities
            risk_fig = plt.figure(figsize=(10, 6))
            plt.hist(df["risk_prob"], bins=20, color='skyblue', edgecolor='black')
            plt.xlabel("Risk Score")
            plt.ylabel("Count")
            plt.title("Risk Analysis: Distribution of Risk Scores")
            plt.grid(alpha=0.3)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches="tight")
            buf.seek(0)
            risk_fig_base64 = base64.b64encode(buf.read()).decode()
            plt.close(risk_fig)

            # Identify high-risk groups
            high_risk = df[df["risk_prob"] > 0.7]
            high_risk_pct = len(high_risk) / len(df) * 100

            risk_metrics = {
                "mean_risk": float(np.mean(df["risk_prob"])),
                "median_risk": float(np.median(df["risk_prob"])),
                "min_risk": float(np.min(df["risk_prob"])),
                "max_risk": float(np.max(df["risk_prob"])),
                "std_risk": float(np.std(df["risk_prob"])),
                "high_risk_count": int(len(high_risk)),
                "high_risk_pct": round(high_risk_pct, 2)
            }

            # ───────────── Save metadata to Supabase ─────────────
            try:
                # Create entry for metadata
                entry = {
                    "id": str(uuid.uuid4()),
                    "created_at": datetime.utcnow().isoformat(),
                    "type": "risk_analysis",
                    "dataset": dataset_name,
                    "parameters": {
                        "target_column": target_column,
                        "drop_columns": drop_columns
                    },
                    "metrics": risk_metrics,
                    "thumbnailData": f"data:image/png;base64,{risk_fig_base64}" if risk_fig_base64 else "",
                    "imageData": f"data:image/png;base64,{risk_fig_base64}" if risk_fig_base64 else "",
                    "visualizations": {
                        "risk_plot": f"data:image/png;base64,{risk_fig_base64}" if risk_fig_base64 else ""
                    }
                }

                try:
                    entry = ensure_json_serializable(entry)
                    with master_db_cm() as db:
                        _append_limited_metadata(user_id, entry, db=db, max_entries=5)
                except Exception as meta_error:
                    print(f"[⚠️] Metadata save error: {meta_error}")

            except Exception as meta_error:
                print(f"[⚠️] Metadata save error: {meta_error}")

            # ───────────── Build response ─────────────
            response_data = {
                "status": "success",
                "user_id": user_id,
                "id": str(uuid.uuid4()),
                "created_at": datetime.utcnow().isoformat(),
                "type": "risk_analysis",
                "dataset": dataset_name,
                "parameters": {
                    "target_column": target_column,
                    "drop_columns": drop_columns
                },
                "message": "Risk Analysis Generated",
                "metrics": risk_metrics,
                "high_risk_percentage": high_risk_pct,
                "mean_risk_score": float(df["risk_prob"].mean()),
                "visualizations": {},
                "insights": {}
            }

            # Dynamically attach visualizations
            if risk_fig_base64:
                response_data["visualizations"]["risk_plot"] = f"data:image/png;base64,{risk_fig_base64}"


            # Return full response
            return response_data
            
    except Exception as e:
        print(f"[⚠️] Error in do_risk_analysis: {e}")
        raise e
def do_decision_paths(
    user_id: str,
    current_user: dict = None,
    file_path: str = None,
    train_path: str = None,
    test_path: str = None,
    target_column: str = "target",
    drop_columns: str = ""
) -> dict:

    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from sklearn.tree import export_text
    import numpy as np
    import os
    import uuid
    import io
    import base64
    import matplotlib.pyplot as plt
    import tempfile
    from pathlib import Path as PathL
    from datetime import datetime

    # ───────────── Validate upload mode ─────────────
    if file_path and (train_path or test_path):
        raise ValueError("Provide either file_path or both train_path+test_path, not both.")
    if (train_path and not test_path) or (test_path and not train_path):
        raise ValueError("Both train_path and test_path must be provided together.")
    if not file_path and not (train_path and test_path):
        raise ValueError("Provide either a full dataset (file_path) or both train_path+test_path.")

    # Handle file downloads from Supabase
    # Note: file_path, train_path, test_path are now Supabase storage paths (e.g., "user123/dataset.csv")
    local_file_path = None
    local_train_path = None
    local_test_path = None
    temp_files = []
    
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download files from Supabase to temporary locations
            if file_path:
                # Download single file from Supabase using the storage path
                file_bytes = download_file_from_supabase(file_path)
                local_file_path = os.path.join(temp_dir, os.path.basename(file_path))
                with open(local_file_path, 'wb') as f:
                    f.write(file_bytes)
                temp_files.append(local_file_path)
            
            if train_path and test_path:
                # Download train and test files from Supabase using the storage paths
                train_bytes = download_file_from_supabase(train_path)
                test_bytes = download_file_from_supabase(test_path)
                
                local_train_path = os.path.join(temp_dir, os.path.basename(train_path))
                local_test_path = os.path.join(temp_dir, os.path.basename(test_path))
                
                with open(local_train_path, 'wb') as f:
                    f.write(train_bytes)
                with open(local_test_path, 'wb') as f:
                    f.write(test_bytes)
                    
                temp_files.extend([local_train_path, local_test_path])
            
            # Create temporary directories for models and outputs
            user_dir = PathL(temp_dir) / "user_outputs"
            user_dir.mkdir(exist_ok=True)

            # ───────────── Load and preprocess data ─────────────
            if local_file_path:
                df = pd.read_csv(local_file_path)
                df = df.dropna(subset=[target_column])
                if drop_columns:
                    drops = [c.strip() for c in drop_columns.split(",") if c.strip() and c in df.columns]
                    df.drop(columns=drops, inplace=True)

                if target_column not in df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in dataset.")

                y = df[target_column]

                X = df.drop(columns=['ID', target_column], errors='ignore')

                X, _, _ = preprocess_data(X)
                X = X.select_dtypes(include=["int", "float", "bool"])
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                X_train = X_train.reset_index(drop=True)
                X_test  = X_test.reset_index(drop=True)
                y_train = y_train.reset_index(drop=True)
                y_test  = y_test.reset_index(drop=True)

                dataset_name = os.path.basename(file_path)

            else:
                train_df = pd.read_csv(local_train_path)
                test_df = pd.read_csv(local_test_path)
                train_df = train_df.dropna(subset=[target_column]).reset_index(drop=True)
                test_df  = test_df.dropna(subset=[target_column]).reset_index(drop=True)
                if drop_columns:
                    drops = [c.strip() for c in drop_columns.split(",") if c.strip()]
                    train_df.drop(columns=[c for c in drops if c in train_df.columns], inplace=True)
                    test_df.drop(columns=[c for c in drops if c in test_df.columns], inplace=True)

                if target_column not in train_df.columns or target_column not in test_df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in training or test dataset.")

                y_train = train_df[target_column]
                y_test = test_df[target_column]

                X_train = train_df.drop(columns=[target_column, 'ID'], errors='ignore')
                X_test = test_df.drop(columns=[target_column, 'ID'], errors='ignore')

                X_train, _, _ = preprocess_data(X_train)
                X_train = X_train.reset_index(drop=True)

                X_test, _, _ = preprocess_data(X_test)
                X_test  = X_test.reset_index(drop=True)

                common_columns = list(set(X_train.columns) & set(X_test.columns))
                X_train = X_train[common_columns]
                X_test = X_test[common_columns]

                dataset_name = f"{os.path.basename(train_path)}+{os.path.basename(test_path)}"

            # ───────────── Train decision tree ─────────────
            clf = DecisionTreeClassifier(max_depth=4, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # ───────────── Calculate metrics ─────────────
            metrics = [
                {"metric": "accuracy", "value": round(float(accuracy_score(y_test, y_pred)), 4)},
                {"metric": "precision", "value": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4)},
                {"metric": "recall", "value": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4)},
                {"metric": "f1_score", "value": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)},
                {"metric": "confusion_matrix", "value": confusion_matrix(y_test, y_pred).tolist()}
            ]
                
            metrics_table = [
                ["accuracy", round(float(accuracy_score(y_test, y_pred)), 4)],
                ["precision", round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4)],
                ["recall", round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4)],
                ["f1_score", round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)],
                ["confusion_matrix", confusion_matrix(y_test, y_pred).tolist()]
            ]
            metrics_headers = ["Metric", "Value"]

            # ───────────── Calculate summary statistics ─────────────
            summary_stats = [
                {
                    "Feature": col,
                    "Mean": float(np.mean(X_train[col])),
                    "Std": float(np.std(X_train[col])),
                    "Min": float(np.min(X_train[col])),
                    "Max": float(np.max(X_train[col])),
                }
                for col in X_train.select_dtypes(include=["number"]).columns
            ]
            summary_stats_table = [
                [col, float(np.mean(X_train[col])), float(np.std(X_train[col])), float(np.min(X_train[col])), float(np.max(X_train[col]))]
                for col in X_train.select_dtypes(include=["number"]).columns
            ]
            summary_stats_headers = ["Feature", "Mean", "Std", "Min", "Max"]

            # ───────────── Generate decision tree visualization ─────────────
            decision_fig = plt.figure(figsize=(20, 10))
            tree.plot_tree(clf, feature_names=X_train.columns, class_names=[str(c) for c in clf.classes_], filled=True)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches="tight")
            buf.seek(0)
            decision_fig_base64 = base64.b64encode(buf.read()).decode()
            plt.close(decision_fig)

            # ───────────── Generate human-readable decision rules ─────────────
            decision_rules = export_text(clf, feature_names=list(X_train.columns), max_depth=4)
            rules_list = decision_rules.strip().split("\n")

            # ───────────── Save metadata to Supabase ─────────────
            try:
                # Create entry for metadata
                entry = {
                    "id": str(uuid.uuid4()),
                    "created_at": datetime.utcnow().isoformat(),
                    "type": "decision_paths",
                    "dataset": dataset_name,
                    "parameters": {
                        "target_column": target_column,
                        "drop_columns": drop_columns
                    },
                    "metrics": metrics_table,
                    "summary_stats": summary_stats_table,
                    "metrics_headers": metrics_headers,
                    "summary_stats_headers": summary_stats_headers,
                    "decision_rules": rules_list,
                    "thumbnailData": f"data:image/png;base64,{decision_fig_base64}",
                    "imageData": f"data:image/png;base64,{decision_fig_base64}",
                    "visualizations": {
                        "decision_paths": f"data:image/png;base64,{decision_fig_base64}"
                    },
                    "summary_stats": summary_stats
                }

                try:
                    entry = ensure_json_serializable(entry)
                    with master_db_cm() as db:
                        _append_limited_metadata(user_id, entry, db=db, max_entries=5)
                except Exception as meta_error:
                    print(f"[⚠️] Metadata save error: {meta_error}")

            except Exception as meta_error:
                print(f"[⚠️] Metadata save error: {meta_error}")

            # ───────────── Build response ─────────────
            response_data = {
                "status": "success",
                "user_id": user_id,
                "created_at": datetime.utcnow().isoformat(),
                "type": "decision_paths",
                "dataset": dataset_name,
                "parameters": {
                    "target_column": target_column,
                    "drop_columns": drop_columns
                },
                "message": "Decision Paths Generated",
                "visualizations": {
                    "decision_paths": f"data:image/png;base64,{decision_fig_base64}"
                },
                "metrics": metrics_table,
                "summary_stats": summary_stats_table,
                "metrics_headers": metrics_headers,
                "summary_stats_headers": summary_stats_headers,
                "decision_rules": rules_list
            }

            # Return full response
            return response_data
            
    except Exception as e:
        print(f"[⚠️] Error in do_decision_paths: {e}")
        raise e
# -------- Finite-safety + robust TS utilities --------
import numpy as np, pandas as pd
from typing import Optional, Dict, Any, Tuple

def _winsorize_quantiles(s: pd.Series, lo=0.001, hi=0.999) -> pd.Series:
    """Clamp extremes by quantiles to prevent overflow/underflow in fitting."""
    if s.empty: return s
    qlo, qhi = s.quantile([lo, hi])
    return s.clip(qlo, qhi)

def _mad_deoutlier(s: pd.Series, k=12.0) -> pd.Series:
    """Remove absurd spikes using median absolute deviation."""
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0:
        return s
    z = 0.6745 * (s - med) / mad
    return s.where(z.abs() <= k, np.nan)

def _stabilize_variance(s: pd.Series) -> pd.Series:
    """
    Stabilize heavy-tailed series safely:
    - Try log1p if series is nonnegative
    - Else Yeo-Johnson-like transform (sign-preserving)
    """
    s = s.astype("float64")
    if (s >= 0).all():
        return np.log1p(s)
    # sign-preserving softplus approximation
    pos = s.clip(lower=0)
    neg = (-s.clip(upper=0))
    pos_t = np.log1p(pos)
    neg_t = -np.log1p(neg)
    return pos_t + neg_t

def _ensure_regular_index(y: pd.Series, freq_hint: Optional[str] = None) -> Tuple[pd.Series, str]:
    """Ensure DatetimeIndex with a frequency; interpolate gaps."""
    if not isinstance(y.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        # Use synthetic daily index
        idx = pd.date_range("2000-01-01", periods=len(y), freq="D")
        y = pd.Series(y.values, index=idx, name=y.name)

    inferred = pd.infer_freq(y.index)
    freq = freq_hint or inferred or "D"
    y = y.asfreq(freq)
    if y.isna().any():
        y = y.interpolate(limit_direction="both")
    return y, freq

def _prepare_ts_strict(df: pd.DataFrame, target_col: str, time_col: Optional[str]) -> Dict[str, Any]:
    """Coerce datetime index, build finite/regularized target series ready for modeling."""
    w = df.copy()

    # 1) Build a datetime index
# replace this block inside _prepare_ts_strict(...)
    if time_col and time_col in w.columns:
        w[time_col] = parse_datetime_smart(w[time_col])
        w = w.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)
    else:
        detected = None
        for col in w.columns:
            if pd.api.types.is_datetime64_any_dtype(w[col]):
                detected = col; break
            if w[col].dtype in (object, "int64", "int32"):
                parsed = parse_datetime_smart(w[col])
                if parsed.notna().sum() >= max(8, int(0.3*len(parsed))):
                    w[col] = parsed; detected = col; break
        if detected:
            w = w.dropna(subset=[detected]).sort_values(detected).set_index(detected)
        else:
            w = w.reset_index(drop=True); w.index.name = "synthetic_time"


    # 2) Extract and coerce target to float; drop non-finite
    if target_col not in w.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(w.columns)}")

    y = pd.to_numeric(w[target_col], errors="coerce").astype("float64")
    y = y.replace([np.inf, -np.inf], np.nan).dropna()

    # 3) Regularize index & interpolate to remove gaps
    y, freq = _ensure_regular_index(y, freq_hint=None)

    # 4) Remove absurd spikes (MAD), winsorize remaining extremes
    y = _mad_deoutlier(y, k=12.0)
    y = y.interpolate(limit_direction="both")
    y = _winsorize_quantiles(y, lo=0.001, hi=0.999)

    # 5) Variance stabilize for heavy tails
    y = _stabilize_variance(y)

    # 6) Final finite check
    y = pd.to_numeric(y, errors="coerce").astype("float64")
    y = y.replace([np.inf, -np.inf], np.nan).dropna()

    return {"y": y, "freq": freq, "nobs": int(y.notna().sum())}

def _is_constant(y: pd.Series) -> bool:
    return y.nunique(dropna=True) <= 1

def _naive_forecast(y: pd.Series, steps: int, freq: str) -> pd.DataFrame:
    """Drift/last-value fallback that is guaranteed finite."""
    last = float(y.iloc[-1])
    idx = pd.date_range(y.index[-1], periods=steps+1, freq=freq)[1:]
    out = pd.DataFrame({
        "timestamp": idx,
        "forecast": np.full(steps, last, dtype="float64"),
        "lower": np.full(steps, last, dtype="float64"),
        "upper": np.full(steps, last, dtype="float64"),
    })
    return out

def _finite_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure frame has only finite floats where numeric."""
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return df.dropna()
# Add this helper near your TS utilities
def parse_datetime_smart(col: pd.Series) -> pd.Series:
    """Fast, consistent datetime parsing without dateutil fallback spam."""
    s = col.astype("string")

    # 1) Try common exact formats (vectorized, fast)
    fmts = [
        "%Y-%m-%d",                # 2025-08-09
        "%Y/%m/%d",
        "%m/%d/%Y",                # 08/09/2025
        "%m/%d/%y",
        "%Y-%m-%d %H:%M:%S",       # 2025-08-09 14:30:00
        "%Y-%m-%dT%H:%M:%S",       # ISO without zone
        "%d-%b-%Y",                # 09-Aug-2025
    ]
    best = None; best_ok = -1
    for fmt in fmts:
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        ok = parsed.notna().sum()
        if ok > best_ok:
            best_ok, best = ok, parsed
        if ok == len(s):  # perfect hit
            return parsed

    # 2) Epoch (seconds) if numeric-ish and not many duds
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().sum() >= max(8, int(0.5 * len(s))):
        epoch_sec = pd.to_datetime(num, unit="s", errors="coerce")
        if epoch_sec.notna().sum() >= best_ok:
            best = epoch_sec

    # 3) Fallback: ISO parser (still vectorized) – no warning spam
    iso = pd.to_datetime(s, errors="coerce")
    if iso.notna().sum() >= best_ok:
        best = iso

    return best
def season_length_from_freq(freq: str) -> int:
    # crude but practical
    if not freq: return 0
    if freq.startswith("H"): return 24
    if freq.startswith("D"): return 7
    if freq.startswith("W"): return 52
    if freq.startswith("M"): return 12
    if freq.startswith("Q"): return 4
    if freq.startswith("A") or freq.startswith("Y"): return 1
    return 0

def do_forecast(
    user_id: str,
    current_user: dict,
    file_path: str = None,
    train_path: str = None,
    test_path: str = None,
    target_column: str = "value",
    drop_columns: str = "",
    periods: int = 24,
    scenarios: List[str] = None,
    time_column: str = None,
    enable_backtesting: bool = False
) -> Dict[str, Any]:
    import os, tempfile, traceback
    import pandas as pd
    import numpy as np

    with tempfile.TemporaryDirectory() as temp_dir:
        # ---------- Load ----------
        if file_path:
            fb = download_file_from_supabase(file_path)
            local = os.path.join(temp_dir, os.path.basename(file_path))
            with open(local, "wb") as f: f.write(fb)
            df = pd.read_csv(local, sep=None, engine="python")
        elif train_path and test_path:
            tb = download_file_from_supabase(train_path)
            vb = download_file_from_supabase(test_path)
            train_file = os.path.join(temp_dir, "train.csv")
            test_file  = os.path.join(temp_dir, "test.csv")
            with open(train_file, "wb") as f: f.write(tb)
            with open(test_file, "wb")  as f: f.write(vb)
            df_train = pd.read_csv(train_file, sep=None, engine="python")
            df_test  = pd.read_csv(test_file,  sep=None, engine="python")
            df = pd.concat([df_train, df_test], ignore_index=True)
        else:
            raise ValueError("No valid input file provided (file_path or train/test).")

        # ---------- Drop columns ----------
        if drop_columns:
            drops = [c.strip() for c in drop_columns.split(",") if c.strip()]
            df.drop(columns=[c for c in drops if c in df.columns], inplace=True, errors="ignore")

       # ... after loading df and dropping columns ...

        # ---------- STRICT TS PREP ----------
        prep = _prepare_ts_strict(df, target_column, time_column)
        y, freq, nobs = prep["y"], prep["freq"], prep["nobs"]

        # Early exits
        if nobs < 3:
            fc = _naive_forecast(y, periods, freq)
            return {
                "series": y.tolist(),
                "parameters": {"periods": periods, "target_column": target_column, "time_column": time_column, "freq": freq, "nobs": nobs, "fallback": "naive"},
                "scenarios": {"naive": {"forecast": _finite_df(fc).to_dict(orient="records")}},
                "metadata": {"data_points": int(nobs), "user_id": user_id}
            }

        if _is_constant(y):
            fc = _naive_forecast(y, periods, freq)
            return {
                "series": y.tolist(),
                "parameters": {"periods": periods, "target_column": target_column, "time_column": time_column, "freq": freq, "nobs": nobs, "fallback": "constant_naive"},
                "scenarios": {"naive": {"forecast": _finite_df(fc).to_dict(orient="records")}},
                "metadata": {"data_points": int(nobs), "user_id": user_id}
            }

        # ---------- INSERT HERE: seasonal length + guarded model selection ----------
        season_L = season_length_from_freq(freq)
        enough_for_seasonal = (season_L > 1) and (nobs >= 3 * season_L)  # need ~3 seasons
        stable_flags = {"enforce_stationarity": True, "enforce_invertibility": True}

        if nobs < 12:
            # short-series guardrails
            arima_model  = ARIMAModel(auto_arima=False, order=(0,1,0), model_kwargs=stable_flags)
            sarima_model = SARIMAModel(order=(0,1,0), seasonal_order=(0,0,0,0), model_kwargs=stable_flags)
        else:
            seas = (0,0,0,0)
            if enough_for_seasonal:
                seas = (1,0,1, season_L)  # simple seasonal ARMA structure
            arima_model  = ARIMAModel(auto_arima=True, model_kwargs=stable_flags)
            sarima_model = SARIMAModel(order=(1,1,1), seasonal_order=seas, model_kwargs=stable_flags)

        # ---------- Scenario setup ----------
        scenario_manager = ScenarioManager()
        scenarios = scenarios or ["comprehensive"]

        if "traditional" in scenarios:
            scenario_manager.add_scenario("traditional", "Statistical models", [arima_model, sarima_model])

        if "machine_learning" in scenarios:
            scenario_manager.add_scenario("machine_learning", "ML models", [RandomForestModel()])

        if "comprehensive" in scenarios:
            scenario_manager.add_scenario("comprehensive", "All models", [arima_model, sarima_model, RandomForestModel()])

        # ---------- Run scenarios ----------
        scenario_results = {}
        for name in scenarios:
            if name not in scenario_manager.scenarios:
                continue
            res = scenario_manager.run_scenario(name, y, periods, freq=freq, enable_backtesting=enable_backtesting)
            try:
                if isinstance(res, dict) and "forecast" in res:
                    fdf = pd.DataFrame(res["forecast"])
                    fdf = _finite_df(fdf)
                    res["forecast"] = fdf.to_dict(orient="records")
            except Exception:
                pass
            scenario_results[name] = res


        return {
            "series": y.tolist(),
            "parameters": {"periods": periods, "target_column": target_column, "time_column": time_column, "freq": freq, "nobs": nobs, "short_series_guard": short_series},
            "scenarios": scenario_results,
            "metadata": {"data_points": int(nobs), "user_id": user_id}
        }


def run_forecast_subprocess(user_id, file_path, forecast_result, output_dir):
    """
    Run forecast visualization subprocess.
    - Sanitizes forecast_result to native Python types.
    - Writes a request.json file.
    - Invokes forecast_runner.py in a subprocess.
    Returns True on success, False on any failure.
    """
    import subprocess
    import sys
    import json
    import traceback
    import numpy as np
    from pathlib import Path as PathL

    # Ensure output directory exists
    output_dir = PathL(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Sanitize forecast_result ───
    clean_result = {}
    for key, val in forecast_result.items():
        # NumPy array → Python list
        if isinstance(val, np.ndarray):
            clean_result[key] = val.tolist()
        # List possibly containing NumPy scalars
        elif isinstance(val, list) and any(isinstance(x, np.generic) for x in val):
            lst = []
            for item in val:
                lst.append(item.item() if isinstance(item, np.generic) else item)
            clean_result[key] = lst
        # NumPy scalar → native Python
        elif isinstance(val, np.generic):
            clean_result[key] = val.item()
        else:
            clean_result[key] = val

    # Build request payload
    request_data = {
        "user_id": str(user_id),
        "file_path": str(file_path),
        "forecast_result": clean_result,
        "output_dir": str(output_dir)
    }

    try:
        # Write request.json
        request_path = output_dir / "request.json"
        with request_path.open("w", encoding="utf-8") as f:
            json.dump(request_data, f, indent=2)

        # Locate the runner script
        runner_script = PathL(__file__).parent / "forecast_runner.py"
        if not runner_script.exists():
            print(f"[⚠️] Forecast runner not found at: {runner_script}", file=sys.stderr)
            raise FileNotFoundError(f"Forecast runner script not found: {runner_script}")

        # Execute subprocess
        result = subprocess.run(
            [sys.executable, str(runner_script.resolve()), str(request_path.resolve())],
            check=True,
            capture_output=True,
            text=True,
            timeout=300  # seconds
        )

        # Log stdout/stderr for debugging
        if result.stdout:
            print(f"[FORECAST STDOUT]:\n{result.stdout}")
        if result.stderr:
            print(f"[FORECAST STDERR]:\n{result.stderr}", file=sys.stderr)

        print(f"[✅] Forecast subprocess ran successfully for user {user_id}")
        return True

    except subprocess.TimeoutExpired:
        print(f"[⚠️] Forecast subprocess timed out after 5 minutes for user {user_id}", file=sys.stderr)
        return False

    except subprocess.CalledProcessError as e:
        print(f"[⚠️] Forecast subprocess failed (exit {e.returncode}):", file=sys.stderr)
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}", file=sys.stderr)
        if e.stderr:
            print(f"STDERR:\n{e.stderr}", file=sys.stderr)
        return False

    except Exception as e:
        print(f"[⚠️] Forecast subprocess error: {e}", file=sys.stderr)
        traceback.print_exc()
        return False

from scipy.stats import chi2_contingency
def do_ab_test(
    user_id: str,
    file_path: str,
    target_column: str = "converted",
    variant_column: str = "variant",
    drop_columns: str = "",
    current_user: dict = None
) -> dict:

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # ─── Download & load dataset ───
            file_bytes = download_file_from_supabase(file_path)
            local_path = os.path.join(temp_dir, os.path.basename(file_path))
            with open(local_path, 'wb') as f:
                f.write(file_bytes)
            df = pd.read_csv(local_path)

            # ─── Clean up unwanted cols ───
            if drop_columns:
                drops = [c.strip() for c in drop_columns.split(",") if c.strip()]
                df.drop(columns=[c for c in drops if c in df.columns], inplace=True)

            # ─── Validate & cast ───
            if target_column not in df.columns or variant_column not in df.columns:
                raise ValueError(f"Missing '{target_column}' or '{variant_column}' column.")
            df = df.dropna(subset=[target_column, variant_column])
            df[target_column] = df[target_column].astype(int)

            # ─── Summarize & χ² test ───
            summary = df.groupby(variant_column)[target_column].agg(["count", "sum"])
            summary.columns = ["total_users", "conversions"]
            summary["conversion_rate"] = summary["conversions"] / summary["total_users"]

            contingency = pd.crosstab(df[variant_column], df[target_column])
            chi2, p_value, _, _ = chi2_contingency(contingency)

            # ─── Identify winner & lift ───
            best = summary["conversion_rate"].idxmax()
            lift = summary.loc[best, "conversion_rate"] - summary["conversion_rate"].min()

            # ─── Chart 1: Conversion-Rate Bar ───
            fig1 = plt.figure(figsize=(6, 4))
            summary["conversion_rate"].plot(kind="bar", rot=0)
            plt.ylabel("Conversion Rate")
            plt.title("A/B Conversion Rates")
            conversion_rate_chart_base64 = plot_to_base64(fig1)
            plt.close(fig1)

            # ─── Chart 2: Contingency Heatmap ───
            fig2 = plt.figure(figsize=(6, 4))
            sns.heatmap(contingency, annot=True, fmt="d", cbar=False)
            plt.ylabel(variant_column)
            plt.xlabel(target_column)
            plt.title("Contingency Table")
            contingency_heatmap_base64 = plot_to_base64(fig2)
            plt.close(fig2)

            # ─── Build metadata including images ───
            entry = {
                "id": str(uuid.uuid4()),
                "created_at": datetime.utcnow().isoformat(),
                "type": "ab_test",
                "dataset": os.path.basename(file_path),
                "target_column": target_column,
                "variant_column": variant_column,
                "summary_stats": summary.reset_index().to_dict("records"),
                "p_value": round(p_value, 6),
                "winner": best,
                "lift": round(lift * 100, 2),
                "conversion_rate": {
                    var: round(rate * 100, 2)
                    for var, rate in summary["conversion_rate"].items()
                },
                "insights": f"Variant '{best}' won with a {round(lift*100,2)}% lift.",
                # Visualizations in the desired format
                "visualizations": {
                    "conversion_rate_chart": f"data:image/png;base64,{conversion_rate_chart_base64}" if conversion_rate_chart_base64 else "",
                    "contingency_heatmap": f"data:image/png;base64,{contingency_heatmap_base64}" if contingency_heatmap_base64 else "",
                }
            }

            # Make sure the entry is serializable to JSON
            entry = ensure_json_serializable(entry)

            # Store in database or append metadata
            with master_db_cm() as db:
                _append_limited_metadata(user_id, entry, db=db, max_entries=5)

            # Return response with visualizations in 'visualizations' key
            response_data = {
                "status": "success",
                "id": entry["id"],
                "user_id": user_id,
                **entry,  # Include the visualizations and metadata in the response
            }

            return response_data

    except Exception as e:
        print(f"[⚠️] Error in run_ab_test: {e}")
        raise



@celery_app.task
def run_classification(
    user_id: str,
    current_user: dict = None,
    file_path: str = None,
    train_path: str = None,
    test_path: str = None,
    target_column: str = "target",
    drop_columns: str = ""
):
    # Optional: logging or throttling
    time.sleep(1)
    # Delegate to your full implementation
    return do_classification(
        user_id,
        file_path=file_path,
        train_path=train_path,
        test_path=test_path,
        target_column=target_column,
        drop_columns=drop_columns
    )
@celery_app.task
def run_classification_predict_score(
    user_id: str,
    file_path: str,
    drop_columns: str = "",
    current_user: dict = None
):
    """
    Celery task for running classification predictions and returning probability scores and risk tiers.
    
    Args:
        user_id: User identifier to locate the saved model
        file_path: Path to CSV file containing new data for predictions
        drop_columns: Comma-separated column names to drop before prediction
        current_user: Optional dictionary with user info
    
    Returns:
        Dictionary containing predictions, probability scores, risk tiers, and metadata
    """
    import time
    time.sleep(1)  # optional throttle or delay

    # Call your probability-score version of the classification function
    return do_classification_predict_score(
        user_id=user_id,
        current_user=current_user,
        file_path=file_path,
        drop_columns=drop_columns
    )
@celery_app.task
def run_classification_predict(
    user_id: str,
    file_path: str,
    drop_columns: str = "",
    current_user: dict = None,
    output_predictions: bool = True
):
    """
    Celery task for running classification predictions on new data.
    
    Args:
        user_id: User identifier to locate the saved model
        prediction_file_path: Path to CSV file containing new data for predictions
        drop_columns: Comma-separated column names to drop before prediction
        output_predictions: Whether to save predictions to a CSV file
    
    Returns:
        Dictionary containing predictions and metadata
    """
    import time
    
    # Optional throttle or delay if needed
    time.sleep(1)
    
    # Call your actual implementation
    return do_classification_predict(
        user_id=user_id,
        current_user=current_user,
        file_path=file_path,
        drop_columns=drop_columns,
        output_predictions=output_predictions
    )
@celery_app.task
def run_clustering(
    user_id: str,
    file_path: str,
    current_user: dict,  # ✅ explicitly a dict
    target_column: str = None,
    time_column: str = None,
    drop_columns: str = "",
    **kwargs
):
    time.sleep(1)
    return do_clustering(
        user_id=user_id,
        current_user=current_user,  # ✅ pass as-is
        file_path=file_path,
        target_column=target_column,
        drop_columns=drop_columns
    )

# Celery tasks
@celery_app.task
def run_segment_analysis(
    user_id: str,
    current_user: dict,  # Changed from str to dict
    file_path: str = None,
    train_path: str = None,
    test_path: str = None,
    target_column: str = None,
    drop_columns: str = ""
):
    """Celery task for running segmentation analysis"""
    time.sleep(1)  # Optional throttle
    
    return do_segment_analysis(
        user_id=user_id,
        current_user=current_user,
        file_path=file_path,
        train_path=train_path,
        test_path=test_path,
        target_column=target_column,
        drop_columns=drop_columns
    )
@celery_app.task
def run_label_clusters(
    user_id: str,
    current_user: dict,
    file_path: str = None,
    train_path: str = None,
    test_path: str = None,
    feature_columns: str = ""
):
    # Optional throttle or delay if needed
    time.sleep(1)

    # Call your actual implementation
    return do_label_clusters(
        user_id=user_id,
        current_user=current_user,
        file_path=file_path,
        train_path=train_path,
        test_path=test_path,
        feature_columns=feature_columns
    )
@celery_app.task
def run_regression(
    user_id: str,
    current_user: dict,
    file_path: str = None,
    train_path: str = None,
    test_path: str = None,
    target_column: str = "target",
    drop_columns: str = ""
):
    # Optional throttle or delay if needed
    time.sleep(1)

    # Call your actual implementation
    return do_regression(
        user_id=user_id,
        current_user = current_user,
        file_path=file_path,
        train_path=train_path,
        test_path=test_path,
        target_column=target_column,
        drop_columns=drop_columns
    )
@celery_app.task
def run_regression_predict(
    user_id: str,
    current_user: dict = None,
    file_path: str = None,
    train_path: str = None,
    test_path: str = None,
    drop_columns: str = ""
):
    import time
    time.sleep(1)  # Optional: artificial delay, like your classification pattern
    
    # Call the actual processing function
    return do_regression_predict(
        user_id=user_id,
        current_user=current_user,
        file_path=file_path,
        train_path=train_path,
        test_path=test_path,
        drop_columns=drop_columns
    )

@celery_app.task
def run_visualization(
    user_id: str,
    current_user: dict = None,
    file_path: str = None,
    train_path: str = None,
    test_path: str = None,
    target_column: str = None,
    feature_column: str = None
):
    import time
    time.sleep(1)
    return do_visualization(
        user_id=user_id,
        current_user=current_user,
        file_path=file_path,
        train_path=train_path,
        test_path=test_path,
        target_column=target_column,
        feature_column=feature_column
    )
import numpy as np

import json

def make_json_safe(obj):
    """
    Recursively convert numpy types and other non-JSON-serializable types to JSON-safe types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: make_json_safe(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_safe(item) for item in obj)
    elif hasattr(obj, 'item'):  # For numpy scalars
        return obj.item()
    else:
        return obj

@celery_app.task
def run_counterfactual(
    user_id: str,
    current_user: dict = None,
    file_path: str = None,
    train_path: str = None,
    test_path: str = None,
    target_column: str = "target",
    drop_columns: str = "",
    sample_id: int = None,
    sample_strategy: str = "random",
    num_samples: int = 1,
    desired_outcome: Union[int, float, str] = None,
    editable_features: List[str] = None,
    max_changes: int = 3,
    proximity_metric: str = "euclidean"
):
    import time
    time.sleep(1)  # Simulate some delay if needed

    # Call your existing function
    result = do_counterfactual(
        user_id=user_id,
        current_user=current_user,
        file_path=file_path,
        train_path=train_path,
        test_path=test_path,
        target_column=target_column,
        drop_columns=drop_columns,
        sample_id=sample_id,
        sample_strategy=sample_strategy,
        num_samples=num_samples,
        desired_outcome=desired_outcome,
        editable_features=editable_features,
        max_changes=max_changes,
        proximity_metric=proximity_metric
    )
    
    # CRITICAL: Make the result JSON-safe before returning
    return make_json_safe(result)

@celery_app.task
def run_survival_analysis(
    user_id: str,
    current_user: dict = None,
    file_path: str = None,
    train_path: str = None,
    test_path: str = None,
    time_col: str = None,
    event_col: str = None,
    drop_cols: str = ""
):
    import time
    time.sleep(1)  # Optional throttle

    return do_survival(
        user_id=user_id,
        current_user=current_user,
        file_path=file_path,
        train_path=train_path,
        test_path=test_path,
        time_col=time_col,
        event_col=event_col,
        drop_cols=drop_cols
    )
@celery_app.task
def run_what_if(
    user_id: str,
    current_user: dict = None,
    file_path: str = None,
    train_path: str = None,
    test_path: str = None,
    sample_id: int = None,
    target_column: str = None,
    feature_changes: str = "",
    drop_columns: str = ""
):
    import time
    time.sleep(1)  # Optional throttle

    return do_what_if(
        user_id=user_id,
        current_user=current_user,
        file_path=file_path,
        train_path=train_path,
        test_path=test_path,
        sample_id=sample_id,
        target_column=target_column,
        feature_changes=feature_changes,
        drop_columns=drop_columns
    )

@celery_app.task
def run_risk_analysis(
    user_id: str,
    current_user: dict = None,
    file_path: str = None,
    train_path: str = None,
    test_path: str = None,
    target_column: str = None,
    drop_columns: str = ""
):
    import time
    time.sleep(1)  # optional throttle

    return do_risk_analysis(
        user_id=user_id,
        current_user=current_user,
        file_path=file_path,
        train_path=train_path,
        test_path=test_path,
        target_column=target_column,
        drop_columns=drop_columns
    )
@celery_app.task
def run_decision_paths(
    user_id: str,
    current_user: dict = None,
    file_path: str = None,
    train_path: str = None,
    test_path: str = None,
    target_column: str = "target",
    drop_columns: str = ""
):
    import time
    time.sleep(1)  # optional throttle
    
    return do_decision_paths(
        user_id=user_id,
        current_user =current_user,
        file_path=file_path,
        train_path=train_path,
        test_path=test_path,
        target_column=target_column,
        drop_columns=drop_columns
    )
# forecast_tasks.py
# Assuming you have celery_app defined elsewhere
# from your_celery_app import celery_app
@celery_app.task(bind=True, max_retries=3)
def run_forecast(
    self,
    user_id: str,
    current_user: dict,
    file_path: str = None,
    train_path: str = None,
    test_path: str = None,
    target_column: str = "value",
    drop_columns: str = "",
    periods: int = 24,
    scenarios: List[str] = None,
    time_column: str = None,
    enable_backtesting: bool = False
):
    try:

        # Resolve user dir
        user_dir = PathL("user_uploads") / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)



        # 🔁 Updated: Forward both file_path or train/test to your internal logic
        results = do_forecast(
            user_id=user_id,
            current_user=current_user,
            file_path=file_path,
            train_path=train_path,
            test_path=test_path,
            target_column=target_column,
            drop_columns=drop_columns,
            periods=periods,
            scenarios=scenarios or ["comprehensive"],
            time_column=time_column,
            enable_backtesting=enable_backtesting
        )

        # Handle visualizations if needed
        subprocess_success = run_forecast_subprocess(
            user_id=user_id,
            file_path=file_path or train_path,  # fallback for output naming
            forecast_result=results,
            output_dir=user_dir
        )

        if not subprocess_success:
            print(f"⚠️ Warning: Visualization subprocess failed for user {user_id}")

        # Load visualizations
        viz_file = PathL("data/visualizations") / f"{user_id}.json"
        if viz_file.exists():
            with viz_file.open("r", encoding="utf-8") as f:
                visualization_data = json.load(f)

            forecast_entries = [v for v in visualization_data if v.get("type") == "enhanced_forecast"]
            if forecast_entries:
                latest_entry = forecast_entries[-1]
                results["visualizations"] = latest_entry.get("visualizations", {})
                results["thumbnailData"] = latest_entry.get("thumbnailData", "")
                results["imageData"] = latest_entry.get("imageData", "")

        return {
            "status": "success",
            "result": results
        }

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error in forecast task: {error_msg}")
        traceback.print_exc()

        # Retry for transient issues
        if any(keyword in error_msg.lower() for keyword in ["timeout", "connection", "network"]):
            try:
                self.retry(countdown=60, max_retries=3)
            except Exception:
                pass

        return {
            "status": "error",
            "error": error_msg,
            "user_id": user_id
        }
@celery_app.task
def run_ab_test(
    user_id: str,
    file_path: str,
    target_column: str = "converted",
    variant_column: str = "variant",
    drop_columns: str = "",
    current_user: dict = None
):
    import time
    time.sleep(1)  # Optional: rate-limiting, simulate delay
    
    # Delegate to full implementation
    return do_ab_test(
        user_id=user_id,
        file_path=file_path,
        target_column=target_column,
        variant_column=variant_column,
        drop_columns=drop_columns,
        current_user=current_user
    )