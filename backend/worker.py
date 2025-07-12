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
import shap
from typing import Union, List, Any, Dict
import tempfile
# -- replace these imports with your actual module paths --
from .preprocessing import preprocess_data
from .classification import ModelClassifyingTrainer, lgb_params_c, cat_params_c, xgb_params_c
from .feature_importance import safe_generate_feature_importance
import subprocess
from .clustering import run_kmeans, find_optimal_k, label_clusters_general
from .regression import ModelTrainer, lgb_params, cat_params, xgb_params, DataPreprocessor, train_regression_models, generate_visualizations_improved
import joblib
from datetime import datetime
from pathlib import Path as PathL
from celery import Celery
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# -- replace these imports with your actual module paths --
from .preprocessing import preprocess_data
from .classification import ModelClassifyingTrainer, lgb_params_c, cat_params_c, xgb_params_c
from .auth import master_db_cm
from queue import Queue
from threading import Thread

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .anomaly_detection import train_best_anomaly_detection
import logging
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
from .storage import upload_file_to_supabase, download_file_from_supabase, handle_file_upload, download_file_from_supabase, list_user_files, delete_file_from_supabase, get_file_url
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

celery_app = Celery(
    "tasks",
    broker="redis://red-d1n270gdl3ps73fqo7fg:6379/0",
    backend="redis://red-d1n270gdl3ps73fqo7fg:6379/1"
)


# celery_app = Celery("worker", broker="redis://redis:6379/0")

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

from .auth import _append_limited_metadata, _append_metadata, _load_metadata, _save_metadata, _get_meta_path
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

                X, _, _ = preprocess_data(X)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
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

                # Drop ID and target from features
                X_train = train_df.drop(columns=[target_column], errors='ignore')
                X_test = test_df.drop(columns=[target_column], errors='ignore')

                X_train.drop(columns=["ID"], inplace=True, errors='ignore')
                X_test.drop(columns=["ID"], inplace=True, errors='ignore')

                # Now preprocess
                X_train, _, _ = preprocess_data(X_train)
                X_test, _, _ = preprocess_data(X_test)

                common_columns = list(set(X_train.columns) & set(X_test.columns))
                X_train = X_train[common_columns]
                X_test = X_test[common_columns]

                train_df = pd.concat([X_train, y_train.rename(target_column)], axis=1)
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
            with open(user_dir / "training_columns.json", "w") as f:
                json.dump(training_columns, f)
            
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
                    "save_filename": f"{user_id}_feature_importance.png"
                }, f)
            
            # Run SHAP Visualizations subprocess
            try:
                # Get the current script's directory to find shap_runner.py
                current_dir = PathL(__file__).parent
                shap_runner_path = current_dir / "shap_runner.py"
                
                # Ensure shap_runner.py exists
                if not shap_runner_path.exists():
                    print(f"[⚠️] SHAP runner not found at: {shap_runner_path}")
                    raise FileNotFoundError(f"SHAP runner script not found: {shap_runner_path}")
                
                # Run the subprocess with proper error handling
                result = subprocess.run(
                    ["python3", "-m", "backend.shap_runner", str(request_json.resolve())],
                    cwd=str(current_dir.parent),  
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode != 0:
                    print(f"[⚠️] SHAP subprocess failed with return code {result.returncode}")
                    print(f"[⚠️] STDERR: {result.stderr}")
                    print(f"[⚠️] STDOUT: {result.stdout}")
                    raise subprocess.CalledProcessError(result.returncode, result.args)
                
                # Load results from SHAP processing
                result_path = user_dir / "result.json"
                if result_path.exists():
                    with open(result_path) as f:
                        shap_result = json.load(f)
                    fi_shap_bar = shap_result.get("shap_bar")
                    fi_shap_dot = shap_result.get("shap_dot")
                    imp_df = pd.DataFrame(shap_result.get("imp_df", []))
                else:
                    print("[⚠️] SHAP result.json not found")
                    fi_shap_bar, fi_shap_dot, imp_df = None, None, pd.DataFrame()
                    
            except subprocess.TimeoutExpired:
                print("[⚠️] SHAP subprocess timed out after 5 minutes")
                fi_shap_bar, fi_shap_dot, imp_df = None, None, pd.DataFrame()
            except subprocess.CalledProcessError as e:
                print(f"[⚠️] SHAP subprocess failed: {e}")
                fi_shap_bar, fi_shap_dot, imp_df = None, None, pd.DataFrame()
            except Exception as viz_error:
                print(f"[⚠️] Visualization error: {viz_error}")
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
                "impact_metrics": impact_metrics
            }
            
            # Add additional visualizations
            if class_dist_base64:
                entry["visualizations"]["class_distribution"] = f"data:image/png;base64,{class_dist_base64}"
            if confusion_matrix_base64:
                entry["visualizations"]["confusion_matrix"] = f"data:image/png;base64,{confusion_matrix_base64}"
            if classification_report_base64:
                entry["visualizations"]["classification_report"] = f"data:image/png;base64,{classification_report_base64}"

            try:
                with master_db_cm() as db:  # ✅ safely create and commit DB session
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
                "parameters": {"drop_columns": drop_columns},
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
        
        model_supabase_path = f"{user_id}/models/{user_id}_best_classifier.pkl"

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

            try:
                with master_db_cm() as db:  # ✅ safely create and commit DB session
                    _append_limited_metadata(user_id, entry, db=db, max_entries=5)
            except Exception as meta_error:
                print(f"[⚠️] Failed to save prediction metadata: {meta_error}")

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
                df = df.dropna(subset=[target_column])
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
                train_df = train_df.dropna(subset=[target_column])
                test_df = test_df.dropna(subset=[target_column])
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
                "id": entry["id"],  # Use the same one
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
                response_data["visualizations"]["cluster_visualization"] = f"data:image/png;base64,{cluster_viz_base64}"
            if elbow_base64:
                response_data["visualizations"]["elbow_method"] = f"data:image/png;base64,{elbow_base64}"
            
            entry = {
                "id": entry["id"],  # Use the same one
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
                # ✅ Add uploaded file references
                "model_url": model_url,
                "data_url": data_url,
                "model_path": model_upload_path,
                "data_path": data_upload_path
            }

            try:
                with master_db_cm() as db:  # ✅ safely create and commit DB session
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
                df = df.dropna(subset=[target_column])
                dataset_name = os.path.basename(file_path)
            else:
                # Train/test pair mode
                train_df = pd.read_csv(local_train_path)
                test_df = pd.read_csv(local_test_path)
                train_df = train_df.dropna(subset=[target_column])
                test_df = test_df.dropna(subset=[target_column])
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
                "id": entry["id"],  # Use the same one
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
                response_data["visualizations"]["cluster_visualization"] = f"data:image/png;base64,{cluster_viz_base64}"
            if elbow_base64:
                response_data["visualizations"]["elbow_method"] = f"data:image/png;base64,{elbow_base64}"

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

            try:
                with master_db_cm() as db:  # ✅ safely create and commit DB session
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

            try:
                with master_db_cm() as db:  # ✅ safely create and commit DB session
                    _append_limited_metadata(user_id, entry, db=db, max_entries=5)
            except Exception as meta_error:
                print(f"[⚠️] Metadata save error: {meta_error}")

            logger.info(f"✅ Cluster labeling completed for user_id: {user_id}")

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
                train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

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

            # ───────────── Upload to Supabase ─────────────
            try:
                model_filename = model_path.name
                preprocessor_filename = preprocessor_path.name

                # Upload model and preprocessor
                model_supabase_path = upload_file_to_supabase(user_id, str(model_path), model_filename)
                preprocessor_supabase_path = upload_file_to_supabase(user_id, str(preprocessor_path), preprocessor_filename)

                # Upload processed dataset
                data_filename = processed_data_path.name
                data_supabase_path = upload_file_to_supabase(user_id, str(processed_data_path), data_filename)

                # Get signed URLs for frontend access (1-hour expiration)
                model_url = get_file_url(model_supabase_path, expires_in=3600)
                preprocessor_url = get_file_url(preprocessor_supabase_path, expires_in=3600)
                data_url = get_file_url(data_supabase_path, expires_in=3600)

                print(f"[✅] Uploaded model: {model_url}")
                print(f"[✅] Uploaded preprocessor: {preprocessor_url}")
                print(f"[✅] Uploaded processed data: {data_url}")

                # Append to response_data
                response_data.update({
                    "model_url": model_url,
                    "preprocessor_url": preprocessor_url,
                    "data_url": data_url
                })

            except Exception as upload_error:
                print(f"[⚠️] Failed to upload regression artifacts to Supabase: {upload_error}")


            # ───────────── Generate Visualizations ─────────────
            try:
                if local_file_path:
                    full_df = pd.read_csv(local_file_path)
                    full_df = full_df.dropna(subset=[target_column])
                    if drop_columns:
                        drops = [c.strip() for c in drop_columns.split(",") if c.strip() and c in full_df.columns]
                        if drops:
                            full_df.drop(columns=drops, inplace=True)
                    full_df, _, _ = preprocess_data(full_df)
                else:
                    full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

                X_full_raw = full_df.drop(columns=["ID", target_column], errors="ignore")
                X_full_processed = results["preprocessor"].transform(X_full_raw)

                if hasattr(results["preprocessor"], "get_feature_names_out"):
                    feature_names = results["preprocessor"].get_feature_names_out()
                else:
                    feature_names = results.get("feature_names")

                X_full_df = pd.DataFrame(X_full_processed, columns=feature_names)
                
                # ───────────── Predictions on Full Data ─────────────
                preds_full = results["final_model"].predict(X_full_processed)

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

                # Save processed data for SHAP
                processed_data_path = user_dir / "processed_full_data.csv"
                X_full_df.to_csv(processed_data_path, index=False)

                # Create request JSON for SHAP
                request_json = user_dir / "request.json"
                with open(request_json, "w") as f:
                    json.dump({
                        "user_id": str(user_id),
                        "model_path": str(model_path.resolve()),
                        "data_path": str(processed_data_path.resolve()),
                        "output_dir": str(user_dir.resolve()),
                        "model_type": "regressor",
                        "target_column": target_column,
                        "drop_columns": drop_columns,
                    }, f)

               
                try:
                    # Build request.json for regression SHAP
                    request_json = user_dir / "request.json"
                    with open(request_json, "w") as f:
                        json.dump({
                            "user_id": str(user_id),
                            "model_path": str(model_path.resolve()),
                            "data_path": str(processed_data_path.resolve()),
                            "output_dir": str(user_dir.resolve()),
                            "model_type": "regression",  # <- updated model_type
                            "target_column": target_column,
                            "save_filename": f"{user_id}_feature_importance.png"
                        }, f)

                    # SHAP subprocess
                    current_dir = PathL(__file__).parent
                    result = subprocess.run(
                        ["python3", "-m", "backend.shap_runner", str(request_json.resolve())],
                        cwd=str(current_dir.parent),  # run from parent of "backend/"
                        capture_output=True,
                        text=True,
                        timeout=300
                    )

                    if result.returncode != 0:
                        print(f"[⚠️] SHAP subprocess failed with return code {result.returncode}")
                        print(f"[⚠️] STDERR: {result.stderr}")
                        print(f"[⚠️] STDOUT: {result.stdout}")
                        raise subprocess.CalledProcessError(result.returncode, result.args)

                    # Load result
                    result_path = user_dir / "result.json"
                    if result_path.exists():
                        with open(result_path) as f:
                            shap_result = json.load(f)
                        fi_shap_bar = shap_result.get("shap_bar")
                        fi_shap_dot = shap_result.get("shap_dot")
                        imp_df = pd.DataFrame(shap_result.get("imp_df", []))
                    else:
                        print("[⚠️] SHAP result.json not found")
                        fi_shap_bar, fi_shap_dot, imp_df = None, None, pd.DataFrame()

                except subprocess.TimeoutExpired:
                    print("[⚠️] SHAP subprocess timed out after 5 minutes")
                    fi_shap_bar, fi_shap_dot, imp_df = None, None, pd.DataFrame()
                except subprocess.CalledProcessError as e:
                    print(f"[⚠️] SHAP subprocess failed: {e}")
                    fi_shap_bar, fi_shap_dot, imp_df = None, None, pd.DataFrame()
                except Exception as viz_error:
                    print(f"[⚠️] Visualization error: {viz_error}")
                    import traceback
                    traceback.print_exc()
                    fi_shap_bar, fi_shap_dot, imp_df = None, None, pd.DataFrame()

                # ───────────── Generate Insights ─────────────
                try:
                    insights = generate_regression_insights(
                        pred_stats=pred_stats,
                        top_features=imp_df.head(10).to_dict("records") if not imp_df.empty else [],
                        financial_inputs=financial_inputs
                    )
                    
                    # Save Metadata for Gallery
                    entry = {
                        "id": str(uuid.uuid4()),
                        "created_at": datetime.utcnow().isoformat(),
                        "type": "regression",
                        "dataset": dataset_name,
                        "parameters": {
                            "target_column": target_column,
                            "drop_columns": drop_columns
                        },
                        "metrics": results.get("metrics", {}),
                        "thumbnailData": f"data:image/png;base64,{fi_shap_bar}" if fi_shap_bar else "",
                        "imageData": f"data:image/png;base64,{fi_shap_dot or fi_shap_bar or ''}",
                        "top_features": imp_df.head(10).to_dict("records") if not imp_df.empty else [],
                        "visualizations": {
                            "shap_bar": f"data:image/png;base64,{fi_shap_bar}" if fi_shap_bar else "",
                            "shap_dot": f"data:image/png;base64,{fi_shap_dot}" if fi_shap_dot else ""
                        },
                        "pred_stats": pred_stats,
                        "insights": insights,
                        "model_url": model_url if 'model_url' in locals() else "",
                        "preprocessor_url": preprocessor_url if 'preprocessor_url' in locals() else "",
                        "data_url": data_url if 'data_url' in locals() else ""
                    }

                    
                    try:
                        with master_db_cm() as db:  # ✅ safely create and commit DB session
                            _append_limited_metadata(user_id, entry, db=db, max_entries=5)
                    except Exception as meta_error:
                        print(f"[⚠️] Metadata save error: {meta_error}")

                except Exception as insight_err:
                    print(f"[⚠️] Insight generation failed: {insight_err}")
                    insights = "Could not generate insights."

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
                    "metrics": results.get("metrics", {}),
                    "message": "Regression model training completed",
                    "pred_stats": pred_stats,
                    "top_features": imp_df.head(10).to_dict("records") if not imp_df.empty else {},
                    "insights": insights if 'insights' in locals() else "Could not generate insights.",
                    "visualizations": {
                        "shap_bar": f"data:image/png;base64,{fi_shap_bar}" if fi_shap_bar else "",
                        "shap_dot": f"data:image/png;base64,{fi_shap_dot}" if fi_shap_dot else ""
                    },
                    "model_url": model_url if 'model_url' in locals() else "",
                    "preprocessor_url": preprocessor_url if 'preprocessor_url' in locals() else "",
                    "data_url": data_url if 'data_url' in locals() else ""
                }


                if fi_shap_bar:
                    response_data["visualizations"]["shap_bar"] = f"data:image/png;base64,{fi_shap_bar}"
                if fi_shap_dot:
                    response_data["visualizations"]["shap_dot"] = f"data:image/png;base64,{fi_shap_dot}"

                return response_data

            except Exception as viz_global_error:
                print(f"[⚠️] Visualization pipeline error: {viz_global_error}")
                return {"status": "failed", "error": str(viz_global_error)}

    except Exception as e:
        print(f"[⚠️] Regression error: {e}")
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
        # Setup paths - models are still stored locally, but data files are in Supabase
        model_path = PathL("models") / f"{user_id}_best_regressor.pkl"
        preprocessor_path = PathL("models") / f"{user_id}_preprocessor.pkl"
        
        # For training columns, we'll also store this in Supabase
        training_columns_supabase_path = f"{user_id}/training_columns.json"

        # Validate required files exist
        if not model_path.exists() or not preprocessor_path.exists():
            raise FileNotFoundError(f"No trained model or preprocessor found for user {user_id}. Please train a model first.")
        
        if file_path is None:
            raise ValueError("You must provide a file_path for prediction input.")

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
        
        # Load the trained model and preprocessor (still stored locally)
        print(f"[📁] Loading model from {model_path}")
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        
        # Create temporary file to load CSV data
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as temp_file:
            temp_file.write(prediction_data)
            temp_csv_path = temp_file.name
        
        try:
            # Load and preprocess prediction data
            print(f"[📊] Loading prediction data from Supabase: {file_path}")
            df = pd.read_csv(temp_csv_path)
            original_shape = df.shape
            
            # Store original dataframe for final output
            original_df = df.copy()
            
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

            try:
                with master_db_cm() as db:  # ✅ safely create and commit DB session
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
            
        finally:
            # Clean up temporary CSV file
            os.unlink(temp_csv_path)

    except Exception as e:
        print(f"[❌] Prediction task error: {e}")
        return {
            "status": "error",
            "user_id": user_id,
            "error_message": str(e),
            "error_type": type(e).__name__
        }
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
                df = df.dropna(subset=[target_column])
                dataset_name = os.path.basename(file_path)
            else:
                train_df = pd.read_csv(local_train_path)
                test_df = pd.read_csv(local_test_path)
                train_df = train_df.dropna(subset=[target_column])
                test_df = test_df.dropna(subset=[target_column])
                df = pd.concat([train_df, test_df], ignore_index=True)
                dataset_name = f"{os.path.basename(train_path)}+{os.path.basename(test_path)}"

            # ───────────── Preprocess data ─────────────
            remove_cols = ["ID", target_column] if "ID" in df.columns else [target_column]
            df, CATS, NUMS = preprocess_data(df, RMV=remove_cols)

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
                    "visualizations": {}
                }

                if scatter_b64:
                    entry["visualizations"]["scatter_plot"] = f"data:image/png;base64,{scatter_b64}"
                
                if pdp_b64:
                    entry["visualizations"]["pdp_plot"] = f"data:image/png;base64,{pdp_b64}"

                # Use the same metadata saving function as classification
                try:
                    with master_db_cm() as db:  # ✅ safely create and commit DB session
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


def clean_data_for_json(data):
    """Recursively clean data structure for JSON serialization"""
    if isinstance(data, dict):
        return {k: clean_data_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif isinstance(data, (int, float, np.integer, np.floating)):
        val = float(data)
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return val
    elif isinstance(data, str):
        return data
    else:
        return str(data)

def safe_float_conversion(value):
    """Convert value to float, handling NaN and infinity"""
    try:
        val = float(value)
        if np.isnan(val) or np.isinf(val):
            return 0.0  # or some default value
        return val
    except (ValueError, TypeError):
        return 0.0
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

                y = df[target_column]
                X = df.drop(columns=['ID', target_column], errors='ignore')
                X, _, _ = preprocess_data(X)

                # Split to simulate train/test consistency
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                train_df = pd.concat([X_train, y_train.rename(target_column)], axis=1)
                test_df = pd.concat([X_test, y_test.rename(target_column)], axis=1)
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

            # ──────── Load trained model ────────
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
                else:
                    raise ValueError("No trained model found for this user")

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

            sample_metrics = {}
            for col in numeric_cols:
                col_data = sample[col]
                sample_metrics[col] = {
                    "mean": safe_float_conversion(col_data.mean()),
                    "median": safe_float_conversion(col_data.median()),
                    "std": safe_float_conversion(col_data.std()),
                    "min": safe_float_conversion(col_data.min()),
                    "max": safe_float_conversion(col_data.max())
                }

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
                    "metrics": sample_metrics
                }

                # Add visualizations to entry
                for i, viz_b64 in enumerate(visualizations):
                    entry["visualizations"][f"counterfactual_{i+1}"] = f"data:image/png;base64,{viz_b64}"

                if summary_b64:
                    entry["visualizations"]["counterfactual_summary"] = f"data:image/png;base64,{summary_b64}"

                # Clean entry for JSON serialization
                entry = clean_data_for_json(entry)

                # Use the same metadata saving function as classification
                try:
                    with master_db_cm() as db:  # ✅ safely create and commit DB session
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
                df = df.dropna(subset=[target_column])
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
                from sklearn.model_selection import train_test_split
                train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

            else:
                train_df = pd.read_csv(local_train_path)
                test_df = pd.read_csv(local_test_path)
                train_df = train_df.dropna(subset=[target_column])
                test_df = test_df.dropna(subset=[target_column])
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

            # Apply cleaning
            train_df = clean_dataframe_for_cox(train_df, time_col, event_col)
            
            # Check if test set has target columns
            has_test_targets = time_col in test_df.columns and event_col in test_df.columns
            if has_test_targets:
                test_df = clean_dataframe_for_cox(test_df, time_col, event_col)
            else:
                # No target columns in test set, only preprocess features
                test_df = clean_dataframe_for_cox(test_df, None, None)

            # Verify no NaN values in final datasets
            print(f"Train set shape: {train_df.shape}")
            print(f"Train set NaN check: {train_df.isna().sum().sum()} total NaN values")
            print(f"Test set shape: {test_df.shape}")
            print(f"Test set NaN check: {test_df.isna().sum().sum()} total NaN values")
            
            # ───────────── Train CoxPH ─────────────
            # First fit on training data only
            cph = CoxPHFitter()
            cph.fit(train_df, duration_col=time_col, event_col=event_col)
            
            # For the full model, only use data that has target columns
            if has_test_targets:
                # Ensure columns match before concatenation
                train_cols = set(train_df.columns)
                test_cols = set(test_df.columns)
                
                # Find common columns
                common_cols = train_cols.intersection(test_cols)
                
                # Make sure target columns are included
                if time_col not in common_cols or event_col not in common_cols:
                    print(f"Warning: Target columns missing in test set. Using training data only for full model.")
                    full_df = train_df.copy()
                else:
                    # Reorder columns to match
                    ordered_cols = list(common_cols)
                    train_subset = train_df[ordered_cols].copy()
                    test_subset = test_df[ordered_cols].copy()
                    
                    # Additional safety check for NaN values before concatenation
                    print(f"Pre-concat train NaN check: {train_subset.isna().sum().sum()}")
                    print(f"Pre-concat test NaN check: {test_subset.isna().sum().sum()}")
                    
                    # Ensure data types match
                    for col in ordered_cols:
                        if train_subset[col].dtype != test_subset[col].dtype:
                            # Convert both to float32 for consistency
                            train_subset[col] = train_subset[col].astype('float32')
                            test_subset[col] = test_subset[col].astype('float32')
                    
                    full_df = pd.concat([train_subset, test_subset], axis=0, ignore_index=True)
                    
                    # Final NaN check after concatenation
                    print(f"Post-concat full_df shape: {full_df.shape}")
                    print(f"Post-concat NaN check: {full_df.isna().sum().sum()}")
                    
                    # If there are still NaNs, handle them
                    if full_df.isna().sum().sum() > 0:
                        print("Warning: NaNs detected after concatenation. Cleaning...")
                        full_df = clean_dataframe_for_cox(full_df, time_col, event_col)
            else:
                print("Test set has no target columns. Using training data only for full model.")
                full_df = train_df.copy()

            # Fit the full model
            cph_full = CoxPHFitter()
            cph_full.fit(full_df, duration_col=time_col, event_col=event_col)
            
            # ──────── Survival Model Metrics ────────
            c_index = cph_full.concordance_index_
            log_likelihood = cph_full.log_likelihood_
            aic = -2 * log_likelihood + 2 * len(cph_full.params_)
            bic = -2 * log_likelihood + len(cph_full.params_) * np.log(full_df.shape[0])
            significant_predictors = int((cph_full.summary["p"] < 0.05).sum())

            survival_metrics = {
                "concordance_index": round(c_index, 4),
                "log_likelihood": round(log_likelihood, 4),
                "AIC": round(aic, 4),
                "BIC": round(bic, 4),
                "significant_predictors": significant_predictors
            }
            
            # Save model to temporary directory first
            temp_model_path = PathL(model_dir) / f"{user_id}_survival_cox.pkl"
            joblib.dump(cph_full, temp_model_path)
            
            # Upload model to Supabase
            try:
                with open(temp_model_path, 'rb') as f:
                    model_bytes = f.read()
                supabase_model_path = f"models/{user_id}_survival_cox.pkl"
                upload_file_to_supabase(model_bytes, supabase_model_path)
                print(f"Model uploaded to Supabase: {supabase_model_path}")
            except Exception as upload_error:
                print(f"Warning: Failed to upload model to Supabase: {upload_error}")

            # ───────────── Generate Visualizations ─────────────
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
                    "retention_rates": retention_rates
                }

                if risk_fig_base64:
                    entry["visualizations"]["risk_distribution"] = f"data:image/png;base64,{risk_fig_base64}"

                # Use the same metadata saving function as visualization
                try:
                    with master_db_cm() as db:  # ✅ safely create and commit DB session
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
                "retention_rates": retention_rates
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
    sample_id: int = None,
    target_column: str = None,
    feature_changes: str = ""
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
    import shap
    from pathlib import Path as PathL
    from datetime import datetime
    
    try:
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
                raise ValueError("No model found")

            model = joblib.load(model_path)

            # Get original sample
            if 'ID' in df.columns:
                original_sample = df[df['ID'] == sample_id]
            else:
                original_sample = df.iloc[sample_id:sample_id+1]

            if len(original_sample) == 0:
                raise ValueError("Sample not found")

            # Create modified sample
            modified_sample = original_sample.copy()
            changes = json.loads(feature_changes)

            for feature, new_value in changes.items():
                if feature in modified_sample.columns:
                    modified_sample[feature] = new_value

            # Get predictions for both original and modified
            features_cols = [col for col in df.columns if col != target_column and col != 'ID']
            original_pred = model[0].predict(original_sample[features_cols])
            modified_pred = model[0].predict(modified_sample[features_cols])
            prediction_diff = float(modified_pred[0]) - float(original_pred[0])
            percent_change = (
                (prediction_diff / float(original_pred[0])) * 100 if original_pred[0] != 0 else None
            )

            # Directional summary for the AI/UX
            if prediction_diff > 0:
                direction = "increase"
            elif prediction_diff < 0:
                direction = "decrease"
            else:
                direction = "no change"

            impact_metrics = {
                "original_prediction": float(original_pred[0]),
                "modified_prediction": float(modified_pred[0]),
                "absolute_change": round(prediction_diff, 4),
                "percent_change": round(percent_change, 2) if percent_change is not None else None,
                "direction": direction
            }

            # Get SHAP values for both
            explainer = shap.TreeExplainer(model[0])
            original_shap = explainer.shap_values(original_sample[features_cols])
            modified_shap = explainer.shap_values(modified_sample[features_cols])

            # Generate comparison visualization
            fig = plt.figure(figsize=(12, 8))
            plt.subplot(1, 2, 1)
            shap.waterfall_plot(
                explainer.expected_value if isinstance(explainer.expected_value, float) else explainer.expected_value[0],
                original_shap[0] if isinstance(original_shap, list) else original_shap[0, :],
                original_sample[features_cols].iloc[0],
                show=False
            )
            plt.title("Original Sample")

            plt.subplot(1, 2, 2)
            shap.waterfall_plot(
                explainer.expected_value if isinstance(explainer.expected_value, float) else explainer.expected_value[0],
                modified_shap[0] if isinstance(modified_shap, list) else modified_shap[0, :],
                modified_sample[features_cols].iloc[0],
                show=False
            )
            plt.title("Modified Sample")

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches="tight")
            buf.seek(0)
            comparison_base64 = base64.b64encode(buf.read()).decode()
            plt.close(fig)

            # ───────────── Save metadata to Supabase ─────────────
            try:
                # Create entry for metadata
                entry = {
                    "id": str(uuid.uuid4()),
                    "created_at": datetime.utcnow().isoformat(),
                    "type": "what_if_analysis",
                    "dataset": dataset_name,
                    "parameters": {
                        "target_column": target_column,
                        "sample_id": sample_id,
                        "feature_changes": changes
                    },
                    "thumbnailData": f"data:image/png;base64,{comparison_base64}" if comparison_base64 else "",
                    "imageData": f"data:image/png;base64,{comparison_base64}" if comparison_base64 else "",
                    "visualizations": {
                        "comparison_plot": f"data:image/png;base64,{comparison_base64}"
                    },
                    "metrics": impact_metrics
                }

                # Use the same metadata saving function as visualization
                try:
                    with master_db_cm() as db:  # ✅ safely create and commit DB session
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
                "type": "what_if_analysis",
                "dataset": dataset_name,
                "parameters": {
                    "target_column": target_column,
                    "sample_id": sample_id,
                    "feature_changes": changes
                },
                "original_prediction": float(original_pred[0]),
                "modified_prediction": float(modified_pred[0]),
                "feature_changes": changes,
                "visualization": f"data:image/png;base64,{comparison_base64}",
                "metrics": impact_metrics,
                "visualizations": {
                    "comparison_plot": f"data:image/png;base64,{comparison_base64}"
                },
                "insights": {}
            }


            # Return full response
            return response_data

    except Exception as e:
        print(f"[⚠️] Error in do_what_if: {e}")
        raise ValueError(f"What-if analysis failed: {str(e)}")
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

                # Use the same metadata saving function as visualization
                try:
                    with master_db_cm() as db:  # ✅ safely create and commit DB session
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
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
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

                if target_column not in train_df.columns or target_column not in test_df.columns:
                    raise ValueError(f"Target column '{target_column}' not found in training or test dataset.")

                y_train = train_df[target_column]
                y_test = test_df[target_column]

                X_train = train_df.drop(columns=[target_column, 'ID'], errors='ignore')
                X_test = test_df.drop(columns=[target_column, 'ID'], errors='ignore')

                X_train, _, _ = preprocess_data(X_train)
                X_test, _, _ = preprocess_data(X_test)

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
                    }
                }

                # Use the same metadata saving function as visualization
                try:
                    with master_db_cm() as db:  # ✅ safely create and commit DB session
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

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Fix: Use proper Path import
from pathlib import Path as PathL
def do_forecast(user_id: str, current_user: dict, file_path: str, target_column: str = "value", drop_columns: str = "", periods: int = 24, compare_models: bool = False):
    """
    Perform ARIMA forecasting on time series data with Supabase integration
    """
    import numpy as np
    import os
    import uuid
    import tempfile
    import json
    from pathlib import Path as PathL
    from datetime import datetime
    
    # Handle file downloads from Supabase
    # Note: file_path is now a Supabase storage path (e.g., "user123/dataset.csv")
    local_file_path = None
    
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download file from Supabase to temporary location
            file_bytes = download_file_from_supabase(file_path)
            local_file_path = os.path.join(temp_dir, os.path.basename(file_path))
            with open(local_file_path, 'wb') as f:
                f.write(file_bytes)
            
            # Create temporary directories for models and outputs
            model_dir = os.path.join(temp_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            
            user_dir = PathL(temp_dir) / "user_outputs"
            user_dir.mkdir(exist_ok=True)

            # Read the CSV file
            try:
                df = pd.read_csv(local_file_path, sep=None, engine="python")
                dataset_name = os.path.basename(file_path)
            except Exception as e:
                raise ValueError(f"Error reading file: {e}")
            
            # Drop specified columns
            if drop_columns:
                drops = [c.strip() for c in drop_columns.split(",") if c.strip() and c.strip() in df.columns]
                if drops:
                    df.drop(columns=drops, inplace=True)
            
            # Check if target column exists
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset. Available columns: {list(df.columns)}")
           
            # Extract the time series
            series = df[target_column].dropna()
            
            if len(series) < 10:
                raise ValueError("Time series too short for ARIMA modeling (minimum 10 observations required)")
            
            # Determine differencing order (d) using ADF test
            d = 0
            adf_stat = None
            p_value = None
            
            for i in range(3):  # Try d = 0, 1, 2
                test_series = series.copy()
                for _ in range(i):
                    test_series = test_series.diff().dropna()
                
                if len(test_series) < 5:  # Need minimum data for ADF test
                    continue
                    
                try:
                    adf_stat, p_value, *_ = adfuller(test_series)
                    if p_value < 0.05:
                        d = i
                        break
                except Exception:
                    continue
            
            # If ADF test failed, default to d=1
            if adf_stat is None or p_value is None:
                d = 1
                adf_stat = 0
                p_value = 1.0
            
            # ARIMA model selection using AIC
            best_aic = float("inf")
            best_order = None
            best_model = None
            
            # Grid search for best ARIMA parameters
            for p in range(3):
                for q in range(3):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        model_fit = model.fit()
                        
                        if model_fit.aic < best_aic:
                            best_aic = model_fit.aic
                            best_order = (p, d, q)
                            best_model = model_fit
                            
                    except Exception as e:
                        continue
            
            if not best_model:
                # Fallback to simple ARIMA(1,1,1) if grid search fails
                try:
                    model = ARIMA(series, order=(1, 1, 1))
                    best_model = model.fit()
                    best_order = (1, 1, 1)
                except Exception:
                    raise RuntimeError("Could not fit any ARIMA model. Please check your data.")
            
            stationary = p_value < 0.05
            
            # Generate forecast
            try:
                forecast_result = best_model.get_forecast(steps=periods)
                forecast_mean = forecast_result.predicted_mean
                conf_int = forecast_result.conf_int()
            except Exception as e:
                raise RuntimeError(f"Error generating forecast: {e}")
            
            # Generate visualization (if matplotlib/seaborn available)
            forecast_plot_b64 = None
            try:
                import matplotlib.pyplot as plt
                
                # Create forecast plot
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot historical data
                ax.plot(range(len(series)), series, label='Historical Data', color='blue')
                
                # Plot forecast
                forecast_start = len(series)
                forecast_range = range(forecast_start, forecast_start + periods)
                ax.plot(forecast_range, forecast_mean, label='Forecast', color='red', linestyle='--')
                
                # Plot confidence intervals
                ax.fill_between(forecast_range, conf_int.iloc[:, 0], conf_int.iloc[:, 1], 
                               color='red', alpha=0.2, label='Confidence Interval')
                
                ax.set_title(f'ARIMA({best_order[0]},{best_order[1]},{best_order[2]}) Forecast')
                ax.set_xlabel('Time Period')
                ax.set_ylabel(target_column)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Convert to base64
                forecast_plot_b64 = plot_to_base64(fig)
                plt.close(fig)
                
            except Exception as viz_error:
                print(f"[⚠️] Visualization generation failed: {viz_error}")
                forecast_plot_b64 = None
            
            # Prepare main response data
            response_data = {
                "status": "success",
                "user_id": user_id,
                "id": str(uuid.uuid4()),
                "created_at": datetime.utcnow().isoformat(),
                "type": "forecast",
                "dataset": dataset_name,
                "dataset_name": dataset_name,
                "series": series.tolist(),
                "model_order": {"p": best_order[0], "d": best_order[1], "q": best_order[2]},
                "model_aic": best_aic,
                "adf_test": {
                    "statistic": float(adf_stat),
                    "p_value": float(p_value),
                    "d_used": d,
                    "is_stationary": str(stationary)
                },
                "parameters": {
                    "target_column": target_column,
                    "drop_columns": drop_columns,
                    "periods": periods,
                    "compare_models": compare_models
                },
                "forecast_periods": periods,
                "forecast_values": forecast_mean.tolist(),
                "confidence_intervals": {
                    "lower": conf_int.iloc[:, 0].tolist(),
                    "upper": conf_int.iloc[:, 1].tolist()
                },
                "visualizations": {},
                "insights": {}
            }
            
            # Add visualization data
            if forecast_plot_b64:
                response_data["visualizations"]["forecast_plot"] = f"data:image/png;base64,{forecast_plot_b64}"
                response_data["thumbnailData"] = f"data:image/png;base64,{forecast_plot_b64}"
                response_data["imageData"] = f"data:image/png;base64,{forecast_plot_b64}"
            
            # Add model_results for visualization compatibility
            model_name = f"ARIMA({best_order[0]},{best_order[1]},{best_order[2]})"
            response_data["model_results"] = {
                model_name: {
                    "forecast": forecast_mean.tolist(),
                    "conf_int": {
                        "lower": conf_int.iloc[:, 0].tolist(),
                        "upper": conf_int.iloc[:, 1].tolist()
                    },
                    "aic": best_aic,
                    "order": f"({best_order[0]},{best_order[1]},{best_order[2]})"
                }
            }
            
            # If compare_models is True, add additional models for comparison
            if compare_models:
                additional_models = [(0, d, 1), (2, d, 0), (1, d, 2)]
                for p, d_val, q in additional_models:
                    if (p, d_val, q) != best_order:
                        try:
                            model = ARIMA(series, order=(p, d_val, q))
                            model_fit = model.fit()
                            forecast_result = model_fit.get_forecast(steps=periods)
                            fc_mean = forecast_result.predicted_mean
                            fc_conf_int = forecast_result.conf_int()
                            
                            model_name = f"ARIMA({p},{d_val},{q})"
                            response_data["model_results"][model_name] = {
                                "forecast": fc_mean.tolist(),
                                "conf_int": {
                                    "lower": fc_conf_int.iloc[:, 0].tolist(),
                                    "upper": fc_conf_int.iloc[:, 1].tolist()
                                },
                                "aic": model_fit.aic,
                                "order": f"({p},{d_val},{q})"
                            }
                        except Exception:
                            continue

            # Save metadata to Supabase (similar to visualization)
            try:
                # Create entry for metadata
                entry = {
                    "id": response_data["id"],
                    "created_at": response_data["created_at"],
                    "type": "forecast",
                    "dataset": dataset_name,
                    "parameters": response_data["parameters"],
                    "thumbnailData": response_data.get("thumbnailData", ""),
                    "imageData": response_data.get("imageData", ""),
                    "visualizations": response_data["visualizations"],
                    "model_results": response_data["model_results"],
                    "forecast_summary": {
                        "model_order": response_data["model_order"],
                        "model_aic": response_data["model_aic"],
                        "forecast_periods": response_data["forecast_periods"],
                        "adf_test": response_data["adf_test"]
                    }
                }

                # Use the same metadata saving function as visualization
                try:
                    with master_db_cm() as db:  # ✅ safely create and commit DB session
                        _append_limited_metadata(user_id, entry, db=db, max_entries=5)
                except Exception as meta_error:
                    print(f"[⚠️] Metadata save error: {meta_error}")

            except Exception as meta_error:
                print(f"[⚠️] Metadata save error: {meta_error}")
            
            return response_data
            
    except Exception as e:
        print(f"[⚠️] Error in do_forecast: {e}")
        raise e
def create_training_test_forecast(df, target_column="value", train_ratio=0.7, order=(1,1,1)):
    """
    Create training/test split and forecast (following your reference implementation)
    """
    series = df[target_column].dropna()
    
    # Create training and test sets
    train_size = int(len(series) * train_ratio)
    train = series[:train_size]
    test = series[train_size:]
    
    # Build model
    model = ARIMA(train, order=order)
    fitted = model.fit()
    
    # Forecast
    forecast_steps = len(test)
    forecast_result = fitted.get_forecast(steps=forecast_steps)
    
    fc = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()
    
    # Create forecast series with proper index
    fc_series = pd.Series(fc.values, index=test.index)
    lower_series = pd.Series(conf_int.iloc[:, 0].values, index=test.index)
    upper_series = pd.Series(conf_int.iloc[:, 1].values, index=test.index)
    
    # Create plot
    plt.figure(figsize=(12, 5), dpi=100)
    plt.plot(train, label='Training', color='blue')
    plt.plot(test, label='Actual', color='green')
    plt.plot(fc_series, label='Forecast', color='red', linestyle='--')
    plt.fill_between(lower_series.index, lower_series, upper_series, 
                     color='red', alpha=0.15)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return {
        'model': fitted,
        'forecast': fc_series,
        'confidence_intervals': (lower_series, upper_series),
        'train': train,
        'test': test
    }

def plot_model_diagnostics(model_fit):
    """
    Plot model diagnostics (residuals analysis)
    """
    residuals = pd.DataFrame(model_fit.resid)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    residuals.plot(title="Residuals", ax=ax[0])
    residuals.plot(kind='kde', title='Density', ax=ax[1])
    plt.tight_layout()
    plt.show()
    
    # Plot predictions
    model_fit.plot_predict(dynamic=False)
    plt.show()
import sys
def run_forecast_subprocess(user_id, file_path, forecast_result, output_dir):
    """
    Run forecast visualization subprocess.
    Saves request.json and executes forecast_runner.py
    """
    output_dir = PathL(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    request_data = {
        "user_id": str(user_id),
        "file_path": str(file_path),
        "forecast_result": forecast_result,
        "output_dir": str(output_dir)
    }

    try:
        # Save request.json into the output directory
        request_path = output_dir / "request.json"
        with request_path.open("w", encoding="utf-8") as f:
            json.dump(request_data, f, indent=2)

        # Locate forecast_runner.py
        current_dir = PathL(__file__).parent
        runner_script = current_dir / "forecast_runner.py"

        if not runner_script.exists():
            print(f"[⚠️] Forecast runner not found at: {runner_script}")
            raise FileNotFoundError(f"Forecast runner script not found: {runner_script}")

        # Run the subprocess
        result = subprocess.run(
            ["python3", "-m", "backend.forecast_runner", str(request_path.resolve())],
            cwd=str(current_dir.parent),  # go up so `backend` is importable
            capture_output=True,
            text=True,
            timeout=300
        )


        if result.returncode != 0:
            print(f"[⚠️] Forecast subprocess failed with return code {result.returncode}")
            print(f"[⚠️] STDERR: {result.stderr}")
            print(f"[⚠️] STDOUT: {result.stdout}")
            raise subprocess.CalledProcessError(result.returncode, result.args)

        print(f"[✅] Forecast subprocess ran successfully for user {user_id}")
        return True

    except subprocess.TimeoutExpired:
        print(f"[⚠️] Forecast subprocess timed out after 5 minutes for user {user_id}")
        return False

    except subprocess.CalledProcessError as e:
        print(f"[⚠️] Forecast subprocess failed: {e}")
        return False

    except Exception as e:
        print(f"[⚠️] Forecast subprocess error: {e}")
        import traceback
        traceback.print_exc()
        return False



@celery_app.task(bind=True, max_retries=3)
def run_forecast(self, user_id: str, current_user: dict, file_path: str, 
                target_column: str = "value", drop_columns: str = "", periods: int = 24, compare_models: bool = False):
    """
    Celery task for running ARIMA forecasting
    """
    try:
        # Validate inputs
        if not user_id:
            raise ValueError("user_id is required")
        if not file_path:
            raise ValueError("file_path is required")
        if not PathL(file_path).exists():
            raise ValueError(f"File does not exist: {file_path}")
        
        # Resolve output dir first
        user_dir = PathL("user_uploads") / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        
        # Normalize current_user to dict format
        if hasattr(current_user, '__dict__'):
            current_user_dict = {
                "id": getattr(current_user, 'id', 'unknown'),
                "email": getattr(current_user, 'email', 'unknown'),
                "subscription": getattr(current_user, 'subscription', 'unknown')
            }
        else:
            current_user_dict = current_user
        
        # Run main forecast logic (model fitting)
        results = do_forecast(
            user_id=user_id,
            current_user=current_user_dict,
            file_path=file_path,
            target_column=target_column,
            drop_columns=drop_columns,
            periods=periods,
            compare_models=compare_models
        )
        
        # Pass results + output_dir to subprocess
        subprocess_success = run_forecast_subprocess(
            user_id=user_id,
            file_path=file_path,
            forecast_result=results,
            output_dir=user_dir
        )
        
        if not subprocess_success:
            print(f"⚠️ Warning: Visualization subprocess failed for user {user_id}")
        
        # Load back visualizations JSON if available
        viz_file = PathL("data/visualizations") / f"{user_id}.json"
        if viz_file.exists():
            try:
                with viz_file.open("r", encoding="utf-8") as f:
                    visualization_data = json.load(f)
                
                # Get the latest forecast entry
                forecast_entries = [v for v in visualization_data if v.get("type") == "forecast"]
                if forecast_entries:
                    results["visualizations"] = forecast_entries[-1].get("visualizations", {})
                    results["thumbnailData"] = forecast_entries[-1].get("thumbnailData", "")
                    results["imageData"] = forecast_entries[-1].get("imageData", "")
            except Exception as e:
                print(f"Warning: Could not load visualization data: {e}")
        
        return {
            "status": "success",
            "result": results
        }
    
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error in forecast task: {error_msg}")
        traceback.print_exc()
        
        # Retry logic for transient errors
        if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
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
    feature_changes: str = ""
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
        feature_changes=feature_changes
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
@celery_app.task
def run_forecast(user_id: str, current_user: dict, file_path: str, target_column: str = "value", drop_columns: str = "", periods: int = 24):
    try:
        # Resolve output dir first
        user_dir = PathL("user_uploads") / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)

        # Run main forecast logic (model fitting)
        results = do_forecast(
            user_id=user_id,
            current_user=current_user,
            file_path=file_path,
            target_column=target_column,
            drop_columns=drop_columns,
            periods=periods
        )

        # Pass results + output_dir to subprocess
        run_forecast_subprocess(
            user_id=user_id,
            file_path=file_path,
            forecast_result=results,
            output_dir=user_dir  # ← add output dir here
        )

        # Load back visualizations JSON if available
        viz_file = PathL("data/visualizations") / f"{user_id}.json"
        if viz_file.exists():
            with viz_file.open("r", encoding="utf-8") as f:
                visualization_data = json.load(f)

            # Get the latest forecast entry
            forecast_entries = [v for v in visualization_data if v.get("type") == "forecast"]
            if forecast_entries:
                results["visualizations"] = forecast_entries[-1].get("visualizations", {})
                results["thumbnailData"] = forecast_entries[-1].get("thumbnailData", "")
                results["imageData"] = forecast_entries[-1].get("imageData", "")

        return {
            "status": "success",
            "result": results
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e)
        }

