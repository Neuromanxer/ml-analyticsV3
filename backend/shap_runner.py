# shap_runner.py

import json
import sys
import os
from pathlib import Path
from datetime import datetime
import traceback

# Force immediate output flushing
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def flush_print(*args, **kwargs):
    """Print with immediate flush to ensure output appears in logs"""
    print(*args, **kwargs)
    sys.stdout.flush()
    sys.stderr.flush()

def main():
    flush_print("🔥 SHAP RUNNER: Starting execution")
    flush_print(f"🔥 SHAP RUNNER: Python version: {sys.version}")
    flush_print(f"🔥 SHAP RUNNER: Current working directory: {os.getcwd()}")
    flush_print(f"🔥 SHAP RUNNER: Script arguments: {sys.argv}")
    
    try:
        # Basic argument validation
        if len(sys.argv) != 2:
            raise ValueError(f"Expected one argument, got {len(sys.argv)-1}: {sys.argv[1:]}")

        request_path = Path(sys.argv[1])
        flush_print(f"[SHAP DEBUG] Request path: {request_path}")
        flush_print(f"[SHAP DEBUG] Request path exists: {request_path.exists()}")
        flush_print(f"[SHAP DEBUG] Request path is file: {request_path.is_file()}")
        
        # Try to read the request file
        try:
            with request_path.open("r") as f:
                req = json.load(f)
            flush_print(f"[SHAP DEBUG] Request loaded successfully")
            flush_print(f"[SHAP DEBUG] Request keys: {list(req.keys())}")
        except Exception as e:
            flush_print(f"[SHAP ERROR] Failed to read request file: {e}")
            raise

        # Extract parameters
        model_path = Path(req["model_path"]).resolve()
        data_path = Path(req["data_path"]).resolve()
        output_dir = Path(req["output_dir"]).resolve()
        user_id = req["user_id"]
        model_type = req.get("model_type", "regression")
        target_column = req.get("target_column", "target")
        save_filename = req.get("save_filename", f"{user_id}_feature_importance.png")

        flush_print(f"[SHAP DEBUG] Model path: {model_path} (exists: {model_path.exists()})")
        flush_print(f"[SHAP DEBUG] Data path: {data_path} (exists: {data_path.exists()})")
        flush_print(f"[SHAP DEBUG] Output dir: {output_dir} (exists: {output_dir.exists()})")
        flush_print(f"[SHAP DEBUG] User ID: {user_id}")
        flush_print(f"[SHAP DEBUG] Model type: {model_type}")

        # Check if required files exist
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        if not output_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {output_dir}")

        # Try to import required modules
        flush_print("[SHAP DEBUG] Importing required modules...")
        try:
            import pandas as pd
            flush_print("[SHAP DEBUG] ✓ pandas imported")
            
            import joblib
            flush_print("[SHAP DEBUG] ✓ joblib imported")
            
            import matplotlib
            matplotlib.use('Agg')
            flush_print("[SHAP DEBUG] ✓ matplotlib imported (Agg backend)")
            
            # Try to import feature_importance
            try:
                from feature_importance import safe_generate_feature_importance
                flush_print("[SHAP DEBUG] ✓ feature_importance imported")
            except ImportError as e:
                flush_print(f"[SHAP ERROR] Failed to import feature_importance: {e}")
                flush_print(f"[SHAP DEBUG] Current directory contents: {list(Path('.').iterdir())}")
                raise
                
        except Exception as e:
            flush_print(f"[SHAP ERROR] Failed to import required modules: {e}")
            raise

        # Load data
        flush_print("[SHAP DEBUG] Loading data...")
        try:
            full_df = pd.read_csv(data_path)
            flush_print(f"[SHAP DEBUG] Data loaded successfully, shape: {full_df.shape}")
            flush_print(f"[SHAP DEBUG] Data columns: {list(full_df.columns)}")
        except Exception as e:
            flush_print(f"[SHAP ERROR] Failed to load data: {e}")
            raise

        # Load training columns
        training_columns_path = output_dir / "training_columns.json"
        flush_print(f"[SHAP DEBUG] Loading training columns from: {training_columns_path}")
        
        if not training_columns_path.exists():
            raise FileNotFoundError(f"Training columns file not found: {training_columns_path}")

        try:
            with training_columns_path.open("r") as f:
                training_columns = json.load(f)
            flush_print(f"[SHAP DEBUG] Training columns loaded: {len(training_columns)} columns")
        except Exception as e:
            flush_print(f"[SHAP ERROR] Failed to load training columns: {e}")
            raise

        # Prepare features
        try:
            X = full_df[training_columns]
            flush_print(f"[SHAP DEBUG] Features prepared, shape: {X.shape}")
        except Exception as e:
            flush_print(f"[SHAP ERROR] Failed to prepare features: {e}")
            flush_print(f"[SHAP DEBUG] Available columns: {list(full_df.columns)}")
            flush_print(f"[SHAP DEBUG] Missing columns: {set(training_columns) - set(full_df.columns)}")
            raise

        # Load model
        flush_print("[SHAP DEBUG] Loading model...")
        try:
            model = joblib.load(model_path)
            flush_print(f"[SHAP DEBUG] Model loaded successfully, type: {type(model)}")
        except Exception as e:
            flush_print(f"[SHAP ERROR] Failed to load model: {e}")
            raise

        # Generate feature importance
        flush_print("[SHAP DEBUG] Starting feature importance generation...")
        try:
            result = safe_generate_feature_importance(
                model=model,
                X=X,
                model_type=model_type,
                output_dir=output_dir,
                save_filename=save_filename,
                return_base64=True
            )
            flush_print(f"[SHAP DEBUG] Feature importance completed successfully")
            flush_print(f"[SHAP DEBUG] Result keys: {list(result.keys())}")
            
        except Exception as e:
            flush_print(f"[SHAP ERROR] Feature importance generation failed: {e}")
            traceback.print_exc()
            # Set default values
            result = {"shap_bar": None, "shap_dot": None, "imp_df": pd.DataFrame()}

        # Extract results
        shap_bar = result.get("shap_bar")
        shap_dot = result.get("shap_dot")
        imp_df = result.get("imp_df", pd.DataFrame())

        flush_print(f"[SHAP DEBUG] Results extracted - bar: {shap_bar is not None}, dot: {shap_dot is not None}, df shape: {imp_df.shape}")

        # Save results
        result_json_path = output_dir / "result.json"
        flush_print(f"[SHAP DEBUG] Saving results to: {result_json_path}")
        
        try:
            result_data = {
                "shap_bar": shap_bar,
                "shap_dot": shap_dot,
                "imp_df": imp_df.to_dict(orient="records") if not imp_df.empty else []
            }
            
            with result_json_path.open("w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2)
            flush_print(f"[SHAP DEBUG] Results saved successfully")
            
        except Exception as e:
            flush_print(f"[SHAP ERROR] Failed to save results: {e}")
            raise

        flush_print("🎉 SHAP RUNNER: Completed successfully!")
        
    except Exception as e:
        flush_print(f"💥 SHAP RUNNER: Fatal error: {e}")
        traceback.print_exc()
        
        # Try to save error result
        try:
            if 'output_dir' in locals():
                result_json_path = output_dir / "result.json"
                error_result = {
                    "shap_bar": None,
                    "shap_dot": None,
                    "imp_df": [],
                    "error": str(e)
                }
                with result_json_path.open("w", encoding="utf-8") as f:
                    json.dump(error_result, f, indent=2)
                flush_print(f"[SHAP DEBUG] Error result saved to {result_json_path}")
        except:
            flush_print("[SHAP ERROR] Failed to save error result")
        
        sys.exit(1)

if __name__ == "__main__":
    main()