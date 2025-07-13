# shap_runner.py

import json
import joblib
import pandas as pd
import base64
import sys
from pathlib import Path
from datetime import datetime
import uuid
import matplotlib
matplotlib.use('Agg')  # Forces non-GUI backend

print("🔥 shap_runner.py started execution")  # <- must appear BEFORE main()

def main():
    print("[SHAP INIT] shap_runner.py file loaded")

    try:
        if len(sys.argv) != 2:
            raise ValueError("Expected one argument: path to request.json")

        request_path = Path(sys.argv[1])
        print(f"[SHAP DEBUG] Reading request from: {request_path}")
        print(f"[SHAP DEBUG] Request path exists: {request_path.exists()}")
        
        with request_path.open("r") as f:
            req = json.load(f)
        print(f"[SHAP DEBUG] Request loaded successfully: {req}")

        model_path = Path(req["model_path"]).resolve()
        data_path = Path(req["data_path"]).resolve()
        output_dir = Path(req["output_dir"]).resolve()
        user_id = req["user_id"]
        model_type = req.get("model_type", "regression")  # Fixed: was "classifier"
        target_column = req.get("target_column", "target")
        save_filename = req.get("save_filename", f"{user_id}_feature_importance.png")

        print(f"[SHAP DEBUG] Model path: {model_path}, exists: {model_path.exists()}")
        print(f"[SHAP DEBUG] Data path: {data_path}, exists: {data_path.exists()}")
        print(f"[SHAP DEBUG] Output dir: {output_dir}, exists: {output_dir.exists()}")
        print(f"[SHAP DEBUG] User ID: {user_id}")
        print(f"[SHAP DEBUG] Model type: {model_type}")
        print(f"[SHAP DEBUG] Target column: {target_column}")

        # Load full dataframe
        print(f"[SHAP DEBUG] Loading CSV from {data_path}")
        try:
            full_df = pd.read_csv(data_path)
            print(f"[SHAP DEBUG] CSV loaded successfully, shape: {full_df.shape}")
            print(f"[SHAP DEBUG] CSV columns: {list(full_df.columns)}")
        except Exception as e:
            print(f"[SHAP ERROR] Failed to load CSV: {e}")
            raise

        # Load training columns
        training_columns_path = output_dir / "training_columns.json"
        print(f"[SHAP DEBUG] Loading training columns from: {training_columns_path}")
        print(f"[SHAP DEBUG] Training columns path exists: {training_columns_path.exists()}")
        
        if not training_columns_path.exists():
            print(f"[SHAP ERROR] Training columns file not found: {training_columns_path}")
            raise FileNotFoundError(f"Training columns file not found: {training_columns_path}")

        try:
            with training_columns_path.open("r") as f:
                training_columns = json.load(f)
            print(f"[SHAP DEBUG] Training columns loaded successfully: {len(training_columns)} columns")
            print(f"[SHAP DEBUG] Training columns: {training_columns}")
        except Exception as e:
            print(f"[SHAP ERROR] Failed to load training columns: {e}")
            raise

        try:
            X = full_df[training_columns]
            print(f"[SHAP DEBUG] X prepared successfully, shape: {X.shape}")
        except Exception as e:
            print(f"[SHAP ERROR] Failed to prepare X from training columns: {e}")
            print(f"[SHAP DEBUG] Available columns in full_df: {list(full_df.columns)}")
            print(f"[SHAP DEBUG] Missing columns: {set(training_columns) - set(full_df.columns)}")
            raise

        print(f"[SHAP DEBUG] Loading model from {model_path}")
        try:
            model = joblib.load(model_path)
            print(f"[SHAP DEBUG] Model loaded successfully")
            print(f"[SHAP DEBUG] Model type: {type(model)}")
        except Exception as e:
            print(f"[SHAP ERROR] Failed to load model: {e}")
            raise

        # ───── Generate Feature Importance ─────
        print(f"[SHAP DEBUG] Starting safe_generate_feature_importance")
        print(f"[SHAP DEBUG] Parameters: model_type={model_type}, output_dir={output_dir}, save_filename={save_filename}")
        
        try:
            # Import the function from the same directory
            from feature_importance import safe_generate_feature_importance
            
            result = safe_generate_feature_importance(
                model=model,
                X=X,
                model_type=model_type,
                output_dir=output_dir,
                save_filename=save_filename,
                return_base64=True
            )
            print(f"[SHAP DEBUG] Feature importance completed successfully")
            print(f"[SHAP DEBUG] Result keys: {list(result.keys())}")
            
            # Check if we got the expected results
            if 'shap_bar' in result:
                print(f"[SHAP DEBUG] shap_bar present: {result['shap_bar'] is not None}")
            if 'shap_dot' in result:
                print(f"[SHAP DEBUG] shap_dot present: {result['shap_dot'] is not None}")
            if 'imp_df' in result:
                print(f"[SHAP DEBUG] imp_df present: {result['imp_df'] is not None}")
                if result['imp_df'] is not None:
                    print(f"[SHAP DEBUG] imp_df shape: {result['imp_df'].shape}")
                    
        except Exception as e:
            print(f"[SHAP ERROR] Feature importance failed: {e}")
            import traceback
            traceback.print_exc()
            # Set default values to continue
            result = {"shap_bar": None, "shap_dot": None, "imp_df": pd.DataFrame()}

        shap_bar = result.get("shap_bar")
        shap_dot = result.get("shap_dot")
        imp_df = result.get("imp_df", pd.DataFrame())

        print(f"[SHAP DEBUG] Extracted results - shap_bar: {shap_bar is not None}, shap_dot: {shap_dot is not None}, imp_df shape: {imp_df.shape}")

        # ───── Save result for use by parent process ─────
        result_json_path = output_dir / "result.json"
        print(f"[SHAP DEBUG] Saving result JSON to: {result_json_path}")
        
        result_data = {
            "shap_bar": shap_bar,
            "shap_dot": shap_dot,
            "imp_df": imp_df.to_dict(orient="records") if not imp_df.empty else []
        }
        
        with result_json_path.open("w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2)
        print(f"[SHAP DEBUG] Result JSON saved successfully")

        print(f"[SHAP DEBUG] SHAP runner completed successfully")

    except Exception as e:
        print(f"[SHAP ERROR] SHAP runner failed with exception: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error result to avoid parent process hanging
        try:
            result_json_path = output_dir / "result.json"
            error_result = {
                "shap_bar": None,
                "shap_dot": None,
                "imp_df": [],
                "error": str(e)
            }
            with result_json_path.open("w", encoding="utf-8") as f:
                json.dump(error_result, f, indent=2)
            print(f"[SHAP DEBUG] Error result saved to {result_json_path}")
        except:
            pass
        
        raise


if __name__ == "__main__":
    main()