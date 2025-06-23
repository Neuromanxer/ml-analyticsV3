# shap_runner.py

import json
import joblib
import pandas as pd
from pathlib import Path
from feature_importance import safe_generate_feature_importance

def main():
    import sys
    if len(sys.argv) != 2:
        raise ValueError("Expected one argument: path to request.json")

    request_path = Path(sys.argv[1])
    with request_path.open("r") as f:
        req = json.load(f)

    model_path = Path(req["model_path"]).resolve()
    data_path = Path(req["data_path"]).resolve()
    output_dir = Path(req["output_dir"]).resolve()
    user_id = req["user_id"]
    model_type = req.get("model_type", "classifier")
    save_filename = req.get("save_filename", f"{user_id}_feature_importance.png")

    # Load full dataframe
    full_df = pd.read_csv(data_path)

    # Load training columns from training time
    training_columns_path = output_dir / "training_columns.json"
    if not training_columns_path.exists():
        raise FileNotFoundError(f"Training columns file not found: {training_columns_path}")

    with open(training_columns_path, "r") as f:
        training_columns = json.load(f)

    # Subset to exact training columns
    X = full_df[training_columns]

    # Load trained model
    model = joblib.load(model_path)

    # Generate feature importance
    result = safe_generate_feature_importance(
        model=model,
        X=X,
        model_type=model_type,
        output_dir=output_dir,
        save_filename=save_filename,
        return_base64=True
    )

    # Save result
    result_path = output_dir / "result.json"
    with result_path.open("w") as f:
        json.dump({
            "shap_bar": result.get("shap_bar"),
            "shap_dot": result.get("shap_dot"),
            "imp_df": result.get("imp_df").to_dict("records")
        }, f)

if __name__ == "__main__":
    main()
