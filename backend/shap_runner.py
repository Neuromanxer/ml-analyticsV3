# shap_runner.py

import json
import joblib
import pandas as pd
import base64
from pathlib import Path
from datetime import datetime
import uuid
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
    target_column = req.get("target_column", "target")
    save_filename = req.get("save_filename", f"{user_id}_feature_importance.png")

    # Load full dataframe
    full_df = pd.read_csv(data_path)

    # Load training columns
    training_columns_path = output_dir / "training_columns.json"
    if not training_columns_path.exists():
        raise FileNotFoundError(f"Training columns file not found: {training_columns_path}")

    with training_columns_path.open("r") as f:
        training_columns = json.load(f)

    X = full_df[training_columns]
    model = joblib.load(model_path)

    # ───── Generate Feature Importance ─────
    result = safe_generate_feature_importance(
        model=model,
        X=X,
        model_type=model_type,
        output_dir=output_dir,
        save_filename=save_filename,
        return_base64=True
    )

    shap_bar = result.get("shap_bar")
    shap_dot = result.get("shap_dot")
    imp_df = result.get("imp_df", pd.DataFrame())

    # ───── Save structured metadata entry ─────
    try:
        meta_dir = Path("data") / "visualizations"
        meta_dir.mkdir(parents=True, exist_ok=True)

        safe_user_id = str(user_id).replace('/', '_').replace('\\', '_')
        output_path = meta_dir / f"{safe_user_id}.json"
        if output_path.exists():
            try:
                with output_path.open("r") as f:
                    content = json.load(f)
                # Force it to be a list if it’s a dict (backward compatibility)
                if isinstance(content, list):
                    meta_list = content
                else:
                    print(f"⚠️ Found non-list content in {output_path}. Starting fresh.")
                    meta_list = []
            except json.JSONDecodeError:
                print(f"⚠️ Corrupted JSON in {output_path}. Starting fresh.")
                meta_list = []
        else:
            meta_list = []


        entry = {
            "id": str(uuid.uuid4()),
            "created_at": datetime.utcnow().isoformat(),
            "type": "classification",
            "dataset": data_path.stem,
            "target_column": target_column,
            "parameters": {"model_type": model_type},
            "thumbnailData": f"data:image/png;base64,{shap_bar or shap_dot or ''}",
            "imageData": f"data:image/png;base64,{shap_dot or shap_bar or ''}",
            "visualizations": {
                "shap_bar": f"data:image/png;base64,{shap_bar}" if shap_bar else "",
                "shap_dot": f"data:image/png;base64,{shap_dot}" if shap_dot else ""
            },
            "top_features": imp_df.head(10).to_dict("records") if not imp_df.empty else []
        }
        # Build visualizations dictionary conditionally
        if shap_bar:
            entry["visualizations"]["shap_bar"] = f"data:image/png;base64,{shap_bar}"
            entry["thumbnailData"] = f"data:image/png;base64,{shap_bar}"  # prioritize shap_bar

        if shap_dot:
            entry["visualizations"]["shap_dot"] = f"data:image/png;base64,{shap_dot}"
            # If shap_bar not used for thumbnail, fallback to shap_dot
            if not shap_bar:
                entry["thumbnailData"] = f"data:image/png;base64,{shap_dot}"

        # imageData prioritizes shap_dot over shap_bar
        if shap_dot:
            entry["imageData"] = f"data:image/png;base64,{shap_dot}"
        elif shap_bar:
            entry["imageData"] = f"data:image/png;base64,{shap_bar}"
        meta_list.append(entry)
        with output_path.open("w") as f:
            json.dump(meta_list, f, indent=2)

        print(f"✅ Saved SHAP metadata for user {user_id}. Total entries: {len(meta_list)}")
        # Save result for use by parent process
        result_json_path = output_dir / "result.json"
        with result_json_path.open("w",  encoding="utf-8") as f:
            json.dump({
                "shap_bar": shap_bar,
                "shap_dot": shap_dot,
                "imp_df": imp_df.to_dict(orient="records")
            }, f, indent=2)

    except Exception as e:
        print("❌ Failed to save SHAP visualization metadata:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
