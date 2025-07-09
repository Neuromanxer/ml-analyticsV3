# prediction_viz_runner.py

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import sys
from pathlib import Path

def to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def main(request_path):
    with open(request_path, "r") as f:
        request = json.load(f)

    user_dir = Path(request["output_dir"])
    data_path = Path(request["data_path"])
    prediction_column = request.get("prediction_column", "prediction")
    output_dir = Path(request["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)  # make sure it exists

    df = pd.read_csv(data_path)

    result = {}
    
    # 1. Distribution Plot
    fig1 = plt.figure()
    sns.histplot(df[prediction_column], kde=True)
    plt.title("Prediction Distribution")
    result["prediction_distribution"] = to_base64(fig1)
    plt.close(fig1)

    # 2. Feature vs Prediction for top N features (by variance or first N columns)
    feature_cols = [col for col in df.columns if col != prediction_column][:3]
    feature_viz = {}

    for col in feature_cols:
        fig = plt.figure()
        sns.scatterplot(data=df, x=col, y=prediction_column)
        plt.title(f"{col} vs Prediction")
        feature_viz[col] = to_base64(fig)
        plt.close(fig)

    result["feature_relationships"] = feature_viz

    with open(user_dir / "prediction_result.json", "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prediction_viz_runner.py <request.json>")
        sys.exit(1)

    main(sys.argv[1])
