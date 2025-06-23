import json
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def main():
    import sys
    if len(sys.argv) != 2:
        raise ValueError("Expected one argument: path to request.json")

    request_path = Path(sys.argv[1])
    with open(request_path, "r") as f:
        req = json.load(f)

    user_id = req["user_id"]
    file_path = req["file_path"]
    forecast_result = req["forecast_result"]

    # Extract back data
    series = pd.Series(forecast_result["series"])
    p = forecast_result["model_order"]["p"]
    d = forecast_result["model_order"]["d"]
    q = forecast_result["model_order"]["q"]
    periods = forecast_result["forecast_periods"]
    fc_vals = np.array(forecast_result["forecast_values"])
    conf_vals_lower = np.array(forecast_result["confidence_intervals"]["lower"])
    conf_vals_upper = np.array(forecast_result["confidence_intervals"]["upper"])
    idx = np.arange(len(series), len(series) + periods)

    # ACF + PACF Plot
    acf_pacf_b64 = None
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(series.diff(d).dropna(), ax=axes[0])
        plot_pacf(series.diff(d).dropna(), ax=axes[1])

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        acf_pacf_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
    except Exception as e:
        print("ACF/PACF generation failed", e)

    # Forecast Plot
    forecast_b64 = None
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(series, label="Historical Data")
        ax.plot(idx, fc_vals, label="Forecast", color="red")
        ax.fill_between(idx, conf_vals_lower, conf_vals_upper, color="red", alpha=0.2)
        ax.legend()
        ax.set_title(f"Time Series Forecast - ARIMA({p},{d},{q})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        forecast_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
    except Exception as e:
        print("Forecast plot generation failed", e)

    # Save result like shap_runner.py style
    try:
        output_dir = Path("data/visualizations") / str(user_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / "result.json"
        result_data = {}
        if forecast_b64:
            result_data["forecast"] = f"data:image/png;base64,{forecast_b64}"

        if acf_pacf_b64:
            result_data["acf_pacf"] = f"data:image/png;base64,{acf_pacf_b64}"


        with result_path.open("w") as f:
            json.dump(result_data, f, indent=2)

    except Exception as e:
        print("Saving result.json failed:", e)

if __name__ == "__main__":
    main()
