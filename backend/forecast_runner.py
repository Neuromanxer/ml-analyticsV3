import json
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import datetime
import uuid


# from .auth import master_db_cm, _append_limited_metadata


from auth import master_db_cm, _append_limited_metadata


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

def main():
    import sys
    if len(sys.argv) != 2:
        raise ValueError("Expected one argument: path to request.json")

    request_path = Path(sys.argv[1])
    with open(request_path, "r", encoding="utf-8") as f:
        req = json.load(f)

    user_id = req["user_id"]
    file_path = req["file_path"]
    forecast_result = req["forecast_result"]
    output_dir = Path(req["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    series = pd.Series(forecast_result["series"])
    periods = forecast_result["forecast_periods"]
    
    # Convert single model result to expected format
    model_results = forecast_result.get("model_results", {})
    if not model_results:
        # Convert single result to multi-model format
        model_order = forecast_result.get("model_order", {"p": 1, "d": 1, "q": 1})
        model_name = f"ARIMA({model_order['p']},{model_order['d']},{model_order['q']})"
        model_results = {
            model_name: {
                "forecast": forecast_result["forecast_values"],
                "conf_int": forecast_result["confidence_intervals"],
                "aic": forecast_result.get("model_aic", 0),
                "order": f"({model_order['p']},{model_order['d']},{model_order['q']})"
            }
        }

    # ACF + PACF Plot
    acf_pacf_b64 = None
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(series.diff().dropna(), ax=axes[0])
        plot_pacf(series.diff().dropna(), ax=axes[1])

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        acf_pacf_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
    except Exception as e:
        print("ACF/PACF generation failed", e)

    # Forecast Comparison Plot
    forecast_b64 = None
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(series, label="Historical")
        colors = ["red", "green", "orange", "blue", "purple"]
        idx = np.arange(len(series), len(series) + periods)

        for i, (label, result) in enumerate(model_results.items()):
            forecast = np.array(result["forecast"])
            lower = np.array(result["conf_int"]["lower"])
            upper = np.array(result["conf_int"]["upper"])
            ax.plot(idx, forecast, label=f"{label} Forecast", color=colors[i % len(colors)])
            ax.fill_between(idx, lower, upper, color=colors[i % len(colors)], alpha=0.2)

        ax.legend()
        ax.set_title("Forecast Comparison")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        forecast_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
    except Exception as e:
        print("Forecast plot generation failed", e)

    # Save result
    try:
        meta_dir = Path("data") / "visualizations"
        meta_dir.mkdir(parents=True, exist_ok=True)
        safe_user_id = str(user_id).replace('/', '_').replace('\\', '_')
        output_path = meta_dir / f"{safe_user_id}.json"

        if output_path.exists():
            with output_path.open("r", encoding="utf-8") as f:
                content = json.load(f)
                meta_list = content if isinstance(content, list) else []
        else:
            meta_list = []

        entry = {
            "id": str(uuid.uuid4()),
            "created_at": datetime.utcnow().isoformat(),
            "type": "forecast",
            "dataset": Path(file_path).stem if file_path else "unknown",
            "parameters": {
                "forecast_periods": periods,
                "models": list(model_results.keys())
            },
            "model_info": {
                label: {"aic": model_results[label].get("aic"), "order": model_results[label].get("order", "N/A")}
                for label in model_results
            },
            "thumbnailData": f"data:image/png;base64,{forecast_b64}" if forecast_b64 else "",
            "imageData": f"data:image/png;base64,{forecast_b64}" if forecast_b64 else "",
            "visualizations": {
                "forecast": f"data:image/png;base64,{forecast_b64}" if forecast_b64 else "",
                "acf_pacf": f"data:image/png;base64,{acf_pacf_b64}" if acf_pacf_b64 else ""
            }
        }

        try:
            # ───────────── Save metadata to Supabase ─────────────
            entry = ensure_json_serializable(entry)
            with master_db_cm() as db:
                _append_limited_metadata(
                    user_id,
                    entry,
                    db=db,
                    max_entries=5
                )
            print(f"[✅] Forecast metadata saved for user {user_id}")

        except Exception as meta_error:
            print(f"[⚠️] Metadata save error: {meta_error}", file=sys.stderr)
            traceback.print_exc()

        # ───────────── Write out the result.json ─────────────
        result_json_path = output_dir / "result.json"
        try:
            with result_json_path.open("w", encoding="utf-8") as f:
                json.dump({
                    "forecast_comparison": forecast_b64,
                    "acf_pacf": acf_pacf_b64,
                    "model_info": entry.get("model_info", {})
                }, f, indent=2, ensure_ascii=False)

            print(f"✅ Forecast visualizations and result.json saved to {output_dir}")

        except Exception as e:
            print(f"[❌] Failed to write result.json: {e}", file=sys.stderr)
            traceback.print_exc()
    except Exception as e:
        print("Saving visualization JSON failed:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()