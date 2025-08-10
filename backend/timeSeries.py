import os
import sys
import json
import uuid
import tempfile
import subprocess
import traceback
import io
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set matplotlib backend for headless environments
plt.switch_backend('Agg')
class ScenarioManager:
    """Manages different forecasting scenarios and models"""
    def __init__(self):
        self.scenarios = {}

    def add_scenario(self, name: str, description: str, models: List):
        self.scenarios[name] = {'description': description, 'models': models}

    def run_scenario(self, scenario_name: str, series: pd.Series, periods: int, **context) -> Dict:
        """
        Run all models in a scenario.
        Extra params (e.g., freq='D', enable_backtesting=True) are passed to models.
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")

        results = {}
        for model in self.scenarios[scenario_name]['models']:
            model_name = model.__class__.__name__
            try:
                # Forward context to models; they can accept what they need, ignore the rest
                result = model.fit_predict(series, periods, **context)
                results[model_name] = result
            except TypeError:
                # Backward-compatible: if model doesn't accept kwargs, call the old way
                result = model.fit_predict(series, periods)
                results[model_name] = result
            except Exception as e:
                print(f"Model {model_name} failed: {e}")
                results[model_name] = {
                    "status": "error", "error": str(e),
                    "predictions": [], "confidence_intervals": []
                }
        return results

    def compare_models(self, series: pd.Series, **context) -> Dict:
        # Placeholder for your backtesting logic; **context available if needed
        return {}

class ARIMAModel:
    def __init__(self, auto_arima: bool = True, order: tuple = (1, 1, 1), model_kwargs: dict | None = None):
        self.auto_arima = auto_arima
        self.order = order
        self.model_kwargs = model_kwargs or {}

    def fit_predict(self, series: pd.Series, periods: int, **kwargs) -> Dict:  # <— accepts extras
        from statsmodels.tsa.arima.model import ARIMA
        try:
            order = (1,1,1) if self.auto_arima else self.order
            model = ARIMA(series, order=order, **self.model_kwargs)
            fitted = model.fit()
            fr = fitted.get_forecast(steps=periods)
            return {
                "status": "success",
                "predictions": fr.predicted_mean.astype(float).tolist(),
                "confidence_intervals": fr.conf_int().to_numpy(dtype=float).tolist(),
                "model_info": {"aic": float(fitted.aic), "bic": float(fitted.bic)}
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "predictions": [], "confidence_intervals": []}
class SARIMAModel:
    def __init__(self, order: tuple = (1, 1, 1), seasonal_order: tuple = (1, 1, 1, 12), model_kwargs: dict | None = None):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_kwargs = model_kwargs or {}

    def fit_predict(self, series: pd.Series, periods: int, **kwargs) -> Dict:  # <— accepts extras
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        try:
            model = SARIMAX(series, order=self.order, seasonal_order=self.seasonal_order, **self.model_kwargs)
            fitted = model.fit(disp=False)
            fr = fitted.get_forecast(steps=periods)
            return {
                "status": "success",
                "predictions": fr.predicted_mean.astype(float).tolist(),
                "confidence_intervals": fr.conf_int().to_numpy(dtype=float).tolist(),
                "model_info": {"aic": float(fitted.aic), "bic": float(fitted.bic)}
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "predictions": [], "confidence_intervals": []}

class ExponentialSmoothingModel:
    def fit_predict(self, series: pd.Series, periods: int, **kwargs) -> Dict:  # <— accepts extras
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        try:
            model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12)
            fitted = model.fit()
            fc = fitted.forecast(steps=periods)
            return {"status": "success", "predictions": fc.astype(float).tolist(),
                    "confidence_intervals": [], "model_info": {}}
        except Exception as e:
            return {"status": "error", "error": str(e), "predictions": [], "confidence_intervals": []}

class LSTMModel:
    """LSTM Model wrapper (placeholder)"""
    
    def __init__(self, sequence_length: int = 10, epochs: int = 50):
        self.sequence_length = sequence_length
        self.epochs = epochs
    
    def fit_predict(self, series: pd.Series, periods: int) -> Dict:
        """Fit LSTM model and make predictions"""
        try:
            # Placeholder implementation
            # You would need tensorflow/keras for actual LSTM implementation
            return {
                "status": "error",
                "error": "LSTM model not implemented - requires tensorflow/keras",
                "predictions": [],
                "confidence_intervals": []
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "predictions": [],
                "confidence_intervals": []
            }

class RandomForestModel:
    def __init__(self, n_estimators: int = 100, lags: int = 5):
        self.n_estimators = n_estimators
        self.lags = lags

    def fit_predict(self, series: pd.Series, periods: int, **kwargs) -> Dict:  # <— accepts extras
        from sklearn.ensemble import RandomForestRegressor
        try:
            X, y = [], []
            for i in range(self.lags, len(series)):
                X.append(series.iloc[i-self.lags:i].values)
                y.append(series.iloc[i])
            if not X:
                raise ValueError("Not enough data for lagged features")
            model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=42)
            model.fit(np.asarray(X), np.asarray(y))
            preds, last = [], series.tail(self.lags).values
            for _ in range(periods):
                p = float(model.predict([last])[0])
                preds.append(p)
                last = np.append(last[1:], p)
            return {"status": "success", "predictions": preds, "confidence_intervals": [], "model_info": {}}
        except Exception as e:
            return {"status": "error", "error": str(e), "predictions": [], "confidence_intervals": []}


def generate_scenario_visualizations(series: pd.Series, results: Dict, periods: int) -> Dict:
    """Generate visualizations for different scenarios"""
    try:
        visualizations = {}
        
        for scenario_name, scenario_results in results.items():
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot historical data
            ax.plot(range(len(series)), series.values, label="Historical", color="blue", linewidth=2)
            
            # Plot forecasts from different models
            colors = ['red', 'green', 'purple', 'orange', 'brown']
            forecast_start = len(series)
            
            for i, (model_name, result) in enumerate(scenario_results.items()):
                if result["status"] == "success" and result["predictions"]:
                    color = colors[i % len(colors)]
                    predictions = result["predictions"]
                    
                    ax.plot(
                        range(forecast_start, forecast_start + len(predictions)),
                        predictions,
                        label=f"{model_name} Forecast",
                        color=color,
                        linestyle="--",
                        linewidth=2
                    )
                    
                    # Add confidence intervals if available
                    if result["confidence_intervals"]:
                        conf_int = np.array(result["confidence_intervals"])
                        ax.fill_between(
                            range(forecast_start, forecast_start + len(predictions)),
                            conf_int[:, 0],
                            conf_int[:, 1],
                            color=color,
                            alpha=0.2
                        )
            
            ax.set_title(f"Forecast Comparison - {scenario_name.title()} Models", fontsize=16)
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format="png", bbox_inches='tight', dpi=150)
            buffer.seek(0)
            
            plot_b64 = base64.b64encode(buffer.read()).decode("utf-8")
            visualizations[scenario_name] = f"data:image/png;base64,{plot_b64}"
            
            plt.close(fig)
        
        return visualizations
        
    except Exception as e:
        print(f"Visualization generation failed: {e}")
        return {}


def main():
    """Enhanced main function with better error handling and structure processing"""
    import sys
    
    if len(sys.argv) != 2:
        raise ValueError("Expected one argument: path to request.json")

    request_path = Path(sys.argv[1])
    
    try:
        with open(request_path, "r", encoding="utf-8") as f:
            req = json.load(f)
    except Exception as e:
        print(f"Failed to load request file: {e}")
        return

    user_id = req["user_id"]
    file_path = req["file_path"]
    forecast_result = req["forecast_result"]
    output_dir = Path(req["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        series = pd.Series(forecast_result["series"])
        periods = forecast_result["parameters"]["periods"]
    except Exception as e:
        print(f"Failed to extract series data: {e}")
        return
    
    # Extract scenarios and models from the new format
    scenarios = forecast_result.get("scenarios", {})
    model_results = {}
    
    print(f"DEBUG: Found scenarios: {list(scenarios.keys())}")
    
    # Convert scenarios to model_results format for visualization
    for scenario_name, scenario_data in scenarios.items():
        print(f"DEBUG: Processing scenario '{scenario_name}' with keys: {list(scenario_data.keys())}")
        
        # Handle different possible structures
        models_data = None
        if "models" in scenario_data:
            models_data = scenario_data["models"]
        elif "results" in scenario_data:
            models_data = scenario_data["results"]
        else:
            # Maybe the scenario_data itself contains the models
            models_data = scenario_data
        
        if models_data:
            print(f"DEBUG: Found models in scenario '{scenario_name}': {list(models_data.keys())}")
            
            for model_name, model_data in models_data.items():
                print(f"DEBUG: Processing model '{model_name}' with status: {model_data.get('status', 'unknown')}")
                
                if model_data.get("status") == "success":
                    # Extract model info from the new format
                    predictions = model_data.get("predictions", [])
                    conf_int = model_data.get("confidence_intervals", [])
                    model_info = model_data.get("model_info", {})
                    
                    print(f"DEBUG: Model '{model_name}' has {len(predictions)} predictions")
                    
                    # Create combined model name with scenario
                    combined_name = f"{scenario_name}_{model_name}"
                    
                    # Convert confidence intervals to expected format
                    if conf_int and len(conf_int) > 0:
                        # Handle different confidence interval formats
                        try:
                            conf_int_array = np.array(conf_int)
                            if conf_int_array.ndim == 2 and conf_int_array.shape[1] == 2:
                                lower_bounds = conf_int_array[:, 0].tolist()
                                upper_bounds = conf_int_array[:, 1].tolist()
                            elif len(conf_int) == 2 and all(isinstance(x, list) for x in conf_int):
                                # Format: [[lower1, lower2, ...], [upper1, upper2, ...]]
                                lower_bounds = conf_int[0]
                                upper_bounds = conf_int[1]
                            else:
                                # Flat list, split in half
                                mid = len(conf_int) // 2
                                lower_bounds = conf_int[:mid]
                                upper_bounds = conf_int[mid:]
                        except Exception as e:
                            print(f"Warning: Could not parse confidence intervals for {model_name}: {e}")
                            # Create dummy confidence intervals
                            lower_bounds = [p * 0.95 for p in predictions]
                            upper_bounds = [p * 1.05 for p in predictions]
                    else:
                        # Create dummy confidence intervals if none provided
                        lower_bounds = [p * 0.95 for p in predictions]
                        upper_bounds = [p * 1.05 for p in predictions]
                    
                    model_results[combined_name] = {
                        "forecast": predictions,
                        "conf_int": {
                            "lower": lower_bounds,
                            "upper": upper_bounds
                        },
                        "aic": model_info.get("aic", 0),
                        "bic": model_info.get("bic", 0),
                        "order": model_info.get("order", "N/A"),
                        "scenario": scenario_name
                    }
                    
                    print(f"DEBUG: Successfully added model '{combined_name}' to results")
                else:
                    print(f"DEBUG: Skipping model '{model_name}' with status '{model_data.get('status', 'unknown')}'")
        else:
            print(f"DEBUG: No models found in scenario '{scenario_name}'")

    # Fallback for old single model format
    if not model_results and "forecast_values" in forecast_result:
        print("DEBUG: Using fallback for old single model format")
        model_order = forecast_result.get("model_order", {"p": 1, "d": 1, "q": 1})
        model_name = f"ARIMA({model_order['p']},{model_order['d']},{model_order['q']})"
        
        # Handle confidence intervals in old format
        conf_int = forecast_result.get("confidence_intervals", {})
        if isinstance(conf_int, dict):
            lower = conf_int.get("lower", [])
            upper = conf_int.get("upper", [])
        else:
            # Assume it's a list/array format
            try:
                conf_int_array = np.array(conf_int)
                if conf_int_array.ndim == 2:
                    lower = conf_int_array[:, 0].tolist()
                    upper = conf_int_array[:, 1].tolist()
                else:
                    lower = upper = []
            except:
                lower = upper = []
        
        model_results = {
            model_name: {
                "forecast": forecast_result["forecast_values"],
                "conf_int": {
                    "lower": lower,
                    "upper": upper
                },
                "aic": forecast_result.get("model_aic", 0),
                "bic": forecast_result.get("model_bic", 0),
                "order": f"({model_order['p']},{model_order['d']},{model_order['q']})",
                "scenario": "single_model"
            }
        }

    if not model_results:
        print("ERROR: No valid model results found")
        return

    # ACF + PACF Plot
    acf_pacf_b64 = None
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Use differenced series for ACF/PACF
        diff_series = series.diff().dropna()
        if len(diff_series) > 10:  # Need enough data points
            max_lags = min(20, len(diff_series)//4)
            plot_acf(diff_series, ax=axes[0], lags=max_lags)
            plot_pacf(diff_series, ax=axes[1], lags=max_lags)
        else:
            # Fallback for short series
            axes[0].text(0.5, 0.5, 'Insufficient data\nfor ACF plot', 
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[1].text(0.5, 0.5, 'Insufficient data\nfor PACF plot', 
                        ha='center', va='center', transform=axes[1].transAxes)
        
        axes[0].set_title('ACF of Differenced Series')
        axes[1].set_title('PACF of Differenced Series')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        acf_pacf_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
    except Exception as e:
        print(f"ACF/PACF generation failed: {e}")
        traceback.print_exc()

    # Forecast Comparison Plot
    forecast_b64 = None
    try:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot historical data
        ax.plot(range(len(series)), series.values, label="Historical Data", color="black", linewidth=2)
        
        # Define colors for different scenarios
        scenario_colors = {
            "traditional": ["#FF6B6B", "#FF8E8E", "#FFB1B1"],
            "machine_learning": ["#4ECDC4", "#7ED8D1", "#A8E6E0"],
            "comprehensive": ["#45B7D1", "#6BC5D8", "#8FD3E0", "#B3E1E8", "#D7EFF0"]
        }
        default_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
        
        # Forecast indices
        forecast_idx = np.arange(len(series), len(series) + periods)
        
        # Group models by scenario for better visualization
        scenarios_plotted = {}
        color_idx = 0
        
        for model_name, result in model_results.items():
            scenario = result.get("scenario", "unknown")
            
            # Get color for this scenario
            if scenario in scenario_colors:
                if scenario not in scenarios_plotted:
                    scenarios_plotted[scenario] = 0
                color = scenario_colors[scenario][scenarios_plotted[scenario] % len(scenario_colors[scenario])]
                scenarios_plotted[scenario] += 1
            else:
                color = default_colors[color_idx % len(default_colors)]
                color_idx += 1
            
            forecast = np.array(result["forecast"])
            conf_int = result["conf_int"]
            
            # Handle different confidence interval formats
            if isinstance(conf_int, dict):
                lower = np.array(conf_int.get("lower", []))
                upper = np.array(conf_int.get("upper", []))
            else:
                # Assume it's a list or array
                try:
                    conf_int_array = np.array(conf_int)
                    if conf_int_array.ndim == 2 and conf_int_array.shape[1] == 2:
                        lower = conf_int_array[:, 0]
                        upper = conf_int_array[:, 1]
                    else:
                        # Create dummy confidence intervals
                        lower = forecast * 0.95
                        upper = forecast * 1.05
                except:
                    lower = forecast * 0.95
                    upper = forecast * 1.05
            
            # Ensure arrays are the right length and handle empty arrays
            if len(forecast) == 0:
                print(f"Warning: No forecast data for model {model_name}")
                continue
                
            min_len = min(len(forecast), len(lower), len(upper), periods)
            if min_len == 0:
                print(f"Warning: No valid data for model {model_name}")
                continue
                
            forecast = forecast[:min_len]
            lower = lower[:min_len]
            upper = upper[:min_len]
            forecast_idx_adj = forecast_idx[:min_len]
            
            # Plot forecast
            aic_val = result.get('aic', 'N/A')
            if isinstance(aic_val, (int, float)) and aic_val != 0:
                label = f"{model_name} (AIC: {aic_val:.2f})"
            else:
                label = model_name
                
            ax.plot(forecast_idx_adj, forecast, label=label, color=color, linewidth=2)
            
            # Plot confidence intervals
            if len(lower) > 0 and len(upper) > 0:
                ax.fill_between(forecast_idx_adj, lower, upper, color=color, alpha=0.2)

        # Customize plot
        ax.set_title("Enhanced Forecast Comparison", fontsize=16, fontweight='bold')
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Only add legend if we have any plots
        if len(ax.lines) > 1:  # More than just historical data
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add vertical line to separate historical from forecast
        ax.axvline(x=len(series)-1, color='red', linestyle='--', alpha=0.7, label='Forecast Start')
        
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        forecast_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
    except Exception as e:
        print(f"Forecast plot generation failed: {e}")
        traceback.print_exc()

    # Model Performance Summary Plot (if multiple models)
    performance_b64 = None
    if len(model_results) > 1:
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            models = list(model_results.keys())
            aics = [model_results[m].get('aic', 0) for m in models]
            bics = [model_results[m].get('bic', 0) for m in models]
            
            # Filter out zero/invalid values for better visualization
            valid_models = []
            valid_aics = []
            valid_bics = []
            
            for i, (model, aic, bic) in enumerate(zip(models, aics, bics)):
                if (isinstance(aic, (int, float)) and aic != 0) or (isinstance(bic, (int, float)) and bic != 0):
                    valid_models.append(model.replace('_', ' '))  # Make names more readable
                    valid_aics.append(aic if isinstance(aic, (int, float)) and aic != 0 else None)
                    valid_bics.append(bic if isinstance(bic, (int, float)) and bic != 0 else None)
            
            if valid_models:
                x = np.arange(len(valid_models))
                width = 0.35
                
                # Plot AIC bars
                aic_values = [aic for aic in valid_aics if aic is not None]
                if aic_values:
                    aic_positions = [i for i, aic in enumerate(valid_aics) if aic is not None]
                    ax.bar([x[i] - width/2 for i in aic_positions], aic_values, 
                          width, label='AIC', alpha=0.8, color='skyblue')
                
                # Plot BIC bars
                bic_values = [bic for bic in valid_bics if bic is not None]
                if bic_values:
                    bic_positions = [i for i, bic in enumerate(valid_bics) if bic is not None]
                    ax.bar([x[i] + width/2 for i in bic_positions], bic_values, 
                          width, label='BIC', alpha=0.8, color='lightcoral')
                
                ax.set_xlabel('Models')
                ax.set_ylabel('Information Criterion')
                ax.set_title('Model Performance Comparison (Lower is Better)')
                ax.set_xticks(x)
                ax.set_xticklabels(valid_models, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                buf.seek(0)
                performance_b64 = base64.b64encode(buf.read()).decode("utf-8")
            
            plt.close(fig)
        except Exception as e:
            print(f"Performance plot generation failed: {e}")
            traceback.print_exc()

    # Save result
    try:
        meta_dir = Path("data") / "visualizations"
        meta_dir.mkdir(parents=True, exist_ok=True)
        safe_user_id = str(user_id).replace('/', '_').replace('\\', '_')
        output_path = meta_dir / f"{safe_user_id}.json"

        # Load existing data
        if output_path.exists():
            try:
                with output_path.open("r", encoding="utf-8") as f:
                    content = json.load(f)
                    meta_list = content if isinstance(content, list) else []
            except Exception as e:
                print(f"Warning: Could not load existing visualization data: {e}")
                meta_list = []
        else:
            meta_list = []

        # Create summary of all models
        model_summary = {}
        for model_name, result in model_results.items():
            model_summary[model_name] = {
                "aic": result.get("aic"),
                "bic": result.get("bic"), 
                "order": result.get("order", "N/A"),
                "scenario": result.get("scenario", "unknown"),
                "forecast_points": len(result.get("forecast", []))
            }

        entry = {
            "id": str(uuid.uuid4()),
            "created_at": datetime.utcnow().isoformat(),
            "type": "enhanced_forecast",
            "dataset": Path(file_path).stem if file_path else "unknown",
            "parameters": {
                "forecast_periods": periods,
                "models": list(model_results.keys()),
                "scenarios": list(scenarios.keys()) if scenarios else ["single_model"]
            },
            "model_info": model_summary,
            "thumbnailData": f"data:image/png;base64,{forecast_b64}" if forecast_b64 else "",
            "imageData": f"data:image/png;base64,{forecast_b64}" if forecast_b64 else "",
            "visualizations": {
                "forecast": f"data:image/png;base64,{forecast_b64}" if forecast_b64 else "",
                "acf_pacf": f"data:image/png;base64,{acf_pacf_b64}" if acf_pacf_b64 else "",
                "performance": f"data:image/png;base64,{performance_b64}" if performance_b64 else ""
            }
        }

        meta_list.append(entry)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(meta_list, f, indent=2, ensure_ascii=False)

        # Also save to result.json for immediate access
        result_json_path = output_dir / "result.json"
        with result_json_path.open("w", encoding="utf-8") as f:
            json.dump({
                "forecast_comparison": forecast_b64,
                "acf_pacf": acf_pacf_b64,
                "performance": performance_b64,
                "model_info": entry["model_info"],
                "success": True
            }, f, indent=2)

        print(f"[SUCCESS] Enhanced forecast visualizations saved to {output_path}")
        print(f"[INFO] Generated {len(model_results)} model visualizations across {len(scenarios)} scenarios")
        
    except Exception as e:
        print("Saving visualization JSON failed:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()