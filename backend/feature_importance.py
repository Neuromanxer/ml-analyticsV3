import os
import io
import base64
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from queue import Queue
from threading import Thread
import time
import traceback
import gc

import shap
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use("Agg")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def flush_print(*args, **kwargs):
    """Print with immediate flush to ensure output appears in logs"""
    print(*args, **kwargs)
    import sys
    sys.stdout.flush()
    sys.stderr.flush()

def _plot_feature_importance(model, X, model_type, output_dir, top_n, save_filename, return_base64, queue):
    """Main plotting function with enhanced error handling and progress tracking"""
    
    def plot_to_base64(fig):
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')  # Reduced DPI for performance
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            return img_str
        except Exception as e:
            flush_print(f"[PLOT ERROR] Failed to convert plot to base64: {e}")
            return None

    standard_result = None
    detailed_result = None
    feature_importance = pd.DataFrame()
    
    start_time = time.time()
    flush_print(f"[SHAP PROCESS] Starting feature importance generation at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Validate inputs
        if X is None or len(X) == 0:
            raise ValueError("Input data X is empty or None")
        
        if model is None:
            raise ValueError("Model is None")
            
        flush_print(f"[SHAP PROCESS] Input validation passed - X shape: {X.shape}, model type: {type(model)}")

        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, save_filename)
        else:
            save_path = save_filename

        # Prepare sample data - use smaller sample for faster processing
        sample_size = min(500, len(X))  # Reduced from 1000 to 500
        flush_print(f"[SHAP PROCESS] Using sample size: {sample_size} out of {len(X)} rows")
        
        if len(X) > sample_size:
            X_sample = X.sample(sample_size, random_state=42)
        else:
            X_sample = X.copy()
        
        # Ensure column names are strings
        X_sample.columns = [str(c) for c in X_sample.columns]
        flush_print(f"[SHAP PROCESS] Sample prepared, shape: {X_sample.shape}")

        # Try SHAP TreeExplainer
        flush_print("[SHAP PROCESS] Creating TreeExplainer...")
        try:
            explainer = shap.TreeExplainer(model)
            flush_print("[SHAP PROCESS] TreeExplainer created successfully")
        except Exception as e:
            flush_print(f"[SHAP ERROR] TreeExplainer failed: {e}")
            raise

        # Generate SHAP values
        flush_print("[SHAP PROCESS] Computing SHAP values...")
        shap_start = time.time()
        try:
            shap_values = explainer.shap_values(X_sample)
            shap_time = time.time() - shap_start
            flush_print(f"[SHAP PROCESS] SHAP values computed in {shap_time:.2f} seconds")
        except Exception as e:
            flush_print(f"[SHAP ERROR] SHAP values computation failed: {e}")
            raise

        # Handle multi-class output
        if model_type == "classifier" and isinstance(shap_values, list) and len(shap_values) > 1:
            class_idx = 1 if len(shap_values) == 2 else 0
            shap_values_plot = shap_values[class_idx]
            flush_print(f"[SHAP PROCESS] Using class index {class_idx} for classification plot")
        else:
            shap_values_plot = shap_values

        # Validate SHAP values
        if isinstance(shap_values_plot, list):
            flush_print(f"[SHAP ERROR] SHAP values still a list after processing")
            raise ValueError("SHAP values dimension mismatch")
            
        if X_sample.shape[1] != shap_values_plot.shape[1]:
            flush_print(f"[SHAP ERROR] Shape mismatch - X: {X_sample.shape[1]}, SHAP: {shap_values_plot.shape[1]}")
            raise ValueError("SHAP and feature dimension mismatch")

        flush_print(f"[SHAP PROCESS] SHAP values shape: {shap_values_plot.shape}")

        # Generate standard bar plot
        flush_print("[SHAP PROCESS] Creating bar plot...")
        try:
            fig_standard = plt.figure(figsize=(10, 6))  # Reduced figure size
            shap.summary_plot(shap_values_plot, X_sample, plot_type="bar", max_display=min(top_n, 15), show=False)
            plt.tight_layout()
            
            if return_base64:
                standard_result = plot_to_base64(fig_standard)
            else:
                fig_standard.savefig(save_path, dpi=150, bbox_inches='tight')
                standard_result = save_path
                
            plt.close(fig_standard)
            flush_print("[SHAP PROCESS] Bar plot created successfully")
            
        except Exception as e:
            flush_print(f"[SHAP ERROR] Bar plot failed: {e}")
            if 'fig_standard' in locals():
                plt.close(fig_standard)

        # Generate detailed dot plot
        flush_print("[SHAP PROCESS] Creating dot plot...")
        try:
            fig_detailed = plt.figure(figsize=(10, 8))  # Reduced figure size
            shap.summary_plot(shap_values_plot, X_sample, plot_type="dot", max_display=min(top_n, 10), show=False)
            plt.tight_layout()
            
            if return_base64:
                detailed_result = plot_to_base64(fig_detailed)
            else:
                detailed_path = save_path.replace('.png', '_detailed.png')
                fig_detailed.savefig(detailed_path, dpi=150, bbox_inches='tight')
                detailed_result = detailed_path
                
            plt.close(fig_detailed)
            flush_print("[SHAP PROCESS] Dot plot created successfully")
            
        except Exception as e:
            flush_print(f"[SHAP ERROR] Dot plot failed: {e}")
            if 'fig_detailed' in locals():
                plt.close(fig_detailed)

        # Create feature importance DataFrame
        flush_print("[SHAP PROCESS] Creating feature importance DataFrame...")
        try:
            feature_importance = pd.DataFrame({
                "Feature": X_sample.columns,
                "Importance": np.abs(shap_values_plot).mean(axis=0)
            }).sort_values("Importance", ascending=False)
            flush_print(f"[SHAP PROCESS] Feature importance DataFrame created with {len(feature_importance)} features")
        except Exception as e:
            flush_print(f"[SHAP ERROR] Feature importance DataFrame failed: {e}")
            feature_importance = pd.DataFrame()

        total_time = time.time() - start_time
        flush_print(f"[SHAP PROCESS] SHAP processing completed in {total_time:.2f} seconds")

    except Exception as e:
        flush_print(f"[SHAP ERROR] SHAP processing failed: {e}")
        traceback.print_exc()
        
        # Try fallback method
        flush_print("[SHAP PROCESS] Attempting fallback feature importance...")
        try:
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
                flush_print("[SHAP PROCESS] Using model.feature_importances_")
            elif hasattr(model, "get_feature_importance"):
                importance = model.get_feature_importance()
                flush_print("[SHAP PROCESS] Using model.get_feature_importance()")
            elif hasattr(model, "coef_"):
                importance = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
                flush_print("[SHAP PROCESS] Using model.coef_")
            else:
                flush_print("[SHAP ERROR] No feature importance method available")
                raise ValueError("Feature importance not available")

            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importance
            }).sort_values('Importance', ascending=False)

            # Create fallback plot
            flush_print("[SHAP PROCESS] Creating fallback plot...")
            fig_fallback = plt.figure(figsize=(10, 6))
            top_features = feature_importance.head(min(top_n, len(feature_importance)))
            plt.barh(range(len(top_features)), top_features['Importance'].values, color="skyblue")
            plt.yticks(range(len(top_features)), top_features['Feature'].values)
            plt.xlabel("Feature Importance")
            plt.ylabel("Feature")
            plt.title("Feature Importance (Fallback Method)")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            if return_base64:
                standard_result = plot_to_base64(fig_fallback)
            else:
                fig_fallback.savefig(save_path, dpi=150, bbox_inches='tight')
                standard_result = save_path
                
            plt.close(fig_fallback)
            flush_print("[SHAP PROCESS] Fallback plot created successfully")

        except Exception as fallback_error:
            flush_print(f"[SHAP ERROR] Fallback feature importance failed: {fallback_error}")
            traceback.print_exc()
            queue.put({"error": str(fallback_error)})
            return

    # Clean up memory
    try:
        gc.collect()
        plt.close('all')
    except:
        pass

    # Return results
    result = {
        "shap_bar": standard_result,
        "shap_dot": detailed_result,
        "imp_df": feature_importance
    }
    
    flush_print(f"[SHAP PROCESS] Returning results - bar: {standard_result is not None}, dot: {detailed_result is not None}, df rows: {len(feature_importance)}")
    queue.put(result)


def safe_generate_feature_importance(
    model,
    X,
    model_type="classifier",
    output_dir=None,
    save_filename="feature_importance.png",
    top_n=20,
    return_base64=True,
    timeout=240  # 4 minutes timeout
):
    """
    Safely generate feature importance with timeout and error handling
    """
    flush_print(f"[MAIN] Starting safe_generate_feature_importance with timeout={timeout}s")
    
    queue = Queue()
    thread = Thread(
        target=_plot_feature_importance,
        args=(model, X, model_type, output_dir, top_n, save_filename, return_base64, queue),
        daemon=True
    )
    
    thread.start()
    flush_print("[MAIN] Thread started, waiting for completion...")
    
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        flush_print(f"[MAIN] Thread timed out after {timeout} seconds")
        # Note: In production, you might want to forcefully terminate the thread
        # but Python doesn't have a clean way to do this
        raise TimeoutError(f"Feature importance generation timed out after {timeout} seconds")
    
    try:
        result = queue.get_nowait()
        flush_print("[MAIN] Results retrieved from queue")
        if "error" in result:
            raise RuntimeError(f"Feature importance generation failed: {result['error']}")
        return result
    except:
        flush_print("[MAIN] No results in queue")
        raise RuntimeError("Feature importance generation failed: No results returned")