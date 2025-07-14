import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import time
import signal
import traceback
import gc
import os
from threading import Thread
from queue import Queue

def flush_print(*args, **kwargs):
    """Print with immediate flush to ensure output appears in logs"""
    print(*args, **kwargs)
    import sys
    sys.stdout.flush()
    sys.stderr.flush()

def timeout_handler(signum, frame):
    """Handle timeout signal"""
    flush_print("⏰ TIMEOUT: Process timed out")
    raise TimeoutError("Process timed out")

def _plot_feature_importance(model, X, model_type, output_dir, top_n, save_filename, return_base64, queue):
    """Main plotting function with enhanced error handling and CatBoost-specific fixes"""
    
    def plot_to_base64(fig):
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            return img_str
        except Exception as e:
            flush_print(f"[PLOT ERROR] Failed to convert plot to base64: {e}")
            return None
        finally:
            plt.close(fig)

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

        # CatBoost-specific handling
        model_class_name = type(model).__name__.lower()
        is_catboost = 'catboost' in model_class_name
        
        if is_catboost:
            flush_print("[SHAP PROCESS] CatBoost model detected - using specialized handling")
            
            # For CatBoost, use a smaller sample size and specific parameters
            sample_size = min(100, len(X))  # Much smaller for CatBoost
            flush_print(f"[SHAP PROCESS] Using CatBoost sample size: {sample_size}")
            
            if len(X) > sample_size:
                X_sample = X.sample(sample_size, random_state=42)
            else:
                X_sample = X.copy()
                
            # Ensure data is in the right format for CatBoost
            X_sample = X_sample.astype(float)
            
            try:
                # For CatBoost, try creating explainer with specific parameters
                flush_print("[SHAP PROCESS] Creating CatBoost TreeExplainer...")
                explainer = shap.TreeExplainer(model, X_sample.iloc[:50])  # Use background data
                flush_print("[SHAP PROCESS] CatBoost TreeExplainer created successfully")
                
                # Use even smaller sample for SHAP computation
                shap_sample = X_sample.iloc[:50]
                flush_print(f"[SHAP PROCESS] Using mini-sample for SHAP computation: {shap_sample.shape}")
                
            except Exception as e:
                flush_print(f"[SHAP ERROR] CatBoost TreeExplainer failed: {e}")
                # Try alternative approach for CatBoost
                try:
                    flush_print("[SHAP PROCESS] Trying CatBoost with no background data...")
                    explainer = shap.TreeExplainer(model)
                    shap_sample = X_sample.iloc[:25]  # Even smaller
                    flush_print(f"[SHAP PROCESS] Alternative CatBoost approach, sample: {shap_sample.shape}")
                except Exception as e2:
                    flush_print(f"[SHAP ERROR] Alternative CatBoost approach failed: {e2}")
                    raise e2
                    
        else:
            # Non-CatBoost models
            sample_size = min(300, len(X))
            flush_print(f"[SHAP PROCESS] Using standard sample size: {sample_size}")
            
            if len(X) > sample_size:
                X_sample = X.sample(sample_size, random_state=42)
            else:
                X_sample = X.copy()
                
            shap_sample = X_sample
            
            try:
                explainer = shap.TreeExplainer(model)
                flush_print("[SHAP PROCESS] Standard TreeExplainer created successfully")
            except Exception as e:
                flush_print(f"[SHAP ERROR] Standard TreeExplainer failed: {e}")
                raise

        # Ensure column names are strings
        shap_sample.columns = [str(c) for c in shap_sample.columns]
        flush_print(f"[SHAP PROCESS] Final sample prepared, shape: {shap_sample.shape}")

        # Generate SHAP values with timeout and progress tracking
        flush_print("[SHAP PROCESS] Computing SHAP values...")
        shap_start = time.time()
        
        try:
            # Set a signal alarm for SHAP computation timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)  # 1 minute timeout for SHAP computation
            
            # Try to compute SHAP values
            if is_catboost:
                # For CatBoost, try with check_additivity=False
                flush_print("[SHAP PROCESS] Computing CatBoost SHAP values (check_additivity=False)...")
                shap_values = explainer.shap_values(shap_sample, check_additivity=False)
            else:
                shap_values = explainer.shap_values(shap_sample)
                
            signal.alarm(0)  # Cancel timeout
            shap_time = time.time() - shap_start
            flush_print(f"[SHAP PROCESS] SHAP values computed in {shap_time:.2f} seconds")
            
        except TimeoutError:
            flush_print("[SHAP ERROR] SHAP computation timed out")
            signal.alarm(0)
            raise
        except Exception as e:
            flush_print(f"[SHAP ERROR] SHAP values computation failed: {e}")
            signal.alarm(0)
            raise

        # Handle different SHAP value formats
        flush_print(f"[SHAP PROCESS] Raw SHAP values type: {type(shap_values)}")
        flush_print(f"[SHAP PROCESS] Model type: {model_type}")
        
        if model_type == "classifier" and isinstance(shap_values, list) and len(shap_values) > 1:
            # Multi-class classification
            class_idx = 1 if len(shap_values) == 2 else 0
            shap_values_plot = shap_values[class_idx]
            flush_print(f"[SHAP PROCESS] Using class index {class_idx} for classification plot")
        elif model_type == "classifier" and isinstance(shap_values, list) and len(shap_values) == 1:
            # Binary classification with single class output
            shap_values_plot = shap_values[0]
            flush_print(f"[SHAP PROCESS] Using single class output for binary classification")
        elif model_type == "regression" or model_type == "regressor":
            # Regression - should be a 2D array
            if isinstance(shap_values, list):
                if len(shap_values) == 1:
                    shap_values_plot = shap_values[0]
                    flush_print(f"[SHAP PROCESS] Regression with list format - using first element")
                else:
                    flush_print(f"[SHAP ERROR] Unexpected list format for regression: {len(shap_values)} elements")
                    shap_values_plot = shap_values[0]  # Try first element anyway
            else:
                shap_values_plot = shap_values
                flush_print(f"[SHAP PROCESS] Regression with array format")
        else:
            # Default case
            shap_values_plot = shap_values
            flush_print(f"[SHAP PROCESS] Using default SHAP values format")

        # Validate SHAP values
        if isinstance(shap_values_plot, list):
            flush_print(f"[SHAP ERROR] SHAP values still a list after processing: {len(shap_values_plot)} elements")
            # Try to convert to array if it's a list of arrays
            try:
                shap_values_plot = np.array(shap_values_plot)
                flush_print(f"[SHAP PROCESS] Converted list to array, shape: {shap_values_plot.shape}")
            except:
                raise ValueError("SHAP values dimension mismatch - cannot convert list to array")
            
        # Check dimensions
        if hasattr(shap_values_plot, 'shape'):
            flush_print(f"[SHAP PROCESS] SHAP values shape: {shap_values_plot.shape}")
            if len(shap_values_plot.shape) == 2 and shap_sample.shape[1] != shap_values_plot.shape[1]:
                flush_print(f"[SHAP ERROR] Shape mismatch - X: {shap_sample.shape[1]}, SHAP: {shap_values_plot.shape[1]}")
                raise ValueError("SHAP and feature dimension mismatch")
        else:
            flush_print(f"[SHAP ERROR] SHAP values has no shape attribute: {type(shap_values_plot)}")
            raise ValueError("Invalid SHAP values format")

        # Create feature importance DataFrame first
        flush_print("[SHAP PROCESS] Creating feature importance DataFrame...")
        try:
            feature_importance = pd.DataFrame({
                "Feature": shap_sample.columns,
                "Importance": np.abs(shap_values_plot).mean(axis=0)
            }).sort_values("Importance", ascending=False)
            flush_print(f"[SHAP PROCESS] Feature importance DataFrame created with {len(feature_importance)} features")
        except Exception as e:
            flush_print(f"[SHAP ERROR] Feature importance DataFrame failed: {e}")
            feature_importance = pd.DataFrame()

        # Generate standard bar plot with timeout protection
        flush_print("[SHAP PROCESS] Creating bar plot...")
        try:
            # Set up timeout for plotting
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30 second timeout for plotting
            
            # Clear any existing plots
            plt.clf()
            plt.cla()
            plt.close('all')
            
            # Create figure with explicit settings
            fig_standard = plt.figure(figsize=(10, 6), facecolor='white')
            plt.rcParams.update({'font.size': 10})
            
            # Try the new SHAP API first
            try:
                shap.plots.bar(shap_values_plot, max_display=min(top_n, 15), show=False)
            except:
                # Fallback to old API
                shap.summary_plot(shap_values_plot, shap_sample, plot_type="bar", max_display=min(top_n, 15), show=False)
            
            plt.tight_layout()
            signal.alarm(0)  # Cancel timeout
            
            if return_base64:
                standard_result = plot_to_base64(fig_standard)
            else:
                fig_standard.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
                standard_result = save_path
                plt.close(fig_standard)
                
            flush_print("[SHAP PROCESS] Bar plot created successfully")
            
        except TimeoutError:
            flush_print("[SHAP ERROR] Bar plot timed out")
            signal.alarm(0)
            plt.close('all')
        except Exception as e:
            flush_print(f"[SHAP ERROR] Bar plot failed: {e}")
            signal.alarm(0)
            plt.close('all')

        # Generate detailed dot plot with timeout protection
        flush_print("[SHAP PROCESS] Creating dot plot...")
        try:
            # Set up timeout for plotting
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30 second timeout for plotting
            
            # Clear any existing plots
            plt.clf()
            plt.cla()
            plt.close('all')
            
            # Create figure with explicit settings
            fig_detailed = plt.figure(figsize=(10, 8), facecolor='white')
            plt.rcParams.update({'font.size': 10})
            
            # Try the new SHAP API first
            try:
                shap.plots.beeswarm(shap_values_plot, max_display=min(top_n, 10), show=False)
            except:
                # Fallback to old API
                shap.summary_plot(shap_values_plot, shap_sample, plot_type="dot", max_display=min(top_n, 10), show=False)
            
            plt.tight_layout()
            signal.alarm(0)  # Cancel timeout
            
            if return_base64:
                detailed_result = plot_to_base64(fig_detailed)
            else:
                detailed_path = save_path.replace('.png', '_detailed.png')
                fig_detailed.savefig(detailed_path, dpi=100, bbox_inches='tight', facecolor='white')
                detailed_result = detailed_path
                plt.close(fig_detailed)
                
            flush_print("[SHAP PROCESS] Dot plot created successfully")
            
        except TimeoutError:
            flush_print("[SHAP ERROR] Dot plot timed out")
            signal.alarm(0)
            plt.close('all')
        except Exception as e:
            flush_print(f"[SHAP ERROR] Dot plot failed: {e}")
            signal.alarm(0)
            plt.close('all')

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

            # Create fallback plot with timeout protection
            flush_print("[SHAP PROCESS] Creating fallback plot...")
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)  # 30 second timeout
                
                plt.clf()
                plt.cla()
                plt.close('all')
                
                fig_fallback = plt.figure(figsize=(10, 6), facecolor='white')
                top_features = feature_importance.head(min(top_n, len(feature_importance)))
                
                plt.barh(range(len(top_features)), top_features['Importance'].values, color="skyblue")
                plt.yticks(range(len(top_features)), top_features['Feature'].values)
                plt.xlabel("Feature Importance")
                plt.ylabel("Feature")
                plt.title("Feature Importance (Fallback Method)")
                plt.gca().invert_yaxis()
                plt.tight_layout()
                
                signal.alarm(0)  # Cancel timeout
                
                if return_base64:
                    standard_result = plot_to_base64(fig_fallback)
                else:
                    fig_fallback.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
                    standard_result = save_path
                    plt.close(fig_fallback)
                    
                flush_print("[SHAP PROCESS] Fallback plot created successfully")
                
            except TimeoutError:
                flush_print("[SHAP ERROR] Fallback plot timed out")
                signal.alarm(0)
                plt.close('all')

        except Exception as fallback_error:
            flush_print(f"[SHAP ERROR] Fallback feature importance failed: {fallback_error}")
            traceback.print_exc()
            signal.alarm(0)
            queue.put({"error": str(fallback_error)})
            return

    finally:
        # Comprehensive cleanup
        try:
            signal.alarm(0)  # Cancel any remaining alarms
            plt.close('all')
            gc.collect()
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
    timeout=120  # Reduced to 2 minutes for CatBoost
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
        # Force cleanup
        try:
            plt.close('all')
            gc.collect()
        except:
            pass
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