import os
import io
import base64
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from queue import Queue
from threading import Thread

import shap
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use("Agg")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _plot_feature_importance(model, X, model_type, output_dir, top_n, save_filename, return_base64, queue):
    def plot_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return img_str

    standard_result = None
    detailed_result = None
    fallback_result = None
    feature_importance = pd.DataFrame()

    try:
        logger.info(f"🔍 Generating SHAP feature importance for {model_type} model")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, save_filename)
        else:
            save_path = save_filename

        sample_size = min(1000, len(X))
        X_sample = X.sample(sample_size, random_state=42) if len(X) > sample_size else X
        X_sample.columns = [str(c) for c in X_sample.columns]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        if model_type == "classifier" and isinstance(shap_values, list) and len(shap_values) > 1:
            class_idx = 1 if len(shap_values) == 2 else 0
            shap_values_plot = shap_values[class_idx]
            logger.info(f"✅ Using class index {class_idx} for SHAP classification plot")
        else:
            shap_values_plot = shap_values

        if isinstance(shap_values_plot, list) or X_sample.shape[1] != shap_values_plot.shape[1]:
            raise ValueError("SHAP and feature dimension mismatch")

        fig_standard = plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_plot, X_sample, plot_type="bar", max_display=top_n, show=False)
        plt.tight_layout()
        standard_result = plot_to_base64(fig_standard) if return_base64 else save_path
        plt.close(fig_standard)

        fig_detailed = plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values_plot, X_sample, plot_type="dot", max_display=min(top_n, 10), show=False)
        plt.tight_layout()
        detailed_result = plot_to_base64(fig_detailed) if return_base64 else save_path.replace('.png', '_detailed.png')
        plt.close(fig_detailed)

        feature_importance = pd.DataFrame({
            "Feature": X_sample.columns,
            "Importance": np.abs(shap_values_plot).mean(axis=0)
        }).sort_values("Importance", ascending=False)

        logger.info("✅ SHAP feature importance generated")

    except Exception as e:
        logger.error(f"❌ SHAP failed: {e}")
        try:
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
            elif hasattr(model, "get_feature_importance"):
                importance = model.get_feature_importance()
            elif hasattr(model, "coef_"):
                importance = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            else:
                raise ValueError("Feature importance not available")

            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importance
            }).sort_values('Importance', ascending=False)

            fig_fallback = plt.figure(figsize=(12, 8))
            plt.barh(feature_importance['Feature'].head(top_n), feature_importance['Importance'].head(top_n), color="skyblue")
            plt.xlabel("Feature Importance")
            plt.ylabel("Feature")
            plt.title("Fallback Feature Importance")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            fallback_result = plot_to_base64(fig_fallback) if return_base64 else save_path
            plt.close(fig_fallback)

        except Exception as fallback_error:
            logger.error(f"❌ Fallback feature importance failed: {fallback_error}")
            queue.put({"error": str(fallback_error)})
            return

    queue.put({
        "shap_bar": standard_result,
        "shap_dot": detailed_result,
        "imp_df": feature_importance
    })


def safe_generate_feature_importance(
    model,
    X,
    model_type="classifier",
    output_dir=None,
    save_filename="feature_importance.png",
    top_n=20,
    return_base64=True
):
    queue = Queue()
    thread = Thread(
        target=_plot_feature_importance,
        args=(model, X, model_type, output_dir, top_n, save_filename, return_base64, queue)
    )
    thread.start()
    thread.join()

    result = queue.get()
    if "error" in result:
        raise RuntimeError(f"Feature importance generation failed: {result['error']}")

    return result
