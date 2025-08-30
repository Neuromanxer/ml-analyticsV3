import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, log_loss
import lightgbm as lgb
import xgboost as xgb
from lightgbm import early_stopping, log_evaluation
import catboost as cb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from queue import Queue
from threading import Thread
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import root_mean_squared_error

import base64
from io import BytesIO


# from .preprocessing import preprocess_data
# from .conformal_utils import prediction_set

from preprocessing import preprocess_data

from conformal_utils import prediction_set
# XGBoost Parameters for Classification
xgb_params_c = {
    "objective": "binary:logistic",
    "n_estimators": 2000,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 10,
    "random_state": 42,
    "use_label_encoder": False,
    "verbosity": 1  # Info-level logs; verbose handled in fit()
}

# LightGBM Parameters for Classification
lgb_params_c = {
    "objective": "binary",
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "max_depth": -1,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42
    # verbose removed: use callback in fit
}

# CatBoost Parameters for Classification
cat_params_c = {
    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "bagging_temperature": 1.0,
    "border_count": 254,
    "loss_function": "Logloss",
    "random_state": 42,
    "verbose": 500  # Supported directly
}
class ModelClassifyingTrainer:
    def __init__(self, data: pd.DataFrame, n_splits: int = 5):
        """
        data: a DataFrame containing ALL training rows, including the target column.
        n_splits: number of CV folds.
        """
        self.raw = data.copy()
        self.n_splits = n_splits

    def train_model(self, params: dict, target: str, title: str):
        """
        Runs Stratified K-Fold CV on the passed data.

        Returns:
          - models: list of fitted models, one per fold
          - oof_preds: numpy array of out-of-fold predictions (length = n_samples)
        """
        df = self.raw
        X = df.drop(columns=['ID', target], errors='ignore')
        y = df[target]

        oof_preds = np.zeros(len(X))
        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        models = []
        for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            # instantiate the right classifier
            if title.startswith('LightGBM'):
                model = LGBMClassifier(**params)
            elif title.startswith('CatBoost'):
                model = CatBoostClassifier(**params)
            else:
                model = XGBClassifier(**params)

            # fit on fold
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
            models.append(model)

            # store OOF predictions
            oof_preds[val_idx] = model.predict(X_val)

        return models, oof_preds

def plot_to_base64(fig):
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    fig.tight_layout()
    buf = BytesIO()
    
    # Use a proper canvas to render the figure
    FigureCanvas(fig).print_png(buf)  # ✅ ensures it's rendered correctly

    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)  # ✅ Close the figure properly
    return img_base64
import numpy as np
from collections import Counter
from sklearn.metrics import log_loss
from sklearn.preprocessing import label_binarize

def multiclass_brier(y_true, probs, class_order):
    """Generalized Brier score for K classes."""
    Y = label_binarize(y_true, classes=class_order)
    if Y.shape[1] != probs.shape[1]:
        # handle binary case where label_binarize returns shape (N,1)
        Y = np.hstack([1 - Y, Y]) if probs.shape[1] == 2 else Y
    return float(np.mean((Y - probs) ** 2))

def reliability_bins(probs, y_true, class_order, n_bins=10):
    """
    Multiclass-friendly reliability: bin by max prob; accuracy = (argmax == y_true).
    Returns [{bin, conf, acc, n}, ...] with 'bin' as bin center.
    """
    pmax = probs.max(axis=1)
    y_pred = probs.argmax(axis=1)
    class_to_idx = {c: i for i, c in enumerate(class_order)}
    y_idx = np.array([class_to_idx[v] for v in y_true])
    correct = (y_pred == y_idx).astype(int)

    # Equal-frequency bins for stability
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(pmax, qs)
    # ensure monotonic
    edges = np.maximum.accumulate(edges)
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1] + 1e-12
        m = (pmax >= lo) & (pmax < hi) if i < n_bins - 1 else (pmax >= lo) & (pmax <= hi)
        n = int(m.sum())
        if n == 0:
            bins.append({"bin": float((lo + hi) / 2), "conf": None, "acc": None, "n": 0})
            continue
        conf = float(pmax[m].mean())
        acc = float(correct[m].mean())
        bins.append({"bin": float((lo + hi) / 2), "conf": conf, "acc": acc, "n": n})
    return bins

def decision_curve_binary(probs_pos, y_true, thresholds=None):
    """
    Compute simple decision curve for binary: volume, precision, recall at thresholds.
    probs_pos: array of P(y=1)
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.linspace(0.1, 0.9, 9)]
    y_true = np.asarray(y_true).astype(int)
    out = []
    for t in thresholds:
        sel = probs_pos >= t
        volume = int(sel.sum())
        if volume == 0:
            out.append({"threshold": float(t), "volume": 0, "precision": None, "recall": None, "expected_profit": None})
            continue
        tp = int(((y_true == 1) & sel).sum())
        fp = int(((y_true == 0) & sel).sum())
        fn = int(((y_true == 1) & (~sel)).sum())
        precision = tp / volume if volume else None
        recall = tp / (tp + fn) if (tp + fn) else None
        out.append({"threshold": float(t), "volume": volume, "precision": float(precision) if precision is not None else None,
                    "recall": float(recall) if recall is not None else None, "expected_profit": None})
    return out
import numpy as np
import matplotlib.pyplot as plt

# All functions return a base64 string (no data: prefix)

def plot_reliability_diagram(bins):
    """
    bins: [{bin: float, conf: float|None, acc: float|None, n: int}, ...]
    """
    if not bins:
        return ""
    xs   = [b["bin"] for b in bins]
    conf = [b["conf"] if b["conf"] is not None else np.nan for b in bins]
    acc  = [b["acc"]  if b["acc"]  is not None else np.nan for b in bins]
    n    = [b["n"] for b in bins]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect")
    ax.plot(xs, conf, marker="o", label="Avg predicted prob")
    ax.plot(xs, acc,  marker="s", label="Observed accuracy")
    ax.set_xlabel("Confidence bin")
    ax.set_ylabel("Value")
    ax.set_title("Reliability Diagram")
    ax.legend(loc="lower right")

    ax2 = ax.twinx()
    # pick a reasonable bin width visually
    width = 0.8 * (xs[1] - xs[0]) if len(xs) > 1 else 0.05
    ax2.bar(xs, n, alpha=0.25, width=width)
    ax2.set_ylabel("Bin count")

    b64 = plot_to_base64(fig)
    plt.close(fig)
    return b64

def plot_decision_curve(points, recommended_t=None):
    """
    points: [{threshold, volume, precision, recall, expected_profit|None}, ...]
    """
    if not points:
        return ""
    t    = [p["threshold"] for p in points]
    vol  = [p["volume"] for p in points]
    prec = [p["precision"] if p["precision"] is not None else np.nan for p in points]
    rec  = [p["recall"]    if p["recall"]    is not None else np.nan for p in points]
    prof = [p.get("expected_profit", None) for p in points]
    prof = [x if x is not None else np.nan for x in prof]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, prec, marker="o", label="Precision")
    ax.plot(t, rec,  marker="s", label="Recall")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Precision / Recall")

    ax2 = ax.twinx()
    ax2.plot(t, prof, marker="^", linestyle=":", label="Expected Profit")
    ax2.set_ylabel("Expected Profit")

    if recommended_t is not None:
        ax.axvline(recommended_t, linestyle="--", linewidth=1)

    ax.set_title("Decision / ROI Curve")
    # Merge legends from both axes
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")

    b64 = plot_to_base64(fig)
    plt.close(fig)
    return b64

def plot_segment_heatmap(segments):
    """
    segments: [{name, n, ece, coverage, avg_set_size, alarm}, ...]
    """
    if not segments:
        return ""
    rows = [s["name"] for s in segments]
    cols = ["ECE", "Coverage", "Avg set size"]
    M = np.array([[s["ece"], s["coverage"], s["avg_set_size"]] for s in segments], dtype=float)

    fig, ax = plt.subplots(figsize=(7, max(1.0, 0.45 * len(rows) + 1)))
    im = ax.imshow(M, aspect="auto")

    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.set_title("Segment Health")

    # annotate each cell
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    b64 = plot_to_base64(fig)
    plt.close(fig)
    return b64

def plot_kpi_abstentions(count, rate):
    """
    count: int of items needing review; rate: float in [0,1]
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(["Needs review"], [count])
    ax.set_ylim(0, max(count, 1))
    ax.set_title(f"Abstentions (rate {rate:.1%})")
    b64 = plot_to_base64(fig)
    plt.close(fig)
    return b64

def plot_trend(y, title="Trend"):
    """
    y: sequence of floats; title: string
    """
    if y is None or len(y) == 0:
        return ""
    x = np.arange(len(y))
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.plot(x, y, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Window")
    ax.set_ylabel("Value")
    b64 = plot_to_base64(fig)
    plt.close(fig)
    return b64
import numpy as np
import pandas as pd

def compute_reliability_bins_multiclass(probs, y_true, class_order, n_bins=10):
    """Top-1 multiclass reliability bins on max prob."""
    probs = np.asarray(probs, float)
    y_true = np.asarray(y_true)
    class_to_idx = {c: i for i, c in enumerate(class_order)}
    y_idx = np.array([class_to_idx[v] for v in y_true])

    pmax = probs.max(axis=1)
    yhat = probs.argmax(axis=1)
    correct = (yhat == y_idx).astype(int)

    # equal-frequency binning
    if len(pmax) == 0:
        return []
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(pmax, qs)
    edges = np.maximum.accumulate(edges)

    out = []
    for i in range(n_bins):
        lo, hi = edges[i], (edges[i+1] + 1e-12)
        m = (pmax >= lo) & (pmax < hi) if i < n_bins - 1 else (pmax >= lo) & (pmax <= hi)
        n = int(m.sum())
        if n == 0:
            out.append({"bin": float((lo + hi)/2), "conf": None, "acc": None, "n": 0})
            continue
        out.append({
            "bin": float((lo + hi)/2),
            "conf": float(pmax[m].mean()),
            "acc": float(correct[m].mean()),
            "n": n
        })
    return out

def ece_from_bins(bins):
    """Expected Calibration Error from reliability bin dicts."""
    n_total = sum(b.get("n", 0) for b in bins)
    if n_total == 0:
        return None
    ece = 0.0
    for b in bins:
        n = b.get("n", 0)
        conf = b.get("conf", None)
        acc  = b.get("acc", None)
        if n > 0 and conf is not None and acc is not None:
            ece += (n / n_total) * abs(acc - conf)
    return float(ece)

def decision_curve_binary(probs_pos, y_true, thresholds=None):
    """volume/precision/recall across thresholds for binary."""
    probs_pos = np.asarray(probs_pos, float)
    y_true = np.asarray(y_true).astype(int)
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.linspace(0.1, 0.9, 9)]
    out = []
    for t in thresholds:
        sel = probs_pos >= t
        vol = int(sel.sum())
        if vol == 0:
            out.append({"threshold": float(t), "volume": 0, "precision": None, "recall": None, "expected_profit": None})
            continue
        tp = int(((y_true == 1) & sel).sum())
        fp = int(((y_true == 0) & sel).sum())
        fn = int(((y_true == 1) & (~sel)).sum())
        precision = tp / vol if vol else None
        recall    = tp / (tp + fn) if (tp + fn) else None
        out.append({
            "threshold": float(t),
            "volume": vol,
            "precision": float(precision) if precision is not None else None,
            "recall": float(recall) if recall is not None else None,
            "expected_profit": None
        })
    return out
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

# assumes you already have:
# - compute_reliability_bins_multiclass(probs, y_true, class_order, n_bins)
# - ece_from_bins(bins)
# - conformal_utils.prediction_set (imported below)

def compute_segments_table(
    test_df: pd.DataFrame,
    y_test: pd.Series,
    test_probs: np.ndarray,
    class_order: np.ndarray,
    thr,                                 # conformal thresholds object
    segment_cols: Optional[List[str]] = None,
    min_n: int = 200,
    max_set_size: int = 2,
    coverage_target: float = 0.90,
    ece_alarm: float = 0.08,
    coverage_tolerance: float = 0.02,
    max_categories: int = 30,
) -> List[Dict[str, Any]]:
    """
    Returns a list of rows like:
      {"name": "channel=email", "n": 5231, "ece": 0.034, "coverage": 0.902,
       "avg_set_size": 1.15, "abstention_rate": 0.03, "alarm": False, "low_n": False}
    """
    from conformal_utils import prediction_set

    # If we don't have probabilities (e.g., estimator lacks predict_proba), no segments.
    if test_probs is None or len(test_probs) == 0:
        return []

    # Infer segment columns if not provided: text/categorical with reasonable cardinality
    if segment_cols is None:
        segment_cols = []
        for c in test_df.columns:
            dt = test_df[c].dtype
            if (dt == "object" or pd.api.types.is_categorical_dtype(dt)) and test_df[c].nunique(dropna=False) <= max_categories:
                segment_cols.append(c)
    if not segment_cols:
        return []

    # Ensure y_test is a Series aligned to test_df rows
    y_series = y_test if isinstance(y_test, pd.Series) else pd.Series(y_test, index=test_df.index)

    # Map labels -> indices once
    class_to_idx = {c: i for i, c in enumerate(class_order)}
    y_idx_all = y_series.map(lambda v: class_to_idx[v]).to_numpy()

    rows = []

    for col in segment_cols:
        s = test_df[col].astype("object").fillna("NA").astype(str)

        # Limit categories to top (by count) to avoid explosion
        top_vals = s.value_counts(dropna=False).head(max_categories).index.tolist()

        for val in top_vals:
            idx = np.where(s.values == val)[0]
            n = int(idx.size)
            if n == 0:
                continue

            low_n = n < min_n
            probs_i = test_probs[idx]
            y_i = y_series.iloc[idx]
            y_idx_i = y_idx_all[idx]

            # --- ECE via reliability bins (top-1, multiclass-friendly) ---
            bins_i = compute_reliability_bins_multiclass(probs_i, y_i, class_order, n_bins=10)
            ece_i = ece_from_bins(bins_i)

            # --- Conformal set stats: coverage, avg set size, abstention rate ---
            covered = 0
            sizes = []
            abst_cnt = 0
            for r in range(len(idx)):
                ids, abst = prediction_set(probs_i[r], thr, max_set_size=max_set_size)
                sizes.append(len(ids))
                if y_idx_i[r] in ids:
                    covered += 1
                if abst:
                    abst_cnt += 1

            coverage = covered / n if n else None
            avg_size = float(np.mean(sizes)) if sizes else None
            abst_rate = abst_cnt / n if n else None

            # Alarm rule: only consider ECE alarm for sufficiently large segments
            alarm = (
                (ece_i is not None and not low_n and ece_i > ece_alarm)
                or (coverage is not None and coverage < coverage_target - coverage_tolerance)
            )

            rows.append({
                "name": f"{col}={val}",
                "n": n,
                "ece": None if ece_i is None else float(ece_i),
                "coverage": None if coverage is None else float(coverage),
                "avg_set_size": None if avg_size is None else float(avg_size),
                "abstention_rate": None if abst_rate is None else float(abst_rate),
                "alarm": bool(alarm),
                "low_n": bool(low_n),
            })
def plot_to_base64(fig):
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    fig.tight_layout()
    buf = BytesIO()
    
    # Use a proper canvas to render the figure
    FigureCanvas(fig).print_png(buf)  # ✅ ensures it's rendered correctly

    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)  # ✅ Close the figure properly
    return img_base64
def reliability_bins(probs, y_true, class_order, n_bins=10):
    """
    Multiclass-friendly reliability: bin by max prob; accuracy = (argmax == y_true).
    Returns [{bin, conf, acc, n}, ...] with 'bin' as bin center.
    """
    pmax = probs.max(axis=1)
    y_pred = probs.argmax(axis=1)
    class_to_idx = {c: i for i, c in enumerate(class_order)}
    y_idx = np.array([class_to_idx[v] for v in y_true])
    correct = (y_pred == y_idx).astype(int)

    # Equal-frequency bins for stability
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(pmax, qs)
    edges = np.maximum.accumulate(edges)  # ensure monotonic

    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1] + 1e-12
        m = (pmax >= lo) & (pmax < hi) if i < n_bins - 1 else (pmax >= lo) & (pmax <= hi)
        n = int(m.sum())
        if n == 0:
            bins.append({"bin": float((lo + hi) / 2), "conf": None, "acc": None, "n": 0})
            continue
        conf = float(pmax[m].mean())
        acc = float(correct[m].mean())
        bins.append({"bin": float((lo + hi) / 2), "conf": conf, "acc": acc, "n": n})
    return bins


def decision_curve_binary(probs_pos, y_true, thresholds=None):
    """
    Compute simple decision curve for binary: volume, precision, recall at thresholds.
    probs_pos: array of P(y=1)
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.linspace(0.1, 0.9, 9)]
    y_true = np.asarray(y_true).astype(int)

    out = []
    for t in thresholds:
        sel = probs_pos >= t
        volume = int(sel.sum())
        if volume == 0:
            out.append({
                "threshold": float(t),
                "volume": 0,
                "precision": None,
                "recall": None,
                "expected_profit": None
            })
            continue

        tp = int(((y_true == 1) & sel).sum())
        fn = int(((y_true == 1) & (~sel)).sum())
        precision = tp / volume if volume else None
        recall = tp / (tp + fn) if (tp + fn) else None

        out.append({
            "threshold": float(t),
            "volume": volume,
            "precision": float(precision) if precision is not None else None,
            "recall": float(recall) if recall is not None else None,
            "expected_profit": None
        })
    return out


# -------------------------------
# Visualization helpers
# -------------------------------

def plot_reliability_diagram(bins):
    """
    bins: [{bin: float, conf: float|None, acc: float|None, n: int}, ...]
    """
    if not bins:
        return ""

    xs = [b["bin"] for b in bins]
    conf = [b["conf"] if b["conf"] is not None else np.nan for b in bins]
    acc = [b["acc"] if b["acc"] is not None else np.nan for b in bins]
    n = [b["n"] for b in bins]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect")
    ax.plot(xs, conf, marker="o", label="Avg predicted prob")
    ax.plot(xs, acc, marker="s", label="Observed accuracy")
    ax.set_xlabel("Confidence bin")
    ax.set_ylabel("Value")
    ax.set_title("Reliability Diagram")
    ax.legend(loc="lower right")

    ax2 = ax.twinx()
    width = 0.8 * (xs[1] - xs[0]) if len(xs) > 1 else 0.05
    ax2.bar(xs, n, alpha=0.25, width=width)
    ax2.set_ylabel("Bin count")

    b64 = plot_to_base64(fig)
    plt.close(fig)
    return b64


def plot_decision_curve(points, recommended_t=None):
    """
    points: [{threshold, volume, precision, recall, expected_profit|None}, ...]
    """
    if not points:
        return ""

    t = [p["threshold"] for p in points]
    prec = [p["precision"] if p["precision"] is not None else np.nan for p in points]
    rec = [p["recall"] if p["recall"] is not None else np.nan for p in points]
    prof = [p.get("expected_profit", np.nan) for p in points]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, prec, marker="o", label="Precision")
    ax.plot(t, rec, marker="s", label="Recall")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Precision / Recall")

    ax2 = ax.twinx()
    ax2.plot(t, prof, marker="^", linestyle=":", label="Expected Profit")
    ax2.set_ylabel("Expected Profit")

    if recommended_t is not None:
        ax.axvline(recommended_t, linestyle="--", linewidth=1)

    ax.set_title("Decision / ROI Curve")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")

    b64 = plot_to_base64(fig)
    plt.close(fig)
    return b64


def plot_segment_heatmap(segments):
    """
    segments: [{name, n, ece, coverage, avg_set_size, alarm}, ...]
    """
    if not segments:
        return ""

    rows = [s["name"] for s in segments]
    cols = ["ECE", "Coverage", "Avg set size"]
    M = np.array([[s["ece"], s["coverage"], s["avg_set_size"]] for s in segments], dtype=float)

    fig, ax = plt.subplots(figsize=(7, max(1.0, 0.45 * len(rows) + 1)))
    im = ax.imshow(M, aspect="auto")

    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.set_title("Segment Health")

    # annotate each cell
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    b64 = plot_to_base64(fig)
    plt.close(fig)
    return b64


def plot_kpi_abstentions(count, rate):
    """
    count: int of items needing review; rate: float in [0,1]
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(["Needs review"], [count])
    ax.set_ylim(0, max(count, 1))
    ax.set_title(f"Abstentions (rate {rate:.1%})")
    b64 = plot_to_base64(fig)
    plt.close(fig)
    return b64


def plot_trend(y, title="Trend"):
    """
    y: sequence of floats; title: string
    """
    if y is None or len(y) == 0:
        return ""
    x = np.arange(len(y))
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.plot(x, y, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Window")
    ax.set_ylabel("Value")
    b64 = plot_to_base64(fig)
    plt.close(fig)
    return b64
def compute_reliability_bins_multiclass(probs, y_true, class_order, n_bins=10):
    """Top-1 multiclass reliability bins on max prob."""
    probs = np.asarray(probs, float)
    y_true = np.asarray(y_true)
    class_to_idx = {c: i for i, c in enumerate(class_order)}
    y_idx = np.array([class_to_idx[v] for v in y_true])

    pmax = probs.max(axis=1)
    yhat = probs.argmax(axis=1)
    correct = (yhat == y_idx).astype(int)

    if len(pmax) == 0:
        return []
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(pmax, qs)
    edges = np.maximum.accumulate(edges)

    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1] + 1e-12
        m = (pmax >= lo) & (pmax < hi) if i < n_bins - 1 else (pmax >= lo) & (pmax <= hi)
        n = int(m.sum())
        if n == 0:
            out.append({"bin": float((lo + hi) / 2), "conf": None, "acc": None, "n": 0})
            continue
        out.append({
            "bin": float((lo + hi) / 2),
            "conf": float(pmax[m].mean()),
            "acc": float(correct[m].mean()),
            "n": n
        })
    return out


def ece_from_bins(bins):
    """Expected Calibration Error from reliability bin dicts."""
    n_total = sum(b.get("n", 0) for b in bins)
    if n_total == 0:
        return None
    ece = 0.0
    for b in bins:
        n = b.get("n", 0)
        conf = b.get("conf", None)
        acc = b.get("acc", None)
        if n > 0 and conf is not None and acc is not None:
            ece += (n / n_total) * abs(acc - conf)
    return float(ece)


# -------------------------------
# Segments table
# -------------------------------

def compute_segments_table(
    test_df: pd.DataFrame,
    y_test: pd.Series,
    test_probs: np.ndarray,
    class_order: np.ndarray,
    thr,  # conformal thresholds object
    segment_cols: Optional[List[str]] = None,
    min_n: int = 200,
    max_set_size: int = 2,
    coverage_target: float = 0.90,
    ece_alarm: float = 0.08,
    coverage_tolerance: float = 0.02,
    max_categories: int = 30,
) -> List[Dict[str, Any]]:
    """
    Returns a list of rows like:
      {"name": "channel=email", "n": 5231, "ece": 0.034, "coverage": 0.902,
       "avg_set_size": 1.15, "abstention_rate": 0.03, "alarm": False, "low_n": False}
    """
    

    if test_probs is None or len(test_probs) == 0:
        return []

    # Infer segment columns if not provided
    if segment_cols is None:
        segment_cols = []
        for c in test_df.columns:
            dt = test_df[c].dtype
            if (dt == "object" or pd.api.types.is_categorical_dtype(dt)) and test_df[c].nunique(dropna=False) <= max_categories:
                segment_cols.append(c)
    if not segment_cols:
        return []

    # Align y_test to DataFrame
    y_series = y_test if isinstance(y_test, pd.Series) else pd.Series(y_test, index=test_df.index)

    class_to_idx = {c: i for i, c in enumerate(class_order)}
    y_idx_all = y_series.map(lambda v: class_to_idx[v]).to_numpy()

    rows = []
    for col in segment_cols:
        s = test_df[col].astype("object").fillna("NA").astype(str)
        top_vals = s.value_counts(dropna=False).head(max_categories).index.tolist()

        for val in top_vals:
            idx = np.where(s.values == val)[0]
            n = int(idx.size)
            if n == 0:
                continue

            low_n = n < min_n
            probs_i = test_probs[idx]
            y_i = y_series.iloc[idx]
            y_idx_i = y_idx_all[idx]

            # Reliability bins → ECE
            bins_i = compute_reliability_bins_multiclass(probs_i, y_i, class_order, n_bins=10)
            ece_i = ece_from_bins(bins_i)

            # Conformal stats
            covered, sizes, abst_cnt = 0, [], 0
            for r in range(len(idx)):
                ids, abst = prediction_set(probs_i[r], thr, max_set_size=max_set_size)
                sizes.append(len(ids))
                if y_idx_i[r] in ids:
                    covered += 1
                if abst:
                    abst_cnt += 1

            coverage = covered / n if n else None
            avg_size = float(np.mean(sizes)) if sizes else None
            abst_rate = abst_cnt / n if n else None

            alarm = (
                (ece_i is not None and not low_n and ece_i > ece_alarm)
                or (coverage is not None and coverage < coverage_target - coverage_tolerance)
            )

            rows.append({
                "name": f"{col}={val}",
                "n": n,
                "ece": None if ece_i is None else float(ece_i),
                "coverage": None if coverage is None else float(coverage),
                "avg_set_size": None if avg_size is None else float(avg_size),
                "abstention_rate": None if abst_rate is None else float(abst_rate),
                "alarm": bool(alarm),
                "low_n": bool(low_n),
            })

    rows.sort(key=lambda r: (not r["alarm"], -r["n"]))
    return rows