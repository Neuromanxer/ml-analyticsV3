import os
import time
import uuid
import json
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path as PathL
from celery import Celery
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.base import clone
import traceback
import json
import numpy as np
import io
import sys
import shap
from typing import Union, List, Any, Dict
import tempfile
def clean_data_for_json(data):
    """Recursively clean data structure for JSON serialization"""
    if isinstance(data, dict):
        return {k: clean_data_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif isinstance(data, (int, float, np.integer, np.floating)):
        val = float(data)
        if np.isnan(val) or np.isinf(val):
            return 0.0
        return val
    elif isinstance(data, str):
        return data
    else:
        return str(data)

def safe_float_conversion(value):
    """Convert value to float, handling NaN and infinity"""
    try:
        val = float(value)
        if np.isnan(val) or np.isinf(val):
            return 0.0  # or some default value
        return val
    except (ValueError, TypeError):
        return 0.0

def _safe_float(x):
    try:
        f = float(x)
        return f if np.isfinite(f) else None
    except Exception:
        return None

def _pct_change(new, old):
    try:
        if not np.isfinite(new) or not np.isfinite(old) or old == 0:
            return None
        return 100.0*(new - old)/abs(old)
    except Exception:
        return None

def detect_task_type(y: pd.Series) -> str:
    y_nonnull = y.dropna()
    nunique = y_nonnull.nunique()
    unique_vals = set(pd.unique(y_nonnull))
    if pd.api.types.is_bool_dtype(y_nonnull) or unique_vals <= {0, 1}:
        return "classification"
    thresh = min(20, max(2, int(0.1 * len(y_nonnull))))
    return "classification" if nunique <= thresh else "regression"

def _safe_float(x):
    try:
        f = float(x)
        return f if np.isfinite(f) else None
    except Exception:
        return None

def _pct_change(new, old):
    try:
        if new is None or old is None or not np.isfinite(new) or not np.isfinite(old) or old == 0:
            return None
        return 100.0 * (new - old) / abs(old)
    except Exception:
        return None

def _to_1d(a):
    if a is None:
        return np.array([])
    a = np.asarray(a)
    if a.ndim == 0:
        a = a.reshape(1)
    if a.ndim > 1 and 1 in a.shape:
        a = a.ravel()
    return a.astype(float)

def _to_1d_proba(p, desired_outcome=None):
    """Return 1D positive-class proba vector from p; handle (n,2), (n,k), (n,), scalars."""
    if p is None:
        return np.array([])
    p = np.asarray(p)
    if p.ndim == 1:
        return p.astype(float)
    if p.ndim == 2:
        n, k = p.shape
        if k == 2:
            return p[:, 1].astype(float)
        # multiclass: use desired_outcome column if provided and valid, else max-prob
        if isinstance(desired_outcome, (int, np.integer)) and 0 <= desired_outcome < k:
            return p[:, int(desired_outcome)].astype(float)
        return p.max(axis=1).astype(float)
    # fallback
    return p.ravel().astype(float)

def compute_summary_metrics(
    sample: pd.DataFrame,
    target_column: str,
    applied_changes: dict,
    original_preds: np.ndarray,
    modified_preds: np.ndarray,
    original_proba: np.ndarray | None = None,
    modified_proba: np.ndarray | None = None,
    desired_outcome: int | float | None = None
) -> dict:
    n = len(sample)
    task_type = detect_task_type(sample[target_column]) if target_column in sample.columns else (
        "classification" if set(np.unique(original_preds)) <= {0, 1} else "regression"
    )

    # ---------- Per-feature change summary ----------
    num_cols = sample.select_dtypes(include=[np.number]).columns.tolist()
    feature_change_rows = []
    for feat, change in (applied_changes or {}).items():
        row = {"feature": str(feat), "applied_change": change, "kind": "auto"}
        if isinstance(change, str) and change.strip().endswith("%"):
            row["kind"] = "percent"
        elif isinstance(change, (int, float, np.number)):
            row["kind"] = "absolute"

        if feat in num_cols and f"{feat}__original" in sample.columns and f"{feat}__modified" in sample.columns:
            base = pd.to_numeric(sample[f"{feat}__original"], errors="coerce")
            new  = pd.to_numeric(sample[f"{feat}__modified"], errors="coerce")
            abs_changes = (new - base).to_numpy()
            with np.errstate(all="ignore"):
                pct_changes = np.where(base.to_numpy() != 0, 100.0 * abs_changes / np.abs(base.to_numpy()), np.nan)
            row.update({
                "mean_abs_change": _safe_float(np.nanmean(abs_changes)) if abs_changes.size else None,
                "mean_pct_change": _safe_float(np.nanmean(pct_changes)) if abs_changes.size else None,
            })
        feature_change_rows.append(row)

    # ---------- Coerce preds/probas & align lengths ----------
    o_pred = _to_1d(original_preds)
    m_pred = _to_1d(modified_preds)
    o_proba = _to_1d_proba(original_proba, desired_outcome=desired_outcome)
    m_proba = _to_1d_proba(modified_proba, desired_outcome=desired_outcome)

    # align by shortest available vector when both present
    def _align(a, b):
        if a.size == 0 or b.size == 0:
            return a, b
        k = min(a.size, b.size)
        return a[:k], b[:k]

    o_pred_aligned, m_pred_aligned = _align(o_pred, m_pred)
    o_proba_aligned, m_proba_aligned = _align(o_proba, m_proba)

    # ---------- Deltas ----------
    if o_pred_aligned.size and m_pred_aligned.size:
        pred_delta = m_pred_aligned - o_pred_aligned
        pred_delta_mean = _safe_float(np.nanmean(pred_delta))
        pred_delta_median = _safe_float(np.nanmedian(pred_delta))
        pred_delta_abs_mean = _safe_float(np.nanmean(np.abs(pred_delta)))
    else:
        pred_delta_mean = pred_delta_median = pred_delta_abs_mean = None

    metrics = {
        "n_samples": int(n),
        "task_type": task_type,
        "pred_delta_mean": pred_delta_mean,
        "pred_delta_median": pred_delta_median,
        "pred_delta_abs_mean": pred_delta_abs_mean,
    }

    # ---------- Task-specific ----------
    if task_type == "classification":
        # Prefer probabilities; fall back to label means if needed
        if o_proba_aligned.size:
            base_pos_rate = _safe_float(np.nanmean(o_proba_aligned))
        else:
            base_pos_rate = _safe_float(np.nanmean(o_pred)) if o_pred.size else None

        if m_proba_aligned.size:
            new_pos_rate = _safe_float(np.nanmean(m_proba_aligned))
        else:
            new_pos_rate = _safe_float(np.nanmean(m_pred)) if m_pred.size else None

        uplift_abs = _safe_float(new_pos_rate - base_pos_rate) if (new_pos_rate is not None and base_pos_rate is not None) else None
        uplift_pct = _pct_change(new_pos_rate, base_pos_rate)

        metrics.update({
            "positive_rate_baseline": base_pos_rate,
            "positive_rate_new": new_pos_rate,
            "positive_rate_uplift_abs": uplift_abs,
            "positive_rate_uplift_pct": uplift_pct,
        })

        if desired_outcome is not None and m_pred.size:
            success = np.sum(m_pred == desired_outcome)
            metrics["success_rate"] = _safe_float(100.0 * success / m_pred.size)

    else:
        base_mean = _safe_float(np.nanmean(o_pred)) if o_pred.size else None
        new_mean  = _safe_float(np.nanmean(m_pred)) if m_pred.size else None
        metrics.update({
            "pred_mean_baseline": base_mean,
            "pred_mean_new": new_mean,
            "pred_mean_change_pct": _pct_change(new_mean, base_mean),
            "total_delta": _safe_float(np.nansum(m_pred_aligned - o_pred_aligned)) if (o_pred_aligned.size and m_pred_aligned.size) else None,
        })

        if desired_outcome is not None and o_pred_aligned.size and m_pred_aligned.size:
            toward = np.mean(np.sign(desired_outcome - o_pred_aligned) == np.sign(m_pred_aligned - o_pred_aligned))
            metrics["moved_toward_goal_pct"] = _safe_float(100.0 * toward)

    # ---------- Attach feature changes & compact profile ----------
    metrics["feature_change_summary"] = feature_change_rows

       # ----- Numeric profile as ready-to-render HTML -----
    try:
        num_summary = (sample.select_dtypes(include=[np.number])
                       .agg(['mean','median','std','min','max'])
                       .T.reset_index().rename(columns={'index':'feature'}))
        for c in ['mean','median','std','min','max']:
            num_summary[c] = num_summary[c].map(_safe_float)

        # Make sure column names are strings
        num_summary.columns = [str(c) for c in num_summary.columns]

        # Minimal, self-contained CSS + HTML table
        css = """
        <style>
          .num-prof-wrap { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
          .num-prof-title { font-weight: 600; margin: 0 0 8px 0; font-size: 14px; }
          table.num-prof-table { border-collapse: collapse; width: 100%; font-size: 13px; }
          .num-prof-table thead th { text-align: left; border-bottom: 1px solid #e5e7eb; padding: 6px 8px; }
          .num-prof-table tbody td { border-bottom: 1px solid #f1f5f9; padding: 6px 8px; }
          .num-prof-table tbody tr:nth-child(odd) { background: #fafafa; }
          .num-prof-table tbody tr:hover { background: #f3f4f6; }
        </style>
        """.strip()

        table_html = num_summary.to_html(index=False, border=0, justify="left", classes="num-prof-table")
        metrics["sample_numeric_profile_html"] = f'{css}<div class="num-prof-wrap"><div class="num-prof-title">Numeric Profile</div>{table_html}</div>'

        # Remove previous non-HTML fields if you were setting them
        metrics.pop("sample_numeric_profile", None)
        metrics.pop("sample_numeric_profile_md", None)

    except Exception:
        metrics["sample_numeric_profile_html"] = "<div>No numeric profile available.</div>"
        metrics.pop("sample_numeric_profile", None)
        metrics.pop("sample_numeric_profile_md", None)

    return metrics
