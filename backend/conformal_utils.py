# conformal_utils.py
import numpy as np
import json

class Thresholds:
    def __init__(self, class_order, global_tau=None, per_class_tau=None, mondrian=False):
        self.class_order = class_order
        self.global_tau = global_tau
        self.per_class_tau = per_class_tau
        self.mondrian = mondrian

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

def fit_thresholds(calib_probs, calib_true_idx, class_order, alpha=0.1, mondrian=False):
    """
    Fit simple conformal prediction thresholds.
    alpha: miscoverage level (1 - coverage target)
    """
    n, k = calib_probs.shape
    if not mondrian:
        scores = 1 - calib_probs[np.arange(n), calib_true_idx]
        tau = np.quantile(scores, 1 - alpha)
        return Thresholds(class_order, global_tau=float(tau), mondrian=False)
    else:
        per_class_tau = {}
        for c in range(k):
            idx = np.where(calib_true_idx == c)[0]
            if idx.size == 0:
                continue
            scores = 1 - calib_probs[idx, c]
            per_class_tau[c] = float(np.quantile(scores, 1 - alpha))
        return Thresholds(class_order, per_class_tau=per_class_tau, mondrian=True)

def prediction_set(probs, thr: Thresholds, max_set_size=2):
    """
    Build a prediction set for one sample.
    Returns (indices, abstention_flag).
    """
    if thr.mondrian:
        idxs = [i for i, p in enumerate(probs) if (1 - p) <= thr.per_class_tau.get(i, 1.0)]
    else:
        idxs = [i for i, p in enumerate(probs) if (1 - p) <= thr.global_tau]

    idxs = sorted(idxs, key=lambda i: -probs[i])[:max_set_size]
    abst = len(idxs) == 0
    return idxs, abst

def batch_metrics(calib_probs, calib_true_idx, thr: Thresholds, max_set_size=2):
    """Compute coverage and avg set size across calibration set."""
    n = calib_probs.shape[0]
    covered, sizes, abst = 0, [], 0
    for i in range(n):
        idxs, abst_i = prediction_set(calib_probs[i], thr, max_set_size=max_set_size)
        sizes.append(len(idxs))
        if calib_true_idx[i] in idxs:
            covered += 1
        if abst_i:
            abst += 1
    return {
        "coverage": covered / n if n else None,
        "avg_set_size": float(np.mean(sizes)) if sizes else None,
        "abstention_rate": abst / n if n else None
    }
