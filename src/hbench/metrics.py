"""Metrics on each V_f, positive class = hallucination (prompt §2.7). Undefined values return 0 and a flag."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import roc_auc_score

METRICS = ("accuracy", "precision", "recall", "f1", "mcc", "auroc", "wacc")


def confusion(y: np.ndarray, yhat: np.ndarray) -> tuple[int, int, int, int]:
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    tp = int(((y == 1) & (yhat == 1)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    return tp, tn, fp, fn


def mcc_from_counts(tp: int, tn: int, fp: int, fn: int) -> tuple[float, bool]:
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if den == 0:
        return 0.0, True
    return (tp * tn - fp * fn) / math.sqrt(den), False


def weighted_accuracy(y: np.ndarray, yhat: np.ndarray) -> float:
    """Inverse class frequency weights w_c = 1 / n_c (prompt §2.7). Equal to ½(TPR + TNR)."""
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    counts = {c: int((y == c).sum()) for c in np.unique(y)}
    w = np.array([1.0 / counts[c] for c in y])
    return float((w * (y == yhat)).sum() / w.sum())


def compute_metrics(y: np.ndarray, scores: np.ndarray, tau: float) -> tuple[dict[str, float], list[str]]:
    y = np.asarray(y).astype(int)
    yhat = (np.asarray(scores) >= tau).astype(int)
    tp, tn, fp, fn = confusion(y, yhat)
    n = tp + tn + fp + fn
    flags: list[str] = []

    def ratio(num: int, den: int, name: str) -> float:
        if den == 0:
            flags.append(f"{name}_undefined")
            return 0.0
        return num / den

    m = {
        "accuracy": (tp + tn) / n,
        "precision": ratio(tp, tp + fp, "precision"),
        "recall": ratio(tp, tp + fn, "recall"),
        "f1": ratio(2 * tp, 2 * tp + fp + fn, "f1"),
    }
    m["mcc"], undef = mcc_from_counts(tp, tn, fp, fn)
    if undef:
        flags.append("mcc_undefined")
    if len(np.unique(y)) < 2:
        m["auroc"] = 0.0
        flags.append("auroc_undefined")
    else:
        m["auroc"] = float(roc_auc_score(y, scores))
    m["wacc"] = weighted_accuracy(y, yhat)
    m.update(tp=tp, tn=tn, fp=fp, fn=fn)
    return m, flags
