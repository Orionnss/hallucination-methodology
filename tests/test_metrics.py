import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from hbench.metrics import compute_metrics, weighted_accuracy


@pytest.mark.parametrize("seed", range(20))
def test_metrics_match_sklearn(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(10, 200))
    y = rng.integers(0, 2, n)
    y[:2] = [0, 1]
    s = np.clip(y * rng.uniform(0, 0.6) + rng.uniform(0, 1, n), 0, 1.6) / 1.6
    tau = float(rng.uniform(0.2, 0.8))
    yhat = (s >= tau).astype(int)
    m, flags = compute_metrics(y, s, tau)
    assert m["accuracy"] == pytest.approx(accuracy_score(y, yhat))
    assert m["precision"] == pytest.approx(precision_score(y, yhat, zero_division=0))
    assert m["recall"] == pytest.approx(recall_score(y, yhat, zero_division=0))
    assert m["f1"] == pytest.approx(f1_score(y, yhat, zero_division=0))
    assert m["mcc"] == pytest.approx(matthews_corrcoef(y, yhat))
    assert m["auroc"] == pytest.approx(roc_auc_score(y, s))
    assert m["wacc"] == pytest.approx(balanced_accuracy_score(y, yhat))


@pytest.mark.parametrize("seed", range(20))
def test_weighted_accuracy_equals_half_tpr_plus_tnr(seed):
    rng = np.random.default_rng(100 + seed)
    y = rng.integers(0, 2, 50)
    y[:2] = [0, 1]
    yhat = rng.integers(0, 2, 50)
    tpr = ((y == 1) & (yhat == 1)).sum() / (y == 1).sum()
    tnr = ((y == 0) & (yhat == 0)).sum() / (y == 0).sum()
    assert weighted_accuracy(y, yhat) == pytest.approx(0.5 * (tpr + tnr))


def test_undefined_values_flagged():
    y = np.array([0, 0, 1, 1])
    m, flags = compute_metrics(y, np.array([0.1, 0.2, 0.3, 0.4]), tau=0.9)  # no positive prediction
    assert m["precision"] == 0 and "precision_undefined" in flags
    assert m["mcc"] == 0 and "mcc_undefined" in flags
    assert "recall_undefined" not in flags
    assert m["auroc"] == pytest.approx(1.0)


def test_auroc_uses_scores_not_predictions():
    y = np.array([0, 0, 1, 1])
    m, _ = compute_metrics(y, np.array([0.1, 0.4, 0.35, 0.8]), tau=0.5)
    assert m["auroc"] == pytest.approx(0.75)
