import numpy as np
import pytest

from hbench.cv import FoldError, outer_folds, run_outer_fold, select_threshold, threshold_candidates


def _data(n=90, d=4, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=n) < 0.35).astype(int)
    X = rng.normal(size=(n, 2, d)) + y[:, None, None] * 0.8
    X[:, 0, 0] = np.arange(n)  # unique row id in layer 0, feature 0 (used by the spy)
    return X, y


def test_outer_folds_stratified_and_deterministic():
    _, y = _data()
    f1 = outer_folds(y, 5, seed=3, repeat=0, where="t")
    f2 = outer_folds(y, 5, seed=3, repeat=0, where="t")
    assert np.array_equal(f1, f2)
    assert not np.array_equal(f1, outer_folds(y, 5, seed=3, repeat=1, where="t"))
    pos = [y[f1 == f].sum() for f in range(5)]
    assert max(pos) - min(pos) <= 1  # stratified: positives spread evenly
    assert set(np.bincount(f1)) <= {17, 18, 19}


def test_fold_error_reports_counts():
    y = np.array([0] * 20 + [1] * 3)
    with pytest.raises(FoldError, match=r"cell X.*\{0: 20, 1: 3\}"):
        outer_folds(y, 5, 0, 0, where="cell X")


def test_threshold_selection():
    assert len(threshold_candidates(np.linspace(0, 1, 1000), 200)) <= 200
    inner = [(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.6, 0.7])), (np.array([0, 1, 0, 1]), np.array([0.3, 0.65, 0.2, 0.9]))]
    tau, mcc = select_threshold(inner, 200)
    assert mcc == pytest.approx(1.0) and tau == pytest.approx(0.6)  # smallest τ with the best mean MCC


class Spy:
    """Classifier that records every row it is given (by the row id in X[:, 0, 0])."""

    log: list[tuple[str, set[int]]] = []

    def __init__(self, layer=0, C=1.0, seed=0):
        self.layer = layer

    def fit(self, X, y):
        Spy.log.append(("fit", set(X[:, 0, 0].astype(int))))
        self.w = X[:, 1].mean(axis=0)
        return self

    def predict_proba(self, X):
        Spy.log.append(("predict", set(X[:, 0, 0].astype(int))))
        z = X[:, 1] @ self.w
        return 1 / (1 + np.exp(-z))


def test_no_leakage_of_outer_test_fold():
    X, y = _data()
    fold = outer_folds(y, 5, 0, 0, where="t")
    for f in range(5):
        Spy.log = []
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        run_outer_fold(X, y, tr, te, Spy, {"layer": [0, 1], "C": [0.1, 1.0]}, 3, 0, 200, "t")
        test_ids = set(te)
        *selection, final = Spy.log
        # Every fit, and every prediction during the inner CV, sees only T_f.
        for kind, ids in selection:
            assert not ids & test_ids, kind
        # The only call that sees V_f is the final prediction, and it sees only V_f.
        assert final == ("predict", test_ids)
        assert sum(1 for k, _ in Spy.log if k == "fit") == 4 * 3 + 1
