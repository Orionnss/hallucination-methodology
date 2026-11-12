"""Nested stratified cross-validation (prompt §2.6, DECISIONS.md D9, D10).

Outer folds are made once per (generator, dataset) cell and shared by all detectors. Inside each
outer training part T_f, an inner stratified CV selects the hyperparameters by mean inner AUROC,
then the threshold τ_f by mean inner MCC. h_f is fitted on all of T_f; V_f is used only for the
final scores and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from hbench.metrics import compute_metrics, confusion, mcc_from_counts


class FoldError(ValueError):
    pass


def check_class_counts(y: np.ndarray, n_splits: int, where: str) -> None:
    counts = {int(c): int((y == c).sum()) for c in (0, 1)}
    if min(counts.values()) < n_splits:
        raise FoldError(
            f"{where}: cannot make {n_splits} stratified folds with both classes in each; class counts {counts}"
        )


def outer_folds(y: np.ndarray, n_splits: int, seed: int, repeat: int, where: str) -> np.ndarray:
    """Return the outer fold index of each sample, for one repeat."""
    check_class_counts(y, n_splits, where)
    fold = np.full(len(y), -1, dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
    for f, (_, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold[te] = f
    for f in range(n_splits):
        if len(np.unique(y[fold == f])) < 2:
            raise FoldError(f"{where}: outer fold {f} has one class only")
    return fold


def threshold_candidates(scores: np.ndarray, max_n: int) -> np.ndarray:
    """Unique pooled inner out-of-fold scores, capped at `max_n` quantile points (D10)."""
    u = np.unique(scores)
    if len(u) > max_n:
        u = np.unique(np.quantile(u, np.linspace(0.0, 1.0, max_n)))
    return u


def select_threshold(inner: list[tuple[np.ndarray, np.ndarray]], max_n: int) -> tuple[float, float]:
    """τ maximising the mean inner-fold MCC. Ties → the smallest τ (D33)."""
    cands = threshold_candidates(np.concatenate([s for _, s in inner]), max_n)
    best_tau, best = float(cands[0]), -np.inf
    for tau in cands:
        mccs = [mcc_from_counts(*confusion(y, s >= tau))[0] for y, s in inner]
        m = float(np.mean(mccs))
        if m > best:
            best_tau, best = float(tau), m
    return best_tau, best


@dataclass
class OuterResult:
    params: dict[str, Any]
    inner_auroc: float
    tau: float
    inner_mcc: float
    scores: np.ndarray  # on V_f
    metrics: dict[str, float]
    flags: list[str]


def _n_workers(n_jobs: int, n_configs: int) -> int:
    import os

    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return max(1, min(n_configs, cpus if n_jobs < 0 else n_jobs))


def _inner_cv(make_clf, params, Xtr, ytr, inner_splits, single_thread: bool = False):
    """Inner CV of one grid point: mean inner AUROC and the inner out-of-fold scores."""
    if single_thread:  # worker process: one thread each, the parallelism is across grid points
        import torch

        torch.set_num_threads(1)
    inner = []
    for tr, va in inner_splits:
        clf = make_clf(**params).fit(Xtr[tr], ytr[tr])
        inner.append((ytr[va], clf.predict_proba(Xtr[va])))
    return float(np.mean([roc_auc_score(yv, s) for yv, s in inner])), inner


def run_outer_fold(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    make_clf: Callable[..., Any],
    grid: dict[str, list[Any]],
    inner_folds: int,
    seed: int,
    max_tau: int,
    where: str,
    n_jobs: int = 1,
) -> OuterResult:
    Xtr, ytr = X[train_idx], y[train_idx]
    check_class_counts(ytr, inner_folds, f"{where} (inner)")
    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    inner_splits = list(skf.split(np.zeros(len(ytr)), ytr))
    for k, (_, va) in enumerate(inner_splits):
        if len(np.unique(ytr[va])) < 2:
            raise FoldError(f"{where}: inner fold {k} has one class only")
    configs = list(ParameterGrid(grid))
    if n_jobs == 1 or len(configs) == 1:
        results = [_inner_cv(make_clf, p, Xtr, ytr, inner_splits) for p in configs]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=_n_workers(n_jobs, len(configs)), backend="loky")(
            delayed(_inner_cv)(make_clf, p, Xtr, ytr, inner_splits, single_thread=True) for p in configs
        )
    best = None
    for params, (auroc, inner) in zip(configs, results):  # grid order, as in the sequential loop
        if best is None or auroc > best[1]:  # ties → first in grid order
            best = (params, auroc, inner)
    params, auroc, inner = best
    tau, inner_mcc = select_threshold(inner, max_tau)
    h = make_clf(**params).fit(Xtr, ytr)
    scores = h.predict_proba(X[test_idx])
    metrics, flags = compute_metrics(y[test_idx], scores, tau)
    return OuterResult(params, auroc, tau, inner_mcc, scores, metrics, flags)
