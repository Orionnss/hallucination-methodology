"""Comparison between detectors (prompt §2.8): corrected resampled t-test per cell (Nadeau & Bengio,
2003), Wilcoxon signed-rank test across cells, Holm correction (DECISIONS.md D54–D56)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    statistic: float
    p: float
    n: int
    flag: str = ""


def corrected_resampled_ttest(d: np.ndarray, test_train_ratio: float) -> TestResult:
    """d: per-fold differences over the k = B·F paired outer folds."""
    d = np.asarray(d, dtype=float)
    k = len(d)
    if k < 2:
        return TestResult(float("nan"), float("nan"), k, "too_few_folds")
    mean, var = d.mean(), d.var(ddof=1)
    if var == 0:
        return (TestResult(0.0, 1.0, k, "zero_variance") if mean == 0 else TestResult(math.copysign(math.inf, mean), 0.0, k, "zero_variance"))
    t = mean / math.sqrt((1.0 / k + test_train_ratio) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return TestResult(float(t), float(p), k)


def wilcoxon_across_cells(diffs: np.ndarray) -> TestResult:
    diffs = np.asarray(diffs, dtype=float)
    nz = diffs[diffs != 0]
    if len(nz) == 0:
        return TestResult(0.0, 1.0, len(diffs), "all_zero")
    r = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", method="exact")
    return TestResult(float(r.statistic), float(r.pvalue), len(diffs))


def min_wilcoxon_p(n: int) -> float:
    """Smallest attainable exact two-sided p with n non-zero differences."""
    return min(1.0, 2.0 / 2**n) if n > 0 else 1.0


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (NaN entries are kept as NaN and not counted)."""
    p = np.asarray(pvals, dtype=float)
    idx = [i for i in range(len(p)) if not np.isnan(p[i])]
    m = len(idx)
    out = np.full(len(p), np.nan)
    order = sorted(idx, key=lambda i: p[i])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out.tolist()
