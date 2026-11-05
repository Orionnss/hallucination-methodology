"""Parity tests (ADDING_A_METHOD.md §6): the authors' original pipeline vs ours, compared in memory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

REL_FLOOR = 1e-6  # D76


@dataclass
class ParityResult:
    method: str
    dtype: str
    n_samples: int
    n_excluded: int  # answer tokens of the original run ≠ stored answer
    max_abs_err: float
    max_rel_err: float
    spearman: float
    verdict: str  # pass | pass_with_note | fail

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def compare(method: str, dtype: str, theirs: list[np.ndarray], ours: list[np.ndarray], n_excluded: int = 0) -> ParityResult:
    """Element-wise comparison. max_rel_err = max |a − b| / max(|b|, 1e-6); Spearman ρ over all values
    of all samples. Verdict: decision 6.A (≤ 1e-3 fp32 / ≤ 1e-2 bf16 → pass; else ρ ≥ 0.99 → pass with note)."""
    for i, (t, o) in enumerate(zip(theirs, ours)):
        if t.shape != o.shape:
            raise ValueError(f"{method}: sample {i} has shape {o.shape} (ours) vs {t.shape} (theirs)")
    a = np.concatenate([np.asarray(o, dtype=np.float64).ravel() for o in ours])
    b = np.concatenate([np.asarray(t, dtype=np.float64).ravel() for t in theirs])
    ok = ~(np.isnan(a) & np.isnan(b))
    a, b = a[ok], b[ok]
    abs_err = np.abs(a - b)
    rel = abs_err / np.maximum(np.abs(b), REL_FLOOR)
    rho = float(spearmanr(a, b).statistic) if len(a) > 1 and np.std(b) > 0 else float("nan")
    tol = 1e-3 if dtype == "float32" else 1e-2
    max_rel = float(rel.max()) if len(rel) else 0.0
    verdict = "pass" if max_rel <= tol else ("pass_with_note" if rho >= 0.99 else "fail")
    return ParityResult(method, dtype, len(ours), n_excluded, float(abs_err.max()) if len(a) else 0.0, max_rel, rho, verdict)
