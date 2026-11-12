"""Feature × classifier cross-product (DECISIONS.md D89).

A combination = a *view* of one stored feature (the feature-specific selection that each published
method does before its classifier) followed by a *head* (the classifier of a published method, run
unchanged on a 2-D input). The three vector features and the three published vector classifiers give
a 3 × 3 grid. The diagonal is the published detectors themselves (lapeigvals, icr_probe,
saplma_reimpl); a combination on the diagonal gives the same predictions as the published detector
(tests/test_combos.py). Only the off-diagonal combinations are registered as new detectors.
CHARM stays outside the grid: its graph is summarised inside its GNN, so its stored feature has no
fixed size and its classifier needs a graph (D89).

Views (selection hyperparameters are chosen in the inner CV, as in the published detectors):
- lapeigvals_feat: top-k Laplacian eigenvalues [L, H, 100] → k ∈ {5, 10, 25, 50, 100} (≤ shortest
  sequence) × layer ∈ {all, each}, flattened (D70, D77).
- icr_feat: per-layer mean ICR score [L], all layers (no selection).
- hidden_feat: last-answer-token hidden state [L, d] → one layer, chosen in the inner CV (D12).
Heads (their own preprocessing kept):
- pca_lr: lapeigvals' PCA(512) + balanced logistic regression (D70).
- icr_mlp: ICR's `ICRProbe` MLP with our training loop from their paper (D68).
- saplma_mlp: SAPLMA's 256-128-64 MLP, inputs standardised in the training fold (D81).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import numpy as np


# ---------- views ----------


@dataclass(frozen=True)
class View:
    name: str
    extractor: Callable[[], Any]
    default_grid: dict[str, Any]
    adjust_grid: Callable[[dict, np.ndarray], dict] | None
    apply: Callable[..., np.ndarray]  # (X, **params) → [N, p]


def _lap_apply(X: np.ndarray, k: int, layer: str | int = "all") -> np.ndarray:
    """The selection of LapEigvalsLR._features: top-k per head, all layers or one, flattened."""
    Z = X[:, :, :, :k] if layer == "all" else X[:, int(layer) : int(layer) + 1, :, :k]
    Z = Z.reshape(len(Z), -1)
    if np.isnan(Z).any():
        raise ValueError(f"k = {k} exceeds the sequence length of some samples")
    return Z


def _flat_apply(X: np.ndarray) -> np.ndarray:
    return X.reshape(len(X), -1)


def _layer_apply(X: np.ndarray, layer: int) -> np.ndarray:
    return X[:, int(layer)]


def _views() -> dict[str, View]:
    from hbench.pool.methods import icr_probe, lapeigvals, saplma_reimpl

    return {
        "lapeigvals_feat": View("lapeigvals_feat", lapeigvals.LapEigvalsExtractor, {"k": lapeigvals.TOP_K_GRID, "layer": "all_and_each"},
                                lapeigvals.adjust_grid, _lap_apply),
        "icr_feat": View("icr_feat", icr_probe.ICRFeature, {}, None, _flat_apply),
        "hidden_feat": View("hidden_feat", saplma_reimpl.SaplmaReimplFeature, {"layer": "all"}, None, _layer_apply),
    }


# ---------- heads: the published classifiers on a 2-D input ----------


class _PcaLR:
    def __init__(self, seed: int):
        self.seed = seed

    def fit(self, Z, y):
        from hbench.pool.methods.lapeigvals import LapEigvalsLR

        self.m = LapEigvalsLR(k=Z.shape[1], layer="all", seed=self.seed).fit(Z[:, None, None, :], y)
        return self

    def predict_proba(self, Z):
        return self.m.predict_proba(Z[:, None, None, :])


class _IcrMLP:
    def __init__(self, seed: int):
        self.seed = seed

    def fit(self, Z, y):
        from hbench.pool.methods.icr_probe import ICRProbeClassifier

        self.m = ICRProbeClassifier(seed=self.seed).fit(Z, y)
        return self

    def predict_proba(self, Z):
        return self.m.predict_proba(Z)


class _SaplmaMLP:
    def __init__(self, seed: int):
        self.seed = seed

    def fit(self, Z, y):
        from hbench.pool.methods.saplma_reimpl import SaplmaReimplMLP

        self.m = SaplmaReimplMLP(layer=0, seed=self.seed).fit(Z[:, None, :], y)
        return self

    def predict_proba(self, Z):
        return self.m.predict_proba(Z[:, None, :])


HEADS: dict[str, type] = {"pca_lr": _PcaLR, "icr_mlp": _IcrMLP, "saplma_mlp": _SaplmaMLP}

# The diagonal: published detector ↔ (view, head).
PUBLISHED = {"lapeigvals": ("lapeigvals_feat", "pca_lr"), "icr_probe": ("icr_feat", "icr_mlp"), "saplma_reimpl": ("hidden_feat", "saplma_mlp")}


class ComboClassifier:
    """View then head. `view_params` are the view's grid values (k, layer)."""

    def __init__(self, view: View, head: str, seed: int = 0, **view_params):
        self.view, self.head, self.seed, self.view_params = view, head, seed, view_params

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ComboClassifier":
        self.h = HEADS[self.head](self.seed).fit(self.view.apply(X, **self.view_params), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.h.predict_proba(self.view.apply(X, **self.view_params))


def combo_name(view: str, head: str) -> str:
    return f"{view}+{head}"


def axes(name: str) -> tuple[str, str] | None:
    """(view, head) of a grid detector, published (diagonal) or combination; None otherwise."""
    if name in PUBLISHED:
        return PUBLISHED[name]
    if "+" in name:
        v, h = name.split("+", 1)
        return v, h
    return None


def register(registry: dict, spec_cls) -> None:
    """Registers the off-diagonal combinations (group "cross")."""
    views = _views()
    diagonal = set(PUBLISHED.values())
    for vname, view in views.items():
        for head in HEADS:
            if (vname, head) in diagonal:
                continue
            name = combo_name(vname, head)
            registry[name] = spec_cls(name, view.extractor, partial(ComboClassifier, view, head), dict(view.default_grid),
                                      adjust_grid=view.adjust_grid, group="cross")
