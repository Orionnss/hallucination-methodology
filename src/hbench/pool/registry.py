"""Detector registry: name → (feature extractor, classifier, default grid). Registered by name in the
config (prompt §3.3). Grid values "all" for `layer` expand to every layer of the stored features."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from hbench.pool.methods.reference import LastAnswerHidden, LayerLogReg, MeanAnswerHidden, SaplmaMLP


@dataclass
class DetectorSpec:
    name: str
    extractor: Callable[..., Any]
    classifier: Callable[..., Any]
    default_grid: dict[str, Any] = field(default_factory=dict)
    adjust_grid: Callable[[dict, np.ndarray], dict] | None = None  # label-free grid adjustment from the features
    group: str = "published"  # report group: published | reimplemented | reference


REGISTRY: dict[str, DetectorSpec] = {
    # Reference detectors of prompt §3.3 (pipeline baselines, not published methods).
    "mean_hidden_probe": DetectorSpec(
        "mean_hidden_probe", MeanAnswerHidden, LayerLogReg, {"layer": "all", "C": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]}, group="reference"
    ),
    "saplma_style_probe": DetectorSpec("saplma_style_probe", LastAnswerHidden, SaplmaMLP, {"layer": "all"}, group="reference"),
}


def _register_published() -> None:
    from hbench.pool.methods import charm, icr_probe, lapeigvals, saplma_reimpl

    REGISTRY["saplma_reimpl"] = DetectorSpec(
        "saplma_reimpl", saplma_reimpl.SaplmaReimplFeature, saplma_reimpl.SaplmaReimplMLP, {"layer": "all"}, group="reimplemented"
    )

    REGISTRY["charm"] = DetectorSpec("charm", charm.CharmGraph, charm.CharmClassifier, {})

    REGISTRY["icr_probe"] = DetectorSpec("icr_probe", icr_probe.ICRFeature, icr_probe.ICRProbeClassifier, {})

    REGISTRY["lapeigvals"] = DetectorSpec(
        "lapeigvals", lapeigvals.LapEigvalsExtractor, lapeigvals.LapEigvalsLR,
        {"k": lapeigvals.TOP_K_GRID, "layer": "all_and_each"}, adjust_grid=lapeigvals.adjust_grid,
    )


_register_published()


def get_detector(name: str) -> DetectorSpec:
    if name not in REGISTRY:
        raise KeyError(f"unknown detector {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name]


def resolve_grid(spec: DetectorSpec, override: dict[str, list[Any]], n_layers: int, X: np.ndarray | None = None) -> dict[str, list[Any]]:
    grid = dict(spec.default_grid)
    grid.update(override)
    if spec.adjust_grid is not None:
        grid = spec.adjust_grid(grid, X)
    if grid.get("layer") == "all":
        grid["layer"] = list(range(n_layers))
    return {k: (v if isinstance(v, list) else [v]) for k, v in grid.items()}
