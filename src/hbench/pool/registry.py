"""Detector registry: name → (feature extractor, classifier, default grid). Registered by name in the
config (prompt §3.3). Grid values "all" for `layer` expand to every layer of the stored features."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from hbench.pool.methods.reference import LastAnswerHidden, LayerLogReg, MeanAnswerHidden, SaplmaMLP


@dataclass
class DetectorSpec:
    name: str
    extractor: Callable[..., Any]
    classifier: Callable[..., Any]
    default_grid: dict[str, Any] = field(default_factory=dict)


REGISTRY: dict[str, DetectorSpec] = {
    "mean_hidden_probe": DetectorSpec(
        "mean_hidden_probe", MeanAnswerHidden, LayerLogReg, {"layer": "all", "C": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]}
    ),
    "saplma_style_probe": DetectorSpec("saplma_style_probe", LastAnswerHidden, SaplmaMLP, {"layer": "all"}),
}


def get_detector(name: str) -> DetectorSpec:
    if name not in REGISTRY:
        raise KeyError(f"unknown detector {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name]


def resolve_grid(spec: DetectorSpec, override: dict[str, list[Any]], n_layers: int) -> dict[str, list[Any]]:
    grid = dict(spec.default_grid)
    grid.update(override)
    if grid.get("layer") == "all":
        grid["layer"] = list(range(n_layers))
    return {k: (v if isinstance(v, list) else [v]) for k, v in grid.items()}
