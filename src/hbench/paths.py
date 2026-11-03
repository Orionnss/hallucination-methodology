"""Run directory layout (PROPOSAL.md)."""

from __future__ import annotations

from pathlib import Path

from hbench.config import ExperimentCfg


class RunPaths:
    def __init__(self, cfg: ExperimentCfg):
        self.root = cfg.run_dir

    def samples(self, dataset: str) -> Path:
        return self.root / "samples" / f"{dataset}.parquet"

    def generations(self, generator: str, dataset: str) -> Path:
        return self.root / "generations" / generator / dataset

    def extra_samples(self, generator: str, dataset: str) -> Path:
        return self.root / "extra_samples" / generator / dataset

    def votes(self, judge: str, generator: str, dataset: str) -> Path:
        return self.root / "votes" / judge / generator / dataset

    def labels(self, generator: str, dataset: str) -> Path:
        return self.root / "labels" / generator / f"{dataset}.parquet"

    def label_stats(self) -> Path:
        return self.root / "labels" / "labelling_stats.parquet"
