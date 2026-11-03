"""Stage `sample` (prompt §2.2): S_j ⊂ D_j, uniform without replacement, |S_j| = n, seed s.
The same S_j is used for all generators (paired design)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from hbench.config import ExperimentCfg
from hbench.data import load_dataset_table
import json

from hbench.io import write_parquet_atomic
from hbench.paths import RunPaths


def draw_sample(table: pd.DataFrame, n: int, seed: int, dataset: str) -> pd.DataFrame:
    if len(table) < n:
        raise ValueError(f"dataset {dataset}: |D| = {len(table)} < n = {n}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(table), size=n, replace=False)
    out = table.iloc[np.sort(idx)].reset_index(drop=True)
    out.insert(0, "dataset", dataset)
    return out


def run(cfg: ExperimentCfg, datasets: list[str] | None = None) -> None:
    paths = RunPaths(cfg)
    for d in cfg.datasets:
        if datasets and d.name not in datasets:
            continue
        path = paths.samples(d.name)
        if path.exists():
            existing = pd.read_parquet(path)
            if len(existing) != cfg.n:
                raise RuntimeError(f"{path} has {len(existing)} rows but n = {cfg.n}; use a new run_id")
            logger.info(f"sample: {d.name} already done ({len(existing)} rows)")
            continue
        table, stats = load_dataset_table(d)
        s = draw_sample(table, cfg.n, cfg.seed, d.name)
        stats.update(rows_after_filters=len(table), n=cfg.n, seed=cfg.seed)
        write_parquet_atomic(s, path)
        path.with_suffix(".meta.json").write_text(json.dumps(stats, indent=2))
        logger.info(f"sample: {d.name} → {len(s)} rows ({stats})")
