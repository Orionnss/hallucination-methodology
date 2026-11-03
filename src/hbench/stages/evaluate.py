"""Stage `evaluate` (prompt §2.6, §2.7): nested stratified CV for each (detector, generator, dataset)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from loguru import logger

from hbench.config import ExperimentCfg
from hbench.cv import outer_folds, run_outer_fold
from hbench.io import write_parquet_atomic
from hbench.paths import RunPaths
from hbench.pool.registry import get_detector, resolve_grid
from hbench.pool.writer import read_features
from hbench.seeding import set_all_seeds


def cell_labels(paths: RunPaths, g: str, d: str) -> pd.DataFrame:
    lab = pd.read_parquet(paths.labels(g, d))
    lab = lab[(lab["status"] == "kept") & ~lab["empty_answer"]]
    return lab.sort_values("sample_id").reset_index(drop=True)


def cell_folds(cfg: ExperimentCfg, paths: RunPaths, g: str, d: str, lab: pd.DataFrame) -> pd.DataFrame:
    """Outer folds of the cell, made once and shared by all detectors (paired design)."""
    p = paths.folds(g, d)
    if p.exists():
        f = pd.read_parquet(p)
        if set(f["sample_id"]) != set(lab["sample_id"]):
            raise RuntimeError(f"{p}: the labelled sample set changed since the folds were made")
        return f
    y = lab["y"].astype(int).to_numpy()
    rows = []
    for b in range(cfg.cv.repeats):
        fold = outer_folds(y, cfg.cv.outer_folds, cfg.seed, b, where=f"{g}/{d} repeat {b}")
        rows.append(pd.DataFrame({"sample_id": lab["sample_id"], "repeat": b, "fold": fold}))
    f = pd.concat(rows, ignore_index=True)
    write_parquet_atomic(f, p)
    return f


def run(cfg: ExperimentCfg, detectors: list[str] | None = None) -> None:
    paths = RunPaths(cfg)
    for g in cfg.generators:
        for d in cfg.datasets:
            lab = cell_labels(paths, g.name, d.name)
            folds = cell_folds(cfg, paths, g.name, d.name, lab)
            y = lab["y"].astype(int).to_numpy()
            for det in cfg.detectors:
                if detectors and det.name not in detectors:
                    continue
                mpath = paths.metrics(det.name, g.name, d.name)
                if mpath.exists():
                    logger.info(f"evaluate: {det.name} on {g.name}/{d.name} already done")
                    continue
                spec = get_detector(det.name)
                ex = spec.extractor()
                X_all, _, sids = read_features(paths.features(g.name, d.name) / ex.name)
                pos = {s: i for i, s in enumerate(sids)}
                missing = [s for s in lab["sample_id"] if s not in pos]
                if missing:
                    raise RuntimeError(f"{det.name}: {len(missing)} labelled samples have no features, e.g. {missing[:3]}")
                X = X_all[[pos[s] for s in lab["sample_id"]]]
                grid = resolve_grid(spec, det.grid, n_layers=X.shape[1])
                fixed = dict(det.params)

                def make_clf(**hp):
                    return spec.classifier(**fixed, **hp, seed=cfg.seed)

                mrows, prow = [], []
                for b in range(cfg.cv.repeats):
                    fb = folds[folds["repeat"] == b].set_index("sample_id").loc[lab["sample_id"], "fold"].to_numpy()
                    for f in range(cfg.cv.outer_folds):
                        set_all_seeds(cfg.seed)
                        tr, te = np.where(fb != f)[0], np.where(fb == f)[0]
                        where = f"{det.name} on {g.name}/{d.name} repeat {b} fold {f}"
                        r = run_outer_fold(X, y, tr, te, make_clf, grid, cfg.cv.inner_folds, cfg.seed + 1000 * b + f, cfg.cv.max_threshold_candidates, where)
                        mrows.append(
                            {
                                "detector": det.name,
                                "generator": g.name,
                                "dataset": d.name,
                                "repeat": b,
                                "fold": f,
                                "n_train": len(tr),
                                "n_test": len(te),
                                "params": json.dumps(r.params),
                                "inner_auroc": r.inner_auroc,
                                "tau": r.tau,
                                "inner_mcc": r.inner_mcc,
                                **r.metrics,
                                "flags": r.flags,
                            }
                        )
                        prow.append(pd.DataFrame({"sample_id": lab["sample_id"].iloc[te].values, "repeat": b, "fold": f, "y": y[te], "score": r.scores, "pred": (r.scores >= r.tau).astype(int)}))
                write_parquet_atomic(pd.concat(prow, ignore_index=True), paths.predictions(det.name, g.name, d.name))
                write_parquet_atomic(pd.DataFrame(mrows), mpath)  # written last: marks the unit as done
                m = pd.DataFrame(mrows)
                logger.info(f"evaluate: {det.name} on {g.name}/{d.name}: AUROC {m['auroc'].mean():.3f} ± {m['auroc'].std():.3f}, MCC {m['mcc'].mean():.3f}")
