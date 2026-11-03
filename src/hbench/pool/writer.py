"""Feature shard files. Only computed features x go to disk (ADDING_A_METHOD.md §5):
`<features>/<method>/shard-XXXXX.h5` with datasets `x` [n, *shape], `key`, `sample_id`."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import h5py
import numpy as np

STR = h5py.string_dtype()


def shard_path(method_dir: Path, shard: int) -> Path:
    return method_dir / f"shard-{shard:05d}.h5"


def write_shard(path: Path, x: np.ndarray, keys: list[str], sample_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset("x", data=x)
        f.create_dataset("key", data=np.array(keys, dtype=object), dtype=STR)
        f.create_dataset("sample_id", data=np.array(sample_ids, dtype=object), dtype=STR)
    os.replace(tmp, path)


def read_features(method_dir: Path) -> tuple[np.ndarray, list[str], list[str]]:
    xs, keys, sids = [], [], []
    for p in sorted(method_dir.glob("shard-*.h5")):
        with h5py.File(p, "r") as f:
            xs.append(f["x"][()])
            keys += [k.decode() if isinstance(k, bytes) else k for k in f["key"][()]]
            sids += [k.decode() if isinstance(k, bytes) else k for k in f["sample_id"][()]]
    if not xs:
        raise FileNotFoundError(f"no feature shards in {method_dir}")
    return np.concatenate(xs), keys, sids


def write_manifest(path: Path, data: dict) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    os.replace(tmp, path)
