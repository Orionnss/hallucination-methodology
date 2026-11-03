"""Disk I/O. Tables are Parquet. A resumable table is a directory of append-only part files."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pandas as pd


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


class PartTable:
    """A table stored as `<dir>/part-XXXXX.parquet`. Each part is written atomically, so a crash
    loses at most the batch in flight. `done_keys()` lets a stage skip the work already done."""

    def __init__(self, directory: Path, key_col: str = "key"):
        self.dir = Path(directory)
        self.key_col = key_col

    def parts(self) -> list[Path]:
        return sorted(self.dir.glob("part-*.parquet")) if self.dir.exists() else []

    def exists(self) -> bool:
        return bool(self.parts())

    def read(self) -> pd.DataFrame:
        parts = self.parts()
        if not parts:
            return pd.DataFrame()
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        # A key written twice (e.g. a crash between two writers) keeps its first occurrence.
        return df.drop_duplicates(subset=[self.key_col], keep="first").reset_index(drop=True)

    def done_keys(self) -> set[str]:
        parts = self.parts()
        if not parts:
            return set()
        keys: set[str] = set()
        for p in parts:
            keys.update(pd.read_parquet(p, columns=[self.key_col])[self.key_col].tolist())
        return keys

    def append(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        idx = len(self.parts())
        while (self.dir / f"part-{idx:05d}.parquet").exists():
            idx += 1
        write_parquet_atomic(df, self.dir / f"part-{idx:05d}.parquet")
