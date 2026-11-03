"""Dataset loaders. Each row → (sample_id, question, expected_answer: list[str]) (prompt §2.1)."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from hbench.config import DatasetCfg


def _get_path(row: Any, dotted: str) -> Any:
    cur = row
    for part in dotted.split("."):
        cur = cur[part]
    return cur


def normalise_answers(value: Any) -> list[str]:
    """`expected_answer` can hold more than one acceptable answer. Return a list of non-empty strings,
    without duplicates, in the original order."""
    if value is None:
        items: list[Any] = []
    elif isinstance(value, str):
        s = value.strip()
        items = [s]
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    items = parsed
            except json.JSONDecodeError:
                pass
    elif isinstance(value, dict):
        raise TypeError(f"expected_answer is a dict; point answer_field at the list inside it: {list(value)}")
    else:
        items = list(value)
    out: list[str] = []
    for x in items:
        x = str(x).strip()
        if x and x not in out:
            out.append(x)
    return out


def _raw_rows(cfg: DatasetCfg) -> list[dict]:
    if cfg.loader == "hf":
        from datasets import load_dataset

        ds = load_dataset(cfg.path, cfg.subset, split=cfg.split)
        return list(ds)
    if cfg.loader == "parquet":
        return pd.read_parquet(cfg.path).to_dict("records")
    if cfg.loader == "csv":
        return pd.read_csv(cfg.path).to_dict("records")
    if cfg.loader == "jsonl":
        return pd.read_json(cfg.path, lines=True).to_dict("records")
    raise ValueError(cfg.loader)


def load_dataset_table(cfg: DatasetCfg) -> pd.DataFrame:
    rows = _raw_rows(cfg)
    recs = []
    for i, r in enumerate(rows):
        sid = str(_get_path(r, cfg.id_field)) if cfg.id_field else str(i)
        recs.append(
            {
                "sample_id": sid,
                "question": str(_get_path(r, cfg.question_field)),
                "expected_answer": normalise_answers(_get_path(r, cfg.answer_field)),
            }
        )
    df = pd.DataFrame(recs)
    if df["sample_id"].duplicated().any():
        dup = df.loc[df["sample_id"].duplicated(), "sample_id"].head(5).tolist()
        raise ValueError(f"dataset {cfg.name}: duplicate sample IDs, e.g. {dup}")
    empty = df["expected_answer"].map(len) == 0
    if empty.any():
        raise ValueError(f"dataset {cfg.name}: {int(empty.sum())} rows have no expected answer")
    return df
