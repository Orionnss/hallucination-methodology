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


def normalise_answers(value: Any, split: str | None = None) -> list[str]:
    """`expected_answer` can hold more than one acceptable answer. Return a list of non-empty strings,
    without duplicates, in the original order."""
    if value is None or (isinstance(value, float) and value != value):  # missing (None / NaN)
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
    if split:
        items = [part for x in items for part in str(x).split(split)]
    out: list[str] = []
    for x in items:
        if x is None or (isinstance(x, float) and x != x):  # missing value (NaN)
            continue
        x = str(x).strip()
        if x and x not in out:
            out.append(x)
    return out


def _raw_rows(cfg: DatasetCfg) -> list[dict]:
    if cfg.loader == "hf":
        from datasets import load_dataset

        ds = load_dataset(cfg.path, cfg.subset, split=cfg.split, revision=cfg.revision)
        return list(ds)
    if cfg.loader == "parquet":
        return pd.read_parquet(cfg.path).to_dict("records")
    if cfg.loader == "csv":
        return pd.read_csv(cfg.path).to_dict("records")
    if cfg.loader == "jsonl":
        return pd.read_json(cfg.path, lines=True).to_dict("records")
    raise ValueError(cfg.loader)


def _answers(r: Any, cfg: DatasetCfg) -> list[str]:
    fields = cfg.answer_field if isinstance(cfg.answer_field, list) else [cfg.answer_field]
    out: list[str] = []
    for f in fields:
        for a in normalise_answers(_get_path(r, f), cfg.answer_split):
            if a not in out:
                out.append(a)
    return out


def load_dataset_table(cfg: DatasetCfg) -> tuple[pd.DataFrame, dict]:
    """Return the normalised table and loading stats (rows read, duplicates and empty answers dropped)."""
    rows = _raw_rows(cfg)
    recs = []
    for i, r in enumerate(rows):
        sid = str(_get_path(r, cfg.id_field)) if cfg.id_field else str(i)
        recs.append({"sample_id": sid, "question": str(_get_path(r, cfg.question_field)), "expected_answer": _answers(r, cfg)})
    df = pd.DataFrame(recs)
    stats = {"rows_read": len(df), "duplicates_dropped": 0, "empty_answers_dropped": 0}
    dup = df["sample_id"].duplicated(keep=False)
    if dup.any():
        if not cfg.dedupe_ids:
            ex = df.loc[dup, "sample_id"].head(5).tolist()
            raise ValueError(f"dataset {cfg.name}: duplicate sample IDs, e.g. {ex}; set dedupe_ids or use another id_field")
        # Identical duplicates keep their first row; an ID whose rows disagree (question or answers)
        # is dropped entirely (D44, D48).
        g = df[dup].assign(_a=df.loc[dup, "expected_answer"].map(tuple)).groupby("sample_id")
        conflicting = g.filter(lambda x: x["question"].nunique() > 1 or x["_a"].nunique() > 1)["sample_id"].unique()
        before = len(df)
        df = df[~df["sample_id"].isin(conflicting)]
        stats["conflicting_ids_dropped"] = len(conflicting)
        stats["conflicting_rows_dropped"] = before - len(df)
        before = len(df)
        df = df.drop_duplicates("sample_id", keep="first").reset_index(drop=True)
        stats["duplicates_dropped"] = before - len(df)
    empty = df["expected_answer"].map(len) == 0
    if empty.any():
        if not cfg.drop_empty_answers:
            raise ValueError(f"dataset {cfg.name}: {int(empty.sum())} rows have no expected answer")
        df = df[~empty].reset_index(drop=True)
        stats["empty_answers_dropped"] = int(empty.sum())
    return df, stats
