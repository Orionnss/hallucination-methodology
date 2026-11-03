"""Stage `aggregate` (prompt §2.4): labels ℓ_i, y_i and the labelling statistics for each (generator, dataset)."""

from __future__ import annotations

import pandas as pd
from loguru import logger

from hbench.config import ExperimentCfg
from hbench.io import PartTable, write_parquet_atomic
from hbench.judging.aggregate import (
    DISCARD_NO_VOTE,
    DISCARD_REFUSAL,
    DISCARD_TIE,
    KEPT,
    map_label,
    mean_pairwise_cohen_kappa,
)
from hbench.judging.judges import judge_prompt_hash
from hbench.judging.labels import PARSE_ERROR
from hbench.stages.judge import vote_key_suffix
from hbench.paths import RunPaths


def aggregate_cell(
    votes: pd.DataFrame, judges: list[str], gen_keys: list[str], empty_keys: set[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """votes: rows (gen_key, sample_id, judge, vote). Every judge must have voted on every generated sample."""
    wide = votes.pivot(index="gen_key", columns="judge", values="vote")
    missing_j = [j for j in judges if j not in wide.columns]
    if missing_j:
        raise RuntimeError(f"no votes from judges {missing_j}; run `judge` first")
    wide = wide.reindex(index=gen_keys, columns=judges)
    if wide.isna().any().any():
        n_missing = int(wide.isna().sum().sum())
        raise RuntimeError(f"{n_missing} votes are missing; run `judge` to completion first")
    sid = votes.drop_duplicates("gen_key").set_index("gen_key")["sample_id"]
    rows = []
    for k, r in wide.iterrows():
        vs = r.tolist()
        lab, y, status = map_label(vs)
        rows.append({"gen_key": k, "sample_id": sid[k], "label": lab, "y": y, "status": status, "votes": vs})
    lab_df = pd.DataFrame(rows)
    lab_df["y"] = lab_df["y"].astype("Int8")
    # An answer that is only EOS has no answer span: excluded for all detectors (DECISIONS.md D30).
    lab_df["empty_answer"] = lab_df["gen_key"].isin(empty_keys or set())
    n = len(lab_df)
    st = lab_df["status"].value_counts()
    labelled = lab_df[lab_df["status"] == KEPT]
    kept = labelled[~labelled["empty_answer"]]
    kappa, n_pairs = mean_pairwise_cohen_kappa({j: wide[j].tolist() for j in judges})
    stats = {
        "n": n,
        "n_kept": len(kept),
        "empty_answer_count": int(lab_df["empty_answer"].sum()),
        "empty_answer_kept_label": int(labelled["empty_answer"].sum()),
        "refusal_rate": st.get(DISCARD_REFUSAL, 0) / n,
        "tie_rate": st.get(DISCARD_TIE, 0) / n,
        "no_valid_vote_rate": st.get(DISCARD_NO_VOTE, 0) / n,
        "parse_error_count": int((wide == PARSE_ERROR).sum().sum()),
        "n_hallucination": int((kept["y"] == 1).sum()),
        "n_correct": int((kept["y"] == 0).sum()),
        "hallucination_ratio": float((kept["y"] == 1).mean()) if len(kept) else float("nan"),
        "mean_pairwise_cohen_kappa": kappa,
        "n_judge_pairs_kappa": n_pairs,
        "n_judges": len(judges),
    }
    return lab_df, stats


def run(cfg: ExperimentCfg) -> pd.DataFrame:
    paths = RunPaths(cfg)
    judges = [j.name for j in cfg.judges]
    jh = judge_prompt_hash(cfg)
    suffix = {j.name: vote_key_suffix(cfg, j, jh) for j in cfg.judges}
    all_stats = []
    for g in cfg.generators:
        for d in cfg.datasets:
            gens = PartTable(paths.generations(g.name, d.name)).read()
            votes = pd.concat([PartTable(paths.votes(j, g.name, d.name)).read() for j in judges], ignore_index=True)
            if gens.empty or votes.empty:
                raise RuntimeError(f"aggregate: missing generations or votes for {g.name}/{d.name}")
            # Only the votes on the current generations, with the current judge settings, count. Votes
            # made with older settings stay on disk (resume never overwrites) but are ignored here.
            votes = votes[votes["key"] == votes["gen_key"] + votes["judge"].map(suffix)]
            empty = set(gens.loc[gens["answer_n_tokens"] == 0, "key"])
            lab_df, stats = aggregate_cell(votes, judges, gens["key"].tolist(), empty)
            lab_df.insert(0, "generator", g.name)
            lab_df.insert(0, "dataset", d.name)
            write_parquet_atomic(lab_df, paths.labels(g.name, d.name))
            all_stats.append({"generator": g.name, "dataset": d.name, **stats})
            logger.info(f"aggregate: {g.name}/{d.name}: {stats}")
    out = pd.DataFrame(all_stats)
    write_parquet_atomic(out, paths.label_stats())
    return out
