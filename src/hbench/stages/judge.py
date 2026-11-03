"""Stage `judge` (prompt §2.4): each judge J_r gives one vote in {CORRECT, INCORRECT, REFUSAL}
for each (generator, dataset, sample). If parsing fails, retry once; if it fails again, the vote
is PARSE_ERROR."""

from __future__ import annotations

import pandas as pd
from loguru import logger

from hbench.config import ExperimentCfg
from hbench.io import PartTable
from hbench.judging.judges import JudgeItem, judge_prompt_hash, make_judge
from hbench.judging.labels import PARSE_ERROR, parse_vote
from hbench.paths import RunPaths
from hbench.provenance import record_model_revision
from hbench.seeding import set_all_seeds

CHUNK = 256  # items written per part file


def vote_with_retry(judge, items: list[JudgeItem], retries: int) -> list[tuple[str, str, int]]:
    """Return (vote, last raw output, attempts) for each item."""
    raws = judge.judge(items, attempt=0)
    result = [(parse_vote(r), r, 1) for r in raws]
    for attempt in range(1, retries + 1):
        bad = [i for i, (v, _, _) in enumerate(result) if v is None]
        if not bad:
            break
        again = judge.judge([items[i] for i in bad], attempt=attempt)
        for i, r in zip(bad, again):
            result[i] = (parse_vote(r), r, attempt + 1)
    return [(v if v is not None else PARSE_ERROR, r, a) for v, r, a in result]


def run(cfg: ExperimentCfg, judges: list[str] | None = None, generators: list[str] | None = None, datasets: list[str] | None = None) -> None:
    paths = RunPaths(cfg)
    jh = judge_prompt_hash(cfg)
    for jc in cfg.judges:
        if judges and jc.name not in judges:
            continue
        judge = None
        for g in cfg.generators:
            if generators and g.name not in generators:
                continue
            for d in cfg.datasets:
                if datasets and d.name not in datasets:
                    continue
                gens = PartTable(paths.generations(g.name, d.name)).read()
                if gens.empty:
                    raise RuntimeError(f"judge: no generations for {g.name}/{d.name}; run `generate` first")
                samples = pd.read_parquet(paths.samples(d.name))[["sample_id", "question", "expected_answer"]]
                gens = gens.merge(samples, on="sample_id", how="left", validate="one_to_one")
                gens["vote_key"] = gens["key"] + f"|{jc.name}:{jh}"
                table = PartTable(paths.votes(jc.name, g.name, d.name))
                todo = gens[~gens["vote_key"].isin(table.done_keys())]
                if todo.empty:
                    logger.info(f"judge: {jc.name} on {g.name}/{d.name} already done")
                    continue
                if judge is None:
                    set_all_seeds(cfg.seed)
                    judge = make_judge(cfg, jc)
                    record_model_revision(cfg, "judge", jc.name, jc.hf_id, judge.commit)
                for s in range(0, len(todo), CHUNK):
                    chunk = todo.iloc[s : s + CHUNK]
                    items = [
                        JudgeItem(key=r.key, question=r.question, answer=r.answer_text, expected=list(r.expected_answer))
                        for r in chunk.itertuples()
                    ]
                    votes = vote_with_retry(judge, items, cfg.judging.retries)
                    table.append(
                        pd.DataFrame(
                            {
                                "key": chunk["vote_key"].values,
                                "gen_key": chunk["key"].values,
                                "dataset": d.name,
                                "sample_id": chunk["sample_id"].values,
                                "generator": g.name,
                                "judge": jc.name,
                                "vote": [v for v, _, _ in votes],
                                "raw_output": [r for _, r, _ in votes],
                                "attempts": [a for _, _, a in votes],
                            }
                        )
                    )
                logger.info(f"judge: {jc.name} on {g.name}/{d.name}: {len(todo)} votes")
        del judge

