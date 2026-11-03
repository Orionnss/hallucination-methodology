"""Stage `generate` (prompt §2.3): a_i = g(π(q_i); θ). Same π and θ for all generators.
Saves the prompt text, the answer text and the token IDs. Each answer is generated once
(ADDING_A_METHOD.md); later stages only read it. Extra samples (for methods that need them)
are made here too, with their own decoding config."""

from __future__ import annotations

import pandas as pd
from loguru import logger
from tqdm import tqdm

from hbench.config import ExperimentCfg
from hbench.generation import build_prompt, generate_batch, strip_eos
from hbench.io import PartTable
from hbench.keys import stable_hash, unit_key
from hbench.models import load_model
from hbench.paths import RunPaths
from hbench.provenance import record_model_revision
from hbench.seeding import set_all_seeds


def prompt_hash(cfg: ExperimentCfg) -> str:
    return stable_hash(cfg.prompt)


def decoding_hash(cfg: ExperimentCfg) -> str:
    return stable_hash(cfg.decoding)


def extra_decoding_hash(cfg: ExperimentCfg) -> str:
    return stable_hash({"extra": cfg.extra_samples.model_dump(mode="json"), "seed": cfg.seed})


def _todo(samples: pd.DataFrame, keys: list[str], done: set[str]) -> pd.DataFrame:
    s = samples.assign(key=keys)
    return s[~s["key"].isin(done)]


def run(cfg: ExperimentCfg, generators: list[str] | None = None, datasets: list[str] | None = None) -> None:
    paths = RunPaths(cfg)
    ph, dh = prompt_hash(cfg), decoding_hash(cfg)
    for g in cfg.generators:
        if generators and g.name not in generators:
            continue
        lm = None
        for d in cfg.datasets:
            if datasets and d.name not in datasets:
                continue
            samples = pd.read_parquet(paths.samples(d.name))
            table = PartTable(paths.generations(g.name, d.name))
            keys = [unit_key(d.name, sid, g.name, ph, dh) for sid in samples["sample_id"]]
            todo = _todo(samples, keys, table.done_keys())
            n_extra = cfg.extra_samples.num_samples
            extra_table = PartTable(paths.extra_samples(g.name, d.name))
            eh = extra_decoding_hash(cfg)
            extra_keys = [unit_key(d.name, sid, g.name, ph, eh) for sid in samples["sample_id"]]
            extra_todo = _todo(samples, extra_keys, extra_table.done_keys()) if n_extra else samples.iloc[:0]
            if todo.empty and extra_todo.empty:
                logger.info(f"generate: {g.name}/{d.name} already done")
                continue
            if lm is None:
                set_all_seeds(cfg.seed)
                lm = load_model(g)
                record_model_revision(cfg, "generator", g.name, g.hf_id, lm.commit)
            _generate_main(cfg, lm, g.name, d.name, todo, table, ph, dh, g.batch_size)
            if n_extra:
                _generate_extra(cfg, lm, g.name, d.name, extra_todo, extra_table, ph, eh, g.batch_size)
        del lm


def _prompts(cfg: ExperimentCfg, lm, questions: list[str]) -> list[tuple[str, list[int]]]:
    return [
        build_prompt(lm, cfg.prompt.template.format(question=q), cfg.prompt.system, cfg.prompt.use_chat_template)
        for q in questions
    ]


def _generate_main(cfg, lm, gname, dname, todo, table, ph, dh, bs) -> None:
    eos = lm.eos_ids
    for start in tqdm(range(0, len(todo), bs), desc=f"generate {gname}/{dname}"):
        chunk = todo.iloc[start : start + bs]
        built = _prompts(cfg, lm, chunk["question"].tolist())
        outs = generate_batch(lm, [ids for _, ids in built], cfg.decoding)
        rows = []
        for (_, r), (ptext, pids), [o] in zip(chunk.iterrows(), built, outs):
            rows.append(
                {
                    "key": r["key"],
                    "dataset": dname,
                    "sample_id": r["sample_id"],
                    "generator": gname,
                    "prompt_hash": ph,
                    "decoding_hash": dh,
                    "prompt_text": ptext,
                    "prompt_ids": pids,
                    "answer_ids": o.answer_ids,
                    "answer_n_tokens": len(strip_eos(o.answer_ids, eos)),  # answer span length, EOS excluded (D8)
                    "answer_text": lm.tokenizer.decode(strip_eos(o.answer_ids, eos), skip_special_tokens=True).strip(),
                    "finish_reason": o.finish_reason,
                }
            )
        table.append(pd.DataFrame(rows))


def _generate_extra(cfg, lm, gname, dname, todo, table, ph, eh, bs) -> None:
    eos = lm.eos_ids
    k = cfg.extra_samples.num_samples
    for start in tqdm(range(0, len(todo), bs), desc=f"extra {gname}/{dname}"):
        chunk = todo.iloc[start : start + bs]
        built = _prompts(cfg, lm, chunk["question"].tolist())
        # Seed from the first key of the batch, so a resumed run reproduces the same batches.
        seed = int(stable_hash([cfg.seed, chunk["key"].iloc[0]], 8), 16)
        outs = generate_batch(lm, [ids for _, ids in built], cfg.extra_samples.decoding, num_return_sequences=k, generator_seed=seed)
        rows = []
        for (_, r), (ptext, pids), samples in zip(chunk.iterrows(), built, outs):
            rows.append(
                {
                    "key": r["key"],
                    "dataset": dname,
                    "sample_id": r["sample_id"],
                    "generator": gname,
                    "prompt_ids": pids,
                    "extra_answer_ids": [o.answer_ids for o in samples],
                    "extra_answer_text": [
                        lm.tokenizer.decode(strip_eos(o.answer_ids, eos), skip_special_tokens=True).strip() for o in samples
                    ],
                }
            )
        table.append(pd.DataFrame(rows))
