"""Stage `parity` (ADDING_A_METHOD.md §6, D65): for one (method, generator, dtype), run the authors'
original pipeline and ours on the first `n_parity` sampled questions of the first dataset, compare
in memory, and write only the result (errors, ρ) to runs/<run_id>/parity/."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import torch
from loguru import logger

from hbench.config import ExperimentCfg
from hbench.generation import build_prompt
from hbench.io import PartTable
from hbench.models import free_gpu, load_model
from hbench.paths import RunPaths
from hbench.pool.batching import reduce_device
from hbench.pool.spans import user_token_span
from hbench.stages.generate import decoding_hash, prompt_hash

METHODS = ("lapeigvals", "icr_probe", "charm")


def run(cfg: ExperimentCfg, method: str, generator: str, dtype: str, n_parity: int = 16, batch_size: int = 4) -> dict:
    if method not in METHODS:
        raise ValueError(f"no parity test for {method!r}; available: {METHODS}")
    paths = RunPaths(cfg)
    g = cfg.generator(generator)
    d = cfg.datasets[0]
    samples = pd.read_parquet(paths.samples(d.name)).head(n_parity)
    torch.set_float32_matmul_precision(cfg.extraction.matmul_precision)  # same setting for both pipelines (D29)
    lm = load_model(g.model_copy(update={"dtype": dtype, "device": cfg.extraction.device}), attn_implementation="eager")
    built = [
        build_prompt(lm, cfg.prompt.template.format(question=q), cfg.prompt.system, cfg.prompt.use_chat_template, g.chat_template_kwargs, template_date=g.template_date)
        for q in samples["question"]
    ]
    prompts = [ids for _, ids in built]
    # Stored answers (if generated already): samples whose original answer differs are excluded and counted.
    gens = PartTable(paths.generations(g.name, d.name)).read()
    stored = None
    if not gens.empty:
        key_prefix = {sid: f"{d.name}|{sid}|{g.name}|{prompt_hash(cfg, g)}|{decoding_hash(cfg)}" for sid in samples["sample_id"]}
        by_key = gens.set_index("key")["answer_ids"]
        stored = [list(by_key[key_prefix[s]]) if key_prefix[s] in by_key.index else None for s in samples["sample_id"]]
        if any(a is None for a in stored):
            stored = None
    mnt = cfg.decoding.max_new_tokens
    out_dev = reduce_device(cfg.extraction.reduce_device, lm.device)  # our side, as in extract (D84)
    if method == "lapeigvals":
        from hbench.parity.lapeigvals import parity

        r = parity(lm, prompts, stored, mnt, batch_size, dtype, out_dev)
    elif method == "icr_probe":
        from hbench.parity.icr import parity

        spans = [user_token_span(lm.tokenizer, t, ids, cfg.prompt.template.format(question=q)) for (t, ids), q in zip(built, samples["question"])]
        r = parity(lm, prompts, spans, stored, mnt, batch_size, dtype, out_dev)
    else:
        from hbench.parity.charm import parity

        r = parity(lm, prompts, stored, mnt, batch_size, dtype, out_dev)
    out = r.as_dict() | {"reduce_device": cfg.extraction.reduce_device,"generator": g.name, "hf_id": g.hf_id, "commit": lm.commit, "dataset": d.name, "date": date.today().isoformat(),
                          "matmul_precision": cfg.extraction.matmul_precision, "stored_answers_checked": stored is not None}
    p = paths.root / "parity" / f"{method}__{g.name}__{dtype}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    logger.info(f"parity {method} / {g.name} / {dtype}: {r.verdict} (max rel err {r.max_rel_err:.2e}, ρ {r.spearman:.4f}, excluded {r.n_excluded}"
                + (f", edges flipped {r.n_edges_flipped}" if r.n_edges_flipped is not None else "") + ")")
    del lm
    free_gpu()
    return out
