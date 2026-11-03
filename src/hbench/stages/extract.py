"""Stage `extract` (prompt §2.5, ADDING_A_METHOD.md §5). Reads the stored answers; never generates.
One teacher-forced HF forward pass per batch on (prompt + stored answer), with the union of the
needs of all methods. Each method's reducer runs on the GPU directly after the pass; then the
activations are deleted. Only feature shards and a manifest are written."""

from __future__ import annotations

import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from hbench.config import ExperimentCfg
from hbench.io import PartTable
from hbench.models import load_model
from hbench.paths import RunPaths
from hbench.pool.adapter import to_generate_output
from hbench.pool.batching import build_pass_batch, to_device
from hbench.pool.interfaces import Activations, Needs
from hbench.pool.registry import get_detector
from hbench.pool.storage import PoolStorage, ReducerStorage
from hbench.pool.writer import shard_path, write_manifest
from hbench.provenance import record_model_revision
from hbench.seeding import set_all_seeds


def _extractors(cfg: ExperimentCfg) -> dict[str, object]:
    """One extractor per distinct extractor name (two detectors can share features)."""
    out: dict[str, object] = {}
    for d in cfg.detectors:
        ex = get_detector(d.name).extractor()
        out.setdefault(ex.name, ex)
    return out


def run(cfg: ExperimentCfg, generators: list[str] | None = None, datasets: list[str] | None = None) -> None:
    paths = RunPaths(cfg)
    xcfg = cfg.extraction
    extractors = _extractors(cfg)
    if not extractors:
        raise ValueError("no detectors in the config")
    needs = Needs()
    for ex in extractors.values():
        needs = needs.union(ex.needs)
    # output_attentions needs eager attention (ADDING_A_METHOD.md §0, consequence 2).
    attn_impl = "eager" if needs.attentions else None
    for g in cfg.generators:
        if generators and g.name not in generators:
            continue
        lm = None
        for d in cfg.datasets:
            if datasets and d.name not in datasets:
                continue
            gens = PartTable(paths.generations(g.name, d.name)).read()
            if gens.empty:
                raise RuntimeError(f"extract: no generations for {g.name}/{d.name}")
            gens = gens.sort_values("key").reset_index(drop=True)
            # Empty answers (only EOS) have no answer span: excluded for all detectors (D30).
            gens = gens[gens["answer_n_tokens"] > 0].reset_index(drop=True)
            fdir = paths.features(g.name, d.name)
            n_shards = (len(gens) + xcfg.shard_size - 1) // xcfg.shard_size
            todo = [s for s in range(n_shards) if not all(shard_path(fdir / m, s).exists() for m in extractors)]
            if not todo:
                logger.info(f"extract: {g.name}/{d.name} already done")
                continue
            if lm is None:
                set_all_seeds(cfg.seed)
                torch.set_float32_matmul_precision(xcfg.matmul_precision)  # D29
                mcfg = g.model_copy(update={"dtype": xcfg.dtype or g.dtype, "device": xcfg.device})
                lm = load_model(mcfg, attn_implementation=attn_impl)
                record_model_revision(cfg, "generator", g.name, g.hf_id, lm.commit)
            storages = {m: ReducerStorage(ex, fdir / m, xcfg.feature_dtype) for m, ex in extractors.items()}
            pool = PoolStorage(storages)
            for s in todo:
                shard = gens.iloc[s * xcfg.shard_size : (s + 1) * xcfg.shard_size]
                pool.start_shard(s)
                _run_shard(lm, shard, pool, needs, xcfg.batch_size, f"extract {g.name}/{d.name} shard {s}")
                pool.flush()
            write_manifest(
                fdir / "manifest.json",
                {
                    "generator": g.name,
                    "hf_id": g.hf_id,
                    "commit": lm.commit,
                    "dataset": d.name,
                    "methods": sorted(extractors),
                    "n_samples": len(gens),
                    "n_shards": n_shards,
                    "shard_size": xcfg.shard_size,
                    "dtype": mcfg.dtype,
                    "feature_dtype": xcfg.feature_dtype,
                    "attn_implementation": attn_impl or "default",
                    "matmul_precision": xcfg.matmul_precision,
                },
            )
            logger.info(f"extract: {g.name}/{d.name}: {len(todo)} shard(s) written")
        del lm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_shard(lm, shard: pd.DataFrame, pool: PoolStorage, needs: Needs, bs: int, desc: str) -> None:
    pad = lm.tokenizer.pad_token_id
    for start in tqdm(range(0, len(shard), bs), desc=desc, leave=False):
        rows = shard.iloc[start : start + bs]
        batch = build_pass_batch(
            rows["key"].tolist(),
            rows["sample_id"].tolist(),
            [list(x) for x in rows["prompt_ids"]],
            [list(x) for x in rows["answer_ids"]],
            rows["answer_n_tokens"].tolist(),
            pad,
        )
        batch = to_device(batch, lm.device)
        with torch.no_grad():
            out = lm.model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                position_ids=batch.position_ids,
                output_hidden_states=needs.hidden_states,
                output_attentions=needs.attentions,
                use_cache=False,
            )
        acts = Activations(
            hidden_states=out.hidden_states if needs.hidden_states else None,
            attentions=out.attentions if needs.attentions else None,
            logits=out.logits if needs.logits else None,
        )
        gen_out = (
            to_generate_output(batch.input_ids, batch.prompt_width, acts.hidden_states, acts.attentions, acts.logits)
            if needs.generate_format
            else None
        )
        pool.update(
            gen_out,
            attention_mask=batch.prompt_attention_mask,
            special_token_mask=None,
            decoder_added_token_mask=None,
            input_length=batch.prompt_width,
            acts=acts,
            batch=batch,
        )
        del out, acts, gen_out
