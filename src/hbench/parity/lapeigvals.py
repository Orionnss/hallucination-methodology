"""lapeigvals parity: their predict_with_llm + AttentionAndLaplacianDiagsFeatureStorage (original
capture, generate(), feature code; output in a temporary directory, deleted after) vs our
teacher-forced pass → to_generate_output → LapEigvalsStorage on the same answer tokens."""

from __future__ import annotations

import glob
import shutil
import tempfile

import numpy as np
import torch
from datasets import Dataset
from transformers.generation import GenerationConfig

import hbench.pool  # noqa: F401
from hallucinations.dirs import DatasetDir
from hallucinations.llm.feature_storage import AttentionAndLaplacianDiagsFeatureStorage
from hallucinations.llm.predict import predict_with_llm
from hbench.models import LoadedModel
from hbench.parity import ParityResult, compare
from hbench.pool.adapter import to_generate_output
from hbench.pool.batching import build_pass_batch, forward_pass
from hbench.pool.interfaces import Needs
from hbench.pool.methods.lapeigvals import LapEigvalsStorage


def _padded_dataset(prompts: list[list[int]], pad: int, batch_size: int) -> Dataset:
    """Left-pad each batch to its longest prompt, like their tokenizer(padding="longest") per batch."""
    ids, mask = [], []
    for s in range(0, len(prompts), batch_size):
        chunk = prompts[s : s + batch_size]
        w = max(map(len, chunk))
        ids += [[pad] * (w - len(p)) + p for p in chunk]
        mask += [[0] * (w - len(p)) + [1] * len(p) for p in chunk]
    ds = Dataset.from_dict({"input_ids": ids, "attention_mask": mask})
    ds.set_format("torch")
    return ds


def run_original(lm: LoadedModel, prompts: list[list[int]], max_new_tokens: int, batch_size: int):
    """Returns (laplacian diags per example, generated answer IDs per example incl. EOS)."""
    tmp = tempfile.mkdtemp(prefix="hbench-parity-lapeig-")
    try:
        storage = AttentionAndLaplacianDiagsFeatureStorage(DatasetDir(tmp), max_save_workers=1, pad_token_id=lm.tokenizer.pad_token_id, verbose=False)
        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens, do_sample=False, return_dict_in_generate=True, output_attentions=True,
            pad_token_id=lm.tokenizer.pad_token_id, eos_token_id=sorted(lm.eos_ids),
        )
        ds = _padded_dataset(prompts, lm.tokenizer.pad_token_id, batch_size)
        # Their DataLoader stacks rows; rows of one batch share a width by construction above.
        predict_with_llm(lm.model, lm.tokenizer, ds, gen_cfg, storage, batch_size=batch_size, num_proc=0)
        storage.flush()
        diags = torch.load(DatasetDir(tmp).laplacian_diags_file, weights_only=True)
        answers = []
        for f in sorted(glob.glob(f"{tmp}/attentions/batch_*.pt"), key=lambda p: int(p.rsplit("_", 1)[1][:-3])):
            st = torch.load(f, weights_only=False)
            W = st["input_length"]
            for row in st["generated_tokens"]:
                seq = row[W:].tolist()
                cut = next((k for k, t in enumerate(seq) if t in lm.eos_ids), None)
                answers.append(seq if cut is None else seq[: cut + 1])
        return [d.float().numpy() for d in diags], answers
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_ours(lm: LoadedModel, prompts: list[list[int]], answers: list[list[int]], batch_size: int, out_device: torch.device) -> list[np.ndarray]:
    st = LapEigvalsStorage(method_dir=None, feature_dtype="float32", tokenizer=lm.tokenizer)
    eos = lm.eos_ids
    for s in range(0, len(prompts), batch_size):
        p, a = prompts[s : s + batch_size], answers[s : s + batch_size]
        n = [len(x) - (1 if x and x[-1] in eos else 0) for x in a]
        keys = [str(i) for i in range(s, s + len(p))]
        acts, b = forward_pass(lm, build_pass_batch(keys, keys, p, a, n, lm.tokenizer.pad_token_id), Needs(attentions=True), out_device)
        gen = to_generate_output(b.input_ids, b.prompt_width, attentions=acts.attentions)
        st.update(gen, b.prompt_attention_mask, None, None, b.prompt_width, batch=b)
        del acts, gen
    diags, _, _ = st.collect()
    return [d.float().numpy() for d in diags]


def parity(lm: LoadedModel, prompts: list[list[int]], stored_answers: list[list[int]] | None, max_new_tokens: int, batch_size: int, dtype: str,
           out_device: torch.device = torch.device("cpu")) -> ParityResult:
    theirs, their_answers = run_original(lm, prompts, max_new_tokens, batch_size)
    keep = list(range(len(prompts)))
    if stored_answers is not None:  # samples whose original answer differs from the stored one are excluded and counted
        keep = [i for i in keep if their_answers[i] == list(stored_answers[i])]
    # Ours runs on all samples, batched exactly as theirs, and the exclusions are applied after: their
    # padding removal keeps a row's EOS unless the row is the longest of its batch (the last column
    # is dropped), so a feature depends on its batch (D85).
    ours = run_ours(lm, prompts, their_answers, batch_size, out_device)
    return compare("lapeigvals", dtype, [theirs[i] for i in keep], [ours[i] for i in keep], n_excluded=len(prompts) - len(keep))
