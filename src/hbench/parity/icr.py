"""ICR parity: their ICRScore on the output of a real generate() call (batch size 1, as in their
README: cached hidden_states and attentions in generate format) vs ours (teacher-forced batch →
per-example adapter output → their ICRScore). Compares the [L × N] ICR matrices."""

from __future__ import annotations

import numpy as np
import torch

from hbench.models import LoadedModel
from hbench.parity import ParityResult, compare
from hbench.pool.batching import build_pass_batch, to_device
from hbench.pool.interfaces import Activations
from hbench.pool.methods.icr_probe import ICR_SETTINGS, _cpu_safe_cuda_device, _icr, icr_matrix


def run_original(lm: LoadedModel, prompts: list[list[int]], spans: list[tuple[int, int]], max_new_tokens: int):
    mats, answers = [], []
    for p, (u0, u1) in zip(prompts, spans):
        ids = torch.tensor([p], device=lm.device)
        with torch.no_grad():
            out = lm.model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens, do_sample=False,
                output_hidden_states=True, output_attentions=True, return_dict_in_generate=True,
                pad_token_id=lm.tokenizer.pad_token_id, eos_token_id=sorted(lm.eos_ids),
            )
        seq = out.sequences[0, len(p) :].tolist()
        cut = next((k for k, t in enumerate(seq) if t in lm.eos_ids), None)
        answers.append(seq if cut is None else seq[: cut + 1])
        with _cpu_safe_cuda_device(lm.device):
            calc = _icr().ICRScore(
                hidden_states=out.hidden_states, attentions=out.attentions, skew_threshold=0, entropy_threshold=1e5,
                core_positions={"user_prompt_start": u0, "user_prompt_end": u1, "response_start": len(p)}, icr_device=lm.device,
            )
            scores, _ = calc.compute_icr(**ICR_SETTINGS)
        mats.append(np.asarray(scores, dtype=np.float64))
    return mats, answers


def run_ours(lm: LoadedModel, prompts, answers, spans, batch_size: int) -> list[np.ndarray]:
    eos = lm.eos_ids
    mats = []
    for s in range(0, len(prompts), batch_size):
        p, a, sp = prompts[s : s + batch_size], answers[s : s + batch_size], spans[s : s + batch_size]
        n = [len(x) - (1 if x and x[-1] in eos else 0) for x in a]
        keys = [str(i) for i in range(s, s + len(p))]
        b = to_device(build_pass_batch(keys, keys, p, a, n, lm.tokenizer.pad_token_id, sp), lm.device)
        with torch.no_grad():
            out = lm.model(input_ids=b.input_ids, attention_mask=b.attention_mask, position_ids=b.position_ids,
                           output_hidden_states=True, output_attentions=True, use_cache=False)
        acts = Activations(hidden_states=out.hidden_states, attentions=out.attentions)
        mats += [icr_matrix(acts, b, i) for i in range(len(p))]
    return mats


def parity(lm: LoadedModel, prompts, spans, stored_answers, max_new_tokens: int, batch_size: int, dtype: str) -> ParityResult:
    theirs, their_answers = run_original(lm, prompts, spans, max_new_tokens)
    keep = [i for i in range(len(prompts)) if stored_answers is None or their_answers[i] == list(stored_answers[i])]
    ours = run_ours(lm, [prompts[i] for i in keep], [their_answers[i] for i in keep], [spans[i] for i in keep], batch_size)
    return compare("icr_probe", dtype, [theirs[i] for i in keep], ours, n_excluded=len(prompts) - len(keep))
