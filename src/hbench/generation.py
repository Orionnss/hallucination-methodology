"""Prompt building and batched HF generation, shared by the generate and judge stages."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import LogitsProcessorList

from hbench.config import DecodingCfg
from hbench.models import LoadedModel


def build_prompt(lm: LoadedModel, user_text: str, system: str | None, use_chat_template: bool) -> tuple[str, list[int]]:
    """Return (prompt_text, prompt_ids). With the chat template, the IDs come from the template
    directly, so special tokens such as BOS are not added twice."""
    tok = lm.tokenizer
    if use_chat_template:
        if tok.chat_template is None:
            raise ValueError(f"{tok.name_or_path} has no chat template; set prompt.use_chat_template: false")
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user_text}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        return text, list(ids)
    text = f"{system}\n\n{user_text}" if system else user_text
    return text, tok(text)["input_ids"]


@dataclass
class GenOut:
    answer_ids: list[int]  # generated tokens up to and including the first EOS; no padding
    finish_reason: str  # "eos" | "length"


def generate_batch(
    lm: LoadedModel,
    prompts: list[list[int]],
    decoding: DecodingCfg,
    logits_processor: LogitsProcessorList | None = None,
    num_return_sequences: int = 1,
    generator_seed: int | None = None,
) -> list[list[GenOut]]:
    tok = lm.tokenizer
    pad = tok.pad_token_id
    width = max(len(p) for p in prompts)
    input_ids = torch.full((len(prompts), width), pad, dtype=torch.long)
    attn = torch.zeros((len(prompts), width), dtype=torch.long)
    for i, p in enumerate(prompts):  # left padding
        input_ids[i, width - len(p) :] = torch.tensor(p)
        attn[i, width - len(p) :] = 1
    kwargs = dict(max_new_tokens=decoding.max_new_tokens, do_sample=decoding.do_sample, pad_token_id=pad)
    if decoding.do_sample:
        kwargs.update(temperature=decoding.temperature, top_p=decoding.top_p)
    else:
        kwargs.update(temperature=None, top_p=None, top_k=None)
    if generator_seed is not None:
        torch.manual_seed(generator_seed)
    with torch.no_grad():
        out = lm.model.generate(
            input_ids=input_ids.to(lm.device),
            attention_mask=attn.to(lm.device),
            logits_processor=logits_processor,
            num_return_sequences=num_return_sequences,
            **kwargs,
        )
    gen = out[:, width:].cpu().tolist()
    eos = lm.eos_ids
    results: list[list[GenOut]] = []
    for i in range(len(prompts)):
        row = []
        for r in range(num_return_sequences):
            seq = gen[i * num_return_sequences + r]
            cut = next((k for k, t in enumerate(seq) if t in eos), None)
            if cut is None:
                row.append(GenOut(answer_ids=seq, finish_reason="length"))
            else:
                row.append(GenOut(answer_ids=seq[: cut + 1], finish_reason="eos"))
        results.append(row)
    return results


def strip_eos(ids: list[int], eos: set[int]) -> list[int]:
    return ids[:-1] if ids and ids[-1] in eos else ids
