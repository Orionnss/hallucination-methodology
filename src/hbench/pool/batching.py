"""Build a teacher-forced batch from the saved token IDs (DECISIONS.md D7). The layout and the
position IDs follow HF generate(), so the to_generate_output() adapter can rebuild its output."""

from __future__ import annotations

import torch

from hbench.pool.interfaces import Activations, Needs, PassBatch


def build_pass_batch(
    keys: list[str],
    sample_ids: list[str],
    prompt_ids: list[list[int]],
    answer_ids: list[list[int]],
    answer_n_tokens: list[int],
    pad_token_id: int,
    user_spans: list[tuple[int, int]] | None = None,
) -> PassBatch:
    B = len(keys)
    W = max(len(p) for p in prompt_ids)
    T = max(len(a) for a in answer_ids)
    ids = torch.full((B, W + T), pad_token_id, dtype=torch.long)
    prompt_mask = torch.zeros((B, W), dtype=torch.long)
    answer_mask = torch.zeros((B, W + T), dtype=torch.bool)
    for i, (p, a, n) in enumerate(zip(prompt_ids, answer_ids, answer_n_tokens)):
        ids[i, W - len(p) : W] = torch.tensor(p)
        ids[i, W : W + len(a)] = torch.tensor(a)
        prompt_mask[i, W - len(p) :] = 1
        answer_mask[i, W : W + n] = True
    # generate() extends the attention mask with ones for every new token (also after EOS).
    attn = torch.cat([prompt_mask, torch.ones((B, T), dtype=torch.long)], dim=1)
    # generate() position IDs: cumsum(mask) - 1, with 1 on masked positions.
    pos = attn.cumsum(-1) - 1
    pos.masked_fill_(attn == 0, 1)
    n = torch.tensor(answer_n_tokens)
    return PassBatch(
        keys=keys,
        sample_ids=sample_ids,
        input_ids=ids,
        attention_mask=attn,
        prompt_attention_mask=prompt_mask,
        position_ids=pos,
        prompt_width=W,
        answer_mask=answer_mask,
        answer_len=n,
        last_answer_pos=W + n - 1,
        prompt_len=[len(p) for p in prompt_ids],
        full_answer_len=[len(a) for a in answer_ids],
        user_span=list(user_spans) if user_spans is not None else [],
    )


def to_device(b: PassBatch, device: torch.device) -> PassBatch:
    for f in ("input_ids", "attention_mask", "prompt_attention_mask", "position_ids", "answer_mask", "answer_len", "last_answer_pos"):
        setattr(b, f, getattr(b, f).to(device))
    return b


def reduce_device(setting: str, model_device: torch.device) -> torch.device:
    """`extraction.reduce_device` → a device (D84)."""
    return torch.device("cpu") if setting == "cpu" else model_device


def forward_pass(lm, b: PassBatch, needs: Needs, out_device: torch.device) -> tuple[Activations, PassBatch]:
    """One teacher-forced forward pass. The outputs the detectors need, and the batch, are moved to
    `out_device` (D84). With a model split over GPUs each layer's outputs sit on its own GPU; the move
    gathers them, so reducers that stack layers see one device."""
    b = to_device(b, lm.device)
    with torch.no_grad():
        out = lm.model(
            input_ids=b.input_ids,
            attention_mask=b.attention_mask,
            position_ids=b.position_ids,
            output_hidden_states=needs.hidden_states,
            output_attentions=needs.attentions,
            use_cache=False,
        )
    acts = Activations(
        hidden_states=tuple(h.to(out_device) for h in out.hidden_states) if needs.hidden_states else None,
        attentions=tuple(a.to(out_device) for a in out.attentions) if needs.attentions else None,
        logits=out.logits.to(out_device) if needs.logits else None,
    )
    del out
    return acts, to_device(b, out_device)
