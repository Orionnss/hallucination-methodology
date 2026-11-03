"""to_generate_output(): convert a teacher-forced forward output into the per-step structure of
HF `GenerateDecoderOnlyOutput` (transformers 4.47.0), for original code that expects generate()
output (ADDING_A_METHOD.md §3, case B).

generate() structure, for a prompt of width W and T generated tokens (sequences: [B, W + T]):
- hidden_states: T steps. Step 0: (L + 1) × [B, W, d] (the whole prompt). Step k ≥ 1:
  (L + 1) × [B, 1, d] (the token at position W + k - 1). The last generated token is never fed back.
- attentions: T steps. Step 0: L × [B, H, W, W]. Step k ≥ 1: L × [B, H, 1, W + k].
- logits / scores: T × [B, V]; step k = the prediction made at position W + k - 1.
The slices are views of the forward tensors (no copy).
"""

from __future__ import annotations

import torch
from transformers.generation import GenerateDecoderOnlyOutput


def to_generate_output(
    sequences: torch.Tensor,
    input_length: int,
    hidden_states: tuple[torch.Tensor, ...] | None = None,
    attentions: tuple[torch.Tensor, ...] | None = None,
    logits: torch.Tensor | None = None,
) -> GenerateDecoderOnlyOutput:
    W = input_length
    T = sequences.shape[1] - W
    if T < 1:
        raise ValueError("no generated tokens")

    def steps_seq(x: torch.Tensor) -> list[torch.Tensor]:  # x: [B, S, ...] over sequence positions
        return [x[:, :W]] + [x[:, W + k - 1 : W + k] for k in range(1, T)]

    hs = None
    if hidden_states is not None:
        per_layer = [steps_seq(h) for h in hidden_states]
        hs = tuple(tuple(layer[k] for layer in per_layer) for k in range(T))
    att = None
    if attentions is not None:
        def steps_attn(a: torch.Tensor) -> list[torch.Tensor]:  # a: [B, H, S, S]
            return [a[:, :, :W, :W]] + [a[:, :, W + k - 1 : W + k, : W + k] for k in range(1, T)]

        per_layer = [steps_attn(a) for a in attentions]
        att = tuple(tuple(layer[k] for layer in per_layer) for k in range(T))
    lg = None
    if logits is not None:
        lg = tuple(logits[:, W + k - 1] for k in range(T))
    return GenerateDecoderOnlyOutput(
        sequences=sequences,
        scores=lg,
        logits=lg,
        attentions=att,
        hidden_states=hs,
        past_key_values=None,
    )
