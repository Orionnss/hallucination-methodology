"""Constrained judge output for HF generation (DECISIONS.md D18): only the token sequences of
{"label": "CORRECT"}, {"label": "INCORRECT"}, {"label": "REFUSAL"}, then EOS, are allowed."""

from __future__ import annotations

import torch
from transformers import LogitsProcessor

from hbench.judging.labels import LABELS, label_json


class AllowedSequencesProcessor(LogitsProcessor):
    def __init__(self, sequences: list[list[int]], eos_id: int, prompt_width: int):
        self.trie: dict = {}
        for seq in sequences:
            node = self.trie
            for t in seq:
                node = node.setdefault(t, {})
            node[None] = {}  # end of an allowed sequence
        self.eos_id = eos_id
        self.prompt_width = prompt_width

    def _allowed(self, generated: list[int]) -> list[int]:
        node = self.trie
        for t in generated:
            if t not in node:
                return [self.eos_id]  # finished (EOS already produced) or off-trie: only EOS
            node = node[t]
        allowed = [t for t in node if t is not None]
        if None in node:
            allowed.append(self.eos_id)
        return allowed or [self.eos_id]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float("-inf"))
        for i in range(input_ids.shape[0]):
            allowed = self._allowed(input_ids[i, self.prompt_width :].tolist())
            mask[i, allowed] = 0.0
        return scores + mask


def label_token_sequences(tokenizer) -> list[list[int]]:
    seqs = [tokenizer(label_json(lab), add_special_tokens=False)["input_ids"] for lab in LABELS]
    assert len({tuple(s) for s in seqs}) == len(seqs)
    return seqs
