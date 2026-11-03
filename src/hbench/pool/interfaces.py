"""Detector plugin interface (prompt §3.3, ADDING_A_METHOD.md §5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch


@dataclass(frozen=True)
class Needs:
    hidden_states: bool = False
    attentions: bool = False
    logits: bool = False
    generate_format: bool = False  # the method's original code expects HF generate() output (adapter)
    extra_samples: int = 0

    def union(self, other: "Needs") -> "Needs":
        return Needs(
            hidden_states=self.hidden_states or other.hidden_states,
            attentions=self.attentions or other.attentions,
            logits=self.logits or other.logits,
            generate_format=self.generate_format or other.generate_format,
            extra_samples=max(self.extra_samples, other.extra_samples),
        )


@dataclass
class PassBatch:
    """One batch of the teacher-forced pass, laid out like HF generate() output: the prompt is
    left-padded to width W, the answer starts at position W and is right-padded (ADDING_A_METHOD.md §0)."""

    keys: list[str]
    sample_ids: list[str]
    input_ids: torch.Tensor  # [B, W + T]
    attention_mask: torch.Tensor  # [B, W + T]; 1 on prompt tokens and on the whole generated part
    prompt_attention_mask: torch.Tensor  # [B, W]; the mask that generate() received
    position_ids: torch.Tensor  # [B, W + T]; same rule as generate()
    prompt_width: int  # W
    answer_mask: torch.Tensor  # [B, W + T] bool; answer tokens, EOS excluded (D8)
    answer_len: torch.Tensor  # [B]
    last_answer_pos: torch.Tensor  # [B]; position of the last answer token (EOS excluded)


@dataclass
class Activations:
    hidden_states: tuple[torch.Tensor, ...] | None = None  # (L + 1) × [B, S, d]; index 0 = embeddings
    attentions: tuple[torch.Tensor, ...] | None = None  # L × [B, H, S, S]
    logits: torch.Tensor | None = None  # [B, S, V]
    extra: dict[str, Any] = field(default_factory=dict)


class FeatureExtractor(Protocol):
    """φ_k. In the shared pass the engine runs the model, so φ_k is a reducer on the activations
    (deviation from the prompt's `extract(model, tokenizer, batch)`, DECISIONS.md D34)."""

    name: str
    needs: Needs
    fitted: bool  # True → also fit(X_raw) / transform(X_raw), called inside the CV folds

    def reduce(self, acts: Activations, batch: PassBatch) -> np.ndarray:  # [B, *feature_shape]
        ...


class Classifier(Protocol):
    name: str
    param_grid: dict

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Classifier": ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # P(y = 1)
        ...
