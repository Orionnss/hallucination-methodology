"""Load HF models and tokenizers (HF everywhere, DECISIONS.md D16)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from hbench.config import ModelCfg

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass
class LoadedModel:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    commit: str | None
    device: torch.device

    @property
    def eos_ids(self) -> set[int]:
        eos = self.model.generation_config.eos_token_id
        ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        return ids


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_model(cfg: ModelCfg, attn_implementation: str | None = None) -> LoadedModel:
    device = resolve_device(cfg.device)
    tok = AutoTokenizer.from_pretrained(cfg.hf_id, revision=cfg.revision)
    # Left padding for batched generation; the pad positions are removed before anything is saved.
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kwargs = {"torch_dtype": _DTYPES[cfg.dtype], "revision": cfg.revision}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(cfg.hf_id, **kwargs).to(device).eval()
    commit = getattr(model.config, "_commit_hash", None) or cfg.revision
    return LoadedModel(model=model, tokenizer=tok, commit=commit, device=device)
