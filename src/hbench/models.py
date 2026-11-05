"""Load HF models and tokenizers (HF everywhere, DECISIONS.md D16)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import transformers
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
    dtype_kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    kwargs = {dtype_kw: _DTYPES[cfg.dtype], "revision": cfg.revision}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if cfg.quantization == "bnb4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
    if device.type == "cuda":
        # Place on the GPU at load time (needed for quantized checkpoints, e.g. bnb4 and gpt-oss MXFP4).
        kwargs["device_map"] = {"": device.index or 0}
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.hf_id, **kwargs)
    except ValueError:
        # Multimodal checkpoints (Gemma-3, Qwen3.5) are registered as image-text-to-text; the text
        # path is used only.
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(cfg.hf_id, **kwargs)
    if device.type != "cuda":
        model = model.to(device)
    model = model.eval()
    commit = getattr(model.config, "_commit_hash", None) or _snapshot_commit(cfg.hf_id, cfg.revision) or cfg.revision
    return LoadedModel(model=model, tokenizer=tok, commit=commit, device=device)


def free_gpu() -> None:
    """Release a model's GPU memory after its last reference is dropped (HF models hold reference cycles)."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _snapshot_commit(hf_id: str, revision: str | None) -> str | None:
    """Commit hash of the loaded snapshot, read from the HF cache path (transformers 5 no longer sets
    `config._commit_hash`). None for a local directory."""
    from pathlib import Path

    if Path(hf_id).is_dir():
        return None
    from huggingface_hub import try_to_load_from_cache

    p = try_to_load_from_cache(hf_id, "config.json", revision=revision)
    if not isinstance(p, str):
        return None
    parts = Path(p).parts
    return parts[parts.index("snapshots") + 1] if "snapshots" in parts else None
