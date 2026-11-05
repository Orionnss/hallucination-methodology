"""PoolStorage(ActivationStorage): the general form of lapeigvals' AttentionAndHiddenStatesFeatureStorage
(`hallucinations/llm/feature_storage.py`). It holds one storage per method and sends each update()
and flush() call to all of them (prompt §3.3, ADDING_A_METHOD.md §3 case B)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from torch import Tensor
from transformers.generation import GenerateDecoderOnlyOutput

import hbench.pool  # noqa: F401  (puts lapeigvals on sys.path)
from hallucinations.llm.activation_storage import ActivationStorage
from hbench.pool.interfaces import Activations, FeatureExtractor, PassBatch
from hbench.pool.writer import shard_path, write_ragged_shard, write_shard


class ReducerStorage(ActivationStorage):
    """Storage for a method written for this pool: runs its reducer on the GPU activations and keeps
    only x. flush() writes the current shard."""

    def __init__(self, extractor: FeatureExtractor, method_dir: Path, feature_dtype: str = "float32"):
        self.extractor = extractor
        self.method_dir = method_dir
        self.dtype = np.dtype(feature_dtype)
        self._x: list[np.ndarray] = []
        self._keys: list[str] = []
        self._sids: list[str] = []
        self.shard: int | None = None

    def update(
        self,
        outputs: GenerateDecoderOnlyOutput | None,
        attention_mask: Tensor,
        special_token_mask: Tensor | None,
        decoder_added_token_mask: Tensor | None,
        input_length: int,
        **kwargs: Any,
    ) -> None:
        acts: Activations = kwargs["acts"]
        batch: PassBatch = kwargs["batch"]
        x = self.extractor.reduce(acts, batch)
        assert len(x) == len(batch.keys)
        if getattr(self.extractor, "ragged", False):
            self._x.append(list(x))  # list of dicts of arrays
        else:
            self._x.append(np.asarray(x, dtype=self.dtype))
        self._keys += batch.keys
        self._sids += batch.sample_ids

    def flush(self) -> None:
        if self.shard is None or not self._keys:
            return
        if getattr(self.extractor, "ragged", False):
            items = [{k: np.asarray(v, dtype=self.dtype) if np.issubdtype(np.asarray(v).dtype, np.floating) else np.asarray(v) for k, v in it.items()}
                     for chunk in self._x for it in chunk]
            write_ragged_shard(shard_path(self.method_dir, self.shard), items, self._keys, self._sids)
        else:
            write_shard(shard_path(self.method_dir, self.shard), np.concatenate(self._x), self._keys, self._sids)
        self._x, self._keys, self._sids = [], [], []


class PoolStorage(ActivationStorage):
    def __init__(self, storages: dict[str, ActivationStorage]):
        super().__init__()
        self.storages = storages

    def start_shard(self, shard: int) -> None:
        for s in self.storages.values():
            if hasattr(s, "shard"):
                s.shard = shard

    def update(self, *args: Any, **kwargs: Any) -> None:
        for s in self.storages.values():
            s.update(*args, **kwargs)

    def flush(self) -> None:
        for s in self.storages.values():
            s.flush()
