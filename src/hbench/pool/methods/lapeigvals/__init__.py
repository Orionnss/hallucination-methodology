"""lapeigvals — "Hallucination Detection in LLMs Using Spectral Features of Attention Maps"
(https://arxiv.org/abs/2502.17598), code https://github.com/graphml-lab-pwr/lapeigvals @ 74f885c.
Card: CARD.md in this directory.

Integration case A0 (ADDING_A_METHOD.md §3): their `AttentionAndLaplacianDiagsFeatureStorage` runs
unchanged on the teacher-forced pass, converted by `to_generate_output()`. A thin subclass only
disables its disk writes (the per-batch `.pt` intermediate states and the final `torch.save`). Their
masks come from their own `get_token_masks`.

Follows the original: the Laplacian diagonal (= eigenvalues, lower-triangular), their padding removal,
their span (prompt + generated tokens except the last one), the per-head descending sort and top-k
(`features/attn_feats.py`), PCA + logistic regression with class_weight="balanced", max_iter 2000
(`probe_models/lr.py`).
Differs: see `deviations_from_original` in CARD.md (transformers 5.20.2, teacher-forced eager pass,
shard-wise collection, k and layer chosen in the inner CV, D70).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import hbench.pool  # noqa: F401  (lapeigvals on sys.path)
from hallucinations.dirs import DatasetDir
from hallucinations.llm.activation_storage import ActivationStorage
from hallucinations.llm.feature_storage import AttentionAndLaplacianDiagsFeatureStorage
from hallucinations.llm.predict import get_token_masks
from hallucinations.probe_models.commons import get_pca_dim
from hallucinations.probe_models.lr import DEFAULT_MAX_ITER
from hbench.pool.interfaces import Needs
from hbench.pool.writer import shard_path, write_shard

TOP_K_MAX = 100  # largest k of their grid TOP_K_EIGVALS = [5, 10, 25, 50, 100]
TOP_K_GRID = [5, 10, 25, 50, 100]
PCA_DIM = 512  # their main stage train_attn_vs_laplacian_pca (D70)


class _NoIntermediateWrites(AttentionAndLaplacianDiagsFeatureStorage):
    """Their storage with I/O disabled: no per-batch .pt files, no final torch.save. The feature
    calculations (update → extract_and_record_features_data) are inherited unchanged."""

    def save(self, intermediate_states: dict[str, Any], batch_idx: int) -> None:
        pass

    def _check_save_dir(self) -> None:
        pass

    def flush(self) -> None:
        self.save_executor.shutdown(wait=True)


def topk_eigvals(lap_diag: torch.Tensor, k: int = TOP_K_MAX) -> np.ndarray:
    """[L, H, T] Laplacian diagonal → [L, H, k] eigenvalues sorted descending, NaN-padded if T < k.
    The sort and the cut are the ones of their get_laplacian_eigvals_per_head_topk."""
    vals = lap_diag.float().sort(dim=-1, descending=True).values[:, :, :k].numpy()
    if vals.shape[-1] < k:
        pad = np.full(vals.shape[:-1] + (k - vals.shape[-1],), np.nan, dtype=vals.dtype)
        vals = np.concatenate([vals, pad], axis=-1)
    return vals


class LapEigvalsStorage(ActivationStorage):
    """PoolStorage member: one instance of their storage per shard, in a temporary directory that is
    removed at flush (their constructor creates empty directories; nothing else is written)."""

    def __init__(self, method_dir: Path, feature_dtype: str, tokenizer):
        self.method_dir = method_dir
        self.dtype = np.dtype(feature_dtype)
        self.tokenizer = tokenizer
        self.shard: int | None = None
        self._tmp: str | None = None
        self._theirs: _NoIntermediateWrites | None = None
        self._keys: list[str] = []
        self._sids: list[str] = []
        self._batch_idx = 0

    def _start(self) -> None:
        self._tmp = tempfile.mkdtemp(prefix="hbench-lapeig-")
        self._theirs = _NoIntermediateWrites(
            DatasetDir(self._tmp), max_save_workers=1, pad_token_id=self.tokenizer.pad_token_id, verbose=False
        )

    def update(self, outputs, attention_mask, special_token_mask, decoder_added_token_mask, input_length, **kwargs: Any) -> None:
        if self._theirs is None:
            self._start()
        batch = kwargs["batch"]
        masks = get_token_masks(outputs.sequences.cpu(), self.tokenizer)  # their masks
        self._theirs.update(
            outputs=outputs,
            attention_mask=attention_mask,
            special_token_mask=masks["special_token_mask"],
            decoder_added_token_mask=masks["decoder_added_token_mask"],
            input_length=input_length,
            batch_idx=self._batch_idx,
        )
        self._batch_idx += 1
        self._keys += batch.keys
        self._sids += batch.sample_ids

    def collect(self) -> tuple[list[torch.Tensor], list[str], list[str]]:
        """Their Laplacian diagonals ([L, H, T] per example) of the current shard, in memory (also used
        by the parity test). Resets the shard."""
        if self._theirs is None:
            return [], [], []
        self._theirs.flush()
        diags, keys, sids = self._theirs.laplacian_diags, self._keys, self._sids
        assert len(diags) == len(keys)
        shutil.rmtree(self._tmp, ignore_errors=True)
        self._theirs, self._tmp, self._keys, self._sids, self._batch_idx = None, None, [], [], 0
        return diags, keys, sids

    def flush(self) -> None:
        diags, keys, sids = self.collect()
        if not keys:
            return
        x = np.stack([topk_eigvals(d) for d in diags]).astype(self.dtype)
        write_shard(shard_path(self.method_dir, self.shard), x, keys, sids)


class LapEigvalsExtractor:
    name = "lapeigvals_laplacian_topk"
    needs = Needs(attentions=True, generate_format=True)
    fitted = False

    def make_storage(self, method_dir: Path, feature_dtype: str, tokenizer) -> ActivationStorage:
        return LapEigvalsStorage(method_dir, feature_dtype, tokenizer)


class LapEigvalsLR:
    """Their probe (probe_models/lr.py): PCA(get_pca_dim(512, X_train)) + LogisticRegression(
    class_weight="balanced", max_iter=2000). Their function trains and scores on a fixed split; this
    is the same sklearn pipeline as a fit/predict object. Input: [N, L, H, 100]; `layer` = "all" or an
    index; `k` = number of top eigenvalues per head."""

    name = "lapeigvals_lr"

    def __init__(self, k: int, layer: str | int = "all", pca_dim: int | None = PCA_DIM, seed: int = 0):
        self.k, self.layer, self.pca_dim, self.seed = k, layer, pca_dim, seed

    def _features(self, X: np.ndarray) -> np.ndarray:
        X = X[:, :, :, : self.k] if self.layer == "all" else X[:, int(self.layer) : int(self.layer) + 1, :, : self.k]
        X = X.reshape(len(X), -1)
        if np.isnan(X).any():
            raise ValueError(f"k = {self.k} exceeds the sequence length of some samples")
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LapEigvalsLR":
        Z = self._features(X)
        pca_dim = get_pca_dim(self.pca_dim, Z)
        lr = LogisticRegression(max_iter=DEFAULT_MAX_ITER, class_weight="balanced", random_state=self.seed)
        self.model = Pipeline([("pca", PCA(n_components=pca_dim, random_state=self.seed)), ("lr", lr)]) if pca_dim else lr
        self.model.fit(Z, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._features(X))[:, 1]


def adjust_grid(grid: dict, X: np.ndarray) -> dict:
    """k ≤ the shortest sequence of the cell (their rule: `eigval <= min(attn_diag.size(-1))`). Uses
    sequence lengths only, no labels. `layer`: "all" plus every single layer (D70)."""
    min_len = int((~np.isnan(X[:, 0, 0, :])).sum(axis=1).min())
    grid = dict(grid)
    grid["k"] = [k for k in grid["k"] if k <= min_len]
    if not grid["k"]:
        raise ValueError(f"lapeigvals: shortest sequence has {min_len} tokens < smallest k")
    if grid.get("layer") == "all_and_each":
        grid["layer"] = ["all"] + list(range(X.shape[1]))
    return grid
