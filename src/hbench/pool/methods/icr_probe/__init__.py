"""ICR Probe — "ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs"
(ACL 2025, https://arxiv.org/abs/2507.16488), code https://github.com/XavierZhang2002/ICR_Probe @ 40ec490.
Card: CARD.md in this directory.

Score (case B): their `ICRScore` (src/icr_score.py) runs unchanged, one example at a time (their code
indexes batch element 0), on the generate-format output of the teacher-forced pass for that example
without padding. Settings follow the paper (D67): top_k = 20, top_p = None, pooling = "mean",
use_induction_head = False (all heads averaged, Eq. 5). Core positions: D69.
Feature: token-wise mean of the [L × N] ICR matrix → [L] (paper §4, Algorithm 2 line 10, App. B.4).
Classifier (case A1 for the model): their `ICRProbe` MLP (src/utils.py), L→128→64→32→1 with batch
norm, dropout 0.3, LeakyReLU, sigmoid, Kaiming init. Their `ICRProbeTrainer` does not run
(missing `_load_data` / `_create_data_loaders`), so the training loop is ours, from paper App. B.4
(D68): BCE, Adam lr 5e-4, 50 epochs, batch 32, ReduceLROnPlateau(factor 0.5, patience 5) on the
validation loss, best-validation-loss epoch kept; validation = 20 % stratified split of the training
fold (as their `test_size = 0.2`).
Differs from the paper: we predict P(hallucination) (the paper's output is P(faithful); same BCE);
a final training batch of one example is dropped (batch norm needs ≥ 2).
"""

from __future__ import annotations

import contextlib
import copy

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from hbench.pool.adapter import to_generate_output
from hbench.pool.interfaces import Activations, Needs, PassBatch
from hbench.pool.thirdparty import THIRD_PARTY, load_package_module

ICR_DIR = THIRD_PARTY / "icr_probe" / "src"
ICR_SETTINGS = dict(top_k=20, top_p=None, pooling="mean", attention_uniform=False, hidden_uniform=False, use_induction_head=False)


def _icr():
    return load_package_module("icr_probe_src", ICR_DIR, "icr_score")


def _utils():
    return load_package_module("icr_probe_src", ICR_DIR, "utils")


@contextlib.contextmanager
def _cpu_safe_cuda_device(device: torch.device):
    """Their code wraps empty_cache() in `torch.cuda.device(icr_device)`, which raises for a CPU
    device. On CPU (tests) this context manager is replaced by a no-op; on GPU nothing changes."""
    if device.type == "cuda":
        yield
        return
    mod = _icr()
    real = mod.torch.cuda.device
    mod.torch.cuda.device = lambda *_a, **_k: contextlib.nullcontext()
    try:
        yield
    finally:
        mod.torch.cuda.device = real


def icr_matrix(acts: Activations, batch: PassBatch, i: int) -> np.ndarray:
    """ICR scores [L, N_out] of example i (their ICRScore.compute_icr)."""
    sl = batch.example_slice(i)
    P = batch.prompt_len[i]
    seq = batch.input_ids[i : i + 1, sl]
    hs = tuple(h[i : i + 1, sl] for h in acts.hidden_states)
    att = tuple(a[i : i + 1, :, sl, sl] for a in acts.attentions)
    gen = to_generate_output(seq, P, hidden_states=hs, attentions=att)
    u0, u1 = batch.user_span[i]
    device = seq.device
    with _cpu_safe_cuda_device(device):
        calc = _icr().ICRScore(
            hidden_states=gen.hidden_states,
            attentions=gen.attentions,
            skew_threshold=0,
            entropy_threshold=1e5,
            core_positions={"user_prompt_start": u0, "user_prompt_end": u1, "response_start": P},
            icr_device=device,
        )
        scores, _ = calc.compute_icr(**ICR_SETTINGS)
    return np.asarray(scores, dtype=np.float64)


class ICRFeature:
    name = "icr_mean"
    needs = Needs(hidden_states=True, attentions=True, user_span=True)
    fitted = False

    def reduce(self, acts: Activations, batch: PassBatch) -> np.ndarray:
        out = []
        for i in range(len(batch.keys)):
            m = icr_matrix(acts, batch, i)
            if m.shape[1] == 0:
                raise ValueError(f"{batch.keys[i]}: no answer token for ICR (answer has one token and no EOS)")
            out.append(m.mean(axis=1))  # token-wise mean → [L]
        return np.stack(out)


class ICRProbeClassifier:
    name = "icr_probe_mlp"

    def __init__(self, seed: int = 0, lr: float = 5e-4, epochs: int = 50, batch_size: int = 32, val_size: float = 0.2,
                 lr_factor: float = 0.5, lr_patience: int = 5):
        self.seed, self.lr, self.epochs, self.batch_size, self.val_size = seed, lr, epochs, batch_size, val_size
        self.lr_factor, self.lr_patience = lr_factor, lr_patience

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ICRProbeClassifier":
        torch.manual_seed(self.seed)
        tr, va = train_test_split(np.arange(len(y)), test_size=self.val_size, stratify=y, random_state=self.seed)
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        self.net = _utils().ICRProbe(input_dim=X.shape[1])
        crit = nn.BCELoss()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=self.lr_factor, patience=self.lr_patience)
        g = torch.Generator().manual_seed(self.seed)
        best, best_state = np.inf, None
        for _ in range(self.epochs):
            self.net.train()
            perm = torch.tensor(tr)[torch.randperm(len(tr), generator=g)]
            for s in range(0, len(perm), self.batch_size):
                b = perm[s : s + self.batch_size]
                if len(b) < 2:
                    continue  # batch norm needs ≥ 2 examples
                opt.zero_grad()
                crit(self.net(Xt[b]), yt[b].unsqueeze(1)).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                val_loss = float(crit(self.net(Xt[va]), yt[va].unsqueeze(1)))
            if val_loss < best:
                best, best_state = val_loss, copy.deepcopy(self.net.state_dict())
            sched.step(val_loss)
        self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.net(torch.tensor(X, dtype=torch.float32)).squeeze(-1).numpy()
