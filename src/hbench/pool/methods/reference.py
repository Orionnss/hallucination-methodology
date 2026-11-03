"""The two reference detectors of prompt §3.3. They are not published methods; they exist to test
the pipeline from start to end.

1. `mean_hidden_probe`: logistic regression on the mean hidden state over the answer tokens at one
   layer. Layer and C are selected in the inner CV (D11, D27).
2. `saplma_style_probe`: an MLP in the style of SAPLMA (Azaria & Mitchell, 2023, "The Internal State
   of an LLM Knows When It's Lying", https://arxiv.org/abs/2304.13734) on the hidden state of the last
   answer token. Follows the paper: 256-128-64 ReLU layers, sigmoid output, Adam, lr 1e-3, BCE,
   5 epochs, batch 32 (D28). Differs from the paper: the layer is selected in the inner CV
   instead of fixed (D12); inputs are standardised inside the training fold; the statement is the
   generated answer in its chat context, not a standalone true/false statement.

The reducers keep every layer (hidden_states 1..L, embeddings excluded): x has shape [L, d]. The
layer choice and the scaler are fitted parts and belong to 𝒜_k (decision 4.A), so they run inside
the CV folds.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn

from hbench.pool.interfaces import Activations, Needs, PassBatch


class MeanAnswerHidden:
    name = "mean_answer_hidden"
    needs = Needs(hidden_states=True)
    fitted = False

    def reduce(self, acts: Activations, batch: PassBatch) -> np.ndarray:
        hs = torch.stack(acts.hidden_states[1:], dim=1)  # [B, L, S, d]
        m = batch.answer_mask[:, None, :, None].to(hs.dtype)
        mean = (hs.float() * m.float()).sum(dim=2) / batch.answer_len[:, None, None].float()
        return mean.cpu().numpy()


class LastAnswerHidden:
    name = "last_answer_hidden"
    needs = Needs(hidden_states=True)
    fitted = False

    def reduce(self, acts: Activations, batch: PassBatch) -> np.ndarray:
        hs = torch.stack(acts.hidden_states[1:], dim=1)  # [B, L, S, d]
        idx = torch.arange(hs.shape[0], device=hs.device)
        return hs[idx, :, batch.last_answer_pos].float().cpu().numpy()  # [B, L, d]


class LayerLogReg:
    name = "layer_logreg"

    def __init__(self, layer: int, C: float = 1.0, seed: int = 0):
        self.layer, self.C, self.seed = layer, C, seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LayerLogReg":
        Z = X[:, self.layer]
        self.scaler = StandardScaler().fit(Z)
        self.model = LogisticRegression(C=self.C, penalty="l2", solver="lbfgs", max_iter=2000, random_state=self.seed)
        self.model.fit(self.scaler.transform(Z), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X[:, self.layer]))[:, 1]


class SaplmaMLP:
    name = "saplma_mlp"

    def __init__(self, layer: int, seed: int = 0, epochs: int = 5, batch_size: int = 32, lr: float = 1e-3):
        self.layer, self.seed, self.epochs, self.batch_size, self.lr = layer, seed, epochs, batch_size, lr

    def _net(self, d: int) -> nn.Module:
        return nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SaplmaMLP":
        Z = X[:, self.layer]
        self.scaler = StandardScaler().fit(Z)
        Zt = torch.tensor(self.scaler.transform(Z), dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        g = torch.Generator().manual_seed(self.seed)
        torch.manual_seed(self.seed)
        self.net = self._net(Zt.shape[1])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()  # sigmoid output + BCE
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(Zt), generator=g)
            for s in range(0, len(Zt), self.batch_size):
                b = perm[s : s + self.batch_size]
                opt.zero_grad()
                loss_fn(self.net(Zt[b]).squeeze(-1), yt[b]).backward()
                opt.step()
        self.net.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Zt = torch.tensor(self.scaler.transform(X[:, self.layer]), dtype=torch.float32)
        with torch.no_grad():
            return torch.sigmoid(self.net(Zt).squeeze(-1)).numpy()
