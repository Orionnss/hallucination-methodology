"""lapeigvals integration: parity against their original pipeline (in memory), and the pool run."""

import numpy as np
import pytest
import torch

from hbench.config import ModelCfg
from hbench.models import load_model
from hbench.parity.lapeigvals import parity, run_original
from hbench.pool.methods.lapeigvals import LapEigvalsLR, adjust_grid, topk_eigvals

PROMPTS = [[5, 6, 7, 8, 9, 10, 11, 12], [13, 14, 15], [16, 17, 18, 19, 20]]


@pytest.fixture(scope="module")
def tiny_eager(tiny_models):
    return load_model(ModelCfg(name="t", hf_id=tiny_models["gen"], dtype="float32", device="cpu"), attn_implementation="eager")


def test_parity_with_original_pipeline(tiny_eager):
    r = parity(tiny_eager, PROMPTS, None, max_new_tokens=6, batch_size=2, dtype="float32")
    assert r.verdict == "pass" and r.max_rel_err <= 1e-3 and r.n_samples == 3, r


def test_parity_excludes_answers_that_differ(tiny_eager):
    _, answers = run_original(tiny_eager, PROMPTS, 6, 2)
    stored = [answers[0], [1, 2, 3], answers[2]]
    r = parity(tiny_eager, PROMPTS, stored, max_new_tokens=6, batch_size=2, dtype="float32")
    assert r.n_excluded == 1 and r.n_samples == 2 and r.verdict == "pass"


def test_topk_and_grid():
    d = torch.tensor([[[0.1, 0.5, 0.3]]])  # [L=1, H=1, T=3]
    v = topk_eigvals(d, k=5)
    assert np.allclose(v[0, 0, :3], [0.5, 0.3, 0.1]) and np.isnan(v[0, 0, 3:]).all()
    X = np.stack([topk_eigvals(torch.rand(2, 2, t), k=100) for t in (12, 30, 8)])
    g = adjust_grid({"k": [5, 10, 25], "layer": "all_and_each"}, X)
    assert g["k"] == [5] and g["layer"] == ["all", 0, 1]


def test_classifier_fits_and_respects_k():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2, 2, 100))
    y = (X[:, 0, 0, 0] > 0).astype(int)
    p = LapEigvalsLR(k=5, layer="all").fit(X, y).predict_proba(X)
    assert p.shape == (40,) and ((p > 0.5) == y).mean() > 0.8
    Xn = X.copy(); Xn[0, :, :, 50:] = np.nan
    with pytest.raises(ValueError, match="exceeds"):
        LapEigvalsLR(k=100).fit(Xn, y)
    assert LapEigvalsLR(k=10, layer=1).fit(Xn, y).predict_proba(Xn).shape == (40,)
