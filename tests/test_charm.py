"""CHARM integration: parity with their capture + graph code, and the classifier wrapper."""

import numpy as np
import pytest

from hbench.config import ModelCfg
from hbench.models import load_model
from hbench.parity.charm import parity, run_original
from hbench.pool.methods.charm import CharmClassifier, act_layer

PROMPTS = [[5, 6, 7, 8, 9], [13, 14, 15], [16, 17, 18, 19]]  # short: tiny-gpt2 attends almost uniformly


@pytest.fixture(scope="module")
def tiny_eager(tiny_models):
    return load_model(ModelCfg(name="t", hf_id=tiny_models["gen"], dtype="float32", device="cpu"), attn_implementation="eager")


def test_charm_parity(tiny_eager):
    r = parity(tiny_eager, PROMPTS, None, max_new_tokens=5, batch_size=3, dtype="float32")
    assert r.verdict == "pass" and r.n_samples == 3, r


def test_act_layer_rule():
    assert act_layer(32) == 24 and act_layer(48) == 36 and act_layer(40) == 30


def test_classifier_uses_their_label_convention(tiny_eager):
    items, _ = run_original(tiny_eager, PROMPTS * 14, 5)
    items = [{**it, "head": np.zeros(it["x"].shape[1], int), "layer": np.zeros(it["x"].shape[1], int),
              "response_index": np.arange(3, it["x"].shape[0]), "prompt_len": np.int64(3), "act_layer": np.int64(2)} for it in items]
    X = np.empty(len(items), dtype=object)
    X[:] = items
    y = np.array([i % 2 for i in range(len(items))])
    clf = CharmClassifier(seed=0)
    clf.args.num_epochs = 2
    p = clf.fit(X, y).predict_proba(X)
    assert p.shape == (len(items),) and ((p >= 0) & (p <= 1)).all()
    assert "best_epoch" in clf.result
