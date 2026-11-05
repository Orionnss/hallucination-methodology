"""ICR Probe integration: parity with their ICRScore on generate() output, the feature and the probe."""

import numpy as np
import pytest

from hbench.config import ModelCfg
from hbench.models import load_model
from hbench.parity.icr import parity
from hbench.pool.methods.icr_probe import ICRProbeClassifier

PROMPTS = [[5, 6, 7, 8, 9, 10, 11, 12], [13, 14, 15, 21, 22], [16, 17, 18, 19, 20, 23]]
SPANS = [(2, 6), (1, 4), (0, 5)]


@pytest.fixture(scope="module")
def tiny_eager(tiny_models):
    return load_model(ModelCfg(name="t", hf_id=tiny_models["gen"], dtype="float32", device="cpu"), attn_implementation="eager")


def test_icr_parity_with_generate(tiny_eager):
    r = parity(tiny_eager, PROMPTS, SPANS, None, max_new_tokens=6, batch_size=3, dtype="float32")
    assert r.verdict == "pass" and r.n_samples == 3, r


def test_icr_probe_classifier_learns():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 12))
    y = (X[:, 3] + 0.3 * rng.normal(size=120) > 0).astype(int)
    clf = ICRProbeClassifier(seed=0, epochs=30).fit(X[:90], y[:90])
    p = clf.predict_proba(X[90:])
    assert p.shape == (30,) and ((p > 0.5) == y[90:]).mean() > 0.7
    p2 = ICRProbeClassifier(seed=0, epochs=30).fit(X[:90], y[:90]).predict_proba(X[90:])
    assert np.allclose(p, p2)  # seeded
