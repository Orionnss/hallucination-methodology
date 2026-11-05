"""Parity rules: scale-aware relative error (D83) and CHARM threshold flips (D82)."""

import numpy as np

from hbench.parity import compare
from hbench.parity.charm import _align_edges


def test_rel_err_floor_is_relative_to_sample_scale():
    theirs = [np.array([100.0, 0.0, 50.0])]
    ours = [np.array([100.0, 0.05, 50.0])]  # abs err 0.05 on a zero value; sample scale 100 → floor 0.1
    r = compare("m", "float32", theirs, ours)
    assert np.isclose(r.max_rel_err, 0.5)
    r = compare("m", "float32", [np.array([1.0, 2.0])], [np.array([1.0005, 2.0])])
    assert r.verdict == "pass"


def _graph(edges, attn):
    return {"edge_index": np.array(edges).T, "edge_attr": np.array(attn, dtype=np.float32)}


def test_charm_threshold_flip_tolerated():
    t = _graph([(0, 1), (1, 2)], [[0.3, 0.0], [0.052, 0.0]])
    o = _graph([(0, 1), (2, 1)], [[0.3, 0.0], [0.0, 0.055]])
    n, a, b = _align_edges(t, o)
    assert n == 2 and a.shape == (1, 2) and np.allclose(a, b)


def test_charm_structural_mismatch_fails():
    t = _graph([(0, 1), (1, 2)], [[0.3, 0.0], [0.2, 0.0]])
    o = _graph([(0, 1)], [[0.3, 0.0]])
    assert _align_edges(t, o) is None
