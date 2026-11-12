"""Feature × classifier grid (D89): the diagonal reproduces the published detectors exactly; the
off-diagonal combinations are registered and run; the report section is built."""

import numpy as np
import pandas as pd
import pytest


def _data(shape, seed=0, n=80):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    X = rng.normal(size=(n, *shape)).astype(np.float32)
    X[y == 1] += 0.3
    return X, y


def test_diagonal_equals_published():
    from hbench.pool.combos import ComboClassifier, _views
    from hbench.pool.methods.icr_probe import ICRProbeClassifier
    from hbench.pool.methods.lapeigvals import LapEigvalsLR
    from hbench.pool.methods.saplma_reimpl import SaplmaReimplMLP

    v = _views()
    X, y = _data((3, 2, 10))
    for hp in ({"k": 5, "layer": "all"}, {"k": 10, "layer": 1}):
        a = ComboClassifier(v["lapeigvals_feat"], "pca_lr", seed=0, **hp).fit(X, y).predict_proba(X)
        b = LapEigvalsLR(seed=0, **hp).fit(X, y).predict_proba(X)
        np.testing.assert_array_equal(a, b)
    X, y = _data((4, 16))
    a = ComboClassifier(v["hidden_feat"], "saplma_mlp", seed=0, layer=2).fit(X, y).predict_proba(X)
    b = SaplmaReimplMLP(layer=2, seed=0).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(a, b)
    X, y = _data((6,))
    a = ComboClassifier(v["icr_feat"], "icr_mlp", seed=0).fit(X, y).predict_proba(X)
    b = ICRProbeClassifier(seed=0).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(a, b)


def test_registry_and_off_diagonal_run():
    from hbench.pool.registry import REGISTRY, resolve_grid

    cross = sorted(n for n, s in REGISTRY.items() if s.group == "cross")
    assert cross == sorted([
        "lapeigvals_feat+icr_mlp", "lapeigvals_feat+saplma_mlp", "icr_feat+pca_lr",
        "icr_feat+saplma_mlp", "hidden_feat+pca_lr", "hidden_feat+icr_mlp",
    ])
    shapes = {"lapeigvals_feat": (3, 2, 10), "icr_feat": (6,), "hidden_feat": (4, 16)}
    for name in cross:
        spec = REGISTRY[name]
        X, y = _data(shapes[name.split("+")[0]])
        grid = resolve_grid(spec, {}, n_layers=X.shape[1], X=X)
        hp = {k: v[0] for k, v in grid.items()}
        p = spec.classifier(**hp, seed=0).fit(X, y).predict_proba(X)
        assert p.shape == (len(y),) and np.all((p >= 0) & (p <= 1)), name
    assert resolve_grid(REGISTRY["hidden_feat+pca_lr"], {}, n_layers=4)["layer"] == [0, 1, 2, 3]


def test_cross_section(tmp_path):
    from hbench.stages.report import cross_section

    dets = ["lapeigvals", "icr_probe", "saplma_reimpl", "icr_feat+pca_lr", "hidden_feat+icr_mlp"]
    rows = [{"detector": d, "generator": "g", "dataset": c, "fold": f, "auroc": 0.6 + 0.05 * i + 0.01 * f, "mcc": 0.2 + 0.01 * i}
            for i, d in enumerate(dets) for c in ("a", "b") for f in range(3)]
    md = "\n".join(cross_section(pd.DataFrame(rows), [("g", "a"), ("g", "b")], tmp_path))
    assert "## Feature × classifier grid" in md and "icr_feat × pca_lr" in md and "icr_feat × icr_mlp †" in md
    mat = pd.read_csv(tmp_path / "cross_auroc_matrix.csv", index_col=0)
    assert mat.loc["icr_feat", "pca_lr"] == pytest.approx(0.6 + 0.05 * 3 + 0.01)
    assert np.isnan(mat.loc["lapeigvals_feat", "icr_mlp"])  # not evaluated → empty cell
    assert r"\textbf{" in (tmp_path / "cross_mcc.tex").read_text()
    assert cross_section(pd.DataFrame(rows[:6]), [("g", "a")], tmp_path) == []  # no combination → no section
