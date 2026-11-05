"""extract → evaluate on a tiny CPU model."""

import json

import h5py
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM

from hbench.config import load_config
from hbench.io import PartTable
from hbench.paths import RunPaths
from hbench.stages import aggregate, evaluate, extract, generate, judge, sample

DETECTORS = [{"name": "mean_hidden_probe", "grid": {"C": [0.1, 1.0]}}, {"name": "saplma_style_probe"}]
ONE_NOISY_JUDGE = [{"name": "mA", "hf_id": "mock", "kind": "mock", "mock": {"seed": 5, "flip_prob": 0.5}}]


def _cfg(make_cfg, n=80, **kw):
    return load_config(make_cfg(n=n, judges=ONE_NOISY_JUDGE, detectors=DETECTORS,
                                extraction={"batch_size": 8, "shard_size": 32, "device": "cpu"}, **kw))


def _upstream(cfg):
    sample.run(cfg)
    generate.run(cfg)
    judge.run(cfg)
    aggregate.run(cfg)


def test_extract_writes_only_features(make_cfg, tiny_models):
    cfg = _cfg(make_cfg)
    _upstream(cfg)
    extract.run(cfg)
    fdir = RunPaths(cfg).features("tiny0", "toy")
    files = sorted(p.relative_to(fdir).as_posix() for p in fdir.rglob("*") if p.is_file())
    assert "manifest.json" in files
    others = [f for f in files if f != "manifest.json"]
    assert others and all(f.split("/")[0] in {"mean_answer_hidden", "last_answer_hidden"} and f.split("/")[1].startswith("shard-") and f.endswith(".h5") for f in others)
    assert len(others) == 2 * 3  # 80 samples / shard size 32 → 3 shards per method
    # Nothing else under the run directory holds activations.
    assert not list(RunPaths(cfg).root.rglob("*.pt"))

    # Features equal an unbatched forward pass on (prompt + answer).
    gens = PartTable(RunPaths(cfg).generations("tiny0", "toy")).read().set_index("key")
    model = AutoModelForCausalLM.from_pretrained(tiny_models["gen"]).eval()
    with h5py.File(fdir / "mean_answer_hidden" / "shard-00000.h5") as fm, h5py.File(fdir / "last_answer_hidden" / "shard-00000.h5") as fl:
        xm, xl, keys = fm["x"][()], fl["x"][()], [k.decode() for k in fm["key"][()]]
    assert xm.shape[1:] == (2, 2)  # tiny-gpt2: 2 layers, d = 2
    for i in (0, 5, 17):
        r = gens.loc[keys[i]]
        p, a, n = list(r["prompt_ids"]), list(r["answer_ids"]), r["answer_n_tokens"]
        with torch.no_grad():
            hs = model(input_ids=torch.tensor([p + a]), output_hidden_states=True).hidden_states
        span = slice(len(p), len(p) + n)
        exp_mean = torch.stack([h[0, span].mean(0) for h in hs[1:]]).numpy()
        exp_last = torch.stack([h[0, len(p) + n - 1] for h in hs[1:]]).numpy()
        np.testing.assert_allclose(xm[i], exp_mean, atol=1e-5)
        np.testing.assert_allclose(xl[i], exp_last, atol=1e-5)

    # Resume: a lost shard is recomputed, the others are skipped.
    mtime = (fdir / "mean_answer_hidden" / "shard-00000.h5").stat().st_mtime_ns
    (fdir / "last_answer_hidden" / "shard-00002.h5").unlink()
    extract.run(cfg)
    assert (fdir / "last_answer_hidden" / "shard-00002.h5").exists()
    assert (fdir / "mean_answer_hidden" / "shard-00000.h5").stat().st_mtime_ns == mtime


def test_evaluate_paired_folds_and_resume(make_cfg):
    cfg = _cfg(make_cfg, n=100)
    _upstream(cfg)
    extract.run(cfg)
    evaluate.run(cfg)
    paths = RunPaths(cfg)
    folds = pd.read_parquet(paths.folds("tiny0", "toy"))
    lab = pd.read_parquet(paths.labels("tiny0", "toy"))
    kept = lab[(lab["status"] == "kept") & ~lab["empty_answer"]]
    assert set(folds["sample_id"]) == set(kept["sample_id"])
    preds = {d["name"]: pd.read_parquet(paths.predictions(d["name"], "tiny0", "toy")) for d in DETECTORS}
    a, b = (p.set_index("sample_id")["fold"].sort_index() for p in preds.values())
    assert a.equals(b)  # same outer folds for all detectors
    for d in DETECTORS:
        m = pd.read_parquet(paths.metrics(d["name"], "tiny0", "toy"))
        assert len(m) == 5 and m["auroc"].between(0, 1).all() and m["wacc"].between(0, 1).all()
        assert (m["n_train"] + m["n_test"] == len(kept)).all()
    mtime = paths.metrics("mean_hidden_probe", "tiny0", "toy").stat().st_mtime_ns
    evaluate.run(cfg)
    assert paths.metrics("mean_hidden_probe", "tiny0", "toy").stat().st_mtime_ns == mtime


def test_evaluate_reruns_when_labels_change(make_cfg):
    cfg = _cfg(make_cfg, n=100)
    _upstream(cfg)
    extract.run(cfg)
    evaluate.run(cfg, detectors=["mean_hidden_probe"])
    paths = RunPaths(cfg)
    m0 = pd.read_parquet(paths.metrics("mean_hidden_probe", "tiny0", "toy"))
    # Re-judge with other settings → other labels → folds and metrics are rebuilt.
    cfg2 = _cfg(make_cfg, n=100) .model_copy()
    cfg2.judges[0].mock = {"seed": 6, "flip_prob": 0.5}
    judge.run(cfg2)
    aggregate.run(cfg2)
    evaluate.run(cfg2, detectors=["mean_hidden_probe"])
    m1 = pd.read_parquet(paths.metrics("mean_hidden_probe", "tiny0", "toy"))
    assert m1["labels_hash"].iloc[0] != m0["labels_hash"].iloc[0]
    assert (pd.read_parquet(paths.folds("tiny0", "toy"))["labels_hash"] == m1["labels_hash"].iloc[0]).all()


def test_lapeigvals_in_pool(make_cfg):
    cfg = load_config(make_cfg(n=80, judges=ONE_NOISY_JUDGE, detectors=[{"name": "lapeigvals", "grid": {"k": [5, 10]}}, {"name": "mean_hidden_probe", "grid": {"C": [1.0]}}],
                               extraction={"batch_size": 8, "shard_size": 32, "device": "cpu"}))
    _upstream(cfg)
    extract.run(cfg)
    fdir = RunPaths(cfg).features("tiny0", "toy")
    files = sorted(p.relative_to(fdir).as_posix() for p in fdir.rglob("*") if p.is_file())
    assert all(f == "manifest.json" or f.endswith(".h5") for f in files)
    assert json.loads((fdir / "manifest.json").read_text())["attn_implementation"] == "eager"
    with h5py.File(fdir / "lapeigvals_laplacian_topk" / "shard-00000.h5") as f:
        assert f["x"].shape[1:] == (2, 2, 100)  # tiny-gpt2: 2 layers × 2 heads × top-100
    evaluate.run(cfg)
    m = pd.read_parquet(RunPaths(cfg).metrics("lapeigvals", "tiny0", "toy"))
    assert len(m) == 5 and m["auroc"].between(0, 1).all()
    assert {json.loads(p)["layer"] for p in m["params"]} <= {"all", 0, 1}
