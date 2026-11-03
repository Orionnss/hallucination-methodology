"""sample → generate → judge → aggregate on a tiny CPU model, and resume behaviour."""

import pandas as pd

from hbench.config import load_config
from hbench.io import PartTable
from hbench.judging.labels import LABELS, PARSE_ERROR
from hbench.paths import RunPaths
from hbench.stages import aggregate, generate, judge, sample


def _gen(cfg, g="tiny0"):
    return PartTable(RunPaths(cfg).generations(g, "toy"))


def test_pipeline_end_to_end_with_mock_judges(make_cfg):
    cfg = load_config(make_cfg(n=12, generators=2))
    sample.run(cfg)
    generate.run(cfg)
    for g in ("tiny0", "tiny1"):
        df = _gen(cfg, g).read()
        assert len(df) == 12
        assert all(len(x) > 0 for x in df["prompt_ids"]) and all(len(x) > 0 for x in df["answer_ids"])
        assert set(df["finish_reason"]) <= {"eos", "length"}
    # Paired design: same sample IDs for all generators.
    assert set(_gen(cfg, "tiny0").read()["sample_id"]) == set(_gen(cfg, "tiny1").read()["sample_id"])
    judge.run(cfg)
    stats = aggregate.run(cfg)
    assert len(stats) == 2 and (stats["n"] == 12).all()
    labels = pd.read_parquet(RunPaths(cfg).labels("tiny0", "toy"))
    assert len(labels) == 12
    assert labels.loc[labels["status"] == "kept", "y"].isin([0, 1]).all()
    assert labels.loc[labels["status"] != "kept", "y"].isna().all()


def test_generate_resume(make_cfg):
    cfg = load_config(make_cfg(n=12))
    sample.run(cfg)
    generate.run(cfg)
    t = _gen(cfg)
    first = t.read().set_index("key")
    parts = t.parts()
    assert len(parts) == 3  # batch_size 4
    generate.run(cfg)  # nothing to do
    assert t.parts() == parts
    parts[1].unlink()  # simulate a crash that lost one batch
    generate.run(cfg)
    again = t.read().set_index("key")
    assert len(t.parts()) == 3 and sorted(again.index) == sorted(first.index)
    # Greedy decoding: the regenerated batch gives the same tokens.
    pd.testing.assert_series_equal(again.loc[first.index, "answer_ids"].map(tuple), first["answer_ids"].map(tuple))


def test_judge_resume_and_retry(make_cfg):
    p = make_cfg(n=8)
    cfg = load_config(p)
    sample.run(cfg)
    generate.run(cfg)
    keys = _gen(cfg).read()["key"].tolist()
    jcfg = [
        {"name": "mA", "hf_id": "mock", "kind": "mock", "mock": {"script": {keys[0]: "garbage", keys[1]: "INCORRECT"}}},
        {"name": "mB", "hf_id": "mock", "kind": "mock", "mock": {"parse_error_prob": 0.5, "seed": 7}},
    ]
    cfg = load_config(make_cfg(n=8, judges=jcfg))
    judge.run(cfg)
    va = PartTable(RunPaths(cfg).votes("mA", "tiny0", "toy")).read().set_index("gen_key")
    assert va.loc[keys[0], "vote"] == PARSE_ERROR and va.loc[keys[0], "attempts"] == 2
    assert va.loc[keys[1], "vote"] == "INCORRECT" and va.loc[keys[1], "attempts"] == 1
    vb = PartTable(RunPaths(cfg).votes("mB", "tiny0", "toy")).read()
    assert (vb["attempts"] == 2).any()  # some first attempts failed and were retried
    n_parts = len(PartTable(RunPaths(cfg).votes("mA", "tiny0", "toy")).parts())
    judge.run(cfg)  # resume: no new votes
    assert len(PartTable(RunPaths(cfg).votes("mA", "tiny0", "toy")).parts()) == n_parts


def test_hf_judge_constrained_output(make_cfg, tiny_models):
    jcfg = [{"name": "hfj", "hf_id": tiny_models["judge"], "dtype": "float32", "batch_size": 4, "device": "cpu"}]
    cfg = load_config(make_cfg(n=6, judges=jcfg))
    sample.run(cfg)
    generate.run(cfg)
    judge.run(cfg)
    v = PartTable(RunPaths(cfg).votes("hfj", "tiny0", "toy")).read()
    # A random tiny model still produces a valid label: the constraint allows nothing else.
    assert len(v) == 6 and v["vote"].isin(LABELS).all() and (v["attempts"] == 1).all()
    stats = aggregate.run(cfg)
    assert stats.loc[0, "parse_error_count"] == 0


def test_extra_samples_generated_once(make_cfg):
    cfg = load_config(make_cfg(n=4, extra_samples={"num_samples": 3, "decoding": {"do_sample": True, "temperature": 1.0, "max_new_tokens": 5}}))
    sample.run(cfg)
    generate.run(cfg)
    t = PartTable(RunPaths(cfg).extra_samples("tiny0", "toy"))
    df = t.read()
    assert len(df) == 4 and all(len(x) == 3 for x in df["extra_answer_ids"])
    parts = t.parts()
    generate.run(cfg)
    assert t.parts() == parts
