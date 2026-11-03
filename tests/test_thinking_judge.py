"""Two-phase thinking judge (DECISIONS.md D51) and judge-settings-aware aggregation."""

import pandas as pd

import hbench.judging.judges as J
from hbench.config import load_config
from hbench.generation import GenOut
from hbench.io import PartTable
from hbench.judging.labels import LABELS
from hbench.paths import RunPaths
from hbench.stages import aggregate, generate, judge, sample


def _thinker(tiny_models, budget=4, **kw):
    return {"name": "thinker", "hf_id": tiny_models["thinker"], "dtype": "float32", "batch_size": 4, "device": "cpu",
            "thinking": {"enabled": True, "max_tokens": budget}, **kw}


def test_forced_close_on_budget(make_cfg, tiny_models):
    cfg = load_config(make_cfg(n=6, judges=[_thinker(tiny_models, budget=4)]))
    sample.run(cfg)
    generate.run(cfg)
    judge.run(cfg)
    v = PartTable(RunPaths(cfg).votes("thinker", "tiny0", "toy")).read()
    assert v["vote"].isin(LABELS).all()
    # A random tiny model does not emit </think> in 4 tokens: every reasoning is closed for it.
    assert v["forced_close"].all() and (v["reasoning_tokens"] <= 4).all()
    assert v["raw_reasoning"].map(lambda r: isinstance(r, str)).all()


def test_natural_close_keeps_marker(make_cfg, tiny_models, monkeypatch):
    cfg = load_config(make_cfg(n=4, judges=[_thinker(tiny_models, budget=50)]))
    hj = J.HFJudge(cfg.judges[0], "Q: {question} A: {answer} Y: {answers}", 16)
    real = J.generate_batch
    seen = {}

    def fake(lm, prompts, decoding, logits_processor=None, stop_ids=None, **kw):
        if stop_ids:  # phase 1: pretend the model reasons 2 tokens then closes
            return [[GenOut(answer_ids=[11, 12, hj.marker_id], finish_reason="eos")] for _ in prompts]
        seen["prompts"] = prompts
        return real(lm, prompts, decoding, logits_processor=logits_processor, **kw)

    monkeypatch.setattr(J, "generate_batch", fake)
    items = [J.JudgeItem(key="k", question="q", answer="a", expected=["a"])]
    [out] = hj.judge(items, attempt=0)
    assert out.forced_close is False and out.reasoning_tokens == 2
    after = hj._tok("\n\n")
    assert seen["prompts"][0][-(3 + len(after)) :] == [11, 12, hj.marker_id] + after
    assert out.raw in {J.label_json(l) for l in LABELS}


def test_aggregate_ignores_votes_from_old_judge_settings(make_cfg, tiny_models):
    mock = {"name": "mA", "hf_id": "mock", "kind": "mock", "mock": {"seed": 1}}
    cfg = load_config(make_cfg(n=6, judges=[mock]))
    sample.run(cfg)
    generate.run(cfg)
    judge.run(cfg)
    cfg2 = load_config(make_cfg(n=6, judges=[{**mock, "mock": {"seed": 1, "flip_prob": 1.0}}]))
    judge.run(cfg2)  # new settings → new vote keys, old votes stay on disk
    assert len(PartTable(RunPaths(cfg2).votes("mA", "tiny0", "toy")).read()) == 12
    st = aggregate.run(cfg2)
    assert st.loc[0, "n"] == 6
    lab = pd.read_parquet(RunPaths(cfg2).labels("tiny0", "toy"))
    st1 = aggregate.run(cfg)
    lab1 = pd.read_parquet(RunPaths(cfg).labels("tiny0", "toy"))
    assert (lab.set_index("gen_key")["label"] != lab1.set_index("gen_key")["label"]).all()  # flipped
