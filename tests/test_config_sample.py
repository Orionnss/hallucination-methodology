import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from hbench.config import load_config
from hbench.data import normalise_answers
from hbench.stages import sample
from hbench.stages.sample import draw_sample


def test_judges_disjoint_from_generators(make_cfg, tiny_models):
    p = make_cfg(judges=[{"name": "j", "hf_id": tiny_models["gen"]}])
    with pytest.raises(ValidationError, match="judges ∩ generators"):
        load_config(p)


def test_judge_name_clash(make_cfg, tiny_models):
    p = make_cfg(judges=[{"name": "tiny0", "hf_id": tiny_models["judge"]}])
    with pytest.raises(ValidationError, match="share a name"):
        load_config(p)


def test_normalise_answers():
    assert normalise_answers('["a", "b", "a", " "]') == ["a", "b"]
    assert normalise_answers("Paris") == ["Paris"]
    assert normalise_answers(["x", 3]) == ["x", "3"]
    assert normalise_answers("[not json") == ["[not json"]


def test_sample_too_small():
    t = pd.DataFrame({"sample_id": ["a", "b"], "question": ["q", "q"], "expected_answer": [["x"], ["y"]]})
    with pytest.raises(ValueError, match="< n"):
        draw_sample(t, 3, 0, "d")


def test_sample_deterministic_and_saved(make_cfg):
    cfg = load_config(make_cfg(n=10))
    sample.run(cfg)
    a = pd.read_parquet(cfg.run_dir / "samples" / "toy.parquet")
    assert len(a) == 10 and a["sample_id"].is_unique
    b = draw_sample(a.drop(columns="dataset"), 10, 0, "toy")
    assert sorted(b["sample_id"]) == sorted(a["sample_id"])
    sample.run(cfg)  # resume: no-op
    assert a.equals(pd.read_parquet(cfg.run_dir / "samples" / "toy.parquet"))


def test_sample_n_changed_needs_new_run(make_cfg):
    p = make_cfg(n=10)
    sample.run(load_config(p))
    raw = yaml.safe_load(p.read_text())
    raw["n"] = 11
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(RuntimeError, match="new run_id"):
        sample.run(load_config(p))
