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
    b = draw_sample(a.drop(columns="dataset"), 10, 0, "toy")  # same seed → same draw from the same table
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


def _ds(tmp_path, rows, **kw):
    from hbench.config import DatasetCfg

    p = tmp_path / "d.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return DatasetCfg(name="d", loader="csv", path=str(p), id_field="id", answer_field="a", **kw)


def test_dedupe_ids(tmp_path):
    from hbench.data import load_dataset_table

    rows = [{"id": "x", "question": "q1", "a": "A"}, {"id": "x", "question": "q1", "a": "A"}, {"id": "y", "question": "q2", "a": "B"}]
    with pytest.raises(ValueError, match="duplicate sample IDs"):
        load_dataset_table(_ds(tmp_path, rows))
    df, st = load_dataset_table(_ds(tmp_path, rows, dedupe_ids=True))
    assert len(df) == 2 and st["duplicates_dropped"] == 1
    rows[1]["a"] = "other"
    df, st = load_dataset_table(_ds(tmp_path, rows, dedupe_ids=True))
    assert df["sample_id"].tolist() == ["y"] and st["conflicting_ids_dropped"] == 1 and st["conflicting_rows_dropped"] == 2


def test_drop_empty_and_split_answers(tmp_path):
    from hbench.data import load_dataset_table

    rows = [{"id": "x", "question": "q", "a": "A; B;A", "b": "C"}, {"id": "y", "question": "q", "a": None, "b": None}]
    with pytest.raises(ValueError, match="no expected answer"):
        load_dataset_table(_ds(tmp_path, rows, answer_split=";"))
    from hbench.config import DatasetCfg

    cfg = _ds(tmp_path, rows, answer_split=";", drop_empty_answers=True).model_copy(update={"answer_field": ["a", "b"]})
    df, st = load_dataset_table(cfg)
    assert df["expected_answer"].tolist() == [["A", "B", "C"]] and st["empty_answers_dropped"] == 1


def test_fixed_template_date():
    from datetime import datetime

    from transformers import AutoTokenizer

    from hbench.generation import fixed_template_date
    from tests.conftest import TINY

    tok = AutoTokenizer.from_pretrained(TINY)
    tok.chat_template = "{{ strftime_now('%Y-%m-%d') }}|{{ messages[0]['content'] }}"
    with fixed_template_date("2001-02-03"):
        assert tok.apply_chat_template([{"role": "user", "content": "q"}], tokenize=False).startswith("2001-02-03|")
    assert tok.apply_chat_template([{"role": "user", "content": "q"}], tokenize=False).startswith(datetime.now().strftime("%Y-%m-%d"))
