from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
TINY = "sshleifer/tiny-gpt2"
CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory) -> dict[str, str]:
    """Two local copies of a tiny model with a chat template: one generator, one judge
    (judges ∩ generators must be empty)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = tmp_path_factory.mktemp("models")
    out = {}
    for name in ("gen", "judge", "thinker"):
        tok = AutoTokenizer.from_pretrained(TINY)
        tok.chat_template = CHAT_TEMPLATE
        model = AutoModelForCausalLM.from_pretrained(TINY)
        if name == "thinker":  # a judge with a single-token think marker
            tok.add_tokens(["</think>"], special_tokens=True)
            model.resize_token_embeddings(len(tok))
        d = base / name
        tok.save_pretrained(d)
        model.save_pretrained(d)
        out[name] = str(d)
    return out


def make_dataset(path: Path, n_rows: int = 120) -> Path:
    rows = [
        {"id": f"q{i}", "question": f"What is {i} plus {i}?", "answers": json.dumps([str(2 * i), f"number {2 * i}"])}
        for i in range(n_rows)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def make_cfg(tmp_path, tiny_models):
    def _make(n: int = 12, judges: list[dict] | None = None, generators: int = 1, **over) -> Path:
        ds = make_dataset(tmp_path / "ds.csv")
        gens = [
            {"name": f"tiny{k}", "hf_id": tiny_models["gen"], "dtype": "float32", "batch_size": 4, "device": "cpu"}
            for k in range(generators)
        ]
        cfg = {
            "run_id": "t",
            "seed": 0,
            "n": n,
            "output_dir": str(tmp_path / "runs"),
            "datasets": [
                {"name": "toy", "loader": "csv", "path": str(ds), "id_field": "id", "question_field": "question", "answer_field": "answers"}
            ],
            "generators": gens,
            "judges": judges
            or [
                {"name": "mA", "hf_id": "mock", "kind": "mock", "mock": {"seed": 1, "flip_prob": 0.3}},
                {"name": "mB", "hf_id": "mock", "kind": "mock", "mock": {"seed": 2, "flip_prob": 0.3, "refusal_prob": 0.1}},
                {"name": "mC", "hf_id": "mock", "kind": "mock", "mock": {"seed": 3, "flip_prob": 0.3, "parse_error_prob": 0.2}},
            ],
            "prompt": {"template": "Answer the following question with a short answer. Do not explain.\nQuestion: {question}", "use_chat_template": True},
            "decoding": {"max_new_tokens": 6},
            "judging": {"prompt_template_file": str(ROOT / "configs" / "judge_prompt.txt"), "max_new_tokens": 16},
        }
        cfg.update(over)
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(cfg))
        return p

    return _make
