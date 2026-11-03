"""Experiment config: one YAML file describes the full experiment (prompt §3.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetCfg(_Strict):
    name: str
    loader: Literal["hf", "parquet", "csv", "jsonl"]
    path: str
    subset: str | None = None
    split: str | None = None
    id_field: str | None = None  # None → row index is the sample ID
    question_field: str = "question"
    answer_field: str = "expected_answer"  # dotted path allowed, e.g. "answer.aliases"


class ModelCfg(_Strict):
    name: str
    hf_id: str
    revision: str | None = None  # resolved to the HF commit hash at load time and logged
    dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    batch_size: int = 16
    device: str = "auto"  # "auto" → cuda if available, else cpu


class JudgeCfg(ModelCfg):
    kind: Literal["hf", "mock"] = "hf"
    mock: dict[str, Any] = Field(default_factory=dict)  # only for kind="mock" (tests)


class PromptCfg(_Strict):
    template: str
    system: str | None = None
    use_chat_template: bool = True


class DecodingCfg(_Strict):
    do_sample: bool = False  # default θ: greedy
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 64


class ExtraSamplesCfg(_Strict):
    num_samples: int = 0
    decoding: DecodingCfg = Field(default_factory=lambda: DecodingCfg(do_sample=True))


class JudgingCfg(_Strict):
    prompt_template_file: str
    retries: int = 1
    max_new_tokens: int = 16


class CVCfg(_Strict):
    outer_folds: int = 5
    inner_folds: int = 3
    repeats: int = 1
    max_threshold_candidates: int = 200


class DetectorCfg(_Strict):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    grid: dict[str, list[Any]] = Field(default_factory=dict)


class ExperimentCfg(_Strict):
    run_id: str
    seed: int = 0
    n: int
    output_dir: str = "runs"
    datasets: list[DatasetCfg]
    generators: list[ModelCfg]
    judges: list[JudgeCfg]
    prompt: PromptCfg
    decoding: DecodingCfg = Field(default_factory=DecodingCfg)
    extra_samples: ExtraSamplesCfg = Field(default_factory=ExtraSamplesCfg)
    judging: JudgingCfg
    cv: CVCfg = Field(default_factory=CVCfg)
    detectors: list[DetectorCfg] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "ExperimentCfg":
        for kind, items in (("dataset", self.datasets), ("generator", self.generators), ("judge", self.judges)):
            names = [x.name for x in items]
            if len(set(names)) != len(names):
                raise ValueError(f"duplicate {kind} names: {names}")
        # Prompt §2.1: 𝓙 ∩ 𝓖 = ∅. Compare the model identity (hf_id + revision), not the display name.
        gen_ids = {(g.hf_id, g.revision) for g in self.generators}
        gen_hf = {g.hf_id for g in self.generators}
        for j in self.judges:
            if j.kind == "mock":
                continue
            if (j.hf_id, j.revision) in gen_ids or j.hf_id in gen_hf:
                raise ValueError(f"judge {j.name!r} ({j.hf_id}) is also a generator: judges ∩ generators must be empty")
        if self.judges and {j.name for j in self.judges} & {g.name for g in self.generators}:
            raise ValueError("a judge and a generator share a name")
        return self

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_id

    def dataset(self, name: str) -> DatasetCfg:
        return _by_name(self.datasets, name)

    def generator(self, name: str) -> ModelCfg:
        return _by_name(self.generators, name)

    def judge(self, name: str) -> JudgeCfg:
        return _by_name(self.judges, name)


def _by_name(items: list[Any], name: str) -> Any:
    for x in items:
        if x.name == name:
            return x
    raise KeyError(name)


def load_config(path: str | Path) -> ExperimentCfg:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    cfg = ExperimentCfg.model_validate(raw)
    # Relative judge-prompt paths are resolved against the config file location.
    p = Path(cfg.judging.prompt_template_file)
    if not p.is_absolute() and not p.exists():
        cfg.judging.prompt_template_file = str((path.parent / p).resolve())
    return cfg
