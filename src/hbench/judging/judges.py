"""Judges. Each judge returns one raw output per item; the stage parses it (prompt §2.4)."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Protocol

from transformers import LogitsProcessorList

from hbench.config import DecodingCfg, ExperimentCfg, JudgeCfg
from hbench.generation import build_prompt, generate_batch, strip_eos
from hbench.judging.constraint import AllowedSequencesProcessor, label_token_sequences
from hbench.judging.labels import CORRECT, INCORRECT, LABELS, REFUSAL, label_json
from hbench.keys import stable_hash


@dataclass
class JudgeOut:
    raw: str  # the label output, parsed by the stage
    reasoning: str | None = None  # thinking judges only (D51)
    forced_close: bool | None = None  # the reasoning hit the budget (or EOS) and was closed for it
    reasoning_tokens: int | None = None


@dataclass
class JudgeItem:
    key: str
    question: str
    answer: str
    expected: list[str]


def render_judge_prompt(template: str, item: JudgeItem) -> str:
    # All acceptable answers Y_i are given, as a JSON list (DECISIONS.md D20).
    return template.format(question=item.question, answer=item.answer, answers=json.dumps(item.expected, ensure_ascii=False))


class Judge(Protocol):
    name: str
    commit: str | None

    def judge(self, items: list[JudgeItem], attempt: int) -> list[JudgeOut]: ...


class HFJudge:
    def __init__(self, cfg: JudgeCfg, template: str, max_new_tokens: int):
        from hbench.models import load_model

        self.name = cfg.name
        self.cfg = cfg
        self.template = template
        self.lm = load_model(cfg)
        self.commit = self.lm.commit
        self.decoding = DecodingCfg(do_sample=False, max_new_tokens=max_new_tokens)
        self.seqs = label_token_sequences(self.lm.tokenizer)
        self.eos_id = next(iter(sorted(self.lm.eos_ids)))
        need = max(len(s) for s in self.seqs) + 1
        if max_new_tokens < need:
            raise ValueError(f"judging.max_new_tokens = {max_new_tokens} < {need} tokens needed for a label")
        th = cfg.thinking
        if th.enabled:
            ids = self.lm.tokenizer(th.end_marker, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise ValueError(f"{cfg.name}: thinking.end_marker {th.end_marker!r} is {len(ids)} tokens; it must be one")
            self.marker_id = ids[0]

    def _tok(self, text: str) -> list[int]:
        return self.lm.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _think(self, prompts: list[list[int]]) -> tuple[list[list[int]], list[JudgeOut]]:
        """Phase 1: greedy reasoning until the end marker or the budget. Returns the prompts for the
        label phase (prompt + reasoning + closing text) and the reasoning metadata."""
        th = self.cfg.thinking
        dec = DecodingCfg(do_sample=False, max_new_tokens=th.max_tokens)
        gens = generate_batch(self.lm, prompts, dec, stop_ids={self.marker_id})
        nxt, meta = [], []
        for p, [g] in zip(prompts, gens):
            ids = g.answer_ids
            natural = g.finish_reason == "eos" and ids and ids[-1] == self.marker_id
            if natural:
                body = ids[:-1]
                cont = ids + self._tok(th.after_marker)
            else:  # budget overrun, or EOS before the marker
                body = strip_eos(ids, self.lm.eos_ids)
                cont = body + self._tok(th.close)
            nxt.append(p + cont)
            meta.append(
                JudgeOut(raw="", reasoning=self.lm.tokenizer.decode(body, skip_special_tokens=True), forced_close=not natural, reasoning_tokens=len(body))
            )
        return nxt, meta

    def judge(self, items: list[JudgeItem], attempt: int) -> list[JudgeOut]:
        out: list[JudgeOut] = []
        bs = self.cfg.batch_size
        for s in range(0, len(items), bs):
            chunk = items[s : s + bs]
            prompts = [
                build_prompt(
                    self.lm, render_judge_prompt(self.template, it), None, True,
                    self.cfg.chat_template_kwargs, self.cfg.assistant_prefix, self.cfg.template_date,
                )[1]
                for it in chunk
            ]
            meta = [JudgeOut(raw="") for _ in chunk]
            if self.cfg.thinking.enabled:
                prompts, meta = self._think(prompts)
            proc = AllowedSequencesProcessor(self.seqs, self.eos_id, max(len(p) for p in prompts))
            gens = generate_batch(self.lm, prompts, self.decoding, logits_processor=LogitsProcessorList([proc]))
            for m, [g] in zip(meta, gens):
                m.raw = self.lm.tokenizer.decode(strip_eos(g.answer_ids, self.lm.eos_ids), skip_special_tokens=True)
                out.append(m)
        return out


class MockJudge:
    """Deterministic judge for tests. `mode: exact_match` votes CORRECT when an acceptable answer
    appears in the generated answer (case-insensitive). Optional noise: `flip_prob`, `refusal_prob`,
    `parse_error_prob` (applied per attempt, so a retry can succeed). `script: {key: label | raw}`
    fixes the output for given keys."""

    def __init__(self, cfg: JudgeCfg):
        self.name = cfg.name
        self.commit = None
        m = cfg.mock
        self.flip = float(m.get("flip_prob", 0.0))
        self.refuse = float(m.get("refusal_prob", 0.0))
        self.perr = float(m.get("parse_error_prob", 0.0))
        self.script: dict[str, str] = m.get("script", {})
        self.seed = int(m.get("seed", 0))
        self.calls = 0

    def _one(self, it: JudgeItem, attempt: int) -> str:
        if it.key in self.script:
            v = self.script[it.key]
            return label_json(v) if v in LABELS else v
        rng = random.Random(int(stable_hash([self.seed, self.name, it.key, attempt], 12), 16))
        if rng.random() < self.perr:
            return "not json"
        ans = it.answer.lower()
        lab = CORRECT if any(re.search(re.escape(e.lower()), ans) for e in it.expected) else INCORRECT
        if rng.random() < self.flip:
            lab = INCORRECT if lab == CORRECT else CORRECT
        if rng.random() < self.refuse:
            lab = REFUSAL
        return label_json(lab)

    def judge(self, items: list[JudgeItem], attempt: int) -> list[JudgeOut]:
        self.calls += len(items)
        return [JudgeOut(raw=self._one(it, attempt)) for it in items]


def make_judge(cfg: ExperimentCfg, jcfg: JudgeCfg) -> Judge:
    if jcfg.kind == "mock":
        return MockJudge(jcfg)
    with open(cfg.judging.prompt_template_file) as f:
        template = f.read()
    return HFJudge(jcfg, template, cfg.judging.max_new_tokens)


def judge_prompt_hash(cfg: ExperimentCfg) -> str:
    with open(cfg.judging.prompt_template_file) as f:
        return stable_hash({"template": f.read(), "max_new_tokens": cfg.judging.max_new_tokens, "constraint": "label-json-v1"})
