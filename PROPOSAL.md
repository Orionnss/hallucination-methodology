# Step 1 proposal: repository layout and config schema

Status: **approved 2026-11-03.** The answers to Q1–Q11 and Q14 are recorded in `DECISIONS.md` (D1–D15). Q12–Q13 are still open.

## Facts found (2026-11-03)

- lapeigvals = `github.com/graphml-lab-pwr/lapeigvals`, HEAD `74f885c69399ed525f4eecd2574c218fac7235d7` (2025-10-18). Package name `graph-hallucinations`, module `hallucinations`.
- **No LICENSE file** (HTTP 404, GitHub reports no license). Check 1.3 fails → import allowed, copy (case C) not allowed.
- Its `pyproject.toml` pins `requires-python >=3.12`, `torch==2.6.0`, `transformers==4.47.0`, plus heavy dependencies (dvc, wandb, langchain, torch_geometric, tuned-lens, jupyterlab…). Recent vLLM versions need a different torch/transformers.
- `PoolStorage` must subclass their `hallucinations.llm.activation_storage.ActivationStorage`, so lapeigvals is a dependency of the extract stage from step 3 on, even before any published method is added.
- Local machine: RTX 4080 SUPER 16 GB, 94 GB RAM, 28 cores. `uv` is installed; micromamba is not installed.

## Repository layout

```
hallucination-methodology/
├── pyproject.toml               # package `hbench`; conflicting uv extras [gen] (vLLM) / [extract] (lapeigvals pins); [dev]
├── uv.lock
├── DECISIONS.md
├── configs/
│   ├── experiment.example.yaml
│   └── local_e2e.yaml           # step 4 run
├── src/hbench/
│   ├── cli.py                   # `hbench <stage> --config X.yaml [--only key=...]`
│   ├── config.py                # pydantic schema, validation (judges ∩ generators = ∅, …), hashes
│   ├── keys.py                  # cache keys, prompt/decoding hashes
│   ├── io.py                    # Parquet/HDF5 helpers, atomic writes, "done" manifests for resume
│   ├── provenance.py            # resolved config, git hash, lib versions, GPU, HF model revisions
│   ├── seeding.py
│   ├── datasets/                # loaders → (sample_id, question, expected_answer: list[str])
│   ├── stages/
│   │   ├── sample.py
│   │   ├── generate.py          # vLLM; main answer + extra samples
│   │   ├── judge.py             # vLLM, constrained output, 1 retry, PARSE_ERROR
│   │   ├── aggregate.py         # plurality vote, label map, labelling stats, Fleiss' κ
│   │   ├── extract.py           # capture engine: one teacher-forced HF pass → PoolStorage
│   │   ├── evaluate.py          # nested stratified CV
│   │   └── report.py            # CSV + LaTeX booktabs + Markdown
│   ├── judging/                 # judge prompt template, parser, mock judge
│   ├── pool/
│   │   ├── storage.py           # PoolStorage(ActivationStorage)
│   │   ├── adapter.py           # to_generate_output(forward_out, input_length)
│   │   ├── masks.py             # answer-token mask from saved token IDs (computed once)
│   │   ├── registry.py          # name → (FeatureExtractor, Classifier)
│   │   ├── interfaces.py        # FeatureExtractor / Classifier / Reducer / Needs protocols
│   │   └── methods/
│   │       ├── mean_hidden_probe/        # reference 1 (+ CARD.md)
│   │       ├── saplma_style_probe/       # reference 2 (+ CARD.md)
│   │       └── <published>/              # later, per ADDING_A_METHOD.md
│   ├── vendor/                  # case C only (empty; lapeigvals cannot go here)
│   ├── cv.py                    # outer/inner folds, threshold selection, leakage guards
│   ├── metrics.py
│   └── stats.py                 # corrected resampled t-test, Wilcoxon, Holm
├── third_party/lapeigvals/      # git submodule @ pinned commit
├── slurm/                       # one template per GPU stage (array jobs)
├── tests/                       # pytest, CPU, tiny model + mock judge, < 2 min
└── runs/<run_id>/               # outputs (git-ignored), see below
```

Run directory (one per experiment):

```
runs/<run_id>/
  provenance/{config.resolved.yaml, git.txt, env.json}
  samples/<dataset>.parquet                     # sample_id, question, expected_answer
  generations/<generator>/<dataset>.parquet     # + prompt_text, answer_text, prompt_ids, answer_ids, key
  extra_samples/<generator>/<dataset>.parquet   # only if a method needs them
  votes/<judge>/<generator>/<dataset>.parquet
  labels/<generator>/<dataset>.parquet          # ℓ_i, y_i, discard reason
  features/<generator>/<dataset>/<extractor>.h5 # x only, key = sample_id; + manifest.json
  folds/<generator>/<dataset>.parquet           # outer fold ids (shared by all detectors)
  metrics/<detector>/<generator>/<dataset>.parquet   # per outer fold
  report/{tables/*.csv, tables/*.tex, summary.md}
```

Resume: each stage writes per-key rows; on restart it reads the keys already present and skips them. Writes are atomic (tmp file + rename).

## Config schema (YAML)

```yaml
run_id: local_e2e_001
seed: 0
n: 200
output_dir: runs/

datasets:
  - name: triviaqa
    loader: hf                     # hf | parquet | csv
    path: mandarjoshi/trivia_qa
    subset: rc.nocontext
    split: validation
    id_field: question_id
    question_field: question
    answer_field: answer.aliases   # normalised to list[str]

generators:
  - name: qwen2.5-1.5b-instruct
    hf_id: Qwen/Qwen2.5-1.5B-Instruct
    revision: <commit>             # resolved and logged if omitted
    dtype: bfloat16
    vllm: {gpu_memory_utilization: 0.85, max_model_len: 2048}

judges:                            # must be disjoint from generators (by hf_id + revision)
  - {name: judge-a, hf_id: ..., revision: ..., dtype: bfloat16}

prompt:
  template: "Answer the question concisely.\nQuestion: {question}\nAnswer:"
  system: null
  use_chat_template: true

decoding:                          # θ, shared by all generators
  temperature: 0.0
  max_new_tokens: 64
  stop: null

extra_decoding:                    # only for methods with extra_samples
  num_samples: 0
  temperature: 1.0
  top_p: 1.0
  max_new_tokens: 64

judging:
  prompt_template_file: configs/judge_prompt.txt
  output: json                     # constrained via vLLM guided decoding
  retries: 1

extraction:
  dtype: float32 | bfloat16
  attn_implementation: auto        # → eager if any method needs attentions
  batch_size: 8
  shard_size: 1000                 # flush() per shard (RAM bound)
  token_source: saved_ids          # see question Q6

detectors:
  - name: mean_hidden_probe
    params: {layers: all}
    grid: {C: [0.01, 0.1, 1, 10]}
  - name: saplma_style_probe
    params: {layer: ...}
    grid: {hidden: [[256,128,64]], lr: [1e-3], weight_decay: [0, 1e-4]}

cv:
  outer_folds: 5
  inner_folds: 3
  repeats: 1                       # B seeds
  hp_selection_metric: ...         # see Q9
  threshold_grid: ...              # see Q9

stats:
  test_metric: auroc               # see Q10
  alpha: 0.05

slurm: {partition: ..., account: ..., gpus_per_node: 4, time: ...}
```

## Open questions

Q1–Q11 and Q14 are resolved; see `DECISIONS.md`. Still open:

- **Q12.** The generator, the dataset and the 3 judges for the local end-to-end run (16 GB), before step 4.
- **Q13.** The prompt template π, the system prompt, max_new_tokens and the judge prompt wording. I will write drafts for your approval in step 2.
- Probe 1 details (standardisation inside the folds, the C grid), before step 3.
