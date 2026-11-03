# hbench — benchmark of white-box hallucination detectors

Pipeline: `sample → generate → judge → aggregate → extract → evaluate → report`. Each stage is a
CLI command that reads from and writes to `runs/<run_id>/`, and resumes by key after a failure.
Design decisions: [DECISIONS.md](DECISIONS.md). Adding a published detector: `ADDING_A_METHOD.md`.

## Setup (uv only)

```bash
git submodule update --init          # third_party/lapeigvals @ 74f885c
uv sync                              # main env (.venv): lapeigvals pins, torch 2.6 / transformers 4.47
UV_PROJECT_ENVIRONMENT=.venv-judge uv sync --no-default-groups --group judge   # judge env (transformers 5.20, bitsandbytes)
export HF_DATASETS_CACHE=~/.cache/hbench/datasets   # datasets 3.1 cannot read newer caches (D38)
```

## Run locally

```bash
uv run hbench sample    -c configs/local_e2e.yaml
uv run hbench generate  -c configs/local_e2e.yaml
./bin/hbench-judge      -c configs/local_e2e.yaml      # judge stage, in the judge env
uv run hbench aggregate -c configs/local_e2e.yaml
uv run hbench extract   -c configs/local_e2e.yaml
uv run hbench evaluate  -c configs/local_e2e.yaml
uv run hbench report    -c configs/local_e2e.yaml      # → runs/<run_id>/report/{summary.md,tables/*.csv,*.tex}
```

Each stage accepts `--generator`, `--dataset` (and `--judge`, `--detector` where relevant) to restrict the work.

## Run on SLURM

```bash
cp slurm/cluster.env.example slurm/cluster.env   # partitions, account, times, caches
slurm/submit_all.sh configs/<experiment>.yaml    # optional 2nd argument: first stage to submit
```

Array jobs: one task per (generator × dataset) for generate / extract / evaluate, and per
(judge × generator × dataset) for judge. Logs go to `slurm/logs/`. Each array task uses one GPU
(`GPUS_PER_TASK`); models must fit on one 80 GB GPU.

## Tests

```bash
uv run pytest -q                     # CPU, tiny model + mock judges, < 2 min
UV_PROJECT_ENVIRONMENT=.venv-judge uv run --no-default-groups --group dev --group judge \
  pytest -q tests/test_votes.py tests/test_thinking_judge.py "tests/test_stages.py::test_hf_judge_constrained_output"
```
