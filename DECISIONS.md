# Decisions

This file records each decision that is not in the task prompt (`PROMPT_hallucination_benchmark.md`) or in `ADDING_A_METHOD.md`. All of them were made by the project owner unless noted.

| # | Date | Topic | Decision | Reason / note |
|---|---|---|---|---|
| D1 | 2026-11-03 | Environments | Two environments: `gen` (vLLM) for sample/generate/judge/aggregate, `extract` (lapeigvals' exact pins: torch 2.6.0, transformers 4.47.0) for extract/evaluate/report. Stages exchange data only through disk. | lapeigvals pins conflict with current vLLM. |
| D2 | 2026-11-03 | Env tooling | **uv only**, locally and on SLURM. No micromamba, no `environment.yml` (deviation from prompt §3.1). One `pyproject.toml` with conflicting extras `gen` / `extract` (`[tool.uv] conflicts`), one `uv.lock`. | Owner's choice. |
| D3 | 2026-11-03 | Python | Python 3.12 in both environments. | lapeigvals requires ≥ 3.12; vLLM supports 3.12. (Chosen by Claude, follows from D1.) |
| D4 | 2026-11-03 | lapeigvals license | Repo has no LICENSE file, but the owner has **explicit approval from the authors** to use their code. Admission check 1.3 is treated as passed. The protocol's order still applies (import first; copy only if needed). | Pinned commit `74f885c69399ed525f4eecd2574c218fac7235d7`. |
| D5 | 2026-11-03 | Judge agreement | Report **mean pairwise Cohen's κ** over all judge pairs instead of Fleiss' κ (deviation from prompt §2.4). Each pair is computed on samples where both votes are valid (not PARSE_ERROR). | Fleiss' κ needs equal raters per sample; PARSE_ERROR votes break this. |
| D6 | 2026-11-03 | No valid vote | A sample whose votes are all PARSE_ERROR is discarded with reason `no_valid_vote`, counted separately from ties and refusals. | Not covered by the plurality rule (empty argmax). |
| D7 | 2026-11-03 | Token source | The HF teacher-forced pass always uses the **saved vLLM token IDs**. The vLLM ↔ HF re-tokenisation test is kept as a reported diagnostic (match rate). | Exact; no re-tokenisation drift. |
| D8 | 2026-11-03 | Answer span | The shared answer mask covers the generated tokens **excluding** the final EOS / end-of-turn token. | Per-method overrides are allowed and recorded in the method card. |
| D9 | 2026-11-03 | HP selection | Hyperparameters (incl. the layer) are selected by **mean inner-fold AUROC**. Then τ is selected for the chosen configuration by mean inner-fold MCC (prompt §2.6). | Threshold-free selection. |
| D10 | 2026-11-03 | τ candidates | Unique pooled inner out-of-fold scores, capped at 200 quantile points. | |
| D11 | 2026-11-03 | Layer candidates | Both reference probes search **all** layers (hidden_states 1..L, embeddings excluded) in the inner CV. | |
| D12 | 2026-11-03 | SAPLMA-style probe | Layer selected in the inner CV (as probe 1). MLP 256-128-64, ReLU (as in SAPLMA). "Last token" = last answer token under D8. | |
| D13 | 2026-11-03 | Significance tests | Run on **AUROC and MCC**. Two Holm families, one per metric, over all detector pairs. | |
| D14 | 2026-11-03 | Repeats | Default B = 1 (5 paired outer folds). Configurable. | |
| D15 | 2026-11-03 | Git | Commit on `main`: an initial commit, then one commit per work step. | |
