# Decisions

This file records each decision that is not in the task prompt (`PROMPT_hallucination_benchmark.md`) or in `ADDING_A_METHOD.md`. All of them were made by the project owner unless noted.

| # | Date | Topic | Decision | Reason / note |
|---|---|---|---|---|
| D1 | 2026-11-03 | Environments | ~~Superseded by D17.~~ Two environments: `gen` (vLLM) for sample/generate/judge/aggregate, `extract` (lapeigvals' exact pins: torch 2.6.0, transformers 4.47.0) for extract/evaluate/report. Stages exchange data only through disk. | lapeigvals pins conflict with current vLLM. |
| D2 | 2026-11-03 | Env tooling | **uv only**, locally and on SLURM. No micromamba, no `environment.yml` (deviation from prompt §3.1). One `pyproject.toml`, one `uv.lock`. | Owner's choice. |
| D3 | 2026-11-03 | Python | Python 3.12. | lapeigvals requires ≥ 3.12. (Chosen by Claude.) |
| D4 | 2026-11-03 | lapeigvals license | Repo has no LICENSE file, but the owner has **explicit approval from the authors** to use their code. Admission check 1.3 is treated as passed. The protocol's order still applies (import first; copy only if needed). | Pinned commit `74f885c69399ed525f4eecd2574c218fac7235d7`. |
| D5 | 2026-11-03 | Judge agreement | Report **mean pairwise Cohen's κ** over all judge pairs instead of Fleiss' κ (deviation from prompt §2.4). Each pair is computed on samples where both votes are valid (not PARSE_ERROR). | Fleiss' κ needs equal raters per sample; PARSE_ERROR votes break this. |
| D6 | 2026-11-03 | No valid vote | A sample whose votes are all PARSE_ERROR is discarded with reason `no_valid_vote`, counted separately from ties and refusals. | Not covered by the plurality rule (empty argmax). |
| D7 | 2026-11-03 | Token source | The HF teacher-forced pass always uses the **saved token IDs** from the generate stage (prompt IDs + answer IDs). It never re-tokenises the text. The vLLM ↔ HF token test of prompt §3.1 is dropped (no vLLM, D16); a re-tokenisation match rate is reported only as a diagnostic. | Exact. |
| D8 | 2026-11-03 | Answer span | The shared answer mask covers the generated tokens **excluding** the final EOS / end-of-turn token. | Per-method overrides are allowed and recorded in the method card. |
| D9 | 2026-11-03 | HP selection | Hyperparameters (incl. the layer) are selected by **mean inner-fold AUROC**. Then τ is selected for the chosen configuration by mean inner-fold MCC (prompt §2.6). | Threshold-free selection. |
| D10 | 2026-11-03 | τ candidates | Unique pooled inner out-of-fold scores, capped at 200 quantile points. | |
| D11 | 2026-11-03 | Layer candidates | Both reference probes search **all** layers (hidden_states 1..L, embeddings excluded) in the inner CV. | |
| D12 | 2026-11-03 | SAPLMA-style probe | Layer selected in the inner CV (as probe 1). MLP 256-128-64, ReLU (as in SAPLMA). "Last token" = last answer token under D8. | |
| D13 | 2026-11-03 | Significance tests | Run on **AUROC and MCC**. Two Holm families, one per metric, over all detector pairs. | |
| D14 | 2026-11-03 | Repeats | Default B = 1 (5 paired outer folds). Configurable. | |
| D15 | 2026-11-03 | Git | Commit on `main`: an initial commit, then one commit per work step. | |
| D16 | 2026-11-03 | Inference backend | **HF `transformers` everywhere**: generation, judges and extraction. No vLLM (deviation from prompt §3.1). Tests use the same code on a tiny model on CPU. | Owner's choice. |
| D17 | 2026-11-03 | Environment | **One environment** for all stages, pinned to lapeigvals' versions (torch 2.6.0, transformers 4.47.0, scikit-learn 1.6.1, datasets 3.1.0, pandas 2.2.3, pydantic 2.11.3, accelerate 1.1.1, pytest 8.3.4). Supersedes D1. | No vLLM conflict left (D16). Generator families not supported by transformers 4.47 need a new decision. |
| D18 | 2026-11-03 | Judge constraint | HF constrained generation: a logits processor allows only the token sequences of `{"label": "CORRECT"}`, `{"label": "INCORRECT"}`, `{"label": "REFUSAL"}`. The parser, the single retry and PARSE_ERROR stay as specified. Judges use greedy decoding and their chat template. | Replaces vLLM guided JSON. |
| D19 | 2026-11-03 | Prompts | Generator prompt π: `Answer the following question with a short answer. Do not explain.\nQuestion: {question}`, through the chat template, no system prompt, greedy, max_new_tokens = 64. Judge prompt: as approved, stored in `configs/judge_prompt.txt`; label-only JSON output. | |
| D20 | 2026-11-03 | Judge prompt rendering | `{answers}` is rendered as a JSON list of all acceptable answers Y_i. | (Chosen by Claude.) |
| D21 | 2026-11-03 | Judge retry | With constrained HF decoding (D18) the output is always parseable, so the retry only matters for mock or future unconstrained judges. A greedy retry would repeat the same output. | (Chosen by Claude.) |
| D22 | 2026-11-03 | Resume storage | Generations and votes are stored as directories of append-only Parquet part files (one per batch, atomic write). A restarted stage skips the keys already present. A vote key = generation key + judge name + judge-prompt hash. | (Chosen by Claude.) Not features: the "no per-batch intermediate files" rule of ADDING_A_METHOD.md applies to the extract stage. |
| D23 | 2026-11-03 | Extra samples seeding | Extra samples (sampling decoding) are seeded per batch from (seed, first key of the batch). This is reproducible with the same batch size and the same resume points. | (Chosen by Claude.) Only used once a method needs extra samples. |
| D24 | 2026-11-03 | Rate denominators | Refusal, tie and no-valid-vote rates are fractions of all n generated samples of the cell. The class ratio is computed on the kept samples. | (Chosen by Claude.) |
