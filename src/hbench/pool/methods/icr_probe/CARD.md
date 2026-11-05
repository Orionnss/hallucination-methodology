# Method card: icr_probe (integrated; cluster parity pending)

```yaml
name: icr_probe
paper: {title: "ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs", venue: "ACL 2025", year: 2025, url: "https://arxiv.org/abs/2507.16488"}
code: {repo: "https://github.com/XavierZhang2002/ICR_Probe", commit: 40ec490e762cadbac6bcefdc24a8f0d5974e8448, license: "Apache-2.0 (LICENSE file; README badge says MIT)"}
integration_case: "B for the score (their ICRScore on per-example generate-format output); A1 for the MLP; own training loop (their trainer does not run, D68)"
capture_point: {file: "none in repo (README: forward pass with cached hidden_states + attentions in generate format, batch size 1)"}
feature_function:
  file: src/icr_score.py
  lines: "18-251 (ICRScore.__init__ → compute_icr)"
  settings: "D67: top_k=20, top_p=None, pooling=mean, use_induction_head=False, attention_uniform=False, hidden_uniform=False"
  output: "ICR [L × N_answer]; probe input = token-wise mean → [L] (paper §4, Alg. 2 line 10, App. B.4)"
classifier:
  file: src/utils.py
  lines: "5-50 (ICRProbe: L→128→64→32→1, BatchNorm, dropout 0.3, LeakyReLU, sigmoid, Kaiming init)"
  training: "D68 (paper B.4): Adam 5e-4, 50 epochs, batch 32, BCE, ReduceLROnPlateau(0.5, 5), best val loss on a 20 % split of T_f"
  grid: none (paper gives fixed values)
needs:
  hidden_states: {layers: all (incl. embeddings), positions: all}
  attentions: {layers: all, heads: all, positions: all}
  masks: [core_positions: user_prompt_start, user_prompt_end, response_start (D69)]
  extra_samples: 0
  other_passes: none
  dtype_in_paper: to check
reducer_ops: [their ICRScore per example (unchanged), token-wise mean over answer tokens]
fitted_ops: [MLP training incl. validation split and best-epoch selection]
deviations_from_original:
  - "transformers 5.20.2 (D62)"
  - "top_k = 20 and all-head mean per paper, not the README call (D67)"
  - "training loop written by us from paper B.4 (their trainer is incomplete, D68)"
  - "core positions defined by D69 (not specified in their code)"
  - "label: we predict P(hallucination); the paper's output is P(non-hallucinated) (same BCE)"
  - "a training batch of one example is skipped (batch norm); CPU-only no-op for their torch.cuda.device context (D78)"
parity_test: {tiny_model_cpu_fp32: {max_rel_err: 2.5e-7, spearman: 1.0, n_samples: 3, date: 2026-11-05}, real_generators: pending (cluster, D65)}  # their ICRScore on generate() output vs ours on adapter output
reproduction_test: not_done
```
