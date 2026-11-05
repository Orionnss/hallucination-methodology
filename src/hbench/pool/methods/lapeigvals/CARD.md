# Method card: lapeigvals (integrated; cluster parity pending)

```yaml
name: lapeigvals
paper: {title: "Hallucination Detection in LLMs Using Spectral Features of Attention Maps", venue: arXiv, year: 2025, url: "https://arxiv.org/abs/2502.17598"}
code: {repo: "https://github.com/graphml-lab-pwr/lapeigvals", commit: 74f885c69399ed525f4eecd2574c218fac7235d7, license: "none in repo; written approval from the authors (DECISIONS.md D4)"}
integration_case: A0 (their storage through to_generate_output); A1 optional after A1-vs-A0 parity
capture_point: {file: hallucinations/llm/predict.py, lines: "48-68 (generate with output_attentions; storage.update with get_token_masks)"}
feature_function:
  file: hallucinations/llm/feature_storage.py
  lines: "28-111 (AttentionAndLaplacianDiagsFeatureStorage.update → extract_and_record_features_data)"
  calls: "stack_attention_matrix → remove_padding_from_intermediate_states(data_type=attn) → laplacian_diagonal_from_attn(A, vertical_edges=False) (features/attention_weights.py:24-69)"
  span: "prompt + generated tokens except the last one (processing.py: generated_tokens[:, :-1]); padding removed by value"
  output: "[#layers, #heads, seq_len] Laplacian diagonal per example; eigenvalues = diagonal (lower-triangular)"
classifier:
  file: hallucinations/probe_models/lr.py
  lines: "19-80 (PCA(pca_dim) + LogisticRegression(class_weight=balanced, max_iter=2000))"
  topk: "features/attn_feats.py:49-70 get_laplacian_eigvals_per_head_topk (sort descending, top_k per head)"
  grid: "D70: k ∈ {5,10,25,50,100} (≤ shortest sequence) × {all layers, each layer}; PCA 512"
needs:
  attentions: {layers: all, heads: all, positions: all (prompt + answer)}
  masks: [attention_mask, special_token_mask, decoder_added_token_mask (their get_token_masks), pad_token_id]
  extra_samples: 0
  other_passes: none
  dtype_in_paper: to check (parity in fp32 and bf16)
reducer_ops: [their Laplacian diagonal (unchanged), per-head descending sort, keep top-100 per head (data-independent)]
fitted_ops: [k selection, layer selection, PCA 512, logistic regression]
deviations_from_original:
  - "transformers 5.20.2 instead of 4.47.0 (D62)"
  - "thin subclass: save() of per-batch .pt intermediate states disabled; _check_save_dir does not need the attentions dir"
  - "flush() per shard (RAM bound, ADDING_A_METHOD.md §5)"
  - "attentions from a teacher-forced eager pass, not generate() (adapter, D36)"
  - "TF32 import side effect undone; precision from the config (D29)"
  - "eager attention for the whole shared pass"
parity_test: {tiny_model_cpu_fp32: {max_rel_err: 6.5e-7, spearman: 1.0, n_samples: 3, date: 2026-11-05}, qwen3_1.7b_gpu_local: {float32: {verdict: pass_with_note, max_rel_err: 1.7e-3, spearman: 1.0000}, bfloat16: {verdict: pass_with_note, max_rel_err: 15.4, spearman: 0.9990}, n_samples: 7, excluded: 1, date: 2026-11-05}, real_generators: pending (cluster, D65), metric: D76 + D83}
reproduction_test: not_done
```

Open items before coding: confirm their RAM cost per shard; confirm `remove_padding_from_intermediate_states` with pad == EOS tokenizers.
