# Method card: charm (DRAFT — integration not started)

```yaml
name: charm
paper: {title: "Neural Message-Passing on Attention Graphs for Hallucination Detection", venue: "ICLR 2026", year: 2026, url: "https://arxiv.org/abs/2509.24770"}
code: {repo: "https://github.com/Noired/charm", commit: 7ab3ab7f7a49fd6ca23defc44ddcd6c794598c61, license: MIT}
integration_case: "B: their get_data_object / get_activations on per-example generate-format output (adapter); their CHARM model and train_model loop imported"
capture_point: {file: data_prep_2/data_collection.py, lines: "18-120 (generate with output_attentions, output_hidden_states; batch size 1)"}
feature_function:
  files: "dataset/graphs.py:92-137 get_data_object (threshold 0.05, prompt_graph=False); dataset/acts.py get_activations"
  output: "attributed graph per sample: nodes = tokens (last prompt token onward for the response), x = attention diagonal per (layer, head), edges = attention ≥ 0.05 in any head, edge_attr = [L·H]; activations of one layer"
classifier:
  files: "model/mp.py:95-197 (CHARM), training.py:269-398 (train_model), 171-199 (get_model), 72-115 (get_transforms)"
  settings: "D71: configs/charm/movies_att_act.yaml; act layer = round(0.75·L)"
  grid: none (their config values)
needs:
  attentions: {layers: all, heads: all, positions: all}
  hidden_states: {layers: [round(0.75·L)], positions: all}
  masks: []
  extra_samples: 0
  other_passes: none
  dtype_in_paper: fp16 storage of graphs and activations (their data_collection)
reducer_ops: [their get_data_object (graph construction, threshold 0.05), their get_activations at one layer]
fitted_ops: [GNN training incl. validation split, early stopping, checkpoint of best epoch]
deviations_from_original:
  - "transformers 5.20.2, torch 2.9.1 (their exp env: 4.52.4, torch 2.7) (D62)"
  - "activation layer = round(0.75·L) for our generators (their 24 of 32)"
  - "validation split carved from T_f (they use fixed 60/20/20 splits); 1 seed (they average 3)"
  - "graphs stored as features in our shard files (they store per-sample .pt)"
  - "wandb not used (their training.py imports it at module level: stub or install)"
parity_test: pending (cluster, D65)
reproduction_test: not_done
```

Open items before coding: PyG wheels (pyg_lib, torch_scatter, torch_sparse) for torch 2.9.1; ragged graph storage in the feature shards; disk size (estimate 2–3 MB per sample for 40 layers × 40 heads).
