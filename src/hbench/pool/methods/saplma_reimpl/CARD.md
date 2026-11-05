# Method card: saplma-reimpl (case D, approved — DECISIONS.md D66)

```yaml
name: saplma-reimpl  # registry name: saplma_reimpl
paper: {title: "The Internal State of an LLM Knows When It's Lying", venue: "Findings of EMNLP 2023", year: 2023, url: "https://arxiv.org/abs/2304.13734"}
code: {repo: none (no official code), commit: n/a, license: n/a}
integration_case: D
feature_function: {file: src/hbench/pool/methods/saplma_reimpl/__init__.py, class: SaplmaReimplFeature (shares LastAnswerHidden features)}
classifier: {file: src/hbench/pool/methods/saplma_reimpl/__init__.py, class: SaplmaReimplMLP, hyperparameters: "paper §4: 256-128-64 ReLU, sigmoid, Adam, 5 epochs; not in the paper: lr 1e-3 (Adam default), batch 32 (Keras default), BCE", grid: "layer (inner CV, D12)"}
needs: {hidden_states: {layers: all, positions: last answer token (D8, D31)}}
reducer_ops: [hidden state of the last answer token at every layer]
fitted_ops: [layer selection, StandardScaler, MLP]
deviations_from_paper: ["layer selected in the inner CV among all layers, not five fixed-layer models (paper §4: last, 28, 24, 20, 16 of 32)", "inputs standardised in the training fold", "statement = generated answer in chat context", "token not specified in the paper: last answer token, EOS excluded"]
parity_test: n/a (no original code)
reproduction_test: not_done
report_group: reimplemented
```
