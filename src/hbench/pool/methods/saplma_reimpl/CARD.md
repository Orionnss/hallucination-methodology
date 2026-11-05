# Method card: saplma-reimpl (case D, approved — DECISIONS.md D66)

```yaml
name: saplma-reimpl
paper: {title: "The Internal State of an LLM Knows When It's Lying", venue: "Findings of EMNLP 2023", year: 2023, url: "https://arxiv.org/abs/2304.13734"}
code: {repo: none (no official code), commit: n/a, license: n/a}
integration_case: D
feature_function: {file: src/hbench/pool/methods/reference.py, class: LastAnswerHidden}
classifier: {file: src/hbench/pool/methods/reference.py, class: SaplmaMLP, hyperparameters: "256-128-64 ReLU, sigmoid, Adam 1e-3, BCE, 5 epochs, batch 32 (D28)", grid: "layer (inner CV, D12)"}
needs: {hidden_states: {layers: all, positions: last answer token (D8, D31)}}
reducer_ops: [hidden state of the last answer token at every layer]
fitted_ops: [layer selection, StandardScaler, MLP]
deviations_from_paper: ["layer selected in the inner CV, not fixed", "inputs standardised in the training fold", "statement = generated answer in chat context"]
parity_test: n/a (no original code)
reproduction_test: not_done
report_group: reimplemented
```
