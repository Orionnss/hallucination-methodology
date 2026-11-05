"""saplma-reimpl — case D reimplementation (no official code; approved, DECISIONS.md D66) of SAPLMA,
"The Internal State of an LLM Knows When It's Lying" (Azaria & Mitchell, Findings of EMNLP 2023,
https://arxiv.org/abs/2304.13734). The paper has no equations; each part below cites its section.

From the paper (Section 4, "SAPLMA"):
- input: the activation values of one hidden layer of the LLM as it reads the statement;
- classifier: feed-forward network with three hidden layers of 256, 128 and 64 units, ReLU, and a
  sigmoid output; Adam optimizer; trained for 5 epochs; no hyperparameter tuning;
- layers: five candidates for a 32-layer model (last, 28th, 24th, 20th, 16th), one model per layer.
Not specified in the paper (our choices): Adam learning rate 1e-3 (the Adam default) and batch size 32
(the Keras default) (D28); the token = the last answer token, EOS excluded (D8, D12, D31); BCE loss
(implied by the sigmoid output).
Differs from the paper: the layer is selected in the inner CV among all layers instead of reporting
five fixed-layer models (D11, D12); inputs are standardised inside the training fold; the statement is
the generated answer in its chat context, not a standalone true/false sentence.
"""

from __future__ import annotations

from hbench.pool.methods.reference import LastAnswerHidden, SaplmaMLP


class SaplmaReimplFeature(LastAnswerHidden):
    """Hidden state of the last answer token at every layer (hidden_states 1..L): x ∈ ℝ^{L × d}."""

    name = "last_answer_hidden"  # same features as the reference probe: shared feature files


class SaplmaReimplMLP(SaplmaMLP):
    """Section 4: 256-128-64 ReLU, sigmoid, Adam, 5 epochs (lr 1e-3 and batch 32 not in the paper)."""

    name = "saplma_reimpl_mlp"
