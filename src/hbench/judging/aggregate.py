"""Vote aggregation (prompt §2.4): plurality vote, tie → discard, PARSE_ERROR = missing vote.
Label mapping: CORRECT → 0, INCORRECT → 1, REFUSAL → discard."""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
from sklearn.metrics import cohen_kappa_score

from hbench.judging.labels import CORRECT, INCORRECT, PARSE_ERROR, REFUSAL

KEPT, DISCARD_REFUSAL, DISCARD_TIE, DISCARD_NO_VOTE = "kept", "refusal", "tie", "no_valid_vote"


def plurality(votes: list[str]) -> tuple[str | None, str]:
    """Return (winning label or None, outcome). Outcome is one of: "label", "tie", "no_valid_vote"."""
    counts = Counter(v for v in votes if v != PARSE_ERROR)
    if not counts:
        return None, DISCARD_NO_VOTE  # DECISIONS.md D6
    top = max(counts.values())
    winners = [c for c, k in counts.items() if k == top]
    if len(winners) > 1:
        return None, DISCARD_TIE
    return winners[0], "label"


def map_label(votes: list[str]) -> tuple[str | None, int | None, str]:
    """Return (ℓ_i, y_i, status)."""
    lab, outcome = plurality(votes)
    if outcome != "label":
        return None, None, outcome
    if lab == REFUSAL:
        return lab, None, DISCARD_REFUSAL
    return lab, {CORRECT: 0, INCORRECT: 1}[lab], KEPT


def mean_pairwise_cohen_kappa(votes_by_judge: dict[str, list[str]]) -> tuple[float, int]:
    """Mean Cohen's κ over judge pairs (DECISIONS.md D5). Each pair uses the samples where both
    votes are valid. Returns (mean κ, number of pairs with a defined κ). NaN if no pair is defined."""
    names = sorted(votes_by_judge)
    kappas = []
    for a, b in combinations(names, 2):
        va, vb = votes_by_judge[a], votes_by_judge[b]
        pairs = [(x, y) for x, y in zip(va, vb) if x != PARSE_ERROR and y != PARSE_ERROR]
        if len(pairs) < 2:
            continue
        xa, xb = zip(*pairs)
        if len(set(xa) | set(xb)) < 2:
            continue  # κ is undefined when both judges always give the same single label
        k = cohen_kappa_score(xa, xb, labels=[CORRECT, INCORRECT, REFUSAL])
        if not np.isnan(k):
            kappas.append(float(k))
    return (float(np.mean(kappas)) if kappas else float("nan")), len(kappas)
