from collections import Counter
from itertools import product

import pandas as pd
import pytest

from hbench.judging.aggregate import (
    DISCARD_NO_VOTE,
    DISCARD_REFUSAL,
    DISCARD_TIE,
    KEPT,
    map_label,
    mean_pairwise_cohen_kappa,
    plurality,
)
from hbench.judging.labels import CORRECT as C, INCORRECT as I, PARSE_ERROR as P, REFUSAL as R, parse_vote
from hbench.stages.aggregate import aggregate_cell

SYMBOLS = [C, I, R, P]


def reference(votes):
    """Direct reading of prompt §2.4: PARSE_ERROR is a missing vote; argmax set size > 1 → tie."""
    valid = [v for v in votes if v != P]
    if not valid:
        return None, "no_valid_vote"
    n = Counter(valid)
    best = {c for c in n if n[c] == max(n.values())}
    return (None, "tie") if len(best) > 1 else (best.pop(), "label")


@pytest.mark.parametrize("R_", [2, 3, 4])
def test_plurality_exhaustive(R_):
    seen = Counter()
    for votes in product(SYMBOLS, repeat=R_):
        got = plurality(list(votes))
        assert got == reference(votes), votes
        seen[got[1]] += 1
    assert seen["tie"] > 0 and seen["label"] > 0 and seen["no_valid_vote"] == 1


@pytest.mark.parametrize(
    "votes,expected",
    [
        # R = 2
        ([C, I], (None, DISCARD_TIE)),
        ([C, R], (None, DISCARD_TIE)),
        ([I, R], (None, DISCARD_TIE)),
        ([C, C], (C, "label")),
        ([C, P], (C, "label")),
        ([P, P], (None, DISCARD_NO_VOTE)),
        # R = 3
        ([C, I, R], (None, DISCARD_TIE)),
        ([C, I, P], (None, DISCARD_TIE)),
        ([C, C, I], (C, "label")),
        ([R, R, C], (R, "label")),
        ([P, P, I], (I, "label")),
        # R = 4
        ([C, C, I, I], (None, DISCARD_TIE)),
        ([C, C, R, R], (None, DISCARD_TIE)),
        ([C, I, R, P], (None, DISCARD_TIE)),
        ([C, C, I, R], (C, "label")),
        ([I, I, P, P], (I, "label")),
        ([C, P, P, P], (C, "label")),
    ],
)
def test_plurality_cases(votes, expected):
    assert plurality(votes) == expected


def test_label_mapping():
    assert map_label([C, C, I]) == (C, 0, KEPT)
    assert map_label([I, I, C]) == (I, 1, KEPT)
    assert map_label([R, R, C]) == (R, None, DISCARD_REFUSAL)
    assert map_label([C, I, R]) == (None, None, DISCARD_TIE)
    assert map_label([P, P, P]) == (None, None, DISCARD_NO_VOTE)


def test_aggregate_cell_counts():
    cells = {
        "k0": [C, C, I],  # kept, y=0
        "k1": [I, I, I],  # kept, y=1
        "k2": [I, C, P],  # tie
        "k3": [R, R, I],  # refusal
        "k4": [P, P, P],  # no valid vote
        "k5": [C, I, R],  # tie
    }
    judges = ["a", "b", "c"]
    rows = [{"gen_key": k, "sample_id": k, "judge": j, "vote": v} for k, vs in cells.items() for j, v in zip(judges, vs)]
    labels, st = aggregate_cell(pd.DataFrame(rows), judges, list(cells))
    assert st["n"] == 6 and st["n_kept"] == 2
    assert st["tie_rate"] == pytest.approx(2 / 6)
    assert st["refusal_rate"] == pytest.approx(1 / 6)
    assert st["no_valid_vote_rate"] == pytest.approx(1 / 6)
    assert st["parse_error_count"] == 4
    assert (st["n_correct"], st["n_hallucination"]) == (1, 1)
    assert labels.set_index("gen_key").loc["k1", "y"] == 1


def test_aggregate_cell_missing_vote_raises():
    rows = [{"gen_key": "k0", "sample_id": "k0", "judge": "a", "vote": C}]
    with pytest.raises(RuntimeError, match="missing"):
        aggregate_cell(pd.DataFrame(rows), ["a"], ["k0", "k1"])


def test_cohen_kappa_mean_and_parse_errors():
    from sklearn.metrics import cohen_kappa_score

    a = [C, I, C, I, R, C]
    b = [C, I, I, I, R, P]
    c = [C, C, C, I, R, C]
    k, n_pairs = mean_pairwise_cohen_kappa({"a": a, "b": b, "c": c})
    labs = [C, I, R]
    ab = cohen_kappa_score(a[:5], b[:5], labels=labs)
    ac = cohen_kappa_score(a, c, labels=labs)
    bc = cohen_kappa_score(b[:5], c[:5], labels=labs)
    assert n_pairs == 3 and k == pytest.approx((ab + ac + bc) / 3)
    assert mean_pairwise_cohen_kappa({"a": a, "b": a})[0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"label": "CORRECT"}', C),
        (' {"label":"REFUSAL"}\n', R),
        ('{"label": "correct"}', None),
        ('{"label": "CORRECT", "x": 1}', None),
        ("CORRECT", None),
        ("", None),
    ],
)
def test_parse_vote(raw, expected):
    assert parse_vote(raw) == expected
