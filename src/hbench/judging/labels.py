from __future__ import annotations

import json

CORRECT, INCORRECT, REFUSAL = "CORRECT", "INCORRECT", "REFUSAL"
LABELS = (CORRECT, INCORRECT, REFUSAL)
PARSE_ERROR = "PARSE_ERROR"


def label_json(label: str) -> str:
    return json.dumps({"label": label})


def parse_vote(raw: str) -> str | None:
    """Parse a judge output into exactly one of the three labels, or None if it is not valid."""
    try:
        obj = json.loads(raw.strip())
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(obj, dict) or set(obj) != {"label"}:
        return None
    lab = obj["label"]
    return lab if lab in LABELS else None
