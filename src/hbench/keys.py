"""Cache keys: (dataset, sample ID, generator, prompt hash, decoding hash, [judge | extractor]) (prompt §3.2)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel


def stable_hash(obj: Any, length: int = 12) -> str:
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json")
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:length]


def unit_key(dataset: str, sample_id: str, generator: str, prompt_hash: str, decoding_hash: str, extra: str = "") -> str:
    parts = [dataset, str(sample_id), generator, prompt_hash, decoding_hash]
    if extra:
        parts.append(extra)
    return "|".join(parts)
