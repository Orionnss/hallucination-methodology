"""Model revision from the HF cache path (transformers 5 no longer sets config._commit_hash)."""

import huggingface_hub

from hbench.models import _snapshot_commit


def test_snapshot_commit_from_cache_path(monkeypatch, tmp_path):
    p = "/c/hub/models--org--m/snapshots/abc123/config.json"
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: p)
    assert _snapshot_commit("org/m", None) == "abc123"


def test_snapshot_commit_local_dir_and_missing(monkeypatch, tmp_path):
    assert _snapshot_commit(str(tmp_path), None) is None
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    assert _snapshot_commit("org/m", None) is None
