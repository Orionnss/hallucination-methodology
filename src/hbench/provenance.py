"""Save the resolved config, the git commit, library versions, the GPU type and model revisions (prompt §3.2, §3.4)."""

from __future__ import annotations

import importlib.metadata as md
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from hbench.config import ExperimentCfg

_LIBS = ["torch", "transformers", "accelerate", "datasets", "scikit-learn", "pandas", "numpy", "pydantic", "pyarrow", "h5py"]


def git_commit() -> dict[str, str | bool]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def environment_info() -> dict:
    libs = {}
    for name in _LIBS:
        try:
            libs[name] = md.version(name)
        except md.PackageNotFoundError:
            libs[name] = None
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": libs,
        "cuda": torch.version.cuda,
        "gpus": gpus,
    }


def save_run_provenance(cfg: ExperimentCfg, stage: str) -> Path:
    """Write the provenance of one stage invocation. Each invocation gets its own file, so a resumed
    stage keeps the record of every run that wrote to the directory."""
    d = cfg.run_dir / "provenance"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.resolved.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    rec = {"stage": stage, "time_utc": stamp, "git": git_commit(), "env": environment_info()}
    path = d / f"{stage}-{stamp}.json"
    path.write_text(json.dumps(rec, indent=2))
    return path


def record_model_revision(cfg: ExperimentCfg, role: str, name: str, hf_id: str, commit: str | None) -> None:
    d = cfg.run_dir / "provenance"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "model_revisions.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    key = f"{role}:{name}"
    if key in data and commit and data[key]["commit"] != commit:
        raise RuntimeError(f"{key}: model revision changed between runs ({data[key]['commit']} → {commit})")
    data[key] = {"hf_id": hf_id, "commit": commit}
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
