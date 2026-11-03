"""Shared-pass feature extraction (ADDING_A_METHOD.md). lapeigvals is a git submodule at the
pinned commit, put on sys.path here (DECISIONS.md D25); its code is not changed."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

LAPEIGVALS_DIR = Path(__file__).resolve().parents[3] / "third_party" / "lapeigvals"
LAPEIGVALS_COMMIT = "74f885c69399ed525f4eecd2574c218fac7235d7"

if str(LAPEIGVALS_DIR) not in sys.path:
    sys.path.insert(0, str(LAPEIGVALS_DIR))

_precision_before = torch.get_float32_matmul_precision()
import hallucinations  # noqa: E402,F401  (sets torch.set_float32_matmul_precision("high") on import)

# Undo the import side effect; the extract stage sets the precision from the config (DECISIONS.md D29).
torch.set_float32_matmul_precision(_precision_before)
