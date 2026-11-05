"""Import modules from the pinned third-party checkouts without changing them. ICR's code lives in a
top-level package named `src`, so it is loaded under a private name instead of being put on sys.path."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

THIRD_PARTY = Path(__file__).resolve().parents[3] / "third_party"


def load_package_module(alias: str, package_dir: Path, module: str):
    """Load `package_dir/module.py` as `alias.module`, with `alias` a package rooted at package_dir
    (so relative imports inside it work)."""
    if alias not in sys.modules:
        pkg = types.ModuleType(alias)
        pkg.__path__ = [str(package_dir)]
        sys.modules[alias] = pkg
    name = f"{alias}.{module}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, package_dir / f"{module}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
