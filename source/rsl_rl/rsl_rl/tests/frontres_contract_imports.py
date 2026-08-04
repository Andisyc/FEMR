"""Package composition helper for isolated FrontRES contract modules.

Focused contracts often load one owner directly from its file.  Production
owners now use normal package imports, so tests must provide the same package
boundary without importing the simulator-heavy runner facade.
"""

from __future__ import annotations

from pathlib import Path
import sys
import types


def _package(name: str, path: Path) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    paths = list(getattr(module, "__path__", []))
    value = str(path)
    if value not in paths:
        paths.append(value)
    module.__path__ = paths
    return module


def install_frontres_contract_packages(rsl_root: str | Path | None = None) -> None:
    """Install lightweight package paths, never the simulator-facing facades."""

    root = Path(rsl_root) if rsl_root is not None else Path(__file__).resolve().parents[1]
    rsl_rl = _package("rsl_rl", root)
    frontres = _package("rsl_rl.frontres", root / "frontres")
    runners = _package("rsl_rl.runners", root / "runners")
    rsl_rl.frontres = frontres
    rsl_rl.runners = runners


__all__ = ["install_frontres_contract_packages"]
