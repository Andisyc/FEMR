from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _install_import_stubs() -> None:
    modules = types.ModuleType("rsl_rl.modules")
    modules.FrontRESActorCritic = type("FrontRESActorCritic", (), {})
    sys.modules["rsl_rl.modules"] = modules

    diagnostics = types.ModuleType("rsl_rl.frontres.frontres_diagnostics")
    diagnostics.format_frontres_floor_alpha_diagnostics = lambda *args, **kwargs: ""
    diagnostics.format_frontres_optimization_diagnostics = lambda *args, **kwargs: ""
    diagnostics.format_frontres_preference_diagnostics = lambda *args, **kwargs: ""
    diagnostics.format_frontres_route_rho_diagnostics = lambda *args, **kwargs: ""
    sys.modules["rsl_rl.frontres.frontres_diagnostics"] = diagnostics


def _load_logging_module():
    _install_import_stubs()
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / "source" / "rsl_rl" / "rsl_rl" / "runners" / "frontres_runner_logging.py"
    spec = importlib.util.spec_from_file_location("frontres_runner_logging_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    logging = _load_logging_module()

    print("=== FrontRES Logging Scalar Guard TEST ONLY ===")
    assert logging._scalar_log_value("burst") is None
    assert logging._scalar_log_value({"mode": "burst"}) is None
    assert logging._scalar_log_value([1.0]) is None
    assert logging._scalar_log_value(1.25) == 1.25
    assert logging._scalar_log_value(True) == 1.0
    print("checks=string diagnostics are not TensorBoard scalars")
    print("result: PASS")


if __name__ == "__main__":
    main()
