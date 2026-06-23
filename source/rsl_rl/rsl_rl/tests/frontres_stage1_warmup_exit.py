from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_warmup_module():
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / "source" / "rsl_rl" / "rsl_rl" / "runners" / "frontres_warmup.py"
    spec = importlib.util.spec_from_file_location("frontres_warmup_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    warmup = _load_warmup_module()
    should_exit = warmup.should_exit_after_frontres_stage1_warmup

    cases = [
        ("stage1_frontres_warmup", {"frontres_stage1_exit_after_warmup": True}, True, 200, True),
        ("stage2_flag_off", {"frontres_stage1_exit_after_warmup": False}, True, 200, False),
        ("not_frontres", {"frontres_stage1_exit_after_warmup": True}, False, 200, False),
        ("no_warmup_ran", {"frontres_stage1_exit_after_warmup": True}, True, 0, False),
        ("missing_flag", {}, True, 200, False),
    ]

    print("=== FrontRES Stage 1 Warmup Exit TEST ONLY ===")
    for name, cfg, is_frontres, warmup_iters, expected in cases:
        got = should_exit(cfg, is_frontres=is_frontres, warmup_iters=warmup_iters)
        status = "PASS" if got == expected else "FAIL"
        print(
            f"{name:22s} is_frontres={is_frontres} warmup_iters={warmup_iters:3d} "
            f"exit={got} expected={expected} {status}"
        )
        if got != expected:
            raise AssertionError(name)
    print("result: PASS")


if __name__ == "__main__":
    main()
