#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checkpointing = _load(
    "frontres_checkpointing_noise_std_contract",
    ROOT / "rsl_rl" / "runners" / "frontres_checkpointing.py",
)


def test_stage3_skips_legacy_12d_std_and_resets_runtime_6d_std() -> None:
    policy = SimpleNamespace(std=torch.ones(6) * 0.25)
    model_state = {"std": torch.arange(12, dtype=torch.float32)}

    loaded = checkpointing._copy_policy_noise_state(policy, model_state)
    assert not loaded
    torch.testing.assert_close(policy.std, torch.ones(6) * 0.25)

    checkpointing._reset_policy_noise_state(
        policy,
        init_noise_std=0.4,
        noise_std_type="scalar",
        device=torch.device("cpu"),
    )
    assert tuple(policy.std.shape) == (6,)
    torch.testing.assert_close(policy.std, torch.ones(6) * 0.4)

    frontres_mean = torch.zeros(3, 6)
    expanded = policy.std.expand_as(frontres_mean)
    assert tuple(expanded.shape) == (3, 6)
    print(
        "[probe stage3_noise_std] "
        f"checkpoint_std_shape={tuple(model_state['std'].shape)} "
        f"runtime_std_shape={tuple(policy.std.shape)} "
        f"expanded_shape={tuple(expanded.shape)}",
        flush=True,
    )


def main() -> None:
    test_stage3_skips_legacy_12d_std_and_resets_runtime_6d_std()
    print("frontres_stage3_noise_std_migration_contract: ok")


if __name__ == "__main__":
    main()
