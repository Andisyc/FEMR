#!/usr/bin/env python3
"""CPU-only regression for the real v015 Adam optimizer step counter."""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch


ROOT = Path(__file__).resolve().parents[4]
ALG_PATH = ROOT / "source" / "rsl_rl" / "rsl_rl" / "algorithms" / "frontres_unified.py"


def _load_algorithm_module():
    rsl_rl_pkg = ModuleType("rsl_rl")
    rsl_rl_pkg.__path__ = []
    modules = ModuleType("rsl_rl.modules")
    storage = ModuleType("rsl_rl.storage")

    class _ActorCritic(torch.nn.Module):
        pass

    class _FrontRESActorCritic(torch.nn.Module):
        pass

    class _ResidualActorCritic(torch.nn.Module):
        pass

    class _RolloutStorage:
        class Transition:
            pass

    modules.ActorCritic = _ActorCritic
    modules.FrontRESActorCritic = _FrontRESActorCritic
    modules.ResidualActorCritic = _ResidualActorCritic
    storage.RolloutStorage = _RolloutStorage
    sys.modules["rsl_rl"] = rsl_rl_pkg
    sys.modules["rsl_rl.modules"] = modules
    sys.modules["rsl_rl.storage"] = storage

    spec = importlib.util.spec_from_file_location("frontres_v015_real_optimizer_owner", ALG_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_algorithm(module, *, formal: bool):
    kwargs = {}
    if formal:
        kwargs.update(
            frontres_training_objective="segment_replay_hrl",
            frontres_segment_replay_enabled=True,
            frontres_segment_live_runner_enabled=True,
            frontres_v015_formal_transaction_enabled=True,
            frontres_v015_local_sentinel_only=True,
            frontres_segment_advantage_normalization="grouped_scale_only",
            lambda_supervised=0.0,
            lambda_supervised_min=0.0,
        )
    return module.FrontRESUnified(torch.nn.Linear(2, 1), **kwargs)


def _take_optimizer_step(optimizer, policy) -> None:
    optimizer.zero_grad()
    policy(torch.ones(1, 2)).sum().backward()
    optimizer.step()


def test_t_real_adam_counter_is_exact_and_persistent() -> None:
    module = _load_algorithm_module()
    algorithm = _build_algorithm(module, formal=True)
    optimizer = algorithm.optimizer

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.frontres_v015_step_count == 0
    _take_optimizer_step(optimizer, algorithm.policy)
    assert optimizer.frontres_v015_step_count == 1

    state = copy.deepcopy(optimizer.state_dict())
    restored = _build_algorithm(module, formal=True)
    restored.optimizer.load_state_dict(state)
    assert restored.optimizer.frontres_v015_step_count == 1
    _take_optimizer_step(restored.optimizer, restored.policy)
    assert restored.optimizer.frontres_v015_step_count == 2
    print("[T-real-adam/T-exact/T-persist] v015 Adam counts every step and survives optimizer state restore", flush=True)


def test_t_non_v015_optimizer_remains_plain_adam() -> None:
    module = _load_algorithm_module()
    algorithm = _build_algorithm(module, formal=False)

    assert type(algorithm.optimizer) is torch.optim.Adam
    assert not hasattr(algorithm.optimizer, "frontres_v015_step_count")
    print("[T-isolation] non-v015 FrontRES keeps the plain Adam owner", flush=True)


def main() -> None:
    test_t_real_adam_counter_is_exact_and_persistent()
    test_t_non_v015_optimizer_remains_plain_adam()
    print("frontres_v015_real_optimizer_counter_contract: ok", flush=True)


if __name__ == "__main__":
    main()
