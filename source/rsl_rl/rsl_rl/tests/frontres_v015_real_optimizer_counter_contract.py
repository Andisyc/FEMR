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
    modules.__path__ = [str(ROOT / "source" / "rsl_rl" / "rsl_rl" / "modules")]
    storage = ModuleType("rsl_rl.storage")
    frontres = ModuleType("rsl_rl.frontres")
    frontres.__path__ = [str(ROOT / "source" / "rsl_rl" / "rsl_rl" / "frontres")]

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
    sys.modules["rsl_rl.frontres"] = frontres

    spec = importlib.util.spec_from_file_location("frontres_v015_real_optimizer_owner", ALG_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_algorithm(module, *, formal: bool, overrides=None):
    kwargs = {}
    policy = module.FrontRESActorCritic()
    policy.residual_actor = torch.nn.Linear(2, 2)
    policy.critic = torch.nn.Linear(2, 1)
    policy.register_buffer("std", torch.ones(2))
    if formal:
        kwargs.update(
            learning_rate=3.0e-7,
            critic_learning_rate=1.0e-5,
            frontres_segment_actor_joint_lr=1.0e-6,
            schedule="fixed",
            frontres_training_objective="segment_replay_hrl",
            frontres_segment_replay_enabled=True,
            frontres_segment_live_runner_enabled=True,
            frontres_formal_transaction_enabled=True,
            frontres_local_sentinel_only=True,
            frontres_segment_advantage_normalization="grouped_scale_only",
            lambda_supervised=0.0,
            lambda_supervised_min=0.0,
            frontres_segment_k_curriculum=(
    (8, 4, 200, 500, 1300, "lower-k8", 0.5, "linear-coupled-v1", 700, 2.381),
    (16, 4, 300, 300, 900, "lower-k16", 0.6, "linear-coupled-v1", 600, 2.381),
    (32, 4, 400, 300, 625, "lower-k32", 0.7, "linear-coupled-v1", 700, 2.381),
            ),
        )
    kwargs.update(dict(overrides or {}))
    return module.FrontRESUnified(policy, **kwargs)


def _take_optimizer_step(optimizer, policy) -> None:
    optimizer.zero_grad()
    (policy.residual_actor(torch.ones(1, 2)).sum() + policy.critic(torch.ones(1, 2)).sum()).backward()
    optimizer.step()


def test_t_real_adam_counter_is_exact_and_persistent() -> None:
    module = _load_algorithm_module()
    algorithm = _build_algorithm(module, formal=True)
    optimizer = algorithm.optimizer

    assert isinstance(optimizer, torch.optim.Adam)
    assert [group["frontres_role"] for group in optimizer.param_groups] == ["actor", "critic"]
    assert [group["lr"] for group in optimizer.param_groups] == [3.0e-7, 1.0e-5]
    actor_ids = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
    critic_ids = {id(parameter) for parameter in optimizer.param_groups[1]["params"]}
    assert actor_ids.isdisjoint(critic_ids)
    assert optimizer.frontres_step_count == 0
    _take_optimizer_step(optimizer, algorithm.policy)
    assert optimizer.frontres_step_count == 1

    state = copy.deepcopy(optimizer.state_dict())
    restored = _build_algorithm(module, formal=True)
    restored.optimizer.load_state_dict(state)
    assert restored.optimizer.frontres_step_count == 1
    _take_optimizer_step(restored.optimizer, restored.policy)
    assert restored.optimizer.frontres_step_count == 2
    print("[T-real-adam/T-exact/T-persist] v015 Adam counts every step and survives optimizer state restore", flush=True)


def test_t_non_v015_optimizer_remains_plain_adam() -> None:
    module = _load_algorithm_module()
    algorithm = _build_algorithm(module, formal=False)

    assert type(algorithm.optimizer) is torch.optim.Adam
    assert not hasattr(algorithm.optimizer, "frontres_step_count")
    print("[T-isolation] non-v015 FrontRES keeps the plain Adam owner", flush=True)


def test_t_v017_value_normalizer_config_fails_closed() -> None:
    module = _load_algorithm_module()
    invalid = (
        {"frontres_critic_value_normalization": "none"},
        {"frontres_critic_value_normalizer_decay": 0.8},
        {"frontres_critic_value_normalizer_scale_floor": 0.5},
    )
    for overrides in invalid:
        try:
            _build_algorithm(module, formal=True, overrides=overrides)
        except ValueError:
            pass
        else:
            raise AssertionError(f"TRAIN-v017 accepted invalid value-normalizer config: {overrides}")


def test_t_critic_only_preserves_actor_parameters_and_adam_state() -> None:
    module = _load_algorithm_module()
    algorithm = _build_algorithm(module, formal=True)
    actor_before = copy.deepcopy(algorithm.policy.residual_actor.state_dict())
    algorithm.optimizer.zero_grad(set_to_none=True)
    algorithm.policy.critic(torch.ones(1, 2)).sum().backward()
    algorithm.optimizer.step()
    for name, value in actor_before.items():
        torch.testing.assert_close(algorithm.policy.residual_actor.state_dict()[name], value)
    assert all(parameter not in algorithm.optimizer.state for parameter in algorithm.policy.residual_actor.parameters())
    assert any(parameter in algorithm.optimizer.state for parameter in algorithm.policy.critic.parameters())
    assert algorithm.optimizer.frontres_step_count == 1


def test_t_split_groups_reject_unowned_trainable_parameter() -> None:
    module = _load_algorithm_module()
    policy = module.FrontRESActorCritic()
    policy.residual_actor = torch.nn.Linear(2, 2)
    policy.critic = torch.nn.Linear(2, 1)
    policy.unowned = torch.nn.Parameter(torch.ones(1))
    policy.register_buffer("std", torch.ones(2))
    try:
        module.FrontRESUnified._collect_trainable_param_groups(
            policy,
            actor_learning_rate=3.0e-7,
            critic_learning_rate=1.0e-5,
        )
    except ValueError as exc:
        assert "exhaust all trainable policy parameters" in str(exc), str(exc)
    else:
        raise AssertionError("v015 split groups must reject an unowned trainable parameter")


def main() -> None:
    test_t_real_adam_counter_is_exact_and_persistent()
    test_t_non_v015_optimizer_remains_plain_adam()
    test_t_v017_value_normalizer_config_fails_closed()
    test_t_critic_only_preserves_actor_parameters_and_adam_state()
    test_t_split_groups_reject_unowned_trainable_parameter()
    print("frontres_v015_real_optimizer_counter_contract: ok", flush=True)


if __name__ == "__main__":
    main()
