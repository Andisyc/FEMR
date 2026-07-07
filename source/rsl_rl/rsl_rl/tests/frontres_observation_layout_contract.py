#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import types

import torch


ROOT = Path(__file__).resolve().parents[4]
RSL_SOURCE = ROOT / "source/rsl_rl"
FRONTRES_RUNTIME = ROOT / "source/rsl_rl/rsl_rl/runners/frontres_runtime.py"
ON_POLICY_RUNNER = ROOT / "source/rsl_rl/rsl_rl/runners/on_policy_runner.py"
if str(RSL_SOURCE) not in sys.path:
    sys.path.insert(0, str(RSL_SOURCE))

frontres_stub = types.ModuleType("rsl_rl.frontres")
runtime_diagnostics_stub = types.ModuleType("rsl_rl.frontres.runtime_diagnostics")
runtime_diagnostics_stub.maybe_print_frontres_restore_debug = lambda *args, **kwargs: None
sys.modules.setdefault("rsl_rl.frontres", frontres_stub)
sys.modules.setdefault("rsl_rl.frontres.runtime_diagnostics", runtime_diagnostics_stub)
spec = importlib.util.spec_from_file_location("frontres_runtime_under_test", FRONTRES_RUNTIME)
frontres_runtime = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(frontres_runtime)
apply_obs_normalizer = frontres_runtime.apply_obs_normalizer
from rsl_rl.modules.front_residual_actor_critic import FrontRESActorCritic
from rsl_rl.modules.frontres_observation_layout import split_frontres_policy_obs


class FakeGMTNormalizer:
    def __init__(self, dim: int):
        self._mean = torch.full((1, dim), 100.0)
        self._std = torch.full((1, dim), 10.0)
        self.calls: list[tuple[int, ...]] = []

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(obs.shape))
        return (obs - self._mean) / (self._std + 1e-8)


class FakeGMTPolicy:
    def __init__(self):
        self.calls: list[tuple[int, ...]] = []

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(obs.shape))
        return obs


def test_frontres_obs_layout_splits_extra_prefix_and_gmt_suffix() -> None:
    extra_dim = 100
    gmt_dim = 770
    extra = torch.full((2, extra_dim), 12.0)
    gmt = torch.full((2, gmt_dim), 130.0)
    obs = torch.cat([extra, gmt], dim=-1)

    extra_part, gmt_part = split_frontres_policy_obs(obs, gmt_dim)

    assert extra_part is not None
    assert tuple(extra_part.shape) == (2, extra_dim)
    assert tuple(gmt_part.shape) == (2, gmt_dim)
    torch.testing.assert_close(extra_part, extra)
    torch.testing.assert_close(gmt_part, gmt)


def test_apply_obs_normalizer_preserves_actor_layout_and_normalizes_suffix() -> None:
    extra_dim = 100
    gmt_dim = 770
    runner = SimpleNamespace(
        _frontres_gmt_obs_dim=gmt_dim,
        _frontres_extra_mean=torch.full((1, extra_dim), 10.0),
        _frontres_extra_std=torch.full((1, extra_dim), 2.0),
        obs_normalizer=FakeGMTNormalizer(gmt_dim),
    )
    obs = torch.cat(
        [
            torch.full((2, extra_dim), 12.0),
            torch.full((2, gmt_dim), 130.0),
        ],
        dim=-1,
    )

    normalized = apply_obs_normalizer(runner, obs)

    expected_extra = torch.full((2, extra_dim), 1.0)
    expected_gmt = torch.full((2, gmt_dim), 3.0)
    print(
        "[FrontRES Observation Layout Contract] "
        f"obs_shape={tuple(obs.shape)} "
        f"extra_shape={(2, extra_dim)} "
        f"gmt_shape={(2, gmt_dim)} "
        f"normalized_shape={tuple(normalized.shape)} "
        f"gmt_normalizer_calls={runner.obs_normalizer.calls}",
        flush=True,
    )
    assert tuple(normalized.shape) == (2, extra_dim + gmt_dim)
    assert runner.obs_normalizer.calls == [(2, gmt_dim)]
    torch.testing.assert_close(normalized[:, :extra_dim], expected_extra)
    torch.testing.assert_close(normalized[:, extra_dim:], expected_gmt)


def test_apply_obs_normalizer_falls_back_for_plain_gmt_obs() -> None:
    gmt_dim = 770
    runner = SimpleNamespace(
        _frontres_gmt_obs_dim=gmt_dim,
        obs_normalizer=FakeGMTNormalizer(gmt_dim),
    )
    obs = torch.full((2, gmt_dim), 120.0)

    normalized = apply_obs_normalizer(runner, obs)

    assert tuple(normalized.shape) == (2, gmt_dim)
    assert runner.obs_normalizer.calls == [(2, gmt_dim)]
    torch.testing.assert_close(normalized, torch.full((2, gmt_dim), 2.0))


def test_gmt_direct_uses_shared_layout_suffix() -> None:
    extra_dim = 100
    gmt_dim = 770
    policy = FrontRESActorCritic.__new__(FrontRESActorCritic)
    policy.gmt_normalizer = FakeGMTNormalizer(gmt_dim)
    policy.gmt_policy = FakeGMTPolicy()
    policy.ref_vel_estimator = None
    policy._cached_full_policy_obs = torch.cat(
        [
            torch.full((2, extra_dim), 12.0),
            torch.full((2, gmt_dim), 130.0),
        ],
        dim=-1,
    )
    policy._pad_observations_for_gmt = lambda obs: obs

    gmt_action = FrontRESActorCritic._run_gmt_direct(policy, torch.empty(2, 4), None, None)

    print(
        "[FrontRES Observation Layout GMT Direct] "
        f"cached_shape={tuple(policy._cached_full_policy_obs.shape)} "
        f"gmt_policy_calls={policy.gmt_policy.calls} "
        f"action_shape={tuple(gmt_action.shape)}",
        flush=True,
    )
    assert policy.gmt_policy.calls == [(2, gmt_dim)]
    assert tuple(gmt_action.shape) == (2, gmt_dim)


def test_frontres_train_mode_keeps_gmt_normalizer_frozen_by_contract() -> None:
    runner_text = ON_POLICY_RUNNER.read_text()
    print(
        "[FrontRES Observation Layout Freeze Contract] "
        "expects_train_mode_to_preserve_frozen_gmt_normalizer=True",
        flush=True,
    )
    assert "if self._frontres_gmt_obs_dim is not None:" in runner_text
    assert "self.obs_normalizer.eval()" in runner_text
    assert "self.obs_normalizer.until = 0" in runner_text


if __name__ == "__main__":
    test_frontres_obs_layout_splits_extra_prefix_and_gmt_suffix()
    test_apply_obs_normalizer_preserves_actor_layout_and_normalizes_suffix()
    test_apply_obs_normalizer_falls_back_for_plain_gmt_obs()
    test_gmt_direct_uses_shared_layout_suffix()
    test_frontres_train_mode_keeps_gmt_normalizer_frozen_by_contract()
    print("frontres_observation_layout_contract: ok")
