#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[4]
RSL_SOURCE = ROOT / "source/rsl_rl"
CONFIG = ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_mosaic_cfg.py"
RUNNER = ROOT / "source/rsl_rl/rsl_rl/runners/on_policy_runner.py"
if str(RSL_SOURCE) not in sys.path:
    sys.path.insert(0, str(RSL_SOURCE))

from rsl_rl.modules.front_residual_actor_critic import FrontRESActorCritic
from rsl_rl.modules.frontres_observation_layout import resolve_frontres_v015_observation_authority


class FakeGMTNormalizer:
    def __init__(self, dim: int):
        self._mean = torch.zeros(1, dim)


class FakeGMTPolicy:
    def __init__(self):
        self.inputs: list[torch.Tensor] = []

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        self.inputs.append(obs.detach().clone())
        return obs


def _policy(*, num_frontres_obs: int) -> FrontRESActorCritic:
    policy = FrontRESActorCritic.__new__(FrontRESActorCritic)
    policy.num_actor_obs = 928
    policy.num_frontres_obs = num_frontres_obs
    policy.num_task_corrections = 6
    policy.gmt_policy_obs_dim = 770
    policy.gmt_actor_input_dim = 770
    policy.gmt_normalizer = FakeGMTNormalizer(770)
    policy.gmt_policy = FakeGMTPolicy()
    policy.ref_vel_estimator = None
    policy._pad_observations_for_gmt = lambda obs: obs
    return policy


def test_v015_layout_resolves_928_to_158_plus_770() -> None:
    authority = resolve_frontres_v015_observation_authority(
        environment_obs_dim=870,
        configured_frontres_prefix_dim=100,
        actor_tail_dim=58,
        gmt_suffix_dim=770,
    )

    assert authority.environment_obs_dim == 870
    assert authority.actor_tail_dim == 58
    assert authority.current_frontres_prefix_dim == 100
    assert authority.combined_obs_dim == 928
    assert authority.frontres_visible_dim == 158
    assert authority.gmt_suffix_dim == 770
    assert authority.combined_obs_dim == authority.frontres_visible_dim + authority.gmt_suffix_dim
    print("T-928-layout/T-158-actor: 870+58=928 and 58+100=158", flush=True)


def test_v015_layout_rejects_zero_frontres_prefix() -> None:
    try:
        resolve_frontres_v015_observation_authority(
            environment_obs_dim=870,
            configured_frontres_prefix_dim=0,
            actor_tail_dim=58,
            gmt_suffix_dim=770,
        )
    except ValueError as exc:
        assert "num_frontres_obs=0" in str(exc)
    else:
        raise AssertionError("v015 must reject num_frontres_obs=0")
    print("T-num-frontres-zero-reject", flush=True)


def test_actor_sees_prefix_while_gmt_sees_unchanged_suffix() -> None:
    policy = _policy(num_frontres_obs=158)
    frontres_prefix = torch.arange(2 * 158, dtype=torch.float32).reshape(2, 158)
    gmt_suffix = 10_000.0 + torch.arange(2 * 770, dtype=torch.float32).reshape(2, 770)
    combined = torch.cat([frontres_prefix, gmt_suffix], dim=-1)

    actor_obs, ref_vel, estimator_obs = FrontRESActorCritic._parse_observations(policy, combined)
    torch.testing.assert_close(actor_obs, frontres_prefix)
    torch.testing.assert_close(policy._cached_full_policy_obs, combined)
    assert tuple(actor_obs.shape) == (2, 158)
    assert not bool((actor_obs >= 10_000.0).any().item())

    gmt_action = FrontRESActorCritic._run_gmt_direct(policy, actor_obs, ref_vel, estimator_obs)
    assert len(policy.gmt_policy.inputs) == 1
    assert tuple(policy.gmt_policy.inputs[0].shape) == (2, 770)
    torch.testing.assert_close(policy.gmt_policy.inputs[0], gmt_suffix)
    torch.testing.assert_close(gmt_action, gmt_suffix)
    print("T-770-GMT/T-frozen-GMT-isolation", flush=True)


def test_task_space_actor_rejects_zero_visibility_boundary() -> None:
    policy = _policy(num_frontres_obs=0)
    combined = torch.zeros(2, 928)
    try:
        FrontRESActorCritic._parse_observations(policy, combined)
    except ValueError as exc:
        assert "num_frontres_obs=0" in str(exc)
    else:
        raise AssertionError("task-space FEMR must not fall back to the full observation")


def test_v015_config_and_runner_bind_the_authority_resolver() -> None:
    config_text = CONFIG.read_text()
    runner_text = RUNNER.read_text()
    assert "num_frontres_obs       = 100" in config_text
    assert "resolve_frontres_v015_observation_authority(" in runner_text
    assert "self.policy_cfg[\"num_frontres_obs\"] = authority.frontres_visible_dim" in runner_text
    print("T-config/T-runner-resolution", flush=True)


if __name__ == "__main__":
    test_v015_layout_resolves_928_to_158_plus_770()
    test_v015_layout_rejects_zero_frontres_prefix()
    test_actor_sees_prefix_while_gmt_sees_unchanged_suffix()
    test_task_space_actor_rejects_zero_visibility_boundary()
    test_v015_config_and_runner_bind_the_authority_resolver()
    print("frontres_v015_observation_authority_contract: ok", flush=True)
