#!/usr/bin/env python3
"""Deterministic TEST-05 contracts for the TRAIN-v016 Critic observation."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rsl_rl.modules.frontres_observation_layout import (
    FRONTRES_FUTURE_INTENT_LAYOUT_VERSION,
    build_frontres_future_intent_tail,
    compose_frontres_v016_critic_observation,
    resolve_frontres_future_intent_layout,
)


def _provenance(batch_size: int) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "intent_q29_provenance": "deployment_noisy_q29",
            "intent_q29_source": "motion_internal_q29",
            "carrier_kind": "local_scenario",
            "clean_continuation_provenance": "clean_gmt_only",
        }
        for _ in range(batch_size)
    )


def _expect_error(call, text: str) -> None:
    try:
        call()
    except (TypeError, ValueError) as exc:
        assert text.lower() in str(exc).lower(), str(exc)
        return
    raise AssertionError("expected observation contract rejection")


def main() -> None:
    layout = resolve_frontres_future_intent_layout((1, 2), FRONTRES_FUTURE_INTENT_LAYOUT_VERSION)
    intent = torch.arange(2 * 3 * 29, dtype=torch.float32).reshape(2, 3, 29)
    tail = build_frontres_future_intent_tail(intent, layout=layout, provenance=_provenance(2))
    current = torch.arange(2 * 289, dtype=torch.float32).reshape(2, 289)

    critic = compose_frontres_v016_critic_observation(current, tail)
    assert tuple(critic.shape) == (2, 347)
    torch.testing.assert_close(critic[:, :289], current)
    torch.testing.assert_close(critic[:, 289:], tail)
    assert not critic.requires_grad and critic.data_ptr() != current.data_ptr()

    order = torch.tensor([1, 0])
    permuted = compose_frontres_v016_critic_observation(current[order], tail[order])
    torch.testing.assert_close(permuted, critic[order])

    _expect_error(lambda: compose_frontres_v016_critic_observation(current, tail[:, :6]), "58")
    _expect_error(lambda: compose_frontres_v016_critic_observation(current[:, :-1], tail), "289")
    bad = tail.clone()
    bad[0, 0] = float("nan")
    _expect_error(lambda: compose_frontres_v016_critic_observation(current, bad), "finite")
    grad_tail = tail.clone().requires_grad_(True)
    _expect_error(lambda: compose_frontres_v016_critic_observation(current, grad_tail), "detached")

    from frontres_v015_one_action_k_contract import _load_owners

    _load_owners()
    one_action = sys.modules["rsl_rl.runners.frontres_segment_one_action_k"]
    policy_rows = torch.arange(4, dtype=torch.long)
    source_index = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    reset_states = torch.zeros(8, 347)
    reset_states[1, 0] = 0.25
    reset_states[2:4] = 1.0
    reset_states[3, 0] += 0.5
    canonical, observed_max = one_action._canonicalize_frontres_v016_segment_state_rows(
        reset_states,
        policy_rows=policy_rows,
        source_index=source_index,
        name="Critic observation",
    )
    assert observed_max == 0.5
    torch.testing.assert_close(canonical[0], canonical[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(canonical[2], canonical[3], rtol=0.0, atol=0.0)
    bad_states = reset_states.clone()
    bad_states[1, 0] = float("nan")
    try:
        one_action._canonicalize_frontres_v016_segment_state_rows(
            bad_states,
            policy_rows=policy_rows,
            source_index=source_index,
            name="Critic observation",
        )
    except ValueError as exc:
        assert "detached finite" in str(exc)
    else:
        raise AssertionError("TRAIN-v016 accepted a non-finite policy observation")

    formal_transaction = sys.modules["rsl_rl.runners.frontres_segment_formal_transaction"]
    trace = {
        "role_row_count": 8,
        "current_command_dim": 58,
        "raw_observation_dim": 870,
        "q29_tail_dim": 58,
        "combined_observation_dim": 928,
        "normalized_observation_dim": 928,
        "femr_visible_dim": 158,
        "gmt_suffix_dim": 770,
        "gmt_input_dim": 770,
        "critic_current_observation_dim": 289,
        "critic_future_intent_dim": 58,
        "critic_observation_dim": 347,
        "post_advance_gmt_read_count": 8,
        "actor_raw_observation_max_abs_diff": 0.25,
        "critic_raw_observation_max_abs_diff": 0.5,
        "actor_segment_state_max_abs_diff": 0.0,
        "critic_segment_state_max_abs_diff": 0.0,
    }
    formal_transaction._require_frontres_v016_observation_trace(
        trace,
        policy_row_count=4,
        label="training",
    )
    stale_trace = dict(trace, critic_observation_dim=289)
    try:
        formal_transaction._require_frontres_v016_observation_trace(
            stale_trace,
            policy_row_count=4,
            label="training",
        )
    except RuntimeError as exc:
        assert "critic_observation_dim" in str(exc) and "347" in str(exc)
    else:
        raise AssertionError("TRAIN-v016 accepted the stale 289D Critic trace")

    raw = torch.arange(2 * 870, dtype=torch.float32).reshape(2, 870)

    class _Env:
        def get_observations(self):
            return raw.clone(), {
                "observations": {
                    "critic": current.clone(),
                    "teacher": current.clone(),
                }
            }

    runner = SimpleNamespace(
        env=_Env(),
        device=torch.device("cpu"),
        policy_obs_type=None,
        privileged_obs_type="critic",
        teacher_obs_type="teacher",
        ref_vel_estimator_obs_type=None,
        alg=SimpleNamespace(
            frontres_future_offsets=(1, 2),
            frontres_formal_transaction_enabled=True,
            policy=SimpleNamespace(num_frontres_obs=158),
        ),
        _frontres_future_intent_layout=layout,
        _frontres_future_intent_actor_context_dim=58,
        _frontres_critic_observation_dim=347,
        _frontres_gmt_obs_dim=770,
        _append_frontres_future_intent_context=lambda obs: torch.cat([tail, obs], dim=-1),
        _apply_obs_normalizer=lambda obs: obs,
        privileged_obs_normalizer=lambda obs: obs,
        teacher_obs_normalizer=lambda obs: obs,
    )
    original_trace = one_action.update_frontres_observation_trace
    one_action.update_frontres_observation_trace = lambda *_args, **_kwargs: None
    try:
        live = one_action._read_live_observations(runner)
    finally:
        one_action.update_frontres_observation_trace = original_trace
    torch.testing.assert_close(live.obs[:, :58], tail)
    torch.testing.assert_close(live.obs[:, 58:158], raw[:, :100])
    torch.testing.assert_close(live.obs[:, 158:], raw[:, 100:])
    torch.testing.assert_close(live.privileged_obs[:, :289], current)
    torch.testing.assert_close(live.privileged_obs[:, 289:], tail)

    print("frontres_v016_state_value_observation_contract: 289+58=347 exact", flush=True)


if __name__ == "__main__":
    main()
