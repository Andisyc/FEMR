#!/usr/bin/env python3
"""Deterministic TEST-18 final-consumer contracts for TRAIN-v017 telemetry."""

from __future__ import annotations

from dataclasses import replace

import torch

from frontres_v015_transaction_route_contract import _request
from rsl_rl.runners.frontres_segment_formal_transaction import run_frontres_formal_transaction_update
from rsl_rl.runners.frontres_segment_runtime_types import open_frontres_checkpoint_transaction_barrier
from rsl_rl.runners.frontres_segment_training_telemetry import build_frontres_transaction_telemetry


def _expect_error(call, text: str) -> None:
    try:
        call()
    except (RuntimeError, ValueError) as exc:
        assert text.lower() in str(exc).lower(), str(exc)
        return
    raise AssertionError("expected telemetry contract rejection")


def main() -> None:
    runner, request, _policy = _request()
    runner.alg.frontres_method_contract_id = "FRS-METHOD-v020"
    runner.alg.frontres_optimization_contract_id = "FRS-PPO-v008"
    runner.alg.frontres_training_contract_id = "FRS-TRAIN-v019"
    runner.alg.frontres_return_utility_id = "symmetric-log-gain-g0-1-v1"
    runner.alg.frontres_return_utility_scale = 1.0
    runner.alg.max_grad_norm = 0.5

    active_m = request.active_m
    shared_actor = torch.tensor([[1.0, 0.0]] * active_m + [[0.0, 1.0]] * active_m)
    shared_critic = torch.zeros(2 * active_m, 449)
    shared_critic[:active_m, 0] = 1.0
    shared_critic[active_m:, 1] = 1.0
    batch = request.candidate_batches[0]
    sealed_batch = replace(
        batch,
        observations=shared_actor,
        privileged_observations=shared_critic,
        advantages=(
            torch.sign(batch.returns) * torch.log1p(torch.abs(batch.returns)) - batch.old_values
        ),
    )
    request = replace(request, candidate_batches=(sealed_batch,))

    open_frontres_checkpoint_transaction_barrier(runner)
    result = run_frontres_formal_transaction_update(runner, request)
    telemetry = build_frontres_transaction_telemetry(result, ppo=result.ppo_result)
    assert telemetry["method_contract_id"] == "FRS-METHOD-v020"
    assert telemetry["optimization_contract_id"] == "FRS-PPO-v008"
    assert telemetry["training_contract_id"] == "FRS-TRAIN-v019"
    assert telemetry["checkpoint_format"] == "frontres-v019-checkpoint-v14"
    assert telemetry["actor_observation_dim"] == 158
    assert telemetry["critic_observation_dim"] == 449
    assert telemetry["gmt_observation_dim"] == 770
    assert telemetry["critic_value_kind"] == "state_value"
    assert telemetry["critic_action_conditioned"] is False
    assert telemetry["critic_target_id"] == "segment-exact-m-mean-symlog-v1"
    assert telemetry["critic_value_targets"] == tuple(result.ppo_result.critic_value_targets)
    assert telemetry["actor_gradient_post_clip_norm"] <= 0.500001
    assert telemetry["critic_gradient_post_clip_norm"] <= 0.500001
    assert telemetry["gradient_clip_identity"] == "separate-actor-critic-v1"
    assert telemetry["critic_value_normalization_id"] == "ema-target-std-nonamplifying-v1"
    assert telemetry["critic_value_scale"] >= 1.0
    assert telemetry["critic_value_normalizer_decay"] == 0.9
    assert telemetry["critic_value_normalizer_scale_floor"] == 1.0
    assert telemetry["critic_value_normalizer_update_count_before"] == 0
    assert telemetry["critic_value_normalizer_update_count_after"] == 1
    assert telemetry["critic_scaled_value_loss"] == result.ppo_result.value_loss.detach().cpu().item()

    missing = dict(result.diagnostics)
    missing.pop("critic_target_id")
    _expect_error(
        lambda: build_frontres_transaction_telemetry(replace(result, diagnostics=missing), ppo=result.ppo_result),
        "critic_target_id",
    )
    nonfinite = dict(result.diagnostics, critic_gradient_post_clip_norm=float("nan"))
    _expect_error(
        lambda: build_frontres_transaction_telemetry(replace(result, diagnostics=nonfinite), ppo=result.ppo_result),
        "critic_gradient_post_clip_norm",
    )
    malformed = dict(result.diagnostics, critic_value_targets=(0.0,))
    _expect_error(
        lambda: build_frontres_transaction_telemetry(replace(result, diagnostics=malformed), ppo=result.ppo_result),
        "raw/utility/target/advantage",
    )
    missing_normalizer = dict(result.diagnostics)
    missing_normalizer.pop("critic_value_scale")
    _expect_error(
        lambda: build_frontres_transaction_telemetry(
            replace(result, diagnostics=missing_normalizer), ppo=result.ppo_result
        ),
        "critic_value_scale",
    )

    print("frontres_v016_runtime_telemetry_contract: final serialized fields exact", flush=True)


if __name__ == "__main__":
    main()
