#!/usr/bin/env python3
"""CPU-only Step 4B-S2 contract for the v015 formal transaction connector."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[4]
RSL_ROOT = ROOT / "source" / "rsl_rl" / "rsl_rl"
CANDIDATE_TEST = RSL_ROOT / "tests" / "frontres_v015_grouped_candidate_adapter_contract.py"
LIVE_SAMPLER_PATH = RSL_ROOT / "runners" / "frontres_segment_live_sampler.py"
LIVE_UPDATE_LOOP_PATH = RSL_ROOT / "runners" / "frontres_segment_live_update_loop.py"
ON_POLICY_RUNNER_PATH = RSL_ROOT / "runners" / "on_policy_runner.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _expect_runtime_error(fn) -> None:
    try:
        fn()
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


class _TrackingSGD(torch.optim.SGD):
    """Test-only explicit optimizer event counter required by fake S2."""

    def __init__(self, params) -> None:
        super().__init__(params, lr=0.05)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure=closure)


def _load_owners():
    candidate_contract = _load("frontres_v015_transaction_candidate_helper", CANDIDATE_TEST)
    owners = candidate_contract._load_owners()
    live_probe = owners[6]
    ppo = owners[8]
    live_probe.FrontRESSegmentPPOConfig = ppo.FrontRESSegmentPPOConfig
    live_probe.compute_frontres_segment_ppo_loss = ppo.compute_frontres_segment_ppo_loss
    live_sampler = sys.modules.get("rsl_rl.runners.frontres_segment_live_sampler")
    if live_sampler is None:
        live_sampler = _load("rsl_rl.runners.frontres_segment_live_sampler", LIVE_SAMPLER_PATH)
    live_update_loop = _load("frontres_v015_transaction_live_update_loop", LIVE_UPDATE_LOOP_PATH)
    return candidate_contract, owners, live_sampler, live_update_loop


def _formal_alg(policy: torch.nn.Module, optimizer: _TrackingSGD) -> SimpleNamespace:
    return SimpleNamespace(
        policy=policy,
        optimizer=optimizer,
        clip_param=0.2,
        value_loss_coef=0.0,
        entropy_coef=0.0,
        use_clipped_value_loss=True,
        max_grad_norm=1.0,
        lambda_supervised=0.0,
        lambda_supervised_min=0.0,
        frontres_v015_formal_transaction_enabled=True,
        frontres_training_objective="segment_replay_hrl",
        frontres_segment_replay_enabled=True,
        frontres_segment_advantage_normalization="grouped_scale_only",
        frontres_hsl_init_enabled=False,
        frontres_hsl_rollout_label_enabled=False,
        frontres_segment_critic_warmup_iterations=0,
        frontres_segment_actor_warmup_iterations=0,
        frontres_future_offsets=(1, 2),
        frontres_future_intent_layout_version="frontres-v015-future-intent-q29-v1",
        frontres_segment_live_train_enabled=False,
        frontres_segment_live_update_loop_only=False,
        frontres_segment_live_single_update_only=False,
    )


def _build_request(candidate_contract, owners, live_sampler):
    gain_contract, one_action, helper, commands, hooks, setup, live_probe, storage, ppo = owners
    captured, _kwargs, _batch = candidate_contract._capture_and_build(
        gain_contract,
        one_action,
        helper,
        commands,
        hooks,
        setup,
        live_probe,
        ppo,
    )
    policy = candidate_contract._ZeroRatioPolicy()
    optimizer = _TrackingSGD(policy.parameters())
    runner = SimpleNamespace(alg=_formal_alg(policy, optimizer))
    snapshot = live_sampler.capture_frontres_frozen_policy_snapshot(runner, transaction_id="tx-v015-formal-s2")
    common = {
        "transaction_id": snapshot.transaction_id,
        "policy_snapshot_id": snapshot.policy_snapshot_id,
        "motion_ids": ("motion-a", "motion-b"),
        "start_frames": torch.tensor([12, 24], dtype=torch.long),
        "segment_ids": torch.tensor([101, 202], dtype=torch.long),
        "source_index": torch.tensor([0, 1], dtype=torch.long),
    }
    attempt_zero = live_probe.build_frontres_v015_grouped_candidate_batch(
        captured.result,
        **common,
        trial_index=torch.tensor([0, 0], dtype=torch.long),
    )
    attempt_one = live_probe.build_frontres_v015_grouped_candidate_batch(
        captured.result,
        **common,
        trial_index=torch.tensor([1, 1], dtype=torch.long),
    )
    metadata = attempt_zero.transaction_metadata
    plan = live_sampler.FrontRESV015FormalTransactionPlan(
        snapshot=snapshot,
        motion_ids=metadata.motion_ids + metadata.motion_ids,
        start_frames=torch.cat((metadata.start_frames, metadata.start_frames), dim=0),
        segment_ids=torch.cat((metadata.segment_ids, metadata.segment_ids), dim=0),
        source_index=torch.tensor([0, 1, 0, 1], dtype=torch.long),
        trial_index=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        horizon_k=torch.cat((metadata.horizon_k, metadata.horizon_k), dim=0),
        scenario_ids=metadata.scenario_ids + metadata.scenario_ids,
        noisy_segment_hashes=metadata.noisy_segment_hashes + metadata.noisy_segment_hashes,
        x_t_identities=metadata.x_t_identities + metadata.x_t_identities,
        intent_q29_provenance=metadata.intent_q29_provenance,
        intent_q29_source=metadata.intent_q29_source,
    )
    request = live_probe.FrontRESV015FormalTransactionRequest(
        plan=plan,
        candidate_batches=(attempt_zero, attempt_one),
        policy_evaluator=policy,
    )
    return SimpleNamespace(
        runner=runner,
        optimizer=optimizer,
        policy=policy,
        request=request,
        storage=storage,
    )


def test_t_connect_order_exact_one_and_diagnostics(candidate_contract, owners, live_sampler, live_update_loop) -> None:
    fixture = _build_request(candidate_contract, owners, live_sampler)
    provider_steps: list[int] = []

    def provider():
        provider_steps.append(fixture.optimizer.step_count)
        assert fixture.runner._frontres_v015_checkpoint_transaction_state == {
            "state": "collecting",
            "phase": "provider",
        }
        return fixture.request

    fixture.runner._frontres_v015_formal_transaction_provider = provider
    original_legacy = fixture.storage.FrontRESSegmentStorageBatch.to_ppo_batch

    def legacy_forbidden(*_args, **_kwargs):
        raise AssertionError("v015 formal transaction must not call legacy to_ppo_batch")

    fixture.storage.FrontRESSegmentStorageBatch.to_ppo_batch = legacy_forbidden
    try:
        result = live_update_loop.run_frontres_v015_formal_transaction_update_loop(fixture.runner)
    finally:
        fixture.storage.FrontRESSegmentStorageBatch.to_ppo_batch = original_legacy

    assert provider_steps == [0]
    assert fixture.optimizer.step_count == 1
    assert result.optimizer_step_before == 0
    assert result.optimizer_step_after == 1
    assert result.optimizer_step_delta == 1
    assert result.update_invocation_count == 1
    assert result.segment_count == 2
    assert result.source_count == 2
    assert result.policy_attempt_count == 4
    assert result.ppo_result.grouped_reduction_active
    assert result.ppo_result.grouped_motion_count == 2
    assert result.ppo_result.grouped_segment_count == 2
    assert result.ppo_result.grouped_attempt_count == 4
    torch.testing.assert_close(
        torch.tensor(result.ppo_result.grouped_attempt_mass_shares),
        torch.full((4,), 0.25),
    )
    assert result.diagnostics["intent_q29_provenance"] == "deployment_noisy_q29"
    assert result.diagnostics["optimizer_step_delta"] == 1
    checkpoint_state = fixture.runner._frontres_v015_checkpoint_transaction_state
    assert checkpoint_state["state"] == "committed"
    assert checkpoint_state["receipt"]["optimizer_step_delta"] == 1
    assert checkpoint_state["receipt"]["collected_policy_attempt_count"] == 4
    assert "clean_continuation" not in repr(checkpoint_state["receipt"])
    runner_source = ON_POLICY_RUNNER_PATH.read_text(encoding="utf-8")
    assert "def run_frontres_v015_formal_transaction(self)" in runner_source
    assert "run_frontres_v015_formal_transaction_update_loop_helper(self)" in runner_source
    print(
        "[T-connect/T-order/T-exact-one-update/T-no-legacy-route/T-diagnostic/T-checkpoint-barrier] "
        "provider barrier seals a 2x2 transaction, then one update yields a metadata-only receipt",
        flush=True,
    )


def test_t_partial_hsl_and_legacy_config_fail_before_step(candidate_contract, owners, live_sampler, live_update_loop) -> None:
    fixture = _build_request(candidate_contract, owners, live_sampler)
    live_probe = owners[6]
    partial = replace(fixture.request, candidate_batches=(fixture.request.candidate_batches[0],))
    _expect_runtime_error(lambda: live_probe.run_frontres_v015_formal_transaction_update(fixture.runner, partial))
    assert fixture.optimizer.step_count == 0

    fixture = _build_request(candidate_contract, owners, live_sampler)
    fixture.runner.alg.frontres_hsl_init_enabled = True
    _expect_runtime_error(
        lambda: live_probe.run_frontres_v015_formal_transaction_update(fixture.runner, fixture.request)
    )
    assert fixture.optimizer.step_count == 0

    fixture = _build_request(candidate_contract, owners, live_sampler)
    fixture.runner.alg.frontres_segment_advantage_normalization = "scale_only"
    _expect_runtime_error(
        lambda: live_probe.run_frontres_v015_formal_transaction_update(fixture.runner, fixture.request)
    )
    assert fixture.optimizer.step_count == 0
    print(
        "[T-partial/T-warmup-isolation/T-fail-closed] partial transaction, HSL, and legacy normalization reject before step",
        flush=True,
    )


def test_t_q29_actor_route_before_normalizer(_candidate_contract, owners, _live_sampler, _live_update_loop) -> None:
    live_probe = owners[6]
    trace: list[str] = []
    obs = torch.zeros(2, 5)
    env = SimpleNamespace(get_observations=lambda: (obs, {"observations": {}}))

    def fixed(value: torch.Tensor) -> torch.Tensor:
        trace.append("fixed")
        return value + 1.0

    def q29(value: torch.Tensor) -> torch.Tensor:
        trace.append("q29")
        return value + 2.0

    def normalize(value: torch.Tensor) -> torch.Tensor:
        trace.append("normalizer")
        return value

    runner = SimpleNamespace(
        env=env,
        device=torch.device("cpu"),
        alg=SimpleNamespace(frontres_future_offsets=(1, 2)),
        policy_obs_type=None,
        privileged_obs_type=None,
        teacher_obs_type=None,
        ref_vel_estimator_obs_type=None,
        _append_frontres_fixed_noisy_future_context=fixed,
        _append_frontres_future_intent_context=q29,
        _apply_obs_normalizer=normalize,
        privileged_obs_normalizer=lambda value: value,
        teacher_obs_normalizer=lambda value: value,
    )
    observations = live_probe._read_live_observations(runner)
    assert trace == ["q29", "normalizer"]
    torch.testing.assert_close(observations.obs, torch.full((2, 5), 2.0))
    print(
        "[T-q29-route] v015 actor observation rejects legacy fixed tail and appends q29 before normalizer",
        flush=True,
    )


def main() -> None:
    candidate_contract, owners, live_sampler, live_update_loop = _load_owners()
    test_t_connect_order_exact_one_and_diagnostics(candidate_contract, owners, live_sampler, live_update_loop)
    test_t_partial_hsl_and_legacy_config_fail_before_step(candidate_contract, owners, live_sampler, live_update_loop)
    test_t_q29_actor_route_before_normalizer(candidate_contract, owners, live_sampler, live_update_loop)
    print("frontres_v015_transaction_route_contract: ok", flush=True)


if __name__ == "__main__":
    main()
