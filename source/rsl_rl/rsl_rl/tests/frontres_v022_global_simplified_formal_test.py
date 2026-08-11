"""One bounded CPU formal test for the complete TRAIN-v022 semantic chain.

The deterministic tensors below replace only simulator rollout evidence.  The
test keeps the production K/DR/LR owner, symmetric-log utility, exact B8/M4
reduction, named Adam groups, Replay Curriculum owner, and persistence schema.
It intentionally never constructs K16/K32/K64 work.
"""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch

from rsl_rl.frontres.frontres_outer_scenario_replay import (
    FRONTRES_OUTER_REPLAY_SCHEMA,
    FrontRESOuterScenarioReplay,
    FrontRESScenarioKey,
)
from rsl_rl.frontres.frontres_return_utility import frontres_symmetric_log_utility
from rsl_rl.frontres.frontres_segment_storage_records import FrontRESV015GroupedCandidateMetadata
from rsl_rl.frontres.frontres_segment_warmup import resolve_frontres_k_stage_identity
from rsl_rl.frontres.frontres_value_normalization import (
    FRONTRES_VALUE_NORMALIZATION_ID,
    FrontRESValueNormalizerState,
)
from rsl_rl.algorithms.frontres_segment_ppo import (
    FrontRESSegmentPPOBatch,
    FrontRESSegmentPPOConfig,
    compute_frontres_segment_ppo_loss,
    install_frontres_v006_scalar_gradients,
    step_frontres_v005_scalar_optimizer,
)
from rsl_rl.algorithms.frontres_unified import FrontRESTrackedAdam
from rsl_rl.modules.front_residual_actor_critic import FrontRESActorCritic
from rsl_rl.frontres.frontres_policy_evaluation import FrontRESSegmentLivePolicyAdapter


SCHEDULE = (
    (8, 4, 4, 4, 24, "lower-k8", 0.5, "linear-coupled-v1", 8, 2.381),
)
METHOD_ID = "FRS-METHOD-v023"
TRAIN_ID = "FRS-TRAIN-v022"


def _production_policy() -> FrontRESActorCritic:
    checkpoint = str(os.environ.get("FRONTRES_GMT_CHECKPOINT", "") or "").strip()
    if not checkpoint or not Path(checkpoint).is_file():
        raise RuntimeError("set FRONTRES_GMT_CHECKPOINT to the frozen production GMT checkpoint")
    return FrontRESActorCritic(
        num_actor_obs=928,
        num_critic_obs=449,
        num_actions=29,
        residual_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[1024, 1024, 512, 256],
        gmt_checkpoint_path=checkpoint,
        q_ref_start_idx=232,
        num_task_corrections=6,
        num_frontres_obs=158,
        init_critic_from_gmt=False,
    )


def _key_for_selection(owner: FrontRESOuterScenarioReplay, selection, transaction: int) -> FrontRESScenarioKey:
    if selection.replay_key_digest is not None:
        matches = tuple(record.key for record in owner.records if record.key.digest == selection.replay_key_digest)
        assert len(matches) == 1
        return matches[0]
    token = f"tx{transaction}-segment{selection.segment_id}-seed{selection.perturbation_seed}"
    return FrontRESScenarioKey(
        motion_id=f"motion-{selection.segment_id}",
        start_frame=selection.segment_id * 8,
        segment_id=selection.segment_id,
        x_t_identity=f"x-{token}",
        perturbation_family=selection.perturbation_family,
        perturbation_strength=selection.perturbation_strength,
        perturbation_seed=selection.perturbation_seed,
        noisy_segment_hash=f"noisy-{token}",
        horizon_k=8,
        future_intent_identity=f"future-{token}",
        planned_support_identity=f"support-{token}",
    )


def _actor_parameters(model: FrontRESActorCritic) -> tuple[torch.nn.Parameter, ...]:
    parameters = list(model.residual_actor.parameters())
    for name in ("std", "log_std"):
        value = getattr(model, name, None)
        if isinstance(value, torch.nn.Parameter) and value.requires_grad:
            parameters.append(value)
    return tuple(parameters)


def _optimizer(model: FrontRESActorCritic) -> FrontRESTrackedAdam:
    optimizer = FrontRESTrackedAdam(
        (
            {"params": _actor_parameters(model), "lr": 3.0e-7, "frontres_role": "actor"},
            {"params": tuple(model.critic.parameters()), "lr": 1.0e-5, "frontres_role": "critic"},
        )
    )
    return optimizer


def _named_groups(optimizer: FrontRESTrackedAdam) -> dict[str, dict]:
    groups = {str(group.get("frontres_role", "")): group for group in optimizer.param_groups}
    assert set(groups) == {"actor", "critic"}
    return groups


def _one_transaction(
    transaction: int,
    *,
    model: FrontRESActorCritic,
    optimizer: FrontRESTrackedAdam,
    replay: FrontRESOuterScenarioReplay,
    value_normalizer: FrontRESValueNormalizerState,
) -> tuple[dict, FrontRESValueNormalizerState]:
    curriculum = resolve_frontres_k_stage_identity(
        schedule=SCHEDULE,
        committed_update_iteration=transaction,
        max_horizon_k=8,
    )
    assert curriculum.active_k == 8 and curriculum.active_m == 4
    phase = curriculum.phase
    groups = _named_groups(optimizer)
    groups["actor"]["lr"] = phase.actor_learning_rate
    assert 3.0e-7 <= groups["actor"]["lr"] <= 1.0e-6
    assert groups["critic"]["lr"] == 1.0e-5

    plan = replay.plan(
        transaction_id=f"tx-{transaction:03d}",
        curriculum=curriculum,
        num_segments=512,
        eligible=lambda _segment_id: True,
        global_family=lambda _segment_id: "local_rp",
    )
    assert len(plan.selections) == 8
    assert transaction == 0 or any(selection.replay_key_digest is not None for selection in plan.selections)
    keys = tuple(_key_for_selection(replay, selection, transaction) for selection in plan.selections)

    segment_value = torch.tensor([selection.segment_id for selection in plan.selections], dtype=torch.float32)
    strength_value = torch.tensor(
        [selection.perturbation_strength for selection in plan.selections],
        dtype=torch.float32,
    )
    seed_phase = torch.tensor(
        [selection.perturbation_seed % 100_000 for selection in plan.selections],
        dtype=torch.float32,
    ) / 100_000.0
    compact = torch.cat(
        (
            (segment_value / 511.0).unsqueeze(1),
            (strength_value / 2.381).unsqueeze(1),
            torch.sin(0.017 * segment_value).unsqueeze(1),
            torch.cos(2.0 * math.pi * seed_phase).unsqueeze(1),
            torch.full((8, 1), float(curriculum.active_k) / 8.0),
        ),
        dim=1,
    )
    actor_features = compact.repeat(1, 32)[:, :158]
    critic_features = compact.repeat(1, 90)[:, :449]
    gmt_features = compact.repeat(1, 154)[:, :770]
    observations = torch.cat((actor_features, gmt_features), dim=1).repeat_interleave(4, dim=0)
    privileged = critic_features.repeat_interleave(4, dim=0)
    with torch.no_grad():
        sealed_actions = model.act(observations).detach().clone()
        old_means = model.action_mean.detach().clone()
        old_sigmas = model.action_std.detach().clone()
        old_log_probs = model.get_actions_log_prob(sealed_actions).detach().clone()
        old_values_by_scenario = model.evaluate(critic_features).reshape(-1).detach().clone()
    old_values = old_values_by_scenario.repeat_interleave(4)
    source_index = torch.arange(8, dtype=torch.long).repeat_interleave(4)
    trial_index = torch.arange(4, dtype=torch.long).repeat(8)
    scenario_signal = (
        torch.sin(0.17 * segment_value)
        + 0.5 * torch.cos(2.0 * math.pi * seed_phase)
        - 0.2 * strength_value
    ).repeat_interleave(4)
    trial_value = trial_index.to(dtype=torch.float32)
    raw_gain = (
        0.35 * scenario_signal
        - 0.08 * trial_value
        - 0.04 * sealed_actions.square().sum(dim=1)
        + 0.12 * torch.sin(trial_value + transaction)
    )
    utility = frontres_symmetric_log_utility(raw_gain)
    advantage = utility - old_values
    scenario_segment_ids = torch.tensor([key.segment_id for key in keys], dtype=torch.long)
    segment_ids = scenario_segment_ids.repeat_interleave(4)
    metadata = FrontRESV015GroupedCandidateMetadata(
        transaction_id=plan.transaction_id,
        policy_snapshot_id=f"policy-{transaction:03d}",
        motion_ids=tuple(key.motion_id for key in keys for _ in range(4)),
        start_frames=torch.tensor([key.start_frame for key in keys], dtype=torch.long).repeat_interleave(4),
        segment_ids=segment_ids,
        source_index=source_index,
        trial_index=trial_index,
        horizon_k=torch.full((32,), 8, dtype=torch.long),
        evidence_valid_step_count=torch.full((32,), 8, dtype=torch.long),
        trial_role=("policy",) * 32,
        noisy_segment_hashes=tuple(key.noisy_segment_hash for key in keys for _ in range(4)),
        scenario_ids=tuple(f"scenario-{key.digest}" for key in keys for _ in range(4)),
        x_t_identities=tuple(key.x_t_identity for key in keys for _ in range(4)),
        intent_q29_provenance="deployment_noisy_q29",
        intent_q29_source="deterministic-simulator-adapter",
    )
    metadata.validate()
    batch = FrontRESSegmentPPOBatch(
        observations=observations,
        privileged_observations=privileged,
        actions=sealed_actions,
        old_log_probs=old_log_probs,
        old_values=old_values,
        returns=raw_gain,
        advantages=advantage,
        valid_mask=torch.ones(32, dtype=torch.bool),
        segment_ids=segment_ids,
        old_means=old_means,
        old_sigmas=old_sigmas,
        transaction_metadata=metadata,
        transaction_row_indices=torch.arange(32, dtype=torch.long),
    )
    cfg = FrontRESSegmentPPOConfig(
        actor_loss_weight=1.0,
        advantage_normalization="grouped_scale_only",
        critic_target_id="segment-exact-m-mean-symlog-v1",
        critic_value_normalization=FRONTRES_VALUE_NORMALIZATION_ID,
        critic_value_normalizer_state=value_normalizer,
    )
    evaluator = FrontRESSegmentLivePolicyAdapter(
        SimpleNamespace(policy=model, use_estimate_ref_vel=False),
        privileged,
    )
    result = compute_frontres_segment_ppo_loss(evaluator, batch, cfg)
    assert torch.isfinite(result.total_loss)
    assert result.grouped_segment_count == 8 and result.grouped_attempt_count == 32

    before_actor = tuple(parameter.detach().clone() for parameter in _actor_parameters(model))
    before_critic = tuple(parameter.detach().clone() for parameter in model.critic.parameters())
    optimizer.zero_grad(set_to_none=True)
    optimizer_parameters = tuple(
        parameter for group in optimizer.param_groups for parameter in group["params"]
    )
    snapshots = {id(parameter): parameter.detach().clone() for parameter in optimizer_parameters}
    gradients = install_frontres_v006_scalar_gradients(
        model,
        result,
        cfg,
        optimizer_parameters,
        max_grad_norm=0.5,
    )
    assert gradients.actor_nonzero_parameter_count > 0 and gradients.critic_nonzero_parameter_count > 0
    committed = step_frontres_v005_scalar_optimizer(
        optimizer,
        gradients.actor_parameters,
        snapshots,
        actor_loss_weight=1.0,
    )
    assert committed.committed_actor_delta_l2 > 0.0
    assert any(not torch.equal(old, new) for old, new in zip(before_actor, _actor_parameters(model), strict=True))
    assert any(not torch.equal(old, new) for old, new in zip(before_critic, model.critic.parameters(), strict=True))

    candidate = replay.stage(
        plan,
        keys=keys,
        actor_advantages=advantage,
        source_index=source_index,
        policy_snapshot_id=f"policy-{transaction:03d}",
        active_m=4,
    )
    telemetry = replay.commit(
        candidate,
        receipt={
            "method_contract_id": METHOD_ID,
            "training_contract_id": TRAIN_ID,
            "transaction_id": plan.transaction_id,
            "policy_snapshot_id": f"policy-{transaction:03d}",
            "optimizer_step_delta": 1,
        },
    )
    assert len(telemetry["slot_purposes"]) == 8
    assert int(telemetry["active_count"]) <= int(telemetry["active_capacity_after"])
    next_normalizer = result.critic_value_normalizer_candidate_state
    assert isinstance(next_normalizer, FrontRESValueNormalizerState)
    return telemetry, next_normalizer


def _checkpoint_roundtrip(
    path: Path,
    *,
    iteration: int,
    model: FrontRESActorCritic,
    optimizer: FrontRESTrackedAdam,
    replay: FrontRESOuterScenarioReplay,
    value_normalizer: FrontRESValueNormalizerState,
) -> tuple[FrontRESActorCritic, FrontRESTrackedAdam, FrontRESOuterScenarioReplay, FrontRESValueNormalizerState]:
    payload = {
        "format": "frontres-v022-global-simplified-test-v1",
        "iteration": iteration,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "outer_replay": replay.state_dict(),
        "value_normalizer": value_normalizer.state_dict(),
    }
    torch.save(payload, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    assert restored["format"] == payload["format"] and restored["iteration"] == iteration
    new_model = _production_policy()
    new_model.load_state_dict(restored["model"], strict=True)
    new_optimizer = _optimizer(new_model)
    new_optimizer.load_state_dict(restored["optimizer"])
    new_replay = FrontRESOuterScenarioReplay(
        capacity_ladder=(64, 128, 256),
        minimum_visits_before_expand=4,
        seed=20260811,
    )
    new_replay.load_state_dict(restored["outer_replay"])
    assert new_replay.state_dict()["schema"] == FRONTRES_OUTER_REPLAY_SCHEMA
    new_value_normalizer = FrontRESValueNormalizerState.from_state_dict(restored["value_normalizer"])
    return new_model, new_optimizer, new_replay, new_value_normalizer


def main() -> None:
    torch.manual_seed(20260811)
    model = _production_policy()
    optimizer = _optimizer(model)
    replay = FrontRESOuterScenarioReplay(
        capacity_ladder=(64, 128, 256),
        minimum_visits_before_expand=4,
        seed=20260811,
    )
    value_normalizer = FrontRESValueNormalizerState()
    capacity_transitions: list[tuple[int, int]] = []
    with tempfile.TemporaryDirectory(prefix="frontres-v022-global-") as directory:
        checkpoint = Path(directory) / "tx8.pt"
        for transaction in range(32):
            telemetry, value_normalizer = _one_transaction(
                transaction,
                model=model,
                optimizer=optimizer,
                replay=replay,
                value_normalizer=value_normalizer,
            )
            capacity_transitions.append(
                (int(telemetry["active_capacity_before"]), int(telemetry["active_capacity_after"]))
            )
            if transaction == 7:
                model, optimizer, replay, value_normalizer = _checkpoint_roundtrip(
                    checkpoint,
                    iteration=8,
                    model=model,
                    optimizer=optimizer,
                    replay=replay,
                    value_normalizer=value_normalizer,
                )

    groups = _named_groups(optimizer)
    assert {int(group["frontres_step_count"]) for group in groups.values()} == {32}
    assert all(before <= after and after in {64, 128, 256} for before, after in capacity_transitions)
    state_before = copy.deepcopy(replay.state_dict())
    curriculum = resolve_frontres_k_stage_identity(schedule=SCHEDULE, committed_update_iteration=32, max_horizon_k=8)
    failed_plan = replay.plan(
        transaction_id="tx-invalid",
        curriculum=curriculum,
        num_segments=512,
        eligible=lambda _segment_id: True,
        global_family=lambda _segment_id: "local_rp",
    )
    keys = tuple(_key_for_selection(replay, selection, 32) for selection in failed_plan.selections)
    candidate = replay.stage(
        failed_plan,
        keys=keys,
        actor_advantages=torch.zeros(32),
        source_index=torch.arange(8).repeat_interleave(4),
        policy_snapshot_id="policy-invalid",
        active_m=4,
    )
    try:
        replay.commit(candidate, receipt={"optimizer_step_delta": 0})
    except ValueError:
        pass
    else:
        raise AssertionError("invalid transaction unexpectedly committed")
    after = replay.state_dict()
    assert state_before.keys() == after.keys()
    assert state_before["records"] == after["records"]
    assert state_before["active_by_k"] == after["active_by_k"]
    assert state_before["capacity_by_k"] == after["capacity_by_k"]
    assert torch.equal(state_before["generator_state"], after["generator_state"])
    assert math.isclose(float(groups["actor"]["lr"]), 1.0e-6)
    print("frontres_v022_global_simplified_formal_test: PASS transactions=32 K=8 B=8 M=4")


if __name__ == "__main__":
    main()
