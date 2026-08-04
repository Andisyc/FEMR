"""FrontRES live rollout storage and paired Gain capture boundary."""





from __future__ import annotations





from typing import Any


import torch


from rsl_rl.frontres.frontres_segment_rollout_storage import FrontRESSegmentRolloutStorage
from rsl_rl.frontres.frontres_segment_storage_records import FrontRESSegmentTransition





from rsl_rl.runners.frontres_segment_runtime_types import (


    FrontRESSegmentLiveRolloutCapture,


)


from rsl_rl.runners.frontres_segment_probe_logging import (
    should_print_once_or_verbose as _should_print_once_or_verbose,
)


from rsl_rl.runners.frontres_segment_live_reset import (
    current_frontres_frozen_transaction_metadata as _current_frozen_transaction_metadata,
    current_frontres_trial_metadata as _current_trial_metadata,
    frontres_trial_metadata_ppo_update_mask as _trial_metadata_ppo_update_mask,
    frontres_trial_metadata_priority_evidence as _trial_metadata_priority_evidence,
)





def _gain_module() -> Any | None:
    try:
        from rsl_rl.frontres import frontres_gain_legacy as frontres_gain
    except (ImportError, ModuleNotFoundError):
        return None
    return frontres_gain


def build_live_segment_storage(runner: Any, capture: FrontRESSegmentLiveRolloutCapture) -> FrontRESSegmentRolloutStorage:
    if (
        capture.transition_obs is None
        or capture.transition_privileged_obs is None
        or capture.transition_actions is None
        or capture.transition_log_probs is None
        or capture.transition_values is None
        or capture.reward_accum is None
        or capture.done_any is None
    ):
        raise RuntimeError("FrontRES Segment live storage probe did not capture a valid first-step PPO tuple.")
    if capture.transition_actions.ndim != 2 or capture.transition_actions.shape[-1] != 6:
        raise ValueError(f"live storage probe requires 6D actions, got {tuple(capture.transition_actions.shape)}")

    batch_size = int(capture.transition_actions.shape[0])
    sample = getattr(runner, "_frontres_segment_live_current_sample", None)
    current_batch = getattr(runner, "_frontres_segment_live_current_batch", None)
    sample_ids = getattr(sample, "segment_ids", None)
    sample_source = getattr(sample, "source", None)
    batch_ids = getattr(current_batch, "segment_ids", None)
    if sample_ids is not None:
        segment_ids = _expand_short_counterfactual_vector(
            sample_ids.to(device=runner.device, dtype=torch.long).reshape(-1),
            name="segment ids",
            batch_size=batch_size,
        )
    elif batch_ids is not None:
        segment_ids = _expand_short_counterfactual_vector(
            batch_ids.to(device=runner.device, dtype=torch.long).reshape(-1),
            name="segment ids",
            batch_size=batch_size,
        )
    else:
        segment_ids = torch.arange(batch_size, device=runner.device, dtype=torch.long)
    if sample_source is not None:
        segment_source = _expand_short_counterfactual_tuple(
            sample_source,
            name="segment source",
            batch_size=batch_size,
        )
    else:
        segment_source = ("live_storage_probe",) * batch_size
    reset_mask = _current_reset_success_mask(runner, batch_size=batch_size, device=runner.device)
    rollout_valid_mask = ~capture.done_any.reshape(-1).bool().to(device=runner.device)
    if capture.actor_update_mask is not None:
        actor_update_mask = capture.actor_update_mask.reshape(-1).bool().to(device=runner.device)
        if int(actor_update_mask.numel()) != batch_size:
            raise ValueError(
                f"actor_update_mask must have {batch_size} rows, got {int(actor_update_mask.numel())}"
            )
    else:
        actor_update_mask = torch.ones(batch_size, device=runner.device, dtype=torch.bool)
    trial_metadata = _current_trial_metadata(runner, batch_size=batch_size, device=runner.device)
    frozen_transaction_metadata = _current_frozen_transaction_metadata(
        runner,
        batch_size=batch_size,
        trial_metadata=trial_metadata,
    )
    if frozen_transaction_metadata is not None:
        policy = getattr(getattr(runner, "alg", None), "policy", None)
        frozen_transaction_metadata.verify_policy(policy)
    ppo_update_mask = _trial_metadata_ppo_update_mask(runner, batch_size=batch_size, device=runner.device)
    valid_mask = rollout_valid_mask & reset_mask & actor_update_mask & ppo_update_mask
    rewards = _segment_storage_rewards(capture, batch_size=batch_size, device=runner.device)
    segment_storage = FrontRESSegmentRolloutStorage(
        capacity=batch_size,
        obs_shape=capture.transition_obs.shape[1:],
        action_dim=6,
        privileged_obs_shape=capture.transition_privileged_obs.shape[1:],
        device=runner.device,
    )
    segment_storage.add_transition(
        FrontRESSegmentTransition(
            observations=capture.transition_obs,
            privileged_observations=capture.transition_privileged_obs,
            actions=capture.transition_actions,
            old_log_probs=capture.transition_log_probs,
            values=capture.transition_values,
            rewards=rewards,
            valid_mask=valid_mask,
            reset_mask=reset_mask,
            segment_ids=segment_ids,
            segment_source=segment_source,
            old_means=capture.transition_means,
            old_sigmas=capture.transition_sigmas,
            audit_transaction_id=capture.audit_transaction_id,
            audit_batch_signature=capture.audit_batch_signature,
            audit_identity_state=capture.audit_identity_state,
            priority_evidence=_trial_metadata_priority_evidence(
                runner,
                batch_size=batch_size,
                device=runner.device,
            ),
            transaction_metadata=frozen_transaction_metadata,
        )
    )
    reward_steps = _segment_storage_reward_steps(capture, batch_size=batch_size, device=runner.device)
    done_steps = _segment_storage_done_steps(capture, batch_size=batch_size, device=runner.device)
    if reward_steps is not None:
        alg = getattr(runner, "alg", None)
        segment_storage.compute_returns_and_advantages(
            reward_steps=reward_steps,
            done_steps=done_steps,
            horizon=capture.horizon_k
            if isinstance(capture.horizon_k, torch.Tensor)
            else max(1, int(getattr(alg, "frontres_segment_k", capture.rollout_k))),
            gamma=float(getattr(alg, "gamma", 1.0)),
        )
    return segment_storage


def _capture_averaged_rewards(
    capture: FrontRESSegmentLiveRolloutCapture,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    reward = capture.reward_accum.reshape(-1).detach().float()
    if device is not None:
        reward = reward.to(device=device)
    if isinstance(capture.horizon_k, torch.Tensor):
        horizon = capture.horizon_k.to(device=reward.device, dtype=torch.float32).reshape(-1)
        if int(horizon.numel()) != int(reward.numel()):
            raise ValueError(f"capture horizon must have {int(reward.numel())} rows, got {int(horizon.numel())}")
        return reward / horizon.clamp_min(1.0)
    return reward / float(max(1, int(capture.rollout_k)))


def _capture_averaged_repair_scores(
    capture: FrontRESSegmentLiveRolloutCapture,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    if capture.repair_score_accum is None:
        raise RuntimeError(
            "paired Segment Replay gain requires repair-specific executability scores; "
            "generic env reward is not a valid fallback"
        )
    score = capture.repair_score_accum.reshape(-1).detach().float()
    if device is not None:
        score = score.to(device=device)
    if isinstance(capture.horizon_k, torch.Tensor):
        horizon = capture.horizon_k.to(device=score.device, dtype=torch.float32).reshape(-1)
        if int(horizon.numel()) != int(score.numel()):
            raise ValueError(f"capture horizon must have {int(score.numel())} rows, got {int(horizon.numel())}")
        return score / horizon.clamp_min(1.0)
    return score / float(max(1, int(capture.rollout_k)))


def _capture_paired_gain(capture: FrontRESSegmentLiveRolloutCapture) -> Any | None:
    n_train = max(0, int(capture.n_train))
    n_base = max(0, int(capture.n_base))
    n_candidate = max(0, int(capture.n_candidate))
    n = min(n_train, n_base)
    gain_module = _gain_module()
    if n <= 0 or capture.gain_config is None or gain_module is None:
        return None
    if capture.done_any is None or capture.survival_steps is None:
        raise RuntimeError("paired Gain requires done_any and survival_steps")
    if capture.transition_action_steps is None:
        raise RuntimeError("paired Gain requires full-6D action steps")
    base_start = n_train + n_candidate
    clean_start = base_start + n_base
    horizon = capture.horizon_k[:n].to(dtype=torch.float32) if isinstance(capture.horizon_k, torch.Tensor) else None
    action_valid_steps = _capture_action_valid_steps(capture)
    clean_action_steps = None
    clean_action_step_mask = None
    if capture.n_clean >= n and int(capture.transition_action_steps.shape[1]) >= clean_start + n:
        clean_action_steps = capture.transition_action_steps[:, clean_start : clean_start + n]
        if action_valid_steps is not None:
            clean_action_step_mask = action_valid_steps[:, clean_start : clean_start + n]
    temporal_mask = None
    repaired_zmp = _average_physics_steps(capture.physics_zmp_repaired_steps, horizon)
    noisy_zmp = _average_physics_steps(capture.physics_zmp_noisy_steps, horizon)
    repaired_contact = _average_physics_steps(capture.physics_contact_repaired_steps, horizon)
    noisy_contact = _average_physics_steps(capture.physics_contact_noisy_steps, horizon)
    if action_valid_steps is not None and capture.motion_clean_body_pos is not None:
        # Style owns the executed trajectory prefix. A terminal fall truncates
        # later frames, but it must not erase the finite pre-fall evidence.
        temporal_mask = action_valid_steps[:, :n].transpose(0, 1)
        expected_shape = tuple(capture.motion_clean_body_pos[:n].shape[:2])
        if tuple(temporal_mask.shape) != expected_shape:
            raise ValueError(
                "paired Style validity must match captured [B,T] motion evidence, "
                f"got {tuple(temporal_mask.shape)} for {expected_shape}"
            )
    elif horizon is not None and capture.motion_clean_body_pos is not None:
        temporal_mask = torch.arange(
            capture.motion_clean_body_pos.shape[1],
            device=capture.motion_clean_body_pos.device,
        ).view(1, -1) < horizon.to(capture.motion_clean_body_pos.device).view(-1, 1)
    return gain_module.compute_segment_gain(
        clean_positions=capture.motion_clean_body_pos[:n] if capture.motion_clean_body_pos is not None else None,
        repaired_positions=capture.motion_repaired_body_pos[:n] if capture.motion_repaired_body_pos is not None else None,
        noisy_positions=capture.motion_noisy_body_pos[:n] if capture.motion_noisy_body_pos is not None else None,
        clean_root_quaternions=capture.motion_clean_root_quat[:n] if capture.motion_clean_root_quat is not None else None,
        repaired_root_quaternions=capture.motion_repaired_root_quat[:n] if capture.motion_repaired_root_quat is not None else None,
        noisy_root_quaternions=capture.motion_noisy_root_quat[:n] if capture.motion_noisy_root_quat is not None else None,
        repaired_success=(~capture.done_any[:n]).reshape(-1),
        noisy_success=(~capture.done_any[base_start : base_start + n]).reshape(-1),
        repaired_survival=capture.survival_steps[:n].reshape(-1),
        noisy_survival=capture.survival_steps[base_start : base_start + n].reshape(-1),
        effective_horizon_k=horizon,
        repaired_zmp_margin=repaired_zmp,
        noisy_zmp_margin=noisy_zmp,
        repaired_contact=repaired_contact,
        noisy_contact=noisy_contact,
        action_steps=capture.transition_action_steps[:, :n],
        config=capture.gain_config,
        audit_transaction_id=capture.audit_transaction_id,
        audit_batch_signature=capture.audit_batch_signature,
        audit_identity_state=capture.audit_identity_state,
        action_step_mask=action_valid_steps[:, :n] if action_valid_steps is not None else None,
        clean_action_steps=clean_action_steps,
        clean_action_step_mask=clean_action_step_mask,
        temporal_mask=temporal_mask,
        # PPO row eligibility still excludes terminal rows in storage. Gain
        # retains their paired pre-fall evidence for diagnostics and replay.
        valid_mask=None,
    )


def capture_frontres_paired_gain(capture: FrontRESSegmentLiveRolloutCapture) -> Any | None:
    """Return the paired Gain already derived by the live-probe owner."""

    return _capture_paired_gain(capture)


def _average_physics_steps(
    values: torch.Tensor | None,
    horizon: torch.Tensor | None,
) -> torch.Tensor | None:
    if not isinstance(values, torch.Tensor) or values.ndim != 2:
        return None
    mask = torch.isfinite(values)
    if horizon is not None and horizon.numel() == values.shape[0]:
        time = torch.arange(values.shape[1], device=values.device).view(1, -1)
        mask = mask & (time < horizon.to(values.device).view(-1, 1))
    count = mask.sum(dim=1)
    summed = torch.where(mask, values.float(), torch.zeros_like(values.float())).sum(dim=1)
    return torch.where(count > 0, summed / count.clamp_min(1), torch.full_like(summed, float("nan")))


def _capture_action_valid_steps(capture: FrontRESSegmentLiveRolloutCapture) -> torch.Tensor | None:
    """Build the executed-action mask from horizon and done-before-step state.

    Status: active, paired repair-cost boundary.
    Upstream: captured action steps, per-row horizon, and raw done trace.
    Downstream: `frontres_gain.compute_repair_cost`.
    Evidence: offline mixed-K/done contract; live population uses the same
    rollout trace but still requires S4 confirmation.
    Gap: none for the captured tensor schema; missing traces return None.
    """
    actions = capture.transition_action_steps
    if not isinstance(actions, torch.Tensor) or actions.ndim != 3:
        return None
    steps, batch_size = int(actions.shape[0]), int(actions.shape[1])
    if not isinstance(capture.horizon_k, torch.Tensor) or int(capture.horizon_k.numel()) != batch_size:
        return None
    time = torch.arange(steps, device=actions.device).view(-1, 1)
    valid = time < capture.horizon_k.to(device=actions.device, dtype=torch.long).reshape(1, -1)
    if isinstance(capture.done_steps, torch.Tensor):
        done_steps = capture.done_steps.to(device=actions.device, dtype=torch.bool)
        if tuple(done_steps.shape) != (steps, batch_size):
            raise ValueError(
                "segment done_steps must match captured action steps, "
                f"got {tuple(done_steps.shape)} for {(steps, batch_size)}"
            )
        done_before = torch.zeros_like(done_steps)
        if steps > 1:
            done_before[1:] = done_steps[:-1].cumsum(dim=0).bool()
        valid = valid & ~done_before
    return valid


def _segment_storage_rewards(
    capture: FrontRESSegmentLiveRolloutCapture,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """选择正式 policy-row reward, 不把 legacy score 当作 Gain.

    Status: active.
    Upstream: paired live capture and FRS-GAIN-v002 component owner.
    Downstream: FrontRESSegmentRolloutStorage.rewards.
    Evidence: contract-confirmed by the formal Gain connectivity test.
    Gap: real rollout population remains live-only.
    """
    reward = _capture_averaged_rewards(capture, device=device)
    n_train = max(0, int(capture.n_train))
    n_candidate = max(0, int(capture.n_candidate))
    n_base = max(0, int(capture.n_base))
    base_start = n_train + n_candidate
    if n_train > 0 and n_base >= n_train and int(reward.numel()) >= base_start + n_train and batch_size == int(reward.numel()):
        paired_gain = _capture_paired_gain(capture)
        if paired_gain is not None:
            if int(paired_gain.gain_total.numel()) != n_train or not bool(torch.isfinite(paired_gain.gain_total).all().item()):
                raise RuntimeError("paired Gain has missing/non-finite training rows; inspect component evidence before PPO")
            reward = reward.clone()
            reward[:n_train] = paired_gain.gain_total.to(device=device)
            return reward
        if capture.gain_config is not None:
            raise RuntimeError(
                "FRS-GAIN formal policy-row reward evidence is unavailable; "
                "refusing legacy repair_score fallback"
            )
        repair_score = _capture_averaged_repair_scores(capture, device=device)
        if int(repair_score.numel()) != batch_size:
            raise ValueError(f"segment repair scores must have {batch_size} rows, got {int(repair_score.numel())}")
        reward = repair_score.clone()
        reward[:n_train] = repair_score[:n_train] - repair_score[base_start : base_start + n_train]
    if int(reward.numel()) != batch_size:
        raise ValueError(f"segment rewards must have {batch_size} rows, got {int(reward.numel())}")
    return reward


def _segment_storage_reward_steps(
    capture: FrontRESSegmentLiveRolloutCapture,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    """选择进入 K-step return 的 policy-row Gain trace.

    Status: active.
    Upstream: per-step paired Gain capture.
    Downstream: storage.compute_returns_and_advantages.
    Evidence: contract-confirmed by the storage and formal-route tests.
    Gap: live finite-value diversity remains unconfirmed.
    """
    if capture.reward_steps is None:
        return None
    reward_steps = capture.reward_steps.to(device=device, dtype=torch.float32)
    if reward_steps.ndim != 2:
        raise ValueError(f"segment reward_steps must be rank-2 [T, B], got {tuple(reward_steps.shape)}")
    if int(reward_steps.shape[1]) != batch_size:
        raise ValueError(f"segment reward_steps must have {batch_size} batch entries, got {int(reward_steps.shape[1])}")

    n_train = max(0, int(capture.n_train))
    n_candidate = max(0, int(capture.n_candidate))
    n_base = max(0, int(capture.n_base))
    base_start = n_train + n_candidate
    if n_train > 0 and n_base >= n_train and batch_size >= base_start + n_train:
        if capture.gain_config is not None:
            if capture.gain_steps is None:
                raise RuntimeError("paired Gain returns require per-step Gain evidence")
            gain_steps = capture.gain_steps.to(device=device, dtype=torch.float32)
            if gain_steps.ndim != 2 or int(gain_steps.shape[1]) != batch_size:
                raise ValueError(f"segment gain_steps must have shape [T, {batch_size}], got {tuple(gain_steps.shape)}")
            if not bool(torch.isfinite(gain_steps[:, :n_train]).all().item()):
                raise RuntimeError("paired Gain step evidence contains missing/non-finite training rows")
            reward_steps = reward_steps.clone()
            reward_steps[:, :n_train] = gain_steps[:, :n_train]
            return reward_steps
        if capture.repair_score_steps is None:
            raise RuntimeError(
                "paired Segment PPO returns require repair-specific executability steps; "
                "generic env reward is not a valid fallback"
            )
        reward_steps = capture.repair_score_steps.to(device=device, dtype=torch.float32)
        if reward_steps.ndim != 2 or int(reward_steps.shape[1]) != batch_size:
            raise ValueError(
                f"segment repair_score_steps must have shape [T, {batch_size}], got {tuple(reward_steps.shape)}"
            )
        reward_steps = reward_steps.clone()
        reward_steps[:, :n_train] = reward_steps[:, :n_train] - reward_steps[:, base_start : base_start + n_train]
    return reward_steps


def _segment_storage_done_steps(
    capture: FrontRESSegmentLiveRolloutCapture,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    if capture.done_steps is None:
        return None
    done_steps = capture.done_steps.to(device=device).bool()
    if done_steps.ndim != 2:
        raise ValueError(f"segment done_steps must be rank-2 [T, B], got {tuple(done_steps.shape)}")
    if int(done_steps.shape[1]) != batch_size:
        raise ValueError(f"segment done_steps must have {batch_size} batch entries, got {int(done_steps.shape[1])}")
    return done_steps


def _select_segment_transition_actions(
    runner: Any,
    *,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if actions.ndim != 2:
        raise ValueError(f"live segment transition actions must be rank-2, got {tuple(actions.shape)}")
    if actions.shape[-1] != 6:
        raise ValueError(
            "active FrontRES transaction requires exact direct [B,6] Delta SE actions; "
            f"legacy mixed-width actions are rejected, got {tuple(actions.shape)}"
        )
    log_probs = runner.alg.transition.actions_log_prob.detach().clone().reshape(-1)
    if _should_print_once_or_verbose(runner.alg, "_frontres_segment_live_probe_trace_printed"):
        print(
            "[FrontRES Segment Live Probe Trace] "
            f"raw_action_shape={tuple(actions.shape)} "
            f"segment_action_shape={tuple(actions.shape)} "
            f"log_prob_shape={tuple(log_probs.shape)} "
            "semantic=storage_uses_native_6d_delta_se_policy",
            flush=True,
        )
    return actions, log_probs


def _select_executed_segment_actions(
    runner: Any,
    *,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Return the full-6D action actually stored after baseline overrides.

    This is intentionally separate from `_select_segment_transition_actions`:
    the latter reconstructs old log-probabilities from raw policy statistics,
    while Repair Cost must observe the executed transition tuple. Candidate,
    baseline, and Clean rows are therefore zero after the baseline override.
    """
    transition_actions = getattr(getattr(runner, "alg", None), "transition", None)
    transition_actions = getattr(transition_actions, "actions", None)
    if isinstance(transition_actions, torch.Tensor) and transition_actions.shape == actions.shape:
        selected = transition_actions
    else:
        selected, _ = _select_segment_transition_actions(runner, actions=actions)
    if selected.ndim != 2 or selected.shape[-1] < 6:
        raise ValueError(f"executed Segment action must expose full 6D Delta SE, got {tuple(selected.shape)}")
    if selected.ndim != 2 or int(selected.shape[-1]) != 6:
        raise ValueError(
            "executed FrontRES action must be exact direct [B,6] Delta SE, "
            f"got {tuple(selected.shape)}"
        )
    return selected.detach().clone()


def _motion_perturber_from_runner(runner: Any) -> Any | None:
    env_raw = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    command_manager = getattr(env_raw, "command_manager", None)
    terms = getattr(command_manager, "_terms", {}) if command_manager is not None else {}
    motion_command = terms.get("motion") if hasattr(terms, "get") else None
    if motion_command is None:
        motion_command = getattr(env_raw, "command", None)
    return getattr(motion_command, "perturber", None)


def _snapshot_frontres_perturbation_rp(runner: Any, *, num_envs: int) -> torch.Tensor | None:
    perturber = _motion_perturber_from_runner(runner)
    roll_state = getattr(perturber, "_roll_state", None)
    pitch_state = getattr(perturber, "_pitch_state", None)
    if not isinstance(roll_state, torch.Tensor) or not isinstance(pitch_state, torch.Tensor):
        return None
    count = max(0, min(int(num_envs), int(roll_state.numel()), int(pitch_state.numel())))
    if count <= 0:
        return None
    rp = torch.stack(
        (
            roll_state[:count].detach().float(),
            pitch_state[:count].detach().float(),
        ),
        dim=-1,
    )
    iid_event_rp = getattr(perturber, "_iid_event_rp", None)
    if isinstance(iid_event_rp, torch.Tensor) and iid_event_rp.ndim == 2 and int(iid_event_rp.shape[0]) >= count:
        rp = rp + iid_event_rp[:count, :2].detach().float()
    family_masks = getattr(perturber, "_family_masks", None)
    if isinstance(family_masks, dict) and isinstance(family_masks.get("local_rp"), torch.Tensor):
        mask = family_masks["local_rp"][:count].to(device=rp.device, dtype=torch.bool)
        rp = rp * mask.to(dtype=rp.dtype).view(-1, 1)
    baseline_mask = getattr(perturber, "_baseline_mask", None)
    if isinstance(baseline_mask, torch.Tensor) and int(baseline_mask.numel()) >= count:
        mask = ~baseline_mask[:count].to(device=rp.device, dtype=torch.bool)
        rp = rp * mask.to(dtype=rp.dtype).view(-1, 1)
    return rp.detach().clone()


def _expand_short_counterfactual_vector(
    tensor: torch.Tensor,
    *,
    name: str,
    batch_size: int,
) -> torch.Tensor:
    rows = int(tensor.numel())
    if rows == int(batch_size):
        return tensor
    if rows > 0 and int(batch_size) % rows == 0:
        return tensor.repeat(int(batch_size) // rows)
    raise ValueError(f"{name} must have {batch_size} rows, got {rows}")


def _expand_short_counterfactual_tuple(value: Any, *, name: str, batch_size: int) -> tuple[str, ...]:
    items = tuple(str(item) for item in value)
    rows = len(items)
    if rows == int(batch_size):
        return items
    if rows > 0 and int(batch_size) % rows == 0:
        return items * (int(batch_size) // rows)
    raise ValueError(f"{name} must have {batch_size} rows, got {rows}")


def _current_reset_success_mask(runner: Any, *, batch_size: int, device: torch.device | str) -> torch.Tensor:
    result = getattr(runner, "_frontres_segment_live_current_reset_result", None)
    if result is None:
        return torch.ones(batch_size, device=device, dtype=torch.bool)
    success_mask = getattr(result, "success_mask", None)
    if success_mask is None:
        return torch.ones(batch_size, device=device, dtype=torch.bool)
    success_mask = success_mask.to(device=device).bool().reshape(-1)
    success_mask = _expand_short_counterfactual_vector(
        success_mask,
        name="segment reset success mask",
        batch_size=batch_size,
    )
    return success_mask.detach()


# Public storage and paired-Gain surface.
frontres_gain_module = _gain_module
select_frontres_segment_transition_actions = _select_segment_transition_actions
select_frontres_executed_segment_actions = _select_executed_segment_actions
snapshot_frontres_perturbation_rp = _snapshot_frontres_perturbation_rp
capture_frontres_averaged_rewards = _capture_averaged_rewards
capture_frontres_averaged_repair_scores = _capture_averaged_repair_scores
