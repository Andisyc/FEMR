"""Imperative frozen-GMT K-step rollout shell."""





from __future__ import annotations





from collections import deque


from typing import Any


import torch


from rsl_rl.frontres.training_schedule import resolve_frontres_mode_state


from rsl_rl.modules import FrontRESActorCritic


from rsl_rl.runners.frontres_training_setup import configure_frontres_pair_layout


from rsl_rl.runners.frontres_rollout_step import frontres_motion_command, prepare_frontres_rollout_step


from rsl_rl.runners.frontres_formal_runtime_audit import print_reset_lifecycle_audit, snapshot_reset_pair_state, snapshot_termination_terms





from rsl_rl.runners.frontres_segment_runtime_types import (


    FrontRESSegmentLiveObservations,


    FrontRESSegmentLiveRolloutCapture,


)


from rsl_rl.runners.frontres_segment_probe_logging import (
    new_live_audit_identity as _new_live_audit_identity,
    print_frontres_dr_runtime_probe as _print_frontres_dr_runtime_probe,
)


from rsl_rl.runners.frontres_segment_live_reset import (
    current_frontres_trial_metadata as _current_trial_metadata,
)


from rsl_rl.runners.frontres_segment_live_storage import (
    frontres_gain_module as _gain_module,
    select_frontres_executed_segment_actions as _select_executed_segment_actions,
    select_frontres_segment_transition_actions as _select_segment_transition_actions,
    snapshot_frontres_perturbation_rp as _snapshot_frontres_perturbation_rp,
)


from rsl_rl.runners.frontres_segment_one_action_k import (
    append_fixed_noisy_actor_context as _append_fixed_noisy_actor_context,
)


from rsl_rl.runners.frontres_segment_physics import (
    capture_frontres_motion_quality_frame as _capture_motion_quality_frame,
    capture_frontres_physics_frame as _capture_physics_frame,
    capture_frontres_root_orientation_frame as _capture_root_orientation_frame,
    stack_frontres_motion_quality_frames as _stack_motion_quality_frames,
)





def _segment_repair_executability_scores(
    runner: Any,
    pair_layout: Any,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Return family-matched repair scores without generic env/task reward."""
    scorer = getattr(runner, "_frontres_executability", None)
    if scorer is None:
        raise RuntimeError("Segment Replay gain requires runner._frontres_executability")
    env = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    command_manager = getattr(env, "command_manager", None)
    command = getattr(command_manager, "_terms", {}).get("motion") if command_manager is not None else None
    if command is None:
        raise RuntimeError("Segment Replay gain requires the motion command executability source")

    _, components = scorer.exec_score(command, return_components=True)
    role_counts = (
        int(pair_layout.n_train),
        int(pair_layout.n_candidate),
        int(pair_layout.n_base),
        int(pair_layout.n_clean),
    )
    if sum(role_counts) != int(batch_size):
        raise ValueError(
            "Segment Replay executability requires an exact quartet row layout; "
            f"counts={role_counts} batch_size={batch_size}"
        )

    cfg = getattr(runner, "cfg", {}) or {}
    specialist = str(cfg.get("frontres_specialist_mode", "") if hasattr(cfg, "get") else "").lower()
    active_modes = tuple(getattr(runner, "_frontres_curriculum_active_modes", ()))
    if specialist in ("rp", "local_rp", "rp_only", "strong_rp"):
        fallback_modes = ("local_rp",)
    elif active_modes:
        fallback_modes = active_modes
    else:
        raise RuntimeError("Segment Replay gain requires an explicit perturbation family")

    max_count = max(role_counts, default=0)
    mode_groups = list(getattr(runner, "_frontres_curriculum_env_mode_groups", ()))[:max_count]
    if len(mode_groups) < max_count:
        mode_groups.extend([fallback_modes] * (max_count - len(mode_groups)))

    scores = torch.empty(batch_size, device=runner.device, dtype=components["rp"].dtype)
    start = 0
    for count in role_counts:
        if count > 0:
            scores[start : start + count] = scorer.exec_score_for_modes(
                components,
                start,
                count,
                mode_groups=mode_groups[:count],
                active_modes=active_modes,
                include_task=False,
            )
        start += count
    return scores.detach()


def _run_live_rollout_capture(
    runner: Any,
    observations: FrontRESSegmentLiveObservations,
    *,
    rollout_steps: int | None = None,
    capture_motion_quality: bool = True,
    zero_segment_action: bool = False,
    reset_lifecycle: dict[str, torch.Tensor] | None = None,
    pair_layout: Any | None = None,
) -> FrontRESSegmentLiveRolloutCapture:
    # FRS3-EVAL-014: step the live env and optionally capture motion-quality frames.
    try:
        v015_command = frontres_motion_command(runner)
    except (RuntimeError, AttributeError):
        v015_command = None
    v015_local_active = getattr(v015_command, "_frontres_local_scenario_active", None)
    if isinstance(v015_local_active, torch.Tensor) and bool(v015_local_active.any()):
        raise RuntimeError(
            "v015 local scenarios are forbidden on the legacy repeated-actor live rollout; "
            "use collect_frontres_v015_one_action_k_evidence() until the formal-route gate is authorized"
        )
    frontres_mode = resolve_frontres_mode_state(runner, FrontRESActorCritic)
    if pair_layout is None:
        pair_layout = configure_frontres_pair_layout(runner, is_frontres=frontres_mode.is_frontres)
    batch_size = int(observations.obs.shape[0])
    # B1: reset 完成后比较四类 role 的 episode_length_buf, 确认生命周期是否只重置了 policy rows.
    # B2: rollout 前比较 policy/candidate/noisy/clean 的 root 与 joint dynamic state, 定位 quartet 配对断点.
    # B3: 每次 env.step 后按 role 分解 done/timeout/physical termination/alive/survival 与 first-done step.
    # AUDIT-RESET-LIFECYCLE-01: 检查 index reset -> quartet dynamic state -> K-step termination 生命周期.
    # Result: quartet reset is live-aligned; anchor_pos alone terminates all 32 rows at step 0, run=E33.
    if reset_lifecycle is not None:
        print_reset_lifecycle_audit(
            runner,
            pair_layout=pair_layout,
            phase="reset",
            pair_state=snapshot_reset_pair_state(runner, pair_layout),
            **reset_lifecycle,
        )
    if rollout_steps is not None:
        rollout_k = max(1, int(rollout_steps))
        horizon_k = torch.full((batch_size,), rollout_k, dtype=torch.long, device=runner.device)
    else:
        metadata = _current_trial_metadata(runner, batch_size=batch_size, device=runner.device)
        horizon_k = metadata.horizon_k.clamp_min(1)
        rollout_k = int(horizon_k.max().item())
    audit_identity = _new_live_audit_identity(
        runner,
        pair_layout=pair_layout,
        batch_size=batch_size,
        horizon_k=horizon_k,
    )
    vel_est_error_buffer = deque(maxlen=1)
    reward_accum = None
    repair_score_accum = None
    done_any = None
    reward_frames = []
    repair_score_frames = []
    gain_step_frames = []
    survival_gain_step_frames = []
    action_step_frames = []
    done_frames = []
    survival_steps = None
    first_done_step = torch.full((batch_size,), -1, dtype=torch.long, device=runner.device)
    actor_update_mask = None
    transition_obs = None
    transition_privileged_obs = None
    transition_actions = None
    transition_log_probs = None
    transition_values = None
    transition_means = None
    transition_sigmas = None
    transition_env_actions = None
    transition_perturbation_rp = None
    transition_supervised_target = None
    action_shape = None
    env_action_shape = None
    clean_body_frames = []
    repaired_body_frames = []
    noisy_body_frames = []
    clean_root_quat_frames = []
    repaired_root_quat_frames = []
    noisy_root_quat_frames = []
    zmp_repaired_frames = []
    zmp_noisy_frames = []
    contact_repaired_frames = []
    contact_noisy_frames = []
    previous_clean_body = None
    previous_repaired_body = None
    previous_noisy_body = None
    previous_clean_root_quat = None
    previous_repaired_root_quat = None
    previous_noisy_root_quat = None
    previous_previous_clean_body = None
    previous_previous_repaired_body = None
    previous_previous_noisy_body = None
    previous_action = None
    gain_module = _gain_module()
    gain_config = (
        gain_module.FrontRESSegmentGainConfig.from_mapping(getattr(runner, "cfg", None))
        if gain_module is not None
        else None
    )
    obs = observations.obs
    privileged_obs = observations.privileged_obs
    teacher_obs = observations.teacher_obs
    ref_vel_estimator_obs = observations.ref_vel_estimator_obs
    last_obs_shape = tuple(obs.shape)

    with torch.inference_mode():
        for rollout_step in range(rollout_k):
            step_plan = prepare_frontres_rollout_step(
                runner,
                obs=obs,
                privileged_obs=privileged_obs,
                teacher_obs=teacher_obs,
                ref_vel_estimator_obs=ref_vel_estimator_obs,
                obs_raw_for_gmt=None,
                vel_est_error_buffer=vel_est_error_buffer,
                iteration=runner.current_learning_iteration,
                rollout_step=rollout_step,
                is_frontres=frontres_mode.is_frontres,
                is_task_space_mode=frontres_mode.is_task_space_mode,
                n_train=pair_layout.n_train,
                n_candidate=pair_layout.n_candidate,
                n_base=pair_layout.n_base,
                n_clean=pair_layout.n_clean,
            )
            actions = step_plan.actions
            env_actions = step_plan.env_actions
            if bool(zero_segment_action) and actions is not None and frontres_mode.is_task_space_mode:
                actions = actions.detach().clone()
                actions[: max(0, min(int(pair_layout.n_train), int(actions.shape[0])))] = 0.0
                runner.alg.transition.actions = actions.detach()
                env_actions = _zero_segment_env_actions(
                    runner,
                    obs=obs,
                    actions=actions,
                    is_frontres=frontres_mode.is_frontres,
                    is_task_space_mode=frontres_mode.is_task_space_mode,
                    n_train=pair_layout.n_train,
                    n_candidate=pair_layout.n_candidate,
                )
            action_shape = tuple(actions.shape) if actions is not None else None
            env_action_shape = tuple(env_actions.shape)
            if rollout_step == 0 and actions is not None:
                transition_obs = runner.alg.transition.observations.detach().clone()
                transition_privileged_obs = runner.alg.transition.privileged_observations.detach().clone()
                transition_env_actions = env_actions.detach().clone()
                transition_perturbation_rp = _snapshot_frontres_perturbation_rp(
                    runner,
                    num_envs=int(actions.shape[0]),
                )
                supervised_target = getattr(runner.alg.transition, "supervised_target", None)
                if supervised_target is not None and supervised_target.ndim == 2 and supervised_target.shape[-1] >= 6:
                    transition_supervised_target = supervised_target.detach().clone()
                selected_actions, selected_log_probs = _select_segment_transition_actions(runner, actions=actions)
                transition_actions = _select_executed_segment_actions(runner, actions=actions)
                transition_log_probs = selected_log_probs.detach().clone().reshape(-1)
                transition_values = runner.alg.transition.values.detach().clone().reshape(-1)
                action_mean = getattr(runner.alg.transition, "action_mean", None)
                action_sigma = getattr(runner.alg.transition, "action_sigma", None)
                if action_mean is not None:
                    if action_mean.ndim != 2 or int(action_mean.shape[-1]) != 6:
                        raise ValueError(
                            "FrontRES rollout mean must be exact direct [B,6], "
                            f"got {tuple(action_mean.shape)}"
                        )
                    transition_means = action_mean.detach().clone()
                if action_sigma is not None:
                    if action_sigma.ndim != 2 or int(action_sigma.shape[-1]) != 6:
                        raise ValueError(
                            "FrontRES rollout sigma must be exact direct [B,6], "
                            f"got {tuple(action_sigma.shape)}"
                        )
                    transition_sigmas = action_sigma.detach().clone()
                actor_update_mask = torch.zeros(actions.shape[0], device=runner.device, dtype=torch.bool)
                actor_update_mask[: max(0, min(int(pair_layout.n_train), actions.shape[0]))] = True

            selected_actions, _ = _select_segment_transition_actions(runner, actions=actions)
            executed_actions = _select_executed_segment_actions(runner, actions=actions)
            action_step_frames.append(executed_actions)

            obs, rewards, dones, infos = runner.env.step(env_actions.to(runner.env.device))
            _print_frontres_dr_runtime_probe(runner, label="after_env_step", rollout_step=rollout_step)
            rewards = rewards.to(runner.device)
            dones = dones.to(runner.device)
            paired_repair_evidence = (
                int(pair_layout.n_train) > 0
                and int(pair_layout.n_base) >= int(pair_layout.n_train)
            )
            repair_scores = (
                _segment_repair_executability_scores(
                    runner,
                    pair_layout,
                    batch_size=batch_size,
                )
                if paired_repair_evidence
                else None
            )
            horizon_active = rollout_step < horizon_k
            alive_before_step = torch.ones_like(horizon_active) if done_any is None else ~done_any
            score_active = horizon_active & alive_before_step
            scored_rewards = rewards.detach() * score_active.to(dtype=rewards.dtype)
            scored_repair = (
                repair_scores * score_active.to(dtype=repair_scores.dtype)
                if repair_scores is not None
                else None
            )
            scored_dones = dones.detach().bool() & score_active
            reward_accum = scored_rewards.clone() if reward_accum is None else reward_accum + scored_rewards
            if scored_repair is not None:
                repair_score_accum = (
                    scored_repair.clone()
                    if repair_score_accum is None
                    else repair_score_accum + scored_repair
                )
            reward_frames.append(rewards.detach().clone())
            if repair_scores is not None:
                repair_score_frames.append(repair_scores.detach().clone())
            done_frames.append(dones.detach().bool().clone())
            if done_any is None:
                done_any = torch.zeros_like(dones.detach(), dtype=torch.bool)
                survival_steps = torch.zeros_like(rewards.detach(), dtype=torch.float32)
            survival_steps = survival_steps + score_active.float()
            newly_done = scored_dones & first_done_step.lt(0)
            first_done_step[newly_done] = int(rollout_step)
            done_any = done_any | scored_dones
            time_outs = infos.get("time_outs") if isinstance(infos, dict) else None
            if isinstance(time_outs, torch.Tensor):
                time_outs = time_outs.to(runner.device).detach().bool()
                terminated = dones.detach().bool() & ~time_outs
            else:
                terminated = None
            print_reset_lifecycle_audit(
                runner,
                pair_layout=pair_layout,
                phase="step",
                rollout_step=rollout_step,
                dones=dones.detach().bool(),
                time_outs=time_outs,
                terminated=terminated,
                alive=~done_any,
                survival_steps=survival_steps,
                termination_terms=snapshot_termination_terms(
                    runner,
                    pair_layout,
                    batch_size=batch_size,
                ),
            )
            if capture_motion_quality:
                clean_body, repaired_body, noisy_body = _capture_motion_quality_frame(runner, pair_layout)
                clean_root_quat, repaired_root_quat, noisy_root_quat = _capture_root_orientation_frame(runner, pair_layout)
                physics_frame = _capture_physics_frame(runner, pair_layout)
                if clean_body is not None and repaired_body is not None and noisy_body is not None:
                    clean_body_frames.append(clean_body)
                    repaired_body_frames.append(repaired_body)
                    noisy_body_frames.append(noisy_body)
                    if clean_root_quat is not None and repaired_root_quat is not None and noisy_root_quat is not None:
                        clean_root_quat_frames.append(clean_root_quat)
                        repaired_root_quat_frames.append(repaired_root_quat)
                        noisy_root_quat_frames.append(noisy_root_quat)
                    if physics_frame is not None:
                        zmp_repaired, zmp_noisy, contact_repaired, contact_noisy = physics_frame
                        zmp_repaired_frames.append(zmp_repaired)
                        zmp_noisy_frames.append(zmp_noisy)
                        contact_repaired_frames.append(contact_repaired)
                        contact_noisy_frames.append(contact_noisy)
                    n_pair = min(int(pair_layout.n_train), int(pair_layout.n_base))
                    if n_pair > 0 and gain_module is not None and gain_config is not None:
                        train_success = (~done_any[:n_pair]).detach()
                        base_start = int(pair_layout.n_train) + int(pair_layout.n_candidate)
                        base_success = (~done_any[base_start : base_start + n_pair]).detach()
                        # B4: 逐步路径传入本步 alive increment, 由 Gain owner 用每行
                        # effective K 转成 survival quality increment. 累计这些增量后,
                        # 才与最终 raw survival_steps / K 的 Segment Gain 同源.
                        train_survival = score_active[:n_pair].float()
                        base_survival = score_active[base_start : base_start + n_pair].float()
                        step_horizon = (
                            horizon_k[:n_pair]
                            if isinstance(horizon_k, torch.Tensor)
                            else None
                        )
                        step_result = gain_module.compute_segment_gain_step(
                            clean_position=clean_body[:n_pair],
                            repaired_position=repaired_body[:n_pair],
                            noisy_position=noisy_body[:n_pair],
                            previous_clean_position=previous_clean_body,
                            previous_repaired_position=previous_repaired_body,
                            previous_noisy_position=previous_noisy_body,
                            previous_previous_clean_position=previous_previous_clean_body,
                            previous_previous_repaired_position=previous_previous_repaired_body,
                            previous_previous_noisy_position=previous_previous_noisy_body,
                            clean_root_quaternion=clean_root_quat,
                            repaired_root_quaternion=repaired_root_quat,
                            noisy_root_quaternion=noisy_root_quat,
                            repaired_zmp_margin=physics_frame[0] if physics_frame is not None else None,
                            noisy_zmp_margin=physics_frame[1] if physics_frame is not None else None,
                            repaired_contact=physics_frame[2] if physics_frame is not None else None,
                            noisy_contact=physics_frame[3] if physics_frame is not None else None,
                            repaired_success=train_success,
                            noisy_success=base_success,
                            repaired_survival=train_survival,
                            noisy_survival=base_survival,
                            effective_horizon_k=step_horizon,
                            action=executed_actions[:n_pair],
                            previous_action=previous_action,
                            config=gain_config,
                        )
                        full_step_gain = torch.full(
                            (batch_size,),
                            float("nan"),
                            device=runner.device,
                            dtype=step_result.gain_total.dtype,
                        )
                        full_step_gain[:n_pair] = step_result.gain_total
                        gain_step_frames.append(full_step_gain)
                        full_step_survival_gain = torch.full(
                            (batch_size,),
                            float("nan"),
                            device=runner.device,
                            dtype=step_result.physics_survival_gain.dtype,
                        )
                        full_step_survival_gain[:n_pair] = step_result.physics_survival_gain
                        survival_gain_step_frames.append(full_step_survival_gain)
                    previous_previous_clean_body = previous_clean_body
                    previous_previous_repaired_body = previous_repaired_body
                    previous_previous_noisy_body = previous_noisy_body
                    previous_clean_body = clean_body
                    previous_repaired_body = repaired_body
                    previous_noisy_body = noisy_body
                    previous_clean_root_quat = clean_root_quat
                    previous_repaired_root_quat = repaired_root_quat
                    previous_noisy_root_quat = noisy_root_quat
            elif int(pair_layout.n_train) > 0:
                gain_step_frames.append(torch.full((batch_size,), float("nan"), device=runner.device))
                survival_gain_step_frames.append(torch.full((batch_size,), float("nan"), device=runner.device))
            previous_action = executed_actions

            obs, privileged_obs, teacher_obs, ref_vel_estimator_obs = _read_step_observations(runner, obs, infos)
            last_obs_shape = tuple(obs.shape)

    print_reset_lifecycle_audit(
        runner,
        pair_layout=pair_layout,
        phase="final",
        first_done_step=first_done_step,
    )

    return FrontRESSegmentLiveRolloutCapture(
        rollout_k=rollout_k,
        reward_mean=float((reward_accum / horizon_k.to(dtype=reward_accum.dtype)).mean().detach().cpu()),
        done_frac=float(done_any.float().mean().detach().cpu()),
        last_obs_shape=last_obs_shape,
        action_shape=action_shape,
        env_action_shape=env_action_shape,
        transition_obs=transition_obs,
        transition_privileged_obs=transition_privileged_obs,
        transition_actions=transition_actions,
        transition_log_probs=transition_log_probs,
        transition_values=transition_values,
        transition_means=transition_means,
        transition_sigmas=transition_sigmas,
        transition_action_steps=torch.stack(action_step_frames, dim=0) if action_step_frames else None,
        reward_accum=reward_accum,
        done_any=done_any,
        reward_steps=torch.stack(reward_frames, dim=0) if reward_frames else None,
        repair_score_accum=repair_score_accum,
        repair_score_steps=torch.stack(repair_score_frames, dim=0) if repair_score_frames else None,
        gain_steps=torch.stack(gain_step_frames, dim=0) if gain_step_frames else None,
        survival_gain_steps=(
            torch.stack(survival_gain_step_frames, dim=0)
            if survival_gain_step_frames
            else None
        ),
        gain_config=gain_config,
        done_steps=torch.stack(done_frames, dim=0) if done_frames else None,
        horizon_k=horizon_k.detach().clone(),
        actor_update_mask=actor_update_mask,
        n_train=int(pair_layout.n_train),
        n_candidate=int(pair_layout.n_candidate),
        n_base=int(pair_layout.n_base),
        n_clean=int(pair_layout.n_clean),
        survival_steps=survival_steps,
        motion_clean_body_pos=_stack_motion_quality_frames(clean_body_frames),
        motion_repaired_body_pos=_stack_motion_quality_frames(repaired_body_frames),
        motion_noisy_body_pos=_stack_motion_quality_frames(noisy_body_frames),
        motion_clean_root_quat=_stack_motion_quality_frames(clean_root_quat_frames),
        motion_repaired_root_quat=_stack_motion_quality_frames(repaired_root_quat_frames),
        motion_noisy_root_quat=_stack_motion_quality_frames(noisy_root_quat_frames),
        physics_zmp_repaired_steps=_stack_motion_quality_frames(zmp_repaired_frames),
        physics_zmp_noisy_steps=_stack_motion_quality_frames(zmp_noisy_frames),
        physics_contact_repaired_steps=_stack_motion_quality_frames(contact_repaired_frames),
        physics_contact_noisy_steps=_stack_motion_quality_frames(contact_noisy_frames),
        env_actions=transition_env_actions,
        transition_perturbation_rp=transition_perturbation_rp,
        transition_supervised_target=transition_supervised_target,
        audit_transaction_id=audit_identity["audit_transaction_id"],
        audit_batch_signature=audit_identity["audit_batch_signature"],
        audit_role_signature=audit_identity["audit_role_signature"],
        audit_k_signature=audit_identity["audit_k_signature"],
        audit_segment_signature=audit_identity["audit_segment_signature"],
        audit_row_count=audit_identity["audit_row_count"],
        audit_identity_state=audit_identity["audit_identity_state"],
    )


def run_frontres_live_rollout_capture(
    runner: Any,
    observations: FrontRESSegmentLiveObservations,
    *,
    rollout_steps: int | None = None,
    capture_motion_quality: bool = True,
    zero_segment_action: bool = False,
    reset_lifecycle: dict[str, torch.Tensor] | None = None,
    pair_layout: Any | None = None,
) -> FrontRESSegmentLiveRolloutCapture:
    """Run the probe-owned rollout through a stable public orchestration seam."""

    return _run_live_rollout_capture(
        runner,
        observations,
        rollout_steps=rollout_steps,
        capture_motion_quality=capture_motion_quality,
        zero_segment_action=zero_segment_action,
        reset_lifecycle=reset_lifecycle,
        pair_layout=pair_layout,
    )


def _zero_segment_env_actions(
    runner: Any,
    *,
    obs: torch.Tensor,
    actions: torch.Tensor,
    is_frontres: bool,
    is_task_space_mode: bool,
    n_train: int,
    n_candidate: int,
) -> torch.Tensor:
    if is_task_space_mode:
        runner._apply_frontres_task_corrections(
            actions,
            n_train,
            allow_oracle=True,
            n_candidate=n_candidate if is_frontres else 0,
        )
        obs_corr, extras_corr = runner.env.get_observations()
        obs_corr_dict = extras_corr.get("observations", {})
        if runner.policy_obs_type is not None and runner.policy_obs_type in obs_corr_dict:
            obs_corr = obs_corr_dict[runner.policy_obs_type]
        obs_corr = _append_fixed_noisy_actor_context(runner, obs_corr.to(runner.device))
        obs_corr = runner._apply_obs_normalizer(obs_corr)
        return runner.alg.policy.get_env_action(obs_corr, actions)
    if hasattr(runner.alg.policy, "get_env_action"):
        return runner.alg.policy.get_env_action(obs, actions)
    return actions


def _read_step_observations(runner: Any, obs: torch.Tensor, infos: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    obs_dict = infos.get("observations", {})
    if runner.policy_obs_type is not None and runner.policy_obs_type in obs_dict:
        obs = obs_dict[runner.policy_obs_type].to(runner.device)
    else:
        obs = obs.to(runner.device)
    obs = _append_fixed_noisy_actor_context(runner, obs)
    obs = runner._apply_obs_normalizer(obs)
    if runner.privileged_obs_type is not None and runner.privileged_obs_type in obs_dict:
        privileged_obs = runner.privileged_obs_normalizer(obs_dict[runner.privileged_obs_type].to(runner.device))
    else:
        privileged_obs = obs
    if runner.teacher_obs_type is not None and runner.teacher_obs_type in obs_dict:
        teacher_obs = runner.teacher_obs_normalizer(obs_dict[runner.teacher_obs_type].to(runner.device))
    else:
        teacher_obs = privileged_obs
    if runner.ref_vel_estimator_obs_type is not None and runner.ref_vel_estimator_obs_type in obs_dict:
        ref_vel_estimator_obs = obs_dict[runner.ref_vel_estimator_obs_type].to(runner.device)
    else:
        ref_vel_estimator_obs = None
    return obs, privileged_obs, teacher_obs, ref_vel_estimator_obs
