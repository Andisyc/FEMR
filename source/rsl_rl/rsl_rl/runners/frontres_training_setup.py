# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

from rsl_rl.runners.frontres_dr_curriculum import (
    GMTFrontierState,
    allowed_perturbation_bases,
    choice_hash as frontres_curriculum_hash,
    choose_perturbation_choices,
    mode_complexity,
    sample_per_env_dr_strength,
    sample_perturbation_mix,
    sample_scalar_dr_strength,
    score_gmt_frontier,
    update_boundary_ema,
    update_gmt_frontier_state,
    warmup_perturbation_mode_groups,
)


@dataclass(frozen=True)
class FrontRESModeState:
    is_frontres: bool
    training_objective: str
    supervised_restore: bool
    hsl_restore: bool
    is_task_space_mode: bool
    critic_warmup_iters: int


@dataclass(frozen=True)
class FrontRESPairLayout:
    use_quartet_reward: bool
    n_train: int
    n_candidate: int
    n_base: int
    n_clean: int
    cur_reward_sum_gmt: torch.Tensor | None


@dataclass(frozen=True)
class FrontRESDRSetup:
    dr_max: float
    dr_min: float
    dr_ema_alpha: float
    dr_scale_init: float
    dr_scale: float
    r_delta_ema: float
    perturb_target: Any | None


@dataclass(frozen=True)
class FrontRESDRScaleEnvPlan:
    scale_vector: torch.Tensor | None
    mix_mode: str
    diag: dict[str, float]


@dataclass(frozen=True)
class FrontRESDRIterationPlan:
    dr_scale: float
    r_delta_ema: float
    effective_dr_scale: float | None
    dr_mix_mode: str | None
    mix_diag: dict[str, float] | None
    critic_warmup: bool
    actor_takeover_active: bool
    hsl_boundary_available: bool
    use_boundary: bool
    gmt_frontier_score: float | None
    gmt_frontier_decision: str | None
    applied: bool


def resolve_frontres_mode_state(runner: Any, policy_cls: type) -> FrontRESModeState:
    """Resolve the current FrontRES objective and task-space mode."""

    is_frontres = isinstance(runner.alg.policy, policy_cls)
    training_objective = str(getattr(
        runner.alg,
        "frontres_training_objective",
        runner.cfg.get("frontres_training_objective", "ppo_hrl"),
    )).lower()
    supervised_restore = (
        is_frontres and training_objective in ("supervised_restore", "basis_restore")
    )
    hsl_restore = (
        is_frontres
        and training_objective in ("supervised_restore", "basis_restore", "hsl_hybrid")
    )
    is_task_space_mode = (
        is_frontres and getattr(runner.alg.policy, "num_task_corrections", 0) > 0
    )
    return FrontRESModeState(
        is_frontres=is_frontres,
        training_objective=training_objective,
        supervised_restore=supervised_restore,
        hsl_restore=hsl_restore,
        is_task_space_mode=is_task_space_mode,
        critic_warmup_iters=int(runner.cfg.get("critic_warmup_iterations", 0)),
    )


def apply_frontres_debug_training_overrides(runner: Any, *, is_frontres: bool) -> None:
    """Apply shortened FrontRES debug-training schedules in-place."""

    if not (is_frontres and bool(runner.cfg.get("frontres_debug_training", False))):
        return

    def debug_value(debug_key: str, default: Any) -> Any:
        return runner.cfg.get(debug_key, default)

    debug_overrides = {
        "supervised_warmup_iterations": int(debug_value("debug_supervised_warmup_iterations", 200)),
        "supervised_warmup_diag_interval": int(debug_value("debug_supervised_warmup_diag_interval", 40)),
        "critic_warmup_iterations": int(debug_value("debug_critic_warmup_iterations", 50)),
        "dr_scale_init": float(debug_value("debug_dr_scale_init", 0.5)),
        "dr_min_scale": float(debug_value("debug_dr_min_scale", 0.3)),
        "dr_ema_alpha": float(debug_value("debug_dr_ema_alpha", 0.90)),
        "dr_p_gain": float(debug_value("debug_dr_p_gain", 0.20)),
        "dr_i_gain": float(debug_value("debug_dr_i_gain", 0.03)),
        "dr_start_ppo_actor_weight": float(debug_value("debug_dr_start_ppo_actor_weight", 1.0)),
        "frontres_safe_gap_per_step": float(debug_value("debug_frontres_safe_gap_per_step", 0.003)),
        "frontres_broken_gap_per_step": float(debug_value("debug_frontres_broken_gap_per_step", 0.08)),
        "frontres_gap_gate_temp": float(debug_value("debug_frontres_gap_gate_temp", 0.005)),
    }
    for key, value in debug_overrides.items():
        runner.cfg[key] = value

    actor_warmup_debug = int(debug_value("debug_ppo_actor_warmup_iterations", 50))
    actor_ramp_debug = int(debug_value("debug_ppo_actor_ramp_iterations", 200))
    runner.cfg["ppo_actor_warmup_iterations"] = actor_warmup_debug
    runner.cfg["ppo_actor_ramp_iterations"] = actor_ramp_debug
    runner.alg_cfg["ppo_actor_warmup_iterations"] = actor_warmup_debug
    runner.alg_cfg["ppo_actor_ramp_iterations"] = actor_ramp_debug

    print(
        "[Runner] === FrontRES DEBUG TRAINING enabled ===\n"
        f"[Runner]   supervised_warmup_iterations={runner.cfg['supervised_warmup_iterations']}, "
        f"critic_warmup_iterations={runner.cfg['critic_warmup_iterations']}\n"
        f"[Runner]   ppo_actor_warmup_iterations={actor_warmup_debug}, "
        f"ppo_actor_ramp_iterations={actor_ramp_debug}\n"
        f"[Runner]   dr_scale_init={runner.cfg['dr_scale_init']}, "
        f"dr_min_scale={runner.cfg['dr_min_scale']}, "
        f"dr_p_gain={runner.cfg['dr_p_gain']}, dr_i_gain={runner.cfg['dr_i_gain']}, "
        f"dr_start_ppo_actor_weight={runner.cfg['dr_start_ppo_actor_weight']}\n"
        f"[Runner]   frontres_safe_gap_per_step={runner.cfg['frontres_safe_gap_per_step']}, "
        f"frontres_broken_gap_per_step={runner.cfg['frontres_broken_gap_per_step']}, "
        f"frontres_gap_gate_temp={runner.cfg['frontres_gap_gate_temp']}",
        flush=True,
    )


def frontres_ppo_actor_weight_for_iter(
    runner: Any,
    *,
    iteration: int,
    is_frontres: bool,
    supervised_restore: bool,
) -> float:
    """Linear supervised-to-PPO takeover schedule for the current iteration."""

    if not (is_frontres and hasattr(runner.alg, "ppo_actor_weight")):
        return 1.0
    if supervised_restore:
        return 0.0
    actor_warmup = int(runner.alg_cfg.get(
        "ppo_actor_warmup_iterations",
        runner.cfg.get("ppo_actor_warmup_iterations", 0),
    ))
    actor_ramp = int(runner.alg_cfg.get(
        "ppo_actor_ramp_iterations",
        runner.cfg.get("ppo_actor_ramp_iterations", 0),
    ))
    phase_iter = max(0, iteration)
    if phase_iter < actor_warmup:
        return 0.0
    if actor_ramp > 0 and phase_iter < actor_warmup + actor_ramp:
        weight = (phase_iter - actor_warmup + 1) / float(actor_ramp)
        return max(0.0, min(1.0, weight))
    return 1.0


def update_frontres_supervised_controller(
    runner: Any,
    *,
    loss_dict: dict,
    positive_gain_frac: float | None,
    harm_rate: float | None,
) -> None:
    """Decay supervised learning into a one-way anchor once PPO is learnable."""

    if not bool(runner.cfg.get("frontres_state_supervised_controller_enabled", True)):
        return
    if not hasattr(runner.alg, "lambda_supervised"):
        return
    runner.alg.state_supervised_controller_enabled = True
    lam = float(getattr(runner.alg, "lambda_supervised", 0.0))
    if lam <= 0.0:
        return

    anchor = float(runner.cfg.get(
        "frontres_supervised_anchor_weight",
        runner.cfg.get("lambda_supervised_min", 0.02),
    ))
    hold_iters = int(runner.cfg.get("frontres_supervised_min_hold_iters", 5))
    seen = int(getattr(runner, "_frontres_supervised_controller_seen", 0)) + 1
    runner._frontres_supervised_controller_seen = seen
    if seen < max(0, hold_iters):
        return

    pos_trigger = float(runner.cfg.get("frontres_supervised_positive_gain_trigger", 0.52))
    harm_limit = float(runner.cfg.get("frontres_supervised_harm_limit", 0.06))
    grad_low = float(runner.cfg.get("frontres_supervised_grad_cos_low", 0.03))
    decay_good = float(runner.cfg.get("frontres_supervised_decay_good", 0.985))
    decay_conflict = float(runner.cfg.get("frontres_supervised_decay_conflict", 0.97))
    grad_cos = float(loss_dict.get("grad_cos_ppo_supervised", 0.0))

    learnable = (
        positive_gain_frac is not None
        and harm_rate is not None
        and float(positive_gain_frac) >= pos_trigger
        and float(harm_rate) <= harm_limit
    )
    factor = 1.0
    if learnable:
        factor = min(factor, decay_good)
    if grad_cos < grad_low and positive_gain_frac is not None and float(positive_gain_frac) >= 0.50:
        factor = min(factor, decay_conflict)
    if factor < 1.0:
        setattr(runner.alg, "lambda_supervised", max(anchor, lam * factor))


def configure_frontres_pair_layout(runner: Any, *, is_frontres: bool) -> FrontRESPairLayout:
    """Configure FrontRES projected/candidate/noisy/clean env layout."""

    if not is_frontres:
        return FrontRESPairLayout(False, 0, 0, 0, 0, None)

    runner.alg.state_supervised_controller_enabled = bool(
        runner.cfg.get("frontres_state_supervised_controller_enabled", True)
    )
    use_quartet_reward = bool(runner.cfg.get("frontres_candidate_rollout_enabled", False))
    n_pair = runner.env.num_envs // (4 if use_quartet_reward else 3)
    n_train = n_pair
    if use_quartet_reward:
        n_candidate = n_pair
        n_base = n_pair
        n_clean = runner.env.num_envs - n_train - n_candidate - n_base
        print(
            f"[Runner] FrontRES B1 quartet reward: "
            f"{n_train} projected envs + {n_candidate} candidate envs + "
            f"{n_base} noisy-GMT envs + {n_clean} clean-GMT envs",
            flush=True,
        )
    else:
        n_candidate = 0
        n_base = n_pair
        n_clean = runner.env.num_envs - n_train - n_base
        print(
            f"[Runner] FrontRES B1 triplet reward: "
            f"{n_train} FrontRES envs + {n_base} noisy-GMT envs + {n_clean} clean-GMT envs",
            flush=True,
        )

    env_pair = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    if hasattr(env_pair, "command_manager") and "motion" in env_pair.command_manager._terms:
        motion_command = env_pair.command_manager._terms["motion"]
        if use_quartet_reward and hasattr(motion_command, "set_frontres_quartet_baseline"):
            motion_command.set_frontres_quartet_baseline(n_train, n_candidate, n_base, n_clean)
            print(
                "[Runner] FrontRES B1 quartet baseline enabled: "
                "projected/candidate/noisy-GMT/clean-GMT share motion/frame; clean-GMT has zero perturbation",
                flush=True,
            )
        elif hasattr(motion_command, "set_frontres_triplet_baseline"):
            motion_command.set_frontres_triplet_baseline(n_train, n_base, n_clean)
            n_candidate = 0
            use_quartet_reward = False
            print(
                "[Runner] FrontRES B1 triplet baseline enabled: "
                "FrontRES/noisy-GMT/clean-GMT share motion/frame; clean-GMT has zero perturbation",
                flush=True,
            )
        elif hasattr(motion_command, "set_frontres_paired_baseline"):
            motion_command.set_frontres_paired_baseline(n_train)
            print(
                "[Runner] FrontRES B1 paired baseline enabled (legacy two-way fallback): "
                "env i and env i+N_train share motion/frame/perturbation",
                flush=True,
            )

    cur_reward_sum_gmt = torch.zeros(
        n_candidate + n_base + n_clean,
        dtype=torch.float,
        device=runner.device,
    )
    return FrontRESPairLayout(
        use_quartet_reward=use_quartet_reward,
        n_train=n_train,
        n_candidate=n_candidate,
        n_base=n_base,
        n_clean=n_clean,
        cur_reward_sum_gmt=cur_reward_sum_gmt,
    )


def build_frontres_task_action_mask(runner: Any, *, is_task_space_mode: bool) -> torch.Tensor | None:
    """Create the active task-action mask used for runtime safety."""

    if not is_task_space_mode:
        return None
    active_dims = runner.cfg.get("frontres_active_task_dims", None)
    if active_dims is None:
        return None
    task_action_dim = int(getattr(runner.alg.policy, "total_output_dim", 8))
    task_action_mask = torch.zeros(task_action_dim, device=runner.device)
    for idx in active_dims:
        idx = int(idx)
        if not 0 <= idx < task_action_dim:
            raise ValueError(
                "frontres_active_task_dims contains an index outside the "
                f"current FrontRES action dim {task_action_dim}."
            )
        task_action_mask[idx] = 1.0
    print(
        "[Runner] FrontRES task-space action mask enabled: "
        f"dim={task_action_dim} mask={task_action_mask.detach().cpu().tolist()}",
        flush=True,
    )
    return task_action_mask


def initialize_frontres_dr_setup(runner: Any, *, is_frontres: bool) -> FrontRESDRSetup:
    """Initialize persistent DR/frontier state and snapshot base perturbation magnitudes."""

    dr_max = float(runner.cfg.get("dr_max_scale", 4.0))
    dr_min = float(runner.cfg.get("dr_min_scale", 0.0))
    dr_ema_alpha = float(runner.cfg.get("dr_ema_alpha", 0.95))
    dr_scale_init = float(runner.cfg.get("dr_scale_init", 0.3))
    dr_scale = float(getattr(runner, "_dr_scale", dr_scale_init))
    dr_scale = max(dr_scale, dr_scale_init)
    r_delta_ema = 0.1

    if is_frontres:
        safe_low = float(getattr(runner, "_frontres_gmt_frontier_safe_low", dr_scale_init))
        safe_low = max(dr_min, min(dr_max, safe_low))
        runner._frontres_gmt_frontier_safe_low = safe_low
        broken_high = getattr(runner, "_frontres_gmt_frontier_broken_high", None)
        if broken_high is not None:
            broken_high = max(safe_low, min(dr_max, float(broken_high)))
        runner._frontres_gmt_frontier_broken_high = broken_high
        runner._frontres_gmt_frontier_probe_score = getattr(
            runner, "_frontres_gmt_frontier_probe_score", None
        )
        runner._frontres_gmt_frontier_decision = getattr(
            runner, "_frontres_gmt_frontier_decision", "init"
        )
        runner._frontres_gmt_frontier_probe_scale = float(getattr(
            runner, "_frontres_gmt_frontier_probe_scale", safe_low
        ))

    perturb_target = _snapshot_frontres_perturbation_target(runner, is_frontres=is_frontres)
    if is_frontres and perturb_target is not None:
        print(
            f"[Runner] Adaptive DR controller: "
            f"boundary_enabled={runner.cfg.get('frontres_boundary_dr_enabled', True)}, "
            f"boundary_takeover={runner.cfg.get('frontres_boundary_dr_during_actor_takeover', False)}, "
            f"fallback_PI=(Kp={runner.cfg.get('dr_p_gain', 0.10)}, "
            f"Ki={runner.cfg.get('dr_i_gain', 0.01)}, "
            f"target={runner.cfg.get('dr_target_r_delta', 0.01)}), "
            f"max_scale={dr_max}, ema_alpha={dr_ema_alpha}, "
            f"start_actor_weight={runner.cfg.get('dr_start_ppo_actor_weight', 1.0)}, "
            f"resume dr_scale={dr_scale:.3f}"
        )
    elif is_frontres:
        print("[Runner] WARNING: FrontRES DR enabled but env.cfg.motion_perturbations not found")

    return FrontRESDRSetup(
        dr_max=dr_max,
        dr_min=dr_min,
        dr_ema_alpha=dr_ema_alpha,
        dr_scale_init=dr_scale_init,
        dr_scale=dr_scale,
        r_delta_ema=r_delta_ema,
        perturb_target=perturb_target,
    )


def frontres_curriculum_allowed_bases(runner: Any) -> tuple[str, ...]:
    """Map the active FrontRES output dimensions to repairable perturbation families."""

    return allowed_perturbation_bases(runner.cfg.get("frontres_active_task_dims", None))


def set_frontres_perturbation_curriculum(
    runner: Any,
    *,
    progress: float,
    seq_idx: int,
    is_frontres: bool,
) -> None:
    """Select one perturbation family group for warmup-style rollout collection."""

    choices, complexity = frontres_curriculum_choices(
        runner,
        progress=progress,
        seq_idx=seq_idx,
        is_frontres=is_frontres,
    )
    choice = choices[frontres_curriculum_hash(seq_idx) % len(choices)]
    runner._frontres_curriculum_active_modes = tuple(choice)
    runner._frontres_curriculum_complexity = mode_complexity(tuple(choice), complexity)


def frontres_curriculum_choices(
    runner: Any,
    *,
    progress: float,
    seq_idx: int,
    is_frontres: bool,
) -> tuple[list[tuple[str, ...]], str]:
    stats = getattr(runner, "_frontres_boundary_ema", None)
    if stats is None:
        stats = getattr(runner, "_last_frontres_boundary_stats", None)
    return choose_perturbation_choices(
        runner.cfg,
        runner.cfg.get("frontres_active_task_dims", None),
        progress,
        seq_idx,
        boundary_stats=stats,
        is_frontres=is_frontres,
    )


def sample_frontres_rollout_perturbation_mix(
    runner: Any,
    *,
    progress: float,
    seq_idx: int,
    n_train: int,
    n_candidate: int,
    n_base: int,
    n_clean: int,
    is_frontres: bool,
) -> None:
    """Assign perturbation mode groups across split-env rollout branches."""

    stats = getattr(runner, "_frontres_boundary_ema", None)
    if stats is None:
        stats = getattr(runner, "_last_frontres_boundary_stats", None)
    plan = sample_perturbation_mix(
        runner.cfg,
        runner.cfg.get("frontres_active_task_dims", None),
        progress,
        seq_idx,
        int(n_train),
        boundary_stats=stats,
        is_frontres=is_frontres,
    )
    groups = plan.groups
    runner._frontres_curriculum_active_modes = plan.active_modes
    runner._frontres_curriculum_complexity = plan.complexity
    runner._frontres_curriculum_env_mode_groups = groups

    env_raw = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    if not (hasattr(env_raw, "command_manager") and "motion" in env_raw.command_manager._terms):
        return
    motion_command = env_raw.command_manager._terms["motion"]
    if not (
        hasattr(motion_command, "perturber")
        and hasattr(motion_command.perturber, "set_family_env_masks")
    ):
        return

    family_masks = {
        "planar": torch.zeros(runner.env.num_envs, dtype=torch.bool, device=runner.device),
        "yaw": torch.zeros(runner.env.num_envs, dtype=torch.bool, device=runner.device),
        "global_z": torch.zeros(runner.env.num_envs, dtype=torch.bool, device=runner.device),
        "local_rp": torch.zeros(runner.env.num_envs, dtype=torch.bool, device=runner.device),
    }
    candidate_start = int(n_train)
    base_start = int(n_train) + int(n_candidate)
    for env_i, group in enumerate(groups[: int(n_train)]):
        for mode in group:
            if mode in family_masks:
                family_masks[mode][env_i] = True
                if env_i < int(n_candidate):
                    family_masks[mode][candidate_start + env_i] = True
                if env_i < int(n_base):
                    family_masks[mode][base_start + env_i] = True
    # Clean-GMT envs intentionally remain all False; baseline masking keeps them clean.
    motion_command.perturber.set_family_env_masks(family_masks)


def maybe_print_frontres_perturbation_curriculum(runner: Any, *, is_frontres: bool) -> None:
    if not (is_frontres and bool(runner.cfg.get("frontres_perturbation_curriculum_enabled", True))):
        return
    allowed_bases = ",".join(frontres_curriculum_allowed_bases(runner))
    print(
        "[Runner] Perturbation curriculum enabled: "
        f"bases=[{allowed_bases}], "
        f"adaptive={runner.cfg.get('frontres_adaptive_perturb_curriculum_enabled', True)}, "
        f"single_until={runner.cfg.get('frontres_curriculum_single_until', 0.30)}, "
        f"two_until={runner.cfg.get('frontres_curriculum_two_until', 0.70)}, "
        f"full_prob={runner.cfg.get('frontres_curriculum_full_prob', 0.05)}",
        flush=True,
    )


def set_frontres_curriculum_modes(runner: Any, modes: tuple[str, ...]) -> None:
    runner._frontres_curriculum_active_modes = tuple(modes)
    runner._frontres_curriculum_complexity = mode_complexity(tuple(modes))


def frontres_warmup_perturbation_mode_groups(
    runner: Any,
    *,
    seq_idx: int,
) -> list[tuple[str, ...]]:
    """Return perturbation families to mix inside one warmup update."""

    return warmup_perturbation_mode_groups(
        runner.cfg,
        runner.cfg.get("frontres_active_task_dims", None),
        seq_idx,
        current_active_modes=tuple(getattr(runner, "_frontres_curriculum_active_modes", ())),
    )


def apply_frontres_dr_scale(
    runner: Any,
    *,
    scale: float,
    is_frontres: bool,
    perturb_target: Any | None,
) -> None:
    """Write the current scalar DR scale and active family gates to the perturber."""

    if not (is_frontres and perturb_target is not None):
        return
    env_raw = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    if not (hasattr(env_raw, "command_manager") and "motion" in env_raw.command_manager._terms):
        return
    motion_command = env_raw.command_manager._terms["motion"]
    if not hasattr(motion_command, "perturber"):
        return

    def pt(attr: str, default: float = 0.0) -> float:
        return getattr(perturb_target, attr, default)

    modes = set(getattr(
        runner,
        "_frontres_curriculum_active_modes",
        ("planar", "yaw", "global_z", "local_rp"),
    ))
    planar = "planar" in modes
    yaw = "yaw" in modes
    global_z = "global_z" in modes
    local_rp = "local_rp" in modes

    motion_command.perturber.cfg.float_prob = pt("float_prob") if global_z else 0.0
    motion_command.perturber.cfg.float_ratio = pt("float_ratio") * scale
    motion_command.perturber.cfg.sink_prob = pt("sink_prob") if global_z else 0.0
    motion_command.perturber.cfg.sink_ratio = pt("sink_ratio") * scale
    motion_command.perturber.cfg.foot_slip_prob = pt("foot_slip_prob") if planar else 0.0
    motion_command.perturber.cfg.foot_slip_ratio = pt("foot_slip_ratio") * scale
    motion_command.perturber.cfg.lateral_drift_prob = pt("lateral_drift_prob") if planar else 0.0
    motion_command.perturber.cfg.lateral_drift_std = pt("lateral_drift_std") * scale
    motion_command.perturber.cfg.root_tilt_prob = pt("root_tilt_prob") if local_rp else 0.0
    motion_command.perturber.cfg.root_tilt_max_rad = pt("root_tilt_max_rad") * scale
    motion_command.perturber.cfg.joint_noise_prob = pt("joint_noise_prob")
    motion_command.perturber.cfg.joint_noise_std = pt("joint_noise_std") * scale
    motion_command.perturber.cfg.iid_prob_z = pt("iid_prob_z") if global_z else 0.0
    motion_command.perturber.cfg.iid_std_z = pt("iid_std_z") * scale
    motion_command.perturber.cfg.iid_prob_xy = pt("iid_prob_xy") if planar else 0.0
    motion_command.perturber.cfg.iid_std_xy = pt("iid_std_xy") * scale
    motion_command.perturber.cfg.iid_prob_rp = pt("iid_prob_rp") if local_rp else 0.0
    motion_command.perturber.cfg.iid_std_rp = pt("iid_std_rp") * scale
    motion_command.perturber.cfg.iid_prob_ya = pt("iid_prob_ya") if yaw else 0.0
    motion_command.perturber.cfg.iid_std_ya = pt("iid_std_ya") * scale
    motion_command.perturber.cfg.local_root_artifact_prob = (
        pt("local_root_artifact_prob") if (planar or yaw) else 0.0
    )
    # Local artifact magnitudes are multiplied by perturber._dr_scale at burst sampling time.
    motion_command.perturber.cfg.local_root_artifact_xy_std = (
        pt("local_root_artifact_xy_std") if planar else 0.0
    )
    motion_command.perturber.cfg.local_root_artifact_yaw_std = (
        pt("local_root_artifact_yaw_std") if yaw else 0.0
    )
    motion_command.perturber._dr_scale = float(scale)
    if hasattr(motion_command.perturber, "set_dr_scale_env"):
        motion_command.perturber.set_dr_scale_env(None)


def frontres_mixed_dr_scale(
    runner: Any,
    *,
    frontier_scale: float,
    enabled: bool,
    seq_idx: int,
    dr_min: float,
    dr_max: float,
) -> tuple[float, str]:
    """Sample one easy/frontier/hard perturbation magnitude around the frontier."""

    return sample_scalar_dr_strength(
        runner.cfg,
        frontier_scale,
        enabled,
        seq_idx,
        dr_min=dr_min,
        dr_max=dr_max,
    )


def frontres_mixed_dr_scale_env(
    runner: Any,
    *,
    frontier_scale: float,
    enabled: bool,
    seq_idx: int,
    n_train: int,
    n_candidate: int,
    n_base: int,
    dr_min: float,
    dr_max: float,
) -> FrontRESDRScaleEnvPlan:
    """Sample per-env easy/frontier/hard DR strength for split rollout branches."""

    plan = sample_per_env_dr_strength(
        runner.cfg,
        frontier_scale,
        enabled,
        seq_idx,
        n_train=int(n_train),
        n_candidate=int(n_candidate),
        n_base=int(n_base),
        num_envs=runner.env.num_envs,
        dr_min=dr_min,
        dr_max=dr_max,
    )
    if plan.scale_vector is None:
        runner._frontres_dr_mix_class_train = None
        return FrontRESDRScaleEnvPlan(None, plan.mix_mode, plan.diag)
    scales = torch.tensor(plan.scale_vector, device=runner.device, dtype=torch.float32)
    if plan.mix_class is not None:
        runner._frontres_dr_mix_class_train = torch.tensor(
            plan.mix_class,
            device=runner.device,
            dtype=torch.long,
        )
    else:
        runner._frontres_dr_mix_class_train = None
    return FrontRESDRScaleEnvPlan(scales, plan.mix_mode, plan.diag)


def apply_frontres_dr_scale_env(
    runner: Any,
    *,
    scale_vec: torch.Tensor,
    is_frontres: bool,
    perturb_target: Any | None,
) -> None:
    """Write unscaled base magnitudes and per-env DR scale vector to the perturber."""

    if not (is_frontres and perturb_target is not None):
        return
    apply_frontres_dr_scale(
        runner,
        scale=1.0,
        is_frontres=is_frontres,
        perturb_target=perturb_target,
    )
    env_raw = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    if not (hasattr(env_raw, "command_manager") and "motion" in env_raw.command_manager._terms):
        return
    motion_command = env_raw.command_manager._terms["motion"]
    if hasattr(motion_command, "perturber") and hasattr(motion_command.perturber, "set_dr_scale_env"):
        motion_command.perturber.set_dr_scale_env(scale_vec)


def apply_frontres_iteration_dr_controller(
    runner: Any,
    *,
    iteration: int,
    is_frontres: bool,
    frontres_hsl_restore: bool,
    perturb_target: Any | None,
    critic_warmup_iters: int,
    ppo_actor_weight_current: float,
    dr_scale: float,
    dr_scale_init: float,
    dr_min: float,
    dr_max: float,
    dr_ema_alpha: float,
    r_delta_ema: float,
    n_train: int,
    n_candidate: int,
    n_base: int,
    n_clean: int,
    lenbuffer_gmt_base: Any,
    lenbuffer_gmt_base_frontier: Any,
    frontres_policy_cls: type,
) -> FrontRESDRIterationPlan:
    """Update and apply the per-iteration FrontRES DR controller state."""

    critic_warmup = (
        isinstance(runner.alg.policy, frontres_policy_cls)
        and int(critic_warmup_iters) > 0
        and int(iteration) < int(critic_warmup_iters)
    )
    dr_start_actor_weight = float(runner.cfg.get("dr_start_ppo_actor_weight", 1.0))
    actor_takeover_active = (
        is_frontres
        and (not critic_warmup)
        and ppo_actor_weight_current < dr_start_actor_weight
    )

    boundary_enabled_for_iter = bool(runner.cfg.get("frontres_boundary_dr_enabled", True))
    boundary_takeover_for_iter = bool(runner.cfg.get("frontres_boundary_dr_during_actor_takeover", False))
    hsl_boundary_available = (
        frontres_hsl_restore
        and perturb_target is not None
        and boundary_enabled_for_iter
        and getattr(runner, "_last_frontres_boundary_stats", None) is not None
        and (not critic_warmup)
        and ((not actor_takeover_active) or boundary_takeover_for_iter)
    )

    gmt_frontier_score = getattr(runner, "_frontres_gmt_frontier_probe_score", None)
    gmt_frontier_decision: str | None = None
    use_boundary = False
    effective_dr_scale: float | None = None
    dr_mix_mode: str | None = None
    mix_diag: dict[str, float] | None = None
    applied = False

    if frontres_hsl_restore and perturb_target is not None and not hsl_boundary_available:
        sup_dr_start = float(runner.cfg.get("frontres_supervised_dr_scale_start", dr_scale_init))
        sup_dr_end = float(runner.cfg.get("frontres_supervised_dr_scale_end", sup_dr_start))
        sup_dr_ramp = max(1, int(runner.cfg.get("frontres_supervised_dr_ramp_iters", 500)))
        sup_dr_delay = max(0, int(runner.cfg.get("frontres_supervised_dr_delay_iters", 0)))
        sup_dr_phase = max(0, int(iteration) - sup_dr_delay)
        sup_dr_frac = min(1.0, max(0.0, sup_dr_phase / float(sup_dr_ramp)))
        dr_scale = sup_dr_start + (sup_dr_end - sup_dr_start) * sup_dr_frac
        if critic_warmup or actor_takeover_active:
            dr_scale = dr_scale_init
        dr_scale = max(dr_min, min(dr_max, dr_scale))
        runner._dr_scale = dr_scale
        curriculum_progress = _frontres_curriculum_progress(runner, iteration)
        if critic_warmup or actor_takeover_active:
            curriculum_progress = 0.0
        sample_frontres_rollout_perturbation_mix(
            runner,
            progress=curriculum_progress,
            seq_idx=iteration,
            n_train=n_train,
            n_candidate=n_candidate,
            n_base=n_base,
            n_clean=n_clean,
            is_frontres=is_frontres,
        )
        mix_strength_enabled = not (critic_warmup or actor_takeover_active)
        scale_plan = frontres_mixed_dr_scale_env(
            runner,
            frontier_scale=dr_scale,
            enabled=mix_strength_enabled,
            seq_idx=iteration,
            n_train=n_train,
            n_candidate=n_candidate,
            n_base=n_base,
            dr_min=dr_min,
            dr_max=dr_max,
        )
        effective_dr_scale, dr_mix_mode, mix_diag = _apply_frontres_strength_plan(
            runner,
            scale_plan=scale_plan,
            dr_scale=dr_scale,
            enabled=mix_strength_enabled,
            seq_idx=iteration,
            dr_min=dr_min,
            dr_max=dr_max,
            is_frontres=is_frontres,
            perturb_target=perturb_target,
        )
        applied = True
    elif is_frontres and perturb_target is not None:
        r_delta_ema = (
            dr_ema_alpha * r_delta_ema
            + (1.0 - dr_ema_alpha) * getattr(runner, "_last_r_delta_mean", 0.0)
        )
        boundary_enabled = bool(runner.cfg.get("frontres_boundary_dr_enabled", True))
        boundary_takeover = bool(runner.cfg.get("frontres_boundary_dr_during_actor_takeover", False))
        boundary_stats = getattr(runner, "_last_frontres_boundary_stats", None)
        use_boundary = (
            boundary_enabled
            and boundary_stats is not None
            and (not critic_warmup)
            and ((not actor_takeover_active) or boundary_takeover)
        )

        if critic_warmup or (actor_takeover_active and not boundary_takeover):
            dr_scale = dr_scale_init
            runner._dr_hold_just_ended = True
        elif use_boundary:
            if getattr(runner, "_dr_hold_just_ended", False):
                if hasattr(runner, "_dr_prev_error"):
                    delattr(runner, "_dr_prev_error")
                runner._dr_hold_just_ended = False

            ema = update_boundary_ema(
                runner.cfg,
                getattr(runner, "_frontres_boundary_ema", None),
                boundary_stats,
            )
            runner._frontres_boundary_ema = ema
            gmt_frontier_enabled = bool(runner.cfg.get("frontres_gmt_frontier_probe_enabled", True))
            gmt_frontier_decision = "disabled"
            gmt_frontier_score = getattr(runner, "_frontres_gmt_frontier_probe_score", None)
            if gmt_frontier_enabled:
                frontier_update = _update_frontres_gmt_frontier(
                    runner,
                    iteration=iteration,
                    dr_scale=dr_scale,
                    dr_scale_init=dr_scale_init,
                    dr_min=dr_min,
                    dr_max=dr_max,
                    lenbuffer_gmt_base=lenbuffer_gmt_base,
                    lenbuffer_gmt_base_frontier=lenbuffer_gmt_base_frontier,
                )
                dr_scale = frontier_update.next_dr_scale
                gmt_frontier_score = getattr(runner, "_frontres_gmt_frontier_probe_score", None)
                gmt_frontier_decision = getattr(runner, "_frontres_gmt_frontier_decision", None)
            else:
                dr_scale = _frontres_boundary_scale_step(
                    runner,
                    ema=ema,
                    dr_scale=dr_scale,
                    dr_min=dr_min,
                    dr_max=dr_max,
                )
        else:
            if getattr(runner, "_dr_hold_just_ended", False):
                dr_scale = dr_scale_init
                if hasattr(runner, "_dr_prev_error"):
                    delattr(runner, "_dr_prev_error")
                runner._dr_hold_just_ended = False
            dr_scale = _frontres_pi_scale_step(
                runner,
                dr_scale=dr_scale,
                dr_min=dr_min,
                dr_max=dr_max,
                r_delta_ema=r_delta_ema,
            )

        runner._dr_scale = dr_scale
        curriculum_progress = _frontres_curriculum_progress(runner, iteration)
        sample_frontres_rollout_perturbation_mix(
            runner,
            progress=curriculum_progress,
            seq_idx=iteration,
            n_train=n_train,
            n_candidate=n_candidate,
            n_base=n_base,
            n_clean=n_clean,
            is_frontres=is_frontres,
        )
        scale_plan = frontres_mixed_dr_scale_env(
            runner,
            frontier_scale=dr_scale,
            enabled=use_boundary,
            seq_idx=iteration,
            n_train=n_train,
            n_candidate=n_candidate,
            n_base=n_base,
            dr_min=dr_min,
            dr_max=dr_max,
        )
        effective_dr_scale, dr_mix_mode, mix_diag = _apply_frontres_strength_plan(
            runner,
            scale_plan=scale_plan,
            dr_scale=dr_scale,
            enabled=use_boundary,
            seq_idx=iteration,
            dr_min=dr_min,
            dr_max=dr_max,
            is_frontres=is_frontres,
            perturb_target=perturb_target,
        )
        applied = True

    return FrontRESDRIterationPlan(
        dr_scale=dr_scale,
        r_delta_ema=r_delta_ema,
        effective_dr_scale=effective_dr_scale,
        dr_mix_mode=dr_mix_mode,
        mix_diag=mix_diag,
        critic_warmup=critic_warmup,
        actor_takeover_active=actor_takeover_active,
        hsl_boundary_available=hsl_boundary_available,
        use_boundary=use_boundary,
        gmt_frontier_score=gmt_frontier_score,
        gmt_frontier_decision=gmt_frontier_decision,
        applied=applied,
    )


def _frontres_curriculum_progress(runner: Any, iteration: int) -> float:
    curriculum_iters = int(runner.cfg.get("frontres_curriculum_total_iterations", 1500))
    curriculum_iters = max(1, curriculum_iters)
    return min(1.0, max(0.0, int(iteration) / float(curriculum_iters)))


def _apply_frontres_strength_plan(
    runner: Any,
    *,
    scale_plan: FrontRESDRScaleEnvPlan,
    dr_scale: float,
    enabled: bool,
    seq_idx: int,
    dr_min: float,
    dr_max: float,
    is_frontres: bool,
    perturb_target: Any | None,
) -> tuple[float, str, dict[str, float]]:
    if scale_plan.scale_vector is None:
        effective_dr_scale, dr_mix_mode = frontres_mixed_dr_scale(
            runner,
            frontier_scale=dr_scale,
            enabled=enabled,
            seq_idx=seq_idx,
            dr_min=dr_min,
            dr_max=dr_max,
        )
        mix_diag = {"easy": 0.0, "frontier": 1.0, "hard": 0.0, "mean": effective_dr_scale}
        runner._frontres_dr_mix_class_train = None
        apply_frontres_dr_scale(
            runner,
            scale=effective_dr_scale,
            is_frontres=is_frontres,
            perturb_target=perturb_target,
        )
    else:
        effective_dr_scale = float(scale_plan.diag["mean"])
        dr_mix_mode = scale_plan.mix_mode
        mix_diag = scale_plan.diag
        apply_frontres_dr_scale_env(
            runner,
            scale_vec=scale_plan.scale_vector,
            is_frontres=is_frontres,
            perturb_target=perturb_target,
        )

    runner._frontres_effective_dr_scale = effective_dr_scale
    runner._frontres_dr_mix_mode = dr_mix_mode
    runner._frontres_dr_frontier_scale = dr_scale
    runner._frontres_dr_mix_easy_frac = float(mix_diag["easy"])
    runner._frontres_dr_mix_frontier_frac = float(mix_diag["frontier"])
    runner._frontres_dr_mix_hard_frac = float(mix_diag["hard"])
    runner._frontres_dr_mix_mean_scale = float(mix_diag["mean"])
    return effective_dr_scale, dr_mix_mode, mix_diag


def _update_frontres_gmt_frontier(
    runner: Any,
    *,
    iteration: int,
    dr_scale: float,
    dr_scale_init: float,
    dr_min: float,
    dr_max: float,
    lenbuffer_gmt_base: Any,
    lenbuffer_gmt_base_frontier: Any,
):
    probe_scale = float(getattr(runner, "_frontres_gmt_frontier_probe_scale", dr_scale))
    probe_scale = max(dr_min, min(dr_max, probe_scale))
    safe_low = float(getattr(runner, "_frontres_gmt_frontier_safe_low", dr_scale_init))
    safe_low = max(dr_min, min(dr_max, safe_low))
    broken_high = getattr(runner, "_frontres_gmt_frontier_broken_high", None)
    score_ref = max(
        1e-6,
        float(runner.cfg.get("frontres_gmt_frontier_ref_episode_len", 0.0) or 0.0),
    )
    if score_ref <= 1e-6:
        score_ref = max(1e-6, float(getattr(runner, "max_episode_length", 1.0)))
    per_env_mix_active = (
        getattr(runner, "_frontres_dr_mix_mode", None) == "per_env"
        and getattr(runner, "_frontres_dr_mix_class_train", None) is not None
    )
    frontier_len_source = lenbuffer_gmt_base_frontier if per_env_mix_active else lenbuffer_gmt_base
    runner._frontres_gmt_frontier_probe_source = "frontier-class" if per_env_mix_active else "all-base"
    runner._frontres_gmt_frontier_probe_samples = len(frontier_len_source)
    gmt_frontier_score = score_gmt_frontier(list(frontier_len_source), score_ref)
    frontier_state = GMTFrontierState(
        safe_low=safe_low,
        broken_high=broken_high,
        probe_scale=probe_scale,
        probe_score=getattr(runner, "_frontres_gmt_frontier_probe_score", None),
        decision=str(getattr(runner, "_frontres_gmt_frontier_decision", "init")),
        confirmed=float(getattr(runner, "_frontres_gmt_frontier_confirmed", dr_scale)),
    )
    frontier_update = update_gmt_frontier_state(
        runner.cfg,
        frontier_state,
        score=gmt_frontier_score,
        samples=len(frontier_len_source),
        dr_scale=dr_scale,
        dr_scale_init=dr_scale_init,
        dr_min=dr_min,
        dr_max=dr_max,
    )
    new_state = frontier_update.state
    runner._frontres_gmt_frontier_safe_low = new_state.safe_low
    runner._frontres_gmt_frontier_broken_high = new_state.broken_high
    runner._frontres_gmt_frontier_confirmed = float(new_state.confirmed)
    runner._frontres_gmt_frontier_probe_scale = new_state.probe_scale
    runner._frontres_gmt_frontier_probe_score = new_state.probe_score
    runner._frontres_gmt_frontier_decision = new_state.decision
    return frontier_update


def _frontres_boundary_scale_step(
    runner: Any,
    *,
    ema: dict[str, float],
    dr_scale: float,
    dr_min: float,
    dr_max: float,
) -> float:
    safe = float(ema.get("safe", 0.0))
    repair = float(ema.get("repair", ema.get("fragile", 0.0)))
    broken = float(ema.get("broken", 0.0))
    gainpos = float(ema.get("positive_gain", 0.5))
    safe_hi = float(runner.cfg.get("frontres_boundary_safe_high", 0.45))
    broken_hi = float(runner.cfg.get("frontres_boundary_broken_high", 0.35))
    broken_target = float(runner.cfg.get("frontres_boundary_broken_target", 0.25))
    repair_lo = float(runner.cfg.get(
        "frontres_boundary_repair_low",
        runner.cfg.get("frontres_boundary_fragile_low", 0.45),
    ))
    repair_hi = float(runner.cfg.get(
        "frontres_boundary_repair_high",
        runner.cfg.get("frontres_boundary_fragile_high", 0.70),
    ))
    gain_hi = float(runner.cfg.get("frontres_boundary_positive_gain_high", 0.55))
    gain_lo = float(runner.cfg.get("frontres_boundary_positive_gain_low", 0.45))
    step = float(runner.cfg.get("frontres_boundary_dr_step", 0.03))
    factor = 1.0
    if broken > broken_hi:
        factor = 1.0 - step * min(3.0, 1.0 + (broken - broken_hi) / max(1.0 - broken_hi, 1e-6))
    elif safe > safe_hi and broken < broken_target:
        factor = 1.0 + step
    elif (repair_lo <= repair <= repair_hi) and gainpos > gain_hi and broken < broken_hi:
        factor = 1.0 + 0.5 * step
    elif gainpos < gain_lo and broken > broken_target:
        factor = 1.0 - 0.5 * step
    factor = max(0.80, min(1.10, factor))
    return max(dr_min, min(dr_max, dr_scale * factor))


def _frontres_pi_scale_step(
    runner: Any,
    *,
    dr_scale: float,
    dr_min: float,
    dr_max: float,
    r_delta_ema: float,
) -> float:
    kp = float(runner.cfg.get("dr_p_gain", 0.10))
    ki = float(runner.cfg.get("dr_i_gain", 0.01))
    dr_target = float(runner.cfg.get("dr_target_r_delta", 0.01))
    error = r_delta_ema - dr_target
    prev_err = getattr(runner, "_dr_prev_error", error)
    delta = kp * (error - prev_err) + ki * error
    runner._dr_prev_error = error
    return max(dr_min, min(dr_max, dr_scale + delta))


def _snapshot_frontres_perturbation_target(runner: Any, *, is_frontres: bool) -> Any | None:
    if not is_frontres:
        return None
    env_raw = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    if not (hasattr(env_raw, "cfg") and hasattr(env_raw.cfg, "motion_perturbations")):
        return None
    pt = env_raw.cfg.motion_perturbations
    return SimpleNamespace(
        float_prob=float(pt.float_prob),
        float_ratio=float(pt.float_ratio),
        sink_prob=float(pt.sink_prob),
        sink_ratio=float(pt.sink_ratio),
        foot_slip_prob=float(pt.foot_slip_prob),
        foot_slip_ratio=float(pt.foot_slip_ratio),
        lateral_drift_prob=float(getattr(pt, "lateral_drift_prob", 0.0)),
        lateral_drift_std=float(getattr(pt, "lateral_drift_std", 0.0)),
        root_tilt_prob=float(getattr(pt, "root_tilt_prob", 0.0)),
        root_tilt_max_rad=float(getattr(pt, "root_tilt_max_rad", 0.0)),
        joint_noise_prob=float(getattr(pt, "joint_noise_prob", 0.0)),
        joint_noise_std=float(getattr(pt, "joint_noise_std", 0.0)),
        iid_prob_z=float(getattr(pt, "iid_prob_z", 0.0)),
        iid_std_z=float(getattr(pt, "iid_std_z", 0.0)),
        iid_prob_xy=float(getattr(pt, "iid_prob_xy", 0.0)),
        iid_std_xy=float(getattr(pt, "iid_std_xy", 0.0)),
        iid_prob_rp=float(getattr(pt, "iid_prob_rp", 0.0)),
        iid_std_rp=float(getattr(pt, "iid_std_rp", 0.0)),
        iid_prob_ya=float(getattr(pt, "iid_prob_ya", 0.0)),
        iid_std_ya=float(getattr(pt, "iid_std_ya", 0.0)),
        local_root_artifact_prob=float(getattr(pt, "local_root_artifact_prob", 0.0)),
        local_root_artifact_xy_std=float(getattr(pt, "local_root_artifact_xy_std", 0.0)),
        local_root_artifact_yaw_std=float(getattr(pt, "local_root_artifact_yaw_std", 0.0)),
    )
