# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-DR FrontRES evaluation helpers."""

from __future__ import annotations

import json
import os
import statistics
from typing import Any

import torch

from rsl_rl.modules import FrontRESActorCritic
from rsl_rl.runners.frontres_runtime import frontres_invalidate_temporal_reference_cache


def evaluate_frontres_dr_sweep(
    runner: Any,
    *,
    dr_scales: list[float],
    num_iterations_per_scale: int,
    output_path: str,
    init_at_random_ep_len: bool = True,
) -> list[dict]:
    """Run a fixed-DR FrontRES-vs-GMT stress sweep without PPO updates."""
    if runner.training_type != "frontres" or not isinstance(runner.alg.policy, FrontRESActorCritic):
        raise ValueError("FrontRES fixed-DR sweep requires a FrontRESActorCritic runner.")
    if getattr(runner.alg.policy, "num_task_corrections", 0) <= 0:
        raise ValueError("FrontRES fixed-DR sweep requires task-space corrections.")
    if not dr_scales:
        raise ValueError("frontres_eval_dr_scales is empty.")

    env_raw = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
    if not (hasattr(env_raw, "command_manager") and "motion" in env_raw.command_manager._terms):
        raise ValueError("FrontRES fixed-DR sweep requires the motion command term.")
    motion_command = env_raw.command_manager._terms["motion"]
    pert_cfg = getattr(getattr(env_raw, "cfg", None), "motion_perturbations", None)
    perturber = getattr(motion_command, "perturber", None)
    if pert_cfg is None or perturber is None:
        raise ValueError("FrontRES fixed-DR sweep requires motion_perturbations and perturber.")

    use_quartet = bool(runner.cfg.get("frontres_candidate_rollout_enabled", False))
    n_pair = runner.env.num_envs // (4 if use_quartet else 3)
    n_train = n_pair
    if use_quartet and hasattr(motion_command, "set_frontres_quartet_baseline"):
        n_candidate = n_pair
        n_base = n_pair
        n_clean = runner.env.num_envs - n_train - n_candidate - n_base
        motion_command.set_frontres_quartet_baseline(n_train, n_candidate, n_base, n_clean)
        layout_name = "quartet"
    elif hasattr(motion_command, "set_frontres_triplet_baseline"):
        n_candidate = 0
        n_base = n_pair
        n_clean = runner.env.num_envs - n_train - n_base
        motion_command.set_frontres_triplet_baseline(n_train, n_base, n_clean)
        layout_name = "triplet"
    else:
        raise ValueError("Motion command does not expose a FrontRES paired baseline layout.")
    base_start = n_train + n_candidate
    base_end = base_start + n_base
    clean_start = base_end
    clean_end = runner.env.num_envs

    def _policy_obs_from_extras(_obs: torch.Tensor, _extras: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
        _obs_dict = _extras.get("observations", {})
        if runner.policy_obs_type is not None and runner.policy_obs_type in _obs_dict:
            _obs = _obs_dict[runner.policy_obs_type]
        _obs = runner._apply_obs_normalizer(_obs.to(runner.device))
        _ref_vel_obs = None
        if runner.ref_vel_estimator_obs_type is not None and runner.ref_vel_estimator_obs_type in _obs_dict:
            _ref_vel_obs = _obs_dict[runner.ref_vel_estimator_obs_type].to(runner.device)
        return _obs, _ref_vel_obs

    def _frontres_policy_input(_obs: torch.Tensor, _ref_vel_obs: torch.Tensor | None) -> torch.Tensor:
        if (
            getattr(runner.alg, "use_estimate_ref_vel", False)
            and getattr(runner.alg, "ref_vel_estimator", None) is not None
        ):
            _estimator_input = _ref_vel_obs if _ref_vel_obs is not None else _obs
            _estimated_ref_vel = runner.alg.ref_vel_estimator(_estimator_input)
            return torch.cat([_obs, _estimated_ref_vel], dim=-1)
        return _obs

    def _allowed_bases() -> tuple[str, ...]:
        mode = str(runner.cfg.get("frontres_specialist_mode", "") or "").lower()
        if mode in ("rp", "local_rp", "rp_only", "strong_rp"):
            return ("local_rp",)
        if mode in ("rp_z", "z_rp", "vertical_contact"):
            return ("global_z", "local_rp")
        channels = str(runner.cfg.get("frontres_perturbation_channels", "") or "").lower()
        if channels in ("rp", "local_rp", "rp_only", "strong_rp"):
            return ("local_rp",)
        if channels in ("rp_z", "z_rp", "vertical_contact"):
            return ("global_z", "local_rp")
        active_dims = runner.cfg.get("frontres_active_task_dims", None)
        if active_dims is None:
            return ("planar", "yaw", "global_z", "local_rp")
        dim_set = {int(idx) for idx in active_dims}
        bases: list[str] = []
        if dim_set.intersection({0, 1}):
            bases.append("planar")
        if 5 in dim_set:
            bases.append("yaw")
        if 2 in dim_set:
            bases.append("global_z")
        if dim_set.intersection({3, 4}):
            bases.append("local_rp")
        return tuple(bases or ["local_rp"])

    allowed_bases = _allowed_bases()

    def _pt(attr: str, default: float = 0.0) -> float:
        return float(getattr(pert_cfg, attr, default) or 0.0)

    base_values = {
        "float_prob": _pt("float_prob"),
        "float_ratio": _pt("float_ratio"),
        "sink_prob": _pt("sink_prob"),
        "sink_ratio": _pt("sink_ratio"),
        "foot_slip_prob": _pt("foot_slip_prob"),
        "foot_slip_ratio": _pt("foot_slip_ratio"),
        "lateral_drift_prob": _pt("lateral_drift_prob"),
        "lateral_drift_std": _pt("lateral_drift_std"),
        "root_tilt_prob": _pt("root_tilt_prob"),
        "root_tilt_max_rad": _pt("root_tilt_max_rad"),
        "joint_noise_prob": _pt("joint_noise_prob"),
        "joint_noise_std": _pt("joint_noise_std"),
        "iid_prob_z": _pt("iid_prob_z"),
        "iid_std_z": _pt("iid_std_z"),
        "iid_prob_xy": _pt("iid_prob_xy"),
        "iid_std_xy": _pt("iid_std_xy"),
        "iid_prob_rp": _pt("iid_prob_rp"),
        "iid_std_rp": _pt("iid_std_rp"),
        "iid_prob_ya": _pt("iid_prob_ya"),
        "iid_std_ya": _pt("iid_std_ya"),
        "local_root_artifact_prob": _pt("local_root_artifact_prob"),
        "local_root_artifact_xy_std": _pt("local_root_artifact_xy_std"),
        "local_root_artifact_yaw_std": _pt("local_root_artifact_yaw_std"),
    }

    def _apply_fixed_scale(scale: float) -> None:
        modes = set(allowed_bases)
        use_planar = "planar" in modes
        use_yaw = "yaw" in modes
        use_global_z = "global_z" in modes
        use_local_rp = "local_rp" in modes
        pert_cfg.float_prob = base_values["float_prob"] if use_global_z else 0.0
        pert_cfg.float_ratio = base_values["float_ratio"] * scale
        pert_cfg.sink_prob = base_values["sink_prob"] if use_global_z else 0.0
        pert_cfg.sink_ratio = base_values["sink_ratio"] * scale
        pert_cfg.foot_slip_prob = base_values["foot_slip_prob"] if use_planar else 0.0
        pert_cfg.foot_slip_ratio = base_values["foot_slip_ratio"] * scale
        pert_cfg.lateral_drift_prob = base_values["lateral_drift_prob"] if use_planar else 0.0
        pert_cfg.lateral_drift_std = base_values["lateral_drift_std"] * scale
        pert_cfg.root_tilt_prob = base_values["root_tilt_prob"] if use_local_rp else 0.0
        pert_cfg.root_tilt_max_rad = base_values["root_tilt_max_rad"] * scale
        pert_cfg.joint_noise_prob = base_values["joint_noise_prob"]
        pert_cfg.joint_noise_std = base_values["joint_noise_std"] * scale
        pert_cfg.iid_prob_z = base_values["iid_prob_z"] if use_global_z else 0.0
        pert_cfg.iid_std_z = base_values["iid_std_z"] * scale
        pert_cfg.iid_prob_xy = base_values["iid_prob_xy"] if use_planar else 0.0
        pert_cfg.iid_std_xy = base_values["iid_std_xy"] * scale
        pert_cfg.iid_prob_rp = base_values["iid_prob_rp"] if use_local_rp else 0.0
        pert_cfg.iid_std_rp = base_values["iid_std_rp"] * scale
        pert_cfg.iid_prob_ya = base_values["iid_prob_ya"] if use_yaw else 0.0
        pert_cfg.iid_std_ya = base_values["iid_std_ya"] * scale
        pert_cfg.local_root_artifact_prob = (
            base_values["local_root_artifact_prob"] if (use_planar or use_yaw) else 0.0
        )
        # Local-root artifact magnitudes are scaled internally by perturber._dr_scale.
        pert_cfg.local_root_artifact_xy_std = (
            base_values["local_root_artifact_xy_std"] if use_planar else 0.0
        )
        pert_cfg.local_root_artifact_yaw_std = (
            base_values["local_root_artifact_yaw_std"] if use_yaw else 0.0
        )
        if hasattr(perturber, "set_dr_scale_env"):
            perturber.set_dr_scale_env(None)
        if hasattr(perturber, "set_family_env_masks"):
            perturber.set_family_env_masks(None)
        if hasattr(perturber, "_dr_scale"):
            perturber._dr_scale = float(scale)
        runner._frontres_curriculum_active_modes = tuple(allowed_bases)
        runner._frontres_curriculum_env_mode_groups = [tuple(allowed_bases)] * n_train
        runner._frontres_curriculum_mix_label = "fixed_eval"
        runner._frontres_dr_scale_mean_last = float(scale)
        runner._frontres_dr_mix_ratio_easy = 0.0
        runner._frontres_dr_mix_ratio_frontier = 1.0
        runner._frontres_dr_mix_ratio_hard = 0.0

    max_episode_length = float(getattr(runner.env, "max_episode_length", 0) or 0)
    rollout_steps = max(1, int(num_iterations_per_scale)) * int(runner.num_steps_per_env)
    results: list[dict] = []
    was_training = runner.alg.policy.training
    runner.eval_mode()
    try:
        with torch.inference_mode():
            for scale in [float(value) for value in dr_scales]:
                _apply_fixed_scale(scale)
                runner.env.reset()
                if init_at_random_ep_len:
                    runner.env.episode_length_buf = torch.randint_like(
                        runner.env.episode_length_buf,
                        high=int(max_episode_length) if max_episode_length > 0 else 1,
                    )
                obs, extras = runner.env.get_observations()
                obs, ref_vel_estimator_obs = _policy_obs_from_extras(obs, extras)
                obs = _frontres_policy_input(obs, ref_vel_estimator_obs)
                cur_episode_length = torch.zeros(runner.env.num_envs, dtype=torch.float, device=runner.device)
                frontres_lengths: list[float] = []
                noisy_gmt_lengths: list[float] = []
                clean_gmt_lengths: list[float] = []
                frontres_terminations = 0
                noisy_gmt_terminations = 0

                for _ in range(rollout_steps):
                    task_corr = runner.alg.policy.get_task_correction_inference(obs)
                    task_corr = runner._mask_frontres_task_actions(task_corr)
                    if (
                        bool(runner.cfg.get("frontres_state_alpha_enabled", True))
                        and hasattr(runner.alg.policy, "get_state_router_alpha")
                        and n_train > 0
                    ):
                        alpha = runner.alg.policy.get_state_router_alpha(obs[:n_train]).view(-1).detach()
                        runner._frontres_state_alpha_prob_next = alpha
                        runner._frontres_state_alpha_pred_last = float(alpha.mean().item())
                    else:
                        runner._frontres_state_alpha_prob_next = torch.zeros(n_train, device=runner.device)
                        runner._frontres_state_alpha_pred_last = 0.0

                    runner._frontres_stable_route_next_mask = torch.zeros(
                        n_train, device=runner.device, dtype=torch.bool
                    )
                    runner._apply_frontres_task_corrections(
                        task_corr,
                        n_train,
                        allow_oracle=False,
                        n_candidate=n_candidate,
                    )
                    obs_corr, extras_corr = runner.env.get_observations()
                    obs_corr, ref_vel_estimator_obs_corr = _policy_obs_from_extras(obs_corr, extras_corr)
                    obs_corr = _frontres_policy_input(obs_corr, ref_vel_estimator_obs_corr)
                    runner.alg.policy._cached_observations = obs_corr
                    env_actions = runner.alg.policy.get_env_action(obs_corr, task_corr)
                    obs, _, dones, infos = runner.env.step(env_actions.to(runner.env.device))
                    dones = dones.to(runner.device).view(-1)
                    cur_episode_length += 1.0
                    time_outs = infos.get("time_outs", torch.zeros_like(dones))
                    time_outs = time_outs.to(runner.device).view(-1).bool()
                    frontres_done = dones[:n_train].bool()
                    noisy_done = dones[base_start:base_end].bool()
                    clean_done = dones[clean_start:clean_end].bool()
                    if frontres_done.any():
                        frontres_lengths.extend(
                            cur_episode_length[:n_train][frontres_done].detach().cpu().tolist()
                        )
                        frontres_terminations += int((frontres_done & ~time_outs[:n_train]).sum().item())
                    if noisy_done.any():
                        noisy_gmt_lengths.extend(
                            cur_episode_length[base_start:base_end][noisy_done].detach().cpu().tolist()
                        )
                        noisy_gmt_terminations += int(
                            (noisy_done & ~time_outs[base_start:base_end]).sum().item()
                        )
                    if clean_done.any():
                        clean_gmt_lengths.extend(
                            cur_episode_length[clean_start:clean_end][clean_done].detach().cpu().tolist()
                        )
                    done_ids = dones.nonzero(as_tuple=False).flatten()
                    if done_ids.numel() > 0:
                        cur_episode_length[done_ids] = 0.0
                        if hasattr(runner.alg.policy, "reset"):
                            runner.alg.policy.reset(dones)
                        frontres_invalidate_temporal_reference_cache(runner, dones)
                    obs, ref_vel_estimator_obs = _policy_obs_from_extras(obs, infos)
                    obs = _frontres_policy_input(obs, ref_vel_estimator_obs)

                frontres_mean = statistics.mean(frontres_lengths) if frontres_lengths else max_episode_length
                noisy_mean = statistics.mean(noisy_gmt_lengths) if noisy_gmt_lengths else max_episode_length
                clean_mean = statistics.mean(clean_gmt_lengths) if clean_gmt_lengths else max_episode_length
                frontres_step_survival = 1.0 - frontres_terminations / max(1, rollout_steps * n_train)
                noisy_step_survival = 1.0 - noisy_gmt_terminations / max(1, rollout_steps * n_base)
                row = {
                    "dr_scale": scale,
                    "layout": layout_name,
                    "num_envs": int(runner.env.num_envs),
                    "num_iterations": int(num_iterations_per_scale),
                    "num_steps": int(rollout_steps),
                    "frontres_episode_length_mean": float(frontres_mean),
                    "noisy_gmt_episode_length_mean": float(noisy_mean),
                    "gmt_episode_length_mean": float(noisy_mean),
                    "clean_gmt_episode_length_mean": float(clean_mean),
                    "frontres_minus_noisy_gmt": float(frontres_mean - noisy_mean),
                    "frontres_minus_gmt": float(frontres_mean - noisy_mean),
                    "frontres_step_survival": float(frontres_step_survival),
                    "frontres_survival_rate": float(frontres_step_survival),
                    "noisy_gmt_step_survival": float(noisy_step_survival),
                    "gmt_survival_rate": float(noisy_step_survival),
                    "frontres_completed_episodes": int(len(frontres_lengths)),
                    "noisy_gmt_completed_episodes": int(len(noisy_gmt_lengths)),
                    "gmt_completed_episodes": int(len(noisy_gmt_lengths)),
                    "clean_gmt_completed_episodes": int(len(clean_gmt_lengths)),
                    "allowed_perturbation_bases": list(allowed_bases),
                }
                results.append(row)
                print(
                    "[FrontRES fixed-DR eval] "
                    f"dr={scale:.4f} FrontRES={frontres_mean:.1f} "
                    f"NoisyGMT={noisy_mean:.1f} diff={frontres_mean - noisy_mean:+.1f} "
                    f"surv={frontres_step_survival:.4f}/{noisy_step_survival:.4f}",
                    flush=True,
                )

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        csv_path = os.path.splitext(output_path)[0] + ".csv"
        if results:
            keys = list(results[0].keys())
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(",".join(keys) + "\n")
                for row in results:
                    f.write(",".join(json.dumps(row[key]) for key in keys) + "\n")
        print(f"[FrontRES fixed-DR eval] wrote {output_path}", flush=True)
        print(f"[FrontRES fixed-DR eval] wrote {csv_path}", flush=True)
    finally:
        if was_training:
            runner.train_mode()
    return results
