# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-DR FrontRES evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import statistics
from typing import Any

import torch

from rsl_rl.modules import FrontRESActorCritic
from rsl_rl.runners.frontres_evaluation_reporting import write_frontres_json_csv_rows


_DR_PARAMETER_NAMES = (
    "float_prob", "float_ratio", "sink_prob", "sink_ratio", "foot_slip_prob", "foot_slip_ratio",
    "lateral_drift_prob", "lateral_drift_std", "root_tilt_prob", "root_tilt_max_rad",
    "joint_noise_prob", "joint_noise_std", "iid_prob_z", "iid_std_z", "iid_prob_xy", "iid_std_xy",
    "iid_prob_rp", "iid_std_rp", "iid_prob_ya", "iid_std_ya", "local_root_artifact_prob",
    "local_root_artifact_xy_std", "local_root_artifact_yaw_std",
)
_RUNNER_DR_STATE_NAMES = (
    "_frontres_curriculum_active_modes", "_frontres_curriculum_env_mode_groups",
    "_frontres_curriculum_mix_label", "_frontres_dr_scale_mean_last", "_frontres_dr_mix_ratio_easy",
    "_frontres_dr_mix_ratio_frontier", "_frontres_dr_mix_ratio_hard",
)
_MISSING = object()


@dataclass
class _FrontRESDRSweepSession:
    """Own fixed-DR mutation, rollout adapters and exact state restoration."""

    runner: Any
    motion_command: Any
    pert_cfg: Any
    perturber: Any
    n_train: int
    n_candidate: int
    n_base: int
    clean_start: int
    clean_end: int
    layout_name: str
    allowed_bases: tuple[str, ...]
    base_values: dict[str, float]
    runner_state: dict[str, Any]
    perturber_scale: Any
    was_training: bool

    @classmethod
    def bind(cls, runner: Any) -> "_FrontRESDRSweepSession":
        # B1: 校验 policy/command/perturber 并配置历史 paired layout, 产出 sweep session.
        if runner.training_type != "frontres" or not isinstance(runner.alg.policy, FrontRESActorCritic):
            raise ValueError("FrontRES fixed-DR sweep requires a FrontRESActorCritic runner.")
        if getattr(runner.alg.policy, "num_task_corrections", 0) <= 0:
            raise ValueError("FrontRES fixed-DR sweep requires task-space corrections.")
        env_raw = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
        manager = getattr(env_raw, "command_manager", None)
        terms = getattr(manager, "_terms", {})
        if "motion" not in terms:
            raise ValueError("FrontRES fixed-DR sweep requires the motion command term.")
        command = terms["motion"]
        pert_cfg = getattr(getattr(env_raw, "cfg", None), "motion_perturbations", None)
        perturber = getattr(command, "perturber", None)
        if pert_cfg is None or perturber is None:
            raise ValueError("FrontRES fixed-DR sweep requires motion_perturbations and perturber.")
        use_quartet = bool(runner.cfg.get("frontres_candidate_rollout_enabled", False))
        n_pair = runner.env.num_envs // (4 if use_quartet else 3)
        if use_quartet and hasattr(command, "set_frontres_quartet_baseline"):
            n_candidate, n_base = n_pair, n_pair
            n_clean = runner.env.num_envs - n_pair - n_candidate - n_base
            command.set_frontres_quartet_baseline(n_pair, n_candidate, n_base, n_clean)
            layout_name = "quartet"
        elif hasattr(command, "set_frontres_triplet_baseline"):
            n_candidate, n_base = 0, n_pair
            n_clean = runner.env.num_envs - n_pair - n_base
            command.set_frontres_triplet_baseline(n_pair, n_base, n_clean)
            layout_name = "triplet"
        else:
            raise ValueError("Motion command does not expose a FrontRES paired baseline layout.")
        mode = str(runner.cfg.get("frontres_specialist_mode", "") or "").lower()
        channels = str(runner.cfg.get("frontres_perturbation_channels", "") or "").lower()
        selector = mode or channels
        if selector in ("rp", "local_rp", "rp_only", "strong_rp"):
            allowed = ("local_rp",)
        elif selector in ("rp_z", "z_rp", "vertical_contact"):
            allowed = ("global_z", "local_rp")
        else:
            allowed = ("planar", "yaw", "global_z", "local_rp")
        base = {name: float(getattr(pert_cfg, name, 0.0) or 0.0) for name in _DR_PARAMETER_NAMES}
        state = {name: getattr(runner, name, _MISSING) for name in _RUNNER_DR_STATE_NAMES}
        return cls(
            runner, command, pert_cfg, perturber, n_pair, n_candidate, n_base,
            n_pair + n_candidate + n_base, runner.env.num_envs, layout_name, allowed, base, state,
            getattr(perturber, "_dr_scale", _MISSING), bool(runner.alg.policy.training),
        )

    def policy_observation(self, obs: torch.Tensor, extras: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
        # B1: 从 env payload 读取 policy/ref rows 并归一化, 产出 rollout observation.
        obs_dict = extras.get("observations", {})
        if self.runner.policy_obs_type is not None and self.runner.policy_obs_type in obs_dict:
            obs = obs_dict[self.runner.policy_obs_type]
        obs = self.runner._apply_obs_normalizer(obs.to(self.runner.device))
        ref = None
        if self.runner.ref_vel_estimator_obs_type is not None and self.runner.ref_vel_estimator_obs_type in obs_dict:
            ref = obs_dict[self.runner.ref_vel_estimator_obs_type].to(self.runner.device)
        if getattr(self.runner.alg, "use_estimate_ref_vel", False) and getattr(
            self.runner.alg, "ref_vel_estimator", None
        ) is not None:
            estimated = self.runner.alg.ref_vel_estimator(ref if ref is not None else obs)
            obs = torch.cat([obs, estimated], dim=-1)
        return obs, ref

    def apply_scale(self, scale: float) -> None:
        # B1: 将 fixed scale 写入允许 perturbation family, 产出本轮唯一 DR state.
        modes = set(self.allowed_bases)
        enabled = {"planar": "planar" in modes, "yaw": "yaw" in modes, "z": "global_z" in modes, "rp": "local_rp" in modes}
        probability_gate = {
            "float_prob": enabled["z"], "sink_prob": enabled["z"], "foot_slip_prob": enabled["planar"],
            "lateral_drift_prob": enabled["planar"], "root_tilt_prob": enabled["rp"],
            "iid_prob_z": enabled["z"], "iid_prob_xy": enabled["planar"],
            "iid_prob_rp": enabled["rp"], "iid_prob_ya": enabled["yaw"],
        }
        unscaled = {"joint_noise_prob", "local_root_artifact_xy_std", "local_root_artifact_yaw_std"}
        for name, base in self.base_values.items():
            if name in probability_gate:
                value = base if probability_gate[name] else 0.0
            elif name == "local_root_artifact_prob":
                value = base if enabled["planar"] or enabled["yaw"] else 0.0
            elif name in unscaled:
                if name.endswith("xy_std") and not enabled["planar"]:
                    value = 0.0
                elif name.endswith("yaw_std") and not enabled["yaw"]:
                    value = 0.0
                else:
                    value = base
            else:
                value = base * float(scale)
            setattr(self.pert_cfg, name, value)
        if hasattr(self.perturber, "set_dr_scale_env"):
            self.perturber.set_dr_scale_env(None)
        if hasattr(self.perturber, "set_family_env_masks"):
            self.perturber.set_family_env_masks(None)
        if hasattr(self.perturber, "_dr_scale"):
            self.perturber._dr_scale = float(scale)
        self.runner._frontres_curriculum_active_modes = self.allowed_bases
        self.runner._frontres_curriculum_env_mode_groups = [self.allowed_bases] * self.n_train
        self.runner._frontres_curriculum_mix_label = "fixed_eval"
        self.runner._frontres_dr_scale_mean_last = float(scale)
        self.runner._frontres_dr_mix_ratio_easy = 0.0
        self.runner._frontres_dr_mix_ratio_frontier = 1.0
        self.runner._frontres_dr_mix_ratio_hard = 0.0

    def restore(self) -> None:
        # B1: 恢复 perturbation/runner/mode snapshot, 关闭 evaluation-only mutation lifecycle.
        for name, value in self.base_values.items():
            setattr(self.pert_cfg, name, value)
        if self.perturber_scale is not _MISSING and hasattr(self.perturber, "_dr_scale"):
            self.perturber._dr_scale = self.perturber_scale
        for name, value in self.runner_state.items():
            if value is _MISSING:
                if hasattr(self.runner, name):
                    delattr(self.runner, name)
            else:
                setattr(self.runner, name, value)
        self.runner.train_mode() if self.was_training else self.runner.eval_mode()


def _frontres_dr_sweep_row(
    session: _FrontRESDRSweepSession,
    scale: float,
    rollout_steps: int,
    *,
    init_at_random_ep_len: bool,
) -> dict[str, Any]:
    # B1: 执行一个 fixed-scale rollout, 产出 FrontRES/Noisy/Clean episode evidence.
    runner = session.runner
    session.apply_scale(scale)
    runner.env.reset()
    if init_at_random_ep_len:
        high = max(1, int(getattr(runner.env, "max_episode_length", 1)))
        runner.env.episode_length_buf = torch.randint_like(runner.env.episode_length_buf, high=high)
    obs, extras = runner.env.get_observations()
    obs, ref = session.policy_observation(obs, extras)
    current = torch.zeros(runner.env.num_envs, dtype=torch.float, device=runner.device)
    lengths = {"frontres": [], "noisy": [], "clean": []}
    terminations = {"frontres": 0, "noisy": 0}
    base_start, base_end = session.n_train + session.n_candidate, session.clean_start
    for _ in range(rollout_steps):
        correction = runner.alg.policy.get_task_correction_inference(obs)
        runner._frontres_stable_route_next_mask = torch.zeros(session.n_train, device=runner.device, dtype=torch.bool)
        runner._apply_frontres_task_corrections(correction, session.n_train, allow_oracle=False, n_candidate=session.n_candidate)
        corrected, corrected_extras = runner.env.get_observations()
        corrected, _ = session.policy_observation(corrected, corrected_extras)
        runner.alg.policy._cached_observations = corrected
        actions = runner.alg.policy.get_env_action(corrected, correction)
        obs, _, dones, infos = runner.env.step(actions.to(runner.env.device))
        dones, current = dones.to(runner.device).view(-1), current + 1.0
        time_outs = infos.get("time_outs", torch.zeros_like(dones)).to(runner.device).view(-1).bool()
        for name, start, end in (
            ("frontres", 0, session.n_train), ("noisy", base_start, base_end),
            ("clean", session.clean_start, session.clean_end),
        ):
            done = dones[start:end].bool()
            if done.any():
                lengths[name].extend(current[start:end][done].detach().cpu().tolist())
                if name in terminations:
                    terminations[name] += int((done & ~time_outs[start:end]).sum().item())
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if done_ids.numel() > 0:
            current[done_ids] = 0.0
            if hasattr(runner.alg.policy, "reset"):
                runner.alg.policy.reset(dones)
        obs, ref = session.policy_observation(obs, infos)
    max_len = float(getattr(runner.env, "max_episode_length", 0) or 0)
    means = {name: statistics.mean(values) if values else max_len for name, values in lengths.items()}
    frontres_survival = 1.0 - terminations["frontres"] / max(1, rollout_steps * session.n_train)
    noisy_survival = 1.0 - terminations["noisy"] / max(1, rollout_steps * session.n_base)
    return {
        "dr_scale": float(scale), "layout": session.layout_name, "num_envs": int(runner.env.num_envs),
        "num_steps": int(rollout_steps), "frontres_episode_length_mean": float(means["frontres"]),
        "noisy_gmt_episode_length_mean": float(means["noisy"]), "gmt_episode_length_mean": float(means["noisy"]),
        "clean_gmt_episode_length_mean": float(means["clean"]),
        "frontres_minus_noisy_gmt": float(means["frontres"] - means["noisy"]),
        "frontres_minus_gmt": float(means["frontres"] - means["noisy"]),
        "frontres_step_survival": float(frontres_survival), "frontres_survival_rate": float(frontres_survival),
        "noisy_gmt_step_survival": float(noisy_survival), "gmt_survival_rate": float(noisy_survival),
        "frontres_completed_episodes": len(lengths["frontres"]), "noisy_gmt_completed_episodes": len(lengths["noisy"]),
        "gmt_completed_episodes": len(lengths["noisy"]), "clean_gmt_completed_episodes": len(lengths["clean"]),
        "allowed_perturbation_bases": list(session.allowed_bases),
    }


def evaluate_frontres_dr_sweep(
    runner: Any,
    *,
    dr_scales: list[float],
    num_iterations_per_scale: int,
    output_path: str,
    init_at_random_ep_len: bool = True,
) -> list[dict]:
    """Run a fixed-DR FrontRES-vs-GMT stress sweep without PPO updates."""

    # B1: 建立唯一 session 与 bounded rollout budget, 产出 sweep execution contract.
    if not dr_scales:
        raise ValueError("frontres_eval_dr_scales is empty.")
    session = _FrontRESDRSweepSession.bind(runner)
    rollout_steps = max(1, int(num_iterations_per_scale)) * int(runner.num_steps_per_env)
    results: list[dict] = []
    runner.eval_mode()
    try:
        # B2: 逐 scale 执行同一 session, 产出纯 result rows.
        with torch.inference_mode():
            for scale in map(float, dr_scales):
                row = _frontres_dr_sweep_row(
                    session,
                    scale,
                    rollout_steps,
                    init_at_random_ep_len=init_at_random_ep_len,
                )
                row["num_iterations"] = int(num_iterations_per_scale)
                results.append(row)
                print(
                    "[FrontRES fixed-DR eval] "
                    f"dr={scale:.4f} FrontRES={row['frontres_episode_length_mean']:.1f} "
                    f"NoisyGMT={row['noisy_gmt_episode_length_mean']:.1f} "
                    f"diff={row['frontres_minus_noisy_gmt']:+.1f} "
                    f"surv={row['frontres_step_survival']:.4f}/{row['noisy_gmt_step_survival']:.4f}",
                    flush=True,
                )
        # B3: 交给 shared reporting owner 原子写出 JSON/CSV, 产出 final artifacts.
        csv_path = write_frontres_json_csv_rows(output_path, results)
        print(f"[FrontRES fixed-DR eval] wrote {output_path}", flush=True)
        print(f"[FrontRES fixed-DR eval] wrote {csv_path}", flush=True)
        return results
    finally:
        session.restore()
