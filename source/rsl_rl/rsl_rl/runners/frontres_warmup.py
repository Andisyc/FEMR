# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class FrontRESWarmupDecision:
    iterations: int
    skip_message: str | None = None


def resolve_frontres_warmup_iterations(
    *,
    configured_iterations: int,
    start_iter: int,
    warmup_complete: bool,
) -> FrontRESWarmupDecision:
    """Resolve whether joint warmup should run for this training process."""
    warmup_iters = max(0, int(configured_iterations))
    if start_iter > 0 and warmup_iters > 0:
        return FrontRESWarmupDecision(
            iterations=0,
            skip_message=f"[Runner] Resuming from iter {start_iter} — skipping supervised warmup",
        )
    if warmup_complete and warmup_iters > 0:
        return FrontRESWarmupDecision(
            iterations=0,
            skip_message="[Runner] Loaded a completed FrontRES warmup checkpoint — skipping supervised warmup",
        )
    return FrontRESWarmupDecision(iterations=warmup_iters)


def should_exit_after_frontres_stage1_warmup(
    cfg: dict[str, Any],
    *,
    is_frontres: bool,
    warmup_iters: int,
) -> bool:
    """Return true when Stage 1 should stop after writing model_warmup.pt."""

    return (
        bool(is_frontres)
        and int(warmup_iters) > 0
        and bool(cfg.get("frontres_stage1_exit_after_warmup", False))
    )


def smoothstep_fraction(index: int, total: int) -> float:
    """Smooth 0..1 warmup progress used for DR scale interpolation."""
    if total > 1:
        frac = index / float(total - 1)
    else:
        frac = 1.0
    return frac * frac * (3.0 - 2.0 * frac)


def interpolate_warmup_scale(start: float, end: float, fraction: float) -> float:
    return float(start) + (float(end) - float(start)) * float(fraction)


def run_frontres_joint_warmup(
    runner: Any,
    *,
    is_frontres: bool,
    warmup_iters: int,
    dr_scale_init: float,
    dr_scale: float,
    n_train: int,
    n_base: int,
    n_clean: int,
    perturb_target: Any,
    curriculum_allowed_bases: Callable[[Any], tuple[str, ...]],
    set_perturbation_curriculum: Callable[..., None],
    set_curriculum_modes: Callable[[Any, tuple[str, ...]], None],
    warmup_perturbation_mode_groups: Callable[..., list[tuple[str, ...]]],
    apply_dr_scale: Callable[..., None],
) -> None:
    """Run the FrontRES joint supervised/energy warmup phase before PPO."""
    if not (is_frontres and warmup_iters > 0):
        return

    self = runner

    def _frontres_curriculum_allowed_bases() -> tuple[str, ...]:
        return curriculum_allowed_bases(self)

    def _set_frontres_perturbation_curriculum(progress: float, seq_idx: int) -> None:
        return set_perturbation_curriculum(
            self,
            progress=progress,
            seq_idx=seq_idx,
            is_frontres=is_frontres,
        )

    def _set_frontres_curriculum_modes(modes: tuple[str, ...]) -> None:
        return set_curriculum_modes(self, tuple(modes))

    def _frontres_warmup_perturbation_mode_groups(seq_idx: int) -> list[tuple[str, ...]]:
        return warmup_perturbation_mode_groups(
            self,
            seq_idx=seq_idx,
        )

    def _apply_frontres_dr_scale(scale: float) -> None:
        return apply_dr_scale(
            self,
            scale=scale,
            is_frontres=is_frontres,
            perturb_target=perturb_target,
        )

    _warmup_dr_scale_end = float(self.cfg.get("supervised_warmup_dr_scale", dr_scale_init))
    _warmup_dr_scale_start = float(self.cfg.get(
        "supervised_warmup_dr_scale_start",
        self.cfg.get("supervised_warmup_dr_scale_min", _warmup_dr_scale_end),
    ))
    _warmup_dr_scale_start = max(0.0, _warmup_dr_scale_start)
    _warmup_dr_scale_end = max(0.0, _warmup_dr_scale_end)
    _warmup_lr = float(self.cfg.get("supervised_warmup_lr", 1e-4))
    _warmup_epochs = int(self.cfg.get("supervised_warmup_epochs", 5))
    _warmup_steps = int(self.cfg.get("supervised_warmup_steps_per_iter", self.num_steps_per_env))
    _warmup_steps = max(1, min(_warmup_steps, self.num_steps_per_env))
    _warmup_max_envs = int(self.cfg.get("supervised_warmup_max_envs_per_step", self.env.num_envs))
    _warmup_max_envs = max(1, min(_warmup_max_envs, self.env.num_envs))
    _warmup_valid_w = float(getattr(self.alg, "supervised_valid_loss_weight", 4.0))
    _warmup_dir_w = float(getattr(self.alg, "supervised_direction_loss_weight", 0.1))
    _warmup_energy_w = float(self.cfg.get("frontres_warmup_energy_loss_weight", 1.0))
    _warmup_diag_interval = int(self.cfg.get(
        "supervised_warmup_diag_interval", max(1, warmup_iters // 5)))
    _warmup_diag_interval = max(1, _warmup_diag_interval)
    _warmup_opt = torch.optim.Adam(
        list(self.alg.policy.residual_actor.parameters())
        + list(self.alg.policy.critic.parameters()),
        lr=_warmup_lr,
    )

    # Import once to avoid per-step overhead.
    from whole_body_tracking.tasks.tracking.mdp.observations import \
        get_supervision_target_task_space as _get_warmup_target

    _env_raw = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
    _nfo = self.alg.policy.num_frontres_obs
    if _nfo <= 0:
        _nfo = self.alg.policy.num_actor_obs

    _warmup_dr_desc = (
        f"{_warmup_dr_scale_start}->{_warmup_dr_scale_end}"
        if abs(_warmup_dr_scale_end - _warmup_dr_scale_start) > 1e-8
        else f"{_warmup_dr_scale_end}"
    )
    print(f"[Runner] === Joint warmup: {warmup_iters} iters "
          f"(dr_scale={_warmup_dr_desc}, lr={_warmup_lr}, epochs={_warmup_epochs}, "
          f"steps_per_iter={_warmup_steps}, "
          f"max_envs_per_step={_warmup_max_envs}, "
          f"frontres_input={_nfo} dims, energy_w={_warmup_energy_w}, "
          f"perturb_schedule={self.cfg.get('supervised_warmup_perturbation_schedule', self.cfg.get('frontres_warmup_perturbation_schedule', 'mixed_single'))}) ===",
          flush=True)

    loss = torch.tensor(0.0, device=self.device)
    for _wu in range(warmup_iters):
        _warmup_frac = smoothstep_fraction(_wu, warmup_iters)
        _warmup_dr_scale = interpolate_warmup_scale(
            _warmup_dr_scale_start,
            _warmup_dr_scale_end,
            _warmup_frac,
        )
        _set_frontres_perturbation_curriculum(_warmup_frac, _wu)
        _warmup_mode_groups = _frontres_warmup_perturbation_mode_groups(_wu)
        if not _warmup_mode_groups:
            _warmup_mode_groups = [tuple(_frontres_curriculum_allowed_bases())]

        _wo_list: list[torch.Tensor] = []
        _wt_list: list[torch.Tensor] = []
        _wc_list: list[torch.Tensor] = []
        _we_list: list[torch.Tensor] = []

        # Use no_grad rather than inference_mode: warmup samples are later fed
        # back through trainable actor/critic networks.
        with torch.no_grad():
            for _step in range(_warmup_steps):
                _mode_group = _warmup_mode_groups[
                    (_wu * max(_warmup_steps, 1) + _step) % len(_warmup_mode_groups)
                ]
                _set_frontres_curriculum_modes(tuple(_mode_group))
                _apply_frontres_dr_scale(_warmup_dr_scale)
                obs, extras = self.env.get_observations()
                obs_dict = extras.get("observations", {})
                _p_obs_raw = obs_dict.get(self.policy_obs_type, obs).to(self.device)
                _p_obs = self._apply_obs_normalizer(_p_obs_raw)

                env_actions = self.alg.policy.get_env_action(
                    _p_obs,
                    torch.zeros(_p_obs.shape[0], self.alg.policy.total_output_dim, device=self.device),
                )

                obs, rewards_wu, dones, extras = self.env.step(env_actions.to(self.env.device))
                obs_dict = extras.get("observations", {})
                _p_obs_raw = obs_dict.get(self.policy_obs_type, obs).to(self.device)
                _p_obs = self._apply_obs_normalizer(_p_obs_raw)
                if self.privileged_obs_type is not None and self.privileged_obs_type in obs_dict:
                    _c_obs = self.privileged_obs_normalizer(
                        obs_dict[self.privileged_obs_type].to(self.device)
                    )
                else:
                    _c_obs = _p_obs
                _target = _get_warmup_target(_env_raw, "motion").to(self.device)
                _mcmd_wu = _env_raw.command_manager._terms.get("motion")
                if _mcmd_wu is not None:
                    _target = self._frontres_action_cone.project_task_target(_mcmd_wu, _target)
                if n_train > 0 and n_base > 0 and n_clean > 0:
                    _n_energy = min(n_train, n_base, n_clean)
                    if _mcmd_wu is not None:
                        _executability = self._frontres_executability
                        _active_modes = tuple(getattr(self, "_frontres_curriculum_active_modes", ()))
                        _, _exec_wu_components = _executability.exec_score(_mcmd_wu, return_components=True)
                        _wu_modes = [
                            tuple(getattr(self, "_frontres_curriculum_active_modes", ()))
                        ] * _n_energy
                        _r_perturbed_wu = _executability.exec_score_for_modes(
                            _exec_wu_components,
                            n_train,
                            _n_energy,
                            mode_groups=_wu_modes,
                            active_modes=_active_modes,
                        ).view(-1)
                        _, _feasible_wu_components = _executability.feasible_oracle_exec_score(
                            _mcmd_wu, n_train, _n_energy, return_components=True
                        )
                        _r_feasible_wu = _executability.exec_score_for_modes(
                            _feasible_wu_components,
                            0,
                            _n_energy,
                            mode_groups=_wu_modes,
                            active_modes=_active_modes,
                        ).to(self.device).view(-1)
                        _energy_target = (_r_feasible_wu - _r_perturbed_wu).clamp(min=0.0).unsqueeze(-1)
                    else:
                        _energy_target = torch.zeros(_n_energy, 1, device=self.device)
                    _p_obs = _p_obs[:_n_energy]
                    _c_obs = _c_obs[:_n_energy]
                    _target = _target[:_n_energy]
                else:
                    _energy_target = torch.zeros(_p_obs.shape[0], 1, device=self.device)

                if _warmup_max_envs < _p_obs.shape[0]:
                    _sample_ids = torch.randperm(_p_obs.shape[0], device=self.device)[:_warmup_max_envs]
                    _p_obs = _p_obs[_sample_ids]
                    _c_obs = _c_obs[_sample_ids]
                    _target = _target[_sample_ids]
                    _energy_target = _energy_target[_sample_ids]

                _wo_list.append(_p_obs[:, :_nfo])
                _wt_list.append(_target)
                _wc_list.append(_c_obs)
                _we_list.append(_energy_target)

        _all_obs = torch.cat(_wo_list, dim=0)
        _all_tgt = torch.cat(_wt_list, dim=0)
        _all_critic_obs = torch.cat(_wc_list, dim=0)
        _all_energy = torch.cat(_we_list, dim=0)
        _N = _all_obs.shape[0]
        _last_actor_loss = torch.tensor(0.0, device=self.device)
        _last_energy_loss = torch.tensor(0.0, device=self.device)
        _sup_mask = None
        _active_sup_dims = getattr(self.alg, "frontres_active_task_dims", None)
        if _active_sup_dims is not None:
            _sup_mask = torch.zeros(_all_tgt.shape[-1], device=self.device, dtype=_all_tgt.dtype)
            for _dim in _active_sup_dims:
                _dim = int(_dim)
                if 0 <= _dim < _sup_mask.numel():
                    _sup_mask[_dim] = 1.0
            if _wu == 0:
                print(
                    "[Runner] Joint warmup supervised active mask: "
                    f"{[float(x) for x in _sup_mask.detach().cpu().tolist()]} "
                    "(dx,dy,dz,droll,dpitch,dyaw)",
                    flush=True,
                )

        for epoch in range(_warmup_epochs):
            perm = torch.randperm(_N, device=self.device)
            for i in range(0, _N, 4096):
                idx = perm[i:i + 4096]
                pred = self.alg.policy.residual_actor(_all_obs[idx])
                if getattr(self.alg.policy, "num_task_corrections", 0) > 0:
                    pred_sup = torch.cat([
                        torch.tanh(pred[:, :3]) * self.alg.policy.max_delta_pos,
                        torch.tanh(pred[:, 3:6]) * self.alg.policy.max_delta_rpy,
                    ], dim=-1)
                    target_sup = torch.cat([
                        _all_tgt[idx, :3].clamp(
                            -self.alg.policy.max_delta_pos, self.alg.policy.max_delta_pos),
                        _all_tgt[idx, 3:].clamp(
                            -self.alg.policy.max_delta_rpy, self.alg.policy.max_delta_rpy),
                    ], dim=-1)
                else:
                    pred_sup = pred[:, :_all_tgt.shape[-1]]
                    target_sup = _all_tgt[idx]
                if _sup_mask is not None:
                    pred_sup = pred_sup * _sup_mask.view(1, -1)
                    target_sup = target_sup * _sup_mask.view(1, -1)

                target_norm = target_sup.norm(dim=-1)
                valid = target_norm > 1e-4
                pos_valid = target_sup[:, :3].norm(dim=-1) > 1e-4
                rpy_valid = target_sup[:, 3:].norm(dim=-1) > 1e-4
                pos_weight = torch.ones_like(target_norm)
                rpy_weight = torch.ones_like(target_norm)
                if pos_valid.any():
                    pos_weight[pos_valid] = _warmup_valid_w
                if rpy_valid.any():
                    rpy_weight[rpy_valid] = _warmup_valid_w
                pos_weight = pos_weight / pos_weight.mean().clamp(min=1e-6)
                rpy_weight = rpy_weight / rpy_weight.mean().clamp(min=1e-6)

                pos_err = torch.nn.functional.huber_loss(
                    pred_sup[:, :3], target_sup[:, :3].detach(), reduction="none").mean(dim=-1)
                rpy_err = torch.nn.functional.huber_loss(
                    pred_sup[:, 3:], target_sup[:, 3:].detach(), reduction="none").mean(dim=-1)
                _rpy_w = float(getattr(self.alg, "supervised_rpy_loss_weight", 1.0))
                loss = (pos_err * pos_weight).mean() + _rpy_w * (rpy_err * rpy_weight).mean()
                if _warmup_dir_w > 0.0:
                    direction_loss = torch.zeros((), device=self.device)
                    if pos_valid.any():
                        direction_loss = direction_loss + (
                            1.0 - torch.nn.functional.cosine_similarity(
                                pred_sup[pos_valid, :3],
                                target_sup[pos_valid, :3].detach(),
                                dim=-1,
                            ).mean()
                        )
                    if rpy_valid.any():
                        direction_loss = direction_loss + (
                            1.0 - torch.nn.functional.cosine_similarity(
                                pred_sup[rpy_valid, 3:],
                                target_sup[rpy_valid, 3:].detach(),
                                dim=-1,
                            ).mean()
                        )
                    loss = loss + _warmup_dir_w * direction_loss
                _conf_w = float(getattr(self.alg, "supervised_conf_loss_weight", 0.0))
                if (
                    getattr(self.alg.policy, "num_task_corrections", 0) > 0
                    and pred.shape[-1] >= 8
                    and _conf_w > 0
                    and int(getattr(self.alg.policy, "task_conf_dim", 2)) == 2
                ):
                    target_conf = valid.view(-1, 1).to(pred.dtype)
                    conf_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        pred[:, 6:8], target_conf.expand(-1, 2))
                    loss = loss + _conf_w * conf_loss
                actor_loss = loss
                value_pred = self.alg.policy.evaluate(_all_critic_obs[idx])
                energy_loss = torch.nn.functional.huber_loss(
                    value_pred, _all_energy[idx].detach(), reduction="mean"
                )
                loss = actor_loss + _warmup_energy_w * energy_loss
                _warmup_opt.zero_grad()
                loss.backward()
                _warmup_opt.step()
                _last_actor_loss = actor_loss.detach()
                _last_energy_loss = energy_loss.detach()

        if (_wu + 1) % _warmup_diag_interval == 0 or (_wu + 1) == warmup_iters:
            with torch.inference_mode():
                _pred_all_raw = self.alg.policy.residual_actor(_all_obs[:, :_nfo])
                if getattr(self.alg.policy, "num_task_corrections", 0) > 0:
                    _pred_all = torch.cat([
                        torch.tanh(_pred_all_raw[:, :3]) * self.alg.policy.max_delta_pos,
                        torch.tanh(_pred_all_raw[:, 3:6]) * self.alg.policy.max_delta_rpy,
                    ], dim=-1)
                    _target_all = torch.cat([
                        _all_tgt[:, :3].clamp(
                            -self.alg.policy.max_delta_pos, self.alg.policy.max_delta_pos),
                        _all_tgt[:, 3:].clamp(
                            -self.alg.policy.max_delta_rpy, self.alg.policy.max_delta_rpy),
                    ], dim=-1)
                else:
                    _pred_all = _pred_all_raw[:, :_all_tgt.shape[-1]]
                    _target_all = _all_tgt
                if _sup_mask is not None:
                    _pred_all = _pred_all * _sup_mask.view(1, -1)
                    _target_all = _target_all * _sup_mask.view(1, -1)

                _valid_all = _target_all.norm(dim=-1) > 1e-4
                _valid_pos = _target_all[:, :3].norm(dim=-1) > 1e-4
                _valid_rpy = _target_all[:, 3:].norm(dim=-1) > 1e-4

                def _masked_cos(a, b, mask):
                    if mask.any():
                        return torch.nn.functional.cosine_similarity(
                            a[mask], b[mask], dim=-1).mean().item()
                    return 0.0

                def _masked_mae(a, b, mask):
                    if mask.any():
                        return (a[mask] - b[mask]).abs().mean().item()
                    return 0.0

                def _masked_norm(a, mask):
                    if mask.any():
                        return a[mask].norm(dim=-1).mean().item()
                    return 0.0

                def _masked_abs_mean(a, mask):
                    if mask.any():
                        return a[mask].abs().mean().item()
                    return 0.0

                def _sign_agreement(a, b, mask):
                    if mask.any():
                        return ((a[mask] * b[mask]) > 0.0).float().mean().item()
                    return 0.0

                if _valid_all.any():
                    _warmup_cos = torch.nn.functional.cosine_similarity(
                        _pred_all[_valid_all], _target_all[_valid_all], dim=-1).mean().item()
                else:
                    _warmup_cos = 0.0
                _valid_frac = _valid_all.float().mean().item()
                _valid_pos_frac = _valid_pos.float().mean().item()
                _valid_rpy_frac = _valid_rpy.float().mean().item()
                _cos_pos = _masked_cos(_pred_all[:, :3], _target_all[:, :3], _valid_pos)
                _cos_rpy = _masked_cos(_pred_all[:, 3:], _target_all[:, 3:], _valid_rpy)
                _valid_roll = _target_all[:, 3].abs() > 1e-4
                _valid_pitch = _target_all[:, 4].abs() > 1e-4
                _valid_yaw = _target_all[:, 5].abs() > 1e-4
                _sign_roll = _sign_agreement(_pred_all[:, 3], _target_all[:, 3], _valid_roll)
                _sign_pitch = _sign_agreement(_pred_all[:, 4], _target_all[:, 4], _valid_pitch)
                _sign_yaw = _sign_agreement(_pred_all[:, 5], _target_all[:, 5], _valid_yaw)
                _abs_tgt_roll = _masked_abs_mean(_target_all[:, 3], _valid_roll)
                _abs_tgt_pitch = _masked_abs_mean(_target_all[:, 4], _valid_pitch)
                _abs_tgt_yaw = _masked_abs_mean(_target_all[:, 5], _valid_yaw)
                _abs_pred_roll = _masked_abs_mean(_pred_all[:, 3], _valid_roll)
                _abs_pred_pitch = _masked_abs_mean(_pred_all[:, 4], _valid_pitch)
                _abs_pred_yaw = _masked_abs_mean(_pred_all[:, 5], _valid_yaw)
                _valid_roll_frac = _valid_roll.float().mean().item()
                _valid_pitch_frac = _valid_pitch.float().mean().item()
                _valid_yaw_frac = _valid_yaw.float().mean().item()
                _valid_x = _target_all[:, 0].abs() > 1e-4
                _valid_y = _target_all[:, 1].abs() > 1e-4
                _valid_z = _target_all[:, 2].abs() > 1e-4
                _valid_x_frac = _valid_x.float().mean().item()
                _valid_y_frac = _valid_y.float().mean().item()
                _valid_z_frac = _valid_z.float().mean().item()
                _mae_pos = _masked_mae(_pred_all[:, :3], _target_all[:, :3], _valid_pos)
                _mae_rpy = _masked_mae(_pred_all[:, 3:], _target_all[:, 3:], _valid_rpy)
                _pred_pos_norm = _masked_norm(_pred_all[:, :3], _valid_pos)
                _tgt_pos_norm = _masked_norm(_target_all[:, :3], _valid_pos)
                _pred_rpy_norm = _masked_norm(_pred_all[:, 3:], _valid_rpy)
                _tgt_rpy_norm = _masked_norm(_target_all[:, 3:], _valid_rpy)
                _obs_pos_best_cos = 0.0
                _obs_rpy_best_cos = 0.0
                _obs_rpy_best_neg_cos = 0.0
                _obs_rpy_best_norm = 0.0
                _obs_z_best_sign = 0.0
                _obs_roll_best_sign = 0.0
                _obs_pitch_best_sign = 0.0
                _obs_z_best_corr = 0.0
                _obs_roll_best_corr = 0.0
                _obs_pitch_best_corr = 0.0
                if _all_obs.shape[-1] >= 30:
                    _extra = _all_obs[:, :30]
                    _target_pos = _target_all[:, :3]
                    _target_rpy = _target_all[:, 3:]

                    def _scalar_corr(a, b, mask):
                        if mask.any():
                            a_m = a[mask] - a[mask].mean()
                            b_m = b[mask] - b[mask].mean()
                            return (a_m * b_m).mean() / (
                                a_m.std(unbiased=False) * b_m.std(unbiased=False)
                            ).clamp(min=1e-6)
                        return torch.tensor(0.0, device=self.device)

                    def _scalar_sign(a, b, mask):
                        if mask.any():
                            return ((a[mask] * b[mask]) > 0.0).float().mean()
                        return torch.tensor(0.0, device=self.device)

                    def _score_extra_layout(_pos_frames, _rpy_frames):
                        _pos_cos_vals = []
                        _rpy_cos_vals = []
                        _rpy_neg_cos_vals = []
                        _rpy_norm_vals = []
                        _z_sign_vals = []
                        _roll_sign_vals = []
                        _pitch_sign_vals = []
                        _z_corr_vals = []
                        _roll_corr_vals = []
                        _pitch_corr_vals = []
                        for _hist_i in range(_pos_frames.shape[1]):
                            _pos_mask_i = _valid_pos & (_pos_frames[:, _hist_i].norm(dim=-1) > 1e-4)
                            _rpy_mask_i = _valid_rpy & (_rpy_frames[:, _hist_i].norm(dim=-1) > 1e-4)
                            if _pos_mask_i.any():
                                _pos_cos_vals.append(torch.nn.functional.cosine_similarity(
                                    _pos_frames[_pos_mask_i, _hist_i],
                                    _target_pos[_pos_mask_i],
                                    dim=-1,
                                ).mean())
                            if _rpy_mask_i.any():
                                _obs_rpy_i = _rpy_frames[_rpy_mask_i, _hist_i]
                                _target_rpy_i = _target_rpy[_rpy_mask_i]
                                _rpy_cos_vals.append(torch.nn.functional.cosine_similarity(
                                    _obs_rpy_i,
                                    _target_rpy_i,
                                    dim=-1,
                                ).mean())
                                _rpy_neg_cos_vals.append(torch.nn.functional.cosine_similarity(
                                    -_obs_rpy_i,
                                    _target_rpy_i,
                                    dim=-1,
                                ).mean())
                                _rpy_norm_vals.append(_obs_rpy_i.norm(dim=-1).mean())
                            _z_mask_i = _target_pos[:, 2].abs() > 1e-4
                            _roll_mask_i = _target_rpy[:, 0].abs() > 1e-4
                            _pitch_mask_i = _target_rpy[:, 1].abs() > 1e-4
                            _z_sign_vals.append(_scalar_sign(
                                _pos_frames[:, _hist_i, 2], _target_pos[:, 2], _z_mask_i))
                            _roll_sign_vals.append(_scalar_sign(
                                _rpy_frames[:, _hist_i, 0], _target_rpy[:, 0], _roll_mask_i))
                            _pitch_sign_vals.append(_scalar_sign(
                                _rpy_frames[:, _hist_i, 1], _target_rpy[:, 1], _pitch_mask_i))
                            _z_corr_vals.append(_scalar_corr(
                                _pos_frames[:, _hist_i, 2], _target_pos[:, 2], _z_mask_i))
                            _roll_corr_vals.append(_scalar_corr(
                                _rpy_frames[:, _hist_i, 0], _target_rpy[:, 0], _roll_mask_i))
                            _pitch_corr_vals.append(_scalar_corr(
                                _rpy_frames[:, _hist_i, 1], _target_rpy[:, 1], _pitch_mask_i))
                        _pos_cos = torch.stack(_pos_cos_vals).max() if _pos_cos_vals else torch.tensor(0.0, device=self.device)
                        _rpy_cos = torch.stack(_rpy_cos_vals).max() if _rpy_cos_vals else torch.tensor(0.0, device=self.device)
                        _rpy_neg_cos = (
                            torch.stack(_rpy_neg_cos_vals).max()
                            if _rpy_neg_cos_vals else torch.tensor(0.0, device=self.device)
                        )
                        _rpy_norm = (
                            torch.stack(_rpy_norm_vals).max()
                            if _rpy_norm_vals else torch.tensor(0.0, device=self.device)
                        )
                        _z_sign = torch.stack(_z_sign_vals).max()
                        _roll_sign = torch.stack(_roll_sign_vals).max()
                        _pitch_sign = torch.stack(_pitch_sign_vals).max()
                        _z_corr = torch.stack(_z_corr_vals).max()
                        _roll_corr = torch.stack(_roll_corr_vals).max()
                        _pitch_corr = torch.stack(_pitch_corr_vals).max()
                        return (
                            _pos_cos, _rpy_cos, _rpy_neg_cos, _rpy_norm,
                            _z_sign, _roll_sign, _pitch_sign,
                            _z_corr, _roll_corr, _pitch_corr,
                        )

                    _frame_extra = _extra.reshape(_all_obs.shape[0], 5, 6)
                    _frame_scores = _score_extra_layout(
                        _frame_extra[:, :, :3],
                        _frame_extra[:, :, 3:],
                    )
                    _term_scores = _score_extra_layout(
                        _extra[:, :15].reshape(_all_obs.shape[0], 5, 3),
                        _extra[:, 15:30].reshape(_all_obs.shape[0], 5, 3),
                    )
                    _best_scores = _frame_scores
                    if _term_scores[0] > _frame_scores[0]:
                        _best_scores = _term_scores
                    _obs_pos_best_cos = _best_scores[0].item()
                    _obs_rpy_best_cos = _best_scores[1].item()
                    _obs_rpy_best_neg_cos = _best_scores[2].item()
                    _obs_rpy_best_norm = _best_scores[3].item()
                    _obs_z_best_sign = _best_scores[4].item()
                    _obs_roll_best_sign = _best_scores[5].item()
                    _obs_pitch_best_sign = _best_scores[6].item()
                    _obs_z_best_corr = _best_scores[7].item()
                    _obs_roll_best_corr = _best_scores[8].item()
                    _obs_pitch_best_corr = _best_scores[9].item()
                _energy_pred_all = self.alg.policy.evaluate(_all_critic_obs)
                _energy_loss_all = torch.nn.functional.huber_loss(
                    _energy_pred_all, _all_energy, reduction="mean").item()
                _energy_mae = (_energy_pred_all - _all_energy).abs().mean().item()
                _energy_target_mean = _all_energy.mean().item()
                _energy_pred_mean = _energy_pred_all.mean().item()
                _energy_target_std = _all_energy.std(unbiased=False).item()
                _energy_pred_std = _energy_pred_all.std(unbiased=False).item()
                _energy_cov = (
                    (_energy_pred_all - _energy_pred_all.mean())
                    * (_all_energy - _all_energy.mean())
                ).mean()
                _energy_corr = (
                    _energy_cov
                    / (_energy_pred_all.std(unbiased=False) * _all_energy.std(unbiased=False)).clamp(min=1e-6)
                ).item()
                _safe_gap_diag = float(self.cfg.get("frontres_safe_gap_per_step", 0.003))
                _broken_gap_diag = float(self.cfg.get("frontres_broken_gap_per_step", 0.08))
                _energy_damage_frac = (_all_energy.view(-1) > _safe_gap_diag).float().mean().item()
                _energy_broken_frac = (_all_energy.view(-1) > _broken_gap_diag).float().mean().item()
            print(f"[Runner]   warmup {_wu + 1}/{warmup_iters}: "
                  f"dr_scale={_warmup_dr_scale:.3f}, "
                  f"mode_mix={tuple(_warmup_mode_groups)}, "
                  f"loss={loss.item():.6f}, actor={_last_actor_loss.item():.6f}, "
                  f"energy={_last_energy_loss.item():.6f}, cos={_warmup_cos:.4f}, "
                  f"valid={_valid_frac:.3f}",
                  flush=True)
            print(f"[Runner]      diag: "
                  f"cos_pos={_cos_pos:+.4f}, cos_rpy={_cos_rpy:+.4f}, "
                  f"valid_pos={_valid_pos_frac:.3f}, valid_rpy={_valid_rpy_frac:.3f}",
                  flush=True)
            print(f"[Runner]      diag_valid_axes: "
                  f"x/y/z={_valid_x_frac:.3f}/{_valid_y_frac:.3f}/{_valid_z_frac:.3f}, "
                  f"r/p/yaw={_valid_roll_frac:.3f}/{_valid_pitch_frac:.3f}/{_valid_yaw_frac:.3f}",
                  flush=True)
            print(f"[Runner]      diag: "
                  f"mae_pos={_mae_pos:.5f}m, mae_rpy={_mae_rpy:.5f}rad, "
                  f"|pred_pos|/|tgt_pos|={_pred_pos_norm:.5f}/{_tgt_pos_norm:.5f}, "
                  f"|pred_rpy|/|tgt_rpy|={_pred_rpy_norm:.5f}/{_tgt_rpy_norm:.5f}",
                  flush=True)
            print(f"[Runner]      diag_rpy: "
                  f"sign_r/p/y={_sign_roll:.3f}/{_sign_pitch:.3f}/{_sign_yaw:.3f}, "
                  f"valid_r/p/y={_valid_roll_frac:.3f}/{_valid_pitch_frac:.3f}/{_valid_yaw_frac:.3f}, "
                  f"|pred_r/p/y|={_abs_pred_roll:.5f}/{_abs_pred_pitch:.5f}/{_abs_pred_yaw:.5f}, "
                  f"|tgt_r/p/y|={_abs_tgt_roll:.5f}/{_abs_tgt_pitch:.5f}/{_abs_tgt_yaw:.5f}",
                  flush=True)
            print(f"[Runner]      diag_obs_target: "
                  f"best_obs_pos_cos={_obs_pos_best_cos:+.4f}, "
                  f"best_obs_rpy_cos={_obs_rpy_best_cos:+.4f}, "
                  f"best_neg_obs_rpy_cos={_obs_rpy_best_neg_cos:+.4f}, "
                  f"best_obs_rpy_norm={_obs_rpy_best_norm:.5f}",
                  flush=True)
            print(f"[Runner]      diag_obs_target_axis: "
                  f"sign_z/r/p={_obs_z_best_sign:.3f}/{_obs_roll_best_sign:.3f}/{_obs_pitch_best_sign:.3f}, "
                  f"corr_z/r/p={_obs_z_best_corr:+.3f}/{_obs_roll_best_corr:+.3f}/{_obs_pitch_best_corr:+.3f}",
                  flush=True)
            print(f"[Runner]      energy: "
                  f"loss={_energy_loss_all:.6f}, mae={_energy_mae:.6f}, "
                  f"pred/target={_energy_pred_mean:.6f}/{_energy_target_mean:.6f}",
                  flush=True)
            print(f"[Runner]      energy: "
                  f"corr={_energy_corr:+.4f}, std_pred/target={_energy_pred_std:.6f}/{_energy_target_std:.6f}, "
                  f"damage_frac={_energy_damage_frac:.3f}, broken_frac={_energy_broken_frac:.3f}",
                  flush=True)

    print(f"[Runner] === Joint warmup complete (final loss={loss.item():.6f}) ===",
          flush=True)
    if self.log_dir is not None:
        self._frontres_warmup_complete = True
        self._dr_scale = dr_scale
        warmup_path = os.path.join(self.log_dir, "model_warmup.pt")
        self.save(warmup_path)
        print(f"[Runner] Warmup checkpoint saved to {warmup_path}", flush=True)
    _apply_frontres_dr_scale(dr_scale)
