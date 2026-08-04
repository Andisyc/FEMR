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


def prepare_frontres_hsl_actor_observation(runner: Any, raw_obs: torch.Tensor) -> torch.Tensor:
    """Build the sealed-q29 Stage-1 actor input before its normalizer consumes it.

    Status: active Stage-1-only route. This helper has no raw-observation
    fallback: the v015 bridge owns q29 provenance and reads only the
    command-owned proposal snapshot on the formal HSL route.
    """

    if not isinstance(raw_obs, torch.Tensor) or raw_obs.ndim != 2:
        raise RuntimeError(f"HSL raw policy observation must be [B,D], got {getattr(raw_obs, 'shape', None)}")
    append_context = getattr(runner, "_append_frontres_future_intent_context", None)
    if not callable(append_context):
        raise RuntimeError("v015 HSL requires the runner future-intent actor-context bridge")
    augmented = append_context(raw_obs)
    if not isinstance(augmented, torch.Tensor) or augmented.ndim != 2:
        raise RuntimeError("v015 HSL actor-context bridge must return [B,D+|H|*29]")
    expected_actor_dim = getattr(getattr(getattr(runner, "alg", None), "policy", None), "num_actor_obs", None)
    if expected_actor_dim is None or int(augmented.shape[-1]) != int(expected_actor_dim):
        raise RuntimeError(
            "v015 HSL actor input dimension disagrees with the q29 policy layout: "
            f"observed={tuple(augmented.shape)} expected_actor_dim={expected_actor_dim}"
        )
    apply_normalizer = getattr(runner, "_apply_obs_normalizer", None)
    if not callable(apply_normalizer):
        raise RuntimeError("v015 HSL requires the runner observation normalizer")
    normalized = apply_normalizer(augmented)
    if not isinstance(normalized, torch.Tensor) or tuple(normalized.shape) != tuple(augmented.shape):
        raise RuntimeError("v015 HSL normalizer must preserve the q29-augmented actor shape")
    if bool(getattr(runner, "_frontres_hsl_live_smoke_enabled", False)):
        policy = runner.alg.policy
        gmt_dim = int(getattr(runner, "_frontres_gmt_obs_dim", 0) or 0)
        prefix_dim = int(getattr(policy, "num_frontres_obs", 0) or 0)
        if (
            tuple(raw_obs.shape)[-1] != 870
            or tuple(augmented.shape)[-1] != 928
            or prefix_dim != 158
            or gmt_dim != 770
        ):
            raise RuntimeError(
                "G2-S4 observation authority drift: "
                f"raw={raw_obs.shape[-1]} combined={augmented.shape[-1]} "
                f"femr={prefix_dim} gmt={gmt_dim}"
            )
        runner._frontres_hsl_smoke_combined_obs = augmented.detach().clone()
        runner._frontres_hsl_smoke_normalized_obs = normalized.detach().clone()
        if not bool(getattr(runner, "_frontres_hsl_smoke_input_emitted", False)):
            snapshot = getattr(runner, "_frontres_hsl_smoke_context_snapshot", None)
            if not isinstance(snapshot, dict):
                raise RuntimeError("G2-S4 input telemetry requires the command-owned proposal snapshot")
            intent = snapshot["intent_q29"]
            artifact_pos = snapshot["artifact_pos"]
            artifact_quat = snapshot["artifact_quat"]
            print(
                "[G2-S4-INPUT] "
                f"artifact_id_head={snapshot['current_root_artifact_ids'][0]} "
                f"context_id_head={snapshot['proposal_context_ids'][0]} "
                f"motion_head={snapshot['motion_indices'][0]} frame_head={snapshot['frame_indices'][0]} "
                f"artifact_pos_head={artifact_pos[0].detach().cpu().tolist()} "
                f"artifact_quat_head={artifact_quat[0].detach().cpu().tolist()} "
                f"q29_shape={tuple(intent.shape)} offsets={snapshot['future_offsets']} "
                f"q29_provenance={snapshot['provenance'][0]['intent_q29_provenance']}",
                flush=True,
            )
            print(
                "[G2-S4-OBS] "
                f"raw={tuple(raw_obs.shape)} combined={tuple(augmented.shape)} "
                f"femr={tuple(normalized[:, :prefix_dim].shape)} "
                f"gmt={tuple(normalized[:, prefix_dim:].shape)}",
                flush=True,
            )
            runner._frontres_hsl_smoke_input_emitted = True
    return normalized


def validate_frontres_hsl_current_frame_target(target: torch.Tensor, command: Any) -> torch.Tensor:
    """Assert that a Stage-1 target is exactly the current anti-DR Delta SE(3)."""

    delta_pos = getattr(command, "anchor_dr_delta_pos", None)
    delta_quat = getattr(command, "anchor_dr_delta_quat_correction", None)
    if (
        not isinstance(target, torch.Tensor)
        or target.ndim != 2
        or target.shape[-1] != 6
        or target.requires_grad
        or not torch.is_floating_point(target)
        or not bool(torch.isfinite(target).all().item())
    ):
        raise RuntimeError("Stage-1 HSL current-frame target must be detached finite [B,6]")
    if (
        not isinstance(delta_pos, torch.Tensor)
        or not isinstance(delta_quat, torch.Tensor)
        or tuple(delta_pos.shape) != (int(target.shape[0]), 3)
        or tuple(delta_quat.shape) != (int(target.shape[0]), 4)
    ):
        raise RuntimeError("Stage-1 HSL target requires current command anti-DR [B,3] and [B,4] fields")
    # B1: 独立重建完整 anti-DR translation, 验证 producer 未隐藏逐轴 mask/clamp.
    expected_pos = -delta_pos.to(device=target.device, dtype=target.dtype)
    quat = delta_quat.to(device=target.device, dtype=target.dtype)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    expected_rpy = torch.stack(
        [
            torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0)),
            torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        ],
        dim=-1,
    )
    expected = torch.cat([expected_pos, expected_rpy], dim=-1)
    if not torch.allclose(target, expected, rtol=1.0e-5, atol=1.0e-6):
        max_error = float((target - expected).abs().max().item())
        raise RuntimeError(
            "Stage-1 HSL target is not the current anti-DR Delta SE(3): "
            f"max_abs_error={max_error:.6g}"
        )
    return target


def capture_frontres_hsl_critic_state(policy: Any) -> tuple[torch.Tensor, ...]:
    """Clone critic parameters for the proposal-only HSL invariance guard."""

    critic = getattr(policy, "critic", None)
    if not isinstance(critic, torch.nn.Module):
        raise RuntimeError("proposal-only HSL requires an explicit critic invariance boundary")
    return tuple(value.detach().clone() for value in critic.parameters())


def assert_frontres_hsl_critic_unchanged(
    policy: Any,
    before: tuple[torch.Tensor, ...],
) -> None:
    """Fail closed if proposal initialization mutates any critic parameter."""

    after = tuple(value.detach() for value in policy.critic.parameters())
    if len(after) != len(before) or any(not torch.equal(value, old) for value, old in zip(after, before)):
        raise RuntimeError("proposal-only HSL critic changed during actor initialization")


def frontres_hsl_critic_grad_count(policy: Any) -> int:
    """Return the number of critic parameters carrying a gradient."""

    return sum(int(isinstance(value.grad, torch.Tensor)) for value in policy.critic.parameters())


def frontres_hsl_critic_max_abs_delta(
    policy: Any,
    before: tuple[torch.Tensor, ...],
) -> float:
    """Return the exact maximum critic parameter delta for S4 telemetry."""

    after = tuple(value.detach() for value in policy.critic.parameters())
    return max(
        (float((value - old).abs().max().item()) for value, old in zip(after, before)),
        default=0.0,
    )


def build_frontres_hsl_actor_only_optimizer(
    policy: Any,
    *,
    learning_rate: float,
) -> torch.optim.Optimizer:
    """Build the only optimizer allowed by proposal-only Stage-1 HSL."""

    actor = getattr(policy, "residual_actor", None)
    if not isinstance(actor, torch.nn.Module):
        raise RuntimeError("proposal-only HSL requires the residual actor owner")
    actor_params = tuple(actor.parameters())
    critic_params = tuple(getattr(policy, "critic").parameters())
    if not actor_params or {id(value) for value in actor_params} & {id(value) for value in critic_params}:
        raise RuntimeError("proposal-only HSL actor and critic parameter ownership must be disjoint")
    return torch.optim.Adam(actor_params, lr=float(learning_rate))


def _require_direct_hsl_proposal(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 2 or int(value.shape[-1]) != 6:
        raise ValueError(f"proposal-only HSL actor must emit exact direct [B,6], got {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("proposal-only HSL actor emitted non-finite direct 6D values")
    return value


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
    """Run proposal-only actor initialization from the current anti-DR target."""
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
    if not bool(getattr(self, "_frontres_hsl_proposal_context_enabled", False)):
        raise RuntimeError("FRS-TRAIN-v007 requires the formal Stage-1 HSL proposal route")
    if float(self.cfg.get("frontres_warmup_energy_loss_weight", 0.0)) != 0.0:
        raise RuntimeError("proposal-only Stage-1 HSL forbids executable-energy critic loss")
    _warmup_diag_interval = int(self.cfg.get(
        "supervised_warmup_diag_interval", max(1, warmup_iters // 5)))
    _warmup_diag_interval = max(1, _warmup_diag_interval)
    _critic_state_before = capture_frontres_hsl_critic_state(self.alg.policy)
    _warmup_opt = build_frontres_hsl_actor_only_optimizer(
        self.alg.policy,
        learning_rate=_warmup_lr,
    )
    _live_smoke = bool(getattr(self, "_frontres_hsl_live_smoke_enabled", False))
    _fresh_reload_shadow = None
    if _live_smoke:
        capture_shadow = getattr(self, "_capture_frontres_hsl_fresh_reload_shadow", None)
        if not callable(capture_shadow):
            raise RuntimeError("G2-S4 requires the strict HSL fresh-reload shadow owner")
        _fresh_reload_shadow = capture_shadow()

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
    print(f"[Runner] === Proposal-only HSL warmup: {warmup_iters} iters "
          f"(dr_scale={_warmup_dr_desc}, lr={_warmup_lr}, epochs={_warmup_epochs}, "
          f"steps_per_iter={_warmup_steps}, "
          f"max_envs_per_step={_warmup_max_envs}, "
          f"frontres_input={_nfo} dims, critic_update=false, "
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

        # Use no_grad rather than inference_mode: warmup samples are later fed
        # back through the trainable residual actor.
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
                _p_obs = prepare_frontres_hsl_actor_observation(self, _p_obs_raw)

                env_actions = self.alg.policy.get_env_action(
                    _p_obs,
                    torch.zeros(_p_obs.shape[0], self.alg.policy.total_output_dim, device=self.device),
                )

                obs, rewards_wu, dones, extras = self.env.step(env_actions.to(self.env.device))
                obs_dict = extras.get("observations", {})
                _p_obs_raw = obs_dict.get(self.policy_obs_type, obs).to(self.device)
                _p_obs = prepare_frontres_hsl_actor_observation(self, _p_obs_raw)
                _mcmd_wu = _env_raw.command_manager._terms.get("motion")
                if _mcmd_wu is None:
                    raise RuntimeError("Stage-1 HSL requires the current motion command anti-DR owner")
                _target = validate_frontres_hsl_current_frame_target(
                    _get_warmup_target(_env_raw, "motion").to(self.device),
                    _mcmd_wu,
                )
                _raw_target = _target.detach().clone()
                if _live_smoke and not bool(getattr(self, "_frontres_hsl_smoke_target_emitted", False)):
                    print(
                        "[G2-S4-TARGET] owner=current_antidr_delta_se3 "
                        f"raw_shape={tuple(_raw_target.shape)} applied_shape={tuple(_target.shape)} "
                        f"raw_head={_raw_target[0].detach().cpu().tolist()} "
                        f"applied_head={_target[0].detach().cpu().tolist()} finite=1",
                        flush=True,
                    )
                    self._frontres_hsl_smoke_target_emitted = True

                if _warmup_max_envs < _p_obs.shape[0]:
                    _sample_ids = torch.randperm(_p_obs.shape[0], device=self.device)[:_warmup_max_envs]
                    _p_obs = _p_obs[_sample_ids]
                    _target = _target[_sample_ids]

                _wo_list.append(_p_obs[:, :_nfo])
                _wt_list.append(_target)

        _all_obs = torch.cat(_wo_list, dim=0)
        _all_tgt = torch.cat(_wt_list, dim=0)
        _N = _all_obs.shape[0]
        _last_actor_loss = torch.tensor(0.0, device=self.device)
        for epoch in range(_warmup_epochs):
            perm = torch.randperm(_N, device=self.device)
            for i in range(0, _N, 4096):
                idx = perm[i:i + 4096]
                pred = self.alg.policy.residual_actor(_all_obs[idx])
                if getattr(self.alg.policy, "num_task_corrections", 0) > 0:
                    pred_sup = _require_direct_hsl_proposal(pred)
                    target_sup = _all_tgt[idx, :6]
                else:
                    pred_sup = pred[:, :_all_tgt.shape[-1]]
                    target_sup = _all_tgt[idx]
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
                actor_loss = loss
                _warmup_opt.zero_grad()
                loss.backward()
                if _live_smoke and not bool(getattr(self, "_frontres_hsl_smoke_grad_emitted", False)):
                    actor_grads = [
                        value.grad.detach()
                        for value in self.alg.policy.residual_actor.parameters()
                        if isinstance(value.grad, torch.Tensor)
                    ]
                    actor_grad_norm = torch.sqrt(
                        sum((value.float().square().sum() for value in actor_grads), torch.zeros((), device=self.device))
                    )
                    critic_grad_count = frontres_hsl_critic_grad_count(self.alg.policy)
                    if not actor_grads or not bool(torch.isfinite(actor_grad_norm).item()) or actor_grad_norm.item() <= 0.0:
                        raise RuntimeError("G2-S4 requires a finite nonzero residual-actor gradient")
                    if critic_grad_count != 0:
                        raise RuntimeError("G2-S4 proposal-only HSL produced a critic gradient")
                    print(
                        "[G2-S4-GRAD] actor_grad_nonzero=1 "
                        f"actor_grad_norm={actor_grad_norm.item():.9g} critic_grad_count=0 optimizer=actor_only",
                        flush=True,
                    )
                    self._frontres_hsl_smoke_grad_emitted = True
                _warmup_opt.step()
                _last_actor_loss = actor_loss.detach()

        if (_wu + 1) % _warmup_diag_interval == 0 or (_wu + 1) == warmup_iters:
            with torch.inference_mode():
                _pred_all_raw = self.alg.policy.residual_actor(_all_obs[:, :_nfo])
                if getattr(self.alg.policy, "num_task_corrections", 0) > 0:
                    _pred_all = _require_direct_hsl_proposal(_pred_all_raw)
                    _target_all = _all_tgt[:, :6]
                else:
                    _pred_all = _pred_all_raw[:, :_all_tgt.shape[-1]]
                    _target_all = _all_tgt

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
            print(f"[Runner]   warmup {_wu + 1}/{warmup_iters}: "
                  f"dr_scale={_warmup_dr_scale:.3f}, "
                  f"mode_mix={tuple(_warmup_mode_groups)}, "
                  f"loss={loss.item():.6f}, actor={_last_actor_loss.item():.6f}, "
                  f"critic_update=false, cos={_warmup_cos:.4f}, "
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

    assert_frontres_hsl_critic_unchanged(self.alg.policy, _critic_state_before)
    if _live_smoke:
        critic_max_abs_delta = frontres_hsl_critic_max_abs_delta(
            self.alg.policy,
            _critic_state_before,
        )
        if critic_max_abs_delta != 0.0:
            raise RuntimeError("G2-S4 critic parameter delta is nonzero")
        print("[G2-S4-CRITIC] critic_max_abs_delta=0 critic_unchanged=1", flush=True)
    print(f"[Runner] === Proposal-only HSL warmup complete (final loss={loss.item():.6f}) ===",
          flush=True)
    if self.log_dir is not None:
        self._dr_scale = dr_scale
        warmup_path = os.path.join(self.log_dir, "model_warmup.pt")
        self.save(warmup_path)
        self._frontres_warmup_complete = True
        print(f"[Runner] Warmup checkpoint saved to {warmup_path}", flush=True)
        if _live_smoke:
            combined_obs = getattr(self, "_frontres_hsl_smoke_combined_obs", None)
            normalized_obs = getattr(self, "_frontres_hsl_smoke_normalized_obs", None)
            verify_reload = getattr(self, "_verify_frontres_hsl_fresh_reload", None)
            if (
                _fresh_reload_shadow is None
                or not isinstance(combined_obs, torch.Tensor)
                or not isinstance(normalized_obs, torch.Tensor)
                or not callable(verify_reload)
            ):
                raise RuntimeError("G2-S4 fresh reload telemetry is incomplete")
            source_actor_input = normalized_obs[:, :158]
            with torch.inference_mode():
                source_proposal = _require_direct_hsl_proposal(
                    self.alg.policy.residual_actor(source_actor_input)
                )
            verify_reload(
                _fresh_reload_shadow,
                checkpoint_path=warmup_path,
                combined_obs=combined_obs,
                source_actor_input=source_actor_input,
                source_proposal=source_proposal,
            )
            print("[G2-S4-COMPLETE] direct_full6_hsl=1 ppo_entered=0", flush=True)
    _apply_frontres_dr_scale(dr_scale)
