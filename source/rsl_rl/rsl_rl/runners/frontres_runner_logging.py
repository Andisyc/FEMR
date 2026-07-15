"""Runner logging for the active MOSAIC and FrontRES paths.

This module owns scalar emission and console formatting only.  It deliberately
does not interpret retired FrontRES objectives or create compatibility routes
for them.
"""

from __future__ import annotations

import statistics
import time
from numbers import Real

import torch

from rsl_rl.modules import FrontRESActorCritic


_SUPERVISED_KEYS = {
    "supervised_cos_sim",
    "supervised_mae",
    "supervised_rmse",
    "supervised_rpy_mae",
    "supervised_rpy_rmse",
    "supervised_restore_ratio",
    "supervised_valid_frac",
    "supervised_l_pos",
    "supervised_l_rot",
    "supervised_l_mag",
    "supervised_l_over",
    "supervised_l_smooth",
    "supervised_l_harm",
    "frontres_supervised_weight",
    "frontres_write_ratio",
    "frontres_proposal_ratio",
    "frontres_axis_leakage",
}

_CURRICULUM_KEYS = {
    "lambda_supervised",
}

_FRONTRES_CONSOLE_KEYS = (
    "frontres_rdelta_mean",
    "frontres_positive_gain_frac_mean",
    "frontres_harm_rate_mean",
    "frontres_damage_gap_mean",
    "frontres_repair_gain_mean",
    "frontres_repair_ratio_mean",
    "frontres_exec_signal_mean",
    "frontres_weighted_exec_signal_mean",
    "frontres_train_reward_mean",
    "frontres_safe_frac_mean",
    "frontres_repair_frac_mean",
    "frontres_broken_frac_mean",
    "frontres_delta_pos_abs_mean",
    "frontres_delta_rpy_abs_mean",
    "frontres_perturb_complexity",
    "frontres_perturb_modes",
    "frontres_dr_scale",
)

_FRONTRES_ACTIVE_KEYS = frozenset(_FRONTRES_CONSOLE_KEYS) | {"frontres_survival_rate"}


def _scalar_log_value(value):
    """Return a TensorBoard-safe scalar, or None for text/containers."""

    if isinstance(value, torch.Tensor):
        detached = value.detach()
        if detached.numel() != 1:
            return None
        return detached.item()
    if isinstance(value, Real):
        return float(value)
    return None


def _mean_buffer(values) -> float | None:
    if not values:
        return None
    return float(statistics.mean(values))


def _frontres_scalar_name(key: str) -> str:
    return key.removeprefix("frontres_")


def _log_frontres_scalars(writer, locs: dict, iteration: int) -> None:
    """Emit current full-6D/Gain diagnostics and ignore retired namespaces."""

    for key, value in locs.items():
        if key not in _FRONTRES_ACTIVE_KEYS:
            continue
        scalar = _scalar_log_value(value)
        if scalar is not None:
            writer.add_scalar(f"FrontRES/{_frontres_scalar_name(key)}", scalar, iteration)


def _log_loss_scalars(writer, self, locs: dict, iteration: int) -> None:
    loss_dict = locs.get("loss_dict", {})
    for key, value in loss_dict.items():
        scalar = _scalar_log_value(value)
        if scalar is None:
            continue
        if key in {"bc_off_policy", "bc_teacher", "lambda_off_policy", "lambda_teacher"} and scalar == 0.0:
            continue
        if key in _SUPERVISED_KEYS:
            namespace = "FrontRES"
        elif key in _CURRICULUM_KEYS:
            namespace = "Curriculum"
        else:
            namespace = "Loss"
        writer.add_scalar(f"{namespace}/{key}", scalar, iteration)
    writer.add_scalar("Loss/learning_rate", float(getattr(self.alg, "learning_rate", 0.0)), iteration)


def _phase_label(self, locs: dict) -> tuple[str, str]:
    if not isinstance(self.alg.policy, FrontRESActorCritic):
        return "PPO", ""
    objective = str(getattr(self.alg, "frontres_training_objective", "")).lower()
    if objective == "supervised_restore":
        return "SUPERVISED RESTORE", "(fitting the full-6D clean restoration target)"
    if locs.get("_supervised_warmup_active", False):
        return "SUPERVISED WARMUP", "(GMT-only, FrontRES corrections disabled)"
    if locs.get("_critic_warmup_active", False):
        return "CRITIC WARMUP", "(critic and supervised terms active; actor frozen)"
    return "FRONTRES HSL", "(full-6D supervised proposal update)"


def log_runner(self, locs: dict, width: int = 80, pad: int = 35):
    """Log one training iteration without changing training state."""

    collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size
    collection_time = float(locs["collection_time"])
    learn_time = float(locs["learn_time"])
    iteration_time = collection_time + learn_time
    self.tot_timesteps += collection_size
    self.tot_time += iteration_time
    iteration = locs["it"]

    ep_string = ""
    if locs["ep_infos"]:
        for key in locs["ep_infos"][0]:
            values = []
            for ep_info in locs["ep_infos"]:
                if key not in ep_info:
                    continue
                value = ep_info[key]
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor([value], device=self.device)
                value = value.reshape(-1).to(self.device)
                values.append(value)
            if not values:
                continue
            mean_value = torch.cat(values).mean()
            log_key = key if "/" in key else ("GMT/" if self.training_type == "supervise" else "Episode/") + key
            if self.training_type == "supervise" and any(token in key.lower() for token in ("rew", "reward")):
                continue
            self.writer.add_scalar(log_key, mean_value, iteration)
            label = key if "/" in key else ("GMT " if self.training_type == "supervise" else "Mean episode ") + key
            ep_string += f"{label + ':':>{pad}} {mean_value.item():.4f}\n"

    if self.training_type == "supervise" and locs["lenbuffer"]:
        self.writer.add_scalar("GMT/mean_episode_length", _mean_buffer(locs["lenbuffer"]), iteration)

    mean_std = float(self.alg.policy.action_std.mean().detach().item())
    fps = int(collection_size / max(iteration_time, 1e-8))
    _log_loss_scalars(self.writer, self, locs, iteration)
    if self.training_type != "supervise":
        self.writer.add_scalar("Policy/mean_noise_std", mean_std, iteration)

    if isinstance(self.alg.policy, FrontRESActorCritic):
        _log_frontres_scalars(self.writer, locs, iteration)
        self.writer.add_scalar("FrontRES/r_delta_ema", float(locs.get("_r_delta_ema", 0.0)), iteration)

    self.writer.add_scalar("Perf/total_fps", fps, iteration)
    self.writer.add_scalar("Perf/collection_time", collection_time, iteration)
    self.writer.add_scalar("Perf/learning_time", learn_time, iteration)

    if self.training_type != "supervise" and locs["rewbuffer"]:
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.writer.add_scalar("Rnd/mean_extrinsic_reward", _mean_buffer(locs["erewbuffer"]), iteration)
            self.writer.add_scalar("Rnd/mean_intrinsic_reward", _mean_buffer(locs["irewbuffer"]), iteration)
            self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, iteration)
        if isinstance(self.alg.policy, FrontRESActorCritic):
            self.writer.add_scalar("Train/mean_r_delta", _mean_buffer(locs["rewbuffer"]), iteration)
            if locs.get("rewbuffer_gmt"):
                self.writer.add_scalar("Train/mean_reward_gmt", _mean_buffer(locs["rewbuffer_gmt"]), iteration)
            if locs.get("lenbuffer_gmt"):
                self.writer.add_scalar("Train/mean_episode_length_gmt", _mean_buffer(locs["lenbuffer_gmt"]), iteration)
        else:
            self.writer.add_scalar("Train/mean_reward", _mean_buffer(locs["rewbuffer"]), iteration)
        self.writer.add_scalar("Train/mean_episode_length", _mean_buffer(locs["lenbuffer"]), iteration)
        if self.logger_type != "wandb":
            self.writer.add_scalar("Train/mean_reward/time", _mean_buffer(locs["rewbuffer"]), self.tot_time)
            self.writer.add_scalar("Train/mean_episode_length/time", _mean_buffer(locs["lenbuffer"]), self.tot_time)

    iter_title = f" \033[1m Learning iteration {iteration}/{locs['tot_iter']} \033[0m "
    if bool(getattr(self.alg, "is_frontres_unified", False)):
        resume_label = "full-resume" if bool(self.cfg.get("is_full_resume", True)) else "checkpoint-init"
        iter_title = f" \033[1m Learning iteration {iteration}/{locs['tot_iter']} [{resume_label}] \033[0m "
    phase, notes = _phase_label(self, locs)
    phase_line = f"  PHASE: {phase}  " + (f"\n  {notes}  " if notes else "")

    log_string = f"{'#' * width}\n{iter_title.center(width, ' ')}\n{phase_line.center(width, ' ')}\n"
    log_string += f"\n{'-' * 12} Performance {'-' * 12}\n"
    log_string += f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning {learn_time:.3f}s)\n"
    if self.training_type != "supervise":
        log_string += f"{'Mean action noise std:':>{pad}} {mean_std:.6f}\n"
    if locs.get("lenbuffer"):
        log_string += f"{'episode length:':>{pad}} {_mean_buffer(locs['lenbuffer']):.1f}\n"
    if locs.get("lenbuffer_gmt"):
        log_string += f"{'GMT episode length:':>{pad}} {_mean_buffer(locs['lenbuffer_gmt']):.1f}\n"
    if ep_string:
        log_string += ep_string

    if isinstance(self.alg.policy, FrontRESActorCritic):
        log_string += f"\n{'-' * 12} FrontRES Gain / Geometry {'-' * 12}\n"
        for key in _FRONTRES_CONSOLE_KEYS:
            value = locs.get(key)
            scalar = _scalar_log_value(value)
            if scalar is not None:
                label = key.removeprefix("frontres_").replace("_mean", "")
                log_string += f"{label + ':':>{pad}} {scalar:+.6f}\n"
        loss_dict = locs.get("loss_dict", {})
        for key in ("supervised_cos_sim", "supervised_restore_ratio", "supervised_valid_frac", "lambda_supervised"):
            scalar = _scalar_log_value(loss_dict.get(key))
            if scalar is not None:
                log_string += f"{key + ':':>{pad}} {scalar:+.6f}\n"
    elif self.training_type == "supervise":
        log_string += f"\n{'-' * 30} STAGE 1 {'-' * 33}\n"
        if "behavior" in locs["loss_dict"]:
            log_string += f"{'behavior loss:':>{pad}} {locs['loss_dict']['behavior']:.4f}\n"
    else:
        log_string += f"{'Mean action noise std:':>{pad}} {mean_std:.6f}\n"
        for key, value in locs["loss_dict"].items():
            scalar = _scalar_log_value(value)
            if scalar is not None:
                log_string += f"{key + ':':>{pad}} {scalar:.4f}\n"

    footer_width = 44 if self.training_type == "frontres" else width
    elapsed = time.strftime("%H:%M:%S", time.gmtime(self.tot_time))
    remaining = locs["start_iter"] + locs["num_learning_iterations"] - iteration
    eta_seconds = self.tot_time / max(1, iteration - locs["start_iter"] + 1) * remaining
    eta = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
    log_string += (
        f"{'-' * footer_width}\n"
        f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
        f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
        f"{'Time elapsed:':>{pad}} {elapsed}\n"
        f"{'ETA:':>{pad}} {eta}\n"
    )
    print(log_string)
