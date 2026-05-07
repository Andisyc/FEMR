"""OU perturbation injection — wraps existing MotionPerturber."""
from __future__ import annotations
import math
import torch


def configure_ou(env_unwrapped, epsilon: float, tau: float = 0.5) -> None:
    """
    Set OU steady-state RMS = epsilon (metres) and time constant tau (seconds).

    Axis ratios vs epsilon:
      Z (float/sink)   : 1.0 × epsilon
      Y (lateral drift): 0.4 × epsilon
      Roll/Pitch tilt  : 5.0 × epsilon  [rad/m] — e.g. 0.05m → 0.25 rad (~14°)

    Foot-slip and joint noise are disabled to keep the root-level effect isolated.
    epsilon = 0 disables all perturbations.
    """
    cmd = _get_motion_cmd(env_unwrapped)
    p = cmd.perturber
    cfg = p.cfg

    if epsilon <= 0.0:
        cfg.float_prob = 0.0
        cfg.sink_prob = 0.0
        cfg.lateral_drift_prob = 0.0
        cfg.root_tilt_prob = 0.0
        cfg.foot_slip_prob = 0.0
        cfg.joint_noise_prob = 0.0
    else:
        cfg.float_prob = 1.0
        cfg.sink_prob = 1.0
        cfg.float_ratio = epsilon
        cfg.sink_ratio = epsilon

        cfg.lateral_drift_prob = 1.0
        cfg.lateral_drift_std = epsilon * 0.4

        cfg.root_tilt_prob = 1.0
        cfg.root_tilt_max_rad = epsilon * 5.0

        cfg.foot_slip_prob = 0.0
        cfg.joint_noise_prob = 0.0

    # Update temporal parameters and recompute beta immediately.
    cfg.ou_time_constant = tau
    p._beta = math.exp(-0.02 / tau)


def reset_ou_states(env_unwrapped, env_ids: torch.Tensor | None = None) -> None:
    """Zero OU states for given envs (all envs if env_ids is None)."""
    cmd = _get_motion_cmd(env_unwrapped)
    p = cmd.perturber
    if env_ids is None:
        env_ids = torch.arange(p.num_envs, device=p.device)
    p.reset_envs(env_ids)


def _get_motion_cmd(env_unwrapped):
    cmd = env_unwrapped.command_manager._terms.get("motion")
    if cmd is None or not hasattr(cmd, "perturber"):
        raise RuntimeError(
            "No MotionPerturber found. Ensure the env uses MultiMotionCommand "
            "and motion_perturbations are configured."
        )
    return cmd
