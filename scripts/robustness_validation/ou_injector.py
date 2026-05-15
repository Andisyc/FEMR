"""Reference-frame perturbation injection — wraps existing MotionPerturber."""
from __future__ import annotations
import math
import torch


PERTURBATION_MODES = ("composite", "xy", "yaw", "z", "rp")


def configure_ou(env_unwrapped, epsilon: float, tau: float = 0.5) -> None:
    """Backward-compatible composite OU/IID perturbation entry point."""
    configure_perturbation(env_unwrapped, epsilon, mode="composite", tau=tau)


def configure_perturbation(
    env_unwrapped,
    epsilon: float,
    mode: str = "composite",
    tau: float = 0.5,
    enable_ou: bool = True,
    enable_iid: bool = True,
) -> None:
    """
    Configure reference-frame perturbations by semantic channel.

    The validation uses a composite condition for the main paper figures and
    individual channels for appendix diagnosis:

      composite: XY + yaw + Z + roll/pitch
      xy:        horizontal root translation
      yaw:       heading jump
      z:         float/sink
      rp:        roll/pitch tilt

    ``epsilon`` is the scalar sweep variable.  Translation channels use metres;
    rotational channels use conservative radian scales tied to epsilon.
    epsilon = 0 disables all perturbations.
    """
    if mode not in PERTURBATION_MODES:
        raise ValueError(f"Unknown perturbation mode {mode!r}; expected one of {PERTURBATION_MODES}")

    cmd = _get_motion_cmd(env_unwrapped)
    p = cmd.perturber
    cfg = p.cfg

    _disable_all(cfg)

    if epsilon <= 0.0:
        _update_tau(p, cfg, tau)
        return

    channels = {"xy", "yaw", "z", "rp"} if mode == "composite" else {mode}

    if enable_ou:
        if "z" in channels:
            cfg.float_prob = 1.0
            cfg.sink_prob = 1.0
            cfg.float_ratio = epsilon
            cfg.sink_ratio = epsilon

        if "xy" in channels:
            cfg.lateral_drift_prob = 1.0
            cfg.lateral_drift_std = epsilon * 0.4

        if "rp" in channels:
            cfg.root_tilt_prob = 1.0
            cfg.root_tilt_max_rad = epsilon * 5.0

    if enable_iid:
        if "z" in channels:
            cfg.iid_prob_z = 0.3
            cfg.iid_std_z = epsilon
        if "xy" in channels:
            cfg.iid_prob_xy = 0.1
            cfg.iid_std_xy = epsilon * 0.4
        if "rp" in channels:
            cfg.iid_prob_rp = 0.1
            cfg.iid_std_rp = epsilon * 5.0
        if "yaw" in channels:
            cfg.iid_prob_ya = 0.1
            cfg.iid_std_ya = epsilon * 5.0

    # Foot-slip and joint noise are disabled to keep the reference-frame effect isolated.
    cfg.foot_slip_prob = 0.0
    cfg.joint_noise_prob = 0.0
    _update_tau(p, cfg, tau)


def _disable_all(cfg) -> None:
    cfg.float_prob = 0.0
    cfg.sink_prob = 0.0
    cfg.lateral_drift_prob = 0.0
    cfg.root_tilt_prob = 0.0
    cfg.foot_slip_prob = 0.0
    cfg.joint_noise_prob = 0.0

    for name in ("iid_prob_z", "iid_prob_xy", "iid_prob_rp", "iid_prob_ya"):
        if hasattr(cfg, name):
            setattr(cfg, name, 0.0)


def _update_tau(p, cfg, tau: float) -> None:
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
