"""Per-step metrics: ZMP lateral margin and fallen detection."""
from __future__ import annotations
import torch


# G1 foot half-width added to the stance to form the support polygon boundary.
_FOOT_HALF_WIDTH_M = 0.05   # metres  (G1 foot ~0.10 m wide)
_FOOT_HALF_LENGTH_M = 0.10  # metres  (G1 foot ~0.20 m long)
_FALLEN_PELVIS_HEIGHT_M = 0.35   # pelvis below this → fallen
_CONTACT_FORCE_THRESHOLD_N = 5.0  # minimum upward force to count as "in contact"


def compute_zmp_margin(
    env_unwrapped,
    left_foot_body_idx: int,
    right_foot_body_idx: int,
) -> torch.Tensor:
    """
    Compute the lateral ZMP margin for all envs: (num_envs,) tensor in metres.

    Positive = ZMP inside support polygon (stable).
    Negative = ZMP outside (about to fall).

    Algorithm
    ---------
    1. Read net contact forces at left/right ankle bodies.
    2. Compute ZMP_y = Σ(F_z_i * y_i) / Σ F_z_i   (lateral component).
    3. Support polygon lateral half-width = ankle_separation / 2 + foot_half_width.
    4. Margin = half_width − |ZMP_y − stance_center_y|.
    """
    robot = env_unwrapped.scene["robot"]
    sensor = env_unwrapped.scene.sensors["contact_forces"]

    # Contact forces: (num_envs, num_bodies, 3)  — world frame
    forces_w = sensor.data.net_forces_w  # (E, B, 3)
    left_fz  = forces_w[:, left_foot_body_idx,  2].clamp(min=0.0)  # (E,)
    right_fz = forces_w[:, right_foot_body_idx, 2].clamp(min=0.0)  # (E,)
    total_fz = left_fz + right_fz + 1e-6  # (E,) avoid division by zero

    # Foot Y positions in world frame: (num_envs, 3) → take Y
    body_pos = robot.data.body_pos_w  # (E, B, 3)
    left_y  = body_pos[:, left_foot_body_idx,  1]  # (E,)
    right_y = body_pos[:, right_foot_body_idx, 1]  # (E,)

    # ZMP lateral coordinate (weighted by vertical force)
    zmp_y = (left_fz * left_y + right_fz * right_y) / total_fz  # (E,)

    # Stance center and half-width of support polygon
    center_y   = (left_y + right_y) * 0.5                              # (E,)
    half_width = (left_y - right_y).abs() * 0.5 + _FOOT_HALF_WIDTH_M   # (E,)

    margin = half_width - (zmp_y - center_y).abs()  # (E,)
    return margin


def compute_zmp_margin_sagittal(
    env_unwrapped,
    left_foot_body_idx: int,
    right_foot_body_idx: int,
) -> torch.Tensor:
    """Sagittal (forward/backward) ZMP margin — identical logic, X axis."""
    robot = env_unwrapped.scene["robot"]
    sensor = env_unwrapped.scene.sensors["contact_forces"]

    forces_w = sensor.data.net_forces_w
    left_fz  = forces_w[:, left_foot_body_idx,  2].clamp(min=0.0)
    right_fz = forces_w[:, right_foot_body_idx, 2].clamp(min=0.0)
    total_fz = left_fz + right_fz + 1e-6

    body_pos = robot.data.body_pos_w
    left_x  = body_pos[:, left_foot_body_idx,  0]
    right_x = body_pos[:, right_foot_body_idx, 0]

    zmp_x   = (left_fz * left_x + right_fz * right_x) / total_fz
    center_x = (left_x + right_x) * 0.5
    half_len = _FOOT_HALF_LENGTH_M

    return half_len - (zmp_x - center_x).abs()


def is_fallen(env_unwrapped) -> torch.Tensor:
    """
    Returns bool tensor (num_envs,): True if env is considered fallen.

    Fallen condition: pelvis height < _FALLEN_PELVIS_HEIGHT_M.
    This detects collapse before the env's own termination triggers a reset,
    complementing the `dones` returned by env.step().
    """
    robot = env_unwrapped.scene["robot"]
    pelvis_z = robot.data.root_pos_w[:, 2]  # (E,)
    return pelvis_z < _FALLEN_PELVIS_HEIGHT_M


def find_body_index(env_unwrapped, body_name: str) -> int:
    """Return integer index of a named body in the robot's body list."""
    robot = env_unwrapped.scene["robot"]
    indices, _ = robot.find_bodies([body_name])
    if len(indices) == 0:
        raise ValueError(f"Body '{body_name}' not found in robot.")
    return int(indices[0])
