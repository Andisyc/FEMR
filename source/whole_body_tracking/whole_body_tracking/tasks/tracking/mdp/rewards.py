from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from whole_body_tracking.tasks.tracking.mdp.balance import (
    frontres_balance_context_from_feet,
    frontres_no_regret_balance_reward,
)
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward

# ===== MOSAIC Expert Teleop-style Rewards (World Frame, Fine-grained) =====

def teleop_body_position_extend(
    env: ManagerBasedRLEnv,
    command_name: str,
    upper_body_std: float = 0.05,
    lower_body_std: float = 0.05,
    upper_weight: float = 1.0,
    lower_weight: float = 1.0,
) -> torch.Tensor:
    """
    Upper/lower body position tracking with separate weights (MOSAIC style).
    Tracks body positions in world frame with fine-grained upper/lower body separation.

    Args:
        env: The environment.
        command_name: Name of the motion command.
        upper_body_std: Std (in meters) for upper body exponential reward.
        lower_body_std: Std (in meters) for lower body exponential reward.
        upper_weight: Weight for upper body reward.
        lower_weight: Weight for lower body reward.

    Returns:
        Combined upper + lower body position reward.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    # Upper body names
    upper_body_names = [
        "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
        "left_elbow_link", "right_shoulder_pitch_link", "right_shoulder_roll_link",
        "right_shoulder_yaw_link", "right_elbow_link", "left_hand_link",
        "right_hand_link", "head_link"
    ]

    # Lower body names
    lower_body_names = [
        "pelvis", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
        "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
        "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
        "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
        "waist_yaw_link", "waist_roll_link", "torso_link"
    ]

    upper_idx = _get_body_indexes(command, upper_body_names)
    lower_idx = _get_body_indexes(command, lower_body_names)

    # Compute errors (world frame)
    if len(upper_idx) > 0:
        upper_diff = command.body_pos_w[:, upper_idx, :] - command.robot_body_pos_w[:, upper_idx, :]
        upper_error = (upper_diff ** 2).mean(dim=-1).mean(dim=-1)  # [N]
        r_upper = torch.exp(-upper_error / (upper_body_std ** 2))
    else:
        r_upper = torch.zeros(env.num_envs, device=env.device)

    if len(lower_idx) > 0:
        lower_diff = command.body_pos_w[:, lower_idx, :] - command.robot_body_pos_w[:, lower_idx, :]
        lower_error = (lower_diff ** 2).mean(dim=-1).mean(dim=-1)
        r_lower = torch.exp(-lower_error / (lower_body_std ** 2))
    else:
        r_lower = torch.zeros(env.num_envs, device=env.device)

    return r_lower * lower_weight + r_upper * upper_weight


def teleop_vr_3point(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.04,
) -> torch.Tensor:
    """
    VR 3-point tracking (head + hands) in world frame.

    Args:
        env: The environment.
        command_name: Name of the motion command.
        std: Std (in meters) for exponential reward.

    Returns:
        VR 3-point position tracking reward.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    keypoint_names = ["head_link", "left_hand_link", "right_hand_link"]
    keypoint_idx = _get_body_indexes(command, keypoint_names)

    if len(keypoint_idx) > 0:
        diff = command.body_pos_w[:, keypoint_idx, :] - command.robot_body_pos_w[:, keypoint_idx, :]
        error = (diff ** 2).mean(dim=-1).mean(dim=-1)
        return torch.exp(-error / (std ** 2))
    else:
        return torch.zeros(env.num_envs, device=env.device)


def teleop_body_position_feet(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.03,
) -> torch.Tensor:
    """
    Feet position tracking in world frame (high precision).

    Args:
        env: The environment.
        command_name: Name of the motion command.
        std: Std (in meters) for exponential reward.

    Returns:
        Feet position tracking reward.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    feet_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    feet_idx = _get_body_indexes(command, feet_names)

    if len(feet_idx) > 0:
        diff = command.body_pos_w[:, feet_idx, :] - command.robot_body_pos_w[:, feet_idx, :]
        error = (diff ** 2).mean(dim=-1).mean(dim=-1)
        return torch.exp(-error / (std ** 2))
    else:
        return torch.zeros(env.num_envs, device=env.device)


def teleop_body_rotation_extend(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.2,
) -> torch.Tensor:
    """
    Full body rotation tracking in world frame.

    Args:
        env: The environment.
        command_name: Name of the motion command.
        std: Std (in radians) for exponential reward.

    Returns:
        Body rotation tracking reward.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    # Use all bodies
    rotation_error = quat_error_magnitude(command.body_quat_w, command.robot_body_quat_w)  # [N, num_bodies]
    error = (rotation_error ** 2).mean(dim=-1)  # [N]
    return torch.exp(-error / (std ** 2))


def teleop_body_velocity_extend(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.5,
) -> torch.Tensor:
    """
    Full body linear velocity tracking in world frame.

    Args:
        env: The environment.
        command_name: Name of the motion command.
        std: Std (in m/s) for exponential reward.

    Returns:
        Body linear velocity tracking reward.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    diff = command.body_lin_vel_w - command.robot_body_lin_vel_w
    error = (diff ** 2).mean(dim=-1).mean(dim=-1)
    return torch.exp(-error / (std ** 2))


def teleop_body_ang_velocity_extend(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 2.0,
) -> torch.Tensor:
    """
    Full body angular velocity tracking in world frame.

    Args:
        env: The environment.
        command_name: Name of the motion command.
        std: Std (in rad/s) for exponential reward.

    Returns:
        Body angular velocity tracking reward.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    diff = command.body_ang_vel_w - command.robot_body_ang_vel_w
    error = (diff ** 2).mean(dim=-1).mean(dim=-1)
    return torch.exp(-error / (std ** 2))


def motion_anchor_linear_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 1.0,
) -> torch.Tensor:
    """
    Anchor (base) linear velocity tracking in world frame.
    Tracks the robot anchor/pelvis linear velocity.

    Args:
        env: The environment.
        command_name: Name of the motion command.
        std: Std for exponential reward.

    Returns:
        Anchor linear velocity tracking reward.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(
        torch.square(command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w),
        dim=-1
    )
    return torch.exp(-error / std**2)

def contact_feet(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float = 0.05,
) -> torch.Tensor:
    """
    Reward function that checks if the robot's foot contact state matches the reference trajectory.

    A foot is considered "in contact" if its height (Z-coordinate) is less than the threshold.
    Scoring: +0.5 per foot if the state (contact or swing) matches the reference, 
    resulting in a max reward of 1.0 for both feet matching.

    Args:
        env: The environment instance.
        command_name: Name of the motion command term.
        threshold: Height threshold (in meters) to determine contact. Defaults to 0.05.

    Returns:
        Contact consistency reward [num_envs].
    """
    # 1. Access the motion command term
    command: MotionCommand = env.command_manager.get_term(command_name)
    feet_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    feet_idx = _get_body_indexes(command, feet_names)

    if len(feet_idx) > 0:
        # 2. Extract Z-coordinates for both robot and reference trajectory
        # Shape: [num_envs, 2]
        current_feet_z = command.robot_body_pos_w[:, feet_idx, 2]
        reference_feet_z = command.body_pos_w[:, feet_idx, 2]

        # 3. Determine contact state (True if height < threshold)
        # Shape: [num_envs, 2] (bool)
        current_contact = current_feet_z < threshold
        reference_contact = reference_feet_z < threshold

        # 4. Compare current state with reference state
        # A match occurs if both are in contact OR both are in swing
        # Shape: [num_envs, 2] (float: 1.0 for match, 0.0 for mismatch)
        matching_states = (current_contact == reference_contact).float()

        # 5. Calculate final reward
        # Sum across the two feet and multiply by 0.5 (max reward 1.0)
        return matching_states.sum(dim=-1) * 0.5
    else:
        return torch.zeros(env.num_envs, device=env.device)


def frontres_dynamic_balance_margin_proxy(
    env: ManagerBasedRLEnv,
    command_name: str,
    foot_body_names: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link"),
    contact_height: float = 0.08,
) -> torch.Tensor:
    """返回 review-only 的保守 balance margin 候选.

    函数名说明:
        `frontres_dynamic_balance_margin_proxy` 是 env-facing metric proxy, 从当前
        机器人状态构造 margin 供 review/候选 reward 比较; 它不是已注册训练 reward,
        也不是 policy observation.

    主链路:
        上游: review/eval/probe 代码显式调用; FrontRES 训练主线在
        `rsl_rl.frontres.frontres_balance.py` 中对四分支 rollout 计算同一类 margin.
        内部: 读取 `MotionCommand` 的 root/foot 状态, 调用
        `frontres_balance_context_from_feet`.
        下游: 输出 min(root_margin, capture_margin), 供 Clean/Noisy/Repaired
        margin 比较, no-regret reward, 或离线审计使用.

    接线状态:
        不注册为普通 RewTerm; 正式训练 reward 从 FrontRES 四分支 reward
        window 接入, 避免把绝对静态 margin 当环境 reward.

    语义:
        从可部署机器人状态计算 min(root_margin, capture_margin).
        比较 Clean/Noisy/Repaired margin, 不作为绝对 reward 直接使用.
    """
    # B1: 从当前 motion command 读取部署可得的机器人状态.
    command: MotionCommand = env.command_manager.get_term(command_name)
    foot_ids = _get_body_indexes(command, list(foot_body_names))
    if len(foot_ids) != 2:
        return torch.zeros(env.num_envs, device=env.device)

    # B2: 提取纯 helper 需要的 root 和 foot 张量.
    root_xy = command.robot.data.root_pos_w[:, :2]
    root_vel_w = getattr(command.robot.data, "root_lin_vel_w", None)
    if root_vel_w is None:
        root_vel_w = command.robot_anchor_vel_w[:, :3]
    root_vel_xy = root_vel_w[:, :2]
    feet_pos = command.robot_body_pos_w[:, foot_ids, :]

    # B3: 复用观测候选 helper, 保证 metric 和 context 共用同一布局.
    context = frontres_balance_context_from_feet(
        root_xy,
        root_vel_xy,
        feet_pos[..., :2],
        feet_pos[..., 2],
        contact_height=contact_height,
        env_origin_z=env.scene.env_origins[:, 2],
    )

    # B4: 将 root/capture 两个 margin 合成一个保守标量.
    root_margin = context[:, -3]
    capture_margin = context[:, -2]
    return torch.minimum(root_margin, capture_margin)


def frontres_no_regret_balance_reward_candidate(
    repaired_margin: torch.Tensor,
    noisy_margin: torch.Tensor,
    clean_margin: torch.Tensor,
    slack: float = 0.02,
) -> torch.Tensor:
    """返回候选 Clean-relative no-regret balance reward.

    函数名说明:
        `frontres_no_regret_balance_reward_candidate` 是 reward adapter, 连接
        rollout 分支 margin 和 mdp.balance 的纯公式 helper; 它不是 observation,
        也不负责从 env 读取 root/foot/contact 状态.

    主链路:
        上游: rollout/eval adapter 传入 Repaired, Noisy/no-op, Clean 三条分支
        的 balance margin.
        内部: 调用 `frontres_no_regret_balance_reward`.
        下游: 输出 Clean-relative no-regret reward; 正式训练主线由
        当前正式 Segment Gain 路径在四分支 rollout 内接入.

    语义:
        Repaired 移除 Noisy 在 Clean 下界以下的额外风险时为正.
    """
    # B1: 保持 rollout adapter 很薄, 公式所有权留在 mdp.balance.
    return frontres_no_regret_balance_reward(
        repaired_margin,
        noisy_margin,
        clean_margin,
        slack=slack,
    )
    
def teleop_body_position_feet_z(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.03,
) -> torch.Tensor:
    """
    Feet position_z tracking in world frame (high precision).

    Args:
        env: The environment.
        command_name: Name of the motion command.
        std: Std (in meters) for exponential reward.

    Returns:
        Feet position tracking reward.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    feet_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    feet_idx = _get_body_indexes(command, feet_names)

    if len(feet_idx) > 0:
        diff = command.body_pos_w[:, feet_idx, 2] - command.robot_body_pos_w[:, feet_idx, 2]
        error = (diff ** 2).mean(dim=-1)
        return torch.exp(-error / (std ** 2))
    else:
        return torch.zeros(env.num_envs, device=env.device)
