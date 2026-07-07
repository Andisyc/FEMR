from __future__ import annotations

import torch


def axis_aligned_support_margin(
    point_xy: torch.Tensor,
    support_center_xy: torch.Tensor,
    support_half_extent_xy: torch.Tensor,
) -> torch.Tensor:
    """返回点到 axis-aligned support box 的有符号裕度.

    函数名说明:
        `axis_aligned_support_margin` 是纯几何 helper, 只计算一个点是否落在
        foot-box support proxy 内; 它不是 reward, observation interface, 或 env adapter.

    主链路:
        上游: `frontres_balance_context_from_feet`.
        下游: 生成 root/capture signed margin, 再被 14D balance context 和
        review-only balance margin proxy 复用.
    """
    slack = support_half_extent_xy - torch.abs(point_xy - support_center_xy)
    return slack.min(dim=-1).values


def frontres_balance_context_from_feet(
    root_xy: torch.Tensor,
    root_vel_xy: torch.Tensor,
    feet_xy: torch.Tensor,
    feet_z: torch.Tensor,
    *,
    contact_height: float = 0.08,
    foot_radius: float = 0.04,
    capture_height: float = 0.8,
    gravity: float = 9.81,
    env_origin_z: torch.Tensor | None = None,
    projected_gravity: torch.Tensor | None = None,
) -> torch.Tensor:
    """构造 FrontRES 可部署的 14D balance context.

    函数名说明:
        `frontres_balance_context_from_feet` 是纯张量布局和公式 helper, 负责把
        root/feet/contact/gravity 张量打包成固定 14D policy 输入; 它不读取 env,
        不注册 ObsTerm, 不计算 reward.

    主链路:
        上游: `frontres_balance_context_proxy` 从 env obs 路径调用;
              `frontres_dynamic_balance_margin_proxy` 从 review metric 路径调用.
        内部: 调用 `axis_aligned_support_margin` 计算 root/capture margin.
        下游: policy obs 前缀使用完整 14D; reward/metric proxy 只复用最后的 margin.

    输出布局:
        contact(2), root_offset(2), capture_offset(2), support_half(2),
        projected_gravity(3), root_margin(1), capture_margin(1), has_contact(1).
    """

    # B1: 用脚高生成接触 mask, 并保证无接触时输出仍然有限.
    if env_origin_z is None:
        feet_height = feet_z
    else:
        feet_height = feet_z - env_origin_z.view(-1, 1)
    contact = feet_height <= contact_height
    has_contact = contact.any(dim=-1, keepdim=True)
    active = torch.where(has_contact, contact, torch.ones_like(contact)).unsqueeze(-1)

    # B2: 用接触脚生成 foot-box support proxy, 供 obs 和 reward proxy 共用.
    inf = torch.finfo(feet_xy.dtype).max
    support_min = torch.where(active, feet_xy, torch.full_like(feet_xy, inf)).amin(dim=1) - foot_radius
    support_max = torch.where(active, feet_xy, torch.full_like(feet_xy, -inf)).amax(dim=1) + foot_radius
    support_center = 0.5 * (support_min + support_max)
    support_half = (0.5 * (support_max - support_min)).clamp(min=foot_radius)

    # B3: 在同一个支撑代理下计算 root margin 和 capture margin.
    omega = (gravity / max(capture_height, 1e-6)) ** 0.5
    capture_xy = root_xy + root_vel_xy / omega
    root_margin = axis_aligned_support_margin(root_xy, support_center, support_half)
    capture_margin = axis_aligned_support_margin(capture_xy, support_center, support_half)

    # B4: 补入身体姿态信号, 让相同 capture 状态可以区分直立和倾斜.
    if projected_gravity is None:
        projected_gravity = torch.zeros(root_xy.shape[0], 3, dtype=root_xy.dtype, device=root_xy.device)

    # B5: 打包稳定 14D 布局, margin 保持在 -3/-2, 方便 reward proxy 复用.
    return torch.cat(
        [
            contact.to(root_xy.dtype),
            root_xy - support_center,
            capture_xy - support_center,
            support_half,
            projected_gravity.to(root_xy.dtype),
            root_margin.unsqueeze(-1),
            capture_margin.unsqueeze(-1),
            has_contact.to(root_xy.dtype),
        ],
        dim=-1,
    )


def frontres_no_regret_balance_reward(
    repaired_margin: torch.Tensor,
    noisy_margin: torch.Tensor,
    clean_margin: torch.Tensor,
    *,
    slack: float = 0.02,
) -> torch.Tensor:
    """返回 Repaired 相对 Noisy 的 Clean-relative 风险降低量.

    函数名说明:
        `frontres_no_regret_balance_reward` 是纯 reward 公式 helper, 只比较三条
        rollout 分支的 balance margin; 它不是 env-facing RewTerm, 也不负责
        从机器人状态构造 margin.

    主链路:
        上游: `frontres_no_regret_balance_reward_candidate` 或后续 rollout adapter
        传入 Repaired, Noisy/no-op, Clean 三条分支的 margin.
        下游: 输出 Clean-relative no-regret 风险降低量, 供候选 reward/review 使用.

    语义:
        正值表示 Repaired 比 no-op/Noisy 分支减少了额外平衡风险.
        Clean 是动态参照下界, 不是绝对静态稳定目标.
    """
    # B1: 用 Clean margin 定义动态动作允许下界.
    clean_floor = clean_margin - slack

    # B2: 只计算低于 Clean 下界的额外风险.
    noisy_risk = torch.relu(clean_floor - noisy_margin)
    repaired_risk = torch.relu(clean_floor - repaired_margin)

    # B3: 奖励 Repaired 相对 Noisy 移除的风险.
    return noisy_risk - repaired_risk
