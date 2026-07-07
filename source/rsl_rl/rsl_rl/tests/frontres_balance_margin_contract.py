#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import importlib.util
from types import SimpleNamespace
import sys
import types

import torch


ROOT = Path(__file__).resolve().parents[4]
RSL_SOURCE = ROOT / "source/rsl_rl"
BALANCE_PATH = ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/balance.py"
G1_RSL_CFG = ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_mosaic_cfg.py"
if str(RSL_SOURCE) not in sys.path:
    sys.path.insert(0, str(RSL_SOURCE))

math_stub = types.ModuleType("isaaclab.utils.math")
math_stub.euler_xyz_from_quat = lambda q: (
    torch.zeros(q.shape[0], device=q.device, dtype=q.dtype),
    torch.zeros(q.shape[0], device=q.device, dtype=q.dtype),
    torch.zeros(q.shape[0], device=q.device, dtype=q.dtype),
)
math_stub.quat_apply = lambda q, v: v
math_stub.quat_from_euler_xyz = lambda roll, pitch, yaw: torch.stack(
    [torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll), torch.zeros_like(roll)],
    dim=-1,
)
math_stub.quat_inv = lambda q: q
math_stub.quat_mul = lambda a, b: b
math_stub.yaw_quat = lambda q: q
sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))
sys.modules.setdefault("isaaclab.utils", types.ModuleType("isaaclab.utils"))
sys.modules["isaaclab.utils.math"] = math_stub

# B1: 只加载纯 helper 模块; 完整 package import 会依赖 IsaacLab tasks.
spec = importlib.util.spec_from_file_location("frontres_balance_under_test", BALANCE_PATH)
assert spec is not None and spec.loader is not None
balance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(balance)

# B2: 将 helper 绑定成本地变量, 让 contract 像普通模块测试一样阅读.
frontres_balance_context_from_feet = balance.frontres_balance_context_from_feet
frontres_no_regret_balance_reward = balance.frontres_no_regret_balance_reward

from rsl_rl.frontres.frontres_reward_window import (  # noqa: E402
    build_frontres_reward_window,
    compose_frontres_reward_delta,
)
from rsl_rl.frontres.frontres_reward_diagnostics import (  # noqa: E402
    initialize_frontres_reward_diagnostic_sums,
    materialize_frontres_reward_diagnostic_means,
)


def test_balance_context_reports_capture_margin() -> None:
    # B1: 构造只改变水平速度的 tiny fixture, 并显式传入 projected gravity.
    root_xy = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    root_vel_xy = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    feet_xy = torch.tensor([[[-0.1, -0.1], [0.1, 0.1]], [[-0.1, -0.1], [0.1, 0.1]]])
    feet_z = torch.zeros(2, 2)
    projected_gravity = torch.tensor([[0.0, 0.0, -1.0], [0.2, 0.0, -0.98]])

    # B2: 计算已接入 obs 的 14D balance context.
    context = frontres_balance_context_from_feet(
        root_xy,
        root_vel_xy,
        feet_xy,
        feet_z,
        projected_gravity=projected_gravity,
    )

    # B3: 检查布局和 capture-point 语义, 而不是绑定具体实现细节.
    assert tuple(context.shape) == (2, 14)
    torch.testing.assert_close(context[:, 8:11], projected_gravity)
    assert context[0, -2] > 0.0
    assert context[1, -2] < context[0, -2]


def test_no_regret_balance_reward_is_clean_relative() -> None:
    # B1: 包含一个位于静态代理外的 Clean 样本, 保护合法动态动作.
    clean_margin = torch.tensor([-0.10, 0.10])
    noisy_margin = torch.tensor([-0.30, -0.10])
    repaired_margin = torch.tensor([-0.20, 0.05])

    # B2: 使用 slack=0, 让 no-regret 期望值可以手算.
    reward = frontres_no_regret_balance_reward(repaired_margin, noisy_margin, clean_margin, slack=0.0)

    # B3: 检查 Clean-relative 风险降低, 而不是绝对静态稳定.
    torch.testing.assert_close(reward, torch.tensor([0.10, 0.15]))
    assert torch.all(reward > 0.0)


def test_balance_reward_enters_frontres_rdelta() -> None:
    # B1: 构造 repairable band 内的 FrontRES reward window, 关闭其他 reward 源.
    device = torch.device("cpu")
    cfg = {
        "frontres_reward_scale_dr_reference": 1.0,
        "frontres_reward_progress_min": 1.0,
        "frontres_constraint_progress_exponent": 1.0,
        "frontres_gap_floor_per_step": 0.001,
        "frontres_safe_gap_per_step": 0.0,
        "frontres_broken_gap_per_step": 1.0,
        "frontres_gap_gate_temp": 0.01,
        "frontres_selective_reward_enabled": True,
        "frontres_exec_reward_signal": "gain",
        "frontres_effective_gain_bonus_weight": 0.0,
        "frontres_candidate_ranking_reward_enabled": False,
        "frontres_balance_reward_enabled": True,
        "frontres_balance_reward_weight": 0.5,
    }
    runner = SimpleNamespace(cfg=cfg)
    n_train = n_exec = 2
    zeros = torch.zeros(n_train, device=device)
    reward_window = build_frontres_reward_window(
        runner=runner,
        cfg=cfg,
        n_train=n_train,
        n_exec=n_exec,
        exec_clean=torch.tensor([0.5, 0.5], device=device),
        exec_perturbed=torch.tensor([0.1, 0.1], device=device),
        exec_feasible=torch.tensor([0.5, 0.5], device=device),
        exec_frontres=torch.tensor([0.1, 0.1], device=device),
        repair_gain=zeros,
        mode_groups=[("planar",)] * n_exec,
        e_raw=torch.ones(n_exec, device=device),
        e_fr=torch.ones(n_exec, device=device),
        intervention_cost=zeros,
        action_activity=zeros,
        under_repair_penalty=zeros.clone(),
        dr_scale=1.0,
        ppo_actor_weight_current=1.0,
        stable_route_active_mask=torch.zeros(n_exec, dtype=torch.bool, device=device),
        device=device,
        balance_reward=torch.tensor([0.04, -0.02], device=device),
        balance_repaired_margin=torch.tensor([0.02, -0.04], device=device),
        balance_candidate_margin=torch.tensor([0.01, -0.05], device=device),
        balance_noisy_margin=torch.tensor([-0.02, -0.02], device=device),
        balance_clean_margin=torch.tensor([0.03, 0.00], device=device),
    )

    # B2: 组合正式 r_delta, 期望 balance weighted bonus 进入训练 reward.
    reward_window = compose_frontres_reward_delta(
        cfg=cfg,
        reward_window=reward_window,
        n_train=n_train,
        n_exec=n_exec,
        n_candidate=0,
        repair_gain=zeros,
        candidate_gain=zeros,
        projection_gain=zeros,
        r_step=zeros,
        r_rescue=zeros,
        intervention_cost=zeros,
        overcorrection_cost=zeros,
        w_exec=0.0,
        repair_scale=1.0,
        w_geom=0.0,
        w_rescue=0.0,
        w_exec_harm=0.0,
        device=device,
    )

    # B3: 检查 raw/weighted balance 项和最终 r_delta 对齐.
    assert reward_window.r_delta is not None
    assert reward_window.balance_bonus is not None
    assert reward_window.balance_weighted_bonus is not None
    torch.testing.assert_close(reward_window.balance_bonus, torch.tensor([0.04, -0.02]))
    torch.testing.assert_close(reward_window.balance_weighted_bonus, torch.tensor([0.02, -0.01]))
    torch.testing.assert_close(reward_window.r_delta, torch.tensor([0.02, -0.01]))


def test_g1_frontres_config_enables_balance_reward_and_metric() -> None:
    # B1: 静态保护正式 G1 FrontRES 配置, 防止 helper 存在但主流程没打开.
    cfg_text = G1_RSL_CFG.read_text()
    assert "frontres_balance_metric_enabled = True" in cfg_text
    assert "frontres_balance_reward_enabled = True" in cfg_text
    assert "frontres_balance_reward_weight = 0.5" in cfg_text
    assert "frontres_balance_foot_body_names = [" in cfg_text


def test_balance_diagnostics_are_absent_until_balance_path_runs() -> None:
    # B1: 没有 balance 样本时必须返回 None, 避免日志把未测量伪装成 0.
    sums = initialize_frontres_reward_diagnostic_sums()
    means = materialize_frontres_reward_diagnostic_means(
        sums,
        is_frontres=True,
        is_task_space_mode=True,
        term_count=0,
        step_count=1,
    )
    assert means["frontres_balance_reward_mean"] is None


if __name__ == "__main__":
    test_balance_context_reports_capture_margin()
    test_no_regret_balance_reward_is_clean_relative()
    test_balance_reward_enters_frontres_rdelta()
    test_g1_frontres_config_enables_balance_reward_and_metric()
    test_balance_diagnostics_are_absent_until_balance_path_runs()
    print("[frontres balance margin contract] PASS")
