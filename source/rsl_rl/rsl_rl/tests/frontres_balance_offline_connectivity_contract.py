#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import types

import torch


ROOT = Path(__file__).resolve().parents[4]
RSL_SOURCE = ROOT / "source/rsl_rl"
BALANCE_PATH = ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/balance.py"
OBS_PATH = ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py"
RUNTIME_PATH = ROOT / "source/rsl_rl/rsl_rl/runners/frontres_runtime.py"
G1_CFG = ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py"

NUM_ENVS = 2
BALANCE_DIM = 14
HISTORY = 5
ANCHOR_EXTRA_DIM = 30
BALANCE_EXTRA_DIM = BALANCE_DIM * HISTORY
FRONTRES_EXTRA_DIM = ANCHOR_EXTRA_DIM + BALANCE_EXTRA_DIM
GMT_SUFFIX_DIM = 770
POLICY_OBS_DIM = FRONTRES_EXTRA_DIM + GMT_SUFFIX_DIM


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # B1: 实现测试需要的 wxyz inverse rotation, 避免依赖 IsaacLab.
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q.unbind(-1)
    q_vec = torch.stack([-x, -y, -z], dim=-1)
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + w.unsqueeze(-1) * t + torch.cross(q_vec, t, dim=-1)


def _install_observation_stubs():
    balance = _load_module("frontres_balance_connectivity_under_test", BALANCE_PATH)

    math_stub = types.ModuleType("isaaclab.utils.math")
    math_stub.quat_rotate_inverse = _quat_rotate_inverse
    math_stub.matrix_from_quat = lambda q: torch.eye(3, device=q.device, dtype=q.dtype).expand(q.shape[0], 3, 3)
    math_stub.subtract_frame_transforms = lambda *args, **kwargs: (args[2] - args[0], args[3])
    math_stub.quat_mul = lambda a, b: a
    math_stub.quat_inv = lambda q: q
    sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))
    sys.modules.setdefault("isaaclab.utils", types.ModuleType("isaaclab.utils"))
    sys.modules["isaaclab.utils.math"] = math_stub

    pkg_names = [
        "whole_body_tracking",
        "whole_body_tracking.tasks",
        "whole_body_tracking.tasks.tracking",
        "whole_body_tracking.tasks.tracking.mdp",
    ]
    for name in pkg_names:
        sys.modules.setdefault(name, types.ModuleType(name))

    commands_stub = types.ModuleType("whole_body_tracking.tasks.tracking.mdp.commands")
    commands_stub.MotionCommand = object
    sys.modules["whole_body_tracking.tasks.tracking.mdp.balance"] = balance
    sys.modules["whole_body_tracking.tasks.tracking.mdp.commands"] = commands_stub
    sys.modules["whole_body_tracking.tasks.tracking.mdp"].balance = balance
    sys.modules["whole_body_tracking.tasks.tracking.mdp"].commands = commands_stub


def _load_observations_module():
    _install_observation_stubs()
    return _load_module("frontres_observations_connectivity_under_test", OBS_PATH)


def _load_runtime_module():
    if str(RSL_SOURCE) not in sys.path:
        sys.path.insert(0, str(RSL_SOURCE))
    runtime_diagnostics_stub = types.ModuleType("rsl_rl.frontres.runtime_diagnostics")
    runtime_diagnostics_stub.maybe_print_frontres_restore_debug = lambda *args, **kwargs: None
    sys.modules.setdefault("rsl_rl.frontres", types.ModuleType("rsl_rl.frontres"))
    sys.modules["rsl_rl.frontres.runtime_diagnostics"] = runtime_diagnostics_stub
    return _load_module("frontres_runtime_connectivity_under_test", RUNTIME_PATH)


class FakeNormalizer:
    def __init__(self, dim: int):
        self.dim = dim
        self._mean = torch.zeros(1, dim)
        self._std = torch.ones(1, dim)
        self.calls: list[tuple[int, ...]] = []

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(obs.shape))
        assert obs.shape[-1] == self.dim
        return obs + 1.0


class FakeGMTPolicy:
    def __init__(self):
        self.calls: list[tuple[int, ...]] = []

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(obs.shape))
        assert obs.shape[-1] == GMT_SUFFIX_DIM
        return obs


def _make_fake_env(
    *,
    body_names: list[str] | None = None,
    root_xy: torch.Tensor | None = None,
    root_vel_xy: torch.Tensor | None = None,
    feet_z: torch.Tensor | None = None,
    env_origin_z: torch.Tensor | None = None,
    include_root_lin_vel: bool = True,
):
    if body_names is None:
        body_names = ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link", "torso_link"]
    if root_xy is None:
        root_xy = torch.tensor([[0.0, 0.0], [0.05, 0.0]], dtype=torch.float32)
    if root_vel_xy is None:
        root_vel_xy = torch.tensor([[0.0, 0.0], [0.6, 0.0]], dtype=torch.float32)
    if feet_z is None:
        feet_z = torch.full((NUM_ENVS, 2), 0.02, dtype=torch.float32)
    if env_origin_z is None:
        env_origin_z = torch.zeros(NUM_ENVS, dtype=torch.float32)

    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    robot_data = SimpleNamespace(
        root_pos_w=torch.cat([root_xy, torch.full((NUM_ENVS, 1), 0.8)], dim=-1),
        root_quat_w=root_quat,
    )
    root_lin_vel_w = torch.cat([root_vel_xy, torch.zeros(NUM_ENVS, 1)], dim=-1)
    if include_root_lin_vel:
        robot_data.root_lin_vel_w = root_lin_vel_w
    robot = SimpleNamespace(data=robot_data)
    robot_body_pos_w = torch.tensor(
        [
            [[0.0, 0.0, 0.8], [-0.1, -0.08, 0.02], [0.1, 0.08, 0.02], [0.0, 0.0, 1.1]],
            [[0.05, 0.0, 0.8], [-0.1, -0.08, 0.02], [0.1, 0.08, 0.02], [0.05, 0.0, 1.1]],
        ],
        dtype=torch.float32,
    )
    robot_body_pos_w[:, 1, 2] = feet_z[:, 0]
    robot_body_pos_w[:, 2, 2] = feet_z[:, 1]
    command = SimpleNamespace(
        cfg=SimpleNamespace(body_names=body_names),
        robot=robot,
        robot_body_pos_w=robot_body_pos_w,
        robot_anchor_vel_w=torch.cat([root_lin_vel_w, torch.zeros(NUM_ENVS, 3)], dim=-1),
    )
    return SimpleNamespace(
        num_envs=NUM_ENVS,
        device=torch.device("cpu"),
        command_manager=SimpleNamespace(get_term=lambda name: command),
        scene=SimpleNamespace(env_origins=torch.cat([torch.zeros(NUM_ENVS, 2), env_origin_z.view(NUM_ENVS, 1)], dim=-1)),
    )


def test_balance_obs_offline_connectivity() -> None:
    # B1: 静态检查正式 G1 FrontRES cfg 已经接到 balance obs helper.
    cfg_text = G1_CFG.read_text()
    assert "self.observations.policy.frontres_balance_context = ObsTerm(" in cfg_text
    assert "func=mdp.frontres_balance_context_proxy" in cfg_text
    assert "870 dims total" in cfg_text

    # B2: 用 fake env 调真实 observation adapter, 检查 env->obs 边界.
    observations = _load_observations_module()
    balance_context = observations.frontres_balance_context_proxy(_make_fake_env(), "motion")
    assert tuple(balance_context.shape) == (NUM_ENVS, BALANCE_DIM)
    assert torch.isfinite(balance_context).all()
    torch.testing.assert_close(balance_context[:, 8:11], torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]))

    # B3: 构造历史展开后的 FrontRES prefix 和 GMT suffix, 检查拼接布局.
    anchor_history = torch.full((NUM_ENVS, ANCHOR_EXTRA_DIM), 2.0)
    balance_history = balance_context.repeat(1, HISTORY)
    extra_prefix = torch.cat([anchor_history, balance_history], dim=-1)
    gmt_suffix = torch.full((NUM_ENVS, GMT_SUFFIX_DIM), 130.0)
    policy_obs = torch.cat([extra_prefix, gmt_suffix], dim=-1)
    assert tuple(extra_prefix.shape) == (NUM_ENVS, FRONTRES_EXTRA_DIM)
    assert tuple(policy_obs.shape) == (NUM_ENVS, POLICY_OBS_DIM)

    # B4: 调真实 runner normalizer 分流, 检查 only suffix enters GMT normalizer.
    runtime = _load_runtime_module()
    runner = SimpleNamespace(
        _frontres_gmt_obs_dim=GMT_SUFFIX_DIM,
        _frontres_extra_mean=torch.zeros(1, FRONTRES_EXTRA_DIM),
        _frontres_extra_std=torch.ones(1, FRONTRES_EXTRA_DIM),
        obs_normalizer=FakeNormalizer(GMT_SUFFIX_DIM),
    )
    normalized = runtime.apply_obs_normalizer(runner, policy_obs)
    assert tuple(normalized.shape) == (NUM_ENVS, POLICY_OBS_DIM)
    assert runner.obs_normalizer.calls == [(NUM_ENVS, GMT_SUFFIX_DIM)]
    torch.testing.assert_close(normalized[:, :FRONTRES_EXTRA_DIM], extra_prefix)
    torch.testing.assert_close(normalized[:, FRONTRES_EXTRA_DIM:], gmt_suffix + 1.0)

    # B5: 调真实 FrontRESActorCritic GMT direct helper, 检查 GMT policy 只接收 770D suffix.
    from rsl_rl.modules.front_residual_actor_critic import FrontRESActorCritic

    policy = FrontRESActorCritic.__new__(FrontRESActorCritic)
    policy.gmt_normalizer = FakeNormalizer(GMT_SUFFIX_DIM)
    policy.gmt_policy = FakeGMTPolicy()
    policy.ref_vel_estimator = None
    policy._cached_full_policy_obs = policy_obs
    policy._pad_observations_for_gmt = lambda obs: obs
    gmt_action = FrontRESActorCritic._run_gmt_direct(policy, torch.empty(NUM_ENVS, 4), None, None)
    assert policy.gmt_policy.calls == [(NUM_ENVS, GMT_SUFFIX_DIM)]
    assert tuple(gmt_action.shape) == (NUM_ENVS, GMT_SUFFIX_DIM)

    print(
        "[FrontRES Balance Offline Connectivity] "
        f"balance_context_shape={tuple(balance_context.shape)} "
        f"extra_prefix_shape={tuple(extra_prefix.shape)} "
        f"policy_obs_shape={tuple(policy_obs.shape)} "
        f"normalizer_calls={runner.obs_normalizer.calls} "
        f"gmt_policy_calls={policy.gmt_policy.calls}",
        flush=True,
    )


def test_balance_obs_broad_fake_schema_and_lifecycle() -> None:
    observations = _load_observations_module()

    # B1: schema fake 覆盖 root_lin_vel_w 存在与缺失时的 fallback.
    direct = observations.frontres_balance_context_proxy(
        _make_fake_env(root_vel_xy=torch.tensor([[0.0, 0.0], [0.4, 0.0]])),
        "motion",
    )
    fallback = observations.frontres_balance_context_proxy(
        _make_fake_env(root_vel_xy=torch.tensor([[0.0, 0.0], [0.4, 0.0]]), include_root_lin_vel=False),
        "motion",
    )
    assert tuple(direct.shape) == (NUM_ENVS, BALANCE_DIM)
    assert tuple(fallback.shape) == (NUM_ENVS, BALANCE_DIM)
    torch.testing.assert_close(direct, fallback)

    # B2: body name 缺失时必须 fail closed, 不能产生错位脚索引.
    missing_feet = observations.frontres_balance_context_proxy(
        _make_fake_env(body_names=["pelvis", "torso_link"]),
        "motion",
    )
    torch.testing.assert_close(missing_feet, torch.zeros(NUM_ENVS, BALANCE_DIM))

    # B3: contact proxy 覆盖双脚接触, 单脚摆动, 无接触 fallback, env_origin_z offset.
    contact_cases = [
        torch.tensor([[0.02, 0.02], [0.02, 0.02]]),
        torch.tensor([[0.02, 0.20], [0.02, 0.20]]),
        torch.tensor([[0.20, 0.20], [0.20, 0.20]]),
    ]
    contexts = [
        observations.frontres_balance_context_proxy(_make_fake_env(feet_z=feet_z), "motion")
        for feet_z in contact_cases
    ]
    assert torch.isfinite(torch.stack(contexts)).all()
    assert torch.all(contexts[0][:, 0:2] == torch.tensor([[1.0, 1.0], [1.0, 1.0]]))
    assert torch.all(contexts[1][:, 0:2] == torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
    assert torch.all(contexts[2][:, -1] == 0.0)
    origin_offset_ctx = observations.frontres_balance_context_proxy(
        _make_fake_env(
            feet_z=torch.tensor([[1.02, 1.02], [1.02, 1.02]]),
            env_origin_z=torch.tensor([1.0, 1.0]),
        ),
        "motion",
    )
    assert torch.all(origin_offset_ctx[:, 0:2] == 1.0)

    # B4: fake reset/step lifecycle, 检查每帧 shape/finite 稳定, 且速度越高 capture margin 越小.
    sequence = [
        _make_fake_env(root_xy=torch.tensor([[0.0, 0.0], [0.0, 0.0]]), root_vel_xy=torch.zeros(NUM_ENVS, 2)),
        _make_fake_env(root_xy=torch.tensor([[0.02, 0.0], [0.02, 0.0]]), root_vel_xy=torch.full((NUM_ENVS, 2), 0.2)),
        _make_fake_env(root_xy=torch.tensor([[0.04, 0.0], [0.04, 0.0]]), root_vel_xy=torch.full((NUM_ENVS, 2), 0.6)),
    ]
    step_contexts = [observations.frontres_balance_context_proxy(env, "motion") for env in sequence]
    for context in step_contexts:
        assert tuple(context.shape) == (NUM_ENVS, BALANCE_DIM)
        assert torch.isfinite(context).all()
    assert torch.all(step_contexts[2][:, -2] < step_contexts[0][:, -2])

    # B5: row permutation metamorphic case, 保护 env row 语义不被聚合打乱.
    permuted_env = _make_fake_env(
        root_xy=torch.tensor([[0.1, 0.0], [-0.1, 0.0]]),
        root_vel_xy=torch.tensor([[0.5, 0.0], [0.0, 0.0]]),
    )
    permuted_ctx = observations.frontres_balance_context_proxy(permuted_env, "motion")
    swapped_env = _make_fake_env(
        root_xy=torch.tensor([[-0.1, 0.0], [0.1, 0.0]]),
        root_vel_xy=torch.tensor([[0.0, 0.0], [0.5, 0.0]]),
    )
    swapped_ctx = observations.frontres_balance_context_proxy(swapped_env, "motion")
    torch.testing.assert_close(permuted_ctx[[1, 0]], swapped_ctx)

    print(
        "[FrontRES Balance Broad Offline Audit] "
        "schema=root_lin_vel_w+fallback "
        "missing_feet=zeros "
        "contact_cases=both/single/none/origin_offset "
        f"lifecycle_steps={len(step_contexts)} "
        "metamorphic=row_permutation",
        flush=True,
    )


if __name__ == "__main__":
    test_balance_obs_offline_connectivity()
    test_balance_obs_broad_fake_schema_and_lifecycle()
    print("frontres_balance_offline_connectivity_contract: ok")
