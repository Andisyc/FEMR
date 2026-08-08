#!/usr/bin/env python3
"""Deterministic S1/S3 contract for FRS-TRAIN-v007 proposal-only HSL."""

from __future__ import annotations

import ast
import copy
import inspect
import importlib.util
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import torch
from frontres_contract_imports import install_frontres_contract_packages


ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = ROOT / "source" / "rsl_rl"
RUNNERS_ROOT = SOURCE_ROOT / "rsl_rl" / "runners"
WARMUP_PATH = RUNNERS_ROOT / "frontres_warmup.py"
RUNTIME_PATH = RUNNERS_ROOT / "frontres_runtime.py"
CHECKPOINT_PATH = RUNNERS_ROOT / "frontres_checkpointing.py"
LEGACY_LABEL_PATH = RUNNERS_ROOT / "frontres_hsl_rollout_target.py"
ROLLOUT_STEP_PATH = RUNNERS_ROOT / "frontres_rollout_step.py"
LAYOUT_PATH = SOURCE_ROOT / "rsl_rl" / "modules" / "frontres_observation_layout.py"
UNIFIED_PATH = SOURCE_ROOT / "rsl_rl" / "algorithms" / "frontres_unified.py"
ON_POLICY_PATH = RUNNERS_ROOT / "on_policy_runner.py"
TRAIN_PATH = ROOT / "scripts" / "rsl_rl" / "train.py"
OBSERVATIONS_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "observations.py"
)
COMMANDS_PATH = OBSERVATIONS_PATH.with_name("commands.py")
G1_CFG_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "config"
    / "g1"
    / "agents"
    / "rsl_rl_mosaic_cfg.py"
)
RESET_HELPER_PATH = RUNNERS_ROOT.parent / "tests" / "frontres_v015_two_role_reset_contract.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_hsl_target_owner():
    """Load only the real target owner body without importing the IsaacLab host."""

    tree = ast.parse(OBSERVATIONS_PATH.read_text())
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_supervision_target_task_space"
    )
    isolated = ast.Module(body=[owner], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace = {
        "torch": torch,
        "ManagerBasedEnv": object,
        "MotionCommand": object,
    }
    exec(compile(isolated, str(OBSERVATIONS_PATH), "exec"), namespace)
    return namespace["get_supervision_target_task_space"]


def _expect_error(exc_type, callback, contains: str) -> None:
    try:
        callback()
    except exc_type as exc:
        assert contains in str(exc), str(exc)
    else:
        raise AssertionError(f"expected {exc_type.__name__} containing {contains!r}")


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


def _load_checkpointing():
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    install_frontres_contract_packages(SOURCE_ROOT / "rsl_rl")
    return _load("frontres_checkpointing_h1_s1_contract", CHECKPOINT_PATH)


def _load_unified_guard():
    rsl_rl = _package("rsl_rl")
    modules = _package("rsl_rl.modules")
    storage = _package("rsl_rl.storage")
    algorithms = _package("rsl_rl.algorithms")
    rsl_rl.modules = modules
    rsl_rl.storage = storage
    rsl_rl.algorithms = algorithms
    modules.ActorCritic = type("ActorCritic", (), {})
    modules.FrontRESActorCritic = type("FrontRESActorCritic", (), {})
    modules.ResidualActorCritic = type("ResidualActorCritic", (), {})
    storage.RolloutStorage = type("RolloutStorage", (), {})
    return _load("rsl_rl.algorithms.frontres_unified", UNIFIED_PATH)


def _load_q29_modules():
    rsl_rl = _package("rsl_rl")
    modules = _package("rsl_rl.modules")
    frontres = _package("rsl_rl.frontres")
    runners = _package("rsl_rl.runners")
    rsl_rl.modules = modules
    rsl_rl.frontres = frontres
    rsl_rl.runners = runners
    modules.FrontRESActorCritic = type("FrontRESActorCritic", (), {})
    layout = _load("rsl_rl.modules.frontres_observation_layout", LAYOUT_PATH)
    modules.frontres_observation_layout = layout
    diagnostics = types.ModuleType("rsl_rl.frontres.runtime_diagnostics")
    diagnostics.maybe_print_frontres_restore_debug = lambda *_args, **_kwargs: None
    sys.modules[diagnostics.__name__] = diagnostics
    frontres.runtime_diagnostics = diagnostics
    runtime = _load("rsl_rl.runners.frontres_runtime", RUNTIME_PATH)
    warmup = _load("frontres_warmup_h1_s1_contract", WARMUP_PATH)
    return layout, runtime, warmup


def _load_stage1_preset():
    tree = ast.parse(TRAIN_PATH.read_text())
    wanted = {
        "_set_if_present",
        "_parse_frontres_v015_future_offsets",
        "_apply_frontres_stage_preset",
        "_configure_frontres_stage1_hsl_env_cfg",
    }
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"RslRlOnPolicyRunnerCfg": object}
    exec(compile(module, str(TRAIN_PATH), "exec"), namespace)
    return (
        namespace["_apply_frontres_stage_preset"],
        namespace["_configure_frontres_stage1_hsl_env_cfg"],
    )


def _intent(batch_size: int = 2, hmax: int = 2) -> torch.Tensor:
    rows = torch.arange(batch_size, dtype=torch.float32).reshape(batch_size, 1, 1) * 1000.0
    frames = torch.arange(hmax + 1, dtype=torch.float32).reshape(1, hmax + 1, 1) * 100.0
    joints = torch.arange(29, dtype=torch.float32).reshape(1, 1, 29)
    return rows + frames + joints


def _provenance(batch_size: int) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "current_root_artifact_provenance": "noisy_root_artifact_t",
            "intent_q29_provenance": "deployment_noisy_q29",
            "intent_q29_source": "motion_internal_q29",
            "clean_continuation_provenance": "clean_gmt_only",
        }
        for _ in range(batch_size)
    )


class _Normalizer:
    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        self.calls.append(value.detach().clone())
        return value / 2.0


class _CheckpointNormalizer(torch.nn.Module):
    def __init__(self, dim: int, *, value: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("_mean", torch.full((1, dim), value))
        self.register_buffer("_var", torch.full((1, dim), value + 1.0))
        self.register_buffer("_std", torch.sqrt(torch.full((1, dim), value + 1.0)))
        self.register_buffer("count", torch.tensor(7, dtype=torch.long))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self._mean) / (self._std + 1.0e-2)


class _CheckpointOptimizer:
    def __init__(self) -> None:
        self.load_calls = 0

    def state_dict(self):
        raise AssertionError("proposal-only HSL save must not read optimizer state")

    def load_state_dict(self, _state) -> None:
        self.load_calls += 1


class _IntentCommand:
    def __init__(self, batch) -> None:
        self.batch = batch

    def frontres_local_scenario_intent_snapshot(self):
        intent = getattr(self.batch, "frontres_local_scenario_intent_q29", None)
        provenance = getattr(self.batch, "frontres_local_scenario_provenance", None)
        if not isinstance(intent, torch.Tensor) or provenance is None:
            raise RuntimeError("v015 sealed local scenario intent snapshot is unavailable")
        batch_size = int(intent.shape[0])
        return {
            "intent_q29": intent.detach().clone(),
            "scenario_ids": tuple(f"hsl-scenario-{row}" for row in range(batch_size)),
            "noisy_segment_hashes": tuple(f"hsl-hash-{row}" for row in range(batch_size)),
            "x_t_identities": tuple(f"hsl-x-{row}" for row in range(batch_size)),
            "roles": ("repair",) * batch_size,
            "provenance": tuple(dict(value) for value in provenance),
        }


class _HSLProposalCommand:
    def __init__(self, snapshot) -> None:
        self.snapshot = snapshot
        self.calls: list[tuple[int, ...]] = []
        batch_size = int(snapshot["intent_q29"].shape[0])
        self.anchor_dr_delta_pos = torch.zeros(batch_size, 3)
        self.anchor_dr_delta_quat_correction = torch.zeros(batch_size, 4)
        self.anchor_dr_delta_quat_correction[:, 0] = 1.0

    def frontres_hsl_proposal_intent_snapshot(self, future_offsets):
        self.calls.append(tuple(int(value) for value in future_offsets))
        return self.snapshot

    def frontres_local_scenario_intent_snapshot(self):
        raise AssertionError("HSL proposal route must not read the Stage-3 local scenario")


class _Env:
    def __init__(self, command) -> None:
        self.unwrapped = self
        self.command_manager = SimpleNamespace(get_term=lambda name: command if name == "motion" else None)


def _q29_runner(layout, runtime, *, provenance=None):
    raw_dim = 5
    gmt_dim = 3
    intent = _intent()
    batch = SimpleNamespace(
        frontres_local_scenario_intent_q29=intent,
        frontres_local_scenario_provenance=_provenance(intent.shape[0]) if provenance is None else provenance,
        frontres_future_offsets=(1, 2),
    )
    normalizer = _Normalizer()
    runner = SimpleNamespace(
        env=_Env(_IntentCommand(batch)),
        device=torch.device("cpu"),
        alg=SimpleNamespace(
            policy=SimpleNamespace(
                num_actor_obs=raw_dim + layout.actor_tail_dim,
                num_frontres_obs=(raw_dim - gmt_dim) + layout.actor_tail_dim,
            )
        ),
        _frontres_gmt_obs_dim=gmt_dim,
        _frontres_future_intent_layout=layout,
        _frontres_future_intent_actor_context_dim=layout.actor_tail_dim,
        _frontres_segment_live_current_batch=batch,
    )
    runner._append_frontres_future_intent_context = (
        lambda obs: runtime.append_frontres_future_intent_context(runner, obs)
    )
    runner._apply_obs_normalizer = normalizer
    return runner, normalizer, intent


def test_t_hsl_proposal_command_carrier() -> None:
    helper = _load("frontres_hsl_v007_reset_helper", RESET_HELPER_PATH)
    commands, _hooks, _setup = helper._load_owners()
    command = helper._make_command(commands, helper._FakeRobot(num_envs=3), num_envs=3)
    command.motion_lengths_minus_one = command.motion_lengths - 1
    command.time_steps[:] = torch.tensor([2, 4, 6], dtype=torch.long)
    command._cached_perturbed_pos[:] = torch.tensor(
        [[2.2, 3.0, 4.0], [4.0, 5.3, 6.0], [6.0, 7.0, 7.6]], dtype=torch.float32
    )

    first = command.frontres_hsl_proposal_intent_snapshot((1, 2))
    assert set(first) == {
        "intent_q29",
        "proposal_context_ids",
        "current_root_artifact_ids",
        "motion_indices",
        "frame_indices",
        "future_offsets",
        "provenance",
    }
    assert tuple(first["intent_q29"].shape) == (3, 3, 29)
    expected_frames = torch.tensor([[2.0, 3.0, 4.0], [4.0, 5.0, 6.0], [6.0, 7.0, 8.0]])
    torch.testing.assert_close(first["intent_q29"][:, :, 0], expected_frames)
    assert first["future_offsets"] == (1, 2)
    assert first["motion_indices"] == (0, 0, 0)
    assert first["frame_indices"] == (2, 4, 6)
    assert all(len(value) == 64 for value in first["proposal_context_ids"])
    assert all(len(value) == 64 for value in first["current_root_artifact_ids"])
    assert all(value["carrier_kind"] == "hsl_proposal" for value in first["provenance"])
    assert all("clean_continuation_provenance" not in value for value in first["provenance"])
    assert not ({"x_t", "clean_continuation", "horizon_k", "roles"} & set(first))

    first["intent_q29"].fill_(-999.0)
    second = command.frontres_hsl_proposal_intent_snapshot((1, 2))
    assert not bool((second["intent_q29"] == -999.0).any())
    assert second["proposal_context_ids"] == first["proposal_context_ids"]
    assert second["current_root_artifact_ids"] == first["current_root_artifact_ids"]

    command.time_steps[0] = 31
    _expect_error(ValueError, lambda: command.frontres_hsl_proposal_intent_snapshot((1, 2)), "cannot clamp")
    command.time_steps[0] = 2
    command._frontres_local_scenario_active[0] = True
    _expect_error(RuntimeError, lambda: command.frontres_hsl_proposal_intent_snapshot((1, 2)), "cannot mix")
    print("[T-HSL-carrier/T-immutability/T-no-C-K] command-owned proposal q29 snapshot is isolated", flush=True)


def test_t_hsl_proposal_runtime_route() -> None:
    layout_module, runtime, warmup = _load_q29_modules()
    layout = layout_module.resolve_frontres_future_intent_layout(
        (1, 2), layout_module.FRONTRES_FUTURE_INTENT_LAYOUT_VERSION
    )
    intent = _intent()
    provenance = tuple(
        {
            "carrier_kind": "hsl_proposal",
            "current_root_artifact_provenance": "noisy_root_artifact_t",
            "intent_q29_provenance": "deployment_noisy_q29",
            "intent_q29_source": "motion_internal_q29",
        }
        for _ in range(int(intent.shape[0]))
    )
    snapshot = {
        "intent_q29": intent,
        "proposal_context_ids": ("a" * 64, "b" * 64),
        "current_root_artifact_ids": ("c" * 64, "d" * 64),
        "motion_indices": (0, 0),
        "frame_indices": (2, 4),
        "future_offsets": (1, 2),
        "provenance": provenance,
    }
    command = _HSLProposalCommand(snapshot)
    normalizer = _Normalizer()
    runner = SimpleNamespace(
        env=_Env(command),
        device=torch.device("cpu"),
        alg=SimpleNamespace(policy=SimpleNamespace(num_actor_obs=63, num_frontres_obs=60)),
        _frontres_gmt_obs_dim=3,
        _frontres_future_intent_layout=layout,
        _frontres_future_intent_actor_context_dim=layout.actor_tail_dim,
        _frontres_hsl_proposal_context_enabled=True,
    )
    runner._append_frontres_future_intent_context = (
        lambda obs: runtime.append_frontres_future_intent_context(runner, obs)
    )
    runner._apply_obs_normalizer = normalizer
    raw_obs = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    prepared = warmup.prepare_frontres_hsl_actor_observation(runner, raw_obs)
    assert command.calls == [(1, 2)]
    assert tuple(prepared.shape) == (2, 63)
    torch.testing.assert_close(normalizer.calls[0][:, :58], intent[:, (1, 2), :].reshape(2, 58))
    torch.testing.assert_close(normalizer.calls[0][:, 58:], raw_obs)
    print("[T-HSL-runtime/T-local-isolation] proposal snapshot reuses the q29 bridge without Segment state", flush=True)


def test_t_hsl_formal_stage1_config_and_layout() -> None:
    apply_preset, configure_env = _load_stage1_preset()
    alg = SimpleNamespace(
        frontres_training_objective="legacy",
        lambda_supervised=1.0,
        lambda_supervised_min=1.0,
        frontres_segment_replay_enabled=True,
        frontres_formal_transaction_enabled=True,
        frontres_future_offsets=(),
        frontres_future_intent_layout_version="",
        frontres_hsl_rollout_label_enabled=True,
        frontres_formal_runtime_audit=False,
    )
    policy = SimpleNamespace(num_frontres_obs=0)
    agent = SimpleNamespace(
        algorithm=alg,
        policy=policy,
        experiment_name="unset",
        frontres_stage1_exit_after_warmup=False,
        critic_warmup_iterations=99,
        frontres_warmup_energy_loss_weight=1.0,
        frontres_hsl_live_smoke_enabled=False,
        supervised_warmup_iterations=200,
        supervised_warmup_steps_per_iter=8,
        max_iterations=800,
        resume=False,
    )
    args = SimpleNamespace(
        frontres_stage="stage1_hsl",
        experiment_name=None,
        frontres_v015_future_offsets=None,
        frontres_segment_live_sentinel_only=False,
        frontres_local_sentinel_only=False,
        frontres_segment_live_probe_only=False,
        frontres_segment_live_storage_write_only=False,
        frontres_segment_live_single_update_only=False,
        frontres_segment_live_update_loop_only=False,
        frontres_hsl_live_smoke=False,
        num_envs=8,
    )
    apply_preset(agent, args)
    assert agent.frontres_stage1_exit_after_warmup is True
    assert agent.frontres_warmup_energy_loss_weight == 0.0
    assert agent.critic_warmup_iterations == 0
    assert alg.frontres_training_objective == "supervised_restore"
    assert alg.frontres_segment_replay_enabled is False
    assert alg.frontres_formal_transaction_enabled is False
    assert alg.frontres_future_offsets == (1, 2)
    assert alg.frontres_future_intent_layout_version == "frontres-v015-future-intent-q29-v1"
    assert alg.lambda_supervised == 0.0
    assert alg.lambda_supervised_min == 0.0
    assert alg.frontres_hsl_rollout_label_enabled is False
    assert policy.num_frontres_obs == 100

    motion_cfg = SimpleNamespace(frontres_required_future_frames=0)
    env_cfg = SimpleNamespace(commands=SimpleNamespace(motion=motion_cfg))
    configure_env(env_cfg, args)
    assert motion_cfg.frontres_required_future_frames == 2

    commands, _hooks, _setup = _load("frontres_hsl_frame_budget_helper", RESET_HELPER_PATH)._load_owners()
    ceilings = commands._frontres_sample_frame_ceiling(torch.tensor([376, 2]), 2)
    assert ceilings.tolist() == [374, 0]
    torch.testing.assert_close(
        commands._frontres_sample_frame_ceiling(torch.tensor([376, 2]), 0),
        torch.tensor([376, 2]),
    )
    _expect_error(
        ValueError,
        lambda: commands._frontres_sample_frame_ceiling(torch.tensor([1]), 2),
        "too short",
    )
    # max_frame=221 and H=2: frame 219 is the last valid current frame.
    # The flag is consumed by the following env.step, before frame 220 can be observed.
    hsl_end = commands._frontres_motion_end_mask(
        torch.tensor([218, 219]),
        torch.tensor([222, 222]),
        2,
    )
    assert hsl_end.tolist() == [False, True]
    ordinary_end = commands._frontres_motion_end_mask(
        torch.tensor([220, 221, 222]),
        torch.tensor([222, 222, 222]),
        0,
    )
    assert ordinary_end.tolist() == [False, False, True]
    fake_clock = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        cfg=SimpleNamespace(frontres_required_future_frames=2),
        motion_lengths=torch.tensor([222]),
        env_motion_indices=torch.tensor([0]),
        time_steps=torch.tensor([218]),
        motion_end_buf=torch.zeros(1, dtype=torch.bool),
    )
    commands.MultiMotionCommand._refresh_frontres_motion_end_buf(fake_clock)
    assert fake_clock.motion_end_buf.tolist() == [False]
    fake_clock.time_steps += 1
    commands.MultiMotionCommand._refresh_frontres_motion_end_buf(fake_clock)
    assert fake_clock.time_steps.tolist() == [219]
    assert fake_clock.motion_end_buf.tolist() == [True]
    command_source = COMMANDS_PATH.read_text()
    assert "sample_frame_ceiling = _frontres_sample_frame_ceiling(" in command_source
    assert "time_steps = torch.minimum(time_steps, sample_frame_ceiling)" in command_source
    assert command_source.count("self._refresh_frontres_motion_end_buf(") == 2
    resample_source = inspect.getsource(commands.MultiMotionCommand._resample_command)
    assert resample_source.index("_sync_frontres_pairs") < resample_source.index(
        "_refresh_frontres_motion_end_buf"
    )
    print(
        "[T-HSL-frame-budget] Stage-1 sampling and motion end preserve two real future frames",
        flush=True,
    )

    args.frontres_hsl_live_smoke = True
    agent.max_iterations = 0
    agent.supervised_warmup_iterations = 1
    agent.supervised_warmup_steps_per_iter = 1
    apply_preset(agent, args)
    assert agent.frontres_hsl_live_smoke_enabled is True
    assert alg.frontres_formal_runtime_audit is True

    agent.max_iterations = 1
    _expect_error(ValueError, lambda: apply_preset(agent, args), "--max_iterations 0")
    agent.max_iterations = 0
    args.num_envs = 3
    _expect_error(ValueError, lambda: apply_preset(agent, args), "even --num_envs")
    args.num_envs = 8
    args.frontres_hsl_live_smoke = False

    args.frontres_v015_future_offsets = "1,3"
    _expect_error(ValueError, lambda: apply_preset(agent, args), "exactly '1,2'")

    runner_source = ON_POLICY_PATH.read_text()
    assert "self._frontres_hsl_proposal_context_enabled" in runner_source
    assert "or self._frontres_hsl_proposal_context_enabled" in runner_source
    assert "obs = self._append_frontres_future_intent_context(obs)" in runner_source
    assert "privileged_obs = privileged_obs.detach()" in runner_source
    assert "teacher_obs = teacher_obs.detach()" in runner_source
    print(
        "[T-HSL-config/T-formal-layout] stage1 preset selects 870+58=928 -> FEMR 158 / GMT 770",
        flush=True,
    )


def test_t_hsl_actor_only_critic_unchanged() -> None:
    _layout, _runtime, warmup = _load_q29_modules()
    torch.manual_seed(17)
    policy = SimpleNamespace(
        residual_actor=torch.nn.Linear(4, 6, bias=False),
        critic=torch.nn.Linear(3, 1, bias=False),
    )
    actor_before = tuple(value.detach().clone() for value in policy.residual_actor.parameters())
    critic_before = warmup.capture_frontres_hsl_critic_state(policy)
    optimizer = warmup.build_frontres_hsl_actor_only_optimizer(policy, learning_rate=1.0e-2)
    prediction = policy.residual_actor(torch.ones(2, 4))
    loss = prediction.square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert any(
        not torch.equal(value, before)
        for value, before in zip(policy.residual_actor.parameters(), actor_before)
    )
    assert all(value.grad is None for value in policy.critic.parameters())
    warmup.assert_frontres_hsl_critic_unchanged(policy, critic_before)

    with torch.no_grad():
        next(policy.critic.parameters()).add_(1.0)
    _expect_error(
        RuntimeError,
        lambda: warmup.assert_frontres_hsl_critic_unchanged(policy, critic_before),
        "critic changed",
    )
    warmup_source = WARMUP_PATH.read_text()
    run_source = warmup_source[warmup_source.index("def run_frontres_joint_warmup") :]
    assert "capture_frontres_hsl_critic_state(self.alg.policy)" in run_source
    assert "build_frontres_hsl_actor_only_optimizer(" in run_source
    assert "assert_frontres_hsl_critic_unchanged(self.alg.policy" in run_source
    assert "policy.evaluate(" not in run_source
    assert "energy_target" not in run_source
    assert "policy.critic.parameters()" not in run_source
    assert "_sup_mask" not in run_source
    assert "frontres_active_task_dims" not in run_source
    print("[T-HSL-actor-only/T-critic-unchanged] actor updates while critic has zero grad and zero delta", flush=True)


def test_t_hsl_legacy_checkpoint_reject() -> None:
    checkpointing = _load_checkpointing()
    v015_runner = SimpleNamespace(
        _frontres_future_intent_layout=object(),
        _frontres_future_intent_actor_context_dim=58,
    )
    legacy_payload = {"frontres_warmup_complete": True, "model_state_dict": {}}
    _expect_error(
        RuntimeError,
        lambda: checkpointing.reject_legacy_frontres_hsl_checkpoint(v015_runner, legacy_payload),
        "FRS-TRAIN-v007",
    )
    checkpointing.reject_legacy_frontres_hsl_checkpoint(
        SimpleNamespace(_frontres_future_intent_layout=None, _frontres_future_intent_actor_context_dim=0),
        legacy_payload,
    )
    checkpoint_source = CHECKPOINT_PATH.read_text()
    load_start = checkpoint_source.index("def load_runner")
    load_end = checkpoint_source.index("# B2:", load_start)
    assert "reject_legacy_frontres_hsl_checkpoint(self, loaded_dict)" in checkpoint_source[load_start:load_end]
    print("[T-HSL-legacy-checkpoint-reject] v015 route rejects old warmup payload before state restoration", flush=True)


def _hsl_checkpoint_runner(checkpointing, layout, gmt_path: Path, *, seed: int):
    torch.manual_seed(seed)
    policy = checkpointing.FrontRESActorCritic.__new__(checkpointing.FrontRESActorCritic)
    torch.nn.Module.__init__(policy)
    policy.residual_actor = torch.nn.Sequential(
        torch.nn.Linear(158, 12),
        torch.nn.ELU(),
        torch.nn.Linear(12, 6),
    )
    policy.critic = torch.nn.Linear(5, 1)
    policy.register_buffer("std", torch.full((6,), 0.05 + seed * 0.001))
    policy.num_actor_obs = 928
    policy.num_frontres_obs = 158
    policy.num_task_corrections = 6
    policy.total_output_dim = 6
    policy.gmt_policy_obs_dim = 770
    policy.gmt_normalizer = _CheckpointNormalizer(770, value=3.0)
    optimizer = _CheckpointOptimizer()
    runner = SimpleNamespace(
        alg=SimpleNamespace(
            policy=policy,
            optimizer=optimizer,
            frontres_training_objective="supervised_restore",
        ),
        policy_cfg={
            "gmt_checkpoint_path": str(gmt_path),
            "noise_std_type": "scalar",
            "init_noise_std": 0.01,
        },
        cfg={"is_full_resume": False},
        device=torch.device("cpu"),
        training_type="frontres",
        empirical_normalization=True,
        _frontres_hsl_proposal_context_enabled=True,
        _frontres_future_intent_layout=layout,
        _frontres_future_intent_actor_context_dim=58,
        _frontres_gmt_obs_dim=770,
        _frontres_extra_normalizer=_CheckpointNormalizer(158, value=float(seed)),
        _frontres_extra_mean=None,
        _frontres_extra_std=None,
        _frontres_extra_stats_layout_version=None,
        obs_normalizer=policy.gmt_normalizer,
        privileged_obs_normalizer=_CheckpointNormalizer(5, value=19.0),
        current_learning_iteration=13,
        logger_type="",
        writer=None,
        disable_logs=True,
    )
    return runner


def _stage3_hsl_initializer_runner(checkpointing, layout, gmt_path: Path, *, seed: int):
    runner = _hsl_checkpoint_runner(checkpointing, layout, gmt_path, seed=seed)
    runner._frontres_hsl_proposal_context_enabled = False
    runner.alg.frontres_training_objective = "segment_replay_hrl"
    runner.alg.frontres_formal_transaction_enabled = True
    runner.alg.frontres_segment_advantage_normalization = "grouped_scale_only"
    runner.alg.frontres_future_offsets = (1, 2)
    runner.alg.frontres_future_intent_layout_version = "frontres-v015-future-intent-q29-v1"
    runner.alg.frontres_hsl_init_enabled = False
    runner.alg.frontres_hsl_rollout_label_enabled = False
    runner.alg.lambda_supervised = 0.0
    runner.alg.lambda_supervised_min = 0.0
    runner.current_learning_iteration = 0
    runner._frontres_segment_sampler = object()
    return runner


def _load_hsl_fresh_connectivity_owners():
    checkpointing = _load_checkpointing()
    rsl_rl = sys.modules["rsl_rl"]
    frontres = _package("rsl_rl.frontres")
    runners = _package("rsl_rl.runners")
    rsl_rl.frontres = frontres
    rsl_rl.runners = runners
    diagnostics = types.ModuleType("rsl_rl.frontres.runtime_diagnostics")
    diagnostics.maybe_print_frontres_restore_debug = lambda *_args, **_kwargs: None
    sys.modules[diagnostics.__name__] = diagnostics
    frontres.runtime_diagnostics = diagnostics
    runtime = _load("rsl_rl.runners.frontres_runtime_hsl_fresh_contract", RUNTIME_PATH)
    warmup = _load("frontres_warmup_hsl_fresh_contract", WARMUP_PATH)
    layout_module = sys.modules["rsl_rl.modules.frontres_observation_layout"]
    return checkpointing, layout_module, runtime, warmup


def _hsl_proposal_snapshot(intent: torch.Tensor) -> dict[str, object]:
    batch_size = int(intent.shape[0])
    provenance = tuple(
        {
            "carrier_kind": "hsl_proposal",
            "current_root_artifact_provenance": "noisy_root_artifact_t",
            "intent_q29_provenance": "deployment_noisy_q29",
            "intent_q29_source": "motion_internal_q29",
        }
        for _ in range(batch_size)
    )
    return {
        "intent_q29": intent.detach().clone(),
        "proposal_context_ids": tuple(f"fresh-context-{row}" for row in range(batch_size)),
        "current_root_artifact_ids": tuple(f"artifact-{row}" for row in range(batch_size)),
        "motion_indices": tuple(range(batch_size)),
        "frame_indices": tuple(10 + row for row in range(batch_size)),
        "future_offsets": (1, 2),
        "provenance": provenance,
    }


def _wire_hsl_fresh_runner(runner, runtime, snapshot: dict[str, object]) -> _HSLProposalCommand:
    command = _HSLProposalCommand(copy.deepcopy(snapshot))
    runner.env = _Env(command)

    def append_context(obs: torch.Tensor) -> torch.Tensor:
        combined = runtime.append_frontres_future_intent_context(runner, obs)
        runner._fresh_trace_combined = combined.detach().clone()
        return combined

    def normalize(obs: torch.Tensor) -> torch.Tensor:
        normalized = runtime.apply_obs_normalizer(runner, obs)
        runner._fresh_trace_normalized = normalized.detach().clone()
        return normalized

    runner._append_frontres_future_intent_context = append_context
    runner._apply_obs_normalizer = normalize
    return command


def _hsl_fresh_trace(runner, warmup, raw_obs: torch.Tensor) -> dict[str, torch.Tensor]:
    normalized = warmup.prepare_frontres_hsl_actor_observation(runner, raw_obs)
    actor_input = normalized[:, :158]
    raw_proposal = runner.alg.policy.residual_actor(actor_input)
    proposal = raw_proposal
    return {
        "combined": runner._fresh_trace_combined.detach().clone(),
        "normalized": runner._fresh_trace_normalized.detach().clone(),
        "actor_input": actor_input.detach().clone(),
        "proposal": proposal.detach().clone(),
    }


def _checkpoint_mutable_state(runner) -> dict[str, object]:
    return {
        "actor": tuple(value.detach().clone() for value in runner.alg.policy.residual_actor.state_dict().values()),
        "critic": tuple(value.detach().clone() for value in runner.alg.policy.critic.state_dict().values()),
        "std": runner.alg.policy.std.detach().clone(),
        "prefix": tuple(
            value.detach().clone() for value in runner._frontres_extra_normalizer.state_dict().values()
        ),
        "privileged": tuple(
            value.detach().clone() for value in runner.privileged_obs_normalizer.state_dict().values()
        ),
        "optimizer_load_calls": runner.alg.optimizer.load_calls,
        "warmup_complete": bool(getattr(runner, "_frontres_warmup_complete", False)),
        "last_loaded_path": getattr(runner, "_frontres_last_loaded_checkpoint_path", None),
    }


def _assert_checkpoint_state_equal(actual: dict[str, object], expected: dict[str, object]) -> None:
    for name in ("actor", "critic", "prefix", "privileged"):
        assert len(actual[name]) == len(expected[name])
        assert all(torch.equal(lhs, rhs) for lhs, rhs in zip(actual[name], expected[name]))
    assert torch.equal(actual["std"], expected["std"])
    assert actual["optimizer_load_calls"] == expected["optimizer_load_calls"]
    assert actual["warmup_complete"] == expected["warmup_complete"]
    assert actual["last_loaded_path"] == expected["last_loaded_path"]


def test_t_hsl_checkpoint_identity_and_pre_mutation() -> None:
    checkpointing = _load_checkpointing()
    layout_module = sys.modules["rsl_rl.modules.frontres_observation_layout"]
    layout = layout_module.resolve_frontres_future_intent_layout(
        (1, 2), layout_module.FRONTRES_FUTURE_INTENT_LAYOUT_VERSION
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        gmt_path = root / "gmt.pt"
        torch.save({"artifact": "frozen-gmt"}, gmt_path)
        checkpoint_path = root / "hsl.pt"
        source = _hsl_checkpoint_runner(checkpointing, layout, gmt_path, seed=3)
        checkpointing.save_runner(source, str(checkpoint_path))
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert set(payload) == {
            "frontres_v015_hsl_checkpoint_identity",
            "model_state_dict",
            "frontres_prefix_norm_state_dict",
        }
        assert set(payload["model_state_dict"]) == {"residual_actor", "std"}
        identity = payload["frontres_v015_hsl_checkpoint_identity"]
        assert identity["format"] == "frontres-v017-hsl-proposal-v2"
        assert identity["method_contract_id"] == "FRS-METHOD-v017"
        assert identity["training_contract_id"] == "FRS-TRAIN-v014"
        inspected = checkpointing.inspect_frontres_quality_checkpoint(checkpoint_path, route="hsl")
        assert inspected.training_contract_id == "FRS-TRAIN-v014"
        assert identity["future_intent_layout"]["actor_dim"] == 928
        assert identity["future_intent_layout"]["prefix_dim"] == 158
        assert identity["future_intent_layout"]["gmt_dim"] == 770
        assert identity["action"] == {
            "kind": "delta_se3",
            "dim": 6,
            "semantics": "direct-world-full6-v1",
        }
        forbidden = {
            "critic",
            "privileged_obs_norm_state_dict",
            "optimizer_state_dict",
            "frontres_segment_sampler_state_dict",
            "frontres_v015_checkpoint_identity",
            "transaction",
            "frontres_gain_config",
            "frontres_warmup_complete",
        }
        assert not (forbidden & set(payload))
        assert not (forbidden & set(payload["model_state_dict"]))

        target = _hsl_checkpoint_runner(checkpointing, layout, gmt_path, seed=11)
        protected_before = _checkpoint_mutable_state(target)
        checkpointing.load_runner(target, str(checkpoint_path), load_optimizer=True, load_critic=True)
        source_state = _checkpoint_mutable_state(source)
        loaded_state = _checkpoint_mutable_state(target)
        for name in ("actor", "prefix"):
            assert all(torch.equal(lhs, rhs) for lhs, rhs in zip(loaded_state[name], source_state[name]))
        assert torch.equal(loaded_state["std"], source_state["std"])
        assert all(
            torch.equal(lhs, rhs)
            for lhs, rhs in zip(loaded_state["critic"], protected_before["critic"])
        )
        assert all(
            torch.equal(lhs, rhs)
            for lhs, rhs in zip(loaded_state["privileged"], protected_before["privileged"])
        )
        assert target.alg.optimizer.load_calls == 0

        tamper_cases = []
        forbidden_payload = copy.deepcopy(payload)
        forbidden_payload["optimizer_state_dict"] = {"forbidden": True}
        tamper_cases.append(("forbidden-key", forbidden_payload, "exact payload"))
        actor_tamper = copy.deepcopy(payload)
        next(iter(actor_tamper["model_state_dict"]["residual_actor"].values())).add_(1.0)
        tamper_cases.append(("actor-tamper", actor_tamper, "fingerprint"))
        prefix_tamper = copy.deepcopy(payload)
        prefix_tamper["frontres_prefix_norm_state_dict"]["_mean"][0, 0] += 1.0
        tamper_cases.append(("prefix-tamper", prefix_tamper, "fingerprint"))
        gmt_tamper = copy.deepcopy(payload)
        gmt_tamper["frontres_v015_hsl_checkpoint_identity"]["gmt"]["checkpoint_sha256"] = "0" * 64
        tamper_cases.append(("gmt-tamper", gmt_tamper, "GMT"))
        stage3_identity_tamper = copy.deepcopy(payload)
        stage3_identity_tamper["frontres_v015_hsl_checkpoint_identity"]["training_contract_id"] = "FRS-TRAIN-v015"
        tamper_cases.append(("stage3-identity", stage3_identity_tamper, "incompatible identity"))
        legacy_payload = {
            "model_state_dict": copy.deepcopy(payload["model_state_dict"]),
            "frontres_warmup_complete": True,
        }
        tamper_cases.append(("legacy", legacy_payload, "identity"))
        old_hsl_v1 = copy.deepcopy(payload)
        old_hsl_v1["frontres_v015_hsl_checkpoint_identity"]["format"] = "frontres-v015-hsl-proposal-v1"
        old_hsl_v1["frontres_v015_hsl_checkpoint_identity"]["method_contract_id"] = "FRS-METHOD-v015"
        old_hsl_v1["frontres_v015_hsl_checkpoint_identity"]["training_contract_id"] = "FRS-TRAIN-v007"
        old_hsl_v1["frontres_v015_hsl_checkpoint_identity"]["action"] = {"kind": "delta_se3", "dim": 6}
        tamper_cases.append(("old-hsl-v1", old_hsl_v1, "incompatible identity"))

        for name, bad_payload, message in tamper_cases:
            bad_path = root / f"{name}.pt"
            torch.save(bad_payload, bad_path)
            rejected = _hsl_checkpoint_runner(checkpointing, layout, gmt_path, seed=21)
            before = _checkpoint_mutable_state(rejected)
            _expect_error(
                RuntimeError,
                lambda p=bad_path, r=rejected: checkpointing.load_runner(
                    r, str(p), load_optimizer=True, load_critic=True
                ),
                message,
            )
            _assert_checkpoint_state_equal(_checkpoint_mutable_state(rejected), before)

    print(
        "[T-HSL-checkpoint/T-pre-mutation] strict actor/std/prefix payload reloads; forbidden/tampered/legacy reject unchanged",
        flush=True,
    )


def test_t_stage3_explicit_hsl_initializer_actor_only() -> None:
    checkpointing = _load_checkpointing()
    layout_module = sys.modules["rsl_rl.modules.frontres_observation_layout"]
    layout = layout_module.resolve_frontres_future_intent_layout(
        (1, 2), layout_module.FRONTRES_FUTURE_INTENT_LAYOUT_VERSION
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        gmt_path = root / "gmt.pt"
        torch.save({"artifact": "frozen-gmt"}, gmt_path)
        checkpoint_path = root / "hsl.pt"
        source = _hsl_checkpoint_runner(checkpointing, layout, gmt_path, seed=31)
        checkpointing.save_runner(source, str(checkpoint_path))

        target = _stage3_hsl_initializer_runner(checkpointing, layout, gmt_path, seed=47)
        protected_before = _checkpoint_mutable_state(target)
        sampler_before = target._frontres_segment_sampler

        _expect_error(
            RuntimeError,
            lambda: checkpointing.load_runner(target, str(checkpoint_path), load_optimizer=False, load_critic=False),
            "active Stage-1 HSL route",
        )
        _assert_checkpoint_state_equal(_checkpoint_mutable_state(target), protected_before)

        receipt = checkpointing.load_frontres_hsl_initializer(target, str(checkpoint_path))
        source_state = _checkpoint_mutable_state(source)
        loaded_state = _checkpoint_mutable_state(target)
        for name in ("actor", "prefix"):
            assert all(torch.equal(lhs, rhs) for lhs, rhs in zip(loaded_state[name], source_state[name]))
        assert torch.equal(loaded_state["std"], source_state["std"])
        assert all(
            torch.equal(lhs, rhs)
            for lhs, rhs in zip(loaded_state["critic"], protected_before["critic"])
        )
        assert all(
            torch.equal(lhs, rhs)
            for lhs, rhs in zip(loaded_state["privileged"], protected_before["privileged"])
        )
        assert target.alg.optimizer.load_calls == 0
        assert target._frontres_segment_sampler is sampler_before
        assert not hasattr(target, "_frontres_checkpoint_transaction_state")
        assert target.alg.frontres_hsl_init_enabled is False
        assert receipt["format"] == "frontres-v017-hsl-proposal-v2"
        assert receipt["restored"] == ("residual_actor", "std", "frontres_prefix_norm_state_dict")

        rejected = _stage3_hsl_initializer_runner(checkpointing, layout, gmt_path, seed=53)
        rejected.current_learning_iteration = 1
        before = _checkpoint_mutable_state(rejected)
        _expect_error(
            RuntimeError,
            lambda: checkpointing.load_frontres_hsl_initializer(rejected, str(checkpoint_path)),
            "before the first Stage-3 iteration",
        )
        _assert_checkpoint_state_equal(_checkpoint_mutable_state(rejected), before)

        rejection_cases = (
            ("full-resume", "actor initialization, not full resume", lambda runner: runner.cfg.__setitem__("is_full_resume", True)),
            ("open-HSL", "HSL flags to be closed", lambda runner: setattr(runner.alg, "frontres_hsl_init_enabled", True)),
            (
                "active-transaction",
                "existing transaction state",
                lambda runner: setattr(
                    runner,
                    "_frontres_checkpoint_transaction_state",
                    {"state": "collecting", "phase": "provider"},
                ),
            ),
        )
        for index, (_name, message, mutate) in enumerate(rejection_cases):
            case = _stage3_hsl_initializer_runner(checkpointing, layout, gmt_path, seed=60 + index)
            mutate(case)
            before = _checkpoint_mutable_state(case)
            sampler = case._frontres_segment_sampler
            _expect_error(
                RuntimeError,
                lambda r=case: checkpointing.load_frontres_hsl_initializer(r, str(checkpoint_path)),
                message,
            )
            _assert_checkpoint_state_equal(_checkpoint_mutable_state(case), before)
            assert case._frontres_segment_sampler is sampler

    print(
        "[T-Stage3-HSL-init] explicit HSL-v2 restores actor/std/158D prefix only; generic load and post-iteration migration reject unchanged",
        flush=True,
    )


def test_t_hsl_fresh_runner_connectivity() -> None:
    checkpointing, layout_module, runtime, warmup = _load_hsl_fresh_connectivity_owners()
    layout = layout_module.resolve_frontres_future_intent_layout(
        (1, 2), layout_module.FRONTRES_FUTURE_INTENT_LAYOUT_VERSION
    )
    intent = _intent(batch_size=2, hmax=2)
    snapshot = _hsl_proposal_snapshot(intent)
    raw_obs = torch.arange(2 * 870, dtype=torch.float32).reshape(2, 870) / 1000.0
    current_artifact = torch.tensor(
        [[0.12, -0.07, 0.03, 1.0, 0.01, 0.02, 0.03], [-0.09, 0.05, -0.02, 1.0, -0.02, 0.01, 0.04]]
    )
    raw_obs[:, :7] = current_artifact

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        gmt_path = root / "gmt.pt"
        torch.save({"artifact": "fresh-connectivity-gmt"}, gmt_path)
        checkpoint_path = root / "hsl.pt"

        source = _hsl_checkpoint_runner(checkpointing, layout, gmt_path, seed=5)
        source_command = _wire_hsl_fresh_runner(source, runtime, snapshot)
        source_trace = _hsl_fresh_trace(source, warmup, raw_obs)
        checkpointing.save_runner(source, str(checkpoint_path))

        fresh = _hsl_checkpoint_runner(checkpointing, layout, gmt_path, seed=23)
        fresh_command = _wire_hsl_fresh_runner(fresh, runtime, snapshot)
        fresh_before = _hsl_fresh_trace(fresh, warmup, raw_obs)
        assert not torch.equal(fresh_before["actor_input"], source_trace["actor_input"])
        assert not torch.equal(fresh_before["proposal"], source_trace["proposal"])
        critic_before = tuple(value.detach().clone() for value in fresh.alg.policy.critic.parameters())

        checkpointing.load_runner(fresh, str(checkpoint_path), load_optimizer=True, load_critic=True)
        fresh_trace = _hsl_fresh_trace(fresh, warmup, raw_obs)

        expected_tail = intent[:, (1, 2), :].reshape(2, 58)
        assert tuple(source_trace["combined"].shape) == (2, 928)
        assert tuple(source_trace["actor_input"].shape) == (2, 158)
        assert tuple(source_trace["proposal"].shape) == (2, 6)
        torch.testing.assert_close(source_trace["combined"][:, :58], expected_tail, rtol=0.0, atol=0.0)
        torch.testing.assert_close(source_trace["combined"][:, 58:65], current_artifact, rtol=0.0, atol=0.0)
        for name in ("combined", "normalized", "actor_input", "proposal"):
            torch.testing.assert_close(fresh_trace[name], source_trace[name], rtol=0.0, atol=0.0)
        assert source_command.calls == [(1, 2)]
        assert fresh_command.calls == [(1, 2), (1, 2)]
        assert all(
            torch.equal(value, before)
            for value, before in zip(fresh.alg.policy.critic.parameters(), critic_before)
        )
        assert fresh.alg.optimizer.load_calls == 0
        assert fresh._frontres_extra_mean is None and fresh._frontres_extra_std is None
        assert fresh._frontres_extra_stats_layout_version is None

    print(
        "[T-HSL-fresh-runner/T-output/T-zero-leak] fixed artifact+q29 -> 928 -> normalized 158 -> 6D proposal is exact after reload",
        flush=True,
    )


def test_t_hsl_live_smoke_connector() -> None:
    checkpointing, layout_module, runtime, warmup = _load_hsl_fresh_connectivity_owners()
    layout = layout_module.resolve_frontres_future_intent_layout(
        (1, 2), layout_module.FRONTRES_FUTURE_INTENT_LAYOUT_VERSION
    )
    intent = _intent(batch_size=2, hmax=2)
    snapshot = _hsl_proposal_snapshot(intent)
    raw_obs = torch.arange(2 * 870, dtype=torch.float32).reshape(2, 870) / 1000.0

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        gmt_path = root / "gmt.pt"
        torch.save({"artifact": "live-smoke-gmt"}, gmt_path)
        checkpoint_path = root / "hsl.pt"
        source = _hsl_checkpoint_runner(checkpointing, layout, gmt_path, seed=5)
        source._frontres_hsl_live_smoke_enabled = True
        _wire_hsl_fresh_runner(source, runtime, snapshot)
        shadow = checkpointing.capture_frontres_hsl_fresh_reload_shadow(source)
        tolerance_shadow = checkpointing.capture_frontres_hsl_fresh_reload_shadow(source)
        reject_shadow = checkpointing.capture_frontres_hsl_fresh_reload_shadow(source)
        with torch.no_grad():
            next(source.alg.policy.residual_actor.parameters()).add_(0.125)
            source._frontres_extra_normalizer._mean.add_(0.25)
        source_trace = _hsl_fresh_trace(source, warmup, raw_obs)
        checkpointing.save_runner(source, str(checkpoint_path))
        result = checkpointing.verify_frontres_hsl_fresh_reload(
            shadow,
            checkpoint_path=str(checkpoint_path),
            combined_obs=source_trace["combined"],
            source_actor_input=source_trace["actor_input"],
            source_proposal=source_trace["proposal"],
        )
        assert result["normalized_158_equal"] is True
        assert result["proposal_6_close"] is True
        assert result["proposal_6_bitwise_equal"] is True
        assert result["proposal_6_max_abs_error"] == 0.0
        assert result["pre_reload_proposal_equal"] is False
        assert not hasattr(shadow.alg.policy, "critic")
        assert not hasattr(shadow.alg, "optimizer")
        assert tuple(source._frontres_hsl_smoke_combined_obs.shape) == (2, 928)
        assert tuple(source._frontres_hsl_smoke_normalized_obs[:, :158].shape) == (2, 158)

        near_proposal = torch.nextafter(
            source_trace["proposal"],
            torch.full_like(source_trace["proposal"], float("inf")),
        )
        near_result = checkpointing.verify_frontres_hsl_fresh_reload(
            tolerance_shadow,
            checkpoint_path=str(checkpoint_path),
            combined_obs=source_trace["combined"],
            source_actor_input=source_trace["actor_input"],
            source_proposal=near_proposal,
        )
        assert near_result["proposal_6_close"] is True
        assert near_result["proposal_6_bitwise_equal"] is False
        direct_tolerance = 1.0e-6 + 1.0e-5 * float(source_trace["proposal"].abs().max())
        assert 0.0 < near_result["proposal_6_max_abs_error"] <= direct_tolerance

        far_proposal = source_trace["proposal"] + torch.full_like(source_trace["proposal"], 1.0e-3)
        _expect_error(
            RuntimeError,
            lambda: checkpointing.verify_frontres_hsl_fresh_reload(
                reject_shadow,
                checkpoint_path=str(checkpoint_path),
                combined_obs=source_trace["combined"],
                source_actor_input=source_trace["actor_input"],
                source_proposal=far_proposal,
            ),
            "max_abs_error",
        )

    warmup_source = WARMUP_PATH.read_text()
    for sentinel in (
        "[G2-S4-INPUT]",
        "[G2-S4-OBS]",
        "[G2-S4-TARGET]",
        "[G2-S4-GRAD]",
        "[G2-S4-CRITIC]",
        "[G2-S4-COMPLETE]",
    ):
        assert sentinel in warmup_source
    print(
        "[T-HSL-live-smoke/T-telemetry/T-shadow-reload] exact state/input plus numerical cross-device proposal tolerance",
        flush=True,
    )


def test_t_hsl_loss_reject() -> None:
    unified = _load_unified_guard()
    _expect_error(
        ValueError,
        lambda: unified.validate_frontres_v015_stage3_supervision_config(
            future_offsets=(1, 3), lambda_supervised=0.0, lambda_supervised_min=0.0
        ),
        "exact deployment offsets (1, 2)",
    )
    _expect_error(
        ValueError,
        lambda: unified.validate_frontres_v015_stage3_supervision_config(
            future_offsets=(1, 2), lambda_supervised=1.0, lambda_supervised_min=0.0
        ),
        "FRS-TRAIN-v018",
    )
    _expect_error(
        ValueError,
        lambda: unified.validate_frontres_v015_stage3_supervision_config(
            future_offsets=(1, 2), lambda_supervised=0.0, lambda_supervised_min=0.2
        ),
        "FRS-TRAIN-v018",
    )
    unified.validate_frontres_v015_stage3_supervision_config(
        future_offsets=(1, 2), lambda_supervised=0.0, lambda_supervised_min=0.0
    )
    unified_source = UNIFIED_PATH.read_text()
    init_start = unified_source.index("def __init__(")
    init_end = unified_source.index("self.frontres_segment_max_horizon_k", init_start)
    assert "validate_frontres_v015_stage3_supervision_config(" in unified_source[init_start:init_end]
    config = G1_CFG_PATH.read_text()
    assert "lambda_supervised             = 0.0" in config
    assert "lambda_supervised_min         = 0.0" in config
    print("[T-HSL-loss-reject] v015 rejects nonzero online supervised loss and floor", flush=True)


def test_t_hsl_layout_and_provenance() -> None:
    layout_module, runtime, warmup = _load_q29_modules()
    layout = layout_module.resolve_frontres_future_intent_layout(
        (1, 2), layout_module.FRONTRES_FUTURE_INTENT_LAYOUT_VERSION
    )
    runner, normalizer, intent = _q29_runner(layout, runtime)
    raw_obs = torch.arange(10, dtype=torch.float32).reshape(2, 5)

    prepared = warmup.prepare_frontres_hsl_actor_observation(runner, raw_obs)

    assert tuple(prepared.shape) == (2, 63)
    assert len(normalizer.calls) == 1
    torch.testing.assert_close(normalizer.calls[0][:, :58], intent[:, (1, 2), :].reshape(2, 58))
    torch.testing.assert_close(normalizer.calls[0][:, 58:], raw_obs)
    torch.testing.assert_close(prepared, normalizer.calls[0] / 2.0)

    runner._frontres_segment_live_current_batch = SimpleNamespace()
    runner.env = _Env(_IntentCommand(SimpleNamespace()))
    _expect_error(
        RuntimeError,
        lambda: warmup.prepare_frontres_hsl_actor_observation(runner, raw_obs),
        "sealed local scenario",
    )

    invalid_provenance = tuple({**row, "intent_q29_provenance": "clean_q29"} for row in _provenance(2))
    bad_runner, _bad_normalizer, _bad_intent = _q29_runner(layout, runtime, provenance=invalid_provenance)
    _expect_error(
        RuntimeError,
        lambda: warmup.prepare_frontres_hsl_actor_observation(bad_runner, raw_obs),
        "invalid",
    )
    print("[T-HSL-layout/provenance] q29-only sealed actor context reaches normalizer before actor", flush=True)


def test_t_hsl_current_antidr_target() -> None:
    _layout, _runtime, warmup = _load_q29_modules()
    command = SimpleNamespace(
        anchor_dr_delta_pos=torch.tensor([[0.25, -0.50, -0.40], [0.0, 0.0, 0.10]]),
        anchor_dr_delta_quat_correction=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        ),
        clean_future=torch.full((2, 4, 65), 99.0),
    )
    target = torch.tensor(
        [[-0.25, 0.50, 0.40, 0.0, 0.0, 0.0], [0.0, 0.0, -0.10, 0.0, 0.0, 0.0]]
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _name: command)
    )
    produced = _load_hsl_target_owner()(env, "motion")
    torch.testing.assert_close(produced, target)
    validated = warmup.validate_frontres_hsl_current_frame_target(target, command)
    torch.testing.assert_close(validated, target)

    altered = target.clone()
    altered[0, 0] += 0.01
    _expect_error(
        RuntimeError,
        lambda: warmup.validate_frontres_hsl_current_frame_target(altered, command),
        "anti-DR",
    )
    _expect_error(
        RuntimeError,
        lambda: warmup.validate_frontres_hsl_current_frame_target(target[:, :5], command),
        "[B,6]",
    )
    warmup_source = WARMUP_PATH.read_text()
    assert "get_supervision_target_task_space as _get_warmup_target" in warmup_source
    assert "build_frontres_hsl_rollout_target" not in warmup_source
    torch.testing.assert_close(validated[:, 2], torch.tensor([0.40, -0.10]))
    print(
        "[T-HSL-target] current anti-DR [B,6] preserves both dz signs without Clean input or axis clamp",
        flush=True,
    )


def _load_legacy_label_owner():
    math_stub = types.ModuleType("isaaclab.utils.math")
    math_stub.quat_inv = lambda quat: quat
    math_stub.quat_mul = lambda lhs, _rhs: lhs
    sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))
    sys.modules.setdefault("isaaclab.utils", types.ModuleType("isaaclab.utils"))
    sys.modules["isaaclab.utils.math"] = math_stub
    return _load("frontres_hsl_rollout_target_h1_s1_contract", LEGACY_LABEL_PATH)


def test_t_hsl_stage3_legacy_reject() -> None:
    legacy = _load_legacy_label_owner()
    transition = SimpleNamespace(sentinel="unchanged")
    runner = SimpleNamespace(alg=SimpleNamespace(transition=transition))
    _expect_error(
        RuntimeError,
        lambda: legacy.build_frontres_hsl_rollout_target(
            runner,
            command=None,
            actions=None,
            dones=None,
            current_pos_correction=None,
            current_quat_correction=None,
            n_train=0,
            n_candidate=0,
            n_base=0,
            n_clean=0,
            quat_to_rotvec_wxyz=lambda value: value,
        ),
        "FRS-TRAIN-v007",
    )
    assert vars(transition) == {"sentinel": "unchanged"}
    assert "root_pos_w" not in LEGACY_LABEL_PATH.read_text()
    assert "supervised_target =" not in LEGACY_LABEL_PATH.read_text()
    assert "from rsl_rl.runners.frontres_hsl_rollout_target import" not in ON_POLICY_PATH.read_text()
    assert "build_frontres_hsl_rollout_target" not in ON_POLICY_PATH.read_text()
    assert "frontres_hsl_rollout_label_enabled = False" in G1_CFG_PATH.read_text()
    print("[T-HSL-stage3-reject] legacy Clean-quartet label cannot read or write transition storage", flush=True)


def test_t_hsl_direct_write_reject() -> None:
    rollout_step = _load("frontres_rollout_step_h1_s1_contract", ROLLOUT_STEP_PATH)
    transition = SimpleNamespace(sentinel="unchanged")
    runner = SimpleNamespace(
        alg=SimpleNamespace(lambda_supervised=1.0, frontres_future_offsets=(1, 2)),
        transition=transition,
    )
    _expect_error(
        RuntimeError,
        lambda: rollout_step._write_supervised_target_before_step(
            runner,
            actions=None,
            iteration=0,
            rollout_step=0,
            is_task_space_mode=True,
            n_train=1,
        ),
        "FRS-TRAIN-v018",
    )
    runner.alg.lambda_supervised = 0.0
    rollout_step._write_supervised_target_before_step(
        runner,
        actions=None,
        iteration=0,
        rollout_step=0,
        is_task_space_mode=True,
        n_train=1,
    )
    assert vars(transition) == {"sentinel": "unchanged"}
    rollout_source = ROLLOUT_STEP_PATH.read_text()
    writer_start = rollout_source.index("def _write_supervised_target_before_step")
    writer_end = rollout_source.index("def _capture_hsl_snapshot_before_step", writer_start)
    assert "_uses_v015_future_intent_route(runner)" in rollout_source[writer_start:writer_end]
    print("[T-HSL-direct-write-reject] v015 blocks nonzero online HSL writer before transition storage", flush=True)


def main() -> None:
    test_t_hsl_legacy_checkpoint_reject()
    test_t_hsl_checkpoint_identity_and_pre_mutation()
    test_t_stage3_explicit_hsl_initializer_actor_only()
    test_t_hsl_fresh_runner_connectivity()
    test_t_hsl_live_smoke_connector()
    test_t_hsl_loss_reject()
    test_t_hsl_proposal_command_carrier()
    test_t_hsl_proposal_runtime_route()
    test_t_hsl_formal_stage1_config_and_layout()
    test_t_hsl_actor_only_critic_unchanged()
    test_t_hsl_layout_and_provenance()
    test_t_hsl_current_antidr_target()
    test_t_hsl_stage3_legacy_reject()
    test_t_hsl_direct_write_reject()
    print("frontres_hsl_v007_s1_contract: ok", flush=True)


if __name__ == "__main__":
    main()
