#!/usr/bin/env python3
"""CPU-only S2 connectivity contract for FRS-TRAIN-v007 proposal-only HSL.

This weak fake proves owner-to-owner tensor connectivity only. It deliberately
does not construct an environment, invoke an optimizer, or claim physics,
formal-route, checkpoint, or live-runtime evidence.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = ROOT / "source" / "rsl_rl"
PKG_ROOT = SOURCE_ROOT / "rsl_rl"
RUNNERS_ROOT = PKG_ROOT / "runners"
MODULES_ROOT = PKG_ROOT / "modules"
ALGORITHMS_ROOT = PKG_ROOT / "algorithms"
STORAGE_ROOT = PKG_ROOT / "storage"

LAYOUT_PATH = MODULES_ROOT / "frontres_observation_layout.py"
RUNTIME_PATH = RUNNERS_ROOT / "frontres_runtime.py"
WARMUP_PATH = RUNNERS_ROOT / "frontres_warmup.py"
ROLLOUT_STEP_PATH = RUNNERS_ROOT / "frontres_rollout_step.py"
UNIFIED_PATH = ALGORITHMS_ROOT / "frontres_unified.py"
STORAGE_PATH = STORAGE_ROOT / "rollout_storage.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


def _load_owners():
    """Load only the real local owners needed by this CPU-only contract."""

    rsl_rl = _package("rsl_rl")
    modules = _package("rsl_rl.modules")
    storage_package = _package("rsl_rl.storage")
    frontres = _package("rsl_rl.frontres")
    runners = _package("rsl_rl.runners")
    algorithms = _package("rsl_rl.algorithms")
    rsl_rl.modules = modules
    rsl_rl.storage = storage_package
    rsl_rl.frontres = frontres
    rsl_rl.runners = runners
    rsl_rl.algorithms = algorithms
    modules.ActorCritic = type("ActorCritic", (), {})
    modules.FrontRESActorCritic = type("FrontRESActorCritic", (), {})
    modules.ResidualActorCritic = type("ResidualActorCritic", (), {})

    utils = types.ModuleType("rsl_rl.utils")
    utils.split_and_pad_trajectories = lambda trajectories, dones: (trajectories, dones)
    sys.modules[utils.__name__] = utils
    rsl_rl.utils = utils

    layout = _load("rsl_rl.modules.frontres_observation_layout", LAYOUT_PATH)
    modules.frontres_observation_layout = layout
    diagnostics = types.ModuleType("rsl_rl.frontres.runtime_diagnostics")
    diagnostics.maybe_print_frontres_restore_debug = lambda *_args, **_kwargs: None
    sys.modules[diagnostics.__name__] = diagnostics
    frontres.runtime_diagnostics = diagnostics

    runtime = _load("rsl_rl.runners.frontres_runtime", RUNTIME_PATH)
    warmup = _load("frontres_warmup_h1_s2_contract", WARMUP_PATH)
    rollout_step = _load("frontres_rollout_step_h1_s2_contract", ROLLOUT_STEP_PATH)
    storage = _load("rsl_rl.storage.rollout_storage", STORAGE_PATH)
    storage_package.RolloutStorage = storage.RolloutStorage
    unified = _load("rsl_rl.algorithms.frontres_unified", UNIFIED_PATH)
    algorithms.frontres_unified = unified
    return layout, runtime, warmup, rollout_step, storage, unified


def _intent(batch_size: int = 2, hmax: int = 3) -> torch.Tensor:
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


class _TraceNormalizer:
    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        self.calls.append(value.detach().clone())
        return value / 2.0


class _TraceResidualActor:
    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        self.calls.append(value.detach().clone())
        return value[:, :6] * 0.0


class _NoStepOptimizer:
    def __init__(self) -> None:
        self.step_calls = 0

    def step(self) -> None:
        self.step_calls += 1
        raise AssertionError("the H1-S2 zero-HSL-loss probe must not step an optimizer")


def _make_runner(layout, runtime, normalizer: _TraceNormalizer):
    intent = _intent()
    batch = SimpleNamespace(
        frontres_local_scenario_intent_q29=intent,
        frontres_local_scenario_provenance=_provenance(intent.shape[0]),
        frontres_future_offsets=(1, 3),
        frontres_local_scenario_current_root_artifact=torch.full((2, 7), -313.0),
        frontres_local_scenario_clean_continuation=torch.full((2, 2, 65), 701.0),
        frontres_local_scenario_id=("s2-connect-0", "s2-connect-1"),
        frontres_noisy_segment_hash=("hash-0", "hash-1"),
    )
    raw_dim = 5
    gmt_dim = 3
    policy = SimpleNamespace(
        num_actor_obs=raw_dim + layout.actor_tail_dim,
        num_frontres_obs=(raw_dim - gmt_dim) + layout.actor_tail_dim,
    )
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        alg=SimpleNamespace(policy=policy),
        _frontres_gmt_obs_dim=gmt_dim,
        _frontres_future_intent_layout=layout,
        _frontres_future_intent_actor_context_dim=layout.actor_tail_dim,
        _frontres_segment_live_current_batch=batch,
        _apply_obs_normalizer=normalizer,
    )
    runner._append_frontres_future_intent_context = (
        lambda obs: runtime.append_frontres_future_intent_context(runner, obs)
    )
    return runner, batch, intent


def test_t_hsl_connect_stage1(layout, runtime, warmup) -> tuple[torch.Tensor, torch.Tensor]:
    normalizer = _TraceNormalizer()
    runner, batch, intent = _make_runner(layout, runtime, normalizer)
    raw_obs = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    actor = _TraceResidualActor()
    command = SimpleNamespace(
        anchor_dr_delta_pos=torch.tensor([[0.25, -0.50, -0.40], [0.0, 0.0, 0.10]]),
        anchor_dr_delta_quat_correction=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        ),
    )
    target = torch.tensor(
        [[-0.25, 0.50, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -0.10, 0.0, 0.0, 0.0]]
    )

    normalized = warmup.prepare_frontres_hsl_actor_observation(runner, raw_obs)
    actor_input = normalized[:, : runner.alg.policy.num_frontres_obs]
    prediction = actor(actor_input)
    checked_target = warmup.validate_frontres_hsl_current_frame_target(target, command)
    proposal_loss = (prediction - checked_target).square().mean()

    expected_tail = intent[:, (1, 3), :].reshape(2, layout.actor_tail_dim)
    assert len(normalizer.calls) == 1
    assert tuple(normalizer.calls[0].shape) == (2, 63)
    torch.testing.assert_close(normalizer.calls[0][:, : layout.actor_tail_dim], expected_tail)
    torch.testing.assert_close(normalizer.calls[0][:, layout.actor_tail_dim :], raw_obs)
    assert tuple(actor_input.shape) == (2, 60)
    torch.testing.assert_close(actor.calls[0], actor_input)
    torch.testing.assert_close(actor_input[:, : layout.actor_tail_dim], expected_tail / 2.0)
    assert not bool((normalizer.calls[0] == 701.0).any().item())
    assert not bool((actor_input == 701.0).any().item())
    assert not checked_target.requires_grad
    assert float(proposal_loss.item()) > 0.0
    assert getattr(batch, "frontres_local_scenario_clean_continuation").shape == (2, 2, 65)
    warmup_source = WARMUP_PATH.read_text()
    run_start = warmup_source.index("def run_frontres_joint_warmup")
    run_source = warmup_source[run_start:]
    assert "_p_obs = prepare_frontres_hsl_actor_observation(self, _p_obs_raw)" in run_source
    assert "_target = validate_frontres_hsl_current_frame_target(" in run_source
    assert "_wo_list.append(_p_obs[:, :_nfo])" in run_source
    assert "pred = self.alg.policy.residual_actor(_all_obs[idx])" in run_source
    print(
        "[T-HSL-connect-stage1] "
        "q29=[2,58] -> normalizer=[2,63] -> actor=[2,60] -> current_target=[2,6], "
        "clean_continuation_actor_values=0",
        flush=True,
    )
    return checked_target, normalized


def _stage3_transition(storage, normalized: torch.Tensor):
    transition = storage.RolloutStorage.Transition()
    batch_size = int(normalized.shape[0])
    transition.observations = normalized.detach().clone()
    transition.actions = torch.zeros(batch_size, 6)
    transition.rewards = torch.zeros(batch_size)
    transition.dones = torch.zeros(batch_size, dtype=torch.uint8)
    transition.values = torch.zeros(batch_size, 1)
    transition.actions_log_prob = torch.zeros(batch_size)
    transition.action_mean = torch.zeros(batch_size, 6)
    transition.action_sigma = torch.ones(batch_size, 6)
    transition.frontres_mask = torch.ones(batch_size, 1)
    return transition


def test_t_hsl_connect_stage3_zero_path(layout, rollout_step, storage, unified, stage1_target, normalized) -> None:
    transition = _stage3_transition(storage, normalized)
    runner = SimpleNamespace(
        alg=SimpleNamespace(
            lambda_supervised=0.0,
            frontres_future_offsets=layout.future_offsets,
            transition=transition,
        ),
        _frontres_future_intent_layout=layout,
    )

    rollout_step._write_supervised_target_before_step(
        runner,
        actions=None,
        iteration=0,
        rollout_step=0,
        is_task_space_mode=True,
        n_train=2,
    )
    assert transition.supervised_target is None
    assert transition.supervised_weight is None
    assert transition.supervised_harm_weight is None

    rollout = storage.RolloutStorage(
        "frontres",
        num_envs=2,
        num_transitions_per_env=1,
        obs_shape=(normalized.shape[-1],),
        privileged_obs_shape=None,
        actions_shape=(6,),
        device="cpu",
    )
    rollout.add_transitions(transition)
    assert not bool(torch.count_nonzero(rollout.supervised_target).item())
    assert bool(torch.all(rollout.supervised_weight == 1.0).item())
    assert not bool(torch.count_nonzero(rollout.supervised_harm_weight).item())

    batch = next(rollout.mini_batch_generator(num_mini_batches=1, num_epochs=1))
    supervised_target_batch = batch[18]
    supervised_weight_batch = batch[19]
    supervised_harm_weight_batch = batch[20]
    torch.testing.assert_close(supervised_target_batch, torch.zeros_like(supervised_target_batch))
    assert not torch.equal(supervised_target_batch, stage1_target)

    optimizer = _NoStepOptimizer()
    loss_owner = SimpleNamespace(
        device=torch.device("cpu"),
        lambda_supervised=0.0,
        optimizer=optimizer,
    )
    loss, cosine, metrics = unified.FrontRESUnified._compute_supervised_loss(
        loss_owner,
        torch.ones_like(supervised_target_batch, requires_grad=True),
        supervised_target_batch,
        int(supervised_target_batch.shape[0]),
        supervised_weight_batch=supervised_weight_batch,
        supervised_harm_weight_batch=supervised_harm_weight_batch,
    )
    assert float(loss.item()) == 0.0
    assert not loss.requires_grad
    assert cosine == 0.0
    assert all(value == 0.0 for value in metrics.values())
    assert optimizer.step_calls == 0
    print(
        "[T-HSL-connect-stage3] "
        "writer_target=None -> storage_target=[2,6] all_zero -> batch=[2,6] -> hsl_loss=0, "
        "optimizer_calls=0",
        flush=True,
    )


def main() -> None:
    layout_module, runtime, warmup, rollout_step, storage, unified = _load_owners()
    layout = layout_module.resolve_frontres_future_intent_layout(
        (1, 3), layout_module.FRONTRES_FUTURE_INTENT_LAYOUT_VERSION
    )
    stage1_target, normalized = test_t_hsl_connect_stage1(layout, runtime, warmup)
    test_t_hsl_connect_stage3_zero_path(
        layout,
        rollout_step,
        storage,
        unified,
        stage1_target,
        normalized,
    )
    print("frontres_hsl_v007_s2_connectivity_contract: ok", flush=True)


if __name__ == "__main__":
    main()
