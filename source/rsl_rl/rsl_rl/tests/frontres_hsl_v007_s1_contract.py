#!/usr/bin/env python3
"""Deterministic S1 contract for FRS-TRAIN-v007 proposal-only HSL."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


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


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


class _Normalizer:
    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        self.calls.append(value.detach().clone())
        return value / 2.0


def _q29_runner(layout, runtime, *, provenance=None):
    raw_dim = 5
    gmt_dim = 3
    intent = _intent()
    batch = SimpleNamespace(
        frontres_local_scenario_intent_q29=intent,
        frontres_local_scenario_provenance=_provenance(intent.shape[0]) if provenance is None else provenance,
        frontres_future_offsets=(1, 3),
    )
    normalizer = _Normalizer()
    runner = SimpleNamespace(
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


def test_t_hsl_loss_reject() -> None:
    unified = _load_unified_guard()
    _expect_error(
        ValueError,
        lambda: unified.validate_frontres_v015_stage3_supervision_config(
            future_offsets=(1, 3), lambda_supervised=1.0, lambda_supervised_min=0.0
        ),
        "FRS-TRAIN-v007",
    )
    _expect_error(
        ValueError,
        lambda: unified.validate_frontres_v015_stage3_supervision_config(
            future_offsets=(1, 3), lambda_supervised=0.0, lambda_supervised_min=0.2
        ),
        "FRS-TRAIN-v007",
    )
    unified.validate_frontres_v015_stage3_supervision_config(
        future_offsets=(1, 3), lambda_supervised=0.0, lambda_supervised_min=0.0
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
        (1, 3), layout_module.FRONTRES_FUTURE_INTENT_LAYOUT_VERSION
    )
    runner, normalizer, intent = _q29_runner(layout, runtime)
    raw_obs = torch.arange(10, dtype=torch.float32).reshape(2, 5)

    prepared = warmup.prepare_frontres_hsl_actor_observation(runner, raw_obs)

    assert tuple(prepared.shape) == (2, 63)
    assert len(normalizer.calls) == 1
    torch.testing.assert_close(normalizer.calls[0][:, :58], intent[:, (1, 3), :].reshape(2, 58))
    torch.testing.assert_close(normalizer.calls[0][:, 58:], raw_obs)
    torch.testing.assert_close(prepared, normalizer.calls[0] / 2.0)

    runner._frontres_segment_live_current_batch = SimpleNamespace()
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
        [[-0.25, 0.50, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -0.10, 0.0, 0.0, 0.0]]
    )
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
    print("[T-HSL-target] current anti-DR [B,6] target is checked without Clean rollout input", flush=True)


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
        alg=SimpleNamespace(lambda_supervised=1.0, frontres_future_offsets=(1, 3)),
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
        "FRS-TRAIN-v007",
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
    test_t_hsl_loss_reject()
    test_t_hsl_layout_and_provenance()
    test_t_hsl_current_antidr_target()
    test_t_hsl_stage3_legacy_reject()
    test_t_hsl_direct_write_reject()
    print("frontres_hsl_v007_s1_contract: ok", flush=True)


if __name__ == "__main__":
    main()
