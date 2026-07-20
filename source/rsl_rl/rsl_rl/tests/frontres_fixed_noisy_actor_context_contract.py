#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = ROOT / "source" / "rsl_rl"

def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


def _load_runtime():
    rsl_rl = _package("rsl_rl")
    modules = _package("rsl_rl.modules")
    frontres = _package("rsl_rl.frontres")
    runners = _package("rsl_rl.runners")
    rsl_rl.modules = modules
    rsl_rl.frontres = frontres
    rsl_rl.runners = runners
    modules.FrontRESActorCritic = type("FrontRESActorCritic", (), {})
    layout = types.ModuleType("rsl_rl.modules.frontres_observation_layout")
    layout.FRONTRES_FUTURE_INTENT_LAYOUT_VERSION = "frontres-v015-future-intent-q29-v1"
    layout.FrontRESFutureIntentLayout = type("FrontRESFutureIntentLayout", (), {})
    layout.build_frontres_future_intent_tail = lambda *_args, **_kwargs: None
    layout.split_frontres_policy_obs = lambda obs, _gmt_dim: (None, obs)
    sys.modules[layout.__name__] = layout
    modules.frontres_observation_layout = layout
    diagnostics = types.ModuleType("rsl_rl.frontres.runtime_diagnostics")
    diagnostics.maybe_print_frontres_restore_debug = lambda *_args, **_kwargs: None
    sys.modules[diagnostics.__name__] = diagnostics
    frontres.runtime_diagnostics = diagnostics
    path = SOURCE_ROOT / "rsl_rl" / "runners" / "frontres_runtime.py"
    spec = importlib.util.spec_from_file_location("rsl_rl.runners.frontres_runtime", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


frontres_runtime = _load_runtime()


class _Command:
    def __init__(self, context: torch.Tensor) -> None:
        self.context = context
        self.offset_calls: list[tuple[int, ...]] = []

    def frontres_fixed_noisy_future_context(self, offsets):
        self.offset_calls.append(tuple(offsets))
        return self.context.clone()


def _runner(*, expected_actor_dim: int, context: torch.Tensor) -> SimpleNamespace:
    command = _Command(context)
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))
    batch = SimpleNamespace(
        frontres_fixed_noisy_tape=torch.ones(context.shape[0], 3, 65),
        frontres_future_offsets=(1, 2),
    )
    return SimpleNamespace(
        env=env,
        device=torch.device("cpu"),
        alg=SimpleNamespace(policy=SimpleNamespace(num_actor_obs=expected_actor_dim, num_frontres_obs=132)),
        _frontres_gmt_obs_dim=3,
        _frontres_segment_live_current_batch=batch,
    )


def test_fixed_noisy_actor_context_is_prepended_and_shape_checked() -> None:
    raw_obs = torch.arange(2 * 5, dtype=torch.float32).reshape(2, 5)
    context = torch.arange(2 * 130, dtype=torch.float32).reshape(2, 130) + 1000.0
    runner = _runner(expected_actor_dim=135, context=context)

    augmented = frontres_runtime.append_frontres_fixed_noisy_future_context(runner, raw_obs)

    assert tuple(augmented.shape) == (2, 135)
    torch.testing.assert_close(augmented[:, :130], context)
    torch.testing.assert_close(augmented[:, 130:], raw_obs)
    assert runner.env.command_manager.get_term("motion").offset_calls == [(1, 2)]
    print(
        "[probe fixed_tape_actor_context] "
        f"raw_shape={tuple(raw_obs.shape)} context_shape={tuple(context.shape)} "
        f"augmented_shape={tuple(augmented.shape)} offsets={(1, 2)}",
        flush=True,
    )


def test_fixed_noisy_actor_context_rejects_legacy_actor_shape() -> None:
    raw_obs = torch.zeros(2, 5)
    runner = _runner(expected_actor_dim=5, context=torch.zeros(2, 130))

    try:
        frontres_runtime.append_frontres_fixed_noisy_future_context(runner, raw_obs)
    except RuntimeError as exc:
        assert "legacy actor layout" in str(exc)
        print(f"[probe fixed_tape_actor_fail_closed] {exc}", flush=True)
    else:
        raise AssertionError("fixed Noisy H context silently accepted the legacy actor layout")


def main() -> None:
    test_fixed_noisy_actor_context_is_prepended_and_shape_checked()
    test_fixed_noisy_actor_context_rejects_legacy_actor_shape()
    print("frontres_fixed_noisy_actor_context_contract: ok")


if __name__ == "__main__":
    main()
