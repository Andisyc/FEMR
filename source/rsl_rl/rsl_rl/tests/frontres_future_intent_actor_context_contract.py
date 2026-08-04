#!/usr/bin/env python3
"""Deterministic S1 contract for the v015 q29 future-intent actor bridge."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

from frontres_contract_imports import install_frontres_contract_packages


ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = ROOT / "source" / "rsl_rl"
LAYOUT_PATH = SOURCE_ROOT / "rsl_rl" / "modules" / "frontres_observation_layout.py"
RUNTIME_PATH = SOURCE_ROOT / "rsl_rl" / "runners" / "frontres_runtime.py"
ON_POLICY_PATH = SOURCE_ROOT / "rsl_rl" / "runners" / "on_policy_runner.py"
RSL_CFG_PATH = SOURCE_ROOT / "rsl_rl" / "modules" / "rsl_rl_cfg.py"
WBT_CFG_PATH = ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "utils" / "rsl_rl_cfg.py"


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_modules():
    rsl_rl = _package("rsl_rl")
    modules = _package("rsl_rl.modules")
    frontres = _package("rsl_rl.frontres")
    runners = _package("rsl_rl.runners")
    rsl_rl.modules = modules
    rsl_rl.frontres = frontres
    rsl_rl.runners = runners
    install_frontres_contract_packages(SOURCE_ROOT / "rsl_rl")
    modules.FrontRESActorCritic = type("FrontRESActorCritic", (), {})
    layout = _load("rsl_rl.modules.frontres_observation_layout", LAYOUT_PATH)
    modules.frontres_observation_layout = layout
    diagnostics = types.ModuleType("rsl_rl.frontres.runtime_diagnostics")
    diagnostics.maybe_print_frontres_restore_debug = lambda *_args, **_kwargs: None
    sys.modules[diagnostics.__name__] = diagnostics
    frontres.runtime_diagnostics = diagnostics
    runtime = _load("rsl_rl.runners.frontres_runtime", RUNTIME_PATH)
    return layout, runtime


class _Normalizer:
    def __init__(self, dim: int, *, divisor: float = 1.0) -> None:
        self._mean = torch.zeros(1, dim)
        self._std = torch.ones(1, dim)
        self.divisor = float(divisor)
        self.calls: list[tuple[int, ...]] = []

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(value.shape))
        return value / self.divisor


class _IntentCommand:
    def __init__(self, batch) -> None:
        self.batch = batch

    def frontres_local_scenario_intent_snapshot(self):
        intent = getattr(self.batch, "frontres_local_scenario_intent_q29", None)
        provenance = getattr(self.batch, "frontres_local_scenario_provenance", None)
        if not isinstance(intent, torch.Tensor) or provenance is None:
            raise RuntimeError("v015 local scenario intent snapshot is unavailable")
        batch_size = int(intent.shape[0])
        return {
            "intent_q29": intent.detach().clone(),
            "scenario_ids": tuple(f"scenario-{row}" for row in range(batch_size)),
            "noisy_segment_hashes": tuple(f"hash-{row}" for row in range(batch_size)),
            "x_t_identities": tuple(f"x-{row}" for row in range(batch_size)),
            "roles": ("repair",) * batch_size,
            "provenance": tuple(dict(value) for value in provenance),
        }


class _CommandManager:
    def __init__(self, command) -> None:
        self.command = command

    def get_term(self, name: str):
        assert name == "motion"
        return self.command


class _Env:
    def __init__(self, command) -> None:
        self.unwrapped = self
        self.command_manager = _CommandManager(command)


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


def _runner(layout, batch, *, raw_dim: int = 5, gmt_dim: int = 3):
    tail_dim = layout.actor_tail_dim
    return SimpleNamespace(
        env=_Env(_IntentCommand(batch)),
        device=torch.device("cpu"),
        alg=SimpleNamespace(
            policy=SimpleNamespace(
                num_actor_obs=raw_dim + tail_dim,
                num_frontres_obs=(raw_dim - gmt_dim) + tail_dim,
            )
        ),
        _frontres_gmt_obs_dim=gmt_dim,
        _frontres_future_intent_layout=layout,
        _frontres_future_intent_layout_version=layout.version,
        _frontres_future_intent_actor_context_dim=tail_dim,
        _frontres_segment_live_current_batch=batch,
    )


def _batch(intent_q29: torch.Tensor, *, provenance=None, offsets=(1, 2)):
    if provenance is None:
        provenance = _provenance(int(intent_q29.shape[0]))
    return SimpleNamespace(
        frontres_local_scenario_intent_q29=intent_q29.detach().clone(),
        frontres_local_scenario_provenance=provenance,
        frontres_local_scenario_clean_continuation=torch.full(
            (intent_q29.shape[0], 2, 65), 17.0, dtype=torch.float32
        ),
        frontres_local_scenario_current_root_artifact_t=torch.full(
            (intent_q29.shape[0], 7), 23.0, dtype=torch.float32
        ),
        frontres_future_offsets=tuple(offsets),
    )


def _expect_error(error_type, fn, text: str) -> None:
    try:
        fn()
    except error_type as exc:
        assert text in str(exc), str(exc)
        return
    raise AssertionError(f"expected {error_type.__name__}")


def test_t_shape_and_offset() -> None:
    layout_module, runtime = _load_modules()
    layout = layout_module.resolve_frontres_future_intent_layout(
        (1, 2), layout_module.FRONTRES_FUTURE_INTENT_LAYOUT_VERSION
    )
    intent = _intent()
    runner = _runner(layout, _batch(intent))
    raw_obs = torch.arange(10, dtype=torch.float32).reshape(2, 5)

    augmented = runtime.append_frontres_future_intent_context(runner, raw_obs)
    expected_tail = torch.cat([intent[:, 1], intent[:, 2]], dim=-1)

    assert tuple(augmented.shape) == (2, 5 + 2 * 29)
    torch.testing.assert_close(augmented[:, : 2 * 29], expected_tail)
    torch.testing.assert_close(augmented[:, 2 * 29 :], raw_obs)
    print(
        "[T-shape/T-offset] "
        f"layout={layout.version} offsets={layout.future_offsets} "
        f"tail_shape={tuple(expected_tail.shape)} actor_shape={tuple(augmented.shape)}",
        flush=True,
    )


def test_t_provenance() -> None:
    layout_module, runtime = _load_modules()
    layout = layout_module.resolve_frontres_future_intent_layout(
        (1, 2), layout_module.FRONTRES_FUTURE_INTENT_LAYOUT_VERSION
    )
    intent = _intent()
    bad_provenance = list(_provenance(2))
    bad_provenance[1] = dict(bad_provenance[1], intent_q29_provenance="clean_q29")
    runner = _runner(layout, _batch(intent, provenance=tuple(bad_provenance)))

    _expect_error(
        RuntimeError,
        lambda: runtime.append_frontres_future_intent_context(runner, torch.zeros(2, 5)),
        "deployment_noisy_q29",
    )
    print("[T-provenance] Clean-labelled q29 is rejected even when numeric values are valid", flush=True)


def test_t_clean_isolation() -> None:
    layout_module, runtime = _load_modules()
    layout = layout_module.resolve_frontres_future_intent_layout(
        (1, 2), layout_module.FRONTRES_FUTURE_INTENT_LAYOUT_VERSION
    )
    intent = _intent()
    first = _batch(intent)
    second = _batch(intent)
    second.frontres_local_scenario_clean_continuation.fill_(9999.0)
    second.frontres_local_scenario_current_root_artifact_t.fill_(-9999.0)
    second.future_root_global = torch.full((2, 4, 7), 5555.0)
    raw_obs = torch.zeros(2, 5)

    left = runtime.append_frontres_future_intent_context(_runner(layout, first), raw_obs)
    right = runtime.append_frontres_future_intent_context(_runner(layout, second), raw_obs)

    torch.testing.assert_close(left[:, : layout.actor_tail_dim], right[:, : layout.actor_tail_dim])
    print("[T-clean-isolation] Clean continuation, current artifact, and future root/global extras do not alter H tail", flush=True)


def test_t_normalizer_layout_rejects_incompatible_stats() -> None:
    layout_module, runtime = _load_modules()
    layout = layout_module.resolve_frontres_future_intent_layout(
        (1, 2), layout_module.FRONTRES_FUTURE_INTENT_LAYOUT_VERSION
    )
    batch = _batch(_intent())
    runner = _runner(layout, batch)
    extra_dim = 2 + layout.actor_tail_dim
    runner._frontres_extra_mean = None
    runner._frontres_extra_std = None
    runner._frontres_extra_normalizer = _Normalizer(extra_dim, divisor=2.0)
    runner.obs_normalizer = _Normalizer(3, divisor=5.0)
    normalized = runtime.apply_obs_normalizer(runner, torch.full((2, extra_dim + 3), 10.0))
    assert tuple(normalized.shape) == (2, extra_dim + 3)
    torch.testing.assert_close(normalized[:, :extra_dim], torch.full((2, extra_dim), 5.0))
    torch.testing.assert_close(normalized[:, extra_dim:], torch.full((2, 3), 2.0))

    bad = _runner(layout, batch)
    bad._frontres_extra_mean = torch.zeros(1, extra_dim)
    bad._frontres_extra_std = torch.ones(1, extra_dim)
    bad._frontres_extra_stats_layout_version = None
    bad._frontres_extra_normalizer = _Normalizer(extra_dim)
    bad.obs_normalizer = _Normalizer(3)
    _expect_error(
        RuntimeError,
        lambda: runtime.apply_obs_normalizer(bad, torch.zeros(2, extra_dim + 3)),
        "normalizer statistics",
    )
    print("[T-normalizer] fresh matching live stats pass; unversioned checkpoint-like prefix stats fail closed", flush=True)


def test_t_legacy_reject() -> None:
    layout_module, runtime = _load_modules()
    layout = layout_module.resolve_frontres_future_intent_layout(
        (1, 2), layout_module.FRONTRES_FUTURE_INTENT_LAYOUT_VERSION
    )
    legacy_intent = torch.zeros(2, 4, 65)
    _expect_error(
        RuntimeError,
        lambda: runtime.append_frontres_future_intent_context(
            _runner(layout, _batch(legacy_intent)), torch.zeros(2, 5)
        ),
        "[B,H_max+1,29]",
    )
    legacy_batch = SimpleNamespace(frontres_fixed_noisy_tape=torch.zeros(2, 4, 65))
    _expect_error(
        RuntimeError,
        lambda: runtime.append_frontres_future_intent_context(
            _runner(layout, legacy_batch), torch.zeros(2, 5)
        ),
        "local scenario",
    )
    _expect_error(
        ValueError,
        lambda: layout_module.resolve_frontres_future_intent_layout((1, 3), "v013-fixed-noisy"),
        "layout version",
    )
    _expect_error(
        ValueError,
        lambda: layout_module.resolve_frontres_future_intent_layout(
            (1, 3), layout_module.FRONTRES_FUTURE_INTENT_LAYOUT_VERSION
        ),
        "exact deployment offsets (1, 2)",
    )
    assert "frontres_future_intent_layout_version" in RSL_CFG_PATH.read_text()
    assert "frontres_future_intent_layout_version" in WBT_CFG_PATH.read_text()
    runner_text = ON_POLICY_PATH.read_text()
    layout_start = runner_text.index("num_actor_obs = num_obs")
    layout_end = runner_text.index("# evaluate the policy class", layout_start)
    layout_block = runner_text[layout_start:layout_end]
    assert "resolve_frontres_future_intent_layout" in layout_block
    assert "layout.actor_tail_dim" in layout_block
    assert "* 65" not in layout_block
    print(
        "[T-legacy-reject] 65D tail, absent local scenario, wrong offsets, and wrong layout version fail closed; "
        "runner config owns q29-only dimensions",
        flush=True,
    )


def main() -> None:
    test_t_shape_and_offset()
    test_t_provenance()
    test_t_clean_isolation()
    test_t_normalizer_layout_rejects_incompatible_stats()
    test_t_legacy_reject()
    print("frontres_future_intent_actor_context_contract: ok", flush=True)


if __name__ == "__main__":
    main()
