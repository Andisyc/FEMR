#!/usr/bin/env python3
"""TEST ONLY: active HSL acceptance payload builds labels and masks correctly."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
import importlib.util

import torch

ROOT = Path(__file__).resolve().parents[4]
FRONTRES_DIR = ROOT / "source/rsl_rl/rsl_rl/frontres"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_transition_module():
    rsl_pkg = ModuleType("rsl_rl")
    frontres_pkg = ModuleType("rsl_rl.frontres")
    rsl_pkg.frontres = frontres_pkg
    sys.modules.setdefault("rsl_rl", rsl_pkg)
    sys.modules.setdefault("rsl_rl.frontres", frontres_pkg)

    _load_module(
        "rsl_rl.frontres.frontres_acceptance_labels",
        FRONTRES_DIR / "frontres_acceptance_labels.py",
    )
    alpha_stub = ModuleType("rsl_rl.frontres.frontres_alpha_router")
    alpha_stub.build_state_alpha_targets = lambda *args, **kwargs: None
    sys.modules["rsl_rl.frontres.frontres_alpha_router"] = alpha_stub
    reward_stub = ModuleType("rsl_rl.frontres.frontres_reward_window")
    reward_stub.FrontRESRewardContext = object
    sys.modules["rsl_rl.frontres.frontres_reward_window"] = reward_stub
    structured_stub = ModuleType("rsl_rl.frontres.frontres_structured_rho")
    structured_stub.build_structured_rho_carrier = lambda *args, **kwargs: None
    sys.modules["rsl_rl.frontres.frontres_structured_rho"] = structured_stub
    return _load_module(
        "frontres_transition_payload_under_test",
        FRONTRES_DIR / "frontres_transition_payload.py",
    )


FrontRESActionCone = _load_module(
    "frontres_action_cone_under_test",
    FRONTRES_DIR / "frontres_action_cone.py",
).FrontRESActionCone
_transition_module = _load_transition_module()
initialize_frontres_acceptance_payload = _transition_module.initialize_frontres_acceptance_payload
_write_active_hsl_acceptance_payload = _transition_module._write_active_hsl_acceptance_payload


def _assert_close(name: str, got: torch.Tensor | float, expected: float) -> None:
    value = float(got.detach().item()) if isinstance(got, torch.Tensor) else float(got)
    if abs(value - expected) > 1.0e-6:
        raise AssertionError(f"{name}: got {value:.6f}, expected {expected:.6f}")


def run_active_payload_mask_check() -> None:
    device = torch.device("cpu")
    dtype = torch.float32
    num_envs = 4
    cfg = {
        "frontres_training_objective": "hsl_hybrid",
        "frontres_per_mode_acceptance_preference_mask": True,
        "frontres_active_task_dims": [0, 1, 3, 4],
    }
    alg = SimpleNamespace(
        transition=SimpleNamespace(action_mean=torch.zeros(num_envs, 12, device=device, dtype=dtype))
    )
    runner = SimpleNamespace(
        cfg=cfg,
        env=SimpleNamespace(num_envs=num_envs),
        device=device,
        alg=alg,
    )
    runner._frontres_action_cone = FrontRESActionCone(cfg, alg)

    payload = initialize_frontres_acceptance_payload(runner)
    actions = torch.zeros(num_envs, 12, device=device, dtype=dtype)
    state_alpha_mask = torch.ones(num_envs, 1, device=device, dtype=dtype)
    ctx = SimpleNamespace(
        candidate_gain=torch.tensor([0.10, -0.20, 0.001, 0.30], device=device, dtype=dtype),
        repair_gain=torch.tensor([0.08, -0.10, 0.0, 0.20], device=device, dtype=dtype),
        mode_groups=[("local_rp",), ("local_rp",), ("local_rp",), ("planar",)],
    )

    payload, accept_target, accept_mask = _write_active_hsl_acceptance_payload(
        runner,
        accept_payload=payload,
        accept_target=payload.accept_target,
        accept_mask=payload.accept_mask,
        n_exec=num_envs,
        pref_margin=0.01,
        ctx=ctx,
        actions=actions,
        state_alpha_mask=state_alpha_mask,
    )

    _assert_close("raw mask sample fraction", payload.pref_raw_mask_frac, 0.75)
    _assert_close("mode mask sample fraction", payload.pref_mode_mask_frac, 0.75)
    _assert_close("dim mask sample fraction", payload.pref_dim_mask_frac, 0.75)
    _assert_close("accept fraction", payload.pref_full_frac, 2.0 / 3.0)
    _assert_close("reject fraction", payload.pref_noop_frac, 1.0 / 3.0)
    _assert_close("ignore fraction", payload.pref_ignore_frac, 0.25)

    expected_target = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        device=device,
        dtype=dtype,
    )
    expected_mask = torch.tensor(
        [
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
        ],
        device=device,
        dtype=dtype,
    )
    torch.testing.assert_close(accept_target, expected_target)
    torch.testing.assert_close(accept_mask, expected_mask)
    torch.testing.assert_close(alg.transition.acceptance_target, expected_target)
    torch.testing.assert_close(alg.transition.acceptance_mask, expected_mask)
    torch.testing.assert_close(alg.transition.acceptance_gt, expected_target)
    torch.testing.assert_close(alg.transition.acceptance_margin[:, 0], ctx.candidate_gain)
    print("FrontRES active acceptance payload mask generation: PASS")


if __name__ == "__main__":
    run_active_payload_mask_check()
