#!/usr/bin/env python3
"""Deterministic S1 contract for the v015 immutable local-scenario kernel."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

from frontres_contract_imports import install_frontres_contract_packages


REPO = Path(__file__).resolve().parents[4]
RSL_ROOT = REPO / "source" / "rsl_rl"
COMMANDS_PATH = (
    REPO
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "commands.py"
)


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


def _install_isaac_stubs() -> None:
    isaaclab = _package("isaaclab")
    assets = _package("isaaclab.assets")
    managers = _package("isaaclab.managers")
    markers = _package("isaaclab.markers")
    markers_config = _package("isaaclab.markers.config")
    utils = _package("isaaclab.utils")
    math_mod = _package("isaaclab.utils.math")

    class _Dummy:
        def __init__(self, *_args, **_kwargs) -> None:
            self.markers = {"frame": SimpleNamespace(scale=None)}

        def replace(self, **_kwargs):
            return self

        def copy(self):
            return self

    assets.Articulation = _Dummy
    managers.CommandTerm = _Dummy
    managers.CommandTermCfg = _Dummy
    markers.VisualizationMarkers = _Dummy
    markers.VisualizationMarkersCfg = _Dummy
    markers_config.FRAME_MARKER_CFG = _Dummy()
    utils.configclass = lambda cls: cls

    def _identity_first(value, *_args, **_kwargs):
        return value

    math_mod.euler_xyz_from_quat = lambda value: (value[..., 0], value[..., 0], value[..., 0])
    math_mod.quat_apply = lambda _quat, value: value
    math_mod.quat_error_magnitude = lambda left, _right: torch.zeros(left.shape[0], device=left.device)
    math_mod.quat_from_euler_xyz = lambda x, _y, _z: torch.stack(
        [torch.ones_like(x), torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)], dim=-1
    )
    math_mod.quat_inv = _identity_first
    math_mod.quat_mul = lambda _left, right: right
    math_mod.sample_uniform = lambda _low, _high, shape, device=None: torch.zeros(shape, device=device)
    math_mod.yaw_quat = _identity_first

    isaaclab.assets = assets
    isaaclab.managers = managers
    isaaclab.markers = markers
    isaaclab.utils = utils

    _package("whole_body_tracking")
    _package("whole_body_tracking.whole_body_tracking")
    _package("whole_body_tracking.whole_body_tracking.tasks")
    _package("whole_body_tracking.whole_body_tracking.tasks.tracking")
    mdp_pkg = _package("whole_body_tracking.whole_body_tracking.tasks.tracking.mdp")
    perturbations = types.ModuleType(
        "whole_body_tracking.whole_body_tracking.tasks.tracking.mdp.motion_perturbations"
    )
    perturbations.MotionPerturber = _Dummy
    sys.modules[perturbations.__name__] = perturbations
    mdp_pkg.motion_perturbations = perturbations


def _load_modules():
    _install_isaac_stubs()
    install_frontres_contract_packages(RSL_ROOT / "rsl_rl")
    commands = _load(
        "whole_body_tracking.whole_body_tracking.tasks.tracking.mdp.commands",
        COMMANDS_PATH,
    )
    live_sampler = _load(
        "frontres_local_scenario_live_sampler_contract",
        RSL_ROOT / "rsl_rl" / "runners" / "frontres_segment_live_sampler.py",
    )
    hooks = _load(
        "frontres_local_scenario_stage1_hooks_contract",
        RSL_ROOT / "rsl_rl" / "frontres" / "frontres_segment_stage1_env_hooks.py",
    )
    return commands, live_sampler, hooks


class _FakeMotionDirLoader:
    def __init__(self) -> None:
        self.joint_pos = torch.zeros(32, 29)
        self.joint_vel = torch.zeros(32, 29)
        self.gathered_attrs: list[str] = []

    def gather(self, attr: str, motion_indices: torch.Tensor, frame_indices: torch.Tensor, out_device):
        del motion_indices
        self.gathered_attrs.append(str(attr))
        frame = frame_indices.to(torch.float32).reshape(-1, 1)
        joint = 10.0 * frame + torch.arange(29, dtype=torch.float32).reshape(1, -1)
        if attr == "joint_pos":
            return joint.to(out_device)
        if attr == "joint_vel":
            return (joint + 1000.0).to(out_device)
        if attr == "body_pos_w":
            return torch.stack([frame[:, 0], frame[:, 0] + 1.0, frame[:, 0] + 2.0], dim=-1).unsqueeze(1).to(out_device)
        if attr == "body_quat_w":
            quat = torch.zeros(frame.shape[0], 1, 4, dtype=torch.float32)
            quat[..., 0] = 1.0
            quat[..., 1] = frame
            return quat.to(out_device)
        raise KeyError(attr)


class _RootOnlyPerturber:
    instances: list["_RootOnlyPerturber"] = []

    def __init__(self, cfg, num_envs: int, device) -> None:
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.device = device
        self.scale = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.family_masks = None
        self.root_calls = 0
        self.quat_calls = 0
        self.joint_calls = 0
        type(self).instances.append(self)

    def set_dr_scale_env(self, scale: torch.Tensor) -> None:
        self.scale = scale.detach().to(self.device, dtype=torch.float32).clone()

    def set_family_env_masks(self, masks) -> None:
        self.family_masks = {name: value.detach().clone() for name, value in masks.items()}

    def apply_perturbations(self, root_pos: torch.Tensor, *_feet: torch.Tensor) -> torch.Tensor:
        self.root_calls += 1
        return root_pos + self.scale[:, None] * torch.tensor([1.0, 2.0, 3.0], device=root_pos.device)

    def apply_quat_perturbation(self, root_quat: torch.Tensor) -> torch.Tensor:
        self.quat_calls += 1
        result = root_quat.clone()
        result[:, 1] += self.scale
        return result

    def apply_joint_perturbation(self, _joint_pos: torch.Tensor) -> torch.Tensor:
        self.joint_calls += 1
        raise AssertionError("v015 q29 intent must not consume the joint perturbation owner")


class _AxisPerturber(_RootOnlyPerturber):
    """Independent hand-answer perturber used only at the materializer boundary."""

    def apply_perturbations(self, root_pos: torch.Tensor, *_feet: torch.Tensor) -> torch.Tensor:
        self.root_calls += 1
        return root_pos.clone()

    def apply_quat_perturbation(self, root_quat: torch.Tensor) -> torch.Tensor:
        del root_quat
        self.quat_calls += 1
        angle = float(self.scale[0]) * float(self.cfg.sign)
        half = torch.tensor(0.5 * angle, dtype=torch.float32, device=self.device)
        result = torch.zeros(1, 4, dtype=torch.float32, device=self.device)
        result[:, 0] = torch.cos(half)
        component = 1 if self.cfg.axis == "roll" else 2
        result[:, component] = torch.sin(half)
        return result


def _command(commands_module):
    command = object.__new__(commands_module.MultiMotionCommand)
    command.num_envs = 1
    command.device = torch.device("cpu")
    command.motion_dir_loader = _FakeMotionDirLoader()
    command.motion_lengths_minus_one = torch.tensor([31], dtype=torch.long)
    command.motion_anchor_body_index = 0
    command.left_foot_idx = 0
    command.right_foot_idx = 0
    command.perturber = _RootOnlyPerturber(SimpleNamespace(), 1, command.device)
    return command


def _materialize(command):
    return command.materialize_frontres_local_scenario(
        motion_index=0,
        start_frame=2,
        horizon_k=3,
        intent_horizon=2,
        perturbation_family="local_rp",
        perturbation_strength=0.25,
    )


def _scenario_parts(live_sampler, command, *, x_t_identity: str = "motion-0:frame-2:segment-7", horizon_k: int = 3):
    sampler = live_sampler._SAMPLER_MODULE
    payload = _materialize(command)
    materialization = sampler.FrontRESLocalScenarioMaterialization(
        current_root_artifact_t=payload["current_root_artifact_t"],
        clean_reference_t=payload["clean_reference_t"],
        intent_q29=payload["intent_q29"],
        clean_continuation=payload["clean_continuation"][:horizon_k],
        expected_support=payload["expected_support"][:horizon_k],
        expected_support_envelope=payload["expected_support_envelope"][:horizon_k],
        provenance=payload["provenance"],
    )
    request = sampler.FrontRESLocalScenarioRequest(
        transaction_id="tx-local",
        scenario_id="tx-local:source-0:segment-7",
        segment_id=7,
        source_index=0,
        x_t_identity=x_t_identity,
        horizon_k=horizon_k,
        future_offsets=(1, 2),
    )
    scenario = sampler.FrontRESLocalScenario.from_materialization(request, materialization)
    return sampler, payload, materialization, request, scenario


def _sample(sampler):
    return sampler.FrontRESSegmentSample(
        segment_ids=torch.tensor([7, 7], dtype=torch.long),
        source=("global", "global"),
        priority=torch.zeros(2),
        staleness=torch.zeros(2),
        valid_mask=torch.ones(2, dtype=torch.bool),
        horizon_k=torch.tensor([3, 3], dtype=torch.long),
        source_index=torch.tensor([0, 0], dtype=torch.long),
        trial_index=torch.tensor([0, 1], dtype=torch.long),
    )


def _expect_error(error_type, fn) -> None:
    try:
        fn()
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


def test_t_schema() -> None:
    commands, live_sampler, _hooks = _load_modules()
    command = _command(commands)
    _sampler, _payload, _materialization, _request, scenario = _scenario_parts(live_sampler, command)
    assert tuple(scenario.current_root_artifact_t.shape) == (7,)
    assert tuple(scenario.intent_q29.shape) == (3, 29)
    assert tuple(scenario.clean_continuation.shape) == (3, 65)
    assert not hasattr(scenario, "reference_sequence")
    print(
        "[T-schema] "
        f"artifact={tuple(scenario.current_root_artifact_t.shape)} "
        f"intent={tuple(scenario.intent_q29.shape)} "
        f"continuation={tuple(scenario.clean_continuation.shape)}",
        flush=True,
    )


def test_single_axis_artifacts_have_hand_computed_root_and_unchanged_q29() -> None:
    commands, _live_sampler, _hooks = _load_modules()
    expected_q29 = torch.stack(
        [20.0 + torch.arange(29), 30.0 + torch.arange(29), 40.0 + torch.arange(29)]
    )
    for axis, sign, component in (("roll", 1.0, 1), ("pitch", -1.0, 2)):
        command = _command(commands)
        command.perturber = _AxisPerturber(
            SimpleNamespace(axis=axis, sign=sign),
            1,
            command.device,
        )
        payload = _materialize(command)
        artifact = payload["current_root_artifact_t"]
        torch.testing.assert_close(artifact[:3], torch.tensor([2.0, 3.0, 4.0]))
        expected_quat = torch.zeros(4)
        expected_quat[0] = torch.cos(torch.tensor(0.125))
        expected_quat[component] = sign * torch.sin(torch.tensor(0.125))
        torch.testing.assert_close(artifact[3:], expected_quat)
        torch.testing.assert_close(payload["intent_q29"], expected_q29)


def test_composite_family_rejects_before_materialization() -> None:
    commands, _live_sampler, hooks = _load_modules()
    command = _command(commands)
    adapter = object.__new__(hooks.FrontRESStage1EnvAdapter)
    adapter.command = command
    adapter._motion_index_for_key = lambda _motion_id: 0
    adapter._frame_index_for_values = lambda frame, _motion_index: int(frame)
    before_instances = len(_RootOnlyPerturber.instances)
    _expect_error(
        ValueError,
        lambda: adapter.materialize_frontres_local_scenario(
            motion_id="motion-0",
            start_frame=2,
            horizon_k=3,
            intent_horizon=2,
            perturbation_family="local_rp+yaw",
            perturbation_strength=0.25,
        ),
    )
    assert len(_RootOnlyPerturber.instances) == before_instances


def test_t_invariant() -> None:
    commands, live_sampler, _hooks = _load_modules()
    command = _command(commands)
    payload = _materialize(command)
    source_q29 = command.motion_dir_loader.gather(
        "joint_pos",
        torch.tensor([0]),
        torch.tensor([2, 3, 4]),
        command.device,
    )
    clean_root = torch.cat(
        [
            command.motion_dir_loader.gather("body_pos_w", torch.tensor([0]), torch.tensor([2]), command.device)[0, 0],
            command.motion_dir_loader.gather("body_quat_w", torch.tensor([0]), torch.tensor([2]), command.device)[0, 0],
        ]
    )
    torch.testing.assert_close(payload["intent_q29"], source_q29)
    torch.testing.assert_close(payload["clean_continuation"][0, :29], source_q29[1])
    assert not torch.equal(payload["current_root_artifact_t"], clean_root)
    isolated = _RootOnlyPerturber.instances[-1]
    assert isolated.root_calls == 1
    assert isolated.quat_calls == 1
    assert isolated.joint_calls == 0
    print("[T-invariant] q29=raw_deployment_carrier root_artifact=selection_time_noisy", flush=True)


def test_t_noisy_q29_owner() -> None:
    commands, _live_sampler, _hooks = _load_modules()
    command = _command(commands)
    intent = command.extract_frontres_noisy_intent_q29(
        motion_index=0,
        start_frame=2,
        intent_horizon=2,
    )
    observed_attrs = tuple(command.motion_dir_loader.gathered_attrs)
    expected = command.motion_dir_loader.gather(
        "joint_pos",
        torch.tensor([0]),
        torch.tensor([2, 3, 4]),
        command.device,
    )
    torch.testing.assert_close(intent, expected)
    assert tuple(intent.shape) == (3, 29)
    assert not intent.requires_grad
    assert observed_attrs and set(observed_attrs) == {"joint_pos"}
    print("[T-noisy-q29-owner] direct owner reads only deployment q29, never root/global", flush=True)


def test_t_noisy_q29_route() -> None:
    commands, _live_sampler, _hooks = _load_modules()
    command = _command(commands)
    supplied = torch.arange(87, dtype=torch.float32).reshape(3, 29)
    calls: list[dict[str, int]] = []

    def extract_frontres_noisy_intent_q29(**kwargs):
        calls.append({name: int(value) for name, value in kwargs.items()})
        return supplied.detach().clone()

    command.extract_frontres_noisy_intent_q29 = extract_frontres_noisy_intent_q29
    payload = _materialize(command)
    assert calls == [{"motion_index": 0, "start_frame": 2, "intent_horizon": 2}]
    torch.testing.assert_close(payload["intent_q29"], supplied)
    print("[T-noisy-q29-route] local materializer consumes the unique q29 extraction owner", flush=True)


def test_t_hash() -> None:
    commands, live_sampler, _hooks = _load_modules()
    command = _command(commands)
    sampler, _payload, materialization, request, scenario = _scenario_parts(live_sampler, command)

    altered_artifact = materialization.current_root_artifact_t.clone()
    altered_artifact[0] += 1.0
    altered_intent = materialization.intent_q29.clone()
    altered_intent[0, 0] += 1.0
    altered_continuation = materialization.clean_continuation.clone()
    altered_continuation[0, 0] += 1.0
    altered_support = materialization.expected_support.clone()
    altered_support[0, 0] = 1.0 - altered_support[0, 0]
    altered_envelope = materialization.expected_support_envelope.clone()
    altered_envelope[0, 0] += 0.1
    altered_source_provenance = dict(materialization.provenance)
    altered_source_provenance["intent_q29_source"] = "motion_internal_q29_v2"
    extended_intent = torch.cat([materialization.intent_q29, materialization.intent_q29[-1:]], dim=0)
    alternatives = (
        sampler.FrontRESLocalScenario.from_materialization(
            request, replace(materialization, current_root_artifact_t=altered_artifact)
        ),
        sampler.FrontRESLocalScenario.from_materialization(request, replace(materialization, intent_q29=altered_intent)),
        sampler.FrontRESLocalScenario.from_materialization(
            request, replace(materialization, clean_continuation=altered_continuation)
        ),
        sampler.FrontRESLocalScenario.from_materialization(
            request, replace(materialization, expected_support=altered_support)
        ),
        sampler.FrontRESLocalScenario.from_materialization(
            request, replace(materialization, expected_support_envelope=altered_envelope)
        ),
        sampler.FrontRESLocalScenario.from_materialization(
            request, replace(materialization, provenance=altered_source_provenance)
        ),
        sampler.FrontRESLocalScenario.from_materialization(
            replace(request, x_t_identity="motion-0:frame-3:segment-7"), materialization
        ),
        sampler.FrontRESLocalScenario.from_materialization(
            replace(request, future_offsets=(1, 3)),
            replace(materialization, intent_q29=extended_intent),
        ),
        sampler.FrontRESLocalScenario.from_materialization(
            replace(request, horizon_k=2),
            replace(
                materialization,
                clean_continuation=materialization.clean_continuation[:2],
                expected_support=materialization.expected_support[:2],
                expected_support_envelope=materialization.expected_support_envelope[:2],
            ),
        ),
    )
    assert all(other.noisy_segment_hash != scenario.noisy_segment_hash for other in alternatives)
    revisit = sampler.FrontRESLocalScenario.from_materialization(
        replace(
            request,
            transaction_id="tx-revisit",
            scenario_id="scenario-revisit",
            source_index=9,
        ),
        materialization,
    )
    assert revisit.noisy_segment_hash == scenario.noisy_segment_hash
    print("[T-hash] x_t/artifact/intent-source/window/continuation/support/K each affect noisy_segment_hash", flush=True)


def test_t_provenance() -> None:
    commands, live_sampler, hooks = _load_modules()
    command = _command(commands)
    adapter = object.__new__(hooks.FrontRESStage1EnvAdapter)
    adapter.command = command
    adapter._motion_index_for_key = lambda _motion_id: 0
    adapter._frame_index_for_values = lambda frame, _motion_index: int(frame)
    payload = adapter.materialize_frontres_local_scenario(
        motion_id="motion-0",
        start_frame=2,
        horizon_k=3,
        intent_horizon=2,
        perturbation_family="local_rp",
        perturbation_strength=0.25,
    )
    assert payload["provenance"]["intent_q29_provenance"] == "deployment_noisy_q29"
    assert payload["provenance"]["clean_continuation_provenance"] == "clean_gmt_only"
    assert payload["provenance"]["expected_support_provenance"] == "clean_gmt_physics_only"
    assert payload["provenance"]["expected_support_envelope_provenance"] == "clean_gmt_physics_only"
    assert "root" not in payload["provenance"]["intent_q29_source"]
    assert "global" not in payload["provenance"]["intent_q29_source"]
    assert tuple(payload["intent_q29"].shape) == (3, 29)
    assert tuple(payload["clean_continuation"].shape) == (3, 65)
    print("[T-provenance] adapter=q29 deployment carrier; continuation=GMT-only Clean carrier", flush=True)


def test_t_frame_budget_rejects_only_current_k_ineligible_segments() -> None:
    commands, _live_sampler, hooks = _load_modules()
    command = _command(commands)
    adapter = object.__new__(hooks.FrontRESStage1EnvAdapter)
    adapter.command = command
    adapter._motion_index_for_key = lambda _motion_id: 0

    assert adapter.frontres_local_scenario_is_materializable(
        motion_id="motion-0", start_frame=23, horizon_k=8, intent_horizon=2
    )
    assert not adapter.frontres_local_scenario_is_materializable(
        motion_id="motion-0", start_frame=24, horizon_k=8, intent_horizon=2
    )
    assert adapter.frontres_local_scenario_is_materializable(
        motion_id="motion-0", start_frame=24, horizon_k=4, intent_horizon=2
    )
    assert not adapter.frontres_local_scenario_is_materializable(
        motion_id="motion-0", start_frame=28, horizon_k=2, intent_horizon=4
    )
    print("[T-frame-budget] eligibility is max(K,H)-conditioned and never clamps the segment start", flush=True)


def test_t_metamorphic() -> None:
    commands, live_sampler, _hooks = _load_modules()
    command = _command(commands)
    sampler, _payload, materialization, _request, _scenario = _scenario_parts(live_sampler, command)
    adapter = SimpleNamespace()
    adapter.calls = []

    def materialize_frontres_local_scenario(**kwargs):
        adapter.calls.append(dict(kwargs))
        return {
            "current_root_artifact_t": materialization.current_root_artifact_t.detach().clone(),
            "clean_reference_t": materialization.clean_reference_t.detach().clone(),
            "intent_q29": materialization.intent_q29.detach().clone(),
            "clean_continuation": materialization.clean_continuation.detach().clone(),
            "expected_support": materialization.expected_support.detach().clone(),
            "expected_support_envelope": materialization.expected_support_envelope.detach().clone(),
            "provenance": dict(materialization.provenance),
        }

    adapter.materialize_frontres_local_scenario = materialize_frontres_local_scenario
    batch = SimpleNamespace(
        segment_ids=torch.tensor([7, 7], dtype=torch.long),
        specs=(
            SimpleNamespace(motion_id="motion-0", start_frame=2),
            SimpleNamespace(motion_id="motion-0", start_frame=2),
        ),
        perturbation_family=("index_only", "index_only"),
        stage3_index_perturbation_family=("local_rp", "local_rp"),
        stage3_index_perturbation_strength=torch.tensor([0.25, 0.25]),
        frontres_segment_budget_horizon_k=torch.tensor([3, 3], dtype=torch.long),
        frontres_segment_source_index=torch.tensor([0, 0], dtype=torch.long),
    )
    runner = SimpleNamespace(
        current_learning_iteration=0,
        alg=SimpleNamespace(frontres_future_offsets=(1, 2)),
        env=SimpleNamespace(_frontres_segment_index_reset_adapter=adapter),
    )
    batch = live_sampler._attach_frontres_local_scenarios(runner, batch, _sample(sampler), update_step=1)
    assert len(adapter.calls) == 1
    assert batch.frontres_local_scenario_ids[0] == batch.frontres_local_scenario_ids[1]
    assert batch.frontres_local_scenario_hashes[0] == batch.frontres_local_scenario_hashes[1]
    assert not hasattr(batch, "frontres_fixed_noisy_tape")
    before = batch.frontres_local_scenario_rows.scenarios[0].intent_q29
    leaked_copy = batch.frontres_local_scenario_rows.scenarios[0].intent_q29
    leaked_copy[0, 0] += 999.0
    torch.testing.assert_close(batch.frontres_local_scenario_rows.scenarios[0].intent_q29, before)
    live_sampler._close_frontres_local_scenarios(batch)
    _expect_error(
        RuntimeError,
        lambda: batch.frontres_local_scenario_lifecycle.bind_rows(_sample(sampler)),
    )
    print("[T-metamorphic] M retries share one sealed identity; accessor mutation and rematerialization are rejected", flush=True)


def test_t_legacy_reject() -> None:
    commands, live_sampler, hooks = _load_modules()
    command = _command(commands)
    sampler, _payload, materialization, request, _scenario = _scenario_parts(live_sampler, command)
    _expect_error(
        ValueError,
        lambda: sampler.FrontRESLocalScenario.from_materialization(
            request,
            replace(materialization, intent_q29=torch.zeros(3, 65)),
        ),
    )
    adapter = object.__new__(hooks.FrontRESStage1EnvAdapter)
    adapter.command = SimpleNamespace(_frontres_fixed_noisy_tape_feature_dim=lambda: 65)
    assert adapter._fixed_noisy_reset_payload(
        SimpleNamespace(frontres_local_scenario_rows=object()),
        source_count=1,
        device=torch.device("cpu"),
    ) is None
    _expect_error(
        ValueError,
        lambda: adapter._v015_local_scenario_reset_payload(
            SimpleNamespace(
                frontres_local_scenario_rows=object(),
                frontres_fixed_noisy_tape=torch.zeros(1, 1, 65),
            ),
            source_count=1,
            device=torch.device("cpu"),
        ),
    )
    print("[T-legacy-reject] 65D intent and local/fixed-tape mixing are rejected", flush=True)


def test_t_fixed_heldout_manifest_item() -> None:
    commands, live_sampler, _hooks = _load_modules()
    command = _command(commands)
    _sampler, _payload, materialization, _request, _scenario = _scenario_parts(
        live_sampler, command
    )
    calls: list[dict[str, object]] = []

    def materialize_frontres_local_scenario(**kwargs):
        calls.append(dict(kwargs))
        horizon_k = int(kwargs["horizon_k"])
        return {
            "current_root_artifact_t": materialization.current_root_artifact_t.detach().clone(),
            "clean_reference_t": materialization.clean_reference_t.detach().clone(),
            "intent_q29": materialization.intent_q29.detach().clone(),
            "clean_continuation": torch.arange(
                horizon_k * 65,
                dtype=torch.float32,
            ).reshape(horizon_k, 65),
            "expected_support": torch.ones(horizon_k, 2, dtype=torch.float32),
            "expected_support_envelope": torch.tensor(
                [[0.0, 0.0, 1.0, 0.0, 0.1, 0.05]], dtype=torch.float32
            ).repeat(horizon_k, 1),
            "provenance": dict(materialization.provenance),
        }

    adapter = SimpleNamespace(materialize_frontres_local_scenario=materialize_frontres_local_scenario)
    spec = SimpleNamespace(
        segment_id=7,
        motion_id="motion-0",
        start_frame=2,
        horizon_k=4,
        perturbation_family="index_only",
    )

    class Dataset:
        _specs = (spec,)

        def get_segments(self, segment_ids):
            count = int(segment_ids.numel())
            return SimpleNamespace(
                segment_ids=segment_ids.detach().clone(),
                batch_size=count,
                specs=(spec,) * count,
                perturbation_family=("index_only",) * count,
                perturbation_strength=torch.zeros(count),
            )

    runner = SimpleNamespace(
        device=torch.device("cpu"),
        current_learning_iteration=0,
        alg=SimpleNamespace(frontres_future_offsets=(1, 2)),
        env=SimpleNamespace(num_envs=8, _frontres_segment_index_reset_adapter=adapter),
        _frontres_segment_dataset=Dataset(),
    )
    item = SimpleNamespace(
        motion_id="motion-0",
        start_frame=2,
        effective_horizon_k=8,
        perturbation_family="local_rp",
        perturbation_parameters=(("dr_scale", 0.25),),
        seed=42,
        comparison_signature="a" * 64,
    )
    rng_before = torch.random.get_rng_state().clone()
    first = live_sampler.prepare_frontres_v015_policy_quality_item_batch(runner, item)
    second = live_sampler.prepare_frontres_v015_policy_quality_item_batch(runner, item)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert len(calls) == 2
    assert all(call["horizon_k"] == 8 for call in calls)
    assert tuple(first.sample.horizon_k.tolist()) == (8, 8, 8, 8)
    assert tuple(first.batch.frontres_local_scenario_clean_continuation.shape) == (4, 8, 65)
    assert tuple(first.sample.source_index.tolist()) == (0, 0, 0, 0)
    assert len(set(first.batch.frontres_local_scenario_ids)) == 1
    assert first.batch.frontres_local_scenario_ids == second.batch.frontres_local_scenario_ids
    assert first.batch.frontres_local_scenario_hashes == second.batch.frontres_local_scenario_hashes
    assert tuple(first.batch.stage3_index_perturbation_family) == ("local_rp",) * 4
    torch.testing.assert_close(
        first.batch.stage3_index_perturbation_strength,
        torch.full((4,), 0.25),
    )
    duplicate_spec = SimpleNamespace(
        segment_id=8,
        motion_id=spec.motion_id,
        start_frame=spec.start_frame,
        horizon_k=8,
        perturbation_family="index_only",
    )
    Dataset._specs = (spec, duplicate_spec)
    _expect_error(
        RuntimeError,
        lambda: live_sampler.prepare_frontres_v015_policy_quality_item_batch(runner, item),
    )
    print(
        "[T-heldout-manifest] K4 index identity -> K8 budget/continuation; "
        "duplicate motion/start rejects without RNG drift"
    )


def test_t_v018_heldout_k16_m4_transaction() -> None:
    commands, live_sampler, _hooks = _load_modules()
    command = _command(commands)
    _sampler, _payload, materialization, _request, _scenario = _scenario_parts(
        live_sampler, command
    )
    calls: list[dict[str, object]] = []

    def materialize_frontres_local_scenario(**kwargs):
        calls.append(dict(kwargs))
        horizon_k = int(kwargs["horizon_k"])
        motion = str(kwargs["motion_id"])
        marker = 1.0 if motion == "motion-a" else 2.0
        return {
            "current_root_artifact_t": torch.full((7,), marker),
            "clean_reference_t": torch.zeros(65),
            "intent_q29": torch.full((3, 29), marker),
            "clean_continuation": torch.full((horizon_k, 65), marker),
            "expected_support": torch.ones(horizon_k, 2),
            "expected_support_envelope": torch.tensor(
                [[0.0, 0.0, 1.0, 0.0, 0.1, 0.05]], dtype=torch.float32
            ).repeat(horizon_k, 1),
            "provenance": dict(materialization.provenance),
        }

    specs = (
        SimpleNamespace(segment_id=7, motion_id="motion-a", start_frame=2, horizon_k=4, perturbation_family="index_only"),
        SimpleNamespace(segment_id=8, motion_id="motion-b", start_frame=5, horizon_k=4, perturbation_family="index_only"),
    )

    class Dataset:
        _specs = specs

        def resolve_segment_spec(self, *, motion_id, start_frame):
            matches = tuple(
                spec for spec in self._specs
                if spec.motion_id == motion_id and spec.start_frame == start_frame
            )
            if len(matches) != 1:
                raise RuntimeError("fixture motion/frame mismatch")
            return matches[0]

        def get_segments(self, segment_ids):
            selected = tuple(self._specs[int(value) - 7] for value in segment_ids.tolist())
            count = int(segment_ids.numel())
            return SimpleNamespace(
                segment_ids=segment_ids.detach().clone(),
                batch_size=count,
                specs=selected,
                perturbation_family=("index_only",) * count,
                perturbation_strength=torch.zeros(count),
            )

    runner = SimpleNamespace(
        device=torch.device("cpu"),
        current_learning_iteration=3500,
        alg=SimpleNamespace(frontres_future_offsets=(1, 2), policy=torch.nn.Linear(2, 2)),
        env=SimpleNamespace(
            num_envs=16,
            _frontres_segment_index_reset_adapter=SimpleNamespace(
                materialize_frontres_local_scenario=materialize_frontres_local_scenario
            ),
        ),
        _frontres_segment_dataset=Dataset(),
    )
    items = (
        SimpleNamespace(
            item_id="a", motion_id="motion-a", start_frame=2, effective_horizon_k=16,
            perturbation_family="local_rp", perturbation_parameters=(("dr_scale", 1.0),),
            seed=42, comparison_signature="a" * 64,
        ),
        SimpleNamespace(
            item_id="b", motion_id="motion-b", start_frame=5, effective_horizon_k=16,
            perturbation_family="local_rp", perturbation_parameters=(("dr_scale", 1.5),),
            seed=43, comparison_signature="b" * 64,
        ),
    )
    rng_before = torch.random.get_rng_state().clone()
    prepared = live_sampler.prepare_frontres_policy_quality_fixed_k_m4_batch(
        runner, items, attempts_per_segment=4
    )
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert len(calls) == 2
    assert tuple(prepared.sample.source_index.tolist()) == (0, 0, 0, 0, 1, 1, 1, 1)
    assert tuple(prepared.sample.trial_index.tolist()) == (0, 1, 2, 3, 0, 1, 2, 3)
    assert tuple(prepared.sample.horizon_k.tolist()) == (16,) * 8
    assert prepared.plan.active_m == 4 and prepared.plan.selected_segment_count == 2
    assert len(set(prepared.plan.scenario_ids[:4])) == 1
    assert len(set(prepared.plan.scenario_ids[4:])) == 1
    assert prepared.plan.scenario_ids[0] != prepared.plan.scenario_ids[4]
    assert tuple(prepared.batch.frontres_local_scenario_clean_continuation.shape) == (8, 16, 65)
    torch.testing.assert_close(
        prepared.batch.stage3_index_perturbation_strength,
        torch.tensor([1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5]),
    )
    k8_items = tuple(
        SimpleNamespace(**{**item.__dict__, "effective_horizon_k": 8})
        for item in items
    )
    prepared_k8 = live_sampler.prepare_frontres_policy_quality_fixed_k_m4_batch(
        runner, k8_items, attempts_per_segment=4
    )
    assert tuple(prepared_k8.sample.horizon_k.tolist()) == (8,) * 8
    assert tuple(prepared_k8.batch.frontres_local_scenario_clean_continuation.shape) == (8, 8, 65)

    original_snapshot = live_sampler.capture_frontres_frozen_policy_snapshot
    original_close = live_sampler.close_frontres_local_scenarios
    closed_batches: list[object] = []

    def fail_snapshot(*_args, **_kwargs):
        raise RuntimeError("injected post-materialization snapshot failure")

    def record_close(batch):
        original_close(batch)
        closed_batches.append(batch)

    live_sampler.capture_frontres_frozen_policy_snapshot = fail_snapshot
    live_sampler.close_frontres_local_scenarios = record_close
    try:
        _expect_error(
            RuntimeError,
            lambda: live_sampler.prepare_frontres_policy_quality_fixed_k_m4_batch(
                runner, k8_items, attempts_per_segment=4
            ),
        )
    finally:
        live_sampler.capture_frontres_frozen_policy_snapshot = original_snapshot
        live_sampler.close_frontres_local_scenarios = original_close
    assert len(closed_batches) == 1
    assert len(tuple(closed_batches[0].frontres_local_scenario_closed_ids)) == 2

    _expect_error(
        RuntimeError,
        lambda: live_sampler.prepare_frontres_policy_quality_fixed_k_m4_batch(
            SimpleNamespace(**{**runner.__dict__, "env": SimpleNamespace(num_envs=12)}),
            items,
            attempts_per_segment=4,
        ),
    )
    print("[T-v018-heldout] two Segment x M4, K16 and source-shared scenario identities are sealed")


def main() -> None:
    test_t_schema()
    test_single_axis_artifacts_have_hand_computed_root_and_unchanged_q29()
    test_composite_family_rejects_before_materialization()
    test_t_invariant()
    test_t_noisy_q29_owner()
    test_t_noisy_q29_route()
    test_t_hash()
    test_t_provenance()
    test_t_frame_budget_rejects_only_current_k_ineligible_segments()
    test_t_metamorphic()
    test_t_fixed_heldout_manifest_item()
    test_t_v018_heldout_k16_m4_transaction()
    test_t_legacy_reject()
    print("frontres_local_scenario_kernel_contract: ok", flush=True)


if __name__ == "__main__":
    main()
