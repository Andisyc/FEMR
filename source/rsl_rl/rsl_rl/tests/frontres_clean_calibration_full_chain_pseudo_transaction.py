#!/usr/bin/env python3
"""Full local manifest-to-receipt transaction for Clean calibration.

The fixture replaces cache payload construction and IsaacLab sensor/device
effects only.  Production owns materialization, reset, K-step collection,
telemetry reduction, read-only lifecycle, adapter validation, and receipt IO.
The success case executes the exact production train.py dispatch helper and
the exact OnPolicyRunner connectors.  IsaacLab cache/sensor effects remain
fakes, so this is an official R1 pseudo-transaction, not live runtime evidence.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from frontres_contract_imports import install_frontres_contract_packages


ROOT = Path(__file__).resolve().parents[4]
RSL_ROOT = ROOT / "source" / "rsl_rl" / "rsl_rl"
install_frontres_contract_packages(RSL_ROOT)

from rsl_rl.runners.frontres_clean_calibration_gateway import (  # noqa: E402
    FRONTRES_CLEAN_CALIBRATION_ROUTE_ID,
    collect_frontres_clean_calibration_from_manifest,
    collect_frontres_clean_calibration_raw_gateway,
    compute_frontres_clean_calibration_state_fingerprint,
)
from rsl_rl.modules.front_residual_actor_critic import FrontRESActorCritic  # noqa: E402


def _load_official_connectors():
    train_tree = ast.parse((ROOT / "scripts" / "rsl_rl" / "train.py").read_text(encoding="utf-8"))
    dispatch_node = next(
        node
        for node in train_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_frontres_clean_calibration_collect_only"
    )
    dispatch_namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=[dispatch_node], type_ignores=[]), "train.py", "exec"), dispatch_namespace)

    runner_tree = ast.parse(
        (RSL_ROOT / "runners" / "on_policy_runner.py").read_text(encoding="utf-8")
    )
    runner_class = next(
        node
        for node in runner_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "OnPolicyRunner"
    )
    method_nodes = [
        node
        for node in runner_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "run_frontres_clean_calibration_collect",
            "run_frontres_clean_calibration_collect_typed",
        }
    ]
    method_namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=method_nodes, type_ignores=[]), "on_policy_runner.py", "exec"), method_namespace)
    return dispatch_namespace["_run_frontres_clean_calibration_collect_only"], method_namespace


_RUN_OFFICIAL_CLEAN_CALIBRATION, _OFFICIAL_RUNNER_METHODS = _load_official_connectors()


class _ContactView:
    def __init__(self, *, x: float, num_envs: int) -> None:
        self._x = float(x)
        self._y = 0.0
        self._num_envs = int(num_envs)
        self._loaded = True

    def set_xy(self, x: float, y: float) -> None:
        self._x = float(x)
        self._y = float(y)

    def set_loaded(self, loaded: bool) -> None:
        self._loaded = bool(loaded)

    def get_contact_data(self, *, dt: float):
        assert dt > 0.0
        count = self._num_envs
        force = torch.full((count,), 100.0 if self._loaded else 0.0)
        points = torch.zeros(count, 3)
        points[:, 0] = self._x
        points[:, 1] = self._y
        normals = torch.zeros(count, 3)
        normals[:, 2] = 1.0
        distance = torch.zeros(count)
        counts = torch.ones(count, dtype=torch.long)
        starts = torch.arange(count, dtype=torch.long)
        return force, points, normals, distance, counts, starts


class _Sensor:
    def __init__(self, *, x: float, num_envs: int) -> None:
        force_matrix = torch.zeros(num_envs, 1, 1, 3)
        force_matrix[..., 2] = 100.0
        self.data = SimpleNamespace(force_matrix_w=force_matrix)
        self.cfg = SimpleNamespace(force_threshold=1.0)
        self.contact_physx_view = _ContactView(x=x, num_envs=num_envs)
        self._frontres_raw_contact_capacity = 4 * num_envs
        self._sim_physics_dt = 0.005

    def set_xy(self, x: float, y: float) -> None:
        self.contact_physx_view.set_xy(x, y)

    def set_loaded(self, loaded: bool) -> None:
        self.data.force_matrix_w[..., 2] = 100.0 if loaded else 0.0
        self.contact_physx_view.set_loaded(loaded)


class _Command:
    def __init__(self, *, num_envs: int) -> None:
        self.num_envs = int(num_envs)
        self.left_foot_idx = 0
        self.right_foot_idx = 1
        self.robot_joint_pos = torch.zeros(num_envs, 29)
        self.robot_anchor_pos_w = torch.tensor([[0.0, 0.0, 0.8]]).repeat(num_envs, 1)
        self.robot_anchor_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_envs, 1)
        self.robot_body_pos_w = torch.zeros(num_envs, 2, 3)
        self.robot_body_pos_w[:, 0, 0] = -0.1
        self.robot_body_pos_w[:, 1, 0] = 0.1
        self.robot_anchor_lin_vel_w = torch.zeros(num_envs, 3)
        self.robot_anchor_ang_vel_w = torch.zeros(num_envs, 3)
        self._active = False
        self._repeat_noise = 0.0
        self._step = 0
        self.begin_count = 0
        self.end_count = 0
        self.advance_count = 0

    def configure_repeat(self, noise: float) -> None:
        self._repeat_noise = float(noise)
        self._step = 0

    def advance_dynamics(self) -> None:
        self._step += 1
        row_factor = 1.0 + 0.05 * torch.arange(self.num_envs, dtype=torch.float32)
        scale = self._repeat_noise * float(self._step) * row_factor
        self.robot_anchor_pos_w[:, 0] = 0.25 * scale
        self.robot_anchor_pos_w[:, 1] = 0.1 * scale
        self.robot_anchor_lin_vel_w[:, 0] = scale
        self.robot_anchor_lin_vel_w[:, 1] = 0.75 * scale
        self.robot_anchor_ang_vel_w[:, 2] = 0.5 * scale
        self.robot_body_pos_w[:, 0, 0] = -0.1 + scale
        self.robot_body_pos_w[:, 1, 0] = 0.1 + scale
        self.robot_body_pos_w[:, :, 1] = 0.2 * scale[:, None]

    def begin_frontres_local_scenario_k_execution(self) -> None:
        if self._active:
            raise RuntimeError("duplicate Clean K execution begin")
        self._active = True
        self.begin_count += 1

    def advance_frontres_local_scenario_k_execution(self) -> dict[str, torch.Tensor]:
        if not self._active:
            raise RuntimeError("Clean K execution advanced before begin")
        self.advance_count += 1
        return {"valid_mask": torch.ones(self.num_envs, dtype=torch.bool)}

    def end_frontres_local_scenario_k_execution(self) -> None:
        if not self._active:
            raise RuntimeError("Clean K execution ended before begin")
        self._active = False
        self.end_count += 1

    def frontres_local_scenario_k_execution_snapshot(self) -> dict[str, torch.Tensor]:
        if not self._active:
            raise RuntimeError("Clean K snapshot requested outside execution")
        envelope = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.1, 0.05]]).repeat(self.num_envs, 1)
        return {
            "expected_support": torch.ones(self.num_envs, 2),
            "expected_support_envelope": envelope,
        }

    def set_frontres_v015_two_role_baseline(self, *, n_repair: int, n_noisy: int) -> None:
        assert (n_repair, n_noisy) == (8, 8)

    def clear_frontres_local_scenario(self) -> None:
        if self._active:
            raise RuntimeError("cannot clear an active Clean K execution")


class _MaterializerAdapter:
    def materialize_frontres_local_scenario(self, **kwargs):
        horizon_k = int(kwargs["horizon_k"])
        marker = 1.0 if str(kwargs["motion_id"]).endswith("motion-a.npz") else 2.0
        return {
            "current_root_artifact_t": torch.full((7,), marker),
            "clean_reference_t": torch.zeros(65),
            "intent_q29": torch.full((3, 29), marker),
            "clean_continuation": torch.full((horizon_k, 65), marker),
            "expected_support": torch.ones(horizon_k, 2),
            "expected_support_envelope": torch.tensor(
                [[0.0, 0.0, 1.0, 0.0, 0.1, 0.05]], dtype=torch.float32
            ).repeat(horizon_k, 1),
            "provenance": {
                "current_root_artifact_provenance": "noisy_root_artifact_t",
                "clean_reference_t_provenance": "clean_gmt_physics_only",
                "intent_q29_provenance": "deployment_noisy_q29",
                "clean_continuation_provenance": "clean_gmt_only",
                "expected_support_provenance": "clean_gmt_physics_only",
                "expected_support_envelope_provenance": "clean_gmt_physics_only",
                "intent_q29_source": "sealed-noisy-command",
            },
        }


class _Dataset:
    def __init__(self) -> None:
        self._specs = (
            SimpleNamespace(segment_id=7, motion_id="motion-a.npz", start_frame=10, horizon_k=4),
            SimpleNamespace(segment_id=8, motion_id="motion-b.npz", start_frame=20, horizon_k=4),
        )
        self.last_batch = None

    def resolve_segment_spec(self, *, motion_id: str, start_frame: int):
        matches = tuple(
            spec for spec in self._specs
            if spec.motion_id == motion_id and int(spec.start_frame) == int(start_frame)
        )
        if len(matches) != 1:
            raise RuntimeError("pseudo cache identity mismatch")
        return matches[0]

    def get_segments(self, segment_ids: torch.Tensor):
        selected = tuple(self._specs[int(value) - 7] for value in segment_ids.tolist())
        count = int(segment_ids.numel())
        self.last_batch = SimpleNamespace(
            segment_ids=segment_ids.detach().clone(),
            batch_size=count,
            specs=selected,
            perturbation_family=("index_only",) * count,
            perturbation_strength=torch.zeros(count, device=segment_ids.device),
        )
        return self.last_batch


class _Environment:
    def __init__(self, command: _Command) -> None:
        self.num_envs = command.num_envs
        self.device = torch.device("cpu")
        self.unwrapped = self
        class CommandManager:
            def __init__(self, motion: _Command) -> None:
                self._terms = {"motion": motion}

            def get_term(self, name: str):
                return self._terms[name]

        self.command_manager = CommandManager(command)
        self._frontres_segment_index_reset_adapter = _MaterializerAdapter()
        class Scene(dict):
            pass

        self.scene = Scene({
            "frontres_left_foot_contacts": _Sensor(x=-0.1, num_envs=self.num_envs),
            "frontres_right_foot_contacts": _Sensor(x=0.1, num_envs=self.num_envs),
        })
        self.scene.env_origins = torch.zeros(self.num_envs, 3)
        self._command = command
        self.reset_count = 0
        self.step_count = 0
        self.close_count = 0
        self.fail_hard_event_on_repeat: int | None = None
        self.mutation_callback = None

    def get_observations(self):
        return torch.zeros(self.num_envs, 770), {"observations": {}}

    def step(self, actions: torch.Tensor):
        assert tuple(actions.shape) == (self.num_envs, 29)
        self.step_count += 1
        self._command.advance_dynamics()
        return (
            torch.zeros(self.num_envs, 770),
            torch.zeros(self.num_envs),
            torch.zeros(self.num_envs, dtype=torch.bool),
            {},
        )

    def apply_frontres_segment_index_reset(self, request):
        assert request.frontres_local_scenario_execution_mode == "clean_baseline"
        self.reset_count += 1
        noise = 1.0e-3 + 2.0e-3 * float(torch.rand(()).item())
        self._command.configure_repeat(noise)
        self.scene["frontres_left_foot_contacts"].set_xy(-0.1 + noise, noise)
        self.scene["frontres_right_foot_contacts"].set_xy(0.1 + noise, noise)
        loaded = self.fail_hard_event_on_repeat != self.reset_count
        self.scene["frontres_left_foot_contacts"].set_loaded(loaded)
        self.scene["frontres_right_foot_contacts"].set_loaded(loaded)
        if self.mutation_callback is not None:
            self.mutation_callback()
        return {}

    def close(self) -> None:
        self.close_count += 1


class _Policy(FrontRESActorCritic):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.probe = nn.Linear(2, 2)
        self.num_task_corrections = 6

    def run_frozen_gmt_from_suffix(self, suffix: torch.Tensor) -> torch.Tensor:
        assert tuple(suffix.shape) == (16, 770)
        return torch.zeros(16, 29, device=suffix.device)


class _Runner:
    def __init__(self) -> None:
        command = _Command(num_envs=16)
        self.device = torch.device("cpu")
        self.env = _Environment(command)
        self.policy_obs_type = None
        self._frontres_gmt_obs_dim = 770
        self._frontres_future_intent_layout = SimpleNamespace(
            version="frontres-v015-future-intent-q29-v1"
        )
        self._frontres_extra_normalizer = None
        self.obs_normalizer = nn.Identity()
        self.privileged_obs_normalizer = nn.Identity()
        self.teacher_obs_normalizer = nn.Identity()
        self.current_learning_iteration = 0
        self._frontres_segment_live_detail_log_enabled = False
        self._frontres_segment_dataset = _Dataset()
        self._frontres_segment_sampler = None
        self._frontres_outer_scenario_replay = None
        policy = _Policy()
        self.alg = SimpleNamespace(
            policy=policy,
            optimizer=torch.optim.Adam(policy.parameters(), lr=1.0e-6),
            frontres_future_offsets=(1, 2),
        )
        self.cfg = {"frontres_specialist_mode": "rp"}
        self._frontres_startup_dispatches: set[str] = set()
        self.frontres_clean_calibration_state_fingerprint = MethodType(
            lambda owner, transaction_id: compute_frontres_clean_calibration_state_fingerprint(
                owner, transaction_id
            ),
            self,
        )
        self.run_frontres_clean_calibration_collect = MethodType(
            _OFFICIAL_RUNNER_METHODS["run_frontres_clean_calibration_collect"], self
        )
        self.run_frontres_clean_calibration_collect_typed = MethodType(
            _OFFICIAL_RUNNER_METHODS["run_frontres_clean_calibration_collect_typed"], self
        )

    def _dispatch_frontres_startup_once(self, route: str, operation):
        if route in self._frontres_startup_dispatches:
            raise RuntimeError(f"duplicate startup route: {route}")
        self._frontres_startup_dispatches.add(route)
        return operation()


def _official_args(manifest: Path, result: Path) -> SimpleNamespace:
    return SimpleNamespace(
        frontres_clean_calibration_collect_only=True,
        frontres_clean_calibration_manifest=str(manifest),
        frontres_clean_calibration_result=str(result),
        frontres_policy_quality_eval_only=False,
        frontres_policy_quality_q2d_eval_only=False,
        frontres_action_gain_direction_collect_only=False,
        frontres_segment_live_sentinel_only=False,
        frontres_local_sentinel_only=False,
        frontres_segment_live_probe_only=False,
        frontres_segment_live_storage_write_only=False,
        frontres_segment_live_single_update_only=False,
        frontres_segment_live_update_loop_only=False,
    )


class _CloseSpy:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


def _manifest() -> dict[str, object]:
    return {
        "route_id": FRONTRES_CLEAN_CALIBRATION_ROUTE_ID,
        "calibration_id": FRONTRES_CLEAN_CALIBRATION_ROUTE_ID,
        "domain_id": "frontres-stage3",
        "field_schema_id": "frontres-clean-calibration-fields-v1",
        "horizon_k": 8,
        "preroll_steps": 2,
        "timestep_seconds": 0.02,
        "seed_protocol_id": "clean-repeat-v1",
        "coverage": 0.95,
        "scenario_source_index": 0,
        "repeats": [
            {"repeat_id": "repeat-00", "seed": 100},
            {"repeat_id": "repeat-01", "seed": 101},
        ],
        "segments": [
            {
                "item_id": "segment-00",
                "motion_id": "motion-a.npz",
                "start_frame": 10,
                "perturbation_family": "local_rp",
                "perturbation_parameters": [["strength", 0.0]],
                "effective_horizon_k": 10,
                "seed": 100,
            },
            {
                "item_id": "segment-01",
                "motion_id": "motion-b.npz",
                "start_frame": 20,
                "perturbation_family": "local_rp",
                "perturbation_parameters": [["strength", 0.0]],
                "effective_horizon_k": 10,
                "seed": 101,
            },
        ],
    }


def main() -> None:
    disabled_env = _CloseSpy()
    assert _RUN_OFFICIAL_CLEAN_CALIBRATION(
        SimpleNamespace(frontres_clean_calibration_collect_only=False), object(), disabled_env
    ) is False
    assert disabled_env.close_count == 0

    invalid_args = _official_args(Path("manifest.json"), Path("result.json"))
    invalid_args.frontres_clean_calibration_manifest = None
    invalid_env = _CloseSpy()
    try:
        _RUN_OFFICIAL_CLEAN_CALIBRATION(invalid_args, object(), invalid_env)
    except ValueError as exc:
        assert "missing required arguments" in str(exc)
    else:
        raise AssertionError("missing Clean calibration manifest must fail closed")
    assert invalid_env.close_count == 1

    runner = _Runner()
    with tempfile.TemporaryDirectory(prefix="frontres-clean-chain-") as directory:
        manifest = Path(directory) / "manifest.json"
        result = Path(directory) / "result.json"
        manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
        with (
            patch(
                "rsl_rl.runners.frontres_clean_calibration_gateway.ensure_frontres_readonly_reset_support",
                side_effect=lambda owner: None,
            ),
            patch(
                "rsl_rl.runners.frontres_clean_calibration_gateway.prepare_frontres_raw_contact_views",
                side_effect=lambda owner: None,
            ),
        ):
            assert _RUN_OFFICIAL_CLEAN_CALIBRATION(
                _official_args(manifest, result), runner, runner.env
            ) is True
        persisted = json.loads(result.read_text(encoding="utf-8"))
        receipt = persisted
        assert persisted == json.loads(json.dumps(receipt, default=str))
        assert receipt["route_id"] == FRONTRES_CLEAN_CALIBRATION_ROUTE_ID
        assert receipt["collected_count"] == 2
        assert tuple(receipt["repeat_ids"]) == ("repeat-00", "repeat-01")
        command = runner.env.command_manager._terms["motion"]
        assert command._active is False
        assert (command.begin_count, command.end_count, command.advance_count) == (2, 2, 20)
        assert (runner.env.reset_count, runner.env.step_count) == (2, 22)
        assert runner.env.close_count == 1
        assert len(tuple(runner._frontres_segment_dataset.last_batch.frontres_local_scenario_closed_ids)) == 2

    def collect_candidate_order(order: tuple[int, int]) -> dict[int, tuple[float | int, ...]]:
        torch.manual_seed(7)
        order_runner = _Runner()
        measurements: dict[int, tuple[float | int, ...]] = {}
        with tempfile.TemporaryDirectory(prefix="frontres-clean-order-") as directory:
            for position, source_index in enumerate(order):
                manifest = Path(directory) / f"manifest-{position}.json"
                result = Path(directory) / f"result-{position}.json"
                payload = _manifest()
                payload["scenario_source_index"] = source_index
                manifest.write_text(json.dumps(payload), encoding="utf-8")
                with (
                    patch(
                        "rsl_rl.runners.frontres_clean_calibration_gateway.ensure_frontres_readonly_reset_support",
                        side_effect=lambda owner: None,
                    ),
                    patch(
                        "rsl_rl.runners.frontres_clean_calibration_gateway.prepare_frontres_raw_contact_views",
                        side_effect=lambda owner: None,
                    ),
                ):
                    collected = collect_frontres_clean_calibration_from_manifest(
                        order_runner,
                        manifest_path=str(manifest),
                        result_path=str(result),
                    )
                assert collected["status"] == "OK"
                calibration = collected["calibration"]
                measurements[source_index] = (
                    int(calibration["repeated_sample_count"]),
                    int(calibration["repeated_pair_count"]),
                    float(calibration["capture_margin_resolution"]),
                    float(calibration["capture_trend_resolution"]),
                    float(calibration["zmp_margin_resolution"]),
                    float(calibration["linear_momentum_resolution"]),
                    float(calibration["angular_momentum_resolution"]),
                    float(calibration["support_drift_resolution"]),
                )
        return measurements

    forward_order = collect_candidate_order((0, 1))
    reverse_order = collect_candidate_order((1, 0))
    assert forward_order[0] != forward_order[1], forward_order
    assert forward_order == reverse_order, (forward_order, reverse_order)

    runner = _Runner()
    with tempfile.TemporaryDirectory(prefix="frontres-clean-atomic-") as directory:
        manifest = Path(directory) / "manifest.json"
        result = Path(directory) / "result.json"
        manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
        original_write_text = Path.write_text

        def fail_result_write(path: Path, data: str, *args, **kwargs):
            if path.name in ("result.json", "result.json.tmp"):
                original_write_text(path, "partial", encoding="utf-8")
                raise OSError("injected evaluation report write failure")
            return original_write_text(path, data, *args, **kwargs)

        with (
            patch(
                "rsl_rl.runners.frontres_clean_calibration_gateway.ensure_frontres_readonly_reset_support",
                side_effect=lambda owner: None,
            ),
            patch(
                "rsl_rl.runners.frontres_clean_calibration_gateway.prepare_frontres_raw_contact_views",
                side_effect=lambda owner: None,
            ),
            patch.object(Path, "write_text", fail_result_write),
        ):
            try:
                collect_frontres_clean_calibration_from_manifest(
                    runner,
                    manifest_path=str(manifest),
                    result_path=str(result),
                )
            except OSError as exc:
                assert "injected evaluation report write failure" in str(exc)
            else:
                raise AssertionError("injected report write failure must abort the transaction")
        assert not result.exists()
        assert not result.with_suffix(result.suffix + ".tmp").exists()

    runner = _Runner()
    runner.env.fail_hard_event_on_repeat = 1
    with tempfile.TemporaryDirectory(prefix="frontres-clean-candidate-fallback-") as directory:
        manifest = Path(directory) / "manifest.json"
        result = Path(directory) / "result.json"
        fallback_manifest = _manifest()
        fallback_manifest["fallback_source_indices"] = [1]
        manifest.write_text(json.dumps(fallback_manifest), encoding="utf-8")
        with (
            patch(
                "rsl_rl.runners.frontres_clean_calibration_gateway.ensure_frontres_readonly_reset_support",
                side_effect=lambda owner: None,
            ),
            patch(
                "rsl_rl.runners.frontres_clean_calibration_gateway.prepare_frontres_raw_contact_views",
                side_effect=lambda owner: None,
            ),
        ):
            assert _RUN_OFFICIAL_CLEAN_CALIBRATION(
                _official_args(manifest, result), runner, runner.env
            ) is True
        fallback_result = json.loads(result.read_text(encoding="utf-8"))
        assert fallback_result["status"] == "OK"
        assert fallback_result["selected_source_index"] == 1
        assert [row["source_index"] for row in fallback_result["rejected_candidates"]] == [0]
        assert fallback_result["rejected_candidates"][0]["hard_events"]["expected_support_no_load"] > 0.0
        assert runner.env.close_count == 1

    runner = _Runner()
    runner.env.fail_hard_event_on_repeat = 2
    with tempfile.TemporaryDirectory(prefix="frontres-clean-hard-event-") as directory:
        manifest = Path(directory) / "manifest.json"
        result = Path(directory) / "result.json"
        manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
        with (
            patch(
                "rsl_rl.runners.frontres_clean_calibration_gateway.ensure_frontres_readonly_reset_support",
                side_effect=lambda owner: None,
            ),
            patch(
                "rsl_rl.runners.frontres_clean_calibration_gateway.prepare_frontres_raw_contact_views",
                side_effect=lambda owner: None,
            ),
        ):
            payload = collect_frontres_clean_calibration_from_manifest(
                runner,
                manifest_path=str(manifest),
                result_path=str(result),
            )
        assert payload["status"] == "TELEMETRY-GAP"
        assert payload["selected_source_index"] is None
        assert tuple(payload["candidate_source_indices"]) == (0,)
        assert len(payload["rejected_candidates"]) == 1
        rejection = payload["rejected_candidates"][0]
        assert rejection["source_index"] == 0
        assert rejection["repeat_id"] == "repeat-01"
        assert rejection["hard_events"]["expected_support_no_load"] > 0
        assert result.is_file()
        assert json.loads(result.read_text(encoding="utf-8"))["status"] == "TELEMETRY-GAP"
        assert runner.env.command_manager._terms["motion"]._active is False
        assert len(tuple(runner._frontres_segment_dataset.last_batch.frontres_local_scenario_closed_ids)) == 2

    runner = _Runner()
    runner.env.mutation_callback = lambda: next(runner.alg.policy.parameters()).data.add_(1.0)
    with tempfile.TemporaryDirectory(prefix="frontres-clean-mutation-") as directory:
        manifest = Path(directory) / "manifest.json"
        result = Path(directory) / "result.json"
        manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
        with (
            patch(
                "rsl_rl.runners.frontres_clean_calibration_gateway.ensure_frontres_readonly_reset_support",
                side_effect=lambda owner: None,
            ),
            patch(
                "rsl_rl.runners.frontres_clean_calibration_gateway.prepare_frontres_raw_contact_views",
                side_effect=lambda owner: None,
            ),
        ):
            try:
                collect_frontres_clean_calibration_from_manifest(
                    runner,
                    manifest_path=str(manifest),
                    result_path=str(result),
                )
            except RuntimeError as exc:
                assert "mutated protected state" in str(exc)
            else:
                raise AssertionError("training-state mutation must fail closed")
        assert not result.exists()
        assert runner.env.command_manager._terms["motion"]._active is False
    print("frontres_clean_calibration_full_chain_pseudo_transaction: OFFICIAL_R1_PSEUDO_PASS")


if __name__ == "__main__":
    main()
