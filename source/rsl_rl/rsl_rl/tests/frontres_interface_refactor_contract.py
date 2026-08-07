#!/usr/bin/env python3
"""Deterministic contract for the interface-oriented FrontRES Stage-3 boundary."""

from __future__ import annotations

import ast
from dataclasses import replace
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
from types import ModuleType, SimpleNamespace
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[4]
RSL_ROOT = ROOT / "source" / "rsl_rl" / "rsl_rl"
INTERFACES_PATH = RSL_ROOT / "frontres" / "frontres_interfaces.py"
ENGINE_PATH = RSL_ROOT / "runners" / "frontres_stage3_engine.py"
UPDATE_LOOP_PATH = RSL_ROOT / "runners" / "frontres_segment_live_update_loop.py"
BOUNDARY_PATH = RSL_ROOT / "runners" / "frontres_segment_runner_boundary.py"
LIVE_PROBE_FACADE_PATH = RSL_ROOT / "runners" / "frontres_segment_live_probe.py"
LIVE_PROBE_OWNER_NAMES = (
    "frontres_segment_runtime_types",
    "frontres_segment_probe_logging",
    "frontres_segment_live_policy",
    "frontres_segment_live_reset",
    "frontres_segment_live_storage",
    "frontres_segment_one_action_k",
    "frontres_segment_formal_transaction",
    "frontres_segment_physics",
    "frontres_segment_live_rollout",
    "frontres_segment_probe_reporting",
    "frontres_segment_legacy_probe",
)

ADJACENT_OWNER_NAMES = (
    "frontres_segment_transaction",
    "frontres_segment_sampler_reporting",
    "frontres_segment_training_telemetry",
    "frontres_checkpoint_quality",
    "frontres_policy_quality_legacy",
)

PHASE_D_DOMAIN_OWNERS = {
    "frontres/frontres_segment_planning.py": 350,
    "frontres/frontres_segment_legacy_scenario.py": 450,
    "frontres/frontres_local_scenario.py": 550,
    "frontres/frontres_segment_evidence.py": 950,
    "frontres/frontres_segment_storage_records.py": 450,
    "frontres/frontres_segment_grouped_adapter.py": 180,
    "frontres/frontres_segment_rollout_storage.py": 550,
    "frontres/frontres_update_diagnostics.py": 180,
    "frontres/frontres_local_evaluation.py": 900,
    "frontres/frontres_segment_reporting.py": 600,
    "algorithms/frontres_constraint_projection.py": 550,
    "runners/frontres_policy_quality_interfaces.py": 120,
    "runners/frontres_policy_quality_state.py": 380,
}


def _package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _owners():
    _package("rsl_rl", RSL_ROOT)
    _package("rsl_rl.frontres", RSL_ROOT / "frontres")
    _package("rsl_rl.runners", RSL_ROOT / "runners")
    interfaces = _load("rsl_rl.frontres.frontres_interfaces", INTERFACES_PATH)
    engine = _load("rsl_rl.runners.frontres_stage3_engine", ENGINE_PATH)
    return interfaces, engine


def test_lazy_public_facade() -> None:
    source_root = ROOT / "source" / "rsl_rl"
    sys.path.insert(0, str(source_root))
    before = set(sys.modules)
    interface_module = importlib.import_module("rsl_rl.frontres.frontres_interfaces")
    imported = set(sys.modules).difference(before)
    assert interface_module.FRONTRES_CHECKPOINT_FORMAT == "frontres-v017-checkpoint-v10"
    assert not any(name.startswith("isaaclab") for name in imported)
    facade = importlib.import_module("rsl_rl.frontres")
    assert "FrontRESActionCone" not in facade.__all__
    assert "FrontRESExecutabilityScorer" not in facade.__dict__


def test_evaluation_reporting_refuses_partial_json_csv_commit() -> None:
    """A conflicting companion CSV must leave neither report artifact behind."""

    reporting_path = RSL_ROOT / "runners" / "frontres_evaluation_reporting.py"
    reporting = _load("rsl_rl.runners.frontres_evaluation_reporting", reporting_path)
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "sweep.json"
        csv_path = output.with_suffix(".csv")
        csv_path.write_text("occupied\n", encoding="utf-8")
        _expect_error(RuntimeError, lambda: reporting.write_frontres_json_csv_rows(output, ({"gain": 1.0},)))
        assert not output.exists()
        assert csv_path.read_text(encoding="utf-8") == "occupied\n"


def test_deployment_carrier_exposes_only_public_sequence_contract() -> None:
    """The evaluator must not cross the carrier boundary through private names."""

    sequence_path = RSL_ROOT / "runners" / "frontres_segment_sequence_eval.py"
    tree = ast.parse(sequence_path.read_text(encoding="utf-8"))
    carrier_imports = [
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "rsl_rl.runners.frontres_deployment_carrier"
        for alias in node.names
    ]
    assert carrier_imports
    assert all(not name.startswith("_") for name in carrier_imports)


def _expect_error(error_type, fn) -> None:
    try:
        fn()
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__}")


class _Rejected(RuntimeError):
    pass


class _FakeRequest:
    def __init__(self, interfaces, *, active_k: int = 8, active_m: int = 2) -> None:
        self._view = interfaces.FrontRESActiveTransactionRequestView(
            identity=interfaces.FrontRESActiveContractIdentity(),
            transaction_id="tx-interface",
            policy_snapshot_id="tx-interface:pi-old",
            shape=interfaces.FrontRESActiveTransactionShape(
                active_k=active_k,
                active_m=active_m,
                selected_segment_count=2,
                policy_row_count=2 * active_m,
                role_row_count=4 * active_m,
            ),
            curriculum_fingerprint="a" * 64,
            k_stage_index={8: 0, 16: 1, 32: 2}.get(active_k, 0),
            k_stage_iteration=0,
            training_iteration=0,
            warmup_phase_name="critic_only",
            warmup_actor_loss_weight=0.0,
            dr_stage_fingerprint="b" * 64,
            dr_progress=0.0,
            d_cap=0.5,
        )

    @property
    def transaction_id(self) -> str:
        return self._view.transaction_id

    def frontres_stage3_request_view(self):
        return self._view


class _FakeBackend:
    def __init__(self, interfaces) -> None:
        self.interfaces = interfaces
        self.steps = 0
        self.open_count = 0
        self.abort_count = 0
        self.close_count = 0
        self.build_count = 0
        self.reject_first = False
        self.transaction_enabled = True
        self.training_enabled = True
        self.sentinel = False
        self.receipt_attempt_delta = 0
        self.receipt_transaction_id = "tx-interface"

    def optimizer_step_count(self) -> int:
        return self.steps

    def open_transaction_barrier(self) -> None:
        self.open_count += 1

    def build_training_request(self, *, init_at_random_ep_len: bool):
        assert init_at_random_ep_len
        self.build_count += 1
        if self.reject_first and self.build_count == 1:
            raise _Rejected("invalid evidence")
        return _FakeRequest(self.interfaces)

    def commit_transaction(self, request) -> object:
        assert request.transaction_id == "tx-interface"
        request_view = request.frontres_stage3_request_view()
        before = self.steps
        self.steps += 1
        return SimpleNamespace(
            transaction_id=self.receipt_transaction_id,
            policy_snapshot_id=request_view.policy_snapshot_id,
            segment_count=2,
            policy_attempt_count=request_view.shape.policy_row_count + self.receipt_attempt_delta,
            valid_row_count=request_view.shape.policy_row_count,
            optimizer_step_before=before,
            optimizer_step_after=self.steps,
            optimizer_step_delta=1,
            update_invocation_count=1,
        )

    def abort_training_collection(self) -> None:
        self.abort_count += 1

    def close_training_request(self) -> None:
        self.close_count += 1

    def is_rejected_evidence(self, error: BaseException) -> bool:
        return isinstance(error, _Rejected)

    def formal_transaction_enabled(self) -> bool:
        return self.transaction_enabled

    def formal_training_enabled(self) -> bool:
        return self.training_enabled

    def sentinel_only(self) -> bool:
        return self.sentinel


def test_schema_and_identity(interfaces) -> None:
    interfaces.FrontRESActiveContractIdentity().validate()
    interfaces.FrontRESActiveObservationAuthority().validate()
    for active_k, active_m in ((8, 2), (16, 3), (32, 4)):
        shape = interfaces.FrontRESActiveTransactionShape(
            active_k=active_k,
            active_m=active_m,
            selected_segment_count=2,
            policy_row_count=2 * active_m,
            role_row_count=4 * active_m,
        )
        shape.validate()
    _expect_error(
        ValueError,
        lambda: replace(interfaces.FrontRESActiveObservationAuthority(), frontres_prefix_dim=157).validate(),
    )
    _expect_error(
        ValueError,
        lambda: interfaces.FrontRESActiveTransactionShape(8, 2, 2, 3, 8).validate(),
    )
    _expect_error(
        ValueError,
        lambda: interfaces.FrontRESActiveTransactionShape(16, 2, 2, 4, 8).validate(),
    )
    ramp_view = replace(
        _FakeRequest(interfaces)._view,
        warmup_phase_name="actor_ramp",
        warmup_actor_loss_weight=0.25,
    )
    ramp_view.validate()
    _expect_error(
        ValueError,
        lambda: replace(ramp_view, warmup_phase_name="actor_warmup").validate(),
    )

    telemetry = {
        "method_contract_id": "FRS-METHOD-v017",
        "gain_contract_id": "FRS-GAIN-v007",
        "optimization_contract_id": "FRS-PPO-v005",
            "training_contract_id": "FRS-TRAIN-v015",
        "scalar_target_id": "clean-anchored-recovery-aware-gain-v1",
        "physics_schema_id": "clean-anchored-contact-zmp-survival-v1",
        "grouped_schema_id": "grouped-all-attempt-scalar-v1",
            "checkpoint_format": "frontres-v017-checkpoint-v10",
        "transaction_id": "tx-interface",
        "active_k": 8,
        "active_m": 2,
        "selected_segment_count": 2,
        "policy_row_count": 4,
        "role_row_count": 8,
        "optimizer_step_delta": 1,
        "update_count": 1,
        "actor_learning_rate": 3.0e-6,
        "critic_learning_rate": 1.0e-5,
    }
    interfaces.FrontRESActiveTelemetryView.from_mapping(telemetry)
    missing_checkpoint = dict(telemetry)
    missing_checkpoint.pop("checkpoint_format")
    _expect_error(ValueError, lambda: interfaces.FrontRESActiveTelemetryView.from_mapping(missing_checkpoint))
    wrong_checkpoint = dict(telemetry, checkpoint_format="frontres-v015-checkpoint-v5")
    _expect_error(ValueError, lambda: interfaces.FrontRESActiveTelemetryView.from_mapping(wrong_checkpoint))


def test_engine_exact_one_and_collection_barrier(interfaces, engine_module) -> None:
    runner = SimpleNamespace()
    backend = _FakeBackend(interfaces)
    engine = engine_module.FrontRESStage3Engine(runner, backend)
    result = engine.run_transaction(lambda: _FakeRequest(interfaces))
    assert result.optimizer_step_delta == 1
    assert backend.open_count == 1
    assert backend.steps == 1
    assert engine.lifecycle_state == "idle"
    assert runner._frontres_checkpoint_transaction_state is engine.transaction
    assert engine.transaction == {"state": "idle"}

    bad_backend = _FakeBackend(interfaces)
    bad_engine = engine_module.FrontRESStage3Engine(runner, bad_backend)

    def mutating_provider():
        bad_backend.steps += 1
        return _FakeRequest(interfaces)

    _expect_error(RuntimeError, lambda: bad_engine.run_transaction(mutating_provider))
    assert bad_backend.steps == 1
    assert bad_engine.lifecycle_state == "idle"

    untyped_backend = _FakeBackend(interfaces)
    untyped_engine = engine_module.FrontRESStage3Engine(SimpleNamespace(), untyped_backend)
    _expect_error(TypeError, lambda: untyped_engine.run_transaction(lambda: SimpleNamespace(transaction_id="tx")))
    assert untyped_backend.steps == 0
    assert untyped_engine.lifecycle_state == "idle"

    wrong_m_backend = _FakeBackend(interfaces)
    wrong_m_engine = engine_module.FrontRESStage3Engine(SimpleNamespace(), wrong_m_backend)
    _expect_error(
        ValueError,
        lambda: wrong_m_engine.run_transaction(lambda: _FakeRequest(interfaces, active_k=16, active_m=2)),
    )
    assert wrong_m_backend.steps == 0

    bad_receipt_backend = _FakeBackend(interfaces)
    bad_receipt_backend.receipt_attempt_delta = 1
    bad_receipt_engine = engine_module.FrontRESStage3Engine(SimpleNamespace(), bad_receipt_backend)
    _expect_error(RuntimeError, lambda: bad_receipt_engine.run_transaction(lambda: _FakeRequest(interfaces)))
    assert bad_receipt_engine.lifecycle_state == "idle"

    mixed_receipt_backend = _FakeBackend(interfaces)
    mixed_receipt_backend.receipt_transaction_id = "tx-other"
    mixed_receipt_engine = engine_module.FrontRESStage3Engine(SimpleNamespace(), mixed_receipt_backend)
    _expect_error(RuntimeError, lambda: mixed_receipt_engine.run_transaction(lambda: _FakeRequest(interfaces)))
    assert mixed_receipt_engine.lifecycle_state == "idle"

    cleanup_runner = SimpleNamespace()
    cleanup_backend = _FakeBackend(interfaces)

    def failed_cleanup() -> None:
        raise OSError("command cleanup failed")

    cleanup_backend.abort_training_collection = failed_cleanup
    cleanup_engine = engine_module.FrontRESStage3Engine(cleanup_runner, cleanup_backend)
    _expect_error(OSError, lambda: cleanup_engine.run_transaction(lambda: (_ for _ in ()).throw(ValueError("bad"))))
    assert cleanup_engine.lifecycle_state == "idle"
    assert cleanup_engine.transaction["state"] == "collecting"


def test_training_recollection_and_cleanup(interfaces, engine_module) -> None:
    backend = _FakeBackend(interfaces)
    backend.reject_first = True
    engine = engine_module.FrontRESStage3Engine(SimpleNamespace(), backend)
    result = engine.run_training_transaction(init_at_random_ep_len=True)
    assert result.optimizer_step_delta == 1
    assert backend.build_count == 2
    assert backend.abort_count == 1
    assert backend.close_count == 1
    assert backend.open_count == 2
    assert backend.steps == 1

    failing = _FakeBackend(interfaces)

    def unexpected_request(*, init_at_random_ep_len: bool):
        raise LookupError("unexpected")

    failing.build_training_request = unexpected_request
    failing_engine = engine_module.FrontRESStage3Engine(SimpleNamespace(), failing)
    _expect_error(LookupError, lambda: failing_engine.run_training_transaction())
    assert failing.abort_count == 1
    assert failing.close_count == 1
    assert failing.steps == 0
    assert failing_engine.lifecycle_state == "idle"


def test_explicit_mosaic_backend_bindings(interfaces, engine_module) -> None:
    events: list[str] = []
    optimizer = SimpleNamespace(frontres_step_count=0)
    runner = SimpleNamespace(
        alg=SimpleNamespace(
            optimizer=optimizer,
            frontres_formal_transaction_enabled=True,
            frontres_segment_live_train_enabled=True,
            frontres_local_sentinel_only=False,
        )
    )

    def build_request(owner, *, init_at_random_ep_len: bool):
        assert owner is runner and init_at_random_ep_len
        events.append("build")
        return _FakeRequest(interfaces)

    def commit_update(owner, request):
        assert owner is runner
        view = request.frontres_stage3_request_view()
        before = optimizer.frontres_step_count
        optimizer.frontres_step_count += 1
        events.append("commit")
        return SimpleNamespace(
            transaction_id=view.transaction_id,
            policy_snapshot_id=view.policy_snapshot_id,
            segment_count=view.shape.selected_segment_count,
            policy_attempt_count=view.shape.policy_row_count,
            valid_row_count=view.shape.policy_row_count,
            optimizer_step_before=before,
            optimizer_step_after=optimizer.frontres_step_count,
            optimizer_step_delta=1,
            update_invocation_count=1,
        )

    bindings = engine_module.FrontRESStage3Bindings(
        rejected_error=_Rejected,
        abort_collection=lambda owner: events.append("abort"),
        build_request=build_request,
        close_request=lambda owner: events.append("close"),
        open_barrier=lambda owner: events.append("open"),
        commit_update=commit_update,
    )
    backend = engine_module.MosaicFrontRESStage3Backend(runner, bindings)
    engine = engine_module.get_frontres_stage3_engine(runner, backend=backend)
    result = engine.run_training_transaction()
    assert result.optimizer_step_delta == 1
    assert optimizer.frontres_step_count == 1
    assert events == ["open", "build", "commit", "close"]


def test_engine_resolution_and_static_route(interfaces, engine_module) -> None:
    runner = SimpleNamespace()
    first = engine_module.get_frontres_stage3_engine(runner, backend=_FakeBackend(interfaces))
    second = engine_module.get_frontres_stage3_engine(runner)
    assert first is second

    foreign = SimpleNamespace(_frontres_stage3_engine=object())
    _expect_error(RuntimeError, lambda: engine_module.get_frontres_stage3_engine(foreign))

    engine_source = ENGINE_PATH.read_text(encoding="utf-8")
    update_source = UPDATE_LOOP_PATH.read_text(encoding="utf-8")
    assert "frontres_segment_live_probe" not in engine_source
    assert "FrontRESStage3Bindings" in update_source
    assert "get_frontres_stage3_engine" in update_source


def test_live_probe_facade_has_deep_acyclic_public_owners() -> None:
    runners_root = RSL_ROOT / "runners"
    owner_paths = {name: runners_root / f"{name}.py" for name in LIVE_PROBE_OWNER_NAMES}
    owner_modules = {f"rsl_rl.runners.{name}": name for name in LIVE_PROBE_OWNER_NAMES}

    facade_tree = ast.parse(LIVE_PROBE_FACADE_PATH.read_text(encoding="utf-8"))
    forbidden_facade_nodes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)
    assert not any(isinstance(node, forbidden_facade_nodes) for node in facade_tree.body)
    assert len(LIVE_PROBE_FACADE_PATH.read_text(encoding="utf-8").splitlines()) < 350

    graph: dict[str, set[str]] = {name: set() for name in LIVE_PROBE_OWNER_NAMES}
    for owner_name, path in owner_paths.items():
        source = path.read_text(encoding="utf-8")
        assert len(source.splitlines()) < 1000, f"owner too large: {owner_name}"
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module not in owner_modules:
                continue
            dependency = owner_modules[node.module]
            graph[owner_name].add(dependency)
            private_names = [alias.name for alias in node.names if alias.name.startswith("_")]
            assert not private_names, f"{owner_name} imports private names from {dependency}: {private_names}"

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(owner_name: str) -> None:
        if owner_name in visited:
            return
        assert owner_name not in visiting, f"cyclic live-probe owner dependency at {owner_name}"
        visiting.add(owner_name)
        for dependency in graph[owner_name]:
            visit(dependency)
        visiting.remove(owner_name)
        visited.add(owner_name)

    for owner_name in graph:
        visit(owner_name)

    facade_module = "rsl_rl.runners.frontres_segment_live_probe"
    facade_consumers: list[str] = []
    for path in runners_root.glob("*.py"):
        if path == LIVE_PROBE_FACADE_PATH:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if any(isinstance(node, ast.ImportFrom) and node.module == facade_module for node in ast.walk(tree)):
            facade_consumers.append(path.name)
    assert facade_consumers == []

    runner_tree = ast.parse((runners_root / "on_policy_runner.py").read_text(encoding="utf-8"))
    runner_imports = {
        node.module for node in ast.walk(runner_tree) if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "rsl_rl.runners.frontres_segment_formal_transaction" in runner_imports
    assert "rsl_rl.runners.frontres_segment_legacy_probe" in runner_imports
    assert "rsl_rl.runners.frontres_segment_live_policy" in runner_imports

    direct_routes = {
        "frontres_segment_live_update_loop.py": "rsl_rl.runners.frontres_segment_formal_transaction",
        "frontres_segment_live_training.py": "rsl_rl.runners.frontres_segment_one_action_k",
        "frontres_policy_quality_eval.py": "rsl_rl.runners.frontres_segment_live_reset",
    }
    for filename, expected_module in direct_routes.items():
        tree = ast.parse((runners_root / filename).read_text(encoding="utf-8"))
        imported_modules = {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert expected_module in imported_modules, f"{filename} bypasses {expected_module}"

    formal_source = (runners_root / "frontres_segment_formal_transaction.py").read_text(encoding="utf-8")
    assert "_frontres_local_scenario_active" not in formal_source


def test_active_generation_and_checkpoint_security_are_isolated() -> None:
    active_gain = (RSL_ROOT / "frontres" / "frontres_gain.py").read_text(encoding="utf-8")
    active_evidence = (RSL_ROOT / "frontres" / "frontres_segment_evidence.py").read_text(encoding="utf-8")
    legacy_gain = (RSL_ROOT / "frontres" / "frontres_gain_legacy.py").read_text(encoding="utf-8")
    legacy_evidence = (RSL_ROOT / "frontres" / "frontres_segment_evidence_legacy.py").read_text(encoding="utf-8")
    interfaces = INTERFACES_PATH.read_text(encoding="utf-8")
    checkpointing = (RSL_ROOT / "runners" / "frontres_checkpointing.py").read_text(encoding="utf-8")
    quality = (RSL_ROOT / "runners" / "frontres_checkpoint_quality.py").read_text(encoding="utf-8")

    assert "FrontRESIntentPhysicsGainConfig" not in active_gain
    assert "compute_segment_gain" not in active_gain
    assert "FrontRESRecoveryAwareGainConfig" not in legacy_gain
    assert "FrontRESV015OneActionKEvidence" not in active_evidence
    assert "FrontRESExecutedKTrajectory" not in legacy_evidence
    assert "frontres_gain_legacy" not in active_gain
    assert "frontres_segment_evidence_legacy" not in active_evidence

    old_public_names = (
        "FrontRESV015ContractIdentity",
        "FrontRESV015TransactionRequestView",
        "FrontRESV015CommittedUpdateView",
        "FrontRESV015TelemetryView",
    )
    assert all(name not in interfaces for name in old_public_names)
    assert "FrontRESActiveContractIdentity" in interfaces
    assert "FrontRESActiveTransactionRequestView" in interfaces

    assert "weights_only=False" not in checkpointing
    assert "weights_only=False" not in quality
    assert "weights_only=True" in quality
    assert "load_frontres_checkpoint_mapping" in checkpointing


def test_explicit_run_mode_rejects_boolean_mixing() -> None:
    boundary_module = _load("frontres_interface_runner_boundary", BOUNDARY_PATH)
    base = {
        "frontres_training_objective": "segment_replay_hrl",
        "frontres_segment_replay_enabled": True,
        "frontres_segment_live_runner_enabled": True,
        "frontres_segment_live_train_enabled": True,
    }
    boundary = boundary_module.FrontRESSegmentRunnerBoundary.from_train_cfg({"algorithm": base})
    assert boundary.run_mode.value == "formal_train"
    boundary.assert_live_runner_ready()

    mixed = dict(base)
    mixed["frontres_segment_live_probe_only"] = True
    invalid = boundary_module.FrontRESSegmentRunnerBoundary.from_train_cfg({"algorithm": mixed})
    _expect_error(ValueError, invalid.assert_live_runner_ready)


def test_adjacent_hotspots_use_public_cohesive_owners() -> None:
    runners_root = RSL_ROOT / "runners"
    for owner_name in ADJACENT_OWNER_NAMES:
        path = runners_root / f"{owner_name}.py"
        assert path.is_file(), f"missing Phase C owner: {owner_name}"
        ast.parse(path.read_text(encoding="utf-8"))

    def top_level_defs(filename: str) -> set[str]:
        tree = ast.parse((runners_root / filename).read_text(encoding="utf-8"))
        return {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }

    sampler_defs = top_level_defs("frontres_segment_live_sampler.py")
    assert "FrontRESFormalTransactionPlan" not in sampler_defs
    assert "FrontRESFormalTransactionAccumulator" not in sampler_defs
    assert "_print_sampler_summary" not in sampler_defs

    training_defs = top_level_defs("frontres_segment_live_training.py")
    assert "build_frontres_transaction_telemetry" not in training_defs
    assert "run_frontres_segment_periodic_eval" not in training_defs
    assert "run_frontres_segment_sequence_offline_eval" not in training_defs
    live_training_tree = ast.parse((runners_root / "frontres_segment_live_training.py").read_text(encoding="utf-8"))
    live_training_imports = {
        node.module
        for node in ast.walk(live_training_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "rsl_rl.runners.frontres_segment_training_evaluation" not in live_training_imports

    runner_tree = ast.parse((runners_root / "on_policy_runner.py").read_text(encoding="utf-8"))
    runner_imports = {
        node.module
        for node in ast.walk(runner_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "rsl_rl.runners.frontres_segment_training_evaluation" not in runner_imports
    assert not (runners_root / "frontres_segment_training_evaluation.py").exists()
    assert not (runners_root / "frontres_segment_legacy_sequence_eval.py").exists()

    checkpoint_defs = top_level_defs("frontres_checkpointing.py")
    assert "FrontRESActiveQualityCheckpointIdentity" not in checkpoint_defs
    assert "inspect_frontres_quality_checkpoint" not in checkpoint_defs
    checkpoint_tree = ast.parse((runners_root / "frontres_checkpointing.py").read_text(encoding="utf-8"))
    quality_imports = [
        node
        for node in checkpoint_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "rsl_rl.runners.frontres_checkpoint_quality"
    ]
    assert len(quality_imports) == 1
    assert not any(alias.name.startswith("_") for alias in quality_imports[0].names)

    sequence_defs = top_level_defs("frontres_segment_sequence_eval.py")
    assert "FrontRESSegmentSequenceEvalPlan" not in sequence_defs
    assert "build_frontres_sequence_eval_plan" not in sequence_defs
    sequence_tree = ast.parse((runners_root / "frontres_segment_sequence_eval.py").read_text(encoding="utf-8"))
    sequence_imports = {
        node.module
        for node in ast.walk(sequence_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "rsl_rl.runners.frontres_segment_legacy_sequence_eval" not in sequence_imports

    quality_defs = top_level_defs("frontres_policy_quality_eval.py")
    assert "FrozenFrontRESTaskActor" not in quality_defs
    assert "run_frontres_policy_quality_counterfactuals" not in quality_defs
    assert "run_frontres_legacy_policy_quality_eval" in quality_defs
    quality_source = (runners_root / "frontres_policy_quality_eval.py").read_text(encoding="utf-8")
    assert "legacy evaluation must use run_frontres_legacy_policy_quality_eval explicitly" in quality_source
    quality_tree = ast.parse(quality_source)
    state_imports = [
        node
        for node in ast.walk(quality_tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "rsl_rl.runners.frontres_policy_quality_state"
    ]
    assert len(state_imports) == 1
    assert not any(alias.name.startswith("_") for alias in state_imports[0].names)

    q2d_tree = ast.parse((runners_root / "frontres_policy_quality_q2d_eval.py").read_text(encoding="utf-8"))
    formal_owner_imports = [
        node
        for node in ast.walk(q2d_tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "rsl_rl.runners.frontres_policy_quality_formal_owners"
    ]
    assert len(formal_owner_imports) == 1
    assert not any(alias.name.startswith("_") for alias in formal_owner_imports[0].names)

    legacy_tree = ast.parse((runners_root / "frontres_policy_quality_legacy.py").read_text(encoding="utf-8"))
    runtime_imports = {
        node.module
        for node in legacy_tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "rsl_rl.runners.frontres_policy_quality_eval" not in runtime_imports

    source_root = str(ROOT / "source" / "rsl_rl")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (source_root, env.get("PYTHONPATH", ""))))
    import_orders = (
        ("rsl_rl.runners.frontres_policy_quality_eval", "rsl_rl.runners.frontres_policy_quality_legacy"),
        ("rsl_rl.runners.frontres_policy_quality_legacy", "rsl_rl.runners.frontres_policy_quality_eval"),
    )
    for first, second in import_orders:
        import_code = f"""
import importlib
from pathlib import Path
import sys
from types import ModuleType

root = Path({str(RSL_ROOT)!r})
for name, path in (
    ("rsl_rl", root),
    ("rsl_rl.frontres", root / "frontres"),
    ("rsl_rl.runners", root / "runners"),
):
    package = ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
importlib.import_module({first!r})
importlib.import_module({second!r})
"""
        result = subprocess.run(
            [sys.executable, "-c", import_code],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    reporting_path = runners_root / "frontres_evaluation_reporting.py"
    assert reporting_path.is_file()
    reporting_defs = top_level_defs("frontres_evaluation_reporting.py")
    assert {"write_frontres_atomic_json", "write_frontres_json_csv_rows"} <= reporting_defs
    for filename in ("frontres_policy_quality_eval.py", "frontres_dr_sweep_eval.py"):
        source = (runners_root / filename).read_text(encoding="utf-8")
        assert ".write_text(" not in source
        assert "with open(" not in source
    dr_tree = ast.parse((runners_root / "frontres_dr_sweep_eval.py").read_text(encoding="utf-8"))
    dr_owner = next(
        node for node in dr_tree.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate_frontres_dr_sweep"
    )
    assert not any(isinstance(node, ast.FunctionDef) for node in ast.walk(dr_owner) if node is not dr_owner)
    assert int(dr_owner.end_lineno or 0) - int(dr_owner.lineno) < 120

    formal_source = (runners_root / "frontres_segment_formal_transaction.py").read_text(encoding="utf-8")
    runtime_source = (runners_root / "frontres_segment_runtime_types.py").read_text(encoding="utf-8")
    assert "from rsl_rl.runners.frontres_segment_live_training import" not in formal_source
    assert "from rsl_rl.runners.frontres_segment_training_telemetry import" in formal_source
    assert "from rsl_rl.runners.frontres_segment_live_sampler import FrontRESFormalTransactionPlan" not in runtime_source
    assert "from rsl_rl.runners.frontres_segment_transaction import FrontRESFormalTransactionPlan" in runtime_source


def test_phase_d_domain_and_algorithm_owners() -> None:
    for relative_path, _review_prompt_lines in PHASE_D_DOMAIN_OWNERS.items():
        path = RSL_ROOT / relative_path
        assert path.is_file(), f"missing Phase D owner: {relative_path}"
        source = path.read_text(encoding="utf-8")
        ast.parse(source)

    for relative_path in (
        "frontres/frontres_segment_storage.py",
        "frontres/frontres_segment_diagnostics.py",
    ):
        tree = ast.parse((RSL_ROOT / relative_path).read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in tree.body
        ), f"compatibility facade regained behavior: {relative_path}"

    production_dynamic_imports: list[str] = []
    for path in RSL_ROOT.rglob("*.py"):
        if "tests" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        if "spec_from_file_location" in source or "importlib.util" in source:
            production_dynamic_imports.append(str(path.relative_to(RSL_ROOT)))
    assert production_dynamic_imports == [], production_dynamic_imports

    formal_source = (RSL_ROOT / "runners" / "frontres_segment_formal_transaction.py").read_text(encoding="utf-8")
    assert "from rsl_rl.runners.frontres_segment_live_update_loop import" not in formal_source
    assert "run_frontres_formal_transaction" in formal_source

    formal_quality = ast.parse(
        (RSL_ROOT / "runners" / "frontres_policy_quality_formal_owners.py").read_text(encoding="utf-8")
    )
    formal_quality_imports = {
        node.module
        for node in ast.walk(formal_quality)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "rsl_rl.runners.frontres_policy_quality_eval" not in formal_quality_imports
    assert "rsl_rl.runners.frontres_policy_quality_interfaces" in formal_quality_imports
    assert "rsl_rl.runners.frontres_policy_quality_state" in formal_quality_imports


def main() -> None:
    test_lazy_public_facade()
    test_evaluation_reporting_refuses_partial_json_csv_commit()
    test_deployment_carrier_exposes_only_public_sequence_contract()
    interfaces, engine = _owners()
    test_schema_and_identity(interfaces)
    test_engine_exact_one_and_collection_barrier(interfaces, engine)
    test_training_recollection_and_cleanup(interfaces, engine)
    test_explicit_mosaic_backend_bindings(interfaces, engine)
    test_engine_resolution_and_static_route(interfaces, engine)
    test_live_probe_facade_has_deep_acyclic_public_owners()
    test_active_generation_and_checkpoint_security_are_isolated()
    test_explicit_run_mode_rejects_boolean_mixing()
    test_adjacent_hotspots_use_public_cohesive_owners()
    test_phase_d_domain_and_algorithm_owners()
    print(
        "[T-interface/T-schema/T-exact-one/T-recollect/T-cleanup] "
        "FrontRES formal Stage-3 route is isolated behind typed fail-closed ports",
        flush=True,
    )


if __name__ == "__main__":
    main()
