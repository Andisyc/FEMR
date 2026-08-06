from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from frontres_contract_imports import install_frontres_contract_packages

install_frontres_contract_packages()

from rsl_rl.frontres.frontres_policy_quality_manifest import FrontRESV017PolicyQualityManifest
import rsl_rl.runners.frontres_checkpointing as frontres_checkpointing
import rsl_rl.runners.frontres_policy_quality_eval as quality
import rsl_rl.runners.frontres_segment_formal_transaction as formal
import rsl_rl.runners.frontres_segment_live_sampler as sampler
import rsl_rl.runners.frontres_segment_runtime_types as runtime_types
from rsl_rl.algorithms.frontres_unified import FrontRESUnified


ROOT = Path(__file__).resolve().parents[4]
MANIFEST = ROOT / "note" / "testing" / "manifests" / "frontres_v017_policy_quality_k16_v1.json"


def test_policy_quality_algorithm_constructs_readonly_stage3_identity() -> None:
    schedule = (
        (8, 2, 200, 500, 1300, "lower-k8", 0.5, "linear-joint-v1", 1300, 2.381),
        (16, 3, 300, 300, 900, "lower-k16", 0.6, "linear-joint-v1", 900, 2.381),
        (32, 4, 400, 300, 625, "lower-k32", 0.7, "linear-joint-v1", 625, 2.381),
    )
    algorithm = FrontRESUnified(
        torch.nn.Linear(2, 1),
        frontres_training_objective="segment_replay_hrl",
        frontres_policy_quality_eval_only=True,
        frontres_segment_replay_enabled=False,
        frontres_segment_live_runner_enabled=False,
        frontres_segment_live_train_enabled=False,
        frontres_formal_transaction_enabled=True,
        frontres_segment_k_curriculum=schedule,
        frontres_segment_advantage_normalization="grouped_scale_only",
        frontres_future_offsets=(1, 2),
        lambda_supervised=0.0,
        lambda_supervised_min=0.0,
    )
    assert algorithm.frontres_policy_quality_eval_only is True
    assert algorithm.frontres_segment_replay_enabled is False
    assert algorithm.frontres_segment_live_runner_enabled is False
    assert algorithm.frontres_segment_live_train_enabled is False
    assert algorithm.optimizer.frontres_step_count == 0

    try:
        FrontRESUnified(
            torch.nn.Linear(2, 1),
            frontres_training_objective="segment_replay_hrl",
            frontres_policy_quality_eval_only=True,
            frontres_segment_replay_enabled=True,
            frontres_segment_live_runner_enabled=True,
            frontres_formal_transaction_enabled=True,
            frontres_segment_k_curriculum=schedule,
            frontres_segment_advantage_normalization="grouped_scale_only",
            frontres_future_offsets=(1, 2),
            lambda_supervised=0.0,
            lambda_supervised_min=0.0,
        )
    except ValueError as exc:
        assert "cannot enable Segment Replay/live training modes" in str(exc)
    else:
        raise AssertionError("read-only quality algorithm must reject mixed training modes")


def test_policy_quality_reset_support_is_readonly_and_sampler_free() -> None:
    runner = SimpleNamespace(_frontres_segment_sampler=None)
    calls: list[str] = []
    original_dataset = sampler._ensure_stage1_cache_dataset
    original_hook = sampler._ensure_stage1_index_reset_hook

    def install_dataset(active_runner):
        active_runner._frontres_segment_dataset = "stage1-index"
        calls.append("dataset")

    def install_hook(active_runner):
        assert active_runner._frontres_segment_dataset == "stage1-index"
        assert active_runner._frontres_segment_sampler is None
        calls.append("reset-hook")

    sampler._ensure_stage1_cache_dataset = install_dataset
    sampler._ensure_stage1_index_reset_hook = install_hook
    try:
        sampler.ensure_frontres_policy_quality_reset_support(runner)
        assert calls == ["dataset", "reset-hook"]
        assert runner._frontres_segment_sampler is None

        sampler._ensure_stage1_cache_dataset = lambda active_runner: setattr(
            active_runner, "_frontres_segment_sampler", object()
        )
        sampler._ensure_stage1_index_reset_hook = lambda _active_runner: None
        try:
            sampler.ensure_frontres_policy_quality_reset_support(runner)
        except RuntimeError as exc:
            assert "must not create or replace the Segment sampler" in str(exc)
        else:
            raise AssertionError("policy-quality reset support must fail if a sampler is created")
    finally:
        sampler._ensure_stage1_cache_dataset = original_dataset
        sampler._ensure_stage1_index_reset_hook = original_hook


def test_policy_quality_collection_lifecycle_preserves_receipt_and_cleans_exceptions() -> None:
    """Read-only evaluation may occupy execution state but never transaction persistence."""

    receipt = {"transaction_id": "committed-before-eval", "optimizer_step_delta": 1}
    runner = SimpleNamespace(
        _frontres_checkpoint_transaction_state={"state": "committed", "receipt": receipt}
    )
    original_close = formal.close_frontres_formal_training_request
    formal.close_frontres_formal_training_request = lambda _runner: None
    try:
        try:
            with formal.frontres_v017_readonly_collection_scope(runner):
                aggregate = runtime_types.frontres_stage3_transaction_aggregate(runner)
                assert aggregate.execution_phase == "evaluating"
                assert aggregate.persistence_phase == "committed"
                try:
                    runtime_types.bind_frontres_collection_context(
                        runner,
                        route="training",
                        sample=object(),
                        batch=object(),
                    )
                except ValueError as route_error:
                    assert "explicit route" in str(route_error)
                else:
                    raise AssertionError("read-only evaluation must reject the training route")
                runtime_types.bind_frontres_collection_context(
                    runner,
                    route="policy_quality",
                    sample=object(),
                    batch=object(),
                )
                raise RuntimeError("deliberate evaluator failure")
        except RuntimeError as exc:
            assert "deliberate evaluator failure" in str(exc)
        else:
            raise AssertionError("the deliberate evaluator failure must escape the read-only scope")
    finally:
        formal.close_frontres_formal_training_request = original_close

    aggregate = runtime_types.frontres_stage3_transaction_aggregate(runner)
    assert aggregate.execution_phase == "idle"
    assert aggregate.as_dict() == {"state": "committed", "receipt": receipt}
    assert aggregate.collection_sample is None and aggregate.collection_batch is None

    aggregate._collection_route = "stale"
    try:
        aggregate.begin_readonly_collection()
    except RuntimeError as exc:
        assert "stale collection context" in str(exc)
    else:
        raise AssertionError("read-only evaluation must reject stale collection context")
    assert aggregate.collection_route == "stale"
    aggregate.clear_collection_context()


@dataclass(frozen=True)
class _Report:
    gain_total: tuple[float, ...]
    policy_actions: tuple[tuple[float, ...], ...]
    clean_execution_count: tuple[int, ...]
    noisy_execution_count: tuple[int, ...]


def test_active_v017_evaluator_serializes_four_readonly_k16_m3_transactions(tmp_path: Path) -> None:
    manifest_bytes = MANIFEST.read_bytes()
    manifest = FrontRESV017PolicyQualityManifest.from_json(manifest_bytes.decode("utf-8"))
    checkpoint = SimpleNamespace(
        format="frontres-v017-checkpoint-v9",
        file_sha256="c" * 64,
    )
    request = quality.FrontRESV017PolicyQualityEvalRequest(
        manifest_path=str(MANIFEST),
        hsl_checkpoint_path=str(tmp_path / "hsl.pt"),
        policy_checkpoint_path=str(tmp_path / "model_3500.pt"),
        result_path=str(tmp_path / "quality.json"),
        manifest=manifest,
        manifest_file_sha256="m" * 64,
        hsl_checkpoint=SimpleNamespace(file_sha256="h" * 64),
        policy_checkpoint=checkpoint,
    )
    policy = SimpleNamespace(
        residual_actor=torch.nn.Linear(2, 6),
        critic=torch.nn.Linear(2, 1),
    )
    optimizer = torch.optim.Adam(
        tuple(policy.residual_actor.parameters()) + tuple(policy.critic.parameters()), lr=1e-3
    )
    runner = SimpleNamespace(
        alg=SimpleNamespace(policy=policy, optimizer=optimizer, frontres_gain_beta=0.02),
        current_learning_iteration=3500,
        _frontres_last_committed_transaction_receipt={"transaction_id": "fixed", "optimizer_step_delta": 1},
    )
    calls: list[tuple[str, ...]] = []
    closes: list[str] = []
    reset_support_calls: list[str] = []

    @contextmanager
    def route_actor(_runner, _path, *, route, expected_file_sha256):
        assert route == "policy" and expected_file_sha256 == "c" * 64
        yield

    def prepare(_runner, items, *, attempts_per_segment):
        assert getattr(_runner, "_frontres_segment_dataset", None) == "quality-readonly-dataset"
        assert getattr(_runner, "_frontres_segment_sampler", None) is None
        calls.append(tuple(item.item_id for item in items))
        assert attempts_per_segment == 3
        index = len(calls)
        return SimpleNamespace(
            sample=object(),
            batch=object(),
            plan=SimpleNamespace(
                transaction_id=f"tx-{index}",
                policy_snapshot_id=f"pi-{index}",
                selected_segment_count=2,
                batch_size=6,
            )
        )

    def collect(_runner, prepared, *, route, label, beta):
        assert route == "policy_quality" and "EVAL-v004" in label and beta == 0.02
        aggregate = runtime_types.frontres_stage3_transaction_aggregate(_runner)
        assert aggregate.execution_phase == "evaluating"
        assert aggregate.persistence_phase == "idle"
        runtime_types.bind_frontres_collection_context(
            _runner,
            route=route,
            sample=prepared.sample,
            batch=prepared.batch,
        )
        return SimpleNamespace(
            observation_trace={"combined_observation_dim": 928, "femr_visible_dim": 158, "gmt_suffix_dim": 770},
            report=_Report(
                gain_total=(0.1,) * 6,
                policy_actions=((0.01,) * 6,) * 6,
                clean_execution_count=(1, 1),
                noisy_execution_count=(1, 1),
            ),
        )

    original_route = frontres_checkpointing.frontres_quality_route_actor
    original_reset_support = sampler.ensure_frontres_policy_quality_reset_support
    original_prepare = sampler.prepare_frontres_v017_policy_quality_batch
    original_collect = formal.collect_frontres_v017_recovery_aware_evaluation
    original_close = formal.close_frontres_formal_training_request
    frontres_checkpointing.frontres_quality_route_actor = route_actor

    def ensure_reset_support(_runner):
        assert getattr(_runner, "_frontres_segment_sampler", None) is None
        _runner._frontres_segment_dataset = "quality-readonly-dataset"
        reset_support_calls.append("installed")

    sampler.ensure_frontres_policy_quality_reset_support = ensure_reset_support
    sampler.prepare_frontres_v017_policy_quality_batch = prepare
    formal.collect_frontres_v017_recovery_aware_evaluation = collect
    formal.close_frontres_formal_training_request = lambda _runner: closes.append("closed")
    try:
        payload = quality.run_frontres_v017_policy_quality_heldout_eval(runner, request=request)
        del runner.alg.frontres_gain_beta
        try:
            quality.run_frontres_v017_policy_quality_heldout_eval(runner, request=request)
        except RuntimeError as exc:
            assert "repair-cost beta" in str(exc)
        else:
            raise AssertionError("EVAL-v004 must reject a missing Gain beta instead of filling a default")
    finally:
        frontres_checkpointing.frontres_quality_route_actor = original_route
        sampler.ensure_frontres_policy_quality_reset_support = original_reset_support
        sampler.prepare_frontres_v017_policy_quality_batch = original_prepare
        formal.collect_frontres_v017_recovery_aware_evaluation = original_collect
        formal.close_frontres_formal_training_request = original_close

    assert len(reset_support_calls) == 2
    assert len(calls) == 4 and len(closes) == 4
    aggregate = runtime_types.frontres_stage3_transaction_aggregate(runner)
    assert aggregate.execution_phase == "idle"
    assert aggregate.persistence_phase == "idle"
    assert aggregate.collection_sample is None and aggregate.collection_batch is None
    assert payload["schema_version"] == "frontres-v017-policy-quality-report-v1"
    assert payload["checkpoint_format"] == "frontres-v017-checkpoint-v9"
    assert (payload["horizon_k"], payload["attempts_per_segment"]) == (16, 3)
    assert all(row["policy_row_count"] == 6 and row["role_row_count"] == 12 for row in payload["transactions"])
    stored = json.loads(Path(request.result_path).read_text(encoding="utf-8"))
    assert stored == json.loads(json.dumps(payload))


def test_training_state_guard_names_the_mutated_owner() -> None:
    runner = SimpleNamespace(current_learning_iteration=3500)
    expected = quality._v015_quality_training_state_field_hashes(runner)
    runner.current_learning_iteration = 3501
    try:
        quality._assert_v015_quality_training_state_unchanged(
            runner,
            expected,
            label="deliberate mutation",
        )
    except RuntimeError as exc:
        message = str(exc)
        assert "deliberate mutation" in message
        assert "differing_fields=('iteration',)" in message
    else:
        raise AssertionError("training-state guard must identify the mutated owner")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_policy_quality_algorithm_constructs_readonly_stage3_identity()
        test_policy_quality_reset_support_is_readonly_and_sampler_free()
        test_policy_quality_collection_lifecycle_preserves_receipt_and_cleans_exceptions()
        test_active_v017_evaluator_serializes_four_readonly_k16_m3_transactions(Path(tmp))
        test_training_state_guard_names_the_mutated_owner()
    print("frontres_v017_policy_quality_eval_contract: ok")
