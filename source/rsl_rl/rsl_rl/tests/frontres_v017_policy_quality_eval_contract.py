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
        _frontres_checkpoint_transaction_state={"state": "committed", "receipt": "fixed"},
        _frontres_last_committed_transaction_receipt="fixed",
    )
    calls: list[tuple[str, ...]] = []
    closes: list[str] = []

    @contextmanager
    def route_actor(_runner, _path, *, route, expected_file_sha256):
        assert route == "policy" and expected_file_sha256 == "c" * 64
        yield

    def prepare(_runner, items, *, attempts_per_segment):
        calls.append(tuple(item.item_id for item in items))
        assert attempts_per_segment == 3
        index = len(calls)
        return SimpleNamespace(
            plan=SimpleNamespace(
                transaction_id=f"tx-{index}",
                policy_snapshot_id=f"pi-{index}",
                selected_segment_count=2,
                batch_size=6,
            )
        )

    def collect(_runner, prepared, *, route, label, beta):
        assert route == "policy_quality" and "EVAL-v004" in label and beta == 0.02
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
    original_prepare = sampler.prepare_frontres_v017_policy_quality_batch
    original_collect = formal.collect_frontres_v017_recovery_aware_evaluation
    original_close = formal.close_frontres_formal_training_request
    frontres_checkpointing.frontres_quality_route_actor = route_actor
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
        sampler.prepare_frontres_v017_policy_quality_batch = original_prepare
        formal.collect_frontres_v017_recovery_aware_evaluation = original_collect
        formal.close_frontres_formal_training_request = original_close

    assert len(calls) == 5 and len(closes) == 5
    assert payload["schema_version"] == "frontres-v017-policy-quality-report-v1"
    assert payload["checkpoint_format"] == "frontres-v017-checkpoint-v9"
    assert (payload["horizon_k"], payload["attempts_per_segment"]) == (16, 3)
    assert all(row["policy_row_count"] == 6 and row["role_row_count"] == 12 for row in payload["transactions"])
    stored = json.loads(Path(request.result_path).read_text(encoding="utf-8"))
    assert stored == json.loads(json.dumps(payload))


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_policy_quality_algorithm_constructs_readonly_stage3_identity()
        test_active_v017_evaluator_serializes_four_readonly_k16_m3_transactions(Path(tmp))
    print("frontres_v017_policy_quality_eval_contract: ok")
