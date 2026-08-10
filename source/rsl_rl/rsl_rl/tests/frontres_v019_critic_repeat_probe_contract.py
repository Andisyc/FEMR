from __future__ import annotations

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from frontres_contract_imports import install_frontres_contract_packages


install_frontres_contract_packages()

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = ROOT / "note/testing/manifests/frontres_v019_critic_repeat_k8_m4_v1.json"

from rsl_rl.frontres.frontres_policy_quality_manifest import (  # noqa: E402
    FrontRESV018PolicyQualityManifest,
)
from rsl_rl.runners import frontres_policy_quality_eval as quality  # noqa: E402
from rsl_rl.runners import frontres_segment_formal_transaction as formal  # noqa: E402
from rsl_rl.runners.frontres_segment_runtime_types import (  # noqa: E402
    FrontRESSegmentLiveObservations,
)


def _transaction(repeat_index: int, target_a: float, target_b: float, action_offset: float) -> dict:
    actions = [
        [action_offset + 0.01 * row + 0.001 * axis for axis in range(6)]
        for row in range(8)
    ]
    return {
        "repeat_index": repeat_index,
        "item_ids": ["fixed-a", "fixed-b"],
        "item_comparison_signatures": ["a" * 64, "b" * 64],
        "x_t_identities": ["x-t-a"] * 4 + ["x-t-b"] * 4,
        "actor_input_rows": [[0.0] * 928, [1.0] * 928],
        "critic_input_rows": [[0.0] * 449, [1.0] * 449],
        "critic_calibration": [
            {
                "source_index": 0,
                "segment_id": 10,
                "scenario_id": "scenario-a",
                "noisy_segment_hash": "hash-a",
                "attempt_count": 4,
                "policy_value": 0.25,
                "raw_target_mean": 10.0 * target_a,
                "target_mean": target_a,
                "value_error": 0.25 - target_a,
            },
            {
                "source_index": 1,
                "segment_id": 20,
                "scenario_id": "scenario-b",
                "noisy_segment_hash": "hash-b",
                "attempt_count": 4,
                "policy_value": -0.1,
                "raw_target_mean": 10.0 * target_b,
                "target_mean": target_b,
                "value_error": -0.1 - target_b,
            },
        ],
        "report": {"policy_actions": actions},
    }


def test_repeat_diagnostics_preserve_fixed_segment_identity_and_hand_statistics() -> None:
    transactions = [
        _transaction(0, 0.0, -1.0, 0.0),
        _transaction(1, 1.0, 0.0, 1.0),
        _transaction(2, 2.0, 1.0, 2.0),
    ]
    result = quality.build_frontres_v019_critic_repeat_diagnostics(transactions, repeat_count=3)

    assert result["schema_version"] == "frontres-v019-critic-repeat-diagnostics-v2"
    assert result["policy_input_contract"] == "first-repeat-frozen-actor-critic-v1"
    assert (result["repeat_count"], result["fixed_segment_count"]) == (3, 2)
    first, second = result["segments"]
    assert first["item_id"] == "fixed-a" and second["item_id"] == "fixed-b"
    assert first["repeat_target_means"] == [0.0, 1.0, 2.0]
    assert first["target_mean"] == 1.0
    assert math.isclose(first["target_std"], math.sqrt(2.0 / 3.0), rel_tol=0.0, abs_tol=1e-12)
    assert (first["target_min"], first["target_max"]) == (0.0, 2.0)
    assert first["critic_policy_value"] == 0.25
    assert first["critic_error_to_repeat_mean"] == -0.75
    assert first["actor_input_max_abs_diff"] == 0.0
    assert first["critic_input_max_abs_diff"] == 0.0
    assert len(set(first["action_fingerprints"])) == 3


def test_repeat_diagnostics_reject_any_used_policy_input_drift() -> None:
    base = [_transaction(0, 0.0, -1.0, 0.0), _transaction(1, 1.0, 0.0, 1.0)]
    for field, label in (("actor_input_rows", "Actor"), ("critic_input_rows", "Critic")):
        drifted = copy.deepcopy(base)
        drifted[1][field][0][17] = 1.0e-7
        try:
            quality.build_frontres_v019_critic_repeat_diagnostics(drifted, repeat_count=2)
        except RuntimeError as exc:
            assert f"{label} input drift" in str(exc), str(exc)
        else:
            raise AssertionError(f"repeat diagnostics must reject any used {label}-input drift")


def _live_observations(*, actor_offset: float, critic_offset: float) -> FrontRESSegmentLiveObservations:
    return FrontRESSegmentLiveObservations(
        obs=torch.full((16, 928), actor_offset, dtype=torch.float32),
        privileged_obs=torch.full((16, 449), critic_offset, dtype=torch.float32),
        teacher_obs=torch.full((16, 289), 3.0, dtype=torch.float32),
        ref_vel_estimator_obs=None,
    )


def test_repeat_policy_input_owner_reuses_first_input_despite_live_history_drift() -> None:
    first_live = _live_observations(actor_offset=0.0, critic_offset=0.0)
    first_used, first_actor_drift, first_critic_drift = formal._resolve_frontres_repeat_policy_observations(
        first_live,
        frozen=None,
        route="policy_quality",
        expected_rows=16,
        device=torch.device("cpu"),
    )
    changed_live = _live_observations(actor_offset=0.25, critic_offset=0.568127841)
    repeated_used, actor_drift, critic_drift = formal._resolve_frontres_repeat_policy_observations(
        changed_live,
        frozen=first_used,
        route="policy_quality",
        expected_rows=16,
        device=torch.device("cpu"),
    )

    assert torch.equal(repeated_used.obs, first_used.obs)
    assert torch.equal(repeated_used.privileged_obs, first_used.privileged_obs)
    assert not torch.equal(repeated_used.privileged_obs, changed_live.privileged_obs)
    assert (first_actor_drift, first_critic_drift) == (0.0, 0.0)
    assert math.isclose(actor_drift, 0.25, rel_tol=0.0, abs_tol=1e-7)
    assert math.isclose(critic_drift, 0.568127841, rel_tol=0.0, abs_tol=1e-7)

    try:
        formal._resolve_frontres_repeat_policy_observations(
            changed_live,
            frozen=first_used,
            route="train",
            expected_rows=16,
            device=torch.device("cpu"),
        )
    except RuntimeError as exc:
        assert "policy-quality" in str(exc)
    else:
        raise AssertionError("training collection must reject frozen repeat inputs")


def test_recovery_aware_collector_consumes_frozen_repeat_inputs_at_public_boundary() -> None:
    frozen = _live_observations(actor_offset=0.0, critic_offset=0.0)
    changed_live = _live_observations(actor_offset=0.25, critic_offset=0.568127841)
    captured: list[FrontRESSegmentLiveObservations] = []

    class _Baseline:
        def __init__(self, **values):
            self.values = values

        def validate(self) -> None:
            return None

    class _Evidence:
        def __init__(self, *, baselines, attempts, active_m):
            self.baselines = baselines
            self.attempts = attempts
            self.active_m = active_m

        def validate(self) -> None:
            return None

        def to_gain_input(self):
            return self

    plan = SimpleNamespace(
        validate=lambda: None,
        batch_size=8,
        selected_segment_count=2,
        active_m=4,
        transaction_id="tx-repeat",
        policy_snapshot_id="pi-frozen",
        source_index=torch.tensor((0, 0, 0, 0, 1, 1, 1, 1), dtype=torch.long),
        segment_ids=torch.tensor((10, 10, 10, 10, 20, 20, 20, 20), dtype=torch.long),
        trial_index=torch.tensor((0, 1, 2, 3, 0, 1, 2, 3), dtype=torch.long),
        horizon_k=torch.full((8,), 8, dtype=torch.long),
        scenario_ids=("scenario-a",) * 4 + ("scenario-b",) * 4,
        noisy_segment_hashes=("hash-a",) * 4 + ("hash-b",) * 4,
        x_t_identities=("x-a",) * 4 + ("x-b",) * 4,
    )
    prepared = SimpleNamespace(batch=object(), sample=object(), plan=plan)
    runner = SimpleNamespace(device=torch.device("cpu"))
    try:
        formal.collect_frontres_recovery_aware_evaluation(
            runner,
            prepared,
            route="train",
            label="forbidden repeat input",
            beta=0.02,
            policy_observations=frozen,
        )
    except RuntimeError as exc:
        assert "policy-quality" in str(exc)
    else:
        raise AssertionError("collector must reject frozen repeat inputs before training-side effects")
    replacements = {
        "prepare_frontres_raw_contact_views": lambda _runner: None,
        "bind_frontres_collection_context": lambda _runner, **_kwargs: None,
        "resolve_frontres_mode_state": lambda _runner, _policy: SimpleNamespace(is_frontres=True),
        "configure_frontres_pair_layout": lambda _runner, **_kwargs: SimpleNamespace(
            n_train=8, n_base=8, n_candidate=0, n_clean=0
        ),
        "_reset_frontres_v017_phase": lambda _runner, **_kwargs: None,
        "collect_frontres_v017_no_actor_baseline": lambda _runner, **_kwargs: (
            object(),
            torch.ones((1, 2), dtype=torch.bool),
        ),
        "select_frontres_v017_trajectory_rows": lambda trajectory, _rows: trajectory,
        "_read_live_observations": lambda _runner: changed_live,
        "FrontRESSegmentBaselineEvidence": _Baseline,
        "FrontRESSealedRecoveryAwareGainBatch": _Evidence,
        "compute_recovery_aware_gain": lambda _input, *, config: SimpleNamespace(config=config),
        "build_frontres_v017_local_evaluation_report": lambda evidence, gain: SimpleNamespace(
            evidence=evidence, gain=gain
        ),
        "frontres_observation_trace": lambda _runner: {"critic_observation_dim": 449},
    }

    def collect_attempts(_runner, observations, **_kwargs):
        captured.append(observations)
        return (object(),) * 8

    replacements["collect_frontres_v017_repair_attempts"] = collect_attempts
    originals = {name: getattr(formal, name) for name in replacements}
    for name, value in replacements.items():
        setattr(formal, name, value)
    try:
        collection = formal.collect_frontres_recovery_aware_evaluation(
            runner,
            prepared,
            route="policy_quality",
            label="EVAL-v004 repeat public-boundary fixture",
            beta=0.02,
            policy_observations=frozen,
        )
    finally:
        for name, value in originals.items():
            setattr(formal, name, value)

    assert len(captured) == 1
    assert torch.equal(captured[0].obs, frozen.obs)
    assert torch.equal(captured[0].privileged_obs, frozen.privileged_obs)
    assert torch.equal(collection.policy_observations.privileged_obs, frozen.privileged_obs)
    assert collection.observation_trace["repeat_policy_input_source"] == "first-repeat-frozen"
    assert math.isclose(
        collection.observation_trace["repeat_live_critic_input_max_abs_diff"],
        0.568127841,
        rel_tol=0.0,
        abs_tol=1e-7,
    )


def test_repeat_diagnostics_fail_closed_on_identity_or_action_collapse() -> None:
    base = [_transaction(0, 0.0, -1.0, 0.0), _transaction(1, 1.0, 0.0, 1.0)]

    drifted = copy.deepcopy(base)
    drifted[1]["critic_calibration"][0]["noisy_segment_hash"] = "changed"
    try:
        quality.build_frontres_v019_critic_repeat_diagnostics(drifted, repeat_count=2)
    except RuntimeError as exc:
        assert "fixed Segment identity" in str(exc), str(exc)
    else:
        raise AssertionError("repeat diagnostics must reject scenario/hash drift")

    collapsed = copy.deepcopy(base)
    collapsed[1]["report"]["policy_actions"] = copy.deepcopy(collapsed[0]["report"]["policy_actions"])
    try:
        quality.build_frontres_v019_critic_repeat_diagnostics(collapsed, repeat_count=2)
    except RuntimeError as exc:
        assert "distinct Repair actions" in str(exc), str(exc)
    else:
        raise AssertionError("repeat diagnostics must reject identical Repair-action groups")


def test_k8_repeat_manifest_is_strict_and_minimal() -> None:
    manifest = FrontRESV018PolicyQualityManifest.from_json(MANIFEST.read_text(encoding="utf-8"))
    assert manifest.horizon_k == 8
    assert manifest.attempts_per_segment == 4
    assert manifest.segments_per_transaction == 2
    assert len(manifest.items) == 2
    assert all(item.effective_horizon_k == 8 for item in manifest.items)


if __name__ == "__main__":
    test_repeat_diagnostics_preserve_fixed_segment_identity_and_hand_statistics()
    test_repeat_diagnostics_reject_any_used_policy_input_drift()
    test_repeat_policy_input_owner_reuses_first_input_despite_live_history_drift()
    test_recovery_aware_collector_consumes_frozen_repeat_inputs_at_public_boundary()
    test_repeat_diagnostics_fail_closed_on_identity_or_action_collapse()
    test_k8_repeat_manifest_is_strict_and_minimal()
    print("frontres_v019_critic_repeat_probe_contract: ok")
