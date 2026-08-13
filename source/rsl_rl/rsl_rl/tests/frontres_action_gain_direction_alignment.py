#!/usr/bin/env python3
"""S1 alignment proof for the offline action-to-Gain direction analyzer."""

from __future__ import annotations

import copy
import math

from frontres_action_gain_direction_analysis import (
    COMPONENTS,
    SCHEMA,
    DirectionAnalysisInputError,
    analyze_payload,
    analyze_scenario,
    centered_cross_moment,
    direction_cosine,
    policy_score_direction,
    vector_norm,
)


def _walsh(index: int, mask: int) -> float:
    return -1.0 if (index & mask).bit_count() % 2 else 1.0


def _actions() -> list[list[float]]:
    return [[0.01 * _walsh(index, mask) for mask in range(1, 7)] for index in range(32)]


def _row(repair_index: int, action: list[float], intent: float, physics: float = 0.0, cost: float = 0.0) -> dict:
    total = intent + physics + cost
    physics_remaining_noisy = 1.0
    physics_remaining_repaired = math.sqrt(physics_remaining_noisy**2 - 2.0 * physics)
    physics_gain = physics_remaining_noisy - physics_remaining_repaired
    recovery_pressure = 0.5 * (physics_remaining_noisy + physics_remaining_repaired)
    weighted_physics = recovery_pressure * physics_gain
    total = intent + weighted_physics + cost
    return {
        "repair_index": repair_index,
        "visit_index": repair_index // 4,
        "attempt_index": repair_index % 4,
        "action_seed": 1000 + repair_index // 4,
        "runtime_seed": 777,
        "checkpoint_file_sha256": "a" * 64,
        "manifest_file_sha256": "b" * 64,
        "action": action,
        "components": {
            "utility": math.copysign(math.log1p(abs(total)), total),
            "raw_return": total,
            "gain_total": total,
            "intent_gain": intent,
            "physics_remaining_noisy": physics_remaining_noisy,
            "physics_remaining_repaired": physics_remaining_repaired,
            "physics_gain": physics_gain,
            "recovery_pressure": recovery_pressure,
            "weighted_physics_gain": weighted_physics,
            "physics_channel_noisy": [physics_remaining_noisy] * 4,
            "physics_channel_repaired": [physics_remaining_repaired] * 4,
            "repair_penalty": -cost,
            "negative_repair_cost": cost,
        },
    }


def _scenario(
    scenario_id: str,
    rows: list[dict],
    *,
    actor_mean: list[float] | None = None,
    actor_sigma: list[float] | None = None,
) -> dict:
    return {
        "scenario_id": scenario_id,
        "active_m": 4,
        "checkpoint_file_sha256": "a" * 64,
        "manifest_file_sha256": "b" * 64,
        "actor_mean": [0.0] * 6 if actor_mean is None else actor_mean,
        "actor_sigma": [1.0] * 6 if actor_sigma is None else actor_sigma,
        "visits": [
            {
                "visit_index": visit,
                "action_seed": 1000 + visit,
                "runtime_seed": 777,
                "actor_input_max_abs_diff": 0.0,
                "critic_input_max_abs_diff": 0.0,
                "live_actor_input_max_abs_diff": 0.0 if visit == 0 else 3.8110761642456055,
                "live_critic_input_max_abs_diff": 0.0 if visit == 0 else 4.439251899719238,
            }
            for visit in range(8)
        ],
        "rows": rows,
    }


def _linear_scenario(
    scenario_id: str,
    *,
    sign: float = 1.0,
    actor_sigma: list[float] | None = None,
) -> dict:
    actions = _actions()
    coefficients = (3.0, -2.0, 1.5, 0.75, -0.5, 0.25)
    rows = []
    for index, action in enumerate(actions):
        intent = sign * sum(weight * value for weight, value in zip(coefficients, action, strict=True))
        rows.append(_row(index, action, intent))
    return _scenario(scenario_id, rows, actor_sigma=actor_sigma)


def _noise_scenario() -> dict:
    actions = _actions()
    return _scenario(
        "orthogonal-no-signal",
        [_row(index, action, 0.1 * _walsh(index, 7)) for index, action in enumerate(actions)],
    )


def _single_axis_scenario() -> dict:
    rows = []
    for index in range(32):
        x_value = -1.0 if index % 2 == 0 else 1.0
        action = [x_value, 0.0, 0.0, 0.0, 0.0, 0.0]
        rows.append(_row(index, action, 2.0 * x_value))
    return _scenario("single-axis-hand-oracle", rows)


def _cancellation_scenario() -> dict:
    actions = _actions()
    rows = []
    for index, action in enumerate(actions):
        intent = 3.0 * action[0] - 2.0 * action[1]
        rows.append(_row(index, action, intent, physics=-intent))
    return _scenario("component-cancellation", rows)


def _inclusive_reference_bias_scenario() -> dict:
    rows = []
    for index in range(32):
        x_value = -1.0 if index % 2 == 0 else 1.0
        action = [x_value, 0.0, 0.0, 0.0, 0.0, 0.0]
        gain = (10.0 if index < 4 else -0.1) * x_value
        rows.append(_row(index, action, gain))
    return _scenario("inclusive-reference-bias", rows)


def _progressive_scenario() -> dict:
    block_directions = (
        (2.0, 4.0),
        (2.0, -4.0),
        (2.0, 2.0),
        (2.0, -2.0),
        (2.0, 0.0),
        (2.0, 0.0),
        (2.0, 0.0),
        (2.0, 0.0),
    )
    rows = []
    for block_index, (gain_x, gain_y) in enumerate(block_directions):
        block_rows = (
            ([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0], -2.0 * gain_x),
            ([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2.0 * gain_x),
            ([0.0, -1.0, 0.0, 0.0, 0.0, 0.0], -2.0 * gain_y),
            ([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 2.0 * gain_y),
        )
        for within_block, (action, gain) in enumerate(block_rows):
            rows.append(_row(block_index * 4 + within_block, action, gain))
    return _scenario("progressive-estimator", rows)


def _payload(*scenarios: dict) -> dict:
    return {"schema": SCHEMA, "scenarios": list(scenarios)}


def _expect_input_error(callable_object, required_text: str) -> None:
    try:
        callable_object()
    except DirectionAnalysisInputError as exc:
        assert required_text.lower() in str(exc).lower(), str(exc)
        return
    raise AssertionError("malformed direction evidence did not fail closed")


def _assert_vectors_close(observed: list[float], expected: list[float], tolerance: float = 1.0e-12) -> None:
    assert len(observed) == len(expected)
    for observed_value, expected_value in zip(observed, expected, strict=True):
        assert math.isclose(observed_value, expected_value, rel_tol=0.0, abs_tol=tolerance), (
            observed,
            expected,
        )


def test_hand_calculated_centered_cross_moment() -> None:
    """S1/C1+C2/T-value: asymmetric four-row direction has a hand oracle."""

    actions = [
        [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    ]
    values = [-3.0, 3.0, -1.0, 1.0]
    observed = centered_cross_moment(actions, values)
    _assert_vectors_close(observed, [1.5, 0.5, 0.0, 0.0, 0.0, 0.0])


def test_common_gain_translation_cancels_from_both_estimators() -> None:
    """S1/C4/T-meta: a shared scalar baseline shift cannot rotate either direction."""

    actions = _actions()
    values = [3.0 * action[0] - 2.0 * action[1] for action in actions]
    shifted = [value + 1000.0 for value in values]
    actor_mean = [0.3, -0.2, 0.1, -0.4, 0.5, -0.6]
    actor_sigma = [0.5, 1.5, 0.75, 2.0, 1.25, 0.9]
    _assert_vectors_close(
        centered_cross_moment(actions, values),
        centered_cross_moment(actions, shifted),
    )
    _assert_vectors_close(
        policy_score_direction(actions, values, actor_mean, actor_sigma),
        policy_score_direction(actions, shifted, actor_mean, actor_sigma),
    )


def test_linear_relation_is_identifiable_at_m16() -> None:
    """S1/C1/T-value+T-dist: both independent M16 halves recover one direction."""

    report = analyze_scenario(_linear_scenario("linear"), partition_count=24, permutation_count=64, seed=17)
    for component in ("utility", "gain_total", "intent_gain"):
        analysis = report["component_analysis"][component]["policy_score"]
        assert analysis["primary_disjoint_m16"]["cosine"] > 0.999
        assert analysis["held_out_complement"]["m16_vs_m16"]["median"] > 0.0
        assert analysis["m32"]["norm"] > 0.0
    assert set(report["component_analysis"]) == set(COMPONENTS)
    assert report["component_analysis"]["negative_repair_cost"]["policy_score"]["m32"]["norm"] == 0.0


def test_m4_m8_m16_direction_and_cosine_hand_oracle() -> None:
    """S1/C1/T-value: every nested estimate has the independent [2,0,...] answer."""

    report = analyze_scenario(_single_axis_scenario(), partition_count=24, permutation_count=64, seed=18)
    analysis = report["component_analysis"]["gain_total"]["policy_score"]
    _assert_vectors_close(analysis["m32"]["direction"], [2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for key in ("m4_vs_m28", "m8_vs_m24", "m16_vs_m16"):
        estimate = analysis["primary_subset_directions"][key]
        _assert_vectors_close(estimate["subset"]["direction"], [2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        _assert_vectors_close(estimate["held_out"]["direction"], [2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert math.isclose(estimate["cosine"], 1.0, rel_tol=0.0, abs_tol=1.0e-12)


def test_m4_m8_m16_progress_toward_independent_direction_oracle() -> None:
    """S1/C2/T-dist: candidate M grows against one unchanged independent M16."""

    report = analyze_scenario(_progressive_scenario(), partition_count=24, permutation_count=64, seed=18)
    analysis = report["component_analysis"]["gain_total"]["policy_score"]
    progression = analysis["fixed_reference_progression"]
    records = [
        progression[key]
        for key in ("m4_vs_fixed_m16", "m8_vs_fixed_m16", "m16_vs_fixed_m16")
    ]
    assert all(record["fixed_reference_indices"] == list(range(16, 32)) for record in records)
    cosines = [record["cosine_summary"]["median"] for record in records]
    assert cosines[0] < cosines[1] < cosines[2]  # type: ignore[operator]
    assert math.isclose(cosines[2], 1.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert records[0]["first_subset"]["estimator_sample_count"] == 4
    assert records[1]["first_subset"]["estimator_sample_count"] == 8
    assert records[2]["first_subset"]["estimator_sample_count"] == 16
    assert records[2]["cosine_summary"]["attempted_count"] == 1
    assert records[2]["cosine_summary"]["defined_count"] == 1


def test_inclusive_m32_reference_can_create_false_high_alignment() -> None:
    """Sensitivity: a subset-containing M32 reference is optimistic; held-out is adverse."""

    scenario = _inclusive_reference_bias_scenario()
    report = analyze_scenario(scenario, partition_count=24, permutation_count=64, seed=20)
    analysis = report["component_analysis"]["gain_total"]["policy_score"]
    m4 = analysis["primary_subset_directions"]["m4_vs_m28"]["subset"]["direction"]
    m32 = analysis["m32"]["direction"]
    assert direction_cosine(m4, m32) > 0.999
    assert analysis["primary_subset_directions"]["m4_vs_m28"]["cosine"] < -0.999
    assert "inclusive-m32 is not a stability verdict" in analyze_payload(
        _payload(scenario),
        partition_count=24,
        permutation_count=64,
        seed=20,
    )["analysis_identity"]["comparison_reference"]


def test_anisotropic_sigma_separates_policy_score_from_raw_covariance() -> None:
    """S1/C4/T-role: score direction uses frozen sigma^-2; raw covariance is auxiliary."""

    sigma = [1.0, 4.0, 1.0, 4.0, 1.0, 4.0]
    report = analyze_scenario(
        _linear_scenario("anisotropic-sigma", actor_sigma=sigma),
        partition_count=24,
        permutation_count=64,
        seed=21,
    )
    component = report["component_analysis"]["gain_total"]
    assert component["primary_estimator"] == "policy_score"
    raw = component["raw_centered_covariance"]["m32"]["direction"]
    score = component["policy_score"]["m32"]["direction"]
    _assert_vectors_close(score, [value / (axis_sigma * axis_sigma) for value, axis_sigma in zip(raw, sigma)])
    assert direction_cosine(raw, score) < 0.95


def test_pure_noise_has_no_direction() -> None:
    """S1/C2/T-value: an exactly orthogonal scalar has no local action direction."""

    report = analyze_scenario(_noise_scenario(), partition_count=24, permutation_count=64, seed=19)
    gain = report["component_analysis"]["gain_total"]["policy_score"]
    assert gain["m32"]["norm"] < 1.0e-14
    assert gain["primary_disjoint_m16"]["left"]["norm"] < 1.0e-14
    assert gain["primary_disjoint_m16"]["right"]["norm"] < 1.0e-14
    assert gain["primary_disjoint_m16"]["cosine"] is None


def test_component_cancellation_remains_visible() -> None:
    """S1/C2/T-role: stable opposite components may erase the total direction."""

    report = analyze_scenario(_cancellation_scenario(), partition_count=24, permutation_count=64, seed=23)
    analyses = {
        component: report["component_analysis"][component]["policy_score"]
        for component in COMPONENTS
    }
    assert analyses["intent_gain"]["primary_disjoint_m16"]["cosine"] > 0.999
    assert analyses["weighted_physics_gain"]["primary_disjoint_m16"]["cosine"] > 0.999
    assert analyses["gain_total"]["m32"]["norm"] < 1.0e-14
    assert analyses["utility"]["m32"]["norm"] < 1.0e-14
    component_cosine = direction_cosine(
        analyses["intent_gain"]["m32"]["direction"],
        analyses["weighted_physics_gain"]["m32"]["direction"],
    )
    assert component_cosine is not None and component_cosine < -0.999


def test_joint_row_permutation_is_invariant_but_misalignment_is_detected() -> None:
    """S1/C4/T-order+T-role: joint rows are identities; action-only swaps are not."""

    scenario = _linear_scenario("row-identity")
    expected = analyze_scenario(scenario, partition_count=24, permutation_count=64, seed=29)
    jointly_permuted = copy.deepcopy(scenario)
    jointly_permuted["rows"] = list(reversed(jointly_permuted["rows"]))
    observed = analyze_scenario(jointly_permuted, partition_count=24, permutation_count=64, seed=29)
    assert observed == expected

    action_misaligned = copy.deepcopy(scenario)
    shifted_actions = [row["action"] for row in action_misaligned["rows"]]
    shifted_actions = shifted_actions[1:] + shifted_actions[:1]
    for row, action in zip(action_misaligned["rows"], shifted_actions, strict=True):
        row["action"] = action
    wrong = analyze_scenario(action_misaligned, partition_count=24, permutation_count=64, seed=29)
    correct_direction = expected["component_analysis"]["gain_total"]["policy_score"]["m32"]["direction"]
    wrong_direction = wrong["component_analysis"]["gain_total"]["policy_score"]["m32"]["direction"]
    assert direction_cosine(correct_direction, wrong_direction) < 0.95


def test_gain_sign_mutation_reverses_direction() -> None:
    """Sensitivity: the public analysis rejects a plausible scalar-sign mutation."""

    positive = analyze_scenario(_linear_scenario("positive"), partition_count=24, permutation_count=64, seed=31)
    negative = analyze_scenario(_linear_scenario("negative", sign=-1.0), partition_count=24, permutation_count=64, seed=31)
    positive_direction = positive["component_analysis"]["gain_total"]["policy_score"]["m32"]["direction"]
    negative_direction = negative["component_analysis"]["gain_total"]["policy_score"]["m32"]["direction"]
    cosine = direction_cosine(positive_direction, negative_direction)
    assert cosine is not None and cosine < -0.999


def test_scenario_grouping_is_preserved_and_equal_weighted() -> None:
    """S1/C4/T-role+T-dist: opposite Scenarios must not be pooled into cancellation."""

    positive = _linear_scenario("scenario-positive")
    negative = _linear_scenario("scenario-negative", sign=-1.0)
    report = analyze_payload(
        _payload(positive, negative),
        partition_count=24,
        permutation_count=64,
        seed=37,
    )
    scenario_cosines = [
        scenario["component_analysis"]["gain_total"]["policy_score"]["primary_disjoint_m16"]["cosine"]
        for scenario in report["scenarios"]
    ]
    assert all(cosine > 0.999 for cosine in scenario_cosines)
    equal_weight = report["equal_scenario_summary"]["gain_total"]["estimator_summaries"]["policy_score"][
        "primary_disjoint_m16_cosine"
    ]
    assert equal_weight["attempted_count"] == 2
    assert equal_weight["defined_count"] == 2
    assert equal_weight["mean"] > 0.999
    pooled_actions = _actions() + _actions()
    positive_values = [row["components"]["gain_total"] for row in positive["rows"]]
    negative_values = [row["components"]["gain_total"] for row in negative["rows"]]
    assert vector_norm(centered_cross_moment(pooled_actions, positive_values + negative_values)) < 1.0e-14


def test_deterministic_partitions_and_permutation_null() -> None:
    """S1/C5/T-meta: seed identity reproduces every diagnostic partition and null."""

    payload = _payload(_linear_scenario("deterministic"))
    first = analyze_payload(payload, partition_count=24, permutation_count=64, seed=41)
    second = analyze_payload(payload, partition_count=24, permutation_count=64, seed=41)
    assert first == second
    assert first["analysis_identity"]["source_transaction_active_m"] == 4
    assert first["analysis_identity"]["estimator_sample_counts"] == [4, 8, 16, 32]
    assert first["scenarios"][0]["source_transaction_identity"]["active_m"] == 4
    analysis = first["scenarios"][0]["component_analysis"]["gain_total"]["policy_score"]
    assert analysis["primary_disjoint_m16"]["scientific_unit"] == "four-complete-m4-visits-per-half"
    assert analysis["primary_disjoint_m16"]["left_visit_indices"] == [0, 1, 2, 3]
    assert analysis["primary_disjoint_m16"]["right_visit_indices"] == [4, 5, 6, 7]
    assert analysis["held_out_complement"]["evidence_role"].startswith("row-level-auxiliary")
    assert analysis["permutation_null_primary_m16"]["permutation_unit"] == "complete-m4-visit-block"
    assert set(analysis["primary_subset_directions"]) == {
        "m4_vs_m28",
        "m8_vs_m24",
        "m16_vs_m16",
    }
    assert len(analysis["primary_subset_directions"]["m4_vs_m28"]["subset"]["direction"]) == 6
    assert len(analysis["primary_subset_directions"]["m8_vs_m24"]["subset"]["direction"]) == 6
    assert len(analysis["primary_subset_directions"]["m16_vs_m16"]["subset"]["direction"]) == 6
    assert analysis["primary_subset_directions"]["m4_vs_m28"]["subset"]["estimator_sample_count"] == 4
    assert analysis["primary_subset_directions"]["m8_vs_m24"]["subset"]["estimator_sample_count"] == 8
    assert analysis["primary_subset_directions"]["m16_vs_m16"]["subset"]["estimator_sample_count"] == 16
    assert analysis["m32"]["estimator_sample_count"] == 32
    assert analysis["held_out_complement"]["m4_vs_m28"]["attempted_count"] == 24
    assert analysis["held_out_complement"]["m8_vs_m24"]["attempted_count"] == 24
    assert analysis["held_out_complement"]["m16_vs_m16"]["attempted_count"] == 24
    assert analysis["permutation_null_primary_m16"]["attempted_count"] == 64


def test_malformed_evidence_fails_closed() -> None:
    """S1/C3/T-shape+T-role: missing, non-finite, wrong identity, and bad sums fail."""

    base = _linear_scenario("malformed")
    short = copy.deepcopy(base)
    short["rows"].pop()
    _expect_input_error(lambda: analyze_scenario(short, partition_count=4, permutation_count=4), "exactly 32")

    wrong_axis = copy.deepcopy(base)
    wrong_axis["rows"][0]["action"].pop()
    _expect_input_error(lambda: analyze_scenario(wrong_axis, partition_count=4, permutation_count=4), "6 values")

    non_finite = copy.deepcopy(base)
    non_finite["rows"][0]["action"][0] = float("nan")
    _expect_input_error(lambda: analyze_scenario(non_finite, partition_count=4, permutation_count=4), "finite")

    missing_component = copy.deepcopy(base)
    missing_component["rows"][0]["components"].pop("intent_gain")
    _expect_input_error(
        lambda: analyze_scenario(missing_component, partition_count=4, permutation_count=4),
        "component identity mismatch",
    )

    bad_sum = copy.deepcopy(base)
    bad_sum["rows"][0]["components"]["gain_total"] += 1.0
    _expect_input_error(lambda: analyze_scenario(bad_sum, partition_count=4, permutation_count=4), "do not sum")

    duplicate_index = copy.deepcopy(base)
    duplicate_index["rows"][1]["repair_index"] = 0
    _expect_input_error(lambda: analyze_scenario(duplicate_index, partition_count=4, permutation_count=4), "duplicate")

    missing_mean = copy.deepcopy(base)
    missing_mean.pop("actor_mean")
    _expect_input_error(lambda: analyze_scenario(missing_mean, partition_count=4, permutation_count=4), "actor_mean")

    invalid_sigma = copy.deepcopy(base)
    invalid_sigma["actor_sigma"][2] = 0.0
    _expect_input_error(lambda: analyze_scenario(invalid_sigma, partition_count=4, permutation_count=4), "positive")

    wrong_active_m = copy.deepcopy(base)
    wrong_active_m["active_m"] = 32
    _expect_input_error(lambda: analyze_scenario(wrong_active_m, partition_count=4, permutation_count=4), "active_m")

    wrong_visit = copy.deepcopy(base)
    wrong_visit["rows"][4]["visit_index"] = 0
    _expect_input_error(
        lambda: analyze_scenario(wrong_visit, partition_count=4, permutation_count=4),
        "visit/attempt/action-seed provenance",
    )

    wrong_visit_seed = copy.deepcopy(base)
    wrong_visit_seed["rows"][4]["action_seed"] += 1
    _expect_input_error(
        lambda: analyze_scenario(wrong_visit_seed, partition_count=4, permutation_count=4),
        "disagree with visit provenance",
    )

    wrong_runtime_seed = copy.deepcopy(base)
    wrong_runtime_seed["visits"][4]["runtime_seed"] += 1
    _expect_input_error(
        lambda: analyze_scenario(wrong_runtime_seed, partition_count=4, permutation_count=4),
        "runtime_seed must remain fixed",
    )

    wrong_physics_gain = copy.deepcopy(base)
    wrong_physics_gain["rows"][0]["components"]["physics_gain"] += 0.25
    _expect_input_error(
        lambda: analyze_scenario(wrong_physics_gain, partition_count=4, permutation_count=4),
        "physics_gain identity failed",
    )

    valid_physics_na = copy.deepcopy(base)
    valid_physics_na["rows"][0]["components"]["physics_channel_noisy"][1] = None
    valid_physics_na["rows"][0]["components"]["physics_channel_repaired"][2] = None
    analyze_scenario(valid_physics_na, partition_count=4, permutation_count=4)

    invalid_required_physics = copy.deepcopy(base)
    invalid_required_physics["rows"][0]["components"]["physics_channel_noisy"][0] = None
    _expect_input_error(
        lambda: analyze_scenario(invalid_required_physics, partition_count=4, permutation_count=4),
        "must be a real scalar",
    )

    wrong_scenario_hash = copy.deepcopy(base)
    wrong_scenario_hash["manifest_file_sha256"] = "c" * 64
    _expect_input_error(
        lambda: analyze_scenario(wrong_scenario_hash, partition_count=4, permutation_count=4),
        "rows disagree",
    )

    wrong_raw_return = copy.deepcopy(base)
    wrong_raw_return["rows"][0]["components"]["raw_return"] += 1.0
    _expect_input_error(
        lambda: analyze_scenario(wrong_raw_return, partition_count=4, permutation_count=4),
        "raw_return must equal gain_total",
    )

    wrong_penalty_sign = copy.deepcopy(base)
    wrong_penalty_sign["rows"][0]["components"]["repair_penalty"] = 1.0
    _expect_input_error(
        lambda: analyze_scenario(wrong_penalty_sign, partition_count=4, permutation_count=4),
        "must equal -repair_penalty",
    )

    wrong_utility = copy.deepcopy(base)
    wrong_utility["rows"][0]["components"]["utility"] += 1.0
    _expect_input_error(
        lambda: analyze_scenario(wrong_utility, partition_count=4, permutation_count=4),
        "utility is not symlog",
    )

    duplicate_scenario = _payload(base, copy.deepcopy(base))
    _expect_input_error(
        lambda: analyze_payload(duplicate_scenario, partition_count=4, permutation_count=4),
        "scenario_id values must be unique",
    )


def main() -> None:
    test_hand_calculated_centered_cross_moment()
    test_common_gain_translation_cancels_from_both_estimators()
    test_linear_relation_is_identifiable_at_m16()
    test_m4_m8_m16_direction_and_cosine_hand_oracle()
    test_m4_m8_m16_progress_toward_independent_direction_oracle()
    test_inclusive_m32_reference_can_create_false_high_alignment()
    test_anisotropic_sigma_separates_policy_score_from_raw_covariance()
    test_pure_noise_has_no_direction()
    test_component_cancellation_remains_visible()
    test_joint_row_permutation_is_invariant_but_misalignment_is_detected()
    test_gain_sign_mutation_reverses_direction()
    test_scenario_grouping_is_preserved_and_equal_weighted()
    test_deterministic_partitions_and_permutation_null()
    test_malformed_evidence_fails_closed()
    print(
        "[T-action-gain-direction] S1 C1/C2/C3/C4/C5 value, grouping, null and sensitivity pass",
        flush=True,
    )


if __name__ == "__main__":
    main()
