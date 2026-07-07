#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from math import exp


@dataclass(frozen=True)
class RewardTerm:
    name: str
    weight: float
    group: str
    raw_min: float
    raw_max: float | None
    note: str


RL_FINETUNE_REWARD_TERMS = (
    RewardTerm("contact_feet", 1.0, "task/contact", 0.0, 1.0, "two-foot contact agreement"),
    RewardTerm("motion_feet_pos", 1.0, "task/feet", 0.0, 1.0, "exp position reward, std=0.05m"),
    RewardTerm("motion_body_pos", 0.5, "task/body", 0.0, 1.0, "exp position reward, std=0.30m"),
    RewardTerm("motion_body_ori", 0.5, "task/body", 0.0, 1.0, "exp orientation reward, std=0.40rad"),
    RewardTerm("motion_body_lin_vel", 0.75, "task/velocity", 0.0, 1.0, "exp linear velocity reward, std=1.00"),
    RewardTerm("motion_body_ang_vel", 0.75, "task/velocity", 0.0, 1.0, "exp angular velocity reward, std=3.14"),
    RewardTerm("motion_global_anchor_pos", 0.5, "task/root", 0.0, 1.0, "exp root position reward, std=0.30m"),
    RewardTerm("motion_global_anchor_ori", 0.5, "task/root", 0.0, 1.0, "exp root orientation reward, std=0.40rad"),
    RewardTerm("undesired_contacts", -2.0, "safety/contact", 0.0, 1.0, "non-foot/hand contact indicator"),
    RewardTerm("joint_limit", -20.0, "safety/joint", 0.0, None, "joint limit violation magnitude"),
    RewardTerm("joint_torque", -2e-4, "naturalness/effort", 0.0, None, "sum squared joint torques"),
)

BALANCE_WEIGHT_CANDIDATES = (0.5, 1.0, 2.0, 5.0, 10.0)


def weighted_total(values: dict[str, float], *, balance_weight: float = 0.0, balance_gain: float = 0.0) -> float:
    total = sum(term.weight * values.get(term.name, 0.0) for term in RL_FINETUNE_REWARD_TERMS)
    return total + balance_weight * balance_gain


def term_contributions(values: dict[str, float]) -> dict[str, float]:
    return {term.name: term.weight * values.get(term.name, 0.0) for term in RL_FINETUNE_REWARD_TERMS}


def positive_reward_ceiling() -> float:
    return weighted_total({term.name: 1.0 for term in RL_FINETUNE_REWARD_TERMS if term.weight > 0.0})


def clean_relative_no_regret_margin(repaired_margin: float, noisy_margin: float, clean_margin: float, slack: float) -> float:
    clean_floor = clean_margin - slack
    noisy_risk = max(clean_floor - noisy_margin, 0.0)
    repaired_risk = max(clean_floor - repaired_margin, 0.0)
    return noisy_risk - repaired_risk


def scan_balance_weights(
    values: dict[str, float],
    balance_gain: float,
    candidates: tuple[float, ...] = BALANCE_WEIGHT_CANDIDATES,
) -> dict[float, float]:
    base = weighted_total(values)
    return {weight: weighted_total(values, balance_weight=weight, balance_gain=balance_gain) - base for weight in candidates}


def exp_reward(error: float, std: float) -> float:
    return exp(-((error / std) ** 2))


def assert_close(actual: float, expected: float, tol: float = 1e-9) -> None:
    assert abs(actual - expected) <= tol, (actual, expected)


def representative_cases() -> dict[str, dict[str, float]]:
    perfect = {term.name: 1.0 for term in RL_FINETUNE_REWARD_TERMS if term.weight > 0.0}
    normal_repair = {
        "contact_feet": 0.8,
        "motion_feet_pos": 0.7,
        "motion_body_pos": 0.6,
        "motion_body_ori": 0.6,
        "motion_body_lin_vel": 0.5,
        "motion_body_ang_vel": 0.5,
        "motion_global_anchor_pos": 0.6,
        "motion_global_anchor_ori": 0.6,
        "undesired_contacts": 0.0,
        "joint_limit": 0.0,
        "joint_torque": 10.0,
    }
    feet_bad = dict(normal_repair, contact_feet=0.0, motion_feet_pos=0.05)
    body_bad = dict(normal_repair, motion_body_pos=0.05, motion_body_ori=0.05)
    root_bad = dict(normal_repair, motion_global_anchor_pos=0.05, motion_global_anchor_ori=0.05)
    unsafe_contact = dict(perfect, undesired_contacts=1.0)
    joint_limit_hit = dict(perfect, joint_limit=1.0)
    high_torque = dict(perfect, joint_torque=10_000.0)
    broken = {
        "contact_feet": 0.0,
        "motion_feet_pos": 0.0,
        "motion_body_pos": 0.05,
        "motion_body_ori": 0.05,
        "motion_body_lin_vel": 0.05,
        "motion_body_ang_vel": 0.05,
        "motion_global_anchor_pos": 0.05,
        "motion_global_anchor_ori": 0.05,
        "undesired_contacts": 1.0,
        "joint_limit": 1.0,
        "joint_torque": 10_000.0,
    }
    return {
        "perfect_tracking": perfect,
        "normal_repair": normal_repair,
        "feet_bad_only": feet_bad,
        "body_bad_only": body_bad,
        "root_bad_only": root_bad,
        "unsafe_contact_perfect_tracking": unsafe_contact,
        "joint_limit_perfect_tracking": joint_limit_hit,
        "high_torque_perfect_tracking": high_torque,
        "broken_multi_failure": broken,
    }


def format_table(rows: list[list[object]]) -> str:
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) for row in rows)


def emit_reward_term_table() -> None:
    rows: list[list[object]] = [["term", "group", "weight", "raw_range", "weighted_range", "note"]]
    for term in RL_FINETUNE_REWARD_TERMS:
        raw_range = f"[{term.raw_min:g}, {term.raw_max:g}]" if term.raw_max is not None else f"[{term.raw_min:g}, inf)"
        weighted_max = "unbounded" if term.raw_max is None else f"{term.weight * term.raw_max:.4g}"
        rows.append([term.name, term.group, f"{term.weight:g}", raw_range, weighted_max, term.note])
    print("[T1 Reward Term Scale]")
    print(format_table(rows))


def emit_case_matrix() -> None:
    ceiling = positive_reward_ceiling()
    rows: list[list[object]] = [["case", "total", "total/max_pos", "top_positive", "top_penalty", "sentinel"]]
    for name, values in representative_cases().items():
        contribs = term_contributions(values)
        positives = {k: v for k, v in contribs.items() if v > 0.0}
        penalties = {k: v for k, v in contribs.items() if v < 0.0}
        top_positive = max(positives.items(), key=lambda item: item[1]) if positives else ("none", 0.0)
        top_penalty = min(penalties.items(), key=lambda item: item[1]) if penalties else ("none", 0.0)
        total = weighted_total(values)
        sentinel = "ok"
        if name == "unsafe_contact_perfect_tracking" and total > 0.0:
            sentinel = "contact penalty is not hard safety"
        if name == "joint_limit_perfect_tracking" and total < 0.0:
            sentinel = "joint limit dominates"
        if name == "high_torque_perfect_tracking" and total > 0.0:
            sentinel = "torque penalty weak unless raw torque is huge"
        rows.append([
            name,
            f"{total:.4f}",
            f"{100.0 * total / ceiling:.1f}%",
            f"{top_positive[0]}={top_positive[1]:.4f}",
            f"{top_penalty[0]}={top_penalty[1]:.4f}",
            sentinel,
        ])
    print("\n[T2 Representative Case Matrix]")
    print(format_table(rows))


def emit_exp_decay_table() -> None:
    specs = (
        ("motion_feet_pos", 0.05, "m"),
        ("motion_body_pos", 0.30, "m"),
        ("motion_body_ori", 0.40, "rad"),
        ("motion_body_lin_vel", 1.00, "unit"),
        ("motion_body_ang_vel", 3.14, "rad/s"),
        ("motion_global_anchor_pos", 0.30, "m"),
        ("motion_global_anchor_ori", 0.40, "rad"),
    )
    multiples = (0.0, 0.5, 1.0, 2.0, 3.0)
    rows: list[list[object]] = [["term", "std", *[f"{m:g}x_std" for m in multiples]]]
    for name, std, unit in specs:
        rows.append([name, f"{std:g}{unit}", *[f"{exp_reward(m * std, std):.4f}" for m in multiples]])
    print("\n[T1 Exp Reward Decay]")
    print(format_table(rows))


def emit_balance_candidate_scan() -> None:
    base_values = representative_cases()["normal_repair"]
    ceiling = positive_reward_ceiling()
    gains = {
        "good_repair": clean_relative_no_regret_margin(-0.10, -0.20, 0.0, 0.0),
        "weak_repair": clean_relative_no_regret_margin(-0.15, -0.20, 0.0, 0.0),
        "no_change": clean_relative_no_regret_margin(-0.20, -0.20, 0.0, 0.0),
        "bad_repair": clean_relative_no_regret_margin(-0.25, -0.10, 0.0, 0.0),
        "dynamic_clean_good": clean_relative_no_regret_margin(-0.20, -0.30, -0.10, 0.0),
    }
    rows: list[list[object]] = [["scenario", "gain", *[f"w={w:g}" for w in BALANCE_WEIGHT_CANDIDATES], "sentinel"]]
    for name, gain in gains.items():
        deltas = scan_balance_weights(base_values, gain)
        sentinel = "ok"
        if max(abs(delta) for delta in deltas.values()) > 0.10 * ceiling:
            sentinel = "w>=10 can dominate small task terms"
        rows.append([name, f"{gain:.4f}", *[f"{deltas[w]:+.4f}" for w in BALANCE_WEIGHT_CANDIDATES], sentinel])
    print("\n[T1 Balance Candidate Weight Sweep]")
    print(format_table(rows))


def emit_hacking_thresholds() -> None:
    ceiling = positive_reward_ceiling()
    torque_raw_to_cancel_all_positive = ceiling / abs(next(t.weight for t in RL_FINETUNE_REWARD_TERMS if t.name == "joint_torque"))
    rows = [
        ["sentinel", "threshold", "meaning"],
        ["max_positive_reward", f"{ceiling:.4f}", "upper bound from positive terms"],
        ["undesired_contact_once", "-2.0000", "can be outweighed by perfect tracking"],
        ["joint_limit_raw_1", "-20.0000", "hard safety dominates positive terms"],
        ["torque_raw_to_cancel_max_positive", f"{torque_raw_to_cancel_all_positive:.1f}", "joint_torque raw sum needed to cancel all positives"],
    ]
    print("\n[Reward-Hacking Thresholds]")
    print(format_table(rows))


def test_rl_finetune_reward_snapshot_matches_current_training_weights() -> None:
    weights = {term.name: term.weight for term in RL_FINETUNE_REWARD_TERMS}

    assert weights["contact_feet"] == 1.0
    assert weights["motion_feet_pos"] == 1.0
    assert weights["motion_body_pos"] == 0.5
    assert weights["motion_body_ori"] == 0.5
    assert weights["motion_body_lin_vel"] == 0.75
    assert weights["motion_body_ang_vel"] == 0.75
    assert weights["motion_global_anchor_pos"] == 0.5
    assert weights["motion_global_anchor_ori"] == 0.5
    assert weights["undesired_contacts"] == -2.0
    assert weights["joint_limit"] == -20.0
    assert weights["joint_torque"] == -2e-4


def test_total_reward_scale_has_expected_positive_and_safety_ranges() -> None:
    max_positive = positive_reward_ceiling()
    assert_close(max_positive, 5.5)

    cases = representative_cases()
    assert weighted_total(cases["joint_limit_perfect_tracking"]) < 0.0
    assert weighted_total(cases["unsafe_contact_perfect_tracking"]) > 0.0
    assert weighted_total(cases["broken_multi_failure"]) < -20.0


def test_balance_weight_scan_is_visible_but_not_dominant_at_low_weights() -> None:
    normal_repair = representative_cases()["normal_repair"]
    base = weighted_total(normal_repair)
    assert 3.0 < base < 4.5

    balance_gain = clean_relative_no_regret_margin(
        repaired_margin=-0.10,
        noisy_margin=-0.20,
        clean_margin=0.00,
        slack=0.0,
    )
    assert_close(balance_gain, 0.10)

    deltas = scan_balance_weights(normal_repair, balance_gain)
    assert_close(deltas[0.5], 0.05)
    assert_close(deltas[1.0], 0.10)
    assert_close(deltas[2.0], 0.20)
    assert_close(deltas[5.0], 0.50)
    assert_close(deltas[10.0], 1.00)

    assert deltas[1.0] > 0.02 * base
    assert deltas[2.0] < 0.10 * positive_reward_ceiling()


def main() -> None:
    test_rl_finetune_reward_snapshot_matches_current_training_weights()
    test_total_reward_scale_has_expected_positive_and_safety_ranges()
    test_balance_weight_scan_is_visible_but_not_dominant_at_low_weights()
    emit_reward_term_table()
    emit_case_matrix()
    emit_exp_decay_table()
    emit_balance_candidate_scan()
    emit_hacking_thresholds()
    print("\nfrontres_total_reward_scale_contract: ok")


if __name__ == "__main__":
    main()
