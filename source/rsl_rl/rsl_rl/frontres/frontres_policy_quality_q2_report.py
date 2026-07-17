"""Offline Q2 counterfactual report for policy-quality result artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from statistics import median
from typing import Any

try:
    from .frontres_policy_quality_manifest import FrontRESPolicyQualityManifest
except ImportError:
    # Keep the offline reporter runnable without importing the simulator-heavy
    # rsl_rl.frontres package initializer.
    _manifest_path = Path(__file__).with_name("frontres_policy_quality_manifest.py")
    _manifest_spec = importlib.util.spec_from_file_location("frontres_policy_quality_manifest", _manifest_path)
    if _manifest_spec is None or _manifest_spec.loader is None:
        raise RuntimeError(f"cannot load policy-quality manifest owner: {_manifest_path}")
    _manifest_module = importlib.util.module_from_spec(_manifest_spec)
    sys.modules[_manifest_spec.name] = _manifest_module
    _manifest_spec.loader.exec_module(_manifest_module)
    FrontRESPolicyQualityManifest = _manifest_module.FrontRESPolicyQualityManifest


_RESULT_SCHEMA = "frontres_policy_quality_result_v1"
_REPORT_SCHEMA = "frontres_policy_quality_q2_report_v1"
_ROUTES = frozenset(("zero", "hsl", "policy"))


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _finite_scalar(values: object, *, name: str) -> float:
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError(f"{name} must contain exactly one scalar")
    value = values[0]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must contain one finite scalar")
    return float(value)


def _bounded(value: object, *, name: str, tolerance: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if abs(float(value)) > tolerance:
        raise ValueError(f"{name} exceeds matched-role tolerance: {value} > {tolerance}")


def _classify(delta: float, epsilon: float) -> str:
    if delta > epsilon:
        return "positive"
    if delta < -epsilon:
        return "negative"
    return "unresolved"


def _route_components(route: dict[str, Any], *, name: str) -> dict[str, float]:
    gain = route.get("gain")
    if not isinstance(gain, dict):
        raise ValueError(f"{name}.gain must be an object")
    return {
        field: _finite_scalar(gain.get(field), name=f"{name}.{field}")
        for field in ("gain_total", "style_gain", "physics_gain", "repair_cost")
    }


def _failure_owner(*, total_delta: float, pre_cost_delta: float, epsilon: float) -> str:
    if total_delta > epsilon:
        return "resolved_improvement"
    if total_delta >= -epsilon:
        return "unresolved_at_zero_noise_floor"
    if pre_cost_delta < -epsilon:
        return "execution_degradation_before_cost"
    if pre_cost_delta > epsilon:
        return "repair_cost_dominance"
    return "insufficient_pre_cost_margin_after_cost"


def _shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    if not value:
        return (0,)
    child_shapes = {_shape(child) for child in value}
    if len(child_shapes) != 1:
        raise ValueError("HSL supervision arrays must be rectangular")
    return (len(value), *child_shapes.pop())


def _flatten_finite(value: object, *, name: str) -> list[float]:
    if isinstance(value, list):
        result: list[float] = []
        for child in value:
            result.extend(_flatten_finite(child, name=name))
        return result
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must contain only finite numeric values")
    return [float(value)]


def _hsl_target_alignment(row: dict[str, Any], *, item_id: str, horizon_k: int) -> dict[str, Any] | None:
    execution = row["routes"]["hsl"].get("execution")
    supervision = execution.get("hsl_supervision") if isinstance(execution, dict) else None
    if supervision is None:
        return None
    if not isinstance(supervision, dict):
        raise ValueError(f"{item_id}.hsl_supervision must be an object")
    required = frozenset(
        (
            "targets",
            "sample_weights",
            "harm_weights",
            "target_nonzero",
            "action_target_l2",
            "action_target_cosine",
            "sign_agree_per_dim",
        )
    )
    if frozenset(supervision) != required:
        raise ValueError(f"{item_id}.hsl_supervision fields do not match the Q2-B schema")
    targets = supervision["targets"]
    target_shape = _shape(targets)
    if len(target_shape) != 3 or target_shape[0] != horizon_k or target_shape[2] != 6 or target_shape[1] <= 0:
        raise ValueError(f"{item_id}.hsl targets must have [K, B, 6] shape")
    batch = target_shape[1]
    expected_shapes = {
        "sample_weights": (horizon_k, batch, 1),
        "harm_weights": (horizon_k, batch, 1),
        "target_nonzero": (horizon_k, batch),
        "action_target_l2": (horizon_k, batch),
        "action_target_cosine": (horizon_k, batch),
        "sign_agree_per_dim": (horizon_k, batch, 6),
    }
    for field, expected in expected_shapes.items():
        if _shape(supervision[field]) != expected:
            raise ValueError(f"{item_id}.hsl_supervision.{field} must have shape {expected}")
    nonzero_raw = supervision["target_nonzero"]
    if any(not isinstance(value, bool) for step in nonzero_raw for value in step):
        raise ValueError(f"{item_id}.hsl target_nonzero must be boolean")
    nonzero = [value for step in nonzero_raw for value in step]
    weights = _flatten_finite(supervision["sample_weights"], name=f"{item_id}.sample_weights")
    harm_weights = _flatten_finite(supervision["harm_weights"], name=f"{item_id}.harm_weights")
    l2 = _flatten_finite(supervision["action_target_l2"], name=f"{item_id}.action_target_l2")
    cosine = _flatten_finite(supervision["action_target_cosine"], name=f"{item_id}.action_target_cosine")
    sign = supervision["sign_agree_per_dim"]
    sign_by_dim = [
        [float(sign[t][b][dim]) for t in range(horizon_k) for b in range(batch)]
        for dim in range(6)
    ]
    active = [index for index, (is_nonzero, weight) in enumerate(zip(nonzero, weights)) if is_nonzero and weight > 0.0]
    mean = lambda values: sum(values) / len(values) if values else 0.0
    return {
        "shape": [horizon_k, batch, 6],
        "active_count": len(active),
        "target_nonzero_frac": mean([float(value) for value in nonzero]),
        "sample_weight_mean": mean(weights),
        "harm_weight_mean": mean(harm_weights),
        "action_target_l2_mean_active": mean([l2[index] for index in active]),
        "action_target_cosine_mean_active": mean([cosine[index] for index in active]),
        "sign_agree_per_dim_mean_active": [
            mean([values[index] for index in active]) for values in sign_by_dim
        ],
    }


def _validate_role_identity(role_identity: object, *, item_id: str) -> None:
    if not isinstance(role_identity, dict):
        raise ValueError(f"{item_id}.role_identity must be an object")
    policy_noisy = role_identity.get("policy_noisy")
    corruption = role_identity.get("corruption_present")
    role_rows = role_identity.get("role_rows")
    if not isinstance(policy_noisy, dict) or not isinstance(corruption, dict) or not isinstance(role_rows, dict):
        raise ValueError(f"{item_id}.role_identity is incomplete")
    if frozenset(role_rows) != frozenset(("policy", "candidate", "noisy", "clean")):
        raise ValueError(f"{item_id}.role_rows must identify the quartet")

    # QUALITY-ID-01: Q2 compares routes only after policy/noisy local dynamics
    # and command perturbation caches are proven to describe one sampled item.
    for field in (
        "joint_pos_max_abs",
        "joint_vel_max_abs",
        "root_quat_max_abs",
        "root_lin_vel_max_abs",
        "root_ang_vel_max_abs",
        "cached_perturbed_pos_max_abs",
        "cached_perturbed_quat_max_abs",
    ):
        _bounded(policy_noisy.get(field), name=f"{item_id}.policy_noisy.{field}", tolerance=1.0e-6)
    _bounded(
        policy_noisy.get("local_root_pos_max_abs"),
        name=f"{item_id}.policy_noisy.local_root_pos_max_abs",
        tolerance=1.0e-5,
    )
    corruption_delta = corruption.get("policy_clean_cached_quat_max_abs")
    if not isinstance(corruption_delta, (int, float)) or not math.isfinite(float(corruption_delta)):
        raise ValueError(f"{item_id}.corruption quaternion delta must be finite")
    if float(corruption_delta) <= 1.0e-8:
        raise ValueError(f"{item_id}.local_rp corruption is absent")


def build_frontres_policy_quality_q2_report(
    manifest: FrontRESPolicyQualityManifest,
    result: object,
    *,
    require_hsl_supervision: bool = False,
) -> dict[str, Any]:
    """Validate one matched Q2 artifact and preserve every item before aggregation."""
    if not isinstance(result, dict):
        raise ValueError("policy-quality result must be an object")
    if result.get("schema_version") != _RESULT_SCHEMA:
        raise ValueError("unsupported policy-quality result schema")
    if result.get("comparison_signature") != manifest.comparison_signature:
        raise ValueError("result and manifest comparison signatures differ")
    rows = result.get("rows")
    if not isinstance(rows, list):
        raise ValueError("policy-quality result rows must be a list")

    expected = {item.item_id: item for item in manifest.items}
    observed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("item"), dict):
            raise ValueError("each result row must contain an item object")
        item_id = row["item"].get("item_id")
        if item_id not in expected:
            raise ValueError(f"unexpected Q2 item: {item_id!r}")
        if item_id in observed:
            raise ValueError(f"duplicate Q2 item: {item_id}")
        item = expected[item_id]
        if row["item"] != item.to_dict() or row.get("comparison_signature") != item.comparison_signature:
            raise ValueError(f"{item_id} does not match its immutable manifest item")
        _validate_role_identity(row.get("role_identity"), item_id=item_id)
        routes = row.get("routes")
        if not isinstance(routes, dict) or frozenset(routes) != _ROUTES:
            raise ValueError(f"{item_id}.routes must be exactly zero/hsl/policy")
        state_hashes = {route.get("initial_state_hash") for route in routes.values() if isinstance(route, dict)}
        if len(state_hashes) != 1 or not next(iter(state_hashes), None):
            raise ValueError(f"{item_id} routes do not share one initial_state_hash")
        observed[item_id] = row
    if frozenset(observed) != frozenset(expected):
        missing = sorted(frozenset(expected) - frozenset(observed))
        raise ValueError(f"Q2 result is missing manifest items: {missing}")

    checkpoint_by_route: dict[str, set[str]] = {route: set() for route in _ROUTES}
    inferred_repair_weights: list[float] = []
    item_reports: list[dict[str, Any]] = []
    for item in manifest.items:
        row = observed[item.item_id]
        route_components: dict[str, dict[str, float]] = {}
        for route_name, route in row["routes"].items():
            checkpoint = route.get("checkpoint_identity")
            if not isinstance(checkpoint, str) or not checkpoint.strip():
                raise ValueError(f"{item.item_id}.{route_name} checkpoint identity is missing")
            checkpoint_by_route[route_name].add(checkpoint)
            route_components[route_name] = _route_components(route, name=f"{item.item_id}.{route_name}")
            components = route_components[route_name]
            if components["repair_cost"] > 1.0e-12:
                inferred_repair_weights.append(
                    (components["style_gain"] + components["physics_gain"] - components["gain_total"])
                    / components["repair_cost"]
                )
        route_gains = {route: values["gain_total"] for route, values in route_components.items()}
        epsilon = abs(route_gains["zero"])
        hsl_zero = route_gains["hsl"] - route_gains["zero"]
        policy_zero = route_gains["policy"] - route_gains["zero"]
        policy_hsl = route_gains["policy"] - route_gains["hsl"]
        pre_cost = {
            route: values["style_gain"] + values["physics_gain"]
            for route, values in route_components.items()
        }
        pre_cost_delta = {
            "hsl_zero": pre_cost["hsl"] - pre_cost["zero"],
            "policy_zero": pre_cost["policy"] - pre_cost["zero"],
            "policy_hsl": pre_cost["policy"] - pre_cost["hsl"],
        }
        hsl_alignment = _hsl_target_alignment(
            row,
            item_id=item.item_id,
            horizon_k=item.effective_horizon_k,
        )
        if require_hsl_supervision and hsl_alignment is None:
            raise ValueError(f"{item.item_id} is missing required Q2-B HSL supervision")
        item_reports.append(
            {
                "item_id": item.item_id,
                "motion_id": item.motion_id,
                "seed": item.seed,
                "effective_horizon_k": item.effective_horizon_k,
                "zero_noise_floor": epsilon,
                "gain": route_gains,
                "components": route_components,
                "pre_cost_style_physics": pre_cost,
                "pre_cost_delta": pre_cost_delta,
                "delta": {
                    "hsl_zero": hsl_zero,
                    "policy_zero": policy_zero,
                    "policy_hsl": policy_hsl,
                },
                "classification": {
                    "hsl_zero": _classify(hsl_zero, epsilon),
                    "policy_zero": _classify(policy_zero, epsilon),
                    "policy_hsl": _classify(policy_hsl, epsilon),
                },
                "failure_owner": {
                    "hsl_zero": _failure_owner(
                        total_delta=hsl_zero, pre_cost_delta=pre_cost_delta["hsl_zero"], epsilon=epsilon
                    ),
                    "policy_zero": _failure_owner(
                        total_delta=policy_zero,
                        pre_cost_delta=pre_cost_delta["policy_zero"],
                        epsilon=epsilon,
                    ),
                    "policy_hsl": _failure_owner(
                        total_delta=policy_hsl,
                        pre_cost_delta=pre_cost_delta["policy_hsl"],
                        epsilon=epsilon,
                    ),
                },
                "hsl_target_alignment": hsl_alignment,
            }
        )

    if any(len(values) != 1 for values in checkpoint_by_route.values()):
        raise ValueError("each Q2 route must use one checkpoint identity across all items")
    if checkpoint_by_route["hsl"] == checkpoint_by_route["policy"]:
        raise ValueError("HSL and policy checkpoints must be distinct")
    repair_weight = median(inferred_repair_weights)
    if not inferred_repair_weights or any(abs(weight - repair_weight) > 1.0e-5 for weight in inferred_repair_weights):
        raise ValueError("Q2 rows do not share one reconstructable repair weight")

    by_motion: dict[str, list[dict[str, Any]]] = {}
    for item_report in item_reports:
        by_motion.setdefault(item_report["motion_id"], []).append(item_report)
    motion_reports: list[dict[str, Any]] = []
    for motion_id, motion_items in sorted(by_motion.items()):
        if len(motion_items) != 2 or len({item["seed"] for item in motion_items}) != 2:
            raise ValueError(f"{motion_id} must have exactly two distinct matched seeds")
        classes: dict[str, str] = {}
        for comparison in ("hsl_zero", "policy_zero", "policy_hsl"):
            observed_classes = {item["classification"][comparison] for item in motion_items}
            classes[comparison] = observed_classes.pop() if len(observed_classes) == 1 else "mixed"
        motion_reports.append(
            {
                "motion_id": motion_id,
                "seeds": sorted(item["seed"] for item in motion_items),
                "classification": classes,
            }
        )

    counts = {
        comparison: {
            label: sum(motion["classification"][comparison] == label for motion in motion_reports)
            for label in ("positive", "negative", "unresolved", "mixed")
        }
        for comparison in ("hsl_zero", "policy_zero", "policy_hsl")
    }
    policy_hsl_values = [item["delta"]["policy_hsl"] for item in item_reports]
    method_review_required = counts["hsl_zero"]["negative"] + counts["hsl_zero"]["unresolved"] >= 3
    return {
        "schema_version": _REPORT_SCHEMA,
        "comparison_signature": manifest.comparison_signature,
        "technical_pass": True,
        "items": item_reports,
        "motions": motion_reports,
        "counts": counts,
        "inferred_repair_weight": repair_weight,
        "failure_owner_counts": {
            comparison: {
                owner: sum(item["failure_owner"][comparison] == owner for item in item_reports)
                for owner in (
                    "resolved_improvement",
                    "unresolved_at_zero_noise_floor",
                    "execution_degradation_before_cost",
                    "repair_cost_dominance",
                    "insufficient_pre_cost_margin_after_cost",
                )
            }
            for comparison in ("hsl_zero", "policy_zero", "policy_hsl")
        },
        "verdict": {
            "oracle_valid": counts["hsl_zero"]["positive"] >= 6,
            "policy_useful": counts["policy_zero"]["positive"] >= 6,
            "ppo_improvement_supported": (
                counts["policy_hsl"]["positive"] >= 5 and median(policy_hsl_values) > 0.0
            ),
            "ppo_regression_supported": (
                counts["policy_hsl"]["negative"] >= 5 and median(policy_hsl_values) < 0.0
            ),
            "method_review_required": method_review_required,
            "policy_hsl_item_median": median(policy_hsl_values),
        },
    }


def write_frontres_policy_quality_q2_report(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize one FrontRES Q2 result artifact.")
    parser.add_argument("manifest")
    parser.add_argument("result")
    parser.add_argument("output")
    parser.add_argument("--require-hsl-supervision", action="store_true")
    args = parser.parse_args()
    manifest = FrontRESPolicyQualityManifest.from_json(Path(args.manifest).read_text(encoding="utf-8"))
    report = build_frontres_policy_quality_q2_report(
        manifest,
        _load_json(args.result),
        require_hsl_supervision=args.require_hsl_supervision,
    )
    write_frontres_policy_quality_q2_report(report, args.output)
    for item in report["items"]:
        delta = item["delta"]
        print(
            f"[Q2 item] id={item['item_id']} seed={item['seed']} epsilon={item['zero_noise_floor']:.6g} "
            f"hsl_zero={delta['hsl_zero']:.6g} policy_zero={delta['policy_zero']:.6g} "
            f"policy_hsl={delta['policy_hsl']:.6g} class={item['classification']['policy_hsl']} "
            f"hsl_owner={item['failure_owner']['hsl_zero']}"
        )
    print(f"[Q2 verdict] {json.dumps(report['verdict'], sort_keys=True)}")


if __name__ == "__main__":
    main()
