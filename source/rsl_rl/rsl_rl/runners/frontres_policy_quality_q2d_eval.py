"""Dedicated Q2-D evaluator; isolated from existing quality and sequence control flows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from rsl_rl.frontres.frontres_policy_quality_q2d import Q2D_SCALE_FACTORS, run_q2d_scale_sweep
from rsl_rl.runners.frontres_policy_quality_interfaces import (
    FrontRESPolicyQualityEvalRequest,
)
from rsl_rl.runners.frontres_policy_quality_state import restore_frontres_policy_quality_state
from rsl_rl.runners.frontres_policy_quality_formal_owners import (
    build_frontres_policy_quality_formal_owner_bundle,
    frontres_policy_quality_json_value,
)


def run_frontres_policy_quality_q2d_scale_eval(
    runner: Any,
    *,
    request: FrontRESPolicyQualityEvalRequest,
    result_path: str,
    scales: Iterable[float] = Q2D_SCALE_FACTORS,
) -> dict[str, Any]:
    """Execute only scaled-HSL routes through the canonical lower-level owners."""

    # B1: Q2-D1 创建独立 owner bundle, 不安装或调用旧 manifest executor.
    owners = build_frontres_policy_quality_formal_owner_bundle(runner, request)
    isolated_before = owners.isolation_state(runner)
    rows = []
    for item in request.manifest.items:
        snapshot, adapters, hooks = owners.prepare_item(runner, item, request)
        hsl_adapter = next(adapter for adapter in adapters if adapter.route == "hsl")

        # B6: QUALITY-ACTION-01 每个 scale 恢复同一 state, 再经过正式 action/Gain owners.
        results = run_q2d_scale_sweep(
            base_adapter=hsl_adapter,
            scales=scales,
            horizon_k=item.effective_horizon_k,
            restore_state=lambda: restore_frontres_policy_quality_state(
                runner,
                snapshot,
                comparison_signature=item.comparison_signature,
            ).initial_state_hash,
            begin_route=hooks.begin_route or (lambda _route: None),
            observe=hooks.observe,
            apply_action=hooks.apply_action,
            step=hooks.step,
            compute_gain=hooks.compute_gain,
            capture_execution=hooks.capture_execution,
            isolation_state=lambda: owners.isolation_state(runner),
            set_audit_identity=(
                lambda route, scale, state_hash: hooks.set_audit_identity(
                    {
                        "audit_transaction_id": f"q2d:{item.comparison_signature[:12]}:{route}",
                        "audit_batch_signature": hashlib.sha1(
                            repr((item.comparison_signature, state_hash, route, scale, item.effective_horizon_k)).encode("utf-8")
                        ).hexdigest()[:16],
                        "audit_identity_state": "complete",
                    }
                )
                if hooks.set_audit_identity is not None
                else None
            ),
        )
        rows.append(
            {
                "item": item.to_dict(),
                "comparison_signature": item.comparison_signature,
                "routes": {
                    result.route: {
                        "scale": result.scale,
                        "initial_state_hash": result.initial_state_hash,
                        "actions": frontres_policy_quality_json_value(result.actions),
                        "gain": frontres_policy_quality_json_value(result.gain),
                        "execution": frontres_policy_quality_json_value(result.execution),
                    }
                    for result in results
                },
            }
        )
    if owners.isolation_state(runner) != isolated_before:
        raise RuntimeError("Q2-D evaluator mutated optimizer/sampler/warmup state")

    # B3: 原子 artifact 保留 scale ordering 和逐 item Gain, 不覆盖 Q1/Q2 result.
    payload = {
        "schema_version": "frontres_policy_quality_q2d_scale_result_v1",
        "comparison_signature": request.manifest.comparison_signature,
        "scales": list(float(scale) for scale in scales),
        "owner_identity": dict(owners.owner_identity),
        "rows": rows,
    }
    destination = Path(result_path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
    temporary.replace(destination)
    return payload
