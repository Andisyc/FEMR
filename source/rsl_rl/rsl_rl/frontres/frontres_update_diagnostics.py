from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any, Callable

import torch

from rsl_rl.frontres.frontres_formal_runtime_probe import emit_formal_runtime_probe


_V004_PROJECTION_STATUSES = {
    "INTENT_FEASIBLE",
    "PROJECTED_INTENT",
    "CONSTRAINT_RECOVERY",
    "NO_EMPIRICAL_DIRECTION",
    "NO_COMMON_FIRST_ORDER_DESCENT",
}
_V004_CONSTRAINT_FAMILIES = {"contact", "zmp", "survival"}


@dataclass(frozen=True)
class FrontRESV004ActualUpdateTelemetry:
    projection_status: str
    actual_projection_status: str
    active_families: tuple[str, ...]
    directional_derivatives: dict[str, float]
    kkt_max_violation: float
    gradient_kkt_max_violation: float
    optimizer_candidate_actor_delta_l2: float
    committed_actor_delta_l2: float
    actor_optimizer_state_preserved: bool
    actor_loss_weight: float


def validate_frontres_v004_actual_update_telemetry(
    diagnostics: Mapping[str, Any],
    *,
    tolerance: float,
) -> FrontRESV004ActualUpdateTelemetry:
    """Validate the final post-optimizer Actor authority before serialization."""

    def finite(name: str) -> float:
        if name not in diagnostics:
            raise RuntimeError(f"v015 formal result is missing {name} telemetry")
        value = float(diagnostics[name])
        if not math.isfinite(value):
            raise RuntimeError(f"v015 formal result has non-finite {name} telemetry")
        return value

    projection_status = str(diagnostics.get("constraint_projection_status", ""))
    if projection_status not in _V004_PROJECTION_STATUSES:
        raise RuntimeError(f"v015 formal result has invalid constraint projection status: {projection_status!r}")
    actual_status = str(diagnostics.get("actual_update_projection_status", ""))
    if actual_status != projection_status:
        raise RuntimeError(
            "v015 formal result actual update status disagrees with its gradient projection: "
            f"gradient={projection_status!r} actual={actual_status!r}"
        )
    active_families = tuple(str(value) for value in diagnostics.get("constraint_active_families", ()))
    if len(set(active_families)) != len(active_families) or not set(active_families) <= _V004_CONSTRAINT_FAMILIES:
        raise RuntimeError(f"v015 formal result has invalid active constraint families: {active_families}")
    raw_derivatives = diagnostics.get("constraint_directional_derivatives")
    if not isinstance(raw_derivatives, Mapping):
        raise RuntimeError("v015 formal result is missing constraint_directional_derivatives telemetry")
    derivatives = {str(key): float(value) for key, value in raw_derivatives.items()}
    if not set(derivatives) <= _V004_CONSTRAINT_FAMILIES or not all(
        math.isfinite(value) for value in derivatives.values()
    ):
        raise RuntimeError("v015 formal result has invalid constraint_directional_derivatives telemetry")

    kkt = finite("constraint_kkt_max_violation")
    if not 0.0 <= kkt <= float(tolerance):
        raise RuntimeError(
            "v015 formal result exceeds the checkpoint-v6 constraint projection tolerance: "
            f"kkt={kkt:.9g} tolerance={float(tolerance):.9g}"
        )
    observed_kkt = max((max(0.0, value) for value in derivatives.values()), default=0.0)
    if abs(observed_kkt - kkt) > float(tolerance):
        raise RuntimeError(
            "v015 formal result has inconsistent constraint KKT telemetry: "
            f"reported={kkt:.9g} observed={observed_kkt:.9g}"
        )
    gradient_kkt = finite("gradient_projection_kkt_max_violation")
    if not 0.0 <= gradient_kkt <= float(tolerance):
        raise RuntimeError("v015 formal result has invalid pre-optimizer projection KKT telemetry")
    candidate_l2 = finite("optimizer_candidate_actor_delta_l2")
    committed_l2 = finite("committed_actor_delta_l2")
    if candidate_l2 < 0.0 or committed_l2 < 0.0:
        raise RuntimeError("v015 formal result has negative Actor delta norm telemetry")
    state_preserved = diagnostics.get("actor_optimizer_state_restored")
    if not isinstance(state_preserved, bool):
        raise RuntimeError("v015 formal result is missing Actor optimizer-state authority telemetry")
    actor_loss_weight = finite("actor_loss_weight")
    must_preserve = actor_loss_weight == 0.0 or projection_status in {
        "NO_EMPIRICAL_DIRECTION",
        "NO_COMMON_FIRST_ORDER_DESCENT",
    }
    if state_preserved != must_preserve:
        raise RuntimeError("v015 formal result Actor optimizer-state preservation disagrees with projection authority")
    if must_preserve and committed_l2 != 0.0:
        raise RuntimeError("v015 formal result committed an Actor delta while Actor authority was frozen")
    if active_families and not must_preserve and committed_l2 == 0.0:
        raise RuntimeError("v015 formal result lost a permitted nonzero Actor update")

    return FrontRESV004ActualUpdateTelemetry(
        projection_status=projection_status,
        actual_projection_status=actual_status,
        active_families=active_families,
        directional_derivatives=derivatives,
        kkt_max_violation=kkt,
        gradient_kkt_max_violation=gradient_kkt,
        optimizer_candidate_actor_delta_l2=candidate_l2,
        committed_actor_delta_l2=committed_l2,
        actor_optimizer_state_preserved=state_preserved,
        actor_loss_weight=actor_loss_weight,
    )

