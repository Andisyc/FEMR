from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Any, Iterable

import torch

try:
    from rsl_rl.frontres.frontres_segment_cache_extractor import extract_robot_rollout_state
    from rsl_rl.frontres.frontres_segment_cache_schema import (
        FrontRESNoisyVariant,
        FrontRESPerturbationDescriptor,
        FrontRESRobotRolloutState,
        FrontRESSegmentIndex,
    )
except ModuleNotFoundError:
    _ROOT = Path(__file__).resolve().parent
    _SCHEMA_SPEC = importlib.util.spec_from_file_location(
        "frontres_segment_cache_schema", _ROOT / "frontres_segment_cache_schema.py"
    )
    _EXTRACTOR_SPEC = importlib.util.spec_from_file_location(
        "frontres_segment_cache_extractor", _ROOT / "frontres_segment_cache_extractor.py"
    )
    if _SCHEMA_SPEC is None or _SCHEMA_SPEC.loader is None or _EXTRACTOR_SPEC is None or _EXTRACTOR_SPEC.loader is None:
        raise
    _SCHEMA_MODULE = importlib.util.module_from_spec(_SCHEMA_SPEC)
    sys.modules[_SCHEMA_SPEC.name] = _SCHEMA_MODULE
    _SCHEMA_SPEC.loader.exec_module(_SCHEMA_MODULE)
    _EXTRACTOR_MODULE = importlib.util.module_from_spec(_EXTRACTOR_SPEC)
    sys.modules[_EXTRACTOR_SPEC.name] = _EXTRACTOR_MODULE
    _EXTRACTOR_SPEC.loader.exec_module(_EXTRACTOR_MODULE)
    extract_robot_rollout_state = _EXTRACTOR_MODULE.extract_robot_rollout_state
    FrontRESNoisyVariant = _SCHEMA_MODULE.FrontRESNoisyVariant
    FrontRESPerturbationDescriptor = _SCHEMA_MODULE.FrontRESPerturbationDescriptor
    FrontRESRobotRolloutState = _SCHEMA_MODULE.FrontRESRobotRolloutState
    FrontRESSegmentIndex = _SCHEMA_MODULE.FrontRESSegmentIndex


@dataclass(frozen=True)
class FrontRESNoisyBaselineResult:
    score: torch.Tensor
    fall: torch.Tensor
    rollout_len: torch.Tensor

    def validate(self, batch: int) -> None:
        _require_shape("baseline.score", self.score, (batch,))
        _require_shape("baseline.fall", self.fall, (batch,))
        _require_shape("baseline.rollout_len", self.rollout_len, (batch,))
        if self.score.requires_grad or self.fall.requires_grad or self.rollout_len.requires_grad:
            raise ValueError("baseline tensors must be detached cache data")
        if not bool(torch.isfinite(self.score).all().item()):
            raise ValueError("baseline score contains non-finite values")


@dataclass(frozen=True)
class FrontRESNoisyCaptureResult:
    variant: FrontRESNoisyVariant
    reset_success: torch.Tensor
    perturbation_success: torch.Tensor
    baseline: FrontRESNoisyBaselineResult

    def validate(self) -> None:
        self.variant.validate()
        batch = self.variant.noisy_state.batch_size
        _require_shape("reset_success", self.reset_success, (batch,))
        _require_shape("perturbation_success", self.perturbation_success, (batch,))
        self.baseline.validate(batch)

    def probe(self) -> dict[str, Any]:
        self.validate()
        variant_probe = self.variant.probe()
        return {
            "segment_id": self.variant.segment_id,
            "perturbation_id": self.variant.perturbation_id,
            "reset_success_count": int(self.reset_success.bool().sum().item()),
            "perturbation_success_count": int(self.perturbation_success.bool().sum().item()),
            "baseline_score_mean": float(self.baseline.score.float().mean().item()),
            "baseline_fall_count": int(self.baseline.fall.bool().sum().item()),
            "rollout_len_mean": float(self.baseline.rollout_len.float().mean().item()),
            "noisy_root_pos_shape": variant_probe["noisy_state.root_pos_shape"],
            "noisy_body_pos_shape": variant_probe["noisy_state.body_pos_shape"],
            "noisy_requires_grad": variant_probe["noisy_state.requires_grad"],
        }


def capture_noisy_variant(
    env: Any,
    *,
    segment: FrontRESSegmentIndex,
    clean_state: FrontRESRobotRolloutState,
    descriptor: FrontRESPerturbationDescriptor,
    env_ids: Iterable[int] | torch.Tensor | None = None,
    robot_name: str = "robot",
) -> FrontRESNoisyCaptureResult:
    segment.validate()
    clean_state.validate(name="clean_state")
    descriptor.validate()
    if int(descriptor.segment_id) != int(segment.segment_id):
        raise ValueError(
            f"descriptor segment_id={descriptor.segment_id} does not match segment_id={segment.segment_id}"
        )
    ids = _normalize_env_ids(env_ids, batch=clean_state.batch_size, device=clean_state.root_pos.device)
    reset_success = apply_clean_state_reset(env, clean_state=clean_state, env_ids=ids)
    perturbation_success = apply_segment_perturbation(env, descriptor=descriptor, env_ids=ids)
    noisy_state = extract_robot_rollout_state(env, env_ids=ids, robot_name=robot_name)
    baseline = rollout_noisy_baseline(env, segment=segment, descriptor=descriptor, env_ids=ids)
    variant = FrontRESNoisyVariant(
        segment=segment,
        descriptor=descriptor,
        noisy_state=noisy_state,
        noisy_baseline_score=baseline.score,
        noisy_fall=baseline.fall,
        noisy_rollout_len=baseline.rollout_len,
    )
    result = FrontRESNoisyCaptureResult(
        variant=variant,
        reset_success=reset_success,
        perturbation_success=perturbation_success,
        baseline=baseline,
    )
    result.validate()
    return result


def apply_clean_state_reset(
    env: Any,
    *,
    clean_state: FrontRESRobotRolloutState,
    env_ids: torch.Tensor,
) -> torch.Tensor:
    for name in ("apply_frontres_clean_state_reset", "set_frontres_rollout_state", "reset_to_frontres_rollout_state"):
        if hasattr(env, name):
            result = getattr(env, name)(clean_state=clean_state, env_ids=env_ids)
            return _success_tensor(result, env_ids)
    raise AttributeError(
        "env must define apply_frontres_clean_state_reset, set_frontres_rollout_state, "
        "or reset_to_frontres_rollout_state"
    )


def apply_segment_perturbation(
    env: Any,
    *,
    descriptor: FrontRESPerturbationDescriptor,
    env_ids: torch.Tensor,
) -> torch.Tensor:
    for name in ("apply_frontres_segment_perturbation", "apply_segment_perturbation", "apply_frontres_perturbation"):
        if hasattr(env, name):
            result = getattr(env, name)(descriptor=descriptor, env_ids=env_ids)
            return _success_tensor(result, env_ids)
    raise AttributeError(
        "env must define apply_frontres_segment_perturbation, apply_segment_perturbation, "
        "or apply_frontres_perturbation"
    )


def rollout_noisy_baseline(
    env: Any,
    *,
    segment: FrontRESSegmentIndex,
    descriptor: FrontRESPerturbationDescriptor,
    env_ids: torch.Tensor,
) -> FrontRESNoisyBaselineResult:
    for name in ("rollout_frontres_noisy_baseline", "evaluate_frontres_noisy_baseline"):
        if hasattr(env, name):
            result = getattr(env, name)(segment=segment, descriptor=descriptor, env_ids=env_ids)
            baseline = _baseline_from_mapping(result, env_ids)
            baseline.validate(int(env_ids.numel()))
            return baseline
    raise AttributeError("env must define rollout_frontres_noisy_baseline or evaluate_frontres_noisy_baseline")


def _normalize_env_ids(
    env_ids: Iterable[int] | torch.Tensor | None,
    *,
    batch: int,
    device: torch.device,
) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(batch, dtype=torch.long, device=device)
    if isinstance(env_ids, torch.Tensor):
        ids = env_ids.to(device=device, dtype=torch.long).flatten()
    else:
        ids = torch.tensor(list(env_ids), dtype=torch.long, device=device)
    if ids.numel() != batch:
        raise ValueError(f"env_ids count {ids.numel()} must match clean_state batch {batch}")
    return ids


def _success_tensor(result: Any, env_ids: torch.Tensor) -> torch.Tensor:
    if result is None:
        return torch.ones(env_ids.numel(), dtype=torch.bool, device=env_ids.device)
    if isinstance(result, torch.Tensor):
        return result.to(device=env_ids.device).bool().flatten()
    if isinstance(result, dict):
        for name in ("success", "success_mask", "reset_success", "perturbation_success"):
            if name in result:
                return result[name].to(device=env_ids.device).bool().flatten()
    raise TypeError(f"unsupported hook result type: {type(result)!r}")


def _baseline_from_mapping(result: Any, env_ids: torch.Tensor) -> FrontRESNoisyBaselineResult:
    if isinstance(result, FrontRESNoisyBaselineResult):
        return result
    if not isinstance(result, dict):
        raise TypeError(f"baseline hook must return mapping or FrontRESNoisyBaselineResult, got {type(result)!r}")
    device = env_ids.device
    return FrontRESNoisyBaselineResult(
        score=result["score"].to(device=device).float().detach().clone(),
        fall=result["fall"].to(device=device).float().detach().clone(),
        rollout_len=result["rollout_len"].to(device=device).float().detach().clone(),
    )


def _require_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
