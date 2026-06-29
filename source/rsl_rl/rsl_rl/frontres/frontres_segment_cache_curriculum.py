from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Any, Sequence

try:
    from rsl_rl.frontres.frontres_dr_curriculum import (
        allowed_perturbation_bases,
        sample_per_env_dr_strength,
        sample_perturbation_mix,
    )
except ModuleNotFoundError:
    _CURRICULUM_PATH = Path(__file__).with_name("frontres_dr_curriculum.py")
    _CURRICULUM_SPEC = importlib.util.spec_from_file_location(
        "frontres_dr_curriculum",
        _CURRICULUM_PATH,
    )
    if _CURRICULUM_SPEC is None or _CURRICULUM_SPEC.loader is None:
        raise
    _CURRICULUM_MODULE = importlib.util.module_from_spec(_CURRICULUM_SPEC)
    sys.modules[_CURRICULUM_SPEC.name] = _CURRICULUM_MODULE
    _CURRICULUM_SPEC.loader.exec_module(_CURRICULUM_MODULE)
    allowed_perturbation_bases = _CURRICULUM_MODULE.allowed_perturbation_bases
    sample_per_env_dr_strength = _CURRICULUM_MODULE.sample_per_env_dr_strength
    sample_perturbation_mix = _CURRICULUM_MODULE.sample_perturbation_mix


MIX_CLASS_NAMES = ("easy", "frontier", "hard")


@dataclass(frozen=True)
class FrontRESStage1CurriculumBankConfig:
    frontier_scale: float
    dr_min: float
    dr_max: float
    n_train: int
    progress: float
    seq_idx: int
    active_dims: Sequence[int] | None = None
    boundary_stats: dict[str, float] | None = None
    include_hard_as_train: bool = False

    def validate(self) -> None:
        if float(self.frontier_scale) < 0.0:
            raise ValueError(f"frontier_scale must be non-negative, got {self.frontier_scale}")
        if float(self.dr_min) < 0.0:
            raise ValueError(f"dr_min must be non-negative, got {self.dr_min}")
        if float(self.dr_max) < float(self.dr_min):
            raise ValueError(f"dr_max must be >= dr_min, got {self.dr_max} < {self.dr_min}")
        if int(self.n_train) <= 0:
            raise ValueError(f"n_train must be positive, got {self.n_train}")
        if not (0.0 <= float(self.progress) <= 1.0):
            raise ValueError(f"progress must be in [0, 1], got {self.progress}")


@dataclass(frozen=True)
class FrontRESStage1CurriculumBankRecord:
    bank_id: int
    family_group: tuple[str, ...]
    mix_class: str
    mix_class_index: int
    frontier_scale: float
    dr_factor: float
    actual_dr_scale: float
    role: str
    seq_idx: int
    env_slot: int

    def validate(self) -> None:
        if int(self.bank_id) < 0:
            raise ValueError(f"bank_id must be non-negative, got {self.bank_id}")
        if not self.family_group:
            raise ValueError("family_group must be non-empty")
        if self.mix_class not in MIX_CLASS_NAMES and self.mix_class != "fixed":
            raise ValueError(f"invalid mix_class {self.mix_class!r}")
        if int(self.mix_class_index) not in (-1, 0, 1, 2):
            raise ValueError(f"invalid mix_class_index {self.mix_class_index}")
        if float(self.frontier_scale) < 0.0:
            raise ValueError(f"frontier_scale must be non-negative, got {self.frontier_scale}")
        if float(self.actual_dr_scale) < 0.0:
            raise ValueError(f"actual_dr_scale must be non-negative, got {self.actual_dr_scale}")
        if float(self.dr_factor) < 0.0:
            raise ValueError(f"dr_factor must be non-negative, got {self.dr_factor}")
        if self.role not in {"train", "boundary_diagnostic"}:
            raise ValueError(f"invalid role {self.role!r}")


@dataclass(frozen=True)
class FrontRESStage1CurriculumBank:
    records: tuple[FrontRESStage1CurriculumBankRecord, ...]
    allowed_bases: tuple[str, ...]
    active_modes: tuple[str, ...]
    complexity: str
    mix_diag: dict[str, float]

    def validate(self) -> None:
        if not self.records:
            raise ValueError("records must be non-empty")
        if not self.allowed_bases:
            raise ValueError("allowed_bases must be non-empty")
        for record in self.records:
            record.validate()


def build_stage1_curriculum_bank(
    cfg: Any,
    bank_cfg: FrontRESStage1CurriculumBankConfig,
) -> FrontRESStage1CurriculumBank:
    bank_cfg.validate()
    allowed_bases = allowed_perturbation_bases(bank_cfg.active_dims)
    mix_plan = sample_perturbation_mix(
        cfg,
        bank_cfg.active_dims,
        float(bank_cfg.progress),
        int(bank_cfg.seq_idx),
        int(bank_cfg.n_train),
        boundary_stats=bank_cfg.boundary_stats,
        is_frontres=True,
    )
    strength_plan = sample_per_env_dr_strength(
        cfg,
        float(bank_cfg.frontier_scale),
        True,
        int(bank_cfg.seq_idx),
        n_train=int(bank_cfg.n_train),
        n_candidate=0,
        n_base=0,
        num_envs=int(bank_cfg.n_train),
        dr_min=float(bank_cfg.dr_min),
        dr_max=float(bank_cfg.dr_max),
    )
    if strength_plan.scale_vector is None or strength_plan.mix_class is None:
        scale_vector = [float(bank_cfg.frontier_scale)] * int(bank_cfg.n_train)
        mix_classes = [-1] * int(bank_cfg.n_train)
        mix_diag = {"easy": 0.0, "frontier": 1.0, "hard": 0.0, "mean": float(bank_cfg.frontier_scale)}
    else:
        scale_vector = list(strength_plan.scale_vector[: int(bank_cfg.n_train)])
        mix_classes = list(strength_plan.mix_class[: int(bank_cfg.n_train)])
        mix_diag = dict(strength_plan.diag)

    records: list[FrontRESStage1CurriculumBankRecord] = []
    for env_slot, (family_group, mix_class_index, actual_scale) in enumerate(
        zip(mix_plan.groups, mix_classes, scale_vector)
    ):
        mix_class = _mix_class_name(mix_class_index)
        role = "boundary_diagnostic" if mix_class == "hard" and not bank_cfg.include_hard_as_train else "train"
        record = FrontRESStage1CurriculumBankRecord(
            bank_id=len(records),
            family_group=tuple(family_group),
            mix_class=mix_class,
            mix_class_index=int(mix_class_index),
            frontier_scale=float(bank_cfg.frontier_scale),
            dr_factor=_dr_factor(float(actual_scale), float(bank_cfg.frontier_scale)),
            actual_dr_scale=float(actual_scale),
            role=role,
            seq_idx=int(bank_cfg.seq_idx),
            env_slot=int(env_slot),
        )
        record.validate()
        records.append(record)

    bank = FrontRESStage1CurriculumBank(
        records=tuple(records),
        allowed_bases=tuple(allowed_bases),
        active_modes=tuple(mix_plan.active_modes),
        complexity=str(mix_plan.complexity),
        mix_diag=mix_diag,
    )
    bank.validate()
    return bank


def stage1_curriculum_bank_probe(bank: FrontRESStage1CurriculumBank) -> dict[str, Any]:
    bank.validate()
    return {
        "record_count": len(bank.records),
        "allowed_bases": bank.allowed_bases,
        "active_modes": bank.active_modes,
        "complexity": bank.complexity,
        "mix_diag": dict(bank.mix_diag),
        "family_groups": [record.family_group for record in bank.records],
        "mix_classes": [record.mix_class for record in bank.records],
        "actual_dr_scales": [record.actual_dr_scale for record in bank.records],
        "dr_factors": [record.dr_factor for record in bank.records],
        "roles": [record.role for record in bank.records],
        "env_slots": [record.env_slot for record in bank.records],
    }


def _mix_class_name(mix_class_index: int) -> str:
    idx = int(mix_class_index)
    if idx < 0:
        return "fixed"
    if idx >= len(MIX_CLASS_NAMES):
        raise ValueError(f"invalid mix_class_index {mix_class_index}")
    return MIX_CLASS_NAMES[idx]


def _dr_factor(actual_scale: float, frontier_scale: float) -> float:
    if abs(float(frontier_scale)) <= 1.0e-8:
        return 0.0
    return float(actual_scale) / float(frontier_scale)
