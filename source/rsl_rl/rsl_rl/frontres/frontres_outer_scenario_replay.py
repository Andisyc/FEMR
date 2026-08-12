"""Committed outer prioritized replay for sealed FrontRES local Scenarios."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import math
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch

from rsl_rl.frontres.frontres_segment_warmup import (
    FRONTRES_V013_DR_CLASS_NAMES,
    FRONTRES_V013_DR_CLASS_WEIGHTS,
    FrontRESKStageIdentity,
    frontres_v021_dr_strength_in_class,
    sample_frontres_v013_dr_strength,
)

FRONTRES_OUTER_REPLAY_SCHEMA = "frontres-outer-scenario-replay-v4"
FRONTRES_OUTER_REPLAY_EMA_DECAY = 0.8
FRONTRES_OUTER_REPLAY_SCENARIO_BATCH = 8
FRONTRES_OUTER_REPLAY_CAPACITY_LADDER = (64, 128, 256)
FRONTRES_OUTER_REPLAY_MIN_VISITS = 4
FRONTRES_REPLAY_POLICY_SYMMETRIC_KL_LIMIT = 0.02
FRONTRES_REPLAY_WINDOW_MAX_VISITS = 32
FRONTRES_REPLAY_WINSOR_FRACTION = 0.2
FRONTRES_REPLAY_CONFIDENCE_Z = 1.96


def _require_nonempty(name: str, value: str) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def frontres_tensor_identity(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        if not isinstance(value, torch.Tensor):
            raise TypeError("tensor identity accepts tensors only")
        data = value.detach().to(device="cpu").contiguous()
        digest.update(repr((tuple(data.shape), str(data.dtype))).encode("utf-8"))
        digest.update(b"\0")
        digest.update(data.numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


@contextmanager
def isolated_frontres_perturbation_rng(seed: int, *, device: torch.device | str) -> Iterator[None]:
    """Seed materialization without advancing Actor/global training RNG state."""

    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("perturbation seed must be a nonnegative integer")
    resolved = torch.device(device)
    devices: list[int] = []
    if resolved.type == "cuda":
        devices = [resolved.index if resolved.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.random.default_generator.manual_seed(int(seed))
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(int(seed))
        yield


@dataclass(frozen=True)
class FrontRESScenarioKey:
    motion_id: str
    start_frame: int
    segment_id: int
    x_t_identity: str
    perturbation_family: str
    perturbation_strength: float
    perturbation_seed: int
    noisy_segment_hash: str
    horizon_k: int
    future_intent_identity: str
    planned_support_identity: str

    def validate(self) -> None:
        for name in (
            "motion_id",
            "x_t_identity",
            "perturbation_family",
            "noisy_segment_hash",
            "future_intent_identity",
            "planned_support_identity",
        ):
            _require_nonempty(name, getattr(self, name))
        for name in ("start_frame", "segment_id", "perturbation_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if isinstance(self.horizon_k, bool) or int(self.horizon_k) <= 0:
            raise ValueError("horizon_k must be positive")
        if not math.isfinite(float(self.perturbation_strength)) or float(self.perturbation_strength) < 0.0:
            raise ValueError("perturbation_strength must be finite and nonnegative")

    @property
    def digest(self) -> str:
        self.validate()
        digest = hashlib.sha256()
        for value in self.to_state().values():
            digest.update(repr(value).encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def to_state(self) -> dict[str, str | int | float]:
        self.validate()
        return {
            "motion_id": self.motion_id,
            "start_frame": int(self.start_frame),
            "segment_id": int(self.segment_id),
            "x_t_identity": self.x_t_identity,
            "perturbation_family": self.perturbation_family,
            "perturbation_strength": float(self.perturbation_strength),
            "perturbation_seed": int(self.perturbation_seed),
            "noisy_segment_hash": self.noisy_segment_hash,
            "horizon_k": int(self.horizon_k),
            "future_intent_identity": self.future_intent_identity,
            "planned_support_identity": self.planned_support_identity,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "FrontRESScenarioKey":
        expected = {
            "motion_id",
            "start_frame",
            "segment_id",
            "x_t_identity",
            "perturbation_family",
            "perturbation_strength",
            "perturbation_seed",
            "noisy_segment_hash",
            "horizon_k",
            "future_intent_identity",
            "planned_support_identity",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("ScenarioKey state has incompatible fields")
        key = cls(**dict(state))
        key.validate()
        return key


@dataclass(frozen=True)
class FrontRESScenarioUtilityWindow:
    """Bounded same-Scenario evidence under one fixed policy anchor."""

    policy_anchor_mean: tuple[float, ...]
    policy_anchor_sigma: tuple[float, ...]
    utility_visits: tuple[tuple[float, ...], ...]
    reset_count: int = 0

    def validate(self) -> None:
        for name, values in (
            ("policy_anchor_mean", self.policy_anchor_mean),
            ("policy_anchor_sigma", self.policy_anchor_sigma),
        ):
            if len(values) != 6 or any(not math.isfinite(float(value)) for value in values):
                raise ValueError(f"{name} must contain six finite values")
        if any(float(value) <= 0.0 for value in self.policy_anchor_sigma):
            raise ValueError("policy_anchor_sigma values must be positive")
        if not self.utility_visits or len(self.utility_visits) > FRONTRES_REPLAY_WINDOW_MAX_VISITS:
            raise ValueError("utility window must contain one to 32 visits")
        if any(
            len(visit) != 4 or any(not math.isfinite(float(value)) for value in visit)
            for visit in self.utility_visits
        ):
            raise ValueError("utility window requires complete finite M4 visits")
        if (
            isinstance(self.reset_count, bool)
            or int(self.reset_count) != self.reset_count
            or int(self.reset_count) < 0
        ):
            raise ValueError("utility window reset_count must be nonnegative")

    @property
    def sample_count(self) -> int:
        return sum(len(visit) for visit in self.utility_visits)

    @property
    def compatible_visit_count(self) -> int:
        return len(self.utility_visits)

    @property
    def robust_mean(self) -> float:
        samples = sorted(float(value) for visit in self.utility_visits for value in visit)
        trim = int(math.floor(FRONTRES_REPLAY_WINSOR_FRACTION * len(samples)))
        if trim:
            low = samples[trim]
            high = samples[-trim - 1]
            samples = [low] * trim + samples[trim : len(samples) - trim] + [high] * trim
        return sum(samples) / len(samples)

    @property
    def outcome_variance(self) -> float:
        samples = [float(value) for visit in self.utility_visits for value in visit]
        if len(samples) < 2:
            return 0.0
        mean = sum(samples) / len(samples)
        return sum((value - mean) ** 2 for value in samples) / (len(samples) - 1)

    @property
    def standard_error(self) -> float:
        return math.sqrt(self.outcome_variance / self.sample_count)

    @property
    def confidence_half_width(self) -> float:
        return FRONTRES_REPLAY_CONFIDENCE_Z * self.standard_error

    @staticmethod
    def _symmetric_kl(
        anchor_mean: tuple[float, ...],
        anchor_sigma: tuple[float, ...],
        current_mean: tuple[float, ...],
        current_sigma: tuple[float, ...],
    ) -> float:
        total = 0.0
        for mean_a, sigma_a, mean_b, sigma_b in zip(
            anchor_mean,
            anchor_sigma,
            current_mean,
            current_sigma,
            strict=True,
        ):
            var_a = float(sigma_a) ** 2
            var_b = float(sigma_b) ** 2
            delta2 = (float(mean_a) - float(mean_b)) ** 2
            kl_ab = 0.5 * (math.log(var_b / var_a) + (var_a + delta2) / var_b - 1.0)
            kl_ba = 0.5 * (math.log(var_a / var_b) + (var_b + delta2) / var_a - 1.0)
            total += 0.5 * (kl_ab + kl_ba)
        return max(total, 0.0)

    @classmethod
    def from_visit(
        cls,
        *,
        utilities: Sequence[float],
        policy_mean: Sequence[float],
        policy_sigma: Sequence[float],
        reset_count: int = 0,
    ) -> "FrontRESScenarioUtilityWindow":
        result = cls(
            policy_anchor_mean=tuple(float(value) for value in policy_mean),
            policy_anchor_sigma=tuple(float(value) for value in policy_sigma),
            utility_visits=(tuple(float(value) for value in utilities),),
            reset_count=int(reset_count),
        )
        result.validate()
        return result

    def preview_visit(
        self,
        *,
        utilities: Sequence[float],
        policy_mean: Sequence[float],
        policy_sigma: Sequence[float],
    ) -> tuple["FrontRESScenarioUtilityWindow", float, bool]:
        self.validate()
        current_mean = tuple(float(value) for value in policy_mean)
        current_sigma = tuple(float(value) for value in policy_sigma)
        visit = tuple(float(value) for value in utilities)
        probe = FrontRESScenarioUtilityWindow.from_visit(
            utilities=visit,
            policy_mean=current_mean,
            policy_sigma=current_sigma,
            reset_count=self.reset_count,
        )
        symmetric_kl = self._symmetric_kl(
            self.policy_anchor_mean,
            self.policy_anchor_sigma,
            current_mean,
            current_sigma,
        )
        if symmetric_kl > FRONTRES_REPLAY_POLICY_SYMMETRIC_KL_LIMIT:
            reset = replace(probe, reset_count=self.reset_count + 1)
            reset.validate()
            return reset, symmetric_kl, True
        visits = (*self.utility_visits, visit)[-FRONTRES_REPLAY_WINDOW_MAX_VISITS:]
        result = replace(self, utility_visits=visits)
        result.validate()
        return result, symmetric_kl, False

    def to_state(self) -> dict[str, Any]:
        self.validate()
        return {
            "policy_anchor_mean": self.policy_anchor_mean,
            "policy_anchor_sigma": self.policy_anchor_sigma,
            "utility_visits": self.utility_visits,
            "reset_count": int(self.reset_count),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "FrontRESScenarioUtilityWindow":
        expected = {"policy_anchor_mean", "policy_anchor_sigma", "utility_visits", "reset_count"}
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("Scenario utility window has incompatible fields")
        result = cls(
            policy_anchor_mean=tuple(float(value) for value in state["policy_anchor_mean"]),
            policy_anchor_sigma=tuple(float(value) for value in state["policy_anchor_sigma"]),
            utility_visits=tuple(
                tuple(float(value) for value in visit) for visit in state["utility_visits"]
            ),
            reset_count=int(state["reset_count"]),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class FrontRESScenarioReplayRecord:
    key: FrontRESScenarioKey
    dr_class: str
    critic_calibration_score_by_k: tuple[tuple[int, float], ...]
    repair_spread_score_by_k: tuple[tuple[int, float], ...]
    staleness: int
    visit_count: int
    last_transaction_id: str
    utility_window: FrontRESScenarioUtilityWindow

    def validate(self) -> None:
        self.key.validate()
        _require_nonempty("dr_class", self.dr_class)
        _require_nonempty("last_transaction_id", self.last_transaction_id)
        if not isinstance(self.utility_window, FrontRESScenarioUtilityWindow):
            raise TypeError("Scenario replay record requires one utility window")
        self.utility_window.validate()
        if (
            isinstance(self.staleness, bool)
            or int(self.staleness) != self.staleness
            or int(self.staleness) < 0
        ):
            raise ValueError("staleness must be nonnegative")
        if (
            isinstance(self.visit_count, bool)
            or int(self.visit_count) != self.visit_count
            or int(self.visit_count) <= 0
        ):
            raise ValueError("visit_count must be positive")
        for name, values in (
            ("critic_calibration_score_by_k", self.critic_calibration_score_by_k),
            ("repair_spread_score_by_k", self.repair_spread_score_by_k),
        ):
            keys = tuple(int(k) for k, _value in values)
            if keys != tuple(sorted(set(keys))) or any(k <= 0 for k in keys):
                raise ValueError(f"{name} must use ordered unique positive K values")
            if any(not math.isfinite(float(value)) or float(value) < 0.0 for _k, value in values):
                raise ValueError(f"{name} values must be finite and nonnegative")
        if tuple(k for k, _ in self.critic_calibration_score_by_k) != tuple(
            k for k, _ in self.repair_spread_score_by_k
        ):
            raise ValueError("TRAIN-v023 replay score maps must contain identical K identities")

    def score_for_k(self, horizon_k: int, *, score_kind: str = "critic_calibration") -> float | None:
        if score_kind == "critic_calibration":
            values = self.critic_calibration_score_by_k
        elif score_kind == "repair_spread":
            values = self.repair_spread_score_by_k
        else:
            raise ValueError(f"unknown TRAIN-v023 replay score kind {score_kind!r}")
        return dict(values).get(int(horizon_k))

    def with_visit(
        self,
        *,
        horizon_k: int,
        critic_calibration_value: float,
        repair_spread_value: float,
        utility_window: FrontRESScenarioUtilityWindow,
        transaction_id: str,
    ) -> "FrontRESScenarioReplayRecord":
        self.validate()
        values = (float(critic_calibration_value), float(repair_spread_value))
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("TRAIN-v023 replay values must be finite and nonnegative")
        calibration_scores = dict(self.critic_calibration_score_by_k)
        repair_scores = dict(self.repair_spread_score_by_k)
        calibration_scores[int(horizon_k)] = float(critic_calibration_value)
        previous_repair = repair_scores.get(int(horizon_k))
        repair_scores[int(horizon_k)] = (
            float(repair_spread_value)
            if previous_repair is None
            else FRONTRES_OUTER_REPLAY_EMA_DECAY * float(previous_repair)
            + (1.0 - FRONTRES_OUTER_REPLAY_EMA_DECAY) * float(repair_spread_value)
        )
        result = replace(
            self,
            critic_calibration_score_by_k=tuple(sorted(calibration_scores.items())),
            repair_spread_score_by_k=tuple(sorted(repair_scores.items())),
            staleness=0,
            visit_count=int(self.visit_count) + 1,
            last_transaction_id=_require_nonempty("transaction_id", transaction_id),
            utility_window=utility_window,
        )
        result.validate()
        return result

    def to_state(self) -> dict[str, Any]:
        self.validate()
        return {
            "key": self.key.to_state(),
            "dr_class": self.dr_class,
            "critic_calibration_score_by_k": tuple(
                (int(k), float(value)) for k, value in self.critic_calibration_score_by_k
            ),
            "repair_spread_score_by_k": tuple(
                (int(k), float(value)) for k, value in self.repair_spread_score_by_k
            ),
            "staleness": int(self.staleness),
            "visit_count": int(self.visit_count),
            "last_transaction_id": self.last_transaction_id,
            "utility_window": self.utility_window.to_state(),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "FrontRESScenarioReplayRecord":
        expected = {
            "key",
            "dr_class",
            "critic_calibration_score_by_k",
            "repair_spread_score_by_k",
            "staleness",
            "visit_count",
            "last_transaction_id",
            "utility_window",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("Scenario replay record has incompatible fields")
        result = cls(
            key=FrontRESScenarioKey.from_state(state["key"]),
            dr_class=str(state["dr_class"]),
            critic_calibration_score_by_k=tuple(
                (int(k), float(value)) for k, value in state["critic_calibration_score_by_k"]
            ),
            repair_spread_score_by_k=tuple(
                (int(k), float(value)) for k, value in state["repair_spread_score_by_k"]
            ),
            staleness=int(state["staleness"]),
            visit_count=int(state["visit_count"]),
            last_transaction_id=str(state["last_transaction_id"]),
            utility_window=FrontRESScenarioUtilityWindow.from_state(state["utility_window"]),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class FrontRESOuterReplaySelection:
    source: str
    segment_id: int
    perturbation_seed: int
    perturbation_family: str
    perturbation_strength: float
    dr_class: str
    purpose: str
    replay_key_digest: str | None
    score: float
    staleness: int

    def validate(self) -> None:
        if self.source not in {"global", "replay", "review"}:
            raise ValueError(f"unsupported Scenario source {self.source!r}")
        if self.purpose not in {"admission", "critic_calibration", "repair_spread", "stale_review"}:
            raise ValueError(f"unsupported Replay Curriculum purpose {self.purpose!r}")
        if int(self.segment_id) < 0 or int(self.perturbation_seed) < 0:
            raise ValueError("selection segment_id and perturbation_seed must be nonnegative")
        _require_nonempty("perturbation_family", self.perturbation_family)
        _require_nonempty("dr_class", self.dr_class)
        if not math.isfinite(float(self.perturbation_strength)) or float(self.perturbation_strength) < 0.0:
            raise ValueError("selection strength must be finite and nonnegative")
        if not math.isfinite(float(self.score)) or float(self.score) < 0.0 or int(self.staleness) < 0:
            raise ValueError("selection score/staleness must be nonnegative")
        if self.source == "global" and self.replay_key_digest is not None:
            raise ValueError("global selection cannot claim an existing replay key")
        if self.source != "global" and not self.replay_key_digest:
            raise ValueError("replay/review selection requires a stable key digest")


@dataclass(frozen=True)
class FrontRESOuterReplayPlan:
    transaction_id: str
    curriculum: FrontRESKStageIdentity
    selections: tuple[FrontRESOuterReplaySelection, ...]
    generator_state_before: torch.Tensor
    generator_state_after: torch.Tensor
    record_state_digest: str
    active_capacity_before: int
    active_capacity_after: int

    @property
    def active_k(self) -> int:
        return int(self.curriculum.active_k)

    @property
    def phase_name(self) -> str:
        return str(self.curriculum.phase.name)

    @property
    def score_kind(self) -> str:
        return "repair_spread" if self.phase_name == "joint" else "critic_calibration"

    def validate(self) -> None:
        _require_nonempty("transaction_id", self.transaction_id)
        if not isinstance(self.curriculum, FrontRESKStageIdentity):
            raise TypeError("outer replay plan requires one immutable TRAIN-v023 curriculum identity")
        if self.phase_name not in {"low_dr_joint_init", "coupled_ramp", "joint"}:
            raise ValueError("outer replay plan has an incompatible TRAIN-v023 phase")
        if int(self.active_k) <= 0 or len(self.selections) != FRONTRES_OUTER_REPLAY_SCENARIO_BATCH:
            raise ValueError("outer replay plan requires positive K and exactly eight selections")
        for selection in self.selections:
            selection.validate()
            if not frontres_v021_dr_strength_in_class(
                self.curriculum,
                class_name=selection.dr_class,
                strength=selection.perturbation_strength,
            ):
                raise ValueError("outer replay selection lies outside its current absolute DR interval")
        if len({selection.segment_id for selection in self.selections}) != FRONTRES_OUTER_REPLAY_SCENARIO_BATCH:
            raise ValueError("outer replay plan requires eight distinct Scenario sources")
        expected_purposes = (
            ("admission",) + ("critic_calibration",) * 6 + ("stale_review",)
            if self.phase_name != "joint"
            else ("admission",) + ("repair_spread",) * 4 + ("critic_calibration",) * 2 + ("stale_review",)
        )
        if tuple(selection.purpose for selection in self.selections) != expected_purposes:
            raise ValueError("outer replay plan violates the phase-specific B8 slot curriculum")
        if self.active_capacity_before <= 0 or self.active_capacity_after <= 0:
            raise ValueError("outer replay plan has an invalid active capacity")
        if self.active_capacity_after < self.active_capacity_before:
            raise ValueError("outer replay plan cannot shrink active capacity")
        if self.active_capacity_after > self.active_capacity_before and self.phase_name != "joint":
            raise ValueError("outer replay capacity cannot expand while DR is adapting")
        for value in (self.generator_state_before, self.generator_state_after):
            if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
                raise ValueError("outer replay plan requires uint8 generator states")
        _require_nonempty("record_state_digest", self.record_state_digest)


@dataclass(frozen=True)
class FrontRESOuterReplayCandidate:
    transaction_id: str
    policy_snapshot_id: str
    plan: FrontRESOuterReplayPlan
    records: tuple[FrontRESScenarioReplayRecord, ...]
    critic_calibration_values: tuple[float, ...]
    repair_spread_values: tuple[float, ...]
    critic_target_means: tuple[float, ...]
    current_utility_means: tuple[float, ...]
    outcome_variances: tuple[float, ...]
    standard_errors: tuple[float, ...]
    confidence_half_widths: tuple[float, ...]
    compatible_sample_counts: tuple[int, ...]
    compatible_visit_counts: tuple[int, ...]
    policy_symmetric_kls: tuple[float, ...]
    policy_window_resets: tuple[bool, ...]
    active_digests_by_k: tuple[tuple[int, tuple[str, ...]], ...]

    def validate(self) -> None:
        self.plan.validate()
        if self.transaction_id != self.plan.transaction_id:
            raise ValueError("candidate transaction differs from its selection plan")
        _require_nonempty("policy_snapshot_id", self.policy_snapshot_id)
        for name, values in (
            ("critic_calibration_values", self.critic_calibration_values),
            ("repair_spread_values", self.repair_spread_values),
            ("outcome_variances", self.outcome_variances),
            ("standard_errors", self.standard_errors),
            ("confidence_half_widths", self.confidence_half_widths),
            ("policy_symmetric_kls", self.policy_symmetric_kls),
        ):
            if len(values) != FRONTRES_OUTER_REPLAY_SCENARIO_BATCH or any(
                not math.isfinite(float(value)) or float(value) < 0.0 for value in values
            ):
                raise ValueError(f"candidate requires eight finite nonnegative {name}")
        if len(self.critic_target_means) != FRONTRES_OUTER_REPLAY_SCENARIO_BATCH or any(
            not math.isfinite(float(value)) for value in self.critic_target_means
        ):
            raise ValueError("candidate requires eight finite critic_target_means")
        if len(self.current_utility_means) != FRONTRES_OUTER_REPLAY_SCENARIO_BATCH or any(
            not math.isfinite(float(value)) for value in self.current_utility_means
        ):
            raise ValueError("candidate requires eight finite current_utility_means")
        for name, values in (
            ("compatible_sample_counts", self.compatible_sample_counts),
            ("compatible_visit_counts", self.compatible_visit_counts),
        ):
            if len(values) != FRONTRES_OUTER_REPLAY_SCENARIO_BATCH or any(
                isinstance(value, bool) or int(value) != value or int(value) <= 0 for value in values
            ):
                raise ValueError(f"candidate requires eight positive {name}")
        if (
            len(self.policy_window_resets) != FRONTRES_OUTER_REPLAY_SCENARIO_BATCH
            or any(not isinstance(value, bool) for value in self.policy_window_resets)
        ):
            raise ValueError("candidate requires eight policy_window_resets")
        if any(
            samples != 4 * visits
            for samples, visits in zip(
                self.compatible_sample_counts,
                self.compatible_visit_counts,
                strict=True,
            )
        ):
            raise ValueError("candidate compatible counts must contain complete M4 visits")
        digests = []
        for record in self.records:
            record.validate()
            digests.append(record.key.digest)
        if len(digests) != len(set(digests)):
            raise ValueError("candidate contains duplicate Scenario records")
        for active_k, active_digests in self.active_digests_by_k:
            if int(active_k) <= 0 or len(active_digests) != len(set(active_digests)):
                raise ValueError("candidate active membership is invalid")
            if any(digest not in set(digests) for digest in active_digests):
                raise ValueError("candidate active membership references a missing archive record")


class FrontRESOuterScenarioReplay:
    """Preview selection, stage evidence, then commit all replay state at once."""

    def __init__(
        self,
        *,
        global_frac: float = 0.4,
        replay_frac: float = 0.5,
        review_frac: float = 0.1,
        min_replay_score: float = 0.05,
        staleness_weight: float = 0.1,
        capacity_ladder: Sequence[int] = FRONTRES_OUTER_REPLAY_CAPACITY_LADDER,
        minimum_visits_before_expand: int = FRONTRES_OUTER_REPLAY_MIN_VISITS,
        seed: int = 0,
    ) -> None:
        fractions = (float(global_frac), float(replay_frac), float(review_frac))
        if min(fractions) < 0.0 or sum(fractions) <= 0.0:
            raise ValueError("outer replay fractions must be nonnegative with positive mass")
        total = sum(fractions)
        self.global_frac, self.replay_frac, self.review_frac = tuple(value / total for value in fractions)
        self.min_replay_score = float(min_replay_score)
        self.staleness_weight = float(staleness_weight)
        ladder = tuple(int(value) for value in capacity_ladder)
        if not ladder or any(value <= 0 for value in ladder) or tuple(sorted(set(ladder))) != ladder:
            raise ValueError("Replay Curriculum capacity ladder must be positive and strictly increasing")
        if int(minimum_visits_before_expand) <= 0:
            raise ValueError("Replay Curriculum minimum visits must be positive")
        self.capacity_ladder = ladder
        self.minimum_visits_before_expand = int(minimum_visits_before_expand)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        self._records: dict[str, FrontRESScenarioReplayRecord] = {}
        self._active_by_k: dict[int, set[str]] = {}
        self._capacity_by_k: dict[int, int] = {}
        self._last_commit: dict[str, Any] | None = None

    @property
    def records(self) -> tuple[FrontRESScenarioReplayRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def _record_digest(self) -> str:
        digest = hashlib.sha256()
        for record in self.records:
            digest.update(repr(record.to_state()).encode("utf-8"))
            digest.update(b"\0")
        digest.update(repr(tuple((k, tuple(sorted(v))) for k, v in sorted(self._active_by_k.items()))).encode("utf-8"))
        digest.update(repr(tuple(sorted(self._capacity_by_k.items()))).encode("utf-8"))
        return digest.hexdigest()

    def _capacity_for_k(self, active_k: int) -> int:
        return int(self._capacity_by_k.get(int(active_k), self.capacity_ladder[0]))

    def _active_records(self, active_k: int) -> tuple[FrontRESScenarioReplayRecord, ...]:
        digests = self._active_by_k.get(int(active_k), set())
        return tuple(self._records[digest] for digest in sorted(digests) if digest in self._records)

    @staticmethod
    def _slot_purposes(phase_name: str) -> tuple[str, ...]:
        if phase_name == "joint":
            return ("admission",) + ("repair_spread",) * 4 + ("critic_calibration",) * 2 + ("stale_review",)
        return ("admission",) + ("critic_calibration",) * 6 + ("stale_review",)

    def _preview_capacity(self, curriculum: FrontRESKStageIdentity) -> tuple[int, int]:
        current = self._capacity_for_k(curriculum.active_k)
        active = self._active_records(curriculum.active_k)
        if curriculum.phase.name != "joint" or len(active) < current:
            return current, current
        if any(
            record.utility_window.compatible_visit_count < self.minimum_visits_before_expand
            for record in active
        ):
            return current, current
        index = self.capacity_ladder.index(current)
        return current, self.capacity_ladder[min(index + 1, len(self.capacity_ladder) - 1)]

    @staticmethod
    def _quota_counts(capacity: int) -> dict[str, int]:
        exact = [float(capacity) * weight for weight in FRONTRES_V013_DR_CLASS_WEIGHTS]
        counts = [int(math.floor(value)) for value in exact]
        remaining = int(capacity) - sum(counts)
        order = sorted(range(len(exact)), key=lambda index: (exact[index] - counts[index], -index), reverse=True)
        for index in order[:remaining]:
            counts[index] += 1
        return dict(zip(FRONTRES_V013_DR_CLASS_NAMES, counts, strict=True))

    def _admission_class(self, *, active_k: int, capacity: int) -> str:
        quotas = self._quota_counts(capacity)
        counts = {name: 0 for name in FRONTRES_V013_DR_CLASS_NAMES}
        for record in self._active_records(active_k):
            counts[record.dr_class] = counts.get(record.dr_class, 0) + 1
        return self._next_quota_class(counts=counts, capacity=capacity)

    @classmethod
    def _next_quota_class(cls, *, counts: Mapping[str, int], capacity: int) -> str:
        quotas = cls._quota_counts(capacity)
        return max(
            FRONTRES_V013_DR_CLASS_NAMES,
            key=lambda name: (quotas[name] - counts.get(name, 0), quotas[name], -FRONTRES_V013_DR_CLASS_NAMES.index(name)),
        )

    @staticmethod
    def _sample_dr_class(
        curriculum: FrontRESKStageIdentity,
        *,
        class_name: str,
        generator: torch.Generator,
    ) -> Any:
        for _ in range(4096):
            sample_key = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
            sample = sample_frontres_v013_dr_strength(curriculum, sample_key=sample_key)
            if sample.class_name == class_name:
                return sample
        raise RuntimeError("Replay Curriculum could not draw the required DR quota class")

    @staticmethod
    def _protected_anchor_digests(
        records: Sequence[FrontRESScenarioReplayRecord],
        *,
        active_k: int,
    ) -> set[str]:
        protected: set[str] = set()
        classes = {record.dr_class for record in records}
        for dr_class in classes:
            rows = [record for record in records if record.dr_class == dr_class]
            for score_kind in ("critic_calibration", "repair_spread"):
                ranked = sorted(
                    rows,
                    key=lambda record: (
                        float(record.score_for_k(active_k, score_kind=score_kind) or 0.0),
                        record.key.digest,
                    ),
                    reverse=True,
                )
                protected.update(record.key.digest for record in ranked[:2])
        return protected

    def _admission_open(self, curriculum: FrontRESKStageIdentity, *, capacity: int) -> bool:
        active = self._active_records(curriculum.active_k)
        if len(active) < int(capacity):
            return True
        protected = self._protected_anchor_digests(active, active_k=curriculum.active_k)
        return any(
            record.utility_window.compatible_visit_count >= self.minimum_visits_before_expand
            and record.key.digest not in protected
            for record in active
        )

    @staticmethod
    def _copy_generator(state: torch.Tensor) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.set_state(state.detach().cpu().clone())
        return generator

    def _choose_source(self, generator: torch.Generator) -> str:
        draw = float(torch.rand((), generator=generator).item())
        if draw < self.global_frac:
            return "global"
        if draw < self.global_frac + self.replay_frac:
            return "replay"
        return "review"

    def _pool(
        self,
        source: str,
        *,
        curriculum: FrontRESKStageIdentity,
        dr_class: str,
        score_kind: str,
        excluded_segments: set[int],
    ) -> list[FrontRESScenarioReplayRecord]:
        rows = [
            record
            for record in self._active_records(curriculum.active_k)
            if record.key.segment_id not in excluded_segments
            and record.score_for_k(curriculum.active_k, score_kind=score_kind) is not None
            and frontres_v021_dr_strength_in_class(
                curriculum,
                class_name=dr_class,
                strength=record.key.perturbation_strength,
            )
        ]
        if source == "replay":
            return [
                record
                for record in rows
                if float(record.score_for_k(curriculum.active_k, score_kind=score_kind)) >= self.min_replay_score
            ]
        if source == "review":
            return [
                record
                for record in rows
                if float(record.score_for_k(curriculum.active_k, score_kind=score_kind)) < self.min_replay_score
            ]
        return []

    def _sample_record(
        self,
        pool: Sequence[FrontRESScenarioReplayRecord],
        *,
        source: str,
        active_k: int,
        score_kind: str,
        generator: torch.Generator,
    ) -> FrontRESScenarioReplayRecord:
        ordered = sorted(
            pool,
            key=lambda record: (float(record.score_for_k(active_k, score_kind=score_kind)), record.key.digest),
        )
        count = len(ordered)
        score_rank = {record.key.digest: (index + 1) / count for index, record in enumerate(ordered)}
        max_staleness = max(1, max(record.staleness for record in ordered))
        weights = []
        for record in ordered:
            rank = score_rank[record.key.digest]
            rank_term = rank if source == "replay" else 1.0 - rank + 1.0 / count
            weights.append(rank_term + self.staleness_weight * record.staleness / max_staleness)
        position = int(torch.multinomial(torch.tensor(weights, dtype=torch.float64), 1, generator=generator).item())
        return ordered[position]

    @staticmethod
    def _sample_global_segment(
        generator: torch.Generator,
        *,
        num_segments: int,
        eligible: Callable[[int], bool],
        excluded_segments: set[int],
        seen_segments: set[int],
    ) -> int:
        if int(num_segments) <= len(excluded_segments):
            raise RuntimeError("outer replay has too few Segment identities")
        for prefer_unseen in (True, False):
            for _ in range(2048):
                segment_id = int(torch.randint(0, int(num_segments), (1,), generator=generator).item())
                if segment_id in excluded_segments or (prefer_unseen and segment_id in seen_segments):
                    continue
                if bool(eligible(segment_id)):
                    return segment_id
        raise RuntimeError("outer replay could not find an eligible global Scenario source")

    def plan(
        self,
        *,
        transaction_id: str,
        curriculum: FrontRESKStageIdentity,
        num_segments: int,
        eligible: Callable[[int], bool],
        global_family: Callable[[int], str],
    ) -> FrontRESOuterReplayPlan:
        _require_nonempty("transaction_id", transaction_id)
        if self._last_commit is not None and self._last_commit.get("transaction_id") == transaction_id:
            raise RuntimeError("outer replay rejects a previously committed transaction before planning")
        if not isinstance(curriculum, FrontRESKStageIdentity):
            raise TypeError("outer replay plan requires a TRAIN-v023 curriculum identity")
        if not callable(eligible) or not callable(global_family):
            raise TypeError("outer replay plan requires eligible and global_family callables")
        before = self.generator.get_state().detach().cpu().clone()
        generator = self._copy_generator(before)
        selected: list[FrontRESOuterReplaySelection] = []
        excluded_segments: set[int] = set()
        seen_segments = {record.key.segment_id for record in self.records}
        capacity_before, capacity_after = self._preview_capacity(curriculum)
        admission_open = self._admission_open(curriculum, capacity=capacity_after)
        active_records = self._active_records(curriculum.active_k)
        replacement_class: str | None = None
        if admission_open and len(active_records) >= capacity_after:
            protected = self._protected_anchor_digests(active_records, active_k=curriculum.active_k)
            replaceable = tuple(
                record
                for record in active_records
                if record.utility_window.compatible_visit_count >= self.minimum_visits_before_expand
                and record.key.digest not in protected
            )
            if not replaceable:
                raise RuntimeError("Replay Curriculum opened admission without a replaceable active Scenario")
            replacement_class = min(
                replaceable,
                key=lambda record: (
                    max(
                        float(record.score_for_k(curriculum.active_k, score_kind="critic_calibration") or 0.0),
                        float(record.score_for_k(curriculum.active_k, score_kind="repair_spread") or 0.0),
                    ),
                    record.key.digest,
                ),
            ).dr_class
        planned_class_counts = {name: 0 for name in FRONTRES_V013_DR_CLASS_NAMES}
        for record in self._active_records(curriculum.active_k):
            planned_class_counts[record.dr_class] = planned_class_counts.get(record.dr_class, 0) + 1
        for purpose in self._slot_purposes(curriculum.phase.name):
            if purpose == "admission":
                dr_sample = self._sample_dr_class(
                    curriculum,
                    class_name=(
                        replacement_class
                        if replacement_class is not None
                        else self._next_quota_class(counts=planned_class_counts, capacity=capacity_after)
                    ),
                    generator=generator,
                )
            else:
                dr_sample_key = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
                dr_sample = sample_frontres_v013_dr_strength(curriculum, sample_key=dr_sample_key)
            admission_gated = purpose == "admission" and not admission_open
            requested_source = (
                "global"
                if purpose == "admission" and not admission_gated
                else ("review" if purpose == "stale_review" else "replay")
            )
            score_kind = "repair_spread" if purpose == "repair_spread" else "critic_calibration"
            if admission_gated:
                pool = tuple(
                    record
                    for record in self._active_records(curriculum.active_k)
                    if record.key.segment_id not in excluded_segments
                    and record.score_for_k(curriculum.active_k, score_kind=score_kind) is not None
                    and frontres_v021_dr_strength_in_class(
                        curriculum,
                        class_name=record.dr_class,
                        strength=record.key.perturbation_strength,
                    )
                )
            else:
                pool = self._pool(
                    requested_source,
                    curriculum=curriculum,
                    dr_class=dr_sample.class_name,
                    score_kind=score_kind,
                    excluded_segments=excluded_segments,
                )
            if purpose == "stale_review" and not pool:
                pool = self._pool(
                    "replay",
                    curriculum=curriculum,
                    dr_class=dr_sample.class_name,
                    score_kind="critic_calibration",
                    excluded_segments=excluded_segments,
                )
            active_fallback = False
            if not pool and requested_source != "global" and self._active_records(curriculum.active_k):
                pool = tuple(
                    record
                    for record in self._active_records(curriculum.active_k)
                    if record.key.segment_id not in excluded_segments
                    and record.score_for_k(curriculum.active_k, score_kind=score_kind) is not None
                    and frontres_v021_dr_strength_in_class(
                        curriculum,
                        class_name=record.dr_class,
                        strength=record.key.perturbation_strength,
                    )
                )
                active_fallback = bool(pool)
            if pool:
                if purpose == "stale_review":
                    record = max(pool, key=lambda item: (item.staleness, item.key.digest))
                else:
                    record = self._sample_record(
                        pool,
                        source="replay",
                        active_k=int(curriculum.active_k),
                        score_kind=score_kind,
                        generator=generator,
                    )
                selection = FrontRESOuterReplaySelection(
                    source=requested_source,
                    segment_id=record.key.segment_id,
                    perturbation_seed=record.key.perturbation_seed,
                    perturbation_family=record.key.perturbation_family,
                    perturbation_strength=record.key.perturbation_strength,
                    dr_class=record.dr_class if admission_gated or active_fallback else dr_sample.class_name,
                    purpose=purpose,
                    replay_key_digest=record.key.digest,
                    score=float(record.score_for_k(curriculum.active_k, score_kind=score_kind)),
                    staleness=record.staleness,
                )
            else:
                if purpose != "admission":
                    quota_class = self._next_quota_class(counts=planned_class_counts, capacity=capacity_after)
                    dr_sample = self._sample_dr_class(
                        curriculum,
                        class_name=quota_class,
                        generator=generator,
                    )
                segment_id = self._sample_global_segment(
                    generator,
                    num_segments=int(num_segments),
                    eligible=eligible,
                    excluded_segments=excluded_segments,
                    seen_segments=seen_segments,
                )
                perturbation_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
                family = _require_nonempty("perturbation_family", global_family(segment_id))
                selection = FrontRESOuterReplaySelection(
                    source="global",
                    segment_id=segment_id,
                    perturbation_seed=perturbation_seed,
                    perturbation_family=family,
                    perturbation_strength=float(torch.tensor(dr_sample.strength, dtype=torch.float32).item()),
                    dr_class=str(dr_sample.class_name),
                    purpose=purpose,
                    replay_key_digest=None,
                    score=0.0,
                    staleness=0,
                )
            selection.validate()
            selected.append(selection)
            excluded_segments.add(selection.segment_id)
            if selection.source == "global":
                planned_class_counts[selection.dr_class] = planned_class_counts.get(selection.dr_class, 0) + 1
        result = FrontRESOuterReplayPlan(
            transaction_id=str(transaction_id),
            curriculum=curriculum,
            selections=tuple(selected),
            generator_state_before=before,
            generator_state_after=generator.get_state().detach().cpu().clone(),
            record_state_digest=self._record_digest(),
            active_capacity_before=capacity_before,
            active_capacity_after=capacity_after,
        )
        result.validate()
        return result

    def stage(
        self,
        plan: FrontRESOuterReplayPlan,
        *,
        keys: Sequence[FrontRESScenarioKey],
        utilities: torch.Tensor,
        old_values: torch.Tensor,
        policy_means: torch.Tensor,
        policy_sigmas: torch.Tensor,
        source_index: torch.Tensor,
        policy_snapshot_id: str,
        active_m: int,
    ) -> FrontRESOuterReplayCandidate:
        plan.validate()
        if self._last_commit is not None and self._last_commit.get("transaction_id") == plan.transaction_id:
            raise RuntimeError("outer replay rejects a previously committed transaction before staging")
        if (
            isinstance(active_m, bool)
            or int(active_m) != 4
            or int(active_m) != int(plan.curriculum.active_m)
        ):
            raise ValueError("FRS-TRAIN-v023 Replay requires the sealed exact M4 identity")
        if plan.record_state_digest != self._record_digest() or not torch.equal(
            plan.generator_state_before, self.generator.get_state().cpu()
        ):
            raise RuntimeError("outer replay state changed after selection preview")
        if len(keys) != FRONTRES_OUTER_REPLAY_SCENARIO_BATCH or any(key.horizon_k != plan.active_k for key in keys):
            raise ValueError("outer replay stage requires eight current-K ScenarioKeys")
        utility_values = torch.as_tensor(utilities, dtype=torch.float32).detach().cpu().flatten()
        value_rows = torch.as_tensor(old_values, dtype=torch.float32).detach().cpu().flatten()
        mean_rows = torch.as_tensor(policy_means, dtype=torch.float32).detach().cpu()
        sigma_rows = torch.as_tensor(policy_sigmas, dtype=torch.float32).detach().cpu()
        sources = torch.as_tensor(source_index, dtype=torch.long).detach().cpu().flatten()
        expected_rows = FRONTRES_OUTER_REPLAY_SCENARIO_BATCH * int(active_m)
        if (
            int(utility_values.numel()) != expected_rows
            or int(value_rows.numel()) != expected_rows
            or tuple(mean_rows.shape) != (expected_rows, 6)
            or tuple(sigma_rows.shape) != (expected_rows, 6)
            or int(sources.numel()) != expected_rows
        ):
            raise ValueError("outer replay stage requires exact B8 x M utility/value/policy evidence")
        if (
            not bool(torch.isfinite(utility_values).all())
            or not bool(torch.isfinite(value_rows).all())
            or not bool(torch.isfinite(mean_rows).all())
            or not bool(torch.isfinite(sigma_rows).all())
            or not bool((sigma_rows > 0.0).all())
            or set(sources.tolist()) != set(range(FRONTRES_OUTER_REPLAY_SCENARIO_BATCH))
        ):
            raise ValueError("outer replay stage requires finite positive row-aligned policy evidence")
        if any(int((sources == index).sum().item()) != int(active_m) for index in range(FRONTRES_OUTER_REPLAY_SCENARIO_BATCH)):
            raise ValueError("outer replay stage requires exact M rows for each Scenario")
        grouped_utilities = tuple(
            utility_values[sources == index] for index in range(FRONTRES_OUTER_REPLAY_SCENARIO_BATCH)
        )
        grouped_old_values = tuple(
            value_rows[sources == index] for index in range(FRONTRES_OUTER_REPLAY_SCENARIO_BATCH)
        )
        grouped_means = tuple(mean_rows[sources == index] for index in range(FRONTRES_OUTER_REPLAY_SCENARIO_BATCH))
        grouped_sigmas = tuple(sigma_rows[sources == index] for index in range(FRONTRES_OUTER_REPLAY_SCENARIO_BATCH))
        for values, means, sigmas in zip(grouped_old_values, grouped_means, grouped_sigmas, strict=True):
            if not torch.equal(values, values[:1].expand_as(values)):
                raise ValueError("outer replay requires one shared old value per Scenario")
            if not torch.equal(means, means[:1].expand_as(means)) or not torch.equal(
                sigmas, sigmas[:1].expand_as(sigmas)
            ):
                raise ValueError("outer replay requires one shared pi_old distribution per Scenario")
        repair_spread_values = tuple(
            float((group - group.mean()).abs().mean().item()) for group in grouped_utilities
        )

        records = {digest: replace(record, staleness=record.staleness + 1) for digest, record in self._records.items()}
        critic_calibration_values: list[float] = []
        critic_target_means: list[float] = []
        current_utility_means: list[float] = []
        outcome_variances: list[float] = []
        standard_errors: list[float] = []
        confidence_half_widths: list[float] = []
        compatible_sample_counts: list[int] = []
        compatible_visit_counts: list[int] = []
        policy_symmetric_kls: list[float] = []
        policy_window_resets: list[bool] = []
        for index, (selection, key, utility_group, value_group, mean_group, sigma_group, spread_value) in enumerate(
            zip(
                plan.selections,
                keys,
                grouped_utilities,
                grouped_old_values,
                grouped_means,
                grouped_sigmas,
                repair_spread_values,
                strict=True,
            )
        ):
            key.validate()
            if (
                key.segment_id != selection.segment_id
                or key.perturbation_seed != selection.perturbation_seed
                or key.perturbation_family != selection.perturbation_family
                or key.perturbation_strength != selection.perturbation_strength
            ):
                raise ValueError(f"materialized ScenarioKey {index} differs from its selection")
            if selection.replay_key_digest is not None and key.digest != selection.replay_key_digest:
                raise ValueError("replayed Scenario did not reproduce its stable key")
            existing = records.get(key.digest)
            policy_mean = tuple(float(value) for value in mean_group[0].tolist())
            policy_sigma = tuple(float(value) for value in sigma_group[0].tolist())
            utility_visit = tuple(float(value) for value in utility_group.tolist())
            if existing is None:
                window = FrontRESScenarioUtilityWindow.from_visit(
                    utilities=utility_visit,
                    policy_mean=policy_mean,
                    policy_sigma=policy_sigma,
                )
                policy_symmetric_kl = 0.0
                policy_window_reset = False
            else:
                window, policy_symmetric_kl, policy_window_reset = existing.utility_window.preview_visit(
                    utilities=utility_visit,
                    policy_mean=policy_mean,
                    policy_sigma=policy_sigma,
                )
            target = window.robust_mean
            confidence = window.confidence_half_width
            calibration_value = max(abs(float(value_group[0].item()) - target) - confidence, 0.0)
            critic_calibration_values.append(calibration_value)
            critic_target_means.append(target)
            current_utility_means.append(float(utility_group.mean().item()))
            outcome_variances.append(window.outcome_variance)
            standard_errors.append(window.standard_error)
            confidence_half_widths.append(confidence)
            compatible_sample_counts.append(window.sample_count)
            compatible_visit_counts.append(window.compatible_visit_count)
            policy_symmetric_kls.append(policy_symmetric_kl)
            policy_window_resets.append(policy_window_reset)
            if existing is None:
                record = FrontRESScenarioReplayRecord(
                    key=key,
                    dr_class=selection.dr_class,
                    critic_calibration_score_by_k=((int(plan.active_k), float(calibration_value)),),
                    repair_spread_score_by_k=((int(plan.active_k), float(spread_value)),),
                    staleness=0,
                    visit_count=1,
                    last_transaction_id=plan.transaction_id,
                    utility_window=window,
                )
            else:
                record = existing.with_visit(
                    horizon_k=plan.active_k,
                    critic_calibration_value=calibration_value,
                    repair_spread_value=spread_value,
                    utility_window=window,
                    transaction_id=plan.transaction_id,
                )
            record.validate()
            records[key.digest] = record
        active_by_k = {int(k): set(values) for k, values in self._active_by_k.items()}
        active = active_by_k.setdefault(int(plan.active_k), set())
        selected_digests = {key.digest for key in keys}
        admitted_digests = {
            key.digest
            for selection, key in zip(plan.selections, keys, strict=True)
            if selection.replay_key_digest is None
        }
        active.update(selected_digests)
        while len(active) > int(plan.active_capacity_after):
            active_records = [records[digest] for digest in active if digest in records]
            protected = self._protected_anchor_digests(active_records, active_k=plan.active_k)
            admission_classes = {
                selection.dr_class for selection in plan.selections if selection.source == "global"
            }
            eligible = [
                record
                for record in active_records
                if record.key.digest not in protected
                and record.key.digest not in admitted_digests
                and record.utility_window.compatible_visit_count >= self.minimum_visits_before_expand
                and (not admission_classes or record.dr_class in admission_classes)
            ]
            if not eligible:
                raise RuntimeError("Replay Curriculum cannot evict without violating visit or anchor protection")
            evicted = min(
                eligible,
                key=lambda record: (
                    max(
                        float(record.score_for_k(plan.active_k, score_kind="critic_calibration") or 0.0),
                        float(record.score_for_k(plan.active_k, score_kind="repair_spread") or 0.0),
                    ),
                    -record.utility_window.compatible_visit_count,
                    record.key.digest,
                ),
            )
            active.remove(evicted.key.digest)
        candidate = FrontRESOuterReplayCandidate(
            transaction_id=plan.transaction_id,
            policy_snapshot_id=_require_nonempty("policy_snapshot_id", policy_snapshot_id),
            plan=plan,
            records=tuple(records[key] for key in sorted(records)),
            critic_calibration_values=tuple(critic_calibration_values),
            repair_spread_values=repair_spread_values,
            critic_target_means=tuple(critic_target_means),
            current_utility_means=tuple(current_utility_means),
            outcome_variances=tuple(outcome_variances),
            standard_errors=tuple(standard_errors),
            confidence_half_widths=tuple(confidence_half_widths),
            compatible_sample_counts=tuple(compatible_sample_counts),
            compatible_visit_counts=tuple(compatible_visit_counts),
            policy_symmetric_kls=tuple(policy_symmetric_kls),
            policy_window_resets=tuple(policy_window_resets),
            active_digests_by_k=tuple((k, tuple(sorted(values))) for k, values in sorted(active_by_k.items())),
        )
        candidate.validate()
        self._copy_generator(plan.generator_state_after)
        return candidate

    def commit(self, candidate: FrontRESOuterReplayCandidate, *, receipt: Mapping[str, Any]) -> dict[str, Any]:
        candidate.validate()
        expected = {
            "method_contract_id": "FRS-METHOD-v024",
            "training_contract_id": "FRS-TRAIN-v023",
            "transaction_id": candidate.transaction_id,
            "policy_snapshot_id": candidate.policy_snapshot_id,
            "optimizer_step_delta": 1,
        }
        if not isinstance(receipt, Mapping) or any(receipt.get(name) != value for name, value in expected.items()):
            raise ValueError("outer replay commit requires its matching committed transaction receipt")
        if self._last_commit is not None and self._last_commit.get("transaction_id") == candidate.transaction_id:
            raise RuntimeError("outer replay rejects duplicate transaction commit")
        if candidate.plan.record_state_digest != self._record_digest() or not torch.equal(
            candidate.plan.generator_state_before, self.generator.get_state().cpu()
        ):
            raise RuntimeError("outer replay owner changed before commit")
        new_records = {record.key.digest: record for record in candidate.records}
        selected_records: list[FrontRESScenarioReplayRecord] = []
        for selection in candidate.plan.selections:
            matches = tuple(
                record
                for record in candidate.records
                if record.last_transaction_id == candidate.transaction_id
                and record.key.segment_id == selection.segment_id
                and record.key.perturbation_seed == selection.perturbation_seed
                and record.key.perturbation_family == selection.perturbation_family
                and record.key.perturbation_strength == selection.perturbation_strength
            )
            if len(matches) != 1:
                raise RuntimeError("outer replay candidate lost its selected Scenario record")
            selected_records.append(matches[0])
        generator = self._copy_generator(candidate.plan.generator_state_after)
        self._records = new_records
        self._active_by_k = {int(k): set(values) for k, values in candidate.active_digests_by_k}
        self._capacity_by_k[int(candidate.plan.active_k)] = int(candidate.plan.active_capacity_after)
        self.generator.set_state(generator.get_state())
        self._last_commit = dict(expected)
        stats = self.stats(active_k=candidate.plan.active_k, score_kind=candidate.plan.score_kind)
        return {
            "transaction_id": candidate.transaction_id,
            "state_delta": 1,
            "score_kind": candidate.plan.score_kind,
            "slot_purposes": tuple(selection.purpose for selection in candidate.plan.selections),
            "critic_calibration_values": candidate.critic_calibration_values,
            "repair_spread_values": candidate.repair_spread_values,
            "critic_target_means": candidate.critic_target_means,
            "current_utility_means": candidate.current_utility_means,
            "outcome_variances": candidate.outcome_variances,
            "standard_errors": candidate.standard_errors,
            "confidence_half_widths": candidate.confidence_half_widths,
            "compatible_sample_counts": candidate.compatible_sample_counts,
            "compatible_visit_counts": candidate.compatible_visit_counts,
            "policy_symmetric_kls": candidate.policy_symmetric_kls,
            "policy_window_resets": candidate.policy_window_resets,
            "record_count": stats["record_count"],
            "archive_count": stats["record_count"],
            "active_count": stats["active_count"],
            "active_capacity_before": candidate.plan.active_capacity_before,
            "active_capacity_after": candidate.plan.active_capacity_after,
            "minimum_active_visits": stats["minimum_active_visits"],
            "replay_pool_size": stats["replay_pool_size"],
            "review_pool_size": stats["review_pool_size"],
            "priority_scores": tuple(
                float(record.score_for_k(candidate.plan.active_k, score_kind=candidate.plan.score_kind))
                for record in selected_records
            ),
            "priority_scores_by_slot": tuple(
                float(
                    record.score_for_k(
                        candidate.plan.active_k,
                        score_kind=("repair_spread" if selection.purpose == "repair_spread" else "critic_calibration"),
                    )
                )
                for record, selection in zip(selected_records, candidate.plan.selections, strict=True)
            ),
            "critic_calibration_scores": tuple(
                float(record.score_for_k(candidate.plan.active_k, score_kind="critic_calibration"))
                for record in selected_records
            ),
            "repair_spread_ema": tuple(
                float(record.score_for_k(candidate.plan.active_k, score_kind="repair_spread"))
                for record in selected_records
            ),
            "visit_counts": tuple(record.visit_count for record in selected_records),
            "staleness": tuple(record.staleness for record in selected_records),
        }

    def stats(self, *, active_k: int, score_kind: str = "critic_calibration") -> dict[str, int | float]:
        active_records = self._active_records(active_k)
        scores = [record.score_for_k(active_k, score_kind=score_kind) for record in active_records]
        active_scores = [float(value) for value in scores if value is not None]
        return {
            "record_count": len(self._records),
            "active_count": len(active_records),
            "active_capacity": self._capacity_for_k(active_k),
            "minimum_active_visits": min(
                (record.utility_window.compatible_visit_count for record in active_records),
                default=0,
            ),
            "replay_pool_size": sum(value >= self.min_replay_score for value in active_scores),
            "review_pool_size": sum(value < self.min_replay_score for value in active_scores),
            "score_mean": sum(active_scores) / len(active_scores) if active_scores else 0.0,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": FRONTRES_OUTER_REPLAY_SCHEMA,
            "min_replay_score": self.min_replay_score,
            "staleness_weight": self.staleness_weight,
            "capacity_ladder": self.capacity_ladder,
            "minimum_visits_before_expand": self.minimum_visits_before_expand,
            "records": tuple(record.to_state() for record in self.records),
            "active_by_k": tuple((k, tuple(sorted(values))) for k, values in sorted(self._active_by_k.items())),
            "capacity_by_k": tuple(sorted(self._capacity_by_k.items())),
            "generator_state": self.generator.get_state().cpu(),
            "last_commit": dict(self._last_commit) if self._last_commit is not None else None,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "schema",
            "min_replay_score",
            "staleness_weight",
            "capacity_ladder",
            "minimum_visits_before_expand",
            "records",
            "active_by_k",
            "capacity_by_k",
            "generator_state",
            "last_commit",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("outer replay checkpoint state has incompatible fields")
        if state["schema"] != FRONTRES_OUTER_REPLAY_SCHEMA:
            raise ValueError("outer replay checkpoint schema mismatch")
        if (
            float(state["min_replay_score"]) != self.min_replay_score
            or float(state["staleness_weight"]) != self.staleness_weight
            or tuple(int(value) for value in state["capacity_ladder"]) != self.capacity_ladder
            or int(state["minimum_visits_before_expand"]) != self.minimum_visits_before_expand
        ):
            raise ValueError("outer replay checkpoint scoring configuration mismatch")
        records = tuple(FrontRESScenarioReplayRecord.from_state(value) for value in state["records"])
        record_map = {record.key.digest: record for record in records}
        if len(record_map) != len(records):
            raise ValueError("outer replay checkpoint contains duplicate ScenarioKeys")
        active_by_k = {int(k): set(values) for k, values in state["active_by_k"]}
        if any(digest not in record_map for values in active_by_k.values() for digest in values):
            raise ValueError("outer replay checkpoint active membership references a missing archive record")
        capacity_by_k = {int(k): int(value) for k, value in state["capacity_by_k"]}
        if any(value not in self.capacity_ladder for value in capacity_by_k.values()):
            raise ValueError("outer replay checkpoint capacity state is invalid")
        generator_state = state["generator_state"]
        if not isinstance(generator_state, torch.Tensor) or generator_state.dtype != torch.uint8:
            raise ValueError("outer replay checkpoint generator state is invalid")
        generator = self._copy_generator(generator_state)
        last_commit = state["last_commit"]
        if last_commit is not None and (
            not isinstance(last_commit, Mapping)
            or set(last_commit)
            != {
                "method_contract_id",
                "training_contract_id",
                "transaction_id",
                "policy_snapshot_id",
                "optimizer_step_delta",
            }
        ):
            raise ValueError("outer replay checkpoint last commit is invalid")
        self._records = record_map
        self._active_by_k = active_by_k
        self._capacity_by_k = capacity_by_k
        self.generator.set_state(generator.get_state())
        self._last_commit = dict(last_commit) if last_commit is not None else None


__all__ = [
    "FRONTRES_OUTER_REPLAY_SCHEMA",
    "FRONTRES_REPLAY_POLICY_SYMMETRIC_KL_LIMIT",
    "FRONTRES_REPLAY_WINDOW_MAX_VISITS",
    "FrontRESOuterReplayCandidate",
    "FrontRESOuterReplayPlan",
    "FrontRESOuterReplaySelection",
    "FrontRESOuterScenarioReplay",
    "FrontRESScenarioKey",
    "FrontRESScenarioReplayRecord",
    "FrontRESScenarioUtilityWindow",
    "frontres_tensor_identity",
    "isolated_frontres_perturbation_rng",
]
