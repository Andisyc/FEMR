"""FRS-TRAIN-v025 outer Replay without scalar utility or Critic targets."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from typing import Any, Callable, Mapping, Sequence

import torch

from rsl_rl.frontres.frontres_outer_scenario_replay import (
    FRONTRES_OUTER_REPLAY_CAPACITY_LADDER,
    FRONTRES_OUTER_REPLAY_MIN_VISITS,
    FRONTRES_OUTER_REPLAY_SCENARIO_BATCH,
    FrontRESOuterReplayPlan,
    FrontRESOuterReplaySelection,
    FrontRESScenarioKey,
)
from rsl_rl.frontres.frontres_segment_warmup import (
    FrontRESKStageIdentity,
    frontres_v021_dr_strength_in_class,
    sample_frontres_v013_dr_strength,
)


FRONTRES_RELATIONAL_REPLAY_SCHEMA = "frontres-relational-scenario-replay-v1"


@dataclass(frozen=True)
class FrontRESRelationalReplaySelection(FrontRESOuterReplaySelection):
    def validate(self) -> None:
        if self.source not in {"global", "replay", "review"}:
            raise ValueError("relational Replay selection has an invalid source")
        if self.purpose not in {"admission", "edge_density", "stale_review"}:
            raise ValueError("relational Replay selection has an invalid purpose")
        if int(self.segment_id) < 0 or int(self.perturbation_seed) < 0:
            raise ValueError("relational Replay selection identity must be nonnegative")
        if not self.perturbation_family or not self.dr_class:
            raise ValueError("relational Replay selection requires family and DR class")
        if not math.isfinite(float(self.perturbation_strength)) or float(self.perturbation_strength) < 0.0:
            raise ValueError("relational Replay selection strength must be finite and nonnegative")
        if not math.isfinite(float(self.score)) or not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("relational Replay edge density must be in [0,1]")
        if int(self.staleness) < 0:
            raise ValueError("relational Replay staleness must be nonnegative")
        if self.source == "global" and self.replay_key_digest is not None:
            raise ValueError("global relational selection cannot claim a Replay key")
        if self.source != "global" and not self.replay_key_digest:
            raise ValueError("replayed relational selection requires a Replay key")


@dataclass(frozen=True)
class FrontRESRelationalReplayPlan(FrontRESOuterReplayPlan):
    @property
    def score_kind(self) -> str:
        return "edge_density"

    def validate(self) -> None:
        if not self.transaction_id or not isinstance(self.curriculum, FrontRESKStageIdentity):
            raise ValueError("relational Replay plan requires transaction and curriculum identity")
        if len(self.selections) != FRONTRES_OUTER_REPLAY_SCENARIO_BATCH or int(self.active_k) <= 0:
            raise ValueError("relational Replay plan requires exactly eight current-K selections")
        if len({value.segment_id for value in self.selections}) != FRONTRES_OUTER_REPLAY_SCENARIO_BATCH:
            raise ValueError("relational Replay plan requires eight distinct Segment sources")
        expected = ("admission",) + ("edge_density",) * 6 + ("stale_review",)
        if tuple(value.purpose for value in self.selections) != expected:
            raise ValueError("relational Replay plan has an invalid B8 slot layout")
        for value in self.selections:
            if not isinstance(value, FrontRESRelationalReplaySelection):
                raise TypeError("relational Replay plan contains a foreign selection")
            value.validate()
            if not frontres_v021_dr_strength_in_class(
                self.curriculum,
                class_name=value.dr_class,
                strength=value.perturbation_strength,
            ):
                raise ValueError("relational Replay selection lies outside the sealed DR interval")
        for state in (self.generator_state_before, self.generator_state_after):
            if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1:
                raise ValueError("relational Replay plan requires uint8 generator states")
        if len(self.record_state_digest) != 64:
            raise ValueError("relational Replay plan requires a SHA-256 state digest")
        if self.active_capacity_before <= 0 or self.active_capacity_after < self.active_capacity_before:
            raise ValueError("relational Replay plan has an invalid capacity transition")
        if self.replacement_digest is not None:
            raise ValueError("relational Replay evictions are staged from the sealed record digest")


@dataclass(frozen=True)
class FrontRESRelationalReplayRecord:
    key: FrontRESScenarioKey
    dr_class: str
    edge_density_by_k: tuple[tuple[int, float], ...]
    staleness: int
    visit_count: int
    last_transaction_id: str

    def validate(self) -> None:
        self.key.validate()
        if not self.dr_class or not self.last_transaction_id:
            raise ValueError("relational Replay record requires complete identity")
        if int(self.staleness) < 0 or int(self.visit_count) <= 0:
            raise ValueError("relational Replay record has invalid lifecycle counters")
        keys = tuple(int(k) for k, _value in self.edge_density_by_k)
        if keys != tuple(sorted(set(keys))) or any(k <= 0 for k in keys):
            raise ValueError("relational Replay edge-density map has invalid K identities")
        if any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0 for _k, value in self.edge_density_by_k):
            raise ValueError("relational Replay edge density must be finite in [0,1]")

    def score_for_k(self, horizon_k: int) -> float | None:
        return dict(self.edge_density_by_k).get(int(horizon_k))

    def with_visit(self, *, horizon_k: int, edge_density: float, transaction_id: str) -> "FrontRESRelationalReplayRecord":
        values = dict(self.edge_density_by_k)
        values[int(horizon_k)] = float(edge_density)
        result = replace(
            self,
            edge_density_by_k=tuple(sorted(values.items())),
            staleness=0,
            visit_count=int(self.visit_count) + 1,
            last_transaction_id=str(transaction_id),
        )
        result.validate()
        return result

    def to_state(self) -> dict[str, Any]:
        self.validate()
        return {
            "key": self.key.to_state(),
            "dr_class": self.dr_class,
            "edge_density_by_k": self.edge_density_by_k,
            "staleness": int(self.staleness),
            "visit_count": int(self.visit_count),
            "last_transaction_id": self.last_transaction_id,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "FrontRESRelationalReplayRecord":
        expected = {"key", "dr_class", "edge_density_by_k", "staleness", "visit_count", "last_transaction_id"}
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("relational Replay record state has incompatible fields")
        result = cls(
            key=FrontRESScenarioKey.from_state(state["key"]),
            dr_class=str(state["dr_class"]),
            edge_density_by_k=tuple((int(k), float(value)) for k, value in state["edge_density_by_k"]),
            staleness=int(state["staleness"]),
            visit_count=int(state["visit_count"]),
            last_transaction_id=str(state["last_transaction_id"]),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class FrontRESRelationalReplayCandidate:
    transaction_id: str
    policy_snapshot_id: str
    plan: FrontRESRelationalReplayPlan
    records: tuple[FrontRESRelationalReplayRecord, ...]
    edge_counts: tuple[int, ...]
    edge_densities: tuple[float, ...]
    active_digests_by_k: tuple[tuple[int, tuple[str, ...]], ...]

    def validate(self) -> None:
        self.plan.validate()
        if self.transaction_id != self.plan.transaction_id or not self.policy_snapshot_id:
            raise ValueError("relational Replay candidate identity differs from its plan")
        if len(self.edge_counts) != 8 or len(self.edge_densities) != 8:
            raise ValueError("relational Replay candidate requires eight Scenario diagnostics")
        if any(int(value) < 0 for value in self.edge_counts) or any(
            not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
            for value in self.edge_densities
        ):
            raise ValueError("relational Replay candidate has invalid edge diagnostics")
        digests = tuple(value.key.digest for value in self.records)
        if len(digests) != len(set(digests)):
            raise ValueError("relational Replay candidate contains duplicate records")


class FrontRESRelationalScenarioReplay:
    """Preview, stage, and atomically commit relation-only Scenario Replay."""

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
        del global_frac, replay_frac, review_frac, staleness_weight
        if not 0.0 <= float(min_replay_score) <= 1.0:
            raise ValueError("relational Replay minimum edge density must be in [0,1]")
        ladder = tuple(int(value) for value in capacity_ladder)
        if not ladder or tuple(sorted(set(ladder))) != ladder or any(value <= 0 for value in ladder):
            raise ValueError("relational Replay capacity ladder must be positive and increasing")
        self.min_replay_score = float(min_replay_score)
        self.capacity_ladder = ladder
        self.minimum_visits_before_expand = int(minimum_visits_before_expand)
        if self.minimum_visits_before_expand <= 0:
            raise ValueError("relational Replay minimum visits must be positive")
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        self._records: dict[str, FrontRESRelationalReplayRecord] = {}
        self._active_by_k: dict[int, set[str]] = {}
        self._capacity_by_k: dict[int, int] = {}
        self._last_commit: dict[str, Any] | None = None

    @property
    def records(self) -> tuple[FrontRESRelationalReplayRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def _digest(self) -> str:
        digest = hashlib.sha256()
        for value in self.records:
            digest.update(repr(value.to_state()).encode())
        digest.update(repr(tuple((k, tuple(sorted(v))) for k, v in sorted(self._active_by_k.items()))).encode())
        digest.update(repr(tuple(sorted(self._capacity_by_k.items()))).encode())
        return digest.hexdigest()

    @staticmethod
    def _copy_generator(state: torch.Tensor) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.set_state(state.detach().cpu().clone())
        return generator

    def _active(self, active_k: int) -> tuple[FrontRESRelationalReplayRecord, ...]:
        return tuple(
            self._records[value]
            for value in sorted(self._active_by_k.get(int(active_k), set()))
            if value in self._records
        )

    def plan(
        self,
        *,
        transaction_id: str,
        curriculum: FrontRESKStageIdentity,
        num_segments: int,
        eligible: Callable[[int], bool],
        global_family: Callable[[int], str],
    ) -> FrontRESRelationalReplayPlan:
        if not transaction_id or not isinstance(curriculum, FrontRESKStageIdentity):
            raise ValueError("relational Replay planning requires transaction and curriculum")
        if self._last_commit is not None and self._last_commit.get("transaction_id") == transaction_id:
            raise RuntimeError("relational Replay rejects a duplicate committed transaction")
        before = self.generator.get_state().cpu().clone()
        generator = self._copy_generator(before)
        excluded: set[int] = set()
        active = self._active(curriculum.active_k)
        selections: list[FrontRESRelationalReplaySelection] = []
        purposes = ("admission",) + ("edge_density",) * 6 + ("stale_review",)
        for purpose in purposes:
            pool = tuple(
                value for value in active
                if value.key.segment_id not in excluded
                and value.score_for_k(curriculum.active_k) is not None
                and float(value.score_for_k(curriculum.active_k)) >= self.min_replay_score
                and frontres_v021_dr_strength_in_class(
                    curriculum,
                    class_name=value.dr_class,
                    strength=value.key.perturbation_strength,
                )
            )
            record = None
            if purpose != "admission" and pool:
                record = max(
                    pool,
                    key=(
                        (lambda value: (value.staleness, value.key.digest))
                        if purpose == "stale_review"
                        else (lambda value: (float(value.score_for_k(curriculum.active_k)), value.staleness, value.key.digest))
                    ),
                )
            if record is not None:
                selection = FrontRESRelationalReplaySelection(
                    source="review" if purpose == "stale_review" else "replay",
                    segment_id=record.key.segment_id,
                    perturbation_seed=record.key.perturbation_seed,
                    perturbation_family=record.key.perturbation_family,
                    perturbation_strength=record.key.perturbation_strength,
                    dr_class=record.dr_class,
                    purpose=purpose,
                    replay_key_digest=record.key.digest,
                    score=float(record.score_for_k(curriculum.active_k)),
                    staleness=record.staleness,
                )
            else:
                for _ in range(4096):
                    segment_id = int(torch.randint(0, int(num_segments), (1,), generator=generator).item())
                    if segment_id not in excluded and bool(eligible(segment_id)):
                        break
                else:
                    raise RuntimeError("relational Replay could not find eight eligible Segment identities")
                sample_key = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
                dr = sample_frontres_v013_dr_strength(curriculum, sample_key=sample_key)
                selection = FrontRESRelationalReplaySelection(
                    source="global",
                    segment_id=segment_id,
                    perturbation_seed=int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item()),
                    perturbation_family=str(global_family(segment_id)),
                    perturbation_strength=float(torch.tensor(dr.strength, dtype=torch.float32).item()),
                    dr_class=str(dr.class_name),
                    purpose=purpose,
                    replay_key_digest=None,
                    score=0.0,
                    staleness=0,
                )
            selection.validate()
            selections.append(selection)
            excluded.add(selection.segment_id)
        capacity_before = int(self._capacity_by_k.get(curriculum.active_k, self.capacity_ladder[0]))
        capacity_after = capacity_before
        active_for_k = self._active(curriculum.active_k)
        if (
            len(active_for_k) >= capacity_before
            and active_for_k
            and all(value.visit_count >= self.minimum_visits_before_expand for value in active_for_k)
        ):
            ladder_index = self.capacity_ladder.index(capacity_before)
            if ladder_index + 1 < len(self.capacity_ladder):
                capacity_after = self.capacity_ladder[ladder_index + 1]
        plan = FrontRESRelationalReplayPlan(
            transaction_id=transaction_id,
            curriculum=curriculum,
            selections=tuple(selections),
            generator_state_before=before,
            generator_state_after=generator.get_state().cpu().clone(),
            record_state_digest=self._digest(),
            active_capacity_before=capacity_before,
            active_capacity_after=capacity_after,
            replacement_digest=None,
        )
        plan.validate()
        if self._last_commit is not None and self._last_commit.get("transaction_id") == plan.transaction_id:
            raise RuntimeError("relational Replay rejects duplicate staging")
        return plan

    def stage(
        self,
        plan: FrontRESRelationalReplayPlan,
        *,
        keys: Sequence[FrontRESScenarioKey],
        preference_edges: Sequence[tuple[int, int]],
        source_index: torch.Tensor,
        policy_snapshot_id: str,
        active_m: int,
    ) -> FrontRESRelationalReplayCandidate:
        plan.validate()
        if plan.record_state_digest != self._digest() or not torch.equal(
            plan.generator_state_before, self.generator.get_state().cpu()
        ):
            raise RuntimeError("relational Replay changed after selection preview")
        if int(active_m) != 4 or len(keys) != 8:
            raise ValueError("relational Replay requires sealed B8 x M4 identity")
        sources = source_index.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        if tuple(sources.shape) != (32,) or any(int((sources == value).sum()) != 4 for value in range(8)):
            raise ValueError("relational Replay requires four policy rows per Scenario")
        edge_counts = [0] * 8
        for winner, loser in preference_edges:
            if not (0 <= int(winner) < 32) or not (0 <= int(loser) < 32) or int(winner) == int(loser):
                raise ValueError("relational Replay preference edge is out of range or self-referential")
            winner_source = int(sources[int(winner)].item())
            loser_source = int(sources[int(loser)].item())
            if winner_source != loser_source:
                raise ValueError("relational Replay rejects cross-Scenario preference edges")
            edge_counts[winner_source] += 1
        densities = tuple(float(value / 6.0) for value in edge_counts)
        records = {key: replace(value, staleness=value.staleness + 1) for key, value in self._records.items()}
        for selection, key, density in zip(plan.selections, keys, densities, strict=True):
            key.validate()
            if (
                key.segment_id != selection.segment_id
                or key.perturbation_seed != selection.perturbation_seed
                or key.perturbation_family != selection.perturbation_family
                or key.perturbation_strength != selection.perturbation_strength
            ):
                raise ValueError("relational Replay materialized Scenario differs from its selection")
            if selection.replay_key_digest is not None and key.digest != selection.replay_key_digest:
                raise ValueError("relational Replay did not reproduce its stable ScenarioKey")
            existing = records.get(key.digest)
            records[key.digest] = (
                FrontRESRelationalReplayRecord(
                    key=key,
                    dr_class=selection.dr_class,
                    edge_density_by_k=((int(plan.active_k), density),),
                    staleness=0,
                    visit_count=1,
                    last_transaction_id=plan.transaction_id,
                )
                if existing is None
                else existing.with_visit(
                    horizon_k=plan.active_k,
                    edge_density=density,
                    transaction_id=plan.transaction_id,
                )
            )
        active_by_k = {int(k): set(values) for k, values in self._active_by_k.items()}
        active_set = active_by_k.setdefault(int(plan.active_k), set())
        selected = {key.digest for key in keys}
        active_set.update(selected)
        capacity = int(plan.active_capacity_after)
        while len(active_set) > capacity:
            removable = [records[value] for value in active_set if value not in selected]
            if not removable:
                raise RuntimeError("relational Replay active capacity cannot preserve the selected transaction")
            victim = min(
                removable,
                key=lambda value: (float(value.score_for_k(plan.active_k) or 0.0), -value.staleness, value.key.digest),
            )
            active_set.remove(victim.key.digest)
        candidate = FrontRESRelationalReplayCandidate(
            transaction_id=plan.transaction_id,
            policy_snapshot_id=str(policy_snapshot_id),
            plan=plan,
            records=tuple(records[key] for key in sorted(records)),
            edge_counts=tuple(edge_counts),
            edge_densities=densities,
            active_digests_by_k=tuple((k, tuple(sorted(values))) for k, values in sorted(active_by_k.items())),
        )
        candidate.validate()
        return candidate

    def commit(self, candidate: FrontRESRelationalReplayCandidate, *, receipt: Mapping[str, Any]) -> dict[str, Any]:
        candidate.validate()
        expected = {
            "method_contract_id": "FRS-METHOD-v026",
            "training_contract_id": "FRS-TRAIN-v025",
            "transaction_id": candidate.transaction_id,
            "policy_snapshot_id": candidate.policy_snapshot_id,
            "optimizer_step_delta": 1,
        }
        if not isinstance(receipt, Mapping) or any(receipt.get(name) != value for name, value in expected.items()):
            raise ValueError("relational Replay commit requires its matching transaction receipt")
        if self._last_commit is not None and self._last_commit.get("transaction_id") == candidate.transaction_id:
            raise RuntimeError("relational Replay rejects duplicate commit")
        if candidate.plan.record_state_digest != self._digest():
            raise RuntimeError("relational Replay owner changed before commit")
        self._records = {value.key.digest: value for value in candidate.records}
        self._active_by_k = {int(k): set(values) for k, values in candidate.active_digests_by_k}
        self._capacity_by_k[int(candidate.plan.active_k)] = int(candidate.plan.active_capacity_after)
        self.generator.set_state(candidate.plan.generator_state_after)
        self._last_commit = dict(expected)
        return {
            "schema": FRONTRES_RELATIONAL_REPLAY_SCHEMA,
            "state_delta": 1,
            "score_kind": "edge_density",
            "edge_counts": candidate.edge_counts,
            "edge_densities": candidate.edge_densities,
            "record_count": len(self._records),
            "active_count": len(self._active_by_k.get(candidate.plan.active_k, set())),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": FRONTRES_RELATIONAL_REPLAY_SCHEMA,
            "min_replay_score": self.min_replay_score,
            "capacity_ladder": self.capacity_ladder,
            "minimum_visits_before_expand": self.minimum_visits_before_expand,
            "records": tuple(value.to_state() for value in self.records),
            "active_by_k": tuple((k, tuple(sorted(values))) for k, values in sorted(self._active_by_k.items())),
            "capacity_by_k": tuple(sorted(self._capacity_by_k.items())),
            "generator_state": self.generator.get_state().cpu(),
            "last_commit": dict(self._last_commit) if self._last_commit is not None else None,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {"schema", "min_replay_score", "capacity_ladder", "minimum_visits_before_expand", "records", "active_by_k", "capacity_by_k", "generator_state", "last_commit"}
        if not isinstance(state, Mapping) or set(state) != expected or state["schema"] != FRONTRES_RELATIONAL_REPLAY_SCHEMA:
            raise ValueError("relational Replay checkpoint schema is incompatible")
        if (
            float(state["min_replay_score"]) != self.min_replay_score
            or tuple(state["capacity_ladder"]) != self.capacity_ladder
            or int(state["minimum_visits_before_expand"]) != self.minimum_visits_before_expand
        ):
            raise ValueError("relational Replay checkpoint configuration differs from runtime")
        records = tuple(FrontRESRelationalReplayRecord.from_state(value) for value in state["records"])
        record_map = {value.key.digest: value for value in records}
        if len(record_map) != len(records):
            raise ValueError("relational Replay checkpoint contains duplicate ScenarioKeys")
        active = {int(k): set(values) for k, values in state["active_by_k"]}
        if any(value not in record_map for values in active.values() for value in values):
            raise ValueError("relational Replay checkpoint active set references a missing record")
        capacity_by_k = {int(k): int(value) for k, value in state["capacity_by_k"]}
        if any(value not in self.capacity_ladder for value in capacity_by_k.values()):
            raise ValueError("relational Replay checkpoint has an invalid active capacity")
        generator = state["generator_state"]
        if not isinstance(generator, torch.Tensor) or generator.dtype != torch.uint8:
            raise ValueError("relational Replay checkpoint generator state is invalid")
        self._records = record_map
        self._active_by_k = active
        self._capacity_by_k = capacity_by_k
        self.generator.set_state(generator)
        self._last_commit = dict(state["last_commit"]) if state["last_commit"] is not None else None


__all__ = (
    "FRONTRES_RELATIONAL_REPLAY_SCHEMA",
    "FrontRESRelationalReplayCandidate",
    "FrontRESRelationalReplayPlan",
    "FrontRESRelationalReplayRecord",
    "FrontRESRelationalReplaySelection",
    "FrontRESRelationalScenarioReplay",
)
