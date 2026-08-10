"""Committed outer prioritized replay for sealed FrontRES local Scenarios."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import math
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch


FRONTRES_OUTER_REPLAY_SCHEMA = "frontres-outer-scenario-replay-v1"
FRONTRES_OUTER_REPLAY_EMA_DECAY = 0.8


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
class FrontRESScenarioReplayRecord:
    key: FrontRESScenarioKey
    dr_class: str
    score_by_k: tuple[tuple[int, float], ...]
    staleness: int
    visit_count: int
    last_transaction_id: str

    def validate(self) -> None:
        self.key.validate()
        _require_nonempty("dr_class", self.dr_class)
        _require_nonempty("last_transaction_id", self.last_transaction_id)
        if isinstance(self.staleness, bool) or int(self.staleness) < 0:
            raise ValueError("staleness must be nonnegative")
        if isinstance(self.visit_count, bool) or int(self.visit_count) <= 0:
            raise ValueError("visit_count must be positive")
        keys = tuple(int(k) for k, _value in self.score_by_k)
        if keys != tuple(sorted(set(keys))) or any(k <= 0 for k in keys):
            raise ValueError("score_by_k must use ordered unique positive K values")
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for _k, value in self.score_by_k):
            raise ValueError("score_by_k values must be finite and nonnegative")

    def score_for_k(self, horizon_k: int) -> float | None:
        return dict(self.score_by_k).get(int(horizon_k))

    def with_visit(self, *, horizon_k: int, learning_value: float, transaction_id: str) -> "FrontRESScenarioReplayRecord":
        self.validate()
        if not math.isfinite(float(learning_value)) or float(learning_value) < 0.0:
            raise ValueError("learning_value must be finite and nonnegative")
        scores = dict(self.score_by_k)
        previous = scores.get(int(horizon_k))
        scores[int(horizon_k)] = (
            float(learning_value)
            if previous is None
            else FRONTRES_OUTER_REPLAY_EMA_DECAY * float(previous)
            + (1.0 - FRONTRES_OUTER_REPLAY_EMA_DECAY) * float(learning_value)
        )
        result = replace(
            self,
            score_by_k=tuple(sorted(scores.items())),
            staleness=0,
            visit_count=int(self.visit_count) + 1,
            last_transaction_id=_require_nonempty("transaction_id", transaction_id),
        )
        result.validate()
        return result

    def to_state(self) -> dict[str, Any]:
        self.validate()
        return {
            "key": self.key.to_state(),
            "dr_class": self.dr_class,
            "score_by_k": tuple((int(k), float(value)) for k, value in self.score_by_k),
            "staleness": int(self.staleness),
            "visit_count": int(self.visit_count),
            "last_transaction_id": self.last_transaction_id,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "FrontRESScenarioReplayRecord":
        expected = {"key", "dr_class", "score_by_k", "staleness", "visit_count", "last_transaction_id"}
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("Scenario replay record has incompatible fields")
        result = cls(
            key=FrontRESScenarioKey.from_state(state["key"]),
            dr_class=str(state["dr_class"]),
            score_by_k=tuple((int(k), float(value)) for k, value in state["score_by_k"]),
            staleness=int(state["staleness"]),
            visit_count=int(state["visit_count"]),
            last_transaction_id=str(state["last_transaction_id"]),
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
    replay_key_digest: str | None
    score: float
    staleness: int

    def validate(self) -> None:
        if self.source not in {"global", "replay", "review"}:
            raise ValueError(f"unsupported Scenario source {self.source!r}")
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
    active_k: int
    selections: tuple[FrontRESOuterReplaySelection, ...]
    generator_state_before: torch.Tensor
    generator_state_after: torch.Tensor
    record_state_digest: str

    def validate(self) -> None:
        _require_nonempty("transaction_id", self.transaction_id)
        if int(self.active_k) <= 0 or len(self.selections) != 2:
            raise ValueError("outer replay plan requires positive K and exactly two selections")
        for selection in self.selections:
            selection.validate()
        if len({selection.segment_id for selection in self.selections}) != 2:
            raise ValueError("outer replay plan requires two distinct Segment sources")
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
    learning_values: tuple[float, float]

    def validate(self) -> None:
        self.plan.validate()
        if self.transaction_id != self.plan.transaction_id:
            raise ValueError("candidate transaction differs from its selection plan")
        _require_nonempty("policy_snapshot_id", self.policy_snapshot_id)
        if len(self.learning_values) != 2 or any(
            not math.isfinite(float(value)) or float(value) < 0.0 for value in self.learning_values
        ):
            raise ValueError("candidate requires two finite nonnegative learning values")
        digests = []
        for record in self.records:
            record.validate()
            digests.append(record.key.digest)
        if len(digests) != len(set(digests)):
            raise ValueError("candidate contains duplicate Scenario records")


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
        seed: int = 0,
    ) -> None:
        fractions = (float(global_frac), float(replay_frac), float(review_frac))
        if min(fractions) < 0.0 or sum(fractions) <= 0.0:
            raise ValueError("outer replay fractions must be nonnegative with positive mass")
        total = sum(fractions)
        self.global_frac, self.replay_frac, self.review_frac = tuple(value / total for value in fractions)
        self.min_replay_score = float(min_replay_score)
        self.staleness_weight = float(staleness_weight)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        self._records: dict[str, FrontRESScenarioReplayRecord] = {}
        self._last_commit: dict[str, Any] | None = None

    @property
    def records(self) -> tuple[FrontRESScenarioReplayRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def _record_digest(self) -> str:
        digest = hashlib.sha256()
        for record in self.records:
            digest.update(repr(record.to_state()).encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

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

    def _pool(self, source: str, *, active_k: int, excluded_segments: set[int]) -> list[FrontRESScenarioReplayRecord]:
        rows = [
            record
            for record in self.records
            if record.key.segment_id not in excluded_segments and record.score_for_k(active_k) is not None
        ]
        if source == "replay":
            return [record for record in rows if float(record.score_for_k(active_k)) >= self.min_replay_score]
        if source == "review":
            return [record for record in rows if float(record.score_for_k(active_k)) < self.min_replay_score]
        return []

    def _sample_record(
        self,
        pool: Sequence[FrontRESScenarioReplayRecord],
        *,
        source: str,
        active_k: int,
        generator: torch.Generator,
    ) -> FrontRESScenarioReplayRecord:
        ordered = sorted(pool, key=lambda record: (float(record.score_for_k(active_k)), record.key.digest))
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
        active_k: int,
        num_segments: int,
        eligible: Callable[[int], bool],
        global_descriptor: Callable[[int, int], tuple[str, float, str]],
    ) -> FrontRESOuterReplayPlan:
        _require_nonempty("transaction_id", transaction_id)
        if not callable(eligible) or not callable(global_descriptor):
            raise TypeError("outer replay plan requires eligible and global_descriptor callables")
        before = self.generator.get_state().detach().cpu().clone()
        generator = self._copy_generator(before)
        selected: list[FrontRESOuterReplaySelection] = []
        excluded_segments: set[int] = set()
        seen_segments = {record.key.segment_id for record in self.records}
        for _slot in range(2):
            requested_source = self._choose_source(generator)
            pool = self._pool(requested_source, active_k=int(active_k), excluded_segments=excluded_segments)
            if pool:
                record = self._sample_record(
                    pool,
                    source=requested_source,
                    active_k=int(active_k),
                    generator=generator,
                )
                selection = FrontRESOuterReplaySelection(
                    source=requested_source,
                    segment_id=record.key.segment_id,
                    perturbation_seed=record.key.perturbation_seed,
                    perturbation_family=record.key.perturbation_family,
                    perturbation_strength=record.key.perturbation_strength,
                    dr_class=record.dr_class,
                    replay_key_digest=record.key.digest,
                    score=float(record.score_for_k(active_k)),
                    staleness=record.staleness,
                )
            else:
                segment_id = self._sample_global_segment(
                    generator,
                    num_segments=int(num_segments),
                    eligible=eligible,
                    excluded_segments=excluded_segments,
                    seen_segments=seen_segments,
                )
                perturbation_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
                family, strength, dr_class = global_descriptor(segment_id, perturbation_seed)
                selection = FrontRESOuterReplaySelection(
                    source="global",
                    segment_id=segment_id,
                    perturbation_seed=perturbation_seed,
                    perturbation_family=str(family),
                    perturbation_strength=float(strength),
                    dr_class=str(dr_class),
                    replay_key_digest=None,
                    score=0.0,
                    staleness=0,
                )
            selection.validate()
            selected.append(selection)
            excluded_segments.add(selection.segment_id)
        result = FrontRESOuterReplayPlan(
            transaction_id=str(transaction_id),
            active_k=int(active_k),
            selections=tuple(selected),
            generator_state_before=before,
            generator_state_after=generator.get_state().detach().cpu().clone(),
            record_state_digest=self._record_digest(),
        )
        result.validate()
        return result

    def stage(
        self,
        plan: FrontRESOuterReplayPlan,
        *,
        keys: Sequence[FrontRESScenarioKey],
        actor_advantages: torch.Tensor,
        source_index: torch.Tensor,
        policy_snapshot_id: str,
        active_m: int,
    ) -> FrontRESOuterReplayCandidate:
        plan.validate()
        if plan.record_state_digest != self._record_digest() or not torch.equal(
            plan.generator_state_before, self.generator.get_state().cpu()
        ):
            raise RuntimeError("outer replay state changed after selection preview")
        if len(keys) != 2 or any(key.horizon_k != plan.active_k for key in keys):
            raise ValueError("outer replay stage requires two current-K ScenarioKeys")
        advantages = torch.as_tensor(actor_advantages, dtype=torch.float32).detach().cpu().flatten()
        sources = torch.as_tensor(source_index, dtype=torch.long).detach().cpu().flatten()
        if int(advantages.numel()) != 2 * int(active_m) or int(sources.numel()) != int(advantages.numel()):
            raise ValueError("outer replay stage requires exact two-Scenario x M advantages")
        if not bool(torch.isfinite(advantages).all()) or set(sources.tolist()) != {0, 1}:
            raise ValueError("outer replay stage requires finite row-aligned source evidence")
        learning_values = tuple(float(advantages[sources == index].abs().mean().item()) for index in range(2))
        if any(int((sources == index).sum().item()) != int(active_m) for index in range(2)):
            raise ValueError("outer replay stage requires exact M rows for each Scenario")

        records = {digest: replace(record, staleness=record.staleness + 1) for digest, record in self._records.items()}
        for index, (selection, key, learning_value) in enumerate(zip(plan.selections, keys, learning_values, strict=True)):
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
            if existing is None:
                record = FrontRESScenarioReplayRecord(
                    key=key,
                    dr_class=selection.dr_class,
                    score_by_k=((int(plan.active_k), float(learning_value)),),
                    staleness=0,
                    visit_count=1,
                    last_transaction_id=plan.transaction_id,
                )
            else:
                record = existing.with_visit(
                    horizon_k=plan.active_k,
                    learning_value=learning_value,
                    transaction_id=plan.transaction_id,
                )
            record.validate()
            records[key.digest] = record
        candidate = FrontRESOuterReplayCandidate(
            transaction_id=plan.transaction_id,
            policy_snapshot_id=_require_nonempty("policy_snapshot_id", policy_snapshot_id),
            plan=plan,
            records=tuple(records[key] for key in sorted(records)),
            learning_values=learning_values,
        )
        candidate.validate()
        self._copy_generator(plan.generator_state_after)
        return candidate

    def commit(self, candidate: FrontRESOuterReplayCandidate, *, receipt: Mapping[str, Any]) -> dict[str, Any]:
        candidate.validate()
        expected = {
            "method_contract_id": "FRS-METHOD-v021",
            "training_contract_id": "FRS-TRAIN-v020",
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
        self.generator.set_state(generator.get_state())
        self._last_commit = dict(expected)
        stats = self.stats(active_k=candidate.plan.active_k)
        return {
            "transaction_id": candidate.transaction_id,
            "state_delta": 1,
            "learning_values": candidate.learning_values,
            "record_count": stats["record_count"],
            "replay_pool_size": stats["replay_pool_size"],
            "review_pool_size": stats["review_pool_size"],
            "ema_scores": tuple(float(record.score_for_k(candidate.plan.active_k)) for record in selected_records),
            "visit_counts": tuple(record.visit_count for record in selected_records),
            "staleness": tuple(record.staleness for record in selected_records),
        }

    def stats(self, *, active_k: int) -> dict[str, int | float]:
        scores = [record.score_for_k(active_k) for record in self.records]
        active_scores = [float(value) for value in scores if value is not None]
        return {
            "record_count": len(self._records),
            "replay_pool_size": sum(value >= self.min_replay_score for value in active_scores),
            "review_pool_size": sum(value < self.min_replay_score for value in active_scores),
            "score_mean": sum(active_scores) / len(active_scores) if active_scores else 0.0,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": FRONTRES_OUTER_REPLAY_SCHEMA,
            "fractions": (self.global_frac, self.replay_frac, self.review_frac),
            "min_replay_score": self.min_replay_score,
            "staleness_weight": self.staleness_weight,
            "records": tuple(record.to_state() for record in self.records),
            "generator_state": self.generator.get_state().cpu(),
            "last_commit": dict(self._last_commit) if self._last_commit is not None else None,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "schema",
            "fractions",
            "min_replay_score",
            "staleness_weight",
            "records",
            "generator_state",
            "last_commit",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("outer replay checkpoint state has incompatible fields")
        if state["schema"] != FRONTRES_OUTER_REPLAY_SCHEMA:
            raise ValueError("outer replay checkpoint schema mismatch")
        fractions = tuple(float(value) for value in state["fractions"])
        if fractions != (self.global_frac, self.replay_frac, self.review_frac):
            raise ValueError("outer replay checkpoint sampling fractions mismatch")
        if (
            float(state["min_replay_score"]) != self.min_replay_score
            or float(state["staleness_weight"]) != self.staleness_weight
        ):
            raise ValueError("outer replay checkpoint scoring configuration mismatch")
        records = tuple(FrontRESScenarioReplayRecord.from_state(value) for value in state["records"])
        record_map = {record.key.digest: record for record in records}
        if len(record_map) != len(records):
            raise ValueError("outer replay checkpoint contains duplicate ScenarioKeys")
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
        self.generator.set_state(generator.get_state())
        self._last_commit = dict(last_commit) if last_commit is not None else None


__all__ = [
    "FRONTRES_OUTER_REPLAY_SCHEMA",
    "FrontRESOuterReplayCandidate",
    "FrontRESOuterReplayPlan",
    "FrontRESOuterReplaySelection",
    "FrontRESOuterScenarioReplay",
    "FrontRESScenarioKey",
    "FrontRESScenarioReplayRecord",
    "frontres_tensor_identity",
    "isolated_frontres_perturbation_rng",
]
