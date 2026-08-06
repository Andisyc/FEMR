"""Interface-oriented owner for one formal FrontRES-v015 Stage-3 transaction.

MOSAIC remains the lifecycle host. This module owns only the FrontRES
transaction state machine and presents the legacy live-probe implementation as
one concrete backend. The frozen MOSAIC runner therefore depends on the two
stable public dispatch functions in ``frontres_segment_live_update_loop``;
FrontRES internals no longer leak through that boundary.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable

from rsl_rl.frontres.frontres_interfaces import (
    FrontRESStage3Backend,
    FrontRESStage3Request,
    FrontRESTransactionProvider,
    FrontRESActiveCommittedUpdateView,
    FrontRESActiveTransactionRequestView,
)


_MAX_REJECTED_COLLECTIONS = 8
_ENGINE_ATTR = "_frontres_stage3_engine"
_TRANSACTION_STATE_ATTR = "_frontres_checkpoint_transaction_state"


class FrontRESStage3TransactionAggregate(Mapping[str, Any]):
    """Single owner for execution phase and checkpoint-visible transaction state."""

    def __init__(self, persisted: Mapping[str, Any] | None = None) -> None:
        self._execution_phase = "idle"
        self._persisted = self._validate_persisted(persisted or {"state": "idle"})
        self._collection_route = ""
        self._sample: Any | None = None
        self._batch: Any | None = None
        self._observation_trace: dict[str, Any] = {}
        self._preupdate_diagnostics: dict[str, Any] = {}

    @staticmethod
    def _validate_persisted(value: Mapping[str, Any]) -> dict[str, Any]:
        state = str(value.get("state", ""))
        if state == "idle" and set(value) == {"state"}:
            return {"state": "idle"}
        if state == "committed" and set(value) == {"state", "receipt"}:
            receipt = value.get("receipt")
            if not isinstance(receipt, Mapping):
                raise ValueError("committed FrontRES transaction requires a receipt mapping")
            return {"state": "committed", "receipt": dict(receipt)}
        if state in {"collecting", "sealed"}:
            return dict(value)
        raise ValueError(f"unsupported FrontRES transaction persistence state={state!r}")

    @property
    def execution_phase(self) -> str:
        return self._execution_phase

    @property
    def persistence_phase(self) -> str:
        return str(self._persisted["state"])

    def begin_collection(self) -> None:
        if self._execution_phase != "idle" or self.persistence_phase in {"collecting", "sealed"}:
            raise RuntimeError(
                "FrontRES transaction cannot begin collection from "
                f"execution={self._execution_phase!r} persistence={self.persistence_phase!r}"
            )
        self._execution_phase = "collecting"
        self._persisted = {"state": "collecting", "phase": "provider"}

    def begin_readonly_collection(self) -> None:
        """Open evaluation context without changing checkpoint-visible transaction state."""

        if self._execution_phase != "idle" or self.persistence_phase in {"collecting", "sealed"}:
            raise RuntimeError(
                "FrontRES read-only collection cannot begin from "
                f"execution={self._execution_phase!r} persistence={self.persistence_phase!r}"
            )
        if (
            self._collection_route
            or self._sample is not None
            or self._batch is not None
            or self._observation_trace
            or self._preupdate_diagnostics
        ):
            raise RuntimeError("FrontRES read-only collection cannot begin with stale collection context")
        # B1: 只占用 execution lifecycle, 保留 committed receipt 和 checkpoint identity.
        self._execution_phase = "evaluating"

    def bind_plan(self, identity: Mapping[str, Any]) -> None:
        if self._execution_phase != "collecting" or self.persistence_phase != "collecting":
            raise RuntimeError("FrontRES transaction plan requires active collection")
        self._persisted = {"state": "collecting", **dict(identity)}

    def bind_collection_context(self, *, route: str, sample: Any, batch: Any) -> None:
        training_collection = self._execution_phase == "collecting" and self.persistence_phase == "collecting"
        readonly_collection = self._execution_phase == "evaluating" and self.persistence_phase not in {
            "collecting",
            "sealed",
        }
        if not training_collection and not readonly_collection:
            raise RuntimeError("FrontRES collection context requires active collection")
        allowed_routes = {"sentinel", "training"} if training_collection else {"policy_quality"}
        if route not in allowed_routes or sample is None or batch is None:
            raise ValueError("FrontRES collection context requires one explicit route, sample, and batch")
        if self._batch is not None:
            raise RuntimeError("FrontRES collection context is already bound")
        self._collection_route = route
        self._sample = sample
        self._batch = batch
        self._observation_trace = {}
        self._preupdate_diagnostics = {}

    @property
    def collection_route(self) -> str:
        return self._collection_route

    @property
    def collection_sample(self) -> Any | None:
        return self._sample

    @property
    def collection_batch(self) -> Any | None:
        return self._batch

    def update_observation_trace(self, **values: Any) -> None:
        if self._batch is None:
            raise RuntimeError("FrontRES observation trace requires a bound collection context")
        self._observation_trace = {**self._observation_trace, **values}

    def observation_trace(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self._observation_trace))

    def publish_preupdate_diagnostics(self, values: Mapping[str, Any]) -> None:
        if self._batch is None or self._preupdate_diagnostics:
            raise RuntimeError("FrontRES pre-update diagnostics require one bound, unpublished context")
        self._preupdate_diagnostics = dict(values)

    def preupdate_diagnostics(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self._preupdate_diagnostics))

    def clear_collection_context(self) -> None:
        self._collection_route = ""
        self._sample = None
        self._batch = None
        self._observation_trace = {}
        self._preupdate_diagnostics = {}

    def seal(self, *, collected_policy_attempt_count: int) -> None:
        if self._execution_phase != "collecting" or self.persistence_phase != "collecting":
            raise RuntimeError("FrontRES transaction seal requires active collection")
        self._persisted = {
            **self._persisted,
            "state": "sealed",
            "collected_policy_attempt_count": int(collected_policy_attempt_count),
        }

    def begin_commit(self) -> None:
        if self._execution_phase != "collecting":
            raise RuntimeError("FrontRES transaction commit requires active collection")
        self._execution_phase = "committing"

    def commit(self, receipt: Mapping[str, Any]) -> None:
        if self._execution_phase != "committing" or self.persistence_phase != "sealed":
            raise RuntimeError("FrontRES transaction receipt requires one sealed commit")
        self._persisted = {"state": "committed", "receipt": dict(receipt)}

    def abort(self) -> None:
        self._execution_phase = "idle"
        self._persisted = {"state": "idle"}
        self.clear_collection_context()

    def finish_execution(self, *, succeeded: bool) -> None:
        self._execution_phase = "idle"
        if succeeded and self.persistence_phase == "collecting":
            self._persisted = {"state": "idle"}

    def finish_readonly_collection(self) -> None:
        """Close evaluation context while preserving its pre-existing receipt."""

        if self._execution_phase != "evaluating":
            raise RuntimeError("FrontRES read-only collection close requires active evaluation")
        # B1: 清除每个 held-out transaction 的临时 carrier, 恢复 idle execution.
        self.clear_collection_context()
        self._execution_phase = "idle"

    def as_dict(self) -> dict[str, Any]:
        result = dict(self._persisted)
        if isinstance(result.get("receipt"), Mapping):
            result["receipt"] = dict(result["receipt"])
        return result

    def __getitem__(self, key: str) -> Any:
        value = self._persisted[key]
        return dict(value) if isinstance(value, Mapping) else value

    def __iter__(self) -> Iterator[str]:
        return iter(self._persisted)

    def __len__(self) -> int:
        return len(self._persisted)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and self.as_dict() == dict(other)


def frontres_stage3_transaction_aggregate(runner: Any) -> FrontRESStage3TransactionAggregate:
    """Resolve the runner projection owned by its Stage-3 engine."""

    engine = getattr(runner, _ENGINE_ATTR, None)
    if isinstance(engine, FrontRESStage3Engine):
        return engine.transaction
    existing = getattr(runner, _TRANSACTION_STATE_ATTR, None)
    if isinstance(existing, FrontRESStage3TransactionAggregate):
        return existing
    if existing is not None and not isinstance(existing, Mapping):
        raise RuntimeError("FrontRES transaction state attribute has a foreign value")
    aggregate = FrontRESStage3TransactionAggregate(existing)
    setattr(runner, _TRANSACTION_STATE_ATTR, aggregate)
    return aggregate


@dataclass(frozen=True)
class FrontRESStage3Bindings:
    """Concrete outer-layer functions selected by the composition root."""

    rejected_error: type[BaseException]
    abort_collection: Callable[[Any], None]
    build_request: Callable[..., FrontRESStage3Request]
    close_request: Callable[[Any], None]
    open_barrier: Callable[[Any], None]
    commit_update: Callable[[Any, FrontRESStage3Request], object]

    def validate(self) -> None:
        if not isinstance(self.rejected_error, type) or not issubclass(self.rejected_error, BaseException):
            raise TypeError("FrontRES Stage-3 bindings require one rejected-evidence exception type")
        for name in ("abort_collection", "build_request", "close_request", "open_barrier", "commit_update"):
            if not callable(getattr(self, name)):
                raise TypeError(f"FrontRES Stage-3 binding {name} must be callable")


class MosaicFrontRESStage3Backend:
    """Adapter from the frozen MOSAIC runner to existing FrontRES owners."""

    def __init__(self, runner: Any, bindings: FrontRESStage3Bindings) -> None:
        self.runner = runner
        if not isinstance(bindings, FrontRESStage3Bindings):
            raise TypeError("MOSAIC FrontRES backend requires explicit Stage-3 bindings")
        bindings.validate()
        self.bindings = bindings

    @property
    def _algorithm(self) -> Any:
        algorithm = getattr(self.runner, "alg", None)
        if algorithm is None:
            raise RuntimeError("FrontRES Stage-3 backend requires runner.alg")
        return algorithm

    def optimizer_step_count(self) -> int:
        optimizer = getattr(self._algorithm, "optimizer", None)
        for name in ("frontres_step_count", "step_count"):
            value = getattr(optimizer, name, None)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return int(value)
        raise RuntimeError("FrontRES Stage-3 optimizer requires an explicit nonnegative step counter")

    def open_transaction_barrier(self) -> None:
        self.bindings.open_barrier(self.runner)

    def build_training_request(self, *, init_at_random_ep_len: bool) -> FrontRESStage3Request:
        return self.bindings.build_request(
            self.runner,
            init_at_random_ep_len=bool(init_at_random_ep_len),
        )

    def commit_transaction(self, request: FrontRESStage3Request) -> object:
        return self.bindings.commit_update(self.runner, request)

    def abort_training_collection(self) -> None:
        self.bindings.abort_collection(self.runner)

    def close_training_request(self) -> None:
        self.bindings.close_request(self.runner)

    def is_rejected_evidence(self, error: BaseException) -> bool:
        return isinstance(error, self.bindings.rejected_error)

    def formal_transaction_enabled(self) -> bool:
        return bool(getattr(self._algorithm, "frontres_formal_transaction_enabled", False))

    def formal_training_enabled(self) -> bool:
        return bool(getattr(self._algorithm, "frontres_segment_live_train_enabled", False))

    def sentinel_only(self) -> bool:
        return bool(getattr(self._algorithm, "frontres_local_sentinel_only", False))


class FrontRESStage3Engine:
    """Fail-closed transaction state machine independent of simulator details."""

    def __init__(self, runner: Any, backend: FrontRESStage3Backend) -> None:
        self.runner = runner
        self.backend = backend
        if not isinstance(self.backend, FrontRESStage3Backend):
            raise TypeError("FrontRES Stage-3 backend does not implement the required port")
        existing = getattr(runner, _TRANSACTION_STATE_ATTR, None)
        if isinstance(existing, FrontRESStage3TransactionAggregate):
            self.transaction = existing
        else:
            if existing is not None and not isinstance(existing, Mapping):
                raise RuntimeError("FrontRES transaction state attribute has a foreign value")
            self.transaction = FrontRESStage3TransactionAggregate(existing)
            setattr(runner, _TRANSACTION_STATE_ATTR, self.transaction)

    @property
    def lifecycle_state(self) -> str:
        return self.transaction.execution_phase

    def run_transaction(self, provider: FrontRESTransactionProvider) -> object:
        """Collect one sealed request and commit exactly one optimizer update."""

        if not self.backend.formal_transaction_enabled():
            raise RuntimeError("v015 formal transaction requires its explicit transaction flag")
        if not callable(provider):
            raise RuntimeError("v015 formal transaction requires a callable request provider")
        if self.lifecycle_state != "idle":
            raise RuntimeError(f"v015 formal transaction cannot enter from {self.lifecycle_state!r}")

        self.transaction.begin_collection()
        before_collection = self.backend.optimizer_step_count()
        succeeded = False
        try:
            self.backend.open_transaction_barrier()
            request = provider()
            if not isinstance(request, FrontRESStage3Request):
                raise TypeError("v015 formal provider must return the typed FrontRES Stage-3 request port")
            request_view = request.frontres_stage3_request_view()
            if not isinstance(request_view, FrontRESActiveTransactionRequestView):
                raise TypeError("v015 formal request port returned a foreign request view")
            request_view.validate()
            after_collection = self.backend.optimizer_step_count()
            if after_collection != before_collection:
                raise RuntimeError(
                    "optimizer step occurred while v015 formal transaction collected attempts: "
                    f"before={before_collection} after={after_collection}"
                )

            result = self.backend.commit_transaction(request)
            view = FrontRESActiveCommittedUpdateView.from_result(result)
            try:
                view.validate(expected_request=request_view)
            except ValueError as error:
                raise RuntimeError("v015 committed receipt diverged from its typed request") from error
            if view.optimizer_step_before != before_collection:
                raise RuntimeError(
                    "v015 committed result step identity drifted from collection boundary: "
                    f"collection_before={before_collection} result_before={view.optimizer_step_before}"
                )
            succeeded = True
            return result
        except Exception:
            # Only pre-update failures are rollback-safe. If commit already
            # stepped, preserve the first-invalid fact instead of pretending
            # the optimizer mutation was reversible.
            if self.backend.optimizer_step_count() == before_collection:
                self.backend.abort_training_collection()
                self.transaction.abort()
            raise
        finally:
            self.transaction.finish_execution(succeeded=succeeded)

    def run_training_transaction(self, *, init_at_random_ep_len: bool = True) -> object:
        """Recollect invalid evidence without ever partially updating the policy."""

        if not self.backend.formal_training_enabled():
            raise RuntimeError("v015 formal training dispatch requires ordinary live training")
        if self.backend.sentinel_only():
            raise RuntimeError("v015 formal training dispatch rejects sentinel mode")

        first_step_count = self.backend.optimizer_step_count()
        last_rejection: Exception | None = None
        try:
            for rejection_count in range(_MAX_REJECTED_COLLECTIONS + 1):
                try:
                    return self.run_transaction(
                        lambda: self.backend.build_training_request(
                            init_at_random_ep_len=bool(init_at_random_ep_len)
                        )
                    )
                except Exception as error:
                    if not self.backend.is_rejected_evidence(error):
                        raise
                    last_rejection = error
                    current_step_count = self.backend.optimizer_step_count()
                    if current_step_count != first_step_count:
                        raise RuntimeError(
                            "v015 rejected transaction changed optimizer state before recollection: "
                            f"before={first_step_count} after={current_step_count}"
                        ) from error
                    if rejection_count >= _MAX_REJECTED_COLLECTIONS:
                        break
                    print(
                        "[FrontRES v015 Transaction Rejected] "
                        f"rejection={rejection_count + 1}/{_MAX_REJECTED_COLLECTIONS} "
                        f"optimizer_step_delta=0 reason={error}",
                        flush=True,
                    )
            raise RuntimeError(
                "v015 formal training exhausted its bounded invalid-evidence recollection budget "
                f"({_MAX_REJECTED_COLLECTIONS}); optimizer_step_delta=0; last={last_rejection}"
            ) from last_rejection
        finally:
            self.backend.close_training_request()


def get_frontres_stage3_engine(
    runner: Any,
    *,
    backend: FrontRESStage3Backend | None = None,
    backend_factory: Callable[[], FrontRESStage3Backend] | None = None,
) -> FrontRESStage3Engine:
    """Resolve exactly one engine instance for a MOSAIC runner."""

    existing = getattr(runner, _ENGINE_ATTR, None)
    if existing is not None:
        if not isinstance(existing, FrontRESStage3Engine):
            raise RuntimeError(f"reserved FrontRES engine attribute {_ENGINE_ATTR!r} has a foreign value")
        if existing.runner is not runner:
            raise RuntimeError("FrontRES Stage-3 engine is bound to a different runner")
        if backend is not None and existing.backend is not backend:
            raise RuntimeError("FrontRES Stage-3 engine is already bound to a different backend")
        return existing

    if backend is not None and backend_factory is not None:
        raise ValueError("FrontRES Stage-3 engine accepts backend or backend_factory, not both")
    if backend is None:
        if backend_factory is None:
            raise RuntimeError("FrontRES Stage-3 engine requires explicit outer-layer backend composition")
        backend = backend_factory()
    engine = FrontRESStage3Engine(runner, backend)
    setattr(runner, _ENGINE_ATTR, engine)
    return engine


__all__ = [
    "FrontRESStage3Engine",
    "FrontRESStage3TransactionAggregate",
    "FrontRESStage3Bindings",
    "MosaicFrontRESStage3Backend",
    "frontres_stage3_transaction_aggregate",
    "get_frontres_stage3_engine",
]
