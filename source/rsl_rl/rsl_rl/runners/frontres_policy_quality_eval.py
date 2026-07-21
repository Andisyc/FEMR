"""State-isolation helpers for the independent FrontRES policy-quality evaluator."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
import pickle
import random
from typing import Any, Callable, Literal, Mapping

import numpy as np
import torch

from rsl_rl.frontres.frontres_policy_quality_manifest import (
    FrontRESPolicyQualityManifest,
    FrontRESPolicyQualityRouteIdentity,
    FrontRESPolicyQualityStateIdentity,
    FrontRESV015PolicyQualityManifest,
)


_COMMAND_STATE_FIELDS = (
    "time_steps",
    "env_motion_indices",
    "_cached_perturbed_pos",
    "_cached_perturbed_quat",
    "_frontres_pos_correction",
    "_frontres_quat_correction",
)


@dataclass(frozen=True)
class _TensorImage:
    dtype: str
    shape: tuple[int, ...]
    data: bytes

    @classmethod
    def capture(cls, tensor: torch.Tensor) -> _TensorImage:
        value = tensor.detach().contiguous().cpu()
        return cls(dtype=str(value.dtype), shape=tuple(value.shape), data=value.numpy().tobytes(order="C"))

    def restore(self, *, device: torch.device | str) -> torch.Tensor:
        dtype_name = self.dtype.removeprefix("torch.")
        dtype = getattr(torch, dtype_name, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"unsupported snapshot dtype: {self.dtype}")
        value = torch.frombuffer(bytearray(self.data), dtype=dtype).clone()
        return value.reshape(self.shape).to(device=device)

    def update_hash(self, digest: Any, *, name: str) -> None:
        digest.update(name.encode("utf-8"))
        digest.update(self.dtype.encode("ascii"))
        digest.update(repr(self.shape).encode("ascii"))
        digest.update(self.data)


@dataclass(frozen=True)
class FrontRESPolicyQualityScoringState:
    comparison_signature: str
    env_ids: tuple[int, ...]
    role_layout: tuple[str, ...]
    root_state_w: _TensorImage
    joint_pos: _TensorImage
    joint_vel: _TensorImage
    env_origins: _TensorImage
    episode_length: _TensorImage
    command_state: tuple[tuple[str, _TensorImage], ...]
    perturber_state: tuple[tuple[str, _TensorImage], ...]
    python_rng_state: bytes
    numpy_rng_state: bytes
    torch_rng_state: _TensorImage
    cuda_rng_state: tuple[_TensorImage, ...]

    @property
    def initial_state_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.comparison_signature.encode("ascii"))
        digest.update(repr(self.env_ids).encode("ascii"))
        digest.update(repr(self.role_layout).encode("utf-8"))
        for name, image in (
            ("root_state_w", self.root_state_w),
            ("joint_pos", self.joint_pos),
            ("joint_vel", self.joint_vel),
            ("env_origins", self.env_origins),
            ("episode_length", self.episode_length),
            *self.command_state,
            *self.perturber_state,
            ("torch_rng_state", self.torch_rng_state),
        ):
            image.update_hash(digest, name=name)
        for index, image in enumerate(self.cuda_rng_state):
            image.update_hash(digest, name=f"cuda_rng_state[{index}]")
        digest.update(self.python_rng_state)
        digest.update(self.numpy_rng_state)
        return digest.hexdigest()

    @property
    def state_identity(self) -> FrontRESPolicyQualityStateIdentity:
        return FrontRESPolicyQualityStateIdentity(
            comparison_signature=self.comparison_signature,
            initial_state_hash=self.initial_state_hash,
        )


@dataclass(frozen=True)
class FrontRESPolicyQualityObservationIdentity:
    expected_obs_dim: int
    actor_input_dim: int
    normalizer_identity: str

    def __post_init__(self) -> None:
        if self.expected_obs_dim <= 0 or not 0 < self.actor_input_dim <= self.expected_obs_dim:
            raise ValueError("quality observation dimensions must satisfy 0 < actor_input_dim <= expected_obs_dim")
        if not self.normalizer_identity.strip():
            raise ValueError("normalizer_identity must be explicit")


@dataclass(frozen=True)
class FrontRESPolicyQualityRouteResult:
    identity: FrontRESPolicyQualityRouteIdentity
    observation_identity: FrontRESPolicyQualityObservationIdentity
    actions: torch.Tensor
    gain: Any
    execution: Any


@dataclass(frozen=True)
class FrontRESPolicyQualityRouteHooks:
    observe: Callable[[], torch.Tensor]
    apply_action: Callable[[torch.Tensor], Any]
    step: Callable[[], Any]
    compute_gain: Callable[[], Any]
    capture_execution: Callable[[], Any]
    begin_route: Callable[[str], None] | None = None
    set_audit_identity: Callable[[Mapping[str, str]], None] | None = None


@dataclass(frozen=True)
class FrontRESPolicyQualityEvalRequest:
    manifest_path: str
    hsl_checkpoint_path: str
    policy_checkpoint_path: str
    result_path: str
    manifest: FrontRESPolicyQualityManifest


@dataclass(frozen=True)
class FrontRESV015PolicyQualityEvalRequest:
    """Pre-mutation v015 manifest/checkpoint identity consumed only by the S2B owner."""

    manifest_path: str
    hsl_checkpoint_path: str
    policy_checkpoint_path: str
    result_path: str
    manifest: FrontRESV015PolicyQualityManifest
    manifest_file_sha256: str
    hsl_checkpoint: Any
    policy_checkpoint: Any


_V015_QUALITY_OWNER_IDENTITY = (
    ("reset", "frontres_segment_stage1_env_hooks"),
    ("observation", "frontres_runtime"),
    ("one_action_k", "frontres_segment_live_probe.collect_frontres_v015_one_action_k_evidence"),
    ("gain", "frontres_gain.compute_intent_physics_local_repair_gain"),
)
_V015_QUALITY_ROUTES = ("zero", "hsl", "policy")
_V015_QUALITY_REPORT_SCHEMA = "frontres-v015-heldout-quality-report-v1"
_V015_GAIN_SOURCE = "FRS-GAIN-v003-intent-physics-local-repair"
_V015_DYNAMIC_STATE_FIELDS = (
    "root_state_w",
    "joint_pos",
    "joint_vel",
    "env_origins",
    "episode_length",
    "command_state",
    "perturber_state",
    "python_rng_state",
    "numpy_rng_state",
    "torch_rng_state",
    "cuda_rng_state",
    "local_scenario",
)


@dataclass(frozen=True)
class FrontRESV015DynamicStateIdentity:
    """Read-only post-reset identity captured before observation or action."""

    comparison_signature: str
    role_layout: tuple[str, ...]
    field_hashes: tuple[tuple[str, str], ...]

    def validate(self) -> None:
        if (
            len(self.comparison_signature) != 64
            or any(char not in "0123456789abcdef" for char in self.comparison_signature)
        ):
            raise ValueError("v015 dynamic-state identity requires the manifest comparison signature")
        if not self.role_layout or any(role not in {"repair", "noisy"} for role in self.role_layout):
            raise ValueError("v015 dynamic-state identity requires Repair/Noisy roles only")
        names = tuple(name for name, _value in self.field_hashes)
        if names != _V015_DYNAMIC_STATE_FIELDS:
            raise ValueError("v015 dynamic-state identity has an incomplete or reordered field schema")
        if any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for _name, value in self.field_hashes
        ):
            raise ValueError("v015 dynamic-state identity field hashes must be SHA-256 values")

    @property
    def full_state_hash(self) -> str:
        self.validate()
        digest = hashlib.sha256()
        digest.update(self.comparison_signature.encode("ascii"))
        digest.update(repr(self.role_layout).encode("ascii"))
        for name, value in self.field_hashes:
            digest.update(name.encode("ascii"))
            digest.update(value.encode("ascii"))
        return digest.hexdigest()

    def to_json(self) -> dict[str, Any]:
        return {
            "full_state_hash": self.full_state_hash,
            "role_layout": list(self.role_layout),
            "field_hashes": dict(self.field_hashes),
        }


@dataclass(frozen=True)
class FrontRESV015PolicyQualityOwnerBundle:
    """S2 connector for active local owners; evaluator owns no training state.

    Status: active formal owner, contract-confirmed through G5-Q1 S1/S2.
    Evidence: matched full-state identity, zero-write inference, and exact item-close.
    Gap: the post-reset identity hash remains pending one bounded S4 sentinel.
    """

    owner_identity: tuple[tuple[str, str], ...]
    collect_one_action_k: Callable[[Any, Any, str], Any]
    close_item: Callable[[Any, Any], None]
    training_state_signature: Callable[[Any], str]

    def __post_init__(self) -> None:
        if tuple(self.owner_identity) != _V015_QUALITY_OWNER_IDENTITY:
            raise ValueError("v015 quality owner identity must name only the active reset/observation/one-action-K/v003 path")
        if (
            not callable(self.collect_one_action_k)
            or not callable(self.close_item)
            or not callable(self.training_state_signature)
        ):
            raise TypeError("v015 quality owner bundle requires collect, item-close, and state-signature callables")


@dataclass(frozen=True)
class FrontRESV015PolicyQualityRouteEvidence:
    """Bind one route/checkpoint identity to one active one-action-K carrier."""

    route: str
    checkpoint_file_sha256: str
    comparison_signature: str
    one_action_k: Any
    dynamic_state_identity: FrontRESV015DynamicStateIdentity

    def validate(self) -> None:
        if self.route not in _V015_QUALITY_ROUTES:
            raise ValueError("v015 quality route evidence has an invalid route")
        if self.route == "zero":
            if self.checkpoint_file_sha256 != "zero":
                raise ValueError("v015 zero route must not claim a checkpoint")
        elif (
            len(self.checkpoint_file_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.checkpoint_file_sha256)
        ):
            raise ValueError("v015 HSL/policy route requires an exact checkpoint SHA-256")
        if (
            len(self.comparison_signature) != 64
            or any(char not in "0123456789abcdef" for char in self.comparison_signature)
        ):
            raise ValueError("v015 quality route evidence requires the manifest item comparison signature")
        if not isinstance(self.dynamic_state_identity, FrontRESV015DynamicStateIdentity):
            raise TypeError("v015 quality route requires a full dynamic-state identity")
        self.dynamic_state_identity.validate()
        if self.dynamic_state_identity.comparison_signature != self.comparison_signature:
            raise ValueError("v015 quality route dynamic-state comparison signature is mixed")
        validate = getattr(self.one_action_k, "validate", None)
        if not callable(validate):
            raise TypeError("v015 route evidence requires validated one-action-K evidence")
        validate()
        if (
            tuple(self.one_action_k.policy_observations.shape) != (int(self.one_action_k.policy_actions.shape[0]), 928)
            or tuple(self.one_action_k.policy_privileged_observations.shape)
            != (int(self.one_action_k.policy_actions.shape[0]), 289)
            or tuple(self.one_action_k.policy_actions.shape) != (int(self.one_action_k.policy_actions.shape[0]), 6)
        ):
            raise ValueError("v015 quality route requires policy/critic/action shapes [B,928]/[B,289]/[B,6]")
        if self.dynamic_state_identity.role_layout != tuple(self.one_action_k.roles):
            raise ValueError("v015 quality route dynamic-state rows are not aligned with Repair/Noisy evidence rows")


def install_frontres_v015_policy_quality_owner_bundle(
    runner: Any,
    bundle: FrontRESV015PolicyQualityOwnerBundle,
) -> None:
    """Install one immutable S2 connector; legacy executor attributes are ignored."""

    if not isinstance(bundle, FrontRESV015PolicyQualityOwnerBundle):
        raise TypeError("v015 policy-quality requires FrontRESV015PolicyQualityOwnerBundle")
    if hasattr(runner, "_frontres_v015_policy_quality_owner_bundle"):
        raise RuntimeError("v015 policy-quality owner bundle is already installed")
    runner._frontres_v015_policy_quality_owner_bundle = bundle


def _v015_quality_hash_state(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, Mapping):
        for key in sorted(value, key=repr):
            digest.update(repr(key).encode("utf-8"))
            _v015_quality_hash_state(digest, value[key])
    elif isinstance(value, (tuple, list)):
        digest.update(type(value).__name__.encode("ascii"))
        for item in value:
            _v015_quality_hash_state(digest, item)
    else:
        digest.update(repr(value).encode("utf-8"))


def _v015_quality_field_hash(name: str, value: Any) -> str:
    digest = hashlib.sha256()
    digest.update(name.encode("ascii"))
    if isinstance(value, _TensorImage):
        value.update_hash(digest, name=name)
    elif isinstance(value, bytes):
        digest.update(value)
    elif isinstance(value, tuple) and all(isinstance(item, _TensorImage) for item in value):
        for index, image in enumerate(value):
            image.update_hash(digest, name=f"{name}[{index}]")
    elif isinstance(value, tuple) and all(
        isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], _TensorImage)
        for item in value
    ):
        for field_name, image in value:
            image.update_hash(digest, name=str(field_name))
    else:
        _v015_quality_hash_state(digest, value)
    return digest.hexdigest()


@contextmanager
def _frontres_v015_quality_inference_mode(runner: Any):
    """Freeze every observation/policy module mode for held-out inference.

    Directly restoring each submodule flag preserves mixed source modes such
    as a training residual actor with an already frozen GMT normalizer.
    """

    roots = (
        getattr(getattr(runner, "alg", None), "policy", None),
        getattr(runner, "_frontres_extra_normalizer", None),
        getattr(runner, "obs_normalizer", None),
        getattr(runner, "privileged_obs_normalizer", None),
        getattr(runner, "teacher_obs_normalizer", None),
    )
    module_modes: dict[torch.nn.Module, bool] = {}
    for root in roots:
        if not isinstance(root, torch.nn.Module):
            continue
        for module in root.modules():
            module_modes.setdefault(module, bool(module.training))
    for module in module_modes:
        module.training = False
    try:
        yield
    finally:
        for module, was_training in module_modes.items():
            module.training = was_training


def _v015_quality_training_state_signature(runner: Any) -> str:
    """Hash every mutable training owner while excluding physical env state."""

    digest = hashlib.sha256()
    alg = getattr(runner, "alg", None)
    policy = getattr(alg, "policy", None)
    for name, value in (
        ("actor", getattr(getattr(policy, "residual_actor", None), "state_dict", lambda: {})()),
        ("critic", getattr(getattr(policy, "critic", None), "state_dict", lambda: {})()),
        ("optimizer", getattr(getattr(alg, "optimizer", None), "state_dict", lambda: {})()),
        (
            "prefix_normalizer",
            getattr(getattr(runner, "_frontres_extra_normalizer", None), "state_dict", lambda: {})(),
        ),
        (
            "gmt_normalizer",
            getattr(getattr(runner, "obs_normalizer", None), "state_dict", lambda: {})(),
        ),
        (
            "privileged_normalizer",
            getattr(getattr(runner, "privileged_obs_normalizer", None), "state_dict", lambda: {})(),
        ),
        (
            "teacher_normalizer",
            getattr(getattr(runner, "teacher_obs_normalizer", None), "state_dict", lambda: {})(),
        ),
        ("prefix_mean", getattr(runner, "_frontres_extra_mean", None)),
        ("prefix_std", getattr(runner, "_frontres_extra_std", None)),
        ("prefix_layout", getattr(runner, "_frontres_extra_stats_layout_version", None)),
        (
            "sampler",
            getattr(getattr(runner, "_frontres_segment_sampler", None), "state_dict", lambda: {})(),
        ),
        ("transaction", getattr(runner, "_frontres_v015_checkpoint_transaction_state", None)),
        ("receipt", getattr(runner, "_frontres_v015_last_committed_transaction_receipt", None)),
        ("warmup", getattr(runner, "_frontres_warmup_complete", None)),
        ("iteration", getattr(runner, "current_learning_iteration", None)),
    ):
        digest.update(name.encode("ascii"))
        _v015_quality_hash_state(digest, value)
    return digest.hexdigest()


def build_frontres_v015_policy_quality_owner_bundle(
    runner: Any,
    request: FrontRESV015PolicyQualityEvalRequest,
) -> FrontRESV015PolicyQualityOwnerBundle:
    """Bind the formal runner to existing v015 scenario, reset, K, and Gain owners."""

    if not isinstance(request, FrontRESV015PolicyQualityEvalRequest):
        raise TypeError("formal v015 quality owner requires the strict request")
    if not bool(getattr(getattr(runner, "alg", None), "frontres_v015_formal_transaction_enabled", False)):
        raise RuntimeError("formal v015 quality owner requires the active grouped transaction configuration")
    from rsl_rl.runners.frontres_checkpointing import frontres_v015_quality_route_actor
    from rsl_rl.runners.frontres_segment_live_probe import (
        _apply_current_segment_reset,
        _read_live_observations,
        collect_frontres_v015_one_action_k_evidence,
    )
    from rsl_rl.runners.frontres_segment_live_sampler import (
        _close_frontres_local_scenarios,
        prepare_frontres_v015_policy_quality_item_batch,
    )
    from rsl_rl.runners.frontres_training_setup import configure_frontres_pair_layout

    pair_layout = configure_frontres_pair_layout(runner, is_frontres=True)
    if (
        int(getattr(pair_layout, "n_train", 0)) != 4
        or int(getattr(pair_layout, "n_base", 0)) != 4
        or int(getattr(pair_layout, "n_candidate", 0)) != 0
        or int(getattr(pair_layout, "n_clean", 0)) != 0
    ):
        raise RuntimeError("formal v015 held-out quality requires exactly 4 Repair + 4 Noisy rows")
    item_batches: dict[str, Any] = {}

    def collect_one_action_k(_runner: Any, item: Any, route: str) -> FrontRESV015PolicyQualityRouteEvidence:
        if _runner is not runner or route not in _V015_QUALITY_ROUTES:
            raise RuntimeError("v015 quality owner received a mixed runner or route identity")
        signature = str(item.comparison_signature)
        prepared = item_batches.get(signature)
        if prepared is None:
            prepared = prepare_frontres_v015_policy_quality_item_batch(runner, item)
            item_batches[signature] = prepared
        runner._frontres_segment_live_current_batch = prepared.batch
        runner._frontres_segment_live_current_sample = prepared.sample
        reset = _apply_current_segment_reset(runner, pair_layout=pair_layout)
        if reset is None or not bool(reset.success_mask.detach().bool().all().item()):
            raise RuntimeError("v015 held-out quality failed to restore the sealed Clean x_t")
        dynamic_state_identity = capture_frontres_v015_policy_quality_dynamic_state_identity(
            runner,
            comparison_signature=signature,
            pair_layout=pair_layout,
        )
        observations = _read_live_observations(runner)
        checkpoint_sha = {
            "zero": "zero",
            "hsl": request.hsl_checkpoint.file_sha256,
            "policy": request.policy_checkpoint.file_sha256,
        }[route]
        checkpoint_path = {
            "hsl": request.hsl_checkpoint_path,
            "policy": request.policy_checkpoint_path,
        }.get(route)
        runner._frontres_v015_quality_action_route = route
        try:
            if checkpoint_path is None:
                evidence = collect_frontres_v015_one_action_k_evidence(
                    runner,
                    observations,
                    pair_layout=pair_layout,
                )
            else:
                with frontres_v015_quality_route_actor(
                    runner,
                    checkpoint_path,
                    route=route,
                    expected_file_sha256=checkpoint_sha,
                ):
                    evidence = collect_frontres_v015_one_action_k_evidence(
                        runner,
                        observations,
                        pair_layout=pair_layout,
                    )
        finally:
            if hasattr(runner, "_frontres_v015_quality_action_route"):
                delattr(runner, "_frontres_v015_quality_action_route")
        return FrontRESV015PolicyQualityRouteEvidence(
            route=route,
            checkpoint_file_sha256=checkpoint_sha,
            comparison_signature=signature,
            one_action_k=evidence,
            dynamic_state_identity=dynamic_state_identity,
        )

    def close_item(_runner: Any, item: Any) -> None:
        """Close one manifest item's command carrier after all counterfactual routes."""

        if _runner is not runner:
            raise RuntimeError("v015 quality item close received a mixed runner identity")
        prepared = item_batches.pop(str(item.comparison_signature), None)
        if prepared is None:
            return
        env = runner.env.unwrapped if hasattr(runner.env, "unwrapped") else runner.env
        manager = getattr(env, "command_manager", None)
        get_term = getattr(manager, "get_term", None)
        command = get_term("motion") if callable(get_term) else getattr(manager, "_terms", {}).get("motion")
        try:
            clear = getattr(command, "clear_frontres_local_scenario", None)
            if not callable(clear):
                raise RuntimeError("v015 quality item close requires command-owned local-scenario lifecycle")
            clear()
        finally:
            try:
                _close_frontres_local_scenarios(prepared.batch)
            finally:
                if getattr(runner, "_frontres_segment_live_current_batch", None) is prepared.batch:
                    runner._frontres_segment_live_current_batch = None
                    runner._frontres_segment_live_current_sample = None

    return FrontRESV015PolicyQualityOwnerBundle(
        owner_identity=_V015_QUALITY_OWNER_IDENTITY,
        collect_one_action_k=collect_one_action_k,
        close_item=close_item,
        training_state_signature=_v015_quality_training_state_signature,
    )


@dataclass(frozen=True)
class FrontRESPolicyQualityFormalOwnerBundle:
    owner_identity: tuple[tuple[str, str], ...]
    prepare_item: Callable[[Any, Any, FrontRESPolicyQualityEvalRequest], tuple[Any, Any, Any]]
    isolation_state: Callable[[Any], str]
    serialize_result: Callable[[Any, tuple[FrontRESPolicyQualityRouteResult, ...]], Mapping[str, Any]]

    def __post_init__(self) -> None:
        owners = dict(self.owner_identity)
        required = {"reset", "observation", "action", "rollout", "gain", "execution"}
        if set(owners) != required or any(not str(value).strip() for value in owners.values()):
            raise ValueError(f"quality formal owner bundle must name exactly {sorted(required)}")


def install_frontres_policy_quality_manifest_executor(
    runner: Any,
    owners: FrontRESPolicyQualityFormalOwnerBundle,
) -> None:
    """Install the only manifest executor; owner callbacks remain module-owned and explicit."""
    # QUALITY-TRAJECTORY-01: 检查 immutable manifest -> ordered checkpoint/item result rows.
    # Result: PENDING_Q_EVIDENCE; Q-E5 only proves offline executor order/schema.
    # B1: manifest item order 是 checkpoint trajectory 的固定比较轴.
    # B2: 每个 item 只调用一次 matched zero/HSL/policy executor.
    # B3: 原子 result artifact 保留 comparison signature 与 owner identity.
    if hasattr(runner, "_frontres_policy_quality_manifest_executor"):
        raise RuntimeError("policy-quality manifest executor is already configured")

    def execute(request: FrontRESPolicyQualityEvalRequest) -> dict[str, Any]:
        isolated_before = owners.isolation_state(runner)
        rows: list[Mapping[str, Any]] = []
        for item in request.manifest.items:
            snapshot, adapters, hooks = owners.prepare_item(runner, item, request)
            results = run_frontres_policy_quality_counterfactuals(
                runner,
                snapshot=snapshot,
                comparison_signature=item.comparison_signature,
                adapters=adapters,
                hooks=hooks,
                horizon_k=item.effective_horizon_k,
                isolation_state=lambda: owners.isolation_state(runner),
            )
            rows.append(owners.serialize_result(item, results))
        if owners.isolation_state(runner) != isolated_before:
            raise RuntimeError("quality manifest executor mutated optimizer/sampler/warmup state")
        payload = {
            "schema_version": "frontres_policy_quality_result_v1",
            "comparison_signature": request.manifest.comparison_signature,
            "owner_identity": dict(owners.owner_identity),
            "rows": rows,
        }
        result_path = Path(request.result_path)
        temporary = result_path.with_suffix(result_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(result_path)
        return payload

    runner._frontres_policy_quality_manifest_executor = execute


def build_frontres_policy_quality_eval_request(
    *,
    manifest_path: str,
    hsl_checkpoint_path: str,
    policy_checkpoint_path: str,
    result_path: str,
) -> FrontRESPolicyQualityEvalRequest:
    paths = {
        "manifest_path": Path(manifest_path).expanduser().resolve(),
        "hsl_checkpoint_path": Path(hsl_checkpoint_path).expanduser().resolve(),
        "policy_checkpoint_path": Path(policy_checkpoint_path).expanduser().resolve(),
        "result_path": Path(result_path).expanduser().resolve(),
    }
    for name in ("manifest_path", "hsl_checkpoint_path", "policy_checkpoint_path"):
        if not paths[name].is_file():
            raise FileNotFoundError(f"policy-quality {name} does not exist: {paths[name]}")
    if paths["hsl_checkpoint_path"] == paths["policy_checkpoint_path"]:
        raise ValueError("policy-quality HSL and tested-policy checkpoints must be explicit distinct files")
    if not paths["result_path"].parent.is_dir():
        raise FileNotFoundError(f"policy-quality result directory does not exist: {paths['result_path'].parent}")
    manifest = FrontRESPolicyQualityManifest.from_json(paths["manifest_path"].read_text(encoding="utf-8"))
    return FrontRESPolicyQualityEvalRequest(
        manifest_path=str(paths["manifest_path"]),
        hsl_checkpoint_path=str(paths["hsl_checkpoint_path"]),
        policy_checkpoint_path=str(paths["policy_checkpoint_path"]),
        result_path=str(paths["result_path"]),
        manifest=manifest,
    )


def build_frontres_v015_policy_quality_eval_request(
    *,
    manifest_path: str,
    hsl_checkpoint_path: str,
    policy_checkpoint_path: str,
    result_path: str,
) -> FrontRESV015PolicyQualityEvalRequest:
    """Inspect strict v015 artifacts and freeze their identities without restoring state."""

    paths = {
        "manifest_path": Path(manifest_path).expanduser().resolve(),
        "hsl_checkpoint_path": Path(hsl_checkpoint_path).expanduser().resolve(),
        "policy_checkpoint_path": Path(policy_checkpoint_path).expanduser().resolve(),
        "result_path": Path(result_path).expanduser().resolve(),
    }
    for name in ("manifest_path", "hsl_checkpoint_path", "policy_checkpoint_path"):
        if not paths[name].is_file():
            raise FileNotFoundError(f"v015 policy-quality {name} does not exist: {paths[name]}")
    if paths["hsl_checkpoint_path"] == paths["policy_checkpoint_path"]:
        raise ValueError("v015 policy-quality HSL and Stage-3 checkpoints must be distinct files")
    if not paths["result_path"].parent.is_dir():
        raise FileNotFoundError(f"v015 policy-quality result directory does not exist: {paths['result_path'].parent}")

    manifest_bytes = paths["manifest_path"].read_bytes()
    manifest = FrontRESV015PolicyQualityManifest.from_json(manifest_bytes.decode("utf-8"))
    from rsl_rl.runners.frontres_checkpointing import inspect_frontres_v015_quality_checkpoint

    hsl = inspect_frontres_v015_quality_checkpoint(paths["hsl_checkpoint_path"], route="hsl")
    policy = inspect_frontres_v015_quality_checkpoint(paths["policy_checkpoint_path"], route="policy")
    expected_layout = {
        "layout_version": manifest.future_intent_layout_version,
        "future_offsets": manifest.future_offsets,
        "intent_dim": 29,
        "actor_tail_dim": 58,
        "environment_obs_dim": manifest.raw_observation_dim,
        "current_frontres_prefix_dim": manifest.actor_input_dim - 58,
        "actor_dim": manifest.combined_observation_dim,
        "prefix_dim": manifest.actor_input_dim,
        "gmt_dim": manifest.gmt_suffix_dim,
    }
    if dict(hsl.future_intent_layout) != expected_layout or dict(policy.future_intent_layout) != expected_layout:
        raise ValueError("v015 policy-quality manifest and checkpoint layouts are mixed")
    if (
        hsl.action_kind != manifest.action_kind
        or policy.action_kind != manifest.action_kind
        or hsl.action_dim != manifest.action_dim
        or policy.action_dim != manifest.action_dim
        or hsl.method_contract_id != manifest.method_contract_id
        or policy.method_contract_id != manifest.method_contract_id
        or hsl.training_contract_id != manifest.training_contract_id
        or policy.training_contract_id != manifest.training_contract_id
        or policy.gain_contract_id != manifest.gain_contract_id
        or policy.ppo_contract_id != manifest.ppo_contract_id
    ):
        raise ValueError("v015 policy-quality manifest and checkpoint contract/action identities are mixed")
    return FrontRESV015PolicyQualityEvalRequest(
        manifest_path=str(paths["manifest_path"]),
        hsl_checkpoint_path=str(paths["hsl_checkpoint_path"]),
        policy_checkpoint_path=str(paths["policy_checkpoint_path"]),
        result_path=str(paths["result_path"]),
        manifest=manifest,
        manifest_file_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        hsl_checkpoint=hsl,
        policy_checkpoint=policy,
    )


def _v015_quality_json_tensor(value: torch.Tensor) -> Any:
    """Serialize finite values and preserve unavailable values as null, never zero."""

    def convert(item: Any) -> Any:
        if isinstance(item, list):
            return [convert(value) for value in item]
        number = float(item)
        return number if np.isfinite(number) else None

    return convert(value.detach().to(device="cpu").tolist())


def _v015_quality_require_same_scenario(anchor: Any, candidate: Any) -> None:
    scalar_identity = (
        "scenario_ids",
        "noisy_segment_hashes",
        "x_t_identities",
        "roles",
        "intent_q29_provenance",
        "intent_q29_source",
    )
    if any(tuple(getattr(anchor, name)) != tuple(getattr(candidate, name)) for name in scalar_identity):
        raise RuntimeError("v015 quality routes lost the same sealed scenario identity")
    tensor_identity = (
        "policy_row_indices",
        "continuation",
        "continuation_valid_mask",
        "horizon_k",
        "intent_q29",
    )
    if any(
        not torch.equal(getattr(anchor, name).detach().to(device="cpu"), getattr(candidate, name).detach().to(device="cpu"))
        for name in tensor_identity
    ):
        raise RuntimeError("v015 quality routes mixed scenario intent, Clean continuation, role rows, or K")


def _v015_quality_route_result(
    evidence: Any,
    *,
    route: str,
    checkpoint_file_sha256: str,
    dynamic_state_identity: FrontRESV015DynamicStateIdentity,
) -> dict[str, Any]:
    """Consume one-action evidence through the active v003 Gain owner only."""

    from rsl_rl.frontres.frontres_gain import (
        FrontRESIntentPhysicsGainConfig,
        FrontRESIntentPhysicsGainInput,
        compute_intent_physics_local_repair_gain,
    )
    from rsl_rl.frontres.frontres_segment_storage import (
        FrontRESV015OneActionKEvidence,
        pair_frontres_v015_gain_facts,
    )

    if route not in _V015_QUALITY_ROUTES or not isinstance(evidence, FrontRESV015OneActionKEvidence):
        raise TypeError("v015 quality route requires active one-action-K evidence")
    evidence.validate()
    if tuple(evidence.roles) != tuple(
        "repair" if index < int(evidence.policy_actions.shape[0]) else "noisy"
        for index in range(int(evidence.t_env_actions.shape[0]))
    ):
        raise ValueError("v015 quality route requires ordered Repair/Noisy rows only")
    if route == "zero" and not bool((evidence.policy_actions == 0.0).all().item()):
        raise ValueError("v015 zero quality route requires an exact zero 6D action")
    facts = pair_frontres_v015_gain_facts(evidence)
    gain_input = FrontRESIntentPhysicsGainInput(
        intent_q29=facts.intent_q29,
        repaired_q29=facts.repaired_q29,
        noisy_q29=facts.noisy_q29,
        intent_q29_provenance=facts.intent_q29_provenance,
        intent_q29_source=facts.intent_q29_source,
        repair_action_steps=facts.policy_actions,
        intent_valid_mask=facts.intent_valid_mask,
        repaired_success=facts.repaired_success,
        noisy_success=facts.noisy_success,
        repaired_survival=facts.repaired_survival,
        noisy_survival=facts.noisy_survival,
        effective_horizon_k=facts.horizon_k,
    )
    gain = compute_intent_physics_local_repair_gain(gain_input, config=FrontRESIntentPhysicsGainConfig())
    valid = facts.intent_valid_mask.bool() & gain.available.bool()
    if not bool(valid.any().item()):
        raise RuntimeError("v015 quality route has no valid v003 Gain row")
    components = {
        name: torch.where(valid, getattr(gain, name).detach().float(), torch.full_like(gain.gain_total, float("nan")))
        for name in ("intent_gain", "physics_gain", "repair_cost", "gain_total")
    }
    repair_rows = evidence.policy_row_indices.detach().to(dtype=torch.long)
    return {
        "route": route,
        "checkpoint_file_sha256": checkpoint_file_sha256,
        "gain_source": _V015_GAIN_SOURCE,
        "scenario_ids": list(evidence.scenario_ids),
        "noisy_segment_hashes": list(evidence.noisy_segment_hashes),
        "x_t_identities": list(evidence.x_t_identities),
        "roles": list(evidence.roles),
        "dynamic_state_identity": dynamic_state_identity.to_json(),
        "horizon_k": [int(value) for value in evidence.horizon_k.tolist()],
        "actor_forward_count": int(evidence.actor_forward_count),
        "later_femr_action_count": int(evidence.later_femr_action_count),
        "policy_actions": _v015_quality_json_tensor(evidence.policy_actions),
        "policy_row_valid": [bool(value) for value in valid.tolist()],
        "evidence_valid_step_count": [
            int(value) for value in evidence.survival_steps.index_select(0, repair_rows).tolist()
        ],
        "intent_q29_provenance": facts.intent_q29_provenance,
        "intent_q29_source": facts.intent_q29_source,
        **{name: _v015_quality_json_tensor(value) for name, value in components.items()},
    }


def run_frontres_v015_policy_quality_heldout_eval(
    runner: Any,
    *,
    request: FrontRESV015PolicyQualityEvalRequest,
    owners: FrontRESV015PolicyQualityOwnerBundle,
) -> dict[str, Any]:
    """Run all held-out routes under one reversible inference-mode boundary."""

    with _frontres_v015_quality_inference_mode(runner):
        return _run_frontres_v015_policy_quality_heldout_eval_inference(
            runner,
            request=request,
            owners=owners,
        )


def _run_frontres_v015_policy_quality_heldout_eval_inference(
    runner: Any,
    *,
    request: FrontRESV015PolicyQualityEvalRequest,
    owners: FrontRESV015PolicyQualityOwnerBundle,
) -> dict[str, Any]:
    """Evaluate zero/HSL/policy on matched v015 evidence and atomically report.

    Status: active formal owner with deterministic G5-Q1 S1/S2 identity checks.
    Full-state live equality and policy quality remain separate S4 evidence.
    """

    if not isinstance(request, FrontRESV015PolicyQualityEvalRequest):
        raise TypeError("v015 held-out quality requires the strict G5-S2A request")
    if not isinstance(owners, FrontRESV015PolicyQualityOwnerBundle):
        raise TypeError("v015 held-out quality requires the active owner bundle")
    baseline_state = str(owners.training_state_signature(runner))
    checkpoint_identity = {
        "zero": "zero",
        "hsl": request.hsl_checkpoint.file_sha256,
        "policy": request.policy_checkpoint.file_sha256,
    }
    item_rows: list[dict[str, Any]] = []
    for item in request.manifest.items:
        anchor = None
        dynamic_state_anchor = None
        routes: list[dict[str, Any]] = []
        try:
            for route in _V015_QUALITY_ROUTES:
                if str(owners.training_state_signature(runner)) != baseline_state:
                    raise RuntimeError("v015 quality training state changed before route collection")
                route_evidence = owners.collect_one_action_k(runner, item, route)
                if not isinstance(route_evidence, FrontRESV015PolicyQualityRouteEvidence):
                    raise TypeError("v015 quality collector must bind route/checkpoint identity to one-action-K evidence")
                route_evidence.validate()
                if (
                    route_evidence.route != route
                    or route_evidence.checkpoint_file_sha256 != checkpoint_identity[route]
                    or route_evidence.comparison_signature != item.comparison_signature
                ):
                    raise RuntimeError("v015 quality route evidence has a mixed manifest/actor/checkpoint identity")
                evidence = route_evidence.one_action_k
                if dynamic_state_anchor is None:
                    dynamic_state_anchor = route_evidence.dynamic_state_identity
                elif route_evidence.dynamic_state_identity != dynamic_state_anchor:
                    anchor_fields = dict(dynamic_state_anchor.field_hashes)
                    observed_fields = dict(route_evidence.dynamic_state_identity.field_hashes)
                    differing = tuple(
                        name
                        for name in _V015_DYNAMIC_STATE_FIELDS
                        if anchor_fields.get(name) != observed_fields.get(name)
                    )
                    if dynamic_state_anchor.role_layout != route_evidence.dynamic_state_identity.role_layout:
                        differing = ("role_layout", *differing)
                    raise RuntimeError(
                        "v015 quality routes did not share one full dynamic state: "
                        f"route={route} differing_fields={differing}"
                    )
                if anchor is None:
                    anchor = evidence
                else:
                    _v015_quality_require_same_scenario(anchor, evidence)
                routes.append(
                    _v015_quality_route_result(
                        evidence,
                        route=route,
                        checkpoint_file_sha256=checkpoint_identity[route],
                        dynamic_state_identity=route_evidence.dynamic_state_identity,
                    )
                )
                if str(owners.training_state_signature(runner)) != baseline_state:
                    raise RuntimeError("v015 quality evaluation mutated training state")
        finally:
            owners.close_item(runner, item)
        if str(owners.training_state_signature(runner)) != baseline_state:
            raise RuntimeError("v015 quality item close mutated training state")
        item_rows.append(
            {
                "item_id": item.item_id,
                "comparison_signature": item.comparison_signature,
                "routes": routes,
            }
        )
    payload = {
        "schema_version": _V015_QUALITY_REPORT_SCHEMA,
        "manifest_file_sha256": request.manifest_file_sha256,
        "comparison_signature": request.manifest.comparison_signature,
        "gain_source": _V015_GAIN_SOURCE,
        "owner_identity": dict(owners.owner_identity),
        "items": item_rows,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    output = Path(request.result_path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"v015 quality atomic report temp path already exists: {temporary}")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return payload


def run_frontres_policy_quality_eval(
    runner: Any,
    *,
    manifest_path: str,
    hsl_checkpoint_path: str,
    policy_checkpoint_path: str,
    result_path: str,
) -> Any:
    """Validate the dedicated request, then delegate only to the quality execution owner."""
    if bool(getattr(getattr(runner, "alg", None), "frontres_v015_formal_transaction_enabled", False)):
        request = build_frontres_v015_policy_quality_eval_request(
            manifest_path=manifest_path,
            hsl_checkpoint_path=hsl_checkpoint_path,
            policy_checkpoint_path=policy_checkpoint_path,
            result_path=result_path,
        )
        owners = getattr(runner, "_frontres_v015_policy_quality_owner_bundle", None)
        if owners is None:
            owners = build_frontres_v015_policy_quality_owner_bundle(runner, request)
            install_frontres_v015_policy_quality_owner_bundle(runner, owners)
        if not isinstance(owners, FrontRESV015PolicyQualityOwnerBundle):
            raise RuntimeError("v015 policy-quality rejects a non-v015 formal owner bundle")
        return run_frontres_v015_policy_quality_heldout_eval(runner, request=request, owners=owners)
    request = build_frontres_policy_quality_eval_request(
        manifest_path=manifest_path,
        hsl_checkpoint_path=hsl_checkpoint_path,
        policy_checkpoint_path=policy_checkpoint_path,
        result_path=result_path,
    )
    # QUALITY-ID-01: 正式 entry -> real owner bundle -> matched manifest executor.
    # Result: Q-E6 OFFLINE PASS; six real owner adapters are installed and reached from the official entry.
    # B1: request 校验后安装 dedicated real-owner bundle, 不进入旧 evaluator.
    # B2: manifest executor 对每个 item 运行 matched zero/HSL/policy route.
    # B3: result 保留 comparison/state/checkpoint/owner identity, 且训练状态不变.
    if not callable(getattr(runner, "_frontres_policy_quality_manifest_executor", None)):
        from rsl_rl.runners.frontres_policy_quality_formal_owners import (
            build_frontres_policy_quality_formal_owner_bundle,
        )

        owners = build_frontres_policy_quality_formal_owner_bundle(runner, request)
        install_frontres_policy_quality_manifest_executor(runner, owners)
    executor = getattr(runner, "_frontres_policy_quality_manifest_executor", None)
    if not callable(executor):
        raise RuntimeError(
            "policy-quality formal owner executor is not configured; "
            "do not run live evaluation before Q1-E offline preflight"
        )
    return executor(request)


class _FrozenFrontRESCheckpointNormalizer(torch.nn.Module):
    """Apply checkpoint prefix stats and frozen-GMT suffix stats without mutating runner state."""

    def __init__(
        self,
        *,
        suffix_template: torch.nn.Module,
        checkpoint_state: dict[str, Any],
        obs_dim: int,
        gmt_dim: int,
    ) -> None:
        super().__init__()
        suffix_state = dict(checkpoint_state)
        template_state = suffix_template.state_dict()
        for key, template_value in template_state.items():
            value = suffix_state.get(key)
            if (
                isinstance(value, torch.Tensor)
                and isinstance(template_value, torch.Tensor)
                and value.ndim > 0
                and template_value.ndim > 0
                and value.shape != template_value.shape
                and value.shape[:-1] == template_value.shape[:-1]
                and value.shape[-1] >= template_value.shape[-1]
            ):
                suffix_state[key] = value[..., -template_value.shape[-1] :]
        suffix_template.load_state_dict(suffix_state, strict=True)
        extra = _extract_checkpoint_extra_stats(checkpoint_state, obs_dim=obs_dim, gmt_dim=gmt_dim)
        if extra is None:
            extra_mean = torch.zeros(obs_dim - gmt_dim)
            extra_std = torch.ones(obs_dim - gmt_dim)
        else:
            extra_mean, extra_std = extra
        self.suffix = suffix_template
        self.gmt_dim = int(gmt_dim)
        self.register_buffer("extra_mean", extra_mean.detach().clone())
        self.register_buffer("extra_std", extra_std.detach().clone())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        extra, suffix = _split_policy_observations(observations, self.gmt_dim)
        if extra is None:
            return self.suffix(suffix)
        normalized_extra = (extra - self.extra_mean.to(extra)) / (self.extra_std.to(extra) + 1e-8)
        return torch.cat((normalized_extra, self.suffix(suffix)), dim=-1)


def _split_policy_observations(observations: torch.Tensor, gmt_dim: int) -> tuple[torch.Tensor | None, torch.Tensor]:
    if observations.shape[-1] <= int(gmt_dim):
        return None, observations
    return observations[..., : -int(gmt_dim)], observations[..., -int(gmt_dim) :]


def _extract_checkpoint_extra_stats(
    state: Mapping[str, Any], *, obs_dim: int, gmt_dim: int
) -> tuple[torch.Tensor, torch.Tensor] | None:
    mean = state.get("_mean")
    std = state.get("_std")
    extra_dim = int(obs_dim) - int(gmt_dim)
    if (
        extra_dim <= 0
        or not isinstance(mean, torch.Tensor)
        or not isinstance(std, torch.Tensor)
        or mean.shape[-1] < int(gmt_dim)
        or std.shape != mean.shape
    ):
        return None
    available = max(0, min(extra_dim, int(mean.shape[-1]) - int(gmt_dim)))
    extra_mean = torch.zeros((*mean.shape[:-1], extra_dim), device=mean.device, dtype=mean.dtype)
    extra_std = torch.ones((*std.shape[:-1], extra_dim), device=std.device, dtype=std.dtype)
    if available:
        extra_mean[..., :available] = mean[..., :available]
        extra_std[..., :available] = std[..., :available]
    return extra_mean, extra_std


class FrozenFrontRESTaskActor:
    """Inference-only actor/normalizer pair loaded without runner training state."""

    def __init__(
        self,
        *,
        route: Literal["hsl", "policy"],
        checkpoint_identity: str,
        actor: torch.nn.Module,
        normalizer: torch.nn.Module,
        observation_identity: FrontRESPolicyQualityObservationIdentity,
        max_delta_pos: float,
        max_delta_rpy: float,
    ) -> None:
        if route not in ("hsl", "policy"):
            raise ValueError("FrozenFrontRESTaskActor route must be hsl or policy")
        if not checkpoint_identity.strip():
            raise ValueError("checkpoint_identity must be explicit")
        if max_delta_pos <= 0 or max_delta_rpy <= 0:
            raise ValueError("task-space action bounds must be positive")
        self.route = route
        self.checkpoint_identity = checkpoint_identity
        self.actor = actor.eval()
        self.normalizer = normalizer.eval()
        self.observation_identity = observation_identity
        self.max_delta_pos = float(max_delta_pos)
        self.max_delta_rpy = float(max_delta_rpy)
        for module in (self.actor, self.normalizer):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    @classmethod
    def from_checkpoint_payload(
        cls,
        *,
        route: Literal["hsl", "policy"],
        checkpoint_identity: str,
        checkpoint_payload: Mapping[str, Any],
        actor_template: torch.nn.Module,
        normalizer_template: torch.nn.Module,
        observation_identity: FrontRESPolicyQualityObservationIdentity,
        max_delta_pos: float,
        max_delta_rpy: float,
        gmt_obs_dim: int | None = None,
    ) -> FrozenFrontRESTaskActor:
        model_state = checkpoint_payload.get("model_state_dict")
        if not isinstance(model_state, Mapping):
            raise ValueError("quality actor checkpoint requires model_state_dict")
        residual_state = model_state.get("residual_actor")
        if not isinstance(residual_state, Mapping):
            student_state = {
                str(key).removeprefix("student."): value
                for key, value in model_state.items()
                if str(key).startswith("student.")
            }
            residual_state = student_state or None
        if not isinstance(residual_state, Mapping):
            raise ValueError("quality actor checkpoint requires residual_actor or student.* weights")
        normalizer_state = checkpoint_payload.get("obs_norm_state_dict")
        if not isinstance(normalizer_state, Mapping):
            raise ValueError("quality actor checkpoint requires obs_norm_state_dict")

        actor = copy.deepcopy(actor_template)
        normalizer = copy.deepcopy(normalizer_template)
        actor.load_state_dict(dict(residual_state), strict=True)
        if gmt_obs_dim is None or observation_identity.expected_obs_dim <= int(gmt_obs_dim):
            normalizer.load_state_dict(dict(normalizer_state), strict=True)
        else:
            normalizer = _FrozenFrontRESCheckpointNormalizer(
                suffix_template=normalizer,
                checkpoint_state=dict(normalizer_state),
                obs_dim=observation_identity.expected_obs_dim,
                gmt_dim=int(gmt_obs_dim),
            )
        return cls(
            route=route,
            checkpoint_identity=checkpoint_identity,
            actor=actor,
            normalizer=normalizer,
            observation_identity=observation_identity,
            max_delta_pos=max_delta_pos,
            max_delta_rpy=max_delta_rpy,
        )

    @torch.inference_mode()
    def action(self, observations: torch.Tensor) -> torch.Tensor:
        _validate_observations(observations, self.observation_identity)
        normalized = self.normalizer(observations)
        if not isinstance(normalized, torch.Tensor) or tuple(normalized.shape) != tuple(observations.shape):
            raise ValueError("quality normalizer must preserve observation shape")
        raw = self.actor(normalized[:, : self.observation_identity.actor_input_dim])
        if tuple(raw.shape) != (int(observations.shape[0]), 6) or not bool(torch.isfinite(raw).all()):
            raise ValueError("quality residual actor must emit finite full-6D raw actions")
        return torch.cat(
            (
                torch.tanh(raw[:, :3]) * self.max_delta_pos,
                torch.tanh(raw[:, 3:]) * self.max_delta_rpy,
            ),
            dim=-1,
        )


class ZeroFrontRESTaskActor:
    def __init__(self, observation_identity: FrontRESPolicyQualityObservationIdentity) -> None:
        self.route = "zero"
        self.checkpoint_identity = "zero:no-checkpoint"
        self.observation_identity = observation_identity

    def action(self, observations: torch.Tensor) -> torch.Tensor:
        _validate_observations(observations, self.observation_identity)
        return torch.zeros((int(observations.shape[0]), 6), dtype=observations.dtype, device=observations.device)


def run_frontres_policy_quality_counterfactuals(
    runner: Any,
    *,
    snapshot: FrontRESPolicyQualityScoringState,
    comparison_signature: str,
    adapters: tuple[ZeroFrontRESTaskActor | FrozenFrontRESTaskActor, ...],
    hooks: FrontRESPolicyQualityRouteHooks,
    horizon_k: int,
    isolation_state: Callable[[], str],
) -> tuple[FrontRESPolicyQualityRouteResult, ...]:
    """Run zero/HSL/policy through the same restored state and canonical owner hooks."""
    # QUALITY-ID-01: 检查 manifest identity -> restored scoring state -> route identity.
    # Result: Q-E1/Q-E2/Q-E3 OFFLINE PASS; real simulator equality pending Q1-F.
    # B1: 进入 route 前校验 adapter order、observation identity 与 comparison signature.
    # B2: 每条 route 在首次 observation 前恢复 initial_state_hash.
    # B3: 返回前要求三路 comparison signature 与 state hash 完全相同.
    if tuple(adapter.route for adapter in adapters) != ("zero", "hsl", "policy"):
        raise ValueError("quality counterfactual order must be exactly zero, hsl, policy")
    if horizon_k <= 0:
        raise ValueError("quality counterfactual horizon_k must be positive")
    observation_identity = adapters[0].observation_identity
    if any(adapter.observation_identity != observation_identity for adapter in adapters[1:]):
        raise ValueError("zero/HSL/policy observation and normalizer identities must match")
    isolated_before = isolation_state()
    results: list[FrontRESPolicyQualityRouteResult] = []

    for adapter in adapters:
        # QUALITY-ID-01: 每条 route 的任何 observation/action 前先恢复并验证同一 scoring state.
        # Result: Q-E3 OFFLINE PASS; zero/HSL/policy route identity 共享 comparison/state hash.
        state_identity = restore_frontres_policy_quality_state(
            runner,
            snapshot,
            comparison_signature=comparison_signature,
        )
        if hooks.begin_route is not None:
            hooks.begin_route(adapter.route)
        route_identity = FrontRESPolicyQualityRouteIdentity(
            route=adapter.route,
            checkpoint_identity=adapter.checkpoint_identity,
            state=state_identity,
        )
        route_actions: list[torch.Tensor] = []
        for _ in range(horizon_k):
            observations = hooks.observe()
            actions = adapter.action(observations)
            if tuple(actions.shape) != (int(observations.shape[0]), 6):
                raise ValueError("quality route action must preserve full-6D identity")
            # QUALITY-ACTION-01: actor source 之后、正式 task-space application owner 之前截获 6D action.
            # Result: Q-E3 OFFLINE PASS; zero/HSL/policy source、shape、bounds 与 frozen state 已闭合.
            route_actions.append(actions.detach().clone())
            hooks.apply_action(actions)
            hooks.step()
        # QUALITY-GAIN-01 / QUALITY-EXEC-01: K-step rollout 完成后只消费注入的
        # canonical Gain 与 execution owner; 本模块不复制公式或执行指标.
        # Result: Q-E3 OFFLINE CONNECTIVITY PASS; real owner wiring remains Q1-D/Q1-F.
        results.append(
            FrontRESPolicyQualityRouteResult(
                identity=route_identity,
                observation_identity=observation_identity,
                actions=torch.stack(route_actions, dim=0),
                gain=hooks.compute_gain(),
                execution=hooks.capture_execution(),
            )
        )

    if isolation_state() != isolated_before:
        raise RuntimeError("quality evaluation mutated optimizer/sampler/warmup isolation state")
    signatures = {result.identity.comparison_signature for result in results}
    state_hashes = {result.identity.state.initial_state_hash for result in results}
    if signatures != {comparison_signature} or state_hashes != {snapshot.initial_state_hash}:
        raise RuntimeError("quality counterfactual routes did not share one matched scoring state")
    return tuple(results)


def _validate_observations(
    observations: torch.Tensor,
    identity: FrontRESPolicyQualityObservationIdentity,
) -> None:
    if not isinstance(observations, torch.Tensor) or observations.ndim != 2:
        raise ValueError("quality observations must be a rank-2 tensor")
    if int(observations.shape[1]) != identity.expected_obs_dim:
        raise ValueError(
            f"quality observation dim mismatch: expected {identity.expected_obs_dim}, got {int(observations.shape[1])}"
        )
    if not bool(torch.isfinite(observations).all()):
        raise ValueError("quality observations must be finite")


def run_zero_frontres_preroll(
    step_fn: Callable[[torch.Tensor], Any],
    *,
    num_envs: int,
    steps: int,
    device: torch.device | str,
) -> None:
    """Advance GMT with exact zero Delta SE(3); no policy object enters this boundary."""
    if num_envs <= 0 or steps < 0:
        raise ValueError("zero preroll requires num_envs > 0 and steps >= 0")
    zero_action = torch.zeros((num_envs, 6), dtype=torch.float32, device=device)
    for _ in range(steps):
        step_fn(zero_action.clone())


def capture_frontres_policy_quality_state(
    runner: Any,
    *,
    env_ids: torch.Tensor | tuple[int, ...] | list[int],
    comparison_signature: str,
    role_layout: tuple[str, ...] | list[str],
) -> FrontRESPolicyQualityScoringState:
    """Capture the complete scoring-start state after the one shared zero preroll."""
    ids = _normalize_env_ids(env_ids)
    roles = _normalize_role_layout(role_layout, count=int(ids.numel()))
    env, raw_env = _resolve_envs(runner)
    robot = _resolve_robot(raw_env)
    command = _resolve_command(raw_env)
    origins = _require_tensor(getattr(getattr(raw_env, "scene", None), "env_origins", None), "env_origins")
    episode = _require_tensor(
        getattr(env, "episode_length_buf", getattr(raw_env, "episode_length_buf", None)),
        "episode_length_buf",
    )
    command_state = tuple(
        (name, _capture_rows(_require_tensor(getattr(command, name, None), f"command.{name}"), ids))
        for name in _COMMAND_STATE_FIELDS
    )
    perturber = getattr(command, "perturber", None)
    if perturber is None:
        raise AttributeError("policy-quality state capture requires command.perturber")
    perturber_state = tuple(
        (f"perturber.{name}", _capture_rows(value, ids))
        for name, value in sorted(vars(perturber).items())
        if isinstance(value, torch.Tensor)
        and value.ndim > 0
        and int(value.shape[0]) > int(ids.max().item())
    )
    if not perturber_state:
        raise AttributeError("policy-quality state capture found no per-env perturber tensors")
    cuda_rng = tuple(_TensorImage.capture(state) for state in torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else ()

    # QUALITY-ID-01: zero preroll 结束后、任一 counterfactual route 开始前冻结动态起点.
    # Result: Q-E2 OFFLINE PASS; 完整 fake lifecycle 可逐字段 restore 并复现 hash.
    # Real simulator state identity remains pending Q1-F live evidence.
    return FrontRESPolicyQualityScoringState(
        comparison_signature=comparison_signature,
        env_ids=tuple(ids.tolist()),
        role_layout=roles,
        root_state_w=_capture_rows(_require_tensor(robot.data.root_state_w, "robot.root_state_w"), ids),
        joint_pos=_capture_rows(_require_tensor(robot.data.joint_pos, "robot.joint_pos"), ids),
        joint_vel=_capture_rows(_require_tensor(robot.data.joint_vel, "robot.joint_vel"), ids),
        env_origins=_capture_rows(origins, ids),
        episode_length=_capture_rows(episode, ids),
        command_state=command_state,
        perturber_state=perturber_state,
        python_rng_state=pickle.dumps(random.getstate(), protocol=5),
        numpy_rng_state=pickle.dumps(np.random.get_state(), protocol=5),
        torch_rng_state=_TensorImage.capture(torch.random.get_rng_state()),
        cuda_rng_state=cuda_rng,
    )


def capture_frontres_v015_policy_quality_dynamic_state_identity(
    runner: Any,
    *,
    comparison_signature: str,
    pair_layout: Any,
) -> FrontRESV015DynamicStateIdentity:
    """Hash the complete active v015 post-reset state without restoring or mutating it."""

    counts = (
        int(getattr(pair_layout, "n_train", 0)),
        int(getattr(pair_layout, "n_base", 0)),
        int(getattr(pair_layout, "n_candidate", 0)),
        int(getattr(pair_layout, "n_clean", 0)),
    )
    if counts != (4, 4, 0, 0):
        raise RuntimeError("v015 dynamic-state identity requires exactly 4 Repair + 4 Noisy rows")
    role_layout = ("repair",) * counts[0] + ("noisy",) * counts[1]
    env, raw_env = _resolve_envs(runner)
    env_count = int(getattr(env, "num_envs", getattr(raw_env, "num_envs", 0)) or 0)
    if env_count != len(role_layout):
        raise RuntimeError("v015 dynamic-state identity requires B=8 role-aligned environment rows")
    env_ids = torch.arange(env_count, dtype=torch.long)
    snapshot = capture_frontres_policy_quality_state(
        runner,
        env_ids=env_ids,
        comparison_signature=comparison_signature,
        role_layout=role_layout,
    )
    command = _resolve_command(raw_env)
    local_snapshot_fn = getattr(command, "frontres_local_scenario_snapshot", None)
    if not callable(local_snapshot_fn):
        raise RuntimeError("v015 dynamic-state identity requires the command-owned local-scenario snapshot")
    local_snapshot = local_snapshot_fn(env_ids)
    if not isinstance(local_snapshot, Mapping):
        raise TypeError("v015 command local-scenario snapshot must be a mapping")
    required_local = {
        "current_root_artifact_t",
        "intent_q29",
        "clean_continuation",
        "horizon_k",
        "continuation_lengths",
        "scenario_ids",
        "noisy_segment_hashes",
        "x_t_identities",
        "roles",
        "provenance",
    }
    if not required_local.issubset(local_snapshot):
        raise RuntimeError("v015 dynamic-state identity local scenario is incomplete")
    if tuple(local_snapshot["roles"]) != role_layout:
        raise RuntimeError("v015 dynamic-state identity local-scenario role alignment is mixed")

    values = {
        "root_state_w": snapshot.root_state_w,
        "joint_pos": snapshot.joint_pos,
        "joint_vel": snapshot.joint_vel,
        "env_origins": snapshot.env_origins,
        "episode_length": snapshot.episode_length,
        "command_state": snapshot.command_state,
        "perturber_state": snapshot.perturber_state,
        "python_rng_state": snapshot.python_rng_state,
        "numpy_rng_state": snapshot.numpy_rng_state,
        "torch_rng_state": snapshot.torch_rng_state,
        "cuda_rng_state": snapshot.cuda_rng_state,
        "local_scenario": {key: local_snapshot[key] for key in sorted(required_local)},
    }
    identity = FrontRESV015DynamicStateIdentity(
        comparison_signature=comparison_signature,
        role_layout=role_layout,
        field_hashes=tuple(
            (name, _v015_quality_field_hash(name, values[name]))
            for name in _V015_DYNAMIC_STATE_FIELDS
        ),
    )
    identity.validate()
    return identity


def restore_frontres_policy_quality_state(
    runner: Any,
    snapshot: FrontRESPolicyQualityScoringState,
    *,
    comparison_signature: str,
) -> FrontRESPolicyQualityStateIdentity:
    """Restore a scoring state and fail closed unless every captured field hashes identically."""
    if comparison_signature != snapshot.comparison_signature:
        raise ValueError("comparison signature mismatch during policy-quality state restore")
    env, raw_env = _resolve_envs(runner)
    robot = _resolve_robot(raw_env)
    command = _resolve_command(raw_env)
    ids = torch.tensor(snapshot.env_ids, dtype=torch.long)

    root_target = _require_tensor(robot.data.root_state_w, "robot.root_state_w")
    joint_pos_target = _require_tensor(robot.data.joint_pos, "robot.joint_pos")
    joint_vel_target = _require_tensor(robot.data.joint_vel, "robot.joint_vel")
    root = snapshot.root_state_w.restore(device=root_target.device)
    joint_pos = snapshot.joint_pos.restore(device=joint_pos_target.device)
    joint_vel = snapshot.joint_vel.restore(device=joint_vel_target.device)
    sim_ids = ids.to(root_target.device)
    robot.write_root_state_to_sim(root, env_ids=sim_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=sim_ids)

    origins = _require_tensor(getattr(getattr(raw_env, "scene", None), "env_origins", None), "env_origins")
    _restore_rows(origins, ids, snapshot.env_origins)
    episode = _require_tensor(
        getattr(env, "episode_length_buf", getattr(raw_env, "episode_length_buf", None)),
        "episode_length_buf",
    )
    _restore_rows(episode, ids, snapshot.episode_length)
    for name, image in snapshot.command_state:
        _restore_rows(_require_tensor(getattr(command, name, None), f"command.{name}"), ids, image)
    perturber = getattr(command, "perturber", None)
    if perturber is None:
        raise AttributeError("policy-quality state restore requires command.perturber")
    for qualified_name, image in snapshot.perturber_state:
        name = qualified_name.removeprefix("perturber.")
        _restore_rows(_require_tensor(getattr(perturber, name, None), qualified_name), ids, image)

    random.setstate(pickle.loads(snapshot.python_rng_state))
    np.random.set_state(pickle.loads(snapshot.numpy_rng_state))
    torch.random.set_rng_state(snapshot.torch_rng_state.restore(device="cpu"))
    if snapshot.cuda_rng_state:
        if not torch.cuda.is_available() or len(snapshot.cuda_rng_state) != torch.cuda.device_count():
            raise RuntimeError("CUDA RNG topology differs from captured policy-quality state")
        torch.cuda.set_rng_state_all([image.restore(device="cpu") for image in snapshot.cuda_rng_state])

    restored = capture_frontres_policy_quality_state(
        runner,
        env_ids=snapshot.env_ids,
        comparison_signature=comparison_signature,
        role_layout=snapshot.role_layout,
    ).state_identity
    if restored.initial_state_hash != snapshot.initial_state_hash:
        raise RuntimeError(
            "policy-quality scoring state restore mismatch: "
            f"expected={snapshot.initial_state_hash} observed={restored.initial_state_hash}"
        )
    return restored


def _normalize_env_ids(env_ids: torch.Tensor | tuple[int, ...] | list[int]) -> torch.Tensor:
    ids = env_ids.detach().to(device="cpu", dtype=torch.long).flatten() if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, dtype=torch.long)
    if ids.numel() == 0 or bool((ids < 0).any()) or int(torch.unique(ids).numel()) != int(ids.numel()):
        raise ValueError("env_ids must be non-empty, non-negative, and unique")
    return ids


def _normalize_role_layout(role_layout: tuple[str, ...] | list[str], *, count: int) -> tuple[str, ...]:
    if not isinstance(role_layout, (tuple, list)) or len(role_layout) != count:
        raise ValueError(f"role_layout must contain exactly {count} entries")
    roles = tuple(str(role).strip() for role in role_layout)
    if any(not role for role in roles):
        raise ValueError("role_layout entries must be non-empty")
    return roles


def _resolve_envs(runner: Any) -> tuple[Any, Any]:
    env = getattr(runner, "env", None)
    if env is None:
        raise AttributeError("policy-quality state capture requires runner.env")
    return env, getattr(env, "unwrapped", env)


def _resolve_robot(raw_env: Any) -> Any:
    scene = getattr(raw_env, "scene", None)
    try:
        robot = scene["robot"]
    except (KeyError, TypeError):
        robot = getattr(scene, "robot", None)
    if robot is None or not hasattr(robot, "data"):
        raise AttributeError("policy-quality state capture requires scene['robot']")
    return robot


def _resolve_command(raw_env: Any) -> Any:
    manager = getattr(raw_env, "command_manager", None)
    if manager is None or not hasattr(manager, "get_term"):
        raise AttributeError("policy-quality state capture requires command_manager.get_term('motion')")
    return manager.get_term("motion")


def _require_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise AttributeError(f"policy-quality state requires tensor {name}")
    return value


def _capture_rows(tensor: torch.Tensor, ids: torch.Tensor) -> _TensorImage:
    if tensor.ndim == 0 or int(tensor.shape[0]) <= int(ids.max().item()):
        raise ValueError(f"state tensor shape {tuple(tensor.shape)} cannot select env_ids={ids.tolist()}")
    return _TensorImage.capture(tensor.index_select(0, ids.to(tensor.device)))


def _restore_rows(target: torch.Tensor, ids: torch.Tensor, image: _TensorImage) -> None:
    values = image.restore(device=target.device)
    target_ids = ids.to(target.device)
    expected = (int(target_ids.numel()), *tuple(target.shape[1:]))
    if tuple(values.shape) != expected:
        raise ValueError(f"snapshot shape {tuple(values.shape)} does not match restore target {expected}")
    # Isaac command caches may be inference tensors. Preserve their object identity
    # and restore rows inside the mode that created them.
    with torch.inference_mode():
        target.index_copy_(0, target_ids, values.to(dtype=target.dtype))
