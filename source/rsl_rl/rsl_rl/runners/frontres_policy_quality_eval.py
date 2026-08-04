"""State-isolation helpers for the independent FrontRES policy-quality evaluator."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from rsl_rl.frontres.frontres_policy_quality_manifest import (
    FrontRESPolicyQualityManifest,
    FrontRESPolicyQualityRouteIdentity,
    FrontRESPolicyQualityStateIdentity,
    FrontRESV015PolicyQualityManifest,
)
from rsl_rl.runners.frontres_policy_quality_interfaces import (
    FrontRESPolicyQualityEvalRequest,
    FrontRESPolicyQualityFormalOwnerBundle,
    FrontRESPolicyQualityObservationIdentity,
    FrontRESPolicyQualityRouteHooks,
    FrontRESPolicyQualityRouteResult,
)
from rsl_rl.runners.frontres_policy_quality_legacy import run_frontres_policy_quality_counterfactuals
from rsl_rl.runners.frontres_policy_quality_state import (
    FrontRESPolicyQualityScoringState,
    FrontRESPolicyQualityTensorImage,
    capture_frontres_policy_quality_state,
    resolve_frontres_policy_quality_command,
    resolve_frontres_policy_quality_envs,
    restore_frontres_policy_quality_state,
)
from rsl_rl.runners.frontres_evaluation_reporting import write_frontres_atomic_json


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
_V015_QUALITY_REPORT_SCHEMA = "frontres-v015-heldout-quality-report-v2"
_V015_GAIN_SOURCE = "FRS-GAIN-v006-loaded-support-zmp-applicability"
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
        # B1: 校验 comparison, role layout 与 field hashes, 产出完整 dynamic-state identity.
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
        # B1: 归约有序 field hashes, 产出 route-start 的整体状态哈希.
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
        # B1: 校验 owner bundle callbacks 与 immutable identity, 产出可执行 dependency set.
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
        # B1: 校验 route/checkpoint/scenario 与 one-action-K fields, 产出 matched route record.
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
    # B1: 验证 bundle 类型并安装到 runner, 产出唯一 policy-quality owner binding.
    """Install one immutable S2 connector; legacy executor attributes are ignored."""

    if not isinstance(bundle, FrontRESV015PolicyQualityOwnerBundle):
        raise TypeError("v015 policy-quality requires FrontRESV015PolicyQualityOwnerBundle")
    if hasattr(runner, "_frontres_v015_policy_quality_owner_bundle"):
        raise RuntimeError("v015 policy-quality owner bundle is already installed")
    runner._frontres_v015_policy_quality_owner_bundle = bundle


def _v015_quality_hash_state(digest: Any, value: Any) -> None:
    # B1: 按稳定类型递归编码 state value, 更新 deterministic SHA-256 digest.
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
    # B1: 将字段名和值共同编码, 产出避免跨字段碰撞的 state hash.
    digest = hashlib.sha256()
    digest.update(name.encode("ascii"))
    if isinstance(value, FrontRESPolicyQualityTensorImage):
        value.update_hash(digest, name=name)
    elif isinstance(value, bytes):
        digest.update(value)
    elif isinstance(value, tuple) and all(isinstance(item, FrontRESPolicyQualityTensorImage) for item in value):
        for index, image in enumerate(value):
            image.update_hash(digest, name=f"{name}[{index}]")
    elif isinstance(value, tuple) and all(
        isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], FrontRESPolicyQualityTensorImage)
        for item in value
    ):
        for field_name, image in value:
            image.update_hash(digest, name=str(field_name))
    else:
        _v015_quality_hash_state(digest, value)
    return digest.hexdigest()


@contextmanager
def _frontres_v015_quality_inference_mode(runner: Any):
    # B1: 捕获并冻结 policy/normalizer modes, 产出可恢复的 inference-only context.
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
    # B1: 哈希 model/optimizer/sampler/normalizer/transaction facts, 产出零写入签名.
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
        ("transaction", getattr(runner, "_frontres_checkpoint_transaction_state", None)),
        ("receipt", getattr(runner, "_frontres_last_committed_transaction_receipt", None)),
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

    # B1: 验证 strict request 与 4 Repair + 4 Noisy layout, 产出唯一 held-out owner context.
    if not isinstance(request, FrontRESV015PolicyQualityEvalRequest):
        raise TypeError("formal v015 quality owner requires the strict request")
    if not bool(getattr(getattr(runner, "alg", None), "frontres_formal_transaction_enabled", False)):
        raise RuntimeError("formal v015 quality owner requires the active grouped transaction configuration")
    from rsl_rl.runners.frontres_checkpointing import frontres_quality_route_actor
    from rsl_rl.runners.frontres_segment_live_reset import apply_frontres_current_segment_reset
    from rsl_rl.runners.frontres_segment_one_action_k import (
        collect_frontres_v015_one_action_k_evidence,
        read_frontres_live_observations,
    )
    from rsl_rl.runners.frontres_segment_live_sampler import (
        close_frontres_local_scenarios,
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
    # B2: 建立 request-scoped item caches, 产出 route-start 与 sealed batch lifecycle state.
    item_batches: dict[str, Any] = {}
    item_route_starts: dict[
        str,
        tuple[FrontRESPolicyQualityScoringState, FrontRESV015DynamicStateIdentity],
    ] = {}
    route_start_roles = ("repair",) * 4 + ("noisy",) * 4
    route_start_env_ids = tuple(range(8))

    # B3: 对每个 manifest item 恢复同一 route-start, 产出 zero/HSL/policy one-action-K evidence.
    def collect_one_action_k(_runner: Any, item: Any, route: str) -> FrontRESV015PolicyQualityRouteEvidence:
        # B1: materialize 或复用 sealed item, 恢复 matched route-start 并收集 K evidence.
        if _runner is not runner or route not in _V015_QUALITY_ROUTES:
            raise RuntimeError("v015 quality owner received a mixed runner or route identity")
        signature = str(item.comparison_signature)
        prepared = item_batches.get(signature)
        if prepared is None:
            prepared = prepare_frontres_v015_policy_quality_item_batch(runner, item)
            item_batches[signature] = prepared
        runner._frontres_segment_live_current_batch = prepared.batch
        runner._frontres_segment_live_current_sample = prepared.sample
        route_start = item_route_starts.get(signature)
        if route_start is None:
            reset = apply_frontres_current_segment_reset(runner, pair_layout=pair_layout)
            if reset is None or not bool(reset.success_mask.detach().bool().all().item()):
                raise RuntimeError("v015 held-out quality failed to restore the sealed Clean x_t")
            snapshot = capture_frontres_policy_quality_state(
                runner,
                env_ids=route_start_env_ids,
                comparison_signature=signature,
                role_layout=route_start_roles,
            )
            expected_identity = capture_frontres_v015_policy_quality_dynamic_state_identity(
                runner,
                comparison_signature=signature,
                pair_layout=pair_layout,
            )
            route_start = (snapshot, expected_identity)
            item_route_starts[signature] = route_start
        snapshot, expected_identity = route_start
        restore_frontres_policy_quality_state(
            runner,
            snapshot,
            comparison_signature=signature,
        )
        dynamic_state_identity = capture_frontres_v015_policy_quality_dynamic_state_identity(
            runner,
            comparison_signature=signature,
            pair_layout=pair_layout,
        )
        if dynamic_state_identity != expected_identity:
            expected_fields = dict(expected_identity.field_hashes)
            observed_fields = dict(dynamic_state_identity.field_hashes)
            differing = tuple(
                name
                for name in _V015_DYNAMIC_STATE_FIELDS
                if expected_fields.get(name) != observed_fields.get(name)
            )
            if expected_identity.role_layout != dynamic_state_identity.role_layout:
                differing = ("role_layout", *differing)
            raise RuntimeError(
                "v015 quality route-start restore did not reproduce the sealed dynamic state: "
                f"route={route} differing_fields={differing}"
            )
        observations = read_frontres_live_observations(runner)
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
                with frontres_quality_route_actor(
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
        # B2: 绑定 route/checkpoint/scenario identities, 产出 immutable route evidence.
        return FrontRESV015PolicyQualityRouteEvidence(
            route=route,
            checkpoint_file_sha256=checkpoint_sha,
            comparison_signature=signature,
            one_action_k=evidence,
            dynamic_state_identity=dynamic_state_identity,
        )

    # B4: 关闭 item-owned scenario/batch 并清除 runner projection, 产出无残留的 evaluation lifecycle.
    def close_item(_runner: Any, item: Any) -> None:
        # B1: 关闭 command carrier, batch 与 runner projections, 清理 item lifecycle.
        """Close one manifest item's command carrier after all counterfactual routes."""

        if _runner is not runner:
            raise RuntimeError("v015 quality item close received a mixed runner identity")
        signature = str(item.comparison_signature)
        item_route_starts.pop(signature, None)
        prepared = item_batches.pop(signature, None)
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
                close_frontres_local_scenarios(prepared.batch)
            finally:
                if getattr(runner, "_frontres_segment_live_current_batch", None) is prepared.batch:
                    runner._frontres_segment_live_current_batch = None
                    runner._frontres_segment_live_current_sample = None

    # B5: 将 request-bound callbacks 封装为 owner bundle, 产出本次 evaluation 的执行接口.
    return FrontRESV015PolicyQualityOwnerBundle(
        owner_identity=_V015_QUALITY_OWNER_IDENTITY,
        collect_one_action_k=collect_one_action_k,
        close_item=close_item,
        training_state_signature=_v015_quality_training_state_signature,
    )


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
        # B1: 将 legacy manifest request 交给显式 owner bundle, 产出兼容 quality payload.
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
        write_frontres_atomic_json(request.result_path, payload, compact=True)
        return payload

    runner._frontres_policy_quality_manifest_executor = execute


def build_frontres_policy_quality_eval_request(
    *,
    manifest_path: str,
    hsl_checkpoint_path: str,
    policy_checkpoint_path: str,
    result_path: str,
) -> FrontRESPolicyQualityEvalRequest:
    # B1: 解析 legacy manifest/checkpoint/report paths, 产出 fail-closed request identity.
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
    # B1: 解析 v015 manifest 与 strict checkpoint identities, 产出 immutable request.
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
    from rsl_rl.runners.frontres_checkpointing import inspect_frontres_quality_checkpoint

    hsl = inspect_frontres_quality_checkpoint(paths["hsl_checkpoint_path"], route="hsl")
    policy = inspect_frontres_quality_checkpoint(paths["policy_checkpoint_path"], route="policy")
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
        or hsl.method_contract_id != "FRS-METHOD-v015"
        or policy.method_contract_id != manifest.method_contract_id
        or hsl.training_contract_id != "FRS-TRAIN-v007"
        or policy.training_contract_id != manifest.training_contract_id
        or policy.gain_contract_id != manifest.gain_contract_id
        or policy.ppo_contract_id != manifest.ppo_contract_id
    ):
        raise ValueError("v015 policy-quality manifest and checkpoint contract/action identities are mixed")
    # B2: 绑定 comparison signature 与 output boundary, 产出经 validate 的 request.
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
    # B1: 将 detached tensor tree 转为 finite JSON values, 产出 serializer-safe payload.
    """Serialize finite values and preserve unavailable values as null, never zero."""

    def convert(item: Any) -> Any:
        if isinstance(item, list):
            return [convert(value) for value in item]
        number = float(item)
        return number if np.isfinite(number) else None

    return convert(value.detach().to(device="cpu").tolist())


def _v015_quality_require_same_scenario(anchor: Any, candidate: Any) -> None:
    # B1: 比较 route 的 scenario/state identities, 拒绝 mixed held-out evidence.
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
    # B1: 归约 action, Intent 与 phase-Physics evidence, 产出单 route report row.
    """Consume one-action evidence through the active v003 Gain owner only."""

    from rsl_rl.frontres.frontres_gain_legacy import (
        FrontRESIntentPhysicsGainConfig,
        FrontRESIntentPhysicsGainInput,
        compute_intent_physics_local_repair_gain,
        evaluate_phase_conditioned_physics,
    )
    from rsl_rl.frontres.frontres_segment_evidence_legacy import (
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
        repaired_zmp_margin=facts.repaired_zmp_margin,
        noisy_zmp_margin=facts.noisy_zmp_margin,
        repaired_contact=facts.repaired_contact,
        noisy_contact=facts.noisy_contact,
        repaired_contact_violation=facts.repaired_contact_violation,
        noisy_contact_violation=facts.noisy_contact_violation,
        repaired_zmp_violation=facts.repaired_zmp_violation,
        noisy_zmp_violation=facts.noisy_zmp_violation,
        expected_support_steps=facts.expected_support_steps,
        repaired_contact_steps=facts.repaired_contact_steps,
        noisy_contact_steps=facts.noisy_contact_steps,
        repaired_zmp_margin_steps=facts.repaired_zmp_margin_steps,
        noisy_zmp_margin_steps=facts.noisy_zmp_margin_steps,
        physics_pair_valid_mask=facts.physics_pair_valid_mask,
    )
    gain = compute_intent_physics_local_repair_gain(gain_input, config=FrontRESIntentPhysicsGainConfig())
    valid = facts.intent_valid_mask.bool() & gain.available.bool()
    if not bool(valid.any().item()):
        raise RuntimeError("v015 quality route has no valid v006 objective/constraint row")
    # B2: 投影 v003 Gain components 与 one-action-K trajectory, 产出 route quality metrics.
    components = {
        name: torch.where(valid, getattr(gain, name).detach().float(), torch.full_like(gain.gain_total, float("nan")))
        for name in (
            "intent_gain", "physics_gain", "repair_cost", "gain_total",
            "intent_quality_repaired", "intent_quality_noisy",
            "physics_admissible_repaired", "physics_admissible_noisy",
            "physics_deficit_repaired", "physics_deficit_noisy",
            "utility_repaired", "utility_noisy", "repair_penalty",
            "contact_constraint", "zmp_constraint", "survival_constraint",
        )
    }
    repair_rows = evidence.policy_row_indices.detach().to(dtype=torch.long)
    required_raw = {
        "physics_survival_repaired_steps": evidence.physics_survival_repaired_steps,
        "physics_survival_noisy_steps": evidence.physics_survival_noisy_steps,
        "evaluation_only_lateral_lean_repaired_steps": evidence.evaluation_only_lateral_lean_repaired_steps,
        "evaluation_only_lateral_lean_noisy_steps": evidence.evaluation_only_lateral_lean_noisy_steps,
    }
    if any(not isinstance(value, torch.Tensor) for value in required_raw.values()):
        missing = tuple(name for name, value in required_raw.items() if not isinstance(value, torch.Tensor))
        raise RuntimeError(f"v015 quality raw Physics/lean evidence is missing: {missing}")
    pair_valid = evidence.physics_pair_valid_mask.detach().bool()
    survival_repaired = required_raw["physics_survival_repaired_steps"].detach()
    survival_noisy = required_raw["physics_survival_noisy_steps"].detach()
    for name, value in (("Repair", survival_repaired), ("Noisy", survival_noisy)):
        if bool(((value != 0) & (value != 1)).any()):
            raise RuntimeError(f"v015 quality {name} survival trajectory must be binary")
    noisy_rows = torch.tensor(
        [index for index, role in enumerate(evidence.roles) if role == "noisy"],
        device=evidence.survival_steps.device,
        dtype=torch.long,
    )
    if not torch.equal(
        survival_repaired.float().sum(dim=0),
        evidence.survival_steps.index_select(0, repair_rows).float(),
    ) or not torch.equal(
        survival_noisy.float().sum(dim=0),
        evidence.survival_steps.index_select(0, noisy_rows).float(),
    ):
        raise RuntimeError("v015 quality raw survival trajectory disagrees with one-action-K evidence")
    phase_kwargs = {
        "expected_support_steps": evidence.physics_expected_support_steps,
        "valid_steps": pair_valid,
    }
    # B3: 按 expected Contact phase 计算 Repair/Noisy ZMP 与 recovery evidence.
    repaired_phase = evaluate_phase_conditioned_physics(
        actual_contact_steps=evidence.physics_contact_repaired_steps,
        zmp_margin_steps=evidence.physics_zmp_repaired_steps,
        **phase_kwargs,
    )
    noisy_phase = evaluate_phase_conditioned_physics(
        actual_contact_steps=evidence.physics_contact_noisy_steps,
        zmp_margin_steps=evidence.physics_zmp_noisy_steps,
        **phase_kwargs,
    )
    applicable_repaired = repaired_phase["zmp_applicable_steps"].bool()
    applicable_noisy = noisy_phase["zmp_applicable_steps"].bool()
    aggregate_repaired = applicable_repaired.any(dim=0) & valid.to(applicable_repaired.device)
    aggregate_noisy = applicable_noisy.any(dim=0) & valid.to(applicable_noisy.device)
    if not torch.equal(gain.zmp_applicable_repaired.to(aggregate_repaired.device), aggregate_repaired):
        raise RuntimeError("v015 quality Gain lost Repair ZMP applicability identity")
    if not torch.equal(gain.zmp_applicable_noisy.to(aggregate_noisy.device), aggregate_noisy):
        raise RuntimeError("v015 quality Gain lost Noisy ZMP applicability identity")
    if not torch.equal(gain.zmp_constraint_applicable.to(aggregate_repaired.device), aggregate_repaired):
        raise RuntimeError("v015 quality PPO ZMP applicability must alias the Repair role")
    supported = pair_valid & evidence.physics_expected_support_steps.detach().bool().any(dim=-1)
    recovery_repaired = supported & evidence.physics_contact_repaired_steps.bool().any(dim=-1) & (~applicable_repaired)
    recovery_noisy = supported & evidence.physics_contact_noisy_steps.bool().any(dim=-1) & (~applicable_noisy)
    nan_steps = torch.full_like(evidence.physics_zmp_repaired_steps.float(), float("nan"))
    repaired_violation = torch.where(applicable_repaired, repaired_phase["zmp_step_violation"], nan_steps)
    noisy_violation = torch.where(applicable_noisy, noisy_phase["zmp_step_violation"], nan_steps)

    def cumulative_mean(value: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(value)
        total = torch.where(finite, value, torch.zeros_like(value)).cumsum(dim=0)
        count = finite.to(dtype=value.dtype).cumsum(dim=0)
        return torch.where(count > 0, total / count.clamp_min(1.0), torch.full_like(value, float("nan")))

    lean_repaired = required_raw["evaluation_only_lateral_lean_repaired_steps"].detach().float()
    lean_noisy = required_raw["evaluation_only_lateral_lean_noisy_steps"].detach().float()
    for name, value in (("Repair", lean_repaired), ("Noisy", lean_noisy)):
        finite = torch.isfinite(value)
        if not bool(finite[pair_valid].all()) or bool(finite[~pair_valid].any()):
            raise RuntimeError(f"v015 quality {name} lateral-lean trajectory has invalid mask semantics")
    # B4: 序列化 action, Intent, Contact, ZMP, survival 与 lean, 产出 report row.
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
        "zmp_applicable_repaired": [bool(value) for value in aggregate_repaired.detach().cpu().tolist()],
        "zmp_applicable_noisy": [bool(value) for value in aggregate_noisy.detach().cpu().tolist()],
        "zmp_constraint_applicable": [
            bool(value) for value in gain.zmp_constraint_applicable.detach().cpu().tolist()
        ],
        "evidence_valid_step_count": [
            int(value) for value in evidence.survival_steps.index_select(0, repair_rows).tolist()
        ],
        "intent_q29_provenance": facts.intent_q29_provenance,
        "intent_q29_source": facts.intent_q29_source,
        "expected_contact_steps": _v015_quality_json_tensor(evidence.physics_expected_support_steps),
        "actual_contact_repaired_steps": _v015_quality_json_tensor(evidence.physics_contact_repaired_steps),
        "actual_contact_noisy_steps": _v015_quality_json_tensor(evidence.physics_contact_noisy_steps),
        "phase_zmp_applicable_repaired_steps": _v015_quality_json_tensor(applicable_repaired),
        "phase_zmp_applicable_noisy_steps": _v015_quality_json_tensor(applicable_noisy),
        "phase_zmp_na_repaired_steps": _v015_quality_json_tensor(~applicable_repaired),
        "phase_zmp_na_noisy_steps": _v015_quality_json_tensor(~applicable_noisy),
        "phase_zmp_margin_repaired_steps": _v015_quality_json_tensor(evidence.physics_zmp_repaired_steps),
        "phase_zmp_margin_noisy_steps": _v015_quality_json_tensor(evidence.physics_zmp_noisy_steps),
        "phase_zmp_violation_repaired_steps": _v015_quality_json_tensor(repaired_violation),
        "phase_zmp_violation_noisy_steps": _v015_quality_json_tensor(noisy_violation),
        "phase_zmp_support_transition_steps": _v015_quality_json_tensor(repaired_phase["support_transition_steps"]),
        "phase_zmp_recovery_repaired_steps": _v015_quality_json_tensor(recovery_repaired),
        "phase_zmp_recovery_noisy_steps": _v015_quality_json_tensor(recovery_noisy),
        "survival_repaired_steps": _v015_quality_json_tensor(required_raw["physics_survival_repaired_steps"]),
        "survival_noisy_steps": _v015_quality_json_tensor(required_raw["physics_survival_noisy_steps"]),
        "evaluation_only_sustained_lean": {
            "repaired_lateral_roll_rad": _v015_quality_json_tensor(lean_repaired),
            "noisy_lateral_roll_rad": _v015_quality_json_tensor(lean_noisy),
            "repaired_cumulative_mean_rad": _v015_quality_json_tensor(cumulative_mean(lean_repaired)),
            "noisy_cumulative_mean_rad": _v015_quality_json_tensor(cumulative_mean(lean_noisy)),
        },
        **{name: _v015_quality_json_tensor(value) for name, value in components.items()},
    }


def run_frontres_v015_policy_quality_heldout_eval(
    runner: Any,
    *,
    request: FrontRESV015PolicyQualityEvalRequest,
    owners: FrontRESV015PolicyQualityOwnerBundle,
) -> dict[str, Any]:
    # B1: 在可恢复 inference mode 中执行 matched evaluator, 产出零写入 report.
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

    # B1: 验证 request/owners 并封存训练状态, 产出三条 route 的共同比较边界.
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
    # B2: 每个 item 执行 matched zero/HSL/policy, 产出 identity-checked route rows.
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
        # B3: 将三条 route 的 matched evidence 固化为一个 item row, 保留 checkpoint 与 state identity.
        item_rows.append(
            {
                "item_id": item.item_id,
                "comparison_signature": item.comparison_signature,
                "routes": routes,
            }
        )
    # B4: 聚合 immutable quality rows 并原子写入, 产出不反馈训练的 report artifact.
    payload = {
        "schema_version": _V015_QUALITY_REPORT_SCHEMA,
        "manifest_file_sha256": request.manifest_file_sha256,
        "comparison_signature": request.manifest.comparison_signature,
        "gain_source": _V015_GAIN_SOURCE,
        "owner_identity": dict(owners.owner_identity),
        "items": item_rows,
    }
    write_frontres_atomic_json(request.result_path, payload, compact=True)
    return payload


def run_frontres_policy_quality_eval(
    runner: Any,
    *,
    manifest_path: str,
    hsl_checkpoint_path: str,
    policy_checkpoint_path: str,
    result_path: str,
) -> Any:
    """Run only the active v015 evaluator; legacy routes are explicit and separate."""

    # B1: 验证 formal transaction 并构造 strict request, 产出本次 quality evaluation identity.
    if not bool(getattr(getattr(runner, "alg", None), "frontres_formal_transaction_enabled", False)):
        raise RuntimeError(
            "active policy-quality evaluation requires the v015 formal transaction route; "
            "legacy evaluation must use run_frontres_legacy_policy_quality_eval explicitly"
        )
    request = build_frontres_v015_policy_quality_eval_request(
        manifest_path=manifest_path,
        hsl_checkpoint_path=hsl_checkpoint_path,
        policy_checkpoint_path=policy_checkpoint_path,
        result_path=result_path,
    )
    # B2: 消费显式注入的单次 bundle 或创建 request-scoped bundle, 禁止跨 request 缓存.
    owners = getattr(runner, "_frontres_v015_policy_quality_owner_bundle", None)
    if owners is None:
        owners = build_frontres_v015_policy_quality_owner_bundle(runner, request)
    else:
        delattr(runner, "_frontres_v015_policy_quality_owner_bundle")
    if not isinstance(owners, FrontRESV015PolicyQualityOwnerBundle):
        raise RuntimeError("v015 policy-quality rejects a non-v015 formal owner bundle")
    # B3: 将 request 与 owner bundle 交给 matched held-out evaluator, 产出 atomic quality report.
    return run_frontres_v015_policy_quality_heldout_eval(runner, request=request, owners=owners)


def run_frontres_legacy_policy_quality_eval(
    runner: Any,
    *,
    manifest_path: str,
    hsl_checkpoint_path: str,
    policy_checkpoint_path: str,
    result_path: str,
) -> Any:
    """Run the historical matched-route evaluator through an explicit legacy entrypoint."""
    request = build_frontres_policy_quality_eval_request(
        manifest_path=manifest_path,
        hsl_checkpoint_path=hsl_checkpoint_path,
        policy_checkpoint_path=policy_checkpoint_path,
        result_path=result_path,
    )
    # QUALITY-ID-01 legacy route -> real owner bundle -> matched manifest executor.
    # Result: Q-E6 OFFLINE PASS; six historical owner adapters remain explicit.
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


def capture_frontres_v015_policy_quality_dynamic_state_identity(
    runner: Any,
    *,
    comparison_signature: str,
    pair_layout: Any,
) -> FrontRESV015DynamicStateIdentity:
    # B1: 捕获并哈希 physical, command, perturber, RNG 与 sealed scenario state.
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
    env, raw_env = resolve_frontres_policy_quality_envs(runner)
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
    command = resolve_frontres_policy_quality_command(raw_env)
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

    # B2: 将 route-start snapshot 投影为固定字段集, 产出逐字段可比较 values.
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
    # B3: 哈希每个字段并校验 role layout, 产出完整 dynamic-state identity.
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
