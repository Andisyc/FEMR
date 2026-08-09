"""Checkpoint, save, and resume helpers for OnPolicyRunner.

This module owns persistence mechanics. The runner keeps its public methods as
thin wrappers so training loops and external scripts keep the same API.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
import random
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any, Literal, Mapping

import torch
import numpy as np

from rsl_rl.frontres.frontres_value_normalization import (
    FRONTRES_VALUE_NORMALIZATION_ID,
    FRONTRES_VALUE_NORMALIZER_DECAY,
    FRONTRES_VALUE_NORMALIZER_SCALE_FLOOR,
    FrontRESValueNormalizerState,
)
from rsl_rl.modules import FrontRESActorCritic, ResidualActorCritic
from rsl_rl.modules.frontres_observation_layout import (
    FRONTRES_FUTURE_INTENT_DIM,
    FRONTRES_FUTURE_INTENT_LAYOUT_VERSION,
    FRONTRES_FUTURE_INTENT_OFFSETS,
    FRONTRES_V015_GMT_SUFFIX_DIM,
    FrontRESFutureIntentLayout,
    compose_frontres_obs_norm_state,
    extract_frontres_extra_norm_stats,
    frontres_extra_norm_stats_for_save,
    resolve_frontres_v015_observation_authority,
)
from rsl_rl.frontres.frontres_segment_warmup import (
    FRONTRES_V011_MAX_ABSOLUTE_ITERATION,
    FRONTRES_V011_REVIEW_BOUNDARIES,
    FRONTRES_V011_SELECTED_SEGMENT_COUNT,
    frontres_k_stage_schedule_tuple,
    frontres_v013_dr_stage_fingerprint,
    require_frontres_v013_campaign_schedule,
    resolve_frontres_k_stage_identity,
)
from rsl_rl.runners.frontres_checkpoint_quality import (
    EMPIRICAL_NORMALIZER_STATE_KEYS as _EMPIRICAL_NORMALIZER_STATE_KEYS,
    FRONTRES_ACTIVE_CHECKPOINT_FORMAT,
    FRONTRES_ACTIVE_CHECKPOINT_IDENTITY_KEY as _V015_CHECKPOINT_IDENTITY_KEY,
    FRONTRES_ACTIVE_GROUPED_CANDIDATE_LAYOUT as _V015_GROUPED_CANDIDATE_LAYOUT,
    FRONTRES_LEGACY_POLICY_CHECKPOINT_FORMAT as _V015_LEGACY_POLICY_CHECKPOINT_FORMAT,
    FRONTRES_HSL_CHECKPOINT_FORMAT as _V015_HSL_CHECKPOINT_FORMAT,
    FRONTRES_HSL_CHECKPOINT_IDENTITY_KEY as _V015_HSL_CHECKPOINT_IDENTITY_KEY,
    FRONTRES_HSL_PREFIX_NORM_KEY as _V015_HSL_PREFIX_NORM_KEY,
    FRONTRES_HSL_ARTIFACT_TRAINING_CONTRACT_ID as _HSL_ARTIFACT_TRAINING_CONTRACT_ID,
    FRONTRES_HSL_TOP_LEVEL_KEYS as _V015_HSL_TOP_LEVEL_KEYS,
    FrontRESActiveQualityCheckpointIdentity,
    frontres_v015_clone_tensor_state as _v015_clone_tensor_state,
    frontres_v015_committed_transaction_receipt as _v015_committed_transaction_receipt,
    frontres_v015_file_sha256 as _v015_file_sha256,
    frontres_v015_state_dict_fingerprint as _v015_state_dict_fingerprint,
    frontres_v015_tensor_fingerprint as _v015_tensor_fingerprint,
    inspect_frontres_quality_checkpoint,
    load_frontres_checkpoint_mapping,
    validate_frontres_v015_normalizer_state as _validate_v015_normalizer_state,
)
from rsl_rl.runners.frontres_checkpoint_legacy import (
    frontres_legacy_gain_config_payload,
    reject_legacy_frontres_hsl_checkpoint,
    validate_frontres_legacy_gain_config_resume,
)

from rsl_rl.runners.frontres_formal_runtime_audit import (
    configure_formal_runtime_probe,
    emit_formal_runtime_probe,
    formal_runtime_audit_enabled,
    print_checkpoint_payload_audit,
    print_checkpoint_reload_audit,
)
from rsl_rl.runners.frontres_stage3_engine import frontres_stage3_transaction_aggregate


_V015_PHYSICS_EVIDENCE_IDENTITY = {
    "zmp_estimator_id": "contact-wrench-zmp-v1",
    "support_envelope_id": "clean-foot-pose-oriented-box-v1",
    "actual_contact_id": "contact-sensor-net-normal-force-threshold-v1",
    "expected_phase_id": "clean-foot-height-phase-v1",
}


def _frontres_v013_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "keys": torch.as_tensor(numpy_state[1].copy(), dtype=torch.int64),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": tuple(state.clone().cpu() for state in torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else (),
    }


def _restore_frontres_v013_rng_state(payload: Mapping[str, Any]) -> None:
    numpy_state = payload["numpy"]
    random.setstate(tuple(payload["python"]))
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            torch.as_tensor(numpy_state["keys"], dtype=torch.int64).cpu().numpy().astype(np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(torch.as_tensor(payload["torch_cpu"], dtype=torch.uint8).cpu())
    cuda_states = tuple(payload.get("torch_cuda", ()))
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([torch.as_tensor(state, dtype=torch.uint8).cpu() for state in cuda_states])
_V015_TRANSACTION_STATE_ATTR = "_frontres_checkpoint_transaction_state"
_V015_LAST_RECEIPT_ATTR = "_frontres_last_committed_transaction_receipt"


@contextmanager
def frontres_quality_route_actor(
    runner: Any,
    checkpoint_path: str | os.PathLike[str],
    *,
    route: Literal["hsl", "policy"],
    expected_file_sha256: str,
):
    """Temporarily install one strict quality policy and restore every touched field.

    This is an inference-only route switch. The Stage-3 route installs the
    checkpoint Actor and Critic because the local report contains both executed
    actions and value calibration. It never restores optimizer, sampler,
    transaction, curriculum, or warmup state. The caller must still restore the
    environment scenario per route.
    """

    identity = inspect_frontres_quality_checkpoint(checkpoint_path, route=route)
    if identity.file_sha256 != str(expected_file_sha256):
        raise RuntimeError("v015 quality route checkpoint changed after request sealing")
    policy = getattr(getattr(runner, "alg", None), "policy", None)
    actor = getattr(policy, "residual_actor", None)
    critic = getattr(policy, "critic", None)
    prefix_normalizer = getattr(runner, "_frontres_extra_normalizer", None)
    critic_observation_normalizer = getattr(runner, "privileged_obs_normalizer", None)
    uses_v018_critic_coordinates = (
        route == "policy" and getattr(identity, "format", None) == FRONTRES_ACTIVE_CHECKPOINT_FORMAT
    )
    if not isinstance(actor, torch.nn.Module):
        raise RuntimeError("v015 quality route requires the residual actor owner")
    if route == "policy" and not isinstance(critic, torch.nn.Module):
        raise RuntimeError("v015 quality policy route requires the Critic owner")
    if uses_v018_critic_coordinates and not isinstance(critic_observation_normalizer, torch.nn.Module):
        raise RuntimeError("v018 quality policy route requires the Critic observation normalizer owner")
    if prefix_normalizer is not None and not isinstance(prefix_normalizer, torch.nn.Module):
        raise RuntimeError("v015 quality route prefix normalizer owner has an invalid type")
    distribution_key, distribution = _v015_hsl_distribution_state(policy)
    actor_before = copy.deepcopy(actor.state_dict())
    critic_before = copy.deepcopy(critic.state_dict()) if isinstance(critic, torch.nn.Module) else None
    distribution_before = distribution.detach().clone()
    prefix_before = copy.deepcopy(prefix_normalizer.state_dict()) if prefix_normalizer is not None else None
    critic_observation_normalizer_before = (
        copy.deepcopy(critic_observation_normalizer.state_dict())
        if isinstance(critic_observation_normalizer, torch.nn.Module)
        else None
    )
    mean_before = copy.deepcopy(getattr(runner, "_frontres_extra_mean", None))
    std_before = copy.deepcopy(getattr(runner, "_frontres_extra_std", None))
    layout_before = getattr(runner, "_frontres_extra_stats_layout_version", None)
    actor_training = bool(actor.training)
    critic_training = bool(critic.training) if isinstance(critic, torch.nn.Module) else False
    prefix_training = bool(prefix_normalizer.training) if prefix_normalizer is not None else False
    critic_observation_normalizer_training = (
        bool(critic_observation_normalizer.training)
        if isinstance(critic_observation_normalizer, torch.nn.Module)
        else False
    )

    checkpoint = load_frontres_checkpoint_mapping(checkpoint_path, map_location="cpu")
    try:
        if route == "hsl":
            validated = _validate_v015_hsl_checkpoint_resume(
                runner,
                checkpoint,
                stage3_initializer=True,
            )
            if validated is None:
                raise RuntimeError("v015 quality HSL route requires strict HSL-v2 identity")
            actor.load_state_dict(validated["actor_state"], strict=True)
            distribution.data.copy_(
                validated["distribution"].to(device=distribution.device, dtype=distribution.dtype)
            )
            if prefix_normalizer is not None:
                prefix_normalizer.load_state_dict(validated["prefix_state"], strict=True)
                runner._frontres_extra_mean = None
                runner._frontres_extra_std = None
                runner._frontres_extra_stats_layout_version = None
            else:
                runner._frontres_extra_mean = validated["prefix_state"]["_mean"].to(runner.device)
                runner._frontres_extra_std = validated["prefix_state"]["_std"].to(runner.device)
                runner._frontres_extra_stats_layout_version = FRONTRES_FUTURE_INTENT_LAYOUT_VERSION
        else:
            identity_format = getattr(identity, "format", FRONTRES_ACTIVE_CHECKPOINT_FORMAT)
            if identity_format == FRONTRES_ACTIVE_CHECKPOINT_FORMAT:
                validated = _validate_v015_checkpoint_resume(
                    runner,
                    checkpoint,
                    validation_scope="quality_inference",
                )
                if validated is None:
                    raise RuntimeError("v017 quality policy route requires strict checkpoint-v14 identity")
                critic_observation_normalizer.load_state_dict(
                    checkpoint["privileged_obs_norm_state_dict"],
                    strict=True,
                )
            elif identity_format != _V015_LEGACY_POLICY_CHECKPOINT_FORMAT:
                raise RuntimeError("quality policy route received an unsupported checkpoint format")
            model_state = checkpoint.get("model_state_dict")
            if (
                not isinstance(model_state, Mapping)
                or not isinstance(model_state.get("residual_actor"), Mapping)
                or not isinstance(model_state.get("critic"), Mapping)
            ):
                raise RuntimeError("v015 quality policy checkpoint is missing Actor/Critic state")
            actor.load_state_dict(model_state["residual_actor"], strict=True)
            critic.load_state_dict(model_state["critic"], strict=True)
            _copy_policy_noise_state(policy, model_state)
            stats = extract_frontres_extra_norm_stats(
                checkpoint.get("obs_norm_state_dict"),
                int(getattr(policy, "num_actor_obs", 0)),
                int(getattr(runner, "_frontres_gmt_obs_dim", 0)),
                runner.device,
            )
            if stats is None or tuple(stats[0].shape[-1:]) != (158,) or tuple(stats[1].shape[-1:]) != (158,):
                raise RuntimeError("v015 quality policy route requires exact 158D prefix statistics")
            runner._frontres_extra_mean, runner._frontres_extra_std = stats
            runner._frontres_extra_stats_layout_version = FRONTRES_FUTURE_INTENT_LAYOUT_VERSION
        actor.eval()
        if route == "policy":
            critic.eval()
            if uses_v018_critic_coordinates:
                critic_observation_normalizer.eval()
        if prefix_normalizer is not None:
            prefix_normalizer.eval()
        yield identity
    finally:
        actor.load_state_dict(actor_before, strict=True)
        if critic_before is not None:
            critic.load_state_dict(critic_before, strict=True)
        distribution.data.copy_(distribution_before.to(device=distribution.device, dtype=distribution.dtype))
        if prefix_normalizer is not None:
            prefix_normalizer.load_state_dict(prefix_before, strict=True)
        if critic_observation_normalizer_before is not None:
            critic_observation_normalizer.load_state_dict(critic_observation_normalizer_before, strict=True)
        actor.train(actor_training)
        if isinstance(critic, torch.nn.Module):
            critic.train(critic_training)
        if prefix_normalizer is not None:
            prefix_normalizer.train(prefix_training)
        if isinstance(critic_observation_normalizer, torch.nn.Module):
            critic_observation_normalizer.train(critic_observation_normalizer_training)
        runner._frontres_extra_mean = mean_before
        runner._frontres_extra_std = std_before
        runner._frontres_extra_stats_layout_version = layout_before
        if _v015_state_dict_fingerprint(actor.state_dict(), label="restored quality actor") != _v015_state_dict_fingerprint(
            actor_before, label="quality actor snapshot"
        ):
            raise RuntimeError("v015 quality route failed to restore the source actor")
        if critic_before is not None and _v015_state_dict_fingerprint(
            critic.state_dict(), label="restored quality Critic"
        ) != _v015_state_dict_fingerprint(critic_before, label="quality Critic snapshot"):
            raise RuntimeError("v015 quality route failed to restore the source Critic")
        if critic_observation_normalizer_before is not None and _v015_state_dict_fingerprint(
            critic_observation_normalizer.state_dict(), label="restored quality Critic observation normalizer"
        ) != _v015_state_dict_fingerprint(
            critic_observation_normalizer_before,
            label="quality Critic observation normalizer snapshot",
        ):
            raise RuntimeError("v018 quality route failed to restore the source Critic observation normalizer")


def _uses_v015_hsl_checkpoint_identity(runner: Any) -> bool:
    layout = getattr(runner, "_frontres_future_intent_layout", None)
    return bool(getattr(runner, "_frontres_hsl_proposal_context_enabled", False)) and isinstance(
        layout, FrontRESFutureIntentLayout
    )


def capture_frontres_hsl_fresh_reload_shadow(runner: Any) -> Any:
    """Capture an independent pre-warmup owner for the direct HSL reload sentinel."""

    if not _uses_v015_hsl_checkpoint_identity(runner):
        raise RuntimeError("G2-S4 fresh reload requires the active proposal-only HSL route")
    policy = runner.alg.policy
    distribution_key, distribution = _v015_hsl_distribution_state(policy)
    shadow_policy = SimpleNamespace(
        residual_actor=copy.deepcopy(policy.residual_actor).to(device="cpu"),
        gmt_normalizer=copy.deepcopy(policy.gmt_normalizer).to(device="cpu"),
        gmt_policy_obs_dim=int(policy.gmt_policy_obs_dim),
        num_task_corrections=int(policy.num_task_corrections),
        total_output_dim=int(policy.total_output_dim),
        num_actor_obs=int(policy.num_actor_obs),
        num_frontres_obs=int(policy.num_frontres_obs),
    )
    setattr(shadow_policy, distribution_key, distribution.detach().to(device="cpu").clone())
    shadow = SimpleNamespace(
        alg=SimpleNamespace(
            policy=shadow_policy,
            frontres_training_objective="supervised_restore",
            frontres_formal_runtime_audit=True,
        ),
        policy_cfg={"gmt_checkpoint_path": runner.policy_cfg.get("gmt_checkpoint_path")},
        empirical_normalization=True,
        _frontres_hsl_proposal_context_enabled=True,
        _frontres_future_intent_layout=runner._frontres_future_intent_layout,
        _frontres_future_intent_actor_context_dim=int(
            runner._frontres_future_intent_actor_context_dim
        ),
        _frontres_gmt_obs_dim=int(runner._frontres_gmt_obs_dim),
        _frontres_extra_normalizer=copy.deepcopy(runner._frontres_extra_normalizer).to(
            device="cpu"
        ),
        _frontres_extra_mean=None,
        _frontres_extra_std=None,
        _frontres_extra_stats_layout_version=None,
    )
    return shadow


def _frontres_hsl_direct_proposal(policy: Any, actor_input: torch.Tensor) -> torch.Tensor:
    if not isinstance(actor_input, torch.Tensor) or actor_input.ndim != 2 or actor_input.shape[-1] != 158:
        raise RuntimeError("G2-S4 fresh reload requires normalized FEMR input [B,158]")
    raw = policy.residual_actor(actor_input)
    if not isinstance(raw, torch.Tensor) or tuple(raw.shape) != (int(actor_input.shape[0]), 6):
        raise RuntimeError("G2-S4 fresh reload requires a full-6D residual actor output")
    if not bool(torch.isfinite(raw).all().item()):
        raise RuntimeError("G2-S4 fresh reload requires a finite full-6D proposal")
    return raw


def verify_frontres_hsl_fresh_reload(
    shadow: Any,
    *,
    checkpoint_path: str,
    combined_obs: torch.Tensor,
    source_actor_input: torch.Tensor,
    source_proposal: torch.Tensor,
) -> dict[str, Any]:
    """Strictly reload HSL v2 and compare its CPU proposal to the live-device output."""

    if not isinstance(combined_obs, torch.Tensor) or combined_obs.ndim != 2 or combined_obs.shape[-1] != 928:
        raise RuntimeError("G2-S4 fresh reload requires the real combined observation [B,928]")
    combined_cpu = combined_obs.detach().to(device="cpu")
    source_input_cpu = source_actor_input.detach().to(device="cpu")
    source_proposal_device = str(source_proposal.device)
    source_proposal_cpu = source_proposal.detach().to(device="cpu")
    shadow._frontres_extra_normalizer.eval()
    with torch.inference_mode():
        before_input = shadow._frontres_extra_normalizer(combined_cpu[:, :158])
        before_proposal = _frontres_hsl_direct_proposal(shadow.alg.policy, before_input)
    if torch.equal(before_proposal, source_proposal_cpu):
        raise RuntimeError("G2-S4 fresh reload shadow did not differ before checkpoint load")

    load_runner(shadow, checkpoint_path, load_optimizer=True, load_critic=True)
    shadow._frontres_extra_normalizer.eval()
    with torch.inference_mode():
        fresh_input = shadow._frontres_extra_normalizer(combined_cpu[:, :158])
        fresh_proposal = _frontres_hsl_direct_proposal(shadow.alg.policy, fresh_input)
    input_equal = torch.equal(fresh_input, source_input_cpu)
    proposal_bitwise_equal = torch.equal(fresh_proposal, source_proposal_cpu)
    proposal_rtol = 1.0e-5
    proposal_atol = 1.0e-6
    proposal_close = torch.allclose(
        fresh_proposal,
        source_proposal_cpu,
        rtol=proposal_rtol,
        atol=proposal_atol,
    )
    proposal_max_abs_error = float((fresh_proposal - source_proposal_cpu).abs().max().item())
    if not input_equal or not proposal_close:
        raise RuntimeError(
            "G2-S4 fresh reload changed the normalized FEMR input or proposal: "
            f"normalized_158_equal={int(input_equal)} proposal_6_close={int(proposal_close)} "
            f"proposal_6_bitwise_equal={int(proposal_bitwise_equal)} "
            f"max_abs_error={proposal_max_abs_error:.9g} rtol={proposal_rtol:.1e} "
            f"atol={proposal_atol:.1e} source_device={source_proposal_device} shadow_device=cpu"
        )
    result = {
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "normalized_158_equal": True,
        "proposal_6_close": True,
        "proposal_6_bitwise_equal": proposal_bitwise_equal,
        "proposal_6_max_abs_error": proposal_max_abs_error,
        "pre_reload_proposal_equal": False,
    }
    print(
        "[G2-S4-FRESH-RELOAD] "
        f"checkpoint={result['checkpoint_path']} normalized_158_equal=1 "
        f"proposal_6_close=1 proposal_6_bitwise_equal={int(proposal_bitwise_equal)} "
        f"max_abs_error={proposal_max_abs_error:.9g} rtol={proposal_rtol:.1e} "
        f"atol={proposal_atol:.1e} source_device={source_proposal_device} "
        "shadow_device=cpu pre_reload_proposal_equal=0",
        flush=True,
    )
    return result


def _v015_hsl_distribution_state(policy: Any) -> tuple[str, torch.Tensor]:
    available = tuple(name for name in ("std", "log_std") if isinstance(getattr(policy, name, None), torch.Tensor))
    if len(available) != 1:
        raise RuntimeError("proposal-only HSL requires exactly one std or log_std distribution tensor")
    name = available[0]
    value = getattr(policy, name)
    if tuple(value.shape) != (6,) or not bool(torch.isfinite(value).all().item()):
        raise RuntimeError("proposal-only HSL distribution state must be finite [6]")
    return name, value


def _v015_hsl_prefix_normalizer_state(runner: Any) -> dict[str, torch.Tensor]:
    if not bool(getattr(runner, "empirical_normalization", False)):
        raise RuntimeError("proposal-only HSL requires empirical 158D prefix normalizer state")
    normalizer = getattr(runner, "_frontres_extra_normalizer", None)
    if not isinstance(normalizer, torch.nn.Module):
        raise RuntimeError("proposal-only HSL requires the live 158D prefix normalizer owner")
    state = _validate_v015_normalizer_state(
        normalizer.state_dict(),
        dim=158,
        label="proposal-only HSL prefix normalizer",
    )
    return _v015_clone_tensor_state(state, label="proposal-only HSL prefix normalizer")


def _v015_frozen_gmt_identity(runner: Any) -> dict[str, Any]:
    policy = getattr(getattr(runner, "alg", None), "policy", None)
    if int(getattr(policy, "gmt_policy_obs_dim", 0) or 0) != FRONTRES_V015_GMT_SUFFIX_DIM:
        raise RuntimeError("FrontRES checkpoint requires the frozen GMT 770D observation identity")
    normalizer = getattr(policy, "gmt_normalizer", None)
    if not isinstance(normalizer, torch.nn.Module):
        raise RuntimeError("FrontRES checkpoint requires the frozen GMT normalizer")
    state = _validate_v015_normalizer_state(
        normalizer.state_dict(),
        dim=FRONTRES_V015_GMT_SUFFIX_DIM,
        label="frozen GMT normalizer",
    )
    policy_cfg = getattr(runner, "policy_cfg", None)
    gmt_path = policy_cfg.get("gmt_checkpoint_path") if isinstance(policy_cfg, Mapping) else None
    if not isinstance(gmt_path, (str, os.PathLike)):
        raise RuntimeError("FrontRES checkpoint requires gmt_checkpoint_path in the policy config")
    return {
        "checkpoint_sha256": _v015_file_sha256(gmt_path),
        "normalizer_dim": FRONTRES_V015_GMT_SUFFIX_DIM,
        "normalizer_fingerprint": _v015_state_dict_fingerprint(
            state, label="frozen GMT normalizer"
        ),
    }


def _uses_v015_formal_checkpoint_identity(runner: Any) -> bool:
    """判断 runner 是否处于已确认的 v015 formal persistence route."""

    layout = getattr(runner, "_frontres_future_intent_layout", None)
    alg = getattr(runner, "alg", None)
    return isinstance(layout, FrontRESFutureIntentLayout) and bool(
        getattr(alg, "frontres_formal_transaction_enabled", False)
    )


def _v015_checkpoint_layout_fields(runner: Any) -> dict[str, int | str | tuple[int, ...]]:
    """在 persistence 或 restore 前读取并验证 R3 精确 observation authority."""

    layout = getattr(runner, "_frontres_future_intent_layout", None)
    if not isinstance(layout, FrontRESFutureIntentLayout):
        raise RuntimeError("v015 checkpoint identity requires FrontRESFutureIntentLayout")
    layout.validate()
    if layout.version != FRONTRES_FUTURE_INTENT_LAYOUT_VERSION:
        raise RuntimeError("v015 checkpoint identity has an incompatible future-intent layout version")
    if tuple(layout.future_offsets) != FRONTRES_FUTURE_INTENT_OFFSETS:
        raise RuntimeError(
            f"v015 checkpoint identity requires future_offsets={FRONTRES_FUTURE_INTENT_OFFSETS}, "
            f"got {tuple(layout.future_offsets)}"
        )
    policy = getattr(getattr(runner, "alg", None), "policy", None)
    actor_dim = getattr(policy, "num_actor_obs", None)
    prefix_dim = getattr(policy, "num_frontres_obs", None)
    gmt_dim = getattr(runner, "_frontres_gmt_obs_dim", None)
    if actor_dim is None or prefix_dim is None or gmt_dim is None:
        raise RuntimeError("v015 checkpoint identity requires actor, prefix, and GMT observation dimensions")
    actor_dim = int(actor_dim)
    prefix_dim = int(prefix_dim)
    gmt_dim = int(gmt_dim)
    environment_obs_dim = actor_dim - layout.actor_tail_dim
    current_frontres_prefix_dim = prefix_dim - layout.actor_tail_dim
    try:
        authority = resolve_frontres_v015_observation_authority(
            environment_obs_dim=environment_obs_dim,
            configured_frontres_prefix_dim=current_frontres_prefix_dim,
            actor_tail_dim=layout.actor_tail_dim,
            gmt_suffix_dim=FRONTRES_V015_GMT_SUFFIX_DIM,
        )
    except ValueError as exc:
        raise RuntimeError(
            "v015 checkpoint actor layout is inconsistent: "
            f"actor={actor_dim} prefix={prefix_dim} gmt={gmt_dim} tail={layout.actor_tail_dim}; {exc}"
        ) from exc
    if (
        actor_dim != authority.combined_obs_dim
        or prefix_dim != authority.frontres_visible_dim
        or gmt_dim != authority.gmt_suffix_dim
    ):
        raise RuntimeError(
            "v015 checkpoint actor layout is inconsistent with R3 authority: "
            f"actor={actor_dim} prefix={prefix_dim} gmt={gmt_dim}; "
            f"expected={authority.combined_obs_dim}/{authority.frontres_visible_dim}/{authority.gmt_suffix_dim}"
        )
    configured_tail = int(getattr(runner, "_frontres_future_intent_actor_context_dim", 0) or 0)
    if configured_tail != layout.actor_tail_dim:
        raise RuntimeError(
            "v015 checkpoint actor tail disagrees with its resolved layout: "
            f"configured={configured_tail} layout={layout.actor_tail_dim}"
        )
    return {
        "layout_version": layout.version,
        "future_offsets": tuple(int(value) for value in layout.future_offsets),
        "intent_dim": FRONTRES_FUTURE_INTENT_DIM,
        "actor_tail_dim": layout.actor_tail_dim,
        "environment_obs_dim": authority.environment_obs_dim,
        "current_frontres_prefix_dim": authority.current_frontres_prefix_dim,
        "actor_dim": actor_dim,
        "prefix_dim": prefix_dim,
        "gmt_dim": gmt_dim,
    }


def _v016_checkpoint_critic_identity(runner: Any) -> dict[str, Any]:
    """Validate and publish the sole TRAIN-v018 Critic/clip persistence identity."""

    # B1: 校验 runtime contract fields 与 449D first layer, 产出可信 Critic identity.
    alg = getattr(runner, "alg", None)
    policy = getattr(alg, "policy", None)
    critic = getattr(policy, "critic", None)
    first_linear = next(
        (module for module in critic.modules() if isinstance(module, torch.nn.Linear)),
        None,
    ) if isinstance(critic, torch.nn.Module) else None
    expected_fields = {
        "frontres_method_contract_id": "FRS-METHOD-v020",
        "frontres_optimization_contract_id": "FRS-PPO-v008",
        "frontres_training_contract_id": "FRS-TRAIN-v019",
        "frontres_critic_value_kind": "state_value",
        "frontres_critic_input_dim": 449,
        "frontres_critic_action_conditioned": False,
        "frontres_critic_target_id": "segment-exact-m-mean-symlog-v1",
        "frontres_return_utility_id": "symmetric-log-gain-g0-1-v1",
        "frontres_critic_support_context_id": "action-pre-support-plan-kmax32-v1",
        "frontres_gradient_clip_identity": "separate-actor-critic-v1",
    }
    for name, expected in expected_fields.items():
        if getattr(alg, name, None) != expected:
            raise RuntimeError(f"checkpoint-v14 requires {name}={expected!r}")
    if (
        not isinstance(first_linear, torch.nn.Linear)
        or int(first_linear.in_features) != 449
        or int(getattr(runner, "_frontres_critic_observation_dim", 0) or 0) != 449
    ):
        raise RuntimeError("checkpoint-v14 requires one exact 449D state-value Critic input")
    if float(getattr(alg, "max_grad_norm", float("nan"))) != 0.5:
        raise RuntimeError("checkpoint-v14 requires separate Actor/Critic max_grad_norm=0.5")
    if float(getattr(alg, "frontres_return_utility_scale", float("nan"))) != 1.0:
        raise RuntimeError("checkpoint-v14 requires fixed return utility G0=1")

    # B2: 发布 Critic semantics, 不从 model shape 猜测 target 或 action authority.
    return {
        "value_kind": "state_value",
        "input_dim": 449,
        "action_conditioned": False,
        "target_id": "segment-exact-m-mean-symlog-v1",
        "return_utility_id": "symmetric-log-gain-g0-1-v1",
        "return_utility_scale": 1.0,
        "support_context_id": "action-pre-support-plan-kmax32-v1",
    }


def _v017_checkpoint_value_normalizer_identity(runner: Any) -> dict[str, Any]:
    alg = getattr(runner, "alg", None)
    normalization_id = str(getattr(alg, "frontres_critic_value_normalization", "")).lower()
    decay = float(getattr(alg, "frontres_critic_value_normalizer_decay", float("nan")))
    scale_floor = float(getattr(alg, "frontres_critic_value_normalizer_scale_floor", float("nan")))
    state = getattr(alg, "frontres_critic_value_normalizer_state", None)
    if (
        normalization_id != FRONTRES_VALUE_NORMALIZATION_ID
        or decay != FRONTRES_VALUE_NORMALIZER_DECAY
        or scale_floor != FRONTRES_VALUE_NORMALIZER_SCALE_FLOOR
        or not isinstance(state, FrontRESValueNormalizerState)
    ):
        raise RuntimeError("checkpoint-v14 requires the fixed Critic value-normalizer identity and state")
    state.validate()
    return {
        "identity": normalization_id,
        "decay": decay,
        "scale_floor": scale_floor,
    }


def _build_v015_hsl_checkpoint_payload(runner: Any) -> dict[str, Any]:
    """Build the frozen v014 HSL-v2 artifact consumed by fresh Stage-3 runs."""

    # B1: 校验 Stage-1 route/layout/action owner, 产出可序列化运行身份.
    if not _uses_v015_hsl_checkpoint_identity(runner):
        raise RuntimeError("proposal-only HSL checkpoint save requires the active Stage-1 route")
    alg = getattr(runner, "alg", None)
    if str(getattr(alg, "frontres_training_objective", "")) != "supervised_restore":
        raise RuntimeError("proposal-only HSL checkpoint requires the supervised_restore objective")
    fields = _v015_checkpoint_layout_fields(runner)
    if (
        fields["environment_obs_dim"] != 870
        or fields["current_frontres_prefix_dim"] != 100
        or fields["actor_dim"] != 928
        or fields["prefix_dim"] != 158
        or fields["gmt_dim"] != 770
        or fields["actor_tail_dim"] != 58
    ):
        raise RuntimeError("proposal-only HSL checkpoint requires exact 870/928/158/770 observation authority")
    policy = getattr(alg, "policy", None)
    actor = getattr(policy, "residual_actor", None)
    if not isinstance(actor, torch.nn.Module):
        raise RuntimeError("proposal-only HSL checkpoint requires the residual actor owner")
    if int(getattr(policy, "num_task_corrections", 0) or 0) != 6 or int(
        getattr(policy, "total_output_dim", 0) or 0
    ) != 6:
        raise RuntimeError("proposal-only HSL checkpoint requires full 6D Delta SE(3) action identity")
    actor_state = _v015_clone_tensor_state(
        actor.state_dict(), label="proposal-only HSL residual actor"
    )
    distribution_key, distribution = _v015_hsl_distribution_state(policy)
    prefix_state = _v015_hsl_prefix_normalizer_state(runner)
    # B2: 封存 Actor/distribution/normalizer fingerprint, 产出 proposal-only payload identity.
    model_keys = {"residual_actor", distribution_key}
    payload_identity = {
        "top_level_keys": tuple(sorted(_V015_HSL_TOP_LEVEL_KEYS)),
        "model_keys": tuple(sorted(model_keys)),
        "residual_actor_fingerprint": _v015_state_dict_fingerprint(
            actor_state, label="proposal-only HSL residual actor"
        ),
        "distribution_key": distribution_key,
        "distribution_fingerprint": _v015_tensor_fingerprint(distribution),
        "prefix_normalizer_keys": tuple(sorted(_EMPIRICAL_NORMALIZER_STATE_KEYS)),
        "prefix_normalizer_fingerprint": _v015_state_dict_fingerprint(
            prefix_state, label="proposal-only HSL prefix normalizer"
        ),
    }
    identity = {
        "format": _V015_HSL_CHECKPOINT_FORMAT,
        "method_contract_id": "FRS-METHOD-v017",
        "training_contract_id": _HSL_ARTIFACT_TRAINING_CONTRACT_ID,
        "objective": "proposal_only_current_antidr_delta_se3",
        "future_intent_layout": fields,
        "action": {
            "kind": "delta_se3",
            "dim": 6,
            "semantics": "direct-world-full6-v1",
        },
        "gmt": _v015_frozen_gmt_identity(runner),
        "payload": payload_identity,
    }
    # B3: 只输出 HSL 允许恢复的三类状态, 不携带 Critic/optimizer/transaction.
    return {
        _V015_HSL_CHECKPOINT_IDENTITY_KEY: identity,
        "model_state_dict": {
            "residual_actor": actor_state,
            distribution_key: distribution.detach().clone(),
        },
        _V015_HSL_PREFIX_NORM_KEY: prefix_state,
    }


def _validate_v015_hsl_tensor_state(
    candidate: Any,
    runtime: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> Mapping[str, torch.Tensor]:
    if not isinstance(candidate, Mapping) or set(candidate) != set(runtime):
        raise RuntimeError(f"proposal-only HSL {label} has an incompatible tensor schema")
    for name, runtime_value in runtime.items():
        value = candidate[name]
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != tuple(runtime_value.shape)
            or value.dtype != runtime_value.dtype
            or (torch.is_floating_point(value) and not bool(torch.isfinite(value).all().item()))
        ):
            raise RuntimeError(f"proposal-only HSL {label} tensor {name} is incompatible")
    return candidate


def _validate_v015_hsl_checkpoint_resume(
    runner: Any,
    checkpoint: Mapping[str, Any],
    *,
    stage3_initializer: bool = False,
) -> dict[str, Any] | None:
    """Validate the frozen HSL-v2 identity before any runner state mutation."""

    # B1: 判定 Stage-1/Stage-3 initializer route, 产出严格 envelope 与 objective 边界.
    uses_hsl = _uses_v015_hsl_checkpoint_identity(runner) or bool(stage3_initializer)
    has_hsl_identity = isinstance(checkpoint, Mapping) and _V015_HSL_CHECKPOINT_IDENTITY_KEY in checkpoint
    if not uses_hsl:
        if has_hsl_identity:
            raise RuntimeError("proposal-only HSL checkpoint requires the active Stage-1 HSL route")
        return None
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("proposal-only HSL checkpoint payload must be a mapping")
    if _V015_HSL_CHECKPOINT_IDENTITY_KEY not in checkpoint:
        raise RuntimeError("proposal-only HSL checkpoint identity is missing; legacy or unversioned payload rejected")
    if set(checkpoint) != _V015_HSL_TOP_LEVEL_KEYS:
        raise RuntimeError("proposal-only HSL checkpoint requires the exact payload field set")
    expected_runtime_objective = "segment_replay_hrl" if stage3_initializer else "supervised_restore"
    if str(getattr(getattr(runner, "alg", None), "frontres_training_objective", "")) != expected_runtime_objective:
        raise RuntimeError(
            "proposal-only HSL checkpoint runtime objective mismatch: "
            f"expected={expected_runtime_objective}"
        )
    identity = checkpoint.get(_V015_HSL_CHECKPOINT_IDENTITY_KEY)
    required_identity = {
        "format",
        "method_contract_id",
        "training_contract_id",
        "objective",
        "future_intent_layout",
        "action",
        "gmt",
        "payload",
    }
    if not isinstance(identity, Mapping) or set(identity) != required_identity:
        raise RuntimeError("proposal-only HSL checkpoint identity is missing, legacy, or malformed")
    # B2: 对齐 HSL artifact/layout/action/GMT identity, 产出可信 payload schema.
    if (
        identity["format"] != _V015_HSL_CHECKPOINT_FORMAT
        or identity["method_contract_id"] != "FRS-METHOD-v017"
        or identity["training_contract_id"] != _HSL_ARTIFACT_TRAINING_CONTRACT_ID
        or identity["objective"] != "proposal_only_current_antidr_delta_se3"
    ):
        raise RuntimeError("proposal-only HSL checkpoint has an incompatible identity")
    fields = _v015_checkpoint_layout_fields(runner)
    if identity["future_intent_layout"] != fields:
        raise RuntimeError("proposal-only HSL checkpoint future-intent layout mismatch")
    if identity["action"] != {
        "kind": "delta_se3",
        "dim": 6,
        "semantics": "direct-world-full6-v1",
    }:
        raise RuntimeError("proposal-only HSL checkpoint action identity mismatch")
    if identity["gmt"] != _v015_frozen_gmt_identity(runner):
        raise RuntimeError("proposal-only HSL checkpoint GMT artifact or normalizer identity mismatch")

    payload_identity = identity["payload"]
    required_payload_identity = {
        "top_level_keys",
        "model_keys",
        "residual_actor_fingerprint",
        "distribution_key",
        "distribution_fingerprint",
        "prefix_normalizer_keys",
        "prefix_normalizer_fingerprint",
    }
    if not isinstance(payload_identity, Mapping) or set(payload_identity) != required_payload_identity:
        raise RuntimeError("proposal-only HSL checkpoint payload identity is malformed")
    if tuple(payload_identity["top_level_keys"]) != tuple(sorted(_V015_HSL_TOP_LEVEL_KEYS)):
        raise RuntimeError("proposal-only HSL checkpoint top-level identity mismatch")

    # B3: 校验三类 tensor state 与 fingerprint, 产出 restore 所需 immutable carrier.
    policy = getattr(getattr(runner, "alg", None), "policy", None)
    actor = getattr(policy, "residual_actor", None)
    if not isinstance(actor, torch.nn.Module):
        raise RuntimeError("proposal-only HSL checkpoint requires the residual actor owner")
    model_state = checkpoint["model_state_dict"]
    distribution_key, runtime_distribution = _v015_hsl_distribution_state(policy)
    expected_model_keys = {"residual_actor", distribution_key}
    if (
        not isinstance(model_state, Mapping)
        or set(model_state) != expected_model_keys
        or tuple(payload_identity["model_keys"]) != tuple(sorted(expected_model_keys))
        or payload_identity["distribution_key"] != distribution_key
    ):
        raise RuntimeError("proposal-only HSL checkpoint model payload is not actor/distribution-only")
    actor_state = _validate_v015_hsl_tensor_state(
        model_state["residual_actor"],
        actor.state_dict(),
        label="residual actor",
    )
    if _v015_state_dict_fingerprint(actor_state, label="proposal-only HSL residual actor") != payload_identity[
        "residual_actor_fingerprint"
    ]:
        raise RuntimeError("proposal-only HSL residual actor fingerprint mismatch")
    distribution = model_state[distribution_key]
    if (
        not isinstance(distribution, torch.Tensor)
        or tuple(distribution.shape) != tuple(runtime_distribution.shape)
        or distribution.dtype != runtime_distribution.dtype
        or not bool(torch.isfinite(distribution).all().item())
        or _v015_tensor_fingerprint(distribution) != payload_identity["distribution_fingerprint"]
    ):
        raise RuntimeError("proposal-only HSL distribution fingerprint or schema mismatch")

    prefix_normalizer = getattr(runner, "_frontres_extra_normalizer", None)
    if not isinstance(prefix_normalizer, torch.nn.Module):
        raise RuntimeError("proposal-only HSL checkpoint requires the live prefix normalizer owner")
    runtime_prefix_state = prefix_normalizer.state_dict()
    prefix_state = _validate_v015_hsl_tensor_state(
        checkpoint[_V015_HSL_PREFIX_NORM_KEY],
        runtime_prefix_state,
        label="prefix normalizer",
    )
    _validate_v015_normalizer_state(
        prefix_state,
        dim=158,
        label="proposal-only HSL prefix normalizer",
    )
    if (
        set(prefix_state) != _EMPIRICAL_NORMALIZER_STATE_KEYS
        or tuple(payload_identity["prefix_normalizer_keys"])
        != tuple(sorted(_EMPIRICAL_NORMALIZER_STATE_KEYS))
        or _v015_state_dict_fingerprint(prefix_state, label="proposal-only HSL prefix normalizer")
        != payload_identity["prefix_normalizer_fingerprint"]
    ):
        raise RuntimeError("proposal-only HSL prefix normalizer fingerprint mismatch")
    return {
        "identity": dict(identity),
        "actor_state": actor_state,
        "distribution_key": distribution_key,
        "distribution": distribution,
        "prefix_state": prefix_state,
    }


def _restore_v015_hsl_checkpoint(runner: Any, validated: Mapping[str, Any], *, path: str) -> bool:
    """Restore only the three states admitted by the validated HSL envelope."""

    policy = runner.alg.policy
    policy.residual_actor.load_state_dict(validated["actor_state"], strict=True)
    distribution = getattr(policy, validated["distribution_key"])
    distribution.data.copy_(validated["distribution"].to(device=distribution.device, dtype=distribution.dtype))
    runner._frontres_extra_normalizer.load_state_dict(validated["prefix_state"], strict=True)
    runner._frontres_extra_mean = None
    runner._frontres_extra_std = None
    runner._frontres_extra_stats_layout_version = None
    runner._frontres_warmup_complete = True
    runner._frontres_last_loaded_checkpoint_path = os.path.abspath(path)
    return True


def _validate_v015_stage3_hsl_initializer_runtime(runner: Any) -> None:
    """Require a fresh formal Stage-3 owner before actor-only HSL migration."""

    alg = getattr(runner, "alg", None)
    if str(getattr(runner, "training_type", "")) != "frontres" or alg is None:
        raise RuntimeError("v015 HSL initializer requires a FrontRES Stage-3 runner")
    if str(getattr(alg, "frontres_training_objective", "")) != "segment_replay_hrl":
        raise RuntimeError("v015 HSL initializer requires the segment_replay_hrl objective")
    if not bool(getattr(alg, "frontres_formal_transaction_enabled", False)):
        raise RuntimeError("v015 HSL initializer requires the formal transaction configuration")
    if str(getattr(alg, "frontres_segment_advantage_normalization", "")) != "grouped_scale_only":
        raise RuntimeError("v015 HSL initializer requires grouped_scale_only configuration")
    if tuple(int(value) for value in (getattr(alg, "frontres_future_offsets", ()) or ())) != (1, 2):
        raise RuntimeError("v015 HSL initializer requires future_offsets=(1, 2)")
    if str(getattr(alg, "frontres_future_intent_layout_version", "")) != FRONTRES_FUTURE_INTENT_LAYOUT_VERSION:
        raise RuntimeError("v015 HSL initializer requires the exact future-intent layout version")
    if any(
        bool(getattr(alg, name, False))
        for name in ("frontres_hsl_init_enabled", "frontres_hsl_rollout_label_enabled")
    ):
        raise RuntimeError("v015 HSL initializer requires HSL flags to be closed before migration")
    if any(
        abs(float(getattr(alg, name, 0.0) or 0.0)) > 1.0e-12
        for name in ("lambda_supervised", "lambda_supervised_min")
    ):
        raise RuntimeError("v015 HSL initializer rejects Stage-3 supervised loss")
    if bool(getattr(runner, "cfg", {}).get("is_full_resume", True)):
        raise RuntimeError("v015 HSL initializer is actor initialization, not full resume")
    if int(getattr(runner, "current_learning_iteration", 0)) != 0:
        raise RuntimeError("v015 HSL initializer must run before the first Stage-3 iteration")
    transaction = getattr(runner, _V015_TRANSACTION_STATE_ATTR, None)
    if transaction is not None and transaction != {"state": "idle"}:
        raise RuntimeError("v015 HSL initializer rejects existing transaction state")


def load_frontres_hsl_initializer(runner: Any, path: str) -> dict[str, Any]:
    """Load only actor, 6D distribution, and 158D prefix stats into fresh Stage 3."""

    _validate_v015_stage3_hsl_initializer_runtime(runner)
    checkpoint_path = Path(path).expanduser().resolve(strict=True)
    if not checkpoint_path.is_file():
        raise RuntimeError(f"v015 HSL initializer is not a file: {checkpoint_path}")
    checkpoint = load_frontres_checkpoint_mapping(checkpoint_path, map_location="cpu")
    validated = _validate_v015_hsl_checkpoint_resume(
        runner,
        checkpoint,
        stage3_initializer=True,
    )
    if validated is None:
        raise RuntimeError("v015 HSL initializer identity is missing")
    _restore_v015_hsl_checkpoint(runner, validated, path=str(checkpoint_path))
    distribution_key = str(validated["distribution_key"])
    identity = dict(validated["identity"])
    receipt = {
        "format": identity["format"],
        "future_intent_layout": dict(identity["future_intent_layout"]),
        "restored": ("residual_actor", distribution_key, _V015_HSL_PREFIX_NORM_KEY),
    }
    print(
        "[FrontRES v017 Stage-3 HSL Init] "
        f"checkpoint={checkpoint_path} format={receipt['format']} "
        f"restored={receipt['restored']} critic=False optimizer=False sampler=False transaction=False",
        flush=True,
    )
    return receipt


def _v015_transaction_checkpoint_payload(runner: Any) -> dict[str, Any]:
    """拒绝 in-flight work, 不序列化 partial candidate batch 或 reference."""

    state = getattr(runner, _V015_TRANSACTION_STATE_ATTR, None)
    last_receipt = getattr(runner, _V015_LAST_RECEIPT_ATTR, None)
    if state is None:
        if isinstance(last_receipt, Mapping):
            return {
                "state": "committed",
                "receipt": _v015_committed_transaction_receipt(
                    {"state": "committed", "receipt": last_receipt}
                ),
            }
        return {"state": "idle"}
    if not isinstance(state, Mapping):
        raise RuntimeError("v015 checkpoint transaction state must be a mapping")
    phase = str(state.get("state", ""))
    if phase == "idle":
        if isinstance(last_receipt, Mapping):
            return {
                "state": "committed",
                "receipt": _v015_committed_transaction_receipt(
                    {"state": "committed", "receipt": last_receipt}
                ),
            }
        return {"state": "idle"}
    if phase == "committed":
        return {"state": "committed", "receipt": _v015_committed_transaction_receipt(state)}
    if phase in {"collecting", "sealed", "failed"}:
        raise RuntimeError(
            "v015 checkpoint save rejects an in-flight formal transaction; "
            f"state={phase}"
        )
    raise RuntimeError(f"v015 checkpoint transaction has unknown state={phase!r}")


def _validate_v013_receipt_curriculum(
    transaction: Mapping[str, Any],
    *,
    schedule: tuple[tuple[object, ...], ...],
    current_iteration: int,
) -> None:
    """Bind the last committed update to the stage immediately before save."""

    if transaction.get("state") != "committed":
        return
    receipt = transaction["receipt"]
    expected_iteration = int(current_iteration) - 1
    if expected_iteration < 0 or int(receipt["training_iteration"]) != expected_iteration:
        raise RuntimeError("FRS-TRAIN-v019 committed receipt is not adjacent to checkpoint iteration")
    expected = resolve_frontres_k_stage_identity(
        schedule=schedule,
        committed_update_iteration=expected_iteration,
        max_horizon_k=max(int(row[0]) for row in schedule),
    )
    if (
        receipt["curriculum_fingerprint"] != expected.schedule_fingerprint
        or int(receipt["k_stage_index"]) != expected.stage_index
        or int(receipt["active_k"]) != expected.active_k
        or int(receipt["active_m"]) != expected.active_m
        or int(receipt["selected_segment_count"]) != FRONTRES_V011_SELECTED_SEGMENT_COUNT
        or int(receipt["policy_row_count"]) != FRONTRES_V011_SELECTED_SEGMENT_COUNT * expected.active_m
        or int(receipt["role_row_count"]) != 2 * FRONTRES_V011_SELECTED_SEGMENT_COUNT * expected.active_m
        or int(receipt["k_stage_iteration"]) != expected.stage_iteration
        or receipt.get("dr_stage_fingerprint") != expected.dr_stage_fingerprint
        or float(receipt.get("dr_progress", -1.0)) != expected.dr_progress
        or float(receipt.get("d_cap", -1.0)) != expected.d_cap
    ):
        raise RuntimeError("FRS-TRAIN-v019 committed receipt has a mismatched K x M x DR stage identity")


def _build_v015_checkpoint_identity(
    runner: Any,
    *,
    obs_norm_state: Mapping[str, Any] | None,
    extra_mean: torch.Tensor | None,
    extra_std: torch.Tensor | None,
) -> dict[str, Any]:
    """从 active runner state 构造唯一允许的 v015 persistence identity."""

    fields = _v015_checkpoint_layout_fields(runner)
    critic_identity = _v016_checkpoint_critic_identity(runner)
    alg = getattr(runner, "alg", None)
    normalization_enabled = bool(getattr(runner, "empirical_normalization", False))
    if str(getattr(alg, "frontres_segment_advantage_normalization", "")) != "grouped_scale_only":
        raise RuntimeError("v015 checkpoint identity requires grouped_scale_only reduction")
    if str(getattr(alg, "frontres_training_objective", "")) != "segment_replay_hrl":
        raise RuntimeError("v015 checkpoint identity requires the segment_replay_hrl objective")
    if normalization_enabled:
        if not isinstance(extra_mean, torch.Tensor) or not isinstance(extra_std, torch.Tensor):
            raise RuntimeError("v015 checkpoint identity requires exact q29-prefix normalizer statistics")
        if int(extra_mean.shape[-1]) != fields["prefix_dim"] or int(extra_std.shape[-1]) != fields["prefix_dim"]:
            raise RuntimeError("v015 checkpoint prefix normalizer shape disagrees with the actor layout")
        if not isinstance(obs_norm_state, Mapping):
            raise RuntimeError("v015 checkpoint identity requires persisted observation normalizer state")
        norm_mean = obs_norm_state.get("_mean")
        norm_std = obs_norm_state.get("_std")
        expected_dim = int(fields["prefix_dim"]) + int(fields["gmt_dim"])
        if (
            not isinstance(norm_mean, torch.Tensor)
            or not isinstance(norm_std, torch.Tensor)
            or int(norm_mean.shape[-1]) != expected_dim
            or int(norm_std.shape[-1]) != expected_dim
        ):
            raise RuntimeError("v015 checkpoint combined normalizer state has an incompatible layout")
        normalizer = {
            "mode": "empirical_prefix_plus_frozen_gmt",
            "prefix_layout_version": fields["layout_version"],
            "prefix_dim": int(fields["prefix_dim"]),
            "combined_dim": expected_dim,
            "prefix_stats_fingerprint": _v015_tensor_fingerprint(extra_mean, extra_std),
        }
        _validate_v015_normalizer_state(
            runner.privileged_obs_normalizer.state_dict(),
            dim=449,
            label="checkpoint-v14 Critic normalizer",
        )
    else:
        normalizer = {
            "mode": "disabled",
            "prefix_layout_version": fields["layout_version"],
            "prefix_dim": int(fields["prefix_dim"]),
            "combined_dim": None,
            "prefix_stats_fingerprint": None,
        }
    iteration = int(getattr(runner, "current_learning_iteration", 0))
    if iteration < 0 or iteration > FRONTRES_V011_MAX_ABSOLUTE_ITERATION:
        raise RuntimeError("FRS-TRAIN-v019 checkpoint iteration must be within [0,8000]")
    schedule = tuple(getattr(alg, "frontres_segment_k_curriculum", ()) or ())
    require_frontres_v013_campaign_schedule(schedule)
    curriculum = resolve_frontres_k_stage_identity(
        schedule=schedule,
        committed_update_iteration=iteration,
        max_horizon_k=int(getattr(alg, "frontres_segment_max_horizon_k", 0)),
    )
    configured_fingerprint = str(getattr(alg, "frontres_segment_k_curriculum_fingerprint", "") or "")
    if configured_fingerprint and configured_fingerprint != curriculum.schedule_fingerprint:
        raise RuntimeError("FRS-TRAIN-v019 checkpoint curriculum fingerprint drifted after config resolution")
    schedule_tuple = frontres_k_stage_schedule_tuple(schedule)
    transaction = _v015_transaction_checkpoint_payload(runner)
    _validate_v013_receipt_curriculum(
        transaction,
        schedule=schedule_tuple,
        current_iteration=iteration,
    )
    return {
        "format": FRONTRES_ACTIVE_CHECKPOINT_FORMAT,
        "method_contract_id": "FRS-METHOD-v020",
        "training_contract_id": "FRS-TRAIN-v019",
        "dr_curriculum_schema_id": "nested-k-dr-four-class-v1",
        "gain_contract_id": "FRS-GAIN-v008",
        "optimization_contract_id": "FRS-PPO-v008",
        "scalar_target_id": "symmetric-log-recovery-aware-utility-v1",
        "return_utility": {
            "identity": "symmetric-log-gain-g0-1-v1",
            "scale": 1.0,
            "placement": "per-attempt-before-exact-m-mean",
        },
        "physics_schema_id": "clean-anchored-contact-zmp-survival-v1",
        "physics_evidence": dict(_V015_PHYSICS_EVIDENCE_IDENTITY),
        "grouped_schema_id": "grouped-all-attempt-scalar-v1",
        "critic": critic_identity,
        "critic_value_normalizer": _v017_checkpoint_value_normalizer_identity(runner),
        "gradient_clip": {
            "identity": "separate-actor-critic-v1",
            "max_norm": 0.5,
        },
        "action": {
            "kind": "delta_se3",
            "dim": 6,
            "semantics": "direct-world-full6-v1",
        },
        "gain": {"beta": float(getattr(alg, "frontres_gain_beta", 0.02))},
        "gmt": _v015_frozen_gmt_identity(runner),
        "future_intent_layout": fields,
        "normalizer": normalizer,
        "grouped_loss": {
            "advantage_normalization": "grouped_scale_only",
            "candidate_layout_version": _V015_GROUPED_CANDIDATE_LAYOUT,
            "policy_rows_per_attempt": 1,
        },
        "transaction": transaction,
        "curriculum": {
            "schedule": frontres_k_stage_schedule_tuple(schedule),
            "schedule_fingerprint": curriculum.schedule_fingerprint,
            "k_stage_index": curriculum.stage_index,
            "active_k": curriculum.active_k,
            "active_m": curriculum.active_m,
            "selected_segment_count": FRONTRES_V011_SELECTED_SEGMENT_COUNT,
            "policy_row_count": FRONTRES_V011_SELECTED_SEGMENT_COUNT * curriculum.active_m,
            "role_row_count": 2 * FRONTRES_V011_SELECTED_SEGMENT_COUNT * curriculum.active_m,
            "maximum_absolute_iteration": FRONTRES_V011_MAX_ABSOLUTE_ITERATION,
            "checkpoint_review_boundaries": FRONTRES_V011_REVIEW_BOUNDARIES,
            "stage_iteration": curriculum.stage_iteration,
            "absolute_iteration": curriculum.absolute_iteration,
            "phase": curriculum.phase.name,
            "phase_iteration": curriculum.phase.phase_iteration,
            "actor_loss_weight": curriculum.phase.actor_loss_weight,
            "dr_stage_fingerprint": curriculum.dr_stage_fingerprint,
            "dr_progress": curriculum.dr_progress,
            "d_cap": curriculum.d_cap,
        },
    }


def _validate_v015_checkpoint_resume(
    runner: Any,
    checkpoint: Mapping[str, Any],
    *,
    validation_scope: Literal["resume", "quality_inference"] = "resume",
) -> dict[str, Any] | None:
    """在 sampler/actor/normalizer/optimizer restore 前校验 v015 identity."""

    if validation_scope not in {"resume", "quality_inference"}:
        raise ValueError(f"unknown v015 checkpoint validation scope: {validation_scope}")

    if not _uses_v015_formal_checkpoint_identity(runner):
        return None
    identity = checkpoint.get(_V015_CHECKPOINT_IDENTITY_KEY)
    if not isinstance(identity, Mapping):
        raise RuntimeError(
            "v015 formal resume requires frontres_v015_checkpoint_identity; "
            "legacy or unversioned checkpoints are forbidden"
        )
    if (
        identity.get("format") != FRONTRES_ACTIVE_CHECKPOINT_FORMAT
        or identity.get("method_contract_id") != "FRS-METHOD-v020"
        or identity.get("training_contract_id") != "FRS-TRAIN-v019"
        or identity.get("dr_curriculum_schema_id") != "nested-k-dr-four-class-v1"
        or identity.get("gain_contract_id") != "FRS-GAIN-v008"
        or identity.get("optimization_contract_id") != "FRS-PPO-v008"
        or identity.get("scalar_target_id") != "symmetric-log-recovery-aware-utility-v1"
        or identity.get("return_utility")
        != {
            "identity": "symmetric-log-gain-g0-1-v1",
            "scale": 1.0,
            "placement": "per-attempt-before-exact-m-mean",
        }
        or identity.get("physics_schema_id") != "clean-anchored-contact-zmp-survival-v1"
        or identity.get("physics_evidence") != _V015_PHYSICS_EVIDENCE_IDENTITY
        or identity.get("grouped_schema_id") != "grouped-all-attempt-scalar-v1"
        or identity.get("action")
        != {"kind": "delta_se3", "dim": 6, "semantics": "direct-world-full6-v1"}
        or identity.get("critic") != _v016_checkpoint_critic_identity(runner)
        or identity.get("critic_value_normalizer") != _v017_checkpoint_value_normalizer_identity(runner)
        or identity.get("gradient_clip")
        != {"identity": "separate-actor-critic-v1", "max_norm": 0.5}
    ):
        raise RuntimeError("v015 checkpoint has an incompatible contract or format identity")
    if identity.get("gmt") != _v015_frozen_gmt_identity(runner):
        raise RuntimeError("v015 checkpoint frozen GMT artifact, 770D layout, or normalizer identity mismatch")
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, Mapping) or not isinstance(model_state.get("residual_actor"), Mapping) or not isinstance(
        model_state.get("critic"), Mapping
    ):
        raise RuntimeError("v015 checkpoint is missing exact actor/Critic state")
    if not isinstance(checkpoint.get("optimizer_state_dict"), Mapping):
        raise RuntimeError("v015 checkpoint is missing optimizer state")
    try:
        value_normalizer_state = FrontRESValueNormalizerState.from_state_dict(
            checkpoint.get("frontres_critic_value_normalizer_state_dict")
        )
    except (TypeError, ValueError, FloatingPointError) as exc:
        raise RuntimeError("checkpoint-v14 Critic value-normalizer state is invalid") from exc
    checkpoint_iteration = checkpoint.get("iter")
    if (
        not isinstance(checkpoint_iteration, int)
        or isinstance(checkpoint_iteration, bool)
        or value_normalizer_state.update_count != checkpoint_iteration
    ):
        raise RuntimeError("checkpoint-v14 Critic value-normalizer count differs from committed iteration")
    rng_state = checkpoint.get("frontres_v013_rng_state")
    if not isinstance(rng_state, Mapping) or set(rng_state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise RuntimeError("FRS-TRAIN-v019 checkpoint is missing complete RNG state")
    numpy_rng = rng_state.get("numpy")
    if not isinstance(numpy_rng, Mapping) or set(numpy_rng) != {
        "bit_generator", "keys", "position", "has_gauss", "cached_gaussian"
    }:
        raise RuntimeError("FRS-TRAIN-v019 checkpoint NumPy RNG state is malformed")
    sampler = getattr(runner, "_frontres_segment_sampler", None)
    if sampler is not None and not isinstance(checkpoint.get("frontres_segment_sampler_state_dict"), Mapping):
        raise RuntimeError("v015 checkpoint is missing sampler state")

    def require_exact_tensor_state(saved: Mapping[str, Any], runtime: Mapping[str, Any], *, label: str) -> None:
        if set(saved) != set(runtime):
            raise RuntimeError(f"v015 checkpoint {label} keys differ from runtime")
        for name, runtime_value in runtime.items():
            saved_value = saved[name]
            if (
                not isinstance(saved_value, torch.Tensor)
                or not isinstance(runtime_value, torch.Tensor)
                or saved_value.shape != runtime_value.shape
                or saved_value.dtype != runtime_value.dtype
            ):
                raise RuntimeError(f"v015 checkpoint {label}.{name} shape/dtype differs from runtime")

    policy = getattr(getattr(runner, "alg", None), "policy", None)
    require_exact_tensor_state(
        model_state["residual_actor"], policy.residual_actor.state_dict(), label="actor"
    )
    require_exact_tensor_state(model_state["critic"], policy.critic.state_dict(), label="Critic")
    optimizer_state = checkpoint["optimizer_state_dict"]
    saved_groups = optimizer_state.get("param_groups")
    saved_slots = optimizer_state.get("state")
    if not isinstance(saved_groups, list) or not isinstance(saved_slots, Mapping):
        raise RuntimeError("v015 checkpoint optimizer state differs from runtime")
    saved_by_role = {
        str(group.get("frontres_role", "")): group for group in saved_groups if isinstance(group, Mapping)
    }
    if len(saved_groups) != 2 or set(saved_by_role) != {"actor", "critic"}:
        raise RuntimeError("v015 checkpoint requires exact actor/critic optimizer groups")
    runtime_by_role: dict[str, Mapping[str, Any]] = {}
    if validation_scope == "resume":
        runtime_groups = getattr(runner.alg.optimizer, "param_groups", None)
        if not isinstance(runtime_groups, list):
            raise RuntimeError("v015 checkpoint optimizer state differs from runtime")
        runtime_by_role = {
            str(group.get("frontres_role", "")): group for group in runtime_groups if isinstance(group, Mapping)
        }
        if len(runtime_groups) != 2 or set(runtime_by_role) != {"actor", "critic"}:
            raise RuntimeError("v015 checkpoint requires exact actor/critic optimizer groups")
    saved_step_counts = set()
    saved_param_ids: set[int] = set()
    runtime_param_ids: set[int] = set()

    def require_finite_lr(value: Any, *, role: str) -> float:
        try:
            learning_rate = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(f"v015 checkpoint {role} optimizer LR identity differs from runtime") from exc
        if not math.isfinite(learning_rate):
            raise RuntimeError(f"v015 checkpoint {role} optimizer LR identity differs from runtime")
        return learning_rate

    for role in ("actor", "critic"):
        saved_group = saved_by_role[role]
        saved_ids = saved_group.get("params") if isinstance(saved_group, Mapping) else None
        policy_params = list((policy.residual_actor if role == "actor" else policy.critic).parameters())
        if not isinstance(saved_ids, list) or len(saved_ids) != len(policy_params):
            raise RuntimeError("v015 checkpoint optimizer parameter groups differ from runtime")
        saved_lr = require_finite_lr(saved_group.get("lr"), role=role)
        expected_lr = 3.0e-6 if role == "actor" else 1.0e-5
        if saved_lr != expected_lr:
            raise RuntimeError(f"v015 checkpoint {role} optimizer LR identity differs from runtime")
        runtime_params = policy_params
        if validation_scope == "resume":
            runtime_group = runtime_by_role[role]
            runtime_params = runtime_group.get("params") if isinstance(runtime_group, Mapping) else None
            if not isinstance(runtime_params, list) or len(saved_ids) != len(runtime_params):
                raise RuntimeError("v015 checkpoint optimizer parameter groups differ from runtime")
            runtime_lr = require_finite_lr(runtime_group.get("lr"), role=role)
            if runtime_lr != expected_lr:
                raise RuntimeError(f"v015 checkpoint {role} optimizer LR identity differs from runtime")
        step_count = saved_group.get("frontres_step_count")
        if not isinstance(step_count, int) or isinstance(step_count, bool) or step_count < 0:
            raise RuntimeError("v015 checkpoint optimizer step count is malformed")
        saved_step_counts.add(step_count)
        if saved_param_ids.intersection(saved_ids):
            raise RuntimeError("v015 checkpoint optimizer parameter membership overlaps across roles")
        saved_param_ids.update(saved_ids)
        if validation_scope == "resume":
            current_runtime_ids = {id(parameter) for parameter in runtime_params}
            if runtime_param_ids.intersection(current_runtime_ids):
                raise RuntimeError("v015 runtime optimizer parameter membership overlaps across roles")
            runtime_param_ids.update(current_runtime_ids)
        for saved_id, runtime_parameter in zip(saved_ids, runtime_params, strict=True):
            slot = saved_slots.get(saved_id, {})
            if not isinstance(slot, Mapping):
                raise RuntimeError("v015 checkpoint optimizer slot state is malformed")
            for slot_value in slot.values():
                if (
                    isinstance(slot_value, torch.Tensor)
                    and slot_value.ndim > 0
                    and slot_value.numel() > 1
                    and slot_value.shape != runtime_parameter.shape
                ):
                    raise RuntimeError("v015 checkpoint optimizer slot shape differs from runtime")
    if len(saved_step_counts) != 1:
        raise RuntimeError("v015 checkpoint optimizer groups disagree on the persisted step count")
    gain_identity = identity.get("gain")
    if not isinstance(gain_identity, Mapping) or float(gain_identity.get("beta", float("nan"))) != float(
        getattr(runner.alg, "frontres_gain_beta", 0.02)
    ):
        raise RuntimeError("v015 checkpoint has an incompatible FRS-GAIN-v008 beta identity")
    curriculum = identity.get("curriculum")
    if not isinstance(curriculum, Mapping):
        raise RuntimeError("FRS-TRAIN-v019 checkpoint curriculum identity is missing")
    runtime_schedule = tuple(getattr(runner.alg, "frontres_segment_k_curriculum", ()) or ())
    require_frontres_v013_campaign_schedule(runtime_schedule)
    runtime_schedule_tuple = frontres_k_stage_schedule_tuple(runtime_schedule)
    if curriculum.get("schedule") != runtime_schedule_tuple:
        raise RuntimeError("FRS-TRAIN-v019 checkpoint schedule differs from the runtime schedule")
    saved_iteration = int(curriculum.get("absolute_iteration", -1))
    if (
        saved_iteration < 0
        or saved_iteration > FRONTRES_V011_MAX_ABSOLUTE_ITERATION
        or int(checkpoint.get("iter", -1)) != saved_iteration
    ):
        raise RuntimeError("FRS-TRAIN-v019 checkpoint iteration identity is inconsistent")
    expected_curriculum = resolve_frontres_k_stage_identity(
        schedule=runtime_schedule,
        committed_update_iteration=saved_iteration,
        max_horizon_k=int(getattr(runner.alg, "frontres_segment_max_horizon_k", 0)),
    )
    expected_curriculum_payload = {
        "schedule": runtime_schedule_tuple,
        "schedule_fingerprint": expected_curriculum.schedule_fingerprint,
        "k_stage_index": expected_curriculum.stage_index,
        "active_k": expected_curriculum.active_k,
        "active_m": expected_curriculum.active_m,
        "selected_segment_count": FRONTRES_V011_SELECTED_SEGMENT_COUNT,
        "policy_row_count": FRONTRES_V011_SELECTED_SEGMENT_COUNT * expected_curriculum.active_m,
        "role_row_count": 2 * FRONTRES_V011_SELECTED_SEGMENT_COUNT * expected_curriculum.active_m,
        "maximum_absolute_iteration": FRONTRES_V011_MAX_ABSOLUTE_ITERATION,
        "checkpoint_review_boundaries": FRONTRES_V011_REVIEW_BOUNDARIES,
        "stage_iteration": expected_curriculum.stage_iteration,
        "absolute_iteration": expected_curriculum.absolute_iteration,
        "phase": expected_curriculum.phase.name,
        "phase_iteration": expected_curriculum.phase.phase_iteration,
        "actor_loss_weight": expected_curriculum.phase.actor_loss_weight,
        "dr_stage_fingerprint": expected_curriculum.dr_stage_fingerprint,
        "dr_progress": expected_curriculum.dr_progress,
        "d_cap": expected_curriculum.d_cap,
    }
    if dict(curriculum) != expected_curriculum_payload:
        raise RuntimeError("FRS-TRAIN-v019 checkpoint curriculum stage/phase/DR identity is inconsistent")
    fields = _v015_checkpoint_layout_fields(runner)
    if identity.get("future_intent_layout") != fields:
        raise RuntimeError(
            "v015 checkpoint future-intent layout mismatch; refusing to reinterpret actor prefix or H offsets"
        )
    grouped = identity.get("grouped_loss")
    if grouped != {
        "advantage_normalization": "grouped_scale_only",
        "candidate_layout_version": _V015_GROUPED_CANDIDATE_LAYOUT,
        "policy_rows_per_attempt": 1,
    }:
        raise RuntimeError("v015 checkpoint has an incompatible grouped-loss identity")
    normalizer = identity.get("normalizer")
    if not isinstance(normalizer, Mapping):
        raise RuntimeError("v015 checkpoint has no normalizer identity")
    normalization_enabled = bool(getattr(runner, "empirical_normalization", False))
    if normalizer.get("prefix_layout_version") != fields["layout_version"] or int(
        normalizer.get("prefix_dim", -1)
    ) != int(fields["prefix_dim"]):
        raise RuntimeError("v015 checkpoint prefix normalizer layout mismatch")
    if normalization_enabled:
        expected_combined_dim = int(fields["prefix_dim"]) + int(fields["gmt_dim"])
        if (
            normalizer.get("mode") != "empirical_prefix_plus_frozen_gmt"
            or int(normalizer.get("combined_dim", -1)) != expected_combined_dim
            or not isinstance(normalizer.get("prefix_stats_fingerprint"), str)
        ):
            raise RuntimeError("v015 checkpoint normalizer identity is incompatible or unversioned")
        state = checkpoint.get("obs_norm_state_dict")
        if not isinstance(state, Mapping):
            raise RuntimeError("v015 checkpoint is missing observation normalizer state")
        mean = state.get("_mean")
        std = state.get("_std")
        if (
            not isinstance(mean, torch.Tensor)
            or not isinstance(std, torch.Tensor)
            or int(mean.shape[-1]) != expected_combined_dim
            or int(std.shape[-1]) != expected_combined_dim
        ):
            raise RuntimeError("v015 checkpoint rejects legacy or incompatible normalizer statistics")
        prefix_dim = int(fields["prefix_dim"])
        observed_fingerprint = _v015_tensor_fingerprint(mean[..., :prefix_dim], std[..., :prefix_dim])
        if observed_fingerprint != normalizer.get("prefix_stats_fingerprint"):
            raise RuntimeError("v015 checkpoint prefix normalizer statistics do not match their identity")
        critic_normalizer_state = checkpoint.get("privileged_obs_norm_state_dict")
        _validate_v015_normalizer_state(
            critic_normalizer_state,
            dim=449,
            label="checkpoint-v14 Critic normalizer",
        )
    elif normalizer.get("mode") != "disabled" or normalizer.get("prefix_stats_fingerprint") is not None:
        raise RuntimeError("v015 checkpoint normalizer mode changed across resume")
    transaction = identity.get("transaction")
    if not isinstance(transaction, Mapping):
        raise RuntimeError("v015 checkpoint has no transaction atomicity identity")
    state = str(transaction.get("state", ""))
    if state == "idle" and set(transaction) == {"state"}:
        result = dict(identity)
        result["transaction"] = {"state": "idle"}
        return result
    if state == "committed" and set(transaction) == {"state", "receipt"}:
        result = dict(identity)
        result["transaction"] = {
            "state": "committed",
            "receipt": _v015_committed_transaction_receipt(transaction),
        }
        _validate_v013_receipt_curriculum(
            result["transaction"],
            schedule=runtime_schedule_tuple,
            current_iteration=saved_iteration,
        )
        return result
    raise RuntimeError("v015 checkpoint resume rejects partial, failed, or malformed transactions")


# Full-resume diagnostic helper; uncomment with the probe prints below when needed.
# def _optimizer_state_debug(state_dict: dict | None) -> str:
#     if not isinstance(state_dict, dict):
#         return "missing"
#     groups = state_dict.get("param_groups", []) or []
#     state = state_dict.get("state", {}) or {}
#     lrs = []
#     param_counts = []
#     for group in groups:
#         if isinstance(group, dict):
#             lrs.append(group.get("lr"))
#             param_counts.append(len(group.get("params", []) or []))
#     return (
#         f"groups={len(groups)} state_entries={len(state)} "
#         f"group_param_counts={param_counts} group_lrs={lrs}"
#     )


def _copy_policy_noise_state(policy, model_state: dict) -> bool:
    """Load std/log_std only when checkpoint and runtime action dims match."""
    if hasattr(policy, "std") and "std" in model_state:
        source = model_state["std"].detach().to(device=policy.std.device, dtype=policy.std.dtype)
        if tuple(source.shape) == tuple(policy.std.shape):
            policy.std.data.copy_(source)
            return True
        print(
            "[Runner] Skipping checkpoint noise std due to action-dim drift: "
            f"checkpoint_shape={tuple(source.shape)} runtime_shape={tuple(policy.std.shape)}",
            flush=True,
        )
        return False
    if hasattr(policy, "log_std") and "log_std" in model_state:
        source = model_state["log_std"].detach().to(device=policy.log_std.device, dtype=policy.log_std.dtype)
        if tuple(source.shape) == tuple(policy.log_std.shape):
            policy.log_std.data.copy_(source)
            return True
        print(
            "[Runner] Skipping checkpoint log_std due to action-dim drift: "
            f"checkpoint_shape={tuple(source.shape)} runtime_shape={tuple(policy.log_std.shape)}",
            flush=True,
        )
        return False
    return False


def _reset_policy_noise_state(policy, *, init_noise_std: float, noise_std_type: str, device) -> None:
    """Reset runtime std/log_std using the current policy tensor shape."""
    if noise_std_type == "scalar" and hasattr(policy, "std"):
        policy.std.data.copy_(torch.ones_like(policy.std, device=device) * init_noise_std)
        print(f"[Runner] Reset noise std → {init_noise_std} shape={tuple(policy.std.shape)}")
    elif noise_std_type == "log" and hasattr(policy, "log_std"):
        policy.log_std.data.copy_(torch.log(torch.ones_like(policy.log_std, device=device) * init_noise_std))
        print(f"[Runner] Reset log_std → log({init_noise_std}) shape={tuple(policy.log_std.shape)}")


def record_frontres_checkpoint_probe(self, locs: dict, checkpoint_path: str) -> None:
    """Persist save-time FrontRES probe metrics and keep the best demo checkpoint.

    This is a lightweight checkpoint selector: it records the triplet
    rollout diagnostics already computed for the checkpoint iteration,
    without resetting the simulator or replaying the full training set.
    """
    if self.training_type != "frontres" or self.log_dir is None:
        return

    def _float(name: str, default: float | None = None) -> float | None:
        value = locs.get(name, default)
        if value is None:
            return default
        try:
            if isinstance(value, torch.Tensor):
                value = value.detach().mean().item()
            return float(value)
        except (TypeError, ValueError):
            return default

    restore_ratio = _float("frontres_restore_ratio_rp_mean")
    if restore_ratio is None:
        return

    residual = _float("frontres_residual_rp_abs_mean", 0.0) or 0.0
    roll_bias = _float("frontres_corr_roll_bias_mean", 0.0) or 0.0
    pitch_bias = _float("frontres_corr_pitch_bias_mean", 0.0) or 0.0
    harm_rate = _float("frontres_harm_rate_mean", 0.0) or 0.0
    harm_mag = _float("frontres_harm_mag_mean", 0.0) or 0.0
    survival = _float("frontres_survival_rate", 1.0)
    r_delta = _float("frontres_rdelta_mean", 0.0) or 0.0
    dr_scale = _float("frontres_dr_scale", None)

    bias_abs = abs(roll_bias) + abs(pitch_bias)
    survival_penalty = 0.0 if survival is None else max(0.0, 1.0 - survival)
    score = (
        restore_ratio
        - 0.25 * harm_rate
        - 2.0 * harm_mag
        - 0.50 * bias_abs
        - 0.10 * residual
        - 2.0 * survival_penalty
    )

    record = {
        "iteration": int(locs.get("it", self.current_learning_iteration)),
        "checkpoint": os.path.basename(checkpoint_path),
        "score": score,
        "restore_ratio_rp": restore_ratio,
        "residual_rp_abs": residual,
        "corr_roll_bias": roll_bias,
        "corr_pitch_bias": pitch_bias,
        "bias_abs": bias_abs,
        "harm_rate": harm_rate,
        "harm_mag": harm_mag,
        "survival_rate": survival,
        "r_delta": r_delta,
        "dr_scale": dr_scale,
        "perturb_modes": locs.get("frontres_perturb_modes"),
        "perturb_complexity": locs.get("frontres_perturb_complexity"),
    }

    probe_path = os.path.join(self.log_dir, "frontres_checkpoint_probe.jsonl")
    with open(probe_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")

    if self.writer is not None and not self.disable_logs:
        self.writer.add_scalar("FrontRES/CheckpointProbe/demo_score", score, record["iteration"])
        self.writer.add_scalar("FrontRES/CheckpointProbe/restore_ratio_rp", restore_ratio, record["iteration"])
        self.writer.add_scalar("FrontRES/CheckpointProbe/bias_abs", bias_abs, record["iteration"])

    best_score = getattr(self, "_frontres_best_probe_score", None)
    best_meta_path = os.path.join(self.log_dir, "frontres_best_probe.json")
    if best_score is None and os.path.exists(best_meta_path):
        try:
            with open(best_meta_path, "r", encoding="utf-8") as f:
                best_score = float(json.load(f).get("score"))
                self._frontres_best_probe_score = best_score
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            best_score = None
    if best_score is None or score > float(best_score):
        self._frontres_best_probe_score = score
        best_path = os.path.join(self.log_dir, "model_best_probe.pt")
        shutil.copyfile(checkpoint_path, best_path)
        with open(best_meta_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, sort_keys=True)
        print(
            "[Runner] New FrontRES probe best: "
            f"score={score:+.4f}, restore_rp={restore_ratio:+.3f}, "
            f"harm={harm_rate:.3f}, bias={bias_abs:.4f} -> {os.path.basename(best_path)}",
            flush=True,
        )

def save_runner(self, path: str, infos=None):
    """保存可恢复 Stage 2/3 语义的完整 runner checkpoint.

    函数名说明:
        `save_runner` 是 FrontRES persistence write owner, 汇总 policy, optimizer,
        normalizer, sampler 和 curriculum state; 它不是模型导出或 eval snapshot.

    主链路:
        上游: periodic/final checkpoint trigger 提供目标 path 和当前 runner state.
        下游: `torch.save` 写盘, `load_runner` 按相同 semantic keys 恢复.

    语义:
        Stage 2 -> Stage 3 必须保存同一个 full-6D actor 和 FrontRES prefix stats.
        Resume-only optimizer/sampler state 也必须与 iteration identity 同源.

    Status:
        v015 branch 是 CPU fake-S3 persistence owner. 上游是显式 fake formal
        transaction barrier, 下游是 `load_runner` 的 pre-mutation validation.
        Evidence 是 code-confirmed 和 contract-confirmed; generic checkpoint
        cadence, simulator, training, 和 live resume 尚未验证.
    """
    # HSL is a migration artifact, not a generic runner checkpoint. Return
    # before critic/optimizer/sampler/Gain/transaction fields are constructed.
    if _uses_v015_hsl_checkpoint_identity(self):
        if infos is not None:
            raise RuntimeError("proposal-only HSL checkpoint forbids generic runner infos payload")
        hsl_payload = _build_v015_hsl_checkpoint_payload(self)
        torch.save(hsl_payload, path)
        hsl_identity = hsl_payload[_V015_HSL_CHECKPOINT_IDENTITY_KEY]
        hsl_layout = hsl_identity["future_intent_layout"]
        print(
            "[G2-S4-HSL-IDENTITY] "
            f"format={hsl_identity['format']} raw={hsl_layout['environment_obs_dim']} "
            f"actor={hsl_layout['actor_dim']} femr={hsl_layout['prefix_dim']} "
            f"gmt={hsl_layout['gmt_dim']} offsets={hsl_layout['future_offsets']} "
            f"gmt_sha256={hsl_identity['gmt']['checkpoint_sha256']} "
            f"top_level_keys={tuple(sorted(hsl_payload))} forbidden_payload=0",
            flush=True,
        )
        print(f"[Runner] Proposal-only HSL checkpoint saved to {path}", flush=True)
        return

    # B1: 汇总 policy, optimizer, iteration 和 active Stage 3 owner state.
    # Check if using ResidualActorCritic (special handling)
    if isinstance(self.alg.policy, (ResidualActorCritic, FrontRESActorCritic)):
        # Save only residual network + critic (GMT is frozen, no need to save)
        model_state_dict = {
            'residual_actor': self.alg.policy.residual_actor.state_dict(),
            'critic': self.alg.policy.critic.state_dict(),}
        # Save noise std parameter
        if hasattr(self.alg.policy, 'std'):
            model_state_dict['std'] = self.alg.policy.std
        elif hasattr(self.alg.policy, 'log_std'):
            model_state_dict['log_std'] = self.alg.policy.log_std
    else:
        # Standard save: entire policy
        model_state_dict = self.alg.policy.state_dict()

    # -- Save model
    saved_dict = {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": self.alg.optimizer.state_dict(),
        "iter": self.current_learning_iteration,
        "infos": infos,}
    if getattr(self.alg, "frontres_training_objective", "") == "segment_replay_hrl":
        saved_dict["frontres_segment_k_curriculum"] = frontres_k_stage_schedule_tuple(
            tuple(getattr(self.alg, "frontres_segment_k_curriculum", ()) or ())
        )
        saved_dict["frontres_v013_rng_state"] = _frontres_v013_rng_state()

    # Legacy adaptive DR is never part of the active TRAIN-v015 payload.
    _is_v013_formal = bool(getattr(self.alg, "frontres_formal_transaction_enabled", False))
    if not _is_v013_formal and hasattr(self, '_dr_scale'):
        saved_dict["dr_scale"] = self._dr_scale
    if hasattr(self, '_dr_prev_error'):
        saved_dict["dr_prev_error"] = self._dr_prev_error
    if getattr(self, '_frontres_boundary_ema', None) is not None:
        saved_dict["frontres_boundary_ema"] = dict(self._frontres_boundary_ema)
    if getattr(self, '_last_frontres_boundary_stats', None) is not None:
        saved_dict["last_frontres_boundary_stats"] = dict(self._last_frontres_boundary_stats)
    if hasattr(self, "_frontres_gmt_frontier_safe_low"):
        saved_dict["frontres_gmt_frontier_safe_low"] = self._frontres_gmt_frontier_safe_low
    if hasattr(self, "_frontres_gmt_frontier_broken_high"):
        saved_dict["frontres_gmt_frontier_broken_high"] = self._frontres_gmt_frontier_broken_high
    if hasattr(self, "_frontres_gmt_frontier_probe_scale"):
        saved_dict["frontres_gmt_frontier_probe_scale"] = self._frontres_gmt_frontier_probe_scale
    if hasattr(self, "_frontres_gmt_frontier_probe_score"):
        saved_dict["frontres_gmt_frontier_probe_score"] = self._frontres_gmt_frontier_probe_score
    if hasattr(self, "_frontres_gmt_frontier_decision"):
        saved_dict["frontres_gmt_frontier_decision"] = self._frontres_gmt_frontier_decision
    if hasattr(self, "_frontres_gmt_frontier_confirmed"):
        saved_dict["frontres_gmt_frontier_confirmed"] = self._frontres_gmt_frontier_confirmed
    for _name in (
        "safe_score_ema",
        "broken_score_ema",
        "safe_count",
        "broken_count",
    ):
        _attr = f"_frontres_exec_floor_{_name}"
        if hasattr(self, _attr):
            saved_dict[f"frontres_exec_floor_{_name}"] = getattr(self, _attr)
    if hasattr(self, "_frontres_exec_floor_source_last"):
        saved_dict["frontres_exec_floor_source_last"] = self._frontres_exec_floor_source_last
    if hasattr(self, '_frontres_warmup_complete'):
        saved_dict["frontres_warmup_complete"] = bool(self._frontres_warmup_complete)
    if str(getattr(self, "training_type", "")) == "frontres" and not _uses_v015_formal_checkpoint_identity(self):
        saved_dict["frontres_gain_config"] = frontres_legacy_gain_config_payload(getattr(self, "cfg", None))
    segment_sampler = getattr(self, "_frontres_segment_sampler", None)
    if segment_sampler is not None and hasattr(segment_sampler, "state_dict"):
        saved_dict["frontres_segment_sampler_state_dict"] = segment_sampler.state_dict()
    
    # -- Save RND model if used
    if hasattr(self.alg, "rnd") and self.alg.rnd:
        saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
        saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
    
    # B2: 先保存 normalizer, 再从其精确 state 构造 v015 identity.
    obs_norm_state = None
    extra_mean = None
    extra_std = None

    # -- Save observation normalizer if used
    if self.empirical_normalization:
        extra_mean, extra_std = frontres_extra_norm_stats_for_save(
            getattr(self, "_frontres_extra_mean", None),
            getattr(self, "_frontres_extra_std", None),
            getattr(self, "_frontres_extra_normalizer", None),
        )
        obs_norm_state = self.obs_normalizer.state_dict()
        obs_norm_state = compose_frontres_obs_norm_state(
            obs_norm_state,
            extra_mean,
            extra_std,
        )
        saved_dict["obs_norm_state_dict"] = obs_norm_state
        saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()
        # Save teacher normalizer for MOSAIC
        if self.training_type == "mosaic" and hasattr(self, 'teacher_obs_normalizer'):
            if not isinstance(self.teacher_obs_normalizer, torch.nn.Identity):
                saved_dict["teacher_obs_norm_state_dict"] = self.teacher_obs_normalizer.state_dict()

    # B3: v015 不做 compatibility conversion, 仅保存精确 layout/normalizer identity 和 completed-or-idle receipt.
    if _uses_v015_formal_checkpoint_identity(self):
        for key in tuple(saved_dict):
            if key in {"dr_scale", "dr_prev_error", "frontres_boundary_ema", "last_frontres_boundary_stats"} or key.startswith(
                ("frontres_gmt_frontier_", "frontres_exec_floor_")
            ):
                saved_dict.pop(key)
        value_normalizer_state = getattr(self.alg, "frontres_critic_value_normalizer_state", None)
        if not isinstance(value_normalizer_state, FrontRESValueNormalizerState):
            raise RuntimeError("checkpoint-v14 save requires one committed Critic value-normalizer state")
        if value_normalizer_state.update_count != int(self.current_learning_iteration):
            raise RuntimeError("checkpoint-v14 save requires value-normalizer count to equal committed iteration")
        saved_dict["frontres_critic_value_normalizer_state_dict"] = value_normalizer_state.state_dict()
        saved_dict[_V015_CHECKPOINT_IDENTITY_KEY] = _build_v015_checkpoint_identity(
            self,
            obs_norm_state=obs_norm_state,
            extra_mean=extra_mean,
            extra_std=extra_std,
        )

    # Full-resume diagnostic probe; uncomment when checking checkpoint payloads.
    # print(
    #     "[FrontRES Checkpoint Save Probe] "
    #     f"path={path} iter={self.current_learning_iteration} "
    #     f"optimizer={_optimizer_state_debug(saved_dict.get('optimizer_state_dict'))} "
    #     f"sampler_state={'frontres_segment_sampler_state_dict' in saved_dict} "
    #     f"dr_scale={saved_dict.get('dr_scale', 'n/a')}",
    #     flush=True,
    # )

    # B2: Validate the complete payload after all semantic owners have contributed state.
    # B3: AUDIT-PERSIST-01 records the exact payload passed to torch.save.
    # Result: E69 LIVE PASS. model_221 保存 model/optimizer/normalizer/sampler/
    # Gain config/warmup payload, 与恢复后的 absolute iter 221 一致.
    print_checkpoint_payload_audit(self, path=path, payload=saved_dict)
    # Formal checkpoint-v14 artifacts publish atomically: failed serialization cannot
    # replace the last committed artifact.
    if _uses_v015_formal_checkpoint_identity(self):
        temp_path = f"{path}.tmp-{os.getpid()}"
        try:
            torch.save(saved_dict, temp_path)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    else:
        torch.save(saved_dict, path)

    if _uses_v015_formal_checkpoint_identity(self) and formal_runtime_audit_enabled(self):
        policy = self.alg.policy
        runner_snapshot = (
            int(self.current_learning_iteration),
            hasattr(self, "_frontres_last_loaded_checkpoint_path"),
            getattr(self, "_frontres_last_loaded_checkpoint_path", None),
            _v015_state_dict_fingerprint(policy.residual_actor.state_dict(), label="AUDIT-B08 Actor"),
            _v015_state_dict_fingerprint(policy.critic.state_dict(), label="AUDIT-B08 Critic"),
        )
        reloaded_payload = load_frontres_checkpoint_mapping(path, map_location="cpu")
        validated_identity = _validate_v015_checkpoint_resume(
            self,
            reloaded_payload,
            validation_scope="resume",
        )
        if not isinstance(validated_identity, Mapping):
            raise RuntimeError("AUDIT-B08 strict checkpoint-v14 readback produced no active identity")
        after_snapshot = (
            int(self.current_learning_iteration),
            hasattr(self, "_frontres_last_loaded_checkpoint_path"),
            getattr(self, "_frontres_last_loaded_checkpoint_path", None),
            _v015_state_dict_fingerprint(policy.residual_actor.state_dict(), label="AUDIT-B08 Actor"),
            _v015_state_dict_fingerprint(policy.critic.state_dict(), label="AUDIT-B08 Critic"),
        )
        if after_snapshot != runner_snapshot:
            raise RuntimeError("AUDIT-B08 checkpoint readback mutated the live runner")
        print_checkpoint_reload_audit(
            self,
            path=path,
            payload=reloaded_payload,
            validated_identity=validated_identity,
            file_sha256=_v015_file_sha256(path),
        )

    # upload model to external logging service
    logger_type = str(getattr(self, "logger_type", getattr(self, "cfg", {}).get("logger", "")) or "").lower()
    writer = getattr(self, "writer", None)
    if logger_type in ["neptune", "wandb"] and writer is not None and not bool(getattr(self, "disable_logs", False)):
        writer.save_model(path, self.current_learning_iteration)

def load_runner(self, path: str, load_optimizer: bool = True, load_critic: bool = True):
    """按 cold-start/resume 语义恢复 FrontRES runner checkpoint.

    函数名说明:
        `load_runner` 是 FrontRES persistence read owner, 区分 HSL 初始化和完整
        Stage 3 resume; 它不是宽松的 shape-compatible state loader.

    主链路:
        上游: train/eval entrypoint 提供 checkpoint path 和 load flags.
        下游: 恢复 full-6D actor, normalizer, optimizer, sampler, warmup 和 iteration
        state, 供 rollout/PPO 立即消费.

    语义:
        checkpoint identity 决定哪些状态允许恢复. Actor head 和 prefix stats
        不能漏载或错载, resume 状态也不能污染 Stage 2 -> Stage 3 cold start.

    Status:
        v015 branch 只恢复 exact layout/normalizer identity 和 committed receipt
        history. 它不重建 partial request 或 candidate batch. Evidence 是
        code-confirmed 和 contract-confirmed; generic/live resume 尚未验证.
    """
    # B1: mutable restore 前先验证 v015 envelope. 只有没有 active v015 identity
    # 的 payload 才能进入 legacy HSL reject boundary, 避免把新 Stage-3 history
    # 误判为旧 HSL checkpoint.
    configure_formal_runtime_probe(
        bool(getattr(getattr(self, "alg", None), "frontres_formal_runtime_audit", False))
    )
    loaded_dict = load_frontres_checkpoint_mapping(path, map_location="cpu")
    hsl_resume = _validate_v015_hsl_checkpoint_resume(self, loaded_dict)
    if hsl_resume is not None:
        return _restore_v015_hsl_checkpoint(self, hsl_resume, path=path)
    v015_resume_identity = _validate_v015_checkpoint_resume(self, loaded_dict)
    if v015_resume_identity is None:
        reject_legacy_frontres_hsl_checkpoint(self, loaded_dict)
    else:
        self.alg.frontres_critic_value_normalizer_state = FrontRESValueNormalizerState.from_state_dict(
            loaded_dict["frontres_critic_value_normalizer_state_dict"]
        )
    # B2: 从同一 payload 恢复 sampler, actor, normalizer, optimizer, Gain 和 warmup identity.
    self._frontres_last_loaded_checkpoint_path = os.path.abspath(path)
    segment_sampler = getattr(self, "_frontres_segment_sampler", None)
    if (
        segment_sampler is not None
        and "frontres_segment_sampler_state_dict" in loaded_dict
        and hasattr(segment_sampler, "load_state_dict")
    ):
        segment_sampler.load_state_dict(loaded_dict["frontres_segment_sampler_state_dict"])
        print("[Runner] Loaded FrontRES Segment sampler state from checkpoint.")
    self._frontres_warmup_complete = bool(loaded_dict.get("frontres_warmup_complete", False))
    if self._frontres_warmup_complete:
        print("[Runner] Checkpoint marks FrontRES supervised warmup as complete.")

    # ── 断点续训模式控制 ────────────────────────────────────────────────────────
    # is_full_resume=True  (Stage2→Stage2 断点续训): 恢复优化器矩估计+学习率, 保留 std
    # is_full_resume=False (Stage1→Stage2 权重迁移): 仅权重, 重置优化器和 std.
    # Joint-warmup checkpoints are a special case: their critic has already
    # learned E(s)=R_feasible_oracle-R_noisy and should be transferred into RL.
    # load_optimizer 参数仍可从外部显式覆盖（例如强制跳过优化器加载）。
    is_full_resume: bool = self.cfg.get('is_full_resume', True)
    if not is_full_resume:
        load_optimizer = False   # 权重迁移模式：强制跳过优化器，从零初始化 Adam
        load_critic = self._frontres_warmup_complete
    if v015_resume_identity is None:
        validate_frontres_legacy_gain_config_resume(self, loaded_dict, is_full_resume=is_full_resume)
    else:
        print(
            "[Runner] Verified FRS-GAIN-v008 and FRS-TRAIN-v019 through the checkpoint-v14 identity; "
            "legacy scalar Gain metadata is excluded from the active v019 owner.",
            flush=True,
        )
    # Full-resume diagnostic probe; uncomment when checking checkpoint reloads.
    # print(
    #     "[FrontRES Resume Probe] "
    #     f"path={os.path.abspath(path)} checkpoint_iter={loaded_dict.get('iter', 'n/a')} "
    #     f"is_full_resume={is_full_resume} "
    #     f"checkpoint_optimizer={_optimizer_state_debug(loaded_dict.get('optimizer_state_dict'))} "
    #     f"sampler_state={'frontres_segment_sampler_state_dict' in loaded_dict} "
    #     f"frontres_warmup_complete={self._frontres_warmup_complete}",
    #     flush=True,
    # )
    print(f"[Runner] is_full_resume={is_full_resume} → "
          f"load_optimizer={load_optimizer}, load_critic={load_critic}, "
          f"reset_noise_std={not is_full_resume}")

    # Check if using ResidualActorCritic (special handling)
    if isinstance(self.alg.policy, (ResidualActorCritic, FrontRESActorCritic)):
        # Stage 2 -> Stage 3 uses the same full-6D residual actor contract.
        if isinstance(self.alg.policy, FrontRESActorCritic) and "student.0.weight" in loaded_dict["model_state_dict"]:
            mapped_dict = {k.replace("student.", ""): v for k, v in loaded_dict["model_state_dict"].items() if k.startswith("student.")}
            self.alg.policy.residual_actor.load_state_dict(mapped_dict, strict=True)
            print("[Runner] Loaded Stage 2 student weights into the full-6D residual actor.")
        else:
            residual_state = loaded_dict["model_state_dict"]["residual_actor"]
            self.alg.policy.residual_actor.load_state_dict(residual_state, strict=True)
        if load_critic:
            if "critic" in loaded_dict["model_state_dict"]:
                self.alg.policy.critic.load_state_dict(loaded_dict["model_state_dict"]["critic"])
            else:
                print("[Runner] No critic weights found. Critic will be initialized from scratch.")
        # Load noise std parameter only when checkpoint and runtime action dims match.
        _copy_policy_noise_state(self.alg.policy, loaded_dict["model_state_dict"])
        if load_critic:
            print("[Runner] Loaded residual network + critic from checkpoint (GMT remains frozen)")
        else:
            print("[Runner] Loaded residual network only (skipping critic from checkpoint)")
        resumed_training = True
    else:
        if load_critic:
            # Standard load: entire policy
            resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        else:
            actor_only_state_dict = {
                key: value
                for key, value in loaded_dict["model_state_dict"].items()
                if not key.startswith("critic.")}
            
            resumed_training = self.alg.policy.load_state_dict(actor_only_state_dict, strict=False)

    # Load RND model if used
    if hasattr(self.alg, "rnd") and self.alg.rnd:
        self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])

    # Load observation normalizers if used
    if self.empirical_normalization:
        if resumed_training:
            # Resuming training: load student obs normalizer
            # For ResidualActorCritic / FrontRESActorCritic, obs_normalizer IS GMT's frozen
            # normalizer — never overwrite it with a checkpoint's normalizer statistics.
            if not isinstance(self.alg.policy, (ResidualActorCritic, FrontRESActorCritic)):
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            elif (isinstance(self.alg.policy, FrontRESActorCritic)
                    and self._frontres_gmt_obs_dim is not None
                    and "obs_norm_state_dict" in loaded_dict):
                # Task-space FrontRES: prefix dims [:num_extra] are not covered by
                # the GMT normalizer.  Restore checkpoint stats for the available
                # prefix dims; newly added prefix dims use identity normalization.
                _s1_sd = loaded_dict["obs_norm_state_dict"]
                gmt_dim = self._frontres_gmt_obs_dim
                obs_dim = int(getattr(self.alg.policy, "num_actor_obs", gmt_dim))
                extra_stats = extract_frontres_extra_norm_stats(_s1_sd, obs_dim, gmt_dim, self.device)
                if extra_stats is not None:
                    self._frontres_extra_mean, self._frontres_extra_std = extra_stats
                    if v015_resume_identity is not None:
                        self._frontres_extra_stats_layout_version = str(
                            v015_resume_identity["future_intent_layout"]["layout_version"]
                        )
                    print(f"[Runner] Loaded FrontRES prefix normalizer stats "
                           f"(dims 0–{self._frontres_extra_mean.shape[-1]}) for FrontRES task-space.")
                else:
                    self._frontres_extra_mean = None
                    self._frontres_extra_std = None
                    self._frontres_extra_stats_layout_version = None
                    print("[Runner] Checkpoint has no compatible FrontRES prefix "
                          "normalizer stats; FrontRES prefix dims pass through unnormalized.")

            if self.training_type == "mosaic":
                # For MOSAIC: determine whether to load privileged_obs_normalizer from checkpoint
                # Only skip loading if teacher_critic was loaded from a separate checkpoint AND is frozen
                load_privileged_normalizer = load_critic
                if hasattr(self.alg, 'teacher_critic_checkpoint_path') and self.alg.teacher_critic_checkpoint_path is not None:
                    if hasattr(self.alg, 'teacher_critic_frozen') and self.alg.teacher_critic_frozen:
                        load_privileged_normalizer = False
                        print("[Runner] Keeping privileged_obs_normalizer from teacher_critic_checkpoint (frozen).")

                if load_privileged_normalizer:
                    # Load critic normalizer from student checkpoint
                    if "privileged_obs_norm_state_dict" in loaded_dict:
                        self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
                        print("[Runner] Loaded privileged_obs_normalizer from checkpoint.")
                    else:
                        print("[Runner] WARNING: No privileged_obs_norm_state_dict in checkpoint!")

                # Load teacher obs normalizer if available (for teacher BC)
                if "teacher_obs_norm_state_dict" in loaded_dict:
                    self.teacher_obs_normalizer.load_state_dict(loaded_dict["teacher_obs_norm_state_dict"])
                    print("[Runner] Loaded teacher_obs_normalizer from checkpoint.")
            else:
                # For PPO and Distillation: load both normalizers
                if load_critic:
                    priv_sd = loaded_dict.get("privileged_obs_norm_state_dict", {})
                    if priv_sd and "_mean" in priv_sd:
                        self.privileged_obs_normalizer.load_state_dict(priv_sd)
                    else:
                        # Stage 1 (SuperviseLearning) checkpoint has no valid
                        # privileged_obs_norm_state_dict — critic normalizer starts fresh.
                        print("[Runner] WARNING: privileged_obs_norm_state_dict missing or invalid — "
                              "privileged_obs_normalizer starts fresh (expected for Stage 1 → Stage 2 transfer).")
        else:
            # Not resuming (e.g., Distillation after RL): load teacher normalizer
            # For Distillation: the checkpoint's obs_norm is the teacher's normalizer
            if load_critic:
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
    # -- load optimizer if used
    if load_optimizer and resumed_training:
        if not load_critic:
            print("[Runner] Skipping optimizer load because load_critic=False.")
        else:
            try:
                # -- algorithm optimizer
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
                print("[Runner] Loaded optimizer state from checkpoint.")
                # Full-resume diagnostic probe; uncomment when checking optimizer state.
                # print(
                #     "[FrontRES Resume Probe] "
                #     f"optimizer_loaded=True runtime_optimizer={_optimizer_state_debug(self.alg.optimizer.state_dict())}",
                #     flush=True,
                # )
                # ── 学习率同步 ─────────────────────────────────────────────────────
                # PPO.update() 每次 epoch 都用 self.alg.learning_rate 覆盖
                # optimizer.param_groups["lr"]。load_state_dict 已将 param_groups["lr"]
                # 恢复为 checkpoint 时的值，但 self.alg.learning_rate 仍是配置初始值。
                # 此处同步，避免第一次 update() 将已恢复的学习率覆盖为初始值。
                if is_full_resume and hasattr(self.alg, 'learning_rate'):
                    reset_lr = bool(self.cfg.get('reset_lr_on_resume', False))
                    named_groups = {
                        str(group.get("frontres_role", "")): group for group in self.alg.optimizer.param_groups
                    }
                    if v015_resume_identity is not None and set(named_groups) != {"actor", "critic"}:
                        raise RuntimeError("v015 full resume restored an invalid split-LR optimizer identity")
                    if v015_resume_identity is not None:
                        if reset_lr:
                            actor_lr = float(self.alg_cfg.get("learning_rate", 3.0e-6))
                            critic_lr = float(self.alg_cfg.get("critic_learning_rate", 1.0e-5))
                            named_groups["actor"]["lr"] = actor_lr
                            named_groups["critic"]["lr"] = critic_lr
                        else:
                            actor_lr = float(named_groups["actor"]["lr"])
                            critic_lr = float(named_groups["critic"]["lr"])
                        self.alg.learning_rate = actor_lr
                        self.alg.actor_learning_rate = actor_lr
                        self.alg.critic_learning_rate = critic_lr
                        print(
                            "[Runner] Synced split learning rates "
                            f"actor={actor_lr:.2e} critic={critic_lr:.2e} reset={reset_lr}"
                        )
                    elif reset_lr:
                        restored_lr = self.alg.optimizer.param_groups[0]['lr']
                        # lr 被 adaptive schedule 压至下限时（如因 desired_kl 配置错误），
                        # 直接重置为算法配置的初始学习率，避免续训起点过低。
                        config_lr = float(self.alg_cfg.get('learning_rate', 5e-4))
                        self.alg.learning_rate = config_lr
                        for pg in self.alg.optimizer.param_groups:
                            pg['lr'] = config_lr
                        print(f"[Runner] Reset learning_rate → {config_lr:.2e} "
                              f"(reset_lr_on_resume=True; checkpoint had {restored_lr:.2e})")
                    else:
                        restored_lr = self.alg.optimizer.param_groups[0]['lr']
                        self.alg.learning_rate = restored_lr
                        print(f"[Runner] Synced learning_rate = {restored_lr:.2e} (from optimizer checkpoint)")
            except (ValueError, KeyError, RuntimeError) as e:
                if v015_resume_identity is not None:
                    raise RuntimeError("v015 full resume rejected incompatible optimizer state") from e
                # Optimizer state mismatch (e.g., different parameter groups between stages)
                # This can happen when:
                # - Stage 1 had frozen critic (optimizer only has actor params)
                # - Stage 2 unfreezes critic (optimizer has actor + critic params)
                print(f"[Runner] WARNING: Could not load optimizer state: {e}")
                print("[Runner] Optimizer will be initialized from scratch (learning rate, momentum, etc. reset)")
                print("[Runner] This is expected when transitioning between training stages with different frozen parameters.")
                # Full-resume diagnostic probe; uncomment when checking optimizer state.
                # print(
                #     "[FrontRES Resume Probe] "
                #     f"optimizer_loaded=False runtime_optimizer={_optimizer_state_debug(self.alg.optimizer.state_dict())}",
                #     flush=True,
                # )

            # -- RND optimizer if used
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
    # -- load current learning iteration
    if resumed_training and is_full_resume and v015_resume_identity is not None:
        saved_schedule = loaded_dict.get("frontres_segment_k_curriculum")
        runtime_schedule = frontres_k_stage_schedule_tuple(
            tuple(getattr(self.alg, "frontres_segment_k_curriculum", ()) or ())
        )
        if saved_schedule != runtime_schedule:
            raise ValueError(
                "FRS-TRAIN-v019 K x M x DR schedule changed across full resume: "
                f"checkpoint={saved_schedule}, runtime={runtime_schedule}."
            )
    if resumed_training:
        if is_full_resume:
            self.current_learning_iteration = loaded_dict["iter"]
        else:
            self.current_learning_iteration = 0
            print("[Runner] Stage1→Stage2 cold-start: current_learning_iteration reset to 0.")
        # Full-resume diagnostic probe; uncomment when checking resume iteration.
        # print(
        #     "[FrontRES Resume Probe] "
        #     f"iteration_after_load={self.current_learning_iteration} checkpoint_iter={loaded_dict.get('iter', 'n/a')} "
        #     f"is_full_resume={is_full_resume}",
        #     flush=True,
        # )

    # ── 噪声 std 控制 ──────────────────────────────────────────────────────────
    # B4: committed receipt 仅是 diagnostic history, resume 后绝不重建旧 request/candidate batch/partial transaction.
    if v015_resume_identity is not None:
        transaction = v015_resume_identity["transaction"]
        if transaction["state"] == "committed":
            setattr(self, _V015_LAST_RECEIPT_ATTR, dict(transaction["receipt"]))
        elif hasattr(self, _V015_LAST_RECEIPT_ATTR):
            delattr(self, _V015_LAST_RECEIPT_ATTR)
        frontres_stage3_transaction_aggregate(self).abort()
        _restore_frontres_v013_rng_state(loaded_dict["frontres_v013_rng_state"])

    # is_full_resume=True:  保留 checkpoint 中已自然适应的 std（断点续训）
    # is_full_resume=False: 重置为 init_noise_std（Stage1→Stage2 冷启动）
    # 向后兼容：若 cfg 中显式设置了 reset_noise_std_on_resume，以其为准。
    reset_noise: bool
    if 'reset_noise_std_on_resume' in self.cfg:
        reset_noise = bool(self.cfg.get('reset_noise_std_on_resume'))
        print(f"[Runner] reset_noise_std_on_resume = {reset_noise} (explicit config override)")
    else:
        reset_noise = not is_full_resume   # is_full_resume=True → 不重置; False → 重置
        print(f"[Runner] reset_noise_std = {reset_noise} (derived from is_full_resume={is_full_resume})")

    if reset_noise and (hasattr(self.alg.policy, 'std') or hasattr(self.alg.policy, 'log_std')):
        init_noise_std = self.policy_cfg.get("init_noise_std", 1.0)
        noise_std_type = self.policy_cfg.get("noise_std_type", "scalar")
        _reset_policy_noise_state(
            self.alg.policy,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            device=self.device,
        )
    else:
        if hasattr(self.alg.policy, 'std'):
            print(f"[Runner] Kept noise std from checkpoint = {self.alg.policy.std.mean().item():.4f}")

    # -- Freeze normalizer if specified in config (for stage transitions)
    # This prevents normalizer statistics from drifting when resuming from distillation
    freeze_normalizer = self.cfg.get("freeze_normalizer_on_resume", False)
    print(f"[Runner] freeze_normalizer_on_resume = {freeze_normalizer}")
    if freeze_normalizer and self.empirical_normalization:
        # Freeze obs normalizer
        self.obs_normalizer.eval()
        if hasattr(self.obs_normalizer, 'until'):
            self.obs_normalizer.until = self.obs_normalizer.count  # Stop updating
        print(f"[Runner] Froze obs_normalizer (count={self.obs_normalizer.count})")

        # Freeze privileged obs normalizer
        self.privileged_obs_normalizer.eval()
        if hasattr(self.privileged_obs_normalizer, 'until'):
            self.privileged_obs_normalizer.until = self.privileged_obs_normalizer.count
        print(f"[Runner] Froze privileged_obs_normalizer (count={self.privileged_obs_normalizer.count})")

    # Restore adaptive DR scale so resume continues from the correct DR level.
    # is_full_resume=True  (Stage2断点续训): 恢复 checkpoint 中的 dr_scale
    # is_full_resume=False (Stage1→Stage2冷启动): 忽略 checkpoint dr_scale，
    #   改用 cfg 中的 dr_scale_init（默认 1.0），确保 Stage2 从 Stage1 训练强度出发，
    #   避免 dr_scale=0 时 Stage1 修正策略作用于干净参考导致的即时崩溃。
    if v015_resume_identity is not None:
        for _attr in tuple(vars(self)):
            if _attr in {"_dr_scale", "_dr_prev_error", "_frontres_boundary_ema", "_last_frontres_boundary_stats"} or _attr.startswith(
                ("_frontres_gmt_frontier_", "_frontres_exec_floor_")
            ):
                delattr(self, _attr)
        print("[Runner] TRAIN-v019 restored explicit per-K DR identity; legacy adaptive DR state excluded")
    elif is_full_resume:
        self._dr_scale      = loaded_dict.get("dr_scale",      0.0)
        self._dr_prev_error = loaded_dict.get("dr_prev_error", 0.0)
        if "frontres_boundary_ema" in loaded_dict:
            self._frontres_boundary_ema = dict(loaded_dict["frontres_boundary_ema"])
        if "last_frontres_boundary_stats" in loaded_dict:
            self._last_frontres_boundary_stats = dict(loaded_dict["last_frontres_boundary_stats"])
        self._frontres_gmt_frontier_safe_low = float(
            loaded_dict.get("frontres_gmt_frontier_safe_low", self._dr_scale)
        )
        self._frontres_gmt_frontier_broken_high = loaded_dict.get(
            "frontres_gmt_frontier_broken_high", None
        )
        if self._frontres_gmt_frontier_broken_high is not None:
            self._frontres_gmt_frontier_broken_high = float(self._frontres_gmt_frontier_broken_high)
        self._frontres_gmt_frontier_probe_scale = float(
            loaded_dict.get("frontres_gmt_frontier_probe_scale", self._dr_scale)
        )
        self._frontres_gmt_frontier_probe_score = loaded_dict.get(
            "frontres_gmt_frontier_probe_score", None
        )
        if self._frontres_gmt_frontier_probe_score is not None:
            self._frontres_gmt_frontier_probe_score = float(self._frontres_gmt_frontier_probe_score)
        self._frontres_gmt_frontier_decision = str(
            loaded_dict.get("frontres_gmt_frontier_decision", "resume")
        )
        self._frontres_gmt_frontier_confirmed = float(
            loaded_dict.get("frontres_gmt_frontier_confirmed", self._frontres_gmt_frontier_safe_low)
        )
        for _name in (
            "safe_score_ema",
            "broken_score_ema",
            "safe_count",
            "broken_count",
        ):
            _key = f"frontres_exec_floor_{_name}"
            _attr = f"_frontres_exec_floor_{_name}"
            if _key in loaded_dict and loaded_dict[_key] is not None:
                setattr(self, _attr, float(loaded_dict[_key]))
            elif hasattr(self, _attr):
                delattr(self, _attr)
        self._frontres_exec_floor_source_last = str(
            loaded_dict.get("frontres_exec_floor_source_last", "resume")
        )
        print(f"[Runner] Adaptive DR scale restored from checkpoint: {self._dr_scale:.4f}")
    else:
        _dr_init = float(self.cfg.get("dr_scale_init", 1.0))
        self._dr_scale = _dr_init
        self._frontres_boundary_ema = None
        self._last_frontres_boundary_stats = None
        self._frontres_gmt_frontier_safe_low = _dr_init
        self._frontres_gmt_frontier_broken_high = None
        self._frontres_gmt_frontier_probe_scale = _dr_init
        self._frontres_gmt_frontier_probe_score = None
        self._frontres_gmt_frontier_decision = "cold_start"
        self._frontres_gmt_frontier_confirmed = _dr_init
        for _name in (
            "safe_score_ema",
            "broken_score_ema",
            "safe_count",
            "broken_count",
        ):
            _attr = f"_frontres_exec_floor_{_name}"
            if hasattr(self, _attr):
                delattr(self, _attr)
        self._frontres_exec_floor_source_last = "cold_start"
        print(f"[Runner] Stage1→Stage2 cold-start: dr_scale initialised to "
              f"dr_scale_init={_dr_init:.4f} (ignoring checkpoint value "
              f"{loaded_dict.get('dr_scale', 0.0):.4f})")

    # B3: AUDIT-HSL-LOAD-01 records the actual loaded actor/normalizer boundary.
    # Result: E69 LIVE PASS. model_220 full-resume 恢复 actor/critic/optimizer,
    # prefix normalizer, sampler, Gain config, warmup phase, std 和 DR scale.
    emit_formal_runtime_probe(
        "AUDIT-HSL-LOAD-01",
        checkpoint_path=self._frontres_last_loaded_checkpoint_path,
        checkpoint_iter=loaded_dict.get("iter", "missing"),
        residual_actor=type(getattr(getattr(self.alg, "policy", None), "residual_actor", None)).__name__,
        obs_normalizer=type(getattr(self, "obs_normalizer", None)).__name__,
        full_resume=bool(is_full_resume),
    )
    return loaded_dict["infos"]
