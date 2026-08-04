"""FrontRES inference and observation-normalization helpers.

Task-space correction and temporal reference cache helpers live under
``rsl_rl.frontres``. This runner module keeps inference wrapping and the
normalizer bridge used by ``OnPolicyRunner``.

Status: active. Upstream: OnPolicyRunner inference/export entrypoints.
Downstream: full-6D task correction and GMT action consumer.
Evidence: code-confirmed and contract-confirmed. Gap: real deployment runtime.
"""

from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from rsl_rl.frontres.frontres_formal_runtime_probe import emit_formal_runtime_probe

from rsl_rl.modules import FrontRESActorCritic
from rsl_rl.modules.frontres_observation_layout import (
    FRONTRES_FUTURE_INTENT_LAYOUT_VERSION,
    FrontRESFutureIntentLayout,
    build_frontres_future_intent_tail,
    split_frontres_policy_obs,
)
from rsl_rl.frontres.runtime_diagnostics import maybe_print_frontres_restore_debug
def _frontres_collection_batch(self):
    resolve = getattr(self, "frontres_stage3_collection_batch", None)
    batch = resolve() if callable(resolve) else None
    return batch if batch is not None else getattr(self, "_frontres_segment_live_current_batch", None)


def _fixed_noisy_context_batch(self):
    batch = _frontres_collection_batch(self)
    tape = getattr(batch, "frontres_fixed_noisy_tape", None) if batch is not None else None
    return batch if isinstance(tape, torch.Tensor) else None


def _future_intent_context_batch(self):
    batch = _frontres_collection_batch(self)
    intent = getattr(batch, "frontres_local_scenario_intent_q29", None) if batch is not None else None
    return batch if isinstance(intent, torch.Tensor) else None


def _fixed_noisy_motion_command(self):
    env = getattr(self, "env", None)
    env = getattr(env, "unwrapped", env)
    manager = getattr(env, "command_manager", None)
    if manager is None:
        return None
    get_term = getattr(manager, "get_term", None)
    if callable(get_term):
        try:
            return get_term("motion")
        except Exception:
            return None
    terms = getattr(manager, "_terms", None)
    return terms.get("motion") if isinstance(terms, dict) else None


def _future_intent_context_snapshot(self):
    command = _fixed_noisy_motion_command(self)
    if bool(getattr(self, "_frontres_hsl_proposal_context_enabled", False)):
        layout = getattr(self, "_frontres_future_intent_layout", None)
        if not isinstance(layout, FrontRESFutureIntentLayout):
            raise RuntimeError("HSL proposal context requires the resolved v015 q29 layout")
        read_proposal = getattr(command, "frontres_hsl_proposal_intent_snapshot", None)
        if not callable(read_proposal):
            raise RuntimeError("HSL proposal context requires the command-owned q29 snapshot")
        snapshot = read_proposal(layout.future_offsets)
        required = {
            "intent_q29",
            "proposal_context_ids",
            "current_root_artifact_ids",
            "motion_indices",
            "frame_indices",
            "future_offsets",
            "provenance",
        }
        if not isinstance(snapshot, dict) or set(snapshot) != required:
            raise RuntimeError("HSL proposal q29 snapshot has an invalid schema")
        if tuple(snapshot["future_offsets"]) != tuple(layout.future_offsets):
            raise RuntimeError("HSL proposal q29 offsets disagree with the runner layout")
        intent_q29 = snapshot["intent_q29"]
        if not isinstance(intent_q29, torch.Tensor) or intent_q29.ndim != 3:
            raise RuntimeError("HSL proposal intent_q29 must be [B,H+1,29]")
        batch_size = int(intent_q29.shape[0])
        aligned = (
            "proposal_context_ids",
            "current_root_artifact_ids",
            "motion_indices",
            "frame_indices",
            "provenance",
        )
        if any(len(snapshot[name]) != batch_size for name in aligned):
            raise RuntimeError("HSL proposal identity/provenance must align one-to-one with actor rows")
        return snapshot

    read_intent = getattr(command, "frontres_local_scenario_intent_snapshot", None)
    if not callable(read_intent):
        return None
    snapshot = read_intent()
    if not isinstance(snapshot, dict):
        raise RuntimeError("v015 future-intent command snapshot must be a dict")
    required = {
        "intent_q29",
        "scenario_ids",
        "noisy_segment_hashes",
        "x_t_identities",
        "roles",
        "provenance",
    }
    if set(snapshot) != required:
        raise RuntimeError(
            "v015 future-intent command snapshot has an invalid schema: "
            f"got={sorted(snapshot)} expected={sorted(required)}"
        )
    return snapshot


def read_frontres_v015_deployment_context(
    self,
    env_ids: torch.Tensor | None = None,
) -> dict[str, object]:
    # B1: 读取 command-owned current/H snapshots, 产出 role-aligned deployment context.
    """Read the Step 5B-S2A deployment current/H carrier without consuming it.

    Status: connector-only. This function validates and clones command-owned
    data; it does not append actor observations, execute GMT, advance a cursor,
    produce metrics, or write training state.
    """

    command = _fixed_noisy_motion_command(self)
    read_snapshot = getattr(command, "frontres_v015_deployment_sequence_snapshot", None)
    if not callable(read_snapshot):
        raise RuntimeError("v015 deployment context requires the command-owned sequence snapshot")
    snapshot = read_snapshot(env_ids)
    required = {
        "env_ids",
        "frame_indices",
        "current_q29_dq29",
        "intent_q29",
        "future_offsets",
        "reference_paths",
        "reference_stream_ids",
        "reference_file_hashes",
        "corruption_ids",
        "corruption_protocol_hashes",
        "corruption_families",
        "corruption_temporal_modes",
        "evaluation_kinds",
        "provenance",
    }
    if not isinstance(snapshot, dict) or set(snapshot) != required:
        raise RuntimeError(
            "v015 deployment context snapshot has an invalid schema: "
            f"got={sorted(snapshot) if isinstance(snapshot, dict) else type(snapshot).__name__}"
        )

    row_ids = snapshot["env_ids"]
    frame_indices = snapshot["frame_indices"]
    current = snapshot["current_q29_dq29"]
    intent = snapshot["intent_q29"]
    offsets = tuple(int(value) for value in snapshot["future_offsets"])
    if (
        not isinstance(row_ids, torch.Tensor)
        or row_ids.ndim != 1
        or row_ids.dtype != torch.long
        or int(torch.unique(row_ids).numel()) != int(row_ids.numel())
        or not isinstance(frame_indices, torch.Tensor)
        or tuple(frame_indices.shape) != tuple(row_ids.shape)
        or frame_indices.dtype != torch.long
    ):
        raise RuntimeError("v015 deployment context requires aligned unique row ids and [B] frame cursors")
    batch_size = int(row_ids.numel())
    max_offset = max(offsets, default=-1)
    tensors = {
        "current_q29_dq29": (current, (batch_size, 58)),
        "intent_q29": (intent, (batch_size, max_offset + 1, 29)),
    }
    for name, (value, shape) in tensors.items():
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.requires_grad
            or not torch.is_floating_point(value)
            or not bool(torch.isfinite(value).all().item())
        ):
            raise RuntimeError(f"v015 deployment {name} must be detached finite {shape} data")
    if not offsets or tuple(sorted(set(offsets))) != offsets or any(value <= 0 for value in offsets):
        raise RuntimeError("v015 deployment context requires ordered unique positive H offsets")

    metadata_names = (
        "reference_paths",
        "reference_stream_ids",
        "reference_file_hashes",
        "corruption_ids",
        "corruption_protocol_hashes",
        "corruption_families",
        "corruption_temporal_modes",
        "evaluation_kinds",
        "provenance",
    )
    if any(not isinstance(snapshot[name], tuple) or len(snapshot[name]) != batch_size for name in metadata_names):
        raise RuntimeError("v015 deployment context identity metadata must align one-to-one with command rows")
    if any(len(set(snapshot[name])) != 1 for name in metadata_names if name != "provenance") or any(
        value != snapshot["provenance"][0] for value in snapshot["provenance"]
    ):
        raise RuntimeError("v015 deployment context rejects mixed reference, protocol, or provenance rows")
    for row in range(batch_size):
        file_hash = str(snapshot["reference_file_hashes"][row])
        protocol_hash = str(snapshot["corruption_protocol_hashes"][row])
        if (
            snapshot["reference_stream_ids"][row] != f"deployment-npz:{file_hash}"
            or len(file_hash) != 64
            or any(ch not in "0123456789abcdef" for ch in file_hash)
            or len(protocol_hash) != 64
            or any(ch not in "0123456789abcdef" for ch in protocol_hash)
            or snapshot["corruption_temporal_modes"][row] != "persistent_full_sequence"
            or snapshot["evaluation_kinds"][row] != "deployment_composition_v015"
        ):
            raise RuntimeError("v015 deployment context has invalid reference/protocol identity")
        provenance = snapshot["provenance"][row]
        if provenance != {
            "reference_provenance": "deployment_reference_stream",
            "current_command_provenance": "deployment_q29_dq29",
            "intent_q29_provenance": "deployment_noisy_q29",
            "intent_q29_source": "deployment_npz_joint_pos",
        }:
            raise RuntimeError("v015 deployment context must retain deployment-only q29 provenance")

    return {
        **{
            name: value.detach().clone()
            for name, value in (
                ("env_ids", row_ids),
                ("frame_indices", frame_indices),
                ("current_q29_dq29", current),
                ("intent_q29", intent),
            )
        },
        "future_offsets": offsets,
        **{
            name: tuple(dict(value) for value in snapshot[name])
            if name == "provenance"
            else tuple(snapshot[name])
            for name in metadata_names
        },
    }


def build_frontres_v015_deployment_observation(
    self,
    obs: torch.Tensor,
    *,
    snapshot: dict[str, object] | None = None,
) -> torch.Tensor:
    # B1: 校验 870D base 与 58D q29 tail, 产出 928D FEMR/GMT observation.
    """Prepend deployment H to one raw 870D observation without actor execution."""

    authoritative = read_frontres_v015_deployment_context(self)
    if snapshot is not None:
        for name in ("env_ids", "frame_indices"):
            if not isinstance(snapshot.get(name), torch.Tensor) or not torch.equal(
                snapshot[name].to(authoritative[name].device), authoritative[name]
            ):
                raise RuntimeError("v015 deployment observation snapshot no longer matches the command cursor")
        for name in ("reference_stream_ids", "corruption_protocol_hashes", "future_offsets"):
            if snapshot.get(name) != authoritative[name]:
                raise RuntimeError("v015 deployment observation snapshot has mixed or stale identity")
    snapshot = authoritative
    layout = getattr(self, "_frontres_future_intent_layout", None)
    if not isinstance(layout, FrontRESFutureIntentLayout) or layout.version != FRONTRES_FUTURE_INTENT_LAYOUT_VERSION:
        raise RuntimeError("v015 deployment observation requires the frozen future-intent layout")
    if tuple(layout.future_offsets) != tuple(snapshot["future_offsets"]):
        raise RuntimeError("v015 deployment observation H offsets disagree with the sealed request")
    if not isinstance(obs, torch.Tensor) or obs.ndim != 2 or int(obs.shape[0]) != int(snapshot["env_ids"].numel()):
        raise RuntimeError("v015 deployment raw observation must be row-aligned [B,D]")
    intent = snapshot["intent_q29"]
    tail = intent[:, layout.future_offsets, :].reshape(int(intent.shape[0]), layout.actor_tail_dim).detach().clone()
    combined = torch.cat([tail.to(device=obs.device, dtype=obs.dtype), obs], dim=-1)
    policy = getattr(getattr(self, "alg", None), "policy", None)
    expected_actor = int(getattr(policy, "num_actor_obs", 0) or 0)
    expected_frontres = int(getattr(policy, "num_frontres_obs", 0) or 0)
    gmt_dim = int(getattr(self, "_frontres_gmt_obs_dim", 0) or 0)
    if (
        expected_actor <= 0
        or int(combined.shape[-1]) != expected_actor
        or gmt_dim <= 0
        or expected_frontres != expected_actor - gmt_dim
        or int(obs.shape[-1]) + layout.actor_tail_dim != expected_actor
    ):
        raise RuntimeError(
            "v015 deployment observation violates FEMR/GMT authority: "
            f"raw={tuple(obs.shape)} combined={tuple(combined.shape)} "
            f"actor={expected_actor} frontres={expected_frontres} gmt={gmt_dim}"
        )
    return combined


def append_frontres_fixed_noisy_future_context(self, obs: torch.Tensor) -> torch.Tensor:
    """Legacy v013 helper for a full 65D fixed-Noisy future tape.

    Status: legacy. The v015 actor route uses
    ``append_frontres_future_intent_context`` and rejects this carrier through
    its q29 actor-dimension check. This function remains only for isolated
    historical contract coverage until the formal route is migrated.
    """

    batch = _fixed_noisy_context_batch(self)
    if batch is None:
        if int(getattr(self, "_frontres_fixed_noisy_actor_context_dim", 0) or 0) > 0:
            raise RuntimeError(
                "fixed Noisy future context requires a selected sealed scenario before actor evaluation; "
                "the legacy raw observation route is forbidden"
            )
        return obs
    offsets = tuple(int(value) for value in (getattr(batch, "frontres_future_offsets", ()) or ()))
    if not offsets or any(value <= 0 for value in offsets) or tuple(sorted(set(offsets))) != offsets:
        raise RuntimeError("fixed Noisy actor context requires nonempty ordered frontres_future_offsets")
    command = _fixed_noisy_motion_command(self)
    read_context = getattr(command, "frontres_fixed_noisy_future_context", None)
    if not callable(read_context):
        raise RuntimeError("fixed Noisy actor context requires MultiMotionCommand future-tape reader")
    context = read_context(offsets)
    if not isinstance(context, torch.Tensor) or context.ndim != 2:
        raise RuntimeError(f"fixed Noisy actor context must be [B,|H|*65], got {getattr(context, 'shape', None)}")
    expected_tail = len(offsets) * 65
    if int(context.shape[0]) != int(obs.shape[0]) or int(context.shape[1]) != expected_tail:
        raise RuntimeError(
            "fixed Noisy actor context shape must match the actor batch and canonical 65D carrier: "
            f"obs={tuple(obs.shape)} context={tuple(context.shape)} expected_tail={expected_tail}"
        )
    augmented = torch.cat([context.to(device=obs.device, dtype=obs.dtype), obs], dim=-1)
    policy = getattr(getattr(self, "alg", None), "policy", None)
    expected_actor_dim = getattr(policy, "num_actor_obs", None)
    if expected_actor_dim is None or int(expected_actor_dim) != int(augmented.shape[-1]):
        raise RuntimeError(
            "fixed Noisy future context requires a v013 actor layout; legacy actor layout is forbidden: "
            f"actor_dim={expected_actor_dim} required={int(augmented.shape[-1])}"
        )
    gmt_dim = getattr(self, "_frontres_gmt_obs_dim", None)
    expected_frontres_dim = getattr(policy, "num_frontres_obs", None)
    if (
        gmt_dim is None
        or expected_frontres_dim is None
        or (
            int(expected_frontres_dim) != 0
            and int(expected_frontres_dim) != int(augmented.shape[-1]) - int(gmt_dim)
        )
    ):
        raise RuntimeError(
            "fixed Noisy future context requires a v013 FrontRES prefix layout; legacy actor layout is forbidden: "
            f"frontres_dim={expected_frontres_dim} required={int(augmented.shape[-1]) - int(gmt_dim or 0)}"
        )
    return augmented


def append_frontres_future_intent_context(self, obs: torch.Tensor) -> torch.Tensor:
    """Prepend the v015 actor-only q29 future-intent tail from the active carrier.

    Status: command-owned bridge. Upstream: proposal-only HSL or role-aligned
    local-scenario intent snapshot. Downstream: FrontRES actor prefix.
    Normalizer/formal observation is offline-S2 contract-confirmed at R5;
    simulator/live-runtime evidence remains open.
    """

    layout = getattr(self, "_frontres_future_intent_layout", None)
    expected_tail = int(getattr(self, "_frontres_future_intent_actor_context_dim", 0) or 0)
    snapshot = _future_intent_context_snapshot(self)
    if snapshot is None:
        if expected_tail > 0:
            raise RuntimeError(
                "v015 future-intent actor context requires the role-aligned command snapshot before actor evaluation; "
                "policy-attempt batch, legacy fixed-Noisy tape, and raw observation fallback are forbidden"
            )
        return obs
    if not isinstance(layout, FrontRESFutureIntentLayout):
        raise RuntimeError("v015 future-intent actor context requires a resolved FrontRESFutureIntentLayout")
    if layout.version != FRONTRES_FUTURE_INTENT_LAYOUT_VERSION:
        raise RuntimeError(
            "v015 future-intent actor context has an incompatible layout version: "
            f"{layout.version!r}"
        )
    intent_q29 = snapshot["intent_q29"]
    provenance = snapshot["provenance"]
    if bool(getattr(self, "_frontres_hsl_live_smoke_enabled", False)):
        command = _fixed_noisy_motion_command(self)
        artifact_pos = getattr(command, "anchor_dr_delta_pos", None)
        artifact_quat = getattr(command, "anchor_dr_delta_quat_correction", None)
        if (
            not isinstance(artifact_pos, torch.Tensor)
            or tuple(artifact_pos.shape) != (int(intent_q29.shape[0]), 3)
            or not isinstance(artifact_quat, torch.Tensor)
            or tuple(artifact_quat.shape) != (int(intent_q29.shape[0]), 4)
        ):
            raise RuntimeError("G2-S4 requires the real current root artifact [B,3]+[B,4]")
        self._frontres_hsl_smoke_context_snapshot = {
            "proposal_context_ids": tuple(snapshot["proposal_context_ids"]),
            "current_root_artifact_ids": tuple(snapshot["current_root_artifact_ids"]),
            "motion_indices": tuple(snapshot["motion_indices"]),
            "frame_indices": tuple(snapshot["frame_indices"]),
            "future_offsets": tuple(snapshot["future_offsets"]),
            "provenance": tuple(dict(value) for value in provenance),
            "intent_q29": intent_q29.detach().clone(),
            "artifact_pos": artifact_pos.detach().clone(),
            "artifact_quat": artifact_quat.detach().clone(),
        }
    try:
        context = build_frontres_future_intent_tail(
            intent_q29,
            layout=layout,
            provenance=provenance,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"v015 future-intent actor context is invalid: {exc}") from exc
    if int(context.shape[0]) != int(obs.shape[0]) or int(context.shape[1]) != layout.actor_tail_dim:
        raise RuntimeError(
            "v015 future-intent actor tail must match the actor batch and canonical q29 layout: "
            f"obs={tuple(obs.shape)} context={tuple(context.shape)} expected_tail={layout.actor_tail_dim}"
        )
    if expected_tail != layout.actor_tail_dim:
        raise RuntimeError(
            "runner future-intent actor-context dimension disagrees with its frozen layout: "
            f"configured={expected_tail} layout={layout.actor_tail_dim}"
        )
    augmented = torch.cat([context.to(device=obs.device, dtype=obs.dtype), obs], dim=-1)
    policy = getattr(getattr(self, "alg", None), "policy", None)
    expected_actor_dim = getattr(policy, "num_actor_obs", None)
    if expected_actor_dim is None or int(expected_actor_dim) != int(augmented.shape[-1]):
        raise RuntimeError(
            "v015 future-intent actor layout requires an exact q29 actor dimension: "
            f"actor_dim={expected_actor_dim} required={int(augmented.shape[-1])}"
        )
    gmt_dim = getattr(self, "_frontres_gmt_obs_dim", None)
    expected_frontres_dim = getattr(policy, "num_frontres_obs", None)
    if (
        gmt_dim is None
        or expected_frontres_dim is None
        or (
            int(expected_frontres_dim) != 0
            and int(expected_frontres_dim) != int(augmented.shape[-1]) - int(gmt_dim)
        )
    ):
        raise RuntimeError(
            "v015 future-intent actor layout requires an exact FrontRES prefix dimension: "
            f"frontres_dim={expected_frontres_dim} required={int(augmented.shape[-1]) - int(gmt_dim or 0)}"
        )
    return augmented


def get_inference_policy_runner(self, device=None):
    self.eval_mode()  # switch to evaluation mode (dropout for example)
    if device is not None:
        self.alg.policy.to(device)
    if self.cfg["empirical_normalization"] and device is not None:
        self.obs_normalizer.to(device)

    is_task_space_frontres = (
        isinstance(self.alg.policy, FrontRESActorCritic)
        and getattr(self.alg.policy, "num_task_corrections", 0) > 0
    )

    if is_task_space_frontres:
        def policy(x):  # noqa: E306
            with torch.inference_mode():
                raw_obs = append_frontres_future_intent_context(self, x.to(self.device))
                norm_obs = self._apply_obs_normalizer(raw_obs) if self.cfg["empirical_normalization"] else raw_obs
                correction = self.alg.policy.get_task_correction_inference(norm_obs)
                self._apply_frontres_task_corrections(correction, correction.shape[0], allow_oracle=False)
                obs_corr, extras_corr = self.env.get_observations()
                obs_corr_dict = extras_corr.get("observations", {})
                if self.policy_obs_type is not None and self.policy_obs_type in obs_corr_dict:
                    obs_corr = obs_corr_dict[self.policy_obs_type]
                obs_corr = append_frontres_future_intent_context(self, obs_corr.to(self.device))
                norm_corr = self._apply_obs_normalizer(obs_corr) if self.cfg["empirical_normalization"] else obs_corr
                return self.alg.policy.get_env_action(norm_corr, correction)
        return policy

    policy = self.alg.policy.act_inference
    if self.cfg["empirical_normalization"]:
        policy = lambda x: self.alg.policy.act_inference(  # noqa: E731
            self._apply_obs_normalizer(append_frontres_future_intent_context(self, x.to(self.device)))
        )
    return policy

def apply_obs_normalizer(self, obs: torch.Tensor) -> torch.Tensor:
    """分别归一化 FrontRES prefix 和 frozen-GMT suffix.

    函数名说明:
        `apply_obs_normalizer` 是 Stage 2/3 observation normalization adapter,
        负责组合两套持久统计; 它不是 observation builder, 也不更新 frozen GMT
        normalizer.

    主链路:
        上游: runner 从 env 取得 870D policy observation.
        下游: normalized observation 直接进入 FrontRES actor, 其中最后 gmt_dim
        维保持 GMT-compatible normalization.

    语义:
        IsaacLab 把 optional FrontRES terms 放在前缀. prefix 必须使用 Stage 2
        保存的 FrontRES stats, suffix 必须沿用 frozen GMT stats; 两者不可混用.
    """
    # B1: 将 policy observation 分为 FrontRES prefix 和 frozen-GMT suffix.
    extra, gmt_obs = split_frontres_policy_obs(obs, self._frontres_gmt_obs_dim)
    if extra is not None:
        num_extra = extra.shape[-1]
        gmt_part = self.obs_normalizer(gmt_obs)
        _s1_mean = getattr(self, '_frontres_extra_mean', None)
        _s1_std  = getattr(self, '_frontres_extra_std',  None)
        extra_normalizer = getattr(self, "_frontres_extra_normalizer", None)
        normalizer_mean = getattr(extra_normalizer, "_mean", None)
        normalizer_std = getattr(extra_normalizer, "_std", None)
        has_matching_checkpoint_stats = (
            isinstance(_s1_mean, torch.Tensor)
            and isinstance(_s1_std, torch.Tensor)
            and int(_s1_mean.shape[-1]) == int(num_extra)
            and int(_s1_std.shape[-1]) == int(num_extra)
        )
        has_matching_live_stats = (
            extra_normalizer is not None
            and isinstance(normalizer_mean, torch.Tensor)
            and isinstance(normalizer_std, torch.Tensor)
            and int(normalizer_mean.shape[-1]) == int(num_extra)
            and int(normalizer_std.shape[-1]) == int(num_extra)
        )
        future_intent_batch = _future_intent_context_batch(self)
        if future_intent_batch is not None:
            layout = getattr(self, "_frontres_future_intent_layout", None)
            if not isinstance(layout, FrontRESFutureIntentLayout) or layout.version != FRONTRES_FUTURE_INTENT_LAYOUT_VERSION:
                raise RuntimeError("v015 future-intent normalizer requires the resolved q29 layout version")
            policy = getattr(getattr(self, "alg", None), "policy", None)
            expected_prefix = getattr(policy, "num_frontres_obs", None)
            if expected_prefix is None or int(expected_prefix) != int(num_extra):
                raise RuntimeError(
                    "v015 future-intent normalizer prefix dimension disagrees with the actor layout: "
                    f"prefix={expected_prefix} observed={num_extra}"
                )
            has_any_checkpoint_stats = _s1_mean is not None or _s1_std is not None
            checkpoint_layout_version = getattr(self, "_frontres_extra_stats_layout_version", None)
            if has_any_checkpoint_stats and checkpoint_layout_version != layout.version:
                raise RuntimeError(
                    "v015 future-intent normalizer statistics are incompatible or unversioned; "
                    "legacy checkpoint prefix statistics are forbidden until persistence migration"
                )
            if not (has_matching_checkpoint_stats or has_matching_live_stats):
                raise RuntimeError(
                    "v015 future-intent normalizer requires exact q29-prefix statistics; "
                    "legacy normalizer layout is forbidden"
                )
        if has_matching_checkpoint_stats:
            extra = (extra - _s1_mean) / (_s1_std + 1e-8)
        else:
            if extra_normalizer is not None:
                extra = extra_normalizer(extra)
        # B2: 使用各自持久统计归一化 prefix 和 suffix, 再恢复原布局.
        normalized = torch.cat([extra, gmt_part], dim=-1)
    else:
        normalized = self.obs_normalizer(obs)
    # B3: AUDIT-OBS-01 截获 actor 实际消费的 normalized tensor.
    # Result: PENDING_LIVE.
    emit_formal_runtime_probe(
        "AUDIT-OBS-01",
        raw_obs=obs,
        frontres_prefix=extra,
        gmt_suffix=gmt_obs,
        normalized_obs=normalized,
    )
    return normalized


def _deployment_object_state_hash(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping{")
            for key in sorted(item, key=lambda entry: str(entry)):
                update(str(key))
                update(item[key])
            digest.update(b"}")
        elif isinstance(item, (tuple, list)):
            digest.update(f"sequence:{len(item)}[".encode("ascii"))
            for entry in item:
                update(entry)
            digest.update(b"]")
        elif item is None or isinstance(item, (str, int, float, bool)):
            digest.update(repr(item).encode("utf-8"))
        elif hasattr(item, "state_dict") and callable(item.state_dict):
            update(item.state_dict())
        elif hasattr(item, "__dict__"):
            update({key: entry for key, entry in vars(item).items() if not callable(entry)})
        else:
            digest.update(type(item).__qualname__.encode("utf-8"))

    update(value)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrontRESV015DeploymentContext:
    """Evaluator-facing deployment cursor and Intent snapshot."""

    env_ids: torch.Tensor
    frame_indices: torch.Tensor
    current_q29_dq29: torch.Tensor
    intent_q29: torch.Tensor
    future_offsets: tuple[int, ...]
    reference_paths: tuple[str, ...]
    reference_stream_ids: tuple[str, ...]
    reference_file_hashes: tuple[str, ...]
    corruption_ids: tuple[str, ...]
    corruption_protocol_hashes: tuple[str, ...]
    corruption_families: tuple[str, ...]
    corruption_temporal_modes: tuple[str, ...]
    evaluation_kinds: tuple[str, ...]
    provenance: tuple[Mapping[str, str], ...]

    def runner_snapshot(self) -> dict[str, object]:
        # B1: 将 validated value object 投影回 runner connector 的既有 snapshot schema.
        return {
            "env_ids": self.env_ids,
            "frame_indices": self.frame_indices,
            "current_q29_dq29": self.current_q29_dq29,
            "intent_q29": self.intent_q29,
            "future_offsets": self.future_offsets,
            "reference_paths": self.reference_paths,
            "reference_stream_ids": self.reference_stream_ids,
            "reference_file_hashes": self.reference_file_hashes,
            "corruption_ids": self.corruption_ids,
            "corruption_protocol_hashes": self.corruption_protocol_hashes,
            "corruption_families": self.corruption_families,
            "corruption_temporal_modes": self.corruption_temporal_modes,
            "evaluation_kinds": self.evaluation_kinds,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class FrontRESV015DeploymentFrameMetrics:
    """Validated row-aligned Physics evidence for one simulator frame."""

    fall: torch.Tensor
    zmp_margin: torch.Tensor
    actual_contact: torch.Tensor
    lateral_roll_rad: torch.Tensor


class FrontRESV015DeploymentRuntimeGateway:
    """Narrow simulator gateway for paired v015 deployment evaluation.

    The evaluator owns request semantics and report reduction. This gateway is
    the only deployment-composition object allowed to know runner-private
    connectors, IsaacLab scene/sensor access, or route-start restore details.
    """

    def __init__(self, runner: Any, command: Any, policy: Any) -> None:
        self._runner = runner
        self.command = command
        self.policy = policy

    @classmethod
    def from_runner(cls, runner: Any) -> "FrontRESV015DeploymentRuntimeGateway":
        """Bind and validate the runner-private deployment dependencies once."""

        # B1: 读取 command、FEMR/GMT 与 runner connectors, 产出 fail-closed dependency set.
        command = _fixed_noisy_motion_command(runner)
        if command is None:
            raise RuntimeError("v015 composition requires the formal motion command owner")
        lifecycle = (
            getattr(command, "set_frontres_v015_deployment_sequence", None),
            getattr(command, "clear_frontres_v015_deployment_sequence", None),
            getattr(command, "advance_frontres_v015_deployment_sequence", None),
        )
        if not all(callable(value) for value in lifecycle):
            raise RuntimeError("v015 composition requires the verified command carrier lifecycle")
        cfg = getattr(command, "cfg", None)
        if int(getattr(cfg, "motion_horizon", 0)) != 1 or not bool(getattr(cfg, "command_velocity", False)):
            raise RuntimeError("v015 composition requires GMT current command [q29,dq29]")
        policy = getattr(getattr(runner, "alg", None), "policy", None)
        if policy is None or int(getattr(policy, "num_task_corrections", 0) or 0) != 6:
            raise RuntimeError("v015 composition requires the full-6D FEMR policy")
        gmt = getattr(policy, "gmt_policy", None)
        if gmt is None or bool(getattr(gmt, "training", True)) or any(
            parameter.requires_grad for parameter in gmt.parameters()
        ):
            raise RuntimeError("v015 composition requires frozen GMT eval parameters")
        connectors = (
            getattr(policy, "get_task_correction_inference", None),
            getattr(policy, "get_env_action", None),
            getattr(runner, "_apply_frontres_task_corrections", None),
            getattr(runner, "_read_frontres_v015_deployment_context", None),
            getattr(runner, "_build_frontres_v015_deployment_observation", None),
        )
        if not all(callable(value) for value in connectors):
            raise RuntimeError("v015 composition formal FEMR/GMT connectors are incomplete")
        # B2: 安装唯一 runtime Gateway 并验证 row authority.
        gateway = cls(runner, command, policy)
        if gateway.row_count <= 0:
            raise RuntimeError("v015 composition requires a positive deployment row count")
        return gateway

    @property
    def device(self) -> torch.device:
        return torch.device(self._runner.device)

    @property
    def row_count(self) -> int:
        return int(getattr(self.command, "num_envs", 0))

    @property
    def config(self) -> Mapping[str, Any]:
        cfg = getattr(self._runner, "cfg", {})
        return cfg if isinstance(cfg, Mapping) else {}

    def set_sequence(self, request: Any) -> None:
        self.command.set_frontres_v015_deployment_sequence(request)

    def clear_sequence(self) -> None:
        self.command.clear_frontres_v015_deployment_sequence()

    def advance_sequence(self) -> None:
        self.command.advance_frontres_v015_deployment_sequence()

    def capture_route_start(self, *, comparison_signature: str) -> Any:
        # B1: 捕获 physical/command/RNG state, 产出 canonical route-start snapshot.
        from rsl_rl.runners.frontres_policy_quality_state import capture_frontres_policy_quality_state

        return capture_frontres_policy_quality_state(
            self._runner,
            env_ids=tuple(range(self.row_count)),
            comparison_signature=comparison_signature,
            role_layout=("deployment",) * self.row_count,
        )

    def restore_route_start(self, snapshot: Any, *, comparison_signature: str) -> str:
        # B1: 恢复 canonical snapshot 并核验 identity, 产出 matched state hash.
        from rsl_rl.runners.frontres_policy_quality_state import restore_frontres_policy_quality_state

        restored = restore_frontres_policy_quality_state(
            self._runner,
            snapshot,
            comparison_signature=comparison_signature,
        )
        if restored.initial_state_hash != snapshot.initial_state_hash:
            raise RuntimeError("v015 composition route-start state identity drifted")
        return str(restored.initial_state_hash)

    def read_context(self) -> FrontRESV015DeploymentContext:
        # B1: 收口 runner-owned payload, 产出 evaluator 只读 typed context.
        snapshot = self._runner._read_frontres_v015_deployment_context()
        if not isinstance(snapshot, Mapping):
            raise RuntimeError("v015 composition context connector must return a validated mapping")
        return FrontRESV015DeploymentContext(**snapshot)

    def read_policy_observation(self) -> torch.Tensor:
        # B1: 从 env observations 选择 policy rows, 产出 finite row-aligned tensor.
        obs, extras = self._runner.env.get_observations()
        obs_dict = extras.get("observations", {}) if isinstance(extras, Mapping) else {}
        obs_type = getattr(self._runner, "policy_obs_type", None)
        if obs_type is not None and obs_type in obs_dict:
            obs = obs_dict[obs_type]
        obs = torch.as_tensor(obs, device=self.device)
        if obs.ndim != 2 or not torch.is_floating_point(obs) or not bool(torch.isfinite(obs).all().item()):
            raise RuntimeError("v015 composition raw policy observation must be finite [B,D]")
        return obs

    def build_observation(
        self,
        obs: torch.Tensor,
        *,
        snapshot: FrontRESV015DeploymentContext,
    ) -> torch.Tensor:
        # B1: 将 typed context 转回唯一 runner connector, 产出正式 928D observation.
        if not isinstance(snapshot, FrontRESV015DeploymentContext):
            raise TypeError("v015 composition observation requires typed deployment context")
        return self._runner._build_frontres_v015_deployment_observation(
            obs,
            snapshot=snapshot.runner_snapshot(),
        )

    def normalize_observation(self, obs: torch.Tensor) -> torch.Tensor:
        # B1: 通过 runner normalizer 处理 928D observation, 保持 FEMR/GMT authority split.
        if not bool(self.config.get("empirical_normalization", False)):
            return obs
        normalize = getattr(self._runner, "_apply_obs_normalizer", None)
        if not callable(normalize):
            raise RuntimeError("v015 composition requires the formal observation normalizer")
        normalized = normalize(obs)
        if tuple(normalized.shape) != tuple(obs.shape) or not bool(torch.isfinite(normalized).all().item()):
            raise RuntimeError("v015 composition normalizer changed shape or produced non-finite values")
        return normalized

    def correction(self, observation: torch.Tensor, *, use_femr: bool) -> torch.Tensor:
        # B1: 执行或禁用 FEMR, 产出 [B,6] correction 且不修改 training state.
        correction = (
            self.policy.get_task_correction_inference(observation)
            if use_femr
            else torch.zeros((int(observation.shape[0]), 6), device=observation.device, dtype=observation.dtype)
        )
        if (
            not isinstance(correction, torch.Tensor)
            or tuple(correction.shape) != (int(observation.shape[0]), 6)
            or correction.requires_grad
            or not bool(torch.isfinite(correction).all().item())
        ):
            raise RuntimeError("v015 composition FEMR correction must be detached finite [B,6]")
        return correction

    def apply_correction(self, correction: torch.Tensor) -> None:
        self._runner._apply_frontres_task_corrections(
            correction, int(correction.shape[0]), allow_oracle=False
        )

    def gmt_action(self, observation: torch.Tensor, correction: torch.Tensor) -> torch.Tensor:
        # B1: 将 correction 注入 command 并调用 frozen GMT, 产出 env action rows.
        action = self.policy.get_env_action(observation, correction)
        if (
            not isinstance(action, torch.Tensor)
            or action.ndim != 2
            or int(action.shape[0]) != self.row_count
            or action.requires_grad
            or not bool(torch.isfinite(action).all().item())
        ):
            raise RuntimeError("v015 composition frozen GMT action must be detached finite [B,A]")
        return action

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, Any]:
        # B1: 推进 simulator 一步并拆分 dones/infos, 产出下一帧 evaluation state.
        env = self._runner.env
        _obs, _reward, dones, infos = env.step(action.to(env.device))
        dones = torch.as_tensor(dones, device=self.device, dtype=torch.bool).flatten()
        if int(dones.numel()) != self.row_count:
            raise RuntimeError("v015 composition dones must align with command rows")
        return dones, infos

    def executed_q29(self) -> torch.Tensor:
        value = getattr(getattr(getattr(self.command, "robot", None), "data", None), "joint_pos", None)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (self.row_count, 29):
            raise RuntimeError("v015 composition requires executed robot q29 aligned to deployment intent")
        return value

    def expected_physics(
        self,
        request: Any,
        *,
        clean_body_pos: torch.Tensor,
        clean_body_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # B1: 从 Clean continuation 派生 expected Contact 与 support envelope.
        provider = getattr(self._runner, "_frontres_v015_deployment_expected_physics_provider", None)
        if callable(provider):
            support, envelope = provider(request=request, command=self.command)
            support = torch.as_tensor(support, device=self.device, dtype=torch.bool)
            envelope = torch.as_tensor(envelope, device=self.device, dtype=torch.float32)
        else:
            left = int(getattr(self.command, "left_foot_idx", -1))
            right = int(getattr(self.command, "right_foot_idx", -1))
            if left < 0 or right < 0 or left == right or max(left, right) >= int(clean_body_pos.shape[1]):
                raise RuntimeError("v015 deployment Physics requires two valid Clean foot indices")
            from rsl_rl.frontres.frontres_balance import expected_support_and_envelope_from_foot_pose

            cfg = getattr(self.command, "cfg", None)
            support, envelope = expected_support_and_envelope_from_foot_pose(
                clean_body_pos[:, (left, right)],
                clean_body_quat[:, (left, right)],
                contact_height=float(getattr(cfg, "frontres_expected_contact_height", 0.08)),
                foot_half_length=float(getattr(cfg, "frontres_expected_foot_half_length", 0.10)),
                foot_half_width=float(getattr(cfg, "frontres_expected_foot_half_width", 0.05)),
            )
        if tuple(support.shape) != (int(request.frame_count), 2) or tuple(envelope.shape) != (
            int(request.frame_count),
            6,
        ):
            raise RuntimeError("v015 deployment expected Physics has an invalid trajectory shape")
        return support.detach().clone(), envelope.detach().clone()

    def prepare_metrics(self) -> None:
        if not callable(getattr(self._runner, "_frontres_v015_deployment_metric_provider", None)):
            from rsl_rl.frontres.frontres_balance import prepare_frontres_raw_contact_views

            prepare_frontres_raw_contact_views(self._runner)

    def evaluate_phase(
        self,
        expected: torch.Tensor,
        actual: torch.Tensor,
        margin: torch.Tensor,
        valid: torch.Tensor,
        *,
        dt: float,
    ) -> Mapping[str, torch.Tensor]:
        # B1: 按 expected support phase 归约 Contact/ZMP, 产出 recovery-aware evidence.
        provider = getattr(self._runner, "_frontres_v015_deployment_phase_provider", None)
        if not callable(provider):
            from rsl_rl.frontres.frontres_gain_legacy import evaluate_phase_conditioned_physics as provider
        return provider(
            expected,
            actual,
            margin,
            valid,
            timing_tolerance=int(self.config.get("frontres_physics_contact_timing_tolerance", 1)),
            recovery_window=int(self.config.get("frontres_physics_zmp_recovery_window", 1)),
            zmp_violation_scale=float(self.config.get("frontres_physics_zmp_violation_scale", 0.05)),
            dt=float(dt),
        )

    def frame_metrics(
        self,
        *,
        frame_index: int,
        dones: torch.Tensor,
        infos: Any,
        expected_support: torch.Tensor,
        expected_support_envelope: torch.Tensor,
    ) -> FrontRESV015DeploymentFrameMetrics:
        """Read one simulator frame and return validated row-aligned Physics evidence."""

        # B1: 调用显式 provider 或 ContactSensor fallback, 产出一帧 raw Physics fields.
        provider = getattr(self._runner, "_frontres_v015_deployment_metric_provider", None)
        if callable(provider):
            values = provider(
                frame_index=frame_index,
                dones=dones,
                infos=infos,
                command=self.command,
                expected_support=expected_support,
                expected_support_envelope=expected_support_envelope,
            )
        else:
            time_outs = infos.get("time_outs") if isinstance(infos, Mapping) else None
            time_outs = (
                torch.as_tensor(time_outs, device=dones.device, dtype=torch.bool)
                if time_outs is not None
                else torch.zeros_like(dones)
            )
            actual_contact, zmp_margin = self._contact_wrench_frame(
                expected_support=expected_support,
                expected_support_envelope=expected_support_envelope,
            )
            values = {
                "fall": dones & ~time_outs,
                "zmp_margin": zmp_margin,
                "actual_contact": actual_contact,
                "lateral_roll_rad": self._lateral_roll(),
            }
        # B2: 严格校验 schema、row shape 与 finite 语义, 产出 detached metric projection.
        required = {"fall", "zmp_margin", "actual_contact", "lateral_roll_rad"}
        if not isinstance(values, Mapping) or set(values) != required:
            raise RuntimeError("v015 composition metric provider returned an invalid schema")
        output: dict[str, torch.Tensor] = {}
        for name in required:
            value = torch.as_tensor(values[name], device=self.device)
            if name == "actual_contact":
                if tuple(value.shape) != (self.row_count, 2):
                    raise RuntimeError("v015 composition actual Contact must be row-aligned [B,2]")
                output[name] = value.bool().detach().clone()
                continue
            value = value.flatten()
            if int(value.numel()) != self.row_count:
                raise RuntimeError(f"v015 composition metric {name} must be row-aligned [B]")
            if name == "fall":
                value = value.bool()
            else:
                value = value.float()
                if name != "zmp_margin" and not bool(torch.isfinite(value).all().item()):
                    raise RuntimeError(f"v015 composition metric {name} must be finite")
            output[name] = value.detach().clone()
        return FrontRESV015DeploymentFrameMetrics(**output)

    def unplanned_contact_steps(self, expected: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        # B1: Gateway 解释 contact timing config, 产出 evaluator-ready unplanned transition mask.
        expected_transition = torch.zeros(expected.shape[:2], device=expected.device, dtype=torch.bool)
        actual_transition = torch.zeros_like(expected_transition)
        if int(expected.shape[0]) > 1:
            expected_transition[1:] = (expected[1:] != expected[:-1]).any(dim=-1)
            actual_transition[1:] = (actual[1:] != actual[:-1]).any(dim=-1)
        planned = torch.zeros_like(expected_transition)
        tolerance = int(self.config.get("frontres_physics_contact_timing_tolerance", 1))
        for delta in range(-tolerance, tolerance + 1):
            source = torch.arange(int(expected.shape[0]), device=expected.device) + delta
            inside = (source >= 0) & (source < int(expected.shape[0]))
            planned |= expected_transition.index_select(
                0,
                source.clamp(0, int(expected.shape[0]) - 1),
            ) & inside.unsqueeze(1)
        return actual_transition & ~planned

    def _contact_wrench_frame(
        self,
        *,
        expected_support: torch.Tensor,
        expected_support_envelope: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Derive actual Contact and loaded-support ZMP margin from foot sensors."""

        # B1: 解析左右脚 ContactSensor, 产出 force-threshold actual Contact 与 raw contact rows.
        env = getattr(self._runner.env, "unwrapped", self._runner.env)
        scene = getattr(env, "scene", None)
        if scene is None:
            raise RuntimeError("v015 deployment Physics requires the formal IsaacLab scene")
        from rsl_rl.frontres.frontres_balance import (
            contact_wrench_zmp_xy,
            expected_support_envelope_margin,
            pad_frontres_raw_contact_slots,
            read_frontres_raw_filtered_contact_rows,
        )

        sensors, actual = [], []
        for name in ("frontres_left_foot_contacts", "frontres_right_foot_contacts"):
            try:
                sensor = scene[name]
            except (KeyError, TypeError) as exc:
                raise RuntimeError(f"v015 deployment Physics is missing {name}") from exc
            force_matrix = getattr(getattr(sensor, "data", None), "force_matrix_w", None)
            threshold = getattr(getattr(sensor, "cfg", None), "force_threshold", None)
            if not isinstance(force_matrix, torch.Tensor) or not isinstance(threshold, (int, float)):
                raise RuntimeError(f"v015 deployment Physics requires filtered force_matrix_w for {name}")
            force_matrix = force_matrix.to(device=self.device, dtype=torch.float32)
            if force_matrix.ndim != 4 or tuple(force_matrix.shape[:2]) != (self.row_count, 1):
                raise RuntimeError(f"{name} filtered force matrix must be [B,1,F,3]")
            if not bool(torch.isfinite(force_matrix).all()) or not math.isfinite(float(threshold)) or float(threshold) <= 0:
                raise RuntimeError(f"{name} filtered contact forces/threshold must be finite with positive threshold")
            actual.append(force_matrix[..., 2].sum(dim=(1, 2)).abs() >= float(threshold))
            sensors.append(sensor)
        actual_contact = torch.stack(actual, dim=-1)
        raw = [
            read_frontres_raw_filtered_contact_rows(sensor, num_envs=self.row_count, device=self.device)
            for sensor in sensors
        ]
        slots = max(int(value[0].shape[2]) for value in raw)
        raw = [pad_frontres_raw_contact_slots(value, contact_slots=slots) for value in raw]
        # B2: 合并 contact points/forces/normals, 产出 contact-wrench ZMP 及 applicability.
        points = torch.cat(tuple(value[0] for value in raw), dim=1)
        forces = torch.cat(tuple(value[1] for value in raw), dim=1)
        normals = torch.cat(tuple(value[2] for value in raw), dim=1)
        valid = torch.cat(tuple(value[3] for value in raw), dim=1)
        zmp_xy, zmp_valid = contact_wrench_zmp_xy(points, forces, normals, valid)
        origins = getattr(scene, "env_origins", None)
        if not isinstance(origins, torch.Tensor) or tuple(origins.shape[:1]) != (self.row_count,):
            raise RuntimeError("v015 deployment Physics requires row-aligned scene.env_origins")
        # B3: 相对 expected support envelope 计算 margin, 无实际承重时产出 ZMP N/A.
        margin = expected_support_envelope_margin(
            zmp_xy,
            expected_support_envelope,
            expected_support,
            env_origins_xy=origins[:, :2].to(device=self.device, dtype=torch.float32),
        )
        required = expected_support.bool().any(dim=-1) & actual_contact.any(dim=-1)
        if bool((required & ~zmp_valid).any()):
            raise RuntimeError("v015 deployment loaded support is missing a finite contact-wrench resultant")
        return actual_contact.detach().clone(), torch.where(
            required, margin, torch.full_like(margin, float("nan"))
        ).detach().clone()

    def _lateral_roll(self) -> torch.Tensor:
        # B1: 从 root orientation 提取 row-aligned roll, 产出 sustained-lean trajectory field.
        quat = getattr(self.command, "robot_anchor_quat_w", None)
        if not isinstance(quat, torch.Tensor) or tuple(quat.shape) != (self.row_count, 4):
            raise RuntimeError("v015 deployment sustained-lean requires robot root quaternion [B,4]")
        w, x, y, z = quat.unbind(dim=-1)
        return torch.atan2(2 * (w * x + y * z), 1 - 2 * (x.square() + y.square())).detach().clone()

    def training_state_fingerprint(self) -> dict[str, str]:
        # B1: 哈希 policy/normalizer/optimizer/sampler facts, 产出 evaluation isolation anchor.
        alg = getattr(self._runner, "alg", None)
        objects = {
            "optimizer": getattr(alg, "optimizer", None),
            "sampler": getattr(self._runner, "_frontres_segment_sampler", None),
            "storage": getattr(self._runner, "storage", None),
            "transition": getattr(alg, "transition", None),
            "prefix_normalizer": getattr(self._runner, "_frontres_extra_normalizer", None),
            "gmt_normalizer": getattr(self._runner, "obs_normalizer", None),
            "privileged_normalizer": getattr(self._runner, "privileged_obs_normalizer", None),
            "teacher_normalizer": getattr(self._runner, "teacher_obs_normalizer", None),
        }
        return {name: _deployment_object_state_hash(value) for name, value in objects.items()}

    @contextmanager
    def inference_mode(self):
        # B1: 捕获并冻结 policy/GMT/normalizer modes, 产出可恢复 inference context.
        roots = (
            self.policy,
            getattr(self._runner, "_frontres_extra_normalizer", None),
            getattr(self._runner, "obs_normalizer", None),
            getattr(self._runner, "privileged_obs_normalizer", None),
            getattr(self._runner, "teacher_obs_normalizer", None),
        )
        modes: dict[torch.nn.Module, bool] = {}
        for root in roots:
            if isinstance(root, torch.nn.Module):
                for module in root.modules():
                    modes.setdefault(module, bool(module.training))
        for module in modes:
            module.training = False
        try:
            yield
        finally:
            for module, was_training in modes.items():
                module.training = was_training
