"""FrontRES inference and observation-normalization helpers.

Task-space correction and temporal reference cache helpers live under
``rsl_rl.frontres``. This runner module keeps inference wrapping and the
normalizer bridge used by ``OnPolicyRunner``.

Status: active. Upstream: OnPolicyRunner inference/export entrypoints.
Downstream: full-6D task correction and GMT action consumer.
Evidence: code-confirmed and contract-confirmed. Gap: real deployment runtime.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import torch

_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "frontres_formal_runtime_probe_runtime",
    Path(__file__).resolve().parents[1] / "frontres" / "frontres_formal_runtime_probe.py",
)
assert _AUDIT_SPEC is not None and _AUDIT_SPEC.loader is not None
_AUDIT_MODULE = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(_AUDIT_MODULE)
emit_formal_runtime_probe = _AUDIT_MODULE.emit_formal_runtime_probe

from rsl_rl.modules import FrontRESActorCritic
from rsl_rl.modules.frontres_observation_layout import (
    FRONTRES_FUTURE_INTENT_LAYOUT_VERSION,
    FrontRESFutureIntentLayout,
    build_frontres_future_intent_tail,
    split_frontres_policy_obs,
)
from rsl_rl.frontres.runtime_diagnostics import maybe_print_frontres_restore_debug


def _fixed_noisy_context_batch(self):
    batch = getattr(self, "_frontres_segment_live_current_batch", None)
    tape = getattr(batch, "frontres_fixed_noisy_tape", None) if batch is not None else None
    return batch if isinstance(tape, torch.Tensor) else None


def _future_intent_context_batch(self):
    batch = getattr(self, "_frontres_segment_live_current_batch", None)
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
    """Prepend the v015 actor-only q29 future-intent tail from a sealed local scenario.

    Status: role-aligned command-owned bridge. Upstream: the sealed
    MultiMotionCommand intent snapshot. Downstream: FrontRES actor prefix.
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
