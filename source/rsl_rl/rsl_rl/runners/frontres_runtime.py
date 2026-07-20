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

    Status: connected only through the explicit pre-live v015 sentinel. Upstream:
    the local-scenario actor bridge. Downstream: FrontRES actor prefix and its
    dedicated normalizer. Generic training and live-runtime evidence remain open.
    """

    layout = getattr(self, "_frontres_future_intent_layout", None)
    expected_tail = int(getattr(self, "_frontres_future_intent_actor_context_dim", 0) or 0)
    batch = _future_intent_context_batch(self)
    if batch is None:
        if expected_tail > 0:
            raise RuntimeError(
                "v015 future-intent actor context requires a sealed local scenario before actor evaluation; "
                "legacy fixed-Noisy tape and raw observation fallback are forbidden"
            )
        return obs
    if not isinstance(layout, FrontRESFutureIntentLayout):
        raise RuntimeError("v015 future-intent actor context requires a resolved FrontRESFutureIntentLayout")
    if layout.version != FRONTRES_FUTURE_INTENT_LAYOUT_VERSION:
        raise RuntimeError(
            "v015 future-intent actor context has an incompatible layout version: "
            f"{layout.version!r}"
        )
    offsets = tuple(int(value) for value in (getattr(batch, "frontres_future_offsets", ()) or ()))
    if offsets != layout.future_offsets:
        raise RuntimeError(
            "sealed local scenario offsets do not match the runner future-intent layout: "
            f"scenario={offsets}, runner={layout.future_offsets}"
        )
    intent_q29 = getattr(batch, "frontres_local_scenario_intent_q29", None)
    provenance = getattr(batch, "frontres_local_scenario_provenance", None)
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
