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
from rsl_rl.modules.frontres_observation_layout import split_frontres_policy_obs
from rsl_rl.frontres.runtime_diagnostics import maybe_print_frontres_restore_debug


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
                raw_obs = x.to(self.device)
                norm_obs = self._apply_obs_normalizer(raw_obs) if self.cfg["empirical_normalization"] else raw_obs
                correction = self.alg.policy.get_task_correction_inference(norm_obs)
                self._apply_frontres_task_corrections(correction, correction.shape[0], allow_oracle=False)
                obs_corr, extras_corr = self.env.get_observations()
                obs_corr_dict = extras_corr.get("observations", {})
                if self.policy_obs_type is not None and self.policy_obs_type in obs_corr_dict:
                    obs_corr = obs_corr_dict[self.policy_obs_type]
                obs_corr = obs_corr.to(self.device)
                norm_corr = self._apply_obs_normalizer(obs_corr) if self.cfg["empirical_normalization"] else obs_corr
                return self.alg.policy.get_env_action(norm_corr, correction)
        return policy

    policy = self.alg.policy.act_inference
    if self.cfg["empirical_normalization"]:
        policy = lambda x: self.alg.policy.act_inference(self._apply_obs_normalizer(x.to(self.device)))  # noqa: E731
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
        if (_s1_mean is not None and _s1_std is not None
                and _s1_mean.shape[-1] == num_extra
                and _s1_std.shape[-1] == num_extra):
            extra = (extra - _s1_mean) / (_s1_std + 1e-8)
        else:
            extra_normalizer = getattr(self, "_frontres_extra_normalizer", None)
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
