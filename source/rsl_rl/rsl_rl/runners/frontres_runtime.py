"""FrontRES inference and observation-normalization helpers.

Task-space correction and temporal reference cache helpers live under
``rsl_rl.frontres``. This runner module keeps inference wrapping and the
normalizer bridge used by ``OnPolicyRunner``.
"""

from __future__ import annotations

import torch

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
                if (
                    bool(self.cfg.get("frontres_state_alpha_enabled", True))
                    and hasattr(self.alg.policy, "get_state_router_alpha")
                ):
                    alpha_obs = norm_obs
                    self._frontres_state_alpha_prob_next = (
                        self.alg.policy.get_state_router_alpha(alpha_obs).view(-1).detach()
                    )
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
    """Apply obs_normalizer, with partial pass-through for FrontRES task-space mode.

    IsaacLab 会把 optional obs term 放在 regular term 前面.
    因此 FrontRES-only 信息位于前缀, 最后 gmt_dim 维保持 GMT-compatible suffix.
    这里只用 frozen GMT normalizer 归一化 suffix, prefix 使用 FrontRES 自己的统计值.
    """
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
        return torch.cat([extra, gmt_part], dim=-1)
    return self.obs_normalizer(obs)
