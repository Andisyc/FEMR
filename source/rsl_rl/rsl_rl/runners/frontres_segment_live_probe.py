from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import torch

from rsl_rl.algorithms import FrontRESUnified
from rsl_rl.algorithms.frontres_segment_ppo import (
    FrontRESSegmentPPOBatch,
    FrontRESSegmentPPOConfig,
    compute_frontres_segment_ppo_loss,
)
from rsl_rl.frontres.frontres_segment_storage import (
    FrontRESSegmentRolloutStorage,
    FrontRESSegmentTransition,
)
from rsl_rl.frontres.training_schedule import configure_frontres_pair_layout, resolve_frontres_mode_state
from rsl_rl.modules import FrontRESActorCritic
from rsl_rl.runners.frontres_rollout_step import prepare_frontres_rollout_step


@dataclass
class FrontRESSegmentLiveObservations:
    obs: torch.Tensor
    privileged_obs: torch.Tensor
    teacher_obs: torch.Tensor
    ref_vel_estimator_obs: torch.Tensor | None


@dataclass
class FrontRESSegmentLiveRolloutCapture:
    rollout_k: int
    reward_mean: float
    done_frac: float
    last_obs_shape: tuple[int, ...]
    action_shape: tuple[int, ...] | None
    env_action_shape: tuple[int, ...] | None
    transition_obs: torch.Tensor | None
    transition_privileged_obs: torch.Tensor | None
    transition_actions: torch.Tensor | None
    transition_log_probs: torch.Tensor | None
    transition_values: torch.Tensor | None
    transition_means: torch.Tensor | None
    transition_sigmas: torch.Tensor | None
    reward_accum: torch.Tensor | None
    done_any: torch.Tensor | None


class FrontRESSegmentLivePolicyAdapter:
    def __init__(self, alg: FrontRESUnified, privileged_observations: torch.Tensor | None):
        self.alg = alg
        self.privileged_observations = privileged_observations

    def evaluate_segment_actions(self, observations: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        if bool(getattr(self.alg, "use_estimate_ref_vel", False)):
            raise NotImplementedError(
                "FrontRES Segment single-update sentinel does not yet store ref_vel_estimator observations."
            )
        self.alg.policy.act(observations)
        value_obs = self.privileged_observations if self.privileged_observations is not None else observations
        entropy = getattr(self.alg.policy, "entropy", None)
        if callable(entropy):
            entropy = entropy()
        if isinstance(entropy, torch.Tensor):
            entropy = entropy.reshape(-1)
            if entropy.numel() == 1 and actions.shape[0] != 1:
                entropy = entropy.expand(actions.shape[0])
        return {
            "log_prob": self.alg._get_actor_log_prob(actions).reshape(-1),
            "value": self.alg.policy.evaluate(value_obs).reshape(-1),
            "entropy": entropy if isinstance(entropy, torch.Tensor) else None,
        }


def run_frontres_segment_live_probe(runner: Any, init_at_random_ep_len: bool = True) -> dict[str, object]:
    single_update, storage_write = _resolve_probe_modes(runner)
    if init_at_random_ep_len:
        runner.env.episode_length_buf = torch.randint_like(
            runner.env.episode_length_buf, high=int(runner.env.max_episode_length)
        )

    observations = _read_live_observations(runner)
    runner.eval_mode()
    capture = _run_live_rollout_capture(runner, observations)
    summary = _initial_live_probe_summary(capture, storage_write=storage_write, single_update=single_update)

    if storage_write:
        segment_storage = build_live_segment_storage(runner, capture)
        storage_stats = segment_storage.stats()
        summary.update(
            {
                "storage_size": storage_stats.size,
                "storage_valid_frac": storage_stats.valid_frac,
                "storage_reward_mean": storage_stats.reward_mean,
            }
        )
        if single_update:
            ppo_result = run_frontres_segment_single_update(runner, segment_storage.full_batch())
            summary.update(
                {
                    "ppo_update": bool(ppo_result.should_step),
                    "ppo_total_loss": float(ppo_result.total_loss.detach().cpu().item()),
                    "ppo_actor_loss": float(ppo_result.actor_loss.detach().cpu().item()),
                    "ppo_value_loss": float(ppo_result.value_loss.detach().cpu().item()),
                    "ppo_valid_count": int(ppo_result.valid_count),
                    "ppo_approx_kl": float(ppo_result.approx_kl),
                    "ppo_clip_frac": float(ppo_result.clip_frac),
                }
            )
    _print_live_probe_summary(runner, capture, summary)
    return summary


def build_live_segment_storage(runner: Any, capture: FrontRESSegmentLiveRolloutCapture) -> FrontRESSegmentRolloutStorage:
    if (
        capture.transition_obs is None
        or capture.transition_privileged_obs is None
        or capture.transition_actions is None
        or capture.transition_log_probs is None
        or capture.transition_values is None
        or capture.reward_accum is None
        or capture.done_any is None
    ):
        raise RuntimeError("FrontRES Segment live storage probe did not capture a valid first-step PPO tuple.")
    if capture.transition_actions.ndim != 2 or capture.transition_actions.shape[-1] != 6:
        raise ValueError(f"live storage probe requires 6D actions, got {tuple(capture.transition_actions.shape)}")

    batch_size = int(capture.transition_actions.shape[0])
    sample = getattr(runner, "_frontres_segment_live_current_sample", None)
    sample_ids = getattr(sample, "segment_ids", None)
    sample_source = getattr(sample, "source", None)
    if sample_ids is not None and int(sample_ids.numel()) == batch_size:
        segment_ids = sample_ids.to(device=runner.device, dtype=torch.long).reshape(-1)
    else:
        segment_ids = torch.arange(batch_size, device=runner.device, dtype=torch.long)
    if sample_source is not None and len(sample_source) == batch_size:
        segment_source = tuple(str(item) for item in sample_source)
    else:
        segment_source = ("live_storage_probe",) * batch_size
    segment_storage = FrontRESSegmentRolloutStorage(
        capacity=batch_size,
        obs_shape=capture.transition_obs.shape[1:],
        action_dim=6,
        privileged_obs_shape=capture.transition_privileged_obs.shape[1:],
        device=runner.device,
    )
    segment_storage.add_transition(
        FrontRESSegmentTransition(
            observations=capture.transition_obs,
            privileged_observations=capture.transition_privileged_obs,
            actions=capture.transition_actions,
            old_log_probs=capture.transition_log_probs,
            values=capture.transition_values,
            rewards=capture.reward_accum.reshape(-1) / float(capture.rollout_k),
            valid_mask=~capture.done_any.reshape(-1).bool(),
            reset_mask=torch.ones(batch_size, device=runner.device, dtype=torch.bool),
            segment_ids=segment_ids,
            segment_source=segment_source,
            old_means=capture.transition_means,
            old_sigmas=capture.transition_sigmas,
            action_mask=torch.ones_like(capture.transition_actions, dtype=torch.bool),
        )
    )
    return segment_storage


def run_frontres_segment_single_update(runner: Any, storage_batch: Any) -> object:
    runner.train_mode()
    ppo_batch = storage_batch.to_ppo_batch(FrontRESSegmentPPOBatch)
    policy_adapter = FrontRESSegmentLivePolicyAdapter(
        runner.alg,
        privileged_observations=storage_batch.privileged_observations,
    )
    ppo_cfg = FrontRESSegmentPPOConfig(
        clip_param=float(getattr(runner.alg, "clip_param", 0.2)),
        value_clip_param=float(getattr(runner.alg, "clip_param", 0.2)),
        value_loss_coef=float(getattr(runner.alg, "value_loss_coef", 1.0)),
        entropy_coef=float(getattr(runner.alg, "entropy_coef", 0.0)),
        use_clipped_value_loss=bool(getattr(runner.alg, "use_clipped_value_loss", True)),
        normalize_advantages=bool(getattr(runner.alg, "normalize_advantage_per_mini_batch", False)),
    )
    ppo_result = compute_frontres_segment_ppo_loss(policy_adapter, ppo_batch, ppo_cfg)
    if ppo_result.should_step:
        runner.alg.optimizer.zero_grad()
        ppo_result.total_loss.backward()
        torch.nn.utils.clip_grad_norm_(runner.alg.policy.parameters(), float(getattr(runner.alg, "max_grad_norm", 1.0)))
        runner.alg.optimizer.step()
    runner.eval_mode()
    return ppo_result


def _resolve_probe_modes(runner: Any) -> tuple[bool, bool]:
    single_update = bool(
        runner._frontres_segment_replay_boundary.live_single_update_only
        or runner._frontres_segment_replay_boundary.live_update_loop_only
        or runner._frontres_segment_replay_boundary.live_train_enabled
    )
    storage_write = bool(runner._frontres_segment_replay_boundary.live_storage_write_only or single_update)
    if not (runner._frontres_segment_replay_boundary.live_probe_only or storage_write):
        raise ValueError(
            "FrontRES Segment live probe requires frontres_segment_live_probe_only=True "
            "or frontres_segment_live_storage_write_only=True "
            "or frontres_segment_live_single_update_only=True "
            "or frontres_segment_live_update_loop_only=True."
        )
    return single_update, storage_write


def _read_live_observations(runner: Any) -> FrontRESSegmentLiveObservations:
    obs, extras = runner.env.get_observations()
    obs_dict = extras.get("observations", {})
    if runner.policy_obs_type is not None and runner.policy_obs_type in obs_dict:
        obs = obs_dict[runner.policy_obs_type]
    privileged_obs = obs_dict.get(runner.privileged_obs_type, obs)
    teacher_obs = obs_dict.get(runner.teacher_obs_type)
    if teacher_obs is None:
        teacher_obs = privileged_obs
    ref_vel_estimator_obs = obs_dict.get(runner.ref_vel_estimator_obs_type)

    obs = runner._apply_obs_normalizer(obs.to(runner.device))
    privileged_obs = runner.privileged_obs_normalizer(privileged_obs.to(runner.device))
    teacher_obs = runner.teacher_obs_normalizer(teacher_obs.to(runner.device))
    if ref_vel_estimator_obs is not None:
        ref_vel_estimator_obs = ref_vel_estimator_obs.to(runner.device)
    return FrontRESSegmentLiveObservations(
        obs=obs,
        privileged_obs=privileged_obs,
        teacher_obs=teacher_obs,
        ref_vel_estimator_obs=ref_vel_estimator_obs,
    )


def _run_live_rollout_capture(
    runner: Any,
    observations: FrontRESSegmentLiveObservations,
) -> FrontRESSegmentLiveRolloutCapture:
    frontres_mode = resolve_frontres_mode_state(runner, FrontRESActorCritic)
    pair_layout = configure_frontres_pair_layout(runner, is_frontres=frontres_mode.is_frontres)
    rollout_k = max(1, int(getattr(runner.alg, "frontres_segment_k", runner._frontres_segment_replay_boundary.segment_k)))
    vel_est_error_buffer = deque(maxlen=1)
    reward_sum = 0.0
    done_sum = 0.0
    reward_accum = None
    done_any = None
    transition_obs = None
    transition_privileged_obs = None
    transition_actions = None
    transition_log_probs = None
    transition_values = None
    transition_means = None
    transition_sigmas = None
    action_shape = None
    env_action_shape = None
    obs = observations.obs
    privileged_obs = observations.privileged_obs
    teacher_obs = observations.teacher_obs
    ref_vel_estimator_obs = observations.ref_vel_estimator_obs
    last_obs_shape = tuple(obs.shape)

    with torch.inference_mode():
        for rollout_step in range(rollout_k):
            step_plan = prepare_frontres_rollout_step(
                runner,
                obs=obs,
                privileged_obs=privileged_obs,
                teacher_obs=teacher_obs,
                ref_vel_estimator_obs=ref_vel_estimator_obs,
                obs_raw_for_gmt=None,
                vel_est_error_buffer=vel_est_error_buffer,
                iteration=runner.current_learning_iteration,
                rollout_step=rollout_step,
                is_frontres=frontres_mode.is_frontres,
                is_task_space_mode=frontres_mode.is_task_space_mode,
                n_train=pair_layout.n_train,
                n_candidate=pair_layout.n_candidate,
                n_base=pair_layout.n_base,
                n_clean=pair_layout.n_clean,
            )
            actions = step_plan.actions
            env_actions = step_plan.env_actions
            action_shape = tuple(actions.shape) if actions is not None else None
            env_action_shape = tuple(env_actions.shape)
            if rollout_step == 0 and actions is not None:
                transition_obs = runner.alg.transition.observations.detach().clone()
                transition_privileged_obs = runner.alg.transition.privileged_observations.detach().clone()
                transition_actions = actions.detach().clone()
                transition_log_probs = runner.alg.transition.actions_log_prob.detach().clone().reshape(-1)
                transition_values = runner.alg.transition.values.detach().clone().reshape(-1)
                action_mean = getattr(runner.alg.transition, "action_mean", None)
                action_sigma = getattr(runner.alg.transition, "action_sigma", None)
                if action_mean is not None and tuple(action_mean.shape) == tuple(transition_actions.shape):
                    transition_means = action_mean.detach().clone()
                if action_sigma is not None and tuple(action_sigma.shape) == tuple(transition_actions.shape):
                    transition_sigmas = action_sigma.detach().clone()

            obs, rewards, dones, infos = runner.env.step(env_actions.to(runner.env.device))
            rewards = rewards.to(runner.device)
            dones = dones.to(runner.device)
            reward_sum += float(rewards.mean().detach().cpu())
            done_sum += float(dones.float().mean().detach().cpu())
            reward_accum = rewards.detach().clone() if reward_accum is None else reward_accum + rewards.detach()
            done_any = dones.detach().clone() if done_any is None else (done_any | dones.detach())

            obs, privileged_obs, teacher_obs, ref_vel_estimator_obs = _read_step_observations(runner, obs, infos)
            last_obs_shape = tuple(obs.shape)

    return FrontRESSegmentLiveRolloutCapture(
        rollout_k=rollout_k,
        reward_mean=reward_sum / float(rollout_k),
        done_frac=done_sum / float(rollout_k),
        last_obs_shape=last_obs_shape,
        action_shape=action_shape,
        env_action_shape=env_action_shape,
        transition_obs=transition_obs,
        transition_privileged_obs=transition_privileged_obs,
        transition_actions=transition_actions,
        transition_log_probs=transition_log_probs,
        transition_values=transition_values,
        transition_means=transition_means,
        transition_sigmas=transition_sigmas,
        reward_accum=reward_accum,
        done_any=done_any,
    )


def _read_step_observations(runner: Any, obs: torch.Tensor, infos: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    obs_dict = infos.get("observations", {})
    if runner.policy_obs_type is not None and runner.policy_obs_type in obs_dict:
        obs = obs_dict[runner.policy_obs_type].to(runner.device)
    else:
        obs = obs.to(runner.device)
    obs = runner._apply_obs_normalizer(obs)
    if runner.privileged_obs_type is not None and runner.privileged_obs_type in obs_dict:
        privileged_obs = runner.privileged_obs_normalizer(obs_dict[runner.privileged_obs_type].to(runner.device))
    else:
        privileged_obs = obs
    if runner.teacher_obs_type is not None and runner.teacher_obs_type in obs_dict:
        teacher_obs = runner.teacher_obs_normalizer(obs_dict[runner.teacher_obs_type].to(runner.device))
    else:
        teacher_obs = privileged_obs
    if runner.ref_vel_estimator_obs_type is not None and runner.ref_vel_estimator_obs_type in obs_dict:
        ref_vel_estimator_obs = obs_dict[runner.ref_vel_estimator_obs_type].to(runner.device)
    else:
        ref_vel_estimator_obs = None
    return obs, privileged_obs, teacher_obs, ref_vel_estimator_obs


def _initial_live_probe_summary(
    capture: FrontRESSegmentLiveRolloutCapture,
    *,
    storage_write: bool,
    single_update: bool,
) -> dict[str, object]:
    return {
        "reward_mean": capture.reward_mean,
        "done_frac": capture.done_frac,
        "valid_mask_frac": 1.0 - capture.done_frac,
        "storage_write": storage_write,
        "storage_size": 0,
        "storage_valid_frac": 0.0,
        "storage_reward_mean": 0.0,
        "single_update": single_update,
        "ppo_update": False,
        "ppo_valid_count": 0,
        "ppo_total_loss": 0.0,
        "ppo_actor_loss": 0.0,
        "ppo_value_loss": 0.0,
        "ppo_approx_kl": 0.0,
        "ppo_clip_frac": 0.0,
    }


def _print_live_probe_summary(
    runner: Any,
    capture: FrontRESSegmentLiveRolloutCapture,
    summary: dict[str, object],
) -> None:
    action_6d = bool(capture.action_shape is not None and len(capture.action_shape) >= 2 and capture.action_shape[-1] == 6)
    print(
        "[FrontRES Segment Live Probe] "
        f"objective={getattr(runner.alg, 'frontres_training_objective', 'n/a')} "
        "segment_id=live_env_current "
        f"reset_mode={runner._frontres_segment_replay_boundary.reset_mode} "
        f"obs_shape={capture.last_obs_shape} "
        f"action_shape={capture.action_shape} "
        f"action_6d={action_6d} "
        f"env_action_shape={capture.env_action_shape} "
        f"valid_mask_frac={float(summary['valid_mask_frac']):.4f} "
        f"rollout_k={capture.rollout_k} "
        f"reward_mean={float(summary['reward_mean']):.6f} "
        f"done_frac={float(summary['done_frac']):.6f} "
        f"storage_write={bool(summary['storage_write'])} "
        f"storage_size={int(summary['storage_size'])} "
        f"storage_valid_frac={float(summary['storage_valid_frac']):.4f} "
        f"storage_reward_mean={float(summary['storage_reward_mean']):.6f} "
        f"single_update={bool(summary['single_update'])} "
        f"ppo_update={bool(summary['ppo_update'])} "
        f"ppo_valid_count={int(summary['ppo_valid_count'])} "
        f"ppo_total_loss={float(summary['ppo_total_loss']):.6f} "
        f"ppo_actor_loss={float(summary['ppo_actor_loss']):.6f} "
        f"ppo_value_loss={float(summary['ppo_value_loss']):.6f} "
        f"ppo_approx_kl={float(summary['ppo_approx_kl']):.6f} "
        f"ppo_clip_frac={float(summary['ppo_clip_frac']):.6f}",
        flush=True,
    )
