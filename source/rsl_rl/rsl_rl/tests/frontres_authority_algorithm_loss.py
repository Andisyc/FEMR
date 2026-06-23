"""TEST ONLY: FrontRES authority actor-critic algorithm loss.

This test uses the formal FrontRESUnified.update() path with synthetic storage.
It checks that authority fields from RolloutStorage train the Stage-2
authority actor/critic without touching the Stage-1 proposal actor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsl_rl.algorithms.frontres_unified import FrontRESUnified
from rsl_rl.storage.rollout_storage import RolloutStorage


class TinyAuthorityPolicy(nn.Module):
    is_recurrent = False
    num_task_corrections = 6
    task_conf_dim = 6
    max_delta_pos = 0.30
    max_delta_rpy = 0.40

    def __init__(self, obs_dim: int, critic_obs_dim: int):
        super().__init__()
        self.residual_actor = nn.Linear(obs_dim, 12)
        self.value_critic = nn.Linear(critic_obs_dim, 1)
        self.authority_actor = nn.Linear(obs_dim + 6, 6)
        self.authority_critic = nn.Linear(obs_dim + 12, 1)
        with torch.no_grad():
            self.residual_actor.weight.zero_()
            self.residual_actor.bias.zero_()
            self.value_critic.weight.zero_()
            self.value_critic.bias.zero_()
            self.authority_actor.weight.zero_()
            self.authority_actor.bias.fill_(torch.logit(torch.tensor(0.25)).item())
            self.authority_critic.weight.zero_()
            self.authority_critic.bias.zero_()
            # Make Q initially increase with rho so the actor has a clear signal,
            # while still leaving critic error against the synthetic K-step return.
            self.authority_critic.weight[:, -6:].fill_(0.5)
        self.action_mean = None
        self.action_std = None
        self.entropy = None

    @property
    def critic(self):
        return self.value_critic

    def update_distribution(self, observations: torch.Tensor) -> None:
        self.action_mean = self.residual_actor(observations)
        self.action_std = torch.ones_like(self.action_mean)
        self.entropy = torch.zeros(observations.shape[0], device=observations.device)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.value_critic(critic_observations)

    def get_authority_rho(
        self,
        observations: torch.Tensor,
        proposal_delta_se: torch.Tensor | None = None,
        *,
        active_task_dims: torch.Tensor | None = None,
        detach_proposal: bool = True,
    ) -> torch.Tensor:
        if proposal_delta_se is None:
            proposal_delta_se = torch.zeros(observations.shape[0], 6, device=observations.device)
        if detach_proposal:
            proposal_delta_se = proposal_delta_se.detach()
        rho = torch.sigmoid(self.authority_actor(torch.cat([observations, proposal_delta_se], dim=-1)))
        if active_task_dims is not None:
            rho = rho * active_task_dims.to(device=rho.device, dtype=rho.dtype)
        return rho

    def evaluate_authority_q(
        self,
        observations: torch.Tensor,
        proposal_delta_se: torch.Tensor,
        authority_rho: torch.Tensor,
        *,
        detach_proposal: bool = True,
    ) -> torch.Tensor:
        if detach_proposal:
            proposal_delta_se = proposal_delta_se.detach()
        return self.authority_critic(torch.cat([observations, proposal_delta_se, authority_rho], dim=-1))


def _build_storage(policy: TinyAuthorityPolicy, *, obs_dim: int, critic_obs_dim: int) -> RolloutStorage:
    num_envs = 6
    storage = RolloutStorage(
        training_type="frontres",
        num_envs=num_envs,
        num_transitions_per_env=1,
        obs_shape=(obs_dim,),
        privileged_obs_shape=(critic_obs_dim,),
        actions_shape=(12,),
        device="cpu",
    )
    obs = torch.linspace(-1.0, 1.0, steps=num_envs * obs_dim).view(num_envs, obs_dim)
    proposal = torch.ones(num_envs, 6) * 0.2
    behavior_rho = torch.zeros(num_envs, 6)
    behavior_rho[:3] = 1.0
    authority_return = torch.zeros(num_envs, 1)
    authority_return[:3] = 6.0

    transition = RolloutStorage.Transition()
    transition.observations = obs
    transition.privileged_observations = torch.zeros(num_envs, critic_obs_dim)
    transition.actions = torch.zeros(num_envs, 12)
    transition.rewards = torch.zeros(num_envs)
    transition.dones = torch.zeros(num_envs)
    transition.values = torch.zeros(num_envs, 1)
    transition.actions_log_prob = torch.zeros(num_envs)
    transition.action_mean = torch.zeros(num_envs, 12)
    transition.action_sigma = torch.ones(num_envs, 12)
    transition.frontres_mask = torch.ones(num_envs, 1)
    transition.frontres_actor_gate = torch.ones(num_envs, 1)
    transition.supervised_target = torch.zeros(num_envs, 6)
    transition.supervised_weight = torch.zeros(num_envs, 1)
    transition.supervised_harm_weight = torch.zeros(num_envs, 1)
    transition.proposal_delta_se = proposal
    transition.authority_action = behavior_rho
    transition.authority_log_prob = torch.zeros(num_envs, 1)
    transition.authority_rho = behavior_rho
    transition.authority_return_k = authority_return
    transition.authority_mask = torch.ones(num_envs, 1)
    storage.add_transitions(transition)
    storage.returns.zero_()
    storage.advantages.zero_()
    return storage


def test_authority_update_moves_actor_and_critic() -> None:
    torch.manual_seed(23)
    obs_dim = 5
    critic_obs_dim = 3
    policy = TinyAuthorityPolicy(obs_dim, critic_obs_dim)
    storage = _build_storage(policy, obs_dim=obs_dim, critic_obs_dim=critic_obs_dim)
    obs = storage.observations.flatten(0, 1)
    proposal = storage.proposal_delta_se.flatten(0, 1)
    behavior_rho = storage.authority_action.flatten(0, 1)
    target_return = storage.authority_return_k.flatten(0, 1)

    with torch.no_grad():
        initial_rho = policy.get_authority_rho(obs, proposal).mean().item()
        initial_q = policy.evaluate_authority_q(obs, proposal, behavior_rho)
        initial_mse = ((initial_q - target_return) ** 2).mean().item()
        initial_proposal_weight = policy.residual_actor.weight.detach().clone()

    alg = FrontRESUnified(
        policy=policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=5.0e-2,
        value_loss_coef=0.0,
        entropy_coef=0.0,
        lambda_supervised=0.0,
        frontres_training_objective="hsl_hybrid",
        frontres_authority_actor_critic_enabled=True,
        frontres_authority_actor_loss_weight=1.0,
        frontres_authority_critic_loss_weight=1.0,
        frontres_structured_joint_rl_enabled=False,
        device="cpu",
    )
    # Authority actor-critic owns rho.  Generic PPO may be scheduled as 1.0 by
    # the runner, but the algorithm must hard-disable it in authority mode.
    alg.ppo_actor_weight = 1.0
    alg.storage = storage
    metrics = alg.update()

    with torch.no_grad():
        final_rho = policy.get_authority_rho(obs, proposal).mean().item()
        final_q = policy.evaluate_authority_q(obs, proposal, behavior_rho)
        final_mse = ((final_q - target_return) ** 2).mean().item()
        proposal_weight_delta = (policy.residual_actor.weight - initial_proposal_weight).abs().max().item()

    if final_rho <= initial_rho:
        raise AssertionError(f"authority actor did not increase rho: {initial_rho:.4f} -> {final_rho:.4f}")
    if final_mse >= initial_mse:
        raise AssertionError(f"authority critic MSE did not improve: {initial_mse:.4f} -> {final_mse:.4f}")
    if proposal_weight_delta > 0.0:
        raise AssertionError("authority loss changed the Stage-1 proposal actor.")
    if metrics["authority_active_frac"] <= 0.0:
        raise AssertionError("authority metrics did not report active samples.")
    if metrics["authority_critic_loss"] <= 0.0:
        raise AssertionError("authority critic loss metric should be positive in this toy setup.")
    if metrics["authority_actor_critic_enabled"] != 1.0:
        raise AssertionError("authority actor-critic enabled sentinel was not reported.")
    if metrics["authority_actor_phase_weight"] != 1.0:
        raise AssertionError("authority actor should be at full phase weight in the default no-warmup test.")
    if metrics["lambda_authority_actor_effective"] != metrics["lambda_authority_actor"]:
        raise AssertionError("effective authority actor weight should equal base weight after takeover.")
    for key in (
        "authority_rho_std",
        "authority_rho_near_zero_frac",
        "authority_rho_near_one_frac",
        "authority_proposal_abs_mean",
        "authority_return_low_rho_mean",
        "authority_q_actor_low_rho_mean",
    ):
        if key not in metrics:
            raise AssertionError(f"missing Step-9 authority diagnostic metric: {key}")
    if metrics["ppo_actor_weight"] != 0.0:
        raise AssertionError("generic PPO actor weight should be disabled in authority actor-critic mode.")

    print("=== FrontRES Authority Algorithm Loss TEST ONLY ===")
    print(
        f"rho_mean: {initial_rho:.3f} -> {final_rho:.3f}; "
        f"critic_mse: {initial_mse:.3f} -> {final_mse:.3f}; "
        f"proposal_delta={proposal_weight_delta:.3e}"
    )
    print("checks=formal update path, actor moves toward higher Q, critic fits K-step return, Stage-1 untouched, generic PPO disabled, Step-9 diagnostics present")
    print("result: PASS")


def test_authority_actor_warmup_holds_actor_and_trains_critic() -> None:
    torch.manual_seed(29)
    obs_dim = 5
    critic_obs_dim = 3
    policy = TinyAuthorityPolicy(obs_dim, critic_obs_dim)
    storage = _build_storage(policy, obs_dim=obs_dim, critic_obs_dim=critic_obs_dim)
    obs = storage.observations.flatten(0, 1)
    proposal = storage.proposal_delta_se.flatten(0, 1)
    behavior_rho = storage.authority_action.flatten(0, 1)
    target_return = storage.authority_return_k.flatten(0, 1)

    with torch.no_grad():
        initial_actor_weight = policy.authority_actor.weight.detach().clone()
        initial_actor_bias = policy.authority_actor.bias.detach().clone()
        initial_q = policy.evaluate_authority_q(obs, proposal, behavior_rho)
        initial_mse = ((initial_q - target_return) ** 2).mean().item()

    alg = FrontRESUnified(
        policy=policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=5.0e-2,
        value_loss_coef=0.0,
        entropy_coef=0.0,
        lambda_supervised=0.0,
        frontres_training_objective="hsl_hybrid",
        frontres_authority_actor_critic_enabled=True,
        frontres_authority_actor_loss_weight=1.0,
        frontres_authority_critic_loss_weight=1.0,
        frontres_authority_actor_warmup_iterations=10,
        frontres_authority_actor_ramp_iterations=5,
        frontres_structured_joint_rl_enabled=False,
        device="cpu",
    )
    alg.current_learning_iteration = 0
    alg.storage = storage
    metrics = alg.update()

    with torch.no_grad():
        actor_weight_delta = (policy.authority_actor.weight - initial_actor_weight).abs().max().item()
        actor_bias_delta = (policy.authority_actor.bias - initial_actor_bias).abs().max().item()
        final_q = policy.evaluate_authority_q(obs, proposal, behavior_rho)
        final_mse = ((final_q - target_return) ** 2).mean().item()

    if actor_weight_delta > 0.0 or actor_bias_delta > 0.0:
        raise AssertionError(
            "authority actor changed during warmup: "
            f"weight_delta={actor_weight_delta:.3e}, bias_delta={actor_bias_delta:.3e}"
        )
    if final_mse >= initial_mse:
        raise AssertionError(f"authority critic did not improve during actor warmup: {initial_mse:.4f} -> {final_mse:.4f}")
    if metrics["authority_actor_phase_weight"] != 0.0:
        raise AssertionError("authority actor phase should be 0 during warmup.")
    if metrics["lambda_authority_actor_effective"] != 0.0:
        raise AssertionError("effective authority actor weight should be 0 during warmup.")
    if metrics["authority_actor_warmup_active"] != 1.0:
        raise AssertionError("warmup sentinel should be active.")

    print("=== FrontRES Authority Actor Warmup TEST ONLY ===")
    print(
        f"actor_delta={max(actor_weight_delta, actor_bias_delta):.3e}; "
        f"critic_mse: {initial_mse:.3f} -> {final_mse:.3f}; "
        f"phase={metrics['authority_actor_phase_weight']:.3f}"
    )
    print("checks=critic learns while Stage-2 authority actor is held during warmup")
    print("result: PASS")


if __name__ == "__main__":
    test_authority_update_moves_actor_and_critic()
    test_authority_actor_warmup_holds_actor_and_trains_critic()
