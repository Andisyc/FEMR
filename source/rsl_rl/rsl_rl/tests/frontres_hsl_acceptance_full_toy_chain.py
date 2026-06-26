#!/usr/bin/env python3
"""Step 10 full toy chain for active FEMR HSL + acceptance.

This test intentionally stays environment-free.  It connects the active modules
in the same conceptual order as training:

mock obs -> Stage-1 proposal + Stage-2 acceptance logits -> rollout evidence
labels -> storage fields -> acceptance loss -> active diagnostics.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsl_rl.algorithms.frontres_unified import FrontRESUnified
from rsl_rl.storage.rollout_storage import RolloutStorage

ROOT = Path(__file__).resolve().parents[4]
POLICY_PATH = ROOT / "source/rsl_rl/rsl_rl/modules/front_residual_actor_critic.py"
DIAG_PATH = ROOT / "source/rsl_rl/rsl_rl/frontres/frontres_diagnostics.py"
LABEL_PATH = ROOT / "source/rsl_rl/rsl_rl/frontres/frontres_acceptance_labels.py"


class ProposalOnlyActor(nn.Module):
    def __init__(self, raw: torch.Tensor):
        super().__init__()
        self.raw = nn.Parameter(raw.clone())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.raw[: obs.shape[0]]


class AcceptanceActor(nn.Module):
    def __init__(self, input_dim: int, logits: torch.Tensor):
        super().__init__()
        self.linear = nn.Linear(input_dim, logits.shape[-1], bias=True)
        nn.init.zeros_(self.linear.weight)
        with torch.no_grad():
            self.linear.bias.copy_(logits[0])
        self.last_input: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x
        return self.linear(x)


class FakePolicy:
    task_conf_dim = 6
    num_task_corrections = 6


class FakeAlgorithm:
    def __init__(self):
        self.device = torch.device("cpu")
        self.policy = FakePolicy()
        self.frontres_training_objective = "hsl_hybrid"
        self.frontres_acceptance_preference_weight = 1.0
        self.frontres_acceptance_preference_focal_gamma = 0.0
        self.frontres_acceptance_preference_balance_min = 1.0
        self.frontres_acceptance_preference_balance_max = 1.0
        self.frontres_active_task_dims = None
        self.frontres_structured_joint_rl_enabled = False
        self.frontres_structured_joint_rl_weight = 0.0
        self.frontres_structured_joint_rl_keep_legacy_bce = False
        self.frontres_authority_actor_critic_enabled = False
        self.frontres_authority_actor_loss_weight = 0.0
        self.frontres_authority_critic_loss_weight = 0.0
        self.ppo_actor_weight = 1.0

    def _structured_joint_rl_enabled(self) -> bool:
        return FrontRESUnified._structured_joint_rl_enabled(self)

    def _ppo_acceptance_only_mode(self) -> bool:
        return FrontRESUnified._ppo_acceptance_only_mode(self)

    def _authority_actor_critic_enabled(self) -> bool:
        return FrontRESUnified._authority_actor_critic_enabled(self)

    def _active_hsl_acceptance_loss_enabled(self) -> bool:
        return FrontRESUnified._active_hsl_acceptance_loss_enabled(self)




def _load_acceptance_label_module():
    spec = importlib.util.spec_from_file_location("frontres_acceptance_labels_under_step10", LABEL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def _load_frontres_policy_class():
    spec = importlib.util.spec_from_file_location("frontres_policy_under_step10", POLICY_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.FrontRESActorCritic


def _load_diagnostics_module():
    spec = importlib.util.spec_from_file_location("frontres_diagnostics_under_step10", DIAG_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _build_minimal_policy(obs_dim: int, proposal_raw: torch.Tensor, acceptance_logits: torch.Tensor):
    FrontRESActorCritic = _load_frontres_policy_class()
    policy = FrontRESActorCritic.__new__(FrontRESActorCritic)
    nn.Module.__init__(policy)
    policy.num_task_corrections = 6
    policy.task_conf_dim = 6
    policy.max_delta_pos = 0.3
    policy.max_delta_rpy = 0.4
    policy.frontres_split_acceptance_head = True
    policy.frontres_authority_actor_critic = False
    policy.frontres_state_router_enabled = False
    policy.residual_actor = ProposalOnlyActor(proposal_raw)
    policy.acceptance_actor = AcceptanceActor(input_dim=obs_dim + 6, logits=acceptance_logits)
    policy.authority_actor = None
    policy.authority_critic = None
    policy.state_router_head = None
    policy._cached_full_policy_obs = torch.arange(proposal_raw.shape[0] * obs_dim, dtype=torch.float32).view(
        proposal_raw.shape[0], obs_dim
    ) / 100.0
    return policy


def _make_transition(
    *,
    obs: torch.Tensor,
    actions: torch.Tensor,
    action_mean: torch.Tensor,
    proposal: torch.Tensor,
    acceptance_gt: torch.Tensor,
    acceptance_mask: torch.Tensor,
    acceptance_margin: torch.Tensor,
) -> RolloutStorage.Transition:
    n = obs.shape[0]
    transition = RolloutStorage.Transition()
    transition.observations = obs.detach().clone()
    transition.actions = actions.detach().clone()
    transition.rewards = torch.zeros(n)
    transition.dones = torch.zeros(n)
    transition.values = torch.zeros(n, 1)
    transition.actions_log_prob = torch.zeros(n)
    transition.action_mean = action_mean.detach().clone()
    transition.action_sigma = torch.ones_like(actions)
    transition.frontres_mask = torch.ones(n, 1)
    transition.frontres_actor_gate = torch.ones(n, 1)
    transition.supervised_target = proposal.detach().clone()
    transition.supervised_weight = torch.ones(n, 1)
    transition.supervised_harm_weight = torch.zeros(n, 1)
    transition.acceptance_gt = acceptance_gt.detach().clone()
    transition.acceptance_mask = acceptance_mask.detach().clone()
    transition.acceptance_margin = acceptance_margin.detach().clone()
    transition.state_alpha_target = torch.zeros(n, 1)
    transition.state_alpha_mask = torch.zeros(n, 1)
    return transition


def main() -> None:
    torch.manual_seed(7)
    n = 4
    obs_dim = 10
    obs = torch.zeros(n, obs_dim)
    raw_proposal = torch.tensor(
        [
            [0.10, -0.05, 0.02, 0.20, -0.10, 0.05],
            [0.08, 0.04, -0.01, -0.15, 0.12, -0.04],
            [0.20, -0.02, 0.00, 0.05, 0.05, 0.05],
            [0.01, 0.02, 0.03, 0.01, 0.02, 0.03],
        ],
        dtype=torch.float32,
    )
    acceptance_logits = torch.tensor(
        [[-1.5, -0.8, -0.2, 0.2, 0.8, 1.5]],
        dtype=torch.float32,
    )
    policy = _build_minimal_policy(obs_dim, raw_proposal, acceptance_logits)

    raw_policy_output = policy._frontres_raw_task_output(obs)
    if tuple(raw_policy_output.shape) != (n, 12):
        raise AssertionError(f"policy output shape mismatch: {tuple(raw_policy_output.shape)}")
    proposal = raw_policy_output[:, :6]
    acceptance_logit = raw_policy_output[:, 6:12]
    acceptance_prob = torch.sigmoid(acceptance_logit)
    if policy.acceptance_actor.last_input is None:
        raise AssertionError("acceptance head was not called")
    if policy.acceptance_actor.last_input[:, 10:].requires_grad:
        raise AssertionError("acceptance head received proposal with live proposal gradient")

    noisy_score = torch.tensor([0.10, 0.30, 0.20, 0.40], dtype=torch.float32)
    candidate_score = torch.tensor([0.35, 0.05, 0.205, 0.18], dtype=torch.float32)
    label_module = _load_acceptance_label_module()
    labels = label_module.build_frontres_acceptance_labels(
        candidate_score=candidate_score,
        noisy_score=noisy_score,
        positive_margin=0.01,
        negative_margin=0.01,
    )
    acceptance_gt, acceptance_mask = label_module.expand_acceptance_labels_to_task_dims(labels, task_dim=6)
    expected_gt = torch.tensor([1.0, 0.0, 0.0, 0.0]).view(n, 1).expand(n, 6)
    expected_sample_mask = torch.tensor([1.0, 1.0, 0.0, 1.0]).view(n, 1).expand(n, 6)
    torch.testing.assert_close(acceptance_gt, expected_gt)
    torch.testing.assert_close(acceptance_mask, expected_sample_mask)

    actions = torch.cat([proposal.detach(), acceptance_prob.detach()], dim=-1)
    action_mean = torch.cat([proposal.detach(), acceptance_logit], dim=-1)
    storage = RolloutStorage(
        training_type="frontres",
        num_envs=n,
        num_transitions_per_env=1,
        obs_shape=(obs_dim,),
        privileged_obs_shape=None,
        actions_shape=(12,),
        device="cpu",
    )
    storage.yield_batch_indices = True
    transition = _make_transition(
        obs=obs,
        actions=actions,
        action_mean=action_mean,
        proposal=proposal,
        acceptance_gt=acceptance_gt,
        acceptance_mask=acceptance_mask,
        acceptance_margin=labels.margin,
    )
    storage.add_transitions(transition)
    torch.testing.assert_close(storage.proposal_delta_se[0], proposal.detach())
    torch.testing.assert_close(storage.acceptance_gt[0], acceptance_gt)
    torch.testing.assert_close(storage.acceptance_mask[0], acceptance_mask)
    torch.testing.assert_close(storage.acceptance_margin[0], labels.margin)
    if float(storage.authority_mask.abs().sum().item()) != 0.0:
        raise AssertionError("authority storage became active in HSL+acceptance toy chain")

    batch = next(storage.mini_batch_generator(num_mini_batches=1, num_epochs=1))
    batch_indices = batch[41]
    mu_batch = storage.mu.flatten(0, 1)[batch_indices].detach().clone().requires_grad_(True)
    gt_batch = batch[39]
    mask_batch = batch[23]
    margin_batch = batch[40]
    torch.testing.assert_close(gt_batch, storage.acceptance_gt.flatten(0, 1)[batch_indices])
    torch.testing.assert_close(mask_batch, storage.acceptance_mask.flatten(0, 1)[batch_indices])
    torch.testing.assert_close(margin_batch, storage.acceptance_margin.flatten(0, 1)[batch_indices])
    alg = FakeAlgorithm()
    loss, metrics = FrontRESUnified._compute_acceptance_preference_loss(
        alg,
        mu_batch,
        gt_batch,
        mask_batch,
        original_batch_size=n,
    )
    if metrics["hsl_acceptance_loss_enabled"] != 1.0:
        raise AssertionError("active HSL acceptance loss did not run")
    for key in ("hsl_acceptance_mask_frac", "hsl_acceptance_gt_mean", "hsl_acceptance_prob_mean", "hsl_acceptance_abs_err"):
        value = float(metrics[key])
        if not 0.0 <= value <= 1.0:
            raise AssertionError(f"{key} out of probability range: {value}")
    if alg._structured_joint_rl_enabled() or alg._authority_actor_critic_enabled():
        raise AssertionError("legacy structured-rho or authority branch became active")
    loss.backward()
    if float(mu_batch.grad[:, :6].abs().sum().item()) != 0.0:
        raise AssertionError("acceptance loss leaked gradient into proposal columns")
    if float(mu_batch.grad[:, 6:12].abs().sum().item()) <= 0.0:
        raise AssertionError("acceptance loss produced no acceptance gradient")

    diag = _load_diagnostics_module()
    loss_dict = dict(metrics)
    loss_dict.update(
        {
            "acceptance_preference_loss": float(loss.detach().item()),
            "lambda_acceptance_preference": 1.0,
            "frontres_accept_pos_mean": float(acceptance_prob[:, :3].mean().detach().item()),
            "frontres_accept_rpy_mean": float(acceptance_prob[:, 3:].mean().detach().item()),
            "frontres_accept_active_frac": float((acceptance_mask.sum(dim=-1) > 0).float().mean().detach().item()),
            "frontres_proposal_ratio": float(proposal.norm(dim=-1).mean().detach().item()),
            "frontres_axis_leakage": 0.0,
        }
    )
    locs = {
        "frontres_accept_pref_mask_mean": float((acceptance_mask.sum(dim=-1) > 0).float().mean().item()),
        "frontres_accept_pref_full_mean": float((labels.accept_gt[:, 0] > 0.5).float().mean().item()),
        "frontres_accept_pref_noop_mean": float(((labels.accept_gt[:, 0] < 0.5) & (labels.accept_mask[:, 0] > 0.5)).float().mean().item()),
        "frontres_accept_pref_ignore_mean": float((labels.accept_mask[:, 0] <= 0.5).float().mean().item()),
        "frontres_accept_pref_margin_mean": float(labels.margin.mean().item()),
    }
    cfg = {
        "frontres_training_objective": "hsl_hybrid",
        "frontres_authority_actor_critic_enabled": False,
        "frontres_structured_joint_rl_enabled": False,
        "frontres_structured_joint_rl_weight": 0.0,
    }
    text = diag.format_frontres_hsl_acceptance_diagnostics(locs, loss_dict, cfg, pad=28)
    if "acceptance loss" not in text or "accept labels" not in text or "accept prob" not in text:
        raise AssertionError(f"active acceptance diagnostics missing expected lines:\n{text}")
    forbidden = ("authority", "rho", "alpha", "structured")
    if any(word in text.lower() for word in forbidden):
        raise AssertionError(f"legacy diagnostic leaked into active HSL acceptance diagnostics:\n{text}")

    print("=== FrontRES HSL+Acceptance Full Toy Chain TEST ONLY ===")
    print("policy: proposal6 + acceptance6, acceptance sees detached proposal")
    print(
        "labels: "
        f"accept={float(labels.accept_frac):.3f}, reject={float(labels.reject_frac):.3f}, "
        f"ignore={float(labels.ignore_frac):.3f}, margin={float(labels.margin_mean):+.4f}"
    )
    print(
        "loss: "
        f"{float(loss.detach()):.6f}, mask={metrics['hsl_acceptance_mask_frac']:.3f}, "
        f"gt={metrics['hsl_acceptance_gt_mean']:.3f}, prob={metrics['hsl_acceptance_prob_mean']:.3f}"
    )
    print("diagnostics:")
    print(text.rstrip())
    print("result: PASS")


if __name__ == "__main__":
    main()
