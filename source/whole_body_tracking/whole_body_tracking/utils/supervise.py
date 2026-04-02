# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# 

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import SuperviseLearning

class SuperviseStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.target_actions = None
            self.dones = None

        def clear(self):
            self.observations = None
            self.target_actions = None
            self.dones = None

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, action_shape, device):
        self.device = device
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.step = 0

        # 核心数据缓冲区
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.target_actions = torch.zeros(num_transitions_per_env, num_envs, *action_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

    def add_transitions(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow.")
        self.observations[self.step].copy_(transition.observations)
        self.target_actions[self.step].copy_(transition.target_actions)
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.step += 1

    def clear(self):
        self.step = 0

    def generator(self):
        for i in range(self.num_transitions_per_env):
            yield self.observations[i], self.target_actions[i], self.dones[i]

class SuperviseTrainer:
    """Supervised learning algorithm for training FrontRES to output delta_q."""

    def __init__(
        self,
        policy: SuperviseLearning,
        num_learning_epochs: int = 1,
        gradient_length: int = 15,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        loss_type: str = "mse",
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None, # Distributed training parameters
    ):
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # Policy (FrontRES Student)
        self.policy = policy
        self.policy.to(self.device)

        # Create the optimizer
        self.optimizer = optim.Adam(self.policy.student.parameters(), lr=learning_rate)

        # Storage
        self.storage = None
        self.transition = SuperviseStorage.Transition()
        self.last_hidden_states = None

        # Training parameters
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        # Initialize the loss function
        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "huber":
            self.loss_fn = nn.functional.huber_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        self.num_updates = 0

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        """initialize buffer"""
        self.storage = SuperviseStorage(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs_shape=actor_obs_shape,
            action_shape=action_shape,
            device=self.device
        )

    def act(self, obs: torch.Tensor, target_delta_q: torch.Tensor) -> torch.Tensor:
        """record delta_q_gt"""
        # Compute the actions
        actions = self.policy.act(obs).detach()
        
        # 使用标准的 transition 流水线记录当前步的数据
        self.transition.observations = obs
        self.transition.target_actions = target_delta_q
        
        return actions

    def process_env_step(self, rewards, dones, infos) -> None:
        """store data into buffer"""
        self.transition.dones = dones
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        pass

    def update(self) -> dict:
        """Run optimization epochs over stored batches and return mean losses."""
        self.num_updates += 1
        mean_behavior_loss = 0
        loss = 0
        cnt = 0

        # Accumulators for diagnostic metrics
        sum_pred_norm = 0.0
        sum_gt_norm = 0.0
        sum_cos_sim = 0.0
        sum_rel_err = 0.0
        sum_valid_ratio = 0.0
        sum_joint_mae = None  # will be (num_actions,) tensor

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()

            # 直接按时间步迭代我们内置的 Buffer
            for obs, target_actions, dones in self.storage.generator():
                # Inference of the FrontRES student
                predicted_actions = self.policy.forward(obs)

                # Mask out terminal transitions: when done=True the robot has fallen.
                # q_sim is from a fallen/reset state → Δq_gt = q_ref - q_sim_fallen is
                # anomalously large and noisy.  Training on it would corrupt FrontRES.
                # valid: (B, 1) float, 1 = valid transition, 0 = terminal (fallen)
                valid = 1.0 - dones.float()         # (B, 1)
                n_valid = valid.sum().clamp(min=1.0) # avoid divide-by-zero

                # Per-sample loss then weighted mean over valid samples only
                per_sample_loss = self.loss_fn(predicted_actions, target_actions, reduction="none")  # (B, A)
                behavior_loss = (per_sample_loss.mean(dim=-1, keepdim=True) * valid).sum() / n_valid

                # Total loss
                loss = loss + behavior_loss
                mean_behavior_loss += behavior_loss.item()
                cnt += 1

                # --- Diagnostic metrics (no grad needed) ---
                with torch.no_grad():
                    # Fraction of valid (non-terminal) samples in this batch
                    sum_valid_ratio += valid.mean().item()

                    pred_norm = predicted_actions.norm(dim=-1)   # (B,)
                    gt_norm   = target_actions.norm(dim=-1)       # (B,)

                    # L2 norm of predicted and ground-truth Δq (batch mean)
                    sum_pred_norm += pred_norm.mean().item()
                    sum_gt_norm   += gt_norm.mean().item()

                    # Cosine similarity between predicted and GT Δq (direction alignment)
                    cos_sim = nn.functional.cosine_similarity(predicted_actions, target_actions, dim=-1)
                    sum_cos_sim += cos_sim.mean().item()

                    # Relative error: ||pred - gt|| / ||gt||  (normalized accuracy)
                    err_norm = (predicted_actions - target_actions).norm(dim=-1)
                    rel_err  = err_norm / (gt_norm + 1e-8)
                    sum_rel_err += rel_err.mean().item()

                    # Per-joint absolute error, accumulated for mean across all batches
                    joint_mae = (predicted_actions - target_actions).abs().mean(dim=0)  # (num_actions,)
                    if sum_joint_mae is None:
                        sum_joint_mae = joint_mae
                    else:
                        sum_joint_mae = sum_joint_mae + joint_mae

                # Gradient step
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0

                # Reset dones
                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        mean_behavior_loss /= cnt
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        # --- Construct the loss dictionary ---
        assert sum_joint_mae is not None, "Storage was empty — no batches processed."
        mean_pred_norm = sum_pred_norm / cnt
        mean_gt_norm   = sum_gt_norm   / cnt
        mean_joint_mae = sum_joint_mae / cnt  # (num_actions,)
        loss_dict = {
            # --- Primary convergence signal ---
            "behavior":            mean_behavior_loss,           # MSE/Huber loss (main signal, should ↓)
            # --- Amplitude calibration ---
            "delta_q_pred_norm":   mean_pred_norm,               # predicted Δq L2 norm
            "delta_q_gt_norm":     mean_gt_norm,                 # GT Δq L2 norm (reference)
            "delta_q_norm_ratio":  mean_pred_norm / (mean_gt_norm + 1e-8),  # pred/gt ≈ 1.0 when calibrated
            # --- Direction alignment ---
            "cosine_similarity":   sum_cos_sim / cnt,            # cos(pred, gt) ∈ [-1,1], target → 1.0
            # --- Normalized accuracy ---
            "relative_error":      sum_rel_err / cnt,            # ||pred-gt||/||gt||, target < 0.1
            # --- Data quality ---
            "valid_ratio":         sum_valid_ratio / cnt,         # fraction of non-terminal steps (fall rate proxy)
            # --- Per-joint breakdown ---
            "joint_mae_mean":      mean_joint_mae.mean().item(), # mean absolute error across joints
            "joint_mae_max":       mean_joint_mae.max().item(),  # worst joint (upper bound on error)
        }

        return loss_dict
