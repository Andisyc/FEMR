# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import os
import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class FrontEndResidualActorCritic(nn.Module):
    """
    Actor-Critic module for Stage 2 RL Finetuning.
    It chains a trainable FrontRES network with a frozen GMT policy.
    """
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        residual_hidden_dims=[1024, 1024, 512, 256],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        gmt_checkpoint_path: str = "",
        q_ref_start_idx: int = 0,
        **kwargs,
    ):
        super().__init__()

        if kwargs:
            print(f"{self.__class__.__name__} got unexpected arguments: {list(kwargs.keys())}")

        activation_fn = resolve_nn_activation(activation)
        self.q_ref_start_idx = q_ref_start_idx
        self.num_actions = num_actions

        # --- 1. FrontRES Network (Trainable) ---
        # This is the 'actor' part that we will be finetuning.
        front_res_layers = []
        front_res_layers.append(nn.Linear(num_actor_obs, residual_hidden_dims[0]))
        front_res_layers.append(activation_fn)
        for i in range(len(residual_hidden_dims) - 1):
            front_res_layers.append(nn.Linear(residual_hidden_dims[i], residual_hidden_dims[i+1]))
            front_res_layers.append(activation_fn)
        front_res_layers.append(nn.Linear(residual_hidden_dims[-1], num_actions))
        self.actor = nn.Sequential(*front_res_layers)
        print(f"FrontRES (Trainable Actor): {self.actor}")

        # --- 2. GMT Policy (Frozen) ---
        if not os.path.exists(gmt_checkpoint_path):
            raise FileNotFoundError(f"GMT checkpoint not found at: {gmt_checkpoint_path}")
        
        print(f"Loading frozen GMT policy from: {gmt_checkpoint_path}")
        gmt_checkpoint = torch.load(gmt_checkpoint_path)
        
        # We need to reconstruct the GMT ActorCritic to load weights
        # Assuming it's a standard ActorCritic for this example
        from .actor_critic import ActorCritic
        gmt_policy_cfg = gmt_checkpoint.get("policy_cfg", {})
        gmt_policy = ActorCritic(
            num_actor_obs=num_actor_obs, # GMT obs dim should match
            num_critic_obs=1, # Dummy value, not used
            num_actions=num_actions,
            **gmt_policy_cfg
        )
        gmt_policy.load_state_dict(gmt_checkpoint['model_state_dict'])
        self.gmt = copy.deepcopy(gmt_policy.actor)
        
        # Freeze GMT parameters
        for param in self.gmt.parameters():
            param.requires_grad = False
        self.gmt.eval()
        print(f"GMT (Frozen): {self.gmt}")

        # --- 3. Critic Network (Trainable) ---
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for i in range(len(critic_hidden_dims) - 1):
            critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i+1]))
            critic_layers.append(activation_fn)
        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.critic = nn.Sequential(*critic_layers)
        print(f"Critic: {self.critic}")

        # --- Action Noise ---
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    def forward(self, observations, privileged_observations):
        """Defines the forward pass for the entire Actor-Critic module."""
        # This is the full pipeline for training (Actor and Critic)
        actions_mean = self.get_actions(observations)
        value = self.critic(privileged_observations)
        return actions_mean, value

    def get_actions(self, observations):
        """Computes the mean of the action distribution."""
        # 1. FrontRES computes Δq
        delta_q = self.actor(observations)

        # 2. Extract q_ref and construct GMT input
        q_ref = observations[:, self.q_ref_start_idx : self.q_ref_start_idx + self.num_actions]
        q_corrected = q_ref + delta_q
        
        gmt_input_obs = observations.clone()
        gmt_input_obs[:, self.q_ref_start_idx : self.q_ref_start_idx + self.num_actions] = q_corrected

        # 3. Frozen GMT computes the final action
        with torch.no_grad():
            action_mean = self.gmt(gmt_input_obs)
            
        return action_mean

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations, **kwargs):
        return self.get_actions(observations)

    def evaluate(self, privileged_observations, **kwargs):
        return self.critic(privileged_observations)

    def update_distribution(self, observations):
        mean = self.get_actions(observations)
        self.distribution = Normal(mean, self.std.expand_as(mean))

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)