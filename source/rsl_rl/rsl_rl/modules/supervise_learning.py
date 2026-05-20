# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# =======================================================================================
# 思路与逻辑梳理 (注释) - legacy 监督学习 helper
#
# 核心架构:
#   1. GMT (.pt checkpoint): 一个 frozen tracker，根据参考运动观测生成动作。
#   2. student (FrontRES): 一个前端网络。legacy joint-space 路径学习 Δq/Δz；
#      task-space 路径学习 ΔSE(3) 监督目标。
#
# 当前主线 FrontRESUnified 的 warmup 在 runner 内实现；本模块保留给旧的
# supervise training_type 和工具脚本使用，不能作为 task-space runner 的 obs
# layout 或 pre-GMT application 语义来源。
# =======================================================================================

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules import ActorCritic, EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation


class SuperviseLearning(nn.Module):
    """
    Legacy supervised FrontRES module for Δq/Δz or task-space target prediction.
    """
    is_recurrent = False
    is_encoding = False  # 适配 OnPolicyRunner 接口

    def __init__(
        self,
        num_actor_obs,          # 对应工厂模式传入的 policy 观测维度
        num_critic_obs,         # 对应工厂模式传入的 critic 观测维度
        num_actions,            # Δq dim (= robot joint DOFs, e.g. 29 for G1)
        student_hidden_dims=[256, 256, 256],
        activation="elu",
        gmt_path: str = None,           # Path to GMT .pt checkpoint
        num_z_outputs: int = 0,         # Additional Δz outputs appended after Δq; 1 → output is [Δq(29), Δz(1)]
        num_task_corrections: int = 0,  # Task-space output dim; when >0, replaces Δq+Δz with [Δpos(3), Δrpy(3)]
        **kwargs,
    ):
        """
        Args:
            num_actor_obs (int): input dim of FrontRES
            num_critic_obs (int): unused
            num_actions (int): number of joint Δq outputs (= robot DOFs)
            gmt_path (str): path to frozen GMT .pt checkpoint
            num_z_outputs (int): number of auxiliary z-correction outputs (0 = legacy behaviour)
        """
        if kwargs:
            print(
                "SuperviseLearning.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation_name = activation          # keep original string for ActorCritic
        activation = resolve_nn_activation(activation)

        # ========== GMT Tracker (专家模型, .pt checkpoint) ==========
        # Uses the same loading logic as FrontRESActorCritic (Stage 2) so the
        # architecture is always inferred correctly from the checkpoint itself.
        self.gmt_policy: ActorCritic | None = None
        self.gmt_normalizer: EmpiricalNormalization | None = None
        if gmt_path:
            print(f"[SuperviseLearning] Loading GMT policy from: {gmt_path}")
            checkpoint = torch.load(gmt_path, map_location="cpu", weights_only=False)
            sd = checkpoint["model_state_dict"]

            # ---- infer architecture ----
            has_skip = "actor.actor_layer1.weight" in sd
            if has_skip:
                layer1_in  = sd["actor.actor_layer1.weight"].shape[1]
                layer1_out = sd["actor.actor_layer1.weight"].shape[0]
                rem0_in    = sd["actor.actor_remaining.0.weight"].shape[1]
                ref_vel_dim = rem0_in - layer1_out
                gmt_actor_in  = layer1_in + ref_vel_dim
                gmt_critic_in = sd["critic.0.weight"].shape[1]
                rem_keys = [k for k in sd if k.startswith("actor.actor_remaining.") and k.endswith(".weight")]
                last_key = max(rem_keys, key=lambda k: int(k.split(".")[2]))
                gmt_num_actions = sd[last_key].shape[0]
                extra_cfg: dict = {"ref_vel_skip_first_layer": True, "ref_vel_dim": ref_vel_dim}
            else:
                gmt_actor_in  = sd["actor.0.weight"].shape[1]
                gmt_critic_in = sd["critic.0.weight"].shape[1]
                act_keys = [k for k in sd if k.startswith("actor.") and k.endswith(".weight")]
                last_key = max(act_keys, key=lambda k: int(k.split(".")[1]))
                gmt_num_actions = sd[last_key].shape[0]
                extra_cfg = {}

            # hidden dims (all layers except the last output layer)
            if has_skip:
                actor_weight_keys = sorted(
                    [k for k in sd if k.startswith("actor.actor_remaining.") and k.endswith(".weight")],
                    key=lambda k: int(k.split(".")[2]))
            else:
                actor_weight_keys = sorted(
                    [k for k in sd if k.startswith("actor.") and k.endswith(".weight")],
                    key=lambda k: int(k.split(".")[1]))
            actor_hidden_dims = [sd[k].shape[0] for k in actor_weight_keys[:-1]]

            critic_weight_keys = sorted(
                [k for k in sd if k.startswith("critic.") and k.endswith(".weight")],
                key=lambda k: int(k.split(".")[1]))
            critic_hidden_dims = [sd[k].shape[0] for k in critic_weight_keys[:-1]]

            noise_std_type = "scalar" if "std" in sd else "log"
            init_noise_std = (sd["std"][0].item() if "std" in sd
                              else torch.exp(sd["log_std"][0]).item())

            self.gmt_policy = ActorCritic(
                num_actor_obs=gmt_actor_in,
                num_critic_obs=gmt_critic_in,
                num_actions=gmt_num_actions,
                actor_hidden_dims=actor_hidden_dims,
                critic_hidden_dims=critic_hidden_dims,
                activation=activation_name,
                init_noise_std=init_noise_std,
                noise_std_type=noise_std_type,
                **extra_cfg,
            )
            self.gmt_policy.load_state_dict(sd)
            self.gmt_policy.eval()
            for p in self.gmt_policy.parameters():
                p.requires_grad = False
            print(f"[SuperviseLearning] GMT policy loaded and frozen "
                  f"(actor_in={gmt_actor_in}, actions={gmt_num_actions})")

            # ---- load frozen obs normalizer ----
            if "obs_norm_state_dict" in checkpoint:
                obs_norm_sd = checkpoint["obs_norm_state_dict"]
                norm_dim = obs_norm_sd["_mean"].shape[1]
                self.gmt_normalizer = EmpiricalNormalization(shape=[norm_dim], until=1.0e8)
                self.gmt_normalizer.load_state_dict(obs_norm_sd)
                self.gmt_normalizer.eval()
                self.gmt_normalizer.until = 0  # freeze statistics
                print(f"[SuperviseLearning] GMT obs normalizer loaded and frozen (dim={norm_dim})")
            else:
                print("[SuperviseLearning] WARNING: no obs_norm_state_dict in GMT checkpoint")

        # ========== student (FrontRES 网络) ==========
        self.num_actions = num_actions
        self.num_z_outputs = num_z_outputs
        self.num_task_corrections = num_task_corrections
        # Legacy task-space supervised mode: output = [Δx, Δy, Δz, Δroll, Δpitch, Δyaw].
        # FrontRESActorCritic PPO mode adds confidence heads separately.
        if num_task_corrections > 0:
            total_output_dim = num_task_corrections
        else:
            total_output_dim = num_actions + num_z_outputs  # e.g. 29 + 1 = 30

        student_layers = []
        student_layers.append(nn.Linear(num_actor_obs, student_hidden_dims[0]))
        student_layers.append(activation)
        for layer_index in range(len(student_hidden_dims)):
            if layer_index == len(student_hidden_dims) - 1:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], total_output_dim))
            else:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], student_hidden_dims[layer_index + 1]))
                student_layers.append(activation)
        self.student = nn.Sequential(*student_layers)

        if num_task_corrections > 0:
            print(f"[SuperviseLearning] Student MLP output: {num_task_corrections} task-space dims [Δpos(3)+Δrpy(3)]")
        else:
            print(f"[SuperviseLearning] Student MLP output: {num_actions} Δq + {num_z_outputs} Δz = {total_output_dim} dims")
        print(f"[SuperviseLearning] Student MLP: {self.student}")

        # 临时存储分布状态 (兼容 rsl_rl 内部流程)
        self._student_pred = None

    def reset(self, dones=None, hidden_states=None):
        pass

    @property
    def action_mean(self):
        return self._student_pred

    @property
    def action_std(self):
        return torch.zeros_like(self._student_pred) if self._student_pred is not None else None

    @property
    def entropy(self):
        return torch.zeros_like(self._student_pred[:, 0]) if self._student_pred is not None else None

    def forward(self, observations):
        """
        Run the student and return the predicted residual target.
        Args:
            observations (torch.Tensor): student obs
        Returns:
            torch.Tensor: the predicted residual target
        """
        return self.student(observations)
        
    def update_distribution(self, observations):
        """updating interface to adapt to RL Runner"""
        self._student_pred = self.student(observations)

    def act(self, observations, **kwargs):
        """
        standard interface, invoke at Runner Rollout
        """
        self.update_distribution(observations)
        return self._student_pred

    def get_actions_log_prob(self, actions):
        """Provide a false Log Prob"""
        return torch.zeros_like(actions[:, 0])

    def act_inference(self, observations, **kwargs):
        return self.act(observations)
        
    def evaluate(self, critic_observations, **kwargs):
        """
        Provide a false Critic evaluation interface
        """
        return torch.zeros((critic_observations.shape[0], 1), device=critic_observations.device)

    @torch.no_grad()
    def get_gmt_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Run GMT (PyTorch .pt) inference on a batch of raw observations.
        obs is normalised internally by gmt_normalizer before being fed to gmt_policy.

        Legacy supervised helper: if the student obs has more dims than the GMT
        normalizer expects, this path keeps the leading GMT-sized prefix.  The
        FrontRESUnified task-space runner uses its own partial-normalization path
        where anchor-error extras are leading dims and GMT obs is the suffix.
        """
        if self.gmt_policy is None:
            raise RuntimeError("GMT policy is not loaded. Cannot compute GMT action.")

        device = obs.device
        # Normalize with GMT's frozen normalizer (same as Stage 2)
        if self.gmt_normalizer is not None:
            _gmt_mean = getattr(self.gmt_normalizer, '_mean', None)
            if _gmt_mean is not None and obs.shape[-1] > _gmt_mean.shape[-1]:
                obs = obs[:, :_gmt_mean.shape[-1]]
            obs = self.gmt_normalizer(obs.to(self.gmt_normalizer._mean.device))
        return self.gmt_policy.act_inference(obs.to(device))

    @staticmethod
    def get_supervision_target(q_sim: torch.Tensor, q_ref: torch.Tensor) -> torch.Tensor:
        """
        Δ_q_gt = q_ref - q_sim
        """
        delta_q = q_ref - q_sim
        return delta_q

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the student network."""
        if any("actor" in key for key in state_dict.keys()):
            student_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    student_state_dict[key.replace("actor.", "")] = value
            self.student.load_state_dict(student_state_dict, strict=strict)
        else:
            super().load_state_dict(state_dict, strict=strict)
        return True

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass
