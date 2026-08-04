# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Residual Learning Policy for Physical-Aware Motion Refiner.

The legacy joint-space path predicts Δq and patches q_ref before frozen GMT.
The current FrontRES task-space path predicts full-6D bounded ΔSE(3); the runner
applies that correction to the command/reference frame before refreshing observations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.frontres.frontres_formal_runtime_probe import emit_formal_runtime_probe
from rsl_rl.modules import ActorCritic, EmpiricalNormalization
from rsl_rl.modules.frontres_observation_layout import split_frontres_policy_obs
from rsl_rl.utils import resolve_nn_activation


def _validate_frontres_task_space_observation_authority(
    *,
    num_actor_obs: int,
    num_frontres_obs: int,
    gmt_policy_obs_dim: int,
) -> None:
    """Fail closed unless task-space FEMR and frozen GMT own disjoint observation slices."""

    if int(num_frontres_obs) <= 0:
        raise ValueError("task-space v015 rejects num_frontres_obs=0; full-observation FEMR fallback is forbidden")
    if int(num_frontres_obs) >= int(num_actor_obs):
        raise ValueError("task-space FEMR prefix must be smaller than the combined policy observation")
    if int(num_frontres_obs) + int(gmt_policy_obs_dim) != int(num_actor_obs):
        raise ValueError(
            "task-space FrontRES/GMT observation authority mismatch: "
            f"{num_frontres_obs}D FEMR + {gmt_policy_obs_dim}D GMT != {num_actor_obs}D combined"
        )


def _gmt_observation_route_messages(
    *,
    environment_obs_dim: int,
    gmt_policy_obs_dim: int,
    gmt_actor_input_dim: int,
    task_space_frontres: bool,
    has_gmt_normalizer: bool,
    has_ref_vel_estimator: bool,
) -> tuple[str, ...]:
    """Describe the effective FrontRES/GMT observation route without changing it."""

    messages: list[str] = []
    if environment_obs_dim > gmt_policy_obs_dim:
        prefix_dim = environment_obs_dim - gmt_policy_obs_dim
        if task_space_frontres and has_gmt_normalizer:
            messages.append(
                f"[FrontRESActorCritic] Observation layout: {environment_obs_dim}D = "
                f"{prefix_dim}D FrontRES-only prefix + {gmt_policy_obs_dim}D GMT-compatible suffix. "
                "GMT consumes the suffix only; no zero padding."
            )
        else:
            messages.append(
                f"[ResidualActorCritic] WARNING: observation provides {environment_obs_dim}D but GMT policy "
                f"expects {gmt_policy_obs_dim}D; no verified suffix-slicing contract is available."
            )
    elif environment_obs_dim < gmt_policy_obs_dim:
        missing_dim = gmt_policy_obs_dim - environment_obs_dim
        messages.append(
            f"[ResidualActorCritic] WARNING: GMT policy observation requires {gmt_policy_obs_dim}D but the "
            f"environment provides {environment_obs_dim}D; the missing {missing_dim}D will be zero padded."
        )

    ref_vel_dim = max(0, gmt_actor_input_dim - gmt_policy_obs_dim)
    if ref_vel_dim > 0:
        if has_ref_vel_estimator:
            messages.append(
                f"[ResidualActorCritic] GMT ref-velocity suffix: {ref_vel_dim}D supplied by the frozen estimator."
            )
        else:
            messages.append(
                f"[ResidualActorCritic] WARNING: GMT expects a {ref_vel_dim}D ref-velocity suffix; "
                "if the caller does not provide it, that suffix will be zero padded."
            )
    return tuple(messages)


class ComposedActor(nn.Module):
    """
    Legacy composed actor for ONNX export: combines frozen GMT + trainable FrontRES.

    This wrapper is for the joint-space Δq path:
        obs → FrontRES → [Δq (num_actions), Δz (num_z_outputs)]
        q_ref_corrected = q_ref + Δq   (modify q_ref inside obs)
        obs_modified → GMT → actions

    Task-space ΔSE(3) FrontRES is applied by the runner/env command term and
    should not be inferred from this ONNX wrapper.
    """
    def __init__(self, gmt_policy: ActorCritic, residual_actor: nn.Module,
                 gmt_actor_input_dim: int, num_actor_obs: int,
                 q_ref_start_idx: int, num_actions: int, num_z_outputs: int = 0):
        super().__init__()
        self.gmt_policy = gmt_policy
        self.residual_actor = residual_actor
        self.gmt_actor_input_dim = gmt_actor_input_dim
        self.num_actor_obs = num_actor_obs
        self.q_ref_start_idx = q_ref_start_idx
        self.num_actions = num_actions
        self.num_z_outputs = num_z_outputs

    def forward(self, observations):
        """FrontRES pre-GMT pipeline: correct q_ref, then run GMT."""
        obs_dim = observations.shape[-1]

        if obs_dim == self.num_actor_obs:
            policy_obs = observations
        elif obs_dim == self.gmt_actor_input_dim:
            policy_obs = observations[:, :self.num_actor_obs]
        else:
            raise ValueError(
                f"Unexpected observation dimension: {obs_dim}. "
                f"Expected {self.num_actor_obs} or {self.gmt_actor_input_dim}")

        # 1. FrontRES computes [Δq, Δz] from policy observations
        frontres_out = self.residual_actor(policy_obs)
        delta_q = frontres_out[:, :self.num_actions]    # (B, 29) joint corrections
        # Δz not applied inside ComposedActor/ONNX — needs env-side z correction hook

        # 2. Apply Δq to q_ref inside the observation vector
        obs_modified = policy_obs.clone()
        q_ref_end_idx = self.q_ref_start_idx + self.num_actions
        obs_modified[:, self.q_ref_start_idx:q_ref_end_idx] = (
            obs_modified[:, self.q_ref_start_idx:q_ref_end_idx] + delta_q
        )

        # 3. Build GMT observation (handle ref_vel suffix or padding if needed)
        if self.gmt_actor_input_dim > self.num_actor_obs:
            # GMT expects more dims than policy_obs.
            # If the original input already contained the ref_vel suffix, restore it;
            # otherwise pad with zeros (ONNX-safe: avoids empty-tensor torch.cat).
            ref_vel_dim = self.gmt_actor_input_dim - self.num_actor_obs
            if obs_dim == self.gmt_actor_input_dim:
                # Caller provided policy_obs + ref_vel concatenated
                ref_vel = observations[:, self.num_actor_obs:]
            else:
                # No ref_vel available — pad with zeros
                ref_vel = torch.zeros(
                    observations.shape[0], ref_vel_dim,
                    device=observations.device,
                    dtype=observations.dtype)
            gmt_obs = torch.cat([obs_modified, ref_vel], dim=-1)
        else:
            # GMT input dim == policy_obs dim (our standard setup: both 770)
            gmt_obs = obs_modified

        # 4. GMT forward (frozen)
        with torch.no_grad():
            actions = self.gmt_policy.act_inference(gmt_obs)

        return actions

    def __getitem__(self, idx):
        """Support subscript access for ONNX exporter (e.g., actor[0].in_features)"""
        return self.residual_actor[idx]


class FrontRESActorCritic(nn.Module):
    """
    Residual learning policy: frozen GMT + trainable residual network.

    Components:
    - residual_actor: Trainable residual network
    - critic: Trainable value function
    - gmt_policy: Frozen teacher policy (loaded from checkpoint)
    - gmt_normalizer: Frozen observation normalizer from GMT checkpoint

    In joint-space mode the residual actor outputs Δq.  In task-space mode it
    outputs full-6D [Δpos, Δrpy]; the command/reference correction is
    applied outside this module by the runner.
    """

    is_recurrent = False
    is_encoding = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        # Residual network configuration
        residual_hidden_dims=[512, 256, 128],
        residual_last_layer_gain=0.01,
        # GMT configuration
        q_ref_start_idx=0, # Added: Index where q_ref begins in the observation vector
        gmt_checkpoint_path=None,
        gmt_policy_cfg=None,  # Optional: specify GMT architecture (auto-inferred if None)
        # Ref vel estimator configuration
        num_ref_vel_estimator_obs=None,  # Dimension of ref_vel_estimator observations (e.g., 305)
        ref_vel_estimator_checkpoint_path=None,  # Path to estimator checkpoint
        ref_vel_estimator_type="mlp",  # Type of estimator: "mlp" or "transformer"
        # Critic configuration
        critic_hidden_dims=[1024, 1024, 512, 256],
        init_critic_from_gmt: bool = False,
        # Standard ActorCritic parameters
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        # Output clipping for Δq: tanh(raw) * max_delta_q bounds each joint correction.
        # Default 0.5 rad ≈ ±28.6°; set to float('inf') to disable.
        max_delta_q: float = 0.5,
        # Additional root z-correction outputs appended after Δq.
        # 0 = legacy behaviour (Δq only); 1 = [Δq (num_actions), Δz (1)].
        # When > 0, residual_actor output dim = num_actions + num_z_outputs.
        num_z_outputs: int = 0,
        # Output clipping for Δz: tanh(raw) * max_delta_z bounds root z correction.
        # 0.3 m covers typical float/sink artifacts (AMASS→G1 conversion errors).
        max_delta_z: float = 0.3,
        # Task-space mode: when >0, replaces Δq+Δz with [Δpos(3), Δrpy(3)]
        # Δq patching is disabled in task-space mode.
        num_task_corrections: int = 0,
        # FrontRES-specific observation subset: when >0, FrontRES only processes the
        # first num_frontres_obs dims of policy_obs (reference-frame data only).
        # GMT continues to receive the full policy_obs. 0 = legacy (full obs for both).
        num_frontres_obs: int = 0,
        **kwargs,
    ):
        retired_task_bounds = {name: kwargs.pop(name) for name in ("max_delta_pos", "max_delta_rpy") if name in kwargs}
        if retired_task_bounds:
            raise ValueError(
                "FRS-TRAIN-v014 rejects retired task-space action bounds: "
                f"{tuple(sorted(retired_task_bounds))}"
            )
        legacy_actor_hidden_dims = kwargs.pop("actor_hidden_dims", None)
        if not residual_hidden_dims and legacy_actor_hidden_dims:
            residual_hidden_dims = list(legacy_actor_hidden_dims)
        if kwargs:
            print(
                "ResidualActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        if gmt_checkpoint_path is None:
            raise ValueError("gmt_checkpoint_path is required for ResidualActorCritic")

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions          # = robot joint DOFs = GMT output dim (e.g. 29)
        self.num_z_outputs = num_z_outputs      # extra Δz outputs (0 = legacy)
        self.num_task_corrections = num_task_corrections  # task-space mode dim (0 = disabled)
        self.total_output_dim = (
            num_task_corrections if num_task_corrections > 0 else num_actions + num_z_outputs
        )
        # FrontRES observation subset: when >0, residual_actor only sees first N dims
        # (reference-frame data). GMT always sees the full observation.
        self.num_frontres_obs = num_frontres_obs
        self.q_ref_start_idx = q_ref_start_idx
        self.noise_std_type = noise_std_type
        self.max_delta_q = max_delta_q          # tanh clip for Δq (rad)
        self.max_delta_z = max_delta_z          # tanh clip for Δz (m)

        activation_fn = resolve_nn_activation(activation)

        # ========== Load GMT Policy ==========

        print(f"[ResidualActorCritic] Loading GMT policy from: {gmt_checkpoint_path}")
        checkpoint = torch.load(gmt_checkpoint_path, map_location="cpu", weights_only=False) # 导入checkpoint

        # Infer GMT architecture from checkpoint
        state_dict = checkpoint["model_state_dict"] # 导入权重

        # Detect checkpoint format: standard or ref_vel skip connection
        has_skip_connection = "actor.actor_layer1.weight" in state_dict # 确定有跳连接的布尔变量

        if has_skip_connection:
            # Skip connection format: actor.actor_layer1, actor.actor_remaining.X
            print(f"[ResidualActorCritic] Detected ref_vel skip connection format in GMT checkpoint")

            # IMPORTANT: Layer1 input dimension tells us the ACTUAL policy_obs_dim used during training
            # This might differ from expected due to bugs or different configurations
            layer1_input_dim = state_dict["actor.actor_layer1.weight"].shape[1] # 设置第一层的输入维度 (观测向量的维度)
            gmt_critic_input_dim = state_dict["critic.0.weight"].shape[1] # 设置GMT的Critic第一层输入维度

            # Infer ref_vel_dim from the second layer input size difference 通过两层维度反推参考速度向量维度
            layer1_output = state_dict["actor.actor_layer1.weight"].shape[0] # 设置第一层的输出维度
            remaining_0_input = state_dict["actor.actor_remaining.0.weight"].shape[1] # 设置第二层输入维度

            # 参考速度维度是第二层输入维度减去第一层输出维度 (第二层输入时拼接了参考速度向量)
            # 在第一层结束才输入ref_vel是因为观测量太高维, 直接输入会导致信息淹没, 但ref_vel
            # 相比其他信息更重要 (代表了往哪走), 因此将第一层作为Encoder, 压缩信息后才进行拼接
            ref_vel_dim = remaining_0_input - layer1_output

            # Calculate gmt_actor_input_dim: layer1_input + ref_vel_dim
            # This is the total observation dimension expected by GMT policy
            gmt_actor_input_dim = layer1_input_dim + ref_vel_dim # 在IsaacLab中注册时需要总观测维度

            # Find the last actor layer in actor_remaining 使用remaining的命名方式是因为ref_vel在第二层才输入, 导致梯度截断, 因此特意申明
            # 寻找最后一层维度 (输出动作维度), 如果无法找到, 就使用输入层的维度
            actor_remaining_keys = [k for k in state_dict.keys() if k.startswith("actor.actor_remaining.") and ".weight" in k]
            if actor_remaining_keys:
                last_actor_key = max(actor_remaining_keys, key=lambda k: int(k.split(".")[2]))
                gmt_num_actions = state_dict[last_actor_key].shape[0]
            else:
                # Fallback: use actor_layer1 output as action dim (shouldn't happen)
                gmt_num_actions = state_dict["actor.actor_layer1.weight"].shape[0]
        else:
            # Standard format: actor.0, actor.2, ...
            gmt_actor_input_dim = state_dict["actor.0.weight"].shape[1]
            gmt_critic_input_dim = state_dict["critic.0.weight"].shape[1]

            # Find the last actor layer
            actor_keys = [k for k in state_dict.keys() if k.startswith("actor.") and ".weight" in k]
            last_actor_key = max(actor_keys, key=lambda k: int(k.split(".")[1]))
            gmt_num_actions = state_dict[last_actor_key].shape[0]

        if gmt_num_actions != num_actions:
            raise ValueError(
                f"GMT action dimension ({gmt_num_actions}) does not match "
                f"specified num_actions ({num_actions})")

        print(f"[ResidualActorCritic] GMT architecture: "
              f"actor_input={gmt_actor_input_dim}, "
              f"critic_input={gmt_critic_input_dim}, "
              f"actions={gmt_num_actions}")

        # Create GMT policy with correct dimensions
        if gmt_policy_cfg is None:
            # Auto-infer architecture from checkpoint
            gmt_policy_cfg = self._infer_gmt_architecture(state_dict, activation)

            # If skip connection format detected, add skip connection config
            if has_skip_connection:
                print("[ResidualActorCritic] GMT uses ref_vel skip connection, creating matching architecture")
                print(f"[ResidualActorCritic] Inferred ref_vel_dim={ref_vel_dim}")
                print(f"[ResidualActorCritic] Layer1 accepts {layer1_input_dim} dims (policy_obs)")
                print(f"[ResidualActorCritic] Setting gmt_actor_input_dim={gmt_actor_input_dim} (layer1_input + ref_vel_dim)")

                gmt_policy_cfg["ref_vel_skip_first_layer"] = True
                gmt_policy_cfg["ref_vel_dim"] = ref_vel_dim

        # GMT实例化: 依据架构创建实例
        self.gmt_policy = ActorCritic(
            num_actor_obs=gmt_actor_input_dim,
            num_critic_obs=gmt_critic_input_dim,
            num_actions=gmt_num_actions,
            **gmt_policy_cfg)

        # Load GMT weights directly (no conversion needed if architectures match)
        self.gmt_policy.load_state_dict(state_dict) # 导入GMT权重

        # Freeze GMT completely
        self.gmt_policy.eval() # GMT设为eval模式
        for param in self.gmt_policy.parameters(): # 梯度置为False
            param.requires_grad = False
        print("[ResidualActorCritic] GMT policy frozen (all parameters require_grad=False)")

        # Load GMT's observation normalizer (critical!)
        self.gmt_normalizer = None # 导入GMT观测量归一器
        gmt_policy_obs_dim = gmt_actor_input_dim
        if "obs_norm_state_dict" in checkpoint:
            # Infer normalizer dimension from checkpoint (usually policy_obs_dim, not gmt_actor_input_dim)
            # This is because normalizer operates on policy_obs before ref_vel is concatenated
            obs_norm_state = checkpoint["obs_norm_state_dict"]
            normalizer_dim = obs_norm_state["_mean"].shape[1]
            gmt_policy_obs_dim = int(normalizer_dim)

            self.gmt_normalizer = EmpiricalNormalization(
                shape=[normalizer_dim], until=1.0e8)
            self.gmt_normalizer.load_state_dict(obs_norm_state)
            self.gmt_normalizer.eval()
            self.gmt_normalizer.until = 0  # Freeze statistics
            print(f"[ResidualActorCritic] GMT observation normalizer loaded (dim={normalizer_dim}) and frozen")
        else:
            print("[ResidualActorCritic] WARNING: No observation normalizer found in GMT checkpoint!")

        self.gmt_policy_obs_dim = int(gmt_policy_obs_dim)
        if self.num_task_corrections > 0:
            _validate_frontres_task_space_observation_authority(
                num_actor_obs=num_actor_obs,
                num_frontres_obs=self.num_frontres_obs,
                gmt_policy_obs_dim=self.gmt_policy_obs_dim,
            )

        # ========== Load Ref Vel Estimator ==========

        self.ref_vel_estimator = None # 导入速度归一器
        self.num_ref_vel_estimator_obs = num_ref_vel_estimator_obs

        if ref_vel_estimator_checkpoint_path is not None:
            if num_ref_vel_estimator_obs is None:
                raise ValueError("num_ref_vel_estimator_obs must be provided when ref_vel_estimator_checkpoint_path is specified")

            print(f"[ResidualActorCritic] Loading ref_vel estimator from: {ref_vel_estimator_checkpoint_path}")
            print(f"[ResidualActorCritic] Estimator type: {ref_vel_estimator_type}")

            # Load estimator based on type
            if ref_vel_estimator_type == "mlp":
                from rsl_rl.modules import VelocityEstimator
                self.ref_vel_estimator = VelocityEstimator.load(
                    ref_vel_estimator_checkpoint_path,
                    device=str(next(self.gmt_policy.parameters()).device))
            elif ref_vel_estimator_type == "transformer":
                from rsl_rl.modules import VelocityEstimatorTransformer
                estimator_checkpoint = torch.load(
                    ref_vel_estimator_checkpoint_path,
                    map_location=str(next(self.gmt_policy.parameters()).device),
                    weights_only=False)
                self.ref_vel_estimator = VelocityEstimatorTransformer(
                    feature_dim=estimator_checkpoint.get('feature_dim', 61),
                    history_length=estimator_checkpoint.get('history_length', 5),
                    d_model=estimator_checkpoint.get('d_model', 128),
                    nhead=estimator_checkpoint.get('nhead', 4),
                    num_layers=estimator_checkpoint.get('num_layers', 2),)
                self.ref_vel_estimator.load_state_dict(estimator_checkpoint['model_state_dict'])
                self.ref_vel_estimator = self.ref_vel_estimator.to(next(self.gmt_policy.parameters()).device)
                print(f"[ResidualActorCritic] Transformer estimator loaded successfully")
            else:
                raise ValueError(f"Unknown ref_vel_estimator_type: {ref_vel_estimator_type}. Must be 'mlp' or 'transformer'")

            # Freeze estimator
            self.ref_vel_estimator.eval()
            for param in self.ref_vel_estimator.parameters():
                param.requires_grad = False
            print("[ResidualActorCritic] Ref vel estimator loaded and frozen")

        # ========== Build Front-End Residual Network ==========

        # Joint-space mode outputs [Δq, Δz]. Task-space mode is full-6D ΔSE(3).
        _frontres_input_dim = num_frontres_obs if num_frontres_obs > 0 else num_actor_obs
        self.residual_actor = self._build_residual_actor(
            input_dim=_frontres_input_dim,
            output_dim=self.total_output_dim,
            hidden_dims=residual_hidden_dims,
            activation=activation_fn,
            last_layer_gain=residual_last_layer_gain)
        if num_task_corrections > 0:
            print(f"[FrontEndResidualActorCritic] FrontRES output: "
                  f"{self.total_output_dim} task-space dims "
                  "[Δpos(3)+Δrpy(3)] — no Δq patching")
        else:
            print(f"[FrontEndResidualActorCritic] FrontRES output: "
                  f"{num_actions} Δq + {num_z_outputs} Δz = {self.total_output_dim} dims")
        print(f"[FrontEndResidualActorCritic] FrontRES network: {self.residual_actor} "
              f"(input_dim={_frontres_input_dim}, output_dim={self.total_output_dim})")

        # ========== Build Critic ==========

        critic_layers: list[nn.Module] = []
        prev_dim = num_critic_obs
        
        # 创建critic网络架构
        if critic_hidden_dims:
            # 创建首层: 线性层+激活层
            critic_layers.append(nn.Linear(prev_dim, critic_hidden_dims[0]))
            critic_layers.append(activation_fn)

            # 创建隐藏层
            for layer_index in range(len(critic_hidden_dims)):
                # 取出每层的输入维度
                in_dim = critic_hidden_dims[layer_index]

                # 达到最后一层时将输出维度设置为1, 因为只需要输出评分
                if layer_index == len(critic_hidden_dims) - 1:
                    critic_layers.append(nn.Linear(in_dim, 1))
                else: # 取出每层的输出维度并创建层级: 线性层+激活层
                    out_dim = critic_hidden_dims[layer_index + 1]
                    critic_layers.append(nn.Linear(in_dim, out_dim))
                    critic_layers.append(activation_fn)
        else:
            critic_layers.append(nn.Linear(prev_dim, 1))

        # List 2 Sequence实例化Critic网络
        self.critic = nn.Sequential(*critic_layers)
        print(f"[ResidualActorCritic] Critic MLP: {self.critic}")

        if init_critic_from_gmt: # 导入GMT的Critic权重作为RES的Critic
            self._load_critic_from_checkpoint(state_dict, num_critic_obs)

        # ========== Action Noise ==========

        # Distribution covers the full residual output. Task-space mode uses
        # the same direct six coordinates for mean, sample, storage and execution.
        #
        # Task-space mode (FrontRES SE3 corrector): σ is a FIXED hyperparameter,
        # not a trainable parameter.
        #
        # Why fixed for task-space:
        #   FrontRES predicts a known target (-OU perturbation). Its uncertainty
        #   should decrease as the mean μ improves — this happens naturally via the
        #   supervised loss driving μ toward the target, not via PPO optimising σ.
        #   Making σ trainable under PPO causes 1/σ² gradient amplification:
        #   at σ=0.1, log_prob gradients are 100× larger than at σ=1.0, allowing
        #   even small negative advantages to drive σ upward explosively.
        #   PPO's objective (maximise reward) is orthogonal to "reduce correction
        #   variance" — it has no reason to keep σ small, and entropy bonuses
        #   actively push it up.  The result is the σ→28 / PhysX corruption spiral.
        #
        # Standard locomotion actors (num_task_corrections == 0) retain trainable
        # σ because exploration breadth is legitimately reward-relevant.
        if self.noise_std_type == "scalar":
            _val = init_noise_std * torch.ones(self.total_output_dim)
            if self.num_task_corrections > 0:
                self.register_buffer('std', _val)
            else:
                self.std = nn.Parameter(_val)
        elif self.noise_std_type == "log":
            _val = torch.log(init_noise_std * torch.ones(self.total_output_dim))
            if self.num_task_corrections > 0:
                self.register_buffer('log_std', _val)
            else:
                self.log_std = nn.Parameter(_val)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None

        # disable args validation for speedup
        Normal.set_default_validate_args(False)

        # ========== Create Composed Actor for ONNX Export ==========

        # ONNX exporter expects policy.actor attribute
        # Create a wrapper module that composes GMT + residual
        self.actor = ComposedActor(
            self.gmt_policy,
            self.residual_actor,
            gmt_actor_input_dim,
            num_actor_obs,
            q_ref_start_idx,
            num_actions,
            num_z_outputs if num_task_corrections == 0 else 0)
        print("[ResidualActorCritic] Created composed actor for ONNX export")

        # ========== Store and report the effective GMT observation route ==========

        # GMT actor input may include a separate ref-velocity suffix; the
        # diagnostic below distinguishes suffix routing from actual padding.
        self.gmt_actor_input_dim = gmt_actor_input_dim
        self.num_actor_obs = num_actor_obs

        # ========== Legacy joint-space curriculum: Δq injection weight ==========
        # alpha ∈ [0, 1]: effective q_dot = q_ref + alpha * Δq
        # Set externally by the runner each iteration according to the schedule:
        #   Phase 0 (Critic warmup):  alpha = alpha_init (fixed, non-zero)
        #   Phase 1 (Ramp):           alpha_init → 1.0 (linear)
        #   Phase 2 (Full training):  alpha = 1.0
        # Default 1.0 so that evaluation / non-curriculum training is unaffected.
        self.delta_q_alpha: float = 1.0

        if self.num_task_corrections <= 0:
            print(f"[ResidualActorCritic] Δq output clipping: tanh * {self.max_delta_q:.3f} rad "
                  f"(≈ ±{self.max_delta_q * 57.3:.1f}°) per joint")

        for message in _gmt_observation_route_messages(
            environment_obs_dim=int(num_actor_obs),
            gmt_policy_obs_dim=int(gmt_policy_obs_dim),
            gmt_actor_input_dim=int(gmt_actor_input_dim),
            task_space_frontres=self.num_task_corrections > 0,
            has_gmt_normalizer=self.gmt_normalizer is not None,
            has_ref_vel_estimator=self.ref_vel_estimator is not None,
        ):
            print(message)

        self.enforce_frozen_gmt_inference()

    def enforce_frozen_gmt_inference(self) -> None:
        """Keep every frozen-GMT component outside the training lifecycle."""

        for module in (self.gmt_policy, self.gmt_normalizer, self.ref_vel_estimator):
            if module is None:
                continue
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        if self.gmt_normalizer is not None and hasattr(self.gmt_normalizer, "until"):
            self.gmt_normalizer.until = 0

    def train(self, mode: bool = True):
        """Train FrontRES Actor/Critic without reopening the frozen GMT family."""

        super().train(mode)
        self.enforce_frozen_gmt_inference()
        return self

    def _frontres_raw_task_output(self, policy_obs: torch.Tensor) -> torch.Tensor:
        """Return raw full-6D task-space correction logits."""

        if not isinstance(policy_obs, torch.Tensor) or policy_obs.ndim != 2:
            raise TypeError("FrontRES actor input must be a rank-2 tensor")
        expected = int(self.num_frontres_obs)
        if expected <= 0 or int(policy_obs.shape[-1]) != expected:
            raise ValueError(
                "FrontRES actor input has the wrong deployable-prefix width: "
                f"expected {expected}, got {int(policy_obs.shape[-1])}"
            )
        if not bool(torch.isfinite(policy_obs).all().item()):
            raise ValueError("FrontRES actor input must be finite")
        return self.residual_actor(policy_obs)

    def _frontres_task_proposal(self, raw: torch.Tensor) -> torch.Tensor:
        """Return the direct world-frame full-6D Delta SE(3) proposal."""

        if raw.ndim != 2 or int(raw.shape[-1]) != 6:
            raise ValueError(f"FrontRES task proposal must have shape [B,6], got {tuple(raw.shape)}")
        if not bool(torch.isfinite(raw).all().item()):
            raise ValueError("FrontRES task proposal must be finite")
        return raw

    def _infer_gmt_architecture(self, state_dict, activation):
        """Infer GMT policy architecture from checkpoint state_dict"""
        # Extract actor hidden dimensions
        actor_hidden_dims = []
        actor_keys = sorted([k for k in state_dict.keys() if k.startswith("actor.") and ".weight" in k])
        for i in range(len(actor_keys) - 1):  # Exclude last layer
            key = actor_keys[i]
            out_dim = state_dict[key].shape[0]
            actor_hidden_dims.append(out_dim)

        # Extract critic hidden dimensions
        critic_hidden_dims = []
        critic_keys = sorted([k for k in state_dict.keys() if k.startswith("critic.") and ".weight" in k])
        for i in range(len(critic_keys) - 1):  # Exclude last layer
            key = critic_keys[i]
            out_dim = state_dict[key].shape[0]
            critic_hidden_dims.append(out_dim)

        # Get noise std type and value
        if "std" in state_dict:
            noise_std_type = "scalar"
            init_noise_std = state_dict["std"][0].item()
        elif "log_std" in state_dict:
            noise_std_type = "log"
            init_noise_std = torch.exp(state_dict["log_std"][0]).item()
        else:
            noise_std_type = "scalar"
            init_noise_std = 1.0

        return {
            "actor_hidden_dims": actor_hidden_dims,
            "critic_hidden_dims": critic_hidden_dims,
            "activation": activation,
            "init_noise_std": init_noise_std,
            "noise_std_type": noise_std_type,
        }

    def _load_critic_from_checkpoint(self, checkpoint_state_dict, expected_input_dim):
        """Load critic weights from a checkpoint state_dict into the residual critic."""
        critic_state_dict = {
            k.replace("critic.", ""): v
            for k, v in checkpoint_state_dict.items()
            if k.startswith("critic.")
        }
        if not critic_state_dict:
            raise ValueError("No critic weights found in GMT checkpoint state_dict.")

        if "0.weight" not in critic_state_dict:
            raise ValueError("GMT critic state_dict missing first layer weights (critic.0.weight).")

        checkpoint_input_dim = critic_state_dict["0.weight"].shape[1]
        if checkpoint_input_dim != expected_input_dim:
            raise ValueError(
                "GMT critic input dim does not match residual critic input dim "
                f"({checkpoint_input_dim} != {expected_input_dim})."
            )

        try:
            self.critic.load_state_dict(critic_state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load GMT critic weights into residual critic. "
                "Check that critic_hidden_dims and input dims match the checkpoint."
            ) from exc

        print("[ResidualActorCritic] Critic weights loaded from GMT checkpoint")

    def _build_residual_actor(self, input_dim, output_dim, hidden_dims, activation, last_layer_gain):
        """
        Build residual network with small-gain Xavier initialization on last layer.

        The small gain (e.g., 0.01) ensures residual starts near zero:
        - Initial behavior: a_final ≈ a_gmt (GMT policy dominates)
        - Gradual learning: residual slowly learns corrections
        - Stable training: avoids large initial perturbations
        """
        layers = []
        prev_dim = input_dim

        # Hidden layers: standard Xavier init
        for hidden_dim in hidden_dims:
            linear = nn.Linear(prev_dim, hidden_dim)
            nn.init.xavier_uniform_(linear.weight, gain=1.0)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(activation)
            prev_dim = hidden_dim

        # Last layer: small gain Xavier (0.01) for near-zero initialization
        last_layer = nn.Linear(prev_dim, output_dim)
        nn.init.xavier_uniform_(last_layer.weight, gain=last_layer_gain)
        nn.init.zeros_(last_layer.bias)
        layers.append(last_layer)

        # Verify initialization
        with torch.no_grad():
            weight_norm = torch.norm(last_layer.weight).item()
            print(f"[ResidualActorCritic] Residual last layer weight norm: {weight_norm:.6f} "
                  f"(gain={last_layer_gain})")

        return nn.Sequential(*layers)

    def _pad_observations_for_gmt(self, observations):
        """
        Pad observations if GMT policy expects more dimensions than provided.

        This handles the case where the GMT checkpoint was created with a different
        observation dimension than the current environment provides.
        """
        if observations.shape[-1] < self.gmt_actor_input_dim:
            # Pad with zeros to match GMT's expected input dimension
            padding_size = self.gmt_actor_input_dim - observations.shape[-1]
            padding = torch.zeros(
                *observations.shape[:-1], padding_size,
                device=observations.device,
                dtype=observations.dtype
            )
            observations = torch.cat([observations, padding], dim=-1)

        return observations

    def reset(self, dones=None):
        """Reset policy state (for recurrent policies)"""
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _parse_observations(self, observations):
        """
        Parse raw observation tensor (or dict) into (policy_obs, ref_vel, ref_vel_estimator_obs).

        When num_frontres_obs > 0, policy_obs is trimmed to the first num_frontres_obs dims
        (reference-frame only: command + motion_anchor_ori_b). The FULL observation is
        stored as self._cached_full_policy_obs so GMT callers can still access it.

        Returns:
            policy_obs (Tensor): obs for residual_actor  [N, num_frontres_obs or num_actor_obs]
            ref_vel    (Tensor | None): reference velocity suffix
            ref_vel_estimator_obs (Tensor | None): obs for vel estimator
        """
        if isinstance(observations, dict):
            full_policy_obs       = observations["policy"]
            ref_vel               = None
            ref_vel_estimator_obs = observations.get("ref_vel_estimator", None)
        elif isinstance(observations, torch.Tensor):
            obs_dim = observations.shape[-1]
            if obs_dim == self.num_actor_obs:
                full_policy_obs       = observations
                ref_vel               = None
                ref_vel_estimator_obs = None
            elif obs_dim == self.gmt_actor_input_dim:
                ref_vel_dim           = self.gmt_actor_input_dim - self.num_actor_obs
                full_policy_obs       = observations[:, :-ref_vel_dim]
                ref_vel               = observations[:, -ref_vel_dim:]
                ref_vel_estimator_obs = None
            else:
                raise ValueError(
                    f"Unexpected observation dimension: {obs_dim}. "
                    f"Expected {self.num_actor_obs} (policy_obs) or "
                    f"{self.gmt_actor_input_dim} (policy_obs+ref_vel)"
                )
        else:
            raise TypeError(f"Unexpected observation type: {type(observations)}")

        # Cache full policy obs for GMT callers
        self._cached_full_policy_obs = full_policy_obs

        if self.num_task_corrections > 0:
            _validate_frontres_task_space_observation_authority(
                num_actor_obs=int(full_policy_obs.shape[-1]),
                num_frontres_obs=self.num_frontres_obs,
                gmt_policy_obs_dim=self.gmt_policy_obs_dim,
            )

        # FrontRES subset: when num_frontres_obs > 0, residual_actor only sees
        # reference-frame data (first N dims), not proprioception.
        if self.num_frontres_obs > 0:
            policy_obs = full_policy_obs[:, :self.num_frontres_obs]
        else:
            policy_obs = full_policy_obs

        return policy_obs, ref_vel, ref_vel_estimator_obs

    def _apply_delta_q_and_run_gmt(self, policy_obs, delta_q, ref_vel, ref_vel_estimator_obs):
        """
        Apply Δq to q_ref, build GMT observation, run frozen GMT.

        Always called inside torch.no_grad() – no gradient needed here because
        FrontRES distributes over Δq directly (see update_distribution).

        When num_frontres_obs > 0, policy_obs is trimmed (reference-only subset).
        GMT always receives the FULL observation so it can access proprioception.
        q_ref indices are within the trimmed prefix, so patching works either way.

        Returns:
            robot_actions (Tensor): motor commands output by GMT  [N, num_actions]
        """
        # Use the FULL (untrimmed) policy obs for GMT.
        # policy_obs may be a FrontRES-specific subset; the full obs is cached.
        full_obs = getattr(self, '_cached_full_policy_obs', policy_obs)
        obs_modified = full_obs.clone()
        # delta_q here is already the Δq-only slice (first num_actions dims).
        # Δz is stored separately in last_delta_z by the callers.
        q_ref_end_idx = self.q_ref_start_idx + self.num_actions
        obs_modified[:, self.q_ref_start_idx:q_ref_end_idx] = (
            obs_modified[:, self.q_ref_start_idx:q_ref_end_idx] + self.delta_q_alpha * delta_q
        )

        # Build GMT observation (add ref_vel suffix if available)
        if ref_vel is not None:
            gmt_obs = torch.cat([obs_modified, ref_vel], dim=-1)
        elif self.ref_vel_estimator is not None and ref_vel_estimator_obs is not None:
            ref_vel = self.ref_vel_estimator(ref_vel_estimator_obs)
            gmt_obs = torch.cat([obs_modified, ref_vel], dim=-1)
        else:
            gmt_obs = self._pad_observations_for_gmt(obs_modified)

        return self.gmt_policy.act_inference(gmt_obs)

    def _run_gmt_direct(self, policy_obs, ref_vel, ref_vel_estimator_obs):
        """Run GMT without q_ref patching (task-space mode).

        policy_obs may be a FrontRES-specific subset (when num_frontres_obs > 0).
        GMT receives only the GMT-compatible suffix from self._cached_full_policy_obs.
        """
        # B1: Start from the full policy obs, then strip FrontRES-only prefix for GMT.
        gmt_input = getattr(self, '_cached_full_policy_obs', policy_obs)
        if self.gmt_normalizer is not None:
            _gmt_mean = getattr(self.gmt_normalizer, '_mean', None)
            if _gmt_mean is not None:
                _, gmt_input = split_frontres_policy_obs(gmt_input, _gmt_mean.shape[-1])

        if ref_vel is not None:
            gmt_obs = torch.cat([gmt_input, ref_vel], dim=-1)
        elif self.ref_vel_estimator is not None and ref_vel_estimator_obs is not None:
            ref_vel = self.ref_vel_estimator(ref_vel_estimator_obs)
            gmt_obs = torch.cat([gmt_input, ref_vel], dim=-1)
        else:
            gmt_obs = self._pad_observations_for_gmt(gmt_input)
        return self.gmt_policy.act_inference(gmt_obs)

    def run_frozen_gmt_from_suffix(self, gmt_observations: torch.Tensor) -> torch.Tensor:
        """Run the frozen GMT from its authoritative normalized suffix only.

        Clean and Noisy baselines have no FrontRES action and therefore must
        not fabricate a 158D actor prefix merely to reuse ``_parse_observations``.
        This is the narrow GMT-owned entrypoint for those baseline executions.
        """

        if not isinstance(gmt_observations, torch.Tensor) or gmt_observations.ndim != 2:
            raise TypeError("frozen GMT suffix must be a rank-2 tensor")
        expected = int(self.gmt_policy_obs_dim)
        if expected <= 0 or int(gmt_observations.shape[-1]) != expected:
            raise ValueError(
                "frozen GMT suffix has the wrong authority width: "
                f"expected {expected}, got {int(gmt_observations.shape[-1])}"
            )
        if not bool(torch.isfinite(gmt_observations).all()):
            raise ValueError("frozen GMT suffix must be finite")
        with torch.inference_mode():
            actions = self.gmt_policy.act_inference(gmt_observations)
        if not isinstance(actions, torch.Tensor) or int(actions.shape[0]) != int(gmt_observations.shape[0]):
            raise RuntimeError("frozen GMT must return one action per suffix row")
        return actions.detach().clone()

    def _frontres_forward(self, observations):
        """
        Full FrontRES → GMT pipeline used only by act_inference (deployment / evaluation).

        For RL training, update_distribution + get_env_action is used instead, so that
        the policy distribution is defined over Δq-space (or task-space) and gradient
        never needs to flow through the frozen GMT network.

        Returns:
            robot_actions (Tensor): final motor commands from GMT
            frontres_out  (Tensor): FrontRES output (Δq or full-6D [Δpos, Δrpy])
        """
        policy_obs, ref_vel, ref_vel_estimator_obs = self._parse_observations(observations)

        raw = self._frontres_raw_task_output(policy_obs)

        if self.num_task_corrections > 0:
            proposal = self._frontres_task_proposal(raw)
            frontres_out = proposal
            self.last_task_correction = proposal.detach()
            self.last_delta_z = None
            with torch.no_grad():
                robot_actions = self._run_gmt_direct(policy_obs, ref_vel, ref_vel_estimator_obs)
        else:
            delta_q = torch.tanh(raw[:, :self.num_actions]) * self.max_delta_q
            if self.num_z_outputs > 0:
                self.last_delta_z = torch.tanh(raw[:, self.num_actions:]) * self.max_delta_z
            else:
                self.last_delta_z = None
            frontres_out = delta_q
            with torch.no_grad():
                robot_actions = self._apply_delta_q_and_run_gmt(
                    policy_obs, delta_q, ref_vel, ref_vel_estimator_obs)

        return robot_actions, frontres_out

    def get_task_correction_inference(self, observations):
        """Deterministic direct task-space correction for runner/env-side application."""
        if self.num_task_corrections <= 0:
            raise RuntimeError("get_task_correction_inference is only valid in task-space FrontRES mode")
        # B1: Parse the policy observation without narrowing the full-6D repair cone.
        policy_obs, _, _ = self._parse_observations(observations)
        raw = self._frontres_raw_task_output(policy_obs)
        proposal = self._frontres_task_proposal(raw)
        correction = proposal
        self.last_task_correction = proposal.detach()
        self.last_delta_z = None
        self.last_residual_actions = correction.detach()
        return correction

    def update_distribution(self, observations):
        """构造 direct Delta SE(3) PPO 的 full-6D action distribution.

        函数名说明:
            `update_distribution` 是 FrontRES policy distribution owner, 产生 mean
            和 sigma; 它不执行 GMT, 也不把 residual action 改写为 motor action.

        主链路:
            上游: `act` 或 PPO evaluate 传入同一布局的 normalized observation.
            下游: `Normal(mean, sigma)` 为 rollout sampling, old stats storage,
            log_prob 和 KL 提供同一个 raw action space.

        语义:
            PPO 优化的是完整 6D Delta SE(3) distribution. 梯度只穿过 FrontRES
            actor, frozen GMT 和机器人动力学属于 environment boundary.
        """
        policy_obs, _, _ = self._parse_observations(observations)

        # # ── DEBUG: 验证 obs 布局与 q_ref_start_idx 是否正确，只打印一次 ──────────
        # if not getattr(self, '_obs_layout_debug_done', False):
        #     obs_dim = policy_obs.shape[1]

        #     # 从 command manager 中拿到当前帧真实 q_ref（需要 env 引用，这里用 obs 间接验证）
        #     # 假设布局：command(58)×5 + ori(6)×5 + ang_vel(3)×5 + jpos(29)×5 + jvel(29)×5 + act(29)×5
        #     single_frame = 58 + 6 + 3 + 29 + 29 + 29  # = 154
        #     history_len  = obs_dim // single_frame

        #     print("\n" + "="*60)
        #     print(f"[DEBUG FrontRES] obs_dim={obs_dim}, single_frame={single_frame}, "
        #           f"inferred history_length={history_len}")
        #     print(f"[DEBUG FrontRES] q_ref_start_idx={self.q_ref_start_idx}, "
        #           f"num_actions={self.num_actions}")

        #     # 打印各帧 q_ref_pos（obs[0:29], obs[58:87], ..., obs[232:261]）的均值
        #     for i in range(history_len):
        #         start = i * 58
        #         end   = start + self.num_actions  # 29
        #         frame_qref = policy_obs[0, start:end]
        #         label = f"t-{history_len-1-i}" if i < history_len - 1 else "t(current)"
        #         print(f"  obs[{start}:{end}] q_ref_pos @ {label}: "
        #               f"mean={frame_qref.mean().item():.4f}, "
        #               f"std={frame_qref.std().item():.4f}, "
        #               f"sample={frame_qref[:3].tolist()}")

        #     # 打印 q_ref_start_idx 处的 slice
        #     idx = self.q_ref_start_idx
        #     target_slice = policy_obs[0, idx : idx + self.num_actions]
        #     print(f"\n  → q_ref_start_idx={idx} 处的 slice: "
        #           f"mean={target_slice.mean().item():.4f}, "
        #           f"std={target_slice.std().item():.4f}, "
        #           f"sample={target_slice[:3].tolist()}")
        #     print(f"  （如果与 t(current) 行一致，则 q_ref_start_idx 正确）")
        #     print("="*60 + "\n")

        #     self._obs_layout_debug_done = True
        # # ── END DEBUG ─────────────────────────────────────────────────────────

        # Cache full observations so get_env_action can access ref_vel if present
        self._cached_observations = observations

        # B2: Produce raw full-6D mean and positive sigma for one Normal distribution.
        # FrontRES forward
        raw = self._frontres_raw_task_output(policy_obs) if self.num_task_corrections > 0 else self.residual_actor(policy_obs)

        if self.num_task_corrections > 0:
            frontres_mean = raw
            self.last_task_correction = self._frontres_task_proposal(raw).detach()
            self.last_delta_z          = None
            self.last_residual_actions = self.last_task_correction
        else:
            # Joint-space mode: output = [Δq (num_actions), Δz (num_z_outputs)]
            delta_q_mean = torch.tanh(raw[:, :self.num_actions]) * self.max_delta_q
            if self.num_z_outputs > 0:
                delta_z_mean  = torch.tanh(raw[:, self.num_actions:]) * self.max_delta_z
                frontres_mean = torch.cat([delta_q_mean, delta_z_mean], dim=-1)
                self.last_delta_z = delta_z_mean.detach()
            else:
                frontres_mean = delta_q_mean
                self.last_delta_z = None
            self.last_residual_actions = delta_q_mean.detach()

        # Build distribution over residual-action space.
        # NOTE: scalar std is an unconstrained nn.Parameter; apply softplus to
        # guarantee std > 0 at all times and avoid Normal(mean, negative_std) crash.
        #
        # WHY clamp(min=0.01) instead of 1e-6:
        #   With 29 action dims and std=1e-6, log_prob(a≈μ) = -29*log(1e-6) ≈ +400.
        #   If old log_prob was computed with std=0.05 (≈+87), then log_ratio=313,
        #   ratio=exp(313) which overflows float32 → Inf surrogate → Inf gradient.
        #   With std_min=0.01: log_prob_max = -29*log(0.01) ≈ +133.
        #   Combined with log_ratio clipping in ppo.py, this prevents Inf IS ratios.
        #   In practice, a well-trained locomotion policy has std ≈ 0.02–0.05,
        #   so 0.01 is a safe floor that does not prevent meaningful exploration decay.
        _STD_MIN = 0.01
        if self.noise_std_type == "scalar":
            if self.num_task_corrections > 0:
                std = self.std.clamp(min=_STD_MIN)
            else:
                std = torch.nn.functional.softplus(self.std).clamp(min=_STD_MIN)
        elif self.noise_std_type == "log":
            if self.num_task_corrections > 0:
                std = self.log_std.clamp(min=_STD_MIN)
            else:
                std = torch.exp(self.log_std).clamp(min=_STD_MIN)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")

        # PyTorch clamp() propagates NaN (NaN.clamp(min=x) == NaN).
        # If self.std becomes NaN via a gradient explosion, Normal(mean, NaN) will crash
        # in .sample() because NaN >= 0 is False.  nan_to_num breaks the death spiral by
        # replacing NaN/Inf std with a safe floor value so training can continue.
        std = std.nan_to_num(nan=_STD_MIN, posinf=5.0, neginf=_STD_MIN)
        std = std.expand_as(frontres_mean)

        # Distribution is over the full residual output. PPO stores residual
        # samples as "actions"; get_env_action maps them to GMT/robot actions.
        self.distribution = Normal(frontres_mean, std)
        # B3: mean 和 sigma 定义 act 与 PPO storage 共用的 raw distribution.

    def act(self, observations, **kwargs):
        """从 FrontRES distribution 采样 rollout residual action.

        函数名说明:
            `act` 返回策略 action, 即 full-6D Delta SE(3) repair; 它不是 GMT
            motor action. Motor action 由后续 `get_env_action` 单独产生.

        主链路:
            上游: rollout step 传入 normalized policy observation.
            下游: direct repair 写入 PPO transition, 同时传给 task correction
            application 和 frozen-GMT execution.

        语义:
            sample 与 mean/sigma 同处于直接 full-6D Delta SE(3) 坐标，PPO
            storage、log_prob 和环境执行不得再引入第二种动作表示。
        """
        self.update_distribution(observations)
        raw_sample = self.distribution.sample()
        action = self._frontres_task_proposal(raw_sample) if self.num_task_corrections > 0 else raw_sample
        # AUDIT-ACTION-01: Record the exact mean/sigma/action tuple at the actor owner.
        # Result: PENDING_LIVE.
        emit_formal_runtime_probe(
            "AUDIT-ACTION-01",
            observations=observations,
            mean=self.distribution.mean,
            sigma=self.distribution.stddev,
            action=action,
        )
        return action

    def get_env_action(self, observations, delta_q_sample: torch.Tensor) -> torch.Tensor:
        """在保存 repair action 后调用 frozen GMT 生成 motor action.

        函数名说明:
            `get_env_action` 是 FrontRES action 到环境 motor action 的 execution
            adapter; 它不重新采样 repair, 也不允许梯度进入 frozen GMT.

        主链路:
            上游: rollout step 传入 `act` 刚采样的 full-6D repair 和同源 obs.
            下游: task-space repair 写入 command owner, frozen GMT 读取修正后的
            reference 并返回 `env.step` 使用的 robot actions.

        语义:
            PPO storage 保存的是 repair action, 不是 robot action. GMT 在
            `torch.no_grad()` 下执行, 其参数必须保持 frozen.
        """
        # Prefer the cached obs from act() which may include ref_vel suffix.
        # If _cached_observations has a different number of environments than delta_q_sample
        # (B1 split-env case: runner calls with sliced obs[:N_train] or obs[N_train:]),
        # fall back to the passed observations to avoid dimension mismatch.
        # B1: 复用与 sampled FrontRES action 配对的 observation.
        cached = getattr(self, '_cached_observations', observations)
        if cached.shape[0] != delta_q_sample.shape[0]:
            cached = observations
        policy_obs, ref_vel, ref_vel_estimator_obs = self._parse_observations(cached)

        # B2: 先记录 full-6D correction, 再在 no_grad 边界执行 frozen GMT.
        if self.num_task_corrections > 0:
            # Task-space mode: delta_q_sample is the full [Δpos(3), Δrpy(3)] sample.
            # Store as last_task_correction so the runner can apply it to the command term.
            self.last_task_correction = delta_q_sample.detach()
            with torch.no_grad():
                robot_actions = self._run_gmt_direct(policy_obs, ref_vel, ref_vel_estimator_obs)
        else:
            # Joint-space mode: delta_q_sample may be [Δq, Δz]; only Δq slice goes to GMT.
            delta_q_only = delta_q_sample[:, :self.num_actions]
            with torch.no_grad():
                robot_actions = self._apply_delta_q_and_run_gmt(
                    policy_obs, delta_q_only, ref_vel, ref_vel_estimator_obs)

        # B3: AUDIT-GMT-01 confirms GMT consumed the repaired path under the no_grad boundary above.
        # Result: PENDING_LIVE.
        emit_formal_runtime_probe(
            "AUDIT-GMT-01",
            policy_obs=policy_obs,
            frontres_action=delta_q_sample,
            robot_actions=robot_actions,
            gmt_training=getattr(self.gmt_policy, "training", "missing"),
        )
        return robot_actions

    def get_actions_log_prob(self, actions):
        """
        Log-probability of residual samples under the current distribution.

        `actions` here are full frontres_output values stored during rollout (NOT robot actions).
        """
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_actions_log_prob_selected(self, actions, selected_dims):
        """Log-probability restricted to selected residual-action dimensions.

        HSL+HRL uses this to let PPO update only the scalar trust/filter head,
        while supervised losses keep ownership of the geometric correction.
        """
        if selected_dims is None:
            return self.get_actions_log_prob(actions)
        dims = torch.as_tensor(selected_dims, device=actions.device, dtype=torch.long)
        dims = dims[(dims >= 0) & (dims < actions.shape[-1])]
        if dims.numel() == 0:
            return torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)

        return self.distribution.log_prob(actions)[:, dims].sum(dim=-1)

    def get_actions_log_prob_per_dim(self, actions, selected_dims):
        """Per-dimension log-probability for selected residual-action dimensions."""
        return self._actions_log_prob_per_dim_from_normal(
            actions,
            self.distribution.mean,
            self.distribution.stddev,
            selected_dims,
        )

    def get_actions_log_prob_per_dim_from_stats(self, actions, mean, std, selected_dims):
        """Per-dimension log-probability under a supplied Gaussian distribution."""
        return self._actions_log_prob_per_dim_from_normal(actions, mean, std, selected_dims)

    def _actions_log_prob_per_dim_from_normal(self, actions, mean, std, selected_dims):
        dims = torch.as_tensor(selected_dims, device=actions.device, dtype=torch.long)
        dims = dims[(dims >= 0) & (dims < actions.shape[-1])]
        if dims.numel() == 0:
            return torch.zeros(actions.shape[0], 0, device=actions.device, dtype=actions.dtype)

        dist = Normal(mean, std)
        return dist.log_prob(actions)[:, dims]

    def act_inference(self, observations):
        """Deterministic robot actions for evaluation / deployment."""
        actions, _ = self._frontres_forward(observations)
        return actions

    def evaluate(self, critic_observations, **kwargs):
        """Evaluate value function"""
        value = self.critic(critic_observations)
        return value
