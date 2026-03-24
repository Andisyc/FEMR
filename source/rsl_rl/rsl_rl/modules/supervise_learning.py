# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# =================================================================================================
# 思路与逻辑梳理 (注释)
#
# 核心架构:
#   你已经实现了一个非常清晰的两阶段架构：
#   1. GMT (ONNX模型): 一个预训练的专家模型，它可以根据给定的参考运动 `q` 生成高质量的动作。
#   2. FrontRES (self.student): 一个前端残差网络。它的目标不是直接生成动作，而是学习预测一个
#      残差 `delta_q_pred`。这个残差被用来修正原始的参考运动 `q_ref`，得到一个
#      "修复后" 的运动 `q_repaired = q_ref + delta_q_pred`。然后，这个修复后的运动
#      再被送入 GMT 模型，以生成最终的、更优的动作。
#
# 监督学习训练流程 (在 `supervise.py` 中实现):
#   1. [数据准备] 从数据集中获取一个参考运动 `q_ref`。
#   2. [获取专家数据] 调用 `get_gmt_action(q_ref)` 来获得 GMT 专家在原始 `q_ref` 上的动作 `a_gmt`。
#   3. [与环境交互] 在仿真环境中执行 `a_gmt`，得到实际的模拟结果 `q_sim`。
#      -> 这一步是关键，它告诉我们 GMT 在这个 `q_ref` 上的实际表现。
#   4. [计算监督目标] 调用 `get_supervision_target(q_sim, q_ref)`，计算出真实的残差
#      `delta_q_gt = q_sim - q_ref`。这就是我们希望 FrontRES 网络学会预测的目标。
#   5. [模型预测] 将 `q_ref` 输入到 FrontRES 网络中，通过调用 `forward(q_ref)` 得到
#      预测的残差 `delta_q_pred`。
#   6. [计算损失] 计算 `delta_q_pred` 和 `delta_q_gt` 之间的损失 (例如 MSELoss)。
#   7. [反向传播] 根据损失更新 FrontRES (self.student) 网络的权重。
#
# 推理/部署流程:
#   当模型训练好后，在实际使用时 (例如，在RL微调或最终部署中)，流程如下:
#   1. 接收一个参考运动 `q_ref`。
#   2. 调用 `act_inference(q_ref)` (它内部会调用 `get_action_with_gmt`)。
#   3. 在 `get_action_with_gmt` 内部:
#      a. FrontRES 网络预测残差: `delta_q_pred = self.forward(q_ref)`。
#      b. 修复运动: `q_repaired = q_ref + delta_q_pred`。
#      c. GMT 生成最终动作: `final_action = self.get_gmt_action(q_repaired)`。
#   4. 在仿真环境中执行 `final_action`。
# =================================================================================================

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
import onnxruntime as ort

from rsl_rl.utils import resolve_nn_activation


class SuperviseLearning(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_student_obs,
        num_actions,
        student_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=0.1,
        gmt_path: str = None,  # Add gmt_path to load the ONNX model
        **kwargs,
    ):
        """
        Args:
            num_student_obs (int): 学生网络(FrontRES)的输入维度。根据你的设计，这应该是 `q_ref` 的维度。
            num_actions (int): 学生网络(FrontRES)的输出维度。根据你的设计，这应该是 `delta_q_pred` 的维度。
            gmt_path (str): 预训练的 GMT ONNX 模型的路径。
        """
        if kwargs:
            print(
                "SuperviseLearning.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)
        self.loaded_teacher = False

        # ========== GMT Tracker (专家模型) ==========
        self.gmt_session = None
        if gmt_path:
            print(f"Loading GMT model from: {gmt_path}")
            try:
                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if torch.cuda.is_available()
                    else ["CPUExecutionProvider"]
                )
                self.gmt_session = ort.InferenceSession(gmt_path, providers=providers)
                self.gmt_input_name = self.gmt_session.get_inputs()[0].name
                self.gmt_output_name = self.gmt_session.get_outputs()[0].name
                print(f"GMT model loaded. Input: '{self.gmt_input_name}', Output: '{self.gmt_output_name}'")
            except Exception as e:
                print(f"Failed to load GMT model from {gmt_path}: {e}")
                self.gmt_session = None

        # FrontRES 网络的输入维度应与 `q_ref` 匹配
        mlp_input_dim_s = num_student_obs

        # ========== student (FrontRES 网络) ==========
        # 这个 MLP 就是学习预测残差 `delta_q` 的学生网络。
        student_layers = []
        student_layers.append(nn.Linear(mlp_input_dim_s, student_hidden_dims[0]))
        student_layers.append(activation)
        for layer_index in range(len(student_hidden_dims)):
            if layer_index == len(student_hidden_dims) - 1:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], num_actions))
            else:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], student_hidden_dims[layer_index + 1]))
                student_layers.append(activation)
        self.student = nn.Sequential(*student_layers)

        print(f"Student MLP: {self.student}")

        # ========== action noise ==========
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None

        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None, hidden_states=None):
        pass

    def forward(self, observations):
        """
        运行学生网络 (FrontRES) 并返回预测的残差 `delta_q_pred`。
        这是监督学习训练的核心。
        Args:
            observations (torch.Tensor): 此处的输入是 `q_ref`。
        Returns:
            torch.Tensor: 预测的残差 `delta_q_pred`。
        """
        return self.student(observations)

    @property
    def action_mean(self):
        # 在这个架构中，action_mean 是预测的 `delta_q_pred`
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        # 分布的均值是 FrontRES 网络对 `q_ref` 的预测结果
        mean = self.forward(observations)
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)
        return self.distribution

    def act(self, observations, **kwargs):
        """为 RL 训练采样动作。此处的动作是 `delta_q` 的随机版本。"""
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        """计算给定动作(delta_q)的对数概率。"""
        return self.distribution.log_prob(actions).sum(dim=-1)

    @torch.no_grad()
    def get_gmt_action(self, q_ref: torch.Tensor) -> torch.Tensor:
        """
        在给定的运动基元 `q` 上运行 GMT 专家模型，以获得专家动作。
        这个方法在两个地方被调用：
        1. 在监督学习数据收集中，输入原始 `q_ref` 来生成 `q_sim`。
        2. 在推理中，输入修复后 `q_repaired` 来生成最终动作。
        """
        if not self.gmt_session:
            raise RuntimeError("GMT model is not loaded. Cannot compute GMT action.")

        gmt_input = {self.gmt_input_name: q_ref.cpu().numpy()}
        gmt_action_np = self.gmt_session.run([self.gmt_output_name], gmt_input)[0]
        return torch.from_numpy(gmt_action_np).to(q_ref.device)

    @staticmethod
    def get_supervision_target(q_sim: torch.Tensor, q_ref: torch.Tensor) -> torch.Tensor:
        """
        [应在训练脚本中调用]
        计算监督学习的目标：`delta_q_gt = q_sim - q_ref`。
        这个 `delta_q_gt` 是 FrontRES 网络需要学习预测的 ground truth。

        重要: 训练脚本需要负责过滤掉无效的转换 (例如导致机器人摔倒的动作)。
        """
        # 监督目标是模拟器实际产生的状态与原始参考状态之间的差异。
        delta_q = q_sim - q_ref
        return delta_q

    def get_action_with_gmt(self, q_ref: torch.Tensor) -> torch.Tensor:
        """
        [用于推理]
        运行完整的 "FrontRES -> 修复 -> GMT" 推理流程。
        这是在RL微调或最终部署时使用的主要方法。
        """
        if not self.gmt_session:
            raise RuntimeError("GMT model is not loaded. Please provide 'gmt_path' in the config.")

        # 1. FrontRES 网络根据 `q_ref` 预测残差
        delta_q_pred = self.forward(q_ref)
        # 2. "修复" 运动基元
        q_repaired = q_ref + delta_q_pred

        # 3. 将修复后的 `q_repaired` 送入 GMT 专家模型以获得最终动作
        final_action = self.get_gmt_action(q_repaired)
        return final_action

    def act_inference(self, observations, **kwargs):
        """
        为环境生成一个确定性的最终动作。
        此处的 `observations` 被假定为 `q_ref`。
        """
        # 调用完整的 FrontRES + GMT 流程
        return self.get_action_with_gmt(observations)

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
