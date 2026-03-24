# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
        if kwargs:
            print(
                "SuperviseLearning.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)
        self.loaded_teacher = False  # indicates if teacher has been loaded

        # ========== GMT Tracker ==========
        self.gmt_session = None
        if gmt_path:
            print(f"Loading GMT model from: {gmt_path}")
            try:
                # Use a list of providers, CUDA first if available
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

        mlp_input_dim_s = num_student_obs

        # ========== student ==========
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
        Runs the student network (FrontRES) and returns the predicted residual (delta_q_pred).
        This is the primary method used for supervised training.
        """
        return self.student(observations)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.forward(observations)
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)
        return self.distribution

    def act(self, observations, **kwargs):
        """Sample actions from distribution (for training with exploration)"""
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        """Compute log probability of actions under current distribution"""
        return self.distribution.log_prob(actions).sum(dim=-1)

    @torch.no_grad()
    def get_gmt_action(self, q_ref: torch.Tensor) -> torch.Tensor:
        """
        Runs the GMT tracker on a reference motion to get the expert action.
        This action should then be applied in the simulator to obtain q_sim.

        Args:
            q_ref (torch.Tensor): The reference motion observation.

        Returns:
            torch.Tensor: The expert action from the GMT model.
        """
        if not self.gmt_session:
            raise RuntimeError("GMT model is not loaded. Cannot compute GMT action.")

        gmt_input = {self.gmt_input_name: q_ref.cpu().numpy()}
        gmt_action_np = self.gmt_session.run([self.gmt_output_name], gmt_input)[0]
        return torch.from_numpy(gmt_action_np).to(q_ref.device)

    @staticmethod
    def get_supervision_target(q_sim: torch.Tensor, q_ref: torch.Tensor) -> torch.Tensor:
        """
        Computes the supervision target delta_q = q_sim - q_ref.
        This method should be called from the training script with valid data.

        IMPORTANT: The training loop is responsible for filtering out invalid transitions.
        If an action led to a fall, the corresponding (q_sim, q_ref) pair should be
        discarded and NOT passed to this function.

        Args:
            q_sim (torch.Tensor): The actual simulated joint state from the environment
                                  after applying the GMT action.
            q_ref (torch.Tensor): The original reference motion observation.

        Returns:
            torch.Tensor: The target residual (delta_q) for supervised training.
        """
        # The supervision target is the difference between what the simulation produced
        # and the original reference motion.
        delta_q = q_sim - q_ref
        return delta_q

    def get_action_with_gmt(self, q_ref: torch.Tensor) -> torch.Tensor:
        """
        Runs the full inference pipeline: FrontRES -> q_repaired -> GMT -> final_action.
        This is used for collecting simulation trajectories (q_sim) during RL finetuning
        or for final deployment.

        Args:
            q_ref (torch.Tensor): The reference motion observation.

        Returns:
            torch.Tensor: The final action to be applied in the simulator.
        """
        if not self.gmt_session:
            raise RuntimeError("GMT model is not loaded. Please provide 'gmt_path' in the config.")

        # The student (FrontRES) predicts the residual based on the reference motion
        delta_q_pred = self.forward(q_ref)
        # The "repaired" motion is the reference + predicted residual
        q_repaired = q_ref + delta_q_pred

        # The input to GMT is the "repaired" motion primitive.
        # This can now be passed to the get_gmt_action method.
        final_action = self.get_gmt_action(q_repaired)
        return final_action

    def act_inference(self, observations, **kwargs):
        """
        Produce a final action for the environment by running the full pipeline.
        During supervised pre-training, this means running FrontRES + GMT.
        """
        # The input 'observations' for the policy is assumed to be 'q_ref' here.
        return self.get_action_with_gmt(observations)

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the student network.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.
        """

        # Load specifically for FrontRES / supervised learning
        if any("actor" in key for key in state_dict.keys()):  # adapting RL checkpoint format
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
