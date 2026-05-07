"""Instantaneous velocity-perturbation push (same pattern as push_by_setting_velocity)."""
from __future__ import annotations
import math
import torch


class PushController:
    """
    Delivers a single-step lateral velocity perturbation to each env at a
    randomized time offset during the observe phase.

    Push magnitude is expressed as Δv (m/s) in the horizontal plane, direction
    is uniformly random in [0, 2π).  This mirrors the IsaacLab training event
    `push_by_setting_velocity`, making it directly comparable to training conditions.
    """

    def __init__(
        self,
        num_envs: int,
        device: str,
        push_offset_range: tuple[int, int] = (0, 40),
    ):
        self.num_envs = num_envs
        self.device = device
        self.push_offset_range = push_offset_range

        # Per-env: step within observe phase at which push is applied
        self.push_at_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        # Per-env: unit direction vector (XY plane)
        self.push_dir = torch.zeros(num_envs, 3, device=device)
        # Whether the push has already been applied
        self.pushed = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def randomize(self, env_ids: torch.Tensor, seed_offset: int = 0) -> None:
        """
        Assign random push timing and direction for env_ids.
        Call once at the start of each observe phase.
        """
        n = len(env_ids)
        low, high = self.push_offset_range
        offsets = torch.randint(low, high + 1, (n,), device=self.device)
        self.push_at_step[env_ids] = offsets
        self.pushed[env_ids] = False

        angles = torch.rand(n, device=self.device) * 2.0 * math.pi
        self.push_dir[env_ids, 0] = torch.cos(angles)
        self.push_dir[env_ids, 1] = torch.sin(angles)
        self.push_dir[env_ids, 2] = 0.0

    def maybe_push(
        self,
        robot,
        observe_step: int,
        delta_v: float,
        alive: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply velocity perturbation to any env where push_at_step == observe_step
        and the env is still alive.

        Returns: bool tensor (num_envs,) — True for envs that were pushed this step.
        """
        should_push = (
            (self.push_at_step == observe_step)
            & (~self.pushed)
            & alive
        )
        push_ids = torch.where(should_push)[0]

        if push_ids.numel() > 0:
            self._apply_velocity_push(robot, push_ids, delta_v)
            self.pushed[push_ids] = True

        return should_push

    # ── internal ──────────────────────────────────────────────────────────────

    def _apply_velocity_push(
        self, robot, env_ids: torch.Tensor, delta_v: float
    ) -> None:
        """Add lateral Δv to root linear velocity for selected envs."""
        velocity_delta = self.push_dir[env_ids] * delta_v  # (N, 3)

        root_state = torch.cat(
            [
                robot.data.root_pos_w[env_ids],         # (N, 3)
                robot.data.root_quat_w[env_ids],        # (N, 4)
                robot.data.root_lin_vel_w[env_ids] + velocity_delta,  # (N, 3)
                robot.data.root_ang_vel_w[env_ids],     # (N, 3)
            ],
            dim=-1,
        )  # (N, 13)
        robot.write_root_state_to_sim(root_state, env_ids=env_ids)
