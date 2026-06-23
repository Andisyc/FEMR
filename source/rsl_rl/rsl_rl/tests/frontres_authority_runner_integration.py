"""TEST ONLY: FrontRES authority runner integration.

This test checks the live-runner plumbing without starting IsaacLab:

1. Stage-1 proposal stays in action columns [:6].
2. Stage-2 authority rho replaces action columns [6:12].
3. The same proposal/rho pair is written to transition storage fields.
4. The post-step executable reward delta is written as the K=1 authority return.
"""

from __future__ import annotations

import sys
import types
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsl_rl.storage.rollout_storage import RolloutStorage

ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _install_post_step_stubs() -> None:
    frontres_pkg = types.ModuleType("rsl_rl.frontres")
    frontres_pkg.__path__ = []
    sys.modules.setdefault("rsl_rl.frontres", frontres_pkg)

    authority_return = _load_module(
        "rsl_rl.frontres.frontres_authority_return",
        ROOT / "frontres" / "frontres_authority_return.py",
    )
    sys.modules["rsl_rl.frontres.frontres_authority_return"] = authority_return

    diagnostics = types.ModuleType("rsl_rl.frontres.frontres_reward_diagnostics")
    diagnostics.accumulate_frontres_reward_diagnostics = lambda *args, **kwargs: None
    sys.modules["rsl_rl.frontres.frontres_reward_diagnostics"] = diagnostics

    reward_window = types.ModuleType("rsl_rl.frontres.frontres_reward_window")
    reward_window.FrontRESRewardContext = object
    reward_window.FrontRESRewardWindow = object
    reward_window.compose_frontres_reward_delta = lambda **kwargs: kwargs["reward_window"]
    sys.modules["rsl_rl.frontres.frontres_reward_window"] = reward_window

    transition_payload = types.ModuleType("rsl_rl.frontres.frontres_transition_payload")
    transition_payload.FrontRESAcceptancePayload = object
    sys.modules["rsl_rl.frontres.frontres_transition_payload"] = transition_payload


rollout_step = _load_module(
    "frontres_rollout_step_test_module",
    ROOT / "runners" / "frontres_rollout_step.py",
)
_install_post_step_stubs()
post_step = _load_module(
    "frontres_post_step_connector_test_module",
    ROOT / "runners" / "frontres_post_step_connector.py",
)
_apply_frontres_authority_rollout_action = rollout_step._apply_frontres_authority_rollout_action
_write_frontres_authority_return = post_step._write_frontres_authority_return
finalize_frontres_authority_k_step_returns = post_step.finalize_frontres_authority_k_step_returns


class TinyAuthorityPolicy:
    task_conf_dim = 6
    authority_actor = object()

    def get_authority_rho(
        self,
        observations: torch.Tensor,
        proposal_delta_se: torch.Tensor,
        *,
        active_task_dims: torch.Tensor | None = None,
        detach_proposal: bool = True,
    ) -> torch.Tensor:
        if not detach_proposal:
            raise AssertionError("Runner must detach Stage-1 proposal before authority rho.")
        rho = torch.sigmoid(proposal_delta_se + observations[:, :6] * 0.1)
        if active_task_dims is not None:
            rho = rho * active_task_dims.to(device=rho.device, dtype=rho.dtype)
        return rho


class TinyAuthorityAlg:
    def __init__(self) -> None:
        self.policy = TinyAuthorityPolicy()
        self.transition = RolloutStorage.Transition()
        self.gamma = 0.9

    def _authority_actor_critic_enabled(self) -> bool:
        return True

    def _authority_active_task_dim_mask(self, *, device, dtype) -> torch.Tensor:
        return torch.tensor([1, 1, 0, 1, 0, 1], device=device, dtype=dtype)


def _make_runner(num_envs: int = 4) -> SimpleNamespace:
    motion = SimpleNamespace(perturber=None)
    command_manager = SimpleNamespace(_terms={"motion": motion})
    return SimpleNamespace(
        alg=TinyAuthorityAlg(),
        env=SimpleNamespace(num_envs=num_envs, command_manager=command_manager),
        cfg={"frontres_authority_live_debug": False, "frontres_authority_return_horizon": 3},
        device=torch.device("cpu"),
    )


class TinyBurstPerturber:
    def __init__(self, states: list[dict[str, torch.Tensor]]) -> None:
        self.cfg = SimpleNamespace(iid_temporal_mode="burst")
        self.states = states
        self.idx = 0

    def frontres_authority_event_state(self, num_envs: int) -> dict[str, torch.Tensor]:
        state = self.states[min(self.idx, len(self.states) - 1)]
        self.idx += 1
        return state


def test_authority_rollout_replaces_only_rho_columns() -> None:
    runner = _make_runner()
    obs = torch.linspace(-0.5, 0.5, steps=runner.env.num_envs * 8).view(runner.env.num_envs, 8)
    actions = torch.zeros(runner.env.num_envs, 12)
    proposal = torch.tensor(
        [
            [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
            [-0.10, -0.20, -0.30, -0.40, -0.50, -0.60],
            [0.90, 0.80, 0.70, 0.60, 0.50, 0.40],
            [-0.90, -0.80, -0.70, -0.60, -0.50, -0.40],
        ]
    )
    actions[:, :6] = proposal
    actions[:, 6:12] = 0.99

    rewritten = _apply_frontres_authority_rollout_action(
        runner,
        obs=obs,
        actions=actions,
        is_frontres=True,
        is_task_space_mode=True,
        n_train=2,
        rollout_step=0,
    )

    expected_rho = runner.alg.policy.get_authority_rho(
        obs,
        proposal,
        active_task_dims=runner.alg._authority_active_task_dim_mask(device=obs.device, dtype=obs.dtype),
        detach_proposal=True,
    )
    torch.testing.assert_close(rewritten[:2, :6], proposal[:2])
    torch.testing.assert_close(rewritten[:2, 6:12], expected_rho[:2])
    torch.testing.assert_close(rewritten[2:, 6:12], actions[2:, 6:12])
    torch.testing.assert_close(runner.alg.transition.actions, rewritten)
    torch.testing.assert_close(runner.alg.transition.proposal_delta_se[:2], proposal[:2])
    torch.testing.assert_close(runner.alg.transition.authority_action[:2], expected_rho[:2])
    torch.testing.assert_close(runner.alg.transition.authority_mask.view(-1), torch.tensor([1.0, 1.0, 0.0, 0.0]))


def test_authority_return_uses_one_step_r_delta() -> None:
    runner = _make_runner()
    runner.alg.transition.proposal_delta_se = torch.zeros(runner.env.num_envs, 6)
    runner.alg.transition.authority_action = torch.zeros(runner.env.num_envs, 6)
    runner.alg.transition.authority_rho = torch.zeros(runner.env.num_envs, 6)
    runner.alg.transition.authority_mask = torch.zeros(runner.env.num_envs, 1)

    r_delta = torch.tensor([1.25, -0.50])
    _write_frontres_authority_return(runner, r_delta=r_delta, n_train=2)

    torch.testing.assert_close(runner.alg.transition.authority_return_k.view(-1), torch.tensor([1.25, -0.50, 0.0, 0.0]))
    torch.testing.assert_close(runner.alg.transition.authority_mask.view(-1), torch.tensor([0.0, 0.0, 0.0, 0.0]))


def test_burst_event_reuses_one_authority_query() -> None:
    runner = _make_runner()
    runner.env.command_manager._terms["motion"].perturber = TinyBurstPerturber(
        [
            {
                "event_start": torch.tensor([True, True, False, False]),
                "event_active": torch.tensor([True, True, False, False]),
                "event_step": torch.tensor([0, 0, 0, 0]),
                "event_duration": torch.tensor([3, 3, 0, 0]),
            },
            {
                "event_start": torch.tensor([False, False, False, False]),
                "event_active": torch.tensor([True, True, False, False]),
                "event_step": torch.tensor([1, 1, 0, 0]),
                "event_duration": torch.tensor([3, 3, 0, 0]),
            },
        ]
    )
    obs0 = torch.zeros(runner.env.num_envs, 8)
    actions0 = torch.zeros(runner.env.num_envs, 12)
    actions0[:, :6] = 0.2
    first = _apply_frontres_authority_rollout_action(
        runner,
        obs=obs0,
        actions=actions0,
        is_frontres=True,
        is_task_space_mode=True,
        n_train=2,
        rollout_step=0,
    )
    first_proposal = first[:2, :6].clone()
    first_rho = first[:2, 6:12].clone()
    torch.testing.assert_close(runner.alg.transition.authority_mask.view(-1), torch.tensor([1.0, 1.0, 0.0, 0.0]))

    obs1 = torch.ones(runner.env.num_envs, 8)
    actions1 = torch.zeros(runner.env.num_envs, 12)
    actions1[:, :6] = -0.8
    second = _apply_frontres_authority_rollout_action(
        runner,
        obs=obs1,
        actions=actions1,
        is_frontres=True,
        is_task_space_mode=True,
        n_train=2,
        rollout_step=1,
    )
    torch.testing.assert_close(second[:2, :6], first_proposal)
    torch.testing.assert_close(second[:2, 6:12], first_rho)
    torch.testing.assert_close(runner.alg.transition.authority_mask.view(-1), torch.tensor([0.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(runner.alg.transition.authority_event_active.view(-1), torch.tensor([1.0, 1.0, 0.0, 0.0]))


def test_finalize_k_step_return_writes_only_event_start_frames() -> None:
    runner = _make_runner(num_envs=2)
    runner.alg.storage = RolloutStorage(
        "frontres",
        num_envs=2,
        num_transitions_per_env=4,
        obs_shape=(3,),
        privileged_obs_shape=(3,),
        actions_shape=(12,),
        device="cpu",
    )
    storage = runner.alg.storage
    storage.rewards[:, :, 0] = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )
    storage.dones.zero_()
    storage.authority_event_start.zero_()
    storage.authority_event_active.zero_()
    storage.authority_event_start[0, 0, 0] = 1.0
    storage.authority_event_start[1, 1, 0] = 1.0
    storage.authority_event_active[:3, 0, 0] = 1.0
    storage.authority_event_active[1:4, 1, 0] = 1.0
    finalize_frontres_authority_k_step_returns(runner, n_train=2)

    expected_env0 = 1.0 + 0.9 * 2.0 + 0.9 * 0.9 * 3.0
    expected_env1 = 20.0 + 0.9 * 30.0 + 0.9 * 0.9 * 40.0
    torch.testing.assert_close(storage.authority_return_k[0, 0, 0], torch.tensor(expected_env0))
    torch.testing.assert_close(storage.authority_return_k[1, 1, 0], torch.tensor(expected_env1))
    torch.testing.assert_close(storage.authority_mask[:, :, 0], torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]))


def main() -> None:
    test_authority_rollout_replaces_only_rho_columns()
    test_authority_return_uses_one_step_r_delta()
    test_burst_event_reuses_one_authority_query()
    test_finalize_k_step_return_writes_only_event_start_frames()
    print("=== FrontRES Authority Runner Integration TEST ONLY ===")
    print("checks=runner action rewrite, burst authority reuse, K-step event return write")
    print("result: PASS")


if __name__ == "__main__":
    main()
