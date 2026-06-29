#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


connector_module = _load("frontres_segment_replay", ROOT / "rsl_rl" / "runners" / "frontres_segment_replay.py")
action_module = _load("frontres_hrl_action", ROOT / "rsl_rl" / "frontres" / "frontres_hrl_action.py")
dataset_module = _load("frontres_segment_dataset", ROOT / "rsl_rl" / "frontres" / "frontres_segment_dataset.py")
diag_module = _load("frontres_segment_diagnostics", ROOT / "rsl_rl" / "frontres" / "frontres_segment_diagnostics.py")
ppo_module = _load("frontres_segment_ppo", ROOT / "rsl_rl" / "algorithms" / "frontres_segment_ppo.py")
reward_module = _load("frontres_segment_reward", ROOT / "rsl_rl" / "frontres" / "frontres_segment_reward.py")
sampler_module = _load("frontres_segment_sampler", ROOT / "rsl_rl" / "frontres" / "frontres_segment_sampler.py")
storage_module = _load("frontres_segment_storage", ROOT / "rsl_rl" / "frontres" / "frontres_segment_storage.py")

FrontRESSegmentReplayConnector = connector_module.FrontRESSegmentReplayConnector
FrontRESHRLActionProjector = action_module.FrontRESHRLActionProjector
FrontRESSegmentBatch = dataset_module.FrontRESSegmentBatch
FrontRESSegmentSpec = dataset_module.FrontRESSegmentSpec
FrontRESSegmentState = dataset_module.FrontRESSegmentState
FrontRESSegmentPPOBatch = ppo_module.FrontRESSegmentPPOBatch
compute_frontres_segment_ppo_loss = ppo_module.compute_frontres_segment_ppo_loss
FrontRESSegmentReward = reward_module.FrontRESSegmentReward
FrontRESSegmentRolloutEvidence = sampler_module.FrontRESSegmentRolloutEvidence
FrontRESSegmentRolloutStorage = storage_module.FrontRESSegmentRolloutStorage
format_segment_replay_log = diag_module.format_segment_replay_log
summarize_segment_batch = diag_module.summarize_segment_batch


@dataclass(frozen=True)
class FakeSample:
    segment_ids: torch.Tensor
    source: tuple[str, ...]
    priority: torch.Tensor
    staleness: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class FakeResetRequest:
    segment_ids: torch.Tensor


@dataclass(frozen=True)
class FakeResetResult:
    success_mask: torch.Tensor
    preroll_mask: torch.Tensor


class FakeSampler:
    def __init__(self) -> None:
        self.updated = None

    def sample(self, batch_size: int) -> FakeSample:
        assert batch_size == 2
        return FakeSample(
            segment_ids=torch.tensor([10, 11]),
            source=("global", "replay"),
            priority=torch.tensor([0.0, 0.5]),
            staleness=torch.tensor([0.0, 2.0]),
            valid_mask=torch.tensor([True, True]),
        )

    def update(self, evidence) -> None:
        self.updated = evidence

    def stats(self):
        return type("Stats", (), {"replay_pool_size": 1})()


class FakeDataset:
    def get_segments(self, segment_ids: torch.Tensor) -> FrontRESSegmentBatch:
        state = FrontRESSegmentState(
            root_pos=torch.zeros(2, 3),
            root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
            root_lin_vel=torch.ones(2, 3),
            root_ang_vel=torch.ones(2, 3),
            dof_pos=torch.zeros(2, 4),
            dof_vel=torch.ones(2, 4),
        )
        return FrontRESSegmentBatch(
            segment_ids=segment_ids,
            specs=(
                FrontRESSegmentSpec(segment_id=10, motion_id=0, start_frame=0, phase=0.25),
                FrontRESSegmentSpec(segment_id=11, motion_id=0, start_frame=1, phase=0.50),
            ),
            clean_state=state,
            reference_window=torch.zeros(2, 4, 3),
            phase=torch.tensor([0.25, 0.50]),
            horizon_k=torch.tensor([4, 4]),
            perturbation_family=("planar", "yaw"),
            perturbation_strength=torch.tensor([0.1, 0.2]),
        )


class FakeResetAdapter:
    def build_request(self, batch, mode: str = "auto") -> FakeResetRequest:
        assert mode == "auto"
        return FakeResetRequest(segment_ids=batch.segment_ids)

    def apply(self, env, request: FakeResetRequest) -> FakeResetResult:
        return FakeResetResult(success_mask=torch.tensor([True, True]), preroll_mask=torch.tensor([False, True]))


class FakeSegmentPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = torch.nn.Linear(4, 6, bias=False)
        self.critic = torch.nn.Linear(4, 1, bias=False)
        torch.nn.init.zeros_(self.actor.weight)
        torch.nn.init.zeros_(self.critic.weight)

    def _observations(self, batch) -> torch.Tensor:
        return torch.stack(
            [
                torch.ones_like(batch.phase),
                batch.phase,
                batch.perturbation_strength,
                torch.zeros_like(batch.phase),
            ],
            dim=-1,
        )

    def act_segment(self, *, batch, reset_result):
        observations = self._observations(batch)
        mean = self.actor(observations)
        value = self.critic(observations).squeeze(-1)
        action = torch.tensor(
            [[0.4, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.3]],
            dtype=observations.dtype,
            device=observations.device,
        )
        return {
            "action": action,
            "observations": observations,
            "log_prob": torch.zeros(2, dtype=observations.dtype),
            "value": value.detach(),
            "mean": mean.detach(),
            "sigma": torch.ones_like(action),
        }

    def evaluate_segment_actions(self, observations: torch.Tensor, actions: torch.Tensor):
        mean = self.actor(observations)
        value = self.critic(observations).squeeze(-1)
        log_prob = -0.5 * (actions - mean).square().sum(dim=-1)
        return {
            "log_prob": log_prob,
            "value": value,
            "entropy": torch.ones_like(log_prob) * 0.5,
            "mean": mean,
            "sigma": torch.ones_like(actions),
        }


class FakeEnv:
    pass


def _rollout_fn(*, env, batch, reset_request, reset_result, repair_action, command):
    assert "frontres_delta_se" in command
    assert repair_action.projected_delta_se.shape == (2, 6)
    return {
        "noisy": torch.tensor([0.2, 0.3]),
        "repaired": torch.tensor([0.8, 0.4]),
        "clean": torch.tensor([1.0, 1.0]),
    }


def test_fake_runner_lifecycle_storage_to_ppo_update() -> None:
    sampler = FakeSampler()
    storage = FrontRESSegmentRolloutStorage(capacity=4, obs_shape=(4,))
    policy = FakeSegmentPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.2)
    before = policy.actor.weight.detach().clone()
    connector = FrontRESSegmentReplayConnector(
        dataset=FakeDataset(),
        sampler=sampler,
        reset_adapter=FakeResetAdapter(),
        action_projector=FrontRESHRLActionProjector(active_task_dims=[0, 1, 5], upward_dz_rule="nonpositive"),
        reward=FrontRESSegmentReward(evidence_type=FrontRESSegmentRolloutEvidence),
        rollout_fn=_rollout_fn,
        transition_writer=storage,
        diagnostics_fn=summarize_segment_batch,
        log_formatter=format_segment_replay_log,
    )

    result = connector.run_step(env=FakeEnv(), policy=policy, batch_size=2)
    assert "FrontRES Segment HRL active" in result.log_string
    assert result.raw_action.shape == (2, 6)
    assert isinstance(result.policy_output, dict)
    assert sampler.updated.segment_ids.tolist() == [10, 11]
    assert storage.stats().size == 2
    assert storage.segment_source == ["global", "replay"]
    assert storage.priority_evidence[0]["segment_ids"].tolist() == [10, 11]

    ppo_batch = storage.full_batch().to_ppo_batch(FrontRESSegmentPPOBatch)
    assert ppo_batch.actions.shape == (2, 6)
    assert ppo_batch.old_log_probs.shape == (2,)
    assert ppo_batch.returns.shape == (2,)
    assert ppo_batch.valid_mask.tolist() == [True, True]
    loss = compute_frontres_segment_ppo_loss(policy, ppo_batch)
    assert loss.should_step
    optimizer.zero_grad(set_to_none=True)
    loss.total_loss.backward()
    optimizer.step()
    assert not torch.allclose(policy.actor.weight.detach(), before)


def main() -> None:
    test_fake_runner_lifecycle_storage_to_ppo_update()
    print("result: PASS")


if __name__ == "__main__":
    main()
