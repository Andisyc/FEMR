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
reward_module = _load("frontres_segment_reward", ROOT / "rsl_rl" / "frontres" / "frontres_segment_reward.py")
sampler_module = _load("frontres_segment_sampler", ROOT / "rsl_rl" / "frontres" / "frontres_segment_sampler.py")

FrontRESSegmentReplayConnector = connector_module.FrontRESSegmentReplayConnector
FrontRESHRLActionProjector = action_module.FrontRESHRLActionProjector
FrontRESSegmentBatch = dataset_module.FrontRESSegmentBatch
FrontRESSegmentSpec = dataset_module.FrontRESSegmentSpec
FrontRESSegmentState = dataset_module.FrontRESSegmentState
FrontRESSegmentReward = reward_module.FrontRESSegmentReward
FrontRESSegmentRolloutEvidence = sampler_module.FrontRESSegmentRolloutEvidence
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
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.updated = None

    def sample(self, batch_size: int) -> FakeSample:
        self.calls.append("sampler.sample")
        assert batch_size == 2
        return FakeSample(
            segment_ids=torch.tensor([0, 1]),
            source=("global", "replay"),
            priority=torch.tensor([0.0, 0.4]),
            staleness=torch.tensor([0.0, 3.0]),
            valid_mask=torch.tensor([True, True]),
        )

    def update(self, evidence) -> None:
        self.calls.append("sampler.update")
        self.updated = evidence

    def stats(self):
        return type("Stats", (), {"replay_pool_size": 1})()


class FakeDataset:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def get_segments(self, segment_ids: torch.Tensor) -> FrontRESSegmentBatch:
        self.calls.append("dataset.get_segments")
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
                FrontRESSegmentSpec(segment_id=0, motion_id=0, start_frame=0, phase=0.2),
                FrontRESSegmentSpec(segment_id=1, motion_id=0, start_frame=1, phase=0.5),
            ),
            clean_state=state,
            reference_window=torch.zeros(2, 4, 3),
            phase=torch.tensor([0.2, 0.5]),
            horizon_k=torch.tensor([3, 3]),
            perturbation_family=("planar", "yaw"),
            perturbation_strength=torch.tensor([0.1, 0.2]),
        )


class FakeResetAdapter:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def build_request(self, batch, mode: str = "auto") -> FakeResetRequest:
        self.calls.append("reset.build_request")
        assert mode == "auto"
        return FakeResetRequest(segment_ids=batch.segment_ids)

    def apply(self, env, request: FakeResetRequest) -> FakeResetResult:
        self.calls.append("reset.apply")
        env.request = request
        return FakeResetResult(success_mask=torch.tensor([True, True]), preroll_mask=torch.tensor([False, True]))


class FakePolicy:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def act_segment(self, *, batch, reset_result):
        self.calls.append("policy.act_segment")
        assert reset_result.success_mask.all()
        return torch.tensor([[1.0, 2.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 3.0]])


class FakeTransitionWriter:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.payload = None

    def write(self, **payload) -> None:
        self.calls.append("transition.write")
        self.payload = payload


def _rollout_fn(*, env, batch, reset_request, reset_result, repair_action, command):
    env.calls.append("rollout_fn")
    assert "frontres_delta_se" in command
    assert repair_action.projected_delta_se.shape == (2, 6)
    return {
        "noisy": torch.tensor([0.2, 0.3]),
        "repaired": torch.tensor([0.6, 0.5]),
        "clean": torch.tensor([1.0, 1.0]),
    }


class FakeEnv:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls


def test_segment_replay_connector_call_order_and_fields() -> None:
    calls: list[str] = []
    sampler = FakeSampler(calls)
    writer = FakeTransitionWriter(calls)
    env = FakeEnv(calls)
    connector = FrontRESSegmentReplayConnector(
        dataset=FakeDataset(calls),
        sampler=sampler,
        reset_adapter=FakeResetAdapter(calls),
        action_projector=FrontRESHRLActionProjector(active_task_dims=[0, 1, 5], upward_dz_rule="nonpositive"),
        reward=FrontRESSegmentReward(evidence_type=FrontRESSegmentRolloutEvidence),
        rollout_fn=_rollout_fn,
        transition_writer=writer,
        diagnostics_fn=summarize_segment_batch,
        log_formatter=format_segment_replay_log,
    )

    result = connector.run_step(env=env, policy=FakePolicy(calls), batch_size=2)
    expected_order = [
        "sampler.sample",
        "dataset.get_segments",
        "reset.build_request",
        "reset.apply",
        "policy.act_segment",
        "rollout_fn",
        "sampler.update",
        "transition.write",
    ]
    assert calls == expected_order
    assert writer.payload is not None
    assert writer.payload["batch"].segment_ids.tolist() == [0, 1]
    assert writer.payload["repair_action"].projected_delta_se.shape == (2, 6)
    torch.testing.assert_close(writer.payload["reward_result"].gain_over_noisy, torch.tensor([0.4, 0.2]))
    assert sampler.updated.segment_ids.tolist() == [0, 1]
    assert "FrontRES Segment HRL active" in result.log_string
    assert "objective=segment_replay_hrl" in result.log_string
    assert result.priority_evidence.gain_over_noisy.shape == (2,)


def main() -> None:
    test_segment_replay_connector_call_order_and_fields()
    print("result: PASS")


if __name__ == "__main__":
    main()
