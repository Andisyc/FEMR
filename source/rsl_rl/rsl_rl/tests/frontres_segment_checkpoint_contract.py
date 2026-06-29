#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checkpointing = _load(
    "frontres_segment_checkpointing",
    ROOT / "rsl_rl" / "runners" / "frontres_segment_checkpointing.py",
)

FrontRESSegmentCheckpointConfig = checkpointing.FrontRESSegmentCheckpointConfig
build_frontres_segment_checkpoint_payload = checkpointing.build_frontres_segment_checkpoint_payload
restore_frontres_segment_checkpoint = checkpointing.restore_frontres_segment_checkpoint


class FakeOptimizer:
    def __init__(self) -> None:
        self.loaded = False

    def state_dict(self):
        return {"momentum": torch.tensor([3.0])}

    def load_state_dict(self, state):
        self.loaded = True
        self.loaded_state = state


class FakeNormalizer:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.loaded_state = None

    def state_dict(self):
        return {"tag": self.tag, "mean": torch.tensor([1.0])}

    def load_state_dict(self, state):
        self.loaded_state = state


class FakeSampler:
    def __init__(self) -> None:
        self.loaded_state = None

    def state_dict(self):
        return {"priority": torch.tensor([0.2, 0.7]), "seen": torch.tensor([True, False])}

    def load_state_dict(self, state):
        self.loaded_state = state


class FakeDataset:
    def __init__(self) -> None:
        self.loaded_metadata = None

    def cache_metadata(self):
        return {"motion_sha": "abc", "num_segments": 2}

    def load_cache_metadata(self, metadata):
        self.loaded_metadata = metadata


def _two_head_stage1_checkpoint() -> dict:
    return {
        "model_state_dict": {
            "residual_actor": {
                "trunk.0.weight": torch.full((5, 4), 0.11),
                "trunk.0.bias": torch.full((5,), 0.12),
                "proposal_head.weight": torch.full((6, 5), 0.21),
                "proposal_head.bias": torch.full((6,), 0.22),
                "acceptance_head.weight": torch.full((1, 5), 9.0),
                "acceptance_head.bias": torch.full((1,), 9.0),
            }
        },
        "optimizer_state_dict": {"old": torch.tensor([1.0])},
        "obs_norm_state_dict": {"mean": torch.tensor([0.5])},
        "privileged_obs_norm_state_dict": {"mean": torch.tensor([0.7])},
        "frontres_segment_sampler_state_dict": {"priority": torch.tensor([1.0, 2.0])},
        "frontres_segment_dataset_cache_metadata": {"motion_sha": "restored"},
    }


def _runner() -> SimpleNamespace:
    repair_actor = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.Tanh(), torch.nn.Linear(5, 6))
    policy = SimpleNamespace(residual_actor=repair_actor, critic=torch.nn.Linear(3, 1), std=torch.ones(6))
    return SimpleNamespace(
        alg=SimpleNamespace(policy=policy, optimizer=FakeOptimizer()),
        current_learning_iteration=12,
        obs_normalizer=FakeNormalizer("obs"),
        privileged_obs_normalizer=FakeNormalizer("priv"),
        _frontres_segment_sampler=FakeSampler(),
        _frontres_segment_dataset=FakeDataset(),
    )


def test_stage3_hsl_init_maps_two_head_proposal_and_ignores_acceptance() -> None:
    runner = _runner()
    result = restore_frontres_segment_checkpoint(
        runner,
        _two_head_stage1_checkpoint(),
        FrontRESSegmentCheckpointConfig(hsl_init_enabled=True, is_full_resume=False),
    )
    state = runner.alg.policy.residual_actor.state_dict()
    torch.testing.assert_close(state["0.weight"], torch.full((5, 4), 0.11))
    torch.testing.assert_close(state["0.bias"], torch.full((5,), 0.12))
    torch.testing.assert_close(state["2.weight"], torch.full((6, 5), 0.21))
    torch.testing.assert_close(state["2.bias"], torch.full((6,), 0.22))
    assert "residual_actor.acceptance_head.weight" in result.ignored_acceptance_keys
    assert result.optimizer_reset
    assert not result.optimizer_loaded
    assert not runner.alg.optimizer.loaded


def test_stage3_can_explicitly_resume_optimizer_and_load_normalizers() -> None:
    runner = _runner()
    result = restore_frontres_segment_checkpoint(
        runner,
        _two_head_stage1_checkpoint(),
        FrontRESSegmentCheckpointConfig(hsl_init_enabled=True, resume_optimizer=True),
    )
    assert result.optimizer_loaded
    assert not result.optimizer_reset
    assert runner.alg.optimizer.loaded
    assert "obs_norm_state_dict" in result.normalizer_keys_loaded
    assert "privileged_obs_norm_state_dict" in result.normalizer_keys_loaded
    torch.testing.assert_close(runner.obs_normalizer.loaded_state["mean"], torch.tensor([0.5]))
    torch.testing.assert_close(runner.privileged_obs_normalizer.loaded_state["mean"], torch.tensor([0.7]))


def test_stage3_restores_sampler_and_dataset_metadata_when_present() -> None:
    runner = _runner()
    result = restore_frontres_segment_checkpoint(
        runner,
        _two_head_stage1_checkpoint(),
        FrontRESSegmentCheckpointConfig(hsl_init_enabled=True),
    )
    assert result.sampler_state_loaded
    assert result.dataset_cache_metadata_loaded
    assert runner._frontres_segment_sampler.loaded_state["priority"].tolist() == [1.0, 2.0]
    assert runner._frontres_segment_dataset.loaded_metadata == {"motion_sha": "restored"}


def test_stage3_payload_saves_sampler_and_dataset_cache_metadata() -> None:
    runner = _runner()
    payload = build_frontres_segment_checkpoint_payload(runner, infos={"tag": "stage3"})
    assert payload["frontres_stage"] == "stage3_segment_hrl"
    assert payload["frontres_training_objective"] == "segment_replay_hrl"
    assert payload["infos"] == {"tag": "stage3"}
    assert "residual_actor" in payload["model_state_dict"]
    assert "optimizer_state_dict" in payload
    assert "obs_norm_state_dict" in payload
    assert "privileged_obs_norm_state_dict" in payload
    torch.testing.assert_close(payload["frontres_segment_sampler_state_dict"]["priority"], torch.tensor([0.2, 0.7]))
    assert payload["frontres_segment_dataset_cache_metadata"] == {"motion_sha": "abc", "num_segments": 2}


def test_stage3_does_not_require_acceptance_actor() -> None:
    runner = _runner()
    assert not hasattr(runner.alg.policy, "acceptance_actor")
    result = restore_frontres_segment_checkpoint(runner, _two_head_stage1_checkpoint())
    assert result.copied_actor_keys == ("0.bias", "0.weight", "2.bias", "2.weight")


def main() -> None:
    test_stage3_hsl_init_maps_two_head_proposal_and_ignores_acceptance()
    test_stage3_can_explicitly_resume_optimizer_and_load_normalizers()
    test_stage3_restores_sampler_and_dataset_metadata_when_present()
    test_stage3_payload_saves_sampler_and_dataset_cache_metadata()
    test_stage3_does_not_require_acceptance_actor()
    print("result: PASS")


if __name__ == "__main__":
    main()
