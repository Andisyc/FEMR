from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]

SAMPLER_PATH = ROOT / "rsl_rl" / "frontres" / "frontres_segment_sampler.py"
sampler_spec = importlib.util.spec_from_file_location("frontres_segment_sampler", SAMPLER_PATH)
sampler_module = importlib.util.module_from_spec(sampler_spec)
assert sampler_spec.loader is not None
sys.modules[sampler_spec.name] = sampler_module
sampler_spec.loader.exec_module(sampler_module)
FrontRESSegmentRolloutEvidence = sampler_module.FrontRESSegmentRolloutEvidence

REWARD_PATH = ROOT / "rsl_rl" / "frontres" / "frontres_segment_reward.py"
reward_spec = importlib.util.spec_from_file_location("frontres_segment_reward", REWARD_PATH)
reward_module = importlib.util.module_from_spec(reward_spec)
assert reward_spec.loader is not None
sys.modules[reward_spec.name] = reward_module
reward_spec.loader.exec_module(reward_module)
FrontRESSegmentReward = reward_module.FrontRESSegmentReward
FrontRESSegmentScoreWindow = reward_module.FrontRESSegmentScoreWindow


def test_reward_is_noisy_relative_gain() -> None:
    reward = FrontRESSegmentReward()
    result = reward.compute(torch.tensor([0.2, 0.7]), torch.tensor([0.6, 0.4]), torch.tensor([1.0, 1.0]))
    assert result.reward[0] > 0.0
    assert result.reward[1] < 0.0
    torch.testing.assert_close(result.gain_over_noisy, torch.tensor([0.4, -0.3]))


def test_noisy_good_is_low_learning_value_and_both_fail_is_hopeless() -> None:
    reward = FrontRESSegmentReward()
    result = reward.compute(torch.tensor([0.95, 0.1]), torch.tensor([0.96, 0.1]), torch.tensor([1.0, 1.0]))
    assert result.solved_mask.tolist() == [True, False]
    assert result.hopeless_mask.tolist() == [False, True]
    assert result.diagnostics["learning_value_mean"] < 0.05


def test_invalid_reset_masks_reward() -> None:
    class ResetResult:
        success_mask = torch.tensor([True, False])

    reward = FrontRESSegmentReward()
    result = reward.compute(torch.tensor([0.1, 0.1]), torch.tensor([0.5, 0.8]), torch.tensor([1.0, 1.0]), ResetResult())
    assert result.valid_mask.tolist() == [True, False]
    assert result.reward[0] > 0.0
    assert result.reward[1].item() == 0.0


def test_full_env_reward_is_ignored_unless_enabled() -> None:
    noisy = FrontRESSegmentScoreWindow(score=torch.tensor([0.8]))
    repaired = FrontRESSegmentScoreWindow(score=torch.tensor([0.2]), full_env_reward=torch.tensor([10.0]))
    clean = FrontRESSegmentScoreWindow(score=torch.tensor([1.0]))

    disabled = FrontRESSegmentReward(use_full_env_reward=False).compute(noisy, repaired, clean)
    enabled = FrontRESSegmentReward(use_full_env_reward=True, full_env_weight=1.0).compute(noisy, repaired, clean)
    assert disabled.reward.item() < 0.0
    assert enabled.reward.item() > 0.0


def test_priority_evidence_uses_segment_reward_result() -> None:
    reward = FrontRESSegmentReward(evidence_type=FrontRESSegmentRolloutEvidence)
    result = reward.compute(torch.tensor([0.2]), torch.tensor([0.7]), torch.tensor([1.0]))
    evidence = reward.priority_evidence(result, torch.tensor([3]), horizon_k=4)
    assert isinstance(evidence, FrontRESSegmentRolloutEvidence)
    assert evidence.segment_ids.tolist() == [3]
    torch.testing.assert_close(evidence.gain_over_noisy, torch.tensor([0.5]))


def main() -> None:
    test_reward_is_noisy_relative_gain()
    test_noisy_good_is_low_learning_value_and_both_fail_is_hopeless()
    test_invalid_reset_masks_reward()
    test_full_env_reward_is_ignored_unless_enabled()
    test_priority_evidence_uses_segment_reward_result()
    print("result: PASS")


if __name__ == "__main__":
    main()
