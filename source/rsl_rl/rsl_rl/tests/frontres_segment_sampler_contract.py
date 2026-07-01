from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULE_PATH = ROOT / "rsl_rl" / "frontres" / "frontres_segment_sampler.py"
spec = importlib.util.spec_from_file_location("frontres_segment_sampler", MODULE_PATH)
sampler_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = sampler_module
spec.loader.exec_module(sampler_module)
FrontRESSegmentRolloutEvidence = sampler_module.FrontRESSegmentRolloutEvidence
FrontRESSegmentSampler = sampler_module.FrontRESSegmentSampler


def _evidence(
    segment_ids: list[int],
    gain: list[float],
    repaired: list[float],
    noisy: list[float],
    *,
    fall: list[bool] | None = None,
    valid: list[bool] | None = None,
) -> FrontRESSegmentRolloutEvidence:
    n = len(segment_ids)
    return FrontRESSegmentRolloutEvidence(
        segment_ids=torch.tensor(segment_ids, dtype=torch.long),
        reset_success=torch.ones(n, dtype=torch.bool),
        score_noisy=torch.tensor(noisy, dtype=torch.float32),
        score_repaired=torch.tensor(repaired, dtype=torch.float32),
        score_clean=torch.ones(n, dtype=torch.float32),
        gain_over_noisy=torch.tensor(gain, dtype=torch.float32),
        fall_repaired=torch.tensor(fall if fall is not None else [False] * n, dtype=torch.bool),
        contact_consistency=torch.ones(n, dtype=torch.float32),
        action_norm=torch.ones(n, dtype=torch.float32),
        valid_reward=torch.tensor(valid if valid is not None else [True] * n, dtype=torch.bool),
        horizon_k=torch.ones(n, dtype=torch.long) * 4,
    )


def test_sampler_global_sampling_visits_unseen_segments() -> None:
    sampler = FrontRESSegmentSampler(5, global_frac=1.0, replay_frac=0.0, review_frac=0.0, seed=3)
    sample = sampler.sample(5)
    assert set(sample.segment_ids.tolist()) == {0, 1, 2, 3, 4}
    assert sample.source == ("global", "global", "global", "global", "global")
    assert sampler.stats().seen_count == 5


def test_sampler_replays_useful_unsolved_segments() -> None:
    sampler = FrontRESSegmentSampler(4, global_frac=0.0, replay_frac=1.0, review_frac=0.0, seed=7)
    sampler.update(_evidence([0, 1, 2, 3], gain=[0.5, 0.02, -0.2, 0.1], repaired=[0.6, 0.98, 0.1, 0.4], noisy=[0.2, 0.96, 0.2, 0.3], fall=[False, False, True, False]))

    assert sampler.priority[0] > sampler.priority[1]
    assert sampler.solved[1].item()
    assert sampler.hopeless[2].item()

    sample = sampler.sample(12)
    assert 0 in sample.segment_ids.tolist()
    assert 2 not in sample.segment_ids.tolist()


def test_sampler_reports_effective_source_after_fallback() -> None:
    sampler = FrontRESSegmentSampler(4, global_frac=0.0, replay_frac=1.0, review_frac=0.0, seed=17)
    sample = sampler.sample(4)
    assert sample.source == ("global", "global", "global", "global")

    sampler = FrontRESSegmentSampler(4, global_frac=0.0, replay_frac=0.0, review_frac=1.0, seed=19)
    sample = sampler.sample(4)
    assert sample.source == ("global", "global", "global", "global")


def test_sampler_update_probe_exposes_priority_boundary() -> None:
    sampler = FrontRESSegmentSampler(4, global_frac=0.0, replay_frac=1.0, review_frac=0.0, seed=23)
    probe = sampler.update_with_probe(
        _evidence([0, 1, 2, 3], gain=[0.5, -0.1, 0.0, 0.2], repaired=[0.6, 0.2, 0.3, 0.7], noisy=[0.2, 0.3, 0.3, 0.4], fall=[False, True, False, False])
    )
    print(
        "[probe sampler_update] "
        f"valid={probe.valid_count} fall={probe.fall_count} "
        f"useful_mean={probe.useful_mean:.6f} useful_max={probe.useful_max:.6f} "
        f"priority_before={probe.priority_before_mean:.6f} priority_after={probe.priority_after_mean:.6f} "
        f"replay_candidates={probe.replay_candidate_count} hopeless={probe.hopeless_count}",
        flush=True,
    )
    assert probe.valid_count == 4
    assert probe.fall_count == 1
    assert probe.useful_max > 0.0
    assert probe.priority_after_mean > probe.priority_before_mean
    assert probe.replay_candidate_count > 0
    assert probe.hopeless_count == 1


def test_sampler_review_and_staleness_keep_coverage() -> None:
    sampler = FrontRESSegmentSampler(3, global_frac=0.0, replay_frac=0.0, review_frac=1.0, seed=11)
    sampler.update(_evidence([0, 1], gain=[0.01, 0.4], repaired=[0.95, 0.5], noisy=[0.94, 0.1]))

    review = sampler.sample(4)
    assert set(review.segment_ids.tolist()) == {0}

    sampler.staleness[1] = 100.0
    sampler.global_frac = 0.0
    sampler.replay_frac = 1.0
    sampler.review_frac = 0.0
    replay = sampler.sample(8)
    assert 1 in replay.segment_ids.tolist()


def test_sampler_invalid_and_state_dict_restore() -> None:
    sampler = FrontRESSegmentSampler(4, global_frac=1.0, replay_frac=0.0, review_frac=0.0, seed=5)
    sampler.mark_invalid([0, 1], "bad reset")
    sample = sampler.sample(12)
    assert not ({0, 1} & set(sample.segment_ids.tolist()))

    restored = FrontRESSegmentSampler(4, seed=5)
    restored.load_state_dict(sampler.state_dict())
    assert restored.invalid.tolist() == sampler.invalid.tolist()
    assert restored.invalid_reasons[0] == "bad reset"


def main() -> None:
    test_sampler_global_sampling_visits_unseen_segments()
    test_sampler_replays_useful_unsolved_segments()
    test_sampler_reports_effective_source_after_fallback()
    test_sampler_update_probe_exposes_priority_boundary()
    test_sampler_review_and_staleness_keep_coverage()
    test_sampler_invalid_and_state_dict_restore()
    print("result: PASS")


if __name__ == "__main__":
    main()
