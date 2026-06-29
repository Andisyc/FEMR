from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]

DIAG_PATH = ROOT / "rsl_rl" / "frontres" / "frontres_segment_diagnostics.py"
diag_spec = importlib.util.spec_from_file_location("frontres_segment_diagnostics", DIAG_PATH)
diag_module = importlib.util.module_from_spec(diag_spec)
assert diag_spec.loader is not None
sys.modules[diag_spec.name] = diag_module
diag_spec.loader.exec_module(diag_module)
format_segment_replay_log = diag_module.format_segment_replay_log
segment_summary_to_scalars = diag_module.segment_summary_to_scalars
summarize_segment_batch = diag_module.summarize_segment_batch


@dataclass(frozen=True)
class FakeSample:
    source: tuple[str, ...]
    priority: torch.Tensor
    horizon_k: torch.Tensor


@dataclass(frozen=True)
class FakeReward:
    score_noisy: torch.Tensor
    score_repaired: torch.Tensor
    score_clean: torch.Tensor
    gain_over_noisy: torch.Tensor
    fall_flag: torch.Tensor
    contact_consistency: torch.Tensor
    valid_mask: torch.Tensor
    solved_mask: torch.Tensor
    hopeless_mask: torch.Tensor


@dataclass(frozen=True)
class FakeReset:
    success_mask: torch.Tensor
    preroll_mask: torch.Tensor


@dataclass(frozen=True)
class FakeActionStats:
    action_norm_mean: float
    per_dim_norm: torch.Tensor


def test_segment_diagnostics_required_keys_and_no_acceptance_keys() -> None:
    summary = summarize_segment_batch(
        FakeSample(
            source=("global", "replay", "review", "replay"),
            priority=torch.tensor([0.1, 0.5, 0.0, 0.2]),
            horizon_k=torch.tensor([4, 4, 4, 4]),
        ),
        FakeReward(
            score_noisy=torch.tensor([0.2, 0.4, 0.9, 0.1]),
            score_repaired=torch.tensor([0.5, 0.6, 0.92, 0.1]),
            score_clean=torch.ones(4),
            gain_over_noisy=torch.tensor([0.3, 0.2, 0.02, 0.0]),
            fall_flag=torch.tensor([False, False, False, True]),
            contact_consistency=torch.tensor([1.0, 0.8, 1.0, 0.2]),
            valid_mask=torch.tensor([True, True, True, False]),
            solved_mask=torch.tensor([False, False, True, False]),
            hopeless_mask=torch.tensor([False, False, False, True]),
        ),
        FakeReset(success_mask=torch.tensor([True, True, True, False]), preroll_mask=torch.tensor([False, True, False, True])),
        FakeActionStats(action_norm_mean=0.7, per_dim_norm=torch.arange(6, dtype=torch.float32)),
    )
    scalars = segment_summary_to_scalars(summary)
    required = {
        "segment/global_frac",
        "segment/replay_frac",
        "segment/review_frac",
        "segment/replay_pool_size",
        "segment/priority_mean",
        "segment/priority_p90",
        "segment/solved_frac",
        "segment/active_frac",
        "segment/hopeless_frac",
        "segment/reset_success_frac",
        "segment/preroll_frac",
        "segment/k",
        "segment/score_noisy",
        "segment/score_repaired",
        "segment/score_clean",
        "segment/gain_over_noisy",
        "segment/fall_frac",
        "segment/contact_consistency",
        "segment/action_norm",
        "segment/action_norm_dx",
        "segment/action_norm_dy",
        "segment/action_norm_dz",
        "segment/action_norm_droll",
        "segment/action_norm_dpitch",
        "segment/action_norm_dyaw",
    }
    assert required.issubset(scalars.keys())
    forbidden = {"acceptance_gt", "acceptance_mask", "acceptance_margin", "acceptance_prob"}
    assert forbidden.isdisjoint(scalars.keys())
    assert scalars["segment/global_frac"] == 0.25
    assert scalars["segment/replay_frac"] == 0.5
    assert scalars["segment/review_frac"] == 0.25
    assert scalars["segment/reset_success_frac"] == 0.75
    assert scalars["segment/preroll_frac"] == 0.5
    assert scalars["segment/action_norm_dyaw"] == 5.0


def test_segment_log_contains_live_path_sentinel() -> None:
    summary = summarize_segment_batch(
        FakeSample(source=("global",), priority=torch.tensor([0.0]), horizon_k=torch.tensor([8])),
        FakeReward(
            score_noisy=torch.tensor([0.2]),
            score_repaired=torch.tensor([0.5]),
            score_clean=torch.tensor([1.0]),
            gain_over_noisy=torch.tensor([0.3]),
            fall_flag=torch.tensor([False]),
            contact_consistency=torch.tensor([1.0]),
            valid_mask=torch.tensor([True]),
            solved_mask=torch.tensor([False]),
            hopeless_mask=torch.tensor([False]),
        ),
        FakeReset(success_mask=torch.tensor([True]), preroll_mask=torch.tensor([False])),
        FakeActionStats(action_norm_mean=0.1, per_dim_norm=torch.zeros(6)),
    )
    log = format_segment_replay_log(summary)
    assert "FrontRES Segment HRL active" in log
    assert "stage=stage3_segment_hrl" in log
    assert "objective=segment_replay_hrl" in log
    assert "k=8" in log
    assert "gain=0.3000" in log


def main() -> None:
    test_segment_diagnostics_required_keys_and_no_acceptance_keys()
    test_segment_log_contains_live_path_sentinel()
    print("result: PASS")


if __name__ == "__main__":
    main()
