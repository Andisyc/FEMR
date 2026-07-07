#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPO_ROOT = ROOT.parents[1]
CFG_PATH = REPO_ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "tasks" / "tracking" / "config" / "g1" / "agents" / "rsl_rl_mosaic_cfg.py"

def _load_dr_curriculum():
    path = ROOT / "rsl_rl" / "frontres" / "frontres_dr_curriculum.py"
    spec = importlib.util.spec_from_file_location("frontres_dr_curriculum_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_DR_CURRICULUM = _load_dr_curriculum()
sample_per_env_dr_strength = _DR_CURRICULUM.sample_per_env_dr_strength
choice_hash = _DR_CURRICULUM.choice_hash


def _cfg_number(text: str, name: str) -> float:
    match = re.search(rf"^\s*{re.escape(name)}\s*=\s*([0-9.]+)", text, re.MULTILINE)
    assert match is not None, f"missing config value: {name}"
    return float(match.group(1))


def _cfg_bool(text: str, name: str) -> bool:
    match = re.search(rf"^\s*{re.escape(name)}\s*=\s*(True|False)", text, re.MULTILINE)
    assert match is not None, f"missing config value: {name}"
    return match.group(1) == "True"


def _current_stage3_cfg() -> tuple[SimpleNamespace, float, float]:
    text = CFG_PATH.read_text(encoding="utf-8")
    cfg = SimpleNamespace(
        frontres_mixed_dr_strength_enabled=_cfg_bool(text, "frontres_mixed_dr_strength_enabled"),
        frontres_mixed_dr_strength_per_env=_cfg_bool(text, "frontres_mixed_dr_strength_per_env"),
        frontres_mixed_dr_low_weight=_cfg_number(text, "frontres_mixed_dr_low_weight"),
        frontres_mixed_dr_mid_weight=_cfg_number(text, "frontres_mixed_dr_mid_weight"),
        frontres_mixed_dr_easy_weight=_cfg_number(text, "frontres_mixed_dr_easy_weight"),
        frontres_mixed_dr_frontier_weight=_cfg_number(text, "frontres_mixed_dr_frontier_weight"),
        frontres_mixed_dr_hard_weight=_cfg_number(text, "frontres_mixed_dr_hard_weight"),
        frontres_mixed_dr_low_hi_frac=_cfg_number(text, "frontres_mixed_dr_low_hi_frac"),
        frontres_mixed_dr_mid_hi_frac=_cfg_number(text, "frontres_mixed_dr_mid_hi_frac"),
        frontres_mixed_dr_hard_hi_frac=_cfg_number(text, "frontres_mixed_dr_hard_hi_frac"),
        frontres_mixed_dr_easy_factor=_cfg_number(text, "frontres_mixed_dr_easy_factor"),
        frontres_mixed_dr_frontier_factor=_cfg_number(text, "frontres_mixed_dr_frontier_factor"),
        frontres_mixed_dr_hard_factor=_cfg_number(text, "frontres_mixed_dr_hard_factor"),
    )
    return cfg, _cfg_number(text, "dr_min_scale"), _cfg_number(text, "dr_max_scale")


def _summary(values: list[float], dr_min: float, dr_max: float) -> dict[str, float]:
    ordered = sorted(values)

    def q(frac: float) -> float:
        idx = min(len(ordered) - 1, max(0, round(frac * (len(ordered) - 1))))
        return ordered[idx]

    return {
        "min": ordered[0],
        "p10": q(0.10),
        "p50": q(0.50),
        "p90": q(0.90),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "below_min": sum(1 for x in ordered if x < dr_min - 1e-9) / len(ordered),
        "at_min": sum(1 for x in ordered if abs(x - dr_min) <= 1e-9) / len(ordered),
        "at_max": sum(1 for x in ordered if abs(x - dr_max) <= 1e-9) / len(ordered),
    }


def _collect_strengths(cfg: SimpleNamespace, frontier_scale: float, dr_min: float, dr_max: float) -> tuple[list[float], dict[int, int]]:
    values: list[float] = []
    classes = {0: 0, 1: 0, 2: 0, 3: 0}
    for seq_idx in range(256):
        plan = sample_per_env_dr_strength(
            cfg,
            frontier_scale,
            True,
            seq_idx,
            n_train=8,
            n_candidate=0,
            n_base=0,
            num_envs=8,
            dr_min=dr_min,
            dr_max=dr_max,
        )
        assert plan.scale_vector is not None
        assert plan.mix_class is not None
        values.extend(float(x) for x in plan.scale_vector)
        for cls in plan.mix_class:
            classes[int(cls)] += 1
    return values, classes


def test_stage3_strength_distribution_uses_frontier_envelope() -> None:
    cfg, dr_min, dr_max = _current_stage3_cfg()
    assert dr_min == 1.25
    assert dr_max == 4.50
    assert cfg.frontres_mixed_dr_strength_per_env is True

    print("[FrontRES Stage3 Frontier Envelope Distribution]")
    print(
        f"config: dr_min={dr_min:.2f} dr_max={dr_max:.2f} "
        f"weights=(low:{cfg.frontres_mixed_dr_low_weight:.2f},mid:{cfg.frontres_mixed_dr_mid_weight:.2f},frontier:{cfg.frontres_mixed_dr_frontier_weight:.2f},hard:{cfg.frontres_mixed_dr_hard_weight:.2f}) "
        f"ranges=(low_hi:{cfg.frontres_mixed_dr_low_hi_frac:.2f},mid_hi:{cfg.frontres_mixed_dr_mid_hi_frac:.2f},hard_hi:{cfg.frontres_mixed_dr_hard_hi_frac:.2f})"
    )

    late_summary = None
    for frontier_scale in (1.25, 1.50, 2.00, 3.00, 4.50):
        values, classes = _collect_strengths(cfg, frontier_scale, dr_min, dr_max)
        summary = _summary(values, dr_min, dr_max)
        g = min(frontier_scale, dr_max)
        total = max(1, sum(classes.values()))
        low_frac = classes[0] / total
        mid_frac = classes[1] / total
        frontier_frac = classes[2] / total
        hard_frac = classes[3] / total
        print(
            "frontier={:.2f} strength min={:.3f} p10={:.3f} p50={:.3f} p90={:.3f} max={:.3f} mean={:.3f} below_old_min={:.1%} at_max={:.1%} class_frac=low:{:.1%},mid:{:.1%},frontier:{:.1%},hard:{:.1%}".format(
                frontier_scale,
                summary["min"],
                summary["p10"],
                summary["p50"],
                summary["p90"],
                summary["max"],
                summary["mean"],
                summary["below_min"],
                summary["at_max"],
                low_frac,
                mid_frac,
                frontier_frac,
                hard_frac,
            )
        )
        assert 0.15 <= low_frac <= 0.25
        assert 0.25 <= mid_frac <= 0.35
        assert 0.35 <= frontier_frac <= 0.45
        assert 0.05 <= hard_frac <= 0.15
        assert summary["min"] < 0.05 * max(1.0, g)
        assert summary["max"] <= dr_max
        if frontier_scale == dr_max:
            late_summary = summary

    assert late_summary is not None
    assert late_summary["p10"] < dr_min
    assert late_summary["p50"] < dr_max
    assert late_summary["at_max"] < 0.01


def main() -> None:
    test_stage3_strength_distribution_uses_frontier_envelope()
    print("frontres_segment_stage3_curriculum_distribution_contract: ok")


if __name__ == "__main__":
    main()
