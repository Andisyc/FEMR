#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "source" / "rsl_rl" / "rsl_rl" / "frontres" / "frontres_segment_cache_indexer.py"
spec = importlib.util.spec_from_file_location("frontres_segment_cache_indexer_stage1_index_only", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load FrontRES indexer from {MODULE_PATH}")
indexer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = indexer
spec.loader.exec_module(indexer)

build_amass_segment_index = indexer.build_amass_segment_index
write_amass_segment_index = indexer.write_amass_segment_index


def _limit(value: str) -> int | None:
    raw = str(value).strip().lower()
    if raw in {"", "all", "auto", "full", "none"}:
        return None
    limit = int(raw)
    return None if limit <= 0 else limit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FrontRES Stage 1 segment index without IsaacLab rollout.")
    parser.add_argument("motion_root")
    parser.add_argument("cache_dir")
    parser.add_argument("--segment-k", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-motions", type=str, default="all")
    parser.add_argument("--max-segments", type=str, default="all")
    args = parser.parse_args()

    segments, summary = build_amass_segment_index(
        args.motion_root,
        horizon_k=max(1, int(args.segment_k)),
        frame_stride=max(1, int(args.frame_stride)),
        max_motions=_limit(args.max_motions),
        max_segments=_limit(args.max_segments),
    )
    write_amass_segment_index(args.cache_dir, segments, summary)
    probe = summary.probe()
    print(
        "[FrontRES Stage1 Segment Index] PASS "
        f"cache_dir={Path(args.cache_dir).resolve()} "
        f"motion_count={probe['motion_count']} "
        f"segment_count={probe['segment_count']} "
        f"horizon_k={probe['horizon_k']} "
        f"frame_stride={probe['frame_stride']} "
        f"max_motions={args.max_motions} "
        f"max_segments={args.max_segments}",
        flush=True,
    )


if __name__ == "__main__":
    main()
