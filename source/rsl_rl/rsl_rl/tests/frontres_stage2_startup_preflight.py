#!/usr/bin/env python3
"""Stage 2 startup smoke test without launching Isaac Sim."""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "run/run_frontres_stage2_acceptance.sh"


def main() -> None:
    with tempfile.NamedTemporaryFile(prefix="frontres_stage1_", suffix=".pt") as ckpt:
        env = os.environ.copy()
        env["FRONTRES_STAGE_PREFLIGHT_ONLY"] = "1"
        result = subprocess.run(
            [
                "bash",
                str(SCRIPT),
                ckpt.name,
                "/tmp/frontres_motion_preflight",
                "8",
                "1",
            ],
            cwd=str(ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.stdout.strip():
        print(result.stdout.rstrip())
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    required = (
        "[FrontRES Stage2 startup preflight] PASS",
        "scripts/rsl_rl/train.py",
        "--frontres_stage stage2_acceptance",
        "--is_full_resume False",
        "--experiment_name g1_flat_frontres_stage2_acceptance",
    )
    for needle in required:
        if needle not in result.stdout:
            raise AssertionError(f"missing startup preflight evidence: {needle}")
    print("PASS: Stage 2 startup preflight expands the FEMR acceptance command.")


if __name__ == "__main__":
    main()
