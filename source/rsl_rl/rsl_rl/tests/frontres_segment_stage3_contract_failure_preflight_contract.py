#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "run" / "run_frontres_stage3_segment_hrl.sh"


def _run_failing_contract_preflight(mode: str = "update_loop") -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        checkpoint = tmp_path / "stage1_model.pt"
        motion_path = tmp_path / "motions"
        suite_stub = tmp_path / "stage3_contract_fail_stub.py"
        checkpoint.write_text("fake checkpoint for failing-contract-preflight test\n")
        motion_path.mkdir()
        suite_stub.write_text(
            "import sys\n"
            "print('[probe step12] failing_contract_suite: entered')\n"
            "sys.exit(37)\n"
        )
        env = os.environ.copy()
        env["FRONTRES_STAGE_PREFLIGHT_ONLY"] = "1"
        env["FRONTRES_STAGE3_RUN_CONTRACTS"] = "1"
        env["FRONTRES_STAGE3_CONTRACT_SUITE"] = str(suite_stub)
        env["FRONTRES_STAGE3_CONTRACT_PYTHON"] = sys.executable
        return subprocess.run(
            [
                "bash",
                str(SCRIPT),
                str(checkpoint),
                str(motion_path),
                "1",
                "2",
                "1",
                mode,
            ],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )


def test_contract_failure_stops_before_stage3_command() -> None:
    result = _run_failing_contract_preflight()
    combined = result.stdout + result.stderr
    print(
        "[probe step12] contract_failure_gate: "
        f"returncode={result.returncode} "
        f"entered={'[probe step12] failing_contract_suite: entered' in combined} "
        f"contract_start={'[FrontRES Stage3 contract preflight] START' in combined} "
        f"contract_pass={'[FrontRES Stage3 contract preflight] PASS' in combined} "
        f"startup_preflight={'[FrontRES Stage3 startup preflight]' in combined} "
        f"command_printed={'Command: ' in combined}",
        flush=True,
    )

    assert result.returncode == 37
    assert "[probe step12] failing_contract_suite: entered" in combined
    assert "[FrontRES Stage3 contract preflight] START" in combined
    assert "[FrontRES Stage3 contract preflight] PASS" not in combined
    assert "[FrontRES Stage3 startup preflight]" not in combined
    assert "Command: " not in combined


if __name__ == "__main__":
    test_contract_failure_stops_before_stage3_command()
    print("frontres_segment_stage3_contract_failure_preflight_contract: ok")
