#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "run" / "run_frontres_stage3_segment_hrl.sh"
SUITE = ROOT / "source" / "rsl_rl" / "rsl_rl" / "tests" / "frontres_segment_all_contract_suite.py"


def _run_contract_preflight(mode: str = "update_loop") -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        checkpoint = tmp_path / "stage1_model.pt"
        motion_path = tmp_path / "motions"
        suite_stub = tmp_path / "stage3_contract_stub.py"
        checkpoint.write_text("fake checkpoint for contract-preflight test\n")
        motion_path.mkdir()
        suite_stub.write_text(
            "print('[probe step10] stub_contract_suite: ok')\n"
            "print('frontres_segment_all_contract_suite: ok')\n"
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


def _line_index(lines: list[str], needle: str) -> int:
    for index, line in enumerate(lines):
        if needle in line:
            return index
    raise AssertionError(f"missing line containing {needle!r}")


def test_contract_gate_runs_before_stage3_command_preflight() -> None:
    result = _run_contract_preflight()
    lines = result.stdout.splitlines()
    start_i = _line_index(lines, "[FrontRES Stage3 contract preflight] START")
    suite_i = _line_index(lines, "frontres_segment_all_contract_suite: ok")
    pass_i = _line_index(lines, "[FrontRES Stage3 contract preflight] PASS")
    command_i = _line_index(lines, "[FrontRES Stage3 startup preflight] PASS mode=update_loop")
    command_line_i = _line_index(lines, "Command: ")
    command_line = lines[command_line_i]

    print(
        "[probe step10] contract_gate_order: "
        f"returncode={result.returncode} "
        f"start_i={start_i} "
        f"suite_i={suite_i} "
        f"pass_i={pass_i} "
        f"command_i={command_i} "
        f"stage3={'--frontres_stage stage3_segment_hrl' in command_line} "
        f"update_loop={'--frontres_segment_live_update_loop_only' in command_line} "
        f"default_suite={SUITE.exists()} "
        f"stub_suite={'[probe step10] stub_contract_suite: ok' in result.stdout}",
        flush=True,
    )

    assert result.returncode == 0, result.stderr
    assert start_i < suite_i < pass_i < command_i < command_line_i
    assert "--frontres_stage stage3_segment_hrl" in command_line
    assert "--frontres_segment_live_update_loop_only" in command_line
    assert SUITE.exists()
    assert "[probe step10] stub_contract_suite: ok" in result.stdout
    assert "frontres_segment_all_contract_suite: ok" in result.stdout


if __name__ == "__main__":
    test_contract_gate_runs_before_stage3_command_preflight()
    print("frontres_segment_stage3_contract_preflight_contract: ok")
