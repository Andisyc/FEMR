#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "run" / "run_frontres_stage3_segment_hrl.sh"
FULL_PREFLIGHT_TIMEOUT_SEC = float(os.environ.get("FRONTRES_STAGE3_FULL_PREFLIGHT_TIMEOUT_SEC", "90"))


def _run_full_preflight(mode: str = "update_loop") -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        checkpoint = tmp_path / "stage1_model.pt"
        motion_path = tmp_path / "motions"
        checkpoint.write_text("fake checkpoint for full-preflight test\n")
        motion_path.mkdir()
        env = os.environ.copy()
        env["FRONTRES_STAGE_PREFLIGHT_ONLY"] = "1"
        env["FRONTRES_STAGE3_RUN_CONTRACTS"] = "1"
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
            timeout=FULL_PREFLIGHT_TIMEOUT_SEC,
        )


def _line_index(lines: list[str], needle: str) -> int:
    for index, line in enumerate(lines):
        if needle in line:
            return index
    raise AssertionError(f"missing line containing {needle!r}")


def test_real_stage3_full_contract_preflight_before_command() -> None:
    try:
        result = _run_full_preflight()
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        print(
            "[probe step11] real_full_preflight_timeout: "
            f"timeout_sec={FULL_PREFLIGHT_TIMEOUT_SEC} "
            f"stdout_has_suite_summary={'[probe step9] suite_summary:' in stdout} "
            f"stdout_has_command={'[FrontRES Stage3 startup preflight]' in stdout}",
            flush=True,
        )
        print(stdout, flush=True)
        print(stderr, flush=True)
        raise AssertionError("Stage 3 full contract preflight timed out") from exc
    lines = result.stdout.splitlines()
    start_i = _line_index(lines, "[FrontRES Stage3 contract preflight] START")
    suite_summary_i = _line_index(lines, "[probe step9] suite_summary:")
    suite_ok_i = _line_index(lines, "frontres_segment_all_contract_suite: ok")
    pass_i = _line_index(lines, "[FrontRES Stage3 contract preflight] PASS")
    command_i = _line_index(lines, "[FrontRES Stage3 startup preflight] PASS mode=update_loop")
    command_line_i = _line_index(lines, "Command: ")
    command_line = lines[command_line_i]

    print(
        "[probe step11] real_full_preflight: "
        f"returncode={result.returncode} "
        f"start_i={start_i} "
        f"suite_summary_i={suite_summary_i} "
        f"suite_ok_i={suite_ok_i} "
        f"pass_i={pass_i} "
        f"command_i={command_i} "
        f"stage3={'--frontres_stage stage3_segment_hrl' in command_line} "
        f"update_loop={'--frontres_segment_live_update_loop_only' in command_line} "
        f"legacy_stage2={'stage2_acceptance' in command_line} "
        f"mosaic_path={'/MOSAIC/' in command_line}",
        flush=True,
    )

    assert result.returncode == 0, result.stderr
    assert start_i < suite_summary_i < suite_ok_i < pass_i < command_i < command_line_i
    assert "[probe step9] suite_summary: contract_count=18 failed_count=0" in result.stdout
    assert "--frontres_stage stage3_segment_hrl" in command_line
    assert "--frontres_segment_live_update_loop_only" in command_line
    assert "stage2_acceptance" not in command_line
    assert "/MOSAIC/" not in command_line


if __name__ == "__main__":
    test_real_stage3_full_contract_preflight_before_command()
    print("frontres_segment_stage3_full_preflight_contract: ok")
