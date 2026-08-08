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


def _run_contract_preflight(mode: str = "train") -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        checkpoint = tmp_path / "stage1_model.pt"
        motion_path = tmp_path / "motions"
        cache_path = tmp_path / "segment_cache"
        suite_stub = tmp_path / "stage3_contract_stub.py"
        checkpoint.write_text("fake checkpoint for contract-preflight test\n")
        motion_path.mkdir()
        cache_path.mkdir()
        suite_stub.write_text(
            "print('[probe step10] stub_contract_suite: ok')\n"
            "print('frontres_segment_all_contract_suite: ok')\n"
        )
        env = os.environ.copy()
        env["CACHE_DIR"] = str(cache_path)
        env["FRONTRES_STAGE_PREFLIGHT_ONLY"] = "1"
        env["FRONTRES_STAGE3_RUN_CONTRACTS"] = "1"
        env["FRONTRES_STAGE3_CONTRACT_SUITE"] = str(suite_stub)
        env["FRONTRES_STAGE3_CONTRACT_PYTHON"] = sys.executable
        env["FRONTRES_V015_K_CURRICULUM"] = "8:4:200:500:1300:lower-k8:0.5:linear-joint-v1:1300:2.381,16:4:300:300:900:lower-k16:0.6:linear-joint-v1:900:2.381,32:4:400:300:625:lower-k32:0.7:linear-joint-v1:625:2.381"
        return subprocess.run(
            [
                "bash",
                str(SCRIPT),
                str(checkpoint),
                str(motion_path),
                "16",
                "1",
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
    command_i = _line_index(lines, "[FrontRES Stage3 startup preflight] PASS mode=train")
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
        f"legacy_update={any(flag in command_line for flag in ('--frontres_segment_live_update_loop_only', '--frontres_segment_live_single_update_only'))} "
        f"default_suite={SUITE.exists()} "
        f"stub_suite={'[probe step10] stub_contract_suite: ok' in result.stdout}",
        flush=True,
    )

    assert result.returncode == 0, result.stderr
    assert start_i < suite_i < pass_i < command_i < command_line_i
    assert "--frontres_stage stage3_segment_hrl" in command_line
    assert "--frontres_segment_live_update_loop_only" not in command_line
    assert "--frontres_segment_live_single_update_only" not in command_line
    assert SUITE.exists()
    assert "[probe step10] stub_contract_suite: ok" in result.stdout
    assert "frontres_segment_all_contract_suite: ok" in result.stdout


def test_retired_optimizer_modes_fail_closed_before_command_construction() -> None:
    for mode in ("single_update", "update_loop"):
        result = _run_contract_preflight(mode)
        assert result.returncode == 4
        assert "FRS-PPO-v007 rejects retired optimizer-writing Stage 3 mode" in result.stderr
        assert "[FrontRES Stage3 startup preflight] PASS" not in result.stdout
        assert "Command: " not in result.stdout


if __name__ == "__main__":
    test_contract_gate_runs_before_stage3_command_preflight()
    test_retired_optimizer_modes_fail_closed_before_command_construction()
    print("frontres_segment_stage3_contract_preflight_contract: ok")
