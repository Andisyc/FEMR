#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "run" / "run_frontres_stage3_segment_hrl.sh"
SENTINEL_FLAGS = {
    "sentinel": "--frontres_segment_live_sentinel_only",
    "probe": "--frontres_segment_live_probe_only",
    "storage": "--frontres_segment_live_storage_write_only",
    "single_update": "--frontres_segment_live_single_update_only",
    "update_loop": "--frontres_segment_live_update_loop_only",
}


def _run_preflight(mode: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        checkpoint = tmp_path / "stage1_model.pt"
        motion_path = tmp_path / "motions"
        checkpoint.write_text("fake checkpoint for launch-contract test\n")
        motion_path.mkdir()
        env = os.environ.copy()
        env["FRONTRES_STAGE_PREFLIGHT_ONLY"] = "1"
        env["FRONTRES_STAGE3_RUN_CONTRACTS"] = "0"
        return subprocess.run(
            [
                "bash",
                str(SCRIPT),
                str(checkpoint),
                str(motion_path),
                "1",
                "2",
                "3",
                mode,
            ],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )


def _command_line(result: subprocess.CompletedProcess[str]) -> str:
    for line in result.stdout.splitlines():
        if line.startswith("Command: "):
            return line
    raise AssertionError(f"preflight command line missing from output:\n{result.stdout}\n{result.stderr}")


def _probe(name: str, command: str) -> None:
    print(
        f"[probe step7] {name}: "
        f"stage3={'--frontres_stage stage3_segment_hrl' in command} "
        f"resume_stage1={'--resume_student_checkpoint' in command} "
        f"is_full_resume_false={'--is_full_resume False' in command} "
        f"update_steps_3={'--frontres_segment_live_update_steps 3' in command} "
        f"update_loop={'--frontres_segment_live_update_loop_only' in command} "
        f"legacy_stage2={'stage2_acceptance' in command} "
        f"mosaic_path={'/MOSAIC/' in command}",
        flush=True,
    )


def test_stage3_train_launch_preflight_builds_femr_command() -> None:
    result = _run_preflight("train")
    assert result.returncode == 0, result.stderr
    command = _command_line(result)
    _probe("stage3_train_launch", command)

    assert "[FrontRES Stage3 startup preflight] PASS mode=train" in result.stdout
    assert "--frontres_stage stage3_segment_hrl" in command
    assert "--resume_student_checkpoint" in command
    assert "--is_full_resume False" in command
    assert "--frontres_segment_live_update_steps 3" in command
    assert "--frontres_segment_live_update_loop_only" not in command
    assert "stage2_acceptance" not in command
    assert "/MOSAIC/" not in command


def test_stage3_update_loop_launch_preflight_adds_only_update_loop_sentinel() -> None:
    for mode, expected_flag in SENTINEL_FLAGS.items():
        result = _run_preflight(mode)
        assert result.returncode == 0, result.stderr
        command = _command_line(result)
        _probe(f"stage3_{mode}_launch", command)

        assert f"[FrontRES Stage3 startup preflight] PASS mode={mode}" in result.stdout
        assert "--frontres_stage stage3_segment_hrl" in command
        assert expected_flag in command
        for other_flag in SENTINEL_FLAGS.values():
            if other_flag != expected_flag:
                assert other_flag not in command


def test_stage3_launch_rejects_unknown_mode_before_training() -> None:
    result = _run_preflight("unknown")
    print(
        f"[probe step7] reject_unknown_mode: returncode={result.returncode} stderr={result.stderr.strip()}",
        flush=True,
    )
    assert result.returncode == 3
    assert "Unknown Stage 3 MODE: unknown" in result.stderr


if __name__ == "__main__":
    test_stage3_train_launch_preflight_builds_femr_command()
    test_stage3_update_loop_launch_preflight_adds_only_update_loop_sentinel()
    test_stage3_launch_rejects_unknown_mode_before_training()
    print("frontres_segment_stage3_launch_command_contract: ok")
