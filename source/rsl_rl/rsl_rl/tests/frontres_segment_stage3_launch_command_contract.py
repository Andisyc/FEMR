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
    "sequence_eval": "--frontres_segment_sequence_offline_eval_only",
}


def _run_preflight(
    mode: str,
    env_overrides: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
    bounds: tuple[str, str, str] = ("1", "2", "3"),
) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        checkpoint = tmp_path / "stage1_model.pt"
        motion_path = tmp_path / "motions"
        checkpoint.write_text("fake checkpoint for launch-contract test\n")
        motion_path.mkdir()
        env = os.environ.copy()
        env["FRONTRES_STAGE_PREFLIGHT_ONLY"] = "1"
        env["FRONTRES_STAGE3_RUN_CONTRACTS"] = "0"
        env["FRONTRES_SPECIALIST_MODE"] = "rp"
        env["FRONTRES_V015_K_CURRICULUM"] = "8:2:3:4,16:2:3:0"
        if env_overrides:
            env.update(env_overrides)
        cmd = [
                "bash",
                str(SCRIPT),
                str(checkpoint),
                str(motion_path),
                *bounds,
                mode,
            ]
        if extra_args:
            cmd.extend(extra_args)
        return subprocess.run(
            cmd,
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
        f"hsl_v1={'--frontres_v015_hsl_initializer_checkpoint' in command} "
        f"offsets={'--frontres_v015_future_offsets 1\\,2' in command} "
        f"resume={'--resume' in command} "
        f"update_steps_3={'--frontres_segment_live_update_steps 3' in command} "
        f"specialist_rp={'--frontres_specialist_mode rp' in command} "
        f"update_loop={'--frontres_segment_live_update_loop_only' in command} "
        f"sequence_eval={'--frontres_segment_sequence_offline_eval_only' in command} "
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
    assert "--frontres_v015_hsl_initializer_checkpoint" in command
    assert "--frontres_v015_future_offsets 1\\,2" in command
    assert "--frontres_segment_k_curriculum 8:2:3:4\\,16:2:3:0" in command
    assert "--resume_student_checkpoint" not in command
    assert "--is_full_resume" not in command
    assert "--resume " not in command
    assert "--frontres_segment_periodic_eval_enabled" not in command
    assert "--frontres_specialist_mode rp" in command
    assert "--frontres_segment_live_update_steps 3" in command
    assert "--frontres_segment_live_update_loop_only" not in command
    assert "--frontres_segment_sequence_offline_eval_only" not in command
    assert "stage2_acceptance" not in command
    assert "/MOSAIC/" not in command


def test_g5_s4_bounded_launch_freezes_8_1_1_and_audit() -> None:
    result = _run_preflight(
        "train",
        {"FRONTRES_G5_S4_BOUNDED": "1"},
        bounds=("8", "1", "1"),
    )
    assert result.returncode == 0, result.stderr
    command = _command_line(result)
    assert "--num_envs=8" in command
    assert "--max_iterations 1" in command
    assert "--frontres_segment_live_update_steps 1" in command
    assert "--frontres_checkpoint_interval 1" in command
    assert "--frontres_formal_runtime_audit" in command


def test_g5_s4_launch_rejects_resume_periodic_and_wrong_bounds() -> None:
    forbidden = (
        ["--resume", "True"],
        ["--resume_student_checkpoint", "/tmp/legacy.pt"],
        ["--is_full_resume", "False"],
        ["--frontres_segment_periodic_eval_enabled"],
    )
    for args in forbidden:
        result = _run_preflight("train", extra_args=args)
        assert result.returncode != 0
        assert "forbids" in result.stderr
    wrong = _run_preflight(
        "train",
        {"FRONTRES_G5_S4_BOUNDED": "1"},
        bounds=("4", "1", "1"),
    )
    assert wrong.returncode != 0
    assert "8 envs, 1 iteration, and 1 update" in wrong.stderr


def test_p4_s1_strict_v5_resume_replaces_hsl_initializer() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        resume_path = Path(tmp) / "model_1.pt"
        resume_path.write_text("semantic checkpoint-v5 fixture\n")
        result = _run_preflight(
            "train",
            {"FRONTRES_V015_RESUME_CHECKPOINT": str(resume_path)},
            bounds=("8", "199", "1"),
        )
    assert result.returncode == 0, result.stderr
    command = _command_line(result)
    assert f"--frontres_v015_resume_checkpoint {resume_path}" in command
    assert "--frontres_v015_hsl_initializer_checkpoint" not in command
    assert "--resume_student_checkpoint" not in command
    assert "--resume " not in command
    assert "--is_full_resume" not in command
    assert "--max_iterations 199" in command


def test_p4_s1_strict_v5_resume_rejects_missing_checkpoint() -> None:
    result = _run_preflight(
        "train",
        {"FRONTRES_V015_RESUME_CHECKPOINT": "/definitely/missing/model_1.pt"},
        bounds=("8", "199", "1"),
    )
    assert result.returncode != 0
    assert "checkpoint-v5 resume checkpoint not found" in result.stderr


def test_stage3_update_loop_launch_preflight_adds_only_update_loop_sentinel() -> None:
    for mode, expected_flag in SENTINEL_FLAGS.items():
        result = _run_preflight(mode)
        assert result.returncode == 0, result.stderr
        command = _command_line(result)
        _probe(f"stage3_{mode}_launch", command)

        assert f"[FrontRES Stage3 startup preflight] PASS mode={mode}" in result.stdout
        assert "--frontres_stage stage3_segment_hrl" in command
        assert "--frontres_specialist_mode rp" in command
        assert expected_flag in command
        for other_flag in SENTINEL_FLAGS.values():
            if other_flag != expected_flag:
                assert other_flag not in command
        if mode == "sequence_eval":
            assert "--frontres_segment_sequence_eval_sequences 10" in command
            assert "--frontres_segment_sequence_eval_max_preroll_steps 2000" in command


def test_stage3_sequence_eval_launch_honors_smoke_eval_env_overrides() -> None:
    result = _run_preflight(
        "sequence_eval",
        {
            "OFFLINE_EVAL_SEQUENCES": "2",
            "OFFLINE_EVAL_STEPS": "120",
            "OFFLINE_EVAL_MAX_PREROLL_STEPS": "120",
        },
    )
    assert result.returncode == 0, result.stderr
    command = _command_line(result)
    _probe("stage3_sequence_eval_smoke_overrides", command)

    assert "--frontres_segment_sequence_offline_eval_only" in command
    assert "--frontres_segment_sequence_eval_sequences 2" in command
    assert "--frontres_segment_sequence_eval_max_preroll_steps 120" in command
    assert "--frontres_segment_offline_eval_steps 120" in command


def test_stage3_launch_passes_explicit_segment_ppo_schedule_and_lr_args() -> None:
    result = _run_preflight(
        "update_loop",
        extra_args=["--frontres_segment_ppo_schedule", "adaptive", "--frontres_segment_ppo_lr", "1e-6"],
    )
    assert result.returncode == 0, result.stderr
    command = _command_line(result)
    _probe("stage3_update_loop_ppo_schedule_lr_args", command)

    assert "--frontres_segment_live_update_loop_only" in command
    assert "--frontres_segment_ppo_schedule adaptive" in command
    assert "--frontres_segment_ppo_lr 1e-6" in command


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
    test_g5_s4_bounded_launch_freezes_8_1_1_and_audit()
    test_g5_s4_launch_rejects_resume_periodic_and_wrong_bounds()
    test_p4_s1_strict_v5_resume_replaces_hsl_initializer()
    test_p4_s1_strict_v5_resume_rejects_missing_checkpoint()
    test_stage3_update_loop_launch_preflight_adds_only_update_loop_sentinel()
    test_stage3_sequence_eval_launch_honors_smoke_eval_env_overrides()
    test_stage3_launch_passes_explicit_segment_ppo_schedule_and_lr_args()
    test_stage3_launch_rejects_unknown_mode_before_training()
    print("frontres_segment_stage3_launch_command_contract: ok")
