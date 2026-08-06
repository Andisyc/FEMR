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
}


def _run_preflight(
    mode: str,
    env_overrides: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
    bounds: tuple[str, str, str] = ("8", "2", "3"),
) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        checkpoint = tmp_path / "stage1_model.pt"
        motion_path = tmp_path / "motions"
        cache_path = tmp_path / "segment_cache"
        checkpoint.write_text("fake checkpoint for launch-contract test\n")
        motion_path.mkdir()
        cache_path.mkdir()
        env = os.environ.copy()
        env["CACHE_DIR"] = str(cache_path)
        env["FRONTRES_STAGE_PREFLIGHT_ONLY"] = "1"
        env["FRONTRES_STAGE3_RUN_CONTRACTS"] = "0"
        env["FRONTRES_SPECIALIST_MODE"] = "rp"
        env["FRONTRES_V015_K_CURRICULUM"] = "8:2:200:500:1300:lower-k8:0.5:linear-joint-v1:1300:2.381,16:3:300:300:900:lower-k16:0.6:linear-joint-v1:900:2.381,32:4:400:300:625:lower-k32:0.7:linear-joint-v1:625:2.381"
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
        f"hsl_v2={'--frontres_v015_hsl_initializer_checkpoint' in command} "
        f"offsets={'--frontres_v015_future_offsets 1\\,2' in command} "
        f"resume={'--resume' in command} "
        f"update_steps_3={'--frontres_segment_live_update_steps 3' in command} "
        f"specialist_rp={'--frontres_specialist_mode rp' in command} "
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
    assert "--frontres_v015_hsl_initializer_checkpoint" in command
    assert "--frontres_v015_future_offsets 1\\,2" in command
    assert "--frontres_segment_k_curriculum 8:2:200:500:1300:lower-k8:0.5:linear-joint-v1:1300:2.381" in command
    assert "--resume_student_checkpoint" not in command
    assert "--is_full_resume" not in command
    assert "--resume " not in command
    assert "--frontres_specialist_mode rp" in command
    assert "--frontres_segment_live_update_steps 3" in command
    assert "--frontres_segment_live_update_loop_only" not in command
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


def test_stage3_train_launch_accepts_explicit_checkpoint_interval() -> None:
    result = _run_preflight(
        "train",
        {"FRONTRES_CHECKPOINT_INTERVAL": "50"},
    )
    assert result.returncode == 0, result.stderr
    assert "--frontres_checkpoint_interval 50" in _command_line(result), result.stdout

    invalid = _run_preflight(
        "train",
        {"FRONTRES_CHECKPOINT_INTERVAL": "0"},
    )
    assert invalid.returncode != 0
    assert "FRONTRES_CHECKPOINT_INTERVAL must be a positive integer" in invalid.stderr


def test_g5_s4_launch_rejects_legacy_resume_and_wrong_bounds() -> None:
    forbidden = (
        ["--resume", "True"],
        ["--resume_student_checkpoint", "/tmp/legacy.pt"],
        ["--is_full_resume", "False"],
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
    assert "fresh K8/M2 campaign requires NUM_ENVS=8" in wrong.stderr


def test_strict_v9_resume_replaces_hsl_initializer() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        resume_path = Path(tmp) / "model_1.pt"
        resume_path.write_text("semantic checkpoint-v9 fixture\n")
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


def test_strict_v9_resume_rejects_missing_checkpoint() -> None:
    result = _run_preflight(
        "train",
        {"FRONTRES_V015_RESUME_CHECKPOINT": "/definitely/missing/model_1.pt"},
        bounds=("8", "199", "1"),
    )
    assert result.returncode != 0
    assert "checkpoint-v9 resume checkpoint not found" in result.stderr


def test_stage3_diagnostic_launch_preflight_adds_only_selected_sentinel() -> None:
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
def test_stage3_train_launch_passes_explicit_segment_ppo_schedule_and_lr_args() -> None:
    result = _run_preflight(
        "train",
        extra_args=["--frontres_segment_ppo_schedule", "adaptive", "--frontres_segment_ppo_lr", "1e-6"],
    )
    assert result.returncode == 0, result.stderr
    command = _command_line(result)
    _probe("stage3_train_ppo_schedule_lr_args", command)

    assert "--frontres_segment_live_update_loop_only" not in command
    assert "--frontres_segment_live_single_update_only" not in command
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


def test_stage3_launch_rejects_legacy_local_evaluation_modes() -> None:
    for mode in ("offline_eval", "sequence_eval", "policy_quality_q2d_eval"):
        result = _run_preflight(mode)
        assert result.returncode == 4
        assert "FRS-EVAL-v004 rejects legacy v002/v006/quartet local evaluation mode" in result.stderr


def test_stage3_launch_builds_active_v017_policy_quality_command() -> None:
    quality_env = {
        "POLICY_QUALITY_MANIFEST": "/tmp/frontres-v017-k16.json",
        "POLICY_QUALITY_POLICY_CHECKPOINT": "/tmp/frontres-v017-model-3500.pt",
        "POLICY_QUALITY_RESULT": "/tmp/frontres-v017-quality.json",
    }
    result = _run_preflight("policy_quality_eval", quality_env, bounds=("12", "0", "1"))
    assert result.returncode == 0, result.stderr
    command = _command_line(result)
    assert "--frontres_policy_quality_eval_only" in command
    assert "--frontres_policy_quality_manifest /tmp/frontres-v017-k16.json" in command
    initializer = command.split("--frontres_v015_hsl_initializer_checkpoint ", 1)[1].split()[0]
    evaluator_hsl = command.split("--frontres_policy_quality_hsl_checkpoint ", 1)[1].split()[0]
    assert evaluator_hsl == initializer
    assert "--frontres_policy_quality_policy_checkpoint /tmp/frontres-v017-model-3500.pt" in command
    assert "--frontres_policy_quality_result /tmp/frontres-v017-quality.json" in command
    assert "--frontres_checkpoint_interval" not in command
    assert "--frontres_segment_live_update_loop_only" not in command

    missing = _run_preflight("policy_quality_eval", bounds=("12", "0", "1"))
    assert missing.returncode == 4
    assert "EVAL-v004 policy quality requires POLICY_QUALITY_MANIFEST" in missing.stderr
    wrong_rows = _run_preflight("policy_quality_eval", quality_env, bounds=("8", "0", "1"))
    assert wrong_rows.returncode == 4
    assert "EVAL-v004 K16/M3 policy quality requires NUM_ENVS=12" in wrong_rows.stderr


def test_stage3_launch_rejects_retired_optimizer_modes() -> None:
    for mode in ("single_update", "update_loop"):
        result = _run_preflight(mode)
        assert result.returncode == 4
        assert "FRS-PPO-v005 rejects retired optimizer-writing Stage 3 mode" in result.stderr
        assert "Command: " not in result.stdout


if __name__ == "__main__":
    test_stage3_train_launch_preflight_builds_femr_command()
    test_g5_s4_bounded_launch_freezes_8_1_1_and_audit()
    test_stage3_train_launch_accepts_explicit_checkpoint_interval()
    test_g5_s4_launch_rejects_legacy_resume_and_wrong_bounds()
    test_strict_v9_resume_replaces_hsl_initializer()
    test_strict_v9_resume_rejects_missing_checkpoint()
    test_stage3_diagnostic_launch_preflight_adds_only_selected_sentinel()
    test_stage3_train_launch_passes_explicit_segment_ppo_schedule_and_lr_args()
    test_stage3_launch_rejects_retired_optimizer_modes()
    test_stage3_launch_rejects_legacy_local_evaluation_modes()
    test_stage3_launch_builds_active_v017_policy_quality_command()
    test_stage3_launch_rejects_unknown_mode_before_training()
    print("frontres_segment_stage3_launch_command_contract: ok")
