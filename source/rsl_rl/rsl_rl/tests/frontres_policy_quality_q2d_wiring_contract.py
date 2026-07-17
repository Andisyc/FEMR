"""Static connectivity contract for the isolated Q2-D real-owner evaluator."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
Q2D = ROOT / "source/rsl_rl/rsl_rl/runners/frontres_policy_quality_q2d_eval.py"
OLD = ROOT / "source/rsl_rl/rsl_rl/runners/frontres_policy_quality_eval.py"
RUNNER = ROOT / "source/rsl_rl/rsl_rl/runners/on_policy_runner.py"
TRAIN = ROOT / "scripts/rsl_rl/train.py"
SHELL = ROOT / "run/run_frontres_stage3_segment_hrl.sh"


def main() -> None:
    source = Q2D.read_text()
    old_source = OLD.read_text()
    runner_source = RUNNER.read_text()
    train_source = TRAIN.read_text()
    shell_source = SHELL.read_text()
    assert "def run_frontres_policy_quality_q2d_scale_eval(" in source
    assert "build_frontres_policy_quality_formal_owner_bundle" in source
    assert "restore_frontres_policy_quality_state" in source
    assert "run_q2d_scale_sweep(" in source
    assert "hooks.compute_gain" in source and "hooks.capture_execution" in source
    assert "install_frontres_policy_quality_manifest_executor" not in source
    assert "run_frontres_policy_quality_q2d_scale_eval" not in old_source
    assert '"frontres_policy_quality_q2d_scale_result_v1"' in source
    assert "def run_frontres_policy_quality_q2d_eval(" in runner_source
    assert "--frontres_policy_quality_q2d_eval_only" in train_source
    assert "runner.run_frontres_policy_quality_q2d_eval(" in train_source
    assert "policy_quality_q2d_eval)" in shell_source
    assert "POLICY_QUALITY_Q2D_RESULT" in shell_source
    print("PASS: isolated Q2-D evaluator reaches canonical owners without modifying old eval control flow.")


if __name__ == "__main__":
    main()
