#!/usr/bin/env python3
"""Offline contract for the independent same-action Gain repeatability probe."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path


PROBE_PATH = Path(__file__).with_name("frontres_same_action_gain_repeatability_probe.py")
CORE_PATH = Path(__file__).with_name("frontres_same_action_gain_repeatability.py")


def _load_probe():
    assert PROBE_PATH.is_file(), "same-action Gain repeatability probe entry is missing"
    spec = importlib.util.spec_from_file_location("frontres_same_action_gain_repeatability_probe", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _snapshot(
    *,
    gain_shift: float = 0.0,
    action_shift: float = 0.0,
    identity_suffix: str = "",
    rank_flip: bool = False,
) -> dict:
    source_index = tuple(row // 4 for row in range(32))
    trial_index = tuple(row % 4 for row in range(32))
    scenario_ids = tuple(f"scenario-{source}{identity_suffix}" for source in source_index)
    noisy_hashes = tuple(f"hash-{source}{identity_suffix}" for source in source_index)
    x_t_identities = tuple(f"x-t-{source}{identity_suffix}" for source in source_index)
    policy_actions = tuple(
        (float(source), float(trial) + action_shift, 0.0, 0.0, 0.0, 0.0)
        for source, trial in zip(source_index, trial_index, strict=True)
    )
    gain_rows = [float(trial - 1) + gain_shift for trial in trial_index]
    if rank_flip:
        gain_rows[0], gain_rows[1] = gain_rows[1], gain_rows[0]
    gain_total = tuple(gain_rows)
    components = {
        "intent_gain": tuple(value + 0.1 for value in gain_total),
        "physics_gain": tuple(value + 0.2 for value in gain_total),
        "recovery_pressure": tuple(0.5 for _ in gain_total),
        "weighted_physics_gain": tuple(value + 0.3 for value in gain_total),
        "repair_cost": tuple(0.25 for _ in gain_total),
        "gain_total": gain_total,
        "raw_returns": gain_total,
        "utility_returns": tuple(math.copysign(math.log1p(abs(value)), value) for value in gain_total),
    }
    sealed = {
        "method_contract_id": "FRS-METHOD-v025",
        "gain_contract_id": "FRS-GAIN-v008",
        "optimization_contract_id": "FRS-PPO-v012",
        "training_contract_id": "FRS-TRAIN-v024",
        "checkpoint_format": "frontres-v024-checkpoint-v19",
        "active_k": 8,
        "active_m": 4,
        "selected_segment_count": 8,
        "policy_row_count": 32,
        "optimizer_step_delta": 1,
        "scenario_ids": scenario_ids,
        "noisy_segment_hashes": noisy_hashes,
        "source_index": source_index,
        "trial_index": trial_index,
        "valid_policy_row_mask": (True,) * 32,
        "policy_actions": policy_actions,
        **components,
    }
    return {
        "scenario_ids": scenario_ids,
        "noisy_segment_hashes": noisy_hashes,
        "x_t_identities": x_t_identities,
        "optimizer_step_delta": 1,
        "exact_one_update": True,
        "sealed_transaction_evidence": sealed,
    }


def _log(snapshot: dict) -> str:
    return "boot\n[FrontRES v017 Live Snapshot] " + json.dumps(snapshot) + "\nclosed\n"


def _expect_error(call, text: str) -> None:
    try:
        call()
    except (RuntimeError, ValueError) as exc:
        assert text.lower() in str(exc).lower(), str(exc)
        return
    raise AssertionError("expected fail-closed probe rejection")


def main() -> None:
    probe = _load_probe()

    first = probe.parse_live_snapshot(_log(_snapshot()), source="repeat-01")
    assert first["identity"]["source_index"] == tuple(row // 4 for row in range(32))
    assert first["identity"]["trial_index"] == tuple(row % 4 for row in range(32))
    assert first["components"]["gain_total"][0:4] == (-1.0, 0.0, 1.0, 2.0)
    assert first["policy_actions"][0] == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    _expect_error(lambda: probe.parse_live_snapshot("no snapshot", source="missing"), "exactly one")
    _expect_error(
        lambda: probe.parse_live_snapshot(_log(_snapshot()) + _log(_snapshot()), source="duplicate"),
        "exactly one",
    )
    malformed = _snapshot()
    malformed["sealed_transaction_evidence"].pop("gain_total")
    _expect_error(lambda: probe.parse_live_snapshot(_log(malformed), source="malformed"), "gain_total")
    malformed_x_t = _snapshot()
    malformed_x_t["x_t_identities"] = (None,) + malformed_x_t["x_t_identities"][1:]
    _expect_error(lambda: probe.parse_live_snapshot(_log(malformed_x_t), source="malformed-x-t"), "x_t_identities")
    malformed_component = _snapshot()
    malformed_component["sealed_transaction_evidence"]["gain_total"] = (
        None,
    ) + malformed_component["sealed_transaction_evidence"]["gain_total"][1:]
    _expect_error(
        lambda: probe.parse_live_snapshot(_log(malformed_component), source="malformed-component"),
        "gain_total",
    )
    wrong_contract = _snapshot()
    wrong_contract["sealed_transaction_evidence"]["gain_contract_id"] = "FRS-GAIN-v007"
    _expect_error(lambda: probe.parse_live_snapshot(_log(wrong_contract), source="wrong-contract"), "FRS-GAIN-v008")

    identity_invalid = [
        probe.parse_live_snapshot(_log(_snapshot(identity_suffix="-changed" if repeat == 3 else "")), source=str(repeat))
        for repeat in range(4)
    ]
    identity_report = probe.compare_repeats(identity_invalid)
    assert identity_report["status"] == "INVALID_IDENTITY"
    assert identity_report["conclusion_authorized"] is False
    assert "comparison" not in identity_report

    action_invalid = [
        probe.parse_live_snapshot(_log(_snapshot(action_shift=0.25 if repeat == 3 else 0.0)), source=str(repeat))
        for repeat in range(4)
    ]
    action_report = probe.compare_repeats(action_invalid)
    assert action_report["status"] == "INVALID_ACTION"
    assert action_report["conclusion_authorized"] is False
    assert "comparison" not in action_report

    valid = [
        probe.parse_live_snapshot(_log(_snapshot(gain_shift=shift)), source=f"repeat-{index + 1:02d}")
        for index, shift in enumerate((0.0, 0.5, -0.5, 1.0))
    ]
    report = probe.compare_repeats(valid)
    assert report["status"] == "DESCRIPTIVE_COMPLETE"
    assert report["conclusion_authorized"] is True
    assert report["identity_gate"]["canonical_identity"]["scenario_ids"][0] == "scenario-0"
    assert len(report["identity_gate"]["policy_actions"]) == 32
    gain_summary = report["comparison"]["field_summaries"]["gain_total"]
    assert gain_summary["max_absolute_range"] == 1.5
    assert gain_summary["sign_flip_row_count"] > 0
    assert report["comparison"]["scenario_m4_mean_spread"]["gain_total"]["0"] == 1.5
    assert report["comparison"]["within_m4_rank_changes"]["gain_total"] == []
    assert len(report["repeats"]) == 4

    rank_changed = [
        probe.parse_live_snapshot(_log(_snapshot(rank_flip=repeat == 3)), source=f"rank-{repeat}")
        for repeat in range(4)
    ]
    rank_report = probe.compare_repeats(rank_changed)
    assert rank_report["comparison"]["within_m4_rank_changes"]["gain_total"] == [0]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        checkpoint = root / "model_200.pt"
        checkpoint.write_bytes(b"trusted-checkpoint-fixture")
        motion = root / "motions"
        cache = root / "cache"
        motion.mkdir()
        cache.mkdir()
        output = root / "probe-output"
        config = probe.ProbeConfig(
            repo_root=Path(__file__).resolve().parents[4],
            checkpoint=checkpoint,
            motion_path=motion,
            cache_dir=cache,
            k_schedule="8,4,200,500,1300,lower-k8,0.5,linear-coupled-v1,700,2.381",
            output_dir=output,
            seed=17,
        )
        command = probe.build_child_command(config)
        assert command[0] == "bash"
        assert command[7] == "train"
        assert command[-3:] == ["--frontres_local_sentinel_only", "--seed", "17"]
        assert "--frontres_segment_live_sentinel_only" not in command
        env = probe.build_child_environment(config, repeat_index=1, base_env={"PATH": "/bin"})
        assert env["FRONTRES_V015_RESUME_CHECKPOINT"] == str(checkpoint.resolve())
        assert env["FEMR_LOG_ROOT"].endswith("repeat_02/runtime")
        assert env["RUN_NAME"].endswith("repeat_02")
        assert env["WANDB_MODE"] == "offline"
        assert env["WANDB_DIR"].endswith("repeat_02/runtime")
        assert env["WANDB_CACHE_DIR"].endswith("repeat_02/runtime/.wandb_cache")
        assert env["FRONTRES_TMPDIR"].endswith("repeat_02/runtime/tmp")
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"
        prepared = probe.prepare_output_directory(config)
        assert prepared == output.resolve()
        _expect_error(lambda: probe.prepare_output_directory(config), "already exists")
        report_path = output / "repeatability_report.json"
        probe.write_json_atomic(report_path, {"status": "DESCRIPTIVE_COMPLETE"})
        assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "DESCRIPTIVE_COMPLETE"
        assert not report_path.with_suffix(report_path.suffix + ".tmp").exists()

    for path in (PROBE_PATH, CORE_PATH):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, (ast.Import, ast.ImportFrom))
            and any(alias.name == "rsl_rl" or alias.name.startswith("rsl_rl.") for alias in node.names)
            for node in ast.walk(tree)
        ), "independent probe must not import the production package"
        assert not any(
            isinstance(node, ast.Call)
            and any(
                keyword.arg == "shell"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in node.keywords
            )
            for node in ast.walk(tree)
        ), "independent probe must never invoke a shell string"

    print("frontres_same_action_gain_repeatability_probe_contract: PASS", flush=True)


if __name__ == "__main__":
    main()
