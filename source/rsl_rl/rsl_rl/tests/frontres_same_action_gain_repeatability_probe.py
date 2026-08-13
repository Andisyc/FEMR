#!/usr/bin/env python3
"""Run four isolated local sentinels and compare final serialized Gain evidence.

Status: test-only ALTERNATE-PATH entry.
Upstream: explicit CLI invocation by a human-approved bounded diagnostic.
Downstream: four official Stage-3 launcher processes and one immutable JSON report.
Evidence: offline contract-confirmed; real repeatability remains BOUNDED-LIVE.
Gap: this entry does not prove formal training reachability or policy quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from frontres_same_action_gain_repeatability import (
    ProbeInputError,
    REPEAT_COUNT,
    compare_repeats,
    parse_live_snapshot,
)


@dataclass(frozen=True)
class ProbeConfig:
    """Validated immutable inputs for one four-process diagnostic."""

    repo_root: Path
    checkpoint: Path
    motion_path: Path
    cache_dir: Path
    k_schedule: str
    output_dir: Path
    seed: int

    def validate(self) -> None:
        repo_root = self.repo_root.resolve()
        if not repo_root.is_dir():
            raise ProbeInputError(f"repo root is not a directory: {repo_root}")
        launcher = repo_root / "run" / "run_frontres_stage3_segment_hrl.sh"
        if not launcher.is_file():
            raise ProbeInputError(f"official Stage-3 launcher is missing: {launcher}")
        if not self.checkpoint.resolve().is_file():
            raise ProbeInputError(f"checkpoint is not a file: {self.checkpoint}")
        if not self.motion_path.resolve().is_dir():
            raise ProbeInputError(f"motion path is not a directory: {self.motion_path}")
        if not self.cache_dir.resolve().is_dir():
            raise ProbeInputError(f"Segment cache path is not a directory: {self.cache_dir}")
        if not self.k_schedule.strip():
            raise ProbeInputError("the active explicit K/M/DR schedule is required")
        if self.seed < 0:
            raise ProbeInputError("seed must be non-negative")


def build_child_command(config: ProbeConfig) -> list[str]:
    """Build the argv-only official launcher call for one local sentinel."""

    launcher = config.repo_root.resolve() / "run" / "run_frontres_stage3_segment_hrl.sh"
    return [
        "bash",
        str(launcher),
        str(config.checkpoint.resolve()),
        str(config.motion_path.resolve()),
        "64",
        "1",
        "1",
        "train",
        "--frontres_local_sentinel_only",
        "--seed",
        str(config.seed),
    ]


def build_child_environment(
    config: ProbeConfig,
    *,
    repeat_index: int,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Seal one child's output, resume, schedule and offline-logger identity."""

    if repeat_index not in range(REPEAT_COUNT):
        raise ProbeInputError(f"repeat_index must be in [0,{REPEAT_COUNT - 1}]")
    repeat_name = f"repeat_{repeat_index + 1:02d}"
    runtime_root = config.output_dir.resolve() / repeat_name / "runtime"
    env = dict(os.environ if base_env is None else base_env)
    env.update(
        {
            "FEMR_ROOT": str(config.repo_root.resolve()),
            "FEMR_LOG_ROOT": str(runtime_root),
            "RUN_NAME": f"frontres_same_action_gain_{repeat_name}",
            "WANDB_MODE": "offline",
            "WANDB_DIR": str(runtime_root),
            "WANDB_CACHE_DIR": str(runtime_root / ".wandb_cache"),
            "FRONTRES_TMPDIR": str(runtime_root / "tmp"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "CACHE_DIR": str(config.cache_dir.resolve()),
            "FRONTRES_V015_K_CURRICULUM": config.k_schedule,
            "FRONTRES_V015_RESUME_CHECKPOINT": str(config.checkpoint.resolve()),
            "FRONTRES_STAGE3_RUN_CONTRACTS": "0",
            "FRONTRES_G5_S4_BOUNDED": "0",
            "NPROC_PER_NODE": "1",
        }
    )
    return env


def prepare_output_directory(config: ProbeConfig) -> Path:
    """Validate immutable inputs and create one new fail-if-present output root."""

    config.validate()
    output_dir = config.output_dir.resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ProbeInputError(f"output directory already exists: {output_dir}") from exc
    return output_dir


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish the final report only through one atomic rename."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_child(config: ProbeConfig, *, repeat_index: int, log_path: Path) -> list[str]:
    command = build_child_command(config)
    env = build_child_environment(config, repeat_index=repeat_index)
    with log_path.open("x", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=config.repo_root.resolve(),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            return_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if return_code != 0:
        raise RuntimeError(f"repeat {repeat_index + 1} exited with code {return_code}; see {log_path}")
    return command


def run_probe(config: ProbeConfig) -> tuple[Path, str]:
    """Run four independent children, gate identities, then publish one report."""

    # B1: 冻结输入 checkpoint 和新输出根, 产出不可覆盖的本次 probe identity.
    output_dir = prepare_output_directory(config)
    checkpoint = config.checkpoint.resolve()
    checkpoint_sha256 = _sha256_file(checkpoint)
    snapshots: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []

    # B2: 顺序运行四个隔离 child, 每次只消费同一 checkpoint 并解析最终 serializer.
    for repeat_index in range(REPEAT_COUNT):
        repeat_dir = output_dir / f"repeat_{repeat_index + 1:02d}"
        repeat_dir.mkdir()
        log_path = repeat_dir / "child.log"
        command = _run_child(config, repeat_index=repeat_index, log_path=log_path)
        if _sha256_file(checkpoint) != checkpoint_sha256:
            raise RuntimeError("input checkpoint changed during the probe")
        snapshots.append(parse_live_snapshot(log_path.read_text(encoding="utf-8"), source=str(log_path)))
        children.append(
            {
                "repeat": repeat_index + 1,
                "command": command,
                "log_path": str(log_path),
            }
        )

    # B3: 先执行 exact identity/action gate, 再原子发布描述性比较.
    comparison = compare_repeats(snapshots)
    report = {
        "schema_version": "frontres-same-action-gain-repeatability/v1",
        "probe_id": "FRS-SAME-ACTION-GAIN-REPEATABILITY-v1",
        "path_class": "ALTERNATE-PATH",
        "evidence_level": "BOUNDED-LIVE",
        "inputs": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "motion_path": str(config.motion_path.resolve()),
            "cache_dir": str(config.cache_dir.resolve()),
            "k_schedule": config.k_schedule,
            "seed": config.seed,
            "repeat_count": REPEAT_COUNT,
        },
        "children": children,
        **comparison,
    }
    report_path = output_dir / "repeatability_report.json"
    write_json_atomic(report_path, report)
    return report_path, str(comparison["status"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run four isolated official local sentinels and compare same-action Gain evidence."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--motion-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--k-schedule", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[4])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = ProbeConfig(
        repo_root=args.repo_root,
        checkpoint=args.checkpoint,
        motion_path=args.motion_path,
        cache_dir=args.cache_dir,
        k_schedule=args.k_schedule,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    try:
        report_path, status = run_probe(config)
    except (OSError, ProbeInputError, RuntimeError, KeyboardInterrupt) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    print(f"frontres_same_action_gain_repeatability_probe: {status} report={report_path}", flush=True)
    return 0 if status == "DESCRIPTIVE_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
