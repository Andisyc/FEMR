"""
Batch launcher for decoupled robustness validation.

Expected motion layout:

    motion_root/
      Walking/*.npz
      Turning/*.npz
      Upper/*.npz
      Lateral/*.npz

Each motion is run as an independent experiment unit and saved to:

    output_dir/
      run_meta.json
      motions/<group>/<motion_stem>/{meta.json,results_raw.npz,summary.csv,status.json}

Pass Isaac/AppLauncher arguments after the normal args; they are forwarded to
run_validation.py unchanged.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


DEFAULT_GROUPS = ["Walking", "Turning", "Upper", "Lateral"]


def _is_completed(path: Path) -> bool:
    status_path = path / "status.json"
    if not status_path.is_file():
        return False
    try:
        return json.loads(status_path.read_text()).get("status") == "completed"
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch robustness validation by motion group.")
    parser.add_argument("--motion_root", type=str, required=True,
                        help="Directory containing Walking/Turning/Upper/Lateral subdirectories.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="GMT checkpoint passed to run_validation.py.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Run root. Defaults to verify/robustness_validation/run_YYYYMMDD_HHMMSS.")
    parser.add_argument("--groups", type=str, nargs="+", default=DEFAULT_GROUPS)
    parser.add_argument("--file_glob", type=str, default="*.npz")
    parser.add_argument("--num_trials", type=int, default=30)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--perturbation_modes", type=str, nargs="+",
                        default=["composite", "xy", "yaw", "z", "rp"])
    parser.add_argument("--skip_completed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop_on_failure", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plot_after", action=argparse.BooleanOptionalAction, default=True,
                        help="Generate aggregate Fig1/Fig2 after the batch finishes.")
    args, passthrough = parser.parse_known_args()

    motion_root = Path(args.motion_root).expanduser().resolve()
    if not motion_root.is_dir():
        raise FileNotFoundError(f"motion_root not found: {motion_root}")

    if args.output_dir is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("verify/robustness_validation") / f"run_{stamp}"
    else:
        output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "motion_root": str(motion_root),
        "checkpoint": args.checkpoint,
        "groups": args.groups,
        "file_glob": args.file_glob,
        "n_trials": args.num_trials,
        "num_envs": args.num_envs,
        "perturbation_modes": args.perturbation_modes,
    }
    (output_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    validation_script = Path(__file__).with_name("run_validation.py")
    jobs: list[tuple[str, Path, Path]] = []
    for group in args.groups:
        group_dir = motion_root / group
        if not group_dir.is_dir():
            print(f"[batch] Missing group directory, skipping: {group_dir}", flush=True)
            continue
        for motion_path in sorted(group_dir.glob(args.file_glob)):
            if not motion_path.is_file():
                continue
            motion_out = output_dir / "motions" / group / motion_path.stem
            jobs.append((group, motion_path, motion_out))

    print(f"[batch] Found {len(jobs)} motion jobs. Output: {output_dir}", flush=True)
    failures = 0

    for idx, (group, motion_path, motion_out) in enumerate(jobs, start=1):
        if args.skip_completed and _is_completed(motion_out):
            print(f"[batch] [{idx}/{len(jobs)}] skip completed: {group}/{motion_path.name}", flush=True)
            continue

        motion_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(validation_script),
            "--motion", str(motion_path),
            "--checkpoint", args.checkpoint,
            "--file_glob", motion_path.name,
            "--num_trials", str(args.num_trials),
            "--num_envs", str(args.num_envs),
            "--output_dir", str(motion_out),
            "--no_timestamp",
            "--no_auto_plot",
            "--motion_group", group,
            "--motion_name", motion_path.stem,
            "--perturbation_modes", *args.perturbation_modes,
            *passthrough,
        ]

        print(f"[batch] [{idx}/{len(jobs)}] running: {group}/{motion_path.name}", flush=True)
        completed = subprocess.run(cmd, cwd=Path.cwd())
        if completed.returncode != 0:
            failures += 1
            status = {
                "status": "failed",
                "motion_group": group,
                "motion_name": motion_path.stem,
                "motion": str(motion_path),
                "returncode": completed.returncode,
                "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            }
            (motion_out / "status.json").write_text(json.dumps(status, indent=2))
            print(f"[batch] FAILED: {group}/{motion_path.name} returncode={completed.returncode}", flush=True)
            if args.stop_on_failure:
                return completed.returncode

    if args.plot_after:
        plot_script = Path(__file__).with_name("plot_results.py")
        print("[batch] Generating aggregate figures...", flush=True)
        subprocess.run([sys.executable, str(plot_script), "--results_dir", str(output_dir)], cwd=Path.cwd())

    print(f"[batch] Done. failures={failures}, output={output_dir}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
