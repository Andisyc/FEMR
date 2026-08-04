#!/usr/bin/env python3
"""Run one v015 deployment-composition sequence without training feedback.

Status: S4-S0 dedicated live entrypoint. The script constructs the formal
IsaacLab runner, loads the frozen GMT and v015 FEMR checkpoints, and dispatches
only the S2B evaluator. Live execution remains user-gated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


_V015_TASK = "FrontRES-Unified-Tracking-Flat-G1-v0"
_V015_LAYOUT_VERSION = "frontres-v015-future-intent-q29-v1"
_PREMATERIALIZED_SOURCE = "pre_materialized_deployment_npz"


@dataclass(frozen=True)
class FrontRESV015DeploymentCLIContract:
    """Validated server-facing identities for one bounded composition run."""

    task: str
    frontres_checkpoint: str
    gmt_checkpoint: str
    source_reference_npz: str
    reference_npz: str
    report_path: str
    future_offsets: tuple[int, ...]
    corruption_id: str
    corruption_family: str
    corruption_seed: int
    corruption_parameters: tuple[tuple[str, str | int | float | bool], ...]
    num_envs: int
    device: str
    cuda_visible_devices: str


def build_parser() -> argparse.ArgumentParser:
    # B1: 声明 v015-only CLI 参数, 产出不包含 training/resume 自由度的 parser.
    parser = argparse.ArgumentParser(
        description="Run one isolated v015 FEMR -> frozen GMT deployment-composition sequence."
    )
    parser.add_argument("--task", default=_V015_TASK)
    parser.add_argument("--frontres_checkpoint", required=True)
    parser.add_argument("--gmt_checkpoint", required=True)
    parser.add_argument("--source_reference_npz", required=True)
    parser.add_argument("--reference_npz", required=True)
    parser.add_argument("--report_path", required=True)
    parser.add_argument("--future_offsets", default="1,2")
    parser.add_argument("--corruption_id", required=True)
    parser.add_argument("--corruption_family", required=True)
    parser.add_argument("--corruption_seed", type=int, required=True)
    parser.add_argument(
        "--corruption_parameters_json",
        required=True,
        help="JSON scalar mapping describing the already materialized persistent corruption.",
    )
    parser.add_argument("--num_envs", type=int, default=2)
    return parser


def _parse_future_offsets(raw: str) -> tuple[int, ...]:
    # B1: 解析 ordered unique positive offsets, 产出 actor future-intent window identity.
    try:
        values = tuple(int(token.strip()) for token in str(raw).split(",") if token.strip())
    except ValueError as exc:
        raise ValueError("future_offsets must be comma-separated positive integers") from exc
    if not values or any(value <= 0 for value in values) or tuple(sorted(set(values))) != values:
        raise ValueError("future_offsets must be ordered unique positive integers")
    return values


def _parse_corruption_parameters(raw: str) -> tuple[tuple[str, str | int | float | bool], ...]:
    # B1: 解析并规范化 JSON scalar mapping, 产出 deterministic corruption parameters.
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("corruption_parameters_json must be a JSON object") from exc
    if not isinstance(values, dict):
        raise ValueError("corruption_parameters_json must be a JSON object")
    normalized: dict[str, str | int | float | bool] = {}
    for raw_name, value in values.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("corruption parameter names must be nonempty strings")
        if type(value) not in (str, int, float, bool):
            raise ValueError("corruption parameter values must be scalar str/int/float/bool")
        normalized[raw_name] = value
    declared_source = normalized.get("source")
    if declared_source is not None and declared_source != _PREMATERIALIZED_SOURCE:
        raise ValueError("deployment CLI rejects non-prematerialized corruption sources")
    normalized["source"] = _PREMATERIALIZED_SOURCE
    return tuple(sorted(normalized.items()))


def _absolute_input_file(raw: str, *, suffix: str, name: str) -> str:
    # B1: 解析绝对 artifact path 并校验 suffix/existence, 产出 strict input identity.
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.suffix.lower() != suffix:
        raise ValueError(f"{name} must be an existing {suffix} file")
    return str(resolved)


def _absolute_new_report(raw: str) -> str:
    # B1: 解析绝对 report path, 拒绝覆盖与缺失 parent, 产出 atomic output boundary.
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        raise ValueError("report_path must be an absolute path")
    if path.suffix.lower() != ".json" or not path.parent.is_dir():
        raise ValueError("report_path must be a new .json under an existing directory")
    if path.exists():
        raise ValueError("report_path already exists; v015 report identity is immutable")
    return str(path.resolve(strict=False))


def _validate_cuda_dispatch(device: str, environ: Mapping[str, str]) -> str:
    # B1: 对齐 CUDA_VISIBLE_DEVICES 与 local cuda:0, 产出显卡选择 identity.
    visible = str(environ.get("CUDA_VISIBLE_DEVICES", "") or "").strip()
    if not str(device).startswith("cuda"):
        raise ValueError("v015 live composition requires a CUDA device")
    if not visible:
        raise ValueError("set CUDA_VISIBLE_DEVICES before invoking the v015 deployment CLI")
    tokens = tuple(token.strip() for token in visible.split(",") if token.strip())
    if not tokens:
        raise ValueError("CUDA_VISIBLE_DEVICES must expose at least one GPU")
    parts = str(device).split(":", maxsplit=1)
    visible_index = 0 if len(parts) == 1 else int(parts[1])
    if visible_index < 0 or visible_index >= len(tokens):
        raise ValueError(
            f"device {device!r} is outside the {len(tokens)} GPU(s) exposed by CUDA_VISIBLE_DEVICES"
        )
    return visible


def validate_frontres_v015_deployment_cli_args(
    args: Any,
    *,
    environ: Mapping[str, str] | None = None,
) -> FrontRESV015DeploymentCLIContract:
    # B1: 校验 task, CUDA, env count 与 artifact paths, 产出 strict CLI identity fields.
    """Fail closed before AppLauncher imports or simulator construction."""

    if str(args.task) != _V015_TASK:
        raise ValueError(f"v015 deployment CLI accepts only task {_V015_TASK!r}")
    num_envs = int(args.num_envs)
    if num_envs <= 0 or num_envs % 2 != 0:
        raise ValueError("v015 deployment CLI requires a positive even num_envs for aligned rows")
    device = str(args.device)
    visible = _validate_cuda_dispatch(device, os.environ if environ is None else environ)
    corruption_id = str(args.corruption_id).strip()
    corruption_family = str(args.corruption_family).strip()
    if not corruption_id or not corruption_family:
        raise ValueError("corruption_id and corruption_family must be nonempty")
    # B2: 封装 corruption 与 future-context 参数, 产出不可变 deployment contract.
    return FrontRESV015DeploymentCLIContract(
        task=_V015_TASK,
        frontres_checkpoint=_absolute_input_file(
            args.frontres_checkpoint, suffix=".pt", name="frontres_checkpoint"
        ),
        gmt_checkpoint=_absolute_input_file(args.gmt_checkpoint, suffix=".pt", name="gmt_checkpoint"),
        source_reference_npz=_absolute_input_file(
            args.source_reference_npz, suffix=".npz", name="source_reference_npz"
        ),
        reference_npz=_absolute_input_file(args.reference_npz, suffix=".npz", name="reference_npz"),
        report_path=_absolute_new_report(args.report_path),
        future_offsets=_parse_future_offsets(args.future_offsets),
        corruption_id=corruption_id,
        corruption_family=corruption_family,
        corruption_seed=int(args.corruption_seed),
        corruption_parameters=_parse_corruption_parameters(args.corruption_parameters_json),
        num_envs=num_envs,
        device=device,
        cuda_visible_devices=visible,
    )


def _set_existing(owner: Any, name: str, value: Any) -> None:
    if owner is None or not hasattr(owner, name):
        raise AttributeError(f"v015 deployment config owner is missing field {name!r}")
    setattr(owner, name, value)


def configure_frontres_v015_deployment_agent_cfg(
    agent_cfg: Any,
    contract: FrontRESV015DeploymentCLIContract,
) -> None:
    # B1: 验证 strict CLI contract 并安装 observation/checkpoint/evaluation-only flags.
    """Install the existing v015 inference/checkpoint identity with all train modes off."""

    if not isinstance(contract, FrontRESV015DeploymentCLIContract):
        raise TypeError("v015 deployment agent config requires its validated CLI contract")
    policy_cfg = getattr(agent_cfg, "policy", None)
    alg_cfg = getattr(agent_cfg, "algorithm", None)
    _set_existing(policy_cfg, "gmt_checkpoint_path", contract.gmt_checkpoint)
    for name, value in (
        ("frontres_training_objective", "deployment_composition_eval"),
        ("frontres_segment_replay_enabled", False),
        ("frontres_segment_live_runner_enabled", False),
        ("frontres_segment_live_train_enabled", False),
        ("frontres_segment_live_sentinel_only", False),
        ("frontres_local_sentinel_only", False),
        ("frontres_segment_live_probe_only", False),
        ("frontres_segment_live_storage_write_only", False),
        ("frontres_segment_live_single_update_only", False),
        ("frontres_segment_live_update_loop_only", False),
        ("frontres_formal_transaction_enabled", True),
        ("frontres_segment_critic_warmup_iterations", 0),
        ("frontres_segment_actor_warmup_iterations", 0),
        ("frontres_future_offsets", contract.future_offsets),
        ("frontres_future_intent_layout_version", _V015_LAYOUT_VERSION),
        ("frontres_segment_advantage_normalization", "grouped_scale_only"),
        ("frontres_hsl_init_enabled", False),
        ("frontres_hsl_rollout_label_enabled", False),
        ("lambda_supervised", 0.0),
        ("lambda_supervised_min", 0.0),
    ):
        _set_existing(alg_cfg, name, value)
    for name, value in (
        ("device", contract.device),
        ("max_iterations", 0),
        ("resume", False),
        ("is_full_resume", True),
        ("supervised_warmup_iterations", 0),
        ("critic_warmup_iterations", 0),
    ):
        _set_existing(agent_cfg, name, value)


def build_frontres_v015_deployment_run_config(
    contract: FrontRESV015DeploymentCLIContract,
    *,
    sequence_module: Any | None = None,
):
    # B1: 从 CLI contract 构造 corruption protocol 与 immutable composition config.
    """Build the already accepted S1/S2B immutable evaluator config."""

    if sequence_module is None:
        from rsl_rl.runners import frontres_segment_sequence_eval as sequence_module

    protocol = sequence_module.build_frontres_v015_persistent_corruption_protocol(
        corruption_id=contract.corruption_id,
        family=contract.corruption_family,
        seed=contract.corruption_seed,
        parameters=dict(contract.corruption_parameters),
    )
    # B2: 绑定 artifact paths, offsets 与 report path, 产出 validated run config.
    config = sequence_module.FrontRESV015DeploymentCompositionRunConfig(
        request_config=sequence_module.FrontRESV015DeploymentCompositionConfig(
            enabled=True,
            source_reference_path=contract.source_reference_npz,
            reference_path=contract.reference_npz,
            future_offsets=contract.future_offsets,
            corruption_protocol=protocol,
        ),
        report_path=contract.report_path,
    )
    config.validate()
    return config


def _optimizer_step_count(runner: Any) -> int:
    optimizer = getattr(getattr(runner, "alg", None), "optimizer", None)
    value = getattr(optimizer, "frontres_step_count", None)
    value = value() if callable(value) else value
    if not isinstance(value, int) or value < 0:
        raise RuntimeError("v015 deployment CLI requires the persistent optimizer step counter")
    return value


def dispatch_frontres_v015_deployment_composition(runner: Any, run_config: Any):
    # B1: 解析 formal dispatch 与 optimizer counter, 产出执行前 mutation anchor.
    """Call only the S2B owner and reject any optimizer-step change."""

    dispatch = getattr(runner, "run_frontres_v015_deployment_composition_eval", None)
    if not callable(dispatch):
        raise RuntimeError("formal runner has no v015 deployment-composition dispatch")
    step_before = _optimizer_step_count(runner)
    # B2: 执行唯一 v015 composition route, 产出 paired deployment report.
    report = dispatch(config=run_config)
    step_after = _optimizer_step_count(runner)
    if step_after != step_before:
        raise RuntimeError(
            f"v015 deployment CLI observed a forbidden optimizer update: {step_before}->{step_after}"
        )
    return report


def format_frontres_v015_deployment_sentinel(
    report: Any,
    contract: FrontRESV015DeploymentCLIContract,
    *,
    optimizer_step_delta: int,
) -> str:
    # B1: 投影 report identity 与质量指标, 产出单行可搜索 runtime sentinel.
    no_feedback = not any(
        bool(getattr(report, name))
        for name in (
            "return_feedback",
            "priority_feedback",
            "ppo_feedback",
            "sampler_feedback",
            "optimizer_feedback",
        )
    )
    return (
        "[FrontRES v015 Deployment Composition S4] "
        f"frontres_checkpoint={contract.frontres_checkpoint} "
        f"reference_hash={report.request.reference_file_hash} "
        f"protocol_hash={report.request.corruption_protocol.protocol_hash} "
        f"reference_frames={report.reference_frame_count} "
        f"evaluated_frames={report.frame_count} "
        f"future_offsets={contract.future_offsets} "
        f"femr_actions={report.femr_action_count} "
        f"failures={report.accumulated_failure_count} "
        f"intent_q29_mean={report.mean_intent_q29_error:.6g} "
        f"contact_preservation={report.contact_preservation_fraction:.6g} "
        f"zmp_violations={report.phase_zmp_violation_count} "
        f"survival={report.survival_fraction:.6g} "
        f"max_cum_roll={report.max_abs_cumulative_lateral_roll_rad:.6g} "
        f"unplanned_contact_events={report.unplanned_contact_event_count} "
        f"optimizer_step_delta={int(optimizer_step_delta)} "
        f"no_feedback={no_feedback} "
        f"report={contract.report_path}"
    )


def _zero_motion_randomization(env_cfg: Any) -> None:
    # B1: 禁用 evaluation motion randomization, 保持 request/reference frame authority.
    motion_cfg = getattr(getattr(env_cfg, "commands", None), "motion", None)
    if motion_cfg is None:
        raise AttributeError("v015 deployment task has no commands.motion config")
    zero_ranges = {name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")}
    for name, value in (
        ("pose_range", dict(zero_ranges)),
        ("velocity_range", dict(zero_ranges)),
        ("joint_position_range", (0.0, 0.0)),
    ):
        _set_existing(motion_cfg, name, value)


def _run_with_hydra(contract: FrontRESV015DeploymentCLIContract, args_cli: Any) -> None:
    # B1: 冻结 Hydra task 与 CLI overrides, 产出 outer composition configuration.
    import gymnasium as gym

    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.hydra import hydra_task_config
    from rsl_rl.runners import OnPolicyRunner
    import whole_body_tracking.tasks  # noqa: F401
    from whole_body_tracking.tasks.tracking.mdp.motion_perturbations import MotionPerturbationCfg

    @hydra_task_config(contract.task, "rsl_rl_cfg_entry_point")
    def run(env_cfg: Any, agent_cfg: Any) -> None:
        # B2: 校验 reference identity 并配置 env/agent, 产出正式 runner inputs.
        configure_frontres_v015_deployment_agent_cfg(agent_cfg, contract)
        env_cfg.scene.num_envs = contract.num_envs
        env_cfg.sim.device = contract.device
        motion_cfg = env_cfg.commands.motion
        # The simulator starts from the Clean physical state. The sealed command
        # carrier installed by the evaluator remains the only Noisy reference.
        motion_cfg.motion = contract.source_reference_npz
        if hasattr(motion_cfg, "motion_file"):
            motion_cfg.motion_file = contract.source_reference_npz
        _set_existing(motion_cfg, "motion_horizon", 1)
        _set_existing(motion_cfg, "command_velocity", True)
        _set_existing(motion_cfg, "start_from_beginning", True)
        _set_existing(motion_cfg, "start_frame", 0)
        with np.load(contract.source_reference_npz, allow_pickle=False) as source_data:
            source_frames = int(np.asarray(source_data["joint_pos"]).shape[0])
            source_fps = float(np.asarray(source_data["fps"]).reshape(()))
        if source_frames <= 0 or not source_fps > 0.0:
            raise ValueError("v015 deployment source requires positive frames/fps")
        env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), source_frames / source_fps + 1.0)
        _zero_motion_randomization(env_cfg)
        if hasattr(env_cfg, "events"):
            env_cfg.events = None
        for group_name in ("policy", "teacher", "critic", "ref_vel_estimator"):
            group = getattr(getattr(env_cfg, "observations", None), group_name, None)
            if group is not None and hasattr(group, "enable_corruption"):
                group.enable_corruption = False
        if hasattr(env_cfg, "motion_perturbations"):
            env_cfg.motion_perturbations = MotionPerturbationCfg()

        # B3: 创建正式 env 与 runner, 严格加载 FEMR/GMT checkpoints.
        env = gym.make(contract.task, cfg=env_cfg, render_mode=None)
        try:
            if isinstance(env.unwrapped, DirectMARLEnv):
                env = multi_agent_to_single_agent(env)
            env = RslRlVecEnvWrapper(env)
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=contract.device)
            runner._move_normalizer_to_device(contract.device)
            runner.load(contract.frontres_checkpoint, load_optimizer=False, load_critic=False)
            runner.alg.policy.eval()
            report = dispatch_frontres_v015_deployment_composition(
                runner,
                build_frontres_v015_deployment_run_config(contract),
            )
            print(
                format_frontres_v015_deployment_sentinel(
                    report,
                    contract,
                    optimizer_step_delta=0,
                ),
                flush=True,
            )
        finally:
            env.close()

    run()


def main() -> None:
    # B1: 解析并验证用户参数, 产出 AppLauncher 可消费的 strict CLI contract.
    parser = build_parser()
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    contract = validate_frontres_v015_deployment_cli_args(args_cli)
    sys.argv = [sys.argv[0], *hydra_args]
    print(
        "[FrontRES v015 Deployment CLI Ready] "
        f"cuda_visible_devices={contract.cuda_visible_devices} device={contract.device} "
        f"task={contract.task} num_envs={contract.num_envs} "
        f"future_offsets={contract.future_offsets} report={contract.report_path}",
        flush=True,
    )
    # B2: 管理 IsaacLab app 生命周期, 在异常或成功后统一关闭 simulator app.
    app_launcher = AppLauncher(args_cli)
    try:
        _run_with_hydra(contract, args_cli)
    finally:
        app_launcher.app.close()


if __name__ == "__main__":
    main()
