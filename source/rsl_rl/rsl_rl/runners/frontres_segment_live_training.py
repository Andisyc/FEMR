from __future__ import annotations

import json
import os
import math
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import torch

from rsl_rl.frontres.frontres_interfaces import FRONTRES_CHECKPOINT_FORMAT, FrontRESActiveTelemetryView
from rsl_rl.frontres.frontres_segment_reporting import (
    action_distribution_health_summary,
    format_segment_motion_quality_log,
    format_segment_train_effect_log,
    motion_quality_summary_to_scalars,
)
from rsl_rl.frontres.frontres_segment_warmup import FRONTRES_V011_MAX_ABSOLUTE_ITERATION
from rsl_rl.runners.frontres_formal_runtime_audit import print_formal_route_audit
from rsl_rl.runners.frontres_checkpoint_quality import load_frontres_checkpoint_mapping
from rsl_rl.runners.frontres_segment_live_reset import apply_frontres_current_segment_reset
from rsl_rl.runners.frontres_segment_live_rollout import run_frontres_live_rollout_capture
from rsl_rl.runners.frontres_segment_live_storage import capture_frontres_paired_gain
from rsl_rl.runners.frontres_segment_one_action_k import read_frontres_live_observations
from rsl_rl.runners.frontres_segment_training_telemetry import (
    build_frontres_formal_update_summary as _v015_formal_update_summary,
    build_frontres_transaction_telemetry,
    require_frontres_committed_result as _require_v015_committed_result,
)

_v015_sealed_transaction_telemetry = build_frontres_transaction_telemetry

from rsl_rl.runners.frontres_segment_live_sampler import (
    build_frontres_current_segment_batch,
    sample_frontres_live_segment_rows,
)


_REQUIRED_SUMMARY_KEYS = (
    "update_steps",
    "update_count",
    "ppo_valid_count",
    "reward_mean",
    "storage_valid_frac",
    "ppo_total_loss_mean",
    "ppo_actor_loss_mean",
    "ppo_value_loss_mean",
    "ppo_approx_kl_mean",
    "ppo_clip_frac_mean",
)

_FINITE_SUMMARY_KEYS = (
    "reward_mean",
    "storage_valid_frac",
    "ppo_total_loss_mean",
    "ppo_actor_loss_mean",
    "ppo_value_loss_mean",
    "ppo_approx_kl_mean",
    "ppo_clip_frac_mean",
)

_LOG_SEPARATOR = "-" * 80
def _fmt_num(value: Any) -> str:
    value = float(value)
    if not math.isfinite(value):
        return str(value)
    abs_value = abs(value)
    if abs_value != 0.0 and (abs_value >= 10000.0 or abs_value < 0.001):
        return f"{value:.3e}"
    return f"{value:.6f}"


def _fmt_pct(value: Any) -> str:
    return f"{100.0 * float(value):.1f}%"


def _live_train_status(summary: Mapping[str, Any]) -> str:
    if int(
        summary.get(
            "ppo_trust_region_rejected_count_sum",
            summary.get("ppo_trust_region_rejected_count", 0),
        )
        or 0
    ) > 0:
        return "WARN_TRUST_REGION_REJECTED"
    total_loss = float(summary["ppo_total_loss_mean"])
    actor_loss = float(summary["ppo_actor_loss_mean"])
    approx_kl = float(summary["ppo_approx_kl_mean"])
    clip_frac = float(summary["ppo_clip_frac_mean"])
    if not all(math.isfinite(v) for v in (total_loss, actor_loss, approx_kl, clip_frac)):
        return "BAD_NONFINITE"
    if abs(actor_loss) >= 1000.0 or abs(total_loss) >= 1000.0:
        return "BAD_LOSS_EXPLOSION"
    if clip_frac >= 0.3:
        return "WARN_HIGH_CLIP"
    if approx_kl < -0.001:
        return "WARN_NEG_KL"
    return "OK"


def _print_live_train_summary(
    runner: Any,
    *,
    local_iteration: int,
    num_learning_iterations: int,
    summary: Mapping[str, Any],
) -> None:
    motion_scalars = motion_quality_summary_to_scalars()
    for key in (
        "segment/motion_mpjpe_repaired_clean",
        "segment/motion_mpjpe_noisy_clean",
        "segment/motion_vel_error_repaired_clean",
        "segment/motion_acc_error_repaired_clean",
        "segment/motion_delta_se_norm",
        "segment/motion_delta_z_up_frac",
    ):
        if key in summary:
            motion_scalars[key] = float(summary[key])
    motion_scalars["segment/motion_delta_se_norm"] = float(
        summary.get("motion_delta_se_norm", motion_scalars["segment/motion_delta_se_norm"])
    )
    motion_scalars["segment/motion_delta_z_up_frac"] = float(
        summary.get("motion_delta_z_up_frac", motion_scalars["segment/motion_delta_z_up_frac"])
    )
    print(
        "\n".join(
            (
                "",
                _LOG_SEPARATOR,
                "",
                "[FrontRES Segment Live Train]",
                "  progress: "
                f"absolute_iter={runner.current_learning_iteration} "
                f"local={local_iteration}/{num_learning_iterations} "
                f"updates={int(summary['update_count'])}/{int(summary['update_steps'])} "
                "runner_learn=True",
                "  data: "
                f"valid={int(summary['ppo_valid_count'])} "
                f"valid_frac={_fmt_pct(summary['storage_valid_frac'])} "
                f"train_reward={_fmt_num(summary.get('train_reward_mean', summary['reward_mean']))} "
                f"env_reward={_fmt_num(summary.get('env_reward_mean', summary['reward_mean']))} "
                f"gain_total={_fmt_num(summary.get('gain_total_mean', float('nan')))}",
                "  trial: "
                f"policy={int(summary.get('trial_policy_count', summary.get('ppo_boundary_policy_rows', 0)))} "
                f"search={int(summary.get('trial_search_count', summary.get('ppo_boundary_search_rows', 0)))} "
                f"evidence={int(summary.get('ppo_boundary_evidence_rows', 0))} "
                f"ppo_valid={int(summary.get('ppo_boundary_eligible_rows', summary.get('ppo_valid_count', 0)))} "
                f"search_evidence_only={int(summary.get('ppo_boundary_search_evidence_only_rows', 0))} "
                f"policy_invalid={int(summary.get('ppo_boundary_policy_invalid_rows', 0))} "
                f"valid_policy={_fmt_pct(summary.get('ppo_boundary_valid_policy_frac', 0.0))} "
                f"valid_evidence={_fmt_pct(summary.get('ppo_boundary_valid_evidence_frac', 0.0))}",
                "  sampler: "
                f"gain={_fmt_num(summary.get('sampler_update_gain_mean', 0.0))} "
                f"gain_pos={_fmt_pct(summary.get('sampler_update_gain_pos_frac', 0.0))} "
                f"useful={_fmt_num(summary.get('sampler_update_useful_mean', 0.0))} "
                f"replay_candidates={int(summary.get('sampler_update_replay_candidate_count', 0))} "
                f"priority={_fmt_num(summary.get('sampler_priority_mean', 0.0))} "
                f"pool={int(summary.get('sampler_replay_pool_size', 0))} "
                f"hopeless={_fmt_pct(summary.get('sampler_hopeless_frac', 0.0))}",
                "  ppo: "
                f"phase={summary.get('ppo_warmup_phase', 'joint')} "
                f"phase_iter={int(summary.get('ppo_warmup_phase_iteration', 0))} "
                f"actor_weight={_fmt_num(summary.get('ppo_actor_loss_weight', 1.0))} "
                f"loss_total={_fmt_num(summary['ppo_total_loss_mean'])} "
                f"actor={_fmt_num(summary['ppo_actor_loss_mean'])} "
                f"value={_fmt_num(summary['ppo_value_loss_mean'])} "
                f"kl={_fmt_num(summary['ppo_approx_kl_mean'])} "
                f"clip={_fmt_pct(summary['ppo_clip_frac_mean'])} "
                f"status={_live_train_status(summary)}",
                "  trust: "
                f"accepted={int(summary.get('ppo_trust_region_accepted_min', summary.get('ppo_trust_region_accepted', 1)))} "
                f"rejected={int(summary.get('ppo_trust_region_rejected_count_sum', summary.get('ppo_trust_region_rejected_count', 0)))} "
                f"lr_before={_fmt_num(summary.get('ppo_adaptive_lr_before_first', summary.get('ppo_adaptive_lr_before', 0.0)))} "
                f"lr_after={_fmt_num(summary.get('ppo_adaptive_lr_after_last', summary.get('ppo_adaptive_lr_after', 0.0)))} "
                f"desired_kl={_fmt_num(summary.get('ppo_adaptive_lr_desired_kl_mean', summary.get('ppo_adaptive_lr_desired_kl', 0.0)))} "
                f"schedule={summary.get('ppo_trust_region_schedule', 'unknown')} "
                f"rollback={bool(summary.get('ppo_trust_region_rollback_enabled_min', summary.get('ppo_trust_region_rollback_enabled', 0)))} "
                f"max_retries={int(summary.get('ppo_trust_region_max_retries_max', summary.get('ppo_trust_region_max_retries', 0)))} "
                f"pre_lr_before={_fmt_num(summary.get('ppo_mosaic_pre_step_adaptive_lr_before_first', summary.get('ppo_mosaic_pre_step_adaptive_lr_before', 0.0)))} "
                f"pre_lr_after={_fmt_num(summary.get('ppo_mosaic_pre_step_adaptive_lr_after_last', summary.get('ppo_mosaic_pre_step_adaptive_lr_after', 0.0)))} "
                f"pre_kl={_fmt_num(summary.get('ppo_mosaic_pre_step_adaptive_lr_kl_mean', 0.0))}",
                "  scale: "
                f"adv_top1={_fmt_pct(summary.get('ppo_advantage_abs_top1_frac_mean', 0.0))} "
                f"old_sigma_min={_fmt_num(summary.get('ppo_old_sigma_min', 0.0))} "
                f"sigma_min={_fmt_num(summary.get('ppo_sigma_min', 0.0))} "
                f"post_mean_delta_l2={_fmt_num(summary.get('ppo_post_update_mean_delta_l2_mean', 0.0))} "
                f"post_mean_delta_max={_fmt_num(summary.get('ppo_post_update_mean_delta_max_abs', 0.0))}",
                "",
                format_segment_train_effect_log(dict(summary)),
                "",
                format_segment_motion_quality_log(motion_scalars),
                "",
            )
        ),
        flush=True,
    )


def _validate_live_update_summary(summary: Mapping[str, Any]) -> None:
    missing = [key for key in _REQUIRED_SUMMARY_KEYS if key not in summary]
    if missing:
        raise KeyError(f"FrontRES Segment live update summary missing keys: {missing}")


def _read_live_guard_cfg(runner: Any) -> tuple[bool, int, bool]:
    alg = getattr(runner, "alg", None)
    fail_on_invalid = bool(getattr(alg, "frontres_segment_live_fail_on_invalid_update", True))
    min_valid_count = max(0, int(getattr(alg, "frontres_segment_live_min_valid_count", 1)))
    fail_on_nonfinite = bool(getattr(alg, "frontres_segment_live_fail_on_nonfinite", True))
    return fail_on_invalid, min_valid_count, fail_on_nonfinite


def _validate_live_update_values(
    summary: Mapping[str, Any],
    *,
    fail_on_invalid: bool,
    min_valid_count: int,
    fail_on_nonfinite: bool,
) -> None:
    if fail_on_nonfinite:
        for key in _FINITE_SUMMARY_KEYS:
            value = float(summary[key])
            if not math.isfinite(value):
                raise FloatingPointError(f"FrontRES Segment live update produced non-finite {key}: {value}")
    if not fail_on_invalid:
        return
    update_count = int(summary["update_count"])
    valid_count = int(summary["ppo_valid_count"])
    if update_count <= 0:
        raise RuntimeError("FrontRES Segment live update produced update_count=0.")
    if valid_count < min_valid_count:
        raise RuntimeError(
            "FrontRES Segment live update has too few valid PPO samples: "
            f"ppo_valid_count={valid_count}, min_valid_count={min_valid_count}."
        )


def _print_v015_formal_train_summary(
    runner: Any,
    *,
    local_iteration: int,
    num_learning_iterations: int,
    summary: Mapping[str, Any],
) -> None:
    telemetry = summary.get("frontres_transaction_telemetry")
    if not isinstance(telemetry, Mapping):
        raise RuntimeError("v015 formal train summary requires sealed transaction telemetry")
    print(
        "[FrontRES v015 Transaction Telemetry] "
        + json.dumps(dict(telemetry), sort_keys=True, separators=(",", ":"), allow_nan=False),
        flush=True,
    )
    print(
        "[FrontRES v015 Formal Train] "
        f"absolute_iter={runner.current_learning_iteration} "
        f"local={local_iteration}/{num_learning_iterations} "
        f"transaction={summary['transaction_id']} "
        f"segments={summary['segment_count']} attempts={summary['policy_attempt_count']} "
        f"valid={summary['ppo_valid_count']} grouped_attempts={summary['grouped_attempt_count']} "
        f"step_delta={summary['optimizer_step_delta']} committed=1",
        flush=True,
    )


def _path_inside_log_dir(path: str, log_dir: str | None) -> bool:
    if log_dir is None:
        return False
    abs_path = os.path.abspath(path)
    abs_log_dir = os.path.abspath(log_dir)
    try:
        return os.path.commonpath([abs_path, abs_log_dir]) == abs_log_dir
    except ValueError:
        return False


def _print_checkpoint_save_probe(runner: Any, checkpoint_path: str) -> None:
    print(
        "\n".join(
            (
                "",
                _LOG_SEPARATOR,
                "",
                "[FrontRES Segment Live Checkpoint]",
                "  save.status: OK",
                f"  save.path: {checkpoint_path}",
                f"  save.in_log_dir: {_path_inside_log_dir(checkpoint_path, runner.log_dir)}",
                f"  save.iteration: {int(getattr(runner, 'current_learning_iteration', 0))}",
                "  route.runner_learn: True",
                "",
            )
        ),
        flush=True,
    )


def _print_checkpoint_save_failure(runner: Any, checkpoint_path: str, exc: BaseException) -> None:
    print(
        "\n".join(
            (
                "",
                _LOG_SEPARATOR,
                "",
                "[FrontRES Segment Live Checkpoint]",
                "  save.status: FAILED",
                f"  save.path: {checkpoint_path}",
                f"  save.in_log_dir: {_path_inside_log_dir(checkpoint_path, runner.log_dir)}",
                f"  save.iteration: {int(getattr(runner, 'current_learning_iteration', 0))}",
                f"  error.type: {type(exc).__name__}",
                f"  error.message: {str(exc)[:240]}",
                "",
            )
        ),
        flush=True,
    )


def _save_live_checkpoint(
    runner: Any,
    *,
    checkpoint_path: str,
    summary: Mapping[str, Any],
    required: bool,
    expected_v015_transaction_id: str | None = None,
) -> bool:
    if expected_v015_transaction_id is not None:
        state = getattr(runner, "_frontres_checkpoint_transaction_state", None)
        receipt = state.get("receipt") if isinstance(state, Mapping) and state.get("state") == "committed" else None
        if (
            not isinstance(receipt, Mapping)
            or str(receipt.get("transaction_id", "")) != str(expected_v015_transaction_id)
            or int(receipt.get("optimizer_step_delta", -1)) != 1
        ):
            raise RuntimeError("v015 checkpoint trigger requires the matching exact-one committed transaction receipt")
    try:
        runner.save(checkpoint_path)
    except (OSError, RuntimeError) as exc:
        _print_checkpoint_save_failure(runner, checkpoint_path, exc)
        if required:
            raise
        return False
    _print_checkpoint_save_probe(runner, checkpoint_path)
    runner._record_frontres_checkpoint_probe(dict(summary), checkpoint_path)
    return True


def finalize_frontres_local_sentinel_checkpoint(runner: Any, result: Any) -> str:
    """Persist and verify the checkpoint-v9 produced by one local sentinel update."""

    summary = _require_v015_committed_result(runner, result)
    telemetry = summary.get("frontres_transaction_telemetry")
    if not isinstance(telemetry, Mapping):
        raise RuntimeError("v015 local sentinel checkpoint requires sealed transaction telemetry")
    existing = getattr(runner, "_frontres_local_sentinel_telemetry", None)
    sealed = existing.get("sealed_transaction_evidence") if isinstance(existing, Mapping) else None
    if not isinstance(sealed, Mapping) or sealed.get("transaction_id") != summary["transaction_id"]:
        raise RuntimeError("v015 local sentinel checkpoint requires the matching final serialized evidence")

    previous_iteration = int(getattr(runner, "current_learning_iteration", -1))
    if previous_iteration < 0 or int(telemetry.get("training_iteration", -1)) != previous_iteration:
        raise RuntimeError("v015 local sentinel checkpoint iteration is not adjacent to its committed update")
    log_dir = str(getattr(runner, "log_dir", "") or "")
    if not log_dir:
        raise RuntimeError("v015 local sentinel checkpoint requires the formal runner log directory")
    next_iteration = previous_iteration + 1
    checkpoint_path = os.path.join(log_dir, f"model_{next_iteration}.pt")
    if os.path.exists(checkpoint_path):
        raise RuntimeError("v015 local sentinel checkpoint refuses to overwrite an existing artifact")
    runner.current_learning_iteration = next_iteration
    try:
        _save_live_checkpoint(
            runner,
            checkpoint_path=checkpoint_path,
            summary=summary,
            required=True,
            expected_v015_transaction_id=summary["transaction_id"],
        )
        payload = load_frontres_checkpoint_mapping(checkpoint_path, map_location="cpu")
        identity = payload.get("frontres_v015_checkpoint_identity") if isinstance(payload, Mapping) else None
        if not isinstance(identity, Mapping):
            raise RuntimeError("v015 local sentinel checkpoint has no checkpoint-v9 identity")
        required_identity = {
            "format": "frontres-v017-checkpoint-v9",
            "method_contract_id": "FRS-METHOD-v017",
            "training_contract_id": "FRS-TRAIN-v014",
            "dr_curriculum_schema_id": "nested-k-dr-four-class-v1",
            "gain_contract_id": "FRS-GAIN-v007",
            "optimization_contract_id": "FRS-PPO-v005",
            "scalar_target_id": "clean-anchored-recovery-aware-gain-v1",
            "physics_schema_id": "clean-anchored-contact-zmp-survival-v1",
            "grouped_schema_id": "grouped-all-attempt-scalar-v1",
        }
        if any(identity.get(name) != value for name, value in required_identity.items()):
            raise RuntimeError("v015 local sentinel checkpoint contract identity drifted after serialization")
        physics_evidence = identity.get("physics_evidence")
        expected_physics = {
            "zmp_estimator_id": "contact-wrench-zmp-v1",
            "support_envelope_id": "clean-foot-pose-oriented-box-v1",
            "actual_contact_id": "contact-sensor-net-normal-force-threshold-v1",
            "expected_phase_id": "clean-foot-height-phase-v1",
        }
        if physics_evidence != expected_physics:
            raise RuntimeError("v015 local sentinel checkpoint Physics evidence identity drifted after serialization")
        transaction = identity.get("transaction")
        receipt = transaction.get("receipt") if isinstance(transaction, Mapping) else None
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("transaction_id") != summary["transaction_id"]
            or int(receipt.get("optimizer_step_delta", -1)) != 1
        ):
            raise RuntimeError("v015 local sentinel checkpoint lost its exact-one committed receipt")
        curriculum = identity.get("curriculum")
        if not isinstance(curriculum, Mapping) or int(curriculum.get("absolute_iteration", -1)) != next_iteration:
            raise RuntimeError("v015 local sentinel checkpoint lost its absolute iteration identity")
    except Exception:
        runner.current_learning_iteration = previous_iteration
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        raise

    checkpoint_evidence = {
        "path": checkpoint_path,
        "iteration": next_iteration,
        **required_identity,
        "physics_evidence": dict(physics_evidence),
        "transaction_id": summary["transaction_id"],
        "optimizer_step_delta": 1,
    }
    updated = dict(existing)
    updated["checkpoint_v6"] = checkpoint_evidence
    runner._frontres_local_sentinel_telemetry = updated
    print(
        "[FrontRES v015 Checkpoint Sentinel] "
        + json.dumps(checkpoint_evidence, sort_keys=True, allow_nan=False),
        flush=True,
    )
    return checkpoint_path


def _print_resume_probe(runner: Any) -> None:
    loaded_checkpoint_path = getattr(runner, "_frontres_last_loaded_checkpoint_path", None)
    if loaded_checkpoint_path is None:
        return
    print(
        "[FrontRES Segment Live Resume] "
        f"loaded_checkpoint_path={loaded_checkpoint_path} "
        f"resumed_iteration={int(getattr(runner, 'current_learning_iteration', 0))} "
        "runner_learn=True "
        "legacy_runner_learn=False",
        flush=True,
    )


def run_frontres_segment_live_training_loop(
    runner: Any,
    *,
    num_learning_iterations: int,
    init_at_random_ep_len: bool = True,
) -> None:
    """运行正式 Stage 3 Segment Replay 训练迭代.

    函数名说明:
        `run_frontres_segment_live_training_loop` 是正式训练 route owner, 负责把
        已完成配置的 runner 送入重复的 live update; 它不是单次 rollout,
        eval 或离线 probe 入口.

    主链路:
        上游: `train.py` 在 `MODE=train` 且 live train 开启时调用本函数.
        下游: v015 每轮调用 formal transaction owner; legacy 配置才调用旧
        update loop. 只有 committed exact-one result 可递增 iteration 或保存.

    语义:
        进入本函数意味着正式 Stage 3 路由已唯一确定. 任何 probe/eval 路径
        都不得伪装成该训练循环.
    """
    boundary = getattr(runner, "_frontres_segment_replay_boundary", None)
    if not bool(getattr(boundary, "live_train_enabled", False)):
        raise ValueError("FrontRES Segment live training requires frontres_segment_live_train_enabled=True.")
    formal_v015 = bool(getattr(getattr(runner, "alg", None), "frontres_formal_transaction_enabled", False))
    # B2: 冻结 live update loop 将消费的正式 iteration budget.
    num_learning_iterations = max(0, int(num_learning_iterations))
    absolute_start = int(getattr(runner, "current_learning_iteration", 0))
    if formal_v015 and absolute_start + num_learning_iterations > FRONTRES_V011_MAX_ABSOLUTE_ITERATION:
        raise RuntimeError(
            "FRS-TRAIN-v014 run would cross maximum_absolute_iteration=8000: "
            f"start={absolute_start} requested={num_learning_iterations}"
        )
    # B3: 首次正式 update iteration 前截获 route identity.
    # AUDIT-ROUTE-01: 检查正式 Stage 3 路由, 位于 train dispatch -> live iteration loop.
    # Result: E70 LIVE PASS. 正式路径从 model_200 进入 absolute iter 700 joint
    # phase, actor_weight=1.0, 完成 4/4 accepted update 并保存 model_701;
    # GMT 仍冻结. 该证据只关闭 runtime connectivity, 不证明 policy quality.
    print_formal_route_audit(runner, num_learning_iterations=num_learning_iterations)
    if num_learning_iterations == 0:
        print(
            "[FrontRES Segment Live Train] "
            "num_learning_iterations=0 update_count=0 runner_learn=True",
            flush=True,
        )
        return

    _print_resume_probe(runner)
    last_checkpoint_probe_path: str | None = None

    for local_iteration in range(num_learning_iterations):
        if formal_v015:
            result = runner.run_frontres_formal_training_transaction(
                init_at_random_ep_len=bool(init_at_random_ep_len and local_iteration == 0),
            )
            summary = _require_v015_committed_result(runner, result)
        else:
            summary = runner.run_frontres_segment_live_update_loop(
                init_at_random_ep_len=bool(init_at_random_ep_len and local_iteration == 0),
                runner_learn=True,
            )
            _validate_live_update_summary(summary)
            fail_on_invalid, min_valid_count, fail_on_nonfinite = _read_live_guard_cfg(runner)
            _validate_live_update_values(
                summary,
                fail_on_invalid=fail_on_invalid,
                min_valid_count=min_valid_count,
                fail_on_nonfinite=fail_on_nonfinite,
            )
        runner.current_learning_iteration += 1
        if formal_v015:
            _print_v015_formal_train_summary(
                runner,
                local_iteration=local_iteration + 1,
                num_learning_iterations=num_learning_iterations,
                summary=summary,
            )
        else:
            _print_live_train_summary(
                runner,
                local_iteration=local_iteration + 1,
                num_learning_iterations=num_learning_iterations,
                summary=summary,
            )
        if (
            runner.log_dir is not None
            and not runner.disable_logs
            and runner.save_interval > 0
            and runner.current_learning_iteration % runner.save_interval == 0
        ):
            checkpoint_path = os.path.join(runner.log_dir, f"model_{runner.current_learning_iteration}.pt")
            if _save_live_checkpoint(
                runner,
                checkpoint_path=checkpoint_path,
                summary=summary,
                required=False,
                expected_v015_transaction_id=(summary["transaction_id"] if formal_v015 else None),
            ):
                last_checkpoint_probe_path = checkpoint_path

    if runner.log_dir is not None and not runner.disable_logs:
        final_checkpoint_path = os.path.join(runner.log_dir, f"model_{runner.current_learning_iteration}.pt")
        if final_checkpoint_path != last_checkpoint_probe_path:
            _save_live_checkpoint(
                runner,
                checkpoint_path=final_checkpoint_path,
                summary=summary,
                required=True,
                expected_v015_transaction_id=(summary["transaction_id"] if formal_v015 else None),
            )
