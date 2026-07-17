"""Checkpoint, save, and resume helpers for OnPolicyRunner.

This module owns persistence mechanics. The runner keeps its public methods as
thin wrappers so training loops and external scripts keep the same API.
"""

from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
import shutil

import torch

from rsl_rl.modules import FrontRESActorCritic, ResidualActorCritic
from rsl_rl.modules.frontres_observation_layout import (
    compose_frontres_obs_norm_state,
    extract_frontres_extra_norm_stats,
    frontres_extra_norm_stats_for_save,
)
_FORMAL_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "frontres_formal_runtime_audit_checkpoint", Path(__file__).resolve().with_name("frontres_formal_runtime_audit.py")
)
_FORMAL_AUDIT_MODULE = importlib.util.module_from_spec(_FORMAL_AUDIT_SPEC)
assert _FORMAL_AUDIT_SPEC.loader is not None
_FORMAL_AUDIT_SPEC.loader.exec_module(_FORMAL_AUDIT_MODULE)
print_checkpoint_payload_audit = _FORMAL_AUDIT_MODULE.print_checkpoint_payload_audit
emit_formal_runtime_probe = _FORMAL_AUDIT_MODULE.emit_formal_runtime_probe
configure_formal_runtime_probe = _FORMAL_AUDIT_MODULE.configure_formal_runtime_probe


_FRONTRES_GAIN_CONFIG_FIELDS = (
    ("style_weight", "frontres_gain_style_weight", 1.0),
    ("physics_weight", "frontres_gain_physics_weight", 1.0),
    ("repair_weight", "frontres_gain_repair_weight", 0.15),
    ("mpjpe_scale", "frontres_gain_mpjpe_scale", 0.10),
    ("velocity_scale", "frontres_gain_velocity_scale", 1.0),
    ("acceleration_scale", "frontres_gain_acceleration_scale", 1.0),
    ("root_orientation_scale", "frontres_gain_root_orientation_scale", 1.0),
    ("repair_norm_scale", "frontres_gain_repair_norm_scale", 1.0),
    ("repair_temporal_scale", "frontres_gain_repair_temporal_scale", 1.0),
)


def _frontres_gain_config_payload(cfg) -> dict[str, object]:
    """Serialize the active FRS-GAIN-v002 scales for checkpoint identity."""
    values = {}
    for serialized_name, cfg_name, default in _FRONTRES_GAIN_CONFIG_FIELDS:
        if isinstance(cfg, dict):
            value = cfg.get(cfg_name, default)
        else:
            value = getattr(cfg, cfg_name, default)
        values[serialized_name] = float(value)
    return {
        "contract_id": "FRS-GAIN-v002",
        "values": values,
    }


def _validate_frontres_gain_config_resume(runner, checkpoint, *, is_full_resume: bool) -> None:
    """Reject full resume when the active Gain scale identity is absent or mismatched."""
    if str(getattr(runner, "training_type", "")) != "frontres":
        return
    checkpoint_config = checkpoint.get("frontres_gain_config")
    if checkpoint_config is None:
        if is_full_resume:
            raise RuntimeError(
                "full FrontRES resume requires frontres_gain_config in the checkpoint; "
                "refusing to resume with ambiguous Gain scales"
            )
        print(
            "[Runner] WARNING: checkpoint has no frontres_gain_config; "
            "using current config for Stage 2 -> Stage 3 initialization.",
            flush=True,
        )
        return
    expected = _frontres_gain_config_payload(getattr(runner, "cfg", None))
    if checkpoint_config != expected:
        raise RuntimeError(
            "FrontRES Gain config mismatch on resume: "
            f"checkpoint={checkpoint_config!r} current={expected!r}"
        )
    print("[Runner] Verified FRS-GAIN-v002 config identity on checkpoint resume.", flush=True)


# Full-resume diagnostic helper; uncomment with the probe prints below when needed.
# def _optimizer_state_debug(state_dict: dict | None) -> str:
#     if not isinstance(state_dict, dict):
#         return "missing"
#     groups = state_dict.get("param_groups", []) or []
#     state = state_dict.get("state", {}) or {}
#     lrs = []
#     param_counts = []
#     for group in groups:
#         if isinstance(group, dict):
#             lrs.append(group.get("lr"))
#             param_counts.append(len(group.get("params", []) or []))
#     return (
#         f"groups={len(groups)} state_entries={len(state)} "
#         f"group_param_counts={param_counts} group_lrs={lrs}"
#     )


def _copy_policy_noise_state(policy, model_state: dict) -> bool:
    """Load std/log_std only when checkpoint and runtime action dims match."""
    if hasattr(policy, "std") and "std" in model_state:
        source = model_state["std"].detach().to(device=policy.std.device, dtype=policy.std.dtype)
        if tuple(source.shape) == tuple(policy.std.shape):
            policy.std.data.copy_(source)
            return True
        print(
            "[Runner] Skipping checkpoint noise std due to action-dim drift: "
            f"checkpoint_shape={tuple(source.shape)} runtime_shape={tuple(policy.std.shape)}",
            flush=True,
        )
        return False
    if hasattr(policy, "log_std") and "log_std" in model_state:
        source = model_state["log_std"].detach().to(device=policy.log_std.device, dtype=policy.log_std.dtype)
        if tuple(source.shape) == tuple(policy.log_std.shape):
            policy.log_std.data.copy_(source)
            return True
        print(
            "[Runner] Skipping checkpoint log_std due to action-dim drift: "
            f"checkpoint_shape={tuple(source.shape)} runtime_shape={tuple(policy.log_std.shape)}",
            flush=True,
        )
        return False
    return False


def _reset_policy_noise_state(policy, *, init_noise_std: float, noise_std_type: str, device) -> None:
    """Reset runtime std/log_std using the current policy tensor shape."""
    if noise_std_type == "scalar" and hasattr(policy, "std"):
        policy.std.data.copy_(torch.ones_like(policy.std, device=device) * init_noise_std)
        print(f"[Runner] Reset noise std → {init_noise_std} shape={tuple(policy.std.shape)}")
    elif noise_std_type == "log" and hasattr(policy, "log_std"):
        policy.log_std.data.copy_(torch.log(torch.ones_like(policy.log_std, device=device) * init_noise_std))
        print(f"[Runner] Reset log_std → log({init_noise_std}) shape={tuple(policy.log_std.shape)}")


def record_frontres_checkpoint_probe(self, locs: dict, checkpoint_path: str) -> None:
    """Persist save-time FrontRES probe metrics and keep the best demo checkpoint.

    This is a lightweight checkpoint selector: it records the triplet
    rollout diagnostics already computed for the checkpoint iteration,
    without resetting the simulator or replaying the full training set.
    """
    if self.training_type != "frontres" or self.log_dir is None:
        return

    def _float(name: str, default: float | None = None) -> float | None:
        value = locs.get(name, default)
        if value is None:
            return default
        try:
            if isinstance(value, torch.Tensor):
                value = value.detach().mean().item()
            return float(value)
        except (TypeError, ValueError):
            return default

    restore_ratio = _float("frontres_restore_ratio_rp_mean")
    if restore_ratio is None:
        return

    residual = _float("frontres_residual_rp_abs_mean", 0.0) or 0.0
    roll_bias = _float("frontres_corr_roll_bias_mean", 0.0) or 0.0
    pitch_bias = _float("frontres_corr_pitch_bias_mean", 0.0) or 0.0
    harm_rate = _float("frontres_harm_rate_mean", 0.0) or 0.0
    harm_mag = _float("frontres_harm_mag_mean", 0.0) or 0.0
    survival = _float("frontres_survival_rate", 1.0)
    r_delta = _float("frontres_rdelta_mean", 0.0) or 0.0
    dr_scale = _float("frontres_dr_scale", None)

    bias_abs = abs(roll_bias) + abs(pitch_bias)
    survival_penalty = 0.0 if survival is None else max(0.0, 1.0 - survival)
    score = (
        restore_ratio
        - 0.25 * harm_rate
        - 2.0 * harm_mag
        - 0.50 * bias_abs
        - 0.10 * residual
        - 2.0 * survival_penalty
    )

    record = {
        "iteration": int(locs.get("it", self.current_learning_iteration)),
        "checkpoint": os.path.basename(checkpoint_path),
        "score": score,
        "restore_ratio_rp": restore_ratio,
        "residual_rp_abs": residual,
        "corr_roll_bias": roll_bias,
        "corr_pitch_bias": pitch_bias,
        "bias_abs": bias_abs,
        "harm_rate": harm_rate,
        "harm_mag": harm_mag,
        "survival_rate": survival,
        "r_delta": r_delta,
        "dr_scale": dr_scale,
        "perturb_modes": locs.get("frontres_perturb_modes"),
        "perturb_complexity": locs.get("frontres_perturb_complexity"),
    }

    probe_path = os.path.join(self.log_dir, "frontres_checkpoint_probe.jsonl")
    with open(probe_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")

    if self.writer is not None and not self.disable_logs:
        self.writer.add_scalar("FrontRES/CheckpointProbe/demo_score", score, record["iteration"])
        self.writer.add_scalar("FrontRES/CheckpointProbe/restore_ratio_rp", restore_ratio, record["iteration"])
        self.writer.add_scalar("FrontRES/CheckpointProbe/bias_abs", bias_abs, record["iteration"])

    best_score = getattr(self, "_frontres_best_probe_score", None)
    best_meta_path = os.path.join(self.log_dir, "frontres_best_probe.json")
    if best_score is None and os.path.exists(best_meta_path):
        try:
            with open(best_meta_path, "r", encoding="utf-8") as f:
                best_score = float(json.load(f).get("score"))
                self._frontres_best_probe_score = best_score
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            best_score = None
    if best_score is None or score > float(best_score):
        self._frontres_best_probe_score = score
        best_path = os.path.join(self.log_dir, "model_best_probe.pt")
        shutil.copyfile(checkpoint_path, best_path)
        with open(best_meta_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, sort_keys=True)
        print(
            "[Runner] New FrontRES probe best: "
            f"score={score:+.4f}, restore_rp={restore_ratio:+.3f}, "
            f"harm={harm_rate:.3f}, bias={bias_abs:.4f} -> {os.path.basename(best_path)}",
            flush=True,
        )

def save_runner(self, path: str, infos=None):
    """保存可恢复 Stage 2/3 语义的完整 runner checkpoint.

    函数名说明:
        `save_runner` 是 FrontRES persistence write owner, 汇总 policy, optimizer,
        normalizer, sampler 和 curriculum state; 它不是模型导出或 eval snapshot.

    主链路:
        上游: periodic/final checkpoint trigger 提供目标 path 和当前 runner state.
        下游: `torch.save` 写盘, `load_runner` 按相同 semantic keys 恢复.

    语义:
        Stage 2 -> Stage 3 必须保存同一个 full-6D actor 和 FrontRES prefix stats.
        Resume-only optimizer/sampler state 也必须与 iteration identity 同源.
    """
    # B1: 汇总 policy, optimizer, iteration 和 active Stage 3 owner state.
    # Check if using ResidualActorCritic (special handling)
    if isinstance(self.alg.policy, (ResidualActorCritic, FrontRESActorCritic)):
        # Save only residual network + critic (GMT is frozen, no need to save)
        model_state_dict = {
            'residual_actor': self.alg.policy.residual_actor.state_dict(),
            'critic': self.alg.policy.critic.state_dict(),}
        # Save noise std parameter
        if hasattr(self.alg.policy, 'std'):
            model_state_dict['std'] = self.alg.policy.std
        elif hasattr(self.alg.policy, 'log_std'):
            model_state_dict['log_std'] = self.alg.policy.log_std
    else:
        # Standard save: entire policy
        model_state_dict = self.alg.policy.state_dict()

    # -- Save model
    saved_dict = {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": self.alg.optimizer.state_dict(),
        "iter": self.current_learning_iteration,
        "infos": infos,}
    if getattr(self.alg, "frontres_training_objective", "") == "segment_replay_hrl":
        saved_dict["frontres_segment_warmup_config"] = {
            "critic_warmup_iterations": int(getattr(self.alg, "frontres_segment_critic_warmup_iterations", 0)),
            "actor_warmup_iterations": int(getattr(self.alg, "frontres_segment_actor_warmup_iterations", 0)),
        }

    # Persist adaptive DR state so resume picks up at the correct scale.
    if hasattr(self, '_dr_scale'):
        saved_dict["dr_scale"] = self._dr_scale
    if hasattr(self, '_dr_prev_error'):
        saved_dict["dr_prev_error"] = self._dr_prev_error
    if getattr(self, '_frontres_boundary_ema', None) is not None:
        saved_dict["frontres_boundary_ema"] = dict(self._frontres_boundary_ema)
    if getattr(self, '_last_frontres_boundary_stats', None) is not None:
        saved_dict["last_frontres_boundary_stats"] = dict(self._last_frontres_boundary_stats)
    if hasattr(self, "_frontres_gmt_frontier_safe_low"):
        saved_dict["frontres_gmt_frontier_safe_low"] = self._frontres_gmt_frontier_safe_low
    if hasattr(self, "_frontres_gmt_frontier_broken_high"):
        saved_dict["frontres_gmt_frontier_broken_high"] = self._frontres_gmt_frontier_broken_high
    if hasattr(self, "_frontres_gmt_frontier_probe_scale"):
        saved_dict["frontres_gmt_frontier_probe_scale"] = self._frontres_gmt_frontier_probe_scale
    if hasattr(self, "_frontres_gmt_frontier_probe_score"):
        saved_dict["frontres_gmt_frontier_probe_score"] = self._frontres_gmt_frontier_probe_score
    if hasattr(self, "_frontres_gmt_frontier_decision"):
        saved_dict["frontres_gmt_frontier_decision"] = self._frontres_gmt_frontier_decision
    if hasattr(self, "_frontres_gmt_frontier_confirmed"):
        saved_dict["frontres_gmt_frontier_confirmed"] = self._frontres_gmt_frontier_confirmed
    for _name in (
        "safe_score_ema",
        "broken_score_ema",
        "safe_count",
        "broken_count",
    ):
        _attr = f"_frontres_exec_floor_{_name}"
        if hasattr(self, _attr):
            saved_dict[f"frontres_exec_floor_{_name}"] = getattr(self, _attr)
    if hasattr(self, "_frontres_exec_floor_source_last"):
        saved_dict["frontres_exec_floor_source_last"] = self._frontres_exec_floor_source_last
    if hasattr(self, '_frontres_warmup_complete'):
        saved_dict["frontres_warmup_complete"] = bool(self._frontres_warmup_complete)
    if str(getattr(self, "training_type", "")) == "frontres":
        saved_dict["frontres_gain_config"] = _frontres_gain_config_payload(getattr(self, "cfg", None))
    segment_sampler = getattr(self, "_frontres_segment_sampler", None)
    if segment_sampler is not None and hasattr(segment_sampler, "state_dict"):
        saved_dict["frontres_segment_sampler_state_dict"] = segment_sampler.state_dict()
    
    # -- Save RND model if used
    if hasattr(self.alg, "rnd") and self.alg.rnd:
        saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
        saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
    
    # -- Save observation normalizer if used
    if self.empirical_normalization:
        extra_mean, extra_std = frontres_extra_norm_stats_for_save(
            getattr(self, "_frontres_extra_mean", None),
            getattr(self, "_frontres_extra_std", None),
            getattr(self, "_frontres_extra_normalizer", None),
        )
        obs_norm_state = self.obs_normalizer.state_dict()
        obs_norm_state = compose_frontres_obs_norm_state(
            obs_norm_state,
            extra_mean,
            extra_std,
        )
        saved_dict["obs_norm_state_dict"] = obs_norm_state
        saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()
        # Save teacher normalizer for MOSAIC
        if self.training_type == "mosaic" and hasattr(self, 'teacher_obs_normalizer'):
            if not isinstance(self.teacher_obs_normalizer, torch.nn.Identity):
                saved_dict["teacher_obs_norm_state_dict"] = self.teacher_obs_normalizer.state_dict()

    # Full-resume diagnostic probe; uncomment when checking checkpoint payloads.
    # print(
    #     "[FrontRES Checkpoint Save Probe] "
    #     f"path={path} iter={self.current_learning_iteration} "
    #     f"optimizer={_optimizer_state_debug(saved_dict.get('optimizer_state_dict'))} "
    #     f"sampler_state={'frontres_segment_sampler_state_dict' in saved_dict} "
    #     f"dr_scale={saved_dict.get('dr_scale', 'n/a')}",
    #     flush=True,
    # )

    # B2: Validate the complete payload after all semantic owners have contributed state.
    # B3: AUDIT-PERSIST-01 records the exact payload passed to torch.save.
    # Result: E69 LIVE PASS. model_221 保存 model/optimizer/normalizer/sampler/
    # Gain config/warmup payload, 与恢复后的 absolute iter 221 一致.
    print_checkpoint_payload_audit(self, path=path, payload=saved_dict)
    # save model
    torch.save(saved_dict, path)

    # upload model to external logging service
    logger_type = str(getattr(self, "logger_type", getattr(self, "cfg", {}).get("logger", "")) or "").lower()
    writer = getattr(self, "writer", None)
    if logger_type in ["neptune", "wandb"] and writer is not None and not bool(getattr(self, "disable_logs", False)):
        writer.save_model(path, self.current_learning_iteration)

def load_runner(self, path: str, load_optimizer: bool = True, load_critic: bool = True):
    """按 cold-start/resume 语义恢复 FrontRES runner checkpoint.

    函数名说明:
        `load_runner` 是 FrontRES persistence read owner, 区分 HSL 初始化和完整
        Stage 3 resume; 它不是宽松的 shape-compatible state loader.

    主链路:
        上游: train/eval entrypoint 提供 checkpoint path 和 load flags.
        下游: 恢复 full-6D actor, normalizer, optimizer, sampler, warmup 和 iteration
        state, 供 rollout/PPO 立即消费.

    语义:
        checkpoint identity 决定哪些状态允许恢复. Actor head 和 prefix stats
        不能漏载或错载, resume 状态也不能污染 Stage 2 -> Stage 3 cold start.
    """
    # B1: 映射 HSL actor/normalizer 前先读取 checkpoint identity.
    configure_formal_runtime_probe(
        bool(getattr(getattr(self, "alg", None), "frontres_formal_runtime_audit", False))
    )
    loaded_dict = torch.load(path, weights_only=False)
    # B2: 从同一 payload 恢复 sampler, actor, normalizer, optimizer, Gain 和 warmup identity.
    self._frontres_last_loaded_checkpoint_path = os.path.abspath(path)
    segment_sampler = getattr(self, "_frontres_segment_sampler", None)
    if (
        segment_sampler is not None
        and "frontres_segment_sampler_state_dict" in loaded_dict
        and hasattr(segment_sampler, "load_state_dict")
    ):
        segment_sampler.load_state_dict(loaded_dict["frontres_segment_sampler_state_dict"])
        print("[Runner] Loaded FrontRES Segment sampler state from checkpoint.")
    self._frontres_warmup_complete = bool(loaded_dict.get("frontres_warmup_complete", False))
    if self._frontres_warmup_complete:
        print("[Runner] Checkpoint marks FrontRES supervised warmup as complete.")

    # ── 断点续训模式控制 ────────────────────────────────────────────────────────
    # is_full_resume=True  (Stage2→Stage2 断点续训): 恢复优化器矩估计+学习率, 保留 std
    # is_full_resume=False (Stage1→Stage2 权重迁移): 仅权重, 重置优化器和 std.
    # Joint-warmup checkpoints are a special case: their critic has already
    # learned E(s)=R_feasible_oracle-R_noisy and should be transferred into RL.
    # load_optimizer 参数仍可从外部显式覆盖（例如强制跳过优化器加载）。
    is_full_resume: bool = self.cfg.get('is_full_resume', True)
    eval_only = bool(
        getattr(self.alg, "frontres_segment_offline_eval_only", False)
        or getattr(self.alg, "frontres_segment_sequence_offline_eval_only", False)
        or self.cfg.get("frontres_segment_offline_eval_only", False)
        or self.cfg.get("frontres_segment_sequence_offline_eval_only", False)
    )
    if not is_full_resume:
        load_optimizer = False   # 权重迁移模式：强制跳过优化器，从零初始化 Adam
        load_critic = self._frontres_warmup_complete
    _validate_frontres_gain_config_resume(self, loaded_dict, is_full_resume=is_full_resume)
    # Full-resume diagnostic probe; uncomment when checking checkpoint reloads.
    # print(
    #     "[FrontRES Resume Probe] "
    #     f"path={os.path.abspath(path)} checkpoint_iter={loaded_dict.get('iter', 'n/a')} "
    #     f"is_full_resume={is_full_resume} "
    #     f"checkpoint_optimizer={_optimizer_state_debug(loaded_dict.get('optimizer_state_dict'))} "
    #     f"sampler_state={'frontres_segment_sampler_state_dict' in loaded_dict} "
    #     f"frontres_warmup_complete={self._frontres_warmup_complete}",
    #     flush=True,
    # )
    print(f"[Runner] is_full_resume={is_full_resume} → "
          f"load_optimizer={load_optimizer}, load_critic={load_critic}, "
          f"reset_noise_std={not is_full_resume}, eval_only={eval_only}")

    # Check if using ResidualActorCritic (special handling)
    if isinstance(self.alg.policy, (ResidualActorCritic, FrontRESActorCritic)):
        # Stage 2 -> Stage 3 uses the same full-6D residual actor contract.
        if isinstance(self.alg.policy, FrontRESActorCritic) and "student.0.weight" in loaded_dict["model_state_dict"]:
            mapped_dict = {k.replace("student.", ""): v for k, v in loaded_dict["model_state_dict"].items() if k.startswith("student.")}
            self.alg.policy.residual_actor.load_state_dict(mapped_dict, strict=True)
            print("[Runner] Loaded Stage 2 student weights into the full-6D residual actor.")
        else:
            residual_state = loaded_dict["model_state_dict"]["residual_actor"]
            self.alg.policy.residual_actor.load_state_dict(residual_state, strict=True)
        if load_critic:
            if "critic" in loaded_dict["model_state_dict"]:
                self.alg.policy.critic.load_state_dict(loaded_dict["model_state_dict"]["critic"])
            else:
                print("[Runner] No critic weights found. Critic will be initialized from scratch.")
        # Load noise std parameter only when checkpoint and runtime action dims match.
        _copy_policy_noise_state(self.alg.policy, loaded_dict["model_state_dict"])
        if load_critic:
            print("[Runner] Loaded residual network + critic from checkpoint (GMT remains frozen)")
        else:
            print("[Runner] Loaded residual network only (skipping critic from checkpoint)")
        resumed_training = True
    else:
        if load_critic:
            # Standard load: entire policy
            resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        else:
            actor_only_state_dict = {
                key: value
                for key, value in loaded_dict["model_state_dict"].items()
                if not key.startswith("critic.")}
            
            resumed_training = self.alg.policy.load_state_dict(actor_only_state_dict, strict=False)

    # Load RND model if used
    if hasattr(self.alg, "rnd") and self.alg.rnd:
        self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])

    # Load observation normalizers if used
    if self.empirical_normalization:
        if resumed_training:
            # Resuming training: load student obs normalizer
            # For ResidualActorCritic / FrontRESActorCritic, obs_normalizer IS GMT's frozen
            # normalizer — never overwrite it with a checkpoint's normalizer statistics.
            if not isinstance(self.alg.policy, (ResidualActorCritic, FrontRESActorCritic)):
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            elif (isinstance(self.alg.policy, FrontRESActorCritic)
                    and self._frontres_gmt_obs_dim is not None
                    and "obs_norm_state_dict" in loaded_dict):
                # Task-space FrontRES: prefix dims [:num_extra] are not covered by
                # the GMT normalizer.  Restore checkpoint stats for the available
                # prefix dims; newly added prefix dims use identity normalization.
                _s1_sd = loaded_dict["obs_norm_state_dict"]
                gmt_dim = self._frontres_gmt_obs_dim
                obs_dim = int(getattr(self.alg.policy, "num_actor_obs", gmt_dim))
                extra_stats = extract_frontres_extra_norm_stats(_s1_sd, obs_dim, gmt_dim, self.device)
                if extra_stats is not None:
                    self._frontres_extra_mean, self._frontres_extra_std = extra_stats
                    print(f"[Runner] Loaded FrontRES prefix normalizer stats "
                          f"(dims 0–{self._frontres_extra_mean.shape[-1]}) for FrontRES task-space.")
                else:
                    self._frontres_extra_mean = None
                    self._frontres_extra_std = None
                    print("[Runner] Checkpoint has no compatible FrontRES prefix "
                          "normalizer stats; FrontRES prefix dims pass through unnormalized.")

            if self.training_type == "mosaic":
                # For MOSAIC: determine whether to load privileged_obs_normalizer from checkpoint
                # Only skip loading if teacher_critic was loaded from a separate checkpoint AND is frozen
                load_privileged_normalizer = load_critic
                if hasattr(self.alg, 'teacher_critic_checkpoint_path') and self.alg.teacher_critic_checkpoint_path is not None:
                    if hasattr(self.alg, 'teacher_critic_frozen') and self.alg.teacher_critic_frozen:
                        load_privileged_normalizer = False
                        print("[Runner] Keeping privileged_obs_normalizer from teacher_critic_checkpoint (frozen).")

                if load_privileged_normalizer:
                    # Load critic normalizer from student checkpoint
                    if "privileged_obs_norm_state_dict" in loaded_dict:
                        self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
                        print("[Runner] Loaded privileged_obs_normalizer from checkpoint.")
                    else:
                        print("[Runner] WARNING: No privileged_obs_norm_state_dict in checkpoint!")

                # Load teacher obs normalizer if available (for teacher BC)
                if "teacher_obs_norm_state_dict" in loaded_dict:
                    self.teacher_obs_normalizer.load_state_dict(loaded_dict["teacher_obs_norm_state_dict"])
                    print("[Runner] Loaded teacher_obs_normalizer from checkpoint.")
            else:
                # For PPO and Distillation: load both normalizers
                if load_critic:
                    priv_sd = loaded_dict.get("privileged_obs_norm_state_dict", {})
                    if priv_sd and "_mean" in priv_sd:
                        self.privileged_obs_normalizer.load_state_dict(priv_sd)
                    else:
                        # Stage 1 (SuperviseLearning) checkpoint has no valid
                        # privileged_obs_norm_state_dict — critic normalizer starts fresh.
                        print("[Runner] WARNING: privileged_obs_norm_state_dict missing or invalid — "
                              "privileged_obs_normalizer starts fresh (expected for Stage 1 → Stage 2 transfer).")
        else:
            # Not resuming (e.g., Distillation after RL): load teacher normalizer
            # For Distillation: the checkpoint's obs_norm is the teacher's normalizer
            if load_critic:
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
    # -- load optimizer if used
    if load_optimizer and resumed_training:
        if not load_critic:
            print("[Runner] Skipping optimizer load because load_critic=False.")
        else:
            try:
                # -- algorithm optimizer
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
                print("[Runner] Loaded optimizer state from checkpoint.")
                # Full-resume diagnostic probe; uncomment when checking optimizer state.
                # print(
                #     "[FrontRES Resume Probe] "
                #     f"optimizer_loaded=True runtime_optimizer={_optimizer_state_debug(self.alg.optimizer.state_dict())}",
                #     flush=True,
                # )
                # ── 学习率同步 ─────────────────────────────────────────────────────
                # PPO.update() 每次 epoch 都用 self.alg.learning_rate 覆盖
                # optimizer.param_groups["lr"]。load_state_dict 已将 param_groups["lr"]
                # 恢复为 checkpoint 时的值，但 self.alg.learning_rate 仍是配置初始值。
                # 此处同步，避免第一次 update() 将已恢复的学习率覆盖为初始值。
                if is_full_resume and hasattr(self.alg, 'learning_rate'):
                    restored_lr = self.alg.optimizer.param_groups[0]['lr']
                    reset_lr = bool(self.cfg.get('reset_lr_on_resume', False))
                    if reset_lr:
                        # lr 被 adaptive schedule 压至下限时（如因 desired_kl 配置错误），
                        # 直接重置为算法配置的初始学习率，避免续训起点过低。
                        config_lr = float(self.alg_cfg.get('learning_rate', 5e-4))
                        self.alg.learning_rate = config_lr
                        for pg in self.alg.optimizer.param_groups:
                            pg['lr'] = config_lr
                        print(f"[Runner] Reset learning_rate → {config_lr:.2e} "
                              f"(reset_lr_on_resume=True; checkpoint had {restored_lr:.2e})")
                    else:
                        self.alg.learning_rate = restored_lr
                        print(f"[Runner] Synced learning_rate = {restored_lr:.2e} (from optimizer checkpoint)")
            except (ValueError, KeyError) as e:
                # Optimizer state mismatch (e.g., different parameter groups between stages)
                # This can happen when:
                # - Stage 1 had frozen critic (optimizer only has actor params)
                # - Stage 2 unfreezes critic (optimizer has actor + critic params)
                print(f"[Runner] WARNING: Could not load optimizer state: {e}")
                print("[Runner] Optimizer will be initialized from scratch (learning rate, momentum, etc. reset)")
                print("[Runner] This is expected when transitioning between training stages with different frozen parameters.")
                # Full-resume diagnostic probe; uncomment when checking optimizer state.
                # print(
                #     "[FrontRES Resume Probe] "
                #     f"optimizer_loaded=False runtime_optimizer={_optimizer_state_debug(self.alg.optimizer.state_dict())}",
                #     flush=True,
                # )

            # -- RND optimizer if used
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
    # -- load current learning iteration
    if resumed_training and is_full_resume and "frontres_segment_warmup_config" in loaded_dict:
        saved_warmup = loaded_dict["frontres_segment_warmup_config"]
        runtime_warmup = {
            "critic_warmup_iterations": int(getattr(self.alg, "frontres_segment_critic_warmup_iterations", 0)),
            "actor_warmup_iterations": int(getattr(self.alg, "frontres_segment_actor_warmup_iterations", 0)),
        }
        if saved_warmup != runtime_warmup:
            if eval_only:
                print(
                    "[Runner] Eval-only checkpoint load: warmup config guard skipped; "
                    f"checkpoint={saved_warmup}, runtime={runtime_warmup}.",
                    flush=True,
                )
            else:
                raise ValueError(
                    "Stage 3 warmup config changed across full resume: "
                    f"checkpoint={saved_warmup}, runtime={runtime_warmup}."
                )
    if resumed_training:
        if is_full_resume:
            self.current_learning_iteration = loaded_dict["iter"]
        else:
            self.current_learning_iteration = 0
            print("[Runner] Stage1→Stage2 cold-start: current_learning_iteration reset to 0.")
        # Full-resume diagnostic probe; uncomment when checking resume iteration.
        # print(
        #     "[FrontRES Resume Probe] "
        #     f"iteration_after_load={self.current_learning_iteration} checkpoint_iter={loaded_dict.get('iter', 'n/a')} "
        #     f"is_full_resume={is_full_resume}",
        #     flush=True,
        # )

    # ── 噪声 std 控制 ──────────────────────────────────────────────────────────
    # is_full_resume=True:  保留 checkpoint 中已自然适应的 std（断点续训）
    # is_full_resume=False: 重置为 init_noise_std（Stage1→Stage2 冷启动）
    # 向后兼容：若 cfg 中显式设置了 reset_noise_std_on_resume，以其为准。
    reset_noise: bool
    if 'reset_noise_std_on_resume' in self.cfg:
        reset_noise = bool(self.cfg.get('reset_noise_std_on_resume'))
        print(f"[Runner] reset_noise_std_on_resume = {reset_noise} (explicit config override)")
    else:
        reset_noise = not is_full_resume   # is_full_resume=True → 不重置; False → 重置
        print(f"[Runner] reset_noise_std = {reset_noise} (derived from is_full_resume={is_full_resume})")

    if reset_noise and (hasattr(self.alg.policy, 'std') or hasattr(self.alg.policy, 'log_std')):
        init_noise_std = self.policy_cfg.get("init_noise_std", 1.0)
        noise_std_type = self.policy_cfg.get("noise_std_type", "scalar")
        _reset_policy_noise_state(
            self.alg.policy,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            device=self.device,
        )
    else:
        if hasattr(self.alg.policy, 'std'):
            print(f"[Runner] Kept noise std from checkpoint = {self.alg.policy.std.mean().item():.4f}")

    # -- Freeze normalizer if specified in config (for stage transitions)
    # This prevents normalizer statistics from drifting when resuming from distillation
    freeze_normalizer = self.cfg.get("freeze_normalizer_on_resume", False)
    print(f"[Runner] freeze_normalizer_on_resume = {freeze_normalizer}")
    if freeze_normalizer and self.empirical_normalization:
        # Freeze obs normalizer
        self.obs_normalizer.eval()
        if hasattr(self.obs_normalizer, 'until'):
            self.obs_normalizer.until = self.obs_normalizer.count  # Stop updating
        print(f"[Runner] Froze obs_normalizer (count={self.obs_normalizer.count})")

        # Freeze privileged obs normalizer
        self.privileged_obs_normalizer.eval()
        if hasattr(self.privileged_obs_normalizer, 'until'):
            self.privileged_obs_normalizer.until = self.privileged_obs_normalizer.count
        print(f"[Runner] Froze privileged_obs_normalizer (count={self.privileged_obs_normalizer.count})")

    # Restore adaptive DR scale so resume continues from the correct DR level.
    # is_full_resume=True  (Stage2断点续训): 恢复 checkpoint 中的 dr_scale
    # is_full_resume=False (Stage1→Stage2冷启动): 忽略 checkpoint dr_scale，
    #   改用 cfg 中的 dr_scale_init（默认 1.0），确保 Stage2 从 Stage1 训练强度出发，
    #   避免 dr_scale=0 时 Stage1 修正策略作用于干净参考导致的即时崩溃。
    if is_full_resume:
        self._dr_scale      = loaded_dict.get("dr_scale",      0.0)
        self._dr_prev_error = loaded_dict.get("dr_prev_error", 0.0)
        if "frontres_boundary_ema" in loaded_dict:
            self._frontres_boundary_ema = dict(loaded_dict["frontres_boundary_ema"])
        if "last_frontres_boundary_stats" in loaded_dict:
            self._last_frontres_boundary_stats = dict(loaded_dict["last_frontres_boundary_stats"])
        self._frontres_gmt_frontier_safe_low = float(
            loaded_dict.get("frontres_gmt_frontier_safe_low", self._dr_scale)
        )
        self._frontres_gmt_frontier_broken_high = loaded_dict.get(
            "frontres_gmt_frontier_broken_high", None
        )
        if self._frontres_gmt_frontier_broken_high is not None:
            self._frontres_gmt_frontier_broken_high = float(self._frontres_gmt_frontier_broken_high)
        self._frontres_gmt_frontier_probe_scale = float(
            loaded_dict.get("frontres_gmt_frontier_probe_scale", self._dr_scale)
        )
        self._frontres_gmt_frontier_probe_score = loaded_dict.get(
            "frontres_gmt_frontier_probe_score", None
        )
        if self._frontres_gmt_frontier_probe_score is not None:
            self._frontres_gmt_frontier_probe_score = float(self._frontres_gmt_frontier_probe_score)
        self._frontres_gmt_frontier_decision = str(
            loaded_dict.get("frontres_gmt_frontier_decision", "resume")
        )
        self._frontres_gmt_frontier_confirmed = float(
            loaded_dict.get("frontres_gmt_frontier_confirmed", self._frontres_gmt_frontier_safe_low)
        )
        for _name in (
            "safe_score_ema",
            "broken_score_ema",
            "safe_count",
            "broken_count",
        ):
            _key = f"frontres_exec_floor_{_name}"
            _attr = f"_frontres_exec_floor_{_name}"
            if _key in loaded_dict and loaded_dict[_key] is not None:
                setattr(self, _attr, float(loaded_dict[_key]))
            elif hasattr(self, _attr):
                delattr(self, _attr)
        self._frontres_exec_floor_source_last = str(
            loaded_dict.get("frontres_exec_floor_source_last", "resume")
        )
        print(f"[Runner] Adaptive DR scale restored from checkpoint: {self._dr_scale:.4f}")
    else:
        _dr_init = float(self.cfg.get("dr_scale_init", 1.0))
        self._dr_scale = _dr_init
        self._frontres_boundary_ema = None
        self._last_frontres_boundary_stats = None
        self._frontres_gmt_frontier_safe_low = _dr_init
        self._frontres_gmt_frontier_broken_high = None
        self._frontres_gmt_frontier_probe_scale = _dr_init
        self._frontres_gmt_frontier_probe_score = None
        self._frontres_gmt_frontier_decision = "cold_start"
        self._frontres_gmt_frontier_confirmed = _dr_init
        for _name in (
            "safe_score_ema",
            "broken_score_ema",
            "safe_count",
            "broken_count",
        ):
            _attr = f"_frontres_exec_floor_{_name}"
            if hasattr(self, _attr):
                delattr(self, _attr)
        self._frontres_exec_floor_source_last = "cold_start"
        print(f"[Runner] Stage1→Stage2 cold-start: dr_scale initialised to "
              f"dr_scale_init={_dr_init:.4f} (ignoring checkpoint value "
              f"{loaded_dict.get('dr_scale', 0.0):.4f})")

    # B3: AUDIT-HSL-LOAD-01 records the actual loaded actor/normalizer boundary.
    # Result: E69 LIVE PASS. model_220 full-resume 恢复 actor/critic/optimizer,
    # prefix normalizer, sampler, Gain config, warmup phase, std 和 DR scale.
    emit_formal_runtime_probe(
        "AUDIT-HSL-LOAD-01",
        checkpoint_path=self._frontres_last_loaded_checkpoint_path,
        checkpoint_iter=loaded_dict.get("iter", "missing"),
        residual_actor=type(getattr(getattr(self.alg, "policy", None), "residual_actor", None)).__name__,
        obs_normalizer=type(getattr(self, "obs_normalizer", None)).__name__,
        full_resume=bool(is_full_resume),
    )
    return loaded_dict["infos"]
