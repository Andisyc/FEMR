"""Checkpoint, save, and resume helpers for OnPolicyRunner.

This module owns persistence mechanics. The runner keeps its public methods as
thin wrappers so training loops and external scripts keep the same API.
"""

from __future__ import annotations

import json
import os
import shutil

import torch

from rsl_rl.modules import FrontRESActorCritic, ResidualActorCritic


def _numeric_module_keys(state_dict: dict, suffix: str) -> list[str]:
    keys = [key for key in state_dict if key.endswith(suffix) and key.split(".", 1)[0].isdigit()]
    return sorted(keys, key=lambda key: int(key.split(".", 1)[0]))


def _load_split_proposal_from_two_head_residual(proposal_actor, residual_state: dict) -> bool:
    """Load Stage-1 two-head residual weights into a Stage-2 proposal-only actor."""

    if not residual_state:
        return False
    if not any(key.startswith("proposal_head.") for key in residual_state):
        return False

    target_state = proposal_actor.state_dict()
    target_weight_keys = _numeric_module_keys(target_state, ".weight")
    if not target_weight_keys:
        return False

    last_weight = target_weight_keys[-1]
    last_bias = last_weight.replace(".weight", ".bias")
    new_state = dict(target_state)
    copied_any = False

    for key, value in target_state.items():
        if key == last_weight:
            src = residual_state.get("proposal_head.weight")
        elif key == last_bias:
            src = residual_state.get("proposal_head.bias")
        else:
            src = residual_state.get(f"trunk.{key}")
        if src is None:
            continue
        if src.shape != value.shape:
            return False
        new_state[key] = src.clone()
        copied_any = True

    if not copied_any:
        return False
    proposal_actor.load_state_dict(new_state, strict=True)
    return True


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
    # Check if using ResidualActorCritic (special handling)
    if isinstance(self.alg.policy, (ResidualActorCritic, FrontRESActorCritic)):
        # Save only residual network + critic (GMT is frozen, no need to save)
        model_state_dict = {
            'residual_actor': self.alg.policy.residual_actor.state_dict(),
            'critic': self.alg.policy.critic.state_dict(),}
        if getattr(self.alg.policy, "acceptance_actor", None) is not None:
            model_state_dict['acceptance_actor'] = self.alg.policy.acceptance_actor.state_dict()
        if getattr(self.alg.policy, "state_router_head", None) is not None:
            model_state_dict['state_router_head'] = self.alg.policy.state_router_head.state_dict()
        if getattr(self.alg.policy, "authority_actor", None) is not None:
            model_state_dict['authority_actor'] = self.alg.policy.authority_actor.state_dict()
        if getattr(self.alg.policy, "authority_critic", None) is not None:
            model_state_dict['authority_critic'] = self.alg.policy.authority_critic.state_dict()
        
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
    segment_sampler = getattr(self, "_frontres_segment_sampler", None)
    if segment_sampler is not None and hasattr(segment_sampler, "state_dict"):
        saved_dict["frontres_segment_sampler_state_dict"] = segment_sampler.state_dict()
    
    # -- Save RND model if used
    if hasattr(self.alg, "rnd") and self.alg.rnd:
        saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
        saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
    
    # -- Save observation normalizer if used
    if self.empirical_normalization:
        saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
        saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()
        # Save teacher normalizer for MOSAIC
        if self.training_type == "mosaic" and hasattr(self, 'teacher_obs_normalizer'):
            if not isinstance(self.teacher_obs_normalizer, torch.nn.Identity):
                saved_dict["teacher_obs_norm_state_dict"] = self.teacher_obs_normalizer.state_dict()

    # save model
    torch.save(saved_dict, path)

    # upload model to external logging service
    logger_type = str(getattr(self, "logger_type", getattr(self, "cfg", {}).get("logger", "")) or "").lower()
    writer = getattr(self, "writer", None)
    if logger_type in ["neptune", "wandb"] and writer is not None and not bool(getattr(self, "disable_logs", False)):
        writer.save_model(path, self.current_learning_iteration)

def load_runner(self, path: str, load_optimizer: bool = True, load_critic: bool = True):
    loaded_dict = torch.load(path, weights_only=False)
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
    if not is_full_resume:
        load_optimizer = False   # 权重迁移模式：强制跳过优化器，从零初始化 Adam
        load_critic = self._frontres_warmup_complete
    print(f"[Runner] is_full_resume={is_full_resume} → "
          f"load_optimizer={load_optimizer}, load_critic={load_critic}, "
          f"reset_noise_std={not is_full_resume}")

    # Check if using ResidualActorCritic (special handling)
    if isinstance(self.alg.policy, (ResidualActorCritic, FrontRESActorCritic)):
        # 智能映射：尝试从阶段一 (SuperviseLearning) 提取 student 权重
        if isinstance(self.alg.policy, FrontRESActorCritic) and "student.0.weight" in loaded_dict["model_state_dict"]:
            mapped_dict = {k.replace("student.", ""): v for k, v in loaded_dict["model_state_dict"].items() if k.startswith("student.")}
            try:
                self.alg.policy.residual_actor.load_state_dict(mapped_dict, strict=True)
                print("[Runner] Success: Auto-mapped Stage 1 'student' weights to Stage 2 'residual_actor'!")
            except RuntimeError:
                migrated = False
                if hasattr(self.alg.policy, "initialize_two_head_from_legacy_state"):
                    migrated = self.alg.policy.initialize_two_head_from_legacy_state(mapped_dict)
                if migrated:
                    print("[Runner] Migrated Stage 1 'student' weights into FrontRES two-head actor.")
                else:
                    raise
        else:
            residual_state = loaded_dict["model_state_dict"]["residual_actor"]
            try:
                self.alg.policy.residual_actor.load_state_dict(residual_state, strict=True)
            except RuntimeError as exc:
                checkpoint_is_two_head = any(
                    key.startswith(("trunk.", "proposal_head.", "acceptance_head."))
                    for key in residual_state
                )
                if checkpoint_is_two_head:
                    migrated = _load_split_proposal_from_two_head_residual(
                        self.alg.policy.residual_actor,
                        residual_state,
                    )
                    if migrated:
                        print("[Runner] Migrated Stage 1 two-head residual_actor into split Stage 2 proposal actor.")
                    else:
                        raise RuntimeError(
                            "Checkpoint residual_actor is a FrontRESTwoHeadActor, but it cannot be "
                            "mapped into the active Stage 2 proposal actor. Check "
                            "frontres_split_acceptance_head, hidden dims, and task correction dims."
                        ) from exc
                else:
                    migrated = False
                    if hasattr(self.alg.policy, "initialize_two_head_from_legacy_state"):
                        migrated = self.alg.policy.initialize_two_head_from_legacy_state(residual_state)
                    if migrated:
                        print("[Runner] Migrated legacy residual_actor weights into FrontRES two-head actor.")
                    else:
                        raise
        if getattr(self.alg.policy, "acceptance_actor", None) is not None:
            if "acceptance_actor" in loaded_dict["model_state_dict"]:
                self.alg.policy.acceptance_actor.load_state_dict(loaded_dict["model_state_dict"]["acceptance_actor"])
                print("[Runner] Loaded split FrontRES acceptance_actor from checkpoint.")
            else:
                migrated = False
                if hasattr(self.alg.policy, "initialize_acceptance_from_residual_state"):
                    migrated = self.alg.policy.initialize_acceptance_from_residual_state(
                        loaded_dict["model_state_dict"].get("residual_actor", {})
                    )
                if migrated:
                    print("[Runner] Initialized split acceptance_actor from legacy residual_actor rho rows.")
                else:
                    print("[Runner] No split acceptance_actor weights found; initialized acceptance head from scratch.")
        elif "acceptance_actor" in loaded_dict["model_state_dict"]:
            print("[Runner] Ignoring split acceptance_actor weights because the active config uses the single two-head FrontRES actor.")
        if getattr(self.alg.policy, "state_router_head", None) is not None:
            if "state_router_head" in loaded_dict["model_state_dict"]:
                self.alg.policy.state_router_head.load_state_dict(
                    loaded_dict["model_state_dict"]["state_router_head"],
                    strict=True,
                )
                print("[Runner] Loaded FrontRES state_router_head from checkpoint.")
            else:
                print("[Runner] No state_router_head weights found; initialized alpha head from scratch.")
        if getattr(self.alg.policy, "authority_actor", None) is not None:
            if "authority_actor" in loaded_dict["model_state_dict"]:
                self.alg.policy.authority_actor.load_state_dict(
                    loaded_dict["model_state_dict"]["authority_actor"],
                    strict=True,
                )
                print("[Runner] Loaded FrontRES authority_actor from checkpoint.")
            else:
                print("[Runner] No authority_actor weights found; initialized authority actor from scratch.")
        elif "authority_actor" in loaded_dict["model_state_dict"]:
            print("[Runner] Ignoring authority_actor weights because authority actor-critic is disabled.")
        if getattr(self.alg.policy, "authority_critic", None) is not None:
            if "authority_critic" in loaded_dict["model_state_dict"]:
                self.alg.policy.authority_critic.load_state_dict(
                    loaded_dict["model_state_dict"]["authority_critic"],
                    strict=True,
                )
                print("[Runner] Loaded FrontRES authority_critic from checkpoint.")
            else:
                print("[Runner] No authority_critic weights found; initialized authority critic from scratch.")
        elif "authority_critic" in loaded_dict["model_state_dict"]:
            print("[Runner] Ignoring authority_critic weights because authority actor-critic is disabled.")

        if load_critic:
            if "critic" in loaded_dict["model_state_dict"]:
                self.alg.policy.critic.load_state_dict(loaded_dict["model_state_dict"]["critic"])
            else:
                print("[Runner] No critic weights found. Critic will be initialized from scratch.")
        # Load noise std parameter
        if "std" in loaded_dict["model_state_dict"]:
            self.alg.policy.std.data = loaded_dict["model_state_dict"]["std"].data
        elif "log_std" in loaded_dict["model_state_dict"]:
            self.alg.policy.log_std.data = loaded_dict["model_state_dict"]["log_std"].data
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
                # Task-space FrontRES: anchor-error dims [:num_extra] are not
                # covered by the GMT normalizer.  Restore Stage-1 empirical stats
                # for those dims when the checkpoint actually contains them.
                # so Stage 2 sees the same normalized scale that Stage 1 trained on.
                _s1_sd   = loaded_dict["obs_norm_state_dict"]
                _s1_mean = _s1_sd.get("_mean", None)  # shape (1, 800)
                _s1_std  = _s1_sd.get("_std",  None)  # shape (1, 800)
                if _s1_mean is not None and _s1_std is not None:
                    gmt_dim = self._frontres_gmt_obs_dim
                    obs_dim = int(getattr(self.alg.policy, "num_actor_obs", gmt_dim))
                    num_extra = max(0, obs_dim - gmt_dim)
                    if num_extra > 0 and _s1_mean.shape[-1] >= obs_dim and _s1_std.shape[-1] >= obs_dim:
                        self._frontres_extra_mean = _s1_mean[:, :num_extra].to(self.device)
                        self._frontres_extra_std  = _s1_std[:,  :num_extra].to(self.device)
                        print(f"[Runner] Loaded Stage-1 anchor-error normalizer stats "
                              f"(dims 0–{num_extra}) for FrontRES task-space.")
                    else:
                        self._frontres_extra_mean = None
                        self._frontres_extra_std = None
                        print("[Runner] Stage-1 checkpoint has no compatible anchor-error "
                              "normalizer stats; FrontRES extra dims pass through unnormalized.")

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

            # -- RND optimizer if used
            if hasattr(self.alg, "rnd") and self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
    # -- load current learning iteration
    if resumed_training:
        if is_full_resume:
            self.current_learning_iteration = loaded_dict["iter"]
        else:
            self.current_learning_iteration = 0
            print("[Runner] Stage1→Stage2 cold-start: current_learning_iteration reset to 0.")

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
        num_actions = (self.alg.policy.std.shape[0] if hasattr(self.alg.policy, 'std')
                       else self.alg.policy.log_std.shape[0])
        if noise_std_type == "scalar":
            self.alg.policy.std.data = torch.ones(num_actions, device=self.device) * init_noise_std
            print(f"[Runner] Reset noise std → {init_noise_std}")
        elif noise_std_type == "log":
            self.alg.policy.log_std.data = torch.log(
                torch.ones(num_actions, device=self.device) * init_noise_std)
            print(f"[Runner] Reset log_std → log({init_noise_std})")
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

    return loaded_dict["infos"]
