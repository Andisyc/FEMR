"""Runner logging for MOSAIC/FrontRES training.

This module owns TensorBoard scalar emission and console string assembly.  The
runner keeps only the public log() entrypoint so existing call sites stay stable.
"""

from __future__ import annotations

import statistics
import time

import torch

from rsl_rl.modules import FrontRESActorCritic
from rsl_rl.frontres.frontres_diagnostics import (
    format_frontres_floor_alpha_diagnostics,
    format_frontres_optimization_diagnostics,
    format_frontres_preference_diagnostics,
    format_frontres_route_rho_diagnostics,
)


def log_runner(self, locs: dict, width: int = 80, pad: int = 35):
    # Compute the collection size
    collection_size = self.num_steps_per_env * self.env.num_envs * self.gpu_world_size

    # Update total time-steps and time
    self.tot_timesteps += collection_size
    self.tot_time += locs["collection_time"] + locs["learn_time"]
    iteration_time = locs["collection_time"] + locs["learn_time"]

    # -- Episode info
    ep_string = ""
    if locs["ep_infos"]:
        for key in locs["ep_infos"][0]:
            infotensor = torch.tensor([], device=self.device)
            for ep_info in locs["ep_infos"]:
                # handle scalar and zero dimensional tensor infos
                if key not in ep_info:
                    continue
                if not isinstance(ep_info[key], torch.Tensor):
                    ep_info[key] = torch.Tensor([ep_info[key]])
                if len(ep_info[key].shape) == 0:
                    ep_info[key] = ep_info[key].unsqueeze(0)
                infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
            value = torch.mean(infotensor)

            if self.training_type == "supervise":
                # Stage 1: only log termination-related keys under GMT/ prefix.
                # Reward and other RL metrics are meaningless here.
                key_lower = key.lower().replace("/", "_")
                if any(r in key_lower for r in ["rew", "reward"]):
                    continue  # skip reward metrics entirely
                # Everything else (e.g. termination reasons) → GMT/ namespace
                log_key = key if "/" in key else f"GMT/{key}"
                self.writer.add_scalar(log_key, value, locs["it"])
                ep_string += f"""{f'GMT {key}:':>{pad}} {value:.4f}\n"""
            else:
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

    # -- Stage 1: GMT episode-length as action-completion proxy
    if self.training_type == "supervise" and len(locs["lenbuffer"]) > 0:
        gmt_ep_len = statistics.mean(locs["lenbuffer"])
        self.writer.add_scalar("GMT/mean_episode_length", gmt_ep_len, locs["it"])

    mean_std = self.alg.policy.action_std.mean()
    fps = int(collection_size / (locs["collection_time"] + locs["learn_time"]))

    # -- Losses
    # Keys suppressed when their controlling lambda is 0 (avoids clutter for unused modes).
    _suppress_if_zero = {"bc_off_policy", "bc_teacher", "lambda_off_policy", "lambda_teacher"}
    # Keys reclassified out of Loss/ for clarity.
    _to_frontres  = {
        "supervised_cos_sim",
        "supervised_mae",
        "supervised_rmse",
        "supervised_rpy_mae",
        "supervised_rpy_rmse",
        "supervised_restore_ratio",
        "supervised_valid_frac",
        "supervised_l_pos",
        "supervised_l_rot",
        "supervised_l_mag",
        "supervised_l_over",
        "supervised_l_smooth",
        "supervised_l_sparse",
        "supervised_l_miss",
        "supervised_l_coeff_smooth",
        "supervised_l_harm",
        "frontres_alpha_mean",
        "frontres_alpha_active_frac",
        "frontres_tau_mean",
        "frontres_tau_active_frac",
        "frontres_rho_pos_mean",
        "frontres_rho_pos_active_frac",
        "frontres_accept_pos_mean",
        "frontres_accept_rpy_mean",
        "frontres_accept_active_frac",
        "frontres_write_ratio",
        "frontres_proposal_ratio",
        "frontres_axis_leakage",
        "state_alpha_loss",
        "lambda_state_alpha",
        "state_alpha_mask_frac",
        "state_alpha_target_mean",
        "state_alpha_pred_mean",
        "state_alpha_abs_err",
        "state_alpha_acc",
        "structured_joint_rl_loss",
        "lambda_structured_joint_rl",
        "structured_joint_rl_enabled",
        "structured_joint_rl_adv_mean",
        "structured_joint_rl_adv_abs_mean",
        "structured_joint_rl_adv_used_mean",
        "structured_joint_rl_weight_mean",
        "structured_joint_rl_weight_all_mean",
        "structured_joint_rl_rho_adv_mean",
        "structured_joint_rl_rho_adv_abs_mean",
        "structured_joint_rl_rho_weight_mean",
        "structured_joint_rl_rho_weight_all_mean",
        "structured_joint_rl_rho_ratio_mean",
        "structured_joint_rl_rho_loss",
        "structured_joint_rl_prior_loss",
        "structured_joint_rl_prior_authority_mean",
        "structured_joint_rl_prior_target_mean",
        "structured_joint_rl_prior_rho_mean",
        "structured_joint_rl_rho_mean",
        "structured_joint_rl_rho_abs_from_half",
        "structured_joint_rl_rho_near_half_frac",
        "structured_joint_rl_adv_pos_frac",
        "structured_joint_rl_adv_neg_frac",
        "structured_joint_rl_adv_near_zero_frac",
        "lambda_structured_joint_prior",
        "structured_joint_rl_ratio_mean",
    }      # diagnostics, not losses
    _to_curriculum = {"lambda_supervised", "ppo_actor_weight"}  # scheduler state, not a loss
    _frontres_diag_keys = {
        "delta_q_norm_mean",
        "delta_q_norm_std",
        "smooth_metric",
        "lambda_reg",
        "grad_cos_ppo_supervised",
        "grad_norm_ratio_ppo_to_supervised",
    }
    _stage1_dz_keys     = {"loss_dz", "dz_pred_abs", "dz_gt_abs", "dz_mae"}
    for key, value in locs["loss_dict"].items():
        if key in _suppress_if_zero and value == 0.0:
            continue
        if self.training_type == "supervise" and key in _stage1_dz_keys:
            self.writer.add_scalar(f"Stage1/DeltaZ/{key}", value, locs["it"])
        elif isinstance(self.alg.policy, FrontRESActorCritic) and key in _frontres_diag_keys:
            self.writer.add_scalar(f"FrontRES/{key}", value, locs["it"])
        elif key in _to_frontres:
            self.writer.add_scalar(f"FrontRES/{key}", value, locs["it"])
        elif key in _to_curriculum:
            self.writer.add_scalar(f"Curriculum/{key}", value, locs["it"])
        else:
            self.writer.add_scalar(f"Loss/{key}", value, locs["it"])
    self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

    # -- Policy (not meaningful during supervised learning)
    if self.training_type != "supervise":
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])

    # -- FrontRES B1 delta-reward + curriculum diagnostics
    if isinstance(self.alg.policy, FrontRESActorCritic):
        _ts_mode = getattr(self.alg.policy, 'num_task_corrections', 0) > 0

        # B1 delta-reward
        if locs.get("frontres_rdelta_mean") is not None:
            self.writer.add_scalar("FrontRES/r_delta_per_step",
                                   locs["frontres_rdelta_mean"], locs["it"])
        if locs.get("frontres_baseline_mean") is not None:
            self.writer.add_scalar("FrontRES/baseline_per_step",
                                   locs["frontres_baseline_mean"], locs["it"])
        for _name in (
            "r_exec", "r_geom", "r_rescue", "intervention_cost",
            "clean_bound_cost", "clean_bound_side_cost", "over_cost", "under_repair_cost",
            "reward_frontres", "reward_clean", "reward_candidate", "reward_oracle",
            "exec_planar", "exec_vertical", "exec_task",
            "damage_gap", "oracle_clean_gap", "oracle_trust",
            "repair_gain", "candidate_gain", "projection_gain", "underwrite",
            "oracle_ub_gain", "oracle_ub_pass",
            "oracle_ub_projected_win", "oracle_ub_candidate_win",
            "oracle_ub_feasible_win", "oracle_ub_noisy_win",
            "accept_pref_mask", "accept_pref_full", "accept_pref_noop",
            "accept_pref_keep", "accept_pref_ignore", "accept_pref_margin",
            "positive_gain_frac", "repair_ratio",
            "exec_signal", "weighted_exec_signal", "train_reward",
            "effective_gain_bonus", "safe_cost", "repair_cost", "broken_cost",
            "reward_progress", "constraint_progress",
            "behavior_fit", "repair_fit_rate", "repair_fit_gain",
            "restore_ratio_rp", "residual_rp_abs", "corr_roll_bias", "corr_pitch_bias",
            "harm_rate", "harm_mag", "safe_harm_rate", "broken_harm_rate",
            "safe_abstain_cost", "broken_abstain_cost",
            "window_mu", "safe_frac", "repair_frac", "broken_frac",
            "candidate_floor_margin", "candidate_floor_pass",
            "exec_floor_value", "exec_floor_safe", "exec_floor_adaptive",
            "exec_floor_safe_count", "exec_floor_broken_count",
            "stable_route_frac",
            "stable_endpoint_frac", "tri_weight_repair", "tri_weight_noisy", "tri_weight_stable",
            "structured_joint_rho_adv", "structured_joint_rho_weight",
            "structured_joint_rho_retention",
            "structured_joint_floor_violation", "structured_joint_full_bonus",
            "rho_regret_up_planar", "rho_regret_up_rp", "rho_regret_up_z",
            "rho_regret_down_planar", "rho_regret_down_rp", "rho_regret_down_z",
            "actor_gate", "exec_gate", "cost_gate",
            "r_z", "r_xy", "r_rp", "r_yaw",
            "dr_z_abs", "dr_xy_abs", "dr_rp_abs", "dr_yaw_abs",
            "corr_z_abs", "corr_xy_abs", "corr_rp_abs", "corr_yaw_abs",
        ):
            _value = locs.get(f"frontres_{_name}_mean")
            if _value is not None:
                self.writer.add_scalar(f"FrontRES/RewardComponents/{_name}",
                                       _value, locs["it"])
        _complexity = locs.get("frontres_perturb_complexity")
        if _complexity is not None:
            _complexity_id = {"single": 1.0, "two": 2.0, "three": 3.0, "full": 4.0}.get(_complexity)
            if _complexity_id is not None:
                self.writer.add_scalar("FrontRES/PerturbationCurriculum/complexity",
                                       _complexity_id, locs["it"])

        # Correction magnitude (task-space: split Δpos/Δrpy; joint-space: Δz)
        if _ts_mode:
            if locs.get("frontres_delta_pos_abs_mean") is not None:
                self.writer.add_scalar("FrontRES/delta_pos_abs_mean",
                                       locs["frontres_delta_pos_abs_mean"], locs["it"])
            if locs.get("frontres_delta_rpy_abs_mean") is not None:
                self.writer.add_scalar("FrontRES/delta_rpy_abs_mean",
                                       locs["frontres_delta_rpy_abs_mean"], locs["it"])
        else:
            if locs.get("frontres_delta_z_abs_mean") is not None:
                self.writer.add_scalar("FrontRES/delta_z_abs_mean",
                                       locs["frontres_delta_z_abs_mean"], locs["it"])

        # Optional shaping penalties (only log when non-zero)
        if locs.get("frontres_smooth_penalty_mean") not in (None, 0.0):
            self.writer.add_scalar("FrontRES/smooth_penalty_per_step",
                                   locs["frontres_smooth_penalty_mean"], locs["it"])
        if locs.get("frontres_reg_penalty_mean") not in (None, 0.0):
            self.writer.add_scalar("FrontRES/reg_penalty_per_step",
                                   locs["frontres_reg_penalty_mean"], locs["it"])

        # Jump-degree gate
        if locs.get("frontres_jump_degree_mean") is not None:
            self.writer.add_scalar("FrontRES/jump_degree_mean",
                                   locs["frontres_jump_degree_mean"], locs["it"])

        # Curriculum / DR schedule
        if locs.get("frontres_dr_scale") is not None:
            self.writer.add_scalar("Curriculum/dr_scale",
                                   locs["frontres_dr_scale"], locs["it"])
            _frontier_scale = getattr(self, "_frontres_dr_frontier_scale", None)
            if _frontier_scale is not None:
                self.writer.add_scalar("Curriculum/dr_frontier_scale",
                                       float(_frontier_scale), locs["it"])
            _gmt_safe = getattr(self, "_frontres_gmt_frontier_safe_low", None)
            _gmt_broken = getattr(self, "_frontres_gmt_frontier_broken_high", None)
            _gmt_score = getattr(self, "_frontres_gmt_frontier_probe_score", None)
            _gmt_probe = getattr(self, "_frontres_gmt_frontier_probe_scale", None)
            _gmt_confirmed = getattr(self, "_frontres_gmt_frontier_confirmed", None)
            if _gmt_safe is not None:
                self.writer.add_scalar("Curriculum/gmt_frontier_safe_low",
                                       float(_gmt_safe), locs["it"])
            if _gmt_broken is not None:
                self.writer.add_scalar("Curriculum/gmt_frontier_broken_high",
                                       float(_gmt_broken), locs["it"])
            if _gmt_score is not None:
                self.writer.add_scalar("Curriculum/gmt_frontier_probe_score",
                                       float(_gmt_score), locs["it"])
            if _gmt_probe is not None:
                self.writer.add_scalar("Curriculum/gmt_frontier_next_probe",
                                       float(_gmt_probe), locs["it"])
            if _gmt_confirmed is not None:
                self.writer.add_scalar("Curriculum/gmt_frontier_confirmed",
                                       float(_gmt_confirmed), locs["it"])
            _mix_easy = getattr(self, "_frontres_dr_mix_easy_frac", None)
            _mix_frontier = getattr(self, "_frontres_dr_mix_frontier_frac", None)
            _mix_hard = getattr(self, "_frontres_dr_mix_hard_frac", None)
            _mix_mean = getattr(self, "_frontres_dr_mix_mean_scale", None)
            if _mix_easy is not None:
                self.writer.add_scalar("Curriculum/dr_mix_easy_frac", float(_mix_easy), locs["it"])
            if _mix_frontier is not None:
                self.writer.add_scalar("Curriculum/dr_mix_frontier_frac", float(_mix_frontier), locs["it"])
            if _mix_hard is not None:
                self.writer.add_scalar("Curriculum/dr_mix_hard_frac", float(_mix_hard), locs["it"])
            if _mix_mean is not None:
                self.writer.add_scalar("Curriculum/dr_mix_mean_scale", float(_mix_mean), locs["it"])
        if locs.get("frontres_survival_rate") is not None:
            self.writer.add_scalar("Curriculum/training_survival_rate",
                                   locs["frontres_survival_rate"], locs["it"])
        self.writer.add_scalar("Curriculum/r_delta_ema",
                               locs.get("_r_delta_ema", 0.0), locs["it"])

        # Δq alpha ramp (legacy joint-space mode only)
    # -- Performance
    self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
    self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
    self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

    # -- Training (RL metrics: skip during supervised learning to avoid confusing oscillation)
    if self.training_type != "supervise" and len(locs["rewbuffer"]) > 0:
        # separate logging for intrinsic and extrinsic rewards
        if hasattr(self.alg, "rnd") and self.alg.rnd:
            self.writer.add_scalar("Rnd/mean_extrinsic_reward", statistics.mean(locs["erewbuffer"]), locs["it"])
            self.writer.add_scalar("Rnd/mean_intrinsic_reward", statistics.mean(locs["irewbuffer"]), locs["it"])
            self.writer.add_scalar("Rnd/weight", self.alg.rnd.weight, locs["it"])

        # everything else
        if isinstance(self.alg.policy, FrontRESActorCritic):
            # B1: rewbuffer = FrontRES r_delta per episode; rewbuffer_gmt = GMT raw reward
            self.writer.add_scalar("Train/mean_r_delta",   statistics.mean(locs["rewbuffer"]),     locs["it"])
            if len(locs.get("rewbuffer_gmt", [])) > 0:
                self.writer.add_scalar("Train/mean_reward_gmt", statistics.mean(locs["rewbuffer_gmt"]), locs["it"])
            if len(locs.get("lenbuffer_gmt", [])) > 0:
                self.writer.add_scalar("Train/mean_episode_length_gmt",
                                       statistics.mean(locs["lenbuffer_gmt"]), locs["it"])
        else:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
        self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
        if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
            self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
            self.writer.add_scalar("Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time)

    iter_title = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

    if self.training_type != "supervise" and len(locs["rewbuffer"]) > 0:
        # ── Phase indicator ──────────────────────────────────────────────
        _ts_mode = getattr(self.alg.policy, 'num_task_corrections', 0) > 0
        if _ts_mode:
            _is_warmup = locs.get("_supervised_warmup_active", False)
            _is_critic = locs.get("_critic_warmup_active", False)
            _is_actor_takeover = locs.get("_actor_takeover_active", False)
            _lam = getattr(self.alg, 'lambda_supervised', 0.0)
            _objective = getattr(self.alg, "frontres_training_objective", "")
            if f"{_objective}".lower() == "supervised_restore":
                _phase = "SUPERVISED RESTORE"
                _notes = "(PPO/HRL update disabled; fitting clean restoration target)"
            elif f"{_objective}".lower() == "basis_restore":
                _phase = "BASIS RESTORE"
                _notes = "(PPO/HRL update disabled; factorized repair coefficients)"
            elif _is_warmup:
                _phase = "SUPERVISED WARMUP"
                _notes = "(GMT-only, FrontRES corrections disabled)"
            elif _is_critic:
                _phase = "CRITIC WARMUP"
                _paw = locs.get("loss_dict", {}).get("ppo_actor_weight", None)
                if _paw is not None and _paw <= 0.0:
                    _notes = "(fixed low DR, PPO actor frozen; critic + supervised train)"
                else:
                    _notes = "(fixed low DR; critic + supervised train)"
            elif _is_actor_takeover:
                _phase = "ACTOR TAKEOVER"
                _notes = "(fixed DR; PPO actor weight ramping)"
            elif _lam > 0.5:
                _phase = "PPO + SUPERVISED ANCHOR"
                _notes = ""
            elif _lam > 0.15:
                _phase = "PPO + WEAK SUPERVISION"
                _notes = ""
            else:
                _phase = "PPO FINE-TUNING"
                _notes = ""
        else:
            _phase = "PPO"
            _notes = ""
        _phase_str = f"  PHASE: {_phase}  "
        if _notes:
            _phase_str += f"\n  {_notes}  "

        _is_frontres_policy = isinstance(self.alg.policy, FrontRESActorCritic)
        if _is_frontres_policy:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{iter_title.center(width, ' ')}\n"""
                f"""{_phase_str.center(width, ' ')}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{iter_title.center(width, ' ')}\n"""
                f"""{_phase_str.center(width, ' ')}\n\n"""
                f"""{'─' * 30} PERFORMANCE {'─' * 30}\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                    'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")

        # ── FrontRES: r_delta + cos_sim + curriculum (compact) ──────────
        if _is_frontres_policy:
            log_string += f"""\n{'-' * 12} Performance {'-' * 12}\n"""
            log_string += f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
            log_string += f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            log_string += f"""{'ep_len_FrontRES:':>{pad}} {statistics.mean(locs['lenbuffer']):.1f}\n"""
            if len(locs.get("lenbuffer_gmt", [])) > 0:
                log_string += f"""{'ep_len_GMT (baseline):':>{pad}} {statistics.mean(locs['lenbuffer_gmt']):.1f}\n"""
            if locs.get("frontres_perturb_complexity") is not None:
                log_string += f"""{'perturb curriculum:':>{pad}} """
                log_string += (
                    f"{locs['frontres_perturb_complexity']} "
                    f"[{locs.get('frontres_perturb_modes', '')}]\n"
                )
            if locs.get("frontres_dr_scale") is not None:
                log_string += f"""{'DR scale:':>{pad}} {locs['frontres_dr_scale']:.4f}\n"""
                _frontier_scale = getattr(self, "_frontres_dr_frontier_scale", None)
                if _frontier_scale is None:
                    _frontier_scale = locs["frontres_dr_scale"]
                _mix_mode = getattr(self, "_frontres_dr_mix_mode", None)
                if _mix_mode is None:
                    _mix_mode = "fixed"
                log_string += f"""{'DR frontier/mix:':>{pad}} """
                log_string += f"{_frontier_scale:.4f} / {_mix_mode}\n"
                _mix_easy = getattr(self, "_frontres_dr_mix_easy_frac", None)
                _mix_frontier = getattr(self, "_frontres_dr_mix_frontier_frac", None)
                _mix_hard = getattr(self, "_frontres_dr_mix_hard_frac", None)
                if _mix_easy is not None and _mix_frontier is not None and _mix_hard is not None:
                    log_string += f"""{'DR train mix e/f/h:':>{pad}} """
                    log_string += f"{float(_mix_easy):.3f} / {float(_mix_frontier):.3f} / {float(_mix_hard):.3f}\n"
                _mix_mean = getattr(self, "_frontres_dr_mix_mean_scale", None)
                if _mix_mean is not None:
                    log_string += f"""{'DR train scale mean:':>{pad}} {float(_mix_mean):.4f}\n"""
                _gmt_score = getattr(self, "_frontres_gmt_frontier_probe_score", None)
                _gmt_decision = getattr(self, "_frontres_gmt_frontier_decision", None)
                _gmt_probe = getattr(self, "_frontres_gmt_frontier_probe_scale", None)
                _gmt_safe = getattr(self, "_frontres_gmt_frontier_safe_low", None)
                _gmt_broken = getattr(self, "_frontres_gmt_frontier_broken_high", None)
                if _gmt_score is not None or _gmt_decision is not None:
                    log_string += f"""{'GMT frontier probe:':>{pad}} """
                    _score_s = "n/a" if _gmt_score is None else f"{float(_gmt_score):.3f}"
                    _probe_s = "n/a" if _gmt_probe is None else f"{float(_gmt_probe):.4f}"
                    _decision_s = _gmt_decision or "n/a"
                    _probe_src = getattr(self, "_frontres_gmt_frontier_probe_source", None)
                    _probe_n = getattr(self, "_frontres_gmt_frontier_probe_samples", None)
                    _src_s = "" if _probe_src is None else f", src={_probe_src}"
                    _n_s = "" if _probe_n is None else f", n={int(_probe_n)}"
                    log_string += f"next={_probe_s}, score={_score_s}, decision={_decision_s}{_src_s}{_n_s}\n"
                if _gmt_safe is not None:
                    log_string += f"""{'GMT bracket safe/broken:':>{pad}} """
                    _broken_s = "open" if _gmt_broken is None else f"{float(_gmt_broken):.4f}"
                    log_string += f"{float(_gmt_safe):.4f} / {_broken_s}\n"
            if locs.get("frontres_survival_rate") is not None:
                log_string += f"""{'survival rate:':>{pad}} {locs['frontres_survival_rate']:.3f}\n"""

            _frontres_supervised_log = (
                f"{getattr(self.alg, 'frontres_training_objective', '')}".lower()
                in ("supervised_restore", "basis_restore", "hsl_hybrid")
            )
            _loss_dict = locs.get("loss_dict", {})

            if _frontres_supervised_log:
                log_string += f"""\n{'-' * 9} Supervised Restore {'-' * 9}\n"""
                _cs = _loss_dict.get("supervised_cos_sim", None)
                if _cs is not None:
                    log_string += f"""{'supervised_cos_sim:':>{pad}} {_cs:.4f}\n"""
                _sup_restore = _loss_dict.get("supervised_restore_ratio", None)
                if _sup_restore is not None:
                    log_string += f"""{'restore ratio:':>{pad}} {_sup_restore:+.3f}\n"""
                if _loss_dict.get("supervised_mae", None) is not None:
                    log_string += f"""{'mae/rmse all:':>{pad}} """
                    log_string += (
                        f"{_loss_dict.get('supervised_mae', 0.0):.5f} / "
                        f"{_loss_dict.get('supervised_rmse', 0.0):.5f}\n"
                    )
                if _loss_dict.get("supervised_rpy_mae", None) is not None:
                    log_string += f"""{'mae/rmse rpy:':>{pad}} """
                    log_string += (
                        f"{_loss_dict.get('supervised_rpy_mae', 0.0):.5f} / "
                        f"{_loss_dict.get('supervised_rpy_rmse', 0.0):.5f}\n"
                    )
                if _loss_dict.get("supervised_valid_frac", None) is not None:
                    log_string += f"""{'valid target frac:':>{pad}} {_loss_dict.get('supervised_valid_frac', 0.0):.3f}\n"""
                if _loss_dict.get("supervised_l_pos", None) is not None:
                    log_string += f"""{'L_pos/L_rot:':>{pad}} """
                    log_string += (
                        f"{_loss_dict.get('supervised_l_pos', 0.0):.6f} / "
                        f"{_loss_dict.get('supervised_l_rot', 0.0):.6f}\n"
                    )
                    log_string += f"""{'L_mag/over/smooth:':>{pad}} """
                    log_string += (
                        f"{_loss_dict.get('supervised_l_mag', 0.0):.6f} / "
                        f"{_loss_dict.get('supervised_l_over', 0.0):.6f} / "
                        f"{_loss_dict.get('supervised_l_smooth', 0.0):.6f}\n"
                    )
                    log_string += f"""{'L_harm/conf:':>{pad}} """
                    log_string += (
                        f"{_loss_dict.get('supervised_l_harm', 0.0):.6f} / "
                        f"{_loss_dict.get('supervised_l_conf', 0.0):.6f}\n"
                    )
                if f"{getattr(self.alg, 'frontres_training_objective', '')}".lower() in ("basis_restore", "hsl_hybrid"):
                    if int(getattr(getattr(self.alg, "policy", None), "task_conf_dim", 2)) == 1:
                        log_string += f"""{'pos rejoin/active:':>{pad}} """
                        log_string += (
                            f"{_loss_dict.get('frontres_rho_pos_mean', _loss_dict.get('frontres_tau_mean', 0.0)):.3f} / "
                            f"{_loss_dict.get('frontres_rho_pos_active_frac', _loss_dict.get('frontres_tau_active_frac', 0.0)):.3f}\n"
                        )
                    elif int(getattr(getattr(self.alg, "policy", None), "task_conf_dim", 2)) == 6:
                        log_string += f"""{'accept pos/rpy:':>{pad}} """
                        log_string += (
                            f"{_loss_dict.get('frontres_accept_pos_mean', 0.0):.3f} / "
                            f"{_loss_dict.get('frontres_accept_rpy_mean', 0.0):.3f}\n"
                        )
                    else:
                        log_string += f"""{'alpha mean/active:':>{pad}} """
                        log_string += (
                            f"{_loss_dict.get('frontres_alpha_mean', 0.0):.3f} / "
                            f"{_loss_dict.get('frontres_alpha_active_frac', 0.0):.3f}\n"
                        )
                    ratio_label = "proposal ratio/leakage"
                    ratio_value = _loss_dict.get(
                        "frontres_proposal_ratio",
                        _loss_dict.get("frontres_write_ratio", 0.0),
                    )
                    log_string += f"""{ratio_label + ':':>{pad}} """
                    log_string += (
                        f"{ratio_value:.3f} / "
                        f"{_loss_dict.get('frontres_axis_leakage', 0.0):.3f}\n"
                    )
                    log_string += f"""{'L_sparse/miss/csmooth:':>{pad}} """
                    log_string += (
                        f"{_loss_dict.get('supervised_l_sparse', 0.0):.6f} / "
                        f"{_loss_dict.get('supervised_l_miss', 0.0):.6f} / "
                        f"{_loss_dict.get('supervised_l_coeff_smooth', 0.0):.6f}\n"
                    )
                    if locs.get("frontres_candidate_gain_mean") is not None:
                        log_string += f"""{"gain proj/cand/bound:":>{pad}} """
                        log_string += (
                            f"{locs['frontres_repair_gain_mean']:+.4f} / "
                            f"{locs['frontres_candidate_gain_mean']:+.4f} / "
                            f"{locs['frontres_projection_gain_mean']:+.4f} "
                            f"(under={locs['frontres_underwrite_mean']:+.4f})\n"
                        )
                    log_string += format_frontres_floor_alpha_diagnostics(locs, _loss_dict, pad=pad)
                    log_string += format_frontres_route_rho_diagnostics(locs, self.cfg, pad=pad)
                    log_string += format_frontres_preference_diagnostics(
                        locs,
                        _loss_dict,
                        self.cfg,
                        pad=pad,
                        structured_label="joint adv pos/neg/near/ign:",
                    )

                log_string += f"""\n{'-' * 10} Correction Geometry {'-' * 10}\n"""
                if locs.get("frontres_delta_pos_abs_mean") is not None:
                    log_string += f"""{'|Δpos|:':>{pad}} {locs['frontres_delta_pos_abs_mean']:.4f} m\n"""
                if locs.get("frontres_delta_rpy_abs_mean") is not None:
                    log_string += f"""{'|Δrpy|:':>{pad}} {locs['frontres_delta_rpy_abs_mean']:.4f} rad\n"""
                if locs.get("frontres_restore_ratio_rp_mean") is not None:
                    log_string += f"""{'restore rp/res/bias:':>{pad}} """
                    log_string += (
                        f"{locs['frontres_restore_ratio_rp_mean']:+.3f} / "
                        f"{locs['frontres_residual_rp_abs_mean']:.4f} / "
                        f"{locs['frontres_corr_roll_bias_mean']:+.4f}, "
                        f"{locs['frontres_corr_pitch_bias_mean']:+.4f}\n"
                    )

                log_string += f"""\n{'-' * 10} Optimization / Update {'-' * 10}\n"""
                _lam = _loss_dict.get("lambda_supervised", None)
                if _lam is not None:
                    log_string += f"""{'λ_supervised:':>{pad}} {_lam:.3f}\n"""
                _paw = _loss_dict.get("ppo_actor_weight", None)
                if _paw is not None:
                    log_string += f"""{'PPO actor weight:':>{pad}} {_paw:.3f}\n"""
                _apl = _loss_dict.get("acceptance_preference_loss", None)
                if _apl is not None:
                    _legacy_pref_disabled = (
                        self._frontres_structured_joint_effective_enabled()
                        and not bool(self.cfg.get("frontres_structured_joint_rl_keep_legacy_bce", False))
                        and float(_loss_dict.get("lambda_acceptance_preference", 0.0)) <= 0.0
                    )
                    if _legacy_pref_disabled:
                        pass
                    else:
                        log_string += f"""{'accept pref loss:':>{pad}} {_apl:.4f} """
                        _low_target_label = (
                            "stable"
                            if str(self.cfg.get("frontres_rho_space", "noisy_to_repair")).lower()
                            in ("stable_to_repair", "stable-repair", "stable")
                            else (
                                "noisy"
                                if str(self.cfg.get("frontres_rho_space", "noisy_to_repair")).lower()
                                in ("tri_anchor", "tri-anchor", "tri")
                                else "noop"
                            )
                        )
                        log_string += (
                            f"(λ={_loss_dict.get('lambda_acceptance_preference', 0.0):.3f}, "
                            f"mask={_loss_dict.get('acceptance_preference_mask_frac', 0.0):.3f}, "
                            f"full={_loss_dict.get('acceptance_preference_full_frac', 0.0):.3f}, "
                            f"{_low_target_label}={_loss_dict.get('acceptance_preference_noop_frac', 0.0):.3f}, "
                            f"eff_full={_loss_dict.get('acceptance_preference_effective_full_frac', 0.0):.3f}, "
                            f"fw={_loss_dict.get('acceptance_preference_full_weight', 1.0):.2f}, "
                            f"low_w={_loss_dict.get('acceptance_preference_noop_weight', 1.0):.2f}, "
                            f"γ={_loss_dict.get('acceptance_preference_focal_gamma', 0.0):.1f}, "
                            f"rho={_loss_dict.get('acceptance_preference_rho_mean', 0.0):.3f}, "
                            f"err={_loss_dict.get('acceptance_preference_abs_err', 0.0):.3f}, "
                            f"corr={_loss_dict.get('acceptance_preference_corr', 0.0):+.3f})\n"
                        )
                log_string += format_frontres_optimization_diagnostics(_loss_dict, pad=pad)
                log_string += f"""{'learning rate:':>{pad}} {getattr(self.alg, 'learning_rate', 0.0):.2e}\n"""
                _objective_name = f"{getattr(self.alg, 'frontres_training_objective', '')}".lower()
                if _objective_name == "hsl_hybrid":
                    _objective_desc = "HSL ΔSE proposal + PPO 6D acceptance"
                elif _objective_name == "basis_restore":
                    _objective_desc = "basis supervised only"
                else:
                    _objective_desc = "supervised only"
                log_string += f"""{'objective:':>{pad}} {_objective_desc}\n"""

            else:
                log_string += f"""\n{'-' * 12} Main Reward {'-' * 12}\n"""
                log_string += f"""{'r_delta (FrontRES):':>{pad}} {statistics.mean(locs['rewbuffer']):.4f}\n"""
                if len(locs.get("rewbuffer_gmt", [])) > 0:
                    log_string += f"""{'reward_GMT (baseline):':>{pad}} {statistics.mean(locs['rewbuffer_gmt']):.4f}\n"""
                _cs = _loss_dict.get("supervised_cos_sim", None)
                if _cs is not None:
                    log_string += f"""{'supervised_cos_sim:':>{pad}} {_cs:.4f}\n"""
                if locs.get("frontres_delta_pos_abs_mean") is not None:
                    log_string += f"""{'|Δpos|:':>{pad}} {locs['frontres_delta_pos_abs_mean']:.4f} m\n"""
                if locs.get("frontres_delta_rpy_abs_mean") is not None:
                    log_string += f"""{'|Δrpy|:':>{pad}} {locs['frontres_delta_rpy_abs_mean']:.4f} rad\n"""
                if locs.get("frontres_r_exec_mean") is not None:
                    if locs.get("frontres_damage_gap_mean") is not None:
                        log_string += f"""{'gap/gain/ratio:':>{pad}} """
                        log_string += (
                            f"{locs['frontres_damage_gap_mean']:+.4f} / "
                            f"{locs['frontres_repair_gain_mean']:+.4f} / "
                            f"{locs['frontres_repair_ratio_mean']:+.4f}\n"
                        )
                        if locs.get("frontres_train_reward_mean") is not None:
                            log_string += f"""{'signal/w_signal/train_r:':>{pad}} """
                            log_string += (
                                f"{locs['frontres_exec_signal_mean']:+.4f} / "
                                f"{locs['frontres_weighted_exec_signal_mean']:+.4f} / "
                                f"{locs['frontres_train_reward_mean']:+.4f}\n"
                            )
                            log_string += f"""{'Clean/Cand/Oracle/Trust:':>{pad}} """
                            log_string += (
                                f"{locs['frontres_reward_clean_mean']:+.4f} / "
                                f"{locs['frontres_reward_candidate_mean']:+.4f} / "
                                f"{locs['frontres_reward_oracle_mean']:+.4f} / "
                                f"{locs['frontres_oracle_trust_mean']:.3f}\n"
                            )
                            log_string += f"""{'oracle gap/cost:':>{pad}} """
                            log_string += (
                                f"{locs['frontres_oracle_clean_gap_mean']:+.4f} / "
                                f"{locs['frontres_clean_bound_cost_mean']:.4f}\n"
                            )
                            log_string += f"""{'side/over/under:':>{pad}} """
                            log_string += (
                                f"{locs['frontres_clean_bound_side_cost_mean']:.4f} / "
                                f"{locs['frontres_over_cost_mean']:.4f} / "
                                f"{locs['frontres_under_repair_cost_mean']:.4f}\n"
                            )
                            log_string += f"""{'bonus/legacy S/R/B:':>{pad}} """
                            log_string += (
                                f"{locs['frontres_effective_gain_bonus_mean']:+.4f} / "
                                f"{locs['frontres_safe_cost_mean']:.4f} / "
                                f"{locs['frontres_repair_cost_mean']:.4f} / "
                                f"{locs['frontres_broken_cost_mean']:.4f}\n"
                            )
                            log_string += f"""{'reward/constraint prog:':>{pad}} """
                            log_string += (
                                f"{locs['frontres_reward_progress_mean']:.4f} / "
                                f"{locs['frontres_constraint_progress_mean']:.4f}\n"
                            )
                        if locs.get("frontres_behavior_fit_mean") is not None:
                            log_string += f"""{'exec legacy fit:':>{pad}} """
                            log_string += (
                                f"{locs['frontres_behavior_fit_mean']:+.3f} / "
                                f"{locs['frontres_repair_fit_rate_mean']:+.3f} / "
                                f"{locs['frontres_repair_fit_gain_mean']:+.4f}\n"
                            )
                            log_string += f"""{'restore rp/res/bias:':>{pad}} """
                            log_string += (
                                f"{locs['frontres_restore_ratio_rp_mean']:+.3f} / "
                                f"{locs['frontres_residual_rp_abs_mean']:.4f} / "
                                f"{locs['frontres_corr_roll_bias_mean']:+.4f}, "
                                f"{locs['frontres_corr_pitch_bias_mean']:+.4f}\n"
                            )
                            log_string += f"""{'harm rate/mag:':>{pad}} """
                            log_string += (
                                f"{locs['frontres_harm_rate_mean']:.3f} / "
                                f"{locs['frontres_harm_mag_mean']:.4f}\n"
                            )
                            log_string += f"""{'safe/broken harm:':>{pad}} """
                            log_string += (
                                f"{locs['frontres_safe_harm_rate_mean']:.3f} / "
                                f"{locs['frontres_broken_harm_rate_mean']:.3f}\n"
                            )
                            log_string += f"""{'safe/broken abstain:':>{pad}} """
                            log_string += (
                                f"{locs['frontres_safe_abstain_cost_mean']:.4f} / "
                                f"{locs['frontres_broken_abstain_cost_mean']:.4f}\n"
                            )
                        if locs.get("frontres_positive_gain_frac_mean") is not None:
                            log_string += f"""{'positive_gain_frac:':>{pad}} """
                            log_string += f"{locs['frontres_positive_gain_frac_mean']:.3f}\n"
                        if locs.get("frontres_oracle_ub_gain_mean") is not None:
                            log_string += f"""{'oracle ub gain/src:':>{pad}} """
                            log_string += (
                                f"{locs['frontres_oracle_ub_gain_mean']:+.4f} / "
                                f"{locs['frontres_oracle_ub_projected_win_mean']:.3f} / "
                                f"{locs['frontres_oracle_ub_candidate_win_mean']:.3f} / "
                                f"{locs['frontres_oracle_ub_feasible_win_mean']:.3f} / "
                                f"{locs['frontres_oracle_ub_noisy_win_mean']:.3f}\n"
                            )
                            log_string += f"""{'oracle ub pass:':>{pad}} """
                            log_string += f"{locs['frontres_oracle_ub_pass_mean']:.3f}\n"
                        log_string += f"""{'safe/repair/broken frac:':>{pad}} """
                        log_string += (
                            f"{locs['frontres_safe_frac_mean']:.3f} / "
                            f"{locs['frontres_repair_frac_mean']:.3f} / "
                            f"{locs['frontres_broken_frac_mean']:.3f}\n"
                        )
                        log_string += format_frontres_floor_alpha_diagnostics(locs, _loss_dict, pad=pad)
                        log_string += format_frontres_route_rho_diagnostics(locs, self.cfg, pad=pad)

                    log_string += f"""\n{'-' * 12} Detail Reward {'-' * 12}\n"""
                    if locs.get("frontres_r_z_mean") is not None:
                        log_string += f"""{'r_z/r_xy/r_rp/r_yaw:':>{pad}} """
                        log_string += (
                            f"{locs['frontres_r_z_mean']:+.4f} / "
                            f"{locs['frontres_r_xy_mean']:+.4f} / "
                            f"{locs['frontres_r_rp_mean']:+.4f} / "
                            f"{locs['frontres_r_yaw_mean']:+.4f}\n"
                        )
                    log_string += f"""{'repair/geom/rescue/action_cost:':>{pad}} """
                    log_string += (
                        f"{locs['frontres_r_exec_mean']:+.4f} / "
                        f"{locs['frontres_r_geom_mean']:+.4f} / "
                        f"{locs['frontres_r_rescue_mean']:+.4f} / "
                        f"{locs['frontres_intervention_cost_mean']:+.4f}\n"
                    )
                    if locs.get("frontres_reward_frontres_mean") is not None and locs.get("frontres_baseline_mean") is not None:
                        log_string += f"""{'exec reward FR/cand/pert:':>{pad}} """
                        log_string += (
                            f"{locs['frontres_reward_frontres_mean']:+.4f} / "
                            f"{locs.get('frontres_reward_candidate_mean', 0.0):+.4f} / "
                            f"{locs['frontres_baseline_mean']:+.4f}\n"
                        )
                    if locs.get("frontres_candidate_gain_mean") is not None:
                        log_string += f"""{'gain proj/cand/bound:':>{pad}} """
                        log_string += (
                            f"{locs['frontres_repair_gain_mean']:+.4f} / "
                            f"{locs['frontres_candidate_gain_mean']:+.4f} / "
                            f"{locs['frontres_projection_gain_mean']:+.4f} "
                            f"(under={locs['frontres_underwrite_mean']:+.4f})\n"
                        )
                    log_string += format_frontres_preference_diagnostics(
                        locs,
                        _loss_dict,
                        self.cfg,
                        pad=pad,
                        structured_label="rho adv pos/neg/near/ign:",
                    )
                    if locs.get("frontres_exec_planar_mean") is not None:
                        log_string += f"""{'exec planar/vertical/task:':>{pad}} """
                        log_string += (
                            f"{locs['frontres_exec_planar_mean']:+.4f} / "
                            f"{locs['frontres_exec_vertical_mean']:+.4f} / "
                            f"{locs['frontres_exec_task_mean']:+.4f}\n"
                        )
                    if locs.get("frontres_window_mu_mean") is not None:
                        log_string += f"""{'exec/cost gate:':>{pad}} """
                        log_string += (
                            f"{locs['frontres_exec_gate_mean']:.3f} / "
                            f"{locs['frontres_cost_gate_mean']:.3f}\n"
                        )

                log_string += f"""\n{'-' * 10} Optimization / Update {'-' * 10}\n"""
                if locs.get("frontres_window_mu_mean") is not None:
                    log_string += f"""{'mu (reward window):':>{pad}} {locs['frontres_window_mu_mean']:.3f}\n"""
                    if not self._frontres_structured_joint_effective_enabled():
                        log_string += f"""{'actor sample weight:':>{pad}} {locs['frontres_actor_gate_mean']:.3f}\n"""
                _gc = _loss_dict.get("grad_cos_ppo_supervised", None)
                if _gc is not None:
                    _gr = _loss_dict.get("grad_norm_ratio_ppo_to_supervised", 0.0)
                    log_string += f"""{'grad cos PPO/Sup:':>{pad}} {_gc:+.4f} (norm ratio={_gr:.3f})\n"""
                _rd_ema = locs.get("_r_delta_ema", 0.0)
                log_string += f"""{'r_delta EMA:':>{pad}} {_rd_ema:.4f}\n"""
                _lam = _loss_dict.get("lambda_supervised", None)
                if _lam is not None:
                    log_string += f"""{'λ_supervised:':>{pad}} {_lam:.3f}\n"""
                _paw = _loss_dict.get("ppo_actor_weight", None)
                if _paw is not None:
                    log_string += f"""{'PPO actor weight:':>{pad}} {_paw:.3f}\n"""
                _apl = _loss_dict.get("acceptance_preference_loss", None)
                if _apl is not None:
                    _legacy_pref_disabled = (
                        self._frontres_structured_joint_effective_enabled()
                        and not bool(self.cfg.get("frontres_structured_joint_rl_keep_legacy_bce", False))
                        and float(_loss_dict.get("lambda_acceptance_preference", 0.0)) <= 0.0
                    )
                    if not _legacy_pref_disabled:
                        log_string += f"""{'accept pref loss:':>{pad}} {_apl:.4f} """
                        _low_target_label = (
                            "stable"
                            if str(self.cfg.get("frontres_rho_space", "noisy_to_repair")).lower()
                            in ("stable_to_repair", "stable-repair", "stable")
                            else (
                                "noisy"
                                if str(self.cfg.get("frontres_rho_space", "noisy_to_repair")).lower()
                                in ("tri_anchor", "tri-anchor", "tri")
                                else "noop"
                            )
                        )
                        log_string += (
                            f"(λ={_loss_dict.get('lambda_acceptance_preference', 0.0):.3f}, "
                            f"mask={_loss_dict.get('acceptance_preference_mask_frac', 0.0):.3f}, "
                            f"full={_loss_dict.get('acceptance_preference_full_frac', 0.0):.3f}, "
                            f"{_low_target_label}={_loss_dict.get('acceptance_preference_noop_frac', 0.0):.3f}, "
                            f"eff_full={_loss_dict.get('acceptance_preference_effective_full_frac', 0.0):.3f}, "
                            f"fw={_loss_dict.get('acceptance_preference_full_weight', 1.0):.2f}, "
                            f"low_w={_loss_dict.get('acceptance_preference_noop_weight', 1.0):.2f}, "
                            f"γ={_loss_dict.get('acceptance_preference_focal_gamma', 0.0):.1f}, "
                            f"rho={_loss_dict.get('acceptance_preference_rho_mean', 0.0):.3f}, "
                            f"err={_loss_dict.get('acceptance_preference_abs_err', 0.0):.3f}, "
                            f"corr={_loss_dict.get('acceptance_preference_corr', 0.0):+.3f})\n"
                        )
                log_string += format_frontres_optimization_diagnostics(_loss_dict, pad=pad)
                if locs.get("frontres_window_mu_mean") is not None:
                    if not self._frontres_structured_joint_effective_enabled():
                        log_string += f"""{'actor/exec/cost:':>{pad}} """
                        log_string += (
                            f"{locs['frontres_actor_gate_mean']:.3f} / "
                            f"{locs['frontres_exec_gate_mean']:.3f} / "
                            f"{locs['frontres_cost_gate_mean']:.3f}\n"
                        )
    else:
        log_string = (
            f"""{'#' * width}\n"""
                f"""{iter_title.center(width, ' ')}\n\n"""
            f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n""")

        if self.training_type == "supervise":
            log_string += f"""{'─' * 30} STAGE 1 {'─' * 33}\n"""
            if "behavior" in locs["loss_dict"]:
                log_string += f"""{'behavior loss:':>{pad}} {locs['loss_dict']['behavior']:.4f}\n"""
            if len(locs["lenbuffer"]) > 0:
                log_string += f"""{'ep length:':>{pad}} {statistics.mean(locs['lenbuffer']):.1f}\n"""
        else:
            log_string += f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

    # Episode_Reward / Metrics / Terminations → wandb only, not console
    _footer_width = 44 if self.training_type == "frontres" else width
    log_string += (
        f"""{'-' * _footer_width}\n"""
        f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
        f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
        f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
        f"""{'ETA:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time / (locs['it'] - locs['start_iter'] + 1) * (
                           locs['start_iter'] + locs['num_learning_iterations'] - locs['it'])))}\n""")
    
    print(log_string)
