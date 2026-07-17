"""Real Stage 3 owner adapters for the independent policy-quality evaluator."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import io
import json
import math
import random
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

from rsl_rl.frontres.frontres_gain import FrontRESSegmentGainConfig, compute_segment_gain
from rsl_rl.runners.frontres_policy_quality_eval import (
    FrontRESPolicyQualityEvalRequest,
    FrontRESPolicyQualityFormalOwnerBundle,
    FrontRESPolicyQualityObservationIdentity,
    FrontRESPolicyQualityRouteHooks,
    FrontRESPolicyQualityRouteResult,
    FrozenFrontRESTaskActor,
    ZeroFrontRESTaskActor,
    capture_frontres_policy_quality_state,
)
from rsl_rl.runners.frontres_hsl_rollout_target import (
    build_frontres_hsl_rollout_target,
    quat_to_rotvec_wxyz,
)
from rsl_rl.runners.frontres_segment_live_probe import (
    _capture_motion_quality_frame,
    _capture_physics_frame,
    _capture_root_orientation_frame,
    _frontres_reset_role_env_ids,
    _index_reset_result_from_mapping,
    _index_segment_reset_hook,
)
from rsl_rl.runners.frontres_segment_live_sampler import ensure_frontres_policy_quality_reset_support
from rsl_rl.runners.frontres_training_setup import configure_frontres_pair_layout


_OWNER_IDENTITY = (
    ("reset", "frontres_segment_stage1_env_hooks.apply_frontres_segment_index_reset"),
    ("observation", "frontres_runtime.apply_obs_normalizer + checkpoint prefix/suffix stats"),
    ("action", "task_space_correction.apply_frontres_task_corrections"),
    ("rollout", "FrontRESActorCritic.get_env_action + env.step"),
    ("gain", "frontres_gain.compute_segment_gain"),
    ("execution", "frontres_segment_live_probe._capture_motion_quality_frame/_capture_physics_frame"),
)


def _checkpoint_payload(path: str, device: torch.device | str) -> Mapping[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"policy-quality checkpoint must contain a mapping: {path}")
    return payload


def _first_linear_input_dim(module: torch.nn.Module) -> int:
    for child in module.modules():
        if isinstance(child, torch.nn.Linear):
            return int(child.in_features)
    raise ValueError("policy-quality residual actor has no Linear input boundary")


def _role_layout(pair_layout: Any) -> tuple[str, ...]:
    return (
        ("policy",) * int(pair_layout.n_train)
        + ("candidate",) * int(pair_layout.n_candidate)
        + ("noisy",) * int(pair_layout.n_base)
        + ("clean",) * int(pair_layout.n_clean)
    )


def _parameter_strength(item: Any) -> float:
    values = dict(item.perturbation_parameters)
    for key in ("strength", "dr_scale", "scale"):
        if key in values:
            return float(values[key])
    raise ValueError("quality manifest perturbation_parameters must include strength, dr_scale, or scale")


def _apply_manifest_reset(runner: Any, item: Any, pair_layout: Any) -> Any:
    """Call the canonical index-reset hook from one immutable manifest item."""
    n_source = int(pair_layout.n_train)
    if n_source <= 0:
        raise ValueError("policy-quality reset requires at least one policy row")
    device = torch.device(runner.device)
    request = SimpleNamespace(
        segment_ids=torch.arange(n_source, dtype=torch.long, device=device),
        motion_ids=tuple(item.motion_id for _ in range(n_source)),
        start_frames=torch.full((n_source,), int(item.start_frame), dtype=torch.long, device=device),
        horizon_k=torch.full((n_source,), int(item.effective_horizon_k), dtype=torch.long, device=device),
        perturbation_family=tuple(item.perturbation_family for _ in range(n_source)),
        perturbation_strength=torch.full((n_source,), _parameter_strength(item), dtype=torch.float32, device=device),
        valid_mask=torch.ones(n_source, dtype=torch.bool, device=device),
        frontres_role_env_ids=_frontres_reset_role_env_ids(pair_layout, source_count=n_source, device=device),
    )
    hook = _index_segment_reset_hook(runner.env)
    if hook is None:
        raise RuntimeError("policy-quality canonical index-reset owner is unavailable")
    result = _index_reset_result_from_mapping(hook(request), request)
    if not bool(result.success_mask.all().item()):
        raise RuntimeError("policy-quality manifest reset failed for one or more source rows")
    return result


class _RouteCapture:
    def __init__(self, runner: Any, pair_layout: Any, horizon_k: int) -> None:
        self.runner = runner
        self.pair_layout = pair_layout
        self.horizon_k = int(horizon_k)
        self.raw_obs: torch.Tensor | None = None
        self.pending_action: torch.Tensor | None = None
        self.actions: list[torch.Tensor] = []
        self.clean_positions: list[torch.Tensor] = []
        self.repaired_positions: list[torch.Tensor] = []
        self.noisy_positions: list[torch.Tensor] = []
        self.clean_root_quat: list[torch.Tensor] = []
        self.repaired_root_quat: list[torch.Tensor] = []
        self.noisy_root_quat: list[torch.Tensor] = []
        self.repaired_zmp: list[torch.Tensor] = []
        self.noisy_zmp: list[torch.Tensor] = []
        self.repaired_contact: list[torch.Tensor] = []
        self.noisy_contact: list[torch.Tensor] = []
        self.repaired_survival = torch.zeros(int(pair_layout.n_train), device=runner.device)
        self.noisy_survival = torch.zeros(int(pair_layout.n_train), device=runner.device)
        self.repaired_done = torch.zeros(int(pair_layout.n_train), dtype=torch.bool, device=runner.device)
        self.noisy_done = torch.zeros(int(pair_layout.n_train), dtype=torch.bool, device=runner.device)
        self.last_gain: Any = None
        self.current_route: str | None = None
        self.hsl_command: Any = None
        self.hsl_pos_snapshot: torch.Tensor | None = None
        self.hsl_quat_snapshot: torch.Tensor | None = None
        self.hsl_targets: list[torch.Tensor] = []
        self.hsl_weights: list[torch.Tensor] = []
        self.hsl_harm_weights: list[torch.Tensor] = []
        self.audit_identity: dict[str, str] = {}

    def begin_route(self, route: str) -> None:
        """Clear route-local evidence after the shared scoring state is restored."""
        self.current_route = route
        self.raw_obs = None
        self.pending_action = None
        self.actions.clear()
        self.clean_positions.clear()
        self.repaired_positions.clear()
        self.noisy_positions.clear()
        self.clean_root_quat.clear()
        self.repaired_root_quat.clear()
        self.noisy_root_quat.clear()
        self.repaired_zmp.clear()
        self.noisy_zmp.clear()
        self.repaired_contact.clear()
        self.noisy_contact.clear()
        self.repaired_survival.zero_()
        self.noisy_survival.zero_()
        self.repaired_done.zero_()
        self.noisy_done.zero_()
        self.last_gain = None
        self.hsl_command = None
        self.hsl_pos_snapshot = None
        self.hsl_quat_snapshot = None
        self.hsl_targets.clear()
        self.hsl_weights.clear()
        self.hsl_harm_weights.clear()
        self.audit_identity.clear()

    def set_audit_identity(self, identity: Mapping[str, str]) -> None:
        required = {"audit_transaction_id", "audit_batch_signature", "audit_identity_state"}
        if set(identity) != required or any(not str(identity[key]).strip() for key in required):
            raise ValueError("quality route audit identity must provide complete transaction/batch/state fields")
        self.audit_identity = {key: str(identity[key]) for key in required}

    def observe(self) -> torch.Tensor:
        obs, extras = self.runner.env.get_observations()
        obs_dict = extras.get("observations", {}) if isinstance(extras, dict) else {}
        if self.runner.policy_obs_type is not None and self.runner.policy_obs_type in obs_dict:
            obs = obs_dict[self.runner.policy_obs_type]
        self.raw_obs = obs.to(self.runner.device)
        return self.raw_obs

    def apply_action(self, actions: torch.Tensor) -> None:
        if tuple(actions.shape) != (int(self.runner.env.num_envs), 6):
            raise ValueError("quality action owner requires [num_envs, 6] full-Delta-SE actions")
        self.pending_action = actions.detach().clone()
        self.runner._apply_frontres_task_corrections(
            self.pending_action,
            int(self.pair_layout.n_train),
            allow_oracle=False,
            n_candidate=0,
        )
        if self.current_route == "hsl":
            env = self.runner.env.unwrapped if hasattr(self.runner.env, "unwrapped") else self.runner.env
            manager = getattr(env, "command_manager", None)
            terms = getattr(manager, "_terms", {}) if manager is not None else {}
            for command in terms.values():
                if hasattr(command, "_frontres_pos_correction") and hasattr(command, "_frontres_quat_correction"):
                    n_train = int(self.pair_layout.n_train)
                    self.hsl_command = command
                    self.hsl_pos_snapshot = command._frontres_pos_correction[:n_train].detach().clone()
                    self.hsl_quat_snapshot = command._frontres_quat_correction[:n_train].detach().clone()
                    break
            if self.hsl_command is None:
                raise RuntimeError("quality HSL target audit could not find the task-space command owner")

    def step(self) -> Any:
        if self.pending_action is None:
            raise RuntimeError("quality rollout step requires action application first")
        corrected_obs, extras = self.runner.env.get_observations()
        obs_dict = extras.get("observations", {}) if isinstance(extras, dict) else {}
        if self.runner.policy_obs_type is not None and self.runner.policy_obs_type in obs_dict:
            corrected_obs = obs_dict[self.runner.policy_obs_type]
        normalized = self.runner._apply_obs_normalizer(corrected_obs.to(self.runner.device))
        executed = self.pending_action.clone()
        executed[int(self.pair_layout.n_train) :] = 0.0
        env_action = self.runner.alg.policy.get_env_action(normalized, executed)
        step_result = self.runner.env.step(env_action.to(self.runner.env.device))
        _, _, dones, _ = step_result
        dones = dones.to(self.runner.device).bool()
        if self.current_route == "hsl":
            target_result = build_frontres_hsl_rollout_target(
                self.runner,
                command=self.hsl_command,
                actions=self.pending_action,
                dones=dones,
                current_pos_correction=self.hsl_pos_snapshot,
                current_quat_correction=self.hsl_quat_snapshot,
                n_train=int(self.pair_layout.n_train),
                n_candidate=int(self.pair_layout.n_candidate),
                n_base=int(self.pair_layout.n_base),
                n_clean=int(self.pair_layout.n_clean),
                quat_to_rotvec_wxyz=quat_to_rotvec_wxyz,
                write_transition=False,
                enforce_training_enable_flag=False,
            )
            if target_result is None:
                raise RuntimeError("quality HSL target audit did not receive a canonical target")
            n_train = int(self.pair_layout.n_train)
            self.hsl_targets.append(target_result.target[:n_train].detach().clone())
            self.hsl_weights.append(target_result.weight[:n_train].detach().clone())
            self.hsl_harm_weights.append(target_result.harm_weight[:n_train].detach().clone())
        n_pair = int(self.pair_layout.n_train)
        base_start = int(self.pair_layout.n_train) + int(self.pair_layout.n_candidate)
        self.repaired_survival += (~self.repaired_done).float()
        self.noisy_survival += (~self.noisy_done).float()
        self.repaired_done |= dones[:n_pair]
        self.noisy_done |= dones[base_start : base_start + n_pair]
        self.actions.append(executed[:n_pair].detach().clone())

        clean, repaired, noisy = _capture_motion_quality_frame(self.runner, self.pair_layout)
        clean_q, repaired_q, noisy_q = _capture_root_orientation_frame(self.runner, self.pair_layout)
        physics = _capture_physics_frame(self.runner, self.pair_layout)
        if clean is None or repaired is None or noisy is None:
            raise RuntimeError("quality execution owner could not capture matched motion rows")
        self.clean_positions.append(clean)
        self.repaired_positions.append(repaired)
        self.noisy_positions.append(noisy)
        if clean_q is not None and repaired_q is not None and noisy_q is not None:
            self.clean_root_quat.append(clean_q)
            self.repaired_root_quat.append(repaired_q)
            self.noisy_root_quat.append(noisy_q)
        if physics is not None:
            repaired_zmp, noisy_zmp, repaired_contact, noisy_contact = physics
            self.repaired_zmp.append(repaired_zmp)
            self.noisy_zmp.append(noisy_zmp)
            self.repaired_contact.append(repaired_contact)
            self.noisy_contact.append(noisy_contact)
        return step_result

    def compute_gain(self) -> Any:
        # Style/orientation owners consume batch-major trajectories [B, T, ...],
        # while repair cost consumes action history [T, B, 6]. Route capture
        # appends one [B, ...] frame per step, so preserve both canonical layouts
        # explicitly instead of letting K masquerade as the paired-sample batch.
        stack_trajectory = lambda values: torch.stack(values, dim=1) if values else None
        stack_action_steps = lambda values: torch.stack(values, dim=0) if values else None
        mean_frames = lambda values: torch.stack(values, dim=0).float().mean(dim=0) if values else None
        n_pair = int(self.pair_layout.n_train)
        effective_k = torch.full((n_pair,), self.horizon_k, dtype=torch.float32, device=self.runner.device)
        self.last_gain = compute_segment_gain(
            clean_positions=stack_trajectory(self.clean_positions),
            repaired_positions=stack_trajectory(self.repaired_positions),
            noisy_positions=stack_trajectory(self.noisy_positions),
            repaired_success=~self.repaired_done,
            noisy_success=~self.noisy_done,
            repaired_survival=self.repaired_survival,
            noisy_survival=self.noisy_survival,
            action_steps=stack_action_steps(self.actions),
            config=FrontRESSegmentGainConfig.from_mapping(getattr(self.runner, "cfg", None)),
            effective_horizon_k=effective_k,
            repaired_zmp_margin=mean_frames(self.repaired_zmp),
            noisy_zmp_margin=mean_frames(self.noisy_zmp),
            repaired_contact=mean_frames(self.repaired_contact),
            noisy_contact=mean_frames(self.noisy_contact),
            clean_root_quaternions=stack_trajectory(self.clean_root_quat),
            repaired_root_quaternions=stack_trajectory(self.repaired_root_quat),
            noisy_root_quaternions=stack_trajectory(self.noisy_root_quat),
            audit_transaction_id=self.audit_identity.get("audit_transaction_id"),
            audit_batch_signature=self.audit_identity.get("audit_batch_signature"),
            audit_identity_state=self.audit_identity.get("audit_identity_state", "UNCONFIRMED"),
        )
        return self.last_gain

    def capture_execution(self) -> Mapping[str, Any]:
        result = {
            "repaired_success": (~self.repaired_done).detach().clone(),
            "noisy_success": (~self.noisy_done).detach().clone(),
            "repaired_survival_steps": self.repaired_survival.detach().clone(),
            "noisy_survival_steps": self.noisy_survival.detach().clone(),
        }
        if self.current_route == "hsl":
            if len(self.hsl_targets) != self.horizon_k or len(self.actions) != self.horizon_k:
                raise RuntimeError("quality HSL target audit must cover every rollout step")
            targets = torch.stack(self.hsl_targets, dim=0)
            actions = torch.stack(self.actions, dim=0)
            weights = torch.stack(self.hsl_weights, dim=0)
            harm_weights = torch.stack(self.hsl_harm_weights, dim=0)
            target_norm = targets.norm(dim=-1)
            action_norm = actions.norm(dim=-1)
            cosine = (actions * targets).sum(dim=-1) / (action_norm * target_norm).clamp_min(1.0e-8)
            target_nonzero = target_norm.gt(1.0e-8)
            cosine = torch.where(target_nonzero, cosine, torch.zeros_like(cosine))
            result["hsl_supervision"] = {
                "targets": targets,
                "sample_weights": weights,
                "harm_weights": harm_weights,
                "target_nonzero": target_nonzero,
                "action_target_l2": (actions - targets).norm(dim=-1),
                "action_target_cosine": cosine,
                "sign_agree_per_dim": ((actions * targets) > 0.0).float(),
            }
        return result


def _training_state_signature(runner: Any) -> str:
    payload: dict[str, Any] = {}
    optimizer = getattr(getattr(runner, "alg", None), "optimizer", None)
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    sampler = getattr(runner, "_frontres_segment_sampler", None)
    if sampler is not None and hasattr(sampler, "state_dict"):
        payload["sampler"] = sampler.state_dict()
    payload["warmup"] = {
        name: getattr(runner, name, None)
        for name in (
            "current_learning_iteration",
            "_frontres_warmup_complete",
            "_frontres_segment_actor_warmup_complete",
        )
    }
    stream = io.BytesIO()
    torch.save(payload, stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().tolist())
    if hasattr(value, "__dataclass_fields__"):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _role_identity_snapshot(snapshot: Any) -> Mapping[str, Any]:
    """Summarize paired role deltas from the immutable scoring-start snapshot."""
    roles = tuple(snapshot.role_layout)
    role_rows = {role: [index for index, value in enumerate(roles) if value == role] for role in set(roles)}
    policy_rows = role_rows.get("policy", [])
    noisy_rows = role_rows.get("noisy", [])
    clean_rows = role_rows.get("clean", [])
    if not policy_rows or len(policy_rows) != len(noisy_rows) or len(policy_rows) != len(clean_rows):
        raise ValueError(f"quality role identity requires equal policy/noisy/clean rows, got {role_rows}")

    root = snapshot.root_state_w.restore(device="cpu").float()
    origins = snapshot.env_origins.restore(device="cpu").float()
    joint_pos = snapshot.joint_pos.restore(device="cpu").float()
    joint_vel = snapshot.joint_vel.restore(device="cpu").float()
    command = {name: image.restore(device="cpu").float() for name, image in snapshot.command_state}
    policy = torch.tensor(policy_rows, dtype=torch.long)
    noisy = torch.tensor(noisy_rows, dtype=torch.long)
    clean = torch.tensor(clean_rows, dtype=torch.long)

    def max_abs_delta(value: torch.Tensor, lhs: torch.Tensor, rhs: torch.Tensor) -> float:
        return float((value.index_select(0, lhs) - value.index_select(0, rhs)).abs().max().item())

    local_root = root[:, :3] - origins
    return {
        "role_rows": {role: rows for role, rows in sorted(role_rows.items())},
        "policy_noisy": {
            "world_root_pos_max_abs": max_abs_delta(root[:, :3], policy, noisy),
            "env_origin_max_abs": max_abs_delta(origins, policy, noisy),
            "local_root_pos_max_abs": max_abs_delta(local_root, policy, noisy),
            "root_quat_max_abs": max_abs_delta(root[:, 3:7], policy, noisy),
            "root_lin_vel_max_abs": max_abs_delta(root[:, 7:10], policy, noisy),
            "root_ang_vel_max_abs": max_abs_delta(root[:, 10:13], policy, noisy),
            "joint_pos_max_abs": max_abs_delta(joint_pos, policy, noisy),
            "joint_vel_max_abs": max_abs_delta(joint_vel, policy, noisy),
            "cached_perturbed_pos_max_abs": max_abs_delta(command["_cached_perturbed_pos"], policy, noisy),
            "cached_perturbed_quat_max_abs": max_abs_delta(command["_cached_perturbed_quat"], policy, noisy),
        },
        "corruption_present": {
            "policy_clean_cached_pos_max_abs": max_abs_delta(command["_cached_perturbed_pos"], policy, clean),
            "policy_clean_cached_quat_max_abs": max_abs_delta(command["_cached_perturbed_quat"], policy, clean),
        },
    }


def _serialize_result(item: Any, results: tuple[FrontRESPolicyQualityRouteResult, ...]) -> Mapping[str, Any]:
    return {
        "item": item.to_dict(),
        "comparison_signature": item.comparison_signature,
        "routes": {
            result.identity.route: {
                "checkpoint_identity": result.identity.checkpoint_identity,
                "initial_state_hash": result.identity.state.initial_state_hash,
                "actions": _json_value(result.actions),
                "gain": _json_value(result.gain),
                "execution": _json_value(result.execution),
            }
            for result in results
        },
    }


def build_frontres_policy_quality_formal_owner_bundle(
    runner: Any,
    request: FrontRESPolicyQualityEvalRequest,
) -> FrontRESPolicyQualityFormalOwnerBundle:
    """Build the production bundle from canonical lower-level Stage 3 owners."""
    ensure_frontres_policy_quality_reset_support(runner)
    pair_layout = configure_frontres_pair_layout(runner, is_frontres=True)
    if int(runner.env.num_envs) != sum(
        int(getattr(pair_layout, name)) for name in ("n_train", "n_candidate", "n_base", "n_clean")
    ):
        raise ValueError("policy-quality pair layout must cover every environment row")

    policy = runner.alg.policy
    actor_template = policy.residual_actor
    obs_dim = int(getattr(policy, "num_actor_obs"))
    gmt_dim = int(getattr(runner, "_frontres_gmt_obs_dim"))
    observation_identity = FrontRESPolicyQualityObservationIdentity(
        expected_obs_dim=obs_dim,
        actor_input_dim=_first_linear_input_dim(actor_template),
        normalizer_identity="checkpoint:frontres-prefix+frozen-gmt-suffix",
    )
    adapters = (
        ZeroFrontRESTaskActor(observation_identity),
        FrozenFrontRESTaskActor.from_checkpoint_payload(
            route="hsl",
            checkpoint_identity=request.hsl_checkpoint_path,
            checkpoint_payload=_checkpoint_payload(request.hsl_checkpoint_path, runner.device),
            actor_template=actor_template,
            normalizer_template=runner.obs_normalizer,
            observation_identity=observation_identity,
            max_delta_pos=float(policy.max_delta_pos),
            max_delta_rpy=float(policy.max_delta_rpy),
            gmt_obs_dim=gmt_dim,
        ),
        FrozenFrontRESTaskActor.from_checkpoint_payload(
            route="policy",
            checkpoint_identity=request.policy_checkpoint_path,
            checkpoint_payload=_checkpoint_payload(request.policy_checkpoint_path, runner.device),
            actor_template=actor_template,
            normalizer_template=runner.obs_normalizer,
            observation_identity=observation_identity,
            max_delta_pos=float(policy.max_delta_pos),
            max_delta_rpy=float(policy.max_delta_rpy),
            gmt_obs_dim=gmt_dim,
        ),
    )
    role_identity_by_signature: dict[str, Mapping[str, Any]] = {}

    def prepare_item(active_runner: Any, item: Any, active_request: FrontRESPolicyQualityEvalRequest):
        if active_request is not request:
            raise RuntimeError("policy-quality formal bundle cannot be reused with another request")
        random.seed(int(item.seed))
        np.random.seed(int(item.seed))
        torch.manual_seed(int(item.seed))
        _apply_manifest_reset(active_runner, item, pair_layout)
        env_ids = torch.arange(int(active_runner.env.num_envs), device=active_runner.device)
        snapshot = capture_frontres_policy_quality_state(
            active_runner,
            env_ids=env_ids,
            comparison_signature=item.comparison_signature,
            role_layout=_role_layout(pair_layout),
        )
        # B4: QUALITY-ID-01 在 reset 后、任一 route 前只读取 frozen scoring state.
        # Result: PENDING_Q_EVIDENCE; 区分 world origin 与 local dynamic/cache mismatch.
        role_identity = _role_identity_snapshot(snapshot)
        role_identity_by_signature[item.comparison_signature] = role_identity
        print(
            "[QUALITY-ID-01 Role Identity] "
            + json.dumps(_json_value(role_identity), sort_keys=True, separators=(",", ":")),
            flush=True,
        )
        capture = _RouteCapture(active_runner, pair_layout, item.effective_horizon_k)
        hooks = FrontRESPolicyQualityRouteHooks(
            observe=capture.observe,
            apply_action=capture.apply_action,
            step=capture.step,
            compute_gain=capture.compute_gain,
            capture_execution=capture.capture_execution,
            begin_route=capture.begin_route,
            set_audit_identity=capture.set_audit_identity,
        )
        return snapshot, adapters, hooks

    def serialize_result(item: Any, results: tuple[FrontRESPolicyQualityRouteResult, ...]) -> Mapping[str, Any]:
        payload = dict(_serialize_result(item, results))
        payload["role_identity"] = role_identity_by_signature[item.comparison_signature]
        return payload

    return FrontRESPolicyQualityFormalOwnerBundle(
        owner_identity=_OWNER_IDENTITY,
        prepare_item=prepare_item,
        isolation_state=_training_state_signature,
        serialize_result=serialize_result,
    )
