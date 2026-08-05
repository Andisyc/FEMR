"""FrontRES deployable observation and one-action-K evidence executor."""

from __future__ import annotations





from typing import Any


import torch


from rsl_rl.algorithms.frontres_segment_ppo import FrontRESSegmentPPOBatch


from rsl_rl.frontres.frontres_segment_evidence import (
    FrontRESExecutedKTrajectory,
    FrontRESRepairAttemptEvidence,
)
from rsl_rl.frontres.frontres_segment_grouped_adapter import build_frontres_v015_grouped_candidate_storage


from rsl_rl.runners.frontres_rollout_step import append_frontres_future_intent_actor_context, frontres_motion_command, prepare_frontres_v015_frozen_gmt_step, prepare_frontres_v015_one_action_at_t
from rsl_rl.runners.frontres_formal_runtime_audit import print_one_action_k_audit





from rsl_rl.runners.frontres_segment_runtime_types import (


    FrontRESSegmentLiveObservations,


    FrontRESV015GainConsumerEvidence,
    frontres_collection_batch,
    frontres_observation_trace,
    update_frontres_observation_trace,


)


from rsl_rl.runners.frontres_segment_live_storage import (
    frontres_gain_module,
)


from rsl_rl.runners.frontres_segment_physics import (
    capture_frontres_physics_frame,
    capture_frontres_quality_lateral_lean_frame,
    capture_frontres_v017_execution_frame,
)





def _resolve_probe_modes(runner: Any) -> tuple[bool, bool]:
    single_update = bool(
        runner._frontres_segment_replay_boundary.live_single_update_only
        or runner._frontres_segment_replay_boundary.live_update_loop_only
        or runner._frontres_segment_replay_boundary.live_train_enabled
    )
    storage_write = bool(runner._frontres_segment_replay_boundary.live_storage_write_only or single_update)
    if not (runner._frontres_segment_replay_boundary.live_probe_only or storage_write):
        raise ValueError(
            "FrontRES Segment live probe requires frontres_segment_live_probe_only=True "
            "or frontres_segment_live_storage_write_only=True "
            "or frontres_segment_live_single_update_only=True "
            "or frontres_segment_live_update_loop_only=True."
        )
    return single_update, storage_write


def _append_fixed_noisy_actor_context(runner: Any, obs: torch.Tensor) -> torch.Tensor:
    append = getattr(runner, "_append_frontres_fixed_noisy_future_context", None)
    if callable(append):
        return append(obs)
    batch = frontres_collection_batch(runner)
    if isinstance(getattr(batch, "frontres_fixed_noisy_tape", None), torch.Tensor):
        raise RuntimeError("fixed Noisy Segment Replay requires runner actor-context connectivity")
    return obs


def _uses_v015_future_intent_route_local(runner: Any) -> bool:
    """本地判定避免 legacy probe stub 依赖新的 rollout-step symbol."""

    alg = getattr(runner, "alg", None)
    offsets = tuple(int(value) for value in (getattr(alg, "frontres_future_offsets", ()) or ()))
    return bool(offsets) or getattr(runner, "_frontres_future_intent_layout", None) is not None


def _read_live_observations(runner: Any) -> FrontRESSegmentLiveObservations:
    """Read env-owned observations and apply the active actor-context/normalizer route.

    Status: R5 offline S2 contract-confirmed; simulator/live timing remains unconfirmed.
    """

    obs, extras = runner.env.get_observations()
    obs_dict = extras.get("observations", {})
    if runner.policy_obs_type is not None and runner.policy_obs_type in obs_dict:
        obs = obs_dict[runner.policy_obs_type]
    privileged_obs = obs_dict.get(runner.privileged_obs_type, obs)
    teacher_obs = obs_dict.get(runner.teacher_obs_type)
    if teacher_obs is None:
        teacher_obs = privileged_obs
    ref_vel_estimator_obs = obs_dict.get(runner.ref_vel_estimator_obs_type)

    obs = obs.to(runner.device)
    raw_obs_dim = int(obs.shape[-1])
    uses_v015_future_intent = _uses_v015_future_intent_route_local(runner)
    # v015 的 actor 只能看到 deployment-q29 intent. 旧 65D fixed tape 只保留给
    # 历史路径, 两者不能在同一个 observation 中拼接.
    if uses_v015_future_intent:
        obs = append_frontres_future_intent_actor_context(runner, obs)
    else:
        obs = _append_fixed_noisy_actor_context(runner, obs)
    combined_obs_dim = int(obs.shape[-1])
    obs = runner._apply_obs_normalizer(obs)
    if uses_v015_future_intent:
        policy = getattr(getattr(runner, "alg", None), "policy", None)
        update_frontres_observation_trace(
            runner,
            role_row_count=int(obs.shape[0]),
            current_command_dim=0,
            raw_observation_dim=raw_obs_dim,
            q29_tail_dim=combined_obs_dim - raw_obs_dim,
            combined_observation_dim=combined_obs_dim,
            normalized_observation_dim=int(obs.shape[-1]),
            femr_visible_dim=int(getattr(policy, "num_frontres_obs", 0) or 0),
            gmt_suffix_dim=int(getattr(runner, "_frontres_gmt_obs_dim", 0) or 0),
            gmt_input_dim=0,
            post_advance_gmt_read_count=0,
        )
    privileged_obs = runner.privileged_obs_normalizer(privileged_obs.to(runner.device))
    teacher_obs = runner.teacher_obs_normalizer(teacher_obs.to(runner.device))
    if ref_vel_estimator_obs is not None:
        ref_vel_estimator_obs = ref_vel_estimator_obs.to(runner.device)
    return FrontRESSegmentLiveObservations(
        obs=obs,
        privileged_obs=privileged_obs,
        teacher_obs=teacher_obs,
        ref_vel_estimator_obs=ref_vel_estimator_obs,
    )


def read_frontres_live_observations(runner: Any) -> FrontRESSegmentLiveObservations:
    """Public observation gateway preserving the 928/158/770 authority route."""

    return _read_live_observations(runner)


def _read_v015_frozen_gmt_observations(
    runner: Any,
    obs: torch.Tensor,
    infos: dict[str, Any],
    *,
    frozen_frontres_prefix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Prepare a GMT execution observation without reopening actor-only q29 context."""

    obs_dict = infos.get("observations", {}) if isinstance(infos, dict) else {}
    if runner.policy_obs_type is not None and runner.policy_obs_type in obs_dict:
        obs = obs_dict[runner.policy_obs_type].to(runner.device)
    else:
        obs = obs.to(runner.device)
    # R5 exact route: t 已经验证并归一化 158D FEMR prefix. K 内 actor 冻结, 因此
    # 只允许重新读取/归一化 fresh 770D GMT suffix, 不得重新打开 q29 actor snapshot.
    policy = getattr(getattr(runner, "alg", None), "policy", None)
    frontres_dim = int(getattr(policy, "num_frontres_obs", 0) or 0)
    gmt_dim = int(getattr(runner, "_frontres_gmt_obs_dim", 0) or 0)
    if isinstance(frozen_frontres_prefix, torch.Tensor) and frontres_dim > 0 and gmt_dim > 0:
        prefix = frozen_frontres_prefix.to(device=runner.device)
        if tuple(prefix.shape) != (int(obs.shape[0]), frontres_dim):
            raise RuntimeError(
                "v015 frozen-GMT route requires the t-time normalized FEMR prefix "
                f"[{int(obs.shape[0])},{frontres_dim}], got {tuple(prefix.shape)}"
            )
        if int(obs.shape[-1]) < gmt_dim:
            raise RuntimeError("v015 frozen-GMT raw observation is smaller than the frozen GMT suffix")
        gmt_raw = obs[..., -gmt_dim:]
        normalize_gmt = getattr(runner, "obs_normalizer", None)
        if not callable(normalize_gmt):
            raise RuntimeError("v015 frozen-GMT route requires the frozen GMT normalizer")
        gmt_obs = normalize_gmt(gmt_raw)
        combined = torch.cat([prefix.to(dtype=gmt_obs.dtype), gmt_obs], dim=-1)
        if int(combined.shape[-1]) != int(getattr(policy, "num_actor_obs", 0) or 0):
            raise RuntimeError("v015 frozen-GMT route lost the exact FEMR/GMT observation authority")
        trace = frontres_observation_trace(runner)
        update_frontres_observation_trace(
            runner,
            gmt_input_dim=int(gmt_obs.shape[-1]),
            post_advance_gmt_read_count=int(trace.get("post_advance_gmt_read_count", 0)) + 1,
        )
        return combined

    if bool(getattr(getattr(runner, "alg", None), "frontres_formal_transaction_enabled", False)):
        raise RuntimeError("v015 formal frozen-GMT route requires the exact R3 158D/770D authority")
    # Candidate-only legacy fixtures without R3 dimensions retain their local test contract.
    obs = append_frontres_future_intent_actor_context(runner, obs)
    return runner._apply_obs_normalizer(obs)


def _require_v015_one_action_k_layout(
    runner: Any,
    observations: FrontRESSegmentLiveObservations,
    pair_layout: Any,
) -> tuple[Any, dict[str, object], torch.Tensor]:
    """Fail closed unless the candidate collector sees exactly the sealed two-role layout."""

    n_repair = int(getattr(pair_layout, "n_train", 0))
    n_noisy = int(getattr(pair_layout, "n_base", 0))
    total = int(observations.obs.shape[0])
    if (
        n_repair <= 0
        or n_noisy != n_repair
        or int(getattr(pair_layout, "n_candidate", 0)) != 0
        or int(getattr(pair_layout, "n_clean", 0)) != 0
        or total != n_repair + n_noisy
    ):
        raise RuntimeError("v015 one-action K collector requires only equal Repair/Noisy role rows")
    command = frontres_motion_command(runner)
    snapshot = command.frontres_local_scenario_snapshot(
        torch.arange(total, device=observations.obs.device, dtype=torch.long)
    )
    roles = tuple(snapshot["roles"])
    expected_roles = ("repair",) * n_repair + ("noisy",) * n_noisy
    if roles != expected_roles:
        raise RuntimeError(
            "v015 one-action K collector requires Repair rows followed by Noisy rows; "
            f"got roles={roles}"
        )
    repair_rows = torch.arange(n_repair, device=observations.obs.device, dtype=torch.long)
    return command, snapshot, repair_rows


def _capture_v015_post_t_executed_q29(command: Any, *, role_count: int, device: torch.device) -> torch.Tensor:
    """读取 post-action robot articulation state, 绝不读取 command/reference q29."""

    executed_q29 = getattr(command, "robot_joint_pos", None)
    if not isinstance(executed_q29, torch.Tensor):
        raise RuntimeError("v015 Gain capture requires command.robot_joint_pos after the t action")
    executed_q29 = executed_q29.detach().to(device=device, dtype=torch.float32).clone()
    if tuple(executed_q29.shape) != (role_count, 29):
        raise RuntimeError(
            "v015 Gain capture requires post-t robot joint state [N,29], "
            f"got {tuple(executed_q29.shape)}"
        )
    return executed_q29


def _v015_intent_provenance_rows(snapshot: dict[str, object], *, role_count: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """只从 sealed command snapshot 提取 deployment-q29 provenance."""

    rows = snapshot.get("provenance")
    if not isinstance(rows, tuple) or len(rows) != role_count:
        raise RuntimeError("v015 Gain capture requires one sealed q29 provenance row per scored role")
    provenance: list[str] = []
    source_rows: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("v015 Gain capture requires mapping-like q29 provenance")
        value = str(row.get("intent_q29_provenance", ""))
        source = str(row.get("intent_q29_source", ""))
        lowered = source.lower()
        if value != "deployment_noisy_q29" or not source or any(
            token in lowered for token in ("clean", "root", "global")
        ):
            raise RuntimeError("v015 Gain capture rejects Clean/root/global q29 provenance")
        provenance.append(value)
        source_rows.append(source)
    return tuple(provenance), tuple(source_rows)


def _read_v017_normalized_gmt_suffix(runner: Any) -> torch.Tensor:
    """Read only the frozen GMT-owned 770D observation authority."""

    raw, infos = runner.env.get_observations()
    obs_dict = infos.get("observations", {}) if isinstance(infos, dict) else {}
    if runner.policy_obs_type is not None and runner.policy_obs_type in obs_dict:
        raw = obs_dict[runner.policy_obs_type]
    raw = raw.to(runner.device)
    gmt_dim = int(getattr(runner, "_frontres_gmt_obs_dim", 0) or 0)
    if gmt_dim <= 0 or int(raw.shape[-1]) < gmt_dim:
        raise RuntimeError("v017 baseline requires the authoritative GMT suffix width")
    normalize = getattr(runner, "obs_normalizer", None)
    if not callable(normalize):
        raise RuntimeError("v017 baseline requires the frozen GMT normalizer")
    suffix = normalize(raw[..., -gmt_dim:])
    if tuple(suffix.shape) != (int(raw.shape[0]), gmt_dim) or not bool(torch.isfinite(suffix).all()):
        raise RuntimeError("v017 baseline produced an invalid normalized GMT suffix")
    return suffix.detach().clone()


def _stack_v017_execution_frames(
    frames: list[Any],
    *,
    survival: list[torch.Tensor],
    valid: list[torch.Tensor],
) -> tuple[FrontRESExecutedKTrajectory, torch.Tensor]:
    if not frames or len(frames) != len(survival) or len(frames) != len(valid):
        raise RuntimeError("v017 K execution requires aligned non-empty frame evidence")
    trajectory = FrontRESExecutedKTrajectory(
        joint_pos=torch.stack([value.joint_pos for value in frames], dim=0),
        root_pos=torch.stack([value.root_pos for value in frames], dim=0),
        root_quat=torch.stack([value.root_quat for value in frames], dim=0),
        key_body_pos=torch.stack([value.key_body_pos for value in frames], dim=0),
        root_lin_vel=torch.stack([value.root_lin_vel for value in frames], dim=0),
        root_ang_vel=torch.stack([value.root_ang_vel for value in frames], dim=0),
        foot_pos=torch.stack([value.foot_pos for value in frames], dim=0),
        contact=torch.stack([value.contact for value in frames], dim=0),
        zmp_margin=torch.stack([value.zmp_margin for value in frames], dim=0),
        survival=torch.stack(survival, dim=0),
        valid_mask=torch.stack(valid, dim=0),
    )
    expected = torch.stack([value.expected_support for value in frames], dim=0)
    trajectory.validate()
    return trajectory, expected.detach().clone()


def _stack_v017_selected_role_execution_frames(
    frames: list[Any],
    *,
    survival: list[torch.Tensor],
    valid: list[torch.Tensor],
    role_rows: torch.Tensor,
) -> tuple[FrontRESExecutedKTrajectory, torch.Tensor]:
    """Stack frames already captured for role_rows with matching selected masks."""

    # B1: 校验 global role-row vectors, 产出与 selected frame batch 对齐的 [K,M] masks.
    ids = role_rows.to(dtype=torch.long).reshape(-1)
    if int(ids.numel()) == 0 or int(torch.unique(ids).numel()) != int(ids.numel()):
        raise ValueError("v017 selected execution requires nonempty unique role rows")
    max_role_row = int(ids.max().item())
    if any(
        not isinstance(value, torch.Tensor)
        or value.ndim != 1
        or int(value.numel()) <= max_role_row
        for value in survival + valid
    ):
        raise ValueError("v017 selected execution masks must contain every requested global role row")
    selected_survival = [value.index_select(0, ids.to(value.device)).detach().clone() for value in survival]
    selected_valid = [value.index_select(0, ids.to(value.device)).detach().clone() for value in valid]

    # B2: 只封装一次 selected role batch, 产出统一 [K,M,...] trajectory.
    trajectory, expected = _stack_v017_execution_frames(
        frames,
        survival=selected_survival,
        valid=selected_valid,
    )
    if int(trajectory.joint_pos.shape[1]) != int(ids.numel()):
        raise ValueError("v017 selected execution frames must already align with role_rows")
    return trajectory, expected


def select_frontres_v017_trajectory_rows(
    trajectory: FrontRESExecutedKTrajectory,
    rows: torch.Tensor,
) -> FrontRESExecutedKTrajectory:
    ids = rows.to(device=trajectory.joint_pos.device, dtype=torch.long).reshape(-1)
    selected = FrontRESExecutedKTrajectory(
        **{
            name: getattr(trajectory, name).index_select(1, ids).detach().clone()
            for name in (
                "joint_pos",
                "root_pos",
                "root_quat",
                "key_body_pos",
                "root_lin_vel",
                "root_ang_vel",
                "foot_pos",
                "contact",
                "zmp_margin",
                "survival",
                "valid_mask",
            )
        }
    )
    selected.validate()
    return selected


def collect_frontres_v017_no_actor_baseline(
    runner: Any,
    *,
    horizon_k: int,
    authoritative_rows: torch.Tensor,
) -> tuple[FrontRESExecutedKTrajectory, torch.Tensor]:
    """Execute one logical Clean/Noisy baseline for the two selected Segments."""

    # B1: 从 command 和 policy public port 取得 sealed phase 与 frozen GMT suffix action.
    command = frontres_motion_command(runner)
    policy = getattr(getattr(runner, "alg", None), "policy", None)
    direct = getattr(policy, "run_frozen_gmt_from_suffix", None)
    if not callable(direct) or int(horizon_k) <= 0:
        raise RuntimeError("v017 baseline requires the frozen GMT suffix port and positive K")
    ids = authoritative_rows.to(device=runner.device, dtype=torch.long).reshape(-1)
    if int(ids.numel()) != 2 or int(torch.unique(ids).numel()) != 2:
        raise ValueError("v017 baseline requires exactly two unique authoritative Segment rows")
    execution_started = False
    try:
        t_actions = direct(_read_v017_normalized_gmt_suffix(runner))
        _raw, _reward, t_done, _infos = runner.env.step(t_actions.to(runner.env.device))
        done_any = t_done.to(runner.device).detach().bool().reshape(-1)
        begin = getattr(command, "begin_frontres_local_scenario_k_execution", None)
        if not callable(begin):
            raise RuntimeError("v017 baseline requires command-owned K lifecycle")
        begin()
        execution_started = True
        # B2: 只推进 GMT-owned K execution, 累积同一 baseline 的动态与 Physics evidence.
        frames: list[Any] = []
        survival: list[torch.Tensor] = []
        valid_rows: list[torch.Tensor] = []
        for _ in range(int(horizon_k)):
            advance = command.advance_frontres_local_scenario_k_execution()
            valid = advance["valid_mask"].to(device=runner.device, dtype=torch.bool).reshape(-1)
            env_actions = direct(_read_v017_normalized_gmt_suffix(runner))
            _raw, _reward, dones, _infos = runner.env.step(env_actions.to(runner.env.device))
            dones = dones.to(runner.device).detach().bool().reshape(-1)
            alive = valid & ~done_any
            frames.append(
                capture_frontres_v017_execution_frame(
                    runner,
                    selected_rows=ids,
                )
            )
            survival.append(alive.index_select(0, ids).detach().clone())
            valid_rows.append(valid.index_select(0, ids).detach().clone())
            done_any = done_any | (dones & alive)
        # B3: 返回 immutable K trajectory 与 expected-support carrier, 不产生 policy row.
        return _stack_v017_execution_frames(frames, survival=survival, valid=valid_rows)
    finally:
        if execution_started:
            end = getattr(command, "end_frontres_local_scenario_k_execution", None)
            if not callable(end):
                raise RuntimeError("v017 baseline requires command-owned K close")
            end()


def _require_direct_policy_stats(transition: Any) -> tuple[torch.Tensor, torch.Tensor]:
    mean = getattr(transition, "action_mean", None)
    sigma = getattr(transition, "action_sigma", None)
    if not isinstance(mean, torch.Tensor) or not isinstance(sigma, torch.Tensor):
        raise RuntimeError("one-action-K evidence requires policy mean and sigma tensors")
    if mean.ndim != 2 or sigma.ndim != 2 or tuple(mean.shape) != tuple(sigma.shape) or int(mean.shape[-1]) != 6:
        raise ValueError(
            "one-action-K policy statistics must be matching direct [B,6] tensors, "
            f"got mean={tuple(mean.shape)} sigma={tuple(sigma.shape)}"
        )
    return mean, sigma


def _record_v017_policy_authority_trace(
    runner: Any,
    *,
    command: Any,
    policy_privileged_observations: torch.Tensor,
    role_row_count: int,
    policy_row_count: int,
) -> None:
    """Record measured command and Critic dimensions for one formal Repair collection."""

    # B1: 从正式 command 和 old-policy tuple 读取真实维度, 拒绝缺失或行错位的 trace.
    current_command = getattr(command, "command", None)
    if (
        not isinstance(current_command, torch.Tensor)
        or current_command.ndim != 2
        or tuple(current_command.shape) != (int(role_row_count), 58)
    ):
        raise RuntimeError(
            "v017 Repair collection requires the role-aligned current GMT command [B,58]"
        )
    if (
        not isinstance(policy_privileged_observations, torch.Tensor)
        or policy_privileged_observations.ndim != 2
        or int(policy_privileged_observations.shape[0]) != int(policy_row_count)
        or int(policy_privileged_observations.shape[1]) <= 0
    ):
        raise RuntimeError(
            "v017 Repair collection requires one non-empty Critic observation per policy row"
        )
    update_frontres_observation_trace(
        runner,
        current_command_dim=int(current_command.shape[-1]),
        critic_observation_dim=int(policy_privileged_observations.shape[-1]),
    )


def collect_frontres_v017_repair_attempts(
    runner: Any,
    observations: FrontRESSegmentLiveObservations,
    *,
    pair_layout: Any,
    transaction_id: str,
    policy_snapshot_id: str,
    source_index: torch.Tensor,
    segment_ids: torch.Tensor,
    trial_index: torch.Tensor,
) -> tuple[FrontRESRepairAttemptEvidence, ...]:
    """Collect exact-M Repair policy rows; companion rows remain unobserved scaffold."""

    # B1: 校验 two-role layout 和 sealed metadata, 冻结本次 M rows 的 old-policy tuple.
    command, snapshot, repair_rows = _require_v015_one_action_k_layout(runner, observations, pair_layout)
    n_repair = int(repair_rows.numel())
    metadata = (source_index, segment_ids, trial_index)
    if any(not isinstance(value, torch.Tensor) or tuple(value.shape) != (n_repair,) for value in metadata):
        raise ValueError("v017 Repair collection requires one source/segment/trial identity per policy row")
    frontres_dim = int(getattr(getattr(runner.alg, "policy", None), "num_frontres_obs", 0) or 0)
    frozen_prefix = observations.obs[:, :frontres_dim].detach().clone()
    execution_started = False
    try:
        plan = prepare_frontres_v015_one_action_at_t(
            runner,
            obs=observations.obs,
            privileged_obs=observations.privileged_obs,
            teacher_obs=observations.teacher_obs,
            ref_vel_estimator_obs=observations.ref_vel_estimator_obs,
            iteration=int(getattr(runner, "current_learning_iteration", 0)),
            n_repair=n_repair,
            n_noisy=int(getattr(pair_layout, "n_base", 0)),
        )
        transition = getattr(runner.alg, "transition", None)
        names = ("observations", "privileged_observations", "actions_log_prob", "values", "action_mean", "action_sigma")
        if transition is None or any(not isinstance(getattr(transition, name, None), torch.Tensor) for name in names):
            raise RuntimeError("v017 Repair collection requires the frozen old-policy tuple")
        policy_rows = {
            "observation": transition.observations.index_select(0, repair_rows).detach().clone(),
            "privileged": transition.privileged_observations.index_select(0, repair_rows).detach().clone(),
            "action": plan.actions.index_select(0, repair_rows).detach().clone(),
            "log_prob": transition.actions_log_prob.index_select(0, repair_rows).detach().clone().reshape(-1),
            "value": transition.values.index_select(0, repair_rows).detach().clone().reshape(-1),
            "mean": _require_direct_policy_stats(transition)[0].index_select(0, repair_rows).detach().clone(),
            "sigma": _require_direct_policy_stats(transition)[1].index_select(0, repair_rows).detach().clone(),
        }
        _record_v017_policy_authority_trace(
            runner,
            command=command,
            policy_privileged_observations=policy_rows["privileged"],
            role_row_count=int(observations.obs.shape[0]),
            policy_row_count=n_repair,
        )
        # B2: 每条 Repair 只在 t 调用一次 FEMR, 随后仅执行 frozen GMT 的 K-step evidence horizon.
        _raw, _reward, t_done, _infos = runner.env.step(plan.env_actions.to(runner.env.device))
        done_any = t_done.to(runner.device).detach().bool().reshape(-1)
        command.begin_frontres_local_scenario_k_execution()
        execution_started = True
        frames: list[Any] = []
        survival: list[torch.Tensor] = []
        valid_rows: list[torch.Tensor] = []
        horizon = snapshot["horizon_k"].detach().long()
        for _ in range(int(horizon.max().item())):
            def provider() -> torch.Tensor:
                fresh, infos = runner.env.get_observations()
                return _read_v015_frozen_gmt_observations(
                    runner, fresh, infos, frozen_frontres_prefix=frozen_prefix
                )

            frozen = prepare_frontres_v015_frozen_gmt_step(runner, gmt_observation_provider=provider)
            _raw, _reward, dones, _infos = runner.env.step(frozen.env_actions.to(runner.env.device))
            dones = dones.to(runner.device).detach().bool().reshape(-1)
            valid = frozen.valid_mask.to(device=runner.device, dtype=torch.bool).reshape(-1)
            alive = valid & ~done_any
            frames.append(capture_frontres_v017_execution_frame(runner, selected_rows=repair_rows))
            survival.append(alive.detach().clone())
            valid_rows.append(valid.detach().clone())
            done_any = done_any | (dones & alive)
        repair_trajectory, _expected = _stack_v017_selected_role_execution_frames(
            frames,
            survival=survival,
            valid=valid_rows,
            role_rows=repair_rows,
        )
        # B3: 将每条 Repair 与其 immutable scenario/policy identity 封装为 typed attempt evidence.
        attempts: list[FrontRESRepairAttemptEvidence] = []
        for row in range(n_repair):
            role_row = int(repair_rows[row].item())
            attempt = FrontRESRepairAttemptEvidence(
                transaction_id=str(transaction_id),
                policy_snapshot_id=str(policy_snapshot_id),
                scenario_id=str(snapshot["scenario_ids"][role_row]),
                noisy_segment_hash=str(snapshot["noisy_segment_hashes"][role_row]),
                x_t_identity=str(snapshot["x_t_identities"][role_row]),
                source_index=int(source_index[row].item()),
                segment_id=int(segment_ids[row].item()),
                trial_index=int(trial_index[row].item()),
                horizon_k=int(horizon[role_row].item()),
                policy_observation=policy_rows["observation"][row],
                policy_privileged_observation=policy_rows["privileged"][row],
                policy_action=policy_rows["action"][row],
                policy_log_prob=policy_rows["log_prob"][row],
                policy_value=policy_rows["value"][row],
                policy_mean=policy_rows["mean"][row],
                policy_sigma=policy_rows["sigma"][row],
                repair=select_frontres_v017_trajectory_rows(
                    repair_trajectory,
                    torch.tensor([row], device=repair_trajectory.joint_pos.device),
                ),
            )
            attempt.validate()
            attempts.append(attempt)
        return tuple(attempts)
    finally:
        if execution_started:
            command.end_frontres_local_scenario_k_execution()
        if hasattr(runner, "_frontres_v015_one_action_k_phase"):
            delattr(runner, "_frontres_v015_one_action_k_phase")


def collect_frontres_v015_one_action_k_evidence(
    runner: Any,
    observations: FrontRESSegmentLiveObservations,
    *,
    pair_layout: Any,
) -> FrontRESV015OneActionKEvidence:
    """Capture one action followed by frozen-GMT K evidence for v015.

    This bypasses the legacy repeated-action loop. It is consumed by the
    explicit pre-live sentinel, formal v015 held-out evaluator, and CPU
    contracts; generic legacy storage/update paths remain isolated.
    """

    from rsl_rl.frontres.frontres_segment_evidence_legacy import FrontRESV015OneActionKEvidence

    command, snapshot, repair_rows = _require_v015_one_action_k_layout(runner, observations, pair_layout)
    n_repair = int(repair_rows.numel())
    frontres_dim = int(getattr(getattr(runner.alg, "policy", None), "num_frontres_obs", 0) or 0)
    # B1: Sentinel 和 ordinary training 从 command owner 读取当前 GMT command
    # 维度. Held-out quality 不消费 training observation trace, 不应触发该读取.
    trace_current_command = bool(
        getattr(runner.alg, "frontres_local_sentinel_only", False)
        or getattr(runner.alg, "frontres_segment_live_train_enabled", False)
    )
    if trace_current_command:
        current_command = command.command
        if not isinstance(current_command, torch.Tensor) or current_command.ndim != 2:
            raise RuntimeError("v015 formal one-action-K requires a rank-2 current GMT command")
        update_frontres_observation_trace(
            runner,
            current_command_dim=int(current_command.shape[-1]),
        )
    frozen_frontres_prefix = (
        observations.obs[:, :frontres_dim].detach().clone()
        if frontres_dim > 0
        else None
    )
    execution_started = False
    actor_forward_count = 0
    later_femr_action_count = 0
    try:
        t_plan = prepare_frontres_v015_one_action_at_t(
            runner,
            obs=observations.obs,
            privileged_obs=observations.privileged_obs,
            teacher_obs=observations.teacher_obs,
            ref_vel_estimator_obs=observations.ref_vel_estimator_obs,
            iteration=int(getattr(runner, "current_learning_iteration", 0)),
            n_repair=n_repair,
            n_noisy=int(getattr(pair_layout, "n_base", 0)),
        )
        actor_forward_count += 1
        transition = getattr(runner.alg, "transition", None)
        required = (
            "observations",
            "privileged_observations",
            "actions_log_prob",
            "values",
            "action_mean",
            "action_sigma",
        )
        if transition is None or any(not isinstance(getattr(transition, name, None), torch.Tensor) for name in required):
            raise RuntimeError("v015 one-action K collector requires the t policy tuple before frozen GMT execution")
        policy_actions = t_plan.actions.index_select(0, repair_rows).detach().clone()
        policy_observations = transition.observations.index_select(0, repair_rows).detach().clone()
        policy_privileged_observations = (
            transition.privileged_observations.index_select(0, repair_rows).detach().clone()
        )
        policy_log_probs = transition.actions_log_prob.index_select(0, repair_rows).detach().clone().reshape(-1)
        policy_values = transition.values.index_select(0, repair_rows).detach().clone().reshape(-1)
        policy_mean, policy_sigma = _require_direct_policy_stats(transition)
        policy_means = policy_mean.index_select(0, repair_rows).detach().clone()
        policy_sigmas = policy_sigma.index_select(0, repair_rows).detach().clone()
        if tuple(policy_actions.shape) != tuple(policy_means.shape) or tuple(policy_actions.shape) != tuple(policy_sigmas.shape):
            raise RuntimeError("v015 one-action K collector requires aligned full-6D old policy statistics")
        if (
            policy_privileged_observations.ndim != 2
            or int(policy_privileged_observations.shape[0]) != n_repair
            or int(policy_privileged_observations.shape[1]) <= 0
        ):
            raise RuntimeError("v015 one-action K collector requires one non-empty t critic observation per Repair row")
        update_frontres_observation_trace(
            runner,
            critic_observation_dim=int(policy_privileged_observations.shape[-1]),
        )

        _raw_obs, _rewards, t_dones, _infos = runner.env.step(t_plan.env_actions.to(runner.env.device))
        t_dones = t_dones.to(runner.device).detach().bool().reshape(-1)
        if int(t_dones.numel()) != int(t_plan.env_actions.shape[0]):
            raise RuntimeError("v015 one-action K collector requires one t done flag per scored role")
        executed_q29_t = _capture_v015_post_t_executed_q29(
            command,
            role_count=int(t_plan.env_actions.shape[0]),
            device=runner.device,
        )
        survival_steps = torch.zeros_like(t_dones, dtype=torch.float32)
        done_any = t_dones.detach().clone()
        begin = getattr(command, "begin_frontres_local_scenario_k_execution", None)
        if not callable(begin):
            raise RuntimeError("v015 one-action K collector requires command Clean-continuation lifecycle ownership")
        begin()
        execution_started = True

        continuation_frames: list[torch.Tensor] = []
        valid_frames: list[torch.Tensor] = []
        gmt_action_frames: list[torch.Tensor] = []
        zmp_repaired_frames: list[torch.Tensor] = []
        zmp_noisy_frames: list[torch.Tensor] = []
        contact_repaired_frames: list[torch.Tensor] = []
        contact_noisy_frames: list[torch.Tensor] = []
        expected_support_frames: list[torch.Tensor] = []
        physics_pair_valid_frames: list[torch.Tensor] = []
        survival_repaired_frames: list[torch.Tensor] = []
        survival_noisy_frames: list[torch.Tensor] = []
        lean_repaired_frames: list[torch.Tensor] = []
        lean_noisy_frames: list[torch.Tensor] = []
        quality_trace_enabled = hasattr(runner, "_frontres_v015_quality_action_route")
        horizon_k = snapshot["horizon_k"].detach().long().clone()
        for _offset in range(int(horizon_k.max().item())):
            if getattr(runner, "_frontres_v015_one_action_k_phase", None) != "frozen":
                raise RuntimeError("v015 one-action K collector lost its frozen-FEMR phase before GMT continuation")

            def post_advance_gmt_observation() -> torch.Tensor:
                fresh_obs, fresh_infos = runner.env.get_observations()
                return _read_v015_frozen_gmt_observations(
                    runner,
                    fresh_obs,
                    fresh_infos,
                    frozen_frontres_prefix=frozen_frontres_prefix,
                )

            frozen_plan = prepare_frontres_v015_frozen_gmt_step(
                runner,
                gmt_observation_provider=post_advance_gmt_observation,
            )
            continuation_frames.append(frozen_plan.continuation)
            valid_frames.append(frozen_plan.valid_mask)
            gmt_action_frames.append(frozen_plan.env_actions)
            _raw_obs, _rewards, frozen_dones, _infos = runner.env.step(frozen_plan.env_actions.to(runner.env.device))
            frozen_dones = frozen_dones.to(runner.device).detach().bool().reshape(-1)
            valid = frozen_plan.valid_mask.to(device=runner.device, dtype=torch.bool).reshape(-1)
            if int(frozen_dones.numel()) != int(valid.numel()):
                raise RuntimeError("v015 one-action K collector requires one frozen-GMT done flag per role")
            alive = valid & (~done_any)
            physics_frame = capture_frontres_physics_frame(runner, pair_layout)
            if physics_frame is None:
                raise RuntimeError(
                    "v015 one-action K collector requires paired ZMP/contact evidence on every executable K step"
                )
            pair_valid = alive[:n_repair] & alive[n_repair : 2 * n_repair]
            survival_repaired_frames.append(alive[:n_repair].detach().clone())
            survival_noisy_frames.append(alive[n_repair : 2 * n_repair].detach().clone())
            nan = torch.full((n_repair,), float("nan"), device=runner.device, dtype=torch.float32)
            expected_support, contact_repaired, contact_noisy = physics_frame[2:]
            for name, frame, destination in (
                ("expected_support", expected_support, expected_support_frames),
                ("contact_repaired", contact_repaired, contact_repaired_frames),
                ("contact_noisy", contact_noisy, contact_noisy_frames),
            ):
                frame = frame.detach().to(device=runner.device, dtype=torch.float32)
                if tuple(frame.shape) != (n_repair, 2):
                    raise RuntimeError(f"v015 one-action K collector received invalid {name} shape {tuple(frame.shape)}")
                destination.append(frame.clone())
            expected_loaded = expected_support.bool().any(dim=-1)
            frame_names = (
                ("zmp_repaired", physics_frame[0], contact_repaired, zmp_repaired_frames),
                ("zmp_noisy", physics_frame[1], contact_noisy, zmp_noisy_frames),
            )
            for name, frame, actual_contact, destination in frame_names:
                frame = frame.detach().to(device=runner.device, dtype=torch.float32).reshape(-1)
                applicable = pair_valid & expected_loaded & actual_contact.bool().any(dim=-1)
                finite = torch.isfinite(frame)
                invalid_physical = pair_valid & ~applicable
                if int(frame.numel()) != n_repair or not bool(finite[applicable].all()) or bool(finite[invalid_physical].any()):
                    raise RuntimeError(
                        f"v015 one-action K collector received invalid loaded-support applicability for {name}"
                    )
                destination.append(torch.where(applicable, frame, nan).detach().clone())
            physics_pair_valid_frames.append(pair_valid.detach().clone())
            if quality_trace_enabled:
                lean_frame = capture_frontres_quality_lateral_lean_frame(runner, pair_layout)
                if lean_frame is None:
                    raise RuntimeError("v015 quality requires evaluation-only paired lateral-lean evidence")
                for frame, destination in zip(lean_frame, (lean_repaired_frames, lean_noisy_frames), strict=True):
                    frame = frame.detach().to(device=runner.device, dtype=torch.float32).reshape(-1)
                    if int(frame.numel()) != n_repair or not bool(torch.isfinite(frame[pair_valid]).all()):
                        raise RuntimeError("v015 quality received invalid lateral-lean evidence")
                    destination.append(torch.where(pair_valid, frame, nan).detach().clone())
            survival_steps = survival_steps + alive.to(dtype=survival_steps.dtype)
            done_any = done_any | (frozen_dones & alive)

        intent_q29_provenance, intent_q29_source = _v015_intent_provenance_rows(
            snapshot,
            role_count=int(t_plan.env_actions.shape[0]),
        )

        evidence = FrontRESV015OneActionKEvidence(
            policy_observations=policy_observations,
            policy_privileged_observations=policy_privileged_observations,
            policy_actions=policy_actions,
            policy_log_probs=policy_log_probs,
            policy_values=policy_values,
            policy_means=policy_means,
            policy_sigmas=policy_sigmas,
            policy_row_indices=repair_rows.detach().clone(),
            t_env_actions=t_plan.env_actions.detach().clone(),
            continuation=torch.stack(continuation_frames, dim=0),
            continuation_valid_mask=torch.stack(valid_frames, dim=0),
            frozen_gmt_env_actions=torch.stack(gmt_action_frames, dim=0),
            actor_forward_count=actor_forward_count,
            later_femr_action_count=later_femr_action_count,
            horizon_k=horizon_k,
            scenario_ids=tuple(snapshot["scenario_ids"]),
            noisy_segment_hashes=tuple(snapshot["noisy_segment_hashes"]),
            x_t_identities=tuple(snapshot["x_t_identities"]),
            roles=tuple(snapshot["roles"]),
            intent_q29=snapshot["intent_q29"].detach().clone(),
            intent_q29_provenance=intent_q29_provenance,
            intent_q29_source=intent_q29_source,
            executed_q29_t=executed_q29_t,
            executed_q29_t_valid_mask=(~t_dones).detach().clone(),
            done_any=done_any.detach().clone(),
            survival_steps=survival_steps.detach().clone(),
            physics_expected_support_steps=torch.stack(expected_support_frames, dim=0),
            physics_zmp_repaired_steps=torch.stack(zmp_repaired_frames, dim=0),
            physics_zmp_noisy_steps=torch.stack(zmp_noisy_frames, dim=0),
            physics_contact_repaired_steps=torch.stack(contact_repaired_frames, dim=0),
            physics_contact_noisy_steps=torch.stack(contact_noisy_frames, dim=0),
            physics_pair_valid_mask=torch.stack(physics_pair_valid_frames, dim=0),
            physics_survival_repaired_steps=torch.stack(survival_repaired_frames, dim=0),
            physics_survival_noisy_steps=torch.stack(survival_noisy_frames, dim=0),
            evaluation_only_lateral_lean_repaired_steps=(
                torch.stack(lean_repaired_frames, dim=0) if quality_trace_enabled else None
            ),
            evaluation_only_lateral_lean_noisy_steps=(
                torch.stack(lean_noisy_frames, dim=0) if quality_trace_enabled else None
            ),
        )
        evidence.validate()
        # AUDIT-B03/B04: 在 one-action-K owner 输出 observation authority 与 frozen-GMT 事实.
        print_one_action_k_audit(runner, evidence=evidence)
        return evidence
    finally:
        if execution_started:
            end = getattr(command, "end_frontres_local_scenario_k_execution", None)
            if not callable(end):
                raise RuntimeError("v015 one-action K collector requires command Clean-continuation close ownership")
            end()
        if hasattr(runner, "_frontres_v015_one_action_k_phase"):
            delattr(runner, "_frontres_v015_one_action_k_phase")


def collect_frontres_v015_gain_return_priority_evidence(
    runner: Any,
    observations: FrontRESSegmentLiveObservations,
    *,
    pair_layout: Any,
    gain_config: Any | None = None,
) -> FrontRESV015GainConsumerEvidence:
    """Run the v015 capture -> v003 Gain -> return/priority chain.

    It consumes a sealed Repair/Noisy reset and one t policy action. The
    explicit pre-live sentinel may pass its immutable result to the candidate
    adapter; it does not mutate sampler state or invoke legacy Gain/storage.
    """

    from rsl_rl.frontres.frontres_segment_evidence_legacy import (
        build_frontres_v015_gain_return_evidence,
        pair_frontres_v015_gain_facts,
    )

    one_action = collect_frontres_v015_one_action_k_evidence(
        runner,
        observations,
        pair_layout=pair_layout,
    )
    facts = pair_frontres_v015_gain_facts(one_action)
    gain_module = frontres_gain_module()
    if gain_module is None:
        raise RuntimeError("v015 Gain consumer chain requires the frontres_gain owner")
    config_cls = getattr(gain_module, "FrontRESIntentPhysicsGainConfig", None)
    input_cls = getattr(gain_module, "FrontRESIntentPhysicsGainInput", None)
    compute = getattr(gain_module, "compute_intent_physics_local_repair_gain", None)
    if not callable(config_cls) or not callable(input_cls) or not callable(compute):
        raise RuntimeError("v015 Gain consumer chain rejects the legacy Clean-global Gain owner")
    config = config_cls() if gain_config is None else gain_config
    gain_input = input_cls(
        intent_q29=facts.intent_q29,
        repaired_q29=facts.repaired_q29,
        noisy_q29=facts.noisy_q29,
        intent_q29_provenance=facts.intent_q29_provenance,
        intent_q29_source=facts.intent_q29_source,
        repair_action_steps=facts.policy_actions,
        intent_valid_mask=facts.intent_valid_mask,
        repaired_success=facts.repaired_success,
        noisy_success=facts.noisy_success,
        repaired_survival=facts.repaired_survival,
        noisy_survival=facts.noisy_survival,
        effective_horizon_k=facts.horizon_k,
        repaired_zmp_margin=facts.repaired_zmp_margin,
        noisy_zmp_margin=facts.noisy_zmp_margin,
        repaired_contact=facts.repaired_contact,
        noisy_contact=facts.noisy_contact,
        repaired_contact_violation=facts.repaired_contact_violation,
        noisy_contact_violation=facts.noisy_contact_violation,
        repaired_zmp_violation=facts.repaired_zmp_violation,
        noisy_zmp_violation=facts.noisy_zmp_violation,
        expected_support_steps=facts.expected_support_steps,
        repaired_contact_steps=facts.repaired_contact_steps,
        noisy_contact_steps=facts.noisy_contact_steps,
        repaired_zmp_margin_steps=facts.repaired_zmp_margin_steps,
        noisy_zmp_margin_steps=facts.noisy_zmp_margin_steps,
        physics_pair_valid_mask=facts.physics_pair_valid_mask,
    )
    gain_result = compute(gain_input, config=config)
    return_evidence = build_frontres_v015_gain_return_evidence(facts, gain_result)
    try:
        from rsl_rl.frontres.frontres_segment_planning import build_frontres_v015_priority_evidence
    except ModuleNotFoundError as exc:
        raise RuntimeError("v015 Gain consumer chain requires the sampler priority-evidence owner") from exc
    priority_evidence = build_frontres_v015_priority_evidence(return_evidence)
    result = FrontRESV015GainConsumerEvidence(
        one_action=one_action,
        return_evidence=return_evidence,
        priority_evidence=priority_evidence,
    )
    result.validate()
    return result


def build_frontres_v015_grouped_candidate_batch(
    candidate_evidence: FrontRESV015GainConsumerEvidence,
    *,
    transaction_id: str,
    policy_snapshot_id: str,
    motion_ids: tuple[str, ...],
    start_frames: torch.Tensor,
    segment_ids: torch.Tensor,
    source_index: torch.Tensor,
    trial_index: torch.Tensor,
) -> FrontRESSegmentPPOBatch:
    """Connect sealed v015 candidate evidence to a grouped PPO batch.

    函数名说明:
        `build_frontres_v015_grouped_candidate_batch` 是 Step 4A candidate connector.
        它不创建 frozen snapshot, 不调用 runner storage, 不执行 loss/backward/step,
        也不触碰 priority 或 sampler state.

    主链路:
        上游: Step 3B v003 Gain return evidence 与显式 transaction row identity.
        下游: explicit pre-live sentinel or CPU formal transaction provider.

    语义:
        每个 Repair policy attempt 只映射到一个 PPO row. `evidence_valid_step_count`
        是 K-step executability metadata, 不参与 actor mass 或 grouped formula.
    """

    # B1: storage owner validates sealed local scenario and one-row policy tuple.
    storage_batch = build_frontres_v015_grouped_candidate_storage(
        candidate_evidence,
        transaction_id=transaction_id,
        policy_snapshot_id=policy_snapshot_id,
        motion_ids=motion_ids,
        start_frames=start_frames,
        segment_ids=segment_ids,
        source_index=source_index,
        trial_index=trial_index,
    )

    # B2: only the explicit grouped candidate adapter may retain v015 metadata.
    candidate_batch = storage_batch.to_grouped_ppo_candidate_batch(FrontRESSegmentPPOBatch)
    metadata = candidate_batch.transaction_metadata
    validate_metadata = getattr(metadata, "validate", None)
    if not callable(validate_metadata):
        raise TypeError("v015 grouped candidate connector lost sealed transaction metadata")
    validate_metadata()

    # B3: return a detached batch; the explicit transaction owner alone may evaluate loss or step.
    return candidate_batch


# Public observation/execution seams. Clean continuation remains internal.
resolve_frontres_probe_modes = _resolve_probe_modes
append_fixed_noisy_actor_context = _append_fixed_noisy_actor_context
