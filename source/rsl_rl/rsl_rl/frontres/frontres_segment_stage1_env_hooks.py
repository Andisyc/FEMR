from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import torch


def _load_same_dir(module_name: str):
    path = Path(__file__).with_name(f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(module_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


try:
    from rsl_rl.frontres.frontres_segment_cache_noisy_capture import FrontRESNoisyBaselineResult
    from rsl_rl.frontres.frontres_segment_cache_schema import (
        FrontRESPerturbationDescriptor,
        FrontRESRobotRolloutState,
        FrontRESSegmentIndex,
    )
except ModuleNotFoundError:
    _noisy_capture = _load_same_dir("frontres_segment_cache_noisy_capture")
    _schema = _load_same_dir("frontres_segment_cache_schema")
    FrontRESNoisyBaselineResult = _noisy_capture.FrontRESNoisyBaselineResult
    FrontRESPerturbationDescriptor = _schema.FrontRESPerturbationDescriptor
    FrontRESRobotRolloutState = _schema.FrontRESRobotRolloutState
    FrontRESSegmentIndex = _schema.FrontRESSegmentIndex


@dataclass
class FrontRESStage1EnvAdapter:
    env: Any
    amass_root: str
    robot_name: str = "robot"
    trace: bool = True
    baseline_rollout_steps: int | None = None
    trace_preview_count: int = 4

    def __post_init__(self) -> None:
        self.base_env = getattr(self.env, "unwrapped", self.env)
        self.command = self._resolve_motion_command()
        self.robot = getattr(self.command, "robot", None) or self._resolve_robot()
        self.motion_path_to_index = self._build_motion_path_index()

    @property
    def unwrapped(self) -> Any:
        return self.base_env

    @property
    def scene(self) -> Any:
        return self.base_env.scene

    def frontres_loaded_motion_paths(self) -> list[str]:
        return [str(path) for path in getattr(self.command.motion_dir_loader, "motion_paths", [])]

    def frontres_motion_loader_probe(self) -> dict[str, Any]:
        loader = getattr(self.command, "motion_dir_loader", None)
        cfg = getattr(self.command, "cfg", None)
        loaded_paths = list(getattr(loader, "motion_paths", []) or [])
        all_paths = list(getattr(loader, "motion_paths_all", []) or [])
        shard_info = dict(getattr(loader, "shard_info", {}) or {})
        return {
            "loaded_motion_count": len(loaded_paths),
            "all_motion_count": len(all_paths),
            "cfg_motion_dataset_load_cap": getattr(cfg, "motion_dataset_load_cap", None),
            "cfg_motion_dataset_shard_across_gpus": getattr(cfg, "motion_dataset_shard_across_gpus", None),
            "shard_selected_motions": shard_info.get("selected_motions"),
            "shard_total_motions": shard_info.get("total_motions"),
            "first_loaded_motion": str(loaded_paths[0]) if loaded_paths else "none",
        }

    def ensure_frontres_env_reset(self) -> dict[str, bool]:
        if bool(getattr(self, "_frontres_env_reset_done", False)):
            return {"reset_called": False, "already_reset": True}
        reset_fn = getattr(self.env, "reset", None)
        if not callable(reset_fn):
            self._frontres_env_reset_done = True
            self._trace("env_reset", reset_called=False, already_reset=False)
            return {"reset_called": False, "already_reset": False}
        result = reset_fn()
        self._frontres_env_reset_done = True
        self._trace("env_reset", reset_called=True, already_reset=False, result_type=type(result).__name__)
        return {"reset_called": True, "already_reset": False}

    def prepare_frontres_clean_segment(self, *, segment: FrontRESSegmentIndex, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        segment.validate()
        ids = self._normalize_env_ids(env_ids)
        motion_index = self._motion_index_for_segment(segment)
        frame_index = self._frame_index_for_segment(segment, motion_index)
        with torch.inference_mode():
            self.command.env_motion_indices[ids] = int(motion_index)
            self.command.time_steps[ids] = int(frame_index)
            if hasattr(self.command, "motion_end_buf"):
                self.command.motion_end_buf[ids] = False
            self._reset_frontres_command_state(ids)
            self._write_command_reference_to_robot(ids)
        self._trace(
            "prepare_clean",
            segment_id=int(segment.segment_id),
            motion_index=int(motion_index),
            frame_index=int(frame_index),
            env_ids=ids.detach().cpu().tolist(),
            root_pos=self.robot.data.root_pos_w.index_select(0, ids),
            joint_pos=self.robot.data.joint_pos.index_select(0, ids),
        )
        return {"success": torch.ones(ids.numel(), dtype=torch.bool, device=ids.device)}

    def materialize_frontres_fixed_noisy_tape(
        self,
        *,
        motion_id: str,
        start_frame: int,
        frame_count: int,
        perturbation_family: str,
        perturbation_strength: float,
    ) -> torch.Tensor:
        """Route one selection-time scenario to the command-owned tape materializer."""

        motion_index = self._motion_index_for_key(str(motion_id))
        frame_index = self._frame_index_for_values(int(start_frame), motion_index)
        materialize = getattr(self.command, "materialize_frontres_fixed_noisy_tape", None)
        if not callable(materialize):
            raise RuntimeError(
                "fixed Noisy Segment requires MultiMotionCommand.materialize_frontres_fixed_noisy_tape()"
            )
        tape = materialize(
            motion_index=motion_index,
            start_frame=frame_index,
            frame_count=int(frame_count),
            perturbation_family=str(perturbation_family),
            perturbation_strength=float(perturbation_strength),
        )
        if not isinstance(tape, torch.Tensor) or tape.ndim != 2:
            raise RuntimeError(f"command fixed Noisy materializer returned invalid shape {getattr(tape, 'shape', None)}")
        expected_dim_fn = getattr(self.command, "_frontres_fixed_noisy_tape_feature_dim", None)
        expected_dim = int(expected_dim_fn()) if callable(expected_dim_fn) else 65
        if (
            tape.requires_grad
            or not torch.is_floating_point(tape)
            or not bool(torch.isfinite(tape).all().item())
            or int(tape.shape[0]) != int(frame_count)
            or int(tape.shape[1]) != expected_dim
        ):
            raise RuntimeError(
                "command fixed Noisy materializer must return detached finite "
                f"[L,{expected_dim}] tape, got {tuple(tape.shape)}"
            )
        return tape.detach().to(device=self.command.device, dtype=torch.float32).clone().contiguous()

    def materialize_frontres_local_scenario(
        self,
        *,
        motion_id: str,
        start_frame: int,
        horizon_k: int,
        intent_horizon: int,
        perturbation_family: str,
        perturbation_strength: float,
    ) -> dict[str, Any]:
        """Route the v015 split local carrier to the command-owned materializer."""

        motion_index = self._motion_index_for_key(str(motion_id))
        frame_index = self._frame_index_for_values(int(start_frame), motion_index)
        materialize = getattr(self.command, "materialize_frontres_local_scenario", None)
        if not callable(materialize):
            raise RuntimeError(
                "v015 local scenario requires MultiMotionCommand.materialize_frontres_local_scenario()"
            )
        payload = materialize(
            motion_index=motion_index,
            start_frame=frame_index,
            horizon_k=int(horizon_k),
            intent_horizon=int(intent_horizon),
            perturbation_family=str(perturbation_family),
            perturbation_strength=float(perturbation_strength),
        )
        if not isinstance(payload, dict):
            raise RuntimeError(f"command local scenario materializer returned {type(payload)!r}, expected dict")
        required = {
            "current_root_artifact_t",
            "intent_q29",
            "clean_continuation",
            "provenance",
        }
        if set(payload) != required:
            raise RuntimeError(
                "command local scenario materializer must return exactly "
                f"{sorted(required)}, got {sorted(payload)}"
            )
        artifact = payload["current_root_artifact_t"]
        intent = payload["intent_q29"]
        continuation = payload["clean_continuation"]
        provenance = payload["provenance"]
        if (
            not isinstance(artifact, torch.Tensor)
            or not isinstance(intent, torch.Tensor)
            or not isinstance(continuation, torch.Tensor)
            or artifact.requires_grad
            or intent.requires_grad
            or continuation.requires_grad
            or not torch.is_floating_point(artifact)
            or not torch.is_floating_point(intent)
            or not torch.is_floating_point(continuation)
            or not bool(torch.isfinite(artifact).all().item())
            or not bool(torch.isfinite(intent).all().item())
            or not bool(torch.isfinite(continuation).all().item())
            or tuple(artifact.shape) != (7,)
            or tuple(intent.shape) != (int(intent_horizon) + 1, 29)
            or tuple(continuation.shape) != (int(horizon_k), 65)
        ):
            raise RuntimeError(
                "command local scenario materializer must return detached finite "
                f"[7], [{int(intent_horizon) + 1},29], [{int(horizon_k)},65] payloads"
            )
        if not isinstance(provenance, dict):
            raise RuntimeError("command local scenario provenance must be a dict")
        return {
            "current_root_artifact_t": artifact.detach().to(device=self.command.device, dtype=torch.float32).clone().contiguous(),
            "intent_q29": intent.detach().to(device=self.command.device, dtype=torch.float32).clone().contiguous(),
            "clean_continuation": continuation.detach().to(device=self.command.device, dtype=torch.float32).clone().contiguous(),
            "provenance": dict(provenance),
        }

    def apply_frontres_segment_index_reset(self, request: Any) -> dict[str, torch.Tensor]:
        """将 sampled Segment 状态重置到显式配对的全部 split-env role.

        状态: active Stage 1 index、Stage 3 quartet reset 与 S2 sealed-tape reset owner.
        上游: live probe 从 pair layout 附加 `frontres_role_env_ids`.
        下游: frozen GMT rollout 消费 role-aligned robot 与 episode state.
        证据: E37 确认 legacy quartet lifecycle；S2 offline contracts 确认 sealed tape reset.
        缺口: v013 actor/normalizer 尚无 live evidence。
        """
        segment_ids = getattr(request, "segment_ids")
        count = int(segment_ids.numel())
        is_v015_local = getattr(request, "frontres_local_scenario_rows", None) is not None
        role_env_ids = self._normalize_frontres_role_env_ids(
            request,
            source_count=count,
            v015_local=is_v015_local,
        )
        source_ids = role_env_ids["repair"] if is_v015_local else role_env_ids["policy"]
        ids = torch.cat(tuple(role_env_ids.values()), dim=0)
        source_rows = torch.arange(count, device=ids.device, dtype=torch.long).repeat(len(role_env_ids))
        local_scenario = self._v015_local_scenario_reset_payload(
            request,
            source_count=count,
            device=ids.device,
        )
        fixed_noisy = None if local_scenario is not None else self._fixed_noisy_reset_payload(
            request,
            source_count=count,
            device=ids.device,
        )
        num_envs = int(getattr(self.base_env, "num_envs", getattr(self.command, "num_envs", count)) or count)
        if (
            int(ids.numel()) > num_envs
            or (ids.numel() and int(ids.min().item()) < 0)
            or (ids.numel() and int(ids.max().item()) >= num_envs)
        ):
            raise ValueError(f"index reset role rows {ids.tolist()} exceed env count {num_envs}")
        motion_ids = tuple(str(item) for item in getattr(request, "motion_ids"))
        start_frames = getattr(request, "start_frames").to(device=ids.device, dtype=torch.long).flatten()
        if len(motion_ids) != count or int(start_frames.numel()) != count:
            raise ValueError("motion_ids and start_frames must match segment_ids count")
        motion_indices = torch.tensor(
            [self._motion_index_for_key(motion_id) for motion_id in motion_ids],
            dtype=torch.long,
            device=ids.device,
        )
        frame_indices = torch.tensor(
            [
                self._frame_index_for_values(int(frame.item()), int(motion_index.item()))
                for frame, motion_index in zip(start_frames, motion_indices, strict=True)
            ],
            dtype=torch.long,
            device=ids.device,
        )
        expanded_motion_indices = motion_indices.index_select(0, source_rows)
        expanded_frame_indices = frame_indices.index_select(0, source_rows)
        with torch.inference_mode():
            # B1: 按稳定 role 顺序展开 sampled policy motion/frame.
            self.command.env_motion_indices[ids] = expanded_motion_indices
            self._write_frontres_motion_groups(ids, expanded_motion_indices)
            self.command.time_steps[ids] = expanded_frame_indices
            if hasattr(self.command, "motion_end_buf"):
                self.command.motion_end_buf[ids] = False
            # B2: role-specific reference 生效前, 所有 role 获得同源 dynamic start.
            self._reset_frontres_command_state(ids, reset_perturber=fixed_noisy is None and local_scenario is None)
            if local_scenario is not None:
                if int(ids.numel()) != int(getattr(self.command, "num_envs", ids.numel())):
                    raise RuntimeError(
                        "v015 local scenario reset requires all command rows so Repair/Noisy artifacts cannot mix"
                    )
                set_local_scenario = getattr(self.command, "set_frontres_local_scenario", None)
                if not callable(set_local_scenario):
                    raise RuntimeError("v015 local scenario reset requires command.set_frontres_local_scenario()")
                applied = set_local_scenario(
                    current_root_artifact_t=local_scenario["current_root_artifact_t"].index_select(0, source_rows),
                    intent_q29=local_scenario["intent_q29"].index_select(0, source_rows),
                    clean_continuation=local_scenario["clean_continuation"].index_select(0, source_rows),
                    horizon_k=local_scenario["horizon_k"].index_select(0, source_rows),
                    continuation_lengths=local_scenario["continuation_lengths"].index_select(0, source_rows),
                    scenario_ids=tuple(local_scenario["scenario_ids"][int(row)] for row in source_rows.tolist()),
                    noisy_segment_hashes=tuple(local_scenario["hashes"][int(row)] for row in source_rows.tolist()),
                    x_t_identities=tuple(local_scenario["x_t_identities"][int(row)] for row in source_rows.tolist()),
                    provenance=tuple(local_scenario["provenance"][int(row)] for row in source_rows.tolist()),
                    roles=tuple(
                        role
                        for role, role_ids in role_env_ids.items()
                        for _ in range(int(role_ids.numel()))
                    ),
                    env_ids=ids,
                )
                if not isinstance(applied, torch.Tensor) or not bool(applied.detach().bool().all()):
                    raise RuntimeError("command rejected one or more v015 local scenario rows during index reset")
                perturbation_state = {
                    "strength": None,
                    "family": tuple(),
                    "family_masks": None,
                    "local_scenario_hashes": local_scenario["hashes"],
                }
            elif fixed_noisy is None:
                local_active = getattr(self.command, "_frontres_local_scenario_active", None)
                if isinstance(local_active, torch.Tensor) and bool(local_active.any()):
                    raise RuntimeError("legacy reset cannot overwrite an active v015 local scenario")
                clear_fixed_tape = getattr(self.command, "clear_frontres_fixed_noisy_tape", None)
                if callable(clear_fixed_tape):
                    clear_fixed_tape(ids)
                perturbation_state = self._apply_index_reset_perturbation_request(request, source_ids)
            else:
                if int(ids.numel()) != int(getattr(self.command, "num_envs", ids.numel())):
                    raise RuntimeError(
                        "fixed Noisy Segment reset requires all command rows so random and sealed references cannot mix"
                    )
                set_fixed_tape = getattr(self.command, "set_frontres_fixed_noisy_tape", None)
                if not callable(set_fixed_tape):
                    raise RuntimeError("fixed Noisy Segment reset requires command.set_frontres_fixed_noisy_tape()")
                role_execution = torch.cat(
                    [
                        torch.full(
                            (int(role_ids.numel()),),
                            role != "clean",
                            dtype=torch.bool,
                            device=ids.device,
                        )
                        for role, role_ids in role_env_ids.items()
                    ],
                    dim=0,
                )
                applied = set_fixed_tape(
                    fixed_noisy["tape"].index_select(0, source_rows),
                    tape_lengths=fixed_noisy["tape_lengths"].index_select(0, source_rows),
                    scenario_ids=tuple(fixed_noisy["scenario_ids"][int(row)] for row in source_rows.tolist()),
                    noisy_segment_hashes=tuple(fixed_noisy["hashes"][int(row)] for row in source_rows.tolist()),
                    execution_mask=role_execution,
                    env_ids=ids,
                )
                if not isinstance(applied, torch.Tensor) or not bool(applied.detach().bool().all()):
                    raise RuntimeError("command rejected one or more fixed Noisy tape rows during index reset")
                perturbation_state = {
                    "strength": None,
                    "family": tuple(),
                    "family_masks": None,
                    "fixed_noisy_hashes": fixed_noisy["hashes"],
                }
            refresh_reference_cache = getattr(
                self.command,
                "refresh_frontres_reference_cache_current_frame",
                None,
            )
            if not callable(refresh_reference_cache):
                raise RuntimeError(
                    "FrontRES index reset requires command.refresh_frontres_reference_cache_current_frame()"
                )
            # Sampled motion/frame 已显式写入, 此处只刷新 cache, 不调用会推进 time_steps 的 _update_command().
            refresh_reference_cache()
            self._write_command_reference_to_robot(ids)
            # B3: Segment index reset 后不得保留随机化的旧 episode age.
            self._reset_frontres_episode_lifecycle(ids)
        success = torch.ones(count, dtype=torch.bool, device=segment_ids.device)
        velocity = torch.zeros(count, dtype=torch.float32, device=segment_ids.device)
        self._trace(
            "index_reset",
            segment_ids=segment_ids,
            motion_ids=motion_ids,
            motion_indices=motion_indices,
            start_frames=start_frames,
            frame_indices=frame_indices,
            env_ids=ids.detach().cpu().tolist(),
            role_env_ids={role: role_ids.detach().cpu().tolist() for role, role_ids in role_env_ids.items()},
            root_pos=self.robot.data.root_pos_w.index_select(0, ids),
            joint_pos=self.robot.data.joint_pos.index_select(0, ids),
            perturbation_strength=perturbation_state.get("strength"),
            perturbation_family=perturbation_state.get("family"),
            perturbation_family_masks=perturbation_state.get("family_masks"),
            cached_perturbed_pos=getattr(self.command, "_cached_perturbed_pos", None),
            perturber_dr_scale_env=getattr(getattr(self.command, "perturber", None), "_dr_scale_env", None),
            perturber_family_masks=getattr(getattr(self.command, "perturber", None), "_family_masks", None),
            fixed_noisy_hashes=perturbation_state.get("fixed_noisy_hashes"),
            local_scenario_hashes=perturbation_state.get("local_scenario_hashes"),
        )
        return {"reset_success": success, "velocity_mismatch": velocity}

    def _v015_local_scenario_reset_payload(
        self,
        request: Any,
        *,
        source_count: int,
        device: torch.device,
    ) -> dict[str, Any] | None:
        if getattr(request, "frontres_local_scenario_rows", None) is None:
            return None
        if getattr(request, "frontres_fixed_noisy_tape", None) is not None:
            raise ValueError("v015 local reset cannot mix a local scenario with a legacy fixed Noisy tape")
        artifact = getattr(request, "frontres_local_scenario_current_root_artifact_t", None)
        intent = getattr(request, "frontres_local_scenario_intent_q29", None)
        continuation = getattr(request, "frontres_local_scenario_clean_continuation", None)
        if not isinstance(artifact, torch.Tensor) or not isinstance(intent, torch.Tensor) or not isinstance(continuation, torch.Tensor):
            raise ValueError("v015 local reset requires tensor artifact, q29 intent, and Clean continuation fields")
        for name, value in (
            ("current_root_artifact_t", artifact),
            ("intent_q29", intent),
            ("clean_continuation", continuation),
        ):
            if (
                value.requires_grad
                or not torch.is_floating_point(value)
                or not bool(torch.isfinite(value).all().item())
            ):
                raise ValueError(f"v015 local {name} must be detached finite floating-point data")
        offsets = tuple(int(value) for value in (getattr(request, "frontres_future_offsets", ()) or ()))
        if not offsets or any(value <= 0 for value in offsets) or tuple(sorted(set(offsets))) != offsets:
            raise ValueError(f"v015 local reset requires nonempty positive ordered future offsets, got {offsets}")
        if (
            tuple(artifact.shape) != (int(source_count), 7)
            or intent.ndim != 3
            or tuple(intent.shape) != (int(source_count), max(offsets) + 1, 29)
            or continuation.ndim != 3
            or int(continuation.shape[0]) != int(source_count)
            or int(continuation.shape[1]) <= 0
            or int(continuation.shape[2]) != 65
        ):
            raise ValueError(
                "v015 local reset requires [B,7] current artifact, [B,H+1,29] q29 intent, and [B,K_max,65] Clean continuation"
            )
        horizon_k = torch.as_tensor(getattr(request, "horizon_k", None), device=device, dtype=torch.long).flatten()
        lengths = torch.as_tensor(
            getattr(request, "frontres_local_scenario_clean_continuation_lengths", None),
            device=device,
            dtype=torch.long,
        ).flatten()
        if (
            int(horizon_k.numel()) != int(source_count)
            or int(lengths.numel()) != int(source_count)
            or bool((horizon_k <= 0).any())
            or not torch.equal(horizon_k, lengths)
            or bool((lengths > int(continuation.shape[1])).any())
        ):
            raise ValueError("v015 local horizon_k and continuation lengths must be equal positive [B] values")
        continuation_mask = torch.as_tensor(
            getattr(request, "frontres_local_scenario_clean_continuation_mask", None),
            device=device,
            dtype=torch.bool,
        )
        expected_mask = torch.arange(int(continuation.shape[1]), device=device).unsqueeze(0) < lengths.unsqueeze(1)
        if tuple(continuation_mask.shape) != tuple(expected_mask.shape) or not torch.equal(continuation_mask, expected_mask):
            raise ValueError("v015 local Clean continuation mask must exactly encode the sealed per-row K lengths")
        scenario_ids = tuple(str(value) for value in (getattr(request, "frontres_local_scenario_ids", ()) or ()))
        hashes = tuple(str(value) for value in (getattr(request, "frontres_local_scenario_hashes", ()) or ()))
        x_t_identities = tuple(str(value) for value in (getattr(request, "frontres_local_scenario_x_t_identities", ()) or ()))
        provenance_raw = tuple(getattr(request, "frontres_local_scenario_provenance", ()) or ())
        if (
            len(scenario_ids) != int(source_count)
            or len(hashes) != int(source_count)
            or len(x_t_identities) != int(source_count)
            or len(provenance_raw) != int(source_count)
            or any(not value for value in scenario_ids)
            or any(not value for value in hashes)
            or any(not value for value in x_t_identities)
            or any(not isinstance(value, dict) for value in provenance_raw)
        ):
            raise ValueError("v015 local reset requires nonempty source-aligned identity and provenance metadata")
        provenance = tuple(dict(value) for value in provenance_raw)
        for row, value in enumerate(provenance):
            if (
                value.get("current_root_artifact_provenance") != "noisy_root_artifact_t"
                or value.get("intent_q29_provenance") != "deployment_noisy_q29"
                or value.get("clean_continuation_provenance") != "clean_gmt_only"
            ):
                raise ValueError(f"v015 local provenance row {row} violates the Noisy-q29/Clean-continuation boundary")
            intent_source = str(value.get("intent_q29_source", "")).lower()
            if not intent_source or "root" in intent_source or "global" in intent_source or "clean" in intent_source:
                raise ValueError(f"v015 local q29 source row {row} may not carry Clean/root/global actor input")
        return {
            "current_root_artifact_t": artifact.detach().to(device=device, dtype=torch.float32).clone().contiguous(),
            "intent_q29": intent.detach().to(device=device, dtype=torch.float32).clone().contiguous(),
            "clean_continuation": continuation.detach().to(device=device, dtype=torch.float32).clone().contiguous(),
            "horizon_k": horizon_k.detach().clone(),
            "continuation_lengths": lengths.detach().clone(),
            "scenario_ids": scenario_ids,
            "hashes": hashes,
            "x_t_identities": x_t_identities,
            "provenance": provenance,
        }

    def _fixed_noisy_reset_payload(self, request: Any, *, source_count: int, device: torch.device) -> dict[str, Any] | None:
        tape = getattr(request, "frontres_fixed_noisy_tape", None)
        if tape is None:
            return None
        if not isinstance(tape, torch.Tensor) or tape.ndim != 3:
            raise ValueError(f"frontres_fixed_noisy_tape must have shape [B,L,65], got {getattr(tape, 'shape', None)}")
        expected_dim_fn = getattr(self.command, "_frontres_fixed_noisy_tape_feature_dim", None)
        expected_dim = int(expected_dim_fn()) if callable(expected_dim_fn) else 65
        if (
            tape.requires_grad
            or not torch.is_floating_point(tape)
            or not bool(torch.isfinite(tape).all().item())
            or int(tape.shape[0]) != int(source_count)
            or int(tape.shape[1]) <= 0
            or int(tape.shape[2]) != expected_dim
        ):
            raise ValueError(
                "frontres_fixed_noisy_tape must be detached finite "
                f"[{source_count},L,{expected_dim}] data, got {tuple(tape.shape)}"
            )
        offsets = tuple(int(value) for value in (getattr(request, "frontres_future_offsets", ()) or ()))
        if not offsets or any(value <= 0 for value in offsets) or tuple(sorted(set(offsets))) != offsets:
            raise ValueError(f"frontres_future_offsets must be nonempty positive ordered offsets, got {offsets}")
        horizon_k = torch.as_tensor(getattr(request, "horizon_k"), device=device, dtype=torch.long).flatten()
        if int(horizon_k.numel()) != int(source_count) or bool((horizon_k <= 0).any()):
            raise ValueError("fixed Noisy reset requires positive source-aligned horizon_k")
        lengths = torch.as_tensor(
            getattr(request, "frontres_fixed_noisy_tape_lengths", None), device=device, dtype=torch.long
        ).flatten()
        if (
            int(lengths.numel()) != int(source_count)
            or bool((lengths <= 0).any())
            or bool((lengths > int(tape.shape[1])).any())
            or bool((lengths < horizon_k + max(offsets)).any())
        ):
            raise ValueError("fixed Noisy tape lengths must cover K + max(H) for every source row")
        scenario_ids = tuple(str(value) for value in (getattr(request, "frontres_fixed_noisy_scenario_ids", ()) or ()))
        hashes = tuple(str(value) for value in (getattr(request, "frontres_fixed_noisy_segment_hashes", ()) or ()))
        if (
            len(scenario_ids) != int(source_count)
            or len(hashes) != int(source_count)
            or any(not value for value in scenario_ids)
            or any(not value for value in hashes)
        ):
            raise ValueError("fixed Noisy reset requires nonempty source-aligned scenario ids and hashes")
        return {
            "tape": tape.detach().to(device=device, dtype=torch.float32).contiguous(),
            "tape_lengths": lengths.detach().clone(),
            "scenario_ids": scenario_ids,
            "hashes": hashes,
        }

    def _normalize_frontres_role_env_ids(
        self,
        request: Any,
        *,
        source_count: int,
        v015_local: bool = False,
    ) -> dict[str, torch.Tensor]:
        raw = getattr(request, "frontres_role_env_ids", None)
        if v015_local:
            if not isinstance(raw, dict) or set(raw) != {"repair", "noisy"}:
                raise ValueError(
                    "v015 local reset requires exactly repair/noisy role rows; policy/candidate/clean roles are rejected"
                )
            result: dict[str, torch.Tensor] = {}
            seen: set[int] = set()
            for role in ("repair", "noisy"):
                role_ids = self._normalize_env_ids(raw[role])
                if int(role_ids.numel()) != int(source_count):
                    raise ValueError(
                        f"v015 local role {role} must have {source_count} rows, got {int(role_ids.numel())}"
                    )
                values = role_ids.detach().cpu().tolist()
                if any(int(value) in seen for value in values):
                    raise ValueError(f"v015 local role {role} overlaps another reset role")
                seen.update(int(value) for value in values)
                result[role] = role_ids
            return result
        if raw is None:
            return {"policy": self._normalize_env_ids(range(source_count))}
        if not isinstance(raw, dict) or "policy" not in raw:
            raise ValueError("frontres_role_env_ids must be a mapping containing policy rows")
        result: dict[str, torch.Tensor] = {}
        seen: set[int] = set()
        for role in ("policy", "candidate", "noisy", "clean"):
            if role not in raw:
                continue
            role_ids = self._normalize_env_ids(raw[role])
            if int(role_ids.numel()) != int(source_count):
                raise ValueError(
                    f"frontres role {role} must have {source_count} rows, got {int(role_ids.numel())}"
                )
            values = role_ids.detach().cpu().tolist()
            if any(int(value) in seen for value in values):
                raise ValueError(f"frontres role {role} overlaps another reset role")
            seen.update(int(value) for value in values)
            result[role] = role_ids
        return result

    def _reset_frontres_episode_lifecycle(self, env_ids: torch.Tensor) -> None:
        for owner in (self.base_env, self.env):
            episode_length_buf = getattr(owner, "episode_length_buf", None)
            if isinstance(episode_length_buf, torch.Tensor):
                episode_length_buf[env_ids.to(episode_length_buf.device)] = 0

    def _write_frontres_motion_groups(self, env_ids: torch.Tensor, motion_indices: torch.Tensor) -> None:
        groups = getattr(self.command, "env_motion_groups", None)
        motion_to_group = getattr(getattr(self.command, "motion_dir_loader", None), "motion_to_group", None)
        group_name_to_idx = getattr(self.command, "group_name_to_idx", None)
        if (
            not isinstance(groups, torch.Tensor)
            or not isinstance(motion_to_group, dict)
            or not isinstance(group_name_to_idx, dict)
        ):
            return
        values = [
            int(group_name_to_idx[motion_to_group.get(int(motion_index.item()), "default")])
            for motion_index in motion_indices
        ]
        groups[env_ids.to(groups.device)] = torch.tensor(values, dtype=groups.dtype, device=groups.device)

    def set_frontres_rollout_state(
        self, *, clean_state: FrontRESRobotRolloutState, env_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        clean_state.validate(name="clean_state")
        ids = self._normalize_env_ids(env_ids)
        if clean_state.batch_size != ids.numel():
            raise ValueError(f"clean_state batch {clean_state.batch_size} does not match env_ids {ids.numel()}")
        root_state = torch.cat(
            [
                clean_state.root_pos.to(ids.device),
                clean_state.root_quat.to(ids.device),
                clean_state.root_lin_vel.to(ids.device),
                clean_state.root_ang_vel.to(ids.device),
            ],
            dim=-1,
        )
        with torch.inference_mode():
            self.robot.write_root_state_to_sim(root_state, env_ids=ids)
            self.robot.write_joint_state_to_sim(
                clean_state.joint_pos.to(ids.device),
                clean_state.joint_vel.to(ids.device),
                env_ids=ids,
            )
        self._trace(
            "reset_clean_state",
            env_ids=ids.detach().cpu().tolist(),
            root_pos=clean_state.root_pos,
            joint_pos=clean_state.joint_pos,
        )
        return {"success": torch.ones(ids.numel(), dtype=torch.bool, device=ids.device)}

    def apply_frontres_segment_perturbation(
        self, *, descriptor: FrontRESPerturbationDescriptor, env_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        descriptor.validate()
        ids = self._normalize_env_ids(env_ids)
        axis = torch.as_tensor(descriptor.params.get("axis", [0.0, 0.0, 0.0]), dtype=torch.float32, device=ids.device)
        if axis.numel() != 3:
            raise ValueError(f"descriptor axis must have 3 values, got {axis.numel()}")
        signed_magnitude = float(
            descriptor.params.get("signed_magnitude", descriptor.params.get("magnitude", descriptor.strength))
        )
        delta = axis.reshape(1, 3) * float(signed_magnitude)
        root_pos = self.robot.data.root_pos_w.index_select(0, ids).clone()
        root_quat = self.robot.data.root_quat_w.index_select(0, ids).clone()
        root_lin_vel = self.robot.data.root_lin_vel_w.index_select(0, ids).clone()
        root_ang_vel = self.robot.data.root_ang_vel_w.index_select(0, ids).clone()
        before_root_pos = root_pos.clone()
        root_pos = root_pos + delta.to(root_pos.device, root_pos.dtype)
        root_lin_vel = root_lin_vel + 0.1 * delta.to(root_lin_vel.device, root_lin_vel.dtype)
        root_state = torch.cat([root_pos, root_quat, root_lin_vel, root_ang_vel], dim=-1)
        with torch.inference_mode():
            self.robot.write_root_state_to_sim(root_state, env_ids=ids)
        self._trace(
            "apply_perturbation",
            segment_id=int(descriptor.segment_id),
            perturbation_id=int(descriptor.perturbation_id),
            strength=float(descriptor.strength),
            env_ids=ids.detach().cpu().tolist(),
            delta=delta,
            root_pos_before=before_root_pos,
            root_pos_after=root_pos,
        )
        return {"success": torch.ones(ids.numel(), dtype=torch.bool, device=ids.device)}

    def rollout_frontres_noisy_baseline(
        self, *, segment: FrontRESSegmentIndex, descriptor: FrontRESPerturbationDescriptor, env_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        self.ensure_frontres_env_reset()
        ids = self._normalize_env_ids(env_ids)
        steps = self._baseline_steps(descriptor)
        fall = torch.zeros(ids.numel(), dtype=torch.float32, device=ids.device)
        rollout_len = torch.zeros(ids.numel(), dtype=torch.float32, device=ids.device)
        rewards = []
        for _ in range(steps):
            action = self._zero_action()
            step_result = self.env.step(action)
            reward, done = self._parse_step_result(step_result)
            if reward is not None:
                rewards.append(reward.index_select(0, ids).detach().float())
            if done is not None:
                fall = torch.maximum(fall, done.index_select(0, ids).detach().float())
            rollout_len += 1.0
        score = self._baseline_score(ids, rewards)
        self._trace(
            "baseline_rollout",
            segment_id=int(segment.segment_id),
            perturbation_id=int(descriptor.perturbation_id),
            steps=int(steps),
            env_ids=ids.detach().cpu().tolist(),
            score=score,
            fall=fall,
            rollout_len=rollout_len,
        )
        baseline = FrontRESNoisyBaselineResult(score=score.detach(), fall=fall.detach(), rollout_len=rollout_len.detach())
        baseline.validate(ids.numel())
        return {"score": baseline.score, "fall": baseline.fall, "rollout_len": baseline.rollout_len}

    def _resolve_motion_command(self) -> Any:
        manager = getattr(self.base_env, "command_manager", None)
        if manager is None or not hasattr(manager, "get_term"):
            raise AttributeError("Stage 1 cache requires base_env.command_manager.get_term('motion').")
        return manager.get_term("motion")

    def _resolve_robot(self) -> Any:
        scene = getattr(self.base_env, "scene", None)
        if scene is None:
            raise AttributeError("Stage 1 cache requires base_env.scene.")
        try:
            return scene[self.robot_name]
        except (KeyError, TypeError):
            pass
        if hasattr(scene, self.robot_name):
            return getattr(scene, self.robot_name)
        raise AttributeError(f"could not resolve robot {self.robot_name!r} from env scene")

    def _build_motion_path_index(self) -> dict[str, int]:
        paths = list(getattr(self.command.motion_dir_loader, "motion_paths", []))
        root = Path(self.amass_root).expanduser().resolve()
        mapping: dict[str, int] = {}
        for idx, value in enumerate(paths):
            path = Path(value).expanduser().resolve()
            mapping[str(path)] = int(idx)
            try:
                mapping[path.relative_to(root).as_posix()] = int(idx)
            except ValueError:
                pass
            mapping[path.name] = int(idx)
        if len(mapping) == 0:
            raise ValueError("motion command has no loaded motion paths")
        return mapping

    def _motion_index_for_segment(self, segment: FrontRESSegmentIndex) -> int:
        return self._motion_index_for_key(str(segment.motion_rel_path))

    def _motion_index_for_key(self, key: str) -> int:
        if key in self.motion_path_to_index:
            return self.motion_path_to_index[key]
        suffix_hits = [
            idx for path_key, idx in self.motion_path_to_index.items() if path_key.endswith(key) or key.endswith(path_key)
        ]
        if len(suffix_hits) == 1:
            return int(suffix_hits[0])
        raise KeyError(f"segment motion path {key!r} is not loaded by the motion command")

    def _frame_index_for_segment(self, segment: FrontRESSegmentIndex, motion_index: int) -> int:
        return self._frame_index_for_values(int(segment.start_frame), motion_index)

    def _frame_index_for_values(self, start_frame: int, motion_index: int) -> int:
        motion_lengths = getattr(self.command, "motion_lengths", None)
        if motion_lengths is None:
            return int(start_frame)
        max_frame = int(motion_lengths[int(motion_index)].item()) - 1
        return min(max(int(start_frame), 0), max(max_frame, 0))

    def _write_command_reference_to_robot(self, env_ids: torch.Tensor) -> None:
        body_pos = self.command._gather_by_motion_for_envs("body_pos_w", env_ids)
        body_quat = self.command._gather_by_motion_for_envs("body_quat_w", env_ids)
        body_lin = self.command._gather_by_motion_for_envs("body_lin_vel_w", env_ids)
        body_ang = self.command._gather_by_motion_for_envs("body_ang_vel_w", env_ids)
        joint_pos = self.command._gather_by_motion_for_envs("joint_pos", env_ids)
        joint_vel = self.command._gather_by_motion_for_envs("joint_vel", env_ids)
        root_pos = body_pos[:, 0] + self.base_env.scene.env_origins[env_ids]
        root_state = torch.cat([root_pos, body_quat[:, 0], body_lin[:, 0], body_ang[:, 0]], dim=-1)
        with torch.inference_mode():
            self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    def _reset_frontres_command_state(self, env_ids: torch.Tensor, *, reset_perturber: bool = True) -> None:
        with torch.inference_mode():
            if hasattr(self.command, "_frontres_pos_correction"):
                self.command._frontres_pos_correction[env_ids] = 0.0
            if hasattr(self.command, "_frontres_quat_correction"):
                self.command._frontres_quat_correction[env_ids] = 0.0
                self.command._frontres_quat_correction[env_ids, 0] = 1.0
            if reset_perturber and hasattr(self.command, "perturber") and hasattr(self.command.perturber, "reset_envs"):
                self.command.perturber.reset_envs(env_ids)

    def _apply_index_reset_perturbation_request(self, request: Any, env_ids: torch.Tensor) -> dict[str, Any]:
        perturber = getattr(self.command, "perturber", None)
        strength_value = getattr(request, "perturbation_strength", None)
        if perturber is None or strength_value is None:
            return {}
        num_envs = int(getattr(self.command, "num_envs", getattr(self.base_env, "num_envs", env_ids.numel())))
        strengths = torch.as_tensor(strength_value, device=env_ids.device, dtype=torch.float32).flatten()
        if int(strengths.numel()) != int(env_ids.numel()):
            raise ValueError(
                f"perturbation_strength has {int(strengths.numel())} rows but reset has {int(env_ids.numel())}"
            )
        scale = torch.zeros(num_envs, dtype=torch.float32, device=env_ids.device)
        scale[env_ids] = strengths.clamp(min=0.0)
        set_scale = getattr(perturber, "set_dr_scale_env", None)
        if callable(set_scale):
            set_scale(scale)

        families = tuple(str(item) for item in (getattr(request, "perturbation_family", ()) or ()))
        if families and len(families) != int(env_ids.numel()):
            raise ValueError(f"perturbation_family has {len(families)} rows but reset has {int(env_ids.numel())}")
        masks = self._family_masks_from_request(families, env_ids, num_envs)
        set_masks = getattr(perturber, "set_family_env_masks", None)
        if callable(set_masks):
            set_masks(masks)
        return {"strength": scale, "family": families, "family_masks": masks}

    @staticmethod
    def _family_masks_from_request(
        families: tuple[str, ...], env_ids: torch.Tensor, num_envs: int
    ) -> dict[str, torch.Tensor] | None:
        if not families:
            return None
        names = ("planar", "yaw", "global_z", "local_rp")
        masks = {name: torch.zeros(num_envs, dtype=torch.bool, device=env_ids.device) for name in names}
        for row, family in enumerate(families):
            parts = {part.strip() for part in str(family).split("+") if part.strip()}
            for name in names:
                if name in parts:
                    masks[name][env_ids[row]] = True
        return masks

    def _baseline_steps(self, descriptor: FrontRESPerturbationDescriptor) -> int:
        if self.baseline_rollout_steps is not None:
            return max(int(self.baseline_rollout_steps), 0)
        return max(int(descriptor.duration), 0)

    def _zero_action(self) -> torch.Tensor:
        num_envs = int(getattr(self.base_env, "num_envs", getattr(self.command, "num_envs", 1)))
        num_actions = int(getattr(self.base_env, "num_actions", 0) or 0)
        if num_actions <= 0:
            action_space = getattr(self.env, "action_space", None)
            shape = getattr(action_space, "shape", None)
            if shape is None or len(shape) == 0:
                raise AttributeError("cannot infer zero action shape for Stage 1 baseline rollout")
            num_actions = int(shape[-1])
        device = torch.device(getattr(self.command, "device", "cpu"))
        return torch.zeros(num_envs, num_actions, dtype=torch.float32, device=device)

    def _parse_step_result(self, step_result: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not isinstance(step_result, tuple):
            return None, None
        if len(step_result) == 4:
            _, rewards, dones, _ = step_result
            return rewards.detach().float(), dones.detach().bool()
        if len(step_result) == 5:
            _, rewards, terminated, truncated, _ = step_result
            dones = torch.logical_or(terminated.bool(), truncated.bool())
            return rewards.detach().float(), dones
        return None, None

    def _baseline_score(self, env_ids: torch.Tensor, rewards: list[torch.Tensor]) -> torch.Tensor:
        if rewards:
            return torch.stack(rewards, dim=0).mean(dim=0).detach()
        if hasattr(self.command, "_update_metrics"):
            self.command._update_metrics()
        anchor_pos = self._metric_value("error_anchor_pos", env_ids)
        anchor_rot = self._metric_value("error_anchor_rot", env_ids)
        return -(anchor_pos + anchor_rot).detach().float()

    def _metric_value(self, name: str, env_ids: torch.Tensor) -> torch.Tensor:
        value = getattr(self.command, "metrics", {}).get(name)
        if value is None:
            return torch.zeros(env_ids.numel(), dtype=torch.float32, device=env_ids.device)
        return value.index_select(0, env_ids).detach().float()

    def _normalize_env_ids(self, env_ids: Iterable[int] | torch.Tensor) -> torch.Tensor:
        device = torch.device(getattr(self.command, "device", "cpu"))
        if isinstance(env_ids, torch.Tensor):
            ids = env_ids.to(device=device, dtype=torch.long).flatten()
        else:
            ids = torch.tensor(list(env_ids), dtype=torch.long, device=device)
        if ids.numel() == 0:
            raise ValueError("env_ids must be non-empty")
        return ids

    def _trace(self, label: str, **items: Any) -> None:
        if not self.trace:
            return
        lines = ["", "-" * 80, "", f"[frontres_stage1_hook trace] {label}"]
        for key, value in items.items():
            lines.append(f"  {key}: {self._format_trace_value(value)}")
        print("\n".join(lines), flush=True)

    def _format_trace_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._format_trace_value(item) for key, item in value.items()}
        if isinstance(value, torch.Tensor):
            t = value.detach()
            if t.numel() == 0:
                return {"shape": tuple(t.shape), "numel": 0}
            nonzero_frac = float((t.reshape(-1) != 0).float().mean().item())
            if torch.is_floating_point(t):
                finite = bool(torch.isfinite(t).all().item())
                result = {
                    "shape": tuple(t.shape),
                    "device": str(t.device),
                    "finite": finite,
                    "min": float(t.min().item()),
                    "max": float(t.max().item()),
                    "mean": float(t.float().mean().item()),
                    "abs_max": float(t.float().abs().max().item()),
                    "nonzero_frac": nonzero_frac,
                    "requires_grad": bool(t.requires_grad),
                }
            else:
                result = {
                    "shape": tuple(t.shape),
                    "device": str(t.device),
                    "min": int(t.min().item()),
                    "max": int(t.max().item()),
                    "nonzero_frac": nonzero_frac,
                }
            if int(t.numel()) <= 16:
                result["values"] = t.reshape(-1).cpu().tolist()
            return result
        if isinstance(value, (list, tuple)):
            return self._format_sequence_trace(value)
        return value

    def _format_sequence_trace(self, value: list[Any] | tuple[Any, ...]) -> Any:
        count = len(value)
        if count <= self.trace_preview_count:
            return list(value)
        first = value[0] if count else None
        last = value[-1] if count else None
        result = {"count": count, "first": first, "last": last}
        if all(isinstance(item, int) for item in value):
            result.update({"min": min(value), "max": max(value)})
        elif all(isinstance(item, str) for item in value):
            result["unique_count"] = len(set(value))
        elif all(isinstance(item, float) and math.isfinite(item) for item in value):
            result.update({"min": min(value), "max": max(value)})
        else:
            result["type"] = type(first).__name__
        return result


def ensure_frontres_segment_index_reset_hook(
    env: Any,
    *,
    amass_root: str,
    robot_name: str = "robot",
    trace: bool = True,
) -> FrontRESStage1EnvAdapter:
    existing = getattr(env, "_frontres_segment_index_reset_adapter", None)
    if isinstance(existing, FrontRESStage1EnvAdapter):
        return existing
    adapter = FrontRESStage1EnvAdapter(env, amass_root=amass_root, robot_name=robot_name, trace=trace)
    setattr(env, "_frontres_segment_index_reset_adapter", adapter)
    setattr(env, "apply_frontres_segment_index_reset", adapter.apply_frontres_segment_index_reset)
    base_env = getattr(adapter, "base_env", None)
    if base_env is not None and base_env is not env:
        setattr(base_env, "_frontres_segment_index_reset_adapter", adapter)
        setattr(base_env, "apply_frontres_segment_index_reset", adapter.apply_frontres_segment_index_reset)
    return adapter
