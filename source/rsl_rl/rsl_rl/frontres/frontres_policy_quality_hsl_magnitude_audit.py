"""Offline Stage 2 HSL magnitude and checkpoint-lineage audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


SUPERVISED_CONFIG_KEYS = (
    "supervised_rpy_loss_weight",
    "supervised_direction_loss_weight",
    "supervised_valid_loss_weight",
    "supervised_magnitude_loss_weight",
    "supervised_over_loss_weight",
    "supervised_smooth_loss_weight",
    "supervised_harm_loss_weight",
)


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().reshape(-1)
    return {
        "min": float(values.min()),
        "p10": float(torch.quantile(values, 0.10)),
        "median": float(values.median()),
        "p90": float(torch.quantile(values, 0.90)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def _weighted_mean(values: torch.Tensor, sample_weight: torch.Tensor, extra_weight=None):
    weight = sample_weight if extra_weight is None else sample_weight * extra_weight
    return (values * weight).sum() / weight.sum().clamp(min=1e-6)


def supervised_component_audit(
    proposal: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
    harm_weight: torch.Tensor,
    *,
    rpy_weight: float = 1.0,
    valid_weight: float = 4.0,
    magnitude_weight: float = 0.5,
    over_weight: float = 0.2,
    direction_weight: float = 0.03,
    harm_loss_weight: float = 1.0,
) -> dict[str, Any]:
    """Replay active scalar losses and their proposal-gradient contribution.

    Status: active offline quality audit, not a training loss owner.
    Upstream: Q2-B matched HSL action/target/weight evidence.
    Downstream: Q2-C evidence report only; no optimizer or runner mutation.
    Evidence: Q-E18 contract-confirmed and held-out-bank-confirmed.
    Gap: complete Stage 2 training distribution and checkpoint config lineage.
    """

    # B5: QUALITY-ACTION-01 拆分方向和尺度梯度, 定位统一 grad clip 前的竞争关系.

    proposal = proposal.detach().float().reshape(-1, 6).requires_grad_(True)
    target = target.detach().float().reshape(-1, 6)
    sample_weight = sample_weight.detach().float().reshape(-1).clamp(min=0.0)
    harm_weight = harm_weight.detach().float().reshape(-1).clamp(min=0.0)
    if not (proposal.shape[0] == target.shape[0] == sample_weight.numel() == harm_weight.numel()):
        raise ValueError("proposal/target/sample_weight/harm_weight row counts must match")

    valid = target.norm(dim=-1) > 1e-4
    pos_valid = target[:, :3].norm(dim=-1) > 1e-4
    rpy_valid = target[:, 3:6].norm(dim=-1) > 1e-4
    pos_row_weight = torch.where(pos_valid, valid_weight, 1.0)
    rpy_row_weight = torch.where(rpy_valid, valid_weight, 1.0)
    pos_row_weight = pos_row_weight / pos_row_weight.mean().clamp(min=1e-6)
    rpy_row_weight = rpy_row_weight / rpy_row_weight.mean().clamp(min=1e-6)

    pos = _weighted_mean(F.huber_loss(proposal[:, :3], target[:, :3], reduction="none").mean(-1), sample_weight, pos_row_weight)
    rot = _weighted_mean(F.huber_loss(proposal[:, 3:6], target[:, 3:6], reduction="none").mean(-1), sample_weight, rpy_row_weight)
    magnitude = _weighted_mean(
        F.huber_loss(proposal.norm(dim=-1), target.norm(dim=-1), reduction="none"),
        sample_weight,
        valid.float(),
    )
    over = _weighted_mean(
        torch.relu(proposal.norm(dim=-1) - target.norm(dim=-1)).square(),
        sample_weight,
        valid.float(),
    )
    direction_pos = torch.zeros((), dtype=proposal.dtype)
    direction_rpy = torch.zeros((), dtype=proposal.dtype)
    if pos_valid.any():
        direction_pos = _weighted_mean(
            1.0 - F.cosine_similarity(proposal[:, :3], target[:, :3], dim=-1),
            sample_weight,
            pos_valid.float(),
        )
    if rpy_valid.any():
        direction_rpy = _weighted_mean(
            1.0 - F.cosine_similarity(proposal[:, 3:6], target[:, 3:6], dim=-1),
            sample_weight,
            rpy_valid.float(),
        )
    direction = direction_pos + direction_rpy
    harm = _weighted_mean(proposal.square().mean(dim=-1), sample_weight, harm_weight)

    weighted = {
        "base_pos": pos,
        "base_rot": rpy_weight * rot,
        "magnitude": magnitude_weight * magnitude,
        "over": over_weight * over,
        "direction_pos": direction_weight * direction_pos,
        "direction_rpy": direction_weight * direction_rpy,
        "harm": harm_loss_weight * harm,
    }
    raw = {
        "base_pos": pos,
        "base_rot": rot,
        "magnitude": magnitude,
        "over": over,
        "direction_pos": direction_pos,
        "direction_rpy": direction_rpy,
        "harm": harm,
    }
    components: dict[str, Any] = {}
    for name, value in weighted.items():
        grad = (
            torch.autograd.grad(value, proposal, retain_graph=True, allow_unused=True)[0]
            if value.requires_grad
            else None
        )
        components[name] = {
            "raw_loss": float(raw[name].detach()),
            "weighted_loss": float(value.detach()),
            "proposal_grad_l2": 0.0 if grad is None else float(grad.norm()),
        }

    action_norm = proposal.detach().norm(dim=-1)
    target_norm = target.norm(dim=-1)
    ratio = action_norm / target_norm.clamp(min=1e-8)
    grad_norms = {name: values["proposal_grad_l2"] for name, values in components.items()}
    scale_grad = grad_norms["magnitude"] + grad_norms["over"]
    return {
        "row_count": int(proposal.shape[0]),
        "valid_count": int(valid.sum()),
        "action_norm": _quantiles(action_norm),
        "target_norm": _quantiles(target_norm),
        "action_target_norm_ratio": _quantiles(ratio),
        "components": components,
        "gradient_competition": {
            "largest_component": max(grad_norms, key=grad_norms.get),
            "direction_to_scale_grad_ratio": (grad_norms["direction_pos"] + grad_norms["direction_rpy"])
            / max(scale_grad, 1e-12),
        },
        "weighted_total_without_smooth": float(sum(weighted.values()).detach()),
        "smooth_component": "unavailable_without_training_batch_indices_and_storage_num_envs",
    }


def _extract_q2_tensors(payload: dict[str, Any]):
    actions, targets, sample_weights, harm_weights = [], [], [], []
    checkpoint_ids = set()
    for row in payload.get("rows", []):
        hsl = row["routes"]["hsl"]
        supervision = hsl["execution"]["hsl_supervision"]
        action = torch.as_tensor(hsl["actions"], dtype=torch.float32)
        target = torch.as_tensor(supervision["targets"], dtype=torch.float32)
        if action.ndim == 3 and target.ndim == 3:
            action = action[:, : target.shape[1], :]
        action = action.reshape(-1, 6)
        target = target.reshape(-1, 6)
        sample = torch.as_tensor(supervision["sample_weights"], dtype=torch.float32).reshape(-1)
        harm = torch.as_tensor(supervision["harm_weights"], dtype=torch.float32).reshape(-1)
        if action.shape[0] != target.shape[0]:
            raise ValueError("HSL action rows do not match canonical target rows")
        actions.append(action)
        targets.append(target)
        sample_weights.append(sample)
        harm_weights.append(harm)
        checkpoint_ids.add(str(hsl.get("checkpoint_identity", "")))
    if not actions:
        raise ValueError("Q2 result contains no HSL rows")
    return (
        torch.cat(actions),
        torch.cat(targets),
        torch.cat(sample_weights),
        torch.cat(harm_weights),
        sorted(checkpoint_ids),
    )


def checkpoint_lineage_audit(path: str | None, checkpoint_ids: list[str]) -> dict[str, Any]:
    """Report whether the artifact itself proves its Stage 2 training semantics."""

    result: dict[str, Any] = {
        "runtime_checkpoint_identities": checkpoint_ids,
        "local_checkpoint_path": path,
        "local_checkpoint_available": bool(path and Path(path).is_file()),
        "required_training_config_keys": list(SUPERVISED_CONFIG_KEYS),
    }
    if not result["local_checkpoint_available"]:
        result.update({"status": "unconfirmed", "reason": "checkpoint_not_available_locally"})
        return result
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    saved_cfg = payload.get("frontres_supervised_config")
    result.update(
        {
            "iteration": payload.get("iter"),
            "top_level_keys": sorted(payload.keys()),
            "source_checkpoint_identity": payload.get("frontres_source_checkpoint_identity"),
            "training_objective": payload.get("frontres_training_objective"),
            "saved_supervised_config": saved_cfg,
            "missing_supervised_config_keys": list(SUPERVISED_CONFIG_KEYS)
            if not isinstance(saved_cfg, dict)
            else [key for key in SUPERVISED_CONFIG_KEYS if key not in saved_cfg],
        }
    )
    result["status"] = "confirmed" if not result["missing_supervised_config_keys"] and result["source_checkpoint_identity"] else "lineage_incomplete"
    return result


def build_audit(result_path: str, checkpoint_path: str | None = None) -> dict[str, Any]:
    payload = json.loads(Path(result_path).read_text())
    action, target, sample_weight, harm_weight, checkpoint_ids = _extract_q2_tensors(payload)
    return {
        "schema_version": "frontres_policy_quality_hsl_magnitude_audit_v1",
        "evidence_scope": "held_out_q2b_bank_not_full_stage2_training_distribution",
        "checkpoint_lineage": checkpoint_lineage_audit(checkpoint_path, checkpoint_ids),
        "effective_config_assumption": {
            "supervised_rpy_loss_weight": 1.0,
            "supervised_direction_loss_weight": 0.03,
            "supervised_valid_loss_weight": 4.0,
            "supervised_magnitude_loss_weight": 0.5,
            "supervised_over_loss_weight": 0.2,
            "supervised_harm_loss_weight": 1.0,
        },
        "loss_gradient_replay": supervised_component_audit(action, target, sample_weight, harm_weight),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json")
    parser.add_argument("output_json")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    audit = build_audit(args.result_json, args.checkpoint)
    Path(args.output_json).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print("PASS: Stage 2 HSL magnitude audit artifact generated.")


if __name__ == "__main__":
    main()
