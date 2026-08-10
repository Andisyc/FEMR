"""Explicit compatibility owner for superseded pre-v7 FrontRES checkpoints.

The active HSL-v2 and Stage3 checkpoint-v15 paths never use these payload
builders. Remove this module when pre-v7 initialization/resume is retired.
"""

from __future__ import annotations

from typing import Any, Mapping


_LEGACY_GAIN_CONFIG_FIELDS = (
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


def frontres_legacy_gain_config_payload(cfg: Any) -> dict[str, object]:
    values = {}
    for serialized_name, cfg_name, default in _LEGACY_GAIN_CONFIG_FIELDS:
        value = cfg.get(cfg_name, default) if isinstance(cfg, Mapping) else getattr(cfg, cfg_name, default)
        values[serialized_name] = float(value)
    return {"contract_id": "FRS-GAIN-v002", "values": values}


def validate_frontres_legacy_gain_config_resume(
    runner: Any,
    checkpoint: Mapping[str, Any],
    *,
    is_full_resume: bool,
) -> None:
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
    expected = frontres_legacy_gain_config_payload(getattr(runner, "cfg", None))
    if checkpoint_config != expected:
        raise RuntimeError(
            "FrontRES Gain config mismatch on resume: "
            f"checkpoint={checkpoint_config!r} current={expected!r}"
        )
    print("[Runner] Verified FRS-GAIN-v002 config identity on checkpoint resume.", flush=True)


def reject_legacy_frontres_hsl_checkpoint(runner: Any, checkpoint: Mapping[str, Any]) -> None:
    layout = getattr(runner, "_frontres_future_intent_layout", None)
    context_dim = int(getattr(runner, "_frontres_future_intent_actor_context_dim", 0) or 0)
    if layout is None and context_dim <= 0:
        return
    if bool(checkpoint.get("frontres_warmup_complete", False)):
        raise RuntimeError(
            "FRS-TRAIN-v007 rejects a legacy HSL warmup checkpoint on the v015 q29 actor layout; "
            "a separately authorized persistence step must define a new checkpoint identity"
        )
