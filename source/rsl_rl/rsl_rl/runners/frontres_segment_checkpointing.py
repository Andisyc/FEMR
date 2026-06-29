from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class FrontRESSegmentCheckpointConfig:
    hsl_init_enabled: bool = True
    is_full_resume: bool = False
    resume_optimizer: bool = False
    load_normalizers: bool = True
    persist_sampler_state: bool = True
    persist_dataset_cache_metadata: bool = True


@dataclass(frozen=True)
class FrontRESSegmentCheckpointResult:
    copied_actor_keys: tuple[str, ...]
    ignored_acceptance_keys: tuple[str, ...]
    optimizer_loaded: bool
    optimizer_reset: bool
    normalizer_keys_loaded: tuple[str, ...]
    sampler_state_loaded: bool
    dataset_cache_metadata_loaded: bool


def build_frontres_segment_checkpoint_payload(
    runner: Any,
    infos: Any | None = None,
    cfg: FrontRESSegmentCheckpointConfig | None = None,
) -> dict[str, Any]:
    cfg = FrontRESSegmentCheckpointConfig() if cfg is None else cfg
    policy = runner.alg.policy
    model_state_dict: dict[str, Any] = {}
    if getattr(policy, "residual_actor", None) is not None:
        model_state_dict["residual_actor"] = policy.residual_actor.state_dict()
    if getattr(policy, "critic", None) is not None:
        model_state_dict["critic"] = policy.critic.state_dict()
    if hasattr(policy, "std"):
        model_state_dict["std"] = policy.std.detach().clone()
    elif hasattr(policy, "log_std"):
        model_state_dict["log_std"] = policy.log_std.detach().clone()

    payload = {
        "model_state_dict": model_state_dict,
        "iter": int(getattr(runner, "current_learning_iteration", 0)),
        "infos": infos,
        "frontres_stage": "stage3_segment_hrl",
        "frontres_training_objective": "segment_replay_hrl",
    }
    optimizer = getattr(runner.alg, "optimizer", None)
    if optimizer is not None and hasattr(optimizer, "state_dict"):
        payload["optimizer_state_dict"] = optimizer.state_dict()

    if cfg.load_normalizers:
        _save_normalizer(payload, "obs_norm_state_dict", getattr(runner, "obs_normalizer", None))
        _save_normalizer(
            payload,
            "privileged_obs_norm_state_dict",
            getattr(runner, "privileged_obs_normalizer", None),
        )
    if cfg.persist_sampler_state:
        sampler = _find_attr(runner, "_frontres_segment_sampler", "frontres_segment_sampler")
        if sampler is not None and hasattr(sampler, "state_dict"):
            payload["frontres_segment_sampler_state_dict"] = sampler.state_dict()
    if cfg.persist_dataset_cache_metadata:
        dataset = _find_attr(runner, "_frontres_segment_dataset", "frontres_segment_dataset")
        metadata = _dataset_cache_metadata(dataset)
        if metadata is not None:
            payload["frontres_segment_dataset_cache_metadata"] = metadata
    return payload


def restore_frontres_segment_checkpoint(
    runner: Any,
    checkpoint: dict[str, Any],
    cfg: FrontRESSegmentCheckpointConfig | None = None,
) -> FrontRESSegmentCheckpointResult:
    cfg = FrontRESSegmentCheckpointConfig() if cfg is None else cfg
    policy = runner.alg.policy
    model_state = checkpoint.get("model_state_dict", checkpoint)
    repair_actor = getattr(policy, "residual_actor", getattr(policy, "repair_actor", None))
    if repair_actor is None:
        raise AttributeError("Stage 3 policy must expose residual_actor or repair_actor")

    if cfg.is_full_resume:
        copied = _load_exact_repair_actor(repair_actor, model_state)
    elif cfg.hsl_init_enabled:
        copied = initialize_repair_actor_from_hsl_checkpoint(repair_actor, model_state)
    else:
        copied = tuple()

    ignored_acceptance = tuple(sorted(key for key in _flatten_model_keys(model_state) if "acceptance" in key))
    normalizers = _restore_normalizers(runner, checkpoint) if cfg.load_normalizers else tuple()
    optimizer_loaded = False
    optimizer_reset = True
    optimizer = getattr(runner.alg, "optimizer", None)
    if cfg.resume_optimizer and optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer_loaded = True
        optimizer_reset = False

    sampler_loaded = _restore_sampler_state(runner, checkpoint)
    dataset_loaded = _restore_dataset_cache_metadata(runner, checkpoint)
    return FrontRESSegmentCheckpointResult(
        copied_actor_keys=copied,
        ignored_acceptance_keys=ignored_acceptance,
        optimizer_loaded=optimizer_loaded,
        optimizer_reset=optimizer_reset,
        normalizer_keys_loaded=normalizers,
        sampler_state_loaded=sampler_loaded,
        dataset_cache_metadata_loaded=dataset_loaded,
    )


def initialize_repair_actor_from_hsl_checkpoint(repair_actor: Any, model_state: dict[str, Any]) -> tuple[str, ...]:
    source = _extract_hsl_actor_state(model_state)
    target = repair_actor.state_dict()
    mapped = _map_source_to_target(target, source)
    if not mapped:
        raise RuntimeError("could not map HSL checkpoint actor into 6D Stage 3 repair actor")
    new_state = dict(target)
    new_state.update(mapped)
    repair_actor.load_state_dict(new_state, strict=True)
    return tuple(sorted(mapped))


def _load_exact_repair_actor(repair_actor: Any, model_state: dict[str, Any]) -> tuple[str, ...]:
    source = _extract_hsl_actor_state(model_state)
    target = repair_actor.state_dict()
    exact = {key: source[key].detach().clone() for key in target if key in source and source[key].shape == target[key].shape}
    if len(exact) != len(target):
        raise RuntimeError("full Stage 3 resume requires an exact 6D repair actor state")
    repair_actor.load_state_dict(exact, strict=True)
    return tuple(sorted(exact))


def _extract_hsl_actor_state(model_state: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "residual_actor" in model_state and isinstance(model_state["residual_actor"], dict):
        return model_state["residual_actor"]
    student = {
        key.removeprefix("student."): value
        for key, value in model_state.items()
        if isinstance(key, str) and key.startswith("student.")
    }
    if student:
        return student
    return {key: value for key, value in model_state.items() if isinstance(value, torch.Tensor)}


def _map_source_to_target(target: dict[str, torch.Tensor], source: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    exact = {key: source[key].detach().clone() for key in target if key in source and source[key].shape == target[key].shape}
    if len(exact) == len(target):
        return exact

    if any(key.startswith("trunk.") for key in source) and any(key.startswith("proposal_head.") for key in source):
        numeric_weights = _numeric_module_keys(target, ".weight")
        numeric_biases = _numeric_module_keys(target, ".bias")
        if not numeric_weights:
            return {}
        last_weight = numeric_weights[-1]
        last_bias = last_weight.replace(".weight", ".bias")
        mapped: dict[str, torch.Tensor] = {}
        for key, value in target.items():
            if key == last_weight:
                src = source.get("proposal_head.weight")
            elif key == last_bias:
                src = source.get("proposal_head.bias")
            else:
                src = source.get(f"trunk.{key}")
            if src is None or src.shape != value.shape:
                return {}
            mapped[key] = src.detach().clone()
        return mapped
    return exact


def _numeric_module_keys(state_dict: dict[str, Any], suffix: str) -> list[str]:
    keys = [key for key in state_dict if key.endswith(suffix) and key.split(".", 1)[0].isdigit()]
    return sorted(keys, key=lambda key: int(key.split(".", 1)[0]))


def _restore_normalizers(runner: Any, checkpoint: dict[str, Any]) -> tuple[str, ...]:
    loaded: list[str] = []
    for key, attr in (
        ("obs_norm_state_dict", "obs_normalizer"),
        ("privileged_obs_norm_state_dict", "privileged_obs_normalizer"),
    ):
        normalizer = getattr(runner, attr, None)
        if normalizer is not None and key in checkpoint and hasattr(normalizer, "load_state_dict"):
            normalizer.load_state_dict(checkpoint[key])
            loaded.append(key)
    return tuple(loaded)


def _restore_sampler_state(runner: Any, checkpoint: dict[str, Any]) -> bool:
    sampler = _find_attr(runner, "_frontres_segment_sampler", "frontres_segment_sampler")
    state = checkpoint.get("frontres_segment_sampler_state_dict")
    if sampler is None or state is None or not hasattr(sampler, "load_state_dict"):
        return False
    sampler.load_state_dict(state)
    return True


def _restore_dataset_cache_metadata(runner: Any, checkpoint: dict[str, Any]) -> bool:
    dataset = _find_attr(runner, "_frontres_segment_dataset", "frontres_segment_dataset")
    metadata = checkpoint.get("frontres_segment_dataset_cache_metadata")
    if dataset is None or metadata is None:
        return False
    if hasattr(dataset, "load_cache_metadata"):
        dataset.load_cache_metadata(metadata)
        return True
    if hasattr(dataset, "load_state_dict"):
        dataset.load_state_dict({"cache_metadata": metadata})
        return True
    return False


def _save_normalizer(payload: dict[str, Any], key: str, normalizer: Any) -> None:
    if normalizer is not None and hasattr(normalizer, "state_dict"):
        payload[key] = normalizer.state_dict()


def _dataset_cache_metadata(dataset: Any) -> Any | None:
    if dataset is None:
        return None
    if hasattr(dataset, "cache_metadata"):
        value = dataset.cache_metadata
        return value() if callable(value) else value
    if hasattr(dataset, "state_dict"):
        state = dataset.state_dict()
        if isinstance(state, dict):
            return state.get("cache_metadata")
    return None


def _find_attr(obj: Any, *names: str) -> Any | None:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    connector = getattr(obj, "_frontres_segment_replay_connector", None)
    for name in names:
        if connector is not None and hasattr(connector, name.replace("_frontres_segment_", "")):
            return getattr(connector, name.replace("_frontres_segment_", ""))
    return None


def _flatten_model_keys(model_state: dict[str, Any]) -> tuple[str, ...]:
    keys: list[str] = []
    for key, value in model_state.items():
        if isinstance(value, dict):
            keys.extend(f"{key}.{subkey}" for subkey in value)
        else:
            keys.append(str(key))
    return tuple(keys)
