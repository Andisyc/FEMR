#!/usr/bin/env python3
"""CPU-only G5-S2A contract for strict v015 quality artifact identity."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[4]
RSL_ROOT = ROOT / "source" / "rsl_rl" / "rsl_rl"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _owners():
    checkpoint_contract = _load(
        "frontres_v015_quality_checkpoint_fixture",
        RSL_ROOT / "tests" / "frontres_v015_checkpoint_resume_contract.py",
    )
    _layout, checkpointing, _policy = checkpoint_contract._load_owners()
    frontres = types.ModuleType("rsl_rl.frontres")
    frontres.__path__ = []
    runners = types.ModuleType("rsl_rl.runners")
    runners.__path__ = []
    sys.modules["rsl_rl.frontres"] = frontres
    sys.modules["rsl_rl.runners"] = runners
    sys.modules["rsl_rl.runners.frontres_checkpointing"] = checkpointing
    manifest = _load(
        "rsl_rl.frontres.frontres_policy_quality_manifest",
        RSL_ROOT / "frontres" / "frontres_policy_quality_manifest.py",
    )
    quality = _load(
        "frontres_v015_quality_request_contract",
        RSL_ROOT / "runners" / "frontres_policy_quality_eval.py",
    )
    return checkpointing, manifest, quality


def _layout() -> dict[str, object]:
    return {
        "layout_version": "frontres-v015-future-intent-q29-v1",
        "future_offsets": (1, 2),
        "intent_dim": 29,
        "actor_tail_dim": 58,
        "environment_obs_dim": 870,
        "current_frontres_prefix_dim": 100,
        "actor_dim": 928,
        "prefix_dim": 158,
        "gmt_dim": 770,
    }


def _normalizer(dim: int) -> dict[str, torch.Tensor]:
    return {
        "_mean": torch.zeros(1, dim),
        "_var": torch.ones(1, dim),
        "_std": torch.ones(1, dim),
        "count": torch.tensor(4.0),
    }


def _actor_state() -> dict[str, torch.Tensor]:
    return {
        "0.weight": torch.arange(6 * 158, dtype=torch.float32).reshape(6, 158) / 1000.0,
        "0.bias": torch.arange(6, dtype=torch.float32) / 100.0,
    }


def _hsl_payload(checkpointing) -> dict[str, object]:
    actor = _actor_state()
    prefix = _normalizer(158)
    distribution = torch.full((6,), 0.1)
    payload_identity = {
        "top_level_keys": (
            "frontres_prefix_norm_state_dict",
            "frontres_v015_hsl_checkpoint_identity",
            "model_state_dict",
        ),
        "model_keys": ("residual_actor", "std"),
        "residual_actor_fingerprint": checkpointing._v015_state_dict_fingerprint(
            actor, label="fixture HSL actor"
        ),
        "distribution_key": "std",
        "distribution_fingerprint": checkpointing._v015_tensor_fingerprint(distribution),
        "prefix_normalizer_keys": ("_mean", "_std", "_var", "count"),
        "prefix_normalizer_fingerprint": checkpointing._v015_state_dict_fingerprint(
            prefix, label="fixture HSL prefix"
        ),
    }
    return {
        "frontres_v015_hsl_checkpoint_identity": {
            "format": "frontres-v015-hsl-proposal-v1",
            "method_contract_id": "FRS-METHOD-v015",
            "training_contract_id": "FRS-TRAIN-v007",
            "objective": "proposal_only_current_antidr_delta_se3",
            "future_intent_layout": _layout(),
            "action": {"kind": "delta_se3", "dim": 6},
            "gmt": {
                "checkpoint_sha256": "a" * 64,
                "normalizer_dim": 770,
                "normalizer_fingerprint": "b" * 64,
            },
            "payload": payload_identity,
        },
        "model_state_dict": {"residual_actor": actor, "std": distribution},
        "frontres_prefix_norm_state_dict": prefix,
    }


def _stage3_payload(checkpointing, *, transaction_state: str = "idle") -> dict[str, object]:
    actor = _actor_state()
    obs_norm = _normalizer(928)
    prefix_fingerprint = checkpointing._v015_tensor_fingerprint(
        obs_norm["_mean"][..., :158],
        obs_norm["_std"][..., :158],
    )
    transaction: dict[str, object] = {"state": transaction_state}
    return {
        "frontres_v015_checkpoint_identity": {
            "format": "frontres-v015-checkpoint-v3",
            "method_contract_id": "FRS-METHOD-v015",
            "training_contract_id": "FRS-TRAIN-v008",
            "gain_contract_id": "FRS-GAIN-v004",
            "ppo_contract_id": "FRS-PPO-v003",
            "future_intent_layout": _layout(),
            "normalizer": {
                "mode": "empirical_prefix_plus_frozen_gmt",
                "prefix_layout_version": "frontres-v015-future-intent-q29-v1",
                "prefix_dim": 158,
                "combined_dim": 928,
                "prefix_stats_fingerprint": prefix_fingerprint,
            },
            "grouped_loss": {
                "advantage_normalization": "grouped_scale_only",
                "candidate_layout_version": "frontres-v015-local-scenario-v1",
                "policy_rows_per_attempt": 1,
            },
            "transaction": transaction,
            "warmup": {
                "critic_warmup_iterations": 200,
                "actor_warmup_iterations": 500,
                "iteration": 1,
                "phase": "critic_only",
            },
        },
        "model_state_dict": {"residual_actor": actor, "std": torch.full((6,), 0.2)},
        "obs_norm_state_dict": obs_norm,
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "iter": 1,
        "infos": None,
    }


def _manifest_payload() -> dict[str, object]:
    return {
        "schema_version": "frontres-v015-policy-quality-manifest-v1",
        "method_contract_id": "FRS-METHOD-v015",
        "training_contract_id": "FRS-TRAIN-v008",
        "gain_contract_id": "FRS-GAIN-v004",
        "ppo_contract_id": "FRS-PPO-v003",
        "future_intent_layout_version": "frontres-v015-future-intent-q29-v1",
        "future_offsets": [1, 2],
        "raw_observation_dim": 870,
        "combined_observation_dim": 928,
        "actor_input_dim": 158,
        "gmt_suffix_dim": 770,
        "action_kind": "delta_se3",
        "action_dim": 6,
        "environment_revision": "isaaclab-mosaic-v015",
        "config_revision": "frontres-stage3-v015",
        "evaluator_version": "frontres-v015-one-action-k-quality-v1",
        "items": [
            {
                "item_id": "motion-a-k8",
                "motion_id": "KIT/example.npz",
                "start_frame": 12,
                "perturbation_family": "local_rp",
                "perturbation_parameters": [["roll_rad", 0.1]],
                "effective_horizon_k": 8,
                "seed": 7,
            }
        ],
    }


def _legacy_manifest_payload() -> dict[str, object]:
    payload = _manifest_payload()
    return {
        "schema_version": "frontres_policy_quality_manifest_v1",
        "environment_revision": payload["environment_revision"],
        "config_revision": payload["config_revision"],
        "evaluator_version": "policy-quality-v1",
        "items": payload["items"],
    }


def _expect_reject(fn, fragment: str) -> None:
    try:
        fn()
    except (RuntimeError, ValueError) as exc:
        assert fragment.lower() in str(exc).lower(), str(exc)
        return
    raise AssertionError(f"expected strict rejection containing {fragment!r}")


def test_strict_v015_quality_identity_and_tamper_rejection() -> None:
    checkpointing, manifest_module, quality = _owners()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest_path = root / "manifest.json"
        hsl_path = root / "hsl.pt"
        policy_path = root / "policy.pt"
        result_path = root / "result.json"
        manifest_path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")
        torch.save(_hsl_payload(checkpointing), hsl_path)
        torch.save(_stage3_payload(checkpointing), policy_path)

        request = quality.build_frontres_v015_policy_quality_eval_request(
            manifest_path=str(manifest_path),
            hsl_checkpoint_path=str(hsl_path),
            policy_checkpoint_path=str(policy_path),
            result_path=str(result_path),
        )
        assert isinstance(request.manifest, manifest_module.FrontRESV015PolicyQualityManifest)
        assert request.manifest.future_offsets == (1, 2)
        assert request.manifest.combined_observation_dim == 928
        assert request.manifest.actor_input_dim == 158
        assert request.manifest.gmt_suffix_dim == 770
        assert request.manifest.action_dim == 6
        assert request.hsl_checkpoint.route == "hsl"
        assert request.hsl_checkpoint.format == "frontres-v015-hsl-proposal-v1"
        assert request.hsl_checkpoint.normalizer_key == "frontres_prefix_norm_state_dict"
        assert request.policy_checkpoint.route == "policy"
        assert request.policy_checkpoint.format == "frontres-v015-checkpoint-v3"
        assert request.policy_checkpoint.normalizer_key == "obs_norm_state_dict"
        assert len(request.manifest_file_sha256) == 64
        assert len(request.hsl_checkpoint.file_sha256) == 64
        assert len(request.policy_checkpoint.file_sha256) == 64
        assert request.hsl_checkpoint.file_sha256 != request.policy_checkpoint.file_sha256

        tampered_hsl = copy.deepcopy(_hsl_payload(checkpointing))
        tampered_hsl["model_state_dict"]["residual_actor"]["0.weight"][0, 0] += 1.0
        tampered_hsl_path = root / "tampered_hsl.pt"
        torch.save(tampered_hsl, tampered_hsl_path)
        _expect_reject(
            lambda: checkpointing.inspect_frontres_v015_quality_checkpoint(tampered_hsl_path, route="hsl"),
            "fingerprint",
        )
        _expect_reject(
            lambda: checkpointing.inspect_frontres_v015_quality_checkpoint(policy_path, route="hsl"),
            "HSL",
        )

        partial_path = root / "partial.pt"
        torch.save(_stage3_payload(checkpointing, transaction_state="sealed"), partial_path)
        _expect_reject(
            lambda: checkpointing.inspect_frontres_v015_quality_checkpoint(partial_path, route="policy"),
            "transaction",
        )

        tampered_policy = _stage3_payload(checkpointing)
        tampered_policy["obs_norm_state_dict"]["_mean"][0, 157] += 1.0
        tampered_policy_path = root / "tampered_policy.pt"
        torch.save(tampered_policy, tampered_policy_path)
        _expect_reject(
            lambda: checkpointing.inspect_frontres_v015_quality_checkpoint(
                tampered_policy_path, route="policy"
            ),
            "fingerprint",
        )

        legacy_manifest = dict(_manifest_payload())
        legacy_manifest["schema_version"] = "frontres_policy_quality_manifest_v1"
        legacy_path = root / "legacy.json"
        legacy_path.write_text(json.dumps(legacy_manifest), encoding="utf-8")
        _expect_reject(
            lambda: quality.build_frontres_v015_policy_quality_eval_request(
                manifest_path=str(legacy_path),
                hsl_checkpoint_path=str(hsl_path),
                policy_checkpoint_path=str(policy_path),
                result_path=str(result_path),
            ),
            "schema",
        )

        v002_manifest = dict(_manifest_payload())
        v002_manifest["gain_contract_id"] = "FRS-GAIN-v002"
        v002_path = root / "v002.json"
        v002_path.write_text(json.dumps(v002_manifest), encoding="utf-8")
        _expect_reject(
            lambda: quality.build_frontres_v015_policy_quality_eval_request(
                manifest_path=str(v002_path),
                hsl_checkpoint_path=str(hsl_path),
                policy_checkpoint_path=str(policy_path),
                result_path=str(result_path),
            ),
            "contract",
        )

        legacy_exact_path = root / "legacy_exact.json"
        legacy_exact_path.write_text(json.dumps(_legacy_manifest_payload()), encoding="utf-8")
        legacy_calls: list[object] = []
        active_runner = SimpleNamespace(
            alg=SimpleNamespace(frontres_v015_formal_transaction_enabled=True),
            _frontres_policy_quality_manifest_executor=lambda request: legacy_calls.append(request),
        )
        _expect_reject(
            lambda: quality.run_frontres_policy_quality_eval(
                active_runner,
                manifest_path=str(legacy_exact_path),
                hsl_checkpoint_path=str(hsl_path),
                policy_checkpoint_path=str(policy_path),
                result_path=str(result_path),
            ),
            "v015",
        )
        assert legacy_calls == []


def test_fixed_v015_heldout_manifest() -> None:
    _checkpointing, manifest_module, _quality = _owners()
    path = ROOT / "note" / "testing" / "manifests" / "frontres_v015_policy_quality_heldout_v1.json"
    manifest = manifest_module.FrontRESV015PolicyQualityManifest.from_json(
        path.read_text(encoding="utf-8")
    )
    assert manifest.schema_version == "frontres-v015-policy-quality-manifest-v1"
    assert len(manifest.items) == 16
    assert manifest.future_offsets == (1, 2)
    assert (manifest.combined_observation_dim, manifest.actor_input_dim, manifest.gmt_suffix_dim) == (
        928,
        158,
        770,
    )
    assert all(item.perturbation_family == "local_rp" for item in manifest.items)
    assert all(item.effective_horizon_k == 8 for item in manifest.items)
    assert len({item.comparison_signature for item in manifest.items}) == 16


if __name__ == "__main__":
    test_strict_v015_quality_identity_and_tamper_rejection()
    test_fixed_v015_heldout_manifest()
    print("frontres_v015_policy_quality_identity_contract: ok", flush=True)
