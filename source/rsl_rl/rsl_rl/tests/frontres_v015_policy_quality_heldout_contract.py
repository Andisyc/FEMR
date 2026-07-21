#!/usr/bin/env python3
"""CPU-only G5-S2B contract for v015 Repair/Noisy held-out quality."""

from __future__ import annotations

import importlib.util
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace

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
    identity_contract = _load(
        "frontres_v015_quality_identity_fixture",
        RSL_ROOT / "tests" / "frontres_v015_policy_quality_identity_contract.py",
    )
    checkpointing, manifest, quality = identity_contract._owners()
    storage = _load(
        "rsl_rl.frontres.frontres_segment_storage",
        RSL_ROOT / "frontres" / "frontres_segment_storage.py",
    )
    gain = _load(
        "rsl_rl.frontres.frontres_gain",
        RSL_ROOT / "frontres" / "frontres_gain.py",
    )
    return identity_contract, checkpointing, manifest, quality, storage, gain


def _evidence(storage, *, route: str):
    action_scale = {"zero": 0.0, "hsl": 0.1, "policy": 0.2}[route]
    repaired_q29 = {"zero": 1.0, "hsl": 0.5, "policy": 0.2}[route]
    repaired_survival = {"zero": 1.0, "hsl": 2.0, "policy": 2.0}[route]
    intent = torch.zeros(2, 3, 29)
    continuation = torch.arange(2 * 65, dtype=torch.float32).reshape(2, 1, 65).repeat(1, 2, 1)
    result = storage.FrontRESV015OneActionKEvidence(
        policy_observations=torch.full((1, 928), action_scale),
        policy_privileged_observations=torch.full((1, 289), action_scale),
        policy_actions=torch.full((1, 6), action_scale),
        policy_log_probs=torch.zeros(1),
        policy_values=torch.zeros(1),
        policy_means=torch.full((1, 6), action_scale),
        policy_sigmas=torch.ones(1, 6),
        policy_row_indices=torch.tensor([0]),
        t_env_actions=torch.zeros(2, 29),
        continuation=continuation,
        continuation_valid_mask=torch.ones(2, 2, dtype=torch.bool),
        frozen_gmt_env_actions=torch.zeros(2, 2, 29),
        actor_forward_count=1,
        later_femr_action_count=0,
        horizon_k=torch.tensor([2, 2]),
        scenario_ids=("scenario-heldout-a", "scenario-heldout-a"),
        noisy_segment_hashes=("noisy-hash-a", "noisy-hash-a"),
        x_t_identities=("x-t-a", "x-t-a"),
        roles=("repair", "noisy"),
        intent_q29=intent,
        intent_q29_provenance=("deployment_noisy_q29", "deployment_noisy_q29"),
        intent_q29_source=("heldout-deployment-q29", "heldout-deployment-q29"),
        executed_q29_t=torch.stack(
            (torch.full((29,), repaired_q29), torch.ones(29)),
            dim=0,
        ),
        executed_q29_t_valid_mask=torch.ones(2, dtype=torch.bool),
        done_any=torch.zeros(2, dtype=torch.bool),
        survival_steps=torch.tensor([repaired_survival, 1.0]),
    )
    result.validate()
    return result


def _expect_reject(fn, fragment: str) -> None:
    try:
        fn()
    except (RuntimeError, ValueError) as exc:
        assert fragment.lower() in str(exc).lower(), str(exc)
        return
    raise AssertionError(f"expected strict rejection containing {fragment!r}")


def test_v015_repair_noisy_one_action_k_atomic_quality() -> None:
    identity, checkpointing, _manifest, quality, storage, _gain = _owners()
    assert callable(getattr(quality, "build_frontres_v015_policy_quality_owner_bundle", None))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest_path = root / "manifest.json"
        hsl_path = root / "hsl.pt"
        policy_path = root / "policy.pt"
        result_path = root / "quality.json"
        manifest_path.write_text(json.dumps(identity._manifest_payload()), encoding="utf-8")
        torch.save(identity._hsl_payload(checkpointing), hsl_path)
        torch.save(identity._stage3_payload(checkpointing), policy_path)

        training_state = {"optimizer_steps": 7, "sampler_cursor": 11, "warmup": "disabled"}
        calls: list[tuple[str, str]] = []
        checkpoint_by_route = {
            "zero": "zero",
            "hsl": checkpointing.inspect_frontres_v015_quality_checkpoint(hsl_path, route="hsl").file_sha256,
            "policy": checkpointing.inspect_frontres_v015_quality_checkpoint(policy_path, route="policy").file_sha256,
        }

        def collect(_runner, item, route: str):
            calls.append((item.item_id, route))
            return quality.FrontRESV015PolicyQualityRouteEvidence(
                route=route,
                checkpoint_file_sha256=checkpoint_by_route[route],
                comparison_signature=item.comparison_signature,
                one_action_k=_evidence(storage, route=route),
            )

        bundle = quality.FrontRESV015PolicyQualityOwnerBundle(
            owner_identity=(
                ("reset", "frontres_segment_stage1_env_hooks"),
                ("observation", "frontres_runtime"),
                ("one_action_k", "frontres_segment_live_probe.collect_frontres_v015_one_action_k_evidence"),
                ("gain", "frontres_gain.compute_intent_physics_local_repair_gain"),
            ),
            collect_one_action_k=collect,
            training_state_signature=lambda _runner: repr(training_state),
        )
        runner = SimpleNamespace(alg=SimpleNamespace(frontres_v015_formal_transaction_enabled=True))
        quality.install_frontres_v015_policy_quality_owner_bundle(runner, bundle)
        payload = quality.run_frontres_policy_quality_eval(
            runner,
            manifest_path=str(manifest_path),
            hsl_checkpoint_path=str(hsl_path),
            policy_checkpoint_path=str(policy_path),
            result_path=str(result_path),
        )

        assert calls == [
            ("motion-a-k8", "zero"),
            ("motion-a-k8", "hsl"),
            ("motion-a-k8", "policy"),
        ]
        assert training_state == {"optimizer_steps": 7, "sampler_cursor": 11, "warmup": "disabled"}
        assert payload == json.loads(result_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == "frontres-v015-heldout-quality-report-v1"
        assert payload["gain_source"] == "FRS-GAIN-v003-intent-physics-local-repair"
        rows = payload["items"][0]["routes"]
        assert tuple(row["route"] for row in rows) == ("zero", "hsl", "policy")
        assert all(row["roles"] == ["repair", "noisy"] for row in rows)
        assert all(row["actor_forward_count"] == 1 and row["later_femr_action_count"] == 0 for row in rows)
        assert all(row["scenario_ids"] == ["scenario-heldout-a", "scenario-heldout-a"] for row in rows)
        assert all(row["noisy_segment_hashes"] == ["noisy-hash-a", "noisy-hash-a"] for row in rows)
        assert rows[0]["policy_actions"] == [[0.0] * 6]
        assert rows[0]["gain_total"] == [0.0]
        assert rows[1]["gain_total"][0] > rows[0]["gain_total"][0]
        assert rows[2]["gain_total"][0] > rows[1]["gain_total"][0]
        assert "return" not in repr(payload).lower()
        assert "priority" not in repr(payload).lower()
        assert "clean" not in repr(payload).lower()

        failed_path = root / "failed.json"

        def mutating_collect(_runner, item, route: str):
            evidence = quality.FrontRESV015PolicyQualityRouteEvidence(
                route=route,
                checkpoint_file_sha256=checkpoint_by_route[route],
                comparison_signature=item.comparison_signature,
                one_action_k=_evidence(storage, route=route),
            )
            if route == "hsl":
                training_state["optimizer_steps"] += 1
            return evidence

        failing_runner = SimpleNamespace(alg=SimpleNamespace(frontres_v015_formal_transaction_enabled=True))
        quality.install_frontres_v015_policy_quality_owner_bundle(
            failing_runner,
            quality.FrontRESV015PolicyQualityOwnerBundle(
                owner_identity=bundle.owner_identity,
                collect_one_action_k=mutating_collect,
                training_state_signature=lambda _runner: repr(training_state),
            ),
        )
        _expect_reject(
            lambda: quality.run_frontres_policy_quality_eval(
                failing_runner,
                manifest_path=str(manifest_path),
                hsl_checkpoint_path=str(hsl_path),
                policy_checkpoint_path=str(policy_path),
                result_path=str(failed_path),
            ),
            "training state",
        )
        assert not failed_path.exists()

        mixed_path = root / "mixed.json"

        def mixed_collect(_runner, item, route: str):
            evidence = _evidence(storage, route=route)
            if route == "policy":
                evidence = replace(
                    evidence,
                    scenario_ids=("scenario-mixed", "scenario-mixed"),
                    noisy_segment_hashes=("mixed-hash", "mixed-hash"),
                )
            return quality.FrontRESV015PolicyQualityRouteEvidence(
                route=route,
                checkpoint_file_sha256=checkpoint_by_route[route],
                comparison_signature=item.comparison_signature,
                one_action_k=evidence,
            )

        mixed_runner = SimpleNamespace(alg=SimpleNamespace(frontres_v015_formal_transaction_enabled=True))
        quality.install_frontres_v015_policy_quality_owner_bundle(
            mixed_runner,
            quality.FrontRESV015PolicyQualityOwnerBundle(
                owner_identity=bundle.owner_identity,
                collect_one_action_k=mixed_collect,
                training_state_signature=lambda _runner: repr(training_state),
            ),
        )
        _expect_reject(
            lambda: quality.run_frontres_policy_quality_eval(
                mixed_runner,
                manifest_path=str(manifest_path),
                hsl_checkpoint_path=str(hsl_path),
                policy_checkpoint_path=str(policy_path),
                result_path=str(mixed_path),
            ),
            "scenario",
        )
        assert not mixed_path.exists()

        wrong_checkpoint_path = root / "wrong_checkpoint.json"

        def wrong_checkpoint_collect(_runner, item, route: str):
            claimed = checkpoint_by_route["policy"] if route == "hsl" else checkpoint_by_route[route]
            return quality.FrontRESV015PolicyQualityRouteEvidence(
                route=route,
                checkpoint_file_sha256=claimed,
                comparison_signature=item.comparison_signature,
                one_action_k=_evidence(storage, route=route),
            )

        wrong_checkpoint_runner = SimpleNamespace(
            alg=SimpleNamespace(frontres_v015_formal_transaction_enabled=True)
        )
        quality.install_frontres_v015_policy_quality_owner_bundle(
            wrong_checkpoint_runner,
            quality.FrontRESV015PolicyQualityOwnerBundle(
                owner_identity=bundle.owner_identity,
                collect_one_action_k=wrong_checkpoint_collect,
                training_state_signature=lambda _runner: repr(training_state),
            ),
        )
        _expect_reject(
            lambda: quality.run_frontres_policy_quality_eval(
                wrong_checkpoint_runner,
                manifest_path=str(manifest_path),
                hsl_checkpoint_path=str(hsl_path),
                policy_checkpoint_path=str(policy_path),
                result_path=str(wrong_checkpoint_path),
            ),
            "checkpoint",
        )
        assert not wrong_checkpoint_path.exists()

        formal_path = root / "formal.json"
        formal_calls: list[tuple[str, str]] = []
        prepared_calls: list[str] = []
        context_calls: list[tuple[str, str]] = []

        @contextmanager
        def route_actor(_runner, _path, *, route: str, expected_file_sha256: str):
            context_calls.append((route, expected_file_sha256))
            yield

        checkpointing.frontres_v015_quality_route_actor = route_actor
        sampler_module = ModuleType("rsl_rl.runners.frontres_segment_live_sampler")

        def prepare_item(_runner, item):
            prepared_calls.append(item.comparison_signature)
            return SimpleNamespace(batch=object(), sample=object())

        sampler_module.prepare_frontres_v015_policy_quality_item_batch = prepare_item
        sys.modules[sampler_module.__name__] = sampler_module
        probe_module = ModuleType("rsl_rl.runners.frontres_segment_live_probe")
        probe_module._apply_current_segment_reset = lambda *_args, **_kwargs: SimpleNamespace(
            success_mask=torch.ones(8, dtype=torch.bool)
        )
        probe_module._read_live_observations = lambda _runner: object()

        def collect_formal(runner, _observations, *, pair_layout):
            route = runner._frontres_v015_quality_action_route
            formal_calls.append((route, f"{pair_layout.n_train}+{pair_layout.n_base}"))
            return _evidence(storage, route=route)

        probe_module.collect_frontres_v015_one_action_k_evidence = collect_formal
        sys.modules[probe_module.__name__] = probe_module
        setup_module = ModuleType("rsl_rl.runners.frontres_training_setup")
        setup_module.configure_frontres_pair_layout = lambda *_args, **_kwargs: SimpleNamespace(
            n_train=4,
            n_base=4,
            n_candidate=0,
            n_clean=0,
        )
        sys.modules[setup_module.__name__] = setup_module

        formal_runner = SimpleNamespace(
            alg=SimpleNamespace(frontres_v015_formal_transaction_enabled=True),
            current_learning_iteration=0,
        )
        formal_payload = quality.run_frontres_policy_quality_eval(
            formal_runner,
            manifest_path=str(manifest_path),
            hsl_checkpoint_path=str(hsl_path),
            policy_checkpoint_path=str(policy_path),
            result_path=str(formal_path),
        )
        assert isinstance(
            formal_runner._frontres_v015_policy_quality_owner_bundle,
            quality.FrontRESV015PolicyQualityOwnerBundle,
        )
        assert prepared_calls == [formal_payload["items"][0]["comparison_signature"]]
        assert formal_calls == [("zero", "4+4"), ("hsl", "4+4"), ("policy", "4+4")]
        assert context_calls == [
            ("hsl", checkpoint_by_route["hsl"]),
            ("policy", checkpoint_by_route["policy"]),
        ]
        assert formal_payload == json.loads(formal_path.read_text(encoding="utf-8"))
        assert not hasattr(formal_runner, "_frontres_v015_quality_action_route")


if __name__ == "__main__":
    test_v015_repair_noisy_one_action_k_atomic_quality()
    print("frontres_v015_policy_quality_heldout_contract: ok", flush=True)
