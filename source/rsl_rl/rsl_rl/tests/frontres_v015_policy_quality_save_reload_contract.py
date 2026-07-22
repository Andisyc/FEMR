#!/usr/bin/env python3
"""Offline G5-S3 contract for real v015 save/reload to atomic quality report."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

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


def _update_state_digest(digest, value) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
        return
    if isinstance(value, dict):
        for key in sorted(value, key=repr):
            digest.update(repr(key).encode("utf-8"))
            _update_state_digest(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        digest.update(type(value).__name__.encode("ascii"))
        for item in value:
            _update_state_digest(digest, item)
        return
    digest.update(repr(value).encode("utf-8"))


def _state_signature(runner, checkpointing) -> str:
    digest = hashlib.sha256()
    digest.update(
        checkpointing._v015_state_dict_fingerprint(
            runner.alg.policy.residual_actor.state_dict(),
            label="G5-S3 actor",
        ).encode("ascii")
    )
    digest.update(
        checkpointing._v015_tensor_fingerprint(
            runner._frontres_extra_mean,
            runner._frontres_extra_std,
        ).encode("ascii")
    )
    _update_state_digest(digest, runner.alg.optimizer.state_dict())
    _update_state_digest(digest, runner._frontres_v015_checkpoint_transaction_state)
    _update_state_digest(digest, runner._frontres_segment_sampler.state_dict())
    _update_state_digest(
        digest, getattr(runner, "_frontres_v015_last_committed_transaction_receipt", None)
    )
    digest.update(str(runner.alg.optimizer.frontres_v015_step_count).encode("ascii"))
    digest.update(str(bool(getattr(runner, "_frontres_warmup_complete", False))).encode("ascii"))
    return digest.hexdigest()


def _bind_route_evidence(base, *, action: torch.Tensor, combined: torch.Tensor):
    action = action[:1].detach().clone()
    combined = combined[:1].detach().clone()
    return replace(
        base,
        policy_observations=combined,
        policy_actions=action,
        policy_means=action,
    )


def test_real_save_fresh_reload_to_atomic_quality_report() -> None:
    hsl_contract = _load(
        "frontres_v015_g5_s3_hsl_fixture",
        RSL_ROOT / "tests" / "frontres_hsl_v007_s1_contract.py",
    )
    hsl_checkpointing, hsl_layout_module, hsl_runtime, warmup = (
        hsl_contract._load_hsl_fresh_connectivity_owners()
    )
    hsl_layout = hsl_layout_module.resolve_frontres_future_intent_layout(
        (1, 2), hsl_layout_module.FRONTRES_FUTURE_INTENT_LAYOUT_VERSION
    )
    assert callable(
        getattr(hsl_checkpointing, "frontres_v015_quality_route_actor", None)
    )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        gmt_path = root / "gmt.pt"
        hsl_path = root / "hsl.pt"
        policy_path = root / "policy.pt"
        manifest_path = root / "manifest.json"
        report_path = root / "quality.json"
        torch.save({"artifact": "g5-s3-frozen-gmt"}, gmt_path)

        intent = hsl_contract._intent(batch_size=2, hmax=2)
        proposal_snapshot = hsl_contract._hsl_proposal_snapshot(intent)
        raw_obs = torch.arange(2 * 870, dtype=torch.float32).reshape(2, 870) / 1000.0
        current_artifact = torch.tensor(
            [
                [0.12, -0.07, 0.03, 1.0, 0.01, 0.02, 0.03],
                [-0.09, 0.05, -0.02, 1.0, -0.02, 0.01, 0.04],
            ]
        )
        raw_obs[:, :7] = current_artifact

        torch.manual_seed(5)
        hsl_source = hsl_contract._hsl_checkpoint_runner(
            hsl_checkpointing, hsl_layout, gmt_path, seed=5
        )
        hsl_source.alg.policy.residual_actor = torch.nn.Linear(158, 6)
        hsl_contract._wire_hsl_fresh_runner(hsl_source, hsl_runtime, proposal_snapshot)
        hsl_before = hsl_contract._hsl_fresh_trace(hsl_source, warmup, raw_obs)
        hsl_checkpointing.save_runner(hsl_source, str(hsl_path))

        torch.manual_seed(23)
        hsl_fresh = hsl_contract._hsl_checkpoint_runner(
            hsl_checkpointing, hsl_layout, gmt_path, seed=23
        )
        hsl_fresh.alg.policy.residual_actor = torch.nn.Linear(158, 6)
        hsl_contract._wire_hsl_fresh_runner(hsl_fresh, hsl_runtime, proposal_snapshot)
        hsl_checkpointing.load_runner(
            hsl_fresh,
            str(hsl_path),
            load_optimizer=False,
            load_critic=False,
        )
        hsl_after = hsl_contract._hsl_fresh_trace(hsl_fresh, warmup, raw_obs)
        for name in ("combined", "normalized", "actor_input", "proposal"):
            torch.testing.assert_close(hsl_after[name], hsl_before[name], rtol=0.0, atol=0.0)

        checkpoint_contract = _load(
            "frontres_v015_g5_s3_checkpoint_fixture",
            RSL_ROOT / "tests" / "frontres_v015_checkpoint_resume_contract.py",
        )
        stage3_layout_module, stage3_checkpointing, policy_cls = checkpoint_contract._load_owners()
        stage3_runtime = checkpoint_contract._load_runtime()
        transaction_template = checkpoint_contract._transaction_template()
        torch.manual_seed(17)
        source = checkpoint_contract._runner(stage3_layout_module, policy_cls, iteration=2)
        source.policy_cfg["gmt_checkpoint_path"] = str(gmt_path)
        source.alg.policy.gmt_policy_obs_dim = hsl_source.alg.policy.gmt_policy_obs_dim
        source.alg.policy.gmt_normalizer = copy.deepcopy(hsl_source.alg.policy.gmt_normalizer)
        source_snapshot = checkpoint_contract._wire_inference_carrier(source, intent)
        pre_update = checkpoint_contract._fresh_inference_trace(source, stage3_runtime, raw_obs)
        request = checkpoint_contract._bind_semantic_transaction(source, transaction_template)
        update = transaction_template.owners[6].run_frontres_v015_formal_transaction_update(
            source, request
        )
        committed = copy.deepcopy(source._frontres_v015_checkpoint_transaction_state)
        assert committed["state"] == "committed"
        assert committed["receipt"]["transaction_id"] == update.transaction_id
        assert committed["receipt"]["optimizer_step_delta"] == 1
        assert source.alg.optimizer.frontres_v015_step_count == 1
        source.current_learning_iteration += 1
        stage3_before = checkpoint_contract._fresh_inference_trace(source, stage3_runtime, raw_obs)
        assert not torch.equal(pre_update["proposal"], stage3_before["proposal"])
        stage3_checkpointing.save_runner(source, str(policy_path))
        saved_policy = torch.load(policy_path, map_location="cpu", weights_only=False)
        assert saved_policy["frontres_v015_checkpoint_identity"]["transaction"] == committed

        torch.manual_seed(29)
        fresh = checkpoint_contract._runner(stage3_layout_module, policy_cls, iteration=0)
        fresh._frontres_extra_mean.fill_(-7.0)
        fresh._frontres_extra_std.fill_(3.0)
        fresh_snapshot = checkpoint_contract._wire_inference_carrier(fresh, intent)
        stage3_checkpointing.load_runner(fresh, str(policy_path), load_optimizer=False)
        stage3_after = checkpoint_contract._fresh_inference_trace(fresh, stage3_runtime, raw_obs)

        torch.testing.assert_close(
            source_snapshot["intent_q29"], fresh_snapshot["intent_q29"], rtol=0.0, atol=0.0
        )
        for name in (
            "scenario_ids",
            "noisy_segment_hashes",
            "x_t_identities",
            "roles",
            "provenance",
        ):
            assert source_snapshot[name] == fresh_snapshot[name]
        assert tuple(stage3_after["combined"].shape) == (2, 928)
        assert tuple(stage3_after["actor_input"].shape) == (2, 158)
        assert tuple(stage3_after["normalized"][:, 158:].shape) == (2, 770)
        assert tuple(stage3_after["proposal"].shape) == (2, 6)
        expected_tail = intent[:, (1, 2), :].reshape(2, 58)
        torch.testing.assert_close(stage3_after["combined"][:, :58], expected_tail)
        for name in ("combined", "normalized", "actor_input", "proposal"):
            torch.testing.assert_close(
                stage3_after[name], stage3_before[name], rtol=0.0, atol=0.0
            )
        torch.testing.assert_close(
            hsl_after["combined"], stage3_after["combined"], rtol=0.0, atol=0.0
        )

        identity_contract = _load(
            "frontres_v015_g5_s3_identity_fixture",
            RSL_ROOT / "tests" / "frontres_v015_policy_quality_identity_contract.py",
        )
        quality_checkpointing, _manifest_owner, quality = identity_contract._owners()
        heldout_contract = _load(
            "frontres_v015_g5_s3_heldout_fixture",
            RSL_ROOT / "tests" / "frontres_v015_policy_quality_heldout_contract.py",
        )
        storage = _load(
            "rsl_rl.frontres.frontres_segment_storage",
            RSL_ROOT / "frontres" / "frontres_segment_storage.py",
        )
        _load(
            "rsl_rl.frontres.frontres_gain",
            RSL_ROOT / "frontres" / "frontres_gain.py",
        )
        manifest_path.write_text(
            json.dumps(identity_contract._manifest_payload()), encoding="utf-8"
        )
        strict_request = quality.build_frontres_v015_policy_quality_eval_request(
            manifest_path=str(manifest_path),
            hsl_checkpoint_path=str(hsl_path),
            policy_checkpoint_path=str(policy_path),
            result_path=str(report_path),
        )
        assert strict_request.hsl_checkpoint.format == "frontres-v015-hsl-proposal-v1"
        assert strict_request.policy_checkpoint.format == "frontres-v015-checkpoint-v4"
        policy_layout = dict(strict_request.policy_checkpoint.future_intent_layout)
        assert policy_layout["actor_dim"] == 928
        assert policy_layout["prefix_dim"] == 158
        assert policy_layout["gmt_dim"] == 770

        source._frontres_extra_normalizer = copy.deepcopy(
            hsl_source._frontres_extra_normalizer
        )
        source_state_before_routes = _state_signature(source, quality_checkpointing)
        with stage3_checkpointing.frontres_v015_quality_route_actor(
            source,
            hsl_path,
            route="hsl",
            expected_file_sha256=strict_request.hsl_checkpoint.file_sha256,
        ):
            hsl_route = checkpoint_contract._fresh_inference_trace(
                source, stage3_runtime, raw_obs
            )
            torch.testing.assert_close(
                hsl_route["proposal"], hsl_after["proposal"], rtol=0.0, atol=0.0
            )
        assert _state_signature(source, quality_checkpointing) == source_state_before_routes
        with stage3_checkpointing.frontres_v015_quality_route_actor(
            source,
            policy_path,
            route="policy",
            expected_file_sha256=strict_request.policy_checkpoint.file_sha256,
        ):
            policy_route = checkpoint_contract._fresh_inference_trace(
                source, stage3_runtime, raw_obs
            )
            torch.testing.assert_close(
                policy_route["proposal"], stage3_after["proposal"], rtol=0.0, atol=0.0
            )
        assert _state_signature(source, quality_checkpointing) == source_state_before_routes

        actions = {
            "zero": torch.zeros_like(stage3_after["proposal"]),
            "hsl": hsl_after["proposal"],
            "policy": stage3_after["proposal"],
        }
        observations = {
            "zero": stage3_after["combined"],
            "hsl": hsl_after["combined"],
            "policy": stage3_after["combined"],
        }
        checkpoint_hashes = {
            "zero": "zero",
            "hsl": strict_request.hsl_checkpoint.file_sha256,
            "policy": strict_request.policy_checkpoint.file_sha256,
        }
        baseline_signature = _state_signature(source, quality_checkpointing)

        def collect(_runner, item, route: str):
            evidence = heldout_contract._evidence(storage, route=route)
            evidence = _bind_route_evidence(
                evidence,
                action=actions[route],
                combined=observations[route],
            )
            return quality.FrontRESV015PolicyQualityRouteEvidence(
                route=route,
                checkpoint_file_sha256=checkpoint_hashes[route],
                comparison_signature=item.comparison_signature,
                one_action_k=evidence,
                dynamic_state_identity=heldout_contract._dynamic_identity(
                    quality,
                    item.comparison_signature,
                ),
            )

        bundle = quality.FrontRESV015PolicyQualityOwnerBundle(
            owner_identity=(
                ("reset", "frontres_segment_stage1_env_hooks"),
                ("observation", "frontres_runtime"),
                (
                    "one_action_k",
                    "frontres_segment_live_probe.collect_frontres_v015_one_action_k_evidence",
                ),
                ("gain", "frontres_gain.compute_intent_physics_local_repair_gain"),
            ),
            collect_one_action_k=collect,
            close_item=lambda _runner, _item: None,
            training_state_signature=lambda runner: _state_signature(
                runner, quality_checkpointing
            ),
        )
        quality.install_frontres_v015_policy_quality_owner_bundle(source, bundle)
        payload = quality.run_frontres_policy_quality_eval(
            source,
            manifest_path=str(manifest_path),
            hsl_checkpoint_path=str(hsl_path),
            policy_checkpoint_path=str(policy_path),
            result_path=str(report_path),
        )

        assert _state_signature(source, quality_checkpointing) == baseline_signature
        assert payload == json.loads(report_path.read_text(encoding="utf-8"))
        assert not report_path.with_suffix(".json.tmp").exists()
        assert payload["manifest_file_sha256"] == strict_request.manifest_file_sha256
        report_routes = payload["items"][0]["routes"]
        assert tuple(row["route"] for row in report_routes) == (
            "zero",
            "hsl",
            "policy",
        )
        assert tuple(row["checkpoint_file_sha256"] for row in report_routes) == (
            "zero",
            strict_request.hsl_checkpoint.file_sha256,
            strict_request.policy_checkpoint.file_sha256,
        )
        assert report_routes[1]["policy_actions"] == hsl_after["proposal"][:1].tolist()
        assert report_routes[2]["policy_actions"] == stage3_after["proposal"][:1].tolist()
        assert fresh._frontres_v015_last_committed_transaction_receipt == committed["receipt"]

    print(
        "[G5-S3/T-commit/T-save/T-fresh-runner/T-928-158-770/T-q29/"
        "T-prefix-normalizer/T-proposal-equality/T-identity/T-atomic-report/T-isolation] PASS",
        flush=True,
    )


if __name__ == "__main__":
    test_real_save_fresh_reload_to_atomic_quality_report()
    print("frontres_v015_policy_quality_save_reload_contract: ok", flush=True)
