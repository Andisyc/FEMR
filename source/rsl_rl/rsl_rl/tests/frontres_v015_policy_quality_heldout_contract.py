#!/usr/bin/env python3
"""CPU-only G5-S2B contract for v015 Repair/Noisy held-out quality."""

from __future__ import annotations

import importlib.util
import hashlib
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


def _dynamic_identity(
    quality,
    comparison_signature: str,
    *,
    role_layout: tuple[str, ...] = ("repair", "noisy"),
    salt: str = "matched",
):
    field_hashes = tuple(
        (name, hashlib.sha256(f"{salt}:{name}".encode("ascii")).hexdigest())
        for name in quality._V015_DYNAMIC_STATE_FIELDS
    )
    return quality.FrontRESV015DynamicStateIdentity(
        comparison_signature=comparison_signature,
        role_layout=role_layout,
        field_hashes=field_hashes,
    )


def _expect_reject(fn, fragment: str) -> None:
    try:
        fn()
    except (RuntimeError, ValueError) as exc:
        assert fragment.lower() in str(exc).lower(), str(exc)
        return
    raise AssertionError(f"expected strict rejection containing {fragment!r}")


class _TrainingStateNormalizer(torch.nn.Module):
    """Expose a live-style running-state write whenever evaluation forgets eval mode."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("forward_updates", torch.zeros((), dtype=torch.long))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.forward_updates.add_(1)
        return value


class _Scene(dict):
    pass


def _semantic_dynamic_state_fixture():
    env_count = 8
    scene = _Scene()
    scene.env_origins = torch.arange(env_count * 3, dtype=torch.float32).reshape(env_count, 3)
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_state_w=torch.arange(env_count * 13, dtype=torch.float32).reshape(env_count, 13),
            joint_pos=torch.arange(env_count * 29, dtype=torch.float32).reshape(env_count, 29),
            joint_vel=torch.arange(env_count * 29, dtype=torch.float32).reshape(env_count, 29) * 0.01,
        )
    )
    scene["robot"] = robot
    roles = ["repair"] * 4 + ["noisy"] * 4
    local_snapshot = {
        "current_root_artifact_t": torch.arange(env_count * 7, dtype=torch.float32).reshape(env_count, 7),
        "intent_q29": torch.arange(env_count * 3 * 29, dtype=torch.float32).reshape(env_count, 3, 29),
        "clean_continuation": torch.arange(env_count * 2 * 65, dtype=torch.float32).reshape(env_count, 2, 65),
        "horizon_k": torch.full((env_count,), 2, dtype=torch.long),
        "continuation_lengths": torch.full((env_count,), 2, dtype=torch.long),
        "scenario_ids": tuple("scenario-q1" for _ in range(env_count)),
        "noisy_segment_hashes": tuple("hash-q1" for _ in range(env_count)),
        "x_t_identities": tuple("x-t-q1" for _ in range(env_count)),
        "roles": roles,
        "provenance": tuple(
            {"intent_q29_provenance": "deployment_noisy_q29", "row": row}
            for row in range(env_count)
        ),
    }
    command = SimpleNamespace(
        time_steps=torch.zeros(env_count, dtype=torch.long),
        env_motion_indices=torch.arange(env_count, dtype=torch.long),
        _cached_perturbed_pos=torch.zeros(env_count, 3),
        _cached_perturbed_quat=torch.zeros(env_count, 4),
        _frontres_pos_correction=torch.zeros(env_count, 3),
        _frontres_quat_correction=torch.zeros(env_count, 4),
        perturber=SimpleNamespace(scale=torch.arange(env_count, dtype=torch.float32)),
        frontres_local_scenario_snapshot=lambda _env_ids: {
            key: value.clone() if isinstance(value, torch.Tensor) else tuple(value)
            for key, value in local_snapshot.items()
        },
    )
    env = SimpleNamespace(
        num_envs=env_count,
        scene=scene,
        episode_length_buf=torch.arange(env_count, dtype=torch.long),
        command_manager=SimpleNamespace(get_term=lambda name: command if name == "motion" else None),
    )
    runner = SimpleNamespace(env=env, device="cpu")
    pair_layout = SimpleNamespace(n_train=4, n_base=4, n_candidate=0, n_clean=0)
    return runner, pair_layout, robot, local_snapshot


def test_v015_full_dynamic_state_identity_probe() -> None:
    _identity, _checkpointing, _manifest, quality, _storage, _gain = _owners()
    runner, pair_layout, robot, local_snapshot = _semantic_dynamic_state_fixture()
    comparison_signature = "a" * 64

    before_root = robot.data.root_state_w.clone()
    first = quality.capture_frontres_v015_policy_quality_dynamic_state_identity(
        runner,
        comparison_signature=comparison_signature,
        pair_layout=pair_layout,
    )
    second = quality.capture_frontres_v015_policy_quality_dynamic_state_identity(
        runner,
        comparison_signature=comparison_signature,
        pair_layout=pair_layout,
    )
    assert first == second
    assert first.role_layout == ("repair",) * 4 + ("noisy",) * 4
    assert tuple(name for name, _value in first.field_hashes) == quality._V015_DYNAMIC_STATE_FIELDS
    assert torch.equal(robot.data.root_state_w, before_root)

    robot.data.root_state_w[[0, 4]] = robot.data.root_state_w[[4, 0]].clone()
    permuted = quality.capture_frontres_v015_policy_quality_dynamic_state_identity(
        runner,
        comparison_signature=comparison_signature,
        pair_layout=pair_layout,
    )
    assert permuted.full_state_hash != first.full_state_hash
    robot.data.root_state_w.copy_(before_root)

    local_snapshot["roles"] = ["noisy"] * 4 + ["repair"] * 4
    _expect_reject(
        lambda: quality.capture_frontres_v015_policy_quality_dynamic_state_identity(
            runner,
            comparison_signature=comparison_signature,
            pair_layout=pair_layout,
        ),
        "role",
    )


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

        normalizers = {
            "prefix": _TrainingStateNormalizer(),
            "gmt": _TrainingStateNormalizer(),
            "privileged": _TrainingStateNormalizer(),
            "teacher": _TrainingStateNormalizer(),
        }
        policy = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout(p=0.5))
        policy[1].eval()
        normalizers["gmt"].eval()

        def normalizer_state() -> tuple[int, ...]:
            return tuple(int(module.forward_updates.item()) for module in normalizers.values())

        def collect(_runner, item, route: str):
            calls.append((item.item_id, route))
            sample = torch.ones(1, 1)
            for module in normalizers.values():
                module(sample)
            return quality.FrontRESV015PolicyQualityRouteEvidence(
                route=route,
                checkpoint_file_sha256=checkpoint_by_route[route],
                comparison_signature=item.comparison_signature,
                one_action_k=_evidence(storage, route=route),
                dynamic_state_identity=_dynamic_identity(quality, item.comparison_signature),
            )

        bundle = quality.FrontRESV015PolicyQualityOwnerBundle(
            owner_identity=(
                ("reset", "frontres_segment_stage1_env_hooks"),
                ("observation", "frontres_runtime"),
                ("one_action_k", "frontres_segment_live_probe.collect_frontres_v015_one_action_k_evidence"),
                ("gain", "frontres_gain.compute_intent_physics_local_repair_gain"),
            ),
            collect_one_action_k=collect,
            close_item=lambda _runner, _item: None,
            training_state_signature=lambda _runner: repr((training_state, normalizer_state())),
        )
        runner = SimpleNamespace(
            alg=SimpleNamespace(
                frontres_v015_formal_transaction_enabled=True,
                policy=policy,
            ),
            _frontres_extra_normalizer=normalizers["prefix"],
            obs_normalizer=normalizers["gmt"],
            privileged_obs_normalizer=normalizers["privileged"],
            teacher_obs_normalizer=normalizers["teacher"],
        )
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
        assert normalizer_state() == (0, 0, 0, 0)
        assert policy.training and not policy[1].training
        assert normalizers["prefix"].training
        assert not normalizers["gmt"].training
        assert normalizers["privileged"].training and normalizers["teacher"].training
        assert payload == json.loads(result_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == "frontres-v015-heldout-quality-report-v1"
        assert payload["gain_source"] == "FRS-GAIN-v003-intent-physics-local-repair"
        rows = payload["items"][0]["routes"]
        assert tuple(row["route"] for row in rows) == ("zero", "hsl", "policy")
        assert all(row["roles"] == ["repair", "noisy"] for row in rows)
        assert all(row["actor_forward_count"] == 1 and row["later_femr_action_count"] == 0 for row in rows)
        assert all(row["scenario_ids"] == ["scenario-heldout-a", "scenario-heldout-a"] for row in rows)
        assert all(row["noisy_segment_hashes"] == ["noisy-hash-a", "noisy-hash-a"] for row in rows)
        assert len({row["dynamic_state_identity"]["full_state_hash"] for row in rows}) == 1
        assert all(
            row["dynamic_state_identity"]["role_layout"] == ["repair", "noisy"]
            for row in rows
        )
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
                dynamic_state_identity=_dynamic_identity(quality, item.comparison_signature),
            )
            if route == "hsl":
                training_state["optimizer_steps"] += 1
            return evidence

        failing_runner = SimpleNamespace(
            alg=SimpleNamespace(
                frontres_v015_formal_transaction_enabled=True,
                policy=policy,
            ),
            _frontres_extra_normalizer=normalizers["prefix"],
            obs_normalizer=normalizers["gmt"],
            privileged_obs_normalizer=normalizers["privileged"],
            teacher_obs_normalizer=normalizers["teacher"],
        )
        quality.install_frontres_v015_policy_quality_owner_bundle(
            failing_runner,
            quality.FrontRESV015PolicyQualityOwnerBundle(
                owner_identity=bundle.owner_identity,
                collect_one_action_k=mutating_collect,
                close_item=lambda _runner, _item: None,
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
        assert policy.training and not policy[1].training
        assert normalizers["prefix"].training
        assert not normalizers["gmt"].training
        assert normalizers["privileged"].training and normalizers["teacher"].training

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
                dynamic_state_identity=_dynamic_identity(quality, item.comparison_signature),
            )

        mixed_runner = SimpleNamespace(alg=SimpleNamespace(frontres_v015_formal_transaction_enabled=True))
        quality.install_frontres_v015_policy_quality_owner_bundle(
            mixed_runner,
            quality.FrontRESV015PolicyQualityOwnerBundle(
                owner_identity=bundle.owner_identity,
                collect_one_action_k=mixed_collect,
                close_item=lambda _runner, _item: None,
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
                dynamic_state_identity=_dynamic_identity(quality, item.comparison_signature),
            )

        wrong_checkpoint_runner = SimpleNamespace(
            alg=SimpleNamespace(frontres_v015_formal_transaction_enabled=True)
        )
        quality.install_frontres_v015_policy_quality_owner_bundle(
            wrong_checkpoint_runner,
            quality.FrontRESV015PolicyQualityOwnerBundle(
                owner_identity=bundle.owner_identity,
                collect_one_action_k=wrong_checkpoint_collect,
                close_item=lambda _runner, _item: None,
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
        reset_calls: list[str] = []
        snapshot_calls: list[str] = []
        restore_calls: list[str] = []
        route_state = {"cuda_rng": 17, "sealed_scenario": "scenario-heldout-a"}
        failing_route = {"value": None}

        @contextmanager
        def route_actor(_runner, _path, *, route: str, expected_file_sha256: str):
            context_calls.append((route, expected_file_sha256))
            yield

        checkpointing.frontres_v015_quality_route_actor = route_actor
        sampler_module = ModuleType("rsl_rl.runners.frontres_segment_live_sampler")
        lifecycle_close_calls: list[object] = []

        def prepare_item(_runner, item):
            prepared_calls.append(item.comparison_signature)
            return SimpleNamespace(batch=object(), sample=object())

        sampler_module.prepare_frontres_v015_policy_quality_item_batch = prepare_item
        sampler_module._close_frontres_local_scenarios = lifecycle_close_calls.append
        sys.modules[sampler_module.__name__] = sampler_module
        probe_module = ModuleType("rsl_rl.runners.frontres_segment_live_probe")
        def reset_route_start(*_args, **_kwargs):
            reset_calls.append("reset")
            return SimpleNamespace(success_mask=torch.ones(8, dtype=torch.bool))

        probe_module._apply_current_segment_reset = reset_route_start
        probe_module._read_live_observations = lambda _runner: object()

        def collect_formal(runner, _observations, *, pair_layout):
            route = runner._frontres_v015_quality_action_route
            formal_calls.append((route, f"{pair_layout.n_train}+{pair_layout.n_base}"))
            evidence = _evidence(storage, route=route)
            route_state["cuda_rng"] += {"zero": 1, "hsl": 3, "policy": 7}[route]
            if failing_route["value"] == route:
                raise RuntimeError(f"deliberate {route} route failure")
            return evidence

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

        command_close_calls: list[str] = []
        command = SimpleNamespace(
            clear_frontres_local_scenario=lambda: command_close_calls.append("clear")
        )
        formal_runner = SimpleNamespace(
            alg=SimpleNamespace(frontres_v015_formal_transaction_enabled=True),
            current_learning_iteration=0,
            env=SimpleNamespace(
                command_manager=SimpleNamespace(
                    get_term=lambda name: command if name == "motion" else None
                )
            ),
        )
        def capture_route_start(_runner, *, env_ids, comparison_signature, role_layout):
            assert tuple(env_ids) == tuple(range(8))
            assert tuple(role_layout) == ("repair",) * 4 + ("noisy",) * 4
            snapshot_calls.append(comparison_signature)
            return SimpleNamespace(
                comparison_signature=comparison_signature,
                cuda_rng=route_state["cuda_rng"],
                sealed_scenario=route_state["sealed_scenario"],
            )

        def restore_route_start(_runner, snapshot, *, comparison_signature):
            assert snapshot.comparison_signature == comparison_signature
            assert route_state["sealed_scenario"] == snapshot.sealed_scenario
            route_state["cuda_rng"] = snapshot.cuda_rng
            restore_calls.append(comparison_signature)

        def capture_dynamic_identity(_runner, *, comparison_signature, pair_layout):
            assert pair_layout.n_train == 4 and pair_layout.n_base == 4
            field_hashes = tuple(
                (
                    name,
                    hashlib.sha256(
                        (
                            f"{name}:{route_state['cuda_rng']}"
                            if name == "cuda_rng_state"
                            else f"{name}:{route_state['sealed_scenario']}"
                        ).encode("ascii")
                    ).hexdigest(),
                )
                for name in quality._V015_DYNAMIC_STATE_FIELDS
            )
            return quality.FrontRESV015DynamicStateIdentity(
                comparison_signature=comparison_signature,
                role_layout=("repair", "noisy"),
                field_hashes=field_hashes,
            )

        quality.capture_frontres_policy_quality_state = capture_route_start
        quality.restore_frontres_policy_quality_state = restore_route_start
        quality.capture_frontres_v015_policy_quality_dynamic_state_identity = capture_dynamic_identity
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
        assert reset_calls == ["reset"]
        assert snapshot_calls == [formal_payload["items"][0]["comparison_signature"]]
        assert restore_calls == [formal_payload["items"][0]["comparison_signature"]] * 3
        assert formal_calls == [("zero", "4+4"), ("hsl", "4+4"), ("policy", "4+4")]
        assert context_calls == [
            ("hsl", checkpoint_by_route["hsl"]),
            ("policy", checkpoint_by_route["policy"]),
        ]
        assert formal_payload == json.loads(formal_path.read_text(encoding="utf-8"))
        assert not hasattr(formal_runner, "_frontres_v015_quality_action_route")
        assert command_close_calls == ["clear"]
        assert len(lifecycle_close_calls) == 1
        route_hashes = {
            row["dynamic_state_identity"]["full_state_hash"]
            for row in formal_payload["items"][0]["routes"]
        }
        assert len(route_hashes) == 1

        request = quality.build_frontres_v015_policy_quality_eval_request(
            manifest_path=str(manifest_path),
            hsl_checkpoint_path=str(hsl_path),
            policy_checkpoint_path=str(policy_path),
            result_path=str(root / "permuted.json"),
        )
        permuted_command_calls: list[str] = []
        permuted_command = SimpleNamespace(
            clear_frontres_local_scenario=lambda: permuted_command_calls.append("clear")
        )
        permuted_runner = SimpleNamespace(
            alg=SimpleNamespace(frontres_v015_formal_transaction_enabled=True),
            current_learning_iteration=0,
            env=SimpleNamespace(
                command_manager=SimpleNamespace(
                    get_term=lambda name: permuted_command if name == "motion" else None
                )
            ),
        )
        route_state["cuda_rng"] = 31
        reset_before = len(reset_calls)
        snapshot_before = len(snapshot_calls)
        restore_before = len(restore_calls)
        permuted_owner = quality.build_frontres_v015_policy_quality_owner_bundle(permuted_runner, request)
        training_before = permuted_owner.training_state_signature(permuted_runner)
        permuted_evidence = [
            permuted_owner.collect_one_action_k(permuted_runner, request.manifest.items[0], route)
            for route in ("policy", "zero", "hsl")
        ]
        permuted_owner.close_item(permuted_runner, request.manifest.items[0])
        assert len(reset_calls) == reset_before + 1
        assert len(snapshot_calls) == snapshot_before + 1
        assert len(restore_calls) == restore_before + 3
        assert len({row.dynamic_state_identity.full_state_hash for row in permuted_evidence}) == 1
        assert permuted_command_calls == ["clear"]
        assert permuted_owner.training_state_signature(permuted_runner) == training_before

        failure_command_calls: list[str] = []
        failure_command = SimpleNamespace(
            clear_frontres_local_scenario=lambda: failure_command_calls.append("clear")
        )
        failure_runner = SimpleNamespace(
            alg=SimpleNamespace(frontres_v015_formal_transaction_enabled=True),
            current_learning_iteration=0,
            env=SimpleNamespace(
                command_manager=SimpleNamespace(
                    get_term=lambda name: failure_command if name == "motion" else None
                )
            ),
        )
        failure_path = root / "route-start-failure.json"
        route_state["cuda_rng"] = 43
        failing_route["value"] = "hsl"
        failed_reset_before = len(reset_calls)
        _expect_reject(
            lambda: quality.run_frontres_policy_quality_eval(
                failure_runner,
                manifest_path=str(manifest_path),
                hsl_checkpoint_path=str(hsl_path),
                policy_checkpoint_path=str(policy_path),
                result_path=str(failure_path),
            ),
            "deliberate hsl route failure",
        )
        assert not failure_path.exists()
        assert failure_command_calls == ["clear"]
        assert len(reset_calls) == failed_reset_before + 1

        failing_route["value"] = None
        route_state["cuda_rng"] = 47
        recovered_path = root / "route-start-recovered.json"
        quality.run_frontres_policy_quality_eval(
            failure_runner,
            manifest_path=str(manifest_path),
            hsl_checkpoint_path=str(hsl_path),
            policy_checkpoint_path=str(policy_path),
            result_path=str(recovered_path),
        )
        assert recovered_path.is_file()
        assert failure_command_calls == ["clear", "clear"]
        assert len(reset_calls) == failed_reset_before + 2

        state_mismatch_path = root / "state-mismatch.json"

        def state_mismatch_collect(_runner, item, route: str):
            return quality.FrontRESV015PolicyQualityRouteEvidence(
                route=route,
                checkpoint_file_sha256=checkpoint_by_route[route],
                comparison_signature=item.comparison_signature,
                one_action_k=_evidence(storage, route=route),
                dynamic_state_identity=_dynamic_identity(
                    quality,
                    item.comparison_signature,
                    salt="mismatch" if route == "policy" else "matched",
                ),
            )

        state_mismatch_runner = SimpleNamespace(
            alg=SimpleNamespace(frontres_v015_formal_transaction_enabled=True)
        )
        quality.install_frontres_v015_policy_quality_owner_bundle(
            state_mismatch_runner,
            quality.FrontRESV015PolicyQualityOwnerBundle(
                owner_identity=quality._V015_QUALITY_OWNER_IDENTITY,
                collect_one_action_k=state_mismatch_collect,
                close_item=lambda _runner, _item: None,
                training_state_signature=lambda _runner: "stable",
            ),
        )
        _expect_reject(
            lambda: quality.run_frontres_policy_quality_eval(
                state_mismatch_runner,
                manifest_path=str(manifest_path),
                hsl_checkpoint_path=str(hsl_path),
                policy_checkpoint_path=str(policy_path),
                result_path=str(state_mismatch_path),
            ),
            "dynamic state",
        )
        assert not state_mismatch_path.exists()


def test_v015_manifest_item_lifecycle_closes_after_routes_and_on_error() -> None:
    identity, checkpointing, _manifest, quality, storage, _gain = _owners()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest_path = root / "manifest.json"
        hsl_path = root / "hsl.pt"
        policy_path = root / "policy.pt"
        result_path = root / "quality.json"
        payload = identity._manifest_payload()
        second = dict(payload["items"][0])
        second["item_id"] = "motion-a-k8-seed-8"
        second["seed"] = 8
        payload["items"] = [payload["items"][0], second]
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        torch.save(identity._hsl_payload(checkpointing), hsl_path)
        torch.save(identity._stage3_payload(checkpointing), policy_path)

        checkpoint_by_route = {
            "zero": "zero",
            "hsl": checkpointing.inspect_frontres_v015_quality_checkpoint(hsl_path, route="hsl").file_sha256,
            "policy": checkpointing.inspect_frontres_v015_quality_checkpoint(policy_path, route="policy").file_sha256,
        }
        events: list[tuple[str, str]] = []

        def collect(_runner, item, route: str):
            events.append((item.item_id, route))
            return quality.FrontRESV015PolicyQualityRouteEvidence(
                route=route,
                checkpoint_file_sha256=checkpoint_by_route[route],
                comparison_signature=item.comparison_signature,
                one_action_k=_evidence(storage, route=route),
                dynamic_state_identity=_dynamic_identity(quality, item.comparison_signature),
            )

        def close_item(_runner, item) -> None:
            events.append((item.item_id, "close"))

        runner = SimpleNamespace(alg=SimpleNamespace(frontres_v015_formal_transaction_enabled=True))
        quality.install_frontres_v015_policy_quality_owner_bundle(
            runner,
            quality.FrontRESV015PolicyQualityOwnerBundle(
                owner_identity=quality._V015_QUALITY_OWNER_IDENTITY,
                collect_one_action_k=collect,
                close_item=close_item,
                training_state_signature=lambda _runner: "stable",
            ),
        )
        quality.run_frontres_policy_quality_eval(
            runner,
            manifest_path=str(manifest_path),
            hsl_checkpoint_path=str(hsl_path),
            policy_checkpoint_path=str(policy_path),
            result_path=str(result_path),
        )
        assert events == [
            ("motion-a-k8", "zero"),
            ("motion-a-k8", "hsl"),
            ("motion-a-k8", "policy"),
            ("motion-a-k8", "close"),
            ("motion-a-k8-seed-8", "zero"),
            ("motion-a-k8-seed-8", "hsl"),
            ("motion-a-k8-seed-8", "policy"),
            ("motion-a-k8-seed-8", "close"),
        ]

        failed_events: list[tuple[str, str]] = []

        def failing_collect(_runner, item, route: str):
            failed_events.append((item.item_id, route))
            if route == "hsl":
                raise RuntimeError("route failure")
            return collect(_runner, item, route)

        failed_runner = SimpleNamespace(alg=SimpleNamespace(frontres_v015_formal_transaction_enabled=True))
        quality.install_frontres_v015_policy_quality_owner_bundle(
            failed_runner,
            quality.FrontRESV015PolicyQualityOwnerBundle(
                owner_identity=quality._V015_QUALITY_OWNER_IDENTITY,
                collect_one_action_k=failing_collect,
                close_item=lambda _runner, item: failed_events.append((item.item_id, "close")),
                training_state_signature=lambda _runner: "stable",
            ),
        )
        _expect_reject(
            lambda: quality.run_frontres_policy_quality_eval(
                failed_runner,
                manifest_path=str(manifest_path),
                hsl_checkpoint_path=str(hsl_path),
                policy_checkpoint_path=str(policy_path),
                result_path=str(root / "failed-lifecycle.json"),
            ),
            "route failure",
        )
        assert failed_events[-1] == ("motion-a-k8", "close")

        close_state = {"value": 0}
        close_mutation_runner = SimpleNamespace(
            alg=SimpleNamespace(frontres_v015_formal_transaction_enabled=True)
        )
        quality.install_frontres_v015_policy_quality_owner_bundle(
            close_mutation_runner,
            quality.FrontRESV015PolicyQualityOwnerBundle(
                owner_identity=quality._V015_QUALITY_OWNER_IDENTITY,
                collect_one_action_k=collect,
                close_item=lambda _runner, _item: close_state.__setitem__("value", 1),
                training_state_signature=lambda _runner: repr(close_state),
            ),
        )
        _expect_reject(
            lambda: quality.run_frontres_policy_quality_eval(
                close_mutation_runner,
                manifest_path=str(manifest_path),
                hsl_checkpoint_path=str(hsl_path),
                policy_checkpoint_path=str(policy_path),
                result_path=str(root / "close-mutation.json"),
            ),
            "item close mutated training state",
        )
        assert not (root / "close-mutation.json").exists()


if __name__ == "__main__":
    test_v015_full_dynamic_state_identity_probe()
    test_v015_repair_noisy_one_action_k_atomic_quality()
    test_v015_manifest_item_lifecycle_closes_after_routes_and_on_error()
    print("frontres_v015_policy_quality_heldout_contract: ok", flush=True)
