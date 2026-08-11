#!/usr/bin/env python3
"""Deterministic TEST-16 contracts for strict checkpoint-v16 persistence."""

from __future__ import annotations

import copy
import contextlib
import hashlib
import io
import json
import tempfile
from pathlib import Path
import sys

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from frontres_v015_checkpoint_resume_contract import (
    SCHEDULE as _LEGACY_SCHEDULE,
    _FrozenNormalizer,
    _load_owners,
    _receipt as _legacy_receipt,
    _runner as _legacy_runner,
)
from rsl_rl.algorithms.frontres_segment_ppo import (
    FRONTRES_VALUE_NORMALIZATION_ID,
    FrontRESValueNormalizerState,
)
from rsl_rl.frontres.frontres_outer_scenario_replay import (
    FRONTRES_OUTER_REPLAY_SCHEMA,
    FrontRESOuterScenarioReplay,
    FrontRESScenarioKey,
)
from rsl_rl.frontres.frontres_segment_warmup import resolve_frontres_k_stage_identity


def _active_schedule(schedule):
    return tuple(
        (
            stage[0],
            4,
            stage[2],
            stage[3],
            stage[4],
            stage[5],
            stage[6],
            "linear-coupled-v1",
            stage[2] + stage[3],
            stage[9],
        )
        for stage in schedule
    )


def _schedule_fingerprint(schedule) -> str:
    return hashlib.sha256(json.dumps(schedule, separators=(",", ":")).encode("ascii")).hexdigest()


ACTIVE_SCHEDULE = _active_schedule(_LEGACY_SCHEDULE)


def _runner(layout, policy_base, *, iteration: int, gmt_checkpoint_path: Path):
    runner = _legacy_runner(layout, policy_base, iteration=iteration, gmt_checkpoint_path=gmt_checkpoint_path)
    runner.alg.frontres_segment_k_curriculum = ACTIVE_SCHEDULE
    runner.alg.frontres_segment_k_curriculum_fingerprint = _schedule_fingerprint(
        runner.alg.frontres_segment_k_curriculum
    )
    policy = runner.alg.policy
    policy.critic = torch.nn.Linear(449, 1)
    runner.alg.optimizer = torch.optim.Adam(
        [
            {
                "params": list(policy.residual_actor.parameters()),
                "lr": 3.0e-6,
                "frontres_role": "actor",
                "frontres_step_count": 0,
            },
            {
                "params": list(policy.critic.parameters()),
                "lr": 1.0e-5,
                "frontres_role": "critic",
                "frontres_step_count": 0,
            },
        ]
    )
    runner.alg.frontres_method_contract_id = "FRS-METHOD-v022"
    runner.alg.frontres_optimization_contract_id = "FRS-PPO-v009"
    runner.alg.frontres_training_contract_id = "FRS-TRAIN-v021"
    runner.alg.frontres_critic_value_kind = "state_value"
    runner.alg.frontres_critic_input_dim = 449
    runner.alg.frontres_critic_action_conditioned = False
    runner.alg.frontres_critic_target_id = "segment-exact-m-mean-symlog-v1"
    runner.alg.frontres_return_utility_id = "symmetric-log-gain-g0-1-v1"
    runner.alg.frontres_return_utility_scale = 1.0
    runner.alg.frontres_critic_support_context_id = "action-pre-support-plan-kmax32-v1"
    runner.alg.frontres_gradient_clip_identity = "separate-actor-critic-v1"
    runner.alg.frontres_critic_value_normalization = FRONTRES_VALUE_NORMALIZATION_ID
    runner.alg.frontres_critic_value_normalizer_decay = 0.9
    runner.alg.frontres_critic_value_normalizer_scale_floor = 1.0
    runner.alg.frontres_critic_value_normalizer_state = FrontRESValueNormalizerState(
        mean=0.25 if iteration else 0.0,
        second_moment=2.0 if iteration else 1.0,
        update_count=iteration,
    )
    runner.alg.max_grad_norm = 0.5
    runner._frontres_critic_observation_dim = 449
    runner.alg.frontres_formal_runtime_audit = True
    runner.empirical_normalization = True
    runner.obs_normalizer = policy.gmt_normalizer
    runner.privileged_obs_normalizer = _FrozenNormalizer(449)
    runner._frontres_extra_mean = torch.zeros(1, 158)
    runner._frontres_extra_std = torch.ones(1, 158)
    runner._frontres_extra_normalizer = None
    runner._frontres_extra_stats_layout_version = runner._frontres_future_intent_layout.version
    runner._frontres_outer_scenario_replay = FrontRESOuterScenarioReplay(seed=17)
    return runner


def _receipt(checkpointing, *, training_iteration: int) -> dict[str, object]:
    value = _legacy_receipt(checkpointing, training_iteration=training_iteration)
    receipt = value["receipt"]
    identity = resolve_frontres_k_stage_identity(
        schedule=ACTIVE_SCHEDULE,
        committed_update_iteration=training_iteration,
        max_horizon_k=32,
    )
    receipt["method_contract_id"] = "FRS-METHOD-v022"
    receipt["gain_contract_id"] = "FRS-GAIN-v008"
    receipt["optimization_contract_id"] = "FRS-PPO-v009"
    receipt["training_contract_id"] = "FRS-TRAIN-v021"
    receipt["scalar_target_id"] = "symmetric-log-recovery-aware-utility-v1"
    receipt["active_m"] = 4
    receipt["expected_policy_row_count"] = 8
    receipt["collected_policy_attempt_count"] = 8
    receipt["valid_policy_row_count"] = 8
    receipt["policy_row_count"] = 8
    receipt["role_row_count"] = 16
    receipt["curriculum_fingerprint"] = _schedule_fingerprint(ACTIVE_SCHEDULE)
    receipt["k_stage_index"] = identity.stage_index
    receipt["active_k"] = identity.active_k
    receipt["active_m"] = identity.active_m
    receipt["k_stage_iteration"] = identity.stage_iteration
    receipt["training_iteration"] = identity.absolute_iteration
    receipt["dr_stage_fingerprint"] = identity.dr_stage_fingerprint
    receipt["dr_progress"] = identity.dr_progress
    receipt["d_cap"] = identity.d_cap
    return value


def _expect_error(call, text: str) -> None:
    try:
        call()
    except RuntimeError as exc:
        assert text.lower() in str(exc).lower(), str(exc)
        return
    raise AssertionError("expected checkpoint-v16 rejection")


def _prime_outer_replay(runner) -> None:
    owner = runner._frontres_outer_scenario_replay
    plan = owner.plan(
        transaction_id="tx-checkpoint-v16-replay",
        curriculum=resolve_frontres_k_stage_identity(
            schedule=runner.alg.frontres_segment_k_curriculum,
            committed_update_iteration=0,
        ),
        num_segments=2,
        eligible=lambda _segment_id: True,
        global_family=lambda _segment_id: "local_rp",
    )
    keys = tuple(
        FrontRESScenarioKey(
            motion_id=f"motion-{index}",
            start_frame=index,
            segment_id=selection.segment_id,
            x_t_identity=f"x-{index}",
            perturbation_family=selection.perturbation_family,
            perturbation_strength=selection.perturbation_strength,
            perturbation_seed=selection.perturbation_seed,
            noisy_segment_hash=f"hash-{index}",
            horizon_k=8,
            future_intent_identity=f"future-{index}",
            planned_support_identity=f"support-{index}",
        )
        for index, selection in enumerate(plan.selections)
    )
    candidate = owner.stage(
        plan,
        keys=keys,
        actor_advantages=torch.tensor([0.5] * 4 + [-0.25] * 4),
        source_index=torch.tensor([0] * 4 + [1] * 4),
        policy_snapshot_id="policy-checkpoint-v16",
        active_m=4,
    )
    owner.commit(
        candidate,
        receipt={
            "method_contract_id": "FRS-METHOD-v022",
            "training_contract_id": "FRS-TRAIN-v021",
            "transaction_id": plan.transaction_id,
            "policy_snapshot_id": "policy-checkpoint-v16",
            "optimizer_step_delta": 1,
        },
    )


def main() -> None:
    layout, checkpointing, policy_base = _load_owners()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        gmt_path = root / "gmt.pt"
        gmt_path.write_bytes(b"frozen GMT artifact v11")
        path = root / "model_3.pt"
        source = _runner(layout, policy_base, iteration=3, gmt_checkpoint_path=gmt_path)
        _prime_outer_replay(source)
        source._frontres_checkpoint_transaction_state = _receipt(checkpointing, training_iteration=2)
        source.privileged_obs_normalizer._var[..., 0] = 0.0
        source.privileged_obs_normalizer._std[..., 0] = 0.0
        actor_state = copy.deepcopy(source.alg.policy.residual_actor.state_dict())
        critic_state = copy.deepcopy(source.alg.policy.critic.state_dict())
        had_loaded_path = hasattr(source, "_frontres_last_loaded_checkpoint_path")
        audit_output = io.StringIO()
        with contextlib.redirect_stdout(audit_output):
            checkpointing.save_runner(source, str(path))
        assert audit_output.getvalue().count("[AUDIT-B08]") == 1
        assert "readback=1" in audit_output.getvalue()
        assert hasattr(source, "_frontres_last_loaded_checkpoint_path") is had_loaded_path

        payload = torch.load(path, weights_only=False)
        identity = payload["frontres_v015_checkpoint_identity"]
        assert identity["format"] == "frontres-v021-checkpoint-v16"
        assert identity["method_contract_id"] == "FRS-METHOD-v022"
        assert identity["optimization_contract_id"] == "FRS-PPO-v009"
        assert identity["training_contract_id"] == "FRS-TRAIN-v021"
        assert identity["outer_replay_schema_id"] == FRONTRES_OUTER_REPLAY_SCHEMA
        assert identity["return_utility"] == {
            "identity": "symmetric-log-gain-g0-1-v1",
            "scale": 1.0,
            "placement": "per-attempt-before-exact-m-mean",
        }
        assert identity["critic"] == {
            "value_kind": "state_value",
            "input_dim": 449,
            "action_conditioned": False,
            "target_id": "segment-exact-m-mean-symlog-v1",
            "return_utility_id": "symmetric-log-gain-g0-1-v1",
            "return_utility_scale": 1.0,
            "support_context_id": "action-pre-support-plan-kmax32-v1",
        }
        assert identity["gradient_clip"] == {
            "identity": "separate-actor-critic-v1",
            "max_norm": 0.5,
        }
        assert identity["critic_value_normalizer"] == {
            "identity": FRONTRES_VALUE_NORMALIZATION_ID,
            "decay": 0.9,
            "scale_floor": 1.0,
        }
        assert payload["frontres_critic_value_normalizer_state_dict"] == {
            "normalization_id": FRONTRES_VALUE_NORMALIZATION_ID,
            "mean": 0.25,
            "second_moment": 2.0,
            "update_count": 3,
        }
        assert payload["privileged_obs_norm_state_dict"]["_var"][0, 0].item() == 0.0
        assert payload["privileged_obs_norm_state_dict"]["_std"][0, 0].item() == 0.0
        active_quality_identity = checkpointing.inspect_frontres_quality_checkpoint(path, route="policy")
        assert active_quality_identity.format == "frontres-v021-checkpoint-v16"
        assert active_quality_identity.ppo_contract_id == "FRS-PPO-v009"
        assert active_quality_identity.training_contract_id == "FRS-TRAIN-v021"

        fresh = _runner(layout, policy_base, iteration=0, gmt_checkpoint_path=gmt_path)
        checkpointing.load_runner(fresh, str(path), load_optimizer=True)
        for name, value in actor_state.items():
            torch.testing.assert_close(fresh.alg.policy.residual_actor.state_dict()[name], value)
        for name, value in critic_state.items():
            torch.testing.assert_close(fresh.alg.policy.critic.state_dict()[name], value)
        assert fresh.alg.frontres_critic_value_normalizer_state == source.alg.frontres_critic_value_normalizer_state
        assert len(fresh._frontres_outer_scenario_replay.records) == 2
        assert torch.equal(
            fresh._frontres_outer_scenario_replay.generator.get_state(),
            source._frontres_outer_scenario_replay.generator.get_state(),
        )

        checkpoint_v11 = copy.deepcopy(payload)
        checkpoint_v11["frontres_v015_checkpoint_identity"].update(
            format="frontres-v017-checkpoint-v11",
            optimization_contract_id="FRS-PPO-v006",
            training_contract_id="FRS-TRAIN-v016",
        )
        checkpoint_v11["frontres_v015_checkpoint_identity"].pop("critic_value_normalizer")
        checkpoint_v11.pop("frontres_critic_value_normalizer_state_dict")
        checkpoint_v11_path = root / "checkpoint-v11.pt"
        torch.save(checkpoint_v11, checkpoint_v11_path)
        checkpoint_v11_target = _runner(layout, policy_base, iteration=0, gmt_checkpoint_path=gmt_path)
        _expect_error(
            lambda: checkpointing.load_runner(checkpoint_v11_target, str(checkpoint_v11_path)),
            "contract or format",
        )
        assert checkpoint_v11_target.alg.frontres_critic_value_normalizer_state.update_count == 0

        legacy_quality = copy.deepcopy(payload)
        legacy_identity = legacy_quality["frontres_v015_checkpoint_identity"]
        legacy_identity.update(
            format="frontres-v017-checkpoint-v10",
            method_contract_id="FRS-METHOD-v017",
            gain_contract_id="FRS-GAIN-v007",
            optimization_contract_id="FRS-PPO-v005",
            training_contract_id="FRS-TRAIN-v015",
            scalar_target_id="clean-anchored-recovery-aware-gain-v1",
        )
        legacy_identity.pop("return_utility")
        legacy_identity.pop("critic")
        legacy_identity.pop("gradient_clip")
        legacy_identity.pop("critic_value_normalizer")
        legacy_quality.pop("frontres_critic_value_normalizer_state_dict")
        legacy_obs_norm = {
            "_mean": torch.zeros(1, 928),
            "_var": torch.ones(1, 928),
            "_std": torch.ones(1, 928),
            "count": torch.tensor(4.0),
        }
        legacy_quality["obs_norm_state_dict"] = legacy_obs_norm
        legacy_identity["normalizer"] = {
            "mode": "empirical_prefix_plus_frozen_gmt",
            "prefix_layout_version": "frontres-v015-future-intent-q29-v1",
            "prefix_dim": 158,
            "combined_dim": 928,
            "prefix_stats_fingerprint": checkpointing._v015_tensor_fingerprint(
                legacy_obs_norm["_mean"][..., :158],
                legacy_obs_norm["_std"][..., :158],
            ),
        }
        legacy_receipt = legacy_identity["transaction"]["receipt"]
        legacy_receipt.update(
            method_contract_id="FRS-METHOD-v017",
            gain_contract_id="FRS-GAIN-v007",
            optimization_contract_id="FRS-PPO-v005",
            training_contract_id="FRS-TRAIN-v015",
            scalar_target_id="clean-anchored-recovery-aware-gain-v1",
        )
        legacy_path = root / "legacy-quality-v10.pt"
        torch.save(legacy_quality, legacy_path)
        read_only_identity = checkpointing.inspect_frontres_quality_checkpoint(legacy_path, route="policy")
        assert read_only_identity.format == "frontres-v017-checkpoint-v10"
        assert read_only_identity.training_contract_id == "FRS-TRAIN-v015"
        legacy_target = _runner(layout, policy_base, iteration=0, gmt_checkpoint_path=gmt_path)
        _expect_error(lambda: checkpointing.load_runner(legacy_target, str(legacy_path)), "contract or format")

        for label, mutate, message in (
            ("missing-critic", lambda item: item.pop("critic"), "contract or format"),
            (
                "missing-value-normalizer-identity",
                lambda item: item.pop("critic_value_normalizer"),
                "contract or format",
            ),
        ):
            tampered = copy.deepcopy(payload)
            mutate(tampered["frontres_v015_checkpoint_identity"])
            tampered_path = root / f"{label}.pt"
            torch.save(tampered, tampered_path)
            target = _runner(layout, policy_base, iteration=0, gmt_checkpoint_path=gmt_path)
            before = copy.deepcopy(target.alg.policy.residual_actor.state_dict())
            _expect_error(lambda: checkpointing.load_runner(target, str(tampered_path)), message)
            for name, value in before.items():
                torch.testing.assert_close(target.alg.policy.residual_actor.state_dict()[name], value)
            assert not hasattr(target, "_frontres_last_loaded_checkpoint_path")

        for label, mutate, message in (
            (
                "missing-value-normalizer-state",
                lambda item: item.pop("frontres_critic_value_normalizer_state_dict"),
                "value-normalizer state is invalid",
            ),
            (
                "nonfinite-value-normalizer-state",
                lambda item: item["frontres_critic_value_normalizer_state_dict"].update(mean=float("nan")),
                "value-normalizer state is invalid",
            ),
            (
                "wrong-value-normalizer-count",
                lambda item: item["frontres_critic_value_normalizer_state_dict"].update(update_count=2),
                "count differs",
            ),
        ):
            tampered = copy.deepcopy(payload)
            mutate(tampered)
            tampered_path = root / f"{label}.pt"
            torch.save(tampered, tampered_path)
            target = _runner(layout, policy_base, iteration=0, gmt_checkpoint_path=gmt_path)
            _expect_error(lambda: checkpointing.load_runner(target, str(tampered_path)), message)
            assert target.alg.frontres_critic_value_normalizer_state.update_count == 0
            assert not hasattr(target, "_frontres_last_loaded_checkpoint_path")

        for label, mutate in (
            (
                "negative-critic-variance",
                lambda state: state["_var"].__setitem__((0, 0), -1.0),
            ),
            (
                "inconsistent-critic-std",
                lambda state: state["_std"].__setitem__((0, 0), 1.0),
            ),
        ):
            tampered = copy.deepcopy(payload)
            mutate(tampered["privileged_obs_norm_state_dict"])
            tampered_path = root / f"{label}.pt"
            torch.save(tampered, tampered_path)
            target = _runner(layout, policy_base, iteration=0, gmt_checkpoint_path=gmt_path)
            _expect_error(
                lambda: checkpointing.load_runner(target, str(tampered_path)),
                "variance/std state is invalid",
            )
            assert not hasattr(target, "_frontres_last_loaded_checkpoint_path")

        atomic_path = root / "atomic.pt"
        atomic_path.write_bytes(b"last committed v11")
        real_save = checkpointing.torch.save

        def _failing_save(_payload, target):
            Path(target).write_bytes(b"partial v11")
            raise OSError("injected v11 serialization failure")

        checkpointing.torch.save = _failing_save
        try:
            try:
                checkpointing.save_runner(source, str(atomic_path))
            except OSError:
                pass
            else:
                raise AssertionError("atomic checkpoint test must inject a save failure")
        finally:
            checkpointing.torch.save = real_save
        assert atomic_path.read_bytes() == b"last committed v11"
        assert not tuple(root.glob("atomic.pt.tmp-*"))

    assert checkpointing._V015_HSL_CHECKPOINT_FORMAT == "frontres-v017-hsl-proposal-v2"
    print("frontres_v016_checkpoint_contract: v16 dual-score replay round-trip and legacy reject", flush=True)


if __name__ == "__main__":
    main()
