#!/usr/bin/env python3
"""Deterministic TEST-16 contracts for strict checkpoint-v11 persistence."""

from __future__ import annotations

import copy
import contextlib
import io
import tempfile
from pathlib import Path

import torch

from frontres_v015_checkpoint_resume_contract import (
    _FrozenNormalizer,
    _load_owners,
    _receipt as _legacy_receipt,
    _runner as _legacy_runner,
)


def _runner(layout, policy_base, *, iteration: int, gmt_checkpoint_path: Path):
    runner = _legacy_runner(layout, policy_base, iteration=iteration, gmt_checkpoint_path=gmt_checkpoint_path)
    policy = runner.alg.policy
    policy.critic = torch.nn.Linear(347, 1)
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
    runner.alg.frontres_method_contract_id = "FRS-METHOD-v018"
    runner.alg.frontres_optimization_contract_id = "FRS-PPO-v006"
    runner.alg.frontres_training_contract_id = "FRS-TRAIN-v016"
    runner.alg.frontres_critic_value_kind = "state_value"
    runner.alg.frontres_critic_input_dim = 347
    runner.alg.frontres_critic_action_conditioned = False
    runner.alg.frontres_critic_target_id = "segment-exact-m-mean-v1"
    runner.alg.frontres_gradient_clip_identity = "separate-actor-critic-v1"
    runner.alg.max_grad_norm = 0.5
    runner._frontres_critic_observation_dim = 347
    runner.alg.frontres_formal_runtime_audit = True
    runner.empirical_normalization = True
    runner.obs_normalizer = policy.gmt_normalizer
    runner.privileged_obs_normalizer = _FrozenNormalizer(347)
    runner._frontres_extra_mean = torch.zeros(1, 158)
    runner._frontres_extra_std = torch.ones(1, 158)
    runner._frontres_extra_normalizer = None
    runner._frontres_extra_stats_layout_version = runner._frontres_future_intent_layout.version
    return runner


def _receipt(checkpointing, *, training_iteration: int) -> dict[str, object]:
    value = _legacy_receipt(checkpointing, training_iteration=training_iteration)
    receipt = value["receipt"]
    receipt["method_contract_id"] = "FRS-METHOD-v018"
    receipt["optimization_contract_id"] = "FRS-PPO-v006"
    receipt["training_contract_id"] = "FRS-TRAIN-v016"
    return value


def _expect_error(call, text: str) -> None:
    try:
        call()
    except RuntimeError as exc:
        assert text.lower() in str(exc).lower(), str(exc)
        return
    raise AssertionError("expected checkpoint-v11 rejection")


def main() -> None:
    layout, checkpointing, policy_base = _load_owners()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        gmt_path = root / "gmt.pt"
        gmt_path.write_bytes(b"frozen GMT artifact v11")
        path = root / "model_3.pt"
        source = _runner(layout, policy_base, iteration=3, gmt_checkpoint_path=gmt_path)
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
        assert identity["format"] == "frontres-v017-checkpoint-v11"
        assert identity["method_contract_id"] == "FRS-METHOD-v018"
        assert identity["optimization_contract_id"] == "FRS-PPO-v006"
        assert identity["training_contract_id"] == "FRS-TRAIN-v016"
        assert identity["critic"] == {
            "value_kind": "state_value",
            "input_dim": 347,
            "action_conditioned": False,
            "target_id": "segment-exact-m-mean-v1",
        }
        assert identity["gradient_clip"] == {
            "identity": "separate-actor-critic-v1",
            "max_norm": 0.5,
        }
        assert payload["privileged_obs_norm_state_dict"]["_var"][0, 0].item() == 0.0
        assert payload["privileged_obs_norm_state_dict"]["_std"][0, 0].item() == 0.0

        fresh = _runner(layout, policy_base, iteration=0, gmt_checkpoint_path=gmt_path)
        checkpointing.load_runner(fresh, str(path), load_optimizer=True)
        for name, value in actor_state.items():
            torch.testing.assert_close(fresh.alg.policy.residual_actor.state_dict()[name], value)
        for name, value in critic_state.items():
            torch.testing.assert_close(fresh.alg.policy.critic.state_dict()[name], value)

        legacy_quality = copy.deepcopy(payload)
        legacy_identity = legacy_quality["frontres_v015_checkpoint_identity"]
        legacy_identity.update(
            format="frontres-v017-checkpoint-v10",
            method_contract_id="FRS-METHOD-v017",
            optimization_contract_id="FRS-PPO-v005",
            training_contract_id="FRS-TRAIN-v015",
        )
        legacy_identity.pop("critic")
        legacy_identity.pop("gradient_clip")
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
            optimization_contract_id="FRS-PPO-v005",
            training_contract_id="FRS-TRAIN-v015",
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
    print("frontres_v016_checkpoint_contract: v11 strict round-trip and v10 reject", flush=True)


if __name__ == "__main__":
    main()
