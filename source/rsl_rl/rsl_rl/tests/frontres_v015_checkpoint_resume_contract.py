#!/usr/bin/env python3
"""CPU-only S3 contract for strict TRAIN-v014 checkpoint-v9 persistence."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import torch

from frontres_contract_imports import install_frontres_contract_packages


ROOT = Path(__file__).resolve().parents[4]
RSL_ROOT = ROOT / "source" / "rsl_rl" / "rsl_rl"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_owners():
    rsl_rl = types.ModuleType("rsl_rl")
    rsl_rl.__path__ = []
    modules = types.ModuleType("rsl_rl.modules")
    modules.__path__ = []
    runners = types.ModuleType("rsl_rl.runners")
    runners.__path__ = []
    frontres = types.ModuleType("rsl_rl.frontres")
    frontres.__path__ = []
    rsl_rl.modules = modules
    rsl_rl.runners = runners
    rsl_rl.frontres = frontres
    sys.modules.update(
        {
            "rsl_rl": rsl_rl,
            "rsl_rl.modules": modules,
            "rsl_rl.runners": runners,
            "rsl_rl.frontres": frontres,
        }
    )
    install_frontres_contract_packages(RSL_ROOT)

    class _FrontRESActorCritic(torch.nn.Module):
        pass

    modules.FrontRESActorCritic = _FrontRESActorCritic
    modules.ResidualActorCritic = _FrontRESActorCritic
    layout = _load(
        "rsl_rl.modules.frontres_observation_layout",
        RSL_ROOT / "modules" / "frontres_observation_layout.py",
    )
    modules.frontres_observation_layout = layout
    checkpointing = _load(
        "rsl_rl.runners.frontres_checkpointing",
        RSL_ROOT / "runners" / "frontres_checkpointing.py",
    )
    return layout, checkpointing, _FrontRESActorCritic


class _Sampler:
    def __init__(self, value: int) -> None:
        self.value = value
        self.loaded = False

    def state_dict(self) -> dict[str, int]:
        return {"value": self.value}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.value = int(state["value"])
        self.loaded = True


class _UnsupportedCheckpointObject:
    pass


class _FrozenNormalizer(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1, dim))
        self.register_buffer("_var", torch.ones(1, dim))
        self.register_buffer("_std", torch.ones(1, dim))
        self.register_buffer("count", torch.ones(1, dtype=torch.long))


SCHEDULE = (
    (8, 2, 200, 500, 1300, "lower-k8", 0.5, "linear-joint-v1", 1300, 2.381),
    (16, 3, 300, 300, 900, "lower-k16", 0.6, "linear-joint-v1", 900, 2.381),
    (32, 4, 400, 300, 625, "lower-k32", 0.7, "linear-joint-v1", 625, 2.381),
)


def _fingerprint() -> str:
    return hashlib.sha256(json.dumps(SCHEDULE, separators=(",", ":")).encode("ascii")).hexdigest()


def _runner(layout_module, policy_base, *, iteration: int, gmt_checkpoint_path: Path):
    layout = layout_module.resolve_frontres_future_intent_layout(
        (1, 2), layout_module.FRONTRES_FUTURE_INTENT_LAYOUT_VERSION
    )

    class _Policy(policy_base):
        def __init__(self) -> None:
            super().__init__()
            self.residual_actor = torch.nn.Linear(158, 6)
            self.critic = torch.nn.Linear(928, 1)
            self.std = torch.nn.Parameter(torch.full((6,), 0.7))
            self.num_actor_obs = 928
            self.num_frontres_obs = 158
            self.num_task_corrections = 6
            self.gmt_policy_obs_dim = 770
            self.gmt_policy = torch.nn.Linear(770, 29)
            self.gmt_normalizer = _FrozenNormalizer(770)
            self.gmt_policy.eval()
            self.gmt_normalizer.eval()
            for parameter in self.gmt_policy.parameters():
                parameter.requires_grad_(False)

    policy = _Policy()
    optimizer = torch.optim.Adam(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=1.0e-3,
    )
    alg = SimpleNamespace(
        policy=policy,
        optimizer=optimizer,
        frontres_training_objective="segment_replay_hrl",
        frontres_formal_transaction_enabled=True,
        frontres_segment_advantage_normalization="grouped_scale_only",
        frontres_segment_k_curriculum=SCHEDULE,
        frontres_segment_k_curriculum_fingerprint=_fingerprint(),
        frontres_segment_max_horizon_k=32,
        frontres_gain_beta=0.02,
        frontres_formal_runtime_audit=False,
        rnd=None,
    )
    return SimpleNamespace(
        alg=alg,
        current_learning_iteration=iteration,
        cfg={"is_full_resume": True},
        alg_cfg={"learning_rate": 1.0e-3},
        policy_cfg={
            "init_noise_std": 1.0,
            "noise_std_type": "scalar",
            "gmt_checkpoint_path": str(gmt_checkpoint_path),
        },
        empirical_normalization=False,
        training_type="frontres",
        logger_type="",
        disable_logs=True,
        writer=None,
        device=torch.device("cpu"),
        _frontres_future_intent_layout=layout,
        _frontres_future_intent_actor_context_dim=58,
        _frontres_gmt_obs_dim=770,
        _frontres_segment_sampler=_Sampler(17),
    )


def _receipt(checkpointing, *, training_iteration: int = 2) -> dict[str, object]:
    identity = checkpointing.resolve_frontres_k_stage_identity(
        schedule=SCHEDULE,
        committed_update_iteration=training_iteration,
        max_horizon_k=32,
    )
    policy_rows = 2 * identity.active_m
    return {
        "state": "committed",
        "receipt": {
            "method_contract_id": "FRS-METHOD-v017",
            "gain_contract_id": "FRS-GAIN-v007",
            "optimization_contract_id": "FRS-PPO-v005",
            "training_contract_id": "FRS-TRAIN-v014",
            "scalar_target_id": "clean-anchored-recovery-aware-gain-v1",
            "physics_schema_id": "clean-anchored-contact-zmp-survival-v1",
            "grouped_schema_id": "grouped-all-attempt-scalar-v1",
            "transaction_id": "tx-v017-checkpoint",
            "policy_snapshot_id": "tx-v017-checkpoint:pi-0123456789abcdef",
            "plan_identity_hash": "a" * 64,
            "scenario_identity_hash": "b" * 64,
            "expected_policy_row_count": policy_rows,
            "collected_policy_attempt_count": policy_rows,
            "valid_policy_row_count": policy_rows,
            "optimizer_step_before": 9,
            "optimizer_step_after": 10,
            "optimizer_step_delta": 1,
            "curriculum_fingerprint": _fingerprint(),
            "k_stage_index": identity.stage_index,
            "active_k": identity.active_k,
            "active_m": identity.active_m,
            "selected_segment_count": 2,
            "policy_row_count": policy_rows,
            "role_row_count": 2 * policy_rows,
            "k_stage_iteration": identity.stage_iteration,
            "training_iteration": identity.absolute_iteration,
            "dr_stage_fingerprint": identity.dr_stage_fingerprint,
            "dr_progress": identity.dr_progress,
            "d_cap": identity.d_cap,
        },
    }


def _expect_error(call, text: str) -> None:
    try:
        call()
    except RuntimeError as exc:
        assert text in str(exc), str(exc)
        return
    raise AssertionError("expected RuntimeError")


def main() -> None:
    layout, checkpointing, policy_base = _load_owners()
    with tempfile.TemporaryDirectory() as directory:
        gmt_path = Path(directory) / "gmt.pt"
        gmt_path.write_bytes(b"frozen GMT artifact A")
        other_gmt_path = Path(directory) / "other-gmt.pt"
        other_gmt_path.write_bytes(b"shape-compatible frozen GMT artifact B")
        path = Path(directory) / "model_3.pt"
        source = _runner(layout, policy_base, iteration=203, gmt_checkpoint_path=gmt_path)
        receipt = _receipt(checkpointing, training_iteration=202)
        source._frontres_checkpoint_transaction_state = receipt
        source.alg.optimizer.zero_grad()
        source.alg.policy.residual_actor.weight.sum().backward()
        source.alg.optimizer.step()
        actor_state = copy.deepcopy(source.alg.policy.residual_actor.state_dict())
        critic_state = copy.deepcopy(source.alg.policy.critic.state_dict())
        checkpointing.save_runner(source, str(path))

        payload = torch.load(path, weights_only=False)
        identity = payload["frontres_v015_checkpoint_identity"]
        assert set(payload["frontres_v013_rng_state"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
        assert identity["format"] == "frontres-v017-checkpoint-v9"
        assert identity["method_contract_id"] == "FRS-METHOD-v017"
        assert identity["gain_contract_id"] == "FRS-GAIN-v007"
        assert identity["optimization_contract_id"] == "FRS-PPO-v005"
        assert identity["training_contract_id"] == "FRS-TRAIN-v014"
        assert identity["dr_curriculum_schema_id"] == "nested-k-dr-four-class-v1"
        assert identity["scalar_target_id"] == "clean-anchored-recovery-aware-gain-v1"
        assert identity["grouped_schema_id"] == "grouped-all-attempt-scalar-v1"
        assert identity["action"] == {
            "kind": "delta_se3",
            "dim": 6,
            "semantics": "direct-world-full6-v1",
        }
        assert identity["gain"] == {"beta": 0.02}
        assert identity["gmt"]["checkpoint_sha256"] == hashlib.sha256(gmt_path.read_bytes()).hexdigest()
        assert identity["gmt"]["normalizer_dim"] == 770
        assert len(identity["gmt"]["normalizer_fingerprint"]) == 64
        assert identity["future_intent_layout"]["actor_dim"] == 928
        assert identity["future_intent_layout"]["prefix_dim"] == 158
        assert identity["future_intent_layout"]["gmt_dim"] == 770
        assert identity["transaction"] == receipt
        assert identity["curriculum"]["phase"] == "actor_ramp"
        assert "projection" not in repr(identity).lower()

        fresh = _runner(layout, policy_base, iteration=0, gmt_checkpoint_path=gmt_path)
        checkpointing.load_runner(fresh, str(path), load_optimizer=True)
        assert fresh.current_learning_iteration == 203
        assert fresh._frontres_segment_sampler.loaded
        assert fresh._frontres_last_committed_transaction_receipt == receipt["receipt"]
        for name, value in actor_state.items():
            torch.testing.assert_close(fresh.alg.policy.residual_actor.state_dict()[name], value)
        for name, value in critic_state.items():
            torch.testing.assert_close(fresh.alg.policy.critic.state_dict()[name], value)

        for label, mutate, message in (
            ("v8", lambda item: item.update(format="frontres-v015-checkpoint-v8"), "contract or format"),
            ("v7", lambda item: item.update(format="frontres-v015-checkpoint-v7"), "contract or format"),
            ("g-k", lambda item: item["curriculum"].update(g_K=2.0), "stage/phase/DR"),
            ("old-phase", lambda item: item["curriculum"].update(phase="actor_warmup"), "stage/phase/DR"),
            ("beta", lambda item: item["gain"].update(beta=0.2), "beta identity"),
            ("partial", lambda item: item.update(transaction={"state": "sealed"}), "partial"),
        ):
            tampered = copy.deepcopy(payload)
            mutate(tampered["frontres_v015_checkpoint_identity"])
            tampered_path = Path(directory) / f"{label}.pt"
            torch.save(tampered, tampered_path)
            target = _runner(layout, policy_base, iteration=0, gmt_checkpoint_path=gmt_path)
            before = copy.deepcopy(target.alg.policy.residual_actor.state_dict())
            _expect_error(lambda: checkpointing.load_runner(target, str(tampered_path)), message)
            for name, value in before.items():
                torch.testing.assert_close(target.alg.policy.residual_actor.state_dict()[name], value)
            assert not hasattr(target, "_frontres_last_loaded_checkpoint_path")

        different_gmt = _runner(
            layout,
            policy_base,
            iteration=0,
            gmt_checkpoint_path=other_gmt_path,
        )
        actor_before = copy.deepcopy(different_gmt.alg.policy.residual_actor.state_dict())
        optimizer_before = copy.deepcopy(different_gmt.alg.optimizer.state_dict())
        sampler_before = different_gmt._frontres_segment_sampler.value
        _expect_error(
            lambda: checkpointing.load_runner(different_gmt, str(path)),
            "frozen GMT artifact",
        )
        for name, value in actor_before.items():
            torch.testing.assert_close(different_gmt.alg.policy.residual_actor.state_dict()[name], value)
        assert different_gmt.alg.optimizer.state_dict() == optimizer_before
        assert different_gmt._frontres_segment_sampler.value == sampler_before
        assert not hasattr(different_gmt, "_frontres_last_loaded_checkpoint_path")

        unsafe_path = Path(directory) / "unsupported-object.pt"
        torch.save({"foreign": _UnsupportedCheckpointObject()}, unsafe_path)
        target = _runner(layout, policy_base, iteration=0, gmt_checkpoint_path=gmt_path)
        actor_before = copy.deepcopy(target.alg.policy.residual_actor.state_dict())
        optimizer_before = copy.deepcopy(target.alg.optimizer.state_dict())
        sampler_before = target._frontres_segment_sampler.value
        _expect_error(
            lambda: checkpointing.load_runner(target, str(unsafe_path)),
            "restricted load failed",
        )
        for name, value in actor_before.items():
            torch.testing.assert_close(target.alg.policy.residual_actor.state_dict()[name], value)
        assert target.alg.optimizer.state_dict() == optimizer_before
        assert target._frontres_segment_sampler.value == sampler_before
        assert not hasattr(target, "_frontres_last_loaded_checkpoint_path")

        collecting = _runner(layout, policy_base, iteration=3, gmt_checkpoint_path=gmt_path)
        collecting._frontres_checkpoint_transaction_state = {"state": "collecting"}
        _expect_error(
            lambda: checkpointing.save_runner(collecting, str(Path(directory) / "partial-save.pt")),
            "in-flight formal transaction",
        )
        atomic_path = Path(directory) / "atomic.pt"
        atomic_path.write_bytes(b"last committed checkpoint")
        real_save = checkpointing.torch.save
        def _failing_save(_payload, target):
            Path(target).write_bytes(b"partial temp")
            raise OSError("injected serialization failure")
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
        assert atomic_path.read_bytes() == b"last committed checkpoint"
        assert not tuple(Path(directory).glob("atomic.pt.tmp-*"))
    print("frontres_v015_checkpoint_resume_contract: v9 strict safe save/resume ok", flush=True)


if __name__ == "__main__":
    main()
