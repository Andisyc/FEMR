#!/usr/bin/env python3
"""CPU-only S3 contract for v015 checkpoint identity and transaction atomicity."""

from __future__ import annotations

import copy
import importlib.util
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[4]
RSL_ROOT = ROOT / "source" / "rsl_rl" / "rsl_rl"
CHECKPOINT_PATH = RSL_ROOT / "runners" / "frontres_checkpointing.py"
LAYOUT_PATH = RSL_ROOT / "modules" / "frontres_observation_layout.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


def _load_owners():
    rsl_rl = _package("rsl_rl")
    modules = _package("rsl_rl.modules")
    rsl_rl.modules = modules

    class _FrontRESActorCritic(torch.nn.Module):
        pass

    class _ResidualActorCritic(torch.nn.Module):
        pass

    modules.FrontRESActorCritic = _FrontRESActorCritic
    modules.ResidualActorCritic = _ResidualActorCritic
    layout = _load("rsl_rl.modules.frontres_observation_layout", LAYOUT_PATH)
    modules.frontres_observation_layout = layout
    checkpointing = _load("frontres_v015_checkpointing_contract", CHECKPOINT_PATH)
    return layout, checkpointing, _FrontRESActorCritic


class _Normalizer(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1, dim))
        self.register_buffer("_std", torch.ones(1, dim))
        self.register_buffer("_var", torch.ones(1, dim))
        self.count = torch.tensor(1.0)
        self.until = 1.0e8

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self._mean) / (self._std + 1.0e-8)


class _TrackingSampler:
    def __init__(self, value: int = 0) -> None:
        self.value = int(value)
        self.loaded = False

    def state_dict(self) -> dict[str, int]:
        return {"value": self.value}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.loaded = True
        self.value = int(state["value"])


def _policy(policy_cls, *, actor_dim: int, prefix_dim: int):
    class _Policy(policy_cls):
        def __init__(self) -> None:
            super().__init__()
            self.residual_actor = torch.nn.Linear(actor_dim, 6)
            self.critic = torch.nn.Linear(actor_dim, 1)
            self.std = torch.nn.Parameter(torch.full((6,), 0.7))
            self.num_actor_obs = actor_dim
            self.num_frontres_obs = prefix_dim

    return _Policy()


def _runner(layout_module, policy_cls, *, offsets=(1, 2), iteration: int = 7):
    layout = layout_module.resolve_frontres_future_intent_layout(
        offsets, layout_module.FRONTRES_FUTURE_INTENT_LAYOUT_VERSION
    )
    gmt_dim = 770
    base_prefix_dim = 100
    prefix_dim = base_prefix_dim + layout.actor_tail_dim
    policy = _policy(policy_cls, actor_dim=prefix_dim + gmt_dim, prefix_dim=prefix_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-3)
    alg = SimpleNamespace(
        policy=policy,
        optimizer=optimizer,
        learning_rate=1.0e-3,
        frontres_training_objective="segment_replay_hrl",
        frontres_v015_formal_transaction_enabled=True,
        frontres_segment_advantage_normalization="grouped_scale_only",
        frontres_segment_critic_warmup_iterations=0,
        frontres_segment_actor_warmup_iterations=0,
        frontres_formal_runtime_audit=False,
        frontres_segment_offline_eval_only=False,
        frontres_segment_sequence_offline_eval_only=False,
        rnd=None,
    )
    runner = SimpleNamespace(
        alg=alg,
        current_learning_iteration=iteration,
        cfg={"is_full_resume": True},
        alg_cfg={"learning_rate": 1.0e-3},
        policy_cfg={"init_noise_std": 1.0, "noise_std_type": "scalar"},
        empirical_normalization=True,
        training_type="frontres",
        logger_type="",
        disable_logs=True,
        writer=None,
        device=torch.device("cpu"),
        _frontres_future_intent_layout=layout,
        _frontres_future_intent_actor_context_dim=layout.actor_tail_dim,
        _frontres_gmt_obs_dim=gmt_dim,
        _frontres_extra_mean=torch.arange(prefix_dim, dtype=torch.float32).reshape(1, prefix_dim),
        _frontres_extra_std=torch.arange(1, prefix_dim + 1, dtype=torch.float32).reshape(1, prefix_dim),
        _frontres_extra_stats_layout_version=None,
        _frontres_extra_normalizer=None,
        obs_normalizer=_Normalizer(gmt_dim),
        privileged_obs_normalizer=_Normalizer(4),
        _frontres_segment_sampler=_TrackingSampler(value=17),
    )
    return runner


def _expect_error(fn, text: str) -> None:
    try:
        fn()
    except RuntimeError as exc:
        assert text in str(exc), str(exc)
        return
    raise AssertionError("expected RuntimeError")


def _committed_state() -> dict[str, object]:
    return {
        "state": "committed",
        "receipt": {
            "transaction_id": "tx-v015-s3",
            "policy_snapshot_id": "tx-v015-s3:pi-0123456789abcdef",
            "plan_identity_hash": "a" * 64,
            "scenario_identity_hash": "b" * 64,
            "expected_policy_row_count": 4,
            "collected_policy_attempt_count": 4,
            "valid_policy_row_count": 4,
            "optimizer_step_before": 9,
            "optimizer_step_after": 10,
            "optimizer_step_delta": 1,
        },
    }


def _saved_payload(path: Path) -> dict:
    return torch.load(path, weights_only=False)


def _assert_unmutated(runner, actor_before: torch.Tensor) -> None:
    torch.testing.assert_close(runner.alg.policy.residual_actor.weight.detach(), actor_before)
    torch.testing.assert_close(runner.obs_normalizer._mean, torch.zeros_like(runner.obs_normalizer._mean))
    torch.testing.assert_close(runner.obs_normalizer._std, torch.ones_like(runner.obs_normalizer._std))
    assert runner._frontres_segment_sampler.loaded is False
    assert not hasattr(runner, "_frontres_last_loaded_checkpoint_path")


def test_t_checkpoint_layout_and_committed_receipt(layout_module, checkpointing, policy_cls) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "v015_committed.pt"
        source = _runner(layout_module, policy_cls)
        source.obs_normalizer._mean.fill_(123.0)
        source.obs_normalizer._std.fill_(2.0)
        source._frontres_v015_checkpoint_transaction_state = _committed_state()
        checkpointing.save_runner(source, str(path))
        payload = _saved_payload(path)
        identity = payload["frontres_v015_checkpoint_identity"]
        assert identity["format"] == "frontres-v015-checkpoint-v2"
        assert identity["gain_contract_id"] == "FRS-GAIN-v003"
        assert identity["future_intent_layout"]["future_offsets"] == (1, 2)
        assert identity["future_intent_layout"]["actor_tail_dim"] == 58
        assert identity["future_intent_layout"]["environment_obs_dim"] == 870
        assert identity["future_intent_layout"]["current_frontres_prefix_dim"] == 100
        assert identity["future_intent_layout"]["actor_dim"] == 928
        assert identity["future_intent_layout"]["prefix_dim"] == 158
        assert identity["future_intent_layout"]["gmt_dim"] == 770
        assert identity["normalizer"]["prefix_layout_version"] == layout_module.FRONTRES_FUTURE_INTENT_LAYOUT_VERSION
        assert identity["normalizer"]["prefix_dim"] == 158
        assert identity["normalizer"]["combined_dim"] == 928
        assert len(identity["normalizer"]["prefix_stats_fingerprint"]) == 64
        assert identity["grouped_loss"]["advantage_normalization"] == "grouped_scale_only"
        assert identity["transaction"] == _committed_state()
        assert "clean_continuation" not in repr(identity)
        assert "intent_q29" not in repr(identity)
        assert tuple(payload["obs_norm_state_dict"]["_mean"].shape) == (1, 928)
        torch.testing.assert_close(
            payload["obs_norm_state_dict"]["_mean"][..., :158],
            source._frontres_extra_mean,
        )
        torch.testing.assert_close(
            payload["obs_norm_state_dict"]["_mean"][..., 158:],
            torch.full((1, 770), 123.0),
        )

        resumed = _runner(layout_module, policy_cls, iteration=0)
        resumed.obs_normalizer._mean.fill_(-321.0)
        resumed.obs_normalizer._std.fill_(3.0)
        checkpointing.load_runner(resumed, str(path), load_optimizer=False)
        assert resumed.current_learning_iteration == 7
        assert resumed._frontres_segment_sampler.loaded is True
        assert resumed._frontres_extra_stats_layout_version == layout_module.FRONTRES_FUTURE_INTENT_LAYOUT_VERSION
        assert resumed._frontres_v015_checkpoint_transaction_state == {"state": "idle"}
        assert resumed._frontres_v015_last_committed_transaction_receipt == _committed_state()["receipt"]
        torch.testing.assert_close(resumed._frontres_extra_mean, source._frontres_extra_mean)
        torch.testing.assert_close(resumed._frontres_extra_std, source._frontres_extra_std)
        torch.testing.assert_close(resumed.obs_normalizer._mean, torch.full((1, 770), -321.0))
        torch.testing.assert_close(resumed.obs_normalizer._std, torch.full((1, 770), 3.0))
        print("[T-checkpoint/T-layout/T-prefix-stats/T-commit-receipt] 928D layout, full 158D prefix fingerprint, frozen 770D GMT suffix, and metadata-only receipt round-trip", flush=True)


def test_t_v015_envelope_distinguishes_completed_hsl_history(layout_module, checkpointing, policy_cls) -> None:
    """合法 v015 Stage-3 envelope 不得把 completed-HSL history 误判为 legacy HSL checkpoint."""

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "v015_after_hsl.pt"
        source = _runner(layout_module, policy_cls)
        source._frontres_warmup_complete = True
        source._frontres_v015_checkpoint_transaction_state = _committed_state()
        checkpointing.save_runner(source, str(path))
        resumed = _runner(layout_module, policy_cls, iteration=0)
        checkpointing.load_runner(resumed, str(path), load_optimizer=False)
        assert resumed._frontres_warmup_complete is True
        assert resumed._frontres_v015_checkpoint_transaction_state == {"state": "idle"}
        print("[T-v015-hsl-history] valid v015 envelope accepts completed-HSL history without accepting a legacy HSL checkpoint", flush=True)


def test_t_resume_rejects_layout_legacy_and_normalizer_before_mutation(layout_module, checkpointing, policy_cls) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        source_path = Path(tmp) / "source.pt"
        checkpointing.save_runner(_runner(layout_module, policy_cls), str(source_path))
        payload = _saved_payload(source_path)

        mismatch = _runner(layout_module, policy_cls, offsets=(1, 3), iteration=0)
        actor_before = mismatch.alg.policy.residual_actor.weight.detach().clone()
        _expect_error(lambda: checkpointing.load_runner(mismatch, str(source_path), load_optimizer=False), "future_offsets=(1, 2)")
        _assert_unmutated(mismatch, actor_before)

        old_v1_payload = copy.deepcopy(payload)
        old_v1_payload["frontres_v015_checkpoint_identity"]["format"] = "frontres-v015-checkpoint-v1"
        old_v1_path = Path(tmp) / "old_v1.pt"
        torch.save(old_v1_payload, old_v1_path)
        old_v1 = _runner(layout_module, policy_cls, iteration=0)
        actor_before = old_v1.alg.policy.residual_actor.weight.detach().clone()
        _expect_error(lambda: checkpointing.load_runner(old_v1, str(old_v1_path), load_optimizer=False), "contract or format identity")
        _assert_unmutated(old_v1, actor_before)

        legacy_payload = copy.deepcopy(payload)
        del legacy_payload["frontres_v015_checkpoint_identity"]
        legacy_payload["obs_norm_state_dict"]["_mean"] = torch.zeros(1, 7 + 2 * 65)
        legacy_payload["obs_norm_state_dict"]["_std"] = torch.ones(1, 7 + 2 * 65)
        legacy_path = Path(tmp) / "legacy_65d.pt"
        torch.save(legacy_payload, legacy_path)
        legacy = _runner(layout_module, policy_cls, iteration=0)
        actor_before = legacy.alg.policy.residual_actor.weight.detach().clone()
        _expect_error(lambda: checkpointing.load_runner(legacy, str(legacy_path), load_optimizer=False), "frontres_v015_checkpoint_identity")
        _assert_unmutated(legacy, actor_before)

        tampered_payload = copy.deepcopy(payload)
        tampered_payload["obs_norm_state_dict"]["_mean"][..., 157] += 1.0
        tampered_path = Path(tmp) / "tampered_stats.pt"
        torch.save(tampered_payload, tampered_path)
        tampered = _runner(layout_module, policy_cls, iteration=0)
        actor_before = tampered.alg.policy.residual_actor.weight.detach().clone()
        _expect_error(lambda: checkpointing.load_runner(tampered, str(tampered_path), load_optimizer=False), "statistics do not match")
        _assert_unmutated(tampered, actor_before)
        print("[T-resume/T-legacy-reject/T-prefix-stats] H mismatch, old v1, old [H,65], and full-prefix tampering reject before mutable restore", flush=True)


def test_t_zero_and_full_observation_prefix_reject_before_save(layout_module, checkpointing, policy_cls) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        for label, prefix_dim in (("zero", 0), ("full_928", 928)):
            path = Path(tmp) / f"{label}.pt"
            runner = _runner(layout_module, policy_cls)
            runner.alg.policy.num_frontres_obs = prefix_dim
            _expect_error(lambda: checkpointing.save_runner(runner, str(path)), "actor layout")
            assert not path.exists()
        print("[T-legacy-zero-reject] num_frontres_obs=0 and full-928 actor visibility reject before checkpoint write", flush=True)


def test_t_atomicity_rejects_partial_save_and_resume(layout_module, checkpointing, policy_cls) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        partial_path = Path(tmp) / "partial_save.pt"
        collecting = _runner(layout_module, policy_cls)
        collecting._frontres_v015_checkpoint_transaction_state = {"state": "collecting", "phase": "provider"}
        _expect_error(lambda: checkpointing.save_runner(collecting, str(partial_path)), "in-flight formal transaction")
        assert not partial_path.exists()

        source_path = Path(tmp) / "source.pt"
        checkpointing.save_runner(_runner(layout_module, policy_cls), str(source_path))
        partial_payload = _saved_payload(source_path)
        partial_payload["frontres_v015_checkpoint_identity"]["transaction"] = {"state": "sealed"}
        partial_resume_path = Path(tmp) / "partial_resume.pt"
        torch.save(partial_payload, partial_resume_path)
        resumed = _runner(layout_module, policy_cls, iteration=0)
        actor_before = resumed.alg.policy.residual_actor.weight.detach().clone()
        _expect_error(lambda: checkpointing.load_runner(resumed, str(partial_resume_path), load_optimizer=False), "partial, failed, or malformed")
        _assert_unmutated(resumed, actor_before)
        print("[T-atomicity] collecting save and sealed resume both fail closed without a later update path", flush=True)


def main() -> None:
    layout_module, checkpointing, policy_cls = _load_owners()
    test_t_checkpoint_layout_and_committed_receipt(layout_module, checkpointing, policy_cls)
    test_t_v015_envelope_distinguishes_completed_hsl_history(layout_module, checkpointing, policy_cls)
    test_t_resume_rejects_layout_legacy_and_normalizer_before_mutation(layout_module, checkpointing, policy_cls)
    test_t_zero_and_full_observation_prefix_reject_before_save(layout_module, checkpointing, policy_cls)
    test_t_atomicity_rejects_partial_save_and_resume(layout_module, checkpointing, policy_cls)
    print("frontres_v015_checkpoint_resume_contract: ok", flush=True)


if __name__ == "__main__":
    main()
