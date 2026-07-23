#!/usr/bin/env python3
"""CPU-only S3 contract for v015 checkpoint identity and transaction atomicity."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[4]
RSL_ROOT = ROOT / "source" / "rsl_rl" / "rsl_rl"
CHECKPOINT_PATH = RSL_ROOT / "runners" / "frontres_checkpointing.py"
RUNTIME_PATH = RSL_ROOT / "runners" / "frontres_runtime.py"
LAYOUT_PATH = RSL_ROOT / "modules" / "frontres_observation_layout.py"
TRANSACTION_TEST_PATH = RSL_ROOT / "tests" / "frontres_v015_transaction_route_contract.py"


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


def _load_runtime():
    rsl_rl = sys.modules["rsl_rl"]
    frontres = _package("rsl_rl.frontres")
    runners = _package("rsl_rl.runners")
    rsl_rl.frontres = frontres
    rsl_rl.runners = runners
    diagnostics = types.ModuleType("rsl_rl.frontres.runtime_diagnostics")
    diagnostics.maybe_print_frontres_restore_debug = lambda *_args, **_kwargs: None
    sys.modules[diagnostics.__name__] = diagnostics
    frontres.runtime_diagnostics = diagnostics
    return _load("rsl_rl.runners.frontres_runtime_v015_checkpoint_contract", RUNTIME_PATH)


def _transaction_template():
    transaction = _load("frontres_v015_checkpoint_transaction_helper", TRANSACTION_TEST_PATH)
    candidate_contract, owners, live_sampler, _live_update_loop = transaction._load_owners()
    fixture = transaction._build_request(candidate_contract, owners, live_sampler)
    return SimpleNamespace(
        transaction=transaction,
        owners=owners,
        live_sampler=live_sampler,
        request=fixture.request,
    )


class _Normalizer(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1, dim))
        self.register_buffer("_std", torch.ones(1, dim))
        self.register_buffer("_var", torch.ones(1, dim))
        self.register_buffer("count", torch.tensor(1.0))
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


class _TrackingAdam(torch.optim.Adam):
    def __init__(self, params) -> None:
        super().__init__(params, lr=1.0e-3)
        self.frontres_v015_step_count = 0

    def step(self, closure=None):
        result = super().step(closure=closure)
        self.frontres_v015_step_count += 1
        return result


def _policy(policy_cls, *, actor_dim: int, prefix_dim: int):
    class _Policy(policy_cls):
        def __init__(self) -> None:
            super().__init__()
            self.residual_actor = torch.nn.Linear(prefix_dim, 6)
            self.critic = torch.nn.Linear(actor_dim, 1)
            self.std = torch.nn.Parameter(torch.full((6,), 0.7))
            self.num_actor_obs = actor_dim
            self.num_frontres_obs = prefix_dim
            self.num_task_corrections = 6
            self.max_delta_pos = 0.3
            self.max_delta_rpy = 0.4

        def evaluate_segment_actions(self, observations: torch.Tensor, actions: torch.Tensor):
            del actions
            return {
                "log_prob": self.residual_actor.weight[0, 0] * observations[:, 0],
                "value": self.critic.weight[0, 0] * observations[:, 1],
                "entropy": torch.zeros_like(observations[:, 0]),
            }

    return _Policy()


_V009_SCHEDULE = ((3, 1, 1, 2), (4, 1, 1, 0))


def _v009_schedule_fingerprint() -> str:
    return hashlib.sha256(json.dumps(_V009_SCHEDULE, separators=(",", ":")).encode("ascii")).hexdigest()


def _runner(layout_module, policy_cls, *, offsets=(1, 2), iteration: int = 3):
    layout = layout_module.resolve_frontres_future_intent_layout(
        offsets, layout_module.FRONTRES_FUTURE_INTENT_LAYOUT_VERSION
    )
    gmt_dim = 770
    base_prefix_dim = 100
    prefix_dim = base_prefix_dim + layout.actor_tail_dim
    policy = _policy(policy_cls, actor_dim=prefix_dim + gmt_dim, prefix_dim=prefix_dim)
    optimizer = _TrackingAdam(policy.parameters())
    alg = SimpleNamespace(
        policy=policy,
        optimizer=optimizer,
        learning_rate=1.0e-3,
        frontres_training_objective="segment_replay_hrl",
        frontres_v015_formal_transaction_enabled=True,
        frontres_segment_advantage_normalization="grouped_scale_only",
        frontres_segment_critic_warmup_iterations=2,
        frontres_segment_actor_warmup_iterations=3,
        frontres_segment_k_curriculum=_V009_SCHEDULE,
        frontres_segment_k_curriculum_fingerprint=_v009_schedule_fingerprint(),
        frontres_segment_max_horizon_k=4,
        frontres_future_offsets=(1, 2),
        frontres_future_intent_layout_version=layout.version,
        frontres_hsl_init_enabled=False,
        frontres_hsl_rollout_label_enabled=False,
        lambda_supervised=0.0,
        lambda_supervised_min=0.0,
        clip_param=0.2,
        value_loss_coef=0.0,
        entropy_coef=0.0,
        use_clipped_value_loss=True,
        max_grad_norm=1.0,
        frontres_segment_live_train_enabled=False,
        frontres_segment_live_update_loop_only=False,
        frontres_segment_live_single_update_only=False,
        frontres_formal_runtime_audit=False,
        frontres_method_contract_id="FRS-METHOD-v016",
        frontres_gain_contract_id="FRS-GAIN-v005",
        frontres_optimization_contract_id="FRS-PPO-v004",
        frontres_training_contract_id="FRS-TRAIN-v010",
        frontres_scalar_target_id="paired-intent-minus-repair-v1",
        frontres_constraint_schema_id="contact-phase_zmp-survival-physical-v1",
        frontres_projection_schema_id="grouped-first-order-constraint-projection-v1",
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


class _IntentCommand:
    def __init__(self, batch) -> None:
        self.batch = batch

    def frontres_local_scenario_intent_snapshot(self):
        intent = self.batch.frontres_local_scenario_intent_q29
        provenance = self.batch.frontres_local_scenario_provenance
        batch_size = int(intent.shape[0])
        return {
            "intent_q29": intent.detach().clone(),
            "scenario_ids": tuple(f"g3-s2-scenario-{row}" for row in range(batch_size)),
            "noisy_segment_hashes": tuple(f"g3-s2-noisy-{row}" for row in range(batch_size)),
            "x_t_identities": tuple(f"g3-s2-x-t-{row}" for row in range(batch_size)),
            "roles": ("repair", "noisy")[:batch_size],
            "provenance": tuple(dict(value) for value in provenance),
        }


def _wire_inference_carrier(runner, intent_q29: torch.Tensor) -> dict[str, object]:
    provenance = tuple(
        {
            "carrier_kind": "local_scenario",
            "current_root_artifact_provenance": "noisy_root_artifact_t",
            "intent_q29_provenance": "deployment_noisy_q29",
            "intent_q29_source": "motion_internal_q29",
            "clean_continuation_provenance": "clean_gmt_only",
        }
        for _ in range(int(intent_q29.shape[0]))
    )
    batch = SimpleNamespace(
        frontres_local_scenario_intent_q29=intent_q29.detach().clone(),
        frontres_local_scenario_provenance=provenance,
        frontres_local_scenario_clean_continuation=torch.full(
            (intent_q29.shape[0], 2, 65), 991.0, dtype=torch.float32
        ),
    )
    command = _IntentCommand(batch)
    runner.env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            command_manager=SimpleNamespace(get_term=lambda name: command if name == "motion" else None)
        )
    )
    runner._frontres_segment_live_current_batch = batch
    runner._frontres_extra_stats_layout_version = runner._frontres_future_intent_layout.version
    return command.frontres_local_scenario_intent_snapshot()


def _fresh_inference_trace(runner, runtime, raw_obs: torch.Tensor) -> dict[str, torch.Tensor]:
    combined = runtime.append_frontres_future_intent_context(runner, raw_obs)
    normalized = runtime.apply_obs_normalizer(runner, combined)
    actor_input = normalized[:, : runner.alg.policy.num_frontres_obs]
    raw_proposal = runner.alg.policy.residual_actor(actor_input)
    proposal = torch.cat(
        (
            torch.tanh(raw_proposal[:, :3]) * runner.alg.policy.max_delta_pos,
            torch.tanh(raw_proposal[:, 3:6]) * runner.alg.policy.max_delta_rpy,
        ),
        dim=-1,
    )
    return {
        "combined": combined.detach().clone(),
        "normalized": normalized.detach().clone(),
        "actor_input": actor_input.detach().clone(),
        "proposal": proposal.detach().clone(),
    }


def _bind_semantic_transaction(runner, template):
    snapshot = template.live_sampler.capture_frontres_frozen_policy_snapshot(
        runner,
        transaction_id=template.request.plan.transaction_id,
    )
    batches = []
    for batch in template.request.candidate_batches:
        metadata = replace(
            batch.transaction_metadata,
            transaction_id=snapshot.transaction_id,
            policy_snapshot_id=snapshot.policy_snapshot_id,
        )
        batches.append(replace(batch, transaction_metadata=metadata))
    plan = replace(template.request.plan, snapshot=snapshot)
    return replace(
        template.request,
        plan=plan,
        candidate_batches=tuple(batches),
        policy_evaluator=runner.alg.policy,
    )


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
            "method_contract_id": "FRS-METHOD-v016",
            "gain_contract_id": "FRS-GAIN-v005",
            "optimization_contract_id": "FRS-PPO-v004",
            "training_contract_id": "FRS-TRAIN-v010",
            "scalar_target_id": "paired-intent-minus-repair-v1",
            "constraint_schema_id": "contact-phase_zmp-survival-physical-v1",
            "projection_schema_id": "grouped-first-order-constraint-projection-v1",
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
            "curriculum_fingerprint": _v009_schedule_fingerprint(),
            "k_stage_index": 0,
            "active_k": 3,
            "k_stage_iteration": 2,
            "training_iteration": 2,
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
        assert identity["format"] == "frontres-v015-checkpoint-v5"
        assert identity["training_contract_id"] == "FRS-TRAIN-v010"
        assert identity["gain_contract_id"] == "FRS-GAIN-v005"
        assert identity["optimization_contract_id"] == "FRS-PPO-v004"
        assert identity["method_contract_id"] == "FRS-METHOD-v016"
        assert identity["curriculum"] == {
            "schedule": _V009_SCHEDULE,
            "schedule_fingerprint": _v009_schedule_fingerprint(),
            "k_stage_index": 0,
            "active_k": 3,
            "stage_iteration": 3,
            "absolute_iteration": 3,
            "phase": "joint",
            "phase_iteration": 1,
            "actor_loss_weight": 1.0,
        }
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
        assert identity["constraint_solver"]["persistent_dual_state"] is False
        assert "frontres_gain_config" not in payload
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
        assert resumed.current_learning_iteration == 3
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

        old_v4_payload = copy.deepcopy(payload)
        old_v4_payload["frontres_v015_checkpoint_identity"].update(
            {
                "format": "frontres-v015-checkpoint-v4",
                "method_contract_id": "FRS-METHOD-v015",
                "gain_contract_id": "FRS-GAIN-v004",
                "optimization_contract_id": "FRS-PPO-v003",
                "training_contract_id": "FRS-TRAIN-v009",
            }
        )
        old_v4_path = Path(tmp) / "old_v4.pt"
        torch.save(old_v4_payload, old_v4_path)
        old_v4 = _runner(layout_module, policy_cls, iteration=0)
        actor_before = old_v4.alg.policy.residual_actor.weight.detach().clone()
        _expect_error(lambda: checkpointing.load_runner(old_v4, str(old_v4_path), load_optimizer=False), "contract or format identity")
        _assert_unmutated(old_v4, actor_before)

        tampered_solver_payload = copy.deepcopy(payload)
        tampered_solver_payload["frontres_v015_checkpoint_identity"]["constraint_solver"]["projection_tolerance"] = 0.5
        tampered_solver_path = Path(tmp) / "tampered_solver.pt"
        torch.save(tampered_solver_payload, tampered_solver_path)
        tampered_solver = _runner(layout_module, policy_cls, iteration=0)
        actor_before = tampered_solver.alg.policy.residual_actor.weight.detach().clone()
        _expect_error(lambda: checkpointing.load_runner(tampered_solver, str(tampered_solver_path), load_optimizer=False), "solver identity")
        _assert_unmutated(tampered_solver, actor_before)

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
        print("[T-resume/T-legacy-reject/T-prefix-stats] H mismatch, old v1/v4, solver tamper, old [H,65], and prefix tamper reject pre-mutation", flush=True)


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


def test_t_v010_transition_identity_and_schedule_reject(layout_module, checkpointing, policy_cls) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "v009_transition.pt"
        source = _runner(layout_module, policy_cls, iteration=4)
        committed = _committed_state()
        committed["receipt"].update(
            {
                "k_stage_iteration": 3,
                "training_iteration": 3,
            }
        )
        source._frontres_v015_checkpoint_transaction_state = committed
        checkpointing.save_runner(source, str(path))
        payload = _saved_payload(path)
        identity = payload["frontres_v015_checkpoint_identity"]
        assert identity["format"] == "frontres-v015-checkpoint-v5"
        assert identity["curriculum"]["k_stage_index"] == 1
        assert identity["curriculum"]["active_k"] == 4
        assert identity["curriculum"]["stage_iteration"] == 0
        assert identity["curriculum"]["phase"] == "critic_only"
        assert identity["curriculum"]["actor_loss_weight"] == 0.0

        resumed = _runner(layout_module, policy_cls, iteration=0)
        checkpointing.load_runner(resumed, str(path), load_optimizer=True)
        assert resumed.current_learning_iteration == 4
        assert resumed.alg.optimizer.param_groups[0]["lr"] == source.alg.optimizer.param_groups[0]["lr"]

        mismatched = _runner(layout_module, policy_cls, iteration=0)
        mismatched_schedule = ((3, 1, 1, 3), (4, 1, 1, 0))
        mismatched.alg.frontres_segment_k_curriculum = mismatched_schedule
        mismatched.alg.frontres_segment_k_curriculum_fingerprint = checkpointing.resolve_frontres_k_stage_identity(
            schedule=mismatched_schedule,
            committed_update_iteration=0,
            max_horizon_k=4,
        ).schedule_fingerprint
        actor_before = mismatched.alg.policy.residual_actor.weight.detach().clone()
        _expect_error(
            lambda: checkpointing.load_runner(mismatched, str(path), load_optimizer=False),
            "schedule differs",
        )
        _assert_unmutated(mismatched, actor_before)

        old_v3 = copy.deepcopy(payload)
        old_v3["frontres_v015_checkpoint_identity"]["format"] = "frontres-v015-checkpoint-v3"
        old_v3["frontres_v015_checkpoint_identity"]["training_contract_id"] = "FRS-TRAIN-v008"
        old_v3_path = Path(tmp) / "old_v3.pt"
        torch.save(old_v3, old_v3_path)
        rejected = _runner(layout_module, policy_cls, iteration=0)
        actor_before = rejected.alg.policy.residual_actor.weight.detach().clone()
        _expect_error(
            lambda: checkpointing.load_runner(rejected, str(old_v3_path), load_optimizer=False),
            "contract or format identity",
        )
        _assert_unmutated(rejected, actor_before)
    print("[T-v010-transition/T-schedule-fingerprint/T-v009-reject] exact new-K critic-only resume and pre-mutation rejection", flush=True)


def test_t_committed_save_to_fresh_inference_equality(
    layout_module,
    checkpointing,
    policy_cls,
    runtime,
    transaction_template,
) -> None:
    """Connect one real committed transaction to strict save/load and 158D inference."""

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "v015_g3_s2_committed.pt"
        torch.manual_seed(17)
        source = _runner(layout_module, policy_cls, iteration=2)
        intent = torch.arange(2 * 3 * 29, dtype=torch.float32).reshape(2, 3, 29) / 100.0
        source_snapshot = _wire_inference_carrier(source, intent)
        raw_obs = torch.arange(2 * 870, dtype=torch.float32).reshape(2, 870) / 1000.0
        pre_update = _fresh_inference_trace(source, runtime, raw_obs)

        request = _bind_semantic_transaction(source, transaction_template)
        result = transaction_template.owners[6].run_frontres_v015_formal_transaction_update(source, request)
        committed_state = copy.deepcopy(source._frontres_v015_checkpoint_transaction_state)
        assert committed_state["state"] == "committed"
        assert committed_state["receipt"]["transaction_id"] == result.transaction_id
        assert committed_state["receipt"]["optimizer_step_delta"] == 1
        assert committed_state["receipt"]["collected_policy_attempt_count"] == 4
        assert source.alg.optimizer.frontres_v015_step_count == 1
        source.current_learning_iteration += 1
        before = _fresh_inference_trace(source, runtime, raw_obs)
        assert not torch.equal(pre_update["proposal"], before["proposal"])

        checkpointing.save_runner(source, str(path))
        payload = _saved_payload(path)
        assert payload["frontres_v015_checkpoint_identity"]["transaction"] == committed_state

        torch.manual_seed(29)
        fresh = _runner(layout_module, policy_cls, iteration=0)
        fresh._frontres_extra_mean.fill_(-7.0)
        fresh._frontres_extra_std.fill_(3.0)
        fresh_snapshot = _wire_inference_carrier(fresh, intent)
        pre_load = _fresh_inference_trace(fresh, runtime, raw_obs)
        assert not torch.equal(pre_load["actor_input"], before["actor_input"])
        assert not torch.equal(pre_load["proposal"], before["proposal"])

        checkpointing.load_runner(fresh, str(path), load_optimizer=False)
        after = _fresh_inference_trace(fresh, runtime, raw_obs)

        torch.testing.assert_close(source_snapshot["intent_q29"], fresh_snapshot["intent_q29"])
        for key in (
            "scenario_ids",
            "noisy_segment_hashes",
            "x_t_identities",
            "roles",
            "provenance",
        ):
            assert source_snapshot[key] == fresh_snapshot[key]
        assert tuple(after["combined"].shape) == (2, 928)
        assert tuple(after["actor_input"].shape) == (2, 158)
        assert tuple(after["normalized"][:, 158:].shape) == (2, 770)
        assert tuple(after["proposal"].shape) == (2, 6)
        expected_q29_tail = intent[:, (1, 2), :].reshape(2, 58)
        torch.testing.assert_close(after["combined"][:, :58], expected_q29_tail)
        for key in ("combined", "normalized", "actor_input", "proposal"):
            torch.testing.assert_close(after[key], before[key], rtol=0.0, atol=0.0)
        assert fresh._frontres_v015_checkpoint_transaction_state == {"state": "idle"}
        assert fresh._frontres_v015_last_committed_transaction_receipt == committed_state["receipt"]
        assert fresh._frontres_extra_stats_layout_version == layout_module.FRONTRES_FUTURE_INTENT_LAYOUT_VERSION
        assert "frontres_fixed_noisy_tape" not in repr(payload)
        assert "clean_continuation" not in repr(payload["frontres_v015_checkpoint_identity"])
        print(
            "[T-save-producer/T-v015-identity/T-commit-receipt/T-fresh-runner/"
            "T-prefix-normalizer/T-proposal-equality/T-legacy-reject] "
            "real committed transaction -> save_runner -> strict fresh load preserves "
            "928/158/770, deployment q29, normalized 158D input, and 6D proposal",
            flush=True,
        )


def main() -> None:
    transaction_template = _transaction_template()
    layout_module, checkpointing, policy_cls = _load_owners()
    runtime = _load_runtime()
    test_t_checkpoint_layout_and_committed_receipt(layout_module, checkpointing, policy_cls)
    test_t_v015_envelope_distinguishes_completed_hsl_history(layout_module, checkpointing, policy_cls)
    test_t_resume_rejects_layout_legacy_and_normalizer_before_mutation(layout_module, checkpointing, policy_cls)
    test_t_zero_and_full_observation_prefix_reject_before_save(layout_module, checkpointing, policy_cls)
    test_t_atomicity_rejects_partial_save_and_resume(layout_module, checkpointing, policy_cls)
    test_t_v010_transition_identity_and_schedule_reject(layout_module, checkpointing, policy_cls)
    test_t_committed_save_to_fresh_inference_equality(
        layout_module,
        checkpointing,
        policy_cls,
        runtime,
        transaction_template,
    )
    print("frontres_v015_checkpoint_resume_contract: ok", flush=True)


if __name__ == "__main__":
    main()
