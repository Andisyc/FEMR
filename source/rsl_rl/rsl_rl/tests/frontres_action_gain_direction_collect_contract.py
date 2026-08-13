"""Focused contract for the bounded checkpoint-v19 direction collector."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch

from frontres_contract_imports import install_frontres_contract_packages
from frontres_action_gain_direction_analysis import analyze_payload


install_frontres_contract_packages()

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = ROOT / "note/testing/manifests/frontres_v024_action_gain_direction_k8_v1.json"

from rsl_rl.runners.frontres_action_gain_direction_collect import (  # noqa: E402
    DIAGNOSTIC_CLASS,
    FrontRESActionGainDirectionOwners,
    FrontRESActionGainDirectionRequest,
    _parse_manifest,
    run_frontres_action_gain_direction_collect,
)
from rsl_rl.runners.frontres_rollout_step import frontres_policy_action_rng_scope  # noqa: E402
from rsl_rl.runners.frontres_checkpoint_quality import (  # noqa: E402
    FrontRESActiveQualityCheckpointIdentity,
)
from rsl_rl.runners.frontres_segment_live_sampler import close_frontres_local_scenarios  # noqa: E402


def _identity() -> FrontRESActiveQualityCheckpointIdentity:
    return FrontRESActiveQualityCheckpointIdentity(
        route="policy",
        format="frontres-v024-checkpoint-v19",
        file_sha256="c" * 64,
        method_contract_id="FRS-METHOD-v025",
        training_contract_id="FRS-TRAIN-v024",
        gain_contract_id="FRS-GAIN-v008",
        ppo_contract_id="FRS-PPO-v012",
        future_intent_layout=(),
        action_kind="delta_se3",
        action_dim=6,
        action_semantics="direct-world-full6-v1",
        normalizer_key="obs_norm_state_dict",
        actor_fingerprint="a" * 64,
        distribution_key="std",
        distribution_fingerprint="d" * 64,
        normalizer_fingerprint="n" * 64,
        critic_fingerprint="e" * 64,
        critic_input_dim=449,
        critic_value_kind="state_value",
        critic_action_conditioned=False,
        critic_target_id="scenario-current-exact-m4-mean-symlog-v1",
    )


class _Aggregate:
    execution_phase = "idle"
    persistence_phase = "idle"


class _Harness:
    def __init__(
        self,
        *,
        fail_collect_at: int | None = None,
        drift_used_input_at: int | None = None,
    ) -> None:
        self.policy_loaded = False
        self.readonly_depth = 0
        self.collect_count = 0
        self.fail_collect_at = fail_collect_at
        self.drift_used_input_at = drift_used_input_at
        self.written = None
        self.prepared_close_count = 0
        self.runtime_rng_fingerprints = []
        self.action_seeds = []

    @contextmanager
    def policy_route(self, runner, checkpoint_path: str, checkpoint_sha: str):
        assert checkpoint_path == "/tmp/checkpoint-v19.pt"
        assert checkpoint_sha == "c" * 64
        assert not self.policy_loaded
        self.policy_loaded = True
        try:
            yield _identity()
        finally:
            self.policy_loaded = False

    @contextmanager
    def readonly_scope(self, runner):
        assert self.policy_loaded and self.readonly_depth == 0
        self.readonly_depth = 1
        try:
            yield
        finally:
            self.readonly_depth = 0

    def prepare(self, runner, items, attempts):
        assert self.readonly_depth == 1 and attempts == 4 and len(items) == 2
        return SimpleNamespace(items=items, closed=False)

    def close_prepared(self, prepared):
        if not prepared.closed:
            prepared.closed = True
            self.prepared_close_count += 1

    def collect(self, runner, prepared, frozen, beta, action_seed):
        call_index = self.collect_count
        self.collect_count += 1
        self.runtime_rng_fingerprints.append(
            (random.random(), float(np.random.random()), float(torch.rand(())))
        )
        self.action_seeds.append(action_seed)
        if self.fail_collect_at == call_index:
            raise RuntimeError("injected collection failure")
        repeat = call_index // 2
        attempts = []
        totals = []
        intents = []
        physics = []
        penalties = []
        physics_noisy = []
        physics_repaired = []
        physics_gain = []
        recovery_pressure = []
        physics_channel_noisy = []
        physics_channel_repaired = []
        for source, item in enumerate(prepared.items):
            mean = torch.full((6,), 0.01 * (source + call_index % 2))
            sigma = torch.full((6,), 0.1)
            for trial in range(4):
                action = mean + 0.001 * (repeat + 1) * torch.tensor(
                    [trial + 1, trial + 2, trial + 3, trial + 4, trial + 5, trial + 6],
                    dtype=torch.float32,
                )
                intent = float(action[0] - action[1])
                weighted_physics = float(0.5 * action[2])
                p_noisy = 1.0 + 0.01 * source
                p_repaired = p_noisy - weighted_physics
                p_gain = p_noisy - p_repaired
                pressure = 0.5 * (p_noisy + p_repaired)
                weighted_physics = pressure * p_gain
                penalty = float(0.02 * torch.linalg.vector_norm(action))
                total = intent + weighted_physics - penalty
                attempts.append(
                    SimpleNamespace(
                        source_index=source,
                        trial_index=trial,
                        scenario_id=f"scenario:{item.item_id}",
                        noisy_segment_hash=f"noisy:{item.item_id}",
                        x_t_identity=f"x-t:{item.item_id}",
                        segment_id=100 + source + 2 * (call_index % 2),
                        policy_mean=mean,
                        policy_sigma=sigma,
                        policy_action=action,
                    )
                )
                totals.append(total)
                intents.append(intent)
                physics.append(weighted_physics)
                physics_noisy.append(p_noisy)
                physics_repaired.append(p_repaired)
                physics_gain.append(p_gain)
                recovery_pressure.append(pressure)
                physics_channel_noisy.append([p_noisy] * 4)
                physics_channel_repaired.append([p_repaired] * 4)
                penalties.append(penalty)
        evidence = SimpleNamespace(ordered_attempts=tuple(attempts))
        gain = SimpleNamespace(
            gain_total=torch.tensor(totals),
            intent_gain=torch.tensor(intents),
            weighted_physics_gain=torch.tensor(physics),
            physics_remaining_noisy=torch.tensor(physics_noisy),
            physics_remaining_repaired=torch.tensor(physics_repaired),
            physics_gain=torch.tensor(physics_gain),
            recovery_pressure=torch.tensor(recovery_pressure),
            physics_channel_noisy=torch.tensor(physics_channel_noisy),
            physics_channel_repaired=torch.tensor(physics_channel_repaired),
            repair_penalty=torch.tensor(penalties),
        )
        used_policy_observations = frozen
        if used_policy_observations is None:
            used_policy_observations = SimpleNamespace(
                obs=torch.arange(16 * 928, dtype=torch.float32).reshape(16, 928),
                privileged_obs=torch.arange(16 * 449, dtype=torch.float32).reshape(16, 449),
            )
        elif self.drift_used_input_at == call_index:
            used_policy_observations = SimpleNamespace(
                obs=used_policy_observations.obs.clone(),
                privileged_obs=used_policy_observations.privileged_obs.clone(),
            )
            used_policy_observations.obs[0, 0] += 1.0e-3
        return SimpleNamespace(
            evidence=evidence,
            gain=gain,
            policy_observations=used_policy_observations,
            observation_trace={
                "repeat_policy_input_source": (
                    "live-first-repeat" if frozen is None else "first-repeat-frozen"
                ),
                "repeat_live_actor_input_max_abs_diff": 0.0 if frozen is None else 3.8110761642456055,
                "repeat_live_critic_input_max_abs_diff": 0.0 if frozen is None else 4.439251899719238,
                "repeat_used_actor_input_max_abs_diff": 0.0,
                "repeat_used_critic_input_max_abs_diff": 0.0,
            },
        )

    def state_hashes(self, runner):
        return {
            "actor": "policy" if self.policy_loaded else "hsl",
            "optimizer": "unchanged",
            "normalizers": "unchanged",
            "transaction": "idle" if self.readonly_depth == 0 else "evaluating",
            "outer_replay": "absent",
        }

    def owners(self) -> FrontRESActionGainDirectionOwners:
        return FrontRESActionGainDirectionOwners(
            ensure_reset_support=lambda runner: None,
            transaction_aggregate=lambda runner: _Aggregate(),
            policy_route=self.policy_route,
            readonly_scope=self.readonly_scope,
            prepare_batch=self.prepare,
            close_prepared=self.close_prepared,
            collect=self.collect,
            training_state_hashes=self.state_hashes,
            replay_owner_present=lambda runner: False,
            write_json=lambda path, payload: setattr(self, "written", (path, payload)),
        )


def _request() -> FrontRESActionGainDirectionRequest:
    return FrontRESActionGainDirectionRequest(
        manifest_path=str(MANIFEST),
        policy_checkpoint_path="/tmp/checkpoint-v19.pt",
        result_path="/tmp/action-gain-direction.json",
        manifest=_parse_manifest(MANIFEST),
        checkpoint=_identity(),
    )


def _runner():
    return SimpleNamespace(
        alg=SimpleNamespace(
            frontres_formal_transaction_enabled=True,
            frontres_policy_quality_eval_only=True,
            frontres_segment_replay_enabled=False,
            frontres_segment_live_runner_enabled=False,
            frontres_gain_beta=0.02,
        )
    )


def _rng_probe():
    return (
        random.getstate(),
        np.random.get_state()[1].copy(),
        torch.random.get_rng_state().clone(),
    )


def test_bounded_collection_emits_analyzer_schema_without_state_writes() -> None:
    harness = _Harness()
    payload = run_frontres_action_gain_direction_collect(
        _runner(),
        request=_request(),
        owners=harness.owners(),
    )
    assert payload["diagnostic_class"] == DIAGNOSTIC_CLASS
    assert payload["formal_pass"] is False
    assert payload["collection_identity"]["replay_constructed"] is False
    assert len(payload["scenarios"]) == 4
    assert all(len(scenario["rows"]) == 32 for scenario in payload["scenarios"])
    assert all(len(scenario["visits"]) == 8 for scenario in payload["scenarios"])
    for scenario in payload["scenarios"]:
        assert [row["repair_index"] for row in scenario["rows"]] == list(range(32))
        assert [row["visit_index"] for row in scenario["rows"]] == [index // 4 for index in range(32)]
        assert [row["attempt_index"] for row in scenario["rows"]] == [index % 4 for index in range(32)]
        assert scenario["visits"][0]["live_actor_input_max_abs_diff"] == 0.0
        assert scenario["visits"][1]["live_actor_input_max_abs_diff"] == 3.8110761642456055
        assert all(visit["actor_input_max_abs_diff"] == 0.0 for visit in scenario["visits"])
        assert all(visit["critic_input_max_abs_diff"] == 0.0 for visit in scenario["visits"])
    assert harness.collect_count == 16
    assert len(set(harness.action_seeds)) == 16
    assert harness.runtime_rng_fingerprints[0] == harness.runtime_rng_fingerprints[2]
    assert harness.runtime_rng_fingerprints[1] == harness.runtime_rng_fingerprints[3]
    assert harness.prepared_close_count == 16
    assert harness.policy_loaded is False and harness.readonly_depth == 0
    assert harness.written is not None and harness.written[1] == payload
    analyzed = analyze_payload(payload, partition_count=4, permutation_count=8, seed=17)
    assert len(analyzed["scenarios"]) == 4


def test_action_rng_is_nested_and_restores_runtime_rng() -> None:
    torch.manual_seed(911)
    before = torch.random.get_rng_state().clone()
    with frontres_policy_action_rng_scope(101):
        action_a = torch.rand(6)
    after_a = torch.random.get_rng_state().clone()
    assert torch.equal(before, after_a)
    with frontres_policy_action_rng_scope(202):
        action_b = torch.rand(6)
    assert not torch.equal(action_a, action_b)
    assert torch.equal(before, torch.random.get_rng_state())


def test_physics_decomposition_is_exported() -> None:
    payload = run_frontres_action_gain_direction_collect(
        _runner(), request=_request(), owners=_Harness().owners()
    )
    components = payload["scenarios"][0]["rows"][0]["components"]
    for field in (
        "physics_remaining_noisy",
        "physics_remaining_repaired",
        "physics_gain",
        "recovery_pressure",
        "physics_channel_noisy",
        "physics_channel_repaired",
    ):
        assert field in components


def test_exception_restores_route_rng_and_readonly_lifecycle() -> None:
    harness = _Harness(fail_collect_at=2)
    before = _rng_probe()
    try:
        run_frontres_action_gain_direction_collect(
            _runner(),
            request=_request(),
            owners=harness.owners(),
        )
    except RuntimeError as exc:
        assert "injected collection failure" in str(exc)
    else:
        raise AssertionError("injected collection failure did not propagate")
    after = _rng_probe()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert torch.equal(before[2], after[2])
    assert harness.policy_loaded is False and harness.readonly_depth == 0
    assert harness.prepared_close_count == 3
    assert harness.written is None


def test_used_policy_input_drift_still_fails_closed() -> None:
    harness = _Harness(drift_used_input_at=2)
    try:
        run_frontres_action_gain_direction_collect(
            _runner(),
            request=_request(),
            owners=harness.owners(),
        )
    except RuntimeError as exc:
        assert "changed used Actor/Critic inputs" in str(exc)
    else:
        raise AssertionError("collector must reject drift in the policy inputs actually consumed")
    assert harness.policy_loaded is False and harness.readonly_depth == 0
    assert harness.prepared_close_count == 3
    assert harness.written is None


def test_real_prepared_cleanup_is_idempotent() -> None:
    closed = []
    lifecycle = SimpleNamespace(close_scenario=lambda scenario_id: closed.append(scenario_id))
    batch = SimpleNamespace(
        frontres_local_scenario_lifecycle=lifecycle,
        frontres_local_scenario_rows=SimpleNamespace(scenario_ids=("a", "a", "b", "b")),
    )
    close_frontres_local_scenarios(batch)
    close_frontres_local_scenarios(batch)
    assert closed == ["a", "b"]
    assert batch.frontres_local_scenario_closed_ids == ("a", "b")


def test_entrypoint_preserves_formal_training_and_historical_eval_boundaries() -> None:
    train_source = (ROOT / "scripts/rsl_rl/train.py").read_text(encoding="utf-8")
    launcher = (ROOT / "run/run_frontres_stage3_segment_hrl.sh").read_text(encoding="utf-8")
    sampler = (
        ROOT / "source/rsl_rl/rsl_rl/runners/frontres_segment_live_sampler.py"
    ).read_text(encoding="utf-8")
    formal = (
        ROOT / "source/rsl_rl/rsl_rl/runners/frontres_segment_formal_transaction.py"
    ).read_text(encoding="utf-8")
    assert "--frontres_action_gain_direction_collect_only" in train_source
    assert "action_gain_direction_collect" in launcher
    assert "prepare_frontres_fixed_k_m4_evaluation_batch" in sampler
    assert "Historical EVAL-v004 wrapper" in sampler
    assert 'request.plan.selected_segment_count != 8' in formal
    assert "formal update rejects mixed-M or non-B8 transactions" in formal
    collector = (
        ROOT / "source/rsl_rl/rsl_rl/runners/frontres_action_gain_direction_collect.py"
    ).read_text(encoding="utf-8")
    for forbidden in ("optimizer.step(", ".backward(", "outer_replay.stage(", "runner.load("):
        assert forbidden not in collector


def main() -> None:
    test_bounded_collection_emits_analyzer_schema_without_state_writes()
    test_action_rng_is_nested_and_restores_runtime_rng()
    test_physics_decomposition_is_exported()
    test_exception_restores_route_rng_and_readonly_lifecycle()
    test_used_policy_input_drift_still_fails_closed()
    test_real_prepared_cleanup_is_idempotent()
    test_entrypoint_preserves_formal_training_and_historical_eval_boundaries()
    print("frontres_action_gain_direction_collect_contract: ok")


if __name__ == "__main__":
    main()
