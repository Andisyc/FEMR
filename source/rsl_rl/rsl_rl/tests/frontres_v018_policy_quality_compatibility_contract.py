from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import torch

from frontres_contract_imports import install_frontres_contract_packages

install_frontres_contract_packages()

from rsl_rl.frontres.frontres_policy_quality_manifest import (  # noqa: E402
    FrontRESV018PolicyQualityManifest,
)
from rsl_rl.modules.normalizer import EmpiricalNormalization  # noqa: E402
import rsl_rl.runners.frontres_checkpointing as checkpointing  # noqa: E402
import rsl_rl.runners.frontres_policy_quality_eval as quality  # noqa: E402


ROOT = Path(__file__).resolve().parents[4]
MANIFEST = ROOT / "note" / "testing" / "manifests" / "frontres_v018_policy_quality_k16_m4_v1.json"


def _assert_tensor_state_equal(first: dict[str, object], second: dict[str, object]) -> None:
    assert set(first) == set(second)
    for name in first:
        left = first[name]
        right = second[name]
        if isinstance(left, torch.Tensor):
            assert isinstance(right, torch.Tensor)
            torch.testing.assert_close(left, right)
        else:
            assert left == right


def test_v018_manifest_and_request_seal_checkpoint_v13_critic_identity(tmp_path: Path) -> None:
    manifest = FrontRESV018PolicyQualityManifest.from_json(MANIFEST.read_text(encoding="utf-8"))
    assert manifest.method_contract_id == "FRS-METHOD-v020"
    assert manifest.training_contract_id == "FRS-TRAIN-v019"
    assert manifest.ppo_contract_id == "FRS-PPO-v008"
    assert manifest.checkpoint_format == "frontres-v019-checkpoint-v14"
    assert manifest.critic_input_dim == 449
    assert manifest.critic_value_kind == "state_value"
    assert manifest.critic_action_conditioned is False
    assert manifest.critic_target_id == "segment-exact-m-mean-symlog-v1"
    assert manifest.critic_support_context_id == "action-pre-support-plan-kmax32-v1"
    assert manifest.critic_value_normalization_id == "ema-target-std-nonamplifying-v1"
    assert (manifest.horizon_k, manifest.attempts_per_segment) == (16, 4)

    hsl_path = tmp_path / "hsl.pt"
    policy_path = tmp_path / "policy.pt"
    result_path = tmp_path / "quality.json"
    hsl_path.write_bytes(b"hsl")
    policy_path.write_bytes(b"policy")
    layout = (
        ("layout_version", manifest.future_intent_layout_version),
        ("future_offsets", manifest.future_offsets),
        ("intent_dim", 29),
        ("actor_tail_dim", 58),
        ("environment_obs_dim", manifest.raw_observation_dim),
        ("current_frontres_prefix_dim", manifest.actor_input_dim - 58),
        ("actor_dim", manifest.combined_observation_dim),
        ("prefix_dim", manifest.actor_input_dim),
        ("gmt_dim", manifest.gmt_suffix_dim),
    )
    policy_critic_dim = 449

    def inspect(path: object, *, route: str) -> SimpleNamespace:
        if route == "hsl":
            return SimpleNamespace(
                format="frontres-v017-hsl-proposal-v2",
                file_sha256="h" * 64,
                method_contract_id="FRS-METHOD-v017",
                training_contract_id="FRS-TRAIN-v014",
                gain_contract_id=None,
                ppo_contract_id=None,
                future_intent_layout=layout,
                action_kind="delta_se3",
                action_dim=6,
                action_semantics="direct-world-full6-v1",
            )
        assert Path(path).resolve() == policy_path.resolve()
        return SimpleNamespace(
            format="frontres-v019-checkpoint-v14",
            file_sha256="p" * 64,
            method_contract_id="FRS-METHOD-v020",
            training_contract_id="FRS-TRAIN-v019",
            gain_contract_id="FRS-GAIN-v008",
            ppo_contract_id="FRS-PPO-v008",
            future_intent_layout=layout,
            action_kind="delta_se3",
            action_dim=6,
            action_semantics="direct-world-full6-v1",
            critic_input_dim=policy_critic_dim,
            critic_value_kind="state_value",
            critic_action_conditioned=False,
            critic_target_id="segment-exact-m-mean-symlog-v1",
            critic_support_context_id="action-pre-support-plan-kmax32-v1",
            critic_value_normalization_id="ema-target-std-nonamplifying-v1",
        )

    original = checkpointing.inspect_frontres_quality_checkpoint
    checkpointing.inspect_frontres_quality_checkpoint = inspect
    try:
        request = quality.build_frontres_v018_policy_quality_eval_request(
            manifest_path=str(MANIFEST),
            hsl_checkpoint_path=str(hsl_path),
            policy_checkpoint_path=str(policy_path),
            result_path=str(result_path),
        )
        assert request.manifest == manifest
        policy_critic_dim = 347
        try:
            quality.build_frontres_v018_policy_quality_eval_request(
                manifest_path=str(MANIFEST),
                hsl_checkpoint_path=str(hsl_path),
                policy_checkpoint_path=str(policy_path),
                result_path=str(result_path),
            )
        except ValueError as exc:
            assert "Critic" in str(exc)
        else:
            raise AssertionError("active Evaluation must reject a non-449D checkpoint before mutation")
    finally:
        checkpointing.inspect_frontres_quality_checkpoint = original


def test_policy_checkpoint_context_installs_and_restores_critic_observation_normalizer() -> None:
    class Policy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.residual_actor = torch.nn.Linear(158, 6)
            self.critic = torch.nn.Linear(449, 1)
            self.register_buffer("std", torch.full((6,), 0.7))
            self.num_actor_obs = 928

    policy = Policy()
    optimizer = torch.optim.Adam(tuple(policy.parameters()))
    privileged_normalizer = EmpiricalNormalization(shape=[449], until=1.0e8)
    runner = SimpleNamespace(
        alg=SimpleNamespace(policy=policy, optimizer=optimizer),
        device=torch.device("cpu"),
        empirical_normalization=True,
        privileged_obs_normalizer=privileged_normalizer,
        _frontres_gmt_obs_dim=770,
        _frontres_extra_normalizer=None,
        _frontres_extra_mean=torch.zeros(158),
        _frontres_extra_std=torch.ones(158),
        _frontres_extra_stats_layout_version="source",
    )
    actor_before = copy.deepcopy(policy.residual_actor.state_dict())
    critic_before = copy.deepcopy(policy.critic.state_dict())
    privileged_before = copy.deepcopy(privileged_normalizer.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())

    checkpoint_actor = {name: value.detach().clone() + 1.0 for name, value in actor_before.items()}
    checkpoint_critic = {name: value.detach().clone() + 2.0 for name, value in critic_before.items()}
    checkpoint_privileged = copy.deepcopy(privileged_before)
    checkpoint_privileged["_mean"] = torch.full_like(checkpoint_privileged["_mean"], 3.0)
    checkpoint_privileged["_var"] = torch.full_like(checkpoint_privileged["_var"], 4.0)
    checkpoint_privileged["_std"] = torch.full_like(checkpoint_privileged["_std"], 2.0)
    checkpoint_privileged["count"] = torch.full_like(checkpoint_privileged["count"], 17)
    checkpoint = {
        "model_state_dict": {
            "residual_actor": checkpoint_actor,
            "critic": checkpoint_critic,
            "std": torch.full((6,), 0.5),
        },
        "obs_norm_state_dict": {
            "_mean": torch.zeros(1, 928),
            "_var": torch.ones(1, 928),
            "_std": torch.ones(1, 928),
            "count": torch.tensor(11),
        },
        "privileged_obs_norm_state_dict": checkpoint_privileged,
    }

    original_inspect = checkpointing.inspect_frontres_quality_checkpoint
    original_load = checkpointing.load_frontres_checkpoint_mapping
    original_validate = checkpointing._validate_v015_checkpoint_resume
    checkpointing.inspect_frontres_quality_checkpoint = lambda *_args, **_kwargs: SimpleNamespace(
        file_sha256="c" * 64,
        format="frontres-v019-checkpoint-v14",
    )
    checkpointing.load_frontres_checkpoint_mapping = lambda *_args, **_kwargs: checkpoint
    checkpointing._validate_v015_checkpoint_resume = lambda *_args, **_kwargs: {}
    try:
        try:
            with checkpointing.frontres_quality_route_actor(
                runner,
                "policy.pt",
                route="policy",
                expected_file_sha256="c" * 64,
            ):
                _assert_tensor_state_equal(privileged_normalizer.state_dict(), checkpoint_privileged)
                for name, value in checkpoint_critic.items():
                    torch.testing.assert_close(policy.critic.state_dict()[name], value)
                raise RuntimeError("deliberate evaluator failure")
        except RuntimeError as exc:
            assert "deliberate evaluator failure" in str(exc)
        else:
            raise AssertionError("the deliberate evaluator failure must escape")
    finally:
        checkpointing.inspect_frontres_quality_checkpoint = original_inspect
        checkpointing.load_frontres_checkpoint_mapping = original_load
        checkpointing._validate_v015_checkpoint_resume = original_validate

    _assert_tensor_state_equal(policy.residual_actor.state_dict(), actor_before)
    _assert_tensor_state_equal(policy.critic.state_dict(), critic_before)
    _assert_tensor_state_equal(privileged_normalizer.state_dict(), privileged_before)
    assert optimizer.state_dict() == optimizer_before


def test_segment_calibration_uses_shared_value_and_exact_m4_mean() -> None:
    report = SimpleNamespace(
        scenario_ids=("s0",) * 4 + ("s1",) * 4,
        noisy_segment_hashes=("h0",) * 4 + ("h1",) * 4,
        valid_policy_row_mask=(True,) * 8,
        policy_values=(2.0,) * 4 + (-1.0,) * 4,
        gain_total=(1.0, 3.0, 5.0, 7.0, -4.0, -2.0, 0.0, 2.0),
    )
    plan = SimpleNamespace(
        active_m=4,
        source_index=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        segment_ids=torch.tensor([10, 10, 10, 10, 20, 20, 20, 20]),
    )
    rows = quality.build_frontres_v018_critic_calibration_rows(report, plan)
    assert len(rows) == 2
    expected_raw = (4.0, -1.0)
    raw_groups = (report.gain_total[:4], report.gain_total[4:])
    for index, row in enumerate(rows):
        utility = torch.sign(torch.tensor(raw_groups[index])) * torch.log1p(torch.abs(torch.tensor(raw_groups[index])))
        expected_target = float(utility.mean())
        assert row["raw_target_mean"] == expected_raw[index]
        assert row["return_utility_id"] == "symmetric-log-gain-g0-1-v1"
        assert row["return_utility_scale"] == 1.0
        torch.testing.assert_close(torch.tensor(row["utility_attempts"]), utility)
        assert abs(row["target_mean"] - expected_target) < 1.0e-6
        assert abs(row["value_error"] - (row["policy_value"] - expected_target)) < 1.0e-6

    permutation = torch.tensor((4, 7, 5, 6, 2, 0, 3, 1), dtype=torch.long)
    permuted_report = SimpleNamespace(
        scenario_ids=tuple(report.scenario_ids[index] for index in permutation.tolist()),
        noisy_segment_hashes=tuple(report.noisy_segment_hashes[index] for index in permutation.tolist()),
        valid_policy_row_mask=tuple(report.valid_policy_row_mask[index] for index in permutation.tolist()),
        policy_values=tuple(report.policy_values[index] for index in permutation.tolist()),
        gain_total=tuple(report.gain_total[index] for index in permutation.tolist()),
    )
    permuted_plan = SimpleNamespace(
        active_m=4,
        source_index=plan.source_index.index_select(0, permutation),
        segment_ids=plan.segment_ids.index_select(0, permutation),
    )
    assert quality.build_frontres_v018_critic_calibration_rows(permuted_report, permuted_plan) == rows

    invalid_m_plan = SimpleNamespace(
        active_m=3,
        source_index=plan.source_index,
        segment_ids=plan.segment_ids,
    )
    try:
        quality.build_frontres_v018_critic_calibration_rows(report, invalid_m_plan)
    except RuntimeError as exc:
        assert "exact M4" in str(exc)
    else:
        raise AssertionError("non-M4 Evaluation must fail closed")

    report.policy_values = (2.0, 2.0, 3.0, 2.0) + (-1.0,) * 4
    try:
        quality.build_frontres_v018_critic_calibration_rows(report, plan)
    except RuntimeError as exc:
        assert "shared Critic value" in str(exc)
    else:
        raise AssertionError("same-Segment value drift must fail closed")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_v018_manifest_and_request_seal_checkpoint_v13_critic_identity(Path(tmp))
    test_policy_checkpoint_context_installs_and_restores_critic_observation_normalizer()
    test_segment_calibration_uses_shared_value_and_exact_m4_mean()
    print("frontres_v018_policy_quality_compatibility_contract: ok")
