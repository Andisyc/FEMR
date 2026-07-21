#!/usr/bin/env python3
"""Deterministic S2 contract for the dedicated v015 deployment live CLI."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[4]
CLI_PATH = ROOT / "scripts" / "rsl_rl" / "frontres_v015_deployment_composition.py"
RUNNER_PATH = ROOT / "source" / "rsl_rl" / "rsl_rl" / "runners" / "on_policy_runner.py"
EVAL_PATH = ROOT / "source" / "rsl_rl" / "rsl_rl" / "runners" / "frontres_segment_sequence_eval.py"
BOUNDARY_PATH = ROOT / "source" / "rsl_rl" / "rsl_rl" / "runners" / "frontres_segment_runner_boundary.py"
TASK_REGISTRY_PATH = ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "tasks" / "tracking" / "config" / "g1" / "__init__.py"
RSL_SOURCE_ROOT = ROOT / "source" / "rsl_rl"
if str(RSL_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(RSL_SOURCE_ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _touch(path: Path) -> str:
    path.write_bytes(b"checkpoint")
    return str(path.resolve())


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        task="FrontRES-Unified-Tracking-Flat-G1-v0",
        frontres_checkpoint=_touch(tmp_path / "frontres.pt"),
        gmt_checkpoint=_touch(tmp_path / "gmt.pt"),
        reference_npz=_touch(tmp_path / "noisy.npz"),
        report_path=str((tmp_path / "report.json").resolve()),
        future_offsets="1,2",
        corruption_id="persistent-rp-test",
        corruption_family="local_rp",
        corruption_seed=17,
        corruption_parameters_json=json.dumps({"std_rp": 0.08}),
        num_envs=2,
        device="cuda:0",
    )


def _agent_cfg() -> SimpleNamespace:
    algorithm_fields = {
        "frontres_training_objective": "legacy",
        "frontres_segment_replay_enabled": True,
        "frontres_segment_live_runner_enabled": True,
        "frontres_segment_live_train_enabled": True,
        "frontres_segment_live_sentinel_only": True,
        "frontres_v015_local_sentinel_only": True,
        "frontres_segment_live_probe_only": True,
        "frontres_segment_live_storage_write_only": True,
        "frontres_segment_live_single_update_only": True,
        "frontres_segment_live_update_loop_only": True,
        "frontres_segment_offline_eval_only": True,
        "frontres_segment_sequence_offline_eval_only": True,
        "frontres_v015_formal_transaction_enabled": False,
        "frontres_segment_critic_warmup_iterations": 200,
        "frontres_segment_actor_warmup_iterations": 500,
        "frontres_future_offsets": (),
        "frontres_future_intent_layout_version": "",
        "frontres_segment_advantage_normalization": "scale_only",
        "frontres_hsl_init_enabled": True,
        "frontres_hsl_rollout_label_enabled": True,
        "lambda_supervised": 1.0,
        "lambda_supervised_min": 1.0,
    }
    return SimpleNamespace(
        policy=SimpleNamespace(gmt_checkpoint_path=None),
        algorithm=SimpleNamespace(**algorithm_fields),
        device="cpu",
        max_iterations=100,
        resume=True,
        is_full_resume=False,
        supervised_warmup_iterations=10,
        critic_warmup_iterations=10,
    )


def test_t_cli_contract_paths_gpu_and_protocol(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    cli = _load("frontres_v015_deployment_live_cli_contract", CLI_PATH)
    args = _args(tmp_path)
    contract = cli.validate_frontres_v015_deployment_cli_args(
        args,
        environ={"CUDA_VISIBLE_DEVICES": "3"},
    )

    assert contract.frontres_checkpoint == str(Path(args.frontres_checkpoint).resolve())
    assert contract.gmt_checkpoint == str(Path(args.gmt_checkpoint).resolve())
    assert contract.reference_npz == str(Path(args.reference_npz).resolve())
    assert contract.report_path == str(Path(args.report_path).resolve())
    assert contract.future_offsets == (1, 2)
    assert contract.cuda_visible_devices == "3"
    assert dict(contract.corruption_parameters) == {
        "source": "pre_materialized_deployment_npz",
        "std_rp": 0.08,
    }

    sequence_module = _load("frontres_v015_deployment_live_cli_sequence", EVAL_PATH)
    run_config = cli.build_frontres_v015_deployment_run_config(
        contract,
        sequence_module=sequence_module,
    )
    run_config.validate()
    assert run_config.request_config.reference_path == contract.reference_npz
    assert run_config.request_config.corruption_protocol.protocol_hash

    bad = SimpleNamespace(**vars(args))
    bad.report_path = "relative-report.json"
    try:
        cli.validate_frontres_v015_deployment_cli_args(bad, environ={"CUDA_VISIBLE_DEVICES": "3"})
    except ValueError as exc:
        assert "absolute" in str(exc).lower()
    else:
        raise AssertionError("relative report path unexpectedly accepted")

    try:
        cli.validate_frontres_v015_deployment_cli_args(args, environ={})
    except ValueError as exc:
        assert "cuda_visible_devices" in str(exc).lower()
    else:
        raise AssertionError("CUDA dispatch unexpectedly accepted without CUDA_VISIBLE_DEVICES")
    print("[T-path/T-gpu/T-protocol] absolute identities and CUDA_VISIBLE_DEVICES are fail-closed", flush=True)


def test_t_agent_config_and_dispatch_are_eval_only(tmp_path: Path) -> None:
    cli = _load("frontres_v015_deployment_live_cli_dispatch", CLI_PATH)
    contract = cli.validate_frontres_v015_deployment_cli_args(
        _args(tmp_path),
        environ={"CUDA_VISIBLE_DEVICES": "3"},
    )
    agent_cfg = _agent_cfg()
    cli.configure_frontres_v015_deployment_agent_cfg(agent_cfg, contract)
    alg = agent_cfg.algorithm

    assert agent_cfg.policy.gmt_checkpoint_path == contract.gmt_checkpoint
    assert agent_cfg.max_iterations == 0
    assert agent_cfg.resume is False
    assert agent_cfg.is_full_resume is True
    assert agent_cfg.supervised_warmup_iterations == 0
    assert agent_cfg.critic_warmup_iterations == 0
    assert alg.frontres_training_objective == "deployment_composition_eval"
    assert alg.frontres_v015_formal_transaction_enabled is True
    assert alg.frontres_segment_critic_warmup_iterations == 0
    assert alg.frontres_segment_actor_warmup_iterations == 0
    assert alg.frontres_future_offsets == (1, 2)
    assert alg.frontres_future_intent_layout_version == "frontres-v015-future-intent-q29-v1"
    assert alg.frontres_segment_replay_enabled is False
    assert alg.frontres_segment_live_runner_enabled is False
    assert alg.frontres_segment_live_train_enabled is False
    assert alg.frontres_hsl_init_enabled is False
    assert alg.frontres_hsl_rollout_label_enabled is False
    assert alg.lambda_supervised == 0.0
    assert alg.lambda_supervised_min == 0.0
    boundary_module = _load("frontres_v015_deployment_live_cli_boundary", BOUNDARY_PATH)
    boundary = boundary_module.FrontRESSegmentRunnerBoundary.from_train_cfg(
        {"algorithm": vars(alg)}
    )
    assert boundary.requested is False
    assert boundary.live_runner_enabled is False

    class FakeRunner:
        def __init__(self):
            self.alg = SimpleNamespace(optimizer=SimpleNamespace(frontres_v015_step_count=11))
            self.calls = 0

        def run_frontres_v015_deployment_composition_eval(self, *, config):
            self.calls += 1
            assert config.report_path == contract.report_path
            return SimpleNamespace(
                request=SimpleNamespace(
                    reference_file_hash="a" * 64,
                    corruption_protocol=SimpleNamespace(protocol_hash="b" * 64),
                ),
                reference_frame_count=9,
                frame_count=7,
                femr_action_count=7,
                accumulated_failure_count=1,
                return_feedback=False,
                priority_feedback=False,
                ppo_feedback=False,
                sampler_feedback=False,
                optimizer_feedback=False,
            )

        def learn(self, *args, **kwargs):
            raise AssertionError("deployment CLI must not enter runner.learn")

    runner = FakeRunner()
    sequence_module = _load("frontres_v015_deployment_live_cli_dispatch_sequence", EVAL_PATH)
    report = cli.dispatch_frontres_v015_deployment_composition(
        runner,
        cli.build_frontres_v015_deployment_run_config(contract, sequence_module=sequence_module),
    )
    assert runner.calls == 1
    assert runner.alg.optimizer.frontres_v015_step_count == 11
    sentinel = cli.format_frontres_v015_deployment_sentinel(report, contract, optimizer_step_delta=0)
    assert "evaluated_frames=7" in sentinel
    assert "femr_actions=7" in sentinel
    assert "optimizer_step_delta=0" in sentinel
    assert "no_feedback=True" in sentinel
    print("[T-config/T-dispatch/T-zero-update] dedicated config calls only S2B and preserves optimizer count", flush=True)


def test_t_source_uses_formal_owners_without_training_route() -> None:
    cli_source = CLI_PATH.read_text(encoding="utf-8")
    runner_source = RUNNER_PATH.read_text(encoding="utf-8")
    task_registry = TASK_REGISTRY_PATH.read_text(encoding="utf-8")

    for required in (
        "AppLauncher.add_app_launcher_args",
        "gym.make",
        "RslRlVecEnvWrapper",
        "OnPolicyRunner(",
        "gmt_checkpoint_path",
        "load_optimizer=False",
        "load_critic=False",
        "run_frontres_v015_deployment_composition_eval",
        "CUDA_VISIBLE_DEVICES",
    ):
        assert required in cli_source, required
    for forbidden in ("runner.learn(", "optimizer.step(", "sampler.update", "run_frontres_segment_sequence_offline_eval"):
        assert forbidden not in cli_source, forbidden
    assert 'id="FrontRES-Unified-Tracking-Flat-G1-v0"' in task_registry
    configure_at = cli_source.index("configure_frontres_v015_deployment_agent_cfg(agent_cfg, contract)")
    runner_at = cli_source.index("runner = OnPolicyRunner(")
    load_at = cli_source.index("runner.load(contract.frontres_checkpoint")
    dispatch_at = cli_source.index("report = dispatch_frontres_v015_deployment_composition(")
    assert configure_at < runner_at < load_at < dispatch_at
    assert 'self.alg_cfg.get("frontres_v015_formal_transaction_enabled", False)' in runner_source
    print("[T-owner/T-formal-entry/T-no-training] IsaacLab runner and checkpoint owners precede the sole S2B dispatch", flush=True)


def main() -> None:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(prefix="frontres-v015-cli-") as temp_dir:
        root = Path(temp_dir)
        test_t_cli_contract_paths_gpu_and_protocol(root / "paths")
    with TemporaryDirectory(prefix="frontres-v015-cli-") as temp_dir:
        root = Path(temp_dir)
        root.mkdir(exist_ok=True)
        test_t_agent_config_and_dispatch_are_eval_only(root)
    test_t_source_uses_formal_owners_without_training_route()
    print("frontres_v015_deployment_live_cli_s4s0_contract: ok", flush=True)


if __name__ == "__main__":
    main()
