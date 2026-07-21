#!/usr/bin/env python3
"""Semantic CPU contract for the v015 deployment sequence command carrier."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]
TEST_ROOT = ROOT / "source" / "rsl_rl" / "rsl_rl" / "tests"
RESET_HELPER_PATH = TEST_ROOT / "frontres_v015_two_role_reset_contract.py"
ACTOR_HELPER_PATH = TEST_ROOT / "frontres_future_intent_actor_context_contract.py"
S1_HELPER_PATH = TEST_ROOT / "frontres_v015_deployment_composition_s1_contract.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _expect_error(error_type, callback, contains: str) -> None:
    try:
        callback()
    except error_type as exc:
        assert contains in str(exc), str(exc)
    else:
        raise AssertionError(f"expected {error_type.__name__} containing {contains!r}")


def _owners():
    reset_helper = _load("frontres_v015_deployment_s2a_reset_helper", RESET_HELPER_PATH)
    commands, _hooks, _setup = reset_helper._load_owners()
    actor_helper = _load("frontres_v015_deployment_s2a_actor_helper", ACTOR_HELPER_PATH)
    _layout, runtime = actor_helper._load_modules()
    s1_helper = _load("frontres_v015_deployment_s2a_s1_helper", S1_HELPER_PATH)
    s1_owner = s1_helper._load_owner()
    return reset_helper, commands, runtime, s1_helper, s1_owner


def _request(s1_helper, s1_owner, root: Path, *, frame_count: int = 6):
    reference = root / "deployment_motion.npz"
    s1_helper._write_npz(reference, frame_count=frame_count)
    protocol = s1_owner.build_frontres_v015_persistent_corruption_protocol(
        corruption_id="persistent-rp-0001",
        family="local_rp",
        seed=20260721,
        parameters={"pitch_std": 0.08, "roll_std": 0.06},
    )
    request = s1_owner.load_frontres_v015_deployment_composition_request(
        s1_owner.FrontRESV015DeploymentCompositionConfig(
            enabled=True,
            reference_path=str(reference),
            future_offsets=(1, 2),
            corruption_protocol=protocol,
        )
    )
    return reference, request


def _command(reset_helper, commands, *, num_envs: int = 4):
    return reset_helper._make_command(commands, reset_helper._FakeRobot(num_envs), num_envs)


def _runner(reset_helper, command):
    return type("Runner", (), {"env": reset_helper._FakeEnv(command, command.robot, command.num_envs)})()


def _expected_arrays(reference: Path):
    with np.load(reference, allow_pickle=False) as data:
        q29 = torch.as_tensor(np.asarray(data["joint_pos"]), dtype=torch.float32)
        dq29 = torch.as_tensor(np.asarray(data["joint_vel"]), dtype=torch.float32)
    return q29, dq29


def test_t_install_current_h_identity_and_provenance() -> None:
    reset_helper, commands, runtime, s1_helper, s1_owner = _owners()
    with tempfile.TemporaryDirectory() as tmp:
        reference, request = _request(s1_helper, s1_owner, Path(tmp))
        command = _command(reset_helper, commands)
        command.set_frontres_v015_deployment_sequence(request)
        snapshot = command.frontres_v015_deployment_sequence_snapshot()
        q29, dq29 = _expected_arrays(reference)

        assert set(snapshot) == {
            "env_ids",
            "frame_indices",
            "current_q29_dq29",
            "intent_q29",
            "future_offsets",
            "reference_paths",
            "reference_stream_ids",
            "reference_file_hashes",
            "corruption_ids",
            "corruption_protocol_hashes",
            "corruption_families",
            "corruption_temporal_modes",
            "evaluation_kinds",
            "provenance",
        }
        assert tuple(snapshot["current_q29_dq29"].shape) == (4, 58)
        assert tuple(snapshot["intent_q29"].shape) == (4, 3, 29)
        torch.testing.assert_close(snapshot["current_q29_dq29"][:, :29], q29[0].expand(4, -1))
        torch.testing.assert_close(snapshot["current_q29_dq29"][:, 29:], dq29[0].expand(4, -1))
        torch.testing.assert_close(snapshot["intent_q29"], q29[:3].unsqueeze(0).expand(4, -1, -1))
        assert snapshot["future_offsets"] == (1, 2)
        assert snapshot["reference_stream_ids"] == (request.reference_stream_id,) * 4
        assert snapshot["reference_file_hashes"] == (request.reference_file_hash,) * 4
        assert snapshot["corruption_ids"] == (request.corruption_protocol.corruption_id,) * 4
        assert snapshot["corruption_protocol_hashes"] == (request.corruption_protocol.protocol_hash,) * 4
        assert snapshot["provenance"] == (
            {
                "reference_provenance": "deployment_reference_stream",
                "current_command_provenance": "deployment_q29_dq29",
                "intent_q29_provenance": "deployment_noisy_q29",
                "intent_q29_source": "deployment_npz_joint_pos",
            },
        ) * 4

        bridge = runtime.read_frontres_v015_deployment_context(_runner(reset_helper, command))
        torch.testing.assert_close(bridge["current_q29_dq29"], snapshot["current_q29_dq29"])
        torch.testing.assert_close(bridge["intent_q29"], snapshot["intent_q29"])
        assert bridge["reference_stream_ids"] == snapshot["reference_stream_ids"]
        assert bridge["corruption_protocol_hashes"] == snapshot["corruption_protocol_hashes"]
    print(
        "[T-install/T-current/T-H/T-identity/T-provenance] "
        "request -> command [B,58]/[B,H+1,29] -> read-only runtime bridge",
        flush=True,
    )


def test_t_frame_order_cursor_boundary_and_read_only() -> None:
    reset_helper, commands, runtime, s1_helper, s1_owner = _owners()
    with tempfile.TemporaryDirectory() as tmp:
        reference, request = _request(s1_helper, s1_owner, Path(tmp), frame_count=5)
        command = _command(reset_helper, commands)
        command.set_frontres_v015_deployment_sequence(request)
        q29, dq29 = _expected_arrays(reference)

        first = command.frontres_v015_deployment_sequence_snapshot()
        first["current_q29_dq29"].fill_(-1.0)
        first["intent_q29"].fill_(-2.0)
        first["provenance"][0]["intent_q29_source"] = "mutated"
        unchanged = command.frontres_v015_deployment_sequence_snapshot()
        torch.testing.assert_close(unchanged["intent_q29"][0], q29[:3])
        assert unchanged["provenance"][0]["intent_q29_source"] == "deployment_npz_joint_pos"

        command.advance_frontres_v015_deployment_sequence()
        second = runtime.read_frontres_v015_deployment_context(_runner(reset_helper, command))
        assert second["frame_indices"].tolist() == [1, 1, 1, 1]
        torch.testing.assert_close(second["current_q29_dq29"][0, :29], q29[1])
        torch.testing.assert_close(second["current_q29_dq29"][0, 29:], dq29[1])
        torch.testing.assert_close(second["intent_q29"][0], q29[1:4])
        command.advance_frontres_v015_deployment_sequence()
        last = command.frontres_v015_deployment_sequence_snapshot()
        assert last["frame_indices"].tolist() == [2, 2, 2, 2]
        torch.testing.assert_close(last["intent_q29"][0], q29[2:5])

        _expect_error(
            RuntimeError,
            command.advance_frontres_v015_deployment_sequence,
            "cannot clamp",
        )
        after_reject = command.frontres_v015_deployment_sequence_snapshot()
        assert after_reject["frame_indices"].tolist() == [2, 2, 2, 2]
    print(
        "[T-frame-order/T-cursor/T-boundary/T-read-only] "
        "cursor advances exactly one frame and rejects before H would clamp",
        flush=True,
    )


def test_t_row_alignment_mixed_reference_and_hash_reject() -> None:
    reset_helper, commands, _runtime, s1_helper, s1_owner = _owners()
    with tempfile.TemporaryDirectory() as tmp:
        reference, request = _request(s1_helper, s1_owner, Path(tmp))
        command = _command(reset_helper, commands)
        command.set_frontres_v015_deployment_sequence(request)
        permutation = torch.tensor([3, 1, 0, 2], dtype=torch.long)
        permuted = command.frontres_v015_deployment_sequence_snapshot(permutation)
        assert torch.equal(permuted["env_ids"], permutation)
        assert permuted["frame_indices"].tolist() == [0, 0, 0, 0]
        assert permuted["reference_stream_ids"] == (request.reference_stream_id,) * 4

        class _MixedCommand:
            robot = command.robot
            num_envs = command.num_envs

            def frontres_v015_deployment_sequence_snapshot(self, env_ids=None):
                snapshot = command.frontres_v015_deployment_sequence_snapshot(env_ids)
                hashes = list(snapshot["corruption_protocol_hashes"])
                hashes[-1] = "a" * 64
                snapshot["corruption_protocol_hashes"] = tuple(hashes)
                return snapshot

        mixed_runner = _runner(reset_helper, _MixedCommand())
        _expect_error(
            RuntimeError,
            lambda: _runtime.read_frontres_v015_deployment_context(mixed_runner),
            "mixed reference",
        )
        _expect_error(
            RuntimeError,
            lambda: command.set_frontres_v015_deployment_sequence(request),
            "already active",
        )

        mixed = _command(reset_helper, commands)
        mixed._frontres_local_scenario_active[0] = True
        _expect_error(
            RuntimeError,
            lambda: mixed.set_frontres_v015_deployment_sequence(request),
            "local scenario",
        )
        mixed_fixed = _command(reset_helper, commands)
        mixed_fixed._frontres_fixed_noisy_tape_context_active[0] = True
        _expect_error(
            RuntimeError,
            lambda: mixed_fixed.set_frontres_v015_deployment_sequence(request),
            "fixed Noisy tape",
        )
        mixed_legacy = _command(reset_helper, commands)
        mixed_legacy._frontres_reference_window_active[0] = True
        _expect_error(
            RuntimeError,
            lambda: mixed_legacy.set_frontres_v015_deployment_sequence(request),
            "legacy reference window",
        )

        stale = _command(reset_helper, commands)
        s1_helper._write_npz(reference, frame_count=6, offset=1.0)
        _expect_error(
            RuntimeError,
            lambda: stale.set_frontres_v015_deployment_sequence(request),
            "file hash",
        )
        assert not bool(stale._frontres_v015_deployment_sequence_active.any())
    print(
        "[T-row-alignment/T-mixed-reference/T-hash] "
        "row permutation is explicit; active local/reinstall/stale file identity reject",
        flush=True,
    )


def test_t_execution_and_training_isolation() -> None:
    reset_helper, commands, runtime, s1_helper, s1_owner = _owners()
    with tempfile.TemporaryDirectory() as tmp:
        _reference, request = _request(s1_helper, s1_owner, Path(tmp))
        command = _command(reset_helper, commands)
        command.set_frontres_v015_deployment_sequence(request)
        before = command.frontres_v015_deployment_sequence_snapshot()
        bridge = runtime.read_frontres_v015_deployment_context(_runner(reset_helper, command))
        after = command.frontres_v015_deployment_sequence_snapshot()
        assert torch.equal(before["frame_indices"], bridge["frame_indices"])
        assert torch.equal(before["frame_indices"], after["frame_indices"])

        append_source = inspect.getsource(runtime.append_frontres_future_intent_context)
        command_source = inspect.getsource(commands.MultiMotionCommand.command.fget)
        clock_source = inspect.getsource(commands.MultiMotionCommand._advance_frontres_command_clock)
        assert "frontres_v015_deployment_sequence" not in append_source
        assert "_gather_future_by_motion" in command_source
        assert "deployment_current_hold" in clock_source
        forbidden = ("storage", "return_evidence", "priority", "ppo", "optimizer", "report")
        bridge_source = inspect.getsource(runtime.read_frontres_v015_deployment_context).lower()
        assert all(name not in bridge_source for name in forbidden)

        command.clear_frontres_v015_deployment_sequence()
        _expect_error(
            RuntimeError,
            command.frontres_v015_deployment_sequence_snapshot,
            "active carrier",
        )
    print(
        "[T-read-isolation/T-no-training-state/T-close] "
        "snapshot reads do not advance; S2B owns the separate command/clock execution connector",
        flush=True,
    )


def test_t_g4_materialized_carrier_to_current_h() -> None:
    reset_helper, commands, _runtime, s1_helper, owner = _owners()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "ordinary_reference.npz"
        carrier_path = root / "controlled_carrier.npz"
        s1_helper._write_npz(source, frame_count=6)
        protocol = owner.build_frontres_v015_persistent_corruption_protocol(
            corruption_id="g4-s2-current-h",
            family="local_rp",
            seed=17,
            parameters={"roll_std": 0.04, "pitch_std": 0.07, "root_body_index": 0},
        )
        carrier = owner.FrontRESV015DeploymentCarrierLifecycle(
            source_path=str(source),
            output_path=str(carrier_path),
            corruption_protocol=protocol,
        ).materialize()
        request = owner.load_frontres_v015_deployment_composition_request(
            owner.FrontRESV015DeploymentCompositionConfig(
                enabled=True,
                reference_path=carrier.carrier_path,
                future_offsets=(1, 2),
                corruption_protocol=protocol,
            )
        )
        assert request.reference_file_hash == carrier.carrier_file_hash
        command = _command(reset_helper, commands, num_envs=2)
        command.set_frontres_v015_deployment_sequence(request)
        snapshot = command.frontres_v015_deployment_sequence_snapshot()
        q29, dq29 = _expected_arrays(carrier_path)
        assert tuple(snapshot["current_q29_dq29"].shape) == (2, 58)
        assert tuple(snapshot["intent_q29"].shape) == (2, 3, 29)
        torch.testing.assert_close(snapshot["current_q29_dq29"][0, :29], q29[0])
        torch.testing.assert_close(snapshot["current_q29_dq29"][0, 29:], dq29[0])
        torch.testing.assert_close(snapshot["intent_q29"][0], q29[:3])
        assert snapshot["reference_file_hashes"] == (carrier.carrier_file_hash,) * 2
        assert snapshot["provenance"] == (
            {
                "reference_provenance": "deployment_reference_stream",
                "current_command_provenance": "deployment_q29_dq29",
                "intent_q29_provenance": "deployment_noisy_q29",
                "intent_q29_source": "deployment_npz_joint_pos",
            },
        ) * 2
    print(
        "[T-G4-S2/T-current-H/T-carrier-identity] deterministic carrier is consumed as [B,58] + [B,H+1,29]",
        flush=True,
    )


def main() -> None:
    test_t_install_current_h_identity_and_provenance()
    test_t_frame_order_cursor_boundary_and_read_only()
    test_t_row_alignment_mixed_reference_and_hash_reject()
    test_t_execution_and_training_isolation()
    test_t_g4_materialized_carrier_to_current_h()
    print("frontres_v015_deployment_carrier_s2a_contract: ok", flush=True)


if __name__ == "__main__":
    main()
