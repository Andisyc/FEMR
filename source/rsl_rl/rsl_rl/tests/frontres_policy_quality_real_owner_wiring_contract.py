#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "source" / "rsl_rl"))

math_stub = types.ModuleType("isaaclab.utils.math")
math_stub.euler_xyz_from_quat = lambda quat: (quat[..., 0], quat[..., 1], quat[..., 2])
math_stub.quat_apply = lambda _quat, value: value
math_stub.quat_inv = lambda quat: quat
math_stub.quat_mul = lambda lhs, _rhs: lhs
math_stub.yaw_quat = lambda quat: quat
math_stub.quat_rotate_inverse = lambda _quat, value: value
sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))
sys.modules.setdefault("isaaclab.utils", types.ModuleType("isaaclab.utils"))
sys.modules["isaaclab.utils.math"] = math_stub

import rsl_rl

runners_package = types.ModuleType("rsl_rl.runners")
runners_package.__path__ = [str(ROOT / "source" / "rsl_rl" / "rsl_rl" / "runners")]
sys.modules["rsl_rl.runners"] = runners_package

from rsl_rl.frontres.frontres_policy_quality_manifest import (
    FrontRESPolicyQualityManifest,
    FrontRESPolicyQualityManifestItem,
    FrontRESPolicyQualityStateIdentity,
)
from rsl_rl.runners import frontres_policy_quality_eval as quality_eval
from rsl_rl.runners import frontres_policy_quality_formal_owners as formal


class _Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.residual_actor = nn.Sequential(nn.Linear(4, 6))
        self.num_actor_obs = 4
        self.max_delta_pos = 0.1
        self.max_delta_rpy = 0.2

    def get_env_action(self, observations: torch.Tensor, corrections: torch.Tensor) -> torch.Tensor:
        return torch.zeros((observations.shape[0], 2), dtype=observations.dtype)


class _Env:
    num_envs = 4
    device = torch.device("cpu")

    def get_observations(self):
        return torch.ones((4, 4)), {"observations": {"policy": torch.ones((4, 4))}}

    def step(self, actions: torch.Tensor):
        return torch.ones((4, 4)), torch.zeros(4), torch.zeros(4, dtype=torch.bool), {}


def _checkpoint(path: Path, policy: _Policy) -> None:
    torch.save(
        {
            "model_state_dict": {"residual_actor": policy.residual_actor.state_dict()},
            "obs_norm_state_dict": {},
        },
        path,
    )


def main() -> None:
    import tempfile

    calls = {name: 0 for name in ("reset", "observation", "action", "rollout", "gain", "execution")}
    policy = _Policy()
    parameter = nn.Parameter(torch.tensor(1.0))
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        env=_Env(),
        policy_obs_type="policy",
        obs_normalizer=nn.Identity(),
        _frontres_gmt_obs_dim=4,
        cfg={},
        current_learning_iteration=9,
        alg=SimpleNamespace(policy=policy, optimizer=torch.optim.SGD([parameter], lr=0.1)),
        _frontres_segment_sampler=SimpleNamespace(state_dict=lambda: {"priority": torch.tensor([1.0])}),
        _frontres_warmup_complete=False,
        _frontres_segment_actor_warmup_complete=False,
    )
    applied_actions = torch.zeros((4, 6))

    def apply_correction(actions, n_train, **kwargs):
        nonlocal applied_actions
        calls["action"] += 1
        assert tuple(actions.shape) == (4, 6)
        assert n_train == 1
        applied_actions = actions.detach().clone()

    runner._apply_frontres_task_corrections = apply_correction
    runner._apply_obs_normalizer = lambda obs: obs

    pair_layout = SimpleNamespace(n_train=1, n_candidate=1, n_base=1, n_clean=1)
    formal.configure_frontres_pair_layout = lambda *_args, **_kwargs: pair_layout
    formal.ensure_frontres_policy_quality_reset_support = lambda _runner: None

    def reset_hook(request):
        calls["reset"] += 1
        assert request.motion_ids == ("motion.npz",)
        assert request.start_frames.tolist() == [7]
        return {"success_mask": torch.ones(1, dtype=torch.bool)}

    formal._index_segment_reset_hook = lambda _env: reset_hook
    formal._index_reset_result_from_mapping = lambda mapping, request: SimpleNamespace(
        success_mask=mapping["success_mask"]
    )
    def capture_state(_runner, **kwargs):
        origins = torch.tensor(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.1], [20.0, 0.0, 0.2], [30.0, 0.0, 0.3]]
        )
        local_root = torch.tensor([[1.0, 2.0, 0.8]]).repeat(4, 1)
        root = torch.zeros((4, 13))
        root[:, :3] = local_root + origins
        root[:, 3] = 1.0
        root[:, 7:13] = 0.25
        cached_pos = torch.tensor([[0.1, -0.2, 0.3], [0.1, -0.2, 0.3], [0.1, -0.2, 0.3], [0.0, 0.0, 0.0]])
        cached_quat = torch.tensor(
            [[0.99, 0.1, 0.0, 0.0], [0.99, 0.1, 0.0, 0.0], [0.99, 0.1, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        )
        image = quality_eval._TensorImage.capture
        return SimpleNamespace(
            comparison_signature=kwargs["comparison_signature"],
            initial_state_hash="a" * 64,
            role_layout=tuple(kwargs["role_layout"]),
            root_state_w=image(root),
            env_origins=image(origins),
            joint_pos=image(torch.full((4, 29), 0.4)),
            joint_vel=image(torch.full((4, 29), 0.2)),
            command_state=(
                ("_cached_perturbed_pos", image(cached_pos)),
                ("_cached_perturbed_quat", image(cached_quat)),
            ),
        )

    formal.capture_frontres_policy_quality_state = capture_state
    quality_eval.restore_frontres_policy_quality_state = lambda _runner, snapshot, **kwargs: (
        FrontRESPolicyQualityStateIdentity(kwargs["comparison_signature"], snapshot.initial_state_hash)
    )

    def motion_capture(_runner, _layout):
        calls["execution"] += 1
        clean = torch.zeros((1, 2, 3))
        noisy = torch.full((1, 2, 3), 0.2)
        repair_effect = applied_actions[:1].abs().mean().clamp(max=0.1)
        repaired = noisy - repair_effect
        return clean, repaired, noisy

    formal._capture_motion_quality_frame = motion_capture
    formal._capture_root_orientation_frame = lambda *_args: (None, None, None)
    def physics_capture(*_args):
        repair_effect = applied_actions[:1].abs().mean().clamp(max=0.1)
        noisy_zmp = torch.tensor([0.1])
        return noisy_zmp + repair_effect, noisy_zmp, torch.ones(1), torch.ones(1)

    formal._capture_physics_frame = physics_capture
    original_gain = formal.compute_segment_gain

    def counted_gain(**kwargs):
        calls["gain"] += 1
        assert kwargs["clean_positions"].shape[:2] == (1, 3)
        assert kwargs["repaired_positions"].shape[:2] == (1, 3)
        assert kwargs["noisy_positions"].shape[:2] == (1, 3)
        assert kwargs["action_steps"].shape[:2] == (3, 1)
        result = original_gain(**kwargs)
        for key in ("style_gain", "physics_gain", "repair_cost", "gain_total"):
            value = getattr(result, key)
            assert value.shape == (1,), (key, value.shape)
        return result

    formal.compute_segment_gain = counted_gain
    original_observe = formal._RouteCapture.observe
    original_step = formal._RouteCapture.step

    def counted_observe(self):
        calls["observation"] += 1
        return original_observe(self)

    def counted_step(self):
        calls["rollout"] += 1
        return original_step(self)

    formal._RouteCapture.observe = counted_observe
    formal._RouteCapture.step = counted_step

    manifest = FrontRESPolicyQualityManifest(
        environment_revision="fake-env-v1",
        config_revision="fake-config-v1",
        evaluator_version="quality-v1",
        items=(
            FrontRESPolicyQualityManifestItem(
                item_id="item-0",
                motion_id="motion.npz",
                start_frame=7,
                perturbation_family="local_rp",
                perturbation_parameters=(("strength", 0.5),),
                effective_horizon_k=3,
                seed=11,
            ),
        ),
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest_path = root / "manifest.json"
        hsl_path = root / "hsl.pt"
        policy_path = root / "policy.pt"
        result_path = root / "result.json"
        manifest_path.write_text(manifest.to_json())
        _checkpoint(hsl_path, policy)
        _checkpoint(policy_path, policy)
        policy_payload = torch.load(policy_path, weights_only=False)
        policy_payload["model_state_dict"]["residual_actor"]["0.bias"] += 0.01
        torch.save(policy_payload, policy_path)

        before = formal._training_state_signature(runner)
        payload = quality_eval.run_frontres_policy_quality_eval(
            runner,
            manifest_path=str(manifest_path),
            hsl_checkpoint_path=str(hsl_path),
            policy_checkpoint_path=str(policy_path),
            result_path=str(result_path),
        )
        after = formal._training_state_signature(runner)

        assert result_path.is_file()
        assert payload["owner_identity"] == dict(formal._OWNER_IDENTITY)
        assert before == after
        assert calls == {
            "reset": 1,
            "observation": 9,
            "action": 9,
            "rollout": 9,
            "gain": 3,
            "execution": 9,
        }
        assert callable(runner._frontres_policy_quality_manifest_executor)
        role_identity = payload["rows"][0]["role_identity"]
        assert role_identity["policy_noisy"]["world_root_pos_max_abs"] == 20.0
        for key, value in role_identity["policy_noisy"].items():
            if key not in ("world_root_pos_max_abs", "env_origin_max_abs"):
                assert value == 0.0, (key, value)
        torch.testing.assert_close(
            torch.tensor(role_identity["corruption_present"]["policy_clean_cached_pos_max_abs"]),
            torch.tensor(0.3),
        )
        torch.testing.assert_close(
            torch.tensor(role_identity["corruption_present"]["policy_clean_cached_quat_max_abs"]),
            torch.tensor(0.1),
        )
        zero_route = payload["rows"][0]["routes"]["zero"]
        assert zero_route["checkpoint_identity"] == "zero:no-checkpoint"
        assert zero_route["actions"] == [[[0.0] * 6] * 4] * 3
        for key in ("style_gain", "physics_gain", "repair_cost", "gain_total"):
            torch.testing.assert_close(torch.tensor(zero_route["gain"][key]), torch.zeros(1), atol=1.0e-7, rtol=0.0)

    print("PASS: official policy-quality entry installs and reaches all six real owner adapters offline.")


if __name__ == "__main__":
    main()
