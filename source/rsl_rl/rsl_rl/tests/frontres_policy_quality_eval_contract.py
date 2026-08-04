from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch
from torch import nn
from frontres_contract_imports import install_frontres_contract_packages


ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = ROOT / "source" / "rsl_rl" / "rsl_rl"
install_frontres_contract_packages(SOURCE_ROOT)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


manifest_module = _load(
    "rsl_rl.frontres.frontres_policy_quality_manifest",
    SOURCE_ROOT / "frontres" / "frontres_policy_quality_manifest.py",
)
quality = _load(
    "rsl_rl.runners.frontres_policy_quality_eval",
    SOURCE_ROOT / "runners" / "frontres_policy_quality_eval.py",
)
legacy_quality = _load(
    "rsl_rl.runners.frontres_policy_quality_legacy",
    SOURCE_ROOT / "runners" / "frontres_policy_quality_legacy.py",
)


class _Normalizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.ones(4))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations * self.scale


class _Robot:
    def __init__(self, rows: int) -> None:
        self.data = SimpleNamespace(
            root_state_w=torch.arange(rows * 13, dtype=torch.float32).reshape(rows, 13),
            joint_pos=torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3),
            joint_vel=torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3) * 0.1,
        )

    def write_root_state_to_sim(self, value: torch.Tensor, *, env_ids: torch.Tensor) -> None:
        self.data.root_state_w.index_copy_(0, env_ids, value)

    def write_joint_state_to_sim(
        self, positions: torch.Tensor, velocities: torch.Tensor, *, env_ids: torch.Tensor
    ) -> None:
        self.data.joint_pos.index_copy_(0, env_ids, positions)
        self.data.joint_vel.index_copy_(0, env_ids, velocities)


def _runner(rows: int = 2) -> tuple[SimpleNamespace, SimpleNamespace]:
    robot = _Robot(rows)
    command = SimpleNamespace(
        time_steps=torch.arange(rows, dtype=torch.long) + 5,
        env_motion_indices=torch.arange(rows, dtype=torch.long),
        _cached_perturbed_pos=torch.zeros(rows, 3),
        _cached_perturbed_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(rows, 1),
        _frontres_pos_correction=torch.zeros(rows, 3),
        _frontres_quat_correction=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(rows, 1),
        perturber=SimpleNamespace(
            _roll_state=torch.arange(rows, dtype=torch.float32) * 0.01,
            _pitch_state=torch.arange(rows, dtype=torch.float32) * 0.02,
        ),
    )
    scene = {"robot": robot}
    scene = SimpleNamespace(env_origins=torch.zeros(rows, 3), __getitem__=scene.__getitem__)

    class _Scene(SimpleNamespace):
        def __getitem__(self, key: str):
            return robot if key == "robot" else None

    raw = SimpleNamespace(
        scene=_Scene(env_origins=torch.zeros(rows, 3)),
        command_manager=SimpleNamespace(get_term=lambda name: command if name == "motion" else None),
        episode_length_buf=torch.zeros(rows, dtype=torch.long),
    )
    env = SimpleNamespace(unwrapped=raw, episode_length_buf=raw.episode_length_buf)
    return SimpleNamespace(env=env), command


def _payload(weight: float, *, poison: str) -> dict[str, object]:
    actor = nn.Linear(4, 6, bias=False)
    with torch.no_grad():
        actor.weight.fill_(weight)
    normalizer = _Normalizer()
    return {
        "model_state_dict": {"residual_actor": actor.state_dict()},
        "obs_norm_state_dict": normalizer.state_dict(),
        "optimizer_state_dict": {"poison": poison},
        "frontres_segment_sampler_state_dict": {"poison": poison},
        "frontres_segment_warmup_config": {"poison": poison},
    }


def test_three_routes_share_state_and_reuse_canonical_hooks() -> None:
    runner, command = _runner()
    comparison_signature = "d" * 64
    snapshot = quality.capture_frontres_policy_quality_state(
        runner,
        env_ids=(0, 1),
        comparison_signature=comparison_signature,
        role_layout=("policy", "clean"),
    )
    observation_identity = quality.FrontRESPolicyQualityObservationIdentity(
        expected_obs_dim=4,
        actor_input_dim=4,
        normalizer_identity="obs-norm:model-specific-sha256",
    )
    actor_template = nn.Linear(4, 6, bias=False)
    normalizer_template = _Normalizer()
    hsl = legacy_quality.FrozenFrontRESTaskActor.from_checkpoint_payload(
        route="hsl",
        checkpoint_identity="model_200:sha-hsl",
        checkpoint_payload=_payload(0.10, poison="hsl"),
        actor_template=actor_template,
        normalizer_template=normalizer_template,
        observation_identity=observation_identity,
    )
    policy = legacy_quality.FrozenFrontRESTaskActor.from_checkpoint_payload(
        route="policy",
        checkpoint_identity="model_701:sha-policy",
        checkpoint_payload=_payload(0.20, poison="policy"),
        actor_template=actor_template,
        normalizer_template=normalizer_template,
        observation_identity=observation_identity,
    )
    zero = legacy_quality.ZeroFrontRESTaskActor(observation_identity)
    assert all(
        not parameter.requires_grad
        for adapter in (hsl, policy)
        for module in (adapter.actor, adapter.normalizer)
        for parameter in module.parameters()
    )

    applied: list[torch.Tensor] = []
    counters = {"step": 0, "gain": 0, "execution": 0}

    def apply_action(actions: torch.Tensor) -> None:
        applied.append(actions.detach().clone())
        command._frontres_pos_correction.copy_(actions[:, :3])

    def step() -> None:
        counters["step"] += 1
        runner.env.unwrapped.scene["robot"].data.root_state_w[:, 0] += 1

    def gain() -> dict[str, float]:
        counters["gain"] += 1
        return {"gain_total": float(applied[-1].sum())}

    def execution() -> dict[str, int]:
        counters["execution"] += 1
        return {"steps": counters["step"]}

    isolation = {"optimizer": "fixed", "sampler": "fixed", "warmup": "fixed"}
    results = legacy_quality.run_frontres_policy_quality_counterfactuals(
        runner,
        snapshot=snapshot,
        comparison_signature=comparison_signature,
        adapters=(zero, hsl, policy),
        hooks=quality.FrontRESPolicyQualityRouteHooks(
            observe=lambda: torch.ones(2, 4),
            apply_action=apply_action,
            step=step,
            compute_gain=gain,
            capture_execution=execution,
        ),
        horizon_k=2,
        isolation_state=lambda: repr(isolation),
    )

    assert tuple(result.identity.route for result in results) == ("zero", "hsl", "policy")
    assert len({result.identity.comparison_signature for result in results}) == 1
    assert len({result.identity.state.initial_state_hash for result in results}) == 1
    assert all(tuple(result.actions.shape) == (2, 2, 6) for result in results)
    assert bool((results[0].actions == 0).all())
    assert not torch.equal(results[1].actions, results[2].actions)
    torch.testing.assert_close(results[0].actions, torch.zeros_like(results[0].actions))
    torch.testing.assert_close(results[2].actions, 2.0 * results[1].actions)
    assert float(results[2].actions.abs().max()) > 0.4
    assert len(applied) == 6 and counters == {"step": 6, "gain": 3, "execution": 3}
    print(
        "[quality counterfactual trace] "
        f"state_hash={results[0].identity.state.initial_state_hash} "
        f"zero_norm={float(results[0].actions.norm())} "
        f"hsl_norm={float(results[1].actions.norm())} policy_norm={float(results[2].actions.norm())}"
    )

    bad_isolation = {"optimizer": "fixed", "sampler": "fixed", "warmup": "fixed"}

    def mutate_training_state() -> None:
        bad_isolation["optimizer"] = "mutated"

    try:
        legacy_quality.run_frontres_policy_quality_counterfactuals(
            runner,
            snapshot=snapshot,
            comparison_signature=comparison_signature,
            adapters=(zero, hsl, policy),
            hooks=quality.FrontRESPolicyQualityRouteHooks(
                observe=lambda: torch.ones(2, 4),
                apply_action=lambda _actions: None,
                step=mutate_training_state,
                compute_gain=lambda: {"gain_total": 0.0},
                capture_execution=lambda: {"steps": 1},
            ),
            horizon_k=1,
            isolation_state=lambda: repr(bad_isolation),
        )
    except RuntimeError as exc:
        assert "optimizer/sampler/warmup" in str(exc)
    else:
        raise AssertionError("training-state mutation must fail closed")


def test_layout_source_and_isolation_fail_closed() -> None:
    assert "supervised_target" not in (SOURCE_ROOT / "runners" / "frontres_policy_quality_eval.py").read_text()
    identity = quality.FrontRESPolicyQualityObservationIdentity(4, 4, "norm-a")
    zero = legacy_quality.ZeroFrontRESTaskActor(identity)
    try:
        zero.action(torch.zeros(2, 5))
    except ValueError as exc:
        assert "dim mismatch" in str(exc)
    else:
        raise AssertionError("observation-layout mismatch must fail closed")

    payload = _payload(0.1, poison="ignored")
    del payload["obs_norm_state_dict"]
    try:
        legacy_quality.FrozenFrontRESTaskActor.from_checkpoint_payload(
            route="hsl",
            checkpoint_identity="model_200",
            checkpoint_payload=payload,
            actor_template=nn.Linear(4, 6, bias=False),
            normalizer_template=_Normalizer(),
            observation_identity=identity,
        )
    except ValueError as exc:
        assert "obs_norm_state_dict" in str(exc)
    else:
        raise AssertionError("missing normalizer identity must fail closed")


if __name__ == "__main__":
    test_three_routes_share_state_and_reuse_canonical_hooks()
    test_layout_source_and_isolation_fail_closed()
    print("PASS: isolated zero/HSL/policy counterfactual execution is closed offline.")
