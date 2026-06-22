# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TEST ONLY: FrontRES update memory pipeline.

Run from the repository root with:

    python source/rsl_rl/rsl_rl/tests/frontres_update_memory_pipeline.py

Optional CUDA stress run with the tiny policy:

    python source/rsl_rl/rsl_rl/tests/frontres_update_memory_pipeline.py --device cuda:0 --live-size

Formal FrontRES policy stress run:

    python source/rsl_rl/rsl_rl/tests/frontres_update_memory_pipeline.py --device cuda:0 --live-size --policy frontres

This is not an IsaacLab run.  It repeatedly fills a synthetic RolloutStorage and
calls the formal FrontRESUnified.update() path.  The test answers two questions:

1. Does update-entry memory rise across repeated updates?
2. Which update stage creates the peak: value, supervised actor, or rho?

If this test is stable but live training still grows, the leak is probably
outside the algorithm update path: runner, environment, diagnostics, storage
ownership, or external GPU pressure.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from rsl_rl.algorithms.frontres_unified import FrontRESUnified
from rsl_rl.modules import FrontRESActorCritic
from rsl_rl.storage.rollout_storage import RolloutStorage
from frontres_region_direct_update_path import TinyFrontRESPolicy


ACTIVE_GMT_CANDIDATES = (
    "/home/yuxuancheng/MOSAIC/model/model_27000.pt",
    "/hdd1/cyx/MOSAIC/model/model_27000.pt",
)


def _gib(value: int) -> float:
    return float(value) / (1024.0 ** 3)


def _cuda_snapshot(device: torch.device, label: str) -> dict[str, float]:
    if device.type != "cuda":
        return {"label": label}
    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "label": label,
        "alloc": _gib(torch.cuda.memory_allocated(device)),
        "reserved": _gib(torch.cuda.memory_reserved(device)),
        "max_alloc": _gib(torch.cuda.max_memory_allocated(device)),
        "max_reserved": _gib(torch.cuda.max_memory_reserved(device)),
        "free": _gib(free_bytes),
        "total": _gib(total_bytes),
    }


def _print_snapshot(snapshot: dict[str, float], *, update_idx: int) -> None:
    if "alloc" not in snapshot:
        print(f"update={update_idx:03d} label={snapshot['label']} device=cpu")
        return
    print(
        f"update={update_idx:03d} label={snapshot['label']} "
        f"alloc={snapshot['alloc']:.2f}GiB reserved={snapshot['reserved']:.2f}GiB "
        f"max_alloc={snapshot['max_alloc']:.2f}GiB max_reserved={snapshot['max_reserved']:.2f}GiB "
        f"free={snapshot['free']:.2f}GiB total={snapshot['total']:.2f}GiB"
    )


def _make_storage(
    *,
    device: torch.device,
    num_envs: int,
    num_transitions: int,
    obs_dim: int,
    critic_obs_dim: int,
) -> RolloutStorage:
    return RolloutStorage(
        training_type="frontres",
        num_envs=num_envs,
        num_transitions_per_env=num_transitions,
        obs_shape=(obs_dim,),
        privileged_obs_shape=(critic_obs_dim,),
        actions_shape=(12,),
        device=str(device),
    )


def _fill_storage(
    storage: RolloutStorage,
    policy: TinyFrontRESPolicy,
    *,
    device: torch.device,
    obs_dim: int,
    critic_obs_dim: int,
) -> None:
    for _ in range(storage.num_transitions_per_env):
        n = storage.num_envs
        obs = torch.randn(n, obs_dim, device=device)
        critic_obs = torch.randn(n, critic_obs_dim, device=device)
        with torch.no_grad():
            policy.update_distribution(obs)
            raw = policy.action_mean.detach()
            actions = raw.clone()
            values = policy.evaluate(critic_obs).detach()
            log_prob = policy.get_actions_log_prob(actions).detach()

        transition = RolloutStorage.Transition()
        transition.observations = obs
        transition.privileged_observations = critic_obs
        transition.actions = actions
        transition.rewards = torch.linspace(0.0, 1.0, n, device=device)
        transition.dones = torch.zeros(n, device=device)
        transition.values = values
        transition.actions_log_prob = log_prob
        transition.action_mean = raw
        transition.action_sigma = torch.full_like(raw, 0.05)
        transition.frontres_mask = torch.ones(n, 1, device=device)
        transition.frontres_actor_gate = torch.ones(n, 1, device=device)
        transition.supervised_target = torch.zeros(n, 6, device=device)
        transition.supervised_weight = torch.ones(n, 1, device=device)
        transition.supervised_harm_weight = torch.zeros(n, 1, device=device)

        base = torch.linspace(-1.0, 1.0, n, device=device).view(n, 1)
        transition.acceptance_target = base.expand(n, 6)
        transition.acceptance_mask = torch.ones(n, 6, device=device)
        transition.rho_prior_authority = (base < -0.6).float()
        transition.rho_prior_target = torch.zeros(n, 6, device=device)
        transition.state_alpha_target = torch.zeros(n, 1, device=device)
        transition.state_alpha_mask = torch.zeros(n, 1, device=device)
        transition.hidden_states = None
        storage.add_transitions(transition)

    storage.returns.copy_(storage.values)
    storage.advantages.zero_()


def _make_algorithm(
    *,
    policy: torch.nn.Module,
    device: torch.device,
    num_learning_epochs: int,
    num_mini_batches: int,
    cuda_memory_debug: bool,
) -> FrontRESUnified:
    return FrontRESUnified(
        policy,
        num_learning_epochs=num_learning_epochs,
        num_mini_batches=num_mini_batches,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1.0e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        device=str(device),
        lambda_supervised=0.25,
        diagnose_gradient_conflict=False,
        frontres_training_objective="hsl_hybrid",
        frontres_acceptance_preference_weight=0.0,
        frontres_state_alpha_weight=0.0,
        frontres_structured_joint_rl_enabled=True,
        frontres_structured_joint_rl_weight=1.0,
        frontres_structured_joint_rl_loss_mode="region_direct",
        frontres_structured_joint_rl_disable_generic_ppo=True,
        frontres_structured_joint_rl_keep_legacy_bce=False,
        frontres_structured_joint_prior_loss_weight=1.0,
        frontres_structured_joint_repair_loss_kind="bce_logit",
        frontres_structured_joint_repair_loss_scale=1.0,
        frontres_cuda_memory_debug=cuda_memory_debug,
    )


def _find_gmt_checkpoint(explicit_path: str | None) -> str:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"explicit --gmt-checkpoint does not exist: {path}")
        return str(path)
    for candidate in ACTIVE_GMT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "No GMT checkpoint found. Pass --gmt-checkpoint explicitly. "
        f"Tried: {list(ACTIVE_GMT_CANDIDATES)}"
    )


def _make_policy(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if args.policy == "tiny":
        return TinyFrontRESPolicy(args.obs_dim, args.critic_obs_dim).to(device)

    gmt_checkpoint = _find_gmt_checkpoint(args.gmt_checkpoint)
    print(f"[test] Using real FrontRESActorCritic with GMT checkpoint: {gmt_checkpoint}")
    policy = FrontRESActorCritic(
        num_actor_obs=args.obs_dim,
        num_critic_obs=args.critic_obs_dim,
        num_actions=args.robot_action_dim,
        residual_hidden_dims=[512, 256, 128],
        residual_last_layer_gain=0.01,
        critic_hidden_dims=[1024, 1024, 512, 256],
        activation="elu",
        init_noise_std=0.01,
        noise_std_type="scalar",
        num_task_corrections=6,
        task_conf_dim=6,
        max_delta_pos=0.3,
        max_delta_rpy=0.4,
        gmt_checkpoint_path=gmt_checkpoint,
        init_critic_from_gmt=False,
        q_ref_start_idx=232,
        num_frontres_obs=0,
        frontres_split_acceptance_head=False,
        num_z_outputs=0,
        max_delta_q=0.5,
    )
    return policy.to(device)


def run_memory_pipeline(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    num_envs = args.num_envs
    num_transitions = args.num_transitions
    if args.live_size:
        num_envs = 36000
        num_transitions = 8

    obs_dim = args.obs_dim
    critic_obs_dim = args.critic_obs_dim
    total_samples = num_envs * num_transitions
    batch_size = total_samples // args.num_mini_batches

    policy = _make_policy(args, device)
    alg = _make_algorithm(
        policy=policy,
        device=device,
        num_learning_epochs=args.num_learning_epochs,
        num_mini_batches=args.num_mini_batches,
        cuda_memory_debug=args.cuda_memory_debug,
    )
    storage = _make_storage(
        device=device,
        num_envs=num_envs,
        num_transitions=num_transitions,
        obs_dim=obs_dim,
        critic_obs_dim=critic_obs_dim,
    )
    alg.storage = storage

    print("=== FrontRES Update Memory Pipeline TEST ONLY ===")
    print(
        f"policy={args.policy} device={device} updates={args.updates} total_samples={total_samples} "
        f"num_envs={num_envs} num_transitions={num_transitions} "
        f"epochs={args.num_learning_epochs} mini_batches={args.num_mini_batches} "
        f"batch={batch_size} obs_dim={obs_dim} critic_obs_dim={critic_obs_dim}"
    )
    print("meaning: update_entry growth => likely algorithm-side retention; stable entry => look outside update path.")

    entry_allocs: list[float] = []
    for update_idx in range(args.updates):
        alg.current_learning_iteration = update_idx
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        _fill_storage(
            storage,
            policy,
            device=device,
            obs_dim=obs_dim,
            critic_obs_dim=critic_obs_dim,
        )
        before = _cuda_snapshot(device, "before_update")
        _print_snapshot(before, update_idx=update_idx)
        entry_allocs.append(float(before.get("alloc", 0.0)))

        try:
            loss_dict = alg.update()
        except torch.cuda.OutOfMemoryError:
            _print_snapshot(_cuda_snapshot(device, "caught_oom_after_update"), update_idx=update_idx)
            raise

        after = _cuda_snapshot(device, "after_update")
        _print_snapshot(after, update_idx=update_idx)
        print(
            f"update={update_idx:03d} loss "
            f"value={loss_dict['value_function']:.4f} "
            f"sup={loss_dict['supervised_loss']:.4f} "
            f"rho={loss_dict['structured_joint_rl_loss']:.4f} "
            f"generic={loss_dict['ppo_actor_weight']:.1f}"
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if device.type == "cuda" and len(entry_allocs) >= 2:
        drift = entry_allocs[-1] - entry_allocs[0]
        print(f"entry_alloc_drift={drift:+.3f}GiB over {len(entry_allocs)} updates")
        if abs(drift) <= args.drift_tolerance_gib:
            print("result: PASS update-entry memory is stable within tolerance.")
        else:
            print("result: FAIL update-entry memory drift exceeds tolerance; inspect retained tensors/graphs.")
    else:
        print("result: PASS CPU path executed; use --device cuda:0 for memory measurements.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--policy", choices=("tiny", "frontres"), default="tiny")
    parser.add_argument("--gmt-checkpoint", default=None)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--num-transitions", type=int, default=4)
    parser.add_argument("--num-learning-epochs", type=int, default=5)
    parser.add_argument("--num-mini-batches", type=int, default=16)
    parser.add_argument("--obs-dim", type=int, default=800)
    parser.add_argument("--critic-obs-dim", type=int, default=859)
    parser.add_argument("--robot-action-dim", type=int, default=29)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--drift-tolerance-gib", type=float, default=0.10)
    parser.add_argument("--cuda-memory-debug", action="store_true")
    parser.add_argument(
        "--live-size",
        action="store_true",
        help="Use 36000 envs x 8 transitions, matching the observed 288k sample / 18k minibatch scale.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_memory_pipeline(parse_args())
