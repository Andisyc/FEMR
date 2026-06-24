"""TEST ONLY: real-batch FrontRES authority critic replay.

This script answers one narrow question:

    Given one real minibatch dumped from formal training, are the authority
    critic targets themselves learnable and directionally consistent?

It does not prove the full method succeeds.  It isolates the critic target
chain:

    obs + proposal + rho -> Q
    Q(..., behavior_rho) -> executed FrontRES K-step delta return
    Q(..., 0)            -> Noisy/GMT endpoint delta return
    Q(..., 1)            -> Candidate/full-write endpoint delta return

Create a real dump during training with:

    FRONTRES_LIVE_BATCH_DUMP=/tmp/frontres_live_batch.pt

Optional controls:

    FRONTRES_LIVE_BATCH_DUMP_IT=20
    FRONTRES_LIVE_BATCH_DUMP_UPDATE=0
    FRONTRES_LIVE_BATCH_DUMP_MAX=20000

Then replay it with:

    python source/rsl_rl/rsl_rl/tests/frontres_authority_critic_replay.py \
        --dump /tmp/frontres_live_batch.pt --device cuda:0
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AuthorityReplayBatch:
    obs: torch.Tensor
    proposal: torch.Tensor
    behavior_rho: torch.Tensor
    target_behavior: torch.Tensor
    target_zero: torch.Tensor
    target_one: torch.Tensor
    mask: torch.Tensor
    config: dict
    source: str


class TinyAuthorityCritic(nn.Module):
    def __init__(self, obs_dim: int, proposal_dim: int, rho_dim: int, hidden_dim: int, linear: bool) -> None:
        super().__init__()
        in_dim = obs_dim + proposal_dim + rho_dim
        if linear:
            self.net = nn.Linear(in_dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, obs: torch.Tensor, proposal: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, proposal, rho], dim=-1))


def _normalize(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-6)
    return (x - mean) / std


def _load_real_batch(path: str, device: torch.device, max_samples: int) -> AuthorityReplayBatch:
    payload = torch.load(path, map_location="cpu")
    required = (
        "obs",
        "proposal_delta_se",
        "authority_action",
        "authority_return_k",
        "authority_return_zero_k",
        "authority_return_one_k",
        "authority_mask",
    )
    missing = [key for key in required if payload.get(key) is None]
    if missing:
        raise KeyError(
            "dump file is missing authority critic replay fields: "
            f"{missing}. Regenerate the dump after the live-batch dump patch."
        )

    n = int(payload["obs"].shape[0])
    if max_samples > 0:
        n = min(n, int(max_samples))
    sample = slice(0, n)

    obs = payload["obs"][sample].float()
    proposal = payload["proposal_delta_se"][sample].float()
    behavior_rho = payload["authority_action"][sample].float()
    target_behavior = payload["authority_return_k"][sample].float().view(n, -1)[:, :1]
    target_zero = payload["authority_return_zero_k"][sample].float().view(n, -1)[:, :1]
    target_one = payload["authority_return_one_k"][sample].float().view(n, -1)[:, :1]
    mask = payload["authority_mask"][sample].float().view(n, -1)[:, :1]

    return AuthorityReplayBatch(
        obs=_normalize(obs).to(device),
        proposal=_normalize(proposal).to(device),
        behavior_rho=behavior_rho.clamp(0.0, 1.0).to(device),
        target_behavior=target_behavior.to(device),
        target_zero=target_zero.to(device),
        target_one=target_one.to(device),
        mask=mask.clamp(0.0, 1.0).to(device),
        config=dict(payload.get("config", {})),
        source=path,
    )


def _make_synthetic_batch(device: torch.device, n: int, seed: int) -> AuthorityReplayBatch:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    obs = torch.randn(n, 12, generator=generator)
    proposal = torch.randn(n, 6, generator=generator) * 0.4
    behavior_rho = torch.rand(n, 6, generator=generator)
    latent = (
        0.55 * proposal[:, 0]
        - 0.35 * proposal[:, 3]
        + 0.25 * obs[:, 1]
        - 0.20 * obs[:, 5]
    ).view(n, 1)
    target_zero = torch.zeros(n, 1)
    target_one = latent.tanh()
    target_behavior = behavior_rho.mean(dim=-1, keepdim=True) * target_one
    mask = torch.ones(n, 1)
    return AuthorityReplayBatch(
        obs=_normalize(obs).to(device),
        proposal=_normalize(proposal).to(device),
        behavior_rho=behavior_rho.to(device),
        target_behavior=target_behavior.to(device),
        target_zero=target_zero.to(device),
        target_one=target_one.to(device),
        mask=mask.to(device),
        config={"synthetic": True},
        source="synthetic",
    )


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(dtype=pred.dtype)
    return (((pred - target) ** 2) * weight).sum() / weight.sum().clamp(min=1.0)


def _losses(model: TinyAuthorityCritic, batch: AuthorityReplayBatch, indices: torch.Tensor) -> tuple[torch.Tensor, dict]:
    obs = batch.obs[indices]
    proposal = batch.proposal[indices]
    behavior_rho = batch.behavior_rho[indices]
    target_behavior = batch.target_behavior[indices]
    target_zero = batch.target_zero[indices]
    target_one = batch.target_one[indices]
    mask = batch.mask[indices]
    zeros = torch.zeros_like(behavior_rho)
    ones = torch.ones_like(behavior_rho)

    q_behavior = model(obs, proposal, behavior_rho)
    q_zero = model(obs, proposal, zeros)
    q_one = model(obs, proposal, ones)

    behavior_loss = _masked_mse(q_behavior, target_behavior, mask)
    zero_loss = _masked_mse(q_zero, target_zero, mask)
    one_loss = _masked_mse(q_one, target_one, mask)
    total = (behavior_loss + zero_loss + one_loss) / 3.0
    return total, {
        "behavior_loss": float(behavior_loss.detach().item()),
        "zero_loss": float(zero_loss.detach().item()),
        "one_loss": float(one_loss.detach().item()),
        "total_loss": float(total.detach().item()),
    }


@torch.no_grad()
def _corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.flatten()
    y = y.flatten()
    if x.numel() < 2:
        return math.nan
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom.item()) <= 1e-12:
        return math.nan
    return float((x * y).sum().div(denom).item())


@torch.no_grad()
def _readout(model: TinyAuthorityCritic, batch: AuthorityReplayBatch, indices: torch.Tensor, margin: float) -> dict:
    obs = batch.obs[indices]
    proposal = batch.proposal[indices]
    behavior_rho = batch.behavior_rho[indices]
    zeros = torch.zeros_like(behavior_rho)
    ones = torch.ones_like(behavior_rho)
    mask = batch.mask[indices] > 0.5
    target_delta = batch.target_one[indices] - batch.target_zero[indices]
    q_delta = model(obs, proposal, ones) - model(obs, proposal, zeros)
    valid = mask & (target_delta.abs() > margin)
    if bool(valid.any().item()):
        sign_acc = (torch.sign(q_delta[valid]) == torch.sign(target_delta[valid])).float().mean().item()
    else:
        sign_acc = math.nan
    return {
        "active": int(mask.sum().item()),
        "target_behavior_mean": float(batch.target_behavior[indices][mask].mean().item()) if bool(mask.any().item()) else math.nan,
        "target_zero_mean": float(batch.target_zero[indices][mask].mean().item()) if bool(mask.any().item()) else math.nan,
        "target_one_mean": float(batch.target_one[indices][mask].mean().item()) if bool(mask.any().item()) else math.nan,
        "target_delta_mean": float(target_delta[mask].mean().item()) if bool(mask.any().item()) else math.nan,
        "target_delta_std": float(target_delta[mask].std(unbiased=False).item()) if bool(mask.any().item()) else math.nan,
        "q_delta_mean": float(q_delta[mask].mean().item()) if bool(mask.any().item()) else math.nan,
        "q_delta_std": float(q_delta[mask].std(unbiased=False).item()) if bool(mask.any().item()) else math.nan,
        "delta_corr": _corr(q_delta[mask], target_delta[mask]) if bool(mask.any().item()) else math.nan,
        "delta_sign_acc": float(sign_acc),
    }


def _print_readout(label: str, losses: dict, readout: dict) -> None:
    print(
        f"{label:<8} "
        f"loss={losses['total_loss']:.5f} "
        f"beh/0/1={losses['behavior_loss']:.5f}/"
        f"{losses['zero_loss']:.5f}/{losses['one_loss']:.5f} "
        f"ret0/1={readout['target_zero_mean']:+.4f}/"
        f"{readout['target_one_mean']:+.4f} "
        f"dret={readout['target_delta_mean']:+.4f}±{readout['target_delta_std']:.4f} "
        f"dq={readout['q_delta_mean']:+.4f}±{readout['q_delta_std']:.4f} "
        f"corr={readout['delta_corr']:+.3f} "
        f"sign={readout['delta_sign_acc']:.3f}"
    )


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if args.dump:
        batch = _load_real_batch(args.dump, device, args.max_samples)
    else:
        batch = _make_synthetic_batch(device, args.synthetic_samples, args.seed)

    active = (batch.mask[:, 0] > 0.5).nonzero(as_tuple=False).flatten()
    if active.numel() == 0:
        raise AssertionError("authority replay batch has no active authority samples.")

    torch.manual_seed(args.seed)
    perm = active[torch.randperm(active.numel(), device=device)]
    holdout = int(round(float(args.holdout_frac) * int(perm.numel())))
    test_idx = perm[:holdout] if holdout > 0 else perm[:0]
    train_idx = perm[holdout:] if holdout > 0 else perm

    model = TinyAuthorityCritic(
        obs_dim=batch.obs.shape[-1],
        proposal_dim=batch.proposal.shape[-1],
        rho_dim=batch.behavior_rho.shape[-1],
        hidden_dim=args.hidden_dim,
        linear=args.linear,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    with torch.no_grad():
        initial_loss, initial_parts = _losses(model, batch, train_idx)
        initial_readout = _readout(model, batch, train_idx, args.sign_margin)

    print("=== FrontRES Authority Critic Real-Batch Replay TEST ONLY ===")
    print(f"source={batch.source}")
    print(
        f"device={device}, active={active.numel()}, train={train_idx.numel()}, "
        f"test={test_idx.numel()}, obs_dim={batch.obs.shape[-1]}, rho_dim={batch.behavior_rho.shape[-1]}"
    )
    print(f"config={batch.config}")
    _print_readout("initial", initial_parts, initial_readout)

    for step in range(1, args.steps + 1):
        if args.batch_size > 0 and args.batch_size < train_idx.numel():
            pick = torch.randint(0, train_idx.numel(), (args.batch_size,), device=device)
            idx = train_idx[pick]
        else:
            idx = train_idx
        loss, _ = _losses(model, batch, idx)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.print_every == 0 or step == args.steps:
            with torch.no_grad():
                train_loss, train_parts = _losses(model, batch, train_idx)
                train_readout = _readout(model, batch, train_idx, args.sign_margin)
            print(f"step={step:05d}")
            _print_readout("train", train_parts, train_readout)
            if test_idx.numel() > 0:
                with torch.no_grad():
                    test_loss, test_parts = _losses(model, batch, test_idx)
                    test_readout = _readout(model, batch, test_idx, args.sign_margin)
                _print_readout("test", test_parts, test_readout)

    eval_idx = test_idx if test_idx.numel() > 0 else train_idx
    with torch.no_grad():
        final_loss, final_parts = _losses(model, batch, eval_idx)
        final_readout = _readout(model, batch, eval_idx, args.sign_margin)
    initial_eval = float(initial_loss.item())
    final_eval = float(final_loss.item())
    loss_ratio = final_eval / max(initial_eval, 1e-12)
    corr = final_readout["delta_corr"]
    sign_acc = final_readout["delta_sign_acc"]

    print("final")
    _print_readout("eval", final_parts, final_readout)
    print(f"loss_ratio={loss_ratio:.4f}")

    corr_ok = math.isnan(corr) or corr >= args.pass_corr
    sign_ok = math.isnan(sign_acc) or sign_acc >= args.pass_sign_acc
    loss_ok = loss_ratio <= args.pass_loss_ratio
    passed = loss_ok and corr_ok and sign_ok
    print(
        "meaning: loss tests whether endpoint targets are learnable; "
        "corr/sign test whether Q(1)-Q(0) matches Candidate-vs-Noisy direction."
    )
    print(f"result: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", default="", help="Path produced by FRONTRES_LIVE_BATCH_DUMP. If omitted, run synthetic self-check.")
    parser.add_argument("--device", default="", help="cpu, cuda, cuda:0, ...")
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--synthetic-samples", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--linear", action="store_true", help="Use a linear critic instead of a 2-layer MLP.")
    parser.add_argument("--sign-margin", type=float, default=1e-4)
    parser.add_argument("--pass-loss-ratio", type=float, default=0.80)
    parser.add_argument("--pass-corr", type=float, default=0.20)
    parser.add_argument("--pass-sign-acc", type=float, default=0.55)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
