"""FrontRES live-batch rho replay test.

This is a TEST-ONLY script.  It answers one question:

    Given one real FrontRES minibatch from formal training, can a small
    state-conditioned rho head learn to assign higher rho to positive
    advantage samples than to negative advantage samples?

The training run must first dump one live minibatch by setting:

    FRONTRES_LIVE_BATCH_DUMP=/tmp/frontres_live_batch.pt

Optional dump controls:

    FRONTRES_LIVE_BATCH_DUMP_IT=1548
    FRONTRES_LIVE_BATCH_DUMP_UPDATE=0
    FRONTRES_LIVE_BATCH_DUMP_MAX=20000

Then run:

    python source/rsl_rl/rsl_rl/tests/frontres_live_batch_replay.py \
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
class Batch:
    obs: torch.Tensor
    rho_advantage: torch.Tensor
    rho_weight: torch.Tensor
    prior_authority: torch.Tensor
    prior_target: torch.Tensor
    current_rho_raw: torch.Tensor
    config: dict


class RhoReplayHead(nn.Module):
    def __init__(self, obs_dim: int, rho_dim: int, hidden_dim: int, linear: bool) -> None:
        super().__init__()
        if linear:
            self.net = nn.Linear(obs_dim, rho_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, rho_dim),
            )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def _load_batch(path: str, device: torch.device) -> Batch:
    payload = torch.load(path, map_location="cpu")
    required = ("obs", "rho_advantage", "rho_weight", "rho_mean_raw")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"dump file is missing required keys: {missing}")

    obs = payload["obs"].float()
    obs_mean = obs.mean(dim=0, keepdim=True)
    obs_std = obs.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-6)
    obs = (obs - obs_mean) / obs_std

    adv = payload["rho_advantage"].float()
    weight = payload["rho_weight"].float().clamp(min=0.0)
    prior_authority = payload.get("rho_prior_authority")
    if prior_authority is None:
        prior_authority = torch.zeros(obs.shape[0], 1)
    prior_authority = prior_authority.float()
    if prior_authority.ndim == 1:
        prior_authority = prior_authority.view(-1, 1)

    prior_target = payload.get("rho_prior_target")
    if prior_target is None:
        prior_target = torch.zeros_like(adv)
    prior_target = prior_target.float()

    return Batch(
        obs=obs.to(device),
        rho_advantage=adv.to(device),
        rho_weight=weight.to(device),
        prior_authority=prior_authority.to(device),
        prior_target=prior_target.to(device),
        current_rho_raw=payload["rho_mean_raw"].float().to(device),
        config=dict(payload.get("config", {})),
    )


def _prepare_advantage(raw_adv: torch.Tensor, active: torch.Tensor, config: dict) -> torch.Tensor:
    adv = raw_adv
    clip = float(config.get("adv_clip", 0.0) or 0.0)
    if clip > 0.0:
        adv = adv.clamp(-clip, clip)
    if bool(config.get("normalize_advantage", False)):
        active_adv = adv[active]
        if int(active_adv.numel()) > 1:
            adv = (adv - active_adv.mean()) / active_adv.std(unbiased=False).clamp(min=1e-6)
    return adv


def _rho_loss(logits: torch.Tensor, batch: Batch, indices: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    if indices is None:
        obs_adv = batch.rho_advantage
        weight = batch.rho_weight
        prior_authority = batch.prior_authority
        prior_target = batch.prior_target
    else:
        obs_adv = batch.rho_advantage[indices]
        weight = batch.rho_weight[indices]
        prior_authority = batch.prior_authority[indices]
        prior_target = batch.prior_target[indices]

    weight = weight[:, : logits.shape[-1]]
    obs_adv = obs_adv[:, : logits.shape[-1]]
    prior_target = prior_target[:, : logits.shape[-1]]
    prior_authority = prior_authority[:, :1].clamp(0.0, 1.0)
    active = weight > 1e-6
    adv = _prepare_advantage(obs_adv, active, batch.config)
    rho = torch.sigmoid(logits)

    repairable_authority = (1.0 - prior_authority).clamp(0.0, 1.0)
    repairable_weight = repairable_authority * active.to(logits.dtype)
    repair_target = (adv > 0.0).to(logits.dtype)
    repair_terms = F.binary_cross_entropy_with_logits(logits, repair_target, reduction="none")
    repair_loss = (repair_terms * adv.abs() * repairable_weight).sum()
    repair_loss = repair_loss / repairable_weight.sum().clamp(min=1e-6)

    prior_dim_weight = prior_authority * active.to(logits.dtype)
    boundary_loss = ((rho - prior_target).pow(2) * prior_dim_weight).sum()
    boundary_loss = boundary_loss / prior_dim_weight.sum().clamp(min=1e-6)

    repair_scale = float(batch.config.get("repair_loss_scale", 1.0) or 1.0)
    prior_weight = float(batch.config.get("prior_loss_weight", 0.0) or 0.0)
    loss = repair_scale * repair_loss + prior_weight * boundary_loss
    return loss, _metrics(rho, obs_adv, weight, prior_authority)


@torch.no_grad()
def _metrics(rho: torch.Tensor, adv: torch.Tensor, weight: torch.Tensor, prior_authority: torch.Tensor) -> dict:
    active = weight[:, : rho.shape[-1]] > 1e-6
    adv = adv[:, : rho.shape[-1]]
    repairable = ((1.0 - prior_authority[:, :1]) > 0.5) & active
    pos = active & (adv > 1e-6)
    neg = active & (adv < -1e-6)
    out = {
        "rho": float(rho[active].mean().item()) if bool(active.any().item()) else math.nan,
        "pos_frac": float(pos.float().mean().item()),
        "neg_frac": float(neg.float().mean().item()),
        "repairable_frac": float(repairable.float().mean().item()),
    }
    if bool(pos.any().item()):
        out["rho_pos"] = float(rho[pos].mean().item())
        out["adv_pos_abs"] = float(adv[pos].abs().mean().item())
    else:
        out["rho_pos"] = math.nan
        out["adv_pos_abs"] = math.nan
    if bool(neg.any().item()):
        out["rho_neg"] = float(rho[neg].mean().item())
        out["adv_neg_abs"] = float(adv[neg].abs().mean().item())
    else:
        out["rho_neg"] = math.nan
        out["adv_neg_abs"] = math.nan
    out["rho_sep"] = out["rho_pos"] - out["rho_neg"] if not math.isnan(out["rho_pos"]) and not math.isnan(out["rho_neg"]) else math.nan
    return out


def _print_metrics(label: str, metrics: dict) -> None:
    print(
        f"{label:<12} "
        f"rho={metrics['rho']:.3f} "
        f"rho+={metrics['rho_pos']:.3f} "
        f"rho-={metrics['rho_neg']:.3f} "
        f"sep={metrics['rho_sep']:+.3f} "
        f"pos/neg={metrics['pos_frac']:.3f}/{metrics['neg_frac']:.3f} "
        f"|adv|+/-={metrics['adv_pos_abs']:.3f}/{metrics['adv_neg_abs']:.3f}"
    )


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    batch = _load_batch(args.dump, device)
    n, obs_dim = batch.obs.shape
    rho_dim = batch.rho_advantage.shape[-1]
    torch.manual_seed(args.seed)
    perm = torch.randperm(n, device=device)
    holdout = int(round(float(args.holdout_frac) * n))
    test_idx = perm[:holdout] if holdout > 0 else perm[:0]
    train_idx = perm[holdout:] if holdout > 0 else perm

    model = RhoReplayHead(obs_dim, rho_dim, args.hidden_dim, args.linear).to(device)
    if args.init_from_current:
        with torch.no_grad():
            mean_logit = batch.current_rho_raw.mean(dim=0)
            last = model.net if isinstance(model.net, nn.Linear) else model.net[-1]
            last.weight.zero_()
            last.bias.copy_(mean_logit[:rho_dim])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=== FrontRES Live Batch Rho Replay TEST ONLY ===")
    print(f"dump={args.dump}")
    print(f"device={device}, samples={n}, train={train_idx.numel()}, test={test_idx.numel()}, obs_dim={obs_dim}, rho_dim={rho_dim}")
    print(f"config={batch.config}")
    _print_metrics("current", _metrics(torch.sigmoid(batch.current_rho_raw[:, :rho_dim]), batch.rho_advantage, batch.rho_weight, batch.prior_authority))

    for step in range(1, args.steps + 1):
        if args.batch_size > 0 and args.batch_size < train_idx.numel():
            pick = torch.randint(0, train_idx.numel(), (args.batch_size,), device=device)
            idx = train_idx[pick]
        else:
            idx = train_idx
        logits = model(batch.obs[idx])
        loss, _ = _rho_loss(logits, batch, idx)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.print_every == 0 or step == args.steps:
            with torch.no_grad():
                train_logits = model(batch.obs[train_idx])
                train_loss, train_metrics = _rho_loss(train_logits, batch, train_idx)
                print(f"step={step:05d} loss={float(train_loss.item()):.5f}")
                _print_metrics("train", train_metrics)
                if test_idx.numel() > 0:
                    test_logits = model(batch.obs[test_idx])
                    test_loss, test_metrics = _rho_loss(test_logits, batch, test_idx)
                    print(f"{'test_loss':<12} {float(test_loss.item()):.5f}")
                    _print_metrics("test", test_metrics)

    with torch.no_grad():
        final_idx = test_idx if test_idx.numel() > 0 else train_idx
        final_logits = model(batch.obs[final_idx])
        _, final_metrics = _rho_loss(final_logits, batch, final_idx)
    passed = final_metrics["rho_sep"] >= args.pass_sep
    print(f"result: {'PASS' if passed else 'FAIL'} (required sep >= {args.pass_sep:.3f})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True, help="Path produced by FRONTRES_LIVE_BATCH_DUMP.")
    parser.add_argument("--device", default="", help="cpu, cuda, cuda:0, ...")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--linear", action="store_true", help="Use a linear rho head instead of a 2-layer MLP.")
    parser.add_argument("--no-init-from-current", dest="init_from_current", action="store_false")
    parser.set_defaults(init_from_current=True)
    parser.add_argument("--pass-sep", type=float, default=0.15)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
