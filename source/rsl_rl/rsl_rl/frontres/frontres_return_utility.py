"""Fixed TRAIN-v019/v020 mapping from raw Recovery-Aware Gain to learning utility."""

from __future__ import annotations

import torch


FRONTRES_RETURN_UTILITY_ID = "symmetric-log-gain-g0-1-v1"
FRONTRES_RETURN_UTILITY_SCALE = 1.0


def frontres_symmetric_log_utility(raw_returns: torch.Tensor) -> torch.Tensor:
    """Map detached finite raw Gain values to same-shape symmetric-log utility."""

    if not isinstance(raw_returns, torch.Tensor) or not raw_returns.is_floating_point():
        raise ValueError("TRAIN-v020 utility requires a floating-point tensor")
    if raw_returns.requires_grad:
        raise ValueError("TRAIN-v020 utility requires detached raw Gain")
    if not bool(torch.isfinite(raw_returns).all().item()):
        raise FloatingPointError("TRAIN-v020 utility requires finite raw Gain")
    utility = torch.sign(raw_returns) * torch.log1p(torch.abs(raw_returns) / FRONTRES_RETURN_UTILITY_SCALE)
    if not bool(torch.isfinite(utility).all().item()):
        raise FloatingPointError("TRAIN-v020 utility produced non-finite values")
    return utility.detach()
