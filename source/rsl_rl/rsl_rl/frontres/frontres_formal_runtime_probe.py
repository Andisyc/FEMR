"""Default-off module-local probes for the formal Stage 3 runtime audit."""

from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Any

import torch


_ENABLED = False
_COUNTS: dict[str, int] = {}


def configure_formal_runtime_probe(enabled: bool) -> None:
    """Enable owner-local audit prints without changing owner return values."""

    global _ENABLED
    enabled = bool(enabled)
    if enabled and not _ENABLED:
        _COUNTS.clear()
    _ENABLED = enabled
    os.environ["FRONTRES_FORMAL_RUNTIME_AUDIT_ACTIVE"] = "1" if enabled else "0"


def formal_runtime_probe_enabled() -> bool:
    return _ENABLED or os.environ.get("FRONTRES_FORMAL_RUNTIME_AUDIT_ACTIVE", "0") == "1"


def _compact(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        finite = bool(torch.isfinite(tensor.float()).all().item()) if tensor.numel() else True
        text = f"shape={tuple(tensor.shape)},dtype={tensor.dtype},device={tensor.device},finite={int(finite)}"
        if tensor.numel() and tensor.dtype != torch.bool:
            numeric = tensor.float()
            text += f",min={numeric.min().item():.6g},max={numeric.max().item():.6g},mean={numeric.mean().item():.6g}"
        return text
    if isinstance(value, Mapping):
        return "{" + ",".join(f"{key}:{_compact(item)}" for key, item in value.items()) + "}"
    if isinstance(value, (tuple, list)):
        head = tuple(value[:8])
        return f"count={len(value)},head={head}"
    return str(value)


def emit_formal_runtime_probe(audit_id: str, *, limit: int = 2, **values: Any) -> None:
    """Print a bounded snapshot at the actual owner boundary when audit mode is enabled."""

    if not formal_runtime_probe_enabled():
        return
    count = _COUNTS.get(audit_id, 0)
    if count >= max(1, int(limit)):
        return
    _COUNTS[audit_id] = count + 1
    fields = " ".join(f"{key}={_compact(value)}" for key, value in values.items())
    print(f"[{audit_id}] occurrence={count + 1} {fields}", flush=True)


__all__ = [
    "configure_formal_runtime_probe",
    "emit_formal_runtime_probe",
    "formal_runtime_probe_enabled",
]
