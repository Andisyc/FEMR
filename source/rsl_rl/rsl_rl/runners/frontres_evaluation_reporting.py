"""Atomic filesystem boundary shared by FrontRES evaluation use cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def write_frontres_atomic_json(
    path: str | Path,
    payload: Mapping[str, Any] | Sequence[Any],
    *,
    compact: bool = False,
) -> None:
    """Validate one unused path and atomically commit JSON evaluation evidence."""

    # B1: 校验 final/tmp identity, 产出唯一可提交的 filesystem boundary.
    output = Path(path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if output.exists() or temporary.exists():
        raise RuntimeError(f"FrontRES evaluation refuses existing report identity: {output}")
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if compact
        else json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    # B2: 写入 temporary 后 replace, 异常时清除 partial artifact.
    try:
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_frontres_json_csv_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Commit DR-sweep rows as JSON plus a stable companion CSV artifact."""

    # B1: 同时预检 JSON/CSV final 与 tmp identity, 禁止半提交.
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    json_temporary = output.with_suffix(output.suffix + ".tmp")
    csv_path = output.with_suffix(".csv")
    csv_temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    if output.exists() or json_temporary.exists() or csv_path.exists() or csv_temporary.exists():
        raise RuntimeError(f"FrontRES evaluation refuses existing CSV identity: {csv_path}")
    # B2: 编码两个 temporary artifacts, 全部成功后才提交 final identities.
    encoded_json = json.dumps(list(rows), indent=2, sort_keys=True, allow_nan=False) + "\n"
    keys = tuple(rows[0]) if rows else ()
    lines = [",".join(keys)] if keys else []
    lines.extend(",".join(json.dumps(row[key], allow_nan=False) for key in keys) for row in rows)
    try:
        json_temporary.write_text(encoded_json, encoding="utf-8")
        csv_temporary.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        csv_temporary.replace(csv_path)
        json_temporary.replace(output)
    except Exception:
        json_temporary.unlink(missing_ok=True)
        csv_temporary.unlink(missing_ok=True)
        if csv_path.exists() and not output.exists():
            csv_path.unlink()
        raise
    return csv_path


__all__ = ["write_frontres_atomic_json", "write_frontres_json_csv_rows"]
