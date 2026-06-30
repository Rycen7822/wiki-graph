"""Pointer-only finalize and rollback helpers for native workspaces."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def finalize_prepared_workspace(
    prepared_workspace_path: Path,
    active_workspace_path: Path,
    history_path: Path,
    *,
    reason: str,
    finalized_at: str | None = None,
) -> dict[str, Any]:
    prepared = _read_json(Path(prepared_workspace_path))
    if prepared.get("status") != "prepared":
        raise ValueError("prepared workspace pointer status must be prepared")
    active = {**prepared, "status": "active"}
    timestamp = finalized_at or _utc_now()
    previous = _read_json(Path(active_workspace_path)) if Path(active_workspace_path).exists() else None
    history = _read_jsonl(Path(history_path))
    history.append(
        {
            "previous": previous,
            "current": active,
            "reason": reason,
            "finalized_at": timestamp,
        }
    )
    _write_jsonl_atomic(Path(history_path), history)
    _write_json_atomic(Path(active_workspace_path), active)
    return active


def rollback_active_workspace(
    active_workspace_path: Path,
    history_path: Path,
) -> dict[str, Any]:
    history = _read_jsonl(Path(history_path))
    if not history:
        raise ValueError("active workspace history is empty")
    last = history[-1]
    if "previous" not in last:
        raise ValueError("active workspace history has no previous pointer")
    previous = last["previous"]
    if previous is None:
        active_path = Path(active_workspace_path)
        if active_path.exists():
            active_path.unlink()
        return {"schema_version": 1, "status": "absent"}
    if not isinstance(previous, dict):
        raise ValueError("active workspace history has no previous pointer")
    _write_json_atomic(Path(active_workspace_path), previous)
    return previous


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path} must contain JSON objects")
        rows.append(row)
    return rows


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
