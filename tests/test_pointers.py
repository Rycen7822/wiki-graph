from __future__ import annotations

import json

import pytest

from llm_wiki_native.pointers import finalize_prepared_workspace, rollback_active_workspace


def _write_json(path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_finalize_prepared_workspace_updates_active_pointer_and_history(tmp_path) -> None:
    prepared_path = tmp_path / "prepared_workspace.json"
    active_path = tmp_path / "active_workspace.json"
    history_path = tmp_path / "active_workspace.history.jsonl"
    prepared = {
        "schema_version": 1,
        "workspace_id": "new-workspace",
        "status": "prepared",
        "sqlite_path": "/tmp/new.sqlite",
        "zvec_path": "/tmp/new.zvec",
    }
    previous = {
        "schema_version": 1,
        "workspace_id": "old-workspace",
        "status": "active",
        "sqlite_path": "/tmp/old.sqlite",
        "zvec_path": "/tmp/old.zvec",
    }
    _write_json(prepared_path, prepared)
    _write_json(active_path, previous)

    active = finalize_prepared_workspace(
        prepared_path,
        active_path,
        history_path,
        reason="shadow gate passed",
        finalized_at="2026-06-28T12:00:00Z",
    )

    assert active["workspace_id"] == "new-workspace"
    assert active["status"] == "active"
    assert json.loads(active_path.read_text(encoding="utf-8")) == active
    assert json.loads(prepared_path.read_text(encoding="utf-8")) == prepared
    history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    assert history == [
        {
            "previous": previous,
            "current": active,
            "reason": "shadow gate passed",
            "finalized_at": "2026-06-28T12:00:00Z",
        }
    ]


def test_rollback_active_workspace_restores_previous_pointer(tmp_path) -> None:
    active_path = tmp_path / "active_workspace.json"
    history_path = tmp_path / "active_workspace.history.jsonl"
    previous = {"schema_version": 1, "workspace_id": "old-workspace", "status": "active"}
    current = {"schema_version": 1, "workspace_id": "new-workspace", "status": "active"}
    _write_json(active_path, current)
    history_path.write_text(
        json.dumps(
            {
                "previous": previous,
                "current": current,
                "reason": "shadow gate passed",
                "finalized_at": "2026-06-28T12:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    restored = rollback_active_workspace(active_path, history_path)

    assert restored == previous
    assert json.loads(active_path.read_text(encoding="utf-8")) == previous


def test_rollback_active_workspace_requires_history(tmp_path) -> None:
    with pytest.raises(ValueError, match="history"):
        rollback_active_workspace(
            tmp_path / "active_workspace.json",
            tmp_path / "active_workspace.history.jsonl",
        )
