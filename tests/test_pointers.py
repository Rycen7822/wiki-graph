from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_wiki_native.cli import main
from llm_wiki_native.pointers import finalize_prepared_workspace, rollback_active_workspace
from support import write_json, write_jsonl


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
    write_json(prepared_path, prepared)
    write_json(active_path, previous)

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


def test_rollback_active_workspace_restores_previous_or_absent_pointer(tmp_path) -> None:
    active_path = tmp_path / "active_workspace.json"
    history_path = tmp_path / "active_workspace.history.jsonl"
    previous = {"schema_version": 1, "workspace_id": "old-workspace", "status": "active"}
    current = {"schema_version": 1, "workspace_id": "new-workspace", "status": "active"}
    write_json(active_path, current)
    write_jsonl(
        history_path,
        [
            {
                "previous": previous,
                "current": current,
                "reason": "shadow gate passed",
                "finalized_at": "2026-06-28T12:00:00Z",
            }
        ],
    )

    restored = rollback_active_workspace(active_path, history_path)

    assert restored == previous
    assert json.loads(active_path.read_text(encoding="utf-8")) == previous

    first_active_path = tmp_path / "first_active_workspace.json"
    first_history_path = tmp_path / "first_active_workspace.history.jsonl"
    write_json(first_active_path, current)
    write_jsonl(first_history_path, [{"previous": None, "current": current, "reason": "first cutover"}])

    absent = rollback_active_workspace(first_active_path, first_history_path)

    assert absent == {"schema_version": 1, "status": "absent"}
    assert not first_active_path.exists()


def test_rollback_active_workspace_requires_history(tmp_path) -> None:
    with pytest.raises(ValueError, match="history"):
        rollback_active_workspace(
            tmp_path / "active_workspace.json",
            tmp_path / "active_workspace.history.jsonl",
        )


def test_cli_pointer_commands_delegate_paths_and_print_json(tmp_path, capsys, monkeypatch) -> None:
    calls: list[tuple] = []

    def fake_finalize_prepared_workspace(
        prepared_workspace_file: Path,
        active_workspace_file: Path,
        history_file: Path,
        *,
        reason: str,
    ) -> dict:
        calls.append(("finalize", prepared_workspace_file, active_workspace_file, history_file, reason))
        return {"workspace_id": "native-test", "status": "active"}

    def fake_rollback_active_workspace(active_workspace_file: Path, history_file: Path) -> dict:
        calls.append(("rollback", active_workspace_file, history_file))
        return {"workspace_id": "old", "status": "active"}

    monkeypatch.setattr("llm_wiki_native.cli.finalize_prepared_workspace", fake_finalize_prepared_workspace)
    monkeypatch.setattr("llm_wiki_native.cli.rollback_active_workspace", fake_rollback_active_workspace)

    prepared_path = tmp_path / "prepared_workspace.json"
    active_path = tmp_path / "active_workspace.json"
    history_path = tmp_path / "active_workspace.history.jsonl"

    assert (
        main(
            [
                "finalize-prepared",
                "--prepared-workspace-file",
                str(prepared_path),
                "--active-workspace-file",
                str(active_path),
                "--history-file",
                str(history_path),
                "--reason",
                "test finalize",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {"workspace_id": "native-test", "status": "active"}
    assert calls[-1] == ("finalize", prepared_path, active_path, history_path, "test finalize")

    assert (
        main(
            [
                "rollback-active",
                "--active-workspace-file",
                str(active_path),
                "--history-file",
                str(history_path),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {"workspace_id": "old", "status": "active"}
    assert calls[-1] == ("rollback", active_path, history_path)
