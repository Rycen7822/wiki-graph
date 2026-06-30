from __future__ import annotations

import json

from llm_wiki_native.cli import main


def test_cli_finalize_prepared_updates_active_pointer(tmp_path, capsys) -> None:
    prepared_path = tmp_path / "prepared_workspace.json"
    active_path = tmp_path / "active_workspace.json"
    history_path = tmp_path / "active_workspace.history.jsonl"
    prepared_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workspace_id": "native-test",
                "status": "prepared",
                "sqlite_path": "/tmp/native.sqlite",
                "zvec_path": "/tmp/zvec_records",
            }
        ),
        encoding="utf-8",
    )

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

    printed = json.loads(capsys.readouterr().out)
    assert printed["workspace_id"] == "native-test"
    assert printed["status"] == "active"
    assert json.loads(active_path.read_text(encoding="utf-8")) == printed
    assert history_path.exists()


def test_cli_rollback_active_restores_previous_pointer(tmp_path, capsys) -> None:
    active_path = tmp_path / "active_workspace.json"
    history_path = tmp_path / "active_workspace.history.jsonl"
    previous = {"schema_version": 1, "workspace_id": "old", "status": "active"}
    current = {"schema_version": 1, "workspace_id": "new", "status": "active"}
    active_path.write_text(json.dumps(current), encoding="utf-8")
    history_path.write_text(
        json.dumps({"previous": previous, "current": current, "reason": "test"}) + "\n",
        encoding="utf-8",
    )

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

    printed = json.loads(capsys.readouterr().out)
    assert printed == previous
    assert json.loads(active_path.read_text(encoding="utf-8")) == previous
