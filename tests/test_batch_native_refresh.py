from __future__ import annotations

import json
import sqlite3
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops import batch_native_refresh  # noqa: E402


def test_native_refresh_ledger_scope_is_distinct_from_wikigraph_batch_ledger(tmp_path) -> None:
    state_dir = tmp_path / "state"
    assert batch_native_refresh.PENDING_NATIVE_REFRESH_LEDGER == "pending_native_refresh.json"
    assert batch_native_refresh.pending_ledger_path(state_dir) == state_dir / batch_native_refresh.PENDING_NATIVE_REFRESH_LEDGER
    assert batch_native_refresh.PENDING_NATIVE_REFRESH_LEDGER != "pending_wikigraph_refresh.json"


def test_status_and_mark_pending_use_native_ledger_under_state(tmp_path, capsys) -> None:
    workdir = tmp_path / "wikigraph"
    root = tmp_path / "wiki"

    assert batch_native_refresh.main(["status", "--workdir", str(workdir), "--root", str(root)]) == 0
    empty = json.loads(capsys.readouterr().out)
    assert empty["pending_count"] == 0
    assert empty["should_refresh"] is False

    assert (
        batch_native_refresh.main(
            [
                "mark-pending",
                "--workdir",
                str(workdir),
                "--root",
                str(root),
                "--reason",
                "manual-smoke",
            ]
        )
        == 0
    )
    marked = json.loads(capsys.readouterr().out)

    assert marked["pending_count"] == 1
    assert Path(marked["ledger_path"]) == workdir / "state" / "pending_native_refresh.json"
    assert not (root / "pending_native_refresh.json").exists()


def test_status_default_paths_are_env_backed_without_repo_local_sandbox(capsys) -> None:
    assert batch_native_refresh.main(["status"]) == 0
    payload = json.loads(capsys.readouterr().out)
    repo_root = Path(__file__).resolve().parents[1]

    assert Path(payload["root"]) == repo_root
    assert Path(payload["state_dir"]) == repo_root / "tmp" / "native_refresh" / "state"


def test_mark_pending_default_paths_are_env_backed_without_repo_local_sandbox(capsys, monkeypatch) -> None:
    calls = []

    def fake_mark_pending(state_dir, root, *, reason):
        calls.append(("mark", state_dir, root, reason))
        return {"reason": reason, "root": str(root)}

    def fake_status(root, state_dir):
        calls.append(("status", state_dir, root))
        return {"pending_count": 1, "should_refresh": True}

    monkeypatch.setattr(batch_native_refresh, "mark_pending", fake_mark_pending)
    monkeypatch.setattr(batch_native_refresh, "status", fake_status)

    assert batch_native_refresh.main(["mark-pending", "--reason", "default-path-smoke"]) == 0
    payload = json.loads(capsys.readouterr().out)
    repo_root = Path(__file__).resolve().parents[1]
    expected_state = repo_root / "tmp" / "native_refresh" / "state"
    expected_root = repo_root

    assert payload["pending_count"] == 1
    assert calls == [
        ("mark", expected_state, expected_root, "default-path-smoke"),
        ("status", expected_state, expected_root),
    ]


def test_mark_pending_accepts_explicit_root_without_hardcoded_local_special_case(tmp_path, monkeypatch) -> None:
    calls = []
    workdir = tmp_path / "workdir"
    root = tmp_path / "operator-wiki"

    def fake_mark_pending(state_dir, root, *, reason):
        calls.append(("mark", state_dir, root, reason))
        return {"reason": reason, "root": str(root)}

    def fake_status(root, state_dir):
        calls.append(("status", state_dir, root))
        return {"pending_count": 1, "should_refresh": True}

    monkeypatch.setattr(batch_native_refresh, "mark_pending", fake_mark_pending)
    monkeypatch.setattr(batch_native_refresh, "status", fake_status)

    assert (
        batch_native_refresh.main(
            [
                "mark-pending",
                "--root",
                str(root),
                "--workdir",
                str(workdir),
                "--reason",
                "operator-root-smoke",
            ]
        )
        == 0
    )

    assert calls == [
        ("mark", workdir / "state", root.resolve(), "operator-root-smoke"),
        ("status", workdir / "state", root.resolve()),
    ]





def test_refresh_prepare_only_updates_prepared_pointer_without_active_or_clear(tmp_path, capsys, monkeypatch) -> None:
    workdir = tmp_path / "wikigraph"
    root = tmp_path / "wiki"
    state_dir = workdir / "state"
    native_dir = state_dir / "native_zvec"
    active_path = native_dir / "active_workspace.json"
    active_path.parent.mkdir(parents=True)
    previous_active = {"schema_version": 1, "workspace_id": "old", "status": "active"}
    active_path.write_text(json.dumps(previous_active), encoding="utf-8")
    batch_native_refresh.mark_pending(state_dir, root, reason="manual-smoke")
    calls = {}

    def fake_build_prepared_workspace(*, root, state_dir, workspace_root, workspace_id, embedding_profile, fill_missing_vectors):
        calls["root"] = root
        calls["state_dir"] = state_dir
        calls["workspace_root"] = workspace_root
        calls["workspace_id"] = workspace_id
        calls["embedding_profile"] = embedding_profile
        calls["fill_missing_vectors"] = fill_missing_vectors
        prepared_path = workspace_root.parent / "prepared_workspace.json"
        prepared_path.write_text(
            json.dumps({"schema_version": 1, "workspace_id": workspace_id, "status": "prepared"}),
            encoding="utf-8",
        )
        return {"ok": True, "prepared_workspace": str(prepared_path), "workspace_id": workspace_id}

    monkeypatch.setattr(batch_native_refresh, "build_prepared_workspace", fake_build_prepared_workspace)

    assert (
        batch_native_refresh.main(
            [
                "refresh",
                "--workdir",
                str(workdir),
                "--root",
                str(root),
                "--prepare-only",
                "--workspace-id",
                "candidate",
                "--embedding-profile",
                "conservative",
                "--fill-missing-vectors",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)

    assert result["prepared_only"] is True
    assert result["active_workspace_unchanged"] is True
    assert json.loads(active_path.read_text(encoding="utf-8")) == previous_active
    assert (native_dir / "prepared_workspace.json").exists()
    assert (state_dir / "pending_native_refresh.json").exists()
    assert calls == {
        "root": root.resolve(),
        "state_dir": state_dir.resolve(),
        "workspace_root": native_dir / "workspaces",
        "workspace_id": "candidate",
        "embedding_profile": "conservative",
        "fill_missing_vectors": True,
    }


def test_refresh_prepare_only_reports_required_unchanged_path_audit(tmp_path, capsys, monkeypatch) -> None:
    workdir = tmp_path / "wikigraph"
    root = tmp_path / "wiki"
    state_dir = workdir / "state"
    watched_dir = tmp_path / "watched"
    watched_dir.mkdir()
    watched_file = watched_dir / "storage.json"
    watched_file.write_text('{"stable":true}', encoding="utf-8")
    batch_native_refresh.mark_pending(state_dir, root, reason="manual-smoke")

    def fake_build_prepared_workspace(*, root, state_dir, workspace_root, workspace_id, embedding_profile, fill_missing_vectors):
        prepared_path = workspace_root.parent / "prepared_workspace.json"
        prepared_path.parent.mkdir(parents=True)
        prepared_path.write_text(
            json.dumps({"schema_version": 1, "workspace_id": workspace_id, "status": "prepared"}),
            encoding="utf-8",
        )
        return {"ok": True, "prepared_workspace": str(prepared_path), "workspace_id": workspace_id}

    monkeypatch.setattr(batch_native_refresh, "build_prepared_workspace", fake_build_prepared_workspace)

    assert (
        batch_native_refresh.main(
            [
                "refresh",
                "--workdir",
                str(workdir),
                "--root",
                str(root),
                "--prepare-only",
                "--workspace-id",
                "candidate",
                "--require-unchanged-path",
                str(watched_dir),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)

    assert result["unchanged_path_audit"]["ok"] is True
    assert result["unchanged_path_audit"]["paths"] == [
        {
            "path": str(watched_dir.resolve()),
            "ok": True,
            "changed": False,
        }
    ]


def test_refresh_without_prepare_only_requires_explicit_cutover(tmp_path) -> None:
    workdir = tmp_path / "wikigraph"

    try:
        batch_native_refresh.main(["refresh", "--workdir", str(workdir)])
    except ValueError as exc:
        assert "--cutover" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("refresh without --prepare-only must fail closed before explicit cutover")















def test_refresh_prepare_only_accepts_explicit_root_without_hardcoded_local_special_case(tmp_path, monkeypatch) -> None:
    state_dir = tmp_path / "native" / "state"
    root = tmp_path / "operator-wiki"
    workspace_root = state_dir / "native_zvec" / "workspaces"
    calls = []

    def fake_status(*args, **kwargs):
        calls.append(("status", args, kwargs))
        return {"should_refresh": True}

    def build_workspace(**kwargs):
        calls.append(("build", kwargs["workspace_id"]))
        return {"ok": True, "workspace_id": kwargs["workspace_id"]}

    monkeypatch.setattr(batch_native_refresh, "status", fake_status)
    monkeypatch.setattr(batch_native_refresh, "build_prepared_workspace", build_workspace)

    result = batch_native_refresh.refresh_prepare_only(
        root=root,
        state_dir=state_dir,
        workspace_root=workspace_root,
        workspace_id="candidate",
        embedding_profile="conservative",
    )

    assert result["prepared_only"] is True
    assert result["skipped"] is False
    assert calls == [("status", (root, state_dir), {}), ("build", "candidate")]
