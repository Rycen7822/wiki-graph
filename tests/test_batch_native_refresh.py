from __future__ import annotations

import json
from pathlib import Path

import pytest

from ops import batch_native_refresh  # noqa: E402

_DEFAULT_PATH_ENV = (
    "LLM_WIKI_ROOT",
    "LLM_WIKI_WORKDIR",
    "WIKI_GRAPH_REPO",
    "LLM_WIKI_NATIVE_REFRESH_WORKDIR",
    "LLM_WIKI_STATE_DIR",
)


def _isolate_default_paths(monkeypatch: pytest.MonkeyPatch) -> Path:
    for name in _DEFAULT_PATH_ENV:
        monkeypatch.delenv(name, raising=False)
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(batch_native_refresh, "DEFAULT_BASE_WORKDIR", repo_root)
    monkeypatch.setattr(batch_native_refresh, "DEFAULT_WIKI_ROOT", repo_root)
    monkeypatch.setattr(batch_native_refresh, "DEFAULT_WORKDIR", repo_root / "tmp" / "native_refresh")
    return repo_root


def test_status_and_mark_pending_use_native_ledger_under_state(tmp_path, capsys) -> None:
    workdir = tmp_path / "wikigraph"
    root = tmp_path / "wiki"
    state_dir = workdir / "state"

    assert batch_native_refresh.pending_ledger_path(state_dir) == state_dir / batch_native_refresh.PENDING_NATIVE_REFRESH_LEDGER

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


def test_status_default_paths_are_env_backed_without_repo_local_sandbox(capsys, monkeypatch) -> None:
    repo_root = _isolate_default_paths(monkeypatch)
    assert batch_native_refresh.main(["status"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert Path(payload["root"]) == repo_root
    assert Path(payload["state_dir"]) == repo_root / "tmp" / "native_refresh" / "state"


def test_mark_pending_default_and_explicit_paths_are_env_backed_without_repo_local_sandbox(capsys, monkeypatch, tmp_path) -> None:
    calls = []
    repo_root = _isolate_default_paths(monkeypatch)
    default_state = repo_root / "tmp" / "native_refresh" / "state"
    explicit_workdir = tmp_path / "workdir"
    explicit_root = (tmp_path / "operator-wiki").resolve()

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

    assert payload["pending_count"] == 1

    assert (
        batch_native_refresh.main(
            [
                "mark-pending",
                "--root",
                str(explicit_root),
                "--workdir",
                str(explicit_workdir),
                "--reason",
                "operator-root-smoke",
            ]
        )
        == 0
    )

    assert calls == [
        ("mark", default_state, repo_root, "default-path-smoke"),
        ("status", default_state, repo_root),
        ("mark", explicit_workdir / "state", explicit_root, "operator-root-smoke"),
        ("status", explicit_workdir / "state", explicit_root),
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
    watched_dir = tmp_path / "watched"
    watched_dir.mkdir()
    (watched_dir / "storage.json").write_text('{"stable":true}', encoding="utf-8")
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
                "--require-unchanged-path",
                str(watched_dir),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)

    assert result["prepared_only"] is True
    assert result["active_workspace_unchanged"] is True
    assert result["unchanged_path_audit"]["ok"] is True
    assert result["unchanged_path_audit"]["paths"][0]["changed"] is False
    assert json.loads(active_path.read_text(encoding="utf-8")) == previous_active
    assert (native_dir / "prepared_workspace.json").exists()
    assert (state_dir / "pending_native_refresh.json").exists()
    assert calls["workspace_id"] == "candidate"
    assert calls["embedding_profile"] == "conservative"
    assert calls["fill_missing_vectors"] is True


def test_refresh_without_prepare_only_requires_explicit_cutover(tmp_path) -> None:
    workdir = tmp_path / "wikigraph"

    with pytest.raises(ValueError, match="--cutover"):
        batch_native_refresh.main(["refresh", "--workdir", str(workdir)])


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
