from __future__ import annotations

from contextlib import contextmanager
import json
import mmap
import multiprocessing
import os
from pathlib import Path

import pytest

from llm_wiki_native.pointers import finalize_prepared_workspace, rollback_active_workspace
from llm_wiki_native.workspace_lock import workspace_mutation_lock
from ops import batch_native_refresh, batch_wiki_integration, native_workspace_retention as retention


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _workspace(root: Path, name: str) -> dict:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "native.sqlite").write_bytes(b"unit-test fixture, not a real database")
    (directory / "zvec_records").mkdir()
    (directory / "zvec_records" / "index").write_bytes(b"index fixture")
    _write(directory / "build_report.json", {"ok": True, "workspace_id": name})
    return {"schema_version": 1, "workspace_id": name, "status": "active", "sqlite_path": str(directory / "native.sqlite"), "zvec_path": str(directory / "zvec_records")}


@pytest.fixture
def workspace_chain(tmp_path):
    root = tmp_path / "state" / "native_zvec" / "workspaces"
    pointers = [_workspace(root, name) for name in ("old", "previous", "current")]
    rows = [{"previous": pointers[i - 1] if i else None, "current": pointer} for i, pointer in enumerate(pointers)]
    (root.parent / "active_workspace.history.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    _write(root.parent / "active_workspace.json", pointers[-1])
    _write(root.parent / "prepared_workspace.json", {**pointers[-1], "status": "prepared"})
    return root, pointers


def _cleanup(root):
    return retention.cleanup_obsolete_workspaces(workspace_root=root, expected_active_id="current")


def test_retention_keeps_current_previous_prepared_and_unknown(workspace_chain):
    root, _ = workspace_chain
    staging = _workspace(root, "staging")
    _workspace(root, "unknown")
    _write(root.parent / "prepared_workspace.json", {**staging, "status": "prepared"})
    pointers_before = retention._pointer_snapshot(root)
    result = _cleanup(root)
    assert result["ok"] is True
    assert result["deleted"] == ["old"]
    assert result["reclaimed_bytes"] > 0
    assert result["protected"] == ["current", "previous", "staging"]
    assert result["retained_unpublished"] == ["unknown"]
    assert sorted(p.name for p in root.iterdir()) == ["current", "previous", "staging", "unknown"]
    assert retention._pointer_snapshot(root) == pointers_before
    assert json.loads(Path(result["report_path"]).read_text()) == result
    assert _cleanup(root)["deleted"] == []


def test_open_fd_and_mmap_protect_historical_workspace(workspace_chain):
    root, _ = workspace_chain
    with (root / "old" / "native.sqlite").open("rb") as handle:
        assert "old" in retention._in_use_workspace_ids(root)
        mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    try:
        assert "old" in retention._in_use_workspace_ids(root)
        result = _cleanup(root)
        assert result["ok"] and not result["deleted"]
    finally:
        mapping.close()
    assert _cleanup(root)["deleted"] == ["old"]


@pytest.mark.parametrize("damage", ["active-id", "history", "prepared", "pointer-path", "missing-previous", "bad-build", "symlink", "nested-symlink"])
def test_invalid_state_fails_closed_without_deleting(workspace_chain, tmp_path, damage):
    root, pointers = workspace_chain
    if damage == "active-id":
        _write(root.parent / "active_workspace.json", pointers[1])
    elif damage == "history":
        (root.parent / "active_workspace.history.jsonl").write_text("not json")
    elif damage == "prepared":
        (root.parent / "prepared_workspace.json").write_text("null")
    elif damage == "pointer-path":
        _write(root.parent / "active_workspace.json", {**pointers[-1], "sqlite_path": str(tmp_path / "outside.sqlite")})
    elif damage == "missing-previous":
        (root / "previous" / "native.sqlite").unlink()
    elif damage == "bad-build":
        _write(root / "old" / "build_report.json", {"ok": False, "workspace_id": "old"})
    else:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "sentinel").write_text("untouched")
        if damage == "symlink":
            (root / "old").rename(root / "old-original")
            (root / "old").symlink_to(outside, target_is_directory=True)
        else:
            (root / "old" / "link").symlink_to(outside, target_is_directory=True)
    result = _cleanup(root)
    assert result["status"] == "warning" and result["warnings"]
    assert result["deleted"] == []
    assert (root / "old").exists()
    if (tmp_path / "outside").exists():
        assert (tmp_path / "outside" / "sentinel").read_text() == "untouched"


def test_process_scan_failure_is_warning_not_deletion(workspace_chain, monkeypatch):
    root, _ = workspace_chain
    def denied(_root):
        raise PermissionError("process inaccessible")
    monkeypatch.setattr(retention, "_in_use_workspace_ids", denied)
    result = _cleanup(root)
    assert result["status"] == "warning" and not result["deleted"]


def test_pointer_drift_aborts_before_delete(workspace_chain, monkeypatch):
    root, pointers = workspace_chain
    def drift(_root):
        _write(root.parent / "active_workspace.json", pointers[1])
        return set()
    monkeypatch.setattr(retention, "_in_use_workspace_ids", drift)
    result = _cleanup(root)
    assert result["status"] == "warning" and not result["deleted"]


def test_removal_error_returns_warning_and_keeps_refresh_success(workspace_chain, monkeypatch):
    root, pointers = workspace_chain
    def denied(*args, **kwargs):
        raise PermissionError("disk read-only")
    setattr(denied, "avoids_symlink_attacks", True)
    monkeypatch.setattr(retention.shutil, "rmtree", denied)
    monkeypatch.setattr(batch_native_refresh, "_refresh_cutover", lambda **kwargs: {"cutover_executed": True, "active": pointers[-1]})
    sentinel = root.parent.parent / "sentinel"
    sentinel.write_text("unchanged")
    result = batch_native_refresh.refresh_cutover(
        root=root.parent, state_dir=root.parent.parent, workspace_root=root,
        workspace_id="current", embedding_profile="conservative",
        query_smoke=lambda **kwargs: {"ok": True}, required_unchanged_paths=[sentinel],
    )
    assert result["cutover_executed"] is True
    assert result["workspace_cleanup"]["status"] == "warning"
    assert (root / "old").is_dir()


def _attempt_lock(pointer_dir: str) -> str:
    try:
        with workspace_mutation_lock(Path(pointer_dir), timeout=0.1):
            return "acquired"
    except TimeoutError:
        return "busy"


@pytest.mark.subprocess
def test_lock_is_reentrant_and_excludes_another_process(tmp_path):
    with workspace_mutation_lock(tmp_path):
        with workspace_mutation_lock(tmp_path):
            with multiprocessing.get_context("spawn").Pool(1) as pool:
                assert pool.apply(_attempt_lock, (str(tmp_path),)) == "busy"
    assert _attempt_lock(str(tmp_path)) == "acquired"


def test_pointer_finalize_and_rollback_share_lifecycle_lock(workspace_chain):
    root, pointers = workspace_chain
    with workspace_mutation_lock(root.parent):
        new = _workspace(root, "new")
        prepared_path = root.parent / "prepared_workspace.json"
        active_path = root.parent / "active_workspace.json"
        history_path = root.parent / "active_workspace.history.jsonl"
        _write(prepared_path, {**new, "status": "prepared"})
        assert finalize_prepared_workspace(prepared_path, active_path, history_path, reason="test")["workspace_id"] == "new"
        assert rollback_active_workspace(active_path, history_path)["workspace_id"] == "current"
    # A rollback leaves history's last current != active: conservative cleanup refuses it.
    assert _cleanup(root)["status"] == "warning"


def test_integration_cleanup_waits_for_managed_teardown(workspace_chain, monkeypatch):
    root, pointers = workspace_chain
    events = []
    monkeypatch.setattr(batch_native_refresh, "status", lambda *args: {"should_refresh": True})
    monkeypatch.setattr(batch_wiki_integration, "_native_refresh_config", lambda *args: {})
    monkeypatch.setattr(batch_wiki_integration, "managed_local_embedding_service_config", lambda *args, **kwargs: None)
    @contextmanager
    def service(_config):
        events.append("service-start")
        yield {"managed": True}
        events.append("service-stop")
    monkeypatch.setattr(batch_wiki_integration, "managed_local_embedding_service", service)
    def run(*args, **kwargs):
        events.extend(["cutover", "query-smoke", "coverage"])
        return 0, {"runs": [{"active": pointers[-1]}], "workspace_root": str(root)}
    monkeypatch.setattr(batch_wiki_integration, "_run_native_refresh_after_wiki_integration", run)
    def cleanup(**kwargs):
        assert events[-1] == "service-stop"
        events.append("cleanup")
        return retention.cleanup_obsolete_workspaces(**kwargs)
    monkeypatch.setattr(batch_wiki_integration, "cleanup_obsolete_workspaces", cleanup)
    code, result = batch_wiki_integration.run_native_refresh_after_wiki_integration(root.parent, root.parent.parent, reason="test")
    assert code == 0 and result["workspace_cleanup"]["deleted"] == ["old"]
    assert events == ["service-start", "cutover", "query-smoke", "coverage", "service-stop", "cleanup"]


@pytest.mark.parametrize("failure_code", [16, 17, 18, 19])
def test_integration_failure_never_cleans(workspace_chain, monkeypatch, failure_code):
    root, pointers = workspace_chain
    monkeypatch.setattr(batch_native_refresh, "status", lambda *args: {"should_refresh": True})
    monkeypatch.setattr(batch_wiki_integration, "_native_refresh_config", lambda *args: {})
    monkeypatch.setattr(batch_wiki_integration, "managed_local_embedding_service_config", lambda *args, **kwargs: None)
    @contextmanager
    def service(_config):
        yield {"managed": False}
    monkeypatch.setattr(batch_wiki_integration, "managed_local_embedding_service", service)
    monkeypatch.setattr(batch_wiki_integration, "_run_native_refresh_after_wiki_integration", lambda *args, **kwargs: (failure_code, {"runs": [{"active": pointers[-1]}]}))
    code, result = batch_wiki_integration.run_native_refresh_after_wiki_integration(root.parent, root.parent.parent, reason="test")
    assert code == failure_code and "workspace_cleanup" not in result
    assert (root / "old").is_dir()


def test_managed_service_teardown_failure_never_cleans(workspace_chain, monkeypatch):
    root, pointers = workspace_chain
    monkeypatch.setattr(batch_native_refresh, "status", lambda *args: {"should_refresh": True})
    monkeypatch.setattr(batch_wiki_integration, "_native_refresh_config", lambda *args: {})
    monkeypatch.setattr(batch_wiki_integration, "managed_local_embedding_service_config", lambda *args, **kwargs: None)

    @contextmanager
    def service(_config):
        yield {"managed": True}
        raise batch_wiki_integration.ManagedLocalEmbeddingServiceError("stop", "test stop failed", report={"stopped": False})

    monkeypatch.setattr(batch_wiki_integration, "managed_local_embedding_service", service)
    monkeypatch.setattr(batch_wiki_integration, "_run_native_refresh_after_wiki_integration", lambda *args, **kwargs: (0, {"runs": [{"active": pointers[-1]}], "workspace_root": str(root)}))
    code, result = batch_wiki_integration.run_native_refresh_after_wiki_integration(root.parent, root.parent.parent, reason="test")
    assert code == 21 and "workspace_cleanup" not in result
    assert (root / "old").is_dir()


def test_symlink_root_is_rejected(workspace_chain, tmp_path):
    root, _ = workspace_chain
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    result = _cleanup(alias)
    assert result["status"] == "warning" and not result["deleted"]
    assert (root / "old").is_dir()


def test_mount_inside_candidate_is_rejected(workspace_chain, monkeypatch):
    root, _ = workspace_chain
    original = Path.is_mount
    monkeypatch.setattr(Path, "is_mount", lambda path: path == root / "old" / "zvec_records" or original(path))
    result = _cleanup(root)
    assert result["status"] == "warning" and not result["deleted"]


def test_new_reader_between_plan_and_delete_is_protected(workspace_chain, monkeypatch):
    root, _ = workspace_chain
    calls = iter([set(), {"old"}])
    monkeypatch.setattr(retention, "_in_use_workspace_ids", lambda _root: next(calls))
    result = _cleanup(root)
    assert result["ok"] and result["deleted"] == []
    assert "old" in result["protected"]


def test_cleanup_report_write_failure_is_warning(workspace_chain, monkeypatch):
    root, _ = workspace_chain
    def fail_report(*args, **kwargs):
        raise OSError("report disk failure")
    monkeypatch.setattr(retention, "atomic_write_json", fail_report)
    result = _cleanup(root)
    assert result["status"] == "warning"
    assert result["deleted"] == ["old"]
    assert not (root / "old").exists()
    assert (root / "previous").is_dir() and (root / "current").is_dir()


def test_build_uses_workspace_lifecycle_lock(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ops import native_zvec_materialize
    calls = []
    @contextmanager
    def lock(directory):
        calls.append(("locked", directory))
        yield
        calls.append(("released", directory))
    monkeypatch.setattr(native_zvec_materialize, "workspace_mutation_lock", lock)
    monkeypatch.setattr(native_zvec_materialize, "_build", lambda *args, **kwargs: calls.append(("build", None)) or {"ok": True})
    assert native_zvec_materialize.build(SimpleNamespace(workspace_root=tmp_path / "workspaces"))["ok"]
    assert calls == [("locked", tmp_path), ("build", None), ("released", tmp_path)]

