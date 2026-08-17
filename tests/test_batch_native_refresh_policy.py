from __future__ import annotations

import json
from pathlib import Path

import pytest

from ops import batch_native_refresh
from support import append_native_history as _append_native_history
def test_native_refresh_status_reports_five_incremental_full_rebuild_policy(tmp_path) -> None:
    state_dir = tmp_path / "wikigraph" / "state"
    root = tmp_path / "wiki"
    _append_native_history(
        state_dir,
        [f"native graph incremental refresh: cutover {idx}" for idx in range(5)],
    )

    status = batch_native_refresh.status(root, state_dir)

    assert status["completed_incremental_refresh_count"] == 5
    assert status["incremental_rebuild_threshold"] == 5
    assert status["next_refresh_kind"] == "full-rebuild"
    assert status["vector_cache_required"] is True
    assert status["vector_cache_path"] == str(state_dir / "vector_cache.sqlite")

    _append_native_history(
        state_dir,
        [
            *[f"native graph incremental refresh: cutover {idx}" for idx in range(5)],
            "native graph full-rebuild refresh: cutover",
        ],
    )

    reset_status = batch_native_refresh.status(root, state_dir)
    assert reset_status["completed_incremental_refresh_count"] == 0
    assert reset_status["next_refresh_kind"] == "incremental"
def test_refresh_cutover_marks_full_rebuild_pending_after_fifth_incremental(tmp_path) -> None:
    state_dir = tmp_path / "wikigraph" / "state"
    root = tmp_path / "wiki"
    workspace_root = state_dir / "native_zvec" / "workspaces"
    watched_dir = tmp_path / "watched"
    watched_dir.mkdir()
    _append_native_history(
        state_dir,
        [f"native graph incremental refresh: cutover {idx}" for idx in range(4)],
    )
    batch_native_refresh.mark_pending(state_dir, root, reason="wiki-integration:threshold")
    calls = []

    def build_workspace(**kwargs):
        calls.append(("build", kwargs["workspace_id"]))
        return {"ok": True, "workspace_id": kwargs["workspace_id"]}

    def finalize_workspace(*, state_dir, reason):
        calls.append(("finalize", reason))
        history = batch_native_refresh.active_workspace_history_path(state_dir)
        with history.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"reason": reason, "current": {"workspace_id": "candidate"}}) + "\n")
        return {"schema_version": 1, "workspace_id": "candidate", "status": "active"}

    def restart_service(*, state_dir):
        calls.append(("restart", str(state_dir)))
        return {"service": "llm-wiki-native", "status": "ok"}

    def query_smoke(*, state_dir, active):
        calls.append(("smoke", str(state_dir), active["workspace_id"]))
        return {"ok": True}

    result = batch_native_refresh.refresh_cutover(
        root=root,
        state_dir=state_dir,
        workspace_root=workspace_root,
        workspace_id="candidate",
        embedding_profile="conservative",
        build_workspace=build_workspace,
        finalize_workspace=finalize_workspace,
        restart_service=restart_service,
        query_smoke=query_smoke,
        required_unchanged_paths=[watched_dir],
    )

    entries = batch_native_refresh.pending_entries(state_dir)
    assert result["refresh_kind"] == "incremental"
    assert result["next_refresh_kind_after_success"] == "full-rebuild"
    assert result["policy_native_pending_marked_count"] == 1
    assert result["policy_native_pending"]["reason"] == batch_native_refresh.FULL_REBUILD_DUE_REASON
    assert result["status_after"]["pending_count"] == 1
    assert [entry["reason"] for entry in entries] == [batch_native_refresh.FULL_REBUILD_DUE_REASON]
    assert calls == [
        ("build", "candidate"),
        ("finalize", "native graph incremental refresh: cutover"),
        ("restart", str(state_dir)),
        ("smoke", str(state_dir), "candidate"),
    ]
def test_refresh_prepare_only_defaults_to_vector_cache_and_full_rebuild_when_policy_due(tmp_path, monkeypatch) -> None:
    state_dir = tmp_path / "wikigraph" / "state"
    root = tmp_path / "wiki"
    workspace_root = state_dir / "native_zvec" / "workspaces"
    _append_native_history(
        state_dir,
        [f"native graph incremental refresh: cutover {idx}" for idx in range(5)],
    )
    batch_native_refresh.mark_pending(state_dir, root, reason="wiki-integration:threshold")
    calls = []

    def fake_build_prepared_workspace(*, root, state_dir, workspace_root, workspace_id, embedding_profile, fill_missing_vectors):
        calls.append((workspace_id, embedding_profile, fill_missing_vectors))
        return {"ok": True, "workspace_id": workspace_id}

    monkeypatch.setattr(batch_native_refresh, "build_prepared_workspace", fake_build_prepared_workspace)

    result = batch_native_refresh.refresh_prepare_only(
        root=root,
        state_dir=state_dir,
        workspace_root=workspace_root,
        workspace_id="candidate",
        embedding_profile="conservative",
    )

    assert result["refresh_kind"] == "full-rebuild"
    assert result["fill_missing_vectors"] is True
    assert result["vector_cache_required"] is True
    assert calls == [("candidate", "conservative", True)]
