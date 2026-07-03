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
def _cutover_context(tmp_path: Path, *, pending: bool = False) -> SimpleNamespace:
    state_dir = tmp_path / "wikigraph" / "state"
    root = tmp_path / "wiki"
    workspace_root = state_dir / "native_zvec" / "workspaces"
    watched_dir = tmp_path / "watched"
    calls: list[tuple] = []
    if pending:
        watched_dir.mkdir(parents=True)
        batch_native_refresh.mark_pending(state_dir, root, reason="manual-smoke")
    return SimpleNamespace(
        state_dir=state_dir,
        root=root,
        workspace_root=workspace_root,
        watched_dir=watched_dir,
        calls=calls,
    )
def _fake_cutover_hooks(
    calls: list[tuple],
    *,
    smoke_ok: bool = True,
    smoke_raises: Exception | None = None,
    mutate_watched_path: Path | None = None,
):
    def build_workspace(**kwargs):
        calls.append(("build", kwargs["workspace_id"]))
        if mutate_watched_path is not None:
            mutate_watched_path.write_text('{"stable":false}', encoding="utf-8")
        return {"ok": True, "workspace_id": kwargs["workspace_id"]}

    def finalize_workspace(*, state_dir, reason):
        calls.append(("finalize", reason))
        return {"schema_version": 1, "workspace_id": "candidate", "status": "active"}

    def restart_service(*, state_dir):
        calls.append(("restart", str(state_dir)))
        return {"service": "llm-wiki-native", "status": "ok"}

    def query_smoke(*, state_dir, active):
        calls.append(("smoke", str(state_dir), active["workspace_id"]))
        if smoke_raises is not None:
            raise smoke_raises
        return {"ok": smoke_ok, "url": "http://127.0.0.1:9621/query/data"}

    return build_workspace, finalize_workspace, restart_service, query_smoke
def test_refresh_cutover_has_named_workflow_seams() -> None:
    expected_helpers = {
        "_validate_refresh_cutover_preconditions",
        "_skipped_refresh_cutover_result",
        "_execute_refresh_cutover",
        "_refresh_cutover_success_result",
    }

    assert {name for name in expected_helpers if callable(getattr(batch_native_refresh, name, None))} == expected_helpers
def test_refresh_cutover_cli_uses_explicit_restart_command_hook(tmp_path, capsys, monkeypatch) -> None:
    workdir = tmp_path / "wikigraph"
    state_dir = workdir / "state"
    root = tmp_path / "wiki"
    watched_dir = tmp_path / "watched"
    watched_dir.mkdir()
    batch_native_refresh.mark_pending(state_dir, root, reason="manual-smoke")
    calls = []

    def fake_build_prepared_workspace(**kwargs):
        calls.append(("build", kwargs["workspace_id"]))
        prepared_path = kwargs["workspace_root"].parent / "prepared_workspace.json"
        prepared_path.parent.mkdir(parents=True)
        prepared_path.write_text(
            json.dumps({"schema_version": 1, "workspace_id": kwargs["workspace_id"], "status": "prepared"}),
            encoding="utf-8",
        )
        return {"ok": True, "prepared_workspace": str(prepared_path), "workspace_id": kwargs["workspace_id"]}

    def fake_finalize_workspace(*, state_dir, reason):
        calls.append(("finalize", reason))
        return {"schema_version": 1, "workspace_id": "candidate", "status": "active"}

    def fake_restart_service_from_args(args):
        calls.append(("restart_args", args.restart_command, args.health_url, args.health_timeout))

        def restart_service(*, state_dir):
            calls.append(("restart", str(state_dir)))
            return {"service": "llm-wiki-native", "status": "ok"}

        return restart_service

    def fake_query_smoke_from_args(args):
        calls.append(("smoke_args", args.smoke_url, args.smoke_query, args.smoke_mode, args.smoke_query_vector_source))

        def query_smoke(*, state_dir, active):
            calls.append(("smoke", str(state_dir), active["workspace_id"]))
            return {"ok": True, "url": args.smoke_url}

        return query_smoke

    monkeypatch.setattr(batch_native_refresh, "build_prepared_workspace", fake_build_prepared_workspace)
    monkeypatch.setattr(batch_native_refresh, "finalize_prepared_workspace_for_state", fake_finalize_workspace)
    monkeypatch.setattr(batch_native_refresh, "restart_service_from_args", fake_restart_service_from_args)
    monkeypatch.setattr(batch_native_refresh, "query_smoke_from_args", fake_query_smoke_from_args)

    assert (
        batch_native_refresh.main(
            [
                "refresh",
                "--workdir",
                str(workdir),
                "--root",
                str(root),
                "--cutover",
                "--workspace-id",
                "candidate",
                "--restart-command",
                "svc restart llm-wiki-native",
                "--health-url",
                "http://127.0.0.1:9621/health",
                "--health-timeout",
                "3",
                "--smoke-url",
                "http://127.0.0.1:9621/query/data",
                "--smoke-query",
                "native cutover smoke",
                "--smoke-query-vector-source",
                "active-first-vector",
                "--require-unchanged-path",
                str(watched_dir),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)

    assert result["cutover"] is True
    assert result["service"]["status"] == "ok"
    assert calls == [
        ("smoke_args", "http://127.0.0.1:9621/query/data", "native cutover smoke", "mix", "active-first-vector"),
        ("restart_args", "svc restart llm-wiki-native", "http://127.0.0.1:9621/health", 3.0),
        ("build", "candidate"),
        ("finalize", "native graph incremental refresh: cutover"),
        ("restart", str(state_dir.resolve())),
        ("smoke", str(state_dir.resolve()), "candidate"),
    ]
def test_refresh_cutover_accepts_explicit_root_without_hardcoded_local_special_case(tmp_path, monkeypatch) -> None:
    state_dir = tmp_path / "native" / "state"
    state_dir.mkdir(parents=True)
    root = tmp_path / "operator-wiki"
    workspace_root = state_dir / "native_zvec" / "workspaces"
    watched_dir = tmp_path / "watched"
    watched_dir.mkdir()
    calls = []

    def fake_status(*args, **kwargs):
        calls.append(("status", args, kwargs))
        return {"should_refresh": True}

    def build_workspace(**kwargs):
        calls.append(("build", kwargs["workspace_id"]))
        return {"ok": True, "workspace_id": kwargs["workspace_id"]}

    def finalize_workspace(*, state_dir, reason):
        calls.append(("finalize", reason))
        return {"schema_version": 1, "workspace_id": "candidate", "status": "active"}

    def restart_service(*, state_dir):
        calls.append(("restart", str(state_dir)))
        return {"service": "llm-wiki-native", "status": "ok"}

    def query_smoke(*, state_dir, active):
        calls.append(("smoke", str(state_dir)))
        return {"ok": True}

    monkeypatch.setattr(batch_native_refresh, "status", fake_status)

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

    assert result["cutover_executed"] is True
    assert calls == [
        ("status", (root, state_dir), {}),
        ("build", "candidate"),
        ("finalize", "native graph incremental refresh: cutover"),
        ("restart", str(state_dir)),
        ("smoke", str(state_dir)),
        ("status", (root, state_dir), {}),
    ]
def test_refresh_cutover_success_preserves_pending_until_smoke_then_clears(tmp_path) -> None:
    state_dir = tmp_path / "wikigraph" / "state"
    root = tmp_path / "wiki"
    workspace_root = state_dir / "native_zvec" / "workspaces"
    watched_dir = tmp_path / "watched"
    watched_dir.mkdir()
    batch_native_refresh.mark_pending(state_dir, root, reason="manual-smoke")
    calls = []

    def build_workspace(**kwargs):
        calls.append(("build", kwargs["workspace_id"], kwargs["fill_missing_vectors"]))
        prepared_path = workspace_root.parent / "prepared_workspace.json"
        prepared_path.parent.mkdir(parents=True)
        prepared_path.write_text(
            json.dumps({"schema_version": 1, "workspace_id": kwargs["workspace_id"], "status": "prepared"}),
            encoding="utf-8",
        )
        return {"ok": True, "prepared_workspace": str(prepared_path), "workspace_id": kwargs["workspace_id"]}

    def finalize_workspace(*, state_dir, reason):
        calls.append(("finalize", reason))
        return {"schema_version": 1, "workspace_id": "candidate", "status": "active"}

    def restart_service(*, state_dir):
        calls.append(("restart", str(state_dir)))
        return {"service": "llm-wiki-native", "status": "ok"}

    def query_smoke(*, state_dir, active):
        calls.append(("smoke", str(state_dir), active["workspace_id"], batch_native_refresh.pending_ledger_path(state_dir).exists()))
        return {"ok": True, "url": "http://127.0.0.1:9621/query/data"}

    result = batch_native_refresh.refresh_cutover(
        root=root,
        state_dir=state_dir,
        workspace_root=workspace_root,
        workspace_id="candidate",
        embedding_profile="conservative",
        fill_missing_vectors=True,
        build_workspace=build_workspace,
        finalize_workspace=finalize_workspace,
        restart_service=restart_service,
        query_smoke=query_smoke,
        required_unchanged_paths=[watched_dir],
    )

    assert result["cutover"] is True
    assert result["cutover_executed"] is True
    assert result["build_executed"] is True
    assert result["restart_executed"] is True
    assert result["query_smoke_executed"] is True
    assert result["pending_clear_executed"] is True
    assert result["query_smoke"]["ok"] is True
    assert result["active"]["status"] == "active"
    assert result["service"]["status"] == "ok"
    assert calls == [
        ("build", "candidate", True),
        ("finalize", "native graph incremental refresh: cutover"),
        ("restart", str(state_dir)),
        ("smoke", str(state_dir), "candidate", True),
    ]
    assert not batch_native_refresh.pending_ledger_path(state_dir).exists()
@pytest.mark.parametrize(
    ("smoke_kwargs", "expected_terms"),
    [
        ({"smoke_raises": RuntimeError("smoke failed")}, ("smoke failed",)),
        ({"smoke_ok": False}, ("query smoke", "ok")),
    ],
)
def test_refresh_cutover_keeps_pending_for_smoke_error_shapes(
    tmp_path,
    smoke_kwargs: dict[str, object],
    expected_terms: tuple[str, ...],
) -> None:
    context = _cutover_context(tmp_path, pending=True)
    build_workspace, finalize_workspace, restart_service, query_smoke = _fake_cutover_hooks(context.calls, **smoke_kwargs)

    with pytest.raises(RuntimeError) as excinfo:
        batch_native_refresh.refresh_cutover(
            root=context.root,
            state_dir=context.state_dir,
            workspace_root=context.workspace_root,
            workspace_id="candidate",
            embedding_profile="conservative",
            build_workspace=build_workspace,
            finalize_workspace=finalize_workspace,
            restart_service=restart_service,
            query_smoke=query_smoke,
            required_unchanged_paths=[context.watched_dir],
        )

    message = str(excinfo.value)
    for term in expected_terms:
        assert term in message
    assert batch_native_refresh.pending_ledger_path(context.state_dir).exists()
def test_refresh_cutover_fails_before_pending_clear_when_required_unchanged_path_changes(tmp_path) -> None:
    context = _cutover_context(tmp_path, pending=True)
    watched_file = context.watched_dir / "storage.json"
    watched_file.write_text('{"stable":true}', encoding="utf-8")
    build_workspace, finalize_workspace, restart_service, query_smoke = _fake_cutover_hooks(
        context.calls,
        mutate_watched_path=watched_file,
    )

    with pytest.raises(RuntimeError, match="required unchanged path changed"):
        batch_native_refresh.refresh_cutover(
            root=context.root,
            state_dir=context.state_dir,
            workspace_root=context.workspace_root,
            workspace_id="candidate",
            embedding_profile="conservative",
            build_workspace=build_workspace,
            finalize_workspace=finalize_workspace,
            restart_service=restart_service,
            query_smoke=query_smoke,
            required_unchanged_paths=[context.watched_dir],
        )

    assert batch_native_refresh.pending_ledger_path(context.state_dir).exists()
def test_refresh_cutover_reuses_active_workspace_when_build_report_fingerprints_match(tmp_path, monkeypatch) -> None:
    context = _cutover_context(tmp_path, pending=True)
    fingerprints = {"custom_kg_manifest.json": {"exists": True, "sha256": "state-hash"}}
    active = {
        "schema_version": 1,
        "workspace_id": "active-a",
        "status": "active",
        "sqlite_path": str(context.workspace_root / "active-a" / "native.sqlite"),
        "zvec_path": str(context.workspace_root / "active-a" / "zvec_records"),
        "source_manifest_hash": "manifest-hash",
        "counts": {"chunks": 1, "entities": 1, "relationships": 0, "sections": 0},
        "lexical_span_count": 2,
        "source_root": str(context.root),
    }
    active_path = batch_native_refresh.active_workspace_path(context.state_dir)
    active_path.parent.mkdir(parents=True)
    active_path.write_text(json.dumps(active), encoding="utf-8")
    build_report_path = context.workspace_root / "active-a" / "build_report.json"
    build_report_path.parent.mkdir(parents=True)
    build_report_path.write_text(
        json.dumps({"ok": True, "workspace_id": "active-a", "input_fingerprints": fingerprints, "native_report": {"source_manifest_hash": "manifest-hash"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(batch_native_refresh, "state_input_fingerprints", lambda state_dir: fingerprints, raising=False)

    def fail_build_workspace(**_kwargs):
        raise AssertionError("active-fresh retry should not build another native workspace")

    def fail_finalize_workspace(**_kwargs):
        raise AssertionError("active-fresh retry should not finalize another native workspace")

    def restart_service(*, state_dir):
        context.calls.append(("restart", str(state_dir)))
        return {"service": "llm-wiki-native", "status": "ok"}

    def query_smoke(*, state_dir, active):
        context.calls.append(("smoke", str(state_dir), active["workspace_id"]))
        return {"ok": True, "url": "http://127.0.0.1:9621/query/data"}

    result = batch_native_refresh.refresh_cutover(
        root=context.root,
        state_dir=context.state_dir,
        workspace_root=context.workspace_root,
        workspace_id="candidate-b",
        embedding_profile="conservative",
        build_workspace=fail_build_workspace,
        finalize_workspace=fail_finalize_workspace,
        restart_service=restart_service,
        query_smoke=query_smoke,
        required_unchanged_paths=[context.watched_dir],
    )

    assert result["active_already_fresh"] is True
    assert result["build_executed"] is False
    assert result["restart_executed"] is True
    assert result["query_smoke_executed"] is True
    assert result["pending_clear_executed"] is True
    assert result["active"]["workspace_id"] == "active-a"
    assert context.calls == [("restart", str(context.state_dir)), ("smoke", str(context.state_dir), "active-a")]
    assert not batch_native_refresh.pending_ledger_path(context.state_dir).exists()


def test_refresh_cutover_does_not_use_active_fresh_skip_for_full_rebuild_policy(tmp_path, monkeypatch) -> None:
    context = _cutover_context(tmp_path, pending=True)
    build_workspace, finalize_workspace, restart_service, query_smoke = _fake_cutover_hooks(context.calls)
    full_policy = {
        "next_refresh_kind": batch_native_refresh.REFRESH_KIND_FULL_REBUILD,
        "completed_incremental_refresh_count": batch_native_refresh.NATIVE_INCREMENTAL_REFRESH_THRESHOLD,
        "incremental_rebuild_threshold": batch_native_refresh.NATIVE_INCREMENTAL_REFRESH_THRESHOLD,
        "vector_cache_required": True,
        "vector_cache_path": str(context.state_dir / "vector_cache.sqlite"),
    }

    def fake_status(root, state_dir):
        return {"should_refresh": True, "refresh_policy": full_policy}

    def fail_if_active_fresh_checked(**_kwargs):
        raise AssertionError("full-rebuild policy must not use active-fresh shortcut")

    monkeypatch.setattr(batch_native_refresh, "status", fake_status)
    monkeypatch.setattr(batch_native_refresh, "active_already_fresh_report", fail_if_active_fresh_checked)

    result = batch_native_refresh.refresh_cutover(
        root=context.root,
        state_dir=context.state_dir,
        workspace_root=context.workspace_root,
        workspace_id="candidate-full",
        embedding_profile="conservative",
        build_workspace=build_workspace,
        finalize_workspace=finalize_workspace,
        restart_service=restart_service,
        query_smoke=query_smoke,
        required_unchanged_paths=[context.watched_dir],
    )

    assert result["refresh_kind"] == batch_native_refresh.REFRESH_KIND_FULL_REBUILD
    assert result["build_executed"] is True
    assert context.calls[0] == ("build", "candidate-full")


def test_refresh_cutover_skipped_result_marks_no_execution(tmp_path) -> None:
    state_dir = tmp_path / "wikigraph" / "state"
    root = tmp_path / "wiki"
    workspace_root = state_dir / "native_zvec" / "workspaces"
    watched_dir = tmp_path / "watched"
    state_dir.mkdir(parents=True)
    watched_dir.mkdir()
    calls = []

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
        restart_service=restart_service,
        query_smoke=query_smoke,
        required_unchanged_paths=[watched_dir],
    )

    assert result["cutover"] is True
    assert result["skipped"] is True
    assert result["cutover_executed"] is False
    assert result["build_executed"] is False
    assert result["restart_executed"] is False
    assert result["query_smoke_executed"] is False
    assert result["pending_clear_executed"] is False
    assert result["unchanged_path_audit"]["ok"] is True
    assert calls == []
