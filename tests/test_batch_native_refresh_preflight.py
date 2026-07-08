from __future__ import annotations

import json
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


def test_refresh_cutover_cli_rejects_invalid_cutover_options_before_calls(tmp_path, monkeypatch) -> None:
    context = _cutover_context(tmp_path, pending=True)
    calls = context.calls

    def fake_status(*args, **kwargs):
        calls.append(("status", args, kwargs))
        raise AssertionError("invalid cutover options must fail before status")

    def fail_build_prepared_workspace(**kwargs):
        calls.append(("build", kwargs["workspace_id"]))
        raise AssertionError("invalid cutover options must fail before build")

    def fail_finalize_workspace(*, state_dir, reason):
        calls.append(("finalize", reason))
        raise AssertionError("invalid cutover options must fail before finalize")

    def fail_restart_service_from_args(args):
        calls.append(("restart_args", args.restart_command))
        raise AssertionError("invalid cutover options must fail before restart hook construction")

    monkeypatch.setattr(batch_native_refresh, "status", fake_status)
    monkeypatch.setattr(batch_native_refresh, "build_prepared_workspace", fail_build_prepared_workspace)
    monkeypatch.setattr(batch_native_refresh, "finalize_prepared_workspace_for_state", fail_finalize_workspace)
    monkeypatch.setattr(batch_native_refresh, "restart_service_from_args", fail_restart_service_from_args)

    args = [
        "refresh",
        "--workdir",
        str(context.state_dir.parent),
        "--root",
        str(context.root),
        "--cutover",
        "--workspace-id",
        "candidate",
        "--health-url",
        "http://127.0.0.1:9621/health",
        "--require-unchanged-path",
        str(context.watched_dir),
        "--smoke-url",
        "http://127.0.0.1:9621/query/data",
        "--smoke-query",
        "native cutover smoke",
    ]

    with pytest.raises(ValueError) as excinfo:
        batch_native_refresh.main(args)

    assert "--restart-command" in str(excinfo.value)
    assert calls == []


def test_preflight_cutover_reports_missing_smoke_pair_without_status_or_build(tmp_path, capsys) -> None:
    workdir = tmp_path / "wikigraph"
    state_dir = workdir / "state"
    state_dir.mkdir(parents=True)
    root = tmp_path / "wiki"
    watched_dir = tmp_path / "watched"
    watched_dir.mkdir()

    code = batch_native_refresh.main(
        [
            "preflight-cutover",
            "--workdir",
            str(workdir),
            "--root",
            str(root),
            "--restart-command",
            "svc restart llm-wiki-native",
            "--smoke-url",
            "http://127.0.0.1:9621/query/data",
            "--require-unchanged-path",
            str(watched_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 1
    assert payload["ok"] is False
    assert "missing_smoke_query" in payload["errors"]
    assert "missing_restart_command" not in payload["errors"]
    assert payload["path_errors"] == []
    assert not batch_native_refresh.pending_ledger_path(state_dir).exists()


def test_preflight_cutover_accepts_valid_guard_shape_without_pending_status(tmp_path, capsys) -> None:
    workdir = tmp_path / "wikigraph"
    state_dir = workdir / "state"
    state_dir.mkdir(parents=True)
    root = tmp_path / "wiki"
    watched_dir = tmp_path / "watched"
    watched_dir.mkdir()

    code = batch_native_refresh.main(
        [
            "preflight-cutover",
            "--workdir",
            str(workdir),
            "--root",
            str(root),
            "--restart-command",
            "svc restart llm-wiki-native",
            "--smoke-url",
            "http://127.0.0.1:9621/query/data",
            "--smoke-query",
            "native cutover smoke",
            "--require-unchanged-path",
            str(watched_dir),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["ok"] is True
    assert payload["errors"] == []
    assert not batch_native_refresh.pending_ledger_path(state_dir).exists()


def test_refresh_cutover_rejects_invalid_pre_status_configuration(tmp_path, monkeypatch) -> None:
    cases = [
        ("missing_state_dir", False, "existing", ("state_dir", "must exist")),
        ("missing_watched_path", True, "missing", ("--require-unchanged-path", "must exist")),
        ("native_output_watched_path", True, "native_output", ("--require-unchanged-path", "native output")),
    ]
    for case_name, state_exists, watch_path_kind, expected_terms in cases:
        context = _cutover_context(tmp_path / case_name)
        if state_exists:
            context.state_dir.mkdir(parents=True)
        if watch_path_kind == "existing":
            context.watched_dir.mkdir(parents=True)
            watched_path = context.watched_dir
        elif watch_path_kind == "missing":
            watched_path = tmp_path / f"missing-storage-boundary-{case_name}"
        elif watch_path_kind == "native_output":
            watched_path = context.workspace_root.parent
            watched_path.mkdir(parents=True)
        else:  # pragma: no cover - test data guard
            raise AssertionError(watch_path_kind)

        build_workspace, finalize_workspace, restart_service, query_smoke = _fake_cutover_hooks(context.calls)

        def fake_status(*args, **kwargs):
            context.calls.append(("status", args, kwargs))
            raise AssertionError(f"{case_name} must fail before status")

        monkeypatch.setattr(batch_native_refresh, "status", fake_status)

        with pytest.raises(ValueError) as excinfo:
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
                required_unchanged_paths=[watched_path],
            )

        message = str(excinfo.value)
        for term in expected_terms:
            assert term in message
        assert context.calls == []


def test_required_unchanged_path_audit_detects_empty_directory_creation(tmp_path) -> None:
    watched_dir = tmp_path / "watched"
    watched_dir.mkdir()
    before = batch_native_refresh.snapshot_required_unchanged_paths([watched_dir])

    (watched_dir / "empty-child").mkdir()
    audit = batch_native_refresh.audit_required_unchanged_paths(before)

    assert audit["ok"] is False
    assert audit["paths"] == [
        {
            "path": str(watched_dir.resolve()),
            "ok": False,
            "changed": True,
        }
    ]
