from __future__ import annotations

import json
from pathlib import Path

import pytest

from ops import batch_native_refresh
from support import cutover_cli_args  # noqa: E402
from support import cutover_context as _cutover_context  # noqa: E402
from support import fake_cutover_hooks as _fake_cutover_hooks  # noqa: E402


def _fail_before(calls: list, recorded, message: str):
    def hook(*args, **kwargs):
        calls.append(recorded(args, kwargs))
        raise AssertionError(message)

    return hook


def test_refresh_cutover_cli_rejects_invalid_cutover_options_before_calls(tmp_path, monkeypatch) -> None:
    context = _cutover_context(tmp_path, pending=True)
    calls = context.calls

    monkeypatch.setattr(
        batch_native_refresh,
        "status",
        _fail_before(calls, lambda args, kwargs: ("status", args, kwargs), "invalid cutover options must fail before status"),
    )
    monkeypatch.setattr(
        batch_native_refresh,
        "build_prepared_workspace",
        _fail_before(calls, lambda _args, kwargs: ("build", kwargs["workspace_id"]), "invalid cutover options must fail before build"),
    )
    monkeypatch.setattr(
        batch_native_refresh,
        "finalize_prepared_workspace_for_state",
        _fail_before(calls, lambda _args, kwargs: ("finalize", kwargs["reason"]), "invalid cutover options must fail before finalize"),
    )
    monkeypatch.setattr(
        batch_native_refresh,
        "restart_service_from_args",
        _fail_before(
            calls,
            lambda args, _kwargs: ("restart_args", args[0].restart_command),
            "invalid cutover options must fail before restart hook construction",
        ),
    )

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


@pytest.mark.parametrize(
    "include_smoke_query,expected_code,expected_ok",
    [(False, 1, False), (True, 0, True)],
    ids=["missing_smoke_query", "valid_guard"],
)
def test_preflight_cutover_validates_smoke_pair_without_pending_status(
    tmp_path, capsys, include_smoke_query: bool, expected_code: int, expected_ok: bool
) -> None:
    workdir = tmp_path / "wikigraph"
    state_dir = workdir / "state"
    state_dir.mkdir(parents=True)
    root = tmp_path / "wiki"
    watched_dir = tmp_path / "watched"
    watched_dir.mkdir()
    extra = [
        "--restart-command",
        "svc restart llm-wiki-native",
        "--smoke-url",
        "http://127.0.0.1:9621/query/data",
    ]
    if include_smoke_query:
        extra.extend(["--smoke-query", "native cutover smoke"])
    extra.extend(["--require-unchanged-path", str(watched_dir)])
    argv = cutover_cli_args(workdir, root, *extra, command="preflight-cutover")

    code = batch_native_refresh.main(argv)
    payload = json.loads(capsys.readouterr().out)

    assert code == expected_code
    assert payload["ok"] is expected_ok
    if include_smoke_query:
        assert payload["errors"] == []
    else:
        assert "missing_smoke_query" in payload["errors"]
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
