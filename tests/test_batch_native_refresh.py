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


def test_status_ignores_unowned_old_refresh_ledger(tmp_path, capsys) -> None:
    old_backend = "light" + "rag"
    workdir = tmp_path / "wikigraph"
    state_dir = workdir / "state"
    old_path = state_dir / f"pending_{old_backend}_refresh.json"
    native_path = state_dir / "pending_native_refresh.json"
    old_path.parent.mkdir(parents=True)
    old_path.write_text(
        json.dumps({"schema_version": 1, "pending": [{"reason": "old"}]}),
        encoding="utf-8",
    )

    assert batch_native_refresh.main(["status", "--workdir", str(workdir)]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["pending_count"] == 0
    assert payload["should_refresh"] is False
    assert old_path.exists()
    assert not native_path.exists()


def test_refresh_without_prepare_only_requires_explicit_cutover(tmp_path) -> None:
    workdir = tmp_path / "wikigraph"

    try:
        batch_native_refresh.main(["refresh", "--workdir", str(workdir)])
    except ValueError as exc:
        assert "--cutover" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("refresh without --prepare-only must fail closed before explicit cutover")


def test_restart_service_command_runs_command_and_optional_health_probe() -> None:
    calls = []

    class Completed:
        returncode = 0
        stdout = "restarted\n"
        stderr = ""

    class Response:
        status = 200

        def read(self) -> bytes:
            return b'{"backend":"native-zvec"}'

        def close(self) -> None:
            calls.append(("close",))

    def runner(command, *, check, capture_output, text, timeout):
        calls.append(("runner", command, check, capture_output, text, timeout))
        return Completed()

    def urlopen(url, *, timeout):
        calls.append(("urlopen", url, timeout))
        return Response()

    result = batch_native_refresh.restart_service_command(
        ["svc", "restart", "llm-wiki-native"],
        health_url="http://127.0.0.1:9621/health",
        timeout_seconds=3,
        runner=runner,
        urlopen=urlopen,
    )

    assert result == {
        "service": "llm-wiki-native",
        "status": "ok",
        "restart": {
            "command": ["svc", "restart", "llm-wiki-native"],
            "returncode": 0,
            "stdout": "restarted\n",
            "stderr": "",
        },
        "health": {
            "url": "http://127.0.0.1:9621/health",
            "status": 200,
            "ok": True,
            "body": '{"backend":"native-zvec"}',
        },
    }
    assert calls == [
        ("runner", ["svc", "restart", "llm-wiki-native"], False, True, True, 3),
        ("urlopen", "http://127.0.0.1:9621/health", 3),
        ("close",),
    ]


def test_query_smoke_request_posts_native_query_data() -> None:
    calls = []

    class Response:
        status = 200

        def read(self) -> bytes:
            return b'{"results":[{"record_id":"doc:a"}],"trace":{"retrieval_backend":"zvec","vector_hit_count":1}}'

        def close(self) -> None:
            calls.append(("close",))

    def urlopen(request, *, timeout):
        calls.append(
            (
                "urlopen",
                request.full_url,
                request.get_method(),
                json.loads(request.data.decode("utf-8")),
                dict(request.header_items()),
                timeout,
            )
        )
        return Response()

    result = batch_native_refresh.query_smoke_request(
        "http://127.0.0.1:9621/query/data",
        query="native smoke",
        workspace_id="candidate",
        timeout_seconds=2,
        urlopen=urlopen,
    )

    assert result == {
        "url": "http://127.0.0.1:9621/query/data",
        "status": 200,
        "ok": True,
        "request": {"query": "native smoke", "mode": "mix", "workspace_id": "candidate"},
        "body": {
            "results": [{"record_id": "doc:a"}],
            "trace": {"retrieval_backend": "zvec", "vector_hit_count": 1},
        },
    }
    assert calls == [
        (
            "urlopen",
            "http://127.0.0.1:9621/query/data",
            "POST",
            {"query": "native smoke", "mode": "mix", "workspace_id": "candidate"},
            {"Content-type": "application/json"},
            2,
        ),
        ("close",),
    ]


def test_query_smoke_request_posts_explicit_query_vector_without_echoing_full_vector() -> None:
    calls = []

    class Response:
        status = 200

        def read(self) -> bytes:
            return b'{"context_blocks":[{"record_id":"doc:a"}],"trace":{"retrieval_backend":"zvec","vector_hit_count":1}}'

        def close(self) -> None:
            calls.append(("close",))

    def urlopen(request, *, timeout):
        calls.append(
            (
                "urlopen",
                request.full_url,
                json.loads(request.data.decode("utf-8")),
                timeout,
            )
        )
        return Response()

    result = batch_native_refresh.query_smoke_request(
        "http://127.0.0.1:9621/query/data",
        query="native vector smoke",
        workspace_id="candidate",
        query_vector=[0.25, 0.75],
        query_vector_source="inline-json",
        timeout_seconds=2,
        urlopen=urlopen,
    )

    assert calls == [
        (
            "urlopen",
            "http://127.0.0.1:9621/query/data",
            {
                "query": "native vector smoke",
                "mode": "mix",
                "workspace_id": "candidate",
                "query_vector": [0.25, 0.75],
            },
            2,
        ),
        ("close",),
    ]
    assert result["ok"] is True
    assert result["request"] == {
        "query": "native vector smoke",
        "mode": "mix",
        "workspace_id": "candidate",
        "query_vector_present": True,
        "query_vector_dim": 2,
        "query_vector_source": "inline-json",
    }
    assert "query_vector" not in result["request"]


def test_query_smoke_from_args_accepts_inline_query_vector_json(monkeypatch) -> None:
    calls = []

    def fake_query_smoke_request(url, **kwargs):
        calls.append((url, kwargs))
        return {"ok": True, "request": {"query_vector_dim": len(kwargs["query_vector"])}}

    monkeypatch.setattr(batch_native_refresh, "query_smoke_request", fake_query_smoke_request)
    args = SimpleNamespace(
        smoke_url="http://127.0.0.1:9621/query/data",
        smoke_query="native vector smoke",
        smoke_mode="mix",
        smoke_workspace_id=None,
        workspace_id="candidate",
        smoke_timeout=3.0,
        smoke_query_vector_json="[0.25, 0.75]",
        smoke_query_vector_file=None,
        smoke_query_vector_source=None,
    )

    query_smoke = batch_native_refresh.query_smoke_from_args(args)
    result = query_smoke(state_dir=Path("/state"), active={"workspace_id": "active-candidate"})

    assert result["ok"] is True
    assert calls == [
        (
            "http://127.0.0.1:9621/query/data",
            {
                "query": "native vector smoke",
                "mode": "mix",
                "workspace_id": "active-candidate",
                "timeout_seconds": 3.0,
                "query_vector": [0.25, 0.75],
                "query_vector_source": "inline-json",
            },
        )
    ]


def test_query_smoke_from_args_loads_active_first_vector_source(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "native.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE vector(workspace_id TEXT, record_type TEXT, record_id TEXT, dim INTEGER, vector_blob BLOB)")
        conn.execute(
            "INSERT INTO vector(workspace_id, record_type, record_id, dim, vector_blob) VALUES(?, ?, ?, ?, ?)",
            ("active-candidate", "chunk", "chunk-a", 2, struct.pack("<2f", 0.25, 0.75)),
        )

    calls = []

    def fake_query_smoke_request(url, **kwargs):
        calls.append((url, kwargs))
        return {"ok": True, "request": {"query_vector_dim": len(kwargs["query_vector"])}}

    monkeypatch.setattr(batch_native_refresh, "query_smoke_request", fake_query_smoke_request)
    args = SimpleNamespace(
        smoke_url="http://127.0.0.1:9621/query/data",
        smoke_query="native vector smoke",
        smoke_mode="mix",
        smoke_workspace_id=None,
        workspace_id="candidate",
        smoke_timeout=3.0,
        smoke_query_vector_json=None,
        smoke_query_vector_file=None,
        smoke_query_vector_source="active-first-vector",
    )

    query_smoke = batch_native_refresh.query_smoke_from_args(args)
    result = query_smoke(state_dir=tmp_path / "state", active={"workspace_id": "active-candidate", "sqlite_path": str(db_path)})

    assert result["ok"] is True
    assert calls == [
        (
            "http://127.0.0.1:9621/query/data",
            {
                "query": "native vector smoke",
                "mode": "mix",
                "workspace_id": "active-candidate",
                "timeout_seconds": 3.0,
                "query_vector": [0.25, 0.75],
                "query_vector_source": "active-first-vector:chunk:chunk-a",
            },
        )
    ]


def test_query_smoke_request_requires_query_data_endpoint_before_http() -> None:
    calls = []

    def urlopen(request, *, timeout):
        calls.append(("urlopen", request.full_url, timeout))
        raise AssertionError("non-/query/data smoke must fail before HTTP")

    try:
        batch_native_refresh.query_smoke_request(
            "http://127.0.0.1:9621/query",
            query="native smoke",
            urlopen=urlopen,
        )
    except ValueError as exc:
        assert "/query/data" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("query smoke endpoint must be /query/data")

    assert calls == []


@pytest.mark.parametrize(
    ("mode", "expected_terms"),
    [
        ("bypass", ("bypass",)),
        ("naive", ("mix", "naive")),
    ],
)
def test_query_smoke_request_rejects_invalid_mode_before_http(mode: str, expected_terms: tuple[str, ...]) -> None:
    calls = []

    def urlopen(request, *, timeout):
        calls.append(("urlopen", request.full_url, timeout))
        raise AssertionError("invalid smoke mode must fail before HTTP")

    try:
        batch_native_refresh.query_smoke_request(
            "http://127.0.0.1:9621/query/data",
            query="native smoke",
            mode=mode,
            urlopen=urlopen,
        )
    except ValueError as exc:
        message = str(exc)
        for term in expected_terms:
            assert term in message
    else:  # pragma: no cover - assertion branch
        raise AssertionError("invalid smoke mode must be rejected")

    assert calls == []


def test_query_smoke_request_requires_zvec_trace_hits() -> None:
    class Response:
        status = 200

        def read(self) -> bytes:
            return b'{"trace":{"retrieval_backend":"bypass","vector_hit_count":0}}'

        def close(self) -> None:
            pass

    def urlopen(request, *, timeout):
        return Response()

    try:
        batch_native_refresh.query_smoke_request(
            "http://127.0.0.1:9621/query/data",
            query="native smoke",
            mode="mix",
            urlopen=urlopen,
        )
    except RuntimeError as exc:
        assert "zvec" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("non-zvec smoke response must fail")


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
        ("finalize", "native refresh cutover"),
        ("restart", str(state_dir.resolve())),
        ("smoke", str(state_dir.resolve()), "candidate"),
    ]


@pytest.mark.parametrize(
    ("case_name", "smoke_url", "smoke_mode", "include_smoke", "include_restart", "expected_terms"),
    [
        ("bypass_mode", "http://127.0.0.1:9621/query/data", "bypass", True, True, ("bypass",)),
        ("non_mix_mode", "http://127.0.0.1:9621/query/data", "naive", True, True, ("mix", "naive")),
        ("query_endpoint", "http://127.0.0.1:9621/query", "mix", True, True, ("/query/data",)),
        ("missing_smoke", "http://127.0.0.1:9621/query/data", "mix", False, True, ("--smoke-url", "--smoke-query")),
        ("missing_restart", "http://127.0.0.1:9621/query/data", "mix", True, False, ("--restart-command",)),
    ],
)
def test_refresh_cutover_cli_rejects_invalid_cutover_options_before_calls(
    tmp_path,
    monkeypatch,
    case_name: str,
    smoke_url: str,
    smoke_mode: str,
    include_smoke: bool,
    include_restart: bool,
    expected_terms: tuple[str, ...],
) -> None:
    context = _cutover_context(tmp_path, pending=True)
    calls = context.calls

    def fake_status(*args, **kwargs):
        calls.append(("status", args, kwargs))
        raise AssertionError(f"{case_name} must fail before status")

    def fail_build_prepared_workspace(**kwargs):
        calls.append(("build", kwargs["workspace_id"]))
        raise AssertionError(f"{case_name} must fail before build")

    def fail_finalize_workspace(*, state_dir, reason):
        calls.append(("finalize", reason))
        raise AssertionError(f"{case_name} must fail before finalize")

    def fail_restart_service_from_args(args):
        calls.append(("restart_args", args.restart_command))
        raise AssertionError(f"{case_name} must fail before restart hook construction")

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
    ]
    if include_restart:
        args.extend(["--restart-command", "svc restart llm-wiki-native"])
    if include_smoke:
        args.extend(["--smoke-url", smoke_url, "--smoke-query", "native cutover smoke"])
    if smoke_mode != "mix":
        args.extend(["--smoke-mode", smoke_mode])

    with pytest.raises(ValueError) as excinfo:
        batch_native_refresh.main(args)

    message = str(excinfo.value)
    for term in expected_terms:
        assert term in message
    assert calls == []


@pytest.mark.parametrize(
    ("case_name", "state_exists", "watch_path_kind", "query_smoke_present", "required_paths", "expected_terms"),
    [
        ("missing_unchanged_path_guard", True, "existing", True, "empty", ("--require-unchanged-path",)),
        ("missing_query_smoke", True, "existing", False, "existing", ("--smoke-url", "--smoke-query")),
        ("missing_state_dir", False, "existing", True, "existing", ("state_dir", "must exist")),
        ("missing_watched_path", True, "missing", True, "watched", ("--require-unchanged-path", "must exist")),
        ("native_output_watched_path", True, "native_output", True, "watched", ("--require-unchanged-path", "native output")),
    ],
)
def test_refresh_cutover_rejects_invalid_pre_status_configuration(
    tmp_path,
    monkeypatch,
    case_name: str,
    state_exists: bool,
    watch_path_kind: str,
    query_smoke_present: bool,
    required_paths: str,
    expected_terms: tuple[str, ...],
) -> None:
    context = _cutover_context(tmp_path)
    if state_exists:
        context.state_dir.mkdir(parents=True)
    if watch_path_kind == "existing":
        context.watched_dir.mkdir(parents=True)
        watched_path = context.watched_dir
    elif watch_path_kind == "missing":
        watched_path = tmp_path / "missing-storage-boundary"
    elif watch_path_kind == "native_output":
        watched_path = context.workspace_root.parent
        watched_path.mkdir(parents=True)
    else:  # pragma: no cover - parametrization guard
        raise AssertionError(watch_path_kind)

    paths = [] if required_paths == "empty" else [watched_path]
    build_workspace, finalize_workspace, restart_service, query_smoke = _fake_cutover_hooks(context.calls)

    def fake_status(*args, **kwargs):
        context.calls.append(("status", args, kwargs))
        raise AssertionError(f"{case_name} must fail before status")

    monkeypatch.setattr(batch_native_refresh, "status", fake_status)

    kwargs = {
        "root": context.root,
        "state_dir": context.state_dir,
        "workspace_root": context.workspace_root,
        "workspace_id": "candidate",
        "embedding_profile": "conservative",
        "build_workspace": build_workspace,
        "finalize_workspace": finalize_workspace,
        "restart_service": restart_service,
        "required_unchanged_paths": paths,
    }
    if query_smoke_present:
        kwargs["query_smoke"] = query_smoke

    with pytest.raises(ValueError) as excinfo:
        batch_native_refresh.refresh_cutover(**kwargs)

    message = str(excinfo.value)
    for term in expected_terms:
        assert term in message
    assert context.calls == []


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
        ("finalize", "native refresh cutover"),
        ("restart", str(state_dir)),
        ("smoke", str(state_dir)),
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
        ("finalize", "native refresh cutover"),
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
