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


def test_query_smoke_request_accepts_zvec_lexical_backend() -> None:
    class Response:
        status = 200

        def read(self) -> bytes:
            return b'{"context_blocks":[],"trace":{"retrieval_backend":"zvec+lexical","vector_hit_count":1}}'

        def close(self) -> None:
            pass

    result = batch_native_refresh.query_smoke_request(
        "http://127.0.0.1:9621/query/data",
        query="native smoke",
        urlopen=lambda request, *, timeout: Response(),
    )

    assert result["ok"] is True


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


def test_query_smoke_request_rejects_invalid_endpoint_or_mode_before_http() -> None:
    cases = [
        ("http://127.0.0.1:9621/query", "mix", ("/query/data",)),
        ("http://127.0.0.1:9621/query/data", "bypass", ("bypass",)),
        ("http://127.0.0.1:9621/query/data", "naive", ("mix", "naive")),
    ]
    for url, mode, expected_terms in cases:
        calls = []

        def urlopen(request, *, timeout):
            calls.append(("urlopen", request.full_url, timeout))
            raise AssertionError("invalid smoke request must fail before HTTP")

        with pytest.raises(ValueError) as excinfo:
            batch_native_refresh.query_smoke_request(
                url,
                query="native smoke",
                mode=mode,
                urlopen=urlopen,
            )
        for term in expected_terms:
            assert term in str(excinfo.value)
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
