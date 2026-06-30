import asyncio
from typing import Any

import httpx

from llm_wiki_native.api.server import create_app
from llm_wiki_native.retrieval.query_engine import NativeQueryEngine
from llm_wiki_native.storage.sqlite_workspace import NativeRecord, SQLiteWorkspace


def _record(workspace_id: str, record_id: str) -> NativeRecord:
    return NativeRecord(
        workspace_id=workspace_id,
        record_type="entity",
        record_id=record_id,
        vector_text="Alpha",
        content_hash=f"{record_id}:content",
        metadata_hash=f"{record_id}:metadata",
        vector_hash=f"{record_id}:vector",
        source_path="alpha.md",
        source_id=record_id,
        payload={"title": "Alpha"},
    )


def _app(tmp_path):
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    db.put_record(_record("native-test", "doc:a"))
    db.put_vector("native-test", "entity", "doc:a", "doc:a:vector", [1.0, 0.0])
    db.mark_audited("native-test", {"chunks": 0, "entities": 1, "relationships": 0, "sections": 0}, require_vectors=True)
    return create_app(NativeQueryEngine(db))


async def _request_async(app, method: str, path: str, *, raise_app_exceptions: bool = True, **kwargs: Any) -> httpx.Response:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, **kwargs)


def _request(app, method: str, path: str, *, raise_app_exceptions: bool = True, **kwargs: Any) -> httpx.Response:
    return asyncio.run(_request_async(app, method, path, raise_app_exceptions=raise_app_exceptions, **kwargs))


def _use_direct_threadpool(monkeypatch) -> None:
    async def direct_threadpool(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("llm_wiki_native.api.server.run_in_threadpool", direct_threadpool)


def test_shadow_api_health_reports_native_port_and_ready(tmp_path) -> None:
    app = _app(tmp_path)

    response = _request(app, "GET", "/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "llm-wiki-native",
        "backend": "native-zvec",
        "default_port": 9622,
        "active_workspace_id": None,
    }


def test_shadow_api_query_data_and_trace_contract(tmp_path, monkeypatch) -> None:
    _use_direct_threadpool(monkeypatch)
    app = _app(tmp_path)
    payload = {
        "workspace_id": "native-test",
        "query": "alpha",
        "query_vector": [1.0, 0.0],
        "mode": "mix",
        "top_k": 1,
        "record_types": ["entity"],
    }

    data_response = _request(app, "POST", "/query/data", json=payload)
    trace_response = _request(app, "POST", "/native/query/trace", json=payload)

    assert data_response.status_code == 200
    assert data_response.json()["context_blocks"][0]["source_path"] == "alpha.md"
    assert trace_response.status_code == 200
    assert trace_response.json()["trace"]["mode"] == "mix"


def test_shadow_api_returns_structured_400_for_validation_errors(tmp_path, monkeypatch) -> None:
    _use_direct_threadpool(monkeypatch)
    app = _app(tmp_path)
    payload = {
        "workspace_id": "native-test",
        "query": "alpha",
        "query_vector": [1.0, 0.0],
        "mode": "mix",
        "record_types": ["unknown"],
    }

    response = _request(app, "POST", "/query/data", json=payload, raise_app_exceptions=False)

    assert response.status_code == 400
    assert "record_type" in response.json()["error"]


def test_shadow_api_rejects_old_graph_mode_as_unsupported(tmp_path, monkeypatch) -> None:
    _use_direct_threadpool(monkeypatch)
    app = _app(tmp_path)
    payload = {"workspace_id": "native-test", "query_vector": [1.0, 0.0], "record_types": ["entity"], "mode": "local"}

    response = _request(app, "POST", "/native/query/trace", json=payload, raise_app_exceptions=False)

    assert response.status_code == 400
    assert "unsupported mode" in response.json()["error"]


def test_shadow_api_requires_bearer_token_when_configured(tmp_path, monkeypatch) -> None:
    _use_direct_threadpool(monkeypatch)
    monkeypatch.setenv("LLM_WIKI_NATIVE_API_KEY", "secret-token")
    app = _app(tmp_path)
    payload = {"workspace_id": "native-test", "query_vector": [1.0, 0.0], "record_types": ["entity"], "top_k": 1}

    assert _request(app, "POST", "/native/query/trace", json=payload, raise_app_exceptions=False).status_code == 401
    assert _request(app, "POST", "/native/query/trace", json=payload, headers={"Authorization": "Bearer wrong"}, raise_app_exceptions=False).status_code == 401
    assert _request(app, "POST", "/native/query/trace", json=payload, headers={"Authorization": "Bearer secret-token"}).status_code == 200


def test_shadow_api_validates_vectors_and_clamps_request_limits(monkeypatch) -> None:
    _use_direct_threadpool(monkeypatch)
    calls = []

    class FakeEngine:
        def query(self, **kwargs):
            calls.append(kwargs)
            return {"hits": [], "trace": {"mode": kwargs["mode"], "top_k": kwargs["top_k"], "neighbor_limit": kwargs["neighbor_limit"], "record_types": list(kwargs["record_types"])}}

    app = create_app(FakeEngine())

    invalid = _request(app, "POST", "/native/query/trace", json={"workspace_id": "native-test", "query_vector": ["nan"]}, raise_app_exceptions=False)
    assert invalid.status_code == 400
    assert "finite" in invalid.json()["error"]

    response = _request(
        app,
        "POST",
        "/native/query/trace",
        json={
            "workspace_id": "native-test",
            "query_vector": [1.0, 0.0],
            "top_k": 999,
            "neighbor_limit": 999,
            "record_types": ["entity", "relationship", "chunk", "section"],
        },
    )

    assert response.status_code == 200
    assert calls[-1]["top_k"] == 100
    assert calls[-1]["neighbor_limit"] == 50
    assert calls[-1]["record_types"] == ("entity", "relationship", "chunk", "section")


def test_shadow_api_rejects_oversized_request_body(tmp_path) -> None:
    app = _app(tmp_path)

    response = _request(app, "POST", "/native/query/trace", content=b"{" + b" " * 1_000_001 + b"}", headers={"content-type": "application/json"}, raise_app_exceptions=False)

    assert response.status_code == 413


def test_shadow_api_runs_sync_query_in_threadpool(monkeypatch) -> None:
    called = {"threadpool": False}

    async def fake_run_in_threadpool(func, *args, **kwargs):
        called["threadpool"] = True
        return func(*args, **kwargs)

    class FakeEngine:
        def query(self, **kwargs):
            return {"hits": [], "trace": {"mode": kwargs["mode"]}}

    monkeypatch.setattr("llm_wiki_native.api.server.run_in_threadpool", fake_run_in_threadpool)
    app = create_app(FakeEngine())

    response = _request(app, "POST", "/native/query/trace", json={"workspace_id": "native-test", "query_vector": [1.0], "record_types": ["entity"]}, raise_app_exceptions=False)

    assert response.status_code == 200
    assert called["threadpool"] is True
