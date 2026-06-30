from typing import Any

from llm_wiki_native.api.server import create_app
from llm_wiki_native.retrieval.query_engine import NativeQueryEngine
from llm_wiki_native.storage.sqlite_workspace import SQLiteWorkspace
from support import native_record, request_asgi as _request


def patch_direct_threadpool(monkeypatch: Any, target: str = "llm_wiki_native.api.server.run_in_threadpool") -> None:
    async def direct_threadpool(func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    monkeypatch.setattr(target, direct_threadpool)


def _app(tmp_path):
    class Hit:
        doc_id = "entity:doc:a"
        score = 1.0
        fields = {"record_type": "entity", "record_id": "doc:a"}

    class ZvecWorkspace:
        def query_mix(self, query: str, query_vector: list[float], top_k: int, filter_expr: str | None):
            return [Hit()]

        def query_vector(self, query_vector: list[float], top_k: int, filter_expr: str | None):
            return [Hit()]

    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    db.put_record(native_record("native-test", "entity", "doc:a", "Alpha", source_path="alpha.md"))
    db.put_vector("native-test", "entity", "doc:a", "doc:a:vector", [1.0, 0.0])
    db.mark_audited("native-test", {"chunks": 0, "entities": 1, "relationships": 0, "sections": 0}, require_vectors=True)
    return create_app(NativeQueryEngine(db, zvec_workspace=ZvecWorkspace()))


def test_native_api_health_reports_native_port_and_ready(tmp_path) -> None:
    app = _app(tmp_path)

    response = _request(app, "GET", "/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "llm-wiki-native",
        "backend": "native-zvec",
        "default_port": 9621,
        "active_workspace_id": None,
    }


def test_native_api_query_data_and_trace_contract(tmp_path, monkeypatch) -> None:
    patch_direct_threadpool(monkeypatch)
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

    assert data_response.status_code == 200
    body = data_response.json()
    assert body["context_blocks"][0]["source_path"] == "alpha.md"
    assert body["trace"]["mode"] == "mix"


def test_native_api_returns_structured_400_for_validation_errors(tmp_path, monkeypatch) -> None:
    patch_direct_threadpool(monkeypatch)
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


def test_native_api_rejects_old_graph_mode_as_unsupported(tmp_path, monkeypatch) -> None:
    patch_direct_threadpool(monkeypatch)
    app = _app(tmp_path)
    payload = {"workspace_id": "native-test", "query_vector": [1.0, 0.0], "record_types": ["entity"], "mode": "local"}

    response = _request(app, "POST", "/query/data", json=payload, raise_app_exceptions=False)

    assert response.status_code == 400
    assert "unsupported mode" in response.json()["error"]


def test_native_api_requires_bearer_token_when_configured(tmp_path, monkeypatch) -> None:
    patch_direct_threadpool(monkeypatch)
    monkeypatch.setenv("LLM_WIKI_NATIVE_API_KEY", "secret-token")
    app = _app(tmp_path)
    payload = {"workspace_id": "native-test", "query_vector": [1.0, 0.0], "record_types": ["entity"], "top_k": 1}

    assert _request(app, "POST", "/query/data", json=payload, raise_app_exceptions=False).status_code == 401
    assert _request(app, "POST", "/query/data", json=payload, headers={"Authorization": "Bearer wrong"}, raise_app_exceptions=False).status_code == 401
    assert _request(app, "POST", "/query/data", json=payload, headers={"Authorization": "Bearer secret-token"}).status_code == 200


def test_native_api_validates_vectors_and_clamps_request_limits(monkeypatch) -> None:
    patch_direct_threadpool(monkeypatch)
    calls = []

    class FakeEngine:
        def query(self, **kwargs):
            calls.append(kwargs)
            return {"hits": [], "trace": {"mode": kwargs["mode"], "top_k": kwargs["top_k"], "neighbor_limit": kwargs["neighbor_limit"], "record_types": list(kwargs["record_types"])}}

    app = create_app(FakeEngine())

    invalid = _request(app, "POST", "/query/data", json={"workspace_id": "native-test", "query_vector": ["nan"]}, raise_app_exceptions=False)
    assert invalid.status_code == 400
    assert "finite" in invalid.json()["error"]

    response = _request(
        app,
        "POST",
        "/query/data",
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


def test_native_api_rejects_oversized_request_body(tmp_path) -> None:
    app = _app(tmp_path)

    response = _request(app, "POST", "/query/data", content=b"{" + b" " * 1_000_001 + b"}", headers={"content-type": "application/json"}, raise_app_exceptions=False)

    assert response.status_code == 413


def test_native_api_runs_sync_query_in_threadpool(monkeypatch) -> None:
    called = {"threadpool": False}

    async def fake_run_in_threadpool(func, *args, **kwargs):
        called["threadpool"] = True
        return func(*args, **kwargs)

    class FakeEngine:
        def query(self, **kwargs):
            return {"hits": [], "trace": {"mode": kwargs["mode"]}}

    monkeypatch.setattr("llm_wiki_native.api.server.run_in_threadpool", fake_run_in_threadpool)
    app = create_app(FakeEngine())

    response = _request(app, "POST", "/query/data", json={"workspace_id": "native-test", "query_vector": [1.0], "record_types": ["entity"]}, raise_app_exceptions=False)

    assert response.status_code == 200
    assert called["threadpool"] is True
