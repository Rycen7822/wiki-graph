from starlette.testclient import TestClient

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


def _client(tmp_path, *, raise_server_exceptions: bool = True) -> TestClient:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    db.put_record(_record("native-test", "doc:a"))
    db.put_vector("native-test", "entity", "doc:a", "doc:a:vector", [1.0, 0.0])
    db.mark_audited("native-test", {"chunks": 0, "entities": 1, "relationships": 0, "sections": 0}, require_vectors=True)
    return TestClient(create_app(NativeQueryEngine(db)), raise_server_exceptions=raise_server_exceptions)


def test_shadow_api_health_reports_native_port_and_ready(tmp_path) -> None:
    client = _client(tmp_path)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "llm-wiki-native", "default_port": 9622}


def test_shadow_api_query_data_and_trace_contract(tmp_path) -> None:
    client = _client(tmp_path)
    payload = {
        "workspace_id": "native-test",
        "query": "alpha",
        "query_vector": [1.0, 0.0],
        "mode": "mix",
        "top_k": 1,
        "record_types": ["entity"],
    }

    data_response = client.post("/query/data", json=payload)
    trace_response = client.post("/native/query/trace", json=payload)

    assert data_response.status_code == 200
    assert data_response.json()["context_blocks"][0]["source_path"] == "alpha.md"
    assert trace_response.status_code == 200
    assert trace_response.json()["trace"]["mode"] == "mix"


def test_shadow_api_returns_structured_400_for_validation_errors(tmp_path) -> None:
    client = _client(tmp_path, raise_server_exceptions=False)
    payload = {
        "workspace_id": "native-test",
        "query": "alpha",
        "query_vector": [1.0, 0.0],
        "mode": "mix",
        "record_types": ["unknown"],
    }

    response = client.post("/query/data", json=payload)

    assert response.status_code == 400
    assert "record_type" in response.json()["error"]


def test_shadow_api_returns_structured_501_for_unimplemented_supported_mode(tmp_path) -> None:
    client = _client(tmp_path, raise_server_exceptions=False)
    payload = {"workspace_id": "native-test", "query_vector": [1.0, 0.0], "record_types": ["entity"], "mode": "local"}

    response = client.post("/native/query/trace", json=payload)

    assert response.status_code == 501
    assert "not implemented" in response.json()["error"]


def test_shadow_api_requires_bearer_token_when_configured(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_WIKI_NATIVE_API_KEY", "secret-token")
    client = _client(tmp_path, raise_server_exceptions=False)
    payload = {"workspace_id": "native-test", "query_vector": [1.0, 0.0], "record_types": ["entity"], "top_k": 1}

    assert client.post("/native/query/trace", json=payload).status_code == 401
    assert client.post("/native/query/trace", json=payload, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/native/query/trace", json=payload, headers={"Authorization": "Bearer secret-token"}).status_code == 200


def test_shadow_api_validates_vectors_and_clamps_request_limits(monkeypatch) -> None:
    calls = []

    class FakeEngine:
        def query(self, **kwargs):
            calls.append(kwargs)
            return {"hits": [], "trace": {"mode": kwargs["mode"], "top_k": kwargs["top_k"], "neighbor_limit": kwargs["neighbor_limit"], "record_types": list(kwargs["record_types"])}}

    client = TestClient(create_app(FakeEngine()), raise_server_exceptions=False)

    invalid = client.post("/native/query/trace", json={"workspace_id": "native-test", "query_vector": ["nan"]})
    assert invalid.status_code == 400
    assert "finite" in invalid.json()["error"]

    response = client.post(
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
    client = _client(tmp_path, raise_server_exceptions=False)

    response = client.post("/native/query/trace", content=b"{" + b" " * 1_000_001 + b"}", headers={"content-type": "application/json"})

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
    client = TestClient(create_app(FakeEngine()), raise_server_exceptions=False)

    response = client.post("/native/query/trace", json={"workspace_id": "native-test", "query_vector": [1.0], "record_types": ["entity"]})

    assert response.status_code == 200
    assert called["threadpool"] is True
