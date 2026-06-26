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
