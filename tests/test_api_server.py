from typing import Any

import pytest

from llm_wiki_native.api.server import _answer_response_payload, _query_kwargs, create_app
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
        "response_profile": "compact",
    }

    data_response = _request(app, "POST", "/query/data", json=payload)

    assert data_response.status_code == 200
    body = data_response.json()
    assert body["context_blocks"][0]["source_path"] == "alpha.md"
    assert "neighbors" not in body["context_blocks"][0]
    assert body["coverage_plan"]["must_read"][0]["source_path"] == "alpha.md"
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


def test_native_api_requires_bearer_token_when_configured(tmp_path, monkeypatch) -> None:
    patch_direct_threadpool(monkeypatch)
    monkeypatch.setenv("LLM_WIKI_NATIVE_API_KEY", "secret-token")
    app = _app(tmp_path)
    payload = {"workspace_id": "native-test", "query_vector": [1.0, 0.0], "record_types": ["entity"], "top_k": 1}

    assert _request(app, "POST", "/query/data", json=payload, raise_app_exceptions=False).status_code == 401
    assert _request(app, "POST", "/query/data", json=payload, headers={"Authorization": "Bearer wrong"}, raise_app_exceptions=False).status_code == 401
    assert _request(app, "POST", "/query/data", json=payload, headers={"Authorization": "Bearer secret-token"}).status_code == 200


def test_native_api_read_span_returns_current_source_text(tmp_path, monkeypatch) -> None:
    patch_direct_threadpool(monkeypatch)
    wiki_root = tmp_path / "wiki"
    (wiki_root / "notes").mkdir(parents=True)
    (wiki_root / "notes" / "alpha.md").write_text("# Alpha\n\nAPI evidence line\n", encoding="utf-8")
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    db.mark_audited("native-test", {"chunks": 0, "entities": 0, "relationships": 0, "sections": 0})
    db.put_lexical_span(
        "native-test",
        span_id="span:api",
        source_path="notes/alpha.md",
        source_id="wiki:alpha",
        source_role="wiki",
        span_kind="doc.paragraph",
        heading_path=["Alpha"],
        start_line=3,
        end_line=3,
        text="API evidence line",
        metadata={},
    )
    app = create_app(NativeQueryEngine(db, zvec_workspace=type("Z", (), {"query_mix": lambda *a, **k: [], "query_vector": lambda *a, **k: []})(), source_root=wiki_root), default_workspace_id="native-test")

    response = _request(app, "POST", "/read/span", json={"span_id": "span:api"})

    assert response.status_code == 200
    body = response.json()
    assert body["workspace_id"] == "native-test"
    assert body["span_id"] == "span:api"
    assert body["source_status"] == "current"
    assert body["text"] == "API evidence line"


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


class FakeAnswerGenerator:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, query: str, context: dict, *, mode: str) -> dict:
        self.calls.append({"query": query, "context": context, "mode": mode})
        return {"response": f"answer for {query}", "references": context["source_paths"]}


def _context() -> dict:
    return {
        "context_blocks": [{"record_id": "doc:a", "text": "Alpha context"}],
        "source_paths": ["alpha.md"],
        "trace": {"mode": "mix", "context_block_count": 1},
    }


def test_answer_response_payload_uses_answer_generator_context() -> None:
    answers = FakeAnswerGenerator()

    payload = _answer_response_payload(
        query="alpha",
        mode="mix",
        context=_context(),
        answer_generator=answers,
    )

    assert payload["query"] == "alpha"
    assert payload["mode"] == "mix"
    assert payload["response"] == "answer for alpha"
    assert payload["references"] == ["alpha.md"]
    assert payload["data"]["context_blocks"][0]["text"] == "Alpha context"
    assert payload["trace"]["context_block_count"] == 1
    assert answers.calls[0]["mode"] == "mix"


def test_answer_response_payload_fails_closed_without_answer_generator() -> None:
    with pytest.raises(NotImplementedError, match="answer generator"):
        _answer_response_payload(
            query="alpha",
            mode="mix",
            context=_context(),
            answer_generator=None,
        )


def test_query_route_bypass_does_not_call_answer_generator() -> None:
    class FakeBypassEngine:
        default_workspace_id = "native-test"

        def query(self, **kwargs) -> dict:
            assert kwargs["mode"] == "bypass"
            assert kwargs["query_vector"] == []
            return {
                "hits": [],
                "trace": {
                    "query": kwargs["query"],
                    "mode": "bypass",
                    "top_k": kwargs["top_k"],
                    "record_types": list(kwargs["record_types"]),
                    "section_kind": kwargs["section_kind"],
                    "vector_hit_count": 0,
                    "retrieval_backend": "bypass",
                },
            }

    answers = FakeAnswerGenerator()
    app = create_app(FakeBypassEngine(), answer_generator=answers, default_workspace_id="native-test")

    response = _request(app, "POST", "/query", json={"query": "alpha", "mode": "bypass"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "bypass"
    assert payload["response"] == ""
    assert payload["references"] == []
    assert payload["data"]["context_blocks"] == []
    assert payload["trace"]["retrieval_backend"] == "bypass"
    assert answers.calls == []


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [0.25, 0.75]


def test_query_kwargs_vector_workspace_and_filter_contracts() -> None:
    provider = FakeEmbeddingProvider()
    kwargs = _query_kwargs(
        {"workspace_id": "native-test", "query": "alpha", "mode": "mix"},
        embedding_provider=provider,
    )
    assert kwargs["query_vector"] == [0.25, 0.75]
    assert provider.queries == ["alpha"]

    provider = FakeEmbeddingProvider()
    kwargs = _query_kwargs(
        {"query": "alpha", "mode": "mix"},
        embedding_provider=provider,
        default_workspace_id="native-default",
    )
    assert kwargs["workspace_id"] == "native-default"
    assert kwargs["query_vector"] == [0.25, 0.75]

    provider = FakeEmbeddingProvider()
    kwargs = _query_kwargs(
        {"workspace_id": "native-test", "query": "alpha", "query_vector": [1, 0]},
        embedding_provider=provider,
    )
    assert provider.queries == []
    assert kwargs["query_vector"] == [1.0, 0.0]

    kwargs = _query_kwargs({"workspace_id": "native-test", "query": "alpha", "mode": "bypass"})
    assert kwargs["query_vector"] == []
    assert kwargs["mode"] == "bypass"

    kwargs = _query_kwargs(
        {
            "workspace_id": "native-test",
            "query": "alpha",
            "query_vector": [1.0],
            "section_kind": "methodology",
        }
    )
    assert kwargs["section_kind"] == "methodology"


def test_query_kwargs_requires_vector_or_embedding_provider_for_retrieval() -> None:
    with pytest.raises(ValueError, match="embedding provider"):
        _query_kwargs({"workspace_id": "native-test", "query": "alpha", "mode": "mix"})
