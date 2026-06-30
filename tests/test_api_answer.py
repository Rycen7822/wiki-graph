from __future__ import annotations

import pytest

from llm_wiki_native.api.server import _answer_response_payload, create_app
from support import request_asgi as _request


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


def test_create_app_registers_query_route_when_answer_generator_is_supported() -> None:
    app = create_app(engine=object(), answer_generator=FakeAnswerGenerator())

    assert "/query" in {getattr(route, "path", None) for route in app.routes}


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
