from __future__ import annotations

import pytest

from llm_wiki_native.api.server import _query_kwargs


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [0.25, 0.75]


def test_query_kwargs_embeds_query_when_vector_is_absent() -> None:
    provider = FakeEmbeddingProvider()

    kwargs = _query_kwargs(
        {"workspace_id": "native-test", "query": "alpha", "mode": "mix"},
        embedding_provider=provider,
    )

    assert kwargs["query_vector"] == [0.25, 0.75]
    assert provider.queries == ["alpha"]


def test_query_kwargs_uses_default_workspace_id_when_payload_omits_it() -> None:
    provider = FakeEmbeddingProvider()

    kwargs = _query_kwargs(
        {"query": "alpha", "mode": "mix"},
        embedding_provider=provider,
        default_workspace_id="native-default",
    )

    assert kwargs["workspace_id"] == "native-default"
    assert kwargs["query_vector"] == [0.25, 0.75]


def test_query_kwargs_preserves_explicit_query_vector_without_embedding() -> None:
    provider = FakeEmbeddingProvider()

    kwargs = _query_kwargs(
        {"workspace_id": "native-test", "query": "alpha", "query_vector": [1, 0]},
        embedding_provider=provider,
    )

    assert kwargs["query_vector"] == [1.0, 0.0]
    assert provider.queries == []


def test_query_kwargs_allows_bypass_without_vector_or_embedding() -> None:
    kwargs = _query_kwargs({"workspace_id": "native-test", "query": "alpha", "mode": "bypass"})

    assert kwargs["query_vector"] == []
    assert kwargs["mode"] == "bypass"


def test_query_kwargs_preserves_section_kind_filter() -> None:
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
