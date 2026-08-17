from __future__ import annotations

import json
import urllib.request

import pytest

from llm_wiki_native.answer import NativeAnswerConfig, NativeAnswerGenerator
from llm_wiki_native.embedding import NativeEmbedding, NativeEmbeddingConfig


def _embed_cfg(**over) -> NativeEmbeddingConfig:
    values = {"base_url": "https://embedding.local/v1", "model": "embed-small", "api_key": "secret"}
    values.update(over)
    return NativeEmbeddingConfig(**values)


def _install_urlopen(monkeypatch, payload: dict, calls: list | None = None):
    def fake(request, timeout):
        if calls is not None:
            calls.append({"request": request, "timeout": timeout})
        return FakeHTTPResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake)


class FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_native_embedding_posts_openai_compatible_request(monkeypatch) -> None:
    calls = []

    _install_urlopen(monkeypatch, {"data": [{"embedding": [0.25, 0.75]}]}, calls)
    provider = NativeEmbedding(_embed_cfg(timeout_seconds=12.5))

    assert provider.embed_query("alpha") == [0.25, 0.75]

    request = calls[0]["request"]
    body = json.loads(request.data.decode("utf-8"))
    assert calls[0]["timeout"] == 12.5
    assert request.full_url == "https://embedding.local/v1/embeddings"
    assert request.headers["Authorization"] == "Bearer secret"
    assert body == {"model": "embed-small", "input": "alpha"}


def test_native_embedding_cache_modes(monkeypatch) -> None:
    for cache_size, mutate_first, expected_calls in [(4, True, 1), (0, False, 2)]:
        calls = []

        _install_urlopen(monkeypatch, {"data": [{"embedding": [0.25, 0.75]}]}, calls)
        provider = NativeEmbedding(_embed_cfg(cache_size=cache_size))

        first = provider.embed_query("alpha")
        if mutate_first:
            first[0] = 99.0

        assert provider.embed_query("alpha") == [0.25, 0.75]
        assert len(calls) == expected_calls


@pytest.mark.parametrize(
    "config_cls,prefix,base_url,model",
    [
        (NativeEmbeddingConfig, "LLM_WIKI_NATIVE_EMBEDDING", "https://embedding.local/v1", "embed-small"),
        (NativeAnswerConfig, "LLM_WIKI_NATIVE_ANSWER", "https://chat.local/v1", "chat-small"),
    ],
    ids=["embedding", "answer"],
)
def test_native_config_from_env_requires_endpoint_model_and_key(config_cls, prefix: str, base_url: str, model: str) -> None:
    with pytest.raises(ValueError, match="BASE_URL"):
        config_cls.from_env({})
    with pytest.raises(ValueError, match="MODEL"):
        config_cls.from_env({f"{prefix}_BASE_URL": base_url})
    with pytest.raises(ValueError, match="API_KEY"):
        config_cls.from_env({f"{prefix}_BASE_URL": base_url, f"{prefix}_MODEL": model})


def test_native_embedding_config_from_env_aliases_and_native_precedence() -> None:
    compatible = NativeEmbeddingConfig.from_env(
        {
            "EMBEDDING_BINDING_HOST": "https://embedding.local/v1",
            "EMBEDDING_MODEL": "BAAI/bge-m3",
            "EMBEDDING_BINDING_API_KEY": "secret",
            "EMBEDDING_TIMEOUT": "77",
            "EMBEDDING_DIM": "1024",
            "EMBEDDING_CACHE_SIZE": "17",
        }
    )
    assert compatible.base_url == "https://embedding.local/v1"
    assert compatible.model == "BAAI/bge-m3"
    assert compatible.api_key == "secret"
    assert compatible.timeout_seconds == 77.0
    assert compatible.embedding_dim == 1024
    assert compatible.cache_size == 17
    assert "secret" not in repr(compatible)

    native = NativeEmbeddingConfig.from_env(
        {
            "LLM_WIKI_NATIVE_EMBEDDING_BASE_URL": "https://native.local/v1",
            "LLM_WIKI_NATIVE_EMBEDDING_MODEL": "native-model",
            "LLM_WIKI_NATIVE_EMBEDDING_API_KEY": "native-secret",
            "LLM_WIKI_NATIVE_EMBEDDING_TIMEOUT_SECONDS": "12",
            "LLM_WIKI_NATIVE_EMBEDDING_DIM": "3",
            "EMBEDDING_BINDING_HOST": "https://embedding.local/v1",
            "EMBEDDING_MODEL": "BAAI/bge-m3",
            "EMBEDDING_BINDING_API_KEY": "compatible-secret",
            "EMBEDDING_TIMEOUT": "77",
            "EMBEDDING_DIM": "1024",
        }
    )
    assert native.base_url == "https://native.local/v1"
    assert native.model == "native-model"
    assert native.api_key == "native-secret"
    assert native.timeout_seconds == 12.0
    assert native.embedding_dim == 3
    assert "native-secret" not in repr(native)


def test_native_embedding_validates_configured_dimension(monkeypatch) -> None:
    _install_urlopen(monkeypatch, {"data": [{"embedding": [0.25, 0.75]}]})
    provider = NativeEmbedding(_embed_cfg(embedding_dim=3))

    with pytest.raises(ValueError, match="dimension"):
        provider.embed_query("alpha")


def test_native_embedding_rejects_bad_embedding_response(monkeypatch) -> None:
    _install_urlopen(monkeypatch, {"data": [{"embedding": [float("nan")]}]})
    provider = NativeEmbedding(_embed_cfg())

    with pytest.raises(ValueError, match="finite"):
        provider.embed_query("alpha")


def test_native_answer_generator_posts_openai_compatible_chat_request(monkeypatch) -> None:
    calls = []

    _install_urlopen(monkeypatch, {"choices": [{"message": {"content": "Alpha answer"}}]}, calls)
    generator = NativeAnswerGenerator(
        NativeAnswerConfig(
            base_url="https://chat.local/v1",
            model="chat-small",
            api_key="secret",
            timeout_seconds=9.0,
        )
    )
    assert "secret" not in repr(generator.config)
    context = {
        "context_blocks": [{"source_path": "alpha.md", "text": "Alpha context"}],
        "source_paths": ["alpha.md"],
    }

    answer = generator.generate("alpha?", context, mode="mix")

    request = calls[0]["request"]
    body = json.loads(request.data.decode("utf-8"))
    assert calls[0]["timeout"] == 9.0
    assert request.full_url == "https://chat.local/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer secret"
    assert body["model"] == "chat-small"
    assert body["messages"][0]["role"] == "system"
    assert "Alpha context" in body["messages"][1]["content"]
    assert answer == {"response": "Alpha answer", "references": ["alpha.md"]}


def test_native_answer_generator_rejects_bad_chat_response(monkeypatch) -> None:
    _install_urlopen(monkeypatch, {"choices": []})
    generator = NativeAnswerGenerator(
        NativeAnswerConfig(
            base_url="https://chat.local/v1",
            model="chat-small",
            api_key="secret",
        )
    )

    with pytest.raises(ValueError, match="message.content"):
        generator.generate("alpha?", {"context_blocks": [], "source_paths": []}, mode="mix")
