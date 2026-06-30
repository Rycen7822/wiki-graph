from __future__ import annotations

import json
import urllib.request

import pytest

from llm_wiki_native.answer import NativeAnswerConfig, NativeAnswerGenerator
from llm_wiki_native.embedding import NativeEmbedding, NativeEmbeddingConfig


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

    def fake_urlopen(request, timeout):
        calls.append({"request": request, "timeout": timeout})
        return FakeHTTPResponse({"data": [{"embedding": [0.25, 0.75]}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = NativeEmbedding(
        NativeEmbeddingConfig(
            base_url="https://embedding.local/v1",
            model="embed-small",
            api_key="secret",
            timeout_seconds=12.5,
        )
    )

    assert provider.embed_query("alpha") == [0.25, 0.75]

    request = calls[0]["request"]
    body = json.loads(request.data.decode("utf-8"))
    assert calls[0]["timeout"] == 12.5
    assert request.full_url == "https://embedding.local/v1/embeddings"
    assert request.headers["Authorization"] == "Bearer secret"
    assert body == {"model": "embed-small", "input": "alpha"}


def test_native_embedding_config_from_env_requires_endpoint_model_and_key() -> None:
    with pytest.raises(ValueError, match="BASE_URL"):
        NativeEmbeddingConfig.from_env({})
    with pytest.raises(ValueError, match="MODEL"):
        NativeEmbeddingConfig.from_env({"LLM_WIKI_NATIVE_EMBEDDING_BASE_URL": "https://embedding.local/v1"})
    with pytest.raises(ValueError, match="API_KEY"):
        NativeEmbeddingConfig.from_env(
            {
                "LLM_WIKI_NATIVE_EMBEDDING_BASE_URL": "https://embedding.local/v1",
                "LLM_WIKI_NATIVE_EMBEDDING_MODEL": "embed-small",
            }
        )


def test_native_embedding_config_from_env_accepts_compatible_binding_names() -> None:
    config = NativeEmbeddingConfig.from_env(
        {
            "EMBEDDING_BINDING_HOST": "https://embedding.local/v1",
            "EMBEDDING_MODEL": "BAAI/bge-m3",
            "EMBEDDING_BINDING_API_KEY": "secret",
            "EMBEDDING_TIMEOUT": "77",
            "EMBEDDING_DIM": "1024",
        }
    )

    assert config.base_url == "https://embedding.local/v1"
    assert config.model == "BAAI/bge-m3"
    assert config.api_key == "secret"
    assert config.timeout_seconds == 77.0
    assert config.embedding_dim == 1024
    assert "secret" not in repr(config)


def test_native_embedding_config_prefers_native_names_over_compatible_names() -> None:
    config = NativeEmbeddingConfig.from_env(
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

    assert config.base_url == "https://native.local/v1"
    assert config.model == "native-model"
    assert config.api_key == "native-secret"
    assert config.timeout_seconds == 12.0
    assert config.embedding_dim == 3
    assert "native-secret" not in repr(config)


def test_native_embedding_validates_configured_dimension(monkeypatch) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: FakeHTTPResponse({"data": [{"embedding": [0.25, 0.75]}]}),
    )
    provider = NativeEmbedding(
        NativeEmbeddingConfig(
            base_url="https://embedding.local/v1",
            model="embed-small",
            api_key="secret",
            embedding_dim=3,
        )
    )

    with pytest.raises(ValueError, match="dimension"):
        provider.embed_query("alpha")


def test_native_embedding_rejects_bad_embedding_response(monkeypatch) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: FakeHTTPResponse({"data": [{"embedding": [float("nan")]}]}),
    )
    provider = NativeEmbedding(
        NativeEmbeddingConfig(
            base_url="https://embedding.local/v1",
            model="embed-small",
            api_key="secret",
        )
    )

    with pytest.raises(ValueError, match="finite"):
        provider.embed_query("alpha")


def test_native_answer_generator_posts_openai_compatible_chat_request(monkeypatch) -> None:
    calls = []

    def fake_urlopen(request, timeout):
        calls.append({"request": request, "timeout": timeout})
        return FakeHTTPResponse({"choices": [{"message": {"content": "Alpha answer"}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
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


def test_native_answer_config_from_env_requires_endpoint_model_and_key() -> None:
    with pytest.raises(ValueError, match="BASE_URL"):
        NativeAnswerConfig.from_env({})
    with pytest.raises(ValueError, match="MODEL"):
        NativeAnswerConfig.from_env({"LLM_WIKI_NATIVE_ANSWER_BASE_URL": "https://chat.local/v1"})
    with pytest.raises(ValueError, match="API_KEY"):
        NativeAnswerConfig.from_env(
            {
                "LLM_WIKI_NATIVE_ANSWER_BASE_URL": "https://chat.local/v1",
                "LLM_WIKI_NATIVE_ANSWER_MODEL": "chat-small",
            }
        )


def test_native_answer_generator_rejects_bad_chat_response(monkeypatch) -> None:
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout: FakeHTTPResponse({"choices": []}),
    )
    generator = NativeAnswerGenerator(
        NativeAnswerConfig(
            base_url="https://chat.local/v1",
            model="chat-small",
            api_key="secret",
        )
    )

    with pytest.raises(ValueError, match="message.content"):
        generator.generate("alpha?", {"context_blocks": [], "source_paths": []}, mode="mix")
