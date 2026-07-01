"""Embedding provider interfaces for native query execution."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import json
import math
import os
from threading import Lock
from typing import Mapping, Protocol
import urllib.request


class EmbeddingProvider(Protocol):
    def embed_query(self, query: str) -> list[float]:
        """Return a dense query embedding for retrieval."""


@dataclass(frozen=True)
class NativeEmbeddingConfig:
    base_url: str
    model: str
    api_key: str = field(repr=False)
    timeout_seconds: float = 60.0
    embedding_dim: int | None = None
    cache_size: int = 512

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "NativeEmbeddingConfig":
        values = os.environ if env is None else env
        base_url = _first_env(
            values,
            "LLM_WIKI_NATIVE_EMBEDDING_BASE_URL",
            "EMBEDDING_BINDING_HOST",
            "OPENAI_BASE_URL",
        )
        model = _first_env(values, "LLM_WIKI_NATIVE_EMBEDDING_MODEL", "EMBEDDING_MODEL")
        api_key = _first_env(
            values,
            "LLM_WIKI_NATIVE_EMBEDDING_API_KEY",
            "EMBEDDING_BINDING_API_KEY",
            "OPENAI_API_KEY",
        )
        if not base_url:
            raise ValueError("LLM_WIKI_NATIVE_EMBEDDING_BASE_URL or EMBEDDING_BINDING_HOST or OPENAI_BASE_URL is required")
        if not model:
            raise ValueError("LLM_WIKI_NATIVE_EMBEDDING_MODEL or EMBEDDING_MODEL is required")
        if not api_key:
            raise ValueError("LLM_WIKI_NATIVE_EMBEDDING_API_KEY or EMBEDDING_BINDING_API_KEY or OPENAI_API_KEY is required")
        timeout = _first_env_float(values, ("LLM_WIKI_NATIVE_EMBEDDING_TIMEOUT_SECONDS", "EMBEDDING_TIMEOUT"), 60.0)
        embedding_dim = _first_env_int(values, ("LLM_WIKI_NATIVE_EMBEDDING_DIM", "EMBEDDING_DIM"))
        cache_size = _first_env_int(values, ("LLM_WIKI_NATIVE_EMBEDDING_CACHE_SIZE", "EMBEDDING_CACHE_SIZE"))
        if cache_size is None:
            cache_size = 512
        if cache_size < 0:
            raise ValueError("LLM_WIKI_NATIVE_EMBEDDING_CACHE_SIZE or EMBEDDING_CACHE_SIZE must be non-negative")
        return cls(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout,
            embedding_dim=embedding_dim,
            cache_size=cache_size,
        )


class NativeEmbedding:
    def __init__(self, config: NativeEmbeddingConfig) -> None:
        self.config = config
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_lock = Lock()

    def embed_query(self, query: str) -> list[float]:
        cached = self._cached(query)
        if cached is not None:
            return cached
        vector = self._fetch_embedding(query)
        self._store_cached(query, vector)
        return list(vector)

    def _fetch_embedding(self, query: str) -> list[float]:
        request = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}/embeddings",
            data=json.dumps({"model": self.config.model, "input": query}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return _embedding_from_payload(payload, expected_dim=self.config.embedding_dim)

    def _cached(self, query: str) -> list[float] | None:
        if self.config.cache_size <= 0:
            return None
        with self._cache_lock:
            vector = self._cache.get(query)
            if vector is None:
                return None
            self._cache.move_to_end(query)
            return list(vector)

    def _store_cached(self, query: str, vector: list[float]) -> None:
        if self.config.cache_size <= 0:
            return
        with self._cache_lock:
            self._cache[query] = list(vector)
            self._cache.move_to_end(query)
            while len(self._cache) > self.config.cache_size:
                self._cache.popitem(last=False)


def _first_env(values: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = values.get(name, "").strip()
        if value:
            return value
    return ""


def _first_env_float(values: Mapping[str, str], names: tuple[str, ...], default: float) -> float:
    for name in names:
        raw = values.get(name, "").strip()
        if raw:
            return float(raw)
    return default


def _first_env_int(values: Mapping[str, str], names: tuple[str, ...]) -> int | None:
    for name in names:
        raw = values.get(name, "").strip()
        if raw:
            return int(raw)
    return None


def _embedding_from_payload(payload: object, *, expected_dim: int | None = None) -> list[float]:
    if not isinstance(payload, dict):
        raise ValueError("embedding response must be a JSON object")
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise ValueError("embedding response must include data[0].embedding")
    first = data[0]
    if not isinstance(first, dict):
        raise ValueError("embedding response must include data[0].embedding")
    embedding = first.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise ValueError("embedding response must include a non-empty embedding list")
    vector: list[float] = []
    for value in embedding:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("embedding response must contain finite numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("embedding response must contain finite numbers")
        vector.append(number)
    if expected_dim is not None and len(vector) != expected_dim:
        raise ValueError(f"embedding response dimension mismatch: expected {expected_dim}, got {len(vector)}")
    return vector
