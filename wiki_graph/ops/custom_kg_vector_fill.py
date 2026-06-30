#!/usr/bin/env python3
"""Native-safe custom KG manifest vector fill helpers."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

EMBEDDING_PROFILES = {
    "conservative": {"EMBEDDING_FUNC_MAX_ASYNC": "1", "EMBEDDING_BATCH_NUM": "10", "MAX_PARALLEL_INSERT": "1"},
    "balanced-medium": {"EMBEDDING_FUNC_MAX_ASYNC": "2", "EMBEDDING_BATCH_NUM": "20", "MAX_PARALLEL_INSERT": "1"},
    "operator-fast": {"EMBEDDING_FUNC_MAX_ASYNC": "4", "EMBEDDING_BATCH_NUM": "32", "MAX_PARALLEL_INSERT": "2"},
}
DEFAULT_EMBEDDING_PROFILE = "conservative"
_SECRET_KEY_TOKENS = ("key", "token", "secret", "password")
_EMBEDDING_ENV_REPORT_KEYS = (
    "EMBEDDING_BINDING",
    "EMBEDDING_BINDING_HOST",
    "OPENAI_BASE_URL",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "EMBEDDING_PARAMS_VERSION",
    "EMBEDDING_TIMEOUT",
    "EMBEDDING_BINDING_API_KEY",
    "OPENAI_API_KEY",
)


def _load_env_file_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def embedding_runtime_env(workdir: Path) -> dict[str, str]:
    values = _load_env_file_values(workdir / ".env")
    for key, value in os.environ.items():
        if value:
            values[key] = value
    return values


def embedding_env_report(env: dict[str, str]) -> dict[str, str]:
    report: dict[str, str] = {}
    for key in _EMBEDDING_ENV_REPORT_KEYS:
        value = env.get(key, "")
        if any(token in key.lower() for token in _SECRET_KEY_TOKENS):
            report[key] = "[REDACTED]" if value else ""
        else:
            report[key] = value
    return report


def env_int(name: str, default: int, env: dict[str, str] | None = None) -> int:
    values = os.environ if env is None else env
    try:
        return int(values.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def embedding_profile_env(profile: str | None = None) -> dict[str, str]:
    name = profile or DEFAULT_EMBEDDING_PROFILE
    if name not in EMBEDDING_PROFILES:
        known = ", ".join(sorted(EMBEDDING_PROFILES))
        raise ValueError(f"unknown embedding profile {name!r}; expected one of: {known}")
    return dict(EMBEDDING_PROFILES[name])


def embedding_profile_report(profile: str | None = None) -> dict[str, Any]:
    name = profile or DEFAULT_EMBEDDING_PROFILE
    env = embedding_profile_env(name)
    return {
        "name": name,
        "env": env,
        "batch_size": int(env["EMBEDDING_BATCH_NUM"]),
        "concurrency": {
            "embedding_func_max_async": int(env["EMBEDDING_FUNC_MAX_ASYNC"]),
            "max_parallel_insert": int(env["MAX_PARALLEL_INSERT"]),
        },
    }


def embed_texts_openai_compatible(
    texts: list[str],
    *,
    workdir: Path,
    embedding_model: str,
    embedding_dim: int,
    timeout: int | None = None,
) -> list[list[float]]:
    """Embed texts through the configured OpenAI-compatible embedding endpoint."""

    env = embedding_runtime_env(workdir)
    host = (env.get("EMBEDDING_BINDING_HOST") or env.get("OPENAI_BASE_URL") or "").rstrip("/")
    api_key = env.get("EMBEDDING_BINDING_API_KEY") or env.get("OPENAI_API_KEY") or ""
    if not host:
        raise RuntimeError("EMBEDDING_BINDING_HOST or OPENAI_BASE_URL is required to fill missing vectors")
    if not api_key:
        raise RuntimeError("EMBEDDING_BINDING_API_KEY or OPENAI_API_KEY is required to fill missing vectors")
    url = f"{host}/embeddings"
    request = urllib.request.Request(
        url,
        data=json.dumps({"model": embedding_model, "input": texts}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout or env_int("EMBEDDING_TIMEOUT", 120, env)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    data = sorted(payload.get("data") or [], key=lambda row: row.get("index", 0))
    vectors = [row.get("embedding") for row in data]
    if len(vectors) != len(texts) or not all(isinstance(vector, list) for vector in vectors):
        raise RuntimeError(f"embedding response count mismatch: expected {len(texts)}, got {len(vectors)}")
    for index, vector in enumerate(vectors):
        if len(vector) != embedding_dim:
            raise RuntimeError(f"embedding response dimension mismatch at {index}: expected {embedding_dim}, got {len(vector)}")
    return vectors  # type: ignore[return-value]


def _missing_manifest_records(manifest: dict[str, Any], vector_report: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    records: list[tuple[str, str, dict[str, Any]]] = []
    missing = vector_report.get("missing", {}) if isinstance(vector_report, dict) else {}
    for collection in ("chunks", "entities", "relationships"):
        collection_records = manifest.get(collection, {})
        if not isinstance(collection_records, dict):
            continue
        for key in missing.get(collection, []) if isinstance(missing, dict) else []:
            record = collection_records.get(key)
            if isinstance(record, dict):
                records.append((collection, str(key), record))
    return records


def fill_missing_manifest_vectors(
    manifest: dict[str, Any],
    vector_report: dict[str, Any],
    cache: Any,
    *,
    workdir: Path,
    embed_texts_func: Any | None = None,
    embedding_profile: str | None = None,
) -> dict[str, Any]:
    """Embed only manifest records that are unresolved after cache/storage seeding."""

    profile_report = embedding_profile_report(embedding_profile)
    runtime_env = embedding_runtime_env(workdir)
    env_report = embedding_env_report(runtime_env)
    batch_size = max(1, int(profile_report["batch_size"]))
    missing_records = _missing_manifest_records(manifest, vector_report)
    if not missing_records:
        return {
            "summary": {"total": 0, "embedded": 0},
            "by_collection": {},
            "embedding_profile": profile_report["name"],
            "batch_size": batch_size,
            "concurrency": profile_report["concurrency"],
            "embedding_env": env_report,
            "total_batches": 0,
            "failed_batches": 0,
            "provider_retries": 0,
            "elapsed_by_collection_s": {},
        }
    metadata = manifest.get("metadata", {}) if isinstance(manifest.get("metadata"), dict) else {}
    embedding_model = str(metadata.get("embedding_model") or runtime_env.get("EMBEDDING_MODEL") or "text-embedding-3-small")
    embedding_dim = int(metadata.get("embedding_dim") or env_int("EMBEDDING_DIM", 1536, runtime_env))
    embedding_params_version = str(metadata.get("embedding_params_version") or runtime_env.get("EMBEDDING_PARAMS_VERSION", "v1"))
    embedder = embed_texts_func or embed_texts_openai_compatible
    cache_records: list[dict[str, Any]] = []
    by_collection: dict[str, int] = {"chunks": 0, "entities": 0, "relationships": 0}
    elapsed_by_collection: dict[str, float] = {"chunks": 0.0, "entities": 0.0, "relationships": 0.0}
    total_batches = 0
    failed_batches = 0
    provider_retries = 0
    for offset in range(0, len(missing_records), batch_size):
        batch = missing_records[offset : offset + batch_size]
        texts = [str(record.get("content") or "") for _collection, _key, record in batch]
        if any(text == "" for text in texts):
            empty_key = batch[texts.index("")][1]
            raise RuntimeError(f"cannot fill missing vector for empty content record: {empty_key}")
        batch_started = time.perf_counter()
        total_batches += 1
        try:
            vectors = embedder(texts, workdir=workdir, embedding_model=embedding_model, embedding_dim=embedding_dim)
        except Exception:
            failed_batches += 1
            raise
        batch_elapsed = round(time.perf_counter() - batch_started, 6)
        for collection in {collection for collection, _key, _record in batch}:
            elapsed_by_collection[collection] = round(elapsed_by_collection.get(collection, 0.0) + batch_elapsed, 6)
        if len(vectors) != len(batch):
            raise RuntimeError(f"embedding fill response count mismatch: expected {len(batch)}, got {len(vectors)}")
        for (collection, key, record), vector in zip(batch, vectors):
            if len(vector) != embedding_dim:
                raise RuntimeError(f"embedding fill dimension mismatch for {collection}:{key}: expected {embedding_dim}, got {len(vector)}")
            required = [
                record.get("vector_hash"),
                record.get("record_type"),
                record.get("record_id"),
                record.get("embedding_model") or embedding_model,
                record.get("embedding_dim") or embedding_dim,
                record.get("embedding_params_version") or embedding_params_version,
            ]
            if any(value in (None, "") for value in required):
                raise RuntimeError(f"cannot fill missing vector with incomplete manifest contract: {collection}:{key}")
            cache_records.append(
                {
                    "vector_hash": str(record["vector_hash"]),
                    "record_type": str(record["record_type"]),
                    "record_id": str(record["record_id"]),
                    "embedding_model": str(record.get("embedding_model") or embedding_model),
                    "embedding_dim": int(record.get("embedding_dim") or embedding_dim),
                    "embedding_params_version": str(record.get("embedding_params_version") or embedding_params_version),
                    "vector": [float(value) for value in vector],
                }
            )
            by_collection[collection] = by_collection.get(collection, 0) + 1
    cache.put_many(cache_records)
    return {
        "summary": {"total": len(cache_records), "embedded": len(cache_records)},
        "by_collection": {key: value for key, value in by_collection.items() if value},
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "embedding_params_version": embedding_params_version,
        "embedding_profile": profile_report["name"],
        "embedding_env": env_report,
        "batch_size": batch_size,
        "concurrency": profile_report["concurrency"],
        "total_batches": total_batches,
        "failed_batches": failed_batches,
        "provider_retries": provider_retries,
        "elapsed_by_collection_s": {key: value for key, value in elapsed_by_collection.items() if value or by_collection.get(key)},
    }
