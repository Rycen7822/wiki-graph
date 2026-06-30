from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from llm_wiki_native.storage.sqlite_workspace import NativeRecord


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_vector_cache(path: Path, vectors: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE vector_cache(vector_hash TEXT PRIMARY KEY, embedding_dim INTEGER NOT NULL, vector_blob BLOB NOT NULL)")
        for vector_hash, vector in vectors.items():
            arr = np.asarray(vector, dtype=np.float32)
            conn.execute(
                "INSERT INTO vector_cache(vector_hash, embedding_dim, vector_blob) VALUES(?, ?, ?)",
                (vector_hash, int(arr.size), arr.tobytes()),
            )


def native_record(
    workspace_id: str,
    record_type: str = "chunk",
    record_id: str = "chunk-a",
    text: str = "Doc A",
    *,
    content_hash: str | None = None,
    metadata_hash: str | None = None,
    vector_hash: str | None = None,
    source_path: str | None = None,
    source_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> NativeRecord:
    return NativeRecord(
        workspace_id=workspace_id,
        record_type=record_type,
        record_id=record_id,
        vector_text=text,
        content_hash=content_hash or f"{record_id}:content",
        metadata_hash=metadata_hash or f"{record_id}:metadata",
        vector_hash=vector_hash or f"{record_id}:vector",
        source_path=source_path or f"{record_id}.md",
        source_id=source_id or record_id,
        payload=payload if payload is not None else {"title": text},
    )


def request_asgi(app: Any, method: str, path: str, *, raise_app_exceptions: bool = True, **kwargs: Any) -> Any:
    import asyncio

    import httpx

    async def request_once() -> httpx.Response:
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(request_once())


def patch_direct_threadpool(monkeypatch: Any, target: str = "llm_wiki_native.api.server.run_in_threadpool") -> None:
    async def direct_threadpool(func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    monkeypatch.setattr(target, direct_threadpool)
