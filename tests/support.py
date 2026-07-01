from __future__ import annotations

import json
import sqlite3
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

from llm_wiki_native.storage.sqlite_workspace import NativeRecord


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sample_wiki(tmp_path: Path) -> Path:
    root = tmp_path / "llm-wiki"
    write(
        root / "index.md",
        "# LLM Wiki Index\n\n> Last updated: 2026-05-18 16:00 | Total pages: 2\n\n"
        "## Concepts\n\n- [[foo]] - Foo page.\n\n## Queries\n\n- [[bar]] - Bar page.\n",
    )
    write(
        root / "SCHEMA.md",
        "# Schema\n\nAllowed tags include agent, rag.\n",
    )
    write(
        root / "concepts/foo.md",
        "---\ntitle: Foo\ntype: concept\ntags: [agent]\nsources: [../raw/clip/2601/26010101_Foo-Paper.md]\nupdated: 2026-05-18 16:00\n---\n# Foo\n\nLinks to [[bar]].\n",
    )
    write(
        root / "queries/bar.md",
        "---\ntitle: Bar\ntype: query\ntags: [rag]\nupdated: 2026-05-18 16:00\n---\n# Bar\n",
    )
    write(
        root / "raw/clip/2601/26010101_Foo-Paper.md",
        "---\ntitle: Foo Paper\nsource: https://arxiv.org/abs/2601.0101\ndomain: paper\nupdated: 2026-05-18 16:00\ntags: [paper]\n---\n# Foo Paper\n\n## Methodology\n\nA direct method with enough structured detail to become a method atom during deterministic extraction.\n\n## 对未来研究的启发\n\n- Future work should connect memory repair with section-level evidence retrieval.\n\n## 可能的局限\n\n- The current benchmark may hide failures in long-horizon transfer.\n\n## 可继续追问的问题\n\n- Which unresolved interface lets agents ask for the right evidence section before planning?\n",
    )
    write(root / "_meta/raw-clip-map.md", "# Raw Clip Map\n\n- raw/clip/2601/26010101_Foo-Paper.md\n")
    write(root / "_meta/topic-map.md", "# Topic Map\n")
    return root


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


def install_fake_zvec(monkeypatch: Any) -> Any:
    fake = types.ModuleType("zvec")

    class DataType:
        INT32 = "INT32"
        STRING = "STRING"
        VECTOR_FP32 = "VECTOR_FP32"

    class MetricType:
        COSINE = "COSINE"

    class CollectionOption:
        def __init__(self, read_only: bool = False, enable_mmap: bool = True) -> None:
            self.read_only = read_only
            self.enable_mmap = enable_mmap

    class InvertIndexParam:
        pass

    class FtsIndexParam:
        def __init__(
            self,
            tokenizer_name: str = "standard",
            filters: list[str] | None = None,
            extra_params: str = "",
        ) -> None:
            self.tokenizer_name = tokenizer_name
            self.filters = filters
            self.extra_params = extra_params

    class HnswIndexParam:
        def __init__(
            self,
            metric_type=MetricType.COSINE,
            m: int = 50,
            ef_construction: int = 500,
        ) -> None:
            self.metric_type = metric_type
            self.m = m
            self.ef_construction = ef_construction

    class HnswQueryParam:
        def __init__(self, ef: int = 300) -> None:
            self.ef = ef

    class FtsQueryParam:
        def __init__(self, default_operator: str = "") -> None:
            self.default_operator = default_operator

    class Fts:
        def __init__(
            self,
            match_string: str | None = None,
            query_string: str | None = None,
        ) -> None:
            self.match_string = match_string
            self.query_string = query_string

    class Query:
        def __init__(self, field_name: str, vector=None, param=None, fts=None) -> None:
            self.field_name = field_name
            self.vector = vector
            self.param = param
            self.fts = fts

    class RrfReRanker:
        def __init__(self, rank_constant: int) -> None:
            self.rank_constant = rank_constant

    class FieldSchema:
        def __init__(
            self,
            name: str,
            data_type,
            nullable: bool = False,
            index_param=None,
        ) -> None:
            self.name = name
            self.data_type = data_type
            self.nullable = nullable
            self.index_param = index_param

    class VectorSchema:
        def __init__(
            self,
            name: str,
            data_type,
            dimension: int,
            index_param=None,
        ) -> None:
            self.name = name
            self.data_type = data_type
            self.dimension = dimension
            self.index_param = index_param

    class CollectionSchema:
        def __init__(self, name: str, fields=None, vectors=None) -> None:
            self.name = name
            self.fields = fields or []
            self.vectors = vectors or []

    fake.calls = []

    def create_and_open(*, path: str, schema, option):
        fake.calls.append(("create_and_open", path, schema, option))
        return {"kind": "created", "path": path}

    def zvec_open(path: str, option):
        fake.calls.append(("open", path, option))
        return {"kind": "opened", "path": path}

    class Doc:
        def __init__(self, id: str, score=None, vectors=None, fields=None) -> None:
            self.id = id
            self.score = score
            self.vectors = vectors or {}
            self.fields = fields or {}

    for name, value in {
        "CollectionOption": CollectionOption,
        "CollectionSchema": CollectionSchema,
        "DataType": DataType,
        "Doc": Doc,
        "FieldSchema": FieldSchema,
        "Fts": Fts,
        "FtsIndexParam": FtsIndexParam,
        "FtsQueryParam": FtsQueryParam,
        "HnswIndexParam": HnswIndexParam,
        "HnswQueryParam": HnswQueryParam,
        "InvertIndexParam": InvertIndexParam,
        "MetricType": MetricType,
        "Query": Query,
        "RrfReRanker": RrfReRanker,
        "VectorSchema": VectorSchema,
        "create_and_open": create_and_open,
        "open": zvec_open,
    }.items():
        setattr(fake, name, value)

    monkeypatch.setitem(sys.modules, "zvec", fake)
    monkeypatch.delitem(sys.modules, "llm_wiki_native.storage.zvec_workspace", raising=False)
    return fake
