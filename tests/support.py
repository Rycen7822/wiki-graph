from __future__ import annotations

import json
import sqlite3
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from llm_wiki_native.section_embedding_store import upsert_rows
from llm_wiki_native.storage.sqlite_workspace import NativeRecord, SQLiteWorkspace

WORKSPACE_TABLES = ("record", "vector", "edge", "lexical_span", "lexical_span_fts")


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


def write_kg_state(
    state: Path,
    *,
    manifest: dict[str, Any],
    section_similarity_edges: list[dict[str, Any]],
    raw_sections: list[dict[str, Any]],
    section_embeddings: list[dict[str, Any]] | None = None,
    vectors: dict[str, list[float]] | None = None,
) -> Path:
    write_json(state / "custom_kg_manifest.json", manifest)
    write_jsonl(state / "section_similarity_edges.jsonl", section_similarity_edges)
    write_jsonl(state / "raw_sections.jsonl", raw_sections)
    if section_embeddings is not None:
        write_jsonl(state / "section_embeddings.jsonl", section_embeddings)
        upsert_rows(state, section_embeddings)
    if vectors is not None:
        write_vector_cache(state / "vector_cache.sqlite", vectors)
    return state


def dump_workspace_tables(db_path: Path, *, mask_workspace_id: bool = False) -> dict[str, list[tuple]]:
    conn = sqlite3.connect(db_path)
    try:
        dump: dict[str, list[tuple]] = {}
        for table in WORKSPACE_TABLES:
            rows = conn.execute(f"SELECT * FROM {table}").fetchall()
            if mask_workspace_id:
                rows = [("", *tuple(row)[1:]) for row in rows]
            dump[table] = sorted(rows, key=lambda row: tuple(str(value) for value in row))
        return dump
    finally:
        conn.close()


def traced_sqlite_connect(db: Any, statements: list[str], *, select_only: bool = False):
    original_connect = db._connect

    def traced_connect():
        connection = original_connect()
        if select_only:
            connection.set_trace_callback(
                lambda statement: statements.append(statement)
                if statement.lstrip().upper().startswith("SELECT")
                else None
            )
        else:
            connection.set_trace_callback(statements.append)
        return connection

    return traced_connect


def structured_raw_fast_note(title: str, source: str) -> str:
    return f"""---
title: \"{title}\"
source: \"{source}\"
capture_route: \"test synthetic route\"
captured: \"2026-06-06 07:00 CST (+0800)\"
---

## 一句话总结

Synthetic take.

## 论文摘要（中文）

Synthetic abstract.

## Motivation

Synthetic motivation.

## Methodology

Formula evidence is integrated here: Eq. (1) defines $loss = x + y$ and the symbols are explained in the method narrative.

## 关键实验结果 / 作者结论

Figure 1 and Table 1 are integrated beside the result claim and record the observed trend.

## 对未来研究的启发

Future work can reuse the verification harness.

## 可能的局限

The tiny fixture is a synthetic limitation, not a real paper.

## 可继续追问的问题

Which wrapper gate catches failed verification before mark-pending?
"""


def cutover_context(tmp_path: Path, *, pending: bool = False) -> SimpleNamespace:
    from ops import batch_native_refresh

    state_dir = tmp_path / "wikigraph" / "state"
    root = tmp_path / "wiki"
    workspace_root = state_dir / "native_zvec" / "workspaces"
    watched_dir = tmp_path / "watched"
    calls: list[tuple] = []
    if pending:
        watched_dir.mkdir(parents=True)
        batch_native_refresh.mark_pending(state_dir, root, reason="manual-smoke")
    return SimpleNamespace(
        state_dir=state_dir,
        root=root,
        workspace_root=workspace_root,
        watched_dir=watched_dir,
        calls=calls,
    )


def fake_cutover_hooks(
    calls: list[tuple],
    *,
    smoke_ok: bool = True,
    smoke_raises: Exception | None = None,
    mutate_watched_path: Path | None = None,
    write_prepared: bool = False,
    record_pending: bool = False,
):
    def build_workspace(**kwargs):
        if write_prepared:
            calls.append(("build", kwargs["workspace_id"], kwargs.get("fill_missing_vectors")))
            prepared_path = kwargs["workspace_root"].parent / "prepared_workspace.json"
            prepared_path.parent.mkdir(parents=True, exist_ok=True)
            prepared_path.write_text(
                json.dumps({"schema_version": 1, "workspace_id": kwargs["workspace_id"], "status": "prepared"}),
                encoding="utf-8",
            )
            return {"ok": True, "prepared_workspace": str(prepared_path), "workspace_id": kwargs["workspace_id"]}
        calls.append(("build", kwargs["workspace_id"]))
        if mutate_watched_path is not None:
            mutate_watched_path.write_text('{"stable":false}', encoding="utf-8")
        return {"ok": True, "workspace_id": kwargs["workspace_id"]}

    def finalize_workspace(*, state_dir, reason):
        calls.append(("finalize", reason))
        return {"schema_version": 1, "workspace_id": "candidate", "status": "active"}

    def restart_service(*, state_dir):
        calls.append(("restart", str(state_dir)))
        return {"service": "llm-wiki-native", "status": "ok"}

    def query_smoke(*, state_dir, active):
        if record_pending:
            from ops import batch_native_refresh

            calls.append(
                (
                    "smoke",
                    str(state_dir),
                    active["workspace_id"],
                    batch_native_refresh.pending_ledger_path(state_dir).exists(),
                )
            )
        else:
            calls.append(("smoke", str(state_dir), active["workspace_id"]))
        if smoke_raises is not None:
            raise smoke_raises
        return {"ok": smoke_ok, "url": "http://127.0.0.1:9621/query/data"}

    return build_workspace, finalize_workspace, restart_service, query_smoke


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


def put_span(
    db: Any,
    *,
    span_id: str,
    text: str,
    span_kind: str,
    source_role: str,
    source_path: str,
    workspace_id: str = "native-test",
    source_id: str | None = None,
    heading_path: list[str] | None = None,
    start_line: int = 1,
    end_line: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    db.put_lexical_span(
        workspace_id,
        span_id=span_id,
        source_path=source_path,
        source_id=source_id if source_id is not None else f"{source_role}:{span_id}",
        source_role=source_role,
        span_kind=span_kind,
        heading_path=["Results"] if heading_path is None else heading_path,
        start_line=start_line,
        end_line=start_line if end_line is None else end_line,
        text=text,
        metadata={} if metadata is None else metadata,
    )


def materialize_argv(
    root: Path,
    state: Path,
    workspace_root: Path,
    *extra: str,
    workspace_id: str = "native-test",
    command: str = "build",
) -> list[str]:
    return [
        command,
        "--root",
        str(root),
        "--state-dir",
        str(state),
        "--workspace-root",
        str(workspace_root),
        "--workspace-id",
        workspace_id,
        *extra,
    ]


def sample_kg_manifest(*, chunk_hash: str = "chunk-hash") -> dict[str, Any]:
    meta = {"embedding_model": "test-embedding", "embedding_dim": 2, "embedding_params_version": "v1"}

    def rec(record_type: str, record_id: str, content: str, **extra: Any) -> dict[str, Any]:
        return {"record_type": record_type, "record_id": record_id, "content": content, **meta, **extra}

    return {
        "metadata": meta,
        "chunks": {
            "chunk-a": rec(
                "chunk", "chunk-a", "Alpha", content_hash=chunk_hash, vector_hash=chunk_hash, source_id="doc:a", file_path="a.md"
            ),
        },
        "entities": {
            "doc:a": rec(
                "entity", "doc:a", "doc:a\nAlpha", vector_hash="entity-vector", metadata_hash="entity-meta",
                source_logical_id="doc:a", file_path="a.md",
            ),
        },
        "relationships": {
            "doc:a<SEP>tag:x": rec(
                "relationship",
                "doc:a<SEP>tag:x",
                "RELATED\tdoc:a\ttag:x\nAlpha tag",
                src_id="doc:a",
                tgt_id="tag:x",
                vector_hash="rel-vector",
                metadata_hash="rel-meta",
                weight=0.6,
                source_logical_id="doc:a",
                file_path="a.md",
            ),
        },
    }


def clear_embedding_env(monkeypatch: Any, *names: str) -> None:
    for name in names:
        monkeypatch.delenv(name, raising=False)


def custom_kg_payload(*, chunks=None, entities=None, relationships=None) -> dict[str, Any]:
    chunk = {"content": "Doc A content", "source_id": "doc:a", "file_path": "a.md", "chunk_order_index": 0}
    entity = lambda name, typ, desc: {"entity_name": name, "entity_type": typ, "description": desc, "source_id": "doc:a", "file_path": "a.md"}
    rel = {
        "src_id": "doc:a", "tgt_id": "topic:x", "description": "Doc discusses Topic X",
        "keywords": "DISCUSSES", "source_id": "doc:a", "weight": 1.0, "file_path": "a.md",
    }
    return {
        "chunks": chunks or [chunk],
        "entities": entities or [entity("doc:a", "DOC", "Doc A"), entity("topic:x", "TOPIC", "Topic X")],
        "relationships": [rel] if relationships is None else relationships,
    }


class Hit:
    def __init__(self, record_type: str = "entity", record_id: str = "doc:a", score: float = 1.0) -> None:
        self.doc_id = f"{record_type}:{record_id}"
        self.score = score
        self.fields = {"record_type": record_type, "record_id": record_id}


def seed_audited_entity_db(db_path: Path, workspace_id: str = "native-test", *, text: str = "Alpha", source_path: str = "alpha.md"):
    db = SQLiteWorkspace(db_path)
    db.create_workspace(workspace_id, "manifest-hash")
    db.put_record(native_record(workspace_id, "entity", "doc:a", text, source_path=source_path))
    db.put_vector(workspace_id, "entity", "doc:a", "doc:a:vector", [1.0, 0.0])
    db.mark_audited(workspace_id, {"chunks": 0, "entities": 1, "relationships": 0, "sections": 0}, require_vectors=True)
    return db


def append_native_history(state_dir: Path, reasons: list[str]) -> None:
    from ops import batch_native_refresh

    history = batch_native_refresh.active_workspace_history_path(state_dir)
    history.parent.mkdir(parents=True, exist_ok=True)
    history.write_text(
        "".join(
            json.dumps({"reason": reason, "current": {"workspace_id": f"ws-{idx}"}}) + "\n"
            for idx, reason in enumerate(reasons)
        ),
        encoding="utf-8",
    )


def cutover_cli_args(workdir: Path, root: Path, *extra: str, command: str = "refresh") -> list[str]:
    return [command, "--workdir", str(workdir), "--root", str(root), *extra]


def mark_pending_batch(
    state: Path,
    root: Path,
    count: int,
    *,
    path_prefix: str = "260108",
    **kwargs: Any,
) -> None:
    from ops.wiki_native_wiki_integration_pending import mark_pending_wiki_integration

    for idx in range(count):
        mark_pending_wiki_integration(
            state,
            root,
            raw_path=f"raw/clip/2601/{path_prefix}{idx:02d}_Paper.md",
            title=f"Paper {idx}",
            **kwargs,
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
    """Import-surface zvec stand-in; collection query/filter stays test-owned."""
    fake = types.ModuleType("zvec")

    def _cls(name: str, fields: tuple[str, ...], defaults: dict[str, Any] | None = None):
        defaults = defaults or {}

        def __init__(self, *args, **kwargs):
            values = {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v) for k, v in defaults.items()}
            for key, arg in zip(fields, args):
                values[key] = arg
            values.update(kwargs)
            self.__dict__.update(values)

        return type(name, (), {"__init__": __init__})

    class DataType:
        INT32, STRING, VECTOR_FP32 = "INT32", "STRING", "VECTOR_FP32"

    class MetricType:
        COSINE = "COSINE"

    class InvertIndexParam:
        pass

    exported = {
        "CollectionOption": _cls("CollectionOption", ("read_only", "enable_mmap"), {"read_only": False, "enable_mmap": True}),
        "CollectionSchema": _cls("CollectionSchema", ("name", "fields", "vectors"), {"fields": [], "vectors": []}),
        "DataType": DataType,
        "Doc": _cls("Doc", ("id", "score", "vectors", "fields"), {"score": None, "vectors": {}, "fields": {}}),
        "FieldSchema": _cls("FieldSchema", ("name", "data_type", "nullable", "index_param"), {"nullable": False, "index_param": None}),
        "Fts": _cls("Fts", ("match_string", "query_string"), {"match_string": None, "query_string": None}),
        "FtsIndexParam": _cls("FtsIndexParam", ("tokenizer_name", "filters", "extra_params"), {"tokenizer_name": "standard", "filters": None, "extra_params": ""}),
        "FtsQueryParam": _cls("FtsQueryParam", ("default_operator",), {"default_operator": ""}),
        "HnswIndexParam": _cls("HnswIndexParam", ("metric_type", "m", "ef_construction"), {"metric_type": MetricType.COSINE, "m": 50, "ef_construction": 500}),
        "HnswQueryParam": _cls("HnswQueryParam", ("ef",), {"ef": 300}),
        "InvertIndexParam": InvertIndexParam,
        "MetricType": MetricType,
        "Query": _cls("Query", ("field_name", "vector", "param", "fts"), {"vector": None, "param": None, "fts": None}),
        "RrfReRanker": _cls("RrfReRanker", ("rank_constant",), {}),
        "VectorSchema": _cls("VectorSchema", ("name", "data_type", "dimension", "index_param"), {"index_param": None}),
    }
    fake.calls = []

    def create_and_open(*, path: str, schema, option):
        fake.calls.append(("create_and_open", path, schema, option))
        return {"kind": "created", "path": path}

    def zvec_open(path: str, option):
        fake.calls.append(("open", path, option))
        return {"kind": "opened", "path": path}

    exported["create_and_open"] = create_and_open
    exported["open"] = zvec_open
    for name, value in exported.items():
        setattr(fake, name, value)
    monkeypatch.setitem(sys.modules, "zvec", fake)
    monkeypatch.delitem(sys.modules, "llm_wiki_native.storage.zvec_workspace", raising=False)
    return fake
