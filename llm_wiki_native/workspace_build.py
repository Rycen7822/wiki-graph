"""Callable build helpers for native llm-wiki workspaces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Callable

import numpy as np

from llm_wiki_native.artifacts import load_custom_kg_manifest, load_raw_sections, load_section_embeddings, load_section_similarity_edges
from llm_wiki_native.build import materialize_zvec_records, raise_if_missing_vectors
from llm_wiki_native.manifest import manifest_summary, materialize_manifest, materialize_raw_sections
from llm_wiki_native.retrieval.lexical import materialize_lexical_spans, spans_from_native_records, spans_from_source_root
from llm_wiki_native.storage.sqlite_workspace import SQLiteWorkspace
from llm_wiki_native.storage.zvec_records import ZvecRecord


def _manifest_hash(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _manifest_vector_hashes(manifest: dict[str, Any]) -> set[str]:
    hashes: set[str] = set()
    for collection in ("chunks", "entities", "relationships"):
        for record in manifest.get(collection, {}).values():
            vector_hash = record.get("vector_hash") or record.get("content_hash") or record.get("record_hash")
            if vector_hash:
                hashes.add(str(vector_hash))
    return hashes


def _load_vector_cache_vectors(state_dir: Path, vector_hashes: set[str]) -> dict[str, list[float]]:
    if not vector_hashes:
        return {}
    db_path = Path(state_dir) / "vector_cache.sqlite"
    if not db_path.exists():
        return {}
    vectors: dict[str, list[float]] = {}
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        ordered = sorted(vector_hashes)
        for start in range(0, len(ordered), 500):
            chunk = ordered[start : start + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT vector_hash, embedding_dim, vector_blob FROM vector_cache WHERE vector_hash IN ({placeholders})",
                chunk,
            ).fetchall()
            for row in rows:
                dim = int(row["embedding_dim"])
                vector = np.frombuffer(bytes(row["vector_blob"]), dtype=np.float32)
                if vector.size != dim:
                    raise ValueError(f"vector_cache dimension mismatch for {row['vector_hash']}: expected {dim}, found {vector.size}")
                vectors[str(row["vector_hash"])] = [float(value) for value in vector]
    return vectors


def _section_embeddings_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["section_id"]): row for row in rows if row.get("section_id") and isinstance(row.get("embedding"), list)}


def _default_zvec_workspace_factory(path: Path, embedding_dim: int) -> Any:
    from llm_wiki_native.storage.zvec_workspace import create_workspace_collection

    return create_workspace_collection(path, embedding_dim)


def _zvec_embedding_dim(records: list[ZvecRecord]) -> int:
    if not records:
        raise ValueError("no zvec records materialized")
    dims = {len(record.embedding) for record in records}
    if len(dims) != 1:
        raise ValueError(f"inconsistent zvec embedding dimensions: {sorted(dims)}")
    return dims.pop()


def _insert_stats_dict(stats: Any) -> dict[str, int]:
    return {
        "attempted": int(stats.attempted),
        "inserted": int(stats.inserted),
        "failed": int(stats.failed),
    }


def _smoke_result_dict(result: Any, expected: int) -> dict[str, Any]:
    checked = int(result.checked)
    passed = int(result.passed)
    failures = list(result.failures)
    return {
        "checked": checked,
        "passed": passed,
        "failures": failures,
        "ok": checked == expected and passed == checked and not failures,
    }


def _materialize_edges(db: SQLiteWorkspace, workspace_id: str, manifest: dict[str, Any], section_edges: list[dict[str, Any]]) -> None:
    for relationship in manifest.get("relationships", {}).values():
        db.put_edge(
            workspace_id,
            "relationship",
            str(relationship["src_id"]),
            str(relationship["tgt_id"]),
            float(relationship.get("weight", 1.0)),
            relationship,
        )
    for edge in section_edges:
        db.put_edge(
            workspace_id,
            "section_similarity",
            str(edge["src_id"]),
            str(edge["tgt_id"]),
            float(edge.get("cosine", edge.get("weight", 1.0))),
            edge,
        )


def build_workspace_from_state(
    state_dir: Path,
    db_path: Path,
    workspace_id: str,
    *,
    zvec_path: Path | None = None,
    prepared_workspace_path: Path | None = None,
    zvec_workspace_factory: Callable[[Path, int], Any] | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    if prepared_workspace_path is not None and zvec_path is None:
        raise ValueError("prepared_workspace_path requires zvec_path")
    manifest = load_custom_kg_manifest(Path(state_dir))
    section_edges = load_section_similarity_edges(Path(state_dir))
    raw_sections = load_raw_sections(Path(state_dir))
    try:
        section_embeddings = load_section_embeddings(Path(state_dir)) if raw_sections else []
    except FileNotFoundError:
        section_embeddings = []
    section_embeddings_by_id = _section_embeddings_by_id(section_embeddings)
    vectors_by_hash = _load_vector_cache_vectors(Path(state_dir), _manifest_vector_hashes(manifest))
    raise_if_missing_vectors(
        manifest,
        raw_sections,
        vectors_by_hash=vectors_by_hash,
        section_embeddings_by_id=section_embeddings_by_id,
    )
    source_manifest_hash = _manifest_hash(manifest)
    db = SQLiteWorkspace(Path(db_path))
    db.create_workspace(workspace_id, source_manifest_hash)
    counts = materialize_manifest(db, workspace_id, manifest, vectors_by_hash=vectors_by_hash)
    counts = {
        **counts,
        "sections": materialize_raw_sections(
            db,
            workspace_id,
            raw_sections,
            section_embeddings_by_id=section_embeddings_by_id,
        ),
    }
    _materialize_edges(db, workspace_id, manifest, section_edges)
    if source_root is not None:
        lexical_spans = spans_from_source_root(Path(source_root), workspace_id)
    else:
        lexical_spans = spans_from_native_records(workspace_id, manifest, raw_sections)
    lexical_span_count = materialize_lexical_spans(db, workspace_id, lexical_spans)
    zvec_report = None
    if zvec_path is not None:
        records = materialize_zvec_records(
            manifest,
            raw_sections,
            vectors_by_hash=vectors_by_hash,
            section_embeddings_by_id=section_embeddings_by_id,
        )
        from llm_wiki_native.storage.zvec_workspace import zvec_doc_id

        embedding_dim = _zvec_embedding_dim(records)
        workspace_factory = zvec_workspace_factory or _default_zvec_workspace_factory
        zvec_workspace = workspace_factory(Path(zvec_path), embedding_dim)
        insert_stats = zvec_workspace.bulk_insert(records)
        flush = getattr(zvec_workspace, "flush_optimize_close", None)
        if callable(flush):
            flush()
        smoke_report = None
        self_nearest_smoke = getattr(zvec_workspace, "self_nearest_smoke", None)
        if callable(self_nearest_smoke):
            sample_doc_ids = [zvec_doc_id(record.record_type, record.record_id) for record in records[: min(len(records), 20)]]
            smoke_report = _smoke_result_dict(self_nearest_smoke(sample_doc_ids), expected=len(sample_doc_ids))
        zvec_report = {
            "path": str(zvec_path),
            "embedding_dim": embedding_dim,
            "record_count": len(records),
            "insert_stats": _insert_stats_dict(insert_stats),
        }
        if smoke_report is not None:
            zvec_report["self_nearest"] = smoke_report
            zvec_report["self_nearest_top1_ok"] = smoke_report["ok"]
    expected = {**manifest_summary(manifest), "sections": len(raw_sections)}
    db.mark_audited(workspace_id, expected, require_vectors=True)
    vector_audit = db.audit_vector_coverage(workspace_id)
    report = {
        "workspace_id": workspace_id,
        "source_manifest_hash": source_manifest_hash,
        "counts": counts,
        "edge_count": db.count_edges(workspace_id),
        "lexical_span_count": lexical_span_count,
        "source_root": str(source_root) if source_root is not None else None,
        "audit": db.audit_counts(workspace_id, expected),
        "vector_audit": vector_audit,
        "status": db.get_workspace_status(workspace_id),
    }
    if zvec_report is not None:
        report["zvec"] = zvec_report
    if prepared_workspace_path is not None:
        pointer = {
            "schema_version": 1,
            "workspace_id": workspace_id,
            "status": "prepared",
            "sqlite_path": str(db_path),
            "zvec_path": str(zvec_path),
            "source_manifest_hash": source_manifest_hash,
            "counts": counts,
            "lexical_span_count": lexical_span_count,
            "source_root": str(source_root) if source_root is not None else None,
            "zvec": zvec_report,
        }
        _write_json_atomic(Path(prepared_workspace_path), pointer)
        report["prepared_workspace"] = str(prepared_workspace_path)
    return report
