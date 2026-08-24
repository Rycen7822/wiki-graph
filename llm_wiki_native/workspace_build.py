"""Callable build helpers for native llm-wiki workspaces."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Callable

import numpy as np

from llm_wiki_native.artifacts import load_custom_kg_manifest, load_raw_sections, load_section_embeddings, load_section_similarity_edges
from llm_wiki_native.build import materialize_zvec_records, raise_if_missing_vectors
from llm_wiki_native.manifest import manifest_summary, materialize_manifest, materialize_raw_sections
from llm_wiki_native.retrieval.lexical import materialize_lexical_spans, spans_from_native_records, spans_from_source_root
from llm_wiki_native.storage.sqlite_workspace import NativeRecord, SQLiteWorkspace
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


@contextmanager
def _timed(timings: dict[str, float], phase: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[phase] = round(timings.get(phase, 0.0) + (time.perf_counter() - start), 3)


def _timed_callable(timings: dict[str, float], phase: str, fn: Callable[[], Any]) -> Callable[[], Any]:
    def run() -> Any:
        with _timed(timings, phase):
            return fn()

    return run


def _manifest_vector_hashes(manifest: dict[str, Any]) -> set[str]:
    hashes: set[str] = set()
    for collection in ("chunks", "entities", "relationships"):
        for record in manifest.get(collection, {}).values():
            vector_hash = record.get("vector_hash") or record.get("content_hash") or record.get("record_hash")
            if vector_hash:
                hashes.add(str(vector_hash))
    return hashes


def _load_vector_cache_vectors(state_dir: Path, vector_hashes: set[str]) -> dict[str, Any]:
    if not vector_hashes:
        return {}
    db_path = Path(state_dir) / "vector_cache.sqlite"
    if not db_path.exists():
        return {}
    # Keep float32 ndarrays zero-copy; conversion to Python floats happens once at the
    # zvec insert boundary (zvec_doc_from_record), not per loader/record hop.
    vectors: dict[str, Any] = {}
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
                vectors[str(row["vector_hash"])] = vector
    return vectors


def _section_embeddings_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["section_id"]): row for row in rows if row.get("section_id") and isinstance(row.get("embedding"), (list, np.ndarray))}


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


def _edge_tuples(manifest: dict[str, Any], section_edges: list[dict[str, Any]]) -> list[tuple[str, str, str, float, dict[str, Any]]]:
    edges: list[tuple[str, str, str, float, dict[str, Any]]] = []
    for relationship in manifest.get("relationships", {}).values():
        edges.append(
            (
                "relationship",
                str(relationship["src_id"]),
                str(relationship["tgt_id"]),
                float(relationship.get("weight", 1.0)),
                relationship,
            )
        )
    for edge in section_edges:
        edges.append(
            (
                "section_similarity",
                str(edge["src_id"]),
                str(edge["tgt_id"]),
                float(edge.get("cosine", edge.get("weight", 1.0))),
                edge,
            )
        )
    return edges


def _materialize_edges(db: SQLiteWorkspace, workspace_id: str, manifest: dict[str, Any], section_edges: list[dict[str, Any]]) -> None:
    db.put_edges(workspace_id, _edge_tuples(manifest, section_edges))


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
    phase_timings: dict[str, float] = {}
    total_start = time.perf_counter()
    with _timed(phase_timings, "load_artifacts"):
        manifest, section_edges, raw_sections, section_embeddings_by_id, vectors_by_hash, source_manifest_hash = _load_build_inputs(Path(state_dir))
    db = SQLiteWorkspace(Path(db_path))
    db.create_workspace(workspace_id, source_manifest_hash)

    def _walk_spans() -> list[Any]:
        if source_root is not None:
            return spans_from_source_root(Path(source_root), workspace_id)
        return spans_from_native_records(workspace_id, manifest, raw_sections)

    def _zvec_chain(target: Path) -> dict[str, Any]:
        with _timed(phase_timings, "zvec_assembly"):
            records = materialize_zvec_records(
                manifest,
                raw_sections,
                vectors_by_hash=vectors_by_hash,
                section_embeddings_by_id=section_embeddings_by_id,
            )
            from llm_wiki_native.storage.zvec_workspace import zvec_doc_id

            embedding_dim = _zvec_embedding_dim(records)
            workspace_factory = zvec_workspace_factory or _default_zvec_workspace_factory
            zvec_workspace = workspace_factory(target, embedding_dim)
        with _timed(phase_timings, "zvec_insert"):
            insert_stats = zvec_workspace.bulk_insert(records)
        flush = getattr(zvec_workspace, "flush_optimize_close", None)
        if callable(flush):
            with _timed(phase_timings, "zvec_optimize"):
                flush()
        smoke_report = None
        self_nearest_smoke = getattr(zvec_workspace, "self_nearest_smoke", None)
        if callable(self_nearest_smoke):
            with _timed(phase_timings, "zvec_smoke"):
                sample_doc_ids = [zvec_doc_id(record.record_type, record.record_id) for record in records[: min(len(records), 20)]]
                smoke_report = _smoke_result_dict(self_nearest_smoke(sample_doc_ids), expected=len(sample_doc_ids))
        report: dict[str, Any] = {
            "path": str(target),
            "embedding_dim": embedding_dim,
            "record_count": len(records),
            "insert_stats": _insert_stats_dict(insert_stats),
        }
        if smoke_report is not None:
            report["self_nearest"] = smoke_report
            report["self_nearest_top1_ok"] = smoke_report["ok"]
        return report

    # Single-writer discipline: every SQLiteWorkspace write stays on the main thread.
    # The spans filesystem walk and the zvec store chain touch neither sqlite nor shared state.
    zvec_report = None
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="workspace-build") as pool:
        spans_future = pool.submit(_timed_callable(phase_timings, "spans_walk", _walk_spans))
        zvec_future = pool.submit(_zvec_chain, Path(zvec_path)) if zvec_path is not None else None
        with _timed(phase_timings, "materialize_manifest"):
            counts = materialize_manifest(db, workspace_id, manifest, vectors_by_hash=vectors_by_hash)
        with _timed(phase_timings, "materialize_sections"):
            counts = {
                **counts,
                "sections": materialize_raw_sections(
                    db,
                    workspace_id,
                    raw_sections,
                    section_embeddings_by_id=section_embeddings_by_id,
                ),
            }
        with _timed(phase_timings, "materialize_edges"):
            _materialize_edges(db, workspace_id, manifest, section_edges)
        lexical_spans = spans_future.result()
        with _timed(phase_timings, "materialize_spans"):
            lexical_span_count = materialize_lexical_spans(db, workspace_id, lexical_spans)
        if zvec_future is not None:
            zvec_report = zvec_future.result()
    with _timed(phase_timings, "audits"):
        expected = {**manifest_summary(manifest), "sections": len(raw_sections)}
        db.mark_audited(workspace_id, expected, require_vectors=True)
        vector_audit = db.audit_vector_coverage(workspace_id)
        audit_counts = db.audit_counts(workspace_id, expected)
    phase_timings["total"] = round(time.perf_counter() - total_start, 3)
    report = {
        "workspace_id": workspace_id,
        "source_manifest_hash": source_manifest_hash,
        "counts": counts,
        "edge_count": db.count_edges(workspace_id),
        "lexical_span_count": lexical_span_count,
        "source_root": str(source_root) if source_root is not None else None,
        "audit": audit_counts,
        "vector_audit": vector_audit,
        "status": db.get_workspace_status(workspace_id),
        "phase_timings": phase_timings,
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


def _rewrite_workspace_id(db_path: Path, old_id: str, new_id: str) -> None:
    """Re-key a copied workspace sqlite from old_id to new_id in one transaction."""
    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        with conn:
            for table in ("record", "vector", "edge", "lexical_span", "lexical_span_fts"):
                conn.execute(f"UPDATE {table} SET workspace_id = ? WHERE workspace_id = ?", (new_id, old_id))
            conn.execute("UPDATE workspace SET workspace_id = ? WHERE workspace_id = ?", (new_id, old_id))
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError(f"workspace rekey left FK violations: {violations[:3]}")
    finally:
        conn.close()


def _load_build_inputs(state_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[float]], str]:
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
    return manifest, section_edges, raw_sections, section_embeddings_by_id, vectors_by_hash, _manifest_hash(manifest)


def apply_incremental_workspace_from_state(
    state_dir: Path,
    db_path: Path,
    workspace_id: str,
    *,
    zvec_path: Path | None = None,
    prepared_workspace_path: Path | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Copy-then-delta build: db_path must already be a byte copy of an audited workspace sqlite.

    Copied rows are re-keyed to workspace_id, then record/edge/span deltas against the
    current state artifacts are applied transactionally; the copied zvec collection is
    upserted/deleted under stable zvec_doc_id identity. Audits and report shape match
    build_workspace_from_state, plus incremental_from/delta/phase_timings metadata.
    """
    from llm_wiki_native.incremental import (
        delta_summary,
        diff_by_key,
        edge_fingerprint,
        fingerprint_for_span_item,
        fingerprints_from_record_rows,
        fingerprints_from_span_rows,
        native_record_fingerprint,
    )
    from llm_wiki_native.manifest import _native_record, _section_record
    from llm_wiki_native.retrieval.lexical import _span_kwargs

    if prepared_workspace_path is not None and zvec_path is None:
        raise ValueError("prepared_workspace_path requires zvec_path")
    phase_timings: dict[str, float] = {}
    total_start = time.perf_counter()
    with _timed(phase_timings, "load_artifacts"):
        manifest, section_edges, raw_sections, section_embeddings_by_id, vectors_by_hash, source_manifest_hash = _load_build_inputs(state_dir)
    with _timed(phase_timings, "rekey"):
        raw_conn = sqlite3.connect(Path(db_path), timeout=30.0)
        try:
            rows = raw_conn.execute("SELECT workspace_id, status FROM workspace").fetchall()
        finally:
            raw_conn.close()
        if len(rows) != 1:
            raise ValueError(f"incremental source workspace must contain exactly one workspace row, found {len(rows)}")
        source_workspace_id, source_status = str(rows[0][0]), str(rows[0][1])
        if source_status != "audited":
            raise ValueError(f"incremental source workspace {source_workspace_id} is not audited (status={source_status})")
        if source_workspace_id != workspace_id:
            _rewrite_workspace_id(Path(db_path), source_workspace_id, workspace_id)
    db = SQLiteWorkspace(Path(db_path))
    db.get_workspace_status(workspace_id)

    current_records: dict[tuple[str, str], NativeRecord] = {}
    for collection_name, record_type in (("chunks", "chunk"), ("entities", "entity"), ("relationships", "relationship")):
        for record_id, record in manifest.get(collection_name, {}).items():
            native = _native_record(workspace_id, record_type, str(record_id), record)
            current_records[(record_type, native.record_id)] = native
    for section in raw_sections:
        section_id = str(section.get("section_id") or "")
        native = _section_record(workspace_id, section, section_embeddings_by_id.get(section_id))
        current_records[("section", native.record_id)] = native

    def _vector_for(native: NativeRecord) -> Any | None:
        if native.record_type == "section":
            embedding_row = section_embeddings_by_id.get(native.record_id)
            if isinstance(embedding_row, dict) and isinstance(embedding_row.get("embedding"), (list, np.ndarray)):
                return embedding_row["embedding"]
            return None
        return vectors_by_hash.get(native.vector_hash)

    def _walk_spans() -> list[Any]:
        if source_root is not None:
            return spans_from_source_root(Path(source_root), workspace_id)
        return spans_from_native_records(workspace_id, manifest, raw_sections)

    def _zvec_delta(target: Path) -> dict[str, Any]:
        from llm_wiki_native.storage.zvec_workspace import open_workspace_collection, zvec_doc_id

        with _timed(phase_timings, "zvec_assembly"):
            all_records = materialize_zvec_records(
                manifest,
                raw_sections,
                vectors_by_hash=vectors_by_hash,
                section_embeddings_by_id=section_embeddings_by_id,
            )
            by_key = {(record.record_type, record.record_id): record for record in all_records}
            embedding_dim = _zvec_embedding_dim(all_records)
            zvec_workspace = open_workspace_collection(target, read_only=False)
        changed_keys = sorted(record_delta.added | record_delta.updated)
        with _timed(phase_timings, "zvec_upsert"):
            upsert_records = []
            for key in changed_keys:
                zvec_record = by_key.get(key)
                if zvec_record is None:
                    raise ValueError(f"zvec record missing for delta key: {key}")
                upsert_records.append(zvec_record)
            upsert_stats = zvec_workspace.upsert_records(upsert_records) if upsert_records else None
        with _timed(phase_timings, "zvec_delete"):
            delete_ids = [zvec_doc_id(record_type, record_id) for record_type, record_id in sorted(record_delta.deleted)]
            delete_stats = zvec_workspace.delete_docs(delete_ids) if delete_ids else None
        flush = getattr(zvec_workspace, "flush_optimize_close", None)
        if callable(flush):
            with _timed(phase_timings, "zvec_optimize"):
                flush()
        smoke_report = None
        self_nearest_smoke = getattr(zvec_workspace, "self_nearest_smoke", None)
        if callable(self_nearest_smoke):
            with _timed(phase_timings, "zvec_smoke"):
                smoke_keys = changed_keys or sorted(current_records)[:20]
                sample_doc_ids = [zvec_doc_id(record_type, record_id) for record_type, record_id in smoke_keys[:20]]
                smoke_report = _smoke_result_dict(self_nearest_smoke(sample_doc_ids), expected=len(sample_doc_ids))
        report: dict[str, Any] = {
            "path": str(target),
            "embedding_dim": embedding_dim,
            "record_count": len(all_records),
            "upsert_stats": _insert_stats_dict(upsert_stats) if upsert_stats is not None else None,
            "delete_stats": {"attempted": delete_stats.attempted, "deleted": delete_stats.deleted, "failed": delete_stats.failed} if delete_stats is not None else None,
        }
        if smoke_report is not None:
            report["self_nearest"] = smoke_report
            report["self_nearest_top1_ok"] = smoke_report["ok"]
        return report

    zvec_report = None
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="workspace-incremental") as pool:
        spans_future = pool.submit(_timed_callable(phase_timings, "spans_walk", _walk_spans))
        with _timed(phase_timings, "diff_records"):
            previous_records = fingerprints_from_record_rows(db.list_record_index_rows(workspace_id))
            current_fingerprints = {key: native_record_fingerprint(native) for key, native in current_records.items()}
            record_delta = diff_by_key(previous_records, current_fingerprints)
        with _timed(phase_timings, "apply_records"):
            changed_keys = sorted(record_delta.added | record_delta.updated)
            upsert_records = [current_records[key] for key in changed_keys]
            vector_upserts: list[tuple[str, str, str, list[float]]] = []
            vector_deletes: list[tuple[str, str]] = []
            for key in changed_keys:
                native = current_records[key]
                vector = _vector_for(native)
                if vector is None:
                    vector_deletes.append((native.record_type, native.record_id))
                else:
                    vector_upserts.append((native.record_type, native.record_id, native.vector_hash, vector))
            db.delete_records(workspace_id, sorted(record_delta.deleted))
            db.put_records(upsert_records)
            db.put_vectors(workspace_id, vector_upserts)
            db.delete_vectors(workspace_id, vector_deletes)
        with _timed(phase_timings, "diff_edges"):
            previous_edges = {
                (row["edge_type"], row["src_id"], row["tgt_id"]): edge_fingerprint(row["weight"], row["payload"])
                for row in db.list_edge_index_rows(workspace_id)
            }
            current_edge_tuples = _edge_tuples(manifest, section_edges)
            current_edge_fingerprints = {
                (edge_type, src_id, tgt_id): edge_fingerprint(weight, payload)
                for edge_type, src_id, tgt_id, weight, payload in current_edge_tuples
            }
            edge_delta = diff_by_key(previous_edges, current_edge_fingerprints)
        with _timed(phase_timings, "apply_edges"):
            edge_by_key = {
                (edge_type, src_id, tgt_id): (edge_type, src_id, tgt_id, weight, payload)
                for edge_type, src_id, tgt_id, weight, payload in current_edge_tuples
            }
            db.delete_edges(workspace_id, sorted(edge_delta.deleted))
            db.put_edges(workspace_id, [edge_by_key[key] for key in sorted(edge_delta.added | edge_delta.updated)])
        lexical_spans = spans_future.result()
        with _timed(phase_timings, "diff_spans"):
            previous_spans = fingerprints_from_span_rows(db.list_lexical_span_index_rows(workspace_id))
            span_items = [span.with_hash() for span in lexical_spans]
            current_span_fingerprints = {("lexical_span", item.span_id): fingerprint_for_span_item(item) for item in span_items}
            span_delta = diff_by_key(previous_spans, current_span_fingerprints)
        with _timed(phase_timings, "apply_spans"):
            kwargs_by_span_id = {item.span_id: _span_kwargs(item) for item in span_items}
            db.delete_lexical_spans(workspace_id, sorted(key[1] for key in span_delta.deleted))
            db.put_lexical_spans(workspace_id, [kwargs_by_span_id[key[1]] for key in sorted(span_delta.added | span_delta.updated)])
        if zvec_path is not None:
            zvec_future = pool.submit(_zvec_delta, Path(zvec_path))
            zvec_report = zvec_future.result()
    with _timed(phase_timings, "audits"):
        db.update_source_manifest_hash(workspace_id, source_manifest_hash)
        expected = {**manifest_summary(manifest), "sections": len(raw_sections)}
        db.mark_audited(workspace_id, expected, require_vectors=True)
        vector_audit = db.audit_vector_coverage(workspace_id)
        audit_counts = db.audit_counts(workspace_id, expected)
    phase_timings["total"] = round(time.perf_counter() - total_start, 3)
    report = {
        "workspace_id": workspace_id,
        "source_manifest_hash": source_manifest_hash,
        "counts": expected,
        "edge_count": db.count_edges(workspace_id),
        "lexical_span_count": len(span_items),
        "source_root": str(source_root) if source_root is not None else None,
        "audit": audit_counts,
        "vector_audit": vector_audit,
        "status": db.get_workspace_status(workspace_id),
        "phase_timings": phase_timings,
        "incremental_from": source_workspace_id,
        "delta": {
            "records": delta_summary(record_delta),
            "edges": delta_summary(edge_delta),
            "spans": delta_summary(span_delta),
        },
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
            "counts": expected,
            "lexical_span_count": len(span_items),
            "source_root": str(source_root) if source_root is not None else None,
            "zvec": zvec_report,
        }
        _write_json_atomic(Path(prepared_workspace_path), pointer)
        report["prepared_workspace"] = str(prepared_workspace_path)
    return report
