"""Callable build helpers for the native shadow workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np

from llm_wiki_native.artifacts import load_custom_kg_manifest, load_raw_sections, load_section_embeddings, load_section_similarity_edges
from llm_wiki_native.manifest import manifest_summary, materialize_manifest, materialize_raw_sections
from llm_wiki_native.storage.sqlite_workspace import SQLiteWorkspace


def _manifest_hash(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


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


def build_workspace_from_state(state_dir: Path, db_path: Path, workspace_id: str) -> dict[str, Any]:
    manifest = load_custom_kg_manifest(Path(state_dir))
    section_edges = load_section_similarity_edges(Path(state_dir))
    raw_sections = load_raw_sections(Path(state_dir))
    try:
        section_embeddings = load_section_embeddings(Path(state_dir)) if raw_sections else []
    except FileNotFoundError as exc:
        raise ValueError(f"missing section vector embeddings: {exc}") from exc
    source_manifest_hash = _manifest_hash(manifest)
    db = SQLiteWorkspace(Path(db_path))
    db.create_workspace(workspace_id, source_manifest_hash)
    vectors_by_hash = _load_vector_cache_vectors(Path(state_dir), _manifest_vector_hashes(manifest))
    counts = materialize_manifest(db, workspace_id, manifest, vectors_by_hash=vectors_by_hash)
    counts = {
        **counts,
        "sections": materialize_raw_sections(
            db,
            workspace_id,
            raw_sections,
            section_embeddings_by_id=_section_embeddings_by_id(section_embeddings),
        ),
    }
    _materialize_edges(db, workspace_id, manifest, section_edges)
    expected = {**manifest_summary(manifest), "sections": len(raw_sections)}
    db.mark_audited(workspace_id, expected, require_vectors=True)
    vector_audit = db.audit_vector_coverage(workspace_id)
    return {
        "workspace_id": workspace_id,
        "source_manifest_hash": source_manifest_hash,
        "counts": counts,
        "edge_count": db.count_edges(workspace_id),
        "audit": db.audit_counts(workspace_id, expected),
        "vector_audit": vector_audit,
        "status": db.get_workspace_status(workspace_id),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and audit llm-wiki native shadow workspaces")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-workspace", help="Build an audited native SQLite workspace from existing state artifacts")
    build.add_argument("--state-dir", type=Path, required=True)
    build.add_argument("--db", type=Path, required=True)
    build.add_argument("--workspace-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "build-workspace":
        report = build_workspace_from_state(args.state_dir, args.db, args.workspace_id)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    parser.error(f"unsupported command: {args.command}")
    return 2
