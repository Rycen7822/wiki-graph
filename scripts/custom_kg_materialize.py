"""Full custom_kg file-storage materialization helpers.

This module writes a fresh, caller-provided shadow storage directory from a
canonical desired manifest and pre-resolved vectors. It intentionally does not
call embedding APIs or touch live ``rag_storage``.
"""

from __future__ import annotations

from array import array
import base64
import json
import shutil
from pathlib import Path
from typing import Any

try:
    import networkx as nx
except Exception:  # pragma: no cover - runtime/audit environment dependency
    nx = None  # type: ignore[assignment]

from custom_kg_incremental import (
    _chunk_storage_record,
    _entity_graph_data,
    _entity_vdb_data,
    _relationship_graph_edge_upserts,
    _relationship_vdb_data,
    _tracking_from_manifest,
)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _resolved_vector(resolved_vectors: dict[str, dict[str, dict[str, Any]]], collection: str, key: str) -> list[float]:
    record = resolved_vectors.get(collection, {}).get(key)
    if not record or "vector" not in record:
        raise RuntimeError(f"missing resolved vectors for {collection}:{key}")
    return list(record["vector"])


def _vdb_payload(records: list[dict[str, Any]], embedding_dim: int, vectors: list[list[float]]) -> dict[str, Any]:
    if len(records) != len(vectors):
        raise RuntimeError("VDB records/vectors length mismatch")
    matrix = array("f")
    for vector in vectors:
        if len(vector) != int(embedding_dim):
            raise RuntimeError(f"VDB vector dimension mismatch: expected {embedding_dim}, found {len(vector)}")
        matrix.extend(float(value) for value in vector)
    return {"embedding_dim": int(embedding_dim), "data": records, "matrix": base64.b64encode(matrix.tobytes()).decode("ascii")}


def materialize_file_storage_from_manifest(
    manifest: dict[str, Any],
    resolved_vectors: dict[str, dict[str, dict[str, Any]]],
    storage_dir: Path,
) -> dict[str, Any]:
    """Write a fresh wikigraph file-backend storage directory from manifest records.

    All vectors must already be present in ``resolved_vectors``. Missing vectors
    fail closed so callers can run an embedding fill phase or cold fallback
    explicitly instead of silently producing incomplete storage.
    """

    if nx is None:
        raise RuntimeError("networkx is required for GraphML materialization")
    storage_dir = Path(storage_dir)
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    embedding_dim = int(manifest.get("metadata", {}).get("embedding_dim", 0) or 0)
    if embedding_dim <= 0:
        raise RuntimeError("manifest metadata missing positive embedding_dim")

    chunk_vdb: list[dict[str, Any]] = []
    chunk_vectors: list[list[float]] = []
    text_chunks: dict[str, dict[str, Any]] = {}
    for chunk_id, record in manifest.get("chunks", {}).items():
        stored = {"__id__": chunk_id, **_chunk_storage_record(record)}
        vector = _resolved_vector(resolved_vectors, "chunks", chunk_id)
        stored["vector"] = vector
        chunk_vectors.append(vector)
        chunk_vdb.append(stored)
        text_chunks[chunk_id] = _chunk_storage_record(record)

    graph = nx.Graph()
    entity_vdb: list[dict[str, Any]] = []
    entity_vectors: list[list[float]] = []
    for entity_name, record in manifest.get("entities", {}).items():
        graph.add_node(entity_name, **_entity_graph_data(record))
        stored = {"__id__": record["vdb_id"], **_entity_vdb_data(record)}
        vector = _resolved_vector(resolved_vectors, "entities", entity_name)
        stored["vector"] = vector
        entity_vectors.append(vector)
        entity_vdb.append(stored)

    relationship_vdb: list[dict[str, Any]] = []
    relationship_vectors: list[list[float]] = []
    relationship_graph_records: list[dict[str, Any]] = []
    for rel_key, record in manifest.get("relationships", {}).items():
        stored = {"__id__": record["vdb_id"], **_relationship_vdb_data(record)}
        vector = _resolved_vector(resolved_vectors, "relationships", rel_key)
        stored["vector"] = vector
        relationship_vectors.append(vector)
        relationship_vdb.append(stored)
        relationship_graph_records.append(record)
    for src, tgt, data in _relationship_graph_edge_upserts(relationship_graph_records):
        graph.add_edge(src, tgt, **data)

    entity_tracking, relation_tracking = _tracking_from_manifest(manifest)
    _write_json(storage_dir / "vdb_chunks.json", _vdb_payload(chunk_vdb, embedding_dim, chunk_vectors))
    _write_json(storage_dir / "vdb_entities.json", _vdb_payload(entity_vdb, embedding_dim, entity_vectors))
    _write_json(storage_dir / "vdb_relationships.json", _vdb_payload(relationship_vdb, embedding_dim, relationship_vectors))
    _write_json(storage_dir / "kv_store_text_chunks.json", text_chunks)
    _write_json(storage_dir / "kv_store_entity_chunks.json", entity_tracking)
    _write_json(storage_dir / "kv_store_relation_chunks.json", relation_tracking)
    nx.write_graphml(graph, storage_dir / "graph_chunk_entity_relation.graphml")

    return {
        "ok": True,
        "storage_dir": str(storage_dir),
        "counts": {
            "chunks": len(chunk_vdb),
            "entities": len(entity_vdb),
            "relationships": len(relationship_vdb),
            "entity_chunks": len(entity_tracking),
            "relation_chunks": len(relation_tracking),
        },
    }
