"""Helpers for native handling of the existing custom_kg manifest."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from llm_wiki_native.storage.sqlite_workspace import NativeRecord, SQLiteWorkspace


def manifest_summary(manifest: dict[str, Any]) -> dict[str, int]:
    return {
        "chunks": len(manifest.get("chunks", {})),
        "entities": len(manifest.get("entities", {})),
        "relationships": len(manifest.get("relationships", {})),
    }


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _native_record(workspace_id: str, record_type: str, record_id: str, record: dict[str, Any]) -> NativeRecord:
    content_hash = str(record.get("content_hash") or record.get("vector_hash") or record.get("record_hash") or "")
    if not content_hash:
        raise ValueError(f"manifest record missing hash: {record_id}")
    return NativeRecord(
        workspace_id=workspace_id,
        record_type=record_type,
        record_id=record_id,
        vector_text=str(record.get("content", "")),
        content_hash=content_hash,
        metadata_hash=str(record.get("metadata_hash") or record.get("record_hash") or content_hash),
        vector_hash=str(record.get("vector_hash") or content_hash),
        source_path=record.get("file_path"),
        source_id=record.get("source_id") or record.get("source_logical_id"),
        payload=record,
    )


def _section_record(workspace_id: str, section: dict[str, Any], embedding_row: dict[str, Any] | None = None) -> NativeRecord:
    section_id = str(section.get("section_id") or "")
    if not section_id:
        raise ValueError("raw section missing section_id")
    content = str(section.get("content", ""))
    content_hash = str(section.get("content_hash") or section.get("text_hash") or _stable_hash(content))
    metadata_hash = str(section.get("metadata_hash") or _stable_hash({key: value for key, value in section.items() if key != "content"}))
    embedding_hash = str(embedding_row.get("text_hash")) if isinstance(embedding_row, dict) and embedding_row.get("text_hash") else None
    vector_hash = str(section.get("vector_hash") or section.get("text_hash") or embedding_hash or content_hash)
    return NativeRecord(
        workspace_id=workspace_id,
        record_type="section",
        record_id=section_id,
        vector_text=content,
        content_hash=content_hash,
        metadata_hash=metadata_hash,
        vector_hash=vector_hash,
        source_path=section.get("source_path"),
        source_id=section.get("source_id"),
        payload=section,
    )


def materialize_raw_sections(
    db: SQLiteWorkspace,
    workspace_id: str,
    sections: list[dict[str, Any]],
    *,
    section_embeddings_by_id: dict[str, dict[str, Any]] | None = None,
) -> int:
    section_embeddings_by_id = section_embeddings_by_id or {}
    for section in sections:
        section_id = str(section.get("section_id") or "")
        embedding_row = section_embeddings_by_id.get(section_id)
        native = _section_record(workspace_id, section, embedding_row)
        db.put_record(native)
        if isinstance(embedding_row, dict) and isinstance(embedding_row.get("embedding"), list):
            db.put_vector(workspace_id, "section", native.record_id, native.vector_hash, embedding_row["embedding"])
    return len(sections)


def materialize_manifest(
    db: SQLiteWorkspace,
    workspace_id: str,
    manifest: dict[str, Any],
    *,
    vectors_by_hash: dict[str, list[float]] | None = None,
) -> dict[str, int]:
    vectors_by_hash = vectors_by_hash or {}
    collections = (
        ("chunks", "chunk"),
        ("entities", "entity"),
        ("relationships", "relationship"),
    )
    for collection_name, record_type in collections:
        for record_id, record in manifest.get(collection_name, {}).items():
            native = _native_record(workspace_id, record_type, str(record_id), record)
            db.put_record(native)
            vector = vectors_by_hash.get(native.vector_hash)
            if vector is not None:
                db.put_vector(workspace_id, record_type, native.record_id, native.vector_hash, vector)
    return manifest_summary(manifest)
