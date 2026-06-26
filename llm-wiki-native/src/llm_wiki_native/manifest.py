"""Helpers for native handling of the existing custom_kg manifest."""

from __future__ import annotations

from typing import Any

from llm_wiki_native.storage.sqlite_workspace import NativeRecord, SQLiteWorkspace


def manifest_summary(manifest: dict[str, Any]) -> dict[str, int]:
    return {
        "chunks": len(manifest.get("chunks", {})),
        "entities": len(manifest.get("entities", {})),
        "relationships": len(manifest.get("relationships", {})),
    }


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


def materialize_manifest(db: SQLiteWorkspace, workspace_id: str, manifest: dict[str, Any]) -> dict[str, int]:
    collections = (
        ("chunks", "chunk"),
        ("entities", "entity"),
        ("relationships", "relationship"),
    )
    for collection_name, record_type in collections:
        for record_id, record in manifest.get(collection_name, {}).items():
            db.put_record(_native_record(workspace_id, record_type, str(record_id), record))
    return manifest_summary(manifest)
