#!/usr/bin/env python3
"""Custom KG manifest and vector-contract helpers for native llm-wiki refresh.

Production refresh materializes native zvec workspaces from state artifacts.
This module owns deterministic custom KG manifest generation, hash helpers,
and native manifest contract checks.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
from pathlib import Path
from typing import Any

from native_runtime_env import env_int
from wiki_native_lib import (
    DEFAULT_STATE_DIR,
    DEFAULT_WIKI_ROOT,
    DEFAULT_WORKDIR,
    build_custom_kg_payload,
    ensure_state_dirs,
    now_stamp,
    print_json,
)

GRAPH_FIELD_SEP = "<SEP>"
MANIFEST_SCHEMA_VERSION = 1
RELATIONSHIP_VECTOR_CONTENT_ALGORITHM = "llm-wiki-typed-directed-relationship:v1"
MANIFEST_FILENAME = "custom_kg_manifest.json"
_CONTROL_CHAR_PATTERN_ALL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_SURROGATE_PATTERN = re.compile(r"[\ud800-\udfff]")
_PLACEHOLDER_DOCUMENT_SOURCES = {"", "unknown", "unknown_source", "none", "null"}
NATIVE_MANIFEST_TOOL_VERSION = "wiki-native-custom-kg-manifest:v1"
NATIVE_CANONICAL_ID_ALGORITHM = "llm-wiki-canonical-id:v1+native-custom-kg:v1"


# ---------------------------------------------------------------------------
# ID and manifest helpers


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Deterministic md5-based manifest id used for custom KG records."""

    try:
        digest = hashlib.md5(str(content).encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        digest = hashlib.md5(str(content).encode("utf-8", errors="replace")).hexdigest()
    return prefix + digest


def entity_record_id(entity_name: str) -> str:
    return compute_mdhash_id(entity_name, prefix="ent-")


def _relation_key_part(value: Any, *, field: str) -> str:
    text = str(value)
    if GRAPH_FIELD_SEP in text:
        raise ValueError(f"relationship {field} contains reserved separator {GRAPH_FIELD_SEP!r}: {text}")
    return text


def relationship_record_id(src_id: str, tgt_id: str, keywords: str) -> str:
    if keywords in (None, ""):
        raise ValueError("relationship keywords are required for typed directed relation IDs")
    return compute_mdhash_id(
        GRAPH_FIELD_SEP.join(
            [
                _relation_key_part(src_id, field="src_id"),
                _relation_key_part(tgt_id, field="tgt_id"),
                _relation_key_part(keywords, field="keywords"),
            ]
        ),
        prefix="rel-",
    )


def relation_chunk_key(src_id: str, tgt_id: str) -> str:
    """Stable endpoint-pair key used for source chunk grouping."""

    return GRAPH_FIELD_SEP.join(sorted((src_id, tgt_id)))


def relationship_vector_content(src_id: str, tgt_id: str, keywords: str, description: str) -> str:
    """Embedding text for a typed, directed relationship record."""

    return f"{keywords}\t{src_id}\n{tgt_id}\n{description}"


def relationship_record_key(src_id: str, tgt_id: str, keywords: str) -> str:
    """Manifest key for a typed, directed relationship record.

    The key includes direction and relationship keywords so distinct typed
    relationships between the same endpoints remain separate native records.
    """

    return GRAPH_FIELD_SEP.join(
        [
            _relation_key_part(src_id, field="src_id"),
            _relation_key_part(tgt_id, field="tgt_id"),
            _relation_key_part(keywords, field="keywords"),
        ]
    )


def split_relation_chunk_key(key: str) -> tuple[str, str]:
    parts = key.split(GRAPH_FIELD_SEP)
    if len(parts) < 2:
        raise ValueError(f"Invalid relation key: {key}")
    return parts[0], parts[1]


def split_source_ids(source_id: Any) -> list[str]:
    if source_id in (None, ""):
        return []
    return [part for part in str(source_id).split(GRAPH_FIELD_SEP) if part]


def native_manifest_sanitize_text(text: Any, replacement_char: str = "") -> str:
    """Normalize custom_kg chunk text before hashing/storage using native code only."""

    value = "" if text is None else str(text)
    if not value:
        return value
    value = value.strip()
    if not value:
        return value
    value = html.unescape(value)
    value = _SURROGATE_PATTERN.sub(replacement_char, value)
    value = _CONTROL_CHAR_PATTERN_ALL.sub(replacement_char, value)
    return value.strip()


def native_manifest_normalize_file_path(file_path: Any) -> str:
    """Normalize stored custom_kg file paths using native code only."""

    source = str(file_path or "").strip()
    if source.lower() in _PLACEHOLDER_DOCUMENT_SOURCES:
        return "unknown_source"
    name = Path(source.replace("\\", "/")).name.strip()
    name = re.sub(r"\.\[[^\]]+\](?=\.)", "", name)
    return name or "unknown_source"


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return compute_mdhash_id(encoded)


_RECORD_HASH_FIELDS = {"record_hash", "vector_hash", "metadata_hash"}
_VECTOR_HASH_FIELDS = ("record_type", "content", "embedding_model", "embedding_dim", "embedding_params_version")


def _record_without_hash_fields(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in _RECORD_HASH_FIELDS}


def _vector_hash_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in _VECTOR_HASH_FIELDS if key in record}


def _stamp_vector_metadata_hashes(record: dict[str, Any]) -> None:
    """Attach full, vector-content, and metadata/provenance hashes to a manifest record."""

    record["vector_hash"] = stable_hash(_vector_hash_payload(record))
    record["metadata_hash"] = stable_hash(
        {key: value for key, value in record.items() if key not in _RECORD_HASH_FIELDS and key != "content"}
    )
    record["record_hash"] = stable_hash(_record_without_hash_fields(record))


def _stamp_identity_and_hashes(record: dict[str, Any], *, record_id: str, canonical_id: str) -> None:
    record["record_id"] = str(record_id)
    record["canonical_id"] = str(canonical_id)
    record["vector_text_hash"] = stable_hash(record.get("content", ""))
    _stamp_vector_metadata_hashes(record)


def manifest_path(state_dir: Path) -> Path:
    return state_dir / MANIFEST_FILENAME


def load_manifest(path_or_state_dir: Path) -> dict[str, Any] | None:
    path = path_or_state_dir
    if path.is_dir():
        path = manifest_path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(state_dir: Path, manifest: dict[str, Any]) -> Path:
    ensure_state_dirs(state_dir)
    path = manifest_path(state_dir)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def metadata_from_environment(
    *,
    native_manifest_tool_version: str | None = None,
    embedding_model: str | None = None,
    embedding_dim: int | None = None,
    embedding_params_version: str | None = None,
    custom_kg_builder_hash: str | None = None,
    section_similarity_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "native_manifest_tool_version": native_manifest_tool_version or NATIVE_MANIFEST_TOOL_VERSION,
        "embedding_model": embedding_model or os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        "embedding_dim": embedding_dim if embedding_dim is not None else env_int("EMBEDDING_DIM", 1536),
        "embedding_params_version": embedding_params_version or os.environ.get("EMBEDDING_PARAMS_VERSION", "v1"),
        "custom_kg_builder_hash": custom_kg_builder_hash or "wiki_native_lib.build_custom_kg_payload:v1",
        "canonical_id_algorithm": NATIVE_CANONICAL_ID_ALGORITHM,
        "relationship_vector_content_algorithm": RELATIONSHIP_VECTOR_CONTENT_ALGORITHM,
        "section_similarity_params": section_similarity_params or {},
        "created_at": now_stamp(),
    }


def build_custom_kg_manifest(
    payload: dict[str, Any],
    *,
    native_manifest_tool_version: str | None = None,
    embedding_model: str | None = None,
    embedding_dim: int | None = None,
    embedding_params_version: str | None = None,
    custom_kg_builder_hash: str | None = None,
    section_similarity_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonicalize a complete custom_kg payload into a desired storage manifest.

    Chunks are content-hashed, entities keep the last declaration by name, and
    relationship identity plus vector text are typed/directed so same endpoint
    pairs with distinct semantics do not silent-last-win collapse.
    Logical ``source_id`` fields are resolved through the complete payload's
    source-to-chunk map before records are emitted.
    """

    metadata = metadata_from_environment(
        native_manifest_tool_version=native_manifest_tool_version,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        embedding_params_version=embedding_params_version,
        custom_kg_builder_hash=custom_kg_builder_hash,
        section_similarity_params=section_similarity_params,
    )
    embedding_contract = {
        "embedding_model": metadata["embedding_model"],
        "embedding_dim": metadata["embedding_dim"],
        "embedding_params_version": metadata["embedding_params_version"],
    }

    chunks: dict[str, dict[str, Any]] = {}
    source_to_chunk: dict[str, str] = {}
    chunk_sources: dict[str, list[str]] = {}
    for index, chunk_data in enumerate(payload.get("chunks", [])):
        content = native_manifest_sanitize_text(chunk_data["content"])
        logical_source_id = str(chunk_data.get("source_id") or chunk_data.get("full_doc_id") or "UNKNOWN")
        chunk_id = compute_mdhash_id(content, prefix="chunk-")
        full_doc_id = str(chunk_data.get("full_doc_id") or logical_source_id)
        file_path = native_manifest_normalize_file_path(chunk_data.get("file_path") or "custom_kg")
        chunk_order_index = int(chunk_data.get("chunk_order_index", index))
        record = {
            "record_type": "chunk",
            "chunk_id": chunk_id,
            "content": content,
            "content_hash": compute_mdhash_id(content),
            "source_id": logical_source_id,
            "full_doc_id": full_doc_id,
            "file_path": file_path,
            "chunk_order_index": chunk_order_index,
            **embedding_contract,
        }
        _stamp_identity_and_hashes(record, record_id=chunk_id, canonical_id=chunk_id)
        chunks[chunk_id] = record
        source_to_chunk[logical_source_id] = chunk_id
        chunk_sources.setdefault(chunk_id, [])
        if logical_source_id not in chunk_sources[chunk_id]:
            chunk_sources[chunk_id].append(logical_source_id)

    deduped_entities: dict[str, dict[str, Any]] = {}
    for entity_data in payload.get("entities", []):
        entity_name = str(entity_data["entity_name"])
        deduped_entities.pop(entity_name, None)
        deduped_entities[entity_name] = entity_data

    entities: dict[str, dict[str, Any]] = {}
    for entity_name, entity_data in deduped_entities.items():
        logical_source_id = str(entity_data.get("source_id", "UNKNOWN"))
        source_chunk_id = source_to_chunk.get(logical_source_id, "UNKNOWN")
        description = str(entity_data.get("description", "No description provided"))
        entity_type = str(entity_data.get("entity_type", "UNKNOWN"))
        file_path = native_manifest_normalize_file_path(entity_data.get("file_path", "custom_kg"))
        record_id = entity_record_id(entity_name)
        record = {
            "record_type": "entity",
            "entity_name": entity_name,
            "entity_type": entity_type,
            "description": description,
            "source_logical_id": logical_source_id,
            "source_chunk_id": source_chunk_id,
            "file_path": file_path,
            "content": entity_name + "\n" + description,
            **embedding_contract,
        }
        _stamp_identity_and_hashes(record, record_id=record_id, canonical_id=entity_name)
        entities[entity_name] = record

    deduped_relationships: dict[str, dict[str, Any]] = {}
    for relationship_data in payload.get("relationships", []):
        src_id = str(relationship_data["src_id"])
        tgt_id = str(relationship_data["tgt_id"])
        keywords = str(relationship_data["keywords"])
        key = relationship_record_key(src_id, tgt_id, keywords)
        deduped_relationships.pop(key, None)
        deduped_relationships[key] = relationship_data

    relationships: dict[str, dict[str, Any]] = {}
    for key, relationship_data in deduped_relationships.items():
        src_id, tgt_id = split_relation_chunk_key(key)
        logical_source_id = str(relationship_data.get("source_id", "UNKNOWN"))
        source_chunk_id = source_to_chunk.get(logical_source_id, "UNKNOWN")
        description = str(relationship_data["description"])
        keywords = str(relationship_data["keywords"])
        file_path = native_manifest_normalize_file_path(relationship_data.get("file_path", "custom_kg"))
        weight = relationship_data.get("weight", 1.0)
        record_id = relationship_record_id(src_id, tgt_id, keywords)
        record = {
            "record_type": "relationship",
            "rel_key": [src_id, tgt_id],
            "chunk_key": key,
            "src_id": src_id,
            "tgt_id": tgt_id,
            "description": description,
            "keywords": keywords,
            "weight": weight,
            "source_logical_id": logical_source_id,
            "source_chunk_id": source_chunk_id,
            "file_path": file_path,
            "content": relationship_vector_content(src_id, tgt_id, keywords, description),
            **embedding_contract,
        }
        _stamp_identity_and_hashes(record, record_id=record_id, canonical_id=key)
        relationships[key] = record

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "metadata": metadata,
        "summary": {
            "chunks": len(chunks),
            "entities": len(entities),
            "relationships": len(relationships),
            "unknown_entity_sources": sum(1 for item in entities.values() if item.get("source_chunk_id") == "UNKNOWN"),
            "unknown_relationship_sources": sum(1 for item in relationships.values() if item.get("source_chunk_id") == "UNKNOWN"),
        },
        "source_to_chunk": source_to_chunk,
        "chunk_sources": {key: sorted(values) for key, values in chunk_sources.items()},
        "chunks": dict(sorted(chunks.items())),
        "entities": dict(sorted(entities.items())),
        "relationships": dict(sorted(relationships.items())),
    }


def build_desired_manifest(root: Path, state_dir: Path, *, limit_docs: int | None = None, limit_edges: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, payload_summary = build_custom_kg_payload(root, state_dir, limit_docs, limit_edges)
    desired = build_custom_kg_manifest(payload)
    return desired, payload_summary


NATIVE_MANIFEST_METADATA_KEYS = {
    "schema_version",
    "native_manifest_tool_version",
    "embedding_model",
    "embedding_dim",
    "embedding_params_version",
    "custom_kg_builder_hash",
    "canonical_id_algorithm",
    "relationship_vector_content_algorithm",
    "section_similarity_params",
    "created_at",
    "last_successful_import_mode",
    "last_successful_import_at",
}
NATIVE_MANIFEST_REQUIRED_METADATA_KEYS = {
    "schema_version",
    "native_manifest_tool_version",
    "embedding_model",
    "embedding_dim",
    "canonical_id_algorithm",
    "relationship_vector_content_algorithm",
}
NATIVE_MANIFEST_EXACT_METADATA_VALUES = {
    "schema_version": MANIFEST_SCHEMA_VERSION,
    "canonical_id_algorithm": NATIVE_CANONICAL_ID_ALGORITHM,
    "relationship_vector_content_algorithm": RELATIONSHIP_VECTOR_CONTENT_ALGORITHM,
}

def audit_manifest_native_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    """Audit generated native manifest metadata before writing state."""

    issues: list[dict[str, str]] = []
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, dict):
        issues.append({"type": "invalid_manifest_metadata", "path": "metadata"})
        metadata = {}
    for key in sorted(set(metadata) - NATIVE_MANIFEST_METADATA_KEYS):
        issues.append({"type": "unexpected_manifest_metadata_key", "path": f"metadata.{key}"})
    for key in sorted(NATIVE_MANIFEST_REQUIRED_METADATA_KEYS - set(metadata)):
        issues.append({"type": "missing_manifest_metadata_key", "path": f"metadata.{key}"})
    for key, expected in NATIVE_MANIFEST_EXACT_METADATA_VALUES.items():
        if key in metadata and metadata.get(key) != expected:
            issues.append({"type": "invalid_manifest_metadata_value", "path": f"metadata.{key}"})
    return {
        "ok": not issues,
        "issue_count": len(issues),
        "issues": issues,
    }


def run_audit_manifest_content(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    state_dir = args.state_dir.resolve()
    workdir = args.workdir.resolve()
    desired_manifest, payload_summary = build_desired_manifest(
        root,
        state_dir,
        limit_docs=getattr(args, "limit_docs", None),
        limit_edges=getattr(args, "limit_edges", None),
    )
    audit = audit_manifest_native_contract(desired_manifest)
    return {
        **audit,
        "command": "audit-manifest-content",
        "wiki_root": str(root),
        "workdir": str(workdir),
        "state_dir": str(state_dir),
        "payload": payload_summary,
        "manifest": desired_manifest.get("summary", {}),
        "desired_manifest_hash": stable_hash(desired_manifest),
    }


def run_export_manifest(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    state_dir = args.state_dir.resolve()
    workdir = args.workdir.resolve()
    desired_manifest, payload_summary = build_desired_manifest(
        root,
        state_dir,
        limit_docs=getattr(args, "limit_docs", None),
        limit_edges=getattr(args, "limit_edges", None),
    )
    audit = audit_manifest_native_contract(desired_manifest)
    if not audit["ok"]:
        issue_paths = ",".join(str(issue.get("path", "unknown")) for issue in audit["issues"][:5]) or "unknown"
        raise RuntimeError(
            "export-manifest refused to write because generated manifest violates native manifest metadata contract; "
            f"issue_count={audit['issue_count']}; first_issue_paths={issue_paths}"
        )
    manifest_out = write_manifest(state_dir, desired_manifest)
    return {
        "ok": True,
        "command": "export-manifest",
        "wiki_root": str(root),
        "workdir": str(workdir),
        "state_dir": str(state_dir),
        "manifest_path": str(manifest_out),
        "payload": payload_summary,
        "manifest": desired_manifest.get("summary", {}),
        "desired_manifest_hash": stable_hash(desired_manifest),
    }


# ---------------------------------------------------------------------------
# CLI


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=DEFAULT_WIKI_ROOT)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)


def main() -> int:
    parser = argparse.ArgumentParser(description="Custom KG manifest helpers for llm-wiki native refresh")
    sub = parser.add_subparsers(dest="command", required=True)

    export_parser = sub.add_parser("export-manifest", help="Write desired custom KG manifest to state without storage mutation")
    add_common_paths(export_parser)
    export_parser.add_argument("--limit-docs", type=int, default=None)
    export_parser.add_argument("--limit-edges", type=int, default=None)

    audit_manifest_parser = sub.add_parser("audit-manifest-content", help="Audit desired manifest content without writing state")
    add_common_paths(audit_manifest_parser)
    audit_manifest_parser.add_argument("--limit-docs", type=int, default=None)
    audit_manifest_parser.add_argument("--limit-edges", type=int, default=None)

    args = parser.parse_args()
    try:
        if args.command == "export-manifest":
            print_json(run_export_manifest(args))
            return 0
        if args.command == "audit-manifest-content":
            result = run_audit_manifest_content(args)
            print_json(result)
            return 0 if result["ok"] else 1
    except Exception as exc:
        print_json({"error": type(exc).__name__, "message": str(exc)})
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
