#!/usr/bin/env python3
"""Custom KG manifest and vector-contract helpers for native llm-wiki refresh.

Production refresh now materializes native zvec workspaces from state artifacts.
This module owns deterministic custom KG manifest generation, diff/hash helpers,
and compatibility checks used by tests. Live file-storage apply, full
materialization, import planning, and activation entrypoints are retired and fail
closed before touching backend storage.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import importlib
import importlib.metadata
import json
import os
import re
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Any

from native_runtime_env import env_int, load_env_file, port_open
from custom_kg_vector_fill import (
    DEFAULT_EMBEDDING_PROFILE,
    EMBEDDING_PROFILES,
    embed_texts_openai_compatible as _native_embed_texts_openai_compatible,
    embedding_profile_env,
    embedding_profile_report,
    fill_missing_manifest_vectors as _native_fill_missing_manifest_vectors,
)
from wiki_wikigraph_compat_names import (
    retired_graph_module_name,
    retired_graph_package_name,
)

try:  # Keep tests usable in the repo Python even when the external graph package is not importable.
    import networkx as nx
except Exception:  # pragma: no cover - audit CLI depends on networkx/external graph env
    nx = None  # type: ignore[assignment]

from wiki_native_lib import (
    DEFAULT_STATE_DIR,
    DEFAULT_WIKI_ROOT,
    DEFAULT_WORKDIR,
    build_custom_kg_payload,
    ensure_state_dirs,
    now_stamp,
    print_json,
    release_process_memory,
)

GRAPH_FIELD_SEP = "<SEP>"
MANIFEST_SCHEMA_VERSION = 1
RELATIONSHIP_VECTOR_CONTENT_ALGORITHM = "llm-wiki-typed-directed-relationship:v1"
MANIFEST_FILENAME = "custom_kg_manifest.json"
DEFAULT_FULL_REBUILD_INTERVAL = 5
PREPARED_WIKIGRAPH_SWAP_ACTIVATION_RETIRED_MESSAGE = (
    "prepared wikigraph storage activation is retired; use batch_native_refresh.py refresh --cutover "
    "or a dedicated audited native migration slice instead of activating the retired file backend"
)
CUSTOM_KG_LIVE_STORAGE_RUNNER_RETIRED_MESSAGE = (
    "custom KG live-storage runner is retired after native zvec production cutover; "
    "use export-manifest plus native_zvec_materialize.py preflight/build for native staging"
)
STORAGE_AUDIT_CONTRACT_VERSION = "audit_custom_kg_storage:v1"
CUSTOM_KG_STORAGE_AUDIT_FILES = {
    "graph": "graph_chunk_entity_relation.graphml",
    "vdb_chunks": "vdb_chunks.json",
    "vdb_entities": "vdb_entities.json",
    "vdb_relationships": "vdb_relationships.json",
    "text_chunks": "kv_store_text_chunks.json",
    "entity_chunks": "kv_store_entity_chunks.json",
    "relation_chunks": "kv_store_relation_chunks.json",
}
_EXTERNAL_GRAPH_PACKAGE = retired_graph_package_name()
_CONTROL_CHAR_PATTERN_ALL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_SURROGATE_PATTERN = re.compile(r"[\ud800-\udfff]")
_PLACEHOLDER_DOCUMENT_SOURCES = {"", "unknown", "unknown_source", "none", "null"}




# ---------------------------------------------------------------------------
# ID and manifest helpers


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Match the external graph package ``compute_mdhash_id`` without importing it."""

    try:
        digest = hashlib.md5(str(content).encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        digest = hashlib.md5(str(content).encode("utf-8", errors="replace")).hexdigest()
    return prefix + digest


def entity_vdb_id(entity_name: str) -> str:
    return compute_mdhash_id(entity_name, prefix="ent-")


def _relation_key_part(value: Any, *, field: str) -> str:
    text = str(value)
    if GRAPH_FIELD_SEP in text:
        raise ValueError(f"relationship {field} contains reserved separator {GRAPH_FIELD_SEP!r}: {text}")
    return text


def relation_vdb_id(src_id: str, tgt_id: str, keywords: str) -> str:
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
    """Compatibility graph/storage pair key for the one-edge-per-pair graph."""

    return GRAPH_FIELD_SEP.join(sorted((src_id, tgt_id)))


def relationship_vector_content(src_id: str, tgt_id: str, keywords: str, description: str) -> str:
    """Embedding text for a typed, directed relationship record."""

    return f"{keywords}\t{src_id}\n{tgt_id}\n{description}"


def relationship_record_key(src_id: str, tgt_id: str, keywords: str) -> str:
    """Manifest key for a typed, directed relationship record.

    The GraphML edge layer is still one undirected edge per endpoint pair,
    but VDB records and relation chunk tracking must preserve distinct relation
    semantics so typed edges do not silent-last-win collapse.
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


def wikigraph_sanitize_text(text: Any, replacement_char: str = "") -> str:
    """Normalize custom_kg chunk text before hashing/storage."""

    value = "" if text is None else str(text)
    try:
        upstream_utils = importlib.import_module(retired_graph_module_name("utils"))
        sanitize_text_for_encoding = upstream_utils.sanitize_text_for_encoding

        return sanitize_text_for_encoding(value, replacement_char=replacement_char)
    except Exception:
        if not value:
            return value
        value = value.strip()
        if not value:
            return value
        value = html.unescape(value)
        value = _SURROGATE_PATTERN.sub(replacement_char, value)
        value = _CONTROL_CHAR_PATTERN_ALL.sub(replacement_char, value)
        return value.strip()


def wikigraph_normalize_file_path(file_path: Any) -> str:
    """Normalize stored custom_kg file paths."""

    try:
        upstream_pipeline = importlib.import_module(retired_graph_module_name("utils_pipeline"))
        normalize_document_file_path = upstream_pipeline.normalize_document_file_path

        return normalize_document_file_path(file_path)
    except Exception:
        source = str(file_path or "").strip()
        if source.lower() in _PLACEHOLDER_DOCUMENT_SOURCES:
            return "unknown_source"
        # Upstream stores the canonical basename and strips parser-hint segments.
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


def _record_vector_hash(record: dict[str, Any]) -> str | None:
    if record.get("vector_hash"):
        return str(record["vector_hash"])
    if "content" not in record:
        return None
    return stable_hash(_vector_hash_payload(record))


def _record_metadata_hash(record: dict[str, Any]) -> str | None:
    if record.get("metadata_hash"):
        return str(record["metadata_hash"])
    if "content" not in record:
        return None
    return stable_hash({key: value for key, value in record.items() if key not in _RECORD_HASH_FIELDS and key != "content"})


def current_wikigraph_tool_version() -> str:
    distribution_names = (f"{_EXTERNAL_GRAPH_PACKAGE}-hku", _EXTERNAL_GRAPH_PACKAGE)
    for distribution in distribution_names:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def _retired_manifest_version_key() -> str:
    return f"{_EXTERNAL_GRAPH_PACKAGE}_version"


def _resolve_wikigraph_tool_version_arg(
    wikigraph_tool_version: str | None,
    compat_kwargs: dict[str, Any],
) -> str | None:
    retired_key = _retired_manifest_version_key()
    if retired_key in compat_kwargs:
        if wikigraph_tool_version is not None:
            raise ValueError("custom KG version was provided through both current and retired keyword names")
        wikigraph_tool_version = compat_kwargs.pop(retired_key)
    if compat_kwargs:
        unknown = ", ".join(sorted(compat_kwargs))
        raise TypeError(f"unexpected custom KG manifest keyword(s): {unknown}")
    return wikigraph_tool_version


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




def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def storage_file_fingerprint(path: Path, *, relative_path: str | None = None) -> dict[str, Any]:
    stat = path.stat()
    return {
        "relative_path": relative_path or path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(path),
    }


def _storage_audit_contract_hash() -> str:
    return stable_hash(
        {
            "version": STORAGE_AUDIT_CONTRACT_VERSION,
            "required_files": CUSTOM_KG_STORAGE_AUDIT_FILES,
        }
    )


def _shadow_audit_issue_count(shadow_audit: dict[str, Any]) -> int:
    issues = shadow_audit.get("issues", [])
    return len(issues) if isinstance(issues, list) else 1


def _coerce_issue_count(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1






def manifest_record_count(manifest: dict[str, Any]) -> dict[str, int]:
    return {
        "chunks": len(manifest.get("chunks", {})),
        "entities": len(manifest.get("entities", {})),
        "relationships": len(manifest.get("relationships", {})),
    }


def metadata_from_environment(
    *,
    wikigraph_tool_version: str | None = None,
    embedding_model: str | None = None,
    embedding_dim: int | None = None,
    embedding_params_version: str | None = None,
    custom_kg_builder_hash: str | None = None,
    section_similarity_params: dict[str, Any] | None = None,
    **compat_kwargs: Any,
) -> dict[str, Any]:
    wikigraph_tool_version = _resolve_wikigraph_tool_version_arg(wikigraph_tool_version, compat_kwargs)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "wikigraph_tool_version": wikigraph_tool_version or current_wikigraph_tool_version(),
        "embedding_model": embedding_model or os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        "embedding_dim": embedding_dim if embedding_dim is not None else env_int("EMBEDDING_DIM", 1536),
        "embedding_params_version": embedding_params_version or os.environ.get("EMBEDDING_PARAMS_VERSION", "v1"),
        "custom_kg_builder_hash": custom_kg_builder_hash or "wiki_native_lib.build_custom_kg_payload:v1",
        "canonical_id_algorithm": "llm-wiki-canonical-id:v1+wikigraph-custom-kg:v1.5",
        "relationship_vector_content_algorithm": RELATIONSHIP_VECTOR_CONTENT_ALGORITHM,
        "section_similarity_params": section_similarity_params or {},
        "incremental_count_since_full": 0,
        "created_at": now_stamp(),
    }


def build_custom_kg_manifest(
    payload: dict[str, Any],
    *,
    wikigraph_tool_version: str | None = None,
    embedding_model: str | None = None,
    embedding_dim: int | None = None,
    embedding_params_version: str | None = None,
    custom_kg_builder_hash: str | None = None,
    section_similarity_params: dict[str, Any] | None = None,
    incremental_count_since_full: int = 0,
    **metadata_compat_kwargs: Any,
) -> dict[str, Any]:
    """Canonicalize a complete custom_kg payload into a desired storage manifest.

    The canonicalization mirrors the external custom_kg storage contract where safe:
    chunks are content-hashed, entities keep the last declaration by name, and
    relationship identity and vector text are typed/directed so same endpoint
    pairs with distinct semantics do not silent-last-win collapse.
    Logical ``source_id`` fields are resolved through the complete payload's
    source-to-chunk map before any diff is attempted.
    """

    metadata = metadata_from_environment(
        wikigraph_tool_version=wikigraph_tool_version,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        embedding_params_version=embedding_params_version,
        custom_kg_builder_hash=custom_kg_builder_hash,
        section_similarity_params=section_similarity_params,
        **metadata_compat_kwargs,
    )
    metadata["incremental_count_since_full"] = incremental_count_since_full
    embedding_contract = {
        "embedding_model": metadata["embedding_model"],
        "embedding_dim": metadata["embedding_dim"],
        "embedding_params_version": metadata["embedding_params_version"],
    }

    chunks: dict[str, dict[str, Any]] = {}
    source_to_chunk: dict[str, str] = {}
    chunk_sources: dict[str, list[str]] = {}
    for index, chunk_data in enumerate(payload.get("chunks", [])):
        content = wikigraph_sanitize_text(chunk_data["content"])
        logical_source_id = str(chunk_data.get("source_id") or chunk_data.get("full_doc_id") or "UNKNOWN")
        chunk_id = compute_mdhash_id(content, prefix="chunk-")
        full_doc_id = str(chunk_data.get("full_doc_id") or logical_source_id)
        file_path = wikigraph_normalize_file_path(chunk_data.get("file_path") or "custom_kg")
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
        file_path = wikigraph_normalize_file_path(entity_data.get("file_path", "custom_kg"))
        vdb_id = entity_vdb_id(entity_name)
        record = {
            "record_type": "entity",
            "entity_name": entity_name,
            "vdb_id": vdb_id,
            "entity_type": entity_type,
            "description": description,
            "source_logical_id": logical_source_id,
            "source_chunk_id": source_chunk_id,
            "file_path": file_path,
            "content": entity_name + "\n" + description,
            **embedding_contract,
        }
        _stamp_identity_and_hashes(record, record_id=vdb_id, canonical_id=entity_name)
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
        file_path = wikigraph_normalize_file_path(relationship_data.get("file_path", "custom_kg"))
        weight = relationship_data.get("weight", 1.0)
        vdb_id = relation_vdb_id(src_id, tgt_id, keywords)
        record = {
            "record_type": "relationship",
            "rel_key": [src_id, tgt_id],
            "chunk_key": key,
            "vdb_id": vdb_id,
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
        _stamp_identity_and_hashes(record, record_id=vdb_id, canonical_id=key)
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


def _embedding_contract(record: dict[str, Any]) -> tuple[str, int | None, str]:
    dim = record.get("embedding_dim")
    try:
        normalized_dim = int(dim) if dim is not None else None
    except (TypeError, ValueError):
        normalized_dim = None
    return (
        str(record.get("embedding_model") or ""),
        normalized_dim,
        str(record.get("embedding_params_version") or ""),
    )


def _diff_collection(collection: str, old_items: dict[str, Any], new_items: dict[str, Any]) -> dict[str, Any]:
    old_keys = set(old_items)
    new_keys = set(new_items)
    add_ids = sorted(new_keys - old_keys)
    delete_ids = sorted(old_keys - new_keys)
    vector_update_ids: list[str] = []
    metadata_update_ids: list[str] = []
    for key in sorted(old_keys & new_keys):
        old_record = old_items[key]
        new_record = new_items[key]
        if old_record.get("record_hash") == new_record.get("record_hash"):
            continue
        old_vector_hash = _record_vector_hash(old_record)
        new_vector_hash = _record_vector_hash(new_record)
        old_metadata_hash = _record_metadata_hash(old_record)
        new_metadata_hash = _record_metadata_hash(new_record)
        if not old_vector_hash or not new_vector_hash:
            vector_update_ids.append(key)
        elif old_vector_hash != new_vector_hash:
            vector_update_ids.append(key)
        elif old_metadata_hash != new_metadata_hash:
            metadata_update_ids.append(key)
        else:
            # The full record hash changed but the known split hashes did not.
            # Treat this as a semantic/vector update so future fields
            # cannot be silently skipped by the optimization.
            vector_update_ids.append(key)
    update_ids = sorted([*vector_update_ids, *metadata_update_ids])
    return {
        "add": len(add_ids),
        "update": len(update_ids),
        "delete": len(delete_ids),
        "add_ids": add_ids,
        "update_ids": update_ids,
        "delete_ids": delete_ids,
        "vector_update": len(vector_update_ids),
        "metadata_update": len(metadata_update_ids),
        "vector_update_ids": vector_update_ids,
        "metadata_update_ids": metadata_update_ids,
    }


def diff_custom_kg_manifests(old_manifest: dict[str, Any], new_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunks": _diff_collection("chunks", old_manifest.get("chunks", {}), new_manifest.get("chunks", {})),
        "entities": _diff_collection("entities", old_manifest.get("entities", {}), new_manifest.get("entities", {})),
        "relationships": _diff_collection("relationships", old_manifest.get("relationships", {}), new_manifest.get("relationships", {})),
    }


def full_materialization_cache_only_blockers(previous_manifest: dict[str, Any] | None, desired_manifest: dict[str, Any]) -> dict[str, Any]:
    """Return diff counts that make cache-only full materialization unsafe.

    The materializer normally assembles storage from already-resolved vectors. If
    desired adds vector records or changes vector text/contract, cache-only mode
    must fail closed; callers that explicitly enable the internal embedding-fill
    phase may continue to seed/resolve/fill and then rely on the final unresolved
    miss guard.
    """

    if previous_manifest is None:
        return {"blocked": False, "total": 0, "collections": {}, "diff": None}
    diff = diff_custom_kg_manifests(previous_manifest, desired_manifest)
    collections: dict[str, dict[str, int]] = {}
    total = 0
    for collection in ("chunks", "entities", "relationships"):
        item = diff[collection]
        add = int(item.get("add", 0) or 0)
        vector_update = int(item.get("vector_update", 0) or 0)
        if add or vector_update:
            collections[collection] = {"add": add, "vector_update": vector_update}
            total += add + vector_update
    return {"blocked": total > 0, "total": total, "collections": collections, "diff": diff}


def compact_vector_cache_report(vector_report: dict[str, Any], *, missing_example_limit: int = 10) -> dict[str, Any]:
    """Return report-safe vector-cache diagnostics without embedding vectors.

    `resolve_manifest_vectors()` keeps full resolved vectors for callers that
    need complete materialization data. Default reports only need compact
    diagnostics; otherwise every report repeats tens of thousands of embedding
    arrays.
    """

    summary = vector_report.get("summary", {}) if isinstance(vector_report, dict) else {}
    missing = vector_report.get("missing", {}) if isinstance(vector_report, dict) else {}
    missing_counts: dict[str, int] = {}
    missing_examples: dict[str, list[str]] = {}
    total_missing = 0
    for collection in ("chunks", "entities", "relationships"):
        values = missing.get(collection, []) if isinstance(missing, dict) else []
        if not isinstance(values, list):
            values = []
        string_values = [str(item) for item in values]
        missing_counts[collection] = len(string_values)
        missing_examples[collection] = string_values[:missing_example_limit]
        total_missing += len(string_values)
    missing_counts["total"] = total_missing
    return {
        "summary": summary,
        "missing_counts": missing_counts,
        "missing_examples": missing_examples,
    }


def embed_texts_openai_compatible(
    texts: list[str],
    *,
    workdir: Path,
    embedding_model: str,
    embedding_dim: int,
    timeout: int | None = None,
) -> list[list[float]]:
    """Compatibility wrapper for the native-safe vector fill module."""

    return _native_embed_texts_openai_compatible(
        texts,
        workdir=workdir,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        timeout=timeout,
    )


def fill_missing_manifest_vectors(
    manifest: dict[str, Any],
    vector_report: dict[str, Any],
    cache: Any,
    *,
    workdir: Path,
    embed_texts_func: Any | None = None,
    embedding_profile: str | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper that preserves monkeypatching of the module embedder."""

    return _native_fill_missing_manifest_vectors(
        manifest,
        vector_report,
        cache,
        workdir=workdir,
        embed_texts_func=embed_texts_func or embed_texts_openai_compatible,
        embedding_profile=embedding_profile,
    )


def choose_refresh_mode(
    previous_manifest: dict[str, Any] | None,
    desired_manifest: dict[str, Any],
    *,
    storage_audit_ok: bool,
    full_rebuild_interval: int = DEFAULT_FULL_REBUILD_INTERVAL,
) -> dict[str, Any]:
    reasons: list[str] = []
    diff: dict[str, Any] | None = None

    if previous_manifest is None:
        reasons.append("missing_manifest")
    else:
        diff = diff_custom_kg_manifests(previous_manifest, desired_manifest)
        if previous_manifest.get("schema_version") != desired_manifest.get("schema_version"):
            reasons.append("manifest_schema_changed")
        previous_meta = previous_manifest.get("metadata", {})
        desired_meta = desired_manifest.get("metadata", {})
        for key in [
            "wikigraph_tool_version",
            "embedding_model",
            "embedding_dim",
            "custom_kg_builder_hash",
            "canonical_id_algorithm",
            "relationship_vector_content_algorithm",
            "section_similarity_params",
        ]:
            if previous_meta.get(key) != desired_meta.get(key):
                reasons.append(f"{key}_changed")
        previous_count = int(previous_meta.get("incremental_count_since_full", 0) or 0)
        if previous_count >= full_rebuild_interval:
            reasons.append("incremental_interval_reached")
    if not storage_audit_ok:
        reasons.append("current_storage_audit_failed")

    selected_mode = "full_rebuild" if reasons else "incremental"
    previous_count = 0
    if previous_manifest is not None:
        previous_count = int(previous_manifest.get("metadata", {}).get("incremental_count_since_full", 0) or 0)
    next_count = 0 if selected_mode == "full_rebuild" else previous_count + 1
    return {
        "selected_mode": selected_mode,
        "reasons": reasons,
        "full_rebuild_interval": full_rebuild_interval,
        "previous_incremental_count_since_full": previous_count,
        "next_incremental_count_since_full": next_count,
        "diff": diff,
        "desired_summary": desired_manifest.get("summary", {}),
        "previous_summary": previous_manifest.get("summary", {}) if previous_manifest else None,
    }


def successful_manifest(
    desired_manifest: dict[str, Any],
    *,
    import_mode: str,
    previous_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = dict(desired_manifest)
    manifest["metadata"] = dict(desired_manifest.get("metadata", {}))
    previous_count = 0
    if previous_manifest is not None:
        previous_count = int(previous_manifest.get("metadata", {}).get("incremental_count_since_full", 0) or 0)
    manifest.setdefault("metadata", {})["incremental_count_since_full"] = 0 if import_mode in {"full_rebuild", "full_materialization"} else previous_count + 1
    manifest["metadata"]["last_successful_import_mode"] = import_mode
    manifest["metadata"]["last_successful_import_at"] = now_stamp()
    return manifest


def write_successful_manifest(
    state_dir: Path,
    desired_manifest: dict[str, Any],
    *,
    import_mode: str,
    previous_manifest: dict[str, Any] | None = None,
) -> Path:
    return write_manifest(state_dir, successful_manifest(desired_manifest, import_mode=import_mode, previous_manifest=previous_manifest))


# ---------------------------------------------------------------------------
# Storage audit helpers


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _load_vdb(path: Path) -> dict[str, dict[str, Any]]:
    raw = _read_json(path, {"data": []})
    if isinstance(raw, dict) and isinstance(raw.get("data"), list):
        records = raw.get("data", [])
    elif isinstance(raw, list):
        records = raw
    else:
        records = []
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        record_id = record.get("__id__") or record.get("id")
        if record_id:
            result[str(record_id)] = record
    return result


def _load_kv(path: Path) -> dict[str, Any]:
    raw = _read_json(path, {})
    return raw if isinstance(raw, dict) else {}


def _normal_tracking(value: Any) -> dict[str, Any]:
    chunk_ids = []
    if isinstance(value, dict):
        chunk_ids = [str(item) for item in value.get("chunk_ids", [])]
    return {"chunk_ids": chunk_ids, "count": len(chunk_ids)}


def _append_issue(issues: list[dict[str, Any]], issue_type: str, **kwargs: Any) -> None:
    issues.append({"type": issue_type, **kwargs})


def audit_custom_kg_storage(storage_dir: Path, manifest: dict[str, Any] | None = None, *, max_samples: int = 10) -> dict[str, Any]:
    storage_dir = storage_dir.resolve()
    issues: list[dict[str, Any]] = []
    required = {name: storage_dir / relative_path for name, relative_path in CUSTOM_KG_STORAGE_AUDIT_FILES.items()}
    for name, path in required.items():
        if not path.exists():
            _append_issue(issues, "missing_storage_file", name=name, path=str(path))
    if issues:
        return {"ok": False, "storage_dir": str(storage_dir), "issues": issues, "counts": {}}
    if nx is None:
        _append_issue(issues, "networkx_unavailable")
        return {"ok": False, "storage_dir": str(storage_dir), "issues": issues, "counts": {}}

    graph = nx.read_graphml(required["graph"])
    graph_nodes = {str(node) for node in graph.nodes}
    graph_pairs = {relation_chunk_key(str(src), str(tgt)) for src, tgt in graph.edges}
    vdb_chunks = _load_vdb(required["vdb_chunks"])
    vdb_entities = _load_vdb(required["vdb_entities"])
    vdb_relationships = _load_vdb(required["vdb_relationships"])
    text_chunks = _load_kv(required["text_chunks"])
    entity_chunks = _load_kv(required["entity_chunks"])
    relation_chunks = _load_kv(required["relation_chunks"])

    chunk_ids = set(vdb_chunks)
    text_chunk_ids = set(text_chunks)
    if chunk_ids != text_chunk_ids:
        _append_issue(
            issues,
            "chunk_store_mismatch",
            missing_text_chunks=sorted(chunk_ids - text_chunk_ids)[:max_samples],
            missing_vdb_chunks=sorted(text_chunk_ids - chunk_ids)[:max_samples],
            vdb_chunks=len(chunk_ids),
            text_chunks=len(text_chunk_ids),
        )

    entity_names: set[str] = set()
    for record_id, record in vdb_entities.items():
        name = record.get("entity_name")
        if name is None:
            _append_issue(issues, "entity_vdb_missing_entity_name", record_id=record_id)
            continue
        entity_names.add(str(name))
    if graph_nodes != entity_names:
        _append_issue(
            issues,
            "graph_entity_vdb_mismatch",
            graph_nodes=len(graph_nodes),
            vdb_entities=len(entity_names),
            missing_vdb=sorted(graph_nodes - entity_names)[:max_samples],
            missing_graph=sorted(entity_names - graph_nodes)[:max_samples],
        )

    rel_pairs_to_ids: dict[str, list[str]] = {}
    manifest_relationship_vdb_ids_by_signature = {
        (str(record.get("src_id")), str(record.get("tgt_id")), str(record.get("keywords"))): str(record.get("vdb_id"))
        for record in (manifest or {}).get("relationships", {}).values()
        if isinstance(record, dict) and record.get("vdb_id") and record.get("src_id") and record.get("tgt_id") and record.get("keywords") not in (None, "")
    }
    for record_id, record in vdb_relationships.items():
        src = record.get("src_id")
        tgt = record.get("tgt_id")
        if not src or not tgt:
            _append_issue(issues, "relationship_vdb_missing_endpoint", record_id=record_id)
            continue
        pair = relation_chunk_key(str(src), str(tgt))
        rel_pairs_to_ids.setdefault(pair, []).append(record_id)
        if record.get("keywords") in (None, ""):
            _append_issue(issues, "relationship_vdb_missing_keywords", record_id=record_id, pair=pair)
            continue
        keywords = str(record["keywords"])
        accepted_ids = {
            manifest_relationship_vdb_ids_by_signature.get((str(src), str(tgt), keywords))
            or relation_vdb_id(str(src), str(tgt), keywords)
        }
        if record_id not in accepted_ids:
            _append_issue(issues, "noncanonical_relationship_id", record_id=record_id, accepted=sorted(accepted_ids)[:max_samples], pair=pair)
    rel_pairs = set(rel_pairs_to_ids)
    if graph_pairs != rel_pairs:
        _append_issue(
            issues,
            "graph_relationship_vdb_mismatch",
            graph_edges=len(graph_pairs),
            vdb_relationships=len(rel_pairs),
            missing_vdb=sorted(graph_pairs - rel_pairs)[:max_samples],
            missing_graph=sorted(rel_pairs - graph_pairs)[:max_samples],
        )

    def check_source(owner_type: str, owner_id: str, source_id: Any) -> None:
        parts = split_source_ids(source_id)
        if not parts or any(part == "UNKNOWN" for part in parts):
            _append_issue(issues, "unknown_source_id", owner_type=owner_type, owner_id=owner_id, source_id=source_id)
            return
        missing = [part for part in parts if part not in chunk_ids or part not in text_chunk_ids]
        if missing:
            _append_issue(issues, "missing_source_chunk", owner_type=owner_type, owner_id=owner_id, missing=missing[:max_samples])

    for node, data in graph.nodes(data=True):
        check_source("graph_node", str(node), data.get("source_id"))
    for src, tgt, data in graph.edges(data=True):
        check_source("graph_edge", relation_chunk_key(str(src), str(tgt)), data.get("source_id"))
    for record_id, record in vdb_entities.items():
        check_source("vdb_entity", str(record.get("entity_name") or record_id), record.get("source_id"))
    for record_id, record in vdb_relationships.items():
        pair = relation_chunk_key(str(record.get("src_id")), str(record.get("tgt_id"))) if record.get("src_id") and record.get("tgt_id") else record_id
        check_source("vdb_relationship", pair, record.get("source_id"))

    expected_entity_tracking: dict[str, dict[str, Any]] = {}
    for node, data in graph.nodes(data=True):
        chunk_list = split_source_ids(data.get("source_id"))
        if chunk_list:
            expected_entity_tracking[str(node)] = {"chunk_ids": chunk_list, "count": len(chunk_list)}
    actual_entity_tracking = {key: _normal_tracking(value) for key, value in entity_chunks.items()}
    if expected_entity_tracking != actual_entity_tracking:
        _append_issue(
            issues,
            "entity_chunk_tracking_mismatch",
            expected=len(expected_entity_tracking),
            actual=len(actual_entity_tracking),
            missing=sorted(set(expected_entity_tracking) - set(actual_entity_tracking))[:max_samples],
            extra=sorted(set(actual_entity_tracking) - set(expected_entity_tracking))[:max_samples],
        )

    if manifest is not None:
        expected_relation_tracking = {
            key: {"chunk_ids": [record["source_chunk_id"]], "count": 1}
            for key, record in manifest.get("relationships", {}).items()
            if record.get("source_chunk_id") not in (None, "", "UNKNOWN")
        }
    else:
        expected_relation_tracking = {}
        for src, tgt, data in graph.edges(data=True):
            chunk_list = split_source_ids(data.get("source_id"))
            if chunk_list:
                key = relation_chunk_key(str(src), str(tgt))
                expected_relation_tracking[key] = {"chunk_ids": chunk_list, "count": len(chunk_list)}
    actual_relation_tracking = {key: _normal_tracking(value) for key, value in relation_chunks.items()}
    if expected_relation_tracking != actual_relation_tracking:
        _append_issue(
            issues,
            "relation_chunk_tracking_mismatch",
            expected=len(expected_relation_tracking),
            actual=len(actual_relation_tracking),
            missing=sorted(set(expected_relation_tracking) - set(actual_relation_tracking))[:max_samples],
            extra=sorted(set(actual_relation_tracking) - set(expected_relation_tracking))[:max_samples],
        )

    if manifest is not None:
        desired_chunks = set(manifest.get("chunks", {}))
        desired_entities = set(manifest.get("entities", {}))
        desired_relationships = manifest.get("relationships", {})
        if desired_chunks != chunk_ids or desired_chunks != text_chunk_ids:
            _append_issue(
                issues,
                "desired_chunk_mismatch",
                desired=len(desired_chunks),
                vdb=len(chunk_ids),
                text=len(text_chunk_ids),
                missing=sorted(desired_chunks - chunk_ids)[:max_samples],
                extra=sorted(chunk_ids - desired_chunks)[:max_samples],
            )
        if desired_entities != graph_nodes or desired_entities != entity_names:
            _append_issue(
                issues,
                "desired_entity_mismatch",
                desired=len(desired_entities),
                graph=len(graph_nodes),
                vdb=len(entity_names),
                missing=sorted(desired_entities - graph_nodes)[:max_samples],
                extra=sorted(graph_nodes - desired_entities)[:max_samples],
            )
        desired_relationship_keys = set(desired_relationships)
        desired_relationship_pairs = {
            relation_chunk_key(str(record["src_id"]), str(record["tgt_id"]))
            for record in desired_relationships.values()
        }
        desired_relationship_vdb_ids = {str(record["vdb_id"]) for record in desired_relationships.values()}
        actual_relation_chunk_keys = set(relation_chunks)
        actual_relationship_vdb_ids = set(vdb_relationships)
        if desired_relationship_pairs != graph_pairs or desired_relationship_pairs != rel_pairs:
            _append_issue(
                issues,
                "desired_relationship_pair_mismatch",
                desired=len(desired_relationship_pairs),
                graph=len(graph_pairs),
                vdb_pairs=len(rel_pairs),
                missing_graph=sorted(desired_relationship_pairs - graph_pairs)[:max_samples],
                extra_graph=sorted(graph_pairs - desired_relationship_pairs)[:max_samples],
                missing_vdb_pairs=sorted(desired_relationship_pairs - rel_pairs)[:max_samples],
                extra_vdb_pairs=sorted(rel_pairs - desired_relationship_pairs)[:max_samples],
            )
        if desired_relationship_keys != actual_relation_chunk_keys:
            _append_issue(
                issues,
                "desired_relationship_chunk_mismatch",
                desired=len(desired_relationship_keys),
                relation_chunks=len(actual_relation_chunk_keys),
                missing=sorted(desired_relationship_keys - actual_relation_chunk_keys)[:max_samples],
                extra=sorted(actual_relation_chunk_keys - desired_relationship_keys)[:max_samples],
            )
        if desired_relationship_vdb_ids != actual_relationship_vdb_ids:
            _append_issue(
                issues,
                "desired_relationship_vdb_mismatch",
                desired=len(desired_relationship_vdb_ids),
                vdb=len(actual_relationship_vdb_ids),
                missing=sorted(desired_relationship_vdb_ids - actual_relationship_vdb_ids)[:max_samples],
                extra=sorted(actual_relationship_vdb_ids - desired_relationship_vdb_ids)[:max_samples],
            )
        for name, entity in manifest.get("entities", {}).items():
            source_chunk = entity.get("source_chunk_id")
            graph_source = graph.nodes[name].get("source_id") if name in graph_nodes else None
            if source_chunk == "UNKNOWN":
                _append_issue(issues, "manifest_unknown_entity_source", entity_name=name)
            elif name in graph_nodes and str(graph_source) != str(source_chunk):
                _append_issue(issues, "desired_entity_source_mismatch", entity_name=name, desired=source_chunk, graph=graph_source)
        for key, relationship in manifest.get("relationships", {}).items():
            source_chunk = relationship.get("source_chunk_id")
            src, tgt = split_relation_chunk_key(key)
            graph_source = graph.edges[src, tgt].get("source_id") if graph.has_edge(src, tgt) else None
            graph_sources = split_source_ids(graph_source)
            if source_chunk == "UNKNOWN":
                _append_issue(issues, "manifest_unknown_relationship_source", relationship=key)
            elif graph.has_edge(src, tgt) and str(source_chunk) not in graph_sources:
                _append_issue(issues, "desired_relationship_source_mismatch", relationship=key, desired=source_chunk, graph=graph_source)

    counts = {
        "graph_nodes": len(graph_nodes),
        "graph_edges": len(graph_pairs),
        "vdb_chunks": len(chunk_ids),
        "text_chunks": len(text_chunk_ids),
        "vdb_entities": len(entity_names),
        "vdb_relationships": len(vdb_relationships),
        "entity_chunks": len(entity_chunks),
        "relation_chunks": len(relation_chunks),
        "isolates": len(list(nx.isolates(graph))),
    }
    return {"ok": not issues, "storage_dir": str(storage_dir), "issues": issues, "counts": counts}


# ---------------------------------------------------------------------------
# Incremental patch implementation


def _chunk_storage_record(record: dict[str, Any]) -> dict[str, Any]:
    content = record["content"]
    return {
        "content": content,
        "source_id": record["source_id"],
        "tokens": int(record.get("tokens", max(1, len(str(content).split())))),
        "chunk_order_index": int(record.get("chunk_order_index", 0)),
        "full_doc_id": record.get("full_doc_id") or record["source_id"],
        "file_path": record.get("file_path", "custom_kg"),
        "status": "processed",
    }


def _entity_graph_data(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "entity_id": record["entity_name"],
        "entity_type": record.get("entity_type", "UNKNOWN"),
        "description": record.get("description", "No description provided"),
        "source_id": record.get("source_chunk_id", "UNKNOWN"),
        "file_path": record.get("file_path", "custom_kg"),
        "created_at": int(time.time()),
    }


def _entity_vdb_data(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": record["content"],
        "entity_name": record["entity_name"],
        "source_id": record.get("source_chunk_id", "UNKNOWN"),
        "description": record.get("description", "No description provided"),
        "entity_type": record.get("entity_type", "UNKNOWN"),
        "file_path": record.get("file_path", "custom_kg"),
    }


def _relationship_graph_data(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "weight": record.get("weight", 1.0),
        "description": record["description"],
        "keywords": record["keywords"],
        "source_id": record.get("source_chunk_id", "UNKNOWN"),
        "file_path": record.get("file_path", "custom_kg"),
        "created_at": int(time.time()),
    }


def _relationship_pair(record: dict[str, Any]) -> str:
    return relation_chunk_key(str(record["src_id"]), str(record["tgt_id"]))


def _aggregate_relationship_graph_data(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate empty relationship record list")
    if len(records) == 1:
        return _relationship_graph_data(records[0])
    ordered = sorted(records, key=lambda record: str(record.get("chunk_key") or relationship_record_key(record["src_id"], record["tgt_id"], record["keywords"])))
    source_ids = sorted({str(record.get("source_chunk_id")) for record in ordered if record.get("source_chunk_id") not in (None, "", "UNKNOWN")})
    keywords = sorted({str(record.get("keywords", "")) for record in ordered if record.get("keywords")})
    file_paths = sorted({str(record.get("file_path", "custom_kg")) for record in ordered if record.get("file_path")})
    return {
        "weight": max(float(record.get("weight", 1.0)) for record in ordered),
        "description": "\n".join(str(record["description"]) for record in ordered),
        "keywords": GRAPH_FIELD_SEP.join(keywords),
        "source_id": GRAPH_FIELD_SEP.join(source_ids) if source_ids else "UNKNOWN",
        "file_path": GRAPH_FIELD_SEP.join(file_paths) if file_paths else "custom_kg",
        "created_at": int(time.time()),
    }


def _relationship_graph_edge_upserts(records: list[dict[str, Any]]) -> list[tuple[str, str, dict[str, Any]]]:
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_pair.setdefault(_relationship_pair(record), []).append(record)
    upserts: list[tuple[str, str, dict[str, Any]]] = []
    for pair in sorted(by_pair):
        records_for_pair = by_pair[pair]
        anchor = sorted(records_for_pair, key=lambda record: str(record.get("chunk_key") or relationship_record_key(record["src_id"], record["tgt_id"], record["keywords"])))[0]
        upserts.append((str(anchor["src_id"]), str(anchor["tgt_id"]), _aggregate_relationship_graph_data(records_for_pair)))
    return upserts


def _relationship_vdb_data(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "src_id": record["src_id"],
        "tgt_id": record["tgt_id"],
        "source_id": record.get("source_chunk_id", "UNKNOWN"),
        "content": record["content"],
        "keywords": record["keywords"],
        "description": record["description"],
        "weight": record.get("weight", 1.0),
        "file_path": record.get("file_path", "custom_kg"),
    }


async def _patch_materialized_vdb_metadata(vdb: Any, updates: dict[str, dict[str, Any]]) -> None:
    """Update NanoVectorDB metadata for unchanged vector content without scheduling re-embedding."""

    if not updates:
        return
    storage = await vdb.client_storage
    data = storage.get("data") if isinstance(storage, dict) else None
    if not isinstance(data, list):
        raise RuntimeError("metadata-only VDB patch requires NanoVectorDB-style client_storage['data']")
    by_id = {record.get("__id__"): record for record in data if isinstance(record, dict) and record.get("__id__")}
    missing: list[str] = []
    for vdb_id, update in updates.items():
        record = by_id.get(vdb_id)
        if record is None:
            missing.append(vdb_id)
            continue
        if "vector" not in record and "__vector__" not in record:
            raise RuntimeError(f"metadata-only VDB patch cannot preserve missing vector for {vdb_id}")
        if record.get("content") != update.get("content"):
            raise RuntimeError(f"metadata-only VDB patch content mismatch for {vdb_id}")
        preserved_vector = record.get("vector")
        preserved_raw_vector = record.get("__vector__")
        for key, value in update.items():
            record[key] = value
        if preserved_vector is not None:
            record["vector"] = preserved_vector
        if preserved_raw_vector is not None:
            record["__vector__"] = preserved_raw_vector
    if missing:
        raise RuntimeError(f"metadata-only VDB patch missing existing vector record(s): {missing[:5]}")
    if hasattr(vdb, "_client_dirty"):
        vdb._client_dirty = True


def _tracking_from_manifest(manifest: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    entity_tracking = {
        name: {"chunk_ids": [record["source_chunk_id"]], "count": 1}
        for name, record in manifest.get("entities", {}).items()
        if record.get("source_chunk_id") not in (None, "", "UNKNOWN")
    }
    relation_tracking = {
        key: {"chunk_ids": [record["source_chunk_id"]], "count": 1}
        for key, record in manifest.get("relationships", {}).items()
        if record.get("source_chunk_id") not in (None, "", "UNKNOWN")
    }
    return entity_tracking, relation_tracking


def _diff_tracking_collection(old_items: dict[str, dict[str, Any]], new_items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    old_keys = set(old_items)
    new_keys = set(new_items)
    add_ids = sorted(new_keys - old_keys)
    delete_ids = sorted(old_keys - new_keys)
    update_ids = sorted(key for key in old_keys & new_keys if old_items[key] != new_items[key])
    upsert_ids = [*add_ids, *update_ids]
    return {
        "add": len(add_ids),
        "update": len(update_ids),
        "delete": len(delete_ids),
        "add_ids": add_ids,
        "update_ids": update_ids,
        "delete_ids": delete_ids,
        "upsert_records": {key: new_items[key] for key in upsert_ids},
    }


def diff_tracking(old_manifest: dict[str, Any], desired_manifest: dict[str, Any]) -> dict[str, Any]:
    old_entity_tracking, old_relation_tracking = _tracking_from_manifest(old_manifest)
    new_entity_tracking, new_relation_tracking = _tracking_from_manifest(desired_manifest)
    return {
        "entities": _diff_tracking_collection(old_entity_tracking, new_entity_tracking),
        "relationships": _diff_tracking_collection(old_relation_tracking, new_relation_tracking),
    }


async def apply_patch_to_storage(
    shadow_storage_dir: Path,
    old_manifest: dict[str, Any],
    desired_manifest: dict[str, Any],
    *,
    workdir: Path,
    tracking_update_mode: str = "full",
) -> dict[str, Any]:
    """Fail closed for the retired ExternalGraph storage patch helper."""

    raise RuntimeError(CUSTOM_KG_LIVE_STORAGE_RUNNER_RETIRED_MESSAGE)




def build_desired_manifest(root: Path, state_dir: Path, *, limit_docs: int | None = None, limit_edges: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, payload_summary = build_custom_kg_payload(root, state_dir, limit_docs, limit_edges)
    desired = build_custom_kg_manifest(payload)
    return desired, payload_summary


def _retired_backend_token_variants() -> tuple[str, str, str]:
    lower = _EXTERNAL_GRAPH_PACKAGE
    mixed = lower[:5].title() + lower[5:].upper()
    return (lower, mixed, lower.upper())


def _retired_backend_token_variant_count(text: str) -> int:
    return sum(1 for token in _retired_backend_token_variants() if token and token in text)


def audit_manifest_content_tokens(manifest: dict[str, Any]) -> dict[str, Any]:
    sources_by_path: dict[str, dict[str, Any]] = {}
    total_variant_count = 0
    for collection in ("chunks", "entities", "relationships"):
        records = manifest.get(collection, {})
        if not isinstance(records, dict):
            continue
        for record_id, record in records.items():
            if not isinstance(record, dict):
                continue
            serialized = json.dumps(record, ensure_ascii=False, sort_keys=True)
            variant_count = _retired_backend_token_variant_count(serialized)
            if not variant_count:
                continue
            total_variant_count += variant_count
            path = str(record.get("file_path") or record.get("source_path") or record_id)
            source = sources_by_path.setdefault(
                path,
                {
                    "source_path": path,
                    "record_count": 0,
                    "token_variant_count": 0,
                    "collections": {},
                },
            )
            source["record_count"] = int(source["record_count"]) + 1
            source["token_variant_count"] = int(source["token_variant_count"]) + variant_count
            collections = source["collections"]
            collections[collection] = int(collections.get(collection, 0)) + 1
    sources = sorted(sources_by_path.values(), key=lambda item: str(item["source_path"]))
    return {
        "ok": total_variant_count == 0,
        "token_variant_count": total_variant_count,
        "source_count": len(sources),
        "sources": sources,
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
    audit = audit_manifest_content_tokens(desired_manifest)
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
    audit = audit_manifest_content_tokens(desired_manifest)
    if not audit["ok"]:
        source_paths = [str(source["source_path"]) for source in audit["sources"][:5]]
        source_hint = ",".join(source_paths) if source_paths else "unknown"
        raise RuntimeError(
            "export-manifest refused to write because generated manifest contains retired backend token variants; "
            f"token_variant_count={audit['token_variant_count']}; first_source_paths={source_hint}"
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


def plan_incremental_import(root: Path, state_dir: Path, workdir: Path, *, full_rebuild_interval: int = DEFAULT_FULL_REBUILD_INTERVAL) -> dict[str, Any]:
    raise RuntimeError(
        "custom KG incremental import planner is retired after native zvec production cutover; "
        "use export-manifest plus native_zvec_materialize.py preflight/build"
    )


def _safe_remove(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _arg_path(args: argparse.Namespace, name: str, default: Path) -> Path:
    value = getattr(args, name, None)
    return Path(value) if value is not None else default


def _record_timing(timings: dict[str, float], key: str, started: float) -> None:
    timings[key] = round(time.perf_counter() - started, 6)


def _count_query_data_items(response: Any) -> dict[str, int]:
    data = response.get("data") if isinstance(response, dict) else None
    if not isinstance(data, dict):
        data = response if isinstance(response, dict) else {}
    counts: dict[str, int] = {}
    for key in ("entities", "relationships", "chunks"):
        value = data.get(key) if isinstance(data, dict) else None
        counts[key] = len(value) if isinstance(value, list) else 0
    return counts




async def run_apply(args: argparse.Namespace) -> dict[str, Any]:
    raise RuntimeError(CUSTOM_KG_LIVE_STORAGE_RUNNER_RETIRED_MESSAGE)




def run_full_materialization_no_swap(args: argparse.Namespace) -> dict[str, Any]:
    """Fail closed for the retired full-materialization runner."""

    raise RuntimeError(CUSTOM_KG_LIVE_STORAGE_RUNNER_RETIRED_MESSAGE)




def run_finalize_prepared_swap(args: argparse.Namespace) -> dict[str, Any]:
    """Retired direct activation entrypoint kept only to fail closed."""

    raise RuntimeError(PREPARED_WIKIGRAPH_SWAP_ACTIVATION_RETIRED_MESSAGE)


# ---------------------------------------------------------------------------
# CLI


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=DEFAULT_WIKI_ROOT)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)


def main() -> int:
    parser = argparse.ArgumentParser(description="Custom KG manifest helpers for llm-wiki native refresh")
    sub = parser.add_subparsers(dest="command", required=True)

    plan_parser = sub.add_parser("plan", help="Retired command; use export-manifest plus native_zvec_materialize.py")
    add_common_paths(plan_parser)
    plan_parser.add_argument("--full-rebuild-interval", type=int, default=DEFAULT_FULL_REBUILD_INTERVAL)

    export_parser = sub.add_parser("export-manifest", help="Write desired custom KG manifest to state without storage mutation")
    add_common_paths(export_parser)
    export_parser.add_argument("--limit-docs", type=int, default=None)
    export_parser.add_argument("--limit-edges", type=int, default=None)

    audit_manifest_parser = sub.add_parser("audit-manifest-content", help="Audit desired manifest content without writing state")
    add_common_paths(audit_manifest_parser)
    audit_manifest_parser.add_argument("--limit-docs", type=int, default=None)
    audit_manifest_parser.add_argument("--limit-edges", type=int, default=None)

    audit_parser = sub.add_parser("audit-storage", help="Retired command; native refresh audits native artifacts")
    add_common_paths(audit_parser)
    audit_parser.add_argument("--manifest", type=Path, default=None)
    audit_parser.add_argument("--storage-dir", type=Path, default=None)

    apply_parser = sub.add_parser("apply", help="Retired command; native staging uses export-manifest plus native_zvec_materialize.py")
    add_common_paths(apply_parser)
    apply_parser.add_argument("--full-rebuild-interval", type=int, default=DEFAULT_FULL_REBUILD_INTERVAL)
    apply_parser.add_argument("--limit-docs", type=int, default=None)
    apply_parser.add_argument("--limit-edges", type=int, default=None)
    apply_parser.add_argument("--server-host", default="127.0.0.1")
    apply_parser.add_argument("--server-port", type=int, default=9621)
    apply_parser.add_argument("--allow-server-running", action="store_true")
    apply_parser.add_argument("--force-incremental", action="store_true")
    apply_parser.add_argument("--no-swap", action="store_true", help="Retained for retired CLI compatibility; command fails before apply")
    apply_parser.add_argument("--prepare-swap", action="store_true", help="Retained for retired CLI compatibility; command fails before apply")
    apply_parser.add_argument("--delete-shadow-on-no-swap", action="store_true")
    apply_parser.add_argument("--write-manifest-without-swap", action="store_true")
    apply_parser.add_argument("--tracking-update-mode", choices=["full", "delta"], default="full", help="Retired compatibility flag; command fails before tracking updates")

    finalize_parser = sub.add_parser("finalize-prepared-swap", help="Retired command; native refresh cutover owns production activation")
    add_common_paths(finalize_parser)
    finalize_parser.add_argument("--prepared-report", type=Path, default=None)
    finalize_parser.add_argument("--server-host", default="127.0.0.1")
    finalize_parser.add_argument("--server-port", type=int, default=9621)
    finalize_parser.add_argument("--allow-server-running", action="store_true")
    finalize_parser.add_argument("--force-shadow-audit", action="store_true", help="Retired compatibility flag; command fails before auditing")
    finalize_parser.add_argument(
        "--allow-current-storage-audit-failure",
        action="store_true",
        help="Retired compatibility flag accepted only so the command can fail closed before activation",
    )

    materialize_parser = sub.add_parser("materialize-full", help="Retired command; native staging uses export-manifest plus native_zvec_materialize.py")
    add_common_paths(materialize_parser)
    materialize_parser.add_argument("--limit-docs", type=int, default=None)
    materialize_parser.add_argument("--limit-edges", type=int, default=None)
    materialize_parser.add_argument("--vector-cache", type=Path, default=None)
    materialize_parser.add_argument("--storage-dir", type=Path, default=None)
    materialize_parser.add_argument("--seed-from-storage", action="store_true", help="Retired compatibility flag; command fails before seeding")
    materialize_parser.add_argument("--seed-storage-dir", type=Path, default=None, help="Retired compatibility path; command fails before use")
    materialize_parser.add_argument("--fill-missing-vectors", action="store_true", help="Retired compatibility flag; command fails before embedding")
    materialize_parser.add_argument("--embedding-profile", choices=sorted(EMBEDDING_PROFILES), default=DEFAULT_EMBEDDING_PROFILE, help="Named embedding env profile for --fill-missing-vectors; default stays conservative")
    materialize_parser.add_argument(
        "--allow-current-storage-audit-failure",
        action="store_true",
        help="Retired compatibility flag accepted only so the command can fail closed before materialization",
    )
    materialize_parser.add_argument("--smoke-query", action="append", default=[], help="Retired compatibility flag; command fails before query smokes")
    materialize_parser.add_argument("--smoke-mode", default="mix", choices=["local", "global", "hybrid", "naive", "mix", "bypass"])
    materialize_parser.add_argument("--smoke-top-k", type=int, default=5)
    materialize_parser.add_argument("--smoke-chunk-top-k", type=int, default=5)
    materialize_parser.add_argument("--no-swap", action="store_true", required=True, help="Retained for retired CLI compatibility; command fails before materialization")
    materialize_parser.add_argument("--prepare-swap", action="store_true", help="Retained for retired CLI compatibility; command fails before materialization")
    materialize_parser.add_argument("--delete-shadow-on-no-swap", action="store_true")

    args = parser.parse_args()
    try:
        if args.command == "plan":
            raise RuntimeError(
                "plan CLI is retired; use export-manifest plus native_zvec_materialize.py preflight/build"
            )
        if args.command == "export-manifest":
            print_json(run_export_manifest(args))
            return 0
        if args.command == "audit-manifest-content":
            result = run_audit_manifest_content(args)
            print_json(result)
            return 0 if result["ok"] else 1
        if args.command == "audit-storage":
            raise RuntimeError(
                "audit-storage CLI is retired; use native_zvec_materialize.py preflight/build artifacts instead"
            )
        if args.command == "apply":
            raise RuntimeError(
                "apply CLI is retired; use export-manifest plus native_zvec_materialize.py "
                "preflight/build for native staging"
            )
        if args.command == "finalize-prepared-swap":
            raise RuntimeError(
                "finalize-prepared-swap CLI is retired; use batch_native_refresh.py cutover "
                "or a dedicated audited native migration slice instead of activating the retired file backend"
            )
        if args.command == "materialize-full":
            raise RuntimeError(
                "materialize-full CLI is retired; use export-manifest plus native_zvec_materialize.py "
                "preflight/build for native staging"
            )
    except Exception as exc:
        print_json({"error": type(exc).__name__, "message": str(exc)})
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
