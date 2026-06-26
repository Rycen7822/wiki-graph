#!/usr/bin/env python3
"""Safe incremental custom_kg maintenance for the llm-wiki LightRAG workspace.

This module is deliberately conservative: it derives a complete desired manifest
from the existing deterministic custom_kg payload, diffs that complete manifest
against the previous successful manifest, patches a shadow copy of rag_storage,
audits the shadow, and only then swaps it into production. The first run without
a manifest, incompatible config/version changes, audit failures, or the periodic
reconciliation interval all fall back to the existing full rebuild path.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from lightrag_runtime_env import env_int, load_env_file, port_open

try:  # Keep tests usable in the repo Python even when LightRAG is not importable.
    import networkx as nx
except Exception:  # pragma: no cover - audit CLI depends on networkx/LightRAG env
    nx = None  # type: ignore[assignment]

from wiki_lightrag_lib import (
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
MANIFEST_FILENAME = "custom_kg_manifest.json"
REPORT_FILENAME = "custom_kg_import_report.json"
PREPARED_SWAP_DIRNAME = "prepared_swaps"
PREPARED_SWAP_REPORT_FILENAME = "custom_kg_prepared_swap.json"
PREPARED_SWAP_MANIFEST_FILENAME = "custom_kg_prepared_manifest.json"
DEFAULT_FULL_REBUILD_INTERVAL = 5
DEFAULT_LIGHTRAG_PYTHON = Path(os.environ.get("LIGHTRAG_PYTHON", "/home/xu/.local/share/uv/tools/lightrag-hku/bin/python"))
_CONTROL_CHAR_PATTERN_ALL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_SURROGATE_PATTERN = re.compile(r"[\ud800-\udfff]")
_PLACEHOLDER_DOCUMENT_SOURCES = {"", "unknown", "unknown_source", "none", "null"}


# ---------------------------------------------------------------------------
# ID and manifest helpers


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Match LightRAG v1.5.0 ``compute_mdhash_id`` without importing LightRAG."""

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


def relation_vdb_id(src_id: str, tgt_id: str, keywords: str | None = None) -> str:
    if keywords is None:
        normalized_src, normalized_tgt = sorted((src_id, tgt_id))
        return compute_mdhash_id(normalized_src + normalized_tgt, prefix="rel-")
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


def legacy_relation_vdb_ids(src_id: str, tgt_id: str) -> list[str]:
    normalized_src, normalized_tgt = sorted((src_id, tgt_id))
    ids = [compute_mdhash_id(normalized_src + normalized_tgt, prefix="rel-")]
    reverse = compute_mdhash_id(normalized_tgt + normalized_src, prefix="rel-")
    if reverse not in ids:
        ids.append(reverse)
    return ids


def relation_vdb_ids(src_id: str, tgt_id: str, keywords: str | None = None) -> list[str]:
    ids: list[str] = []
    if keywords is not None:
        ids.append(relation_vdb_id(src_id, tgt_id, keywords))
    for legacy_id in legacy_relation_vdb_ids(src_id, tgt_id):
        if legacy_id not in ids:
            ids.append(legacy_id)
    return ids


def relation_chunk_key(src_id: str, tgt_id: str) -> str:
    """Compatibility graph/storage pair key for LightRAG's one-edge-per-pair graph."""

    return GRAPH_FIELD_SEP.join(sorted((src_id, tgt_id)))


def relationship_record_key(src_id: str, tgt_id: str, keywords: str) -> str:
    """Manifest key for a typed, directed relationship record.

    LightRAG's GraphML edge layer is still one undirected edge per endpoint pair,
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


def lightrag_sanitize_text(text: Any, replacement_char: str = "") -> str:
    """Mirror LightRAG custom_kg chunk sanitization before hashing/storage."""

    value = "" if text is None else str(text)
    try:
        from lightrag.utils import sanitize_text_for_encoding  # type: ignore

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


def lightrag_normalize_file_path(file_path: Any) -> str:
    """Mirror LightRAG's stored custom_kg file_path normalization."""

    try:
        from lightrag.utils_pipeline import normalize_document_file_path  # type: ignore

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


def current_lightrag_version() -> str:
    for distribution in ("lightrag-hku", "lightrag"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    if DEFAULT_LIGHTRAG_PYTHON.exists():
        try:
            completed = subprocess.run(
                [
                    str(DEFAULT_LIGHTRAG_PYTHON),
                    "-c",
                    "import importlib.metadata as m; print(m.version('lightrag-hku'))",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=True,
            )
            version = completed.stdout.strip()
            if version:
                return version
        except Exception:
            pass
    return "unknown"


def manifest_path(state_dir: Path) -> Path:
    return state_dir / MANIFEST_FILENAME


def prepared_swap_dir(state_dir: Path) -> Path:
    return state_dir / PREPARED_SWAP_DIRNAME


def prepared_swap_report_path(state_dir: Path) -> Path:
    return prepared_swap_dir(state_dir) / PREPARED_SWAP_REPORT_FILENAME


def prepared_swap_manifest_path(state_dir: Path) -> Path:
    return prepared_swap_dir(state_dir) / PREPARED_SWAP_MANIFEST_FILENAME


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


def write_prepared_swap_bundle(state_dir: Path, desired_manifest: dict[str, Any], report: dict[str, Any]) -> tuple[Path, Path]:
    """Persist an audited shadow swap bundle without mutating production manifest/storage."""

    ensure_state_dirs(state_dir)
    bundle_dir = prepared_swap_dir(state_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = prepared_swap_manifest_path(state_dir)
    report_out = prepared_swap_report_path(state_dir)
    manifest_out.write_text(json.dumps(desired_manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    report = {**report, "desired_manifest_path": str(manifest_out), "report_path": str(report_out)}
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report_out, manifest_out


def manifest_record_count(manifest: dict[str, Any]) -> dict[str, int]:
    return {
        "chunks": len(manifest.get("chunks", {})),
        "entities": len(manifest.get("entities", {})),
        "relationships": len(manifest.get("relationships", {})),
    }


def metadata_from_environment(
    *,
    lightrag_version: str | None = None,
    embedding_model: str | None = None,
    embedding_dim: int | None = None,
    embedding_params_version: str | None = None,
    custom_kg_builder_hash: str | None = None,
    section_similarity_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "lightrag_version": lightrag_version or current_lightrag_version(),
        "embedding_model": embedding_model or os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        "embedding_dim": embedding_dim if embedding_dim is not None else env_int("EMBEDDING_DIM", 1536),
        "embedding_params_version": embedding_params_version or os.environ.get("EMBEDDING_PARAMS_VERSION", "v1"),
        "custom_kg_builder_hash": custom_kg_builder_hash or "wiki_lightrag_lib.build_custom_kg_payload:v1",
        "canonical_id_algorithm": "llm-wiki-canonical-id:v1+lightrag-custom-kg:v1.5",
        "section_similarity_params": section_similarity_params or {},
        "incremental_count_since_full": 0,
        "created_at": now_stamp(),
    }


def build_custom_kg_manifest(
    payload: dict[str, Any],
    *,
    lightrag_version: str | None = None,
    embedding_model: str | None = None,
    embedding_dim: int | None = None,
    embedding_params_version: str | None = None,
    custom_kg_builder_hash: str | None = None,
    section_similarity_params: dict[str, Any] | None = None,
    incremental_count_since_full: int = 0,
) -> dict[str, Any]:
    """Canonicalize a complete custom_kg payload into a desired storage manifest.

    The canonicalization mirrors LightRAG v1.5.0 custom_kg semantics:
    chunks are content-hashed, entities keep the last declaration by name,
    relationships keep the last declaration by unordered endpoint pair, and
    logical ``source_id`` fields are resolved through the complete payload's
    source-to-chunk map before any diff is attempted.
    """

    metadata = metadata_from_environment(
        lightrag_version=lightrag_version,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        embedding_params_version=embedding_params_version,
        custom_kg_builder_hash=custom_kg_builder_hash,
        section_similarity_params=section_similarity_params,
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
        content = lightrag_sanitize_text(chunk_data["content"])
        logical_source_id = str(chunk_data.get("source_id") or chunk_data.get("full_doc_id") or "UNKNOWN")
        chunk_id = compute_mdhash_id(content, prefix="chunk-")
        full_doc_id = str(chunk_data.get("full_doc_id") or logical_source_id)
        file_path = lightrag_normalize_file_path(chunk_data.get("file_path") or "custom_kg")
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
        file_path = lightrag_normalize_file_path(entity_data.get("file_path", "custom_kg"))
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
        file_path = lightrag_normalize_file_path(relationship_data.get("file_path", "custom_kg"))
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
            "content": f"{keywords}\t{src_id}\n{tgt_id}\n{description}",
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


def _diff_collection(old_items: dict[str, Any], new_items: dict[str, Any]) -> dict[str, Any]:
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
            # Treat this as a semantic/vector update so legacy or future fields
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
        "chunks": _diff_collection(old_manifest.get("chunks", {}), new_manifest.get("chunks", {})),
        "entities": _diff_collection(old_manifest.get("entities", {}), new_manifest.get("entities", {})),
        "relationships": _diff_collection(old_manifest.get("relationships", {}), new_manifest.get("relationships", {})),
    }


def full_materialization_cache_only_blockers(previous_manifest: dict[str, Any] | None, desired_manifest: dict[str, Any]) -> dict[str, Any]:
    """Return diff counts that make cache-only full materialization unsafe.

    The current materialize-full path only assembles storage from already-resolved
    vectors; it has no embedding-fill phase. If desired adds vector records or
    changes vector text/contract, the caller must choose cold import or a future
    explicit embedding-fill path instead of relying on cache-only assembly.
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
            "lightrag_version",
            "embedding_model",
            "embedding_dim",
            "custom_kg_builder_hash",
            "canonical_id_algorithm",
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
    required = {
        "graph": storage_dir / "graph_chunk_entity_relation.graphml",
        "vdb_chunks": storage_dir / "vdb_chunks.json",
        "vdb_entities": storage_dir / "vdb_entities.json",
        "vdb_relationships": storage_dir / "vdb_relationships.json",
        "text_chunks": storage_dir / "kv_store_text_chunks.json",
        "entity_chunks": storage_dir / "kv_store_entity_chunks.json",
        "relation_chunks": storage_dir / "kv_store_relation_chunks.json",
    }
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
    manifest_relationship_vdb_ids = {
        str(record.get("vdb_id"))
        for record in (manifest or {}).get("relationships", {}).values()
        if isinstance(record, dict) and record.get("vdb_id")
    }
    for record_id, record in vdb_relationships.items():
        src = record.get("src_id")
        tgt = record.get("tgt_id")
        if not src or not tgt:
            _append_issue(issues, "relationship_vdb_missing_endpoint", record_id=record_id)
            continue
        pair = relation_chunk_key(str(src), str(tgt))
        rel_pairs_to_ids.setdefault(pair, []).append(record_id)
        keywords = str(record.get("keywords")) if record.get("keywords") not in (None, "") else None
        accepted_ids = set(relation_vdb_ids(str(src), str(tgt), keywords)) | manifest_relationship_vdb_ids
        if record_id not in accepted_ids:
            _append_issue(issues, "legacy_or_noncanonical_relationship_id", record_id=record_id, accepted=sorted(accepted_ids)[:max_samples], pair=pair)
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
            if source_chunk == "UNKNOWN":
                _append_issue(issues, "manifest_unknown_relationship_source", relationship=key)
            elif graph.has_edge(src, tgt) and str(graph_source) != str(source_chunk):
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
    """Apply manifest diff to a shadow LightRAG storage directory using storage APIs."""

    from import_custom_kg import build_rag  # Imported lazily to avoid circular imports in full rebuild.

    diff = diff_custom_kg_manifests(old_manifest, desired_manifest)
    if tracking_update_mode not in {"full", "delta"}:
        raise ValueError(f"tracking_update_mode must be 'full' or 'delta', got {tracking_update_mode!r}")
    tracking_diff = diff_tracking(old_manifest, desired_manifest)
    diff["tracking_update_mode"] = tracking_update_mode
    diff["tracking"] = tracking_diff
    rag = build_rag(workdir, storage_dir=shadow_storage_dir)
    await rag.initialize_storages()
    try:
        old_rels = old_manifest.get("relationships", {})
        new_rels = desired_manifest.get("relationships", {})
        rel_vector_update_ids = set(diff["relationships"].get("vector_update_ids", diff["relationships"]["update_ids"]))
        rel_metadata_update_ids = set(diff["relationships"].get("metadata_update_ids", [])) - rel_vector_update_ids
        rels_to_remove = sorted(set(diff["relationships"]["delete_ids"]) | rel_vector_update_ids)
        affected_relationship_pairs: set[str] = set()
        for key in rels_to_remove:
            if key in old_rels:
                affected_relationship_pairs.add(_relationship_pair(old_rels[key]))
        if rels_to_remove:
            await rag.chunk_entity_relation_graph.remove_edges([tuple(old_rels[key]["rel_key"]) for key in rels_to_remove if key in old_rels])
            delete_rel_vdb_ids = sorted(
                {
                    rel_id
                    for key in rels_to_remove
                    if key in old_rels
                    for rel_id in [str(old_rels[key].get("vdb_id") or ""), *relation_vdb_ids(old_rels[key]["src_id"], old_rels[key]["tgt_id"], str(old_rels[key].get("keywords", "")))]
                    if rel_id
                }
            )
            if delete_rel_vdb_ids:
                await rag.relationships_vdb.delete(delete_rel_vdb_ids)
            await rag.relation_chunks.delete([key for key in rels_to_remove if key in old_rels])

        old_entities = old_manifest.get("entities", {})
        entity_vector_update_ids = set(diff["entities"].get("vector_update_ids", diff["entities"]["update_ids"]))
        entity_metadata_update_ids = set(diff["entities"].get("metadata_update_ids", [])) - entity_vector_update_ids
        entities_to_delete = diff["entities"]["delete_ids"]
        if entities_to_delete:
            await rag.chunk_entity_relation_graph.remove_nodes(entities_to_delete)
            await rag.entities_vdb.delete([old_entities[key]["vdb_id"] for key in entities_to_delete if key in old_entities])
            await rag.entity_chunks.delete([key for key in entities_to_delete if key in old_entities])

        chunks_to_delete = diff["chunks"]["delete_ids"]
        if chunks_to_delete:
            await rag.chunks_vdb.delete(chunks_to_delete)
            await rag.text_chunks.delete(chunks_to_delete)

        chunk_upsert_ids = sorted(set(diff["chunks"]["add_ids"]) | set(diff["chunks"]["update_ids"]))
        if chunk_upsert_ids:
            chunk_payload = {chunk_id: _chunk_storage_record(desired_manifest["chunks"][chunk_id]) for chunk_id in chunk_upsert_ids}
            await asyncio.gather(rag.chunks_vdb.upsert(chunk_payload), rag.text_chunks.upsert(chunk_payload))

        entity_graph_upsert_ids = sorted(set(diff["entities"]["add_ids"]) | set(diff["entities"]["update_ids"]))
        if entity_graph_upsert_ids:
            entities = [desired_manifest["entities"][entity_name] for entity_name in entity_graph_upsert_ids]
            await rag.chunk_entity_relation_graph.upsert_nodes_batch([(record["entity_name"], _entity_graph_data(record)) for record in entities])
        entity_vdb_upsert_ids = sorted(set(diff["entities"]["add_ids"]) | entity_vector_update_ids)
        if entity_vdb_upsert_ids:
            entities = [desired_manifest["entities"][entity_name] for entity_name in entity_vdb_upsert_ids]
            await rag.entities_vdb.upsert({record["vdb_id"]: _entity_vdb_data(record) for record in entities})
        entity_metadata_records = [desired_manifest["entities"][entity_name] for entity_name in sorted(entity_metadata_update_ids)]
        if entity_metadata_records:
            await _patch_materialized_vdb_metadata(
                rag.entities_vdb,
                {record["vdb_id"]: _entity_vdb_data(record) for record in entity_metadata_records},
            )

        rel_graph_upsert_ids = sorted(set(diff["relationships"]["add_ids"]) | set(diff["relationships"]["update_ids"]))
        for key in rel_graph_upsert_ids:
            if key in new_rels:
                affected_relationship_pairs.add(_relationship_pair(new_rels[key]))
        rel_graph_records = [record for record in new_rels.values() if _relationship_pair(record) in affected_relationship_pairs]
        if rel_graph_records:
            await rag.chunk_entity_relation_graph.upsert_edges_batch(_relationship_graph_edge_upserts(rel_graph_records))
        rel_vdb_upsert_ids = sorted(set(diff["relationships"]["add_ids"]) | rel_vector_update_ids)
        if rel_vdb_upsert_ids:
            relationships = [new_rels[key] for key in rel_vdb_upsert_ids]
            await rag.relationships_vdb.upsert({record["vdb_id"]: _relationship_vdb_data(record) for record in relationships})
            legacy_ids = sorted({rel_id for record in relationships for rel_id in legacy_relation_vdb_ids(record["src_id"], record["tgt_id"])})
            if legacy_ids:
                await rag.relationships_vdb.delete(legacy_ids)
        rel_metadata_records = [new_rels[key] for key in sorted(rel_metadata_update_ids)]
        if rel_metadata_records:
            await _patch_materialized_vdb_metadata(
                rag.relationships_vdb,
                {record["vdb_id"]: _relationship_vdb_data(record) for record in rel_metadata_records},
            )

        if tracking_update_mode == "delta":
            entity_tracking_delta = tracking_diff["entities"]
            relation_tracking_delta = tracking_diff["relationships"]
            if entity_tracking_delta["delete_ids"]:
                await rag.entity_chunks.delete(entity_tracking_delta["delete_ids"])
            if relation_tracking_delta["delete_ids"]:
                await rag.relation_chunks.delete(relation_tracking_delta["delete_ids"])
            if entity_tracking_delta["upsert_records"]:
                await rag.entity_chunks.upsert(entity_tracking_delta["upsert_records"])
            if relation_tracking_delta["upsert_records"]:
                await rag.relation_chunks.upsert(relation_tracking_delta["upsert_records"])
        else:
            entity_tracking, relation_tracking = _tracking_from_manifest(desired_manifest)
            await rag.entity_chunks.drop()
            await rag.relation_chunks.drop()
            if entity_tracking:
                await rag.entity_chunks.upsert(entity_tracking)
            if relation_tracking:
                await rag.relation_chunks.upsert(relation_tracking)

        # This patch includes deletes, so use plain _insert_done instead of the
        # upsert-only cleanup helper. The shadow directory is discarded on error.
        await rag._insert_done()
    finally:
        await rag.finalize_storages()
    return diff


def build_desired_manifest(root: Path, state_dir: Path, *, limit_docs: int | None = None, limit_edges: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, payload_summary = build_custom_kg_payload(root, state_dir, limit_docs, limit_edges)
    desired = build_custom_kg_manifest(payload)
    return desired, payload_summary


def plan_incremental_import(root: Path, state_dir: Path, workdir: Path, *, full_rebuild_interval: int = DEFAULT_FULL_REBUILD_INTERVAL) -> dict[str, Any]:
    ensure_state_dirs(state_dir)
    load_env_file(workdir / ".env")
    desired_manifest, payload_summary = build_desired_manifest(root, state_dir)
    previous_manifest = load_manifest(state_dir)
    storage_audit = {"ok": True, "issues": [], "counts": {}}
    if previous_manifest is not None:
        storage_audit = audit_custom_kg_storage(workdir / "rag_storage", previous_manifest)
    plan = choose_refresh_mode(previous_manifest, desired_manifest, storage_audit_ok=bool(storage_audit.get("ok")), full_rebuild_interval=full_rebuild_interval)
    plan.update(
        {
            "manifest_path": str(manifest_path(state_dir)),
            "payload": payload_summary,
            "desired_summary": desired_manifest.get("summary", {}),
            "storage_audit": storage_audit,
        }
    )
    del desired_manifest, previous_manifest, storage_audit
    release_process_memory()
    return plan


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


def run_shadow_query_data_smokes(
    *,
    workdir: Path,
    storage_dir: Path,
    queries: list[str],
    mode: str = "mix",
    top_k: int = 5,
    chunk_top_k: int = 5,
) -> dict[str, Any]:
    """Run direct LightRAG ``aquery_data`` smokes against an explicit shadow storage dir."""

    from import_custom_kg import build_rag
    from lightrag import QueryParam  # type: ignore[import-not-found]

    workdir = Path(workdir).resolve()
    storage_dir = Path(storage_dir).resolve()
    load_env_file(workdir / ".env")

    async def _run() -> dict[str, Any]:
        rag = build_rag(workdir, storage_dir=storage_dir)
        await rag.initialize_storages()
        results: list[dict[str, Any]] = []
        try:
            for query in queries:
                response = await rag.aquery_data(
                    query,
                    param=QueryParam(mode=mode, top_k=top_k, chunk_top_k=chunk_top_k, max_total_tokens=8000),
                )
                counts = _count_query_data_items(response)
                results.append({"query": query, "ok": any(counts.values()), "counts": counts})
        finally:
            await rag.finalize_storages()
        return {"ok": bool(results) and all(item["ok"] for item in results), "queries": results}

    return asyncio.run(_run())


async def run_apply(args: argparse.Namespace) -> dict[str, Any]:
    total_started = time.perf_counter()
    timings: dict[str, float] = {}
    root = args.root.resolve()
    state_dir = args.state_dir.resolve()
    workdir = args.workdir.resolve()

    phase_started = time.perf_counter()
    ensure_state_dirs(state_dir)
    load_env_file(workdir / ".env")
    _record_timing(timings, "load_env_s", phase_started)

    phase_started = time.perf_counter()
    previous_manifest = load_manifest(state_dir)
    _record_timing(timings, "load_previous_manifest_s", phase_started)

    phase_started = time.perf_counter()
    desired_manifest, payload_summary = build_desired_manifest(root, state_dir, limit_docs=args.limit_docs, limit_edges=args.limit_edges)
    _record_timing(timings, "build_desired_manifest_s", phase_started)

    phase_started = time.perf_counter()
    storage_audit = {"ok": False, "issues": [{"type": "missing_manifest"}], "counts": {}}
    if previous_manifest is not None:
        storage_audit = audit_custom_kg_storage(workdir / "rag_storage", previous_manifest)
    _record_timing(timings, "audit_live_storage_s", phase_started)
    release_process_memory()

    phase_started = time.perf_counter()
    plan = choose_refresh_mode(previous_manifest, desired_manifest, storage_audit_ok=bool(storage_audit.get("ok")), full_rebuild_interval=args.full_rebuild_interval)
    _record_timing(timings, "choose_mode_s", phase_started)
    if plan["selected_mode"] != "incremental" and not args.force_incremental:
        raise RuntimeError(f"Incremental apply is not allowed; selected_mode={plan['selected_mode']} reasons={plan['reasons']}")
    if previous_manifest is None:
        raise RuntimeError("Incremental apply requires a previous custom_kg manifest")
    prepare_swap = bool(getattr(args, "prepare_swap", False))
    if port_open(args.server_host, args.server_port) and not args.allow_server_running and not args.no_swap and not prepare_swap:
        raise RuntimeError(f"{args.server_host}:{args.server_port} is listening. Stop lightrag-server before swapping rag_storage.")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    live_storage = workdir / "rag_storage"
    if not live_storage.exists():
        raise RuntimeError(f"missing live rag_storage: {live_storage}")
    shadow_storage = workdir / f"rag_storage.shadow.{stamp}"
    backup_dir = state_dir / "backups" / f"rag_storage_incremental_{stamp}"

    phase_started = time.perf_counter()
    _safe_remove(shadow_storage)
    shutil.copytree(live_storage, shadow_storage)
    _record_timing(timings, "copy_live_to_shadow_s", phase_started)

    started_at = now_stamp()
    phase_started = time.perf_counter()
    diff = await apply_patch_to_storage(
        shadow_storage,
        previous_manifest,
        desired_manifest,
        workdir=workdir,
        tracking_update_mode=getattr(args, "tracking_update_mode", "full"),
    )
    _record_timing(timings, "apply_patch_to_shadow_s", phase_started)
    release_process_memory()

    phase_started = time.perf_counter()
    shadow_audit = audit_custom_kg_storage(shadow_storage, desired_manifest)
    _record_timing(timings, "audit_shadow_storage_s", phase_started)
    release_process_memory()
    if not shadow_audit.get("ok"):
        raise RuntimeError(f"shadow storage audit failed: {json.dumps(shadow_audit.get('issues', [])[:10], ensure_ascii=False)}")

    phase_started = time.perf_counter()
    swapped = False
    if not args.no_swap and not prepare_swap:
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        _safe_remove(backup_dir)
        try:
            live_storage.rename(backup_dir)
            shadow_storage.rename(live_storage)
            swapped = True
        except Exception:
            if not live_storage.exists() and backup_dir.exists():
                backup_dir.rename(live_storage)
            raise
    _record_timing(timings, "swap_shadow_to_live_s", phase_started)

    phase_started = time.perf_counter()
    if args.no_swap and not prepare_swap and args.delete_shadow_on_no_swap:
        shutil.rmtree(shadow_storage, ignore_errors=True)
    _record_timing(timings, "cleanup_shadow_s", phase_started)

    phase_started = time.perf_counter()
    final_manifest = successful_manifest(desired_manifest, import_mode="incremental", previous_manifest=previous_manifest)
    manifest_written = None
    if swapped or args.write_manifest_without_swap:
        manifest_written = str(write_manifest(state_dir, final_manifest))
    _record_timing(timings, "write_manifest_s", phase_started)
    timings["total_s"] = round(time.perf_counter() - total_started, 6)

    report = {
        "started_at": started_at,
        "finished_at": now_stamp(),
        "wiki_root": str(root),
        "workdir": str(workdir),
        "state_dir": str(state_dir),
        "dry_run": False,
        "import_mode": "incremental",
        "full_rebuild_interval": args.full_rebuild_interval,
        "payload": payload_summary,
        "manifest": final_manifest.get("summary", {}),
        "manifest_path": manifest_written,
        "diff": diff,
        "plan": plan,
        "pre_audit": storage_audit,
        "shadow_audit": shadow_audit,
        "shadow_storage": str(shadow_storage),
        "backup_dir": str(backup_dir) if (swapped or prepare_swap) else None,
        "swapped": swapped,
        "prepared_for_swap": prepare_swap,
        "previous_manifest_hash": stable_hash(previous_manifest),
        "desired_manifest_hash": stable_hash(desired_manifest),
        "timings": timings,
    }
    if prepare_swap:
        report_path, desired_manifest_path = write_prepared_swap_bundle(state_dir, desired_manifest, report)
        report["report_path"] = str(report_path)
        report["desired_manifest_path"] = str(desired_manifest_path)
    if swapped:
        report_path = state_dir / REPORT_FILENAME
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["report_path"] = str(report_path)
    return report


def run_full_materialization_no_swap(args: argparse.Namespace) -> dict[str, Any]:
    """Build and audit a full materialized shadow storage directory without swapping live state."""

    from custom_kg_materialize import materialize_file_storage_from_manifest
    from vector_cache import VectorCache, resolve_manifest_vectors, seed_vector_cache_from_storage

    total_started = time.perf_counter()
    timings: dict[str, float] = {}
    root = args.root.resolve()
    state_dir = args.state_dir.resolve()
    workdir = args.workdir.resolve()

    phase_started = time.perf_counter()
    ensure_state_dirs(state_dir)
    load_env_file(workdir / ".env")
    prepare_swap = bool(getattr(args, "prepare_swap", False))
    if prepare_swap and getattr(args, "delete_shadow_on_no_swap", False):
        raise RuntimeError("--prepare-swap cannot be combined with --delete-shadow-on-no-swap; prepared shadow must remain for finalize")
    _record_timing(timings, "load_env_s", phase_started)

    phase_started = time.perf_counter()
    previous_manifest = load_manifest(state_dir)
    _record_timing(timings, "load_previous_manifest_s", phase_started)

    phase_started = time.perf_counter()
    desired_manifest, payload_summary = build_desired_manifest(
        root,
        state_dir,
        limit_docs=getattr(args, "limit_docs", None),
        limit_edges=getattr(args, "limit_edges", None),
    )
    _record_timing(timings, "build_desired_manifest_s", phase_started)

    cache_only_blockers = full_materialization_cache_only_blockers(previous_manifest, desired_manifest)
    if cache_only_blockers.get("blocked"):
        raise RuntimeError(
            "cache-only full materialization is unsafe without an embedding-fill phase; "
            f"new_or_vector_updated_records={cache_only_blockers['total']} "
            f"collections={json.dumps(cache_only_blockers['collections'], ensure_ascii=False, sort_keys=True)}"
        )

    phase_started = time.perf_counter()
    cache_path = _arg_path(args, "vector_cache", state_dir / "vector_cache.sqlite").resolve()
    cache = VectorCache(cache_path)
    vector_seed_report = None
    if getattr(args, "seed_from_storage", False):
        if previous_manifest is None:
            raise RuntimeError("--seed-from-storage requires a previous custom_kg manifest so storage vectors can be matched to their original vector_hash")
        seed_storage_dir = _arg_path(args, "seed_storage_dir", workdir / "rag_storage").resolve()
        vector_seed_report = seed_vector_cache_from_storage(desired_manifest, seed_storage_dir, cache, previous_manifest=previous_manifest)
    _record_timing(timings, "seed_vector_cache_s", phase_started)

    phase_started = time.perf_counter()
    vector_report = resolve_manifest_vectors(desired_manifest, cache)
    _record_timing(timings, "resolve_vector_cache_s", phase_started)
    misses = int(vector_report.get("summary", {}).get("total", {}).get("misses", 0) or 0)
    if misses:
        raise RuntimeError(f"full materialization requires all vectors resolved from cache; misses={misses}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    shadow_storage = _arg_path(args, "storage_dir", workdir / f"rag_storage.full_materialize.{stamp}").resolve()
    started_at = now_stamp()

    phase_started = time.perf_counter()
    materialize_report = materialize_file_storage_from_manifest(desired_manifest, vector_report["resolved"], shadow_storage)
    _record_timing(timings, "materialize_shadow_storage_s", phase_started)

    phase_started = time.perf_counter()
    shadow_audit = audit_custom_kg_storage(shadow_storage, desired_manifest)
    _record_timing(timings, "audit_shadow_storage_s", phase_started)
    if not shadow_audit.get("ok"):
        raise RuntimeError(f"full materialized shadow audit failed: {json.dumps(shadow_audit.get('issues', [])[:10], ensure_ascii=False)}")

    phase_started = time.perf_counter()
    pre_audit = None
    if prepare_swap:
        if previous_manifest is None:
            raise RuntimeError("full materialization prepare-swap requires a current custom_kg manifest")
        live_storage = workdir / "rag_storage"
        if not live_storage.exists():
            raise RuntimeError(f"missing live rag_storage before prepared full materialization: {live_storage}")
        pre_audit = audit_custom_kg_storage(live_storage, previous_manifest)
        if not pre_audit.get("ok"):
            raise RuntimeError(f"live storage audit failed before prepared full materialization: {json.dumps(pre_audit.get('issues', [])[:10], ensure_ascii=False)}")
    _record_timing(timings, "audit_live_storage_s", phase_started)

    phase_started = time.perf_counter()
    query_smoke = None
    smoke_queries = list(getattr(args, "smoke_query", None) or [])
    if smoke_queries:
        query_smoke = run_shadow_query_data_smokes(
            workdir=workdir,
            storage_dir=shadow_storage,
            queries=smoke_queries,
            mode=str(getattr(args, "smoke_mode", "mix") or "mix"),
            top_k=int(getattr(args, "smoke_top_k", 5) or 5),
            chunk_top_k=int(getattr(args, "smoke_chunk_top_k", 5) or 5),
        )
        if not query_smoke.get("ok"):
            raise RuntimeError(f"shadow query smoke failed: {json.dumps(query_smoke, ensure_ascii=False)[:1000]}")
    _record_timing(timings, "query_shadow_storage_s", phase_started)

    phase_started = time.perf_counter()
    shadow_deleted = False
    if getattr(args, "delete_shadow_on_no_swap", False) and not prepare_swap:
        shutil.rmtree(shadow_storage, ignore_errors=True)
        shadow_deleted = True
    _record_timing(timings, "cleanup_shadow_s", phase_started)
    timings["total_s"] = round(time.perf_counter() - total_started, 6)

    report = {
        "started_at": started_at,
        "finished_at": now_stamp(),
        "wiki_root": str(root),
        "workdir": str(workdir),
        "state_dir": str(state_dir),
        "dry_run": False,
        "import_mode": "full_materialization",
        "payload": payload_summary,
        "manifest": desired_manifest.get("summary", {}),
        "manifest_path": None,
        "materialize": materialize_report,
        "vector_cache_path": str(cache_path),
        "vector_cache_seed": vector_seed_report,
        "vector_cache": vector_report,
        "pre_audit": pre_audit,
        "shadow_audit": shadow_audit,
        "query_smoke": query_smoke,
        "shadow_storage": str(shadow_storage),
        "shadow_deleted": shadow_deleted,
        "backup_dir": str(state_dir / "backups" / f"rag_storage_full_materialization_{stamp}") if prepare_swap else None,
        "swapped": False,
        "prepared_for_swap": prepare_swap,
        "previous_manifest_hash": stable_hash(previous_manifest) if previous_manifest is not None else None,
        "desired_manifest_hash": stable_hash(desired_manifest),
        "timings": timings,
    }
    if prepare_swap:
        report_path, desired_manifest_path = write_prepared_swap_bundle(state_dir, desired_manifest, report)
        report["report_path"] = str(report_path)
        report["desired_manifest_path"] = str(desired_manifest_path)
        report["finalize_command"] = [
            "custom_kg_incremental.py",
            "finalize-prepared-swap",
            "--prepared-report",
            str(report_path),
        ]
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report


def run_finalize_prepared_swap(args: argparse.Namespace) -> dict[str, Any]:
    """Swap an already audited prepared shadow into production after live state re-checks."""

    total_started = time.perf_counter()
    timings: dict[str, float] = {}
    root = args.root.resolve()
    state_dir = args.state_dir.resolve()
    workdir = args.workdir.resolve()
    load_env_file(workdir / ".env")
    report_path = args.prepared_report.resolve() if getattr(args, "prepared_report", None) else prepared_swap_report_path(state_dir)
    if not report_path.exists():
        raise RuntimeError(f"missing prepared swap report: {report_path}")
    prepared_report = json.loads(report_path.read_text(encoding="utf-8"))
    if not prepared_report.get("prepared_for_swap"):
        raise RuntimeError(f"prepared swap report is not marked prepared_for_swap: {report_path}")
    if port_open(args.server_host, args.server_port) and not args.allow_server_running:
        raise RuntimeError(f"{args.server_host}:{args.server_port} is listening. Stop lightrag-server before finalizing prepared swap.")

    desired_manifest_path = Path(str(prepared_report.get("desired_manifest_path") or prepared_swap_manifest_path(state_dir))).resolve()
    desired_manifest = load_manifest(desired_manifest_path)
    if desired_manifest is None:
        raise RuntimeError(f"missing prepared desired manifest: {desired_manifest_path}")
    desired_hash = stable_hash(desired_manifest)
    if desired_hash != prepared_report.get("desired_manifest_hash"):
        raise RuntimeError("prepared desired manifest hash mismatch")

    phase_started = time.perf_counter()
    current_manifest = load_manifest(state_dir)
    if current_manifest is None:
        raise RuntimeError("missing current custom_kg manifest before prepared swap finalize")
    current_hash = stable_hash(current_manifest)
    if current_hash != prepared_report.get("previous_manifest_hash"):
        raise RuntimeError("live manifest changed since prepared swap was created")
    _record_timing(timings, "load_and_check_manifest_s", phase_started)

    live_storage = workdir / "rag_storage"
    shadow_storage = Path(str(prepared_report.get("shadow_storage", ""))).resolve()
    backup_dir = Path(str(prepared_report.get("backup_dir") or (state_dir / "backups" / f"rag_storage_incremental_finalize_{time.strftime('%Y%m%d_%H%M%S')}"))).resolve()
    if not live_storage.exists():
        raise RuntimeError(f"missing live rag_storage: {live_storage}")
    if not shadow_storage.exists():
        raise RuntimeError(f"missing prepared shadow storage: {shadow_storage}")

    phase_started = time.perf_counter()
    live_audit = audit_custom_kg_storage(live_storage, current_manifest)
    _record_timing(timings, "audit_live_storage_s", phase_started)
    if not live_audit.get("ok"):
        raise RuntimeError(f"live storage audit failed before prepared swap: {json.dumps(live_audit.get('issues', [])[:10], ensure_ascii=False)}")

    phase_started = time.perf_counter()
    shadow_audit = audit_custom_kg_storage(shadow_storage, desired_manifest)
    _record_timing(timings, "audit_shadow_storage_s", phase_started)
    if not shadow_audit.get("ok"):
        raise RuntimeError(f"prepared shadow storage audit failed: {json.dumps(shadow_audit.get('issues', [])[:10], ensure_ascii=False)}")

    phase_started = time.perf_counter()
    backup_dir.parent.mkdir(parents=True, exist_ok=True)
    _safe_remove(backup_dir)
    swapped = False
    try:
        live_storage.rename(backup_dir)
        shadow_storage.rename(live_storage)
        swapped = True
    except Exception:
        if not live_storage.exists() and backup_dir.exists():
            backup_dir.rename(live_storage)
        raise
    _record_timing(timings, "swap_shadow_to_live_s", phase_started)

    phase_started = time.perf_counter()
    prepared_import_mode = str(prepared_report.get("import_mode") or "incremental")
    final_manifest = successful_manifest(desired_manifest, import_mode=prepared_import_mode, previous_manifest=current_manifest)
    manifest_written = str(write_manifest(state_dir, final_manifest))
    _record_timing(timings, "write_manifest_s", phase_started)
    timings["total_s"] = round(time.perf_counter() - total_started, 6)

    final_report = {
        "started_at": prepared_report.get("started_at"),
        "finished_at": now_stamp(),
        "wiki_root": str(root),
        "workdir": str(workdir),
        "state_dir": str(state_dir),
        "dry_run": False,
        "import_mode": prepared_import_mode,
        "full_rebuild_interval": prepared_report.get("full_rebuild_interval"),
        "payload": prepared_report.get("payload", {}),
        "manifest": final_manifest.get("summary", {}),
        "manifest_path": manifest_written,
        "diff": prepared_report.get("diff", {}),
        "plan": prepared_report.get("plan", {}),
        "pre_audit": prepared_report.get("pre_audit", {}),
        "live_audit": live_audit,
        "shadow_audit": shadow_audit,
        "prepared_report_path": str(report_path),
        "shadow_storage": str(shadow_storage),
        "backup_dir": str(backup_dir),
        "swapped": swapped,
        "prepared_for_swap": False,
        "finalized_prepared_swap": True,
        "prepare_timings": prepared_report.get("timings", {}),
        "timings": timings,
    }
    import_report_path = state_dir / REPORT_FILENAME
    import_report_path.write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")
    final_report["report_path"] = str(import_report_path)

    prepared_report["finalized"] = True
    prepared_report["finalized_at"] = final_report["finished_at"]
    prepared_report["final_report_path"] = str(import_report_path)
    report_path.write_text(json.dumps(prepared_report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return final_report


# ---------------------------------------------------------------------------
# CLI


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=DEFAULT_WIKI_ROOT)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)


def main() -> int:
    parser = argparse.ArgumentParser(description="Safe incremental custom_kg planner/apply/audit for llm-wiki LightRAG")
    sub = parser.add_subparsers(dest="command", required=True)

    plan_parser = sub.add_parser("plan", help="Build desired manifest and decide full vs incremental without mutation")
    add_common_paths(plan_parser)
    plan_parser.add_argument("--full-rebuild-interval", type=int, default=DEFAULT_FULL_REBUILD_INTERVAL)

    audit_parser = sub.add_parser("audit-storage", help="Audit rag_storage consistency, optionally against the saved manifest")
    add_common_paths(audit_parser)
    audit_parser.add_argument("--manifest", type=Path, default=None)
    audit_parser.add_argument("--storage-dir", type=Path, default=None)

    apply_parser = sub.add_parser("apply", help="Patch a shadow copy and swap it into rag_storage after audit")
    add_common_paths(apply_parser)
    apply_parser.add_argument("--full-rebuild-interval", type=int, default=DEFAULT_FULL_REBUILD_INTERVAL)
    apply_parser.add_argument("--limit-docs", type=int, default=None)
    apply_parser.add_argument("--limit-edges", type=int, default=None)
    apply_parser.add_argument("--server-host", default="127.0.0.1")
    apply_parser.add_argument("--server-port", type=int, default=9621)
    apply_parser.add_argument("--allow-server-running", action="store_true")
    apply_parser.add_argument("--force-incremental", action="store_true")
    apply_parser.add_argument("--no-swap", action="store_true", help="Patch and audit shadow storage but do not swap it into production")
    apply_parser.add_argument("--prepare-swap", action="store_true", help="Patch and audit shadow storage while service may remain live; persist a prepared swap bundle for finalize-prepared-swap")
    apply_parser.add_argument("--delete-shadow-on-no-swap", action="store_true")
    apply_parser.add_argument("--write-manifest-without-swap", action="store_true")
    apply_parser.add_argument("--tracking-update-mode", choices=["full", "delta"], default="full", help="How to update entity/relation chunk tracking stores inside the audited shadow copy")

    finalize_parser = sub.add_parser("finalize-prepared-swap", help="Swap a previously prepared audited shadow into production and write the successful manifest/report")
    add_common_paths(finalize_parser)
    finalize_parser.add_argument("--prepared-report", type=Path, default=None)
    finalize_parser.add_argument("--server-host", default="127.0.0.1")
    finalize_parser.add_argument("--server-port", type=int, default=9621)
    finalize_parser.add_argument("--allow-server-running", action="store_true")

    materialize_parser = sub.add_parser("materialize-full", help="Materialize and audit a full shadow storage directory without swapping live rag_storage")
    add_common_paths(materialize_parser)
    materialize_parser.add_argument("--limit-docs", type=int, default=None)
    materialize_parser.add_argument("--limit-edges", type=int, default=None)
    materialize_parser.add_argument("--vector-cache", type=Path, default=None)
    materialize_parser.add_argument("--storage-dir", type=Path, default=None)
    materialize_parser.add_argument("--seed-from-storage", action="store_true", help="Seed the vector cache from an explicit file-backend storage directory before resolving vectors")
    materialize_parser.add_argument("--seed-storage-dir", type=Path, default=None, help="Storage directory used with --seed-from-storage; defaults to <workdir>/rag_storage")
    materialize_parser.add_argument("--smoke-query", action="append", default=[], help="Run a direct LightRAG aquery_data smoke against the materialized shadow before optional cleanup; repeat for multiple queries")
    materialize_parser.add_argument("--smoke-mode", default="mix", choices=["local", "global", "hybrid", "naive", "mix", "bypass"])
    materialize_parser.add_argument("--smoke-top-k", type=int, default=5)
    materialize_parser.add_argument("--smoke-chunk-top-k", type=int, default=5)
    materialize_parser.add_argument("--no-swap", action="store_true", required=True, help="Required guard: this command only materializes shadow storage; use --prepare-swap to persist a finalizeable bundle")
    materialize_parser.add_argument("--prepare-swap", action="store_true", help="Persist an audited full-materialization prepared swap bundle; does not swap live rag_storage")
    materialize_parser.add_argument("--delete-shadow-on-no-swap", action="store_true")

    args = parser.parse_args()
    try:
        if args.command == "plan":
            print_json(plan_incremental_import(args.root.resolve(), args.state_dir.resolve(), args.workdir.resolve(), full_rebuild_interval=args.full_rebuild_interval))
            return 0
        if args.command == "audit-storage":
            state_dir = args.state_dir.resolve()
            workdir = args.workdir.resolve()
            manifest = load_manifest(args.manifest.resolve()) if args.manifest else load_manifest(state_dir)
            storage_dir = args.storage_dir.resolve() if args.storage_dir else workdir / "rag_storage"
            audit = audit_custom_kg_storage(storage_dir, manifest)
            print_json(audit)
            return 0 if audit.get("ok") else 1
        if args.command == "apply":
            print_json(asyncio.run(run_apply(args)))
            return 0
        if args.command == "finalize-prepared-swap":
            print_json(run_finalize_prepared_swap(args))
            return 0
        if args.command == "materialize-full":
            print_json(run_full_materialization_no_swap(args))
            return 0
    except Exception as exc:
        print_json({"error": type(exc).__name__, "message": str(exc)})
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
