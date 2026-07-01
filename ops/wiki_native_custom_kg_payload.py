#!/usr/bin/env python3
"""Native-owned deterministic custom KG payload assembly."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from llm_wiki_native.source_docs import WikiDoc, canonical_id_for, collect_source_docs, display_scalar, generated_docs_from_state
from ops.wiki_native_ingest_text import make_ingest_text
from ops.wiki_native_jsonl import jsonl_read
from ops.wiki_native_section_similarity import section_similarity_edge_to_custom_kg_relationship


def custom_kg_entity_type(node_id: str) -> str:
    if node_id.startswith("compiled:"):
        return "LLM_WIKI_PAGE"
    if node_id.startswith("raw_section:"):
        return "LLM_WIKI_RAW_SECTION"
    if node_id.startswith("raw_clip:") or node_id.startswith("raw:"):
        return "LLM_WIKI_RAW"
    if node_id.startswith("tag:"):
        return "LLM_WIKI_TAG"
    if node_id.startswith("method_atom:") or node_id.startswith("method:"):
        return "LLM_WIKI_METHOD_ATOM"
    return "LLM_WIKI_NODE"


def custom_kg_doc_description(doc: WikiDoc) -> str:
    parts = [doc.title, doc.rel_path, doc.doc_type]
    tags = display_scalar(doc.frontmatter.get("tags"))
    if tags:
        parts.append(f"tags={tags}")
    topic_hints = display_scalar(doc.frontmatter.get("topic_hints"))
    if topic_hints:
        parts.append(f"topic_hints={topic_hints}")
    updated = display_scalar(doc.frontmatter.get("updated"))
    if updated:
        parts.append(f"updated={updated}")
    return " | ".join(part for part in parts if part)[:900]


def _chunk_record(content: str, source_id: str, file_path: str) -> dict[str, Any]:
    return {
        "content": content,
        "source_id": source_id,
        "file_path": file_path,
        "chunk_order_index": 0,
    }


def _entity_record(entity_name: str, description: str, source_id: str, file_path: str) -> dict[str, Any]:
    return {
        "entity_name": entity_name,
        "entity_type": custom_kg_entity_type(entity_name),
        "description": description,
        "source_id": source_id,
        "file_path": file_path,
    }


def _add_source_doc_records(
    docs: list[WikiDoc],
    chunks: list[dict[str, Any]],
    entities: dict[str, dict[str, Any]],
) -> set[str]:
    doc_ids = {doc.canonical_id for doc in docs}
    for doc in docs:
        chunks.append(_chunk_record(make_ingest_text(doc), doc.canonical_id, doc.rel_path))
        entities[doc.canonical_id] = _entity_record(
            doc.canonical_id,
            custom_kg_doc_description(doc),
            doc.canonical_id,
            doc.rel_path,
        )
    return doc_ids


def _add_method_atom_records(
    root: Path,
    state_dir: Path,
    chunks: list[dict[str, Any]],
    entities: dict[str, dict[str, Any]],
    relationships: list[dict[str, Any]],
    doc_ids: set[str],
    fallback_source: str,
) -> None:
    for method_doc in generated_docs_from_state(state_dir, kind="method_atom"):
        chunks.append(_chunk_record(method_doc.text, method_doc.canonical_id, method_doc.rel_path))
        doc_ids.add(method_doc.canonical_id)
        entities[method_doc.canonical_id] = _entity_record(
            method_doc.canonical_id,
            f"MethodAtom {method_doc.title}",
            method_doc.canonical_id,
            method_doc.rel_path,
        )
        source_match = re.search(r"^source_path:\s*(.+)$", method_doc.text, flags=re.M)
        if not source_match:
            continue
        source_rel = source_match.group(1).strip()
        source_path = root / source_rel
        if not source_path.exists():
            continue
        source_id = canonical_id_for(root, source_path)
        entities.setdefault(
            source_id,
            _entity_record(
                source_id,
                f"Source for {method_doc.canonical_id}",
                source_id if source_id in doc_ids else fallback_source,
                source_rel,
            ),
        )
        relationships.append(
            {
                "src_id": method_doc.canonical_id,
                "tgt_id": source_id,
                "description": f"{method_doc.canonical_id} METHOD_ATOM_FROM {source_id}; evidence: {method_doc.rel_path}.",
                "keywords": "METHOD_ATOM_FROM",
                "source_id": method_doc.canonical_id,
                "weight": 0.9,
                "file_path": method_doc.rel_path,
            }
        )


def _add_raw_section_records(
    state_dir: Path,
    chunks: list[dict[str, Any]],
    entities: dict[str, dict[str, Any]],
    relationships: list[dict[str, Any]],
    doc_ids: set[str],
    fallback_source: str,
) -> int:
    raw_section_chunk_count = 0
    for section_doc in generated_docs_from_state(state_dir, kind="raw_section"):
        raw_section_chunk_count += 1
        chunks.append(_chunk_record(section_doc.text, section_doc.canonical_id, section_doc.rel_path))
        doc_ids.add(section_doc.canonical_id)
        entities[section_doc.canonical_id] = _entity_record(
            section_doc.canonical_id,
            f"RawSection {section_doc.title}",
            section_doc.canonical_id,
            section_doc.rel_path,
        )
        source_id_match = re.search(r"^source_id:\s*(.+)$", section_doc.text, flags=re.M)
        source_path_match = re.search(r"^source_path:\s*(.+)$", section_doc.text, flags=re.M)
        section_kind_match = re.search(r"^section_kind:\s*(.+)$", section_doc.text, flags=re.M)
        source_id = source_id_match.group(1).strip() if source_id_match else ""
        source_rel = source_path_match.group(1).strip() if source_path_match else ""
        section_kind = section_kind_match.group(1).strip() if section_kind_match else "section"
        if not source_id:
            continue
        entities.setdefault(
            source_id,
            _entity_record(
                source_id,
                f"Source for {section_doc.canonical_id}",
                source_id if source_id in doc_ids else fallback_source,
                source_rel or section_doc.rel_path,
            ),
        )
        relationships.append(
            {
                "src_id": section_doc.canonical_id,
                "tgt_id": source_id,
                "description": f"{section_doc.canonical_id} RAW_SECTION_OF {source_id}; section_kind: {section_kind}; evidence: {section_doc.rel_path}.",
                "keywords": "RAW_SECTION_OF",
                "source_id": section_doc.canonical_id,
                "weight": 0.85,
                "file_path": section_doc.rel_path,
            }
        )
    return raw_section_chunk_count


def _add_section_similarity_records(
    state_dir: Path,
    entities: dict[str, dict[str, Any]],
    relationships: list[dict[str, Any]],
    doc_ids: set[str],
    fallback_source: str,
) -> int:
    relationship_count = 0
    for edge in jsonl_read(state_dir / "section_similarity_edges.jsonl"):
        src = str(edge.get("src_id", "")).strip()
        tgt = str(edge.get("tgt_id", "")).strip()
        if not src or not tgt:
            continue
        for node, path_key, title_key in [
            (src, "source_path", "source_title"),
            (tgt, "target_path", "target_title"),
        ]:
            entities.setdefault(
                node,
                _entity_record(
                    node,
                    f"Semantic section neighbor endpoint {edge.get(title_key, node)}",
                    node if node in doc_ids else fallback_source,
                    str(edge.get(path_key, "section_similarity_edges.jsonl")),
                ),
            )
        relationships.append(section_similarity_edge_to_custom_kg_relationship(edge))
        relationship_count += 1
    return relationship_count


def _add_seed_edge_records(
    state_dir: Path,
    entities: dict[str, dict[str, Any]],
    relationships: list[dict[str, Any]],
    doc_ids: set[str],
    fallback_source: str,
    limit_edges: int | None = None,
) -> None:
    seed_edges = jsonl_read(state_dir / "seed_edges.jsonl")
    if limit_edges is not None:
        seed_edges = seed_edges[:limit_edges]
    for edge in seed_edges:
        src = str(edge.get("src_id", "")).strip()
        tgt = str(edge.get("tgt_id", "")).strip()
        if not src or not tgt:
            continue
        evidence_path = str(edge.get("evidence_path", "custom_kg"))
        for node in (src, tgt):
            entities.setdefault(
                node,
                _entity_record(
                    node,
                    f"Deterministic llm-wiki node {node}",
                    src if src in doc_ids else (tgt if tgt in doc_ids else fallback_source),
                    evidence_path,
                ),
            )
        source_id = src if src in doc_ids else (tgt if tgt in doc_ids else fallback_source)
        edge_type = str(edge.get("type", "RELATED_TO"))
        relationships.append(
            {
                "src_id": src,
                "tgt_id": tgt,
                "description": f"{src} {edge_type} {tgt}; evidence: {evidence_path} ({edge.get('evidence_field', '')}).",
                "keywords": edge_type,
                "source_id": source_id,
                "weight": float(edge.get("weight", 1.0)),
                "file_path": evidence_path,
            }
        )


def build_custom_kg_payload(
    root: Path,
    state_dir: Path,
    limit_docs: int | None = None,
    limit_edges: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a deterministic custom_kg payload without touching the wiki root."""
    docs = collect_source_docs(root)
    if limit_docs is not None:
        docs = docs[:limit_docs]
    chunks: list[dict[str, Any]] = []
    entities: dict[str, dict[str, Any]] = {}
    relationships: list[dict[str, Any]] = []

    doc_ids = _add_source_doc_records(docs, chunks, entities)
    fallback_source = docs[0].canonical_id if docs else "UNKNOWN"
    _add_method_atom_records(root, state_dir, chunks, entities, relationships, doc_ids, fallback_source)
    raw_section_chunk_count = _add_raw_section_records(state_dir, chunks, entities, relationships, doc_ids, fallback_source)
    section_similarity_relationship_count = _add_section_similarity_records(
        state_dir,
        entities,
        relationships,
        doc_ids,
        fallback_source,
    )
    _add_seed_edge_records(state_dir, entities, relationships, doc_ids, fallback_source, limit_edges)

    payload = {"chunks": chunks, "entities": list(entities.values()), "relationships": relationships}
    summary = {
        "chunks": len(chunks),
        "raw_section_chunks": raw_section_chunk_count,
        "section_similarity_relationships": section_similarity_relationship_count,
        "entities": len(entities),
        "relationships": len(relationships),
        "limit_docs": limit_docs,
        "limit_edges": limit_edges,
    }
    return payload, summary
