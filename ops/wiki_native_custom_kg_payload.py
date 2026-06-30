#!/usr/bin/env python3
"""Native-owned deterministic custom KG payload assembly."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ops.wiki_native_docs import WikiDoc, canonical_id_for, collect_source_docs, display_scalar, generated_docs_from_state
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
    doc_ids = {doc.canonical_id for doc in docs}
    chunks: list[dict[str, Any]] = []
    entities: dict[str, dict[str, Any]] = {}
    relationships: list[dict[str, Any]] = []
    for doc in docs:
        chunks.append(
            {
                "content": make_ingest_text(doc),
                "source_id": doc.canonical_id,
                "file_path": doc.rel_path,
                "chunk_order_index": 0,
            }
        )
        entities[doc.canonical_id] = {
            "entity_name": doc.canonical_id,
            "entity_type": custom_kg_entity_type(doc.canonical_id),
            "description": custom_kg_doc_description(doc),
            "source_id": doc.canonical_id,
            "file_path": doc.rel_path,
        }

    fallback_source = docs[0].canonical_id if docs else "UNKNOWN"
    for method_doc in generated_docs_from_state(state_dir, kind="method_atom"):
        chunks.append(
            {
                "content": method_doc.text,
                "source_id": method_doc.canonical_id,
                "file_path": method_doc.rel_path,
                "chunk_order_index": 0,
            }
        )
        doc_ids.add(method_doc.canonical_id)
        entities[method_doc.canonical_id] = {
            "entity_name": method_doc.canonical_id,
            "entity_type": custom_kg_entity_type(method_doc.canonical_id),
            "description": f"MethodAtom {method_doc.title}",
            "source_id": method_doc.canonical_id,
            "file_path": method_doc.rel_path,
        }
        m = re.search(r"^source_path:\s*(.+)$", method_doc.text, flags=re.M)
        if m:
            source_rel = m.group(1).strip()
            source_path = root / source_rel
            if source_path.exists():
                source_id = canonical_id_for(root, source_path)
                entities.setdefault(
                    source_id,
                    {
                        "entity_name": source_id,
                        "entity_type": custom_kg_entity_type(source_id),
                        "description": f"Source for {method_doc.canonical_id}",
                        "source_id": source_id if source_id in doc_ids else fallback_source,
                        "file_path": source_rel,
                    },
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

    raw_section_chunk_count = 0
    for section_doc in generated_docs_from_state(state_dir, kind="raw_section"):
        raw_section_chunk_count += 1
        chunks.append(
            {
                "content": section_doc.text,
                "source_id": section_doc.canonical_id,
                "file_path": section_doc.rel_path,
                "chunk_order_index": 0,
            }
        )
        doc_ids.add(section_doc.canonical_id)
        entities[section_doc.canonical_id] = {
            "entity_name": section_doc.canonical_id,
            "entity_type": custom_kg_entity_type(section_doc.canonical_id),
            "description": f"RawSection {section_doc.title}",
            "source_id": section_doc.canonical_id,
            "file_path": section_doc.rel_path,
        }
        source_id_match = re.search(r"^source_id:\s*(.+)$", section_doc.text, flags=re.M)
        source_path_match = re.search(r"^source_path:\s*(.+)$", section_doc.text, flags=re.M)
        section_kind_match = re.search(r"^section_kind:\s*(.+)$", section_doc.text, flags=re.M)
        source_id = source_id_match.group(1).strip() if source_id_match else ""
        source_rel = source_path_match.group(1).strip() if source_path_match else ""
        section_kind = section_kind_match.group(1).strip() if section_kind_match else "section"
        if source_id:
            entities.setdefault(
                source_id,
                {
                    "entity_name": source_id,
                    "entity_type": custom_kg_entity_type(source_id),
                    "description": f"Source for {section_doc.canonical_id}",
                    "source_id": source_id if source_id in doc_ids else fallback_source,
                    "file_path": source_rel or section_doc.rel_path,
                },
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

    section_similarity_relationship_count = 0
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
                {
                    "entity_name": node,
                    "entity_type": custom_kg_entity_type(node),
                    "description": f"Semantic section neighbor endpoint {edge.get(title_key, node)}",
                    "source_id": node if node in doc_ids else fallback_source,
                    "file_path": str(edge.get(path_key, "section_similarity_edges.jsonl")),
                },
            )
        relationships.append(section_similarity_edge_to_custom_kg_relationship(edge))
        section_similarity_relationship_count += 1

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
                {
                    "entity_name": node,
                    "entity_type": custom_kg_entity_type(node),
                    "description": f"Deterministic llm-wiki node {node}",
                    "source_id": src if src in doc_ids else (tgt if tgt in doc_ids else fallback_source),
                    "file_path": evidence_path,
                },
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
