#!/usr/bin/env python3
"""Native deterministic artifact builders for llm-wiki."""

from __future__ import annotations

import functools
import os
import re
from pathlib import Path
from typing import Any

from wiki_native_docs import (
    WikiDoc,
    collect_source_docs,
    display_scalar,
    generated_doc_filename,
    make_wiki_doc,
    section_text,
)
from wiki_native_ingest_text import as_list, find_wikilinks, first_sentences
from wiki_native_jsonl import jsonl_write
from wiki_native_query_events import slugify
from wiki_native_state import ensure_state_dirs

WIKI_SOURCE_ROOT_PREFIXES = frozenset({"raw", "entities", "concepts", "comparisons", "queries", "_meta"})


def _lexical_norm(path: Path | str) -> str:
    """Normalize a path lexically without touching the filesystem."""

    return os.path.normpath(os.path.abspath(os.fspath(path)))


def _is_lexically_under(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([root, path]) == root
    except ValueError:
        return False


@functools.lru_cache(maxsize=20000)
def _resolve_source_cached(page_dir: str, source: str, root_str: str | None) -> str | None:
    source = source.strip()
    if not source or source.startswith("http://") or source.startswith("https://"):
        return None

    root_norm = _lexical_norm(root_str) if root_str else None
    source_path = Path(source)
    if source_path.is_absolute():
        candidate = _lexical_norm(source_path)
    else:
        first_segment = source.split("/", 1)[0]
        base = Path(root_norm) if root_norm is not None and first_segment in WIKI_SOURCE_ROOT_PREFIXES else Path(page_dir)
        candidate = _lexical_norm(base / source)

    if root_norm is not None and not _is_lexically_under(candidate, root_norm):
        return None
    return candidate


def resolve_source(page: Path, source: str, root: Path | None = None) -> Path | None:
    root_str = os.fspath(root) if root is not None else None
    resolved = _resolve_source_cached(os.fspath(page.parent), source.strip(), root_str)
    return Path(resolved) if resolved else None


def _write_text_if_changed(path: Path, content: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def bullet_items(text: str, limit: int = 5) -> list[str]:
    items = []
    for line in text.splitlines():
        s = re.sub(r"^[-*]\s+", "", line.strip())
        if s and len(s) > 8:
            items.append(s[:240])
        if len(items) >= limit:
            break
    if not items and text.strip():
        items.append(first_sentences(text, 240))
    return items


def method_type_for(doc: WikiDoc) -> str:
    hay = " ".join([doc.title, display_scalar(doc.frontmatter.get("tags")), display_scalar(doc.frontmatter.get("topic_hints")), doc.body[:2000]]).lower()
    if any(k in hay for k in ["retrieval", "rag", "search"]):
        return "retrieval_or_memory_design"
    if any(k in hay for k in ["evaluation", "benchmark", "judge"]):
        return "evaluation_protocol"
    if any(k in hay for k in ["training", "rl", "reward", "distillation"]):
        return "training_signal_construction"
    if any(k in hay for k in ["agent", "workflow", "harness"]):
        return "agent_workflow_design"
    return "methodological_pattern"


def extract_method_atoms(root: Path, state_dir: Path) -> dict[str, Any]:
    ensure_state_dirs(state_dir)
    rows = []
    docs_dir = state_dir / "method_atom_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for stale in docs_dir.glob("*.md"):
        stale.unlink()
    for doc in collect_source_docs(root):
        if doc.doc_type != "raw_note":
            continue
        methodology = section_text(doc.body, ["methodology", "方法"])
        if not methodology or len(methodology) < 80:
            continue
        limitations = section_text(doc.body, ["局限", "limitation", "failure"])
        future = section_text(doc.body, ["未来", "启发", "future", "question"])
        atom_id = f"method:{slugify(doc.path.stem, 100)}:001"
        atom = {
            "atom_id": atom_id,
            "source_path": doc.rel_path,
            "paper_title": doc.title,
            "method_name": doc.title,
            "method_type": method_type_for(doc),
            "problem_form": first_sentences(section_text(doc.body, ["motivation", "问题", "background"]) or doc.body, 300),
            "core_mechanism": first_sentences(methodology, 700),
            "transferable_unit": method_type_for(doc).replace("_", " "),
            "assumptions": [],
            "failure_modes": bullet_items(limitations, 3),
            "possible_transfer_targets": bullet_items(future, 3),
            "evidence_headings": ["## Methodology"],
            "confidence": 0.62,
            "created_by": "extract_method_atoms.py",
        }
        rows.append(atom)
        md = method_atom_markdown(atom)
        (docs_dir / generated_doc_filename(atom_id, 120)).write_text(md, encoding="utf-8")
    jsonl = state_dir / "method_atoms.jsonl"
    count = jsonl_write(jsonl, rows)
    return {"method_atoms": count, "jsonl": str(jsonl), "docs_dir": str(docs_dir)}


def method_atom_markdown(atom: dict[str, Any]) -> str:
    lines = [
        "[LLM_WIKI_METHOD_ATOM]",
        f"atom_id: {atom['atom_id']}",
        f"source_path: {atom['source_path']}",
        f"paper_title: {atom['paper_title']}",
        f"method_name: {atom['method_name']}",
        f"method_type: {atom['method_type']}",
        f"transferable_unit: {atom['transferable_unit']}",
        f"confidence: {atom['confidence']}",
        "[/LLM_WIKI_METHOD_ATOM]",
        "",
        f"# MethodAtom: {atom['method_name']}",
        "",
        "## Problem Form",
        atom.get("problem_form") or "Not specified.",
        "",
        "## Core Mechanism",
        atom.get("core_mechanism") or "Not specified.",
        "",
        "## Transferable Unit",
        atom.get("transferable_unit") or "Not specified.",
        "",
        "## Failure Modes",
    ]
    for item in atom.get("failure_modes") or ["Not specified."]:
        lines.append(f"- {item}")
    lines.extend(["", "## Possible Transfer Targets"])
    for item in atom.get("possible_transfer_targets") or ["Not specified."]:
        lines.append(f"- {item}")
    lines.extend(["", "## Evidence Source", atom["source_path"], ""])
    return "\n".join(lines)


def build_seed_edges(root: Path, state_dir: Path) -> dict[str, Any]:
    ensure_state_dirs(state_dir)
    docs = collect_source_docs(root)
    by_rel = {d.rel_path: d for d in docs}
    by_stem = {Path(d.rel_path).stem: d for d in docs}
    edges: list[dict[str, Any]] = []

    def add(src: str, tgt: str, typ: str, weight: float, evidence_path: str, field: str) -> None:
        edge_id = f"edge:{typ.lower()}:{len(edges)+1:06d}"
        edges.append(
            {
                "edge_id": edge_id,
                "src_id": src,
                "tgt_id": tgt,
                "type": typ,
                "weight": weight,
                "evidence_path": evidence_path,
                "evidence_field": field,
                "created_by": "build_seed_edges.py",
            }
        )

    for doc in docs:
        if not doc.doc_type.startswith("compiled_") and doc.doc_type != "meta_map":
            continue
        for link in find_wikilinks(doc.text):
            target = by_stem.get(link)
            if target:
                add(doc.canonical_id, target.canonical_id, "WIKILINKS_TO", 1.0, doc.rel_path, "body.wikilinks")
        for src in as_list(doc.frontmatter.get("sources")):
            s = display_scalar(src)
            if not s or s.startswith("http://") or s.startswith("https://"):
                continue
            resolved = resolve_source(doc.path, s, root)
            if resolved and resolved.exists():
                rel = resolved.relative_to(root).as_posix()
                target = by_rel.get(rel) or make_wiki_doc(root, resolved)
                add(doc.canonical_id, target.canonical_id, "SOURCED_BY", 1.0, doc.rel_path, "frontmatter.sources")
        for tag in as_list(doc.frontmatter.get("tags")):
            tag_s = display_scalar(tag)
            if tag_s:
                add(doc.canonical_id, f"tag:{tag_s}", "HAS_TAG", 0.7, doc.rel_path, "frontmatter.tags")
    jsonl = state_dir / "seed_edges.jsonl"
    count = jsonl_write(jsonl, edges)
    docs_dir = state_dir / "edge_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    desired_docs = {generated_doc_filename(edge["edge_id"], 120): edge_markdown(edge) for edge in edges}
    for stale in docs_dir.glob("*.md"):
        if stale.name not in desired_docs:
            stale.unlink()
    edge_docs_written = 0
    for filename, content in desired_docs.items():
        if _write_text_if_changed(docs_dir / filename, content):
            edge_docs_written += 1
    return {"seed_edges": count, "jsonl": str(jsonl), "docs_dir": str(docs_dir), "edge_docs_total": len(desired_docs), "edge_docs_written": edge_docs_written}


def edge_markdown(edge: dict[str, Any]) -> str:
    return "\n".join(
        [
            "[LLM_WIKI_EDGE_DOC]",
            f"edge_id: {edge['edge_id']}",
            f"edge_type: {edge['type']}",
            f"source: {edge['src_id']}",
            f"target: {edge['tgt_id']}",
            f"evidence_path: {edge['evidence_path']}",
            f"evidence_field: {edge['evidence_field']}",
            f"confidence: {edge['weight']}",
            "[/LLM_WIKI_EDGE_DOC]",
            "",
            f"The llm-wiki document {edge['src_id']} has a high-confidence {edge['type']} relation to {edge['tgt_id']}.",
            f"Evidence: {edge['evidence_path']} ({edge['evidence_field']}).",
            "",
        ]
    )
