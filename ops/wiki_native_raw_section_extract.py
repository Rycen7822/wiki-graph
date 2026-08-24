#!/usr/bin/env python3
"""Native raw-section extraction and materialization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_wiki_native.source_docs import collect_source_docs, generated_doc_filename, markdown_sections
from ops.wiki_native_raw_sections import raw_section_specs_for_heading
from ops.wiki_native_state import ensure_state_dirs


def extract_raw_note_sections(doc: Any) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    seen: set[str] = set()
    for title, text in markdown_sections(doc.body):
        specs = raw_section_specs_for_heading(title)
        if not specs or not text.strip():
            continue
        for spec in specs:
            kind = str(spec["kind"])
            if kind in seen:
                continue
            seen.add(kind)
            section_id = f"raw_section:{doc.path.stem}:{kind}"
            sections.append(
                {
                    "section_id": section_id,
                    "source_id": doc.canonical_id,
                    "source_path": doc.rel_path,
                    "paper_title": doc.title,
                    "section_kind": kind,
                    "section_title": title.strip() or spec["canonical_title"],
                    "canonical_section_title": spec["canonical_title"],
                    "content": text.strip(),
                    "created_by": "extract_raw_sections.py",
                }
            )
    return sections


def raw_section_markdown(section: dict[str, Any]) -> str:
    return "\n".join(
        [
            "[LLM_WIKI_RAW_SECTION]",
            f"section_id: {section['section_id']}",
            f"source_id: {section['source_id']}",
            f"source_path: {section['source_path']}",
            f"paper_title: {section['paper_title']}",
            f"section_kind: {section['section_kind']}",
            f"section_title: {section['section_title']}",
            f"canonical_section_title: {section['canonical_section_title']}",
            "[/LLM_WIKI_RAW_SECTION]",
            "",
            f"# RawSection: {section['paper_title']} / {section['section_title']}",
            "",
            "## Section Content",
            section.get("content") or "Not specified.",
            "",
            "## Evidence Source",
            section["source_path"],
            "",
        ]
    )


def extract_raw_sections(root: Path, state_dir: Path, *, docs: list | None = None) -> dict[str, Any]:
    from ops.wiki_native_lib import jsonl_write

    ensure_state_dirs(state_dir)
    rows: list[dict[str, Any]] = []
    docs_dir = state_dir / "raw_section_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for stale in docs_dir.glob("*.md"):
        stale.unlink()
    for doc in (docs if docs is not None else collect_source_docs(root)):
        if doc.doc_type != "raw_note":
            continue
        for section in extract_raw_note_sections(doc):
            rows.append(section)
            filename = generated_doc_filename(section["section_id"], 120)
            (docs_dir / filename).write_text(raw_section_markdown(section), encoding="utf-8")
    jsonl = state_dir / "raw_sections.jsonl"
    count = jsonl_write(jsonl, rows)
    by_kind: dict[str, int] = {}
    for row in rows:
        kind = str(row.get("section_kind", ""))
        by_kind[kind] = by_kind.get(kind, 0) + 1
    return {"raw_sections": count, "by_kind": by_kind, "jsonl": str(jsonl), "docs_dir": str(docs_dir)}
