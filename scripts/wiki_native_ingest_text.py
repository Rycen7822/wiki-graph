#!/usr/bin/env python3
"""Native wiki ingest-text rendering helpers."""

from __future__ import annotations

import re
from typing import Any

from wiki_native_docs import WikiDoc, display_scalar, section_text


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def find_wikilinks(text: str) -> list[str]:
    out: list[str] = []
    for raw in re.findall(r"\[\[([^\]]+)\]\]", text):
        target = raw.split("|", 1)[0].split("#", 1)[0].strip()
        if target:
            out.append(target)
    return out


def source_urls(frontmatter: dict[str, Any]) -> list[str]:
    keys = ["source", "url", "doi", "project", "github", "official_code", "source_code", "paper", "arxiv"]
    urls: list[str] = []
    for key in keys:
        for value in as_list(frontmatter.get(key)):
            s = display_scalar(value)
            if s and (s.startswith("http://") or s.startswith("https://")):
                urls.append(s)
    return sorted(set(urls))


def first_sentences(text: str, max_chars: int = 600) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned[:max_chars].rstrip()


def compact_body_for_ingest(doc: WikiDoc) -> str:
    """Keep graph-ingest text bounded while leaving canonical Markdown untouched."""
    if len(doc.text) <= 2500:
        return doc.text
    if doc.doc_type == "compiled_index":
        lines = [line for line in doc.text.splitlines() if line.startswith("#") or line.startswith(">")]
        return "\n".join(lines)[:900]
    if doc.doc_type == "raw_note":
        parts = []
        for title, keys in [
            ("Summary", ["一句话", "摘要", "summary"]),
            ("Motivation", ["motivation", "问题", "背景"]),
            ("Methodology", ["methodology", "方法"]),
            ("Findings", ["关键实验", "结论", "result"]),
            ("Limitations", ["局限", "limitation"]),
            ("Future", ["启发", "未来", "追问", "future"]),
        ]:
            section = section_text(doc.body, keys)
            if section:
                parts.append(f"## {title}\n{first_sentences(section, 350)}")
        if parts:
            return "\n\n".join([f"# {doc.title}", *parts])[:2400]
    kept = []
    for line in doc.text.splitlines():
        if line.startswith("#") or len("\n".join(kept)) < 1700:
            kept.append(line)
        if len("\n".join(kept)) >= 2400:
            break
    return "\n".join(kept)[:2400]


def limited_scalar(value: Any, max_chars: int = 1200) -> str:
    text = display_scalar(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ...[truncated]"


def make_ingest_text(doc: WikiDoc) -> str:
    fm = doc.frontmatter
    header = {
        "canonical_id": doc.canonical_id,
        "path": doc.rel_path,
        "doc_type": doc.doc_type,
        "title": doc.title,
        "domain": display_scalar(fm.get("domain")),
        "tags": display_scalar(fm.get("tags")),
        "topic_hints": display_scalar(fm.get("topic_hints")),
        "updated": display_scalar(fm.get("updated")),
        "source_urls": limited_scalar(source_urls(fm), 500),
        "wikilinks_out": limited_scalar(find_wikilinks(doc.text), 500),
        "source_refs": limited_scalar(fm.get("sources"), 500),
    }
    lines = ["[LLM_WIKI_DOC]"]
    for key, value in header.items():
        if value:
            lines.append(f"{key}: {value}")
    lines.append("[/LLM_WIKI_DOC]")
    return "\n".join(lines) + "\n\n" + compact_body_for_ingest(doc)
