"""Deterministic lexical sidecar span extraction for native workspaces."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


@dataclass(frozen=True)
class LexicalSpan:
    span_id: str
    source_path: str
    source_id: str
    source_role: str
    span_kind: str
    heading_path: list[str]
    start_line: int
    end_line: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    text_hash: str = ""

    def with_hash(self) -> "LexicalSpan":
        if self.text_hash:
            return self
        return LexicalSpan(
            span_id=self.span_id,
            source_path=self.source_path,
            source_id=self.source_id,
            source_role=self.source_role,
            span_kind=self.span_kind,
            heading_path=self.heading_path,
            start_line=self.start_line,
            end_line=self.end_line,
            text=self.text,
            metadata=self.metadata,
            text_hash=_sha256_text(self.text),
        )


def spans_from_source_root(root: Path, workspace_id: str) -> list[LexicalSpan]:
    """Extract source-backed lexical spans from the human wiki root."""

    del workspace_id  # span ids are workspace-independent; DB rows carry workspace_id.
    from llm_wiki_native.source_docs import collect_source_docs

    root = Path(root)
    if not root.exists():
        return []
    spans: list[LexicalSpan] = []
    for doc in collect_source_docs(root):
        spans.extend(_spans_from_markdown(doc.rel_path, doc.canonical_id, _source_role_for_doc_type(doc.doc_type), doc.text))
    return [_dedupe_span_id(span, index).with_hash() for index, span in enumerate(spans)]


def spans_from_native_records(workspace_id: str, manifest: dict[str, Any], raw_sections: list[dict[str, Any]]) -> list[LexicalSpan]:
    """Build snapshot spans from state artifacts when no source root is available."""

    del workspace_id
    spans: list[LexicalSpan] = []
    for collection, role in (("chunks", "compiled"), ("entities", "compiled"), ("relationships", "compiled")):
        for record_id, record in manifest.get(collection, {}).items():
            if not isinstance(record, dict):
                continue
            text = str(record.get("content") or "")
            if not text.strip():
                continue
            source_path = str(record.get("file_path") or record.get("source_path") or "")
            spans.append(
                LexicalSpan(
                    span_id=f"snapshot:{collection}:{record_id}",
                    source_path=source_path or f"{record_id}.md",
                    source_id=str(record.get("source_id") or record.get("source_logical_id") or record_id),
                    source_role=role,
                    span_kind="record.snapshot",
                    heading_path=[],
                    start_line=0,
                    end_line=0,
                    text=text,
                    metadata={"collection": collection, "record_id": str(record_id)},
                )
            )
    for section in raw_sections:
        section_id = str(section.get("section_id") or "")
        text = str(section.get("content") or "")
        if not section_id or not text.strip():
            continue
        spans.append(
            LexicalSpan(
                span_id=f"raw-section:{section_id}",
                source_path=str(section.get("source_path") or ""),
                source_id=str(section.get("source_id") or ""),
                source_role="raw",
                span_kind="raw.section",
                heading_path=[str(section.get("section_title") or section.get("canonical_section_title") or "")],
                start_line=int(section.get("start_line") or 0),
                end_line=int(section.get("end_line") or 0),
                text=text,
                metadata={
                    "section_id": section_id,
                    "section_kind": section.get("section_kind"),
                    "section_title": section.get("section_title"),
                },
            )
        )
    return [_dedupe_span_id(span, index).with_hash() for index, span in enumerate(spans)]


def _span_kwargs(item: LexicalSpan) -> dict[str, Any]:
    return {
        "span_id": item.span_id,
        "source_path": item.source_path,
        "source_id": item.source_id,
        "source_role": item.source_role,
        "span_kind": item.span_kind,
        "heading_path": item.heading_path,
        "start_line": item.start_line,
        "end_line": item.end_line,
        "text": item.text,
        "metadata": item.metadata,
        "text_hash": item.text_hash,
    }


def materialize_lexical_spans(db: Any, workspace_id: str, spans: Iterable[LexicalSpan]) -> int:
    items = [_span_kwargs(span.with_hash()) for span in spans]
    if not items:
        return 0
    return db.put_lexical_spans(workspace_id, items)


def _spans_from_markdown(rel_path: str, source_id: str, source_role: str, text: str) -> list[LexicalSpan]:
    lines = text.splitlines()
    heading_stack: list[tuple[int, str]] = []
    spans: list[LexicalSpan] = []
    in_frontmatter = bool(lines and lines[0].strip() == "---")
    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        if index == 1 and stripped == "---":
            continue
        if in_frontmatter:
            if stripped == "---":
                in_frontmatter = False
            continue
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            heading_stack = [(existing_level, existing_title) for existing_level, existing_title in heading_stack if existing_level < level]
            heading_stack.append((level, title))
            spans.append(
                _line_span(
                    rel_path,
                    source_id,
                    source_role,
                    "doc.heading",
                    [title for _, title in heading_stack],
                    index,
                    line,
                    {"heading_level": level},
                )
            )
            continue
        if _is_table_row(stripped):
            spans.append(_line_span(rel_path, source_id, source_role, "table.row", _heading_path(heading_stack), index, line, {}))
            continue
        if source_role == "meta_map" and _is_map_row(stripped):
            spans.append(_line_span(rel_path, source_id, source_role, "map.row", _heading_path(heading_stack), index, line, {}))
            continue
    return spans


def _line_span(
    rel_path: str,
    source_id: str,
    source_role: str,
    span_kind: str,
    heading_path: list[str],
    line_no: int,
    text: str,
    metadata: dict[str, Any],
) -> LexicalSpan:
    return LexicalSpan(
        span_id=f"lexical:{rel_path}:{line_no}:{span_kind}",
        source_path=rel_path,
        source_id=source_id,
        source_role=source_role,
        span_kind=span_kind,
        heading_path=heading_path,
        start_line=line_no,
        end_line=line_no,
        text=text.strip(),
        metadata=metadata,
    )


def _heading_path(stack: list[tuple[int, str]]) -> list[str]:
    return [title for _, title in stack]


def _is_table_row(stripped: str) -> bool:
    if not stripped.startswith("|") or stripped.count("|") < 2:
        return False
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    if cells and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells):
        return False
    return True


def _is_map_row(stripped: str) -> bool:
    return stripped.startswith("- ") or stripped.startswith("* ")


def _source_role_for_doc_type(doc_type: str) -> str:
    if doc_type == "raw_note":
        return "raw"
    if doc_type == "meta_map":
        return "meta_map"
    if doc_type.startswith("compiled"):
        return "compiled"
    if doc_type.startswith("raw_section"):
        return "raw"
    return "wiki"


def _dedupe_span_id(span: LexicalSpan, index: int) -> LexicalSpan:
    stable = f"{span.source_path}\n{span.start_line}\n{span.span_kind}\n{span.text}"
    digest = hashlib.sha1(stable.encode("utf-8")).hexdigest()[:12]
    span_id = f"{span.span_id}:{digest}" if span.span_id else f"lexical:{index}:{digest}"
    return LexicalSpan(
        span_id=span_id,
        source_path=span.source_path,
        source_id=span.source_id,
        source_role=span.source_role,
        span_kind=span.span_kind,
        heading_path=span.heading_path,
        start_line=span.start_line,
        end_line=span.end_line,
        text=span.text,
        metadata=span.metadata,
        text_hash=span.text_hash,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
