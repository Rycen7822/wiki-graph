#!/usr/bin/env python3
"""Native wiki validation and audit check helpers."""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from pathlib import Path
from typing import Any, Iterable

from wiki_native_docs import (
    COMPILED_DIR_TYPES,
    META_FILES,
    collect_source_docs,
    display_scalar,
    markdown_sections,
    parse_frontmatter,
    raw_clip_files,
    read_text,
)
from wiki_native_raw_sections import (
    RAW_NOTE_CONTRACT_REQUIRED_KINDS,
    RAW_NOTE_CONTRACT_SECTION_KINDS,
    likely_raw_section_kinds_for_unmatched_heading,
    raw_section_spec_for_heading,
    raw_section_specs_for_heading,
    summary_heading_matches,
)
from wiki_wikigraph_compat_names import retired_graph_package_name

VALIDATION_REPORT_FRESHNESS_SCHEMA_VERSION = 1
_RETIRED_BACKEND = retired_graph_package_name()
_RETIRED_SYNC_DB_NAME = f"{_RETIRED_BACKEND}_sync.db"
_RETIRED_MANIFEST_NAME = f"{_RETIRED_BACKEND}_manifest.jsonl"
POLLUTION_DIRECT_NAMES = {
    ".llm-wiki",
    "wikigraph_sync.db",
    "wikigraph_manifest.jsonl",
    _RETIRED_SYNC_DB_NAME,
    _RETIRED_MANIFEST_NAME,
    "seed_edges.jsonl",
    "method_atoms.jsonl",
    "raw_sections.jsonl",
    "retrieval_eval_queries.jsonl",
    "connection_review_queue.jsonl",
    "rag_storage",
    "inputs",
    "edge_docs",
    "method_atom_docs",
    "raw_section_docs",
    "raw_section_audits",
    "section_similarity_reports",
    "section_embeddings.jsonl",
    "section_similarity_edges.candidates.jsonl",
    "section_similarity_edges.jsonl",
    "evidence_packs",
    "validation_reports",
    "scripts",
}
POLLUTION_RECURSIVE_NAMES = {
    "wikigraph_sync.db",
    "wikigraph_manifest.jsonl",
    _RETIRED_SYNC_DB_NAME,
    _RETIRED_MANIFEST_NAME,
    "seed_edges.jsonl",
    "method_atoms.jsonl",
    "raw_sections.jsonl",
    "raw_section_audits",
    "section_similarity_reports",
    "section_embeddings.jsonl",
    "section_similarity_edges.candidates.jsonl",
    "section_similarity_edges.jsonl",
    "retrieval_eval_queries.jsonl",
    "connection_review_queue.jsonl",
}


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def validation_report_is_fresh(
    report: dict[str, Any],
    current: dict[str, Any],
    *,
    required_surfaces: Iterable[str],
    reason: str,
) -> dict[str, Any]:
    """Fail-closed freshness check for reusing a validation report."""

    rejections: list[str] = []
    expected_schema = current.get("schema_version", VALIDATION_REPORT_FRESHNESS_SCHEMA_VERSION)
    if report.get("schema_version") != expected_schema:
        rejections.append("schema_version_mismatch")
    if not report.get("generated_at"):
        rejections.append("missing_generated_at")
    if "warnings" not in report:
        rejections.append("missing_warnings")
    elif report.get("warnings"):
        rejections.append("report_has_warnings")
    if "errors" not in report:
        rejections.append("missing_errors")
    elif report.get("errors"):
        rejections.append("report_has_errors")

    for key in ("root", "state_dir", "workdir"):
        current_value = current.get(key)
        if current_value is not None and report.get(key) != current_value:
            rejections.append(f"{key}_mismatch")

    covered_surfaces = {str(surface) for surface in report.get("covered_surfaces") or []}
    for surface in required_surfaces:
        surface_name = str(surface)
        if surface_name not in covered_surfaces:
            rejections.append(f"missing_required_surface:{surface_name}")

    valid_reasons = {str(item) for item in report.get("valid_for_reasons") or []}
    if reason and reason not in valid_reasons:
        rejections.append(f"missing_valid_reason:{reason}")

    report_fingerprints = report.get("input_fingerprints")
    if not isinstance(report_fingerprints, dict):
        report_fingerprints = {}
    current_fingerprints = current.get("input_fingerprints")
    if not isinstance(current_fingerprints, dict) or not current_fingerprints:
        rejections.append("missing_current_fingerprints")
        current_fingerprints = {}
    for rel_path, current_fingerprint in current_fingerprints.items():
        if rel_path not in report_fingerprints:
            rejections.append(f"missing_fingerprint:{rel_path}")
        elif report_fingerprints[rel_path] != current_fingerprint:
            rejections.append(f"fingerprint_mismatch:{rel_path}")
    for rel_path in sorted(report_fingerprints):
        if rel_path not in current_fingerprints:
            rejections.append(f"stale_fingerprint:{rel_path}")

    return {"fresh": not rejections, "rejections": rejections}


def validation_input_fingerprints(root: Path) -> dict[str, dict[str, Any]]:
    root = root.resolve()
    paths = [root / "index.md", root / "SCHEMA.md", *indexed_markdown_pages(root), *raw_clip_files(root)]
    fingerprints: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel in fingerprints:
            continue
        stat = path.stat()
        fingerprints[rel] = {
            "path": rel,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return dict(sorted(fingerprints.items()))


def validation_freshness_context(root: Path, state_dir: Path, workdir: Path | None = None) -> dict[str, Any]:
    return {
        "schema_version": VALIDATION_REPORT_FRESHNESS_SCHEMA_VERSION,
        "root": str(root.resolve()),
        "state_dir": str(state_dir.resolve()),
        "workdir": str(workdir.resolve()) if workdir else None,
        "input_fingerprints": validation_input_fingerprints(root),
    }


def wiki_root_machine_pollution(root: Path) -> list[Path]:
    root = root.resolve()
    polluted: list[Path] = []
    for name in POLLUTION_DIRECT_NAMES:
        p = root / name
        if p.exists():
            polluted.append(Path(name))
    for name in POLLUTION_RECURSIVE_NAMES:
        for p in root.rglob(name):
            rel = p.relative_to(root)
            if rel not in polluted:
                polluted.append(rel)
    return sorted(polluted, key=lambda p: p.as_posix())


def compiled_pages(root: Path) -> list[Path]:
    paths: list[Path] = []
    for dirname in COMPILED_DIR_TYPES:
        d = root / dirname
        if d.exists():
            paths.extend(sorted(d.glob("*.md")))
    return paths


def indexed_markdown_pages(root: Path) -> list[Path]:
    paths = compiled_pages(root)
    for rel in META_FILES:
        p = root / rel
        if p.exists():
            paths.append(p)
    return sorted(paths, key=lambda p: p.relative_to(root).as_posix())


def index_stats(root: Path) -> tuple[int | None, int]:
    path = root / "index.md"
    if not path.exists():
        return None, 0
    text = read_text(path)
    m = re.search(r"Total pages:\s*(\d+)", text)
    total = int(m.group(1)) if m else None
    wikilinks = len(re.findall(r"^-\s+\[\[[^\]]+\]\]", text, flags=re.M))
    return total, wikilinks


def is_structured_raw_note(text: str) -> bool:
    meta, _body = parse_frontmatter(text)
    note_type = display_scalar(meta.get("type")).strip().strip('"\'').lower()
    domain = display_scalar(meta.get("domain")).strip().strip('"\'').lower()
    return note_type in {"raw-note", "paper-note"} or domain == "paper"


def structured_heading_warnings(path: Path, text: str) -> list[str]:
    if not is_structured_raw_note(text):
        return []
    sections = markdown_sections(text)
    titles = [title for title, _body in sections]
    present = {
        "## 一句话总结": any(summary_heading_matches(title) for title in titles),
        "## 论文摘要": False,
        "## Motivation": False,
        "## Methodology": False,
        "## 关键实验结果": False,
    }
    kind_to_heading = {
        "abstract": "## 论文摘要",
        "motivation": "## Motivation",
        "methodology": "## Methodology",
        "results": "## 关键实验结果",
    }
    for title in titles:
        spec = raw_section_spec_for_heading(title)
        if spec and spec.get("kind") in kind_to_heading:
            present[kind_to_heading[spec["kind"]]] = True
    warnings = []
    for heading, has_heading in present.items():
        if not has_heading:
            warnings.append(f"{path.as_posix()} missing heading prefix {heading}")
    return warnings


def audit_raw_note_section_contracts(root: Path, include_legacy: bool = True, issue_limit: int | None = None) -> dict[str, Any]:
    root = root.resolve()
    raw_total = 0
    structured_total = 0
    issue_rows: list[dict[str, Any]] = []
    heading_occurrences: dict[str, int] = {}
    doc_counts_by_kind: dict[str, int] = {}

    def add_issue(issue: dict[str, Any]) -> None:
        issue_rows.append(issue)

    for doc in collect_source_docs(root):
        if doc.doc_type != "raw_note":
            continue
        raw_total += 1
        structured = is_structured_raw_note(doc.text)
        if structured:
            structured_total += 1
        if not structured and not include_legacy:
            continue
        titles_by_kind: dict[str, list[str]] = {kind: [] for kind in RAW_NOTE_CONTRACT_SECTION_KINDS}
        for title, body in markdown_sections(doc.body):
            kinds: list[str] = []
            if summary_heading_matches(title):
                kinds.append("summary")
            matched_specs = raw_section_specs_for_heading(title)
            kinds.extend(str(spec["kind"]) for spec in matched_specs)
            kinds = list(dict.fromkeys(kinds))
            if kinds:
                for kind in kinds:
                    titles_by_kind.setdefault(kind, []).append(title)
                    heading_occurrences[kind] = heading_occurrences.get(kind, 0) + 1
                if len(kinds) > 1:
                    add_issue(
                        {
                            "type": "combined_section_heading",
                            "severity": "warning",
                            "path": doc.rel_path,
                            "title": title,
                            "section_kinds": kinds,
                            "message": "One heading maps to multiple retrieval sections; split it or keep the new multi-indexing behavior in mind.",
                        }
                    )
                continue
            suggestions = likely_raw_section_kinds_for_unmatched_heading(title)
            if suggestions:
                add_issue(
                    {
                        "type": "near_miss_heading",
                        "severity": "info",
                        "path": doc.rel_path,
                        "title": title,
                        "suggested_section_kinds": suggestions,
                        "message": "Heading looks semantically section-like but is not part of the section-level retrieval contract.",
                    }
                )
        for kind, titles in titles_by_kind.items():
            if titles:
                doc_counts_by_kind[kind] = doc_counts_by_kind.get(kind, 0) + 1
            if len(titles) > 1:
                add_issue(
                    {
                        "type": "duplicate_section_kind",
                        "severity": "warning",
                        "path": doc.rel_path,
                        "section_kind": kind,
                        "titles": titles,
                        "message": "Multiple headings map to the same retrieval section; extraction keeps the first section per kind.",
                    }
                )
        if structured:
            for kind in RAW_NOTE_CONTRACT_REQUIRED_KINDS:
                if not titles_by_kind.get(kind):
                    add_issue(
                        {
                            "type": "missing_section",
                            "severity": "warning",
                            "path": doc.rel_path,
                            "section_kind": kind,
                            "message": "Structured raw paper note is missing a canonical retrieval section.",
                        }
                    )
    issues_by_type: dict[str, int] = {}
    issues_by_severity: dict[str, int] = {}
    for issue in issue_rows:
        issue_type = str(issue.get("type", "unknown"))
        severity = str(issue.get("severity", "warning"))
        issues_by_type[issue_type] = issues_by_type.get(issue_type, 0) + 1
        issues_by_severity[severity] = issues_by_severity.get(severity, 0) + 1
    returned_issues = issue_rows[:issue_limit] if issue_limit is not None else issue_rows
    return {
        "generated_at": now_stamp(),
        "root": root.as_posix(),
        "raw_notes": raw_total,
        "structured_raw_notes": structured_total,
        "include_legacy": include_legacy,
        "contract_section_kinds": RAW_NOTE_CONTRACT_SECTION_KINDS,
        "required_structured_section_kinds": RAW_NOTE_CONTRACT_REQUIRED_KINDS,
        "heading_occurrences_by_kind": dict(sorted(heading_occurrences.items())),
        "docs_with_section_by_kind": dict(sorted(doc_counts_by_kind.items())),
        "issue_count": len(issue_rows),
        "returned_issue_count": len(returned_issues),
        "issues_by_type": dict(sorted(issues_by_type.items())),
        "issues_by_severity": dict(sorted(issues_by_severity.items())),
        "issues": returned_issues,
    }
