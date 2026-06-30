#!/usr/bin/env python3
"""Native-safe facade for llm-wiki script helpers.

This module exposes active native/wiki helpers only. It intentionally does not
re-export retired backend HTTP import/query helpers. Refresh pending marks route
to the native refresh ledger.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ops.batch_native_refresh import mark_pending as mark_native_refresh_pending
from ops.batch_native_refresh import pending_entries as pending_native_refresh_entries
from ops.batch_native_refresh import pending_ledger_path as pending_native_refresh_ledger_path
from ops.batch_native_refresh import status as pending_native_refresh_status
from ops.wiki_native_artifacts import build_seed_edges, extract_method_atoms, resolve_source
from ops.wiki_native_cli import (
    DEFAULT_SERVER,
    DEFAULT_STATE_DIR,
    DEFAULT_WIKI_ROOT,
    DEFAULT_WORKDIR,
    common_paths_parser,
    print_json,
    release_process_memory,
)
from ops.wiki_native_docs import (
    COMPILED_DIR_TYPES,
    WikiDoc,
    collect_source_docs,
    display_scalar,
    generated_doc_filename,
    generated_docs_from_state,
    markdown_sections,
    parse_frontmatter,
    raw_clip_files,
    read_text,
    sha256_text,
)
from ops.wiki_native_custom_kg_payload import build_custom_kg_payload, custom_kg_doc_description, custom_kg_entity_type
from ops.wiki_native_ingest_text import as_list, find_wikilinks
from ops.wiki_native_jsonl import jsonl_read, jsonl_write
from ops.wiki_native_section_similarity import (
    build_section_similarity_edges,
    build_section_similarity_edges_from_index,
    section_similarity_edge_to_custom_kg_relationship,
    section_similarity_embedding_text,
    section_similarity_index_summary,
    section_similarity_report_summary,
    select_section_similarity_edges,
    write_section_similarity_index,
)
from ops.wiki_native_wiki_checks import (
    audit_raw_note_section_contracts,
    compiled_pages,
    index_stats,
    indexed_markdown_pages,
    now_stamp,
    structured_heading_warnings,
    validation_freshness_context,
    validation_report_is_fresh,
    wiki_root_machine_pollution,
)
from ops.wiki_native_wiki_integration_pending import (
    DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD,
    clear_pending_wiki_integration_after_success as _clear_pending_wiki_integration_pending_after_success,
    load_pending_wiki_integration_ledger,
    mark_pending_wiki_integration,
    pending_wiki_integration_ledger_path,
    pending_wiki_integration_status,
    record_pending_wiki_integration_failure,
    save_pending_wiki_integration_ledger,
)
from ops.wiki_native_raw_section_extract import extract_raw_sections
from ops.wiki_native_raw_sections import (
    RAW_NOTE_CONTRACT_REQUIRED_KINDS,
    RAW_NOTE_CONTRACT_SECTION_KINDS,
    raw_section_query_for_kind,
)
from ops.wiki_native_query_events import add_query_event, save_evidence_pack
from ops.wiki_native_state import ensure_state_dirs
from ops.wiki_native_validation import secret_hits, validate_wiki

__all__ = [
    "COMPILED_DIR_TYPES",
    "DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD",
    "DEFAULT_SERVER",
    "DEFAULT_STATE_DIR",
    "DEFAULT_WIKI_ROOT",
    "DEFAULT_WORKDIR",
    "RAW_NOTE_CONTRACT_REQUIRED_KINDS",
    "RAW_NOTE_CONTRACT_SECTION_KINDS",
    "WikiDoc",
    "add_query_event",
    "as_list",
    "audit_raw_note_section_contracts",
    "build_section_similarity_edges",
    "build_section_similarity_edges_from_index",
    "build_custom_kg_payload",
    "build_seed_edges",
    "clear_pending_wiki_integration_after_success",
    "compiled_pages",
    "collect_source_docs",
    "common_paths_parser",
    "custom_kg_doc_description",
    "custom_kg_entity_type",
    "display_scalar",
    "ensure_state_dirs",
    "extract_method_atoms",
    "extract_raw_sections",
    "find_wikilinks",
    "generated_docs_from_state",
    "index_stats",
    "indexed_markdown_pages",
    "jsonl_read",
    "jsonl_write",
    "load_pending_wiki_integration_ledger",
    "mark_native_refresh_pending",
    "mark_pending_wiki_integration",
    "now_stamp",
    "pending_native_refresh_entries",
    "pending_native_refresh_ledger_path",
    "pending_native_refresh_status",
    "pending_wiki_integration_ledger_path",
    "pending_wiki_integration_status",
    "parse_frontmatter",
    "print_json",
    "raw_clip_files",
    "raw_section_query_for_kind",
    "read_text",
    "record_pending_wiki_integration_failure",
    "release_process_memory",
    "resolve_source",
    "save_evidence_pack",
    "save_pending_wiki_integration_ledger",
    "section_similarity_embedding_text",
    "section_similarity_edge_to_custom_kg_relationship",
    "section_similarity_index_summary",
    "section_similarity_report_summary",
    "select_section_similarity_edges",
    "sha256_text",
    "structured_heading_warnings",
    "validate_wiki",
    "validation_freshness_context",
    "validation_report_is_fresh",
    "wiki_root_machine_pollution",
    "write_section_similarity_index",
]


def clear_pending_wiki_integration_after_success(
    root: Path,
    state_dir: Path,
    integrated_paths: list[str] | None = None,
    reason: str = "integration",
) -> dict[str, Any]:
    """Clear wiki integration pending and carry cleared work into native refresh pending."""

    result = _clear_pending_wiki_integration_pending_after_success(
        root,
        state_dir,
        integrated_paths=integrated_paths,
        reason=reason,
    )
    marked_native_pending: list[dict[str, Any]] = []
    if int(result.get("cleared_count") or 0) > 0:
        marked_native_pending.append(
            mark_native_refresh_pending(
                state_dir,
                root,
                reason=f"wiki-integration:{reason}",
            )
    )
    return {
        **result,
        "marked_native_pending": marked_native_pending,
        "marked_native_pending_count": len(marked_native_pending),
        "native_ledger_path": str(pending_native_refresh_ledger_path(state_dir)),
    }
