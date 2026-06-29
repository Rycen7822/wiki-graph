#!/usr/bin/env python3
"""Compatibility utilities for the llm-wiki wikigraph migration.

Generated state stays under the configured workdir state directory.
The Markdown wiki root is read-only input for these scripts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from wiki_native_artifacts import (
    WIKI_SOURCE_ROOT_PREFIXES,
    build_seed_edges,
    extract_method_atoms,
    resolve_source,
)
from wiki_native_cli import (
    DEFAULT_SERVER,
    DEFAULT_STATE_DIR,
    DEFAULT_WIKI_ROOT,
    DEFAULT_WORKDIR,
    common_paths_parser,
    print_json,
    release_process_memory,
)
from wiki_native_docs import (
    COMPILED_DIR_TYPES,
    META_FILES,
    WikiDoc,
    canonical_id_for,
    collect_source_docs,
    display_scalar,
    doc_type_for,
    fallback_frontmatter_load,
    generated_doc_id,
    generated_docs_from_state,
    markdown_sections,
    parse_frontmatter,
    raw_clip_files,
    read_text,
    sha256_text,
    title_for,
)
from wiki_native_custom_kg_payload import build_custom_kg_payload, custom_kg_doc_description, custom_kg_entity_type
from wiki_native_ingest_text import (
    as_list,
    compact_body_for_ingest,
    find_wikilinks,
    first_sentences,
    limited_scalar,
    make_ingest_text,
    source_urls,
)
from wiki_native_jsonl import jsonl_read, jsonl_write
from wiki_native_query_response import (
    expand_wikigraph_data_response_with_section_neighbors,
    filter_wikigraph_data_response_by_section_kind,
)
from wiki_native_section_similarity import (
    _section_rank_lists,
    _section_rank_lists_scalar,
    build_section_similarity_edges,
    build_section_similarity_edges_from_index,
    cosine_similarity,
    section_similarity_edge_to_custom_kg_relationship,
    section_similarity_embedding_text,
    section_similarity_index_summary,
    section_similarity_report_summary,
    select_section_similarity_edges,
    write_section_similarity_index,
)
from wiki_native_state import STATE_SUBDIRS, ensure_state_dirs
from wiki_native_wiki_checks import (
    POLLUTION_DIRECT_NAMES,
    POLLUTION_RECURSIVE_NAMES,
    VALIDATION_REPORT_FRESHNESS_SCHEMA_VERSION,
    audit_raw_note_section_contracts,
    compiled_pages,
    index_stats,
    indexed_markdown_pages,
    is_structured_raw_note,
    now_stamp,
    structured_heading_warnings,
    validation_freshness_context,
    validation_input_fingerprints,
    validation_report_is_fresh,
    wiki_root_machine_pollution,
)
from wiki_native_lib import (
    clear_pending_wiki_integration_after_success as _native_clear_pending_wiki_integration_after_success,
)
from wiki_native_wiki_integration_pending import (
    DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD,
    PENDING_WIKI_INTEGRATION_LEDGER,
    WIKI_INTEGRATION_ACTIONABLE_STATUSES,
    WIKI_INTEGRATION_REVIEW_STATUSES,
    WIKI_INTEGRATION_TERMINAL_STATUSES,
    default_pending_wiki_integration_ledger,
    load_pending_wiki_integration_ledger,
    mark_pending_wiki_integration,
    pending_wiki_integration_ledger_path,
    pending_wiki_integration_status,
    record_pending_wiki_integration_failure,
    save_pending_wiki_integration_ledger,
)
from wiki_native_raw_sections import (
    RAW_NOTE_CONTRACT_REQUIRED_KINDS,
    RAW_NOTE_CONTRACT_SECTION_KINDS,
    RAW_NOTE_NEAR_MISS_TOKENS,
    RAW_NOTE_SUMMARY_ALIASES,
    RAW_SECTION_QUERY_ALIASES,
    RAW_SECTION_SPECS,
)


from wiki_native_query_events import slugify


def clear_pending_wiki_integration_after_success(
    root: Path,
    state_dir: Path,
    integrated_paths: list[str] | None = None,
    reason: str = "integration",
    mark_native_pending: bool = True,
) -> dict[str, Any]:
    return _native_clear_pending_wiki_integration_after_success(
        root,
        state_dir,
        integrated_paths=integrated_paths,
        reason=reason,
        mark_native_pending=mark_native_pending,
    )


from wiki_native_query_events import init_manifest_db


from wiki_native_validation import secret_hits, validate_wiki



from wiki_native_raw_section_extract import extract_raw_note_sections, extract_raw_sections, raw_section_markdown
from wiki_native_raw_sections import (
    likely_raw_section_kinds_for_unmatched_heading,
    normalized_heading_key,
    raw_section_matches_heading,
    raw_section_spec_for_heading,
    raw_section_specs_for_heading,
    summary_heading_matches,
)


from wiki_native_raw_sections import raw_section_id_from_content, raw_section_kind_from_content, raw_section_query_for_kind


from wiki_native_query_events import add_query_event, save_evidence_pack
