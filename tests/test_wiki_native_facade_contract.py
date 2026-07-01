from __future__ import annotations

import ast
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"
SCRIPTS = OPS
sys.path.insert(0, str(ROOT))

from ops import batch_native_refresh  # noqa: E402
from ops import wiki_native_lib  # noqa: E402
PURE_VALIDATION_HELPERS = {
    "COMPILED_DIR_TYPES",
    "as_list",
    "compiled_pages",
    "display_scalar",
    "find_wikilinks",
    "index_stats",
    "indexed_markdown_pages",
    "parse_frontmatter",
    "raw_clip_files",
    "resolve_source",
    "structured_heading_warnings",
    "wiki_root_machine_pollution",
}


FACADE_OWNER_SYMBOLS = {
    "ops.wiki_native_artifacts": ("build_seed_edges", "extract_method_atoms", "resolve_source"),
    "ops.wiki_native_cli": ("common_paths_parser", "print_json", "release_process_memory"),
    "ops.wiki_native_custom_kg_payload": ("build_custom_kg_payload", "custom_kg_doc_description", "custom_kg_entity_type"),
    "llm_wiki_native.source_docs": (
        "COMPILED_DIR_TYPES",
        "WikiDoc",
        "collect_source_docs",
        "display_scalar",
        "generated_docs_from_state",
        "parse_frontmatter",
        "raw_clip_files",
        "read_text",
        "sha256_text",
    ),
    "ops.wiki_native_ingest_text": ("as_list", "find_wikilinks"),
    "ops.wiki_native_jsonl": ("jsonl_read", "jsonl_write"),
    "ops.wiki_native_query_events": ("add_query_event", "save_evidence_pack"),
    "ops.wiki_native_raw_section_extract": ("extract_raw_sections",),
    "ops.wiki_native_raw_sections": (
        "RAW_NOTE_CONTRACT_REQUIRED_KINDS",
        "RAW_NOTE_CONTRACT_SECTION_KINDS",
        "raw_section_query_for_kind",
    ),
    "ops.wiki_native_section_similarity": (
        "build_section_similarity_edges",
        "build_section_similarity_edges_from_index",
        "section_similarity_edge_to_custom_kg_relationship",
        "section_similarity_embedding_text",
        "section_similarity_index_summary",
        "section_similarity_report_summary",
        "select_section_similarity_edges",
        "write_section_similarity_index",
    ),
    "ops.wiki_native_state": ("ensure_state_dirs",),
    "ops.wiki_native_validation": ("validate_wiki",),
    "ops.wiki_native_wiki_checks": (
        "audit_raw_note_section_contracts",
        "compiled_pages",
        "index_stats",
        "indexed_markdown_pages",
        "now_stamp",
        "structured_heading_warnings",
        "validation_freshness_context",
        "validation_report_is_fresh",
        "wiki_root_machine_pollution",
    ),
    "ops.wiki_native_wiki_integration_bridge": ("clear_pending_wiki_integration_after_success",),
    "ops.wiki_native_wiki_integration_pending": (
        "load_pending_wiki_integration_ledger",
        "mark_pending_wiki_integration",
        "pending_wiki_integration_ledger_path",
        "pending_wiki_integration_status",
        "record_pending_wiki_integration_failure",
        "save_pending_wiki_integration_ledger",
    ),
}

FACADE_OWNER_EQUALITY_SYMBOLS = {
    "ops.wiki_native_cli": ("DEFAULT_SERVER", "DEFAULT_STATE_DIR", "DEFAULT_WIKI_ROOT", "DEFAULT_WORKDIR"),
    "ops.wiki_native_wiki_integration_pending": ("DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD",),
}

FACADE_OWNER_ALIASES = (
    ("ops.batch_native_refresh", "mark_native_refresh_pending", "mark_pending"),
    ("ops.batch_native_refresh", "pending_native_refresh_entries", "pending_entries"),
    ("ops.batch_native_refresh", "pending_native_refresh_ledger_path", "pending_ledger_path"),
    ("ops.batch_native_refresh", "pending_native_refresh_status", "status"),
)
def test_native_facade_exports_active_owner_symbols() -> None:
    for module_name, names in FACADE_OWNER_SYMBOLS.items():
        owner = importlib.import_module(module_name)
        for name in names:
            assert name in wiki_native_lib.__all__
            assert getattr(wiki_native_lib, name) is getattr(owner, name)

    for module_name, names in FACADE_OWNER_EQUALITY_SYMBOLS.items():
        owner = importlib.import_module(module_name)
        for name in names:
            assert name in wiki_native_lib.__all__
            assert getattr(wiki_native_lib, name) == getattr(owner, name)

    for module_name, facade_name, owner_name in FACADE_OWNER_ALIASES:
        owner = importlib.import_module(module_name)
        assert facade_name in wiki_native_lib.__all__
        assert getattr(wiki_native_lib, facade_name) is getattr(owner, owner_name)
def test_save_evidence_pack_accepts_string_references_from_answer_route(tmp_path: Path) -> None:
    native_query_events = importlib.import_module("ops.wiki_native_query_events")

    pack = native_query_events.save_evidence_pack(
        tmp_path / "state",
        "answer route query",
        "mix",
        {
            "response": "answer",
            "references": ["raw/example.md"],
            "data": {"context_blocks": [{"source_path": "raw/example.md", "text": "context"}]},
            "trace": {"retrieval_backend": "zvec", "context_block_count": 1},
        },
    )

    text = pack.read_text(encoding="utf-8")
    assert "- file_path: `raw/example.md`" in text
def test_native_facade_exports_pure_validation_helpers() -> None:
    for name in PURE_VALIDATION_HELPERS:
        assert name in wiki_native_lib.__all__
        assert hasattr(wiki_native_lib, name)
