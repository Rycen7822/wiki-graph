import importlib
import inspect
import json
import sqlite3
import subprocess
import sys
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
RAW_FAST_VERIFIER = Path.home() / ".hermes" / "skills" / "research" / "llm-wiki" / "scripts" / "raw_fast_note_verify.py"
sys.path.insert(0, str(SCRIPTS))

import batch_native_refresh  # noqa: E402
import batch_wikigraph_refresh  # noqa: E402
import batch_wiki_integration  # noqa: E402
import raw_fast_closeout  # noqa: E402
import validate_wiki as validate_wiki_cli  # noqa: E402

from wiki_wikigraph_compat_lib import (  # noqa: E402
    _section_rank_lists,
    _section_rank_lists_scalar,
    build_custom_kg_payload,
    build_section_similarity_edges,
    build_seed_edges,
    canonical_id_for,
    collect_source_docs,
    ensure_state_dirs,
    extract_method_atoms,
    expand_wikigraph_data_response_with_section_neighbors,
    extract_raw_sections,
    fallback_frontmatter_load,
    filter_wikigraph_data_response_by_section_kind,
    generated_docs_from_state,
    init_manifest_db,
    DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD,
    clear_pending_wiki_integration_after_success,
    load_pending_wiki_integration_ledger,
    make_ingest_text,
    mark_pending_wiki_integration,
    parse_frontmatter,
    pending_wiki_integration_status,
    audit_raw_note_section_contracts,
    raw_section_query_for_kind,
    raw_section_specs_for_heading,
    resolve_source,
    section_similarity_embedding_text,
    section_similarity_report_summary,
    select_section_similarity_edges,
    structured_heading_warnings,
    validate_wiki,
    wiki_root_machine_pollution,
)
from wiki_wikigraph_refresh_pending import (  # noqa: E402
    DEFAULT_PENDING_WIKIGRAPH_REFRESH_THRESHOLD,
    clear_wikigraph_refresh_pending_after_success,
    load_wikigraph_refresh_ledger,
    mark_wikigraph_refresh_pending,
    pending_wikigraph_refresh_status,
)


def _retired_full_import_commands() -> list[list[str]]:
    return [batch_wikigraph_refresh.retired_wikigraph_cold_import_command()]


def _refresh_command_groups(artifact_commands: list[list[str]]) -> dict[str, list[list[str]]]:
    return {"artifact": artifact_commands, "full_import": _retired_full_import_commands()}


def test_refresh_tests_patch_command_groups_not_obsolete_flat_builder() -> None:
    text = Path(__file__).read_text(encoding="utf-8")
    target = "build_" + "refresh_commands"
    assert f'monkeypatch.setattr(batch_wikigraph_refresh, "{target}"' not in text
    assert f"monkeypatch.setattr(batch_wikigraph_refresh, '{target}'" not in text


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sample_wiki(tmp_path: Path) -> Path:
    root = tmp_path / "llm-wiki"
    write(
        root / "index.md",
        "# LLM Wiki Index\n\n> Last updated: 2026-05-18 16:00 | Total pages: 2\n\n"
        "## Concepts\n\n- [[foo]] - Foo page.\n\n## Queries\n\n- [[bar]] - Bar page.\n",
    )
    write(
        root / "SCHEMA.md",
        "# Schema\n\nAllowed tags include agent, rag.\n",
    )
    write(
        root / "concepts/foo.md",
        "---\ntitle: Foo\ntype: concept\ntags: [agent]\nsources: [../raw/clip/2601/26010101_Foo-Paper.md]\nupdated: 2026-05-18 16:00\n---\n# Foo\n\nLinks to [[bar]].\n",
    )
    write(
        root / "queries/bar.md",
        "---\ntitle: Bar\ntype: query\ntags: [rag]\nupdated: 2026-05-18 16:00\n---\n# Bar\n",
    )
    write(
        root / "raw/clip/2601/26010101_Foo-Paper.md",
        "---\ntitle: Foo Paper\nsource: https://arxiv.org/abs/2601.0101\ndomain: paper\nupdated: 2026-05-18 16:00\ntags: [paper]\n---\n# Foo Paper\n\n## Methodology\n\nA direct method with enough structured detail to become a method atom during deterministic extraction.\n\n## 对未来研究的启发\n\n- Future work should connect memory repair with section-level evidence retrieval.\n\n## 可能的局限\n\n- The current benchmark may hide failures in long-horizon transfer.\n\n## 可继续追问的问题\n\n- Which unresolved interface lets agents ask for the right evidence section before planning?\n",
    )
    write(root / "_meta/raw-clip-map.md", "# Raw Clip Map\n\n- raw/clip/2601/26010101_Foo-Paper.md\n")
    write(root / "_meta/topic-map.md", "# Topic Map\n")
    return root


def validation_reuse_wiki(tmp_path: Path) -> Path:
    root = sample_wiki(tmp_path)
    write(
        root / "index.md",
        "# LLM Wiki Index\n\n> Last updated: 2026-05-18 16:00 | Total pages: 4\n\n"
        "## Concepts\n\n- [[foo]] - Foo page.\n\n"
        "## Queries\n\n- [[bar]] - Bar page.\n\n"
        "## Meta\n\n- [[raw-clip-map]] - Raw clip map.\n- [[topic-map]] - Topic map.\n",
    )
    write(
        root / "raw/clip/2601/26010101_Foo-Paper.md",
        "---\n"
        "title: Foo Paper\n"
        "source: https://arxiv.org/abs/2601.0101\n"
        "domain: paper\n"
        "updated: 2026-05-18 16:00\n"
        "tags: [paper]\n"
        "---\n"
        "# Foo Paper\n\n"
        "## 一句话总结\n\nA concise summary.\n\n"
        "## 论文摘要\n\nA compact abstract.\n\n"
        "## Motivation\n\nA clear motivation.\n\n"
        "## Methodology\n\nA direct method with enough structured detail.\n\n"
        "## 关键实验结果\n\nThe main result is stable.\n\n"
        "## 对未来研究的启发\n\n- Future work should connect evidence retrieval with planning.\n\n"
        "## 可能的局限\n\n- The benchmark may hide failures in long-horizon transfer.\n\n"
        "## 可继续追问的问题\n\n- Which unresolved interface asks for the right evidence section before planning?\n",
    )
    return root


def test_parse_frontmatter_and_canonical_id(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    path = root / "concepts/foo.md"
    meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
    assert meta["title"] == "Foo"
    assert body.startswith("# Foo")
    assert canonical_id_for(root, path) == "compiled:concept:foo"
    assert canonical_id_for(root, root / "raw/clip/2601/26010101_Foo-Paper.md") == "raw_clip:26010101_Foo-Paper"


def test_fallback_frontmatter_loader_handles_inline_lists() -> None:
    meta = fallback_frontmatter_load("title: Foo\ntags: [agent, rag]\nsources:\n  - raw/clip/a.md\n")
    assert meta["title"] == "Foo"
    assert meta["tags"] == ["agent", "rag"]
    assert meta["sources"] == ["raw/clip/a.md"]


def test_collect_source_docs_excludes_schema_log_and_includes_meta(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    write(root / "log.md", "# Log\n")
    docs = collect_source_docs(root)
    rels = {doc.rel_path for doc in docs}
    assert "index.md" in rels
    assert "concepts/foo.md" in rels
    assert "queries/bar.md" in rels
    assert "raw/clip/2601/26010101_Foo-Paper.md" in rels
    assert "_meta/raw-clip-map.md" in rels
    assert "SCHEMA.md" not in rels
    assert "log.md" not in rels


def test_resolve_source_keeps_wiki_sources_inside_root_without_following_external_escape(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    page = root / "concepts/foo.md"

    assert resolve_source(page, "../raw/clip/2601/26010101_Foo-Paper.md", root) == root / "raw/clip/2601/26010101_Foo-Paper.md"
    assert resolve_source(page, "raw/clip/2601/26010101_Foo-Paper.md", root) == root / "raw/clip/2601/26010101_Foo-Paper.md"
    assert resolve_source(page, "https://arxiv.org/abs/2601.0101", root) is None
    assert resolve_source(page, "../../../outside.md", root) is None


def test_state_dirs_and_manifest_are_external_to_wiki_root(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    workdir = tmp_path / "work" / "wikigraph"
    state = workdir / "state"
    ensure_state_dirs(state)
    assert (state / "edge_docs").is_dir()
    assert not (root / ".llm-wiki").exists()
    db = init_manifest_db(state)
    retired_backend = "light" + "rag"
    assert db == state / "wikigraph_sync.db"
    with sqlite3.connect(db) as conn:
        tables = {row[0] for row in conn.execute("select name from sqlite_master where type='table'")}
        columns = {row[1] for row in conn.execute("pragma table_info(docs)")}
    assert {"docs", "sync_events", "query_events"} <= tables
    assert {"wikigraph_track_id", "wikigraph_doc_status"} <= columns
    assert f"{retired_backend}_track_id" not in columns
    assert f"{retired_backend}_doc_status" not in columns


def test_query_event_helpers_reexport_from_native_owner() -> None:
    import wiki_wikigraph_compat_lib
    import wiki_native_query_events

    assert wiki_wikigraph_compat_lib.slugify is wiki_native_query_events.slugify
    assert wiki_wikigraph_compat_lib.init_manifest_db is wiki_native_query_events.init_manifest_db
    assert wiki_wikigraph_compat_lib.save_evidence_pack is wiki_native_query_events.save_evidence_pack
    assert wiki_wikigraph_compat_lib.add_query_event is wiki_native_query_events.add_query_event

    text = (SCRIPTS / "wiki_wikigraph_compat_lib.py").read_text(encoding="utf-8")
    assert "from wiki_native_query_events import" in text
    for name in ("slugify", "init_manifest_db", "save_evidence_pack", "add_query_event"):
        assert f"def {name}(" not in text


def test_raw_section_contract_helpers_reexport_from_native_owner() -> None:
    import wiki_wikigraph_compat_lib
    import wiki_native_raw_sections

    assert wiki_wikigraph_compat_lib.RAW_SECTION_SPECS is wiki_native_raw_sections.RAW_SECTION_SPECS
    assert wiki_wikigraph_compat_lib.RAW_NOTE_CONTRACT_SECTION_KINDS is wiki_native_raw_sections.RAW_NOTE_CONTRACT_SECTION_KINDS
    assert wiki_wikigraph_compat_lib.raw_section_specs_for_heading is wiki_native_raw_sections.raw_section_specs_for_heading
    assert wiki_wikigraph_compat_lib.raw_section_query_for_kind is wiki_native_raw_sections.raw_section_query_for_kind

    text = (SCRIPTS / "wiki_wikigraph_compat_lib.py").read_text(encoding="utf-8")
    assert "from wiki_native_raw_sections import" in text
    for pattern in (
        "RAW_SECTION_SPECS = [",
        "RAW_SECTION_QUERY_ALIASES = {",
        "RAW_NOTE_CONTRACT_SECTION_KINDS = [",
        "def normalized_heading_key(",
        "def raw_section_specs_for_heading(",
        "def raw_section_query_for_kind(",
    ):
        assert pattern not in text


def test_raw_section_extract_helpers_reexport_from_native_owner() -> None:
    import wiki_wikigraph_compat_lib
    import wiki_native_raw_section_extract

    assert wiki_wikigraph_compat_lib.extract_raw_note_sections is wiki_native_raw_section_extract.extract_raw_note_sections
    assert wiki_wikigraph_compat_lib.raw_section_markdown is wiki_native_raw_section_extract.raw_section_markdown
    assert wiki_wikigraph_compat_lib.extract_raw_sections is wiki_native_raw_section_extract.extract_raw_sections

    text = (SCRIPTS / "wiki_wikigraph_compat_lib.py").read_text(encoding="utf-8")
    assert "from wiki_native_raw_section_extract import" in text
    for pattern in (
        "def extract_raw_note_sections(",
        "def raw_section_markdown(",
        "def extract_raw_sections(",
    ):
        assert pattern not in text


def test_document_helpers_reexport_from_native_owner() -> None:
    import wiki_wikigraph_compat_lib
    import wiki_native_docs

    assert wiki_wikigraph_compat_lib.WikiDoc is wiki_native_docs.WikiDoc
    assert wiki_wikigraph_compat_lib.COMPILED_DIR_TYPES is wiki_native_docs.COMPILED_DIR_TYPES
    assert wiki_wikigraph_compat_lib.collect_source_docs is wiki_native_docs.collect_source_docs
    assert wiki_wikigraph_compat_lib.generated_docs_from_state is wiki_native_docs.generated_docs_from_state
    assert wiki_wikigraph_compat_lib.parse_frontmatter is wiki_native_docs.parse_frontmatter
    assert wiki_wikigraph_compat_lib.display_scalar is wiki_native_docs.display_scalar
    assert wiki_wikigraph_compat_lib.sha256_text is wiki_native_docs.sha256_text

    text = (SCRIPTS / "wiki_wikigraph_compat_lib.py").read_text(encoding="utf-8")
    assert "from wiki_native_docs import" in text
    for pattern in (
        "COMPILED_DIR_TYPES = {",
        "META_FILES = [",
        "class WikiDoc",
        "def generated_doc_filename(",
        "def sha256_text(",
        "def read_text(",
        "def fallback_frontmatter_load(",
        "def parse_frontmatter(",
        "def display_scalar(",
        "def canonical_id_for(",
        "def doc_type_for(",
        "def title_for(",
        "def make_wiki_doc(",
        "def raw_clip_files(",
        "def collect_source_docs(",
        "def markdown_sections(",
        "def section_text(",
        "def generated_docs_from_state(",
        "def generated_doc_id(",
    ):
        assert pattern not in text


def test_state_dir_helper_reexport_from_native_owner() -> None:
    import wiki_wikigraph_compat_lib
    import wiki_native_state

    assert wiki_wikigraph_compat_lib.STATE_SUBDIRS is wiki_native_state.STATE_SUBDIRS
    assert wiki_wikigraph_compat_lib.ensure_state_dirs is wiki_native_state.ensure_state_dirs

    text = (SCRIPTS / "wiki_wikigraph_compat_lib.py").read_text(encoding="utf-8")
    assert "from wiki_native_state import" in text
    assert "STATE_SUBDIRS = [" not in text
    assert "def ensure_state_dirs(" not in text


def test_ingest_text_helpers_reexport_from_native_owner() -> None:
    import wiki_wikigraph_compat_lib
    import wiki_native_ingest_text

    assert wiki_wikigraph_compat_lib.as_list is wiki_native_ingest_text.as_list
    assert wiki_wikigraph_compat_lib.find_wikilinks is wiki_native_ingest_text.find_wikilinks
    assert wiki_wikigraph_compat_lib.first_sentences is wiki_native_ingest_text.first_sentences
    assert wiki_wikigraph_compat_lib.source_urls is wiki_native_ingest_text.source_urls
    assert wiki_wikigraph_compat_lib.compact_body_for_ingest is wiki_native_ingest_text.compact_body_for_ingest
    assert wiki_wikigraph_compat_lib.limited_scalar is wiki_native_ingest_text.limited_scalar
    assert wiki_wikigraph_compat_lib.make_ingest_text is wiki_native_ingest_text.make_ingest_text

    text = (SCRIPTS / "wiki_wikigraph_compat_lib.py").read_text(encoding="utf-8")
    assert "from wiki_native_ingest_text import" in text
    for pattern in (
        "def as_list(",
        "def find_wikilinks(",
        "def first_sentences(",
        "def source_urls(",
        "def compact_body_for_ingest(",
        "def limited_scalar(",
        "def make_ingest_text(",
    ):
        assert pattern not in text


def test_jsonl_helpers_reexport_from_native_owner() -> None:
    import wiki_wikigraph_compat_lib
    import wiki_native_jsonl

    assert wiki_wikigraph_compat_lib.jsonl_read is wiki_native_jsonl.jsonl_read
    assert wiki_wikigraph_compat_lib.jsonl_write is wiki_native_jsonl.jsonl_write

    text = (SCRIPTS / "wiki_wikigraph_compat_lib.py").read_text(encoding="utf-8")
    assert "from wiki_native_jsonl import" in text
    assert "def jsonl_read(" not in text
    assert "def jsonl_write(" not in text


def test_query_response_helpers_reexport_from_native_owner() -> None:
    import wiki_wikigraph_compat_lib
    import wiki_native_query_response

    retired_backend = "light" + "rag"
    assert wiki_wikigraph_compat_lib.expand_wikigraph_data_response_with_section_neighbors is wiki_native_query_response.expand_wikigraph_data_response_with_section_neighbors
    assert wiki_wikigraph_compat_lib.filter_wikigraph_data_response_by_section_kind is wiki_native_query_response.filter_wikigraph_data_response_by_section_kind
    assert not hasattr(wiki_wikigraph_compat_lib, f"expand_{retired_backend}_data_response_with_section_neighbors")
    assert not hasattr(wiki_wikigraph_compat_lib, f"filter_{retired_backend}_data_response_by_section_kind")
    assert not hasattr(wiki_native_query_response, f"expand_{retired_backend}_data_response_with_section_neighbors")
    assert not hasattr(wiki_native_query_response, f"filter_{retired_backend}_data_response_by_section_kind")

    text = (SCRIPTS / "wiki_wikigraph_compat_lib.py").read_text(encoding="utf-8")
    assert "from wiki_native_query_response import" in text
    assert "expand_wikigraph_data_response_with_section_neighbors" in text
    assert "filter_wikigraph_data_response_by_section_kind" in text
    assert f"def expand_{retired_backend}_data_response_with_section_neighbors(" not in text
    assert f"def filter_{retired_backend}_data_response_by_section_kind(" not in text


def test_section_similarity_helpers_reexport_from_native_owner() -> None:
    import wiki_wikigraph_compat_lib
    import wiki_native_section_similarity

    for name in (
        "_section_rank_lists",
        "_section_rank_lists_scalar",
        "build_section_similarity_edges",
        "build_section_similarity_edges_from_index",
        "cosine_similarity",
        "section_similarity_edge_to_custom_kg_relationship",
        "section_similarity_embedding_text",
        "section_similarity_index_summary",
        "section_similarity_report_summary",
        "select_section_similarity_edges",
        "write_section_similarity_index",
    ):
        assert getattr(wiki_wikigraph_compat_lib, name) is getattr(wiki_native_section_similarity, name)

    text = (SCRIPTS / "wiki_wikigraph_compat_lib.py").read_text(encoding="utf-8")
    assert "from wiki_native_section_similarity import" in text
    for pattern in (
        "def section_similarity_embedding_text(",
        "def _section_rank_lists_scalar(",
        "def _section_rank_lists(",
        "def write_section_similarity_index(",
        "def build_section_similarity_edges_from_index(",
        "def build_section_similarity_edges(",
        "def section_similarity_edge_to_custom_kg_relationship(",
        "def section_similarity_report_summary(",
        "def select_section_similarity_edges(",
    ):
        assert pattern not in text


def test_custom_kg_payload_helpers_reexport_from_native_owner() -> None:
    import wiki_wikigraph_compat_lib
    import wiki_native_custom_kg_payload

    for name in (
        "build_custom_kg_payload",
        "custom_kg_doc_description",
        "custom_kg_entity_type",
    ):
        assert getattr(wiki_wikigraph_compat_lib, name) is getattr(wiki_native_custom_kg_payload, name)

    text = (SCRIPTS / "wiki_wikigraph_compat_lib.py").read_text(encoding="utf-8")
    assert "from wiki_native_custom_kg_payload import" in text
    for pattern in (
        "def custom_kg_entity_type(",
        "def custom_kg_doc_description(",
        "def build_custom_kg_payload(",
    ):
        assert pattern not in text


def test_cli_defaults_and_helpers_reexport_from_native_owner() -> None:
    import wiki_wikigraph_compat_lib
    import wiki_native_cli

    for name in (
        "DEFAULT_SERVER",
        "DEFAULT_STATE_DIR",
        "DEFAULT_WIKI_ROOT",
        "DEFAULT_WORKDIR",
    ):
        assert getattr(wiki_wikigraph_compat_lib, name) == getattr(wiki_native_cli, name)
    for name in ("common_paths_parser", "print_json", "release_process_memory"):
        assert getattr(wiki_wikigraph_compat_lib, name) is getattr(wiki_native_cli, name)

    text = (SCRIPTS / "wiki_wikigraph_compat_lib.py").read_text(encoding="utf-8")
    assert "from wiki_native_cli import" in text
    for pattern in (
        "DEFAULT_WIKI_ROOT =",
        "DEFAULT_WORKDIR =",
        "def common_paths_parser(",
        "def print_json(",
        "def release_process_memory(",
    ):
        assert pattern not in text


def test_artifact_builder_helpers_reexport_from_native_owner() -> None:
    import wiki_wikigraph_compat_lib
    import wiki_native_artifacts

    for name in (
        "WIKI_SOURCE_ROOT_PREFIXES",
        "build_seed_edges",
        "extract_method_atoms",
        "resolve_source",
    ):
        assert getattr(wiki_wikigraph_compat_lib, name) is getattr(wiki_native_artifacts, name)

    text = (SCRIPTS / "wiki_wikigraph_compat_lib.py").read_text(encoding="utf-8")
    assert "from wiki_native_artifacts import" in text
    for pattern in (
        "def resolve_source(",
        "def _resolve_source_cached(",
        "def _lexical_norm(",
        "def _is_lexically_under(",
        "def bullet_items(",
        "def method_type_for(",
        "def extract_method_atoms(",
        "def method_atom_markdown(",
        "def build_seed_edges(",
        "def edge_markdown(",
        "def write_text_if_changed(",
    ):
        assert pattern not in text


def test_wiki_check_helpers_reexport_from_native_owner() -> None:
    import wiki_wikigraph_compat_lib
    import wiki_native_wiki_checks

    for name in (
        "POLLUTION_DIRECT_NAMES",
        "POLLUTION_RECURSIVE_NAMES",
        "VALIDATION_REPORT_FRESHNESS_SCHEMA_VERSION",
        "audit_raw_note_section_contracts",
        "compiled_pages",
        "index_stats",
        "indexed_markdown_pages",
        "is_structured_raw_note",
        "now_stamp",
        "structured_heading_warnings",
        "validation_freshness_context",
        "validation_input_fingerprints",
        "validation_report_is_fresh",
        "wiki_root_machine_pollution",
    ):
        assert getattr(wiki_wikigraph_compat_lib, name) is getattr(wiki_native_wiki_checks, name)

    text = (SCRIPTS / "wiki_wikigraph_compat_lib.py").read_text(encoding="utf-8")
    assert "from wiki_native_wiki_checks import" in text
    for pattern in (
        "VALIDATION_REPORT_FRESHNESS_SCHEMA_VERSION =",
        "POLLUTION_DIRECT_NAMES =",
        "POLLUTION_RECURSIVE_NAMES =",
        "def now_stamp(",
        "def validation_report_is_fresh(",
        "def validation_input_fingerprints(",
        "def validation_freshness_context(",
        "def wiki_root_machine_pollution(",
        "def compiled_pages(",
        "def indexed_markdown_pages(",
        "def index_stats(",
        "def is_structured_raw_note(",
        "def structured_heading_warnings(",
        "def audit_raw_note_section_contracts(",
    ):
        assert pattern not in text


def test_wiki_integration_pending_helpers_reexport_from_native_owner() -> None:
    import wiki_wikigraph_compat_lib
    import wiki_native_wiki_integration_pending

    for name in (
        "PENDING_WIKI_INTEGRATION_LEDGER",
        "DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD",
        "WIKI_INTEGRATION_ACTIONABLE_STATUSES",
        "WIKI_INTEGRATION_REVIEW_STATUSES",
        "WIKI_INTEGRATION_TERMINAL_STATUSES",
        "default_pending_wiki_integration_ledger",
        "load_pending_wiki_integration_ledger",
        "mark_pending_wiki_integration",
        "pending_wiki_integration_ledger_path",
        "pending_wiki_integration_status",
        "record_pending_wiki_integration_failure",
        "save_pending_wiki_integration_ledger",
    ):
        assert getattr(wiki_wikigraph_compat_lib, name) is getattr(wiki_native_wiki_integration_pending, name)

    assert (
        wiki_wikigraph_compat_lib.clear_pending_wiki_integration_after_success
        is not wiki_native_wiki_integration_pending.clear_pending_wiki_integration_after_success
    )

    text = (SCRIPTS / "wiki_wikigraph_compat_lib.py").read_text(encoding="utf-8")
    assert "from wiki_native_wiki_integration_pending import" in text
    assert (
        "clear_pending_wiki_integration_after_success as _native_clear_pending_wiki_integration_after_success"
        in text
    )
    for pattern in (
        "PENDING_WIKI_INTEGRATION_LEDGER =",
        "DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD =",
        "WIKI_INTEGRATION_ACTIONABLE_STATUSES =",
        "WIKI_INTEGRATION_REVIEW_STATUSES =",
        "WIKI_INTEGRATION_TERMINAL_STATUSES =",
        "def pending_wiki_integration_ledger_path(",
        "def default_pending_wiki_integration_ledger(",
        "def load_pending_wiki_integration_ledger(",
        "def save_pending_wiki_integration_ledger(",
        "def mark_pending_wiki_integration(",
        "def pending_wiki_integration_status(",
        "def record_pending_wiki_integration_failure(",
    ):
        assert pattern not in text


def test_wikigraph_refresh_pending_owner_is_not_reexported_through_compatibility_facade() -> None:
    import wiki_wikigraph_compat_lib
    import wiki_wikigraph_refresh_pending

    old_backend = "light" + "rag"
    old_module = f"wiki_{old_backend}_refresh_pending"
    new_module = "wiki_wikigraph_refresh_pending"

    assert not (SCRIPTS / f"{old_module}.py").exists()
    assert (SCRIPTS / f"{new_module}.py").exists()

    for name in (
        "clear_wikigraph_refresh_pending_after_success",
        "default_wikigraph_refresh_ledger",
        "wikigraph_refresh_import_summary",
        "load_wikigraph_refresh_ledger",
        "mark_wikigraph_refresh_pending",
        "pending_wikigraph_refresh_ledger_path",
        "pending_wikigraph_refresh_status",
        "record_wikigraph_refresh_failure",
        "save_wikigraph_refresh_ledger",
        "wiki_markdown_latest_mtime",
        "PENDING_WIKIGRAPH_REFRESH_LEDGER",
        "DEFAULT_PENDING_WIKIGRAPH_REFRESH_THRESHOLD",
    ):
        assert hasattr(wiki_wikigraph_refresh_pending, name)
        assert not hasattr(wiki_wikigraph_compat_lib, name)

    for old_name in (
        f"PENDING_{old_backend.upper()}_REFRESH_LEDGER",
        f"DEFAULT_PENDING_{old_backend.upper()}_REFRESH_THRESHOLD",
        f"clear_{old_backend}_refresh_pending_after_success",
        f"default_{old_backend}_refresh_ledger",
        f"{old_backend}_refresh_import_summary",
        f"load_{old_backend}_refresh_ledger",
        f"mark_{old_backend}_refresh_pending",
        f"pending_{old_backend}_refresh_ledger_path",
        f"pending_{old_backend}_refresh_status",
        f"record_{old_backend}_refresh_failure",
        f"save_{old_backend}_refresh_ledger",
    ):
        assert not hasattr(wiki_wikigraph_compat_lib, old_name)

    module_text = (SCRIPTS / f"{new_module}.py").read_text(encoding="utf-8")
    for pattern in (
        "PENDING_WIKIGRAPH_REFRESH_LEDGER =",
        "DEFAULT_PENDING_WIKIGRAPH_REFRESH_THRESHOLD =",
        "def pending_wikigraph_refresh_ledger_path(",
        "def default_wikigraph_refresh_ledger(",
        "def load_wikigraph_refresh_ledger(",
        "def save_wikigraph_refresh_ledger(",
        "def wikigraph_refresh_import_summary(",
        "def _parse_refresh_time(",
        "wiki_markdown_latest_mtime",
        "def mark_wikigraph_refresh_pending(",
        "def pending_wikigraph_refresh_status(",
        "def clear_wikigraph_refresh_pending_after_success(",
        "def record_wikigraph_refresh_failure(",
    ):
        assert pattern in module_text
    assert "import wiki_wikigraph_compat_lib" not in module_text
    assert "from wiki_wikigraph_compat_lib import" not in module_text
    assert old_backend not in module_text

    text = (SCRIPTS / "wiki_wikigraph_compat_lib.py").read_text(encoding="utf-8")
    assert f"from {new_module} import" not in text
    assert f"from {old_module} import" not in text
    for pattern in (
        f"PENDING_{old_backend.upper()}_REFRESH_LEDGER =",
        f"DEFAULT_PENDING_{old_backend.upper()}_REFRESH_THRESHOLD =",
        f"def pending_{old_backend}_refresh_ledger_path(",
        f"def default_{old_backend}_refresh_ledger(",
        f"def load_{old_backend}_refresh_ledger(",
        f"def save_{old_backend}_refresh_ledger(",
        f"def {old_backend}_refresh_import_summary(",
        "def _parse_refresh_time(",
        "def wiki_markdown_latest_mtime(",
        f"def mark_{old_backend}_refresh_pending(",
        f"def pending_{old_backend}_refresh_status(",
        f"def clear_{old_backend}_refresh_pending_after_success(",
        f"def record_{old_backend}_refresh_failure(",
    ):
        assert pattern not in text


def test_old_http_query_compatibility_module_is_removed() -> None:
    import wiki_wikigraph_compat_lib

    old_backend = "light" + "rag"
    module_name = f"wiki_{old_backend}_http"
    sys.modules.pop(module_name, None)

    assert not (SCRIPTS / f"{module_name}.py").exists()
    with pytest.raises(ModuleNotFoundError):
        __import__(module_name)

    retired_http_helpers = {
        "health",
        "http_json",
        f"load_{old_backend}_api_key",
        f"query_{old_backend}",
        f"query_{old_backend}_data",
        "TERMINAL_STATUSES",
        "SUCCESS_STATUSES",
        "insert_texts",
        "track_status",
        "wait_for_track",
        "manifest_rows",
        "upsert_doc_event",
        "write_manifest_jsonl",
        f"sync_docs_to_{old_backend}",
    }
    for name in retired_http_helpers:
        assert not hasattr(wiki_wikigraph_compat_lib, name)

    text = (SCRIPTS / "wiki_wikigraph_compat_lib.py").read_text(encoding="utf-8")
    assert f"from wiki_{old_backend}_http import" not in text
    for pattern in (
        f"def load_{old_backend}_api_key(",
        "def http_json(",
        "def health(",
        f"def query_{old_backend}(",
        f"def query_{old_backend}_data(",
        "TERMINAL_STATUSES =",
        "SUCCESS_STATUSES =",
        "def insert_texts(",
        "def track_status(",
        "def wait_for_track(",
        "def manifest_rows(",
        "def upsert_doc_event(",
        "def write_manifest_jsonl(",
        f"def sync_docs_to_{old_backend}(",
    ):
        assert pattern not in text


def test_jsonl_read_streams_rows_in_order_and_skips_blank_lines(tmp_path: Path) -> None:
    from wiki_wikigraph_compat_lib import jsonl_read

    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n  {"b": 2}  \n', encoding="utf-8")

    assert jsonl_read(path) == [{"a": 1}, {"b": 2}]


def test_validate_wiki_default_does_not_write_report(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"

    report = validate_wiki(root, state, tmp_path / "work" / "wikigraph")

    assert "report_path" not in report
    assert not state.exists()
    assert not list((state / "validation_reports").glob("*_validate.json"))


def test_validate_wiki_report_uses_native_not_retired_output_fields(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    workdir = tmp_path / "work" / "wikigraph"

    report = validate_wiki(root, state, workdir)

    assert report["native_unresolved_references"] == 0
    assert report["native_state_dir"] == str(state.resolve())
    assert report["native_workdir"] == str(workdir.resolve())
    retired_backend = "light" + "rag"
    assert f"{retired_backend}_unresolved_references" not in report
    assert f"{retired_backend}_state_dir" not in report
    assert f"{retired_backend}_workdir" not in report


def test_validate_wiki_without_write_report_does_not_hash_freshness_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import wiki_native_wiki_checks

    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"

    def fail_if_called(_root: Path) -> dict[str, dict[str, object]]:
        raise AssertionError("validation_input_fingerprints should only run for persisted validation reports")

    monkeypatch.setattr(wiki_native_wiki_checks, "validation_input_fingerprints", fail_if_called)
    report = validate_wiki(root, state, tmp_path / "work" / "wikigraph")

    assert "input_fingerprints" not in report
    assert "schema_version" not in report


def test_validate_wiki_write_report_is_explicit(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"

    report = validate_wiki(root, state, tmp_path / "work" / "wikigraph", write_report=True)

    report_path = Path(report["report_path"])
    assert report_path.exists()
    assert report_path.parent == state / "validation_reports"


def test_validate_wiki_full_report_contains_reusable_freshness_contract(tmp_path: Path) -> None:
    import wiki_wikigraph_compat_lib

    root = validation_reuse_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    workdir = tmp_path / "work" / "wikigraph"

    report = validate_wiki(root, state, workdir, full=True, write_report=True)
    current = wiki_wikigraph_compat_lib.validation_freshness_context(root, state, workdir)
    freshness = wiki_wikigraph_compat_lib.validation_report_is_fresh(
        report,
        current,
        required_surfaces=["index", "compiled", "_meta", "raw"],
        reason="refresh-artifact",
    )

    assert freshness == {"fresh": True, "rejections": []}
    assert report["schema_version"] == wiki_wikigraph_compat_lib.VALIDATION_REPORT_FRESHNESS_SCHEMA_VERSION
    assert report["root"] == str(root.resolve())
    assert report["state_dir"] == str(state.resolve())
    assert report["workdir"] == str(workdir.resolve())
    assert "refresh-artifact" in report["valid_for_reasons"]
    assert "index.md" in report["input_fingerprints"]
    assert "SCHEMA.md" in report["input_fingerprints"]
    assert "raw/clip/2601/26010101_Foo-Paper.md" in report["input_fingerprints"]


def test_validate_wiki_cli_reuses_fresh_report_without_running_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    root = validation_reuse_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    workdir = tmp_path / "work" / "wikigraph"
    report = validate_wiki(root, state, workdir, full=True, write_report=True)
    report_path = Path(report["report_path"])

    def fail_validate(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("fresh validation report should skip validate_wiki()")

    monkeypatch.setattr(validate_wiki_cli, "validate_wiki", fail_validate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_wiki.py",
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--workdir",
            str(workdir),
            "--full",
            "--reuse-validation-report",
            str(report_path),
        ],
    )

    assert validate_wiki_cli.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["validation_reuse"] == {"fresh": True, "path": str(report_path.resolve())}
    assert payload["reused_validation_report"] == str(report_path.resolve())
    assert payload["errors"] == []


def test_validate_wiki_cli_falls_back_when_reuse_report_is_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    root = validation_reuse_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    workdir = tmp_path / "work" / "wikigraph"
    report = validate_wiki(root, state, workdir, full=True, write_report=True)
    report_path = Path(report["report_path"])
    write(root / "index.md", (root / "index.md").read_text(encoding="utf-8") + "\n<!-- changed -->\n")
    calls: list[tuple[Path, Path, Path | None, bool, bool]] = []

    def fake_validate(root_arg: Path, state_arg: Path, workdir_arg: Path | None = None, *, full: bool = False, write_report: bool = False) -> dict[str, object]:
        calls.append((root_arg, state_arg, workdir_arg, full, write_report))
        return {"errors": [], "warnings": [], "validation_ran": True}

    monkeypatch.setattr(validate_wiki_cli, "validate_wiki", fake_validate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_wiki.py",
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--workdir",
            str(workdir),
            "--full",
            "--reuse-validation-report",
            str(report_path),
        ],
    )

    assert validate_wiki_cli.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert calls == [(root, state, workdir, True, False)]
    assert payload["validation_ran"] is True
    assert payload["validation_reuse"]["fresh"] is False
    assert payload["validation_reuse"]["path"] == str(report_path.resolve())
    assert "fingerprint_mismatch:index.md" in payload["validation_reuse"]["rejections"]


def test_validation_split_module_reexports_existing_public_functions() -> None:
    import wiki_native_validation

    old_backend = "light" + "rag"
    compat_module_name = "wiki_wikigraph_compat_lib"
    validation_module_name = f"wiki_{old_backend}_validation"
    compat_lib = importlib.import_module(compat_module_name)

    assert not (SCRIPTS / f"{validation_module_name}.py").exists()
    sys.modules.pop(validation_module_name, None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(validation_module_name)

    assert compat_lib.validate_wiki is wiki_native_validation.validate_wiki
    assert compat_lib.secret_hits is wiki_native_validation.secret_hits

    text = (SCRIPTS / f"{compat_module_name}.py").read_text(encoding="utf-8")
    assert "from wiki_native_validation import secret_hits, validate_wiki" in text
    assert f"from {validation_module_name} import secret_hits, validate_wiki" not in text


def _fresh_validation_report_inputs() -> tuple[dict[str, object], dict[str, object]]:
    fingerprint = {"sha256": "abc123", "size": 12, "mtime_ns": 345}
    report = {
        "schema_version": 1,
        "generated_at": "2026-06-28 04:00",
        "root": "/wiki",
        "state_dir": "/state",
        "workdir": "/workdir",
        "covered_surfaces": ["compiled", "_meta", "raw"],
        "valid_for_reasons": ["refresh-artifact", "wiki-clear-success"],
        "warnings": [],
        "errors": [],
        "input_fingerprints": {"index.md": fingerprint},
    }
    current = {
        "schema_version": 1,
        "root": "/wiki",
        "state_dir": "/state",
        "workdir": "/workdir",
        "input_fingerprints": {"index.md": fingerprint},
    }
    return report, current


def test_validation_report_freshness_accepts_matching_report() -> None:
    import wiki_wikigraph_compat_lib

    report, current = _fresh_validation_report_inputs()

    result = wiki_wikigraph_compat_lib.validation_report_is_fresh(
        report,
        current,
        required_surfaces=["compiled", "_meta"],
        reason="refresh-artifact",
    )

    assert result == {"fresh": True, "rejections": []}


@pytest.mark.parametrize(
    ("mutator", "required_surfaces", "reason", "expected_rejection"),
    [
        (lambda report, current: report.update({"warnings": ["secret warning"]}), ["compiled"], "refresh-artifact", "report_has_warnings"),
        (lambda report, current: report.update({"errors": ["broken_wikilinks=1"]}), ["compiled"], "refresh-artifact", "report_has_errors"),
        (lambda report, current: report.update({"schema_version": 0}), ["compiled"], "refresh-artifact", "schema_version_mismatch"),
        (lambda report, current: None, ["queries"], "refresh-artifact", "missing_required_surface:queries"),
        (lambda report, current: None, ["compiled"], "final-status", "missing_valid_reason:final-status"),
        (
            lambda report, current: current["input_fingerprints"].update({"index.md": {"sha256": "changed", "size": 12, "mtime_ns": 345}}),
            ["compiled"],
            "refresh-artifact",
            "fingerprint_mismatch:index.md",
        ),
        (lambda report, current: report.update({"input_fingerprints": {}}), ["compiled"], "refresh-artifact", "missing_fingerprint:index.md"),
        (
            lambda report, current: report["input_fingerprints"].update({"deleted.md": {"sha256": "gone", "size": 4, "mtime_ns": 5}}),
            ["compiled"],
            "refresh-artifact",
            "stale_fingerprint:deleted.md",
        ),
    ],
)
def test_validation_report_freshness_rejects_stale_or_unsafe_reports(mutator, required_surfaces: list[str], reason: str, expected_rejection: str) -> None:
    import wiki_wikigraph_compat_lib

    report, current = _fresh_validation_report_inputs()
    mutator(report, current)

    result = wiki_wikigraph_compat_lib.validation_report_is_fresh(report, current, required_surfaces=required_surfaces, reason=reason)

    assert result["fresh"] is False
    assert expected_rejection in result["rejections"]


def test_false_changed_only_flags_are_removed_from_cli_help() -> None:
    for script_name in [
        "validate_wiki.py",
        "build_seed_edges.py",
        "extract_method_atoms.py",
        "extract_raw_sections.py",
        "sync_virtual_docs.py",
    ]:
        result = subprocess.run([sys.executable, str(SCRIPTS / script_name), "--help"], check=True, text=True, capture_output=True)
        assert "--changed-only" not in result.stdout


def test_native_runtime_env_helpers_share_env_and_redaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from native_runtime_env import env_int, load_env_file, redact_summary

    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=secret-value\nEMBEDDING_DIM=1536\nBAD_INT=nope\n", encoding="utf-8")
    monkeypatch.setenv("EMBEDDING_DIM", "3072")

    loaded = load_env_file(env_file)

    assert loaded["EMBEDDING_DIM"] == "1536"
    assert env_int("EMBEDDING_DIM", 1) == 3072
    assert env_int("BAD_INT", 9) == 9
    assert redact_summary({"OPENAI_API_KEY": loaded["OPENAI_API_KEY"], "MODEL": "x"}) == {"OPENAI_API_KEY": "[REDACTED]", "MODEL": "x"}


def test_build_evidence_pack_alias_preserves_output_and_records_query_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import build_evidence_pack
    import wiki_search

    state = tmp_path / "state"
    workdir = tmp_path / "work"

    def fake_http_json(method: str, url: str, payload: dict, *, timeout: int = 60) -> dict:
        return {"response": f"answer for {payload['query']}", "references": []}

    monkeypatch.setattr(wiki_search, "http_json", fake_http_json)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_evidence_pack.py",
            "alias query",
            "--state-dir",
            str(state),
            "--workdir",
            str(workdir),
            "--server",
            "http://127.0.0.1:9",
        ],
    )

    assert build_evidence_pack.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert set(payload) == {"evidence_pack"}
    pack = Path(payload["evidence_pack"])
    assert pack.exists()
    with sqlite3.connect(state / "wikigraph_sync.db") as conn:
        rows = conn.execute("SELECT query, mode, evidence_pack_path FROM query_events").fetchall()
    assert rows == [("alias query", "mix", str(pack))]


# Retired wikigraph pending writes are no longer part of production behavior.

def test_pending_wikigraph_refresh_status_reads_existing_ledger_without_writes(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    ensure_state_dirs(state)
    ledger_path = state / "pending_wikigraph_refresh.json"
    ledger_path.write_text(
        json.dumps(
            {
                "version": 1,
                "threshold": 2,
                "last_successful_refresh_at": None,
                "last_successful_raw_count": None,
                "last_successful_import_payload": {},
                "pending": [
                    {"raw_path": "raw/clip/2601/26010101_Foo-Paper.md", "title": "Foo Paper"},
                    {"raw_path": "raw/clip/2601/26010102_Bar-Paper.md", "title": "Bar Paper"},
                ],
                "dirty": True,
                "last_failed_refresh": None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    status = pending_wikigraph_refresh_status(root, state, reason="threshold")

    assert status["retired"] is True
    assert status["pending_count"] == 2
    assert status["threshold"] == 2
    assert status["would_refresh_if_unblocked"] is True
    assert status["should_refresh"] is False
    assert status["next_required_action"] == "native_refresh"
    assert json.loads(ledger_path.read_text(encoding="utf-8"))["dirty"] is True
    with pytest.raises(RuntimeError, match="ledger writes are retired"):
        mark_wikigraph_refresh_pending(state, root, raw_path="raw/clip/2601/26010103_Baz.md")
    with pytest.raises(RuntimeError, match="ledger writes are retired"):
        clear_wikigraph_refresh_pending_after_success(root, state)


def test_pending_wikigraph_refresh_status_is_readonly_when_state_is_absent(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "absent" / "state"

    status = pending_wikigraph_refresh_status(root, state, reason="threshold")

    assert status["retired"] is True
    assert status["should_refresh"] is False
    assert status["pending_count"] == 0
    assert status["raw_fast_pending_wiki_integration_count"] == 0
    assert not state.exists()


def test_mark_pending_wiki_integration_tracks_raw_fast_queue_without_wiki_pollution(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    entry = mark_pending_wiki_integration(
        state,
        root,
        raw_path="raw/clip/2601/26010101_Foo-Paper.md",
        title="Foo Paper",
        source_id="https://arxiv.org/abs/2601.0101",
        topic_hints=["agents", "rag"],
        required_sections=["summary", "methodology"],
        resource_status_summary="official abs/pdf verified",
    )
    ledger = load_pending_wiki_integration_ledger(state)
    assert entry["status"] == "raw_saved"
    assert ledger["threshold"] == DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD == 10
    assert ledger["dirty"] is True
    assert len(ledger["pending"]) == 1
    assert ledger["pending"][0]["source_id"] == "https://arxiv.org/abs/2601.0101"
    assert ledger["pending"][0]["topic_hints"] == ["agents", "rag"]
    assert (state / "pending_wiki_integration.json").exists()
    assert not (root / "pending_wiki_integration.json").exists()
    assert wiki_root_machine_pollution(root) == []
    status = pending_wiki_integration_status(root, state)
    assert status["pending_count"] == 1
    assert status["should_integrate"] is False


def test_pending_wiki_integration_status_triggers_at_threshold_and_clears_after_success(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    for idx in range(9):
        mark_pending_wiki_integration(state, root, raw_path=f"raw/clip/2601/260101{idx:02d}_Paper.md", title=f"Paper {idx}")
    below = pending_wiki_integration_status(root, state)
    assert below["pending_count"] == 9
    assert below["actionable_pending_count"] == 9
    assert below["threshold"] == DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD == 10
    assert below["should_integrate"] is False

    mark_pending_wiki_integration(state, root, raw_path="raw/clip/2601/26010109_Paper.md", title="Paper 9")
    status = pending_wiki_integration_status(root, state)
    assert status["pending_count"] == 10
    assert status["actionable_pending_count"] == 10
    assert status["threshold"] == DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD == 10
    assert status["should_integrate"] is True
    assert "pending_threshold_reached" in status["reasons"]

    cleared = clear_pending_wiki_integration_after_success(root, state, reason="threshold")
    assert cleared["cleared_count"] == 10
    assert cleared["remaining_pending_count"] == 0
    ledger = load_pending_wiki_integration_ledger(state)
    assert ledger["pending"] == []
    assert ledger["dirty"] is False
    assert ledger["last_successful_integration_raw_count"] == 1


def test_pending_wiki_integration_status_uses_persisted_threshold(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    for idx in range(5):
        mark_pending_wiki_integration(state, root, raw_path=f"raw/clip/2601/260102{idx:02d}_Paper.md", title=f"Paper {idx}", threshold=5)

    status = pending_wiki_integration_status(root, state)

    assert status["pending_count"] == 5
    assert status["actionable_pending_count"] == 5
    assert status["threshold"] == 5
    assert status["should_integrate"] is True
    assert "pending_threshold_reached" in status["reasons"]


def test_terminal_wiki_integration_statuses_do_not_trigger_threshold_or_wikigraph_block(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    for idx in range(10):
        mark_pending_wiki_integration(state, root, raw_path=f"raw/clip/2601/260103{idx:02d}_Duplicate.md", title=f"Duplicate {idx}", status="skipped_duplicate")

    wiki_status = pending_wiki_integration_status(root, state)
    graph_status = pending_wikigraph_refresh_status(root, state, reason="pre-query")

    assert wiki_status["pending_count"] == 10
    assert wiki_status["actionable_pending_count"] == 0
    assert wiki_status["terminal_pending_count"] == 10
    assert wiki_status["should_integrate"] is False
    assert graph_status["blocked_by_pending_wiki_integration"] is False
    assert graph_status["raw_fast_pending_wiki_integration_count"] == 0


def test_review_wiki_integration_status_blocks_wikigraph_with_manual_review_action(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    mark_pending_wiki_integration(state, root, raw_path="raw/clip/2601/26010400_Needs-Review.md", title="Needs Review", status="needs_review")

    wiki_status = pending_wiki_integration_status(root, state, reason="pre-query")
    graph_status = pending_wikigraph_refresh_status(root, state, reason="pre-query")

    assert wiki_status["actionable_pending_count"] == 0
    assert wiki_status["review_pending_count"] == 1
    assert "pending_items_need_review" in wiki_status["reasons"]
    assert graph_status["blocked_by_pending_wiki_integration"] is True
    assert graph_status["next_required_action"] == "manual_review"
    assert graph_status["raw_fast_pending_wiki_integration_count"] == 1


def test_wiki_integration_plan_is_order_independent_and_keeps_ambiguous_items_in_review_queue(tmp_path: Path) -> None:
    from wiki_integration_plan import build_wiki_integration_plan

    root_a = sample_wiki(tmp_path / "a")
    root_b = sample_wiki(tmp_path / "b")
    state_a = tmp_path / "a" / "state"
    state_b = tmp_path / "b" / "state"
    items = [
        {
            "raw_path": "raw/clip/2601/26010102_Routed.md",
            "title": "Routed",
            "source_id": "source:routed",
            "topic_hints": ["retrieval", "agents"],
            "required_sections": ["summary"],
        },
        {
            "raw_path": "raw/clip/2601/26010101_Ambiguous.md",
            "title": "Ambiguous",
            "source_id": "source:ambiguous",
            "topic_hints": [],
            "required_sections": ["summary"],
        },
    ]
    for item in items:
        mark_pending_wiki_integration(state_a, root_a, **item)
    for item in reversed(items):
        mark_pending_wiki_integration(state_b, root_b, **item)

    plan_a = build_wiki_integration_plan(root_a, state_a, reason="manual")
    plan_b = build_wiki_integration_plan(root_b, state_b, reason="manual")

    assert plan_a["plan_hash"] == plan_b["plan_hash"]
    assert plan_a["dry_run"] is True
    assert plan_a["writes_wiki"] is False
    assert plan_a["compiled_page_writes"] == []
    operations = plan_a["operations"]
    assert [op["raw_path"] for op in operations if op["op"] == "raw_map_upsert"] == [
        "raw/clip/2601/26010102_Routed.md"
    ]
    review_ops = [op for op in operations if op["op"] == "review_queue_add"]
    assert review_ops == [
        {
            "op": "review_queue_add",
            "raw_path": "raw/clip/2601/26010101_Ambiguous.md",
            "title": "Ambiguous",
            "reason": "missing_topic_hints",
        }
    ]
    assert wiki_root_machine_pollution(root_a) == []


def test_clear_pending_wiki_integration_marks_integrated_items_for_native_refresh(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    first = "raw/clip/2601/26010102_Raw-Fast-A.md"
    second = "raw/clip/2601/26010103_Raw-Fast-B.md"
    mark_pending_wiki_integration(state, root, raw_path=first, title="Raw Fast A", required_sections=["summary", "methodology"])
    mark_pending_wiki_integration(state, root, raw_path=second, title="Raw Fast B", required_sections=["summary"])

    cleared = clear_pending_wiki_integration_after_success(root, state, integrated_paths=[first], reason="threshold")

    assert cleared["cleared_count"] == 1
    assert cleared["remaining_pending_count"] == 1
    assert cleared["marked_native_pending_count"] == 1
    wiki_ledger = load_pending_wiki_integration_ledger(state)
    assert [item["raw_path"] for item in wiki_ledger["pending"]] == [second]
    assert wiki_ledger["dirty"] is True
    assert not (state / "pending_wikigraph_refresh.json").exists()
    native_entries = batch_native_refresh.pending_entries(state)
    assert len(native_entries) == 1
    assert native_entries[0]["reason"] == "wiki-integration:threshold"


def test_clear_pending_wiki_integration_without_integrated_paths_marks_native_refresh_once(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    first = "raw/clip/2601/26010102_Raw-Fast-A.md"
    second = "raw/clip/2601/26010103_Raw-Fast-B.md"
    mark_pending_wiki_integration(state, root, raw_path=first, title="Raw Fast A")
    mark_pending_wiki_integration(state, root, raw_path=second, title="Raw Fast B")

    cleared = clear_pending_wiki_integration_after_success(root, state, reason="threshold")

    assert cleared["cleared_count"] == 2
    assert cleared["remaining_pending_count"] == 0
    assert cleared["marked_native_pending_count"] == 1
    assert load_pending_wiki_integration_ledger(state)["pending"] == []
    assert not (state / "pending_wikigraph_refresh.json").exists()
    assert len(batch_native_refresh.pending_entries(state)) == 1


# Retired batch_wikigraph_refresh behavior is now covered by tests/test_wikigraph_refresh.py.


def test_batch_wiki_integration_cli_status_mark_and_clear_are_external(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    script = SCRIPTS / "batch_wiki_integration.py"
    mark = subprocess.run(
        [
            sys.executable,
            str(script),
            "mark-pending",
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--raw-path",
            "raw/clip/2601/26010101_Foo-Paper.md",
            "--title",
            "Foo Paper",
            "--source-id",
            "https://arxiv.org/abs/2601.0101",
            "--topic-hint",
            "agents",
            "--required-section",
            "methodology",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    mark_payload = json.loads(mark.stdout)
    assert mark_payload["pending_count"] == 1
    assert mark_payload["threshold"] == DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD == 10
    assert mark_payload["pending"][0]["topic_hints"] == ["agents"]
    assert mark_payload["pending"][0]["required_sections"] == ["methodology"]

    status = subprocess.run([sys.executable, str(script), "status", "--root", str(root), "--state-dir", str(state)], check=True, text=True, capture_output=True)
    assert json.loads(status.stdout)["should_integrate"] is False
    assert not (root / "pending_wiki_integration.json").exists()

    clear = subprocess.run(
        [sys.executable, str(script), "clear-success", "--root", str(root), "--state-dir", str(state), "--integrated-path", "raw/clip/2601/26010101_Foo-Paper.md"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(clear.stdout)["cleared_count"] == 1
    assert load_pending_wiki_integration_ledger(state)["pending"] == []


def test_batch_wiki_integration_prompt_uses_repo_local_workdir_paths(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wiki-graph" / "state"
    status = {
        "pending_count": 1,
        "actionable_pending_count": 1,
        "threshold": 10,
        "actionable_pending": [
            {
                "raw_path": "raw/clip/2606/26062906_skill-neologisms-towards-skill-based-continual-learning.md",
                "title": "Skill Neologisms",
                "source_id": "arXiv:2605.04970v2",
                "topic_hints": ["skill neologisms", "continual learning"],
            }
        ],
    }

    prompt = batch_wiki_integration.build_auto_integration_prompt(root, state, status, "manual")

    assert ("/home/" + "xu/project/wiki/wikigraph") not in prompt
    assert f"Native refresh workdir: `{SCRIPTS.parent}`" in prompt
    assert f"python {SCRIPTS / 'batch_wiki_integration.py'} clear-success" in prompt
    assert f"python {SCRIPTS / 'batch_native_refresh.py'} status" in prompt


def test_batch_wiki_integration_auto_integrate_runs_configured_runner_at_threshold_and_requires_cleared_ledger(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    script = SCRIPTS / "batch_wiki_integration.py"
    for idx in range(10):
        mark_pending_wiki_integration(state, root, raw_path=f"raw/clip/2601/260106{idx:02d}_Paper.md", title=f"Paper {idx}", required_sections=["summary"])
    fake_runner = tmp_path / "fake_wiki_integrator.py"
    write(
        fake_runner,
        "import os, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
        "from wiki_native_lib import clear_pending_wiki_integration_after_success\n"
        "root = Path(os.environ['LLM_WIKI_ROOT'])\n"
        "state = Path(os.environ['LLM_WIKI_STATE_DIR'])\n"
        "prompt_path = Path(os.environ['LLM_WIKI_INTEGRATION_PROMPT'])\n"
        "seen = state / 'fake_runner_seen.txt'\n"
        "seen.write_text(prompt_path.read_text(encoding='utf-8')[:500], encoding='utf-8')\n"
        "clear_pending_wiki_integration_after_success(root, state, reason=os.environ.get('LLM_WIKI_INTEGRATION_REASON', 'threshold'))\n",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "auto-integrate",
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--reason",
            "threshold",
            "--integration-command",
            f"{sys.executable} {fake_runner}",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["ran"] is True
    assert payload["pre_status"]["should_integrate"] is True
    assert payload["post_status"]["should_integrate"] is False
    assert payload["post_status"]["pending_count"] == 0
    assert payload["prompt_path"].endswith(".md")
    assert "batch wiki integration" in (state / "fake_runner_seen.txt").read_text(encoding="utf-8")
    assert load_pending_wiki_integration_ledger(state)["pending"] == []
    assert len(batch_native_refresh.pending_entries(state)) == 1
    assert not (state / "pending_wikigraph_refresh.json").exists()


def test_batch_wiki_integration_auto_integrate_records_failure_if_runner_leaves_ledger_pending(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    script = SCRIPTS / "batch_wiki_integration.py"
    for idx in range(10):
        mark_pending_wiki_integration(state, root, raw_path=f"raw/clip/2601/260107{idx:02d}_Paper.md", title=f"Paper {idx}")
    noop_runner = tmp_path / "noop_integrator.py"
    write(noop_runner, "print('noop integration runner returned without clearing ledger')\n")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "auto-integrate",
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--reason",
            "threshold",
            "--integration-command",
            f"{sys.executable} {noop_runner}",
        ],
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 12
    assert payload["ran"] is True
    assert payload["post_status"]["should_integrate"] is True
    assert payload["failure"]["reason"] == "auto-integrate-incomplete"
    assert load_pending_wiki_integration_ledger(state)["last_failed_integration"]["reason"] == "auto-integrate-incomplete"


def test_machine_pollution_detection(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    assert wiki_root_machine_pollution(root) == []
    write(root / ".llm-wiki" / "bad.txt", "bad")
    retired_backend = "light" + "rag"
    retired_manifest_name = f"{retired_backend}_manifest.jsonl"
    write(root / retired_manifest_name, "{}\n")
    polluted = {p.as_posix() for p in wiki_root_machine_pollution(root)}
    assert ".llm-wiki" in polluted
    assert retired_manifest_name in polluted


def test_ingest_text_has_stable_machine_header(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    doc = next(doc for doc in collect_source_docs(root) if doc.rel_path == "concepts/foo.md")
    text = make_ingest_text(doc)
    assert text.startswith("[LLM_WIKI_DOC]\n")
    assert "canonical_id: compiled:concept:foo" in text
    assert "path: concepts/foo.md" in text
    assert "doc_type: compiled_concept" in text
    assert "[/LLM_WIKI_DOC]" in text
    assert "# Foo" in text


def test_ingest_text_preserves_raw_note_tags_and_topic_hints_as_indexing_cues(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    raw = root / "raw/clip/2601/26010101_Foo-Paper.md"
    raw.write_text(
        raw.read_text(encoding="utf-8").replace(
            "tags: [paper]\n",
            "tags: [paper, benchmark]\ntopic_hints: [\"continual learning\", \"stateful evaluation\"]\n",
        ),
        encoding="utf-8",
    )
    doc = next(doc for doc in collect_source_docs(root) if doc.rel_path == "raw/clip/2601/26010101_Foo-Paper.md")
    text = make_ingest_text(doc)

    assert "tags: paper; benchmark" in text
    assert "topic_hints: continual learning; stateful evaluation" in text
    assert "source_pdf" not in text.split("[/LLM_WIKI_DOC]", 1)[0]


def test_custom_kg_payload_reuses_external_seed_edges(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    build_seed_edges(root, state)
    payload, summary = build_custom_kg_payload(root, state, limit_docs=3, limit_edges=2)
    assert summary["chunks"] == 3
    assert summary["relationships"] <= 2
    assert all(chunk["file_path"].endswith(".md") for chunk in payload["chunks"])
    entity_names = {entity["entity_name"] for entity in payload["entities"]}
    assert "compiled:concept:foo" in entity_names
    assert payload["relationships"]
    assert {"src_id", "tgt_id", "description", "keywords", "source_id"} <= set(payload["relationships"][0])


def test_custom_kg_payload_includes_method_atoms_without_wiki_pollution(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    ensure_state_dirs(state)
    write(
        state / "method_atom_docs" / "method-demo.md",
        "\n".join(
            [
                "[LLM_WIKI_METHOD_ATOM]",
                "atom_id: method:demo:001",
                "source_path: raw/clip/2601/26010101_Foo-Paper.md",
                "[/LLM_WIKI_METHOD_ATOM]",
                "",
                "# MethodAtom Demo",
            ]
        ),
    )
    build_seed_edges(root, state)
    payload, summary = build_custom_kg_payload(root, state, limit_docs=2, limit_edges=0)
    assert summary["chunks"] == 3
    entity_by_name = {entity["entity_name"]: entity for entity in payload["entities"]}
    assert entity_by_name["method:demo:001"]["entity_type"] == "LLM_WIKI_METHOD_ATOM"
    assert any(rel["keywords"] == "METHOD_ATOM_FROM" for rel in payload["relationships"])
    assert not (root / ".llm-wiki").exists()


def test_extract_raw_sections_indexes_summary_heading_as_retrieval_section(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    write(
        root / "raw/clip/2601/26010105_Summary-Section.md",
        "---\ntitle: Summary Section\nupdated: 2026-05-19 04:10\n---\n"
        "# Summary Section\n\n"
        "## 一句话总结\n\n"
        "A short summary should become a raw_section summary node for paper-level semantic neighborhoods.\n",
    )
    state = tmp_path / "work" / "wikigraph" / "state"
    extract_raw_sections(root, state)
    docs = generated_docs_from_state(state, kind="raw_section")
    summary_docs = [doc for doc in docs if doc.canonical_id == "raw_section:26010105_Summary-Section:summary"]
    assert len(summary_docs) == 1
    assert "section_kind: summary" in summary_docs[0].text
    assert "section_title: 一句话总结" in summary_docs[0].text


def test_generated_docs_from_state_is_read_only_when_state_is_missing(tmp_path: Path) -> None:
    state = tmp_path / "work" / "wikigraph" / "state"

    assert generated_docs_from_state(state, kind="raw_section") == []
    assert not state.exists()


def test_extract_raw_sections_recognizes_summary_motivation_methodology_title_variants(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    write(
        root / "raw" / "clip" / "2601" / "26010102_Variant-Sections.md",
        "---\ntitle: Variant Sections\nupdated: 2026-05-19 01:20\n---\n"
        "# Variant Sections\n\n"
        "## 论文摘要（中文）\n\n"
        "This abstract variant summarizes the paper contribution.\n\n"
        "## 研究动机 / 为什么要重新审视检索增强推理\n\n"
        "This motivation variant explains the unresolved setup pressure.\n\n"
        "## 方法拆解\n\n"
        "This methodology variant details a staged retrieval and verification pipeline.\n\n"
        "## 明确失败案例说明方法不是万能\n\n"
        "This caveat heading contains 方法 but is not the methodology section.\n",
    )
    state = tmp_path / "work" / "wikigraph" / "state"
    extract_raw_sections(root, state)
    docs = generated_docs_from_state(state, kind="raw_section")
    variant_docs = {doc.canonical_id.rsplit(":", 1)[-1]: doc for doc in docs if "Variant-Sections" in doc.canonical_id}
    assert {"abstract", "motivation", "methodology"} <= set(variant_docs)
    assert "section_title: 论文摘要（中文）" in variant_docs["abstract"].text
    assert "section_title: 研究动机 / 为什么要重新审视检索增强推理" in variant_docs["motivation"].text
    assert "section_title: 方法拆解" in variant_docs["methodology"].text
    assert "This caveat heading contains 方法" not in variant_docs["methodology"].text


def test_extract_raw_sections_indexes_combined_limitation_future_heading_for_both_kinds(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    write(
        root / "raw/clip/2601/26010103_Combined-Sections.md",
        "---\ntitle: Combined Sections\nupdated: 2026-05-19 03:00\n---\n"
        "# Combined Sections\n\n"
        "## 局限 / Future Works\n\n"
        "The same source section discusses remaining failure modes and follow-up research directions.\n",
    )
    state = tmp_path / "work" / "wikigraph" / "state"
    extract_raw_sections(root, state)
    docs = generated_docs_from_state(state, kind="raw_section")
    combined = {doc.canonical_id.rsplit(":", 1)[-1]: doc for doc in docs if "Combined-Sections" in doc.canonical_id}
    assert {"future", "limitations"} <= set(combined)
    assert "section_title: 局限 / Future Works" in combined["future"].text
    assert "section_title: 局限 / Future Works" in combined["limitations"].text
    assert {spec["kind"] for spec in raw_section_specs_for_heading("局限 / Future Works")} == {"future", "limitations"}


def test_extract_raw_sections_integrates_formula_and_figure_evidence_into_context_sections(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    write(
        root / "raw/clip/2601/26010106_Visual-Evidence.md",
        "---\ntitle: Visual Evidence\ndomain: paper\nupdated: 2026-05-19 04:20\n---\n"
        "# Visual Evidence\n\n"
        "## Methodology\n\n"
        "- Formula evidence is integrated here: Eq. (3) defines the routing objective; symbols and the baseline delta are interpreted in the method narrative.\n\n"
        "## 关键实验结果 / 作者结论\n\n"
        "- Figure 2 has three panels; the x-axis, y-axis, trend, and supported claim are recorded next to the result it supports.\n",
    )
    state = tmp_path / "work" / "wikigraph" / "state"
    extract_raw_sections(root, state)
    docs = generated_docs_from_state(state, kind="raw_section")
    visual_docs = {doc.canonical_id.rsplit(":", 1)[-1]: doc for doc in docs if "Visual-Evidence" in doc.canonical_id}
    assert {"methodology", "results"} <= set(visual_docs)
    assert "section_kind: methodology" in visual_docs["methodology"].text
    assert "Eq. (3) defines the routing objective" in visual_docs["methodology"].text
    assert "section_kind: results" in visual_docs["results"].text
    assert "Figure 2 has three panels" in visual_docs["results"].text
    assert raw_section_specs_for_heading("关键公式 / 机制推导") == []
    assert raw_section_specs_for_heading("关键图表 / 读图笔记") == []
    assert "equation" in raw_section_query_for_kind("methodology", "routing objective")
    assert "figure" in raw_section_query_for_kind("results", "ablation trend")


def test_raw_note_section_contract_audit_reports_combined_and_duplicate_headings(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    write(
        root / "raw/clip/2601/26010104_Strict-But-Messy.md",
        "---\ntitle: Strict But Messy\ndomain: paper\nupdated: 2026-05-19 03:00\n---\n"
        "# Strict But Messy\n\n"
        "## 一句话总结\n\nA compact take.\n\n"
        "## 论文摘要（中文）\n\nAbstract content.\n\n"
        "## Motivation\n\nMotivation content.\n\n"
        "## 方法拆解\n\nPrimary method content.\n\n"
        "## Method: ablation detail\n\nA duplicate method-style section that should be visible to the audit.\n\n"
        "## 局限 / Future Works\n\nCombined limits and future work content.\n\n"
        "## 可继续追问的问题\n\nQuestion content.\n",
    )
    audit = audit_raw_note_section_contracts(root)
    issues = [issue for issue in audit["issues"] if issue["path"].endswith("Strict-But-Messy.md")]
    assert any(issue["type"] == "duplicate_section_kind" and issue["section_kind"] == "methodology" for issue in issues)
    assert any(issue["type"] == "combined_section_heading" and set(issue["section_kinds"]) == {"future", "limitations"} for issue in issues)
    assert audit["issues_by_type"]["duplicate_section_kind"] >= 1


def test_extract_raw_sections_creates_section_virtual_docs_without_wiki_pollution(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    ensure_state_dirs(state)
    write(state / "raw_section_docs" / "stale.md", "stale")
    result = extract_raw_sections(root, state)
    assert result["raw_sections"] == 4
    assert not (state / "raw_section_docs" / "stale.md").exists()
    docs = generated_docs_from_state(state, kind="raw_section")
    by_id = {doc.canonical_id: doc for doc in docs}
    assert "raw_section:26010101_Foo-Paper:future" in by_id
    assert "raw_section:26010101_Foo-Paper:limitations" in by_id
    assert "raw_section:26010101_Foo-Paper:questions" in by_id
    assert "raw_section:26010101_Foo-Paper:methodology" in by_id
    future = by_id["raw_section:26010101_Foo-Paper:future"].text
    assert "section_kind: future" in future
    assert "section_title: 对未来研究的启发" in future
    assert "Future work should connect memory repair" in future
    assert not (root / ".llm-wiki").exists()


def test_extract_raw_sections_uses_unique_filenames_for_long_raw_stems(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    shared = "Long-Section-Retrieval-" + "A" * 150
    for suffix in ["X", "Y"]:
        write(
            root / "raw" / "clip" / "2601" / f"26010100_{shared}{suffix}.md",
            "---\ntitle: Long Stem Paper\nupdated: 2026-05-18 16:00\n---\n# Long Stem Paper\n\n## 对未来研究的启发\n\n- A unique long-stem future section should survive as a separate virtual doc.\n",
        )
    state = tmp_path / "work" / "wikigraph" / "state"
    result = extract_raw_sections(root, state)
    files = list((state / "raw_section_docs").glob("*.md"))
    assert len(files) == result["raw_sections"]


def test_custom_kg_payload_includes_raw_section_chunks_and_relationships(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    extract_raw_sections(root, state)
    build_seed_edges(root, state)
    payload, summary = build_custom_kg_payload(root, state, limit_docs=2, limit_edges=0)
    assert summary["raw_section_chunks"] == 4
    assert summary["chunks"] == 6
    section_chunks = [chunk for chunk in payload["chunks"] if chunk["source_id"].startswith("raw_section:")]
    assert {chunk["source_id"].rsplit(":", 1)[-1] for chunk in section_chunks} == {"methodology", "future", "limitations", "questions"}
    assert any("section_kind: limitations" in chunk["content"] for chunk in section_chunks)
    assert any("section_kind: methodology" in chunk["content"] for chunk in section_chunks)
    entity_by_name = {entity["entity_name"]: entity for entity in payload["entities"]}
    assert entity_by_name["raw_section:26010101_Foo-Paper:questions"]["entity_type"] == "LLM_WIKI_RAW_SECTION"
    assert any(rel["keywords"] == "RAW_SECTION_OF" and rel["tgt_id"] == "raw_clip:26010101_Foo-Paper" for rel in payload["relationships"])


def test_custom_kg_manifest_resolves_sources_and_preserves_typed_relationships(tmp_path: Path) -> None:
    from custom_kg_incremental import build_custom_kg_manifest, relation_vdb_id, stable_hash

    payload = {
        "chunks": [
            {"content": "Doc A content", "source_id": "doc:a", "file_path": "a.md", "chunk_order_index": 0},
        ],
        "entities": [
            {"entity_name": "doc:a", "entity_type": "DOC", "description": "Doc A", "source_id": "doc:a", "file_path": "a.md"},
            {"entity_name": "topic:x", "entity_type": "TOPIC", "description": "Topic", "source_id": "doc:a", "file_path": "a.md"},
        ],
        "relationships": [
            {"src_id": "topic:x", "tgt_id": "doc:a", "description": "old", "keywords": "OLD", "source_id": "doc:a", "weight": 0.1, "file_path": "old.md"},
            {"src_id": "doc:a", "tgt_id": "topic:x", "description": "new", "keywords": "NEW", "source_id": "doc:a", "weight": 0.9, "file_path": "new.md"},
        ],
    }

    manifest = build_custom_kg_manifest(payload, native_manifest_tool_version="native-test", embedding_model="test-embed", embedding_dim=3)

    metadata = manifest["metadata"]
    retired_backend = "light" + "rag"
    assert metadata["native_manifest_tool_version"] == "native-test"
    assert "wikigraph_tool_version" not in metadata
    assert f"{retired_backend}_version" not in metadata
    assert metadata["canonical_id_algorithm"] == "llm-wiki-canonical-id:v1+native-custom-kg:v1"
    assert metadata["relationship_vector_content_algorithm"] == "llm-wiki-typed-directed-relationship:v1"
    assert retired_backend not in metadata["canonical_id_algorithm"]
    assert retired_backend not in metadata["relationship_vector_content_algorithm"]

    chunk_id = next(iter(manifest["chunks"]))
    chunk = manifest["chunks"][chunk_id]
    assert manifest["source_to_chunk"]["doc:a"] == chunk_id
    assert chunk["record_type"] == "chunk"
    assert chunk["record_id"] == chunk_id
    assert chunk["canonical_id"] == chunk_id
    assert chunk["vector_text_hash"] == stable_hash(chunk["content"])
    assert manifest["entities"]["topic:x"]["source_chunk_id"] == chunk_id
    topic = manifest["entities"]["topic:x"]
    assert topic["record_type"] == "entity"
    assert topic["record_id"] == topic["vdb_id"]
    assert topic["canonical_id"] == "topic:x"
    assert topic["vector_text_hash"] == stable_hash(topic["content"])
    assert "UNKNOWN" not in json.dumps(manifest, ensure_ascii=False)
    assert len(manifest["relationships"]) == 2
    rels_by_keyword = {record["keywords"]: record for record in manifest["relationships"].values()}
    assert set(rels_by_keyword) == {"OLD", "NEW"}
    old_rel = rels_by_keyword["OLD"]
    new_rel = rels_by_keyword["NEW"]
    assert old_rel["description"] == "old"
    assert old_rel["src_id"] == "topic:x"
    assert old_rel["tgt_id"] == "doc:a"
    assert old_rel["vdb_id"] == relation_vdb_id("topic:x", "doc:a", "OLD")
    assert new_rel["description"] == "new"
    assert new_rel["src_id"] == "doc:a"
    assert new_rel["tgt_id"] == "topic:x"
    assert new_rel["source_chunk_id"] == chunk_id
    assert new_rel["vdb_id"] == relation_vdb_id("doc:a", "topic:x", "NEW")
    assert new_rel["record_type"] == "relationship"
    assert new_rel["record_id"] == new_rel["vdb_id"]
    assert new_rel["canonical_id"] == new_rel["chunk_key"]
    assert new_rel["vector_text_hash"] == stable_hash(new_rel["content"])


def test_custom_kg_manifest_matches_wikigraph_sanitized_chunk_ids_and_basenames() -> None:
    import custom_kg_incremental
    from custom_kg_incremental import build_custom_kg_manifest, compute_mdhash_id, wikigraph_sanitize_text

    retired_backend = "light" + "rag"
    assert not hasattr(custom_kg_incremental, f"{retired_backend}_sanitize_text")
    assert not hasattr(custom_kg_incremental, f"{retired_backend}_normalize_file_path")

    raw_content = "  A &amp; B\x1f  "
    payload = {
        "chunks": [
            {"content": raw_content, "source_id": "doc:sanitized", "file_path": "nested/doc.[native-iet].md", "chunk_order_index": 0},
        ],
        "entities": [
            {"entity_name": "doc:sanitized", "entity_type": "DOC", "description": "Doc", "source_id": "doc:sanitized", "file_path": "nested/doc.[native-iet].md"},
        ],
        "relationships": [],
    }

    manifest = build_custom_kg_manifest(payload, native_manifest_tool_version="1.5.0", embedding_model="test-embed", embedding_dim=3)
    sanitized = wikigraph_sanitize_text(raw_content)
    chunk_id = compute_mdhash_id(sanitized, prefix="chunk-")

    assert sanitized == "A & B"
    assert custom_kg_incremental.wikigraph_normalize_file_path("nested/doc.[native-iet].md") == "doc.md"
    assert set(manifest["chunks"]) == {chunk_id}
    assert manifest["chunks"][chunk_id]["content"] == sanitized
    assert manifest["chunks"][chunk_id]["file_path"] == "doc.md"
    assert manifest["entities"]["doc:sanitized"]["source_chunk_id"] == chunk_id
    assert manifest["entities"]["doc:sanitized"]["file_path"] == "doc.md"


def test_custom_kg_incremental_source_uses_native_runtime_wording() -> None:
    text = (SCRIPTS / "custom_kg_incremental.py").read_text(encoding="utf-8")
    retired_backend = "light" + "rag"

    assert retired_backend not in text.lower()
    assert "native zvec" in text.lower()
    assert "importlib.import_module(retired_graph_module_name" not in text
    assert "importlib.metadata.version" not in text
    assert "current_wikigraph_tool_version" not in text
    assert "def _resolve_wikigraph_tool_version_arg" not in text
    assert '"wikigraph_tool_version":' not in text
    assert "wikigraph-custom-kg" not in text


def test_import_custom_kg_source_uses_wikigraph_external_graph_wording() -> None:
    text = (SCRIPTS / "import_custom_kg.py").read_text(encoding="utf-8")
    retired_backend = "light" + "rag"

    assert retired_backend not in text.lower()
    assert "wikigraph" in text.lower()


def test_wikigraph_compat_tests_use_wikigraph_workdir_fixture_paths() -> None:
    text = (Path(__file__).read_text(encoding="utf-8"))
    retired_backend = "light" + "rag"
    retired_path_fragment = f'"work" / "{retired_backend}"'

    assert retired_path_fragment not in text
    assert '"work" / "wikigraph"' in text


def test_wikigraph_compat_tests_use_wikigraph_status_key_names() -> None:
    text = Path(__file__).read_text(encoding="utf-8")
    retired_backend = "light" + "rag"
    forbidden_tokens = [
        f"test_{retired_backend}",
        f"{retired_backend}_status",
        f"{retired_backend}_state_dir",
        f"{retired_backend}_workdir",
        f"{retired_backend}_unresolved",
        f"{retired_backend}_block",
        f"{retired_backend}_manifest",
    ]

    for token in forbidden_tokens:
        assert token not in text


def test_wikigraph_compat_tests_construct_fake_external_module_names() -> None:
    text = Path(__file__).read_text(encoding="utf-8")
    retired_backend = "light" + "rag"
    fake_class_token = "Fake" + "Light" + "RAG"
    forbidden_tokens = [
        f"fake_{retired_backend}",
        fake_class_token,
        f'types.ModuleType("{retired_backend}',
        f'sys.modules, "{retired_backend}',
    ]

    for token in forbidden_tokens:
        assert token not in text


def test_native_sync_db_sources_use_wikigraph_schema_names() -> None:
    retired_backend = "light" + "rag"
    sources = [
        SCRIPTS / "wiki_native_query_events.py",
        SCRIPTS / "wiki_native_wiki_checks.py",
        Path(__file__),
    ]
    forbidden_tokens = [
        f"{retired_backend}_sync.db",
        f"{retired_backend}_track_id",
        f"{retired_backend}_doc_status",
    ]

    for source in sources:
        text = source.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in text


def test_successful_manifest_stamps_metadata_without_mutating_desired_manifest() -> None:
    from custom_kg_incremental import build_custom_kg_manifest, successful_manifest

    payload = {
        "chunks": [{"content": "Doc A", "source_id": "doc:a", "file_path": "a.md"}],
        "entities": [{"entity_name": "doc:a", "entity_type": "DOC", "description": "Doc A", "source_id": "doc:a", "file_path": "a.md"}],
        "relationships": [],
    }
    desired = build_custom_kg_manifest(payload, native_manifest_tool_version="1.5.0", embedding_model="test", embedding_dim=3)
    original_metadata = dict(desired["metadata"])

    final = successful_manifest(desired, import_mode="incremental", previous_manifest=desired)

    assert desired["metadata"] == original_metadata
    assert final["metadata"]["last_successful_import_mode"] == "incremental"
    assert final["metadata"]["incremental_count_since_full"] == 1
    assert final["chunks"] == desired["chunks"]
    assert final["entities"] == desired["entities"]
    assert final["relationships"] == desired["relationships"]


# External tool-python version fallback was removed with the retired backend runner.


def test_custom_kg_manifest_diff_tracks_add_update_delete() -> None:
    from custom_kg_incremental import build_custom_kg_manifest, diff_custom_kg_manifests

    old_payload = {
        "chunks": [{"content": "Doc A v1", "source_id": "doc:a", "file_path": "a.md"}],
        "entities": [{"entity_name": "doc:a", "entity_type": "DOC", "description": "v1", "source_id": "doc:a", "file_path": "a.md"}],
        "relationships": [{"src_id": "doc:a", "tgt_id": "topic:x", "description": "edge", "keywords": "RELATED", "source_id": "doc:a", "file_path": "a.md"}],
    }
    new_payload = {
        "chunks": [{"content": "Doc A v2", "source_id": "doc:a", "file_path": "a.md"}],
        "entities": [
            {"entity_name": "doc:a", "entity_type": "DOC", "description": "v2", "source_id": "doc:a", "file_path": "a.md"},
            {"entity_name": "topic:y", "entity_type": "TOPIC", "description": "new", "source_id": "doc:a", "file_path": "a.md"},
        ],
        "relationships": [],
    }

    diff = diff_custom_kg_manifests(
        build_custom_kg_manifest(old_payload, native_manifest_tool_version="1.5.0", embedding_model="test", embedding_dim=3),
        build_custom_kg_manifest(new_payload, native_manifest_tool_version="1.5.0", embedding_model="test", embedding_dim=3),
    )

    assert diff["chunks"]["add"] == 1
    assert diff["chunks"]["delete"] == 1
    assert diff["entities"]["add"] == 1
    assert diff["entities"]["update"] == 1
    assert diff["relationships"]["delete"] == 1


def test_custom_kg_diff_splits_metadata_only_relationship_and_entity_updates() -> None:
    from custom_kg_incremental import build_custom_kg_manifest, diff_custom_kg_manifests, relationship_record_key

    chunks = [
        {"content": "Doc A", "source_id": "doc:a", "file_path": "a.md"},
        {"content": "Doc B", "source_id": "doc:b", "file_path": "b.md"},
    ]
    old_payload = {
        "chunks": chunks,
        "entities": [{"entity_name": "topic:x", "entity_type": "TOPIC", "description": "same", "source_id": "doc:a", "file_path": "old.md"}],
        "relationships": [{"src_id": "doc:a", "tgt_id": "topic:x", "description": "same edge", "keywords": "RELATED", "source_id": "doc:a", "file_path": "old.md"}],
    }
    new_payload = {
        "chunks": chunks,
        "entities": [{"entity_name": "topic:x", "entity_type": "TOPIC", "description": "same", "source_id": "doc:b", "file_path": "new.md"}],
        "relationships": [{"src_id": "doc:a", "tgt_id": "topic:x", "description": "same edge", "keywords": "RELATED", "source_id": "doc:b", "file_path": "new.md"}],
    }

    diff = diff_custom_kg_manifests(
        build_custom_kg_manifest(old_payload, native_manifest_tool_version="1.5.0", embedding_model="test", embedding_dim=3),
        build_custom_kg_manifest(new_payload, native_manifest_tool_version="1.5.0", embedding_model="test", embedding_dim=3),
    )
    rel_key = relationship_record_key("doc:a", "topic:x", "RELATED")

    assert diff["entities"]["update_ids"] == ["topic:x"]
    assert diff["entities"]["metadata_update_ids"] == ["topic:x"]
    assert diff["entities"]["vector_update_ids"] == []
    assert diff["relationships"]["update_ids"] == [rel_key]
    assert diff["relationships"]["metadata_update_ids"] == [rel_key]
    assert diff["relationships"]["vector_update_ids"] == []


def test_custom_kg_vector_hash_includes_embedding_contract() -> None:
    from custom_kg_incremental import build_custom_kg_manifest

    payload = {
        "chunks": [{"content": "Doc A", "source_id": "doc:a", "file_path": "a.md"}],
        "entities": [{"entity_name": "topic:x", "entity_type": "TOPIC", "description": "same", "source_id": "doc:a", "file_path": "a.md"}],
        "relationships": [{"src_id": "doc:a", "tgt_id": "topic:x", "description": "same edge", "keywords": "RELATED", "source_id": "doc:a", "file_path": "a.md"}],
    }

    base = build_custom_kg_manifest(payload, native_manifest_tool_version="1.5.0", embedding_model="embed-a", embedding_dim=3, embedding_params_version="v1")
    model_changed = build_custom_kg_manifest(payload, native_manifest_tool_version="1.5.0", embedding_model="embed-b", embedding_dim=3, embedding_params_version="v1")
    dim_changed = build_custom_kg_manifest(payload, native_manifest_tool_version="1.5.0", embedding_model="embed-a", embedding_dim=4, embedding_params_version="v1")
    params_changed = build_custom_kg_manifest(payload, native_manifest_tool_version="1.5.0", embedding_model="embed-a", embedding_dim=3, embedding_params_version="v2")

    entity_hash = next(iter(base["entities"].values()))["vector_hash"]
    relationship_hash = next(iter(base["relationships"].values()))["vector_hash"]
    chunk_hash = next(iter(base["chunks"].values()))["vector_hash"]

    assert next(iter(model_changed["entities"].values()))["vector_hash"] != entity_hash
    assert next(iter(dim_changed["relationships"].values()))["vector_hash"] != relationship_hash
    assert next(iter(params_changed["chunks"].values()))["vector_hash"] != chunk_hash


def test_custom_kg_metadata_only_change_preserves_vector_hash_with_embedding_contract() -> None:
    from custom_kg_incremental import build_custom_kg_manifest

    chunks = [{"content": "Doc A", "source_id": "doc:a", "file_path": "a.md"}]
    old_manifest = build_custom_kg_manifest(
        {
            "chunks": chunks,
            "entities": [{"entity_name": "topic:x", "entity_type": "TOPIC", "description": "same", "source_id": "doc:a", "file_path": "old.md"}],
            "relationships": [{"src_id": "doc:a", "tgt_id": "topic:x", "description": "same edge", "keywords": "RELATED", "source_id": "doc:a", "file_path": "old.md"}],
        },
        native_manifest_tool_version="1.5.0",
        embedding_model="embed-a",
        embedding_dim=3,
        embedding_params_version="v1",
    )
    new_manifest = build_custom_kg_manifest(
        {
            "chunks": chunks,
            "entities": [{"entity_name": "topic:x", "entity_type": "TOPIC", "description": "same", "source_id": "doc:a", "file_path": "new.md"}],
            "relationships": [{"src_id": "doc:a", "tgt_id": "topic:x", "description": "same edge", "keywords": "RELATED", "source_id": "doc:a", "file_path": "new.md"}],
        },
        native_manifest_tool_version="1.5.0",
        embedding_model="embed-a",
        embedding_dim=3,
        embedding_params_version="v1",
    )

    assert next(iter(old_manifest["entities"].values()))["vector_hash"] == next(iter(new_manifest["entities"].values()))["vector_hash"]
    assert next(iter(old_manifest["relationships"].values()))["vector_hash"] == next(iter(new_manifest["relationships"].values()))["vector_hash"]


def test_custom_kg_diff_derives_split_hashes_for_legacy_manifest_records() -> None:
    from custom_kg_incremental import build_custom_kg_manifest, diff_custom_kg_manifests, relationship_record_key

    chunks = [
        {"content": "Doc A", "source_id": "doc:a", "file_path": "a.md"},
        {"content": "Doc B", "source_id": "doc:b", "file_path": "b.md"},
    ]
    old_manifest = build_custom_kg_manifest(
        {
            "chunks": chunks,
            "entities": [{"entity_name": "topic:x", "entity_type": "TOPIC", "description": "same", "source_id": "doc:a", "file_path": "old.md"}],
            "relationships": [{"src_id": "doc:a", "tgt_id": "topic:x", "description": "same edge", "keywords": "RELATED", "source_id": "doc:a", "file_path": "old.md"}],
        },
        native_manifest_tool_version="1.5.0",
        embedding_model="test",
        embedding_dim=3,
    )
    new_manifest = build_custom_kg_manifest(
        {
            "chunks": chunks,
            "entities": [{"entity_name": "topic:x", "entity_type": "TOPIC", "description": "same", "source_id": "doc:b", "file_path": "new.md"}],
            "relationships": [{"src_id": "doc:a", "tgt_id": "topic:x", "description": "same edge", "keywords": "RELATED", "source_id": "doc:b", "file_path": "new.md"}],
        },
        native_manifest_tool_version="1.5.0",
        embedding_model="test",
        embedding_dim=3,
    )
    rel_key = relationship_record_key("doc:a", "topic:x", "RELATED")
    for collection in ("entities", "relationships"):
        for record in old_manifest[collection].values():
            record.pop("vector_hash", None)
            record.pop("metadata_hash", None)

    diff = diff_custom_kg_manifests(old_manifest, new_manifest)

    assert diff["entities"]["metadata_update_ids"] == ["topic:x"]
    assert diff["entities"]["vector_update_ids"] == []
    assert diff["relationships"]["metadata_update_ids"] == [rel_key]
    assert diff["relationships"]["vector_update_ids"] == []


# Direct storage patching via ExternalGraph APIs was removed with the retired live-storage runner.


def test_incremental_refresh_mode_requires_manifest_and_full_after_five_incrementals() -> None:
    from custom_kg_incremental import build_custom_kg_manifest, choose_refresh_mode

    payload = {
        "chunks": [{"content": "Doc A", "source_id": "doc:a", "file_path": "a.md"}],
        "entities": [{"entity_name": "doc:a", "entity_type": "DOC", "description": "Doc A", "source_id": "doc:a", "file_path": "a.md"}],
        "relationships": [],
    }
    desired = build_custom_kg_manifest(payload, native_manifest_tool_version="1.5.0", embedding_model="test", embedding_dim=3)

    no_manifest = choose_refresh_mode(None, desired, native_preflight_ok=True, full_rebuild_interval=5)
    assert no_manifest["selected_mode"] == "full_rebuild"
    assert "missing_manifest" in no_manifest["reasons"]

    previous = json.loads(json.dumps(desired))
    previous["metadata"]["incremental_count_since_full"] = 4
    fifth_incremental = choose_refresh_mode(previous, desired, native_preflight_ok=True, full_rebuild_interval=5)
    assert fifth_incremental["selected_mode"] == "incremental"
    assert fifth_incremental["next_incremental_count_since_full"] == 5

    previous["metadata"]["incremental_count_since_full"] = 5
    after_five = choose_refresh_mode(previous, desired, native_preflight_ok=True, full_rebuild_interval=5)
    assert after_five["selected_mode"] == "full_rebuild"
    assert "incremental_interval_reached" in after_five["reasons"]


# Retired incremental apply runner behavior is covered by direct fail-closed tests; low-level diff helpers remain tested above.


def test_custom_kg_storage_audit_is_retired_without_reading_old_storage(tmp_path: Path) -> None:
    from custom_kg_incremental import audit_custom_kg_storage

    storage = tmp_path / "rag_storage"
    storage.mkdir()
    (storage / "graph_chunk_entity_relation.graphml").write_text("not graphml", encoding="utf-8")

    with pytest.raises(RuntimeError, match="retired"):
        audit_custom_kg_storage(storage, {"chunks": {}, "entities": {}, "relationships": {}})

    source = (SCRIPTS / "custom_kg_incremental.py").read_text(encoding="utf-8")
    forbidden = ["_load_vdb", "_load_kv", "nx.read_graphml", "CUSTOM_KG_STORAGE_AUDIT_FILES", "storage_audit_ok", "current_storage_audit_failed"]
    assert [token for token in forbidden if token in source] == []


def test_section_similarity_embedding_text_uses_clean_section_content_without_sidecar_boilerplate() -> None:
    section = {
        "section_id": "raw_section:26010101_Foo-Paper:future",
        "source_id": "raw_clip:26010101_Foo-Paper",
        "source_path": "raw/clip/2601/26010101_Foo-Paper.md",
        "paper_title": "Foo Paper",
        "section_kind": "future",
        "section_title": "对未来研究的启发",
        "content": "- Future work should connect memory repair with section-level evidence retrieval.",
    }
    text = section_similarity_embedding_text(section)
    assert "Title: Foo Paper" in text
    assert "Section kind: future" in text
    assert "Future work should connect memory repair" in text
    assert "LLM_WIKI_RAW_SECTION" not in text
    assert "RAW_SECTION_OF" not in text


def test_build_section_similarity_edges_keeps_sparse_mutual_edges_and_excludes_same_raw_note() -> None:
    sections = [
        {"section_id": "raw_section:a:future", "source_id": "raw_clip:a", "source_path": "raw/clip/a.md", "paper_title": "A", "section_kind": "future", "section_title": "Future", "content": "alpha"},
        {"section_id": "raw_section:b:future", "source_id": "raw_clip:b", "source_path": "raw/clip/b.md", "paper_title": "B", "section_kind": "future", "section_title": "Future", "content": "alpha neighbor"},
        {"section_id": "raw_section:c:future", "source_id": "raw_clip:c", "source_path": "raw/clip/c.md", "paper_title": "C", "section_kind": "future", "section_title": "Future", "content": "orthogonal"},
        {"section_id": "raw_section:a:limitations", "source_id": "raw_clip:a", "source_path": "raw/clip/a.md", "paper_title": "A", "section_kind": "limitations", "section_title": "Limitations", "content": "same raw note should be excluded"},
        {"section_id": "raw_section:d:questions", "source_id": "raw_clip:d", "source_path": "raw/clip/d.md", "paper_title": "D", "section_kind": "questions", "section_title": "Questions", "content": "problem neighbor"},
    ]
    embeddings = {
        "raw_section:a:future": [1.0, 0.0, 0.0],
        "raw_section:b:future": [0.96, 0.28, 0.0],
        "raw_section:c:future": [0.0, 1.0, 0.0],
        "raw_section:a:limitations": [1.0, 0.0, 0.0],
        "raw_section:d:questions": [1.0, 0.0, 0.0],
    }
    edges = build_section_similarity_edges(
        sections,
        embeddings,
        same_kind_k=1,
        cross_kind_k=1,
        same_kind_min_cosine=0.9,
        cross_kind_min_cosine=0.9,
        cross_kind_pairs=[("future", "questions"), ("future", "limitations")],
        mutual=True,
    )
    edge_pairs = {(edge["src_id"], edge["tgt_id"], edge["source_section_kind"], edge["target_section_kind"]) for edge in edges}
    assert ("raw_section:a:future", "raw_section:b:future", "future", "future") in edge_pairs
    assert ("raw_section:a:future", "raw_section:d:questions", "future", "questions") in edge_pairs
    assert all(not {edge["src_id"], edge["tgt_id"]} == {"raw_section:a:future", "raw_section:a:limitations"} for edge in edges)
    assert all(edge["type"] == "SEMANTIC_SECTION_NEIGHBOR" for edge in edges)
    assert all(edge["mutual_knn"] for edge in edges)


def test_section_similarity_index_round_trips_full_builder_edges(tmp_path: Path) -> None:
    from wiki_wikigraph_compat_lib import build_section_similarity_edges_from_index, section_similarity_index_summary

    sections = [
        {"section_id": "raw_section:a:future", "source_id": "raw_clip:a", "source_path": "raw/clip/a.md", "paper_title": "A", "section_kind": "future", "section_title": "Future", "content": "alpha"},
        {"section_id": "raw_section:b:future", "source_id": "raw_clip:b", "source_path": "raw/clip/b.md", "paper_title": "B", "section_kind": "future", "section_title": "Future", "content": "alpha peer"},
        {"section_id": "raw_section:c:future", "source_id": "raw_clip:c", "source_path": "raw/clip/c.md", "paper_title": "C", "section_kind": "future", "section_title": "Future", "content": "other peer"},
        {"section_id": "raw_section:d:questions", "source_id": "raw_clip:d", "source_path": "raw/clip/d.md", "paper_title": "D", "section_kind": "questions", "section_title": "Questions", "content": "question peer"},
        {"section_id": "raw_section:e:questions", "source_id": "raw_clip:e", "source_path": "raw/clip/e.md", "paper_title": "E", "section_kind": "questions", "section_title": "Questions", "content": "question peer 2"},
    ]
    embeddings = {
        "raw_section:a:future": [1.0, 0.0, 0.0],
        "raw_section:b:future": [0.96, 0.28, 0.0],
        "raw_section:c:future": [0.7, 0.714142842854285, 0.0],
        "raw_section:d:questions": [1.0, 0.0, 0.0],
        "raw_section:e:questions": [0.96, 0.28, 0.0],
    }
    index_path = tmp_path / "section_similarity_index.sqlite"

    full_edges = build_section_similarity_edges(
        sections,
        embeddings,
        same_kind_k=2,
        cross_kind_k=1,
        same_kind_min_cosine=0.7,
        cross_kind_min_cosine=0.9,
        cross_kind_pairs=[("future", "questions")],
        mutual=True,
        embedding_model="test-embedding",
        embedding_dim=3,
        index_path=index_path,
    )
    indexed_edges = build_section_similarity_edges_from_index(
        index_path,
        sections,
        embeddings,
        cross_kind_pairs=[("future", "questions")],
        mutual=True,
        embedding_model="test-embedding",
        embedding_dim=3,
    )

    def comparable(edge: dict[str, object]) -> tuple[object, ...]:
        return (
            edge["edge_id"],
            edge["src_id"],
            edge["tgt_id"],
            edge["pair_kind"],
            edge["cosine"],
            edge["source_rank"],
            edge["target_rank"],
        )

    assert [comparable(edge) for edge in indexed_edges] == [comparable(edge) for edge in full_edges]
    summary = section_similarity_index_summary(index_path)
    assert summary["directed_rows"] >= len(full_edges)
    assert summary["family_count"] >= 2


def test_build_section_similarity_graph_writes_section_similarity_index_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import build_section_similarity_graph
    from wiki_wikigraph_compat_lib import jsonl_read, section_similarity_index_summary

    root = tmp_path / "wiki"
    state = tmp_path / "state"
    workdir = tmp_path / "workdir"
    state.mkdir(parents=True)
    workdir.mkdir(parents=True)
    rows = [
        {"section_id": "raw_section:a:future", "source_id": "raw_clip:a", "source_path": "raw/clip/a.md", "paper_title": "A", "section_kind": "future", "section_title": "Future", "content": "alpha"},
        {"section_id": "raw_section:b:future", "source_id": "raw_clip:b", "source_path": "raw/clip/b.md", "paper_title": "B", "section_kind": "future", "section_title": "Future", "content": "alpha peer"},
    ]
    write(state / "raw_sections.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

    monkeypatch.setattr(
        build_section_similarity_graph,
        "embedding_config",
        lambda _workdir: {"model": "test-embedding", "embedding_dim": 3, "env": {}, "batch_size": 10},
    )
    monkeypatch.setattr(
        build_section_similarity_graph,
        "build_embedding_rows",
        lambda rows_arg, config, cache_path, reuse_cache=True: (
            [
                {**rows_arg[0], "text_hash": "h-a", "embedding_model": config["model"], "embedding_dim": 3, "embedding": [1.0, 0.0, 0.0]},
                {**rows_arg[1], "text_hash": "h-b", "embedding_model": config["model"], "embedding_dim": 3, "embedding": [1.0, 0.0, 0.0]},
            ],
            {"cache_hits": 0, "embedded": 2, "total": 2, "cache_path": cache_path.as_posix()},
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_section_similarity_graph.py",
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--workdir",
            str(workdir),
            "--same-kind-k",
            "1",
            "--same-kind-min-cosine",
            "0.9",
            "--cross-kind-pairs",
            "",
            "--min-content-chars",
            "1",
            "--sample-edges",
            "1",
        ],
    )

    assert build_section_similarity_graph.main() == 0
    index_path = state / "section_similarity_index.sqlite"
    summary = section_similarity_index_summary(index_path)
    assert summary["directed_rows"] == 2
    reports = sorted((state / "section_similarity_reports").glob("*_section_similarity_report.json"))
    assert reports
    report = json.loads(reports[-1].read_text(encoding="utf-8"))
    assert report["section_similarity_index"]["directed_rows"] == 2
    assert len(jsonl_read(state / "section_similarity_edges.candidates.jsonl")) == 1


def test_build_section_similarity_graph_reports_provider_failure_without_partial_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import build_section_similarity_graph

    root = tmp_path / "wiki"
    state = tmp_path / "state"
    workdir = tmp_path / "workdir"
    state.mkdir(parents=True)
    workdir.mkdir(parents=True)
    row = {
        "section_id": "raw_section:a:future",
        "source_id": "raw_clip:a",
        "source_path": "raw/clip/a.md",
        "paper_title": "A",
        "section_kind": "future",
        "section_title": "Future",
        "content": "alpha content long enough for embedding",
    }
    write(state / "raw_sections.jsonl", json.dumps(row, ensure_ascii=False) + "\n")
    for name in [
        "EMBEDDING_BINDING_HOST",
        "OPENAI_BASE_URL",
        "EMBEDDING_BINDING_API_KEY",
        "OPENAI_API_KEY",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_section_similarity_graph.py",
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--workdir",
            str(workdir),
            "--section-kinds",
            "future",
            "--min-content-chars",
            "1",
        ],
    )

    assert build_section_similarity_graph.main() == 1

    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is False
    assert printed["error"] == "section_embedding_failed"
    assert printed["section_count"] == 1
    assert "EMBEDDING_BINDING_HOST or OPENAI_BASE_URL is required" in printed["message"]
    assert not (state / "section_embeddings.jsonl").exists()
    assert not (state / "section_similarity_edges.candidates.jsonl").exists()
    assert list((state / "section_similarity_reports").glob("*_section_similarity_report.json")) == []


def test_section_rank_lists_fast_matches_scalar_reference_on_boundary_fixture() -> None:
    sections = [
        {"section_id": "raw_section:src:future", "source_id": "raw_clip:src", "source_path": "raw/clip/src.md", "paper_title": "Src", "section_kind": "future", "section_title": "Future", "content": "source"},
        {"section_id": "raw_section:a_target:future", "source_id": "raw_clip:a", "source_path": "raw/clip/a.md", "paper_title": "A", "section_kind": "future", "section_title": "Future", "content": "tie a"},
        {"section_id": "raw_section:b_target:future", "source_id": "raw_clip:b", "source_path": "raw/clip/b.md", "paper_title": "B", "section_kind": "future", "section_title": "Future", "content": "tie b"},
        {"section_id": "raw_section:c_below:future", "source_id": "raw_clip:c", "source_path": "raw/clip/c.md", "paper_title": "C", "section_kind": "future", "section_title": "Future", "content": "below threshold"},
        {"section_id": "raw_section:same_note:future", "source_id": "raw_clip:src", "source_path": "raw/clip/src.md", "paper_title": "Src", "section_kind": "future", "section_title": "Future", "content": "same raw note"},
        {"section_id": "raw_section:q1:questions", "source_id": "raw_clip:q1", "source_path": "raw/clip/q1.md", "paper_title": "Q1", "section_kind": "questions", "section_title": "Questions", "content": "cross"},
        {"section_id": "raw_section:q2:questions", "source_id": "raw_clip:q2", "source_path": "raw/clip/q2.md", "paper_title": "Q2", "section_kind": "questions", "section_title": "Questions", "content": "cross lower"},
    ]
    embeddings = {
        "raw_section:src:future": [1.0, 0.0, 0.0],
        "raw_section:a_target:future": [0.8, 0.6, 0.0],
        "raw_section:b_target:future": [0.8, 0.6, 0.0],
        "raw_section:c_below:future": [0.719999, 0.6949759247363014, 0.0],
        "raw_section:same_note:future": [1.0, 0.0, 0.0],
        "raw_section:q1:questions": [0.76, 0.6499230723708769, 0.0],
        "raw_section:q2:questions": [0.759999, 0.6499242414779487, 0.0],
    }

    fast_same = _section_rank_lists(sections, embeddings, "future", "future", 2, 0.72)
    scalar_same = _section_rank_lists_scalar(sections, embeddings, "future", "future", 2, 0.72)
    fast_cross = _section_rank_lists(sections, embeddings, "future", "questions", 1, 0.76)
    scalar_cross = _section_rank_lists_scalar(sections, embeddings, "future", "questions", 1, 0.76)

    assert fast_same == scalar_same
    assert fast_cross == scalar_cross
    assert ("raw_section:src:future", "raw_section:a_target:future") in fast_same
    assert ("raw_section:src:future", "raw_section:b_target:future") in fast_same
    assert ("raw_section:src:future", "raw_section:c_below:future") not in fast_same
    assert ("raw_section:src:future", "raw_section:same_note:future") not in fast_same
    assert fast_same[("raw_section:src:future", "raw_section:a_target:future")]["rank"] == 1
    assert fast_same[("raw_section:src:future", "raw_section:b_target:future")]["rank"] == 2


def test_section_rank_lists_fast_falls_back_to_scalar_for_zero_and_mismatched_vectors() -> None:
    sections = [
        {"section_id": "raw_section:zero:future", "source_id": "raw_clip:zero", "source_path": "raw/clip/zero.md", "paper_title": "Zero", "section_kind": "future", "section_title": "Future", "content": "zero"},
        {"section_id": "raw_section:short:future", "source_id": "raw_clip:short", "source_path": "raw/clip/short.md", "paper_title": "Short", "section_kind": "future", "section_title": "Future", "content": "short"},
        {"section_id": "raw_section:long:future", "source_id": "raw_clip:long", "source_path": "raw/clip/long.md", "paper_title": "Long", "section_kind": "future", "section_title": "Future", "content": "long"},
    ]
    embeddings = {
        "raw_section:zero:future": [0.0, 0.0],
        "raw_section:short:future": [1.0, 0.0],
        "raw_section:long:future": [1.0, 0.0, 0.0],
    }

    assert _section_rank_lists(sections, embeddings, "future", "future", 2, 0.0) == _section_rank_lists_scalar(
        sections, embeddings, "future", "future", 2, 0.0
    )


def test_section_similarity_report_summary_counts_edges_and_hubs() -> None:
    sections = [
        {"section_id": "raw_section:a:future", "section_kind": "future"},
        {"section_id": "raw_section:b:future", "section_kind": "future"},
        {"section_id": "raw_section:c:questions", "section_kind": "questions"},
    ]
    edges = [
        {"src_id": "raw_section:a:future", "tgt_id": "raw_section:b:future", "pair_kind": "future:future", "cosine": 0.91},
        {"src_id": "raw_section:a:future", "tgt_id": "raw_section:c:questions", "pair_kind": "future:questions", "cosine": 0.84},
    ]
    report = section_similarity_report_summary(sections, edges)
    assert report["section_count"] == 3
    assert report["section_count_by_kind"] == {"future": 2, "questions": 1}
    assert report["edge_count"] == 2
    assert report["edge_count_by_pair_kind"] == {"future:future": 1, "future:questions": 1}
    assert report["top_hubs"][0]["section_id"] == "raw_section:a:future"
    assert report["cosine_by_pair_kind"]["future:future"]["max"] == 0.91


def test_select_section_similarity_edges_marks_reviewed_high_value_pairs() -> None:
    candidates = [
        {"src_id": "a", "tgt_id": "b", "pair_kind": "summary:summary", "cosine": 0.9},
        {"src_id": "c", "tgt_id": "d", "pair_kind": "future:future", "cosine": 0.8},
        {"src_id": "e", "tgt_id": "f", "pair_kind": "limitations:questions", "cosine": 0.81},
    ]
    selected = select_section_similarity_edges(candidates, allowed_pair_kinds={"future:future", "limitations:questions"})
    assert [edge["pair_kind"] for edge in selected] == ["future:future", "limitations:questions"]
    assert all(edge["review_status"] == "phase2_selected" for edge in selected)
    assert all("Semantic proximity only" in edge["review_note"] for edge in selected)


def test_custom_kg_payload_includes_reviewed_section_similarity_edges(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    extract_raw_sections(root, state)
    build_seed_edges(root, state)
    jsonl = state / "section_similarity_edges.jsonl"
    write(
        jsonl,
        json.dumps(
            {
                "edge_id": "semantic_section_neighbor:test",
                "type": "SEMANTIC_SECTION_NEIGHBOR",
                "src_id": "raw_section:26010101_Foo-Paper:future",
                "tgt_id": "raw_section:26010101_Foo-Paper:questions",
                "source_section_kind": "future",
                "target_section_kind": "questions",
                "source_path": "raw/clip/2601/26010101_Foo-Paper.md",
                "target_path": "raw/clip/2601/26010101_Foo-Paper.md",
                "cosine": 0.82,
                "mutual_knn": True,
                "embedding_model": "test-embedding",
            },
            ensure_ascii=False,
        )
        + "\n",
    )
    payload, summary = build_custom_kg_payload(root, state, limit_docs=2, limit_edges=0)
    assert summary["section_similarity_relationships"] == 1
    rels = [rel for rel in payload["relationships"] if rel["keywords"] == "SEMANTIC_SECTION_NEIGHBOR"]
    assert len(rels) == 1
    assert rels[0]["src_id"] == "raw_section:26010101_Foo-Paper:future"
    assert rels[0]["tgt_id"] == "raw_section:26010101_Foo-Paper:questions"
    assert "cosine=0.820000" in rels[0]["description"]


def test_structured_heading_warnings_accept_legacy_raw_heading_variants(tmp_path: Path) -> None:
    path = tmp_path / "raw/clip/2601/26010101_Legacy.md"
    text = """---
title: Legacy Paper
domain: paper
---
# Legacy Paper

## 一句话结论

A compact take.

## 中文摘要 / 核心内容

An abstract variant.

## 研究动机 / 为什么这个问题重要

A motivation variant.

## 方法拆解

A methodology variant with integrated formula evidence: Eq. (2) defines the objective and its variables.

## 关键实验结果 / 作者结论

A result variant with integrated visual evidence: Figure 2 and Table 1 are read next to the claims they support.
"""
    assert structured_heading_warnings(path, text) == []


def test_structured_heading_warnings_report_truly_missing_sections(tmp_path: Path) -> None:
    path = tmp_path / "raw/clip/2601/26010102_Broken.md"
    text = """---
title: Broken Paper
domain: paper
---
# Broken Paper

## 一句话总结

A compact take.
"""
    warnings = structured_heading_warnings(path, text)
    assert len(warnings) == 4
    assert any("missing heading prefix ## 论文摘要" in warning for warning in warnings)
    assert any("missing heading prefix ## Motivation" in warning for warning in warnings)
    assert any("missing heading prefix ## Methodology" in warning for warning in warnings)
    assert any("missing heading prefix ## 关键实验结果" in warning for warning in warnings)
    assert not any("## 关键公式" in warning or "## 关键图表" in warning for warning in warnings)


def test_structured_heading_warnings_skip_legacy_non_structured_arxiv_clippings(tmp_path: Path) -> None:
    path = tmp_path / "raw/clip/2601/26010103_Legacy-Article.md"
    text = """---
title: Legacy Article
domain: "arxiv.org"
source: "https://arxiv.org/abs/2601.0103"
---
# Legacy Article

## Original article

This preserved source clipping predates the structured paper-note schema.
"""
    assert structured_heading_warnings(path, text) == []


def test_section_kind_query_prefix_and_response_filter_target_raw_section_chunks() -> None:
    query = raw_section_query_for_kind("methodology", "retrieval verification")
    assert "raw_section section_kind methodology" in query
    assert "Methodology" in query
    assert "方法拆解" in query
    response = {
        "data": {
            "chunks": [
                {"file_path": "raw_section_docs/a.md", "content": "section_kind: methodology\nA"},
                {"file_path": "raw_section_docs/b.md", "content": "section_kind: abstract\nB"},
                {"file_path": "concepts/c.md", "content": "section_kind: methodology\nC"},
            ],
            "entities": [1],
        },
        "status": "ok",
    }
    filtered = filter_wikigraph_data_response_by_section_kind(response, "methodology")
    chunks = filtered["data"]["chunks"]
    assert len(chunks) == 1
    assert chunks[0]["file_path"] == "raw_section_docs/a.md"
    assert response["data"]["chunks"][1]["file_path"] == "raw_section_docs/b.md"


def test_expand_wikigraph_data_response_with_section_neighbors_keeps_direct_hits_separate(tmp_path: Path) -> None:
    state = tmp_path / "state"
    ensure_state_dirs(state)
    write(
        state / "section_similarity_edges.jsonl",
        json.dumps(
            {
                "src_id": "raw_section:a:future",
                "tgt_id": "raw_section:b:future",
                "source_section_kind": "future",
                "target_section_kind": "future",
                "source_path": "raw/clip/a.md",
                "target_path": "raw/clip/b.md",
                "source_title": "A",
                "target_title": "B",
                "cosine": 0.88,
                "pair_kind": "future:future",
            },
            ensure_ascii=False,
        )
        + "\n",
    )
    response = {
        "data": {
            "chunks": [
                {"file_path": "raw_section_docs/a.md", "content": "section_id: raw_section:a:future\nsection_kind: future\n"},
            ]
        }
    }
    expanded = expand_wikigraph_data_response_with_section_neighbors(response, state, neighbor_k=1, section_kind="future")
    neighbors = expanded["data"]["section_neighbor_expansions"]
    assert len(neighbors) == 1
    assert neighbors[0]["seed_section_id"] == "raw_section:a:future"
    assert neighbors[0]["neighbor_section_id"] == "raw_section:b:future"
    assert neighbors[0]["cosine"] == 0.88
    assert expanded["data"]["chunks"] == response["data"]["chunks"]


def test_import_custom_kg_run_import_retired_before_storage_access(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import asyncio
    import import_custom_kg

    root = sample_wiki(tmp_path)
    workdir = tmp_path / "work" / "wikigraph"
    state_dir = workdir / "state"
    ensure_state_dirs(state_dir)

    def fail_if_called(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("build_rag should not be called by retired cold import")

    monkeypatch.setattr(import_custom_kg, "build_rag", fail_if_called)
    args = types.SimpleNamespace(
        root=root,
        workdir=workdir,
        state_dir=state_dir,
        limit_docs=None,
        limit_edges=None,
        dry_run=False,
        server_host="127.0.0.1",
        server_port=9621,
        allow_server_running=False,
    )

    with pytest.raises(RuntimeError, match="custom KG cold import is retired"):
        asyncio.run(import_custom_kg.run_import(args))

    assert not (state_dir / "custom_kg_manifest.json").exists()
    assert not (state_dir / "custom_kg_import_report.json").exists()


def test_import_custom_kg_build_rag_is_retired_without_constructing_external_graph(tmp_path: Path) -> None:
    from import_custom_kg import build_rag

    with pytest.raises(RuntimeError, match="object construction is retired"):
        build_rag(tmp_path)


def test_generated_virtual_doc_builders_remove_stale_markdown(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    write(
        root / "raw/clip/2601/26010101_Bar-Paper.md",
        "---\ntitle: Bar Paper\nupdated: 2026-05-18 16:00\n---\n# Bar Paper\n\n## Methodology\n\n"
        "This is another direct method with enough detail to become a separate method atom. "
        "It intentionally shares the same date prefix as Foo Paper.\n",
    )
    state = tmp_path / "work" / "wikigraph" / "state"
    ensure_state_dirs(state)
    write(state / "edge_docs" / "stale.md", "stale")
    write(state / "method_atom_docs" / "stale.md", "stale")
    edge_result = build_seed_edges(root, state)
    method_result = extract_method_atoms(root, state)
    assert not (state / "edge_docs" / "stale.md").exists()
    assert not (state / "method_atom_docs" / "stale.md").exists()
    edge_docs = sorted((state / "edge_docs").glob("*.md"))
    edge_doc_contents = {path.name: path.read_text(encoding="utf-8") for path in edge_docs}
    assert len(edge_docs) == edge_result["seed_edges"]
    assert len(list((state / "method_atom_docs").glob("*.md"))) == method_result["method_atoms"]
    assert method_result["method_atoms"] == 2

    second_edge_result = build_seed_edges(root, state)
    second_edge_docs = sorted((state / "edge_docs").glob("*.md"))
    assert second_edge_result["edge_docs_total"] == edge_result["seed_edges"]
    assert second_edge_result["edge_docs_written"] == 0
    assert {path.name: path.read_text(encoding="utf-8") for path in second_edge_docs} == edge_doc_contents


def _write_tiny_pdf(path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Tiny Paper\nAbstract\nThis paper has Eq. (1), Figure 1, Table 1, and https://github.com/example/tiny-paper",
    )
    doc.save(path)
    doc.close()


def test_raw_fast_evidence_bundle_title_guess_strips_markdown_heading_prefix() -> None:
    import raw_fast_evidence_bundle

    assert raw_fast_evidence_bundle.title_from_text("## SWE-Marathon: Can Agents Work?\n\nAbstract") == "SWE-Marathon: Can Agents Work?"


def _assert_timing_step(payload: dict, step: str) -> None:
    timings = payload["timings"]
    entry = timings["steps"][step]
    assert isinstance(entry["elapsed_seconds"], (int, float))
    assert entry["elapsed_seconds"] >= 0


def test_raw_fast_evidence_bundle_direct_pdf_writes_temp_only_and_defaults_docling(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    source_pdf = tmp_path / "source.pdf"
    _write_tiny_pdf(source_pdf)
    workdir = tmp_path / "bundle"
    script = SCRIPTS / "raw_fast_evidence_bundle.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--url",
            source_pdf.as_uri(),
            "--kind",
            "direct-pdf",
            "--root",
            str(root),
            "--workdir",
            str(workdir),
            "--probe",
            "none",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["ok"] is True
    assert payload["kind"] == "direct-pdf"
    assert payload["pdf_backend_requested"] == "docling"
    assert payload["files"]["layout_text"] == "paper.layout.txt"
    assert payload["files"]["note_skeleton"] == "note_skeleton.md"
    assert payload["timings"]["total_seconds"] >= 0
    _assert_timing_step(payload, "fetch_pdf")
    _assert_timing_step(payload, "pdfinfo")
    _assert_timing_step(payload, "pdftotext_layout")
    _assert_timing_step(payload, "pdftotext_raw")
    _assert_timing_step(payload, "resource_probe")
    assert json.loads((workdir / "evidence_bundle.json").read_text(encoding="utf-8"))["timings"]["steps"]["fetch_pdf"]["elapsed_seconds"] >= 0
    assert (workdir / "paper.pdf").exists()
    assert (workdir / "paper.layout.txt").exists()
    assert (workdir / "evidence_bundle.json").exists()
    assert "raw/clip/" in payload["next_raw_path"]
    secret_scan = json.loads((workdir / "secret_scan.json").read_text(encoding="utf-8"))
    assert secret_scan["strict_secret_hits"] == []
    assert not (root / "evidence_bundle.json").exists()
    assert wiki_root_machine_pollution(root) == []


def _write_tiny_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000a49444154789c636000000200015d0b2a0000000049454e44ae426082"))


def _write_tiny_pdf_figure(path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=144, height=96)
    page.draw_rect(fitz.Rect(12, 12, 132, 84), color=(0, 0, 0), fill=(0.9, 0.95, 1.0))
    page.insert_text((24, 48), "PDF Figure")
    doc.save(path)
    doc.close()


def _write_sidecar_source_fixture(workdir: Path) -> None:
    _write_tiny_png(workdir / "source" / "figs" / "method.png")
    write(
        workdir / "source" / "main.tex",
        r"""
\title{Fixture Sidecar Paper}
\begin{document}
\begin{abstract}
This paper introduces Sidecar Method with exact evidence and an official code link.
\end{abstract}
\section{Method}
We optimize $\mathcal{L}=x+y$ and release code at \url{https://github.com/example/sidecar-paper}.
\begin{equation}
\mathcal{L}=x+y
\label{eq:loss}
\end{equation}
\begin{figure}
\includegraphics{figs/method.png}
\caption{Method figure shows the pipeline.}
\label{fig:method}
\end{figure}
\begin{table}
\caption{Main results on the fixture benchmark.}
\label{tab:main}
\begin{tabular}{lr}
Method & Score \\
Sidecar & 42
\end{tabular}
\end{table}
\section{Limitations}
The fixture has one synthetic limitation.
\end{document}
""".strip(),
    )


def test_raw_fast_evidence_bundle_paper_digest_resource_draft_and_local_figures_are_sidecars(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    source_pdf = tmp_path / "source.pdf"
    _write_tiny_pdf(source_pdf)
    workdir = tmp_path / "bundle-sidecars"
    _write_sidecar_source_fixture(workdir)
    script = SCRIPTS / "raw_fast_evidence_bundle.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--url",
            source_pdf.as_uri(),
            "--kind",
            "direct-pdf",
            "--root",
            str(root),
            "--workdir",
            str(workdir),
            "--probe",
            "none",
            "--paper-digest",
            "--resource-draft",
            "--localize-figures",
            "--image-slug",
            "fixture-sidecar-paper",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["ok"] is True
    for key in ["paper_digest", "paper_digest_markdown", "resource_boundary_draft", "resource_boundary_draft_markdown", "note_block_drafts", "localized_figures", "localized_figures_markdown"]:
        assert key in payload["files"]
        assert (workdir / payload["files"][key]).exists()
    for step in ["paper_digest", "resource_boundary_draft", "localize_figures", "note_block_drafts"]:
        _assert_timing_step(payload, step)

    digest = json.loads((workdir / "paper_digest.json").read_text(encoding="utf-8"))
    assert digest["ok"] is True
    assert digest["metadata_card"]["title"] == "Fixture Sidecar Paper"
    assert any(card.get("label") == "eq:loss" and "x+y" in card.get("formula", "") for card in digest["equation_cards"])
    assert any(card.get("label") == "fig:method" and card.get("localizable") is True for card in digest["figure_cards"])
    assert any(card.get("label") == "tab:main" and "Sidecar" in card.get("body_excerpt", "") for card in digest["table_cards"])
    localized = json.loads((workdir / "localized_figures.json").read_text(encoding="utf-8"))
    assert localized["ok"] is True
    localized_entry = localized["entries"][0]
    assert localized_entry["dest_rel"].startswith("localized_figures_assets/fixture-sidecar-paper/")
    assert localized_entry["sha256"]
    assert localized_entry["raw_note_policy"] == "temporary_inspection_only_do_not_embed_markdown_image"
    assert "markdown" not in localized_entry
    assert (workdir / localized_entry["dest_rel"]).exists()
    assert not (root / "raw" / "images").exists()
    assert not (root / "paper_digest.json").exists()
    assert not (root / "resource_boundary_draft.json").exists()
    assert not (root / "note_block_drafts.md").exists()
    assert wiki_root_machine_pollution(root) == []


def test_raw_fast_evidence_bundle_localize_figures_does_not_overwrite_existing_asset(tmp_path: Path) -> None:
    import raw_fast_evidence_bundle

    root = sample_wiki(tmp_path)
    workdir = tmp_path / "bundle-sidecar-conflict"
    _write_sidecar_source_fixture(workdir)
    conflict = workdir / "localized_figures_assets" / "fixture-sidecar-paper" / "figure-01-method-figure-shows-the-pipeline.png"
    conflict.parent.mkdir(parents=True, exist_ok=True)
    conflict.write_bytes(b"do-not-overwrite")

    localized = raw_fast_evidence_bundle.localize_source_figures(root, workdir, "fixture-sidecar-paper")

    assert localized["ok"] is True
    assert conflict.read_bytes() == b"do-not-overwrite"
    assert localized["entries"]
    assert localized["entries"][0]["dest_rel"] != conflict.relative_to(workdir).as_posix()
    assert localized["entries"][0]["dest_rel"].endswith("-02.png")
    assert (workdir / localized["entries"][0]["dest_rel"]).exists()
    assert not (root / "raw" / "images").exists()


def test_raw_fast_evidence_bundle_localize_figures_resolves_source_root_relative_paths(tmp_path: Path) -> None:
    import raw_fast_evidence_bundle

    root = sample_wiki(tmp_path)
    workdir = tmp_path / "bundle-source-root-fig"
    _write_tiny_png(workdir / "source" / "fig" / "method.png")
    write(
        workdir / "source" / "tex" / "main.tex",
        r"""
\begin{figure}
\includegraphics{fig/method.png}
\caption{Source root method figure}
\label{fig:source-root}
\end{figure}
""".strip(),
    )

    localized = raw_fast_evidence_bundle.localize_source_figures(root, workdir, "source-root-paper")

    assert localized["ok"] is True
    assert len(localized["entries"]) == 1
    assert localized["entries"][0]["label"] == "fig:source-root"
    assert localized["entries"][0]["source_rel"] == "fig/method.png"
    assert localized["refused"] == []
    assert localized["entries"][0]["dest_rel"].startswith("localized_figures_assets/source-root-paper/")
    assert (workdir / localized["entries"][0]["dest_rel"]).exists()
    assert not (root / "raw" / "images").exists()


def test_raw_fast_evidence_bundle_localize_figures_renders_pdf_source_figures_to_png(tmp_path: Path) -> None:
    import raw_fast_evidence_bundle

    root = sample_wiki(tmp_path)
    workdir = tmp_path / "bundle-pdf-fig"
    _write_tiny_pdf_figure(workdir / "source" / "fig" / "frontier.pdf")
    write(
        workdir / "source" / "tex" / "main.tex",
        r"""
\begin{figure}
\includegraphics{fig/frontier.pdf}
\caption{Frontier curve as PDF}
\label{fig:frontier-pdf}
\end{figure}
""".strip(),
    )

    localized = raw_fast_evidence_bundle.localize_source_figures(root, workdir, "pdf-figure-paper")

    assert localized["ok"] is True
    assert len(localized["entries"]) == 1
    entry = localized["entries"][0]
    assert entry["label"] == "fig:frontier-pdf"
    assert entry["source_rel"] == "fig/frontier.pdf"
    assert entry["dest_rel"].endswith(".png")
    assert entry["localization_method"] == "pdf_render_first_page"
    assert entry["source_sha256"]
    assert entry["sha256"] != entry["source_sha256"]
    assert entry["width"] > 0 and entry["height"] > 0
    assert entry["dest_rel"].startswith("localized_figures_assets/pdf-figure-paper/")
    assert (workdir / entry["dest_rel"]).exists()
    assert not (root / "raw" / "images").exists()


def test_raw_fast_evidence_bundle_paper_digest_prefers_arxiv_api_title_over_image_tex_title(tmp_path: Path) -> None:
    import raw_fast_evidence_bundle

    workdir = tmp_path / "bundle-api-title"
    write(
        workdir / "api.xml",
        """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns='http://www.w3.org/2005/Atom'>
  <entry><title>API Grounded Paper Title</title></entry>
</feed>
""".strip(),
    )
    write(workdir / "source" / "main.tex", r"\title{<!-- image -->}\begin{document}\begin{abstract}API title should win.\end{abstract}\end{document}")

    digest = raw_fast_evidence_bundle.build_paper_digest(workdir, "<!-- image -->", "https://arxiv.org/abs/2600.00000")

    assert digest["metadata_card"]["title"] == "API Grounded Paper Title"


def test_raw_fast_evidence_bundle_resource_boundary_draft_distinguishes_absence_failure_and_unrelated_hits() -> None:
    import raw_fast_evidence_bundle

    resource_probe = {
        "ok": True,
        "probes": [
            {
                "ok": True,
                "type": "github_repo",
                "repo": "example/sidecar-paper",
                "html_url": "https://github.com/example/sidecar-paper",
                "private": False,
                "fork": False,
                "archived": False,
                "license": "MIT",
                "default_branch": "main",
                "evidence": {"root_files": ["README.md", "src"], "readme_excerpt": "Fixture repo README", "commit": "abc123"},
            },
            {"ok": True, "type": "hf_models", "query": "Fixture Sidecar Paper", "count": 0, "items": []},
            {"ok": False, "type": "hf_datasets", "query": "Fixture Sidecar Paper", "error": "TimeoutExpired", "message": "timed out"},
            {"ok": True, "type": "hf_spaces", "query": "q0", "count": 1, "items": [{"id": "unrelated/q0-demo", "likes": 1}]},
        ],
    }

    draft = raw_fast_evidence_bundle.summarize_resource_boundary(resource_probe, metadata={"title": "Fixture Sidecar Paper"})
    markdown = raw_fast_evidence_bundle.render_resource_boundary_markdown(draft)

    assert draft["github"][0]["status"] == "verified"
    assert draft["github"][0]["license"] == "MIT"
    assert draft["hf"]["models"]["status"] == "verified_absent"
    assert draft["hf"]["datasets"]["status"] == "probe_failed"
    assert draft["hf"]["spaces"]["status"] == "candidates_unverified"
    assert draft["hf"]["spaces"]["unrelated_candidates"][0]["id"] == "unrelated/q0-demo"
    assert "probe_failed" in markdown
    assert "verified_absent" in markdown


def test_raw_fast_evidence_bundle_localize_figures_refuses_unsafe_sources(tmp_path: Path) -> None:
    import raw_fast_evidence_bundle

    root = sample_wiki(tmp_path)
    workdir = tmp_path / "bundle-localize"
    _write_tiny_png(workdir / "source" / "figs" / "method.png")
    write(workdir / "source" / "figs" / "secret.txt", "not an image")
    write(
        workdir / "source" / "main.tex",
        r"""
\begin{figure}\includegraphics{figs/method.png}\caption{Safe figure}\label{fig:safe}\end{figure}
\begin{figure}\includegraphics{../outside.png}\caption{Traversal figure}\label{fig:outside}\end{figure}
\begin{figure}\includegraphics{figs/secret.txt}\caption{Text file}\label{fig:text}\end{figure}
""".strip(),
    )

    localized = raw_fast_evidence_bundle.localize_source_figures(root, workdir, "safe-slug")

    assert localized["ok"] is True
    assert len(localized["entries"]) == 1
    assert localized["entries"][0]["label"] == "fig:safe"
    refusal_reasons = {item["reason"] for item in localized["refused"]}
    assert "path_traversal" in refusal_reasons
    assert "unsupported_extension" in refusal_reasons
    assert localized["entries"][0]["dest_rel"].startswith("localized_figures_assets/safe-slug/")
    assert (workdir / localized["entries"][0]["dest_rel"]).exists()
    assert not (root / "raw" / "images").exists()


def _structured_raw_fast_note(title: str, source: str) -> str:
    return f"""---
title: \"{title}\"
source: \"{source}\"
capture_route: \"test synthetic route\"
captured: \"2026-06-06 07:00 CST (+0800)\"
---

## 一句话总结

Synthetic take.

## 论文摘要（中文）

Synthetic abstract.

## Motivation

Synthetic motivation.

## Methodology

Formula evidence is integrated here: Eq. (1) defines $loss = x + y$ and the symbols are explained in the method narrative.

## 关键实验结果 / 作者结论

Figure 1 and Table 1 are integrated beside the result claim and record the observed trend.

## 对未来研究的启发

Future work can reuse the verification harness.

## 可能的局限

The tiny fixture is a synthetic limitation, not a real paper.

## 可继续追问的问题

Which wrapper gate catches failed verification before mark-pending?
"""


def test_raw_fast_verifier_rejects_resource_status_and_extra_frontmatter_metadata(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    raw_rel = "raw/clip/2601/26010107_Bloated-Meta-Paper.md"
    note = _structured_raw_fast_note("Bloated Meta Paper", "https://example.test/bloated-meta.pdf").replace(
        "capture_route: \"test synthetic route\"\n",
        "capture_route: \"test synthetic route\"\ntags: [benchmark, memory]\ntopic_hints: [\"compact metadata\", \"graph routing\"]\nresource_status: \"legacy resource status should stay outside raw notes\"\nsource_pdf: \"https://example.test/bloated-meta.pdf\"\nauthors: [\"A. Author\"]\narxiv_version: \"v1\"\n",
    )
    write(root / raw_rel, note)

    result = subprocess.run(
        [
            sys.executable,
            str(RAW_FAST_VERIFIER),
            "--wiki",
            str(root),
            "--raw-file",
            raw_rel,
            "--structured-paper",
        ],
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode != 0
    assert payload["frontmatter_fields_extra"] == ["resource_status", "source_pdf", "authors", "arxiv_version"]
    assert "frontmatter_fields_extra" in payload["raw_fast_blockers"]


def test_raw_fast_evidence_bundle_candidate_frontmatter_stays_compact() -> None:
    import raw_fast_evidence_bundle

    fm = raw_fast_evidence_bundle.build_frontmatter(
        "Compact Candidate Paper",
        "https://arxiv.org/abs/2601.0101",
        "arxiv",
    )

    assert set(fm) <= {
        "title",
        "source",
        "created",
        "updated",
        "type",
        "domain",
        "tags",
        "topic_hints",
        "capture_route",
        "captured",
    }
    assert "resource_status" not in fm
    assert "source_id" not in fm

    skeleton = raw_fast_evidence_bundle.build_note_skeleton(fm)
    assert "## 资源与复现状态" not in skeleton
    assert "## Evidence trail" not in skeleton


def test_raw_fast_verifier_rejects_remote_markdown_images(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    raw_rel = "raw/clip/2601/26010108_Remote-Image-Paper.md"
    write(root / raw_rel, _structured_raw_fast_note("Remote Image Paper", "https://example.test/remote-image.pdf") + "\n![remote](https://example.test/figure.png)\n")

    result = subprocess.run(
        [
            sys.executable,
            str(RAW_FAST_VERIFIER),
            "--wiki",
            str(root),
            "--raw-file",
            raw_rel,
            "--structured-paper",
        ],
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode != 0
    assert payload["remote_markdown_images"] == 1
    assert "remote_markdown_images" in payload["raw_fast_blockers"]


def test_raw_fast_closeout_source_has_no_retired_helper_aliases() -> None:
    text = (SCRIPTS / "raw_fast_closeout.py").read_text(encoding="utf-8")
    old_backend = "light" + "rag"

    for name in [
        f"compact_{old_backend}_status",
        f"run_{old_backend}_status",
        f"run_{old_backend}_refresh_if_needed",
        f"synthesize_blocked_{old_backend}_status",
    ]:
        assert name not in text
    assert "wiki/wikigraph follow-through" not in text


def test_raw_fast_closeout_marks_pending_after_verifier_and_cleans_tmp(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    raw_rel = "raw/clip/2601/26010109_Tiny-Wrapper-Paper.md"
    title = "Tiny Wrapper Paper"
    source = "https://example.test/tiny-wrapper-paper.pdf"
    write(root / raw_rel, _structured_raw_fast_note(title, source))
    fetch_tmp = tmp_path / "fetch-tmp"
    write(fetch_tmp / "scratch.txt", "temporary evidence")
    write(
        fetch_tmp / "evidence_bundle.json",
        json.dumps(
            {
                "ok": True,
                "kind": "direct-pdf",
                "source_url": source,
                "title_guess": title,
                "warnings": [],
                "files": {"pdf": "paper.pdf", "paper_digest": "paper_digest.json", "localized_figures": "localized_figures.json"},
                "timings": {"total_seconds": 12.5, "steps": {"fetch_pdf": {"elapsed_seconds": 1.25}}},
            }
        ),
    )
    script = SCRIPTS / "raw_fast_closeout.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--workdir",
            str(SCRIPTS.parent),
            "--raw-file",
            raw_rel,
            "--title",
            title,
            "--source-id",
            source,
            "--pattern",
            title,
            "--pattern",
            source,
            "--topic-hint",
            "wrapper-test",
            "--resource-status-summary",
            "synthetic resources checked",
            "--tmp",
            str(fetch_tmp),
            "--auto-integrate",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["raw_fast_ok"] is True
    assert payload["pre_verify"]["duplicate_strict_ok"] is True
    assert payload["control_scan"]["control_count"] == 0
    assert payload["marked"]["raw_path"] == raw_rel
    assert payload["wiki_integration"]["pending_count"] == 1
    assert payload["wiki_integration"]["should_integrate"] is False
    assert payload["native_refresh_status"]["blocked_by_pending_wiki_integration"] is True
    old_backend = "light" + "rag"
    assert old_backend not in payload
    assert f"{old_backend}_refresh" not in payload
    assert payload["timings"]["total_seconds"] >= 0
    _assert_timing_step(payload, "pre_verify")
    _assert_timing_step(payload, "control_scan")
    _assert_timing_step(payload, "evidence_report_capture")
    _assert_timing_step(payload, "mark_pending")
    _assert_timing_step(payload, "final_verify")
    _assert_timing_step(payload, "native_refresh_status")
    retired_backend = "light" + "rag"
    assert f"{retired_backend}_status" not in payload["timings"]["steps"]
    assert payload["evidence_reports"]["count"] == 1
    evidence_report_path = Path(payload["evidence_reports"]["summaries"][0]["report_path"])
    assert evidence_report_path.exists()
    evidence_report = json.loads(evidence_report_path.read_text(encoding="utf-8"))
    assert evidence_report["timings"]["steps"]["fetch_pdf"]["elapsed_seconds"] == 1.25
    assert payload["final_verify"]["tmp_absent"][str(fetch_tmp)] is True
    assert not fetch_tmp.exists()
    ledger = load_pending_wiki_integration_ledger(state)
    assert ledger["pending"][0]["raw_path"] == raw_rel
    assert "resources" not in ledger["pending"][0]["required_sections"]


def test_raw_fast_closeout_native_status_uses_batch_native_refresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = types.SimpleNamespace(
        root=tmp_path / "wiki",
        state_dir=tmp_path / "work" / "wikigraph" / "state",
        workdir=SCRIPTS.parent,
        timeout=17,
    )
    calls: list[dict[str, object]] = []

    def fake_run_json(command, *, cwd, timeout):
        calls.append({"command": command, "cwd": cwd, "timeout": timeout})
        return {"returncode": 0, "json": {"pending_count": 1, "should_refresh": True}}

    monkeypatch.setattr(raw_fast_closeout, "run_json", fake_run_json)

    status = raw_fast_closeout.run_native_refresh_status(args)

    command = calls[0]["command"]
    assert isinstance(command, list)
    assert str(command[1]).endswith("scripts/batch_native_refresh.py")
    assert command[2] == "status"
    assert "--workdir" in command
    assert "--no-migrate-legacy" not in command
    assert status["pending_count"] == 1
    assert status["command_returncode"] == 0


def test_raw_fast_closeout_native_status_has_no_legacy_migration_knob(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = types.SimpleNamespace(
        root=tmp_path / "wiki",
        state_dir=tmp_path / "work" / "wikigraph" / "state",
        workdir=SCRIPTS.parent,
        timeout=17,
    )
    calls: list[dict[str, object]] = []

    def fake_run_json(command, *, cwd, timeout):
        calls.append({"command": command, "cwd": cwd, "timeout": timeout})
        return {"returncode": 0, "json": {"pending_count": 1, "should_refresh": True}}

    monkeypatch.setattr(raw_fast_closeout, "run_json", fake_run_json)

    assert "migrate_legacy" not in inspect.signature(raw_fast_closeout.run_native_refresh_status).parameters
    status = raw_fast_closeout.run_native_refresh_status(args)

    command = calls[0]["command"]
    assert isinstance(command, list)
    assert "--no-migrate-legacy" not in command
    assert status["pending_count"] == 1
    assert status["command_returncode"] == 0


def test_raw_fast_closeout_native_refresh_runs_prepare_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = types.SimpleNamespace(
        root=tmp_path / "wiki",
        state_dir=tmp_path / "work" / "wikigraph" / "state",
        workdir=SCRIPTS.parent,
        refresh_timeout=23,
    )
    status = {"command_returncode": 0, "pending_count": 1, "should_refresh": True}
    calls: list[dict[str, object]] = []

    def fake_run_json(command, *, cwd, timeout):
        calls.append({"command": command, "cwd": cwd, "timeout": timeout})
        return {
            "returncode": 0,
            "json": {
                "prepared_only": True,
                "skipped": False,
                "status_before": status,
                "build": {"ok": True},
            },
        }

    monkeypatch.setattr(raw_fast_closeout, "run_json", fake_run_json)

    result = raw_fast_closeout.run_native_refresh_if_needed(args, status)

    command = calls[0]["command"]
    assert str(command[1]).endswith("scripts/batch_native_refresh.py")
    assert command[2:4] == ["refresh", "--prepare-only"]
    assert result["ran"] is True
    assert result["prepared_only"] is True
    assert result["status"]["pending_count"] == 1


def test_raw_fast_closeout_does_not_mark_pending_when_verifier_fails(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    raw_rel = "raw/clip/2601/26010110_Bad-Wrapper-Paper.md"
    write(root / raw_rel, "---\ntitle: Bad\nsource: https://example.test/bad.pdf\n---\n\n## Methodology\n\nTODO\n")
    script = SCRIPTS / "raw_fast_closeout.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--workdir",
            str(SCRIPTS.parent),
            "--raw-file",
            raw_rel,
            "--title",
            "Bad Wrapper Paper",
            "--source-id",
            "https://example.test/bad.pdf",
            "--pattern",
            "https://example.test/bad.pdf",
        ],
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode != 0
    assert payload["stage"] == "pre_verify"
    assert payload["raw_fast_ok"] is False
    _assert_timing_step(payload, "pre_verify")
    assert "mark_pending" not in payload["timings"]["steps"]
    assert load_pending_wiki_integration_ledger(state)["pending"] == []


def test_raw_fast_evidence_bundle_refuses_workdir_inside_wiki_root(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    source_pdf = tmp_path / "source.pdf"
    _write_tiny_pdf(source_pdf)
    script = SCRIPTS / "raw_fast_evidence_bundle.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--url",
            source_pdf.as_uri(),
            "--kind",
            "direct-pdf",
            "--root",
            str(root),
            "--workdir",
            str(root / "raw" / "clip" / "bundle"),
            "--probe",
            "none",
        ],
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode != 0
    assert payload["stage"] == "preflight"
    assert payload["error"] == "workdir_inside_wiki_root"
    assert not (root / "raw" / "clip" / "bundle" / "evidence_bundle.json").exists()


def test_raw_fast_evidence_bundle_arxiv_doi_probe_schema_is_explicit() -> None:
    import raw_fast_evidence_bundle

    payload = raw_fast_evidence_bundle.build_resource_probe(
        "See arXiv:2604.08999 and DOI 10.1234/example.paper for details.",
        {"links": []},
        "https://example.test/paper.pdf",
        ["arxiv", "doi"],
        timeout=1,
    )
    probes = {(item["type"], item.get("id") or item.get("doi")): item for item in payload["probes"]}

    assert ("arxiv", "2604.08999") in probes
    assert probes[("arxiv", "2604.08999")]["ok"] is True
    assert probes[("arxiv", "2604.08999")]["status"] == "detected"
    assert probes[("arxiv", "2604.08999")]["evidence"]
    assert ("doi", "10.1234/example.paper") in probes
    assert probes[("doi", "10.1234/example.paper")]["status"] == "detected"


def test_raw_fast_evidence_bundle_github_probe_collects_root_files_and_readme(monkeypatch: pytest.MonkeyPatch) -> None:
    import base64
    import raw_fast_evidence_bundle

    readme_content = base64.b64encode(
        b"Fixture README sk-" + b"a" * 24 + b" details\nBenchmark artifacts are released for reproduction."
    ).decode("ascii")
    pyproject_content = base64.b64encode(
        b"""
[project]
name = "sidecar-paper"
dependencies = ["numpy"]

[project.scripts]
sidecar = "sidecar.cli:main"
""".strip()
    ).decode("ascii")

    def fake_fetch_text(url: str, timeout: int) -> dict:
        if url == "https://api.github.com/repos/example/sidecar-paper":
            return {"ok": True, "status": 200, "text": json.dumps({"html_url": "https://github.com/example/sidecar-paper", "private": False, "fork": False, "archived": False, "disabled": False, "default_branch": "main", "license": {"spdx_id": "MIT"}, "description": "fixture", "stargazers_count": 7, "full_name": "example/sidecar-paper", "pushed_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-02T00:00:00Z"})}
        if url == "https://api.github.com/repos/example/sidecar-paper/contents?ref=main":
            return {"ok": True, "status": 200, "text": json.dumps([{"name": "README.md", "type": "file"}, {"name": "pyproject.toml", "type": "file"}, {"name": "src", "type": "dir"}])}
        if url == "https://api.github.com/repos/example/sidecar-paper/readme?ref=main":
            return {"ok": True, "status": 200, "text": json.dumps({"encoding": "base64", "content": readme_content})}
        if url == "https://api.github.com/repos/example/sidecar-paper/branches/main":
            return {"ok": True, "status": 200, "text": json.dumps({"commit": {"sha": "abc123def456"}})}
        if url == "https://api.github.com/repos/example/sidecar-paper/contents/pyproject.toml?ref=main":
            return {"ok": True, "status": 200, "text": json.dumps({"encoding": "base64", "content": pyproject_content})}
        raise AssertionError(url)

    monkeypatch.setattr(raw_fast_evidence_bundle, "fetch_text", fake_fetch_text)

    payload = raw_fast_evidence_bundle.probe_github_repo("example/sidecar-paper", timeout=1)

    assert payload["ok"] is True
    assert payload["evidence"]["root_files"] == ["README.md", "pyproject.toml", "src"]
    assert payload["evidence"]["readme_excerpt"].startswith("Fixture README [REDACTED] details")
    assert "sk-" not in payload["evidence"]["readme_excerpt"]
    assert payload["evidence"]["commit"] == "abc123def456"
    assert payload["evidence"]["pyproject"]["project_name"] == "sidecar-paper"
    assert payload["evidence"]["pyproject"]["scripts"] == ["sidecar"]
    assert payload["evidence"]["readme_resource_mentions"] == ["Benchmark artifacts are released for reproduction."]
    assert payload["license"] == "MIT"

    draft = raw_fast_evidence_bundle.summarize_resource_boundary({"ok": True, "probes": [payload]}, metadata={"title": "Sidecar Paper"})
    markdown = raw_fast_evidence_bundle.render_resource_boundary_markdown(draft)
    assert draft["github"][0]["commit"] == "abc123def456"
    assert draft["github"][0]["pyproject"]["project_name"] == "sidecar-paper"
    assert draft["github"][0]["readme_resource_mentions"] == ["Benchmark artifacts are released for reproduction."]
    assert "commit=abc123def456" in markdown
    assert "pyproject=sidecar-paper" in markdown


def test_raw_fast_closeout_refuses_non_tmp_cleanup_before_marking(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    raw_rel = "raw/clip/2601/26010111_Unsafe-Cleanup-Paper.md"
    title = "Unsafe Cleanup Paper"
    source = "https://example.test/unsafe-cleanup-paper.pdf"
    write(root / raw_rel, _structured_raw_fast_note(title, source))
    unsafe_tmp = root / "raw" / "clip" / "unsafe-bundle"
    write(unsafe_tmp / "scratch.txt", "must not be deleted")
    script = SCRIPTS / "raw_fast_closeout.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--workdir",
            str(SCRIPTS.parent),
            "--raw-file",
            raw_rel,
            "--title",
            title,
            "--source-id",
            source,
            "--pattern",
            source,
            "--tmp",
            str(unsafe_tmp),
            "--auto-integrate",
        ],
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode != 0
    assert payload["stage"] == "cleanup_preflight"
    assert payload["cleanup_preflight"][0]["ok"] is False
    assert unsafe_tmp.exists()
    assert load_pending_wiki_integration_ledger(state)["pending"] == []


def test_raw_fast_closeout_final_verify_only_waives_post_integration_non_raw_hits() -> None:
    import raw_fast_closeout

    pre = {"raw_fast_ok": True, "non_raw_wiki_hits": [], "raw_fast_blockers": []}
    final = {"raw_fast_ok": False, "non_raw_wiki_hits": ["concepts/after.md"], "raw_fast_blockers": ["non_raw_wiki_hits"]}
    assert raw_fast_closeout.final_verify_acceptable(pre, final) is True

    final_with_secret = dict(final, raw_fast_blockers=["non_raw_wiki_hits", "strict_secret_hits"])
    assert raw_fast_closeout.final_verify_acceptable(pre, final_with_secret) is False

    final_with_tmp_left = dict(final, tmp_absent={"/tmp/raw-fast": False})
    assert raw_fast_closeout.final_verify_acceptable(pre, final_with_tmp_left) is False


def test_raw_fast_closeout_fast_final_verify_records_tmp_absence(tmp_path: Path) -> None:
    state = tmp_path / "state"
    args = types.SimpleNamespace(state_dir=state, raw_file="raw/clip/2601/26010112_Fast-Final.md")
    tmp_bundle = tmp_path / "already-cleaned"
    pre = {
        "command_returncode": 0,
        "raw_fast_ok": True,
        "note_exists": True,
        "nonzero_size": True,
        "has_frontmatter": True,
        "frontmatter_fields_missing": [],
        "structured_sections_missing": [],
        "structured_evidence_sections_insufficient": [],
        "deprecated_standalone_evidence_sections": [],
        "structured_heading_order_ok": True,
        "duplicate_strict_ok": True,
        "non_raw_wiki_hits": [],
        "remote_markdown_images": 0,
        "data_uri_images": 0,
        "missing_local_images": 0,
        "strict_secret_hits": 0,
    }

    report = raw_fast_closeout.fast_final_verify_from_pre(args, pre, [tmp_bundle])

    assert report["raw_fast_ok"] is True
    assert report["fast_final_verify"] is True
    assert report["tmp_absent"] == {str(tmp_bundle): True}
    assert Path(report["report_path"]).exists()


def test_raw_fast_closeout_compact_log_entry_is_bounded() -> None:
    args = types.SimpleNamespace(
        title="Compact Log Paper",
        source_id="https://arxiv.org/abs/2601.0101",
        raw_file="raw/clip/2601/26010112_Compact-Log-Paper.md",
        resource_status_summary="official abs/pdf/source verified; claimed code unresolved",
    )
    output = {
        "raw_fast_ok": True,
        "final_verify": {"report_path": "/state/raw_fast_reports/compact_final_verify.json"},
        "wiki_integration": {"pending_count": 4, "actionable_pending_count": 4, "threshold": 10, "should_integrate": False, "next_required_action": "none"},
        "native_refresh_status": {"blocked_by_pending_wiki_integration": True, "graph_ready_pending_count": 0, "should_refresh": False},
    }

    entry = raw_fast_closeout.build_compact_log_entry(args, output)

    assert len(entry.splitlines()) <= 5
    assert "26010112_Compact-Log-Paper.md" in entry
    assert "raw_fast_ok=true" in entry
    assert "checksums" not in entry.lower()


def test_raw_fast_closeout_blocked_log_distinguishes_standalone_native_ledger(tmp_path: Path) -> None:
    args = types.SimpleNamespace(
        title="Blocked Native Ledger Paper",
        source_id="https://arxiv.org/abs/2601.0102",
        raw_file="raw/clip/2601/26010113_Blocked-Native-Ledger-Paper.md",
        resource_status_summary="official abs/pdf/source verified",
    )
    wiki_status = {
        "pending_count": 2,
        "actionable_pending_count": 2,
        "review_pending_count": 0,
        "blocking_pending_count": 2,
        "threshold": 10,
        "should_integrate": False,
        "next_required_action": "wiki_integration",
    }
    standalone_status = {
        "command_returncode": 0,
        "pending_count": 1,
        "should_refresh": True,
        "ledger_path": str(tmp_path / "state" / "pending_native_refresh.json"),
    }

    native_status = raw_fast_closeout.synthesize_blocked_native_refresh_status(
        args,
        wiki_status,
        standalone_status=standalone_status,
    )
    output = {
        "raw_fast_ok": True,
        "final_verify": {"report_path": "/state/raw_fast_reports/blocked_final_verify.json"},
        "wiki_integration": wiki_status,
        "native_refresh_status": raw_fast_closeout.compact_native_refresh_status(native_status),
    }

    entry = raw_fast_closeout.build_compact_log_entry(args, output)

    assert native_status["blocked_by_pending_wiki_integration"] is True
    assert native_status["graph_ready_pending_count"] == 0
    assert native_status["standalone_native_pending_count"] == 1
    assert native_status["standalone_native_should_refresh"] is True
    assert "graph-ready pending `0`, `should_refresh=false`" in entry
    assert "standalone native ledger pending `1`, `should_refresh=true`" in entry


def test_threshold_wikigraph_status_skips_prequery_freshness_scan(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    mark_pending_wiki_integration(state, root, raw_path="raw/clip/2601/26010113_Pending.md", title="Pending")

    status = pending_wikigraph_refresh_status(root, state, reason="threshold")

    assert status["blocked_by_pending_wiki_integration"] is True
    assert status["latest_wiki_markdown_mtime"] is None
    assert status["import_report"] == {}
