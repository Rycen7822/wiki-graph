import json
import sqlite3
import subprocess
import sys
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import batch_lightrag_refresh  # noqa: E402
import raw_fast_closeout  # noqa: E402

from wiki_lightrag_lib import (  # noqa: E402
    _section_rank_lists,
    _section_rank_lists_scalar,
    build_custom_kg_payload,
    build_section_similarity_edges,
    build_seed_edges,
    canonical_id_for,
    clear_lightrag_refresh_pending_after_success,
    collect_source_docs,
    ensure_state_dirs,
    extract_method_atoms,
    expand_lightrag_data_response_with_section_neighbors,
    extract_raw_sections,
    fallback_frontmatter_load,
    filter_lightrag_data_response_by_section_kind,
    generated_docs_from_state,
    init_manifest_db,
    DEFAULT_PENDING_LIGHTRAG_REFRESH_THRESHOLD,
    DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD,
    clear_pending_wiki_integration_after_success,
    load_lightrag_refresh_ledger,
    load_pending_wiki_integration_ledger,
    make_ingest_text,
    mark_lightrag_refresh_pending,
    mark_pending_wiki_integration,
    parse_frontmatter,
    pending_lightrag_refresh_status,
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
    workdir = tmp_path / "work" / "lightrag"
    state = workdir / "state"
    ensure_state_dirs(state)
    assert (state / "edge_docs").is_dir()
    assert not (root / ".llm-wiki").exists()
    db = init_manifest_db(state)
    assert db == state / "lightrag_sync.db"
    with sqlite3.connect(db) as conn:
        tables = {row[0] for row in conn.execute("select name from sqlite_master where type='table'")}
    assert {"docs", "sync_events", "query_events"} <= tables


def test_jsonl_read_streams_rows_in_order_and_skips_blank_lines(tmp_path: Path) -> None:
    from wiki_lightrag_lib import jsonl_read

    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n  {"b": 2}  \n', encoding="utf-8")

    assert jsonl_read(path) == [{"a": 1}, {"b": 2}]


def test_validate_wiki_default_does_not_write_report(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"

    report = validate_wiki(root, state, tmp_path / "work" / "lightrag")

    assert "report_path" not in report
    assert not state.exists()
    assert not list((state / "validation_reports").glob("*_validate.json"))


def test_validate_wiki_write_report_is_explicit(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"

    report = validate_wiki(root, state, tmp_path / "work" / "lightrag", write_report=True)

    report_path = Path(report["report_path"])
    assert report_path.exists()
    assert report_path.parent == state / "validation_reports"


def test_validation_split_module_reexports_existing_public_functions() -> None:
    import wiki_lightrag_lib
    import wiki_lightrag_validation

    assert wiki_lightrag_lib.validate_wiki is wiki_lightrag_validation.validate_wiki
    assert wiki_lightrag_lib.secret_hits is wiki_lightrag_validation.secret_hits


def test_false_changed_only_flags_are_removed_from_cli_help() -> None:
    for script_name in [
        "validate_wiki.py",
        "build_seed_edges.py",
        "extract_method_atoms.py",
        "extract_raw_sections.py",
        "lightrag_sync.py",
        "sync_virtual_docs.py",
    ]:
        result = subprocess.run([sys.executable, str(SCRIPTS / script_name), "--help"], check=True, text=True, capture_output=True)
        assert "--changed-only" not in result.stdout


def test_lightrag_runtime_env_helpers_share_env_and_redaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from lightrag_runtime_env import env_int, load_env_file, redact_summary

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

    def fake_query(server: str, api_key: str, query: str, mode: str, top_k: int = 20, chunk_top_k: int = 10) -> dict:
        return {"response": f"answer for {query}", "references": []}

    monkeypatch.setattr(build_evidence_pack, "query_lightrag", fake_query, raising=False)
    monkeypatch.setattr(wiki_search, "query_lightrag", fake_query)
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
    with sqlite3.connect(state / "lightrag_sync.db") as conn:
        rows = conn.execute("SELECT query, mode, evidence_pack_path FROM query_events").fetchall()
    assert rows == [("alias query", "mix", str(pack))]


def test_mark_lightrag_refresh_pending_creates_external_ledger_without_wiki_pollution(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"
    entry = mark_lightrag_refresh_pending(
        state,
        root,
        raw_path="raw/clip/2601/26010101_Foo-Paper.md",
        title="Foo Paper",
        event_type="new_raw_note",
        changed_surfaces=["raw", "compiled", "meta", "log"],
        expected_sections=["summary", "abstract", "motivation", "methodology", "future", "limitations", "questions"],
    )
    ledger = load_lightrag_refresh_ledger(state)
    assert entry["raw_path"] == "raw/clip/2601/26010101_Foo-Paper.md"
    assert ledger["threshold"] == DEFAULT_PENDING_LIGHTRAG_REFRESH_THRESHOLD == 10
    assert ledger["dirty"] is True
    assert len(ledger["pending"]) == 1
    assert (state / "pending_lightrag_refresh.json").exists()
    assert not (root / "pending_lightrag_refresh.json").exists()
    assert wiki_root_machine_pollution(root) == []
    threshold_status = pending_lightrag_refresh_status(root, state, reason="threshold")
    pre_query_status = pending_lightrag_refresh_status(root, state, reason="pre-query")
    assert threshold_status["should_refresh"] is False
    assert pre_query_status["should_refresh"] is True
    assert "pending_items_for_pre_query" in pre_query_status["reasons"]


def test_mark_pending_wiki_integration_tracks_raw_fast_queue_without_wiki_pollution(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"
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
    state = tmp_path / "work" / "lightrag" / "state"
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
    state = tmp_path / "work" / "lightrag" / "state"
    for idx in range(5):
        mark_pending_wiki_integration(state, root, raw_path=f"raw/clip/2601/260102{idx:02d}_Paper.md", title=f"Paper {idx}", threshold=5)

    status = pending_wiki_integration_status(root, state)

    assert status["pending_count"] == 5
    assert status["actionable_pending_count"] == 5
    assert status["threshold"] == 5
    assert status["should_integrate"] is True
    assert "pending_threshold_reached" in status["reasons"]


def test_terminal_wiki_integration_statuses_do_not_trigger_threshold_or_lightrag_block(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"
    for idx in range(10):
        mark_pending_wiki_integration(state, root, raw_path=f"raw/clip/2601/260103{idx:02d}_Duplicate.md", title=f"Duplicate {idx}", status="skipped_duplicate")

    wiki_status = pending_wiki_integration_status(root, state)
    graph_status = pending_lightrag_refresh_status(root, state, reason="pre-query")

    assert wiki_status["pending_count"] == 10
    assert wiki_status["actionable_pending_count"] == 0
    assert wiki_status["terminal_pending_count"] == 10
    assert wiki_status["should_integrate"] is False
    assert graph_status["blocked_by_pending_wiki_integration"] is False
    assert graph_status["raw_fast_pending_wiki_integration_count"] == 0


def test_review_wiki_integration_status_blocks_lightrag_with_manual_review_action(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"
    mark_pending_wiki_integration(state, root, raw_path="raw/clip/2601/26010400_Needs-Review.md", title="Needs Review", status="needs_review")

    wiki_status = pending_wiki_integration_status(root, state, reason="pre-query")
    graph_status = pending_lightrag_refresh_status(root, state, reason="pre-query")

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


def test_pending_lightrag_refresh_status_triggers_at_threshold(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"
    for idx in range(9):
        mark_lightrag_refresh_pending(state, root, raw_path=f"raw/clip/2601/260101{idx:02d}_Paper.md", title=f"Paper {idx}")
    below = pending_lightrag_refresh_status(root, state, reason="threshold")
    assert below["pending_count"] == 9
    assert below["graph_ready_pending_count"] == 9
    assert below["raw_fast_pending_wiki_integration_count"] == 0
    assert below["threshold"] == DEFAULT_PENDING_LIGHTRAG_REFRESH_THRESHOLD == 10
    assert below["should_refresh"] is False

    mark_lightrag_refresh_pending(state, root, raw_path="raw/clip/2601/26010109_Paper.md", title="Paper 9")
    status = pending_lightrag_refresh_status(root, state, reason="threshold")
    assert status["pending_count"] == 10
    assert status["graph_ready_pending_count"] == 10
    assert status["raw_fast_pending_wiki_integration_count"] == 0
    assert status["threshold"] == DEFAULT_PENDING_LIGHTRAG_REFRESH_THRESHOLD == 10
    assert status["should_refresh"] is True
    assert "pending_threshold_reached" in status["reasons"]


def test_pending_lightrag_refresh_status_uses_persisted_threshold(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"
    for idx in range(5):
        mark_lightrag_refresh_pending(state, root, raw_path=f"raw/clip/2601/260105{idx:02d}_Paper.md", title=f"Paper {idx}", threshold=5)

    status = pending_lightrag_refresh_status(root, state, reason="threshold")

    assert status["pending_count"] == 5
    assert status["graph_ready_pending_count"] == 5
    assert status["threshold"] == 5
    assert status["should_refresh"] is True
    assert "pending_threshold_reached" in status["reasons"]


def test_lightrag_status_surfaces_raw_fast_pending_as_upstream_blocker(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"
    mark_lightrag_refresh_pending(state, root, raw_path="raw/clip/2601/26010101_Foo-Paper.md", title="Foo Paper")
    mark_pending_wiki_integration(state, root, raw_path="raw/clip/2601/26010102_Raw-Fast-A.md", title="Raw Fast A")
    mark_pending_wiki_integration(state, root, raw_path="raw/clip/2601/26010103_Raw-Fast-B.md", title="Raw Fast B")

    status = pending_lightrag_refresh_status(root, state, reason="pre-query")

    assert status["pending_count"] == 1
    assert status["graph_ready_pending_count"] == 1
    assert status["raw_fast_pending_wiki_integration_count"] == 2
    assert status["blocked_by_pending_wiki_integration"] is True
    assert status["should_refresh"] is False
    assert status["next_required_action"] == "wiki_integration"
    assert "pending_wiki_integration_before_lightrag_refresh" in status["blocked_reasons"]
    assert status["upstream_wiki_integration"]["pending_count"] == 2


def test_clear_pending_wiki_integration_marks_integrated_items_for_lightrag_refresh(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"
    first = "raw/clip/2601/26010102_Raw-Fast-A.md"
    second = "raw/clip/2601/26010103_Raw-Fast-B.md"
    mark_pending_wiki_integration(state, root, raw_path=first, title="Raw Fast A", required_sections=["summary", "methodology"])
    mark_pending_wiki_integration(state, root, raw_path=second, title="Raw Fast B", required_sections=["summary"])

    cleared = clear_pending_wiki_integration_after_success(root, state, integrated_paths=[first], reason="threshold")

    assert cleared["cleared_count"] == 1
    assert cleared["remaining_pending_count"] == 1
    assert cleared["marked_lightrag_pending_count"] == 1
    wiki_ledger = load_pending_wiki_integration_ledger(state)
    assert [item["raw_path"] for item in wiki_ledger["pending"]] == [second]
    assert wiki_ledger["dirty"] is True
    graph_ledger = load_lightrag_refresh_ledger(state)
    assert len(graph_ledger["pending"]) == 1
    assert graph_ledger["pending"][0]["raw_path"] == first
    assert graph_ledger["pending"][0]["event_type"] == "batch-wiki-integration"
    assert graph_ledger["pending"][0]["expected_sections"] == ["summary", "methodology"]


def test_clear_pending_wiki_integration_without_integrated_paths_clears_all_actionable_items(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"
    first = "raw/clip/2601/26010102_Raw-Fast-A.md"
    second = "raw/clip/2601/26010103_Raw-Fast-B.md"
    mark_pending_wiki_integration(state, root, raw_path=first, title="Raw Fast A")
    mark_pending_wiki_integration(state, root, raw_path=second, title="Raw Fast B")

    cleared = clear_pending_wiki_integration_after_success(root, state, reason="threshold")

    assert cleared["cleared_count"] == 2
    assert cleared["remaining_pending_count"] == 0
    assert cleared["marked_lightrag_pending_count"] == 2
    assert load_pending_wiki_integration_ledger(state)["pending"] == []
    assert {item["raw_path"] for item in load_lightrag_refresh_ledger(state)["pending"]} == {first, second}


def test_mark_lightrag_refresh_pending_deduplicates_by_raw_path(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"
    raw_path = "raw/clip/2601/26010101_Foo-Paper.md"
    mark_lightrag_refresh_pending(state, root, raw_path=raw_path, title="Foo Paper")
    mark_lightrag_refresh_pending(state, root, raw_path=raw_path, title="Foo Paper Updated", event_type="resource_refresh")
    ledger = load_lightrag_refresh_ledger(state)
    assert len(ledger["pending"]) == 1
    assert ledger["pending"][0]["title"] == "Foo Paper Updated"
    assert ledger["pending"][0]["event_type"] == "resource_refresh"


def test_clear_lightrag_refresh_pending_after_success_records_import_summary(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"
    mark_lightrag_refresh_pending(state, root, raw_path="raw/clip/2601/26010101_Foo-Paper.md", title="Foo Paper")
    import_report = state / "custom_kg_import_report.json"
    ensure_state_dirs(state)
    write(
        import_report,
        json.dumps(
            {
                "finished_at": "2026-05-20 12:00:00",
                "payload": {"chunks": 10, "entities": 11, "relationships": 12, "raw_section_chunks": 4, "section_similarity_relationships": 2},
            },
            ensure_ascii=False,
        ),
    )
    cleared = clear_lightrag_refresh_pending_after_success(root, state, import_report_path=import_report, reason="threshold")
    assert cleared["cleared_count"] == 1
    ledger = load_lightrag_refresh_ledger(state)
    assert ledger["pending"] == []
    assert ledger["dirty"] is False
    assert ledger["last_successful_raw_count"] == 1
    assert ledger["last_successful_refresh_at"] == "2026-05-20 12:00:00"
    assert ledger["last_successful_import_payload"]["relationships"] == 12


def test_batch_lightrag_refresh_cli_status_and_dry_run_are_non_mutating(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"
    script = SCRIPTS / "batch_lightrag_refresh.py"
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
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    mark_payload = json.loads(mark.stdout)
    assert mark_payload["pending_count"] == 1
    assert mark_payload["threshold"] == DEFAULT_PENDING_LIGHTRAG_REFRESH_THRESHOLD == 10
    dry = subprocess.run(
        [sys.executable, str(script), "refresh", "--root", str(root), "--state-dir", str(state), "--workdir", str(tmp_path / "work" / "lightrag"), "--reason", "pre-query", "--dry-run"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(dry.stdout)
    assert payload["dry_run"] is True
    assert payload["would_run"] is True
    assert any("validate_wiki.py" in " ".join(cmd) for cmd in payload["commands"])
    assert load_lightrag_refresh_ledger(state)["pending"]


def test_batch_lightrag_refresh_cli_exit_code_signals_upstream_wiki_integration(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"
    script = SCRIPTS / "batch_lightrag_refresh.py"
    mark_pending_wiki_integration(state, root, raw_path="raw/clip/2601/26010102_Raw-Fast-A.md", title="Raw Fast A")

    result = subprocess.run(
        [sys.executable, str(script), "should-refresh", "--root", str(root), "--state-dir", str(state), "--reason", "pre-query", "--exit-code"],
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 11
    assert payload["should_refresh"] is False
    assert payload["blocked_by_pending_wiki_integration"] is True
    assert payload["next_required_action"] == "wiki_integration"
    assert payload["upstream_wiki_integration"]["should_integrate"] is True
    assert "pending_items_for_wiki_integration" in payload["upstream_wiki_integration"]["reasons"]


def test_batch_lightrag_refresh_command_groups_preserve_order_without_positional_slice(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"
    workdir = tmp_path / "work" / "lightrag"

    groups = batch_lightrag_refresh.build_refresh_command_groups(root, state, workdir)
    flattened = batch_lightrag_refresh.build_refresh_commands(root, state, workdir)

    assert list(groups) == ["artifact", "full_import"]
    assert flattened == groups["artifact"] + groups["full_import"]
    assert all("systemctl" not in command[0] for command in groups["artifact"])
    assert groups["full_import"][0] == ["systemctl", "--user", "stop", batch_lightrag_refresh.SERVICE_NAME]


def test_batch_lightrag_refresh_restarts_service_after_import_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"
    workdir = tmp_path / "work" / "lightrag"
    artifact_log = state / "refresh_logs" / "artifact.log"
    import_log = state / "refresh_logs" / "import.log"
    commands = [["artifact", str(idx)] for idx in range(7)] + [
        ["systemctl", "--user", "stop", batch_lightrag_refresh.SERVICE_NAME],
        ["python-internal", "reset-rag-storage", str(workdir / "rag_storage"), str(workdir / "inputs")],
        [str(batch_lightrag_refresh.LIGHTRAG_PYTHON), str(workdir / "scripts" / "import_custom_kg.py")],
        ["systemctl", "--user", "start", batch_lightrag_refresh.SERVICE_NAME],
        ["python-internal", "health", "http://127.0.0.1:9621/health"],
    ]
    calls: list[list[str]] = []

    monkeypatch.setattr(batch_lightrag_refresh, "build_refresh_commands", lambda *_args: commands)
    monkeypatch.setattr(batch_lightrag_refresh, "reset_rag_storage", lambda *_args: calls.append(["python-internal", "reset-rag-storage"]))

    def fake_run_subprocess(command: list[str], *_args, **_kwargs) -> None:
        calls.append(command)
        if command and command[0] == str(batch_lightrag_refresh.LIGHTRAG_PYTHON):
            raise RuntimeError("import failed")

    monkeypatch.setattr(batch_lightrag_refresh, "run_subprocess", fake_run_subprocess)

    with pytest.raises(RuntimeError, match="import failed"):
        batch_lightrag_refresh.run_real_refresh(root, state, workdir, "threshold", artifact_log, import_log)

    assert ["systemctl", "--user", "stop", batch_lightrag_refresh.SERVICE_NAME] in calls
    assert ["systemctl", "--user", "start", batch_lightrag_refresh.SERVICE_NAME] in calls


def test_batch_wiki_integration_cli_status_mark_and_clear_are_external(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"
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


def test_batch_wiki_integration_auto_integrate_runs_configured_runner_at_threshold_and_requires_cleared_ledger(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"
    script = SCRIPTS / "batch_wiki_integration.py"
    for idx in range(10):
        mark_pending_wiki_integration(state, root, raw_path=f"raw/clip/2601/260106{idx:02d}_Paper.md", title=f"Paper {idx}", required_sections=["summary"])
    fake_runner = tmp_path / "fake_wiki_integrator.py"
    write(
        fake_runner,
        "import os, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
        "from wiki_lightrag_lib import clear_pending_wiki_integration_after_success\n"
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
    assert len(load_lightrag_refresh_ledger(state)["pending"]) == 10


def test_batch_wiki_integration_auto_integrate_records_failure_if_runner_leaves_ledger_pending(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"
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
    write(root / "lightrag_manifest.jsonl", "{}\n")
    polluted = {p.as_posix() for p in wiki_root_machine_pollution(root)}
    assert ".llm-wiki" in polluted
    assert "lightrag_manifest.jsonl" in polluted


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
    state = tmp_path / "work" / "lightrag" / "state"
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
    state = tmp_path / "work" / "lightrag" / "state"
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
    state = tmp_path / "work" / "lightrag" / "state"
    extract_raw_sections(root, state)
    docs = generated_docs_from_state(state, kind="raw_section")
    summary_docs = [doc for doc in docs if doc.canonical_id == "raw_section:26010105_Summary-Section:summary"]
    assert len(summary_docs) == 1
    assert "section_kind: summary" in summary_docs[0].text
    assert "section_title: 一句话总结" in summary_docs[0].text


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
    state = tmp_path / "work" / "lightrag" / "state"
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
    state = tmp_path / "work" / "lightrag" / "state"
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
    state = tmp_path / "work" / "lightrag" / "state"
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
    state = tmp_path / "work" / "lightrag" / "state"
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
    state = tmp_path / "work" / "lightrag" / "state"
    result = extract_raw_sections(root, state)
    files = list((state / "raw_section_docs").glob("*.md"))
    assert len(files) == result["raw_sections"]


def test_custom_kg_payload_includes_raw_section_chunks_and_relationships(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"
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


def test_custom_kg_manifest_resolves_sources_and_dedupes_relationships(tmp_path: Path) -> None:
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

    manifest = build_custom_kg_manifest(payload, lightrag_version="1.5.0", embedding_model="test-embed", embedding_dim=3)

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
    assert len(manifest["relationships"]) == 1
    rel = next(iter(manifest["relationships"].values()))
    assert rel["description"] == "new"
    assert rel["keywords"] == "NEW"
    assert rel["source_chunk_id"] == chunk_id
    assert rel["vdb_id"] == relation_vdb_id("topic:x", "doc:a")
    assert rel["record_type"] == "relationship"
    assert rel["record_id"] == rel["vdb_id"]
    assert rel["canonical_id"] == rel["chunk_key"]
    assert rel["vector_text_hash"] == stable_hash(rel["content"])


def test_custom_kg_manifest_matches_lightrag_sanitized_chunk_ids_and_basenames() -> None:
    from custom_kg_incremental import build_custom_kg_manifest, compute_mdhash_id, lightrag_sanitize_text

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

    manifest = build_custom_kg_manifest(payload, lightrag_version="1.5.0", embedding_model="test-embed", embedding_dim=3)
    sanitized = lightrag_sanitize_text(raw_content)
    chunk_id = compute_mdhash_id(sanitized, prefix="chunk-")

    assert sanitized == "A & B"
    assert set(manifest["chunks"]) == {chunk_id}
    assert manifest["chunks"][chunk_id]["content"] == sanitized
    assert manifest["chunks"][chunk_id]["file_path"] == "doc.md"
    assert manifest["entities"]["doc:sanitized"]["source_chunk_id"] == chunk_id
    assert manifest["entities"]["doc:sanitized"]["file_path"] == "doc.md"


def test_successful_manifest_stamps_metadata_without_mutating_desired_manifest() -> None:
    from custom_kg_incremental import build_custom_kg_manifest, successful_manifest

    payload = {
        "chunks": [{"content": "Doc A", "source_id": "doc:a", "file_path": "a.md"}],
        "entities": [{"entity_name": "doc:a", "entity_type": "DOC", "description": "Doc A", "source_id": "doc:a", "file_path": "a.md"}],
        "relationships": [],
    }
    desired = build_custom_kg_manifest(payload, lightrag_version="1.5.0", embedding_model="test", embedding_dim=3)
    original_metadata = dict(desired["metadata"])

    final = successful_manifest(desired, import_mode="incremental", previous_manifest=desired)

    assert desired["metadata"] == original_metadata
    assert final["metadata"]["last_successful_import_mode"] == "incremental"
    assert final["metadata"]["incremental_count_since_full"] == 1
    assert final["chunks"] == desired["chunks"]
    assert final["entities"] == desired["entities"]
    assert final["relationships"] == desired["relationships"]


def test_current_lightrag_version_uses_tool_python_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import custom_kg_incremental

    tool_python = tmp_path / "fake-lightrag-python"
    tool_python.write_text("#!/bin/sh\nprintf '9.9.9\\n'\n", encoding="utf-8")
    tool_python.chmod(0o755)

    def missing_distribution(_name: str) -> str:
        raise custom_kg_incremental.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(custom_kg_incremental.importlib.metadata, "version", missing_distribution)
    monkeypatch.setattr(custom_kg_incremental, "DEFAULT_LIGHTRAG_PYTHON", tool_python)

    assert custom_kg_incremental.current_lightrag_version() == "9.9.9"


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
        build_custom_kg_manifest(old_payload, lightrag_version="1.5.0", embedding_model="test", embedding_dim=3),
        build_custom_kg_manifest(new_payload, lightrag_version="1.5.0", embedding_model="test", embedding_dim=3),
    )

    assert diff["chunks"]["add"] == 1
    assert diff["chunks"]["delete"] == 1
    assert diff["entities"]["add"] == 1
    assert diff["entities"]["update"] == 1
    assert diff["relationships"]["delete"] == 1


def test_tracking_diff_identifies_entity_and_relationship_tracking_deltas() -> None:
    from custom_kg_incremental import diff_tracking

    old_manifest = {
        "entities": {
            "doc:keep": {"source_chunk_id": "chunk:keep"},
            "doc:update": {"source_chunk_id": "chunk:old"},
            "doc:delete": {"source_chunk_id": "chunk:delete"},
            "doc:unknown": {"source_chunk_id": "UNKNOWN"},
        },
        "relationships": {
            "doc:keep<SEP>topic:keep": {"source_chunk_id": "chunk:keep"},
            "doc:update<SEP>topic:update": {"source_chunk_id": "chunk:old"},
            "doc:delete<SEP>topic:delete": {"source_chunk_id": "chunk:delete"},
        },
    }
    new_manifest = {
        "entities": {
            "doc:keep": {"source_chunk_id": "chunk:keep"},
            "doc:update": {"source_chunk_id": "chunk:new"},
            "doc:add": {"source_chunk_id": "chunk:add"},
        },
        "relationships": {
            "doc:keep<SEP>topic:keep": {"source_chunk_id": "chunk:keep"},
            "doc:update<SEP>topic:update": {"source_chunk_id": "chunk:new"},
            "doc:add<SEP>topic:add": {"source_chunk_id": "chunk:add"},
        },
    }

    diff = diff_tracking(old_manifest, new_manifest)

    assert diff["entities"]["add_ids"] == ["doc:add"]
    assert diff["entities"]["update_ids"] == ["doc:update"]
    assert diff["entities"]["delete_ids"] == ["doc:delete"]
    assert diff["entities"]["upsert_records"] == {
        "doc:add": {"chunk_ids": ["chunk:add"], "count": 1},
        "doc:update": {"chunk_ids": ["chunk:new"], "count": 1},
    }
    assert diff["relationships"]["add_ids"] == ["doc:add<SEP>topic:add"]
    assert diff["relationships"]["update_ids"] == ["doc:update<SEP>topic:update"]
    assert diff["relationships"]["delete_ids"] == ["doc:delete<SEP>topic:delete"]


def test_custom_kg_diff_splits_metadata_only_relationship_and_entity_updates() -> None:
    from custom_kg_incremental import build_custom_kg_manifest, diff_custom_kg_manifests, relation_chunk_key

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
        build_custom_kg_manifest(old_payload, lightrag_version="1.5.0", embedding_model="test", embedding_dim=3),
        build_custom_kg_manifest(new_payload, lightrag_version="1.5.0", embedding_model="test", embedding_dim=3),
    )
    rel_key = relation_chunk_key("doc:a", "topic:x")

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

    base = build_custom_kg_manifest(payload, lightrag_version="1.5.0", embedding_model="embed-a", embedding_dim=3, embedding_params_version="v1")
    model_changed = build_custom_kg_manifest(payload, lightrag_version="1.5.0", embedding_model="embed-b", embedding_dim=3, embedding_params_version="v1")
    dim_changed = build_custom_kg_manifest(payload, lightrag_version="1.5.0", embedding_model="embed-a", embedding_dim=4, embedding_params_version="v1")
    params_changed = build_custom_kg_manifest(payload, lightrag_version="1.5.0", embedding_model="embed-a", embedding_dim=3, embedding_params_version="v2")

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
        lightrag_version="1.5.0",
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
        lightrag_version="1.5.0",
        embedding_model="embed-a",
        embedding_dim=3,
        embedding_params_version="v1",
    )

    assert next(iter(old_manifest["entities"].values()))["vector_hash"] == next(iter(new_manifest["entities"].values()))["vector_hash"]
    assert next(iter(old_manifest["relationships"].values()))["vector_hash"] == next(iter(new_manifest["relationships"].values()))["vector_hash"]


def test_custom_kg_diff_derives_split_hashes_for_legacy_manifest_records() -> None:
    from custom_kg_incremental import build_custom_kg_manifest, diff_custom_kg_manifests, relation_chunk_key

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
        lightrag_version="1.5.0",
        embedding_model="test",
        embedding_dim=3,
    )
    new_manifest = build_custom_kg_manifest(
        {
            "chunks": chunks,
            "entities": [{"entity_name": "topic:x", "entity_type": "TOPIC", "description": "same", "source_id": "doc:b", "file_path": "new.md"}],
            "relationships": [{"src_id": "doc:a", "tgt_id": "topic:x", "description": "same edge", "keywords": "RELATED", "source_id": "doc:b", "file_path": "new.md"}],
        },
        lightrag_version="1.5.0",
        embedding_model="test",
        embedding_dim=3,
    )
    rel_key = relation_chunk_key("doc:a", "topic:x")
    for collection in ("entities", "relationships"):
        for record in old_manifest[collection].values():
            record.pop("vector_hash", None)
            record.pop("metadata_hash", None)

    diff = diff_custom_kg_manifests(old_manifest, new_manifest)

    assert diff["entities"]["metadata_update_ids"] == ["topic:x"]
    assert diff["entities"]["vector_update_ids"] == []
    assert diff["relationships"]["metadata_update_ids"] == [rel_key]
    assert diff["relationships"]["vector_update_ids"] == []


def test_incremental_apply_patches_metadata_only_relationship_without_reembedding(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import asyncio
    import custom_kg_incremental
    import import_custom_kg

    class FakeVDB:
        def __init__(self, records: list[dict[str, object]] | None = None) -> None:
            self.storage = {"data": records or [], "matrix": ""}
            self.deleted: list[str] = []
            self.upserted: list[dict[str, dict[str, object]]] = []

        async def delete(self, ids: list[str]) -> None:
            self.deleted.extend(ids)
            self.storage["data"] = [record for record in self.storage["data"] if record.get("__id__") not in set(ids)]

        async def upsert(self, data: dict[str, dict[str, object]]) -> None:
            self.upserted.append(data)

        @property
        async def client_storage(self) -> dict[str, object]:
            return self.storage

    class FakeKV:
        def __init__(self) -> None:
            self.deleted: list[str] = []
            self.upserted: list[dict[str, object]] = []
            self.dropped = False

        async def delete(self, ids: list[str]) -> None:
            self.deleted.extend(ids)

        async def upsert(self, records: dict[str, object]) -> None:
            self.upserted.append(records)

        async def drop(self) -> None:
            self.dropped = True

    class FakeGraph:
        def __init__(self) -> None:
            self.removed_edges: list[tuple[str, str]] = []
            self.upserted_edges: list[tuple[str, str, dict[str, object]]] = []

        async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
            self.removed_edges.extend(edges)

        async def remove_nodes(self, _nodes: list[str]) -> None:
            pass

        async def upsert_nodes_batch(self, _nodes: list[tuple[str, dict[str, object]]]) -> None:
            pass

        async def upsert_edges_batch(self, edges: list[tuple[str, str, dict[str, object]]]) -> None:
            self.upserted_edges.extend(edges)

    chunks = [
        {"content": "Doc A", "source_id": "doc:a", "file_path": "a.md"},
        {"content": "Doc B", "source_id": "doc:b", "file_path": "b.md"},
    ]
    old_manifest = custom_kg_incremental.build_custom_kg_manifest(
        {
            "chunks": chunks,
            "entities": [{"entity_name": "topic:x", "entity_type": "TOPIC", "description": "same", "source_id": "doc:a", "file_path": "old.md"}],
            "relationships": [{"src_id": "doc:a", "tgt_id": "topic:x", "description": "same edge", "keywords": "RELATED", "source_id": "doc:a", "file_path": "old.md"}],
        },
        lightrag_version="1.5.0",
        embedding_model="test",
        embedding_dim=3,
    )
    desired_manifest = custom_kg_incremental.build_custom_kg_manifest(
        {
            "chunks": chunks,
            "entities": [{"entity_name": "topic:x", "entity_type": "TOPIC", "description": "same", "source_id": "doc:a", "file_path": "old.md"}],
            "relationships": [{"src_id": "doc:a", "tgt_id": "topic:x", "description": "same edge", "keywords": "RELATED", "source_id": "doc:b", "file_path": "new.md"}],
        },
        lightrag_version="1.5.0",
        embedding_model="test",
        embedding_dim=3,
    )
    old_rel = next(iter(old_manifest["relationships"].values()))
    desired_rel = next(iter(desired_manifest["relationships"].values()))
    rel_record = {"__id__": old_rel["vdb_id"], "vector": "PRESERVED_VECTOR", **custom_kg_incremental._relationship_vdb_data(old_rel)}

    fake_rag = types.SimpleNamespace(
        chunk_entity_relation_graph=FakeGraph(),
        relationships_vdb=FakeVDB([rel_record]),
        entities_vdb=FakeVDB(),
        chunks_vdb=FakeVDB(),
        text_chunks=FakeKV(),
        entity_chunks=FakeKV(),
        relation_chunks=FakeKV(),
        initialize_storages=lambda: None,
        _insert_done=lambda: None,
        finalize_storages=lambda: None,
    )

    async def initialize_storages() -> None:
        pass

    async def insert_done() -> None:
        pass

    async def finalize_storages() -> None:
        pass

    fake_rag.initialize_storages = initialize_storages
    fake_rag._insert_done = insert_done
    fake_rag.finalize_storages = finalize_storages
    monkeypatch.setattr(import_custom_kg, "build_rag", lambda *_args, **_kwargs: fake_rag)

    diff = asyncio.run(custom_kg_incremental.apply_patch_to_storage(tmp_path, old_manifest, desired_manifest, workdir=tmp_path, tracking_update_mode="delta"))

    assert diff["relationships"]["metadata_update_ids"] == [desired_rel["chunk_key"]]
    assert diff["relationships"]["vector_update_ids"] == []
    assert fake_rag.relationships_vdb.deleted == []
    assert fake_rag.relationships_vdb.upserted == []
    assert fake_rag.relation_chunks.deleted == []
    assert fake_rag.relation_chunks.upserted == [{desired_rel["chunk_key"]: {"chunk_ids": [desired_rel["source_chunk_id"]], "count": 1}}]
    assert fake_rag.chunk_entity_relation_graph.removed_edges == []
    assert fake_rag.chunk_entity_relation_graph.upserted_edges[-1][2]["source_id"] == desired_rel["source_chunk_id"]
    assert rel_record["vector"] == "PRESERVED_VECTOR"
    assert rel_record["source_id"] == desired_rel["source_chunk_id"]
    assert rel_record["file_path"] == "new.md"


def test_incremental_refresh_mode_requires_manifest_and_full_after_five_incrementals() -> None:
    from custom_kg_incremental import build_custom_kg_manifest, choose_refresh_mode

    payload = {
        "chunks": [{"content": "Doc A", "source_id": "doc:a", "file_path": "a.md"}],
        "entities": [{"entity_name": "doc:a", "entity_type": "DOC", "description": "Doc A", "source_id": "doc:a", "file_path": "a.md"}],
        "relationships": [],
    }
    desired = build_custom_kg_manifest(payload, lightrag_version="1.5.0", embedding_model="test", embedding_dim=3)

    no_manifest = choose_refresh_mode(None, desired, storage_audit_ok=True, full_rebuild_interval=5)
    assert no_manifest["selected_mode"] == "full_rebuild"
    assert "missing_manifest" in no_manifest["reasons"]

    previous = json.loads(json.dumps(desired))
    previous["metadata"]["incremental_count_since_full"] = 4
    fifth_incremental = choose_refresh_mode(previous, desired, storage_audit_ok=True, full_rebuild_interval=5)
    assert fifth_incremental["selected_mode"] == "incremental"
    assert fifth_incremental["next_incremental_count_since_full"] == 5

    previous["metadata"]["incremental_count_since_full"] = 5
    after_five = choose_refresh_mode(previous, desired, storage_audit_ok=True, full_rebuild_interval=5)
    assert after_five["selected_mode"] == "full_rebuild"
    assert "incremental_interval_reached" in after_five["reasons"]


def test_incremental_apply_report_includes_phase_timings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    import custom_kg_incremental

    payload = {
        "chunks": [{"content": "Doc A", "source_id": "doc:a", "file_path": "a.md"}],
        "entities": [{"entity_name": "doc:a", "entity_type": "DOC", "description": "Doc A", "source_id": "doc:a", "file_path": "a.md"}],
        "relationships": [],
    }
    previous = custom_kg_incremental.build_custom_kg_manifest(payload, lightrag_version="1.5.0", embedding_model="test", embedding_dim=3)
    desired = json.loads(json.dumps(previous))
    root = sample_wiki(tmp_path)
    workdir = tmp_path / "work" / "lightrag"
    state = workdir / "state"
    ensure_state_dirs(state)
    (workdir / "rag_storage").mkdir(parents=True)
    write(workdir / "rag_storage" / "placeholder.txt", "ok")
    custom_kg_incremental.write_manifest(state, previous)

    async def fake_apply_patch_to_storage(_shadow_storage, _old_manifest, _desired_manifest, *, workdir, tracking_update_mode="full"):
        return custom_kg_incremental.diff_custom_kg_manifests(_old_manifest, _desired_manifest)

    monkeypatch.setattr(custom_kg_incremental, "build_desired_manifest", lambda *_args, **_kwargs: (desired, {"chunks": 1, "entities": 1, "relationships": 0}))
    monkeypatch.setattr(custom_kg_incremental, "audit_custom_kg_storage", lambda *_args, **_kwargs: {"ok": True, "issues": [], "counts": {}})
    monkeypatch.setattr(custom_kg_incremental, "port_open", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(custom_kg_incremental, "apply_patch_to_storage", fake_apply_patch_to_storage)

    args = types.SimpleNamespace(
        root=root,
        state_dir=state,
        workdir=workdir,
        limit_docs=None,
        limit_edges=None,
        full_rebuild_interval=5,
        force_incremental=False,
        server_host="127.0.0.1",
        server_port=9621,
        allow_server_running=False,
        no_swap=True,
        delete_shadow_on_no_swap=True,
        write_manifest_without_swap=False,
    )

    report = asyncio.run(custom_kg_incremental.run_apply(args))

    assert report["swapped"] is False
    assert report["timings"]["build_desired_manifest_s"] >= 0
    assert report["timings"]["copy_live_to_shadow_s"] >= 0
    assert report["timings"]["apply_patch_to_shadow_s"] >= 0
    assert report["timings"]["audit_shadow_storage_s"] >= 0
    assert report["timings"]["total_s"] >= 0


def test_incremental_apply_prepare_swap_writes_audited_shadow_without_manifest_swap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    import custom_kg_incremental

    payload = {
        "chunks": [{"content": "Doc A", "source_id": "doc:a", "file_path": "a.md"}],
        "entities": [{"entity_name": "doc:a", "entity_type": "DOC", "description": "Doc A", "source_id": "doc:a", "file_path": "a.md"}],
        "relationships": [],
    }
    previous = custom_kg_incremental.build_custom_kg_manifest(payload, lightrag_version="1.5.0", embedding_model="test", embedding_dim=3)
    desired = json.loads(json.dumps(previous))
    first_chunk_id = next(iter(desired["chunks"]))
    desired["chunks"][first_chunk_id]["content"] = "Doc A updated"
    desired["chunks"][first_chunk_id]["record_hash"] = custom_kg_incremental.stable_hash(desired["chunks"][first_chunk_id])
    root = sample_wiki(tmp_path)
    workdir = tmp_path / "work" / "lightrag"
    state = workdir / "state"
    ensure_state_dirs(state)
    live_storage = workdir / "rag_storage"
    live_storage.mkdir(parents=True)
    write(live_storage / "live.txt", "live")
    custom_kg_incremental.write_manifest(state, previous)

    async def fake_apply_patch_to_storage(shadow_storage, _old_manifest, _desired_manifest, *, workdir, tracking_update_mode="full"):
        write(Path(shadow_storage) / "shadow.txt", "prepared")
        return custom_kg_incremental.diff_custom_kg_manifests(_old_manifest, _desired_manifest)

    monkeypatch.setattr(custom_kg_incremental, "build_desired_manifest", lambda *_args, **_kwargs: (desired, {"chunks": 1, "entities": 1, "relationships": 0}))
    monkeypatch.setattr(custom_kg_incremental, "audit_custom_kg_storage", lambda *_args, **_kwargs: {"ok": True, "issues": [], "counts": {}})
    monkeypatch.setattr(custom_kg_incremental, "port_open", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(custom_kg_incremental, "apply_patch_to_storage", fake_apply_patch_to_storage)

    args = types.SimpleNamespace(
        root=root,
        state_dir=state,
        workdir=workdir,
        limit_docs=None,
        limit_edges=None,
        full_rebuild_interval=5,
        force_incremental=False,
        server_host="127.0.0.1",
        server_port=9621,
        allow_server_running=False,
        no_swap=False,
        prepare_swap=True,
        delete_shadow_on_no_swap=False,
        write_manifest_without_swap=False,
        tracking_update_mode="full",
    )

    report = asyncio.run(custom_kg_incremental.run_apply(args))

    assert report["prepared_for_swap"] is True
    assert report["swapped"] is False
    assert report["manifest_path"] is None
    assert custom_kg_incremental.load_manifest(state) == previous
    prepared_report = custom_kg_incremental.prepared_swap_report_path(state)
    prepared_manifest = custom_kg_incremental.prepared_swap_manifest_path(state)
    assert prepared_report.exists()
    assert prepared_manifest.exists()
    prepared_payload = json.loads(prepared_report.read_text(encoding="utf-8"))
    assert prepared_payload["previous_manifest_hash"] == custom_kg_incremental.stable_hash(previous)
    assert prepared_payload["desired_manifest_hash"] == custom_kg_incremental.stable_hash(desired)
    assert Path(report["shadow_storage"]).exists()
    assert (Path(report["shadow_storage"]) / "shadow.txt").exists()
    assert (live_storage / "live.txt").exists()


def test_finalize_prepared_swap_refuses_stale_live_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import custom_kg_incremental

    payload = {
        "chunks": [{"content": "Doc A", "source_id": "doc:a", "file_path": "a.md"}],
        "entities": [{"entity_name": "doc:a", "entity_type": "DOC", "description": "Doc A", "source_id": "doc:a", "file_path": "a.md"}],
        "relationships": [],
    }
    previous = custom_kg_incremental.build_custom_kg_manifest(payload, lightrag_version="1.5.0", embedding_model="test", embedding_dim=3)
    stale = json.loads(json.dumps(previous))
    stale["metadata"]["changed"] = True
    desired = json.loads(json.dumps(previous))
    root = sample_wiki(tmp_path)
    workdir = tmp_path / "work" / "lightrag"
    state = workdir / "state"
    ensure_state_dirs(state)
    live_storage = workdir / "rag_storage"
    shadow_storage = workdir / "rag_storage.shadow.test"
    live_storage.mkdir(parents=True)
    shadow_storage.mkdir(parents=True)
    write(live_storage / "live.txt", "live")
    write(shadow_storage / "shadow.txt", "shadow")
    custom_kg_incremental.write_manifest(state, stale)
    desired_path = custom_kg_incremental.prepared_swap_manifest_path(state)
    desired_path.parent.mkdir(parents=True, exist_ok=True)
    desired_path.write_text(json.dumps(desired, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = custom_kg_incremental.prepared_swap_report_path(state)
    report_path.write_text(
        json.dumps(
            {
                "prepared_for_swap": True,
                "shadow_storage": str(shadow_storage),
                "backup_dir": str(state / "backups" / "backup"),
                "desired_manifest_path": str(desired_path),
                "previous_manifest_hash": custom_kg_incremental.stable_hash(previous),
                "desired_manifest_hash": custom_kg_incremental.stable_hash(desired),
                "payload": {"chunks": 1, "entities": 1, "relationships": 0},
                "diff": {},
                "plan": {},
                "pre_audit": {"ok": True, "issues": []},
                "timings": {},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(custom_kg_incremental, "port_open", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(custom_kg_incremental, "audit_custom_kg_storage", lambda *_args, **_kwargs: {"ok": True, "issues": [], "counts": {}})
    args = types.SimpleNamespace(root=root, state_dir=state, workdir=workdir, prepared_report=report_path, server_host="127.0.0.1", server_port=9621, allow_server_running=False)

    with pytest.raises(RuntimeError, match="live manifest changed"):
        custom_kg_incremental.run_finalize_prepared_swap(args)

    assert (live_storage / "live.txt").exists()
    assert (shadow_storage / "shadow.txt").exists()


def test_finalize_prepared_swap_swaps_shadow_and_writes_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import custom_kg_incremental

    payload = {
        "chunks": [{"content": "Doc A", "source_id": "doc:a", "file_path": "a.md"}],
        "entities": [{"entity_name": "doc:a", "entity_type": "DOC", "description": "Doc A", "source_id": "doc:a", "file_path": "a.md"}],
        "relationships": [],
    }
    previous = custom_kg_incremental.build_custom_kg_manifest(payload, lightrag_version="1.5.0", embedding_model="test", embedding_dim=3)
    desired = json.loads(json.dumps(previous))
    desired["metadata"]["prepared"] = True
    root = sample_wiki(tmp_path)
    workdir = tmp_path / "work" / "lightrag"
    state = workdir / "state"
    ensure_state_dirs(state)
    live_storage = workdir / "rag_storage"
    shadow_storage = workdir / "rag_storage.shadow.test"
    live_storage.mkdir(parents=True)
    shadow_storage.mkdir(parents=True)
    write(live_storage / "live.txt", "live")
    write(shadow_storage / "shadow.txt", "shadow")
    custom_kg_incremental.write_manifest(state, previous)
    desired_path = custom_kg_incremental.prepared_swap_manifest_path(state)
    desired_path.parent.mkdir(parents=True, exist_ok=True)
    desired_path.write_text(json.dumps(desired, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = custom_kg_incremental.prepared_swap_report_path(state)
    report_path.write_text(
        json.dumps(
            {
                "prepared_for_swap": True,
                "shadow_storage": str(shadow_storage),
                "backup_dir": str(state / "backups" / "backup"),
                "desired_manifest_path": str(desired_path),
                "previous_manifest_hash": custom_kg_incremental.stable_hash(previous),
                "desired_manifest_hash": custom_kg_incremental.stable_hash(desired),
                "payload": {"chunks": 1, "entities": 1, "relationships": 0},
                "diff": {"chunks": {"add": 1, "update": 0, "delete": 0}, "entities": {"add": 0, "update": 0, "delete": 0}, "relationships": {"add": 0, "update": 0, "delete": 0}},
                "plan": {"selected_mode": "incremental"},
                "pre_audit": {"ok": True, "issues": []},
                "timings": {"apply_patch_to_shadow_s": 1.0},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(custom_kg_incremental, "port_open", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(custom_kg_incremental, "audit_custom_kg_storage", lambda *_args, **_kwargs: {"ok": True, "issues": [], "counts": {}})
    args = types.SimpleNamespace(root=root, state_dir=state, workdir=workdir, prepared_report=report_path, server_host="127.0.0.1", server_port=9621, allow_server_running=False)

    report = custom_kg_incremental.run_finalize_prepared_swap(args)

    assert report["swapped"] is True
    assert (workdir / "rag_storage" / "shadow.txt").exists()
    assert (state / "backups" / "backup" / "live.txt").exists()
    final_manifest = custom_kg_incremental.load_manifest(state)
    assert final_manifest["metadata"]["last_successful_import_mode"] == "incremental"
    assert Path(report["manifest_path"]).exists()
    assert (state / custom_kg_incremental.REPORT_FILENAME).exists()


def test_custom_kg_storage_audit_detects_graph_vdb_mismatch_and_unknown_source(tmp_path: Path) -> None:
    import networkx as nx
    from custom_kg_incremental import audit_custom_kg_storage, build_custom_kg_manifest

    payload = {
        "chunks": [{"content": "Doc A", "source_id": "doc:a", "file_path": "a.md"}],
        "entities": [
            {"entity_name": "doc:a", "entity_type": "DOC", "description": "Doc A", "source_id": "doc:a", "file_path": "a.md"},
            {"entity_name": "topic:x", "entity_type": "TOPIC", "description": "Topic", "source_id": "doc:a", "file_path": "a.md"},
        ],
        "relationships": [{"src_id": "doc:a", "tgt_id": "topic:x", "description": "edge", "keywords": "RELATED", "source_id": "doc:a", "file_path": "a.md"}],
    }
    manifest = build_custom_kg_manifest(payload, lightrag_version="1.5.0", embedding_model="test", embedding_dim=3)
    storage = tmp_path / "rag_storage"
    storage.mkdir()
    chunk_id = next(iter(manifest["chunks"]))
    graph = nx.Graph()
    for entity in manifest["entities"].values():
        graph.add_node(entity["entity_name"], entity_id=entity["entity_name"], entity_type=entity["entity_type"], description=entity["description"], source_id=entity["source_chunk_id"], file_path=entity["file_path"])
    rel = next(iter(manifest["relationships"].values()))
    graph.add_edge(rel["src_id"], rel["tgt_id"], description=rel["description"], keywords=rel["keywords"], source_id=rel["source_chunk_id"], file_path=rel["file_path"])
    nx.write_graphml(graph, storage / "graph_chunk_entity_relation.graphml")
    write(storage / "vdb_chunks.json", json.dumps({"embedding_dim": 3, "data": [{"__id__": chunk_id, "content": "Doc A", "full_doc_id": "doc:a", "file_path": "a.md"}], "matrix": ""}))
    write(storage / "kv_store_text_chunks.json", json.dumps({chunk_id: {"content": "Doc A", "source_id": "doc:a", "full_doc_id": "doc:a", "file_path": "a.md"}}))
    write(storage / "vdb_entities.json", json.dumps({"embedding_dim": 3, "data": [{"__id__": entity["vdb_id"], "entity_name": entity["entity_name"], "source_id": entity["source_chunk_id"], "file_path": entity["file_path"]} for entity in manifest["entities"].values()], "matrix": ""}))
    write(storage / "vdb_relationships.json", json.dumps({"embedding_dim": 3, "data": [{"__id__": rel["vdb_id"], "src_id": rel["src_id"], "tgt_id": rel["tgt_id"], "source_id": rel["source_chunk_id"], "file_path": rel["file_path"]}], "matrix": ""}))
    write(storage / "kv_store_entity_chunks.json", json.dumps({name: {"chunk_ids": [entity["source_chunk_id"]], "count": 1} for name, entity in manifest["entities"].items()}))
    write(storage / "kv_store_relation_chunks.json", json.dumps({rel["chunk_key"]: {"chunk_ids": [rel["source_chunk_id"]], "count": 1}}))

    ok = audit_custom_kg_storage(storage, manifest)
    assert ok["ok"] is True

    graph.nodes["topic:x"]["source_id"] = "UNKNOWN"
    nx.write_graphml(graph, storage / "graph_chunk_entity_relation.graphml")
    bad = audit_custom_kg_storage(storage, manifest)
    assert bad["ok"] is False
    assert any(issue["type"] == "unknown_source_id" for issue in bad["issues"])


def test_batch_lightrag_refresh_uses_incremental_apply_when_plan_allows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"
    workdir = tmp_path / "work" / "lightrag"
    artifact_log = state / "refresh_logs" / "artifact.log"
    import_log = state / "refresh_logs" / "import.log"
    artifact_commands = [["artifact", str(idx)] for idx in range(7)]
    full_import_commands = [
        ["systemctl", "--user", "stop", batch_lightrag_refresh.SERVICE_NAME],
        ["python-internal", "reset-rag-storage", str(workdir / "rag_storage"), str(workdir / "inputs")],
        [str(batch_lightrag_refresh.LIGHTRAG_PYTHON), str(workdir / "scripts" / "import_custom_kg.py")],
        ["systemctl", "--user", "start", batch_lightrag_refresh.SERVICE_NAME],
        ["python-internal", "health", "http://127.0.0.1:9621/health"],
    ]
    calls: list[list[str]] = []
    release_calls: list[str] = []

    monkeypatch.setattr(batch_lightrag_refresh, "build_refresh_commands", lambda *_args: artifact_commands + full_import_commands)
    monkeypatch.setattr(
        batch_lightrag_refresh,
        "plan_incremental_import_mode",
        lambda *_args, **_kwargs: {
            "selected_mode": "incremental",
            "reasons": [],
            "diff": {
                "chunks": {"add": 1, "update": 0, "delete": 0},
                "entities": {"add": 0, "update": 0, "delete": 0},
                "relationships": {"add": 0, "update": 0, "delete": 0},
            },
        },
    )
    monkeypatch.setattr(batch_lightrag_refresh, "wait_health", lambda *_args, **_kwargs: {"status": "healthy", "pipeline_busy": False})
    monkeypatch.setattr(batch_lightrag_refresh, "clear_lightrag_refresh_pending_after_success", lambda *_args, **_kwargs: {"cleared_count": 1})
    monkeypatch.setattr(batch_lightrag_refresh, "release_process_memory", lambda: release_calls.append("released"))

    def fake_run_subprocess(command: list[str], *_args, **_kwargs) -> None:
        calls.append(command)

    monkeypatch.setattr(batch_lightrag_refresh, "run_subprocess", fake_run_subprocess)

    result = batch_lightrag_refresh.run_real_refresh(root, state, workdir, "threshold", artifact_log, import_log)

    assert result["import_mode"]["selected_mode"] == "incremental"
    assert release_calls
    assert not any(command[:2] == ["python-internal", "reset-rag-storage"] for command in calls)
    prepare_idx = next(idx for idx, command in enumerate(calls) if "custom_kg_incremental.py" in " ".join(command) and "apply" in command)
    stop_idx = next(idx for idx, command in enumerate(calls) if command[:3] == ["systemctl", "--user", "stop"])
    finalize_idx = next(idx for idx, command in enumerate(calls) if "custom_kg_incremental.py" in " ".join(command) and "finalize-prepared-swap" in command)
    start_idx = next(idx for idx, command in enumerate(calls) if command[:3] == ["systemctl", "--user", "start"])
    assert prepare_idx < stop_idx < finalize_idx < start_idx
    assert "--prepare-swap" in calls[prepare_idx]
    assert "--prepared-report" in calls[finalize_idx]


def test_batch_lightrag_refresh_uses_full_materialization_for_scheduled_full_rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"
    workdir = tmp_path / "work" / "lightrag"
    artifact_log = state / "refresh_logs" / "artifact.log"
    import_log = state / "refresh_logs" / "import.log"
    artifact_commands = [["artifact", str(idx)] for idx in range(7)]
    cold_full_commands = [
        ["systemctl", "--user", "stop", batch_lightrag_refresh.SERVICE_NAME],
        ["python-internal", "reset-rag-storage", str(workdir / "rag_storage"), str(workdir / "inputs")],
        [str(batch_lightrag_refresh.LIGHTRAG_PYTHON), str(workdir / "scripts" / "import_custom_kg.py")],
        ["systemctl", "--user", "start", batch_lightrag_refresh.SERVICE_NAME],
        ["python-internal", "health", "http://127.0.0.1:9621/health"],
    ]
    calls: list[list[str]] = []

    monkeypatch.setattr(batch_lightrag_refresh, "build_refresh_commands", lambda *_args: artifact_commands + cold_full_commands)
    monkeypatch.setattr(
        batch_lightrag_refresh,
        "plan_incremental_import_mode",
        lambda *_args, **_kwargs: {
            "selected_mode": "full_rebuild",
            "reasons": ["incremental_interval_reached"],
            "diff": {},
        },
    )
    monkeypatch.setattr(batch_lightrag_refresh, "wait_health", lambda *_args, **_kwargs: {"status": "healthy", "pipeline_busy": False})
    monkeypatch.setattr(batch_lightrag_refresh, "clear_lightrag_refresh_pending_after_success", lambda *_args, **_kwargs: {"cleared_count": 1})
    monkeypatch.setattr(batch_lightrag_refresh, "run_subprocess", lambda command, *_args, **_kwargs: calls.append(command))

    result = batch_lightrag_refresh.run_real_refresh(root, state, workdir, "threshold", artifact_log, import_log)

    assert result["import_mode"]["selected_mode"] == "full_rebuild"
    assert not any(command[:2] == ["python-internal", "reset-rag-storage"] for command in calls)
    assert not any("import_custom_kg.py" in " ".join(command) for command in calls)
    prepare_idx = next(idx for idx, command in enumerate(calls) if "custom_kg_incremental.py" in " ".join(command) and "materialize-full" in command)
    stop_idx = next(idx for idx, command in enumerate(calls) if command[:3] == ["systemctl", "--user", "stop"])
    finalize_idx = next(idx for idx, command in enumerate(calls) if "custom_kg_incremental.py" in " ".join(command) and "finalize-prepared-swap" in command)
    start_idx = next(idx for idx, command in enumerate(calls) if command[:3] == ["systemctl", "--user", "start"])
    assert prepare_idx < stop_idx < finalize_idx < start_idx
    prepare = calls[prepare_idx]
    assert "--no-swap" in prepare
    assert "--prepare-swap" in prepare
    assert "--seed-from-storage" in prepare
    assert "--vector-cache" in prepare
    assert "--prepared-report" in calls[finalize_idx]


def test_batch_lightrag_refresh_keeps_cold_full_import_for_unsafe_full_rebuild_reasons(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"
    workdir = tmp_path / "work" / "lightrag"
    cold_full_commands = [
        ["systemctl", "--user", "stop", batch_lightrag_refresh.SERVICE_NAME],
        ["python-internal", "reset-rag-storage", str(workdir / "rag_storage"), str(workdir / "inputs")],
        [str(batch_lightrag_refresh.LIGHTRAG_PYTHON), str(workdir / "scripts" / "import_custom_kg.py")],
        ["systemctl", "--user", "start", batch_lightrag_refresh.SERVICE_NAME],
        ["python-internal", "health", "http://127.0.0.1:9621/health"],
    ]

    for reasons in (["missing_manifest"], ["current_storage_audit_failed"], ["embedding_model_changed"], ["embedding_dim_changed"]):
        commands = batch_lightrag_refresh.select_import_commands(
            root,
            state,
            workdir,
            cold_full_commands,
            {"selected_mode": "full_rebuild", "reasons": reasons},
        )
        assert commands == cold_full_commands


def test_batch_lightrag_refresh_falls_back_to_cold_full_when_materialization_prepare_fails_before_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"
    workdir = tmp_path / "work" / "lightrag"
    artifact_log = state / "refresh_logs" / "artifact.log"
    import_log = state / "refresh_logs" / "import.log"
    calls: list[list[str]] = []

    monkeypatch.setattr(
        batch_lightrag_refresh,
        "plan_incremental_import_mode",
        lambda *_args, **_kwargs: {"selected_mode": "full_rebuild", "reasons": ["incremental_interval_reached"], "diff": {}},
    )
    monkeypatch.setattr(batch_lightrag_refresh, "wait_health", lambda *_args, **_kwargs: {"status": "healthy", "pipeline_busy": False})
    monkeypatch.setattr(batch_lightrag_refresh, "clear_lightrag_refresh_pending_after_success", lambda *_args, **_kwargs: {"cleared_count": 1})
    monkeypatch.setattr(batch_lightrag_refresh, "reset_rag_storage", lambda *_args: calls.append(["python-internal", "reset-rag-storage"]))

    def fake_run_subprocess(command: list[str], *_args, **_kwargs) -> None:
        calls.append(command)
        if "materialize-full" in command:
            raise RuntimeError("cache miss before stop")

    monkeypatch.setattr(batch_lightrag_refresh, "run_subprocess", fake_run_subprocess)

    result = batch_lightrag_refresh.run_real_refresh(root, state, workdir, "threshold", artifact_log, import_log)

    assert result["import_mode"]["full_materialization_fallback"] == "cold_full_import"
    materialize_idx = next(idx for idx, command in enumerate(calls) if "materialize-full" in command)
    reset_idx = next(idx for idx, command in enumerate(calls) if command[:2] == ["python-internal", "reset-rag-storage"])
    import_idx = next(idx for idx, command in enumerate(calls) if "import_custom_kg.py" in " ".join(command))
    assert materialize_idx < reset_idx < import_idx


def test_batch_lightrag_refresh_skips_import_when_incremental_plan_has_empty_diff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"
    workdir = tmp_path / "work" / "lightrag"
    artifact_log = state / "refresh_logs" / "artifact.log"
    import_log = state / "refresh_logs" / "import.log"
    artifact_commands = [["artifact", str(idx)] for idx in range(7)]
    full_import_commands = [
        ["systemctl", "--user", "stop", batch_lightrag_refresh.SERVICE_NAME],
        ["python-internal", "reset-rag-storage", str(workdir / "rag_storage"), str(workdir / "inputs")],
        [str(batch_lightrag_refresh.LIGHTRAG_PYTHON), str(workdir / "scripts" / "import_custom_kg.py")],
        ["systemctl", "--user", "start", batch_lightrag_refresh.SERVICE_NAME],
        ["python-internal", "health", "http://127.0.0.1:9621/health"],
    ]
    calls: list[list[str]] = []
    clear_calls: list[dict[str, object]] = []
    empty_diff = {
        "chunks": {"add": 0, "update": 0, "delete": 0},
        "entities": {"add": 0, "update": 0, "delete": 0},
        "relationships": {"add": 0, "update": 0, "delete": 0},
    }

    monkeypatch.setattr(batch_lightrag_refresh, "build_refresh_commands", lambda *_args: artifact_commands + full_import_commands)
    monkeypatch.setattr(batch_lightrag_refresh, "plan_incremental_import_mode", lambda *_args, **_kwargs: {"selected_mode": "incremental", "reasons": [], "diff": empty_diff})

    def fake_clear(*_args, **kwargs):
        clear_calls.append(kwargs)
        return {"cleared_count": 1}

    monkeypatch.setattr(batch_lightrag_refresh, "clear_lightrag_refresh_pending_after_success", fake_clear)
    monkeypatch.setattr(batch_lightrag_refresh, "run_subprocess", lambda command, *_args, **_kwargs: calls.append(command))

    result = batch_lightrag_refresh.run_real_refresh(root, state, workdir, "threshold", artifact_log, import_log)

    assert result["import_skipped"] is True
    assert result["import_skip_reason"] == "incremental_empty_diff"
    assert not any(command and command[0] == "systemctl" for command in calls)
    assert not any("custom_kg_incremental.py" in " ".join(command) and "apply" in command for command in calls)
    assert clear_calls == [{"reason": "threshold"}]


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
    from wiki_lightrag_lib import build_section_similarity_edges_from_index, section_similarity_index_summary

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
    from wiki_lightrag_lib import jsonl_read, section_similarity_index_summary

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
    state = tmp_path / "work" / "lightrag" / "state"
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
    filtered = filter_lightrag_data_response_by_section_kind(response, "methodology")
    chunks = filtered["data"]["chunks"]
    assert len(chunks) == 1
    assert chunks[0]["file_path"] == "raw_section_docs/a.md"
    assert response["data"]["chunks"][1]["file_path"] == "raw_section_docs/b.md"


def test_expand_lightrag_data_response_with_section_neighbors_keeps_direct_hits_separate(tmp_path: Path) -> None:
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
    expanded = expand_lightrag_data_response_with_section_neighbors(response, state, neighbor_k=1, section_kind="future")
    neighbors = expanded["data"]["section_neighbor_expansions"]
    assert len(neighbors) == 1
    assert neighbors[0]["seed_section_id"] == "raw_section:a:future"
    assert neighbors[0]["neighbor_section_id"] == "raw_section:b:future"
    assert neighbors[0]["cosine"] == 0.88
    assert expanded["data"]["chunks"] == response["data"]["chunks"]


def test_import_custom_kg_build_rag_honors_embedding_throttle_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from import_custom_kg import build_rag

    captured: dict[str, object] = {}

    class FakeLightRAG:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    class FakeEmbeddingFunc:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    async def fake_complete(*args: object, **kwargs: object) -> str:
        return ""

    async def fake_embed(*args: object, **kwargs: object) -> list[list[float]]:
        return []

    fake_lightrag = types.ModuleType("lightrag")
    fake_lightrag.LightRAG = FakeLightRAG
    fake_openai = types.ModuleType("lightrag.llm.openai")
    fake_openai.openai_complete_if_cache = fake_complete
    fake_openai.openai_embed = types.SimpleNamespace(func=fake_embed)
    fake_utils = types.ModuleType("lightrag.utils")
    fake_utils.EmbeddingFunc = FakeEmbeddingFunc
    monkeypatch.setitem(sys.modules, "lightrag", fake_lightrag)
    monkeypatch.setitem(sys.modules, "lightrag.llm", types.ModuleType("lightrag.llm"))
    monkeypatch.setitem(sys.modules, "lightrag.llm.openai", fake_openai)
    monkeypatch.setitem(sys.modules, "lightrag.utils", fake_utils)
    monkeypatch.setenv("EMBEDDING_BATCH_NUM", "2")
    monkeypatch.setenv("EMBEDDING_FUNC_MAX_ASYNC", "1")

    build_rag(tmp_path)

    assert captured["embedding_batch_num"] == 2
    assert captured["embedding_func_max_async"] == 1


def test_generated_virtual_doc_builders_remove_stale_markdown(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    write(
        root / "raw/clip/2601/26010101_Bar-Paper.md",
        "---\ntitle: Bar Paper\nupdated: 2026-05-18 16:00\n---\n# Bar Paper\n\n## Methodology\n\n"
        "This is another direct method with enough detail to become a separate method atom. "
        "It intentionally shares the same date prefix as Foo Paper.\n",
    )
    state = tmp_path / "work" / "lightrag" / "state"
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
            "/home/xu/.hermes/skills/research/llm-wiki/scripts/raw_fast_note_verify.py",
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
            "/home/xu/.hermes/skills/research/llm-wiki/scripts/raw_fast_note_verify.py",
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


def test_raw_fast_closeout_marks_pending_after_verifier_and_cleans_tmp(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"
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
    assert payload["lightrag"]["blocked_by_pending_wiki_integration"] is True
    assert payload["timings"]["total_seconds"] >= 0
    _assert_timing_step(payload, "pre_verify")
    _assert_timing_step(payload, "control_scan")
    _assert_timing_step(payload, "evidence_report_capture")
    _assert_timing_step(payload, "mark_pending")
    _assert_timing_step(payload, "final_verify")
    _assert_timing_step(payload, "lightrag_status")
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


def test_raw_fast_closeout_does_not_mark_pending_when_verifier_fails(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"
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
    state = tmp_path / "work" / "lightrag" / "state"
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
        "lightrag": {"blocked_by_pending_wiki_integration": True, "graph_ready_pending_count": 0, "should_refresh": False},
    }

    entry = raw_fast_closeout.build_compact_log_entry(args, output)

    assert len(entry.splitlines()) <= 5
    assert "26010112_Compact-Log-Paper.md" in entry
    assert "raw_fast_ok=true" in entry
    assert "checksums" not in entry.lower()


def test_threshold_lightrag_status_skips_prequery_freshness_scan(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "lightrag" / "state"
    mark_pending_wiki_integration(state, root, raw_path="raw/clip/2601/26010113_Pending.md", title="Pending")

    status = pending_lightrag_refresh_status(root, state, reason="threshold")

    assert status["blocked_by_pending_wiki_integration"] is True
    assert status["latest_wiki_markdown_mtime"] is None
    assert status["import_report"] == {}
