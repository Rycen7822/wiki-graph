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
import batch_wiki_integration  # noqa: E402
import raw_fast_closeout  # noqa: E402
import validate_wiki as validate_wiki_cli  # noqa: E402

from wiki_native_artifacts import build_seed_edges, extract_method_atoms, resolve_source  # noqa: E402
from wiki_native_custom_kg_payload import build_custom_kg_payload  # noqa: E402
from wiki_native_docs import (  # noqa: E402
    canonical_id_for,
    collect_source_docs,
    fallback_frontmatter_load,
    generated_docs_from_state,
    parse_frontmatter,
)
from wiki_native_ingest_text import make_ingest_text  # noqa: E402
from wiki_native_jsonl import jsonl_read  # noqa: E402
from wiki_native_lib import clear_pending_wiki_integration_after_success  # noqa: E402
from wiki_native_query_events import init_query_events_db  # noqa: E402
from wiki_native_query_response import (  # noqa: E402
    expand_native_data_response_with_section_neighbors,
    filter_native_data_response_by_section_kind,
)
from wiki_native_raw_section_extract import extract_raw_sections  # noqa: E402
from wiki_native_raw_sections import raw_section_query_for_kind, raw_section_specs_for_heading  # noqa: E402
from wiki_native_section_similarity import (  # noqa: E402
    _section_rank_lists,
    _section_rank_lists_scalar,
    build_section_similarity_edges,
    build_section_similarity_edges_from_index,
    section_similarity_embedding_text,
    section_similarity_index_summary,
    section_similarity_report_summary,
    select_section_similarity_edges,
)
from wiki_native_state import ensure_state_dirs  # noqa: E402
from wiki_native_validation import validate_wiki  # noqa: E402
from wiki_native_wiki_checks import (  # noqa: E402
    VALIDATION_REPORT_FRESHNESS_SCHEMA_VERSION,
    audit_raw_note_section_contracts,
    structured_heading_warnings,
    validation_freshness_context,
    validation_report_is_fresh,
    wiki_root_machine_pollution,
)
from wiki_native_wiki_integration_pending import (  # noqa: E402
    DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD,
    load_pending_wiki_integration_ledger,
    mark_pending_wiki_integration,
    pending_wiki_integration_status,
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


def test_state_dirs_and_query_event_db_are_external_to_wiki_root(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    workdir = tmp_path / "work" / "native"
    state = workdir / "state"
    ensure_state_dirs(state)
    assert (state / "edge_docs").is_dir()
    assert not (root / ".llm-wiki").exists()
    db = init_query_events_db(state)
    assert db == state / "native_query_events.db"
    with sqlite3.connect(db) as conn:
        tables = {row[0] for row in conn.execute("select name from sqlite_master where type='table'")}
        columns = {row[1] for row in conn.execute("pragma table_info(query_events)")}
    assert tables == {"query_events", "sqlite_sequence"}
    assert {"query", "mode", "rewritten_queries", "evidence_pack_path", "created_at"} <= columns
    assert "docs" not in tables
    assert "sync_events" not in tables


def test_jsonl_read_streams_rows_in_order_and_skips_blank_lines(tmp_path: Path) -> None:

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
    root = validation_reuse_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    workdir = tmp_path / "work" / "wikigraph"

    report = validate_wiki(root, state, workdir, full=True, write_report=True)
    current = validation_freshness_context(root, state, workdir)
    freshness = validation_report_is_fresh(
        report,
        current,
        required_surfaces=["index", "compiled", "_meta", "raw"],
        reason="refresh-artifact",
    )

    assert freshness == {"fresh": True, "rejections": []}
    assert report["schema_version"] == VALIDATION_REPORT_FRESHNESS_SCHEMA_VERSION
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
    report, current = _fresh_validation_report_inputs()

    result = validation_report_is_fresh(
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
    report, current = _fresh_validation_report_inputs()
    mutator(report, current)

    result = validation_report_is_fresh(report, current, required_surfaces=required_surfaces, reason=reason)

    assert result["fresh"] is False
    assert expected_rejection in result["rejections"]


def test_false_changed_only_flags_are_removed_from_cli_help() -> None:
    for script_name in [
        "validate_wiki.py",
        "build_seed_edges.py",
        "extract_method_atoms.py",
        "extract_raw_sections.py",
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



def test_query_evidence_pack_uses_canonical_wiki_search_cli() -> None:
    import importlib.util

    assert importlib.util.find_spec("build_evidence_pack") is None
    assert not (SCRIPTS / "build_evidence_pack.py").exists()
    assert (SCRIPTS / "wiki_search.py").exists()


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


def test_review_wiki_integration_status_routes_manual_review_items_without_old_graph_status(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    mark_pending_wiki_integration(state, root, raw_path="raw/clip/2601/26010400_Needs-Review.md", title="Needs Review", status="needs_review")

    wiki_status = pending_wiki_integration_status(root, state, reason="pre-query")

    assert wiki_status["actionable_pending_count"] == 0
    assert wiki_status["review_pending_count"] == 1
    assert "pending_items_need_review" in wiki_status["reasons"]
    assert wiki_status["next_required_action"] == "manual_review"


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
    assert "references/raw-fast-batch-wiki-integration.md" in prompt
    assert "references/wiki-core-operations.md" in prompt
    assert "references/wiki-operational-pitfalls.md" not in prompt


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
    from custom_kg_incremental import build_custom_kg_manifest, entity_record_id, relationship_record_id, stable_hash

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
    assert topic["record_id"] == entity_record_id("topic:x")
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
    assert old_rel["record_id"] == relationship_record_id("topic:x", "doc:a", "OLD")
    assert new_rel["description"] == "new"
    assert new_rel["src_id"] == "doc:a"
    assert new_rel["tgt_id"] == "topic:x"
    assert new_rel["source_chunk_id"] == chunk_id
    assert new_rel["record_id"] == relationship_record_id("doc:a", "topic:x", "NEW")
    assert new_rel["record_type"] == "relationship"
    assert new_rel["canonical_id"] == new_rel["chunk_key"]
    assert "vdb_id" not in json.dumps(manifest, ensure_ascii=False)
    assert new_rel["vector_text_hash"] == stable_hash(new_rel["content"])


def test_custom_kg_manifest_matches_native_sanitized_chunk_ids_and_basenames() -> None:
    import custom_kg_incremental
    from custom_kg_incremental import build_custom_kg_manifest, compute_mdhash_id, native_manifest_sanitize_text

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
    sanitized = native_manifest_sanitize_text(raw_content)
    chunk_id = compute_mdhash_id(sanitized, prefix="chunk-")

    assert sanitized == "A & B"
    assert custom_kg_incremental.native_manifest_normalize_file_path("nested/doc.[native-iet].md") == "doc.md"
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
    filtered = filter_native_data_response_by_section_kind(response, "methodology")
    chunks = filtered["data"]["chunks"]
    assert len(chunks) == 1
    assert chunks[0]["file_path"] == "raw_section_docs/a.md"
    assert response["data"]["chunks"][1]["file_path"] == "raw_section_docs/b.md"


def test_expand_native_data_response_with_section_neighbors_keeps_direct_hits_separate(tmp_path: Path) -> None:
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
    expanded = expand_native_data_response_with_section_neighbors(response, state, neighbor_k=1, section_kind="future")
    neighbors = expanded["data"]["section_neighbor_expansions"]
    assert len(neighbors) == 1
    assert neighbors[0]["seed_section_id"] == "raw_section:a:future"
    assert neighbors[0]["neighbor_section_id"] == "raw_section:b:future"
    assert neighbors[0]["cosine"] == 0.88
    assert expanded["data"]["chunks"] == response["data"]["chunks"]


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


def test_raw_fast_evidence_bundle_probe_choices_are_current_only() -> None:
    source = (SCRIPTS / "raw_fast_evidence_bundle.py").read_text(encoding="utf-8")
    help_result = subprocess.run(
        [sys.executable, str(SCRIPTS / "raw_fast_evidence_bundle.py"), "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert 'choices=["github", "hf", "project", "arxiv", "doi", "none"]' not in source
    assert "{arxiv,doi,none}" in help_result.stdout
    assert "GitHub/HF/project probes" not in help_result.stdout


def test_raw_fast_evidence_bundle_has_no_retired_deep_probe_helpers() -> None:
    source = (SCRIPTS / "raw_fast_evidence_bundle.py").read_text(encoding="utf-8")

    for retired in [
        "def probe_github_repo",
        "def probe_hf_search",
        '"verified_absent"',
        '"candidates_unverified"',
    ]:
        assert retired not in source


def test_raw_fast_evidence_bundle_resource_boundary_defaults_to_not_checked_without_exact_link_report() -> None:
    import raw_fast_evidence_bundle

    resource_probe = {
        "ok": True,
        "probes": [
            {"ok": True, "type": "doi", "doi": "10.1234/example.paper", "status": "detected", "url": "https://doi.org/10.1234/example.paper"},
            {"ok": True, "type": "arxiv", "id": "2604.08999", "status": "detected", "url": "https://arxiv.org/abs/2604.08999"},
            {"ok": True, "type": "github_repo", "repo": "example/ignored"},
            {"ok": True, "type": "hf_models", "query": "Fixture", "count": 0, "items": []},
            {"ok": True, "type": "project_page", "url": "https://example.test/project"},
        ],
    }

    draft = raw_fast_evidence_bundle.summarize_resource_boundary(resource_probe, metadata={"title": "Fixture Sidecar Paper"})
    markdown = raw_fast_evidence_bundle.render_resource_boundary_markdown(draft)

    assert draft["github"] == []
    assert draft["project_pages"] == []
    assert {kind: bucket["status"] for kind, bucket in draft["hf"].items()} == {"models": "not_checked", "datasets": "not_checked", "spaces": "not_checked"}
    assert draft["doi"][0]["doi"] == "10.1234/example.paper"
    assert draft["arxiv"][0]["id"] == "2604.08999"
    assert "verified_absent" not in markdown
    assert "candidates_unverified" not in markdown
    assert "not_checked" in markdown


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
        {"links": [{"uri": "https://github.com/example/sidecar"}]},
        ["arxiv", "doi"],
    )
    probes = {(item["type"], item.get("id") or item.get("doi")): item for item in payload["probes"]}

    assert ("arxiv", "2604.08999") in probes
    assert probes[("arxiv", "2604.08999")]["ok"] is True
    assert probes[("arxiv", "2604.08999")]["status"] == "detected"
    assert probes[("arxiv", "2604.08999")]["evidence"]
    assert ("doi", "10.1234/example.paper") in probes
    assert probes[("doi", "10.1234/example.paper")]["status"] == "detected"
    assert "https://github.com/example/sidecar" in payload["urls"]
    assert "github_repos" not in payload


def test_raw_fast_evidence_bundle_none_probe_keeps_url_inventory() -> None:
    import raw_fast_evidence_bundle

    payload = raw_fast_evidence_bundle.build_resource_probe(
        "Project page: https://example.test/project and arXiv:2604.08999.",
        {"links": [{"uri": "https://huggingface.co/example/model"}]},
        ["none"],
    )

    assert payload["ok"] is True
    assert payload["skipped"] is True
    assert payload["probes"] == []
    assert payload["urls"] == ["https://example.test/project", "https://huggingface.co/example/model"]


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
