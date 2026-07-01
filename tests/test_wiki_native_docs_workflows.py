import sys
import json
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from support import sample_wiki, write  # noqa: E402
import ops.validate_wiki as validate_wiki_cli  # noqa: E402
from ops.wiki_native_artifacts import resolve_source  # noqa: E402
from llm_wiki_native.source_docs import canonical_id_for  # noqa: E402
from llm_wiki_native.source_docs import collect_source_docs  # noqa: E402
from llm_wiki_native.source_docs import fallback_frontmatter_load  # noqa: E402
from llm_wiki_native.source_docs import generated_docs_from_state  # noqa: E402
from llm_wiki_native.source_docs import parse_frontmatter  # noqa: E402
from ops.wiki_native_ingest_text import make_ingest_text  # noqa: E402
from ops.wiki_native_query_events import init_query_events_db  # noqa: E402
from ops.wiki_native_state import ensure_state_dirs  # noqa: E402
from ops.wiki_native_validation import validate_wiki  # noqa: E402
from ops.wiki_native_wiki_checks import VALIDATION_REPORT_FRESHNESS_SCHEMA_VERSION  # noqa: E402
from ops.wiki_native_wiki_checks import validation_freshness_context  # noqa: E402
from ops.wiki_native_wiki_checks import validation_report_is_fresh  # noqa: E402
from ops.wiki_native_wiki_checks import wiki_root_machine_pollution  # noqa: E402


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


def test_validate_wiki_default_does_not_write_report(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"

    report = validate_wiki(root, state, tmp_path / "work" / "wikigraph")

    assert "report_path" not in report
    assert not state.exists()
    assert not list((state / "validation_reports").glob("*_validate.json"))


def test_validate_wiki_report_uses_native_output_fields(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    workdir = tmp_path / "work" / "wikigraph"

    report = validate_wiki(root, state, workdir)

    assert report["native_unresolved_references"] == 0
    assert report["native_state_dir"] == str(state.resolve())
    assert report["native_workdir"] == str(workdir.resolve())


def test_validate_wiki_can_sync_raw_clip_map_snapshot_when_explicit(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    write(
        root / "raw" / "clip" / "2601" / "26010102_Bar-Paper.md",
        "---\ntitle: Bar Paper\nsource: https://arxiv.org/abs/2601.0102\ndomain: paper\nupdated: 2026-05-18 16:00\ntags: [paper]\n---\n# Bar Paper\n",
    )
    raw_map = root / "_meta" / "raw-clip-map.md"
    write(raw_map, "# Raw Clip Map\n\nActive raw clips: 1\n")
    before_text = raw_map.read_text(encoding="utf-8")

    stale = validate_wiki(root, tmp_path / "state", tmp_path / "work")

    assert raw_map.read_text(encoding="utf-8") == before_text
    assert stale["active_raw_clips"] == 2
    assert stale["raw_clip_map_snapshot"] == 1
    assert "active_raw_clips != raw-clip-map snapshot (2 != 1)" in stale["warnings"]

    synced = validate_wiki(root, tmp_path / "state", tmp_path / "work", sync_raw_map_snapshot=True)

    assert synced["active_raw_clips"] == 2
    assert synced["raw_clip_map_snapshot"] == 2
    assert synced["raw_clip_map_sync"]["changed"] is True
    assert synced["raw_clip_map_sync"]["previous_snapshot"] == 1
    assert not [warning for warning in synced["warnings"] if "raw-clip-map snapshot" in warning]
    assert "Active raw clips: 2" in raw_map.read_text(encoding="utf-8")


def test_validate_wiki_non_full_does_not_read_raw_bodies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ops import wiki_native_lib

    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    original_read_text = wiki_native_lib.read_text

    def fail_on_raw_body_reads(path: Path) -> str:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
        if rel.startswith("raw/"):
            raise AssertionError(f"non-full validation should not read raw body: {rel}")
        return original_read_text(path)

    monkeypatch.setattr(wiki_native_lib, "read_text", fail_on_raw_body_reads)

    report = validate_wiki(root, state, tmp_path / "work" / "wikigraph", full=False)

    assert report["active_raw_clips"] == 1
    with pytest.raises(AssertionError, match="non-full validation should not read raw body"):
        validate_wiki(root, state, tmp_path / "work" / "wikigraph", full=True)


def test_validate_wiki_without_write_report_does_not_hash_freshness_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ops import wiki_native_wiki_checks

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


def test_machine_pollution_detection(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    assert wiki_root_machine_pollution(root) == []
    write(root / ".llm-wiki" / "bad.txt", "bad")
    retired_backend = "light" + "rag"
    retired_manifest_name = f"{retired_backend}_manifest.jsonl"
    write(root / retired_manifest_name, "{}\n")
    write(root / "raw" / "clip" / "2601" / "seed_edges.jsonl", "{}\n")
    polluted = {p.as_posix() for p in wiki_root_machine_pollution(root)}
    assert ".llm-wiki" in polluted
    assert retired_manifest_name in polluted
    assert "raw/clip/2601/seed_edges.jsonl" in polluted


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


def test_generated_docs_from_state_is_read_only_when_state_is_missing(tmp_path: Path) -> None:
    state = tmp_path / "work" / "wikigraph" / "state"

    assert generated_docs_from_state(state, kind="raw_section") == []
    assert not state.exists()
