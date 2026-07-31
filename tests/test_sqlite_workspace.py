import sqlite3

import pytest

from llm_wiki_native.storage.sqlite_workspace import SQLiteWorkspace
from support import native_record


def _workspace_record(workspace_id: str):
    return native_record(
        workspace_id,
        content_hash="content-hash",
        metadata_hash="metadata-hash",
        vector_hash="vector-hash",
        source_path="a.md",
        source_id="doc:a",
        payload={"file_path": "a.md"},
    )


def test_create_workspace_initializes_schema_and_status(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    db.put_record(_workspace_record("native-test"))
    db.mark_audited("native-test", {"chunks": 1, "entities": 0, "relationships": 0, "sections": 0})

    assert db.get_workspace_status("native-test") == "audited"
    assert db.get_workspace_metadata("native-test") == {
        "workspace_id": "native-test",
        "source_manifest_hash": "manifest-hash",
        "schema_version": 1,
        "status": "audited",
    }
    assert db.audit_counts("native-test", {"chunks": 1, "entities": 0, "relationships": 0, "sections": 0})["ok"] is True


def test_open_existing_read_only_reads_without_allowing_writes(tmp_path) -> None:
    db_path = tmp_path / "native.sqlite"
    writable = SQLiteWorkspace(db_path)
    writable.create_workspace("native-test", "manifest-hash")
    writable.put_record(_workspace_record("native-test"))
    writable.mark_audited("native-test", {"chunks": 1, "entities": 0, "relationships": 0, "sections": 0})

    read_only = SQLiteWorkspace.open_existing(db_path, read_only=True)

    assert read_only.get_workspace_status("native-test") == "audited"
    assert read_only.get_record("native-test", "chunk", "chunk-a")["vector_text"]
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        read_only.create_workspace("should-fail", "manifest-hash")


def test_open_existing_read_only_keeps_resolved_path_after_cwd_change(
    tmp_path, monkeypatch
) -> None:
    workspace_dir = tmp_path / "workspace"
    db_path = workspace_dir / "native.sqlite"
    writable = SQLiteWorkspace(db_path)
    writable.create_workspace("native-test", "manifest-hash")
    writable.mark_audited(
        "native-test",
        {"chunks": 0, "entities": 0, "relationships": 0, "sections": 0},
    )

    monkeypatch.chdir(tmp_path)
    read_only = SQLiteWorkspace.open_existing(
        db_path.relative_to(tmp_path),
        read_only=True,
    )
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)

    assert read_only.get_workspace_status("native-test") == "audited"


def test_workspace_audit_reports_record_count_mismatch(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    db.put_record(_workspace_record("native-test"))

    audit = db.audit_counts("native-test", {"chunks": 2, "entities": 0, "relationships": 0, "sections": 0})

    assert audit["ok"] is False
    assert audit["counts"]["chunks"] == 1
    assert "chunks" in audit["issues"][0]


def test_workspace_operations_reject_unknown_workspace(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")

    with pytest.raises(KeyError, match="missing-workspace"):
        db.audit_counts("missing-workspace", {"chunks": 0, "entities": 0, "relationships": 0, "sections": 0})
    with pytest.raises(KeyError, match="missing-workspace"):
        db.put_record(_workspace_record("missing-workspace"))


def test_lexical_sidecar_queries_table_and_map_rows(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    db.put_record(_workspace_record("native-test"))
    db.mark_audited("native-test", {"chunks": 1, "entities": 0, "relationships": 0, "sections": 0})

    db.put_lexical_span(
        "native-test",
        span_id="span:doc-section",
        source_path="concepts/alpha.md",
        source_id="compiled:concept:alpha",
        source_role="compiled",
        span_kind="doc.section",
        heading_path=["Alpha"],
        start_line=3,
        end_line=6,
        text="General section text",
        metadata={"title": "Alpha"},
    )
    db.put_lexical_span(
        "native-test",
        span_id="span:table-row",
        source_path="concepts/alpha.md",
        source_id="compiled:concept:alpha",
        source_role="compiled",
        span_kind="table.row",
        heading_path=["Alpha", "Results"],
        start_line=10,
        end_line=10,
        text="| Method | CalibrationWinner | strong table evidence |",
        metadata={"columns": ["Method", "Result"]},
    )
    db.put_lexical_span(
        "native-test",
        span_id="span:map-row",
        source_path="_meta/raw-clip-map.md",
        source_id="meta:raw-clip-map",
        source_role="meta_map",
        span_kind="map.row",
        heading_path=["Raw Clip Map"],
        start_line=4,
        end_line=4,
        text="- raw/clip/2601/26010101_Foo-Paper.md :: MapOnlyNeedle",
        metadata={"map": "raw-clip"},
    )
    db.put_lexical_span(
        "native-test",
        span_id="span:late-term",
        source_path="concepts/late.md",
        source_id="compiled:concept:late",
        source_role="compiled",
        span_kind="doc.section",
        heading_path=["Late"],
        start_line=1,
        end_line=1,
        text="LateIdentifier appears only after the legacy recall cutoff",
        metadata={},
    )
    db.put_lexical_span(
        "native-test",
        span_id="span:low-coverage",
        source_path="a-first.md",
        source_id="source:low",
        source_role="raw",
        span_kind="table.row",
        heading_path=["Low"],
        start_line=1,
        end_line=1,
        text="alpha only",
        metadata={},
    )
    db.put_lexical_span(
        "native-test",
        span_id="span:high-coverage",
        source_path="z-last.md",
        source_id="source:high",
        source_role="raw",
        span_kind="table.row",
        heading_path=["High"],
        start_line=1,
        end_line=1,
        text="alpha needle",
        metadata={},
    )

    table_hits = db.query_lexical_spans("native-test", "CalibrationWinner", limit=5)
    map_hits = db.query_lexical_spans("native-test", "MapOnlyNeedle", limit=5, source_roles=("meta_map",))

    assert db.count_lexical_spans("native-test") == 6
    assert table_hits[0]["span_id"] == "span:table-row"
    assert table_hits[0]["span_kind"] == "table.row"
    assert table_hits[0]["source_path"] == "concepts/alpha.md"
    assert table_hits[0]["source_role"] == "compiled"
    assert table_hits[0]["start_line"] == 10
    assert table_hits[0]["end_line"] == 10
    assert "CalibrationWinner" in table_hits[0]["text"]
    assert table_hits[0]["route"] in {"lexical_fts", "lexical_like"}
    assert map_hits[0]["span_id"] == "span:map-row"
    assert map_hits[0]["source_role"] == "meta_map"

    long_query = "one two three four five six seven eight LateIdentifier"
    assert db.query_lexical_spans("native-test", long_query, limit=5) == []
    normalized_hits = db.query_lexical_spans(
        "native-test",
        long_query,
        limit=5,
        normalized_terms=(
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "lateidentifier",
        ),
    )
    assert [hit["span_id"] for hit in normalized_hits] == ["span:late-term"]
    assert db.query_lexical_spans(
        "native-test",
        "LateIdentifier",
        limit=5,
        normalized_terms=(),
    ) == []
    ranked_like = db.query_lexical_spans(
        "native-test",
        "alpha needle missing",
        limit=1,
        normalized_terms=("alpha", "needle", "missing"),
    )
    assert [hit["span_id"] for hit in ranked_like] == ["span:high-coverage"]


@pytest.mark.parametrize(
    ("distractor_text", "target_text", "terms"),
    (
        (
            "ablation table generic DistinctiveAnchor filler",
            "ablation DistinctiveAnchor SecondaryMarker ThirdMarker FourthMarker decisive evidence",
            (
                "in",
                "the",
                "ablation",
                "table",
                "what",
                "happened",
                "to",
                "generic",
                "distinctiveanchor",
                "secondarymarker",
                "thirdmarker",
                "fourthmarker",
            ),
        ),
        (
            "alpha beta distinctiveanchor secondarymarker thirdmarker fourthmarker",
            "mido midt midr midf midv mids distinctiveanchor secondarymarker thirdmarker fourthmarker",
            (
                "alpha",
                "beta",
                "mido",
                "midt",
                "midr",
                "midf",
                "midv",
                "mids",
                "distinctiveanchor",
                "secondarymarker",
                "thirdmarker",
                "fourthmarker",
            ),
        ),
    ),
)
def test_like_ranking_scores_first_eight_and_anchor_terms_before_limit(
    tmp_path,
    distractor_text,
    target_text,
    terms,
) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    for index in range(50):
        db.put_lexical_span(
            "native-test",
            span_id=f"distractor-{index:02d}",
            source_path=f"a/{index:02d}.md",
            source_id=f"raw:distractor-{index:02d}",
            source_role="raw",
            span_kind="table.row",
            heading_path=["Results"],
            start_line=1,
            end_line=1,
            text=distractor_text,
            metadata={},
        )
    db.put_lexical_span(
        "native-test",
        span_id="target",
        source_path="z/target.md",
        source_id="raw:target",
        source_role="raw",
        span_kind="doc.section",
        heading_path=["Results"],
        start_line=1,
        end_line=1,
        text=target_text,
        metadata={},
    )

    hits = db.query_lexical_spans(
        "native-test",
        " ".join(terms),
        limit=40,
        normalized_terms=terms,
    )

    assert hits[0]["span_id"] == "target"


def test_normalized_numeric_like_prioritizes_grouped_and_plain_values_before_limit(
    tmp_path,
    monkeypatch,
) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    for index in range(50):
        db.put_lexical_span(
            "native-test",
            span_id=f"distractor-{index:02d}",
            source_path=f"a/{index:02d}.md",
            source_id=f"raw:distractor-{index:02d}",
            source_role="raw",
            span_kind="table.row",
            heading_path=["Results"],
            start_line=1,
            end_line=1,
            text="exact filler",
            metadata={},
        )
    db.put_lexical_span(
        "native-test",
        span_id="target",
        source_path="z/target.md",
        source_id="raw:target",
        source_role="raw",
        span_kind="table.row",
        heading_path=["Results"],
        start_line=1,
        end_line=1,
        text=r"|D|=36{,}193",
        metadata={},
    )
    db.put_lexical_span(
        "native-test",
        span_id="plain-target",
        source_path="y/plain-target.md",
        source_id="raw:plain-target",
        source_role="raw",
        span_kind="table.row",
        heading_path=["Results"],
        start_line=1,
        end_line=1,
        text="|P|=237",
        metadata={},
    )
    for span_id, source_path, text in (
        ("comma-target", "zz/comma-target.md", "|D|=36,193"),
        ("ungrouped-target", "zz/ungrouped-target.md", "|D|=36193"),
    ):
        db.put_lexical_span(
            "native-test",
            span_id=span_id,
            source_path=source_path,
            source_id=f"raw:{span_id}",
            source_role="raw",
            span_kind="table.row",
            heading_path=["Results"],
            start_line=1,
            end_line=1,
            text=text,
            metadata={},
        )
    statements: list[str] = []
    original_connect = db._connect

    def traced_connect():
        connection = original_connect()
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(db, "_connect", traced_connect)

    hits = db.query_lexical_spans(
        "native-test",
        "exact 36,193",
        limit=40,
        normalized_terms=("exact", "36193"),
    )

    assert hits[0]["span_id"] == "target"
    assert {"target", "comma-target", "ungrouped-target"} <= {
        hit["span_id"] for hit in hits
    }
    primary = next(statement for statement in statements if "term_match_count" in statement)
    assert "REPLACE(" not in primary
    plain_hits = db.query_lexical_spans(
        "native-test",
        "exact prompt count 237",
        limit=40,
        normalized_terms=("exact", "prompt", "count", "237"),
    )
    assert plain_hits[0]["span_id"] == "plain-target"


def test_normalized_numeric_like_uses_source_path_relevance_before_limit(
    tmp_path,
) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    for index in range(50):
        db.put_lexical_span(
            "native-test",
            span_id=f"distractor-{index:02d}",
            source_path=f"notes/generic-{index:02d}.md",
            source_id=f"raw:distractor-{index:02d}",
            source_role="raw",
            span_kind="table.row",
            heading_path=["Results"],
            start_line=1,
            end_line=1,
            text="capacity |N|=10",
            metadata={},
        )
    db.put_lexical_span(
        "native-test",
        span_id="target",
        source_path="notes/project-orchid.md",
        source_id="raw:target",
        source_role="raw",
        span_kind="table.row",
        heading_path=["Results"],
        start_line=1,
        end_line=1,
        text="|N|=10",
        metadata={},
    )

    hits = db.query_lexical_spans(
        "native-test",
        "project orchid capacity 10",
        limit=40,
        normalized_terms=("project", "orchid", "capacity", "10"),
    )

    assert hits[0]["span_id"] == "target"


def test_structured_query_admits_path_match_before_text_only_limit(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    for index in range(50):
        db.put_lexical_span(
            "native-test",
            span_id=f"distractor-{index:02d}",
            source_path=f"notes/value-model-{index:02d}.md",
            source_id=f"raw:distractor-{index:02d}",
            source_role="raw",
            span_kind="table.row",
            heading_path=["Method"],
            start_line=1,
            end_line=1,
            text="formula row generic evidence",
            metadata={},
        )
    db.put_lexical_span(
        "native-test",
        span_id="target",
        source_path="notes/project-graphpo.md",
        source_id="raw:target",
        source_role="raw",
        span_kind="table.row",
        heading_path=["Method"],
        start_line=190,
        end_line=190,
        text=r"|V^\star(u)-V^\star(v)|\le\delta_\kappa,",
        metadata={},
    )

    hits = db.query_lexical_spans(
        "native-test",
        "graphpo value exact formula row",
        limit=40,
        normalized_terms=("graphpo", "value", "exact", "formula", "row"),
    )

    assert hits[0]["span_id"] == "target"


def test_plain_numeric_terms_do_not_enable_grouped_decimal_sql_scan(
    tmp_path,
    monkeypatch,
) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    db.put_lexical_span(
        "native-test",
        span_id="target",
        source_path="target.md",
        source_id="raw:target",
        source_role="raw",
        span_kind="table.row",
        heading_path=["Results"],
        start_line=1,
        end_line=1,
        text="410 tokens per millisecond",
        metadata={},
    )
    statements: list[str] = []
    original_connect = db._connect

    def traced_connect():
        connection = original_connect()
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(db, "_connect", traced_connect)

    db.query_lexical_spans(
        "native-test",
        "exact 410 to 480",
        limit=40,
        normalized_terms=("exact", "410", "to", "480"),
    )

    assert not any("REPLACE(" in statement for statement in statements)
    primary = next(
        statement for statement in statements if "term_match_count" in statement
    )
    assert primary.count("source_path LIKE") == 1


def test_vector_coverage_audit_reports_records_without_vectors(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    db.put_record(native_record("native-test", "entity", "doc:a", "Alpha", payload={"entity_name": "doc:a"}))
    db.put_record(native_record("native-test", "entity", "doc:b", "Beta", payload={"entity_name": "doc:b"}))
    db.put_vector("native-test", "entity", "doc:a", "doc:a:vector", [1.0, 0.0])

    audit = db.audit_vector_coverage("native-test")

    assert audit["ok"] is False
    assert audit["missing"]["entity"] == ["doc:b"]
    assert audit["counts"]["entity"] == {"records": 2, "vectors": 1, "missing": 1}
    with pytest.raises(ValueError, match="vector coverage"):
        db.mark_audited("native-test", {"chunks": 0, "entities": 2, "relationships": 0, "sections": 0}, require_vectors=True)

def test_vector_insert_rejects_hash_mismatch(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    db.put_record(native_record("native-test", "entity", "doc:a", "Alpha", payload={"entity_name": "doc:a"}))

    with pytest.raises(ValueError, match="vector_hash"):
        db.put_vector("native-test", "entity", "doc:a", "wrong-vector-hash", [1.0, 0.0])
