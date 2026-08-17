import sqlite3
from dataclasses import replace

import pytest

from llm_wiki_native.storage.sqlite_workspace import SQLiteWorkspace
from support import dump_workspace_tables, native_record, put_span, traced_sqlite_connect


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


def _put_distractors(db, text: str, path_pattern: str, *, heading_path: list[str] | None = None) -> None:
    for index in range(50):
        put_span(
            db,
            span_id=f"distractor-{index:02d}",
            source_path=path_pattern.format(index=index),
            source_role="raw",
            span_kind="table.row",
            text=text,
            heading_path=heading_path,
        )


def _bulk_fixture():
    records = [native_record("native-test", "chunk", f"chunk-{i}", text=f"Chunk {i}") for i in range(3)]
    records.append(native_record("native-test", "entity", "entity-0", text="Entity 0"))
    vectors = [("chunk", f"chunk-{i}", f"chunk-{i}:vector", [float(i), 1.0]) for i in range(3)]
    vectors.append(("entity", "entity-0", "entity-0:vector", [9.0, 9.0]))
    edges = [
        ("relationship", "entity-0", "chunk-0", 0.9, {"kind": "mentions"}),
        ("section_similarity", "chunk-0", "chunk-1", 0.5, {"cosine": 0.5}),
    ]
    spans = [
        {
            "span_id": f"span-{i}",
            "source_path": f"f{i}.md",
            "source_id": f"s{i}",
            "source_role": "raw_note",
            "span_kind": "paragraph",
            "heading_path": ["H", f"S{i}"],
            "start_line": 1,
            "end_line": 2,
            "text": f"text {i}",
            "metadata": {"k": i},
        }
        for i in range(2)
    ]
    return records, vectors, edges, spans


def test_bulk_puts_match_per_item_writes_exactly(tmp_path) -> None:
    records, vectors, edges, spans = _bulk_fixture()

    per_item_path = tmp_path / "per-item.sqlite"
    db_a = SQLiteWorkspace(per_item_path)
    db_a.create_workspace("native-test", "manifest-hash")
    for record in records:
        db_a.put_record(record)
    for record_type, record_id, vector_hash, vector in vectors:
        db_a.put_vector("native-test", record_type, record_id, vector_hash, vector)
    for edge_type, src_id, tgt_id, weight, payload in edges:
        db_a.put_edge("native-test", edge_type, src_id, tgt_id, weight, payload)
    for span in spans:
        db_a.put_lexical_span("native-test", **span)

    bulk_path = tmp_path / "bulk.sqlite"
    db_b = SQLiteWorkspace(bulk_path)
    db_b.create_workspace("native-test", "manifest-hash")
    assert db_b.put_records(records) == len(records)
    assert db_b.put_vectors("native-test", vectors) == len(vectors)
    assert db_b.put_edges("native-test", edges) == len(edges)
    assert db_b.put_lexical_spans("native-test", spans) == len(spans)

    assert dump_workspace_tables(bulk_path) == dump_workspace_tables(per_item_path)


def test_bulk_puts_preserve_validation_errors(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    record = _workspace_record("native-test")
    db.put_record(record)

    with pytest.raises(ValueError, match="unknown record_type"):
        db.put_records([replace(record, record_type="bogus")])
    with pytest.raises(ValueError, match="vector_hash mismatch"):
        db.put_vectors("native-test", [("chunk", record.record_id, "wrong-hash", [1.0])])
    with pytest.raises(ValueError, match="edge_type, src_id, and tgt_id are required"):
        db.put_edges("native-test", [("relationship", "", "chunk-a", 1.0, {})])
    with pytest.raises(ValueError, match="span_id is required"):
        db.put_lexical_spans(
            "native-test",
            [{"span_id": " ", "source_path": "a.md", "source_id": "s", "source_role": "raw_note", "span_kind": "paragraph", "text": "t"}],
        )
    with pytest.raises(KeyError):
        db.put_records([native_record("missing-workspace")])

    assert db.put_records([]) == 0
    assert db.put_vectors("native-test", []) == 0
    assert db.put_edges("native-test", []) == 0
    assert db.put_lexical_spans("native-test", []) == 0


def test_delete_primitives_remove_rows_and_companions(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    records, vectors, edges, spans = _bulk_fixture()
    db.put_records(records)
    db.put_vectors("native-test", vectors)
    db.put_edges("native-test", edges)
    db.put_lexical_spans("native-test", spans)

    assert db.delete_records("native-test", [("chunk", "chunk-0"), ("chunk", "missing")]) == 1
    assert db.delete_edges("native-test", [("relationship", "entity-0", "chunk-0")]) == 1
    assert db.delete_lexical_spans("native-test", ["span-0"]) == 1
    assert db.delete_records("native-test", []) == 0
    assert db.delete_edges("native-test", []) == 0
    assert db.delete_lexical_spans("native-test", []) == 0
    assert db.delete_vectors("native-test", []) == 0

    dump = dump_workspace_tables(tmp_path / "native.sqlite")
    for table, predicate in (
        ("record", lambda row: row[1] == "chunk" and row[2] == "chunk-0"),
        ("vector", lambda row: row[1] == "chunk" and row[2] == "chunk-0"),
        ("edge", lambda row: row[1] == "relationship" and row[2] == "entity-0"),
        ("lexical_span", lambda row: row[1] == "span-0"),
        ("lexical_span_fts", lambda row: row[1] == "span-0"),
    ):
        assert [row for row in dump[table] if predicate(row)] == []
    assert db.count_records("native-test") == {"chunks": 2, "entities": 1, "relationships": 0, "sections": 0}


def test_delete_primitives_on_read_only_leave_workspace_untouched(tmp_path) -> None:
    db_path = tmp_path / "native.sqlite"
    writable = SQLiteWorkspace(db_path)
    writable.create_workspace("native-test", "manifest-hash")
    records, _, _, _ = _bulk_fixture()
    writable.put_records(records)
    before = dump_workspace_tables(db_path)

    read_only = SQLiteWorkspace.open_existing(db_path, read_only=True)
    with pytest.raises(sqlite3.OperationalError):
        read_only.delete_records("native-test", [("chunk", "chunk-0")])

    assert dump_workspace_tables(db_path) == before
    assert read_only.get_record("native-test", "chunk", "chunk-0")["record_id"] == "chunk-0"


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
    assert read_only.get_record("native-test", "chunk", "chunk-a")["vector_text"] == "Doc A"
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

    put_span(
        db,
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
    put_span(
        db,
        span_id="span:table-row",
        source_path="concepts/alpha.md",
        source_id="compiled:concept:alpha",
        source_role="compiled",
        span_kind="table.row",
        heading_path=["Alpha", "Results"],
        start_line=10,
        text="| Method | CalibrationWinner | strong table evidence |",
        metadata={"columns": ["Method", "Result"]},
    )
    put_span(
        db,
        span_id="span:map-row",
        source_path="_meta/raw-clip-map.md",
        source_id="meta:raw-clip-map",
        source_role="meta_map",
        span_kind="map.row",
        heading_path=["Raw Clip Map"],
        start_line=4,
        text="- raw/clip/2601/26010101_Foo-Paper.md :: MapOnlyNeedle",
        metadata={"map": "raw-clip"},
    )
    put_span(
        db,
        span_id="span:late-term",
        source_path="concepts/late.md",
        source_id="compiled:concept:late",
        source_role="compiled",
        span_kind="doc.section",
        heading_path=["Late"],
        text="LateIdentifier appears only after the legacy recall cutoff",
    )
    put_span(
        db,
        span_id="span:low-coverage",
        source_path="a-first.md",
        source_id="source:low",
        source_role="raw",
        span_kind="table.row",
        heading_path=["Low"],
        text="alpha only",
    )
    put_span(
        db,
        span_id="span:high-coverage",
        source_path="z-last.md",
        source_id="source:high",
        source_role="raw",
        span_kind="table.row",
        heading_path=["High"],
        text="alpha needle",
    )

    table_hits = db.query_lexical_spans("native-test", "CalibrationWinner", limit=5)
    map_hits = db.query_lexical_spans("native-test", "MapOnlyNeedle", limit=5, source_roles=("meta_map",))

    assert db.count_lexical_spans("native-test") == 6
    assert table_hits[0]["span_id"] == "span:table-row"
    assert table_hits[0]["span_kind"] == "table.row"
    assert table_hits[0]["source_path"] == "concepts/alpha.md"
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
    _put_distractors(db, distractor_text, "a/{index:02d}.md")
    put_span(
        db,
        span_id="target",
        source_path="z/target.md",
        source_role="raw",
        span_kind="doc.section",
        text=target_text,
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
    _put_distractors(db, "exact filler", "a/{index:02d}.md")
    put_span(db, span_id="target", source_path="z/target.md", source_role="raw", span_kind="table.row", text=r"|D|=36{,}193")
    put_span(db, span_id="plain-target", source_path="y/plain-target.md", source_role="raw", span_kind="table.row", text="|P|=237")
    for span_id, source_path, text in (
        ("comma-target", "zz/comma-target.md", "|D|=36,193"),
        ("ungrouped-target", "zz/ungrouped-target.md", "|D|=36193"),
    ):
        put_span(db, span_id=span_id, source_path=source_path, source_role="raw", span_kind="table.row", text=text)
    statements: list[str] = []
    monkeypatch.setattr(db, "_connect", traced_sqlite_connect(db, statements))

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


@pytest.mark.parametrize(
    ("distractor_path", "distractor_text", "heading", "target_path", "target_text", "start_line", "query", "terms"),
    [
        (
            "notes/generic-{index:02d}.md",
            "capacity |N|=10",
            ["Results"],
            "notes/project-orchid.md",
            "|N|=10",
            1,
            "project orchid capacity 10",
            ("project", "orchid", "capacity", "10"),
        ),
        (
            "notes/value-model-{index:02d}.md",
            "formula row generic evidence",
            ["Method"],
            "notes/project-graphpo.md",
            r"|V^\star(u)-V^\star(v)|\le\delta_\kappa,",
            190,
            "graphpo value exact formula row",
            ("graphpo", "value", "exact", "formula", "row"),
        ),
    ],
)
def test_normalized_query_ranks_path_match_before_text_only_limit(
    tmp_path,
    distractor_path,
    distractor_text,
    heading,
    target_path,
    target_text,
    start_line,
    query,
    terms,
) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    _put_distractors(db, distractor_text, distractor_path, heading_path=heading)
    put_span(
        db,
        span_id="target",
        source_path=target_path,
        source_role="raw",
        span_kind="table.row",
        heading_path=heading,
        start_line=start_line,
        text=target_text,
    )

    hits = db.query_lexical_spans(
        "native-test",
        query,
        limit=40,
        normalized_terms=terms,
    )

    assert hits[0]["span_id"] == "target"


def test_plain_numeric_terms_do_not_enable_grouped_decimal_sql_scan(
    tmp_path,
    monkeypatch,
) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    put_span(db, span_id="target", source_path="target.md", source_role="raw", span_kind="table.row", text="410 tokens per millisecond")
    statements: list[str] = []
    monkeypatch.setattr(db, "_connect", traced_sqlite_connect(db, statements))

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
