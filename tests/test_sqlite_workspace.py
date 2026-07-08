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

    table_hits = db.query_lexical_spans("native-test", "CalibrationWinner", limit=5)
    map_hits = db.query_lexical_spans("native-test", "MapOnlyNeedle", limit=5, source_roles=("meta_map",))

    assert db.count_lexical_spans("native-test") == 3
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
