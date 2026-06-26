import pytest

from llm_wiki_native.storage.sqlite_workspace import NativeRecord, SQLiteWorkspace


def _record(workspace_id: str, record_id: str, text: str) -> NativeRecord:
    return NativeRecord(
        workspace_id=workspace_id,
        record_type="entity",
        record_id=record_id,
        vector_text=text,
        content_hash=f"{record_id}:content",
        metadata_hash=f"{record_id}:metadata",
        vector_hash=f"{record_id}:vector",
        source_path=f"{record_id}.md",
        source_id=record_id,
        payload={"entity_name": record_id},
    )


def test_exact_vector_index_returns_stable_cosine_top_k(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    db.put_record(_record("native-test", "doc:a", "Alpha"))
    db.put_record(_record("native-test", "doc:b", "Beta"))
    db.put_record(_record("native-test", "doc:c", "Gamma"))
    db.put_vector("native-test", "entity", "doc:a", "doc:a:vector", [1.0, 0.0])
    db.put_vector("native-test", "entity", "doc:b", "doc:b:vector", [0.0, 1.0])
    db.put_vector("native-test", "entity", "doc:c", "doc:c:vector", [1.0, 0.0])

    results = db.nearest_vectors("native-test", "entity", [1.0, 0.0], top_k=3)

    assert [(item["record_id"], round(item["score"], 6)) for item in results] == [
        ("doc:a", 1.0),
        ("doc:c", 1.0),
        ("doc:b", 0.0),
    ]


def test_exact_vector_index_rejects_dimension_mismatch(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    db.put_record(_record("native-test", "doc:a", "Alpha"))
    db.put_vector("native-test", "entity", "doc:a", "doc:a:vector", [1.0, 0.0])

    with pytest.raises(ValueError, match="dimension"):
        db.nearest_vectors("native-test", "entity", [1.0], top_k=1)


def test_vector_coverage_audit_reports_records_without_vectors(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    db.put_record(_record("native-test", "doc:a", "Alpha"))
    db.put_record(_record("native-test", "doc:b", "Beta"))
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
    db.put_record(_record("native-test", "doc:a", "Alpha"))

    with pytest.raises(ValueError, match="vector_hash"):
        db.put_vector("native-test", "entity", "doc:a", "wrong-vector-hash", [1.0, 0.0])
