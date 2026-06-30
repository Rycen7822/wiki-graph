import pytest

from llm_wiki_native.storage.sqlite_workspace import SQLiteWorkspace
from support import native_record


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
