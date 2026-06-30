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

    assert db.get_workspace_status("native-test") == "building"
    assert db.audit_counts("native-test", {"chunks": 0, "entities": 0, "relationships": 0, "sections": 0})["ok"] is True


def test_workspace_audit_reports_record_count_mismatch(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    db.put_record(_workspace_record("native-test"))

    audit = db.audit_counts("native-test", {"chunks": 2, "entities": 0, "relationships": 0, "sections": 0})

    assert audit["ok"] is False
    assert audit["counts"]["chunks"] == 1
    assert "chunks" in audit["issues"][0]


def test_workspace_no_longer_exposes_serving_activation_state(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    db.put_record(_workspace_record("native-test"))
    db.mark_audited("native-test", {"chunks": 1, "entities": 0, "relationships": 0, "sections": 0})

    assert db.get_workspace_status("native-test") == "audited"
    assert not hasattr(SQLiteWorkspace, "activate_workspace")


def test_workspace_operations_reject_unknown_workspace(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")

    with pytest.raises(KeyError, match="missing-workspace"):
        db.audit_counts("missing-workspace", {"chunks": 0, "entities": 0, "relationships": 0, "sections": 0})
    with pytest.raises(KeyError, match="missing-workspace"):
        db.put_record(_workspace_record("missing-workspace"))
