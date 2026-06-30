import pytest

from llm_wiki_native.manifest import materialize_manifest, manifest_summary
from llm_wiki_native.storage.sqlite_workspace import SQLiteWorkspace


def test_materialize_manifest_preserves_counts_sources_and_hashes(tmp_path) -> None:
    manifest = {
        "chunks": {
            "chunk-a": {
                "chunk_id": "chunk-a",
                "content": "Doc A",
                "content_hash": "chunk-content-hash",
                "source_id": "doc:a",
                "file_path": "a.md",
            }
        },
        "entities": {
            "doc:a": {
                "entity_name": "doc:a",
                "content": "doc:a\nDoc A",
                "vector_hash": "entity-vector-hash",
                "metadata_hash": "entity-metadata-hash",
                "source_logical_id": "doc:a",
                "file_path": "a.md",
            }
        },
        "relationships": {
            "doc:a<SEP>tag:x": {
                "src_id": "doc:a",
                "tgt_id": "tag:x",
                "content": "RELATED\tdoc:a\ntag:x\nDoc A related to tag X",
                "vector_hash": "rel-vector-hash",
                "metadata_hash": "rel-metadata-hash",
                "source_logical_id": "doc:a",
                "file_path": "a.md",
            }
        },
    }
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")

    result = materialize_manifest(db, "native-test", manifest)
    audit = db.audit_counts("native-test", {"chunks": 1, "entities": 1, "relationships": 1, "sections": 0})

    assert result == {"chunks": 1, "entities": 1, "relationships": 1}
    assert manifest_summary(manifest) == {"chunks": 1, "entities": 1, "relationships": 1}
    assert audit["ok"] is True
    stored = db.get_record("native-test", "chunk", "chunk-a")
    assert stored["source_path"] == "a.md"
    assert stored["source_id"] == "doc:a"
    assert stored["content_hash"] == "chunk-content-hash"
    assert stored["vector_hash"] == "chunk-content-hash"


def test_materialize_manifest_rejects_records_without_hashes(tmp_path) -> None:
    manifest = {"chunks": {"chunk-a": {"content": "Doc A", "source_id": "doc:a", "file_path": "a.md"}}, "entities": {}, "relationships": {}}
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")

    with pytest.raises(ValueError, match="chunk-a"):
        materialize_manifest(db, "native-test", manifest)
