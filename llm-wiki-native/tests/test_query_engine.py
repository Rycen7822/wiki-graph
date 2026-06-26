import pytest

from llm_wiki_native.retrieval.query_engine import NativeQueryEngine
from llm_wiki_native.storage.sqlite_workspace import NativeRecord, SQLiteWorkspace


def _record(workspace_id: str, record_type: str, record_id: str, text: str) -> NativeRecord:
    return NativeRecord(
        workspace_id=workspace_id,
        record_type=record_type,
        record_id=record_id,
        vector_text=text,
        content_hash=f"{record_id}:content",
        metadata_hash=f"{record_id}:metadata",
        vector_hash=f"{record_id}:vector",
        source_path=f"{record_id}.md",
        source_id=record_id,
        payload={"title": text},
    )


def test_data_only_query_engine_returns_ranked_hits_with_trace(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    db.put_record(_record("native-test", "entity", "doc:a", "Alpha"))
    db.put_record(_record("native-test", "entity", "doc:b", "Beta"))
    db.put_vector("native-test", "entity", "doc:a", "doc:a:vector", [1.0, 0.0])
    db.put_vector("native-test", "entity", "doc:b", "doc:b:vector", [0.0, 1.0])
    db.put_edge("native-test", "relationship", "doc:a", "tag:x", 0.8, {"kind": "related"})
    db.mark_audited("native-test", {"chunks": 0, "entities": 2, "relationships": 0, "sections": 0}, require_vectors=True)
    engine = NativeQueryEngine(db)

    result = engine.query("native-test", "alpha query", [1.0, 0.0], mode="mix", top_k=1, record_types=("entity",), neighbor_limit=1)

    assert result["hits"][0]["record_id"] == "doc:a"
    assert result["hits"][0]["record"]["vector_text"] == "Alpha"
    assert result["hits"][0]["neighbors"][0]["neighbor_id"] == "tag:x"
    assert result["trace"]["mode"] == "mix"
    assert result["trace"]["query"] == "alpha query"
    assert result["trace"]["vector_hit_count"] == 1


def test_data_only_query_engine_rejects_building_workspace(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    db.put_record(_record("native-test", "entity", "doc:a", "Alpha"))
    db.put_vector("native-test", "entity", "doc:a", "doc:a:vector", [1.0, 0.0])
    engine = NativeQueryEngine(db)

    with pytest.raises(ValueError, match="audited or active"):
        engine.query("native-test", "alpha query", [1.0, 0.0], mode="mix", record_types=("entity",))


def test_data_only_query_engine_rejects_unknown_mode(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    engine = NativeQueryEngine(db)

    with pytest.raises(ValueError, match="mode"):
        engine.query("native-test", "alpha query", [1.0], mode="unsupported")


def test_data_only_query_engine_rejects_unimplemented_supported_mode(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    engine = NativeQueryEngine(db)

    with pytest.raises(NotImplementedError, match="not implemented"):
        engine.query("native-test", "alpha query", [1.0], mode="local")


def test_data_only_query_engine_rejects_unknown_record_type(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    engine = NativeQueryEngine(db)

    with pytest.raises(ValueError, match="record_type"):
        engine.query("native-test", "alpha query", [1.0], mode="mix", record_types=("unknown",))
