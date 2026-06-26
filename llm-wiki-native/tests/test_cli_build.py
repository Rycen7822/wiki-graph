import json
import sqlite3
from pathlib import Path

import numpy as np

from llm_wiki_native.cli import build_workspace_from_state, main
from llm_wiki_native.storage.sqlite_workspace import SQLiteWorkspace


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_vector_cache(path: Path, vectors: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE vector_cache(vector_hash TEXT PRIMARY KEY, embedding_dim INTEGER NOT NULL, vector_blob BLOB NOT NULL)")
        for vector_hash, vector in vectors.items():
            arr = np.asarray(vector, dtype=np.float32)
            conn.execute(
                "INSERT INTO vector_cache(vector_hash, embedding_dim, vector_blob) VALUES(?, ?, ?)",
                (vector_hash, int(arr.size), arr.tobytes()),
            )


def _sample_state(tmp_path: Path, *, with_vectors: bool = True) -> Path:
    state = tmp_path / "state"
    manifest = {
        "chunks": {"chunk-a": {"content": "Alpha", "content_hash": "chunk-hash", "source_id": "doc:a", "file_path": "a.md"}},
        "entities": {"doc:a": {"content": "doc:a\nAlpha", "vector_hash": "entity-vector", "metadata_hash": "entity-meta", "source_logical_id": "doc:a", "file_path": "a.md"}},
        "relationships": {
            "doc:a<SEP>tag:x": {
                "src_id": "doc:a",
                "tgt_id": "tag:x",
                "content": "RELATED\tdoc:a\ntag:x\nAlpha tag",
                "vector_hash": "rel-vector",
                "metadata_hash": "rel-meta",
                "weight": 0.6,
                "source_logical_id": "doc:a",
                "file_path": "a.md",
            }
        },
    }
    _write_json(state / "custom_kg_manifest.json", manifest)
    _write_jsonl(state / "section_similarity_edges.jsonl", [{"src_id": "doc:a", "tgt_id": "doc:b", "cosine": 0.9}])
    _write_jsonl(
        state / "raw_sections.jsonl",
        [
            {
                "section_id": "raw_section:doc-a:method",
                "source_id": "doc:a",
                "source_path": "a.md",
                "paper_title": "Alpha",
                "section_kind": "method",
                "section_title": "Method",
                "canonical_section_title": "method",
                "content": "Method body",
            }
        ],
    )
    if with_vectors:
        _write_vector_cache(
            state / "vector_cache.sqlite",
            {
                "chunk-hash": [1.0, 0.0],
                "entity-vector": [0.9, 0.1],
                "rel-vector": [0.0, 1.0],
            },
        )
        _write_jsonl(
            state / "section_embeddings.jsonl",
            [{"section_id": "raw_section:doc-a:method", "text_hash": "section-vector", "embedding": [0.5, 0.5]}],
        )
    return state


def test_build_workspace_from_state_materializes_manifest_and_edges_without_activation(tmp_path) -> None:
    state = _sample_state(tmp_path)
    db_path = tmp_path / "native.sqlite"

    report = build_workspace_from_state(state, db_path, "native-test")

    db = SQLiteWorkspace(db_path)
    assert report["audit"]["ok"] is True
    assert report["status"] == "audited"
    assert report["counts"] == {"chunks": 1, "entities": 1, "relationships": 1, "sections": 1}
    assert report["edge_count"] == 2
    assert report["vector_audit"]["ok"] is True
    assert db.get_workspace_status("native-test") == "audited"
    assert db.nearest_vectors("native-test", "chunk", [1.0, 0.0], top_k=1)[0]["record_id"] == "chunk-a"
    assert db.nearest_vectors("native-test", "section", [0.5, 0.5], top_k=1)[0]["record_id"] == "raw_section:doc-a:method"
    section = db.get_record("native-test", "section", "raw_section:doc-a:method")
    assert section["source_path"] == "a.md"
    assert section["source_id"] == "doc:a"
    assert section["payload"]["section_kind"] == "method"
    assert [item["neighbor_id"] for item in db.neighbors("native-test", "doc:a")] == ["doc:b", "tag:x"]


def test_build_workspace_from_state_fails_closed_when_vectors_are_missing(tmp_path) -> None:
    state = _sample_state(tmp_path, with_vectors=False)

    try:
        build_workspace_from_state(state, tmp_path / "native.sqlite", "native-test")
    except ValueError as exc:
        assert "vector" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("build_workspace_from_state should require complete vector coverage")


def test_cli_main_builds_workspace_and_prints_json_report(tmp_path, capsys) -> None:
    state = _sample_state(tmp_path)
    db_path = tmp_path / "native.sqlite"

    assert main(["build-workspace", "--state-dir", str(state), "--db", str(db_path), "--workspace-id", "native-test"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["workspace_id"] == "native-test"
    assert report["audit"]["ok"] is True
