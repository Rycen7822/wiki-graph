import json
from pathlib import Path

from llm_wiki_native.cli import build_workspace_from_state, main
from llm_wiki_native.storage.sqlite_workspace import SQLiteWorkspace


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _sample_state(tmp_path: Path) -> Path:
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
    _write_jsonl(state / "raw_sections.jsonl", [])
    return state


def test_build_workspace_from_state_materializes_manifest_and_edges_without_activation(tmp_path) -> None:
    state = _sample_state(tmp_path)
    db_path = tmp_path / "native.sqlite"

    report = build_workspace_from_state(state, db_path, "native-test")

    db = SQLiteWorkspace(db_path)
    assert report["audit"]["ok"] is True
    assert report["status"] == "audited"
    assert report["counts"] == {"chunks": 1, "entities": 1, "relationships": 1}
    assert report["edge_count"] == 2
    assert db.get_workspace_status("native-test") == "audited"
    assert [item["neighbor_id"] for item in db.neighbors("native-test", "doc:a")] == ["doc:b", "tag:x"]


def test_cli_main_builds_workspace_and_prints_json_report(tmp_path, capsys) -> None:
    state = _sample_state(tmp_path)
    db_path = tmp_path / "native.sqlite"

    assert main(["build-workspace", "--state-dir", str(state), "--db", str(db_path), "--workspace-id", "native-test"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["workspace_id"] == "native-test"
    assert report["audit"]["ok"] is True
