import json
from pathlib import Path

from llm_wiki_native.artifacts import (
    load_custom_kg_manifest,
    load_raw_sections,
    load_section_similarity_edges,
)
from llm_wiki_native.manifest import manifest_summary


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_artifact_loader_reads_existing_manifest_and_sidecars_without_writes(tmp_path: Path) -> None:
    state = tmp_path / "state"
    manifest = {
        "schema_version": 1,
        "summary": {"chunks": 1, "entities": 1, "relationships": 1},
        "chunks": {"chunk-a": {"chunk_id": "chunk-a", "content": "Doc A", "file_path": "a.md"}},
        "entities": {"doc:a": {"entity_name": "doc:a", "content": "doc:a\nDoc A", "file_path": "a.md"}},
        "relationships": {
            "doc:a<SEP>tag:x": {
                "src_id": "doc:a",
                "tgt_id": "tag:x",
                "content": "RELATED\tdoc:a\ntag:x\nDoc A related to tag X",
                "file_path": "a.md",
            }
        },
    }
    edge = {"edge_id": "edge-1", "src_id": "raw:a", "tgt_id": "raw:b", "cosine": 0.9, "source_path": "raw/a.md"}
    section = {"section_id": "raw:a#method", "section_kind": "method", "source_path": "raw/a.md"}
    _write_json(state / "custom_kg_manifest.json", manifest)
    _write_jsonl(state / "section_similarity_edges.jsonl", [edge])
    _write_jsonl(state / "raw_sections.jsonl", [section])
    before = sorted(path.relative_to(state).as_posix() for path in state.rglob("*"))

    loaded_manifest = load_custom_kg_manifest(state)
    loaded_edges = load_section_similarity_edges(state)
    loaded_sections = load_raw_sections(state)

    assert manifest_summary(loaded_manifest) == {"chunks": 1, "entities": 1, "relationships": 1}
    assert loaded_edges == [edge]
    assert loaded_sections == [section]
    after = sorted(path.relative_to(state).as_posix() for path in state.rglob("*"))
    assert after == before
