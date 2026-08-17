from pathlib import Path

from support import write_json, write_jsonl

from llm_wiki_native.artifacts import (
    load_custom_kg_manifest,
    load_raw_sections,
    load_section_similarity_edges,
)
from llm_wiki_native.manifest import manifest_summary
from ops.wiki_native_jsonl import jsonl_read, jsonl_write


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
    write_json(state / "custom_kg_manifest.json", manifest)
    write_jsonl(state / "section_similarity_edges.jsonl", [edge])
    write_jsonl(state / "raw_sections.jsonl", [section])
    before = sorted(path.relative_to(state).as_posix() for path in state.rglob("*"))

    loaded_manifest = load_custom_kg_manifest(state)
    loaded_edges = load_section_similarity_edges(state)
    loaded_sections = load_raw_sections(state)

    assert manifest_summary(loaded_manifest) == {"chunks": 1, "entities": 1, "relationships": 1}
    assert loaded_edges == [edge]
    assert loaded_sections == [section]
    after = sorted(path.relative_to(state).as_posix() for path in state.rglob("*"))
    assert after == before


def test_jsonl_read_streams_rows_in_order_and_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n  {"b": 2}  \n', encoding="utf-8")

    assert jsonl_read(path) == [{"a": 1}, {"b": 2}]

    roundtrip = tmp_path / "roundtrip.jsonl"
    assert jsonl_write(roundtrip, [{"b": 2}, {"a": 1}]) == 2
    assert jsonl_read(roundtrip) == [{"b": 2}, {"a": 1}]
