import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from support import sample_wiki, write  # noqa: E402
from ops.wiki_native_artifacts import build_seed_edges  # noqa: E402
from ops.wiki_native_custom_kg_payload import build_custom_kg_payload  # noqa: E402
from ops.wiki_native_raw_section_extract import extract_raw_sections  # noqa: E402
from ops.wiki_native_state import ensure_state_dirs  # noqa: E402


def test_custom_kg_payload_reuses_external_seed_edges(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    build_seed_edges(root, state)
    payload, summary = build_custom_kg_payload(root, state, limit_docs=3, limit_edges=2)
    assert summary["chunks"] == 3
    assert summary["relationships"] <= 2
    assert all(chunk["file_path"].endswith(".md") for chunk in payload["chunks"])
    entity_names = {entity["entity_name"] for entity in payload["entities"]}
    assert "compiled:concept:foo" in entity_names
    assert payload["relationships"]
    assert {"src_id", "tgt_id", "description", "keywords", "source_id"} <= set(payload["relationships"][0])


def test_custom_kg_payload_includes_method_atoms_without_wiki_pollution(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    ensure_state_dirs(state)
    write(
        state / "method_atom_docs" / "method-demo.md",
        "\n".join(
            [
                "[LLM_WIKI_METHOD_ATOM]",
                "atom_id: method:demo:001",
                "source_path: raw/clip/2601/26010101_Foo-Paper.md",
                "[/LLM_WIKI_METHOD_ATOM]",
                "",
                "# MethodAtom Demo",
            ]
        ),
    )
    build_seed_edges(root, state)
    payload, summary = build_custom_kg_payload(root, state, limit_docs=2, limit_edges=0)
    assert summary["chunks"] == 3
    entity_by_name = {entity["entity_name"]: entity for entity in payload["entities"]}
    assert entity_by_name["method:demo:001"]["entity_type"] == "LLM_WIKI_METHOD_ATOM"
    assert any(rel["keywords"] == "METHOD_ATOM_FROM" for rel in payload["relationships"])
    assert not (root / ".llm-wiki").exists()


def test_custom_kg_payload_includes_raw_section_chunks_and_relationships(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    extract_raw_sections(root, state)
    build_seed_edges(root, state)
    payload, summary = build_custom_kg_payload(root, state, limit_docs=2, limit_edges=0)
    assert summary["raw_section_chunks"] == 4
    assert summary["chunks"] == 6
    section_chunks = [chunk for chunk in payload["chunks"] if chunk["source_id"].startswith("raw_section:")]
    assert {chunk["source_id"].rsplit(":", 1)[-1] for chunk in section_chunks} == {"methodology", "future", "limitations", "questions"}
    assert any("section_kind: limitations" in chunk["content"] for chunk in section_chunks)
    assert any("section_kind: methodology" in chunk["content"] for chunk in section_chunks)
    entity_by_name = {entity["entity_name"]: entity for entity in payload["entities"]}
    assert entity_by_name["raw_section:26010101_Foo-Paper:questions"]["entity_type"] == "LLM_WIKI_RAW_SECTION"
    assert any(rel["keywords"] == "RAW_SECTION_OF" and rel["tgt_id"] == "raw_clip:26010101_Foo-Paper" for rel in payload["relationships"])


def test_custom_kg_manifest_resolves_sources_and_preserves_typed_relationships(tmp_path: Path) -> None:
    from ops.custom_kg_incremental import build_custom_kg_manifest, entity_record_id, relationship_record_id, stable_hash

    payload = {
        "chunks": [
            {"content": "Doc A content", "source_id": "doc:a", "file_path": "a.md", "chunk_order_index": 0},
        ],
        "entities": [
            {"entity_name": "doc:a", "entity_type": "DOC", "description": "Doc A", "source_id": "doc:a", "file_path": "a.md"},
            {"entity_name": "topic:x", "entity_type": "TOPIC", "description": "Topic", "source_id": "doc:a", "file_path": "a.md"},
        ],
        "relationships": [
            {"src_id": "topic:x", "tgt_id": "doc:a", "description": "old", "keywords": "OLD", "source_id": "doc:a", "weight": 0.1, "file_path": "old.md"},
            {"src_id": "doc:a", "tgt_id": "topic:x", "description": "new", "keywords": "NEW", "source_id": "doc:a", "weight": 0.9, "file_path": "new.md"},
        ],
    }

    manifest = build_custom_kg_manifest(payload, native_manifest_tool_version="native-test", embedding_model="test-embed", embedding_dim=3)

    metadata = manifest["metadata"]
    assert metadata["native_manifest_tool_version"] == "native-test"
    assert metadata["canonical_id_algorithm"] == "llm-wiki-canonical-id:v1+native-custom-kg:v1"
    assert metadata["relationship_vector_content_algorithm"] == "llm-wiki-typed-directed-relationship:v1"

    chunk_id = next(iter(manifest["chunks"]))
    chunk = manifest["chunks"][chunk_id]
    assert manifest["source_to_chunk"]["doc:a"] == chunk_id
    assert chunk["record_type"] == "chunk"
    assert chunk["record_id"] == chunk_id
    assert chunk["canonical_id"] == chunk_id
    assert chunk["vector_text_hash"] == stable_hash(chunk["content"])
    assert manifest["entities"]["topic:x"]["source_chunk_id"] == chunk_id
    topic = manifest["entities"]["topic:x"]
    assert topic["record_type"] == "entity"
    assert topic["record_id"] == entity_record_id("topic:x")
    assert topic["canonical_id"] == "topic:x"
    assert topic["vector_text_hash"] == stable_hash(topic["content"])
    assert "UNKNOWN" not in json.dumps(manifest, ensure_ascii=False)
    assert len(manifest["relationships"]) == 2
    rels_by_keyword = {record["keywords"]: record for record in manifest["relationships"].values()}
    assert set(rels_by_keyword) == {"OLD", "NEW"}
    old_rel = rels_by_keyword["OLD"]
    new_rel = rels_by_keyword["NEW"]
    assert old_rel["description"] == "old"
    assert old_rel["src_id"] == "topic:x"
    assert old_rel["tgt_id"] == "doc:a"
    assert old_rel["record_id"] == relationship_record_id("topic:x", "doc:a", "OLD")
    assert new_rel["description"] == "new"
    assert new_rel["src_id"] == "doc:a"
    assert new_rel["tgt_id"] == "topic:x"
    assert new_rel["source_chunk_id"] == chunk_id
    assert new_rel["record_id"] == relationship_record_id("doc:a", "topic:x", "NEW")
    assert new_rel["record_type"] == "relationship"
    assert new_rel["canonical_id"] == new_rel["chunk_key"]
    assert "vdb_id" not in json.dumps(manifest, ensure_ascii=False)
    assert new_rel["vector_text_hash"] == stable_hash(new_rel["content"])


def test_custom_kg_manifest_matches_native_sanitized_chunk_ids_and_basenames() -> None:
    from ops import custom_kg_incremental
    from ops.custom_kg_incremental import build_custom_kg_manifest, compute_mdhash_id, native_manifest_sanitize_text

    raw_content = "  A &amp; B\x1f  "
    payload = {
        "chunks": [
            {"content": raw_content, "source_id": "doc:sanitized", "file_path": "nested/doc.[native-iet].md", "chunk_order_index": 0},
        ],
        "entities": [
            {"entity_name": "doc:sanitized", "entity_type": "DOC", "description": "Doc", "source_id": "doc:sanitized", "file_path": "nested/doc.[native-iet].md"},
        ],
        "relationships": [],
    }

    manifest = build_custom_kg_manifest(payload, native_manifest_tool_version="1.5.0", embedding_model="test-embed", embedding_dim=3)
    sanitized = native_manifest_sanitize_text(raw_content)
    chunk_id = compute_mdhash_id(sanitized, prefix="chunk-")

    assert sanitized == "A & B"
    assert custom_kg_incremental.native_manifest_normalize_file_path("nested/doc.[native-iet].md") == "doc.md"
    assert set(manifest["chunks"]) == {chunk_id}
    assert manifest["chunks"][chunk_id]["content"] == sanitized
    assert manifest["chunks"][chunk_id]["file_path"] == "doc.md"
    assert manifest["entities"]["doc:sanitized"]["source_chunk_id"] == chunk_id
    assert manifest["entities"]["doc:sanitized"]["file_path"] == "doc.md"


def test_custom_kg_vector_hash_embedding_contract_and_metadata_only_paths() -> None:
    from ops.custom_kg_incremental import build_custom_kg_manifest

    payload = {
        "chunks": [{"content": "Doc A", "source_id": "doc:a", "file_path": "a.md"}],
        "entities": [{"entity_name": "topic:x", "entity_type": "TOPIC", "description": "same", "source_id": "doc:a", "file_path": "a.md"}],
        "relationships": [{"src_id": "doc:a", "tgt_id": "topic:x", "description": "same edge", "keywords": "RELATED", "source_id": "doc:a", "file_path": "a.md"}],
    }
    path_changed_payload = {
        "chunks": payload["chunks"],
        "entities": [{"entity_name": "topic:x", "entity_type": "TOPIC", "description": "same", "source_id": "doc:a", "file_path": "new.md"}],
        "relationships": [{"src_id": "doc:a", "tgt_id": "topic:x", "description": "same edge", "keywords": "RELATED", "source_id": "doc:a", "file_path": "new.md"}],
    }

    base = build_custom_kg_manifest(payload, native_manifest_tool_version="1.5.0", embedding_model="embed-a", embedding_dim=3, embedding_params_version="v1")
    model_changed = build_custom_kg_manifest(payload, native_manifest_tool_version="1.5.0", embedding_model="embed-b", embedding_dim=3, embedding_params_version="v1")
    dim_changed = build_custom_kg_manifest(payload, native_manifest_tool_version="1.5.0", embedding_model="embed-a", embedding_dim=4, embedding_params_version="v1")
    params_changed = build_custom_kg_manifest(payload, native_manifest_tool_version="1.5.0", embedding_model="embed-a", embedding_dim=3, embedding_params_version="v2")
    path_changed = build_custom_kg_manifest(path_changed_payload, native_manifest_tool_version="1.5.0", embedding_model="embed-a", embedding_dim=3, embedding_params_version="v1")

    entity_hash = next(iter(base["entities"].values()))["vector_hash"]
    relationship_hash = next(iter(base["relationships"].values()))["vector_hash"]
    chunk_hash = next(iter(base["chunks"].values()))["vector_hash"]

    assert next(iter(model_changed["entities"].values()))["vector_hash"] != entity_hash
    assert next(iter(dim_changed["relationships"].values()))["vector_hash"] != relationship_hash
    assert next(iter(params_changed["chunks"].values()))["vector_hash"] != chunk_hash
    assert next(iter(path_changed["entities"].values()))["vector_hash"] == entity_hash
    assert next(iter(path_changed["relationships"].values()))["vector_hash"] == relationship_hash


def test_custom_kg_payload_includes_reviewed_section_similarity_edges(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    extract_raw_sections(root, state)
    build_seed_edges(root, state)
    jsonl = state / "section_similarity_edges.jsonl"
    write(
        jsonl,
        json.dumps(
            {
                "edge_id": "semantic_section_neighbor:test",
                "type": "SEMANTIC_SECTION_NEIGHBOR",
                "src_id": "raw_section:26010101_Foo-Paper:future",
                "tgt_id": "raw_section:26010101_Foo-Paper:questions",
                "source_section_kind": "future",
                "target_section_kind": "questions",
                "source_path": "raw/clip/2601/26010101_Foo-Paper.md",
                "target_path": "raw/clip/2601/26010101_Foo-Paper.md",
                "cosine": 0.82,
                "mutual_knn": True,
                "embedding_model": "test-embedding",
            },
            ensure_ascii=False,
        )
        + "\n",
    )
    payload, summary = build_custom_kg_payload(root, state, limit_docs=2, limit_edges=0)
    assert summary["section_similarity_relationships"] == 1
    rels = [rel for rel in payload["relationships"] if rel["keywords"] == "SEMANTIC_SECTION_NEIGHBOR"]
    assert len(rels) == 1
    assert rels[0]["src_id"] == "raw_section:26010101_Foo-Paper:future"
    assert rels[0]["tgt_id"] == "raw_section:26010101_Foo-Paper:questions"
    assert "cosine=0.820000" in rels[0]["description"]
