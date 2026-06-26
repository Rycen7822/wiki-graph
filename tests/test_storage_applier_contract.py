import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from custom_kg_incremental import build_custom_kg_manifest  # noqa: E402
from custom_kg_storage_applier import FileBackendStorageApplier  # noqa: E402
from vector_cache import VectorCache, resolve_manifest_vectors  # noqa: E402


def _payload() -> dict:
    return {
        "chunks": [{"content": "Doc A content", "source_id": "doc:a", "file_path": "a.md", "chunk_order_index": 0}],
        "entities": [
            {"entity_name": "doc:a", "entity_type": "DOC", "description": "Doc A", "source_id": "doc:a", "file_path": "a.md"},
            {"entity_name": "topic:x", "entity_type": "TOPIC", "description": "Topic X", "source_id": "doc:a", "file_path": "a.md"},
        ],
        "relationships": [
            {"src_id": "doc:a", "tgt_id": "topic:x", "description": "Doc discusses Topic X", "keywords": "DISCUSSES", "source_id": "doc:a", "weight": 1.0, "file_path": "a.md"},
        ],
    }


def _resolved_manifest_vectors(tmp_path: Path) -> tuple[dict, dict]:
    manifest = build_custom_kg_manifest(_payload(), lightrag_version="1.5.0", embedding_model="embed-a", embedding_dim=2, embedding_params_version="v1")
    cache = VectorCache(tmp_path / "vector_cache.sqlite")
    ordinal = 1
    for collection in ("chunks", "entities", "relationships"):
        for record in manifest[collection].values():
            cache.put(
                record["vector_hash"],
                record_type=record["record_type"],
                record_id=record["record_id"],
                embedding_model=record["embedding_model"],
                embedding_dim=record["embedding_dim"],
                embedding_params_version=record["embedding_params_version"],
                vector=[float(ordinal), float(ordinal + 1)],
            )
            ordinal += 1
    resolved = resolve_manifest_vectors(manifest, cache)
    assert resolved["summary"]["total"]["misses"] == 0
    return manifest, resolved["resolved"]


def test_file_backend_storage_applier_materializes_audits_and_hashes_stable_workspace(tmp_path: Path) -> None:
    manifest, resolved = _resolved_manifest_vectors(tmp_path)
    applier = FileBackendStorageApplier(tmp_path / "workspaces", pointer_path=tmp_path / "active_workspace.json")

    first = applier.materialize("candidate-a", manifest, resolved)
    second = applier.materialize("candidate-b", manifest, resolved)

    assert first["backend"] == "file"
    assert first["audit"]["ok"] is True
    assert first["counts"] == second["counts"]
    assert first["semantic_digest"] == second["semantic_digest"]
    assert Path(first["storage_dir"]).exists()


def test_file_backend_storage_applier_pointer_rollback_restores_previous_workspace(tmp_path: Path) -> None:
    manifest, resolved = _resolved_manifest_vectors(tmp_path)
    applier = FileBackendStorageApplier(tmp_path / "workspaces", pointer_path=tmp_path / "active_workspace.json")
    first = applier.materialize("candidate-a", manifest, resolved)
    second = applier.materialize("candidate-b", manifest, resolved)

    applier.activate(first)
    applier.activate(second)
    rolled_back = applier.rollback_active()

    assert rolled_back["active_workspace_id"] == "candidate-a"
    assert applier.active_workspace()["active_workspace_id"] == "candidate-a"
    assert applier.active_workspace()["previous_workspace_id"] == "candidate-b"
