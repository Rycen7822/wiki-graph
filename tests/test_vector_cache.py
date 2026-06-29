import base64
import hashlib
import json
import sqlite3
import struct
import sys
import zlib
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from vector_cache import VectorCache, resolve_manifest_vectors, seed_vector_cache_from_storage  # noqa: E402


def _pair_only_rel_id(src_id: str, tgt_id: str) -> str:
    normalized_src, normalized_tgt = sorted((src_id, tgt_id))
    return "rel-" + hashlib.md5((normalized_src + normalized_tgt).encode("utf-8")).hexdigest()


def test_vector_cache_resolves_matching_embedding_contract(tmp_path) -> None:
    cache = VectorCache(tmp_path / "vector_cache.sqlite")
    cache.put(
        "hash-a",
        record_type="entity",
        record_id="doc:a",
        embedding_model="embed-a",
        embedding_dim=3,
        embedding_params_version="v1",
        vector=[1.0, 2.0, 3.0],
    )

    cached = cache.resolve("hash-a", embedding_model="embed-a", embedding_dim=3, embedding_params_version="v1")

    assert cached is not None
    assert cached["record_type"] == "entity"
    assert cached["record_id"] == "doc:a"
    assert cached["vector"] == pytest.approx([1.0, 2.0, 3.0])


def test_vector_cache_misses_on_embedding_contract_mismatch(tmp_path) -> None:
    cache = VectorCache(tmp_path / "vector_cache.sqlite")
    cache.put(
        "hash-a",
        record_type="entity",
        record_id="doc:a",
        embedding_model="embed-a",
        embedding_dim=3,
        embedding_params_version="v1",
        vector=[1.0, 2.0, 3.0],
    )

    assert cache.resolve("hash-a", embedding_model="embed-b", embedding_dim=3, embedding_params_version="v1") is None
    assert cache.resolve("hash-a", embedding_model="embed-a", embedding_dim=4, embedding_params_version="v1") is None
    assert cache.resolve("hash-a", embedding_model="embed-a", embedding_dim=3, embedding_params_version="v2") is None


def test_vector_cache_treats_checksum_corruption_as_miss(tmp_path) -> None:
    path = tmp_path / "vector_cache.sqlite"
    cache = VectorCache(path)
    cache.put(
        "hash-a",
        record_type="entity",
        record_id="doc:a",
        embedding_model="embed-a",
        embedding_dim=3,
        embedding_params_version="v1",
        vector=[1.0, 2.0, 3.0],
    )
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE vector_cache SET vector_blob = ? WHERE vector_hash = ?", (b"corrupt", "hash-a"))

    assert cache.resolve("hash-a", embedding_model="embed-a", embedding_dim=3, embedding_params_version="v1") is None


def test_vector_cache_rejects_non_finite_vectors(tmp_path) -> None:
    cache = VectorCache(tmp_path / "vector_cache.sqlite")

    with pytest.raises(ValueError, match="finite"):
        cache.put(
            "hash-a",
            record_type="entity",
            record_id="doc:a",
            embedding_model="embed-a",
            embedding_dim=1,
            embedding_params_version="v1",
            vector=[float("nan")],
        )


def test_manifest_vector_resolver_reports_hits_and_misses(tmp_path) -> None:
    cache = VectorCache(tmp_path / "vector_cache.sqlite")
    cache.put(
        "hash-hit",
        record_type="entity",
        record_id="cached-old-id",
        embedding_model="embed-a",
        embedding_dim=2,
        embedding_params_version="v1",
        vector=[0.25, 0.75],
    )
    manifest = {
        "chunks": {},
        "entities": {
            "topic:x": {
                "record_type": "entity",
                "record_id": "ent-new-id",
                "canonical_id": "topic:x",
                "vector_hash": "hash-hit",
                "embedding_model": "embed-a",
                "embedding_dim": 2,
                "embedding_params_version": "v1",
            },
            "topic:y": {
                "record_type": "entity",
                "record_id": "ent-miss-id",
                "canonical_id": "topic:y",
                "vector_hash": "hash-miss",
                "embedding_model": "embed-a",
                "embedding_dim": 2,
                "embedding_params_version": "v1",
            },
        },
        "relationships": {
            "topic:x<SEP>topic:y": {
                "record_type": "relationship",
                "record_id": "rel-missing-contract",
                "canonical_id": "topic:x<SEP>topic:y",
                "vector_hash": "hash-no-contract",
            }
        },
    }

    result = resolve_manifest_vectors(manifest, cache)

    hit = result["resolved"]["entities"]["topic:x"]
    assert hit["record_id"] == "ent-new-id"
    assert hit["cached_record_id"] == "cached-old-id"
    assert hit["vector"] == pytest.approx([0.25, 0.75])
    assert result["summary"]["entities"] == {"total": 2, "hits": 1, "misses": 1}
    assert result["summary"]["relationships"] == {"total": 1, "hits": 0, "misses": 1}
    assert result["missing"]["entities"] == ["topic:y"]
    assert result["missing"]["relationships"] == ["topic:x<SEP>topic:y"]


def test_manifest_vector_resolver_uses_bulk_cache_lookup(tmp_path, monkeypatch) -> None:
    cache = VectorCache(tmp_path / "vector_cache.sqlite")
    manifest = {"chunks": {}, "entities": {}, "relationships": {}}
    for idx in range(12):
        key = f"chunk-{idx:02d}"
        cache.put(
            f"hash-{idx:02d}",
            record_type="chunk",
            record_id=key,
            embedding_model="embed-a",
            embedding_dim=2,
            embedding_params_version="v1",
            vector=[float(idx), float(idx + 1)],
        )
        manifest["chunks"][key] = {
            "record_type": "chunk",
            "record_id": key,
            "canonical_id": key,
            "vector_hash": f"hash-{idx:02d}",
            "embedding_model": "embed-a",
            "embedding_dim": 2,
            "embedding_params_version": "v1",
        }

    def fail_per_record_resolve(*_args, **_kwargs):
        raise AssertionError("resolve_manifest_vectors must use bulk cache lookup")

    monkeypatch.setattr(cache, "resolve", fail_per_record_resolve)

    result = resolve_manifest_vectors(manifest, cache)

    assert result["summary"]["total"] == {"total": 12, "hits": 12, "misses": 0}
    assert result["resolved"]["chunks"]["chunk-07"]["vector"] == pytest.approx([7.0, 8.0])


def test_seed_vector_cache_uses_bulk_cache_write(tmp_path, monkeypatch) -> None:
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    data = [
        {"__id__": f"chunk-{idx:02d}", "vector": [float(idx), float(idx + 1)]}
        for idx in range(10)
    ]
    (storage_dir / "vdb_chunks.json").write_text(json.dumps({"embedding_dim": 2, "data": data, "matrix": ""}), encoding="utf-8")
    (storage_dir / "vdb_entities.json").write_text(json.dumps({"embedding_dim": 2, "data": [], "matrix": ""}), encoding="utf-8")
    (storage_dir / "vdb_relationships.json").write_text(json.dumps({"embedding_dim": 2, "data": [], "matrix": ""}), encoding="utf-8")
    manifest = {"chunks": {}, "entities": {}, "relationships": {}}
    for idx in range(10):
        key = f"chunk-{idx:02d}"
        manifest["chunks"][key] = {
            "record_type": "chunk",
            "record_id": key,
            "canonical_id": key,
            "vector_hash": f"hash-{idx:02d}",
            "embedding_model": "embed-a",
            "embedding_dim": 2,
            "embedding_params_version": "v1",
        }
    cache = VectorCache(tmp_path / "cache.sqlite")

    def fail_per_record_put(*_args, **_kwargs):
        raise AssertionError("seed_vector_cache_from_storage must use bulk cache write")

    monkeypatch.setattr(cache, "put", fail_per_record_put)

    seed = seed_vector_cache_from_storage(manifest, storage_dir, cache)
    resolved = resolve_manifest_vectors(manifest, cache)

    assert seed["summary"]["total"] == {"total": 10, "seeded": 10, "missing": 0}
    assert resolved["summary"]["total"] == {"total": 10, "hits": 10, "misses": 0}


def test_seed_vector_cache_skips_storage_vectors_when_previous_vector_hash_differs(tmp_path) -> None:
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    (storage_dir / "vdb_chunks.json").write_text(json.dumps({"embedding_dim": 2, "data": [{"__id__": "chunk-a", "vector": [1.0, 2.0]}], "matrix": ""}), encoding="utf-8")
    (storage_dir / "vdb_entities.json").write_text(json.dumps({"embedding_dim": 2, "data": [], "matrix": ""}), encoding="utf-8")
    (storage_dir / "vdb_relationships.json").write_text(json.dumps({"embedding_dim": 2, "data": [], "matrix": ""}), encoding="utf-8")
    previous_manifest = {
        "chunks": {
            "chunk-a": {
                "record_type": "chunk",
                "record_id": "chunk-a",
                "canonical_id": "chunk-a",
                "vector_hash": "old-vector-hash",
                "embedding_model": "embed-a",
                "embedding_dim": 2,
                "embedding_params_version": "v1",
            }
        },
        "entities": {},
        "relationships": {},
    }
    desired_manifest = json.loads(json.dumps(previous_manifest))
    desired_manifest["chunks"]["chunk-a"]["vector_hash"] = "new-vector-hash"
    cache = VectorCache(tmp_path / "cache.sqlite")

    seed = seed_vector_cache_from_storage(desired_manifest, storage_dir, cache, previous_manifest=previous_manifest)
    resolved = resolve_manifest_vectors(desired_manifest, cache)

    assert seed["summary"]["total"] == {"total": 1, "seeded": 0, "missing": 1}
    assert seed["skipped_vector_hash_mismatch"]["chunks"] == ["chunk-a"]
    assert resolved["summary"]["total"] == {"total": 1, "hits": 0, "misses": 1}


def test_seed_vector_cache_does_not_probe_pair_only_relationship_storage(tmp_path) -> None:
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    src_id = "raw_section:z"
    tgt_id = "raw_clip:a"
    pair_only_id = _pair_only_rel_id(src_id, tgt_id)
    desired_content = "RAW_SECTION_OF\traw_section:z\nraw_clip:a\nraw_section:z RAW_SECTION_OF raw_clip:a"
    (storage_dir / "vdb_chunks.json").write_text(json.dumps({"embedding_dim": 2, "data": [], "matrix": ""}), encoding="utf-8")
    (storage_dir / "vdb_entities.json").write_text(json.dumps({"embedding_dim": 2, "data": [], "matrix": ""}), encoding="utf-8")
    (storage_dir / "vdb_relationships.json").write_text(
        json.dumps(
            {
                "embedding_dim": 2,
                "data": [
                    {
                        "__id__": pair_only_id,
                        "src_id": "raw_clip:a",
                        "tgt_id": "raw_section:z",
                        "keywords": "RAW_SECTION_OF",
                        "content": desired_content,
                        "vector": [1.0, 2.0],
                    }
                ],
                "matrix": "",
            }
        ),
        encoding="utf-8",
    )
    key = "raw_section:z<SEP>raw_clip:a<SEP>RAW_SECTION_OF"
    previous_manifest = {
        "metadata": {"embedding_model": "embed-a", "embedding_dim": 2, "embedding_params_version": "v1"},
        "chunks": {},
        "entities": {},
        "relationships": {
            key: {
                "record_type": "relationship",
                "record_id": "rel-typed",
                "canonical_id": key,
                "src_id": src_id,
                "tgt_id": tgt_id,
                "keywords": "RAW_SECTION_OF",
                "content": "RAW_SECTION_OF\traw_section:z\nraw_clip:a\nraw_section:z RAW_SECTION_OF raw_clip:a",
                "vector_hash": "old-directed-hash",
                "embedding_model": "embed-a",
                "embedding_dim": 2,
                "embedding_params_version": "v1",
            }
        },
    }
    desired_manifest = json.loads(json.dumps(previous_manifest))
    desired_manifest["relationships"][key]["content"] = desired_content
    desired_manifest["relationships"][key]["vector_hash"] = "new-directed-hash"
    cache = VectorCache(tmp_path / "cache.sqlite")

    seed = seed_vector_cache_from_storage(desired_manifest, storage_dir, cache, previous_manifest=previous_manifest)
    resolved = resolve_manifest_vectors(desired_manifest, cache)

    assert seed["summary"]["total"] == {"total": 1, "seeded": 0, "missing": 1}
    assert seed["skipped_vector_hash_mismatch"]["relationships"] == [key]
    assert resolved["summary"]["total"] == {"total": 1, "hits": 0, "misses": 1}


def test_seed_vector_cache_rejects_pair_only_relationship_reuse_when_embedding_contract_changes(tmp_path) -> None:
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    src_id = "raw_section:z"
    tgt_id = "raw_clip:a"
    pair_only_id = _pair_only_rel_id(src_id, tgt_id)
    desired_content = "RAW_SECTION_OF\traw_section:z\nraw_clip:a\nraw_section:z RAW_SECTION_OF raw_clip:a"
    (storage_dir / "vdb_chunks.json").write_text(json.dumps({"embedding_dim": 3, "data": [], "matrix": ""}), encoding="utf-8")
    (storage_dir / "vdb_entities.json").write_text(json.dumps({"embedding_dim": 3, "data": [], "matrix": ""}), encoding="utf-8")
    (storage_dir / "vdb_relationships.json").write_text(
        json.dumps(
            {
                "embedding_dim": 3,
                "data": [
                    {
                        "__id__": pair_only_id,
                        "src_id": "raw_clip:a",
                        "tgt_id": "raw_section:z",
                        "keywords": "RAW_SECTION_OF",
                        "content": desired_content,
                        "vector": [1.0, 2.0, 3.0],
                    }
                ],
                "matrix": "",
            }
        ),
        encoding="utf-8",
    )
    key = "raw_section:z<SEP>raw_clip:a<SEP>RAW_SECTION_OF"
    previous_manifest = {
        "metadata": {"embedding_model": "embed-old", "embedding_dim": 3, "embedding_params_version": "v1"},
        "chunks": {},
        "entities": {},
        "relationships": {
            key: {
                "record_type": "relationship",
                "record_id": "rel-typed",
                "canonical_id": key,
                "src_id": src_id,
                "tgt_id": tgt_id,
                "keywords": "RAW_SECTION_OF",
                "content": desired_content,
                "vector_hash": "old-contract-hash",
                "embedding_model": "embed-old",
                "embedding_dim": 3,
                "embedding_params_version": "v1",
            }
        },
    }
    desired_manifest = json.loads(json.dumps(previous_manifest))
    desired_manifest["metadata"]["embedding_model"] = "embed-new"
    desired_manifest["relationships"][key]["embedding_model"] = "embed-new"
    desired_manifest["relationships"][key]["vector_hash"] = "new-contract-hash"
    cache = VectorCache(tmp_path / "cache.sqlite")

    seed = seed_vector_cache_from_storage(desired_manifest, storage_dir, cache, previous_manifest=previous_manifest)
    resolved = resolve_manifest_vectors(desired_manifest, cache)

    assert seed["summary"]["total"] == {"total": 1, "seeded": 0, "missing": 1}
    assert seed["skipped_vector_hash_mismatch"]["relationships"] == [key]
    assert resolved["summary"]["total"] == {"total": 1, "hits": 0, "misses": 1}


def test_seed_vector_cache_decodes_nanovectordb_compressed_float16_vectors(tmp_path) -> None:
    storage_dir = tmp_path / "storage"
    storage_dir.mkdir()
    encoded = base64.b64encode(zlib.compress(b"".join(struct.pack("<e", value) for value in [0.25, -0.5]))).decode("ascii")
    (storage_dir / "vdb_chunks.json").write_text(json.dumps({"embedding_dim": 2, "data": [{"__id__": "chunk-a", "vector": encoded}], "matrix": ""}), encoding="utf-8")
    (storage_dir / "vdb_entities.json").write_text(json.dumps({"embedding_dim": 2, "data": [], "matrix": ""}), encoding="utf-8")
    (storage_dir / "vdb_relationships.json").write_text(json.dumps({"embedding_dim": 2, "data": [], "matrix": ""}), encoding="utf-8")
    manifest = {
        "chunks": {
            "chunk-a": {
                "record_type": "chunk",
                "record_id": "chunk-a",
                "canonical_id": "chunk-a",
                "vector_hash": "hash-a",
                "embedding_model": "embed-a",
                "embedding_dim": 2,
                "embedding_params_version": "v1",
            }
        },
        "entities": {},
        "relationships": {},
    }
    cache = VectorCache(tmp_path / "cache.sqlite")

    seed = seed_vector_cache_from_storage(manifest, storage_dir, cache)
    resolved = resolve_manifest_vectors(manifest, cache)

    assert seed["summary"]["total"] == {"total": 1, "seeded": 1, "missing": 0}
    assert resolved["resolved"]["chunks"]["chunk-a"]["vector"] == pytest.approx([0.25, -0.5])
