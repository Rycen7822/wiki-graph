import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"
SCRIPTS = OPS
sys.path.insert(0, str(ROOT))

from ops.vector_cache import VectorCache, resolve_manifest_vectors  # noqa: E402


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


def test_vector_cache_has_no_retired_storage_seed_api() -> None:
    from ops import vector_cache

    source = (Path(__file__).resolve().parents[1] / "ops" / "vector_cache.py").read_text(encoding="utf-8")
    assert not hasattr(vector_cache, "seed_vector_cache_from_storage")
    assert "def seed_vector_cache_from_storage(" not in source
    assert "SEED_VECTOR_CACHE_FROM_STORAGE_RETIRED_MESSAGE" not in source


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
