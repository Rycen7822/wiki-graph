from array import array
import base64
import copy
import json
import sys
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import custom_kg_incremental  # noqa: E402
from custom_kg_incremental import audit_custom_kg_storage, build_custom_kg_manifest  # noqa: E402
from custom_kg_materialize import materialize_file_storage_from_manifest  # noqa: E402
from vector_cache import VectorCache, resolve_manifest_vectors, seed_vector_cache_from_storage  # noqa: E402


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


def _assert_compact_vector_cache_report(report: dict) -> None:
    vector_cache = report["vector_cache"]
    assert set(vector_cache) == {"summary", "missing_counts", "missing_examples"}
    assert vector_cache["summary"]["total"]["misses"] == 0
    serialized = json.dumps(report, ensure_ascii=False)
    assert '"resolved"' not in serialized
    assert '"vector"' not in serialized
    assert '"vector": [' not in serialized


def _seed_manifest_vectors(manifest: dict, cache: VectorCache) -> None:
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


def _materialize_manifest_storage(manifest: dict, storage_dir: Path, cache_path: Path) -> VectorCache:
    cache = VectorCache(cache_path)
    _seed_manifest_vectors(manifest, cache)
    materialize_file_storage_from_manifest(manifest, resolve_manifest_vectors(manifest, cache)["resolved"], storage_dir)
    return cache


def _write_prepared_swap_report(
    state_dir: Path,
    desired_manifest: dict,
    previous_manifest: dict,
    shadow_storage: Path,
    backup_dir: Path,
    shadow_audit: dict,
    *,
    fingerprint: dict | None = None,
) -> Path:
    custom_kg_incremental.write_manifest(state_dir, previous_manifest)
    prepared_manifest = custom_kg_incremental.prepared_swap_manifest_path(state_dir)
    prepared_manifest.parent.mkdir(parents=True, exist_ok=True)
    prepared_manifest.write_text(json.dumps(desired_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "started_at": "2026-06-26 00:00",
        "prepared_for_swap": True,
        "import_mode": "full_materialization",
        "previous_manifest_hash": custom_kg_incremental.stable_hash(previous_manifest),
        "desired_manifest_hash": custom_kg_incremental.stable_hash(desired_manifest),
        "desired_manifest_path": str(prepared_manifest),
        "shadow_storage": str(shadow_storage),
        "backup_dir": str(backup_dir),
        "payload": {},
        "manifest": desired_manifest.get("summary", {}),
        "shadow_audit": shadow_audit,
    }
    if fingerprint is not None:
        report["prepared_shadow_fingerprint"] = fingerprint
    report_path = custom_kg_incremental.prepared_swap_report_path(state_dir)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def _prepared_finalize_fixture(tmp_path: Path, *, backup_name: str = "prepared-backup") -> types.SimpleNamespace:
    manifest = build_custom_kg_manifest(_payload(), lightrag_version="1.5.0", embedding_model="embed-a", embedding_dim=2, embedding_params_version="v1")
    state_dir = tmp_path / "state"
    workdir = tmp_path / "workdir"
    root = tmp_path / "wiki"
    live_storage = workdir / "rag_storage"
    shadow_storage = tmp_path / "prepared_shadow"
    backup_dir = state_dir / "backups" / backup_name
    state_dir.mkdir()
    workdir.mkdir()
    root.mkdir()
    live_storage.mkdir()
    (live_storage / "live.txt").write_text("live", encoding="utf-8")
    _materialize_manifest_storage(manifest, shadow_storage, tmp_path / "cache.sqlite")
    shadow_audit = audit_custom_kg_storage(shadow_storage, manifest)
    fingerprint = custom_kg_incremental.prepared_shadow_fingerprint(shadow_storage, custom_kg_incremental.stable_hash(manifest), shadow_audit)
    report_path = _write_prepared_swap_report(state_dir, manifest, manifest, shadow_storage, backup_dir, shadow_audit, fingerprint=fingerprint)
    return types.SimpleNamespace(
        manifest=manifest,
        state_dir=state_dir,
        workdir=workdir,
        root=root,
        live_storage=live_storage,
        shadow_storage=shadow_storage,
        backup_dir=backup_dir,
        shadow_audit=shadow_audit,
        report_path=report_path,
    )


def test_embedding_profile_env_defaults_and_rejects_unknown() -> None:
    assert custom_kg_incremental.embedding_profile_env("conservative") == {
        "EMBEDDING_FUNC_MAX_ASYNC": "1",
        "EMBEDDING_BATCH_NUM": "10",
        "MAX_PARALLEL_INSERT": "1",
    }
    assert custom_kg_incremental.embedding_profile_env("shadow-medium")["EMBEDDING_BATCH_NUM"] == "20"
    with pytest.raises(ValueError, match="unknown embedding profile"):
        custom_kg_incremental.embedding_profile_env("surprise-fast")


def test_fill_missing_manifest_vectors_reports_embedding_profile_metrics(tmp_path) -> None:
    manifest = {
        "metadata": {"embedding_model": "embed-a", "embedding_dim": 2, "embedding_params_version": "v1"},
        "chunks": {
            "chunk:a": {
                "record_type": "chunk",
                "record_id": "chunk:a",
                "vector_hash": "hash-a",
                "content": "Doc A content",
                "embedding_model": "embed-a",
                "embedding_dim": 2,
                "embedding_params_version": "v1",
            }
        },
        "entities": {},
        "relationships": {},
    }
    vector_report = {"missing": {"chunks": ["chunk:a"], "entities": [], "relationships": []}}
    cache = VectorCache(tmp_path / "cache.sqlite")

    def fake_embed(texts, **_kwargs):
        assert texts == ["Doc A content"]
        return [[0.1, 0.2]]

    report = custom_kg_incremental.fill_missing_manifest_vectors(
        manifest,
        vector_report,
        cache,
        workdir=tmp_path,
        embed_texts_func=fake_embed,
        embedding_profile="shadow-medium",
    )

    assert report["embedding_profile"] == "shadow-medium"
    assert report["batch_size"] == 20
    assert report["concurrency"] == {"embedding_func_max_async": 2, "max_parallel_insert": 1}
    assert report["total_batches"] == 1
    assert report["failed_batches"] == 0
    assert report["provider_retries"] == 0
    assert report["elapsed_by_collection_s"]["chunks"] >= 0


def test_manifest_and_materialization_preserve_same_endpoint_typed_relationships(tmp_path) -> None:
    payload = _payload()
    payload["relationships"] = [
        {"src_id": "doc:a", "tgt_id": "topic:x", "description": "Doc links to topic", "keywords": "WIKILINKS_TO", "source_id": "doc:a", "weight": 1.0, "file_path": "a.md"},
        {"src_id": "topic:x", "tgt_id": "doc:a", "description": "Topic cites doc", "keywords": "SOURCED_BY", "source_id": "doc:a", "weight": 0.8, "file_path": "a.md"},
    ]
    manifest = build_custom_kg_manifest(payload, lightrag_version="1.5.0", embedding_model="embed-a", embedding_dim=2, embedding_params_version="v1")

    assert len(manifest["relationships"]) == 2
    assert {record["keywords"] for record in manifest["relationships"].values()} == {"WIKILINKS_TO", "SOURCED_BY"}
    assert len({record["vdb_id"] for record in manifest["relationships"].values()}) == 2

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
    storage_dir = tmp_path / "shadow_storage"

    materialize_file_storage_from_manifest(manifest, resolve_manifest_vectors(manifest, cache)["resolved"], storage_dir)

    audit = audit_custom_kg_storage(storage_dir, manifest)
    assert audit["ok"] is True
    vdb_relationships = json.loads((storage_dir / "vdb_relationships.json").read_text(encoding="utf-8"))
    assert len(vdb_relationships["data"]) == 2
    relation_chunks = json.loads((storage_dir / "kv_store_relation_chunks.json").read_text(encoding="utf-8"))
    assert set(relation_chunks) == set(manifest["relationships"])


def test_materialized_audit_accepts_aggregated_graph_sources_for_same_pair_typed_relationships(tmp_path) -> None:
    payload = {
        "chunks": [
            {"content": "Doc A content", "source_id": "doc:a", "file_path": "a.md", "chunk_order_index": 0},
            {"content": "Doc B content", "source_id": "doc:b", "file_path": "b.md", "chunk_order_index": 1},
        ],
        "entities": [
            {"entity_name": "doc:a", "entity_type": "DOC", "description": "Doc A", "source_id": "doc:a", "file_path": "a.md"},
            {"entity_name": "topic:x", "entity_type": "TOPIC", "description": "Topic X", "source_id": "doc:a", "file_path": "a.md"},
        ],
        "relationships": [
            {"src_id": "doc:a", "tgt_id": "topic:x", "description": "Doc A links to topic", "keywords": "WIKILINKS_TO", "source_id": "doc:a", "weight": 1.0, "file_path": "a.md"},
            {"src_id": "topic:x", "tgt_id": "doc:a", "description": "Topic cites Doc A from Doc B context", "keywords": "SOURCED_BY", "source_id": "doc:b", "weight": 0.8, "file_path": "b.md"},
        ],
    }
    manifest = build_custom_kg_manifest(payload, lightrag_version="1.5.0", embedding_model="embed-a", embedding_dim=2, embedding_params_version="v1")
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
    storage_dir = tmp_path / "shadow_storage"

    materialize_file_storage_from_manifest(manifest, resolve_manifest_vectors(manifest, cache)["resolved"], storage_dir)
    audit = audit_custom_kg_storage(storage_dir, manifest)

    assert audit["ok"] is True


def test_relationship_vector_content_uses_pair_sorted_endpoint_order() -> None:
    payload = _payload()
    payload["relationships"] = [
        {
            "src_id": "topic:z",
            "tgt_id": "doc:a",
            "description": "topic:z SOURCED_BY doc:a",
            "keywords": "SOURCED_BY",
            "source_id": "doc:a",
            "weight": 1.0,
            "file_path": "a.md",
        }
    ]

    manifest = build_custom_kg_manifest(payload, lightrag_version="1.5.0", embedding_model="embed-a", embedding_dim=2, embedding_params_version="v1")
    relationship = next(iter(manifest["relationships"].values()))

    assert relationship["src_id"] == "topic:z"
    assert relationship["tgt_id"] == "doc:a"
    assert relationship["content"] == "SOURCED_BY\tdoc:a\ntopic:z\ntopic:z SOURCED_BY doc:a"


def test_full_materialization_blocker_ignores_legacy_relationship_endpoint_order_only_change() -> None:
    payload = _payload()
    payload["relationships"] = [
        {
            "src_id": "topic:z",
            "tgt_id": "doc:a",
            "description": "topic:z SOURCED_BY doc:a",
            "keywords": "SOURCED_BY",
            "source_id": "doc:a",
            "weight": 1.0,
            "file_path": "a.md",
        }
    ]
    desired = build_custom_kg_manifest(payload, lightrag_version="1.5.0", embedding_model="embed-a", embedding_dim=2, embedding_params_version="v1")
    previous = copy.deepcopy(desired)
    key, previous_relationship = next(iter(previous["relationships"].items()))
    previous_relationship["content"] = "SOURCED_BY\ttopic:z\ndoc:a\ntopic:z SOURCED_BY doc:a"
    custom_kg_incremental._stamp_identity_and_hashes(
        previous_relationship,
        record_id=previous_relationship["vdb_id"],
        canonical_id=key,
    )

    blockers = custom_kg_incremental.full_materialization_cache_only_blockers(previous, desired)

    assert blockers["blocked"] is False
    assert blockers["collections"] == {}
    assert blockers["diff"]["relationships"]["vector_update"] == 0
    assert blockers["diff"]["relationships"]["metadata_update"] == 1


def test_materialize_file_storage_from_all_hit_vector_cache_passes_audit(tmp_path) -> None:
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

    storage_dir = tmp_path / "shadow_storage"
    report = materialize_file_storage_from_manifest(manifest, resolved["resolved"], storage_dir)

    assert report["ok"] is True
    audit = audit_custom_kg_storage(storage_dir, manifest)
    assert audit["ok"] is True
    chunk_id = next(iter(manifest["chunks"]))
    vdb_chunks = json.loads((storage_dir / "vdb_chunks.json").read_text(encoding="utf-8"))
    assert vdb_chunks["embedding_dim"] == 2
    assert vdb_chunks["data"][0]["__id__"] == chunk_id
    text_chunks = json.loads((storage_dir / "kv_store_text_chunks.json").read_text(encoding="utf-8"))
    assert text_chunks[chunk_id]["tokens"] == 3


def test_materialize_file_storage_writes_nanovectordb_matrix_buffer(tmp_path) -> None:
    manifest = build_custom_kg_manifest(_payload(), lightrag_version="1.5.0", embedding_model="embed-a", embedding_dim=2, embedding_params_version="v1")
    cache = VectorCache(tmp_path / "vector_cache.sqlite")
    for collection in ("chunks", "entities", "relationships"):
        for idx, record in enumerate(manifest[collection].values(), start=1):
            cache.put(
                record["vector_hash"],
                record_type=record["record_type"],
                record_id=record["record_id"],
                embedding_model=record["embedding_model"],
                embedding_dim=record["embedding_dim"],
                embedding_params_version=record["embedding_params_version"],
                vector=[float(idx), float(idx + 1)],
            )
    storage_dir = tmp_path / "shadow_storage"

    materialize_file_storage_from_manifest(manifest, resolve_manifest_vectors(manifest, cache)["resolved"], storage_dir)

    vdb_chunks = json.loads((storage_dir / "vdb_chunks.json").read_text(encoding="utf-8"))
    matrix = array("f")
    matrix.frombytes(base64.b64decode(vdb_chunks["matrix"]))
    assert len(matrix) == len(vdb_chunks["data"]) * vdb_chunks["embedding_dim"]
    assert list(matrix[:2]) == pytest.approx([1.0, 2.0])


def test_seed_vector_cache_from_materialized_file_storage_roundtrips_all_vectors(tmp_path) -> None:
    manifest = build_custom_kg_manifest(_payload(), lightrag_version="1.5.0", embedding_model="embed-a", embedding_dim=2, embedding_params_version="v1")
    source_cache = VectorCache(tmp_path / "source_cache.sqlite")
    ordinal = 1
    for collection in ("chunks", "entities", "relationships"):
        for record in manifest[collection].values():
            source_cache.put(
                record["vector_hash"],
                record_type=record["record_type"],
                record_id=record["record_id"],
                embedding_model=record["embedding_model"],
                embedding_dim=record["embedding_dim"],
                embedding_params_version=record["embedding_params_version"],
                vector=[float(ordinal), float(ordinal + 1)],
            )
            ordinal += 1
    storage_dir = tmp_path / "shadow_storage"
    materialize_file_storage_from_manifest(manifest, resolve_manifest_vectors(manifest, source_cache)["resolved"], storage_dir)
    fresh_cache = VectorCache(tmp_path / "fresh_cache.sqlite")

    seed_report = seed_vector_cache_from_storage(manifest, storage_dir, fresh_cache)
    resolved = resolve_manifest_vectors(manifest, fresh_cache)

    assert seed_report["summary"]["total"] == {"total": 4, "seeded": 4, "missing": 0}
    assert resolved["summary"]["total"] == {"total": 4, "hits": 4, "misses": 0}


def test_materialize_file_storage_fails_closed_when_vectors_are_missing(tmp_path) -> None:
    manifest = build_custom_kg_manifest(_payload(), lightrag_version="1.5.0", embedding_model="embed-a", embedding_dim=2, embedding_params_version="v1")

    try:
        materialize_file_storage_from_manifest(manifest, {"chunks": {}, "entities": {}, "relationships": {}}, tmp_path / "shadow_storage")
    except RuntimeError as exc:
        assert "missing resolved vectors" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("materializer should fail closed when vectors are missing")


def test_run_full_materialization_no_swap_blocks_cache_only_when_diff_needs_new_vectors(monkeypatch, tmp_path) -> None:
    previous = build_custom_kg_manifest(_payload(), lightrag_version="1.5.0", embedding_model="embed-a", embedding_dim=2, embedding_params_version="v1")
    desired_payload = _payload()
    desired_payload["entities"].append({"entity_name": "topic:new", "entity_type": "TOPIC", "description": "New topic", "source_id": "doc:a", "file_path": "a.md"})
    desired = build_custom_kg_manifest(desired_payload, lightrag_version="1.5.0", embedding_model="embed-a", embedding_dim=2, embedding_params_version="v1")
    state_dir = tmp_path / "state"
    workdir = tmp_path / "workdir"
    root = tmp_path / "wiki"
    shadow = tmp_path / "shadow_full"
    state_dir.mkdir()
    workdir.mkdir()
    root.mkdir()
    custom_kg_incremental.write_manifest(state_dir, previous)
    cache = VectorCache(state_dir / "vector_cache.sqlite")
    ordinal = 1
    for collection in ("chunks", "entities", "relationships"):
        for record in desired[collection].values():
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
    monkeypatch.setattr(custom_kg_incremental, "build_desired_manifest", lambda *_args, **_kwargs: (desired, {"chunks": 1, "entities": 3, "relationships": 1}))
    args = types.SimpleNamespace(
        root=root,
        state_dir=state_dir,
        workdir=workdir,
        vector_cache=state_dir / "vector_cache.sqlite",
        storage_dir=shadow,
        delete_shadow_on_no_swap=False,
        limit_docs=None,
        limit_edges=None,
        seed_from_storage=False,
        seed_storage_dir=None,
        fill_missing_vectors=False,
    )

    with pytest.raises(RuntimeError, match="cache-only full materialization is unsafe"):
        custom_kg_incremental.run_full_materialization_no_swap(args)
    assert not shadow.exists()


def test_run_full_materialization_no_swap_fills_true_adds_after_cache_only_blocker(monkeypatch, tmp_path) -> None:
    previous = build_custom_kg_manifest(_payload(), lightrag_version="1.5.0", embedding_model="embed-a", embedding_dim=2, embedding_params_version="v1")
    desired_payload = _payload()
    desired_payload["entities"].append({"entity_name": "topic:new", "entity_type": "TOPIC", "description": "New topic", "source_id": "doc:a", "file_path": "a.md"})
    desired = build_custom_kg_manifest(desired_payload, lightrag_version="1.5.0", embedding_model="embed-a", embedding_dim=2, embedding_params_version="v1")
    state_dir = tmp_path / "state"
    workdir = tmp_path / "workdir"
    root = tmp_path / "wiki"
    shadow = tmp_path / "shadow_full"
    state_dir.mkdir()
    workdir.mkdir()
    root.mkdir()
    custom_kg_incremental.write_manifest(state_dir, previous)
    cache = VectorCache(state_dir / "vector_cache.sqlite")
    ordinal = 1
    for collection in ("chunks", "entities", "relationships"):
        for record in previous[collection].values():
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
    new_entity = desired["entities"]["topic:new"]
    embedded_texts = []

    def fake_embed_texts(texts, **_kwargs):
        embedded_texts.extend(texts)
        return [[9.0, 10.0] for _text in texts]

    monkeypatch.setattr(custom_kg_incremental, "embed_texts_openai_compatible", fake_embed_texts, raising=False)
    monkeypatch.setattr(custom_kg_incremental, "build_desired_manifest", lambda *_args, **_kwargs: (desired, {"chunks": 1, "entities": 3, "relationships": 1}))
    args = types.SimpleNamespace(
        root=root,
        state_dir=state_dir,
        workdir=workdir,
        vector_cache=state_dir / "vector_cache.sqlite",
        storage_dir=shadow,
        delete_shadow_on_no_swap=False,
        limit_docs=None,
        limit_edges=None,
        seed_from_storage=False,
        seed_storage_dir=None,
        fill_missing_vectors=True,
        smoke_query=[],
        prepare_swap=False,
    )

    report = custom_kg_incremental.run_full_materialization_no_swap(args)

    assert embedded_texts == [new_entity["content"]]
    assert report["cache_only_blockers"]["blocked"] is True
    assert report["cache_only_blockers"]["collections"] == {"entities": {"add": 1, "vector_update": 0}}
    assert report["vector_cache_fill"]["summary"] == {"total": 1, "embedded": 1}
    assert report["vector_cache"]["summary"]["total"]["misses"] == 0
    _assert_compact_vector_cache_report(report)
    assert report["shadow_audit"]["ok"] is True
    assert Path(report["shadow_storage"]).exists()


def test_run_full_materialization_no_swap_fills_missing_vectors_when_enabled(monkeypatch, tmp_path) -> None:
    manifest = build_custom_kg_manifest(_payload(), lightrag_version="1.5.0", embedding_model="embed-a", embedding_dim=2, embedding_params_version="v1")
    state_dir = tmp_path / "state"
    workdir = tmp_path / "workdir"
    root = tmp_path / "wiki"
    shadow = tmp_path / "shadow_full"
    state_dir.mkdir()
    workdir.mkdir()
    root.mkdir()
    cache = VectorCache(state_dir / "vector_cache.sqlite")
    ordinal = 1
    for collection in ("chunks", "entities"):
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
    missing_relationship = next(iter(manifest["relationships"].values()))
    embedded_texts = []

    def fake_embed_texts(texts, **_kwargs):
        embedded_texts.extend(texts)
        return [[9.0, 10.0] for _text in texts]

    monkeypatch.setattr(custom_kg_incremental, "embed_texts_openai_compatible", fake_embed_texts, raising=False)
    monkeypatch.setattr(custom_kg_incremental, "build_desired_manifest", lambda *_args, **_kwargs: (manifest, {"chunks": 1, "entities": 2, "relationships": 1}))
    args = types.SimpleNamespace(
        root=root,
        state_dir=state_dir,
        workdir=workdir,
        vector_cache=state_dir / "vector_cache.sqlite",
        storage_dir=shadow,
        delete_shadow_on_no_swap=False,
        limit_docs=None,
        limit_edges=None,
        seed_from_storage=False,
        seed_storage_dir=None,
        fill_missing_vectors=True,
        smoke_query=[],
        prepare_swap=False,
    )

    report = custom_kg_incremental.run_full_materialization_no_swap(args)

    assert embedded_texts == [missing_relationship["content"]]
    assert report["vector_cache_fill"]["summary"] == {"total": 1, "embedded": 1}
    assert report["vector_cache"]["summary"]["total"]["misses"] == 0
    _assert_compact_vector_cache_report(report)
    assert report["shadow_audit"]["ok"] is True
    assert Path(report["shadow_storage"]).exists()


def test_run_full_materialization_no_swap_builds_audited_shadow(monkeypatch, tmp_path) -> None:
    manifest = build_custom_kg_manifest(_payload(), lightrag_version="1.5.0", embedding_model="embed-a", embedding_dim=2, embedding_params_version="v1")
    state_dir = tmp_path / "state"
    workdir = tmp_path / "workdir"
    root = tmp_path / "wiki"
    shadow = tmp_path / "shadow_full"
    state_dir.mkdir()
    workdir.mkdir()
    root.mkdir()
    cache = VectorCache(state_dir / "vector_cache.sqlite")
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
    monkeypatch.setattr(custom_kg_incremental, "build_desired_manifest", lambda *_args, **_kwargs: (manifest, {"chunks": 1, "entities": 2, "relationships": 1}))
    args = types.SimpleNamespace(
        root=root,
        state_dir=state_dir,
        workdir=workdir,
        vector_cache=state_dir / "vector_cache.sqlite",
        storage_dir=shadow,
        delete_shadow_on_no_swap=False,
        limit_docs=None,
        limit_edges=None,
    )

    report = custom_kg_incremental.run_full_materialization_no_swap(args)

    assert report["import_mode"] == "full_materialization"
    assert report["swapped"] is False
    assert report["vector_cache"]["summary"]["total"]["misses"] == 0
    _assert_compact_vector_cache_report(report)
    assert report["shadow_audit"]["ok"] is True
    assert Path(report["shadow_storage"]).exists()


def test_run_full_materialization_prepare_swap_writes_bundle_without_live_swap(monkeypatch, tmp_path) -> None:
    manifest = build_custom_kg_manifest(_payload(), lightrag_version="1.5.0", embedding_model="embed-a", embedding_dim=2, embedding_params_version="v1")
    state_dir = tmp_path / "state"
    workdir = tmp_path / "workdir"
    root = tmp_path / "wiki"
    shadow = tmp_path / "prepared_shadow"
    state_dir.mkdir()
    workdir.mkdir()
    root.mkdir()
    live_storage = workdir / "rag_storage"
    cache = VectorCache(state_dir / "vector_cache.sqlite")
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
    resolved = resolve_manifest_vectors(manifest, cache)["resolved"]
    materialize_file_storage_from_manifest(manifest, resolved, live_storage)
    custom_kg_incremental.write_manifest(state_dir, manifest)
    monkeypatch.setattr(custom_kg_incremental, "build_desired_manifest", lambda *_args, **_kwargs: (manifest, {"chunks": 1, "entities": 2, "relationships": 1}))
    args = types.SimpleNamespace(
        root=root,
        state_dir=state_dir,
        workdir=workdir,
        vector_cache=state_dir / "vector_cache.sqlite",
        storage_dir=shadow,
        delete_shadow_on_no_swap=False,
        limit_docs=None,
        limit_edges=None,
        prepare_swap=True,
        seed_from_storage=False,
        seed_storage_dir=None,
        smoke_query=[],
    )

    report = custom_kg_incremental.run_full_materialization_no_swap(args)

    assert report["import_mode"] == "full_materialization"
    assert report["prepared_for_swap"] is True
    assert report["swapped"] is False
    assert report["manifest_path"] is None
    assert custom_kg_incremental.load_manifest(state_dir) == manifest
    assert Path(report["shadow_storage"]).exists()
    assert (live_storage / "vdb_chunks.json").exists()
    prepared_report = custom_kg_incremental.prepared_swap_report_path(state_dir)
    prepared_manifest = custom_kg_incremental.prepared_swap_manifest_path(state_dir)
    assert prepared_report.exists()
    assert prepared_manifest.exists()
    persisted = json.loads(prepared_report.read_text(encoding="utf-8"))
    assert persisted["prepared_for_swap"] is True
    assert persisted["previous_manifest_hash"] == custom_kg_incremental.stable_hash(manifest)
    assert persisted["desired_manifest_hash"] == custom_kg_incremental.stable_hash(manifest)
    assert persisted["desired_manifest_path"] == str(prepared_manifest)
    _assert_compact_vector_cache_report(report)
    _assert_compact_vector_cache_report(persisted)
    assert prepared_report.stat().st_size < 20_000


def test_run_full_materialization_prepare_swap_can_repair_current_storage_audit_failure(monkeypatch, tmp_path) -> None:
    manifest = build_custom_kg_manifest(_payload(), lightrag_version="1.5.0", embedding_model="embed-a", embedding_dim=2, embedding_params_version="v1")
    state_dir = tmp_path / "state"
    workdir = tmp_path / "workdir"
    root = tmp_path / "wiki"
    shadow = tmp_path / "prepared_shadow"
    state_dir.mkdir()
    workdir.mkdir()
    root.mkdir()
    (workdir / "rag_storage").mkdir()
    cache = VectorCache(state_dir / "vector_cache.sqlite")
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
    custom_kg_incremental.write_manifest(state_dir, manifest)
    monkeypatch.setattr(custom_kg_incremental, "build_desired_manifest", lambda *_args, **_kwargs: (manifest, {"chunks": 1, "entities": 2, "relationships": 1}))
    args = types.SimpleNamespace(
        root=root,
        state_dir=state_dir,
        workdir=workdir,
        vector_cache=state_dir / "vector_cache.sqlite",
        storage_dir=shadow,
        delete_shadow_on_no_swap=False,
        limit_docs=None,
        limit_edges=None,
        prepare_swap=True,
        seed_from_storage=False,
        seed_storage_dir=None,
        fill_missing_vectors=False,
        smoke_query=[],
        allow_current_storage_audit_failure=True,
    )

    report = custom_kg_incremental.run_full_materialization_no_swap(args)

    assert report["prepared_for_swap"] is True
    assert report["pre_audit"]["ok"] is False
    assert report["shadow_audit"]["ok"] is True


def test_finalize_prepared_swap_can_replace_current_storage_that_fails_audit(tmp_path, monkeypatch) -> None:
    manifest = build_custom_kg_manifest(_payload(), lightrag_version="1.5.0", embedding_model="embed-a", embedding_dim=2, embedding_params_version="v1")
    previous = json.loads(json.dumps(manifest))
    desired = json.loads(json.dumps(manifest))
    state_dir = tmp_path / "state"
    workdir = tmp_path / "workdir"
    root = tmp_path / "wiki"
    live_storage = workdir / "rag_storage"
    shadow_storage = tmp_path / "prepared_shadow"
    backup_dir = state_dir / "backups" / "bad-live-backup"
    state_dir.mkdir()
    workdir.mkdir()
    root.mkdir()
    live_storage.mkdir()
    shadow_storage.mkdir()
    (live_storage / "bad.txt").write_text("bad", encoding="utf-8")
    (shadow_storage / "shadow.txt").write_text("shadow", encoding="utf-8")
    custom_kg_incremental.write_manifest(state_dir, previous)
    prepared_manifest = custom_kg_incremental.prepared_swap_manifest_path(state_dir)
    prepared_manifest.parent.mkdir(parents=True, exist_ok=True)
    prepared_manifest.write_text(json.dumps(desired), encoding="utf-8")
    prepared_report = custom_kg_incremental.prepared_swap_report_path(state_dir)
    prepared_report.write_text(
        json.dumps(
            {
                "started_at": "2026-06-26 00:00",
                "prepared_for_swap": True,
                "import_mode": "full_materialization",
                "previous_manifest_hash": custom_kg_incremental.stable_hash(previous),
                "desired_manifest_hash": custom_kg_incremental.stable_hash(desired),
                "desired_manifest_path": str(prepared_manifest),
                "shadow_storage": str(shadow_storage),
                "backup_dir": str(backup_dir),
                "payload": {},
                "manifest": desired.get("summary", {}),
                "shadow_audit": {"ok": True, "issues": [], "counts": {}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(custom_kg_incremental, "port_open", lambda *_args, **_kwargs: False)

    def fake_audit(storage_dir, *_args, **_kwargs):
        if Path(storage_dir) == live_storage:
            return {"ok": False, "issues": [{"type": "known_bad_live"}], "counts": {}}
        return {"ok": True, "issues": [], "counts": {}}

    monkeypatch.setattr(custom_kg_incremental, "audit_custom_kg_storage", fake_audit)
    args = types.SimpleNamespace(
        root=root,
        state_dir=state_dir,
        workdir=workdir,
        prepared_report=prepared_report,
        server_host="127.0.0.1",
        server_port=9621,
        allow_server_running=False,
        allow_current_storage_audit_failure=True,
    )

    report = custom_kg_incremental.run_finalize_prepared_swap(args)

    assert report["swapped"] is True
    assert report["live_audit"]["ok"] is False
    assert (workdir / "rag_storage" / "shadow.txt").exists()
    assert (backup_dir / "bad.txt").exists()


def test_run_full_materialization_no_swap_can_smoke_query_shadow(monkeypatch, tmp_path) -> None:
    manifest = build_custom_kg_manifest(_payload(), lightrag_version="1.5.0", embedding_model="embed-a", embedding_dim=2, embedding_params_version="v1")
    cache = VectorCache(tmp_path / "cache.sqlite")
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
    state_dir = tmp_path / "state"
    workdir = tmp_path / "workdir"
    root = tmp_path / "wiki"
    shadow = tmp_path / "shadow_full"
    state_dir.mkdir()
    workdir.mkdir()
    root.mkdir()
    monkeypatch.setattr(custom_kg_incremental, "build_desired_manifest", lambda *_args, **_kwargs: (manifest, {"chunks": 1, "entities": 2, "relationships": 1}))
    called = {}

    def fake_smoke(*, workdir, storage_dir, queries, mode, top_k, chunk_top_k):
        called["workdir"] = workdir
        called["storage_dir"] = storage_dir
        called["queries"] = list(queries)
        called["mode"] = mode
        called["top_k"] = top_k
        called["chunk_top_k"] = chunk_top_k
        assert Path(storage_dir).exists()
        return {"ok": True, "queries": [{"query": "graph retrieval", "ok": True, "counts": {"chunks": 1}}]}

    monkeypatch.setattr(custom_kg_incremental, "run_shadow_query_data_smokes", fake_smoke)
    args = types.SimpleNamespace(
        root=root,
        state_dir=state_dir,
        workdir=workdir,
        vector_cache=tmp_path / "cache.sqlite",
        storage_dir=shadow,
        delete_shadow_on_no_swap=True,
        limit_docs=None,
        limit_edges=None,
        seed_from_storage=False,
        seed_storage_dir=None,
        smoke_query=["graph retrieval"],
        smoke_mode="mix",
        smoke_top_k=3,
        smoke_chunk_top_k=4,
    )

    report = custom_kg_incremental.run_full_materialization_no_swap(args)

    assert report["query_smoke"]["ok"] is True
    assert called == {
        "workdir": workdir.resolve(),
        "storage_dir": shadow.resolve(),
        "queries": ["graph retrieval"],
        "mode": "mix",
        "top_k": 3,
        "chunk_top_k": 4,
    }
    assert report["shadow_deleted"] is True
    assert not shadow.exists()


def test_prepared_shadow_fingerprint_verifies_and_rejects_modified_storage_file(tmp_path) -> None:
    manifest = build_custom_kg_manifest(_payload(), lightrag_version="1.5.0", embedding_model="embed-a", embedding_dim=2, embedding_params_version="v1")
    storage_dir = tmp_path / "prepared_shadow"
    _materialize_manifest_storage(manifest, storage_dir, tmp_path / "cache.sqlite")
    audit = audit_custom_kg_storage(storage_dir, manifest)
    desired_hash = custom_kg_incremental.stable_hash(manifest)

    fingerprint = custom_kg_incremental.prepared_shadow_fingerprint(storage_dir, desired_hash, audit)
    prepared_report = {
        "shadow_storage": str(storage_dir),
        "desired_manifest_hash": desired_hash,
        "shadow_audit": audit,
        "prepared_shadow_fingerprint": fingerprint,
    }

    assert custom_kg_incremental.verify_prepared_shadow_fingerprint(prepared_report)["ok"] is True

    vdb_chunks = storage_dir / "vdb_chunks.json"
    vdb_chunks.write_text(vdb_chunks.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    verification = custom_kg_incremental.verify_prepared_shadow_fingerprint(prepared_report)
    assert verification["ok"] is False
    assert "changed_file:vdb_chunks.json" in verification["reasons"]


def test_prepared_shadow_fingerprint_rejects_desired_manifest_hash_mismatch(tmp_path) -> None:
    manifest = build_custom_kg_manifest(_payload(), lightrag_version="1.5.0", embedding_model="embed-a", embedding_dim=2, embedding_params_version="v1")
    storage_dir = tmp_path / "prepared_shadow"
    _materialize_manifest_storage(manifest, storage_dir, tmp_path / "cache.sqlite")
    audit = audit_custom_kg_storage(storage_dir, manifest)
    desired_hash = custom_kg_incremental.stable_hash(manifest)

    prepared_report = {
        "shadow_storage": str(storage_dir),
        "desired_manifest_hash": "different-manifest-hash",
        "shadow_audit": audit,
        "prepared_shadow_fingerprint": custom_kg_incremental.prepared_shadow_fingerprint(storage_dir, desired_hash, audit),
    }

    verification = custom_kg_incremental.verify_prepared_shadow_fingerprint(prepared_report)
    assert verification["ok"] is False
    assert "desired_manifest_hash_mismatch" in verification["reasons"]

    malformed_report = copy.deepcopy(prepared_report)
    malformed_report["desired_manifest_hash"] = desired_hash
    malformed_report["prepared_shadow_fingerprint"]["shadow_audit"]["issue_count"] = "not-an-int"
    malformed_verification = custom_kg_incremental.verify_prepared_shadow_fingerprint(malformed_report)
    assert malformed_verification["ok"] is False
    assert "prepared_shadow_audit_not_clean" in malformed_verification["reasons"]


def test_prepare_swap_report_contains_prepared_shadow_fingerprint(monkeypatch, tmp_path) -> None:
    manifest = build_custom_kg_manifest(_payload(), lightrag_version="1.5.0", embedding_model="embed-a", embedding_dim=2, embedding_params_version="v1")
    state_dir = tmp_path / "state"
    workdir = tmp_path / "workdir"
    root = tmp_path / "wiki"
    live_storage = workdir / "rag_storage"
    state_dir.mkdir()
    workdir.mkdir()
    root.mkdir()
    cache = _materialize_manifest_storage(manifest, live_storage, tmp_path / "cache.sqlite")
    custom_kg_incremental.write_manifest(state_dir, manifest)
    monkeypatch.setattr(custom_kg_incremental, "build_desired_manifest", lambda *_args, **_kwargs: (manifest, {"chunks": 1, "entities": 2, "relationships": 1}))
    args = types.SimpleNamespace(
        root=root,
        state_dir=state_dir,
        workdir=workdir,
        vector_cache=cache.path,
        storage_dir=tmp_path / "prepared_shadow",
        delete_shadow_on_no_swap=False,
        limit_docs=None,
        limit_edges=None,
        seed_from_storage=False,
        seed_storage_dir=None,
        fill_missing_vectors=False,
        prepare_swap=True,
        allow_current_storage_audit_failure=False,
        smoke_query=[],
        smoke_mode="mix",
        smoke_top_k=5,
        smoke_chunk_top_k=5,
    )

    report = custom_kg_incremental.run_full_materialization_no_swap(args)
    persisted = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))

    assert report["prepared_shadow_fingerprint"]["desired_manifest_hash"] == report["desired_manifest_hash"]
    assert persisted["prepared_shadow_fingerprint"] == report["prepared_shadow_fingerprint"]
    assert custom_kg_incremental.verify_prepared_shadow_fingerprint(persisted)["ok"] is True


def test_finalize_prepared_swap_reuses_verified_shadow_audit_but_keeps_live_audit(monkeypatch, tmp_path) -> None:
    fixture = _prepared_finalize_fixture(tmp_path, backup_name="reuse-backup")
    calls: list[Path] = []

    def fake_audit(storage_dir, _manifest, **_kwargs):
        calls.append(Path(storage_dir).resolve())
        return {"ok": True, "issues": [], "counts": {}}

    monkeypatch.setattr(custom_kg_incremental, "port_open", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(custom_kg_incremental, "audit_custom_kg_storage", fake_audit)
    args = types.SimpleNamespace(root=fixture.root, state_dir=fixture.state_dir, workdir=fixture.workdir, prepared_report=fixture.report_path, server_host="127.0.0.1", server_port=9621, allow_server_running=False, force_shadow_audit=False)

    report = custom_kg_incremental.run_finalize_prepared_swap(args)

    assert calls == [fixture.live_storage.resolve()]
    assert report["shadow_audit_reused"] is True
    assert report["shadow_audit_reuse"]["reason"] == "fingerprint_verified"
    assert report["live_audit"]["ok"] is True
    assert report["shadow_audit"] == fixture.shadow_audit


def test_finalize_prepared_swap_reruns_shadow_audit_when_fingerprint_changes(monkeypatch, tmp_path) -> None:
    fixture = _prepared_finalize_fixture(tmp_path, backup_name="fallback-backup")
    vdb_chunks = fixture.shadow_storage / "vdb_chunks.json"
    vdb_chunks.write_text(vdb_chunks.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    calls: list[Path] = []

    def fake_audit(storage_dir, _manifest, **_kwargs):
        calls.append(Path(storage_dir).resolve())
        return {"ok": True, "issues": [], "counts": {"path": Path(storage_dir).name}}

    monkeypatch.setattr(custom_kg_incremental, "port_open", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(custom_kg_incremental, "audit_custom_kg_storage", fake_audit)
    args = types.SimpleNamespace(root=fixture.root, state_dir=fixture.state_dir, workdir=fixture.workdir, prepared_report=fixture.report_path, server_host="127.0.0.1", server_port=9621, allow_server_running=False, force_shadow_audit=False)

    report = custom_kg_incremental.run_finalize_prepared_swap(args)

    assert calls == [fixture.live_storage.resolve(), fixture.shadow_storage.resolve()]
    assert report["shadow_audit_reused"] is False
    assert "changed_file:vdb_chunks.json" in report["shadow_audit_reuse"]["verification"]["reasons"]


def test_finalize_prepared_swap_force_shadow_audit_uses_old_path(monkeypatch, tmp_path) -> None:
    fixture = _prepared_finalize_fixture(tmp_path, backup_name="forced-backup")
    calls: list[Path] = []

    def fake_audit(storage_dir, _manifest, **_kwargs):
        calls.append(Path(storage_dir).resolve())
        return {"ok": True, "issues": [], "counts": {}}

    monkeypatch.setattr(custom_kg_incremental, "port_open", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(custom_kg_incremental, "audit_custom_kg_storage", fake_audit)
    args = types.SimpleNamespace(root=fixture.root, state_dir=fixture.state_dir, workdir=fixture.workdir, prepared_report=fixture.report_path, server_host="127.0.0.1", server_port=9621, allow_server_running=False, force_shadow_audit=True)

    report = custom_kg_incremental.run_finalize_prepared_swap(args)

    assert calls == [fixture.live_storage.resolve(), fixture.shadow_storage.resolve()]
    assert report["shadow_audit_reused"] is False
    assert report["shadow_audit_reuse"]["reason"] == "force_shadow_audit"


def test_custom_kg_incremental_materialize_full_cli_accepts_embedding_profile(monkeypatch, tmp_path, capsys) -> None:
    called = {}

    def fake_runner(args):
        called["embedding_profile"] = args.embedding_profile
        return {"ok": True, "import_mode": "full_materialization", "embedding_profile": args.embedding_profile}

    monkeypatch.setattr(custom_kg_incremental, "run_full_materialization_no_swap", fake_runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "custom_kg_incremental.py",
            "materialize-full",
            "--root",
            str(tmp_path / "wiki"),
            "--state-dir",
            str(tmp_path / "state"),
            "--workdir",
            str(tmp_path / "work"),
            "--no-swap",
            "--embedding-profile",
            "shadow-medium",
        ],
    )

    assert custom_kg_incremental.main() == 0
    out = json.loads(capsys.readouterr().out)
    assert called["embedding_profile"] == "shadow-medium"
    assert out["embedding_profile"] == "shadow-medium"


def test_custom_kg_incremental_materialize_full_cli_routes_to_no_swap_runner(monkeypatch, tmp_path, capsys) -> None:
    called = {}

    def fake_runner(args):
        called["no_swap"] = args.no_swap
        called["vector_cache"] = args.vector_cache
        called["storage_dir"] = args.storage_dir
        return {"ok": True, "import_mode": "full_materialization", "swapped": False}

    monkeypatch.setattr(custom_kg_incremental, "run_full_materialization_no_swap", fake_runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "custom_kg_incremental.py",
            "materialize-full",
            "--root",
            str(tmp_path / "wiki"),
            "--state-dir",
            str(tmp_path / "state"),
            "--workdir",
            str(tmp_path / "work"),
            "--vector-cache",
            str(tmp_path / "state" / "vector_cache.sqlite"),
            "--storage-dir",
            str(tmp_path / "shadow"),
            "--no-swap",
            "--delete-shadow-on-no-swap",
        ],
    )

    assert custom_kg_incremental.main() == 0
    out = json.loads(capsys.readouterr().out)
    assert out["import_mode"] == "full_materialization"
    assert out["swapped"] is False
    assert called["no_swap"] is True
    assert called["vector_cache"].name == "vector_cache.sqlite"
    assert called["storage_dir"].name == "shadow"


def test_custom_kg_incremental_materialize_full_cli_accepts_prepare_swap(monkeypatch, tmp_path, capsys) -> None:
    called = {}

    def fake_runner(args):
        called["prepare_swap"] = args.prepare_swap
        called["delete_shadow_on_no_swap"] = args.delete_shadow_on_no_swap
        called["allow_current_storage_audit_failure"] = args.allow_current_storage_audit_failure
        return {"ok": True, "import_mode": "full_materialization", "swapped": False, "prepared_for_swap": args.prepare_swap}

    monkeypatch.setattr(custom_kg_incremental, "run_full_materialization_no_swap", fake_runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "custom_kg_incremental.py",
            "materialize-full",
            "--root",
            str(tmp_path / "wiki"),
            "--state-dir",
            str(tmp_path / "state"),
            "--workdir",
            str(tmp_path / "work"),
            "--no-swap",
            "--prepare-swap",
            "--allow-current-storage-audit-failure",
        ],
    )

    assert custom_kg_incremental.main() == 0
    out = json.loads(capsys.readouterr().out)
    assert out["prepared_for_swap"] is True
    assert called == {
        "prepare_swap": True,
        "delete_shadow_on_no_swap": False,
        "allow_current_storage_audit_failure": True,
    }


def test_finalize_prepared_swap_preserves_full_materialization_mode_and_resets_counter(tmp_path, monkeypatch) -> None:
    manifest = build_custom_kg_manifest(_payload(), lightrag_version="1.5.0", embedding_model="embed-a", embedding_dim=2, embedding_params_version="v1")
    previous = json.loads(json.dumps(manifest))
    previous["metadata"]["incremental_count_since_full"] = 5
    desired = json.loads(json.dumps(manifest))
    state_dir = tmp_path / "state"
    workdir = tmp_path / "workdir"
    root = tmp_path / "wiki"
    live_storage = workdir / "rag_storage"
    shadow_storage = tmp_path / "prepared_shadow"
    backup_dir = state_dir / "backups" / "full-materialization-backup"
    state_dir.mkdir()
    workdir.mkdir()
    root.mkdir()
    live_storage.mkdir()
    shadow_storage.mkdir()
    (live_storage / "live.txt").write_text("live", encoding="utf-8")
    (shadow_storage / "shadow.txt").write_text("shadow", encoding="utf-8")
    custom_kg_incremental.write_manifest(state_dir, previous)
    prepared_manifest = custom_kg_incremental.prepared_swap_manifest_path(state_dir)
    prepared_manifest.parent.mkdir(parents=True, exist_ok=True)
    prepared_manifest.write_text(json.dumps(desired), encoding="utf-8")
    prepared_report = custom_kg_incremental.prepared_swap_report_path(state_dir)
    prepared_report.write_text(
        json.dumps(
            {
                "started_at": "2026-06-26 00:00",
                "prepared_for_swap": True,
                "import_mode": "full_materialization",
                "previous_manifest_hash": custom_kg_incremental.stable_hash(previous),
                "desired_manifest_hash": custom_kg_incremental.stable_hash(desired),
                "desired_manifest_path": str(prepared_manifest),
                "shadow_storage": str(shadow_storage),
                "backup_dir": str(backup_dir),
                "payload": {},
                "manifest": desired.get("summary", {}),
                "shadow_audit": {"ok": True, "issues": [], "counts": {}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(custom_kg_incremental, "port_open", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(custom_kg_incremental, "audit_custom_kg_storage", lambda *_args, **_kwargs: {"ok": True, "issues": [], "counts": {}})
    args = types.SimpleNamespace(root=root, state_dir=state_dir, workdir=workdir, prepared_report=prepared_report, server_host="127.0.0.1", server_port=9621, allow_server_running=False)

    report = custom_kg_incremental.run_finalize_prepared_swap(args)

    final_manifest = custom_kg_incremental.load_manifest(state_dir)
    assert report["swapped"] is True
    assert report["import_mode"] == "full_materialization"
    assert final_manifest["metadata"]["last_successful_import_mode"] == "full_materialization"
    assert final_manifest["metadata"]["incremental_count_since_full"] == 0
    assert (workdir / "rag_storage" / "shadow.txt").exists()
    assert (backup_dir / "live.txt").exists()


def test_run_full_materialization_no_swap_can_seed_cache_from_explicit_storage(monkeypatch, tmp_path) -> None:
    manifest = build_custom_kg_manifest(_payload(), lightrag_version="1.5.0", embedding_model="embed-a", embedding_dim=2, embedding_params_version="v1")
    seed_storage = tmp_path / "seed_storage"
    source_cache = VectorCache(tmp_path / "source_cache.sqlite")
    ordinal = 1
    for collection in ("chunks", "entities", "relationships"):
        for record in manifest[collection].values():
            source_cache.put(
                record["vector_hash"],
                record_type=record["record_type"],
                record_id=record["record_id"],
                embedding_model=record["embedding_model"],
                embedding_dim=record["embedding_dim"],
                embedding_params_version=record["embedding_params_version"],
                vector=[float(ordinal), float(ordinal + 1)],
            )
            ordinal += 1
    materialize_file_storage_from_manifest(manifest, resolve_manifest_vectors(manifest, source_cache)["resolved"], seed_storage)
    state_dir = tmp_path / "state"
    workdir = tmp_path / "workdir"
    root = tmp_path / "wiki"
    state_dir.mkdir()
    workdir.mkdir()
    root.mkdir()
    custom_kg_incremental.write_manifest(state_dir, manifest)
    monkeypatch.setattr(custom_kg_incremental, "build_desired_manifest", lambda *_args, **_kwargs: (manifest, {"chunks": 1, "entities": 2, "relationships": 1}))
    args = types.SimpleNamespace(
        root=root,
        state_dir=state_dir,
        workdir=workdir,
        vector_cache=tmp_path / "fresh_cache.sqlite",
        storage_dir=tmp_path / "shadow_full",
        delete_shadow_on_no_swap=False,
        limit_docs=None,
        limit_edges=None,
        seed_from_storage=True,
        seed_storage_dir=seed_storage,
    )

    report = custom_kg_incremental.run_full_materialization_no_swap(args)

    assert report["vector_cache_seed"]["summary"]["total"] == {"total": 4, "seeded": 4, "missing": 0}
    assert report["vector_cache"]["summary"]["total"] == {"total": 4, "hits": 4, "misses": 0}
    assert report["shadow_audit"]["ok"] is True
