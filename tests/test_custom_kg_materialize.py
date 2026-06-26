from array import array
import base64
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
        ],
    )

    assert custom_kg_incremental.main() == 0
    out = json.loads(capsys.readouterr().out)
    assert out["prepared_for_swap"] is True
    assert called == {"prepare_swap": True, "delete_shadow_on_no_swap": False}


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
