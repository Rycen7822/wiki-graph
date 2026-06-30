import copy
import json
import sys
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import custom_kg_incremental  # noqa: E402
import custom_kg_vector_fill  # noqa: E402
from custom_kg_incremental import build_custom_kg_manifest  # noqa: E402
from custom_kg_materialize import materialize_file_storage_from_manifest  # noqa: E402
from vector_cache import VectorCache  # noqa: E402


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


def test_run_export_manifest_writes_manifest_without_storage_mutation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest = build_custom_kg_manifest(_payload(), native_manifest_tool_version="1.5.0", embedding_model="embed-a", embedding_dim=2)
    state_dir = tmp_path / "state"
    workdir = tmp_path / "work"
    root = tmp_path / "wiki"

    monkeypatch.setattr(
        custom_kg_incremental,
        "build_desired_manifest",
        lambda *_args, **_kwargs: (manifest, {"chunks": 1, "entities": 2, "relationships": 1}),
    )

    report = custom_kg_incremental.run_export_manifest(
        types.SimpleNamespace(root=root, state_dir=state_dir, workdir=workdir, limit_docs=None, limit_edges=None)
    )

    manifest_path = state_dir / "custom_kg_manifest.json"
    assert report["ok"] is True
    assert report["command"] == "export-manifest"
    assert report["manifest_path"] == str(manifest_path)
    assert custom_kg_incremental.load_manifest(state_dir) == manifest
    assert not workdir.exists()
    assert not (state_dir / "prepared_swap").exists()


def test_run_export_manifest_refuses_unexpected_manifest_metadata_without_write(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest = build_custom_kg_manifest(_payload(), native_manifest_tool_version="1.5.0", embedding_model="embed-a", embedding_dim=2)
    manifest["metadata"]["unexpected_tool_version"] = "old"
    state_dir = tmp_path / "state"

    monkeypatch.setattr(
        custom_kg_incremental,
        "build_desired_manifest",
        lambda *_args, **_kwargs: (manifest, {"chunks": 1, "entities": 2, "relationships": 1}),
    )

    with pytest.raises(RuntimeError, match="native manifest metadata contract") as exc_info:
        custom_kg_incremental.run_export_manifest(
            types.SimpleNamespace(
                root=tmp_path / "wiki",
                state_dir=state_dir,
                workdir=tmp_path / "work",
                limit_docs=None,
                limit_edges=None,
            )
        )

    assert "unexpected_tool_version" in str(exc_info.value)
    assert not (state_dir / "custom_kg_manifest.json").exists()
    assert not state_dir.exists()


def test_run_audit_manifest_content_reports_native_contract_issues_without_write(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest = build_custom_kg_manifest(_payload(), native_manifest_tool_version="1.5.0", embedding_model="embed-a", embedding_dim=2)
    manifest["metadata"]["canonical_id_algorithm"] = "llm-wiki-canonical-id:v0"
    state_dir = tmp_path / "state"
    workdir = tmp_path / "work"

    monkeypatch.setattr(
        custom_kg_incremental,
        "build_desired_manifest",
        lambda *_args, **_kwargs: (manifest, {"chunks": 1, "entities": 2, "relationships": 1}),
    )

    report = custom_kg_incremental.run_audit_manifest_content(
        types.SimpleNamespace(root=tmp_path / "wiki", state_dir=state_dir, workdir=workdir, limit_docs=None, limit_edges=None)
    )

    assert report["ok"] is False
    assert report["command"] == "audit-manifest-content"
    assert report["issue_count"] == 1
    assert report["issues"] == [
        {"type": "invalid_manifest_metadata_value", "path": "metadata.canonical_id_algorithm"}
    ]
    assert not state_dir.exists()
    assert not workdir.exists()


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


# Prepared shadow/swap internals were removed with the retired live-storage runner.


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


def test_openai_compatible_vector_fill_provider_loads_workdir_env(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "EMBEDDING_BINDING=openai",
                "EMBEDDING_BINDING_HOST=https://embedding.local/v1",
                "EMBEDDING_BINDING_API_KEY=secret",
                "EMBEDDING_MODEL=BAAI/bge-m3",
                "EMBEDDING_DIM=2",
                "EMBEDDING_TIMEOUT=17",
            ]
        ),
        encoding="utf-8",
    )
    for name in ("EMBEDDING_BINDING_HOST", "EMBEDDING_BINDING_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _tb) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"data": [{"index": 0, "embedding": [0.1, 0.2]}]}).encode("utf-8")

    def fake_urlopen(request, timeout):
        calls.append({"request": request, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr(custom_kg_vector_fill.urllib.request, "urlopen", fake_urlopen)

    vectors = custom_kg_vector_fill.embed_texts_openai_compatible(
        ["Doc A content"],
        workdir=tmp_path,
        embedding_model="BAAI/bge-m3",
        embedding_dim=2,
    )

    assert vectors == [[0.1, 0.2]]
    assert calls[0]["timeout"] == 17
    assert calls[0]["request"].full_url == "https://embedding.local/v1/embeddings"
    assert calls[0]["request"].headers["Authorization"] == "Bearer secret"
    assert json.loads(calls[0]["request"].data.decode("utf-8")) == {"model": "BAAI/bge-m3", "input": ["Doc A content"]}


def test_fill_missing_manifest_vectors_reports_redacted_embedding_env(tmp_path) -> None:
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "EMBEDDING_BINDING=openai",
                "EMBEDDING_BINDING_HOST=https://embedding.local/v1",
                "EMBEDDING_BINDING_API_KEY=secret",
                "EMBEDDING_MODEL=BAAI/bge-m3",
                "EMBEDDING_DIM=2",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "metadata": {},
        "chunks": {
            "chunk:a": {
                "record_type": "chunk",
                "record_id": "chunk:a",
                "vector_hash": "hash-a",
                "content": "Doc A content",
                "embedding_params_version": "v1",
            }
        },
        "entities": {},
        "relationships": {},
    }
    vector_report = {"missing": {"chunks": ["chunk:a"], "entities": [], "relationships": []}}
    cache = VectorCache(tmp_path / "cache.sqlite")

    report = custom_kg_vector_fill.fill_missing_manifest_vectors(
        manifest,
        vector_report,
        cache,
        workdir=tmp_path,
        embed_texts_func=lambda texts, **_kwargs: [[0.1, 0.2]],
    )

    assert report["embedding_model"] == "BAAI/bge-m3"
    assert report["embedding_dim"] == 2
    assert report["embedding_env"]["EMBEDDING_BINDING_API_KEY"] == "[REDACTED]"
    assert report["embedding_env"]["EMBEDDING_BINDING_HOST"] == "https://embedding.local/v1"
    assert "secret" not in json.dumps(report, ensure_ascii=False)


def test_relationship_vector_content_uses_typed_directed_endpoint_order() -> None:
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

    manifest = build_custom_kg_manifest(payload, native_manifest_tool_version="1.5.0", embedding_model="embed-a", embedding_dim=2, embedding_params_version="v1")
    relationship = next(iter(manifest["relationships"].values()))

    assert relationship["src_id"] == "topic:z"
    assert relationship["tgt_id"] == "doc:a"
    assert relationship["content"] == "SOURCED_BY\ttopic:z\ndoc:a\ntopic:z SOURCED_BY doc:a"


def test_full_materialization_blocker_treats_endpoint_order_content_change_as_vector_update() -> None:
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
    desired = build_custom_kg_manifest(payload, native_manifest_tool_version="1.5.0", embedding_model="embed-a", embedding_dim=2, embedding_params_version="v1")
    previous = copy.deepcopy(desired)
    key, previous_relationship = next(iter(previous["relationships"].items()))
    previous_relationship["content"] = "SOURCED_BY\tdoc:a\ntopic:z\ntopic:z SOURCED_BY doc:a"
    custom_kg_incremental._stamp_identity_and_hashes(
        previous_relationship,
        record_id=previous_relationship["vdb_id"],
        canonical_id=key,
    )

    blockers = custom_kg_incremental.full_materialization_cache_only_blockers(previous, desired)

    assert blockers["blocked"] is True
    assert blockers["collections"] == {"relationships": {"add": 0, "vector_update": 1}}
    assert blockers["diff"]["relationships"]["vector_update"] == 1
    assert blockers["diff"]["relationships"]["metadata_update"] == 0


def test_materialize_file_storage_from_manifest_is_retired_without_writing_storage(tmp_path) -> None:
    manifest = build_custom_kg_manifest(_payload(), native_manifest_tool_version="1.5.0", embedding_model="embed-a", embedding_dim=2)
    storage_dir = tmp_path / "shadow_storage"

    with pytest.raises(RuntimeError, match="retired"):
        materialize_file_storage_from_manifest(manifest, {"chunks": {}, "entities": {}, "relationships": {}}, storage_dir)

    assert not storage_dir.exists()


# Retired full-materialization runner behavior is covered by direct fail-closed tests below; old file-storage writer helpers fail closed.


def test_run_finalize_prepared_swap_is_retired_without_reading_prepared_bundle(tmp_path) -> None:
    args = types.SimpleNamespace(
        root=tmp_path / "wiki",
        state_dir=tmp_path / "state",
        workdir=tmp_path / "workdir",
        prepared_report=tmp_path / "state" / "prepared.json",
        server_host="127.0.0.1",
        server_port=9621,
        allow_server_running=False,
        allow_current_storage_audit_failure=True,
        force_shadow_audit=False,
    )

    with pytest.raises(RuntimeError, match="prepared wikigraph storage activation is retired"):
        custom_kg_incremental.run_finalize_prepared_swap(args)

    assert not args.workdir.exists()
    assert not args.prepared_report.exists()


def test_custom_kg_incremental_materialize_full_cli_is_retired(monkeypatch, tmp_path, capsys) -> None:
    called = {}

    def fake_runner(args):
        called["materialize_full"] = True
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
            "--prepare-swap",
            "--allow-current-storage-audit-failure",
        ],
    )

    assert custom_kg_incremental.main() == 1
    out = json.loads(capsys.readouterr().out)
    assert out["error"] == "RuntimeError"
    assert "materialize-full CLI is retired" in out["message"]
    assert called == {}


def test_custom_kg_incremental_export_manifest_cli_routes_to_export_runner(monkeypatch, tmp_path, capsys) -> None:
    captured = {}

    def fake_runner(args):
        captured.update(
            {
                "root": args.root,
                "state_dir": args.state_dir,
                "workdir": args.workdir,
                "limit_docs": args.limit_docs,
                "limit_edges": args.limit_edges,
            }
        )
        return {"ok": True, "command": "export-manifest", "manifest_path": str(tmp_path / "state" / "custom_kg_manifest.json")}

    monkeypatch.setattr(custom_kg_incremental, "run_export_manifest", fake_runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "custom_kg_incremental.py",
            "export-manifest",
            "--root",
            str(tmp_path / "wiki"),
            "--state-dir",
            str(tmp_path / "state"),
            "--workdir",
            str(tmp_path / "work"),
            "--limit-docs",
            "2",
            "--limit-edges",
            "3",
        ],
    )

    assert custom_kg_incremental.main() == 0
    assert captured == {
        "root": tmp_path / "wiki",
        "state_dir": tmp_path / "state",
        "workdir": tmp_path / "work",
        "limit_docs": 2,
        "limit_edges": 3,
    }
    assert json.loads(capsys.readouterr().out)["command"] == "export-manifest"


def test_custom_kg_incremental_audit_manifest_content_cli_routes_to_audit_runner(monkeypatch, tmp_path, capsys) -> None:
    captured = {}

    def fake_runner(args):
        captured.update(
            {
                "root": args.root,
                "state_dir": args.state_dir,
                "workdir": args.workdir,
                "limit_docs": args.limit_docs,
                "limit_edges": args.limit_edges,
            }
        )
        return {"ok": False, "command": "audit-manifest-content", "token_variant_count": 1, "sources": []}

    monkeypatch.setattr(custom_kg_incremental, "run_audit_manifest_content", fake_runner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "custom_kg_incremental.py",
            "audit-manifest-content",
            "--root",
            str(tmp_path / "wiki"),
            "--state-dir",
            str(tmp_path / "state"),
            "--workdir",
            str(tmp_path / "work"),
            "--limit-docs",
            "2",
            "--limit-edges",
            "3",
        ],
    )

    assert custom_kg_incremental.main() == 1
    assert captured == {
        "root": tmp_path / "wiki",
        "state_dir": tmp_path / "state",
        "workdir": tmp_path / "work",
        "limit_docs": 2,
        "limit_edges": 3,
    }
    assert json.loads(capsys.readouterr().out)["command"] == "audit-manifest-content"


def test_custom_kg_incremental_apply_cli_is_retired(monkeypatch, tmp_path, capsys) -> None:
    called = {}

    async def fake_apply(_args):
        called["apply"] = True
        return {"swapped": True}

    monkeypatch.setattr(custom_kg_incremental, "run_apply", fake_apply)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "custom_kg_incremental.py",
            "apply",
            "--root",
            str(tmp_path / "wiki"),
            "--state-dir",
            str(tmp_path / "state"),
            "--workdir",
            str(tmp_path / "work"),
            "--no-swap",
        ],
    )

    assert custom_kg_incremental.main() == 1
    out = json.loads(capsys.readouterr().out)
    assert out["error"] == "RuntimeError"
    assert "apply CLI is retired" in out["message"]
    assert called == {}


def test_custom_kg_incremental_finalize_prepared_swap_cli_is_retired(monkeypatch, tmp_path, capsys) -> None:
    called = {}

    def fake_finalize(_args):
        called["finalize"] = True
        return {"swapped": True}

    monkeypatch.setattr(custom_kg_incremental, "run_finalize_prepared_swap", fake_finalize)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "custom_kg_incremental.py",
            "finalize-prepared-swap",
            "--root",
            str(tmp_path / "wiki"),
            "--state-dir",
            str(tmp_path / "state"),
            "--workdir",
            str(tmp_path / "work"),
            "--prepared-report",
            str(tmp_path / "state" / "prepared.json"),
        ],
    )

    assert custom_kg_incremental.main() == 1
    out = json.loads(capsys.readouterr().out)
    assert out["error"] == "RuntimeError"
    assert "finalize-prepared-swap CLI is retired" in out["message"]
    assert called == {}


def test_custom_kg_incremental_direct_live_storage_runners_are_retired(tmp_path) -> None:
    import asyncio

    root = tmp_path / "wiki"
    state_dir = tmp_path / "state"
    workdir = tmp_path / "work"
    root.mkdir()
    state_dir.mkdir()
    workdir.mkdir()
    materialize_args = types.SimpleNamespace(
        root=root,
        state_dir=state_dir,
        workdir=workdir,
        vector_cache=tmp_path / "cache.sqlite",
        storage_dir=tmp_path / "shadow_full",
        delete_shadow_on_no_swap=False,
        limit_docs=None,
        limit_edges=None,
        seed_from_storage=False,
        seed_storage_dir=None,
        fill_missing_vectors=False,
        prepare_swap=False,
        allow_current_storage_audit_failure=False,
        smoke_query=[],
        smoke_mode="mix",
        smoke_top_k=5,
        smoke_chunk_top_k=5,
    )
    apply_args = types.SimpleNamespace(
        root=root,
        state_dir=state_dir,
        workdir=workdir,
        limit_docs=None,
        limit_edges=None,
        full_rebuild_interval=5,
        force_incremental=False,
        server_host="127.0.0.1",
        server_port=9621,
        allow_server_running=False,
        no_swap=True,
        prepare_swap=False,
        delete_shadow_on_no_swap=True,
        write_manifest_without_swap=False,
        tracking_update_mode="full",
    )

    with pytest.raises(RuntimeError, match="live-storage runner is retired"):
        custom_kg_incremental.run_full_materialization_no_swap(materialize_args)
    with pytest.raises(RuntimeError, match="live-storage runner is retired"):
        asyncio.run(custom_kg_incremental.run_apply(apply_args))
    with pytest.raises(RuntimeError, match="live-storage runner is retired"):
        asyncio.run(custom_kg_incremental.apply_patch_to_storage(tmp_path / "shadow", {}, {}, workdir=workdir))
    assert not (tmp_path / "shadow_full").exists()
    assert not (tmp_path / "shadow").exists()
    assert not (workdir / "rag_storage").exists()


def test_custom_kg_incremental_old_plan_and_storage_audit_cli_are_retired(monkeypatch, tmp_path, capsys) -> None:
    for command in ("plan", "audit-storage"):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "custom_kg_incremental.py",
                command,
                "--root",
                str(tmp_path / "wiki"),
                "--state-dir",
                str(tmp_path / "state"),
                "--workdir",
                str(tmp_path / "work"),
            ],
        )
        assert custom_kg_incremental.main() == 1
        out = json.loads(capsys.readouterr().out)
        assert out["error"] == "RuntimeError"
        assert "retired" in out["message"]
