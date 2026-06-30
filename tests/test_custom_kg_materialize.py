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
    old_medium_profile = "shadow" + "-medium"
    assert custom_kg_vector_fill.embedding_profile_env("conservative") == {
        "EMBEDDING_FUNC_MAX_ASYNC": "1",
        "EMBEDDING_BATCH_NUM": "10",
        "MAX_PARALLEL_INSERT": "1",
    }
    assert custom_kg_vector_fill.embedding_profile_env("balanced-medium")["EMBEDDING_BATCH_NUM"] == "20"
    with pytest.raises(ValueError, match="unknown embedding profile"):
        custom_kg_vector_fill.embedding_profile_env(old_medium_profile)
    with pytest.raises(ValueError, match="unknown embedding profile"):
        custom_kg_vector_fill.embedding_profile_env("surprise-fast")


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

    report = custom_kg_vector_fill.fill_missing_manifest_vectors(
        manifest,
        vector_report,
        cache,
        workdir=tmp_path,
        embed_texts_func=fake_embed,
        embedding_profile="balanced-medium",
    )

    assert report["embedding_profile"] == "balanced-medium"
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


def test_custom_kg_incremental_exposes_only_native_manifest_cli_surface() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts" / "custom_kg_incremental.py").read_text(encoding="utf-8")
    forbidden_symbols = [
        "def audit_custom_kg_storage(",
        "def apply_patch_to_storage(",
        "def run_apply(",
        "def run_full_materialization_no_swap(",
        "def run_finalize_prepared_swap(",
        "def embed_texts_openai_compatible(",
        "def fill_missing_manifest_vectors(",
        "def entity_vdb_id(",
        "def relation_vdb_id(",
        "def wikigraph_sanitize_text(",
        "def wikigraph_normalize_file_path(",
        "vdb_id",
        "VDB",
        "GraphML",
        "CUSTOM_KG_LIVE_STORAGE_RUNNER_RETIRED_MESSAGE",
        "PREPARED_WIKIGRAPH_SWAP_ACTIVATION_RETIRED_MESSAGE",
        "CUSTOM_KG_STORAGE_AUDIT_RETIRED_MESSAGE",
        "def diff_custom_kg_manifests(",
        "def full_materialization_cache_only_blockers(",
        "def compact_vector_cache_report(",
        "def choose_refresh_mode(",
        "def successful_manifest(",
        "def write_successful_manifest(",
        "def manifest_record_count(",
        "DEFAULT_FULL_REBUILD_INTERVAL",
        "def _diff_collection(",
        "incremental_count_since_full",
    ]
    assert [symbol for symbol in forbidden_symbols if symbol in source] == []

    parser_commands = [
        'sub.add_parser("plan"',
        'sub.add_parser("audit-storage"',
        'sub.add_parser("apply"',
        'sub.add_parser("finalize-prepared-swap"',
        'sub.add_parser("materialize-full"',
        "--allow-current-storage-audit-failure",
        "--seed-from-storage",
        "--tracking-update-mode",
    ]
    assert [symbol for symbol in parser_commands if symbol in source] == []
    assert 'sub.add_parser("export-manifest"' in source
    assert 'sub.add_parser("audit-manifest-content"' in source


def test_retired_custom_kg_file_storage_materializer_module_is_removed() -> None:
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "custom_kg_materialize.py"
    assert not module_path.exists()

    import importlib.util

    assert importlib.util.find_spec("custom_kg_materialize") is None


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
