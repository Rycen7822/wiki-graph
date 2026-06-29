import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_wiki_native.storage.sqlite_workspace import NativeRecord, SQLiteWorkspace

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import wiki_search  # noqa: E402


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.queries = []

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [1.0, 0.0]


def test_wiki_search_default_native_api_uses_server_without_query_vector(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_http_json(method, url, payload, *, timeout=60):
        calls.append({"method": method, "url": url, "payload": payload, "timeout": timeout})
        return {"context_blocks": [{"source_path": "alpha.md"}], "hits": []}

    monkeypatch.setattr(wiki_search, "http_json", fake_http_json)
    args = SimpleNamespace(
        backend="native",
        native_db=None,
        native_workspace=None,
        query_vector=None,
        mode="mix",
        top_k=3,
        chunk_top_k=2,
        section_kind="methodology",
        data_only=True,
        expand_section_neighbors=False,
        save_evidence_pack=False,
        state_dir=tmp_path,
        workdir=tmp_path,
        server="http://127.0.0.1:9621",
        neighbor_k=5,
    )

    result = wiki_search.run_query(args, "GraphRAG bottleneck")

    assert result["backend"] == "native"
    assert calls[0]["method"] == "POST"
    assert calls[0]["url"] == "http://127.0.0.1:9621/query/data"
    assert "query_vector" not in calls[0]["payload"]
    assert calls[0]["payload"]["section_kind"] == "methodology"
    assert result["response"]["context_blocks"][0]["source_path"] == "alpha.md"


def test_wiki_search_does_not_export_retired_query_helpers() -> None:
    retired = "light" + "rag"
    assert not hasattr(wiki_search, f"query_{retired}")
    assert not hasattr(wiki_search, f"query_{retired}_data")


def test_wiki_search_cli_help_is_native_only() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "wiki_search.py"), "--help"],
        check=True,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )

    assert "--backend" not in result.stdout
    assert ("light" + "rag") not in result.stdout.lower()


def test_wiki_search_source_has_no_retired_backend_branch() -> None:
    retired = "light" + "rag"
    text = (SCRIPTS / "wiki_search.py").read_text(encoding="utf-8")

    assert retired not in text.lower()
    assert f'choices=["native", "{retired}"]' not in text
    assert ("Light" + "RAG backend") not in text


def test_wiki_search_rejects_stale_non_native_backend_override(tmp_path) -> None:
    args = SimpleNamespace(
        backend="legacy",
        native_db=None,
        native_workspace=None,
        query_vector=None,
        mode="mix",
        top_k=3,
        chunk_top_k=2,
        section_kind=None,
        data_only=True,
        expand_section_neighbors=False,
        save_evidence_pack=False,
        state_dir=tmp_path,
        workdir=tmp_path,
        server="http://127.0.0.1:9621",
        neighbor_k=5,
    )

    with pytest.raises(ValueError, match="native-only"):
        wiki_search.run_query(args, "alpha")


def test_wiki_search_native_backend_is_explicit_and_default_off(tmp_path) -> None:
    db_path = tmp_path / "native.sqlite"
    db = SQLiteWorkspace(db_path)
    db.create_workspace("native-test", "manifest-hash")
    db.put_record(
        NativeRecord(
            workspace_id="native-test",
            record_type="entity",
            record_id="doc:a",
            vector_text="Alpha",
            content_hash="doc:a:content",
            metadata_hash="doc:a:metadata",
            vector_hash="doc:a:vector",
            source_path="alpha.md",
            source_id="doc:a",
            payload={"title": "Alpha"},
        )
    )
    db.put_vector("native-test", "entity", "doc:a", "doc:a:vector", [1.0, 0.0])
    db.mark_audited("native-test", {"chunks": 0, "entities": 1, "relationships": 0, "sections": 0}, require_vectors=True)
    args = SimpleNamespace(
        backend="native",
        native_db=db_path,
        native_workspace="native-test",
        query_vector=json.dumps([1.0, 0.0]),
        mode="mix",
        top_k=1,
        chunk_top_k=10,
        section_kind=None,
        data_only=True,
        expand_section_neighbors=False,
        save_evidence_pack=False,
        state_dir=tmp_path,
        workdir=tmp_path,
        server="http://127.0.0.1:9621",
        neighbor_k=5,
    )

    result = wiki_search.run_query(args, "alpha")

    assert result["backend"] == "native"
    assert result["response"]["context_blocks"][0]["source_path"] == "alpha.md"


def test_wiki_search_native_backend_can_embed_when_query_vector_is_absent(tmp_path) -> None:
    db_path = tmp_path / "native.sqlite"
    db = SQLiteWorkspace(db_path)
    db.create_workspace("native-test", "manifest-hash")
    db.put_record(
        NativeRecord(
            workspace_id="native-test",
            record_type="entity",
            record_id="doc:a",
            vector_text="Alpha",
            content_hash="doc:a:content",
            metadata_hash="doc:a:metadata",
            vector_hash="doc:a:vector",
            source_path="alpha.md",
            source_id="doc:a",
            payload={"title": "Alpha"},
        )
    )
    db.put_vector("native-test", "entity", "doc:a", "doc:a:vector", [1.0, 0.0])
    db.mark_audited("native-test", {"chunks": 0, "entities": 1, "relationships": 0, "sections": 0}, require_vectors=True)
    provider = FakeEmbeddingProvider()
    args = SimpleNamespace(
        backend="native",
        native_db=db_path,
        native_workspace="native-test",
        query_vector=None,
        mode="mix",
        top_k=1,
        chunk_top_k=10,
        section_kind=None,
        data_only=True,
        expand_section_neighbors=False,
        save_evidence_pack=False,
        state_dir=tmp_path,
        workdir=tmp_path,
        server="http://127.0.0.1:9621",
        neighbor_k=5,
    )

    result = wiki_search.run_query(args, "alpha", native_embedding_provider=provider)

    assert result["backend"] == "native"
    assert provider.queries == ["alpha"]
    assert result["response"]["context_blocks"][0]["source_path"] == "alpha.md"


def test_wiki_search_native_backend_prefers_explicit_query_vector(tmp_path) -> None:
    db_path = tmp_path / "native.sqlite"
    db = SQLiteWorkspace(db_path)
    db.create_workspace("native-test", "manifest-hash")
    db.put_record(
        NativeRecord(
            workspace_id="native-test",
            record_type="entity",
            record_id="doc:a",
            vector_text="Alpha",
            content_hash="doc:a:content",
            metadata_hash="doc:a:metadata",
            vector_hash="doc:a:vector",
            source_path="alpha.md",
            source_id="doc:a",
            payload={"title": "Alpha"},
        )
    )
    db.put_vector("native-test", "entity", "doc:a", "doc:a:vector", [1.0, 0.0])
    db.mark_audited("native-test", {"chunks": 0, "entities": 1, "relationships": 0, "sections": 0}, require_vectors=True)
    provider = FakeEmbeddingProvider()
    args = SimpleNamespace(
        backend="native",
        native_db=db_path,
        native_workspace="native-test",
        query_vector=json.dumps([1.0, 0.0]),
        mode="mix",
        top_k=1,
        chunk_top_k=10,
        section_kind=None,
        data_only=True,
        expand_section_neighbors=False,
        save_evidence_pack=False,
        state_dir=tmp_path,
        workdir=tmp_path,
        server="http://127.0.0.1:9621",
        neighbor_k=5,
    )

    result = wiki_search.run_query(args, "alpha", native_embedding_provider=provider)

    assert provider.queries == []
    assert result["response"]["context_blocks"][0]["source_path"] == "alpha.md"


def test_wiki_search_native_backend_requires_vector_or_embedding_provider(tmp_path) -> None:
    args = SimpleNamespace(
        backend="native",
        native_db=tmp_path / "native.sqlite",
        native_workspace="native-test",
        query_vector=None,
        mode="mix",
        top_k=1,
        chunk_top_k=10,
        section_kind=None,
        data_only=True,
        expand_section_neighbors=False,
        save_evidence_pack=False,
        state_dir=tmp_path,
        workdir=tmp_path,
        server="http://127.0.0.1:9621",
        neighbor_k=5,
    )

    with pytest.raises(ValueError, match="query-vector JSON or embedding provider"):
        wiki_search.run_query(args, "alpha")


def test_wiki_search_local_native_expands_sqlite_edge_neighbors(tmp_path) -> None:
    db_path = tmp_path / "native.sqlite"
    db = SQLiteWorkspace(db_path)
    db.create_workspace("native-test", "manifest-hash")
    db.put_record(
        NativeRecord(
            workspace_id="native-test",
            record_type="section",
            record_id="raw_section:a:future",
            vector_text="Alpha",
            content_hash="doc:a:content",
            metadata_hash="doc:a:metadata",
            vector_hash="doc:a:vector",
            source_path="alpha.md",
            source_id="raw_section:a:future",
            payload={"title": "Alpha"},
        )
    )
    db.put_record(
        NativeRecord(
            workspace_id="native-test",
            record_type="section",
            record_id="raw_section:b:future",
            vector_text="Beta",
            content_hash="doc:b:content",
            metadata_hash="doc:b:metadata",
            vector_hash="doc:b:vector",
            source_path="beta.md",
            source_id="raw_section:b:future",
            payload={"title": "Beta"},
        )
    )
    db.put_vector("native-test", "section", "raw_section:a:future", "doc:a:vector", [1.0, 0.0])
    db.put_vector("native-test", "section", "raw_section:b:future", "doc:b:vector", [0.0, 1.0])
    db.put_edge("native-test", "relationship", "raw_section:a:future", "raw_clip:a", 0.99, {"relation": "RAW_SECTION_OF"})
    db.put_edge("native-test", "section_similarity", "raw_section:a:future", "raw_section:b:future", 0.91, {"pair_kind": "future:future"})
    db.mark_audited("native-test", {"chunks": 0, "entities": 0, "relationships": 0, "sections": 2}, require_vectors=True)
    args = SimpleNamespace(
        backend="native",
        native_db=db_path,
        native_workspace="native-test",
        query_vector=json.dumps([1.0, 0.0]),
        mode="mix",
        top_k=1,
        chunk_top_k=10,
        section_kind=None,
        data_only=True,
        expand_section_neighbors=True,
        save_evidence_pack=False,
        state_dir=tmp_path,
        workdir=tmp_path,
        server="http://127.0.0.1:9621",
        neighbor_k=2,
    )

    result = wiki_search.run_query(args, "alpha")

    expansions = result["response"]["section_neighbor_expansions"]
    assert expansions == [
        {
            "seed_record_id": "raw_section:a:future",
            "seed_source_path": "alpha.md",
            "neighbor_id": "raw_section:b:future",
            "edge_type": "section_similarity",
            "weight": 0.91,
            "payload": {"pair_kind": "future:future"},
        }
    ]
