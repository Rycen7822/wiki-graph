from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.requires_zvec]


def _real_zvec_workspace_module():
    pytest.importorskip("zvec")
    sys.modules.pop("llm_wiki_native.storage.zvec_workspace", None)
    return importlib.import_module("llm_wiki_native.storage.zvec_workspace")


def _record(module, *, record_type: str, record_id: str, content: str, embedding: list[float], section_kind: str | None = None):
    return module.ZvecRecord(
        record_type=record_type,
        record_id=record_id,
        canonical_id=record_id,
        source_id="source:fixture",
        source_kind_code=1,
        source_path_hash=f"path:{record_id}",
        source_path=f"fixtures/{record_id}.md",
        title=record_id.title(),
        vector_hash=f"vector:{record_id}",
        content_hash=f"content:{record_id}",
        metadata_hash=f"metadata:{record_id}",
        content=content,
        tokens=len(content.split()),
        embedding=embedding,
        section_kind=section_kind,
    )


def test_real_zvec_workspace_adapter_canonical_smoke(tmp_path: Path) -> None:
    module = _real_zvec_workspace_module()
    workspace = module.create_workspace_collection(tmp_path / "zvec-real", embedding_dim=2)
    punctuation_id = "raw/section:doc-a:method"
    long_id = "compiled:comparison:adaptive-memory-retrieval-vs-persistent-knowledge-base"
    records = [
        _record(
            module,
            record_type="section",
            record_id=punctuation_id,
            content="alpha target phrase appears in this methodology section",
            embedding=[1.0, 0.0],
            section_kind="methodology",
        ),
        _record(
            module,
            record_type="entity",
            record_id=long_id,
            content="compiled comparison entity without the target phrase",
            embedding=[0.0, 1.0],
        ),
    ]
    doc_ids = [module.zvec_doc_id(record.record_type, record.record_id) for record in records]
    punctuation_doc_id, long_doc_id = doc_ids

    stats = workspace.bulk_insert(records, batch_size=1)
    fetched = workspace.fetch(doc_ids)
    vector_hits = workspace.query_vector([1.0, 0.0], top_k=2, filter_expr=None)
    section_only_hits = workspace.query_vector([1.0, 0.0], top_k=2, filter_expr="record_type_code in (4)")
    mix_hits = workspace.query_mix("alpha target", [0.0, 1.0], top_k=2, filter_expr=None)
    smoke = workspace.self_nearest_smoke(doc_ids)

    assert ":" not in punctuation_doc_id
    assert "/" not in punctuation_doc_id
    assert len(long_doc_id) <= module.MAX_ZVEC_DOC_ID_LENGTH
    assert stats == module.InsertStats(attempted=2, inserted=2, failed=0)
    assert set(fetched) == set(doc_ids)
    assert fetched[punctuation_doc_id].fields["record_id"] == punctuation_id
    assert fetched[long_doc_id].fields["record_id"] == long_id
    assert fetched[long_doc_id].fields["content"] == records[1].content
    assert workspace.stats()["doc_count"] == 2

    assert vector_hits[0].fields["record_id"] == punctuation_id
    assert {hit.fields["record_id"] for hit in vector_hits} == {punctuation_id, long_id}

    assert [hit.fields["record_type"] for hit in section_only_hits] == ["section"]
    assert section_only_hits[0].fields["record_id"] == punctuation_id

    # The text query favors the section even though the vector favors the entity, proving the FTS leg participates.
    assert mix_hits[0].fields["record_id"] == punctuation_id
    assert {hit.fields["record_id"] for hit in mix_hits} == {punctuation_id, long_id}

    assert smoke == module.SmokeResult(checked=2, passed=2, failures=[])
    workspace.flush_optimize_close()


def _workspace_ab(tmp_path: Path, name: str):
    module = _real_zvec_workspace_module()
    workspace = module.create_workspace_collection(tmp_path / name, embedding_dim=3)
    records = [
        _record(module, record_type="chunk", record_id="a", content="content a", embedding=[1.0, 0.0, 0.0]),
        _record(module, record_type="chunk", record_id="b", content="content b", embedding=[0.0, 1.0, 0.0]),
    ]
    return module, workspace, records


def test_real_zvec_upsert_updates_existing_doc_and_smoke_passes(tmp_path: Path) -> None:
    module, workspace, (rec_a, rec_b) = _workspace_ab(tmp_path, "zvec-upsert")
    stats = workspace.bulk_insert([rec_a, rec_b])
    assert stats.failed == 0

    updated = _record(module, record_type="chunk", record_id="a", content="content a2", embedding=[0.0, 0.0, 1.0])
    up = workspace.upsert_records([updated])
    assert (up.attempted, up.inserted, up.failed) == (1, 1, 0)

    doc_id = module.zvec_doc_id("chunk", "a")
    fetched = workspace.fetch([doc_id])
    assert fetched[doc_id].fields["content"] == "content a2"
    assert workspace.stats()["doc_count"] == 2
    smoke = workspace.self_nearest_smoke([doc_id, module.zvec_doc_id("chunk", "b")])
    assert smoke.failures == []
    workspace.flush_optimize_close()


def test_real_zvec_delete_docs_removes_from_fetch_and_query(tmp_path: Path) -> None:
    module, workspace, records = _workspace_ab(tmp_path, "zvec-delete")
    workspace.bulk_insert(records)
    doc_id = module.zvec_doc_id("chunk", "a")
    stats = workspace.delete_docs([doc_id])
    assert (stats.attempted, stats.deleted, stats.failed) == (1, 1, 0)
    assert workspace.fetch([doc_id]).get(doc_id) is None
    hits = workspace.query_vector([1.0, 0.0, 0.0], top_k=5, filter_expr=None)
    assert all(hit.fields["record_id"] != "a" for hit in hits)
    empty = workspace.delete_docs([])
    assert (empty.attempted, empty.deleted, empty.failed) == (0, 0, 0)
    workspace.flush_optimize_close()


def test_real_native_refresh_retention_and_http_retrieval(tmp_path: Path, monkeypatch) -> None:
    """Four real temp builds/cutovers: retain two, query live, then verify rollback."""
    import gc
    import json
    import socket
    import threading
    import time
    import urllib.request

    import uvicorn
    from llm_wiki_native.api.server import create_app
    from llm_wiki_native.pointers import rollback_active_workspace
    from llm_wiki_native.runtime import load_engine_from_workspace_pointer
    from ops import batch_native_refresh
    from support import sample_kg_manifest, write_kg_state

    _real_zvec_workspace_module()
    monkeypatch.delenv("LLM_WIKI_NATIVE_API_KEY", raising=False)
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "a.md").write_text("# Alpha\n\nAlpha retrieval test.\n")
    state = write_kg_state(
        tmp_path / "state", manifest=sample_kg_manifest(),
        section_similarity_edges=[], raw_sections=[],
        vectors={"chunk-hash": [1.0, 0.0], "entity-vector": [0.9, 0.1], "rel-vector": [0.0, 1.0]},
    )
    root = batch_native_refresh.default_workspace_root(state)
    pointer = batch_native_refresh.active_workspace_path(state)
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("must not change")
    running = []
    evidence = []

    def stop():
        if running:
            server, thread, sock = running.pop()
            server.should_exit = True
            thread.join(timeout=10)
            assert not thread.is_alive()
            sock.close()
            del server, thread
            gc.collect()

    def restart_service(*, state_dir):
        stop()
        engine = load_engine_from_workspace_pointer(pointer)
        active_id = json.loads(pointer.read_text())["workspace_id"]
        app = create_app(engine, default_workspace_id=active_id)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        running.append((server, thread, sock))
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            assert thread.is_alive() and time.monotonic() < deadline
            time.sleep(0.01)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as response:
            health = json.load(response)
        assert health["active_workspace_id"] == active_id
        return {"ok": True, "health": health}

    def query_smoke(*, state_dir, active):
        port = running[0][2].getsockname()[1]
        payload = {"query": "Alpha", "query_vector": [1.0, 0.0], "mode": "mix", "top_k": 2, "response_profile": "compact"}
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/query/data", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            result = json.load(response)
        assert result["context_blocks"]
        return {"ok": True, "block_count": len(result["context_blocks"])}

    try:
        for index in range(4):
            name = f"native-retention-{index}"
            batch_native_refresh.mark_pending(state, wiki, reason="retention rehearsal")
            result = batch_native_refresh.refresh_cutover(
                root=wiki, state_dir=state, workspace_root=root, workspace_id=name,
                embedding_profile="conservative", fill_missing_vectors=False, force=True,
                restart_service=restart_service, query_smoke=query_smoke,
                required_unchanged_paths=[sentinel],
            )
            cleanup = result["workspace_cleanup"]
            assert cleanup["ok"], cleanup
            assert sorted(p.name for p in root.iterdir()) == [f"native-retention-{i}" for i in range(max(0, index - 1), index + 1)]
            evidence.append({"workspace": name, "deleted": cleanup["deleted"], "http_blocks": result["query_smoke"]["block_count"]})
        assert sentinel.read_text() == "must not change"
        stop()
        previous = rollback_active_workspace(pointer, batch_native_refresh.active_workspace_history_path(state))
        assert previous["workspace_id"] == "native-retention-2"
        restart_service(state_dir=state)
        assert query_smoke(state_dir=state, active=previous)["ok"]
        print(json.dumps({"real_retention_cycles": evidence, "rollback_http_query": "ok"}))
    finally:
        stop()
