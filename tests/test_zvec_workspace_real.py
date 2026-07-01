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


@pytest.mark.requires_zvec
def test_real_zvec_accepts_encoded_doc_id_with_punctuation(tmp_path) -> None:
    sys.modules.pop("llm_wiki_native.storage.zvec_workspace", None)
    module = importlib.import_module("llm_wiki_native.storage.zvec_workspace")

    record = module.ZvecRecord(
        record_type="section",
        record_id="raw/section:doc-a:method",
        canonical_id="raw/section:doc-a:method",
        source_id="doc:a",
        source_kind_code=2,
        source_path_hash="path-hash",
        source_path="raw/doc-a.md",
        title="Method",
        vector_hash="vector-hash",
        content_hash="content-hash",
        metadata_hash="metadata-hash",
        content="method text",
        tokens=2,
        embedding=[1.0, 0.0],
        section_kind="methodology",
    )
    doc = module.zvec_doc_from_record(record)

    assert ":" not in doc.id
    assert "/" not in doc.id
    workspace = module.create_workspace_collection(tmp_path / "zvec", embedding_dim=2)
    stats = workspace.bulk_insert([record], batch_size=1)
    fetched = workspace.fetch([doc.id])

    assert stats == module.InsertStats(attempted=1, inserted=1, failed=0)
    assert fetched[doc.id].fields["record_id"] == "raw/section:doc-a:method"
    workspace.flush_optimize_close()
    hits = workspace.query_vector([1.0, 0.0], top_k=1, filter_expr="record_type_code in (4)")

    assert hits[0].doc_id == doc.id
    assert hits[0].fields["record_id"] == "raw/section:doc-a:method"


@pytest.mark.requires_zvec
def test_real_zvec_accepts_long_record_id_with_short_stable_doc_id(tmp_path) -> None:
    sys.modules.pop("llm_wiki_native.storage.zvec_workspace", None)
    module = importlib.import_module("llm_wiki_native.storage.zvec_workspace")

    record = module.ZvecRecord(
        record_type="entity",
        record_id="compiled:comparison:adaptive-memory-retrieval-vs-persistent-knowledge-base",
        canonical_id="compiled:comparison:adaptive-memory-retrieval-vs-persistent-knowledge-base",
        source_id="compiled:comparison:adaptive-memory-retrieval-vs-persistent-knowledge-base",
        source_kind_code=1,
        source_path_hash="path-hash",
        source_path="adaptive-memory-retrieval-vs-persistent-knowledge-base.md",
        title="",
        vector_hash="vector-hash",
        content_hash="content-hash",
        metadata_hash="metadata-hash",
        content="compiled:comparison:adaptive-memory-retrieval-vs-persistent-knowledge-base\nAdaptive Memory Retrieval",
        tokens=2,
        embedding=[1.0, 0.0],
    )
    workspace = module.create_workspace_collection(tmp_path / "zvec", embedding_dim=2)
    doc_id = module.zvec_doc_id(record.record_type, record.record_id)

    stats = workspace.bulk_insert([record], batch_size=1)
    fetched = workspace.fetch([doc_id])

    assert len(doc_id) <= module.MAX_ZVEC_DOC_ID_LENGTH
    assert stats == module.InsertStats(attempted=1, inserted=1, failed=0)
    assert fetched[doc_id].fields["record_id"] == record.record_id
    assert fetched[doc_id].fields["content"] == record.content

def test_real_zvec_workspace_insert_fetch_query_mix_filter_and_self_nearest(tmp_path: Path) -> None:
    module = _real_zvec_workspace_module()
    workspace = module.create_workspace_collection(tmp_path / "zvec-real", embedding_dim=2)
    records = [
        _record(
            module,
            record_type="chunk",
            record_id="alpha-chunk",
            content="alpha target phrase appears in this chunk",
            embedding=[1.0, 0.0],
            section_kind="methodology",
        ),
        _record(
            module,
            record_type="entity",
            record_id="beta-entity",
            content="gamma delta entity without the target phrase",
            embedding=[0.0, 1.0],
        ),
    ]
    doc_ids = [module.zvec_doc_id(record.record_type, record.record_id) for record in records]

    stats = workspace.bulk_insert(records, batch_size=1)
    fetched = workspace.fetch(doc_ids)
    vector_hits = workspace.query_vector([1.0, 0.0], top_k=2, filter_expr=None)
    entity_only_hits = workspace.query_vector([1.0, 0.0], top_k=2, filter_expr="record_type_code in (2)")
    mix_hits = workspace.query_mix("alpha target", [0.0, 1.0], top_k=2, filter_expr=None)
    smoke = workspace.self_nearest_smoke(doc_ids)

    assert stats == module.InsertStats(attempted=2, inserted=2, failed=0)
    assert set(fetched) == set(doc_ids)
    assert fetched[module.zvec_doc_id("chunk", "alpha-chunk")].fields["record_id"] == "alpha-chunk"
    assert fetched[module.zvec_doc_id("entity", "beta-entity")].fields["record_type"] == "entity"
    assert workspace.stats()["doc_count"] == 2

    assert vector_hits[0].fields["record_id"] == "alpha-chunk"
    assert {hit.fields["record_id"] for hit in vector_hits} == {"alpha-chunk", "beta-entity"}

    assert [hit.fields["record_type"] for hit in entity_only_hits] == ["entity"]
    assert entity_only_hits[0].fields["record_id"] == "beta-entity"

    # The text query favors alpha even though the vector favors beta, proving the FTS leg participates in query_mix.
    assert mix_hits[0].fields["record_id"] == "alpha-chunk"
    assert {hit.fields["record_id"] for hit in mix_hits} == {"alpha-chunk", "beta-entity"}

    assert smoke == module.SmokeResult(checked=2, passed=2, failures=[])
    workspace.flush_optimize_close()
