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
