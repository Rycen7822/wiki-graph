from __future__ import annotations

import pytest

from llm_wiki_native.build import MissingNativeVectorsError, materialize_zvec_records, missing_vector_report
from llm_wiki_native.contracts import SOURCE_KIND_CODES
from llm_wiki_native.storage.zvec_records import ZvecRecord


def test_materialize_zvec_records_from_manifest_and_sections() -> None:
    manifest = {
        "chunks": {
            "chunk-a": {
                "chunk_id": "chunk-a",
                "content": "Alpha chunk",
                "content_hash": "chunk-hash",
                "source_id": "doc:a",
                "file_path": "a.md",
            }
        },
        "entities": {
            "doc:a": {
                "entity_name": "doc:a",
                "content": "doc:a\nAlpha",
                "vector_hash": "entity-vector",
                "metadata_hash": "entity-meta",
                "source_logical_id": "doc:a",
                "file_path": "a.md",
            }
        },
        "relationships": {
            "doc:a<SEP>tag:x": {
                "src_id": "doc:a",
                "tgt_id": "tag:x",
                "content": "RELATED\tdoc:a\ntag:x\nAlpha tag",
                "vector_hash": "rel-vector",
                "metadata_hash": "rel-meta",
                "source_logical_id": "doc:a",
                "file_path": "a.md",
            }
        },
    }
    sections = [
        {
            "section_id": "raw_section:doc-a:method",
            "source_id": "doc:a",
            "source_path": "a.md",
            "paper_title": "Alpha",
            "section_kind": "method",
            "content": "Method body",
        }
    ]
    records = materialize_zvec_records(
        manifest,
        sections,
        vectors_by_hash={
            "chunk-hash": [1.0, 0.0],
            "entity-vector": [0.9, 0.1],
            "rel-vector": [0.0, 1.0],
        },
        section_embeddings_by_id={
            "raw_section:doc-a:method": {
                "text_hash": "section-vector",
                "embedding": [0.5, 0.5],
            }
        },
    )

    assert all(isinstance(record, ZvecRecord) for record in records)
    assert [(record.record_type, record.record_id) for record in records] == [
        ("chunk", "chunk-a"),
        ("entity", "doc:a"),
        ("relationship", "doc:a<SEP>tag:x"),
        ("section", "raw_section:doc-a:method"),
    ]
    chunk, entity, relationship, section = records
    assert chunk.source_kind_code == SOURCE_KIND_CODES["compiled"]
    assert chunk.canonical_id == "chunk-a"
    assert chunk.source_id == "doc:a"
    assert chunk.embedding == [1.0, 0.0]
    assert chunk.tokens == 2
    assert entity.content_hash == "entity-vector"
    assert relationship.vector_hash == "rel-vector"
    assert section.source_kind_code == SOURCE_KIND_CODES["raw"]
    assert section.vector_hash == "section-vector"
    assert section.section_kind == "methodology"
    assert section.title == "Alpha"
    assert len(section.source_path_hash) == 64


def test_materialize_zvec_records_fails_closed_on_missing_vectors() -> None:
    manifest = {
        "chunks": {
            "chunk-a": {
                "content": "Alpha chunk",
                "content_hash": "chunk-hash",
                "file_path": "a.md",
            }
        }
    }

    with pytest.raises(MissingNativeVectorsError) as exc:
        materialize_zvec_records(
            manifest,
            [],
            vectors_by_hash={},
            section_embeddings_by_id={},
        )
    assert exc.value.report["total_missing"] == 1
    assert exc.value.report["by_type"] == {"chunk": 1}
    assert exc.value.report["manifest"] == [
        {
            "record_type": "chunk",
            "record_id": "chunk-a",
            "vector_hash": "chunk-hash",
            "collection": "chunks",
            "source_path": "a.md",
        }
    ]


def test_missing_vector_report_lists_manifest_and_section_gaps() -> None:
    manifest = {
        "chunks": {
            "chunk-a": {
                "content": "Alpha chunk",
                "content_hash": "chunk-hash",
                "file_path": "a.md",
            }
        },
        "entities": {
            "doc:a": {
                "content": "doc:a\nAlpha",
                "vector_hash": "entity-vector",
                "file_path": "a.md",
            }
        },
    }
    sections = [
        {
            "section_id": "raw_section:doc-a:method",
            "source_path": "a.md",
            "content": "Method body",
        }
    ]

    report = missing_vector_report(
        manifest,
        sections,
        vectors_by_hash={"entity-vector": [0.9, 0.1]},
        section_embeddings_by_id={},
    )

    assert report == {
        "total_missing": 2,
        "by_type": {"chunk": 1, "section": 1},
        "manifest": [
            {
                "collection": "chunks",
                "record_id": "chunk-a",
                "record_type": "chunk",
                "source_path": "a.md",
                "vector_hash": "chunk-hash",
            }
        ],
        "sections": [
            {
                "record_id": "raw_section:doc-a:method",
                "record_type": "section",
                "section_id": "raw_section:doc-a:method",
                "source_path": "a.md",
                "vector_hash": "",
            }
        ],
    }


def test_materialize_zvec_records_fails_closed_on_unknown_section_kind() -> None:
    with pytest.raises(ValueError, match="unknown section_kind"):
        materialize_zvec_records(
            {},
            [
                {
                    "section_id": "raw_section:doc-a:appendix",
                    "section_kind": "appendix",
                    "content": "Appendix",
                }
            ],
            vectors_by_hash={},
            section_embeddings_by_id={
                "raw_section:doc-a:appendix": {
                    "text_hash": "section-vector",
                    "embedding": [0.5, 0.5],
                }
            },
        )
