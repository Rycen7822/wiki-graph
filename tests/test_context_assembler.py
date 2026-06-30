import pytest

from llm_wiki_native.retrieval.context import assemble_context


def test_context_assembler_deduplicates_sources_and_limits_text() -> None:
    query_result = {
        "hits": [
            {
                "record_id": "doc:a",
                "record_type": "entity",
                "score": 0.9,
                "record": {"vector_text": "Alpha text is long enough", "source_path": "a.md", "source_id": "doc:a"},
                "neighbors": [{"neighbor_id": "tag:x", "weight": 0.8}],
            },
            {
                "record_id": "doc:a#dup",
                "record_type": "chunk",
                "score": 0.8,
                "record": {"vector_text": "Duplicate source text", "source_path": "a.md", "source_id": "doc:a"},
                "neighbors": [],
            },
            {
                "record_id": "doc:b",
                "record_type": "entity",
                "score": 0.7,
                "record": {"vector_text": "Beta", "source_path": "b.md", "source_id": "doc:b"},
                "neighbors": [],
            },
        ],
        "trace": {"query": "alpha", "mode": "mix"},
    }

    assembled = assemble_context(query_result, max_chars_per_block=10)

    assert [(block["record_id"], block["source_path"], block["text"]) for block in assembled["context_blocks"]] == [
        ("doc:a", "a.md", "Alpha text"),
        ("doc:b", "b.md", "Beta"),
    ]
    assert assembled["source_paths"] == ["a.md", "b.md"]
    assert assembled["trace"]["context_block_count"] == 2


def test_context_assembler_rejects_non_positive_block_limit() -> None:
    with pytest.raises(ValueError, match="max_chars_per_block"):
        assemble_context({"hits": []}, max_chars_per_block=0)
