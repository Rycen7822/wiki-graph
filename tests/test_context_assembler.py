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


def test_context_assembler_profiles_preserve_coverage_anchors() -> None:
    query_result = {
        "hits": [
            {
                "record_id": "span:meta",
                "record_type": "lexical_span",
                "score": 42.0,
                "routes": ["lexical_fts"],
                "score_breakdown": {"source_role": 30.0, "span_kind": 30.0},
                "record": {
                    "vector_text": "- raw/clip/2601/26010101_Foo-Paper.md :: MapOnlyNeedle",
                    "source_path": "_meta/raw-clip-map.md",
                    "source_id": "meta:raw-clip-map",
                    "payload": {"source_role": "meta_map", "span_kind": "map.row", "start_line": 4, "end_line": 4},
                },
                "neighbors": [{"neighbor_id": "raw_clip:foo", "weight": 0.9}],
            },
            {
                "record_id": "raw_section:foo:method",
                "record_type": "section",
                "score": 21.0,
                "routes": ["zvec"],
                "score_breakdown": {"zvec_route": 1.0},
                "record": {
                    "vector_text": "Methodology source text",
                    "source_path": "raw/clip/2601/26010101_Foo-Paper.md",
                    "source_id": "raw_clip:foo",
                    "payload": {"section_kind": "methodology", "source_role": "raw"},
                },
                "neighbors": [],
            },
            {
                "record_id": "compiled:concept:foo",
                "record_type": "entity",
                "score": 12.0,
                "routes": ["zvec"],
                "score_breakdown": {"zvec_route": 0.5},
                "record": {
                    "vector_text": "Compiled synthesis",
                    "source_path": "concepts/foo.md",
                    "source_id": "compiled:concept:foo",
                    "payload": {"source_role": "compiled"},
                },
                "neighbors": [],
            },
        ],
        "trace": {"query": "foo", "mode": "mix", "warnings": ["sample-warning"]},
    }

    compact = assemble_context(query_result, max_chars_per_block=12, response_profile="compact")
    debug = assemble_context(query_result, max_chars_per_block=12, response_profile="debug")

    assert compact["source_paths"] == ["_meta/raw-clip-map.md", "raw/clip/2601/26010101_Foo-Paper.md", "concepts/foo.md"]
    assert compact["coverage_plan"]["by_source_role"]["meta_map"] == ["_meta/raw-clip-map.md"]
    assert compact["coverage_plan"]["by_source_role"]["raw"] == ["raw/clip/2601/26010101_Foo-Paper.md"]
    assert compact["coverage_plan"]["by_source_role"]["compiled"] == ["concepts/foo.md"]
    assert compact["coverage_plan"]["must_read"][0]["source_path"] == "_meta/raw-clip-map.md"
    assert "neighbors" not in compact["context_blocks"][0]
    assert compact["context_blocks"][0]["text"] == "- raw/clip/2"
    assert compact["trace"]["warnings"] == ["sample-warning"]
    assert debug["context_blocks"][0]["neighbors"] == [{"neighbor_id": "raw_clip:foo", "weight": 0.9}]
    assert debug["retrieval_debug"]["hits"][0]["score_breakdown"]["source_role"] == 30.0
