import json

import pytest

from llm_wiki_native.retrieval.context import assemble_context


def test_context_assembler_preserves_selected_same_source_blocks_and_unique_source_paths() -> None:
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
                "source_key": "a.md",
                "score": 0.8,
                "record": {
                    "vector_text": "Duplicate source text",
                    "source_path": "derived-a.md",
                    "source_id": "doc:a",
                },
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
        "trace": {"query": "alpha", "mode": "mix", "retrieval_goal": "focused", "coverage_fill_pass_used": False},
    }

    assembled = assemble_context(query_result, max_chars_per_block=10)

    assert [(block["record_id"], block["source_path"], block["text"]) for block in assembled["context_blocks"]] == [
        ("doc:a", "a.md", "Alpha text"),
        ("doc:a#dup", "a.md", "Duplicate "),
        ("doc:b", "b.md", "Beta"),
    ]
    assert assembled["source_paths"] == ["a.md", "b.md"]
    assert assembled["trace"]["context_block_count"] == 3
    assert assembled["coverage_plan"]["blocks_per_source"] == {"a.md": 2, "b.md": 1}
    assert assembled["coverage_plan"]["distinct_source_count"] == 2


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


def test_context_assembler_uses_late_query_excerpt_and_projects_debug_safely() -> None:
    long_text = ("irrelevant prefix line\n" * 80) + "decisive needle answer line\ntrailing"
    query_result = {
        "hits": [
            {
                "record_id": "raw_section:late",
                "record_type": "section",
                "score": 0.91,
                "ranking_contract": "relevance-v1",
                "route_ranks": {"zvec_section": 1},
                "relevance_score_breakdown": {
                    "route_rank": 1.0,
                    "source_rank": 1.0,
                    "term_coverage": 1.0,
                    "evidence_quality": 1.0,
                },
                "score_breakdown": {"zvec_route": 1.0},
                "routes": ["zvec"],
                "record": {
                    "vector_text": long_text,
                    "content_hash": "section-content-hash",
                    "source_path": "raw/late.md",
                    "source_id": "raw:late",
                    "payload": {"source_role": "raw", "section_kind": "results"},
                },
                "neighbors": [],
            }
        ],
        "trace": {
            "query": "decisive needle",
            "mode": "mix",
            "retrieval_goal": "focused",
            "coverage_fill_pass_used": False,
            "source_scope": [{"source_key": "raw/late.md", "source_score": 1.0}],
            "planner_decisions": [
                {"record_id": "raw_section:late", "decision": "selected", "reason": "selected"},
                {
                    "record_id": "hidden",
                    "decision": "rejected",
                    "reason": "ranking_cutoff",
                    "evidence_group": "heading:UNSELECTED SECRET HEADING",
                    "content": "UNSELECTED SECRET",
                },
            ],
            "candidate_cards": [
                {
                    "record_type": "section",
                    "record_id": "raw_section:late",
                    "source_path": "raw/late.md",
                    "source_id": "raw:late",
                    "source_key": "raw/late.md",
                    "route_family": "zvec_section",
                    "route_rank": 1,
                    "routes": ["zvec"],
                    "content": "UNSELECTED SECRET",
                }
            ],
        },
    }

    compact = assemble_context(query_result, max_chars_per_block=120, response_profile="compact")
    standard = assemble_context(query_result, max_chars_per_block=120, response_profile="standard")
    debug = assemble_context(query_result, max_chars_per_block=120, response_profile="debug")

    assert "decisive needle" in compact["context_blocks"][0]["text"]
    assert compact["context_blocks"][0]["ranking_contract"] == "relevance-v1"
    assert "excerpt" not in compact["context_blocks"][0]
    assert standard["context_blocks"][0]["excerpt"]["reason"] == "matched_term_window"
    assert standard["context_blocks"][0]["read_span"] == {
        "span_id": "raw_section:late",
        "source_path": "raw/late.md",
        "start_line": 0,
        "end_line": 0,
        "text_hash": "section-content-hash",
    }
    assert standard["trace"]["timings_ms"]["context_assembly"] >= 0
    assert "candidate_cards" not in standard["trace"]
    assert debug["retrieval_debug"]["hits"][0]["ranking_contract"] == "relevance-v1"
    assert debug["retrieval_debug"]["source_scope"][0]["source_key"] == "raw/late.md"
    assert debug["retrieval_debug"]["source_scope"][0]["source_path"] == "raw/late.md"
    assert debug["retrieval_debug"]["candidate_cards"][0]["record_id"] == "raw_section:late"
    assert {
        card["record_id"]
        for card in debug["retrieval_debug"]["decisions"]
        if card.get("decision") == "selected"
    } == {hit["record_id"] for hit in debug["retrieval_debug"]["hits"]}
    assert "UNSELECTED SECRET" not in json.dumps(debug, ensure_ascii=False)


def test_context_assembler_preserves_raw_query_answer_cues_for_excerpt() -> None:
    text = (
        "diffusion training PCA single component\n"
        + ("background filler " * 20)
        + "\nminimizing diffusion training loss is equivalent to PCA; "
        "when $N<d$ the estimator fails."
    )
    query_result = {
        "hits": [
            {
                "record_id": "raw_section:answer",
                "record_type": "section",
                "source_key": "raw/answer.md",
                "record": {
                    "record_type": "section",
                    "source_path": "raw/answer.md",
                    "vector_text": text,
                },
            }
        ],
        "trace": {
            "query": (
                "When is diffusion training equivalent to PCA and what fails "
                "when N is below d?"
            ),
            "retrieval_goal": "focused",
        },
    }

    assembled = assemble_context(query_result, max_chars_per_block=160)

    excerpt = assembled["context_blocks"][0]["text"]
    assert "minimizing diffusion training loss is equivalent to PCA" in excerpt
    assert "when $N<d$ the estimator fails" in excerpt


def test_context_profiles_remain_bounded_at_maximum_public_card_counts() -> None:
    hits = [
        {
            "record_id": f"section:{index:03d}",
            "record_type": "section",
            "score": 1.0,
            "ranking_contract": "relevance-v1",
            "record": {
                "vector_text": ("evidence line\n" * 2000) + f"needle {index}",
                "content_hash": f"hash-{index}",
                "source_path": f"raw/{index:03d}.md",
                "source_id": f"raw:{index:03d}",
                "payload": {"source_role": "raw", "section_kind": "results"},
            },
            "neighbors": [],
        }
        for index in range(100)
    ]
    trace = {
        "query": "needle",
        "retrieval_goal": "coverage",
        "coverage_fill_pass_used": True,
        "source_scope": [{"source_key": f"raw/{index:03d}.md", "source_score": 1.0} for index in range(200)],
        "planner_decisions": [
            {"record_id": f"section:{index:03d}", "decision": "selected", "reason": "selected"}
            for index in range(200)
        ],
        "candidate_cards": [
            {
                "record_type": "section",
                "record_id": f"section:{index:03d}",
                "source_path": f"raw/{index % 200:03d}.md",
                "source_key": f"raw/{index % 200:03d}.md",
                "route_family": "zvec_section",
                "route_rank": index + 1,
                "content": "UNSELECTED SECRET",
            }
            for index in range(600)
        ],
    }

    for profile in ("compact", "standard", "debug"):
        result = assemble_context(
            {"hits": hits, "trace": trace},
            max_chars_per_block=20_000,
            response_profile=profile,
        )
        assert len(result["context_blocks"]) == 100
        assert all(len(block["text"]) <= 20_000 for block in result["context_blocks"])
        assert len(json.dumps(result, ensure_ascii=False).encode("utf-8")) <= 1_048_576
        if profile != "debug":
            assert "retrieval_debug" not in result
            assert "candidate_cards" not in result["trace"]
