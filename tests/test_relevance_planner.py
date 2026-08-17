from __future__ import annotations

import ast
import itertools
import json
from pathlib import Path
import tracemalloc

import pytest

from llm_wiki_native.contracts import SECTION_KIND_CODES
from llm_wiki_native.retrieval import relevance as relevance_module
from llm_wiki_native.retrieval.relevance import (
    build_source_scope,
    candidate_limit,
    merge_candidates,
    normalize_query_terms,
    plan_relevance,
    query_aware_excerpt,
    rank_route_candidates,
    scope_limit,
    source_key,
)


def _candidate(
    record_id: str,
    *,
    source_path: str = "raw/a.md",
    source_id: str = "source:a",
    record_type: str = "section",
    route_family: str = "zvec_section",
    route_rank: int = 1,
    raw_score: float = 1.0,
    content: str = "alpha evidence",
    content_hash: str | None = None,
    source_kind_code: int = 2,
    section_kind_code: int = 4,
    span_kind: str | None = None,
    heading_path: list[str] | None = None,
    start_line: int = 0,
    routes: list[str] | None = None,
) -> dict:
    candidate = {
        "record_id": record_id,
        "record_type": record_type,
        "source_path": source_path,
        "source_id": source_id,
        "route_family": route_family,
        "route_rank": route_rank,
        "raw_score": raw_score,
        "content": content,
        "content_hash": content_hash or f"hash:{record_id}",
        "source_kind_code": source_kind_code,
        "section_kind_code": section_kind_code,
        "start_line": start_line,
        "routes": routes or (["zvec"] if route_family.startswith("zvec") else ["lexical_fts"]),
    }
    if span_kind is not None:
        candidate["span_kind"] = span_kind
    if heading_path is not None:
        candidate["heading_path"] = heading_path
    return candidate


def _heading(record_id: str, *, source_path: str, rank: int = 1) -> dict:
    return _candidate(
        record_id,
        source_path=source_path,
        source_id=f"source:{source_path}",
        record_type="lexical_span",
        route_family="lexical",
        route_rank=rank,
        source_kind_code=2,
        section_kind_code=0,
        span_kind="doc.heading",
        heading_path=[record_id],
        content=f"heading {record_id}",
    )


def test_limits_are_bounded_without_public_tuning_knobs() -> None:
    assert candidate_limit(1) == 40
    assert candidate_limit(20) == 100
    assert candidate_limit(100) == 200
    assert candidate_limit(6, "coverage") == 40
    assert scope_limit(1) == 20
    assert scope_limit(20) == 60
    assert scope_limit(100) == 200
    assert scope_limit(6, "coverage") == 200
    with pytest.raises(ValueError):
        candidate_limit(True)
    with pytest.raises(ValueError):
        scope_limit(0)


def test_query_term_normalization_handles_ascii_cjk_order_dedupe_and_cap() -> None:
    assert normalize_query_terms("A ab AB x9 短中文 超长中文检索测试") == [
        "ab",
        "x9",
        "短中文",
        "超长",
        "长中",
        "中文",
        "文检",
        "检索",
        "索测",
        "测试",
    ]
    capped = normalize_query_terms(" ".join(f"term{index}" for index in range(40)))
    assert capped == [f"term{index}" for index in range(32)]

    supplementary = normalize_query_terms("𠀀𠀁𠀂𠀃𠀄")
    assert supplementary == ["𠀀𠀁", "𠀁𠀂", "𠀂𠀃", "𠀃𠀄"]

    tracemalloc.start()
    try:
        assert normalize_query_terms("测" * 100_000) == ["测测"]
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 15_000_000


def test_query_term_normalization_keeps_grouped_decimal_values_atomic() -> None:
    assert normalize_query_terms("Rows 36,193 and 1,125") == [
        "rows",
        "36193",
        "and",
        "1125",
    ]


@pytest.mark.parametrize(
    ("record_id", "content", "query", "expected"),
    [
        ("grouped-value", r"|D|=36{,}193", "36,193", 1.0),
        ("numeric-value", "|P|=237.", "prompt 237", pytest.approx(2 / 3)),
    ],
)
def test_term_coverage_matches_table_numeric_values(record_id, content, query, expected) -> None:
    table = _candidate(
        record_id,
        record_type="lexical_span",
        route_family="lexical",
        section_kind_code=0,
        span_kind="table.row",
        content=content,
    )

    selected = plan_relevance([table], query, 1, "focused")["selected"][0]

    assert selected["relevance_score_breakdown"]["term_coverage"] == expected


def test_route_ranking_normalizes_backend_ties_before_assigning_rank() -> None:
    ranked = rank_route_candidates(
        [
            _candidate("b", raw_score=0.5),
            _candidate("a", raw_score=0.5),
            _candidate("c", raw_score=0.7),
        ],
        route_family="zvec_section",
    )
    assert [candidate["record_id"] for candidate in ranked] == ["c", "a", "b"]
    assert [candidate["route_rank"] for candidate in ranked] == [1, 2, 3]


def test_candidate_identity_merge_keeps_best_rank_per_family_and_external_routes() -> None:
    merged = merge_candidates(
        [
            _candidate("same", route_family="zvec_section", route_rank=3, routes=["zvec"]),
            _candidate(
                "same",
                route_family="zvec_section",
                route_rank=1,
                raw_score=0.9,
                routes=["zvec"],
            ),
            _candidate(
                "same",
                route_family="lexical",
                route_rank=2,
                record_type="section",
                routes=["lexical_fts"],
            ),
        ]
    )
    assert len(merged) == 1
    assert merged[0]["route_ranks"] == {"zvec_section": 1, "lexical": 2}
    assert merged[0]["routes"] == ["zvec", "lexical_fts"]

    three_route = merge_candidates(
        [
            _candidate("shared-three", route_family="zvec_navigation", route_rank=4),
            _candidate("shared-three", route_family="zvec_section", route_rank=2),
            _candidate(
                "shared-three",
                route_family="lexical",
                route_rank=3,
                routes=["lexical_fts"],
            ),
        ]
    )
    assert three_route[0]["route_ranks"] == {
        "zvec_navigation": 4,
        "zvec_section": 2,
        "lexical": 3,
    }
    scope = build_source_scope(three_route, top_k=1)
    assert scope[0]["source_score"] == pytest.approx(1 / 2 + 0.25 * (1 / 3 + 1 / 4))


def test_source_key_falls_back_without_merging_empty_paths() -> None:
    assert source_key(_candidate("a", source_path="raw/a.md", source_id="source:a")) == "raw/a.md"
    assert source_key(_candidate("a", source_path="", source_id="source:a")) == "source:a"
    assert source_key(_candidate("a", source_path="", source_id="")) == "a"
    scope = build_source_scope(
        [
            _candidate("empty-a", source_path="", source_id="", route_rank=1),
            _candidate("empty-b", source_path="", source_id="", route_rank=2),
        ],
        top_k=20,
    )
    assert [card["source_key"] for card in scope] == ["empty-a", "empty-b"]


def test_source_score_uses_one_best_value_per_family_and_rewards_cross_route() -> None:
    same_family = merge_candidates(
        [
            _candidate("a-1", source_path="raw/a.md", route_rank=1),
            _candidate("a-2", source_path="raw/a.md", route_rank=2),
            _candidate("b-1", source_path="raw/b.md", source_id="source:b", route_rank=1),
        ]
    )
    scope = build_source_scope(same_family, top_k=20)
    scores = {card["source_key"]: card["source_score"] for card in scope}
    assert scores["raw/a.md"] == scores["raw/b.md"] == 1.0

    cross_route = merge_candidates(
        [
            *same_family,
            _candidate(
                "a-lexical",
                source_path="raw/a.md",
                route_family="lexical",
                route_rank=2,
                record_type="lexical_span",
            ),
        ]
    )
    scope = build_source_scope(cross_route, top_k=20)
    assert scope[0]["source_key"] == "raw/a.md"
    assert scope[0]["source_score"] == 1.125


def test_raw_section_and_chunk_aliases_share_one_coverage_source() -> None:
    stem = "26062814_evoembedding-evolvable-representations"
    derived_stem = stem.replace("_", "-")
    canonical_path = f"raw/clip/2606/{stem}.md"
    candidates = [
        _candidate(
            "section-a",
            source_path=canonical_path,
            source_id=f"raw_clip:{stem}",
            route_rank=1,
        ),
        _candidate(
            "derived-a-1",
            source_path=f"raw-section-{stem}-summary-deadbeef.md",
            source_id=f"raw_section:{derived_stem}:summary",
            record_type="chunk",
            route_family="zvec_navigation",
            route_rank=2,
            source_kind_code=1,
        ),
        _candidate(
            "derived-a-2",
            source_path=f"raw-section-{stem}-future-feedface.md",
            source_id=f"raw_section:{derived_stem}:future",
            record_type="chunk",
            route_family="zvec_navigation",
            route_rank=3,
            source_kind_code=1,
        ),
        _candidate(
            "section-b",
            source_path="raw/clip/2606/26060000_other.md",
            source_id="raw_clip:26060000_other",
            route_rank=4,
        ),
    ]
    result = plan_relevance(candidates, "alpha", 3, "coverage")
    selected_sources = [hit["source_key"] for hit in result["selected"]]
    assert selected_sources.count(canonical_path) == 2
    assert "raw/clip/2606/26060000_other.md" in selected_sources


@pytest.mark.parametrize(("top_k", "blockers"), [(1, 20), (20, 60), (100, 200)])
def test_ineligible_candidates_do_not_consume_source_scope(top_k: int, blockers: int) -> None:
    candidates = [
        _candidate(
            f"raw-chunk-{index:03d}",
            record_type="chunk",
            source_kind_code=2,
            source_path=f"raw/a-{index:03d}.md",
            route_rank=index + 1,
        )
        for index in range(blockers)
    ]
    candidates.append(
        _candidate(
            "valid-section",
            source_path="raw/zz-valid.md",
            route_rank=blockers + 1,
        )
    )
    result = plan_relevance(candidates, "alpha", top_k, "focused")
    assert "valid-section" in {hit["record_id"] for hit in result["selected"]}


def test_relevance_score_contract_uses_fixed_components() -> None:
    result = plan_relevance([_candidate("evidence")], "alpha", 1, "focused")
    selected = result["selected"][0]
    assert selected["record_id"] == "evidence"
    assert selected["score"] == 1.0
    assert selected["ranking_contract"] == "relevance-v1"
    assert selected["relevance_score_breakdown"] == {
        "local_rank_value": 1.0,
        "source_rank_value": 1.0,
        "term_coverage": 1.0,
        "evidence_quality": 1.0,
    }


def test_rare_query_terms_can_rerank_low_vector_rank_section() -> None:
    candidates = [
        _candidate(
            "common-diffusion",
            route_rank=1,
            source_path="raw/common.md",
            content="diffusion training method",
        ),
        _candidate(
            "specific-pca",
            route_rank=12,
            source_path="raw/specific.md",
            content="single component molrg analysis equivalent pca fails below",
        ),
    ]
    query = "Under the single component MoLRG analysis, when is diffusion training equivalent to PCA and what fails below?"
    result = plan_relevance(candidates, query, 1, "focused")
    assert result["selected"][0]["record_id"] == "specific-pca"


def test_coverage_scoring_can_use_source_identity_terms_without_affecting_focused() -> None:
    candidates = [
        {
            **_candidate("target", route_rank=100, content="generic evidence"),
            "title": "FlashMemory sparse attention",
        },
        _candidate("distractor", route_rank=1, source_path="raw/other.md", content="generic evidence"),
    ]
    coverage = plan_relevance(candidates, "FlashMemory sparse attention", 1, "coverage")
    focused = plan_relevance(candidates, "FlashMemory sparse attention", 1, "focused")
    assert coverage["selected"][0]["record_id"] == "target"
    assert focused["selected"][0]["record_id"] == "distractor"


@pytest.mark.parametrize(
    ("candidate", "quality", "primary"),
    [
        (_candidate("raw-section"), 1.0, True),
        (
            _candidate(
                "future-section",
                section_kind_code=SECTION_KIND_CODES["future"],
            ),
            0.75,
            True,
        ),
        (
            _candidate(
                "snapshot-section",
                record_type="lexical_span",
                route_family="lexical",
                span_kind="raw.section",
            ),
            1.0,
            True,
        ),
        (
            _candidate(
                "table-row",
                record_type="lexical_span",
                route_family="lexical",
                span_kind="table.row",
            ),
            1.0,
            True,
        ),
        (_candidate("compiled-chunk", record_type="chunk", source_kind_code=1), 0.7, True),
        (_heading("heading", source_path="raw/h.md"), 0.5, False),
        (
            _candidate(
                "map-row",
                record_type="lexical_span",
                route_family="lexical",
                span_kind="map.row",
            ),
            0.4,
            False,
        ),
        (
            {
                **_candidate(
                    "meta-table-row",
                    record_type="lexical_span",
                    route_family="lexical",
                    span_kind="table.row",
                    source_path="_meta/raw-clip-map.md",
                ),
                "source_role": "meta_map",
            },
            0.4,
            False,
        ),
        (_candidate("entity", record_type="entity"), 0.2, False),
    ],
)
def test_evidence_classes_have_fixed_quality_and_primary_status(
    candidate: dict, quality: float, primary: bool
) -> None:
    selected = plan_relevance([candidate], "", 1, "focused")["selected"][0]
    assert selected["relevance_score_breakdown"]["evidence_quality"] == quality
    assert selected["is_primary"] is primary


def test_primary_evidence_is_selected_before_higher_ranked_navigation_fallback() -> None:
    candidates = [
        _heading("heading", source_path="raw/nav.md", rank=1),
        _candidate("primary", source_path="raw/evidence.md", route_rank=20),
    ]
    result = plan_relevance(candidates, "heading", 1, "focused")
    assert [hit["record_id"] for hit in result["selected"]] == ["primary"]

    same_evidence = [
        _heading("fallback-duplicate", source_path="raw/same.md", rank=1),
        _candidate(
            "primary-duplicate",
            source_path="raw/same.md",
            route_rank=20,
            content_hash="hash:fallback-duplicate",
        ),
    ]
    deduplicated = plan_relevance(same_evidence, "heading", 1, "focused")
    assert [hit["record_id"] for hit in deduplicated["selected"]] == ["primary-duplicate"]
    assert any(
        decision["record_id"] == "fallback-duplicate"
        and decision["reason"] == "duplicate_evidence"
        for decision in deduplicated["decisions"]
    )


def test_high_term_coverage_heading_is_promoted_for_exact_title_retrieval() -> None:
    source_path = "raw/title.md"
    candidates = [
        _candidate(
            f"section-{index}",
            source_path=source_path,
            section_kind_code=section_kind,
            route_rank=index,
            content=f"primary section {index}",
        )
        for index, section_kind in enumerate((1, 2, 3, 4), start=1)
    ]
    candidates.append(
        {
            **_heading("exact-heading", source_path=source_path, rank=1),
            "content": "dataflex unified framework training",
        }
    )
    result = plan_relevance(candidates, "dataflex unified framework training", 4, "focused")
    selected = {hit["record_id"]: hit for hit in result["selected"]}
    assert selected["exact-heading"]["is_primary"] is True


def test_focused_prefers_distinct_groups_then_uses_deferred_duplicates_with_cap_three() -> None:
    candidates = [
        _candidate("method-best", route_rank=1, section_kind_code=4),
        _candidate("method-duplicate", route_rank=2, section_kind_code=4),
        _candidate("results", route_rank=3, section_kind_code=5),
        _candidate("summary", route_rank=4, section_kind_code=1),
        _candidate("future", route_rank=5, section_kind_code=6),
    ]
    result = plan_relevance(candidates, "alpha", 4, "focused")
    assert [hit["record_id"] for hit in result["selected"]] == [
        "method-best",
        "results",
        "summary",
    ]
    assert result["trace"]["selected_block_count"] == 3

    deferred = plan_relevance(candidates[:3], "alpha", 3, "focused")
    assert [hit["record_id"] for hit in deferred["selected"]] == [
        "method-best",
        "results",
        "method-duplicate",
    ]


def test_zvec_chunk_without_heading_uses_hash_group_without_payload_hydration() -> None:
    chunk = _candidate(
        "chunk",
        record_type="chunk",
        source_kind_code=1,
        section_kind_code=0,
        heading_path=None,
        content_hash="chunk-hash",
    )
    result = plan_relevance([chunk], "alpha", 1, "focused")
    assert result["selected"][0]["evidence_group"] == "hash:chunk-hash"


def test_coverage_uses_distinct_sources_first_then_bounded_fill_pass() -> None:
    candidates = [
        _candidate("a-1", source_path="raw/a.md", route_rank=1),
        _candidate("a-2", source_path="raw/a.md", route_rank=2, section_kind_code=5),
        _candidate("b-1", source_path="raw/b.md", source_id="source:b", route_rank=3),
        _candidate("c-1", source_path="raw/c.md", source_id="source:c", route_rank=4),
    ]
    first_pass = plan_relevance(candidates, "alpha", 3, "coverage")
    assert [hit["record_id"] for hit in first_pass["selected"]] == ["a-1", "b-1", "c-1"]
    assert first_pass["trace"]["coverage_fill_pass_used"] is False

    filled = plan_relevance(candidates, "alpha", 4, "coverage")
    assert [hit["record_id"] for hit in filled["selected"]] == ["a-1", "b-1", "c-1", "a-2"]
    assert filled["trace"]["coverage_fill_pass_used"] is True
    assert max(
        sum(hit["source_key"] == key for hit in filled["selected"])
        for key in {hit["source_key"] for hit in filled["selected"]}
    ) <= 2

    exhausted = plan_relevance(candidates[:1], "alpha", 4, "coverage")
    assert [hit["record_id"] for hit in exhausted["selected"]] == ["a-1"]
    assert exhausted["trace"]["coverage_fill_pass_used"] is False


def test_coverage_first_pass_prefers_summary_representative_per_source() -> None:
    candidates = [
        _candidate(
            "a-future",
            route_rank=1,
            source_path="raw/a.md",
            section_kind_code=SECTION_KIND_CODES["future"],
            content="topic future",
        ),
        _candidate(
            "a-summary",
            route_rank=3,
            source_path="raw/a.md",
            section_kind_code=SECTION_KIND_CODES["summary"],
            content="topic summary",
        ),
        _candidate("b-method", route_rank=2, source_path="raw/b.md", content="topic method"),
    ]
    result = plan_relevance(candidates, "topic", 2, "coverage")
    assert [hit["record_id"] for hit in result["selected"]] == ["a-summary", "b-method"]

    fallback_only = plan_relevance(
        [
            _heading("heading-a", source_path="raw/a.md", rank=1),
            _heading("heading-b", source_path="raw/b.md", rank=2),
        ],
        "heading",
        2,
        "coverage",
    )
    assert {hit["source_key"] for hit in fallback_only["selected"]} == {
        "raw/a.md",
        "raw/b.md",
    }
    assert fallback_only["trace"]["coverage_fill_pass_used"] is False

    primary_before_fallback = plan_relevance(
        [
            _candidate("primary-a-1", source_path="raw/a.md", route_rank=1),
            _candidate("primary-a-2", source_path="raw/a.md", route_rank=2),
            _heading("fallback-b", source_path="raw/b.md", rank=1),
        ],
        "alpha",
        2,
        "coverage",
    )
    assert [hit["record_id"] for hit in primary_before_fallback["selected"]] == [
        "primary-a-1",
        "primary-a-2",
    ]
    assert primary_before_fallback["trace"]["coverage_fill_pass_used"] is True


def test_rejected_decisions_distinguish_ranking_from_source_quota() -> None:
    ranking = plan_relevance(
        [
            _candidate(str(index), source_path=f"raw/{index}.md", route_rank=index)
            for index in range(1, 5)
        ],
        "alpha",
        3,
        "focused",
    )
    assert next(
        decision["reason"] for decision in ranking["decisions"] if decision["record_id"] == "4"
    ) == "ranking_cutoff"

    quota = plan_relevance(
        [_candidate(str(index), route_rank=index) for index in range(1, 5)],
        "alpha",
        4,
        "focused",
    )
    assert next(
        decision["reason"] for decision in quota["decisions"] if decision["record_id"] == "4"
    ) == "source_quota"


def test_focused_reserves_third_source_slot_for_structured_evidence() -> None:
    query = "locate exact alpha beta gamma delta number 36193"
    prose = [
        _candidate(
            "results",
            route_rank=1,
            section_kind_code=SECTION_KIND_CODES["results"],
            content=query,
        ),
        _candidate(
            "methodology",
            route_rank=2,
            section_kind_code=SECTION_KIND_CODES["methodology"],
            content=query,
        ),
        _candidate(
            "compiled",
            record_type="chunk",
            source_kind_code=1,
            route_rank=1,
            content=query,
        ),
    ]
    table = _candidate(
        "exact-table-value",
        record_type="lexical_span",
        route_family="lexical",
        route_rank=20,
        section_kind_code=0,
        span_kind="table.row",
        content="36193",
    )

    result = plan_relevance([*prose, table], query, 3, "focused")

    assert [hit["record_id"] for hit in result["selected"]] == [
        "results",
        "methodology",
        "exact-table-value",
    ]


def test_focused_prefers_raw_sections_over_same_source_compiled_chunk() -> None:
    query = "alpha beta gamma delta exact evidence"
    candidates = [
        _candidate(
            "compiled",
            record_type="chunk",
            source_kind_code=1,
            route_rank=1,
            content=query,
        ),
        _candidate(
            "results",
            route_rank=20,
            section_kind_code=SECTION_KIND_CODES["results"],
            content="alpha evidence",
        ),
        _candidate(
            "methodology",
            route_rank=30,
            section_kind_code=SECTION_KIND_CODES["methodology"],
            content="beta evidence",
        ),
    ]

    result = plan_relevance(candidates, query, 2, "focused")

    assert [hit["record_id"] for hit in result["selected"]] == [
        "results",
        "methodology",
    ]


def test_evidence_dedup_uses_source_and_hash_after_record_identity_merge() -> None:
    candidates = [
        _candidate("best", route_rank=1, content_hash="same-hash"),
        _candidate("duplicate", route_rank=2, content_hash="same-hash"),
    ]
    result = plan_relevance(candidates, "alpha", 2, "focused")
    assert [hit["record_id"] for hit in result["selected"]] == ["best"]
    assert any(
        decision["record_id"] == "duplicate" and decision["reason"] == "duplicate_evidence"
        for decision in result["decisions"]
    )

    normalized_a = _candidate("normalized-a", route_rank=1, content="same   evidence")
    normalized_b = _candidate("normalized-b", route_rank=2, content="same evidence")
    normalized_a.pop("content_hash")
    normalized_b.pop("content_hash")
    normalized = plan_relevance([normalized_a, normalized_b], "same", 2, "focused")
    assert [hit["record_id"] for hit in normalized["selected"]] == ["normalized-a"]


def test_planner_output_is_invariant_to_candidate_input_order() -> None:
    candidates = [
        _candidate("section-a", source_path="raw/a.md", route_rank=2),
        _candidate("section-b", source_path="raw/b.md", source_id="source:b", route_rank=1),
        _heading("heading-c", source_path="raw/c.md", rank=1),
        _candidate(
            "section-a",
            source_path="raw/a.md",
            route_family="lexical",
            route_rank=3,
            routes=["lexical_like"],
        ),
    ]
    outputs = [
        plan_relevance(permutation, "alpha", 3, "focused")
        for permutation in itertools.permutations(candidates)
    ]
    assert all(output == outputs[0] for output in outputs[1:])


def test_malformed_identity_is_skipped_and_debug_scores_remain_strict_json() -> None:
    malformed = _candidate("malformed")
    malformed["record_id"] = ["not", "hashable"]
    missing_score = _candidate("missing-score", source_path="raw/missing.md")
    missing_score.pop("raw_score")
    nan_score = _candidate("nan-score", source_path="raw/nan.md")
    nan_score["raw_score"] = float("nan")

    result = plan_relevance(
        [malformed, missing_score, nan_score],
        "alpha",
        3,
        "focused",
    )
    assert {hit["record_id"] for hit in result["selected"]} == {
        "missing-score",
        "nan-score",
    }
    json.dumps(result, allow_nan=False)


def test_equal_key_duplicate_representative_is_independent_of_input_order() -> None:
    alpha = _candidate("same", content="alpha")
    zeta = _candidate("same", content="zeta")
    alpha.pop("content_hash")
    zeta.pop("content_hash")

    forward = plan_relevance([alpha, zeta], "alpha", 1, "focused")
    reverse = plan_relevance([zeta, alpha], "alpha", 1, "focused")
    assert forward == reverse


def test_planner_is_stable_for_ties_empty_candidates_exhaustion_and_top_k_bounds() -> None:
    tied = [
        _candidate("b", source_path="raw/b.md", source_id="source:b"),
        _candidate("a", source_path="raw/a.md", source_id="source:a"),
    ]
    assert [hit["record_id"] for hit in plan_relevance(tied, "none", 2, "focused")["selected"]] == [
        "a",
        "b",
    ]
    assert plan_relevance([], "anything", 20, "focused")["selected"] == []
    assert len(plan_relevance(tied, "none", 100, "coverage")["selected"]) == 2
    with pytest.raises(ValueError):
        plan_relevance(tied, "none", 1, "unknown")


def test_decisions_and_source_scope_are_bounded_by_internal_limits() -> None:
    candidates = [
        _candidate(
            f"record-{index:03d}",
            source_path=f"raw/{index:03d}.md",
            source_id=f"source:{index:03d}",
            route_rank=index + 1,
        )
        for index in range(300)
    ]
    result = plan_relevance(candidates, "alpha", 100, "focused")
    assert len(result["decisions"]) <= candidate_limit(100)
    assert len(result["source_scope"]) <= scope_limit(100)


def test_bounded_decisions_always_include_every_selected_record() -> None:
    candidates = [
        _candidate(
            f"record-{index:03d}",
            source_path=f"raw/{index:03d}.md",
            source_id=f"source:{index:03d}",
            route_rank=index + 1,
        )
        for index in range(100)
    ]
    result = plan_relevance(candidates, "alpha", 1, "focused")
    selected_ids = {hit["record_id"] for hit in result["selected"]}
    explained_ids = {
        decision["record_id"]
        for decision in result["decisions"]
        if decision["decision"] == "selected"
    }
    assert explained_ids == selected_ids
    assert len(result["decisions"]) <= candidate_limit(1)


def test_query_aware_excerpt_uses_best_line_prefix_fallback_and_centered_long_line() -> None:
    text = "intro\nalpha only\nalpha beta decisive evidence\ntrailer"
    excerpt = query_aware_excerpt(text, ["alpha", "beta", "decisive", "evidence"], 30)
    assert excerpt["text"] in text
    assert "alpha beta decisive evidence" in excerpt["text"]
    assert excerpt["metadata"]["reason"] == "matched_term_window"

    tie = query_aware_excerpt("alpha first\nalpha second\nend", ["alpha"], 12)
    assert tie["text"].startswith("alpha first")

    prefix = query_aware_excerpt("0123456789abcdefghij", ["missing"], 8)
    assert prefix["text"] == "01234567"
    assert prefix["metadata"]["reason"] == "prefix_no_match"

    multiline_prefix = query_aware_excerpt("a\nb\nc\nd", ["missing"], 4)
    assert multiline_prefix["text"] == "a\nb\n"
    assert multiline_prefix["metadata"]["line_start"] == 0
    assert multiline_prefix["metadata"]["line_end"] == 2

    long_line = "x" * 80 + "TARGET" + "y" * 80
    centered = query_aware_excerpt(long_line, ["target"], 40)
    assert centered["text"] in long_line
    assert "TARGET" in centered["text"]
    assert len(centered["text"]) == 40
    assert centered["metadata"]["reason"] == "matched_term_center"

    unicode_prefix = "İ" * 80 + "TARGET" + "y" * 80
    unicode_centered = query_aware_excerpt(unicode_prefix, ["target"], 40)
    start = unicode_centered["metadata"]["char_start"]
    end = unicode_centered["metadata"]["char_end"]
    assert "TARGET" in unicode_centered["text"]
    assert unicode_centered["text"] == unicode_prefix[start:end]


def test_query_aware_excerpt_maximizes_terms_across_the_whole_window() -> None:
    text = (
        "Paper / Technique / Code node types\n"
        + ("background filler\n" * 45)
        + "Implementation Edge connects executable graph nodes\n"
        + ("later filler\n" * 35)
        + "executable graph edge types edge types\n"
    )
    excerpt = query_aware_excerpt(
        text,
        "Paper Technique Code executable graph edge types",
        900,
    )
    assert "Paper / Technique / Code" in excerpt["text"]
    assert "Implementation Edge" in excerpt["text"]

    cross_line = (
        ("x" * 1000)
        + " fixed capacity FIFO latent memory queue\n"
        + ("y" * 220)
        + " capacity 512 evidence "
        + ("z" * 1200)
    )
    refined = query_aware_excerpt(cross_line, "FIFO memory capacity", 1200)
    assert "FIFO latent memory queue" in refined["text"]
    assert "capacity 512 evidence" in refined["text"]
    assert excerpt["metadata"]["reason"] == "matched_term_window"


def test_query_aware_excerpt_ignores_question_stopwords_when_ranking_windows() -> None:
    text = (
        "how does use to its raw training pool\n"
        + ("background filler " * 20)
        + "\npreference aware influence functions purify training pool\n"
    )

    excerpt = query_aware_excerpt(
        text,
        "How does a model use preference-aware influence functions to purify its raw training pool?",
        120,
    )

    assert "preference aware influence functions purify" in excerpt["text"]


def test_query_aware_excerpt_keeps_matching_prefix_before_repeated_term_ties() -> None:
    text = (
        "Paper / Technique / Code node types and edge types form an executable "
        "knowledge graph with an Implementation Edge.\n"
        + ("background filler " * 8)
        + (
            "Paper Technique Code node types edge types form executable "
            "knowledge graph "
        )
        * 3
    )

    excerpt = query_aware_excerpt(
        text,
        "What Paper, Technique, and Code node types and edge types form a graph?",
        220,
    )

    assert excerpt["text"].startswith("Paper / Technique / Code")
    assert "Implementation Edge" in excerpt["text"]


def test_query_aware_excerpt_ignores_structured_metadata_matches() -> None:
    text = (
        "[LLM_WIKI_RAW_SECTION]\n"
        "paper_title: Advances in Temporal Point Processes\n"
        "[/LLM_WIKI_RAW_SECTION]\n"
        "# RawSection: Advances in Temporal Point Processes\n"
        "## Section Content\n"
        "RNN models trade speed for memory; Transformer handles range; "
        "ODE/SDE models time continuously; diffusion provides global generation.\n"
    )

    excerpt = query_aware_excerpt(
        text,
        "Advances in Temporal Point Processes RNN Transformer ODE SDE diffusion",
        220,
    )

    assert "RNN models trade speed for memory" in excerpt["text"]
    assert "diffusion provides global generation" in excerpt["text"]


def test_query_aware_excerpt_does_not_regex_scan_every_line_for_every_term(
    monkeypatch,
) -> None:
    original = relevance_module._term_position
    original_positions = relevance_module._term_positions
    calls = 0
    position_scan_lengths: list[int] = []

    def counted_term_position(text: str, term: str) -> int:
        nonlocal calls
        calls += 1
        return original(text, term)

    def counted_term_positions(text: str, term: str, *, limit: int = 4) -> list[int]:
        position_scan_lengths.append(len(text))
        return original_positions(text, term, limit=limit)

    monkeypatch.setattr(relevance_module, "_term_position", counted_term_position)
    monkeypatch.setattr(relevance_module, "_term_positions", counted_term_positions)
    text = ("background filler\n" * 2_000) + "alpha beta target\n"

    excerpt = query_aware_excerpt(text, ["alpha", "beta", "target"], 1_200)

    assert "alpha beta target" in excerpt["text"]
    assert calls <= 6
    assert max(position_scan_lengths, default=0) < 100


def test_relevance_module_has_no_runtime_io_imports() -> None:
    path = Path(__file__).parents[1] / "llm_wiki_native" / "retrieval" / "relevance.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    forbidden = ("sqlite", "zvec", "llm_wiki_native.storage", "llm_wiki_native.api", "ops")
    assert not any(any(name.startswith(prefix) for prefix in forbidden) for name in imported)
