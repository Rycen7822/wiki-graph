import json
from pathlib import Path

import pytest

from llm_wiki_native.contracts import (
    DEFAULT_QUERY_RECORD_TYPES,
    DEFAULT_RETRIEVAL_GOAL,
    DEFAULT_NATIVE_PORT,
    NATIVE_SCHEMA_VERSION,
    RECORD_TYPE_CODES,
    RECORD_TYPES,
    SECTION_KIND_CODES,
    SOURCE_KIND_CODES,
    SUPPORTED_RETRIEVAL_GOALS,
    SUPPORTED_QUERY_MODES,
    WORKSPACE_SCHEMA_VERSION,
)
from llm_wiki_native.query_contract import (
    STRUCTURED_QUERY_SUITE_KEYS,
    engine_query_kwargs,
    query_request_metadata,
    query_suite_payload,
)
from llm_wiki_native.reports import validate_query_suite_row, validate_relevance_quality_row


def test_native_contract_constants_match_zvec_plan() -> None:
    assert NATIVE_SCHEMA_VERSION == 1
    assert WORKSPACE_SCHEMA_VERSION == 1
    assert DEFAULT_NATIVE_PORT == 9621
    assert SUPPORTED_QUERY_MODES == {"mix", "naive", "bypass"}
    assert SUPPORTED_RETRIEVAL_GOALS == {"focused", "coverage"}
    assert DEFAULT_RETRIEVAL_GOAL == "focused"
    assert DEFAULT_QUERY_RECORD_TYPES == ("entity", "relationship", "chunk", "section")
    assert RECORD_TYPES == {"chunk", "entity", "relationship", "section"}
    assert RECORD_TYPE_CODES == {"chunk": 1, "entity": 2, "relationship": 3, "section": 4}
    assert SOURCE_KIND_CODES == {"compiled": 1, "raw": 2, "generated": 3, "debug": 4}
    assert SECTION_KIND_CODES == {
        "none": 0,
        "summary": 1,
        "abstract": 2,
        "motivation": 3,
        "methodology": 4,
        "results": 5,
        "future": 6,
        "limitations": 7,
        "questions": 8,
        "other": 99,
    }


def test_query_suite_row_requires_must_hit_fields() -> None:
    row = {
        "id": "q001",
        "query": "How should evidence retrieval be tested?",
        "mode": "mix",
        "top_k": 20,
        "must_include_paths": ["concepts/evidence.md"],
        "must_include_entities": ["compiled:evidence"],
        "notes": "covers path and entity must-hit checks",
    }
    validate_query_suite_row(row)

    bad = dict(row)
    bad.pop("must_include_paths")
    with pytest.raises(ValueError, match="must_include_paths"):
        validate_query_suite_row(bad)

    bad_top_k = dict(row)
    bad_top_k["top_k"] = 0
    with pytest.raises(ValueError, match="top_k"):
        validate_query_suite_row(bad_top_k)


def test_minimal_query_suite_fixture_matches_schema() -> None:
    fixture = Path(__file__).parent / "fixtures" / "query_suite_minimal.jsonl"
    rows = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 1
    validate_query_suite_row(rows[0])


def test_retrieval_goal_is_forwarded_across_structured_query_contract() -> None:
    row = {
        "query": "alpha evidence",
        "mode": "mix",
        "top_k": 4,
        "retrieval_goal": "coverage",
        "query_vector": [1.0, 0.0],
    }
    assert "retrieval_goal" in STRUCTURED_QUERY_SUITE_KEYS
    assert query_suite_payload(row, workspace_id="native-test")["retrieval_goal"] == "coverage"
    assert query_suite_payload(
        {"id": "default", "query": "alpha", "mode": "mix", "top_k": 20},
        workspace_id="native-test",
    )["retrieval_goal"] == "focused"
    assert query_request_metadata(row)["retrieval_goal"] == "coverage"
    assert query_request_metadata({})["retrieval_goal"] == "focused"
    kwargs = engine_query_kwargs(
        {"workspace_id": "native-test", "query": "alpha"},
        normalized_query_vector=[1.0, 0.0],
    )
    assert kwargs["retrieval_goal"] == "focused"
    assert kwargs["record_types"] == DEFAULT_QUERY_RECORD_TYPES

    with pytest.raises(ValueError, match="retrieval_goal"):
        engine_query_kwargs(
            {"workspace_id": "native-test", "retrieval_goal": "wide"},
            normalized_query_vector=[1.0],
        )
    with pytest.raises(ValueError, match="record_types"):
        engine_query_kwargs(
            {"workspace_id": "native-test", "record_types": []},
            normalized_query_vector=[1.0],
        )
    with pytest.raises(ValueError, match="top_k"):
        engine_query_kwargs(
            {"workspace_id": "native-test", "top_k": True},
            normalized_query_vector=[1.0],
        )
    with pytest.raises(ValueError, match="neighbor_limit"):
        engine_query_kwargs(
            {"workspace_id": "native-test", "neighbor_limit": 1.5},
            normalized_query_vector=[1.0],
        )


def _quality_row(**overrides: object) -> dict:
    row = {
        "id": "quality-001",
        "query": "alpha evidence",
        "mode": "mix",
        "top_k": 4,
        "must_include_paths": ["raw/a.md"],
        "must_include_entities": [],
        "must_include_evidence": [
            {"source_path": "raw/a.md", "text_contains": ["exact anchor"]},
        ],
        "notes": "quality fixture",
        "retrieval_goal": "focused",
        "critical": False,
        "partition": "calibration",
        "minimum_distinct_sources": 1,
        "query_vector": [1.0, 0.0],
        "response_profile": "debug",
    }
    row.update(overrides)
    return row


def test_generic_query_suite_accepts_optional_quality_fields_without_requiring_them() -> None:
    validate_query_suite_row(_quality_row())
    fixture = Path(__file__).parent / "fixtures" / "query_suite_minimal.jsonl"
    minimal = json.loads(fixture.read_text(encoding="utf-8"))
    validate_query_suite_row(minimal)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("retrieval_goal", "wide", "retrieval_goal"),
        ("critical", 1, "critical"),
        ("minimum_distinct_sources", True, "minimum_distinct_sources"),
        ("must_include_evidence", "raw/a.md", "must_include_evidence"),
        ("must_include_evidence", [{"source_path": "", "text_contains": ["anchor"]}], "source_path"),
        ("must_include_evidence", [{"source_path": "raw/a.md", "text_contains": []}], "text_contains"),
    ],
)
def test_generic_query_suite_strictly_validates_optional_quality_fields(
    field: str, value: object, message: str
) -> None:
    row = _quality_row()
    row[field] = value
    with pytest.raises(ValueError, match=message):
        validate_query_suite_row(row)


def test_relevance_quality_row_accepts_default_and_explicit_filter_rows() -> None:
    validate_relevance_quality_row(_quality_row())
    validate_relevance_quality_row(_quality_row(record_types=["section"], section_kind="results"))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"id": 7}, "quality.id"),
        ({"must_include_entities": [7]}, "must_include_entities"),
        ({"top_k": True}, "top_k"),
        ({"top_k": 101}, "top_k"),
        ({"critical": True}, "critical.*partition"),
        ({"partition": "holdout", "critical": False}, "critical.*partition"),
        ({"must_include_paths": []}, "must_include_paths"),
        ({"must_include_paths": [""]}, "must_include_paths"),
        ({"must_include_evidence": []}, "must_include_evidence"),
        ({"query_vector": []}, "query_vector"),
        ({"query_vector": [float("inf")]}, "query_vector"),
        ({"query_vector": [0.0] * 8193}, "query_vector"),
        ({"response_profile": "standard"}, "response_profile"),
        ({"record_types": []}, "record_types"),
        ({"record_types": ["unknown"]}, "record_types"),
        ({"retrieval_goal": "focused", "minimum_distinct_sources": 2}, "minimum_distinct_sources"),
        (
            {
                "retrieval_goal": "coverage",
                "minimum_distinct_sources": 2,
                "must_include_evidence": [
                    {"source_path": "raw/a.md", "text_contains": ["one"]},
                ],
            },
            "distinct evidence source",
        ),
    ],
)
def test_relevance_quality_row_rejects_malformed_contract(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_relevance_quality_row(_quality_row(**overrides))


def test_relevance_quality_row_rejects_duplicate_evidence_items() -> None:
    evidence = {"source_path": "raw/a.md", "text_contains": ["exact anchor"]}
    with pytest.raises(ValueError, match="duplicate evidence"):
        validate_relevance_quality_row(_quality_row(must_include_evidence=[evidence, dict(evidence)]))


def test_relevance_quality_row_rejects_noncanonical_evidence_items() -> None:
    with pytest.raises(ValueError, match="exactly source_path and text_contains"):
        validate_relevance_quality_row(
            _quality_row(
                must_include_evidence=[
                    {"source_path": "raw/a.md", "text_contains": ["exact anchor"], "label": "extra"},
                ]
            )
        )
    with pytest.raises(ValueError, match="duplicate anchors"):
        validate_relevance_quality_row(
            _quality_row(
                must_include_evidence=[
                    {"source_path": "raw/a.md", "text_contains": ["same", "same"]},
                ]
            )
        )
