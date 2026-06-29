import json
from pathlib import Path

import pytest

from llm_wiki_native.contracts import (
    DEFAULT_NATIVE_PORT,
    NATIVE_SCHEMA_VERSION,
    RECORD_TYPE_CODES,
    RECORD_TYPES,
    SECTION_KIND_CODES,
    SOURCE_KIND_CODES,
    SUPPORTED_QUERY_MODES,
    WORKSPACE_SCHEMA_VERSION,
)
from llm_wiki_native.reports import validate_query_suite_row, validate_shadow_report


def test_native_contract_constants_match_zvec_plan() -> None:
    assert NATIVE_SCHEMA_VERSION == 1
    assert WORKSPACE_SCHEMA_VERSION == 1
    assert DEFAULT_NATIVE_PORT == 9622
    assert SUPPORTED_QUERY_MODES == {"local", "global", "hybrid", "naive", "mix", "bypass"}
    from llm_wiki_native.contracts import IMPLEMENTED_QUERY_MODES

    assert IMPLEMENTED_QUERY_MODES == {"mix", "naive", "bypass"}
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
        "chunk_top_k": 10,
        "must_include_paths": ["concepts/evidence.md"],
        "must_include_entities": ["compiled:evidence"],
        "notes": "covers path and entity must-hit checks",
    }
    validate_query_suite_row(row)

    bad = dict(row)
    bad.pop("must_include_paths")
    with pytest.raises(ValueError, match="must_include_paths"):
        validate_query_suite_row(bad)


def test_native_shadow_report_requires_trace_and_baseline_sections() -> None:
    report = {
        "schema_version": 1,
        "query_suite": "/tmp/query_suite.jsonl",
        "baseline": {"server": "http://127.0.0.1:9621", "results": []},
        "native": {"server": "http://127.0.0.1:9622", "results": []},
        "trace_paths": ["/tmp/native-trace.json"],
        "promotion_blockers": [],
    }
    validate_shadow_report(report)

    bad = dict(report)
    bad.pop("trace_paths")
    with pytest.raises(ValueError, match="trace_paths"):
        validate_shadow_report(bad)

    old = dict(report)
    old.pop("baseline")
    old["light" + "rag"] = {"server": "http://127.0.0.1:9621", "results": []}
    with pytest.raises(ValueError, match="baseline"):
        validate_shadow_report(old)


def test_minimal_query_suite_fixture_matches_schema() -> None:
    fixture = Path(__file__).parent / "fixtures" / "query_suite_minimal.jsonl"
    rows = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 1
    validate_query_suite_row(rows[0])
