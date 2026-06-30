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
from llm_wiki_native.reports import validate_query_suite_row


def test_native_contract_constants_match_zvec_plan() -> None:
    assert NATIVE_SCHEMA_VERSION == 1
    assert WORKSPACE_SCHEMA_VERSION == 1
    assert DEFAULT_NATIVE_PORT == 9621
    assert SUPPORTED_QUERY_MODES == {"mix", "naive", "bypass"}
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
