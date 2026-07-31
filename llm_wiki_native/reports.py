"""Validators for native query-suite inputs."""

from __future__ import annotations

import math

from .contracts import (
    MAX_QUERY_VECTOR_DIM,
    MAX_TOP_K,
    RECORD_TYPES,
    SECTION_KIND_CODES,
    SUPPORTED_QUERY_MODES,
    SUPPORTED_RETRIEVAL_GOALS,
)

_QUERY_ROW_REQUIRED = {
    "id",
    "query",
    "mode",
    "top_k",
    "must_include_paths",
    "must_include_entities",
    "notes",
}


def _require_keys(data: dict, required: set[str], label: str) -> None:
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"missing {label}.{missing[0]}")


def _require_list(data: dict, key: str, label: str) -> None:
    if not isinstance(data.get(key), list):
        raise ValueError(f"{label}.{key} must be a list")


def _validate_evidence_items(value: object, *, allow_empty: bool) -> list[dict]:
    if not isinstance(value, list):
        raise ValueError("query.must_include_evidence must be a list")
    if not value and not allow_empty:
        raise ValueError("quality.must_include_evidence must be non-empty")
    for index, item in enumerate(value):
        label = f"query.must_include_evidence[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{label} must be an object")
        if set(item) != {"source_path", "text_contains"}:
            raise ValueError(f"{label} must contain exactly source_path and text_contains")
        source_path = item.get("source_path")
        if not isinstance(source_path, str) or not source_path.strip():
            raise ValueError(f"{label}.source_path must be non-empty")
        anchors = item.get("text_contains")
        if not isinstance(anchors, list) or not anchors:
            raise ValueError(f"{label}.text_contains must be a non-empty list")
        if any(not isinstance(anchor, str) or not anchor for anchor in anchors):
            raise ValueError(f"{label}.text_contains must contain non-empty strings")
        if len(set(anchors)) != len(anchors):
            raise ValueError(f"{label}.text_contains contains duplicate anchors")
    return value


def validate_query_suite_row(row: dict) -> None:
    """Validate one native query-suite row."""

    _require_keys(row, _QUERY_ROW_REQUIRED, "query")
    if not str(row.get("id", "")).strip():
        raise ValueError("query.id must be non-empty")
    if not str(row.get("query", "")).strip():
        raise ValueError("query.query must be non-empty")
    if row.get("mode") not in SUPPORTED_QUERY_MODES:
        raise ValueError(f"query.mode must be one of {sorted(SUPPORTED_QUERY_MODES)}")
    if int(row["top_k"]) <= 0:
        raise ValueError("query.top_k must be positive")
    _require_list(row, "must_include_paths", "query")
    _require_list(row, "must_include_entities", "query")
    if "retrieval_goal" in row and row["retrieval_goal"] not in SUPPORTED_RETRIEVAL_GOALS:
        raise ValueError(f"query.retrieval_goal must be one of {sorted(SUPPORTED_RETRIEVAL_GOALS)}")
    if "critical" in row and not isinstance(row["critical"], bool):
        raise ValueError("query.critical must be a boolean")
    if "minimum_distinct_sources" in row:
        minimum = row["minimum_distinct_sources"]
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
            raise ValueError("query.minimum_distinct_sources must be a positive integer")
    if "must_include_evidence" in row:
        _validate_evidence_items(row["must_include_evidence"], allow_empty=True)


def validate_relevance_quality_row(row: dict) -> None:
    """Validate one row of the frozen ``relevance-v1`` quality contract."""

    validate_query_suite_row(row)
    required = {
        "retrieval_goal",
        "critical",
        "partition",
        "must_include_evidence",
        "minimum_distinct_sources",
        "query_vector",
        "response_profile",
    }
    _require_keys(row, required, "quality")

    for key in ("id", "query"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"quality.{key} must be a non-empty string")
    if not isinstance(row.get("notes"), str):
        raise ValueError("quality.notes must be a string")
    entities = row.get("must_include_entities")
    if not isinstance(entities, list) or any(
        not isinstance(entity, str) or not entity.strip() for entity in entities
    ):
        raise ValueError("quality.must_include_entities must contain non-empty strings")

    top_k = row.get("top_k")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"quality.top_k must be an integer from 1 to {MAX_TOP_K}")

    partition = row.get("partition")
    if partition not in {"calibration", "holdout"}:
        raise ValueError("quality.partition must be calibration or holdout")
    critical = row.get("critical")
    if not isinstance(critical, bool) or critical != (partition == "holdout"):
        raise ValueError("quality.critical must be true exactly when quality.partition is holdout")

    paths = row.get("must_include_paths")
    if (
        not isinstance(paths, list)
        or not paths
        or any(not isinstance(path, str) or not path.strip() for path in paths)
    ):
        raise ValueError("quality.must_include_paths must contain non-empty paths")

    evidence = _validate_evidence_items(row.get("must_include_evidence"), allow_empty=False)
    evidence_keys: set[tuple[str, tuple[str, ...]]] = set()
    evidence_paths: set[str] = set()
    for item in evidence:
        key = (item["source_path"], tuple(item["text_contains"]))
        if key in evidence_keys:
            raise ValueError("quality.must_include_evidence contains a duplicate evidence item")
        evidence_keys.add(key)
        evidence_paths.add(item["source_path"])

    goal = row.get("retrieval_goal")
    minimum_sources = row.get("minimum_distinct_sources")
    if goal == "focused" and minimum_sources != 1:
        raise ValueError("focused quality.minimum_distinct_sources must equal 1")
    if goal == "coverage":
        if (
            not isinstance(minimum_sources, int)
            or isinstance(minimum_sources, bool)
            or not 2 <= minimum_sources <= top_k
        ):
            raise ValueError("coverage quality.minimum_distinct_sources must be from 2 through top_k")
        if len(evidence_paths) < minimum_sources:
            raise ValueError("coverage row must label at least minimum_distinct_sources distinct evidence source paths")

    vector = row.get("query_vector")
    if not isinstance(vector, list) or not 1 <= len(vector) <= MAX_QUERY_VECTOR_DIM:
        raise ValueError("quality.query_vector must be a non-empty list")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in vector
    ):
        raise ValueError("quality.query_vector must contain only finite numbers")

    if row.get("response_profile") != "debug":
        raise ValueError("quality.response_profile must be debug")

    if "record_types" in row:
        record_types = row["record_types"]
        if not isinstance(record_types, list) or not record_types:
            raise ValueError("quality.record_types must be a non-empty list")
        if any(
            not isinstance(record_type, str) or record_type not in RECORD_TYPES
            for record_type in record_types
        ):
            raise ValueError(f"quality.record_types must contain only {sorted(RECORD_TYPES)}")
    if "section_kind" in row and row["section_kind"] not in SECTION_KIND_CODES:
        raise ValueError(f"quality.section_kind must be one of {sorted(SECTION_KIND_CODES)}")
    if "workspace_id" in row and (
        not isinstance(row["workspace_id"], str) or not row["workspace_id"].strip()
    ):
        raise ValueError("quality.workspace_id must be non-empty when present")
