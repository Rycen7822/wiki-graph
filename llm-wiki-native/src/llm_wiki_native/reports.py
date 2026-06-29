"""Schema validators for native shadow benchmark inputs and reports."""

from __future__ import annotations

from typing import Any

from .contracts import NATIVE_SCHEMA_VERSION, SUPPORTED_QUERY_MODES

_QUERY_ROW_REQUIRED = {
    "id",
    "query",
    "mode",
    "top_k",
    "chunk_top_k",
    "must_include_paths",
    "must_include_entities",
    "notes",
}
_SHADOW_REPORT_REQUIRED = {
    "schema_version",
    "query_suite",
    "baseline",
    "native",
    "trace_paths",
    "promotion_blockers",
}


def _require_keys(data: dict, required: set[str], label: str) -> None:
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"missing {label}.{missing[0]}")


def _require_list(data: dict, key: str, label: str) -> None:
    if not isinstance(data.get(key), list):
        raise ValueError(f"{label}.{key} must be a list")


def validate_query_suite_row(row: dict) -> None:
    """Validate one native shadow benchmark query-suite row."""

    _require_keys(row, _QUERY_ROW_REQUIRED, "query")
    if not str(row.get("id", "")).strip():
        raise ValueError("query.id must be non-empty")
    if not str(row.get("query", "")).strip():
        raise ValueError("query.query must be non-empty")
    if row.get("mode") not in SUPPORTED_QUERY_MODES:
        raise ValueError(f"query.mode must be one of {sorted(SUPPORTED_QUERY_MODES)}")
    for key in ("top_k", "chunk_top_k"):
        if int(row[key]) <= 0:
            raise ValueError(f"query.{key} must be positive")
    _require_list(row, "must_include_paths", "query")
    _require_list(row, "must_include_entities", "query")


def validate_shadow_report(report: dict) -> None:
    """Validate a baseline-vs-native shadow benchmark report shell."""

    _require_keys(report, _SHADOW_REPORT_REQUIRED, "report")
    if int(report.get("schema_version", -1)) != NATIVE_SCHEMA_VERSION:
        raise ValueError(f"report.schema_version must be {NATIVE_SCHEMA_VERSION}")
    if not isinstance(report.get("baseline"), dict):
        raise ValueError("report.baseline must be an object")
    if not isinstance(report.get("native"), dict):
        raise ValueError("report.native must be an object")
    _require_list(report, "trace_paths", "report")
    _require_list(report, "promotion_blockers", "report")


def _collect_paths(response: dict[str, Any]) -> set[str]:
    paths = {str(path) for path in response.get("source_paths", []) if path}
    for block in response.get("context_blocks", []):
        if block.get("source_path"):
            paths.add(str(block["source_path"]))
    return paths


def _collect_entities(response: dict[str, Any]) -> set[str]:
    entities: set[str] = set()
    for block in response.get("context_blocks", []):
        if block.get("source_id"):
            entities.add(str(block["source_id"]))
        if block.get("record_id"):
            entities.add(str(block["record_id"]))
    for hit in response.get("hits", []):
        if hit.get("record_id"):
            entities.add(str(hit["record_id"]))
    return entities


def compare_shadow_response(
    query_id: str,
    baseline_response: dict[str, Any],
    native_response: dict[str, Any],
    *,
    must_include_paths: list[str] | None = None,
    must_include_entities: list[str] | None = None,
) -> dict[str, Any]:
    baseline_paths = _collect_paths(baseline_response)
    native_paths = _collect_paths(native_response)
    native_entities = _collect_entities(native_response)
    blockers: list[str] = []
    for path in must_include_paths or []:
        if path not in native_paths:
            blockers.append(f"missing required path: {path}")
    for entity in must_include_entities or []:
        if entity not in native_entities:
            blockers.append(f"missing required entity: {entity}")
    return {
        "query_id": query_id,
        "ok": not blockers,
        "blockers": blockers,
        "baseline_paths": sorted(baseline_paths),
        "native_paths": sorted(native_paths),
        "path_overlap": sorted(baseline_paths & native_paths),
        "native_entities": sorted(native_entities),
    }
