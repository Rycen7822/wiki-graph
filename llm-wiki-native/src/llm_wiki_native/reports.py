"""Validators for native query-suite inputs."""

from __future__ import annotations

from .contracts import SUPPORTED_QUERY_MODES

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


def _require_keys(data: dict, required: set[str], label: str) -> None:
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"missing {label}.{missing[0]}")


def _require_list(data: dict, key: str, label: str) -> None:
    if not isinstance(data.get(key), list):
        raise ValueError(f"{label}.{key} must be a list")


def validate_query_suite_row(row: dict) -> None:
    """Validate one native query-suite row."""

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
