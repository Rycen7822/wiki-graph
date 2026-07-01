"""Shared native query request contract helpers."""

from __future__ import annotations

import math
from typing import Any

from llm_wiki_native.contracts import (
    DEFAULT_MAX_CHARS_PER_BLOCK,
    DEFAULT_NEIGHBOR_LIMIT,
    DEFAULT_QUERY_MODE,
    DEFAULT_QUERY_RECORD_TYPES,
    DEFAULT_RESPONSE_PROFILE,
    DEFAULT_TOP_K,
    MAX_CHARS_PER_BLOCK,
    MAX_NEIGHBOR_LIMIT,
    MAX_QUERY_VECTOR_DIM,
    MAX_TOP_K,
)

QUERY_PAYLOAD_OPTIONAL_FIELDS = (
    "query_vector",
    "section_kind",
    "record_types",
    "neighbor_limit",
    "max_chars_per_block",
    "response_profile",
)
QUERY_REQUEST_METADATA_FIELDS = (
    "section_kind",
    "record_types",
    "neighbor_limit",
    "max_chars_per_block",
    "response_profile",
)
STRUCTURED_QUERY_SUITE_KEYS = frozenset(
    {
        "mode",
        "top_k",
        "query_vector",
        "record_types",
        "section_kind",
        "neighbor_limit",
        "max_chars_per_block",
        "response_profile",
        "must_include_paths",
        "must_include_entities",
    }
)


def bounded_int(value: Any, *, default: int, minimum: int, maximum: int, field: str) -> int:
    """Parse a bounded integer using the existing native API clamp semantics."""

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def query_vector(value: Any) -> list[float]:
    """Validate and normalize an explicit query vector."""

    if not isinstance(value, list):
        raise ValueError("query_vector must be a list of finite numbers")
    if not value:
        raise ValueError("query_vector must not be empty")
    if len(value) > MAX_QUERY_VECTOR_DIM:
        raise ValueError(f"query_vector exceeds max dimension {MAX_QUERY_VECTOR_DIM}")
    vector: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("query_vector must contain only finite numbers")
        numeric = float(item)
        if not math.isfinite(numeric):
            raise ValueError("query_vector must contain only finite numbers")
        vector.append(numeric)
    return vector


def query_mode(payload: dict[str, Any]) -> str:
    return str(payload.get("mode", DEFAULT_QUERY_MODE))


def engine_query_kwargs(
    payload: dict[str, Any],
    *,
    normalized_query_vector: list[float],
    default_workspace_id: str | None = None,
) -> dict[str, Any]:
    """Build kwargs for ``NativeQueryEngine.query`` from a normalized payload."""

    workspace_id = payload.get("workspace_id") or default_workspace_id
    if not workspace_id:
        raise ValueError("workspace_id is required")
    return {
        "workspace_id": str(workspace_id),
        "query": str(payload.get("query", "")),
        "query_vector": normalized_query_vector,
        "mode": query_mode(payload),
        "top_k": bounded_int(payload.get("top_k", DEFAULT_TOP_K), default=DEFAULT_TOP_K, minimum=1, maximum=MAX_TOP_K, field="top_k"),
        "record_types": tuple(payload.get("record_types", DEFAULT_QUERY_RECORD_TYPES)),
        "section_kind": str(payload["section_kind"]) if payload.get("section_kind") else None,
        "neighbor_limit": bounded_int(
            payload.get("neighbor_limit", DEFAULT_NEIGHBOR_LIMIT),
            default=DEFAULT_NEIGHBOR_LIMIT,
            minimum=0,
            maximum=MAX_NEIGHBOR_LIMIT,
            field="neighbor_limit",
        ),
    }


def response_max_chars(payload: dict[str, Any]) -> int:
    return bounded_int(
        payload.get("max_chars_per_block", DEFAULT_MAX_CHARS_PER_BLOCK),
        default=DEFAULT_MAX_CHARS_PER_BLOCK,
        minimum=1,
        maximum=MAX_CHARS_PER_BLOCK,
        field="max_chars_per_block",
    )


def response_profile(payload: dict[str, Any]) -> str:
    return str(payload.get("response_profile", DEFAULT_RESPONSE_PROFILE))


def query_suite_payload(row: dict[str, Any], *, workspace_id: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": row["query"],
        "mode": row.get("mode", DEFAULT_QUERY_MODE),
        "top_k": int(row.get("top_k", DEFAULT_TOP_K)),
    }
    if workspace_id:
        payload["workspace_id"] = workspace_id
    for key in QUERY_PAYLOAD_OPTIONAL_FIELDS:
        if key in row:
            payload[key] = row[key]
    return payload


def query_request_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "top_k": int(row.get("top_k", DEFAULT_TOP_K)),
    }
    vector = row.get("query_vector")
    if isinstance(vector, list):
        metadata["query_vector_dim"] = len(vector)
    for key in QUERY_REQUEST_METADATA_FIELDS:
        if key in row:
            metadata[key] = row[key]
    return metadata
