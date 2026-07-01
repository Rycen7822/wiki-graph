"""Coverage and must-read planning for native retrieval contexts."""

from __future__ import annotations

from typing import Any


def build_coverage_plan(hits: list[dict[str, Any]]) -> dict[str, Any]:
    by_source_role: dict[str, list[str]] = {}
    by_span_kind: dict[str, list[str]] = {}
    must_read: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for hit in hits:
        record = hit.get("record", {}) if isinstance(hit.get("record"), dict) else {}
        payload = record.get("payload", {}) if isinstance(record.get("payload"), dict) else {}
        source_path = str(record.get("source_path") or hit.get("record_id") or "")
        if not source_path:
            continue
        source_role = str(payload.get("source_role") or _role_from_path(source_path))
        span_kind = str(payload.get("span_kind") or hit.get("record_type") or "unknown")
        by_source_role.setdefault(source_role, [])
        if source_path not in by_source_role[source_role]:
            by_source_role[source_role].append(source_path)
        by_span_kind.setdefault(span_kind, [])
        if source_path not in by_span_kind[span_kind]:
            by_span_kind[span_kind].append(source_path)
        if source_path not in seen_paths:
            seen_paths.add(source_path)
            must_read.append(
                {
                    "source_path": source_path,
                    "source_id": record.get("source_id"),
                    "source_role": source_role,
                    "span_kind": span_kind,
                    "record_id": hit.get("record_id"),
                    "record_type": hit.get("record_type"),
                    "score": hit.get("score"),
                    "decision": "must_read" if len(must_read) < 8 else "supporting",
                }
            )
    return {
        "by_source_role": by_source_role,
        "by_span_kind": by_span_kind,
        "must_read": must_read,
        "coverage_gaps": _coverage_gaps(by_source_role),
    }


def _role_from_path(source_path: str) -> str:
    if source_path.startswith("_meta/"):
        return "meta_map"
    if source_path.startswith("raw/"):
        return "raw"
    if source_path.startswith(("concepts/", "entities/", "comparisons/", "queries/")) or source_path == "index.md":
        return "compiled"
    return "unknown"


def _coverage_gaps(by_source_role: dict[str, list[str]]) -> list[dict[str, str]]:
    gaps: list[dict[str, str]] = []
    for role in ("raw", "compiled"):
        if not by_source_role.get(role):
            gaps.append({"source_role": role, "severity": "advisory", "message": f"no {role} source in top retrieval context"})
    return gaps
