"""Deterministic context assembly for native query hits."""

from __future__ import annotations

from typing import Any

from llm_wiki_native.retrieval.coverage import build_coverage_plan

RESPONSE_PROFILES = {"compact", "standard", "debug"}


def assemble_context(
    query_result: dict[str, Any],
    *,
    max_chars_per_block: int = 1200,
    response_profile: str = "standard",
) -> dict[str, Any]:
    if max_chars_per_block <= 0:
        raise ValueError("max_chars_per_block must be positive")
    if response_profile not in RESPONSE_PROFILES:
        raise ValueError(f"unsupported response_profile: {response_profile}")
    seen_sources: set[str] = set()
    context_blocks: list[dict[str, Any]] = []
    included_hits: list[dict[str, Any]] = []
    for hit in query_result.get("hits", []):
        record = hit.get("record", {})
        source_path = str(record.get("source_path") or hit.get("record_id") or "")
        if source_path in seen_sources:
            continue
        seen_sources.add(source_path)
        included_hits.append(hit)
        text = str(record.get("vector_text", ""))[:max_chars_per_block]
        block = {
            "record_id": hit.get("record_id"),
            "record_type": hit.get("record_type"),
            "score": hit.get("score"),
            "source_path": source_path,
            "source_id": record.get("source_id"),
            "text": text,
        }
        read_span = _read_span_card(hit, record)
        if read_span:
            block["read_span"] = read_span
        if response_profile != "compact":
            block["neighbors"] = hit.get("neighbors", [])
            if hit.get("routes"):
                block["routes"] = hit.get("routes")
            if hit.get("score_breakdown"):
                block["score_breakdown"] = hit.get("score_breakdown")
        context_blocks.append(block)
    trace = dict(query_result.get("trace", {}))
    trace["context_block_count"] = len(context_blocks)
    result: dict[str, Any] = {
        "context_blocks": context_blocks,
        "source_paths": [block["source_path"] for block in context_blocks],
        "coverage_plan": build_coverage_plan(included_hits),
        "trace": trace,
    }
    if response_profile == "debug":
        result["retrieval_debug"] = {
            "hit_count": len(query_result.get("hits", [])),
            "hits": [
                {
                    "record_id": hit.get("record_id"),
                    "record_type": hit.get("record_type"),
                    "source_path": (hit.get("record") or {}).get("source_path") if isinstance(hit.get("record"), dict) else None,
                    "score": hit.get("score"),
                    "routes": hit.get("routes", []),
                    "score_breakdown": hit.get("score_breakdown", {}),
                }
                for hit in query_result.get("hits", [])
            ],
        }
    return result


def _read_span_card(hit: dict[str, Any], record: dict[str, Any]) -> dict[str, Any] | None:
    payload = record.get("payload", {}) if isinstance(record.get("payload"), dict) else {}
    start_line = payload.get("start_line")
    end_line = payload.get("end_line")
    if start_line in (None, "") and end_line in (None, ""):
        return None
    return {
        "span_id": hit.get("record_id"),
        "source_path": record.get("source_path"),
        "start_line": int(start_line or 0),
        "end_line": int(end_line or start_line or 0),
        "text_hash": payload.get("text_hash"),
    }
