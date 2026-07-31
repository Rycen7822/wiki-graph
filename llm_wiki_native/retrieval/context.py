"""Deterministic context assembly for native query hits."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from llm_wiki_native.retrieval.coverage import build_coverage_plan
from llm_wiki_native.retrieval.relevance import query_aware_excerpt

RESPONSE_PROFILES = {"compact", "standard", "debug"}
MAX_CONTEXT_TEXT_CHARS = 512 * 1024


def assemble_context(
    query_result: dict[str, Any],
    *,
    max_chars_per_block: int = 1200,
    response_profile: str = "standard",
) -> dict[str, Any]:
    assembly_started = perf_counter()
    if max_chars_per_block <= 0:
        raise ValueError("max_chars_per_block must be positive")
    if response_profile not in RESPONSE_PROFILES:
        raise ValueError(f"unsupported response_profile: {response_profile}")
    raw_trace = dict(query_result.get("trace", {}))
    query_text = str(raw_trace.get("query") or "")
    hits = list(query_result.get("hits", []))
    block_char_budget = min(
        max_chars_per_block,
        max(1, MAX_CONTEXT_TEXT_CHARS // max(1, len(hits))),
    )
    context_blocks: list[dict[str, Any]] = []
    included_hits: list[dict[str, Any]] = []
    source_paths: list[str] = []
    seen_source_paths: set[str] = set()
    for hit in hits:
        record = hit.get("record", {})
        source_path = str(
            hit.get("source_key") or record.get("source_path") or hit.get("record_id") or ""
        )
        included_hits.append(hit)
        if source_path not in seen_source_paths:
            seen_source_paths.add(source_path)
            source_paths.append(source_path)
        excerpt = _block_excerpt(
            str(record.get("vector_text") or ""),
            record_type=str(hit.get("record_type") or record.get("record_type") or ""),
            query_text=query_text,
            max_chars=block_char_budget,
        )
        block = {
            "record_id": hit.get("record_id"),
            "record_type": hit.get("record_type"),
            "score": hit.get("score"),
            "evidence_hash": hit.get("evidence_hash"),
            "source_path": source_path,
            "source_id": record.get("source_id"),
            "text": excerpt["text"],
        }
        if "ranking_contract" in hit:
            block["ranking_contract"] = hit["ranking_contract"]
        read_span = _read_span_card(hit, record)
        if read_span:
            block["read_span"] = read_span
        if response_profile != "compact":
            if excerpt.get("metadata") is not None:
                block["excerpt"] = excerpt["metadata"]
            block["neighbors"] = hit.get("neighbors", [])
            if hit.get("routes"):
                block["routes"] = hit.get("routes")
            if hit.get("score_breakdown"):
                block["score_breakdown"] = hit.get("score_breakdown")
            if "relevance_score_breakdown" in hit:
                block["relevance_score_breakdown"] = hit["relevance_score_breakdown"]
            if response_profile == "debug" and "route_ranks" in hit:
                block["route_ranks"] = hit["route_ranks"]
        context_blocks.append(block)
    trace = _public_summary_trace(raw_trace)
    trace["context_block_count"] = len(context_blocks)
    result: dict[str, Any] = {
        "context_blocks": context_blocks,
        "source_paths": source_paths,
        "coverage_plan": build_coverage_plan(included_hits, trace=raw_trace),
        "trace": trace,
    }
    if response_profile == "debug":
        result["retrieval_debug"] = {
            "hit_count": len(query_result.get("hits", [])),
            "hits": [_debug_hit(hit) for hit in hits],
            "source_scope": [_source_scope_card(card) for card in raw_trace.get("source_scope", [])],
            "decisions": [_decision_card(card) for card in raw_trace.get("planner_decisions", [])],
            "candidate_cards": [_candidate_card(card) for card in raw_trace.get("candidate_cards", [])],
        }
    result["trace"].setdefault("timings_ms", {})["context_assembly"] = (
        perf_counter() - assembly_started
    ) * 1000
    return result


def _block_excerpt(
    text: str,
    *,
    record_type: str,
    query_text: str,
    max_chars: int,
) -> dict[str, Any]:
    if record_type in {"chunk", "section"} and len(text) > max_chars:
        excerpt = query_aware_excerpt(text, query_text, max_chars)
        return {
            "text": excerpt["text"],
            "metadata": excerpt["metadata"],
        }
    return {"text": text[:max_chars], "metadata": None}


def _public_summary_trace(trace: dict[str, Any]) -> dict[str, Any]:
    internal_keys = {"source_scope", "planner_decisions", "candidate_cards", "planner"}
    result = {key: value for key, value in trace.items() if key not in internal_keys}
    if isinstance(trace.get("timings_ms"), dict):
        result["timings_ms"] = dict(trace["timings_ms"])
    return result


def _debug_hit(hit: dict[str, Any]) -> dict[str, Any]:
    record_value = hit.get("record")
    record = record_value if isinstance(record_value, dict) else {}
    return {
        "record_id": hit.get("record_id"),
        "record_type": hit.get("record_type"),
        "source_path": hit.get("source_key") or record.get("source_path"),
        "score": hit.get("score"),
        "evidence_hash": hit.get("evidence_hash"),
        "routes": list(hit.get("routes") or []),
        "score_breakdown": dict(hit.get("score_breakdown") or {}),
        "ranking_contract": hit.get("ranking_contract"),
        "route_ranks": dict(hit.get("route_ranks") or {}),
        "relevance_score_breakdown": dict(hit.get("relevance_score_breakdown") or {}),
    }


def _source_scope_card(card: object) -> dict[str, Any]:
    if not isinstance(card, dict):
        return {}
    keys = ("source_key", "source_rank", "source_score", "route_values")
    result = {key: card[key] for key in keys if key in card}
    if isinstance(card.get("source_key"), str):
        result["source_path"] = card["source_key"]
    return result


def _decision_card(card: object) -> dict[str, Any]:
    if not isinstance(card, dict):
        return {}
    keys = (
        "record_type",
        "record_id",
        "source_key",
        "decision",
        "reason",
        "score",
        "is_primary",
    )
    return {key: card[key] for key in keys if key in card}


def _candidate_card(card: object) -> dict[str, Any]:
    if not isinstance(card, dict):
        return {}
    keys = (
        "record_type",
        "record_id",
        "source_path",
        "source_id",
        "source_key",
        "route_family",
        "route_rank",
        "routes",
    )
    return {key: card[key] for key in keys if key in card}


def _read_span_card(hit: dict[str, Any], record: dict[str, Any]) -> dict[str, Any] | None:
    payload = record.get("payload", {}) if isinstance(record.get("payload"), dict) else {}
    start_line = payload.get("start_line")
    end_line = payload.get("end_line")
    if start_line in (None, "") and end_line in (None, ""):
        if hit.get("record_type") != "section":
            return None
        start_line = 0
        end_line = 0
    return {
        "span_id": hit.get("record_id"),
        "source_path": record.get("source_path"),
        "start_line": int(start_line or 0),
        "end_line": int(end_line or start_line or 0),
        "text_hash": payload.get("text_hash") or record.get("content_hash"),
    }
