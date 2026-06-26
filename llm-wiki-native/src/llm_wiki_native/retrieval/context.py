"""Deterministic context assembly for native query hits."""

from __future__ import annotations

from typing import Any


def assemble_context(query_result: dict[str, Any], *, max_chars_per_block: int = 1200) -> dict[str, Any]:
    if max_chars_per_block <= 0:
        raise ValueError("max_chars_per_block must be positive")
    seen_sources: set[str] = set()
    context_blocks: list[dict[str, Any]] = []
    for hit in query_result.get("hits", []):
        record = hit.get("record", {})
        source_path = str(record.get("source_path") or hit.get("record_id") or "")
        if source_path in seen_sources:
            continue
        seen_sources.add(source_path)
        text = str(record.get("vector_text", ""))[:max_chars_per_block]
        context_blocks.append(
            {
                "record_id": hit.get("record_id"),
                "record_type": hit.get("record_type"),
                "score": hit.get("score"),
                "source_path": source_path,
                "source_id": record.get("source_id"),
                "text": text,
                "neighbors": hit.get("neighbors", []),
            }
        )
    trace = dict(query_result.get("trace", {}))
    trace["context_block_count"] = len(context_blocks)
    return {
        "context_blocks": context_blocks,
        "source_paths": [block["source_path"] for block in context_blocks],
        "trace": trace,
    }
