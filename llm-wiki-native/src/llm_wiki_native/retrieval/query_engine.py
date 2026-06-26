"""Data-only native query engine.

This module intentionally accepts a precomputed query vector. Embedding calls,
LLM answer generation, HTTP serving, and LightRAG compatibility adapters live in
later layers.
"""

from __future__ import annotations

from typing import Any

from llm_wiki_native.contracts import RECORD_TYPES, SUPPORTED_QUERY_MODES
from llm_wiki_native.storage.sqlite_workspace import SQLiteWorkspace


class NativeQueryEngine:
    def __init__(self, db: SQLiteWorkspace) -> None:
        self.db = db

    def query(
        self,
        workspace_id: str,
        query: str,
        query_vector: list[float],
        *,
        mode: str,
        top_k: int = 20,
        record_types: tuple[str, ...] = ("entity", "relationship", "chunk"),
        neighbor_limit: int = 5,
    ) -> dict[str, Any]:
        if mode not in SUPPORTED_QUERY_MODES:
            raise ValueError(f"unsupported mode: {mode}")
        unknown_record_types = [record_type for record_type in record_types if record_type not in RECORD_TYPES]
        if unknown_record_types:
            raise ValueError(f"unsupported record_type: {unknown_record_types[0]}")
        vector_hits: list[dict[str, Any]] = []
        for record_type in record_types:
            vector_hits.extend(self.db.nearest_vectors(workspace_id, record_type, query_vector, top_k))
        vector_hits.sort(key=lambda item: (-item["score"], item["record_type"], item["record_id"]))
        hits: list[dict[str, Any]] = []
        for item in vector_hits[:top_k]:
            record = self.db.get_record(workspace_id, item["record_type"], item["record_id"])
            hits.append(
                {
                    **item,
                    "record": record,
                    "neighbors": self.db.neighbors(workspace_id, item["record_id"], limit=neighbor_limit),
                }
            )
        return {
            "hits": hits,
            "trace": {
                "query": query,
                "mode": mode,
                "top_k": top_k,
                "record_types": list(record_types),
                "vector_hit_count": len(hits),
            },
        }
