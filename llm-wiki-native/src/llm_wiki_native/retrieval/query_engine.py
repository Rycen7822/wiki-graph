"""Data-only native query engine.

This module intentionally accepts a precomputed query vector. Embedding calls,
LLM answer generation, HTTP serving, and retired-backend compatibility adapters live in
later layers.
"""

from __future__ import annotations

import base64
from typing import Any

from llm_wiki_native.contracts import (
    IMPLEMENTED_QUERY_MODES,
    RECORD_TYPE_CODES,
    RECORD_TYPES,
    SECTION_KIND_CODES,
    SUPPORTED_QUERY_MODES,
)
from llm_wiki_native.storage.sqlite_workspace import SQLiteWorkspace


class NativeQueryEngine:
    def __init__(self, db: SQLiteWorkspace, zvec_workspace: Any | None = None) -> None:
        self.db = db
        self.zvec_workspace = zvec_workspace

    def query(
        self,
        workspace_id: str,
        query: str,
        query_vector: list[float],
        *,
        mode: str,
        top_k: int = 20,
        record_types: tuple[str, ...] = ("entity", "relationship", "chunk"),
        section_kind: str | None = None,
        neighbor_limit: int = 5,
    ) -> dict[str, Any]:
        if mode not in SUPPORTED_QUERY_MODES:
            raise ValueError(f"unsupported mode: {mode}")
        if mode not in IMPLEMENTED_QUERY_MODES:
            raise NotImplementedError(f"native query mode is not implemented yet: {mode}")
        unknown_record_types = [record_type for record_type in record_types if record_type not in RECORD_TYPES]
        if unknown_record_types:
            raise ValueError(f"unsupported record_type: {unknown_record_types[0]}")
        status = self.db.get_workspace_status(workspace_id)
        if status not in {"audited", "active"}:
            raise ValueError(f"workspace must be audited or active before query: {workspace_id} status={status}")
        if section_kind is not None and self.zvec_workspace is None:
            raise NotImplementedError("section_kind requires a zvec workspace")
        if mode == "bypass":
            return {
                "hits": [],
                "trace": {
                    "query": query,
                    "mode": mode,
                    "top_k": top_k,
                    "record_types": list(record_types),
                    "section_kind": section_kind,
                    "vector_hit_count": 0,
                    "retrieval_backend": "bypass",
                },
            }
        if self.zvec_workspace is not None:
            return self._query_zvec(
                workspace_id,
                query,
                query_vector,
                mode=mode,
                top_k=top_k,
                record_types=record_types,
                section_kind=section_kind,
                neighbor_limit=neighbor_limit,
            )
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
                "section_kind": section_kind,
                "vector_hit_count": len(hits),
                "retrieval_backend": "sqlite",
            },
        }

    def _query_zvec(
        self,
        workspace_id: str,
        query: str,
        query_vector: list[float],
        *,
        mode: str,
        top_k: int,
        record_types: tuple[str, ...],
        section_kind: str | None,
        neighbor_limit: int,
    ) -> dict[str, Any]:
        if mode == "mix":
            zvec_hits = self.zvec_workspace.query_mix(
                query,
                query_vector,
                top_k,
                _zvec_filter(record_types, section_kind),
            )
        elif mode == "naive":
            zvec_hits = self.zvec_workspace.query_vector(
                query_vector,
                top_k,
                _zvec_filter(("chunk", "section"), section_kind),
            )
        else:
            raise NotImplementedError(f"zvec query mode is not implemented yet: {mode}")

        hits = []
        for hit in zvec_hits[:top_k]:
            fields = dict(hit.fields)
            record_type, record_id = _record_identity_from_hit(hit.doc_id, fields)
            item = {
                "doc_id": hit.doc_id,
                "score": float(hit.score),
                "record_type": record_type,
                "record_id": record_id,
                "record": self.db.get_record(workspace_id, record_type, record_id),
                "neighbors": self.db.neighbors(workspace_id, record_id, limit=neighbor_limit),
            }
            hits.append(item)
        return {
            "hits": hits,
            "trace": {
                "query": query,
                "mode": mode,
                "top_k": top_k,
                "record_types": list(record_types),
                "section_kind": section_kind,
                "vector_hit_count": len(hits),
                "retrieval_backend": "zvec",
            },
        }


def _zvec_filter(record_types: tuple[str, ...], section_kind: str | None) -> str:
    if section_kind is not None:
        try:
            section_kind_code = SECTION_KIND_CODES[section_kind]
        except KeyError as exc:
            raise ValueError(f"unknown section_kind: {section_kind}") from exc
        return f"record_type_code in (4) and section_kind_code in ({section_kind_code})"
    return _record_type_filter(record_types)


def _record_type_filter(record_types: tuple[str, ...]) -> str:
    codes = sorted(RECORD_TYPE_CODES[record_type] for record_type in record_types)
    return f"record_type_code in ({','.join(str(code) for code in codes)})"


def _record_identity_from_hit(doc_id: str, fields: dict[str, Any]) -> tuple[str, str]:
    record_type = fields.get("record_type")
    record_id = fields.get("record_id")
    if record_type and record_id:
        return str(record_type), str(record_id)
    if ":" not in doc_id:
        return _record_identity_from_encoded_doc_id(doc_id)
    fallback_type, fallback_id = doc_id.split(":", 1)
    return fallback_type, fallback_id


def _record_identity_from_encoded_doc_id(doc_id: str) -> tuple[str, str]:
    if "__" not in doc_id:
        raise ValueError(f"zvec hit missing record identity: {doc_id}")
    record_type, encoded_id = doc_id.split("__", 1)
    if record_type not in RECORD_TYPES or not encoded_id:
        raise ValueError(f"zvec hit missing record identity: {doc_id}")
    padding = "=" * (-len(encoded_id) % 4)
    try:
        record_id = base64.urlsafe_b64decode(f"{encoded_id}{padding}").decode("utf-8")
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"zvec hit missing record identity: {doc_id}") from exc
    if not record_id:
        raise ValueError(f"zvec hit missing record identity: {doc_id}")
    return record_type, record_id
