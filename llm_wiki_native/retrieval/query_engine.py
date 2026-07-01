"""Data-only native query engine.

This module intentionally accepts a precomputed query vector. Embedding calls,
LLM answer generation, and HTTP serving live in the API/search layers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, cast

from llm_wiki_native.contracts import (
    RECORD_TYPE_CODES,
    RECORD_TYPES,
    SECTION_KIND_CODES,
    SUPPORTED_QUERY_MODES,
)
from llm_wiki_native.storage.sqlite_workspace import SQLiteWorkspace


class NativeQueryEngine:
    def __init__(self, db: SQLiteWorkspace, zvec_workspace: Any, *, source_root: Path | None = None) -> None:
        if zvec_workspace is None:
            raise ValueError("zvec workspace is required for native query engine")
        self.db = db
        self.zvec_workspace = zvec_workspace
        self.source_root = Path(source_root).resolve() if source_root is not None else None

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
        include_lexical: bool = True,
        lexical_top_k: int | None = None,
        source_roles: tuple[str, ...] | None = None,
        span_kinds: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        if mode not in SUPPORTED_QUERY_MODES:
            raise ValueError(f"unsupported mode: {mode}")
        unknown_record_types = [record_type for record_type in record_types if record_type not in RECORD_TYPES]
        if unknown_record_types:
            raise ValueError(f"unsupported record_type: {unknown_record_types[0]}")
        status = self.db.get_workspace_status(workspace_id)
        if status != "audited":
            raise ValueError(f"workspace must be audited before query: {workspace_id} status={status}")
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
        return self._query_zvec(
            workspace_id,
            query,
            query_vector,
            mode=mode,
            top_k=top_k,
            record_types=record_types,
            section_kind=section_kind,
            neighbor_limit=neighbor_limit,
            include_lexical=include_lexical,
            lexical_top_k=lexical_top_k or top_k,
            source_roles=source_roles,
            span_kinds=span_kinds,
        )

    def read_span(self, workspace_id: str, span_id: str) -> dict[str, Any]:
        span = self.db.get_lexical_span(workspace_id, span_id)
        response = _span_response_base(workspace_id, span)
        if self.source_root is None:
            return {
                **response,
                "source_status": "snapshot",
                "relocation": "snapshot_only",
                "text": span["text"],
            }
        source_path = _safe_source_path(self.source_root, str(span["source_path"]))
        if source_path is None or not source_path.exists() or not source_path.is_file():
            return {
                **response,
                "source_status": "missing",
                "relocation": "source_missing",
                "text": span["text"],
            }
        lines = source_path.read_text(encoding="utf-8").splitlines()
        stored_text = str(span["text"])
        start_line = int(span.get("start_line") or 0)
        end_line = int(span.get("end_line") or start_line or 0)
        current_text = _text_for_line_range(lines, start_line, end_line)
        if current_text == stored_text:
            return {
                **response,
                "source_status": "current",
                "relocation": "stored_lines",
                "start_line": start_line,
                "end_line": end_line,
                "text": current_text,
            }
        relocated = _find_exact_text(lines, stored_text)
        if relocated is not None:
            relocated_start, relocated_end, relocated_text = relocated
            return {
                **response,
                "source_status": "current",
                "relocation": "exact_text",
                "start_line": relocated_start,
                "end_line": relocated_end,
                "text": relocated_text,
            }
        return {
            **response,
            "source_status": "stale",
            "relocation": "not_found",
            "start_line": start_line,
            "end_line": end_line,
            "text": stored_text,
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
        include_lexical: bool,
        lexical_top_k: int,
        source_roles: tuple[str, ...] | None,
        span_kinds: tuple[str, ...] | None,
    ) -> dict[str, Any]:
        zvec_hits = _query_zvec_hits(
            self.zvec_workspace,
            query,
            query_vector,
            mode=mode,
            top_k=top_k,
            record_types=record_types,
            section_kind=section_kind,
        )
        zvec_items = _zvec_items(
            self.db,
            workspace_id,
            zvec_hits,
            top_k=top_k,
            neighbor_limit=neighbor_limit,
        )
        lexical_items = _query_lexical_items(
            self.db,
            workspace_id,
            query,
            limit=lexical_top_k,
            source_roles=source_roles,
            span_kinds=span_kinds,
        ) if include_lexical and mode == "mix" else []
        hits = _rank_hybrid_hits([*zvec_items, *lexical_items], query=query)[:top_k]
        return _query_response(
            hits,
            query=query,
            mode=mode,
            top_k=top_k,
            record_types=record_types,
            section_kind=section_kind,
            zvec_items=zvec_items,
            lexical_items=lexical_items,
        )


def _query_zvec_hits(
    zvec_workspace: Any,
    query: str,
    query_vector: list[float],
    *,
    mode: str,
    top_k: int,
    record_types: tuple[str, ...],
    section_kind: str | None,
) -> Sequence[Any]:
    filter_expr = _zvec_filter(record_types, section_kind)
    if mode == "mix":
        return zvec_workspace.query_mix(query, query_vector, top_k, filter_expr)
    if mode == "naive":
        return zvec_workspace.query_vector(query_vector, top_k, filter_expr)
    raise NotImplementedError(f"zvec query mode is not implemented yet: {mode}")


def _zvec_items(
    db: Any,
    workspace_id: str,
    zvec_hits: Sequence[Any],
    *,
    top_k: int,
    neighbor_limit: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for rank, hit in enumerate(zvec_hits[:top_k], start=1):
        fields = dict(hit.fields)
        record_type, record_id = _record_identity_from_hit(hit.doc_id, fields)
        record = db.get_record(workspace_id, record_type, record_id)
        items.append(
            {
                "doc_id": hit.doc_id,
                "zvec_score": float(hit.score),
                "record_type": record_type,
                "record_id": record_id,
                "record": record,
                "neighbors": db.neighbors(workspace_id, record_id, limit=neighbor_limit),
                "routes": ["zvec"],
                "route_ranks": {"zvec": rank},
            }
        )
    return items


def _query_response(
    hits: list[dict[str, Any]],
    *,
    query: str,
    mode: str,
    top_k: int,
    record_types: tuple[str, ...],
    section_kind: str | None,
    zvec_items: list[dict[str, Any]],
    lexical_items: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "hits": hits,
        "trace": {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "record_types": list(record_types),
            "section_kind": section_kind,
            "vector_hit_count": len(zvec_items),
            "lexical_hit_count": len(lexical_items),
            "route_counts": {"zvec": len(zvec_items), "lexical": len(lexical_items)},
            "retrieval_backend": "zvec+lexical" if lexical_items else "zvec",
        },
    }


def _span_response_base(workspace_id: str, span: dict[str, Any]) -> dict[str, Any]:
    return {
        "workspace_id": workspace_id,
        "span_id": span.get("span_id"),
        "source_path": span.get("source_path"),
        "source_id": span.get("source_id"),
        "source_role": span.get("source_role"),
        "span_kind": span.get("span_kind"),
        "heading_path": span.get("heading_path", []),
        "start_line": int(span.get("start_line") or 0),
        "end_line": int(span.get("end_line") or span.get("start_line") or 0),
        "text_hash": span.get("text_hash"),
        "metadata": span.get("metadata", {}),
    }


def _safe_source_path(source_root: Path, source_path: str) -> Path | None:
    candidate = (source_root / source_path).resolve()
    try:
        candidate.relative_to(source_root)
    except ValueError:
        return None
    return candidate


def _text_for_line_range(lines: list[str], start_line: int, end_line: int) -> str:
    if start_line <= 0 or end_line <= 0 or end_line < start_line:
        return ""
    start = start_line - 1
    end = end_line
    if start >= len(lines):
        return ""
    return "\n".join(lines[start:min(end, len(lines))])


def _find_exact_text(lines: list[str], text: str) -> tuple[int, int, str] | None:
    if not text:
        return None
    wanted = text.splitlines()
    if not wanted:
        return None
    width = len(wanted)
    for index in range(0, len(lines) - width + 1):
        current = lines[index:index + width]
        if current == wanted:
            return index + 1, index + width, "\n".join(current)
    return None


def _query_lexical_items(
    db: Any,
    workspace_id: str,
    query: str,
    *,
    limit: int,
    source_roles: tuple[str, ...] | None,
    span_kinds: tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    query_lexical_spans = getattr(db, "query_lexical_spans", None)
    if not callable(query_lexical_spans):
        return []
    rows = cast(
        list[dict[str, Any]],
        query_lexical_spans(
            workspace_id,
            query,
            limit=limit,
            source_roles=source_roles,
            span_kinds=span_kinds,
        ),
    )
    items: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        row = dict(row)
        span_id = str(row.get("span_id") or "")
        if not span_id:
            continue
        items.append(
            {
                "doc_id": f"lexical:{span_id}",
                "record_type": "lexical_span",
                "record_id": span_id,
                "record": _record_from_lexical_span(row),
                "neighbors": [],
                "routes": [str(row.get("route") or "lexical")],
                "route_ranks": {"lexical": rank},
                "lexical_span": row,
            }
        )
    return items


def _record_from_lexical_span(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_type": "lexical_span",
        "record_id": row.get("span_id"),
        "vector_text": row.get("text", ""),
        "source_path": row.get("source_path"),
        "source_id": row.get("source_id"),
        "payload": {
            "source_role": row.get("source_role"),
            "span_kind": row.get("span_kind"),
            "heading_path": row.get("heading_path", []),
            "start_line": row.get("start_line"),
            "end_line": row.get("end_line"),
            "text_hash": row.get("text_hash"),
            "metadata": row.get("metadata", {}),
        },
    }


def _rank_hybrid_hits(items: list[dict[str, Any]], *, query: str) -> list[dict[str, Any]]:
    scored = []
    terms = _query_terms(query)
    for item in items:
        breakdown = _score_breakdown(item, terms)
        score = sum(float(value) for value in breakdown.values())
        item["score"] = score
        item["score_breakdown"] = breakdown
        scored.append(item)
    scored.sort(key=lambda item: (-float(item.get("score", 0.0)), str(item.get("record", {}).get("source_path") or ""), str(item.get("record_id") or "")))
    return scored


def _score_breakdown(item: dict[str, Any], terms: list[str]) -> dict[str, float]:
    ranks = item.get("route_ranks", {}) if isinstance(item.get("route_ranks"), dict) else {}
    zvec_rank = int(ranks.get("zvec", 0) or 0)
    lexical_rank = int(ranks.get("lexical", 0) or 0)
    breakdown = {
        "zvec_route": 80.0 / (60 + zvec_rank) if zvec_rank else 0.0,
        "lexical_route": 160.0 / (60 + lexical_rank) if lexical_rank else 0.0,
        "zvec_score": min(float(item.get("zvec_score", 0.0)) * 0.01, 5.0),
        "source_role": 0.0,
        "span_kind": 0.0,
        "exact_terms": 0.0,
    }
    record = item.get("record", {}) if isinstance(item.get("record"), dict) else {}
    payload = record.get("payload", {}) if isinstance(record.get("payload"), dict) else {}
    source_role = str(payload.get("source_role") or "")
    span_kind = str(payload.get("span_kind") or "")
    if source_role == "meta_map":
        breakdown["source_role"] = 30.0
    elif source_role == "raw":
        breakdown["source_role"] = 18.0
    elif source_role == "compiled":
        breakdown["source_role"] = 8.0
    if span_kind == "map.row":
        breakdown["span_kind"] = 30.0
    elif span_kind == "table.row":
        breakdown["span_kind"] = 24.0
    elif span_kind.startswith("raw."):
        breakdown["span_kind"] = 12.0
    haystack = " ".join(
        str(part or "")
        for part in [
            record.get("source_path"),
            record.get("source_id"),
            record.get("vector_text"),
            " ".join(str(part) for part in payload.get("heading_path", []) if part),
        ]
    ).lower()
    matched = sum(1 for term in terms if term.lower() in haystack)
    breakdown["exact_terms"] = min(40.0, matched * 10.0)
    return breakdown


def _query_terms(query: str) -> list[str]:
    import re

    return [term for term in re.findall(r"[0-9A-Za-z_\u4e00-\u9fff]+", query) if term]


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
    record_type = str(fields.get("record_type") or "")
    record_id = str(fields.get("record_id") or "")
    if not record_type or not record_id:
        raise ValueError(f"zvec hit missing record_type/record_id fields: {doc_id}")
    return record_type, record_id
