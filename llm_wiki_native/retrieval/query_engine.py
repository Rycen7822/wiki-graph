"""Data-only native query engine.

This module intentionally accepts a precomputed query vector. Embedding calls,
LLM answer generation, and HTTP serving live in the API/search layers.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Sequence, cast

from llm_wiki_native.contracts import (
    DEFAULT_NEIGHBOR_LIMIT,
    DEFAULT_QUERY_MODE,
    DEFAULT_QUERY_RECORD_TYPES,
    DEFAULT_RETRIEVAL_GOAL,
    DEFAULT_TOP_K,
    MAX_NEIGHBOR_LIMIT,
    MAX_TOP_K,
    RECORD_TYPE_CODES,
    RECORD_TYPES,
    SECTION_KIND_CODES,
    SUPPORTED_QUERY_MODES,
    SUPPORTED_RETRIEVAL_GOALS,
    WORKSPACE_SCHEMA_VERSION,
)
from llm_wiki_native.retrieval.relevance import (
    candidate_limit,
    normalize_query_terms,
    plan_relevance,
    rank_route_candidates,
    source_key,
)



class NativeQueryEngine:
    def __init__(
        self,
        db: Any,
        zvec_workspace: Any,
        *,
        source_root: Path | None = None,
    ) -> None:
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
        mode: str = DEFAULT_QUERY_MODE,
        top_k: int = DEFAULT_TOP_K,
        record_types: tuple[str, ...] | list[str] = DEFAULT_QUERY_RECORD_TYPES,
        section_kind: str | None = None,
        neighbor_limit: int = DEFAULT_NEIGHBOR_LIMIT,
        retrieval_goal: str = DEFAULT_RETRIEVAL_GOAL,
        include_lexical: bool = True,
        source_roles: tuple[str, ...] | None = None,
        span_kinds: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        if mode not in SUPPORTED_QUERY_MODES:
            raise ValueError(f"unsupported mode: {mode}")
        if retrieval_goal not in SUPPORTED_RETRIEVAL_GOALS:
            raise ValueError(f"unsupported retrieval_goal: {retrieval_goal}")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be an integer between 1 and {MAX_TOP_K}")
        if not isinstance(record_types, (tuple, list)) or not record_types:
            raise ValueError("record_types must be a non-empty list")
        if any(not isinstance(record_type, str) for record_type in record_types):
            raise ValueError("record_types must contain only strings")
        normalized_record_types = tuple(dict.fromkeys(record_types))
        unknown_record_types = [
            record_type
            for record_type in normalized_record_types
            if record_type not in RECORD_TYPES
        ]
        if unknown_record_types:
            raise ValueError(f"unsupported record_type: {unknown_record_types[0]}")
        if section_kind is not None and section_kind not in SECTION_KIND_CODES:
            raise ValueError(f"unknown section_kind: {section_kind}")
        if (
            isinstance(neighbor_limit, bool)
            or not isinstance(neighbor_limit, int)
            or not 0 <= neighbor_limit <= MAX_NEIGHBOR_LIMIT
        ):
            raise ValueError(
                f"neighbor_limit must be an integer between 0 and {MAX_NEIGHBOR_LIMIT}"
            )

        metadata = self.db.get_workspace_metadata(workspace_id)
        if metadata.get("workspace_id") != workspace_id:
            raise ValueError(
                f"workspace metadata mismatch: requested={workspace_id} "
                f"actual={metadata.get('workspace_id')}"
            )
        if metadata["schema_version"] != WORKSPACE_SCHEMA_VERSION:
            raise ValueError(
                f"workspace schema mismatch: {workspace_id} "
                f"schema={metadata['schema_version']} expected={WORKSPACE_SCHEMA_VERSION}"
            )
        if metadata["status"] != "audited":
            raise ValueError(
                f"workspace must be audited before query: "
                f"{workspace_id} status={metadata['status']}"
            )
        if mode == "bypass":
            return _bypass_result(
                query=query,
                top_k=top_k,
                record_types=normalized_record_types,
                section_kind=section_kind,
                retrieval_goal=retrieval_goal,
                workspace_metadata=metadata,
            )
        return self._query_relevance(
            workspace_id,
            query,
            query_vector,
            mode=mode,
            top_k=top_k,
            record_types=normalized_record_types,
            section_kind=section_kind,
            neighbor_limit=neighbor_limit,
            retrieval_goal=retrieval_goal,
            include_lexical=include_lexical,
            source_roles=source_roles,
            span_kinds=span_kinds,
            workspace_metadata=metadata,
        )

    def read_span(self, workspace_id: str, span_id: str) -> dict[str, Any]:
        try:
            span = self.db.get_lexical_span(workspace_id, span_id)
        except KeyError:
            span = _span_from_section_record(
                self.db.get_record(workspace_id, "section", span_id)
            )
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
        relocations = _find_exact_text_matches(lines, stored_text)
        if len(relocations) == 1:
            relocated_start, relocated_end, relocated_text = relocations[0]
            return {
                **response,
                "source_status": "current",
                "relocation": "exact_text",
                "start_line": relocated_start,
                "end_line": relocated_end,
                "text": relocated_text,
            }
        if len(relocations) > 1:
            return {
                **response,
                "source_status": "ambiguous",
                "relocation": "multiple_exact_text",
                "start_line": start_line,
                "end_line": end_line,
                "text": stored_text,
            }
        return {
            **response,
            "source_status": "stale",
            "relocation": "not_found",
            "start_line": start_line,
            "end_line": end_line,
            "text": stored_text,
        }

    def _query_relevance(
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
        retrieval_goal: str,
        include_lexical: bool,
        source_roles: tuple[str, ...] | None,
        span_kinds: tuple[str, ...] | None,
        workspace_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        base_budget = candidate_limit(top_k)
        section_budget = base_budget
        budget = max(base_budget, section_budget)
        route_started = perf_counter()
        navigation_candidates, section_candidates, lexical_candidates = (
            _collect_route_candidates(
                self.zvec_workspace,
                self.db,
                workspace_id,
                query,
                query_vector,
                mode=mode,
                navigation_limit=base_budget,
                section_limit=section_budget,
                lexical_limit=base_budget,
                record_types=record_types,
                section_kind=section_kind,
                include_lexical=include_lexical,
                source_roles=source_roles,
                span_kinds=span_kinds,
            )
        )
        route_ms = (perf_counter() - route_started) * 1000

        planner_started = perf_counter()
        plan = plan_relevance(
            [*navigation_candidates, *section_candidates, *lexical_candidates],
            query,
            top_k,
            retrieval_goal,
        )
        planner_ms = (perf_counter() - planner_started) * 1000
        hydrate_started = perf_counter()
        hits, record_calls, neighbor_calls = _hydrate_selected(
            self.db,
            workspace_id,
            plan["selected"],
            neighbor_limit=neighbor_limit,
        )
        hydrate_ms = (perf_counter() - hydrate_started) * 1000
        response = _query_response(
            hits,
            query=query,
            mode=mode,
            top_k=top_k,
            record_types=record_types,
            section_kind=section_kind,
            retrieval_goal=retrieval_goal,
            navigation_candidates=navigation_candidates,
            section_candidates=section_candidates,
            lexical_candidates=lexical_candidates,
            plan=plan,
            db_record_calls=record_calls,
            db_neighbor_calls=neighbor_calls,
            workspace_metadata=workspace_metadata,
            timings_ms={
                "route": route_ms,
                "planner": planner_ms,
                "hydrate": hydrate_ms,
            },
        )
        active_routes = _active_route_count(
            mode=mode,
            record_types=record_types,
            section_kind=section_kind,
            include_lexical=include_lexical,
        )
        route_candidates = [*navigation_candidates, *section_candidates, *lexical_candidates]
        response["trace"].update(
            _relevance_trace_details(
                workspace_metadata=workspace_metadata,
                plan=plan,
                route_candidates=route_candidates,
                active_routes=active_routes,
                budget=budget,
            )
        )
        return response


def _bypass_result(
    *,
    query: str,
    top_k: int,
    record_types: tuple[str, ...],
    section_kind: str | None,
    retrieval_goal: str,
    workspace_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "hits": [],
        "trace": {
            "query": query,
            "mode": "bypass",
            "top_k": top_k,
            "record_types": list(record_types),
            "section_kind": section_kind,
            "retrieval_goal": retrieval_goal,
            "vector_hit_count": 0,
            "lexical_hit_count": 0,
            "route_counts": {"zvec": 0, "lexical": 0},
            "family_candidate_counts": {
                "zvec_navigation": 0,
                "zvec_section": 0,
                "lexical": 0,
            },
            "retrieval_backend": "bypass",
            "db_record_calls": 0,
            "db_neighbor_calls": 0,
            "workspace_metadata": workspace_metadata,
            "workspace_id": workspace_metadata["workspace_id"],
            "source_manifest_hash": workspace_metadata["source_manifest_hash"],
            "workspace_schema_version": workspace_metadata["schema_version"],
            "workspace_status": workspace_metadata["status"],
            "merged_candidate_count": 0,
            "source_scope_count": 0,
            "eligible_evidence_count": 0,
            "selected_block_count": 0,
            "distinct_selected_source_count": 0,
            "coverage_fill_pass_used": False,
            "active_route_count": 0,
            "candidate_card_limit": 0,
            "candidate_cards": [],
            "source_scope": [],
            "planner_decisions": [],
            "timings_ms": {"route": 0.0, "planner": 0.0, "hydrate": 0.0},
        },
    }


def _collect_route_candidates(
    zvec_workspace: Any,
    db: Any,
    workspace_id: str,
    query: str,
    query_vector: list[float],
    *,
    mode: str,
    navigation_limit: int,
    section_limit: int,
    lexical_limit: int,
    record_types: tuple[str, ...],
    section_kind: str | None,
    include_lexical: bool,
    source_roles: tuple[str, ...] | None,
    span_kinds: tuple[str, ...] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if section_kind is not None:
        navigation_candidates: list[dict[str, Any]] = []
        section_candidates = _query_zvec_candidates(
            zvec_workspace,
            query,
            query_vector,
            mode=mode,
            limit=section_limit,
            record_types=("section",),
            section_kind=section_kind,
            route_family="zvec_section",
        )
    else:
        navigation_types = tuple(
            record_type for record_type in record_types if record_type != "section"
        )
        navigation_candidates = _query_zvec_candidates(
            zvec_workspace,
            query,
            query_vector,
            mode=mode,
            limit=navigation_limit,
            record_types=navigation_types,
            section_kind=None,
            route_family="zvec_navigation",
        )
        section_candidates = _query_zvec_candidates(
            zvec_workspace,
            query,
            query_vector,
            mode=mode,
            limit=section_limit,
            record_types=("section",) if "section" in record_types else (),
            section_kind=None,
            route_family="zvec_section",
        )
    lexical_candidates = (
        _query_lexical_candidates(
            db,
            workspace_id,
            query,
            normalized_terms=normalize_query_terms(query),
            limit=lexical_limit,
            source_roles=source_roles,
            span_kinds=span_kinds,
        )
        if include_lexical and mode == "mix" and section_kind is None
        else []
    )
    return navigation_candidates, section_candidates, lexical_candidates


def _active_route_count(
    *,
    mode: str,
    record_types: tuple[str, ...],
    section_kind: str | None,
    include_lexical: bool,
) -> int:
    if section_kind is not None:
        return 1
    count = int(any(record_type != "section" for record_type in record_types))
    count += int("section" in record_types)
    count += int(mode == "mix" and include_lexical)
    return count


def _candidate_identity_cards(candidates: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for candidate in candidates[:limit]:
        cards.append(
            {
                "record_type": candidate["record_type"],
                "record_id": candidate["record_id"],
                "source_path": candidate.get("source_path"),
                "source_id": candidate.get("source_id"),
                "source_key": source_key(candidate),
                "route_family": candidate.get("route_family"),
                "route_rank": candidate.get("route_rank"),
                "routes": list(candidate.get("routes") or []),
            }
        )
    return cards


def _relevance_trace_details(
    *,
    workspace_metadata: dict[str, Any],
    plan: dict[str, Any],
    route_candidates: list[dict[str, Any]],
    active_routes: int,
    budget: int,
) -> dict[str, Any]:
    planner_trace = plan["trace"]
    return {
        "workspace_id": workspace_metadata["workspace_id"],
        "source_manifest_hash": workspace_metadata["source_manifest_hash"],
        "workspace_schema_version": workspace_metadata["schema_version"],
        "workspace_status": workspace_metadata["status"],
        "merged_candidate_count": planner_trace["merged_candidate_count"],
        "source_scope_count": planner_trace["source_scope_count"],
        "eligible_evidence_count": planner_trace["eligible_evidence_count"],
        "selected_block_count": planner_trace["selected_block_count"],
        "distinct_selected_source_count": planner_trace["distinct_selected_source_count"],
        "coverage_fill_pass_used": planner_trace["coverage_fill_pass_used"],
        "active_route_count": active_routes,
        "candidate_card_limit": active_routes * budget,
        "candidate_cards": _candidate_identity_cards(
            route_candidates,
            limit=active_routes * budget,
        ),
    }


def _query_zvec_candidates(
    zvec_workspace: Any,
    query: str,
    query_vector: list[float],
    *,
    mode: str,
    limit: int,
    record_types: tuple[str, ...],
    section_kind: str | None,
    route_family: str,
) -> list[dict[str, Any]]:
    if not record_types:
        return []
    hits = _query_zvec_hits(
        zvec_workspace,
        query,
        query_vector,
        mode=mode,
        top_k=limit,
        record_types=record_types,
        section_kind=section_kind,
    )
    candidates = [_candidate_from_zvec_hit(hit) for hit in hits[:limit]]
    return rank_route_candidates(candidates, route_family=route_family)


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


def _candidate_from_zvec_hit(hit: Any) -> dict[str, Any]:
    fields = dict(hit.fields)
    record_type, record_id = _record_identity_from_hit(hit.doc_id, fields)
    return {
        "doc_id": str(hit.doc_id),
        "record_type": record_type,
        "record_id": record_id,
        "raw_score": float(hit.score),
        "content": str(fields.get("content") or ""),
        "content_hash": str(fields.get("content_hash") or ""),
        "source_path": str(fields.get("source_path") or ""),
        "source_id": str(fields.get("source_id") or ""),
        "source_kind_code": fields.get("source_kind_code"),
        "section_kind_code": fields.get("section_kind_code"),
        "title": str(fields.get("title") or ""),
        "routes": ["zvec"],
    }


def _query_lexical_candidates(
    db: Any,
    workspace_id: str,
    query: str,
    *,
    normalized_terms: list[str],
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
            normalized_terms=tuple(normalized_terms),
        ),
    )
    candidates: list[dict[str, Any]] = []
    for rank, raw_row in enumerate(rows[:limit], start=1):
        row = dict(raw_row)
        span_id = str(row.get("span_id") or "")
        if not span_id:
            continue
        candidates.append(
            {
                "doc_id": f"lexical:{span_id}",
                "record_type": "lexical_span",
                "record_id": span_id,
                "raw_score": float(row.get("lexical_rank") or 0.0),
                "content": str(row.get("text") or ""),
                "text_hash": str(row.get("text_hash") or ""),
                "source_path": str(row.get("source_path") or ""),
                "source_id": str(row.get("source_id") or ""),
                "source_role": str(row.get("source_role") or ""),
                "span_kind": str(row.get("span_kind") or ""),
                "heading_path": list(row.get("heading_path") or []),
                "start_line": int(row.get("start_line") or 0),
                "end_line": int(row.get("end_line") or 0),
                "routes": [str(row.get("route") or "lexical")],
                "route_family": "lexical",
                "route_rank": rank,
                "lexical_span": row,
            }
        )
    return candidates


def _hydrate_selected(
    db: Any,
    workspace_id: str,
    selected: list[dict[str, Any]],
    *,
    neighbor_limit: int,
) -> tuple[list[dict[str, Any]], int, int]:
    hits: list[dict[str, Any]] = []
    record_calls = 0
    neighbor_calls = 0
    for candidate in selected:
        record_type = str(candidate["record_type"])
        record_id = str(candidate["record_id"])
        if record_type == "lexical_span":
            span = dict(candidate.get("lexical_span") or {})
            record = _record_from_lexical_span(span)
            neighbors: list[dict[str, Any]] = []
        else:
            record_calls += 1
            record = db.get_record(workspace_id, record_type, record_id)
            if neighbor_limit > 0:
                neighbor_calls += 1
                neighbors = db.neighbors(workspace_id, record_id, limit=neighbor_limit)
            else:
                neighbors = []
        hydrated_source_key = str(
            candidate.get("_source_key")
            or candidate.get("source_path")
            or record.get("source_path")
            or candidate.get("source_id")
            or record.get("source_id")
            or record_id
        )
        hit: dict[str, Any] = {
            "doc_id": candidate.get("doc_id") or f"{record_type}:{record_id}",
            "record_type": record_type,
            "record_id": record_id,
            "source_key": hydrated_source_key,
            "evidence_hash": candidate.get("evidence_hash"),
            "record": record,
            "neighbors": neighbors,
            "routes": list(candidate.get("routes") or []),
            "route_ranks": dict(candidate.get("route_ranks") or {}),
            "score": float(candidate["score"]),
            "ranking_contract": candidate["ranking_contract"],
            "relevance_score_breakdown": dict(candidate["relevance_score_breakdown"]),
        }
        raw_scores = candidate.get("raw_scores")
        if isinstance(raw_scores, dict):
            zvec_scores = [
                float(score)
                for family, score in raw_scores.items()
                if str(family).startswith("zvec_")
            ]
            if zvec_scores:
                hit["zvec_score"] = max(zvec_scores)
        if record_type == "lexical_span":
            hit["lexical_span"] = dict(candidate.get("lexical_span") or {})
        hit["score_breakdown"] = _compatibility_score_breakdown(candidate)
        hits.append(hit)
    return hits, record_calls, neighbor_calls


def _compatibility_score_breakdown(candidate: dict[str, Any]) -> dict[str, float]:
    ranks = candidate.get("route_ranks")
    route_ranks = ranks if isinstance(ranks, dict) else {}
    navigation_rank = int(route_ranks.get("zvec_navigation", 0) or 0)
    section_rank = int(route_ranks.get("zvec_section", 0) or 0)
    zvec_rank = min(
        (rank for rank in (navigation_rank, section_rank) if rank > 0),
        default=0,
    )
    lexical_rank = int(route_ranks.get("lexical", 0) or 0)
    raw_scores = candidate.get("raw_scores")
    score_values = raw_scores if isinstance(raw_scores, dict) else {}
    zvec_scores = [
        float(score)
        for family, score in score_values.items()
        if str(family).startswith("zvec_")
    ]
    relevance = candidate.get("relevance_score_breakdown")
    relevance_values = relevance if isinstance(relevance, dict) else {}
    return {
        "zvec_route": 1.0 / zvec_rank if zvec_rank else 0.0,
        "lexical_route": 1.0 / lexical_rank if lexical_rank else 0.0,
        "zvec_score": max(zvec_scores, default=0.0),
        "source_role": 1.0 if candidate.get("source_role") else 0.0,
        "span_kind": 1.0 if candidate.get("span_kind") else 0.0,
        "exact_terms": float(relevance_values.get("term_coverage", 0.0)),
    }


def _query_response(
    hits: list[dict[str, Any]],
    *,
    query: str,
    mode: str,
    top_k: int,
    record_types: tuple[str, ...],
    section_kind: str | None,
    retrieval_goal: str,
    navigation_candidates: list[dict[str, Any]],
    section_candidates: list[dict[str, Any]],
    lexical_candidates: list[dict[str, Any]],
    plan: dict[str, Any],
    db_record_calls: int,
    db_neighbor_calls: int,
    workspace_metadata: dict[str, Any],
    timings_ms: dict[str, float],
) -> dict[str, Any]:
    zvec_count = len(navigation_candidates) + len(section_candidates)
    lexical_count = len(lexical_candidates)
    return {
        "hits": hits,
        "trace": {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "record_types": list(record_types),
            "section_kind": section_kind,
            "retrieval_goal": retrieval_goal,
            "vector_hit_count": zvec_count,
            "lexical_hit_count": lexical_count,
            "route_counts": {"zvec": zvec_count, "lexical": lexical_count},
            "family_candidate_counts": {
                "zvec_navigation": len(navigation_candidates),
                "zvec_section": len(section_candidates),
                "lexical": lexical_count,
            },
            "retrieval_backend": "zvec+lexical" if lexical_count else "zvec",
            "db_record_calls": db_record_calls,
            "db_neighbor_calls": db_neighbor_calls,
            "workspace_metadata": workspace_metadata,
            "source_scope": plan["source_scope"],
            "planner_decisions": plan["decisions"],
            "planner": plan["trace"],
            "timings_ms": timings_ms,
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


def _span_from_section_record(record: dict[str, Any]) -> dict[str, Any]:
    payload_value = record.get("payload")
    payload = payload_value if isinstance(payload_value, dict) else {}
    return {
        "span_id": record.get("record_id"),
        "source_path": record.get("source_path"),
        "source_id": record.get("source_id"),
        "source_role": payload.get("source_role") or "raw",
        "span_kind": "raw.section",
        "heading_path": payload.get("heading_path", []),
        "start_line": 0,
        "end_line": 0,
        "text": record.get("vector_text", ""),
        "text_hash": payload.get("text_hash") or record.get("content_hash"),
        "metadata": {"section_kind": payload.get("section_kind")},
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


def _find_exact_text_matches(lines: list[str], text: str) -> list[tuple[int, int, str]]:
    if not text:
        return []
    wanted = text.splitlines()
    if not wanted:
        return []
    width = len(wanted)
    matches: list[tuple[int, int, str]] = []
    for index in range(0, len(lines) - width + 1):
        current = lines[index:index + width]
        if current == wanted:
            matches.append((index + 1, index + width, "\n".join(current)))
            if len(matches) == 2:
                break
    return matches


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


def _zvec_filter(record_types: tuple[str, ...], section_kind: str | None) -> str:
    if section_kind is not None:
        try:
            section_kind_code = SECTION_KIND_CODES[section_kind]
        except KeyError as exc:
            raise ValueError(f"unknown section_kind: {section_kind}") from exc
        return (
            f"record_type_code in (4) and "
            f"section_kind_code in ({section_kind_code})"
        )
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
