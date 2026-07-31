"""Pure deterministic relevance planning for lightweight retrieval candidates."""

from __future__ import annotations

from bisect import bisect_right
import hashlib
import math
import re
from typing import Any, Iterable

from llm_wiki_native.contracts import SECTION_KIND_CODES, SOURCE_KIND_CODES

_ROUTE_FAMILIES = ("zvec_navigation", "zvec_section", "lexical")
_EXTERNAL_ROUTES = ("zvec", "lexical_fts", "lexical_like")
_ROUTE_FAMILY_ORDER = {family: index for index, family in enumerate(_ROUTE_FAMILIES)}
_EXTERNAL_ROUTE_ORDER = {"zvec": 0, "lexical_fts": 1, "lexical_like": 2}
_EXACT_HEADING_TERM_COVERAGE = 0.75
_EXACT_HEADING_MIN_MATCHED_TERMS = 3
_NUMERIC_ANSWER_TERMS = {"capacity", "count", "number", "size", "many"}
_LOCAL_RANK_WEIGHT = 0.15
_SOURCE_RANK_WEIGHT = 0.05
_TERM_COVERAGE_WEIGHT = 0.50
_EVIDENCE_QUALITY_WEIGHT = 0.30
_PROSE_EVIDENCE_KINDS = frozenset({"raw_section", "raw.section", "compiled_chunk"})
_RANKING_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "below",
        "by",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "of",
        "or",
        "the",
        "to",
        "under",
        "used",
        "using",
        "was",
        "were",
        "what",
        "when",
        "which",
        "with",
    }
)
_SECTION_KIND_BY_CODE = {code: name for name, code in SECTION_KIND_CODES.items()}
_SOURCE_KIND_BY_CODE = {code: name for name, code in SOURCE_KIND_CODES.items()}
_ASCII_OR_CJK_RUN = re.compile(
    r"(?P<ascii>\d{1,3}(?:,\d{3})+|[A-Za-z0-9]+)|"
    r"(?P<cjk>[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\U00020000-\U0002ebef\U00030000-\U0003134f]+)"
)


def _validated_top_k(top_k: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    return top_k


def candidate_limit(top_k: int, retrieval_goal: str | None = None) -> int:
    value = _validated_top_k(top_k)
    if retrieval_goal not in {None, "focused", "coverage"}:
        raise ValueError(f"unsupported retrieval_goal: {retrieval_goal}")
    return min(200, max(40, value * 5))


def scope_limit(top_k: int, retrieval_goal: str | None = None) -> int:
    value = _validated_top_k(top_k)
    if retrieval_goal == "coverage":
        return 200
    return min(candidate_limit(value, retrieval_goal), max(20, value * 3))


def normalize_query_terms(query: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for match in _ASCII_OR_CJK_RUN.finditer(str(query)):
        ascii_run = match.group("ascii")
        if ascii_run is not None:
            normalized = ascii_run.replace(",", "").lower()
            additions: Iterable[str] = (normalized,) if len(normalized) >= 2 else ()
        else:
            cjk_run = match.group("cjk") or ""
            additions = (
                (cjk_run,)
                if len(cjk_run) <= 4
                else (cjk_run[index : index + 2] for index in range(len(cjk_run) - 1))
            )
        for term in additions:
            if term in seen:
                continue
            seen.add(term)
            terms.append(term)
            if len(terms) == 32:
                return terms
    return terms


def _finite_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("-inf")
    score = float(value)
    return score if math.isfinite(score) else float("-inf")


def rank_route_candidates(
    candidates: Iterable[dict[str, Any]], *, route_family: str
) -> list[dict[str, Any]]:
    if route_family not in _ROUTE_FAMILIES:
        raise ValueError(f"unsupported route family: {route_family}")
    ranked = sorted(
        (dict(candidate) for candidate in candidates),
        key=lambda candidate: (
            -_finite_score(candidate.get("raw_score")),
            str(candidate.get("record_type") or ""),
            str(candidate.get("record_id") or ""),
        ),
    )
    for rank, candidate in enumerate(ranked, start=1):
        if not math.isfinite(_finite_score(candidate.get("raw_score"))):
            candidate.pop("raw_score", None)
        candidate["route_family"] = route_family
        candidate["route_rank"] = rank
    return ranked


def _route_rank_items(candidate: dict[str, Any]) -> list[tuple[str, int]]:
    existing = candidate.get("route_ranks")
    if isinstance(existing, dict):
        items = []
        for family, rank in existing.items():
            if family in _ROUTE_FAMILIES and isinstance(rank, int) and not isinstance(rank, bool) and rank > 0:
                items.append((family, rank))
        return items
    family = candidate.get("route_family")
    rank = candidate.get("route_rank")
    if family in _ROUTE_FAMILIES and isinstance(rank, int) and not isinstance(rank, bool) and rank > 0:
        return [(str(family), rank)]
    return []


def _best_route_rank(candidate: dict[str, Any]) -> int:
    items = _route_rank_items(candidate)
    return min((rank for _, rank in items), default=10**9)


def _family_order(family: str) -> tuple[int, str]:
    try:
        return (_ROUTE_FAMILIES.index(family), family)
    except ValueError:
        return (len(_ROUTE_FAMILIES), family)


def _ordered_external_routes(routes: set[str]) -> list[str]:
    known = [route for route in _EXTERNAL_ROUTES if route in routes]
    return [*known, *sorted(routes - set(_EXTERNAL_ROUTES))]


def merge_candidates(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for raw_candidate in candidates:
        candidate = dict(raw_candidate)
        record_type = candidate.get("record_type")
        record_id = candidate.get("record_id")
        if not isinstance(record_type, str) or not record_type:
            continue
        if not isinstance(record_id, str) or not record_id:
            continue
        grouped.setdefault((record_type, record_id), []).append(candidate)

    merged: list[dict[str, Any]] = []
    for identity in sorted(grouped):
        members = grouped[identity]
        representative = min(
            members,
            key=lambda candidate: (
                _best_route_rank(candidate),
                _family_order(str(candidate.get("route_family") or "")),
                str(candidate.get("source_path") or ""),
                str(candidate.get("source_id") or ""),
                str(candidate.get("content_hash") or candidate.get("text_hash") or ""),
                _evidence_hash(candidate),
            ),
        )
        item = dict(representative)
        if not math.isfinite(_finite_score(item.get("raw_score"))):
            item.pop("raw_score", None)
        ranks: dict[str, int] = {}
        raw_scores: dict[str, float] = {}
        external_routes: set[str] = set()
        for member in members:
            for family, rank in _route_rank_items(member):
                ranks[family] = min(rank, ranks.get(family, rank))
                member_raw_scores = member.get("raw_scores")
                raw_value = (
                    member_raw_scores.get(family)
                    if isinstance(member_raw_scores, dict)
                    else member.get("raw_score")
                )
                score = _finite_score(raw_value)
                if math.isfinite(score):
                    raw_scores[family] = max(score, raw_scores.get(family, score))
            routes = member.get("routes")
            if isinstance(routes, list):
                external_routes.update(route for route in routes if isinstance(route, str) and route)
        item["route_ranks"] = {
            family: ranks[family] for family in sorted(ranks, key=_family_order)
        }
        item["raw_scores"] = {
            family: raw_scores[family] for family in sorted(raw_scores, key=_family_order)
        }
        item["route_rank"] = min(ranks.values(), default=10**9)
        item["routes"] = _ordered_external_routes(external_routes)
        merged.append(item)
    return merged


def source_key(candidate: dict[str, Any]) -> str:
    for key in ("_source_key", "source_path", "source_id", "record_id"):
        value = candidate.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _raw_source_stem(source_id: object) -> str | None:
    if not isinstance(source_id, str):
        return None
    if source_id.startswith("raw_clip:"):
        return source_id.removeprefix("raw_clip:") or None
    if source_id.startswith("raw_section:"):
        parts = source_id.split(":", 2)
        return parts[1] if len(parts) == 3 and parts[1] else None
    if source_id.startswith("method:"):
        parts = source_id.split(":", 2)
        return parts[1] if len(parts) == 3 and parts[1] else None
    return None


def _source_stem_key(stem: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", stem.lower())


def _assign_canonical_source_keys(candidates: list[dict[str, Any]]) -> None:
    raw_paths: dict[str, str] = {}
    for candidate in candidates:
        stem = _raw_source_stem(candidate.get("source_id"))
        path = candidate.get("source_path")
        if stem and isinstance(path, str) and path.startswith("raw/clip/"):
            raw_paths[_source_stem_key(stem)] = path
    for candidate in candidates:
        stem = _raw_source_stem(candidate.get("source_id"))
        if not stem:
            continue
        path = raw_paths.get(_source_stem_key(stem))
        if path is None and len(stem) >= 4 and stem[:4].isdigit():
            path = f"raw/clip/{stem[:4]}/{stem}.md"
        if path:
            candidate["_source_key"] = path


def build_source_scope(
    candidates: Iterable[dict[str, Any]],
    *,
    top_k: int,
    retrieval_goal: str | None = None,
) -> list[dict[str, Any]]:
    per_source: dict[str, dict[str, float]] = {}
    for candidate in candidates:
        key = source_key(candidate)
        if not key:
            continue
        route_values = per_source.setdefault(key, {})
        for family, rank in _route_rank_items(candidate):
            value = 1.0 / rank
            route_values[family] = max(value, route_values.get(family, 0.0))

    cards: list[dict[str, Any]] = []
    for key, route_values in per_source.items():
        ordered_values = sorted(route_values.values(), reverse=True)
        if not ordered_values:
            continue
        score = ordered_values[0] + 0.25 * sum(ordered_values[1:])
        cards.append(
            {
                "source_key": key,
                "source_score": score,
                "route_values": {
                    family: route_values[family]
                    for family in sorted(route_values, key=_family_order)
                },
            }
        )
    cards.sort(key=lambda card: (-card["source_score"], card["source_key"]))
    cards = cards[: scope_limit(top_k, retrieval_goal)]
    for rank, card in enumerate(cards, start=1):
        card["source_rank"] = rank
    return cards


def _candidate_text(candidate: dict[str, Any], *, include_identity: bool = False) -> str:
    content = ""
    for key in ("content", "text", "vector_text"):
        value = candidate.get(key)
        if isinstance(value, str):
            content = value
            break
    if not include_identity:
        return content
    parts = [content]
    for key in ("title", "source_path"):
        value = candidate.get(key)
        if isinstance(value, str) and value:
            parts.append(value)
    heading_path = candidate.get("heading_path")
    if isinstance(heading_path, (list, tuple)):
        parts.extend(str(part) for part in heading_path if str(part))
    return "\n".join(parts)


def _evidence_hash(candidate: dict[str, Any]) -> str:
    for key in ("text_hash", "content_hash"):
        value = candidate.get(key)
        if isinstance(value, str) and value:
            return value
    normalized = " ".join(_candidate_text(candidate).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _section_kind(candidate: dict[str, Any]) -> str | None:
    value = candidate.get("section_kind")
    if isinstance(value, str) and value and value != "none":
        return value
    code = candidate.get("section_kind_code")
    if isinstance(code, int) and not isinstance(code, bool):
        decoded = _SECTION_KIND_BY_CODE.get(code)
        return decoded if decoded and decoded != "none" else None
    return None


def _evidence_group(candidate: dict[str, Any], evidence_hash: str) -> str:
    kind = _section_kind(candidate)
    if kind:
        return f"section:{kind}"
    heading = candidate.get("heading_path")
    if isinstance(heading, (list, tuple)):
        parts = [str(part) for part in heading if str(part)]
        if parts:
            return "heading:" + "\x1f".join(parts)
    return f"hash:{evidence_hash}"


def _section_evidence_quality(candidate: dict[str, Any]) -> float:
    kind = _section_kind(candidate)
    if kind in {"methodology", "results", "claims"}:
        return 1.0
    if kind == "summary":
        return 0.95
    if kind == "abstract":
        return 0.85
    if kind in {"questions", "future"}:
        return 0.75
    return 0.9


def _evidence_class(candidate: dict[str, Any]) -> tuple[bool, float, str] | None:
    record_type = str(candidate.get("record_type") or "")
    span_kind = str(candidate.get("span_kind") or "")
    source_role = str(candidate.get("source_role") or "")
    source_path = str(candidate.get("source_path") or "")
    if source_role == "meta_map" or source_path.startswith("_meta/"):
        return False, 0.4, "map"
    source_kind = candidate.get("source_kind")
    if not isinstance(source_kind, str):
        source_kind_code = candidate.get("source_kind_code")
        source_kind = (
            _SOURCE_KIND_BY_CODE.get(source_kind_code, "")
            if isinstance(source_kind_code, int) and not isinstance(source_kind_code, bool)
            else ""
        )

    if record_type == "section" and source_kind == "raw":
        return (True, _section_evidence_quality(candidate), "raw_section")
    if span_kind == "raw.section":
        return (True, _section_evidence_quality(candidate), span_kind)
    if span_kind == "table.row":
        return (True, 1.0, span_kind)
    if record_type == "chunk" and source_kind == "compiled":
        return (True, 0.7, "compiled_chunk")
    if span_kind in {"doc.heading", "heading"}:
        return (False, 0.5, "heading")
    if span_kind == "map.row":
        return (False, 0.4, "map_row")
    if record_type in {"entity", "relationship"}:
        return (False, 0.2, record_type)
    return None


def _term_coverage(
    text: str,
    terms: list[str],
    *,
    numeric_weight: int = 1,
) -> float:
    if not terms:
        return 0.0
    lowered = text.lower()
    numeric_text = lowered
    if any(term.isascii() and term.isdecimal() for term in terms):
        numeric_text = lowered.replace(",", "").replace("{", "").replace("}", "")
    matched_weight = 0
    total_weight = 0
    for term in terms:
        numeric = term.isascii() and term.isdecimal()
        weight = numeric_weight if numeric else 1
        total_weight += weight
        if numeric:
            term_matched = (
                re.search(rf"(?<!\d){re.escape(term)}(?!\d)", numeric_text)
                is not None
            )
        else:
            term_matched = term in lowered
        if term_matched:
            matched_weight += weight
    return matched_weight / total_weight


def _evidence_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    start_line = candidate.get("start_line")
    if isinstance(start_line, bool) or not isinstance(start_line, int):
        start_line = 0
    return (
        -candidate["score"],
        candidate["source_rank"],
        candidate["source_key"],
        start_line,
        str(candidate.get("record_id") or ""),
    )


def _decorate_evidence(
    candidate: dict[str, Any],
    *,
    terms: list[str],
    source_rank: int,
    include_identity_terms: bool,
    classification: tuple[bool, float, str],
) -> dict[str, Any]:
    primary, quality, evidence_kind = classification
    route_ranks = dict(candidate.get("route_ranks") or {})
    best_rank = min(route_ranks.values())
    local_rank_value = 1.0 / best_rank
    source_rank_value = 1.0 / source_rank
    term_coverage = _term_coverage(
        _candidate_text(candidate, include_identity=include_identity_terms),
        terms,
        numeric_weight=len(terms) if evidence_kind == "table.row" else 1,
    )
    score = (
        _LOCAL_RANK_WEIGHT * local_rank_value
        + _SOURCE_RANK_WEIGHT * source_rank_value
        + _TERM_COVERAGE_WEIGHT * term_coverage
        + _EVIDENCE_QUALITY_WEIGHT * quality
    )
    evidence_hash = _evidence_hash(candidate)
    item = dict(candidate)
    item.update(
        {
            "source_key": source_key(candidate),
            "source_rank": source_rank,
            "evidence_hash": evidence_hash,
            "evidence_group": _evidence_group(candidate, evidence_hash),
            "evidence_kind": evidence_kind,
            "is_primary": primary,
            "score": score,
            "ranking_contract": "relevance-v1",
            "relevance_score_breakdown": {
                "local_rank_value": local_rank_value,
                "source_rank_value": source_rank_value,
                "term_coverage": term_coverage,
                "evidence_quality": quality,
            },
        }
    )
    return item


def _focused_select(
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    counts: dict[str, int],
    groups: dict[str, set[str]],
    *,
    top_k: int,
) -> None:
    prose_counts: dict[str, int] = {}
    for candidate in selected:
        if candidate["evidence_kind"] in _PROSE_EVIDENCE_KINDS:
            key = candidate["source_key"]
            prose_counts[key] = prose_counts.get(key, 0) + 1
    raw_section_sources = {
        candidate["source_key"]
        for candidate in candidates
        if candidate["evidence_kind"] in {"raw_section", "raw.section"}
    }
    prose_deferred: list[dict[str, Any]] = []
    compiled_deferred: list[dict[str, Any]] = []
    repeated_group_deferred: list[dict[str, Any]] = []
    for candidate in candidates:
        if len(selected) >= top_k:
            break
        key = candidate["source_key"]
        if counts.get(key, 0) >= 3:
            continue
        group = candidate["evidence_group"]
        is_prose = candidate["evidence_kind"] in _PROSE_EVIDENCE_KINDS
        if candidate["evidence_kind"] == "compiled_chunk" and key in raw_section_sources:
            compiled_deferred.append(candidate)
            continue
        if group in groups.setdefault(key, set()):
            repeated_group_deferred.append(candidate)
            continue
        if is_prose and prose_counts.get(key, 0) >= 2:
            prose_deferred.append(candidate)
            continue
        selected.append(candidate)
        counts[key] = counts.get(key, 0) + 1
        groups[key].add(group)
        if is_prose:
            prose_counts[key] = prose_counts.get(key, 0) + 1
    for candidate in [
        *prose_deferred,
        *compiled_deferred,
        *repeated_group_deferred,
    ]:
        if len(selected) >= top_k:
            break
        key = candidate["source_key"]
        if counts.get(key, 0) >= 3:
            continue
        selected.append(candidate)
        counts[key] = counts.get(key, 0) + 1
        if candidate["evidence_kind"] in _PROSE_EVIDENCE_KINDS:
            prose_counts[key] = prose_counts.get(key, 0) + 1


def _coverage_select(
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    counts: dict[str, int],
    *,
    top_k: int,
) -> bool:
    chosen_ids = {id(candidate) for candidate in selected}
    candidates_by_source: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        candidates_by_source.setdefault(candidate["source_key"], []).append(candidate)
    for key, source_candidates in candidates_by_source.items():
        if len(selected) >= top_k:
            return False
        if counts.get(key, 0) > 0:
            continue
        representative = next(
            (
                item
                for item in source_candidates
                if item.get("evidence_group") == "section:summary"
            ),
            source_candidates[0],
        )
        selected.append(representative)
        chosen_ids.add(id(representative))
        counts[key] = 1
    if len(selected) >= top_k:
        return False
    fill_used = False
    for candidate in candidates:
        if len(selected) >= top_k:
            break
        if id(candidate) in chosen_ids:
            continue
        key = candidate["source_key"]
        if counts.get(key, 0) < 2:
            fill_used = True
            selected.append(candidate)
            chosen_ids.add(id(candidate))
            counts[key] = counts.get(key, 0) + 1
    return fill_used


def _decision(candidate: dict[str, Any], *, decision: str, reason: str) -> dict[str, Any]:
    return {
        "record_id": candidate.get("record_id"),
        "record_type": candidate.get("record_type"),
        "source_key": candidate.get("source_key") or source_key(candidate),
        "decision": decision,
        "reason": reason,
        "score": candidate.get("score"),
        "is_primary": candidate.get("is_primary"),
        "evidence_group": candidate.get("evidence_group"),
    }


def _unselected_reason(
    candidate: dict[str, Any],
    *,
    selected_count: int,
    selected_source_counts: dict[str, int],
    top_k: int,
    retrieval_goal: str,
) -> str:
    source_cap = 3 if retrieval_goal == "focused" else 2
    if selected_source_counts.get(candidate["source_key"], 0) >= source_cap:
        return "source_quota"
    if selected_count >= top_k:
        return "ranking_cutoff"
    return "source_quota"


def _prepare_evidence(
    merged: list[dict[str, Any]],
    *,
    terms: list[str],
    top_k: int,
    retrieval_goal: str,
    decision_limit: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[tuple[dict[str, Any], str]],
    list[tuple[dict[str, Any], str]],
]:
    scope_candidates = [
        candidate
        for candidate in merged
        if _evidence_class(candidate) is not None and _route_rank_items(candidate)
    ]
    source_scope = build_source_scope(
        scope_candidates,
        top_k=top_k,
        retrieval_goal=retrieval_goal,
    )
    source_ranks = {card["source_key"]: card["source_rank"] for card in source_scope}
    eligible: list[dict[str, Any]] = []
    ineligible_specs: list[tuple[dict[str, Any], str]] = []
    outside_specs: list[tuple[dict[str, Any], str]] = []
    for candidate in merged:
        classification = _evidence_class(candidate)
        if classification is None or not _route_rank_items(candidate):
            if len(ineligible_specs) < decision_limit:
                ineligible_specs.append((candidate, "ineligible_evidence"))
            continue
        source_rank = source_ranks.get(source_key(candidate))
        if source_rank is None:
            if len(outside_specs) < decision_limit:
                outside_specs.append((candidate, "outside_source_scope"))
            continue
        decorated = _decorate_evidence(
            candidate,
            terms=terms,
            source_rank=source_rank,
            include_identity_terms=retrieval_goal == "coverage",
            classification=classification,
        )
        if (
            decorated["evidence_kind"] == "heading"
            and decorated["relevance_score_breakdown"]["term_coverage"]
            >= _EXACT_HEADING_TERM_COVERAGE
            and round(
                decorated["relevance_score_breakdown"]["term_coverage"] * len(terms)
            )
            >= _EXACT_HEADING_MIN_MATCHED_TERMS
        ):
            decorated["is_primary"] = True
        eligible.append(decorated)
    return source_scope, eligible, ineligible_specs, outside_specs


def _deduplicate_evidence(
    eligible: list[dict[str, Any]], *, decision_limit: int
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], str]]]:
    eligible.sort(key=_evidence_sort_key)
    evidence_winners: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_specs: list[tuple[dict[str, Any], str]] = []
    for candidate in eligible:
        identity = (candidate["source_key"], candidate["evidence_hash"])
        existing = evidence_winners.get(identity)
        if existing is None:
            evidence_winners[identity] = candidate
            continue
        if candidate["is_primary"] and not existing["is_primary"]:
            evidence_winners[identity] = candidate
            duplicate = existing
        else:
            duplicate = candidate
        if len(duplicate_specs) < decision_limit:
            duplicate_specs.append((duplicate, "duplicate_evidence"))
    return sorted(evidence_winners.values(), key=_evidence_sort_key), duplicate_specs


def _select_evidence(
    candidates: list[dict[str, Any]], *, retrieval_goal: str, top_k: int
) -> tuple[list[dict[str, Any]], dict[str, int], bool]:
    primary = [candidate for candidate in candidates if candidate["is_primary"]]
    fallback = [candidate for candidate in candidates if not candidate["is_primary"]]
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    coverage_fill_pass_used = False
    if retrieval_goal == "focused":
        groups: dict[str, set[str]] = {}
        _focused_select(primary, selected, counts, groups, top_k=top_k)
        if len(selected) < top_k:
            _focused_select(fallback, selected, counts, groups, top_k=top_k)
    else:
        coverage_fill_pass_used = _coverage_select(primary, selected, counts, top_k=top_k)
        if len(selected) < top_k:
            coverage_fill_pass_used = (
                _coverage_select(fallback, selected, counts, top_k=top_k)
                or coverage_fill_pass_used
            )
    return selected, counts, coverage_fill_pass_used


def _bounded_decision_records(
    *,
    selected: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    selected_source_counts: dict[str, int],
    retrieval_goal: str,
    top_k: int,
    decision_limit: int,
    duplicate_specs: list[tuple[dict[str, Any], str]],
    ineligible_specs: list[tuple[dict[str, Any], str]],
    outside_specs: list[tuple[dict[str, Any], str]],
) -> list[dict[str, Any]]:
    selected_ids = {(candidate["record_type"], candidate["record_id"]) for candidate in selected}
    unselected_specs: list[tuple[dict[str, Any], str]] = []
    for candidate in candidates:
        identity = (candidate["record_type"], candidate["record_id"])
        if identity in selected_ids or len(unselected_specs) >= decision_limit:
            continue
        unselected_specs.append(
            (
                candidate,
                _unselected_reason(
                    candidate,
                    selected_count=len(selected),
                    selected_source_counts=selected_source_counts,
                    top_k=top_k,
                    retrieval_goal=retrieval_goal,
                ),
            )
        )
    decisions = [
        _decision(candidate, decision="selected", reason="selected")
        for candidate in selected
    ]
    remaining = decision_limit - len(decisions)
    ordered_rejections = [
        *duplicate_specs,
        *unselected_specs,
        *ineligible_specs,
        *outside_specs,
    ][:remaining]
    decisions.extend(
        _decision(candidate, decision="rejected", reason=reason)
        for candidate, reason in ordered_rejections
    )
    return decisions


def plan_relevance(
    candidates: Iterable[dict[str, Any]],
    query: str,
    top_k: int,
    retrieval_goal: str,
) -> dict[str, Any]:
    visible_limit = _validated_top_k(top_k)
    if retrieval_goal not in {"focused", "coverage"}:
        raise ValueError(f"unsupported retrieval_goal: {retrieval_goal}")
    terms = normalize_query_terms(query)
    scoring_terms = [term for term in terms if term not in _RANKING_STOPWORDS] or terms
    merged = merge_candidates(candidates)
    _assign_canonical_source_keys(merged)
    decision_limit = candidate_limit(visible_limit, retrieval_goal)
    source_scope, eligible, ineligible_specs, outside_specs = _prepare_evidence(
        merged,
        terms=scoring_terms,
        top_k=visible_limit,
        retrieval_goal=retrieval_goal,
        decision_limit=decision_limit,
    )
    deduplicated, duplicate_specs = _deduplicate_evidence(
        eligible,
        decision_limit=decision_limit,
    )
    selected, counts, coverage_fill_pass_used = _select_evidence(
        deduplicated,
        retrieval_goal=retrieval_goal,
        top_k=visible_limit,
    )

    decisions = _bounded_decision_records(
        selected=selected,
        candidates=deduplicated,
        selected_source_counts=counts,
        retrieval_goal=retrieval_goal,
        top_k=visible_limit,
        decision_limit=decision_limit,
        duplicate_specs=duplicate_specs,
        ineligible_specs=ineligible_specs,
        outside_specs=outside_specs,
    )

    return {
        "selected": selected,
        "source_scope": source_scope,
        "decisions": decisions,
        "normalized_terms": terms,
        "trace": {
            "retrieval_goal": retrieval_goal,
            "merged_candidate_count": len(merged),
            "source_scope_count": len(source_scope),
            "eligible_evidence_count": len(deduplicated),
            "selected_block_count": len(selected),
            "distinct_selected_source_count": len({candidate["source_key"] for candidate in selected}),
            "coverage_fill_pass_used": coverage_fill_pass_used,
        },
    }


def _normalized_terms_input(query_or_terms: str | Iterable[str]) -> list[str]:
    if isinstance(query_or_terms, str):
        return normalize_query_terms(query_or_terms)
    terms: list[str] = []
    seen: set[str] = set()
    for value in query_or_terms:
        term = str(value).lower()
        if term and term not in seen:
            seen.add(term)
            terms.append(term)
            if len(terms) == 32:
                break
    return terms


def _term_position(text: str, term: str) -> int:
    if term.isascii():
        match = re.search(re.escape(term), text, flags=re.IGNORECASE | re.ASCII)
        return -1 if match is None else match.start()
    return text.find(term)


def _term_positions(text: str, term: str, *, limit: int = 4) -> list[int]:
    if term.isascii():
        positions: list[int] = []
        for match in re.finditer(re.escape(term), text, flags=re.IGNORECASE | re.ASCII):
            positions.append(match.start())
            if len(positions) == limit:
                break
        return positions
    positions: list[int] = []
    start = 0
    while len(positions) < limit:
        position = text.find(term, start)
        if position < 0:
            break
        positions.append(position)
        start = position + max(1, len(term))
    return positions


def _numeric_answer_score(text: str, terms: list[str]) -> int:
    if not _NUMERIC_ANSWER_TERMS.intersection(terms):
        return 0
    values: list[str] = []
    for index, match in enumerate(re.finditer(r"\d+(?:\.\d+)?", text)):
        if index == 64:
            break
        values.append(match.group(0))
    if not values:
        return 0
    if {"capacity", "size"}.intersection(terms):
        for value in values:
            if "." in value:
                continue
            integer = int(value)
            if integer >= 8 and integer & (integer - 1) == 0:
                return 2
    return 1


def _math_answer_score(text: str, variables: set[str], raw_query: str) -> int:
    best = 0
    lowered_query = raw_query.lower()
    query_words = set(lowered_query.split())
    for match in re.finditer(r"\$([^$]{1,80})\$", text):
        expression = match.group(1)
        covered = sum(
            bool(re.search(rf"(?<![A-Za-z]){re.escape(variable)}(?![A-Za-z])", expression))
            for variable in variables
        )
        relation_bonus = 0
        if {"below", "less"}.intersection(query_words) and "<" in expression:
            relation_bonus = 2
        elif ("above" in lowered_query or "at least" in lowered_query) and (
            ">" in expression or "\\ge" in expression
        ):
            relation_bonus = 2
        best = max(best, covered + relation_bonus)
    return 1 + best if best else 0


def query_aware_excerpt(
    text: str,
    query_or_terms: str | Iterable[str],
    max_chars: int,
) -> dict[str, Any]:
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    if len(text) <= max_chars:
        return {
            "text": text,
            "metadata": {
                "reason": "full_text",
                "char_start": 0,
                "char_end": len(text),
                "line_start": 0,
                "line_end": len(text.splitlines()),
            },
        }

    raw_query = query_or_terms if isinstance(query_or_terms, str) else ""
    terms = _normalized_terms_input(query_or_terms)
    if isinstance(query_or_terms, str):
        terms = [term for term in terms if term not in _RANKING_STOPWORDS] or terms
    ranking_start = 0
    if text.startswith("[LLM_WIKI_RAW_SECTION]"):
        content_marker = "\n## Section Content\n"
        marker_start = text.find(content_marker)
        if marker_start >= 0:
            ranking_start = marker_start + len(content_marker)
    lines = text.splitlines(keepends=True) or [text]
    starts: list[int] = []
    position = 0
    for line in lines:
        starts.append(position)
        position += len(line)
    lowered_text = text.lower()
    lowered_lines = lowered_text.splitlines(keepends=True) or [lowered_text]
    lowered_starts: list[int] = []
    lowered_position = 0
    for line in lowered_lines:
        lowered_starts.append(lowered_position)
        lowered_position += len(line)
    lowered_ranking_start = len(text[:ranking_start].lower())
    line_term_sets: list[set[str]] = [set() for _ in lines]
    term_line_indices: dict[str, list[int]] = {}
    for term in terms:
        search_start = lowered_ranking_start
        while (match_start := lowered_text.find(term, search_start)) >= 0:
            line_index = max(0, bisect_right(lowered_starts, match_start) - 1)
            if term not in line_term_sets[line_index]:
                line_term_sets[line_index].add(term)
                term_line_indices.setdefault(term, []).append(line_index)
            search_start = lowered_starts[line_index] + len(lowered_lines[line_index])
    line_scores = [len(line_terms) for line_terms in line_term_sets]
    best_line = max(range(len(lines)), key=lambda index: (line_scores[index], -index))
    if line_scores[best_line] == 0:
        prefix_line_end = sum(start < max_chars for start in starts)
        return {
            "text": text[:max_chars],
            "metadata": {
                "reason": "prefix_no_match",
                "char_start": 0,
                "char_end": max_chars,
                "line_start": 0,
                "line_end": prefix_line_end,
            },
        }

    line_start = starts[best_line]
    line_end = line_start + len(lines[best_line])
    if len(lines[best_line]) > max_chars:
        matches = [position for term in terms if (position := _term_position(lines[best_line], term)) >= 0]
        match_start = line_start + min(matches)
        start = max(line_start, match_start - max_chars // 2)
        end = min(line_end, start + max_chars)
        start = max(line_start, end - max_chars)
        return {
            "text": text[start:end],
            "metadata": {
                "reason": "matched_term_center",
                "char_start": start,
                "char_end": end,
                "line_start": best_line,
                "line_end": best_line + 1,
            },
        }

    left = 0
    total = 0
    term_counts: dict[str, int] = {}
    best: tuple[int, int, int, int] | None = None
    for right, line in enumerate(lines):
        total += len(line)
        for term in line_term_sets[right]:
            term_counts[term] = term_counts.get(term, 0) + 1
        while left <= right and total > max_chars:
            for term in line_term_sets[left]:
                count = term_counts[term] - 1
                if count:
                    term_counts[term] = count
                else:
                    del term_counts[term]
            total -= len(lines[left])
            left += 1
        if left <= right:
            candidate = (len(term_counts), -left, total, right)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        return {
            "text": text[:max_chars],
            "metadata": {
                "reason": "prefix",
                "char_start": 0,
                "char_end": max_chars,
                "line_start": 0,
                "line_end": 1,
            },
        }
    _, negative_left, _, right = best
    left = -negative_left
    start = starts[left]
    end = starts[right] + len(lines[right])
    max_start = max(0, len(text) - max_chars)
    candidate_windows = {
        (start, end),
        (0, min(len(text), max_chars)),
        (
            max(0, min(max_start, end - max_chars)),
            min(len(text), max(0, min(max_start, end - max_chars)) + max_chars),
        ),
    }
    term_positions_by_term: dict[str, list[int]] = {}
    for term in terms:
        positions: list[int] = []
        for line_index in term_line_indices.get(term, []):
            positions.extend(
                starts[line_index] + position
                for position in _term_positions(
                    lines[line_index],
                    term,
                    limit=4 - len(positions),
                )
            )
            if len(positions) == 4:
                break
        term_positions_by_term[term] = positions
    for term, match_positions in term_positions_by_term.items():
        for match_position in match_positions:
            for candidate_start in (
                max(0, min(max_start, match_position - max_chars // 2)),
                max(0, min(max_start, match_position + len(term) - max_chars)),
            ):
                candidate_windows.add(
                    (candidate_start, min(len(text), candidate_start + max_chars))
                )
    numeric_answer_intent = bool(_NUMERIC_ANSWER_TERMS.intersection(terms))
    math_query_variables = {
        variable
        for variable in re.findall(r"\b[A-Za-z]\b", raw_query)
        if variable.lower() not in {"a", "i"}
    }
    math_answer_intent = any(variable.isupper() for variable in math_query_variables)
    prefix_anchor_intent = sum(_term_position(lines[0], term) >= 0 for term in terms) >= 2
    numeric_matches: list[tuple[int, int, int]] = []
    if numeric_answer_intent:
        for index, match in enumerate(re.finditer(r"\d+(?:\.\d+)?", text)):
            if index == 16:
                break
            numeric_matches.append(
                (match.start(), match.end(), _numeric_answer_score(match.group(0), terms))
            )
            candidate_start = max(0, min(max_start, match.start() - max_chars // 2))
            candidate_windows.add(
                (candidate_start, min(len(text), candidate_start + max_chars))
            )
    math_matches: list[tuple[int, int, int]] = []
    if math_answer_intent:
        for index, match in enumerate(re.finditer(r"\$[^$]{1,80}\$", text)):
            if index == 16:
                break
            math_matches.append(
                (
                    match.start(),
                    match.end(),
                    _math_answer_score(match.group(0), math_query_variables, raw_query),
                )
            )
            candidate_start = max(0, min(max_start, match.start() - max_chars // 2))
            candidate_windows.add(
                (candidate_start, min(len(text), candidate_start + max_chars))
            )
    line_ends = {starts[index] + len(line) for index, line in enumerate(lines)}
    best_char_window: tuple[int, int, int, int, int, int, int, int] | None = None
    for candidate_start, candidate_end in candidate_windows:
        matched_positions: list[tuple[int, int]] = []
        matched_occurrences = 0
        for term, positions in term_positions_by_term.items():
            inside = [
                position
                for position in positions
                if candidate_start <= position and position + len(term) <= candidate_end
            ]
            if inside:
                matched_positions.append((inside[0] - candidate_start, len(term)))
                matched_occurrences += min(2, len(inside))
        matched_terms = len(matched_positions)
        answer_cue = max(
            (
                score
                for match_start, match_end, score in [*numeric_matches, *math_matches]
                if candidate_start <= match_start and match_end <= candidate_end
            ),
            default=0,
        )
        heading_cue = int(prefix_anchor_intent and candidate_start == 0)
        boundary_count = int(candidate_start in starts) + int(candidate_end in line_ends)
        edge_margin = min(
            (
                min(
                    position,
                    candidate_end - candidate_start - position - term_length,
                )
                for position, term_length in matched_positions
            ),
            default=0,
        )
        candidate_key = (
            answer_cue,
            matched_terms,
            heading_cue,
            matched_occurrences,
            boundary_count,
            edge_margin,
            -candidate_start,
            candidate_end - candidate_start,
        )
        if best_char_window is None or candidate_key > best_char_window:
            best_char_window = candidate_key
            start, end = candidate_start, candidate_end
    line_start_index = max(0, bisect_right(starts, start) - 1)
    line_end_index = max(line_start_index + 1, bisect_right(starts, max(start, end - 1)))
    return {
        "text": text[start:end],
        "metadata": {
            "reason": "matched_term_window",
            "char_start": start,
            "char_end": end,
            "line_start": line_start_index,
            "line_end": line_end_index,
        },
    }
