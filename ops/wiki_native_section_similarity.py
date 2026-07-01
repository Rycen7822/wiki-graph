#!/usr/bin/env python3
"""Native-owned raw-section similarity helpers for llm-wiki artifacts."""

from __future__ import annotations

import datetime as dt
import re
import sqlite3
from pathlib import Path
from typing import Any

from llm_wiki_native.source_docs import sha256_text


def _now_stamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def section_similarity_embedding_text(section: dict[str, Any], max_content_chars: int = 6000) -> str:
    """Build clean text for section-to-section embedding without sidecar boilerplate."""
    content = re.sub(r"\s+", " ", str(section.get("content", "")).strip())
    if max_content_chars > 0 and len(content) > max_content_chars:
        content = content[:max_content_chars].rstrip()
    lines = [
        f"Title: {section.get('paper_title', '')}",
        f"Section kind: {section.get('section_kind', '')}",
        f"Section title: {section.get('section_title', '')}",
        f"Source path: {section.get('source_path', '')}",
        "",
        content,
    ]
    return "\n".join(line for line in lines if line is not None).strip() + "\n"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _ordered_section_pair_key(src_id: str, tgt_id: str) -> str:
    digest = sha256_text("\t".join(sorted([src_id, tgt_id])))[:12]
    return f"semantic_section_neighbor:{digest}"


def _section_rank_lists_scalar(
    sections: list[dict[str, Any]],
    embeddings: dict[str, list[float]],
    src_kind: str,
    tgt_kind: str,
    k: int,
    min_cosine: float,
) -> dict[tuple[str, str], dict[str, Any]]:
    sources = [section for section in sections if section.get("section_kind") == src_kind and section.get("section_id") in embeddings]
    targets = [section for section in sections if section.get("section_kind") == tgt_kind and section.get("section_id") in embeddings]
    directed: dict[tuple[str, str], dict[str, Any]] = {}
    for src in sources:
        src_id = str(src["section_id"])
        scored: list[tuple[float, dict[str, Any]]] = []
        for tgt in targets:
            tgt_id = str(tgt["section_id"])
            if src_id == tgt_id:
                continue
            if src.get("source_id") and src.get("source_id") == tgt.get("source_id"):
                continue
            score = cosine_similarity(embeddings[src_id], embeddings[tgt_id])
            if score >= min_cosine:
                scored.append((score, tgt))
        scored.sort(key=lambda item: (-item[0], str(item[1].get("section_id", ""))))
        for rank, (score, tgt) in enumerate(scored[: max(k, 0)], start=1):
            directed[(src_id, str(tgt["section_id"]))] = {"cosine": score, "rank": rank, "src": src, "tgt": tgt}
    return directed


def _section_rank_lists_vectorized(
    sections: list[dict[str, Any]],
    embeddings: dict[str, list[float]],
    src_kind: str,
    tgt_kind: str,
    k: int,
    min_cosine: float,
) -> dict[tuple[str, str], dict[str, Any]] | None:
    if k <= 0:
        return {}
    sources = [section for section in sections if section.get("section_kind") == src_kind and section.get("section_id") in embeddings]
    targets = [section for section in sections if section.get("section_kind") == tgt_kind and section.get("section_id") in embeddings]
    if not sources or not targets:
        return {}
    try:
        import numpy as np
    except Exception:
        return None

    try:

        def matrix_for(items: list[dict[str, Any]]):
            vectors: list[list[float]] = []
            dim: int | None = None
            for section in items:
                vector = embeddings.get(str(section["section_id"]))
                if not isinstance(vector, list) or not vector:
                    return None
                if dim is None:
                    dim = len(vector)
                elif len(vector) != dim:
                    return None
                vectors.append(vector)
            matrix = np.asarray(vectors, dtype=np.float64)
            if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
                return None
            return matrix

        source_matrix = matrix_for(sources)
        target_matrix = matrix_for(targets)
        if source_matrix is None or target_matrix is None or source_matrix.shape[1] != target_matrix.shape[1]:
            return None

        source_norms = np.linalg.norm(source_matrix, axis=1)
        target_norms = np.linalg.norm(target_matrix, axis=1)
        source_normalized = np.divide(
            source_matrix,
            source_norms[:, None],
            out=np.zeros_like(source_matrix, dtype=np.float64),
            where=source_norms[:, None] != 0,
        )
        target_normalized = np.divide(
            target_matrix,
            target_norms[:, None],
            out=np.zeros_like(target_matrix, dtype=np.float64),
            where=target_norms[:, None] != 0,
        )

        source_ids = [str(section["section_id"]) for section in sources]
        target_ids = [str(section["section_id"]) for section in targets]
        target_id_array = np.asarray(target_ids, dtype=object)
        target_source_ids = np.asarray([str(section.get("source_id") or "") for section in targets], dtype=object)
        score_eps = 1e-9
        block_size = 256
        threshold_floor = min_cosine - score_eps
        directed: dict[tuple[str, str], dict[str, Any]] = {}

        for block_start in range(0, len(sources), block_size):
            score_block = source_normalized[block_start : block_start + block_size] @ target_normalized.T
            for local_index, row in enumerate(score_block):
                source_index = block_start + local_index
                src = sources[source_index]
                src_id = source_ids[source_index]
                valid_mask = target_id_array != src_id
                src_source_id = str(src.get("source_id") or "")
                if src_source_id:
                    valid_mask = valid_mask & (target_source_ids != src_source_id)
                candidate_mask = valid_mask & (row >= threshold_floor)
                if not bool(candidate_mask.any()):
                    continue

                candidate_indices = np.flatnonzero(candidate_mask)
                if len(candidate_indices) > k:
                    candidate_scores = row[candidate_indices]
                    kth_index = len(candidate_scores) - k
                    kth_score = float(np.partition(candidate_scores, kth_index)[kth_index])
                    candidate_indices = candidate_indices[candidate_scores >= kth_score - score_eps]

                scored: list[tuple[float, dict[str, Any]]] = []
                for target_index in candidate_indices.tolist():
                    tgt = targets[int(target_index)]
                    tgt_id = target_ids[int(target_index)]
                    score = cosine_similarity(embeddings[src_id], embeddings[tgt_id])
                    if score >= min_cosine:
                        scored.append((score, tgt))
                scored.sort(key=lambda item: (-item[0], str(item[1].get("section_id", ""))))
                for rank, (score, tgt) in enumerate(scored[: max(k, 0)], start=1):
                    directed[(src_id, str(tgt["section_id"]))] = {"cosine": score, "rank": rank, "src": src, "tgt": tgt}
        return directed
    except Exception:
        return None


def _section_rank_lists(
    sections: list[dict[str, Any]],
    embeddings: dict[str, list[float]],
    src_kind: str,
    tgt_kind: str,
    k: int,
    min_cosine: float,
) -> dict[tuple[str, str], dict[str, Any]]:
    vectorized = _section_rank_lists_vectorized(sections, embeddings, src_kind, tgt_kind, k, min_cosine)
    if vectorized is not None:
        return vectorized
    return _section_rank_lists_scalar(sections, embeddings, src_kind, tgt_kind, k, min_cosine)


def _section_similarity_directed_rank_lists_for_parameters(
    sections_by_id: dict[str, dict[str, Any]],
    embeddings: dict[str, list[float]],
    same_kind_k: int,
    cross_kind_k: int,
    same_kind_min_cosine: float,
    cross_kind_min_cosine: float,
    cross_kind_pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], dict[str, Any]]:
    section_values = list(sections_by_id.values())
    section_kinds = sorted({str(section.get("section_kind", "")) for section in section_values if section.get("section_kind")})
    directed: dict[tuple[str, str], dict[str, Any]] = {}
    for kind in section_kinds:
        directed.update(_section_rank_lists(section_values, embeddings, kind, kind, same_kind_k, same_kind_min_cosine))
    for left, right in cross_kind_pairs:
        directed.update(_section_rank_lists(section_values, embeddings, left, right, cross_kind_k, cross_kind_min_cosine))
        directed.update(_section_rank_lists(section_values, embeddings, right, left, cross_kind_k, cross_kind_min_cosine))
    return directed


def _section_similarity_index_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        create table if not exists section_similarity_rank(
          family_id text not null,
          src_id text not null,
          tgt_id text not null,
          src_kind text not null,
          tgt_kind text not null,
          cosine real not null,
          rank integer not null,
          src_text_hash text not null,
          tgt_text_hash text not null,
          embedding_model text not null,
          embedding_dim integer,
          updated_at text not null,
          primary key(family_id, src_id, tgt_id)
        )
        """
    )
    conn.execute("create index if not exists idx_section_similarity_rank_src on section_similarity_rank(src_kind, tgt_kind, src_id, rank)")


def write_section_similarity_index(
    index_path: Path,
    sections_by_id: dict[str, dict[str, Any]],
    directed: dict[tuple[str, str], dict[str, Any]],
    *,
    embedding_model: str = "unknown",
    embedding_dim: int | None = None,
) -> dict[str, Any]:
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[Any, ...]] = []
    updated_at = _now_stamp()
    for (src_id, tgt_id), item in sorted(directed.items(), key=lambda pair: (pair[0][0], int(pair[1].get("rank", 0)), pair[0][1])):
        src = sections_by_id[src_id]
        tgt = sections_by_id[tgt_id]
        src_kind = str(src.get("section_kind") or "")
        tgt_kind = str(tgt.get("section_kind") or "")
        rows.append(
            (
                f"{src_kind}:{tgt_kind}",
                src_id,
                tgt_id,
                src_kind,
                tgt_kind,
                float(item["cosine"]),
                int(item["rank"]),
                sha256_text(section_similarity_embedding_text(src)),
                sha256_text(section_similarity_embedding_text(tgt)),
                embedding_model,
                int(embedding_dim) if embedding_dim is not None else None,
                updated_at,
            )
        )
    with sqlite3.connect(index_path) as conn:
        _section_similarity_index_schema(conn)
        conn.execute("delete from section_similarity_rank")
        conn.executemany(
            """
            insert into section_similarity_rank
            (family_id, src_id, tgt_id, src_kind, tgt_kind, cosine, rank, src_text_hash, tgt_text_hash, embedding_model, embedding_dim, updated_at)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return section_similarity_index_summary(index_path)


def section_similarity_index_summary(index_path: Path) -> dict[str, Any]:
    index_path = Path(index_path)
    if not index_path.exists():
        return {"index_path": index_path.as_posix(), "directed_rows": 0, "family_count": 0, "families": {}}
    with sqlite3.connect(index_path) as conn:
        _section_similarity_index_schema(conn)
        directed_rows = int(conn.execute("select count(*) from section_similarity_rank").fetchone()[0])
        family_rows = conn.execute("select family_id, count(*) from section_similarity_rank group by family_id order by family_id").fetchall()
    return {
        "index_path": index_path.as_posix(),
        "directed_rows": directed_rows,
        "family_count": len(family_rows),
        "families": {str(family): int(count) for family, count in family_rows},
    }


def _read_section_similarity_index_directed(
    index_path: Path,
    sections_by_id: dict[str, dict[str, Any]],
    *,
    embedding_model: str | None = None,
    embedding_dim: int | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    directed: dict[tuple[str, str], dict[str, Any]] = {}
    with sqlite3.connect(index_path) as conn:
        _section_similarity_index_schema(conn)
        rows = conn.execute(
            """
            select src_id, tgt_id, cosine, rank, src_text_hash, tgt_text_hash, embedding_model, embedding_dim
            from section_similarity_rank
            order by family_id, src_id, rank, tgt_id
            """
        ).fetchall()
    for src_id, tgt_id, cosine, rank, src_text_hash, tgt_text_hash, row_model, row_dim in rows:
        src_id = str(src_id)
        tgt_id = str(tgt_id)
        if src_id not in sections_by_id or tgt_id not in sections_by_id:
            continue
        src = sections_by_id[src_id]
        tgt = sections_by_id[tgt_id]
        if src_text_hash != sha256_text(section_similarity_embedding_text(src)):
            raise ValueError(f"stale section similarity index row for {src_id}: src_text_hash mismatch")
        if tgt_text_hash != sha256_text(section_similarity_embedding_text(tgt)):
            raise ValueError(f"stale section similarity index row for {tgt_id}: tgt_text_hash mismatch")
        if embedding_model is not None and str(row_model) != embedding_model:
            raise ValueError(f"stale section similarity index row for {src_id}->{tgt_id}: embedding_model mismatch")
        if embedding_dim is not None and row_dim is not None and int(row_dim) != int(embedding_dim):
            raise ValueError(f"stale section similarity index row for {src_id}->{tgt_id}: embedding_dim mismatch")
        directed[(src_id, tgt_id)] = {"cosine": float(cosine), "rank": int(rank), "src": src, "tgt": tgt}
    return directed


def _build_section_similarity_edges_from_directed(
    sections_by_id: dict[str, dict[str, Any]],
    embeddings: dict[str, list[float]],
    directed: dict[tuple[str, str], dict[str, Any]],
    *,
    cross_kind_pairs: list[tuple[str, str]],
    mutual: bool,
    embedding_model: str,
    embedding_dim: int | None,
) -> list[dict[str, Any]]:
    section_kinds = sorted({str(section.get("section_kind", "")) for section in sections_by_id.values() if section.get("section_kind")})
    edges: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    def add_edge(src_id: str, tgt_id: str, orientation: tuple[str, str] | None = None) -> None:
        if (src_id, tgt_id) not in directed:
            return
        if mutual and (tgt_id, src_id) not in directed:
            return
        src = sections_by_id[src_id]
        tgt = sections_by_id[tgt_id]
        unordered = tuple(sorted([src_id, tgt_id]))
        if unordered in seen_pairs:
            return
        seen_pairs.add(unordered)
        forward = directed[(src_id, tgt_id)]
        reverse = directed.get((tgt_id, src_id), {})
        cosine = float(forward["cosine"])
        edge = {
            "edge_id": _ordered_section_pair_key(src_id, tgt_id),
            "type": "SEMANTIC_SECTION_NEIGHBOR",
            "src_id": src_id,
            "tgt_id": tgt_id,
            "source_section_kind": src.get("section_kind"),
            "target_section_kind": tgt.get("section_kind"),
            "source_path": src.get("source_path"),
            "target_path": tgt.get("source_path"),
            "source_title": src.get("paper_title"),
            "target_title": tgt.get("paper_title"),
            "cosine": round(cosine, 6),
            "source_rank": forward.get("rank"),
            "target_rank": reverse.get("rank"),
            "mutual_knn": bool((tgt_id, src_id) in directed),
            "embedding_model": embedding_model,
            "embedding_dim": embedding_dim or (len(embeddings[src_id]) if src_id in embeddings else None),
            "source_text_hash": sha256_text(section_similarity_embedding_text(src)),
            "target_text_hash": sha256_text(section_similarity_embedding_text(tgt)),
            "created_by": "build_section_similarity_graph.py",
        }
        if orientation:
            edge["pair_kind"] = f"{orientation[0]}:{orientation[1]}"
        edges.append(edge)

    for kind in section_kinds:
        same_ids = sorted(str(section["section_id"]) for section in sections_by_id.values() if section.get("section_kind") == kind)
        for i, src_id in enumerate(same_ids):
            for tgt_id in same_ids[i + 1 :]:
                if (src_id, tgt_id) in directed:
                    add_edge(src_id, tgt_id, (kind, kind))
                elif (tgt_id, src_id) in directed:
                    add_edge(tgt_id, src_id, (kind, kind))
    for left, right in cross_kind_pairs:
        left_ids = sorted(str(section["section_id"]) for section in sections_by_id.values() if section.get("section_kind") == left)
        right_ids = sorted(str(section["section_id"]) for section in sections_by_id.values() if section.get("section_kind") == right)
        for src_id in left_ids:
            for tgt_id in right_ids:
                add_edge(src_id, tgt_id, (left, right))
    edges.sort(key=lambda edge: (str(edge.get("pair_kind", "")), -float(edge.get("cosine", 0)), str(edge.get("src_id")), str(edge.get("tgt_id"))))
    return edges


def build_section_similarity_edges_from_index(
    index_path: Path,
    sections: list[dict[str, Any]],
    embeddings: dict[str, list[float]],
    *,
    cross_kind_pairs: list[tuple[str, str]] | None = None,
    mutual: bool = True,
    embedding_model: str = "unknown",
    embedding_dim: int | None = None,
) -> list[dict[str, Any]]:
    sections_by_id = {str(section.get("section_id")): section for section in sections if section.get("section_id") in embeddings}
    directed = _read_section_similarity_index_directed(index_path, sections_by_id, embedding_model=embedding_model, embedding_dim=embedding_dim)
    return _build_section_similarity_edges_from_directed(
        sections_by_id,
        embeddings,
        directed,
        cross_kind_pairs=cross_kind_pairs or [],
        mutual=mutual,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
    )


def build_section_similarity_edges(
    sections: list[dict[str, Any]],
    embeddings: dict[str, list[float]],
    same_kind_k: int = 5,
    cross_kind_k: int = 3,
    same_kind_min_cosine: float = 0.72,
    cross_kind_min_cosine: float = 0.76,
    cross_kind_pairs: list[tuple[str, str]] | None = None,
    mutual: bool = True,
    embedding_model: str = "unknown",
    embedding_dim: int | None = None,
    index_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Build sparse semantic-neighbor candidate edges between raw-section embeddings."""
    sections_by_id = {str(section.get("section_id")): section for section in sections if section.get("section_id") in embeddings}
    cross_kind_pairs = cross_kind_pairs or []
    directed = _section_similarity_directed_rank_lists_for_parameters(
        sections_by_id,
        embeddings,
        same_kind_k,
        cross_kind_k,
        same_kind_min_cosine,
        cross_kind_min_cosine,
        cross_kind_pairs,
    )
    if index_path is not None:
        write_section_similarity_index(index_path, sections_by_id, directed, embedding_model=embedding_model, embedding_dim=embedding_dim)
    return _build_section_similarity_edges_from_directed(
        sections_by_id,
        embeddings,
        directed,
        cross_kind_pairs=cross_kind_pairs,
        mutual=mutual,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
    )


def section_similarity_edge_to_custom_kg_relationship(edge: dict[str, Any]) -> dict[str, Any]:
    cosine = float(edge.get("cosine", 0.0))
    src = str(edge.get("src_id", ""))
    tgt = str(edge.get("tgt_id", ""))
    source_kind = edge.get("source_section_kind", "section")
    target_kind = edge.get("target_section_kind", "section")
    description = (
        f"{src} SEMANTIC_SECTION_NEIGHBOR {tgt}; "
        f"source_section_kind={source_kind}; target_section_kind={target_kind}; "
        f"cosine={cosine:.6f}; mutual_knn={edge.get('mutual_knn')}; "
        f"embedding_model={edge.get('embedding_model', 'unknown')}."
    )
    return {
        "src_id": src,
        "tgt_id": tgt,
        "description": description,
        "keywords": "SEMANTIC_SECTION_NEIGHBOR",
        "source_id": src,
        "weight": cosine,
        "file_path": str(edge.get("source_path") or edge.get("target_path") or "section_similarity_edges.jsonl"),
    }


def _small_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    values_sorted = sorted(values)
    return {
        "min": round(values_sorted[0], 6),
        "mean": round(sum(values_sorted) / len(values_sorted), 6),
        "max": round(values_sorted[-1], 6),
    }


def section_similarity_report_summary(sections: list[dict[str, Any]], edges: list[dict[str, Any]], top_hubs: int = 20) -> dict[str, Any]:
    section_count_by_kind: dict[str, int] = {}
    for section in sections:
        kind = str(section.get("section_kind", "unknown"))
        section_count_by_kind[kind] = section_count_by_kind.get(kind, 0) + 1
    edge_count_by_pair_kind: dict[str, int] = {}
    cosine_values_by_pair: dict[str, list[float]] = {}
    hub_degree: dict[str, int] = {}
    for edge in edges:
        pair_kind = str(edge.get("pair_kind") or f"{edge.get('source_section_kind', 'section')}:{edge.get('target_section_kind', 'section')}")
        edge_count_by_pair_kind[pair_kind] = edge_count_by_pair_kind.get(pair_kind, 0) + 1
        cosine_values_by_pair.setdefault(pair_kind, []).append(float(edge.get("cosine", 0.0)))
        for node in [str(edge.get("src_id", "")), str(edge.get("tgt_id", ""))]:
            if node:
                hub_degree[node] = hub_degree.get(node, 0) + 1
    hubs = [
        {"section_id": section_id, "degree": degree}
        for section_id, degree in sorted(hub_degree.items(), key=lambda item: (-item[1], item[0]))[:top_hubs]
    ]
    return {
        "section_count": len(sections),
        "section_count_by_kind": dict(sorted(section_count_by_kind.items())),
        "edge_count": len(edges),
        "edge_count_by_pair_kind": dict(sorted(edge_count_by_pair_kind.items())),
        "cosine_by_pair_kind": {pair: _small_stats(values) for pair, values in sorted(cosine_values_by_pair.items())},
        "top_hubs": hubs,
    }


def select_section_similarity_edges(candidates: list[dict[str, Any]], allowed_pair_kinds: set[str]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for edge in candidates:
        if str(edge.get("pair_kind", "")) not in allowed_pair_kinds:
            continue
        row = dict(edge)
        row["review_status"] = "phase2_selected"
        row["review_note"] = "Sparse mutual-kNN high-value section pair selected for custom_kg import. Semantic proximity only, not a factual relation."
        selected.append(row)
    selected.sort(key=lambda edge: (str(edge.get("pair_kind", "")), -float(edge.get("cosine", 0.0)), str(edge.get("src_id", "")), str(edge.get("tgt_id", ""))))
    return selected
