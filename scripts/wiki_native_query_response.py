#!/usr/bin/env python3
"""Native query-response transforms for raw-section retrieval helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wiki_native_jsonl import jsonl_read
from wiki_native_raw_sections import raw_section_id_from_content, raw_section_kind_from_content


def expand_native_data_response_with_section_neighbors(
    response: dict[str, Any],
    state_dir: Path,
    neighbor_k: int = 5,
    section_kind: str | None = None,
) -> dict[str, Any]:
    cloned = json.loads(json.dumps(response, ensure_ascii=False))
    data = cloned.get("data")
    if not isinstance(data, dict):
        return cloned
    chunks = data.get("chunks")
    if not isinstance(chunks, list):
        return cloned
    seed_ids: list[str] = []
    seed_kind_by_id: dict[str, str] = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        content = str(chunk.get("content", ""))
        section_id = raw_section_id_from_content(content)
        if not section_id:
            continue
        kind = raw_section_kind_from_content(content)
        if section_kind and kind and kind != section_kind.strip().lower():
            continue
        if section_id not in seed_ids:
            seed_ids.append(section_id)
            seed_kind_by_id[section_id] = kind
    seed_set = set(seed_ids)
    grouped: dict[str, list[dict[str, Any]]] = {seed_id: [] for seed_id in seed_ids}
    for edge in jsonl_read(state_dir / "section_similarity_edges.jsonl"):
        src = str(edge.get("src_id", ""))
        tgt = str(edge.get("tgt_id", ""))
        orientations = []
        if src in seed_set:
            orientations.append((src, tgt, edge.get("source_section_kind"), edge.get("target_section_kind"), edge.get("target_path"), edge.get("target_title")))
        if tgt in seed_set:
            orientations.append((tgt, src, edge.get("target_section_kind"), edge.get("source_section_kind"), edge.get("source_path"), edge.get("source_title")))
        for seed_id, neighbor_id, seed_kind, neighbor_kind, neighbor_path, neighbor_title in orientations:
            if section_kind and str(seed_kind or seed_kind_by_id.get(seed_id, "")).lower() != section_kind.strip().lower():
                continue
            grouped.setdefault(seed_id, []).append(
                {
                    "seed_section_id": seed_id,
                    "neighbor_section_id": neighbor_id,
                    "seed_section_kind": seed_kind or seed_kind_by_id.get(seed_id),
                    "neighbor_section_kind": neighbor_kind,
                    "neighbor_source_path": neighbor_path,
                    "neighbor_title": neighbor_title,
                    "cosine": edge.get("cosine"),
                    "pair_kind": edge.get("pair_kind"),
                    "edge_id": edge.get("edge_id"),
                    "expansion_source": "section_similarity_edges.jsonl",
                }
            )
    expansions: list[dict[str, Any]] = []
    for seed_id in seed_ids:
        rows = sorted(grouped.get(seed_id, []), key=lambda row: (-float(row.get("cosine") or 0), str(row.get("neighbor_section_id", ""))))
        expansions.extend(rows[: max(0, neighbor_k)])
    data["section_neighbor_expansions"] = expansions
    return cloned


def filter_native_data_response_by_section_kind(response: dict[str, Any], section_kind: str) -> dict[str, Any]:
    kind = section_kind.strip().lower()
    cloned = json.loads(json.dumps(response, ensure_ascii=False))
    data = cloned.get("data")
    if not isinstance(data, dict):
        return cloned
    chunks = data.get("chunks")
    if not isinstance(chunks, list):
        return cloned
    filtered = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        file_path = str(chunk.get("file_path", ""))
        content = str(chunk.get("content", ""))
        if file_path.startswith("raw_section_docs/") and raw_section_kind_from_content(content) == kind:
            filtered.append(chunk)
    data["chunks"] = filtered
    data["section_kind_filter"] = kind
    data["section_kind_filter_kept"] = len(filtered)
    data["section_kind_filter_original_chunks"] = len(chunks)
    return cloned
