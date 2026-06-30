#!/usr/bin/env python3
"""Build sidecar raw-section embedding similarity candidates for llm-wiki.

Phase 1 writes only external llm-wiki state artifacts. It does not import semantic
neighbor edges into custom_kg; use a separate reviewed Phase 2 for that.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from wiki_graph.ops.native_runtime_env import load_env_file, redact_summary
from wiki_graph.ops.wiki_native_lib import (
    RAW_NOTE_CONTRACT_SECTION_KINDS,
    build_section_similarity_edges,
    common_paths_parser,
    ensure_state_dirs,
    jsonl_read,
    jsonl_write,
    now_stamp,
    print_json,
    release_process_memory,
    section_similarity_embedding_text,
    section_similarity_index_summary,
    section_similarity_report_summary,
    sha256_text,
)

DEFAULT_CROSS_KIND_PAIRS = [
    ("limitations", "questions"),
    ("future", "limitations"),
    ("future", "questions"),
    ("methodology", "future"),
    ("methodology", "results"),
    ("results", "future"),
    ("results", "limitations"),
]


def parse_section_kinds(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_cross_kind_pairs(raw: str) -> list[tuple[str, str]]:
    if not raw.strip():
        return []
    pairs: list[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"cross-kind pair must be KIND:KIND, got {item!r}")
        left, right = [part.strip() for part in item.split(":", 1)]
        if not left or not right:
            raise ValueError(f"cross-kind pair must be KIND:KIND, got {item!r}")
        pairs.append((left, right))
    return pairs


def embedding_config(workdir: Path) -> dict[str, Any]:
    env_values = load_env_file(workdir / ".env")
    model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    host = os.environ.get("EMBEDDING_BINDING_HOST") or os.environ.get("OPENAI_BASE_URL")
    key = os.environ.get("EMBEDDING_BINDING_API_KEY") or os.environ.get("OPENAI_API_KEY")
    dim_raw = os.environ.get("EMBEDDING_DIM", "")
    dim = int(dim_raw) if dim_raw.isdigit() else None
    timeout = int(os.environ.get("EMBEDDING_TIMEOUT", "120") or "120")
    batch = int(os.environ.get("EMBEDDING_BATCH_NUM", "10") or "10")
    return {
        "model": model,
        "host": host,
        "api_key": key,
        "embedding_dim": dim,
        "timeout": timeout,
        "batch_size": batch,
        "env": redact_summary({
            "EMBEDDING_BINDING": env_values.get("EMBEDDING_BINDING", ""),
            "EMBEDDING_BINDING_HOST": env_values.get("EMBEDDING_BINDING_HOST", ""),
            "EMBEDDING_MODEL": env_values.get("EMBEDDING_MODEL", ""),
            "EMBEDDING_DIM": env_values.get("EMBEDDING_DIM", ""),
            "EMBEDDING_BINDING_API_KEY": env_values.get("EMBEDDING_BINDING_API_KEY", ""),
        }),
    }


def openai_compatible_embed(texts: list[str], config: dict[str, Any], max_attempts: int = 3) -> list[list[float]]:
    host = (config.get("host") or "").rstrip("/")
    api_key = config.get("api_key") or ""
    model = config.get("model") or "text-embedding-3-small"
    if not host:
        raise RuntimeError("EMBEDDING_BINDING_HOST or OPENAI_BASE_URL is required for section embeddings")
    if not api_key:
        raise RuntimeError("EMBEDDING_BINDING_API_KEY or OPENAI_API_KEY is required for section embeddings")
    url = f"{host}/embeddings"
    body = json.dumps({"model": model, "input": texts}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=int(config.get("timeout") or 120)) as response:
                payload = json.loads(response.read().decode("utf-8"))
            data = payload.get("data") or []
            data = sorted(data, key=lambda row: row.get("index", 0))
            embeddings = [row.get("embedding") for row in data]
            if len(embeddings) != len(texts) or not all(isinstance(vec, list) for vec in embeddings):
                raise RuntimeError(f"embedding response count mismatch: expected {len(texts)}, got {len(embeddings)}")
            return embeddings  # type: ignore[return-value]
        except Exception as exc:  # pragma: no cover - network retry path
            last_exc = exc
            if attempt >= max_attempts:
                break
            time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"embedding request failed after {max_attempts} attempts: {last_exc!r}")


def load_cached_embeddings(path: Path, model: str) -> dict[str, dict[str, Any]]:
    cached: dict[str, dict[str, Any]] = {}
    for row in jsonl_read(path):
        if row.get("embedding_model") == model and row.get("section_id") and row.get("text_hash"):
            cached[str(row["section_id"])] = row
    return cached


def build_embedding_rows(sections: list[dict[str, Any]], config: dict[str, Any], cache_path: Path, reuse_cache: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = str(config.get("model") or "unknown")
    dim = config.get("embedding_dim")
    cached = load_cached_embeddings(cache_path, model) if reuse_cache else {}
    rows: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], str, str]] = []
    cache_hits = 0
    for section in sections:
        text = section_similarity_embedding_text(section)
        text_hash = sha256_text(text)
        section_id = str(section["section_id"])
        cached_row = cached.get(section_id)
        if cached_row and cached_row.get("text_hash") == text_hash and isinstance(cached_row.get("embedding"), list):
            rows.append(cached_row)
            cache_hits += 1
        else:
            pending.append((section, text, text_hash))
    batch_size = max(1, int(config.get("batch_size") or 10))
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        texts = [item[1] for item in batch]
        vectors = openai_compatible_embed(texts, config)
        for (section, text, text_hash), vector in zip(batch, vectors):
            rows.append(
                {
                    "section_id": section["section_id"],
                    "source_id": section.get("source_id"),
                    "source_path": section.get("source_path"),
                    "paper_title": section.get("paper_title"),
                    "section_kind": section.get("section_kind"),
                    "section_title": section.get("section_title"),
                    "text_hash": text_hash,
                    "text_chars": len(text),
                    "embedding_model": model,
                    "embedding_dim": dim or len(vector),
                    "embedding": vector,
                }
            )
    rows.sort(key=lambda row: str(row.get("section_id", "")))
    return rows, {"cache_hits": cache_hits, "embedded": len(pending), "total": len(rows), "cache_path": cache_path.as_posix()}


def main() -> int:
    parser = common_paths_parser("Build sidecar raw-section semantic-neighbor candidates")
    parser.add_argument("--section-kinds", default=",".join(RAW_NOTE_CONTRACT_SECTION_KINDS))
    parser.add_argument("--cross-kind-pairs", default=",".join(f"{a}:{b}" for a, b in DEFAULT_CROSS_KIND_PAIRS))
    parser.add_argument("--same-kind-k", type=int, default=5)
    parser.add_argument("--cross-kind-k", type=int, default=3)
    parser.add_argument("--same-kind-min-cosine", type=float, default=0.72)
    parser.add_argument("--cross-kind-min-cosine", type=float, default=0.76)
    parser.add_argument("--no-mutual", action="store_true", help="Keep directed top-k edges without mutual-kNN filtering")
    parser.add_argument("--min-content-chars", type=int, default=20)
    parser.add_argument("--limit-sections", type=int, default=None)
    parser.add_argument("--no-reuse-cache", action="store_true")
    parser.add_argument("--section-similarity-index", type=Path, default=None, help="SQLite directed-rank index path; defaults to state/section_similarity_index.sqlite")
    parser.add_argument("--no-section-similarity-index", action="store_true", help="Do not write the persisted exact section-similarity rank index")
    parser.add_argument("--sample-edges", type=int, default=50)
    args = parser.parse_args()

    ensure_state_dirs(args.state_dir)
    section_kinds = set(parse_section_kinds(args.section_kinds))
    cross_kind_pairs = parse_cross_kind_pairs(args.cross_kind_pairs)
    raw_sections_path = args.state_dir / "raw_sections.jsonl"
    rows = [row for row in jsonl_read(raw_sections_path) if row.get("section_kind") in section_kinds and len(str(row.get("content", "")).strip()) >= args.min_content_chars]
    if args.limit_sections is not None:
        rows = rows[: args.limit_sections]
    if not rows:
        raise RuntimeError(f"No raw sections found in {raw_sections_path}; run extract_raw_sections.py first")
    index_path = None if args.no_section_similarity_index else (args.section_similarity_index or args.state_dir / "section_similarity_index.sqlite")

    config = embedding_config(args.workdir)
    embedding_path = args.state_dir / "section_embeddings.jsonl"
    try:
        embedding_rows, embedding_stats = build_embedding_rows(rows, config, embedding_path, reuse_cache=not args.no_reuse_cache)
    except Exception as exc:
        print_json(
            {
                "ok": False,
                "error": "section_embedding_failed",
                "message": str(exc),
                "exception_type": type(exc).__name__,
                "root": str(args.root.resolve()),
                "state_dir": str(args.state_dir.resolve()),
                "workdir": str(args.workdir.resolve()),
                "raw_sections_path": raw_sections_path.as_posix(),
                "embedding_path": embedding_path.as_posix(),
                "section_count": len(rows),
                "embedding_env": config["env"],
            }
        )
        return 1
    jsonl_write(embedding_path, embedding_rows)
    embeddings = {str(row["section_id"]): row["embedding"] for row in embedding_rows}
    del embedding_rows
    release_process_memory()
    edges = build_section_similarity_edges(
        rows,
        embeddings,
        same_kind_k=args.same_kind_k,
        cross_kind_k=args.cross_kind_k,
        same_kind_min_cosine=args.same_kind_min_cosine,
        cross_kind_min_cosine=args.cross_kind_min_cosine,
        cross_kind_pairs=cross_kind_pairs,
        mutual=not args.no_mutual,
        embedding_model=str(config.get("model") or "unknown"),
        embedding_dim=config.get("embedding_dim"),
        index_path=index_path,
    )
    index_summary = section_similarity_index_summary(index_path) if index_path is not None else None
    candidates_path = args.state_dir / "section_similarity_edges.candidates.jsonl"
    jsonl_write(candidates_path, edges)
    report = {
        "generated_at": now_stamp(),
        "phase": "sidecar-dry-run",
        "imported_to_custom_kg": False,
        "raw_sections_path": raw_sections_path.as_posix(),
        "embedding_path": embedding_path.as_posix(),
        "candidate_edges_path": candidates_path.as_posix(),
        "parameters": {
            "section_kinds": sorted(section_kinds),
            "cross_kind_pairs": cross_kind_pairs,
            "same_kind_k": args.same_kind_k,
            "cross_kind_k": args.cross_kind_k,
            "same_kind_min_cosine": args.same_kind_min_cosine,
            "cross_kind_min_cosine": args.cross_kind_min_cosine,
            "mutual": not args.no_mutual,
            "min_content_chars": args.min_content_chars,
            "limit_sections": args.limit_sections,
        },
        "embedding_env": config["env"],
        "embedding_stats": embedding_stats,
        "section_similarity_index": index_summary,
        "summary": section_similarity_report_summary(rows, edges),
        "sample_edges": edges[: args.sample_edges],
    }
    report_dir = args.state_dir / "section_similarity_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_section_similarity_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = report_path.as_posix()
    print_json({
        "section_count": report["summary"]["section_count"],
        "edge_count": report["summary"]["edge_count"],
        "embedding_stats": embedding_stats,
        "embedding_path": embedding_path.as_posix(),
        "candidate_edges_path": candidates_path.as_posix(),
        "section_similarity_index": index_summary,
        "report_path": report_path.as_posix(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
