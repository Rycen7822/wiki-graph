#!/usr/bin/env python3
"""Native query event and evidence-pack helpers."""

from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from ops.wiki_native_state import ensure_state_dirs

NATIVE_QUERY_EVENTS_DB_FILENAME = "native_query_events.db"


def _now_stamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ensure_state_dirs(state_dir: Path) -> None:
    ensure_state_dirs(state_dir)


def slugify(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", text.strip()).strip("-").lower()
    return (slug or "item")[:max_len].strip("-") or "item"


def init_query_events_db(state_dir: Path) -> Path:
    _ensure_state_dirs(state_dir)
    db = state_dir / NATIVE_QUERY_EVENTS_DB_FILENAME
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS query_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              query TEXT NOT NULL,
              mode TEXT NOT NULL,
              rewritten_queries TEXT,
              evidence_pack_path TEXT,
              created_at TEXT NOT NULL
            )
            """
        )
    return db


def save_evidence_pack(
    state_dir: Path,
    query: str,
    mode: str,
    response: dict[str, Any],
    *,
    request_metadata: dict[str, Any] | None = None,
) -> Path:
    _ensure_state_dirs(state_dir)
    slug = slugify(query, 70)
    path = state_dir / "evidence_packs" / f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{slug}.md"
    refs = response.get("references") or []
    safe_metadata = _bounded_request_metadata(request_metadata or {})
    lines = [
        f"# Evidence Pack: {query}",
        "",
        f"Generated: {_now_stamp()}",
        f"Mode: {mode}",
        f"Retrieval goal: {safe_metadata.get('retrieval_goal', 'focused')}",
        "Intent: query",
        "",
        "## Request Metadata",
        "",
        "```json",
        json.dumps(safe_metadata, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## 1. Response",
        "",
        str(response.get("response", "")),
        "",
        "## 2. References",
        "",
    ]
    for i, ref in enumerate(refs, 1):
        lines.append(f"### Reference {i}")
        if isinstance(ref, dict):
            file_path = ref.get("file_path") or ref.get("source") or ""
            content = ref.get("content")
        else:
            file_path = str(ref)
            content = None
        lines.append(f"- file_path: `{file_path}`")
        if isinstance(content, list):
            for chunk in content[:3]:
                lines.extend(["", "```text", str(chunk)[:1200], "```"])
        lines.append("")
    data = response.get("data") if isinstance(response, dict) else None
    if isinstance(data, dict):
        lines.extend(["", "## 3. Retrieval Data", ""])
        for key in ["entities", "relationships", "chunks"]:
            values = data.get(key) or []
            lines.append(f"### {key} ({len(values)})")
            for item in values[:8]:
                lines.append("")
                lines.append("```json")
                lines.append(json.dumps(item, ensure_ascii=False, indent=2)[:1600])
                lines.append("```")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _bounded_request_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "retrieval_goal",
        "mode",
        "top_k",
        "neighbor_limit",
        "section_kind",
        "response_profile",
        "workspace_id",
        "record_types",
    )
    result: dict[str, Any] = {"retrieval_goal": "focused"}
    for key in allowed:
        if key not in metadata:
            continue
        value = metadata.get(key)
        if key == "retrieval_goal" and not isinstance(value, str):
            continue
        if isinstance(value, str):
            result[key] = value[:200]
        elif isinstance(value, (int, float, bool)) or value is None:
            result[key] = value
        elif isinstance(value, (list, tuple)):
            result[key] = [str(item)[:200] for item in value[:20]]

    while len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > 1600 and len(result) > 1:
        result.pop(next(reversed(result)))
    return result


def add_query_event(state_dir: Path, query: str, mode: str, evidence_pack_path: str | None = None) -> None:
    db = init_query_events_db(state_dir)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO query_events(query, mode, rewritten_queries, evidence_pack_path, created_at) VALUES(?,?,?,?,?)",
            (query, mode, None, evidence_pack_path, _now_stamp()),
        )
