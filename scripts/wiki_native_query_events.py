#!/usr/bin/env python3
"""Native query event and evidence-pack helpers."""

from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from wiki_native_state import ensure_state_dirs

WIKIGRAPH_SYNC_DB_FILENAME = "wikigraph_sync.db"


def _now_stamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ensure_state_dirs(state_dir: Path) -> None:
    ensure_state_dirs(state_dir)


def slugify(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", text.strip()).strip("-").lower()
    return (slug or "item")[:max_len].strip("-") or "item"


def init_manifest_db(state_dir: Path) -> Path:
    _ensure_state_dirs(state_dir)
    db = state_dir / WIKIGRAPH_SYNC_DB_FILENAME
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS docs (
              canonical_id TEXT PRIMARY KEY,
              rel_path TEXT NOT NULL,
              doc_type TEXT NOT NULL,
              sha256 TEXT NOT NULL,
              title TEXT,
              updated TEXT,
              wikigraph_track_id TEXT,
              wikigraph_doc_status TEXT,
              last_synced_at TEXT,
              deleted INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS sync_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              canonical_id TEXT NOT NULL,
              event_type TEXT NOT NULL,
              old_sha256 TEXT,
              new_sha256 TEXT,
              track_id TEXT,
              status TEXT,
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS query_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              query TEXT NOT NULL,
              mode TEXT NOT NULL,
              rewritten_queries TEXT,
              evidence_pack_path TEXT,
              created_at TEXT NOT NULL
            );
            """
        )
    return db


def save_evidence_pack(state_dir: Path, query: str, mode: str, response: dict[str, Any]) -> Path:
    _ensure_state_dirs(state_dir)
    slug = slugify(query, 70)
    path = state_dir / "evidence_packs" / f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{slug}.md"
    refs = response.get("references") or []
    lines = [
        f"# Evidence Pack: {query}",
        "",
        f"Generated: {_now_stamp()}",
        f"Mode: {mode}",
        "Intent: query",
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


def add_query_event(state_dir: Path, query: str, mode: str, evidence_pack_path: str | None = None) -> None:
    db = init_manifest_db(state_dir)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO query_events(query, mode, rewritten_queries, evidence_pack_path, created_at) VALUES(?,?,?,?,?)",
            (query, mode, None, evidence_pack_path, _now_stamp()),
        )
