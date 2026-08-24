"""SQLite sidecar for section_embeddings.jsonl.

The jsonl file (~200MB) is rewritten and re-parsed several times per refresh.
This sidecar stores the same rows keyed by section_id: the full row as JSON
(minus the embedding) plus the embedding as little-endian float64 bytes, which
round-trips bit-exactly with the JSON floats written by jsonl_write.

Transition policy: the jsonl write was dropped on 2026-08-21 after
migration verification E-migration-f and explicit approval. The sidecar is
now the only write target; readers prefer the sidecar and fall back to a
legacy jsonl only for pre-migration state directories.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SIDECAR_NAME = "section_embeddings.sqlite"
JSONL_NAME = "section_embeddings.jsonl"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS section_embedding(
    section_id TEXT PRIMARY KEY,
    row_json TEXT NOT NULL,
    embedding BLOB NOT NULL
)
"""


def sidecar_path(state_dir: Path) -> Path:
    return Path(state_dir) / SIDECAR_NAME


def jsonl_path(state_dir: Path) -> Path:
    return Path(state_dir) / JSONL_NAME


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)


def upsert_rows(state_dir: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Insert/replace rows (embedding list -> float64 blob). Returns row count."""
    rows = list(rows)
    if not rows:
        return 0
    conn = sqlite3.connect(sidecar_path(state_dir))
    try:
        _ensure_schema(conn)
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO section_embedding(section_id, row_json, embedding) VALUES (?, ?, ?)",
                [
                    (
                        str(row["section_id"]),
                        json.dumps({k: v for k, v in row.items() if k != "embedding"}, ensure_ascii=False, sort_keys=True),
                        np.asarray(row["embedding"], dtype="<f8").tobytes(),
                    )
                    for row in rows
                ],
            )
    finally:
        conn.close()
    return len(rows)


def load_rows(state_dir: Path) -> list[dict[str, Any]] | None:
    """Return rows sorted by section_id with embedding as float64 ndarray, or None if no sidecar."""
    path = sidecar_path(state_dir)
    if not path.exists():
        return None
    conn = sqlite3.connect(path)
    try:
        _ensure_schema(conn)
        fetched = conn.execute(
            "SELECT section_id, row_json, embedding FROM section_embedding ORDER BY section_id"
        ).fetchall()
    finally:
        conn.close()
    rows: list[dict[str, Any]] = []
    for section_id, row_json, blob in fetched:
        row = json.loads(row_json)
        row["section_id"] = section_id
        row["embedding"] = np.frombuffer(blob, dtype="<f8")
        rows.append(row)
    return rows
