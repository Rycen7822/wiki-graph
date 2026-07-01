"""SQLite versioned workspace storage for the native retrieval kernel."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

import numpy as np

from llm_wiki_native.contracts import RECORD_TYPES

_COUNT_KEYS = {
    "chunk": "chunks",
    "entity": "entities",
    "relationship": "relationships",
    "section": "sections",
}
_ZERO_COUNTS = {value: 0 for value in _COUNT_KEYS.values()}


@dataclass(frozen=True)
class NativeRecord:
    workspace_id: str
    record_type: str
    record_id: str
    vector_text: str
    content_hash: str
    metadata_hash: str
    vector_hash: str
    source_path: str | None = None
    source_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LexicalSpan:
    workspace_id: str
    span_id: str
    source_path: str
    source_id: str
    source_role: str
    span_kind: str
    heading_path: list[str]
    start_line: int
    end_line: int
    text: str
    text_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _as_float32_vector(values: list[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("vector must be one-dimensional and non-empty")
    if not np.isfinite(vector).all():
        raise ValueError("vector must contain only finite values")
    if float(np.linalg.norm(vector)) == 0.0:
        raise ValueError("vector norm must be non-zero")
    return vector


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _query_terms(query: str) -> list[str]:
    return [term for term in re.findall(r"[0-9A-Za-z_\u4e00-\u9fff]+", query) if term]


def _fts_query(query: str) -> str:
    terms = _query_terms(query)
    return " ".join(f'"{term.replace(chr(34), "")}"' for term in terms[:8])


def _append_in_filter(filters: list[str], params: list[Any], column: str, values: tuple[str, ...]) -> None:
    cleaned = tuple(str(value) for value in values if str(value).strip())
    if not cleaned:
        return
    placeholders = ",".join("?" for _ in cleaned)
    filters.append(f"{column} IN ({placeholders})")
    params.extend(cleaned)


def _lexical_span_from_row(row: sqlite3.Row, *, route: str, lexical_rank: float | None = None) -> dict[str, Any]:
    data = dict(row)
    heading_path = json.loads(str(data.pop("heading_path_json")))
    metadata = json.loads(str(data.pop("metadata_json")))
    result = {
        **data,
        "heading_path": heading_path,
        "metadata": metadata,
        "route": route,
    }
    if lexical_rank is not None:
        result["lexical_rank"] = lexical_rank
    return result


class SQLiteWorkspace:

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._read_only = False
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @classmethod
    def open_existing(cls, db_path: Path, *, read_only: bool = True) -> "SQLiteWorkspace":
        path = Path(db_path)
        if not path.exists():
            raise FileNotFoundError(path)
        workspace = cls.__new__(cls)
        workspace.db_path = path
        workspace._read_only = bool(read_only)
        return workspace

    def _connect(self) -> sqlite3.Connection:
        if self._read_only:
            conn = sqlite3.connect(f"{self.db_path.resolve().as_uri()}?mode=ro", timeout=30.0, uri=True)
        else:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        if self._read_only:
            conn.execute("PRAGMA query_only = ON")
        else:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workspace(
                  workspace_id TEXT PRIMARY KEY,
                  source_manifest_hash TEXT NOT NULL,
                  schema_version INTEGER NOT NULL,
                  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  status TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS record(
                  workspace_id TEXT NOT NULL,
                  record_type TEXT NOT NULL,
                  record_id TEXT NOT NULL,
                  vector_text TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  metadata_hash TEXT NOT NULL,
                  vector_hash TEXT NOT NULL,
                  source_path TEXT,
                  source_id TEXT,
                  payload_json TEXT NOT NULL,
                  PRIMARY KEY(workspace_id, record_type, record_id),
                  FOREIGN KEY(workspace_id) REFERENCES workspace(workspace_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vector(
                  workspace_id TEXT NOT NULL,
                  record_type TEXT NOT NULL,
                  record_id TEXT NOT NULL,
                  vector_hash TEXT NOT NULL,
                  dim INTEGER NOT NULL,
                  vector_blob BLOB NOT NULL,
                  PRIMARY KEY(workspace_id, record_type, record_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS edge(
                  workspace_id TEXT NOT NULL,
                  edge_type TEXT NOT NULL,
                  src_id TEXT NOT NULL,
                  tgt_id TEXT NOT NULL,
                  weight REAL NOT NULL,
                  payload_json TEXT NOT NULL,
                  PRIMARY KEY(workspace_id, edge_type, src_id, tgt_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edge_workspace_src ON edge(workspace_id, src_id, edge_type, weight)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_edge_workspace_tgt ON edge(workspace_id, tgt_id, edge_type, weight)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS lexical_span(
                  workspace_id TEXT NOT NULL,
                  span_id TEXT NOT NULL,
                  source_path TEXT NOT NULL,
                  source_id TEXT NOT NULL,
                  source_role TEXT NOT NULL,
                  span_kind TEXT NOT NULL,
                  heading_path_json TEXT NOT NULL,
                  start_line INTEGER NOT NULL,
                  end_line INTEGER NOT NULL,
                  text TEXT NOT NULL,
                  text_hash TEXT NOT NULL,
                  metadata_json TEXT NOT NULL,
                  PRIMARY KEY(workspace_id, span_id),
                  FOREIGN KEY(workspace_id) REFERENCES workspace(workspace_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lexical_span_workspace_role ON lexical_span(workspace_id, source_role, span_kind)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lexical_span_workspace_source ON lexical_span(workspace_id, source_path, start_line)")
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS lexical_span_fts USING fts5(
                  workspace_id UNINDEXED,
                  span_id UNINDEXED,
                  source_path,
                  source_id,
                  source_role,
                  span_kind,
                  heading_path,
                  text,
                  metadata
                )
                """
            )

    def create_workspace(self, workspace_id: str, source_manifest_hash: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO workspace(workspace_id, source_manifest_hash, schema_version, status)
                VALUES(?, ?, 1, 'building')
                """,
                (workspace_id, source_manifest_hash),
            )

    def get_workspace_status(self, workspace_id: str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM workspace WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()
        if row is None:
            raise KeyError(workspace_id)
        return str(row["status"])

    def put_record(self, record: NativeRecord) -> None:
        self.get_workspace_status(record.workspace_id)
        if record.record_type not in RECORD_TYPES:
            raise ValueError(f"unknown record_type: {record.record_type}")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO record(
                  workspace_id, record_type, record_id, vector_text, content_hash,
                  metadata_hash, vector_hash, source_path, source_id, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.workspace_id,
                    record.record_type,
                    record.record_id,
                    record.vector_text,
                    record.content_hash,
                    record.metadata_hash,
                    record.vector_hash,
                    record.source_path,
                    record.source_id,
                    json.dumps(record.payload, ensure_ascii=False, sort_keys=True),
                ),
            )

    def count_records(self, workspace_id: str) -> dict[str, int]:
        self.get_workspace_status(workspace_id)
        counts = dict(_ZERO_COUNTS)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT record_type, COUNT(*) AS count
                FROM record
                WHERE workspace_id = ?
                GROUP BY record_type
                """,
                (workspace_id,),
            ).fetchall()
        for row in rows:
            key = _COUNT_KEYS.get(str(row["record_type"]))
            if key:
                counts[key] = int(row["count"])
        return counts

    def get_record(self, workspace_id: str, record_type: str, record_id: str) -> dict[str, Any]:
        self.get_workspace_status(workspace_id)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM record
                WHERE workspace_id = ? AND record_type = ? AND record_id = ?
                """,
                (workspace_id, record_type, record_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"{workspace_id}:{record_type}:{record_id}")
        data = dict(row)
        data["payload"] = json.loads(str(data.pop("payload_json")))
        return data

    def put_vector(self, workspace_id: str, record_type: str, record_id: str, vector_hash: str, vector: list[float]) -> None:
        record = self.get_record(workspace_id, record_type, record_id)
        if record["vector_hash"] != vector_hash:
            raise ValueError(f"vector_hash mismatch for {record_id}")
        vector_array = _as_float32_vector(vector)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO vector(workspace_id, record_type, record_id, vector_hash, dim, vector_blob)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (workspace_id, record_type, record_id, vector_hash, int(vector_array.size), vector_array.tobytes()),
            )

    def put_edge(self, workspace_id: str, edge_type: str, src_id: str, tgt_id: str, weight: float, payload: dict[str, Any]) -> None:
        self.get_workspace_status(workspace_id)
        if not edge_type or not src_id or not tgt_id:
            raise ValueError("edge_type, src_id, and tgt_id are required")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO edge(workspace_id, edge_type, src_id, tgt_id, weight, payload_json)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (workspace_id, edge_type, src_id, tgt_id, float(weight), json.dumps(payload, ensure_ascii=False, sort_keys=True)),
            )

    def neighbors(
        self,
        workspace_id: str,
        node_id: str,
        *,
        edge_types: set[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self.get_workspace_status(workspace_id)
        if limit is not None and limit <= 0:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT edge_type, src_id, tgt_id, weight, payload_json
                FROM edge
                WHERE workspace_id = ? AND src_id = ?
                UNION ALL
                SELECT edge_type, src_id, tgt_id, weight, payload_json
                FROM edge
                WHERE workspace_id = ? AND tgt_id = ? AND src_id <> ?
                """,
                (workspace_id, node_id, workspace_id, node_id, node_id),
            ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            edge_type = str(row["edge_type"])
            if edge_types is not None and edge_type not in edge_types:
                continue
            src_id = str(row["src_id"])
            tgt_id = str(row["tgt_id"])
            results.append(
                {
                    "edge_type": edge_type,
                    "neighbor_id": tgt_id if src_id == node_id else src_id,
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "weight": float(row["weight"]),
                    "payload": json.loads(str(row["payload_json"])),
                }
            )
        results.sort(key=lambda item: (-item["weight"], item["edge_type"], item["neighbor_id"]))
        return results[:limit] if limit is not None else results

    def put_lexical_span(
        self,
        workspace_id: str,
        *,
        span_id: str,
        source_path: str,
        source_id: str,
        source_role: str,
        span_kind: str,
        heading_path: list[str] | tuple[str, ...] | None = None,
        start_line: int = 0,
        end_line: int = 0,
        text: str,
        metadata: dict[str, Any] | None = None,
        text_hash: str | None = None,
    ) -> None:
        self.get_workspace_status(workspace_id)
        if not span_id.strip():
            raise ValueError("span_id is required")
        if not source_path.strip():
            raise ValueError("source_path is required")
        if not source_role.strip() or not span_kind.strip():
            raise ValueError("source_role and span_kind are required")
        if end_line and start_line and end_line < start_line:
            raise ValueError("end_line must be >= start_line")
        heading = [str(part) for part in (heading_path or [])]
        metadata = dict(metadata or {})
        text = str(text)
        text_hash = text_hash or _sha256_text(text)
        heading_json = json.dumps(heading, ensure_ascii=False, sort_keys=True)
        metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO lexical_span(
                  workspace_id, span_id, source_path, source_id, source_role, span_kind,
                  heading_path_json, start_line, end_line, text, text_hash, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workspace_id,
                    span_id,
                    source_path,
                    source_id,
                    source_role,
                    span_kind,
                    heading_json,
                    int(start_line),
                    int(end_line),
                    text,
                    text_hash,
                    metadata_json,
                ),
            )
            conn.execute("DELETE FROM lexical_span_fts WHERE workspace_id = ? AND span_id = ?", (workspace_id, span_id))
            conn.execute(
                """
                INSERT INTO lexical_span_fts(
                  workspace_id, span_id, source_path, source_id, source_role, span_kind,
                  heading_path, text, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workspace_id,
                    span_id,
                    source_path,
                    source_id,
                    source_role,
                    span_kind,
                    " / ".join(heading),
                    text,
                    metadata_json,
                ),
            )

    def get_lexical_span(self, workspace_id: str, span_id: str) -> dict[str, Any]:
        self.get_workspace_status(workspace_id)
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM lexical_span
                WHERE workspace_id = ? AND span_id = ?
                """,
                (workspace_id, span_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"{workspace_id}:lexical_span:{span_id}")
        return _lexical_span_from_row(row, route="lexical_lookup")

    def count_lexical_spans(self, workspace_id: str) -> int:
        self.get_workspace_status(workspace_id)
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM lexical_span WHERE workspace_id = ?", (workspace_id,)).fetchone()
        return int(row["count"])

    def query_lexical_spans(
        self,
        workspace_id: str,
        query: str,
        *,
        limit: int = 20,
        source_roles: tuple[str, ...] | list[str] | None = None,
        span_kinds: tuple[str, ...] | list[str] | None = None,
    ) -> list[dict[str, Any]]:
        self.get_workspace_status(workspace_id)
        if limit <= 0:
            return []
        source_roles = tuple(source_roles or ())
        span_kinds = tuple(span_kinds or ())
        fts_query = _fts_query(query)
        if fts_query:
            try:
                hits = self._query_lexical_spans_fts(workspace_id, fts_query, limit=limit, source_roles=source_roles, span_kinds=span_kinds)
            except sqlite3.OperationalError:
                hits = []
            if hits:
                return hits
        return self._query_lexical_spans_like(workspace_id, query, limit=limit, source_roles=source_roles, span_kinds=span_kinds)

    def _query_lexical_spans_fts(
        self,
        workspace_id: str,
        fts_query: str,
        *,
        limit: int,
        source_roles: tuple[str, ...],
        span_kinds: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        filters = ["lexical_span_fts MATCH ?", "lexical_span_fts.workspace_id = ?"]
        params: list[Any] = [fts_query, workspace_id]
        _append_in_filter(filters, params, "s.source_role", source_roles)
        _append_in_filter(filters, params, "s.span_kind", span_kinds)
        params.append(limit)
        sql = f"""
            SELECT s.*, bm25(lexical_span_fts) AS lexical_rank
            FROM lexical_span_fts
            JOIN lexical_span AS s
              ON s.workspace_id = lexical_span_fts.workspace_id
             AND s.span_id = lexical_span_fts.span_id
            WHERE {' AND '.join(filters)}
            ORDER BY lexical_rank ASC, s.source_path ASC, s.start_line ASC, s.span_id ASC
            LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_lexical_span_from_row(row, route="lexical_fts", lexical_rank=float(row["lexical_rank"])) for row in rows]

    def _query_lexical_spans_like(
        self,
        workspace_id: str,
        query: str,
        *,
        limit: int,
        source_roles: tuple[str, ...],
        span_kinds: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        terms = _query_terms(query) or [query.strip()]
        filters = ["workspace_id = ?"]
        params: list[Any] = [workspace_id]
        _append_in_filter(filters, params, "source_role", source_roles)
        _append_in_filter(filters, params, "span_kind", span_kinds)
        like_parts = []
        for term in terms[:8]:
            pattern = f"%{term}%"
            like_parts.append("(text LIKE ? OR source_path LIKE ? OR source_id LIKE ? OR heading_path_json LIKE ? OR metadata_json LIKE ?)")
            params.extend([pattern, pattern, pattern, pattern, pattern])
        if like_parts:
            filters.append("(" + " OR ".join(like_parts) + ")")
        params.append(limit)
        sql = f"""
            SELECT * FROM lexical_span
            WHERE {' AND '.join(filters)}
            ORDER BY source_path ASC, start_line ASC, span_id ASC
            LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_lexical_span_from_row(row, route="lexical_like") for row in rows]

    def count_edges(self, workspace_id: str) -> int:
        self.get_workspace_status(workspace_id)
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM edge WHERE workspace_id = ?", (workspace_id,)).fetchone()
        return int(row["count"])

    def audit_counts(self, workspace_id: str, expected: dict[str, int]) -> dict[str, Any]:
        counts = self.count_records(workspace_id)
        issues = [
            f"{key}: expected {int(expected.get(key, 0))}, found {counts.get(key, 0)}"
            for key in sorted(_ZERO_COUNTS)
            if counts.get(key, 0) != int(expected.get(key, 0))
        ]
        return {"ok": not issues, "counts": counts, "expected": {**_ZERO_COUNTS, **expected}, "issues": issues}

    def audit_vector_coverage(self, workspace_id: str) -> dict[str, Any]:
        self.get_workspace_status(workspace_id)
        counts = {record_type: {"records": 0, "vectors": 0, "missing": 0} for record_type in sorted(RECORD_TYPES)}
        missing = {record_type: [] for record_type in sorted(RECORD_TYPES)}
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT r.record_type, r.record_id, v.record_id AS vector_record_id
                FROM record AS r
                LEFT JOIN vector AS v
                  ON v.workspace_id = r.workspace_id
                 AND v.record_type = r.record_type
                 AND v.record_id = r.record_id
                WHERE r.workspace_id = ?
                ORDER BY r.record_type, r.record_id
                """,
                (workspace_id,),
            ).fetchall()
        for row in rows:
            record_type = str(row["record_type"])
            counts.setdefault(record_type, {"records": 0, "vectors": 0, "missing": 0})
            missing.setdefault(record_type, [])
            counts[record_type]["records"] += 1
            if row["vector_record_id"] is None:
                counts[record_type]["missing"] += 1
                missing[record_type].append(str(row["record_id"]))
            else:
                counts[record_type]["vectors"] += 1
        missing = {record_type: ids for record_type, ids in missing.items() if ids}
        return {"ok": not missing, "counts": counts, "missing": missing}

    def mark_audited(self, workspace_id: str, expected: dict[str, int], *, require_vectors: bool = False) -> None:
        audit = self.audit_counts(workspace_id, expected)
        if not audit["ok"]:
            raise ValueError(f"workspace audit failed: {audit['issues']}")
        if require_vectors:
            vector_audit = self.audit_vector_coverage(workspace_id)
            if not vector_audit["ok"]:
                raise ValueError(f"workspace vector coverage failed: {vector_audit['missing']}")
        with self._connect() as conn:
            conn.execute("UPDATE workspace SET status = 'audited' WHERE workspace_id = ?", (workspace_id,))
