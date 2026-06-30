"""SQLite versioned workspace storage for the native retrieval kernel."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
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


def _as_float32_vector(values: list[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("vector must be one-dimensional and non-empty")
    if not np.isfinite(vector).all():
        raise ValueError("vector must contain only finite values")
    if float(np.linalg.norm(vector)) == 0.0:
        raise ValueError("vector norm must be non-zero")
    return vector


def _vector_from_blob(blob: bytes, dim: int) -> np.ndarray:
    vector = np.frombuffer(blob, dtype=np.float32)
    if vector.size != dim:
        raise ValueError(f"stored vector dimension mismatch: expected {dim}, found {vector.size}")
    return vector


class SQLiteWorkspace:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
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

    def nearest_vectors(self, workspace_id: str, record_type: str, query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
        self.get_workspace_status(workspace_id)
        if top_k <= 0:
            return []
        query = _as_float32_vector(query_vector)
        query_norm = float(np.linalg.norm(query))
        results: list[dict[str, Any]] = []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT record_id, vector_hash, dim, vector_blob
                FROM vector
                WHERE workspace_id = ? AND record_type = ?
                """,
                (workspace_id, record_type),
            ).fetchall()
        for row in rows:
            vector = _vector_from_blob(bytes(row["vector_blob"]), int(row["dim"]))
            if vector.size != query.size:
                raise ValueError(f"dimension mismatch for {row['record_id']}: expected {vector.size}, found {query.size}")
            score = float(np.dot(query, vector) / (query_norm * float(np.linalg.norm(vector))))
            results.append({"record_type": record_type, "record_id": str(row["record_id"]), "vector_hash": str(row["vector_hash"]), "score": score})
        results.sort(key=lambda item: (-item["score"], item["record_id"]))
        return results[:top_k]

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
                WHERE workspace_id = ? AND (src_id = ? OR tgt_id = ?)
                """,
                (workspace_id, node_id, node_id),
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

    def activate_workspace(self, workspace_id: str) -> None:
        if self.get_workspace_status(workspace_id) != "audited":
            raise ValueError("workspace must be audited before activation")
        with self._connect() as conn:
            conn.execute("UPDATE workspace SET status = 'retired' WHERE status = 'active'")
            conn.execute("UPDATE workspace SET status = 'active' WHERE workspace_id = ?", (workspace_id,))
