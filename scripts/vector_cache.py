"""Persistent vector cache for llm-wiki full materialization."""

from __future__ import annotations

from array import array
import base64
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any


GRAPH_FIELD_SEP = "<SEP>"


def _compute_mdhash_id(content: str, prefix: str = "") -> str:
    try:
        digest = hashlib.md5(str(content).encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        digest = hashlib.md5(str(content).encode("utf-8", errors="replace")).hexdigest()
    return prefix + digest


class VectorCache:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vector_cache(
                  vector_hash TEXT PRIMARY KEY,
                  record_type TEXT NOT NULL,
                  record_id TEXT NOT NULL,
                  embedding_model TEXT NOT NULL,
                  embedding_dim INTEGER NOT NULL,
                  embedding_params_version TEXT NOT NULL,
                  vector_blob BLOB NOT NULL,
                  vector_sha256 TEXT NOT NULL,
                  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_used_at TEXT
                )
                """
            )

    @staticmethod
    def _encode_vector(vector: list[float], embedding_dim: int) -> bytes:
        if len(vector) != embedding_dim:
            raise ValueError(f"vector dimension mismatch: expected {embedding_dim}, found {len(vector)}")
        if any(not math.isfinite(value) for value in vector):
            raise ValueError("vector must contain only finite values")
        return array("f", vector).tobytes()

    @staticmethod
    def _decode_vector(blob: bytes, embedding_dim: int) -> list[float] | None:
        values = array("f")
        try:
            values.frombytes(blob)
        except ValueError:
            return None
        if len(values) != embedding_dim:
            return None
        return list(values)

    @staticmethod
    def _sha256(blob: bytes) -> str:
        return hashlib.sha256(blob).hexdigest()

    @staticmethod
    def _chunks(values: list[str], size: int = 900) -> list[list[str]]:
        return [values[index : index + size] for index in range(0, len(values), size)]

    def put(
        self,
        vector_hash: str,
        *,
        record_type: str,
        record_id: str,
        embedding_model: str,
        embedding_dim: int,
        embedding_params_version: str,
        vector: list[float],
    ) -> None:
        self.put_many(
            [
                {
                    "vector_hash": vector_hash,
                    "record_type": record_type,
                    "record_id": record_id,
                    "embedding_model": embedding_model,
                    "embedding_dim": embedding_dim,
                    "embedding_params_version": embedding_params_version,
                    "vector": vector,
                }
            ]
        )

    def put_many(self, records: list[dict[str, Any]]) -> int:
        """Insert or replace cached vectors in one SQLite transaction."""

        rows: list[tuple[str, str, str, str, int, str, bytes, str]] = []
        for record in records:
            embedding_dim = int(record["embedding_dim"])
            blob = self._encode_vector(record["vector"], embedding_dim)
            rows.append(
                (
                    str(record["vector_hash"]),
                    str(record["record_type"]),
                    str(record["record_id"]),
                    str(record["embedding_model"]),
                    embedding_dim,
                    str(record["embedding_params_version"]),
                    blob,
                    self._sha256(blob),
                )
            )
        if not rows:
            return 0
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO vector_cache(
                  vector_hash, record_type, record_id, embedding_model, embedding_dim,
                  embedding_params_version, vector_blob, vector_sha256, last_used_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                rows,
            )
        return len(rows)

    def resolve(
        self,
        vector_hash: str,
        *,
        embedding_model: str,
        embedding_dim: int,
        embedding_params_version: str,
    ) -> dict[str, Any] | None:
        return self.resolve_many(
            [
                {
                    "vector_hash": vector_hash,
                    "embedding_model": embedding_model,
                    "embedding_dim": embedding_dim,
                    "embedding_params_version": embedding_params_version,
                }
            ]
        ).get(str(vector_hash))

    def resolve_many(self, requests: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Resolve cached vectors for many embedding contracts with batched SQLite IO."""

        wanted: dict[str, dict[str, Any]] = {}
        for request in requests:
            vector_hash = request.get("vector_hash")
            embedding_model = request.get("embedding_model")
            embedding_dim = request.get("embedding_dim")
            embedding_params_version = request.get("embedding_params_version")
            if not vector_hash or not embedding_model or embedding_dim is None or not embedding_params_version:
                continue
            wanted[str(vector_hash)] = {
                "embedding_model": str(embedding_model),
                "embedding_dim": int(embedding_dim),
                "embedding_params_version": str(embedding_params_version),
            }
        if not wanted:
            return {}

        rows: dict[str, sqlite3.Row] = {}
        resolved: dict[str, dict[str, Any]] = {}
        vector_hashes = list(wanted)
        with self._connect() as conn:
            for chunk in self._chunks(vector_hashes):
                placeholders = ",".join("?" for _ in chunk)
                for row in conn.execute(f"SELECT * FROM vector_cache WHERE vector_hash IN ({placeholders})", chunk):
                    rows[str(row["vector_hash"])] = row
            used: list[str] = []
            for vector_hash, contract in wanted.items():
                row = rows.get(vector_hash)
                if row is None:
                    continue
                if str(row["embedding_model"]) != contract["embedding_model"]:
                    continue
                if int(row["embedding_dim"]) != contract["embedding_dim"]:
                    continue
                if str(row["embedding_params_version"]) != contract["embedding_params_version"]:
                    continue
                blob = bytes(row["vector_blob"])
                if self._sha256(blob) != str(row["vector_sha256"]):
                    continue
                vector = self._decode_vector(blob, int(row["embedding_dim"]))
                if vector is None:
                    continue
                resolved[vector_hash] = {
                    "record_type": str(row["record_type"]),
                    "record_id": str(row["record_id"]),
                    "vector": vector,
                }
                used.append(vector_hash)
            for chunk in self._chunks(used):
                placeholders = ",".join("?" for _ in chunk)
                conn.execute(f"UPDATE vector_cache SET last_used_at = CURRENT_TIMESTAMP WHERE vector_hash IN ({placeholders})", chunk)
        return resolved


def _empty_collection_result() -> tuple[dict[str, dict[str, Any]], list[str], dict[str, int]]:
    return {}, [], {"total": 0, "hits": 0, "misses": 0}


_COLLECTION_RECORD_TYPES = {"chunks": "chunk", "entities": "entity", "relationships": "relationship"}


def resolve_manifest_vectors(manifest: dict[str, Any], cache: VectorCache) -> dict[str, Any]:
    """Resolve cached vectors for all vector-like records in a custom_kg manifest.

    Misses are reported instead of embedded. Callers can use the report to decide
    whether to run an embedding fill phase or fall back to cold materialization.
    """

    resolved: dict[str, dict[str, dict[str, Any]]] = {}
    missing: dict[str, list[str]] = {}
    summary: dict[str, dict[str, int]] = {}
    entries: list[tuple[str, str, dict[str, Any], str | None]] = []
    requests: list[dict[str, Any]] = []
    for collection in ("chunks", "entities", "relationships"):
        records = manifest.get(collection, {})
        if not isinstance(records, dict):
            records = {}
        for key in sorted(records):
            record = records[key]
            vector_hash = record.get("vector_hash") if isinstance(record, dict) else None
            embedding_model = record.get("embedding_model") if isinstance(record, dict) else None
            embedding_dim = record.get("embedding_dim") if isinstance(record, dict) else None
            embedding_params_version = record.get("embedding_params_version") if isinstance(record, dict) else None
            valid_hash = str(vector_hash) if vector_hash and embedding_model and embedding_dim is not None and embedding_params_version else None
            if valid_hash is not None:
                request_embedding_dim = int(str(embedding_dim))
                requests.append(
                    {
                        "vector_hash": valid_hash,
                        "embedding_model": str(embedding_model),
                        "embedding_dim": request_embedding_dim,
                        "embedding_params_version": str(embedding_params_version),
                    }
                )
            entries.append((collection, str(key), record if isinstance(record, dict) else {}, valid_hash))

    cached_by_hash = cache.resolve_many(requests)
    for collection in ("chunks", "entities", "relationships"):
        collection_resolved, collection_missing, collection_summary = _empty_collection_result()
        for entry_collection, key, record, vector_hash in entries:
            if entry_collection != collection:
                continue
            collection_summary["total"] += 1
            cached = cached_by_hash.get(vector_hash) if vector_hash is not None else None
            if cached is None:
                collection_summary["misses"] += 1
                collection_missing.append(key)
                continue
            collection_summary["hits"] += 1
            collection_resolved[key] = {
                "record_type": str(record.get("record_type") or _COLLECTION_RECORD_TYPES[collection]),
                "record_id": str(record.get("record_id") or key),
                "canonical_id": str(record.get("canonical_id") or key),
                "vector_hash": str(vector_hash),
                "cached_record_id": cached["record_id"],
                "vector": cached["vector"],
            }
        resolved[collection] = collection_resolved
        missing[collection] = collection_missing
        summary[collection] = collection_summary
    summary["total"] = {
        "total": sum(item["total"] for key, item in summary.items() if key != "total"),
        "hits": sum(item["hits"] for key, item in summary.items() if key != "total"),
        "misses": sum(item["misses"] for key, item in summary.items() if key != "total"),
    }
    return {"resolved": resolved, "missing": missing, "summary": summary}
