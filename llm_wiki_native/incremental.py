"""Pure diff planning for incremental native workspace builds (speedup plan M5/P-07).

The planner is IO-free: callers map both sides to ``(record_class, key) -> fingerprint``
dictionaries and receive add/update/delete key sets. Fingerprints must cover payload-only
mutations, not just the manifest hash fields.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping

from llm_wiki_native.storage.sqlite_workspace import NativeRecord


def stable_fingerprint(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def native_record_fingerprint(record: NativeRecord) -> str:
    """Content fingerprint that also covers payload-only mutations."""
    return stable_fingerprint(
        {
            "content_hash": record.content_hash,
            "metadata_hash": record.metadata_hash,
            "vector_hash": record.vector_hash,
            "source_path": record.source_path,
            "source_id": record.source_id,
            "payload": record.payload,
        }
    )


@dataclass(frozen=True)
class Delta:
    added: frozenset[Any]
    updated: frozenset[Any]
    deleted: frozenset[Any]

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.deleted)


def diff_by_key(previous: Mapping[Any, str], current: Mapping[Any, str]) -> Delta:
    previous_keys = set(previous)
    current_keys = set(current)
    return Delta(
        added=frozenset(current_keys - previous_keys),
        updated=frozenset(key for key in previous_keys & current_keys if previous[key] != current[key]),
        deleted=frozenset(previous_keys - current_keys),
    )


def delta_summary(delta: Delta) -> dict[str, Any]:
    return {
        "added": len(delta.added),
        "updated": len(delta.updated),
        "deleted": len(delta.deleted),
        "changed": delta.changed,
    }


def record_row_fingerprint(row: Mapping[str, Any]) -> str:
    """Fingerprint shape matching native_record_fingerprint, for stored record index rows."""
    return stable_fingerprint(
        {
            "content_hash": row["content_hash"],
            "metadata_hash": row["metadata_hash"],
            "vector_hash": row["vector_hash"],
            "source_path": row["source_path"],
            "source_id": row["source_id"],
            "payload": row["payload"],
        }
    )


def fingerprints_from_record_rows(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], str]:
    return {(str(row["record_type"]), str(row["record_id"])): record_row_fingerprint(row) for row in rows}


def edge_fingerprint(weight: float, payload: Mapping[str, Any]) -> str:
    return stable_fingerprint({"weight": float(weight), "payload": payload})


def fingerprints_from_edge_rows(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str, str], str]:
    return {
        (str(row["edge_type"]), str(row["src_id"]), str(row["tgt_id"])): edge_fingerprint(float(row["weight"]), row["payload"])
        for row in rows
    }


def span_fingerprint(
    *,
    source_path: str,
    source_id: str,
    source_role: str,
    span_kind: str,
    heading_path: Iterable[str],
    start_line: int,
    end_line: int,
    text_hash: str,
    metadata: Mapping[str, Any],
) -> str:
    return stable_fingerprint(
        {
            "source_path": source_path,
            "source_id": source_id,
            "source_role": source_role,
            "span_kind": span_kind,
            "heading_path": [str(part) for part in heading_path],
            "start_line": int(start_line),
            "end_line": int(end_line),
            "text_hash": text_hash,
            "metadata": dict(metadata),
        }
    )


def fingerprints_from_span_rows(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], str]:
    return {
        ("lexical_span", str(row["span_id"])): span_fingerprint(
            source_path=str(row["source_path"]),
            source_id=str(row["source_id"]),
            source_role=str(row["source_role"]),
            span_kind=str(row["span_kind"]),
            heading_path=row["heading_path"],
            start_line=int(row["start_line"]),
            end_line=int(row["end_line"]),
            text_hash=str(row["text_hash"]),
            metadata=row["metadata"],
        )
        for row in rows
    }


def fingerprint_for_span_item(item: Any) -> str:
    """Fingerprint a LexicalSpan (post with_hash) with the same shape as stored span rows."""
    return span_fingerprint(
        source_path=item.source_path,
        source_id=item.source_id,
        source_role=item.source_role,
        span_kind=item.span_kind,
        heading_path=item.heading_path,
        start_line=item.start_line,
        end_line=item.end_line,
        text_hash=item.text_hash,
        metadata=item.metadata,
    )
