"""Pure build helpers for native zvec workspace materialization."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from llm_wiki_native.contracts import SECTION_KIND_CODES, SOURCE_KIND_CODES
from llm_wiki_native.storage.zvec_records import ZvecRecord

_SECTION_KIND_ALIASES = {"method": "methodology"}
_MANIFEST_COLLECTIONS = (
    ("chunks", "chunk"),
    ("entities", "entity"),
    ("relationships", "relationship"),
)


class MissingNativeVectorsError(ValueError):
    """Raised when native materialization lacks required vector inputs."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        super().__init__(f"missing native vectors: {report['total_missing']} missing; by_type={report['by_type']}")


def missing_vector_report(
    manifest: dict[str, Any],
    raw_sections: list[dict[str, Any]],
    *,
    vectors_by_hash: dict[str, list[float]],
    section_embeddings_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    manifest_missing: list[dict[str, str]] = []
    section_missing: list[dict[str, str]] = []
    by_type: dict[str, int] = {}

    def add_count(record_type: str) -> None:
        by_type[record_type] = by_type.get(record_type, 0) + 1

    for collection_name, record_type in _MANIFEST_COLLECTIONS:
        records = manifest.get(collection_name, {})
        if not isinstance(records, dict):
            continue
        for record_id, record in records.items():
            if not isinstance(record, dict):
                continue
            vector_hash = _manifest_vector_hash(record)
            if vector_hash in vectors_by_hash:
                continue
            add_count(record_type)
            manifest_missing.append(
                {
                    "record_type": record_type,
                    "record_id": str(record_id),
                    "vector_hash": vector_hash,
                    "collection": collection_name,
                    "source_path": str(record.get("file_path") or record.get("source_path") or ""),
                }
            )

    for section in raw_sections:
        section_id = str(section.get("section_id") or "")
        if not section_id:
            continue
        embedding_row = section_embeddings_by_id.get(section_id)
        if isinstance(embedding_row, dict) and isinstance(embedding_row.get("embedding"), list):
            continue
        add_count("section")
        section_missing.append(
            {
                "record_type": "section",
                "record_id": section_id,
                "section_id": section_id,
                "source_path": str(section.get("source_path") or ""),
                "vector_hash": str(section.get("vector_hash") or section.get("text_hash") or ""),
            }
        )

    return {
        "total_missing": len(manifest_missing) + len(section_missing),
        "by_type": by_type,
        "manifest": manifest_missing,
        "sections": section_missing,
    }


def raise_if_missing_vectors(
    manifest: dict[str, Any],
    raw_sections: list[dict[str, Any]],
    *,
    vectors_by_hash: dict[str, list[float]],
    section_embeddings_by_id: dict[str, dict[str, Any]],
) -> None:
    report = missing_vector_report(
        manifest,
        raw_sections,
        vectors_by_hash=vectors_by_hash,
        section_embeddings_by_id=section_embeddings_by_id,
    )
    if report["total_missing"]:
        raise MissingNativeVectorsError(report)


def materialize_zvec_records(
    manifest: dict[str, Any],
    raw_sections: list[dict[str, Any]],
    *,
    vectors_by_hash: dict[str, list[float]],
    section_embeddings_by_id: dict[str, dict[str, Any]],
) -> list[ZvecRecord]:
    raise_if_missing_vectors(
        manifest,
        raw_sections,
        vectors_by_hash=vectors_by_hash,
        section_embeddings_by_id=section_embeddings_by_id,
    )
    records: list[ZvecRecord] = []
    for collection_name, record_type in _MANIFEST_COLLECTIONS:
        for record_id, record in manifest.get(collection_name, {}).items():
            records.append(
                _manifest_record(
                    record_type,
                    str(record_id),
                    record,
                    vectors_by_hash=vectors_by_hash,
                )
            )
    for section in raw_sections:
        records.append(
            _section_record(
                section,
                section_embeddings_by_id=section_embeddings_by_id,
            )
        )
    return records


def _manifest_record(
    record_type: str,
    record_id: str,
    record: dict[str, Any],
    *,
    vectors_by_hash: dict[str, list[float]],
) -> ZvecRecord:
    content_hash = _manifest_content_hash(record)
    if not content_hash:
        raise ValueError(f"manifest record missing hash: {record_id}")
    vector_hash = str(record.get("vector_hash") or content_hash)
    vector = vectors_by_hash.get(vector_hash)
    if vector is None:
        raise ValueError(f"missing vector for {record_type}: {record_id}")
    content = str(record.get("content", ""))
    source_path = str(record.get("file_path") or record.get("source_path") or "")
    return ZvecRecord(
        record_type=record_type,
        record_id=record_id,
        canonical_id=str(
            record.get("canonical_id")
            or record.get("chunk_id")
            or record.get("entity_name")
            or record_id
        ),
        source_id=str(record.get("source_id") or record.get("source_logical_id") or ""),
        source_kind_code=SOURCE_KIND_CODES["compiled"],
        source_path_hash=_hash_text(source_path),
        source_path=source_path,
        title=str(record.get("title") or record.get("paper_title") or ""),
        vector_hash=vector_hash,
        content_hash=content_hash,
        metadata_hash=str(record.get("metadata_hash") or record.get("record_hash") or content_hash),
        content=content,
        tokens=_token_count(content),
        embedding=list(vector),
    )


def _manifest_content_hash(record: dict[str, Any]) -> str:
    return str(
        record.get("content_hash")
        or record.get("vector_hash")
        or record.get("record_hash")
        or ""
    )


def _manifest_vector_hash(record: dict[str, Any]) -> str:
    return str(record.get("vector_hash") or _manifest_content_hash(record))


def _section_record(
    section: dict[str, Any],
    *,
    section_embeddings_by_id: dict[str, dict[str, Any]],
) -> ZvecRecord:
    section_id = str(section.get("section_id") or "")
    if not section_id:
        raise ValueError("raw section missing section_id")
    embedding_row = section_embeddings_by_id.get(section_id)
    if not isinstance(embedding_row, dict) or not isinstance(embedding_row.get("embedding"), list):
        raise ValueError(f"missing section embedding for section: {section_id}")
    content = str(section.get("content", ""))
    content_hash = str(section.get("content_hash") or section.get("text_hash") or _stable_hash(content))
    embedding_hash = str(embedding_row.get("text_hash") or "")
    vector_hash = str(section.get("vector_hash") or section.get("text_hash") or embedding_hash or content_hash)
    metadata_hash = str(
        section.get("metadata_hash")
        or _stable_hash({key: value for key, value in section.items() if key != "content"})
    )
    source_path = str(section.get("source_path") or "")
    return ZvecRecord(
        record_type="section",
        record_id=section_id,
        canonical_id=str(section.get("canonical_id") or section_id),
        source_id=str(section.get("source_id") or ""),
        source_kind_code=SOURCE_KIND_CODES["raw"],
        source_path_hash=_hash_text(source_path),
        source_path=source_path,
        title=str(section.get("paper_title") or section.get("section_title") or ""),
        vector_hash=vector_hash,
        content_hash=content_hash,
        metadata_hash=metadata_hash,
        content=content,
        tokens=_token_count(content),
        embedding=list(embedding_row["embedding"]),
        section_kind=_normalize_section_kind(section.get("section_kind")),
    )


def _normalize_section_kind(value: Any) -> str | None:
    if value is None or value == "":
        return None
    section_kind = _SECTION_KIND_ALIASES.get(str(value).strip().lower(), str(value).strip().lower())
    if section_kind not in SECTION_KIND_CODES:
        raise ValueError(f"unknown section_kind: {value}")
    return section_kind


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _token_count(value: str) -> int:
    return len(value.split())
