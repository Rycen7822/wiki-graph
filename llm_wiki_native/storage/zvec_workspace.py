"""Zvec collection owner seam for native retrieval workspaces."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable

from llm_wiki_native.contracts import (
    RECORD_TYPE_CODES,
    SECTION_KIND_CODES,
    WORKSPACE_SCHEMA_VERSION,
)
from llm_wiki_native.storage.zvec_records import ZvecRecord

from zvec import (
    CollectionOption,
    CollectionSchema,
    DataType,
    Doc,
    FieldSchema,
    Fts,
    FtsIndexParam,
    FtsQueryParam,
    HnswIndexParam,
    HnswQueryParam,
    InvertIndexParam,
    MetricType,
    Query,
    RrfReRanker,
    VectorSchema,
    create_and_open,
    open as zvec_open,
)

COLLECTION_NAME = "llm_wiki_records_v1"
MAX_ZVEC_WRITE_BATCH_SIZE = 1024
MAX_ZVEC_DOC_ID_LENGTH = 64


def zvec_doc_id(record_type: str, record_id: str) -> str:
    record_type_code(record_type)
    if not record_id.strip():
        raise ValueError("record_id must not be empty")
    encoded = base64.urlsafe_b64encode(record_id.encode("utf-8")).decode("ascii").rstrip("=")
    doc_id = f"{record_type}__{encoded}"
    if len(doc_id) <= MAX_ZVEC_DOC_ID_LENGTH:
        return doc_id
    digest = hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:32]
    return f"{record_type}__h_{digest}"


def record_type_code(record_type: str) -> int:
    try:
        return RECORD_TYPE_CODES[record_type]
    except KeyError as exc:
        raise ValueError(f"unknown record_type: {record_type}") from exc


def section_kind_code(section_kind: str | None) -> int:
    if section_kind is None or section_kind == "":
        return SECTION_KIND_CODES["none"]
    try:
        return SECTION_KIND_CODES[section_kind]
    except KeyError as exc:
        raise ValueError(f"unknown section_kind: {section_kind}") from exc


@dataclass(frozen=True)
class InsertStats:
    attempted: int
    inserted: int
    failed: int


@dataclass(frozen=True)
class DeleteStats:
    attempted: int
    deleted: int
    failed: int


@dataclass(frozen=True)
class SmokeResult:
    checked: int
    passed: int
    failures: list[str]


@dataclass(frozen=True)
class ZvecHit:
    doc_id: str
    score: float
    fields: dict[str, Any]


@dataclass(frozen=True)
class ZvecWorkspace:
    collection: Any

    def bulk_insert(
        self,
        records: Iterable[ZvecRecord],
        batch_size: int = MAX_ZVEC_WRITE_BATCH_SIZE,
    ) -> InsertStats:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        attempted = 0
        inserted = 0
        for batch in _batches(records, batch_size):
            docs = [zvec_doc_from_record(record) for record in batch]
            statuses = self.collection.insert(docs)
            attempted += len(docs)
            inserted += sum(1 for status in statuses if status.ok())
        return InsertStats(
            attempted=attempted,
            inserted=inserted,
            failed=attempted - inserted,
        )

    def upsert_records(
        self,
        records: Iterable[ZvecRecord],
        batch_size: int = MAX_ZVEC_WRITE_BATCH_SIZE,
    ) -> InsertStats:
        """Upsert docs under their stable zvec_doc_id identity (existing ids are updated, not duplicated)."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        attempted = 0
        written = 0
        for batch in _batches(records, batch_size):
            docs = [zvec_doc_from_record(record) for record in batch]
            statuses = self.collection.upsert(docs)
            if not isinstance(statuses, list):
                statuses = [statuses]
            attempted += len(docs)
            written += sum(1 for status in statuses if status.ok())
        return InsertStats(
            attempted=attempted,
            inserted=written,
            failed=attempted - written,
        )

    def delete_docs(self, doc_ids: Iterable[str]) -> DeleteStats:
        """Delete docs by id in a single batch chain; missing ids are not failures."""
        ids = [str(doc_id) for doc_id in doc_ids]
        if not ids:
            return DeleteStats(attempted=0, deleted=0, failed=0)

        attempted = 0
        deleted = 0
        for batch in _batches(ids, MAX_ZVEC_WRITE_BATCH_SIZE):
            statuses = self.collection.delete(batch)
            if not isinstance(statuses, list):
                statuses = [statuses]
            attempted += len(batch)
            deleted += sum(1 for status in statuses if status.ok())
        return DeleteStats(
            attempted=attempted,
            deleted=deleted,
            failed=attempted - deleted,
        )

    def flush_optimize_close(self) -> None:
        self.collection.flush()
        self.collection.optimize()
        close = getattr(self.collection, "close", None)
        if callable(close):
            close()

    def self_nearest_smoke(self, sample_doc_ids: list[str]) -> SmokeResult:
        docs = self.collection.fetch(sample_doc_ids, include_vector=True)
        checked = 0
        passed = 0
        failures: list[str] = []
        for doc_id in sample_doc_ids:
            doc = docs.get(doc_id)
            if doc is None:
                failures.append(f"{doc_id} missing")
                continue
            vector = _embedding_vector(doc)
            if not vector:
                failures.append(f"{doc_id} missing embedding")
                continue
            checked += 1
            hits = self.query_vector(vector, top_k=1, filter_expr=None)
            nearest = hits[0].doc_id if hits else "<none>"
            if nearest == doc_id:
                passed += 1
            else:
                failures.append(f"{doc_id} nearest {nearest}")
        return SmokeResult(checked=checked, passed=passed, failures=failures)

    def fetch(self, doc_ids: list[str]) -> dict[str, Doc]:
        return self.collection.fetch(doc_ids, include_vector=False)

    def query_vector(
        self,
        query_vector: list[float],
        top_k: int,
        filter_expr: str | None,
    ) -> list[ZvecHit]:
        docs = self.collection.query(
            queries=Query(
                field_name="embedding",
                vector=query_vector,
                param=HnswQueryParam(ef=200),
            ),
            topk=top_k,
            filter=filter_expr,
            include_vector=False,
        )
        return [_hit_from_doc(doc) for doc in docs]

    def query_mix(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int,
        filter_expr: str | None,
    ) -> list[ZvecHit]:
        docs = self.collection.query(
            queries=[
                Query(
                    field_name="embedding",
                    vector=query_vector,
                    param=HnswQueryParam(ef=200),
                ),
                Query(
                    field_name="content",
                    fts=Fts(match_string=query_text),
                    param=FtsQueryParam(default_operator="AND"),
                ),
            ],
            topk=top_k,
            filter=filter_expr,
            include_vector=False,
            reranker=RrfReRanker(rank_constant=60),
        )
        return [_hit_from_doc(doc) for doc in docs]

    def stats(self) -> dict[str, int]:
        return {"doc_count": int(self.collection.stats.doc_count)}


def zvec_doc_from_record(record: ZvecRecord) -> Doc:
    if not record.embedding:
        raise ValueError("embedding must not be empty")
    return Doc(
        id=zvec_doc_id(record.record_type, record.record_id),
        vectors={"embedding": list(record.embedding)},
        fields={
            "record_type_code": record_type_code(record.record_type),
            "section_kind_code": section_kind_code(record.section_kind),
            "source_kind_code": int(record.source_kind_code),
            "workspace_schema_version": WORKSPACE_SCHEMA_VERSION,
            "record_type": record.record_type,
            "record_id": record.record_id,
            "canonical_id": record.canonical_id,
            "source_id": record.source_id,
            "source_path_hash": record.source_path_hash,
            "source_path": record.source_path,
            "title": record.title,
            "vector_hash": record.vector_hash,
            "content_hash": record.content_hash,
            "metadata_hash": record.metadata_hash,
            "content": record.content,
            "tokens": int(record.tokens),
        },
    )


def _batches(items: Iterable[Any], batch_size: int) -> Iterable[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _hit_from_doc(doc: Doc) -> ZvecHit:
    return ZvecHit(doc_id=str(doc.id), score=float(doc.score), fields=dict(doc.fields))


def _embedding_vector(doc: Doc) -> list[float] | None:
    vectors = getattr(doc, "vectors", None)
    if isinstance(vectors, dict) and isinstance(vectors.get("embedding"), list):
        return vectors["embedding"]
    vector = getattr(doc, "vector", None)
    if callable(vector):
        value = vector("embedding")
        if isinstance(value, list):
            return value
    return None


def build_schema(embedding_dim: int) -> CollectionSchema:
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")

    fields = [
        FieldSchema(
            "record_type_code",
            DataType.INT32,
            nullable=False,
            index_param=InvertIndexParam(),
        ),
        FieldSchema(
            "section_kind_code",
            DataType.INT32,
            nullable=False,
            index_param=InvertIndexParam(),
        ),
        FieldSchema(
            "source_kind_code",
            DataType.INT32,
            nullable=False,
            index_param=InvertIndexParam(),
        ),
        FieldSchema(
            "workspace_schema_version",
            DataType.INT32,
            nullable=False,
            index_param=InvertIndexParam(),
        ),
        FieldSchema("record_type", DataType.STRING, nullable=False),
        FieldSchema("record_id", DataType.STRING, nullable=False),
        FieldSchema("canonical_id", DataType.STRING, nullable=False),
        FieldSchema("source_id", DataType.STRING, nullable=False),
        FieldSchema(
            "source_path_hash",
            DataType.STRING,
            nullable=False,
            index_param=InvertIndexParam(),
        ),
        FieldSchema("source_path", DataType.STRING, nullable=False),
        FieldSchema("title", DataType.STRING, nullable=False),
        FieldSchema("vector_hash", DataType.STRING, nullable=False),
        FieldSchema("content_hash", DataType.STRING, nullable=False),
        FieldSchema("metadata_hash", DataType.STRING, nullable=False),
        FieldSchema(
            "content",
            DataType.STRING,
            nullable=False,
            index_param=FtsIndexParam(tokenizer_name="standard", filters=["lowercase"]),
        ),
        FieldSchema("tokens", DataType.INT32, nullable=False),
    ]
    return CollectionSchema(
        name=COLLECTION_NAME,
        fields=fields,
        vectors=[
            VectorSchema(
                "embedding",
                DataType.VECTOR_FP32,
                dimension=embedding_dim,
                index_param=HnswIndexParam(
                    metric_type=MetricType.COSINE,
                    m=32,
                    ef_construction=400,
                ),
            )
        ],
    )


def create_workspace_collection(path: Path, embedding_dim: int) -> ZvecWorkspace:
    collection = create_and_open(
        path=str(path),
        schema=build_schema(embedding_dim),
        option=CollectionOption(read_only=False, enable_mmap=True),
    )
    return ZvecWorkspace(collection=collection)


def open_workspace_collection(path: Path, read_only: bool = True) -> ZvecWorkspace:
    collection = zvec_open(
        str(path),
        option=CollectionOption(read_only=read_only, enable_mmap=True),
    )
    return ZvecWorkspace(collection=collection)
