from __future__ import annotations

import importlib
import sys
import tomllib
import types
from pathlib import Path

import pytest


def _install_fake_zvec(monkeypatch):
    fake = types.ModuleType("zvec")

    class DataType:
        INT32 = "INT32"
        STRING = "STRING"
        VECTOR_FP32 = "VECTOR_FP32"

    class MetricType:
        COSINE = "COSINE"

    class CollectionOption:
        def __init__(self, read_only: bool = False, enable_mmap: bool = True) -> None:
            self.read_only = read_only
            self.enable_mmap = enable_mmap

    class InvertIndexParam:
        pass

    class FtsIndexParam:
        def __init__(
            self,
            tokenizer_name: str = "standard",
            filters: list[str] | None = None,
            extra_params: str = "",
        ) -> None:
            self.tokenizer_name = tokenizer_name
            self.filters = filters
            self.extra_params = extra_params

    class HnswIndexParam:
        def __init__(
            self,
            metric_type=MetricType.COSINE,
            m: int = 50,
            ef_construction: int = 500,
        ) -> None:
            self.metric_type = metric_type
            self.m = m
            self.ef_construction = ef_construction

    class HnswQueryParam:
        def __init__(self, ef: int = 300) -> None:
            self.ef = ef

    class FtsQueryParam:
        def __init__(self, default_operator: str = "") -> None:
            self.default_operator = default_operator

    class Fts:
        def __init__(
            self,
            match_string: str | None = None,
            query_string: str | None = None,
        ) -> None:
            self.match_string = match_string
            self.query_string = query_string

    class Query:
        def __init__(self, field_name: str, vector=None, param=None, fts=None) -> None:
            self.field_name = field_name
            self.vector = vector
            self.param = param
            self.fts = fts

    class RrfReRanker:
        def __init__(self, rank_constant: int) -> None:
            self.rank_constant = rank_constant

    class FieldSchema:
        def __init__(
            self,
            name: str,
            data_type,
            nullable: bool = False,
            index_param=None,
        ) -> None:
            self.name = name
            self.data_type = data_type
            self.nullable = nullable
            self.index_param = index_param

    class VectorSchema:
        def __init__(
            self,
            name: str,
            data_type,
            dimension: int,
            index_param=None,
        ) -> None:
            self.name = name
            self.data_type = data_type
            self.dimension = dimension
            self.index_param = index_param

    class CollectionSchema:
        def __init__(self, name: str, fields=None, vectors=None) -> None:
            self.name = name
            self.fields = fields or []
            self.vectors = vectors or []

    fake.calls = []

    def create_and_open(*, path: str, schema, option):
        fake.calls.append(("create_and_open", path, schema, option))
        return {"kind": "created", "path": path}

    def zvec_open(path: str, option):
        fake.calls.append(("open", path, option))
        return {"kind": "opened", "path": path}

    class Doc:
        def __init__(self, id: str, score=None, vectors=None, fields=None) -> None:
            self.id = id
            self.score = score
            self.vectors = vectors or {}
            self.fields = fields or {}

    for name, value in {
        "CollectionOption": CollectionOption,
        "CollectionSchema": CollectionSchema,
        "DataType": DataType,
        "Doc": Doc,
        "FieldSchema": FieldSchema,
        "Fts": Fts,
        "FtsIndexParam": FtsIndexParam,
        "FtsQueryParam": FtsQueryParam,
        "HnswIndexParam": HnswIndexParam,
        "HnswQueryParam": HnswQueryParam,
        "InvertIndexParam": InvertIndexParam,
        "MetricType": MetricType,
        "Query": Query,
        "RrfReRanker": RrfReRanker,
        "VectorSchema": VectorSchema,
        "create_and_open": create_and_open,
        "open": zvec_open,
    }.items():
        setattr(fake, name, value)

    monkeypatch.setitem(sys.modules, "zvec", fake)
    monkeypatch.delitem(sys.modules, "llm_wiki_native.storage.zvec_workspace", raising=False)
    return fake


def test_build_schema_matches_plan_numeric_filters_and_hnsw(monkeypatch) -> None:
    fake_zvec = _install_fake_zvec(monkeypatch)
    module = importlib.import_module("llm_wiki_native.storage.zvec_workspace")

    schema = module.build_schema(embedding_dim=8)

    assert schema.name == "llm_wiki_records_v1"
    fields = {field.name: field for field in schema.fields}
    assert fields["record_type_code"].data_type == fake_zvec.DataType.INT32
    assert isinstance(fields["record_type_code"].index_param, fake_zvec.InvertIndexParam)
    assert fields["section_kind_code"].data_type == fake_zvec.DataType.INT32
    assert isinstance(fields["section_kind_code"].index_param, fake_zvec.InvertIndexParam)
    assert fields["source_kind_code"].data_type == fake_zvec.DataType.INT32
    assert fields["workspace_schema_version"].data_type == fake_zvec.DataType.INT32
    assert fields["record_type"].index_param is None
    assert isinstance(fields["source_path_hash"].index_param, fake_zvec.InvertIndexParam)
    assert isinstance(fields["content"].index_param, fake_zvec.FtsIndexParam)
    assert fields["content"].index_param.tokenizer_name == "standard"
    assert fields["content"].index_param.filters == ["lowercase"]

    assert len(schema.vectors) == 1
    vector = schema.vectors[0]
    assert vector.name == "embedding"
    assert vector.data_type == fake_zvec.DataType.VECTOR_FP32
    assert vector.dimension == 8
    assert isinstance(vector.index_param, fake_zvec.HnswIndexParam)
    assert vector.index_param.metric_type == fake_zvec.MetricType.COSINE
    assert vector.index_param.m == 32
    assert vector.index_param.ef_construction == 400


def test_pyproject_declares_zvec_phase1_dependency() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "llm-wiki-native" / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert "zvec>=0.5.1,<0.6" in data["project"]["dependencies"]


def test_doc_ids_and_numeric_codes_are_strict_plan_mappings(monkeypatch) -> None:
    _install_fake_zvec(monkeypatch)
    module = importlib.import_module("llm_wiki_native.storage.zvec_workspace")

    assert module.zvec_doc_id("chunk", "chunk-1") == "chunk__Y2h1bmstMQ"
    assert module.zvec_doc_id("entity", "Evidence") == "entity__RXZpZGVuY2U"
    assert module.zvec_doc_id("relationship", "rel-key") == "relationship__cmVsLWtleQ"
    assert module.zvec_doc_id("section", "sec-1") == "section__c2VjLTE"

    assert module.record_type_code("chunk") == 1
    assert module.record_type_code("entity") == 2
    assert module.record_type_code("relationship") == 3
    assert module.record_type_code("section") == 4
    assert module.section_kind_code(None) == 0
    assert module.section_kind_code("") == 0
    assert module.section_kind_code("summary") == 1
    assert module.section_kind_code("questions") == 8
    assert module.section_kind_code("other") == 99

    with pytest.raises(ValueError, match="unknown record_type"):
        module.zvec_doc_id("note", "x")
    with pytest.raises(ValueError, match="record_id must not be empty"):
        module.zvec_doc_id("chunk", "")
    with pytest.raises(ValueError, match="unknown record_type"):
        module.record_type_code("note")
    with pytest.raises(ValueError, match="unknown section_kind"):
        module.section_kind_code("appendix")


def test_zvec_record_to_doc_maps_all_plan_fields(monkeypatch) -> None:
    fake_zvec = _install_fake_zvec(monkeypatch)
    module = importlib.import_module("llm_wiki_native.storage.zvec_workspace")

    record = module.ZvecRecord(
        record_type="section",
        record_id="sec-1",
        canonical_id="canonical:paper",
        source_id="source:paper",
        source_kind_code=2,
        source_path_hash="sha256:path",
        source_path="raw/paper.md",
        title="Paper title",
        vector_hash="sha256:vector",
        content_hash="sha256:content",
        metadata_hash="sha256:metadata",
        content="dense retrievable text",
        tokens=17,
        embedding=[0.1, 0.2],
        section_kind="summary",
    )

    doc = module.zvec_doc_from_record(record)

    assert isinstance(doc, fake_zvec.Doc)
    assert doc.id == "section__c2VjLTE"
    assert doc.vectors == {"embedding": [0.1, 0.2]}
    assert doc.fields == {
        "record_type_code": 4,
        "section_kind_code": 1,
        "source_kind_code": 2,
        "workspace_schema_version": 1,
        "record_type": "section",
        "record_id": "sec-1",
        "canonical_id": "canonical:paper",
        "source_id": "source:paper",
        "source_path_hash": "sha256:path",
        "source_path": "raw/paper.md",
        "title": "Paper title",
        "vector_hash": "sha256:vector",
        "content_hash": "sha256:content",
        "metadata_hash": "sha256:metadata",
        "content": "dense retrievable text",
        "tokens": 17,
    }

    with pytest.raises(ValueError, match="embedding must not be empty"):
        module.zvec_doc_from_record(
            module.ZvecRecord(
                record_type="chunk",
                record_id="chunk-1",
                canonical_id="chunk-1",
                source_id="source",
                source_kind_code=1,
                source_path_hash="path",
                source_path="source.md",
                title="",
                vector_hash="vector",
                content_hash="content",
                metadata_hash="metadata",
                content="text",
                tokens=1,
                embedding=[],
            )
        )


def test_real_zvec_accepts_encoded_doc_id_with_punctuation(tmp_path) -> None:
    sys.modules.pop("llm_wiki_native.storage.zvec_workspace", None)
    module = importlib.import_module("llm_wiki_native.storage.zvec_workspace")

    record = module.ZvecRecord(
        record_type="section",
        record_id="raw/section:doc-a:method",
        canonical_id="raw/section:doc-a:method",
        source_id="doc:a",
        source_kind_code=2,
        source_path_hash="path-hash",
        source_path="raw/doc-a.md",
        title="Method",
        vector_hash="vector-hash",
        content_hash="content-hash",
        metadata_hash="metadata-hash",
        content="method text",
        tokens=2,
        embedding=[1.0, 0.0],
        section_kind="methodology",
    )
    doc = module.zvec_doc_from_record(record)

    assert ":" not in doc.id
    assert "/" not in doc.id
    workspace = module.create_workspace_collection(tmp_path / "zvec", embedding_dim=2)
    stats = workspace.bulk_insert([record], batch_size=1)
    fetched = workspace.fetch([doc.id])

    assert stats == module.InsertStats(attempted=1, inserted=1, failed=0)
    assert fetched[doc.id].fields["record_id"] == "raw/section:doc-a:method"
    workspace.flush_optimize_close()
    hits = workspace.query_vector([1.0, 0.0], top_k=1, filter_expr="record_type_code in (4)")

    assert hits[0].doc_id == doc.id
    assert hits[0].fields["record_id"] == "raw/section:doc-a:method"


def test_real_zvec_accepts_long_record_id_with_short_stable_doc_id(tmp_path) -> None:
    sys.modules.pop("llm_wiki_native.storage.zvec_workspace", None)
    module = importlib.import_module("llm_wiki_native.storage.zvec_workspace")

    record = module.ZvecRecord(
        record_type="entity",
        record_id="compiled:comparison:adaptive-memory-retrieval-vs-persistent-knowledge-base",
        canonical_id="compiled:comparison:adaptive-memory-retrieval-vs-persistent-knowledge-base",
        source_id="compiled:comparison:adaptive-memory-retrieval-vs-persistent-knowledge-base",
        source_kind_code=1,
        source_path_hash="path-hash",
        source_path="adaptive-memory-retrieval-vs-persistent-knowledge-base.md",
        title="",
        vector_hash="vector-hash",
        content_hash="content-hash",
        metadata_hash="metadata-hash",
        content="compiled:comparison:adaptive-memory-retrieval-vs-persistent-knowledge-base\nAdaptive Memory Retrieval",
        tokens=2,
        embedding=[1.0, 0.0],
    )
    workspace = module.create_workspace_collection(tmp_path / "zvec", embedding_dim=2)
    doc_id = module.zvec_doc_id(record.record_type, record.record_id)

    stats = workspace.bulk_insert([record], batch_size=1)
    fetched = workspace.fetch([doc_id])

    assert len(doc_id) <= module.MAX_ZVEC_DOC_ID_LENGTH
    assert stats == module.InsertStats(attempted=1, inserted=1, failed=0)
    assert fetched[doc_id].fields["record_id"] == record.record_id
    assert fetched[doc_id].fields["content"] == record.content


def test_bulk_insert_batches_docs_and_counts_statuses(monkeypatch) -> None:
    _install_fake_zvec(monkeypatch)
    module = importlib.import_module("llm_wiki_native.storage.zvec_workspace")

    class Status:
        def __init__(self, ok: bool) -> None:
            self._ok = ok

        def ok(self) -> bool:
            return self._ok

    class Collection:
        def __init__(self) -> None:
            self.insert_calls = []

        def insert(self, docs):
            self.insert_calls.append(docs)
            return [Status(not doc.fields["record_id"].endswith("3")) for doc in docs]

    records = [
        module.ZvecRecord(
            record_type="chunk",
            record_id=f"chunk-{index}",
            canonical_id=f"chunk-{index}",
            source_id="source",
            source_kind_code=1,
            source_path_hash="path",
            source_path="source.md",
            title="",
            vector_hash=f"vector-{index}",
            content_hash=f"content-{index}",
            metadata_hash=f"metadata-{index}",
            content=f"text {index}",
            tokens=index,
            embedding=[float(index), 0.0],
        )
        for index in range(1, 4)
    ]
    workspace = module.ZvecWorkspace(collection=Collection())

    stats = workspace.bulk_insert(records, batch_size=2)

    assert stats == module.InsertStats(attempted=3, inserted=2, failed=1)
    assert [len(call) for call in workspace.collection.insert_calls] == [2, 1]
    assert [doc.fields["record_id"] for call in workspace.collection.insert_calls for doc in call] == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]


def test_bulk_insert_default_batch_size_respects_zvec_write_limit(monkeypatch) -> None:
    _install_fake_zvec(monkeypatch)
    module = importlib.import_module("llm_wiki_native.storage.zvec_workspace")

    class Status:
        def ok(self) -> bool:
            return True

    class Collection:
        def __init__(self) -> None:
            self.insert_call_sizes = []

        def insert(self, docs):
            if len(docs) > 1024:
                raise ValueError(f"Too many docs: {len(docs)} exceeds max write batch size of 1024")
            self.insert_call_sizes.append(len(docs))
            return [Status() for _doc in docs]

    records = [
        module.ZvecRecord(
            record_type="chunk",
            record_id=f"chunk-{index}",
            canonical_id=f"chunk-{index}",
            source_id="source",
            source_kind_code=1,
            source_path_hash="path",
            source_path="source.md",
            title="",
            vector_hash=f"vector-{index}",
            content_hash=f"content-{index}",
            metadata_hash=f"metadata-{index}",
            content=f"text {index}",
            tokens=index,
            embedding=[float(index), 0.0],
        )
        for index in range(1025)
    ]
    workspace = module.ZvecWorkspace(collection=Collection())

    stats = workspace.bulk_insert(records)

    assert stats == module.InsertStats(attempted=1025, inserted=1025, failed=0)
    assert workspace.collection.insert_call_sizes == [1024, 1]


def test_flush_optimize_close_delegates_without_destroy(monkeypatch) -> None:
    _install_fake_zvec(monkeypatch)
    module = importlib.import_module("llm_wiki_native.storage.zvec_workspace")

    class Collection:
        def __init__(self) -> None:
            self.calls = []

        def flush(self) -> None:
            self.calls.append("flush")

        def optimize(self) -> None:
            self.calls.append("optimize")

        def close(self) -> None:
            self.calls.append("close")

        def destroy(self) -> None:  # pragma: no cover - must not be called
            raise AssertionError("destroy must not be called")

    collection = Collection()
    module.ZvecWorkspace(collection=collection).flush_optimize_close()

    assert collection.calls == ["flush", "optimize", "close"]


def test_self_nearest_smoke_fetches_vectors_and_reports_failures(monkeypatch) -> None:
    _install_fake_zvec(monkeypatch)
    module = importlib.import_module("llm_wiki_native.storage.zvec_workspace")

    class FetchedDoc:
        def __init__(self, doc_id: str, vector: list[float]) -> None:
            self.id = doc_id
            self.vectors = {"embedding": vector}

    class QueryDoc:
        def __init__(self, doc_id: str) -> None:
            self.id = doc_id
            self.score = 1.0
            self.fields = {}

    class Collection:
        def __init__(self) -> None:
            self.fetch_calls = []

        def fetch(self, doc_ids, *, include_vector: bool):
            self.fetch_calls.append((doc_ids, include_vector))
            return {
                "doc:1": FetchedDoc("doc:1", [1.0, 0.0]),
                "doc:2": FetchedDoc("doc:2", [0.0, 1.0]),
            }

        def query(self, **kwargs):
            vector = kwargs["queries"].vector
            return [QueryDoc("doc:1" if vector == [1.0, 0.0] else "wrong:doc")]

    collection = Collection()
    result = module.ZvecWorkspace(collection=collection).self_nearest_smoke(["doc:1", "doc:2"])

    assert collection.fetch_calls == [(["doc:1", "doc:2"], True)]
    assert result == module.SmokeResult(
        checked=2,
        passed=1,
        failures=["doc:2 nearest wrong:doc"],
    )


def test_create_and_open_workspace_collections_use_mmap_options(
    monkeypatch,
    tmp_path,
) -> None:
    fake_zvec = _install_fake_zvec(monkeypatch)
    module = importlib.import_module("llm_wiki_native.storage.zvec_workspace")

    created = module.create_workspace_collection(
        tmp_path / "workspace",
        embedding_dim=8,
    )
    opened = module.open_workspace_collection(tmp_path / "workspace", read_only=True)

    assert created.collection == {
        "kind": "created",
        "path": str(tmp_path / "workspace"),
    }
    assert opened.collection == {
        "kind": "opened",
        "path": str(tmp_path / "workspace"),
    }
    create_call = fake_zvec.calls[0]
    assert create_call[0] == "create_and_open"
    assert create_call[1] == str(tmp_path / "workspace")
    assert create_call[2].name == "llm_wiki_records_v1"
    assert create_call[3].read_only is False
    assert create_call[3].enable_mmap is True
    open_call = fake_zvec.calls[1]
    assert open_call[0] == "open"
    assert open_call[1] == str(tmp_path / "workspace")
    assert open_call[2].read_only is True
    assert open_call[2].enable_mmap is True


def test_workspace_fetch_and_stats_are_thin_collection_calls(monkeypatch) -> None:
    _install_fake_zvec(monkeypatch)
    module = importlib.import_module("llm_wiki_native.storage.zvec_workspace")

    class Stats:
        doc_count = 42

    class Collection:
        stats = Stats()

        def __init__(self) -> None:
            self.fetch_calls = []
            self.doc = object()

        def fetch(self, doc_ids, *, include_vector: bool):
            self.fetch_calls.append((doc_ids, include_vector))
            return {"doc:1": self.doc}

    collection = Collection()
    workspace = module.ZvecWorkspace(collection=collection)

    assert workspace.fetch(["doc:1"]) == {"doc:1": collection.doc}
    assert collection.fetch_calls == [(["doc:1"], False)]
    assert workspace.stats() == {"doc_count": 42}


def test_query_vector_uses_hnsw_param_filter_and_maps_hits(monkeypatch) -> None:
    fake_zvec = _install_fake_zvec(monkeypatch)
    module = importlib.import_module("llm_wiki_native.storage.zvec_workspace")

    class Doc:
        id = "doc:1"
        score = 0.75
        fields = {"record_id": "one"}

    class Collection:
        def __init__(self) -> None:
            self.query_calls = []

        def query(self, **kwargs):
            self.query_calls.append(kwargs)
            return [Doc()]

    collection = Collection()
    workspace = module.ZvecWorkspace(collection=collection)

    hits = workspace.query_vector(
        [1.0, 0.0],
        top_k=3,
        filter_expr="record_type_code in (1)",
    )

    assert hits == [
        module.ZvecHit(doc_id="doc:1", score=0.75, fields={"record_id": "one"})
    ]
    call = collection.query_calls[0]
    assert call["topk"] == 3
    assert call["filter"] == "record_type_code in (1)"
    assert call["include_vector"] is False
    query = call["queries"]
    assert isinstance(query, fake_zvec.Query)
    assert query.field_name == "embedding"
    assert query.vector == [1.0, 0.0]
    assert isinstance(query.param, fake_zvec.HnswQueryParam)
    assert query.param.ef == 200


def test_query_mix_uses_vector_fts_and_rrf(monkeypatch) -> None:
    fake_zvec = _install_fake_zvec(monkeypatch)
    module = importlib.import_module("llm_wiki_native.storage.zvec_workspace")

    class Collection:
        def __init__(self) -> None:
            self.query_calls = []

        def query(self, **kwargs):
            self.query_calls.append(kwargs)
            return []

    collection = Collection()
    workspace = module.ZvecWorkspace(collection=collection)

    assert (
        workspace.query_mix("alpha beta", [1.0, 0.0], top_k=5, filter_expr=None)
        == []
    )

    call = collection.query_calls[0]
    assert call["topk"] == 5
    assert call["filter"] is None
    assert call["include_vector"] is False
    assert isinstance(call["reranker"], fake_zvec.RrfReRanker)
    assert call["reranker"].rank_constant == 60
    vector_query, fts_query = call["queries"]
    assert vector_query.field_name == "embedding"
    assert isinstance(vector_query.param, fake_zvec.HnswQueryParam)
    assert vector_query.param.ef == 200
    assert fts_query.field_name == "content"
    assert fts_query.fts.match_string == "alpha beta"
    assert isinstance(fts_query.param, fake_zvec.FtsQueryParam)
    assert fts_query.param.default_operator == "AND"
