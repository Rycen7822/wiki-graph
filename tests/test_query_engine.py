import pytest

from llm_wiki_native.retrieval.query_engine import NativeQueryEngine, _record_identity_from_hit
from llm_wiki_native.storage.sqlite_workspace import SQLiteWorkspace
from support import native_record


class _ZvecHit:
    def __init__(self, record_type: str, record_id: str, score: float = 1.0) -> None:
        self.doc_id = f"{record_type}:{record_id}"
        self.score = score
        self.fields = {"record_type": record_type, "record_id": record_id}


class _ZvecWorkspace:
    def __init__(self, hits: list[_ZvecHit] | None = None) -> None:
        self.hits = list(hits or [])
        self.calls = []

    def query_mix(self, query: str, query_vector: list[float], top_k: int, filter_expr: str | None):
        self.calls.append(("mix", query, query_vector, top_k, filter_expr))
        return self.hits

    def query_vector(self, query_vector: list[float], top_k: int, filter_expr: str | None):
        self.calls.append(("vector", query_vector, top_k, filter_expr))
        return self.hits


def test_query_engine_requires_zvec_workspace(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")

    with pytest.raises(ValueError, match="zvec workspace"):
        NativeQueryEngine(db, zvec_workspace=None)


def test_data_only_query_engine_returns_ranked_hits_with_trace(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    db.put_record(native_record("native-test", "entity", "doc:a", "Alpha"))
    db.put_record(native_record("native-test", "entity", "doc:b", "Beta"))
    db.put_vector("native-test", "entity", "doc:a", "doc:a:vector", [1.0, 0.0])
    db.put_vector("native-test", "entity", "doc:b", "doc:b:vector", [0.0, 1.0])
    db.put_edge("native-test", "relationship", "doc:a", "tag:x", 0.8, {"kind": "related"})
    db.mark_audited("native-test", {"chunks": 0, "entities": 2, "relationships": 0, "sections": 0}, require_vectors=True)
    zvec = _ZvecWorkspace([_ZvecHit("entity", "doc:a")])
    engine = NativeQueryEngine(db, zvec_workspace=zvec)

    result = engine.query("native-test", "alpha query", [1.0, 0.0], mode="mix", top_k=1, record_types=("entity",), neighbor_limit=1)

    assert zvec.calls == [("mix", "alpha query", [1.0, 0.0], 1, "record_type_code in (2)")]

    assert result["hits"][0]["record_id"] == "doc:a"
    assert result["hits"][0]["record"]["vector_text"] == "Alpha"
    assert result["hits"][0]["neighbors"][0]["neighbor_id"] == "tag:x"
    assert result["trace"]["mode"] == "mix"
    assert result["trace"]["query"] == "alpha query"
    assert result["trace"]["vector_hit_count"] == 1


def test_query_engine_routes_mix_to_zvec_hybrid_when_workspace_is_supplied() -> None:
    class Hit:
        doc_id = "chunk:chunk-a"
        score = 0.75
        fields = {"record_type": "chunk", "record_id": "chunk-a"}

    class DB:
        def get_workspace_status(self, workspace_id: str) -> str:
            return "audited"

        def get_record(self, workspace_id: str, record_type: str, record_id: str):
            return {"record_type": record_type, "record_id": record_id, "vector_text": "Alpha"}

        def neighbors(self, workspace_id: str, record_id: str, *, limit: int):
            return [{"neighbor_id": "doc:b", "limit": limit}]

    class ZvecWorkspace:
        def __init__(self) -> None:
            self.calls = []

        def query_mix(self, query: str, query_vector: list[float], top_k: int, filter_expr: str | None):
            self.calls.append(("mix", query, query_vector, top_k, filter_expr))
            return [Hit()]

    zvec = ZvecWorkspace()
    engine = NativeQueryEngine(DB(), zvec_workspace=zvec)

    result = engine.query(
        "native-test",
        "alpha query",
        [1.0, 0.0],
        mode="mix",
        top_k=3,
        record_types=("chunk", "entity"),
        neighbor_limit=2,
    )

    assert zvec.calls == [
        ("mix", "alpha query", [1.0, 0.0], 3, "record_type_code in (1,2)")
    ]
    assert result["hits"][0]["doc_id"] == "chunk:chunk-a"
    assert result["hits"][0]["record"]["vector_text"] == "Alpha"
    assert result["hits"][0]["neighbors"] == [{"neighbor_id": "doc:b", "limit": 2}]
    assert result["trace"]["retrieval_backend"] == "zvec"


def test_query_engine_routes_naive_and_bypass_with_zvec_workspace() -> None:
    class Hit:
        doc_id = "section:sec-a"
        score = 0.5
        fields = {"record_type": "section", "record_id": "sec-a"}

    class DB:
        def get_workspace_status(self, workspace_id: str) -> str:
            return "audited"

        def get_record(self, workspace_id: str, record_type: str, record_id: str):
            return {"record_type": record_type, "record_id": record_id}

        def neighbors(self, workspace_id: str, record_id: str, *, limit: int):
            return []

    class ZvecWorkspace:
        def __init__(self) -> None:
            self.calls = []

        def query_vector(self, query_vector: list[float], top_k: int, filter_expr: str | None):
            self.calls.append(("vector", query_vector, top_k, filter_expr))
            return [Hit()]

        def query_mix(self, *args, **kwargs):
            raise AssertionError("naive/bypass must not call query_mix")

    zvec = ZvecWorkspace()
    engine = NativeQueryEngine(DB(), zvec_workspace=zvec)

    naive = engine.query(
        "native-test",
        "alpha query",
        [1.0, 0.0],
        mode="naive",
        top_k=5,
        record_types=("entity",),
    )
    bypass = engine.query(
        "native-test",
        "alpha query",
        [1.0, 0.0],
        mode="bypass",
        top_k=5,
    )

    assert zvec.calls == [("vector", [1.0, 0.0], 5, "record_type_code in (2)")]
    assert naive["hits"][0]["doc_id"] == "section:sec-a"
    assert naive["trace"]["retrieval_backend"] == "zvec"
    assert bypass["hits"] == []
    assert bypass["trace"]["retrieval_backend"] == "bypass"
    assert bypass["trace"]["vector_hit_count"] == 0


def test_query_engine_routes_section_kind_as_numeric_zvec_filter() -> None:
    class DB:
        def get_workspace_status(self, workspace_id: str) -> str:
            return "audited"

    class ZvecWorkspace:
        def __init__(self) -> None:
            self.calls = []

        def query_mix(self, query: str, query_vector: list[float], top_k: int, filter_expr: str | None):
            self.calls.append(("mix", filter_expr))
            return []

    zvec = ZvecWorkspace()
    engine = NativeQueryEngine(DB(), zvec_workspace=zvec)

    result = engine.query(
        "native-test",
        "method query",
        [1.0, 0.0],
        mode="mix",
        top_k=3,
        record_types=("chunk", "section"),
        section_kind="methodology",
    )

    assert zvec.calls == [("mix", "record_type_code in (4) and section_kind_code in (4)")]
    assert result["trace"]["section_kind"] == "methodology"


def test_data_only_query_engine_rejects_building_workspace(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    db.put_record(native_record("native-test", "entity", "doc:a", "Alpha"))
    db.put_vector("native-test", "entity", "doc:a", "doc:a:vector", [1.0, 0.0])
    engine = NativeQueryEngine(db, zvec_workspace=_ZvecWorkspace())

    with pytest.raises(ValueError, match="workspace must be audited before query"):
        engine.query("native-test", "alpha query", [1.0, 0.0], mode="mix", record_types=("entity",))


def test_data_only_query_engine_rejects_unknown_mode(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    engine = NativeQueryEngine(db, zvec_workspace=_ZvecWorkspace())

    with pytest.raises(ValueError, match="mode"):
        engine.query("native-test", "alpha query", [1.0], mode="unsupported")


def test_data_only_query_engine_rejects_unknown_record_type(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    engine = NativeQueryEngine(db, zvec_workspace=_ZvecWorkspace())

    with pytest.raises(ValueError, match="record_type"):
        engine.query("native-test", "alpha query", [1.0], mode="mix", record_types=("unknown",))


def test_record_identity_requires_zvec_hit_fields() -> None:
    assert _record_identity_from_hit(
        "chunk__Y2h1bmstYQ",
        {"record_type": "chunk", "record_id": "chunk-a"},
    ) == ("chunk", "chunk-a")

    for doc_id in ("chunk__Y2h1bmstYQ", "chunk:legacy-id"):
        with pytest.raises(ValueError, match="zvec hit missing record_type/record_id fields"):
            _record_identity_from_hit(doc_id, {})

    with pytest.raises(ValueError, match="zvec hit missing record_type/record_id fields"):
        _record_identity_from_hit("chunk__Y2h1bmstYQ", {"record_type": "chunk"})
