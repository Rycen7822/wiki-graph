import re
from typing import Any

import pytest

from llm_wiki_native.contracts import RECORD_TYPE_CODES, SECTION_KIND_CODES
from llm_wiki_native.retrieval.context import assemble_context
from llm_wiki_native.retrieval.query_engine import NativeQueryEngine, _record_identity_from_hit
from llm_wiki_native.storage.sqlite_workspace import SQLiteWorkspace
from support import native_record, put_span, traced_sqlite_connect


def _db(tmp_path, *, create: bool = True):
    workspace = SQLiteWorkspace(tmp_path / "native.sqlite")
    if create:
        workspace.create_workspace("native-test", "manifest-hash")
    return workspace


def _audit(db, *, require_vectors: bool = False, **counts):
    db.mark_audited(
        "native-test",
        {"chunks": 0, "entities": 0, "relationships": 0, "sections": 0, **counts},
        require_vectors=require_vectors,
    )


class _ZvecHit:
    def __init__(
        self,
        record_type: str,
        record_id: str,
        score: float = 1.0,
        *,
        source_path: str | None = None,
        source_id: str | None = None,
        source_kind_code: int | None = None,
        section_kind_code: int | None = None,
        content: str | None = None,
    ) -> None:
        self.doc_id = f"{record_type}:{record_id}"
        self.score = score
        self.fields = {
            "record_type": record_type,
            "record_id": record_id,
            "source_path": source_path or f"raw/{record_id}.md",
            "source_id": source_id or f"source:{record_id}",
            "source_kind_code": (
                source_kind_code
                if source_kind_code is not None
                else (2 if record_type == "section" else 1)
            ),
            "section_kind_code": (
                section_kind_code
                if section_kind_code is not None
                else (SECTION_KIND_CODES["methodology"] if record_type == "section" else 0)
            ),
            "content": content or f"Evidence for {record_id}",
            "content_hash": f"{record_id}:content",
        }


class _StubDb:
    def __init__(self, *, record_extra: dict[str, Any] | None = None, neighbors: list[dict[str, Any]] | None = None) -> None:
        self._record_extra = record_extra or {}
        self._neighbors = neighbors

    def get_workspace_metadata(self, workspace_id: str) -> dict:
        return {
            "workspace_id": workspace_id,
            "source_manifest_hash": "manifest-hash",
            "schema_version": 1,
            "status": "audited",
        }

    def get_record(self, workspace_id: str, record_type: str, record_id: str):
        return {"record_type": record_type, "record_id": record_id, **self._record_extra}

    def neighbors(self, workspace_id: str, record_id: str, *, limit: int):
        if self._neighbors is None:
            return []
        return [{**row, "limit": limit} for row in self._neighbors]


class _ZvecWorkspace:
    def __init__(
        self,
        hits: list[_ZvecHit] | None = None,
        *,
        apply_filter: bool = True,
        reject_mix: bool = False,
    ) -> None:
        self.hits = list(hits or [])
        self.calls = []
        self.apply_filter = apply_filter
        self.reject_mix = reject_mix

    def query_mix(self, query: str, query_vector: list[float], top_k: int, filter_expr: str | None):
        if self.reject_mix:
            raise AssertionError("naive/bypass must not call query_mix")
        self.calls.append(("mix", query, query_vector, top_k, filter_expr))
        return self._filtered(filter_expr)[:top_k]

    def query_vector(self, query_vector: list[float], top_k: int, filter_expr: str | None):
        self.calls.append(("vector", query_vector, top_k, filter_expr))
        return self._filtered(filter_expr)[:top_k]

    def _filtered(self, filter_expr: str | None) -> list[_ZvecHit]:
        if not self.apply_filter or not filter_expr:
            return list(self.hits)
        type_match = re.search(r"record_type_code in \(([^)]+)\)", filter_expr)
        allowed_types = (
            {int(value.strip()) for value in type_match.group(1).split(",")}
            if type_match
            else set(RECORD_TYPE_CODES.values())
        )
        kind_match = re.search(r"section_kind_code in \(([^)]+)\)", filter_expr)
        allowed_kinds = (
            {int(value.strip()) for value in kind_match.group(1).split(",")}
            if kind_match
            else None
        )
        return [
            hit
            for hit in self.hits
            if RECORD_TYPE_CODES[hit.fields["record_type"]] in allowed_types
            and (
                allowed_kinds is None
                or int(hit.fields["section_kind_code"]) in allowed_kinds
            )
        ]


def test_query_engine_requires_zvec_workspace(tmp_path) -> None:
    db = _db(tmp_path, create=False)

    with pytest.raises(ValueError, match="zvec workspace"):
        NativeQueryEngine(db, zvec_workspace=None)


def test_data_only_query_engine_returns_ranked_hits_with_trace(tmp_path) -> None:
    db = _db(tmp_path)
    db.put_record(native_record("native-test", "entity", "doc:a", "Alpha"))
    db.put_record(native_record("native-test", "entity", "doc:b", "Beta"))
    db.put_vector("native-test", "entity", "doc:a", "doc:a:vector", [1.0, 0.0])
    db.put_vector("native-test", "entity", "doc:b", "doc:b:vector", [0.0, 1.0])
    db.put_edge("native-test", "relationship", "doc:a", "tag:x", 0.8, {"kind": "related"})
    _audit(db, entities=2, require_vectors=True)
    zvec = _ZvecWorkspace([_ZvecHit("entity", "doc:a")])
    engine = NativeQueryEngine(db, zvec_workspace=zvec)

    result = engine.query("native-test", "alpha query", [1.0, 0.0], mode="mix", top_k=1, record_types=("entity",), neighbor_limit=1)

    assert zvec.calls == [("mix", "alpha query", [1.0, 0.0], 40, "record_type_code in (2)")]

    assert result["hits"][0]["record_id"] == "doc:a"
    assert result["hits"][0]["record"]["vector_text"] == "Alpha"
    assert result["hits"][0]["neighbors"][0]["neighbor_id"] == "tag:x"
    assert result["trace"]["mode"] == "mix"
    assert result["trace"]["query"] == "alpha query"
    assert result["trace"]["vector_hit_count"] == 1


def test_mix_query_parity_covers_routes_scores_trace_neighbors_and_profiles(tmp_path) -> None:
    db = _db(tmp_path)
    db.put_record(native_record("native-test", "chunk", "chunk-weak", "Weak semantic chunk", source_path="concepts/weak.md"))
    db.put_vector("native-test", "chunk", "chunk-weak", "chunk-weak:vector", [1.0, 0.0])
    db.put_edge("native-test", "relationship", "chunk-weak", "tag:x", 0.5, {"kind": "related"})
    _audit(db, chunks=1, require_vectors=True)
    put_span(
        db,
        span_id="span:raw-map-row",
        source_path="_meta/raw-clip-map.md",
        source_id="meta:raw-clip-map",
        source_role="meta_map",
        span_kind="map.row",
        heading_path=["Raw Clip Map"],
        start_line=4,
        text="- raw/clip/2601/26010101_Foo-Paper.md :: MapOnlyNeedle",
        metadata={"map": "raw-clip"},
    )
    zvec = _ZvecWorkspace([_ZvecHit("chunk", "chunk-weak", score=100.0)])
    engine = NativeQueryEngine(db, zvec_workspace=zvec)

    result = engine.query(
        "native-test",
        "MapOnlyNeedle raw map",
        [1.0, 0.0],
        mode="mix",
        top_k=2,
        record_types=("chunk",),
        neighbor_limit=1,
    )
    compact = assemble_context(result, max_chars_per_block=16, response_profile="compact")
    standard = assemble_context(result, max_chars_per_block=16, response_profile="standard")
    debug = assemble_context(result, max_chars_per_block=16, response_profile="debug")

    assert [hit["record_id"] for hit in result["hits"]] == ["chunk-weak", "span:raw-map-row"]
    assert result["hits"][0]["ranking_contract"] == "relevance-v1"
    assert result["hits"][1]["record"]["source_path"] == "_meta/raw-clip-map.md"
    assert result["hits"][0]["routes"] == ["zvec"]
    assert result["hits"][1]["routes"] == ["lexical_fts"]
    assert result["hits"][0]["neighbors"] == [
        {
            "edge_type": "relationship",
            "neighbor_id": "tag:x",
            "src_id": "chunk-weak",
            "tgt_id": "tag:x",
            "weight": 0.5,
            "payload": {"kind": "related"},
        }
    ]
    assert result["trace"]["route_counts"] == {"zvec": 1, "lexical": 1}
    assert result["trace"]["retrieval_goal"] == "focused"
    assert result["trace"]["db_record_calls"] == 1
    assert result["trace"]["db_neighbor_calls"] == 1
    assert set(result["trace"]["timings_ms"]) == {"route", "planner", "hydrate"}
    assert compact["coverage_plan"]["by_source_role"]["meta_map"] == ["_meta/raw-clip-map.md"]
    assert "neighbors" not in compact["context_blocks"][0]
    assert standard["context_blocks"][0]["routes"] == ["zvec"]
    assert debug["retrieval_debug"]["hits"][1]["score_breakdown"]["source_role"] == 1.0


def test_default_query_runs_navigation_and_section_routes(tmp_path) -> None:
    db = _db(tmp_path)
    db.put_record(native_record("native-test", "entity", "entity-a", "Entity evidence"))
    db.put_record(
        native_record(
            "native-test",
            "section",
            "section-a",
            "Section evidence",
            payload={"section_kind": "methodology"},
        )
    )
    _audit(db, entities=1, sections=1)
    zvec = _ZvecWorkspace(
        [_ZvecHit("entity", "entity-a"), _ZvecHit("section", "section-a")]
    )

    result = NativeQueryEngine(db, zvec_workspace=zvec).query(
        "native-test",
        "evidence",
        [1.0, 0.0],
        mode="mix",
        top_k=2,
        neighbor_limit=0,
    )

    assert zvec.calls == [
        ("mix", "evidence", [1.0, 0.0], 40, "record_type_code in (1,2,3)"),
        ("mix", "evidence", [1.0, 0.0], 40, "record_type_code in (4)"),
    ]
    assert [hit["record_id"] for hit in result["hits"]] == ["section-a", "entity-a"]
    assert result["trace"]["retrieval_goal"] == "focused"
    assert result["trace"]["record_types"] == ["entity", "relationship", "chunk", "section"]
    assert result["trace"]["family_candidate_counts"] == {
        "zvec_navigation": 1,
        "zvec_section": 1,
        "lexical": 0,
    }
    assert result["trace"]["workspace_id"] == "native-test"
    assert result["trace"]["workspace_schema_version"] == 1
    assert result["trace"]["selected_block_count"] == 2
    assert len(result["trace"]["candidate_cards"]) <= result["trace"]["candidate_card_limit"]
    assert all(
        not ({"content", "text", "vector_text"} & card.keys())
        for card in result["trace"]["candidate_cards"]
    )


def test_explicit_section_kind_runs_only_filtered_section_route_without_lexical(
    tmp_path,
    monkeypatch,
) -> None:
    db = _db(tmp_path)
    db.put_record(
        native_record(
            "native-test",
            "section",
            "section-results",
            "Result evidence",
            payload={"section_kind": "results"},
        )
    )
    _audit(db, sections=1)
    zvec = _ZvecWorkspace(
        [
            _ZvecHit(
                "section",
                "section-results",
                section_kind_code=SECTION_KIND_CODES["results"],
            )
        ]
    )
    lexical_calls = 0
    original_lexical = db.query_lexical_spans

    def count_lexical(*args, **kwargs):
        nonlocal lexical_calls
        lexical_calls += 1
        return original_lexical(*args, **kwargs)

    monkeypatch.setattr(db, "query_lexical_spans", count_lexical)
    result = NativeQueryEngine(db, zvec_workspace=zvec).query(
        "native-test",
        "results",
        [1.0, 0.0],
        mode="mix",
        top_k=1,
        record_types=("entity",),
        section_kind="results",
        neighbor_limit=0,
        retrieval_goal="coverage",
    )

    assert zvec.calls == [
        (
            "mix",
            "results",
            [1.0, 0.0],
            40,
            f"record_type_code in (4) and section_kind_code in ({SECTION_KIND_CODES['results']})",
        )
    ]
    assert lexical_calls == 0
    assert [hit["record_id"] for hit in result["hits"]] == ["section-results"]
    assert result["hits"][0]["relevance_score_breakdown"]["evidence_quality"] == 1.0
    selected_decision = next(
        decision
        for decision in result["trace"]["planner_decisions"]
        if decision["decision"] == "selected"
    )
    assert selected_decision["evidence_group"] == "section:results"
    assert result["trace"]["retrieval_goal"] == "coverage"


def _oversample_workspace(tmp_path, monkeypatch):
    db = _db(tmp_path)
    db.put_record(native_record("native-test", "chunk", "chunk-000", "alpha evidence"))
    db.put_edge("native-test", "supports", "chunk-000", "neighbor-001", 1.0, {})
    _audit(db, chunks=1)
    zvec = _ZvecWorkspace(
        [
            _ZvecHit(
                "chunk",
                f"chunk-{index:03d}",
                score=float(100 - index),
                content="alpha evidence",
            )
            for index in range(40)
        ]
    )
    counters = {"connections": 0, "gets": [], "neighbors": [], "selects": []}
    original_connect = db._connect
    original_get = db.get_record
    original_neighbors = db.neighbors

    def tracked_connect():
        counters["connections"] += 1
        connection = original_connect()
        connection.set_trace_callback(
            lambda statement: counters["selects"].append(statement)
            if statement.lstrip().upper().startswith("SELECT")
            else None
        )
        return connection

    def count_get(workspace_id: str, record_type: str, record_id: str):
        counters["gets"].append((record_type, record_id))
        return original_get(workspace_id, record_type, record_id)

    def count_neighbors(workspace_id: str, record_id: str, *, limit: int):
        counters["neighbors"].append(record_id)
        return original_neighbors(workspace_id, record_id, limit=limit)

    monkeypatch.setattr(db, "_connect", tracked_connect)
    monkeypatch.setattr(db, "get_record", count_get)
    monkeypatch.setattr(db, "neighbors", count_neighbors)
    return db, zvec, counters


def _reset_oversample_counters(counters: dict) -> None:
    counters["connections"] = 0
    counters["gets"].clear()
    counters["neighbors"].clear()
    counters["selects"].clear()


@pytest.mark.parametrize(("neighbor_limit", "cost_cap"), [(0, 3), (2, 5)])
def test_candidate_oversampling_does_not_expand_selected_cost(
    tmp_path,
    monkeypatch,
    neighbor_limit,
    cost_cap,
) -> None:
    db, zvec, counters = _oversample_workspace(tmp_path, monkeypatch)
    large_hits = list(zvec.hits)
    result = NativeQueryEngine(db, zvec_workspace=zvec).query(
        "native-test",
        "alpha",
        [1.0, 0.0],
        mode="naive",
        top_k=1,
        record_types=("chunk",),
        neighbor_limit=neighbor_limit,
    )
    large_cost = (counters["connections"], len(counters["selects"]))
    if neighbor_limit == 0:
        assert zvec.calls[0][2] == 40
        assert [hit["record_id"] for hit in result["hits"]] == ["chunk-000"]
        assert counters["gets"] == [("chunk", "chunk-000")]
        assert counters["neighbors"] == []
        assert result["trace"]["db_record_calls"] == 1
        assert result["trace"]["db_neighbor_calls"] == 0
    else:
        assert result["trace"]["db_neighbor_calls"] == 1
        assert len(result["hits"][0]["neighbors"]) == 1
        assert counters["neighbors"] == ["chunk-000"]

    _reset_oversample_counters(counters)
    zvec.hits = large_hits[:1]
    NativeQueryEngine(db, zvec_workspace=zvec).query(
        "native-test",
        "alpha",
        [1.0, 0.0],
        mode="naive",
        top_k=1,
        record_types=("chunk",),
        neighbor_limit=neighbor_limit,
    )
    assert (counters["connections"], len(counters["selects"])) == large_cost
    assert all(0 < value <= cost_cap for value in large_cost)


def test_zvec_equal_scores_are_normalized_before_route_rank(tmp_path) -> None:
    db = _db(tmp_path)
    for record_id in ("chunk-a", "chunk-b"):
        db.put_record(native_record("native-test", "chunk", record_id, "alpha evidence"))
    _audit(db, chunks=2)
    zvec = _ZvecWorkspace(
        [_ZvecHit("chunk", "chunk-b", score=1.0), _ZvecHit("chunk", "chunk-a", score=1.0)]
    )
    result = NativeQueryEngine(db, zvec_workspace=zvec).query(
        "native-test",
        "alpha",
        [1.0, 0.0],
        mode="naive",
        top_k=1,
        record_types=("chunk",),
        neighbor_limit=0,
    )
    assert [hit["record_id"] for hit in result["hits"]] == ["chunk-a"]
    assert result["hits"][0]["route_ranks"] == {"zvec_navigation": 1}


@pytest.mark.parametrize(
    ("top_k", "expected_candidate_budget"),
    [(1, 40), (20, 100), (100, 200)],
)
def test_selected_only_sql_cost_scales_with_selected_hits_not_candidate_budget(
    tmp_path,
    monkeypatch,
    top_k: int,
    expected_candidate_budget: int,
) -> None:
    db = _db(tmp_path)
    hits = []
    for index in range(100):
        record_id = f"chunk-{index:03d}"
        source_path = f"raw/{record_id}.md"
        db.put_record(
            native_record(
                "native-test",
                "chunk",
                record_id,
                "alpha evidence",
                source_path=source_path,
            )
        )
        hits.append(
            _ZvecHit(
                "chunk",
                record_id,
                score=float(100 - index),
                source_path=source_path,
                content="alpha evidence",
            )
        )
    _audit(db, chunks=100)
    connections = 0
    selects: list[str] = []
    traced = traced_sqlite_connect(db, selects, select_only=True)

    def tracked_connect():
        nonlocal connections
        connections += 1
        return traced()

    monkeypatch.setattr(db, "_connect", tracked_connect)
    zvec = _ZvecWorkspace(hits)
    result = NativeQueryEngine(db, zvec_workspace=zvec).query(
        "native-test",
        "alpha",
        [1.0, 0.0],
        mode="naive",
        top_k=top_k,
        record_types=("chunk",),
        neighbor_limit=0,
        retrieval_goal="coverage",
    )

    assert zvec.calls[0][2] == expected_candidate_budget
    assert len(result["hits"]) == top_k
    assert result["trace"]["db_record_calls"] == top_k
    assert result["trace"]["db_neighbor_calls"] == 0
    assert connections <= 1 + 2 * top_k
    assert len(selects) <= 1 + 2 * top_k


def test_engine_lexical_route_receives_all_normalized_terms(tmp_path, monkeypatch) -> None:
    db = _db(tmp_path)
    _audit(db)
    put_span(
        db,
        span_id="span:late",
        source_path="raw/late.md",
        source_id="source:late",
        source_role="raw",
        span_kind="table.row",
        heading_path=["Late"],
        text="LateIdentifier exact evidence",
    )
    captured_terms: list[tuple[str, ...]] = []
    original_lexical = db.query_lexical_spans

    def capture_terms(*args, **kwargs):
        captured_terms.append(tuple(kwargs.get("normalized_terms") or ()))
        return original_lexical(*args, **kwargs)

    monkeypatch.setattr(db, "query_lexical_spans", capture_terms)
    query = "one two three four five six seven eight LateIdentifier"
    result = NativeQueryEngine(db, zvec_workspace=_ZvecWorkspace()).query(
        "native-test",
        query,
        [1.0, 0.0],
        mode="mix",
        top_k=1,
        record_types=("entity",),
        neighbor_limit=0,
    )
    assert captured_terms == [
        ("one", "two", "three", "four", "five", "six", "seven", "eight", "lateidentifier")
    ]
    assert [hit["record_id"] for hit in result["hits"]] == ["span:late"]


def test_query_engine_routes_mix_to_zvec_hybrid_when_workspace_is_supplied() -> None:
    zvec = _ZvecWorkspace([_ZvecHit("chunk", "chunk-a", score=0.75, content="Alpha")])
    engine = NativeQueryEngine(
        _StubDb(record_extra={"vector_text": "Alpha"}, neighbors=[{"neighbor_id": "doc:b"}]),
        zvec_workspace=zvec,
    )

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
        ("mix", "alpha query", [1.0, 0.0], 40, "record_type_code in (1,2)")
    ]
    assert result["hits"][0]["doc_id"] == "chunk:chunk-a"
    assert result["hits"][0]["record"]["vector_text"] == "Alpha"
    assert result["hits"][0]["neighbors"] == [{"neighbor_id": "doc:b", "limit": 2}]
    assert result["trace"]["retrieval_backend"] == "zvec"


def test_query_engine_routes_naive_and_bypass_with_zvec_workspace() -> None:
    zvec = _ZvecWorkspace(
        [
            _ZvecHit(
                "section",
                "sec-a",
                score=0.5,
                source_kind_code=2,
                section_kind_code=4,
                content="Alpha section",
            )
        ],
        apply_filter=False,
        reject_mix=True,
    )
    engine = NativeQueryEngine(_StubDb(), zvec_workspace=zvec)

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

    assert zvec.calls == [("vector", [1.0, 0.0], 40, "record_type_code in (2)")]
    assert naive["hits"][0]["doc_id"] == "section:sec-a"
    assert naive["trace"]["retrieval_backend"] == "zvec"
    assert bypass["hits"] == []
    assert bypass["trace"]["retrieval_backend"] == "bypass"
    assert bypass["trace"]["vector_hit_count"] == 0
    assert bypass["trace"]["candidate_cards"] == []
    assert bypass["trace"]["merged_candidate_count"] == 0


def test_query_engine_routes_section_kind_as_numeric_zvec_filter() -> None:
    zvec = _ZvecWorkspace()
    engine = NativeQueryEngine(_StubDb(), zvec_workspace=zvec)

    result = engine.query(
        "native-test",
        "method query",
        [1.0, 0.0],
        mode="mix",
        top_k=3,
        record_types=("chunk", "section"),
        section_kind="methodology",
    )

    assert zvec.calls == [
        ("mix", "method query", [1.0, 0.0], 40, "record_type_code in (4) and section_kind_code in (4)")
    ]
    assert result["trace"]["section_kind"] == "methodology"


def test_data_only_query_engine_rejects_building_workspace(tmp_path) -> None:
    db = _db(tmp_path)
    db.put_record(native_record("native-test", "entity", "doc:a", "Alpha"))
    db.put_vector("native-test", "entity", "doc:a", "doc:a:vector", [1.0, 0.0])
    engine = NativeQueryEngine(db, zvec_workspace=_ZvecWorkspace())

    with pytest.raises(ValueError, match="workspace must be audited before query"):
        engine.query("native-test", "alpha query", [1.0, 0.0], mode="mix", record_types=("entity",))


def test_query_engine_rejects_mismatched_workspace_metadata_before_routes() -> None:
    class Db:
        def get_workspace_metadata(self, workspace_id: str) -> dict:
            return {
                "workspace_id": "different-workspace",
                "source_manifest_hash": "manifest-hash",
                "schema_version": 1,
                "status": "audited",
            }

    zvec = _ZvecWorkspace()
    engine = NativeQueryEngine(Db(), zvec_workspace=zvec)
    with pytest.raises(ValueError, match="workspace metadata mismatch"):
        engine.query("native-test", "alpha", [1.0], mode="mix")
    assert zvec.calls == []


def test_data_only_query_engine_rejects_unknown_or_retired_modes(tmp_path) -> None:
    db = _db(tmp_path, create=False)
    engine = NativeQueryEngine(db, zvec_workspace=_ZvecWorkspace())

    for mode in ["unsupported", "local", "global", "hybrid"]:
        with pytest.raises(ValueError, match="mode"):
            engine.query("native-test", "alpha query", [1.0], mode=mode)


def test_data_only_query_engine_rejects_invalid_record_types_and_goal_before_query(tmp_path) -> None:
    db = _db(tmp_path, create=False)
    zvec = _ZvecWorkspace()
    engine = NativeQueryEngine(db, zvec_workspace=zvec)

    with pytest.raises(ValueError, match="record_type"):
        engine.query("native-test", "alpha query", [1.0], mode="mix", record_types=("unknown",))
    invalid_record_types: list[Any] = [(), [], "entity"]
    for record_types in invalid_record_types:
        with pytest.raises(ValueError, match="record_types"):
            engine.query(
                "native-test",
                "alpha query",
                [1.0],
                mode="mix",
                record_types=record_types,
            )
    with pytest.raises(ValueError, match="retrieval_goal"):
        engine.query(
            "native-test",
            "alpha query",
            [1.0],
            mode="mix",
            retrieval_goal="wide",
        )
    assert zvec.calls == []


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


def test_read_span_rereads_current_source_and_relocates_exact_text(tmp_path) -> None:
    db = _db(tmp_path)
    _audit(db)
    wiki_root = tmp_path / "wiki"
    (wiki_root / "notes").mkdir(parents=True)
    source = wiki_root / "notes" / "alpha.md"
    source.write_text("# Alpha\n\nStable evidence line\nTail\n", encoding="utf-8")
    put_span(
        db,
        span_id="span:current",
        source_path="notes/alpha.md",
        source_id="wiki:alpha",
        source_role="wiki",
        span_kind="doc.paragraph",
        heading_path=["Alpha"],
        start_line=3,
        text="Stable evidence line",
    )
    put_span(
        db,
        span_id="span:moved",
        source_path="notes/alpha.md",
        source_id="wiki:alpha",
        source_role="wiki",
        span_kind="doc.paragraph",
        heading_path=["Alpha"],
        start_line=2,
        text="Relocated evidence line",
    )
    source.write_text("# Alpha\n\nStable evidence line\nInserted\nRelocated evidence line\n", encoding="utf-8")
    engine = NativeQueryEngine(db, zvec_workspace=_ZvecWorkspace(), source_root=wiki_root)

    current = engine.read_span("native-test", "span:current")
    moved = engine.read_span("native-test", "span:moved")
    snapshot = NativeQueryEngine(db, zvec_workspace=_ZvecWorkspace()).read_span("native-test", "span:moved")

    assert current["source_status"] == "current"
    assert current["text"] == "Stable evidence line"
    assert current["start_line"] == 3
    assert moved["source_status"] == "current"
    assert moved["relocation"] == "exact_text"
    assert moved["start_line"] == 5
    assert moved["text"] == "Relocated evidence line"
    assert snapshot["source_status"] == "snapshot"
    assert snapshot["text"] == "Relocated evidence line"


def test_read_span_falls_back_to_section_and_rejects_ambiguous_relocation(tmp_path) -> None:
    db = _db(tmp_path)
    db.put_record(
        native_record(
            "native-test",
            "section",
            "section:unique",
            "Unique section evidence",
            source_path="notes/unique.md",
            source_id="source:unique",
            payload={"section_kind": "results", "heading_path": ["Results"]},
        )
    )
    db.put_record(
        native_record(
            "native-test",
            "section",
            "section:duplicate",
            "Repeated template text",
            source_path="notes/duplicate.md",
            source_id="source:duplicate",
            payload={"section_kind": "methodology", "heading_path": ["Method"]},
        )
    )
    _audit(db, sections=2)
    wiki_root = tmp_path / "wiki"
    (wiki_root / "notes").mkdir(parents=True)
    (wiki_root / "notes" / "unique.md").write_text(
        "# Unique\n\nUnique section evidence\n",
        encoding="utf-8",
    )
    (wiki_root / "notes" / "duplicate.md").write_text(
        "Repeated template text\nMiddle\nRepeated template text\n",
        encoding="utf-8",
    )
    engine = NativeQueryEngine(db, zvec_workspace=_ZvecWorkspace(), source_root=wiki_root)

    unique = engine.read_span("native-test", "section:unique")
    ambiguous = engine.read_span("native-test", "section:duplicate")
    snapshot = NativeQueryEngine(
        db,
        zvec_workspace=_ZvecWorkspace(),
    ).read_span("native-test", "section:unique")

    assert unique["span_kind"] == "raw.section"
    assert unique["source_status"] == "current"
    assert unique["relocation"] == "exact_text"
    assert unique["start_line"] == 3
    assert ambiguous["source_status"] == "ambiguous"
    assert ambiguous["relocation"] == "multiple_exact_text"
    assert ambiguous["start_line"] == 0
    assert snapshot["source_status"] == "snapshot"
    assert snapshot["start_line"] == 0
