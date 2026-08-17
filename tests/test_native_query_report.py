from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from support import write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def _quality_argv(tmp_path: Path, *extra: str, partition: str | None = "all") -> list[str]:
    argv = [
        "--quality-contract",
        "relevance-v1",
        "--query-suite",
        str(tmp_path / "suite.jsonl"),
        "--workspace-file",
        str(tmp_path / "pointer.json"),
        "--runtime-code-root",
        str(ROOT),
        "--server",
        "http://127.0.0.1:19637",
    ]
    if partition is not None:
        argv[2:2] = ["--partition", partition]
    argv.extend(extra)
    return argv


def _collect(suite, pointer, *, partition="all", **kwargs):
    return collect_native_query_report.collect_quality_report(
        query_suite_path=suite,
        server="http://127.0.0.1:19637",
        workspace_file=pointer,
        runtime_code_root=ROOT,
        partition=partition,
        warmup_runs=0,
        repetitions=1,
        **kwargs,
    )


def _legacy_query_row(row_id: str, query: str, **extra) -> dict:
    row = {
        "id": row_id,
        "query": query,
        "mode": "mix",
        "top_k": 20,
        "must_include_paths": ["a.md"],
        "must_include_entities": [],
        "notes": "fixture",
    }
    row.update(extra)
    return row


def _ok_get(active_id: str = "native-fixture"):
    def fake_get(_url: str, *, timeout: int) -> dict:
        return {"status": "ok", "active_workspace_id": active_id}

    return fake_get

from llm_wiki_native.retrieval.context import assemble_context  # noqa: E402
from ops import collect_native_query_report  # noqa: E402


def _quality_rows() -> list[dict]:
    rows: list[dict] = []
    for index in range(36):
        holdout = index >= 24
        coverage = 16 <= index < 24 or index >= 32
        source_path = f"raw/{index + 1:02d}.md"
        second_source_path = f"raw/{index + 1:02d}-comparison.md"
        rows.append(
            {
                "id": f"quality-{index + 1:02d}",
                "query": f"evidence query {index + 1}",
                "mode": "mix",
                "top_k": 4,
                "must_include_paths": [source_path, second_source_path] if coverage else [source_path],
                "must_include_entities": [],
                "must_include_evidence": (
                    [
                        {"source_path": source_path, "text_contains": [f"anchor-{index + 1:02d}"]},
                        {"source_path": second_source_path, "text_contains": [f"compare-{index + 1:02d}"]},
                    ]
                    if coverage
                    else [{"source_path": source_path, "text_contains": [f"anchor-{index + 1:02d}"]}]
                ),
                "notes": "strict quality fixture",
                "retrieval_goal": "coverage" if coverage else "focused",
                "critical": holdout,
                "partition": "holdout" if holdout else "calibration",
                "minimum_distinct_sources": 2 if coverage else 1,
                "query_vector": [float(index + 1), 0.0],
                "response_profile": "debug",
            }
        )
    return rows


def _workspace_pointer(tmp_path: Path) -> Path:
    source_root = tmp_path / "wiki"
    source_root.mkdir()
    zvec_path = tmp_path / "zvec"
    zvec_path.mkdir()
    sqlite_path = tmp_path / "workspace.sqlite3"
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute(
            "CREATE TABLE workspace("
            "workspace_id TEXT PRIMARY KEY, source_manifest_hash TEXT, "
            "schema_version INTEGER, status TEXT)"
        )
        conn.execute(
            "INSERT INTO workspace VALUES (?, ?, ?, ?)",
            ("native-fixture", "manifest-fixture", 1, "audited"),
        )
    pointer = tmp_path / "active_workspace.json"
    pointer.write_text(
        json.dumps(
            {
                "workspace_id": "native-fixture",
                "source_manifest_hash": "manifest-fixture",
                "schema_version": 1,
                "status": "active",
                "embedding_dim": 2,
                "source_root": str(source_root),
                "sqlite_path": str(sqlite_path),
                "zvec_path": str(zvec_path),
            }
        ),
        encoding="utf-8",
    )
    return pointer


def test_relevance_quality_reader_freezes_counts_ids_dimensions_and_partitions(tmp_path: Path) -> None:
    suite = tmp_path / "quality.jsonl"
    write_jsonl(suite, _quality_rows())

    rows = collect_native_query_report.read_query_suite(suite, quality_contract="relevance-v1")

    assert len(rows) == 36
    assert len(collect_native_query_report.select_quality_partition(rows, "calibration")) == 24
    assert len(collect_native_query_report.select_quality_partition(rows, "holdout")) == 12
    assert collect_native_query_report.select_quality_partition(rows, "all") == rows


@pytest.mark.parametrize("mutation", ["empty", "duplicate", "dimension", "partition-count", "goal-count"])
def test_relevance_quality_reader_rejects_invalid_suite_shape(tmp_path: Path, mutation: str) -> None:
    rows = _quality_rows()
    if mutation == "empty":
        rows = []
    elif mutation == "duplicate":
        rows[-1]["id"] = rows[0]["id"]
    elif mutation == "dimension":
        rows[-1]["query_vector"] = [1.0]
    elif mutation == "partition-count":
        rows[23]["partition"] = "holdout"
        rows[23]["critical"] = True
    else:
        rows[15]["retrieval_goal"] = "coverage"
        rows[15]["minimum_distinct_sources"] = 2
        rows[15]["must_include_paths"].append("raw/extra.md")
        rows[15]["must_include_evidence"].append(
            {"source_path": "raw/extra.md", "text_contains": ["extra-anchor"]}
        )
    suite = tmp_path / "quality.jsonl"
    write_jsonl(suite, rows)

    with pytest.raises(ValueError):
        collect_native_query_report.read_query_suite(suite, quality_contract="relevance-v1")


def test_corpus_identity_cross_checks_frozen_pointer_and_sqlite_metadata(tmp_path: Path) -> None:
    pointer = _workspace_pointer(tmp_path)

    identity = collect_native_query_report.build_corpus_identity(pointer)

    assert identity["frozen_pointer_sha256"] == hashlib.sha256(pointer.read_bytes()).hexdigest()
    assert identity["workspace_id"] == "native-fixture"
    assert identity["source_manifest_hash"] == "manifest-fixture"
    assert identity["workspace_schema_version"] == 1
    assert identity["sqlite_status"] == "audited"
    assert identity["resolved_targets"]["source_root"] == str((tmp_path / "wiki").resolve())

    payload = json.loads(pointer.read_text(encoding="utf-8"))
    payload["source_manifest_hash"] = "drifted"
    pointer.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source_manifest_hash"):
        collect_native_query_report.build_corpus_identity(pointer)


def test_corpus_identity_reads_embedding_dimension_from_production_zvec_shape(tmp_path: Path) -> None:
    pointer = _workspace_pointer(tmp_path)
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    embedding_dim = payload.pop("embedding_dim")
    payload["zvec"] = {"embedding_dim": embedding_dim}
    pointer.write_text(json.dumps(payload), encoding="utf-8")

    identity = collect_native_query_report.build_corpus_identity(pointer)

    assert identity["embedding_dim"] == 2


def _scoring_row(**overrides: object) -> dict:
    row = _quality_rows()[0]
    row.update(
        {
            "top_k": 2,
            "must_include_paths": ["raw/a.md"],
            "must_include_entities": ["section:a"],
            "must_include_evidence": [
                {"source_path": "raw/a.md", "text_contains": ["Exact", "Anchor"]},
            ],
        }
    )
    row.update(overrides)
    return row


def _scoring_response() -> dict:
    return {
        "source_paths": ["raw/a.md", "raw/b.md"],
        "context_blocks": [
            {"record_id": "section:a", "source_path": "raw/a.md", "text": "Exact evidence Anchor"},
            {"record_id": "section:b", "source_path": "raw/b.md", "text": "Other evidence"},
        ],
        "retrieval_debug": {
            "source_scope": [{"source_path": "raw/a.md"}, {"source_path": "raw/c.md"}],
            "candidate_cards": [{"record_id": "section:a", "source_id": "source:a"}],
            "selection": {"goal": "focused", "elapsed_ms": 7.0, "pid": 123},
        },
        "trace": {"route_ms": 2.0, "timestamp": "later"},
    }


def test_quality_scoring_covers_scope_visible_entities_evidence_and_bounds() -> None:
    metrics = collect_native_query_report.score_quality_response(_scoring_row(), _scoring_response())

    assert metrics["first_required_path_rank"] == 1
    assert metrics["distinct_source_count"] == 2
    assert metrics["relevant_distinct_source_count"] == 1
    assert metrics["minimum_relevant_sources_met"] is True
    assert metrics["duplicate_block_rate"] == 0.0
    assert metrics["response_bound_ok"] is True
    assert metrics["response_size_bound_ok"] is True


def test_assembled_debug_response_matches_frozen_quality_scorer_contract() -> None:
    row = _scoring_row()
    response = assemble_context(
        {
            "hits": [
                {
                    "record_id": "section:a",
                    "record_type": "section",
                    "score": 1.0,
                    "ranking_contract": "relevance-v1",
                    "record": {
                        "vector_text": "Exact text with Anchor in one block",
                        "content_hash": "section-a-hash",
                        "source_path": "raw/a.md",
                        "source_id": "source:a",
                        "payload": {"source_role": "raw", "section_kind": "results"},
                    },
                    "neighbors": [],
                }
            ],
            "trace": {
                "query": "alpha evidence",
                "retrieval_goal": "focused",
                "coverage_fill_pass_used": False,
                "source_scope": [{"source_key": "raw/a.md", "source_score": 1.0}],
                "planner_decisions": [
                    {"record_id": "section:a", "record_type": "section", "decision": "selected", "reason": "selected"}
                ],
                "candidate_cards": [
                    {
                        "record_type": "section",
                        "record_id": "section:a",
                        "source_path": "raw/a.md",
                        "source_id": "source:a",
                        "source_key": "raw/a.md",
                        "route_family": "zvec_section",
                        "route_rank": 1,
                        "routes": ["zvec"],
                    }
                ],
            },
        },
        response_profile="debug",
    )

    metrics = collect_native_query_report.score_quality_response(row, response)

    assert metrics["scope_path_recall"] == 1.0
    assert metrics["visible_path_recall"] == 1.0
    assert metrics["candidate_entity_recall"] == 1.0
    assert metrics["visible_evidence_recall"] == 1.0
    assert metrics["quality_pass"] is True


def test_quality_scoring_requires_all_case_sensitive_anchors_in_one_visible_block() -> None:
    response = _scoring_response()
    response["context_blocks"] = [
        {"record_id": "section:a", "source_path": "raw/a.md", "text": "Exact only"},
        {"record_id": "section:b", "source_path": "raw/a.md", "text": "anchor only"},
    ]
    response["source_paths"] = ["raw/a.md"]

    metrics = collect_native_query_report.score_quality_response(_scoring_row(), response)

    assert metrics["visible_evidence_recall"] == 0.0
    assert metrics["duplicate_block_rate"] == 0.5


def test_quality_scoring_uses_debug_hits_as_legacy_scope_and_empty_entities_pass() -> None:
    row = _scoring_row(must_include_paths=["raw/c.md"], must_include_entities=[])
    response = _scoring_response()
    response["retrieval_debug"] = {"hits": [{"source_path": "raw/c.md"}]}

    metrics = collect_native_query_report.score_quality_response(row, response)

    assert metrics["scope_path_recall"] == 1.0
    assert metrics["visible_path_recall"] == 0.0
    assert metrics["candidate_entity_recall"] == 1.0
    assert metrics["first_required_path_rank"] is None


def test_quality_response_fingerprint_excludes_process_diagnostics_but_includes_text() -> None:
    first = _scoring_response()
    second = json.loads(json.dumps(first))
    second["retrieval_debug"]["selection"]["elapsed_ms"] = 99.0
    second["retrieval_debug"]["selection"]["pid"] = 999
    second["trace"]["timestamp"] = "different"

    assert collect_native_query_report.quality_response_fingerprint(first) == (
        collect_native_query_report.quality_response_fingerprint(second)
    )
    second["context_blocks"][0]["text"] = "different evidence"
    assert collect_native_query_report.quality_response_fingerprint(first) != (
        collect_native_query_report.quality_response_fingerprint(second)
    )


def test_code_fingerprints_are_stable_and_owner_specific(tmp_path: Path) -> None:
    (tmp_path / "llm_wiki_native").mkdir()
    (tmp_path / "llm_wiki_native" / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "wiki_search.py").write_text("CLIENT = 1\n", encoding="utf-8")
    (tmp_path / "ops" / "wiki_native_query_events.py").write_text("EVENTS = 1\n", encoding="utf-8")

    runtime_before = collect_native_query_report.runtime_package_fingerprint(tmp_path)
    client_before = collect_native_query_report.client_fingerprint(tmp_path)
    (tmp_path / "llm_wiki_native" / "runtime.py").write_text("VALUE = 2\n", encoding="utf-8")

    assert collect_native_query_report.runtime_package_fingerprint(tmp_path) != runtime_before
    assert collect_native_query_report.client_fingerprint(tmp_path) == client_before


def test_evaluator_fingerprint_tracks_only_evaluator_files_and_frozen_constants(tmp_path: Path) -> None:
    (tmp_path / "llm_wiki_native").mkdir()
    (tmp_path / "llm_wiki_native" / "reports.py").write_text("REPORTS = 1\n", encoding="utf-8")
    (tmp_path / "ops").mkdir()
    collector = tmp_path / "ops" / "collect_native_query_report.py"
    collector.write_text("COLLECTOR = 1\n", encoding="utf-8")

    before = collect_native_query_report.evaluator_fingerprint(tmp_path)
    (tmp_path / "llm_wiki_native" / "unrelated.py").write_text("UNRELATED = 2\n", encoding="utf-8")
    assert collect_native_query_report.evaluator_fingerprint(tmp_path) == before
    collector.write_text("COLLECTOR = 2\n", encoding="utf-8")
    assert collect_native_query_report.evaluator_fingerprint(tmp_path) != before


def test_quality_repetitions_record_each_score_and_require_determinism() -> None:
    first = _scoring_response()
    second = json.loads(json.dumps(first))
    second["trace"]["route_ms"] = 99.0

    result = collect_native_query_report.evaluate_quality_repetitions(
        _scoring_row(),
        responses=[first, second],
        latencies_ms=[10.0, 20.0],
    )

    assert len(result["repetition_results"]) == 2
    assert result["response_deterministic"] is True
    assert result["quality_pass"] is True
    assert result["latency_median_ms"] == 15.0
    assert result["metrics"]["visible_evidence_recall"] == 1.0
    assert all(item["response_bytes"] > 0 for item in result["repetition_results"])

    second["context_blocks"][1]["text"] = "changed but still irrelevant"
    nondeterministic = collect_native_query_report.evaluate_quality_repetitions(
        _scoring_row(),
        responses=[first, second],
        latencies_ms=[10.0, 20.0],
    )
    assert nondeterministic["response_deterministic"] is False
    assert nondeterministic["quality_pass"] is False


def test_quality_repetition_assigns_a_primary_class_to_same_source_hash_failure() -> None:
    response = _scoring_response()
    response["source_paths"] = ["raw/a.md"]
    response["context_blocks"] = [
        {"record_id": "section:a", "source_path": "raw/a.md", "text": "Exact evidence Anchor"},
        {"record_id": "section:a-copy", "source_path": "raw/a.md", "text": "Exact evidence Anchor"},
    ]

    result = collect_native_query_report.evaluate_quality_repetitions(
        _scoring_row(),
        responses=[response],
        latencies_ms=[5.0],
    )

    assert result["quality_pass"] is False
    assert result["failure_class"] == "evidence_ranking"


def test_quality_aggregates_use_nearest_rank_p95_and_emit_goal_partition_groups() -> None:
    focused = collect_native_query_report.evaluate_quality_repetitions(
        _scoring_row(id="focused", partition="calibration", critical=False),
        responses=[_scoring_response()] * 5,
        latencies_ms=[1.0, 2.0, 3.0, 4.0, 100.0],
    )
    coverage_row = _scoring_row(
        id="coverage",
        retrieval_goal="coverage",
        partition="holdout",
        critical=True,
        minimum_distinct_sources=2,
        must_include_paths=["raw/a.md", "raw/b.md"],
        must_include_entities=[],
        must_include_evidence=[
            {"source_path": "raw/a.md", "text_contains": ["Exact", "Anchor"]},
            {"source_path": "raw/b.md", "text_contains": ["Other", "evidence"]},
        ],
    )
    coverage_response = _scoring_response()
    coverage_response["retrieval_debug"]["source_scope"] = [
        {"source_path": "raw/a.md"},
        {"source_path": "raw/b.md"},
    ]
    coverage = collect_native_query_report.evaluate_quality_repetitions(
        coverage_row,
        responses=[coverage_response] * 5,
        latencies_ms=[5.0] * 5,
    )

    aggregates = collect_native_query_report.aggregate_quality_results([focused, coverage])

    assert aggregates["overall"]["latency"]["p95_ms"] == 100.0
    assert aggregates["focused"]["row_count"] == 1
    assert aggregates["coverage"]["row_count"] == 1
    assert aggregates["partition"]["calibration"]["row_count"] == 1
    assert aggregates["partition"]["holdout"]["critical_failure_ids"] == []


def _report_result(row: dict, *, latency_ms: float = 10.0) -> dict:
    metrics = {
        "scope_path_recall": 1.0,
        "visible_path_recall": 1.0,
        "candidate_entity_recall": 1.0,
        "visible_evidence_recall": 1.0,
        "first_required_path_rank": 1,
        "distinct_source_count": row["minimum_distinct_sources"],
        "relevant_distinct_source_count": row["minimum_distinct_sources"],
        "minimum_relevant_sources_met": True,
        "duplicate_block_rate": 0.0,
        "response_chars": 100,
        "response_bytes": 200,
        "response_bound_ok": True,
        "response_size_bound_ok": True,
        "focused_evidence_hashes_distinct": True,
        "coverage_distinct_first_pass_ok": True,
    }
    repetitions = [
        {"latency_ms": latency_ms, "response_chars": 100, "response_bytes": 200}
        for _ in range(5)
    ]
    return {
        "id": row["id"],
        "retrieval_goal": row["retrieval_goal"],
        "partition": row["partition"],
        "critical": row["critical"],
        "repetition_results": repetitions,
        "response_deterministic": True,
        "latencies_ms": [latency_ms] * 5,
        "latency_median_ms": latency_ms,
        "metrics": metrics,
        "quality_pass": True,
        "failure_class": None,
        "effective_request_sha256": f"request-{row['id']}",
    }


def _quality_report(*, partition: str, latency_ms: float = 10.0) -> dict:
    rows = _quality_rows()
    if partition != "all":
        rows = [row for row in rows if row["partition"] == partition]
    results = [_report_result(row, latency_ms=latency_ms) for row in rows]
    return {
        "quality_contract": "relevance-v1",
        "report_contract_version": "relevance-v1-report-v1",
        "partition": partition,
        "query_suite_sha256": "suite-sha",
        "selected_row_ids": [result["id"] for result in results],
        "corpus_identity": {
            "frozen_pointer_path": "/provenance-only/pointer.json",
            "frozen_pointer_sha256": "pointer-sha",
            "workspace_id": "native-fixture",
            "source_manifest_hash": "manifest-fixture",
            "workspace_schema_version": 1,
            "status": "active",
            "embedding_dim": 2,
            "resolved_targets": {
                "source_root": "/wiki",
                "sqlite_path": "/workspace.sqlite3",
                "zvec_path": "/zvec",
            },
        },
        "runtime_package_fingerprint": "runtime-candidate",
        "runtime_git_revision": None,
        "evaluator_fingerprint": "evaluator",
        "client_fingerprint": "client",
        "endpoint": "/query/data",
        "timing_scope": "data_only",
        "warmup_runs": 1,
        "repetitions": 5,
        "results": results,
        "aggregates": collect_native_query_report.aggregate_quality_results(results),
    }


def test_quality_comparison_projects_baseline_and_does_not_claim_calibration_critical_gate() -> None:
    baseline = _quality_report(partition="all")
    candidate = _quality_report(partition="calibration")

    comparison = collect_native_query_report.compare_quality_reports(candidate, baseline)

    assert comparison["identity_valid"] is True
    assert comparison["gates_passed"] is True
    assert comparison["critical_gate"]["applicable"] is False
    assert comparison["paired_latency"]["success_rate"] == 1.0


@pytest.mark.parametrize(
    ("mutation", "failure"),
    [
        ("evaluator", "evaluator_fingerprint"),
        ("workspace", "corpus_identity"),
        ("endpoint", "endpoint"),
        ("request", "effective_request"),
    ],
)
def test_quality_comparison_rejects_identity_mismatch(mutation: str, failure: str) -> None:
    baseline = _quality_report(partition="all")
    candidate = _quality_report(partition="calibration")
    if mutation == "evaluator":
        candidate["evaluator_fingerprint"] = "different"
    elif mutation == "workspace":
        candidate["corpus_identity"]["workspace_id"] = "different"
    elif mutation == "endpoint":
        candidate["endpoint"] = "/query"
    else:
        candidate["results"][0]["effective_request_sha256"] = "different"

    comparison = collect_native_query_report.compare_quality_reports(candidate, baseline)

    assert comparison["identity_valid"] is False
    assert comparison["gates_passed"] is False
    assert failure in comparison["identity_failures"]


def test_quality_comparison_rejects_holdout_rows_in_calibration_projection() -> None:
    baseline = _quality_report(partition="all")
    candidate = _quality_report(partition="calibration")
    leaked = json.loads(json.dumps(baseline["results"][24]))
    candidate["results"][-1] = leaked
    candidate["selected_row_ids"][-1] = leaked["id"]
    candidate["aggregates"] = collect_native_query_report.aggregate_quality_results(candidate["results"])

    comparison = collect_native_query_report.compare_quality_reports(candidate, baseline)

    assert comparison["identity_valid"] is False
    assert "candidate_partition" in comparison["identity_failures"]
    assert comparison["gates_passed"] is False


def test_quality_comparison_enforces_endpoint_and_paired_row_latency_gates() -> None:
    baseline = _quality_report(partition="all", latency_ms=10.0)
    candidate = _quality_report(partition="all", latency_ms=100.0)

    comparison = collect_native_query_report.compare_quality_reports(candidate, baseline)

    assert comparison["endpoint_latency"]["passed"] is False
    assert comparison["paired_latency"]["success_rate"] == 0.0
    assert comparison["gates_passed"] is False


def test_active_comparison_requires_accepted_candidate_runtime_and_client_identity() -> None:
    baseline = _quality_report(partition="all")
    accepted = _quality_report(partition="all")
    active = json.loads(json.dumps(accepted))
    active["runtime_package_fingerprint"] = "different-runtime"
    active["client_fingerprint"] = "different-client"

    comparison = collect_native_query_report.compare_quality_reports(
        active,
        baseline,
        accepted_candidate_report=accepted,
    )

    assert comparison["active_identity_passed"] is False
    assert set(comparison["active_identity_failures"]) == {
        "runtime_package_fingerprint",
        "client_fingerprint",
    }
    assert comparison["gates_passed"] is False


def _quality_response_for_row(row: dict) -> dict:
    blocks = [
        {
            "record_id": f"section:{index}",
            "source_path": item["source_path"],
            "text": " | ".join(item["text_contains"]),
        }
        for index, item in enumerate(row["must_include_evidence"], start=1)
    ]
    return {
        "source_paths": [block["source_path"] for block in blocks],
        "context_blocks": blocks,
        "retrieval_debug": {
            "source_scope": [{"source_path": path} for path in row["must_include_paths"]],
            "candidate_cards": [],
            "selection": {"goal": row["retrieval_goal"]},
        },
        "trace": {"context_block_count": len(blocks)},
    }


def test_quality_collector_baseline_preflights_and_never_forwards_scoring_labels(tmp_path: Path) -> None:
    suite = tmp_path / "quality.jsonl"
    rows = _quality_rows()
    write_jsonl(suite, rows)
    pointer = _workspace_pointer(tmp_path)
    calls: list[dict] = []

    def fake_get(_url: str, *, timeout: int) -> dict:
        assert timeout == 120
        return {"status": "ok", "active_workspace_id": "native-fixture"}

    def fake_post(_url: str, payload: dict, *, timeout: int) -> dict:
        calls.append(payload)
        return _quality_response_for_row(rows[len(calls) - 1])

    report = _collect(suite, pointer, get_json=fake_get, post_json=fake_post)

    assert len(calls) == 36
    assert all(call["workspace_id"] == "native-fixture" for call in calls)
    assert all("retrieval_goal" not in call for call in calls)
    assert all(
        not ({"partition", "critical", "must_include_evidence", "minimum_distinct_sources"} & set(call))
        for call in calls
    )
    assert report["report_role"] == "baseline"
    assert report["corpus_identity"]["workspace_id"] == "native-fixture"
    assert report["runtime_package_fingerprint"]
    assert report["evaluator_fingerprint"]
    assert report["client_fingerprint"]
    assert report["runtime_process_binding"]["bound"] is False
    assert report["gates_passed"] is None
    assert report["aggregates"]["overall"]["all_quality_pass"] is True


def test_quality_collector_candidate_forwards_goal_and_uses_baseline_projection(tmp_path: Path) -> None:
    suite = tmp_path / "quality.jsonl"
    rows = _quality_rows()
    write_jsonl(suite, rows)
    pointer = _workspace_pointer(tmp_path)

    fake_get = _ok_get()
    baseline_index = 0

    def baseline_post(_url: str, _payload: dict, *, timeout: int) -> dict:
        nonlocal baseline_index
        response = _quality_response_for_row(rows[baseline_index])
        baseline_index += 1
        return response

    baseline = _collect(suite, pointer, get_json=fake_get, post_json=baseline_post)
    calibration_rows = rows[:24]
    candidate_calls: list[dict] = []

    def candidate_post(_url: str, payload: dict, *, timeout: int) -> dict:
        candidate_calls.append(payload)
        return _quality_response_for_row(calibration_rows[len(candidate_calls) - 1])

    candidate = _collect(
        suite,
        pointer,
        partition="calibration",
        baseline_report=baseline,
        get_json=fake_get,
        post_json=candidate_post,
    )

    assert [call["retrieval_goal"] for call in candidate_calls] == [
        row["retrieval_goal"] for row in calibration_rows
    ]
    assert candidate["report_role"] == "candidate"
    assert candidate["comparison"]["baseline_projection_row_ids"] == [row["id"] for row in calibration_rows]
    assert candidate["gates_passed"] is True


@pytest.mark.parametrize("mismatch", ["cli-workspace", "row-workspace", "vector-dim", "health"])
def test_quality_collector_rejects_identity_mismatch_before_query_post(
    tmp_path: Path, mismatch: str
) -> None:
    suite = tmp_path / "quality.jsonl"
    rows = _quality_rows()
    workspace_id = None
    if mismatch == "cli-workspace":
        workspace_id = "wrong"
    elif mismatch == "row-workspace":
        rows[0]["workspace_id"] = "wrong"
    elif mismatch == "vector-dim":
        for row in rows:
            row["query_vector"] = [1.0, 0.0, 0.0]
    write_jsonl(suite, rows)
    pointer = _workspace_pointer(tmp_path)
    posts: list[dict] = []

    def fake_post(_url: str, payload: dict, *, timeout: int) -> dict:
        posts.append(payload)
        return {}

    with pytest.raises(ValueError):
        _collect(
            suite,
            pointer,
            workspace_id=workspace_id,
            get_json=_ok_get("wrong" if mismatch == "health" else "native-fixture"),
            post_json=fake_post,
        )

    assert posts == []


def test_quality_cli_writes_attempt_promotes_on_pass_and_never_serializes_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}", encoding="utf-8")
    output = tmp_path / "attempt.json"
    promoted = tmp_path / "accepted.json"
    calls: list[dict] = []
    monkeypatch.setenv("QUALITY_API_KEY", "super-secret-value")

    def fake_collect(**kwargs: object) -> dict:
        calls.append(kwargs)
        return {"gates_passed": True, "marker": "accepted"}

    monkeypatch.setattr(collect_native_query_report, "collect_quality_report", fake_collect)
    argv = _quality_argv(
        tmp_path,
        "--baseline-report",
        str(baseline),
        "--api-key-env",
        "QUALITY_API_KEY",
        "--require-gates",
        "--fail-if-output-exists",
        "--output",
        str(output),
        "--promote-on-pass",
        str(promoted),
    )

    assert collect_native_query_report.main(argv) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["marker"] == "accepted"
    assert promoted.read_bytes() == output.read_bytes()
    assert b"super-secret-value" not in output.read_bytes()
    assert calls[0]["api_key"] == "super-secret-value"

    assert collect_native_query_report.main(argv) != 0
    assert len(calls) == 1


def test_quality_cli_retains_failed_attempt_and_does_not_promote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}", encoding="utf-8")
    output = tmp_path / "failed-attempt.json"
    promoted = tmp_path / "must-not-exist.json"
    monkeypatch.setattr(
        collect_native_query_report,
        "collect_quality_report",
        lambda **_kwargs: {"gates_passed": False, "failure": "quality"},
    )

    result = collect_native_query_report.main(
        _quality_argv(
            tmp_path,
            "--baseline-report",
            str(baseline),
            "--require-gates",
            "--output",
            str(output),
            partition="calibration",
        )
    )

    assert result != 0
    assert json.loads(output.read_text(encoding="utf-8"))["gates_passed"] is False
    assert not promoted.exists()


def test_atomic_promotion_is_no_overwrite_and_preserves_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "accepted.json"
    target.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        collect_native_query_report.write_report_atomic(
            target,
            {"new": True},
            no_overwrite=True,
        )

    assert target.read_text(encoding="utf-8") == "existing"
    assert not list(tmp_path.glob("*.tmp"))


def test_quality_cli_fails_before_collection_when_promotion_target_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}", encoding="utf-8")
    promoted = tmp_path / "accepted.json"
    promoted.write_text("existing", encoding="utf-8")
    calls: list[dict] = []

    def fake_collect(**kwargs: object) -> dict:
        calls.append(kwargs)
        return {"gates_passed": True}

    monkeypatch.setattr(collect_native_query_report, "collect_quality_report", fake_collect)
    result = collect_native_query_report.main(
        _quality_argv(
            tmp_path,
            "--baseline-report",
            str(baseline),
            "--output",
            str(tmp_path / "new-attempt.json"),
            "--promote-on-pass",
            str(promoted),
            partition=None,
        )
    )

    assert result != 0
    assert calls == []
    assert promoted.read_text(encoding="utf-8") == "existing"


def test_default_cli_keeps_legacy_collection_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "legacy.json"
    calls: list[dict] = []

    def fake_collect(**kwargs: object) -> dict:
        calls.append(kwargs)
        return {"legacy": True}

    monkeypatch.setattr(collect_native_query_report, "collect_query_report", fake_collect)

    assert (
        collect_native_query_report.main(
            [
                "--query-suite",
                str(tmp_path / "minimal.jsonl"),
                "--server",
                "http://127.0.0.1:19637",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8")) == {"legacy": True}
    assert len(calls) == 1


def test_native_query_report_collector_posts_suite_rows_and_records_latency(tmp_path: Path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    write_jsonl(
        query_suite,
        [
            _legacy_query_row(
                "q1",
                "alpha evidence",
                query_vector=[1.0, 0.0],
                section_kind="methodology",
                neighbor_limit=2,
                response_profile="compact",
            ),
            _legacy_query_row("q2", "beta evidence", mode="naive", must_include_paths=["b.md"], query_vector=[0.0, 1.0]),
        ],
    )
    calls: list[dict] = []
    responses = [
        {"source_paths": ["a.md"], "context_blocks": [{"source_path": "a.md"}]},
        {"source_paths": ["b.md"], "context_blocks": [{"source_path": "b.md"}]},
    ]
    ticks = iter([1.00, 1.01, 2.00, 2.03])

    def fake_post(url: str, payload: dict, *, timeout: int) -> dict:
        calls.append({"url": url, "payload": payload, "timeout": timeout})
        return responses[len(calls) - 1]

    report = collect_native_query_report.collect_query_report(
        query_suite_path=query_suite,
        server="http://127.0.0.1:19637",
        workspace_id="native-test",
        post_json=fake_post,
        timer=lambda: next(ticks),
    )

    assert report["query_suite"] == str(query_suite)
    assert report["query_suite_sha256"] == hashlib.sha256(query_suite.read_bytes()).hexdigest()
    assert report["endpoint"] == "/query/data"
    assert report["timing_scope"] == "data_only"
    assert report["summary"]["count"] == 2
    assert report["summary"]["min_ms"] == 10.0
    assert report["summary"]["max_ms"] == 30.0
    assert report["summary"]["p95_ms"] == 29.0
    assert [row["id"] for row in report["results"]] == ["q1", "q2"]
    assert report["results"][0]["elapsed_ms"] == 10.0
    assert report["results"][1]["elapsed_ms"] == 30.0
    assert report["results"][0]["request"] == {
        "top_k": 20,
        "retrieval_goal": "focused",
        "query_vector_dim": 2,
        "section_kind": "methodology",
        "neighbor_limit": 2,
        "response_profile": "compact",
    }
    assert calls[0]["url"] == "http://127.0.0.1:19637/query/data"
    assert calls[0]["payload"]["workspace_id"] == "native-test"
    assert calls[0]["payload"]["query_vector"] == [1.0, 0.0]
    assert calls[1]["payload"]["mode"] == "naive"


def test_native_query_report_collector_marks_endpoint_embedding_timing_for_missing_query_vectors(tmp_path: Path) -> None:
    for index, query_vector in enumerate([None, []]):
        query_suite = tmp_path / f"query_suite_{index}.jsonl"
        row = _legacy_query_row("q1", "alpha evidence")
        if query_vector is not None:
            row["query_vector"] = query_vector
        write_jsonl(query_suite, [row])

        def fake_post(_url: str, _payload: dict, *, timeout: int) -> dict:
            return {"source_paths": ["a.md"]}

        report = collect_native_query_report.collect_query_report(
            query_suite_path=query_suite,
            server="http://127.0.0.1:19637",
            post_json=fake_post,
            timer=iter([1.00, 1.01]).__next__,
        )

        assert report["timing_scope"] == "endpoint_includes_embedding"


def test_native_query_report_collector_records_repeated_samples_without_duplicate_results(tmp_path: Path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    write_jsonl(
        query_suite,
        [_legacy_query_row("q1", "alpha evidence", query_vector=[1.0, 0.0])],
    )
    calls: list[dict] = []
    responses = [
        {"source_paths": ["warmup.md"]},
        {"source_paths": ["a.md"], "context_blocks": [{"source_path": "a.md"}]},
        {"source_paths": ["a.md"], "context_blocks": [{"source_path": "a.md"}]},
    ]
    ticks = iter([1.00, 1.02, 2.00, 2.03])

    def fake_post(url: str, payload: dict, *, timeout: int) -> dict:
        calls.append({"url": url, "payload": payload, "timeout": timeout})
        return responses[len(calls) - 1]

    report = collect_native_query_report.collect_query_report(
        query_suite_path=query_suite,
        server="http://127.0.0.1:19637",
        workspace_id="native-test",
        warmup_runs=1,
        repetitions=2,
        post_json=fake_post,
        timer=lambda: next(ticks),
    )

    assert len(calls) == 3
    assert report["summary"]["count"] == 2
    assert report["summary"]["p95_ms"] == 29.5
    assert report["samples"] == [
        {"id": "q1", "rep": 1, "elapsed_ms": 20.0},
        {"id": "q1", "rep": 2, "elapsed_ms": 30.0},
    ]
    assert len(report["results"]) == 1
    assert report["results"][0]["id"] == "q1"
    assert report["results"][0]["elapsed_ms"] == 29.5
