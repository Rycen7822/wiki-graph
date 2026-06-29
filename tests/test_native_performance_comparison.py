from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_native_performance_comparison  # noqa: E402
import collect_wikigraph_query_report  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _query_suite_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _query_suite(path: Path) -> None:
    _write_jsonl(
        path,
        [
            {
                "id": "q1",
                "query": "alpha",
                "mode": "mix",
                "top_k": 20,
                "chunk_top_k": 10,
                "must_include_paths": ["a.md"],
                "must_include_entities": [],
                "notes": "fixture",
                "query_vector": [1.0, 0.0],
            },
            {
                "id": "q2",
                "query": "beta",
                "mode": "naive",
                "top_k": 20,
                "chunk_top_k": 10,
                "must_include_paths": ["c.md"],
                "must_include_entities": [],
                "notes": "fixture",
                "query_vector": [0.0, 1.0],
            },
        ],
    )


def test_ordered_paths_extracts_source_path_from_saved_baseline_chunks() -> None:
    response = {
        "data": {
            "chunks": [
                {
                    "content": (
                        "[LLM_WIKI_RAW_SECTION]\n"
                        "source_path: raw/clip/2604/example.md\n"
                        "paper_title: Example\n"
                        "[/LLM_WIKI_RAW_SECTION]\n\n"
                        "body"
                    )
                },
                {
                    "content": (
                        "[LLM_WIKI_METHOD_ATOM]\n"
                        "source_path: raw/clip/2606/other.md\n"
                        "[/LLM_WIKI_METHOD_ATOM]\n\n"
                        "body"
                    )
                },
            ]
        }
    }

    assert audit_native_performance_comparison._ordered_paths(response) == [
        "raw/clip/2604/example.md",
        "raw/clip/2606/other.md",
    ]


def _query_report(
    path: Path,
    *,
    native: bool,
    p95_ms: float,
    query_suite_sha256: str | None = None,
    timing_scope: str | None = "data_only",
    measurement_role: str | None = "auto",
    endpoint: str | None = "/query/data",
    baseline_vector_contract: dict | bool | None = None,
) -> None:
    first_paths = ["a.md", "b.md"] if native else ["a.md", "b.md"]
    second_paths = ["c.md", "d.md"] if native else ["c.md", "d.md"]
    first_response = {"source_paths": first_paths}
    second_response = {"context_blocks": [{"source_path": p} for p in second_paths]}
    if native:
        first_response["trace"] = {"retrieval_backend": "zvec", "vector_hit_count": 2}
        second_response["trace"] = {"retrieval_backend": "zvec", "vector_hit_count": 2}
    payload = {
        "summary": {"p95_ms": p95_ms},
        "results": [
            {
                "id": "q1",
                "query": "alpha",
                "mode": "mix",
                "elapsed_ms": p95_ms - 1,
                "request": {
                    "top_k": 20,
                    "chunk_top_k": 10,
                    "query_vector_dim": 2,
                },
                "response": first_response,
            },
            {
                "id": "q2",
                "query": "beta",
                "mode": "naive",
                "elapsed_ms": p95_ms,
                "request": {
                    "top_k": 20,
                    "chunk_top_k": 10,
                    "query_vector_dim": 2,
                },
                "response": second_response,
            },
        ],
    }
    if query_suite_sha256 is not None:
        payload["query_suite_sha256"] = query_suite_sha256
    if timing_scope is not None:
        payload["timing_scope"] = timing_scope
    if measurement_role == "auto":
        payload["measurement_role"] = "native_query" if native else "baseline_query"
    elif measurement_role is not None:
        payload["measurement_role"] = measurement_role
    if endpoint is not None:
        payload["endpoint"] = endpoint
    if not native and baseline_vector_contract is not False:
        payload["baseline_vector_contract"] = baseline_vector_contract or {
            "endpoint": "/query/data",
            "explicit_vector_comparable": True,
            "query_vector_declared": True,
            "blockers": [],
        }
    _write_json(path, payload)


def _refresh_report(path: Path, *, role: str, seconds: float) -> None:
    _write_json(path, {"measurement_role": role, "total_seconds": seconds})


def _add_query_report_samples(path: Path, *, reps_by_query: dict[str, list[int]]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["samples"] = [
        {"id": query_id, "rep": rep, "elapsed_ms": 30.0}
        for query_id, reps in reps_by_query.items()
        for rep in reps
    ]
    _write_json(path, payload)


def test_performance_comparison_fails_closed_when_inputs_are_missing(tmp_path, capsys) -> None:
    result = audit_native_performance_comparison.main(
        [
            "--query-suite",
            str(tmp_path / "missing-query-suite.jsonl"),
            "--baseline-query-report",
            str(tmp_path / "missing-baseline-query.json"),
            "--native-query-report",
            str(tmp_path / "missing-native-query.json"),
            "--baseline-refresh-report",
            str(tmp_path / "missing-baseline-refresh.json"),
            "--native-refresh-report",
            str(tmp_path / "missing-native-refresh.json"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 1
    assert payload["ok"] is False
    assert payload["error"] == "missing_inputs"
    assert len(payload["missing_inputs"]) == 5


def test_performance_comparison_passes_matched_recall_and_three_x_thresholds(tmp_path, capsys) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_suite(query_suite)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=query_suite_digest)
    _query_report(native_query, native=True, p95_ms=90.0, query_suite_sha256=query_suite_digest)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=30.0)

    result = audit_native_performance_comparison.main(
        [
            "--query-suite",
            str(query_suite),
            "--baseline-query-report",
            str(baseline_query),
            "--native-query-report",
            str(native_query),
            "--baseline-refresh-report",
            str(baseline_refresh),
            "--native-refresh-report",
            str(native_refresh),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["ok"] is True
    assert payload["blockers"] == []
    assert payload["source_path_recall_at_20"] == 1.0
    assert payload["timing_scope"] == {
        "required": "data_only",
        "baseline": "data_only",
        "native": "data_only",
    }
    assert payload["retrieval"]["ok"] is True
    assert payload["refresh"]["ok"] is True


def test_performance_comparison_blocks_refresh_reports_without_roles(tmp_path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_suite(query_suite)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=query_suite_digest)
    _query_report(native_query, native=True, p95_ms=90.0, query_suite_sha256=query_suite_digest)
    _write_json(baseline_refresh, {"total_seconds": 120.0})
    _write_json(native_refresh, {"total_seconds": 30.0})

    audit = audit_native_performance_comparison.audit_comparison(
        query_suite_path=query_suite,
        baseline_query_report_path=baseline_query,
        native_query_report_path=native_query,
        baseline_refresh_report_path=baseline_refresh,
        native_refresh_report_path=native_refresh,
    )

    assert audit["ok"] is False
    assert audit["refresh_role_policy"] == {
        "required": {
            "baseline": "baseline_refresh",
            "native": "native_refresh",
        },
        "baseline": None,
        "native": None,
    }
    assert "baseline refresh report missing measurement_role=baseline_refresh" in audit["blockers"]
    assert "native refresh report missing measurement_role=native_refresh" in audit["blockers"]


def test_performance_comparison_blocks_missing_baseline_vector_contract_proof(tmp_path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_suite(query_suite)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(
        baseline_query,
        native=False,
        p95_ms=300.0,
        query_suite_sha256=query_suite_digest,
        baseline_vector_contract=False,
    )
    _query_report(native_query, native=True, p95_ms=90.0, query_suite_sha256=query_suite_digest)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=30.0)

    audit = audit_native_performance_comparison.audit_comparison(
        query_suite_path=query_suite,
        baseline_query_report_path=baseline_query,
        native_query_report_path=native_query,
        baseline_refresh_report_path=baseline_refresh,
        native_refresh_report_path=native_refresh,
    )

    assert audit["ok"] is False
    assert audit["baseline_vector_contract_policy"] == {
        "required": {
            "endpoint": "/query/data",
            "explicit_vector_comparable": True,
            "query_vector_declared": True,
        },
        "endpoint": None,
        "explicit_vector_comparable": None,
        "query_vector_declared": None,
        "blockers": ["baseline vector contract missing explicit_vector_comparable=true"],
    }
    assert "baseline vector contract missing explicit_vector_comparable=true" in audit["blockers"]


def test_performance_comparison_cli_uses_baseline_vector_contract_report(tmp_path, capsys) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    vector_contract = tmp_path / "baseline_vector_contract.json"
    _query_suite(query_suite)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=query_suite_digest)
    _query_report(native_query, native=True, p95_ms=90.0, query_suite_sha256=query_suite_digest)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=30.0)
    _write_json(
        vector_contract,
        {
            "endpoint": "/query/data",
            "explicit_vector_comparable": False,
            "query_vector_declared": False,
            "blockers": ["baseline /query/data request schema does not declare query_vector"],
        },
    )

    result = audit_native_performance_comparison.main(
        [
            "--query-suite",
            str(query_suite),
            "--baseline-query-report",
            str(baseline_query),
            "--native-query-report",
            str(native_query),
            "--baseline-refresh-report",
            str(baseline_refresh),
            "--native-refresh-report",
            str(native_refresh),
            "--baseline-vector-contract-report",
            str(vector_contract),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 1
    assert payload["ok"] is False
    assert payload["baseline_vector_contract_policy"]["query_vector_declared"] is False
    assert payload["baseline_vector_contract_policy"]["explicit_vector_comparable"] is False
    assert "baseline /query/data request schema does not declare query_vector" in payload["blockers"]


def test_performance_comparison_cli_accepts_explicit_baseline_timing_policy(tmp_path, capsys) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    vector_contract = tmp_path / "baseline_vector_contract.json"
    timing_policy = tmp_path / "baseline_timing_policy.json"
    _query_suite(query_suite)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=query_suite_digest)
    _query_report(native_query, native=True, p95_ms=90.0, query_suite_sha256=query_suite_digest)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=30.0)
    _write_json(
        vector_contract,
        {
            "endpoint": "/query/data",
            "explicit_vector_comparable": False,
            "query_vector_declared": False,
            "blockers": ["baseline /query/data request schema does not declare query_vector"],
        },
    )
    _write_json(
        timing_policy,
        {
            "policy_type": "baseline_endpoint_timing",
            "approved": True,
            "baseline_endpoint": "/query/data",
            "applies_to_query_suite_sha256": query_suite_digest,
            "waives_baseline_vector_contract": True,
            "allows_endpoint_includes_embedding": True,
            "approved_by": "test-operator",
            "approved_at": "2026-06-29T14:55:00+08:00",
            "reason": "explicit test approval for endpoint-level baseline timing",
            "known_limitations": ["baseline endpoint does not declare query_vector"],
        },
    )

    result = audit_native_performance_comparison.main(
        [
            "--query-suite",
            str(query_suite),
            "--baseline-query-report",
            str(baseline_query),
            "--native-query-report",
            str(native_query),
            "--baseline-refresh-report",
            str(baseline_refresh),
            "--native-refresh-report",
            str(native_refresh),
            "--baseline-vector-contract-report",
            str(vector_contract),
            "--baseline-timing-policy-report",
            str(timing_policy),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["ok"] is True
    assert payload["blockers"] == []
    assert payload["baseline_vector_contract_policy"]["query_vector_declared"] is False
    assert payload["baseline_timing_policy"]["accepted"] is True
    assert payload["baseline_timing_policy"]["waived_vector_contract_blockers"] == [
        "baseline vector contract missing explicit_vector_comparable=true",
        "baseline vector contract missing query_vector_declared=true",
        "baseline /query/data request schema does not declare query_vector",
    ]


def test_performance_comparison_blocks_missing_or_wrong_query_measurement_roles(tmp_path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_suite(query_suite)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(
        baseline_query,
        native=False,
        p95_ms=300.0,
        query_suite_sha256=query_suite_digest,
        measurement_role=None,
    )
    _query_report(
        native_query,
        native=True,
        p95_ms=90.0,
        query_suite_sha256=query_suite_digest,
        measurement_role="baseline_query",
    )
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=30.0)

    audit = audit_native_performance_comparison.audit_comparison(
        query_suite_path=query_suite,
        baseline_query_report_path=baseline_query,
        native_query_report_path=native_query,
        baseline_refresh_report_path=baseline_refresh,
        native_refresh_report_path=native_refresh,
    )

    assert audit["ok"] is False
    assert audit["query_role_policy"] == {
        "required": {
            "baseline": "baseline_query",
            "native": "native_query",
        },
        "baseline": None,
        "native": "baseline_query",
    }
    assert "baseline query report missing measurement_role=baseline_query" in audit["blockers"]
    assert "native query report measurement_role baseline_query is not native_query" in audit["blockers"]


def test_performance_comparison_blocks_missing_or_wrong_query_endpoints(tmp_path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_suite(query_suite)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(
        baseline_query,
        native=False,
        p95_ms=300.0,
        query_suite_sha256=query_suite_digest,
        endpoint=None,
    )
    _query_report(
        native_query,
        native=True,
        p95_ms=90.0,
        query_suite_sha256=query_suite_digest,
        endpoint="/query",
    )
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=30.0)

    audit = audit_native_performance_comparison.audit_comparison(
        query_suite_path=query_suite,
        baseline_query_report_path=baseline_query,
        native_query_report_path=native_query,
        baseline_refresh_report_path=baseline_refresh,
        native_refresh_report_path=native_refresh,
    )

    assert audit["ok"] is False
    assert audit["query_endpoint_policy"] == {
        "required": "/query/data",
        "baseline": None,
        "native": "/query",
    }
    assert "baseline query report missing endpoint=/query/data" in audit["blockers"]
    assert "native query report endpoint /query is not /query/data" in audit["blockers"]


def test_performance_comparison_rejects_invalid_refresh_seconds(tmp_path, capsys) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_suite(query_suite)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=query_suite_digest)
    _query_report(native_query, native=True, p95_ms=90.0, query_suite_sha256=query_suite_digest)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=-1.0)

    result = audit_native_performance_comparison.main(
        [
            "--query-suite",
            str(query_suite),
            "--baseline-query-report",
            str(baseline_query),
            "--native-query-report",
            str(native_query),
            "--baseline-refresh-report",
            str(baseline_refresh),
            "--native-refresh-report",
            str(native_refresh),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 1
    assert payload["ok"] is False
    assert payload["error"] == "invalid_input"
    assert "positive finite" in payload["message"]


def test_performance_comparison_rejects_invalid_query_p95_ms(tmp_path, capsys) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_suite(query_suite)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=query_suite_digest)
    _query_report(native_query, native=True, p95_ms=-1.0, query_suite_sha256=query_suite_digest)
    native_payload = json.loads(native_query.read_text(encoding="utf-8"))
    native_payload["results"][0]["elapsed_ms"] = 89.0
    native_payload["results"][1]["elapsed_ms"] = 90.0
    _write_json(native_query, native_payload)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=30.0)

    result = audit_native_performance_comparison.main(
        [
            "--query-suite",
            str(query_suite),
            "--baseline-query-report",
            str(baseline_query),
            "--native-query-report",
            str(native_query),
            "--baseline-refresh-report",
            str(baseline_refresh),
            "--native-refresh-report",
            str(native_refresh),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 1
    assert payload["ok"] is False
    assert payload["error"] == "invalid_input"
    assert "p95_ms" in payload["message"]
    assert "positive finite" in payload["message"]


def test_performance_comparison_rejects_summary_p95_below_observed_elapsed_ms(tmp_path, capsys) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_suite(query_suite)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=query_suite_digest)
    _query_report(native_query, native=True, p95_ms=90.0, query_suite_sha256=query_suite_digest)
    native_payload = json.loads(native_query.read_text(encoding="utf-8"))
    native_payload["samples"] = [
        {"id": "q1", "rep": 1, "elapsed_ms": 300.0},
        {"id": "q1", "rep": 2, "elapsed_ms": 320.0},
        {"id": "q2", "rep": 1, "elapsed_ms": 310.0},
        {"id": "q2", "rep": 2, "elapsed_ms": 330.0},
    ]
    _write_json(native_query, native_payload)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=30.0)

    result = audit_native_performance_comparison.main(
        [
            "--query-suite",
            str(query_suite),
            "--baseline-query-report",
            str(baseline_query),
            "--native-query-report",
            str(native_query),
            "--baseline-refresh-report",
            str(baseline_refresh),
            "--native-refresh-report",
            str(native_refresh),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 1
    assert payload["ok"] is False
    assert payload["error"] == "invalid_input"
    assert "summary.p95_ms" in payload["message"]
    assert "observed p95" in payload["message"]


def test_performance_comparison_rejects_summary_p95_below_result_elapsed_ms_when_samples_exist(tmp_path, capsys) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_suite(query_suite)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=query_suite_digest)
    _query_report(native_query, native=True, p95_ms=90.0, query_suite_sha256=query_suite_digest)
    native_payload = json.loads(native_query.read_text(encoding="utf-8"))
    native_payload["results"][0]["elapsed_ms"] = 300.0
    native_payload["results"][1]["elapsed_ms"] = 330.0
    native_payload["samples"] = [
        {"id": "q1", "rep": 1, "elapsed_ms": 80.0},
        {"id": "q2", "rep": 1, "elapsed_ms": 85.0},
    ]
    _write_json(native_query, native_payload)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=30.0)

    result = audit_native_performance_comparison.main(
        [
            "--query-suite",
            str(query_suite),
            "--baseline-query-report",
            str(baseline_query),
            "--native-query-report",
            str(native_query),
            "--baseline-refresh-report",
            str(baseline_refresh),
            "--native-refresh-report",
            str(native_refresh),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 1
    assert payload["ok"] is False
    assert payload["error"] == "invalid_input"
    assert "summary.p95_ms" in payload["message"]
    assert "observed p95" in payload["message"]


def test_performance_comparison_uses_sample_p95_when_summary_is_missing(tmp_path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_suite(query_suite)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=query_suite_digest)
    _query_report(native_query, native=True, p95_ms=90.0, query_suite_sha256=query_suite_digest)
    native_payload = json.loads(native_query.read_text(encoding="utf-8"))
    del native_payload["summary"]
    native_payload["samples"] = [
        {"id": "q1", "rep": 1, "elapsed_ms": 300.0},
        {"id": "q1", "rep": 2, "elapsed_ms": 320.0},
        {"id": "q2", "rep": 1, "elapsed_ms": 310.0},
        {"id": "q2", "rep": 2, "elapsed_ms": 330.0},
    ]
    _write_json(native_query, native_payload)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=30.0)

    audit = audit_native_performance_comparison.audit_comparison(
        query_suite_path=query_suite,
        baseline_query_report_path=baseline_query,
        native_query_report_path=native_query,
        baseline_refresh_report_path=baseline_refresh,
        native_refresh_report_path=native_refresh,
    )

    assert audit["ok"] is False
    assert audit["retrieval"]["native_p95_ms"] == 328.5
    assert audit["retrieval"]["latency_evidence"]["native"]["source"] == "observed"
    assert audit["retrieval"]["latency_evidence"]["native"]["effective_p95_ms"] == 328.5
    assert audit["retrieval"]["latency_evidence"]["native"]["sample_observed_p95_ms"] == 328.5
    assert any(blocker.startswith("retrieval p95 328.500ms exceeds threshold") for blocker in audit["blockers"])


def test_performance_comparison_blocks_unknown_and_duplicate_report_ids(tmp_path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_suite(query_suite)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=query_suite_digest)
    _query_report(native_query, native=True, p95_ms=90.0, query_suite_sha256=query_suite_digest)
    native_payload = json.loads(native_query.read_text(encoding="utf-8"))
    native_payload["results"].append(
        {"id": "q1", "elapsed_ms": 80.0, "response": {"source_paths": ["a.md", "b.md"]}}
    )
    native_payload["results"].append(
        {"id": "unknown-result", "elapsed_ms": 80.0, "response": {"source_paths": ["x.md"]}}
    )
    native_payload["samples"] = [
        {"id": "q1", "rep": 1, "elapsed_ms": 80.0},
        {"id": "q1", "rep": 2, "elapsed_ms": 80.0},
        {"id": "q1", "rep": 3, "elapsed_ms": 80.0},
        {"id": "q2", "rep": 1, "elapsed_ms": 90.0},
        {"id": "q2", "rep": 2, "elapsed_ms": 90.0},
        {"id": "q2", "rep": 3, "elapsed_ms": 90.0},
        {"id": "unknown", "rep": 1, "elapsed_ms": 1.0},
        {"rep": 1, "elapsed_ms": 1.0},
    ]
    _write_json(native_query, native_payload)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=30.0)

    audit = audit_native_performance_comparison.audit_comparison(
        query_suite_path=query_suite,
        baseline_query_report_path=baseline_query,
        native_query_report_path=native_query,
        baseline_refresh_report_path=baseline_refresh,
        native_refresh_report_path=native_refresh,
        min_samples_per_query=3,
    )

    assert audit["ok"] is False
    assert "native query report duplicate result id: q1" in audit["blockers"]
    assert "native query report unknown result id: unknown-result" in audit["blockers"]
    assert "native query report unknown sample id: unknown" in audit["blockers"]
    assert "native query report sample row missing id" in audit["blockers"]


def test_performance_comparison_blocks_missing_required_entities(tmp_path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_suite(query_suite)
    rows = [json.loads(line) for line in query_suite.read_text(encoding="utf-8").splitlines()]
    rows[0]["must_include_entities"] = ["entity:a"]
    _write_jsonl(query_suite, rows)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=query_suite_digest)
    _query_report(native_query, native=True, p95_ms=90.0, query_suite_sha256=query_suite_digest)
    native_payload = json.loads(native_query.read_text(encoding="utf-8"))
    native_payload["results"][0]["response"] = {
        "source_paths": ["a.md", "b.md"],
        "context_blocks": [{"source_path": "a.md", "source_id": "entity:other", "record_id": "entity:other"}],
    }
    _write_json(native_query, native_payload)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=30.0)

    audit = audit_native_performance_comparison.audit_comparison(
        query_suite_path=query_suite,
        baseline_query_report_path=baseline_query,
        native_query_report_path=native_query,
        baseline_refresh_report_path=baseline_refresh,
        native_refresh_report_path=native_refresh,
    )

    assert audit["ok"] is False
    assert "missing required entity for q1: entity:a" in audit["blockers"]


def test_performance_comparison_blocks_result_metadata_mismatch(tmp_path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_suite(query_suite)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=query_suite_digest)
    _query_report(native_query, native=True, p95_ms=90.0, query_suite_sha256=query_suite_digest)
    native_payload = json.loads(native_query.read_text(encoding="utf-8"))
    native_payload["results"][0]["query"] = "wrong query text"
    native_payload["results"][1]["mode"] = "mix"
    _write_json(native_query, native_payload)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=30.0)

    audit = audit_native_performance_comparison.audit_comparison(
        query_suite_path=query_suite,
        baseline_query_report_path=baseline_query,
        native_query_report_path=native_query,
        baseline_refresh_report_path=baseline_refresh,
        native_refresh_report_path=native_refresh,
    )

    assert audit["ok"] is False
    assert "native query report result q1 query metadata mismatch" in audit["blockers"]
    assert "native query report result q2 mode metadata mismatch" in audit["blockers"]


def test_performance_comparison_blocks_missing_result_metadata(tmp_path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_suite(query_suite)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=query_suite_digest)
    _query_report(native_query, native=True, p95_ms=90.0, query_suite_sha256=query_suite_digest)
    native_payload = json.loads(native_query.read_text(encoding="utf-8"))
    native_payload["results"][0].pop("mode", None)
    native_payload["results"][1].pop("query", None)
    _write_json(native_query, native_payload)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=30.0)

    audit = audit_native_performance_comparison.audit_comparison(
        query_suite_path=query_suite,
        baseline_query_report_path=baseline_query,
        native_query_report_path=native_query,
        baseline_refresh_report_path=baseline_refresh,
        native_refresh_report_path=native_refresh,
    )

    assert audit["ok"] is False
    assert "native query report result q1 mode metadata missing" in audit["blockers"]
    assert "native query report result q2 query metadata missing" in audit["blockers"]


def test_performance_comparison_blocks_missing_or_wrong_request_metadata(tmp_path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    _query_suite(query_suite)
    suite_hash = _query_suite_sha256(query_suite)
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=suite_hash)
    _query_report(native_query, native=True, p95_ms=80.0, query_suite_sha256=suite_hash)
    payload = json.loads(native_query.read_text(encoding="utf-8"))
    del payload["results"][0]["request"]["chunk_top_k"]
    payload["results"][1]["request"]["top_k"] = 5
    payload["results"][1]["request"]["query_vector_dim"] = 3
    _write_json(native_query, payload)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=300.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=80.0)
    _add_query_report_samples(baseline_query, reps_by_query={"q1": [1, 2, 3], "q2": [1, 2, 3]})
    _add_query_report_samples(native_query, reps_by_query={"q1": [1, 2, 3], "q2": [1, 2, 3]})

    audit = audit_native_performance_comparison.audit_comparison(
        query_suite_path=query_suite,
        baseline_query_report_path=baseline_query,
        native_query_report_path=native_query,
        baseline_refresh_report_path=baseline_refresh,
        native_refresh_report_path=native_refresh,
        min_samples_per_query=3,
    )

    assert audit["ok"] is False
    assert "native query report result q1 request.chunk_top_k metadata missing" in audit["blockers"]
    assert "native query report result q2 request.top_k metadata mismatch" in audit["blockers"]
    assert "native query report result q2 request.query_vector_dim metadata mismatch" in audit["blockers"]


def test_performance_comparison_blocks_missing_or_wrong_native_backend_trace(tmp_path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    _query_suite(query_suite)
    suite_hash = _query_suite_sha256(query_suite)
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=suite_hash)
    _query_report(native_query, native=True, p95_ms=80.0, query_suite_sha256=suite_hash)
    payload = json.loads(native_query.read_text(encoding="utf-8"))
    del payload["results"][0]["response"]["trace"]
    payload["results"][1]["response"]["trace"]["retrieval_backend"] = "sqlite"
    _write_json(native_query, payload)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=300.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=80.0)
    _add_query_report_samples(baseline_query, reps_by_query={"q1": [1, 2, 3], "q2": [1, 2, 3]})
    _add_query_report_samples(native_query, reps_by_query={"q1": [1, 2, 3], "q2": [1, 2, 3]})

    audit = audit_native_performance_comparison.audit_comparison(
        query_suite_path=query_suite,
        baseline_query_report_path=baseline_query,
        native_query_report_path=native_query,
        baseline_refresh_report_path=baseline_refresh,
        native_refresh_report_path=native_refresh,
        min_samples_per_query=3,
    )

    assert audit["ok"] is False
    assert "native query report result q1 response trace retrieval_backend missing" in audit["blockers"]
    assert "native query report result q2 response trace retrieval_backend sqlite is not zvec" in audit["blockers"]


def test_performance_comparison_reports_native_backend_trace_per_query(tmp_path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    _query_suite(query_suite)
    suite_hash = _query_suite_sha256(query_suite)
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=suite_hash)
    _query_report(native_query, native=True, p95_ms=80.0, query_suite_sha256=suite_hash)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=300.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=80.0)
    _add_query_report_samples(baseline_query, reps_by_query={"q1": [1, 2, 3], "q2": [1, 2, 3]})
    _add_query_report_samples(native_query, reps_by_query={"q1": [1, 2, 3], "q2": [1, 2, 3]})

    audit = audit_native_performance_comparison.audit_comparison(
        query_suite_path=query_suite,
        baseline_query_report_path=baseline_query,
        native_query_report_path=native_query,
        baseline_refresh_report_path=baseline_refresh,
        native_refresh_report_path=native_refresh,
        min_samples_per_query=3,
    )

    assert audit["ok"] is True
    assert [row["native_retrieval_backend"] for row in audit["per_query"]] == ["zvec", "zvec"]


def test_performance_comparison_reports_distinct_sample_counts_per_query(tmp_path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    _query_suite(query_suite)
    suite_hash = _query_suite_sha256(query_suite)
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=suite_hash)
    _query_report(native_query, native=True, p95_ms=80.0, query_suite_sha256=suite_hash)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=300.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=80.0)
    _add_query_report_samples(baseline_query, reps_by_query={"q1": [1, 2, 3], "q2": [1, 2, 3]})
    _add_query_report_samples(native_query, reps_by_query={"q1": [1, 2, 3], "q2": [1, 2, 3]})

    audit = audit_native_performance_comparison.audit_comparison(
        query_suite_path=query_suite,
        baseline_query_report_path=baseline_query,
        native_query_report_path=native_query,
        baseline_refresh_report_path=baseline_refresh,
        native_refresh_report_path=native_refresh,
        min_samples_per_query=3,
    )

    assert audit["ok"] is True
    assert [row["baseline_distinct_sample_count"] for row in audit["per_query"]] == [3, 3]
    assert [row["native_distinct_sample_count"] for row in audit["per_query"]] == [3, 3]


def test_performance_comparison_reports_latency_evidence(tmp_path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    _query_suite(query_suite)
    suite_hash = _query_suite_sha256(query_suite)
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=suite_hash)
    _query_report(native_query, native=True, p95_ms=80.0, query_suite_sha256=suite_hash)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=300.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=80.0)
    _add_query_report_samples(baseline_query, reps_by_query={"q1": [1, 2, 3], "q2": [1, 2, 3]})
    _add_query_report_samples(native_query, reps_by_query={"q1": [1, 2, 3], "q2": [1, 2, 3]})

    audit = audit_native_performance_comparison.audit_comparison(
        query_suite_path=query_suite,
        baseline_query_report_path=baseline_query,
        native_query_report_path=native_query,
        baseline_refresh_report_path=baseline_refresh,
        native_refresh_report_path=native_refresh,
        min_samples_per_query=3,
    )

    assert audit["ok"] is True
    evidence = audit["retrieval"]["latency_evidence"]
    assert evidence["baseline"]["source"] == "summary"
    assert evidence["baseline"]["summary_p95_ms"] == 300.0
    assert evidence["baseline"]["effective_p95_ms"] == 300.0
    assert evidence["baseline"]["result_observed_p95_ms"] == 299.95
    assert evidence["native"]["source"] == "summary"
    assert evidence["native"]["summary_p95_ms"] == 80.0
    assert evidence["native"]["effective_p95_ms"] == 80.0
    assert evidence["native"]["sample_observed_p95_ms"] == 30.0


def test_performance_comparison_rejects_invalid_result_elapsed_ms_with_valid_summary(tmp_path, capsys) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_suite(query_suite)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=query_suite_digest)
    _query_report(native_query, native=True, p95_ms=90.0, query_suite_sha256=query_suite_digest)
    native_payload = json.loads(native_query.read_text(encoding="utf-8"))
    native_payload["results"][0]["elapsed_ms"] = -1.0
    _write_json(native_query, native_payload)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=30.0)

    result = audit_native_performance_comparison.main(
        [
            "--query-suite",
            str(query_suite),
            "--baseline-query-report",
            str(baseline_query),
            "--native-query-report",
            str(native_query),
            "--baseline-refresh-report",
            str(baseline_refresh),
            "--native-refresh-report",
            str(native_refresh),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 1
    assert payload["ok"] is False
    assert payload["error"] == "invalid_input"
    assert "results[].elapsed_ms" in payload["message"]
    assert "positive finite" in payload["message"]


def test_performance_comparison_rejects_missing_result_elapsed_ms_with_valid_summary(tmp_path, capsys) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_suite(query_suite)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=query_suite_digest)
    _query_report(native_query, native=True, p95_ms=90.0, query_suite_sha256=query_suite_digest)
    native_payload = json.loads(native_query.read_text(encoding="utf-8"))
    del native_payload["results"][0]["elapsed_ms"]
    _write_json(native_query, native_payload)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=30.0)

    result = audit_native_performance_comparison.main(
        [
            "--query-suite",
            str(query_suite),
            "--baseline-query-report",
            str(baseline_query),
            "--native-query-report",
            str(native_query),
            "--baseline-refresh-report",
            str(baseline_refresh),
            "--native-refresh-report",
            str(native_refresh),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 1
    assert payload["ok"] is False
    assert payload["error"] == "invalid_input"
    assert "results[].elapsed_ms" in payload["message"]
    assert "required" in payload["message"]


def test_performance_comparison_rejects_invalid_sample_elapsed_ms_with_valid_counts(tmp_path, capsys) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_suite(query_suite)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=query_suite_digest)
    _query_report(native_query, native=True, p95_ms=90.0, query_suite_sha256=query_suite_digest)
    for report_path in (baseline_query, native_query):
        _add_query_report_samples(report_path, reps_by_query={"q1": [1, 2, 3], "q2": [1, 2, 3]})
    native_payload = json.loads(native_query.read_text(encoding="utf-8"))
    native_payload["samples"][0]["elapsed_ms"] = -1.0
    _write_json(native_query, native_payload)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=30.0)

    result = audit_native_performance_comparison.main(
        [
            "--query-suite",
            str(query_suite),
            "--baseline-query-report",
            str(baseline_query),
            "--native-query-report",
            str(native_query),
            "--baseline-refresh-report",
            str(baseline_refresh),
            "--native-refresh-report",
            str(native_refresh),
            "--min-samples-per-query",
            "3",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 1
    assert payload["ok"] is False
    assert payload["error"] == "invalid_input"
    assert "samples[].elapsed_ms" in payload["message"]
    assert "positive finite" in payload["message"]


def test_performance_comparison_rejects_missing_sample_elapsed_ms_with_valid_counts(tmp_path, capsys) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_suite(query_suite)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=query_suite_digest)
    _query_report(native_query, native=True, p95_ms=90.0, query_suite_sha256=query_suite_digest)
    for report_path in (baseline_query, native_query):
        _add_query_report_samples(report_path, reps_by_query={"q1": [1, 2, 3], "q2": [1, 2, 3]})
    native_payload = json.loads(native_query.read_text(encoding="utf-8"))
    del native_payload["samples"][0]["elapsed_ms"]
    _write_json(native_query, native_payload)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=30.0)

    result = audit_native_performance_comparison.main(
        [
            "--query-suite",
            str(query_suite),
            "--baseline-query-report",
            str(baseline_query),
            "--native-query-report",
            str(native_query),
            "--baseline-refresh-report",
            str(baseline_refresh),
            "--native-refresh-report",
            str(native_refresh),
            "--min-samples-per-query",
            "3",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 1
    assert payload["ok"] is False
    assert payload["error"] == "invalid_input"
    assert "samples[].elapsed_ms" in payload["message"]
    assert "required" in payload["message"]


def test_performance_comparison_counts_distinct_sample_repetitions(tmp_path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_suite(query_suite)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=query_suite_digest)
    _query_report(native_query, native=True, p95_ms=90.0, query_suite_sha256=query_suite_digest)
    _add_query_report_samples(baseline_query, reps_by_query={"q1": [1, 2, 3], "q2": [1, 2, 3]})
    _add_query_report_samples(native_query, reps_by_query={"q1": [1, 1, 2], "q2": [1, 2, 3]})
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=30.0)

    audit = audit_native_performance_comparison.audit_comparison(
        query_suite_path=query_suite,
        baseline_query_report_path=baseline_query,
        native_query_report_path=native_query,
        baseline_refresh_report_path=baseline_refresh,
        native_refresh_report_path=native_refresh,
        min_samples_per_query=3,
    )

    assert audit["ok"] is False
    assert "native samples for q1 2 below required 3" in audit["blockers"]


def test_performance_comparison_rejects_missing_sample_rep_with_valid_counts(tmp_path, capsys) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_suite(query_suite)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=query_suite_digest)
    _query_report(native_query, native=True, p95_ms=90.0, query_suite_sha256=query_suite_digest)
    _add_query_report_samples(baseline_query, reps_by_query={"q1": [1, 2, 3], "q2": [1, 2, 3]})
    _add_query_report_samples(native_query, reps_by_query={"q1": [1, 2, 3], "q2": [1, 2, 3]})
    native_payload = json.loads(native_query.read_text(encoding="utf-8"))
    del native_payload["samples"][0]["rep"]
    _write_json(native_query, native_payload)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=30.0)

    result = audit_native_performance_comparison.main(
        [
            "--query-suite",
            str(query_suite),
            "--baseline-query-report",
            str(baseline_query),
            "--native-query-report",
            str(native_query),
            "--baseline-refresh-report",
            str(baseline_refresh),
            "--native-refresh-report",
            str(native_refresh),
            "--min-samples-per-query",
            "3",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 1
    assert payload["ok"] is False
    assert payload["error"] == "invalid_input"
    assert "samples[].rep" in payload["message"]
    assert "required" in payload["message"]


def test_performance_comparison_reports_recall_and_threshold_blockers(tmp_path, capsys) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_suite(query_suite)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=query_suite_digest)
    _write_json(
        native_query,
        {
            "query_suite_sha256": query_suite_digest,
            "summary": {"p95_ms": 150.0},
            "results": [
                {"id": "q1", "elapsed_ms": 140.0, "response": {"source_paths": ["x.md"]}},
                {"id": "q2", "elapsed_ms": 150.0, "response": {"source_paths": ["c.md"]}},
            ],
        },
    )
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=50.0)

    result = audit_native_performance_comparison.main(
        [
            "--query-suite",
            str(query_suite),
            "--baseline-query-report",
            str(baseline_query),
            "--native-query-report",
            str(native_query),
            "--baseline-refresh-report",
            str(baseline_refresh),
            "--native-refresh-report",
            str(native_refresh),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 1
    assert payload["ok"] is False
    assert "missing required path for q1: a.md" in payload["blockers"]
    assert any(blocker.startswith("source_path Recall@20") for blocker in payload["blockers"])
    assert any(blocker.startswith("retrieval p95") for blocker in payload["blockers"])
    assert any(blocker.startswith("refresh seconds") for blocker in payload["blockers"])


def test_wikigraph_query_report_collector_posts_suite_rows_and_records_latency(tmp_path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    _write_jsonl(
        query_suite,
        [
            {
                "id": "q1",
                "query": "alpha evidence",
                "mode": "mix",
                "top_k": 20,
                "chunk_top_k": 10,
                "must_include_paths": ["a.md"],
                "must_include_entities": [],
                "notes": "fixture",
                "query_vector": [1.0, 0.0],
                "section_kind": "methodology",
                "neighbor_limit": 2,
            },
            {
                "id": "q2",
                "query": "beta evidence",
                "mode": "naive",
                "top_k": 20,
                "chunk_top_k": 10,
                "must_include_paths": ["b.md"],
                "must_include_entities": [],
                "notes": "fixture",
                "query_vector": [0.0, 1.0],
            },
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

    report = collect_wikigraph_query_report.collect_query_report(
        query_suite_path=query_suite,
        server="http://127.0.0.1:19637",
        workspace_id="native-test",
        post_json=fake_post,
        timer=lambda: next(ticks),
    )

    assert report["query_suite"] == str(query_suite)
    assert report["query_suite_sha256"] == _query_suite_sha256(query_suite)
    assert report["server"] == "http://127.0.0.1:19637"
    assert report["endpoint"] == "/query/data"
    assert report["measurement_role"] == "native_query"
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
        "chunk_top_k": 10,
        "query_vector_dim": 2,
        "section_kind": "methodology",
        "neighbor_limit": 2,
    }
    assert report["results"][0]["response"]["source_paths"] == ["a.md"]
    assert calls[0]["url"] == "http://127.0.0.1:19637/query/data"
    assert calls[0]["payload"]["workspace_id"] == "native-test"
    assert calls[0]["payload"]["chunk_top_k"] == 10
    assert calls[0]["payload"]["query_vector"] == [1.0, 0.0]
    assert calls[0]["payload"]["section_kind"] == "methodology"
    assert calls[0]["payload"]["neighbor_limit"] == 2
    assert calls[1]["payload"]["mode"] == "naive"


def test_wikigraph_query_report_collector_marks_endpoint_embedding_timing_when_vector_is_missing(tmp_path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    _write_jsonl(
        query_suite,
        [
            {
                "id": "q1",
                "query": "alpha evidence",
                "mode": "mix",
                "top_k": 20,
                "chunk_top_k": 10,
                "must_include_paths": ["a.md"],
                "must_include_entities": [],
                "notes": "fixture",
            }
        ],
    )

    def fake_post(_url: str, _payload: dict, *, timeout: int) -> dict:
        return {"source_paths": ["a.md"]}

    report = collect_wikigraph_query_report.collect_query_report(
        query_suite_path=query_suite,
        server="http://127.0.0.1:19637",
        post_json=fake_post,
        timer=iter([1.00, 1.01]).__next__,
    )

    assert report["timing_scope"] == "endpoint_includes_embedding"


def test_wikigraph_query_report_collector_allows_baseline_measurement_role(tmp_path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    _query_suite(query_suite)
    responses = [
        {"source_paths": ["a.md", "b.md"], "trace": {"retrieval_backend": "zvec", "vector_hit_count": 2}},
        {
            "context_blocks": [{"source_path": "c.md"}, {"source_path": "d.md"}],
            "trace": {"retrieval_backend": "zvec", "vector_hit_count": 2},
        },
    ]

    def fake_post(_url: str, _payload: dict, *, timeout: int) -> dict:
        return responses.pop(0)

    report = collect_wikigraph_query_report.collect_query_report(
        query_suite_path=query_suite,
        server="http://127.0.0.1:19637",
        measurement_role="baseline_query",
        post_json=fake_post,
        timer=iter([1.00, 1.02, 2.00, 2.03]).__next__,
    )

    assert report["measurement_role"] == "baseline_query"
    assert report["timing_scope"] == "data_only"


def test_wikigraph_query_report_collector_output_feeds_performance_auditor(tmp_path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    _query_suite(query_suite)
    responses = [
        {"source_paths": ["a.md", "b.md"], "trace": {"retrieval_backend": "zvec", "vector_hit_count": 2}},
        {
            "context_blocks": [{"source_path": "c.md"}, {"source_path": "d.md"}],
            "trace": {"retrieval_backend": "zvec", "vector_hit_count": 2},
        },
    ]
    ticks = iter([1.00, 1.02, 2.00, 2.03])

    def fake_post(_url: str, _payload: dict, *, timeout: int) -> dict:
        return responses.pop(0)

    native_report = collect_wikigraph_query_report.collect_query_report(
        query_suite_path=query_suite,
        server="http://127.0.0.1:19637",
        post_json=fake_post,
        timer=lambda: next(ticks),
    )
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=_query_suite_sha256(query_suite))
    _write_json(native_query, native_report)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=30.0)

    audit = audit_native_performance_comparison.audit_comparison(
        query_suite_path=query_suite,
        baseline_query_report_path=baseline_query,
        native_query_report_path=native_query,
        baseline_refresh_report_path=baseline_refresh,
        native_refresh_report_path=native_refresh,
    )

    assert audit["ok"] is True
    assert audit["blockers"] == []
    assert audit["retrieval"]["native_p95_ms"] == 29.5


def test_performance_comparison_blocks_missing_query_suite_fingerprint(tmp_path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_suite(query_suite)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(baseline_query, native=False, p95_ms=300.0)
    _query_report(native_query, native=True, p95_ms=90.0, query_suite_sha256=query_suite_digest)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=30.0)

    audit = audit_native_performance_comparison.audit_comparison(
        query_suite_path=query_suite,
        baseline_query_report_path=baseline_query,
        native_query_report_path=native_query,
        baseline_refresh_report_path=baseline_refresh,
        native_refresh_report_path=native_refresh,
    )

    assert audit["ok"] is False
    assert "baseline query report missing query_suite_sha256" in audit["blockers"]


def test_performance_comparison_blocks_query_suite_fingerprint_mismatch(tmp_path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_suite(query_suite)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256="mismatched-suite")
    _query_report(native_query, native=True, p95_ms=90.0, query_suite_sha256=query_suite_digest)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=30.0)

    audit = audit_native_performance_comparison.audit_comparison(
        query_suite_path=query_suite,
        baseline_query_report_path=baseline_query,
        native_query_report_path=native_query,
        baseline_refresh_report_path=baseline_refresh,
        native_refresh_report_path=native_refresh,
    )

    assert audit["ok"] is False
    assert "baseline query report query_suite_sha256 mismatch" in audit["blockers"]


def test_performance_comparison_blocks_reports_without_data_only_timing_scope(tmp_path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_suite(query_suite)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(
        baseline_query,
        native=False,
        p95_ms=300.0,
        query_suite_sha256=query_suite_digest,
        timing_scope=None,
    )
    _query_report(
        native_query,
        native=True,
        p95_ms=90.0,
        query_suite_sha256=query_suite_digest,
        timing_scope="endpoint_includes_embedding",
    )
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=30.0)

    audit = audit_native_performance_comparison.audit_comparison(
        query_suite_path=query_suite,
        baseline_query_report_path=baseline_query,
        native_query_report_path=native_query,
        baseline_refresh_report_path=baseline_refresh,
        native_refresh_report_path=native_refresh,
    )

    assert audit["ok"] is False
    assert "baseline query report missing timing_scope=data_only" in audit["blockers"]
    assert "native query report timing_scope endpoint_includes_embedding is not data_only" in audit["blockers"]


def test_performance_comparison_blocks_data_only_reports_when_query_suite_lacks_explicit_vectors(tmp_path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _write_jsonl(
        query_suite,
        [
            {
                "id": "q1",
                "query": "alpha",
                "mode": "mix",
                "top_k": 20,
                "chunk_top_k": 10,
                "must_include_paths": ["a.md"],
                "must_include_entities": [],
                "notes": "fixture",
            },
            {
                "id": "q2",
                "query": "beta",
                "mode": "naive",
                "top_k": 20,
                "chunk_top_k": 10,
                "must_include_paths": ["c.md"],
                "must_include_entities": [],
                "notes": "fixture",
                "query_vector": [],
            },
            {
                "id": "q3",
                "query": "gamma",
                "mode": "mix",
                "top_k": 20,
                "chunk_top_k": 10,
                "must_include_paths": ["e.md"],
                "must_include_entities": [],
                "notes": "fixture",
                "query_vector": [True],
            },
        ],
    )
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=query_suite_digest)
    _query_report(native_query, native=True, p95_ms=90.0, query_suite_sha256=query_suite_digest)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=30.0)

    audit = audit_native_performance_comparison.audit_comparison(
        query_suite_path=query_suite,
        baseline_query_report_path=baseline_query,
        native_query_report_path=native_query,
        baseline_refresh_report_path=baseline_refresh,
        native_refresh_report_path=native_refresh,
    )

    assert audit["ok"] is False
    assert audit["query_vector_policy"] == {
        "required_for_timing_scope": "data_only",
        "total": 3,
        "valid_explicit_query_vector_count": 0,
        "missing_or_invalid_ids": ["q1", "q2", "q3"],
    }
    assert "query suite explicit query_vector coverage 0/3 is required for data_only timing" in audit["blockers"]


def test_wikigraph_query_report_collector_records_repeated_samples_without_duplicate_results(tmp_path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    _write_jsonl(
        query_suite,
        [
            {
                "id": "q1",
                "query": "alpha evidence",
                "mode": "mix",
                "top_k": 20,
                "chunk_top_k": 10,
                "must_include_paths": ["a.md"],
                "must_include_entities": [],
                "notes": "fixture",
                "query_vector": [1.0, 0.0],
            }
        ],
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

    report = collect_wikigraph_query_report.collect_query_report(
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
    assert report["results"][0]["response"]["source_paths"] == ["a.md"]


def test_wikigraph_query_report_collector_requires_non_empty_query_vectors_for_data_only_scope(tmp_path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    _write_jsonl(
        query_suite,
        [
            {
                "id": "q1",
                "query": "alpha evidence",
                "mode": "mix",
                "top_k": 20,
                "chunk_top_k": 10,
                "must_include_paths": ["a.md"],
                "must_include_entities": [],
                "notes": "fixture",
                "query_vector": [],
            }
        ],
    )

    def fake_post(_url: str, _payload: dict, *, timeout: int) -> dict:
        return {"source_paths": ["a.md"]}

    report = collect_wikigraph_query_report.collect_query_report(
        query_suite_path=query_suite,
        server="http://127.0.0.1:19637",
        post_json=fake_post,
        timer=iter([1.00, 1.01]).__next__,
    )

    assert report["timing_scope"] == "endpoint_includes_embedding"


def test_performance_comparison_blocks_reports_with_too_few_samples(tmp_path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_suite(query_suite)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=query_suite_digest)
    _query_report(native_query, native=True, p95_ms=90.0, query_suite_sha256=query_suite_digest)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=30.0)

    audit = audit_native_performance_comparison.audit_comparison(
        query_suite_path=query_suite,
        baseline_query_report_path=baseline_query,
        native_query_report_path=native_query,
        baseline_refresh_report_path=baseline_refresh,
        native_refresh_report_path=native_refresh,
        min_samples_per_query=3,
    )

    assert audit["ok"] is False
    assert "baseline samples for q1 1 below required 3" in audit["blockers"]
    assert "native samples for q1 1 below required 3" in audit["blockers"]
    assert "baseline samples for q2 1 below required 3" in audit["blockers"]
    assert "native samples for q2 1 below required 3" in audit["blockers"]


def test_performance_comparison_cli_accepts_min_samples_per_query(tmp_path, capsys) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    baseline_query = tmp_path / "baseline_query.json"
    native_query = tmp_path / "native_query.json"
    baseline_refresh = tmp_path / "baseline_refresh.json"
    native_refresh = tmp_path / "native_refresh.json"
    _query_suite(query_suite)
    query_suite_digest = _query_suite_sha256(query_suite)
    _query_report(baseline_query, native=False, p95_ms=300.0, query_suite_sha256=query_suite_digest)
    _query_report(native_query, native=True, p95_ms=90.0, query_suite_sha256=query_suite_digest)
    _refresh_report(baseline_refresh, role="baseline_refresh", seconds=120.0)
    _refresh_report(native_refresh, role="native_refresh", seconds=30.0)

    result = audit_native_performance_comparison.main(
        [
            "--query-suite",
            str(query_suite),
            "--baseline-query-report",
            str(baseline_query),
            "--native-query-report",
            str(native_query),
            "--baseline-refresh-report",
            str(baseline_refresh),
            "--native-refresh-report",
            str(native_refresh),
            "--min-samples-per-query",
            "3",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 1
    assert payload["ok"] is False
    assert payload["sample_policy"]["min_samples_per_query"] == 3
    assert any(blocker.startswith("baseline samples for q1") for blocker in payload["blockers"])
