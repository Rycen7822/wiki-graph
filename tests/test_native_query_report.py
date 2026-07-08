from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from support import write_jsonl

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"
SCRIPTS = OPS
sys.path.insert(0, str(ROOT))

from ops import collect_native_query_report  # noqa: E402


def _query_suite_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _query_suite(path: Path) -> None:
    write_jsonl(
        path,
        [
            {
                "id": "q1",
                "query": "alpha evidence",
                "mode": "mix",
                "top_k": 20,
                "must_include_paths": ["a.md"],
                "must_include_entities": [],
                "notes": "fixture",
                "query_vector": [1.0, 0.0],
            },
            {
                "id": "q2",
                "query": "beta evidence",
                "mode": "naive",
                "top_k": 20,
                "must_include_paths": ["b.md"],
                "must_include_entities": [],
                "notes": "fixture",
                "query_vector": [0.0, 1.0],
            },
        ],
    )


def test_native_query_report_collector_posts_suite_rows_and_records_latency(tmp_path: Path) -> None:
    query_suite = tmp_path / "query_suite.jsonl"
    write_jsonl(
        query_suite,
        [
            {
                "id": "q1",
                "query": "alpha evidence",
                "mode": "mix",
                "top_k": 20,
                "must_include_paths": ["a.md"],
                "must_include_entities": [],
                "notes": "fixture",
                "query_vector": [1.0, 0.0],
                "section_kind": "methodology",
                "neighbor_limit": 2,
                "response_profile": "compact",
            },
            {
                "id": "q2",
                "query": "beta evidence",
                "mode": "naive",
                "top_k": 20,
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

    report = collect_native_query_report.collect_query_report(
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
        "query_vector_dim": 2,
        "section_kind": "methodology",
        "neighbor_limit": 2,
        "response_profile": "compact",
    }
    assert calls[0]["url"] == "http://127.0.0.1:19637/query/data"
    assert calls[0]["payload"]["workspace_id"] == "native-test"
    assert calls[0]["payload"]["query_vector"] == [1.0, 0.0]
    assert calls[0]["payload"]["section_kind"] == "methodology"
    assert calls[0]["payload"]["neighbor_limit"] == 2
    assert calls[0]["payload"]["response_profile"] == "compact"
    assert calls[1]["payload"]["mode"] == "naive"


def test_native_query_report_collector_marks_endpoint_embedding_timing_for_missing_query_vectors(tmp_path: Path) -> None:
    for index, query_vector in enumerate([None, []]):
        query_suite = tmp_path / f"query_suite_{index}.jsonl"
        row = {
            "id": "q1",
            "query": "alpha evidence",
            "mode": "mix",
            "top_k": 20,
            "must_include_paths": ["a.md"],
            "must_include_entities": [],
            "notes": "fixture",
        }
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
        [
            {
                "id": "q1",
                "query": "alpha evidence",
                "mode": "mix",
                "top_k": 20,
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
