#!/usr/bin/env python3
"""Collect wikigraph query latency reports from a saved query suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable
import urllib.error
import urllib.request

NATIVE_SRC = Path(__file__).resolve().parents[1] / "llm-wiki-native" / "src"
if str(NATIVE_SRC) not in sys.path:
    sys.path.insert(0, str(NATIVE_SRC))

from llm_wiki_native.reports import validate_query_suite_row  # noqa: E402

Timer = Callable[[], float]


def _read_query_suite(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        validate_query_suite_row(row)
    return rows


def _query_suite_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    return ordered[lower] * (upper - rank) + ordered[upper] * (rank - lower)


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "count": len(values),
        "min_ms": round(min(values), 3),
        "p50_ms": round(_percentile(values, 50), 3),
        "p95_ms": round(_percentile(values, 95), 3),
        "max_ms": round(max(values), 3),
    }


def _post_json(url: str, payload: dict[str, Any], *, timeout: int) -> dict[str, Any]:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=raw, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code} {url}: {body}") from exc
    return json.loads(body) if body else {}


def _query_payload(row: dict[str, Any], *, workspace_id: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": row["query"],
        "mode": row["mode"],
        "top_k": int(row["top_k"]),
        "chunk_top_k": int(row["chunk_top_k"]),
    }
    if workspace_id:
        payload["workspace_id"] = workspace_id
    for key in ("query_vector", "section_kind", "record_types", "neighbor_limit", "max_chars_per_block"):
        if key in row:
            payload[key] = row[key]
    return payload


def _request_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "top_k": int(row["top_k"]),
        "chunk_top_k": int(row["chunk_top_k"]),
    }
    vector = row.get("query_vector")
    if isinstance(vector, list):
        metadata["query_vector_dim"] = len(vector)
    for key in ("section_kind", "record_types", "neighbor_limit", "max_chars_per_block"):
        if key in row:
            metadata[key] = row[key]
    return metadata


def _has_explicit_query_vector(row: dict[str, Any]) -> bool:
    vector = row.get("query_vector")
    if not isinstance(vector, list) or not vector:
        return False
    return all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
        for value in vector
    )


def _timing_scope(rows: list[dict[str, Any]]) -> str:
    if rows and all(_has_explicit_query_vector(row) for row in rows):
        return "data_only"
    return "endpoint_includes_embedding"


def collect_query_report(
    *,
    query_suite_path: Path,
    server: str,
    workspace_id: str | None = None,
    endpoint: str = "/query/data",
    measurement_role: str = "native_query",
    timeout: int = 120,
    warmup_runs: int = 0,
    repetitions: int = 1,
    post_json: Callable[..., dict[str, Any]] = _post_json,
    timer: Timer = time.perf_counter,
) -> dict[str, Any]:
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    rows = _read_query_suite(query_suite_path)
    endpoint_path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    url = server.rstrip("/") + endpoint_path
    results: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    elapsed_values: list[float] = []

    for row in rows:
        payload = _query_payload(row, workspace_id=workspace_id)
        for _ in range(warmup_runs):
            post_json(url, payload, timeout=timeout)
        response: dict[str, Any] = {}
        row_elapsed_values: list[float] = []
        for rep in range(1, repetitions + 1):
            started = timer()
            response = post_json(url, payload, timeout=timeout)
            elapsed_ms = round((timer() - started) * 1000.0, 3)
            row_elapsed_values.append(elapsed_ms)
            elapsed_values.append(elapsed_ms)
            samples.append({"id": str(row["id"]), "rep": rep, "elapsed_ms": elapsed_ms})
        results.append(
            {
                "id": str(row["id"]),
                "query": str(row["query"]),
                "mode": str(row["mode"]),
                "elapsed_ms": round(_percentile(row_elapsed_values, 95), 3),
                "request": _request_metadata(row),
                "response": response,
            }
        )

    return {
        "query_suite": str(query_suite_path),
        "query_suite_sha256": _query_suite_sha256(query_suite_path),
        "server": server.rstrip("/"),
        "endpoint": endpoint_path,
        "measurement_role": measurement_role,
        "workspace_id": workspace_id,
        "timing_scope": _timing_scope(rows),
        "warmup_runs": warmup_runs,
        "repetitions": repetitions,
        "summary": _summary(elapsed_values),
        "samples": samples,
        "results": results,
    }


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect wikigraph query report artifacts")
    parser.add_argument("--query-suite", type=Path, required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--workspace-id")
    parser.add_argument("--endpoint", default="/query/data")
    parser.add_argument("--measurement-role", choices=("native_query", "baseline_query"), default="native_query")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        report = collect_query_report(
            query_suite_path=args.query_suite,
            server=args.server,
            workspace_id=args.workspace_id,
            endpoint=args.endpoint,
            measurement_role=args.measurement_role,
            timeout=args.timeout,
            warmup_runs=args.warmup_runs,
            repetitions=args.repetitions,
        )
    except Exception as exc:
        print_json({"ok": False, "error": "collect_failed", "message": str(exc), "exception_type": type(exc).__name__})
        return 1

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print_json(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
