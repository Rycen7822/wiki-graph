#!/usr/bin/env python3
"""Audit saved native-vs-baseline performance artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

NATIVE_SRC = Path(__file__).resolve().parents[1] / "llm-wiki-native" / "src"
if str(NATIVE_SRC) not in sys.path:
    sys.path.insert(0, str(NATIVE_SRC))

from llm_wiki_native.reports import validate_query_suite_row  # noqa: E402

REQUIRED_RECALL_AT_20 = 0.75
SPEEDUP_FACTOR = 3.0
P95_CONSISTENCY_TOLERANCE_MS = 0.001
SOURCE_PATH_PREFIX = "source_path:"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_query_suite(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        validate_query_suite_row(row)
    return rows


def _query_suite_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _embedded_source_paths(text: Any) -> list[str]:
    if not isinstance(text, str):
        return []
    paths: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(SOURCE_PATH_PREFIX):
            path = stripped[len(SOURCE_PATH_PREFIX) :].strip()
            if path:
                paths.append(path)
    return paths


def _ordered_paths(response: dict[str, Any], *, limit: int = 20) -> list[str]:
    paths: list[str] = []
    for path in response.get("source_paths", []) or []:
        if path:
            paths.append(str(path))
    for block in response.get("context_blocks", []) or []:
        if not isinstance(block, dict):
            continue
        if block.get("source_path"):
            paths.append(str(block["source_path"]))
        paths.extend(_embedded_source_paths(block.get("text")))
        paths.extend(_embedded_source_paths(block.get("content")))
    data = response.get("data")
    chunks = data.get("chunks", []) if isinstance(data, dict) else []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        paths.extend(_embedded_source_paths(chunk.get("content")))
    deduped = list(dict.fromkeys(paths))
    return deduped[:limit]


def _response_entities(response: dict[str, Any]) -> set[str]:
    entities: set[str] = set()
    for block in response.get("context_blocks", []) or []:
        if not isinstance(block, dict):
            continue
        if block.get("source_id"):
            entities.add(str(block["source_id"]))
        if block.get("record_id"):
            entities.add(str(block["record_id"]))
    for hit in response.get("hits", []) or []:
        if isinstance(hit, dict) and hit.get("record_id"):
            entities.add(str(hit["record_id"]))
    return entities


def _results_by_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results = report.get("results", [])
    if not isinstance(results, list):
        raise ValueError("query report results must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    for row in results:
        if isinstance(row, dict) and row.get("id"):
            indexed[str(row["id"])] = row
    return indexed


def _report_id_policy(*, report: dict[str, Any], query_ids: set[str]) -> dict[str, Any]:
    results = report.get("results", [])
    samples = report.get("samples", [])
    result_ids: list[str] = []
    duplicate_result_ids: list[str] = []
    unknown_result_ids: list[str] = []
    missing_result_id_rows = 0
    non_object_result_rows = 0
    seen_result_ids: set[str] = set()
    for row in results if isinstance(results, list) else []:
        if not isinstance(row, dict):
            non_object_result_rows += 1
            continue
        if not row.get("id"):
            missing_result_id_rows += 1
            continue
        query_id = str(row["id"])
        result_ids.append(query_id)
        if query_id in seen_result_ids:
            duplicate_result_ids.append(query_id)
        seen_result_ids.add(query_id)
        if query_id not in query_ids:
            unknown_result_ids.append(query_id)

    unknown_sample_ids: list[str] = []
    missing_sample_id_rows = 0
    non_object_sample_rows = 0
    sample_count = 0
    for row in samples if isinstance(samples, list) else []:
        if not isinstance(row, dict):
            non_object_sample_rows += 1
            continue
        sample_count += 1
        if not row.get("id"):
            missing_sample_id_rows += 1
            continue
        query_id = str(row["id"])
        if query_id not in query_ids:
            unknown_sample_ids.append(query_id)

    return {
        "query_count": len(query_ids),
        "result_count": len(result_ids),
        "sample_count": sample_count,
        "missing_result_ids": sorted(query_ids - set(result_ids)),
        "duplicate_result_ids": sorted(set(duplicate_result_ids)),
        "unknown_result_ids": sorted(set(unknown_result_ids)),
        "missing_result_id_rows": missing_result_id_rows,
        "non_object_result_rows": non_object_result_rows,
        "unknown_sample_ids": sorted(set(unknown_sample_ids)),
        "missing_sample_id_rows": missing_sample_id_rows,
        "non_object_sample_rows": non_object_sample_rows,
    }


def _add_report_id_blockers(*, label: str, policy: dict[str, Any], blockers: list[str]) -> None:
    for query_id in policy["missing_result_ids"]:
        blockers.append(f"{label} query report missing result id: {query_id}")
    for query_id in policy["duplicate_result_ids"]:
        blockers.append(f"{label} query report duplicate result id: {query_id}")
    for query_id in policy["unknown_result_ids"]:
        blockers.append(f"{label} query report unknown result id: {query_id}")
    for query_id in policy["unknown_sample_ids"]:
        blockers.append(f"{label} query report unknown sample id: {query_id}")
    if policy["missing_result_id_rows"]:
        blockers.append(f"{label} query report result row missing id")
    if policy["missing_sample_id_rows"]:
        blockers.append(f"{label} query report sample row missing id")
    if policy["non_object_result_rows"]:
        blockers.append(f"{label} query report result row must be an object")
    if policy["non_object_sample_rows"]:
        blockers.append(f"{label} query report sample row must be an object")


def _add_result_metadata_blockers(
    *,
    label: str,
    report: dict[str, Any],
    query_suite_by_id: dict[str, dict[str, Any]],
    blockers: list[str],
) -> None:
    results = report.get("results", [])
    if not isinstance(results, list):
        return
    for row in results:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        query_id = str(row["id"])
        expected = query_suite_by_id.get(query_id)
        if expected is None:
            continue
        if row.get("query") is None:
            blockers.append(f"{label} query report result {query_id} query metadata missing")
        elif str(row["query"]) != str(expected.get("query", "")):
            blockers.append(f"{label} query report result {query_id} query metadata mismatch")
        if row.get("mode") is None:
            blockers.append(f"{label} query report result {query_id} mode metadata missing")
        elif str(row["mode"]) != str(expected.get("mode", "")):
            blockers.append(f"{label} query report result {query_id} mode metadata mismatch")


def _expected_request_metadata(row: dict[str, Any]) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "top_k": int(row["top_k"]),
        "chunk_top_k": int(row["chunk_top_k"]),
    }
    vector = row.get("query_vector")
    if isinstance(vector, list):
        expected["query_vector_dim"] = len(vector)
    for key in ("section_kind", "record_types", "neighbor_limit", "max_chars_per_block"):
        if key in row:
            expected[key] = row[key]
    return expected


def _add_result_request_metadata_blockers(
    *,
    label: str,
    report: dict[str, Any],
    query_suite_by_id: dict[str, dict[str, Any]],
    blockers: list[str],
) -> None:
    results = report.get("results", [])
    if not isinstance(results, list):
        return
    for row in results:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        query_id = str(row["id"])
        expected_row = query_suite_by_id.get(query_id)
        if expected_row is None:
            continue
        request = row.get("request")
        if not isinstance(request, dict):
            blockers.append(f"{label} query report result {query_id} request metadata missing")
            continue
        for key, expected in _expected_request_metadata(expected_row).items():
            if key not in request:
                blockers.append(f"{label} query report result {query_id} request.{key} metadata missing")
            elif request[key] != expected:
                blockers.append(f"{label} query report result {query_id} request.{key} metadata mismatch")


def _native_retrieval_backend(row: dict[str, Any]) -> Any:
    response = row.get("response")
    trace = response.get("trace") if isinstance(response, dict) else None
    return trace.get("retrieval_backend") if isinstance(trace, dict) else None


def _add_native_backend_trace_blockers(*, report: dict[str, Any], blockers: list[str]) -> None:
    results = report.get("results", [])
    if not isinstance(results, list):
        return
    for row in results:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        query_id = str(row["id"])
        backend = _native_retrieval_backend(row)
        if backend is None:
            blockers.append(f"native query report result {query_id} response trace retrieval_backend missing")
        elif str(backend) != "zvec":
            blockers.append(f"native query report result {query_id} response trace retrieval_backend {backend} is not zvec")


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    return ordered[lower] * (upper - rank) + ordered[upper] * (rank - lower)


def _positive_finite_milliseconds(value: Any, *, label: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"query report {label} must be positive finite milliseconds")
    return numeric


def _query_elapsed_values(report: dict[str, Any], *, key: str) -> list[float]:
    values: list[float] = []
    for row in report.get(key, []):
        if not isinstance(row, dict):
            continue
        if row.get("elapsed_ms") is None:
            raise ValueError(f"query report {key}[].elapsed_ms is required")
        values.append(_positive_finite_milliseconds(row["elapsed_ms"], label=f"{key}[].elapsed_ms"))
    return values


def _observed_query_p95s(*, sample_values: list[float], result_values: list[float]) -> list[float]:
    return [_percentile(values, 95) for values in (sample_values, result_values) if values]


def _query_latency_evidence(report: dict[str, Any]) -> dict[str, Any]:
    result_values = _query_elapsed_values(report, key="results")
    sample_values = _query_elapsed_values(report, key="samples")
    result_observed_p95 = _percentile(result_values, 95) if result_values else None
    sample_observed_p95 = _percentile(sample_values, 95) if sample_values else None
    observed_p95s = [
        value for value in (sample_observed_p95, result_observed_p95) if value is not None
    ]
    summary = report.get("summary")
    summary_p95 = None
    if isinstance(summary, dict) and summary.get("p95_ms") is not None:
        summary_p95 = _positive_finite_milliseconds(summary["p95_ms"], label="summary.p95_ms")
        if observed_p95s:
            observed_p95 = max(observed_p95s)
            if summary_p95 + P95_CONSISTENCY_TOLERANCE_MS < observed_p95:
                raise ValueError(
                    "query report summary.p95_ms "
                    f"{summary_p95:.3f} is below observed p95 {observed_p95:.3f}"
                )
        effective_p95 = summary_p95
        source = "summary"
    else:
        if not result_values:
            raise ValueError("query report needs summary.p95_ms or result elapsed_ms values")
        effective_p95 = max(observed_p95s)
        source = "observed"
    return {
        "effective_p95_ms": effective_p95,
        "source": source,
        "summary_p95_ms": summary_p95,
        "result_observed_p95_ms": result_observed_p95,
        "sample_observed_p95_ms": sample_observed_p95,
        "observed_p95_ms": max(observed_p95s) if observed_p95s else None,
    }


def _query_p95_ms(report: dict[str, Any]) -> float:
    return float(_query_latency_evidence(report)["effective_p95_ms"])


def _positive_sample_rep(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("query report samples[].rep must be a positive integer")
    return value


def _sample_counts_by_id(report: dict[str, Any]) -> dict[str, int]:
    samples = report.get("samples")
    if isinstance(samples, list):
        reps_by_id: dict[str, set[int]] = {}
        for sample in samples:
            if isinstance(sample, dict) and sample.get("id"):
                query_id = str(sample["id"])
                if sample.get("rep") is None:
                    raise ValueError("query report samples[].rep is required")
                reps_by_id.setdefault(query_id, set()).add(_positive_sample_rep(sample["rep"]))
        return {query_id: len(reps) for query_id, reps in reps_by_id.items()}
    return {query_id: 1 for query_id in _results_by_id(report)}


def _add_query_suite_fingerprint_blockers(
    *,
    label: str,
    report: dict[str, Any],
    expected_sha256: str,
    blockers: list[str],
) -> None:
    actual = report.get("query_suite_sha256")
    if not actual:
        blockers.append(f"{label} query report missing query_suite_sha256")
    elif str(actual) != expected_sha256:
        blockers.append(f"{label} query report query_suite_sha256 mismatch")


def _add_timing_scope_blockers(*, label: str, report: dict[str, Any], blockers: list[str]) -> None:
    actual = report.get("timing_scope")
    if not actual:
        blockers.append(f"{label} query report missing timing_scope=data_only")
    elif str(actual) != "data_only":
        blockers.append(f"{label} query report timing_scope {actual} is not data_only")


def _query_role_policy(*, baseline_query: dict[str, Any], native_query: dict[str, Any]) -> dict[str, Any]:
    return {
        "required": {
            "baseline": "baseline_query",
            "native": "native_query",
        },
        "baseline": baseline_query.get("measurement_role"),
        "native": native_query.get("measurement_role"),
    }


def _add_query_role_blockers(*, policy: dict[str, Any], blockers: list[str]) -> None:
    required = policy["required"]
    for label in ("baseline", "native"):
        actual = policy.get(label)
        expected = required[label]
        if not actual:
            blockers.append(f"{label} query report missing measurement_role={expected}")
        elif str(actual) != expected:
            blockers.append(f"{label} query report measurement_role {actual} is not {expected}")


def _query_endpoint_policy(*, baseline_query: dict[str, Any], native_query: dict[str, Any]) -> dict[str, Any]:
    return {
        "required": "/query/data",
        "baseline": baseline_query.get("endpoint"),
        "native": native_query.get("endpoint"),
    }


def _add_query_endpoint_blockers(*, policy: dict[str, Any], blockers: list[str]) -> None:
    expected = policy["required"]
    for label in ("baseline", "native"):
        actual = policy.get(label)
        if not actual:
            blockers.append(f"{label} query report missing endpoint={expected}")
        elif str(actual) != expected:
            blockers.append(f"{label} query report endpoint {actual} is not {expected}")


def _baseline_vector_contract_policy(contract: Any) -> dict[str, Any]:
    required = {
        "endpoint": "/query/data",
        "explicit_vector_comparable": True,
        "query_vector_declared": True,
    }
    if not isinstance(contract, dict):
        return {
            "required": required,
            "endpoint": None,
            "explicit_vector_comparable": None,
            "query_vector_declared": None,
            "blockers": ["baseline vector contract missing explicit_vector_comparable=true"],
        }

    endpoint = contract.get("endpoint")
    explicit_vector_comparable = contract.get("explicit_vector_comparable")
    query_vector_declared = contract.get("query_vector_declared")
    blockers: list[str] = []
    if endpoint != required["endpoint"]:
        blockers.append(f"baseline vector contract endpoint {endpoint} is not /query/data")
    if explicit_vector_comparable is not True:
        blockers.append("baseline vector contract missing explicit_vector_comparable=true")
    if query_vector_declared is not True:
        blockers.append("baseline vector contract missing query_vector_declared=true")
    raw_blockers = contract.get("blockers")
    if isinstance(raw_blockers, list):
        blockers.extend(str(blocker) for blocker in raw_blockers if blocker)
    return {
        "required": required,
        "endpoint": endpoint,
        "explicit_vector_comparable": explicit_vector_comparable,
        "query_vector_declared": query_vector_declared,
        "blockers": blockers,
    }


def _add_baseline_vector_contract_blockers(*, policy: dict[str, Any], blockers: list[str]) -> None:
    blockers.extend(policy["blockers"])


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _baseline_timing_policy(
    policy: Any,
    *,
    query_suite_sha256: str,
    vector_contract_blockers: list[str],
) -> dict[str, Any]:
    required = {
        "policy_type": "baseline_endpoint_timing",
        "approved": True,
        "baseline_endpoint": "/query/data",
        "applies_to_query_suite_sha256": query_suite_sha256,
        "waives_baseline_vector_contract": True,
        "allows_endpoint_includes_embedding": True,
    }
    requires_policy = bool(vector_contract_blockers)
    if not isinstance(policy, dict):
        blockers = []
        if requires_policy:
            blockers.append(
                "baseline timing policy required when baseline vector contract is not explicit-vector comparable"
            )
        return {
            "required": required,
            "required_when": "baseline vector contract is not explicit-vector comparable",
            "present": False,
            "accepted": False,
            "policy_type": None,
            "baseline_endpoint": None,
            "applies_to_query_suite_sha256": None,
            "approved_by_present": False,
            "approved_at_present": False,
            "reason_present": False,
            "known_limitations_count": 0,
            "waived_vector_contract_blockers": [],
            "blockers": blockers,
        }

    blockers: list[str] = []
    for key, expected in required.items():
        actual = policy.get(key)
        if actual != expected:
            blockers.append(f"baseline timing policy {key} {actual} is not {expected}")
    if not _is_non_empty_string(policy.get("approved_by")):
        blockers.append("baseline timing policy approved_by must be a non-empty string")
    if not _is_non_empty_string(policy.get("approved_at")):
        blockers.append("baseline timing policy approved_at must be a non-empty string")
    if not _is_non_empty_string(policy.get("reason")):
        blockers.append("baseline timing policy reason must be a non-empty string")
    known_limitations = policy.get("known_limitations")
    if not isinstance(known_limitations, list) or not any(_is_non_empty_string(item) for item in known_limitations):
        blockers.append("baseline timing policy known_limitations must be a non-empty list")
        known_limitations_count = 0
    else:
        known_limitations_count = sum(1 for item in known_limitations if _is_non_empty_string(item))

    accepted = not blockers
    return {
        "required": required,
        "required_when": "baseline vector contract is not explicit-vector comparable",
        "present": True,
        "accepted": accepted,
        "policy_type": policy.get("policy_type"),
        "baseline_endpoint": policy.get("baseline_endpoint"),
        "applies_to_query_suite_sha256": policy.get("applies_to_query_suite_sha256"),
        "approved_by_present": _is_non_empty_string(policy.get("approved_by")),
        "approved_at_present": _is_non_empty_string(policy.get("approved_at")),
        "reason_present": _is_non_empty_string(policy.get("reason")),
        "known_limitations_count": known_limitations_count,
        "waived_vector_contract_blockers": list(vector_contract_blockers) if accepted and requires_policy else [],
        "blockers": blockers,
    }


def _has_explicit_query_vector(row: dict[str, Any]) -> bool:
    vector = row.get("query_vector")
    if not isinstance(vector, list) or not vector:
        return False
    return all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
        for value in vector
    )


def _query_vector_policy(query_suite: list[dict[str, Any]]) -> dict[str, Any]:
    missing_or_invalid_ids = [str(row["id"]) for row in query_suite if not _has_explicit_query_vector(row)]
    return {
        "required_for_timing_scope": "data_only",
        "total": len(query_suite),
        "valid_explicit_query_vector_count": len(query_suite) - len(missing_or_invalid_ids),
        "missing_or_invalid_ids": missing_or_invalid_ids,
    }


def _refresh_seconds(report: dict[str, Any]) -> float:
    for key in ("total_seconds", "elapsed_seconds", "duration_seconds"):
        if report.get(key) is not None:
            value = float(report[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"refresh report {key} must be positive finite seconds")
            return value
    raise ValueError("refresh report needs total_seconds, elapsed_seconds, or duration_seconds")


def _refresh_role_policy(*, baseline_refresh: dict[str, Any], native_refresh: dict[str, Any]) -> dict[str, Any]:
    return {
        "required": {
            "baseline": "baseline_refresh",
            "native": "native_refresh",
        },
        "baseline": baseline_refresh.get("measurement_role"),
        "native": native_refresh.get("measurement_role"),
    }


def _add_refresh_role_blockers(*, policy: dict[str, Any], blockers: list[str]) -> None:
    required = policy["required"]
    for label in ("baseline", "native"):
        actual = policy.get(label)
        expected = required[label]
        if not actual:
            blockers.append(f"{label} refresh report missing measurement_role={expected}")
        elif str(actual) != expected:
            blockers.append(f"{label} refresh report measurement_role {actual} is not {expected}")


def audit_comparison(
    *,
    query_suite_path: Path,
    baseline_query_report_path: Path,
    native_query_report_path: Path,
    baseline_refresh_report_path: Path,
    native_refresh_report_path: Path,
    baseline_vector_contract_report_path: Path | None = None,
    baseline_timing_policy_report_path: Path | None = None,
    min_samples_per_query: int = 1,
) -> dict[str, Any]:
    if min_samples_per_query <= 0:
        raise ValueError("min_samples_per_query must be positive")
    query_suite = _read_query_suite(query_suite_path)
    baseline_query = _read_json(baseline_query_report_path)
    native_query = _read_json(native_query_report_path)
    baseline_refresh = _read_json(baseline_refresh_report_path)
    native_refresh = _read_json(native_refresh_report_path)
    baseline_vector_contract = (
        _read_json(baseline_vector_contract_report_path)
        if baseline_vector_contract_report_path
        else baseline_query.get("baseline_vector_contract")
    )
    baseline_timing_policy = (
        _read_json(baseline_timing_policy_report_path)
        if baseline_timing_policy_report_path
        else baseline_query.get("baseline_timing_policy")
    )

    query_suite_by_id = {str(row["id"]): row for row in query_suite}
    query_ids = set(query_suite_by_id)
    baseline_results = _results_by_id(baseline_query)
    native_results = _results_by_id(native_query)
    blockers: list[str] = []
    query_suite_sha256 = _query_suite_sha256(query_suite_path)
    _add_query_suite_fingerprint_blockers(
        label="baseline",
        report=baseline_query,
        expected_sha256=query_suite_sha256,
        blockers=blockers,
    )
    _add_query_suite_fingerprint_blockers(
        label="native",
        report=native_query,
        expected_sha256=query_suite_sha256,
        blockers=blockers,
    )
    _add_timing_scope_blockers(label="baseline", report=baseline_query, blockers=blockers)
    _add_timing_scope_blockers(label="native", report=native_query, blockers=blockers)
    query_role_policy = _query_role_policy(baseline_query=baseline_query, native_query=native_query)
    _add_query_role_blockers(policy=query_role_policy, blockers=blockers)
    query_endpoint_policy = _query_endpoint_policy(baseline_query=baseline_query, native_query=native_query)
    _add_query_endpoint_blockers(policy=query_endpoint_policy, blockers=blockers)
    baseline_vector_contract_policy = _baseline_vector_contract_policy(baseline_vector_contract)
    baseline_timing_policy = _baseline_timing_policy(
        baseline_timing_policy,
        query_suite_sha256=query_suite_sha256,
        vector_contract_blockers=baseline_vector_contract_policy["blockers"],
    )
    if baseline_vector_contract_policy["blockers"] and not baseline_timing_policy["accepted"]:
        _add_baseline_vector_contract_blockers(policy=baseline_vector_contract_policy, blockers=blockers)
    blockers.extend(baseline_timing_policy["blockers"])
    query_vector_policy = _query_vector_policy(query_suite)
    if query_vector_policy["missing_or_invalid_ids"]:
        blockers.append(
            "query suite explicit query_vector coverage "
            f"{query_vector_policy['valid_explicit_query_vector_count']}/{query_vector_policy['total']} "
            "is required for data_only timing"
        )
    report_id_policy = {
        "baseline": _report_id_policy(report=baseline_query, query_ids=query_ids),
        "native": _report_id_policy(report=native_query, query_ids=query_ids),
    }
    _add_report_id_blockers(label="baseline", policy=report_id_policy["baseline"], blockers=blockers)
    _add_report_id_blockers(label="native", policy=report_id_policy["native"], blockers=blockers)
    _add_result_metadata_blockers(
        label="baseline",
        report=baseline_query,
        query_suite_by_id=query_suite_by_id,
        blockers=blockers,
    )
    _add_result_metadata_blockers(
        label="native",
        report=native_query,
        query_suite_by_id=query_suite_by_id,
        blockers=blockers,
    )
    _add_result_request_metadata_blockers(
        label="baseline",
        report=baseline_query,
        query_suite_by_id=query_suite_by_id,
        blockers=blockers,
    )
    _add_result_request_metadata_blockers(
        label="native",
        report=native_query,
        query_suite_by_id=query_suite_by_id,
        blockers=blockers,
    )
    _add_native_backend_trace_blockers(report=native_query, blockers=blockers)
    refresh_role_policy = _refresh_role_policy(baseline_refresh=baseline_refresh, native_refresh=native_refresh)
    _add_refresh_role_blockers(policy=refresh_role_policy, blockers=blockers)
    recall_hits = 0
    recall_total = 0
    per_query: list[dict[str, Any]] = []
    baseline_sample_counts = _sample_counts_by_id(baseline_query)
    native_sample_counts = _sample_counts_by_id(native_query)

    for row in query_suite:
        query_id = str(row["id"])
        baseline_row = baseline_results.get(query_id)
        native_row = native_results.get(query_id)
        baseline_sample_count = baseline_sample_counts.get(query_id, 0)
        native_sample_count = native_sample_counts.get(query_id, 0)
        if baseline_sample_count < min_samples_per_query:
            blockers.append(f"baseline samples for {query_id} {baseline_sample_count} below required {min_samples_per_query}")
        if native_sample_count < min_samples_per_query:
            blockers.append(f"native samples for {query_id} {native_sample_count} below required {min_samples_per_query}")
        if baseline_row is None:
            blockers.append(f"missing baseline result for {query_id}")
            continue
        if native_row is None:
            blockers.append(f"missing native result for {query_id}")
            continue
        baseline_paths = _ordered_paths(dict(baseline_row.get("response") or {}))
        native_response = dict(native_row.get("response") or {})
        native_paths = _ordered_paths(native_response)
        native_entities = _response_entities(native_response)
        native_backend = _native_retrieval_backend(native_row)
        if not baseline_paths:
            blockers.append(f"baseline result for {query_id} has no source paths")
        for required_path in row.get("must_include_paths", []) or []:
            if str(required_path) not in native_paths:
                blockers.append(f"missing required path for {query_id}: {required_path}")
        for required_entity in row.get("must_include_entities", []) or []:
            if str(required_entity) not in native_entities:
                blockers.append(f"missing required entity for {query_id}: {required_entity}")
        baseline_set = set(baseline_paths)
        native_set = set(native_paths)
        recall_hits += len(baseline_set & native_set)
        recall_total += len(baseline_set)
        per_query.append(
            {
                "id": query_id,
                "baseline_distinct_sample_count": baseline_sample_count,
                "native_distinct_sample_count": native_sample_count,
                "baseline_path_count_at_20": len(baseline_set),
                "native_path_count_at_20": len(native_set),
                "path_overlap_at_20": len(baseline_set & native_set),
                "native_retrieval_backend": native_backend,
            }
        )

    recall = (recall_hits / recall_total) if recall_total else 0.0
    if recall < REQUIRED_RECALL_AT_20:
        blockers.append(f"source_path Recall@20 {recall:.3f} is below {REQUIRED_RECALL_AT_20:.2f}")

    baseline_latency_evidence = _query_latency_evidence(baseline_query)
    native_latency_evidence = _query_latency_evidence(native_query)
    baseline_p95 = float(baseline_latency_evidence["effective_p95_ms"])
    native_p95 = float(native_latency_evidence["effective_p95_ms"])
    retrieval_threshold = baseline_p95 / SPEEDUP_FACTOR
    if native_p95 > retrieval_threshold:
        blockers.append(f"retrieval p95 {native_p95:.3f}ms exceeds threshold {retrieval_threshold:.3f}ms")

    baseline_seconds = _refresh_seconds(baseline_refresh)
    native_seconds = _refresh_seconds(native_refresh)
    refresh_threshold = baseline_seconds / SPEEDUP_FACTOR
    if native_seconds > refresh_threshold:
        blockers.append(f"refresh seconds {native_seconds:.3f}s exceeds threshold {refresh_threshold:.3f}s")

    return {
        "ok": not blockers,
        "blockers": blockers,
        "query_count": len(query_suite),
        "query_suite_sha256": query_suite_sha256,
        "source_path_recall_at_20": round(recall, 6),
        "required_recall_at_20": REQUIRED_RECALL_AT_20,
        "sample_policy": {
            "min_samples_per_query": min_samples_per_query,
        },
        "timing_scope": {
            "required": "data_only",
            "baseline": baseline_query.get("timing_scope"),
            "native": native_query.get("timing_scope"),
        },
        "query_role_policy": query_role_policy,
        "query_endpoint_policy": query_endpoint_policy,
        "baseline_vector_contract_policy": baseline_vector_contract_policy,
        "baseline_timing_policy": baseline_timing_policy,
        "query_vector_policy": query_vector_policy,
        "report_id_policy": report_id_policy,
        "refresh_role_policy": refresh_role_policy,
        "retrieval": {
            "ok": native_p95 <= retrieval_threshold,
            "baseline_p95_ms": baseline_p95,
            "native_p95_ms": native_p95,
            "threshold_ms": retrieval_threshold,
            "speedup_factor": SPEEDUP_FACTOR,
            "latency_evidence": {
                "baseline": baseline_latency_evidence,
                "native": native_latency_evidence,
            },
        },
        "refresh": {
            "ok": native_seconds <= refresh_threshold,
            "baseline_seconds": baseline_seconds,
            "native_seconds": native_seconds,
            "threshold_seconds": refresh_threshold,
            "speedup_factor": SPEEDUP_FACTOR,
        },
        "per_query": per_query,
    }


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit saved native-vs-baseline performance comparison artifacts")
    parser.add_argument("--query-suite", type=Path, required=True)
    parser.add_argument("--baseline-query-report", type=Path, required=True)
    parser.add_argument("--native-query-report", type=Path, required=True)
    parser.add_argument("--baseline-refresh-report", type=Path, required=True)
    parser.add_argument("--native-refresh-report", type=Path, required=True)
    parser.add_argument("--baseline-vector-contract-report", type=Path)
    parser.add_argument("--baseline-timing-policy-report", type=Path)
    parser.add_argument("--min-samples-per-query", type=int, default=1)
    args = parser.parse_args(argv)

    input_paths = [
        args.query_suite,
        args.baseline_query_report,
        args.native_query_report,
        args.baseline_refresh_report,
        args.native_refresh_report,
    ]
    if args.baseline_vector_contract_report:
        input_paths.append(args.baseline_vector_contract_report)
    if args.baseline_timing_policy_report:
        input_paths.append(args.baseline_timing_policy_report)
    missing = [str(path) for path in input_paths if not path.exists()]
    if missing:
        print_json({"ok": False, "error": "missing_inputs", "missing_inputs": missing})
        return 1
    try:
        report = audit_comparison(
            query_suite_path=args.query_suite,
            baseline_query_report_path=args.baseline_query_report,
            native_query_report_path=args.native_query_report,
            baseline_refresh_report_path=args.baseline_refresh_report,
            native_refresh_report_path=args.native_refresh_report,
            baseline_vector_contract_report_path=args.baseline_vector_contract_report,
            baseline_timing_policy_report_path=args.baseline_timing_policy_report,
            min_samples_per_query=args.min_samples_per_query,
        )
    except Exception as exc:
        print_json({"ok": False, "error": "invalid_input", "message": str(exc), "exception_type": type(exc).__name__})
        return 1
    print_json(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
