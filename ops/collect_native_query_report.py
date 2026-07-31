#!/usr/bin/env python3
"""Collect native query latency reports from a saved query suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import statistics
import subprocess
import tempfile
import time
from typing import Any, Callable
import urllib.error
import urllib.request

from llm_wiki_native.contracts import (
    MAX_CHARS_PER_BLOCK,
    MAX_QUERY_VECTOR_DIM,
    MAX_TOP_K,
    RESPONSE_PROFILES,
    SUPPORTED_RETRIEVAL_GOALS,
    WORKSPACE_SCHEMA_VERSION,
)
from llm_wiki_native.query_contract import query_request_metadata, query_suite_payload, response_max_chars
from llm_wiki_native.reports import validate_query_suite_row, validate_relevance_quality_row  # noqa: E402

Timer = Callable[[], float]

QUALITY_CONTRACT = "relevance-v1"
QUALITY_REPORT_CONTRACT_VERSION = "relevance-v1-report-v1"
RESPONSE_SIZE_BOUND_BYTES = 1_048_576


def read_query_suite(path: Path, *, quality_contract: str | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if quality_contract == QUALITY_CONTRACT:
            validate_relevance_quality_row(row)
        elif quality_contract is None:
            validate_query_suite_row(row)
        else:
            raise ValueError(f"unsupported quality contract: {quality_contract}")
    if quality_contract == QUALITY_CONTRACT:
        if not rows:
            raise ValueError("relevance-v1 query suite must not be empty")
        row_ids = [row["id"] for row in rows]
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("relevance-v1 query suite contains duplicate row IDs")
        if len({len(row["query_vector"]) for row in rows}) != 1:
            raise ValueError("relevance-v1 query vectors must have one consistent dimension")
        counts = {
            partition: sum(row["partition"] == partition for row in rows)
            for partition in ("calibration", "holdout")
        }
        if counts != {"calibration": 24, "holdout": 12}:
            raise ValueError("relevance-v1 query suite must contain 24 calibration and 12 holdout rows")
        goal_counts = {
            (partition, goal): sum(
                row["partition"] == partition and row["retrieval_goal"] == goal for row in rows
            )
            for partition in ("calibration", "holdout")
            for goal in ("focused", "coverage")
        }
        if goal_counts != {
            ("calibration", "focused"): 16,
            ("calibration", "coverage"): 8,
            ("holdout", "focused"): 8,
            ("holdout", "coverage"): 4,
        }:
            raise ValueError("relevance-v1 query suite must preserve the frozen focused/coverage distribution")
    return rows


def select_quality_partition(rows: list[dict[str, Any]], partition: str) -> list[dict[str, Any]]:
    if partition == "all":
        return list(rows)
    if partition not in {"calibration", "holdout"}:
        raise ValueError(f"unsupported quality partition: {partition}")
    return [row for row in rows if row["partition"] == partition]


def _query_suite_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved_target(pointer_path: Path, value: object, *, kind: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"workspace pointer {kind} must be a non-empty path")
    target = Path(value).expanduser()
    if not target.is_absolute():
        target = pointer_path.parent / target
    target = target.resolve()
    target_checks = {
        "source_root": target.is_dir,
        "sqlite_path": target.is_file,
        "zvec_path": target.is_dir,
    }
    if not target_checks[kind]():
        raise ValueError(f"workspace pointer {kind} target is missing: {target}")
    return target


def build_corpus_identity(workspace_file: Path) -> dict[str, Any]:
    pointer_path = Path(workspace_file).expanduser().resolve()
    if not pointer_path.is_file():
        raise FileNotFoundError(pointer_path)
    pointer_bytes = pointer_path.read_bytes()
    pointer = json.loads(pointer_bytes)
    if not isinstance(pointer, dict):
        raise ValueError("workspace pointer must contain a JSON object")

    workspace_id = pointer.get("workspace_id")
    source_manifest_hash = pointer.get("source_manifest_hash")
    schema_version = pointer.get("schema_version")
    status = pointer.get("status")
    embedding_dim = pointer.get("embedding_dim")
    zvec_metadata = pointer.get("zvec")
    if embedding_dim is None and isinstance(zvec_metadata, dict):
        embedding_dim = zvec_metadata.get("embedding_dim")
    if not isinstance(workspace_id, str) or not workspace_id:
        raise ValueError("workspace pointer workspace_id must be non-empty")
    if not isinstance(source_manifest_hash, str) or not source_manifest_hash:
        raise ValueError("workspace pointer source_manifest_hash must be non-empty")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("workspace pointer schema_version must be an integer")
    if schema_version != WORKSPACE_SCHEMA_VERSION:
        raise ValueError(f"workspace pointer schema_version must equal {WORKSPACE_SCHEMA_VERSION}")
    if status != "active":
        raise ValueError("workspace pointer status must be active")
    if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int) or embedding_dim < 1:
        raise ValueError("workspace pointer embedding_dim must be a positive integer")

    targets = {
        key: _resolved_target(pointer_path, pointer.get(key), kind=key)
        for key in ("source_root", "sqlite_path", "zvec_path")
    }
    sqlite_path = targets["sqlite_path"]
    with sqlite3.connect(f"{sqlite_path.as_uri()}?mode=ro", uri=True) as conn:
        row = conn.execute(
            "SELECT workspace_id, source_manifest_hash, schema_version, status "
            "FROM workspace WHERE workspace_id = ?",
            (workspace_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"workspace_id not found in SQLite metadata: {workspace_id}")
    metadata: dict[str, object] = dict(
        zip(("workspace_id", "source_manifest_hash", "schema_version", "status"), row, strict=True)
    )
    expected_metadata: dict[str, object] = {
        "workspace_id": workspace_id,
        "source_manifest_hash": source_manifest_hash,
        "schema_version": schema_version,
    }
    for key, expected in expected_metadata.items():
        if metadata[key] != expected:
            raise ValueError(f"workspace pointer {key} does not match SQLite metadata")
    if metadata["status"] != "audited":
        raise ValueError("SQLite workspace status must be audited")

    return {
        "frozen_pointer_path": str(pointer_path),
        "frozen_pointer_sha256": hashlib.sha256(pointer_bytes).hexdigest(),
        "workspace_id": workspace_id,
        "source_manifest_hash": source_manifest_hash,
        "workspace_schema_version": schema_version,
        "status": status,
        "sqlite_status": metadata["status"],
        "embedding_dim": embedding_dim,
        "resolved_targets": {key: str(path) for key, path in targets.items()},
    }


def _fingerprint_files(root: Path, relative_paths: list[Path], *, suffix: bytes = b"") -> str:
    digest = hashlib.sha256()
    for relative_path in sorted(relative_paths, key=lambda path: path.as_posix()):
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(relative_path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    digest.update(suffix)
    return digest.hexdigest()


def runtime_package_fingerprint(runtime_code_root: Path) -> str:
    root = Path(runtime_code_root).expanduser().resolve()
    package_root = root / "llm_wiki_native"
    relative_paths = [path.relative_to(root) for path in package_root.rglob("*.py")]
    if not relative_paths:
        raise ValueError(f"no llm_wiki_native Python files found under {root}")
    return _fingerprint_files(root, relative_paths)


def client_fingerprint(code_root: Path) -> str:
    root = Path(code_root).expanduser().resolve()
    return _fingerprint_files(
        root,
        [Path("ops/wiki_search.py"), Path("ops/wiki_native_query_events.py")],
    )


def runtime_git_revision(runtime_code_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(Path(runtime_code_root).resolve()), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision or None


def evaluator_fingerprint(code_root: Path) -> str:
    root = Path(code_root).expanduser().resolve()
    contract = {
        "quality_contract": QUALITY_CONTRACT,
        "report_contract_version": QUALITY_REPORT_CONTRACT_VERSION,
        "retrieval_goals": sorted(SUPPORTED_RETRIEVAL_GOALS),
        "response_profiles": sorted(RESPONSE_PROFILES),
        "max_top_k": MAX_TOP_K,
        "max_query_vector_dim": MAX_QUERY_VECTOR_DIM,
        "max_chars_per_block": MAX_CHARS_PER_BLOCK,
        "response_size_bound_bytes": RESPONSE_SIZE_BOUND_BYTES,
    }
    return _fingerprint_files(
        root,
        [Path("llm_wiki_native/reports.py"), Path("ops/collect_native_query_report.py")],
        suffix=b"quality-contract\0" + _canonical_json_bytes(contract),
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


_PROCESS_DIAGNOSTIC_KEYS = {
    "created_at",
    "ended_at",
    "elapsed",
    "latency",
    "pid",
    "process_diagnostics",
    "process_id",
    "started_at",
    "timestamp",
    "timings",
    "updated_at",
}


def _without_process_diagnostics(value: object) -> object:
    if isinstance(value, list):
        return [_without_process_diagnostics(item) for item in value]
    if not isinstance(value, dict):
        return value
    result: dict[str, object] = {}
    for key, item in value.items():
        normalized = str(key).lower()
        if (
            normalized in _PROCESS_DIAGNOSTIC_KEYS
            or normalized.endswith("_ms")
            or "latency" in normalized
            or "timestamp" in normalized
            or "duration" in normalized
        ):
            continue
        result[str(key)] = _without_process_diagnostics(item)
    return result


def quality_response_fingerprint(response: dict[str, Any]) -> str:
    raw_blocks = response.get("context_blocks")
    blocks = raw_blocks if isinstance(raw_blocks, list) else []
    debug = _as_dict(response.get("retrieval_debug"))
    selection_keys = (
        "candidate_cards",
        "decision_cards",
        "decisions",
        "hits",
        "identity_cards",
        "selection",
        "selection_summary",
        "source_scope",
    )
    selection = {key: debug[key] for key in selection_keys if key in debug}
    fingerprint_payload = {
        "source_paths": response.get("source_paths", []),
        "context_blocks": [
            {
                "record_id": block.get("record_id"),
                "source_path": block.get("source_path"),
                "text": block.get("text"),
            }
            for block in blocks
            if isinstance(block, dict)
        ],
        "selection_summary": _without_process_diagnostics(selection),
        "trace_summary": _without_process_diagnostics(response.get("trace", {})),
    }
    return hashlib.sha256(_canonical_json_bytes(fingerprint_payload)).hexdigest()


def _card_source_path(card: object) -> str | None:
    if isinstance(card, str) and card:
        return card
    if not isinstance(card, dict):
        return None
    value = card.get("source_path")
    if isinstance(value, str) and value:
        return value
    record = card.get("record")
    if isinstance(record, dict) and isinstance(record.get("source_path"), str):
        return record["source_path"] or None
    return None


def _recall(expected: list[str], observed: set[str]) -> float:
    if not expected:
        return 1.0
    return sum(item in observed for item in expected) / len(expected)


def score_quality_response(row: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    raw_blocks = response.get("context_blocks")
    blocks = [block for block in raw_blocks if isinstance(block, dict)] if isinstance(raw_blocks, list) else []
    raw_source_paths = response.get("source_paths")
    source_paths = [path for path in raw_source_paths if isinstance(path, str)] if isinstance(raw_source_paths, list) else []
    block_source_paths = [
        path for block in blocks if (path := _card_source_path(block)) is not None
    ]
    distinct_block_paths = _ordered_unique(block_source_paths)

    debug = _as_dict(response.get("retrieval_debug"))
    scope_cards = debug.get("source_scope")
    if not isinstance(scope_cards, list):
        scope_cards = debug.get("hits") if isinstance(debug.get("hits"), list) else []
    scope_paths = {
        path for card in scope_cards if (path := _card_source_path(card)) is not None
    }

    candidate_cards: object = debug.get("candidate_cards")
    if not isinstance(candidate_cards, list):
        candidate_cards = debug.get("identity_cards")
    if not isinstance(candidate_cards, list):
        candidate_cards = debug.get("hits") if isinstance(debug.get("hits"), list) else []
    candidate_ids: set[str] = set()
    for card in candidate_cards:
        if not isinstance(card, dict):
            continue
        for key in ("record_id", "source_id"):
            value = card.get(key)
            if isinstance(value, str) and value:
                candidate_ids.add(value)

    required_paths = row["must_include_paths"]
    expected_entities = row["must_include_entities"]
    evidence_matches: list[bool] = []
    matched_evidence_paths: set[str] = set()
    for item in row["must_include_evidence"]:
        matched = any(
            _card_source_path(block) == item["source_path"]
            and isinstance(block.get("text"), str)
            and all(anchor in block["text"] for anchor in item["text_contains"])
            for block in blocks
        )
        evidence_matches.append(matched)
        if matched:
            matched_evidence_paths.add(item["source_path"])

    ranks = [source_paths.index(path) + 1 for path in required_paths if path in source_paths]
    block_count = len(blocks)
    distinct_source_count = len(distinct_block_paths)
    duplicate_rate = (block_count - distinct_source_count) / block_count if block_count else 0.0
    max_chars = response_max_chars(row)
    text_shape_ok = all(isinstance(block.get("text"), str) for block in blocks)
    source_shape_ok = isinstance(raw_source_paths, list) and all(
        isinstance(path, str) and path for path in raw_source_paths
    )
    source_order_ok = source_shape_ok and source_paths == distinct_block_paths
    response_bound_ok = (
        block_count <= row["top_k"]
        and text_shape_ok
        and all(len(block["text"]) <= max_chars for block in blocks)
        and source_order_ok
    )
    response_bytes = len(_canonical_json_bytes(response))

    hashes_by_source: dict[str, list[str]] = {}
    for block in blocks:
        source_path = _card_source_path(block)
        text = block.get("text")
        if source_path is None or not isinstance(text, str):
            continue
        evidence_hash = block.get("evidence_hash") or block.get("content_hash")
        if not isinstance(evidence_hash, str) or not evidence_hash:
            evidence_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        hashes_by_source.setdefault(source_path, []).append(evidence_hash)
    focused_hashes_distinct = all(len(values) == len(set(values)) for values in hashes_by_source.values())

    seen_sources: set[str] = set()
    fill_pass_started = False
    coverage_first_pass_ok = True
    for source_path in block_source_paths:
        if source_path in seen_sources:
            fill_pass_started = True
        elif fill_pass_started:
            coverage_first_pass_ok = False
        seen_sources.add(source_path)

    metrics = {
        "scope_path_recall": _recall(required_paths, scope_paths),
        "visible_path_recall": _recall(required_paths, set(source_paths)),
        "candidate_entity_recall": _recall(expected_entities, candidate_ids),
        "visible_evidence_recall": sum(evidence_matches) / len(evidence_matches),
        "first_required_path_rank": min(ranks) if ranks else None,
        "distinct_source_count": distinct_source_count,
        "relevant_distinct_source_count": len(matched_evidence_paths),
        "minimum_relevant_sources_met": len(matched_evidence_paths) >= row["minimum_distinct_sources"],
        "duplicate_block_rate": duplicate_rate,
        "response_chars": sum(len(block["text"]) for block in blocks if isinstance(block.get("text"), str)),
        "response_bytes": response_bytes,
        "response_bound_ok": response_bound_ok,
        "response_size_bound_ok": response_bytes <= RESPONSE_SIZE_BOUND_BYTES,
        "focused_evidence_hashes_distinct": focused_hashes_distinct,
        "coverage_distinct_first_pass_ok": coverage_first_pass_ok,
        "response_fingerprint": quality_response_fingerprint(response),
    }
    metrics["quality_pass"] = all(
        (
            metrics["scope_path_recall"] == 1.0,
            metrics["candidate_entity_recall"] == 1.0,
            metrics["visible_evidence_recall"] == 1.0,
            metrics["minimum_relevant_sources_met"],
            metrics["response_bound_ok"],
            metrics["response_size_bound_ok"],
        )
    )
    return metrics


def _quality_failure_class(
    metrics: dict[str, Any], *, response_deterministic: bool, retrieval_goal: str
) -> str | None:
    if metrics["scope_path_recall"] < 1.0:
        return "scope_miss"
    if metrics["candidate_entity_recall"] < 1.0:
        return "primary_candidate_miss"
    if metrics["visible_evidence_recall"] < 1.0:
        return "evidence_ranking"
    if not metrics["minimum_relevant_sources_met"]:
        return "quota"
    if not metrics["response_bound_ok"] or not metrics["response_size_bound_ok"]:
        return "truncation"
    if retrieval_goal == "focused" and not metrics["focused_evidence_hashes_distinct"]:
        return "evidence_ranking"
    if retrieval_goal == "coverage" and not metrics["coverage_distinct_first_pass_ok"]:
        return "quota"
    if not response_deterministic:
        return "identity_staleness"
    return None


def evaluate_quality_repetitions(
    row: dict[str, Any],
    *,
    responses: list[dict[str, Any]],
    latencies_ms: list[float],
) -> dict[str, Any]:
    if not responses or len(responses) != len(latencies_ms):
        raise ValueError("quality repetitions require equal non-empty response and latency lists")
    repetition_results: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    for response, latency_ms in zip(responses, latencies_ms, strict=True):
        score = score_quality_response(row, response)
        scores.append(score)
        repetition_results.append({"latency_ms": latency_ms, **score})

    fingerprints = [score["response_fingerprint"] for score in scores]
    response_deterministic = len(set(fingerprints)) == 1
    ranks = [score["first_required_path_rank"] for score in scores if score["first_required_path_rank"]]
    metrics = {
        "scope_path_recall": max(score["scope_path_recall"] for score in scores),
        "visible_path_recall": max(score["visible_path_recall"] for score in scores),
        "candidate_entity_recall": max(score["candidate_entity_recall"] for score in scores),
        "visible_evidence_recall": max(score["visible_evidence_recall"] for score in scores),
        "first_required_path_rank": min(ranks) if ranks else None,
        "distinct_source_count": max(score["distinct_source_count"] for score in scores),
        "relevant_distinct_source_count": max(score["relevant_distinct_source_count"] for score in scores),
        "minimum_relevant_sources_met": any(score["minimum_relevant_sources_met"] for score in scores),
        "duplicate_block_rate": min(score["duplicate_block_rate"] for score in scores),
        "response_chars": max(score["response_chars"] for score in scores),
        "response_bytes": max(score["response_bytes"] for score in scores),
        "response_bound_ok": all(score["response_bound_ok"] for score in scores),
        "response_size_bound_ok": all(score["response_size_bound_ok"] for score in scores),
        "focused_evidence_hashes_distinct": all(
            score["focused_evidence_hashes_distinct"] for score in scores
        ),
        "coverage_distinct_first_pass_ok": all(
            score["coverage_distinct_first_pass_ok"] for score in scores
        ),
    }
    quality_pass = (
        response_deterministic
        and all(score["quality_pass"] for score in scores)
        and (row["retrieval_goal"] != "focused" or metrics["focused_evidence_hashes_distinct"])
        and (row["retrieval_goal"] != "coverage" or metrics["coverage_distinct_first_pass_ok"])
    )
    return {
        "id": row["id"],
        "retrieval_goal": row["retrieval_goal"],
        "partition": row["partition"],
        "critical": row["critical"],
        "response": responses[0],
        "repetition_results": repetition_results,
        "response_deterministic": response_deterministic,
        "latencies_ms": list(latencies_ms),
        "latency_median_ms": statistics.median(latencies_ms),
        "metrics": metrics,
        "quality_pass": quality_pass,
        "failure_class": _quality_failure_class(
            metrics,
            response_deterministic=response_deterministic,
            retrieval_goal=row["retrieval_goal"],
        ),
    }


def _nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _quality_value_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "p50": statistics.median(values),
        "p95": _nearest_rank(values, 0.95),
        "max": max(values),
    }


def _quality_latency_summary(values: list[float]) -> dict[str, float | int]:
    summary = _quality_value_summary(values)
    return {
        "count": summary["count"],
        "min_ms": summary["min"],
        "p50_ms": summary["p50"],
        "p95_ms": summary["p95"],
        "max_ms": summary["max"],
    }


def _aggregate_quality_group(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "row_count": 0,
            "scope_path_recall": 0.0,
            "visible_path_recall": 0.0,
            "candidate_entity_recall": 0.0,
            "visible_evidence_recall": 0.0,
            "first_path_mrr": 0.0,
            "relevant_source_coverage_success": 0.0,
            "duplicate_block_rate": 0.0,
            "critical_failure_ids": [],
            "failure_ids": [],
            "failure_classes": {},
            "latency": _quality_latency_summary([]),
            "response_chars": _quality_value_summary([]),
            "response_bytes": _quality_value_summary([]),
            "all_response_deterministic": False,
            "all_quality_pass": False,
        }
    metric_rows = [result["metrics"] for result in results]

    def mean(key: str) -> float:
        return sum(float(metrics[key]) for metrics in metric_rows) / len(metric_rows)

    first_ranks = [
        metrics["first_required_path_rank"]
        for metrics in metric_rows
        if metrics["first_required_path_rank"] is not None
    ]
    latencies = [latency for result in results for latency in result["latencies_ms"]]
    response_chars = [
        float(repetition["response_chars"])
        for result in results
        for repetition in result["repetition_results"]
    ]
    response_bytes = [
        float(repetition["response_bytes"])
        for result in results
        for repetition in result["repetition_results"]
    ]
    failures = [result for result in results if not result["quality_pass"]]
    return {
        "row_count": len(results),
        "scope_path_recall": mean("scope_path_recall"),
        "visible_path_recall": mean("visible_path_recall"),
        "candidate_entity_recall": mean("candidate_entity_recall"),
        "visible_evidence_recall": mean("visible_evidence_recall"),
        "first_path_mrr": (
            sum(1.0 / rank for rank in first_ranks) / len(first_ranks) if first_ranks else 0.0
        ),
        "relevant_source_coverage_success": sum(
            bool(metrics["minimum_relevant_sources_met"]) for metrics in metric_rows
        )
        / len(metric_rows),
        "duplicate_block_rate": mean("duplicate_block_rate"),
        "latency": _quality_latency_summary(latencies),
        "response_chars": _quality_value_summary(response_chars),
        "response_bytes": _quality_value_summary(response_bytes),
        "all_response_deterministic": all(result["response_deterministic"] for result in results),
        "all_quality_pass": not failures,
        "critical_failure_ids": [result["id"] for result in failures if result["critical"]],
        "failure_ids": [result["id"] for result in failures],
        "failure_classes": {
            result["id"]: result["failure_class"] for result in failures
        },
    }


def aggregate_quality_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": _aggregate_quality_group(results),
        "focused": _aggregate_quality_group(
            [result for result in results if result["retrieval_goal"] == "focused"]
        ),
        "coverage": _aggregate_quality_group(
            [result for result in results if result["retrieval_goal"] == "coverage"]
        ),
        "partition": {
            partition: _aggregate_quality_group(
                [result for result in results if result["partition"] == partition]
            )
            for partition in ("calibration", "holdout")
        },
    }


def _semantic_corpus_identity(identity: object) -> object:
    if not isinstance(identity, dict):
        return identity
    return {key: value for key, value in identity.items() if key != "frozen_pointer_path"}


def _report_results_by_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results = report.get("results")
    if not isinstance(results, list):
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("id"), str):
            continue
        indexed[result["id"]] = result
    return indexed


def _report_partition_shape_valid(report: dict[str, Any]) -> bool:
    partition = report.get("partition")
    results = list(_report_results_by_id(report).values())
    expected_counts = {"calibration": 24, "holdout": 12, "all": 36}
    if partition not in expected_counts or len(results) != expected_counts[partition]:
        return False
    expected_distribution = {
        "calibration": {("calibration", "focused"): 16, ("calibration", "coverage"): 8},
        "holdout": {("holdout", "focused"): 8, ("holdout", "coverage"): 4},
        "all": {
            ("calibration", "focused"): 16,
            ("calibration", "coverage"): 8,
            ("holdout", "focused"): 8,
            ("holdout", "coverage"): 4,
        },
    }
    actual_distribution: dict[tuple[str, str], int] = {}
    for result in results:
        row_partition = result.get("partition")
        retrieval_goal = result.get("retrieval_goal")
        if not isinstance(row_partition, str) or not isinstance(retrieval_goal, str):
            return False
        key = (row_partition, retrieval_goal)
        actual_distribution[key] = actual_distribution.get(key, 0) + 1
        if bool(result.get("critical")) != (row_partition == "holdout"):
            return False
    return actual_distribution == expected_distribution[partition]


def _comparison_identity_failures(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> tuple[list[str], list[dict[str, Any]]]:
    failures: list[str] = []
    for field in (
        "quality_contract",
        "report_contract_version",
        "query_suite_sha256",
        "evaluator_fingerprint",
        "endpoint",
        "timing_scope",
        "warmup_runs",
        "repetitions",
    ):
        if candidate.get(field) != baseline.get(field):
            failures.append(field)
    if not _report_partition_shape_valid(baseline) or baseline.get("partition") != "all":
        failures.append("baseline_partition")
    if not _report_partition_shape_valid(candidate):
        failures.append("candidate_partition")
    if _semantic_corpus_identity(candidate.get("corpus_identity")) != _semantic_corpus_identity(
        baseline.get("corpus_identity")
    ):
        failures.append("corpus_identity")

    candidate_results = _report_results_by_id(candidate)
    baseline_results = _report_results_by_id(baseline)
    selected_ids = candidate.get("selected_row_ids")
    if not isinstance(selected_ids, list) or selected_ids != list(candidate_results):
        failures.append("selected_row_ids")
        selected_ids = list(candidate_results)
    selected_id_set = set(selected_ids)
    baseline_projection = [row_id for row_id in baseline_results if row_id in selected_id_set]
    if baseline_projection != selected_ids:
        failures.append("baseline_projection")

    request_differences: list[dict[str, Any]] = []
    for row_id in selected_ids:
        candidate_result = candidate_results.get(row_id)
        baseline_result = baseline_results.get(row_id)
        if candidate_result is None or baseline_result is None:
            continue
        if candidate_result.get("effective_request_sha256") != baseline_result.get(
            "effective_request_sha256"
        ):
            request_differences.append({"id": row_id})
    if request_differences:
        failures.append("effective_request")
    return list(dict.fromkeys(failures)), request_differences


def _active_identity_failures(active: dict[str, Any], accepted: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if _semantic_corpus_identity(active.get("corpus_identity")) != _semantic_corpus_identity(
        accepted.get("corpus_identity")
    ):
        failures.append("corpus_identity")
    for field in (
        "evaluator_fingerprint",
        "runtime_package_fingerprint",
        "client_fingerprint",
    ):
        if active.get(field) != accepted.get(field):
            failures.append(field)
    return failures


def compare_quality_reports(
    candidate_report: dict[str, Any],
    baseline_report: dict[str, Any],
    *,
    accepted_candidate_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    identity_failures, request_differences = _comparison_identity_failures(
        candidate_report, baseline_report
    )
    candidate_results = list(_report_results_by_id(candidate_report).values())
    baseline_by_id = _report_results_by_id(baseline_report)
    baseline_projection = [
        baseline_by_id[result["id"]]
        for result in candidate_results
        if result["id"] in baseline_by_id
    ]
    candidate_aggregates = aggregate_quality_results(candidate_results)
    baseline_aggregates = aggregate_quality_results(baseline_projection)

    candidate_p95 = float(candidate_aggregates["overall"]["latency"]["p95_ms"])
    baseline_p95 = float(baseline_aggregates["overall"]["latency"]["p95_ms"])
    endpoint_threshold = max(baseline_p95 * 1.20, baseline_p95 + 25.0)
    endpoint_latency = {
        "baseline_p95_ms": baseline_p95,
        "candidate_p95_ms": candidate_p95,
        "threshold_ms": endpoint_threshold,
        "passed": candidate_p95 <= endpoint_threshold,
    }

    paired_rows: list[dict[str, Any]] = []
    for candidate_result in candidate_results:
        baseline_result = baseline_by_id.get(candidate_result["id"])
        if baseline_result is None:
            continue
        baseline_median = float(baseline_result["latency_median_ms"])
        candidate_median = float(candidate_result["latency_median_ms"])
        threshold = max(baseline_median * 1.20, baseline_median + 25.0)
        paired_rows.append(
            {
                "id": candidate_result["id"],
                "baseline_median_ms": baseline_median,
                "candidate_median_ms": candidate_median,
                "delta_ms": candidate_median - baseline_median,
                "ratio": candidate_median / baseline_median if baseline_median else None,
                "threshold_ms": threshold,
                "passed": candidate_median <= threshold,
            }
        )
    paired_success_rate = (
        sum(row["passed"] for row in paired_rows) / len(paired_rows) if paired_rows else 0.0
    )
    paired_latency = {
        "row_count": len(paired_rows),
        "success_rate": paired_success_rate,
        "required_success_rate": 0.90,
        "passed": bool(paired_rows) and paired_success_rate >= 0.90,
        "rows": paired_rows,
    }

    candidate_overall = candidate_aggregates["overall"]
    baseline_overall = baseline_aggregates["overall"]
    candidate_focused = candidate_aggregates["focused"]
    baseline_focused = baseline_aggregates["focused"]
    candidate_coverage = candidate_aggregates["coverage"]
    baseline_coverage = baseline_aggregates["coverage"]
    quality_gates = {
        "all_rows_pass": bool(candidate_results) and candidate_overall["all_quality_pass"],
        "scope_path_recall": candidate_overall["scope_path_recall"] >= 0.95,
        "visible_evidence_recall": (
            candidate_overall["visible_evidence_recall"] >= 0.90
            and candidate_overall["visible_evidence_recall"]
            >= baseline_overall["visible_evidence_recall"]
        ),
        "focused_visible_evidence_recall": (
            candidate_focused["visible_evidence_recall"]
            >= baseline_focused["visible_evidence_recall"]
        ),
        "coverage_relevant_sources": (
            candidate_coverage["relevant_source_coverage_success"] == 1.0
        ),
        "coverage_visible_path_recall": (
            candidate_coverage["visible_path_recall"] >= baseline_coverage["visible_path_recall"]
        ),
        "coverage_first_path_mrr": (
            candidate_coverage["first_path_mrr"] >= baseline_coverage["first_path_mrr"]
        ),
        "candidate_entity_recall": (
            candidate_overall["candidate_entity_recall"]
            >= baseline_overall["candidate_entity_recall"]
        ),
        "response_deterministic": candidate_overall["all_response_deterministic"],
        "response_bounds": all(
            result["metrics"]["response_bound_ok"]
            and result["metrics"]["response_size_bound_ok"]
            for result in candidate_results
        ),
        "focused_evidence_hashes": all(
            result["metrics"]["focused_evidence_hashes_distinct"]
            for result in candidate_results
            if result["retrieval_goal"] == "focused"
        ),
        "coverage_distinct_first_pass": all(
            result["metrics"]["coverage_distinct_first_pass_ok"]
            for result in candidate_results
            if result["retrieval_goal"] == "coverage"
        ),
    }

    critical_results = [result for result in candidate_results if result["critical"]]
    critical_applicable = candidate_report.get("partition") == "all"
    critical_gate = {
        "applicable": critical_applicable,
        "expected_count": 12 if critical_applicable else None,
        "observed_count": len(critical_results),
        "failure_ids": [result["id"] for result in critical_results if not result["quality_pass"]],
        "passed": (
            len(critical_results) == 12 and all(result["quality_pass"] for result in critical_results)
            if critical_applicable
            else None
        ),
    }

    active_identity_failures = (
        _active_identity_failures(candidate_report, accepted_candidate_report)
        if accepted_candidate_report is not None
        else []
    )
    active_identity_passed = not active_identity_failures
    gates_passed = (
        not identity_failures
        and all(quality_gates.values())
        and endpoint_latency["passed"]
        and paired_latency["passed"]
        and (not critical_applicable or critical_gate["passed"] is True)
        and active_identity_passed
    )
    return {
        "identity_valid": not identity_failures,
        "identity_failures": identity_failures,
        "request_differences": request_differences,
        "baseline_projection_row_ids": [result["id"] for result in baseline_projection],
        "baseline_aggregates": baseline_aggregates,
        "candidate_aggregates": candidate_aggregates,
        "quality_gates": quality_gates,
        "critical_gate": critical_gate,
        "endpoint_latency": endpoint_latency,
        "paired_latency": paired_latency,
        "active_identity_passed": active_identity_passed,
        "active_identity_failures": active_identity_failures,
        "gates_passed": gates_passed,
    }


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


def _request_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _get_json(url: str, *, timeout: int, api_key: str | None = None) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=_request_headers(api_key), method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    payload = json.loads(body) if body else {}
    if not isinstance(payload, dict):
        raise ValueError(f"GET {url} did not return a JSON object")
    return payload


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: int,
    api_key: str | None = None,
) -> dict[str, Any]:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=raw, headers=_request_headers(api_key), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code} {url}: {body}") from exc
    result = json.loads(body) if body else {}
    if not isinstance(result, dict):
        raise ValueError(f"POST {url} did not return a JSON object")
    return result


def _query_payload(row: dict[str, Any], *, workspace_id: str | None) -> dict[str, Any]:
    return query_suite_payload(row, workspace_id=workspace_id)


def _request_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return query_request_metadata(row)


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


def _call_get_json(
    get_json: Callable[..., dict[str, Any]],
    url: str,
    *,
    timeout: int,
    api_key: str | None,
) -> dict[str, Any]:
    if api_key is None:
        return get_json(url, timeout=timeout)
    return get_json(url, timeout=timeout, api_key=api_key)


def _call_post_json(
    post_json: Callable[..., dict[str, Any]],
    url: str,
    payload: dict[str, Any],
    *,
    timeout: int,
    api_key: str | None,
) -> dict[str, Any]:
    if api_key is None:
        return post_json(url, payload, timeout=timeout)
    return post_json(url, payload, timeout=timeout, api_key=api_key)


def _effective_request_sha256(payload: dict[str, Any]) -> str:
    comparison_payload = dict(payload)
    comparison_payload.pop("retrieval_goal", None)
    return hashlib.sha256(_canonical_json_bytes(comparison_payload)).hexdigest()


def collect_quality_report(
    *,
    query_suite_path: Path,
    server: str,
    workspace_file: Path,
    runtime_code_root: Path,
    partition: str = "all",
    workspace_id: str | None = None,
    endpoint: str = "/query/data",
    timeout: int = 120,
    warmup_runs: int = 1,
    repetitions: int = 5,
    baseline_report: dict[str, Any] | None = None,
    accepted_candidate_report: dict[str, Any] | None = None,
    api_key: str | None = None,
    get_json: Callable[..., dict[str, Any]] = _get_json,
    post_json: Callable[..., dict[str, Any]] = _post_json,
    timer: Timer = time.perf_counter,
) -> dict[str, Any]:
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if accepted_candidate_report is not None and baseline_report is None:
        raise ValueError("accepted candidate comparison requires a baseline report")
    if baseline_report is None and partition != "all":
        raise ValueError("baseline quality report must collect the all partition")

    all_rows = read_query_suite(query_suite_path, quality_contract=QUALITY_CONTRACT)
    rows = select_quality_partition(all_rows, partition)
    corpus_identity = build_corpus_identity(workspace_file)
    canonical_workspace_id = corpus_identity["workspace_id"]
    if workspace_id is not None and workspace_id != canonical_workspace_id:
        raise ValueError("CLI workspace_id does not match frozen workspace pointer")
    for row in all_rows:
        row_workspace_id = row.get("workspace_id")
        if row_workspace_id is not None and row_workspace_id != canonical_workspace_id:
            raise ValueError(f"row {row['id']} workspace_id does not match frozen workspace pointer")
        if len(row["query_vector"]) != corpus_identity["embedding_dim"]:
            raise ValueError(f"row {row['id']} query_vector dimension does not match frozen workspace")

    runtime_root = Path(runtime_code_root).expanduser().resolve()
    runtime_fingerprint = runtime_package_fingerprint(runtime_root)
    evaluator_root = Path(__file__).resolve().parents[1]
    endpoint_path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    base_url = server.rstrip("/")
    health = _call_get_json(
        get_json,
        base_url + "/health",
        timeout=timeout,
        api_key=api_key,
    )
    if health.get("status") != "ok":
        raise ValueError("quality server health status is not ok")
    if health.get("active_workspace_id") != canonical_workspace_id:
        raise ValueError("quality server health workspace ID does not match frozen workspace")

    report_role = (
        "active"
        if accepted_candidate_report is not None
        else "candidate"
        if baseline_report is not None
        else "baseline"
    )
    forward_goal = report_role != "baseline"
    query_url = base_url + endpoint_path
    results: list[dict[str, Any]] = []
    for row in rows:
        payload = _query_payload(row, workspace_id=canonical_workspace_id)
        if forward_goal:
            payload["retrieval_goal"] = row["retrieval_goal"]
        else:
            payload.pop("retrieval_goal", None)
        for _ in range(warmup_runs):
            _call_post_json(
                post_json,
                query_url,
                payload,
                timeout=timeout,
                api_key=api_key,
            )
        responses: list[dict[str, Any]] = []
        row_latencies: list[float] = []
        for _ in range(repetitions):
            started = timer()
            response = _call_post_json(
                post_json,
                query_url,
                payload,
                timeout=timeout,
                api_key=api_key,
            )
            row_latencies.append(round((timer() - started) * 1000.0, 3))
            responses.append(response)
        result = evaluate_quality_repetitions(
            row,
            responses=responses,
            latencies_ms=row_latencies,
        )
        request_metadata = _request_metadata(row)
        request_metadata["retrieval_goal"] = row["retrieval_goal"]
        result.update(
            {
                "query": row["query"],
                "request": request_metadata,
                "effective_request_sha256": _effective_request_sha256(payload),
            }
        )
        results.append(result)

    report: dict[str, Any] = {
        "quality_contract": QUALITY_CONTRACT,
        "report_contract_version": QUALITY_REPORT_CONTRACT_VERSION,
        "report_role": report_role,
        "partition": partition,
        "query_suite": str(Path(query_suite_path).resolve()),
        "query_suite_sha256": _query_suite_sha256(query_suite_path),
        "selected_row_ids": [row["id"] for row in rows],
        "selected_row_count": len(rows),
        "corpus_identity": corpus_identity,
        "runtime_code_root": str(runtime_root),
        "runtime_package_fingerprint": runtime_fingerprint,
        "runtime_git_revision": runtime_git_revision(runtime_root),
        "evaluator_fingerprint": evaluator_fingerprint(evaluator_root),
        "client_fingerprint": client_fingerprint(evaluator_root),
        "runtime_process_binding": {
            "bound": False,
            "reason": "collector_has_no_controlled-process_attestation",
        },
        "server": base_url,
        "server_health": {
            "status": health.get("status"),
            "active_workspace_id": health.get("active_workspace_id"),
        },
        "endpoint": endpoint_path,
        "measurement_role": "native_relevance_quality",
        "workspace_id": canonical_workspace_id,
        "timing_scope": _timing_scope(rows),
        "warmup_runs": warmup_runs,
        "repetitions": repetitions,
        "results": results,
        "aggregates": aggregate_quality_results(results),
    }
    comparison = (
        compare_quality_reports(
            report,
            baseline_report,
            accepted_candidate_report=accepted_candidate_report,
        )
        if baseline_report is not None
        else None
    )
    report["comparison"] = comparison
    report["gates_passed"] = comparison["gates_passed"] if comparison is not None else None
    return report


def collect_query_report(
    *,
    query_suite_path: Path,
    server: str,
    workspace_id: str | None = None,
    endpoint: str = "/query/data",
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
    rows = read_query_suite(query_suite_path)
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
        "measurement_role": "native_query",
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


def write_report_atomic(path: Path, report: dict[str, Any], *, no_overwrite: bool) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if no_overwrite:
            os.link(temporary, target)
            temporary.unlink()
        else:
            os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report must contain a JSON object: {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect native query report artifacts")
    parser.add_argument("--query-suite", type=Path, required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--workspace-id")
    parser.add_argument("--endpoint", default="/query/data")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--quality-contract", choices=[QUALITY_CONTRACT])
    parser.add_argument("--partition", choices=["calibration", "holdout", "all"], default="all")
    parser.add_argument("--workspace-file", type=Path)
    parser.add_argument("--runtime-code-root", type=Path)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--accepted-candidate-report", type=Path)
    parser.add_argument("--require-gates", action="store_true")
    parser.add_argument("--fail-if-output-exists", action="store_true")
    parser.add_argument("--promote-on-pass", type=Path)
    parser.add_argument("--api-key-env")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    api_key: str | None = None
    try:
        if args.fail_if_output_exists and args.output is None:
            raise ValueError("--fail-if-output-exists requires --output")
        if args.fail_if_output_exists and args.output.exists():
            raise FileExistsError(args.output)
        if args.promote_on_pass is not None:
            if args.output is None:
                raise ValueError("--promote-on-pass requires --output")
            if args.promote_on_pass.resolve() == args.output.resolve():
                raise ValueError("promotion target must differ from attempt output")
            if args.promote_on_pass.exists():
                raise FileExistsError(args.promote_on_pass)

        if args.quality_contract == QUALITY_CONTRACT:
            if args.workspace_file is None or args.runtime_code_root is None:
                raise ValueError("relevance-v1 requires --workspace-file and --runtime-code-root")
            if args.require_gates and args.baseline_report is None:
                raise ValueError("--require-gates requires --baseline-report")
            if args.promote_on_pass is not None and args.partition != "all":
                raise ValueError("only the all partition can be promoted")
            baseline = _read_json_object(args.baseline_report) if args.baseline_report else None
            accepted = (
                _read_json_object(args.accepted_candidate_report)
                if args.accepted_candidate_report
                else None
            )
            if args.api_key_env:
                api_key = os.environ.get(args.api_key_env)
                if not api_key:
                    raise ValueError(f"API key environment variable is unset: {args.api_key_env}")
            report = collect_quality_report(
                query_suite_path=args.query_suite,
                server=args.server,
                workspace_file=args.workspace_file,
                runtime_code_root=args.runtime_code_root,
                partition=args.partition,
                workspace_id=args.workspace_id,
                endpoint=args.endpoint,
                timeout=args.timeout,
                warmup_runs=args.warmup_runs,
                repetitions=args.repetitions,
                baseline_report=baseline,
                accepted_candidate_report=accepted,
                api_key=api_key,
            )
        else:
            quality_only_values = (
                args.workspace_file,
                args.runtime_code_root,
                args.baseline_report,
                args.accepted_candidate_report,
                args.promote_on_pass,
                args.api_key_env,
            )
            if args.partition != "all" or args.require_gates or any(quality_only_values):
                raise ValueError("quality-only options require --quality-contract relevance-v1")
            report = collect_query_report(
                query_suite_path=args.query_suite,
                server=args.server,
                workspace_id=args.workspace_id,
                endpoint=args.endpoint,
                timeout=args.timeout,
                warmup_runs=args.warmup_runs,
                repetitions=args.repetitions,
            )

        if args.output:
            write_report_atomic(args.output, report, no_overwrite=args.fail_if_output_exists)
        if args.require_gates and report.get("gates_passed") is not True:
            print_json(report)
            return 2
        if args.promote_on_pass is not None and report.get("gates_passed") is True:
            write_report_atomic(args.promote_on_pass, report, no_overwrite=True)
    except Exception as exc:
        message = str(exc)
        if api_key:
            message = message.replace(api_key, "[REDACTED]")
        print_json(
            {
                "ok": False,
                "error": "collect_failed",
                "message": message,
                "exception_type": type(exc).__name__,
            }
        )
        return 1

    print_json(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
