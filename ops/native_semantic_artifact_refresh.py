#!/usr/bin/env python3
"""Rebuild and verify semantic artifacts before native wiki cutover.

The batch wiki integration runner calls this owner seam after wiki metadata is
updated and before native materialization. It replaces ad-hoc file/SQLite
inspection with a deterministic rebuild, content-level coverage gate, and a
compact machine-readable report.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from llm_wiki_native.source_docs import collect_source_docs
from ops.custom_kg_incremental import manifest_path
from ops.native_runtime_env import env_int, load_env_file
from ops.wiki_native_cli import common_paths_parser, print_json
from ops.wiki_native_raw_section_extract import extract_raw_note_sections
from ops.wiki_native_wiki_integration_pending import load_pending_wiki_integration_ledger

StageRunner = Callable[[str, list[str], Path, dict[str, str], int], dict[str, Any]]

PARALLEL_STAGES = (
    ("method_atoms", "ops.extract_method_atoms"),
    ("raw_sections", "ops.extract_raw_sections"),
    ("seed_edges", "ops.build_seed_edges"),
)
SERIAL_STAGES = (
    ("section_similarity", "ops.build_section_similarity_graph"),
    ("custom_kg_manifest", "ops.custom_kg_incremental", "export-manifest"),
)
REPORT_VERSION = 1


class SemanticArtifactRefreshError(RuntimeError):
    def __init__(self, message: str, *, report: dict[str, Any]):
        super().__init__(message)
        self.report = report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _compact_stage_payload(payload: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "atom_count",
        "candidate_edges_path",
        "desired_manifest_hash",
        "edge_count",
        "embedding_path",
        "embedding_stats",
        "manifest",
        "manifest_path",
        "method_atom_count",
        "payload",
        "raw_section_count",
        "report_path",
        "section_count",
        "section_similarity_index",
        "seed_edge_count",
        "source_count",
    }
    return {key: payload[key] for key in sorted(payload) if key in keep}


def _redacted_tail(text: str, env: dict[str, str], max_chars: int = 1200) -> str:
    redacted = text
    for key, value in env.items():
        if value and any(token in key.lower() for token in ("key", "token", "secret", "password")):
            redacted = redacted.replace(value, "[REDACTED]")
    return redacted.strip()[-max_chars:]


def run_stage_subprocess(
    stage: str,
    command: list[str],
    workdir: Path,
    env: dict[str, str],
    timeout: int,
) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=workdir,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    duration = round(time.monotonic() - started, 3)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{stage} failed with exit {completed.returncode}: "
            f"{_redacted_tail(completed.stderr or completed.stdout, env)}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{stage} returned non-JSON output: {_redacted_tail(completed.stdout, env)}") from exc
    if not isinstance(payload, dict) or payload.get("ok") is False or payload.get("error"):
        raise RuntimeError(
            f"{stage} returned failure payload: "
            f"{_redacted_tail(json.dumps(payload, ensure_ascii=False), env)}"
        )
    return {
        "stage": stage,
        "exit_code": completed.returncode,
        "duration_seconds": duration,
        "result": _compact_stage_payload(payload),
    }


def _runtime_contract(workdir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    loaded = load_env_file(workdir / ".env")
    contract = {
        "embedding_model": os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        "embedding_dim": env_int("EMBEDDING_DIM", 1536),
        "embedding_params_version": os.environ.get("EMBEDDING_PARAMS_VERSION", "v1"),
    }
    return contract, loaded


def _vector_cache_contracts(state_dir: Path) -> list[dict[str, Any]]:
    path = state_dir / "vector_cache.sqlite"
    if not path.exists():
        return []
    with sqlite3.connect(path) as conn:
        table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vector_cache'"
        ).fetchone()
        if table is None:
            return []
        rows = conn.execute(
            "SELECT embedding_model, embedding_dim, embedding_params_version, COUNT(*) "
            "FROM vector_cache GROUP BY embedding_model, embedding_dim, embedding_params_version "
            "ORDER BY COUNT(*) DESC"
        ).fetchall()
    return [
        {
            "embedding_model": str(model),
            "embedding_dim": int(dim),
            "embedding_params_version": str(version),
            "rows": int(count),
        }
        for model, dim, version, count in rows
    ]


def _integration_paths(state_dir: Path, requested: list[str] | None) -> list[str]:
    if requested:
        values = requested
    else:
        ledger = load_pending_wiki_integration_ledger(state_dir)
        values = list(ledger.get("last_integrated_paths") or [])
    return sorted({str(value).replace("\\", "/").lstrip("./") for value in values if value})


def _artifact_paths(state_dir: Path) -> dict[str, Path]:
    return {
        "raw_sections": state_dir / "raw_sections.jsonl",
        "method_atoms": state_dir / "method_atoms.jsonl",
        "seed_edges": state_dir / "seed_edges.jsonl",
        "section_embeddings": state_dir / "section_embeddings.jsonl",
        "section_similarity_edges": state_dir / "section_similarity_edges.candidates.jsonl",
        "custom_kg_manifest": manifest_path(state_dir),
    }


def validate_semantic_artifacts(
    root: Path,
    state_dir: Path,
    *,
    integrated_paths: list[str],
    runtime_contract: dict[str, Any],
    allow_embedding_contract_change: bool = False,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    normalized_paths = _integration_paths(state_dir, integrated_paths)
    docs = {str(doc.rel_path).replace("\\", "/"): doc for doc in collect_source_docs(root)}
    actual_sections = {
        str(row.get("section_id")): row
        for row in _jsonl_rows(state_dir / "raw_sections.jsonl")
        if row.get("section_id")
    }

    expected_source_ids: set[str] = set()
    expected_section_count = 0
    covered_paths: list[str] = []
    for rel_path in normalized_paths:
        doc = docs.get(rel_path)
        if doc is None:
            failures.append({"code": "integrated-source-missing", "path": rel_path})
            continue
        expected_source_ids.add(str(doc.canonical_id))
        expected_sections = extract_raw_note_sections(doc)
        expected_section_count += len(expected_sections)
        path_ok = True
        for expected in expected_sections:
            section_id = str(expected["section_id"])
            expected_source_ids.add(section_id)
            actual = actual_sections.get(section_id)
            if actual is None:
                failures.append({"code": "raw-section-missing", "path": rel_path, "section_id": section_id})
                path_ok = False
                continue
            if str(actual.get("content") or "").strip() != str(expected.get("content") or "").strip():
                failures.append({"code": "raw-section-stale", "path": rel_path, "section_id": section_id})
                path_ok = False
        if path_ok:
            covered_paths.append(rel_path)

    manifest_file = manifest_path(state_dir)
    manifest: dict[str, Any] = {}
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    else:
        failures.append({"code": "custom-kg-manifest-missing", "path": str(manifest_file)})
    manifest_sources = {
        str(record.get("source_id"))
        for record in (manifest.get("chunks", {}) or {}).values()
        if isinstance(record, dict) and record.get("source_id")
    }
    missing_manifest_sources = sorted(expected_source_ids - manifest_sources)
    if missing_manifest_sources:
        failures.append(
            {
                "code": "custom-kg-source-coverage-missing",
                "count": len(missing_manifest_sources),
                "sample": missing_manifest_sources[:10],
            }
        )

    manifest_metadata = manifest.get("metadata") or {}
    manifest_contract = {
        "embedding_model": manifest_metadata.get("embedding_model"),
        "embedding_dim": manifest_metadata.get("embedding_dim"),
        "embedding_params_version": manifest_metadata.get("embedding_params_version"),
    }
    if manifest_contract != runtime_contract:
        failures.append(
            {
                "code": "manifest-runtime-embedding-contract-mismatch",
                "runtime": runtime_contract,
                "manifest": manifest_contract,
            }
        )

    cache_contracts = _vector_cache_contracts(state_dir)
    cache_keys = {
        (row["embedding_model"], row["embedding_dim"], row["embedding_params_version"])
        for row in cache_contracts
    }
    runtime_key = (
        runtime_contract["embedding_model"],
        runtime_contract["embedding_dim"],
        runtime_contract["embedding_params_version"],
    )
    if cache_keys and runtime_key not in cache_keys and not allow_embedding_contract_change:
        failures.append(
            {
                "code": "vector-cache-embedding-contract-change-requires-opt-in",
                "runtime": runtime_contract,
                "cache_contracts": cache_contracts,
            }
        )

    artifacts = {}
    for name, path in _artifact_paths(state_dir).items():
        if path.exists():
            artifacts[name] = {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
        else:
            failures.append({"code": "semantic-artifact-missing", "artifact": name, "path": str(path)})

    return {
        "ok": not failures,
        "integrated_path_count": len(normalized_paths),
        "covered_path_count": len(covered_paths),
        "expected_section_count": expected_section_count,
        "expected_manifest_source_count": len(expected_source_ids),
        "missing_manifest_source_count": len(missing_manifest_sources),
        "runtime_embedding_contract": runtime_contract,
        "vector_cache_contracts": cache_contracts,
        "failures": failures,
        "artifacts": artifacts,
    }


def validate_active_workspace_coverage(
    root: Path,
    state_dir: Path,
    *,
    integrated_paths: list[str],
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    pointer_path = state_dir / "native_zvec" / "active_workspace.json"
    if not pointer_path.exists():
        return {
            "ok": False,
            "workspace_id": None,
            "pointer_path": str(pointer_path),
            "covered_path_count": 0,
            "failures": [{"code": "active-workspace-pointer-missing", "path": str(pointer_path)}],
            "paths": [],
        }
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    workspace_id = pointer.get("workspace_id")
    pointer_source_root = str(pointer.get("source_root") or "")
    if pointer_source_root and Path(pointer_source_root).resolve() != root.resolve():
        failures.append(
            {
                "code": "active-workspace-source-root-mismatch",
                "expected": str(root.resolve()),
                "actual": pointer_source_root,
            }
        )
    sqlite_value = pointer.get("sqlite_path")
    sqlite_path = Path(str(sqlite_value)) if sqlite_value else Path()
    if not sqlite_value or not sqlite_path.exists():
        failures.append({"code": "active-workspace-sqlite-missing", "path": str(sqlite_path)})
        return {
            "ok": False,
            "workspace_id": workspace_id,
            "pointer_path": str(pointer_path),
            "sqlite_path": str(sqlite_path),
            "covered_path_count": 0,
            "failures": failures,
            "paths": [],
        }

    normalized_paths = _integration_paths(state_dir, integrated_paths)
    docs = {str(doc.rel_path).replace("\\", "/"): doc for doc in collect_source_docs(root)}
    path_results: list[dict[str, Any]] = []
    with sqlite3.connect(sqlite_path) as conn:
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if "record" not in tables or "lexical_span" not in tables:
            failures.append(
                {
                    "code": "active-workspace-coverage-table-missing",
                    "missing": sorted({"record", "lexical_span"} - tables),
                }
            )
        else:
            for rel_path in normalized_paths:
                doc = docs.get(rel_path)
                expected_sections = len(extract_raw_note_sections(doc)) if doc is not None else 0
                record_count = int(
                    conn.execute("SELECT COUNT(*) FROM record WHERE source_path = ?", (rel_path,)).fetchone()[0]
                )
                section_count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM record WHERE source_path = ? AND record_type = 'section'",
                        (rel_path,),
                    ).fetchone()[0]
                )
                lexical_count = int(
                    conn.execute("SELECT COUNT(*) FROM lexical_span WHERE source_path = ?", (rel_path,)).fetchone()[0]
                )
                path_ok = record_count > 0 and lexical_count > 0 and section_count >= expected_sections
                path_results.append(
                    {
                        "path": rel_path,
                        "ok": path_ok,
                        "expected_section_count": expected_sections,
                        "record_count": record_count,
                        "section_record_count": section_count,
                        "lexical_span_count": lexical_count,
                    }
                )
                if not path_ok:
                    failures.append(
                        {
                            "code": "active-workspace-source-coverage-missing",
                            "path": rel_path,
                            "expected_section_count": expected_sections,
                            "record_count": record_count,
                            "section_record_count": section_count,
                            "lexical_span_count": lexical_count,
                        }
                    )

    return {
        "ok": not failures,
        "workspace_id": workspace_id,
        "pointer_path": str(pointer_path),
        "sqlite_path": str(sqlite_path),
        "integrated_path_count": len(normalized_paths),
        "covered_path_count": sum(1 for row in path_results if row["ok"]),
        "failures": failures,
        "paths": path_results,
    }


def refresh_semantic_artifacts(
    root: Path,
    state_dir: Path,
    *,
    workdir: Path,
    integrated_paths: list[str] | None = None,
    allow_embedding_contract_change: bool = False,
    stage_runner: StageRunner = run_stage_subprocess,
    timeout: int = 7200,
) -> dict[str, Any]:
    root = root.resolve()
    state_dir = state_dir.resolve()
    workdir = workdir.resolve()
    paths = _integration_paths(state_dir, integrated_paths)
    runtime_contract, loaded_env = _runtime_contract(workdir)
    env = os.environ.copy()
    common = ["--root", str(root), "--state-dir", str(state_dir), "--workdir", str(workdir)]

    report: dict[str, Any] = {
        "version": REPORT_VERSION,
        "ok": False,
        "integrated_paths": paths,
        "runtime_embedding_contract": runtime_contract,
        "env_file_loaded": bool(loaded_env),
        "stages": [],
    }
    try:
        parallel_commands = [
            (stage, [sys.executable, "-m", module, *common])
            for stage, module in PARALLEL_STAGES
        ]
        parallel_results: dict[str, dict[str, Any]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(parallel_commands)) as executor:
            futures = {
                stage: executor.submit(stage_runner, stage, command, workdir, env, timeout)
                for stage, command in parallel_commands
            }
            for stage, _command in parallel_commands:
                parallel_results[stage] = futures[stage].result()
        report["stages"].extend(parallel_results[stage] for stage, _module in PARALLEL_STAGES)

        for spec in SERIAL_STAGES:
            stage, module, *subcommand = spec
            command = [sys.executable, "-m", module, *subcommand, *common]
            report["stages"].append(stage_runner(stage, command, workdir, env, timeout))

        try:
            validation = validate_semantic_artifacts(
                root,
                state_dir,
                integrated_paths=paths,
                runtime_contract=runtime_contract,
                allow_embedding_contract_change=allow_embedding_contract_change,
            )
        except Exception as exc:  # pragma: no cover - defensive conversion for unattended runs
            validation = {
                "ok": False,
                "integrated_paths": paths,
                "failures": [
                    {
                        "code": "semantic-artifact-validator-error",
                        "error_type": type(exc).__name__,
                        "message": _redacted_tail(str(exc), env),
                    }
                ],
            }
        report["validation"] = validation
        report["ok"] = validation["ok"]
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": _redacted_tail(str(exc), env)}

    report_dir = state_dir / "semantic_artifact_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_semantic_artifact_refresh.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if not report["ok"]:
        raise SemanticArtifactRefreshError("semantic artifact refresh did not pass its coverage gate", report=report)
    return report


def main() -> int:
    parser = common_paths_parser("Rebuild and verify semantic artifacts before native cutover")
    parser.add_argument("--integrated-path", action="append", default=[])
    parser.add_argument("--allow-embedding-contract-change", action="store_true")
    args = parser.parse_args()
    try:
        result = refresh_semantic_artifacts(
            args.root,
            args.state_dir,
            workdir=args.workdir,
            integrated_paths=args.integrated_path or None,
            allow_embedding_contract_change=args.allow_embedding_contract_change,
        )
    except SemanticArtifactRefreshError as exc:
        print_json(exc.report)
        return 18
    print_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
