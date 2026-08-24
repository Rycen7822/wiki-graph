#!/usr/bin/env python3
"""Manage pending llm-wiki native zvec refreshes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sqlite3
import struct
import subprocess
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse
import urllib.request

from llm_wiki_native.contracts import DEFAULT_QUERY_MODE
from llm_wiki_native.query_contract import query_vector as normalize_native_query_vector

PENDING_NATIVE_REFRESH_LEDGER = "pending_native_refresh.json"
NATIVE_INCREMENTAL_REFRESH_THRESHOLD = 5
FULL_REBUILD_DUE_REASON = "policy:full-rebuild-due-after-incrementals"
REFRESH_KIND_INCREMENTAL = "incremental"
REFRESH_KIND_FULL_REBUILD = "full-rebuild"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if not raw:
        return default
    return Path(raw).expanduser()


DEFAULT_BASE_WORKDIR = _env_path("LLM_WIKI_WORKDIR", _env_path("WIKI_GRAPH_REPO", REPO_ROOT))
DEFAULT_WIKI_ROOT = _env_path("LLM_WIKI_ROOT", DEFAULT_BASE_WORKDIR)
DEFAULT_WORKDIR = _env_path("LLM_WIKI_NATIVE_REFRESH_WORKDIR", DEFAULT_BASE_WORKDIR / "tmp" / "native_refresh")


def pending_ledger_path(state_dir: Path) -> Path:
    return Path(state_dir) / PENDING_NATIVE_REFRESH_LEDGER


def native_dir(state_dir: Path) -> Path:
    return Path(state_dir) / "native_zvec"


def default_workspace_root(state_dir: Path) -> Path:
    return native_dir(state_dir) / "workspaces"


def prepared_workspace_path(state_dir: Path) -> Path:
    return native_dir(state_dir) / "prepared_workspace.json"


def active_workspace_path(state_dir: Path) -> Path:
    return native_dir(state_dir) / "active_workspace.json"


def active_workspace_history_path(state_dir: Path) -> Path:
    return native_dir(state_dir) / "active_workspace.history.jsonl"


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def refresh_kind_from_reason(reason: Any) -> str | None:
    text = str(reason or "").lower()
    if REFRESH_KIND_FULL_REBUILD in text or "full rebuild" in text:
        return REFRESH_KIND_FULL_REBUILD
    if REFRESH_KIND_INCREMENTAL in text:
        return REFRESH_KIND_INCREMENTAL
    return None


def completed_incremental_refresh_count(state_dir: Path) -> int:
    count = 0
    for row in reversed(_read_jsonl_objects(active_workspace_history_path(state_dir))):
        kind = refresh_kind_from_reason(row.get("reason"))
        if kind == REFRESH_KIND_FULL_REBUILD:
            break
        if kind == REFRESH_KIND_INCREMENTAL:
            count += 1
    return count


def next_refresh_kind(state_dir: Path) -> str:
    if completed_incremental_refresh_count(state_dir) >= NATIVE_INCREMENTAL_REFRESH_THRESHOLD:
        return REFRESH_KIND_FULL_REBUILD
    return REFRESH_KIND_INCREMENTAL


def native_refresh_reason(kind: str, reason: str = "native refresh") -> str:
    normalized = kind if kind in {REFRESH_KIND_INCREMENTAL, REFRESH_KIND_FULL_REBUILD} else REFRESH_KIND_INCREMENTAL
    return f"native graph {normalized} refresh: {reason}"


def native_refresh_policy_status(state_dir: Path) -> dict[str, Any]:
    completed_incrementals = completed_incremental_refresh_count(state_dir)
    return {
        "incremental_rebuild_threshold": NATIVE_INCREMENTAL_REFRESH_THRESHOLD,
        "completed_incremental_refresh_count": completed_incrementals,
        "next_refresh_kind": REFRESH_KIND_FULL_REBUILD
        if completed_incrementals >= NATIVE_INCREMENTAL_REFRESH_THRESHOLD
        else REFRESH_KIND_INCREMENTAL,
        "vector_cache_required": True,
        "vector_cache_path": str(Path(state_dir) / "vector_cache.sqlite"),
        "history_path": str(active_workspace_history_path(state_dir)),
    }


def _path_snapshot(path: Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {"exists": False}
    if target.is_file():
        stat = target.stat()
        return {
            "exists": True,
            "kind": "file",
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    entries: list[dict[str, Any]] = []
    for child in sorted(target.rglob("*")):
        if not (child.is_file() or child.is_dir()):
            continue
        stat = child.stat()
        entries.append(
            {
                "path": child.relative_to(target).as_posix(),
                "kind": "dir" if child.is_dir() else "file",
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return {"exists": True, "kind": "dir", "entries": entries}


def snapshot_required_unchanged_paths(paths: list[Path] | None) -> dict[str, dict[str, Any]]:
    return {str(Path(path).resolve()): _path_snapshot(Path(path).resolve()) for path in paths or []}


def audit_required_unchanged_paths(before: dict[str, dict[str, Any]]) -> dict[str, Any]:
    paths: list[dict[str, Any]] = []
    for path, before_snapshot in before.items():
        changed = _path_snapshot(Path(path)) != before_snapshot
        paths.append({"path": path, "ok": not changed, "changed": changed})
    return {"ok": all(row["ok"] for row in paths), "paths": paths}


def assert_required_unchanged_paths(before: dict[str, dict[str, Any]]) -> dict[str, Any]:
    audit = audit_required_unchanged_paths(before)
    if not audit["ok"]:
        changed = [row["path"] for row in audit["paths"] if row["changed"]]
        raise RuntimeError(f"required unchanged path changed: {changed}")
    return audit


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def pending_entries(state_dir: Path) -> list[dict[str, Any]]:
    ledger = _read_json(pending_ledger_path(state_dir))
    pending = ledger.get("pending", [])
    if not isinstance(pending, list):
        raise ValueError(f"{PENDING_NATIVE_REFRESH_LEDGER} pending must be a list")
    return [dict(item) for item in pending if isinstance(item, dict)]


def status_has_wiki_integration_pending(current_status: dict[str, Any]) -> bool:
    """Return true when pending native work came from successful wiki integration."""

    pending = current_status.get("pending") or []
    if not isinstance(pending, list):
        return False
    return any(
        isinstance(item, dict) and str(item.get("reason") or "").startswith("wiki-integration:")
        for item in pending
    )


def mark_pending(state_dir: Path, root: Path, *, reason: str) -> dict[str, Any]:
    entries = pending_entries(state_dir)
    entry = {
        "reason": reason,
        "root": str(Path(root)),
        "marked_at": _utc_now(),
        "status": "pending",
    }
    entries.append(entry)
    _write_json_atomic(
        pending_ledger_path(state_dir),
        {
            "schema_version": 1,
            "pending": entries,
            "updated_at": entry["marked_at"],
        },
    )
    return entry


def mark_full_rebuild_pending_if_due(
    state_dir: Path,
    root: Path,
    *,
    refresh_kind: str,
    status_after_policy: dict[str, Any],
) -> dict[str, Any]:
    if refresh_kind == REFRESH_KIND_FULL_REBUILD:
        return {"marked": False, "reason": "full_rebuild_completed"}
    if status_after_policy.get("next_refresh_kind") != REFRESH_KIND_FULL_REBUILD:
        return {"marked": False, "reason": "full_rebuild_not_due"}
    for entry in pending_entries(state_dir):
        if entry.get("reason") == FULL_REBUILD_DUE_REASON:
            return {"marked": False, "reason": "already_pending", "entry": entry}
    entry = mark_pending(state_dir, root, reason=FULL_REBUILD_DUE_REASON)
    return {"marked": True, "reason": FULL_REBUILD_DUE_REASON, "entry": entry}


def status(root: Path, state_dir: Path) -> dict[str, Any]:
    entries = pending_entries(state_dir)
    refresh_policy = native_refresh_policy_status(state_dir)
    return {
        "root": str(root),
        "state_dir": str(state_dir),
        "ledger_path": str(pending_ledger_path(state_dir)),
        "pending_count": len(entries),
        "pending": entries,
        "should_refresh": bool(entries),
        "native_dir": str(native_dir(state_dir)),
        "prepared_workspace": str(prepared_workspace_path(state_dir)),
        "active_workspace": str(active_workspace_path(state_dir)),
        "history_path": str(active_workspace_history_path(state_dir)),
        "refresh_policy": refresh_policy,
        "next_refresh_kind": refresh_policy["next_refresh_kind"],
        "completed_incremental_refresh_count": refresh_policy["completed_incremental_refresh_count"],
        "incremental_rebuild_threshold": refresh_policy["incremental_rebuild_threshold"],
        "vector_cache_required": refresh_policy["vector_cache_required"],
        "vector_cache_path": refresh_policy["vector_cache_path"],
    }


def build_prepared_workspace(
    *,
    root: Path,
    state_dir: Path,
    workspace_root: Path,
    workspace_id: str,
    embedding_profile: str,
    fill_missing_vectors: bool = True,
) -> dict[str, Any]:
    from ops import native_zvec_materialize

    return native_zvec_materialize.build(
        SimpleNamespace(
            root=root,
            state_dir=state_dir,
            workspace_root=workspace_root,
            workspace_id=workspace_id,
            embedding_profile=embedding_profile,
            fill_missing_vectors=fill_missing_vectors,
            prepare_only=True,
        )
    )


def refresh_prepare_only(
    *,
    root: Path,
    state_dir: Path,
    workspace_root: Path,
    workspace_id: str,
    embedding_profile: str,
    fill_missing_vectors: bool = True,
    force: bool = False,
    required_unchanged_paths: list[Path] | None = None,
) -> dict[str, Any]:
    before_active = active_workspace_path(state_dir).read_text(encoding="utf-8") if active_workspace_path(state_dir).exists() else None
    unchanged_before = snapshot_required_unchanged_paths(required_unchanged_paths)
    current_status = status(root, state_dir)
    refresh_policy = current_status.get("refresh_policy") or native_refresh_policy_status(state_dir)
    refresh_kind = refresh_policy["next_refresh_kind"]
    if not current_status["should_refresh"] and not force:
        result = {
            "prepared_only": True,
            "skipped": True,
            "status": current_status,
            "refresh_kind": refresh_kind,
            "refresh_policy": refresh_policy,
            "vector_cache_required": True,
            "fill_missing_vectors": fill_missing_vectors,
        }
        if unchanged_before:
            result["unchanged_path_audit"] = assert_required_unchanged_paths(unchanged_before)
        return result
    report = build_prepared_workspace(
        root=root,
        state_dir=state_dir,
        workspace_root=workspace_root,
        workspace_id=workspace_id,
        embedding_profile=embedding_profile,
        fill_missing_vectors=fill_missing_vectors,
    )
    after_active = active_workspace_path(state_dir).read_text(encoding="utf-8") if active_workspace_path(state_dir).exists() else None
    result = {
        "prepared_only": True,
        "skipped": False,
        "active_workspace_unchanged": before_active == after_active,
        "pending_retained": pending_ledger_path(state_dir).exists(),
        "build": report,
        "status_before": current_status,
        "refresh_kind": refresh_kind,
        "refresh_policy": refresh_policy,
        "vector_cache_required": True,
        "fill_missing_vectors": fill_missing_vectors,
    }
    if unchanged_before:
        result["unchanged_path_audit"] = assert_required_unchanged_paths(unchanged_before)
    return result


def clear_pending(state_dir: Path) -> bool:
    path = pending_ledger_path(state_dir)
    if not path.exists():
        return False
    path.unlink()
    return True


def finalize_prepared_workspace_for_state(*, state_dir: Path, reason: str) -> dict[str, Any]:
    from ops import native_zvec_materialize

    workspace_root = default_workspace_root(state_dir)
    return native_zvec_materialize.finalize_prepared_workspace(
        native_zvec_materialize.prepared_workspace_path(workspace_root),
        native_zvec_materialize.active_workspace_path(workspace_root),
        native_zvec_materialize.history_path(workspace_root),
        reason=reason,
    )


def restart_service_command(
    command: list[str],
    *,
    health_url: str | None = None,
    timeout_seconds: float = 10.0,
    runner: Any = subprocess.run,
    urlopen: Any = urllib.request.urlopen,
) -> dict[str, Any]:
    if not command:
        raise ValueError("restart command must not be empty")
    completed = runner(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    restart = {
        "command": list(command),
        "returncode": int(completed.returncode),
        "stdout": str(completed.stdout),
        "stderr": str(completed.stderr),
    }
    if int(completed.returncode) != 0:
        raise RuntimeError(f"restart command failed: returncode={completed.returncode} stderr={completed.stderr}")
    result: dict[str, Any] = {
        "service": "llm-wiki-native",
        "status": "ok",
        "restart": restart,
    }
    if health_url:
        response = urlopen(health_url, timeout=timeout_seconds)
        try:
            body = response.read().decode("utf-8", errors="replace")
            status_code = int(getattr(response, "status", getattr(response, "code", 0)) or 0)
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
        health = {
            "url": health_url,
            "status": status_code,
            "ok": 200 <= status_code < 300,
            "body": body,
        }
        if not health["ok"]:
            raise RuntimeError(f"health check failed after restart: url={health_url} status={status_code}")
        result["health"] = health
    return result


def restart_service_from_args(args: argparse.Namespace) -> Any:
    command = shlex.split(str(args.restart_command or ""))
    if not command:
        raise ValueError("--restart-command must not be empty for --cutover")

    def restart_service(*, state_dir: Path) -> dict[str, Any]:
        result = restart_service_command(
            command,
            health_url=args.health_url,
            timeout_seconds=args.health_timeout,
        )
        result["state_dir"] = str(state_dir)
        return result

    return restart_service


def query_smoke_request(
    url: str,
    *,
    query: str,
    mode: str = DEFAULT_QUERY_MODE,
    workspace_id: str | None = None,
    query_vector: list[float] | None = None,
    query_vector_source: str | None = None,
    timeout_seconds: float = 10.0,
    urlopen: Any = urllib.request.urlopen,
) -> dict[str, Any]:
    if mode != "mix":
        raise ValueError(f"native refresh cutover query smoke requires mix mode; got {mode}")
    if urlparse(url).path != "/query/data":
        raise ValueError("native refresh cutover query smoke requires /query/data endpoint")
    payload: dict[str, Any] = {"query": query, "mode": mode}
    request_summary: dict[str, Any] = {"query": query, "mode": mode}
    if workspace_id:
        payload["workspace_id"] = workspace_id
        request_summary["workspace_id"] = workspace_id
    if query_vector is not None:
        vector = normalize_query_vector(query_vector)
        payload["query_vector"] = vector
        request_summary["query_vector_present"] = True
        request_summary["query_vector_dim"] = len(vector)
        if query_vector_source:
            request_summary["query_vector_source"] = query_vector_source
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    response = urlopen(request, timeout=timeout_seconds)
    try:
        body_text = response.read().decode("utf-8", errors="replace")
        status_code = int(getattr(response, "status", getattr(response, "code", 0)) or 0)
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()
    try:
        body = json.loads(body_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"query smoke returned non-JSON response: url={url} status={status_code}") from exc
    result = {
        "url": url,
        "status": status_code,
        "ok": 200 <= status_code < 300,
        "request": request_summary,
        "body": body,
    }
    if not result["ok"]:
        raise RuntimeError(f"query smoke failed: url={url} status={status_code}")
    trace = body.get("trace") if isinstance(body, dict) else None
    retrieval_backend = trace.get("retrieval_backend") if isinstance(trace, dict) else None
    if not isinstance(trace, dict) or not str(retrieval_backend or "").startswith("zvec"):
        raise RuntimeError(f"query smoke did not prove zvec retrieval: url={url} status={status_code}")
    vector_hit_count = trace.get("vector_hit_count")
    if type(vector_hit_count) is not int or vector_hit_count <= 0:
        raise RuntimeError(f"query smoke did not return positive zvec hits: url={url} status={status_code}")
    return result


def normalize_query_vector(value: Any) -> list[float]:
    try:
        return normalize_native_query_vector(value)
    except ValueError as exc:
        if str(exc) in {"query_vector must be a list of finite numbers", "query_vector must not be empty"}:
            raise ValueError("query_vector must be a non-empty JSON list") from exc
        raise


def parse_query_vector_json(text: str, *, source: str) -> list[float]:
    payload = json.loads(text)
    if isinstance(payload, dict) and "query_vector" in payload:
        payload = payload["query_vector"]
    try:
        return normalize_query_vector(payload)
    except ValueError as exc:
        raise ValueError(f"{source} query_vector is invalid: {exc}") from exc


def load_query_vector_file(path: Path) -> list[float]:
    target = Path(path)
    return parse_query_vector_json(target.read_text(encoding="utf-8"), source=str(target))


def load_active_first_query_vector(active: dict[str, Any]) -> tuple[list[float], str]:
    workspace_id = active.get("workspace_id")
    sqlite_path = active.get("sqlite_path")
    if not workspace_id or not sqlite_path:
        raise ValueError("active-first-vector source requires active workspace_id and sqlite_path")
    with sqlite3.connect(Path(sqlite_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT record_type, record_id, dim, vector_blob
            FROM vector
            WHERE workspace_id = ?
            ORDER BY record_type, record_id
            LIMIT 1
            """,
            (str(workspace_id),),
        ).fetchone()
    if row is None:
        raise ValueError(f"active-first-vector source found no vectors for workspace_id={workspace_id}")
    dim = int(row["dim"])
    blob = bytes(row["vector_blob"])
    expected_bytes = dim * 4
    if len(blob) != expected_bytes:
        raise ValueError(f"active-first-vector blob has {len(blob)} bytes, expected {expected_bytes}")
    vector = [float(value) for value in struct.unpack(f"<{dim}f", blob)]
    source = f"active-first-vector:{row['record_type']}:{row['record_id']}"
    return vector, source


def static_query_vector_from_args(args: argparse.Namespace) -> tuple[list[float] | None, str | None]:
    vector_json = getattr(args, "smoke_query_vector_json", None)
    vector_file = getattr(args, "smoke_query_vector_file", None)
    vector_source = getattr(args, "smoke_query_vector_source", None)
    provided = [name for name, value in [("json", vector_json), ("file", vector_file), ("source", vector_source)] if value]
    if len(provided) > 1:
        raise ValueError("provide only one of --smoke-query-vector-json, --smoke-query-vector-file, or --smoke-query-vector-source")
    if vector_json:
        return parse_query_vector_json(str(vector_json), source="--smoke-query-vector-json"), "inline-json"
    if vector_file:
        path = Path(vector_file)
        return load_query_vector_file(path), f"file:{path}"
    if vector_source and vector_source != "active-first-vector":
        raise ValueError(f"unsupported --smoke-query-vector-source: {vector_source}")
    return None, None


def query_smoke_from_args(args: argparse.Namespace) -> Any | None:
    if bool(args.smoke_url) != bool(args.smoke_query):
        raise ValueError("--smoke-url and --smoke-query must be provided together")
    if not args.smoke_url:
        raise ValueError("native refresh cutover requires explicit --smoke-url and --smoke-query guards")
    if args.smoke_mode != "mix":
        raise ValueError(f"native refresh cutover query smoke requires mix mode; got {args.smoke_mode}")
    if urlparse(args.smoke_url).path != "/query/data":
        raise ValueError("native refresh cutover query smoke requires /query/data endpoint")
    static_query_vector, static_query_vector_source = static_query_vector_from_args(args)

    def query_smoke(*, state_dir: Path, active: dict[str, Any]) -> dict[str, Any]:
        workspace_id = args.smoke_workspace_id or active.get("workspace_id") or args.workspace_id
        query_vector = static_query_vector
        query_vector_source = static_query_vector_source
        if getattr(args, "smoke_query_vector_source", None) == "active-first-vector":
            query_vector, query_vector_source = load_active_first_query_vector(active)
        result = query_smoke_request(
            args.smoke_url,
            query=args.smoke_query,
            mode=args.smoke_mode,
            workspace_id=str(workspace_id) if workspace_id else None,
            query_vector=query_vector,
            query_vector_source=query_vector_source,
            timeout_seconds=args.smoke_timeout,
        )
        result["state_dir"] = str(state_dir)
        return result

    return query_smoke


def require_existing_cutover_state_dir(state_dir: Path) -> None:
    target = Path(state_dir)
    if not target.exists():
        raise ValueError(f"native refresh cutover state_dir must exist before cutover: {target}")
    if not target.is_dir():
        raise ValueError(f"native refresh cutover state_dir must be a directory: {target}")


def require_existing_unchanged_paths(paths: list[Path] | None) -> None:
    for path in paths or []:
        target = Path(path)
        if not target.exists():
            raise ValueError(f"native refresh cutover --require-unchanged-path must exist before cutover: {target}")


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def require_unchanged_paths_outside_native_outputs(paths: list[Path] | None, *, workspace_root: Path) -> None:
    native_output_root = Path(workspace_root).resolve(strict=False).parent
    for path in paths or []:
        target = Path(path).resolve(strict=False)
        if _paths_overlap(target, native_output_root):
            raise ValueError(
                "native refresh cutover --require-unchanged-path must watch existing non-native storage; "
                f"path overlaps native output root: {target}"
            )


def cutover_guard_report(
    *,
    state_dir: Path,
    workspace_root: Path,
    restart_command: str | None,
    smoke_url: str | None,
    smoke_query: str | None,
    smoke_mode: str = DEFAULT_QUERY_MODE,
    required_unchanged_paths: list[Path] | None = None,
    check_paths: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    required_unchanged_paths = required_unchanged_paths or []
    if not str(restart_command or "").strip():
        errors.append("missing_restart_command")
    if smoke_url and not smoke_query:
        errors.append("missing_smoke_query")
    if smoke_query and not smoke_url:
        errors.append("missing_smoke_url")
    if not smoke_url and not smoke_query:
        errors.extend(["missing_smoke_url", "missing_smoke_query"])
    if smoke_mode != "mix":
        errors.append("invalid_smoke_mode")
    if smoke_url and urlparse(smoke_url).path != "/query/data":
        errors.append("invalid_smoke_endpoint")
    if not required_unchanged_paths:
        errors.append("missing_required_unchanged_path")
    path_errors: list[dict[str, str]] = []
    if check_paths:
        target_state = Path(state_dir)
        if not target_state.exists():
            errors.append("missing_state_dir")
            path_errors.append({"path": str(target_state), "error": "missing_state_dir"})
        elif not target_state.is_dir():
            errors.append("state_dir_not_directory")
            path_errors.append({"path": str(target_state), "error": "state_dir_not_directory"})
        native_output_root = Path(workspace_root).resolve(strict=False).parent
        for path in required_unchanged_paths:
            target = Path(path)
            if not target.exists():
                errors.append("missing_required_unchanged_path_target")
                path_errors.append({"path": str(target), "error": "missing_required_unchanged_path_target"})
                continue
            if _paths_overlap(target.resolve(strict=False), native_output_root):
                errors.append("required_unchanged_path_overlaps_native_output")
                path_errors.append({"path": str(target), "error": "required_unchanged_path_overlaps_native_output"})
    unique_errors = list(dict.fromkeys(errors))
    return {
        "ok": not unique_errors,
        "errors": unique_errors,
        "path_errors": path_errors,
        "state_dir": str(Path(state_dir).resolve(strict=False)),
        "workspace_root": str(Path(workspace_root).resolve(strict=False)),
        "restart_command_present": bool(str(restart_command or "").strip()),
        "smoke_url": smoke_url,
        "smoke_query_present": bool(smoke_query),
        "smoke_mode": smoke_mode,
        "required_unchanged_paths": [str(Path(path).resolve(strict=False)) for path in required_unchanged_paths],
    }


def _validate_refresh_cutover_preconditions(
    *,
    state_dir: Path,
    workspace_root: Path,
    required_unchanged_paths: list[Path] | None,
    query_smoke: Any | None,
) -> None:
    if not required_unchanged_paths:
        raise ValueError("native refresh cutover requires at least one --require-unchanged-path guard")
    if query_smoke is None:
        raise ValueError("native refresh cutover requires explicit --smoke-url and --smoke-query guards")
    require_existing_cutover_state_dir(state_dir)
    require_existing_unchanged_paths(required_unchanged_paths)
    require_unchanged_paths_outside_native_outputs(required_unchanged_paths, workspace_root=workspace_root)


def _skipped_refresh_cutover_result(
    *,
    current_status: dict[str, Any],
    refresh_kind: str,
    refresh_policy: dict[str, Any],
    fill_missing_vectors: bool,
    unchanged_before: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result = {
        "cutover": True,
        "skipped": True,
        "cutover_executed": False,
        "build_executed": False,
        "restart_executed": False,
        "query_smoke_executed": False,
        "pending_clear_executed": False,
        "status": current_status,
        "refresh_kind": refresh_kind,
        "refresh_policy": refresh_policy,
        "vector_cache_required": True,
        "fill_missing_vectors": fill_missing_vectors,
    }
    if unchanged_before:
        result["unchanged_path_audit"] = assert_required_unchanged_paths(unchanged_before)
    return result


def state_input_fingerprints(state_dir: Path) -> dict[str, dict[str, Any]]:
    from ops import native_zvec_materialize

    return native_zvec_materialize.state_input_fingerprints(state_dir)


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def active_already_fresh_report(*, state_dir: Path, workspace_root: Path) -> dict[str, Any]:
    active = _read_json_object(active_workspace_path(state_dir))
    if not active:
        return {"fresh": False, "reason": "active_pointer_missing"}
    workspace_id = str(active.get("workspace_id") or "")
    if not workspace_id or active.get("status") != "active":
        return {"fresh": False, "reason": "active_pointer_not_active", "active": active}
    report_path = Path(workspace_root) / workspace_id / "build_report.json"
    report = _read_json_object(report_path)
    if not report or report.get("ok") is not True:
        return {"fresh": False, "reason": "active_build_report_missing_or_not_ok", "active": active, "build_report": str(report_path)}
    if str(report.get("workspace_id") or "") != workspace_id:
        return {"fresh": False, "reason": "workspace_id_mismatch", "active": active, "build_report": str(report_path)}
    current_fingerprints = state_input_fingerprints(state_dir)
    if report.get("input_fingerprints") != current_fingerprints:
        return {
            "fresh": False,
            "reason": "state_input_fingerprints_mismatch",
            "active": active,
            "build_report": str(report_path),
        }
    raw_native_report = report.get("native_report")
    native_report = raw_native_report if isinstance(raw_native_report, dict) else {}
    active_hash = active.get("source_manifest_hash")
    report_hash = native_report.get("source_manifest_hash") or report.get("source_manifest_hash")
    if active_hash and report_hash and active_hash != report_hash:
        return {"fresh": False, "reason": "source_manifest_hash_mismatch", "active": active, "build_report": str(report_path)}
    if report.get("counts") and active.get("counts") and report.get("counts") != active.get("counts"):
        return {"fresh": False, "reason": "counts_mismatch", "active": active, "build_report": str(report_path)}
    return {
        "fresh": True,
        "reason": "active_build_report_input_fingerprints_match",
        "active": active,
        "build_report": str(report_path),
    }


def _execute_active_already_fresh_cutover(
    *,
    state_dir: Path,
    active: dict[str, Any],
    restart_service: Any,
    query_smoke: Any,
    unchanged_before: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    service = restart_service(state_dir=state_dir)
    smoke = query_smoke(state_dir=state_dir, active=active)
    if not isinstance(smoke, dict) or smoke.get("ok") is not True:
        raise RuntimeError(f"native refresh active-fresh query smoke did not report ok=true: {smoke!r}")
    unchanged_path_audit = assert_required_unchanged_paths(unchanged_before) if unchanged_before else None
    return {"active": active, "service": service, "query_smoke": smoke, "unchanged_path_audit": unchanged_path_audit}


def _active_already_fresh_success_result(
    *,
    root: Path,
    state_dir: Path,
    current_status: dict[str, Any],
    refresh_kind: str,
    refresh_policy: dict[str, Any],
    fill_missing_vectors: bool,
    freshness: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, Any]:
    pending_cleared = clear_pending(state_dir)
    status_after = status(root, state_dir)
    status_after_policy = status_after.get("refresh_policy") or native_refresh_policy_status(state_dir)
    result = {
        "cutover": True,
        "skipped": False,
        "active_already_fresh": True,
        "cutover_executed": True,
        "build_executed": False,
        "restart_executed": True,
        "query_smoke_executed": True,
        "pending_clear_executed": True,
        "active": execution["active"],
        "service": execution["service"],
        "pending_cleared": pending_cleared,
        "status_before": current_status,
        "status_after": status_after,
        "refresh_kind": refresh_kind,
        "refresh_policy": refresh_policy,
        "next_refresh_kind_after_success": status_after_policy["next_refresh_kind"],
        "policy_native_pending": {"marked": False, "reason": "active_already_fresh"},
        "policy_native_pending_marked_count": 0,
        "vector_cache_required": True,
        "fill_missing_vectors": fill_missing_vectors,
        "query_smoke": execution["query_smoke"],
        "active_freshness": freshness,
    }
    if execution.get("unchanged_path_audit") is not None:
        result["unchanged_path_audit"] = execution["unchanged_path_audit"]
    return result


def _execute_refresh_cutover(
    *,
    root: Path,
    state_dir: Path,
    workspace_root: Path,
    workspace_id: str,
    embedding_profile: str,
    refresh_kind: str,
    build_workspace: Any,
    finalize_workspace: Any,
    restart_service: Any,
    query_smoke: Any,
    fill_missing_vectors: bool,
    unchanged_before: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    report = build_workspace(
        root=root,
        state_dir=state_dir,
        workspace_root=workspace_root,
        workspace_id=workspace_id,
        embedding_profile=embedding_profile,
        fill_missing_vectors=fill_missing_vectors,
    )
    active = finalize_workspace(state_dir=state_dir, reason=native_refresh_reason(refresh_kind, "cutover"))
    service = restart_service(state_dir=state_dir)
    smoke = query_smoke(state_dir=state_dir, active=active)
    if not isinstance(smoke, dict) or smoke.get("ok") is not True:
        raise RuntimeError(f"native refresh cutover query smoke did not report ok=true: {smoke!r}")
    unchanged_path_audit = assert_required_unchanged_paths(unchanged_before) if unchanged_before else None
    return {
        "build": report,
        "active": active,
        "service": service,
        "query_smoke": smoke,
        "unchanged_path_audit": unchanged_path_audit,
    }


def _refresh_cutover_success_result(
    *,
    root: Path,
    state_dir: Path,
    current_status: dict[str, Any],
    refresh_kind: str,
    refresh_policy: dict[str, Any],
    fill_missing_vectors: bool,
    execution: dict[str, Any],
) -> dict[str, Any]:
    pending_cleared = clear_pending(state_dir)
    status_after = status(root, state_dir)
    status_after_policy = status_after.get("refresh_policy") or native_refresh_policy_status(state_dir)
    policy_native_pending = mark_full_rebuild_pending_if_due(
        state_dir,
        root,
        refresh_kind=refresh_kind,
        status_after_policy=status_after_policy,
    )
    if policy_native_pending.get("marked"):
        status_after = status(root, state_dir)
        status_after_policy = status_after.get("refresh_policy") or native_refresh_policy_status(state_dir)
    result = {
        "cutover": True,
        "skipped": False,
        "cutover_executed": True,
        "build_executed": True,
        "restart_executed": True,
        "query_smoke_executed": True,
        "pending_clear_executed": True,
        "build": execution["build"],
        "active": execution["active"],
        "service": execution["service"],
        "pending_cleared": pending_cleared,
        "status_before": current_status,
        "status_after": status_after,
        "refresh_kind": refresh_kind,
        "refresh_policy": refresh_policy,
        "next_refresh_kind_after_success": status_after_policy["next_refresh_kind"],
        "policy_native_pending": policy_native_pending,
        "policy_native_pending_marked_count": 1 if policy_native_pending.get("marked") else 0,
        "vector_cache_required": True,
        "fill_missing_vectors": fill_missing_vectors,
        "query_smoke": execution["query_smoke"],
    }
    if execution.get("unchanged_path_audit") is not None:
        result["unchanged_path_audit"] = execution["unchanged_path_audit"]
    return result


def refresh_cutover(
    *,
    root: Path,
    state_dir: Path,
    workspace_root: Path,
    workspace_id: str,
    embedding_profile: str,
    build_workspace: Any = build_prepared_workspace,
    finalize_workspace: Any = finalize_prepared_workspace_for_state,
    restart_service: Any | None = None,
    query_smoke: Any | None = None,
    fill_missing_vectors: bool = True,
    force: bool = False,
    required_unchanged_paths: list[Path] | None = None,
) -> dict[str, Any]:
    _validate_refresh_cutover_preconditions(
        state_dir=state_dir,
        workspace_root=workspace_root,
        required_unchanged_paths=required_unchanged_paths,
        query_smoke=query_smoke,
    )
    current_status = status(root, state_dir)
    refresh_policy = current_status.get("refresh_policy") or native_refresh_policy_status(state_dir)
    refresh_kind = refresh_policy["next_refresh_kind"]
    unchanged_before = snapshot_required_unchanged_paths(required_unchanged_paths)
    if not current_status["should_refresh"] and not force:
        return _skipped_refresh_cutover_result(
            current_status=current_status,
            refresh_kind=refresh_kind,
            refresh_policy=refresh_policy,
            fill_missing_vectors=fill_missing_vectors,
            unchanged_before=unchanged_before,
        )
    if restart_service is None:
        raise ValueError("native refresh cutover requires an explicit restart_service hook")
    if not force and refresh_kind == REFRESH_KIND_INCREMENTAL and not status_has_wiki_integration_pending(current_status):
        freshness = active_already_fresh_report(state_dir=state_dir, workspace_root=workspace_root)
        if freshness.get("fresh") is True:
            execution = _execute_active_already_fresh_cutover(
                state_dir=state_dir,
                active=freshness["active"],
                restart_service=restart_service,
                query_smoke=query_smoke,
                unchanged_before=unchanged_before,
            )
            return _active_already_fresh_success_result(
                root=root,
                state_dir=state_dir,
                current_status=current_status,
                refresh_kind=refresh_kind,
                refresh_policy=refresh_policy,
                fill_missing_vectors=fill_missing_vectors,
                freshness=freshness,
                execution=execution,
            )

    execution = _execute_refresh_cutover(
        root=root,
        state_dir=state_dir,
        workspace_root=workspace_root,
        workspace_id=workspace_id,
        embedding_profile=embedding_profile,
        refresh_kind=refresh_kind,
        build_workspace=build_workspace,
        finalize_workspace=finalize_workspace,
        restart_service=restart_service,
        query_smoke=query_smoke,
        fill_missing_vectors=fill_missing_vectors,
        unchanged_before=unchanged_before,
    )
    return _refresh_cutover_success_result(
        root=root,
        state_dir=state_dir,
        current_status=current_status,
        refresh_kind=refresh_kind,
        refresh_policy=refresh_policy,
        fill_missing_vectors=fill_missing_vectors,
        execution=execution,
    )


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage pending llm-wiki native zvec refreshes")
    sub = parser.add_subparsers(dest="command", required=True)

    def common_paths(target: argparse.ArgumentParser) -> None:
        target.add_argument("--root", type=Path, default=DEFAULT_WIKI_ROOT)
        target.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
        target.add_argument("--state-dir", type=Path)

    status_parser = sub.add_parser("status", help="Show native refresh pending state")
    common_paths(status_parser)

    mark_parser = sub.add_parser("mark-pending", help="Mark a native refresh as pending")
    common_paths(mark_parser)
    mark_parser.add_argument("--reason", default="manual")

    refresh_parser = sub.add_parser("refresh", help="Run a native refresh")
    common_paths(refresh_parser)
    refresh_parser.add_argument("--prepare-only", action="store_true")
    refresh_parser.add_argument("--cutover", action="store_true")
    refresh_parser.add_argument("--force", action="store_true")
    refresh_parser.add_argument("--workspace-id", default=None)
    refresh_parser.add_argument("--workspace-root", type=Path)
    refresh_parser.add_argument("--embedding-profile", default="conservative")
    refresh_parser.add_argument("--fill-missing-vectors", dest="fill_missing_vectors", action="store_true", default=True, help="Use vector_cache.sqlite and fill missing manifest vectors before native graph update/rebuild (default)")
    refresh_parser.add_argument("--no-fill-missing-vectors", dest="fill_missing_vectors", action="store_false", help="Disable vector-cache fill only for controlled tests or emergency debugging")
    refresh_parser.add_argument("--restart-command", default=None)
    refresh_parser.add_argument("--health-url", default=None)
    refresh_parser.add_argument("--health-timeout", type=float, default=10.0)
    refresh_parser.add_argument("--smoke-url", default=None)
    refresh_parser.add_argument("--smoke-query", default=None)
    refresh_parser.add_argument("--smoke-mode", default="mix")
    refresh_parser.add_argument("--smoke-workspace-id", default=None)
    refresh_parser.add_argument("--smoke-query-vector-json", default=None)
    refresh_parser.add_argument("--smoke-query-vector-file", type=Path, default=None)
    refresh_parser.add_argument("--smoke-query-vector-source", choices=["active-first-vector"], default=None)
    refresh_parser.add_argument("--smoke-timeout", type=float, default=10.0)
    refresh_parser.add_argument("--require-unchanged-path", dest="required_unchanged_paths", action="append", type=Path, default=[])

    preflight_parser = sub.add_parser("preflight-cutover", help="Validate native refresh cutover guards without build/status writes")
    common_paths(preflight_parser)
    preflight_parser.add_argument("--workspace-root", type=Path)
    preflight_parser.add_argument("--restart-command", default=None)
    preflight_parser.add_argument("--smoke-url", default=None)
    preflight_parser.add_argument("--smoke-query", default=None)
    preflight_parser.add_argument("--smoke-mode", default="mix")
    preflight_parser.add_argument("--require-unchanged-path", dest="required_unchanged_paths", action="append", type=Path, default=[])
    preflight_parser.add_argument("--skip-path-checks", action="store_true", help="Validate option shape only; skip state/path existence checks")

    args = parser.parse_args(argv)
    root = args.root.resolve()
    workdir = args.workdir.resolve()
    state_dir = (args.state_dir or workdir / "state").resolve()
    if args.command == "status":
        print_json(status(root, state_dir))
        return 0
    if args.command == "mark-pending":
        entry = mark_pending(state_dir, root, reason=args.reason)
        print_json({"marked": entry, **status(root, state_dir)})
        return 0
    if args.command == "preflight-cutover":
        workspace_root = (args.workspace_root or default_workspace_root(state_dir)).resolve()
        report = cutover_guard_report(
            state_dir=state_dir,
            workspace_root=workspace_root,
            restart_command=args.restart_command,
            smoke_url=args.smoke_url,
            smoke_query=args.smoke_query,
            smoke_mode=args.smoke_mode,
            required_unchanged_paths=args.required_unchanged_paths,
            check_paths=not args.skip_path_checks,
        )
        print_json(report)
        return 0 if report["ok"] else 1
    if args.command == "refresh":
        if args.prepare_only and args.cutover:
            raise ValueError("native refresh accepts either --prepare-only or --cutover, not both")
        workspace_root = (args.workspace_root or default_workspace_root(state_dir)).resolve()
        workspace_id = args.workspace_id or f"native-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        if not args.prepare_only:
            if not args.cutover:
                raise ValueError("native refresh production mode requires --prepare-only or explicit --cutover")
            if not args.restart_command:
                raise ValueError("native refresh cutover requires explicit --restart-command")
            query_smoke = query_smoke_from_args(args)
            restart_service = restart_service_from_args(args) if args.restart_command else None
            print_json(
                refresh_cutover(
                    root=root,
                    state_dir=state_dir,
                    workspace_root=workspace_root,
                    workspace_id=workspace_id,
                    embedding_profile=args.embedding_profile,
                    build_workspace=build_prepared_workspace,
                    finalize_workspace=finalize_prepared_workspace_for_state,
                    restart_service=restart_service,
                    query_smoke=query_smoke,
                    fill_missing_vectors=args.fill_missing_vectors,
                    force=args.force,
                    required_unchanged_paths=args.required_unchanged_paths,
                )
            )
            return 0
        print_json(
            refresh_prepare_only(
                root=root,
                state_dir=state_dir,
                workspace_root=workspace_root,
                workspace_id=workspace_id,
                embedding_profile=args.embedding_profile,
                fill_missing_vectors=args.fill_missing_vectors,
                force=args.force,
                required_unchanged_paths=args.required_unchanged_paths,
            )
        )
        return 0
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
