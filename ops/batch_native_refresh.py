#!/usr/bin/env python3
"""Manage pending llm-wiki native zvec refreshes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
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

PENDING_NATIVE_REFRESH_LEDGER = "pending_native_refresh.json"
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


def status(root: Path, state_dir: Path) -> dict[str, Any]:
    entries = pending_entries(state_dir)
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
    }


def build_prepared_workspace(
    *,
    root: Path,
    state_dir: Path,
    workspace_root: Path,
    workspace_id: str,
    embedding_profile: str,
    fill_missing_vectors: bool = False,
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
    fill_missing_vectors: bool = False,
    force: bool = False,
    required_unchanged_paths: list[Path] | None = None,
) -> dict[str, Any]:
    before_active = active_workspace_path(state_dir).read_text(encoding="utf-8") if active_workspace_path(state_dir).exists() else None
    unchanged_before = snapshot_required_unchanged_paths(required_unchanged_paths)
    current_status = status(root, state_dir)
    if not current_status["should_refresh"] and not force:
        result = {"prepared_only": True, "skipped": True, "status": current_status}
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
    mode: str = "mix",
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
    if not isinstance(trace, dict) or trace.get("retrieval_backend") != "zvec":
        raise RuntimeError(f"query smoke did not prove zvec retrieval: url={url} status={status_code}")
    vector_hit_count = trace.get("vector_hit_count")
    if type(vector_hit_count) is not int or vector_hit_count <= 0:
        raise RuntimeError(f"query smoke did not return positive zvec hits: url={url} status={status_code}")
    return result


def normalize_query_vector(value: Any) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError("query_vector must be a non-empty JSON list")
    vector: list[float] = []
    for item in value:
        if type(item) not in (int, float):
            raise ValueError("query_vector must contain only finite numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError("query_vector must contain only finite numbers")
        vector.append(number)
    return vector


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
    fill_missing_vectors: bool = False,
    force: bool = False,
    required_unchanged_paths: list[Path] | None = None,
) -> dict[str, Any]:
    if not required_unchanged_paths:
        raise ValueError("native refresh cutover requires at least one --require-unchanged-path guard")
    if query_smoke is None:
        raise ValueError("native refresh cutover requires explicit --smoke-url and --smoke-query guards")
    require_existing_cutover_state_dir(state_dir)
    require_existing_unchanged_paths(required_unchanged_paths)
    require_unchanged_paths_outside_native_outputs(required_unchanged_paths, workspace_root=workspace_root)
    current_status = status(root, state_dir)
    if not current_status["should_refresh"] and not force:
        result = {
            "cutover": True,
            "skipped": True,
            "cutover_executed": False,
            "build_executed": False,
            "restart_executed": False,
            "query_smoke_executed": False,
            "pending_clear_executed": False,
            "status": current_status,
        }
        unchanged_before = snapshot_required_unchanged_paths(required_unchanged_paths)
        if unchanged_before:
            result["unchanged_path_audit"] = assert_required_unchanged_paths(unchanged_before)
        return result
    if restart_service is None:
        raise ValueError("native refresh cutover requires an explicit restart_service hook")

    unchanged_before = snapshot_required_unchanged_paths(required_unchanged_paths)
    report = build_workspace(
        root=root,
        state_dir=state_dir,
        workspace_root=workspace_root,
        workspace_id=workspace_id,
        embedding_profile=embedding_profile,
        fill_missing_vectors=fill_missing_vectors,
    )
    active = finalize_workspace(state_dir=state_dir, reason="native refresh cutover")
    service = restart_service(state_dir=state_dir)
    smoke = query_smoke(state_dir=state_dir, active=active) if query_smoke is not None else None
    if not isinstance(smoke, dict) or smoke.get("ok") is not True:
        raise RuntimeError(f"native refresh cutover query smoke did not report ok=true: {smoke!r}")
    unchanged_path_audit = assert_required_unchanged_paths(unchanged_before) if unchanged_before else None
    pending_cleared = clear_pending(state_dir)
    result = {
        "cutover": True,
        "skipped": False,
        "cutover_executed": True,
        "build_executed": True,
        "restart_executed": True,
        "query_smoke_executed": smoke is not None,
        "pending_clear_executed": True,
        "build": report,
        "active": active,
        "service": service,
        "pending_cleared": pending_cleared,
        "status_before": current_status,
    }
    if smoke is not None:
        result["query_smoke"] = smoke
    if unchanged_path_audit is not None:
        result["unchanged_path_audit"] = unchanged_path_audit
    return result


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
    refresh_parser.add_argument("--fill-missing-vectors", action="store_true")
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
