#!/usr/bin/env python3
"""Latency-aware pending ledger and batch LightRAG refresh runner for llm-wiki.

The default workflow is deliberately conservative: mark Markdown/wiki changes as pending, refresh only when the pending threshold is reached or before LightRAG-heavy queries, and keep all state under the external LightRAG workdir/state tree rather than the human wiki root.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from wiki_lightrag_lib import (
    DEFAULT_PENDING_LIGHTRAG_REFRESH_THRESHOLD,
    DEFAULT_STATE_DIR,
    DEFAULT_WIKI_ROOT,
    DEFAULT_WORKDIR,
    clear_lightrag_refresh_pending_after_success,
    ensure_state_dirs,
    mark_lightrag_refresh_pending,
    now_stamp,
    pending_lightrag_refresh_status,
    print_json,
    record_lightrag_refresh_failure,
    release_process_memory,
)
from custom_kg_incremental import DEFAULT_FULL_REBUILD_INTERVAL, plan_incremental_import, prepared_swap_report_path

LIGHTRAG_PYTHON = Path("/home/xu/.local/share/uv/tools/lightrag-hku/bin/python")
SERVICE_NAME = "lightrag-server.service"
MATERIALIZABLE_FULL_REBUILD_REASONS = {"incremental_interval_reached"}


def script_path(workdir: Path, name: str) -> str:
    return str(workdir / "scripts" / name)


def build_refresh_command_groups(root: Path, state_dir: Path, workdir: Path) -> dict[str, list[list[str]]]:
    py = sys.executable
    artifact = [
        [py, script_path(workdir, "validate_wiki.py"), "--root", str(root), "--state-dir", str(state_dir), "--workdir", str(workdir), "--full"],
        [py, script_path(workdir, "audit_raw_note_sections.py"), "--root", str(root), "--state-dir", str(state_dir), "--workdir", str(workdir), "--write-report", "--limit-issues", "200"],
        [py, script_path(workdir, "build_seed_edges.py"), "--root", str(root), "--state-dir", str(state_dir), "--workdir", str(workdir)],
        [py, script_path(workdir, "extract_method_atoms.py"), "--root", str(root), "--state-dir", str(state_dir), "--workdir", str(workdir)],
        [py, script_path(workdir, "extract_raw_sections.py"), "--root", str(root), "--state-dir", str(state_dir), "--workdir", str(workdir)],
        [
            py,
            script_path(workdir, "build_section_similarity_graph.py"),
            "--root",
            str(root),
            "--state-dir",
            str(state_dir),
            "--workdir",
            str(workdir),
            "--same-kind-k",
            "5",
            "--cross-kind-k",
            "3",
            "--same-kind-min-cosine",
            "0.72",
            "--cross-kind-min-cosine",
            "0.76",
        ],
        [py, script_path(workdir, "select_section_similarity_edges.py"), "--root", str(root), "--state-dir", str(state_dir), "--workdir", str(workdir)],
    ]
    full_import = [
        ["systemctl", "--user", "stop", SERVICE_NAME],
        ["python-internal", "reset-rag-storage", str(workdir / "rag_storage"), str(workdir / "inputs")],
        [
            str(LIGHTRAG_PYTHON),
            script_path(workdir, "import_custom_kg.py"),
            "--root",
            str(root),
            "--state-dir",
            str(state_dir),
            "--workdir",
            str(workdir),
        ],
        ["systemctl", "--user", "start", SERVICE_NAME],
        ["python-internal", "health", "http://127.0.0.1:9621/health"],
    ]
    return {"artifact": artifact, "full_import": full_import}


def build_refresh_commands(root: Path, state_dir: Path, workdir: Path) -> list[list[str]]:
    groups = build_refresh_command_groups(root, state_dir, workdir)
    return groups["artifact"] + groups["full_import"]


def build_incremental_import_commands(root: Path, state_dir: Path, workdir: Path) -> list[list[str]]:
    prepared_report = prepared_swap_report_path(state_dir)
    return [
        [
            str(LIGHTRAG_PYTHON),
            script_path(workdir, "custom_kg_incremental.py"),
            "apply",
            "--root",
            str(root),
            "--state-dir",
            str(state_dir),
            "--workdir",
            str(workdir),
            "--full-rebuild-interval",
            str(DEFAULT_FULL_REBUILD_INTERVAL),
            "--prepare-swap",
        ],
        ["systemctl", "--user", "stop", SERVICE_NAME],
        [
            str(LIGHTRAG_PYTHON),
            script_path(workdir, "custom_kg_incremental.py"),
            "finalize-prepared-swap",
            "--root",
            str(root),
            "--state-dir",
            str(state_dir),
            "--workdir",
            str(workdir),
            "--prepared-report",
            str(prepared_report),
        ],
        ["systemctl", "--user", "start", SERVICE_NAME],
        ["python-internal", "health", "http://127.0.0.1:9621/health"],
    ]


def build_full_materialization_import_commands(root: Path, state_dir: Path, workdir: Path) -> list[list[str]]:
    prepared_report = prepared_swap_report_path(state_dir)
    return [
        [
            str(LIGHTRAG_PYTHON),
            script_path(workdir, "custom_kg_incremental.py"),
            "materialize-full",
            "--root",
            str(root),
            "--state-dir",
            str(state_dir),
            "--workdir",
            str(workdir),
            "--vector-cache",
            str(state_dir / "vector_cache.sqlite"),
            "--seed-from-storage",
            "--seed-storage-dir",
            str(workdir / "rag_storage"),
            "--no-swap",
            "--prepare-swap",
        ],
        ["systemctl", "--user", "stop", SERVICE_NAME],
        [
            str(LIGHTRAG_PYTHON),
            script_path(workdir, "custom_kg_incremental.py"),
            "finalize-prepared-swap",
            "--root",
            str(root),
            "--state-dir",
            str(state_dir),
            "--workdir",
            str(workdir),
            "--prepared-report",
            str(prepared_report),
        ],
        ["systemctl", "--user", "start", SERVICE_NAME],
        ["python-internal", "health", "http://127.0.0.1:9621/health"],
    ]


def should_use_full_materialization(import_mode: dict[str, Any]) -> bool:
    if import_mode.get("selected_mode") != "full_rebuild":
        return False
    reasons = {str(reason) for reason in (import_mode.get("reasons") or [])}
    return bool(reasons) and reasons <= MATERIALIZABLE_FULL_REBUILD_REASONS


def plan_incremental_import_mode(root: Path, state_dir: Path, workdir: Path) -> dict[str, Any]:
    return plan_incremental_import(root, state_dir, workdir, full_rebuild_interval=DEFAULT_FULL_REBUILD_INTERVAL)


def _diff_collection_is_empty(collection: Any) -> bool:
    if not isinstance(collection, dict):
        return False
    for key in ("add", "update", "delete"):
        try:
            if int(collection.get(key, 0) or 0) != 0:
                return False
        except (TypeError, ValueError):
            return False
        if collection.get(f"{key}_ids"):
            return False
    return True


def incremental_diff_is_empty(diff: Any) -> bool:
    if not isinstance(diff, dict):
        return False
    return all(_diff_collection_is_empty(diff.get(name)) for name in ("chunks", "entities", "relationships"))


def should_skip_incremental_import(import_mode: dict[str, Any]) -> bool:
    return import_mode.get("selected_mode") == "incremental" and incremental_diff_is_empty(import_mode.get("diff"))


def select_import_commands(root: Path, state_dir: Path, workdir: Path, full_import_commands: list[list[str]], import_mode: dict[str, Any]) -> list[list[str]]:
    if should_skip_incremental_import(import_mode):
        return []
    if import_mode.get("selected_mode") == "incremental":
        return build_incremental_import_commands(root, state_dir, workdir)
    if should_use_full_materialization(import_mode):
        return build_full_materialization_import_commands(root, state_dir, workdir)
    return full_import_commands


def append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(message)
        if not message.endswith("\n"):
            f.write("\n")


def run_subprocess(command: list[str], log_path: Path, env: dict[str, str] | None = None, timeout: int | None = None) -> None:
    append_log(log_path, "\n$ " + " ".join(command) + "\n")
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    append_log(log_path, completed.stdout or "")
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit {completed.returncode}: {' '.join(command)}")


def reset_rag_storage(workdir: Path, log_path: Path) -> None:
    append_log(log_path, f"\n[reset-rag-storage] {workdir / 'rag_storage'} {workdir / 'inputs'}\n")
    for rel in ["rag_storage", "inputs"]:
        path = workdir / rel
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def wait_health(url: str, log_path: Path, attempts: int = 12, delay_s: float = 5.0) -> dict[str, Any]:
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
            append_log(log_path, f"[health:{attempt}] {json.dumps(payload, ensure_ascii=False)}\n")
            if payload.get("status") == "healthy" and payload.get("pipeline_busy") is False:
                return payload
            last_error = f"unhealthy payload: {payload}"
        except Exception as exc:  # pragma: no cover - external service path
            last_error = repr(exc)
            append_log(log_path, f"[health:{attempt}] {last_error}\n")
        time.sleep(delay_s)
    raise RuntimeError(f"LightRAG health check did not become healthy: {last_error}")


def run_real_refresh(root: Path, state_dir: Path, workdir: Path, reason: str, artifact_log: Path, import_log: Path) -> dict[str, Any]:
    ensure_state_dirs(state_dir)
    refresh_groups = build_refresh_command_groups(root, state_dir, workdir)
    artifact_commands = refresh_groups["artifact"]
    full_import_commands = refresh_groups["full_import"]
    append_log(artifact_log, f"[batch-refresh] start {now_stamp()} reason={reason}\n")
    for command in artifact_commands:
        run_subprocess(command, artifact_log, timeout=600)
    append_log(artifact_log, f"[batch-refresh] artifacts complete {now_stamp()}\n")

    import_mode = plan_incremental_import_mode(root, state_dir, workdir)
    release_process_memory()
    import_commands = select_import_commands(root, state_dir, workdir, full_import_commands, import_mode)
    append_log(import_log, f"[batch-refresh] import start {now_stamp()} reason={reason} mode={import_mode.get('selected_mode')} reasons={import_mode.get('reasons')}\n")
    if should_skip_incremental_import(import_mode):
        append_log(import_log, f"[batch-refresh] import skipped {now_stamp()} reason=incremental_empty_diff\n")
        clear = clear_lightrag_refresh_pending_after_success(root, state_dir, reason=reason)
        return {"artifact_log": str(artifact_log), "import_log": str(import_log), "health": None, "clear": clear, "import_mode": import_mode, "import_skipped": True, "import_skip_reason": "incremental_empty_diff"}
    def _run_import_commands(commands: list[list[str]]) -> tuple[dict[str, Any] | None, bool, bool]:
        service_stopped = False
        service_started = False
        health: dict[str, Any] | None = None
        try:
            for command in commands:
                if command[:2] == ["python-internal", "reset-rag-storage"]:
                    reset_rag_storage(workdir, import_log)
                elif command[:2] == ["python-internal", "health"]:
                    health = wait_health(command[2], import_log)
                else:
                    env = os.environ.copy()
                    if command and command[0] == str(LIGHTRAG_PYTHON):
                        env.update({"EMBEDDING_FUNC_MAX_ASYNC": "1", "EMBEDDING_BATCH_NUM": "10", "MAX_PARALLEL_INSERT": "1"})
                    run_subprocess(command, import_log, env=env, timeout=None)
                    if command[:3] == ["systemctl", "--user", "stop"] and command[-1] == SERVICE_NAME:
                        service_stopped = True
                    if command[:3] == ["systemctl", "--user", "start"] and command[-1] == SERVICE_NAME:
                        service_started = True
        except Exception as exc:
            setattr(exc, "_lightrag_service_stopped", service_stopped)
            if service_stopped and not service_started:
                start_command = ["systemctl", "--user", "start", SERVICE_NAME]
                append_log(import_log, f"[batch-refresh] recovery start after failure: {type(exc).__name__}: {exc}\n")
                try:
                    run_subprocess(start_command, import_log, timeout=180)
                    service_started = True
                except Exception as recovery_exc:  # pragma: no cover - defensive recovery path
                    append_log(import_log, f"[batch-refresh] recovery start failed: {type(recovery_exc).__name__}: {recovery_exc}\n")
                    raise RuntimeError(f"{exc}; additionally failed to restart {SERVICE_NAME}: {recovery_exc}") from exc
            raise
        return health, service_stopped, service_started

    try:
        health, _service_stopped, _service_started = _run_import_commands(import_commands)
    except Exception as exc:
        if should_use_full_materialization(import_mode) and not bool(getattr(exc, "_lightrag_service_stopped", False)):
            append_log(import_log, f"[batch-refresh] full materialization failed before service stop; falling back to cold full import: {type(exc).__name__}: {exc}\n")
            fallback_mode = dict(import_mode)
            fallback_mode["full_materialization_fallback"] = "cold_full_import"
            fallback_mode["fallback_reason"] = f"{type(exc).__name__}: {exc}"
            import_mode = fallback_mode
            health, _service_stopped, _service_started = _run_import_commands(full_import_commands)
        else:
            raise
    append_log(import_log, f"[batch-refresh] import done {now_stamp()} mode={import_mode.get('selected_mode')}\n")
    clear = clear_lightrag_refresh_pending_after_success(root, state_dir, reason=reason)
    return {"artifact_log": str(artifact_log), "import_log": str(import_log), "health": health, "clear": clear, "import_mode": import_mode}


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=DEFAULT_WIKI_ROOT)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage pending llm-wiki LightRAG batch refreshes")
    sub = parser.add_subparsers(dest="command", required=True)

    status_parser = sub.add_parser("status", help="Show pending-refresh state and freshness decision")
    add_common_paths(status_parser)
    status_parser.add_argument("--reason", default="threshold", choices=["threshold", "pre-query", "manual", "full-graph-fresh"])
    status_parser.add_argument("--threshold", type=int, default=None, help="Override ledger threshold for this status check")

    mark_parser = sub.add_parser("mark-pending", help="Mark one raw/wiki change as pending for the next batch refresh")
    add_common_paths(mark_parser)
    mark_parser.add_argument("--raw-path", default="")
    mark_parser.add_argument("--title", default="")
    mark_parser.add_argument("--event-type", default="new_raw_note")
    mark_parser.add_argument("--changed-surface", action="append", default=[])
    mark_parser.add_argument("--expected-section", action="append", default=[])
    mark_parser.add_argument("--threshold", type=int, default=None, help=f"Set/override graph-refresh batch threshold for this ledger (default {DEFAULT_PENDING_LIGHTRAG_REFRESH_THRESHOLD} for new ledgers)")

    should_parser = sub.add_parser("should-refresh", help="Return whether a refresh should run for this reason")
    add_common_paths(should_parser)
    should_parser.add_argument("--reason", default="threshold", choices=["threshold", "pre-query", "manual", "full-graph-fresh"])
    should_parser.add_argument("--threshold", type=int, default=None, help="Override ledger threshold for this decision")
    should_parser.add_argument("--exit-code", action="store_true", help="Exit 10 when refresh is needed, 11 when upstream wiki integration/manual review is needed, 0 when no action is needed")

    refresh_parser = sub.add_parser("refresh", help="Run or dry-run a threshold/pre-query/manual full custom_kg refresh")
    add_common_paths(refresh_parser)
    refresh_parser.add_argument("--reason", default="threshold", choices=["threshold", "pre-query", "manual", "full-graph-fresh"])
    refresh_parser.add_argument("--threshold", type=int, default=None, help="Override ledger threshold for this run/dry-run decision")
    refresh_parser.add_argument("--dry-run", action="store_true")
    refresh_parser.add_argument("--force", action="store_true")

    clear_parser = sub.add_parser("clear-success", help="Clear pending ledger after an externally completed successful refresh")
    add_common_paths(clear_parser)
    clear_parser.add_argument("--reason", default="external-success")
    clear_parser.add_argument("--import-report", type=Path, default=None)

    args = parser.parse_args()
    root = args.root.resolve()
    state_dir = args.state_dir.resolve()
    workdir = args.workdir.resolve()

    if args.command == "status":
        print_json(pending_lightrag_refresh_status(root, state_dir, reason=args.reason, threshold=args.threshold))
        return 0
    if args.command == "mark-pending":
        entry = mark_lightrag_refresh_pending(
            state_dir,
            root,
            raw_path=args.raw_path,
            title=args.title,
            event_type=args.event_type,
            changed_surfaces=args.changed_surface or None,
            expected_sections=args.expected_section or None,
            threshold=args.threshold,
        )
        status = pending_lightrag_refresh_status(root, state_dir, reason="threshold", threshold=args.threshold)
        print_json({"marked": entry, **status})
        return 0
    if args.command == "should-refresh":
        status = pending_lightrag_refresh_status(root, state_dir, reason=args.reason, threshold=args.threshold)
        print_json(status)
        if args.exit_code:
            if status.get("blocked_by_pending_wiki_integration"):
                return 11
            if status["should_refresh"]:
                return 10
        return 0
    if args.command == "clear-success":
        print_json(clear_lightrag_refresh_pending_after_success(root, state_dir, import_report_path=args.import_report, reason=args.reason))
        return 0
    if args.command == "refresh":
        status = pending_lightrag_refresh_status(root, state_dir, reason=args.reason, threshold=args.threshold)
        force_blocked = bool(args.force and status.get("blocked_by_pending_wiki_integration"))
        should_run = bool((status["should_refresh"] or args.force) and not status.get("blocked_by_pending_wiki_integration"))
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_dir = state_dir / "refresh_logs"
        artifact_log = log_dir / f"{stamp}_batch_lightrag_refresh_artifacts.log"
        import_log = log_dir / f"{stamp}_batch_lightrag_refresh_import.log"
        command_groups = build_refresh_command_groups(root, state_dir, workdir)
        commands = command_groups["artifact"] + command_groups["full_import"]
        import_mode = None
        if should_run:
            try:
                import_mode = plan_incremental_import_mode(root, state_dir, workdir)
                commands = command_groups["artifact"] + select_import_commands(root, state_dir, workdir, command_groups["full_import"], import_mode)
            except Exception as exc:
                import_mode = {"selected_mode": "full_rebuild", "reasons": ["incremental_plan_failed"], "error": type(exc).__name__, "message": str(exc)}
        if args.dry_run:
            print_json({"dry_run": True, "would_run": should_run, "force_blocked_by_pending_wiki_integration": force_blocked, "status": status, "import_mode": import_mode, "commands": commands, "artifact_log": str(artifact_log), "import_log": str(import_log)})
            return 0
        if not should_run:
            print_json({"dry_run": False, "would_run": False, "force_blocked_by_pending_wiki_integration": force_blocked, "skipped": True, "status": status})
            return 0
        try:
            result = run_real_refresh(root, state_dir, workdir, args.reason, artifact_log, import_log)
        except Exception as exc:
            failure = record_lightrag_refresh_failure(state_dir, reason=args.reason, log_path=str(import_log), message=str(exc))
            print_json({"error": type(exc).__name__, "message": str(exc), "failure": failure, "artifact_log": str(artifact_log), "import_log": str(import_log)})
            return 1
        print_json({"dry_run": False, "would_run": True, "status_before": status, **result})
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
