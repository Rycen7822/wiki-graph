#!/usr/bin/env python3
"""Retired wikigraph refresh wrapper.

Production refresh moved to the native zvec backend.  This module remains
importable only so historical tests/tools get an explicit fail-closed error
instead of silently planning old backend work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from wiki_native_lib import DEFAULT_STATE_DIR, DEFAULT_WIKI_ROOT, DEFAULT_WORKDIR, print_json
from wiki_wikigraph_refresh_pending import pending_wikigraph_refresh_status

RETIRED_WIKIGRAPH_ACTIVATION_CODE = "retired-wikigraph-activation-retired"
RETIRED_WIKIGRAPH_COLD_IMPORT_CODE = "retired-wikigraph-cold-import-retired"
RETIRED_WIKIGRAPH_FULL_MATERIALIZATION_CODE = "retired-wikigraph-full-materialization-retired"
RETIRED_WIKIGRAPH_REFRESH_CODE = "retired-wikigraph-refresh-retired"
REPLACEMENT_COMMAND = "batch_native_refresh.py refresh"
RETIRED_WIKIGRAPH_MESSAGE = (
    "batch_wikigraph_refresh.py is retired after native zvec production cutover; "
    "use batch_native_refresh.py for production refresh work"
)
RETIRED_WIKIGRAPH_ACTIVATION_MESSAGE = RETIRED_WIKIGRAPH_MESSAGE
RETIRED_WIKIGRAPH_COLD_IMPORT_MESSAGE = RETIRED_WIKIGRAPH_MESSAGE
RETIRED_WIKIGRAPH_FULL_MATERIALIZATION_MESSAGE = RETIRED_WIKIGRAPH_MESSAGE


class RetiredWikigraphActivationError(RuntimeError):
    pass


def _retired_error() -> RetiredWikigraphActivationError:
    return RetiredWikigraphActivationError(RETIRED_WIKIGRAPH_MESSAGE)


def _retired_payload(*, command: str, status: dict[str, Any] | None = None, dry_run: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "retired": True,
        "code": RETIRED_WIKIGRAPH_REFRESH_CODE,
        "message": RETIRED_WIKIGRAPH_MESSAGE,
        "replacement_command": REPLACEMENT_COMMAND,
        "command": command,
        "dry_run": dry_run,
        "would_run": False,
        "commands": [],
    }
    if status is not None:
        payload["retired_backend_status"] = status
    return payload


def script_path(workdir: Path, name: str) -> str:
    return str(workdir / "scripts" / name)


def retired_wikigraph_activation_command() -> list[str]:
    return ["python-internal", RETIRED_WIKIGRAPH_ACTIVATION_CODE, RETIRED_WIKIGRAPH_ACTIVATION_MESSAGE]


def retired_wikigraph_cold_import_command() -> list[str]:
    return ["python-internal", RETIRED_WIKIGRAPH_COLD_IMPORT_CODE, RETIRED_WIKIGRAPH_COLD_IMPORT_MESSAGE]


def retired_wikigraph_full_materialization_command() -> list[str]:
    return ["python-internal", RETIRED_WIKIGRAPH_FULL_MATERIALIZATION_CODE, RETIRED_WIKIGRAPH_FULL_MATERIALIZATION_MESSAGE]


def build_refresh_command_groups(root: Path, state_dir: Path, workdir: Path, reuse_validation_report: Path | None = None) -> dict[str, list[list[str]]]:
    return {"artifact": [], "full_import": []}


def build_refresh_commands(root: Path, state_dir: Path, workdir: Path, reuse_validation_report: Path | None = None) -> list[list[str]]:
    return []


def build_incremental_import_commands(root: Path, state_dir: Path, workdir: Path) -> list[list[str]]:
    raise _retired_error()


def build_full_materialization_import_commands(root: Path, state_dir: Path, workdir: Path, embedding_profile: str | None = None) -> list[list[str]]:
    raise _retired_error()


def forced_full_rebuild_reuses_vector_cache(import_mode: dict[str, Any]) -> bool:
    return False


def should_use_full_materialization(import_mode: dict[str, Any]) -> bool:
    return False


def threshold_defaults_to_forced_full_rebuild(reason: str) -> bool:
    return False


def apply_refresh_mode_policy(
    import_mode: dict[str, Any],
    *,
    reason: str,
    force_full_rebuild: bool = False,
    reuse_vector_cache: bool = False,
) -> dict[str, Any]:
    retired_mode = dict(import_mode)
    retired_mode.update(
        {
            "selected_mode": "retired",
            "retired": True,
            "reasons": ["native_zvec_production_cutover"],
            "force_full_rebuild": False,
            "reuse_vector_cache": False,
        }
    )
    return retired_mode


def plan_incremental_import_mode(root: Path, state_dir: Path, workdir: Path) -> dict[str, Any]:
    return {"selected_mode": "retired", "retired": True, "reasons": ["native_zvec_production_cutover"]}


def incremental_diff_is_empty(diff: Any) -> bool:
    return True


def should_skip_incremental_import(import_mode: dict[str, Any]) -> bool:
    return True


def select_import_commands(root: Path, state_dir: Path, workdir: Path, full_import_commands: list[list[str]], import_mode: dict[str, Any], embedding_profile: str | None = None) -> list[list[str]]:
    return []




def run_subprocess(command: list[str], log_path: Path, env: dict[str, str] | None = None, timeout: int | None = None) -> None:
    raise _retired_error()


def reset_retired_backend_storage(workdir: Path, log_path: Path) -> None:
    raise _retired_error()


def wait_health(url: str, log_path: Path, attempts: int = 1, delay_s: float = 0.0) -> dict[str, Any]:
    raise _retired_error()


def run_real_refresh(
    root: Path,
    state_dir: Path,
    workdir: Path,
    reason: str,
    artifact_log: Path,
    import_log: Path,
    *,
    force_full_rebuild: bool = False,
    reuse_vector_cache: bool = False,
    reuse_validation_report: Path | None = None,
    embedding_profile: str | None = None,
) -> dict[str, Any]:
    raise _retired_error()


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=DEFAULT_WIKI_ROOT)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)


def _readonly_status(root: Path, state_dir: Path, *, reason: str, threshold: int | None = None) -> dict[str, Any]:
    status = pending_wikigraph_refresh_status(root, state_dir, reason=reason, threshold=threshold)
    payload = _retired_payload(command="status", status=status, dry_run=True)
    payload["should_refresh"] = False
    payload["blocked_by_pending_wiki_integration"] = bool(status.get("blocked_by_pending_wiki_integration"))
    payload["next_required_action"] = "native_refresh"
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Retired llm-wiki wikigraph refresh wrapper")
    sub = parser.add_subparsers(dest="command", required=True)

    status_parser = sub.add_parser("status", help="Show read-only retired-wrapper status")
    add_common_paths(status_parser)
    status_parser.add_argument("--reason", default="threshold", choices=["threshold", "pre-query", "manual", "full-graph-fresh"])
    status_parser.add_argument("--threshold", type=int, default=None)

    mark_parser = sub.add_parser("mark-pending", help="Fail closed; use native refresh state instead")
    add_common_paths(mark_parser)
    mark_parser.add_argument("--raw-path", default="")
    mark_parser.add_argument("--title", default="")
    mark_parser.add_argument("--event-type", default="new_raw_note")
    mark_parser.add_argument("--changed-surface", action="append", default=[])
    mark_parser.add_argument("--expected-section", action="append", default=[])
    mark_parser.add_argument("--threshold", type=int, default=None)

    should_parser = sub.add_parser("should-refresh", help="Return retired read-only decision")
    add_common_paths(should_parser)
    should_parser.add_argument("--reason", default="threshold", choices=["threshold", "pre-query", "manual", "full-graph-fresh"])
    should_parser.add_argument("--threshold", type=int, default=None)
    should_parser.add_argument("--exit-code", action="store_true")

    refresh_parser = sub.add_parser("refresh", help="Fail closed; use native refresh")
    add_common_paths(refresh_parser)
    refresh_parser.add_argument("--reason", default="threshold", choices=["threshold", "pre-query", "manual", "full-graph-fresh"])
    refresh_parser.add_argument("--threshold", type=int, default=None)
    refresh_parser.add_argument("--dry-run", action="store_true")
    refresh_parser.add_argument("--force", action="store_true")
    refresh_parser.add_argument("--force-full-rebuild", action="store_true")
    refresh_parser.add_argument("--reuse-vector-cache", action="store_true")
    refresh_parser.add_argument("--embedding-profile", default="native")
    refresh_parser.add_argument("--reuse-validation-report", type=Path, default=None)

    clear_parser = sub.add_parser("clear-success", help="Fail closed; old refresh success cannot be recorded")
    add_common_paths(clear_parser)
    clear_parser.add_argument("--reason", default="external-success")
    clear_parser.add_argument("--import-report", type=Path, default=None)

    args = parser.parse_args()
    root = args.root.resolve()
    state_dir = args.state_dir.resolve()

    if args.command == "status":
        print_json(_readonly_status(root, state_dir, reason=args.reason, threshold=args.threshold))
        return 0
    if args.command == "should-refresh":
        print_json(_readonly_status(root, state_dir, reason=args.reason, threshold=args.threshold))
        return 0
    if args.command == "refresh":
        payload = _readonly_status(root, state_dir, reason=args.reason, threshold=args.threshold)
        payload.update({"command": "refresh", "dry_run": bool(args.dry_run), "would_run": False, "commands": []})
        print_json(payload)
        return 0 if args.dry_run else 1
    if args.command == "mark-pending":
        print_json(_retired_payload(command="mark-pending", dry_run=False))
        return 1
    if args.command == "clear-success":
        print_json(_retired_payload(command="clear-success", dry_run=False))
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
