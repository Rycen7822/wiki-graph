#!/usr/bin/env python3
"""CLI entrypoints for native llm-wiki workspaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_wiki_native.pointers import finalize_prepared_workspace, rollback_active_workspace
from llm_wiki_native.workspace_build import build_workspace_from_state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and audit llm-wiki native workspaces")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-workspace", help="Build an audited native SQLite workspace from existing state artifacts")
    build.add_argument("--state-dir", type=Path, required=True)
    build.add_argument("--db", type=Path, required=True)
    build.add_argument("--workspace-id", required=True)
    build.add_argument("--zvec-workspace", type=Path, required=True)
    build.add_argument("--root", type=Path, help="Optional source wiki root for lexical sidecar spans")
    build.add_argument("--prepared-workspace-file", type=Path)
    finalize = sub.add_parser("finalize-prepared", help="Promote a prepared native workspace pointer to active")
    finalize.add_argument("--prepared-workspace-file", type=Path, required=True)
    finalize.add_argument("--active-workspace-file", type=Path, required=True)
    finalize.add_argument("--history-file", type=Path, required=True)
    finalize.add_argument("--reason", required=True)
    rollback = sub.add_parser("rollback-active", help="Restore the previous active native workspace pointer")
    rollback.add_argument("--active-workspace-file", type=Path, required=True)
    rollback.add_argument("--history-file", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build-workspace":
        report = build_workspace_from_state(
            args.state_dir,
            args.db,
            args.workspace_id,
            zvec_path=args.zvec_workspace,
            prepared_workspace_path=args.prepared_workspace_file,
            source_root=args.root,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "finalize-prepared":
        active = finalize_prepared_workspace(
            args.prepared_workspace_file,
            args.active_workspace_file,
            args.history_file,
            reason=args.reason,
        )
        print(json.dumps(active, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "rollback-active":
        active = rollback_active_workspace(
            args.active_workspace_file,
            args.history_file,
        )
        print(json.dumps(active, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
