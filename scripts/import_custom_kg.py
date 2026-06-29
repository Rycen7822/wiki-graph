#!/usr/bin/env python3
"""Retired cold custom KG import shim.

Dry-run still summarizes deterministic custom KG payloads. Non-dry cold import
is fail-closed; native state export and native zvec materialization own staging.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from native_runtime_env import load_env_file, redact_summary
from wiki_native_lib import (
    DEFAULT_STATE_DIR,
    DEFAULT_WIKI_ROOT,
    DEFAULT_WORKDIR,
    build_custom_kg_payload,
    ensure_state_dirs,
    now_stamp,
    print_json,
)
from wiki_wikigraph_compat_names import retired_graph_package_name


def build_rag(workdir: Path, storage_dir: Path | None = None):
    raise RuntimeError(
        f"direct {retired_graph_package_name()} object construction is retired after native zvec production cutover; "
        "use export-manifest plus native_zvec_materialize.py preflight/build"
    )


async def run_import(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    workdir = args.workdir.resolve()
    state_dir = args.state_dir.resolve()
    ensure_state_dirs(state_dir)
    env_values = load_env_file(workdir / ".env")

    payload, payload_summary = build_custom_kg_payload(root, state_dir, args.limit_docs, args.limit_edges)
    from custom_kg_incremental import build_custom_kg_manifest

    desired_manifest = build_custom_kg_manifest(payload)
    summary: dict[str, Any] = {
        "started_at": now_stamp(),
        "wiki_root": str(root),
        "workdir": str(workdir),
        "state_dir": str(state_dir),
        "payload": payload_summary,
        "manifest": desired_manifest.get("summary", {}),
        "import_mode": "full_rebuild",
        "dry_run": args.dry_run,
        "env": redact_summary({k: env_values.get(k, "") for k in [
            "LLM_BINDING", "LLM_BINDING_HOST", "LLM_MODEL",
            "EMBEDDING_BINDING", "EMBEDDING_BINDING_HOST", "EMBEDDING_MODEL", "EMBEDDING_DIM",
            "MAX_GLEANING", "MAX_EXTRACT_INPUT_TOKENS",
        ]}),
    }
    if args.dry_run:
        return summary
    raise RuntimeError(
        "custom KG cold import is retired; use custom_kg_incremental.py export-manifest "
        "and native_zvec_materialize.py build/preflight for native staging"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Retired cold custom_kg import shim; dry-run only")
    parser.add_argument("--root", type=Path, default=DEFAULT_WIKI_ROOT)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--limit-edges", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-server-running", action="store_true")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=9621)
    args = parser.parse_args()
    try:
        print_json(asyncio.run(run_import(args)))
        return 0
    except Exception as exc:
        print_json({"error": type(exc).__name__, "message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
