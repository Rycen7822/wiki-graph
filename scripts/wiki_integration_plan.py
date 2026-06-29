#!/usr/bin/env python3
"""Deterministic dry-run planning for pending llm-wiki raw integration.

This script plans machine-owned integration surfaces from the external pending
wiki-integration ledger. It does not edit compiled pages, metadata maps, or the
human wiki root; callers must review/apply a generated plan separately.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from wiki_native_lib import (
    DEFAULT_STATE_DIR,
    DEFAULT_WIKI_ROOT,
    ensure_state_dirs,
    now_stamp,
    pending_wiki_integration_status,
    print_json,
    sha256_text,
)


def _stable_unique_strings(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return sorted({str(item).strip() for item in values if str(item).strip()})


def _pending_sort_key(item: dict[str, Any]) -> tuple[str, str, str]:
    return (str(item.get("raw_path") or ""), str(item.get("title") or ""), str(item.get("source_id") or ""))


def _canonical_plan_hash(operations: list[dict[str, Any]], compiled_page_writes: list[dict[str, Any]]) -> str:
    return sha256_text(
        json.dumps(
            {
                "operations": operations,
                "compiled_page_writes": compiled_page_writes,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def build_wiki_integration_plan(root: Path, state_dir: Path, reason: str = "manual", threshold: int | None = None) -> dict[str, Any]:
    """Return a deterministic dry-run plan for pending raw-note integration.

    Ambiguous records are conservative: when there is no topic hint, the plan adds
    a review-queue operation and never writes a compiled page operation.
    """
    root = Path(root)
    state_dir = Path(state_dir)
    status = pending_wiki_integration_status(root, state_dir, reason=reason, threshold=threshold)
    actionable = [item for item in status.get("actionable_pending") or [] if isinstance(item, dict)]
    operations: list[dict[str, Any]] = []
    routed_paths: list[str] = []
    review_paths: list[str] = []

    for item in sorted(actionable, key=_pending_sort_key):
        raw_path = str(item.get("raw_path") or "")
        title = str(item.get("title") or Path(raw_path).stem)
        topic_hints = _stable_unique_strings(item.get("topic_hints"))
        required_sections = _stable_unique_strings(item.get("required_sections"))
        if not topic_hints:
            review_paths.append(raw_path)
            operations.append(
                {
                    "op": "review_queue_add",
                    "raw_path": raw_path,
                    "title": title,
                    "reason": "missing_topic_hints",
                }
            )
            continue
        routed_paths.append(raw_path)
        operations.append(
            {
                "op": "raw_map_upsert",
                "raw_path": raw_path,
                "title": title,
                "source_id": str(item.get("source_id") or ""),
                "required_sections": required_sections,
            }
        )
        for topic in topic_hints:
            operations.append(
                {
                    "op": "topic_map_route",
                    "topic": topic,
                    "raw_path": raw_path,
                    "title": title,
                }
            )
    if routed_paths or review_paths:
        operations.append(
            {
                "op": "log_batch_entry",
                "routed_paths": sorted(routed_paths),
                "review_paths": sorted(review_paths),
                "reason": reason,
            }
        )

    compiled_page_writes: list[dict[str, Any]] = []
    plan_hash = _canonical_plan_hash(operations, compiled_page_writes)
    return {
        "schema_version": 1,
        "created_at": now_stamp(),
        "dry_run": True,
        "writes_wiki": False,
        "root": str(root),
        "state_dir": str(state_dir),
        "reason": reason,
        "pending_count": status.get("pending_count"),
        "actionable_pending_count": status.get("actionable_pending_count"),
        "review_pending_count": status.get("review_pending_count"),
        "plan_hash": plan_hash,
        "operations": operations,
        "compiled_page_writes": compiled_page_writes,
        "status_reasons": status.get("reasons") or [],
    }


def write_wiki_integration_plan_report(state_dir: Path, plan: dict[str, Any], output: Path | None = None) -> Path:
    state_dir = Path(state_dir)
    ensure_state_dirs(state_dir)
    if output is None:
        report_dir = state_dir / "wiki_integration_plans"
        report_dir.mkdir(parents=True, exist_ok=True)
        output = report_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_wiki_integration_plan.json"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic dry-run plan for pending llm-wiki raw integration")
    parser.add_argument("--root", type=Path, default=DEFAULT_WIKI_ROOT)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--reason", default="manual", choices=["threshold", "pre-query", "query", "manual", "integrate", "wiki-query"])
    parser.add_argument("--threshold", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Report path; defaults to state/wiki_integration_plans/<stamp>_wiki_integration_plan.json")
    args = parser.parse_args()

    plan = build_wiki_integration_plan(args.root, args.state_dir, reason=args.reason, threshold=args.threshold)
    report_path = write_wiki_integration_plan_report(args.state_dir, plan, output=args.output)
    print_json({"plan_hash": plan["plan_hash"], "dry_run": True, "writes_wiki": False, "operations": len(plan["operations"]), "report_path": report_path.as_posix()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
