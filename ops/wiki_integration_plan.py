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

from ops.wiki_native_lib import (
    DEFAULT_STATE_DIR,
    DEFAULT_WIKI_ROOT,
    ensure_state_dirs,
    now_stamp,
    pending_wiki_integration_status,
    print_json,
    sha256_text,
)


SUPPORTED_LOCAL_APPLY_OPS = {"raw_map_upsert", "topic_map_route", "log_batch_entry"}


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


def load_wiki_integration_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def validate_wiki_integration_plan(
    plan: dict[str, Any],
    root: Path,
    state_dir: Path,
    *,
    expected_hash: str | None = None,
) -> dict[str, Any]:
    operations = plan.get("operations")
    if not isinstance(operations, list):
        operations = []
    compiled_page_writes = plan.get("compiled_page_writes")
    if not isinstance(compiled_page_writes, list):
        compiled_page_writes = []

    errors: list[str] = []
    if plan.get("schema_version") != 1:
        errors.append("schema_version_unsupported")
    plan_root = plan.get("root")
    if plan_root and Path(str(plan_root)).resolve() != Path(root).resolve():
        errors.append("root_mismatch")
    plan_state_dir = plan.get("state_dir")
    if plan_state_dir and Path(str(plan_state_dir)).resolve() != Path(state_dir).resolve():
        errors.append("state_dir_mismatch")

    canonical_hash = _canonical_plan_hash(operations, compiled_page_writes)
    plan_hash = str(plan.get("plan_hash") or "")
    if plan_hash != canonical_hash:
        errors.append("plan_hash_mismatch")
    if expected_hash and plan_hash != expected_hash:
        errors.append("expected_plan_hash_mismatch")
    if compiled_page_writes:
        errors.append("compiled_page_writes_not_supported")

    for op in operations:
        if not isinstance(op, dict):
            errors.append("unsupported_operation")
            continue
        op_name = str(op.get("op") or "")
        if op_name == "review_queue_add":
            errors.append("manual_review_required")
        elif op_name not in SUPPORTED_LOCAL_APPLY_OPS:
            errors.append("unsupported_operation")
    return {
        "ok": not errors,
        "errors": sorted(set(errors)),
        "plan_hash": plan_hash,
        "canonical_plan_hash": canonical_hash,
        "operations": len(operations),
    }


def _read_text_or_default(path: Path, default: str) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return default


def _write_text_if_changed(path: Path, text: str, changed_paths: set[str]) -> None:
    old = path.read_text(encoding="utf-8") if path.exists() else None
    if old == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    changed_paths.add(path.as_posix())


def _source_ref(raw_path: str) -> str:
    return f"../{raw_path.strip().lstrip('/')}"


def _ensure_frontmatter_source(text: str, source_ref: str) -> str:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    end_idx = next((idx for idx, line in enumerate(lines[1:], start=1) if line.strip() == "---"), None)
    if end_idx is None:
        return text
    frontmatter = lines[1:end_idx]
    source_line = f"- {source_ref}"
    if source_line in frontmatter:
        return text
    sources_idx = next((idx for idx, line in enumerate(frontmatter) if line.strip() == "sources:"), None)
    if sources_idx is None:
        frontmatter.extend(["sources:", source_line])
    else:
        insert_at = sources_idx + 1
        while insert_at < len(frontmatter) and frontmatter[insert_at].startswith("- "):
            insert_at += 1
        frontmatter.insert(insert_at, source_line)
    rebuilt = ["---", *frontmatter, "---", *lines[end_idx + 1 :]]
    suffix = "\n" if text.endswith("\n") else ""
    return "\n".join(rebuilt) + suffix


def _append_unique_line(text: str, heading: str, line: str) -> str:
    if line in text:
        return text
    base = text.rstrip()
    if heading not in base:
        base = f"{base}\n\n{heading}\n" if base else f"{heading}\n"
    return f"{base.rstrip()}\n{line}\n"


def _raw_map_line(op: dict[str, Any]) -> str:
    raw_path = str(op.get("raw_path") or "").strip()
    title = str(op.get("title") or Path(raw_path).stem).strip()
    source_id = str(op.get("source_id") or "").strip()
    sections = _stable_unique_strings(op.get("required_sections"))
    suffix_parts = []
    if source_id:
        suffix_parts.append(f"source: {source_id}")
    if sections:
        suffix_parts.append("sections: " + ", ".join(sections))
    suffix = f" — {'; '.join(suffix_parts)}" if suffix_parts else ""
    return f"- `{raw_path}` — {title}{suffix}"


def _topic_map_line(op: dict[str, Any]) -> str:
    topic = str(op.get("topic") or "").strip()
    raw_path = str(op.get("raw_path") or "").strip()
    title = str(op.get("title") or Path(raw_path).stem).strip()
    return f"- **{topic}** → `{raw_path}` — {title}"


def _log_batch_entry(plan: dict[str, Any], reason: str) -> str:
    plan_hash = str(plan.get("plan_hash") or "")
    routed: set[str] = set()
    reviewed: set[str] = set()
    for op in plan.get("operations") or []:
        if not isinstance(op, dict):
            continue
        if op.get("op") == "raw_map_upsert" and op.get("raw_path"):
            routed.add(str(op["raw_path"]))
        if op.get("op") == "log_batch_entry":
            routed.update(str(path) for path in op.get("routed_paths") or [])
            reviewed.update(str(path) for path in op.get("review_paths") or [])
    lines = [
        f"## [{now_stamp()}] raw-fast batch wiki integration | deterministic local apply",
        "",
        f"- Trigger: `batch_wiki_integration.py integrate-local`; reason=`{reason}`; plan_hash=`{plan_hash}`; routed_raw_notes={len(routed)}; review_paths={len(reviewed)}.",
    ]
    if routed:
        lines.append("- Routed raw notes: " + ", ".join(f"`{path}`" for path in sorted(routed)) + ".")
    return "\n".join(lines).rstrip() + "\n"


def apply_wiki_integration_plan(
    root: Path,
    state_dir: Path,
    plan: dict[str, Any],
    *,
    reason: str = "integrate",
    dry_run: bool = False,
    expected_hash: str | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    state_dir = Path(state_dir).resolve()
    validation = validate_wiki_integration_plan(plan, root, state_dir, expected_hash=expected_hash)
    if not validation["ok"]:
        return {
            "applied": False,
            "dry_run": dry_run,
            "writes_wiki": False,
            "errors": validation["errors"],
            "plan_hash": validation["plan_hash"],
            "canonical_plan_hash": validation["canonical_plan_hash"],
        }

    raw_map_path = root / "_meta" / "raw-clip-map.md"
    topic_map_path = root / "_meta" / "topic-map.md"
    log_path = root / "log.md"
    raw_map_text = _read_text_or_default(raw_map_path, "# Raw Clip Map\n")
    topic_map_text = _read_text_or_default(topic_map_path, "# Topic Map\n")
    log_text = _read_text_or_default(log_path, "# Wiki Log\n")
    changed_paths: set[str] = set()

    for op in plan.get("operations") or []:
        if not isinstance(op, dict):
            continue
        op_name = op.get("op")
        raw_path = str(op.get("raw_path") or "").strip()
        if op_name == "raw_map_upsert" and raw_path:
            raw_map_text = _ensure_frontmatter_source(raw_map_text, _source_ref(raw_path))
            raw_map_text = _append_unique_line(raw_map_text, "## Deterministic integration routes", _raw_map_line(op))
        elif op_name == "topic_map_route" and raw_path:
            topic_map_text = _ensure_frontmatter_source(topic_map_text, _source_ref(raw_path))
            topic_map_text = _append_unique_line(topic_map_text, "## Deterministic topic routes", _topic_map_line(op))
        elif op_name == "log_batch_entry":
            plan_hash = str(plan.get("plan_hash") or "")
            if plan_hash and plan_hash not in log_text:
                log_text = f"{log_text.rstrip()}\n\n{_log_batch_entry(plan, reason)}"

    if not dry_run:
        _write_text_if_changed(raw_map_path, raw_map_text, changed_paths)
        _write_text_if_changed(topic_map_path, topic_map_text, changed_paths)
        _write_text_if_changed(log_path, log_text, changed_paths)
    return {
        "applied": not dry_run,
        "dry_run": dry_run,
        "writes_wiki": not dry_run,
        "errors": [],
        "plan_hash": validation["plan_hash"],
        "operations_applied": validation["operations"],
        "changed_paths": sorted(changed_paths),
    }


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
