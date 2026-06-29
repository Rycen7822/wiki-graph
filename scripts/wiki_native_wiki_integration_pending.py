"""Pending wiki-integration ledger helpers shared by native-safe scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from wiki_native_docs import raw_clip_files
from wiki_native_state import ensure_state_dirs
from wiki_native_wiki_checks import now_stamp

PENDING_WIKI_INTEGRATION_LEDGER = "pending_wiki_integration.json"
DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD = 10
WIKI_INTEGRATION_ACTIONABLE_STATUSES = {"raw_saved"}
WIKI_INTEGRATION_REVIEW_STATUSES = {"needs_review"}
WIKI_INTEGRATION_TERMINAL_STATUSES = {"failed", "skipped_duplicate"}
PENDING_WIKI_INTEGRATION_LEDGER_FIELDS = {
    "version",
    "threshold",
    "last_successful_integration_at",
    "last_successful_integration_raw_count",
    "last_integrated_paths",
    "pending",
    "dirty",
    "last_failed_integration",
    "last_pending_update_at",
    "current_raw_count_at_last_pending_update",
    "last_integration_reason",
    "last_cleared_pending",
    "last_cleared_pending_count",
    "last_remaining_pending_count",
}


def normalize_pending_wiki_integration_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    """Keep only the current pending-wiki-integration ledger schema fields."""

    return {key: value for key, value in ledger.items() if str(key) in PENDING_WIKI_INTEGRATION_LEDGER_FIELDS}


def pending_wiki_integration_ledger_path(state_dir: Path) -> Path:
    return state_dir / PENDING_WIKI_INTEGRATION_LEDGER


def default_pending_wiki_integration_ledger(threshold: int | None = None) -> dict[str, Any]:
    return {
        "version": 1,
        "threshold": int(threshold or DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD),
        "last_successful_integration_at": None,
        "last_successful_integration_raw_count": None,
        "last_integrated_paths": [],
        "pending": [],
        "dirty": False,
        "last_failed_integration": None,
    }


def load_pending_wiki_integration_ledger(state_dir: Path, threshold: int | None = None) -> dict[str, Any]:
    ensure_state_dirs(state_dir)
    path = pending_wiki_integration_ledger_path(state_dir)
    if not path.exists():
        return default_pending_wiki_integration_ledger(threshold)
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(ledger, dict):
        raise ValueError(f"{path} must contain a JSON object")
    merged = default_pending_wiki_integration_ledger(threshold)
    merged.update(normalize_pending_wiki_integration_ledger(ledger))
    merged["threshold"] = int(
        threshold if threshold is not None else (merged.get("threshold") or DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD)
    )
    pending = merged.get("pending") or []
    if not isinstance(pending, list):
        raise ValueError(f"{path} field pending must be a list")
    merged["pending"] = pending
    merged["dirty"] = bool(merged.get("dirty"))
    return merged


def save_pending_wiki_integration_ledger(state_dir: Path, ledger: dict[str, Any]) -> Path:
    ensure_state_dirs(state_dir)
    path = pending_wiki_integration_ledger_path(state_dir)
    tmp = path.with_suffix(path.suffix + ".tmp")
    ledger = normalize_pending_wiki_integration_ledger(ledger)
    tmp.write_text(json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def mark_pending_wiki_integration(
    state_dir: Path,
    root: Path,
    raw_path: str = "",
    title: str = "",
    source_id: str = "",
    topic_hints: list[str] | None = None,
    required_sections: list[str] | None = None,
    resource_status_summary: str = "",
    status: str = "raw_saved",
    threshold: int | None = None,
) -> dict[str, Any]:
    ledger = load_pending_wiki_integration_ledger(state_dir, threshold=threshold)
    effective_threshold = int(
        threshold if threshold is not None else (ledger.get("threshold") or DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD)
    )
    ledger["threshold"] = effective_threshold
    captured_at = now_stamp()
    entry = {
        "raw_path": raw_path,
        "title": title or (Path(raw_path).stem if raw_path else ""),
        "source_id": source_id,
        "captured_at": captured_at,
        "status": status,
        "topic_hints": topic_hints or [],
        "required_sections": required_sections or [],
        "resource_status_summary": resource_status_summary,
    }
    pending = list(ledger.get("pending") or [])
    replaced = False
    if raw_path:
        for index, old in enumerate(pending):
            if isinstance(old, dict) and old.get("raw_path") == raw_path:
                pending[index] = {**old, **entry}
                replaced = True
                break
    if not replaced:
        pending.append(entry)
    ledger["pending"] = pending
    ledger["dirty"] = True
    ledger["last_pending_update_at"] = captured_at
    ledger["current_raw_count_at_last_pending_update"] = len(raw_clip_files(root))
    save_pending_wiki_integration_ledger(state_dir, ledger)
    return entry


def pending_wiki_integration_status(
    root: Path,
    state_dir: Path,
    reason: str = "threshold",
    threshold: int | None = None,
) -> dict[str, Any]:
    ledger = load_pending_wiki_integration_ledger(state_dir, threshold=threshold)
    effective_threshold = int(
        threshold if threshold is not None else (ledger.get("threshold") or DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD)
    )
    pending = ledger.get("pending") or []
    pending_count = len(pending)
    actionable_pending = [
        item
        for item in pending
        if isinstance(item, dict) and str(item.get("status") or "raw_saved") in WIKI_INTEGRATION_ACTIONABLE_STATUSES
    ]
    terminal_pending = [
        item
        for item in pending
        if isinstance(item, dict) and str(item.get("status") or "raw_saved") in WIKI_INTEGRATION_TERMINAL_STATUSES
    ]
    review_pending = [
        item
        for item in pending
        if isinstance(item, dict)
        and str(item.get("status") or "raw_saved") not in WIKI_INTEGRATION_ACTIONABLE_STATUSES
        and str(item.get("status") or "raw_saved") not in WIKI_INTEGRATION_TERMINAL_STATUSES
    ]
    actionable_pending_count = len(actionable_pending)
    terminal_pending_count = len(terminal_pending)
    review_pending_count = len(review_pending)
    blocking_pending_count = actionable_pending_count + review_pending_count
    raw_count = len(raw_clip_files(root))
    normalized_reason = reason.strip().lower().replace("_", "-")
    reasons: list[str] = []
    if normalized_reason == "manual" and blocking_pending_count:
        reasons.append("manual_requested")
    if actionable_pending_count >= effective_threshold:
        reasons.append("pending_threshold_reached")
    if normalized_reason in {"pre-query", "query", "integrate", "wiki-query"} and actionable_pending_count:
        reasons.append("pending_items_for_wiki_integration")
    if review_pending_count:
        reasons.append("pending_items_need_review")
    if normalized_reason in {"pre-query", "query", "integrate", "wiki-query", "manual"}:
        if ledger.get("dirty") and actionable_pending_count:
            reasons.append("dirty_ledger_for_wiki_integration")
    integration_reasons = {
        "manual_requested",
        "pending_threshold_reached",
        "pending_items_for_wiki_integration",
        "dirty_ledger_for_wiki_integration",
    }
    should_integrate = bool(integration_reasons.intersection(reasons))
    should_review = review_pending_count > 0
    if should_integrate:
        next_required_action = "wiki_integration"
    elif should_review:
        next_required_action = "manual_review"
    else:
        next_required_action = "none"
    return {
        "reason": normalized_reason,
        "should_integrate": should_integrate,
        "should_review": should_review,
        "next_required_action": next_required_action,
        "reasons": sorted(set(reasons)),
        "pending_count": pending_count,
        "actionable_pending_count": actionable_pending_count,
        "terminal_pending_count": terminal_pending_count,
        "review_pending_count": review_pending_count,
        "blocking_pending_count": blocking_pending_count,
        "threshold": effective_threshold,
        "dirty": bool(ledger.get("dirty")),
        "raw_clip_count": raw_count,
        "last_successful_integration_raw_count": ledger.get("last_successful_integration_raw_count"),
        "last_successful_integration_at": ledger.get("last_successful_integration_at"),
        "ledger_path": str(pending_wiki_integration_ledger_path(state_dir)),
        "pending": pending,
        "actionable_pending": actionable_pending,
        "terminal_pending": terminal_pending,
        "review_pending": review_pending,
    }


def clear_pending_wiki_integration_after_success(
    root: Path,
    state_dir: Path,
    integrated_paths: list[str] | None = None,
    reason: str = "integration",
    mark_graph_pending: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ledger = load_pending_wiki_integration_ledger(state_dir)
    pending = list(ledger.get("pending") or [])
    integrated_path_set = {path for path in (integrated_paths or []) if path}
    marked_graph_pending: list[dict[str, Any]] = []
    cleared_pending: list[dict[str, Any]] = []
    remaining_pending: list[Any] = []
    for item in pending:
        if not isinstance(item, dict):
            remaining_pending.append(item)
            continue
        raw_path = str(item.get("raw_path") or "")
        item_status = str(item.get("status") or "raw_saved")
        should_clear = (
            bool(raw_path and raw_path in integrated_path_set)
            if integrated_path_set
            else item_status in WIKI_INTEGRATION_ACTIONABLE_STATUSES
        )
        if not should_clear:
            remaining_pending.append(item)
            continue
        cleared_pending.append(item)
        if mark_graph_pending and raw_path and item_status in WIKI_INTEGRATION_ACTIONABLE_STATUSES:
            marked_graph_pending.append(mark_graph_pending(raw_path, item))
    cleared_at = now_stamp()
    cleared_paths = [str(item.get("raw_path") or "") for item in cleared_pending if item.get("raw_path")]
    remaining_blocking = [
        item
        for item in remaining_pending
        if isinstance(item, dict) and str(item.get("status") or "raw_saved") not in WIKI_INTEGRATION_TERMINAL_STATUSES
    ]
    ledger["last_successful_integration_at"] = cleared_at
    ledger["last_successful_integration_raw_count"] = len(raw_clip_files(root))
    ledger["last_integration_reason"] = reason
    ledger["last_integrated_paths"] = integrated_paths or cleared_paths
    ledger["last_cleared_pending"] = cleared_pending
    ledger["last_cleared_pending_count"] = len(cleared_pending)
    ledger["last_remaining_pending_count"] = len(remaining_pending)
    ledger["last_marked_graph_pending"] = marked_graph_pending
    ledger["last_marked_graph_pending_count"] = len(marked_graph_pending)
    ledger["pending"] = remaining_pending
    ledger["dirty"] = bool(remaining_blocking)
    ledger["last_failed_integration"] = None
    save_pending_wiki_integration_ledger(state_dir, ledger)
    return {
        "cleared_count": len(cleared_pending),
        "remaining_pending_count": len(remaining_pending),
        "last_successful_integration_at": ledger["last_successful_integration_at"],
        "last_successful_integration_raw_count": ledger["last_successful_integration_raw_count"],
        "last_integrated_paths": ledger["last_integrated_paths"],
        "marked_graph_pending_count": len(marked_graph_pending),
        "marked_graph_pending": marked_graph_pending,
        "ledger_path": str(pending_wiki_integration_ledger_path(state_dir)),
    }


def record_pending_wiki_integration_failure(state_dir: Path, reason: str, message: str = "") -> dict[str, Any]:
    ledger = load_pending_wiki_integration_ledger(state_dir)
    failure = {"at": now_stamp(), "reason": reason, "message": message}
    ledger["last_failed_integration"] = failure
    ledger["dirty"] = True
    save_pending_wiki_integration_ledger(state_dir, ledger)
    return failure
