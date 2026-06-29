"""Read-only diagnostics for the retired wikigraph refresh pending ledger."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from wiki_native_docs import collect_source_docs, raw_clip_files
from wiki_native_state import ensure_state_dirs
from wiki_native_wiki_checks import now_stamp
from wiki_native_wiki_integration_pending import pending_wiki_integration_status

PENDING_WIKIGRAPH_REFRESH_LEDGER = "pending_wikigraph_refresh.json"
DEFAULT_PENDING_WIKIGRAPH_REFRESH_THRESHOLD = 10
WIKIGRAPH_REFRESH_LEDGER_WRITE_RETIRED_MESSAGE = (
    "wikigraph refresh pending ledger writes are retired after native zvec production cutover; "
    "use batch_native_refresh.py status/refresh and pending_native_refresh.json"
)


def pending_wikigraph_refresh_ledger_path(state_dir: Path) -> Path:
    return state_dir / PENDING_WIKIGRAPH_REFRESH_LEDGER


def default_wikigraph_refresh_ledger(threshold: int | None = None) -> dict[str, Any]:
    return {
        "version": 1,
        "threshold": int(threshold or DEFAULT_PENDING_WIKIGRAPH_REFRESH_THRESHOLD),
        "last_successful_refresh_at": None,
        "last_successful_raw_count": None,
        "last_successful_import_payload": {},
        "pending": [],
        "dirty": False,
        "last_failed_refresh": None,
    }


def load_wikigraph_refresh_ledger(state_dir: Path, threshold: int | None = None) -> dict[str, Any]:
    ensure_state_dirs(state_dir)
    path = pending_wikigraph_refresh_ledger_path(state_dir)
    if not path.exists():
        return default_wikigraph_refresh_ledger(threshold)
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(ledger, dict):
        raise ValueError(f"{path} must contain a JSON object")
    merged = default_wikigraph_refresh_ledger(threshold)
    merged.update(ledger)
    merged["threshold"] = int(
        threshold if threshold is not None else (merged.get("threshold") or DEFAULT_PENDING_WIKIGRAPH_REFRESH_THRESHOLD)
    )
    pending = merged.get("pending") or []
    if not isinstance(pending, list):
        raise ValueError(f"{path} field pending must be a list")
    merged["pending"] = pending
    merged["dirty"] = bool(merged.get("dirty"))
    return merged


def save_wikigraph_refresh_ledger(state_dir: Path, ledger: dict[str, Any]) -> Path:
    raise RuntimeError(WIKIGRAPH_REFRESH_LEDGER_WRITE_RETIRED_MESSAGE)


def wikigraph_refresh_import_summary(import_report_path: Path) -> dict[str, Any]:
    if not import_report_path.exists():
        return {}
    report = json.loads(import_report_path.read_text(encoding="utf-8", errors="ignore"))
    payload = report.get("payload") if isinstance(report, dict) else {}
    if not isinstance(payload, dict):
        payload = {}
    return {
        "finished_at": report.get("finished_at") if isinstance(report, dict) else None,
        "payload": {
            key: payload.get(key)
            for key in [
                "chunks",
                "entities",
                "relationships",
                "raw_section_chunks",
                "section_similarity_relationships",
            ]
            if key in payload
        },
        "report_path": str(import_report_path),
    }


def _parse_refresh_time(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"]:
        try:
            return dt.datetime.strptime(value[: len(dt.datetime.now().strftime(fmt))], fmt)
        except Exception:
            continue
    return None


def wiki_markdown_latest_mtime(root: Path) -> str | None:
    """Latest mtime among Markdown files that participate in custom_kg."""
    mtimes = [doc.path.stat().st_mtime for doc in collect_source_docs(root) if doc.path.exists()]
    if not mtimes:
        return None
    return dt.datetime.fromtimestamp(max(mtimes)).strftime("%Y-%m-%d %H:%M:%S")


def mark_wikigraph_refresh_pending(
    state_dir: Path,
    root: Path,
    raw_path: str = "",
    title: str = "",
    event_type: str = "new_raw_note",
    changed_surfaces: list[str] | None = None,
    expected_sections: list[str] | None = None,
    threshold: int | None = None,
) -> dict[str, Any]:
    raise RuntimeError(WIKIGRAPH_REFRESH_LEDGER_WRITE_RETIRED_MESSAGE)


def pending_wikigraph_refresh_status(
    root: Path,
    state_dir: Path,
    reason: str = "threshold",
    threshold: int | None = None,
    import_report_path: Path | None = None,
) -> dict[str, Any]:
    normalized_reason = reason.strip().lower().replace("_", "-")
    ledger = load_wikigraph_refresh_ledger(state_dir, threshold=threshold)
    upstream_reason = normalized_reason if normalized_reason in {"pre-query", "query", "manual"} else "threshold"
    upstream_wiki_integration = pending_wiki_integration_status(root, state_dir, reason=upstream_reason)
    upstream_actionable_count = int(upstream_wiki_integration.get("actionable_pending_count") or 0)
    upstream_review_count = int(upstream_wiki_integration.get("review_pending_count") or 0)
    upstream_pending_count = int(
        upstream_wiki_integration.get("blocking_pending_count") or (upstream_actionable_count + upstream_review_count)
    )
    effective_threshold = int(
        threshold if threshold is not None else (ledger.get("threshold") or DEFAULT_PENDING_WIKIGRAPH_REFRESH_THRESHOLD)
    )
    pending = ledger.get("pending") or []
    pending_count = len(pending)
    raw_count = len(raw_clip_files(root))
    last_success_count = ledger.get("last_successful_raw_count")
    prequery_like = normalized_reason in {"pre-query", "query", "full-graph-fresh", "manual"}
    report_path = import_report_path or (state_dir / "custom_kg_import_report.json")
    import_summary = wikigraph_refresh_import_summary(report_path) if prequery_like else {}
    last_success_at = ledger.get("last_successful_refresh_at") or import_summary.get("finished_at")
    reasons: list[str] = []
    blocked_reasons: list[str] = []
    if normalized_reason == "manual":
        reasons.append("manual_requested")
    if pending_count >= effective_threshold:
        reasons.append("pending_threshold_reached")
    if normalized_reason in {"pre-query", "query", "full-graph-fresh"} and pending_count:
        reasons.append("pending_items_for_pre_query")
    if prequery_like and ledger.get("dirty"):
        reasons.append("dirty_ledger_for_pre_query")
    if isinstance(last_success_count, int) and raw_count > last_success_count and prequery_like:
        reasons.append("raw_count_newer_than_last_success")
    if prequery_like and not import_summary:
        reasons.append("import_report_missing")
    latest_mtime = wiki_markdown_latest_mtime(root) if prequery_like else None
    latest_dt = _parse_refresh_time(latest_mtime)
    success_dt = _parse_refresh_time(str(last_success_at) if last_success_at else None)
    if prequery_like and latest_dt and success_dt and latest_dt > success_dt:
        reasons.append("wiki_markdown_newer_than_last_import")
    if upstream_actionable_count:
        blocked_reasons.append("pending_wiki_integration_before_wikigraph_refresh")
    if upstream_review_count:
        blocked_reasons.append("pending_wiki_integration_needs_manual_review")
    blocked_by_pending_wiki_integration = bool(blocked_reasons)
    would_refresh_if_unblocked = bool(reasons)
    should_refresh = bool(would_refresh_if_unblocked and not blocked_by_pending_wiki_integration)
    if upstream_actionable_count:
        next_required_action = "wiki_integration"
    elif upstream_review_count:
        next_required_action = "manual_review"
    elif should_refresh:
        next_required_action = "wikigraph_refresh"
    else:
        next_required_action = "none"
    return {
        "reason": normalized_reason,
        "should_refresh": should_refresh,
        "would_refresh_if_unblocked": would_refresh_if_unblocked,
        "blocked_by_pending_wiki_integration": blocked_by_pending_wiki_integration,
        "blocked_reasons": sorted(set(blocked_reasons)),
        "next_required_action": next_required_action,
        "reasons": sorted(set(reasons)),
        "pending_count": pending_count,
        "graph_ready_pending_count": pending_count,
        "raw_fast_pending_wiki_integration_count": upstream_pending_count,
        "raw_fast_actionable_wiki_integration_count": upstream_actionable_count,
        "raw_fast_review_wiki_integration_count": upstream_review_count,
        "total_not_graph_fresh_count": pending_count + upstream_pending_count,
        "pending_count_excludes_raw_fast": True,
        "threshold": effective_threshold,
        "dirty": bool(ledger.get("dirty")),
        "raw_clip_count": raw_count,
        "last_successful_raw_count": last_success_count,
        "last_successful_refresh_at": last_success_at,
        "latest_wiki_markdown_mtime": latest_mtime,
        "ledger_path": str(pending_wikigraph_refresh_ledger_path(state_dir)),
        "pending": pending,
        "upstream_wiki_integration": upstream_wiki_integration,
        "import_report": import_summary,
    }


def clear_wikigraph_refresh_pending_after_success(
    root: Path,
    state_dir: Path,
    import_report_path: Path | None = None,
    reason: str = "refresh",
) -> dict[str, Any]:
    raise RuntimeError(WIKIGRAPH_REFRESH_LEDGER_WRITE_RETIRED_MESSAGE)


def record_wikigraph_refresh_failure(state_dir: Path, reason: str, log_path: str = "", message: str = "") -> dict[str, Any]:
    raise RuntimeError(WIKIGRAPH_REFRESH_LEDGER_WRITE_RETIRED_MESSAGE)
