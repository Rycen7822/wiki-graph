#!/usr/bin/env python3
"""Deterministic closeout wrapper for llm-wiki raw-fast paper notes.

The wrapper intentionally keeps the canonical raw-note verifier in the llm-wiki
Hermes skill. It orchestrates the existing verifier and ledgers so agents do not
forget pre-mark verification, pending marking, cleanup proof, or threshold-gated
wiki/native graph follow-through.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_WORKDIR = Path("/home/xu/project/wiki/wiki-graph")
DEFAULT_ROOT = DEFAULT_WORKDIR / "wiki_test"
DEFAULT_STATE_DIR = DEFAULT_WORKDIR / "tmp" / "wiki_test_state"
DEFAULT_VERIFIER = DEFAULT_WORKDIR / ".agents" / "skills" / "llm-wiki" / "scripts" / "raw_fast_note_verify.py"
DEFAULT_REQUIRED_SECTIONS = [
    "summary",
    "abstract",
    "methodology",
    "results",
    "future",
    "limitations",
    "questions",
]


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


class TimingRecorder:
    """Small JSON-serializable wall-clock timing collector for closeout reports."""

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self.started_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        self.steps: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _round_seconds(value: float) -> float:
        return round(max(value, 0.0), 6)

    def _entry_from_result(self, elapsed: float, result: Any) -> dict[str, Any]:
        entry: dict[str, Any] = {"elapsed_seconds": self._round_seconds(elapsed)}
        if isinstance(result, dict):
            for key in ["ok", "raw_fast_ok", "returncode", "command_returncode", "error", "message", "skipped", "skip_reason", "fast_final_verify"]:
                if key in result and result.get(key) is not None:
                    entry[key] = result.get(key)
        elif isinstance(result, list):
            entry["result_type"] = "list"
            entry["count"] = len(result)
            if all(isinstance(item, dict) and "ok" in item for item in result):
                entry["ok"] = all(bool(item.get("ok")) for item in result)
        else:
            entry["result_type"] = type(result).__name__
        return entry

    def record(self, name: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            self.steps[name] = {
                "elapsed_seconds": self._round_seconds(time.perf_counter() - started),
                "ok": False,
                "error": type(exc).__name__,
                "message": str(exc),
            }
            raise
        self.steps[name] = self._entry_from_result(time.perf_counter() - started, result)
        return result

    def snapshot(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "total_seconds": self._round_seconds(time.perf_counter() - self._start),
            "steps": self.steps,
        }


def parse_json_maybe(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return None


def run_json(command: list[str], *, cwd: Path | None = None, timeout: int = 7200) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(cwd) if cwd else None, timeout=timeout)
        payload = parse_json_maybe(completed.stdout)
        return {
            "returncode": completed.returncode,
            "json": payload,
            "stdout_tail": (completed.stdout or "")[-4000:],
            "stderr_tail": (completed.stderr or "")[-4000:],
            "command": command,
        }
    except subprocess.TimeoutExpired as exc:
        return {"returncode": 124, "json": None, "stdout_tail": (exc.stdout or "")[-4000:], "stderr_tail": (exc.stderr or "")[-4000:], "command": command, "error": "TimeoutExpired", "message": f"command exceeded {timeout}s"}
    except Exception as exc:  # pragma: no cover - subprocess environment failure
        return {"returncode": 1, "json": None, "stdout_tail": "", "stderr_tail": "", "command": command, "error": type(exc).__name__, "message": str(exc)}


def raw_fast_ok(report: Any) -> bool:
    return isinstance(report, dict) and bool(report.get("raw_fast_ok"))


def report_path_for(args: argparse.Namespace, label: str) -> Path:
    safe_raw = re.sub(r"[^0-9A-Za-z_.-]+", "_", args.raw_file).strip("_") or "raw_note"
    path = args.state_dir / "raw_fast_reports" / f"{safe_raw}_{label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_report(args: argparse.Namespace, label: str, report: dict[str, Any]) -> str:
    path = report_path_for(args, label)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def _load_json_file(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def capture_tmp_evidence_reports(args: argparse.Namespace) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for idx, tmp_path in enumerate(args.tmp):
        evidence_path = tmp_path / "evidence_bundle.json"
        payload = _load_json_file(evidence_path)
        if payload is None:
            continue
        files = payload.get("files") if isinstance(payload.get("files"), dict) else {}
        summary: dict[str, Any] = {
            "ok": True,
            "source_tmp": str(tmp_path),
            "evidence_bundle": str(evidence_path),
            "kind": payload.get("kind"),
            "source_url": payload.get("source_url"),
            "title_guess": payload.get("title_guess"),
            "warnings": payload.get("warnings") or [],
            "files": files,
            "timings": payload.get("timings") or {},
        }
        localized_rel = files.get("localized_figures") if isinstance(files, dict) else None
        if localized_rel:
            localized = _load_json_file(tmp_path / str(localized_rel))
            if localized is not None:
                summary["localized_figures"] = {
                    "entry_count": localized.get("entry_count"),
                    "refused_count": localized.get("refused_count"),
                }
        digest_rel = files.get("paper_digest") if isinstance(files, dict) else None
        if digest_rel:
            digest = _load_json_file(tmp_path / str(digest_rel))
            if digest is not None:
                summary["paper_digest"] = {
                    "equation_count": len(digest.get("equation_cards") or []),
                    "table_count": len(digest.get("table_cards") or []),
                    "figure_count": len(digest.get("figure_cards") or []),
                    "warning_count": len(digest.get("quality_warnings") or []),
                }
        label = "evidence_bundle_summary" if idx == 0 else f"evidence_bundle_summary_{idx + 1}"
        report_path = report_path_for(args, label)
        summary["report_path"] = str(report_path)
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summaries.append(summary)
    return {"ok": True, "count": len(summaries), "summaries": summaries}


def verify_note(args: argparse.Namespace, tmp_paths: list[Path] | None = None, label: str = "verify") -> dict[str, Any]:
    command = [
        sys.executable,
        str(args.verifier),
        "--wiki",
        str(args.root),
        "--raw-file",
        args.raw_file,
        "--structured-paper",
    ]
    patterns = [p for p in args.pattern if p]
    if patterns:
        command.append("--patterns")
        command.extend(patterns)
    if tmp_paths:
        command.append("--tmp")
        command.extend(str(p) for p in tmp_paths)
    result = run_json(command, cwd=args.workdir, timeout=args.timeout)
    report = result.get("json") if isinstance(result.get("json"), dict) else {}
    report = {"command_returncode": result["returncode"], **report, "stderr_tail": result.get("stderr_tail", "")}
    if result.get("error"):
        report["runner_error"] = {"error": result.get("error"), "message": result.get("message")}
    report["report_path"] = write_report(args, label, report)
    return report


def fast_final_verify_from_pre(args: argparse.Namespace, pre_verify: dict[str, Any], tmp_paths: list[Path] | None = None, label: str = "final_verify") -> dict[str, Any]:
    """Avoid a second full raw-note duplicate/image scan after safe cleanup.

    Raw-fast closeout mutates only external ledgers between `pre_verify` and
    cleanup. When `pre_verify` already passed, the only new final-closeout fact
    is declared temp-path absence. This keeps the durable final report shape
    compatible with the canonical verifier while avoiding a second whole-raw
    duplicate/image scan on the hot path.
    """
    tmp_paths = tmp_paths or []
    tmp_absent = {str(path): not Path(path).exists() for path in tmp_paths}
    report = dict(pre_verify)
    report["fast_final_verify"] = True
    report["tmp_absent"] = tmp_absent
    report["cleanup_verified_by"] = "safe_cleanup+tmp_absent"
    report["raw_fast_ok"] = raw_fast_ok(pre_verify) and all(tmp_absent.values())
    report["report_path"] = write_report(args, label, report)
    return report


def control_scan(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    disallowed: list[dict[str, Any]] = []
    tabs = 0
    for idx, ch in enumerate(text):
        code = ord(ch)
        if ch == "\t":
            tabs += 1
            continue
        if ch in {"\n", "\r"}:
            continue
        if code < 32 or code == 127:
            line = text.count("\n", 0, idx) + 1
            col = idx - text.rfind("\n", 0, idx)
            disallowed.append({"line": line, "col": col, "codepoint": code})
    return {"control_count": len(disallowed), "tabs": tabs, "control_hits": disallowed[:50]}


def cleanup_preflight(tmp_paths: list[Path], *, root: Path, workdir: Path, allow_non_tmp_cleanup: bool = False) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    root_resolved = root.resolve()
    workdir_resolved = workdir.resolve()
    home_resolved = Path.home().resolve()
    protected = {root_resolved, workdir_resolved, home_resolved, Path("/").resolve(), Path("/tmp").resolve(), Path("/home").resolve()}
    for path in tmp_paths:
        resolved = path.resolve(strict=False)
        entry: dict[str, Any] = {"path": str(path), "resolved": str(resolved), "exists": path.exists()}
        if not path.exists():
            entry.update({"ok": True, "reason": "already_absent"})
        elif resolved in protected:
            entry.update({"ok": False, "reason": "refused_protected_path"})
        elif resolved.is_relative_to(root_resolved):
            entry.update({"ok": False, "reason": "refused_inside_wiki_root"})
        elif resolved.is_relative_to(workdir_resolved):
            entry.update({"ok": False, "reason": "refused_inside_workdir"})
        elif not str(resolved).startswith("/tmp/") and not allow_non_tmp_cleanup:
            entry.update({"ok": False, "reason": "refused_non_tmp_without_override"})
        elif resolved.is_relative_to(home_resolved) and any(part in {".ssh", ".gnupg", ".hermes"} for part in resolved.parts):
            entry.update({"ok": False, "reason": "refused_sensitive_home_path"})
        else:
            entry.update({"ok": True, "reason": "cleanup_allowed"})
        results.append(entry)
    return results


def safe_cleanup(tmp_paths: list[Path], *, root: Path, workdir: Path, allow_non_tmp_cleanup: bool = False) -> list[dict[str, Any]]:
    preflight = cleanup_preflight(tmp_paths, root=root, workdir=workdir, allow_non_tmp_cleanup=allow_non_tmp_cleanup)
    results: list[dict[str, Any]] = []
    for entry, path in zip(preflight, tmp_paths):
        item = dict(entry)
        item["existed_before"] = path.exists()
        if not entry.get("ok") or not path.exists():
            item.update({"removed": False})
        else:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                item.update({"removed": True})
            except Exception as exc:  # pragma: no cover - filesystem race guard
                item.update({"removed": False, "ok": False, "error": type(exc).__name__, "message": str(exc)})
        item["exists_after"] = path.exists()
        results.append(item)
    return results


def compact_mark_payload(mark_result: dict[str, Any]) -> dict[str, Any]:
    payload = mark_result.get("json") if isinstance(mark_result.get("json"), dict) else {}
    marked = payload.get("marked") if isinstance(payload, dict) else None
    status = payload.get("status_after_mark") if isinstance(payload, dict) else payload
    auto = payload.get("auto_integrate") if isinstance(payload, dict) else None
    compact: dict[str, Any] = {
        "returncode": mark_result.get("returncode"),
        "marked": marked,
        "status_after_mark": compact_wiki_status(status if isinstance(status, dict) else {}),
    }
    if isinstance(auto, dict):
        compact["auto_integrate"] = compact_auto_integrate(auto)
    if mark_result.get("stderr_tail"):
        compact["stderr_tail"] = mark_result["stderr_tail"]
    return compact


def compact_wiki_status(status: dict[str, Any]) -> dict[str, Any]:
    return {
        key: status.get(key)
        for key in [
            "pending_count",
            "actionable_pending_count",
            "review_pending_count",
            "terminal_pending_count",
            "threshold",
            "should_integrate",
            "should_review",
            "next_required_action",
            "dirty",
        ]
        if key in status
    }


def compact_auto_integrate(auto: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: auto.get(key)
        for key in [
            "dry_run",
            "would_run",
            "ran",
            "skipped",
            "skip_reason",
            "runner_returncode",
            "prompt_path",
        ]
        if key in auto
    }
    if isinstance(auto.get("pre_status"), dict):
        compact["pre_status"] = compact_wiki_status(auto["pre_status"])
    if isinstance(auto.get("post_status"), dict):
        compact["post_status"] = compact_wiki_status(auto["post_status"])
    if auto.get("failure"):
        compact["failure"] = auto.get("failure")
    return compact


def compact_native_refresh_status(status: dict[str, Any]) -> dict[str, Any]:
    return {
        key: status.get(key)
        for key in [
            "pending_count",
            "graph_ready_pending_count",
            "standalone_native_pending_count",
            "standalone_native_should_refresh",
            "standalone_native_ledger_path",
            "standalone_native_command_returncode",
            "standalone_native_error",
            "raw_fast_pending_wiki_integration_count",
            "raw_fast_actionable_wiki_integration_count",
            "raw_fast_review_wiki_integration_count",
            "threshold",
            "should_refresh",
            "blocked_by_pending_wiki_integration",
            "next_required_action",
        ]
        if key in status
    }


def final_verify_acceptable(pre_verify: dict[str, Any], final_verify: dict[str, Any]) -> bool:
    tmp_absent = final_verify.get("tmp_absent")
    if isinstance(tmp_absent, dict) and not all(tmp_absent.values()):
        return False
    if raw_fast_ok(final_verify):
        return True
    blockers = {str(item) for item in final_verify.get("raw_fast_blockers") or []}
    return blockers == {"non_raw_wiki_hits"} and bool(final_verify.get("non_raw_wiki_hits")) and not bool(pre_verify.get("non_raw_wiki_hits"))


def run_mark_pending(args: argparse.Namespace) -> dict[str, Any]:
    script = args.workdir / "scripts" / "batch_wiki_integration.py"
    command = [
        sys.executable,
        str(script),
        "mark-pending",
        "--root",
        str(args.root),
        "--state-dir",
        str(args.state_dir),
        "--raw-path",
        args.raw_file,
        "--title",
        args.title,
        "--source-id",
        args.source_id,
        "--resource-status-summary",
        args.resource_status_summary,
    ]
    for hint in args.topic_hint:
        command.extend(["--topic-hint", hint])
    required = args.required_section or DEFAULT_REQUIRED_SECTIONS
    for section in required:
        command.extend(["--required-section", section])
    if args.threshold is not None:
        command.extend(["--threshold", str(args.threshold)])
    if args.auto_integrate:
        command.append("--auto-integrate")
        command.extend(["--auto-integrate-timeout", str(args.integration_timeout)])
        if args.auto_integrate_dry_run:
            command.append("--auto-integrate-dry-run")
        if args.integration_command:
            command.extend(["--integration-command", args.integration_command])
    return run_json(command, cwd=args.workdir, timeout=args.integration_timeout + 30)


def run_wiki_status(args: argparse.Namespace) -> dict[str, Any]:
    script = args.workdir / "scripts" / "batch_wiki_integration.py"
    command = [sys.executable, str(script), "status", "--root", str(args.root), "--state-dir", str(args.state_dir), "--reason", "threshold"]
    if args.threshold is not None:
        command.extend(["--threshold", str(args.threshold)])
    result = run_json(command, cwd=args.workdir, timeout=args.timeout)
    return result.get("json") if isinstance(result.get("json"), dict) else {"error": "status_json_missing", "returncode": result.get("returncode")}


def run_native_refresh_status(args: argparse.Namespace, *, migrate_legacy: bool = True) -> dict[str, Any]:
    script = args.workdir / "scripts" / "batch_native_refresh.py"
    command = [sys.executable, str(script), "status", "--root", str(args.root), "--state-dir", str(args.state_dir), "--workdir", str(args.workdir)]
    if not migrate_legacy:
        command.append("--no-migrate-legacy")
    result = run_json(command, cwd=args.workdir, timeout=args.timeout)
    payload = result.get("json") if isinstance(result.get("json"), dict) else {"error": "status_json_missing", "returncode": result.get("returncode")}
    payload["command_returncode"] = result.get("returncode")
    return payload


def run_native_refresh_if_needed(args: argparse.Namespace, status: dict[str, Any]) -> dict[str, Any]:
    compact_status = compact_native_refresh_status(status)
    if status.get("command_returncode") not in (None, 0):
        return {"ok": False, "ran": False, "returncode": status.get("command_returncode"), "error": status.get("error") or "native_refresh_status_failed", **compact_status}
    if status.get("blocked_by_pending_wiki_integration") or not status.get("should_refresh"):
        return {"ok": True, "ran": False, "skipped": True, "skip_reason": "not_runnable_or_not_required", **compact_status}
    script = args.workdir / "scripts" / "batch_native_refresh.py"
    command = [sys.executable, str(script), "refresh", "--prepare-only", "--root", str(args.root), "--state-dir", str(args.state_dir), "--workdir", str(args.workdir)]
    result = run_json(command, cwd=args.workdir, timeout=args.refresh_timeout)
    payload = result.get("json") if isinstance(result.get("json"), dict) else {}
    refresh_status = payload.get("status") or payload.get("status_before") or status
    ok = result.get("returncode") == 0 and not payload.get("error")
    build = payload.get("build") if isinstance(payload.get("build"), dict) else {}
    return {
        "ok": ok,
        "ran": ok and not bool(payload.get("skipped")),
        "returncode": result.get("returncode"),
        "status": compact_native_refresh_status(refresh_status if isinstance(refresh_status, dict) else status),
        "skipped": payload.get("skipped"),
        "prepared_only": payload.get("prepared_only"),
        "build_ok": build.get("ok"),
        "error": payload.get("error") or result.get("error"),
        "message": payload.get("message") or result.get("message"),
        "stderr_tail": result.get("stderr_tail") or None,
    }


def compact_standalone_native_refresh_status(status: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(status, dict) or not status:
        return {}
    compact: dict[str, Any] = {
        "standalone_native_command_returncode": status.get("command_returncode"),
    }
    if "pending_count" in status:
        compact["standalone_native_pending_count"] = status.get("pending_count")
    if "should_refresh" in status:
        compact["standalone_native_should_refresh"] = status.get("should_refresh")
    if "ledger_path" in status:
        compact["standalone_native_ledger_path"] = status.get("ledger_path")
    if status.get("error"):
        compact["standalone_native_error"] = status.get("error")
    return compact


def synthesize_blocked_native_refresh_status(
    args: argparse.Namespace,
    wiki_status: dict[str, Any],
    *,
    standalone_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return threshold native refresh status implied by raw-fast pending wiki integration.

    When actionable/review raw-fast notes are still pending wiki integration,
    native refresh cannot run. Calling `batch_native_refresh.py status` in
    that state repeats a whole-wiki freshness scan only to report the same
    blocker, so closeout can synthesize the compact blocked status.
    """
    actionable = int(wiki_status.get("actionable_pending_count") or 0)
    review = int(wiki_status.get("review_pending_count") or 0)
    blocking = int(wiki_status.get("blocking_pending_count") or (actionable + review))
    if not blocking:
        return {}
    blocked_reasons: list[str] = []
    if actionable:
        blocked_reasons.append("pending_wiki_integration_before_native_refresh")
    if review:
        blocked_reasons.append("pending_wiki_integration_needs_manual_review")
    status = {
        "reason": "threshold",
        "should_refresh": False,
        "would_refresh_if_unblocked": False,
        "blocked_by_pending_wiki_integration": True,
        "blocked_reasons": sorted(set(blocked_reasons)),
        "next_required_action": "manual_review" if review and not actionable else "wiki_integration",
        "reasons": [],
        "pending_count": 0,
        "graph_ready_pending_count": 0,
        "raw_fast_pending_wiki_integration_count": blocking,
        "raw_fast_actionable_wiki_integration_count": actionable,
        "raw_fast_review_wiki_integration_count": review,
        "total_not_graph_fresh_count": blocking,
        "pending_count_excludes_raw_fast": True,
        "threshold": args.refresh_threshold or 10,
        "dirty": False,
        "raw_clip_count": wiki_status.get("raw_clip_count"),
        "skipped": True,
        "skip_reason": "blocked_by_pending_wiki_integration_from_wiki_status",
        "command_returncode": 0,
    }
    status.update(compact_standalone_native_refresh_status(standalone_status))
    return status


def short_text(value: str, limit: int = 220) -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


def build_compact_log_entry(args: argparse.Namespace, output: dict[str, Any]) -> str:
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    title = short_text(args.title, 160)
    source = short_text(args.source_id, 180)
    resource = short_text(args.resource_status_summary or "see closeout artifacts", 220)
    resource_sentence = resource if resource.endswith((".", "!", "?", "。", "！", "？")) else f"{resource}."
    final_verify = output.get("final_verify") if isinstance(output.get("final_verify"), dict) else {}
    wiki = output.get("wiki_integration") if isinstance(output.get("wiki_integration"), dict) else {}
    native_refresh = output.get("native_refresh_status") if isinstance(output.get("native_refresh_status"), dict) else {}
    report_path = final_verify.get("report_path") or output.get("report_path") or ""
    standalone_native = ""
    if "standalone_native_pending_count" in native_refresh or "standalone_native_should_refresh" in native_refresh:
        standalone_native = (
            f"; standalone native ledger pending `{native_refresh.get('standalone_native_pending_count')}`, "
            f"`should_refresh={str(bool(native_refresh.get('standalone_native_should_refresh'))).lower()}`"
        )
    lines = [
        f"## [{stamp}] raw-fast ingest | {title}",
        "",
        f"- Saved `{args.raw_file}` from `{source}`; closeout `raw_fast_ok={str(bool(output.get('raw_fast_ok'))).lower()}`; final report `{report_path}`.",
        f"- Resource status: {resource_sentence}",
        f"- Queue: wiki pending/actionable `{wiki.get('pending_count')}/{wiki.get('actionable_pending_count')}` of threshold `{wiki.get('threshold')}`, `should_integrate={str(bool(wiki.get('should_integrate'))).lower()}`, next `{wiki.get('next_required_action')}`; native graph `blocked_by_pending_wiki_integration={str(bool(native_refresh.get('blocked_by_pending_wiki_integration'))).lower()}`, graph-ready pending `{native_refresh.get('graph_ready_pending_count')}`, `should_refresh={str(bool(native_refresh.get('should_refresh'))).lower()}`{standalone_native}.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def append_compact_log(args: argparse.Namespace, output: dict[str, Any]) -> dict[str, Any]:
    log_path = args.root / "log.md"
    entry = build_compact_log_entry(args, output)
    if log_path.exists():
        existing = log_path.read_text(encoding="utf-8", errors="replace")
    else:
        existing = "# llm-wiki log\n"
    if args.raw_file in existing:
        return {"ok": True, "appended": False, "skip_reason": "raw_file_already_in_log", "log_path": str(log_path), "entry": entry}
    sep = "\n\n" if existing and not existing.endswith("\n\n") else ""
    log_path.write_text(existing.rstrip() + sep + entry, encoding="utf-8")
    return {"ok": True, "appended": True, "log_path": str(log_path), "entry": entry}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Close out one llm-wiki raw-fast raw note")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--verifier", type=Path, default=DEFAULT_VERIFIER)
    parser.add_argument("--raw-file", required=True, help="Raw note path relative to the wiki root")
    parser.add_argument("--title", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--pattern", action="append", default=[])
    parser.add_argument("--topic-hint", action="append", default=[])
    parser.add_argument("--required-section", action="append", default=[])
    parser.add_argument("--resource-status-summary", default="")
    parser.add_argument("--tmp", action="append", default=[], type=Path)
    parser.add_argument("--allow-non-tmp-cleanup", action="store_true", help="Permit cleanup outside /tmp after protection checks; default refuses non-/tmp paths")
    parser.add_argument("--threshold", type=int, default=None, help="Override pending wiki-integration threshold")
    parser.add_argument("--refresh-threshold", type=int, default=None, help="Compatibility option retained for legacy callers; native refresh status has no threshold")
    parser.add_argument("--auto-integrate", action="store_true", help="Launch wiki integration automatically when threshold says it should run")
    parser.add_argument("--auto-integrate-dry-run", action="store_true")
    parser.add_argument("--integration-command", default=None)
    parser.add_argument("--fast-final-verify", action="store_true", help="After successful pre-verify and cleanup, synthesize the final verifier report from pre-verify plus declared temp-path absence instead of rescanning all raw notes")
    parser.add_argument("--append-log", action="store_true", help="Append an idempotent compact raw-fast entry to root/log.md after successful closeout; no post-log wiki validation is run")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--integration-timeout", type=int, default=7200)
    parser.add_argument("--refresh-timeout", type=int, default=7200)
    return parser.parse_args()


def fail(stage: str, payload: dict[str, Any], code: int = 1, timings: TimingRecorder | None = None) -> int:
    output = {"ok": False, "stage": stage, **payload}
    if timings is not None:
        output["timings"] = timings.snapshot()
    print_json(output)
    return code


def main() -> int:
    timings = TimingRecorder()
    args = parse_args()
    args.root = args.root.resolve()
    args.state_dir = args.state_dir.resolve()
    args.workdir = args.workdir.resolve()
    args.verifier = args.verifier.resolve()
    if Path(args.raw_file).is_absolute() or ".." in Path(args.raw_file).parts or not args.raw_file.startswith("raw/clip/"):
        return fail("preflight", {"raw_fast_ok": False, "error": "invalid_raw_file", "raw_file": args.raw_file}, 1, timings)
    raw_note = args.root / args.raw_file

    pre_verify = timings.record("pre_verify", verify_note, args, label="pre_verify")
    if not raw_fast_ok(pre_verify):
        return fail("pre_verify", {"raw_fast_ok": False, "pre_verify": pre_verify}, 1, timings)

    scan = timings.record("control_scan", control_scan, raw_note)
    if scan["control_count"]:
        return fail("control_scan", {"raw_fast_ok": False, "pre_verify": pre_verify, "control_scan": scan}, 1, timings)

    cleanup_check = timings.record("cleanup_preflight", cleanup_preflight, args.tmp, root=args.root, workdir=args.workdir, allow_non_tmp_cleanup=args.allow_non_tmp_cleanup)
    if any(not item.get("ok") for item in cleanup_check):
        return fail("cleanup_preflight", {"raw_fast_ok": False, "pre_verify": pre_verify, "control_scan": scan, "cleanup_preflight": cleanup_check}, 1, timings)

    evidence_reports = timings.record("evidence_report_capture", capture_tmp_evidence_reports, args)

    mark_result = timings.record("mark_pending", run_mark_pending, args)
    marked_payload = compact_mark_payload(mark_result)
    if mark_result.get("returncode") != 0:
        return fail("mark_pending", {"raw_fast_ok": False, "pre_verify": pre_verify, "control_scan": scan, "mark_pending": marked_payload}, int(mark_result.get("returncode") or 1), timings)

    cleanup = timings.record("cleanup", safe_cleanup, args.tmp, root=args.root, workdir=args.workdir, allow_non_tmp_cleanup=args.allow_non_tmp_cleanup)
    if any(not item.get("ok") for item in cleanup):
        return fail("cleanup", {"raw_fast_ok": False, "pre_verify": pre_verify, "control_scan": scan, "marked": marked_payload.get("marked"), "cleanup": cleanup}, 1, timings)

    if args.fast_final_verify:
        final_verify = timings.record("final_verify", fast_final_verify_from_pre, args, pre_verify, args.tmp, label="final_verify")
    else:
        final_verify = timings.record("final_verify", verify_note, args, tmp_paths=args.tmp, label="final_verify") if args.tmp else timings.record("final_verify", verify_note, args, label="final_verify")
    if not final_verify_acceptable(pre_verify, final_verify):
        return fail(
            "final_verify",
            {"raw_fast_ok": False, "pre_verify": pre_verify, "control_scan": scan, "marked": marked_payload.get("marked"), "cleanup": cleanup, "final_verify": final_verify},
            1,
            timings,
        )

    wiki_status = timings.record("wiki_status", run_wiki_status, args)
    synthesized_native = synthesize_blocked_native_refresh_status(args, wiki_status)
    if synthesized_native:
        standalone_native_status = timings.record("standalone_native_refresh_status", run_native_refresh_status, args, migrate_legacy=False)
        native_refresh_status = timings.record(
            "native_refresh_status",
            synthesize_blocked_native_refresh_status,
            args,
            wiki_status,
            standalone_status=standalone_native_status,
        )
    else:
        native_refresh_status = timings.record("native_refresh_status", run_native_refresh_status, args)
    native_refresh = timings.record("native_refresh", run_native_refresh_if_needed, args, native_refresh_status)
    compact_native_status = compact_native_refresh_status(native_refresh_status)
    if not native_refresh.get("ok", True):
        return fail(
            "native_refresh",
            {
                "raw_fast_ok": False,
                "pre_verify": pre_verify,
                "control_scan": scan,
                "marked": marked_payload.get("marked"),
                "cleanup": cleanup,
                "final_verify": final_verify,
                "wiki_integration": compact_wiki_status(wiki_status),
                "native_refresh_status": compact_native_status,
                "native_refresh": native_refresh,
            },
            int(native_refresh.get("returncode") or 1),
            timings,
        )

    output = {
        "ok": True,
        "stage": "complete",
        "raw_fast_ok": True,
        "raw_file": args.raw_file,
        "pre_verify": pre_verify,
        "control_scan": scan,
        "marked": marked_payload.get("marked"),
        "mark_pending": marked_payload,
        "cleanup": cleanup,
        "evidence_reports": evidence_reports,
        "final_verify": final_verify,
        "wiki_integration": compact_wiki_status(wiki_status),
        "native_refresh_status": compact_native_status,
        "native_refresh": native_refresh,
        "timings": timings.snapshot(),
    }
    if args.append_log:
        log_append = timings.record("append_log", append_compact_log, args, output)
        output["log_append"] = {key: value for key, value in log_append.items() if key != "entry"}
        output["compact_log_entry"] = log_append.get("entry")
        output["timings"] = timings.snapshot()
    else:
        output["compact_log_entry"] = build_compact_log_entry(args, output)

    print_json(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
