#!/usr/bin/env python3
"""External pending ledger and runner hook for raw-fast llm-wiki integration.

Raw-fast clipping saves canonical raw notes first, then queues `_meta`/compiled/log integration until the threshold batch or a query/integration preflight needs the pending notes. `status` and `should-integrate` are read-only decision surfaces; `auto-integrate` writes a self-contained Hermes prompt and launches either the default Hermes CLI runner or a configured command. The runner is considered successful only if it performs Markdown/wiki integration, validation, and `clear-success`, so a no-op external runner cannot silently clear or bypass the ledger.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from wiki_native_lib import (
    DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD,
    DEFAULT_STATE_DIR,
    DEFAULT_WIKI_ROOT,
    clear_pending_wiki_integration_after_success,
    mark_pending_wiki_integration,
    pending_wiki_integration_status,
    print_json,
    record_pending_wiki_integration_failure,
)


DEFAULT_NATIVE_WORKDIR = Path(__file__).resolve().parents[1]


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=DEFAULT_WIKI_ROOT)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)


def _tail_text(text: str, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def build_auto_integration_prompt(root: Path, state_dir: Path, status: dict[str, Any], reason: str, workdir: Path | None = None) -> str:
    native_workdir = workdir or DEFAULT_NATIVE_WORKDIR
    wiki_integration_script = native_workdir / "scripts" / "batch_wiki_integration.py"
    native_refresh_script = native_workdir / "scripts" / "batch_native_refresh.py"
    pending = status.get("actionable_pending") or []
    pending_lines = []
    for idx, item in enumerate(pending, 1):
        if not isinstance(item, dict):
            continue
        pending_lines.append(
            f"{idx}. `{item.get('raw_path', '')}` — {item.get('title', '')} — source: {item.get('source_id', '')} — hints: {', '.join(str(h) for h in (item.get('topic_hints') or []))}"
        )
    pending_block = "\n".join(pending_lines) or "(no actionable pending items)"
    return f"""You are an autonomous Hermes sub-run launched by `batch_wiki_integration.py auto-integrate` because the llm-wiki raw-fast batch wiki integration threshold was reached.

Task: perform the batch wiki integration for the pending raw-fast notes, then clear the pending wiki-integration ledger only after Markdown/wiki validation passes.

Context:
- Wiki root: `{root}`
- Native refresh workdir: `{native_workdir}`
- State dir: `{state_dir}`
- Pending ledger: `{state_dir / 'pending_wiki_integration.json'}`
- Trigger reason: `{reason}`
- Pending threshold status: pending_count={status.get('pending_count')}, actionable_pending_count={status.get('actionable_pending_count')}, threshold={status.get('threshold')}

Pending actionable raw notes:
{pending_block}

Required workflow:
1. Load and follow the `llm-wiki` skill, especially `references/raw-fast-batch-wiki-integration.md` and `references/wiki-core-operations.md`.
2. Read `SCHEMA.md`, `index.md`, recent `log.md`, `_meta/raw-clip-map.md`, and `_meta/topic-map.md` before editing.
3. Read every pending raw note listed above; reconcile duplicates or review blockers conservatively.
4. Batch-update the Markdown/wiki layer: `_meta/raw-clip-map.md`, `_meta/topic-map.md`, relevant compiled pages where the notes meet page/update thresholds, `index.md` if compiled pages were added/removed, and `log.md` with one batch entry. Do not mechanically create one compiled page per raw note.
5. Run the appropriate wiki validation checks. Do not put generated machine artifacts under the human wiki root.
6. Only after validation passes, run:
   `python {wiki_integration_script} clear-success --root {root} --state-dir {state_dir} --reason {reason}`
   This carries integrated raw notes into the native graph pending ledger.
7. Run `python {native_refresh_script} status --root {root} --state-dir {state_dir} --workdir {native_workdir}` and report whether native graph refresh is now pending.

Important constraints:
- Do not ask the user questions; make reasonable conservative integration choices.
- Do not clear pending items before wiki/meta/log edits and validation are complete.
- Do not run a full native refresh unless explicitly required by the current prompt; a status check is enough for this auto-integration closeout.
- Do not expose secrets or tokens. If any appear in sources or logs, redact as `[REDACTED]`.
"""


def write_auto_integration_prompt(state_dir: Path, prompt: str) -> Path:
    run_dir = state_dir / "wiki_integration_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_auto_integration_prompt.md"
    path.write_text(prompt, encoding="utf-8")
    return path


def build_runner_command(integration_command: str | None, prompt: str, prompt_path: Path, root: Path, state_dir: Path, reason: str) -> list[str]:
    if integration_command:
        replacements = {
            "{prompt_path}": str(prompt_path),
            "{root}": str(root),
            "{state_dir}": str(state_dir),
            "{ledger_path}": str(state_dir / "pending_wiki_integration.json"),
            "{reason}": reason,
        }
        command = []
        for token in shlex.split(integration_command):
            for key, value in replacements.items():
                token = token.replace(key, value)
            command.append(token)
        return command
    return ["hermes", "-z", prompt, "--skills", "llm-wiki"]


def run_auto_integration(
    root: Path,
    state_dir: Path,
    reason: str,
    threshold: int | None = None,
    dry_run: bool = False,
    integration_command: str | None = None,
    timeout: int = 7200,
) -> tuple[int, dict[str, Any]]:
    pre_status = pending_wiki_integration_status(root, state_dir, reason=reason, threshold=threshold)
    prompt = build_auto_integration_prompt(root, state_dir, pre_status, reason)
    prompt_path = write_auto_integration_prompt(state_dir, prompt)
    command = build_runner_command(integration_command, prompt, prompt_path, root, state_dir, reason)
    redacted_command = ["<prompt>" if part == prompt else part for part in command]
    base: dict[str, Any] = {
        "dry_run": dry_run,
        "would_run": bool(pre_status.get("should_integrate")) and not bool(pre_status.get("should_review")),
        "ran": False,
        "pre_status": pre_status,
        "prompt_path": str(prompt_path),
        "command": redacted_command,
    }
    if pre_status.get("should_review"):
        return 11, {**base, "skipped": True, "skip_reason": "manual_review_required"}
    if not pre_status.get("should_integrate"):
        return 0, {**base, "skipped": True, "skip_reason": "integration_not_required"}
    if dry_run:
        return 0, {**base, "skipped": False}

    env = os.environ.copy()
    env.update(
        {
            "LLM_WIKI_ROOT": str(root),
            "LLM_WIKI_STATE_DIR": str(state_dir),
            "LLM_WIKI_LEDGER": str(state_dir / "pending_wiki_integration.json"),
            "LLM_WIKI_INTEGRATION_PROMPT": str(prompt_path),
            "LLM_WIKI_INTEGRATION_REASON": reason,
        }
    )
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=timeout)
    post_status = pending_wiki_integration_status(root, state_dir, reason=reason, threshold=threshold)
    result = {
        **base,
        "ran": True,
        "runner_returncode": completed.returncode,
        "runner_stdout_tail": _tail_text(completed.stdout or ""),
        "runner_stderr_tail": _tail_text(completed.stderr or ""),
        "post_status": post_status,
    }
    if completed.returncode != 0:
        failure = record_pending_wiki_integration_failure(state_dir, "auto-integrate-runner-failed", f"runner exited {completed.returncode}")
        return completed.returncode or 13, {**result, "failure": failure}
    if post_status.get("should_integrate") or post_status.get("should_review"):
        failure = record_pending_wiki_integration_failure(state_dir, "auto-integrate-incomplete", "runner returned successfully but pending wiki integration ledger still requires action")
        return 12, {**result, "failure": failure}
    return 0, result


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage raw-fast llm-wiki pending integration batches")
    sub = parser.add_subparsers(dest="command", required=True)

    status_parser = sub.add_parser("status", help="Show pending wiki-integration state and threshold decision")
    add_common_paths(status_parser)
    status_parser.add_argument("--reason", default="threshold", choices=["threshold", "pre-query", "query", "manual", "integrate", "wiki-query"])
    status_parser.add_argument("--threshold", type=int, default=None, help="Override ledger threshold for this status check")

    mark_parser = sub.add_parser("mark-pending", help="Mark one raw-fast raw note as pending for batch wiki integration")
    add_common_paths(mark_parser)
    mark_parser.add_argument("--raw-path", default="")
    mark_parser.add_argument("--title", default="")
    mark_parser.add_argument("--source-id", default="")
    mark_parser.add_argument("--topic-hint", action="append", default=[])
    mark_parser.add_argument("--required-section", action="append", default=[])
    mark_parser.add_argument("--resource-status-summary", default="")
    mark_parser.add_argument("--status", default="raw_saved", choices=["raw_saved", "needs_review", "failed", "skipped_duplicate"])
    mark_parser.add_argument("--threshold", type=int, default=None, help=f"Set/override pending batch threshold for this ledger (default {DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD} for new ledgers)")
    mark_parser.add_argument("--auto-integrate", action="store_true", help="After marking, immediately launch the configured wiki-integration runner if the threshold/preflight says integration is required")
    mark_parser.add_argument("--auto-integrate-dry-run", action="store_true", help="With --auto-integrate, print the generated runner prompt/command without launching it")
    mark_parser.add_argument("--integration-command", default=None, help="Optional command template for auto integration; placeholders: {prompt_path}, {root}, {state_dir}, {ledger_path}, {reason}. Defaults to Hermes CLI.")
    mark_parser.add_argument("--auto-integrate-timeout", type=int, default=7200)

    auto_parser = sub.add_parser("auto-integrate", help="Launch the configured wiki-integration runner when pending raw-fast notes require integration")
    add_common_paths(auto_parser)
    auto_parser.add_argument("--reason", default="threshold", choices=["threshold", "pre-query", "query", "manual", "integrate", "wiki-query"])
    auto_parser.add_argument("--threshold", type=int, default=None, help="Override ledger threshold for this decision")
    auto_parser.add_argument("--dry-run", action="store_true", help="Write the prompt and print the command without launching the runner")
    auto_parser.add_argument("--integration-command", default=None, help="Optional command template; placeholders: {prompt_path}, {root}, {state_dir}, {ledger_path}, {reason}. Defaults to Hermes CLI.")
    auto_parser.add_argument("--timeout", type=int, default=7200)

    should_parser = sub.add_parser("should-integrate", help="Return whether batch wiki integration should run for this reason")
    add_common_paths(should_parser)
    should_parser.add_argument("--reason", default="threshold", choices=["threshold", "pre-query", "query", "manual", "integrate", "wiki-query"])
    should_parser.add_argument("--threshold", type=int, default=None, help="Override ledger threshold for this decision")
    should_parser.add_argument("--exit-code", action="store_true", help="Exit 10 when integration is needed, 11 when manual review is needed, 0 when no action is needed")

    clear_parser = sub.add_parser("clear-success", help="Clear pending wiki-integration ledger after successful batch integration and validation")
    add_common_paths(clear_parser)
    clear_parser.add_argument("--reason", default="external-success")
    clear_parser.add_argument("--integrated-path", action="append", default=[])
    clear_parser.add_argument(
        "--no-mark-native-pending",
        dest="no_mark_native_pending",
        action="store_true",
        help="Clear only; do not carry integrated raw notes into the native graph pending ledger",
    )

    args = parser.parse_args()
    root = args.root.resolve()
    state_dir = args.state_dir.resolve()

    if args.command == "status":
        print_json(pending_wiki_integration_status(root, state_dir, reason=args.reason, threshold=args.threshold))
        return 0
    if args.command == "mark-pending":
        entry = mark_pending_wiki_integration(
            state_dir,
            root,
            raw_path=args.raw_path,
            title=args.title,
            source_id=args.source_id,
            topic_hints=args.topic_hint or None,
            required_sections=args.required_section or None,
            resource_status_summary=args.resource_status_summary,
            status=args.status,
            threshold=args.threshold,
        )
        status = pending_wiki_integration_status(root, state_dir, reason="threshold", threshold=args.threshold)
        if args.auto_integrate:
            code, auto_result = run_auto_integration(
                root,
                state_dir,
                reason="threshold",
                threshold=args.threshold,
                dry_run=args.auto_integrate_dry_run,
                integration_command=args.integration_command,
                timeout=args.auto_integrate_timeout,
            )
            print_json({"marked": entry, "status_after_mark": status, "auto_integrate": auto_result})
            return code
        print_json({"marked": entry, **status})
        return 0
    if args.command == "auto-integrate":
        code, result = run_auto_integration(
            root,
            state_dir,
            reason=args.reason,
            threshold=args.threshold,
            dry_run=args.dry_run,
            integration_command=args.integration_command,
            timeout=args.timeout,
        )
        print_json(result)
        return code
    if args.command == "should-integrate":
        status = pending_wiki_integration_status(root, state_dir, reason=args.reason, threshold=args.threshold)
        print_json(status)
        if args.exit_code:
            if status["should_integrate"]:
                return 10
            if status.get("should_review"):
                return 11
        return 0
    if args.command == "clear-success":
        print_json(clear_pending_wiki_integration_after_success(root, state_dir, integrated_paths=args.integrated_path, reason=args.reason, mark_native_pending=not args.no_mark_native_pending))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
