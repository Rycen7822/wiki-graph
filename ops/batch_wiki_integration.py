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

from ops.wiki_native_cli import DEFAULT_STATE_DIR, DEFAULT_WIKI_ROOT, print_json
from ops.wiki_native_wiki_integration_bridge import clear_pending_wiki_integration_after_success
from ops.wiki_native_wiki_integration_pending import (
    DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD,
    mark_pending_wiki_integration,
    pending_wiki_integration_status,
    record_pending_wiki_integration_failure,
)
from ops.validate_wiki import validation_summary
from ops.wiki_integration_plan import (
    apply_wiki_integration_plan,
    build_wiki_integration_plan,
    load_wiki_integration_plan,
    write_wiki_integration_plan_report,
)
from ops.wiki_native_validation import validate_wiki


DEFAULT_NATIVE_WORKDIR = Path(__file__).resolve().parents[1]


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=DEFAULT_WIKI_ROOT)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)


def _tail_text(text: str, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def build_auto_integration_prompt(
    root: Path,
    state_dir: Path,
    status: dict[str, Any],
    reason: str,
    workdir: Path | None = None,
    *,
    plan: dict[str, Any] | None = None,
    plan_path: Path | None = None,
) -> str:
    native_workdir = workdir or DEFAULT_NATIVE_WORKDIR
    wiki_integration_command = "python -m ops.batch_wiki_integration"
    native_refresh_command = "python -m ops.batch_native_refresh"
    validate_command = "python -m ops.validate_wiki"
    plan_hash = str((plan or {}).get("plan_hash") or "not-written")
    operations = (plan or {}).get("operations") or []
    routed_count = sum(1 for op in operations if isinstance(op, dict) and op.get("op") == "raw_map_upsert")
    review_count = sum(1 for op in operations if isinstance(op, dict) and op.get("op") == "review_queue_add")
    plan_path_text = str(plan_path) if plan_path else "(auto-integrate writes this before launch)"
    return f"""You are an autonomous Hermes sub-run launched by `batch_wiki_integration.py auto-integrate` for llm-wiki raw-fast batch wiki integration.

Task: execute the deterministic plan artifact, validate once with raw-map snapshot sync, clear the wiki integration ledger after validation passes, then run guarded native refresh.

Context:
- Wiki root: `{root}`
- Native refresh workdir: `{native_workdir}`
- State dir: `{state_dir}`
- Pending ledger: `{state_dir / 'pending_wiki_integration.json'}`
- Trigger reason: `{reason}`
- Pending status: pending_count={status.get('pending_count')}, actionable_pending_count={status.get('actionable_pending_count')}, threshold={status.get('threshold')}
- Plan artifact: `{plan_path_text}`
- plan_hash: `{plan_hash}`; operations={len(operations)}; routed_raw_notes={routed_count}; review_queue_additions={review_count}

Required references:
- Load `llm-wiki` and use `references/raw-fast-batch-wiki-integration.md` plus `references/wiki-core-operations.md`.
- Use bounded reads/searches for `_meta/raw-clip-map.md` and `_meta/topic-map.md`; keep large map files out of context.

Execution contract:
1. Read the plan artifact JSON first and apply the `operations` deterministically to `_meta/raw-clip-map.md`, `_meta/topic-map.md`, and `log.md`; compiled page edits follow only when the plan explicitly names them.
2. Run one persisted wiki validation with raw-map snapshot sync:
   `{validate_command} --root {root} --state-dir {state_dir} --workdir {native_workdir} --full --write-report --sync-raw-map-snapshot`
   Save stdout to a temp JSON file, inspect `errors` and `warnings`, and keep the report path for closeout.
3. After validation passes, run:
   `{wiki_integration_command} clear-success --root {root} --state-dir {state_dir} --reason {reason}`
   This carries integrated raw notes into the native graph pending ledger.
4. Read native status:
   `{native_refresh_command} status --root {root} --state-dir {state_dir} --workdir {native_workdir}`
   Use `next_refresh_kind`; after 5 completed incremental graph updates the policy queues `full-rebuild`.
5. Validate cutover guards before the build attempt:
   `{native_refresh_command} preflight-cutover --root {root} --state-dir {state_dir} --workdir {native_workdir} --restart-command "$LLM_WIKI_NATIVE_RESTART_COMMAND" --smoke-url "$LLM_WIKI_NATIVE_SMOKE_URL" --smoke-query "$LLM_WIKI_NATIVE_SMOKE_QUERY" --require-unchanged-path "$LLM_WIKI_NATIVE_UNCHANGED_PATH"`
6. When guards are available, execute guarded cutover with vector cache:
   `{native_refresh_command} refresh --root {root} --state-dir {state_dir} --workdir {native_workdir} --cutover --fill-missing-vectors --restart-command "$LLM_WIKI_NATIVE_RESTART_COMMAND" --health-url "$LLM_WIKI_NATIVE_HEALTH_URL" --smoke-url "$LLM_WIKI_NATIVE_SMOKE_URL" --smoke-query "$LLM_WIKI_NATIVE_SMOKE_QUERY" --smoke-query-vector-source active-first-vector --require-unchanged-path "$LLM_WIKI_NATIVE_UNCHANGED_PATH"`
   Live graph freshness is complete after cutover, service restart, `/health`, `/query/data`, unchanged-path audit, and native pending clear succeed; prepare-only output is a prepared artifact, not live graph freshness.

Closeout:
- Redact secrets as `[REDACTED]` in logs.
- Reuse a fresh passing validation report for log-only closeout edits.
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


def run_integrate_local(
    root: Path,
    state_dir: Path,
    *,
    reason: str,
    plan_path: Path,
    workdir: Path | None = None,
    dry_run: bool = False,
) -> tuple[int, dict[str, Any]]:
    plan = load_wiki_integration_plan(plan_path)
    apply_result = apply_wiki_integration_plan(root, state_dir, plan, reason=reason, dry_run=dry_run)
    if apply_result.get("errors"):
        return 13, {"local_runner": True, "plan_path": str(plan_path), **apply_result}
    if dry_run:
        return 0, {"local_runner": True, "plan_path": str(plan_path), **apply_result}

    validation = validate_wiki(
        root,
        state_dir,
        workdir or DEFAULT_NATIVE_WORKDIR,
        full=True,
        write_report=True,
        sync_raw_map_snapshot=True,
    )
    validation_compact = validation_summary(validation)
    validation_ok = not bool(validation.get("errors"))
    if not validation_ok:
        failure = record_pending_wiki_integration_failure(state_dir, "local-validation-failed", "local integration validation reported errors")
        return 14, {
            "local_runner": True,
            "plan_path": str(plan_path),
            "apply": apply_result,
            "validation": validation_compact,
            "failure": failure,
        }
    clear = clear_pending_wiki_integration_after_success(root, state_dir, reason=reason)
    post_status = pending_wiki_integration_status(root, state_dir, reason=reason)
    return 0, {
        "local_runner": True,
        "plan_path": str(plan_path),
        "apply": apply_result,
        "validation": validation_compact,
        "clear_success": clear,
        "post_status": post_status,
    }


def run_auto_integration(
    root: Path,
    state_dir: Path,
    reason: str,
    threshold: int | None = None,
    dry_run: bool = False,
    integration_command: str | None = None,
    timeout: int = 7200,
    runner: str = "local",
) -> tuple[int, dict[str, Any]]:
    pre_status = pending_wiki_integration_status(root, state_dir, reason=reason, threshold=threshold)
    plan = build_wiki_integration_plan(root, state_dir, reason=reason, threshold=threshold)
    plan_path = write_wiki_integration_plan_report(state_dir, plan)
    prompt = build_auto_integration_prompt(root, state_dir, pre_status, reason, plan=plan, plan_path=plan_path)
    prompt_path = write_auto_integration_prompt(state_dir, prompt)
    effective_runner = "external" if integration_command else runner
    if effective_runner == "local":
        command = ["integrate-local", "--plan", str(plan_path)]
    else:
        command = build_runner_command(integration_command, prompt, prompt_path, root, state_dir, reason)
    redacted_command = ["<prompt>" if part == prompt else part for part in command]
    base: dict[str, Any] = {
        "dry_run": dry_run,
        "would_run": bool(pre_status.get("should_integrate")) and not bool(pre_status.get("should_review")),
        "ran": False,
        "runner": effective_runner,
        "pre_status": pre_status,
        "prompt_path": str(prompt_path),
        "plan_path": str(plan_path),
        "plan_hash": plan.get("plan_hash"),
        "plan_operations": len(plan.get("operations") or []),
        "prompt_chars": len(prompt),
        "command": redacted_command,
    }
    if pre_status.get("should_review"):
        return 11, {**base, "skipped": True, "skip_reason": "manual_review_required"}
    if not pre_status.get("should_integrate"):
        return 0, {**base, "skipped": True, "skip_reason": "integration_not_required"}
    if dry_run:
        return 0, {**base, "skipped": False}
    if effective_runner == "local":
        local_code, local_result = run_integrate_local(
            root,
            state_dir,
            reason=reason,
            plan_path=plan_path,
            workdir=DEFAULT_NATIVE_WORKDIR,
        )
        result = {**base, "ran": True, "runner_returncode": local_code, "local_result": local_result}
        if local_code != 0:
            return local_code, result
        post_status = pending_wiki_integration_status(root, state_dir, reason=reason, threshold=threshold)
        result["post_status"] = post_status
        if post_status.get("should_integrate") or post_status.get("should_review"):
            failure = record_pending_wiki_integration_failure(state_dir, "auto-integrate-incomplete", "local integration returned successfully but pending wiki integration ledger still requires action")
            return 12, {**result, "failure": failure}
        return 0, result

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


def main(argv: list[str] | None = None) -> int:
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
    mark_parser.add_argument("--integration-command", default=None, help="Optional command template for auto integration; placeholders: {prompt_path}, {root}, {state_dir}, {ledger_path}, {reason}. Defaults to local deterministic integration.")
    mark_parser.add_argument("--auto-integrate-runner", choices=["local", "hermes"], default="local", help="Runner for --auto-integrate when --integration-command is not set")
    mark_parser.add_argument("--auto-integrate-timeout", type=int, default=7200)

    auto_parser = sub.add_parser("auto-integrate", help="Launch the configured wiki-integration runner when pending raw-fast notes require integration")
    add_common_paths(auto_parser)
    auto_parser.add_argument("--reason", default="threshold", choices=["threshold", "pre-query", "query", "manual", "integrate", "wiki-query"])
    auto_parser.add_argument("--threshold", type=int, default=None, help="Override ledger threshold for this decision")
    auto_parser.add_argument("--dry-run", action="store_true", help="Write the prompt and print the command without launching the runner")
    auto_parser.add_argument("--integration-command", default=None, help="Optional command template; placeholders: {prompt_path}, {root}, {state_dir}, {ledger_path}, {reason}. Defaults to local deterministic integration.")
    auto_parser.add_argument("--runner", choices=["local", "hermes"], default="local", help="Runner to use when --integration-command is not set")
    auto_parser.add_argument("--timeout", type=int, default=7200)

    should_parser = sub.add_parser("should-integrate", help="Return whether batch wiki integration should run for this reason")
    add_common_paths(should_parser)
    should_parser.add_argument("--reason", default="threshold", choices=["threshold", "pre-query", "query", "manual", "integrate", "wiki-query"])
    should_parser.add_argument("--threshold", type=int, default=None, help="Override ledger threshold for this decision")
    should_parser.add_argument("--exit-code", action="store_true", help="Exit 10 when integration is needed, 11 when manual review is needed, 0 when no action is needed")

    apply_parser = sub.add_parser("apply-plan", help="Apply a deterministic wiki integration plan to machine-owned map/log surfaces")
    add_common_paths(apply_parser)
    apply_parser.add_argument("--plan", type=Path, required=True)
    apply_parser.add_argument("--reason", default="integrate")
    apply_parser.add_argument("--dry-run", action="store_true")

    local_parser = sub.add_parser("integrate-local", help="Apply a deterministic plan, validate, and clear pending wiki integration")
    add_common_paths(local_parser)
    local_parser.add_argument("--plan", type=Path, default=None)
    local_parser.add_argument("--reason", default="threshold", choices=["threshold", "pre-query", "query", "manual", "integrate", "wiki-query"])
    local_parser.add_argument("--threshold", type=int, default=None)
    local_parser.add_argument("--dry-run", action="store_true")

    clear_parser = sub.add_parser("clear-success", help="Clear pending wiki-integration ledger after successful batch integration and validation")
    add_common_paths(clear_parser)
    clear_parser.add_argument("--reason", default="external-success")
    clear_parser.add_argument("--integrated-path", action="append", default=[])

    args = parser.parse_args(argv)
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
                runner=args.auto_integrate_runner,
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
            runner=args.runner,
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
    if args.command == "apply-plan":
        plan = load_wiki_integration_plan(args.plan)
        result = apply_wiki_integration_plan(root, state_dir, plan, reason=args.reason, dry_run=args.dry_run)
        print_json(result)
        return 0 if not result.get("errors") else 13
    if args.command == "integrate-local":
        plan_path = args.plan
        if plan_path is None:
            plan = build_wiki_integration_plan(root, state_dir, reason=args.reason, threshold=args.threshold)
            plan_path = write_wiki_integration_plan_report(state_dir, plan)
        code, result = run_integrate_local(root, state_dir, reason=args.reason, plan_path=plan_path, dry_run=args.dry_run)
        print_json(result)
        return code
    if args.command == "clear-success":
        print_json(clear_pending_wiki_integration_after_success(root, state_dir, integrated_paths=args.integrated_path, reason=args.reason))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
