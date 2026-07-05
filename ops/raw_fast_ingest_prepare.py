#!/usr/bin/env python3
"""One-command non-mutating preparation for llm-wiki raw-fast clipping.

This wrapper intentionally delegates extraction to `ops.raw_fast_evidence_bundle`.
It writes only the resolved tmp/workdir and returns the compact file an agent
should read next. Manual references stay hidden unless preparation or a bounded
sub-stage reports a concrete failure/manual-required reason.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import urllib.parse
from pathlib import Path
from typing import Any

from ops.raw_fast_evidence_bundle import ASSEMBLED_RAW_NOTE_REPORT_FILE, RAW_BODY_DRAFT_FILE, detect_kind, manual_reference_policy, render_agent_handoff_markdown, slugify
from ops.raw_fast_closeout import derive_closeout_args_from_bundle

PROD_WIKI_ROOT = Path("/mnt/d/data/Clippings/llm-wiki")
PROD_STATE_DIR = Path("/home/xu/project/wiki/storage/zvec/llm-wiki-prod")
PROD_TMP_ROOT = Path("/home/xu/tmp/llm_wiki_raw_fast")
DEFAULT_TMP_ROOT = PROD_TMP_ROOT
MANUAL_REFERENCE_PATHS = [
    "/home/xu/.hermes/skills/research/llm-wiki/references/structured-paper-ingest-router.md",
    "/home/xu/.hermes/skills/research/llm-wiki/references/raw-fast-resource-probe-boundaries.md",
    "/home/xu/.hermes/skills/research/llm-wiki/references/raw-fast-batch-wiki-integration.md",
]


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def manual_stop_message(payload: dict[str, Any]) -> str:
    if payload.get("manual_reference_paths"):
        return "raw_fast_ingest_prepare stopped: read only manual_reference_paths from the JSON output, then follow manual_reason."
    return "raw_fast_ingest_prepare stopped without manual_reference_paths: report the automation failure and do not guess fallback references."


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else None


def default_slug(url: str, kind: str) -> str:
    parsed = urllib.parse.urlparse(url)
    stem = Path(parsed.path).stem or kind or "source"
    date_prefix = dt.datetime.now().strftime("%y%m%d")
    return f"{date_prefix}_{slugify(stem).lower()}"


def normalize_source_url(url: str) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    parts = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
    if host == "github.com" and len(parts) >= 5 and parts[2] in {"blob", "raw"} and parts[-1].lower().endswith(".pdf"):
        owner, repo, marker, ref = parts[:4]
        path_parts = parts[4:]
        quoted = [urllib.parse.quote(part, safe="") for part in [owner, repo, ref, *path_parts]]
        normalized = f"https://raw.githubusercontent.com/{'/'.join(quoted)}"
        reason = "github_blob_pdf" if marker == "blob" else "github_raw_pdf"
        return {"url": normalized, "original_url": url, "normalized": normalized != url, "reason": reason}
    return {"url": url, "original_url": url, "normalized": False, "reason": None}


def resolve_prepare_paths(args: argparse.Namespace) -> dict[str, Any]:
    profile = args.profile
    explicit_root = args.root is not None
    if profile == "env":
        profile_root = _env_path("LLM_WIKI_ROOT") or PROD_WIKI_ROOT
        profile_state = _env_path("LLM_WIKI_STATE_DIR") or PROD_STATE_DIR
        profile_tmp = _env_path("LLM_WIKI_RAW_FAST_TMP_ROOT") or _env_path("LLM_WIKI_TMP_ROOT") or PROD_TMP_ROOT
    else:
        profile_root = PROD_WIKI_ROOT
        profile_state = PROD_STATE_DIR
        profile_tmp = _env_path("LLM_WIKI_RAW_FAST_TMP_ROOT") or PROD_TMP_ROOT
    root = (args.root or profile_root).expanduser().resolve()
    state_dir = (args.state_dir.expanduser().resolve() if args.state_dir else (profile_state.expanduser().resolve() if not explicit_root else None))
    tmp_root = (args.tmp_root or profile_tmp).expanduser().resolve()
    source_url_normalization = normalize_source_url(args.url)
    args.url = str(source_url_normalization["url"])
    kind = detect_kind(args.url, args.kind)
    args.kind = kind
    workdir = (args.workdir.expanduser().resolve() if args.workdir else (tmp_root / (args.slug or default_slug(args.url, kind))).resolve())
    return {
        "profile": profile,
        "root": root,
        "state_dir": state_dir,
        "tmp_root": tmp_root,
        "workdir": workdir,
        "kind": kind,
        "supplied_url": source_url_normalization["original_url"],
        "source_url": source_url_normalization["url"],
        "source_url_normalization": source_url_normalization,
    }


def build_prepare_command(args: argparse.Namespace, paths: dict[str, Any]) -> list[str]:
    workdir: Path = paths["workdir"]
    command = [
        sys.executable,
        "-m",
        "ops.raw_fast_evidence_bundle",
        "--url",
        args.url,
        "--kind",
        paths["kind"],
        "--root",
        str(paths["root"]),
        "--workdir",
        str(workdir),
        "--pdf-backend",
        args.pdf_backend,
        "--timeout",
        str(args.timeout),
        "--paper-digest",
        "--resource-draft",
        "--resource-health",
        args.resource_health,
    ]
    if paths.get("state_dir"):
        command.extend(["--state-dir", str(paths["state_dir"])])
    if args.strict_pdf_backend:
        command.append("--strict-pdf-backend")
    probes = args.probe if args.probe is not None else None
    if probes:
        for probe in probes:
            command.extend(["--probe", probe])
    if args.localize_figures:
        command.append("--localize-figures")
        command.extend(["--image-slug", args.image_slug or slugify(workdir.name).lower()])
    return command


def agent_next_reads(workdir: Path, *, ok: bool = True) -> list[str]:
    if not ok:
        return []
    return [str((workdir / "agent_handoff.md").resolve())]


def prepare_automation_next_action(workdir: Path, *, ok: bool, reason: str | None = None) -> dict[str, Any]:
    if ok:
        return {
            "action": "read_agent_handoff",
            "read_path": str((workdir / "agent_handoff.md").resolve()),
            "manual_reference_policy": "only_on_manual_required",
        }
    return {
        "action": "read_manual_reference_paths",
        "reason": reason or "script_failed",
        "manual_reference_policy": "only_on_manual_required",
    }


def run_json_command(command: list[str], *, cwd: Path, timeout: int) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = None
    return {
        "returncode": completed.returncode,
        "json": payload,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "command": command,
    }


def _manual_reason_from_result(result: dict[str, Any]) -> dict[str, Any]:
    raw_payload = result.get("json")
    payload = raw_payload if isinstance(raw_payload, dict) else {}
    raw_fetch = payload.get("fetch")
    fetch = raw_fetch if isinstance(raw_fetch, dict) else {}
    return {
        "stage": payload.get("stage") or "evidence_bundle_failed",
        "error": payload.get("error") or fetch.get("error") or "prepare_failed",
        "message": payload.get("message") or fetch.get("message") or result.get("stderr_tail") or result.get("stdout_tail"),
        "returncode": result.get("returncode"),
    }


def write_closeout_artifacts(workdir: Path, paths: dict[str, Any]) -> dict[str, Any]:
    bundle_path = workdir / "evidence_bundle.json"
    closeout_args = derive_closeout_args_from_bundle(bundle_path)
    args_path = workdir / "closeout_args.json"
    args_path.write_text(json.dumps(closeout_args, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    preview_path = workdir / "closeout_command.preview.sh"
    command = [
        sys.executable,
        "-m",
        "ops.raw_fast_closeout",
        "--root",
        str(paths["root"]),
    ]
    if paths.get("state_dir"):
        command.extend(["--state-dir", str(paths["state_dir"])])
    command.extend(["--workdir", str(Path(__file__).resolve().parents[1])])
    if closeout_args.get("ok"):
        command.extend(closeout_args.get("argv_tail") or [])
        command.extend(["--allow-non-tmp-cleanup", "--fast-final-verify", "--append-log", "--auto-integrate", "--native-refresh-mode", "status"])
    preview_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(Path(__file__).resolve().parents[1])) + "\n" + " ".join(shlex.quote(str(part)) for part in command) + "\n", encoding="utf-8")
    return {"closeout_args": closeout_args, "closeout_args_path": str(args_path.resolve()), "closeout_command_preview_path": str(preview_path.resolve())}


def write_assemble_artifacts(workdir: Path, paths: dict[str, Any], closeout: dict[str, Any]) -> dict[str, Any]:
    raw_closeout_args = closeout.get("closeout_args")
    closeout_args = raw_closeout_args if isinstance(raw_closeout_args, dict) else {}
    preview_path = workdir / "assemble_command.preview.sh"
    report_path = workdir / ASSEMBLED_RAW_NOTE_REPORT_FILE
    command = [
        sys.executable,
        "-m",
        "ops.raw_fast_note_assemble",
        "--root",
        str(paths["root"]),
        "--workdir",
        str(workdir),
        "--body-draft",
        RAW_BODY_DRAFT_FILE,
        "--output-report",
        str(report_path),
    ]
    if closeout_args.get("raw_file"):
        command.extend(["--raw-file", str(closeout_args["raw_file"])])
    preview_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(Path(__file__).resolve().parents[1])) + "\n" + " ".join(shlex.quote(str(part)) for part in command) + "\n", encoding="utf-8")
    return {
        "command_preview_path": str(preview_path.resolve()),
        "report_path": str(report_path.resolve()),
        "body_draft_path": str((workdir / RAW_BODY_DRAFT_FILE).resolve()),
        "raw_file": closeout_args.get("raw_file"),
    }


def update_agent_handoff(workdir: Path, closeout: dict[str, Any], assemble: dict[str, Any] | None = None) -> dict[str, Any]:
    handoff_path = workdir / "agent_handoff.json"
    if handoff_path.exists():
        handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    else:
        handoff = {
            "ok": True,
            "status": "ready",
            "manual_reference_paths": [],
            "manual_reference_policy": manual_reference_policy(visible=False),
            "automation_next_action": {"action": "read_agent_handoff", "manual_reference_policy": "only_on_manual_required"},
            "resource_review_required": False,
            "agent_actions": [],
        }
    handoff["body_draft"] = {"path": RAW_BODY_DRAFT_FILE, "contract": "body_only_no_frontmatter"}
    raw_source_refs = handoff.get("source_refs")
    source_refs = raw_source_refs if isinstance(raw_source_refs, dict) else {}
    source_refs["body_draft"] = RAW_BODY_DRAFT_FILE
    handoff["source_refs"] = source_refs
    handoff["closeout_args"] = closeout["closeout_args"]
    handoff["closeout_args_path"] = closeout["closeout_args_path"]
    handoff["closeout_command_preview_path"] = closeout["closeout_command_preview_path"]
    if assemble is not None:
        handoff["assemble"] = {
            "command_preview_path": assemble["command_preview_path"],
            "report_path": assemble["report_path"],
            "body_draft_path": assemble["body_draft_path"],
            "raw_file": assemble.get("raw_file"),
        }
    handoff["agent_actions"] = [
        "Read this handoff first; use Default source reads and the sanitized scientific digest for body synthesis.",
        f"Write the raw-note body only to `{RAW_BODY_DRAFT_FILE}`; do not write YAML frontmatter or metadata.",
        "Run assemble_command.preview.sh to create the canonical raw note from script-owned metadata, then run closeout_command.preview.sh.",
    ]
    if not closeout["closeout_args"].get("ok"):
        handoff["status"] = "manual_required"
        handoff["manual_reason"] = {"stage": "closeout_args", "error": closeout["closeout_args"].get("error")}
        handoff["manual_reference_policy"] = manual_reference_policy(visible=True)
        handoff["automation_next_action"] = {"action": "read_manual_reference_paths", "reason": "manual_required", "manual_reference_policy": "only_on_manual_required"}
        handoff["manual_reference_paths"] = MANUAL_REFERENCE_PATHS
    elif handoff.get("resource_review_required"):
        handoff["status"] = "manual_required"
        handoff["manual_reference_policy"] = manual_reference_policy(visible=True)
        handoff["automation_next_action"] = {"action": "read_manual_reference_paths", "reason": "manual_required", "manual_reference_policy": "only_on_manual_required"}
        handoff["manual_reference_paths"] = MANUAL_REFERENCE_PATHS
    handoff_path.write_text(json.dumps(handoff, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (workdir / "agent_handoff.md").write_text(render_agent_handoff_markdown(handoff), encoding="utf-8")
    return handoff


def run_prepare(args: argparse.Namespace, paths: dict[str, Any]) -> dict[str, Any]:
    root: Path = paths["root"]
    workdir: Path = paths["workdir"]
    if workdir == root or workdir.is_relative_to(root):
        return {
            "ok": False,
            "stage": "preflight",
            "error": "workdir_inside_wiki_root",
            "root": str(root),
            "workdir": str(workdir),
            "supplied_url": paths.get("supplied_url"),
            "source_url": paths.get("source_url"),
            "source_url_normalization": paths.get("source_url_normalization"),
            "manual_required": True,
            "manual_reason": {"stage": "preflight", "error": "workdir_inside_wiki_root"},
            "manual_reference_policy": manual_reference_policy(visible=True),
            "automation_next_action": prepare_automation_next_action(workdir, ok=False, reason="script_failed"),
            "manual_reference_paths": MANUAL_REFERENCE_PATHS,
            "agent_next_reads": [],
        }
    workdir.mkdir(parents=True, exist_ok=True)
    command = build_prepare_command(args, paths)
    if args.print_command:
        return {
            "ok": True,
            "stage": "command",
            "profile": paths["profile"],
            "root": str(root),
            "state_dir": str(paths["state_dir"]) if paths.get("state_dir") else None,
            "workdir": str(workdir),
            "supplied_url": paths.get("supplied_url"),
            "source_url": paths.get("source_url"),
            "source_url_normalization": paths.get("source_url_normalization"),
            "command": command,
            "agent_next_reads": agent_next_reads(workdir),
            "manual_reference_policy": manual_reference_policy(visible=False),
            "automation_next_action": prepare_automation_next_action(workdir, ok=True),
            "ready_message": "Raw-fast evidence command prepared; execute without manual resource setup.",
        }
    result = run_json_command(command, cwd=Path(__file__).resolve().parents[1], timeout=args.timeout + 30)
    payload = result.get("json") if isinstance(result.get("json"), dict) else None
    ok = result["returncode"] == 0 and bool(payload and payload.get("ok"))
    output: dict[str, Any] = {
        "ok": ok,
        "stage": "prepared" if ok else "evidence_bundle_failed",
        "profile": paths["profile"],
        "root": str(root),
        "state_dir": str(paths["state_dir"]) if paths.get("state_dir") else None,
        "workdir": str(workdir),
        "supplied_url": paths.get("supplied_url"),
        "source_url": paths.get("source_url"),
        "source_url_normalization": paths.get("source_url_normalization"),
        "command": command,
        "command_returncode": result["returncode"],
        "agent_next_reads": agent_next_reads(workdir, ok=ok),
        "manual_reference_policy": manual_reference_policy(visible=not ok),
        "automation_next_action": prepare_automation_next_action(workdir, ok=ok),
    }
    if payload is not None:
        output["evidence_bundle"] = payload
    else:
        output["stdout_tail"] = result.get("stdout_tail")
    if not ok:
        output["stderr_tail"] = result.get("stderr_tail")
        output["manual_required"] = True
        output["manual_reason"] = _manual_reason_from_result(result)
        output["manual_reference_paths"] = MANUAL_REFERENCE_PATHS
        output["agent_next_reads"] = []
        return output

    closeout = write_closeout_artifacts(workdir, paths)
    assemble = write_assemble_artifacts(workdir, paths, closeout)
    handoff = update_agent_handoff(workdir, closeout, assemble)
    resource_review_required = bool(handoff.get("resource_review_required")) or handoff.get("status") == "manual_required"
    output.update(
        {
            "ready_message": "Raw-fast evidence prepared; read agent_handoff.md first and do not run manual resource discovery unless the handoff says manual_required.",
            "resource_review_required": resource_review_required,
            "closeout_args_path": closeout["closeout_args_path"],
            "closeout_command_preview_path": closeout["closeout_command_preview_path"],
            "assemble_command_preview_path": assemble["command_preview_path"],
        }
    )
    if resource_review_required:
        output["manual_required"] = True
        output["manual_reason"] = handoff.get("manual_reason")
        output["manual_reference_policy"] = manual_reference_policy(visible=True)
        output["automation_next_action"] = prepare_automation_next_action(workdir, ok=False, reason="handoff_manual_required")
        output["manual_reference_paths"] = handoff.get("manual_reference_paths") or MANUAL_REFERENCE_PATHS
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare tmp raw-fast evidence, resource triage, agent handoff, and closeout args without writing the wiki")
    parser.add_argument("--url", required=True)
    parser.add_argument("--profile", choices=["prod", "env"], default="prod")
    parser.add_argument("--kind", choices=["auto", "direct-pdf", "arxiv"], default="auto")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--tmp-root", type=Path, default=None)
    parser.add_argument("--slug", default=None)
    parser.add_argument("--probe", action="append", default=None, choices=["arxiv", "doi", "none"])
    parser.add_argument("--resource-health", choices=["direct", "none"], default="direct")
    parser.add_argument("--pdf-backend", choices=["docling", "auto"], default="docling")
    parser.add_argument("--strict-pdf-backend", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--localize-figures", action="store_true")
    parser.add_argument("--image-slug", default=None)
    parser.add_argument("--print-command", action="store_true", help="Print the derived evidence-bundle command without executing it")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = resolve_prepare_paths(args)
    output = run_prepare(args, paths)
    print_json(output)
    if not output.get("ok"):
        print(manual_stop_message(output), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
