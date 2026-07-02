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

from ops.raw_fast_evidence_bundle import detect_kind, render_agent_handoff_markdown, slugify
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


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else None


def default_slug(url: str, kind: str) -> str:
    parsed = urllib.parse.urlparse(url)
    stem = Path(parsed.path).stem or kind or "source"
    date_prefix = dt.datetime.now().strftime("%y%m%d")
    return f"{date_prefix}_{slugify(stem).lower()}"


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
    kind = detect_kind(args.url, args.kind)
    args.kind = kind
    workdir = (args.workdir.expanduser().resolve() if args.workdir else (tmp_root / (args.slug or default_slug(args.url, kind))).resolve())
    return {"profile": profile, "root": root, "state_dir": state_dir, "tmp_root": tmp_root, "workdir": workdir, "kind": kind}


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


def update_agent_handoff(workdir: Path, closeout: dict[str, Any]) -> dict[str, Any]:
    handoff_path = workdir / "agent_handoff.json"
    if handoff_path.exists():
        handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    else:
        handoff = {"ok": True, "status": "ready", "manual_reference_paths": [], "resource_review_required": False}
    handoff["closeout_args"] = closeout["closeout_args"]
    handoff["closeout_args_path"] = closeout["closeout_args_path"]
    handoff["closeout_command_preview_path"] = closeout["closeout_command_preview_path"]
    if not closeout["closeout_args"].get("ok"):
        handoff["status"] = "manual_required"
        handoff["manual_reason"] = {"stage": "closeout_args", "error": closeout["closeout_args"].get("error")}
        handoff["manual_reference_paths"] = MANUAL_REFERENCE_PATHS
    elif handoff.get("resource_review_required"):
        handoff["status"] = "manual_required"
        handoff["manual_reference_paths"] = MANUAL_REFERENCE_PATHS
    handoff_path.write_text(json.dumps(handoff, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (workdir / "agent_handoff.md").write_text(render_agent_handoff_markdown(handoff), encoding="utf-8")
    return handoff


def run_prepare(args: argparse.Namespace, paths: dict[str, Any]) -> dict[str, Any]:
    root: Path = paths["root"]
    workdir: Path = paths["workdir"]
    if workdir == root or workdir.is_relative_to(root):
        return {"ok": False, "stage": "preflight", "error": "workdir_inside_wiki_root", "root": str(root), "workdir": str(workdir), "manual_required": True, "manual_reason": {"stage": "preflight", "error": "workdir_inside_wiki_root"}, "manual_reference_paths": MANUAL_REFERENCE_PATHS, "agent_next_reads": []}
    workdir.mkdir(parents=True, exist_ok=True)
    command = build_prepare_command(args, paths)
    if args.print_command:
        return {"ok": True, "stage": "command", "profile": paths["profile"], "root": str(root), "state_dir": str(paths["state_dir"]) if paths.get("state_dir") else None, "workdir": str(workdir), "command": command, "agent_next_reads": agent_next_reads(workdir), "ready_message": "Raw-fast evidence command prepared; execute without manual resource setup."}
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
        "command": command,
        "command_returncode": result["returncode"],
        "agent_next_reads": agent_next_reads(workdir, ok=ok),
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
    handoff = update_agent_handoff(workdir, closeout)
    resource_review_required = bool(handoff.get("resource_review_required")) or handoff.get("status") == "manual_required"
    output.update(
        {
            "ready_message": "Raw-fast evidence prepared; read agent_handoff.md first and do not run manual resource discovery unless the handoff says manual_required.",
            "resource_review_required": resource_review_required,
            "closeout_args_path": closeout["closeout_args_path"],
            "closeout_command_preview_path": closeout["closeout_command_preview_path"],
        }
    )
    if resource_review_required:
        output["manual_required"] = True
        output["manual_reason"] = handoff.get("manual_reason")
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
    parser.add_argument("--pdf-backend", choices=["docling", "pdftotext", "auto"], default="docling")
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
    return 0 if output.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
