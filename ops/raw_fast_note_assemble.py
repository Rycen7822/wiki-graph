#!/usr/bin/env python3
"""Assemble one raw-fast raw note from script-owned metadata and an agent body draft.

This module is intentionally narrow: evidence bundle owns metadata sidecars,
this assembler owns the single wiki raw-note write, and raw_fast_closeout owns
verification/ledger/log/status after the note exists.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from ops.raw_fast_evidence_bundle import yamlish

DEFAULT_BODY_DRAFT = "raw_body_draft.md"
DEFAULT_FRONTMATTER = "candidate_frontmatter.json"
DEFAULT_HANDOFF = "agent_handoff.json"
DEFAULT_REPORT = "assembled_raw_note_report.json"
DEFAULT_VERIFIER = Path.home() / ".hermes" / "skills" / "research" / "llm-wiki" / "scripts" / "raw_fast_note_verify.py"

REQUIRED_HEADINGS = [
    "一句话总结",
    "论文摘要（中文）",
    "Motivation",
    "Methodology",
    "关键实验结果 / 作者结论",
    "对未来研究的启发",
    "可能的局限",
    "可继续追问的问题",
]

FRONTMATTER_METADATA_RE = re.compile(
    r"^(title|source|created|updated|type|domain|tags|topic_hints|github_links|"
    r"huggingface_model_links|huggingface_dataset_links|capture_route|captured)\s*:",
    re.IGNORECASE | re.MULTILINE,
)
RESOURCE_OR_URL_RE = re.compile(
    r"https?://|\[[^\]]+\]\([^\)]+\)|\b(?:link[-_ ]?health|resource_status|resource status|HEAD\s+\d{3}|PapersWithCode|Hugging Face|GitHub)\b",
    re.IGNORECASE,
)


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fail(stage: str, error: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "stage": stage, "error": error, **extra}


def _safe_relative_raw_file(raw_file: str | None) -> tuple[bool, str | None, str | None]:
    if not raw_file:
        return False, None, "missing_raw_file"
    rel = Path(str(raw_file))
    if rel.is_absolute() or ".." in rel.parts or not str(raw_file).startswith("raw/clip/") or rel.suffix.lower() != ".md":
        return False, str(raw_file), "invalid_raw_file"
    return True, str(raw_file), None


def _resolve_workdir_path(workdir: Path, path: Path | str | None, default_name: str) -> Path:
    candidate = Path(path) if path is not None else Path(default_name)
    if candidate.is_absolute():
        return candidate
    return workdir / candidate


def resolve_raw_file(workdir: Path, explicit_raw_file: str | None = None) -> dict[str, Any]:
    if explicit_raw_file:
        ok, value, error = _safe_relative_raw_file(explicit_raw_file)
        return {"ok": ok, "raw_file": value, "error": error, "source": "argument"}

    handoff = _load_json(workdir / DEFAULT_HANDOFF) or {}
    raw_anchors = handoff.get("protected_anchors")
    anchors = raw_anchors if isinstance(raw_anchors, dict) else {}
    raw_file = anchors.get("next_raw_path")
    if not raw_file:
        evidence = _load_json(workdir / "evidence_bundle.json") or {}
        raw_preflight = evidence.get("preflight")
        preflight = raw_preflight if isinstance(raw_preflight, dict) else {}
        raw_file = evidence.get("next_raw_path") or preflight.get("next_raw_path")
        source = "evidence_bundle"
    else:
        source = "agent_handoff"
    ok, value, error = _safe_relative_raw_file(str(raw_file) if raw_file else None)
    return {"ok": ok, "raw_file": value, "error": error, "source": source}


def refresh_frontmatter_timestamps(frontmatter: dict[str, Any], now: dt.datetime | None = None) -> dict[str, Any]:
    now = now or dt.datetime.now().astimezone()
    refreshed = dict(frontmatter)
    refreshed["created"] = now.strftime("%Y-%m-%d")
    refreshed["updated"] = now.strftime("%Y-%m-%d %H:%M")
    refreshed["captured"] = now.strftime("%Y-%m-%d %H:%M:%S %Z (%z)")
    return refreshed


def validate_body_draft(text: str) -> dict[str, Any]:
    stripped = text.lstrip()
    if stripped.startswith("---"):
        return _fail("body_draft", "body_frontmatter_forbidden")
    metadata_match = FRONTMATTER_METADATA_RE.search(text)
    if metadata_match:
        return _fail("body_draft", "body_metadata_line_forbidden", field=metadata_match.group(1))
    leakage_match = RESOURCE_OR_URL_RE.search(text)
    if leakage_match:
        return _fail("body_draft", "body_resource_or_url_leakage", match=leakage_match.group(0))
    headings = re.findall(r"^##\s+(.+?)\s*$", text, flags=re.MULTILINE)
    missing = [heading for heading in REQUIRED_HEADINGS if heading not in headings]
    if missing:
        return _fail("body_draft", "required_headings_missing", missing_headings=missing, heading_count=len(headings))
    return {"ok": True, "stage": "body_draft", "heading_count": len(headings), "missing_headings": []}


def run_verifier(root: Path, raw_file: str, *, verifier: Path, frontmatter: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    patterns = [str(frontmatter.get("title") or "").strip(), str(frontmatter.get("source") or "").strip()]
    command = [sys.executable, str(verifier), "--wiki", str(root), "--raw-file", raw_file, "--structured-paper"]
    unique_patterns = [value for idx, value in enumerate(patterns) if value and value not in patterns[:idx]]
    if unique_patterns:
        command.append("--patterns")
        command.extend(unique_patterns)
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = None
    return {
        "ok": completed.returncode == 0 and bool(isinstance(payload, dict) and payload.get("raw_fast_ok")),
        "returncode": completed.returncode,
        "command": command,
        "raw_fast_ok": bool(isinstance(payload, dict) and payload.get("raw_fast_ok")),
        "payload": payload,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def _finalize_report(workdir: Path, output_report: Path | str | None, payload: dict[str, Any]) -> dict[str, Any]:
    report_path = _resolve_workdir_path(workdir, output_report, DEFAULT_REPORT)
    payload = {**payload, "report_path": str(report_path.resolve())}
    _write_json(report_path, payload)
    return payload


def assemble_raw_note(
    *,
    root: Path,
    workdir: Path,
    body_draft: Path | str | None = None,
    candidate_frontmatter: Path | str | None = None,
    raw_file: str | None = None,
    output_report: Path | str | None = None,
    overwrite_existing: bool = False,
    verify: bool = False,
    verifier: Path | None = None,
    verify_timeout: int = 120,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    workdir = workdir.expanduser().resolve()
    frontmatter_path = _resolve_workdir_path(workdir, candidate_frontmatter, DEFAULT_FRONTMATTER)
    body_path = _resolve_workdir_path(workdir, body_draft, DEFAULT_BODY_DRAFT)

    frontmatter = _load_json(frontmatter_path)
    if not frontmatter:
        return _finalize_report(workdir, output_report, _fail("frontmatter", "invalid_candidate_frontmatter", frontmatter_path=str(frontmatter_path)))

    resolved = resolve_raw_file(workdir, raw_file)
    if not resolved.get("ok"):
        return _finalize_report(workdir, output_report, _fail("raw_file", str(resolved.get("error") or "invalid_raw_file"), raw_file=resolved.get("raw_file"), source=resolved.get("source")))
    raw_rel = str(resolved["raw_file"])
    raw_path = root / raw_rel
    if raw_path.exists() and not overwrite_existing:
        return _finalize_report(workdir, output_report, _fail("raw_file", "raw_file_exists", raw_file=raw_rel, raw_path=str(raw_path)))

    try:
        body_text = body_path.read_text(encoding="utf-8")
    except OSError as exc:
        return _finalize_report(workdir, output_report, _fail("body_draft", "body_draft_missing", body_draft=str(body_path), detail=str(exc)))
    body_checks = validate_body_draft(body_text)
    if not body_checks.get("ok"):
        return _finalize_report(workdir, output_report, {**body_checks, "body_draft": str(body_path)})

    final_frontmatter = refresh_frontmatter_timestamps(frontmatter)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_text = yamlish(final_frontmatter).rstrip() + "\n\n" + body_text.strip() + "\n"
    raw_path.write_text(raw_text, encoding="utf-8")

    payload: dict[str, Any] = {
        "ok": True,
        "stage": "assembled",
        "raw_file": raw_rel,
        "raw_path": str(raw_path),
        "root": str(root),
        "workdir": str(workdir),
        "frontmatter_path": str(frontmatter_path),
        "body_draft": str(body_path),
        "frontmatter_fields": list(final_frontmatter.keys()),
        "body_checks": body_checks,
    }
    if verify:
        verify_result = run_verifier(root, raw_rel, verifier=(verifier or DEFAULT_VERIFIER), frontmatter=final_frontmatter, timeout=verify_timeout)
        payload["verify"] = verify_result
        if not verify_result.get("ok"):
            payload["ok"] = False
            payload["stage"] = "verify"
            payload["error"] = "verifier_failed"
    return _finalize_report(workdir, output_report, payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble a raw-fast raw note from script-owned metadata and a body-only draft")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--body-draft", type=Path, default=Path(DEFAULT_BODY_DRAFT))
    parser.add_argument("--candidate-frontmatter", type=Path, default=Path(DEFAULT_FRONTMATTER))
    parser.add_argument("--raw-file", default=None)
    parser.add_argument("--output-report", type=Path, default=Path(DEFAULT_REPORT))
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verifier", type=Path, default=DEFAULT_VERIFIER)
    parser.add_argument("--verify-timeout", type=int, default=120)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = assemble_raw_note(
        root=args.root,
        workdir=args.workdir,
        body_draft=args.body_draft,
        candidate_frontmatter=args.candidate_frontmatter,
        raw_file=args.raw_file,
        output_report=args.output_report,
        overwrite_existing=args.overwrite_existing,
        verify=args.verify,
        verifier=args.verifier,
        verify_timeout=args.verify_timeout,
    )
    print_json(payload)
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
