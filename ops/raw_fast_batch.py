#!/usr/bin/env python3
"""Staging-only batch orchestration with one canonical closeout writer."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any

from ops.batch_wiki_integration import run_auto_integration
from ops.raw_fast_closeout import build_compact_log_entry
from ops.raw_fast_note_assemble import render_raw_note
from ops.raw_fast_publish import (
    RawPublishError,
    allocate_raw_paths_locked,
    publish_raw_text_exclusive_locked,
    validate_raw_relative_path,
)
from ops.wiki_mutation_lock import atomic_write_json, wiki_mutation_lock
from ops.wiki_native_wiki_integration_pending import (
    mark_pending_wiki_integration_batch,
    pending_wiki_integration_status,
)

DEFAULT_TMP_ROOT = Path("/home/xu/tmp/llm_wiki_raw_fast_batches")
MANIFEST_NAME = "batch_manifest.json"
REPORT_NAME = "batch_closeout_report.json"
WORKER_RESULT_NAME = "worker_result.json"
WORKER_CONTRACT_NAME = "worker_contract.json"
WORKER_PROMPT_NAME = "worker_prompt.md"
BODY_NAME = "raw_body_draft.md"
FRONTMATTER_NAME = "candidate_frontmatter.json"
WORKER_CONTRACT_VERSION = 1
SOURCE_LINE_RE = re.compile(r"^source:\s*[\"']?(.*?)[\"']?\s*$", re.MULTILINE)
SAFE_BATCH_ID_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z_.-]{0,95}$")


def _now_stamp() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _source_hash(source_url: str) -> str:
    return hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:12]


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_child(batch_root: Path, child: Path) -> Path:
    resolved_root = batch_root.expanduser().resolve()
    resolved_child = child.expanduser().resolve()
    if resolved_child != resolved_root and resolved_root not in resolved_child.parents:
        raise ValueError(f"item path escapes batch root: {resolved_child}")
    return resolved_child


def _batch_id(value: str | None = None) -> str:
    if value:
        if not SAFE_BATCH_ID_RE.fullmatch(value):
            raise ValueError("batch_id must match [0-9A-Za-z][0-9A-Za-z_.-]{0,95}")
        return value
    return f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:8]}"


def init_batch(
    *,
    root: Path,
    state_dir: Path,
    urls: list[str],
    tmp_root: Path = DEFAULT_TMP_ROOT,
    batch_id: str | None = None,
    threshold: int | None = None,
    runner: str = "local",
) -> dict[str, Any]:
    if not urls:
        raise ValueError("at least one --url is required")
    root = root.expanduser().resolve()
    state_dir = state_dir.expanduser().resolve()
    tmp_root = tmp_root.expanduser().resolve()
    resolved_batch_id = _batch_id(batch_id)
    batch_root = tmp_root / resolved_batch_id
    manifest_path = batch_root / MANIFEST_NAME
    if manifest_path.exists():
        raise FileExistsError(f"batch already exists: {manifest_path}")
    batch_root.mkdir(parents=True, exist_ok=False)
    items: list[dict[str, Any]] = []
    for input_index, source_url in enumerate(urls):
        source_url = str(source_url).strip()
        if not source_url:
            raise ValueError(f"empty source URL at input index {input_index}")
        source_hash = _source_hash(source_url)
        workdir = batch_root / f"{input_index:03d}-{source_hash}"
        workdir.mkdir(parents=False, exist_ok=False)
        prepare_command = [
            sys.executable,
            "-m",
            "ops.raw_fast_ingest_prepare",
            "--url",
            source_url,
            "--root",
            str(root),
            "--state-dir",
            str(state_dir),
            "--workdir",
            str(workdir),
        ]
        items.append(
            {
                "input_index": input_index,
                "source_url": source_url,
                "source_hash": source_hash,
                "workdir": str(workdir),
                "status": "staging",
                "prepare_command": prepare_command,
                "worker_contract": {
                    "body_draft": BODY_NAME,
                    "candidate_frontmatter": FRONTMATTER_NAME,
                    "worker_result": WORKER_RESULT_NAME,
                    "canonical_writes_allowed": False,
                },
            }
        )
    manifest = {
        "schema_version": 1,
        "batch_id": resolved_batch_id,
        "phase": "staging",
        "created_at": _now_stamp(),
        "updated_at": _now_stamp(),
        "root": str(root),
        "state_dir": str(state_dir),
        "batch_root": str(batch_root),
        "threshold": threshold,
        "runner": runner,
        "items": items,
    }
    atomic_write_json(manifest_path, manifest)
    return {"ok": True, **manifest, "manifest_path": str(manifest_path)}


def _load_manifest(manifest_path: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = _load_json(manifest_path)
    if manifest is None or manifest.get("schema_version") != 1:
        raise ValueError(f"invalid batch manifest: {manifest_path}")
    batch_root = Path(str(manifest.get("batch_root") or manifest_path.parent)).expanduser().resolve()
    if manifest_path != batch_root / MANIFEST_NAME:
        raise ValueError("manifest path does not match batch_root")
    items = manifest.get("items")
    if not isinstance(items, list):
        raise ValueError("manifest items must be a list")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("manifest item must be an object")
        _safe_child(batch_root, Path(str(item.get("workdir") or "")))
    return manifest_path, manifest


def _render_worker_prompt(contract_path: Path) -> str:
    return f"""# Raw-fast batch worker

Authority: `{contract_path}`. Treat its identity and paths as literal values; do not infer or rewrite them.

1. Load `llm-wiki` once, then read the contract, its handoff, writing contracts, and only the source reads named by the handoff.
2. Work only inside the contract workdir. Do not edit the batch manifest or canonical wiki, and do not run assemble, closeout, integration, or native refresh.
3. Batch independent reads and vision calls. Use fallback source spans only for a named evidence gap. After one visual batch, retry at most one still-required visual with a targeted crop.
4. Write the body once to the contract body path. In candidate frontmatter, edit only `domain`, `tags`, or `topic_hints`.
5. Run the contract finish command exactly. If it fails, repair only the reported issue and rerun once. Do not write `worker_result.json` or create an ad hoc verifier.
6. Report only ready/blocked, output paths, and any unresolved evidence gap.
"""


def materialize_worker_contracts(manifest_path: Path) -> dict[str, Any]:
    manifest_path, manifest = _load_manifest(manifest_path)
    batch_root = Path(str(manifest["batch_root"])).resolve()
    tasks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for item in sorted(manifest["items"], key=lambda value: int(value.get("input_index", 0))):
        input_index = int(item["input_index"])
        workdir = _safe_child(batch_root, Path(str(item["workdir"])))
        handoff = _load_json(workdir / "agent_handoff.json")
        frontmatter = _load_json(workdir / FRONTMATTER_NAME)
        error = None
        if not handoff:
            error = "agent_handoff_missing"
        elif handoff.get("status") != "ready" or handoff.get("resource_review_required"):
            error = "handoff_not_ready"
        elif not frontmatter:
            error = "candidate_frontmatter_missing"
        title = str((frontmatter or {}).get("title") or "").strip()
        source_id = str((frontmatter or {}).get("source") or "").strip()
        if error is None and (not title or not source_id):
            error = "title_or_source_missing"
        if error:
            item.update({"status": "worker_contract_failed", "error": error})
            failures.append({"input_index": input_index, "error": error, "workdir": str(workdir)})
            continue
        assert handoff is not None and frontmatter is not None

        writing_contracts = [
            str(ref.get("path"))
            for ref in handoff.get("writing_contract_refs") or []
            if isinstance(ref, dict) and str(ref.get("path") or "").strip()
        ]
        if not writing_contracts:
            error = "writing_contract_missing"
            item.update({"status": "worker_contract_failed", "error": error})
            failures.append({"input_index": input_index, "error": error, "workdir": str(workdir)})
            continue
        contract_path = workdir / WORKER_CONTRACT_NAME
        prompt_path = workdir / WORKER_PROMPT_NAME
        contract = {
            "version": WORKER_CONTRACT_VERSION,
            "workdir": str(workdir),
            "title": title,
            "source_id": source_id,
            "handoff": str((workdir / "agent_handoff.md").resolve()),
            "writing_contracts": writing_contracts,
            "candidate_frontmatter": str((workdir / FRONTMATTER_NAME).resolve()),
            "body_draft": str((workdir / BODY_NAME).resolve()),
            "worker_result": str((workdir / WORKER_RESULT_NAME).resolve()),
            "finish": {
                "cwd": str(Path(__file__).resolve().parents[1]),
                "argv": [sys.executable, "-m", "ops.raw_fast_batch", "finish-worker", "--contract", str(contract_path)],
            },
        }
        atomic_write_json(contract_path, contract)
        prompt_path.write_text(_render_worker_prompt(contract_path), encoding="utf-8")
        item.pop("error", None)
        item.update(
            {
                "status": "worker_ready",
                "worker_contract": {
                    "path": str(contract_path),
                    "prompt": str(prompt_path),
                    "canonical_writes_allowed": False,
                },
            }
        )
        tasks.append(
            {
                "goal": f"完成 raw-fast batch item {input_index} 的隔离 staging。",
                "context": f"先加载 `llm-wiki`，再严格执行 `{prompt_path}`。只写该 item workdir。",
            }
        )
    manifest["phase"] = "workers_ready" if tasks else "worker_contract_failed"
    _manifest_save(manifest_path, manifest)
    return {
        "ok": bool(tasks),
        "phase": manifest["phase"],
        "manifest_path": str(manifest_path),
        "ready_count": len(tasks),
        "failed_count": len(failures),
        "worker_tasks": tasks,
        "failures": failures,
    }


def finish_worker(contract_path: Path) -> dict[str, Any]:
    contract_path = contract_path.expanduser().resolve()
    contract = _load_json(contract_path)
    if not contract or contract.get("version") != WORKER_CONTRACT_VERSION:
        return {"ok": False, "error": "worker_contract_invalid", "contract_path": str(contract_path)}
    workdir = Path(str(contract.get("workdir") or "")).expanduser().resolve()
    if contract_path != workdir / WORKER_CONTRACT_NAME:
        return {"ok": False, "error": "worker_contract_path_invalid", "contract_path": str(contract_path)}

    expected_paths = {
        "candidate_frontmatter": workdir / FRONTMATTER_NAME,
        "body_draft": workdir / BODY_NAME,
        "worker_result": workdir / WORKER_RESULT_NAME,
    }
    for key, expected in expected_paths.items():
        actual = Path(str(contract.get(key) or "")).expanduser().resolve()
        if actual != expected:
            return {"ok": False, "error": "worker_contract_path_invalid", "field": key, "contract_path": str(contract_path)}
    result_path = expected_paths["worker_result"]
    result_path.unlink(missing_ok=True)

    frontmatter = _load_json(expected_paths["candidate_frontmatter"])
    if not frontmatter:
        return {"ok": False, "error": "candidate_frontmatter_missing", "contract_path": str(contract_path)}
    title = str(frontmatter.get("title") or "").strip()
    source_id = str(frontmatter.get("source") or "").strip()
    if title != str(contract.get("title") or "") or source_id != str(contract.get("source_id") or ""):
        return {
            "ok": False,
            "error": "protected_identity_conflict",
            "expected_title": contract.get("title"),
            "actual_title": title,
            "expected_source_id": contract.get("source_id"),
            "actual_source_id": source_id,
        }
    try:
        body_text = expected_paths["body_draft"].read_text(encoding="utf-8")
    except OSError as exc:
        return {"ok": False, "error": "body_draft_missing", "detail": str(exc)}
    rendered = render_raw_note(frontmatter, body_text)
    if not rendered.get("ok"):
        return {"ok": False, "error": str(rendered.get("error") or "staged_note_invalid"), "validation": rendered}

    handoff = _load_json(workdir / "agent_handoff.json") or {}
    raw_closeout_args = handoff.get("closeout_args")
    closeout_args = raw_closeout_args if isinstance(raw_closeout_args, dict) else {}
    raw_topic_hints = frontmatter.get("topic_hints")
    topic_hints = raw_topic_hints if isinstance(raw_topic_hints, list) else []
    worker_result = {
        "ok": True,
        "status": "ready",
        "source_id": source_id,
        "topic_hints": [str(value) for value in topic_hints if str(value).strip()],
        "required_sections": ["index.md"],
        "resource_status_summary": str(closeout_args.get("resource_status_summary") or "batch staging"),
    }
    atomic_write_json(result_path, worker_result)
    return {
        "ok": True,
        "status": "ready",
        "contract_path": str(contract_path),
        "body_draft": str(expected_paths["body_draft"]),
        "worker_result": str(result_path),
        "source_id": source_id,
    }


def _frontmatter_source(raw_text: str) -> str:
    if not raw_text.startswith("---\n"):
        return ""
    parts = raw_text.split("---", 2)
    if len(parts) != 3:
        return ""
    match = SOURCE_LINE_RE.search(parts[1])
    return match.group(1).strip() if match else ""


def _find_raw_by_source(root: Path, source: str) -> Path | None:
    if not source:
        return None
    raw_root = root / "raw" / "clip"
    if not raw_root.exists():
        return None
    for path in sorted(raw_root.glob("*/*.md")):
        try:
            head = path.read_text(encoding="utf-8", errors="replace")[:16384]
        except OSError:
            continue
        if _frontmatter_source(head) == source:
            return path
    return None


def _worker_payload(item: dict[str, Any], batch_root: Path) -> dict[str, Any]:
    workdir = _safe_child(batch_root, Path(str(item["workdir"])))
    worker = _load_json(workdir / WORKER_RESULT_NAME)
    if worker is None:
        return {"ok": False, "error": "worker_result_missing"}
    if not worker.get("ok") or str(worker.get("status") or "ready") not in {"ready", "completed"}:
        return {"ok": False, "error": str(worker.get("error") or "worker_failed"), "worker_result": worker}
    frontmatter = _load_json(workdir / FRONTMATTER_NAME)
    if not frontmatter:
        return {"ok": False, "error": "candidate_frontmatter_missing"}
    try:
        body_text = (workdir / BODY_NAME).read_text(encoding="utf-8")
    except OSError as exc:
        return {"ok": False, "error": "body_draft_missing", "detail": str(exc)}
    rendered = render_raw_note(frontmatter, body_text)
    if not rendered.get("ok"):
        return {"ok": False, "error": str(rendered.get("error") or "staged_note_invalid"), "validation": rendered}
    title = str(rendered["frontmatter"].get("title") or "").strip()
    source_id = str(rendered["frontmatter"].get("source") or "").strip()
    if not title or not source_id:
        return {"ok": False, "error": "title_or_source_missing"}
    declared_source = str(worker.get("source_id") or "").strip()
    if declared_source and declared_source != source_id:
        return {"ok": False, "error": "worker_source_conflict", "declared_source": declared_source, "rendered_source": source_id}
    topic_hints = worker.get("topic_hints") if isinstance(worker.get("topic_hints"), list) else rendered["frontmatter"].get("topic_hints")
    required_sections = worker.get("required_sections") if isinstance(worker.get("required_sections"), list) else []
    return {
        "ok": True,
        "workdir": workdir,
        "title": title,
        "source_id": source_id,
        "raw_text": str(rendered["raw_text"]),
        "content_sha256": _content_hash(str(rendered["raw_text"])),
        "topic_hints": [str(value) for value in (topic_hints or []) if str(value).strip()],
        "required_sections": [str(value) for value in (required_sections or []) if str(value).strip()],
        "resource_status_summary": str(worker.get("resource_status_summary") or "batch staging"),
    }


def _append_batch_log_locked(
    root: Path,
    state_dir: Path,
    manifest_path: Path,
    items: list[dict[str, Any]],
    wiki_status: dict[str, Any],
) -> list[str]:
    log_path = root / "log.md"
    existing = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else "# llm-wiki log\n"
    entries: list[str] = []
    for item in items:
        raw_file = str(item.get("raw_file") or "")
        if not raw_file or raw_file in existing:
            continue
        args = argparse.Namespace(
            title=str(item.get("title") or ""),
            source_id=str(item.get("source_id") or ""),
            resource_status_summary=str(item.get("resource_status_summary") or "batch staging"),
            raw_file=raw_file,
            root=root,
            state_dir=state_dir,
        )
        output = {
            "raw_fast_ok": True,
            "report_path": str(manifest_path),
            "final_verify": {"report_path": str(manifest_path)},
            "wiki_integration": wiki_status,
            "native_refresh_status": {
                "blocked_by_pending_wiki_integration": bool(wiki_status.get("blocking_pending_count")),
                "graph_ready_pending_count": 0,
                "should_refresh": False,
            },
        }
        entry = build_compact_log_entry(args, output)
        entries.append(entry)
        existing = existing.rstrip() + "\n\n" + entry
    if entries:
        log_path.write_text(existing, encoding="utf-8")
    return entries


def _manifest_save(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _now_stamp()
    atomic_write_json(manifest_path, manifest)


def _report_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    statuses = [str(item.get("status") or "") for item in items]
    return {
        "input_count": len(items),
        "ready_count": sum(bool(item.get("staged_ok")) for item in items),
        "published_count": statuses.count("published"),
        "reused_count": statuses.count("reused") + statuses.count("recovered"),
        "failed_count": sum(status in {"failed", "conflict"} for status in statuses),
    }


def _write_report(batch_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    report_path = batch_root / REPORT_NAME
    report = {**report, "report_path": str(report_path)}
    atomic_write_json(report_path, report)
    return report


def closeout_batch(
    manifest_path: Path,
    *,
    auto_integrate: bool = True,
    runner: str | None = None,
    threshold: int | None = None,
    integration_timeout: int = 7200,
) -> dict[str, Any]:
    manifest_path, manifest = _load_manifest(manifest_path)
    batch_root = Path(str(manifest["batch_root"])).resolve()
    root = Path(str(manifest["root"])).expanduser().resolve()
    state_dir = Path(str(manifest["state_dir"])).expanduser().resolve()
    items = sorted(manifest["items"], key=lambda item: int(item.get("input_index", 0)))
    effective_runner = runner or str(manifest.get("runner") or "local")
    effective_threshold = threshold if threshold is not None else manifest.get("threshold")
    manifest["phase"] = "fan_in"

    staged: dict[int, dict[str, Any]] = {}
    for item in items:
        raw_file = str(item.get("raw_file") or "")
        if raw_file and (root / raw_file).is_file():
            item["staged_ok"] = True
            staged[int(item["input_index"])] = {"ok": True, "recorded": True}
            continue
        worker_payload = _worker_payload(item, batch_root)
        if not worker_payload.get("ok"):
            item.update({"status": "failed", "staged_ok": False, "error": worker_payload.get("error"), "failure": worker_payload})
            continue
        item.update(
            {
                "staged_ok": True,
                "title": worker_payload["title"],
                "source_id": worker_payload["source_id"],
                "resource_status_summary": worker_payload["resource_status_summary"],
            }
        )
        staged[int(item["input_index"])] = worker_payload
    _manifest_save(manifest_path, manifest)

    successful_items: list[dict[str, Any]] = []
    publish_error: dict[str, Any] | None = None
    pending_before: dict[str, Any] = {}
    pending_after_publish: dict[str, Any] = {}
    log_entries: list[str] = []
    try:
        with wiki_mutation_lock(state_dir):
            pending_before = pending_wiki_integration_status(root, state_dir, reason="threshold", threshold=effective_threshold)
            new_items: list[tuple[dict[str, Any], dict[str, Any]]] = []
            source_owners: dict[str, dict[str, Any]] = {}
            duplicate_items: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for item in items:
                payload = staged.get(int(item["input_index"]))
                if not payload:
                    continue
                recorded_raw = str(item.get("raw_file") or "")
                if recorded_raw:
                    validate_raw_relative_path(recorded_raw)
                    raw_path = root / recorded_raw
                    if not raw_path.is_file():
                        raise RawPublishError("recorded_raw_missing", f"manifest raw path is missing: {recorded_raw}", raw_file=recorded_raw)
                    raw_text = raw_path.read_text(encoding="utf-8")
                    actual_hash = _content_hash(raw_text)
                    expected_hash = str(item.get("content_sha256") or "")
                    if expected_hash and actual_hash != expected_hash:
                        raise RawPublishError("recorded_raw_hash_conflict", f"manifest raw hash mismatch: {recorded_raw}", raw_file=recorded_raw)
                    item.update(
                        {
                            "status": "recovered",
                            "content_sha256": actual_hash,
                            "title": item.get("title") or recorded_raw,
                            "source_id": item.get("source_id") or _frontmatter_source(raw_text[:16384]),
                        }
                    )
                    successful_items.append(item)
                    continue

                source_id = str(payload["source_id"])
                existing = _find_raw_by_source(root, source_id)
                if existing is not None:
                    raw_text = existing.read_text(encoding="utf-8")
                    item.update(
                        {
                            "status": "reused",
                            "raw_file": existing.relative_to(root).as_posix(),
                            "content_sha256": _content_hash(raw_text),
                            "title": payload["title"],
                            "source_id": source_id,
                            "resource_status_summary": payload["resource_status_summary"],
                            "topic_hints": payload["topic_hints"],
                            "required_sections": payload["required_sections"],
                        }
                    )
                    source_owners[source_id] = item
                    successful_items.append(item)
                    _manifest_save(manifest_path, manifest)
                    continue
                if source_id in source_owners:
                    duplicate_items.append((item, payload))
                    continue
                source_owners[source_id] = item
                new_items.append((item, payload))

            raw_files = allocate_raw_paths_locked(root, [str(payload["title"]) for _item, payload in new_items])
            for (item, payload), raw_file in zip(new_items, raw_files, strict=True):
                publication = publish_raw_text_exclusive_locked(root, raw_file, str(payload["raw_text"]))
                item.update(
                    {
                        "status": "published",
                        "raw_file": raw_file,
                        "content_sha256": publication["content_sha256"],
                        "published_at": _now_stamp(),
                        "title": payload["title"],
                        "source_id": payload["source_id"],
                        "resource_status_summary": payload["resource_status_summary"],
                        "topic_hints": payload["topic_hints"],
                        "required_sections": payload["required_sections"],
                    }
                )
                successful_items.append(item)
                _manifest_save(manifest_path, manifest)

            for item, payload in duplicate_items:
                owner = source_owners[str(payload["source_id"])]
                item.update(
                    {
                        "status": "reused",
                        "raw_file": owner["raw_file"],
                        "content_sha256": owner["content_sha256"],
                        "title": payload["title"],
                        "source_id": payload["source_id"],
                        "resource_status_summary": payload["resource_status_summary"],
                        "topic_hints": payload["topic_hints"],
                        "required_sections": payload["required_sections"],
                        "duplicate_of_input_index": owner["input_index"],
                    }
                )
                successful_items.append(item)
                _manifest_save(manifest_path, manifest)

            pending_entries = [
                {
                    "raw_path": item["raw_file"],
                    "title": item.get("title") or item["raw_file"],
                    "source_id": item.get("source_id") or "",
                    "topic_hints": item.get("topic_hints") or [],
                    "required_sections": item.get("required_sections") or [],
                    "resource_status_summary": item.get("resource_status_summary") or "batch staging",
                    "status": "raw_saved",
                }
                for item in successful_items
            ]
            mark_pending_wiki_integration_batch(state_dir, root, pending_entries, threshold=effective_threshold)
            pending_after_publish = pending_wiki_integration_status(root, state_dir, reason="threshold", threshold=effective_threshold)
            log_entries = _append_batch_log_locked(root, state_dir, manifest_path, successful_items, pending_after_publish)
            manifest["phase"] = "published"
            _manifest_save(manifest_path, manifest)
    except (OSError, ValueError, RawPublishError) as exc:
        code = exc.code if isinstance(exc, RawPublishError) else type(exc).__name__
        details = exc.details if isinstance(exc, RawPublishError) else {}
        publish_error = {"error": code, "detail": str(exc), **details}
        manifest["phase"] = "publish_failed"
        manifest["failure"] = publish_error
        _manifest_save(manifest_path, manifest)

    integration_attempted = False
    integration_code: int | None = None
    integration_result: dict[str, Any] | None = None
    integration_deferred_reason: str | None = None
    if publish_error is None:
        if not auto_integrate:
            integration_deferred_reason = "auto_integration_disabled"
        elif effective_runner != "local":
            integration_deferred_reason = "external_runner_deferred"
        elif pending_after_publish.get("should_review"):
            integration_deferred_reason = "manual_review_required"
        elif pending_after_publish.get("should_integrate"):
            integration_attempted = True
            integration_code, integration_result = run_auto_integration(
                root,
                state_dir,
                reason="threshold",
                threshold=effective_threshold,
                timeout=integration_timeout,
                runner="local",
                defer_native_refresh=True,
            )
        else:
            integration_deferred_reason = "threshold_not_reached"

    final_wiki = pending_wiki_integration_status(root, state_dir, reason="threshold", threshold=effective_threshold)
    counts = _report_counts(items)
    successful_count = counts["published_count"] + counts["reused_count"]
    if successful_count == counts["input_count"]:
        raw_status = "complete"
    elif successful_count:
        raw_status = "partial"
    else:
        raw_status = "failed"
    if integration_code not in {None, 0}:
        wiki_status = "integration_failed"
    elif final_wiki.get("should_review"):
        wiki_status = "needs_review"
    elif final_wiki.get("blocking_pending_count"):
        wiki_status = "pending"
    else:
        wiki_status = "integrated"
    ok = publish_error is None and integration_code in {None, 0}
    manifest["phase"] = "closed" if ok else ("integration_failed" if publish_error is None else "publish_failed")
    _manifest_save(manifest_path, manifest)
    report = {
        "ok": ok,
        "batch_id": manifest["batch_id"],
        "phase": manifest["phase"],
        **counts,
        "raw_status": raw_status,
        "wiki_status": wiki_status,
        "graph_status": "pending" if successful_count else "unchanged",
        "raw_files": [str(item.get("raw_file")) for item in items if item.get("raw_file")],
        "wiki": {
            "pending_before": pending_before.get("pending_count", 0),
            "pending_after_publish": pending_after_publish.get("pending_count", 0),
            "pending_final": final_wiki.get("pending_count", 0),
            "threshold": final_wiki.get("threshold", effective_threshold),
            "decision_count": 1 if publish_error is None else 0,
            "integration_attempted": integration_attempted,
            "integration_runner": effective_runner,
            "integration_returncode": integration_code,
            "integration_deferred_reason": integration_deferred_reason,
            "integration_run_id": (
                str((integration_result or {}).get("plan_path") or (integration_result or {}).get("prompt_path") or "") or None
            ),
        },
        "log_entries_appended": len(log_entries),
        "failure": publish_error or ((integration_result or {}).get("failure") if integration_code not in {None, 0} else None),
        "items": items,
    }
    return _write_report(batch_root, report)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage and close out parallel raw-fast clipping batches")
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init", help="Create one manifest and isolated item workdirs")
    init_parser.add_argument("--root", required=True, type=Path)
    init_parser.add_argument("--state-dir", required=True, type=Path)
    init_parser.add_argument("--tmp-root", type=Path, default=DEFAULT_TMP_ROOT)
    init_parser.add_argument("--batch-id", default=None)
    init_parser.add_argument("--url", action="append", required=True)
    init_parser.add_argument("--threshold", type=int, default=None)
    init_parser.add_argument("--runner", choices=["local", "external"], default="local")

    contracts_parser = sub.add_parser("worker-contracts", help="Materialize concise worker contracts after item prepares")
    contracts_parser.add_argument("--manifest", required=True, type=Path)

    finish_parser = sub.add_parser("finish-worker", help="Validate one staged worker draft and write worker_result.json")
    finish_parser.add_argument("--contract", required=True, type=Path)

    closeout_parser = sub.add_parser("closeout", help="Fan in staged items and perform one canonical closeout")
    closeout_parser.add_argument("--manifest", required=True, type=Path)
    closeout_parser.add_argument("--auto-integrate", action=argparse.BooleanOptionalAction, default=True)
    closeout_parser.add_argument("--runner", choices=["local", "external"], default=None)
    closeout_parser.add_argument("--threshold", type=int, default=None)
    closeout_parser.add_argument("--integration-timeout", type=int, default=7200)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "init":
            payload = init_batch(
                root=args.root,
                state_dir=args.state_dir,
                urls=args.url,
                tmp_root=args.tmp_root,
                batch_id=args.batch_id,
                threshold=args.threshold,
                runner=args.runner,
            )
        elif args.command == "worker-contracts":
            payload = materialize_worker_contracts(args.manifest)
        elif args.command == "finish-worker":
            payload = finish_worker(args.contract)
        else:
            payload = closeout_batch(
                args.manifest,
                auto_integrate=args.auto_integrate,
                runner=args.runner,
                threshold=args.threshold,
                integration_timeout=args.integration_timeout,
            )
    except (OSError, ValueError) as exc:
        payload = {"ok": False, "error": type(exc).__name__, "detail": str(exc)}
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
