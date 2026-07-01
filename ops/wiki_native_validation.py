#!/usr/bin/env python3
"""Native wiki validation helpers."""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

from llm_wiki_native.source_docs import COMPILED_DIR_TYPES, display_scalar, parse_frontmatter, raw_clip_files, read_text
from ops.wiki_native_artifacts import resolve_source
from ops.wiki_native_ingest_text import as_list, find_wikilinks
from ops.wiki_native_state import ensure_state_dirs
from ops.wiki_native_wiki_checks import (
    compiled_pages,
    index_stats,
    indexed_markdown_pages,
    now_stamp,
    structured_heading_warnings,
    validation_freshness_context,
    wiki_root_machine_pollution,
)

RAW_CLIP_MAP_SNAPSHOT_RE = re.compile(r"(Active raw clips[：:]\s*)(\d+)")


def secret_hits(path: Path, text: str) -> list[str]:
    hits = []
    pattern = re.compile(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*['\"]?(?!\[REDACTED\])([A-Za-z0-9_\-]{28,})")
    for match in pattern.finditer(text):
        hits.append(f"{path.as_posix()}:{match.start()}: {match.group(1)}=[REDACTED]")
    return hits


def raw_clip_map_snapshot_count(root: Path) -> int | None:
    raw_map = Path(root) / "_meta" / "raw-clip-map.md"
    if not raw_map.exists():
        return None
    match = RAW_CLIP_MAP_SNAPSHOT_RE.search(read_text(raw_map))
    return int(match.group(2)) if match else None


def sync_raw_clip_map_snapshot(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    raw_map = root / "_meta" / "raw-clip-map.md"
    active_raw_clips = len(raw_clip_files(root))
    base: dict[str, Any] = {
        "path": str(raw_map),
        "active_raw_clips": active_raw_clips,
        "snapshot_found": False,
        "previous_snapshot": None,
        "changed": False,
    }
    if not raw_map.exists():
        return {**base, "ok": True, "reason": "raw_clip_map_missing"}
    text = read_text(raw_map)
    match = RAW_CLIP_MAP_SNAPSHOT_RE.search(text)
    if not match:
        return {**base, "ok": True, "reason": "raw_clip_map_snapshot_marker_missing"}
    previous_snapshot = int(match.group(2))
    if previous_snapshot == active_raw_clips:
        return {
            **base,
            "ok": True,
            "snapshot_found": True,
            "previous_snapshot": previous_snapshot,
            "current_snapshot": active_raw_clips,
            "reason": "already_current",
        }

    def replace_snapshot(snapshot_match: re.Match[str]) -> str:
        return f"{snapshot_match.group(1)}{active_raw_clips}"

    raw_map.write_text(RAW_CLIP_MAP_SNAPSHOT_RE.sub(replace_snapshot, text, count=1), encoding="utf-8")
    return {
        **base,
        "ok": True,
        "snapshot_found": True,
        "previous_snapshot": previous_snapshot,
        "current_snapshot": active_raw_clips,
        "changed": True,
        "reason": "snapshot_synchronized",
    }


def _validate_index_surface(root: Path, errors: list[str]) -> dict[str, Any]:
    compiled = indexed_markdown_pages(root)
    index_total, index_wikilinks = index_stats(root)
    compiled_count = len(compiled)
    if index_total is None:
        errors.append("index.md missing Total pages marker")
    elif compiled_count != index_total:
        errors.append(f"compiled_count != index_total ({compiled_count} != {index_total})")
    if index_total is not None and index_wikilinks != index_total:
        errors.append(f"index_wikilinks != index_total ({index_wikilinks} != {index_total})")
    slug_set = {p.stem for p in compiled}
    slug_set.update({"index", "raw-clip-map", "topic-map", "source-map"})
    return {
        "compiled_count": compiled_count,
        "index_total": index_total,
        "index_wikilinks": index_wikilinks,
        "slug_set": slug_set,
    }


def _validate_compiled_surface(root: Path, slug_set: set[str], errors: list[str], warnings: list[str]) -> dict[str, Any]:
    broken: list[str] = []
    missing_sources: list[str] = []
    image_issues: list[str] = []
    for path in compiled_pages(root):
        text = read_text(path)
        fm, _body = parse_frontmatter(text)
        rel = path.relative_to(root).as_posix()
        expected_type = COMPILED_DIR_TYPES[path.parent.name]
        for key in ["title", "type", "tags", "updated"]:
            if key not in fm:
                errors.append(f"{rel} missing frontmatter.{key}")
        if fm.get("type") != expected_type:
            errors.append(f"{rel} type {fm.get('type')!r} != {expected_type!r}")
        for link in find_wikilinks(text):
            if link not in slug_set:
                broken.append(f"{rel} -> {link}")
        for src in as_list(fm.get("sources")):
            source = display_scalar(src)
            if not source or source.startswith("http://") or source.startswith("https://"):
                continue
            resolved = resolve_source(path, source, root)
            if resolved and not resolved.exists():
                missing_sources.append(f"{rel} -> {source}")
        if re.search(r"!\[[^\]]*\]\(https?://", text) or "data:image" in text:
            image_issues.append(rel)
        warnings.extend(secret_hits(path, text))
    if broken:
        errors.append(f"broken_wikilinks={len(broken)}")
    if missing_sources:
        errors.append(f"missing_source_refs={len(missing_sources)}")
    if image_issues:
        errors.append(f"image_hygiene={len(image_issues)}")
    return {
        "broken_wikilinks": len(broken),
        "broken_wikilink_examples": broken[:20],
        "missing_source_refs": len(missing_sources),
        "missing_source_ref_examples": missing_sources[:20],
        "image_hygiene": len(image_issues),
    }


def _validate_raw_surface(root: Path, full: bool, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    raw_filename_drift: list[str] = []
    raw_files = raw_clip_files(root)
    for raw in raw_files:
        rel = raw.relative_to(root).as_posix()
        if " " in raw.name or "-md.md" in raw.name or raw.name.count(".md") != 1:
            raw_filename_drift.append(rel)
        if full:
            raw_text = read_text(raw)
            warnings.extend(structured_heading_warnings(raw, raw_text))
            warnings.extend(secret_hits(raw, raw_text))
    raw_count, map_count = len(raw_files), None
    raw_map = root / "_meta" / "raw-clip-map.md"
    if raw_map.exists():
        map_count = raw_clip_map_snapshot_count(root)
        if map_count is not None:
            if map_count != raw_count:
                warnings.append(f"active_raw_clips != raw-clip-map snapshot ({raw_count} != {map_count})")
    if raw_filename_drift:
        errors.append(f"raw_filename_drift={len(raw_filename_drift)}")
    return {
        "active_raw_clips": raw_count,
        "raw_clip_map_snapshot": map_count,
        "raw_filename_drift": len(raw_filename_drift),
        "raw_filename_drift_examples": raw_filename_drift[:20],
    }


def _validate_root_hygiene_surface(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    pollution = wiki_root_machine_pollution(root)
    if pollution:
        errors.append("wiki_root_machine_pollution detected: " + ", ".join(p.as_posix() for p in pollution))
    log_path = root / "log.md"
    if log_path.exists():
        log_entries = len(re.findall(r"^##\s+", read_text(log_path), re.M))
        if log_entries > 500:
            warnings.append(f"log.md has {log_entries} entries; rotate warning")
    return {
        "wiki_root_machine_pollution": len(pollution),
        "wiki_root_machine_pollution_paths": [p.as_posix() for p in pollution],
    }


def _validation_freshness_contract(root: Path, state_dir: Path, workdir: Path | None, full: bool, write_report: bool) -> dict[str, Any]:
    if not write_report:
        return {}
    valid_for_reasons = ["wiki-clear-success", "final-status"]
    if full:
        valid_for_reasons.append("refresh-artifact")
    return {
        **validation_freshness_context(root, state_dir, workdir),
        "covered_surfaces": ["index", "compiled", "_meta", "raw"],
        "valid_for_reasons": valid_for_reasons,
    }


def _write_validation_report(report: dict[str, Any], state_dir: Path) -> Path:
    ensure_state_dirs(state_dir)
    out = state_dir / "validation_reports" / f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_validate.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def validate_wiki(
    root: Path,
    state_dir: Path,
    workdir: Path | None = None,
    full: bool = False,
    write_report: bool = False,
    sync_raw_map_snapshot: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    raw_map_sync = sync_raw_clip_map_snapshot(root) if sync_raw_map_snapshot else None
    errors: list[str] = []
    warnings: list[str] = []

    index_report = _validate_index_surface(root, errors)
    compiled_report = _validate_compiled_surface(root, index_report["slug_set"], errors, warnings)
    raw_report = _validate_raw_surface(root, full, errors, warnings)
    hygiene_report = _validate_root_hygiene_surface(root, errors, warnings)
    freshness_contract = _validation_freshness_contract(root, state_dir, workdir, full, write_report)

    report = {
        **freshness_contract,
        "generated_at": now_stamp(),
        "compiled_count": index_report["compiled_count"],
        "index_total": index_report["index_total"],
        "index_wikilinks": index_report["index_wikilinks"],
        "active_raw_clips": raw_report["active_raw_clips"],
        "raw_clip_map_snapshot": raw_report["raw_clip_map_snapshot"],
        "broken_wikilinks": compiled_report["broken_wikilinks"],
        "broken_wikilink_examples": compiled_report["broken_wikilink_examples"],
        "missing_source_refs": compiled_report["missing_source_refs"],
        "missing_source_ref_examples": compiled_report["missing_source_ref_examples"],
        "raw_filename_drift": raw_report["raw_filename_drift"],
        "raw_filename_drift_examples": raw_report["raw_filename_drift_examples"],
        "native_unresolved_references": 0,
        "wiki_root_machine_pollution": hygiene_report["wiki_root_machine_pollution"],
        "wiki_root_machine_pollution_paths": hygiene_report["wiki_root_machine_pollution_paths"],
        "native_state_dir": str(state_dir.resolve()),
        "native_workdir": str(workdir.resolve()) if workdir else None,
        "warnings": warnings[:200],
        "errors": errors,
    }
    if raw_map_sync is not None:
        report["raw_clip_map_sync"] = raw_map_sync
    if write_report:
        report["report_path"] = str(_write_validation_report(report, state_dir))
    return report
