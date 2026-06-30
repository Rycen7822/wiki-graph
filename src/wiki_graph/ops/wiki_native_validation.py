#!/usr/bin/env python3
"""Native wiki validation helpers."""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any


def secret_hits(path: Path, text: str) -> list[str]:
    hits = []
    pattern = re.compile(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*['\"]?(?!\[REDACTED\])([A-Za-z0-9_\-]{28,})")
    for match in pattern.finditer(text):
        hits.append(f"{path.as_posix()}:{match.start()}: {match.group(1)}=[REDACTED]")
    return hits


def validate_wiki(
    root: Path,
    state_dir: Path,
    workdir: Path | None = None,
    full: bool = False,
    write_report: bool = False,
) -> dict[str, Any]:
    import wiki_graph.ops.wiki_native_lib as lib

    root = root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    compiled = lib.indexed_markdown_pages(root)
    non_meta_compiled = lib.compiled_pages(root)
    index_total, index_wikilinks = lib.index_stats(root)
    compiled_count = len(compiled)
    if index_total is None:
        errors.append("index.md missing Total pages marker")
    elif compiled_count != index_total:
        errors.append(f"compiled_count != index_total ({compiled_count} != {index_total})")
    if index_total is not None and index_wikilinks != index_total:
        errors.append(f"index_wikilinks != index_total ({index_wikilinks} != {index_total})")
    slug_set = {p.stem for p in compiled}
    slug_set.update({"index", "raw-clip-map", "topic-map", "source-map"})
    broken, missing_sources, image_issues, raw_filename_drift = [], [], [], []
    for path in non_meta_compiled:
        text = lib.read_text(path)
        fm, body = lib.parse_frontmatter(text)
        rel = path.relative_to(root).as_posix()
        expected_type = lib.COMPILED_DIR_TYPES[path.parent.name]
        for key in ["title", "type", "tags", "updated"]:
            if key not in fm:
                errors.append(f"{rel} missing frontmatter.{key}")
        if fm.get("type") != expected_type:
            errors.append(f"{rel} type {fm.get('type')!r} != {expected_type!r}")
        for link in lib.find_wikilinks(text):
            if link not in slug_set:
                broken.append(f"{rel} -> {link}")
        for src in lib.as_list(fm.get("sources")):
            s = lib.display_scalar(src)
            if not s or s.startswith("http://") or s.startswith("https://"):
                continue
            resolved = lib.resolve_source(path, s, root)
            if resolved and not resolved.exists():
                missing_sources.append(f"{rel} -> {s}")
        if re.search(r"!\[[^\]]*\]\(https?://", text) or "data:image" in text:
            image_issues.append(rel)
        warnings.extend(secret_hits(path, text))
    raw_files = lib.raw_clip_files(root)
    for raw in raw_files:
        rel = raw.relative_to(root).as_posix()
        if " " in raw.name or "-md.md" in raw.name or raw.name.count(".md") != 1:
            raw_filename_drift.append(rel)
        raw_text = lib.read_text(raw)
        if full:
            warnings.extend(lib.structured_heading_warnings(raw, raw_text))
            warnings.extend(secret_hits(raw, raw_text))
    raw_count, map_count = len(raw_files), None
    raw_map = root / "_meta" / "raw-clip-map.md"
    if raw_map.exists():
        m = re.search(r"Active raw clips[：:]\s*(\d+)", lib.read_text(raw_map))
        if m:
            map_count = int(m.group(1))
            if map_count != raw_count:
                warnings.append(f"active_raw_clips != raw-clip-map snapshot ({raw_count} != {map_count})")
    pollution = lib.wiki_root_machine_pollution(root)
    if pollution:
        errors.append("wiki_root_machine_pollution detected: " + ", ".join(p.as_posix() for p in pollution))
    log_path = root / "log.md"
    if log_path.exists():
        log_entries = len(re.findall(r"^##\s+", lib.read_text(log_path), re.M))
        if log_entries > 500:
            warnings.append(f"log.md has {log_entries} entries; rotate warning")
    if broken:
        errors.append(f"broken_wikilinks={len(broken)}")
    if missing_sources:
        errors.append(f"missing_source_refs={len(missing_sources)}")
    if raw_filename_drift:
        errors.append(f"raw_filename_drift={len(raw_filename_drift)}")
    if image_issues:
        errors.append(f"image_hygiene={len(image_issues)}")
    freshness_contract: dict[str, Any] = {}
    if write_report:
        valid_for_reasons = ["wiki-clear-success", "final-status"]
        if full:
            valid_for_reasons.append("refresh-artifact")
        freshness_contract = {
            **lib.validation_freshness_context(root, state_dir, workdir),
            "covered_surfaces": ["index", "compiled", "_meta", "raw"],
            "valid_for_reasons": valid_for_reasons,
        }
    report = {
        **freshness_contract,
        "generated_at": lib.now_stamp(),
        "compiled_count": compiled_count,
        "index_total": index_total,
        "index_wikilinks": index_wikilinks,
        "active_raw_clips": raw_count,
        "raw_clip_map_snapshot": map_count,
        "broken_wikilinks": len(broken),
        "broken_wikilink_examples": broken[:20],
        "missing_source_refs": len(missing_sources),
        "missing_source_ref_examples": missing_sources[:20],
        "raw_filename_drift": len(raw_filename_drift),
        "raw_filename_drift_examples": raw_filename_drift[:20],
        "native_unresolved_references": 0,
        "wiki_root_machine_pollution": len(pollution),
        "wiki_root_machine_pollution_paths": [p.as_posix() for p in pollution],
        "native_state_dir": str(state_dir.resolve()),
        "native_workdir": str(workdir.resolve()) if workdir else None,
        "warnings": warnings[:200],
        "errors": errors,
    }
    if write_report:
        lib.ensure_state_dirs(state_dir)
        out = state_dir / "validation_reports" / f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_validate.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["report_path"] = str(out)
    return report
