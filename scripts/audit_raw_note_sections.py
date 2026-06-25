#!/usr/bin/env python3
"""Audit raw-note section headings for section-level LightRAG retrieval."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from wiki_lightrag_lib import (
    audit_raw_note_section_contracts,
    common_paths_parser,
    ensure_state_dirs,
    print_json,
)


def main() -> int:
    parser = common_paths_parser("Audit raw-note section headings for LightRAG raw-section retrieval")
    parser.add_argument(
        "--structured-only",
        action="store_true",
        help="Only report issues for explicit structured raw paper notes; still reports raw/structured counts.",
    )
    parser.add_argument(
        "--limit-issues",
        type=int,
        default=None,
        help="Return only the first N issue rows in stdout/report while preserving issue_count totals.",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write JSON report under state/raw_section_audits/.",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Exit non-zero if warning-severity issues are present.",
    )
    args = parser.parse_args()

    result = audit_raw_note_section_contracts(
        args.root,
        include_legacy=not args.structured_only,
        issue_limit=args.limit_issues,
    )
    if args.write_report:
        ensure_state_dirs(args.state_dir)
        report_dir = args.state_dir / "raw_section_audits"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_raw_section_audit.json"
        report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["report_path"] = report_path.as_posix()
    print_json(result)
    if args.fail_on_warning and result.get("issues_by_severity", {}).get("warning", 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
