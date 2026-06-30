#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Any

from wiki_graph.ops.wiki_native_lib import (
    common_paths_parser,
    print_json,
    validate_wiki,
    validation_freshness_context,
    validation_report_is_fresh,
)


REQUIRED_VALIDATION_REUSE_SURFACES = ["index", "compiled", "_meta", "raw"]


def validation_reuse_reason(full: bool) -> str:
    return "refresh-artifact" if full else "wiki-clear-success"


def validation_reuse_status(path: Path, *, fresh: bool, rejections: list[str] | None = None) -> dict[str, Any]:
    status: dict[str, Any] = {"fresh": fresh, "path": str(path.resolve())}
    if rejections is not None:
        status["rejections"] = rejections
    return status


def load_reusable_validation_report(path: Path, root: Path, state_dir: Path, workdir: Path | None, *, full: bool) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return None, validation_reuse_status(path, fresh=False, rejections=["reuse_report_unreadable"])
    except json.JSONDecodeError:
        return None, validation_reuse_status(path, fresh=False, rejections=["reuse_report_invalid_json"])
    if not isinstance(report, dict):
        return None, validation_reuse_status(path, fresh=False, rejections=["reuse_report_not_object"])

    current = validation_freshness_context(root, state_dir, workdir)
    freshness = validation_report_is_fresh(
        report,
        current,
        required_surfaces=REQUIRED_VALIDATION_REUSE_SURFACES,
        reason=validation_reuse_reason(full),
    )
    if not freshness["fresh"]:
        return None, validation_reuse_status(path, fresh=False, rejections=list(freshness["rejections"]))

    reused = dict(report)
    status = validation_reuse_status(path, fresh=True)
    reused["validation_reuse"] = status
    reused["reused_validation_report"] = str(path.resolve())
    return reused, status


def main() -> int:
    parser = common_paths_parser("Validate llm-wiki before/after native refresh")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument("--reuse-validation-report", type=Path)
    args = parser.parse_args()

    reuse_status = None
    if args.reuse_validation_report:
        reused_report, reuse_status = load_reusable_validation_report(
            args.reuse_validation_report,
            args.root,
            args.state_dir,
            args.workdir,
            full=args.full,
        )
        if reused_report is not None:
            print_json(reused_report)
            return 0 if args.allow_errors or not reused_report["errors"] else 1

    report = validate_wiki(args.root, args.state_dir, args.workdir, full=args.full, write_report=args.write_report)
    if reuse_status is not None:
        report["validation_reuse"] = reuse_status
    print_json(report)
    return 0 if args.allow_errors or not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
