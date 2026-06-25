#!/usr/bin/env python3
from wiki_lightrag_lib import common_paths_parser, print_json, validate_wiki


def main() -> int:
    parser = common_paths_parser("Validate llm-wiki before/after LightRAG sync")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--allow-errors", action="store_true")
    args = parser.parse_args()
    report = validate_wiki(args.root, args.state_dir, args.workdir, full=args.full, write_report=args.write_report)
    print_json(report)
    return 0 if args.allow_errors or not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
