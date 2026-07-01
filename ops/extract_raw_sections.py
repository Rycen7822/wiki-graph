#!/usr/bin/env python3
from ops.wiki_native_cli import common_paths_parser, print_json
from ops.wiki_native_raw_section_extract import extract_raw_sections


def main() -> int:
    parser = common_paths_parser("Extract section-level raw-note virtual docs for native retrieval")
    args = parser.parse_args()
    result = extract_raw_sections(args.root, args.state_dir)
    print_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
