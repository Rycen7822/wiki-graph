#!/usr/bin/env python3
from wiki_lightrag_lib import common_paths_parser, extract_raw_sections, print_json


def main() -> int:
    parser = common_paths_parser("Extract section-level raw-note virtual docs for LightRAG retrieval")
    args = parser.parse_args()
    result = extract_raw_sections(args.root, args.state_dir)
    print_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
