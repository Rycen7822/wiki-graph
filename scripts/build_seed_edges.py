#!/usr/bin/env python3
from wiki_lightrag_lib import build_seed_edges, common_paths_parser, print_json


def main() -> int:
    parser = common_paths_parser("Build deterministic high-confidence llm-wiki seed edges")
    args = parser.parse_args()
    result = build_seed_edges(args.root, args.state_dir)
    print_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
