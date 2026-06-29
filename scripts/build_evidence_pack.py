#!/usr/bin/env python3
"""Deprecated alias for ``wiki_search.py --save-evidence-pack``."""

from wiki_native_lib import common_paths_parser, print_json
from wiki_search import run_query


def main() -> int:
    parser = common_paths_parser("Deprecated alias: use wiki_search.py --save-evidence-pack")
    parser.add_argument("query")
    parser.add_argument("--mode", default="mix")
    parser.add_argument("--data-only", action="store_true", help="Use /query/data retrieval without LLM answer generation")
    args = parser.parse_args()
    args.top_k = 20
    args.chunk_top_k = 10
    args.section_kind = None
    args.save_evidence_pack = True
    args.expand_section_neighbors = False
    args.neighbor_k = 5
    result = run_query(args, args.query)
    print_json({"evidence_pack": result["evidence_pack"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
