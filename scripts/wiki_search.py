#!/usr/bin/env python3
import json
from pathlib import Path

from wiki_lightrag_lib import (
    add_query_event,
    common_paths_parser,
    expand_lightrag_data_response_with_section_neighbors,
    filter_lightrag_data_response_by_section_kind,
    load_lightrag_api_key,
    print_json,
    query_lightrag,
    query_lightrag_data,
    RAW_NOTE_CONTRACT_SECTION_KINDS,
    raw_section_query_for_kind,
    save_evidence_pack,
)


def run_query(args, query: str) -> dict:
    api_key = load_lightrag_api_key(args.workdir)
    section_kind = (args.section_kind or "").strip().lower()
    effective_query = raw_section_query_for_kind(section_kind, query) if section_kind else query
    top_k = max(args.top_k, 40) if section_kind else args.top_k
    chunk_top_k = max(args.chunk_top_k, 40) if section_kind else args.chunk_top_k
    query_func = query_lightrag_data if args.data_only else query_lightrag
    response = query_func(args.server, api_key, effective_query, mode=args.mode, top_k=top_k, chunk_top_k=chunk_top_k)
    if section_kind and args.data_only:
        response = filter_lightrag_data_response_by_section_kind(response, section_kind)
    if args.expand_section_neighbors:
        if not args.data_only:
            raise ValueError("--expand-section-neighbors requires --data-only so direct hits and expansions remain explicit")
        response = expand_lightrag_data_response_with_section_neighbors(response, args.state_dir, neighbor_k=args.neighbor_k, section_kind=section_kind or None)
    pack = None
    if args.save_evidence_pack:
        pack = save_evidence_pack(args.state_dir, effective_query, args.mode, response)
    add_query_event(args.state_dir, effective_query, args.mode, str(pack) if pack else None)
    return {"query": effective_query, "mode": args.mode, "section_kind": section_kind or None, "evidence_pack": str(pack) if pack else None, "response": response}


def main() -> int:
    parser = common_paths_parser("Query LightRAG and optionally save an evidence pack")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--mode", default="mix", choices=["local", "global", "hybrid", "naive", "mix", "bypass"])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--chunk-top-k", type=int, default=10)
    parser.add_argument(
        "--section-kind",
        choices=RAW_NOTE_CONTRACT_SECTION_KINDS,
        help="Prefix a raw_section query and, with --data-only, return only matching raw_section_docs chunks.",
    )
    parser.add_argument("--intent", default="query")
    parser.add_argument("--driver", default="hermes-current")
    parser.add_argument("--save-evidence-pack", action="store_true")
    parser.add_argument("--data-only", action="store_true", help="Use /query/data retrieval without LLM answer generation")
    parser.add_argument("--expand-section-neighbors", action="store_true", help="With --data-only, append reviewed semantic section-neighbor expansions from state/section_similarity_edges.jsonl")
    parser.add_argument("--neighbor-k", type=int, default=5, help="Max semantic section-neighbor expansions per direct raw-section hit")
    parser.add_argument("--benchmark", type=Path)
    args = parser.parse_args()
    if args.benchmark:
        rows = []
        for line in args.benchmark.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(run_query(args, item.get("query") or item.get("q") or line))
        print_json({"benchmark": str(args.benchmark), "runs": rows})
        return 0
    if not args.query:
        parser.error("query is required unless --benchmark is set")
    print_json(run_query(args, args.query))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
