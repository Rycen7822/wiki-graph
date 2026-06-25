#!/usr/bin/env python3
from wiki_lightrag_lib import common_paths_parser, generated_docs_from_state, print_json, sync_docs_to_lightrag


def main() -> int:
    parser = common_paths_parser("Sync generated edge, MethodAtom, and raw-section docs into LightRAG")
    parser.add_argument("--kind", choices=["all", "edge", "method_atom", "raw_section"], default="all")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    docs = generated_docs_from_state(args.state_dir, args.kind)
    if args.limit:
        docs = docs[: args.limit]
    result = sync_docs_to_lightrag(
        docs,
        args.state_dir,
        args.server,
        args.workdir,
        full=args.full,
        force=args.force,
        batch_size=args.batch_size,
        wait=not args.no_wait,
        timeout_s=args.timeout,
    )
    print_json(result)
    return 0 if not result.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
