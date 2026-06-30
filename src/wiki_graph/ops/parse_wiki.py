#!/usr/bin/env python3
from wiki_graph.ops.wiki_native_lib import collect_source_docs, common_paths_parser, print_json


def main() -> int:
    parser = common_paths_parser("Parse llm-wiki source documents and print a compact summary")
    args = parser.parse_args()
    docs = collect_source_docs(args.root)
    by_type = {}
    for doc in docs:
        by_type[doc.doc_type] = by_type.get(doc.doc_type, 0) + 1
    print_json({"root": str(args.root), "docs": len(docs), "by_type": by_type})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
