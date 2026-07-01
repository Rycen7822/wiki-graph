#!/usr/bin/env python3
from llm_wiki_native.source_docs import collect_source_docs
from ops.wiki_native_cli import common_paths_parser, print_json


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
