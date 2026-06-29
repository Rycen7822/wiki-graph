#!/usr/bin/env python3
from __future__ import annotations

from wiki_native_lib import common_paths_parser, print_json


def unsupported_document_sync_payload(script_name: str) -> dict[str, object]:
    return {
        "ok": False,
        "error": "unsupported_native_document_sync",
        "script": script_name,
        "operation": "document_sync",
        "unsupported_endpoint": "/documents/texts",
        "message": "Generated-document ingestion is disabled for the native-zvec production backend.",
        "native_refresh": {
            "command": "scripts/batch_native_refresh.py refresh --prepare-only",
            "cutover_command": "scripts/batch_native_refresh.py refresh --cutover",
            "status_command": "scripts/batch_native_refresh.py status",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = common_paths_parser("Unsupported generated-doc sync entrypoint")
    parser.add_argument("--kind", choices=["all", "edge", "method_atom", "raw_section"], default="all")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    parser.parse_args(argv)
    print_json(unsupported_document_sync_payload("sync_virtual_docs.py"))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
