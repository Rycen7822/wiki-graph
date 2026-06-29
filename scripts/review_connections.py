#!/usr/bin/env python3
import json
from pathlib import Path

from wiki_native_lib import common_paths_parser, ensure_state_dirs, print_json


def main() -> int:
    parser = common_paths_parser("Review pending llm-wiki native graph connection candidates")
    parser.add_argument("--list", choices=["pending", "accepted", "rejected", "merged", "stale"], default="pending")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    ensure_state_dirs(args.state_dir)
    queue = args.state_dir / "connection_review_queue.jsonl"
    if not queue.exists():
        queue.write_text("", encoding="utf-8")
    rows = []
    for line in queue.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("status", "pending") == args.list:
            rows.append(item)
    print_json({"queue": str(queue), "status": args.list, "count": len(rows), "items": rows[: args.top]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
