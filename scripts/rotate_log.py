#!/usr/bin/env python3
from pathlib import Path

from wiki_lightrag_lib import common_paths_parser, now_stamp, print_json, read_text


def main() -> int:
    parser = common_paths_parser("Warn or rotate llm-wiki log.md when it grows too large")
    parser.add_argument("--max-entries", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    log = args.root / "log.md"
    if not log.exists():
        print_json({"log": str(log), "exists": False})
        return 0
    text = read_text(log)
    entries = text.count("\n## ")
    result = {"log": str(log), "entries": entries, "max_entries": args.max_entries, "rotated": False}
    if entries <= args.max_entries or args.dry_run:
        print_json(result)
        return 0
    archive = args.root / "raw" / "_archive" / "logs"
    archive.mkdir(parents=True, exist_ok=True)
    out = archive / f"log-archive-{now_stamp().replace(':', '').replace(' ', '-')}.md"
    out.write_text(text, encoding="utf-8")
    log.write_text("# llm-wiki log\n\n> Older log archived to `" + out.relative_to(args.root).as_posix() + "`.\n", encoding="utf-8")
    result.update({"rotated": True, "archive": str(out)})
    print_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
