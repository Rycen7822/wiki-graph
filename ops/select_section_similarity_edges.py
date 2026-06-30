#!/usr/bin/env python3
"""Select reviewed section-similarity candidates for custom_kg import."""

from __future__ import annotations

import json
from pathlib import Path

from ops.wiki_native_lib import (
    common_paths_parser,
    ensure_state_dirs,
    jsonl_read,
    jsonl_write,
    print_json,
    select_section_similarity_edges,
)

DEFAULT_PAIR_KINDS = [
    "methodology:methodology",
    "results:results",
    "future:future",
    "limitations:limitations",
    "questions:questions",
    "limitations:questions",
    "future:limitations",
    "future:questions",
    "methodology:future",
    "methodology:results",
    "results:future",
    "results:limitations",
]


def main() -> int:
    parser = common_paths_parser("Select reviewed section-similarity edges for custom_kg import")
    parser.add_argument("--candidate-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--pair-kinds", default=",".join(DEFAULT_PAIR_KINDS))
    args = parser.parse_args()
    ensure_state_dirs(args.state_dir)
    candidate_path = args.candidate_path or (args.state_dir / "section_similarity_edges.candidates.jsonl")
    output_path = args.output_path or (args.state_dir / "section_similarity_edges.jsonl")
    candidates = jsonl_read(candidate_path)
    allowed = {item.strip() for item in args.pair_kinds.split(",") if item.strip()}
    selected = select_section_similarity_edges(candidates, allowed)
    jsonl_write(output_path, selected)
    counts: dict[str, int] = {}
    for edge in selected:
        pair = str(edge.get("pair_kind", "unknown"))
        counts[pair] = counts.get(pair, 0) + 1
    print_json(
        {
            "candidate_path": candidate_path.as_posix(),
            "output_path": output_path.as_posix(),
            "candidate_edges": len(candidates),
            "selected_edges": len(selected),
            "selected_by_pair_kind": dict(sorted(counts.items())),
            "allowed_pair_kinds": sorted(allowed),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
