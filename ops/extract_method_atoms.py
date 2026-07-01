#!/usr/bin/env python3
from ops.wiki_native_artifacts import extract_method_atoms
from ops.wiki_native_cli import common_paths_parser, print_json


def main() -> int:
    parser = common_paths_parser("Extract deterministic MethodAtom virtual docs from structured paper notes")
    args = parser.parse_args()
    result = extract_method_atoms(args.root, args.state_dir)
    print_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
