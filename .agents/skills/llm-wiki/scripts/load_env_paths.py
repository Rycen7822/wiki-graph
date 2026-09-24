#!/usr/bin/env python3
"""Resolve project-local llm-wiki paths from the ignored repository .env.

Only path variables are emitted. Credentials in .env never reach stdout.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))
from ops.native_runtime_env import load_env_file  # noqa: E402

PATH_KEYS = (
    "WIKI_GRAPH_REPO",
    "LLM_WIKI_ROOT",
    "LLM_WIKI_STATE_DIR",
    "LLM_WIKI_STORAGE_ROOT",
    "LLM_WIKI_RAW_FAST_TMP_ROOT",
    "LLM_WIKI_RAW_FAST_BATCH_TMP_ROOT",
)
OPTIONAL_PATH_KEYS = ("LLM_WIKI_OBCLIP_WRAPPER", "LLM_WIKI_OBCLIP_PROFILE", "OPENREVIEW_ENV_PATH")


def resolve_paths(env_file: Path) -> dict[str, str]:
    if not env_file.is_file():
        raise ValueError("repository .env is missing")
    values = load_env_file(env_file)
    missing = [key for key in PATH_KEYS if not values.get(key)]
    if missing:
        raise ValueError("missing path keys in repository .env: " + ", ".join(missing))
    paths = {key: Path(values[key]).expanduser() for key in PATH_KEYS}
    paths.update({key: Path(values[key]).expanduser() for key in OPTIONAL_PATH_KEYS if values.get(key)})
    if any(not path.is_absolute() for path in paths.values()):
        raise ValueError("all llm-wiki path keys must be absolute")
    if paths["WIKI_GRAPH_REPO"].resolve() != REPO_ROOT:
        raise ValueError("WIKI_GRAPH_REPO must point to this skill's repository")
    skill_dir = REPO_ROOT / ".agents" / "skills" / "llm-wiki"
    verifier = skill_dir / "scripts" / "raw_fast_note_verify.py"
    if not verifier.is_file():
        raise ValueError("project-local raw-fast verifier is missing")
    paths["LLM_WIKI_SKILL_DIR"] = skill_dir
    paths["LLM_WIKI_RAW_FAST_VERIFIER"] = verifier
    return {key: str(path) for key, path in paths.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Validate paths without displaying their values")
    mode.add_argument("--shell", action="store_true", help="Print shell-quoted exports for path keys only")
    args = parser.parse_args()
    try:
        paths = resolve_paths(args.env_file)
    except ValueError as exc:
        print(f"llm-wiki path configuration: {exc}", file=sys.stderr)
        return 2
    if args.check:
        print(json.dumps({"ok": True, "path_keys": list(paths)}, sort_keys=True))
    else:
        for key, value in paths.items():
            print(f"export {key}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
