#!/usr/bin/env python3
"""Native-owned shared CLI defaults and lightweight process helpers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if not raw:
        return default
    return Path(raw).expanduser()


DEFAULT_WORKDIR = _env_path("LLM_WIKI_WORKDIR", _env_path("WIKI_GRAPH_REPO", REPO_ROOT))
DEFAULT_WIKI_ROOT = _env_path("LLM_WIKI_ROOT", DEFAULT_WORKDIR)
DEFAULT_STATE_DIR = _env_path("LLM_WIKI_STATE_DIR", DEFAULT_WORKDIR / "tmp" / "native_refresh" / "state")
DEFAULT_SERVER = os.environ.get("LLM_WIKI_SERVER", "http://127.0.0.1:9621")


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def common_paths_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--root", type=Path, default=DEFAULT_WIKI_ROOT)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--server", default=DEFAULT_SERVER)
    return parser


def release_process_memory() -> bool:
    """Best-effort return of freed Python/glibc arenas to the OS."""
    import gc

    gc.collect()
    if os.name != "posix":
        return False
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        return bool(libc.malloc_trim(0))
    except Exception:
        return False
