#!/usr/bin/env python3
"""Native-owned shared CLI defaults and lightweight process helpers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_WIKI_ROOT = Path("/mnt/d/data/Clippings/llm-wiki")
DEFAULT_WORKDIR = Path("/home/xu/project/wiki/wikigraph")
DEFAULT_STATE_DIR = DEFAULT_WORKDIR / "state"
DEFAULT_SERVER = "http://127.0.0.1:9621"


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
