#!/usr/bin/env python3
"""Small runtime/env helpers shared by native llm-wiki maintenance scripts."""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "env_int",
    "load_env_file",
    "redact_summary",
]

_SECRET_KEY_TOKENS = ("key", "token", "secret", "password")


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
        os.environ.setdefault(key, value)
    return values


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def redact_summary(values: dict[str, str]) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key, value in values.items():
        if any(token in key.lower() for token in _SECRET_KEY_TOKENS):
            redacted[key] = "[REDACTED]" if value else ""
        else:
            redacted[key] = value
    return redacted
