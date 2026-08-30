"""Cross-process lock and atomic JSON helpers for wiki mutations."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterator


DEFAULT_WIKI_MUTATION_LOCK_TIMEOUT = 30.0
_LOCK_RELATIVE_PATH = Path("locks") / "wiki_mutation.lock"
_LOCAL = threading.local()


class WikiMutationLockTimeout(TimeoutError):
    """Raised when the wiki mutation lock cannot be acquired in time."""

    code = "mutation_lock_timeout"

    def __init__(self, path: Path, timeout: float) -> None:
        self.path = Path(path)
        self.timeout = float(timeout)
        super().__init__(f"timed out after {self.timeout:.3f}s waiting for wiki mutation lock: {self.path}")


def wiki_mutation_lock_path(state_dir: Path) -> Path:
    return Path(state_dir) / _LOCK_RELATIVE_PATH


def _held_locks() -> dict[str, dict[str, Any]]:
    locks = getattr(_LOCAL, "held_locks", None)
    if locks is None:
        locks = {}
        _LOCAL.held_locks = locks
    return locks


@contextlib.contextmanager
def wiki_mutation_lock(
    state_dir: Path,
    *,
    timeout: float = DEFAULT_WIKI_MUTATION_LOCK_TIMEOUT,
    poll_interval: float = 0.05,
) -> Iterator[None]:
    """Acquire the shared wiki mutation lock, with same-thread reentrancy."""

    lock_path = wiki_mutation_lock_path(state_dir).resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    key = str(lock_path)
    held = _held_locks()
    current = held.get(key)
    if current is not None:
        current["depth"] += 1
        try:
            yield
        finally:
            current["depth"] -= 1
        return

    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.monotonic() + max(float(timeout), 0.0)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise WikiMutationLockTimeout(lock_path, timeout)
                time.sleep(max(float(poll_interval), 0.001))
        held[key] = {"fd": fd, "depth": 1}
        try:
            yield
        finally:
            held.pop(key, None)
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a JSON object using a unique same-directory temp file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_path, target)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
