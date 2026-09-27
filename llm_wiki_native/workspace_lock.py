"""Serialize workspace builds, pointer changes, and retention across processes."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import threading
import time
from typing import Iterator


_LOCAL = threading.local()


@contextmanager
def workspace_mutation_lock(pointer_dir: Path, *, timeout: float = 30.0) -> Iterator[None]:
    """Reentrant within one thread/process; readers do not acquire this lock."""
    directory = Path(pointer_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "workspace_mutation.lock"
    held = getattr(_LOCAL, "held", None)
    if held is None:
        held = _LOCAL.held = set()
    key = (os.getpid(), str(path))
    if key in held:
        yield
        return
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"workspace mutation lock busy: {path}") from None
                time.sleep(0.05)
        held.add(key)
        try:
            yield
        finally:
            held.remove(key)
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
