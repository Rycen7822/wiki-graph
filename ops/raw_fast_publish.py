"""Single owner for final raw-note path allocation and publication."""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Sequence

from ops.raw_fast_evidence_bundle import slugify
from ops.wiki_mutation_lock import wiki_mutation_lock

RAW_PATH_RE = re.compile(r"^raw/clip/(?P<yymm>\d{4})/(?P<yymmdd>\d{6})(?P<seq>\d{2})_(?P<slug>[^/]+)\.md$")


class RawPublishError(RuntimeError):
    """Structured raw publication failure."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


class RawPublishConflict(RawPublishError):
    pass


def default_mutation_state_dir(root: Path) -> Path:
    """Return a shared lock location outside the human wiki root."""
    root = root.expanduser().resolve()
    return root.parent / f".{root.name}-wiki-graph-state"


def validate_raw_relative_path(raw_file: str) -> re.Match[str]:
    rel = Path(str(raw_file))
    match = RAW_PATH_RE.fullmatch(str(raw_file))
    if rel.is_absolute() or ".." in rel.parts or match is None:
        raise RawPublishError("invalid_raw_file", f"invalid raw clip path: {raw_file}", raw_file=raw_file)
    if match.group("yymmdd")[:4] != match.group("yymm"):
        raise RawPublishError("invalid_raw_file", f"raw clip month/date mismatch: {raw_file}", raw_file=raw_file)
    return match


def _used_sequences(root: Path, yymm: str, yymmdd: str) -> set[int]:
    used: set[int] = set()
    month_dir = root / "raw" / "clip" / yymm
    if not month_dir.exists():
        return used
    for path in month_dir.glob(f"{yymmdd}??_*.md"):
        match = RAW_PATH_RE.fullmatch(path.relative_to(root).as_posix())
        if match:
            used.add(int(match.group("seq")))
    return used


def allocate_raw_paths_locked(
    root: Path,
    titles: Sequence[str],
    *,
    requested_raw_files: Sequence[str | None] | None = None,
    now: dt.datetime | None = None,
) -> list[str]:
    """Allocate unique paths while the caller holds ``wiki_mutation_lock``."""
    root = root.expanduser().resolve()
    if requested_raw_files is None:
        requested_raw_files = [None] * len(titles)
    if len(requested_raw_files) != len(titles):
        raise ValueError("requested_raw_files must align with titles")
    now = now or dt.datetime.now().astimezone()
    used_by_day: dict[tuple[str, str], set[int]] = {}
    allocated: list[str] = []
    for title, requested in zip(titles, requested_raw_files, strict=True):
        requested_match = validate_raw_relative_path(requested) if requested else None
        yymmdd = requested_match.group("yymmdd") if requested_match else now.strftime("%y%m%d")
        yymm = yymmdd[:4]
        key = (yymm, yymmdd)
        used = used_by_day.setdefault(key, _used_sequences(root, yymm, yymmdd))
        requested_seq = int(requested_match.group("seq")) if requested_match else None
        if requested_seq is not None and requested_seq not in used:
            seq = requested_seq
            raw_file = str(requested)
        else:
            seq = (max(used) + 1) if used else 1
            raw_file = f"raw/clip/{yymm}/{yymmdd}{seq:02d}_{slugify(str(title))}.md"
        if seq > 99:
            raise RawPublishError(
                "raw_sequence_exhausted",
                f"raw clip sequence exhausted for {yymmdd}",
                yymmdd=yymmdd,
                max_sequence=max(used) if used else 0,
            )
        used.add(seq)
        allocated.append(raw_file)
    return allocated


def publish_raw_text_exclusive_locked(root: Path, raw_file: str, raw_text: str) -> dict[str, Any]:
    """Create one canonical raw note with O_EXCL while the caller holds the lock."""
    validate_raw_relative_path(raw_file)
    root = root.expanduser().resolve()
    raw_path = root / raw_file
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(raw_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise RawPublishConflict(
            "raw_file_conflict",
            f"canonical raw path already exists: {raw_file}",
            raw_file=raw_file,
            raw_path=str(raw_path),
        ) from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(raw_text)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raw_path.unlink(missing_ok=True)
        raise
    return {
        "raw_file": raw_file,
        "raw_path": str(raw_path),
        "content_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "published": True,
        "refreshed": False,
    }


def refresh_raw_text_locked(root: Path, raw_file: str, raw_text: str) -> dict[str, Any]:
    """Atomically replace an explicitly selected existing raw note under the lock."""
    validate_raw_relative_path(raw_file)
    root = root.expanduser().resolve()
    raw_path = root / raw_file
    if not raw_path.is_file():
        raise RawPublishError("raw_file_missing", f"existing raw note not found: {raw_file}", raw_file=raw_file)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{raw_path.name}.", suffix=".tmp", dir=raw_path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(raw_text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, raw_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return {
        "raw_file": raw_file,
        "raw_path": str(raw_path),
        "content_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "published": False,
        "refreshed": True,
    }


def publish_raw_note(
    root: Path,
    state_dir: Path | None,
    *,
    title: str,
    raw_text: str,
    requested_raw_file: str | None = None,
    overwrite_existing: bool = False,
    now: dt.datetime | None = None,
    lock_timeout: float = 30.0,
) -> dict[str, Any]:
    """Allocate and publish one note through the canonical lock-protected path."""
    root = root.expanduser().resolve()
    state_dir = (state_dir or default_mutation_state_dir(root)).expanduser().resolve()
    with wiki_mutation_lock(state_dir, timeout=lock_timeout):
        if overwrite_existing:
            if not requested_raw_file:
                raise RawPublishError("missing_raw_file", "overwrite requires an explicit raw_file")
            result = refresh_raw_text_locked(root, requested_raw_file, raw_text)
        else:
            raw_file = allocate_raw_paths_locked(
                root,
                [title],
                requested_raw_files=[requested_raw_file],
                now=now,
            )[0]
            result = publish_raw_text_exclusive_locked(root, raw_file, raw_text)
    return {
        **result,
        "state_dir": str(state_dir),
        "requested_raw_file": requested_raw_file,
        "reallocated": bool(requested_raw_file and result["raw_file"] != requested_raw_file),
    }
