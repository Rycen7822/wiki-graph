"""Conservative, best-effort retention after a fully verified native refresh."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Any

from llm_wiki_native.workspace_lock import workspace_mutation_lock
from ops.wiki_mutation_lock import atomic_write_json


_POINTER_FILES = ("active_workspace.json", "prepared_workspace.json", "active_workspace.history.jsonl")


def _pointer_id(pointer: dict[str, Any], root: Path) -> str:
    name = pointer["workspace_id"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("unsafe workspace id")
    directory = root / name
    if Path(pointer["sqlite_path"]) != directory / "native.sqlite" or Path(pointer["zvec_path"]) != directory / "zvec_records":
        raise ValueError(f"workspace pointer escapes canonical layout: {name}")
    return name


def _pointer_snapshot(root: Path) -> dict[str, bytes | None]:
    result = {}
    for name in _POINTER_FILES:
        path = root.parent / name
        if path.is_symlink():
            raise ValueError(f"symlink pointer rejected: {name}")
        result[name] = path.read_bytes() if path.exists() else None
    return result


def _in_use_workspace_ids(root: Path) -> set[str]:
    """Protect same-user fd/cwd/mmap references; fail closed on unknown scan gaps."""
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        raise OSError("workspace retention requires /proc process inspection")
    prefix = str(root) + "/"
    result: set[str] = set()
    for process in proc_root.iterdir():
        if not process.name.isdigit():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                # Cross-user native readers may be invisible in fd/maps. Refuse
                # cleanup when their command identifies this runtime/workspace.
                command = (process / "cmdline").read_bytes().decode(errors="replace")
                if "llm_wiki_native" in command or str(root) in command:
                    raise PermissionError(f"cross-user native process cannot be safely inspected: {process.name}")
                continue
            try:
                links = [process / "cwd", *(process / "fd").iterdir()]
                maps = (process / "maps").read_text()
            except PermissionError:
                # Linux's non-dumpable session-manager/PAM processes are not native readers.
                if (process / "comm").read_text().strip() in {"systemd", "(sd-pam)"}:
                    continue
                raise
            targets = []
            for link in links:
                try:
                    targets.append(os.readlink(link))
                except FileNotFoundError:
                    continue
            targets.extend(maps.splitlines())
            for target in targets:
                if prefix in target:
                    result.add(target.split(prefix, 1)[1].split("/", 1)[0])
        except (FileNotFoundError, ProcessLookupError):
            continue
    return result


def _tree_inventory(directory: Path, root: Path) -> tuple[int, tuple[int, int]]:
    """Reject symlinks/mounts/special files; count allocated, not apparent bytes."""
    if directory.parent != root or directory.resolve() != directory:
        raise ValueError(f"unsafe workspace path: {directory}")
    root_device = root.stat().st_dev
    total = 0
    pending = [directory]
    while pending:
        path = pending.pop()
        info = path.lstat()
        if info.st_dev != root_device or path.is_mount() or info.st_uid != os.getuid():
            raise ValueError(f"foreign filesystem/owner in workspace: {path}")
        if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise ValueError(f"symlink or special file in workspace: {path}")
        total += info.st_blocks * 512
        if stat.S_ISDIR(info.st_mode):
            pending.extend(path.iterdir())
    info = directory.stat()
    return total, (info.st_dev, info.st_ino)


def _cleanup(root: Path, expected_active_id: str, report: dict[str, Any]) -> None:
    snapshot = _pointer_snapshot(root)
    active = json.loads(snapshot["active_workspace.json"] or b"null")
    active_id = _pointer_id(active, root)
    if active.get("status") != "active" or active_id != expected_active_id:
        raise ValueError("active workspace changed since refresh verification")
    history = [json.loads(line) for line in (snapshot["active_workspace.history.jsonl"] or b"").splitlines() if line.strip()]
    if not history or _pointer_id(history[-1]["current"], root) != active_id:
        raise ValueError("active pointer and latest history disagree (possibly rolled back)")
    protected = {active_id}
    previous = history[-1]["previous"]
    if previous is not None:
        protected.add(_pointer_id(previous, root))
    prepared_bytes = snapshot["prepared_workspace.json"]
    if prepared_bytes is not None:
        prepared = json.loads(prepared_bytes)
        if prepared.get("status") != "prepared":
            raise ValueError("invalid prepared workspace status")
        protected.add(_pointer_id(prepared, root))
    for name in protected:
        directory = root / name
        if directory.is_symlink() or not (directory / "native.sqlite").is_file() or not (directory / "zvec_records").is_dir():
            raise ValueError(f"protected workspace missing or invalid: {name}")
    # Only previously published, successfully built workspaces are eligible.
    # Unknown/failed/unpublished staging directories are deliberately left for inspection.
    published = set()
    for row in history:
        for key in ("current", "previous"):
            if row.get(key) is not None:
                published.add(_pointer_id(row[key], root))
    in_use = _in_use_workspace_ids(root)
    report["protected"] = sorted(protected | in_use)
    candidates = []
    for directory in sorted(root.iterdir()):
        name = directory.name
        if name in protected or name in in_use:
            continue
        if name not in published:
            report["retained_unpublished"].append(name)
            continue
        size, identity = _tree_inventory(directory, root)
        build = json.loads((directory / "build_report.json").read_text())
        if build.get("ok") is not True or build.get("workspace_id") != name:
            raise ValueError(f"unverified historical build: {name}")
        candidates.append((directory, size, identity))
    # Finish all validation before deleting any candidate.
    if not shutil.rmtree.avoids_symlink_attacks:
        raise OSError("safe descriptor-based rmtree unavailable")
    for directory, size, identity in candidates:
        if _pointer_snapshot(root) != snapshot:
            raise ValueError("workspace pointers changed during cleanup")
        if directory.name in _in_use_workspace_ids(root):
            report["protected"].append(directory.name)
            continue
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != identity:
            raise ValueError(f"workspace replaced during cleanup: {directory.name}")
        shutil.rmtree(directory)
        if directory.exists():
            raise OSError(f"workspace removal incomplete: {directory.name}")
        report["deleted"].append(directory.name)
        report["reclaimed_bytes"] += size
    if _pointer_snapshot(root) != snapshot:
        raise ValueError("workspace pointers changed during cleanup")
    report["ok"] = True
    report["status"] = "completed"


def cleanup_obsolete_workspaces(*, workspace_root: Path, expected_active_id: str) -> dict[str, Any]:
    """Keep active/previous/prepared/in-use; never invalidate a successful refresh.

    Call only after every caller-owned acceptance gate (including semantic coverage)
    has passed. The expected id binds deletion to the workspace actually verified.
    """
    root = Path(workspace_root).absolute()
    report: dict[str, Any] = {
        "ok": False, "status": "warning", "workspace_root": str(root),
        "protected": [], "retained_unpublished": [], "deleted": [],
        "reclaimed_bytes": 0, "warnings": [],
    }
    try:
        if root.resolve() != root or not root.is_dir():
            raise ValueError("workspace root missing or contains symlinks")
        with workspace_mutation_lock(root.parent, timeout=0):
            try:
                _cleanup(root, expected_active_id, report)
            except Exception as exc:
                report["warnings"].append(f"{type(exc).__name__}: {exc}")
            # A small last-run report; history remains untouched for policy/audit use.
            report_path = root.parent / "workspace_cleanup.json"
            if report_path.is_symlink():
                raise ValueError("symlink cleanup report rejected")
            report["report_path"] = str(report_path)
            atomic_write_json(report_path, report)
    except Exception as exc:
        report["ok"] = False
        report["status"] = "warning"
        report["warnings"].append(f"{type(exc).__name__}: {exc}")
    return report
