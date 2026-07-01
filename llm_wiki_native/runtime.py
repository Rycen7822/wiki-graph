"""Runtime loading helpers for native zvec workspace pointers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from llm_wiki_native.retrieval.query_engine import NativeQueryEngine
from llm_wiki_native.storage.sqlite_workspace import SQLiteWorkspace


def load_engine_from_workspace_pointer(
    pointer_path: Path,
    *,
    allowed_statuses: tuple[str, ...] = ("active",),
    sqlite_workspace_factory: Callable[[Path], Any] = SQLiteWorkspace,
    zvec_workspace_factory: Callable[..., Any] | None = None,
) -> NativeQueryEngine:
    pointer = _read_pointer(Path(pointer_path))
    if pointer.get("schema_version") != 1:
        raise ValueError("workspace pointer schema_version must be 1")
    status = str(pointer.get("status", ""))
    if status not in set(allowed_statuses):
        allowed = ", ".join(allowed_statuses)
        raise ValueError(f"workspace pointer status must be one of: {allowed}")
    sqlite_path = Path(str(pointer["sqlite_path"]))
    zvec_path = Path(str(pointer["zvec_path"]))
    db = sqlite_workspace_factory(sqlite_path)
    if zvec_workspace_factory is None:
        from llm_wiki_native.storage.zvec_workspace import open_workspace_collection

        zvec_workspace_factory = open_workspace_collection
    zvec_workspace = zvec_workspace_factory(zvec_path, read_only=True)
    source_root = Path(str(pointer["source_root"])).resolve() if pointer.get("source_root") else None
    engine = NativeQueryEngine(db, zvec_workspace=zvec_workspace, source_root=source_root)
    engine.default_workspace_id = str(pointer["workspace_id"])  # type: ignore[attr-defined]
    return engine


def _read_pointer(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("workspace pointer must be a JSON object")
    for key in ("workspace_id", "sqlite_path", "zvec_path"):
        if not payload.get(key):
            raise ValueError(f"workspace pointer missing {key}")
    return payload
