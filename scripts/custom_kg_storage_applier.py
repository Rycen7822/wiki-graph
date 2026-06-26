"""Backend-independent custom KG storage applier contract.

The first implementation is a FileBackend adapter over the existing
JSON/NanoVectorDB/NetworkX materializer. DB/versioned backends must satisfy the
same materialize/audit/digest/pointer contract before any runtime promotion.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from custom_kg_incremental import audit_custom_kg_storage
from custom_kg_materialize import materialize_file_storage_from_manifest


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def storage_tree_digest(storage_dir: Path) -> str:
    """Hash semantic storage files independent of their absolute workspace path."""
    storage_dir = Path(storage_dir)
    digest = hashlib.sha256()
    for path in sorted(p for p in storage_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(storage_dir).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class FileBackendStorageApplier:
    """Versioned FileBackend storage applier for custom KG materialization."""

    backend = "file"

    def __init__(self, workspace_root: Path, pointer_path: Path | None = None) -> None:
        self.workspace_root = Path(workspace_root)
        self.pointer_path = Path(pointer_path) if pointer_path is not None else self.workspace_root / "active_workspace.json"
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def storage_dir_for(self, workspace_id: str) -> Path:
        workspace_id = str(workspace_id).strip()
        if not workspace_id:
            raise ValueError("workspace_id is required")
        if "/" in workspace_id or "\\" in workspace_id or workspace_id in {".", ".."}:
            raise ValueError(f"unsafe workspace_id: {workspace_id!r}")
        return self.workspace_root / workspace_id / "rag_storage"

    def materialize(self, workspace_id: str, manifest: dict[str, Any], resolved_vectors: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
        storage_dir = self.storage_dir_for(workspace_id)
        materialize_report = materialize_file_storage_from_manifest(manifest, resolved_vectors, storage_dir)
        audit = audit_custom_kg_storage(storage_dir, manifest)
        if not audit.get("ok"):
            raise RuntimeError(f"materialized workspace audit failed for {workspace_id}: {audit}")
        return {
            "backend": self.backend,
            "workspace_id": workspace_id,
            "status": "audited",
            "storage_dir": storage_dir.as_posix(),
            "counts": materialize_report.get("counts", {}),
            "audit": audit,
            "semantic_digest": storage_tree_digest(storage_dir),
        }

    def active_workspace(self) -> dict[str, Any]:
        if not self.pointer_path.exists():
            return {
                "active_workspace_id": None,
                "active_storage_dir": None,
                "previous_workspace_id": None,
                "previous_storage_dir": None,
            }
        data = json.loads(self.pointer_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{self.pointer_path} must contain a JSON object")
        return data

    def activate(self, workspace_report: dict[str, Any]) -> dict[str, Any]:
        current = self.active_workspace()
        next_pointer = {
            "backend": self.backend,
            "active_workspace_id": str(workspace_report["workspace_id"]),
            "active_storage_dir": str(workspace_report["storage_dir"]),
            "active_semantic_digest": str(workspace_report.get("semantic_digest") or ""),
            "previous_workspace_id": current.get("active_workspace_id"),
            "previous_storage_dir": current.get("active_storage_dir"),
            "previous_semantic_digest": current.get("active_semantic_digest"),
        }
        _write_json(self.pointer_path, next_pointer)
        return next_pointer

    def rollback_active(self) -> dict[str, Any]:
        current = self.active_workspace()
        previous_workspace_id = current.get("previous_workspace_id")
        previous_storage_dir = current.get("previous_storage_dir")
        if not previous_workspace_id or not previous_storage_dir:
            raise RuntimeError("no previous workspace to roll back to")
        rolled_back = {
            "backend": self.backend,
            "active_workspace_id": previous_workspace_id,
            "active_storage_dir": previous_storage_dir,
            "active_semantic_digest": current.get("previous_semantic_digest"),
            "previous_workspace_id": current.get("active_workspace_id"),
            "previous_storage_dir": current.get("active_storage_dir"),
            "previous_semantic_digest": current.get("active_semantic_digest"),
        }
        _write_json(self.pointer_path, rolled_back)
        return rolled_back
