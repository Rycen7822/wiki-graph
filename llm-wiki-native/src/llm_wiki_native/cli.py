"""Callable build helpers for the native shadow workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from llm_wiki_native.artifacts import load_custom_kg_manifest, load_section_similarity_edges
from llm_wiki_native.manifest import manifest_summary, materialize_manifest
from llm_wiki_native.storage.sqlite_workspace import SQLiteWorkspace


def _manifest_hash(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _materialize_edges(db: SQLiteWorkspace, workspace_id: str, manifest: dict[str, Any], section_edges: list[dict[str, Any]]) -> None:
    for relationship in manifest.get("relationships", {}).values():
        db.put_edge(
            workspace_id,
            "relationship",
            str(relationship["src_id"]),
            str(relationship["tgt_id"]),
            float(relationship.get("weight", 1.0)),
            relationship,
        )
    for edge in section_edges:
        db.put_edge(
            workspace_id,
            "section_similarity",
            str(edge["src_id"]),
            str(edge["tgt_id"]),
            float(edge.get("cosine", edge.get("weight", 1.0))),
            edge,
        )


def build_workspace_from_state(state_dir: Path, db_path: Path, workspace_id: str) -> dict[str, Any]:
    manifest = load_custom_kg_manifest(Path(state_dir))
    section_edges = load_section_similarity_edges(Path(state_dir))
    source_manifest_hash = _manifest_hash(manifest)
    db = SQLiteWorkspace(Path(db_path))
    db.create_workspace(workspace_id, source_manifest_hash)
    counts = materialize_manifest(db, workspace_id, manifest)
    _materialize_edges(db, workspace_id, manifest, section_edges)
    expected = {**manifest_summary(manifest), "sections": 0}
    db.mark_audited(workspace_id, expected)
    return {
        "workspace_id": workspace_id,
        "source_manifest_hash": source_manifest_hash,
        "counts": counts,
        "edge_count": db.count_edges(workspace_id),
        "audit": db.audit_counts(workspace_id, expected),
        "status": db.get_workspace_status(workspace_id),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and audit llm-wiki native shadow workspaces")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-workspace", help="Build an audited native SQLite workspace from existing state artifacts")
    build.add_argument("--state-dir", type=Path, required=True)
    build.add_argument("--db", type=Path, required=True)
    build.add_argument("--workspace-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "build-workspace":
        report = build_workspace_from_state(args.state_dir, args.db, args.workspace_id)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    parser.error(f"unsupported command: {args.command}")
    return 2
