#!/usr/bin/env python3
"""Build, audit, finalize, and roll back native zvec workspaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm_wiki_native.build import MissingNativeVectorsError  # noqa: E402
from llm_wiki_native.cli import build_workspace_from_state  # noqa: E402
from llm_wiki_native.pointers import finalize_prepared_workspace, rollback_active_workspace  # noqa: E402


class VectorFillFailedError(RuntimeError):
    def __init__(self, message: str, *, before_missing: int, vector_cache_path: Path, exception_type: str) -> None:
        super().__init__(message)
        self.before_missing = before_missing
        self.vector_cache_path = vector_cache_path
        self.exception_type = exception_type


REQUIRED_STATE_INPUTS = (
    "custom_kg_manifest.json",
    "raw_sections.jsonl",
    "section_similarity_edges.jsonl",
)
MANIFEST_VECTOR_COLLECTIONS = ("chunks", "entities", "relationships")


def pointer_dir(workspace_root: Path) -> Path:
    return Path(workspace_root).parent


def workspace_dir(workspace_root: Path, workspace_id: str) -> Path:
    return Path(workspace_root) / workspace_id


def prepared_workspace_path(workspace_root: Path) -> Path:
    return pointer_dir(workspace_root) / "prepared_workspace.json"


def active_workspace_path(workspace_root: Path) -> Path:
    return pointer_dir(workspace_root) / "active_workspace.json"


def history_path(workspace_root: Path) -> Path:
    return pointer_dir(workspace_root) / "active_workspace.history.jsonl"


def build_report_path(workspace_root: Path, workspace_id: str) -> Path:
    return workspace_dir(workspace_root, workspace_id) / "build_report.json"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _vector_missing_count(vector_audit: dict[str, Any]) -> int:
    missing = vector_audit.get("missing", {})
    if not isinstance(missing, dict):
        return 0
    return sum(len(ids) for ids in missing.values() if isinstance(ids, list))


def _sqlite_record_count(counts: dict[str, Any]) -> int:
    return sum(int(value) for value in counts.values())


def _build_ok(native_report: dict[str, Any]) -> bool:
    zvec = native_report.get("zvec") or {}
    insert_stats = zvec.get("insert_stats") or {}
    return bool(
        native_report.get("audit", {}).get("ok")
        and native_report.get("vector_audit", {}).get("ok")
        and int(insert_stats.get("failed", 0)) == 0
        and int(insert_stats.get("inserted", -1)) == int(zvec.get("record_count", -2))
    )


def _manifest_record_count(manifest: dict[str, Any]) -> int:
    total = 0
    for collection in MANIFEST_VECTOR_COLLECTIONS:
        records = manifest.get(collection, {})
        if isinstance(records, dict):
            total += len(records)
    return total


def _jsonl_has_rows(path: Path) -> bool:
    with path.open("r", encoding="utf-8") as f:
        return any(bool(line.strip()) for line in f)


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    state_dir = args.state_dir.resolve()
    workspace_root = args.workspace_root.resolve()
    required = [state_dir / name for name in REQUIRED_STATE_INPUTS]
    manifest_path = state_dir / "custom_kg_manifest.json"
    raw_sections_path = state_dir / "raw_sections.jsonl"
    if manifest_path.exists() and _manifest_record_count(json.loads(manifest_path.read_text(encoding="utf-8"))):
        required.append(state_dir / "vector_cache.sqlite")
    if raw_sections_path.exists() and _jsonl_has_rows(raw_sections_path):
        required.append(state_dir / "section_embeddings.jsonl")
    missing = [str(path) for path in required if not path.exists()]
    return {
        "ok": not missing,
        "command": "preflight",
        "root": str(root),
        "state_dir": str(state_dir),
        "workspace_root": str(workspace_root),
        "workspace_id": args.workspace_id,
        "required": [str(path) for path in required],
        "missing": missing,
    }


def fill_missing_manifest_vectors_for_state(
    state_dir: Path,
    *,
    workdir: Path,
    embedding_profile: str,
    embed_texts_func: Any | None = None,
) -> dict[str, Any]:
    from ops.custom_kg_vector_fill import fill_missing_manifest_vectors
    from llm_wiki_native.artifacts import load_custom_kg_manifest
    from ops.vector_cache import VectorCache, resolve_manifest_vectors

    manifest = load_custom_kg_manifest(state_dir)
    cache_path = state_dir / "vector_cache.sqlite"
    cache = VectorCache(cache_path)
    vector_report = resolve_manifest_vectors(manifest, cache)
    before_missing = int(vector_report.get("summary", {}).get("total", {}).get("misses", 0) or 0)
    if not before_missing:
        return {
            "enabled": True,
            "skipped": True,
            "vector_cache_path": str(cache_path),
            "before_missing": 0,
            "after_missing": 0,
            "fill": None,
        }
    try:
        fill_report = fill_missing_manifest_vectors(
            manifest,
            vector_report,
            cache,
            workdir=workdir,
            embed_texts_func=embed_texts_func,
            embedding_profile=embedding_profile,
        )
    except Exception as exc:
        raise VectorFillFailedError(
            str(exc),
            before_missing=before_missing,
            vector_cache_path=cache_path,
            exception_type=type(exc).__name__,
        ) from exc
    after_report = resolve_manifest_vectors(manifest, cache)
    after_missing = int(after_report.get("summary", {}).get("total", {}).get("misses", 0) or 0)
    if after_missing:
        raise RuntimeError(f"manifest vector fill left unresolved vectors: before={before_missing} after={after_missing}")
    return {
        "enabled": True,
        "skipped": False,
        "vector_cache_path": str(cache_path),
        "before_missing": before_missing,
        "after_missing": after_missing,
        "fill": fill_report,
    }


def build(args: argparse.Namespace, *, embed_texts_func: Any | None = None) -> dict[str, Any]:
    root = args.root.resolve()
    state_dir = args.state_dir.resolve()
    workspace_root = args.workspace_root.resolve()
    candidate_dir = workspace_dir(workspace_root, args.workspace_id)
    sqlite_path = candidate_dir / "native.sqlite"
    zvec_path = candidate_dir / "zvec_records"
    prepared_path = prepared_workspace_path(workspace_root) if args.prepare_only else None
    vector_fill_report = None
    if bool(getattr(args, "fill_missing_vectors", False)):
        vector_fill_report = fill_missing_manifest_vectors_for_state(
            state_dir,
            workdir=state_dir.parent,
            embedding_profile=args.embedding_profile,
            embed_texts_func=embed_texts_func,
        )
    native_report = build_workspace_from_state(
        state_dir,
        sqlite_path,
        args.workspace_id,
        zvec_path=zvec_path,
        prepared_workspace_path=prepared_path,
    )
    zvec = native_report["zvec"]
    report = {
        "ok": _build_ok(native_report),
        "command": "build",
        "prepare_only": bool(args.prepare_only),
        "root": str(root),
        "state_dir": str(state_dir),
        "workspace_root": str(workspace_root),
        "workspace_dir": str(candidate_dir),
        "workspace_id": args.workspace_id,
        "embedding_profile": args.embedding_profile,
        "sqlite_path": str(sqlite_path),
        "zvec_path": str(zvec_path),
        "prepared_workspace": str(prepared_path) if prepared_path else None,
        "build_report": str(build_report_path(workspace_root, args.workspace_id)),
        "counts": native_report["counts"],
        "sqlite_record_count": _sqlite_record_count(native_report["counts"]),
        "sqlite_edge_count": int(native_report["edge_count"]),
        "zvec_doc_count": int(zvec["insert_stats"]["inserted"]),
        "self_nearest_top1_ok": bool(zvec.get("self_nearest_top1_ok")),
        "vector_coverage": {
            "ok": bool(native_report["vector_audit"]["ok"]),
            "missing": _vector_missing_count(native_report["vector_audit"]),
        },
        "native_report": native_report,
    }
    if vector_fill_report is not None:
        report["vector_fill"] = vector_fill_report
    _write_json_atomic(Path(report["build_report"]), report)
    return report


def audit(args: argparse.Namespace) -> dict[str, Any]:
    report_path = build_report_path(args.workspace_root.resolve(), args.workspace_id)
    if not report_path.exists():
        return {
            "ok": False,
            "command": "audit",
            "workspace_id": args.workspace_id,
            "build_report": str(report_path),
            "build_report_found": False,
        }
    report = _read_json(report_path)
    return {
        "ok": bool(report.get("ok")),
        "command": "audit",
        "workspace_id": args.workspace_id,
        "build_report": str(report_path),
        "build_report_found": True,
        "prepared_workspace": report.get("prepared_workspace"),
        "sqlite_record_count": report.get("sqlite_record_count"),
        "zvec_doc_count": report.get("zvec_doc_count"),
        "vector_coverage": report.get("vector_coverage"),
    }


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize native zvec workspaces from llm-wiki state artifacts")
    sub = parser.add_subparsers(dest="command", required=True)

    build_parser = sub.add_parser("build", help="Build a staging native zvec workspace and build report")
    build_parser.add_argument("--root", type=Path, required=True)
    build_parser.add_argument("--state-dir", type=Path, required=True)
    build_parser.add_argument("--workspace-root", type=Path, required=True)
    build_parser.add_argument("--workspace-id", required=True)
    build_parser.add_argument("--embedding-profile", default="conservative")
    build_parser.add_argument("--fill-missing-vectors", action="store_true")
    build_parser.add_argument("--prepare-only", action="store_true")

    preflight_parser = sub.add_parser("preflight", help="Check native zvec staging inputs without workspace writes")
    preflight_parser.add_argument("--root", type=Path, required=True)
    preflight_parser.add_argument("--state-dir", type=Path, required=True)
    preflight_parser.add_argument("--workspace-root", type=Path, required=True)
    preflight_parser.add_argument("--workspace-id", required=True)

    audit_parser = sub.add_parser("audit", help="Read and summarize an existing native zvec build report")
    audit_parser.add_argument("--workspace-root", type=Path, required=True)
    audit_parser.add_argument("--workspace-id", required=True)

    finalize_parser = sub.add_parser("finalize", help="Promote prepared native workspace pointer to active")
    finalize_parser.add_argument("--workspace-root", type=Path, required=True)
    finalize_parser.add_argument("--reason", required=True)

    rollback_parser = sub.add_parser("rollback", help="Restore the previous active native workspace pointer")
    rollback_parser.add_argument("--workspace-root", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "build":
        try:
            print_json(build(args))
        except VectorFillFailedError as exc:
            print_json(
                {
                    "ok": False,
                    "command": "build",
                    "error": "vector_fill_failed",
                    "message": str(exc),
                    "exception_type": exc.exception_type,
                    "root": str(args.root.resolve()),
                    "state_dir": str(args.state_dir.resolve()),
                    "workspace_root": str(args.workspace_root.resolve()),
                    "workspace_id": args.workspace_id,
                    "embedding_profile": args.embedding_profile,
                    "prepare_only": bool(args.prepare_only),
                    "before_missing": exc.before_missing,
                    "vector_cache_path": str(exc.vector_cache_path),
                }
            )
            return 1
        except MissingNativeVectorsError as exc:
            print_json(
                {
                    "ok": False,
                    "command": "build",
                    "error": "missing_native_vectors",
                    "root": str(args.root.resolve()),
                    "state_dir": str(args.state_dir.resolve()),
                    "workspace_root": str(args.workspace_root.resolve()),
                    "workspace_id": args.workspace_id,
                    "embedding_profile": args.embedding_profile,
                    "prepare_only": bool(args.prepare_only),
                    "missing_vectors": exc.report,
                }
            )
            return 1
        return 0
    if args.command == "preflight":
        result = preflight(args)
        print_json(result)
        return 0 if result["ok"] else 1
    if args.command == "audit":
        result = audit(args)
        print_json(result)
        return 0 if result["ok"] else 1
    if args.command == "finalize":
        active = finalize_prepared_workspace(
            prepared_workspace_path(args.workspace_root),
            active_workspace_path(args.workspace_root),
            history_path(args.workspace_root),
            reason=args.reason,
        )
        print_json(active)
        return 0
    if args.command == "rollback":
        active = rollback_active_workspace(
            active_workspace_path(args.workspace_root),
            history_path(args.workspace_root),
        )
        print_json(active)
        return 0
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
