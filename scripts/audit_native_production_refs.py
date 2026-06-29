#!/usr/bin/env python3
"""Audit active native production surfaces for old backend compatibility refs."""

from __future__ import annotations

import argparse
import importlib
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any, Callable

from wiki_wikigraph_compat_names import (
    retired_graph_class_name,
    retired_graph_package_name,
    retired_graph_service_name,
    retired_graph_tool_python_path,
    retired_refresh_ledger_name,
)

EXACT_ACTIVE_SURFACES = (
    "scripts/batch_native_refresh.py",
    "scripts/batch_wiki_integration.py",
    "scripts/raw_fast_closeout.py",
    "scripts/sync_virtual_docs.py",
    "scripts/wiki_search.py",
)
ACTIVE_GLOBS = (
    "scripts/wiki_native_*.py",
    "llm-wiki-native/src/**/*.py",
)
ACTIVE_IMPORT_MODULES = (
    "batch_native_refresh",
    "raw_fast_closeout",
    "sync_virtual_docs",
    "wiki_search",
    "llm_wiki_native.api.server",
    "llm_wiki_native.retrieval.query_engine",
    "llm_wiki_native.runtime",
)
ALLOWED_REF_REASONS: dict[str, str] = {}
RETIRED_WIKIGRAPH_WRAPPER_MARKERS = (
    ("retired_wikigraph_refresh_module", "batch_wikigraph_refresh"),
    ("retired_wikigraph_refresh_script", "batch_wikigraph_refresh.py"),
)

_ISOLATED_PACKAGE_ABSENCE_SMOKE_CODE = r"""
import importlib
import importlib.abc
import json
from pathlib import Path
import sys


class BlockedPackageFinder(importlib.abc.MetaPathFinder):
    def __init__(self, blocked_package):
        self.blocked_package = blocked_package

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.blocked_package or fullname.startswith(f"{self.blocked_package}."):
            raise ImportError(f"blocked retired external package import: {fullname}")
        return None


payload = json.loads(sys.stdin.read())
root = Path(payload["repo_root"]).resolve()
blocked_package = payload["blocked_package"]
module_names = payload["module_names"]
for path in (root / "scripts", root / "llm-wiki-native" / "src"):
    sys.path.insert(0, str(path))
for name in list(sys.modules):
    if name == blocked_package or name.startswith(f"{blocked_package}."):
        sys.modules.pop(name, None)
sys.meta_path.insert(0, BlockedPackageFinder(blocked_package))

imports = []
for module_name in module_names:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        imports.append({
            "module": module_name,
            "ok": False,
            "exception_type": type(exc).__name__,
            "message": str(exc),
        })
    else:
        imports.append({"module": module_name, "ok": True})

try:
    audit = importlib.import_module("audit_native_production_refs")
    runtime_smoke = audit._package_independence_runtime_smoke()
    checks = runtime_smoke["checks"]
    for row in checks:
        if row.get("name") == "batch_native_refresh_status_empty" and isinstance(row.get("status"), dict):
            row["pending_count"] = row["status"].get("pending_count")
            row["should_refresh"] = row["status"].get("should_refresh")
except Exception as exc:
    checks = [{
        "name": "isolated_runtime_smoke",
        "ok": False,
        "exception_type": type(exc).__name__,
        "message": str(exc),
    }]

print(json.dumps({
    "ok": all(row["ok"] for row in imports) and all(row["ok"] for row in checks),
    "blocked_package_label": "external_package_name",
    "module_count": len(module_names),
    "imports": imports,
    "checks": checks,
    "production_uninstall_proven": False,
}, sort_keys=True))
"""


class _BlockedPackageFinder(importlib.abc.MetaPathFinder):
    def __init__(self, blocked_package: str) -> None:
        self.blocked_package = blocked_package

    def find_spec(self, fullname: str, path: object = None, target: object = None) -> object:
        if fullname == self.blocked_package or fullname.startswith(f"{self.blocked_package}."):
            raise ImportError(f"blocked retired external package import: {fullname}")
        return None


def compatibility_marker_specs() -> list[dict[str, str]]:
    """Return centrally generated old-backend markers with neutral labels."""

    return [
        {"label": "compat_registry_module", "value": "wiki_wikigraph_compat_names"},
        {"label": "registry_function_prefix", "value": "retired_graph_"},
        {"label": "external_package_name", "value": retired_graph_package_name()},
        {"label": "external_class_name", "value": retired_graph_class_name()},
        {"label": "external_service_name", "value": retired_graph_service_name()},
        {"label": "external_tool_python_path", "value": retired_graph_tool_python_path()},
        {"label": "old_refresh_ledger_name", "value": retired_refresh_ledger_name()},
        {"label": "old_storage_dir_name", "value": "rag_storage"},
    ]


def active_production_surface_paths(repo_root: Path) -> list[Path]:
    root = Path(repo_root)
    paths: set[Path] = set()
    for rel_path in EXACT_ACTIVE_SURFACES:
        path = root / rel_path
        if path.exists():
            paths.add(path)
    for pattern in ACTIVE_GLOBS:
        paths.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(paths)


def _marker_hits(text: str, marker_specs: list[dict[str, str]]) -> list[str]:
    return [spec["label"] for spec in marker_specs if spec["value"] and spec["value"] in text]


def audit_retired_wikigraph_wrapper_refs(root: Path, active_paths: list[Path]) -> dict[str, Any]:
    repo_root = Path(root).resolve()
    offenders: list[dict[str, Any]] = []
    for path in active_paths:
        rel = path.relative_to(repo_root).as_posix()
        text = path.read_text(encoding="utf-8")
        marker_labels = [label for label, marker in RETIRED_WIKIGRAPH_WRAPPER_MARKERS if marker in text]
        if marker_labels:
            offenders.append({"path": rel, "marker_labels": marker_labels})
    return {
        "ok": not offenders,
        "marker_labels": [label for label, _marker in RETIRED_WIKIGRAPH_WRAPPER_MARKERS],
        "offender_count": len(offenders),
        "offenders": offenders,
    }


def _sanitize_query_trace(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": trace.get("mode"),
        "top_k": trace.get("top_k"),
        "record_types": trace.get("record_types"),
        "section_kind": trace.get("section_kind"),
        "vector_hit_count": trace.get("vector_hit_count"),
        "retrieval_backend": trace.get("retrieval_backend"),
    }


def _read_query_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("repo-local query payload must be a JSON object")
    query_vector = payload.get("query_vector")
    if not isinstance(query_vector, list) or not query_vector:
        raise ValueError("repo-local query payload must include non-empty query_vector")
    if not all(isinstance(value, (int, float)) for value in query_vector):
        raise ValueError("repo-local query payload query_vector must contain numbers")
    return payload


def _payload_int(payload: dict[str, Any], key: str, default: int) -> int:
    if key not in payload or payload[key] is None:
        return default
    return int(payload[key])


def _repo_local_active_pointer_query_check(
    active_pointer_path: Path,
    query_payload_path: Path,
    *,
    sqlite_workspace_factory: Callable[[Path], Any] | None = None,
    zvec_workspace_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    check_name = "repo_local_active_pointer_query"
    try:
        from llm_wiki_native.runtime import load_engine_from_prepared_workspace

        pointer = json.loads(Path(active_pointer_path).read_text(encoding="utf-8"))
        if not isinstance(pointer, dict):
            raise ValueError("repo-local active pointer must be a JSON object")
        payload = _read_query_payload(query_payload_path)
        query_vector = [float(value) for value in payload["query_vector"]]
        record_types_payload = payload.get("record_types") or ["entity", "relationship", "chunk", "section"]
        if not isinstance(record_types_payload, list):
            raise ValueError("repo-local query payload record_types must be a list")
        record_types = tuple(str(value) for value in record_types_payload)
        workspace_id = str(payload.get("workspace_id") or pointer.get("workspace_id") or "")
        loader_kwargs: dict[str, Any] = {}
        if sqlite_workspace_factory is not None:
            loader_kwargs["sqlite_workspace_factory"] = sqlite_workspace_factory
        if zvec_workspace_factory is not None:
            loader_kwargs["zvec_workspace_factory"] = zvec_workspace_factory
        engine = load_engine_from_prepared_workspace(
            Path(active_pointer_path),
            allowed_statuses=("prepared", "active"),
            **loader_kwargs,
        )
        result = engine.query(
            workspace_id,
            str(payload.get("query") or ""),
            query_vector,
            mode=str(payload.get("mode") or "mix"),
            top_k=_payload_int(payload, "top_k", 3),
            record_types=record_types,
            section_kind=payload.get("section_kind"),
            neighbor_limit=_payload_int(payload, "neighbor_limit", 5),
        )
        trace = dict(result.get("trace") or {})
        hit_count = len(result.get("hits") or [])
        return {
            "name": check_name,
            "ok": trace.get("retrieval_backend") == "zvec" and hit_count > 0,
            "active_pointer_path": str(active_pointer_path),
            "query_payload_path": str(query_payload_path),
            "workspace_id": workspace_id,
            "status": pointer.get("status"),
            "query_vector_dim": len(query_vector),
            "mode": str(payload.get("mode") or "mix"),
            "top_k": _payload_int(payload, "top_k", 3),
            "record_types": list(record_types),
            "hit_count": hit_count,
            "trace": _sanitize_query_trace(trace),
            "production_uninstall_proven": False,
        }
    except Exception as exc:
        return {
            "name": check_name,
            "ok": False,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "production_uninstall_proven": False,
        }


def _package_independence_runtime_smoke(
    *,
    repo_local_active_pointer_path: Path | None = None,
    repo_local_query_payload_path: Path | None = None,
    sqlite_workspace_factory: Callable[[Path], Any] | None = None,
    zvec_workspace_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    check_name = "native_query_engine_zvec_naive"

    class _Db:
        def get_workspace_status(self, workspace_id: str) -> str:
            return "active"

        def get_record(self, workspace_id: str, record_type: str, record_id: str) -> dict[str, Any]:
            return {
                "workspace_id": workspace_id,
                "record_type": record_type,
                "record_id": record_id,
                "content": "runtime smoke record",
            }

        def neighbors(self, workspace_id: str, record_id: str, limit: int) -> list[dict[str, Any]]:
            return []

    class _Hit:
        doc_id = "chunk:runtime-smoke"
        score = 1.0
        fields = {"record_type": "chunk", "record_id": "runtime-smoke"}

    class _Zvec:
        def query_vector(self, query_vector: list[float], top_k: int, filter_expr: str | None) -> list[_Hit]:
            return [_Hit()]

    checks: list[dict[str, Any]] = []
    try:
        from llm_wiki_native.retrieval.query_engine import NativeQueryEngine

        result = NativeQueryEngine(_Db(), _Zvec()).query(
            "package-independence-smoke",
            "runtime smoke",
            [1.0, 0.0],
            mode="naive",
            top_k=1,
        )
        trace = dict(result.get("trace") or {})
        checks.append(
            {
                "name": check_name,
                "ok": trace.get("retrieval_backend") == "zvec" and trace.get("vector_hit_count") == 1,
                "trace": trace,
            }
        )
    except Exception as exc:
        checks.append(
            {
                "name": check_name,
                "ok": False,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
        )
    try:
        import batch_native_refresh

        root = Path("__wikigraph_package_independence_smoke_root__")
        state_dir = Path("__wikigraph_package_independence_smoke_state__")
        status = batch_native_refresh.status(root, state_dir)
        checks.append(
            {
                "name": "batch_native_refresh_status_empty",
                "ok": status.get("pending_count") == 0 and status.get("should_refresh") is False,
                "status": status,
            }
        )
    except Exception as exc:
        checks.append(
            {
                "name": "batch_native_refresh_status_empty",
                "ok": False,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
        )
    try:
        from llm_wiki_native.runtime import load_engine_from_prepared_workspace

        class _LoaderDb:
            pass

        class _LoaderZvec:
            pass

        calls: dict[str, Any] = {}
        with TemporaryDirectory(prefix="wikigraph-active-loader-") as tmp:
            tmp_path = Path(tmp)
            pointer_path = tmp_path / "active_workspace.json"
            sqlite_path = tmp_path / "records.sqlite"
            zvec_path = tmp_path / "zvec_records"
            pointer_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "workspace_id": "package-independence-active",
                        "status": "active",
                        "sqlite_path": str(sqlite_path),
                        "zvec_path": str(zvec_path),
                    }
                ),
                encoding="utf-8",
            )

            def sqlite_factory(path: Path) -> _LoaderDb:
                calls["sqlite_path"] = path
                return _LoaderDb()

            def zvec_factory(path: Path, *, read_only: bool) -> _LoaderZvec:
                calls["zvec_path"] = path
                calls["read_only"] = read_only
                return _LoaderZvec()

            engine = load_engine_from_prepared_workspace(
                pointer_path,
                allowed_statuses=("prepared", "active"),
                sqlite_workspace_factory=sqlite_factory,
                zvec_workspace_factory=zvec_factory,
            )
            checks.append(
                {
                    "name": "active_pointer_loader_status_allowance",
                    "ok": (
                        getattr(engine, "default_workspace_id", None) == "package-independence-active"
                        and calls == {"sqlite_path": sqlite_path, "zvec_path": zvec_path, "read_only": True}
                    ),
                    "workspace_id": getattr(engine, "default_workspace_id", None),
                    "status": "active",
                }
            )
    except Exception as exc:
        checks.append(
            {
                "name": "active_pointer_loader_status_allowance",
                "ok": False,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
        )
    try:
        from llm_wiki_native.pointers import rollback_active_workspace

        with TemporaryDirectory(prefix="wikigraph-pointer-rollback-") as tmp:
            tmp_path = Path(tmp)
            active_path = tmp_path / "active_workspace.json"
            history_path = tmp_path / "active_workspace.history.jsonl"
            previous = {"schema_version": 1, "workspace_id": "rollback-old", "status": "active"}
            current = {"schema_version": 1, "workspace_id": "rollback-new", "status": "active"}
            active_path.write_text(json.dumps(current, ensure_ascii=False), encoding="utf-8")
            history_path.write_text(
                json.dumps(
                    {
                        "previous": previous,
                        "current": current,
                        "reason": "package independence rollback smoke",
                        "finalized_at": "2026-06-29T00:00:00Z",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            restored = rollback_active_workspace(active_path, history_path)
            restored_file = json.loads(active_path.read_text(encoding="utf-8"))
            checks.append(
                {
                    "name": "native_pointer_rollback_previous",
                    "ok": restored == previous and restored_file == previous,
                    "workspace_id": restored.get("workspace_id"),
                    "status": restored.get("status"),
                    "active_pointer_restored": restored_file == previous,
                    "production_uninstall_proven": False,
                }
            )
    except Exception as exc:
        checks.append(
            {
                "name": "native_pointer_rollback_previous",
                "ok": False,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "production_uninstall_proven": False,
            }
        )
    if repo_local_active_pointer_path is not None or repo_local_query_payload_path is not None:
        if repo_local_active_pointer_path is None or repo_local_query_payload_path is None:
            checks.append(
                {
                    "name": "repo_local_active_pointer_query",
                    "ok": False,
                    "message": "repo-local active pointer and query payload paths must be provided together",
                    "production_uninstall_proven": False,
                }
            )
        else:
            checks.append(
                _repo_local_active_pointer_query_check(
                    repo_local_active_pointer_path,
                    repo_local_query_payload_path,
                    sqlite_workspace_factory=sqlite_workspace_factory,
                    zvec_workspace_factory=zvec_workspace_factory,
                )
            )
    return {"ok": all(row["ok"] for row in checks), "checks": checks}


def _bounded_output(text: str, limit: int = 1200) -> str:
    return text[-limit:]


def _isolated_package_absence_smoke(repo_root: Path, module_names: tuple[str, ...], blocked_package: str) -> dict[str, Any]:
    payload = {
        "repo_root": str(Path(repo_root).resolve()),
        "blocked_package": blocked_package,
        "module_names": list(module_names),
    }
    try:
        process = subprocess.run(
            [sys.executable, "-c", _ISOLATED_PACKAGE_ABSENCE_SMOKE_CODE],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        return {
            "ok": False,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "production_uninstall_proven": False,
            "imports": [],
            "checks": [],
        }
    if process.returncode != 0:
        return {
            "ok": False,
            "returncode": process.returncode,
            "stdout_tail": _bounded_output(process.stdout),
            "stderr_tail": _bounded_output(process.stderr),
            "production_uninstall_proven": False,
            "imports": [],
            "checks": [],
        }
    try:
        report = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "stdout_tail": _bounded_output(process.stdout),
            "stderr_tail": _bounded_output(process.stderr),
            "production_uninstall_proven": False,
            "imports": [],
            "checks": [],
        }
    if not isinstance(report, dict):
        return {
            "ok": False,
            "message": "isolated package-absence smoke returned non-object JSON",
            "production_uninstall_proven": False,
            "imports": [],
            "checks": [],
        }
    report["production_uninstall_proven"] = False
    return report


def audit_package_independent_imports(
    repo_root: Path,
    module_names: tuple[str, ...] = ACTIVE_IMPORT_MODULES,
    *,
    repo_local_active_pointer_path: Path | None = None,
    repo_local_query_payload_path: Path | None = None,
    sqlite_workspace_factory: Callable[[Path], Any] | None = None,
    zvec_workspace_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    blocked_package = retired_graph_package_name()
    finder = _BlockedPackageFinder(blocked_package)
    original_path = list(sys.path)
    original_meta_path = list(sys.meta_path)
    parent_module_names = {
        ".".join(parts[:index])
        for module_name in module_names
        for parts in [module_name.split(".")]
        for index in range(1, len(parts))
    }
    module_names_to_restore = set(module_names) | parent_module_names | {
        name for name in sys.modules if name == blocked_package or name.startswith(f"{blocked_package}.")
    }
    module_snapshots = {name: sys.modules.get(name) for name in module_names_to_restore}
    module_was_present = {name: name in sys.modules for name in module_names_to_restore}
    imports: list[dict[str, Any]] = []
    try:
        for path in (root / "scripts", root / "llm-wiki-native" / "src"):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)
        for name in module_names_to_restore:
            sys.modules.pop(name, None)
        sys.meta_path.insert(0, finder)
        for module_name in module_names:
            try:
                importlib.import_module(module_name)
            except Exception as exc:
                imports.append(
                    {
                        "module": module_name,
                        "ok": False,
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
            else:
                imports.append({"module": module_name, "ok": True})
        runtime_smoke = _package_independence_runtime_smoke(
            repo_local_active_pointer_path=repo_local_active_pointer_path,
            repo_local_query_payload_path=repo_local_query_payload_path,
            sqlite_workspace_factory=sqlite_workspace_factory,
            zvec_workspace_factory=zvec_workspace_factory,
        )
        isolated_process_smoke = _isolated_package_absence_smoke(root, module_names, blocked_package)
    finally:
        sys.meta_path[:] = original_meta_path
        sys.path[:] = original_path
        for module_name in module_names:
            sys.modules.pop(module_name, None)
        for name, module in module_snapshots.items():
            if module_was_present[name]:
                sys.modules[name] = module
            else:
                sys.modules.pop(name, None)
    return {
        "ok": all(row["ok"] for row in imports) and runtime_smoke["ok"] and isolated_process_smoke["ok"],
        "blocked_package_label": "external_package_name",
        "module_count": len(module_names),
        "imports": imports,
        "runtime_smoke": runtime_smoke,
        "isolated_process_smoke": isolated_process_smoke,
    }


def audit_active_production_refs(
    repo_root: Path,
    *,
    repo_local_active_pointer_path: Path | None = None,
    repo_local_query_payload_path: Path | None = None,
    sqlite_workspace_factory: Callable[[Path], Any] | None = None,
    zvec_workspace_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    marker_specs = compatibility_marker_specs()
    marker_labels = [spec["label"] for spec in marker_specs]
    active_paths = active_production_surface_paths(root)
    checked_paths: list[str] = []
    allowed_refs: dict[str, dict[str, Any]] = {}
    offenders: list[dict[str, Any]] = []

    for path in active_paths:
        rel = path.relative_to(root).as_posix()
        checked_paths.append(rel)
        hits = _marker_hits(path.read_text(encoding="utf-8"), marker_specs)
        if not hits:
            continue
        if rel in ALLOWED_REF_REASONS:
            allowed_refs[rel] = {
                "reason": ALLOWED_REF_REASONS[rel],
                "marker_labels": hits,
            }
            continue
        offenders.append({"path": rel, "marker_labels": hits})

    retired_wikigraph_wrapper_refs = audit_retired_wikigraph_wrapper_refs(root, active_paths)
    package_independence = audit_package_independent_imports(
        root,
        repo_local_active_pointer_path=repo_local_active_pointer_path,
        repo_local_query_payload_path=repo_local_query_payload_path,
        sqlite_workspace_factory=sqlite_workspace_factory,
        zvec_workspace_factory=zvec_workspace_factory,
    )

    return {
        "ok": not offenders and retired_wikigraph_wrapper_refs["ok"] and package_independence["ok"],
        "repo_root": str(root),
        "checked_count": len(checked_paths),
        "checked_paths": checked_paths,
        "marker_labels": marker_labels,
        "allowed_refs": allowed_refs,
        "offender_count": len(offenders),
        "offenders": offenders,
        "retired_wikigraph_wrapper_refs": retired_wikigraph_wrapper_refs,
        "package_independence": package_independence,
    }


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit active native production refs")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--repo-local-active-pointer", type=Path)
    parser.add_argument("--repo-local-query-payload", type=Path)
    args = parser.parse_args(argv)
    if (args.repo_local_active_pointer is None) != (args.repo_local_query_payload is None):
        parser.error("--repo-local-active-pointer and --repo-local-query-payload must be provided together")

    report = audit_active_production_refs(
        args.repo_root,
        repo_local_active_pointer_path=args.repo_local_active_pointer,
        repo_local_query_payload_path=args.repo_local_query_payload,
    )
    print_json(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
