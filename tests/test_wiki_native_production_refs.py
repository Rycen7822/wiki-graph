from __future__ import annotations

import ast
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"
SCRIPTS = OPS
sys.path.insert(0, str(ROOT))

from ops import batch_native_refresh  # noqa: E402
from ops import wiki_native_lib  # noqa: E402
def test_active_production_surfaces_restrict_retired_compat_registry_refs() -> None:
    audit_native_production_refs = importlib.import_module("ops.audit_native_production_refs")

    report = audit_native_production_refs.audit_active_production_refs(ROOT)

    assert report["ok"] is True
    assert report["offenders"] == []
    assert "ops/batch_native_refresh.py" not in report["allowed_refs"]
    assert "ops/wiki_search.py" in report["checked_paths"]
    assert "ops/sync_virtual_docs.py" not in report["checked_paths"]
    assert "ops/custom_kg_incremental.py" in report["checked_paths"]
    assert "ops/vector_cache.py" in report["checked_paths"]
    assert "ops/native_zvec_materialize.py" in report["checked_paths"]
    assert "ops/raw_fast_evidence_bundle.py" in report["checked_paths"]
    assert "llm_wiki_native/api/server.py" in report["checked_paths"]
    assert "compat_registry_module" not in report["marker_labels"]
    assert "registry_function_prefix" not in report["marker_labels"]
    assert report["allowed_refs"] == {}
    assert "retired_wikigraph_wrapper_refs" not in report
    assert "legacy" + "_wikigraph_wrapper_refs" not in report
def test_audit_native_production_refs_imports_active_modules_with_retired_package_blocked() -> None:
    audit_native_production_refs = importlib.import_module("ops.audit_native_production_refs")

    report = audit_native_production_refs.audit_active_production_refs(ROOT)
    package_independence = report["package_independence"]

    assert package_independence["ok"] is True
    imported = {row["module"] for row in package_independence["imports"]}
    assert {
        "ops.batch_native_refresh",
        "ops.custom_kg_incremental",
        "ops.custom_kg_vector_fill",
        "ops.native_zvec_materialize",
        "ops.raw_fast_closeout",
        "ops.raw_fast_evidence_bundle",
        "ops.vector_cache",
        "ops.wiki_native_cli",
        "ops.wiki_search",
        "llm_wiki_native.api.server",
        "llm_wiki_native.retrieval.query_engine",
        "llm_wiki_native.runtime",
    }.issubset(imported)
    assert all(row["ok"] is True for row in package_independence["imports"])
    runtime_smoke = package_independence["runtime_smoke"]
    assert runtime_smoke["ok"] is True
    checks = {row["name"]: row for row in runtime_smoke["checks"]}
    assert checks["native_query_engine_zvec_naive"]["ok"] is True
    assert checks["native_query_engine_zvec_naive"]["trace"]["retrieval_backend"] == "zvec"
    assert checks["native_query_engine_zvec_naive"]["trace"]["vector_hit_count"] == 1
    refresh_status = checks["batch_native_refresh_status_empty"]
    assert refresh_status["ok"] is True
    assert refresh_status["status"]["pending_count"] == 0
    assert refresh_status["status"]["should_refresh"] is False
    active_loader = checks["active_pointer_loader_default"]
    assert active_loader["ok"] is True
    assert active_loader["workspace_id"] == "package-independence-active"
    assert active_loader["status"] == "active"
    rollback = checks["native_pointer_rollback_previous"]
    assert rollback["ok"] is True
    assert rollback["workspace_id"] == "rollback-old"
    assert rollback["status"] == "active"
    assert rollback["active_pointer_restored"] is True
    isolated = package_independence["isolated_process_smoke"]
    assert isolated["ok"] is True
    assert isolated["production_uninstall_proven"] is False
    isolated_imports = {row["module"] for row in isolated["imports"]}
    assert imported.issubset(isolated_imports)
    isolated_checks = {row["name"]: row for row in isolated["checks"]}
    assert isolated_checks["native_query_engine_zvec_naive"]["ok"] is True
    assert isolated_checks["native_query_engine_zvec_naive"]["trace"]["retrieval_backend"] == "zvec"
    assert isolated_checks["batch_native_refresh_status_empty"]["ok"] is True
    assert isolated_checks["batch_native_refresh_status_empty"]["pending_count"] == 0
    assert isolated_checks["native_pointer_rollback_previous"]["ok"] is True
def test_audit_native_production_refs_can_query_repo_local_active_pointer_with_retired_package_blocked(
    tmp_path: Path,
) -> None:
    audit_native_production_refs = importlib.import_module("ops.audit_native_production_refs")
    pointer_path = tmp_path / "active_workspace.json"
    sqlite_path = tmp_path / "native.sqlite"
    zvec_path = tmp_path / "zvec_records"
    payload_path = tmp_path / "query_payload.json"
    pointer_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workspace_id": "wikigraph-zvec192-active",
                "status": "active",
                "sqlite_path": str(sqlite_path),
                "zvec_path": str(zvec_path),
            }
        ),
        encoding="utf-8",
    )
    payload_path.write_text(
        json.dumps(
            {
                "workspace_id": "wikigraph-zvec192-active",
                "query": "repo-local active pointer smoke",
                "query_vector": [1.0, 0.0],
                "mode": "mix",
                "top_k": 1,
                "neighbor_limit": 0,
                "record_types": ["chunk"],
            }
        ),
        encoding="utf-8",
    )

    class FakeDb:
        def get_workspace_status(self, workspace_id: str) -> str:
            assert workspace_id == "wikigraph-zvec192-active"
            return "audited"

        def get_record(self, workspace_id: str, record_type: str, record_id: str) -> dict[str, str]:
            return {"workspace_id": workspace_id, "record_type": record_type, "record_id": record_id}

        def neighbors(self, workspace_id: str, record_id: str, limit: int) -> list[dict[str, str]]:
            assert limit == 0
            return []

    class FakeHit:
        doc_id = "chunk:runtime-smoke"
        score = 1.0
        fields = {"record_type": "chunk", "record_id": "runtime-smoke"}

    class FakeZvec:
        def query_mix(self, query: str, query_vector: list[float], top_k: int, filter_expr: str | None) -> list[FakeHit]:
            assert query == "repo-local active pointer smoke"
            assert query_vector == [1.0, 0.0]
            assert top_k == 1
            assert filter_expr == "record_type_code in (1)"
            return [FakeHit()]

    def sqlite_factory(path: Path) -> FakeDb:
        assert path == sqlite_path
        return FakeDb()

    def zvec_factory(path: Path, *, read_only: bool) -> FakeZvec:
        assert path == zvec_path
        assert read_only is True
        return FakeZvec()

    report = audit_native_production_refs.audit_active_production_refs(
        ROOT,
        repo_local_active_pointer_path=pointer_path,
        repo_local_query_payload_path=payload_path,
        sqlite_workspace_factory=sqlite_factory,
        zvec_workspace_factory=zvec_factory,
    )

    checks = {
        row["name"]: row
        for row in report["package_independence"]["runtime_smoke"]["checks"]
    }
    active_pointer = checks["repo_local_active_pointer_query"]
    assert active_pointer["ok"] is True
    assert active_pointer["workspace_id"] == "wikigraph-zvec192-active"
    assert active_pointer["status"] == "active"
    assert active_pointer["query_vector_dim"] == 2
    assert active_pointer["hit_count"] == 1
    assert active_pointer["trace"] == {
        "mode": "mix",
        "top_k": 1,
        "record_types": ["chunk"],
        "section_kind": None,
        "vector_hit_count": 1,
        "retrieval_backend": "zvec",
    }
    assert "hits" not in active_pointer
    assert "query" not in active_pointer["trace"]
def test_audit_native_production_refs_cli_outputs_structured_report(capsys: pytest.CaptureFixture[str]) -> None:
    audit_native_production_refs = importlib.import_module("ops.audit_native_production_refs")

    result = audit_native_production_refs.main(["--repo-root", str(ROOT)])
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["ok"] is True
    assert payload["offender_count"] == 0
    assert "allowed_refs" in payload
