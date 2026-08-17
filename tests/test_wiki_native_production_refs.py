from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def active_production_audit() -> dict:
    audit_native_production_refs = importlib.import_module("ops.audit_native_production_refs")
    return audit_native_production_refs.audit_active_production_refs(ROOT)


def test_active_production_surfaces_restrict_retired_compat_registry_refs(active_production_audit) -> None:
    report = active_production_audit

    assert report["ok"] is True
    assert report["offenders"] == []
    assert {
        "ops/wiki_search.py",
        "ops/custom_kg_incremental.py",
        "ops/vector_cache.py",
        "ops/native_zvec_materialize.py",
        "ops/raw_fast_evidence_bundle.py",
        "llm_wiki_native/api/server.py",
    } <= set(report["checked_paths"])
    assert "ops/sync_virtual_docs.py" not in report["checked_paths"]
    assert "compat_registry_module" not in report["marker_labels"]
    assert "registry_function_prefix" not in report["marker_labels"]
    assert report["allowed_refs"] == {}
    assert "retired_wikigraph_wrapper_refs" not in report
    assert "legacy" + "_wikigraph_wrapper_refs" not in report


def test_audit_native_production_refs_imports_active_modules_with_retired_package_blocked(
    active_production_audit,
) -> None:
    report = active_production_audit
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
                "retrieval_goal": "coverage",
                "top_k": 2,
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

        def get_workspace_metadata(self, workspace_id: str) -> dict:
            assert workspace_id == "wikigraph-zvec192-active"
            return {
                "workspace_id": workspace_id,
                "source_manifest_hash": "manifest-hash",
                "schema_version": 1,
                "status": "audited",
            }

        def get_record(self, workspace_id: str, record_type: str, record_id: str) -> dict[str, str]:
            return {"workspace_id": workspace_id, "record_type": record_type, "record_id": record_id}

        def neighbors(self, workspace_id: str, record_id: str, limit: int) -> list[dict[str, str]]:
            assert limit == 0
            return []

        def query_lexical_spans(self, workspace_id: str, query: str, limit: int, **kwargs) -> list[dict]:
            assert kwargs["normalized_terms"]
            return [
                {
                    "span_id": "span:runtime-smoke",
                    "source_path": "lexical-smoke.md",
                    "source_id": "lexical-smoke",
                    "source_role": "raw",
                    "span_kind": "table.row",
                    "heading_path": ["Runtime smoke"],
                    "start_line": 1,
                    "end_line": 1,
                    "text": "repo-local active pointer smoke",
                    "text_hash": "lexical-smoke-content",
                    "metadata": {},
                    "lexical_route": "lexical_fts",
                }
            ]

    class FakeHit:
        doc_id = "chunk:runtime-smoke"
        score = 1.0
        fields = {
            "record_type": "chunk",
            "record_id": "runtime-smoke",
            "source_path": "runtime-smoke.md",
            "source_id": "runtime-smoke",
            "source_kind_code": 1,
            "section_kind_code": 0,
            "content": "repo-local active pointer smoke",
            "content_hash": "runtime-smoke-content",
        }

    class FakeZvec:
        def query_mix(self, query: str, query_vector: list[float], top_k: int, filter_expr: str | None) -> list[FakeHit]:
            assert query == "repo-local active pointer smoke"
            assert query_vector == [1.0, 0.0]
            assert top_k == 40
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
    assert active_pointer["retrieval_goal"] == "coverage"
    assert active_pointer["hit_count"] == 2
    assert active_pointer["trace"] == {
        "mode": "mix",
        "top_k": 2,
        "record_types": ["chunk"],
        "section_kind": None,
        "vector_hit_count": 1,
        "retrieval_backend": "zvec+lexical",
    }
    assert "hits" not in active_pointer
    assert "query" not in active_pointer["trace"]

    invalid_payload = json.loads(payload_path.read_text(encoding="utf-8"))
    invalid_payload["record_types"] = []
    payload_path.write_text(json.dumps(invalid_payload), encoding="utf-8")
    invalid_report = audit_native_production_refs.audit_active_production_refs(
        ROOT,
        repo_local_active_pointer_path=pointer_path,
        repo_local_query_payload_path=payload_path,
        sqlite_workspace_factory=sqlite_factory,
        zvec_workspace_factory=zvec_factory,
    )
    invalid_check = next(
        row
        for row in invalid_report["package_independence"]["runtime_smoke"]["checks"]
        if row["name"] == "repo_local_active_pointer_query"
    )
    assert invalid_check["ok"] is False
    assert "record_types" in invalid_check["message"]


def test_audit_native_production_refs_cli_outputs_structured_report(capsys: pytest.CaptureFixture[str]) -> None:
    audit_native_production_refs = importlib.import_module("ops.audit_native_production_refs")

    result = audit_native_production_refs.main(["--repo-root", str(ROOT)])
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["ok"] is True
    assert payload["offender_count"] == 0
    assert "allowed_refs" in payload
