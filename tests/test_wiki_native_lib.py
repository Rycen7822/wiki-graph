from __future__ import annotations

import importlib
import inspect
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
from ops import wiki_native_wiki_integration_pending  # noqa: E402

PURE_NATIVE_FACADE_SCRIPTS = [
    "audit_raw_note_sections.py",
    "build_section_similarity_graph.py",
    "build_seed_edges.py",
    "extract_method_atoms.py",
    "extract_raw_sections.py",
    "parse_wiki.py",
    "review_connections.py",
    "rotate_log.py",
    "select_section_similarity_edges.py",
    "validate_wiki.py",
    "wiki_integration_plan.py",
]

ACTIVE_NATIVE_ONLY_HELP_TEXT_SCRIPTS = [
    "audit_raw_note_sections.py",
    "extract_raw_sections.py",
    "review_connections.py",
    "validate_wiki.py",
]

PURE_VALIDATION_HELPERS = {
    "COMPILED_DIR_TYPES",
    "as_list",
    "compiled_pages",
    "display_scalar",
    "find_wikilinks",
    "index_stats",
    "indexed_markdown_pages",
    "parse_frontmatter",
    "raw_clip_files",
    "resolve_source",
    "structured_heading_warnings",
    "wiki_root_machine_pollution",
}


def test_native_facade_does_not_export_retired_http_helpers() -> None:
    retired_backend = "light" + "rag"
    banned = {
        "health",
        "http_json",
        "insert_texts",
        f"load_{retired_backend}_api_key",
        f"query_{retired_backend}",
        f"query_{retired_backend}_data",
        f"sync_docs_to_{retired_backend}",
        "track_status",
        "wait_for_track",
    }

    for name in banned:
        assert name not in wiki_native_lib.__all__
        assert not hasattr(wiki_native_lib, name)

    assert callable(wiki_native_lib.common_paths_parser)
    assert callable(wiki_native_lib.save_evidence_pack)


def test_python_sources_do_not_spell_retired_backend_token_directly() -> None:
    variants = [
        "light" + "rag",
        "Light" + "RAG",
        "LIGHT" + "RAG",
    ]
    source_roots = [
        SCRIPTS,
        ROOT / "llm_wiki_native",
        ROOT / "tests",
    ]
    offenders: list[str] = []

    for source_root in source_roots:
        for path in sorted(source_root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for token in variants:
                if token in text:
                    offenders.append(f"{path.relative_to(ROOT)} contains retired backend token variant")
                    break

    assert offenders == []


def test_retired_external_compatibility_name_registry_is_removed() -> None:
    registry_path = SCRIPTS / "wiki_wikigraph_compat_names.py"
    assert not registry_path.exists()

    import importlib.util

    assert importlib.util.find_spec("wiki_wikigraph_compat_names") is None
    audit_text = (SCRIPTS / "audit_native_production_refs.py").read_text(encoding="utf-8")
    assert "from wiki_wikigraph_compat_names import" not in audit_text
    assert "retired_graph_package_name" not in audit_text


@pytest.mark.parametrize(
    "module_name",
    (
        "wiki_wikigraph_compat_lib",
        "import_custom_kg",
        "batch_wikigraph_refresh",
        "wiki_wikigraph_refresh_pending",
        "collect_wikigraph_query_report",
        "probe_wikigraph_baseline_vector_contract",
        "probe_baseline_query_vector_contract",
        "audit_native_performance_comparison",
        ("light" + "rag_runtime_env"),
        ("wiki_" + "light" + "rag" + "_validation"),
        ("light" + "rag_sync"),
        "sync_virtual_docs",
        "build_evidence_pack",
    ),
)
def test_retired_top_level_script_entrypoints_are_absent(module_name: str) -> None:
    import importlib.util

    sys.modules.pop(module_name, None)
    assert not (SCRIPTS / f"{module_name}.py").exists()
    assert importlib.util.find_spec(module_name) is None


def test_native_query_report_entrypoint_remains_available() -> None:
    assert (SCRIPTS / "collect_native_query_report.py").exists()
    assert importlib.util.find_spec("ops.collect_native_query_report") is not None


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


def test_active_native_diagnostics_do_not_import_retired_compat_name_registry() -> None:
    checks_text = (SCRIPTS / "wiki_native_wiki_checks.py").read_text(encoding="utf-8")
    facade_text = (SCRIPTS / "wiki_native_lib.py").read_text(encoding="utf-8")
    audit_text = (SCRIPTS / "audit_native_production_refs.py").read_text(encoding="utf-8")

    assert "wiki_wikigraph_compat_names" not in checks_text
    assert "retired_graph_package_name" not in checks_text
    assert "wiki_wikigraph_refresh_pending" not in facade_text
    assert "compatibility_marker_specs" not in audit_text
    assert "__wikigraph_" not in audit_text
    assert "old backend compatibility refs" not in audit_text


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


def test_native_facade_exports_pure_artifact_helpers_for_active_scripts() -> None:
    required = {
        "audit_raw_note_section_contracts",
        "build_section_similarity_edges",
        "build_section_similarity_edges_from_index",
        "build_custom_kg_payload",
        "build_seed_edges",
        "custom_kg_doc_description",
        "custom_kg_entity_type",
        "extract_method_atoms",
        "extract_raw_sections",
        "now_stamp",
        "read_text",
        "release_process_memory",
        "section_similarity_embedding_text",
        "section_similarity_edge_to_custom_kg_relationship",
        "section_similarity_index_summary",
        "section_similarity_report_summary",
        "select_section_similarity_edges",
        "sha256_text",
        "write_section_similarity_index",
    }

    for name in required:
        assert name in wiki_native_lib.__all__
        assert hasattr(wiki_native_lib, name)


def test_native_query_events_module_owns_active_query_helpers() -> None:
    native_query_events = importlib.import_module("ops.wiki_native_query_events")

    assert wiki_native_lib.save_evidence_pack is native_query_events.save_evidence_pack
    assert wiki_native_lib.add_query_event is native_query_events.add_query_event

    module_text = (SCRIPTS / "wiki_native_query_events.py").read_text(encoding="utf-8")
    assert "def save_evidence_pack(" in module_text
    assert "def add_query_event(" in module_text
    assert "def init_query_events_db(" in module_text
    assert "def init_manifest_db(" not in module_text
    assert "CREATE TABLE IF NOT EXISTS docs" not in module_text
    assert "CREATE TABLE IF NOT EXISTS sync_events" not in module_text
    assert "def slugify(" in module_text
    assert "import wiki_wikigraph_compat_lib" not in module_text
    assert "from wiki_wikigraph_compat_lib import" not in module_text

    facade_text = (SCRIPTS / "wiki_native_lib.py").read_text(encoding="utf-8")
    assert "from ops.wiki_native_query_events import add_query_event, save_evidence_pack" in facade_text


def test_save_evidence_pack_accepts_string_references_from_answer_route(tmp_path: Path) -> None:
    native_query_events = importlib.import_module("ops.wiki_native_query_events")

    pack = native_query_events.save_evidence_pack(
        tmp_path / "state",
        "answer route query",
        "mix",
        {
            "response": "answer",
            "references": ["raw/example.md"],
            "data": {"context_blocks": [{"source_path": "raw/example.md", "text": "context"}]},
            "trace": {"retrieval_backend": "zvec", "context_block_count": 1},
        },
    )

    text = pack.read_text(encoding="utf-8")
    assert "- file_path: `raw/example.md`" in text


def test_native_cli_module_owns_defaults_and_common_helpers(capsys: pytest.CaptureFixture[str]) -> None:
    native_cli = importlib.import_module("ops.wiki_native_cli")

    for name in (
        "DEFAULT_SERVER",
        "DEFAULT_STATE_DIR",
        "DEFAULT_WIKI_ROOT",
        "DEFAULT_WORKDIR",
    ):
        assert getattr(wiki_native_lib, name) == getattr(native_cli, name)
    for name in ("common_paths_parser", "print_json", "release_process_memory"):
        assert getattr(wiki_native_lib, name) is getattr(native_cli, name)

    parser = wiki_native_lib.common_paths_parser("demo")
    args = parser.parse_args([])
    assert args.root == native_cli.DEFAULT_WIKI_ROOT
    assert args.state_dir == native_cli.DEFAULT_STATE_DIR
    assert args.workdir == native_cli.DEFAULT_WORKDIR
    assert args.server == native_cli.DEFAULT_SERVER

    wiki_native_lib.print_json({"ok": True})
    assert '"ok": true' in capsys.readouterr().out

    module_text = (SCRIPTS / "wiki_native_cli.py").read_text(encoding="utf-8")
    for pattern in (
        "DEFAULT_WIKI_ROOT =",
        "DEFAULT_WORKDIR =",
        "def common_paths_parser(",
        "def print_json(",
        "def release_process_memory(",
    ):
        assert pattern in module_text
    assert "import wiki_wikigraph_compat_lib" not in module_text
    assert "from wiki_wikigraph_compat_lib import" not in module_text

    facade_text = (SCRIPTS / "wiki_native_lib.py").read_text(encoding="utf-8")
    assert "from ops.wiki_native_cli import" in facade_text


def test_native_cli_defaults_are_portable_and_env_backed(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "LLM_WIKI_ROOT": str(tmp_path / "wiki-root"),
            "LLM_WIKI_STATE_DIR": str(tmp_path / "state-dir"),
            "WIKI_GRAPH_REPO": str(tmp_path / "repo-root"),
            "LLM_WIKI_SERVER": "http://127.0.0.1:9999",
        }
    )
    code = (
        "import json, sys; "
        f"sys.path.insert(0, {str(ROOT)!r}); "
        "from ops import wiki_native_cli; "
        "print(json.dumps({"
        "'root': str(wiki_native_cli.DEFAULT_WIKI_ROOT), "
        "'state': str(wiki_native_cli.DEFAULT_STATE_DIR), "
        "'workdir': str(wiki_native_cli.DEFAULT_WORKDIR), "
        "'server': wiki_native_cli.DEFAULT_SERVER"
        "}, sort_keys=True))"
    )

    completed = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, env=env, check=True)
    payload = json.loads(completed.stdout)

    assert payload == {
        "root": str(tmp_path / "wiki-root"),
        "state": str(tmp_path / "state-dir"),
        "workdir": str(tmp_path / "repo-root"),
        "server": "http://127.0.0.1:9999",
    }


def test_active_sources_do_not_embed_operator_local_paths() -> None:
    scanned = [
        SCRIPTS / "wiki_native_cli.py",
        SCRIPTS / "batch_native_refresh.py",
        SCRIPTS / "raw_fast_closeout.py",
        SCRIPTS / "raw_fast_evidence_bundle.py",
        SCRIPTS / "batch_wiki_integration.py",
        SCRIPTS / "wiki_search.py",
    ]
    private_clip_root = Path("/mnt") / "d" / "data" / ("Clip" + "pings")
    patterns = [str(Path.home()), str(Path.home() / ".local" / "share" / "uv" / "tools"), str(private_clip_root)]

    offenders = []
    for path in scanned:
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            if pattern in text:
                offenders.append(f"{path.relative_to(ROOT)} contains {pattern}")

    assert offenders == []




def test_native_artifacts_module_owns_source_and_builder_helpers() -> None:
    native_artifacts = importlib.import_module("ops.wiki_native_artifacts")

    for name in ("build_seed_edges", "extract_method_atoms", "resolve_source"):
        assert getattr(wiki_native_lib, name) is getattr(native_artifacts, name)

    module_text = (SCRIPTS / "wiki_native_artifacts.py").read_text(encoding="utf-8")
    for pattern in (
        "def resolve_source(",
        "def extract_method_atoms(",
        "def build_seed_edges(",
        "def method_atom_markdown(",
        "def edge_markdown(",
    ):
        assert pattern in module_text
    assert "import wiki_wikigraph_compat_lib" not in module_text
    assert "from wiki_wikigraph_compat_lib import" not in module_text

    facade_text = (SCRIPTS / "wiki_native_lib.py").read_text(encoding="utf-8")
    assert "from ops.wiki_native_artifacts import" in facade_text


def test_native_wiki_checks_module_owns_validation_and_audit_helpers() -> None:
    native_wiki_checks = importlib.import_module("ops.wiki_native_wiki_checks")

    for name in (
        "audit_raw_note_section_contracts",
        "compiled_pages",
        "index_stats",
        "indexed_markdown_pages",
        "now_stamp",
        "structured_heading_warnings",
        "validation_freshness_context",
        "validation_report_is_fresh",
        "wiki_root_machine_pollution",
    ):
        assert getattr(wiki_native_lib, name) is getattr(native_wiki_checks, name)

    module_text = (SCRIPTS / "wiki_native_wiki_checks.py").read_text(encoding="utf-8")
    for pattern in (
        "def now_stamp(",
        "def validation_report_is_fresh(",
        "def validation_freshness_context(",
        "def wiki_root_machine_pollution(",
        "def structured_heading_warnings(",
        "def audit_raw_note_section_contracts(",
    ):
        assert pattern in module_text
    assert "import wiki_wikigraph_compat_lib" not in module_text
    assert "from wiki_wikigraph_compat_lib import" not in module_text

    facade_text = (SCRIPTS / "wiki_native_lib.py").read_text(encoding="utf-8")
    assert "from ops.wiki_native_wiki_checks import" in facade_text


def test_native_wiki_integration_pending_module_owns_active_pending_helpers() -> None:
    native_pending = importlib.import_module("ops.wiki_native_wiki_integration_pending")

    assert (
        wiki_native_lib.DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD
        == native_pending.DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD
    )
    for name in (
        "load_pending_wiki_integration_ledger",
        "mark_pending_wiki_integration",
        "pending_wiki_integration_ledger_path",
        "pending_wiki_integration_status",
        "record_pending_wiki_integration_failure",
        "save_pending_wiki_integration_ledger",
    ):
        assert getattr(wiki_native_lib, name) is getattr(native_pending, name)

    module_text = (SCRIPTS / "wiki_native_wiki_integration_pending.py").read_text(encoding="utf-8")
    for pattern in (
        "PENDING_WIKI_INTEGRATION_LEDGER =",
        "DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD =",
        "WIKI_INTEGRATION_ACTIONABLE_STATUSES =",
        "WIKI_INTEGRATION_REVIEW_STATUSES =",
        "WIKI_INTEGRATION_TERMINAL_STATUSES =",
        "def pending_wiki_integration_ledger_path(",
        "def default_pending_wiki_integration_ledger(",
        "def load_pending_wiki_integration_ledger(",
        "def save_pending_wiki_integration_ledger(",
        "def mark_pending_wiki_integration(",
        "def pending_wiki_integration_status(",
        "def clear_pending_wiki_integration_after_success(",
        "def record_pending_wiki_integration_failure(",
    ):
        assert pattern in module_text
    old_backend = "light" + "rag"
    assert f"mark_{old_backend}_refresh_pending" not in module_text
    assert f"marked_{old_backend}_pending" not in module_text
    assert "import wiki_wikigraph_compat_lib" not in module_text
    assert "from wiki_wikigraph_compat_lib import" not in module_text

    facade_text = (SCRIPTS / "wiki_native_lib.py").read_text(encoding="utf-8")
    assert "from ops.wiki_native_wiki_integration_pending import" in facade_text
    assert "_legacy_clear_pending_wiki_integration_after_success" not in facade_text


def test_native_raw_sections_module_owns_active_raw_section_helpers() -> None:
    native_raw_sections = importlib.import_module("ops.wiki_native_raw_sections")

    assert wiki_native_lib.RAW_NOTE_CONTRACT_SECTION_KINDS is native_raw_sections.RAW_NOTE_CONTRACT_SECTION_KINDS
    assert wiki_native_lib.RAW_NOTE_CONTRACT_REQUIRED_KINDS is native_raw_sections.RAW_NOTE_CONTRACT_REQUIRED_KINDS
    assert wiki_native_lib.raw_section_query_for_kind is native_raw_sections.raw_section_query_for_kind

    module_text = (SCRIPTS / "wiki_native_raw_sections.py").read_text(encoding="utf-8")
    assert "RAW_SECTION_SPECS = [" in module_text
    assert "def raw_section_specs_for_heading(" in module_text
    assert "def raw_section_query_for_kind(" in module_text
    assert "import wiki_wikigraph_compat_lib" not in module_text
    assert "from wiki_wikigraph_compat_lib import" not in module_text

    facade_text = (SCRIPTS / "wiki_native_lib.py").read_text(encoding="utf-8")
    assert "from ops.wiki_native_raw_sections import" in facade_text


def test_native_raw_section_extract_module_owns_active_extract_helpers() -> None:
    native_extract = importlib.import_module("ops.wiki_native_raw_section_extract")

    assert wiki_native_lib.extract_raw_sections is native_extract.extract_raw_sections

    module_text = (SCRIPTS / "wiki_native_raw_section_extract.py").read_text(encoding="utf-8")
    assert "def extract_raw_note_sections(" in module_text
    assert "def raw_section_markdown(" in module_text
    assert "def extract_raw_sections(" in module_text
    assert "from ops.wiki_native_raw_sections import" in module_text
    assert "import wiki_wikigraph_compat_lib" not in module_text
    assert "from wiki_wikigraph_compat_lib import" not in module_text

    facade_text = (SCRIPTS / "wiki_native_lib.py").read_text(encoding="utf-8")
    assert "from ops.wiki_native_raw_section_extract import extract_raw_sections" in facade_text


def test_native_docs_module_owns_active_document_helpers() -> None:
    native_docs = importlib.import_module("ops.wiki_native_docs")

    assert wiki_native_lib.WikiDoc is native_docs.WikiDoc
    assert wiki_native_lib.COMPILED_DIR_TYPES is native_docs.COMPILED_DIR_TYPES
    assert wiki_native_lib.collect_source_docs is native_docs.collect_source_docs
    assert wiki_native_lib.generated_docs_from_state is native_docs.generated_docs_from_state
    assert wiki_native_lib.parse_frontmatter is native_docs.parse_frontmatter
    assert wiki_native_lib.read_text is native_docs.read_text
    assert wiki_native_lib.display_scalar is native_docs.display_scalar
    assert wiki_native_lib.sha256_text is native_docs.sha256_text

    module_text = (SCRIPTS / "wiki_native_docs.py").read_text(encoding="utf-8")
    for pattern in (
        "class WikiDoc",
        "def generated_doc_filename(",
        "def sha256_text(",
        "def read_text(",
        "def parse_frontmatter(",
        "def display_scalar(",
        "def canonical_id_for(",
        "def collect_source_docs(",
        "def markdown_sections(",
        "def generated_docs_from_state(",
    ):
        assert pattern in module_text
    assert "import wiki_wikigraph_compat_lib" not in module_text
    assert "from wiki_wikigraph_compat_lib import" not in module_text

    facade_text = (SCRIPTS / "wiki_native_lib.py").read_text(encoding="utf-8")
    assert "from ops.wiki_native_docs import" in facade_text


def test_native_state_module_owns_state_dir_helper(tmp_path: Path) -> None:
    native_state = importlib.import_module("ops.wiki_native_state")

    assert wiki_native_lib.ensure_state_dirs is native_state.ensure_state_dirs
    state = tmp_path / "state"
    wiki_native_lib.ensure_state_dirs(state)
    for name in native_state.STATE_SUBDIRS:
        assert (state / name).is_dir()

    module_text = (SCRIPTS / "wiki_native_state.py").read_text(encoding="utf-8")
    assert "STATE_SUBDIRS = [" in module_text
    assert "def ensure_state_dirs(" in module_text
    assert "import wiki_wikigraph_compat_lib" not in module_text
    assert "from wiki_wikigraph_compat_lib import" not in module_text

    facade_text = (SCRIPTS / "wiki_native_lib.py").read_text(encoding="utf-8")
    assert "from ops.wiki_native_state import ensure_state_dirs" in facade_text


def test_native_ingest_text_module_owns_active_text_helpers() -> None:
    native_text = importlib.import_module("ops.wiki_native_ingest_text")

    assert wiki_native_lib.as_list is native_text.as_list
    assert wiki_native_lib.find_wikilinks is native_text.find_wikilinks

    module_text = (SCRIPTS / "wiki_native_ingest_text.py").read_text(encoding="utf-8")
    for pattern in (
        "def as_list(",
        "def find_wikilinks(",
        "def first_sentences(",
        "def source_urls(",
        "def compact_body_for_ingest(",
        "def limited_scalar(",
        "def make_ingest_text(",
    ):
        assert pattern in module_text
    assert "import wiki_wikigraph_compat_lib" not in module_text
    assert "from wiki_wikigraph_compat_lib import" not in module_text

    facade_text = (SCRIPTS / "wiki_native_lib.py").read_text(encoding="utf-8")
    assert "from ops.wiki_native_ingest_text import" in facade_text


def test_native_jsonl_module_owns_active_jsonl_helpers(tmp_path: Path) -> None:
    native_jsonl = importlib.import_module("ops.wiki_native_jsonl")

    assert wiki_native_lib.jsonl_read is native_jsonl.jsonl_read
    assert wiki_native_lib.jsonl_write is native_jsonl.jsonl_write

    path = tmp_path / "rows.jsonl"
    assert wiki_native_lib.jsonl_write(path, [{"b": 2}, {"a": 1}]) == 2
    assert wiki_native_lib.jsonl_read(path) == [{"b": 2}, {"a": 1}]

    module_text = (SCRIPTS / "wiki_native_jsonl.py").read_text(encoding="utf-8")
    assert "def jsonl_write(" in module_text
    assert "def jsonl_read(" in module_text
    assert "import wiki_wikigraph_compat_lib" not in module_text
    assert "from wiki_wikigraph_compat_lib import" not in module_text

    facade_text = (SCRIPTS / "wiki_native_lib.py").read_text(encoding="utf-8")
    assert "from ops.wiki_native_jsonl import jsonl_read, jsonl_write" in facade_text


def test_native_query_response_module_owns_section_response_helpers() -> None:
    native_response = importlib.import_module("ops.wiki_native_query_response")

    module_text = (SCRIPTS / "wiki_native_query_response.py").read_text(encoding="utf-8")
    retired_backend = "light" + "rag"
    assert "def expand_native_data_response_with_section_neighbors(" in module_text
    assert "def filter_native_data_response_by_section_kind(" in module_text
    assert "def expand_wikigraph_data_response_with_section_neighbors(" not in module_text
    assert "def filter_wikigraph_data_response_by_section_kind(" not in module_text
    assert f"def expand_{retired_backend}_data_response_with_section_neighbors(" not in module_text
    assert f"def filter_{retired_backend}_data_response_by_section_kind(" not in module_text
    assert "from ops.wiki_native_jsonl import jsonl_read" in module_text
    assert "from ops.wiki_native_raw_sections import" in module_text
    assert "import wiki_wikigraph_compat_lib" not in module_text
    assert "from wiki_wikigraph_compat_lib import" not in module_text
    assert callable(native_response.expand_native_data_response_with_section_neighbors)
    assert callable(native_response.filter_native_data_response_by_section_kind)
    assert not hasattr(native_response, "expand_wikigraph_data_response_with_section_neighbors")
    assert not hasattr(native_response, "filter_wikigraph_data_response_by_section_kind")


def test_native_section_similarity_module_owns_graph_helpers() -> None:
    native_similarity = importlib.import_module("ops.wiki_native_section_similarity")

    for name in (
        "build_section_similarity_edges",
        "build_section_similarity_edges_from_index",
        "section_similarity_edge_to_custom_kg_relationship",
        "section_similarity_embedding_text",
        "section_similarity_index_summary",
        "section_similarity_report_summary",
        "select_section_similarity_edges",
        "write_section_similarity_index",
    ):
        assert getattr(wiki_native_lib, name) is getattr(native_similarity, name)

    module_text = (SCRIPTS / "wiki_native_section_similarity.py").read_text(encoding="utf-8")
    for pattern in (
        "def section_similarity_embedding_text(",
        "def _section_rank_lists_scalar(",
        "def _section_rank_lists(",
        "def write_section_similarity_index(",
        "def build_section_similarity_edges_from_index(",
        "def build_section_similarity_edges(",
        "def section_similarity_edge_to_custom_kg_relationship(",
        "def section_similarity_report_summary(",
        "def select_section_similarity_edges(",
    ):
        assert pattern in module_text
    assert "import wiki_wikigraph_compat_lib" not in module_text
    assert "from wiki_wikigraph_compat_lib import" not in module_text

    facade_text = (SCRIPTS / "wiki_native_lib.py").read_text(encoding="utf-8")
    assert "from ops.wiki_native_section_similarity import" in facade_text


def test_native_custom_kg_payload_module_owns_payload_helpers() -> None:
    native_payload = importlib.import_module("ops.wiki_native_custom_kg_payload")

    for name in (
        "build_custom_kg_payload",
        "custom_kg_doc_description",
        "custom_kg_entity_type",
    ):
        assert getattr(wiki_native_lib, name) is getattr(native_payload, name)

    module_text = (SCRIPTS / "wiki_native_custom_kg_payload.py").read_text(encoding="utf-8")
    for pattern in (
        "def custom_kg_entity_type(",
        "def custom_kg_doc_description(",
        "def build_custom_kg_payload(",
    ):
        assert pattern in module_text
    assert "from ops.wiki_native_docs import" in module_text
    assert "from ops.wiki_native_ingest_text import" in module_text
    assert "from ops.wiki_native_jsonl import" in module_text
    assert "from ops.wiki_native_section_similarity import" in module_text
    assert "import wiki_wikigraph_compat_lib" not in module_text
    assert "from wiki_wikigraph_compat_lib import" not in module_text

    facade_text = (SCRIPTS / "wiki_native_lib.py").read_text(encoding="utf-8")
    assert "from ops.wiki_native_custom_kg_payload import" in facade_text


def test_native_facade_exports_pure_validation_helpers() -> None:
    for name in PURE_VALIDATION_HELPERS:
        assert name in wiki_native_lib.__all__
        assert hasattr(wiki_native_lib, name)


def test_selected_active_scripts_import_native_facade_not_retired_graph_lib() -> None:
    for script_name in PURE_NATIVE_FACADE_SCRIPTS:
        text = (SCRIPTS / script_name).read_text(encoding="utf-8")
        assert "from wiki_wikigraph_compat_lib import" not in text
        assert "import wiki_wikigraph_compat_lib" not in text


def test_selected_active_script_help_text_is_native_only() -> None:
    for script_name in ACTIVE_NATIVE_ONLY_HELP_TEXT_SCRIPTS:
        text = (SCRIPTS / script_name).read_text(encoding="utf-8")
        assert ("Light" + "RAG") not in text
        assert ("light" + "rag") not in text


def test_native_runtime_env_helpers_share_env_and_redaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    native_runtime_env = importlib.import_module("ops.native_runtime_env")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=secret-value\nEMBEDDING_DIM=1536\nBAD_INT=nope\n", encoding="utf-8")
    monkeypatch.setenv("EMBEDDING_DIM", "3072")

    loaded = native_runtime_env.load_env_file(env_file)

    assert loaded["EMBEDDING_DIM"] == "1536"
    assert native_runtime_env.env_int("EMBEDDING_DIM", 1) == 3072
    assert native_runtime_env.env_int("BAD_INT", 9) == 9
    assert native_runtime_env.redact_summary({"OPENAI_API_KEY": loaded["OPENAI_API_KEY"], "MODEL": "x"}) == {"OPENAI_API_KEY": "[REDACTED]", "MODEL": "x"}

    source = (SCRIPTS / "native_runtime_env.py").read_text(encoding="utf-8")
    assert not hasattr(native_runtime_env, "env_bool")
    assert not hasattr(native_runtime_env, "env_float")
    assert not hasattr(native_runtime_env, "port_open")
    assert hasattr(native_runtime_env, "env_int")
    assert hasattr(native_runtime_env, "load_env_file")
    assert hasattr(native_runtime_env, "redact_summary")
    assert "def env_bool(" not in source
    assert "def env_float(" not in source
    assert "def port_open(" not in source


def test_active_section_similarity_builder_imports_native_runtime_env() -> None:
    text = (SCRIPTS / "build_section_similarity_graph.py").read_text(encoding="utf-8")
    old_backend = "light" + "rag"

    assert "from ops.native_runtime_env import load_env_file, redact_summary" in text
    assert f"from {old_backend}_runtime_env import" not in text


def test_custom_kg_scripts_import_native_runtime_env() -> None:
    old_backend = "light" + "rag"

    for script_name in ["custom_kg_incremental.py"]:
        text = (SCRIPTS / script_name).read_text(encoding="utf-8")
        assert "from ops.native_runtime_env import" in text
        assert f"from {old_backend}_runtime_env import" not in text


def test_custom_kg_scripts_import_native_facade() -> None:
    old_backend = "light" + "rag"

    for script_name in ["custom_kg_incremental.py"]:
        text = (SCRIPTS / script_name).read_text(encoding="utf-8")
        assert "from ops.wiki_native_lib import" in text
        assert f"from wiki_{old_backend}_lib import" not in text










def test_native_validation_module_owns_validation_implementation() -> None:
    native_validation = importlib.import_module("ops.wiki_native_validation")

    assert wiki_native_lib.validate_wiki is native_validation.validate_wiki
    assert wiki_native_lib.secret_hits is native_validation.secret_hits

    text = (SCRIPTS / "wiki_native_validation.py").read_text(encoding="utf-8")
    assert "def validate_wiki(" in text
    assert "def secret_hits(" in text
    assert "import ops.wiki_native_lib as lib" in text
    assert "from wiki_wikigraph_compat_lib import" not in text
    assert "import wiki_wikigraph_compat_lib" not in text




def test_native_facade_imports_validation_from_native_owner() -> None:
    text = (SCRIPTS / "wiki_native_lib.py").read_text(encoding="utf-8")
    old_backend = "light" + "rag"

    assert "from ops.wiki_native_validation import secret_hits, validate_wiki" in text
    assert f"from wiki_{old_backend}_validation import" not in text






def test_clear_success_native_surface_has_no_legacy_backend_aliases() -> None:
    old_backend = "light" + "rag"
    native_bypass_param = "mark_native" + "_pending"
    native_bypass_flag = "--no-mark-native" + "-pending"

    signature = inspect.signature(wiki_native_lib.clear_pending_wiki_integration_after_success)
    assert native_bypass_param not in signature.parameters
    assert f"mark_{old_backend}_pending" not in signature.parameters

    text = (SCRIPTS / "batch_wiki_integration.py").read_text(encoding="utf-8")
    assert native_bypass_flag not in text
    assert f"--no-mark-{old_backend}-pending" not in text
    assert f"mark_{old_backend}_pending=" not in text
    assert native_bypass_param + "=" not in text


def test_low_level_wiki_integration_clear_has_no_graph_pending_hook() -> None:
    signature = inspect.signature(wiki_native_wiki_integration_pending.clear_pending_wiki_integration_after_success)

    assert "mark_graph_pending" not in signature.parameters


def test_low_level_wiki_integration_clear_result_has_no_graph_pending_fields(tmp_path) -> None:
    root = tmp_path / "wiki"
    state_dir = tmp_path / "state"
    root.mkdir()
    wiki_native_wiki_integration_pending.mark_pending_wiki_integration(
        state_dir,
        root,
        raw_path="raw/clip/26010101_Test.md",
        title="Test",
    )

    result = wiki_native_wiki_integration_pending.clear_pending_wiki_integration_after_success(
        root,
        state_dir,
        reason="unit",
    )

    assert "marked_graph_pending" not in result
    assert "marked_graph_pending_count" not in result


def test_clear_success_marks_native_refresh_not_legacy_backend_refresh(tmp_path) -> None:
    old_backend = "light" + "rag"
    root = tmp_path / "wiki"
    state_dir = tmp_path / "state"
    wiki_native_lib.mark_pending_wiki_integration(
        state_dir,
        root,
        raw_path="raw/clip/26010101_Test.md",
        title="Test",
    )

    result = wiki_native_lib.clear_pending_wiki_integration_after_success(
        root,
        state_dir,
        reason="integration-smoke",
    )

    assert result["cleared_count"] == 1
    assert f"marked_{old_backend}_pending" not in result
    assert f"marked_{old_backend}_pending_count" not in result
    assert result["marked_native_pending_count"] == 1
    assert not (state_dir / f"pending_{old_backend}_refresh.json").exists()
    assert (state_dir / "pending_native_refresh.json").exists()
    native_entries = batch_native_refresh.pending_entries(state_dir)
    assert len(native_entries) == 1
    assert native_entries[0]["reason"] == "wiki-integration:integration-smoke"


def test_wiki_integration_ledger_normalizes_to_current_schema(tmp_path) -> None:
    old_backend = "light" + "rag"
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    ledger_path = state_dir / "pending_wiki_integration.json"
    ledger_path.write_text(
        json.dumps(
            {
                "version": 1,
                "threshold": 10,
                "pending": [],
                "dirty": False,
                f"last_marked_{old_backend}_pending": [{"raw_path": "raw/clip/old.md"}],
                f"last_marked_{old_backend}_pending_count": 1,
                "unexpected_future_or_retired_field": "drop me",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    loaded = wiki_native_lib.load_pending_wiki_integration_ledger(state_dir)
    wiki_native_lib.save_pending_wiki_integration_ledger(state_dir, loaded)
    persisted = json.loads(ledger_path.read_text(encoding="utf-8"))

    assert f"last_marked_{old_backend}_pending" not in loaded
    assert f"last_marked_{old_backend}_pending_count" not in loaded
    assert "unexpected_future_or_retired_field" not in loaded
    assert f"last_marked_{old_backend}_pending" not in persisted
    assert f"last_marked_{old_backend}_pending_count" not in persisted
    assert "unexpected_future_or_retired_field" not in persisted
