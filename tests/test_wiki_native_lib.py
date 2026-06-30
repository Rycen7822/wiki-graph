from __future__ import annotations

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


FACADE_OWNER_SYMBOLS = {
    "ops.wiki_native_artifacts": ("build_seed_edges", "extract_method_atoms", "resolve_source"),
    "ops.wiki_native_cli": ("common_paths_parser", "print_json", "release_process_memory"),
    "ops.wiki_native_custom_kg_payload": ("build_custom_kg_payload", "custom_kg_doc_description", "custom_kg_entity_type"),
    "ops.wiki_native_docs": (
        "COMPILED_DIR_TYPES",
        "WikiDoc",
        "collect_source_docs",
        "display_scalar",
        "generated_docs_from_state",
        "parse_frontmatter",
        "raw_clip_files",
        "read_text",
        "sha256_text",
    ),
    "ops.wiki_native_ingest_text": ("as_list", "find_wikilinks"),
    "ops.wiki_native_jsonl": ("jsonl_read", "jsonl_write"),
    "ops.wiki_native_query_events": ("add_query_event", "save_evidence_pack"),
    "ops.wiki_native_raw_section_extract": ("extract_raw_sections",),
    "ops.wiki_native_raw_sections": (
        "RAW_NOTE_CONTRACT_REQUIRED_KINDS",
        "RAW_NOTE_CONTRACT_SECTION_KINDS",
        "raw_section_query_for_kind",
    ),
    "ops.wiki_native_section_similarity": (
        "build_section_similarity_edges",
        "build_section_similarity_edges_from_index",
        "section_similarity_edge_to_custom_kg_relationship",
        "section_similarity_embedding_text",
        "section_similarity_index_summary",
        "section_similarity_report_summary",
        "select_section_similarity_edges",
        "write_section_similarity_index",
    ),
    "ops.wiki_native_state": ("ensure_state_dirs",),
    "ops.wiki_native_validation": ("validate_wiki",),
    "ops.wiki_native_wiki_checks": (
        "audit_raw_note_section_contracts",
        "compiled_pages",
        "index_stats",
        "indexed_markdown_pages",
        "now_stamp",
        "structured_heading_warnings",
        "validation_freshness_context",
        "validation_report_is_fresh",
        "wiki_root_machine_pollution",
    ),
    "ops.wiki_native_wiki_integration_pending": (
        "load_pending_wiki_integration_ledger",
        "mark_pending_wiki_integration",
        "pending_wiki_integration_ledger_path",
        "pending_wiki_integration_status",
        "record_pending_wiki_integration_failure",
        "save_pending_wiki_integration_ledger",
    ),
}

FACADE_OWNER_EQUALITY_SYMBOLS = {
    "ops.wiki_native_cli": ("DEFAULT_SERVER", "DEFAULT_STATE_DIR", "DEFAULT_WIKI_ROOT", "DEFAULT_WORKDIR"),
    "ops.wiki_native_wiki_integration_pending": ("DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD",),
}

FACADE_OWNER_ALIASES = (
    ("ops.batch_native_refresh", "mark_native_refresh_pending", "mark_pending"),
    ("ops.batch_native_refresh", "pending_native_refresh_entries", "pending_entries"),
    ("ops.batch_native_refresh", "pending_native_refresh_ledger_path", "pending_ledger_path"),
    ("ops.batch_native_refresh", "pending_native_refresh_status", "status"),
)



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


def test_native_facade_exports_active_owner_symbols() -> None:
    for module_name, names in FACADE_OWNER_SYMBOLS.items():
        owner = importlib.import_module(module_name)
        for name in names:
            assert name in wiki_native_lib.__all__
            assert getattr(wiki_native_lib, name) is getattr(owner, name)

    for module_name, names in FACADE_OWNER_EQUALITY_SYMBOLS.items():
        owner = importlib.import_module(module_name)
        for name in names:
            assert name in wiki_native_lib.__all__
            assert getattr(wiki_native_lib, name) == getattr(owner, name)

    for module_name, facade_name, owner_name in FACADE_OWNER_ALIASES:
        owner = importlib.import_module(module_name)
        assert facade_name in wiki_native_lib.__all__
        assert getattr(wiki_native_lib, facade_name) is getattr(owner, owner_name)


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


def test_native_cli_common_helpers_preserve_defaults_and_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    native_cli = importlib.import_module("ops.wiki_native_cli")

    parser = wiki_native_lib.common_paths_parser("demo")
    args = parser.parse_args([])
    assert args.root == native_cli.DEFAULT_WIKI_ROOT
    assert args.state_dir == native_cli.DEFAULT_STATE_DIR
    assert args.workdir == native_cli.DEFAULT_WORKDIR
    assert args.server == native_cli.DEFAULT_SERVER

    wiki_native_lib.print_json({"ok": True})
    assert '"ok": true' in capsys.readouterr().out


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


def test_native_state_ensure_state_dirs_creates_expected_subdirs(tmp_path: Path) -> None:
    native_state = importlib.import_module("ops.wiki_native_state")

    state = tmp_path / "state"
    wiki_native_lib.ensure_state_dirs(state)

    for name in native_state.STATE_SUBDIRS:
        assert (state / name).is_dir()


def test_native_facade_exports_pure_validation_helpers() -> None:
    for name in PURE_VALIDATION_HELPERS:
        assert name in wiki_native_lib.__all__
        assert hasattr(wiki_native_lib, name)


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
