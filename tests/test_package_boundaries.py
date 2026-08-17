from __future__ import annotations

import ast
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def _import_offenders(paths, matches) -> list[str]:
    offenders: list[str] = []
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and matches(node.module or ""):
                offenders.append(f"{rel}:{node.lineno}:{node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if matches(alias.name):
                        offenders.append(f"{rel}:{node.lineno}:{alias.name}")
    return offenders


def test_leaf_entrypoint_wrappers_import_owner_modules_directly() -> None:
    wrapper_paths = [
        ROOT / rel
        for rel in (
            "ops/audit_raw_note_sections.py",
            "ops/build_seed_edges.py",
            "ops/extract_method_atoms.py",
            "ops/extract_raw_sections.py",
            "ops/parse_wiki.py",
            "ops/review_connections.py",
            "ops/rotate_log.py",
            "ops/select_section_similarity_edges.py",
        )
    ]
    assert _import_offenders(wrapper_paths, lambda name: name == "ops.wiki_native_lib") == []


def test_native_package_does_not_import_ops_modules() -> None:
    assert _import_offenders(
        sorted((ROOT / "llm_wiki_native").rglob("*.py")),
        lambda name: name == "ops" or name.startswith("ops."),
    ) == []

def test_package_isolation_smoke_script_documents_installed_entrypoints() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["scripts"] == {
        "llm-wiki-native": "llm_wiki_native.cli:main",
        "wiki-graph-audit-native-refs": "ops.audit_native_production_refs:main",
        "wiki-graph-batch-native-refresh": "ops.batch_native_refresh:main",
        "wiki-graph-batch-wiki-integration": "ops.batch_wiki_integration:main",
        "wiki-graph-custom-kg": "ops.custom_kg_incremental:main",
        "wiki-graph-native-query-report": "ops.collect_native_query_report:main",
        "wiki-graph-native-server-control": "ops.native_server_control:main",
        "wiki-graph-native-zvec-materialize": "ops.native_zvec_materialize:main",
        "wiki-graph-raw-fast-closeout": "ops.raw_fast_closeout:main",
        "wiki-graph-raw-fast-evidence-bundle": "ops.raw_fast_evidence_bundle:main",
        "wiki-graph-search": "ops.wiki_search:main",
    }
