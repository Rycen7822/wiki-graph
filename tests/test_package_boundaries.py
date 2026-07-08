from __future__ import annotations

import ast
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]

PACKAGE_ISOLATION_SMOKE_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail
python3 -m venv /tmp/wiki-graph-package-smoke
/tmp/wiki-graph-package-smoke/bin/python -m pip install -U pip
/tmp/wiki-graph-package-smoke/bin/python -m pip install -e .
/tmp/wiki-graph-package-smoke/bin/llm-wiki-native --help
/tmp/wiki-graph-package-smoke/bin/wiki-graph-batch-native-refresh --help
/tmp/wiki-graph-package-smoke/bin/wiki-graph-search --help
/tmp/wiki-graph-package-smoke/bin/wiki-graph-native-query-report --help
/tmp/wiki-graph-package-smoke/bin/wiki-graph-native-server-control --help
/tmp/wiki-graph-package-smoke/bin/python -c "import llm_wiki_native, ops.batch_native_refresh, ops.wiki_search"
"""

def test_leaf_entrypoint_wrappers_import_owner_modules_directly() -> None:
    wrapper_paths = [
        "ops/audit_raw_note_sections.py",
        "ops/build_seed_edges.py",
        "ops/extract_method_atoms.py",
        "ops/extract_raw_sections.py",
        "ops/parse_wiki.py",
        "ops/review_connections.py",
        "ops/rotate_log.py",
        "ops/select_section_similarity_edges.py",
    ]
    offenders: list[str] = []
    for rel_path in wrapper_paths:
        path = ROOT / rel_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "ops.wiki_native_lib":
                names = ",".join(alias.name for alias in node.names)
                offenders.append(f"{rel_path}:{node.lineno}:{names}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "ops.wiki_native_lib":
                        offenders.append(f"{rel_path}:{node.lineno}:{alias.name}")

    assert offenders == []

def test_native_package_does_not_import_ops_modules() -> None:
    offenders: list[str] = []
    for path in sorted((ROOT / "llm_wiki_native").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "ops" or alias.name.startswith("ops."):
                        offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}:{alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "ops" or module.startswith("ops."):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}:{module}")

    assert offenders == []

def test_package_isolation_smoke_script_documents_installed_entrypoints() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]
    expected = {
        "llm-wiki-native": "llm_wiki_native.cli:main",
        "wiki-graph-batch-native-refresh": "ops.batch_native_refresh:main",
        "wiki-graph-search": "ops.wiki_search:main",
        "wiki-graph-native-query-report": "ops.collect_native_query_report:main",
        "wiki-graph-native-server-control": "ops.native_server_control:main",
    }

    assert {name: scripts[name] for name in expected} == expected
    for entrypoint in expected:
        assert f"/tmp/wiki-graph-package-smoke/bin/{entrypoint} --help" in PACKAGE_ISOLATION_SMOKE_SCRIPT
    assert "PYTHONPATH" not in PACKAGE_ISOLATION_SMOKE_SCRIPT
