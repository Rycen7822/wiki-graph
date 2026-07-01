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
def test_native_query_report_entrypoint_remains_available() -> None:
    assert (SCRIPTS / "collect_native_query_report.py").exists()
    assert importlib.util.find_spec("ops.collect_native_query_report") is not None
