from __future__ import annotations

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
