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
