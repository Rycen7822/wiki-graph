from __future__ import annotations

import json
from pathlib import Path

import pytest

from ops import native_server_control


ENV_KEYS = [
    "WIKI_GRAPH_REPO",
    "LLM_WIKI_WORKDIR",
    "LLM_WIKI_ROOT",
    "LLM_WIKI_STATE_DIR",
    "LLM_WIKI_NATIVE_HOST",
    "LLM_WIKI_NATIVE_PORT",
    "LLM_WIKI_NATIVE_WORKSPACE_FILE",
    "LLM_WIKI_NATIVE_SERVER_LOG_DIR",
    "LLM_WIKI_NATIVE_SERVER_PIDFILE",
    "LLM_WIKI_NATIVE_HEALTH_URL",
]


def clear_native_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def write_env(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "WIKI_GRAPH_REPO=/repo/from-env",
                "LLM_WIKI_ROOT=/wiki/from-env",
                "LLM_WIKI_STATE_DIR=/state/from-env",
                "LLM_WIKI_NATIVE_HOST=127.0.0.2",
                "LLM_WIKI_NATIVE_PORT=9765",
                "LLM_WIKI_NATIVE_WORKSPACE_FILE=/state/from-env/native_zvec/active_workspace.json",
                "LLM_WIKI_NATIVE_SERVER_LOG_DIR=/state/from-env/native_zvec/server_logs",
                "LLM_WIKI_NATIVE_SERVER_PIDFILE=/state/from-env/native_zvec/native_server_9765.pid",
                "LLM_WIKI_NATIVE_HEALTH_URL=http://127.0.0.2:9765/health",
                "EMBEDDING_BINDING_API_KEY=secret-value",
            ]
        ),
        encoding="utf-8",
    )


def test_resolve_config_reads_paths_and_port_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clear_native_env(monkeypatch)
    env_file = tmp_path / ".env"
    write_env(env_file)

    config = native_server_control.resolve_config(env_file)

    assert config.repo == Path("/repo/from-env")
    assert config.state_dir == Path("/state/from-env")
    assert config.host == "127.0.0.2"
    assert config.port == 9765
    assert config.workspace_file == Path("/state/from-env/native_zvec/active_workspace.json")
    assert config.log_dir == Path("/state/from-env/native_zvec/server_logs")
    assert config.pidfile == Path("/state/from-env/native_zvec/native_server_9765.pid")
    assert config.health_url == "http://127.0.0.2:9765/health"
    assert config.server_command[1:4] == ["-m", "llm_wiki_native.api.server", "--host"]
    assert "--state-dir" in config.server_command
    assert "--workspace-file" in config.server_command


def test_config_command_redacts_secrets_and_dry_run_does_not_start_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    clear_native_env(monkeypatch)
    env_file = tmp_path / ".env"
    write_env(env_file)

    assert native_server_control.main(["--env-file", str(env_file), "restart", "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["config"]["repo"] == "/repo/from-env"
    assert payload["config"]["server_command"][1:4] == ["-m", "llm_wiki_native.api.server", "--host"]
    assert payload["env"]["EMBEDDING_BINDING_API_KEY"] == "[REDACTED]"
    assert "secret-value" not in json.dumps(payload, ensure_ascii=False)


def test_resolve_config_requires_env_backed_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clear_native_env(monkeypatch)
    env_file = tmp_path / ".env"
    env_file.write_text("WIKI_GRAPH_REPO=/repo/from-env\n", encoding="utf-8")

    try:
        native_server_control.resolve_config(env_file)
    except ValueError as exc:
        assert "LLM_WIKI_STATE_DIR" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("resolve_config should require LLM_WIKI_STATE_DIR")
