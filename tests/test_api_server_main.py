from __future__ import annotations

from pathlib import Path

import pytest

from llm_wiki_native import runtime
from llm_wiki_native.api import server


def test_provider_factories_cover_server_wrapper_env_states() -> None:
    assert server.embedding_provider_from_env({}) is None
    assert server.answer_generator_from_env({}) is None

    with pytest.raises(ValueError, match="MODEL"):
        server.embedding_provider_from_env({"LLM_WIKI_NATIVE_EMBEDDING_BASE_URL": "https://embedding.local/v1"})
    with pytest.raises(ValueError, match="MODEL"):
        server.answer_generator_from_env({"LLM_WIKI_NATIVE_ANSWER_BASE_URL": "https://chat.local/v1"})

    assert server.embedding_provider_from_env(
        {
            "LLM_WIKI_NATIVE_EMBEDDING_BASE_URL": "https://embedding.local/v1",
            "LLM_WIKI_NATIVE_EMBEDDING_MODEL": "embed-small",
            "LLM_WIKI_NATIVE_EMBEDDING_API_KEY": "secret",
        }
    ) is not None
    assert server.answer_generator_from_env(
        {
            "LLM_WIKI_NATIVE_ANSWER_BASE_URL": "https://chat.local/v1",
            "LLM_WIKI_NATIVE_ANSWER_MODEL": "chat-small",
            "LLM_WIKI_NATIVE_ANSWER_API_KEY": "secret",
        }
    ) is not None


def test_run_server_loads_engine_builds_app_and_calls_runner(tmp_path) -> None:
    pointer_path = tmp_path / "active_workspace.json"
    calls = {}

    class Engine:
        default_workspace_id = "loaded-workspace"

    def engine_loader(path: Path):
        calls["workspace_file"] = path
        return Engine()

    def runner(app, *, host: str, port: int):
        calls["app"] = app
        calls["host"] = host
        calls["port"] = port

    result = server.run_server(
        [
            "--host",
            "127.0.0.1",
            "--port",
            "9622",
            "--workspace-file",
            str(pointer_path),
            "--state-dir",
            str(tmp_path / "state"),
        ],
        engine_loader=engine_loader,
        runner=runner,
        env={},
    )

    assert result == 0
    assert calls["workspace_file"] == pointer_path
    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 9622
    assert calls["app"].state.default_workspace_id == "loaded-workspace"
    assert "/query/data" in {getattr(route, "path", None) for route in calls["app"].routes}


@pytest.mark.parametrize(
    "argv_extra,expected_statuses,workspace_id,check_port",
    [
        ([], ("active",), "active-workspace", True),
        (["--allow-workspace-status", "prepared"], ("active", "prepared"), "prepared-workspace", False),
    ],
    ids=["active_default", "explicit_prepared"],
)
def test_run_server_passes_allowed_statuses_to_default_loader(
    tmp_path, monkeypatch, argv_extra: list[str], expected_statuses: tuple[str, ...], workspace_id: str, check_port: bool
) -> None:
    pointer_path = tmp_path / "workspace.json"
    calls = {}

    class Engine:
        default_workspace_id = workspace_id

    def fake_loader(path: Path, *, allowed_statuses: tuple[str, ...]):
        calls["workspace_file"] = path
        calls["allowed_statuses"] = allowed_statuses
        return Engine()

    def runner(app, *, host: str, port: int):
        calls["app"] = app
        calls["host"] = host
        calls["port"] = port

    monkeypatch.setattr(runtime, "load_engine_from_workspace_pointer", fake_loader, raising=False)

    result = server.run_server(
        ["--workspace-file", str(pointer_path), *argv_extra],
        runner=runner,
        env={},
    )

    assert result == 0
    assert calls["workspace_file"] == pointer_path
    assert calls["allowed_statuses"] == expected_statuses
    assert calls["app"].state.default_workspace_id == workspace_id
    if check_port:
        assert calls["port"] == 9621
