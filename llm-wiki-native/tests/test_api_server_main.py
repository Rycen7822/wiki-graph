from __future__ import annotations

from pathlib import Path

import pytest

from llm_wiki_native import runtime
from llm_wiki_native.api import server


def test_provider_factories_are_none_when_unconfigured_and_fail_on_partial_env() -> None:
    assert server.embedding_provider_from_env({}) is None
    assert server.answer_generator_from_env({}) is None

    with pytest.raises(ValueError, match="MODEL"):
        server.embedding_provider_from_env({"LLM_WIKI_NATIVE_EMBEDDING_BASE_URL": "https://embedding.local/v1"})
    with pytest.raises(ValueError, match="MODEL"):
        server.answer_generator_from_env({"LLM_WIKI_NATIVE_ANSWER_BASE_URL": "https://chat.local/v1"})


def test_embedding_provider_factory_accepts_compatible_binding_env() -> None:
    provider = server.embedding_provider_from_env(
        {
            "EMBEDDING_BINDING_HOST": "https://embedding.local/v1",
            "EMBEDDING_MODEL": "BAAI/bge-m3",
            "EMBEDDING_BINDING_API_KEY": "secret",
            "EMBEDDING_DIM": "1024",
        }
    )

    assert provider is not None
    assert provider.config.base_url == "https://embedding.local/v1"
    assert provider.config.model == "BAAI/bge-m3"
    assert provider.config.embedding_dim == 1024


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


def test_run_server_passes_active_default_to_default_loader(tmp_path, monkeypatch) -> None:
    pointer_path = tmp_path / "active_workspace.json"
    calls = {}

    class Engine:
        default_workspace_id = "active-workspace"

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
        [
            "--workspace-file",
            str(pointer_path),
        ],
        runner=runner,
        env={},
    )

    assert result == 0
    assert calls["workspace_file"] == pointer_path
    assert calls["allowed_statuses"] == ("active",)
    assert calls["port"] == 9621
    assert calls["app"].state.default_workspace_id == "active-workspace"


def test_run_server_appends_explicit_prepared_status_for_staging_loader(tmp_path, monkeypatch) -> None:
    pointer_path = tmp_path / "prepared_workspace.json"
    calls = {}

    class Engine:
        default_workspace_id = "prepared-workspace"

    def fake_loader(path: Path, *, allowed_statuses: tuple[str, ...]):
        calls["workspace_file"] = path
        calls["allowed_statuses"] = allowed_statuses
        return Engine()

    def runner(app, *, host: str, port: int):
        calls["app"] = app

    monkeypatch.setattr(runtime, "load_engine_from_workspace_pointer", fake_loader, raising=False)

    result = server.run_server(
        [
            "--workspace-file",
            str(pointer_path),
            "--allow-workspace-status",
            "prepared",
        ],
        runner=runner,
        env={},
    )

    assert result == 0
    assert calls["workspace_file"] == pointer_path
    assert calls["allowed_statuses"] == ("active", "prepared")
    assert calls["app"].state.default_workspace_id == "prepared-workspace"
