import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest

from ops import batch_native_refresh, batch_wiki_integration, local_embedding_service
from support import sample_wiki


def _config(tmp_path: Path) -> local_embedding_service.ManagedLocalEmbeddingServiceConfig:
    return local_embedding_service.ManagedLocalEmbeddingServiceConfig(
        compose_dir=tmp_path,
        service="api",
        health_url="http://127.0.0.1:8000/health",
        start_timeout=10.0,
        health_interval=0.01,
        stop_timeout=10.0,
    )


class _HealthyResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_managed_local_embedding_service_starts_waits_and_verifies_stop(tmp_path: Path, monkeypatch) -> None:
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(local_embedding_service.subprocess, "run", fake_run)
    monkeypatch.setattr(local_embedding_service.urllib.request, "urlopen", lambda *args, **kwargs: _HealthyResponse())

    with local_embedding_service.managed_local_embedding_service(_config(tmp_path)) as report:
        assert report["started"] is True
        assert report["healthy"] is True
        assert report["stopped"] is False

    assert commands == [
        ["docker", "compose", "up", "-d", "api"],
        ["docker", "compose", "stop", "api"],
        ["docker", "compose", "ps", "--status", "running", "--services", "api"],
    ]
    assert report["stop_requested"] is True
    assert report["stopped"] is True


def test_managed_local_embedding_service_stops_when_task_raises(tmp_path: Path, monkeypatch) -> None:
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(local_embedding_service.subprocess, "run", fake_run)
    monkeypatch.setattr(local_embedding_service.urllib.request, "urlopen", lambda *args, **kwargs: _HealthyResponse())

    with pytest.raises(RuntimeError, match="refresh failed"):
        with local_embedding_service.managed_local_embedding_service(_config(tmp_path)):
            raise RuntimeError("refresh failed")

    assert [command[2] for command in commands] == ["up", "stop", "ps"]


def test_managed_local_embedding_service_stops_after_failed_start(tmp_path: Path, monkeypatch) -> None:
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            command,
            1 if command[2] == "up" else 0,
            stdout="",
            stderr="start failed" if command[2] == "up" else "",
        )

    monkeypatch.setattr(local_embedding_service.subprocess, "run", fake_run)

    with pytest.raises(local_embedding_service.ManagedLocalEmbeddingServiceError) as caught:
        with local_embedding_service.managed_local_embedding_service(_config(tmp_path)):
            raise AssertionError("must not enter body")

    assert caught.value.phase == "start"
    assert [command[2] for command in commands] == ["up", "stop", "ps"]
    assert caught.value.report["stopped"] is True


def test_native_refresh_skip_does_not_resolve_or_start_embedding_service(tmp_path: Path, monkeypatch) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "state"
    monkeypatch.setattr(
        batch_wiki_integration,
        "managed_local_embedding_service_config",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("service config must not be resolved")),
    )

    code, payload = batch_wiki_integration.run_native_refresh_after_wiki_integration(
        root,
        state,
        workdir=tmp_path,
        reason="threshold",
    )

    assert code == 0
    assert payload["skip_reason"] == "native_refresh_not_required"
    assert payload["embedding_service"]["managed"] is False


def test_native_refresh_owner_stops_managed_service_after_success_and_failure(tmp_path: Path, monkeypatch) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "state"
    batch_native_refresh.mark_pending(state, root, reason="wiki-integration:threshold")
    events = []

    @contextmanager
    def fake_manager(config):
        report = {"managed": True, "started": True, "healthy": True, "stopped": False}
        events.append("start")
        try:
            yield report
        finally:
            report["stopped"] = True
            events.append("stop")

    monkeypatch.setattr(batch_wiki_integration, "managed_local_embedding_service_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(batch_wiki_integration, "managed_local_embedding_service", fake_manager)
    monkeypatch.setattr(
        batch_wiki_integration,
        "_run_native_refresh_after_wiki_integration",
        lambda *args, **kwargs: (0, {"native_refresh": True, "runs": [], "status_after": {"should_refresh": False}}),
    )

    code, payload = batch_wiki_integration.run_native_refresh_after_wiki_integration(
        root,
        state,
        workdir=tmp_path,
        reason="threshold",
    )

    assert code == 0
    assert events == ["start", "stop"]
    assert payload["embedding_service"]["stopped"] is True

    events.clear()
    monkeypatch.setattr(
        batch_wiki_integration,
        "_run_native_refresh_after_wiki_integration",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("refresh failed")),
    )
    with pytest.raises(RuntimeError, match="refresh failed"):
        batch_wiki_integration.run_native_refresh_after_wiki_integration(
            root,
            state,
            workdir=tmp_path,
            reason="threshold",
        )
    assert events == ["start", "stop"]
