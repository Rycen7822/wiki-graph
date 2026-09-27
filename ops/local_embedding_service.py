"""Managed Docker lifecycle for the local embedding service."""

from __future__ import annotations

import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping


MANAGED_ENV = "LLM_WIKI_NATIVE_LOCAL_EMBEDDING_DOCKER_MANAGED"
COMPOSE_DIR_ENV = "LLM_WIKI_NATIVE_LOCAL_EMBEDDING_COMPOSE_DIR"
COMPOSE_SERVICE_ENV = "LLM_WIKI_NATIVE_LOCAL_EMBEDDING_COMPOSE_SERVICE"
HEALTH_URL_ENV = "LLM_WIKI_NATIVE_LOCAL_EMBEDDING_HEALTH_URL"
START_TIMEOUT_ENV = "LLM_WIKI_NATIVE_LOCAL_EMBEDDING_START_TIMEOUT"
HEALTH_INTERVAL_ENV = "LLM_WIKI_NATIVE_LOCAL_EMBEDDING_HEALTH_INTERVAL"
STOP_TIMEOUT_ENV = "LLM_WIKI_NATIVE_LOCAL_EMBEDDING_STOP_TIMEOUT"


class ManagedLocalEmbeddingServiceError(RuntimeError):
    """Raised when a configured local embedding service cannot be managed."""

    def __init__(self, phase: str, message: str, *, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.phase = phase
        self.report = report


@dataclass(frozen=True)
class ManagedLocalEmbeddingServiceConfig:
    compose_dir: Path
    service: str
    health_url: str
    start_timeout: float
    health_interval: float
    stop_timeout: float


def _value(values: Mapping[str, str], name: str) -> str | None:
    value = str(values.get(name) or "").strip()
    return value or None


def _enabled(values: Mapping[str, str]) -> bool:
    value = (_value(values, MANAGED_ENV) or "false").lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{MANAGED_ENV} must be true or false")


def _positive_float(values: Mapping[str, str], name: str, default: float) -> float:
    raw = _value(values, name)
    try:
        value = default if raw is None else float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value


def managed_local_embedding_service_config(
    values: Mapping[str, str],
    *,
    workdir: Path,
) -> ManagedLocalEmbeddingServiceConfig | None:
    """Resolve the opt-in Compose lifecycle from native refresh config."""

    if not _enabled(values):
        return None
    compose_dir_value = _value(values, COMPOSE_DIR_ENV)
    health_url = _value(values, HEALTH_URL_ENV)
    if compose_dir_value is None:
        raise ValueError(f"{COMPOSE_DIR_ENV} is required when {MANAGED_ENV}=true")
    if health_url is None:
        raise ValueError(f"{HEALTH_URL_ENV} is required when {MANAGED_ENV}=true")
    compose_dir = Path(compose_dir_value).expanduser()
    if not compose_dir.is_absolute():
        compose_dir = workdir / compose_dir
    compose_dir = compose_dir.resolve()
    if not compose_dir.is_dir():
        raise ValueError(f"managed local embedding compose directory does not exist: {compose_dir}")
    return ManagedLocalEmbeddingServiceConfig(
        compose_dir=compose_dir,
        service=_value(values, COMPOSE_SERVICE_ENV) or "api",
        health_url=health_url,
        start_timeout=_positive_float(values, START_TIMEOUT_ENV, 1800.0),
        health_interval=_positive_float(values, HEALTH_INTERVAL_ENV, 2.0),
        stop_timeout=_positive_float(values, STOP_TIMEOUT_ENV, 180.0),
    )


def _run_compose(
    config: ManagedLocalEmbeddingServiceConfig,
    args: list[str],
    *,
    phase: str,
    timeout: float,
    report: dict[str, Any],
) -> subprocess.CompletedProcess[str]:
    command = ["docker", "compose", *args]
    try:
        completed = subprocess.run(
            command,
            cwd=config.compose_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ManagedLocalEmbeddingServiceError(
            phase,
            f"managed local embedding Docker {phase} command failed: {type(exc).__name__}: {exc}",
            report=report,
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "docker compose returned nonzero").strip()
        raise ManagedLocalEmbeddingServiceError(
            phase,
            f"managed local embedding Docker {phase} command returned {completed.returncode}: {detail[-1200:]}",
            report=report,
        )
    return completed


def _wait_until_healthy(config: ManagedLocalEmbeddingServiceConfig, report: dict[str, Any]) -> None:
    deadline = time.monotonic() + config.start_timeout
    last_error = "health endpoint did not respond"
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ManagedLocalEmbeddingServiceError(
                "health",
                f"managed local embedding service did not become healthy within {config.start_timeout:g}s: {last_error}",
                report=report,
            )
        try:
            with urllib.request.urlopen(config.health_url, timeout=min(5.0, remaining)) as response:
                status = int(getattr(response, "status", 200))
                if 200 <= status < 300:
                    report["healthy"] = True
                    return
                last_error = f"HTTP {status}"
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(min(config.health_interval, max(0.0, remaining)))


def _verify_stopped(config: ManagedLocalEmbeddingServiceConfig, report: dict[str, Any]) -> None:
    completed = _run_compose(
        config,
        ["ps", "--status", "running", "--services", config.service],
        phase="stop-verification",
        timeout=min(config.stop_timeout, 30.0),
        report=report,
    )
    if completed.stdout.strip():
        raise ManagedLocalEmbeddingServiceError(
            "stop-verification",
            f"managed local embedding Docker service is still running: {config.service}",
            report=report,
        )
    report["stopped"] = True


@contextmanager
def managed_local_embedding_service(
    config: ManagedLocalEmbeddingServiceConfig | None,
) -> Iterator[dict[str, Any]]:
    """Start, health-check, and always stop the configured Compose service."""

    if config is None:
        yield {"managed": False, "skip_reason": "managed_local_embedding_disabled"}
        return

    report: dict[str, Any] = {
        "managed": True,
        "compose_dir": str(config.compose_dir),
        "service": config.service,
        "health_url": config.health_url,
        "start_requested": True,
        "started": False,
        "healthy": False,
        "stop_requested": False,
        "stopped": False,
    }
    try:
        _run_compose(
            config,
            ["up", "-d", config.service],
            phase="start",
            timeout=config.start_timeout,
            report=report,
        )
        report["started"] = True
        _wait_until_healthy(config, report)
        yield report
    finally:
        report["stop_requested"] = True
        _run_compose(
            config,
            ["stop", config.service],
            phase="stop",
            timeout=config.stop_timeout,
            report=report,
        )
        _verify_stopped(config, report)
