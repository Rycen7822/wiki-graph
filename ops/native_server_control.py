#!/usr/bin/env python3
"""Reusable control plane for the llm-wiki native API server.

This module replaces one-off restart shell snippets. It reads the repository
`.env`, derives a single native server configuration, and can either print the
configuration/command or restart the process guarded by pidfile + health check.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request

from llm_wiki_native.contracts import DEFAULT_NATIVE_PORT
from ops.native_runtime_env import load_env_file, redact_summary

REPO_ROOT = Path(__file__).resolve().parents[1]


def _first_env(env: dict[str, str], *names: str) -> str | None:
    for name in names:
        value = env.get(name) or os.environ.get(name)
        if value not in (None, ""):
            return str(value)
    return None


def _env_path(env: dict[str, str], *names: str, default: Path | None = None, required: bool = False) -> Path:
    raw = _first_env(env, *names)
    if raw:
        return Path(raw).expanduser()
    if default is not None and not required:
        return default
    joined = " or ".join(names)
    raise ValueError(f"{joined} must be set in .env")


def _env_int(env: dict[str, str], *names: str, default: int) -> int:
    raw = _first_env(env, *names)
    if raw in (None, ""):
        return default
    try:
        return int(str(raw))
    except ValueError as exc:
        joined = " or ".join(names)
        raise ValueError(f"{joined} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class NativeServerConfig:
    env_file: Path
    repo: Path
    state_dir: Path
    host: str
    port: int
    workspace_file: Path
    log_dir: Path
    pidfile: Path
    health_url: str
    stop_timeout_seconds: float = 4.0
    health_timeout_seconds: float = 60.0
    health_interval_seconds: float = 1.0

    @property
    def server_command(self) -> list[str]:
        return [
            sys.executable,
            "-m",
            "llm_wiki_native.api.server",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--state-dir",
            str(self.state_dir),
            "--workspace-file",
            str(self.workspace_file),
            "--allow-workspace-status",
            "active",
        ]

    def summary(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        payload["server_command"] = self.server_command
        return payload


def resolve_config(env_file: Path | None = None) -> NativeServerConfig:
    target_env = (env_file or (REPO_ROOT / ".env")).expanduser()
    values = load_env_file(target_env)
    repo = _env_path(values, "WIKI_GRAPH_REPO", "LLM_WIKI_WORKDIR", default=REPO_ROOT, required=True).resolve()
    state_dir = _env_path(values, "LLM_WIKI_STATE_DIR", required=True).resolve()
    host = _first_env(values, "LLM_WIKI_NATIVE_HOST", "HOST") or "127.0.0.1"
    port = _env_int(values, "LLM_WIKI_NATIVE_PORT", "PORT", default=DEFAULT_NATIVE_PORT)
    workspace_file = _env_path(
        values,
        "LLM_WIKI_NATIVE_WORKSPACE_FILE",
        default=state_dir / "native_zvec" / "active_workspace.json",
    ).resolve()
    log_dir = _env_path(
        values,
        "LLM_WIKI_NATIVE_SERVER_LOG_DIR",
        default=state_dir / "native_zvec" / "server_logs",
    ).resolve()
    pidfile = _env_path(
        values,
        "LLM_WIKI_NATIVE_SERVER_PIDFILE",
        default=state_dir / "native_zvec" / f"native_server_{port}.pid",
    ).resolve()
    health_url = _first_env(values, "LLM_WIKI_NATIVE_HEALTH_URL", "LLM_WIKI_SERVER") or f"http://{host}:{port}/health"
    if health_url.rstrip("/").endswith(f":{port}"):
        health_url = health_url.rstrip("/") + "/health"
    return NativeServerConfig(
        env_file=target_env.resolve(strict=False),
        repo=repo,
        state_dir=state_dir,
        host=host,
        port=port,
        workspace_file=workspace_file,
        log_dir=log_dir,
        pidfile=pidfile,
        health_url=health_url,
    )


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _read_pid(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def stop_pidfile_process(pidfile: Path, *, timeout_seconds: float) -> dict[str, Any]:
    pid = _read_pid(pidfile)
    if pid is None:
        return {"pidfile": str(pidfile), "old_pid": None, "stopped": False, "reason": "pidfile_missing_or_invalid"}
    if not _pid_is_running(pid):
        return {"pidfile": str(pidfile), "old_pid": pid, "stopped": False, "reason": "pid_not_running"}
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _pid_is_running(pid):
            return {"pidfile": str(pidfile), "old_pid": pid, "stopped": True, "killed": False}
        time.sleep(0.2)
    if _pid_is_running(pid):
        os.kill(pid, signal.SIGKILL)
        return {"pidfile": str(pidfile), "old_pid": pid, "stopped": True, "killed": True}
    return {"pidfile": str(pidfile), "old_pid": pid, "stopped": True, "killed": False}


def _tail_text(path: Path, *, max_lines: int = 120) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    return "\n".join(lines[-max_lines:])


def wait_for_health(process: subprocess.Popen[Any], *, health_url: str, log_path: Path, timeout_seconds: float, interval_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"native server exited before health became ready: pid={process.pid} returncode={returncode} log={log_path}\n{_tail_text(log_path)}"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=min(3.0, max(1.0, interval_seconds + 1.0))) as response:
                body = response.read().decode("utf-8", errors="replace")
                status = int(getattr(response, "status", getattr(response, "code", 0)) or 0)
            if 200 <= status < 300:
                return {"ok": True, "url": health_url, "status": status, "body": body}
            last_error = f"status={status}"
        except (OSError, urllib.error.URLError) as exc:
            last_error = repr(exc)
        time.sleep(interval_seconds)
    raise RuntimeError(
        f"native server health did not become ready: pid={process.pid} url={health_url} last_error={last_error} log={log_path}\n{_tail_text(log_path)}"
    )


def restart_native_server(config: NativeServerConfig) -> dict[str, Any]:
    config.log_dir.mkdir(parents=True, exist_ok=True)
    config.pidfile.parent.mkdir(parents=True, exist_ok=True)
    stopped = stop_pidfile_process(config.pidfile, timeout_seconds=config.stop_timeout_seconds)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = config.log_dir / f"native_server_{stamp}.log"
    log_handle = log_path.open("ab")
    try:
        process = subprocess.Popen(
            config.server_command,
            cwd=str(config.repo),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()
    config.pidfile.write_text(f"{process.pid}\n", encoding="utf-8")
    health = wait_for_health(
        process,
        health_url=config.health_url,
        log_path=log_path,
        timeout_seconds=config.health_timeout_seconds,
        interval_seconds=config.health_interval_seconds,
    )
    return {
        "ok": True,
        "service": "llm-wiki-native",
        "pid": process.pid,
        "pidfile": str(config.pidfile),
        "log": str(log_path),
        "stopped": stopped,
        "health": health,
        "config": config.summary(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Control the reusable llm-wiki native API server")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("config", help="Print resolved server config without starting anything")
    restart = sub.add_parser("restart", help="Restart native API server from .env-backed config")
    restart.add_argument("--dry-run", action="store_true", help="Print the resolved command without killing or starting a process")
    args = parser.parse_args(argv)
    config = resolve_config(args.env_file)
    if args.command == "config" or getattr(args, "dry_run", False):
        payload = {"ok": True, "dry_run": bool(getattr(args, "dry_run", False)), "config": config.summary()}
        payload["env"] = redact_summary(load_env_file(config.env_file))
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    result = restart_native_server(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
