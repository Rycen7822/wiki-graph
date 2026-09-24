"""Project-local llm-wiki skill packaging and path resolution."""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / ".agents" / "skills" / "llm-wiki"
LOADER = SKILL / "scripts" / "load_env_paths.py"


def _env_file(tmp_path: Path) -> Path:
    env_file = tmp_path / ".env"
    fields = {
        "WIKI_GRAPH_REPO": ROOT,
        "LLM_WIKI_ROOT": tmp_path / "wiki",
        "LLM_WIKI_STATE_DIR": tmp_path / "state",
        "LLM_WIKI_STORAGE_ROOT": tmp_path / "storage",
        "LLM_WIKI_RAW_FAST_TMP_ROOT": tmp_path / f"scratch folder;$(touch {tmp_path / 'unexpected'})",
        "LLM_WIKI_RAW_FAST_BATCH_TMP_ROOT": tmp_path / "batches",
        "OPENREVIEW_ENV_PATH": tmp_path / "private credential file",
    }
    env_file.write_text(
        "\n".join(f"{key}={value}" for key, value in fields.items())
        + "\nLLM_BINDING_API_KEY=fixture-only-sensitive-value\n",
        encoding="utf-8",
    )
    return env_file


def _run_loader(env_file: Path, mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(LOADER), "--env-file", str(env_file), mode],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.subprocess
def test_env_example_bootstraps_project_paths_without_secrets(tmp_path: Path) -> None:
    template = (ROOT / ".env.example").read_text(encoding="utf-8")
    assignments = dict(
        line.split("=", 1)
        for line in template.splitlines()
        if line and not line.startswith("#") and "=" in line
    )
    required_paths = {
        "WIKI_GRAPH_REPO",
        "LLM_WIKI_ROOT",
        "LLM_WIKI_STATE_DIR",
        "LLM_WIKI_STORAGE_ROOT",
        "LLM_WIKI_RAW_FAST_TMP_ROOT",
        "LLM_WIKI_RAW_FAST_BATCH_TMP_ROOT",
    }
    assert all(assignments.get(key, "").startswith("/absolute/path/") for key in required_paths)
    assert not any(re.search(r"(?:API_KEY|TOKEN|SECRET|PASSWORD)$", key) for key in assignments)
    assert not re.search(r"(?<![A-Za-z])(?:/home/|/mnt/[A-Za-z]/|~/|[A-Za-z]:[/\\])", template)
    env_file = tmp_path / ".env"
    env_file.write_text(
        template.replace("WIKI_GRAPH_REPO=/absolute/path/to/wiki-graph", f"WIKI_GRAPH_REPO={ROOT}"),
        encoding="utf-8",
    )
    result = _run_loader(env_file, "--check")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ok"] is True
    assert "/absolute/path/" not in result.stdout


@pytest.mark.subprocess
def test_project_skill_loader_exports_only_quoted_path_keys(tmp_path: Path) -> None:
    env_file = _env_file(tmp_path)
    check = _run_loader(env_file, "--check")
    assert check.returncode == 0, check.stderr
    report = json.loads(check.stdout)
    assert report["ok"] is True
    assert "LLM_WIKI_RAW_FAST_VERIFIER" in report["path_keys"]
    assert "fixture-only-sensitive-value" not in check.stdout
    assert str(tmp_path) not in check.stdout

    shell = _run_loader(env_file, "--shell")
    assert shell.returncode == 0, shell.stderr
    assert "LLM_BINDING_API_KEY" not in shell.stdout
    assert "fixture-only-sensitive-value" not in shell.stdout
    tricky_path = tmp_path / f"scratch folder;$(touch {tmp_path / 'unexpected'})"
    assert f"export LLM_WIKI_RAW_FAST_TMP_ROOT={shlex.quote(str(tricky_path))}" in shell.stdout
    assert f"export LLM_WIKI_SKILL_DIR={shlex.quote(str(SKILL))}" in shell.stdout
    assert f"export LLM_WIKI_RAW_FAST_VERIFIER={shlex.quote(str(SKILL / 'scripts/raw_fast_note_verify.py'))}" in shell.stdout
    assert "export OPENREVIEW_ENV_PATH=" in shell.stdout
    evaluated = subprocess.run(
        ["bash", "-c", 'eval "$1"; test "$LLM_WIKI_SKILL_DIR" = "$2" && test "$LLM_WIKI_RAW_FAST_TMP_ROOT" = "$3"',
         "bash", shell.stdout, str(SKILL), str(tricky_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert evaluated.returncode == 0, evaluated.stderr
    assert not (tmp_path / "unexpected").exists()


@pytest.mark.subprocess
def test_project_skill_loader_rejects_missing_or_wrong_repo(tmp_path: Path) -> None:
    env_file = _env_file(tmp_path)
    env_file.write_text(env_file.read_text().replace("LLM_WIKI_STATE_DIR=", "UNSET_STATE_DIR="))
    missing = _run_loader(env_file, "--shell")
    assert missing.returncode == 2
    assert missing.stdout == ""
    assert "LLM_WIKI_STATE_DIR" in missing.stderr
    assert "fixture-only-sensitive-value" not in missing.stderr

    env_file = _env_file(tmp_path)
    env_file.write_text(env_file.read_text().replace(f"WIKI_GRAPH_REPO={ROOT}", f"WIKI_GRAPH_REPO={tmp_path}"))
    wrong_repo = _run_loader(env_file, "--shell")
    assert wrong_repo.returncode == 2
    assert wrong_repo.stdout == ""
    assert "WIKI_GRAPH_REPO" in wrong_repo.stderr
    assert str(tmp_path) not in wrong_repo.stderr


@pytest.mark.subprocess
def test_raw_fast_handoff_uses_project_local_skill_when_configured() -> None:
    env = os.environ.copy()
    env["LLM_WIKI_SKILL_DIR"] = str(SKILL)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; from ops.raw_fast_evidence_bundle import WRITING_CONTRACT_REFS; "
            "from ops.raw_fast_ingest_prepare import MANUAL_REFERENCE_PATHS; "
            "root = Path(WRITING_CONTRACT_REFS[0]['path']).parents[1]; "
            "assert root == Path(MANUAL_REFERENCE_PATHS[0]).parents[1]; "
            "assert all(Path(path).is_file() for path in MANUAL_REFERENCE_PATHS); "
            "print(root.name)",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "llm-wiki"


@pytest.mark.subprocess
def test_raw_fast_defaults_to_tracked_project_skill() -> None:
    env = os.environ.copy()
    env.pop("LLM_WIKI_SKILL_DIR", None)
    env.pop("LLM_WIKI_RAW_FAST_VERIFIER", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; from ops.raw_fast_evidence_bundle import LLM_WIKI_SKILL_ROOT; "
            "from ops.raw_fast_closeout import DEFAULT_VERIFIER; "
            "from ops.raw_fast_note_assemble import DEFAULT_VERIFIER as ASSEMBLER_VERIFIER; "
            "assert (LLM_WIKI_SKILL_ROOT / 'SKILL.md').is_file(); "
            "assert DEFAULT_VERIFIER == LLM_WIKI_SKILL_ROOT / 'scripts' / 'raw_fast_note_verify.py'; "
            "assert ASSEMBLER_VERIFIER == DEFAULT_VERIFIER; "
            "print(LLM_WIKI_SKILL_ROOT.name)",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "llm-wiki"


def test_tracked_skill_contains_no_machine_specific_paths() -> None:
    candidates = (p for p in SKILL.rglob("*") if p.is_file() and p.suffix in {".md", ".py", ".yaml", ".json"})
    machine_path = re.compile(r"(?<![A-Za-z])(?:/home/|/mnt/[A-Za-z]/|~/|[A-Za-z]:[/\\])")
    offenders = [str(p.relative_to(SKILL)) for p in candidates if machine_path.search(p.read_text(encoding="utf-8"))]
    assert not offenders
