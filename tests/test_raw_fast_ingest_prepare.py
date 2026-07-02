import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from support import sample_wiki  # noqa: E402
from ops import raw_fast_ingest_prepare  # noqa: E402
from ops.wiki_native_wiki_checks import wiki_root_machine_pollution  # noqa: E402


def _write_tiny_pdf(path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Tiny Prepare Paper\nAbstract\nThis paper has Figure 1 and DOI 10.1234/example.prepare")
    doc.save(path)
    doc.close()


def test_raw_fast_ingest_prepare_url_only_prod_profile_prints_command(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ops.raw_fast_ingest_prepare",
            "--url",
            "https://arxiv.org/abs/2606.32032",
            "--tmp-root",
            str(tmp_path / "raw-fast-tmp"),
            "--print-command",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["ok"] is True
    assert payload["profile"] == "prod"
    assert payload["root"] == str(raw_fast_ingest_prepare.PROD_WIKI_ROOT)
    assert payload["state_dir"] == str(raw_fast_ingest_prepare.PROD_STATE_DIR)
    assert payload["workdir"].startswith(str((tmp_path / "raw-fast-tmp").resolve()))
    assert payload["agent_next_reads"] == [str((Path(payload["workdir"]) / "agent_handoff.md").resolve())]
    command = payload["command"]
    assert command[command.index("--root") + 1] == str(raw_fast_ingest_prepare.PROD_WIKI_ROOT)
    assert command[command.index("--state-dir") + 1] == str(raw_fast_ingest_prepare.PROD_STATE_DIR)
    assert "manual_reference_paths" not in payload


def test_raw_fast_ingest_prepare_failure_payload_exposes_manual_reference_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = raw_fast_ingest_prepare.parse_args(
        [
            "--url",
            "https://example.test/fail.pdf",
            "--root",
            str(tmp_path / "wiki"),
            "--state-dir",
            str(tmp_path / "state"),
            "--tmp-root",
            str(tmp_path / "tmp"),
        ]
    )
    paths = raw_fast_ingest_prepare.resolve_prepare_paths(args)

    def fake_run_json_command(command, *, cwd, timeout):
        return {
            "returncode": 2,
            "json": {"ok": False, "stage": "fetch_pdf", "error": "HTTPError", "message": "403 Forbidden"},
            "stdout_tail": "{}",
            "stderr_tail": "fetch failed",
            "command": command,
        }

    monkeypatch.setattr(raw_fast_ingest_prepare, "run_json_command", fake_run_json_command)

    output = raw_fast_ingest_prepare.run_prepare(args, paths)

    assert output["ok"] is False
    assert output["manual_required"] is True
    assert output["manual_reason"]["error"] == "HTTPError"
    assert "manual_reference_paths" in output
    assert any(path.endswith("structured-paper-ingest-router.md") for path in output["manual_reference_paths"])
    assert output["agent_next_reads"] == []


@pytest.mark.requires_fitz
def test_raw_fast_ingest_prepare_wrapper_writes_single_handoff_and_closeout_args(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    source_pdf = tmp_path / "source.pdf"
    _write_tiny_pdf(source_pdf)
    workdir = tmp_path / "prepare-workdir"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ops.raw_fast_ingest_prepare",
            "--url",
            source_pdf.as_uri(),
            "--kind",
            "direct-pdf",
            "--root",
            str(root),
            "--workdir",
            str(workdir),
            "--probe",
            "none",
            "--pdf-backend",
            "pdftotext",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["ok"] is True
    assert payload["stage"] == "prepared"
    assert payload["workdir"] == str(workdir.resolve())
    evidence = payload["evidence_bundle"]
    assert evidence["ok"] is True
    for key in ["raw_fast_preflight", "agent_brief", "evidence_report", "note_candidate"]:
        assert key in evidence["files"]
        assert (workdir / evidence["files"][key]).exists()
    assert payload["agent_next_reads"] == [str((workdir / "agent_handoff.md").resolve())]
    assert payload["ready_message"].startswith("Raw-fast evidence prepared")
    assert payload["resource_review_required"] is False
    for rel in ["agent_handoff.json", "agent_handoff.md", "closeout_args.json", "closeout_command.preview.sh"]:
        assert (workdir / rel).exists()
    handoff = json.loads((workdir / "agent_handoff.json").read_text(encoding="utf-8"))
    assert handoff["resource_review_required"] is False
    assert handoff["manual_reference_paths"] == []
    assert handoff["closeout_args"]["ok"] is True
    assert handoff["closeout_args_path"] == str((workdir / "closeout_args.json").resolve())
    closeout_args = json.loads((workdir / "closeout_args.json").read_text(encoding="utf-8"))
    assert closeout_args["ok"] is True
    assert "--resource-status-summary" in closeout_args["argv_tail"]
    assert wiki_root_machine_pollution(root) == []
