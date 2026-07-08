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

LEGACY_EXTRACTOR_KEY = "pdf" + "totext"
LEGACY_RAW_TEXT_KEY = "raw" + "_text"
LEGACY_LAYOUT_TEXT_KEY = "layout" + "_text"


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
    expected_handoff = str((Path(payload["workdir"]) / "agent_handoff.md").resolve())
    assert payload["agent_next_reads"][0] == expected_handoff
    assert any(path.endswith("structured-paper-note-contract.md") for path in payload["agent_next_reads"])
    assert payload["writing_contract_refs"][0]["path"].endswith("structured-paper-note-contract.md")
    assert payload["automation_next_action"]["action"] == "read_agent_handoff_then_writing_contract"
    assert payload["automation_next_action"]["read_path"] == payload["agent_next_reads"][0]
    assert payload["manual_reference_policy"]["mode"] == "only_on_manual_required"
    assert payload["manual_reference_policy"]["manual_references_visible"] is False
    command = payload["command"]
    assert command[command.index("--root") + 1] == str(raw_fast_ingest_prepare.PROD_WIKI_ROOT)
    assert command[command.index("--state-dir") + 1] == str(raw_fast_ingest_prepare.PROD_STATE_DIR)
    assert "manual_reference_paths" not in payload


def test_raw_fast_ingest_prepare_print_command_normalizes_github_blob_pdf_to_raw_download(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ops.raw_fast_ingest_prepare",
            "--url",
            "https://github.com/areal-project/AReaL/blob/main/docs/paper/AReaL2.0_report.pdf",
            "--tmp-root",
            str(tmp_path / "raw-fast-tmp"),
            "--print-command",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    command = payload["command"]

    normalized = "https://raw.githubusercontent.com/areal-project/AReaL/main/docs/paper/AReaL2.0_report.pdf"
    assert payload["source_url"] == normalized
    assert payload["supplied_url"] == "https://github.com/areal-project/AReaL/blob/main/docs/paper/AReaL2.0_report.pdf"
    assert payload["source_url_normalization"]["normalized"] is True
    assert payload["source_url_normalization"]["reason"] == "github_blob_pdf"
    assert command[command.index("--url") + 1] == normalized
    assert command[command.index("--kind") + 1] == "direct-pdf"


def test_raw_fast_ingest_prepare_print_command_plumbs_cross_site_arxiv_kind(tmp_path: Path) -> None:
    urls = [
        "https://paperswithcode.co/paper/2606.28436",
        "https://modelscope.ai/papers/2606.07591",
    ]
    for url in urls:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ops.raw_fast_ingest_prepare",
                "--url",
                url,
                "--tmp-root",
                str(tmp_path / "raw-fast-tmp"),
                "--print-command",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        payload = json.loads(result.stdout)

        command = payload["command"]
        assert command[command.index("--kind") + 1] == "arxiv"


def test_raw_fast_ingest_prepare_print_command_plumbs_openreview_canonical_pdf(tmp_path: Path) -> None:
    url = "https://openreview.net/attachment?id=mLhZzo7BIb&name=pdf"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ops.raw_fast_ingest_prepare",
            "--url",
            url,
            "--tmp-root",
            str(tmp_path / "raw-fast-tmp"),
            "--print-command",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    command = payload["command"]
    canonical_pdf = "https://openreview.net/pdf?id=mLhZzo7BIb"

    assert payload["source_url"] == canonical_pdf
    assert payload["supplied_url"] == url
    assert payload["source_url_normalization"]["normalized"] == (url != canonical_pdf)
    assert payload["source_url_normalization"]["reason"] == "openreview_canonical_pdf"
    assert command[command.index("--url") + 1] == canonical_pdf
    assert command[command.index("--kind") + 1] == "openreview"


def test_raw_fast_ingest_prepare_print_command_passes_max_download_bytes(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ops.raw_fast_ingest_prepare",
            "--url",
            "https://openreview.net/forum?id=mLhZzo7BIb",
            "--tmp-root",
            str(tmp_path / "raw-fast-tmp"),
            "--max-download-bytes",
            "none",
            "--print-command",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    command = payload["command"]

    assert command[command.index("--max-download-bytes") + 1] == "none"


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
    assert output["manual_reference_policy"]["mode"] == "only_on_manual_required"
    assert output["automation_next_action"]["action"] == "read_manual_reference_paths"
    assert output["automation_next_action"]["reason"] == "script_failed"
    assert any(path.endswith("structured-paper-ingest-router.md") for path in output["manual_reference_paths"])
    assert "diagnostic_hint" in output
    assert "manual_reference_paths" in output["diagnostic_hint"] or "Access is blocked" in output["diagnostic_hint"]
    assert output["agent_next_reads"] == []


def test_raw_fast_ingest_prepare_main_failure_raises_manual_reference_reminder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fake_run_json_command(command, *, cwd, timeout):
        return {
            "returncode": 2,
            "json": {"ok": False, "stage": "fetch_pdf", "error": "HTTPError", "message": "403 Forbidden"},
            "stdout_tail": "{}",
            "stderr_tail": "fetch failed",
            "command": command,
        }

    monkeypatch.setattr(raw_fast_ingest_prepare, "run_json_command", fake_run_json_command)

    code = raw_fast_ingest_prepare.main(
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
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == 1
    assert payload["manual_required"] is True
    assert payload["manual_reference_paths"]
    assert payload["diagnostic_hint"]
    assert "read only manual_reference_paths" in captured.err


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
    assert evidence["pdf_backend_effective"] == "docling"
    assert evidence["source_read_plan"]["source_kind"] == "docling_pdf"
    for key in ["raw_fast_preflight", "agent_brief", "evidence_report", "note_candidate"]:
        assert key in evidence["files"]
        assert (workdir / evidence["files"][key]).exists()
    assert payload["agent_next_reads"][0] == str((workdir / "agent_handoff.md").resolve())
    assert any(path.endswith("structured-paper-note-contract.md") for path in payload["agent_next_reads"])
    assert payload["writing_contract_refs"][0]["read_before"] == "source_read_plan"
    assert payload["automation_next_action"]["action"] == "read_agent_handoff_then_writing_contract"
    assert payload["manual_reference_policy"]["manual_references_visible"] is False
    assert payload["assemble_command_preview_path"] == str((workdir / "assemble_command.preview.sh").resolve())
    for rel in ["agent_handoff.json", "agent_handoff.md", "closeout_args.json", "closeout_command.preview.sh", "assemble_command.preview.sh"]:
        assert (workdir / rel).exists()
    handoff = json.loads((workdir / "agent_handoff.json").read_text(encoding="utf-8"))
    assert handoff["automation_next_action"]["action"] == "read_writing_contract_then_follow_source_read_plan"
    assert handoff["body_draft"] == {"path": "raw_body_draft.md", "contract": "body_only_no_frontmatter"}
    assert handoff["source_refs"]["scientific_digest"] == "paper_digest.md"
    assert handoff["assemble"]["command_preview_path"] == str((workdir / "assemble_command.preview.sh").resolve())
    assert handoff["closeout_args"]["ok"] is True
    preview = (workdir / "assemble_command.preview.sh").read_text(encoding="utf-8")
    assert "ops.raw_fast_note_assemble" in preview
    assert "raw_body_draft.md" in preview
    closeout_args = json.loads((workdir / "closeout_args.json").read_text(encoding="utf-8"))
    assert closeout_args["ok"] is True
    assert "--resource-status-summary" in closeout_args["argv_tail"]
    assert wiki_root_machine_pollution(root) == []


def test_raw_fast_ingest_prepare_wrapper_preserves_filtered_tex_fallback_contract(tmp_path: Path) -> None:
    workdir = tmp_path / "filtered-tex-wrapper"
    workdir.mkdir()
    handoff_path = workdir / "agent_handoff.json"
    handoff_path.write_text(
        json.dumps(
            {
                "ok": True,
                "status": "ready",
                "resource_review_required": False,
                "manual_reference_paths": [],
                "manual_reference_policy": raw_fast_ingest_prepare.manual_reference_policy(visible=False),
                "automation_next_action": {"action": "read_writing_contract_then_filtered_tex_source"},
                "source_read_plan": {
                    "source_kind": "tex_source",
                    "first_reads": [{"path": "source/sections/method.tex", "offset": 1, "limit": 180, "reason": "source overview"}],
                },
                "source_refs": {
                    "scientific_digest": "paper_digest.md",
                    "tex_agent_source": "paper_source.agent.tex",
                    "tex_agent_audit": "tex_agent_ir_audit.md",
                },
                "protected_anchors": {"title": "Semantic Wrapper Fixture", "next_raw_path": "raw/clip/2607/26070799_Semantic-Wrapper-Fixture.md"},
                "quality_gate": dict(raw_fast_ingest_prepare.RAW_FAST_QUALITY_GATE),
                "writing_contract_refs": [dict(ref) for ref in raw_fast_ingest_prepare.WRITING_CONTRACT_REFS],
                "duplicate_summary": {},
                "evidence_cards": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    closeout = {
        "closeout_args": {"ok": True, "raw_file": "raw/clip/2607/26070799_Semantic-Wrapper-Fixture.md"},
        "closeout_args_path": str(workdir / "closeout_args.json"),
        "closeout_command_preview_path": str(workdir / "closeout_command.preview.sh"),
    }
    assemble = {"command_preview_path": str(workdir / "assemble_command.preview.sh"), "report_path": str(workdir / "assembled_raw_note_report.json"), "body_draft_path": str(workdir / "raw_body_draft.md"), "raw_file": "raw/clip/2607/26070799_Semantic-Wrapper-Fixture.md"}

    handoff = raw_fast_ingest_prepare.update_agent_handoff(workdir, closeout, assemble=assemble)
    handoff_md = (workdir / "agent_handoff.md").read_text(encoding="utf-8")

    joined_actions = "\n".join(handoff["agent_actions"])
    assert "source_refs/source_read_plan route" not in joined_actions
    assert "filtered original TeX source" in joined_actions
    assert "paper_source.agent.tex" in joined_actions
    assert "parsed " + "Markdown sidecars" not in joined_actions
    for retired in ["map", "core", "objects"]:
        assert f"paper_{retired}.md" not in joined_actions
    assert "fallback source span:" not in handoff_md
