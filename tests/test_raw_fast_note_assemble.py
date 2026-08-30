import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW_FAST_VERIFIER = Path.home() / ".hermes" / "skills" / "research" / "llm-wiki" / "scripts" / "raw_fast_note_verify.py"

from support import sample_wiki, write  # noqa: E402
from ops import raw_fast_note_assemble  # noqa: E402


STRUCTURED_BODY = """## 一句话总结

Assembler Paper shows that a body-only raw-fast draft can be combined with script-owned metadata without agent-written YAML.

## 论文摘要（中文）

This synthetic paper studies how deterministic assembly can keep metadata stable while the agent focuses on scientific prose and source-grounded interpretation.

## Motivation

The motivating problem is that manual frontmatter editing mixes operational metadata with reading-note synthesis and makes fast clipping brittle.

## Methodology

The method separates two channels: a script-owned metadata channel and an agent-owned body channel. The body explains the objective with $loss = x + y$ and keeps formulas in normal prose rather than separate buckets.

## 关键实验结果 / 作者结论

Figure 1 and Table 1 are integrated as prose evidence: the assembled output preserves structure, keeps the body URL-clean, and remains compatible with the existing verifier.

## 对未来研究的启发

Future raw-fast work can reduce prompt context by moving deterministic note scaffolding into scripts while preserving human-readable body synthesis.

## 可能的局限

The assembler cannot judge whether the scientific prose is insightful; it only enforces metadata ownership, body hygiene, and structural gates.

## 可继续追问的问题

Which additional semantic hints can be derived deterministically without reintroducing agent-written frontmatter?
"""


def _write_assembly_fixture(tmp_path: Path, *, raw_file: str = "raw/clip/2607/26070401_Assembler-Paper.md") -> tuple[Path, Path, str]:
    root = sample_wiki(tmp_path)
    workdir = tmp_path / "assemble-workdir"
    workdir.mkdir(parents=True, exist_ok=True)
    write(
        workdir / "candidate_frontmatter.json",
        json.dumps(
            {
                "title": "Assembler Paper",
                "created": "1999-01-01",
                "updated": "1999-01-01 00:00",
                "type": "raw-note",
                "domain": "paper",
                "source": "https://arxiv.org/abs/2607.0401",
                "github_links": ["https://github.com/example/assembler-paper"],
                "huggingface_model_links": ["https://huggingface.co/example/assembler-model"],
                "capture_route": "raw-fast evidence bundle (arxiv)",
                "captured": "1999-01-01 00:00:00 CST (+0800)",
            }
        ),
    )
    write(
        workdir / "agent_handoff.json",
        json.dumps(
            {
                "ok": True,
                "protected_anchors": {
                    "title": "Assembler Paper",
                    "source_url": "https://arxiv.org/abs/2607.0401",
                    "next_raw_path": raw_file,
                },
            }
        ),
    )
    write(workdir / "raw_body_draft.md", STRUCTURED_BODY)
    write(workdir / "evidence_report.json", json.dumps({"ok": True, "status": "test"}))
    return root, workdir, raw_file


def test_raw_fast_note_assemble_writes_script_frontmatter_and_body_draft(tmp_path: Path) -> None:
    root, workdir, raw_file = _write_assembly_fixture(tmp_path)

    payload = raw_fast_note_assemble.assemble_raw_note(root=root, workdir=workdir)

    assert payload["ok"] is True
    assert payload["raw_file"] == raw_file
    assert payload["body_checks"]["heading_count"] >= 8
    raw_path = root / raw_file
    assert raw_path.exists()
    text = raw_path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert 'title: "Assembler Paper"' in text
    assert 'source: "https://arxiv.org/abs/2607.0401"' in text
    assert "github_links:\n  - \"https://github.com/example/assembler-paper\"" in text
    assert "huggingface_model_links:\n  - \"https://huggingface.co/example/assembler-model\"" in text
    assert "1999-01-01" not in text
    body = text.split("---", 2)[-1]
    assert "## Methodology" in body
    assert "https://" not in body
    assert (workdir / "assembled_raw_note_report.json").exists()


def test_raw_fast_note_assemble_preserves_created_when_overwriting_existing_note(tmp_path: Path) -> None:
    root, workdir, raw_file = _write_assembly_fixture(tmp_path)
    raw_path = root / raw_file
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        '---\n'
        'title: "Existing Paper"\n'
        'source: "https://example.com/existing"\n'
        'created: "2001-02-03"\n'
        'updated: "2001-02-03 00:00"\n'
        'type: "raw-note"\n'
        'domain: "machine-learning"\n'
        'captured: "2001-02-03 00:00:00 UTC (+0000)"\n'
        '---\n\n'
        'old body\n',
        encoding="utf-8",
    )

    payload = raw_fast_note_assemble.assemble_raw_note(
        root=root,
        workdir=workdir,
        overwrite_existing=True,
    )

    assert payload["ok"] is True
    assert payload["preserved_created"] == "2001-02-03"
    text = raw_path.read_text(encoding="utf-8")
    assert 'created: "2001-02-03"' in text
    assert 'title: "Assembler Paper"' in text
    assert "old body" not in text


@pytest.mark.parametrize(
    "body,expected_error",
    [
        ("---\ntitle: Agent YAML\n---\n\n" + STRUCTURED_BODY, "body_frontmatter_forbidden"),
        (
            STRUCTURED_BODY + "\nResource status: HEAD 200 at https://github.com/example/assembler-paper\n",
            "body_resource_or_url_leakage",
        ),
    ],
)
def test_raw_fast_note_assemble_rejects_invalid_body(tmp_path: Path, body: str, expected_error: str) -> None:
    root, workdir, _raw_file = _write_assembly_fixture(tmp_path)
    write(workdir / "raw_body_draft.md", body)

    payload = raw_fast_note_assemble.assemble_raw_note(root=root, workdir=workdir)

    assert payload["ok"] is False
    assert payload["stage"] == "body_draft"
    assert payload["error"] == expected_error
    assert not (root / "raw/clip/2607/26070401_Assembler-Paper.md").exists()


def test_raw_fast_note_assemble_refuses_unsafe_and_reallocates_existing_sequence(tmp_path: Path) -> None:
    root, workdir, _raw_file = _write_assembly_fixture(tmp_path, raw_file="../bad.md")

    unsafe = raw_fast_note_assemble.assemble_raw_note(root=root, workdir=workdir)
    assert unsafe["ok"] is False
    assert unsafe["stage"] == "raw_file"
    assert unsafe["error"] == "invalid_raw_file"

    root, workdir, raw_file = _write_assembly_fixture(tmp_path, raw_file="raw/clip/2607/26070402_Existing-Paper.md")
    write(root / raw_file, "already here")
    write(
        workdir / "evidence_bundle.json",
        json.dumps(
            {
                "kind": "arxiv",
                "source_url": "https://arxiv.org/abs/2607.0401",
                "title_guess": "Assembler Paper",
                "next_raw_path": raw_file,
                "preflight": {"next_raw_path": raw_file, "source_identity": {"arxiv_id_base": "2607.0401"}},
                "files": {"evidence_report": "evidence_report.json"},
            }
        ),
    )
    state_dir = tmp_path / "state"
    allocated = raw_fast_note_assemble.assemble_raw_note(root=root, workdir=workdir, state_dir=state_dir)
    assert allocated["ok"] is True
    assert allocated["raw_file"] == "raw/clip/2607/26070403_Assembler-Paper.md"
    assert allocated["publication"]["reallocated"] is True
    assert (root / raw_file).read_text(encoding="utf-8") == "already here"
    assert (root / allocated["raw_file"]).is_file()
    closeout_args = json.loads((workdir / "closeout_args.json").read_text(encoding="utf-8"))
    assert closeout_args["raw_file"] == allocated["raw_file"]
    assert allocated["raw_file"] in (workdir / "closeout_command.preview.sh").read_text(encoding="utf-8")
    assert f"--state-dir {state_dir}" in (workdir / "assemble_command.preview.sh").read_text(encoding="utf-8")


@pytest.mark.external_skill
@pytest.mark.subprocess
def test_raw_fast_note_assemble_cli_can_verify_assembled_note(tmp_path: Path) -> None:
    if not RAW_FAST_VERIFIER.is_file():
        pytest.skip(f"external skill missing: {RAW_FAST_VERIFIER}")
    root, workdir, raw_file = _write_assembly_fixture(tmp_path, raw_file="raw/clip/2607/26070403_Roundtrip-Paper.md")

    assemble = subprocess.run(
        [
            sys.executable,
            "-m",
            "ops.raw_fast_note_assemble",
            "--root",
            str(root),
            "--workdir",
            str(workdir),
            "--verifier",
            str(RAW_FAST_VERIFIER),
            "--verify",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assembled = json.loads(assemble.stdout)
    assert assembled["ok"] is True
    assert assembled["verify"]["raw_fast_ok"] is True
    assert (root / raw_file).exists()
