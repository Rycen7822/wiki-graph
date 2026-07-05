import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_FAST_VERIFIER = Path.home() / ".hermes" / "skills" / "research" / "llm-wiki" / "scripts" / "raw_fast_note_verify.py"
sys.path.insert(0, str(ROOT))

from support import sample_wiki, write  # noqa: E402
from ops import raw_fast_note_assemble  # noqa: E402
from ops.wiki_native_wiki_integration_pending import load_pending_wiki_integration_ledger  # noqa: E402


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
    report = json.loads((workdir / "assembled_raw_note_report.json").read_text(encoding="utf-8"))
    assert report["raw_file"] == raw_file


def test_raw_fast_note_assemble_rejects_agent_frontmatter(tmp_path: Path) -> None:
    root, workdir, _raw_file = _write_assembly_fixture(tmp_path)
    write(workdir / "raw_body_draft.md", "---\ntitle: Agent YAML\n---\n\n" + STRUCTURED_BODY)

    payload = raw_fast_note_assemble.assemble_raw_note(root=root, workdir=workdir)

    assert payload["ok"] is False
    assert payload["stage"] == "body_draft"
    assert payload["error"] == "body_frontmatter_forbidden"
    assert not (root / "raw/clip/2607/26070401_Assembler-Paper.md").exists()


def test_raw_fast_note_assemble_rejects_resource_leakage_in_body(tmp_path: Path) -> None:
    root, workdir, _raw_file = _write_assembly_fixture(tmp_path)
    write(workdir / "raw_body_draft.md", STRUCTURED_BODY + "\nResource status: HEAD 200 at https://github.com/example/assembler-paper\n")

    payload = raw_fast_note_assemble.assemble_raw_note(root=root, workdir=workdir)

    assert payload["ok"] is False
    assert payload["stage"] == "body_draft"
    assert payload["error"] == "body_resource_or_url_leakage"


def test_raw_fast_note_assemble_refuses_unsafe_and_existing_raw_paths(tmp_path: Path) -> None:
    root, workdir, raw_file = _write_assembly_fixture(tmp_path, raw_file="../bad.md")

    unsafe = raw_fast_note_assemble.assemble_raw_note(root=root, workdir=workdir)
    assert unsafe["ok"] is False
    assert unsafe["stage"] == "raw_file"
    assert unsafe["error"] == "invalid_raw_file"

    root, workdir, raw_file = _write_assembly_fixture(tmp_path, raw_file="raw/clip/2607/26070402_Existing-Paper.md")
    write(root / raw_file, "already here")
    existing = raw_fast_note_assemble.assemble_raw_note(root=root, workdir=workdir)
    assert existing["ok"] is False
    assert existing["stage"] == "raw_file"
    assert existing["error"] == "raw_file_exists"


def test_raw_fast_note_assemble_cli_can_verify_and_closeout_roundtrip(tmp_path: Path) -> None:
    root, workdir, raw_file = _write_assembly_fixture(tmp_path, raw_file="raw/clip/2607/26070403_Roundtrip-Paper.md")
    state = tmp_path / "state"

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

    closeout = subprocess.run(
        [
            sys.executable,
            "-m",
            "ops.raw_fast_closeout",
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--workdir",
            str(ROOT),
            "--raw-file",
            raw_file,
            "--title",
            "Assembler Paper",
            "--source-id",
            "https://arxiv.org/abs/2607.0401",
            "--pattern",
            "Assembler Paper",
            "--pattern",
            "https://arxiv.org/abs/2607.0401",
            "--resource-status-summary",
            "synthetic resources checked",
            "--tmp",
            str(workdir),
            "--auto-integrate",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    closed = json.loads(closeout.stdout)
    assert closed["raw_fast_ok"] is True
    assert closed["marked"]["raw_path"] == raw_file
    assert load_pending_wiki_integration_ledger(state)["pending"][0]["raw_path"] == raw_file
    assert not workdir.exists()
