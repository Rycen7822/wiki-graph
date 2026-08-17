from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

import pytest

from raw_fast_evidence_fixtures import (
    LEGACY_LAYOUT_TEXT_FILE,
    LEGACY_RAW_TEXT_FILE,
    LEGACY_RAW_TEXT_KEY,
    _fake_latexpand_success,
    _write_tex_agent_ir_source_fixture,
    _write_tiny_pdf,
    bundle_cli_argv,
    install_fake_arxiv_fetch,
    run_process_arxiv,
    sample_wiki,
    wiki_root_machine_pollution,
    write,
)


def test_raw_fast_evidence_bundle_arxiv_uses_tex_source_without_pdf_text_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ops import raw_fast_evidence_bundle

    root = sample_wiki(tmp_path)
    workdir = tmp_path / "tex-first-arxiv"

    def fake_fetch_text(url: str, timeout: int) -> dict:
        if "export.arxiv.org" in url:
            return {"ok": True, "status": 200, "text": "<feed><entry><title>TeX First Fixture Paper</title></entry></feed>"}
        return {"ok": True, "status": 200, "text": "<html><title>TeX First Fixture Paper</title></html>"}

    def forbidden_process_pdf(*args, **kwargs):
        raise AssertionError("arXiv TeX-success route must not call process_pdf or parse PDF text")

    install_fake_arxiv_fetch(
        monkeypatch,
        raw_fast_evidence_bundle,
        workdir,
        fetch_text=fake_fetch_text,
        write_source=_write_tex_agent_ir_source_fixture,
        extracted_count=4,
        extra=(("process_pdf", forbidden_process_pdf),),
    )
    _fake_latexpand_success(monkeypatch, raw_fast_evidence_bundle, workdir)

    payload = run_process_arxiv(
        raw_fast_evidence_bundle,
        "https://arxiv.org/abs/2600.00001",
        root,
        workdir,
        timed=True,
        resource_health="none",
    )

    assert payload["ok"] is True
    assert payload["pdf_backend_effective"] == "tex_source"
    assert LEGACY_RAW_TEXT_KEY not in payload["files"]
    read_plan = json.loads((workdir / "tex_read_plan.json").read_text(encoding="utf-8"))
    assert read_plan["source_kind"] == "tex_source"
    assert read_plan["main_tex"] == "source/main.tex"
    required_sidecars = {"tex_agent_ir", "tex_agent_source", "tex_agent_audit", "tex_agent_audit_markdown"}
    assert required_sidecars <= set(payload["files"])
    for key in required_sidecars:
        assert (workdir / payload["files"][key]).exists()
    retired_sidecars = {f"tex_agent_{suffix}" for suffix in ["map", "core", "objects", "full"]}
    assert not retired_sidecars & set(payload["files"])
    for retired in ["map", "core", "objects"]:
        assert not (workdir / f"paper_{retired}.md").exists()
    assert not (workdir / ("paper_" + "full.agent.md")).exists()

    filtered_tex = (workdir / payload["files"]["tex_agent_source"]).read_text(encoding="utf-8")
    audit_json = json.loads((workdir / payload["files"]["tex_agent_audit"]).read_text(encoding="utf-8"))
    ir_json = json.loads((workdir / payload["files"]["tex_agent_ir"]).read_text(encoding="utf-8"))

    assert "The filtered TeX source exposes theorem-like statements" in filtered_tex
    assert r"\mathcal{L}=x+y" in filtered_tex
    assert r"\includegraphics" in filtered_tex
    assert "Appendix Details" in filtered_tex
    assert "Supplementary detail should stay discoverable" in filtered_tex
    assert r"\title" not in filtered_tex
    assert r"\newcommand" not in filtered_tex
    assert r"\input" not in filtered_tex
    assert "Source comments should not pollute" not in filtered_tex
    assert "Bibliographic detail should not be in the filtered agent source" not in filtered_tex
    assert r"\bibliography" not in filtered_tex
    assert audit_json["coverage"]["statements"] >= 1
    assert ir_json["objects"]["statements"][0]["line_start"]
    assert ir_json["filtered_source_path"] == "paper_source.agent.tex"

    handoff = json.loads((workdir / "agent_handoff.json").read_text(encoding="utf-8"))
    assert handoff["source_read_plan"]["source_kind"] == "tex_source"
    assert not retired_sidecars & set(handoff["source_refs"])
    brief = json.loads((workdir / "agent_brief.json").read_text(encoding="utf-8"))
    assert brief["evidence_cards"][0]["kind"] == "abstract"


def test_tex_source_units_resolve_dotted_stem_includes(tmp_path: Path) -> None:
    from ops import raw_fast_evidence_bundle

    workdir = tmp_path / "dotted-stem-tex"
    write(
        workdir / "source" / "main.tex",
        r"""
\begin{document}
\input{sections/1.introduction}
\input{sections/method.v2}
\end{document}
""".strip(),
    )
    write(workdir / "source" / "sections" / "1.introduction.tex", r"\section{Introduction} Dotted numeric section stem.")
    write(workdir / "source" / "sections" / "method.v2.tex", r"\section{Method} Dotted method stem.")

    units, diagnostics = raw_fast_evidence_bundle.build_tex_source_units(workdir, "source/main.tex")

    assert diagnostics == []
    assert [unit["path"] for unit in units] == [
        "main.tex",
        "sections/1.introduction.tex",
        "sections/method.v2.tex",
    ]


@pytest.mark.requires_fitz
@pytest.mark.subprocess
@pytest.mark.parametrize("workdir_inside_wiki", [False, True], ids=["temp_workdir", "inside_wiki_root"])
def test_raw_fast_evidence_bundle_direct_pdf_respects_workdir_guard(tmp_path: Path, workdir_inside_wiki: bool) -> None:
    root = sample_wiki(tmp_path)
    source_pdf = tmp_path / "source.pdf"
    _write_tiny_pdf(source_pdf)
    workdir = root / "raw" / "clip" / "bundle" if workdir_inside_wiki else tmp_path / "bundle"
    result = subprocess.run(
        bundle_cli_argv(source_pdf.as_uri(), root, workdir),
        check=not workdir_inside_wiki,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    if workdir_inside_wiki:
        assert result.returncode != 0
        assert payload["stage"] == "preflight"
        assert payload["error"] == "workdir_inside_wiki_root"
        assert not (workdir / "evidence_bundle.json").exists()
        return

    assert payload["ok"] is True
    assert payload["kind"] == "direct-pdf"
    assert payload["pdf_backend_requested"] == "docling"
    assert payload["pdf_backend_effective"] == "docling"
    assert (workdir / "paper.pdf").exists()
    assert (workdir / "docling.md").exists()
    assert (workdir / "docling.json").exists()
    assert not (workdir / LEGACY_LAYOUT_TEXT_FILE).exists()
    assert not (workdir / LEGACY_RAW_TEXT_FILE).exists()
    assert (workdir / "source_read_plan.json").exists()
    assert (workdir / "note_skeleton.md").exists()
    assert (workdir / "evidence_bundle.json").exists()
    assert "raw/clip/" in payload["next_raw_path"]
    assert payload["source_read_plan"]["source_kind"] == "docling_pdf"
    secret_scan = json.loads((workdir / "secret_scan.json").read_text(encoding="utf-8"))
    assert secret_scan["strict_secret_hits"] == []
    assert not (root / "evidence_bundle.json").exists()
    assert wiki_root_machine_pollution(root) == []

@pytest.mark.requires_fitz
def test_raw_fast_evidence_bundle_paper_digest_prefers_arxiv_api_title_over_image_tex_title(tmp_path: Path) -> None:
    from ops import raw_fast_evidence_bundle

    workdir = tmp_path / "bundle-api-title"
    write(
        workdir / "api.xml",
        """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns='http://www.w3.org/2005/Atom'>
  <entry><title>API Grounded Paper Title</title></entry>
</feed>
""".strip(),
    )
    write(workdir / "source" / "main.tex", r"\title{<!-- image -->}\begin{document}\begin{abstract}API title should win.\end{abstract}\end{document}")

    digest = raw_fast_evidence_bundle.build_paper_digest(workdir, "<!-- image -->", "https://arxiv.org/abs/2600.00000")

    assert digest["metadata_card"]["title"] == "API Grounded Paper Title"


def test_raw_fast_evidence_bundle_arxiv_api_title_decodes_xml_entities(tmp_path: Path) -> None:
    from ops import raw_fast_evidence_bundle

    workdir = tmp_path / "bundle-api-entity-title"
    write(
        workdir / "api.xml",
        """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns='http://www.w3.org/2005/Atom'>
  <entry><title>ResearchArena: Automated AI R&amp;D</title></entry>
</feed>
""".strip(),
    )

    assert raw_fast_evidence_bundle.arxiv_api_title(workdir) == "ResearchArena: Automated AI R&D"


def test_raw_fast_evidence_bundle_title_fallbacks_use_abs_html_and_generic_title_macro(tmp_path: Path) -> None:
    from ops import raw_fast_evidence_bundle

    workdir = tmp_path / "bundle-title-fallbacks"
    write(workdir / "abs.html", '<meta name="citation_title" content="Abs &amp; HTML Grounded Title">')
    write(
        workdir / "source" / "main.tex",
        r"""
\paperTitle{Source Grounded Title}
\begin{document}
\begin{abstract}Title fallback should not become Untitled when the API is down.\end{abstract}
\end{document}
""".strip(),
    )

    assert raw_fast_evidence_bundle.arxiv_abs_title(workdir) == "Abs & HTML Grounded Title"
    assert raw_fast_evidence_bundle.tex_title_from_text((workdir / "source" / "main.tex").read_text()) == "Source Grounded Title"
    digest = raw_fast_evidence_bundle.build_paper_digest(workdir, "Untitled Paper", "https://arxiv.org/abs/2600.00000")

    assert digest["metadata_card"]["title"] == "Abs & HTML Grounded Title"


def test_raw_fast_evidence_bundle_preflight_and_agent_sidecars_are_tmp_only(tmp_path: Path) -> None:
    from ops import raw_fast_evidence_bundle

    root = sample_wiki(tmp_path)
    workdir = tmp_path / "agent-automation-sidecars"
    state_dir = tmp_path / "state"
    write(
        state_dir / "pending_wiki_integration.json",
        json.dumps({"pending": [{"raw_path": "raw/clip/2601/26010101_Foo-Paper.md", "status": "pending"}]}),
    )
    source_url = "https://arxiv.org/abs/2601.0101"
    title = "Foo Paper"
    frontmatter = raw_fast_evidence_bundle.build_frontmatter(title, source_url, "arxiv")
    preflight = raw_fast_evidence_bundle.build_raw_fast_preflight(
        root,
        title,
        source_url,
        "arxiv",
        workdir=workdir,
        state_dir=state_dir,
        now=dt.datetime(2026, 1, 1, 12, 0),
    )
    digest = {
        "ok": True,
        "metadata_card": {"title": title, "source_url": source_url},
        "abstract_cards": [{"text": "A compact abstract card grounded in the source."}],
        "section_cards": [{"section": "Method", "text": "A method card."}],
        "equation_cards": [{"label": "eq:loss", "text": "L = x"}],
        "table_cards": [],
        "figure_cards": [],
        "result_cards": [{"text": "A result card."}],
        "quality_warnings": ["source text is short"],
    }
    resource_boundary = raw_fast_evidence_bundle.summarize_resource_boundary(
        {"ok": True, "probes": [{"ok": True, "type": "arxiv", "id": "2601.0101", "status": "detected", "url": source_url}], "urls": [source_url]},
        metadata={"title": title, "source_url": source_url},
    )
    files: dict[str, str] = {}
    bundle_payload = {
        "ok": True,
        "kind": "arxiv",
        "source_url": source_url,
        "title_guess": title,
        "warnings": [],
        "files": files,
        "next_raw_path": preflight["next_raw_path"],
    }

    summary = raw_fast_evidence_bundle.write_agent_automation_sidecars(
        workdir,
        files,
        bundle_payload,
        frontmatter=frontmatter,
        preflight=preflight,
        digest=digest,
        resource_boundary=resource_boundary,
    )

    assert preflight["ok"] is True
    assert preflight["next_raw_path"] == "raw/clip/2601/26010102_Foo-Paper.md"
    assert summary["ok"] is True
    required_files = {
        "raw_fast_preflight",
        "agent_brief",
        "evidence_report",
        "note_candidate",
        "agent_handoff",
    }
    assert required_files <= set(files)
    for key in required_files:
        assert (workdir / files[key]).exists()
    brief = json.loads((workdir / "agent_brief.json").read_text(encoding="utf-8"))
    assert brief["protected_anchors"]["next_raw_path"] == preflight["next_raw_path"]
    assert brief["source_refs"]["body_draft"] == "raw_body_draft.md"
    evidence_report = json.loads((workdir / "evidence_report.json").read_text(encoding="utf-8"))
    assert evidence_report["resource_boundary"]["status"] == "standardized"
    handoff = json.loads((workdir / "agent_handoff.json").read_text(encoding="utf-8"))
    assert handoff["body_draft"] == {"path": "raw_body_draft.md", "contract": "body_only_no_frontmatter"}
    assert handoff["source_refs"]["scientific_digest"] == "paper_digest.md"
    assert "resource_status_summary" not in handoff
    candidate_md = (workdir / "note_candidate.md").read_text(encoding="utf-8")
    body = candidate_md.split("---", 2)[-1]
    assert all(marker not in body for marker in ["https://", "## 资源与复现状态", "## Evidence trail"])
    assert "<!-- advisory:" in body
    assert wiki_root_machine_pollution(root) == []
