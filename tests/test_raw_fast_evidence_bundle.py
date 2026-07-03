import sys
import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW_FAST_VERIFIER = Path.home() / ".hermes" / "skills" / "research" / "llm-wiki" / "scripts" / "raw_fast_note_verify.py"
sys.path.insert(0, str(ROOT))

from support import sample_wiki, write  # noqa: E402
from ops import raw_fast_closeout  # noqa: E402
from ops.wiki_native_wiki_checks import wiki_root_machine_pollution  # noqa: E402
from ops.wiki_native_wiki_integration_pending import load_pending_wiki_integration_ledger  # noqa: E402

LEGACY_EXTRACTOR_KEY = "pdf" + "totext"
LEGACY_RAW_TEXT_KEY = "raw" + "_text"
LEGACY_LAYOUT_TEXT_KEY = "layout" + "_text"
LEGACY_RAW_TEXT_FILE = "paper." + "raw" + ".txt"
LEGACY_LAYOUT_TEXT_FILE = "paper." + "layout" + ".txt"


def _write_tiny_pdf(path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Tiny Paper\nAbstract\nThis paper has Eq. (1), Figure 1, Table 1, and https://github.com/example/tiny-paper",
    )
    doc.save(path)
    doc.close()
def _assert_timing_step(payload: dict, step: str) -> None:
    timings = payload["timings"]
    entry = timings["steps"][step]
    assert isinstance(entry["elapsed_seconds"], (int, float))
    assert entry["elapsed_seconds"] >= 0
def _write_tiny_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000a49444154789c636000000200015d0b2a0000000049454e44ae426082"))
def _write_tiny_pdf_figure(path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=144, height=96)
    page.draw_rect(fitz.Rect(12, 12, 132, 84), color=(0, 0, 0), fill=(0.9, 0.95, 1.0))
    page.insert_text((24, 48), "PDF Figure")
    doc.save(path)
    doc.close()
def _write_sidecar_source_fixture(workdir: Path) -> None:
    _write_tiny_png(workdir / "source" / "figs" / "method.png")
    write(
        workdir / "source" / "main.tex",
        r"""
\title{Fixture Sidecar Paper}
\begin{document}
\begin{abstract}
This paper introduces Sidecar Method with exact evidence and an official code link.
\end{abstract}
\section{Method}
We optimize $\mathcal{L}=x+y$ and release code at \url{https://github.com/example/sidecar-paper}.
\begin{equation}
\mathcal{L}=x+y
\label{eq:loss}
\end{equation}
\begin{figure}
\includegraphics{figs/method.png}
\caption{Method figure shows the pipeline.}
\label{fig:method}
\end{figure}
\begin{table}
\caption{Main results on the fixture benchmark.}
\label{tab:main}
\begin{tabular}{lr}
Method & Score \\
Sidecar & 42
\end{tabular}
\end{table}
\section{Limitations}
The fixture has one synthetic limitation.
\end{document}
""".strip(),
    )


def _write_tex_first_source_fixture(workdir: Path) -> None:
    write(workdir / "source" / "defs.tex", r"\newcommand{\loss}{\mathcal{L}}")
    write(
        workdir / "source" / "main.tex",
        r"""
\title{TeX First Fixture Paper}
\begin{document}
\maketitle
\begin{abstract}
This paper tests that raw-fast uses arXiv TeX source before any PDF text extraction.
\end{abstract}
\section{Introduction}
The TeX source contains the main paper prose and should be the default evidence.
\section{Method}
The method minimizes $\loss=x+y$ and defines the exact objective in source.
\begin{equation}
\loss=x+y
\label{eq:tex-first-loss}
\end{equation}
\section{Results}
The TeX-backed result states that source-first handoff is sufficient.
\section{Limitations}
The fixture limitation is synthetic and intentionally short.
\end{document}
""".strip(),
    )


def _write_fake_docling_outputs(workdir: Path, markdown: str = "# Docling Fallback Paper\n\nAbstract\nDocling text is the PDF fallback.\n\n## Method\nFallback method text.") -> None:
    write(workdir / "docling.md", markdown)
    write(workdir / "docling.json", json.dumps({"ok": True, "markdown_chars": len(markdown)}) + "\n")


def test_raw_fast_evidence_bundle_title_from_text_skips_docling_image_placeholder() -> None:
    from ops import raw_fast_evidence_bundle

    text = "<!-- image -->\n\n## Next-Generation Agentic Reinforcement Learning Systems Enable Self-Evolving Agents\n\n## Abstract\nBody"

    assert raw_fast_evidence_bundle.title_from_text(text) == "Next-Generation Agentic Reinforcement Learning Systems Enable Self-Evolving Agents"


def test_raw_fast_evidence_bundle_pdf_title_falls_back_to_url_stem_when_docling_only_has_placeholders() -> None:
    from ops import raw_fast_evidence_bundle

    title = raw_fast_evidence_bundle.pdf_title_from_text_or_url("<!-- image -->\n\n<!-- formula-not-decoded -->", "https://raw.githubusercontent.com/areal-project/AReaL/main/docs/paper/AReaL2.0_report.pdf")

    assert title == "AReaL2.0_report"


@pytest.mark.parametrize(
    ("url", "expected_id"),
    [
        ("https://paperswithcode.co/paper/2606.28436", "2606.28436"),
        ("https://paperswithcode.co/paper/arxiv/2606.30410v2", "2606.30410"),
        ("https://huggingface.co/papers/2606.02572", "2606.02572"),
        ("https://www.alphaxiv.org/abs/2606.04036", "2606.04036"),
        ("https://alphaxiv.org/overview/2606.04036v3?tab=discussion", "2606.04036"),
    ],
)
def test_raw_fast_evidence_bundle_recognizes_cross_site_arxiv_routes(url: str, expected_id: str) -> None:
    from ops import raw_fast_evidence_bundle

    assert raw_fast_evidence_bundle.arxiv_id_from_url(url) == expected_id
    assert raw_fast_evidence_bundle.detect_kind(url, "auto") == "arxiv"


def test_raw_fast_evidence_bundle_cross_site_arxiv_uses_canonical_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ops import raw_fast_evidence_bundle

    root = sample_wiki(tmp_path)
    workdir = tmp_path / "cross-site-arxiv"
    supplied_url = "https://huggingface.co/papers/2606.02572"

    def fake_fetch_text(url: str, timeout: int) -> dict:
        if "export.arxiv.org" in url:
            return {"ok": True, "status": 200, "text": "<feed><entry><title>Cross Site Fixture Paper</title></entry></feed>"}
        return {"ok": True, "status": 200, "text": "<html><title>Cross Site Fixture Paper</title></html>"}

    def fake_fetch_url_to_file(url: str, dest: Path, timeout: int) -> dict:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake tar")
        return {"ok": True, "url": url, "dest": str(dest), "bytes": dest.stat().st_size, "sha256": "fake"}

    def fake_extract_tar(tar_path: Path, dest: Path) -> dict:
        _write_tex_first_source_fixture(workdir)
        return {"ok": True, "extracted_count": 2, "errors": []}

    monkeypatch.setattr(raw_fast_evidence_bundle, "fetch_text", fake_fetch_text)
    monkeypatch.setattr(raw_fast_evidence_bundle, "fetch_url_to_file", fake_fetch_url_to_file)
    monkeypatch.setattr(raw_fast_evidence_bundle, "safe_extract_tar", fake_extract_tar)

    payload = raw_fast_evidence_bundle.process_arxiv(
        supplied_url,
        root,
        workdir,
        "docling",
        False,
        ["none"],
        30,
        paper_digest=True,
        resource_draft=True,
    )

    assert payload["ok"] is True
    assert payload["kind"] == "arxiv"
    assert payload["source_url"] == "https://arxiv.org/abs/2606.02572"
    assert payload["supplied_url"] == supplied_url
    assert payload["arxiv"]["id"] == "2606.02572"
    assert payload["arxiv"]["supplied_url"] == supplied_url
    frontmatter = json.loads((workdir / "candidate_frontmatter.json").read_text(encoding="utf-8"))
    assert frontmatter["source"] == "https://arxiv.org/abs/2606.02572"
    assert supplied_url not in frontmatter.values()


def test_raw_fast_evidence_bundle_process_pdf_has_named_sidecar_seams() -> None:
    from ops import raw_fast_evidence_bundle

    expected_helpers = {
        "_write_pdf_resource_boundary_sidecars",
        "_write_pdf_localized_figure_sidecars",
        "_write_pdf_digest_sidecars",
        "_write_pdf_note_block_drafts",
        "_write_pdf_enrichment_sidecars",
    }

    assert {name for name in expected_helpers if callable(getattr(raw_fast_evidence_bundle, name, None))} == expected_helpers


def test_raw_fast_evidence_bundle_arxiv_uses_tex_source_without_pdf_text_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ops import raw_fast_evidence_bundle

    root = sample_wiki(tmp_path)
    workdir = tmp_path / "tex-first-arxiv"

    def fake_fetch_text(url: str, timeout: int) -> dict:
        if "export.arxiv.org" in url:
            return {"ok": True, "status": 200, "text": "<feed><entry><title>TeX First Fixture Paper</title></entry></feed>"}
        return {"ok": True, "status": 200, "text": "<html><title>TeX First Fixture Paper</title></html>"}

    def fake_fetch_url_to_file(url: str, dest: Path, timeout: int) -> dict:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake eprint tarball")
        return {"ok": True, "status": 200, "dest": str(dest), "bytes": dest.stat().st_size}

    def fake_extract_tar(tar_path: Path, dest: Path) -> dict:
        _write_tex_first_source_fixture(workdir)
        return {"ok": True, "extracted_count": 2, "errors": []}

    def forbidden_process_pdf(*args, **kwargs):
        raise AssertionError("arXiv TeX-success route must not call process_pdf or parse PDF text")

    monkeypatch.setattr(raw_fast_evidence_bundle, "fetch_text", fake_fetch_text)
    monkeypatch.setattr(raw_fast_evidence_bundle, "fetch_url_to_file", fake_fetch_url_to_file)
    monkeypatch.setattr(raw_fast_evidence_bundle, "safe_extract_tar", fake_extract_tar)
    monkeypatch.setattr(raw_fast_evidence_bundle, "process_pdf", forbidden_process_pdf)

    payload = raw_fast_evidence_bundle.process_arxiv(
        "https://arxiv.org/abs/2600.00001",
        root,
        workdir,
        "docling",
        False,
        ["none"],
        5,
        raw_fast_evidence_bundle.TimingRecorder(),
        paper_digest=True,
        resource_draft=True,
        resource_health="none",
    )

    assert payload["ok"] is True
    assert payload["pdf_backend_effective"] == "tex_source"
    assert LEGACY_EXTRACTOR_KEY not in payload
    assert LEGACY_RAW_TEXT_KEY not in payload["files"]
    assert LEGACY_LAYOUT_TEXT_KEY not in payload["files"]
    assert not (workdir / LEGACY_RAW_TEXT_FILE).exists()
    assert not (workdir / LEGACY_LAYOUT_TEXT_FILE).exists()
    read_plan = json.loads((workdir / "tex_read_plan.json").read_text(encoding="utf-8"))
    assert read_plan["source_kind"] == "tex_source"
    assert read_plan["fallback_used"] is False
    assert read_plan["main_tex"] == "source/main.tex"
    assert read_plan["first_reads"]
    handoff = json.loads((workdir / "agent_handoff.json").read_text(encoding="utf-8"))
    assert handoff["source_read_plan"]["source_kind"] == "tex_source"
    assert handoff["manual_reference_policy"]["mode"] == "only_on_manual_required"
    assert handoff["manual_reference_policy"]["manual_references_visible"] is False
    assert handoff["automation_next_action"]["action"] == "follow_source_read_plan"
    assert handoff["manual_reference_paths"] == []
    brief = json.loads((workdir / "agent_brief.json").read_text(encoding="utf-8"))
    assert brief["evidence_cards"][0]["kind"] == "abstract"
    assert brief["evidence_cards"][0]["source_bucket"] == "abstract_card"
    handoff_md = (workdir / "agent_handoff.md").read_text(encoding="utf-8")
    assert "Default next action: read the TeX spans" in handoff_md
    assert "Manual references are hidden unless this handoff reports manual_required=true" in handoff_md
    assert LEGACY_RAW_TEXT_FILE not in handoff_md
    assert LEGACY_LAYOUT_TEXT_FILE not in handoff_md


def test_raw_fast_evidence_bundle_direct_pdf_uses_docling_without_legacy_pdf_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ops import raw_fast_evidence_bundle

    root = sample_wiki(tmp_path)
    workdir = tmp_path / "docling-direct-pdf"

    def fake_fetch_url_to_file(url: str, dest: Path, timeout: int) -> dict:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"%PDF fake fixture")
        return {"ok": True, "status": 200, "dest": str(dest), "bytes": dest.stat().st_size}

    def fake_docling(pdf: Path, workdir_arg: Path, strict: bool = False, timeout: int = 120) -> dict:
        _write_fake_docling_outputs(workdir_arg)
        return {"ok": True, "markdown": str(workdir_arg / "docling.md"), "json": str(workdir_arg / "docling.json"), "markdown_chars": (workdir_arg / "docling.md").stat().st_size}

    assert not hasattr(raw_fast_evidence_bundle, "run_" + LEGACY_EXTRACTOR_KEY)
    monkeypatch.setattr(raw_fast_evidence_bundle, "fetch_url_to_file", fake_fetch_url_to_file)
    monkeypatch.setattr(raw_fast_evidence_bundle, "run_pdfinfo", lambda pdf, out: {"ok": True, "returncode": 0})
    monkeypatch.setattr(raw_fast_evidence_bundle, "extract_pdf_links", lambda pdf, out: {"ok": True, "links": []})
    monkeypatch.setattr(raw_fast_evidence_bundle, "run_docling", fake_docling)

    payload = raw_fast_evidence_bundle.process_pdf(
        "https://example.test/docling-only.pdf",
        "direct-pdf",
        root,
        workdir,
        "docling",
        False,
        ["none"],
        5,
        raw_fast_evidence_bundle.TimingRecorder(),
        paper_digest=True,
        resource_draft=True,
        resource_health="none",
    )

    assert payload["ok"] is True
    assert payload["pdf_backend_effective"] == "docling"
    assert LEGACY_EXTRACTOR_KEY not in payload
    assert LEGACY_RAW_TEXT_KEY not in payload["files"]
    assert LEGACY_LAYOUT_TEXT_KEY not in payload["files"]
    assert (workdir / "docling.md").exists()
    assert (workdir / "docling.json").exists()
    assert not (workdir / LEGACY_RAW_TEXT_FILE).exists()
    assert not (workdir / LEGACY_LAYOUT_TEXT_FILE).exists()
    handoff = json.loads((workdir / "agent_handoff.json").read_text(encoding="utf-8"))
    assert handoff["source_read_plan"]["source_kind"] == "docling_pdf"


def test_raw_fast_evidence_bundle_digest_ignores_stale_legacy_pdf_text_files(tmp_path: Path) -> None:
    from ops import raw_fast_evidence_bundle

    workdir = tmp_path / "digest-source-priority"
    _write_fake_docling_outputs(workdir)
    write(workdir / LEGACY_LAYOUT_TEXT_FILE, f"Stale {LEGACY_EXTRACTOR_KEY} layout must not become a source priority.")
    write(workdir / LEGACY_RAW_TEXT_FILE, f"Stale {LEGACY_EXTRACTOR_KEY} raw text must not become a source priority.")

    digest = raw_fast_evidence_bundle.build_paper_digest(
        workdir,
        "Docling Fallback Paper",
        "https://example.test/docling-only.pdf",
        files={"docling_markdown": "docling.md"},
    )

    assert digest["source_priority"] == ["docling"]
    assert LEGACY_EXTRACTOR_KEY not in digest["source_priority"]
    assert LEGACY_LAYOUT_TEXT_FILE not in digest["files_used"]
    assert LEGACY_RAW_TEXT_FILE not in digest["files_used"]
@pytest.mark.requires_fitz
def test_raw_fast_evidence_bundle_direct_pdf_writes_temp_only_and_defaults_docling(tmp_path: Path) -> None:
    pytest.importorskip("docling.document_converter")
    root = sample_wiki(tmp_path)
    source_pdf = tmp_path / "source.pdf"
    _write_tiny_pdf(source_pdf)
    workdir = tmp_path / "bundle"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ops.raw_fast_evidence_bundle",
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
def test_raw_fast_evidence_bundle_paper_digest_resource_draft_and_local_figures_are_sidecars(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    source_pdf = tmp_path / "source.pdf"
    _write_tiny_pdf(source_pdf)
    workdir = tmp_path / "bundle-sidecars"
    _write_sidecar_source_fixture(workdir)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ops.raw_fast_evidence_bundle",
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
            "--paper-digest",
            "--resource-draft",
            "--localize-figures",
            "--image-slug",
            "fixture-sidecar-paper",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["ok"] is True
    for key in [
        "paper_digest",
        "paper_digest_markdown",
        "resource_boundary_draft",
        "resource_boundary_draft_markdown",
        "note_block_drafts",
        "localized_figures",
        "localized_figures_markdown",
        "raw_fast_preflight",
        "agent_brief",
        "agent_brief_markdown",
        "evidence_report",
        "evidence_report_markdown",
        "note_candidate",
        "note_candidate_markdown",
        "agent_handoff",
        "agent_handoff_markdown",
    ]:
        assert key in payload["files"]
        assert (workdir / payload["files"][key]).exists()
    assert payload["preflight"]["ok"] is True
    assert payload["agent_automation"]["ok"] is True

    digest = json.loads((workdir / "paper_digest.json").read_text(encoding="utf-8"))
    assert digest["ok"] is True
    resource_draft = json.loads((workdir / "resource_boundary_draft.json").read_text(encoding="utf-8"))
    assert resource_draft["github"][0]["url"] == "https://github.com/example/sidecar-paper"
    assert resource_draft["github"][0]["status"] == "not_checked"
    assert resource_draft["github"][0]["source"] == "source_exposed_exact_url"
    evidence_report = json.loads((workdir / "evidence_report.json").read_text(encoding="utf-8"))
    assert evidence_report["resource_boundary"]["github_count"] >= 1
    assert evidence_report["resource_boundary"]["review_required"] is False
    handoff = json.loads((workdir / "agent_handoff.json").read_text(encoding="utf-8"))
    assert handoff["resource_review_required"] is False
    assert handoff["manual_reference_paths"] == []
    assert "https://github.com/example/sidecar-paper" in handoff["resource_status_summary"]
    localized = json.loads((workdir / "localized_figures.json").read_text(encoding="utf-8"))
    assert localized["ok"] is True
    localized_entry = localized["entries"][0]
    assert localized_entry["dest_rel"].startswith("localized_figures_assets/fixture-sidecar-paper/")
    assert localized_entry["raw_note_policy"] == "temporary_inspection_only_do_not_embed_markdown_image"
    assert "markdown" not in localized_entry
    assert (workdir / localized_entry["dest_rel"]).exists()
    assert not (root / "raw" / "images").exists()
    assert not (root / "paper_digest.json").exists()
    assert not (root / "resource_boundary_draft.json").exists()
    assert not (root / "note_block_drafts.md").exists()
    assert wiki_root_machine_pollution(root) == []
def test_raw_fast_evidence_bundle_localize_figures_does_not_overwrite_existing_asset(tmp_path: Path) -> None:
    from ops import raw_fast_evidence_bundle

    root = sample_wiki(tmp_path)
    workdir = tmp_path / "bundle-sidecar-conflict"
    _write_sidecar_source_fixture(workdir)
    conflict = workdir / "localized_figures_assets" / "fixture-sidecar-paper" / "figure-01-method-figure-shows-the-pipeline.png"
    conflict.parent.mkdir(parents=True, exist_ok=True)
    conflict.write_bytes(b"do-not-overwrite")

    localized = raw_fast_evidence_bundle.localize_source_figures(root, workdir, "fixture-sidecar-paper")

    assert localized["ok"] is True
    assert conflict.read_bytes() == b"do-not-overwrite"
    assert localized["entries"]
    assert localized["entries"][0]["dest_rel"] != conflict.relative_to(workdir).as_posix()
    assert localized["entries"][0]["dest_rel"].endswith("-02.png")
    assert (workdir / localized["entries"][0]["dest_rel"]).exists()
    assert not (root / "raw" / "images").exists()
def test_raw_fast_evidence_bundle_localize_figures_resolves_source_root_relative_paths(tmp_path: Path) -> None:
    from ops import raw_fast_evidence_bundle

    root = sample_wiki(tmp_path)
    workdir = tmp_path / "bundle-source-root-fig"
    _write_tiny_png(workdir / "source" / "fig" / "method.png")
    write(
        workdir / "source" / "tex" / "main.tex",
        r"""
\begin{figure}
\includegraphics{fig/method.png}
\caption{Source root method figure}
\label{fig:source-root}
\end{figure}
""".strip(),
    )

    localized = raw_fast_evidence_bundle.localize_source_figures(root, workdir, "source-root-paper")

    assert localized["ok"] is True
    assert len(localized["entries"]) == 1
    assert localized["entries"][0]["label"] == "fig:source-root"
    assert localized["entries"][0]["source_rel"] == "fig/method.png"
    assert localized["refused"] == []
    assert localized["entries"][0]["dest_rel"].startswith("localized_figures_assets/source-root-paper/")
    assert (workdir / localized["entries"][0]["dest_rel"]).exists()
    assert not (root / "raw" / "images").exists()
@pytest.mark.requires_fitz
def test_raw_fast_evidence_bundle_localize_figures_renders_pdf_source_figures_to_png(tmp_path: Path) -> None:
    from ops import raw_fast_evidence_bundle

    root = sample_wiki(tmp_path)
    workdir = tmp_path / "bundle-pdf-fig"
    _write_tiny_pdf_figure(workdir / "source" / "fig" / "frontier.pdf")
    write(
        workdir / "source" / "tex" / "main.tex",
        r"""
\begin{figure}
\includegraphics{fig/frontier.pdf}
\caption{Frontier curve as PDF}
\label{fig:frontier-pdf}
\end{figure}
""".strip(),
    )

    localized = raw_fast_evidence_bundle.localize_source_figures(root, workdir, "pdf-figure-paper")

    assert localized["ok"] is True
    assert len(localized["entries"]) == 1
    entry = localized["entries"][0]
    assert entry["label"] == "fig:frontier-pdf"
    assert entry["source_rel"] == "fig/frontier.pdf"
    assert entry["dest_rel"].endswith(".png")
    assert entry["localization_method"] == "pdf_render_first_page"
    assert entry["source_sha256"]
    assert entry["sha256"] != entry["source_sha256"]
    assert entry["width"] > 0 and entry["height"] > 0
    assert entry["dest_rel"].startswith("localized_figures_assets/pdf-figure-paper/")
    assert (workdir / entry["dest_rel"]).exists()
    assert not (root / "raw" / "images").exists()
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
def test_raw_fast_evidence_bundle_resource_boundary_defaults_to_not_checked_without_exact_link_report() -> None:
    from ops import raw_fast_evidence_bundle

    resource_probe = {
        "ok": True,
        "probes": [
            {"ok": True, "type": "doi", "doi": "10.1234/example.paper", "status": "detected", "url": "https://doi.org/10.1234/example.paper"},
            {"ok": True, "type": "arxiv", "id": "2604.08999", "status": "detected", "url": "https://arxiv.org/abs/2604.08999"},
            {"ok": True, "type": "github_repo", "repo": "example/ignored"},
            {"ok": True, "type": "hf_models", "query": "Fixture", "count": 0, "items": []},
            {"ok": True, "type": "project_page", "url": "https://example.test/project"},
        ],
    }

    draft = raw_fast_evidence_bundle.summarize_resource_boundary(resource_probe, metadata={"title": "Fixture Sidecar Paper"})
    markdown = raw_fast_evidence_bundle.render_resource_boundary_markdown(draft)

    assert draft["github"] == []
    assert draft["project_pages"] == []
    assert {kind: bucket["status"] for kind, bucket in draft["hf"].items()} == {"models": "not_checked", "datasets": "not_checked", "spaces": "not_checked"}
    assert draft["doi"][0]["doi"] == "10.1234/example.paper"
    assert draft["arxiv"][0]["id"] == "2604.08999"
    assert "verified_absent" not in markdown
    assert "candidates_unverified" not in markdown
    assert "not_checked" in markdown
def test_raw_fast_evidence_bundle_localize_figures_refuses_unsafe_sources(tmp_path: Path) -> None:
    from ops import raw_fast_evidence_bundle

    root = sample_wiki(tmp_path)
    workdir = tmp_path / "bundle-localize"
    _write_tiny_png(workdir / "source" / "figs" / "method.png")
    write(workdir / "source" / "figs" / "secret.txt", "not an image")
    write(
        workdir / "source" / "main.tex",
        r"""
\begin{figure}\includegraphics{figs/method.png}\caption{Safe figure}\label{fig:safe}\end{figure}
\begin{figure}\includegraphics{../outside.png}\caption{Traversal figure}\label{fig:outside}\end{figure}
\begin{figure}\includegraphics{figs/secret.txt}\caption{Text file}\label{fig:text}\end{figure}
""".strip(),
    )

    localized = raw_fast_evidence_bundle.localize_source_figures(root, workdir, "safe-slug")

    assert localized["ok"] is True
    assert len(localized["entries"]) == 1
    assert localized["entries"][0]["label"] == "fig:safe"
    refusal_reasons = {item["reason"] for item in localized["refused"]}
    assert "path_traversal" in refusal_reasons
    assert "unsupported_extension" in refusal_reasons
    assert localized["entries"][0]["dest_rel"].startswith("localized_figures_assets/safe-slug/")
    assert (workdir / localized["entries"][0]["dest_rel"]).exists()
    assert not (root / "raw" / "images").exists()
def test_raw_fast_evidence_bundle_candidate_frontmatter_stays_compact() -> None:
    from ops import raw_fast_evidence_bundle

    fm = raw_fast_evidence_bundle.build_frontmatter(
        "Compact Candidate Paper",
        "https://arxiv.org/abs/2601.0101",
        "arxiv",
    )

    assert set(fm) <= {
        "title",
        "source",
        "created",
        "updated",
        "type",
        "domain",
        "tags",
        "topic_hints",
        "capture_route",
        "captured",
    }
    assert "resource_status" not in fm
    assert "source_id" not in fm

    skeleton = raw_fast_evidence_bundle.build_note_skeleton(fm)
    assert "## 资源与复现状态" not in skeleton
    assert "## Evidence trail" not in skeleton
@pytest.mark.requires_fitz
def test_raw_fast_evidence_bundle_refuses_workdir_inside_wiki_root(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    source_pdf = tmp_path / "source.pdf"
    _write_tiny_pdf(source_pdf)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ops.raw_fast_evidence_bundle",
            "--url",
            source_pdf.as_uri(),
            "--kind",
            "direct-pdf",
            "--root",
            str(root),
            "--workdir",
            str(root / "raw" / "clip" / "bundle"),
            "--probe",
            "none",
        ],
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode != 0
    assert payload["stage"] == "preflight"
    assert payload["error"] == "workdir_inside_wiki_root"
    assert not (root / "raw" / "clip" / "bundle" / "evidence_bundle.json").exists()
def test_raw_fast_evidence_bundle_resource_probe_contracts_are_explicit() -> None:
    from ops import raw_fast_evidence_bundle

    detected = raw_fast_evidence_bundle.build_resource_probe(
        "See arXiv:2604.08999 and DOI 10.1234/example.paper for details.",
        {"links": [{"uri": "https://github.com/example/sidecar"}]},
        ["arxiv", "doi"],
    )
    probes = {(item["type"], item.get("id") or item.get("doi")): item for item in detected["probes"]}

    assert ("arxiv", "2604.08999") in probes
    assert probes[("arxiv", "2604.08999")]["ok"] is True
    assert probes[("arxiv", "2604.08999")]["status"] == "detected"
    assert probes[("arxiv", "2604.08999")]["evidence"]
    assert ("doi", "10.1234/example.paper") in probes
    assert probes[("doi", "10.1234/example.paper")]["status"] == "detected"
    assert "https://github.com/example/sidecar" in detected["urls"]

    skipped = raw_fast_evidence_bundle.build_resource_probe(
        "Project page: https://example.test/project and arXiv:2604.08999.",
        {"links": [{"uri": "https://huggingface.co/example/model"}]},
        ["none"],
    )

    assert skipped["ok"] is True
    assert skipped["skipped"] is True
    assert skipped["probes"] == []
    assert skipped["urls"] == ["https://example.test/project", "https://huggingface.co/example/model"]


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
    assert preflight["duplicate_hits"]["raw"]
    assert preflight["queue_status"]["wiki_pending_count"] == 1
    assert summary["ok"] is True
    for key in ["raw_fast_preflight", "agent_brief", "agent_brief_markdown", "evidence_report", "evidence_report_markdown", "note_candidate", "note_candidate_markdown", "agent_handoff", "agent_handoff_markdown"]:
        assert key in files
        assert (workdir / files[key]).exists()
    brief = json.loads((workdir / "agent_brief.json").read_text(encoding="utf-8"))
    assert brief["protected_anchors"]["next_raw_path"] == preflight["next_raw_path"]
    assert brief["duplicate_summary"]["raw"] >= 1
    assert brief["source_refs"]["paper_digest"] == "paper_digest.json"
    evidence_report = json.loads((workdir / "evidence_report.json").read_text(encoding="utf-8"))
    assert evidence_report["resource_boundary"]["status"] == "standardized"
    candidate_md = (workdir / "note_candidate.md").read_text(encoding="utf-8")
    body = candidate_md.split("---", 2)[-1]
    assert "https://" not in body
    assert "## 资源与复现状态" not in body
    assert "## Evidence trail" not in body
    assert "<!-- advisory:" in body
    assert not (root / "agent_brief.json").exists()
    assert not (root / "evidence_report.json").exists()
    assert not (root / "note_candidate.md").exists()
    assert wiki_root_machine_pollution(root) == []
