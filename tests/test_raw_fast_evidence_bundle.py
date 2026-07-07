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
This paper introduces Sidecar Method with exact evidence and official code at \url{https://github.com/example/sidecar-paper}.
\end{abstract}
\section{Method}
We optimize $\mathcal{L}=x+y$.
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


def _write_tex_agent_ir_source_fixture(workdir: Path) -> None:
    _write_tiny_png(workdir / "source" / "figures" / "cost_curve.png")
    write(
        workdir / "source" / "main.tex",
        r"""
\title{TeX Agent IR Fixture}
\newcommand{\system}{AgentIR}
\begin{document}
\begin{abstract}
This paper tests semantic TeX sidecars for source-grounded raw-fast reading.
\end{abstract}
\input{sections/method}
\appendix
\input{sections/appendix}
\end{document}
""".strip(),
    )
    write(
        workdir / "source" / "sections" / "method.tex",
        r"""
\section{Method}
\system{} minimizes a compact loss while preserving source anchors.
\begin{equation}
\mathcal{L}=x+y
\label{eq:agent-ir-loss}
\end{equation}
\begin{figure}
\includegraphics[width=0.8\linewidth]{figures/cost_curve}
\caption{Cost curve plot shows the main result trend.}
\label{fig:cost-curve}
\end{figure}
\begin{table}
\caption{Main result table for the semantic sidecar fixture.}
\label{tab:agent-ir-main}
\begin{tabular}{lr}
Method & Score \\
AgentIR & 42
\end{tabular}
\end{table}
""".strip(),
    )
    write(
        workdir / "source" / "sections" / "appendix.tex",
        r"""
\section{Appendix Details}
Supplementary detail should stay discoverable but not become the default body-only core.
""".strip(),
    )


def _fake_latexpand_success(monkeypatch: pytest.MonkeyPatch, raw_fast_evidence_bundle, workdir: Path) -> None:
    original_which = raw_fast_evidence_bundle.shutil.which
    original_subprocess_run = raw_fast_evidence_bundle.subprocess.run

    def fake_which(name: str) -> str | None:
        if name == "latexpand":
            return "/usr/bin/latexpand"
        return original_which(name)

    def fake_subprocess_run(command, *, cwd=None, capture_output=False, text=False, timeout=None):
        if Path(command[0]).name == "latexpand":
            assert command[-1] == "main.tex"
            assert cwd == str((workdir / "source").resolve())
            assert capture_output is True
            assert text is True
            flattened = "\n".join(
                (workdir / rel).read_text(encoding="utf-8")
                for rel in ["source/main.tex", "source/sections/method.tex", "source/sections/appendix.tex"]
            )
            return raw_fast_evidence_bundle.subprocess.CompletedProcess(command, 0, flattened, "")
        return original_subprocess_run(command, cwd=cwd, capture_output=capture_output, text=text, timeout=timeout)

    monkeypatch.setattr(raw_fast_evidence_bundle.shutil, "which", fake_which)
    monkeypatch.setattr(raw_fast_evidence_bundle.subprocess, "run", fake_subprocess_run)


def _write_sectioned_main_with_long_appendix_fixture(workdir: Path) -> None:
    write(
        workdir / "source" / "sec" / "vla_main.tex",
        r"""
\section{Introduction}
Main paper introduction states the core problem and deployment tension.
\section{Method}
Main paper method defines the action-horizon correction objective.
\begin{equation}
J(\theta)=\mathbb{E}[r(a_{1:H})]
\label{eq:main-objective}
\end{equation}
\section{Experiments}
Main paper experiments report MetaWorld and LIBERO task results.
\section{Conclusion}
Main paper conclusion names the practical boundary.
""".strip(),
    )
    appendix_sections = "\n".join(
        f"\\section{{Appendix Ablation {idx}}}\nSupplementary implementation and ablation detail {idx}. " * 8
        for idx in range(1, 18)
    )
    write(workdir / "source" / "sec" / "vla_appendix.tex", appendix_sections)


def _write_fake_docling_outputs(workdir: Path, markdown: str = "# Docling Fallback Paper\n\nAbstract\nDocling text is the PDF fallback.\n\n## Method\nFallback method text.") -> None:
    write(workdir / "docling.md", markdown)
    write(workdir / "docling.json", json.dumps({"ok": True, "markdown_chars": len(markdown)}) + "\n")


def test_raw_fast_evidence_bundle_tex_read_plan_prefers_main_sections_over_long_appendix(tmp_path: Path) -> None:
    from ops import raw_fast_evidence_bundle

    _write_sectioned_main_with_long_appendix_fixture(tmp_path)

    selection = raw_fast_evidence_bundle.select_main_tex_source(tmp_path)
    assert selection["ok"] is True
    assert selection["main_tex"] == "source/sec/vla_main.tex"

    plan = raw_fast_evidence_bundle.build_source_read_plan(
        tmp_path,
        source_kind="tex_source",
        main_rel=selection["main_tex"],
        fallback_used=False,
    )
    first_reads = plan["first_reads"]
    assert first_reads
    assert {item["path"] for item in first_reads} == {"source/sec/vla_main.tex"}
    assert first_reads[0]["limit"] == 180
    assert any(item["limit"] == 220 for item in first_reads[1:])
    assert "fallback locators" in plan["reading_strategy"]
    assert "not a read queue" in plan["reading_strategy"]
    assert "continue reading original source" not in plan["reading_strategy"]


def test_raw_fast_evidence_bundle_title_from_text_skips_docling_image_placeholder() -> None:
    from ops import raw_fast_evidence_bundle

    text = "<!-- image -->\n\n## Next-Generation Agentic Reinforcement Learning Systems Enable Self-Evolving Agents\n\n## Abstract\nBody"

    assert raw_fast_evidence_bundle.title_from_text(text) == "Next-Generation Agentic Reinforcement Learning Systems Enable Self-Evolving Agents"


def test_raw_fast_evidence_bundle_pdf_title_falls_back_to_url_stem_when_docling_only_has_placeholders() -> None:
    from ops import raw_fast_evidence_bundle

    title = raw_fast_evidence_bundle.pdf_title_from_text_or_url("<!-- image -->\n\n<!-- formula-not-decoded -->", "https://raw.githubusercontent.com/areal-project/AReaL/main/docs/paper/AReaL2.0_report.pdf")

    assert title == "AReaL2.0_report"


def test_raw_fast_evidence_bundle_tex_title_skips_macro_definition_pollution() -> None:
    from ops import raw_fast_evidence_bundle

    tex = r"""
\newcommand{\makefntext}[1]{\title{makefntext 1 xt 1em makefnmark hbox}}
\renewcommand{\title}[1]{#1}
\title[Short Fixture]{Real Semantic Handoff Paper}
\begin{document}
\maketitle
\end{document}
"""

    assert raw_fast_evidence_bundle.tex_title_from_text(tex) == "Real Semantic Handoff Paper"


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


def test_raw_fast_evidence_bundle_pdf_download_limit_is_large_and_configurable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ops import raw_fast_evidence_bundle

    assert raw_fast_evidence_bundle.DEFAULT_MAX_DOWNLOAD_BYTES == 2 * 1024 * 1024 * 1024
    assert raw_fast_evidence_bundle.parse_byte_limit("512MiB") == 512 * 1024 * 1024
    assert raw_fast_evidence_bundle.parse_byte_limit("1.5GiB") == int(1.5 * 1024**3)
    assert raw_fast_evidence_bundle.parse_byte_limit("none") is None
    monkeypatch.setenv("LLM_WIKI_RAW_FAST_MAX_DOWNLOAD_BYTES", "3GiB")
    assert raw_fast_evidence_bundle.configured_max_download_bytes() == 3 * 1024**3
    monkeypatch.setenv("LLM_WIKI_RAW_FAST_MAX_DOWNLOAD_BYTES", "0")
    assert raw_fast_evidence_bundle.configured_max_download_bytes() is None

    source = tmp_path / "large-ok.pdf"
    source.write_bytes(b"%PDF-" + b"x" * 16)
    too_small = raw_fast_evidence_bundle.fetch_url_to_file(source.as_uri(), tmp_path / "too-small.pdf", timeout=1, max_bytes=8)
    assert too_small["ok"] is False
    assert too_small["error"] == "FileTooLarge"
    unlimited = raw_fast_evidence_bundle.fetch_url_to_file(source.as_uri(), tmp_path / "unlimited.pdf", timeout=1, max_bytes=None)
    assert unlimited["ok"] is True
    assert unlimited["bytes"] == source.stat().st_size


@pytest.mark.parametrize(
    "url",
    [
        "https://openreview.net/forum?id=mLhZzo7BIb",
        "https://openreview.net/pdf?id=mLhZzo7BIb",
        "https://openreview.net/attachment?id=mLhZzo7BIb&name=pdf",
    ],
)
def test_raw_fast_evidence_bundle_recognizes_openreview_routes(url: str) -> None:
    from ops import raw_fast_evidence_bundle

    assert raw_fast_evidence_bundle.openreview_id_from_url(url) == "mLhZzo7BIb"
    assert raw_fast_evidence_bundle.detect_kind(url, "auto") == "openreview"
    assert raw_fast_evidence_bundle.canonical_openreview_pdf_url("mLhZzo7BIb") == "https://openreview.net/pdf?id=mLhZzo7BIb"


def test_raw_fast_evidence_bundle_openreview_uses_authenticated_pdf_and_canonical_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ops import raw_fast_evidence_bundle

    root = sample_wiki(tmp_path)
    workdir = tmp_path / "openreview"
    supplied_url = "https://openreview.net/attachment?id=mLhZzo7BIb&name=pdf"
    canonical_pdf = "https://openreview.net/pdf?id=mLhZzo7BIb"

    def fail_anonymous_fetch(url: str, dest: Path, timeout: int, max_bytes: int = raw_fast_evidence_bundle.DEFAULT_MAX_DOWNLOAD_BYTES) -> dict:
        return {"ok": False, "url": url, "dest": str(dest), "error": "HTTPError", "message": "403 Forbidden"}

    def fake_openreview_fetch(note_id: str, dest: Path, timeout: int, max_bytes: int = raw_fast_evidence_bundle.DEFAULT_MAX_DOWNLOAD_BYTES) -> dict:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"%PDF-1.4\n% fixture openreview pdf\n")
        return {
            "ok": True,
            "url": canonical_pdf,
            "status": 200,
            "content_type": "application/pdf",
            "bytes": dest.stat().st_size,
            "sha256": raw_fast_evidence_bundle.sha256_file(dest),
            "dest": str(dest),
            "openreview": {
                "id": note_id,
                "forum_url": "https://openreview.net/forum?id=mLhZzo7BIb",
                "pdf_url": canonical_pdf,
                "metadata_ok": True,
                "auth_used": True,
                "title": "OpenReview Fixture Paper",
            },
        }

    def fake_docling(pdf_path: Path, workdir_arg: Path, strict: bool, timeout: int) -> dict:
        _write_fake_docling_outputs(
            workdir_arg,
            "# OpenReview Fixture Paper\n\nAbstract\nThis fixture confirms authenticated OpenReview PDF retrieval.\n\n## Method\nThe method is a test fixture.",
        )
        return {"ok": True, "returncode": 0, "markdown_chars": (workdir_arg / "docling.md").stat().st_size}

    monkeypatch.setattr(raw_fast_evidence_bundle, "fetch_url_to_file", fail_anonymous_fetch)
    monkeypatch.setattr(raw_fast_evidence_bundle, "fetch_openreview_pdf_to_file", fake_openreview_fetch)
    monkeypatch.setattr(raw_fast_evidence_bundle, "run_pdfinfo", lambda pdf, out: {"ok": True, "stdout": "Pages: 1"})
    monkeypatch.setattr(raw_fast_evidence_bundle, "extract_pdf_links", lambda pdf, out: {"ok": True, "links": []})
    monkeypatch.setattr(raw_fast_evidence_bundle, "run_docling", fake_docling)

    payload = raw_fast_evidence_bundle.process_openreview(
        supplied_url,
        root,
        workdir,
        "docling",
        False,
        ["none"],
        30,
        paper_digest=True,
        resource_draft=True,
        resource_health="none",
    )

    assert payload["ok"] is True
    assert payload["kind"] == "openreview"
    assert payload["source_url"] == canonical_pdf
    assert payload["supplied_url"] == supplied_url
    assert payload["fetch"]["openreview"]["auth_used"] is True
    assert payload["openreview"]["id"] == "mLhZzo7BIb"
    assert payload["files"]["openreview_metadata"] == "openreview_metadata.json"
    frontmatter = json.loads((workdir / "candidate_frontmatter.json").read_text(encoding="utf-8"))
    assert frontmatter["source"] == canonical_pdf
    assert frontmatter["title"] == "OpenReview Fixture Paper"
    assert frontmatter["domain"] == "machine-learning"
    assert frontmatter["capture_route"] == "raw-fast evidence bundle (openreview)"
    assert "source_pdf" not in frontmatter
    assert "openreview_id" not in frontmatter


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
    assert frontmatter["domain"] == "machine-learning"
    assert supplied_url not in frontmatter.values()


def test_raw_fast_evidence_bundle_resource_classifier_separates_hf_paper_index_collection_from_artifacts() -> None:
    from ops import raw_fast_evidence_bundle

    assert raw_fast_evidence_bundle._resource_url_candidate("https://huggingface.co/papers/2606.32039") is None
    assert raw_fast_evidence_bundle._resource_url_candidate("https://huggingface.co/models?other=arxiv:2606.32039") is None

    model = raw_fast_evidence_bundle._resource_url_candidate("https://huggingface.co/BinLin203/GEAR-VQ")
    dataset = raw_fast_evidence_bundle._resource_url_candidate("https://huggingface.co/datasets/BinLin203/GEAR-Data")
    collection = raw_fast_evidence_bundle._resource_url_candidate("https://huggingface.co/collections/BinLin203/gear-models")

    assert model is not None
    assert model["type"] == "hf"
    assert model["hf_kind"] == "models"
    assert model["repo"] == "BinLin203/GEAR-VQ"
    assert dataset is not None
    assert dataset["type"] == "hf"
    assert dataset["hf_kind"] == "datasets"
    assert dataset["repo"] == "BinLin203/GEAR-Data"
    assert collection is not None
    assert collection["type"] == "hf_collection"
    assert collection["hf_kind"] == "collections"
    assert collection["repo"] == "BinLin203/gear-models"


def test_raw_fast_evidence_bundle_pwc_supplied_resources_feed_metadata_without_hf_paper_confusion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ops import raw_fast_evidence_bundle

    root = sample_wiki(tmp_path)
    workdir = tmp_path / "pwc-supplied-resource-metadata"
    supplied_url = "https://paperswithcode.co/paper/2606.32039"

    pwc_payload = {
        "arxiv_id": "2606.32039",
        "url_abs": "https://arxiv.org/abs/2606.32039",
        "url_pdf": "https://arxiv.org/pdf/2606.32039.pdf",
        "repositories": [
            {"url": "https://github.com/Tencent-Hunyuan/GEAR", "is_official": True, "source": "hf_api"},
        ],
        "project_pages": [
            {"url": "https://linb203.github.io/gear", "is_official": True, "source": "hf_api"},
        ],
        "hf_artifact_summary": {
            "best_kind": "model",
            "best_count": 2,
            "best_url": "https://huggingface.co/models?other=arxiv:2606.32039",
        },
        "urls_extracted": True,
    }
    hf_paper_payload = {
        "id": "2606.32039",
        "githubRepo": "https://github.com/Tencent-Hunyuan/GEAR",
        "projectPage": "https://linb203.github.io/gear",
        "linkedModels": [
            {"id": "BinLin203/Warmup-LFQ", "repoType": "model"},
            {"id": "BinLin203/GEAR-VQ", "repoType": "model"},
        ],
        "linkedDatasets": [
            {"id": "BinLin203/GEAR-Data", "repoType": "dataset"},
        ],
        "linkedSpaces": [
            {"id": "BinLin203/GEAR-Demo", "repoType": "space"},
        ],
    }
    hf_repos_payload = {
        "models": [
            {"id": "BinLin203/Warmup-LFQ"},
            {"id": "BinLin203/GEAR-VQ"},
        ],
        "datasets": [
            {"id": "BinLin203/GEAR-Data"},
        ],
        "spaces": [
            {"id": "BinLin203/GEAR-Demo"},
        ],
    }

    def fake_fetch_text(url: str, timeout: int) -> dict:
        if "export.arxiv.org" in url:
            return {"ok": True, "status": 200, "text": "<feed><entry><title>GEAR Fixture Paper</title></entry></feed>"}
        if url == "https://arxiv.org/abs/2606.32039":
            return {"ok": True, "status": 200, "text": "<html><title>GEAR Fixture Paper</title></html>"}
        if "paperswithcode.co/api/v1/papers/2606.32039" in url:
            return {"ok": True, "status": 200, "text": json.dumps(pwc_payload)}
        if url == "https://huggingface.co/api/papers/2606.32039":
            return {"ok": True, "status": 200, "text": json.dumps(hf_paper_payload)}
        if url == "https://huggingface.co/api/arxiv/2606.32039/repos":
            return {"ok": True, "status": 200, "text": json.dumps(hf_repos_payload)}
        raise AssertionError(f"unexpected fetch_text URL: {url}")

    def fake_fetch_url_to_file(url: str, dest: Path, timeout: int) -> dict:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake eprint tarball")
        return {"ok": True, "status": 200, "dest": str(dest), "bytes": dest.stat().st_size}

    def fake_extract_tar(tar_path: Path, dest: Path) -> dict:
        _write_tex_first_source_fixture(workdir)
        return {"ok": True, "extracted_count": 2, "errors": []}

    monkeypatch.setattr(raw_fast_evidence_bundle, "fetch_text", fake_fetch_text)
    monkeypatch.setattr(raw_fast_evidence_bundle, "fetch_url_to_file", fake_fetch_url_to_file)
    monkeypatch.setattr(raw_fast_evidence_bundle, "safe_extract_tar", fake_extract_tar)
    monkeypatch.setattr(
        raw_fast_evidence_bundle,
        "probe_exact_link_health",
        lambda url, **kwargs: {"ok": True, "url": url, "status": "verified_present"},
    )

    payload = raw_fast_evidence_bundle.process_arxiv(
        supplied_url,
        root,
        workdir,
        "docling",
        False,
        ["arxiv", "doi"],
        5,
        raw_fast_evidence_bundle.TimingRecorder(),
        paper_digest=True,
        resource_draft=True,
        resource_health="direct",
    )

    assert payload["ok"] is True
    frontmatter = json.loads((workdir / "candidate_frontmatter.json").read_text(encoding="utf-8"))
    assert frontmatter["source"] == "https://arxiv.org/abs/2606.32039"
    assert frontmatter["github_links"] == ["https://github.com/Tencent-Hunyuan/GEAR"]
    assert frontmatter["huggingface_model_links"] == [
        "https://huggingface.co/BinLin203/GEAR-VQ",
        "https://huggingface.co/BinLin203/Warmup-LFQ",
    ]
    assert frontmatter["huggingface_dataset_links"] == ["https://huggingface.co/datasets/BinLin203/GEAR-Data"]
    metadata_text = json.dumps(frontmatter, sort_keys=True)
    assert "https://huggingface.co/papers/2606.32039" not in metadata_text
    assert "https://huggingface.co/models?other=arxiv:2606.32039" not in metadata_text
    assert "https://huggingface.co/spaces/BinLin203/GEAR-Demo" not in metadata_text
    supplied_resources = json.loads((workdir / "supplied_page_resources.json").read_text(encoding="utf-8"))
    assert supplied_resources["ok"] is True
    assert set(supplied_resources["platforms_checked"]) == {"paperswithcode", "huggingface"}



def test_raw_fast_evidence_bundle_metadata_resource_links_require_verified_health() -> None:
    from ops import raw_fast_evidence_bundle

    links = raw_fast_evidence_bundle.metadata_resource_links(
        {
            "source_exposed_resources": {
                "github": [
                    {"url": "https://github.com/example/reachable", "status": "verified_present"},
                    {"url": "https://github.com/example/unresolved", "status": "probe_failed"},
                ],
                "hf": [
                    {"url": "https://huggingface.co/example/model-ok", "hf_kind": "models", "status": "verified_present"},
                    {"url": "https://huggingface.co/example/model-bad", "hf_kind": "models", "status": "probe_failed"},
                    {"url": "https://huggingface.co/datasets/example/data-ok", "hf_kind": "datasets", "status": "verified_present"},
                    {"url": "https://huggingface.co/spaces/example/demo", "hf_kind": "spaces", "status": "verified_present"},
                ],
            }
        }
    )

    assert links == {
        "github_links": ["https://github.com/example/reachable"],
        "huggingface_model_links": ["https://huggingface.co/example/model-ok"],
        "huggingface_dataset_links": ["https://huggingface.co/datasets/example/data-ok"],
    }


def test_raw_fast_evidence_bundle_only_abstract_source_links_enter_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    from ops import raw_fast_evidence_bundle

    monkeypatch.setattr(
        raw_fast_evidence_bundle,
        "probe_exact_link_health",
        lambda url, **kwargs: {"ok": True, "url": url, "status": "verified_present"},
    )

    probe = raw_fast_evidence_bundle.build_resource_probe(
        r"""
        \begin{abstract}
        Official implementation: https://github.com/example/paper-code and model weights: https://huggingface.co/example/paper-model.
        \end{abstract}
        % Optional math commands from https://github.com/goodfeli/dlbook_notation.
        \\input{math_commands.tex}
        Related work uses https://github.com/example/reference-code and https://huggingface.co/example/reference-model.
        """,
        {"links": []},
        [],
        health_mode="direct",
    )

    links = raw_fast_evidence_bundle.metadata_resource_links(probe)
    assert links == {
        "github_links": ["https://github.com/example/paper-code"],
        "huggingface_model_links": ["https://huggingface.co/example/paper-model"],
    }
    source_exposed = probe["source_exposed_resources"]
    assert [item["url"] for item in source_exposed["github"]] == ["https://github.com/example/paper-code"]
    assert [item["url"] for item in source_exposed["hf"]] == ["https://huggingface.co/example/paper-model"]
    ignored_reasons = {item["url"]: item["reason"] for item in source_exposed["ignored"]}
    assert ignored_reasons["https://github.com/goodfeli/dlbook_notation"] == "auxiliary_tex_notation_or_template_repo"
    assert ignored_reasons["https://github.com/example/reference-code"] == "non_abstract_source_resource_link"
    assert ignored_reasons["https://huggingface.co/example/reference-model"] == "non_abstract_source_resource_link"
    assert probe["abstract_urls"] == [
        "https://github.com/example/paper-code",
        "https://huggingface.co/example/paper-model",
    ]


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
        _write_tex_agent_ir_source_fixture(workdir)
        return {"ok": True, "extracted_count": 4, "errors": []}

    def forbidden_process_pdf(*args, **kwargs):
        raise AssertionError("arXiv TeX-success route must not call process_pdf or parse PDF text")

    monkeypatch.setattr(raw_fast_evidence_bundle, "fetch_text", fake_fetch_text)
    monkeypatch.setattr(raw_fast_evidence_bundle, "fetch_url_to_file", fake_fetch_url_to_file)
    monkeypatch.setattr(raw_fast_evidence_bundle, "safe_extract_tar", fake_extract_tar)
    monkeypatch.setattr(raw_fast_evidence_bundle, "process_pdf", forbidden_process_pdf)
    _fake_latexpand_success(monkeypatch, raw_fast_evidence_bundle, workdir)

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
    assert LEGACY_RAW_TEXT_KEY not in payload["files"]
    read_plan = json.loads((workdir / "tex_read_plan.json").read_text(encoding="utf-8"))
    assert read_plan["source_kind"] == "tex_source"
    assert read_plan["main_tex"] == "source/main.tex"
    required_sidecars = {
        "tex_agent_ir",
        "tex_agent_map",
        "tex_agent_core",
        "tex_agent_objects",
        "tex_agent_full",
        "tex_agent_audit",
    }
    assert required_sidecars <= set(payload["files"])
    for key in required_sidecars:
        assert (workdir / payload["files"][key]).exists()

    objects_md = (workdir / payload["files"]["tex_agent_objects"]).read_text(encoding="utf-8")
    assert "image_path: `source/figures/cost_curve.png`" in objects_md
    assert "fallback_locator: source_tex=`sections/method.tex`" in objects_md
    assert "source_tex: `sections/method.tex`" not in objects_md
    assert "Appendix Details" not in (workdir / payload["files"]["tex_agent_core"]).read_text(encoding="utf-8")

    handoff = json.loads((workdir / "agent_handoff.json").read_text(encoding="utf-8"))
    assert handoff["source_read_plan"]["source_kind"] == "tex_source"
    assert handoff["automation_next_action"]["action"] == "read_writing_contract_then_tex_semantic_sidecars"
    assert handoff["automation_next_action"]["digest_read_before_source"] == "paper_digest.md"
    assert handoff["automation_next_action"]["semantic_first_reads"] == [
        "paper_map.md",
        "paper_core.md",
        "paper_objects.md",
        "tex_agent_ir_audit.md",
    ]
    assert {"tex_agent_map", "tex_agent_core", "tex_agent_objects", "tex_agent_full"} <= set(handoff["source_refs"])
    brief = json.loads((workdir / "agent_brief.json").read_text(encoding="utf-8"))
    assert brief["evidence_cards"][0]["kind"] == "abstract"


@pytest.mark.requires_fitz
def test_raw_fast_evidence_bundle_direct_pdf_writes_temp_only_and_defaults_docling(tmp_path: Path) -> None:
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
    required_files = {
        "paper_digest",
        "paper_digest_markdown",
        "localized_figures",
        "raw_fast_preflight",
        "agent_brief",
        "evidence_report",
        "note_candidate",
        "agent_handoff",
    }
    assert required_files <= set(payload["files"])
    for key in required_files:
        assert (workdir / payload["files"][key]).exists()
    assert payload["preflight"]["ok"] is True
    assert payload["agent_automation"]["ok"] is True

    digest = json.loads((workdir / "paper_digest.json").read_text(encoding="utf-8"))
    assert digest["ok"] is True
    digest_md = (workdir / "paper_digest.md").read_text(encoding="utf-8")
    assert "https://" not in digest_md
    evidence_report = json.loads((workdir / "evidence_report.json").read_text(encoding="utf-8"))
    assert evidence_report["resource_boundary"]["github_count"] >= 1
    assert evidence_report["resource_boundary"]["review_required"] is False
    handoff = json.loads((workdir / "agent_handoff.json").read_text(encoding="utf-8"))
    assert handoff["source_refs"].get("scientific_digest") == "paper_digest.md"
    assert handoff["manual_reference_paths"] == []
    localized = json.loads((workdir / "localized_figures.json").read_text(encoding="utf-8"))
    assert localized["ok"] is True
    localized_entry = localized["entries"][0]
    assert localized_entry["dest_rel"].startswith("localized_figures_assets/fixture-sidecar-paper/")
    assert localized_entry["raw_note_policy"] == "temporary_inspection_only_do_not_embed_markdown_image"
    assert "markdown" not in localized_entry
    assert (workdir / localized_entry["dest_rel"]).exists()
    assert not (workdir / "resource_boundary_draft.json").exists()
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
    summary = raw_fast_evidence_bundle.resource_status_summary(draft)

    assert draft["github"] == []
    assert draft["project_pages"] == []
    assert {kind: bucket["status"] for kind, bucket in draft["hf"].items()} == {"models": "not_checked", "datasets": "not_checked", "spaces": "not_checked"}
    assert draft["doi"][0]["doi"] == "10.1234/example.paper"
    assert draft["arxiv"][0]["id"] == "2604.08999"
    assert "verified_absent" not in summary
    assert "candidates_unverified" not in summary
    assert "review_required=no" in summary
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

    assert fm["domain"] == "machine-learning"
    assert fm["domain"] != "paper"
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


def test_raw_fast_evidence_bundle_arxiv_category_domains_are_semantic() -> None:
    from ops import raw_fast_evidence_bundle

    assert raw_fast_evidence_bundle.domain_from_arxiv_categories(["cs.CV", "cs.LG"]) == "computer-vision"
    assert raw_fast_evidence_bundle.domain_from_arxiv_categories(["cs.LG", "stat.ML"]) == "machine-learning"
    assert raw_fast_evidence_bundle.domain_from_arxiv_categories(["cs.RO"]) == "robotics"
    assert raw_fast_evidence_bundle.domain_from_arxiv_categories([]) == "machine-learning"
    assert raw_fast_evidence_bundle.domain_from_arxiv_categories(["cs.LG"]) != "paper"


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
    assert preflight["queue_status"]["wiki_pending_count"] == 1
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
