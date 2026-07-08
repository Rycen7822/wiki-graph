from __future__ import annotations

import json
from pathlib import Path

import pytest

from raw_fast_evidence_fixtures import (
    _write_fake_docling_outputs,
    _write_sectioned_main_with_long_appendix_fixture,
    _write_tex_first_source_fixture,
    sample_wiki,
)


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

def test_raw_fast_evidence_bundle_title_resolution_contracts() -> None:
    from ops import raw_fast_evidence_bundle

    tex = r"""
\newcommand{\makefntext}[1]{\title{makefntext 1 xt 1em makefnmark hbox}}
\renewcommand{\title}[1]{#1}
\title[Short Fixture]{Real Semantic Handoff Paper}
\begin{document}
\maketitle
\end{document}
"""
    text = "<!-- image -->\n\n## Next-Generation Agentic Reinforcement Learning Systems Enable Self-Evolving Agents\n\n## Abstract\nBody"

    assert raw_fast_evidence_bundle.title_from_text(text) == "Next-Generation Agentic Reinforcement Learning Systems Enable Self-Evolving Agents"
    assert (
        raw_fast_evidence_bundle.pdf_title_from_text_or_url(
            "<!-- image -->\n\n<!-- formula-not-decoded -->",
            "https://raw.githubusercontent.com/areal-project/AReaL/main/docs/paper/AReaL2.0_report.pdf",
        )
        == "AReaL2.0_report"
    )
    assert raw_fast_evidence_bundle.tex_title_from_text(tex) == "Real Semantic Handoff Paper"

def test_raw_fast_evidence_bundle_recognizes_cross_site_arxiv_routes() -> None:
    from ops import raw_fast_evidence_bundle

    cases = [
        ("https://paperswithcode.co/paper/2606.28436", "2606.28436"),
        ("https://paperswithcode.co/paper/arxiv/2606.30410v2", "2606.30410"),
        ("https://huggingface.co/papers/2606.02572", "2606.02572"),
        ("https://modelscope.ai/papers/2606.07591", "2606.07591"),
        ("https://www.modelscope.ai/papers/2606.07591v2", "2606.07591"),
        ("https://www.alphaxiv.org/abs/2606.04036", "2606.04036"),
        ("https://alphaxiv.org/overview/2606.04036v3?tab=discussion", "2606.04036"),
    ]

    for url, expected_id in cases:
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

def test_raw_fast_evidence_bundle_recognizes_openreview_routes() -> None:
    from ops import raw_fast_evidence_bundle

    for url in [
        "https://openreview.net/forum?id=mLhZzo7BIb",
        "https://openreview.net/pdf?id=mLhZzo7BIb",
        "https://openreview.net/attachment?id=mLhZzo7BIb&name=pdf",
    ]:
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
