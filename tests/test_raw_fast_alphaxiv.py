from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ops import raw_fast_evidence_bundle as bundle
from ops import raw_fast_ingest_prepare as prepare
from raw_fast_evidence_fixtures import _write_fake_docling_outputs, sample_wiki

NATIVE_ID = "2610.looped-models-fixed-points"
NATIVE_ABS = f"https://www.alphaxiv.org/abs/{NATIVE_ID}"


def native_metadata(**overrides):
    return {
        "type": "public",
        "sourceName": "alphaXiv",
        "universalId": NATIVE_ID,
        "versionOrder": 2,
        "versionId": "fixture-version",
        "title": "Rethinking at Fixed Points",
        **overrides,
    }


@pytest.mark.parametrize("url,kind,identifier", [
    (NATIVE_ABS, "alphaxiv-native", NATIVE_ID),
    (f"https://alphaxiv.org/pdf/{NATIVE_ID}v1?tab=paper", "alphaxiv-native", NATIVE_ID + "v1"),
    (NATIVE_ABS + ".pdf", "alphaxiv-native", NATIVE_ID),
    (NATIVE_ABS + ".md", "alphaxiv-native", NATIVE_ID),
    (f"https://alphaxiv.org/overview/{NATIVE_ID}", "alphaxiv-native", NATIVE_ID),
    (f"https://pdfs.assets.alphaxiv.org/{NATIVE_ID}v2.pdf", "alphaxiv-native", NATIVE_ID + "v2"),
    ("https://www.alphaxiv.org/abs/2606.04036", "arxiv", "2606.04036"),
    ("https://alphaxiv.org/overview/2606.04036v3?tab=discussion", "arxiv", "2606.04036v3"),
    ("https://www.alphaxiv.org/pdf/2606.04036v3", "arxiv", "2606.04036v3"),
    ("https://www.alphaxiv.org/abs/2606.04036.pdf", "arxiv", "2606.04036"),
])
def test_alphaxiv_routes_keep_native_and_arxiv_identities_separate(url, kind, identifier):
    assert bundle.alphaxiv_paper_id_from_url(url) == identifier
    assert bundle.detect_kind(url, "auto") == kind
    assert bundle.alphaxiv_native_id_from_url(url) == (identifier if kind == "alphaxiv-native" else None)
    assert bundle.arxiv_id_from_url(url) == ("2606.04036" if kind == "arxiv" else None)


@pytest.mark.parametrize("path", [
    "abs/2610.12345junk", "abs/2610.12345-extra", "abs/2613.invalid-month",
    "abs/2610.looped/2606.04036", "blog/2610.looped", "abs/2610.looped.md.pdf",
])
def test_alphaxiv_malformed_ids_do_not_become_arxiv_or_pdf_pages(path):
    url = "https://www.alphaxiv.org/" + path
    assert bundle.arxiv_id_from_url(url) is None
    with pytest.raises(ValueError):
        bundle.detect_kind(url, "auto")


@pytest.mark.parametrize("url,requested", [
    (NATIVE_ABS, "arxiv"), (NATIVE_ABS, "direct-pdf"),
    ("https://alphaxiv.org/abs/2606.04036", "alphaxiv-native"),
    ("https://arxiv.org/abs/2606.04036", "alphaxiv-native"),
])
def test_alphaxiv_explicit_kind_mismatch_is_rejected(url, requested):
    with pytest.raises(ValueError):
        bundle.detect_kind(url, requested)


@pytest.mark.parametrize("url", [NATIVE_ABS, f"https://alphaxiv.org/pdf/{NATIVE_ID}"])
def test_prepare_command_normalizes_native_abs_without_guessing_a_version(tmp_path, url):
    args = prepare.parse_args(["--url", url, "--tmp-root", str(tmp_path), "--print-command"])
    paths = prepare.resolve_prepare_paths(args)
    payload = prepare.run_prepare(args, paths)
    assert payload["ok"] is True
    assert payload["source_url"] == NATIVE_ABS
    assert payload["supplied_url"] == url
    assert paths["kind"] == "alphaxiv-native"
    assert "2610-looped-models-fixed-points" in paths["workdir"].name
    command = payload["command"]
    assert command[command.index("--kind") + 1] == "alphaxiv-native"
    assert command[command.index("--url") + 1] == NATIVE_ABS


def test_native_download_resolves_metadata_version_and_preserves_limit(tmp_path, monkeypatch):
    calls = []

    def text(url, timeout):
        calls.append(url)
        return {"ok": True, "text": json.dumps(native_metadata(versionOrder=1))}

    def download(url, dest, timeout, max_bytes):
        calls.append((url, max_bytes))
        dest.write_bytes(b"%PDF-1.7\nfixture")
        return {"ok": True, "url": url, "bytes": dest.stat().st_size}

    monkeypatch.setattr(bundle, "fetch_text", text)
    monkeypatch.setattr(bundle, "fetch_url_to_file", download)
    result = bundle.fetch_pdf_source_to_file(NATIVE_ABS + "v1", tmp_path / "paper.pdf", 5, max_bytes=128)
    assert result["ok"] is True
    assert result["alphaxiv"]["version_order"] == 1
    assert result["alphaxiv"]["requested_id"] == NATIVE_ID + "v1"
    assert result["alphaxiv"]["source_name"] == "alphaXiv"
    assert calls == [
        f"https://api.alphaxiv.org/papers/v3/{NATIVE_ID}v1",
        (f"https://pdfs.assets.alphaxiv.org/{NATIVE_ID}v1.pdf", 128),
    ]


@pytest.mark.parametrize("metadata,error", [
    (native_metadata(sourceName="arXiv"), "AlphaXivNotNative"),
    (native_metadata(type="private"), "AlphaXivNotNative"),
    (native_metadata(universalId="2610.other-paper"), "AlphaXivIdentityMismatch"),
    (native_metadata(universalId="2610.12345"), "AlphaXivMetadataInvalid"),
    (native_metadata(versionOrder=True), "AlphaXivMetadataInvalid"),
    (native_metadata(versionOrder=0), "AlphaXivMetadataInvalid"),
    (native_metadata(title=""), "AlphaXivMetadataInvalid"),
    ([], "AlphaXivNotNative"),
    ("<html>blocked</html>", "AlphaXivMetadataInvalid"),
])
def test_native_metadata_failure_stops_before_download(tmp_path, monkeypatch, metadata, error):
    response = metadata if isinstance(metadata, str) else json.dumps(metadata)
    monkeypatch.setattr(bundle, "fetch_text", lambda *args: {"ok": True, "text": response})
    monkeypatch.setattr(bundle, "fetch_url_to_file", lambda *args, **kwargs: pytest.fail("invalid metadata must not reach a download"))
    result = bundle.fetch_alphaxiv_pdf_to_file(NATIVE_ID, tmp_path / "paper.pdf", 5)
    assert result["ok"] is False
    assert result["error"] == error
    assert not (tmp_path / "paper.pdf").exists()


def test_native_metadata_unavailable_does_not_guess_latest_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, "fetch_text", lambda *args: {"ok": False, "error": "HTTPError", "message": "403 Forbidden"})
    result = bundle.fetch_alphaxiv_pdf_to_file(NATIVE_ID, tmp_path / "paper.pdf", 5)
    assert result["ok"] is False
    assert result["error"] == "AlphaXivMetadataUnavailable"


def test_native_requested_version_cannot_silently_use_latest(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, "fetch_text", lambda *args: {"ok": True, "text": json.dumps(native_metadata())})
    result = bundle.fetch_alphaxiv_pdf_to_file(NATIVE_ID + "v1", tmp_path / "paper.pdf", 5)
    assert result["ok"] is False
    assert result["error"] == "AlphaXivIdentityMismatch"


def test_native_download_rejects_html_even_with_http_200(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, "fetch_text", lambda *args: {"ok": True, "text": json.dumps(native_metadata())})

    def download(url, dest, timeout, max_bytes):
        dest.write_text("<!doctype html><title>Blocked</title>")
        return {"ok": True, "url": url, "status": 200}

    monkeypatch.setattr(bundle, "fetch_url_to_file", download)
    result = bundle.fetch_alphaxiv_pdf_to_file(NATIVE_ID, tmp_path / "paper.pdf", 5)
    assert result["ok"] is False
    assert result["error"] == "InvalidPDF"
    assert not (tmp_path / "paper.pdf").exists()


def test_native_pdf_cli_keeps_origin_title_handoff_and_sidecar(tmp_path, monkeypatch, capsys):
    root = sample_wiki(tmp_path)
    workdir = tmp_path / "native"
    calls = []

    def text(url, timeout):
        calls.append(url)
        return {"ok": True, "text": json.dumps(native_metadata())}

    def download(url, dest, timeout, max_bytes):
        calls.append(url)
        dest.write_bytes(b"%PDF-1.7\nfixture")
        return {"ok": True, "url": url, "status": 200, "bytes": dest.stat().st_size}

    def docling(pdf, directory, strict, timeout):
        _write_fake_docling_outputs(directory, "# Incorrect extraction title\n\n## Abstract\nNative PDF evidence.\n\n## Method\nA fixed-point method.")
        return {"ok": True}

    monkeypatch.setattr(bundle, "fetch_text", text)
    monkeypatch.setattr(bundle, "fetch_url_to_file", download)
    monkeypatch.setattr(bundle, "run_pdfinfo", lambda *args: {"ok": True, "stdout": "Pages: 1"})
    monkeypatch.setattr(bundle, "extract_pdf_links", lambda *args: {"ok": True, "links": []})
    monkeypatch.setattr(bundle, "run_docling", docling)
    monkeypatch.setattr(sys, "argv", ["bundle", "--url", NATIVE_ABS, "--root", str(root), "--workdir", str(workdir), "--probe", "none", "--paper-digest", "--resource-draft"])
    assert bundle.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "alphaxiv-native"
    assert payload["source_url"] == NATIVE_ABS
    assert payload["title_guess"] == "Rethinking at Fixed Points"
    assert payload["alphaxiv"]["version_order"] == 2
    assert "arxiv" not in payload
    assert not (workdir / "eprint.tar").exists()
    assert not (workdir / "source").exists()
    frontmatter = json.loads((workdir / "candidate_frontmatter.json").read_text())
    assert frontmatter["source"] == NATIVE_ABS
    assert frontmatter["title"] == "Rethinking at Fixed Points"
    assert frontmatter["capture_route"] == "raw-fast evidence bundle (alphaxiv-native)"
    assert "pdf_url" not in frontmatter
    assert payload["files"]["alphaxiv_metadata"] == "alphaxiv_metadata.json"
    handoff = (workdir / "agent_handoff.md").read_text()
    assert "source kind: alphaxiv-native" in handoff
    assert "alphaXiv native PDF version: v2" in handoff
    assert "no arXiv e-print/TeX source was fetched" in handoff
    prepare.update_agent_handoff(workdir, {
        "closeout_args": {"ok": True},
        "closeout_args_path": "closeout_args.json",
        "closeout_command_preview_path": "closeout_command.preview.sh",
    })
    updated_handoff = (workdir / "agent_handoff.md").read_text()
    assert "source kind: alphaxiv-native" in updated_handoff
    assert "alphaXiv native PDF version: v2" in updated_handoff
    assert "no arXiv e-print/TeX source was fetched" in updated_handoff
    summary = prepare.summarize_evidence_bundle(payload)
    assert summary["alphaxiv"]["version_order"] == 2
    assert "arxiv" not in summary
    assert calls == [f"https://api.alphaxiv.org/papers/v3/{NATIVE_ID}", f"https://pdfs.assets.alphaxiv.org/{NATIVE_ID}v2.pdf"]


def test_native_wrong_explicit_kind_cli_reports_json_before_network(tmp_path):
    result = subprocess.run([
        sys.executable, "-m", "ops.raw_fast_ingest_prepare", "--url", NATIVE_ABS,
        "--kind", "arxiv", "--tmp-root", str(tmp_path), "--print-command",
    ], text=True, capture_output=True)
    assert result.returncode == 1
    assert json.loads(result.stdout)["error"] == "SourceKindMismatch"
    assert not (tmp_path / "paper.pdf").exists()
