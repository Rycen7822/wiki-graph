import sys
import argparse
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


def _structured_raw_fast_note(title: str, source: str) -> str:
    return f"""---
title: \"{title}\"
source: \"{source}\"
capture_route: \"test synthetic route\"
captured: \"2026-06-06 07:00 CST (+0800)\"
---

## 一句话总结

Synthetic take.

## 论文摘要（中文）

Synthetic abstract.

## Motivation

Synthetic motivation.

## Methodology

Formula evidence is integrated here: Eq. (1) defines $loss = x + y$ and the symbols are explained in the method narrative.

## 关键实验结果 / 作者结论

Figure 1 and Table 1 are integrated beside the result claim and record the observed trend.

## 对未来研究的启发

Future work can reuse the verification harness.

## 可能的局限

The tiny fixture is a synthetic limitation, not a real paper.

## 可继续追问的问题

Which wrapper gate catches failed verification before mark-pending?
"""


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
    assert (workdir / "paper.pdf").exists()
    assert (workdir / "paper.layout.txt").exists()
    assert (workdir / "note_skeleton.md").exists()
    assert (workdir / "evidence_bundle.json").exists()
    assert "raw/clip/" in payload["next_raw_path"]
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
    for key in ["paper_digest", "paper_digest_markdown", "resource_boundary_draft", "resource_boundary_draft_markdown", "note_block_drafts", "localized_figures", "localized_figures_markdown"]:
        assert key in payload["files"]
        assert (workdir / payload["files"][key]).exists()

    digest = json.loads((workdir / "paper_digest.json").read_text(encoding="utf-8"))
    assert digest["ok"] is True
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


def test_raw_fast_verifier_rejects_resource_status_and_extra_frontmatter_metadata(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    raw_rel = "raw/clip/2601/26010107_Bloated-Meta-Paper.md"
    note = _structured_raw_fast_note("Bloated Meta Paper", "https://example.test/bloated-meta.pdf").replace(
        "capture_route: \"test synthetic route\"\n",
        "capture_route: \"test synthetic route\"\ntags: [benchmark, memory]\ntopic_hints: [\"compact metadata\", \"graph routing\"]\nresource_status: \"legacy resource status should stay outside raw notes\"\nsource_pdf: \"https://example.test/bloated-meta.pdf\"\nauthors: [\"A. Author\"]\narxiv_version: \"v1\"\n",
    )
    write(root / raw_rel, note)

    result = subprocess.run(
        [
            sys.executable,
            str(RAW_FAST_VERIFIER),
            "--wiki",
            str(root),
            "--raw-file",
            raw_rel,
            "--structured-paper",
        ],
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode != 0
    assert {"resource_status", "source_pdf", "authors", "arxiv_version"} <= set(payload["frontmatter_fields_extra"])
    assert "frontmatter_fields_extra" in payload["raw_fast_blockers"]


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


def test_raw_fast_verifier_rejects_remote_markdown_images(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    raw_rel = "raw/clip/2601/26010108_Remote-Image-Paper.md"
    write(root / raw_rel, _structured_raw_fast_note("Remote Image Paper", "https://example.test/remote-image.pdf") + "\n![remote](https://example.test/figure.png)\n")

    result = subprocess.run(
        [
            sys.executable,
            str(RAW_FAST_VERIFIER),
            "--wiki",
            str(root),
            "--raw-file",
            raw_rel,
            "--structured-paper",
        ],
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode != 0
    assert payload["remote_markdown_images"] == 1
    assert "remote_markdown_images" in payload["raw_fast_blockers"]


def test_raw_fast_closeout_marks_pending_after_verifier_and_cleans_tmp(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    raw_rel = "raw/clip/2601/26010109_Tiny-Wrapper-Paper.md"
    title = "Tiny Wrapper Paper"
    source = "https://example.test/tiny-wrapper-paper.pdf"
    write(root / raw_rel, _structured_raw_fast_note(title, source))
    fetch_tmp = tmp_path / "fetch-tmp"
    write(fetch_tmp / "scratch.txt", "temporary evidence")
    write(
        fetch_tmp / "evidence_bundle.json",
        json.dumps(
            {
                "ok": True,
                "kind": "direct-pdf",
                "source_url": source,
                "title_guess": title,
                "warnings": [],
                "files": {"pdf": "paper.pdf", "paper_digest": "paper_digest.json", "localized_figures": "localized_figures.json"},
                "timings": {"total_seconds": 12.5, "steps": {"fetch_pdf": {"elapsed_seconds": 1.25}}},
            }
        ),
    )
    result = subprocess.run(
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
            raw_rel,
            "--title",
            title,
            "--source-id",
            source,
            "--pattern",
            title,
            "--pattern",
            source,
            "--topic-hint",
            "wrapper-test",
            "--resource-status-summary",
            "synthetic resources checked",
            "--tmp",
            str(fetch_tmp),
            "--auto-integrate",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["raw_fast_ok"] is True
    assert payload["pre_verify"]["duplicate_strict_ok"] is True
    assert payload["control_scan"]["control_count"] == 0
    assert payload["marked"]["raw_path"] == raw_rel
    assert payload["wiki_integration"]["pending_count"] == 1
    assert payload["wiki_integration"]["should_integrate"] is False
    assert payload["native_refresh_status"]["blocked_by_pending_wiki_integration"] is True
    timing_steps = set(payload["timings"]["steps"])
    assert {"pre_verify", "mark_pending"} <= timing_steps
    assert payload["evidence_reports"]["count"] == 1
    evidence_report_path = Path(payload["evidence_reports"]["summaries"][0]["report_path"])
    assert evidence_report_path.exists()
    assert payload["final_verify"]["tmp_absent"][str(fetch_tmp)] is True
    assert not fetch_tmp.exists()
    ledger = load_pending_wiki_integration_ledger(state)
    assert ledger["pending"][0]["raw_path"] == raw_rel
    assert "resources" not in ledger["pending"][0]["required_sections"]


def _raw_fast_closeout_native_args(tmp_path: Path, mode: str) -> argparse.Namespace:
    return argparse.Namespace(
        root=tmp_path / "wiki",
        state_dir=tmp_path / "work" / "wikigraph" / "state",
        workdir=ROOT,
        timeout=17,
        refresh_timeout=23,
        native_refresh_mode=mode,
    )


def test_raw_fast_closeout_native_refresh_defaults_to_status_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _raw_fast_closeout_native_args(tmp_path, "status")
    status = {"command_returncode": 0, "pending_count": 1, "should_refresh": True}
    calls: list[dict[str, object]] = []

    def fake_run_json(command, *, cwd, timeout):
        calls.append({"command": command, "cwd": cwd, "timeout": timeout})
        if command[3] == "status":
            return {"returncode": 0, "json": {"pending_count": 1, "should_refresh": True}}
        raise AssertionError("default raw-fast closeout should not run native prepare-only")

    monkeypatch.setattr(raw_fast_closeout, "run_json", fake_run_json)

    status_result = raw_fast_closeout.run_native_refresh_status(args)
    refresh_result = raw_fast_closeout.run_native_refresh_if_needed(args, status)

    assert len(calls) == 1
    status_command = calls[0]["command"]
    assert isinstance(status_command, list)
    assert status_command[1:4] == ["-m", "ops.batch_native_refresh", "status"]
    assert "--workdir" in status_command
    assert status_result["pending_count"] == 1
    assert status_result["command_returncode"] == 0
    assert refresh_result["ran"] is False
    assert refresh_result["skipped"] is True
    assert refresh_result["skip_reason"] == "native_refresh_status_only"
    assert refresh_result["refresh_mode"] == "status"
    assert refresh_result["pending_count"] == 1


def test_raw_fast_closeout_native_refresh_prepare_mode_uses_batch_native_refresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _raw_fast_closeout_native_args(tmp_path, "prepare")
    status = {"command_returncode": 0, "pending_count": 1, "should_refresh": True}
    calls: list[dict[str, object]] = []

    def fake_run_json(command, *, cwd, timeout):
        calls.append({"command": command, "cwd": cwd, "timeout": timeout})
        if command[3] == "status":
            return {"returncode": 0, "json": {"pending_count": 1, "should_refresh": True}}
        return {
            "returncode": 0,
            "json": {
                "prepared_only": True,
                "skipped": False,
                "status_before": status,
                "build": {"ok": True},
            },
        }

    monkeypatch.setattr(raw_fast_closeout, "run_json", fake_run_json)

    status_result = raw_fast_closeout.run_native_refresh_status(args)
    refresh_result = raw_fast_closeout.run_native_refresh_if_needed(args, status)

    assert len(calls) == 2
    refresh_command = calls[1]["command"]
    assert isinstance(refresh_command, list)
    assert refresh_command[1:5] == ["-m", "ops.batch_native_refresh", "refresh", "--prepare-only"]
    assert status_result["pending_count"] == 1
    assert status_result["command_returncode"] == 0
    assert refresh_result["ran"] is True
    assert refresh_result["prepared_only"] is True
    assert refresh_result["refresh_mode"] == "prepare"
    assert refresh_result["status"]["pending_count"] == 1


def test_raw_fast_closeout_does_not_mark_pending_when_verifier_fails(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    raw_rel = "raw/clip/2601/26010110_Bad-Wrapper-Paper.md"
    write(root / raw_rel, "---\ntitle: Bad\nsource: https://example.test/bad.pdf\n---\n\n## Methodology\n\nTODO\n")
    result = subprocess.run(
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
            raw_rel,
            "--title",
            "Bad Wrapper Paper",
            "--source-id",
            "https://example.test/bad.pdf",
            "--pattern",
            "https://example.test/bad.pdf",
        ],
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode != 0
    assert payload["stage"] == "pre_verify"
    assert payload["raw_fast_ok"] is False
    _assert_timing_step(payload, "pre_verify")
    assert "mark_pending" not in payload["timings"]["steps"]
    assert load_pending_wiki_integration_ledger(state)["pending"] == []


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


def test_raw_fast_closeout_refuses_non_tmp_cleanup_before_marking(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    raw_rel = "raw/clip/2601/26010111_Unsafe-Cleanup-Paper.md"
    title = "Unsafe Cleanup Paper"
    source = "https://example.test/unsafe-cleanup-paper.pdf"
    write(root / raw_rel, _structured_raw_fast_note(title, source))
    unsafe_tmp = root / "raw" / "clip" / "unsafe-bundle"
    write(unsafe_tmp / "scratch.txt", "must not be deleted")
    result = subprocess.run(
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
            raw_rel,
            "--title",
            title,
            "--source-id",
            source,
            "--pattern",
            source,
            "--tmp",
            str(unsafe_tmp),
            "--auto-integrate",
        ],
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode != 0
    assert payload["stage"] == "cleanup_preflight"
    assert payload["cleanup_preflight"][0]["ok"] is False
    assert unsafe_tmp.exists()
    assert load_pending_wiki_integration_ledger(state)["pending"] == []


def test_raw_fast_closeout_final_verify_only_waives_post_integration_non_raw_hits() -> None:
    from ops import raw_fast_closeout

    pre = {"raw_fast_ok": True, "non_raw_wiki_hits": [], "raw_fast_blockers": []}
    final = {"raw_fast_ok": False, "non_raw_wiki_hits": ["concepts/after.md"], "raw_fast_blockers": ["non_raw_wiki_hits"]}
    assert raw_fast_closeout.final_verify_acceptable(pre, final) is True

    final_with_secret = dict(final, raw_fast_blockers=["non_raw_wiki_hits", "strict_secret_hits"])
    assert raw_fast_closeout.final_verify_acceptable(pre, final_with_secret) is False

    final_with_tmp_left = dict(final, tmp_absent={"/tmp/raw-fast": False})
    assert raw_fast_closeout.final_verify_acceptable(pre, final_with_tmp_left) is False


def test_raw_fast_closeout_fast_final_verify_records_tmp_absence(tmp_path: Path) -> None:
    state = tmp_path / "state"
    args = argparse.Namespace(state_dir=state, raw_file="raw/clip/2601/26010112_Fast-Final.md")
    tmp_bundle = tmp_path / "already-cleaned"
    pre = {
        "command_returncode": 0,
        "raw_fast_ok": True,
        "note_exists": True,
        "nonzero_size": True,
        "has_frontmatter": True,
        "frontmatter_fields_missing": [],
        "structured_sections_missing": [],
        "structured_evidence_sections_insufficient": [],
        "deprecated_standalone_evidence_sections": [],
        "structured_heading_order_ok": True,
        "duplicate_strict_ok": True,
        "non_raw_wiki_hits": [],
        "remote_markdown_images": 0,
        "data_uri_images": 0,
        "missing_local_images": 0,
        "strict_secret_hits": 0,
    }

    report = raw_fast_closeout.fast_final_verify_from_pre(args, pre, [tmp_bundle])

    assert report["raw_fast_ok"] is True
    assert report["fast_final_verify"] is True
    assert report["tmp_absent"] == {str(tmp_bundle): True}
    assert Path(report["report_path"]).exists()


def test_raw_fast_closeout_compact_log_entry_is_bounded() -> None:
    args = argparse.Namespace(
        title="Compact Log Paper",
        source_id="https://arxiv.org/abs/2601.0101",
        raw_file="raw/clip/2601/26010112_Compact-Log-Paper.md",
        resource_status_summary="official abs/pdf/source verified; claimed code unresolved",
    )
    output = {
        "raw_fast_ok": True,
        "final_verify": {"report_path": "/state/raw_fast_reports/compact_final_verify.json"},
        "wiki_integration": {"pending_count": 4, "actionable_pending_count": 4, "threshold": 10, "should_integrate": False, "next_required_action": "none"},
        "native_refresh_status": {"blocked_by_pending_wiki_integration": True, "graph_ready_pending_count": 0, "should_refresh": False},
    }

    entry = raw_fast_closeout.build_compact_log_entry(args, output)

    assert len(entry.splitlines()) <= 5
    assert "26010112_Compact-Log-Paper.md" in entry
    assert "raw_fast_ok=true" in entry
    assert "checksums" not in entry.lower()


def test_raw_fast_closeout_blocked_log_distinguishes_standalone_native_ledger(tmp_path: Path) -> None:
    args = argparse.Namespace(
        title="Blocked Native Ledger Paper",
        source_id="https://arxiv.org/abs/2601.0102",
        raw_file="raw/clip/2601/26010113_Blocked-Native-Ledger-Paper.md",
        resource_status_summary="official abs/pdf/source verified",
    )
    wiki_status = {
        "pending_count": 2,
        "actionable_pending_count": 2,
        "review_pending_count": 0,
        "blocking_pending_count": 2,
        "threshold": 10,
        "should_integrate": False,
        "next_required_action": "wiki_integration",
    }
    standalone_status = {
        "command_returncode": 0,
        "pending_count": 1,
        "should_refresh": True,
        "ledger_path": str(tmp_path / "state" / "pending_native_refresh.json"),
    }

    native_status = raw_fast_closeout.synthesize_blocked_native_refresh_status(
        args,
        wiki_status,
        standalone_status=standalone_status,
    )
    output = {
        "raw_fast_ok": True,
        "final_verify": {"report_path": "/state/raw_fast_reports/blocked_final_verify.json"},
        "wiki_integration": wiki_status,
        "native_refresh_status": raw_fast_closeout.compact_native_refresh_status(native_status),
    }

    entry = raw_fast_closeout.build_compact_log_entry(args, output)

    assert native_status["blocked_by_pending_wiki_integration"] is True
    assert native_status["graph_ready_pending_count"] == 0
    assert native_status["standalone_native_pending_count"] == 1
    assert native_status["standalone_native_should_refresh"] is True
    assert "graph-ready pending `0`" in entry
    assert "standalone native ledger pending `1`" in entry
