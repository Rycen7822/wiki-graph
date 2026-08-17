from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from raw_fast_evidence_fixtures import (
    _write_sidecar_source_fixture,
    _write_tiny_pdf,
    _write_tiny_pdf_figure,
    _write_tiny_png,
    bundle_cli_argv,
    sample_wiki,
    wiki_root_machine_pollution,
    write,
)


@pytest.mark.requires_fitz
@pytest.mark.subprocess
def test_raw_fast_evidence_bundle_paper_digest_resource_draft_and_local_figures_are_sidecars(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    source_pdf = tmp_path / "source.pdf"
    _write_tiny_pdf(source_pdf)
    workdir = tmp_path / "bundle-sidecars"
    _write_sidecar_source_fixture(workdir)
    result = subprocess.run(
        bundle_cli_argv(
            source_pdf.as_uri(),
            root,
            workdir,
            "--paper-digest",
            "--resource-draft",
            "--localize-figures",
            "--image-slug",
            "fixture-sidecar-paper",
        ),
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
    assert (workdir / localized_entry["dest_rel"]).exists()
    assert not (workdir / "resource_boundary_draft.json").exists()
    assert wiki_root_machine_pollution(root) == []

def test_raw_fast_evidence_bundle_localize_figures_copy_and_refusal_contracts(tmp_path: Path) -> None:
    from ops import raw_fast_evidence_bundle

    root = sample_wiki(tmp_path)

    conflict_workdir = tmp_path / "bundle-sidecar-conflict"
    _write_sidecar_source_fixture(conflict_workdir)
    conflict = conflict_workdir / "localized_figures_assets" / "fixture-sidecar-paper" / "figure-01-method-figure-shows-the-pipeline.png"
    conflict.parent.mkdir(parents=True, exist_ok=True)
    conflict.write_bytes(b"do-not-overwrite")

    localized = raw_fast_evidence_bundle.localize_source_figures(root, conflict_workdir, "fixture-sidecar-paper")

    assert localized["ok"] is True
    assert conflict.read_bytes() == b"do-not-overwrite"
    assert localized["entries"]
    assert localized["entries"][0]["dest_rel"] != conflict.relative_to(conflict_workdir).as_posix()
    assert localized["entries"][0]["dest_rel"].endswith("-02.png")
    assert (conflict_workdir / localized["entries"][0]["dest_rel"]).exists()

    source_root_workdir = tmp_path / "bundle-source-root-fig"
    _write_tiny_png(source_root_workdir / "source" / "fig" / "method.png")
    write(
        source_root_workdir / "source" / "tex" / "main.tex",
        r"""
\begin{figure}
\includegraphics{fig/method.png}
\caption{Source root method figure}
\label{fig:source-root}
\end{figure}
""".strip(),
    )

    localized = raw_fast_evidence_bundle.localize_source_figures(root, source_root_workdir, "source-root-paper")

    assert localized["ok"] is True
    assert len(localized["entries"]) == 1
    assert localized["entries"][0]["label"] == "fig:source-root"
    assert localized["entries"][0]["source_rel"] == "fig/method.png"
    assert localized["refused"] == []
    assert localized["entries"][0]["dest_rel"].startswith("localized_figures_assets/source-root-paper/")
    assert (source_root_workdir / localized["entries"][0]["dest_rel"]).exists()

    unsafe_workdir = tmp_path / "bundle-localize"
    _write_tiny_png(unsafe_workdir / "source" / "figs" / "method.png")
    write(unsafe_workdir / "source" / "figs" / "secret.txt", "not an image")
    write(
        unsafe_workdir / "source" / "main.tex",
        r"""
\begin{figure}\includegraphics{figs/method.png}\caption{Safe figure}\label{fig:safe}\end{figure}
\begin{figure}\includegraphics{../outside.png}\caption{Traversal figure}\label{fig:outside}\end{figure}
\begin{figure}\includegraphics{figs/secret.txt}\caption{Text file}\label{fig:text}\end{figure}
""".strip(),
    )

    localized = raw_fast_evidence_bundle.localize_source_figures(root, unsafe_workdir, "safe-slug")

    assert localized["ok"] is True
    assert len(localized["entries"]) == 1
    assert localized["entries"][0]["label"] == "fig:safe"
    refusal_reasons = {item["reason"] for item in localized["refused"]}
    assert "path_traversal" in refusal_reasons
    assert "unsupported_extension" in refusal_reasons
    assert localized["entries"][0]["dest_rel"].startswith("localized_figures_assets/safe-slug/")
    assert (unsafe_workdir / localized["entries"][0]["dest_rel"]).exists()
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
