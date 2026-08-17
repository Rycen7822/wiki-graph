import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from support import sample_wiki, write  # noqa: E402
from ops.wiki_native_wiki_checks import wiki_root_machine_pollution  # noqa: E402

LEGACY_RAW_TEXT_KEY = "raw" + "_text"
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
This paper tests filtered TeX source for source-grounded raw-fast reading.
\end{abstract}
\input{sections/method}
% Source comments should not pollute the filtered agent view.
\section{References}
Bibliographic detail should not be in the filtered agent source.
\bibliography{fixture_refs}
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
\begin{theorem}[Line-aware filtered source statement]
\label{thm:agent-ir-line-aware}
The filtered TeX source exposes theorem-like statements while preserving raw source structure.
\end{theorem}
\begin{proof}[Proof sketch]
The parser keeps a bounded statement card and proof excerpt with exact source lines.
\end{proof}
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
\caption{Main result table for the filtered source fixture.}
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


def bundle_cli_argv(url: str, root: Path, workdir: Path, *extra: str, kind: str = "direct-pdf") -> list[str]:
    return [
        sys.executable,
        "-m",
        "ops.raw_fast_evidence_bundle",
        "--url",
        url,
        "--kind",
        kind,
        "--root",
        str(root),
        "--workdir",
        str(workdir),
        "--probe",
        "none",
        *extra,
    ]


def install_fake_arxiv_fetch(
    monkeypatch: pytest.MonkeyPatch,
    bundle_mod,
    workdir: Path,
    *,
    fetch_text,
    write_source=_write_tex_first_source_fixture,
    extracted_count: int = 2,
    extra: tuple = (),
) -> None:
    def fake_fetch_url_to_file(url: str, dest: Path, timeout: int, max_bytes=None) -> dict:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake eprint tarball")
        return {"ok": True, "url": url, "dest": str(dest), "bytes": dest.stat().st_size, "sha256": "fake"}

    def fake_extract_tar(tar_path: Path, dest: Path) -> dict:
        write_source(workdir)
        return {"ok": True, "extracted_count": extracted_count, "errors": []}

    monkeypatch.setattr(bundle_mod, "fetch_text", fetch_text)
    monkeypatch.setattr(bundle_mod, "fetch_url_to_file", fake_fetch_url_to_file)
    monkeypatch.setattr(bundle_mod, "safe_extract_tar", fake_extract_tar)
    for name, value in extra:
        monkeypatch.setattr(bundle_mod, name, value)


def run_process_arxiv(bundle_mod, url: str, root: Path, workdir: Path, *, probes=None, timeout: int = 5, timed: bool = False, **kwargs):
    kwargs.setdefault("paper_digest", True)
    kwargs.setdefault("resource_draft", True)
    if timed:
        kwargs["timings"] = bundle_mod.TimingRecorder()
    return bundle_mod.process_arxiv(url, root, workdir, "docling", False, list(probes or ["none"]), timeout, **kwargs)
