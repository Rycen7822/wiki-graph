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


def test_raw_fast_verifier_accepts_source_resource_link_metadata(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    raw_rel = "raw/clip/2601/26010110_Resource-Link-Metadata-Paper.md"
    note = _structured_raw_fast_note("Resource Link Metadata Paper", "https://arxiv.org/abs/2606.32039").replace(
        "capture_route: \"test synthetic route\"\n",
        "capture_route: \"test synthetic route\"\n"
        "github_links:\n"
        "  - https://github.com/Tencent-Hunyuan/GEAR\n"
        "huggingface_model_links:\n"
        "  - https://huggingface.co/BinLin203/GEAR-VQ\n"
        "huggingface_dataset_links:\n"
        "  - https://huggingface.co/datasets/BinLin203/GEAR-Data\n",
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

    assert result.returncode == 0
    assert payload["frontmatter_fields_extra"] == []
    assert "frontmatter_fields_extra" not in payload["raw_fast_blockers"]


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


def test_raw_fast_verifier_rejects_non_obsidian_math_delimiters(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    raw_rel = "raw/clip/2601/26010109_Renderable-Math-Paper.md"
    note = _structured_raw_fast_note("Renderable Math Paper", "https://example.test/renderable-math.pdf").replace(
        "Eq. (1) defines $loss = x + y$ and the symbols are explained in the method narrative.",
        r"the method defines \(M=\\mathrm{Extract}(D)\) and display math \[R(q)=R_f(q)\\cup R_p(q)\], with symbols explained in the method narrative.",
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
    assert "obsidian_math_delimiters" in payload["raw_fast_blockers"]
    assert payload["obsidian_math_delimiter_issues"][0]["line"] > 0
    assert "\\(" in payload["obsidian_math_delimiter_issues"][0]["snippet"]
    diagnostics = {item["code"]: item for item in payload["blocker_diagnostics"]}
    assert diagnostics["obsidian_math_delimiters"]["fix_hint"].startswith("Use Obsidian-renderable")


def test_raw_fast_verifier_explains_visual_evidence_blocker_without_broad_search(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    raw_rel = "raw/clip/2601/26010109_Visual-Evidence-Missing.md"
    note = _structured_raw_fast_note("Visual Evidence Missing", "https://example.test/visual-evidence.pdf").replace(
        "Figure 1 and Table 1 are integrated beside the result claim and record the observed trend.",
        "The experimental conclusion is summarized in prose without naming any figure or table evidence.",
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
    assert payload["verifier_path"].endswith("raw_fast_note_verify.py")
    assert payload["diagnostic_hint"]["path"].endswith("raw_fast_note_verify.py")
    assert "repair anchor" in payload["diagnostic_hint"]["message"]
    assert "figure_table_evidence_integrated" in payload["structured_evidence_sections_insufficient"]
    diagnostics = {item["code"]: item for item in payload["blocker_diagnostics"]}
    assert "structured_evidence_sections_insufficient" in diagnostics
    assert diagnostics["figure_table_evidence_integrated"]["fix_hint"].startswith("Integrate figure/table")
    assert diagnostics["figure_table_evidence_integrated"]["owner_path"].endswith("raw_fast_note_verify.py")
