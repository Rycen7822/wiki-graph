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
