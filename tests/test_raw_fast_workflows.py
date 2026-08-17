import sys
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW_FAST_VERIFIER = Path.home() / ".hermes" / "skills" / "research" / "llm-wiki" / "scripts" / "raw_fast_note_verify.py"

from support import sample_wiki, structured_raw_fast_note, write  # noqa: E402

pytestmark = [pytest.mark.external_skill, pytest.mark.subprocess]

if not RAW_FAST_VERIFIER.is_file():
    pytest.skip(f"external skill missing: {RAW_FAST_VERIFIER}", allow_module_level=True)


def _run_verifier(root: Path, raw_rel: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
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


@pytest.mark.parametrize(
    "title,slug,source,kind",
    [
        ("Bloated Meta Paper", "Bloated-Meta-Paper", "https://example.test/bloated-meta.pdf", "frontmatter"),
        ("Resource Link Metadata Paper", "Resource-Link-Metadata-Paper", "https://arxiv.org/abs/2606.32039", "resource_links"),
        ("Remote Image Paper", "Remote-Image-Paper", "https://example.test/remote-image.pdf", "remote_image"),
        ("Renderable Math Paper", "Renderable-Math-Paper", "https://example.test/renderable-math.pdf", "math"),
    ],
)
def test_raw_fast_verifier_structured_paper_cases(tmp_path: Path, title: str, slug: str, source: str, kind: str) -> None:
    root = sample_wiki(tmp_path)
    raw_rel = f"raw/clip/2601/26010107_{slug}.md"
    note = structured_raw_fast_note(title, source)
    if kind == "frontmatter":
        note = note.replace(
            "capture_route: \"test synthetic route\"\n",
            "capture_route: \"test synthetic route\"\ntags: [benchmark, memory]\ntopic_hints: [\"compact metadata\", \"graph routing\"]\nresource_status: \"legacy resource status should stay outside raw notes\"\nsource_pdf: \"https://example.test/bloated-meta.pdf\"\nauthors: [\"A. Author\"]\narxiv_version: \"v1\"\n",
        )
    elif kind == "resource_links":
        note = note.replace(
            "capture_route: \"test synthetic route\"\n",
            "capture_route: \"test synthetic route\"\n"
            "github_links:\n"
            "  - https://github.com/Tencent-Hunyuan/GEAR\n"
            "huggingface_model_links:\n"
            "  - https://huggingface.co/BinLin203/GEAR-VQ\n"
            "huggingface_dataset_links:\n"
            "  - https://huggingface.co/datasets/BinLin203/GEAR-Data\n",
        )
    elif kind == "remote_image":
        note = note + "\n![remote](https://example.test/figure.png)\n"
    else:
        note = note.replace(
            "Eq. (1) defines $loss = x + y$ and the symbols are explained in the method narrative.",
            r"the method defines \(M=\\mathrm{Extract}(D)\) and display math \[R(q)=R_f(q)\\cup R_p(q)\], with symbols explained in the method narrative.",
        )
    write(root / raw_rel, note)

    result = _run_verifier(root, raw_rel)
    payload = json.loads(result.stdout)

    if kind == "resource_links":
        assert result.returncode == 0
        assert payload["frontmatter_fields_extra"] == []
        assert "frontmatter_fields_extra" not in payload["raw_fast_blockers"]
        return

    assert result.returncode != 0
    if kind == "frontmatter":
        assert {"resource_status", "source_pdf", "authors", "arxiv_version"} <= set(payload["frontmatter_fields_extra"])
        assert "frontmatter_fields_extra" in payload["raw_fast_blockers"]
    elif kind == "remote_image":
        assert payload["remote_markdown_images"] == 1
        assert "remote_markdown_images" in payload["raw_fast_blockers"]
    else:
        assert "obsidian_math_delimiters" in payload["raw_fast_blockers"]
        assert payload["obsidian_math_delimiter_issues"][0]["line"] > 0
        assert "\\(" in payload["obsidian_math_delimiter_issues"][0]["snippet"]
        diagnostics = {item["code"]: item for item in payload["blocker_diagnostics"]}
        assert diagnostics["obsidian_math_delimiters"]["fix_hint"].startswith("Use Obsidian-renderable")


def test_raw_fast_verifier_rejects_unintegrated_visual_evidence_patterns(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    cases = [
        (
            "Visual Evidence Missing",
            "visual-evidence",
            "The experimental conclusion is summarized in prose without naming any figure or table evidence.",
            "figure_table_evidence_integrated",
        ),
        (
            "Chart Inventory Paper",
            "chart-inventory",
            "图表证据先给出整体读法：headline Figure 把同一 prompt 下的四类输出并列展示，"
            "主图用原模型 probe score 作为 x-axis、最终模型 probe score 作为 y-axis，"
            "右侧堆叠图表按 KL penalty 与 detector penalty 展示 policy type 分布。",
            "figure_table_inventory_style",
        ),
        (
            "Broad Visual Conclusion Paper",
            "broad-visual",
            "Figure 1 支撑核心定位：图表证据显示该系统位于复杂但可解释的区域，"
            "结论是该 substrate 的卖点不是简化单元，而是暴露复杂交互。",
            "figure_table_broad_conclusion_without_key_data",
        ),
    ]
    for title, slug, replacement, expected_code in cases:
        raw_rel = f"raw/clip/2601/26010109_{slug}.md"
        note = structured_raw_fast_note(title, f"https://example.test/{slug}.pdf").replace(
            "Figure 1 and Table 1 are integrated beside the result claim and record the observed trend.",
            replacement,
        )
        write(root / raw_rel, note)

        result = _run_verifier(root, raw_rel)
        payload = json.loads(result.stdout)
        diagnostics = {item["code"]: item for item in payload["blocker_diagnostics"]}

        assert result.returncode != 0
        assert payload["verifier_path"].endswith("raw_fast_note_verify.py")
        assert "structured_evidence_sections_insufficient" in payload["raw_fast_blockers"]
        assert expected_code in payload["structured_evidence_sections_insufficient"]
        assert diagnostics[expected_code]["fix_hint"]
