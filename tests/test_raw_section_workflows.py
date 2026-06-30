import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from support import sample_wiki, write  # noqa: E402
from ops.wiki_native_artifacts import build_seed_edges  # noqa: E402
from ops.wiki_native_artifacts import extract_method_atoms  # noqa: E402
from ops.wiki_native_docs import generated_docs_from_state  # noqa: E402
from ops.wiki_native_query_response import expand_native_data_response_with_section_neighbors  # noqa: E402
from ops.wiki_native_query_response import filter_native_data_response_by_section_kind  # noqa: E402
from ops.wiki_native_raw_section_extract import extract_raw_sections  # noqa: E402
from ops.wiki_native_raw_sections import raw_section_query_for_kind  # noqa: E402
from ops.wiki_native_raw_sections import raw_section_specs_for_heading  # noqa: E402
from ops.wiki_native_state import ensure_state_dirs  # noqa: E402
from ops.wiki_native_wiki_checks import audit_raw_note_section_contracts  # noqa: E402
from ops.wiki_native_wiki_checks import structured_heading_warnings  # noqa: E402


def test_extract_raw_sections_indexes_summary_heading_as_retrieval_section(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    write(
        root / "raw/clip/2601/26010105_Summary-Section.md",
        "---\ntitle: Summary Section\nupdated: 2026-05-19 04:10\n---\n"
        "# Summary Section\n\n"
        "## 一句话总结\n\n"
        "A short summary should become a raw_section summary node for paper-level semantic neighborhoods.\n",
    )
    state = tmp_path / "work" / "wikigraph" / "state"
    extract_raw_sections(root, state)
    docs = generated_docs_from_state(state, kind="raw_section")
    summary_docs = [doc for doc in docs if doc.canonical_id == "raw_section:26010105_Summary-Section:summary"]
    assert len(summary_docs) == 1
    assert "section_kind: summary" in summary_docs[0].text
    assert "section_title: 一句话总结" in summary_docs[0].text


def test_extract_raw_sections_recognizes_summary_motivation_methodology_title_variants(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    write(
        root / "raw" / "clip" / "2601" / "26010102_Variant-Sections.md",
        "---\ntitle: Variant Sections\nupdated: 2026-05-19 01:20\n---\n"
        "# Variant Sections\n\n"
        "## 论文摘要（中文）\n\n"
        "This abstract variant summarizes the paper contribution.\n\n"
        "## 研究动机 / 为什么要重新审视检索增强推理\n\n"
        "This motivation variant explains the unresolved setup pressure.\n\n"
        "## 方法拆解\n\n"
        "This methodology variant details a staged retrieval and verification pipeline.\n\n"
        "## 明确失败案例说明方法不是万能\n\n"
        "This caveat heading contains 方法 but is not the methodology section.\n",
    )
    state = tmp_path / "work" / "wikigraph" / "state"
    extract_raw_sections(root, state)
    docs = generated_docs_from_state(state, kind="raw_section")
    variant_docs = {doc.canonical_id.rsplit(":", 1)[-1]: doc for doc in docs if "Variant-Sections" in doc.canonical_id}
    assert {"abstract", "motivation", "methodology"} <= set(variant_docs)
    assert "section_title: 论文摘要（中文）" in variant_docs["abstract"].text
    assert "section_title: 研究动机 / 为什么要重新审视检索增强推理" in variant_docs["motivation"].text
    assert "section_title: 方法拆解" in variant_docs["methodology"].text
    assert "This caveat heading contains 方法" not in variant_docs["methodology"].text


def test_extract_raw_sections_indexes_combined_limitation_future_heading_for_both_kinds(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    write(
        root / "raw/clip/2601/26010103_Combined-Sections.md",
        "---\ntitle: Combined Sections\nupdated: 2026-05-19 03:00\n---\n"
        "# Combined Sections\n\n"
        "## 局限 / Future Works\n\n"
        "The same source section discusses remaining failure modes and follow-up research directions.\n",
    )
    state = tmp_path / "work" / "wikigraph" / "state"
    extract_raw_sections(root, state)
    docs = generated_docs_from_state(state, kind="raw_section")
    combined = {doc.canonical_id.rsplit(":", 1)[-1]: doc for doc in docs if "Combined-Sections" in doc.canonical_id}
    assert {"future", "limitations"} <= set(combined)
    assert "section_title: 局限 / Future Works" in combined["future"].text
    assert "section_title: 局限 / Future Works" in combined["limitations"].text
    assert {spec["kind"] for spec in raw_section_specs_for_heading("局限 / Future Works")} == {"future", "limitations"}


def test_extract_raw_sections_integrates_formula_and_figure_evidence_into_context_sections(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    write(
        root / "raw/clip/2601/26010106_Visual-Evidence.md",
        "---\ntitle: Visual Evidence\ndomain: paper\nupdated: 2026-05-19 04:20\n---\n"
        "# Visual Evidence\n\n"
        "## Methodology\n\n"
        "- Formula evidence is integrated here: Eq. (3) defines the routing objective; symbols and the baseline delta are interpreted in the method narrative.\n\n"
        "## 关键实验结果 / 作者结论\n\n"
        "- Figure 2 has three panels; the x-axis, y-axis, trend, and supported claim are recorded next to the result it supports.\n",
    )
    state = tmp_path / "work" / "wikigraph" / "state"
    extract_raw_sections(root, state)
    docs = generated_docs_from_state(state, kind="raw_section")
    visual_docs = {doc.canonical_id.rsplit(":", 1)[-1]: doc for doc in docs if "Visual-Evidence" in doc.canonical_id}
    assert {"methodology", "results"} <= set(visual_docs)
    assert "section_kind: methodology" in visual_docs["methodology"].text
    assert "Eq. (3) defines the routing objective" in visual_docs["methodology"].text
    assert "section_kind: results" in visual_docs["results"].text
    assert "Figure 2 has three panels" in visual_docs["results"].text
    assert raw_section_specs_for_heading("关键公式 / 机制推导") == []
    assert raw_section_specs_for_heading("关键图表 / 读图笔记") == []
    assert "equation" in raw_section_query_for_kind("methodology", "routing objective")
    assert "figure" in raw_section_query_for_kind("results", "ablation trend")


def test_raw_note_section_contract_audit_reports_combined_and_duplicate_headings(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    write(
        root / "raw/clip/2601/26010104_Strict-But-Messy.md",
        "---\ntitle: Strict But Messy\ndomain: paper\nupdated: 2026-05-19 03:00\n---\n"
        "# Strict But Messy\n\n"
        "## 一句话总结\n\nA compact take.\n\n"
        "## 论文摘要（中文）\n\nAbstract content.\n\n"
        "## Motivation\n\nMotivation content.\n\n"
        "## 方法拆解\n\nPrimary method content.\n\n"
        "## Method: ablation detail\n\nA duplicate method-style section that should be visible to the audit.\n\n"
        "## 局限 / Future Works\n\nCombined limits and future work content.\n\n"
        "## 可继续追问的问题\n\nQuestion content.\n",
    )
    audit = audit_raw_note_section_contracts(root)
    issues = [issue for issue in audit["issues"] if issue["path"].endswith("Strict-But-Messy.md")]
    assert any(issue["type"] == "duplicate_section_kind" and issue["section_kind"] == "methodology" for issue in issues)
    assert any(issue["type"] == "combined_section_heading" and set(issue["section_kinds"]) == {"future", "limitations"} for issue in issues)
    assert audit["issues_by_type"]["duplicate_section_kind"] >= 1


def test_extract_raw_sections_creates_section_virtual_docs_without_wiki_pollution(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    ensure_state_dirs(state)
    write(state / "raw_section_docs" / "stale.md", "stale")
    result = extract_raw_sections(root, state)
    assert result["raw_sections"] == 4
    assert not (state / "raw_section_docs" / "stale.md").exists()
    docs = generated_docs_from_state(state, kind="raw_section")
    by_id = {doc.canonical_id: doc for doc in docs}
    assert "raw_section:26010101_Foo-Paper:future" in by_id
    assert "raw_section:26010101_Foo-Paper:limitations" in by_id
    assert "raw_section:26010101_Foo-Paper:questions" in by_id
    assert "raw_section:26010101_Foo-Paper:methodology" in by_id
    future = by_id["raw_section:26010101_Foo-Paper:future"].text
    assert "section_kind: future" in future
    assert "section_title: 对未来研究的启发" in future
    assert "Future work should connect memory repair" in future
    assert not (root / ".llm-wiki").exists()


def test_extract_raw_sections_uses_unique_filenames_for_long_raw_stems(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    shared = "Long-Section-Retrieval-" + "A" * 150
    for suffix in ["X", "Y"]:
        write(
            root / "raw" / "clip" / "2601" / f"26010100_{shared}{suffix}.md",
            "---\ntitle: Long Stem Paper\nupdated: 2026-05-18 16:00\n---\n# Long Stem Paper\n\n## 对未来研究的启发\n\n- A unique long-stem future section should survive as a separate virtual doc.\n",
        )
    state = tmp_path / "work" / "wikigraph" / "state"
    result = extract_raw_sections(root, state)
    files = list((state / "raw_section_docs").glob("*.md"))
    assert len(files) == result["raw_sections"]


def test_structured_heading_warnings_accept_legacy_raw_heading_variants(tmp_path: Path) -> None:
    path = tmp_path / "raw/clip/2601/26010101_Legacy.md"
    text = """---
title: Legacy Paper
domain: paper
---
# Legacy Paper

## 一句话结论

A compact take.

## 中文摘要 / 核心内容

An abstract variant.

## 研究动机 / 为什么这个问题重要

A motivation variant.

## 方法拆解

A methodology variant with integrated formula evidence: Eq. (2) defines the objective and its variables.

## 关键实验结果 / 作者结论

A result variant with integrated visual evidence: Figure 2 and Table 1 are read next to the claims they support.
"""
    assert structured_heading_warnings(path, text) == []


def test_structured_heading_warnings_report_truly_missing_sections(tmp_path: Path) -> None:
    path = tmp_path / "raw/clip/2601/26010102_Broken.md"
    text = """---
title: Broken Paper
domain: paper
---
# Broken Paper

## 一句话总结

A compact take.
"""
    warnings = structured_heading_warnings(path, text)
    assert len(warnings) == 4
    assert any("missing heading prefix ## 论文摘要" in warning for warning in warnings)
    assert any("missing heading prefix ## Motivation" in warning for warning in warnings)
    assert any("missing heading prefix ## Methodology" in warning for warning in warnings)
    assert any("missing heading prefix ## 关键实验结果" in warning for warning in warnings)
    assert not any("## 关键公式" in warning or "## 关键图表" in warning for warning in warnings)


def test_structured_heading_warnings_skip_legacy_non_structured_arxiv_clippings(tmp_path: Path) -> None:
    path = tmp_path / "raw/clip/2601/26010103_Legacy-Article.md"
    text = """---
title: Legacy Article
domain: "arxiv.org"
source: "https://arxiv.org/abs/2601.0103"
---
# Legacy Article

## Original article

This preserved source clipping predates the structured paper-note schema.
"""
    assert structured_heading_warnings(path, text) == []


def test_section_kind_query_prefix_and_response_filter_target_raw_section_chunks() -> None:
    query = raw_section_query_for_kind("methodology", "retrieval verification")
    assert "raw_section section_kind methodology" in query
    assert "Methodology" in query
    assert "方法拆解" in query
    response = {
        "data": {
            "chunks": [
                {"file_path": "raw_section_docs/a.md", "content": "section_kind: methodology\nA"},
                {"file_path": "raw_section_docs/b.md", "content": "section_kind: abstract\nB"},
                {"file_path": "concepts/c.md", "content": "section_kind: methodology\nC"},
            ],
            "entities": [1],
        },
        "status": "ok",
    }
    filtered = filter_native_data_response_by_section_kind(response, "methodology")
    chunks = filtered["data"]["chunks"]
    assert len(chunks) == 1
    assert chunks[0]["file_path"] == "raw_section_docs/a.md"
    assert response["data"]["chunks"][1]["file_path"] == "raw_section_docs/b.md"


def test_expand_native_data_response_with_section_neighbors_keeps_direct_hits_separate(tmp_path: Path) -> None:
    state = tmp_path / "state"
    ensure_state_dirs(state)
    write(
        state / "section_similarity_edges.jsonl",
        json.dumps(
            {
                "src_id": "raw_section:a:future",
                "tgt_id": "raw_section:b:future",
                "source_section_kind": "future",
                "target_section_kind": "future",
                "source_path": "raw/clip/a.md",
                "target_path": "raw/clip/b.md",
                "source_title": "A",
                "target_title": "B",
                "cosine": 0.88,
                "pair_kind": "future:future",
            },
            ensure_ascii=False,
        )
        + "\n",
    )
    response = {
        "data": {
            "chunks": [
                {"file_path": "raw_section_docs/a.md", "content": "section_id: raw_section:a:future\nsection_kind: future\n"},
            ]
        }
    }
    expanded = expand_native_data_response_with_section_neighbors(response, state, neighbor_k=1, section_kind="future")
    neighbors = expanded["data"]["section_neighbor_expansions"]
    assert len(neighbors) == 1
    assert neighbors[0]["seed_section_id"] == "raw_section:a:future"
    assert neighbors[0]["neighbor_section_id"] == "raw_section:b:future"
    assert neighbors[0]["cosine"] == 0.88
    assert expanded["data"]["chunks"] == response["data"]["chunks"]


def test_generated_virtual_doc_builders_remove_stale_markdown(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    write(
        root / "raw/clip/2601/26010101_Bar-Paper.md",
        "---\ntitle: Bar Paper\nupdated: 2026-05-18 16:00\n---\n# Bar Paper\n\n## Methodology\n\n"
        "This is another direct method with enough detail to become a separate method atom. "
        "It intentionally shares the same date prefix as Foo Paper.\n",
    )
    state = tmp_path / "work" / "wikigraph" / "state"
    ensure_state_dirs(state)
    write(state / "edge_docs" / "stale.md", "stale")
    write(state / "method_atom_docs" / "stale.md", "stale")
    edge_result = build_seed_edges(root, state)
    method_result = extract_method_atoms(root, state)
    assert not (state / "edge_docs" / "stale.md").exists()
    assert not (state / "method_atom_docs" / "stale.md").exists()
    edge_docs = sorted((state / "edge_docs").glob("*.md"))
    edge_doc_contents = {path.name: path.read_text(encoding="utf-8") for path in edge_docs}
    assert len(edge_docs) == edge_result["seed_edges"]
    assert len(list((state / "method_atom_docs").glob("*.md"))) == method_result["method_atoms"]
    assert method_result["method_atoms"] == 2

    second_edge_result = build_seed_edges(root, state)
    second_edge_docs = sorted((state / "edge_docs").glob("*.md"))
    assert second_edge_result["edge_docs_total"] == edge_result["seed_edges"]
    assert second_edge_result["edge_docs_written"] == 0
    assert {path.name: path.read_text(encoding="utf-8") for path in second_edge_docs} == edge_doc_contents
