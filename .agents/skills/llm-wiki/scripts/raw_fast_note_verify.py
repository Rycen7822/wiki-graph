#!/usr/bin/env python3
"""Raw-fast note verifier for llm-wiki.

This verifier is intentionally narrower than the structured-ingest verifier: raw-fast notes are saved before `_meta`, `log.md`, compiled pages, index totals, and native zvec state are updated, so checks that require raw-map or graph parity are false positives in raw-fast mode.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_VERIFY_HEAD_BYTES = 16 * 1024
_VERIFY_WORKERS = 8


def _read_head_text(path: Path, max_bytes: int = _VERIFY_HEAD_BYTES) -> str | None:
    try:
        with open(path, "rb") as fh:
            return fh.read(max_bytes).decode("utf-8", errors="ignore")
    except OSError:
        return None

STRUCTURED_PAPER_SECTIONS = [
    "## 一句话总结",
    "## 论文摘要（中文）",
    "## Motivation",
    "## Methodology",
    "## 关键实验结果 / 作者结论",
    "## 对未来研究的启发",
    "## 可能的局限",
    "## 可继续追问的问题",
]
DEPRECATED_STANDALONE_EVIDENCE_SECTIONS = [
    "## 关键公式 / 机制推导",
    "## 关键图表 / 读图笔记",
    "## 资源与复现状态",
    "# 资源与复现状态",
    "## Evidence trail",
    "# Evidence trail",
    "#Evidence trail",
    "#Evidencetrai1",
]
PLACEHOLDER_RE = re.compile(
    r"(?i)\b(todo|tbd)\b|待补|未处理|略过|未(?:读|阅读)(?:完|全文|原文|论文|材料|source|paper)"
)
FORMULA_EVIDENCE_RE = re.compile(
    r"(?i)(eq\.?\s*\(?\d+\)?|equation|formula|objective|loss|derivation|KL|FLOPs|softmax|logits?|公式|方程|目标函数|损失|机制推导|推导|符号|变量|\\\(|\\\[|\$[^$]+\$)"
)
FORMULA_ABSENCE_RE = re.compile(r"(?i)(无|没有|not expose|no explicit|checked absence).{0,24}(中心|关键|显式|可复用)?(公式|方程|目标函数|equation|formula)")
OBSIDIAN_MATH_DELIMITER_RE = re.compile(r"\\+[()\[\]]")
VISUAL_EVIDENCE_RE = re.compile(
    r"(?i)(fig\.?\s*\d+|figure\s*\d+|table\s*\d+|图\s*\d+|表\s*\d+|图表|表格|曲线|趋势|panel|axis|axes|x-axis|y-axis|ablation|benchmark|metric|吞吐|消融|主表|结果表)"
)
VISUAL_ABSENCE_RE = re.compile(r"(?i)(无|没有|not expose|no useful|checked absence).{0,24}(关键|可复用|中心)?(图表|图|表格|figure|table)")
VISUAL_INVENTORY_STYLE_RE = re.compile(
    r"(?i)(图表证据.{0,24}(整体读法|先给出)|headline\s+figure\s+把|主图用.{0,80}作为.{0,12}x-axis|右侧.{0,40}(堆叠|展示)|把.{0,40}(并列展示|分开)|后续.{0,40}(bar chart|表格).{0,40}支撑)"
)
VISUAL_BROAD_CONCLUSION_RE = re.compile(
    r"(?i)(fig\.?\s*\d+|figure\s*\d+|table\s*\d+|图\s*\d+|表\s*\d+).{0,80}(支撑|支持|结论|显示|说明|定位).{0,120}(核心|重要|关键|多通道|复杂|区域|框架|不是泛泛|卖点)"
)
VISUAL_KEY_DATA_ANCHOR_RE = re.compile(
    r"(?i)(`[^`]{3,}`|\(\s*\d+(?:\.\d+)?\s*,|\d+(?:\.\d+)?\s*%|0\s*[–-]\s*10|->|→|列名|字段|坐标|标尺|x\s*轴|y\s*轴|z\s*轴|axis|score|metric|AUC|accuracy|channel text|question text)"
)
REQUIRED_FRONTMATTER_FIELDS = [
    "title",
    "source",
    "capture_route",
    "captured",
]
ALLOWED_FRONTMATTER_FIELDS = [
    "title",
    "source",
    "created",
    "updated",
    "type",
    "domain",
    "tags",
    "topic_hints",
    "github_links",
    "huggingface_model_links",
    "huggingface_dataset_links",
    "capture_route",
    "captured",
]
STRICT_SECRET_RE = re.compile(r"(?<![A-Za-z0-9_-])sk-(?:proj-)?[A-Za-z0-9_]{20,}(?![A-Za-z0-9_-])")
VERIFIER_PATH = Path(__file__).resolve()

BLOCKER_DIAGNOSTICS = {
    "frontmatter_fields_missing": ("Required raw-note frontmatter fields are missing.", "Add the missing allowed frontmatter fields reported by frontmatter_fields_missing."),
    "frontmatter_fields_extra": ("Raw-note frontmatter contains fields outside the compact contract.", "Move route/resource/probe metadata out of the raw note and keep only allowed frontmatter fields."),
    "structured_sections_missing": ("The structured paper H2 section contract is incomplete.", "Add the missing H2 sections reported by structured_sections_missing."),
    "structured_evidence_sections_insufficient": ("Formula or figure/table evidence is not integrated into the allowed narrative sections.", "Use the nested structured evidence issue diagnostics; edit the raw note content, not the verifier code."),
    "deprecated_standalone_evidence_sections": ("The note still has deprecated standalone evidence/resource headings.", "Fold formulas and figure/table conclusions into Methodology, Results, or Limitations and remove deprecated headings."),
    "non_raw_wiki_hits": ("Duplicate guard found the same source/title outside raw notes.", "Inspect the listed non_raw_wiki_hits before marking this raw note as new."),
    "remote_markdown_images": ("Raw notes must not embed remote Markdown images.", "Remove remote image embeds; keep visual evidence as prose conclusions."),
    "data_uri_images": ("Raw notes must not embed data URI images.", "Remove embedded data URI images; keep visual evidence as prose conclusions."),
    "missing_local_images": ("A local Markdown image reference points to a missing file.", "Remove or repair the local image reference; structured paper raw notes normally should not embed images."),
    "strict_secret_hits": ("Strict secret-like tokens were detected in the raw note.", "Remove or redact secret-like text before closeout."),
    "structured_markdown_images": ("Structured paper raw notes should not contain Markdown image embeds.", "Convert figure/table evidence into prose and remove Markdown image embeds."),
    "obsidian_math_delimiters": ("Raw-note formulas use non-rendering MathJax delimiters for Obsidian.", "Use Obsidian-renderable dollar math: convert inline \\(...\\) to $...$ and display \\[...\\] to $$...$$."),
    "note_exists": ("The raw note path does not exist.", "Create the raw note at the requested raw-file path before closeout."),
    "nonzero_size": ("The raw note file is empty.", "Write the raw note body before closeout."),
    "has_frontmatter": ("The raw note is missing YAML frontmatter.", "Add compact raw-note frontmatter at the top of the file."),
    "structured_heading_order_ok": ("Structured paper sections are out of order.", "Reorder the structured H2 sections to match the raw-fast contract."),
    "duplicate_strict_ok": ("Duplicate patterns matched unexpected raw notes.", "Inspect duplicate_hits and refresh the existing canonical raw note if needed."),
    "tmp_absent": ("One or more temporary paths expected to be cleaned still exist.", "Run safe closeout cleanup or remove only the reported temporary paths."),
}

STRUCTURED_EVIDENCE_DIAGNOSTICS = {
    "formula_evidence_in_methodology": ("Methodology lacks formula/objective evidence or an explicit checked absence statement.", "Integrate formula/objective evidence into Methodology, or explicitly state that the source exposes no central reusable formula/objective."),
    "figure_table_evidence_integrated": ("Methodology, Results, and Limitations lack figure/table evidence or an explicit checked absence statement.", "Integrate figure/table evidence into Methodology, Results, or Limitations, or explicitly state that the source exposes no useful figures/tables."),
    "figure_table_inventory_style": ("Figure/table prose looks like an inventory of chart layout rather than evidence-derived conclusions.", "Replace chart layout descriptions with the scientific conclusion supported by the figure/table: trend, contrast, boundary, metric change, or failure mode."),
    "figure_table_broad_conclusion_without_key_data": ("Figure/table prose states a broad conclusion but lacks the key data/field anchors from the visual/table evidence.", "Add the figure/table's concrete anchors beside the claim: coordinates, row/column labels, metric values, axis values, channel/question fields, or the specific examples that make the conclusion checkable."),
}


def diagnostic_entry(code: str, *, parent_code: str | None = None) -> dict[str, str]:
    summary, fix_hint = STRUCTURED_EVIDENCE_DIAGNOSTICS.get(code) or BLOCKER_DIAGNOSTICS.get(
        code,
        ("Raw-fast verifier reported this blocker.", "Use the report fields and verifier path as the repair anchor."),
    )
    entry = {
        "code": code,
        "summary": summary,
        "fix_hint": fix_hint,
        "owner_path": str(VERIFIER_PATH),
        "owner_function": "integrated_evidence_issues" if code in STRUCTURED_EVIDENCE_DIAGNOSTICS else "main",
        "inspect_hint": "Read owner_path for verifier rule details when needed.",
    }
    if parent_code:
        entry["parent_code"] = parent_code
    return entry


def attach_diagnostics(report: dict, raw_note: Path) -> None:
    diagnostics: list[dict[str, str]] = []
    for blocker in report.get("raw_fast_blockers") or []:
        diagnostics.append(diagnostic_entry(str(blocker)))
        if blocker == "structured_evidence_sections_insufficient":
            for issue in report.get("structured_evidence_sections_insufficient") or []:
                diagnostics.append(diagnostic_entry(str(issue), parent_code=blocker))
    report["verifier_path"] = str(VERIFIER_PATH)
    report["blocker_diagnostics"] = diagnostics
    report["diagnostic_hint"] = {
        "path": str(VERIFIER_PATH),
        "message": "Use blocker_diagnostics as the repair anchor; rerun verification after updating the raw note.",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify one llm-wiki raw-fast note without requiring wiki integration parity")
    p.add_argument("--wiki", required=True, type=Path, help="Wiki root (LLM_WIKI_ROOT from the repository .env)")
    p.add_argument("--raw-file", required=True, help="Raw note path relative to wiki root")
    p.add_argument("--patterns", nargs="*", default=[], help="Duplicate patterns expected to hit only this raw note")
    p.add_argument("--structured-paper", action="store_true", help="Require the structured paper H2 contract")
    p.add_argument("--tmp", nargs="*", default=[], help="Temp paths expected to be absent")
    return p.parse_args()


def has_frontmatter(text: str) -> bool:
    return text.startswith("---\n") and text.find("\n---\n", 4) != -1


def frontmatter(text: str) -> str:
    if not has_frontmatter(text):
        return ""
    return text[4:text.find("\n---\n", 4)]


def frontmatter_keys(fm: str) -> list[str]:
    keys = []
    for line in fm.splitlines():
        if line.startswith((" ", "\t", "-")):
            continue
        match = re.match(r"^([A-Za-z][A-Za-z0-9_-]*):", line)
        if match:
            keys.append(match.group(1))
    return keys


def h2_body(text: str, heading: str) -> str:
    match = re.search(rf"^{re.escape(heading)}\s*$", text, re.M)
    if not match:
        return ""
    next_match = re.search(r"^##\s+", text[match.end():], re.M)
    end = match.end() + next_match.start() if next_match else len(text)
    return text[match.end():end].strip()


def compact_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def obsidian_math_delimiter_issues(text: str) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not OBSIDIAN_MATH_DELIMITER_RE.search(line):
            continue
        snippet = re.sub(r"\s+", " ", line).strip()
        issues.append({"line": line_no, "snippet": snippet[:180]})
    return issues


def broad_visual_conclusion_without_key_data(text: str) -> bool:
    for paragraph in re.split(r"\n\s*\n", text):
        if VISUAL_BROAD_CONCLUSION_RE.search(paragraph) and not VISUAL_KEY_DATA_ANCHOR_RE.search(paragraph):
            return True
    return False


def integrated_evidence_issues(text: str) -> list[str]:
    issues = []
    methodology_body = h2_body(text, "## Methodology")
    if compact_len(methodology_body) < 40 or PLACEHOLDER_RE.search(methodology_body) or not (
        FORMULA_EVIDENCE_RE.search(methodology_body) or FORMULA_ABSENCE_RE.search(methodology_body)
    ):
        issues.append("formula_evidence_in_methodology")

    visual_context = "\n".join(
        h2_body(text, heading)
        for heading in ["## Methodology", "## 关键实验结果 / 作者结论", "## 可能的局限"]
    )
    if compact_len(visual_context) < 40 or PLACEHOLDER_RE.search(visual_context) or not (
        VISUAL_EVIDENCE_RE.search(visual_context) or VISUAL_ABSENCE_RE.search(visual_context)
    ):
        issues.append("figure_table_evidence_integrated")
    if VISUAL_INVENTORY_STYLE_RE.search(visual_context):
        issues.append("figure_table_inventory_style")
    if broad_visual_conclusion_without_key_data(visual_context):
        issues.append("figure_table_broad_conclusion_without_key_data")
    return issues


def deprecated_standalone_evidence_sections(text: str) -> list[str]:
    return [heading for heading in DEPRECATED_STANDALONE_EVIDENCE_SECTIONS if re.search(rf"^{re.escape(heading)}\s*$", text, re.M)]


def main() -> int:
    args = parse_args()
    wiki = args.wiki.resolve()
    raw_rel = args.raw_file
    raw_note = wiki / raw_rel
    text = raw_note.read_text(encoding="utf-8", errors="replace") if raw_note.exists() else ""
    fm = frontmatter(text)

    raw_files = sorted((wiki / "raw/clip").rglob("*.md")) if (wiki / "raw/clip").exists() else []
    fm_keys = frontmatter_keys(fm)
    hits = {pattern: [] for pattern in args.patterns}
    def _check(f: Path) -> list[tuple[str, str]] | None:
        head = _read_head_text(f)
        if head is None:
            return None
        rel = str(f.relative_to(wiki))
        return [(pattern, rel) for pattern in args.patterns if pattern in head]

    if args.patterns:
        with ThreadPoolExecutor(max_workers=_VERIFY_WORKERS) as executor:
            for matched in executor.map(_check, raw_files):
                if matched:
                    for pattern, rel in matched:
                        hits[pattern].append(rel)

    non_raw_hits = []
    for sub in ["_meta", "concepts", "comparisons", "queries", "entities"]:
        d = wiki / sub
        if not d.exists():
            continue
        for f in d.glob("*.md"):
            ftext = f.read_text(encoding="utf-8", errors="ignore")
            if any(pattern and pattern in ftext for pattern in args.patterns):
                non_raw_hits.append(str(f.relative_to(wiki)))

    heading_positions = [text.find(section) for section in STRUCTURED_PAPER_SECTIONS if section in text]
    report = {
        "note_exists": raw_note.exists(),
        "nonzero_size": raw_note.exists() and raw_note.stat().st_size > 0,
        "has_frontmatter": has_frontmatter(text),
        "frontmatter_fields_missing": [field for field in REQUIRED_FRONTMATTER_FIELDS if not re.search(rf"^{re.escape(field)}:", fm, re.M)],
        "frontmatter_fields_extra": [field for field in fm_keys if field not in ALLOWED_FRONTMATTER_FIELDS],
        "structured_sections_missing": [section for section in STRUCTURED_PAPER_SECTIONS if section not in text] if args.structured_paper else [],
        "structured_evidence_sections_insufficient": integrated_evidence_issues(text) if args.structured_paper else [],
        "deprecated_standalone_evidence_sections": deprecated_standalone_evidence_sections(text) if args.structured_paper else [],
        "structured_heading_order_ok": heading_positions == sorted(heading_positions),
        "duplicate_hits": hits,
        "duplicate_strict_ok": all(set(v) == {raw_rel} for v in hits.values()) if args.patterns else True,
        "non_raw_wiki_hits": non_raw_hits,
        "remote_markdown_images": len(re.findall(r"!\[[^\]]*\]\(https?://", text)),
        "data_uri_images": len(re.findall(r"!\[[^\]]*\]\(data:image", text)),
        "markdown_images": len(re.findall(r"!\[[^\]]*\]\([^)]*\)", text)),
        "structured_markdown_images": len(re.findall(r"!\[[^\]]*\]\([^)]*\)", text)) if args.structured_paper else 0,
        "obsidian_math_delimiter_issues": obsidian_math_delimiter_issues(text),
        "missing_local_images": 0,
        "strict_secret_hits": len(STRICT_SECRET_RE.findall(text)),
        "tmp_absent": {path: not Path(path).exists() for path in args.tmp},
    }

    for match in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", text):
        dest = match.group(1).strip()
        if not dest or dest.startswith(("http://", "https://", "data:image")):
            continue
        if not (raw_note.parent / dest).resolve().exists():
            report["missing_local_images"] += 1

    blockers = []
    for key in [
        "frontmatter_fields_missing",
        "frontmatter_fields_extra",
        "structured_sections_missing",
        "structured_evidence_sections_insufficient",
        "deprecated_standalone_evidence_sections",
        "non_raw_wiki_hits",
    ]:
        if report[key]:
            blockers.append(key)
    for key in ["remote_markdown_images", "data_uri_images", "missing_local_images", "strict_secret_hits"]:
        if report[key]:
            blockers.append(key)
    if report["obsidian_math_delimiter_issues"]:
        blockers.append("obsidian_math_delimiters")
    if args.structured_paper and report["structured_markdown_images"]:
        blockers.append("structured_markdown_images")
    for key in ["note_exists", "nonzero_size", "has_frontmatter", "structured_heading_order_ok", "duplicate_strict_ok"]:
        if not report[key]:
            blockers.append(key)
    if not all(report["tmp_absent"].values()):
        blockers.append("tmp_absent")
    report["raw_fast_blockers"] = blockers
    report["raw_fast_ok"] = not blockers
    attach_diagnostics(report, raw_note)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["raw_fast_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
