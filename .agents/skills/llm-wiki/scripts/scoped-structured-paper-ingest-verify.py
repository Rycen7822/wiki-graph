#!/usr/bin/env python3
"""Scoped verification for a structured paper ingest in the layered llm-wiki.

Usage:
  python3 ${WIKI_GRAPH_REPO}/.agents/skills/llm-wiki/scripts/scoped-structured-paper-ingest-verify.py \\
    --wiki ${LLM_WIKI_ROOT} \
    --raw-file raw/clip/YYMM/YYMMDDNN_Title.md \
    --patterns "arXiv ID" "Exact Title" "https://github.com/owner/repo" \
    --changed concepts/foo.md comparisons/bar.md _meta/raw-clip-map.md _meta/topic-map.md \
    --batch-start 260501 --batch-end 260510

The script prints JSON and exits 0 only when the scoped checks pass. It is meant
for post-write verification after a single structured paper note has already
been created and navigation pages have been updated.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REQUIRED_SECTIONS = [
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
PLACEHOLDER_RE = re.compile(r"(?i)\b(todo|tbd)\b|待补|未读|未阅读|略过|未处理")
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

SCOPED_DIAGNOSTICS = {
    "note_required_sections_missing": ("The structured paper H2 section contract is incomplete.", "Add the missing H2 sections reported by note_required_sections_missing."),
    "note_evidence_sections_insufficient": ("Formula or figure/table evidence is not integrated into the allowed narrative sections.", "Use nested structured evidence diagnostics; edit the raw note content, not verifier code."),
    "deprecated_standalone_evidence_sections": ("The note still has deprecated standalone evidence/resource headings.", "Fold formulas and figure/table conclusions into Methodology, Results, or Limitations and remove deprecated headings."),
    "note_frontmatter_fields_missing": ("Required raw-note frontmatter fields are missing.", "Add the missing allowed frontmatter fields reported by note_frontmatter_fields_missing."),
    "note_frontmatter_fields_extra": ("Raw-note frontmatter contains fields outside the compact contract.", "Move route/resource/probe metadata out of the raw note and keep only allowed frontmatter fields."),
    "note_markdown_images": ("Structured paper raw notes should not contain Markdown image embeds.", "Convert figure/table evidence into prose and remove Markdown image embeds."),
    "obsidian_math_delimiters": ("Raw-note formulas use non-rendering MathJax delimiters for Obsidian.", "Use Obsidian-renderable dollar math: convert inline \\(...\\) to $...$ and display \\[...\\] to $$...$$."),
}

STRUCTURED_EVIDENCE_DIAGNOSTICS = {
    "formula_evidence_in_methodology": ("Methodology lacks formula/objective evidence or an explicit checked absence statement.", "Integrate formula/objective evidence into Methodology, or explicitly state that the source exposes no central reusable formula/objective."),
    "figure_table_evidence_integrated": ("Methodology, Results, and Limitations lack figure/table evidence or an explicit checked absence statement.", "Integrate figure/table evidence into Methodology, Results, or Limitations, or explicitly state that the source exposes no useful figures/tables."),
    "figure_table_inventory_style": ("Figure/table prose looks like an inventory of chart layout rather than evidence-derived conclusions.", "Replace chart layout descriptions with the scientific conclusion supported by the figure/table: trend, contrast, boundary, metric change, or failure mode."),
    "figure_table_broad_conclusion_without_key_data": ("Figure/table prose states a broad conclusion but lacks the key data/field anchors from the visual/table evidence.", "Add the figure/table's concrete anchors beside the claim: coordinates, row/column labels, metric values, axis values, channel/question fields, or the specific examples that make the conclusion checkable."),
}


def diagnostic_entry(code: str, *, parent_code: str | None = None) -> dict[str, str]:
    summary, fix_hint = STRUCTURED_EVIDENCE_DIAGNOSTICS.get(code) or SCOPED_DIAGNOSTICS.get(
        code,
        ("Scoped verifier reported this blocker.", "Use the report fields and verifier path as the repair anchor."),
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
    for key in [
        "note_required_sections_missing",
        "note_evidence_sections_insufficient",
        "deprecated_standalone_evidence_sections",
        "note_frontmatter_fields_missing",
        "note_frontmatter_fields_extra",
    ]:
        if report.get(key):
            diagnostics.append(diagnostic_entry(key))
            if key == "note_evidence_sections_insufficient":
                for issue in report.get(key) or []:
                    diagnostics.append(diagnostic_entry(str(issue), parent_code=key))
    if report.get("note_markdown_images"):
        diagnostics.append(diagnostic_entry("note_markdown_images"))
    if report.get("obsidian_math_delimiter_issues"):
        diagnostics.append(diagnostic_entry("obsidian_math_delimiters"))
    report["verifier_path"] = str(VERIFIER_PATH)
    report["blocker_diagnostics"] = diagnostics
    report["diagnostic_hint"] = {
        "path": str(VERIFIER_PATH),
        "message": "Use blocker_diagnostics as the repair anchor; rerun scoped verification after updating the note.",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--wiki", required=True, type=Path)
    p.add_argument("--raw-file", required=True, help="Path relative to wiki root, e.g. raw/clip/2605/26051834_foo.md")
    p.add_argument("--patterns", nargs="*", default=[], help="Duplicate-search strings that should only hit the canonical raw note")
    p.add_argument("--changed", nargs="*", default=[], help="Changed files relative to wiki root, excluding or including the raw note")
    p.add_argument("--batch-start", help="Optional raw filename prefix lower bound, e.g. 260501")
    p.add_argument("--batch-end", help="Optional raw filename prefix upper bound, e.g. 260510")
    p.add_argument("--tmp", nargs="*", default=[], help="Temp paths expected to be absent after cleanup")
    return p.parse_args()


def has_frontmatter(text: str) -> bool:
    return text.startswith("---\n") and text.find("\n---\n", 4) != -1


def frontmatter(text: str) -> str:
    if not has_frontmatter(text):
        return ""
    return text[4 : text.find("\n---\n", 4)]


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


def resolve_ref(wiki: Path, page: Path, ref: str) -> bool:
    if ref.startswith(("http://", "https://")):
        return True
    if any(ch in ref for ch in ["*", "{", "}"]):
        return True
    if ref.startswith("/"):
        return Path(ref).exists()
    if ref.startswith(("../", "./")):
        return (page.parent / ref).resolve().exists()
    return (wiki / ref).exists()


def main() -> int:
    args = parse_args()
    wiki = args.wiki.resolve()
    raw_rel = args.raw_file
    raw_note = wiki / raw_rel
    changed_rels = list(dict.fromkeys([raw_rel] + args.changed))
    changed = [wiki / rel for rel in changed_rels]

    report: dict[str, object] = {}
    text = raw_note.read_text(encoding="utf-8") if raw_note.exists() else ""
    fm = frontmatter(text)
    fm_keys = frontmatter_keys(fm)
    report["note_exists"] = raw_note.exists()
    report["note_required_sections_missing"] = [s for s in REQUIRED_SECTIONS if s not in text]
    report["note_evidence_sections_insufficient"] = integrated_evidence_issues(text)
    report["deprecated_standalone_evidence_sections"] = deprecated_standalone_evidence_sections(text)
    report["note_frontmatter_fields_missing"] = [
        k for k in REQUIRED_FRONTMATTER_FIELDS if not re.search(rf"^{re.escape(k)}:", fm, re.M)
    ]
    report["note_frontmatter_fields_extra"] = [field for field in fm_keys if field not in ALLOWED_FRONTMATTER_FIELDS]
    report["note_markdown_images"] = len(re.findall(r"!\[[^\]]*\]\([^)]*\)", text))
    report["obsidian_math_delimiter_issues"] = obsidian_math_delimiter_issues(text)

    hits: dict[str, list[str]] = {p: [] for p in args.patterns}
    raw_dir = wiki / "raw/clip"
    raw_files = sorted(path for path in raw_dir.rglob("*.md") if path.is_file())
    for f in raw_files:
        ftext = f.read_text(encoding="utf-8", errors="ignore")
        for pattern in args.patterns:
            if pattern in ftext:
                hits[pattern].append(str(f.relative_to(wiki)))
    canonical_set = {raw_rel}
    report["duplicate_hits"] = hits
    report["duplicate_strict_ok"] = all(set(v) == canonical_set for v in hits.values()) if args.patterns else True

    report["raw_count"] = len(raw_files)
    raw_map = (wiki / "_meta/raw-clip-map.md").read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Active raw clips：([0-9]+)", raw_map)
    report["raw_map_active"] = int(m.group(1)) if m else None
    if args.batch_start and args.batch_end:
        batch_count = sum(1 for f in raw_files if args.batch_start <= f.name[:6] <= args.batch_end)
        pattern = rf"\| {re.escape(args.batch_start)}–{re.escape(args.batch_end)} \|\s*([0-9]+)\s*\|"
        bm = re.search(pattern, raw_map)
        report["batch_count"] = batch_count
        report["raw_map_batch"] = int(bm.group(1)) if bm else None

    fm_bad: list[str] = []
    updated_bad: list[str] = []
    source_missing: list[tuple[str, str]] = []
    source_checked = 0
    for page in changed:
        if not page.exists() or page.name in {"log.md", "index.md"}:
            continue
        ptext = page.read_text(encoding="utf-8", errors="ignore")
        if not has_frontmatter(ptext):
            fm_bad.append(str(page.relative_to(wiki)))
            continue
        fm = frontmatter(ptext)
        if page != raw_note:
            um = re.search(r"^updated:\s*(.+)$", fm, re.M)
            if not (um and re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", um.group(1))):
                updated_bad.append(str(page.relative_to(wiki)))
        sm = re.search(r"^sources:\s*\[(.*)\]\s*$", fm, re.M | re.S)
        if sm:
            refs = [r.strip().strip("\"'") for r in sm.group(1).split(",") if r.strip()]
            for ref in refs:
                source_checked += 1
                if not resolve_ref(wiki, page, ref):
                    source_missing.append((str(page.relative_to(wiki)), ref))
    report["frontmatter_bad"] = fm_bad
    report["updated_bad"] = updated_bad
    report["source_refs_checked"] = source_checked
    report["source_refs_missing"] = source_missing[:50]
    report["source_refs_missing_count"] = len(source_missing)

    all_pages = {f.stem for d in ["concepts", "comparisons", "entities", "queries", "_meta"] for f in (wiki / d).glob("*.md")}
    all_pages.update(["index", "SCHEMA", "log"])
    broken: list[tuple[str, str]] = []
    for page in changed:
        if not page.exists() or str(page.relative_to(wiki)).startswith("raw/") or page.name == "log.md":
            continue
        ptext = re.sub(r"```.*?```", "", page.read_text(encoding="utf-8", errors="ignore"), flags=re.S)
        for match in re.finditer(r"\[\[([^\]|#]+)", ptext):
            target = match.group(1).strip()
            if target and target not in all_pages:
                broken.append((str(page.relative_to(wiki)), target))
    report["scoped_broken_wikilinks"] = broken[:50]
    report["scoped_broken_wikilinks_count"] = len(broken)

    remote_images = data_images = missing_images = 0
    strict_secret_count = 0
    for page in changed:
        if not page.exists():
            continue
        ptext = page.read_text(encoding="utf-8", errors="ignore")
        strict_secret_count += sum(1 for line in ptext.splitlines() if STRICT_SECRET_RE.search(line))
        for match in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", ptext):
            dest = match.group(1).strip()
            if dest.startswith("data:image"):
                data_images += 1
            elif dest.startswith(("http://", "https://")):
                remote_images += 1
            elif dest and not (page.parent / dest).resolve().exists():
                missing_images += 1
    report["remote_markdown_images"] = remote_images
    report["data_uri_images"] = data_images
    report["missing_local_images"] = missing_images
    report["strict_secret_count"] = strict_secret_count

    index = (wiki / "index.md").read_text(encoding="utf-8", errors="ignore")
    im = re.search(r"Total pages:\s*(\d+)", index)
    report["index_total"] = int(im.group(1)) if im else None
    report["index_wikilinks"] = len(set(re.findall(r"\[\[([^\]|#]+)", index)))
    report["tmp_absent"] = {path: not Path(path).exists() for path in args.tmp}

    expected_batch_ok = True
    if args.batch_start and args.batch_end:
        expected_batch_ok = report.get("batch_count") == report.get("raw_map_batch")
    report["overall_ok"] = all(
        [
            report["note_exists"],
            not report["note_required_sections_missing"],
            not report["note_evidence_sections_insufficient"],
            not report["deprecated_standalone_evidence_sections"],
            not report["note_frontmatter_fields_missing"],
            not report["note_frontmatter_fields_extra"],
            report["note_markdown_images"] == 0,
            not report["obsidian_math_delimiter_issues"],
            report["duplicate_strict_ok"],
            report["raw_count"] == report["raw_map_active"],
            expected_batch_ok,
            not fm_bad,
            not updated_bad,
            report["source_refs_missing_count"] == 0,
            report["scoped_broken_wikilinks_count"] == 0,
            remote_images == 0,
            data_images == 0,
            missing_images == 0,
            strict_secret_count == 0,
            report["index_total"] == report["index_wikilinks"],
            all(report["tmp_absent"].values()),
        ]
    )
    attach_diagnostics(report, raw_note)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["overall_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
