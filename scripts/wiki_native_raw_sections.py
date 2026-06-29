#!/usr/bin/env python3
"""Native raw-section contract and query helpers."""

from __future__ import annotations

import re
from typing import Any

RAW_SECTION_SPECS = [
    {
        "kind": "summary",
        "canonical_title": "一句话总结",
        "prefixes": ["一句话总结", "一句话结论", "tl;dr", "tldr", "最小闭环", "先说整体判断", "核心要点"],
    },
    {
        "kind": "abstract",
        "canonical_title": "论文摘要",
        "prefixes": ["论文摘要", "中文摘要", "摘要", "abstract", "paper abstract"],
    },
    {
        "kind": "motivation",
        "canonical_title": "Motivation",
        "prefixes": ["motivation", "研究动机", "显式 motivation", "动机"],
    },
    {
        "kind": "methodology",
        "canonical_title": "Methodology",
        "prefixes": ["methodology", "method:", "method：", "method /", "method ", "方法", "方法论", "技术路线", "实验方法", "阅读方法"],
    },
    {
        "kind": "results",
        "canonical_title": "关键实验结果 / 作者结论",
        "prefixes": ["关键实验结果", "实验结果", "作者结论", "主要结果", "结果", "评估结果", "results", "experiments", "evaluation"],
        "contains": ["关键实验结果", "作者结论", "experimental results", "main results", "evaluation results", "benchmark results"],
    },
    {
        "kind": "future",
        "canonical_title": "对未来研究的启发",
        "contains": ["对未来研究的启发", "未来研究", "启发", "future work", "future research", "inspiration"],
    },
    {
        "kind": "limitations",
        "canonical_title": "可能的局限",
        "contains": ["可能的局限", "局限", "limitations", "limitation", "failure mode", "failure modes"],
    },
    {
        "kind": "questions",
        "canonical_title": "可继续追问的问题",
        "contains": ["可继续追问的问题", "继续追问", "追问", "open question", "open questions", "unresolved question"],
    },
]
RAW_SECTION_QUERY_ALIASES = {
    "summary": "一句话总结 一句话结论 TLDR paper takeaway concise summary",
    "abstract": "论文摘要 中文摘要 abstract paper summary contribution",
    "motivation": "Motivation 研究动机 problem framing why motivation",
    "methodology": "Methodology 方法拆解 方法机制 方法总览 技术路线 关键公式 机制推导 目标函数 损失函数 method mechanism pipeline architecture equation formula objective derivation symbol definition",
    "results": "关键实验结果 作者结论 实验结果 评估结果 figure table plot chart diagram axes panels ablation trend metrics benchmark results conclusion",
    "future": "对未来研究的启发 future work future research inspiration research ideas",
    "limitations": "可能的局限 limitations limitation failure modes bottlenecks caveats",
    "questions": "可继续追问的问题 open questions unresolved questions research questions",
}
RAW_NOTE_SUMMARY_ALIASES = [
    "一句话总结",
    "一句话结论",
    "tl;dr",
    "tldr",
    "最小闭环",
    "先说整体判断",
    "核心要点",
]
RAW_NOTE_CONTRACT_SECTION_KINDS = ["summary", "abstract", "motivation", "methodology", "results", "future", "limitations", "questions"]
RAW_NOTE_CONTRACT_REQUIRED_KINDS = RAW_NOTE_CONTRACT_SECTION_KINDS
RAW_NOTE_NEAR_MISS_TOKENS = {
    "motivation": ["为什么", "痛点", "背景", "problem framing", "真正回答的问题", "问题定义"],
    "methodology": ["机制", "流程", "pipeline", "架构", "算法", "实验设计", "评估口径", "怎么做", "如何", "公式", "方程", "目标函数", "损失", "objective", "equation", "derivation", "symbol"],
    "results": ["图表", "读图", "图像", "表格", "figure", "fig.", "table", "chart", "plot", "diagram", "实验结果", "benchmark", "metric", "消融"],
    "future": ["展望", "takeaway", "后续研究"],
    "limitations": ["边界", "trade-off", "tradeoff", "caveat", "保守", "不足"],
    "questions": ["open question", "unresolved question", "继续追问"],
}


def normalized_heading_key(title: str) -> str:
    key = re.sub(r"\s+", " ", title.lower()).strip()
    key = re.sub(r"^[\s#>*`_~\-–—•·]+", "", key).strip()
    return key


def raw_section_matches_heading(spec: dict[str, Any], key: str) -> bool:
    prefixes = [str(prefix).lower() for prefix in spec.get("prefixes", [])]
    contains = [str(keyword).lower() for keyword in spec.get("contains", spec.get("keywords", []))]
    return any(key.startswith(prefix) for prefix in prefixes) or any(keyword in key for keyword in contains)


def raw_section_specs_for_heading(title: str) -> list[dict[str, Any]]:
    key = normalized_heading_key(title)
    return [spec for spec in RAW_SECTION_SPECS if raw_section_matches_heading(spec, key)]


def raw_section_spec_for_heading(title: str) -> dict[str, Any] | None:
    specs = raw_section_specs_for_heading(title)
    return specs[0] if specs else None


def summary_heading_matches(title: str) -> bool:
    key = normalized_heading_key(title)
    return any(alias.lower() in key for alias in RAW_NOTE_SUMMARY_ALIASES)


def likely_raw_section_kinds_for_unmatched_heading(title: str) -> list[str]:
    key = normalized_heading_key(title)
    if not key or raw_section_specs_for_heading(title) or summary_heading_matches(title):
        return []
    suggestions = []
    for kind, tokens in RAW_NOTE_NEAR_MISS_TOKENS.items():
        if any(token.lower() in key for token in tokens):
            suggestions.append(kind)
    return suggestions


def raw_section_query_for_kind(section_kind: str, query: str) -> str:
    kind = section_kind.strip().lower()
    aliases = RAW_SECTION_QUERY_ALIASES.get(kind, "")
    return " ".join(part for part in ["raw_section", f"section_kind {kind}", aliases, query.strip()] if part).strip()


def raw_section_kind_from_content(content: str) -> str:
    match = re.search(r"^section_kind:\s*(.+)$", content or "", flags=re.M)
    return match.group(1).strip().lower() if match else ""


def raw_section_id_from_content(content: str) -> str:
    match = re.search(r"^section_id:\s*(.+)$", content or "", flags=re.M)
    return match.group(1).strip() if match else ""
