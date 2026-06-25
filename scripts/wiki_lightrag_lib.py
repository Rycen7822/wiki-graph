#!/usr/bin/env python3
"""Small utilities for the llm-wiki -> LightRAG sidecar upgrade.

All generated state stays under /home/xu/project/wiki/lightrag/state.
The Markdown wiki root is read-only input for these scripts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import hashlib
import json
import os
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ModuleNotFoundError:  # LightRAG uv tool env may omit PyYAML.
    yaml = None  # type: ignore[assignment]

DEFAULT_WIKI_ROOT = Path("/mnt/d/data/Clippings/llm-wiki")
DEFAULT_WORKDIR = Path("/home/xu/project/wiki/lightrag")
DEFAULT_STATE_DIR = DEFAULT_WORKDIR / "state"
DEFAULT_SERVER = "http://127.0.0.1:9621"
PENDING_LIGHTRAG_REFRESH_LEDGER = "pending_lightrag_refresh.json"
PENDING_WIKI_INTEGRATION_LEDGER = "pending_wiki_integration.json"
DEFAULT_PENDING_LIGHTRAG_REFRESH_THRESHOLD = 10
DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD = 10
WIKI_INTEGRATION_ACTIONABLE_STATUSES = {"raw_saved"}
WIKI_INTEGRATION_REVIEW_STATUSES = {"needs_review"}
WIKI_INTEGRATION_TERMINAL_STATUSES = {"failed", "skipped_duplicate"}

COMPILED_DIR_TYPES = {
    "entities": "entity",
    "concepts": "concept",
    "comparisons": "comparison",
    "queries": "query",
}
WIKI_SOURCE_ROOT_PREFIXES = frozenset({"raw", "entities", "concepts", "comparisons", "queries", "_meta"})
META_FILES = ["_meta/source-map.md", "_meta/raw-clip-map.md", "_meta/topic-map.md"]
STATE_SUBDIRS = [
    "edge_docs",
    "method_atom_docs",
    "raw_section_docs",
    "raw_section_audits",
    "section_similarity_reports",
    "evidence_packs",
    "validation_reports",
]
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
POLLUTION_DIRECT_NAMES = {
    ".llm-wiki",
    "lightrag_sync.db",
    "lightrag_manifest.jsonl",
    "seed_edges.jsonl",
    "method_atoms.jsonl",
    "raw_sections.jsonl",
    "retrieval_eval_queries.jsonl",
    "connection_review_queue.jsonl",
    "rag_storage",
    "inputs",
    "edge_docs",
    "method_atom_docs",
    "raw_section_docs",
    "raw_section_audits",
    "section_similarity_reports",
    "section_embeddings.jsonl",
    "section_similarity_edges.candidates.jsonl",
    "section_similarity_edges.jsonl",
    "evidence_packs",
    "validation_reports",
    "scripts",
}
POLLUTION_RECURSIVE_NAMES = {
    "lightrag_sync.db",
    "lightrag_manifest.jsonl",
    "seed_edges.jsonl",
    "method_atoms.jsonl",
    "raw_sections.jsonl",
    "raw_section_audits",
    "section_similarity_reports",
    "section_embeddings.jsonl",
    "section_similarity_edges.candidates.jsonl",
    "section_similarity_edges.jsonl",
    "retrieval_eval_queries.jsonl",
    "connection_review_queue.jsonl",
}
TERMINAL_STATUSES = {"PROCESSED", "FAILED", "processed", "failed"}
SUCCESS_STATUSES = {"PROCESSED", "processed"}


@dataclass(frozen=True)
class WikiDoc:
    path: Path
    rel_path: str
    canonical_id: str
    doc_type: str
    title: str
    frontmatter: dict[str, Any]
    body: str
    text: str
    sha256: str


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def slugify(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", text.strip()).strip("-").lower()
    return (slug or "item")[:max_len].strip("-") or "item"


def generated_doc_filename(identifier: str, max_slug: int = 120) -> str:
    digest = sha256_text(identifier)[:12]
    slug = slugify(identifier.replace(":", "_"), max_slug)
    return f"{slug}-{digest}.md"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def fallback_frontmatter_load(raw: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_key: str | None = None
    current_items: list[str] = []
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if current_key and line.startswith("  - "):
            current_items.append(line.split("- ", 1)[1].strip().strip('"').strip("'"))
            continue
        if current_key:
            data[current_key] = current_items
            current_key = None
            current_items = []
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            current_key = key
            current_items = []
        elif value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            data[key] = [item.strip().strip('"').strip("'") for item in inner.split(",") if item.strip()]
        elif value.lower() in {"true", "false"}:
            data[key] = value.lower() == "true"
        else:
            data[key] = value.strip('"').strip("'")
    if current_key:
        data[current_key] = current_items
    return data


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    match = re.match(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", text, re.S)
    if not match:
        return {}, text
    raw = match.group(1)
    try:
        if yaml is not None:
            data = yaml.safe_load(raw) or {}
        else:
            data = fallback_frontmatter_load(raw)
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    return data, text[match.end() :]


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def display_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(display_scalar(v) for v in value if display_scalar(v))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def canonical_id_for(root: Path, path: Path) -> str:
    rel = path.relative_to(root).as_posix()
    parts = rel.split("/")
    stem = path.stem
    if rel == "index.md":
        return "compiled:index"
    if len(parts) >= 3 and parts[0] == "raw" and parts[1] == "clip":
        return f"raw_clip:{stem}"
    if len(parts) >= 2 and parts[0] in COMPILED_DIR_TYPES:
        return f"compiled:{COMPILED_DIR_TYPES[parts[0]]}:{stem}"
    if len(parts) >= 2 and parts[0] == "_meta":
        return f"meta:{stem}"
    return f"wiki:{rel[:-3] if rel.endswith('.md') else rel}"


def doc_type_for(root: Path, path: Path) -> str:
    rel = path.relative_to(root).as_posix()
    parts = rel.split("/")
    if rel == "index.md":
        return "compiled_index"
    if len(parts) >= 3 and parts[0] == "raw" and parts[1] == "clip":
        return "raw_note"
    if len(parts) >= 2 and parts[0] in COMPILED_DIR_TYPES:
        return f"compiled_{COMPILED_DIR_TYPES[parts[0]]}"
    if len(parts) >= 2 and parts[0] == "_meta":
        return "meta_map"
    return "wiki_markdown"


def title_for(path: Path, frontmatter: dict[str, Any], body: str) -> str:
    title = display_scalar(frontmatter.get("title"))
    if title:
        return title
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem.replace("-", " ")


def make_wiki_doc(root: Path, path: Path) -> WikiDoc:
    text = read_text(path)
    fm, body = parse_frontmatter(text)
    rel = path.relative_to(root).as_posix()
    return WikiDoc(
        path=path,
        rel_path=rel,
        canonical_id=canonical_id_for(root, path),
        doc_type=doc_type_for(root, path),
        title=title_for(path, fm, body),
        frontmatter=fm,
        body=body,
        text=text,
        sha256=sha256_text(text),
    )


def raw_clip_files(root: Path) -> list[Path]:
    raw_clip = root / "raw" / "clip"
    if not raw_clip.exists():
        return []
    return sorted(path for path in raw_clip.rglob("*.md") if path.is_file())


def collect_source_docs(root: Path) -> list[WikiDoc]:
    root = root.resolve()
    paths: list[Path] = []
    if (root / "index.md").exists():
        paths.append(root / "index.md")
    for rel in META_FILES:
        p = root / rel
        if p.exists():
            paths.append(p)
    for dirname in COMPILED_DIR_TYPES:
        d = root / dirname
        if d.exists():
            paths.extend(sorted(d.glob("*.md")))
    paths.extend(raw_clip_files(root))
    unique = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return [make_wiki_doc(root, p) for p in unique]


def find_wikilinks(text: str) -> list[str]:
    out: list[str] = []
    for raw in re.findall(r"\[\[([^\]]+)\]\]", text):
        target = raw.split("|", 1)[0].split("#", 1)[0].strip()
        if target:
            out.append(target)
    return out


def source_urls(frontmatter: dict[str, Any]) -> list[str]:
    keys = ["source", "url", "doi", "project", "github", "official_code", "source_code", "paper", "arxiv"]
    urls: list[str] = []
    for key in keys:
        for value in as_list(frontmatter.get(key)):
            s = display_scalar(value)
            if s and (s.startswith("http://") or s.startswith("https://")):
                urls.append(s)
    return sorted(set(urls))


def compact_body_for_ingest(doc: WikiDoc) -> str:
    """Keep ingestion small enough for the Hermes-backed LightRAG extractor.

    The canonical Markdown stays untouched. This only controls the text handed to
    LightRAG; MethodAtom/edge docs preserve additional retrieval structure.
    """
    if len(doc.text) <= 2500:
        return doc.text
    if doc.doc_type == "compiled_index":
        lines = [line for line in doc.text.splitlines() if line.startswith("#") or line.startswith(">")]
        return "\n".join(lines)[:900]
    if doc.doc_type == "raw_note":
        parts = []
        for title, keys in [
            ("Summary", ["一句话", "摘要", "summary"]),
            ("Motivation", ["motivation", "问题", "背景"]),
            ("Methodology", ["methodology", "方法"]),
            ("Findings", ["关键实验", "结论", "result"]),
            ("Limitations", ["局限", "limitation"]),
            ("Future", ["启发", "未来", "追问", "future"]),
        ]:
            section = section_text(doc.body, keys)
            if section:
                parts.append(f"## {title}\n{first_sentences(section, 350)}")
        if parts:
            return "\n\n".join([f"# {doc.title}", *parts])[:2400]
    # Compiled topic pages keep their opening argument and headings.
    kept = []
    for line in doc.text.splitlines():
        if line.startswith("#") or len("\n".join(kept)) < 1700:
            kept.append(line)
        if len("\n".join(kept)) >= 2400:
            break
    return "\n".join(kept)[:2400]


def limited_scalar(value: Any, max_chars: int = 1200) -> str:
    text = display_scalar(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ...[truncated]"


def make_ingest_text(doc: WikiDoc) -> str:
    fm = doc.frontmatter
    header = {
        "canonical_id": doc.canonical_id,
        "path": doc.rel_path,
        "doc_type": doc.doc_type,
        "title": doc.title,
        "domain": display_scalar(fm.get("domain")),
        "tags": display_scalar(fm.get("tags")),
        "topic_hints": display_scalar(fm.get("topic_hints")),
        "updated": display_scalar(fm.get("updated")),
        "source_urls": limited_scalar(source_urls(fm), 500),
        "wikilinks_out": limited_scalar(find_wikilinks(doc.text), 500),
        "source_refs": limited_scalar(fm.get("sources"), 500),
    }
    lines = ["[LLM_WIKI_DOC]"]
    for key, value in header.items():
        if value:
            lines.append(f"{key}: {value}")
    lines.append("[/LLM_WIKI_DOC]")
    return "\n".join(lines) + "\n\n" + compact_body_for_ingest(doc)


def ensure_state_dirs(state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    for name in STATE_SUBDIRS:
        (state_dir / name).mkdir(parents=True, exist_ok=True)


def pending_lightrag_refresh_ledger_path(state_dir: Path) -> Path:
    return state_dir / PENDING_LIGHTRAG_REFRESH_LEDGER


def default_lightrag_refresh_ledger(threshold: int | None = None) -> dict[str, Any]:
    return {
        "version": 1,
        "threshold": int(threshold or DEFAULT_PENDING_LIGHTRAG_REFRESH_THRESHOLD),
        "last_successful_refresh_at": None,
        "last_successful_raw_count": None,
        "last_successful_import_payload": {},
        "pending": [],
        "dirty": False,
        "last_failed_refresh": None,
    }


def load_lightrag_refresh_ledger(state_dir: Path, threshold: int | None = None) -> dict[str, Any]:
    ensure_state_dirs(state_dir)
    path = pending_lightrag_refresh_ledger_path(state_dir)
    if not path.exists():
        return default_lightrag_refresh_ledger(threshold)
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(ledger, dict):
        raise ValueError(f"{path} must contain a JSON object")
    merged = default_lightrag_refresh_ledger(threshold)
    merged.update(ledger)
    merged["threshold"] = int(threshold if threshold is not None else (merged.get("threshold") or DEFAULT_PENDING_LIGHTRAG_REFRESH_THRESHOLD))
    pending = merged.get("pending") or []
    if not isinstance(pending, list):
        raise ValueError(f"{path} field pending must be a list")
    merged["pending"] = pending
    merged["dirty"] = bool(merged.get("dirty"))
    return merged


def save_lightrag_refresh_ledger(state_dir: Path, ledger: dict[str, Any]) -> Path:
    ensure_state_dirs(state_dir)
    path = pending_lightrag_refresh_ledger_path(state_dir)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def pending_wiki_integration_ledger_path(state_dir: Path) -> Path:
    return state_dir / PENDING_WIKI_INTEGRATION_LEDGER


def default_pending_wiki_integration_ledger(threshold: int | None = None) -> dict[str, Any]:
    return {
        "version": 1,
        "threshold": int(threshold or DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD),
        "last_successful_integration_at": None,
        "last_successful_integration_raw_count": None,
        "last_integrated_paths": [],
        "pending": [],
        "dirty": False,
        "last_failed_integration": None,
    }


def load_pending_wiki_integration_ledger(state_dir: Path, threshold: int | None = None) -> dict[str, Any]:
    ensure_state_dirs(state_dir)
    path = pending_wiki_integration_ledger_path(state_dir)
    if not path.exists():
        return default_pending_wiki_integration_ledger(threshold)
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(ledger, dict):
        raise ValueError(f"{path} must contain a JSON object")
    merged = default_pending_wiki_integration_ledger(threshold)
    merged.update(ledger)
    merged["threshold"] = int(threshold if threshold is not None else (merged.get("threshold") or DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD))
    pending = merged.get("pending") or []
    if not isinstance(pending, list):
        raise ValueError(f"{path} field pending must be a list")
    merged["pending"] = pending
    merged["dirty"] = bool(merged.get("dirty"))
    return merged


def save_pending_wiki_integration_ledger(state_dir: Path, ledger: dict[str, Any]) -> Path:
    ensure_state_dirs(state_dir)
    path = pending_wiki_integration_ledger_path(state_dir)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def mark_pending_wiki_integration(
    state_dir: Path,
    root: Path,
    raw_path: str = "",
    title: str = "",
    source_id: str = "",
    topic_hints: list[str] | None = None,
    required_sections: list[str] | None = None,
    resource_status_summary: str = "",
    status: str = "raw_saved",
    threshold: int | None = None,
) -> dict[str, Any]:
    ledger = load_pending_wiki_integration_ledger(state_dir, threshold=threshold)
    effective_threshold = int(threshold if threshold is not None else (ledger.get("threshold") or DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD))
    ledger["threshold"] = effective_threshold
    captured_at = now_stamp()
    entry = {
        "raw_path": raw_path,
        "title": title or (Path(raw_path).stem if raw_path else ""),
        "source_id": source_id,
        "captured_at": captured_at,
        "status": status,
        "topic_hints": topic_hints or [],
        "required_sections": required_sections or [],
        "resource_status_summary": resource_status_summary,
    }
    pending = list(ledger.get("pending") or [])
    replaced = False
    if raw_path:
        for index, old in enumerate(pending):
            if isinstance(old, dict) and old.get("raw_path") == raw_path:
                pending[index] = {**old, **entry}
                replaced = True
                break
    if not replaced:
        pending.append(entry)
    ledger["pending"] = pending
    ledger["dirty"] = True
    ledger["last_pending_update_at"] = captured_at
    ledger["current_raw_count_at_last_pending_update"] = len(raw_clip_files(root))
    save_pending_wiki_integration_ledger(state_dir, ledger)
    return entry


def pending_wiki_integration_status(
    root: Path,
    state_dir: Path,
    reason: str = "threshold",
    threshold: int | None = None,
) -> dict[str, Any]:
    ledger = load_pending_wiki_integration_ledger(state_dir, threshold=threshold)
    effective_threshold = int(threshold if threshold is not None else (ledger.get("threshold") or DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD))
    pending = ledger.get("pending") or []
    pending_count = len(pending)
    actionable_pending = [item for item in pending if isinstance(item, dict) and str(item.get("status") or "raw_saved") in WIKI_INTEGRATION_ACTIONABLE_STATUSES]
    terminal_pending = [item for item in pending if isinstance(item, dict) and str(item.get("status") or "raw_saved") in WIKI_INTEGRATION_TERMINAL_STATUSES]
    review_pending = [
        item
        for item in pending
        if isinstance(item, dict)
        and str(item.get("status") or "raw_saved") not in WIKI_INTEGRATION_ACTIONABLE_STATUSES
        and str(item.get("status") or "raw_saved") not in WIKI_INTEGRATION_TERMINAL_STATUSES
    ]
    actionable_pending_count = len(actionable_pending)
    terminal_pending_count = len(terminal_pending)
    review_pending_count = len(review_pending)
    blocking_pending_count = actionable_pending_count + review_pending_count
    raw_count = len(raw_clip_files(root))
    normalized_reason = reason.strip().lower().replace("_", "-")
    reasons: list[str] = []
    if normalized_reason == "manual" and blocking_pending_count:
        reasons.append("manual_requested")
    if actionable_pending_count >= effective_threshold:
        reasons.append("pending_threshold_reached")
    if normalized_reason in {"pre-query", "query", "integrate", "wiki-query"} and actionable_pending_count:
        reasons.append("pending_items_for_wiki_integration")
    if review_pending_count:
        reasons.append("pending_items_need_review")
    if normalized_reason in {"pre-query", "query", "integrate", "wiki-query", "manual"} and ledger.get("dirty") and actionable_pending_count:
        reasons.append("dirty_ledger_for_wiki_integration")
    should_integrate = any(reason in set(reasons) for reason in {"manual_requested", "pending_threshold_reached", "pending_items_for_wiki_integration", "dirty_ledger_for_wiki_integration"})
    should_review = review_pending_count > 0
    if should_integrate:
        next_required_action = "wiki_integration"
    elif should_review:
        next_required_action = "manual_review"
    else:
        next_required_action = "none"
    return {
        "reason": normalized_reason,
        "should_integrate": should_integrate,
        "should_review": should_review,
        "next_required_action": next_required_action,
        "reasons": sorted(set(reasons)),
        "pending_count": pending_count,
        "actionable_pending_count": actionable_pending_count,
        "terminal_pending_count": terminal_pending_count,
        "review_pending_count": review_pending_count,
        "blocking_pending_count": blocking_pending_count,
        "threshold": effective_threshold,
        "dirty": bool(ledger.get("dirty")),
        "raw_clip_count": raw_count,
        "last_successful_integration_raw_count": ledger.get("last_successful_integration_raw_count"),
        "last_successful_integration_at": ledger.get("last_successful_integration_at"),
        "ledger_path": str(pending_wiki_integration_ledger_path(state_dir)),
        "pending": pending,
        "actionable_pending": actionable_pending,
        "terminal_pending": terminal_pending,
        "review_pending": review_pending,
    }


def clear_pending_wiki_integration_after_success(
    root: Path,
    state_dir: Path,
    integrated_paths: list[str] | None = None,
    reason: str = "integration",
    mark_lightrag_pending: bool = True,
) -> dict[str, Any]:
    ledger = load_pending_wiki_integration_ledger(state_dir)
    pending = list(ledger.get("pending") or [])
    integrated_path_set = {path for path in (integrated_paths or []) if path}
    marked_lightrag_pending: list[dict[str, Any]] = []
    cleared_pending: list[dict[str, Any]] = []
    remaining_pending: list[Any] = []
    for item in pending:
        if not isinstance(item, dict):
            remaining_pending.append(item)
            continue
        raw_path = str(item.get("raw_path") or "")
        item_status = str(item.get("status") or "raw_saved")
        should_clear = bool(raw_path and raw_path in integrated_path_set) if integrated_path_set else item_status in WIKI_INTEGRATION_ACTIONABLE_STATUSES
        if not should_clear:
            remaining_pending.append(item)
            continue
        cleared_pending.append(item)
        if mark_lightrag_pending and raw_path and item_status in WIKI_INTEGRATION_ACTIONABLE_STATUSES:
            marked_lightrag_pending.append(
                mark_lightrag_refresh_pending(
                    state_dir,
                    root,
                    raw_path=raw_path,
                    title=str(item.get("title") or ""),
                    event_type="batch-wiki-integration",
                    changed_surfaces=["raw-note", "_meta", "compiled-anchors", "log"],
                    expected_sections=[str(section) for section in (item.get("required_sections") or [])],
                )
            )
    cleared_at = now_stamp()
    cleared_paths = [str(item.get("raw_path") or "") for item in cleared_pending if isinstance(item, dict) and item.get("raw_path")]
    remaining_blocking = [
        item
        for item in remaining_pending
        if isinstance(item, dict) and str(item.get("status") or "raw_saved") not in WIKI_INTEGRATION_TERMINAL_STATUSES
    ]
    ledger["last_successful_integration_at"] = cleared_at
    ledger["last_successful_integration_raw_count"] = len(raw_clip_files(root))
    ledger["last_integration_reason"] = reason
    ledger["last_integrated_paths"] = integrated_paths or cleared_paths
    ledger["last_cleared_pending"] = cleared_pending
    ledger["last_cleared_pending_count"] = len(cleared_pending)
    ledger["last_remaining_pending_count"] = len(remaining_pending)
    ledger["last_marked_lightrag_pending"] = marked_lightrag_pending
    ledger["last_marked_lightrag_pending_count"] = len(marked_lightrag_pending)
    ledger["pending"] = remaining_pending
    ledger["dirty"] = bool(remaining_blocking)
    ledger["last_failed_integration"] = None
    save_pending_wiki_integration_ledger(state_dir, ledger)
    return {
        "cleared_count": len(cleared_pending),
        "remaining_pending_count": len(remaining_pending),
        "last_successful_integration_at": ledger["last_successful_integration_at"],
        "last_successful_integration_raw_count": ledger["last_successful_integration_raw_count"],
        "last_integrated_paths": ledger["last_integrated_paths"],
        "marked_lightrag_pending_count": len(marked_lightrag_pending),
        "marked_lightrag_pending": marked_lightrag_pending,
        "ledger_path": str(pending_wiki_integration_ledger_path(state_dir)),
    }


def record_pending_wiki_integration_failure(state_dir: Path, reason: str, message: str = "") -> dict[str, Any]:
    ledger = load_pending_wiki_integration_ledger(state_dir)
    failure = {"at": now_stamp(), "reason": reason, "message": message}
    ledger["last_failed_integration"] = failure
    ledger["dirty"] = True
    save_pending_wiki_integration_ledger(state_dir, ledger)
    return failure


def lightrag_refresh_import_summary(import_report_path: Path) -> dict[str, Any]:
    if not import_report_path.exists():
        return {}
    report = json.loads(import_report_path.read_text(encoding="utf-8", errors="ignore"))
    payload = report.get("payload") if isinstance(report, dict) else {}
    if not isinstance(payload, dict):
        payload = {}
    return {
        "finished_at": report.get("finished_at") if isinstance(report, dict) else None,
        "payload": {key: payload.get(key) for key in ["chunks", "entities", "relationships", "raw_section_chunks", "section_similarity_relationships"] if key in payload},
        "report_path": str(import_report_path),
    }


def _parse_refresh_time(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"]:
        try:
            return dt.datetime.strptime(value[: len(dt.datetime.now().strftime(fmt))], fmt)
        except Exception:
            continue
    return None


def wiki_markdown_latest_mtime(root: Path) -> str | None:
    """Latest mtime among Markdown files that participate in custom_kg.

    `log.md` and `SCHEMA.md` are intentionally ignored by `collect_source_docs`, so they should not make the LightRAG graph look stale after a post-refresh log entry.
    """
    mtimes = [doc.path.stat().st_mtime for doc in collect_source_docs(root) if doc.path.exists()]
    if not mtimes:
        return None
    return dt.datetime.fromtimestamp(max(mtimes)).strftime("%Y-%m-%d %H:%M:%S")


def mark_lightrag_refresh_pending(
    state_dir: Path,
    root: Path,
    raw_path: str = "",
    title: str = "",
    event_type: str = "new_raw_note",
    changed_surfaces: list[str] | None = None,
    expected_sections: list[str] | None = None,
    threshold: int | None = None,
) -> dict[str, Any]:
    ledger = load_lightrag_refresh_ledger(state_dir, threshold=threshold)
    effective_threshold = int(threshold if threshold is not None else (ledger.get("threshold") or DEFAULT_PENDING_LIGHTRAG_REFRESH_THRESHOLD))
    ledger["threshold"] = effective_threshold
    entry = {
        "raw_path": raw_path,
        "title": title or (Path(raw_path).stem if raw_path else ""),
        "event_type": event_type,
        "added_at": now_stamp(),
        "changed_surfaces": changed_surfaces or ["raw", "compiled", "meta", "log"],
        "expected_sections": expected_sections or [],
    }
    pending = list(ledger.get("pending") or [])
    replaced = False
    if raw_path:
        for index, old in enumerate(pending):
            if isinstance(old, dict) and old.get("raw_path") == raw_path:
                pending[index] = {**old, **entry}
                replaced = True
                break
    if not replaced:
        pending.append(entry)
    ledger["pending"] = pending
    ledger["dirty"] = True
    ledger["last_pending_update_at"] = entry["added_at"]
    ledger["current_raw_count_at_last_pending_update"] = len(raw_clip_files(root))
    save_lightrag_refresh_ledger(state_dir, ledger)
    return entry


def pending_lightrag_refresh_status(
    root: Path,
    state_dir: Path,
    reason: str = "threshold",
    threshold: int | None = None,
    import_report_path: Path | None = None,
) -> dict[str, Any]:
    normalized_reason = reason.strip().lower().replace("_", "-")
    ledger = load_lightrag_refresh_ledger(state_dir, threshold=threshold)
    upstream_reason = normalized_reason if normalized_reason in {"pre-query", "query", "manual"} else "threshold"
    upstream_wiki_integration = pending_wiki_integration_status(root, state_dir, reason=upstream_reason)
    upstream_actionable_count = int(upstream_wiki_integration.get("actionable_pending_count") or 0)
    upstream_review_count = int(upstream_wiki_integration.get("review_pending_count") or 0)
    upstream_pending_count = int(upstream_wiki_integration.get("blocking_pending_count") or (upstream_actionable_count + upstream_review_count))
    effective_threshold = int(threshold if threshold is not None else (ledger.get("threshold") or DEFAULT_PENDING_LIGHTRAG_REFRESH_THRESHOLD))
    pending = ledger.get("pending") or []
    pending_count = len(pending)
    raw_count = len(raw_clip_files(root))
    last_success_count = ledger.get("last_successful_raw_count")
    prequery_like = normalized_reason in {"pre-query", "query", "full-graph-fresh", "manual"}
    report_path = import_report_path or (state_dir / "custom_kg_import_report.json")
    import_summary = lightrag_refresh_import_summary(report_path) if prequery_like else {}
    last_success_at = ledger.get("last_successful_refresh_at") or import_summary.get("finished_at")
    reasons: list[str] = []
    blocked_reasons: list[str] = []
    if normalized_reason == "manual":
        reasons.append("manual_requested")
    if pending_count >= effective_threshold:
        reasons.append("pending_threshold_reached")
    if normalized_reason in {"pre-query", "query", "full-graph-fresh"} and pending_count:
        reasons.append("pending_items_for_pre_query")
    if prequery_like and ledger.get("dirty"):
        reasons.append("dirty_ledger_for_pre_query")
    if isinstance(last_success_count, int) and raw_count > last_success_count and prequery_like:
        reasons.append("raw_count_newer_than_last_success")
    if prequery_like and not import_summary:
        reasons.append("import_report_missing")
    latest_mtime = wiki_markdown_latest_mtime(root) if prequery_like else None
    latest_dt = _parse_refresh_time(latest_mtime)
    success_dt = _parse_refresh_time(str(last_success_at) if last_success_at else None)
    if prequery_like and latest_dt and success_dt and latest_dt > success_dt:
        reasons.append("wiki_markdown_newer_than_last_import")
    if upstream_actionable_count:
        blocked_reasons.append("pending_wiki_integration_before_lightrag_refresh")
    if upstream_review_count:
        blocked_reasons.append("pending_wiki_integration_needs_manual_review")
    blocked_by_pending_wiki_integration = bool(blocked_reasons)
    would_refresh_if_unblocked = bool(reasons)
    should_refresh = bool(would_refresh_if_unblocked and not blocked_by_pending_wiki_integration)
    if upstream_actionable_count:
        next_required_action = "wiki_integration"
    elif upstream_review_count:
        next_required_action = "manual_review"
    elif should_refresh:
        next_required_action = "lightrag_refresh"
    else:
        next_required_action = "none"
    return {
        "reason": normalized_reason,
        "should_refresh": should_refresh,
        "would_refresh_if_unblocked": would_refresh_if_unblocked,
        "blocked_by_pending_wiki_integration": blocked_by_pending_wiki_integration,
        "blocked_reasons": sorted(set(blocked_reasons)),
        "next_required_action": next_required_action,
        "reasons": sorted(set(reasons)),
        "pending_count": pending_count,
        "graph_ready_pending_count": pending_count,
        "raw_fast_pending_wiki_integration_count": upstream_pending_count,
        "raw_fast_actionable_wiki_integration_count": upstream_actionable_count,
        "raw_fast_review_wiki_integration_count": upstream_review_count,
        "total_not_graph_fresh_count": pending_count + upstream_pending_count,
        "pending_count_excludes_raw_fast": True,
        "threshold": effective_threshold,
        "dirty": bool(ledger.get("dirty")),
        "raw_clip_count": raw_count,
        "last_successful_raw_count": last_success_count,
        "last_successful_refresh_at": last_success_at,
        "latest_wiki_markdown_mtime": latest_mtime,
        "ledger_path": str(pending_lightrag_refresh_ledger_path(state_dir)),
        "pending": pending,
        "upstream_wiki_integration": upstream_wiki_integration,
        "import_report": import_summary,
    }


def clear_lightrag_refresh_pending_after_success(
    root: Path,
    state_dir: Path,
    import_report_path: Path | None = None,
    reason: str = "refresh",
) -> dict[str, Any]:
    ledger = load_lightrag_refresh_ledger(state_dir)
    pending = list(ledger.get("pending") or [])
    report_path = import_report_path or (state_dir / "custom_kg_import_report.json")
    import_summary = lightrag_refresh_import_summary(report_path)
    cleared_at = now_stamp()
    ledger["last_successful_refresh_at"] = import_summary.get("finished_at") or cleared_at
    ledger["last_successful_raw_count"] = len(raw_clip_files(root))
    ledger["last_successful_import_payload"] = import_summary.get("payload") or {}
    ledger["last_successful_import_report"] = import_summary.get("report_path")
    ledger["last_refresh_reason"] = reason
    ledger["last_cleared_pending"] = pending
    ledger["last_cleared_pending_count"] = len(pending)
    ledger["pending"] = []
    ledger["dirty"] = False
    ledger["last_failed_refresh"] = None
    save_lightrag_refresh_ledger(state_dir, ledger)
    return {
        "cleared_count": len(pending),
        "last_successful_refresh_at": ledger["last_successful_refresh_at"],
        "last_successful_raw_count": ledger["last_successful_raw_count"],
        "last_successful_import_payload": ledger["last_successful_import_payload"],
        "ledger_path": str(pending_lightrag_refresh_ledger_path(state_dir)),
    }


def record_lightrag_refresh_failure(state_dir: Path, reason: str, log_path: str = "", message: str = "") -> dict[str, Any]:
    ledger = load_lightrag_refresh_ledger(state_dir)
    failure = {"at": now_stamp(), "reason": reason, "log_path": log_path, "message": message}
    ledger["last_failed_refresh"] = failure
    ledger["dirty"] = True
    save_lightrag_refresh_ledger(state_dir, ledger)
    return failure


def init_manifest_db(state_dir: Path) -> Path:
    ensure_state_dirs(state_dir)
    db = state_dir / "lightrag_sync.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS docs (
              canonical_id TEXT PRIMARY KEY,
              rel_path TEXT NOT NULL,
              doc_type TEXT NOT NULL,
              sha256 TEXT NOT NULL,
              title TEXT,
              updated TEXT,
              lightrag_track_id TEXT,
              lightrag_doc_status TEXT,
              last_synced_at TEXT,
              deleted INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS sync_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              canonical_id TEXT NOT NULL,
              event_type TEXT NOT NULL,
              old_sha256 TEXT,
              new_sha256 TEXT,
              track_id TEXT,
              status TEXT,
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS query_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              query TEXT NOT NULL,
              mode TEXT NOT NULL,
              rewritten_queries TEXT,
              evidence_pack_path TEXT,
              created_at TEXT NOT NULL
            );
            """
        )
    return db


def load_lightrag_api_key(workdir: Path = DEFAULT_WORKDIR) -> str:
    if os.environ.get("LIGHTRAG_API_KEY"):
        return os.environ["LIGHTRAG_API_KEY"]
    env_path = workdir / ".env"
    if not env_path.exists():
        return ""
    for line in read_text(env_path).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == "LIGHTRAG_API_KEY":
            return value.strip().strip('"').strip("'")
    return ""


def http_json(method: str, url: str, payload: Any | None = None, api_key: str = "", timeout: int = 60) -> Any:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code} {url}: {body}") from exc
    if not raw:
        return None
    return json.loads(raw)


def health(server: str, api_key: str) -> dict[str, Any]:
    return http_json("GET", server.rstrip("/") + "/health", api_key=api_key, timeout=20)


def insert_texts(server: str, api_key: str, texts: list[str], file_sources: list[str]) -> dict[str, Any]:
    payload = {"texts": texts, "file_sources": file_sources}
    return http_json("POST", server.rstrip("/") + "/documents/texts", payload, api_key=api_key, timeout=120)


def track_status(server: str, api_key: str, track_id: str) -> dict[str, Any]:
    quoted = urllib.parse.quote(track_id, safe="")
    return http_json("GET", server.rstrip("/") + f"/documents/track_status/{quoted}", api_key=api_key, timeout=30)


def wait_for_track(server: str, api_key: str, track_id: str, timeout_s: int = 900, poll_s: float = 5.0) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last: dict[str, Any] = {"track_id": track_id, "documents": [], "status_summary": {}}
    while time.time() < deadline:
        last = track_status(server, api_key, track_id)
        docs = last.get("documents") or []
        if docs and all(str(doc.get("status")) in TERMINAL_STATUSES for doc in docs):
            return last
        time.sleep(poll_s)
    return last


def manifest_rows(db: Path) -> dict[str, dict[str, Any]]:
    if not db.exists():
        return {}
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        return {row["canonical_id"]: dict(row) for row in conn.execute("SELECT * FROM docs")}


def upsert_doc_event(
    db: Path,
    canonical_id: str,
    rel_path: str,
    doc_type: str,
    title: str,
    updated: str,
    old_sha: str | None,
    new_sha: str,
    track_id: str | None,
    status: str,
) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            INSERT INTO docs(canonical_id, rel_path, doc_type, sha256, title, updated, lightrag_track_id, lightrag_doc_status, last_synced_at, deleted)
            VALUES(?,?,?,?,?,?,?,?,?,0)
            ON CONFLICT(canonical_id) DO UPDATE SET
              rel_path=excluded.rel_path,
              doc_type=excluded.doc_type,
              sha256=excluded.sha256,
              title=excluded.title,
              updated=excluded.updated,
              lightrag_track_id=excluded.lightrag_track_id,
              lightrag_doc_status=excluded.lightrag_doc_status,
              last_synced_at=excluded.last_synced_at,
              deleted=0
            """,
            (canonical_id, rel_path, doc_type, new_sha, title, updated, track_id, status, now_stamp()),
        )
        conn.execute(
            """
            INSERT INTO sync_events(canonical_id, event_type, old_sha256, new_sha256, track_id, status, created_at)
            VALUES(?,?,?,?,?,?,?)
            """,
            (canonical_id, "sync", old_sha, new_sha, track_id, status, now_stamp()),
        )


def sync_docs_to_lightrag(
    docs: list[WikiDoc],
    state_dir: Path,
    server: str,
    workdir: Path,
    full: bool = False,
    force: bool = False,
    batch_size: int = 8,
    wait: bool = True,
    timeout_s: int = 900,
) -> dict[str, Any]:
    ensure_state_dirs(state_dir)
    db = init_manifest_db(state_dir)
    rows = manifest_rows(db)
    sync_shas = {doc.canonical_id: sha256_text(make_ingest_text(doc)) for doc in docs}
    changed: list[WikiDoc] = []
    for doc in docs:
        old = rows.get(doc.canonical_id)
        sync_sha = sync_shas[doc.canonical_id]
        if full or force or old is None or old.get("sha256") != sync_sha:
            changed.append(doc)
    result = {"seen": len(docs), "changed": len(changed), "batches": [], "errors": []}
    if not changed:
        return result
    api_key = load_lightrag_api_key(workdir)
    health(server, api_key)
    for start in range(0, len(changed), batch_size):
        batch = changed[start : start + batch_size]
        texts = [make_ingest_text(doc) for doc in batch]
        sources = [doc.rel_path for doc in batch]
        response = insert_texts(server, api_key, texts, sources)
        track_id = response.get("track_id")
        status = response.get("status", "submitted")
        final = None
        if track_id and wait:
            final = wait_for_track(server, api_key, track_id, timeout_s=timeout_s)
            statuses = {str(item.get("status")) for item in final.get("documents", [])}
            if statuses and not statuses <= SUCCESS_STATUSES:
                result["errors"].append({"track_id": track_id, "statuses": sorted(statuses)})
            status = ",".join(sorted(statuses)) if statuses else status
        for doc in batch:
            old_sha = rows.get(doc.canonical_id, {}).get("sha256")
            upsert_doc_event(
                db,
                doc.canonical_id,
                doc.rel_path,
                doc.doc_type,
                doc.title,
                display_scalar(doc.frontmatter.get("updated")),
                old_sha,
                sync_shas[doc.canonical_id],
                track_id,
                status,
            )
        result["batches"].append({"count": len(batch), "track_id": track_id, "status": status})
    write_manifest_jsonl(state_dir, manifest_rows(db).values())
    return result


def write_manifest_jsonl(state_dir: Path, rows: Iterable[dict[str, Any]]) -> Path:
    path = state_dir / "lightrag_manifest.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    return path


def wiki_root_machine_pollution(root: Path) -> list[Path]:
    root = root.resolve()
    polluted: list[Path] = []
    for name in POLLUTION_DIRECT_NAMES:
        p = root / name
        if p.exists():
            polluted.append(Path(name))
    for name in POLLUTION_RECURSIVE_NAMES:
        for p in root.rglob(name):
            rel = p.relative_to(root)
            if rel not in polluted:
                polluted.append(rel)
    return sorted(polluted, key=lambda p: p.as_posix())


def compiled_pages(root: Path) -> list[Path]:
    paths: list[Path] = []
    for dirname in COMPILED_DIR_TYPES:
        d = root / dirname
        if d.exists():
            paths.extend(sorted(d.glob("*.md")))
    return paths


def indexed_markdown_pages(root: Path) -> list[Path]:
    paths = compiled_pages(root)
    for rel in META_FILES:
        p = root / rel
        if p.exists():
            paths.append(p)
    return sorted(paths, key=lambda p: p.relative_to(root).as_posix())


def index_stats(root: Path) -> tuple[int | None, int]:
    path = root / "index.md"
    if not path.exists():
        return None, 0
    text = read_text(path)
    m = re.search(r"Total pages:\s*(\d+)", text)
    total = int(m.group(1)) if m else None
    wikilinks = len(re.findall(r"^-\s+\[\[[^\]]+\]\]", text, flags=re.M))
    return total, wikilinks


def _lexical_norm(path: Path | str) -> str:
    """Normalize a path lexically without touching the filesystem."""

    return os.path.normpath(os.path.abspath(os.fspath(path)))


def _is_lexically_under(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([root, path]) == root
    except ValueError:
        return False


@functools.lru_cache(maxsize=20000)
def _resolve_source_cached(page_dir: str, source: str, root_str: str | None) -> str | None:
    source = source.strip()
    if not source or source.startswith("http://") or source.startswith("https://"):
        return None

    root_norm = _lexical_norm(root_str) if root_str else None
    source_path = Path(source)
    if source_path.is_absolute():
        candidate = _lexical_norm(source_path)
    else:
        first_segment = source.split("/", 1)[0]
        base = Path(root_norm) if root_norm is not None and first_segment in WIKI_SOURCE_ROOT_PREFIXES else Path(page_dir)
        candidate = _lexical_norm(base / source)

    if root_norm is not None and not _is_lexically_under(candidate, root_norm):
        return None
    return candidate


def resolve_source(page: Path, source: str, root: Path | None = None) -> Path | None:
    root_str = os.fspath(root) if root is not None else None
    resolved = _resolve_source_cached(os.fspath(page.parent), source.strip(), root_str)
    return Path(resolved) if resolved else None


def is_structured_raw_note(text: str) -> bool:
    meta, _body = parse_frontmatter(text)
    note_type = display_scalar(meta.get("type")).strip().strip('"\'').lower()
    domain = display_scalar(meta.get("domain")).strip().strip('"\'').lower()
    return note_type in {"raw-note", "paper-note"} or domain == "paper"


def structured_heading_warnings(path: Path, text: str) -> list[str]:
    if not is_structured_raw_note(text):
        return []
    sections = markdown_sections(text)
    titles = [title for title, _body in sections]
    present = {
        "## 一句话总结": any(summary_heading_matches(title) for title in titles),
        "## 论文摘要": False,
        "## Motivation": False,
        "## Methodology": False,
        "## 关键实验结果": False,
    }
    kind_to_heading = {
        "abstract": "## 论文摘要",
        "motivation": "## Motivation",
        "methodology": "## Methodology",
        "results": "## 关键实验结果",
    }
    for title in titles:
        spec = raw_section_spec_for_heading(title)
        if spec and spec.get("kind") in kind_to_heading:
            present[kind_to_heading[spec["kind"]]] = True
    warnings = []
    for heading, has_heading in present.items():
        if not has_heading:
            warnings.append(f"{path.as_posix()} missing heading prefix {heading}")
    return warnings


def audit_raw_note_section_contracts(root: Path, include_legacy: bool = True, issue_limit: int | None = None) -> dict[str, Any]:
    root = root.resolve()
    raw_total = 0
    structured_total = 0
    issue_rows: list[dict[str, Any]] = []
    heading_occurrences: dict[str, int] = {}
    doc_counts_by_kind: dict[str, int] = {}

    def add_issue(issue: dict[str, Any]) -> None:
        issue_rows.append(issue)

    for doc in collect_source_docs(root):
        if doc.doc_type != "raw_note":
            continue
        raw_total += 1
        structured = is_structured_raw_note(doc.text)
        if structured:
            structured_total += 1
        if not structured and not include_legacy:
            continue
        titles_by_kind: dict[str, list[str]] = {kind: [] for kind in RAW_NOTE_CONTRACT_SECTION_KINDS}
        for title, body in markdown_sections(doc.body):
            kinds: list[str] = []
            if summary_heading_matches(title):
                kinds.append("summary")
            matched_specs = raw_section_specs_for_heading(title)
            kinds.extend(str(spec["kind"]) for spec in matched_specs)
            kinds = list(dict.fromkeys(kinds))
            if kinds:
                for kind in kinds:
                    titles_by_kind.setdefault(kind, []).append(title)
                    heading_occurrences[kind] = heading_occurrences.get(kind, 0) + 1
                if len(kinds) > 1:
                    add_issue(
                        {
                            "type": "combined_section_heading",
                            "severity": "warning",
                            "path": doc.rel_path,
                            "title": title,
                            "section_kinds": kinds,
                            "message": "One heading maps to multiple retrieval sections; split it or keep the new multi-indexing behavior in mind.",
                        }
                    )
                continue
            suggestions = likely_raw_section_kinds_for_unmatched_heading(title)
            if suggestions:
                add_issue(
                    {
                        "type": "near_miss_heading",
                        "severity": "info",
                        "path": doc.rel_path,
                        "title": title,
                        "suggested_section_kinds": suggestions,
                        "message": "Heading looks semantically section-like but is not part of the section-level retrieval contract.",
                    }
                )
        for kind, titles in titles_by_kind.items():
            if titles:
                doc_counts_by_kind[kind] = doc_counts_by_kind.get(kind, 0) + 1
            if len(titles) > 1:
                add_issue(
                    {
                        "type": "duplicate_section_kind",
                        "severity": "warning",
                        "path": doc.rel_path,
                        "section_kind": kind,
                        "titles": titles,
                        "message": "Multiple headings map to the same retrieval section; extraction keeps the first section per kind.",
                    }
                )
        if structured:
            for kind in RAW_NOTE_CONTRACT_REQUIRED_KINDS:
                if not titles_by_kind.get(kind):
                    add_issue(
                        {
                            "type": "missing_section",
                            "severity": "warning",
                            "path": doc.rel_path,
                            "section_kind": kind,
                            "message": "Structured raw paper note is missing a canonical retrieval section.",
                        }
                    )
    issues_by_type: dict[str, int] = {}
    issues_by_severity: dict[str, int] = {}
    for issue in issue_rows:
        issue_type = str(issue.get("type", "unknown"))
        severity = str(issue.get("severity", "warning"))
        issues_by_type[issue_type] = issues_by_type.get(issue_type, 0) + 1
        issues_by_severity[severity] = issues_by_severity.get(severity, 0) + 1
    returned_issues = issue_rows[:issue_limit] if issue_limit is not None else issue_rows
    return {
        "generated_at": now_stamp(),
        "root": root.as_posix(),
        "raw_notes": raw_total,
        "structured_raw_notes": structured_total,
        "include_legacy": include_legacy,
        "contract_section_kinds": RAW_NOTE_CONTRACT_SECTION_KINDS,
        "required_structured_section_kinds": RAW_NOTE_CONTRACT_REQUIRED_KINDS,
        "heading_occurrences_by_kind": dict(sorted(heading_occurrences.items())),
        "docs_with_section_by_kind": dict(sorted(doc_counts_by_kind.items())),
        "issue_count": len(issue_rows),
        "returned_issue_count": len(returned_issues),
        "issues_by_type": dict(sorted(issues_by_type.items())),
        "issues_by_severity": dict(sorted(issues_by_severity.items())),
        "issues": returned_issues,
    }


from wiki_lightrag_validation import secret_hits, validate_wiki



def release_process_memory() -> bool:
    """Best-effort return of freed Python/glibc arenas to the OS.

    This helper is a memory-pressure hint for large refresh phases. It must not
    be required for correctness: unsupported platforms or allocator failures
    simply return False after running normal garbage collection.
    """
    import gc

    gc.collect()
    if os.name != "posix":
        return False
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        return bool(libc.malloc_trim(0))
    except Exception:
        return False


def jsonl_write(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def jsonl_read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_text_if_changed(path: Path, content: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def markdown_sections(markdown: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_title: str | None = None
    current_level = 0
    current_lines: list[str] = []
    for line in markdown.splitlines():
        match = re.match(r"^(#{2,6})\s+(.+?)\s*#*\s*$", line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            if current_title is None:
                current_title = title
                current_level = level
                current_lines = []
                continue
            if level <= current_level:
                sections.append((current_title, "\n".join(current_lines).strip()))
                current_title = title
                current_level = level
                current_lines = []
                continue
        if current_title is not None:
            current_lines.append(line)
    if current_title is not None:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return sections


def section_text(markdown: str, heading_keywords: list[str]) -> str:
    for title, text in markdown_sections(markdown):
        key = title.lower()
        if any(keyword.lower() in key for keyword in heading_keywords):
            return text
    return ""


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


def extract_raw_note_sections(doc: WikiDoc) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    seen: set[str] = set()
    for title, text in markdown_sections(doc.body):
        specs = raw_section_specs_for_heading(title)
        if not specs or not text.strip():
            continue
        for spec in specs:
            kind = str(spec["kind"])
            if kind in seen:
                continue
            seen.add(kind)
            section_id = f"raw_section:{doc.path.stem}:{kind}"
            sections.append(
                {
                    "section_id": section_id,
                    "source_id": doc.canonical_id,
                    "source_path": doc.rel_path,
                    "paper_title": doc.title,
                    "section_kind": kind,
                    "section_title": title.strip() or spec["canonical_title"],
                    "canonical_section_title": spec["canonical_title"],
                    "content": text.strip(),
                    "created_by": "extract_raw_sections.py",
                }
            )
    return sections


def first_sentences(text: str, max_chars: int = 600) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned[:max_chars].rstrip()


def bullet_items(text: str, limit: int = 5) -> list[str]:
    items = []
    for line in text.splitlines():
        s = re.sub(r"^[-*]\s+", "", line.strip())
        if s and len(s) > 8:
            items.append(s[:240])
        if len(items) >= limit:
            break
    if not items and text.strip():
        items.append(first_sentences(text, 240))
    return items


def method_type_for(doc: WikiDoc) -> str:
    hay = " ".join([doc.title, display_scalar(doc.frontmatter.get("tags")), display_scalar(doc.frontmatter.get("topic_hints")), doc.body[:2000]]).lower()
    if any(k in hay for k in ["retrieval", "rag", "search"]):
        return "retrieval_or_memory_design"
    if any(k in hay for k in ["evaluation", "benchmark", "judge"]):
        return "evaluation_protocol"
    if any(k in hay for k in ["training", "rl", "reward", "distillation"]):
        return "training_signal_construction"
    if any(k in hay for k in ["agent", "workflow", "harness"]):
        return "agent_workflow_design"
    return "methodological_pattern"


def extract_method_atoms(root: Path, state_dir: Path) -> dict[str, Any]:
    ensure_state_dirs(state_dir)
    rows = []
    docs_dir = state_dir / "method_atom_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for stale in docs_dir.glob("*.md"):
        stale.unlink()
    for doc in collect_source_docs(root):
        if doc.doc_type != "raw_note":
            continue
        methodology = section_text(doc.body, ["methodology", "方法"])
        if not methodology or len(methodology) < 80:
            continue
        limitations = section_text(doc.body, ["局限", "limitation", "failure"])
        future = section_text(doc.body, ["未来", "启发", "future", "question"])
        atom_id = f"method:{slugify(doc.path.stem, 100)}:001"
        atom = {
            "atom_id": atom_id,
            "source_path": doc.rel_path,
            "paper_title": doc.title,
            "method_name": doc.title,
            "method_type": method_type_for(doc),
            "problem_form": first_sentences(section_text(doc.body, ["motivation", "问题", "background"]) or doc.body, 300),
            "core_mechanism": first_sentences(methodology, 700),
            "transferable_unit": method_type_for(doc).replace("_", " "),
            "assumptions": [],
            "failure_modes": bullet_items(limitations, 3),
            "possible_transfer_targets": bullet_items(future, 3),
            "evidence_headings": ["## Methodology"],
            "confidence": 0.62,
            "created_by": "extract_method_atoms.py",
        }
        rows.append(atom)
        md = method_atom_markdown(atom)
        (docs_dir / generated_doc_filename(atom_id, 120)).write_text(md, encoding="utf-8")
    jsonl = state_dir / "method_atoms.jsonl"
    count = jsonl_write(jsonl, rows)
    return {"method_atoms": count, "jsonl": str(jsonl), "docs_dir": str(docs_dir)}


def method_atom_markdown(atom: dict[str, Any]) -> str:
    lines = [
        "[LLM_WIKI_METHOD_ATOM]",
        f"atom_id: {atom['atom_id']}",
        f"source_path: {atom['source_path']}",
        f"paper_title: {atom['paper_title']}",
        f"method_name: {atom['method_name']}",
        f"method_type: {atom['method_type']}",
        f"transferable_unit: {atom['transferable_unit']}",
        f"confidence: {atom['confidence']}",
        "[/LLM_WIKI_METHOD_ATOM]",
        "",
        f"# MethodAtom: {atom['method_name']}",
        "",
        "## Problem Form",
        atom.get("problem_form") or "Not specified.",
        "",
        "## Core Mechanism",
        atom.get("core_mechanism") or "Not specified.",
        "",
        "## Transferable Unit",
        atom.get("transferable_unit") or "Not specified.",
        "",
        "## Failure Modes",
    ]
    for item in atom.get("failure_modes") or ["Not specified."]:
        lines.append(f"- {item}")
    lines.extend(["", "## Possible Transfer Targets"])
    for item in atom.get("possible_transfer_targets") or ["Not specified."]:
        lines.append(f"- {item}")
    lines.extend(["", "## Evidence Source", atom["source_path"], ""])
    return "\n".join(lines)


def raw_section_markdown(section: dict[str, Any]) -> str:
    return "\n".join(
        [
            "[LLM_WIKI_RAW_SECTION]",
            f"section_id: {section['section_id']}",
            f"source_id: {section['source_id']}",
            f"source_path: {section['source_path']}",
            f"paper_title: {section['paper_title']}",
            f"section_kind: {section['section_kind']}",
            f"section_title: {section['section_title']}",
            f"canonical_section_title: {section['canonical_section_title']}",
            "[/LLM_WIKI_RAW_SECTION]",
            "",
            f"# RawSection: {section['paper_title']} / {section['section_title']}",
            "",
            "## Section Content",
            section.get("content") or "Not specified.",
            "",
            "## Evidence Source",
            section["source_path"],
            "",
        ]
    )


def extract_raw_sections(root: Path, state_dir: Path) -> dict[str, Any]:
    ensure_state_dirs(state_dir)
    rows: list[dict[str, Any]] = []
    docs_dir = state_dir / "raw_section_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for stale in docs_dir.glob("*.md"):
        stale.unlink()
    for doc in collect_source_docs(root):
        if doc.doc_type != "raw_note":
            continue
        for section in extract_raw_note_sections(doc):
            rows.append(section)
            (docs_dir / generated_doc_filename(section["section_id"], 120)).write_text(raw_section_markdown(section), encoding="utf-8")
    jsonl = state_dir / "raw_sections.jsonl"
    count = jsonl_write(jsonl, rows)
    by_kind: dict[str, int] = {}
    for row in rows:
        kind = str(row.get("section_kind", ""))
        by_kind[kind] = by_kind.get(kind, 0) + 1
    return {"raw_sections": count, "by_kind": by_kind, "jsonl": str(jsonl), "docs_dir": str(docs_dir)}


def section_similarity_embedding_text(section: dict[str, Any], max_content_chars: int = 6000) -> str:
    """Build clean text for section-to-section embedding without sidecar boilerplate."""
    content = re.sub(r"\s+", " ", str(section.get("content", "")).strip())
    if max_content_chars > 0 and len(content) > max_content_chars:
        content = content[:max_content_chars].rstrip()
    lines = [
        f"Title: {section.get('paper_title', '')}",
        f"Section kind: {section.get('section_kind', '')}",
        f"Section title: {section.get('section_title', '')}",
        f"Source path: {section.get('source_path', '')}",
        "",
        content,
    ]
    return "\n".join(line for line in lines if line is not None).strip() + "\n"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _ordered_section_pair_key(src_id: str, tgt_id: str) -> str:
    digest = sha256_text("\t".join(sorted([src_id, tgt_id])))[:12]
    return f"semantic_section_neighbor:{digest}"


def _section_rank_lists_scalar(
    sections: list[dict[str, Any]],
    embeddings: dict[str, list[float]],
    src_kind: str,
    tgt_kind: str,
    k: int,
    min_cosine: float,
) -> dict[tuple[str, str], dict[str, Any]]:
    sources = [section for section in sections if section.get("section_kind") == src_kind and section.get("section_id") in embeddings]
    targets = [section for section in sections if section.get("section_kind") == tgt_kind and section.get("section_id") in embeddings]
    directed: dict[tuple[str, str], dict[str, Any]] = {}
    for src in sources:
        src_id = str(src["section_id"])
        scored: list[tuple[float, dict[str, Any]]] = []
        for tgt in targets:
            tgt_id = str(tgt["section_id"])
            if src_id == tgt_id:
                continue
            if src.get("source_id") and src.get("source_id") == tgt.get("source_id"):
                continue
            score = cosine_similarity(embeddings[src_id], embeddings[tgt_id])
            if score >= min_cosine:
                scored.append((score, tgt))
        scored.sort(key=lambda item: (-item[0], str(item[1].get("section_id", ""))))
        for rank, (score, tgt) in enumerate(scored[: max(k, 0)], start=1):
            directed[(src_id, str(tgt["section_id"]))] = {"cosine": score, "rank": rank, "src": src, "tgt": tgt}
    return directed


def _section_rank_lists_vectorized(
    sections: list[dict[str, Any]],
    embeddings: dict[str, list[float]],
    src_kind: str,
    tgt_kind: str,
    k: int,
    min_cosine: float,
) -> dict[tuple[str, str], dict[str, Any]] | None:
    if k <= 0:
        return {}
    sources = [section for section in sections if section.get("section_kind") == src_kind and section.get("section_id") in embeddings]
    targets = [section for section in sections if section.get("section_kind") == tgt_kind and section.get("section_id") in embeddings]
    if not sources or not targets:
        return {}
    try:
        import numpy as np
    except Exception:
        return None

    try:
        def matrix_for(items: list[dict[str, Any]]):
            vectors: list[list[float]] = []
            dim: int | None = None
            for section in items:
                vector = embeddings.get(str(section["section_id"]))
                if not isinstance(vector, list) or not vector:
                    return None
                if dim is None:
                    dim = len(vector)
                elif len(vector) != dim:
                    return None
                vectors.append(vector)
            matrix = np.asarray(vectors, dtype=np.float64)
            if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
                return None
            return matrix

        source_matrix = matrix_for(sources)
        target_matrix = matrix_for(targets)
        if source_matrix is None or target_matrix is None or source_matrix.shape[1] != target_matrix.shape[1]:
            return None

        source_norms = np.linalg.norm(source_matrix, axis=1)
        target_norms = np.linalg.norm(target_matrix, axis=1)
        source_normalized = np.divide(
            source_matrix,
            source_norms[:, None],
            out=np.zeros_like(source_matrix, dtype=np.float64),
            where=source_norms[:, None] != 0,
        )
        target_normalized = np.divide(
            target_matrix,
            target_norms[:, None],
            out=np.zeros_like(target_matrix, dtype=np.float64),
            where=target_norms[:, None] != 0,
        )

        source_ids = [str(section["section_id"]) for section in sources]
        target_ids = [str(section["section_id"]) for section in targets]
        target_id_array = np.asarray(target_ids, dtype=object)
        target_source_ids = np.asarray([str(section.get("source_id") or "") for section in targets], dtype=object)
        score_eps = 1e-9
        block_size = 256
        threshold_floor = min_cosine - score_eps
        directed: dict[tuple[str, str], dict[str, Any]] = {}

        for block_start in range(0, len(sources), block_size):
            score_block = source_normalized[block_start : block_start + block_size] @ target_normalized.T
            for local_index, row in enumerate(score_block):
                source_index = block_start + local_index
                src = sources[source_index]
                src_id = source_ids[source_index]
                valid_mask = target_id_array != src_id
                src_source_id = str(src.get("source_id") or "")
                if src_source_id:
                    valid_mask = valid_mask & (target_source_ids != src_source_id)
                candidate_mask = valid_mask & (row >= threshold_floor)
                if not bool(candidate_mask.any()):
                    continue

                candidate_indices = np.flatnonzero(candidate_mask)
                if len(candidate_indices) > k:
                    candidate_scores = row[candidate_indices]
                    kth_index = len(candidate_scores) - k
                    kth_score = float(np.partition(candidate_scores, kth_index)[kth_index])
                    candidate_indices = candidate_indices[candidate_scores >= kth_score - score_eps]

                scored: list[tuple[float, dict[str, Any]]] = []
                for target_index in candidate_indices.tolist():
                    tgt = targets[int(target_index)]
                    tgt_id = target_ids[int(target_index)]
                    score = cosine_similarity(embeddings[src_id], embeddings[tgt_id])
                    if score >= min_cosine:
                        scored.append((score, tgt))
                scored.sort(key=lambda item: (-item[0], str(item[1].get("section_id", ""))))
                for rank, (score, tgt) in enumerate(scored[: max(k, 0)], start=1):
                    directed[(src_id, str(tgt["section_id"]))] = {"cosine": score, "rank": rank, "src": src, "tgt": tgt}
        return directed
    except Exception:
        return None


def _section_rank_lists(
    sections: list[dict[str, Any]],
    embeddings: dict[str, list[float]],
    src_kind: str,
    tgt_kind: str,
    k: int,
    min_cosine: float,
) -> dict[tuple[str, str], dict[str, Any]]:
    vectorized = _section_rank_lists_vectorized(sections, embeddings, src_kind, tgt_kind, k, min_cosine)
    if vectorized is not None:
        return vectorized
    return _section_rank_lists_scalar(sections, embeddings, src_kind, tgt_kind, k, min_cosine)


def build_section_similarity_edges(
    sections: list[dict[str, Any]],
    embeddings: dict[str, list[float]],
    same_kind_k: int = 5,
    cross_kind_k: int = 3,
    same_kind_min_cosine: float = 0.72,
    cross_kind_min_cosine: float = 0.76,
    cross_kind_pairs: list[tuple[str, str]] | None = None,
    mutual: bool = True,
    embedding_model: str = "unknown",
    embedding_dim: int | None = None,
) -> list[dict[str, Any]]:
    """Build sparse semantic-neighbor candidate edges between raw-section embeddings."""
    sections_by_id = {str(section.get("section_id")): section for section in sections if section.get("section_id") in embeddings}
    section_kinds = sorted({str(section.get("section_kind", "")) for section in sections_by_id.values() if section.get("section_kind")})
    cross_kind_pairs = cross_kind_pairs or []
    directed: dict[tuple[str, str], dict[str, Any]] = {}
    for kind in section_kinds:
        directed.update(_section_rank_lists(list(sections_by_id.values()), embeddings, kind, kind, same_kind_k, same_kind_min_cosine))
    for left, right in cross_kind_pairs:
        directed.update(_section_rank_lists(list(sections_by_id.values()), embeddings, left, right, cross_kind_k, cross_kind_min_cosine))
        directed.update(_section_rank_lists(list(sections_by_id.values()), embeddings, right, left, cross_kind_k, cross_kind_min_cosine))

    edges: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    def add_edge(src_id: str, tgt_id: str, orientation: tuple[str, str] | None = None) -> None:
        if (src_id, tgt_id) not in directed:
            return
        if mutual and (tgt_id, src_id) not in directed:
            return
        src = sections_by_id[src_id]
        tgt = sections_by_id[tgt_id]
        unordered = tuple(sorted([src_id, tgt_id]))
        if unordered in seen_pairs:
            return
        seen_pairs.add(unordered)
        forward = directed[(src_id, tgt_id)]
        reverse = directed.get((tgt_id, src_id), {})
        cosine = float(forward["cosine"])
        edge = {
            "edge_id": _ordered_section_pair_key(src_id, tgt_id),
            "type": "SEMANTIC_SECTION_NEIGHBOR",
            "src_id": src_id,
            "tgt_id": tgt_id,
            "source_section_kind": src.get("section_kind"),
            "target_section_kind": tgt.get("section_kind"),
            "source_path": src.get("source_path"),
            "target_path": tgt.get("source_path"),
            "source_title": src.get("paper_title"),
            "target_title": tgt.get("paper_title"),
            "cosine": round(cosine, 6),
            "source_rank": forward.get("rank"),
            "target_rank": reverse.get("rank"),
            "mutual_knn": bool((tgt_id, src_id) in directed),
            "embedding_model": embedding_model,
            "embedding_dim": embedding_dim or (len(embeddings[src_id]) if src_id in embeddings else None),
            "source_text_hash": sha256_text(section_similarity_embedding_text(src)),
            "target_text_hash": sha256_text(section_similarity_embedding_text(tgt)),
            "created_by": "build_section_similarity_graph.py",
        }
        if orientation:
            edge["pair_kind"] = f"{orientation[0]}:{orientation[1]}"
        edges.append(edge)

    for kind in section_kinds:
        same_ids = sorted(str(section["section_id"]) for section in sections_by_id.values() if section.get("section_kind") == kind)
        for i, src_id in enumerate(same_ids):
            for tgt_id in same_ids[i + 1 :]:
                if (src_id, tgt_id) in directed:
                    add_edge(src_id, tgt_id, (kind, kind))
                elif (tgt_id, src_id) in directed:
                    add_edge(tgt_id, src_id, (kind, kind))
    for left, right in cross_kind_pairs:
        left_ids = sorted(str(section["section_id"]) for section in sections_by_id.values() if section.get("section_kind") == left)
        right_ids = sorted(str(section["section_id"]) for section in sections_by_id.values() if section.get("section_kind") == right)
        for src_id in left_ids:
            for tgt_id in right_ids:
                add_edge(src_id, tgt_id, (left, right))
    edges.sort(key=lambda edge: (str(edge.get("pair_kind", "")), -float(edge.get("cosine", 0)), str(edge.get("src_id")), str(edge.get("tgt_id"))))
    return edges


def section_similarity_edge_to_custom_kg_relationship(edge: dict[str, Any]) -> dict[str, Any]:
    cosine = float(edge.get("cosine", 0.0))
    src = str(edge.get("src_id", ""))
    tgt = str(edge.get("tgt_id", ""))
    source_kind = edge.get("source_section_kind", "section")
    target_kind = edge.get("target_section_kind", "section")
    description = (
        f"{src} SEMANTIC_SECTION_NEIGHBOR {tgt}; "
        f"source_section_kind={source_kind}; target_section_kind={target_kind}; "
        f"cosine={cosine:.6f}; mutual_knn={edge.get('mutual_knn')}; "
        f"embedding_model={edge.get('embedding_model', 'unknown')}."
    )
    return {
        "src_id": src,
        "tgt_id": tgt,
        "description": description,
        "keywords": "SEMANTIC_SECTION_NEIGHBOR",
        "source_id": src,
        "weight": cosine,
        "file_path": str(edge.get("source_path") or edge.get("target_path") or "section_similarity_edges.jsonl"),
    }


def _small_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    values_sorted = sorted(values)
    return {
        "min": round(values_sorted[0], 6),
        "mean": round(sum(values_sorted) / len(values_sorted), 6),
        "max": round(values_sorted[-1], 6),
    }


def section_similarity_report_summary(sections: list[dict[str, Any]], edges: list[dict[str, Any]], top_hubs: int = 20) -> dict[str, Any]:
    section_count_by_kind: dict[str, int] = {}
    for section in sections:
        kind = str(section.get("section_kind", "unknown"))
        section_count_by_kind[kind] = section_count_by_kind.get(kind, 0) + 1
    edge_count_by_pair_kind: dict[str, int] = {}
    cosine_values_by_pair: dict[str, list[float]] = {}
    hub_degree: dict[str, int] = {}
    for edge in edges:
        pair_kind = str(edge.get("pair_kind") or f"{edge.get('source_section_kind', 'section')}:{edge.get('target_section_kind', 'section')}")
        edge_count_by_pair_kind[pair_kind] = edge_count_by_pair_kind.get(pair_kind, 0) + 1
        cosine_values_by_pair.setdefault(pair_kind, []).append(float(edge.get("cosine", 0.0)))
        for node in [str(edge.get("src_id", "")), str(edge.get("tgt_id", ""))]:
            if node:
                hub_degree[node] = hub_degree.get(node, 0) + 1
    hubs = [
        {"section_id": section_id, "degree": degree}
        for section_id, degree in sorted(hub_degree.items(), key=lambda item: (-item[1], item[0]))[:top_hubs]
    ]
    return {
        "section_count": len(sections),
        "section_count_by_kind": dict(sorted(section_count_by_kind.items())),
        "edge_count": len(edges),
        "edge_count_by_pair_kind": dict(sorted(edge_count_by_pair_kind.items())),
        "cosine_by_pair_kind": {pair: _small_stats(values) for pair, values in sorted(cosine_values_by_pair.items())},
        "top_hubs": hubs,
    }


def select_section_similarity_edges(candidates: list[dict[str, Any]], allowed_pair_kinds: set[str]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for edge in candidates:
        if str(edge.get("pair_kind", "")) not in allowed_pair_kinds:
            continue
        row = dict(edge)
        row["review_status"] = "phase2_selected"
        row["review_note"] = "Sparse mutual-kNN high-value section pair selected for custom_kg import. Semantic proximity only, not a factual relation."
        selected.append(row)
    selected.sort(key=lambda edge: (str(edge.get("pair_kind", "")), -float(edge.get("cosine", 0.0)), str(edge.get("src_id", "")), str(edge.get("tgt_id", ""))))
    return selected


def build_seed_edges(root: Path, state_dir: Path) -> dict[str, Any]:
    ensure_state_dirs(state_dir)
    docs = collect_source_docs(root)
    by_rel = {d.rel_path: d for d in docs}
    by_stem = {Path(d.rel_path).stem: d for d in docs}
    edges: list[dict[str, Any]] = []

    def add(src: str, tgt: str, typ: str, weight: float, evidence_path: str, field: str) -> None:
        edge_id = f"edge:{typ.lower()}:{len(edges)+1:06d}"
        edges.append(
            {
                "edge_id": edge_id,
                "src_id": src,
                "tgt_id": tgt,
                "type": typ,
                "weight": weight,
                "evidence_path": evidence_path,
                "evidence_field": field,
                "created_by": "build_seed_edges.py",
            }
        )

    for doc in docs:
        if not doc.doc_type.startswith("compiled_") and doc.doc_type != "meta_map":
            continue
        for link in find_wikilinks(doc.text):
            target = by_stem.get(link)
            if target:
                add(doc.canonical_id, target.canonical_id, "WIKILINKS_TO", 1.0, doc.rel_path, "body.wikilinks")
        for src in as_list(doc.frontmatter.get("sources")):
            s = display_scalar(src)
            if not s or s.startswith("http://") or s.startswith("https://"):
                continue
            resolved = resolve_source(doc.path, s, root)
            if resolved and resolved.exists():
                rel = resolved.relative_to(root).as_posix()
                target = by_rel.get(rel) or make_wiki_doc(root, resolved)
                add(doc.canonical_id, target.canonical_id, "SOURCED_BY", 1.0, doc.rel_path, "frontmatter.sources")
        for tag in as_list(doc.frontmatter.get("tags")):
            tag_s = display_scalar(tag)
            if tag_s:
                add(doc.canonical_id, f"tag:{tag_s}", "HAS_TAG", 0.7, doc.rel_path, "frontmatter.tags")
    jsonl = state_dir / "seed_edges.jsonl"
    count = jsonl_write(jsonl, edges)
    docs_dir = state_dir / "edge_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    desired_docs = {generated_doc_filename(edge["edge_id"], 120): edge_markdown(edge) for edge in edges}
    for stale in docs_dir.glob("*.md"):
        if stale.name not in desired_docs:
            stale.unlink()
    edge_docs_written = 0
    for filename, content in desired_docs.items():
        if write_text_if_changed(docs_dir / filename, content):
            edge_docs_written += 1
    return {"seed_edges": count, "jsonl": str(jsonl), "docs_dir": str(docs_dir), "edge_docs_total": len(desired_docs), "edge_docs_written": edge_docs_written}


def edge_markdown(edge: dict[str, Any]) -> str:
    return "\n".join(
        [
            "[LLM_WIKI_EDGE_DOC]",
            f"edge_id: {edge['edge_id']}",
            f"edge_type: {edge['type']}",
            f"source: {edge['src_id']}",
            f"target: {edge['tgt_id']}",
            f"evidence_path: {edge['evidence_path']}",
            f"evidence_field: {edge['evidence_field']}",
            f"confidence: {edge['weight']}",
            "[/LLM_WIKI_EDGE_DOC]",
            "",
            f"The llm-wiki document {edge['src_id']} has a high-confidence {edge['type']} relation to {edge['tgt_id']}.",
            f"Evidence: {edge['evidence_path']} ({edge['evidence_field']}).",
            "",
        ]
    )


def custom_kg_entity_type(node_id: str) -> str:
    if node_id.startswith("compiled:"):
        return "LLM_WIKI_PAGE"
    if node_id.startswith("raw_section:"):
        return "LLM_WIKI_RAW_SECTION"
    if node_id.startswith("raw_clip:") or node_id.startswith("raw:"):
        return "LLM_WIKI_RAW"
    if node_id.startswith("tag:"):
        return "LLM_WIKI_TAG"
    if node_id.startswith("method_atom:") or node_id.startswith("method:"):
        return "LLM_WIKI_METHOD_ATOM"
    return "LLM_WIKI_NODE"


def custom_kg_doc_description(doc: WikiDoc) -> str:
    parts = [doc.title, doc.rel_path, doc.doc_type]
    tags = display_scalar(doc.frontmatter.get("tags"))
    if tags:
        parts.append(f"tags={tags}")
    topic_hints = display_scalar(doc.frontmatter.get("topic_hints"))
    if topic_hints:
        parts.append(f"topic_hints={topic_hints}")
    updated = display_scalar(doc.frontmatter.get("updated"))
    if updated:
        parts.append(f"updated={updated}")
    return " | ".join(part for part in parts if part)[:900]


def build_custom_kg_payload(
    root: Path,
    state_dir: Path,
    limit_docs: int | None = None,
    limit_edges: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a deterministic LightRAG custom_kg payload without touching the wiki root."""
    docs = collect_source_docs(root)
    if limit_docs is not None:
        docs = docs[:limit_docs]
    doc_ids = {doc.canonical_id for doc in docs}
    chunks: list[dict[str, Any]] = []
    entities: dict[str, dict[str, Any]] = {}
    relationships: list[dict[str, Any]] = []
    for doc in docs:
        chunks.append(
            {
                "content": make_ingest_text(doc),
                "source_id": doc.canonical_id,
                "file_path": doc.rel_path,
                "chunk_order_index": 0,
            }
        )
        entities[doc.canonical_id] = {
            "entity_name": doc.canonical_id,
            "entity_type": custom_kg_entity_type(doc.canonical_id),
            "description": custom_kg_doc_description(doc),
            "source_id": doc.canonical_id,
            "file_path": doc.rel_path,
        }

    fallback_source = docs[0].canonical_id if docs else "UNKNOWN"
    for method_doc in generated_docs_from_state(state_dir, kind="method_atom"):
        chunks.append(
            {
                "content": method_doc.text,
                "source_id": method_doc.canonical_id,
                "file_path": method_doc.rel_path,
                "chunk_order_index": 0,
            }
        )
        doc_ids.add(method_doc.canonical_id)
        entities[method_doc.canonical_id] = {
            "entity_name": method_doc.canonical_id,
            "entity_type": custom_kg_entity_type(method_doc.canonical_id),
            "description": f"MethodAtom {method_doc.title}",
            "source_id": method_doc.canonical_id,
            "file_path": method_doc.rel_path,
        }
        m = re.search(r"^source_path:\s*(.+)$", method_doc.text, flags=re.M)
        if m:
            source_rel = m.group(1).strip()
            source_path = root / source_rel
            if source_path.exists():
                source_id = canonical_id_for(root, source_path)
                entities.setdefault(
                    source_id,
                    {
                        "entity_name": source_id,
                        "entity_type": custom_kg_entity_type(source_id),
                        "description": f"Source for {method_doc.canonical_id}",
                        "source_id": source_id if source_id in doc_ids else fallback_source,
                        "file_path": source_rel,
                    },
                )
                relationships.append(
                    {
                        "src_id": method_doc.canonical_id,
                        "tgt_id": source_id,
                        "description": f"{method_doc.canonical_id} METHOD_ATOM_FROM {source_id}; evidence: {method_doc.rel_path}.",
                        "keywords": "METHOD_ATOM_FROM",
                        "source_id": method_doc.canonical_id,
                        "weight": 0.9,
                        "file_path": method_doc.rel_path,
                    }
                )

    raw_section_chunk_count = 0
    for section_doc in generated_docs_from_state(state_dir, kind="raw_section"):
        raw_section_chunk_count += 1
        chunks.append(
            {
                "content": section_doc.text,
                "source_id": section_doc.canonical_id,
                "file_path": section_doc.rel_path,
                "chunk_order_index": 0,
            }
        )
        doc_ids.add(section_doc.canonical_id)
        entities[section_doc.canonical_id] = {
            "entity_name": section_doc.canonical_id,
            "entity_type": custom_kg_entity_type(section_doc.canonical_id),
            "description": f"RawSection {section_doc.title}",
            "source_id": section_doc.canonical_id,
            "file_path": section_doc.rel_path,
        }
        source_id_match = re.search(r"^source_id:\s*(.+)$", section_doc.text, flags=re.M)
        source_path_match = re.search(r"^source_path:\s*(.+)$", section_doc.text, flags=re.M)
        section_kind_match = re.search(r"^section_kind:\s*(.+)$", section_doc.text, flags=re.M)
        source_id = source_id_match.group(1).strip() if source_id_match else ""
        source_rel = source_path_match.group(1).strip() if source_path_match else ""
        section_kind = section_kind_match.group(1).strip() if section_kind_match else "section"
        if source_id:
            entities.setdefault(
                source_id,
                {
                    "entity_name": source_id,
                    "entity_type": custom_kg_entity_type(source_id),
                    "description": f"Source for {section_doc.canonical_id}",
                    "source_id": source_id if source_id in doc_ids else fallback_source,
                    "file_path": source_rel or section_doc.rel_path,
                },
            )
            relationships.append(
                {
                    "src_id": section_doc.canonical_id,
                    "tgt_id": source_id,
                    "description": f"{section_doc.canonical_id} RAW_SECTION_OF {source_id}; section_kind: {section_kind}; evidence: {section_doc.rel_path}.",
                    "keywords": "RAW_SECTION_OF",
                    "source_id": section_doc.canonical_id,
                    "weight": 0.85,
                    "file_path": section_doc.rel_path,
                }
            )

    section_similarity_relationship_count = 0
    for edge in jsonl_read(state_dir / "section_similarity_edges.jsonl"):
        src = str(edge.get("src_id", "")).strip()
        tgt = str(edge.get("tgt_id", "")).strip()
        if not src or not tgt:
            continue
        for node, path_key, title_key in [
            (src, "source_path", "source_title"),
            (tgt, "target_path", "target_title"),
        ]:
            entities.setdefault(
                node,
                {
                    "entity_name": node,
                    "entity_type": custom_kg_entity_type(node),
                    "description": f"Semantic section neighbor endpoint {edge.get(title_key, node)}",
                    "source_id": node if node in doc_ids else fallback_source,
                    "file_path": str(edge.get(path_key, "section_similarity_edges.jsonl")),
                },
            )
        relationships.append(section_similarity_edge_to_custom_kg_relationship(edge))
        section_similarity_relationship_count += 1

    seed_edges = jsonl_read(state_dir / "seed_edges.jsonl")
    if limit_edges is not None:
        seed_edges = seed_edges[:limit_edges]
    for edge in seed_edges:
        src = str(edge.get("src_id", "")).strip()
        tgt = str(edge.get("tgt_id", "")).strip()
        if not src or not tgt:
            continue
        evidence_path = str(edge.get("evidence_path", "custom_kg"))
        for node in (src, tgt):
            entities.setdefault(
                node,
                {
                    "entity_name": node,
                    "entity_type": custom_kg_entity_type(node),
                    "description": f"Deterministic llm-wiki node {node}",
                    "source_id": src if src in doc_ids else (tgt if tgt in doc_ids else fallback_source),
                    "file_path": evidence_path,
                },
            )
        source_id = src if src in doc_ids else (tgt if tgt in doc_ids else fallback_source)
        edge_type = str(edge.get("type", "RELATED_TO"))
        relationships.append(
            {
                "src_id": src,
                "tgt_id": tgt,
                "description": f"{src} {edge_type} {tgt}; evidence: {evidence_path} ({edge.get('evidence_field', '')}).",
                "keywords": edge_type,
                "source_id": source_id,
                "weight": float(edge.get("weight", 1.0)),
                "file_path": evidence_path,
            }
        )
    payload = {"chunks": chunks, "entities": list(entities.values()), "relationships": relationships}
    summary = {
        "chunks": len(chunks),
        "raw_section_chunks": raw_section_chunk_count,
        "section_similarity_relationships": section_similarity_relationship_count,
        "entities": len(entities),
        "relationships": len(relationships),
        "limit_docs": limit_docs,
        "limit_edges": limit_edges,
    }
    return payload, summary


def generated_docs_from_state(state_dir: Path, kind: str = "all") -> list[WikiDoc]:
    ensure_state_dirs(state_dir)
    dirs: list[tuple[str, str]] = []
    if kind in {"all", "edge"}:
        dirs.append(("edge_docs", "edge_doc"))
    if kind in {"all", "method_atom"}:
        dirs.append(("method_atom_docs", "method_atom"))
    if kind in {"all", "raw_section"}:
        dirs.append(("raw_section_docs", "raw_section"))
    docs: list[WikiDoc] = []
    for dirname, doc_type in dirs:
        for path in sorted((state_dir / dirname).glob("*.md")):
            text = read_text(path)
            rel = path.relative_to(state_dir).as_posix()
            cid = generated_doc_id(text, path, doc_type)
            docs.append(
                WikiDoc(
                    path=path,
                    rel_path=rel,
                    canonical_id=cid,
                    doc_type=doc_type,
                    title=path.stem,
                    frontmatter={},
                    body=text,
                    text=text,
                    sha256=sha256_text(text),
                )
            )
    return docs


def generated_doc_id(text: str, path: Path, doc_type: str) -> str:
    if doc_type == "edge_doc":
        m = re.search(r"^edge_id:\s*(.+)$", text, re.M)
        return m.group(1).strip() if m else f"edge_doc:{path.stem}"
    if doc_type == "method_atom":
        m = re.search(r"^atom_id:\s*(.+)$", text, re.M)
        return m.group(1).strip() if m else f"method_atom:{path.stem}"
    if doc_type == "raw_section":
        m = re.search(r"^section_id:\s*(.+)$", text, re.M)
        return m.group(1).strip() if m else f"raw_section:{path.stem}"
    return f"generated:{path.stem}"


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


def expand_lightrag_data_response_with_section_neighbors(
    response: dict[str, Any],
    state_dir: Path,
    neighbor_k: int = 5,
    section_kind: str | None = None,
) -> dict[str, Any]:
    cloned = json.loads(json.dumps(response, ensure_ascii=False))
    data = cloned.get("data")
    if not isinstance(data, dict):
        return cloned
    chunks = data.get("chunks")
    if not isinstance(chunks, list):
        return cloned
    seed_ids: list[str] = []
    seed_kind_by_id: dict[str, str] = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        content = str(chunk.get("content", ""))
        section_id = raw_section_id_from_content(content)
        if not section_id:
            continue
        kind = raw_section_kind_from_content(content)
        if section_kind and kind and kind != section_kind.strip().lower():
            continue
        if section_id not in seed_ids:
            seed_ids.append(section_id)
            seed_kind_by_id[section_id] = kind
    seed_set = set(seed_ids)
    grouped: dict[str, list[dict[str, Any]]] = {seed_id: [] for seed_id in seed_ids}
    for edge in jsonl_read(state_dir / "section_similarity_edges.jsonl"):
        src = str(edge.get("src_id", ""))
        tgt = str(edge.get("tgt_id", ""))
        orientations = []
        if src in seed_set:
            orientations.append((src, tgt, edge.get("source_section_kind"), edge.get("target_section_kind"), edge.get("target_path"), edge.get("target_title")))
        if tgt in seed_set:
            orientations.append((tgt, src, edge.get("target_section_kind"), edge.get("source_section_kind"), edge.get("source_path"), edge.get("source_title")))
        for seed_id, neighbor_id, seed_kind, neighbor_kind, neighbor_path, neighbor_title in orientations:
            if section_kind and str(seed_kind or seed_kind_by_id.get(seed_id, "")).lower() != section_kind.strip().lower():
                continue
            grouped.setdefault(seed_id, []).append(
                {
                    "seed_section_id": seed_id,
                    "neighbor_section_id": neighbor_id,
                    "seed_section_kind": seed_kind or seed_kind_by_id.get(seed_id),
                    "neighbor_section_kind": neighbor_kind,
                    "neighbor_source_path": neighbor_path,
                    "neighbor_title": neighbor_title,
                    "cosine": edge.get("cosine"),
                    "pair_kind": edge.get("pair_kind"),
                    "edge_id": edge.get("edge_id"),
                    "expansion_source": "section_similarity_edges.jsonl",
                }
            )
    expansions: list[dict[str, Any]] = []
    for seed_id in seed_ids:
        rows = sorted(grouped.get(seed_id, []), key=lambda row: (-float(row.get("cosine") or 0), str(row.get("neighbor_section_id", ""))))
        expansions.extend(rows[: max(0, neighbor_k)])
    data["section_neighbor_expansions"] = expansions
    return cloned


def filter_lightrag_data_response_by_section_kind(response: dict[str, Any], section_kind: str) -> dict[str, Any]:
    kind = section_kind.strip().lower()
    cloned = json.loads(json.dumps(response, ensure_ascii=False))
    data = cloned.get("data")
    if not isinstance(data, dict):
        return cloned
    chunks = data.get("chunks")
    if not isinstance(chunks, list):
        return cloned
    filtered = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        file_path = str(chunk.get("file_path", ""))
        content = str(chunk.get("content", ""))
        if file_path.startswith("raw_section_docs/") and raw_section_kind_from_content(content) == kind:
            filtered.append(chunk)
    data["chunks"] = filtered
    data["section_kind_filter"] = kind
    data["section_kind_filter_kept"] = len(filtered)
    data["section_kind_filter_original_chunks"] = len(chunks)
    return cloned


def query_lightrag(server: str, api_key: str, query: str, mode: str = "mix", top_k: int = 20, chunk_top_k: int = 10) -> dict[str, Any]:
    payload = {
        "query": query,
        "mode": mode,
        "top_k": top_k,
        "chunk_top_k": chunk_top_k,
        "include_references": True,
        "include_chunk_content": True,
    }
    return http_json("POST", server.rstrip("/") + "/query", payload, api_key=api_key, timeout=300)


def query_lightrag_data(server: str, api_key: str, query: str, mode: str = "mix", top_k: int = 20, chunk_top_k: int = 10) -> dict[str, Any]:
    payload = {
        "query": query,
        "mode": mode,
        "top_k": top_k,
        "chunk_top_k": chunk_top_k,
        "max_total_tokens": 8000,
    }
    return http_json("POST", server.rstrip("/") + "/query/data", payload, api_key=api_key, timeout=180)


def save_evidence_pack(state_dir: Path, query: str, mode: str, response: dict[str, Any]) -> Path:
    ensure_state_dirs(state_dir)
    slug = slugify(query, 70)
    path = state_dir / "evidence_packs" / f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{slug}.md"
    refs = response.get("references") or []
    lines = [
        f"# Evidence Pack: {query}",
        "",
        f"Generated: {now_stamp()}",
        f"Mode: {mode}",
        "Intent: query",
        "",
        "## 1. Response",
        "",
        str(response.get("response", "")),
        "",
        "## 2. References",
        "",
    ]
    for i, ref in enumerate(refs, 1):
        lines.append(f"### Reference {i}")
        lines.append(f"- file_path: `{ref.get('file_path') or ref.get('source') or ''}`")
        content = ref.get("content")
        if isinstance(content, list):
            for chunk in content[:3]:
                lines.extend(["", "```text", str(chunk)[:1200], "```"])
        lines.append("")
    data = response.get("data") if isinstance(response, dict) else None
    if isinstance(data, dict):
        lines.extend(["", "## 3. Retrieval Data", ""])
        for key in ["entities", "relationships", "chunks"]:
            values = data.get(key) or []
            lines.append(f"### {key} ({len(values)})")
            for item in values[:8]:
                lines.append("")
                lines.append("```json")
                lines.append(json.dumps(item, ensure_ascii=False, indent=2)[:1600])
                lines.append("```")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def add_query_event(state_dir: Path, query: str, mode: str, evidence_pack_path: str | None = None) -> None:
    db = init_manifest_db(state_dir)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO query_events(query, mode, rewritten_queries, evidence_pack_path, created_at) VALUES(?,?,?,?,?)",
            (query, mode, None, evidence_pack_path, now_stamp()),
        )


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def common_paths_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--root", type=Path, default=DEFAULT_WIKI_ROOT)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--server", default=DEFAULT_SERVER)
    return parser
