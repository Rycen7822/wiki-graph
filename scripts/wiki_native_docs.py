#!/usr/bin/env python3
"""Native wiki document parsing and generated-doc helpers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None  # type: ignore[assignment]

from wiki_native_query_events import slugify

COMPILED_DIR_TYPES = {
    "entities": "entity",
    "concepts": "concept",
    "comparisons": "comparison",
    "queries": "query",
}
META_FILES = ["_meta/source-map.md", "_meta/raw-clip-map.md", "_meta/topic-map.md"]


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


def generated_docs_from_state(state_dir: Path, kind: str = "all") -> list[WikiDoc]:
    dirs: list[tuple[str, str]] = []
    if kind in {"all", "edge"}:
        dirs.append(("edge_docs", "edge_doc"))
    if kind in {"all", "method_atom"}:
        dirs.append(("method_atom_docs", "method_atom"))
    if kind in {"all", "raw_section"}:
        dirs.append(("raw_section_docs", "raw_section"))
    docs: list[WikiDoc] = []
    for dirname, doc_type in dirs:
        docs_dir = state_dir / dirname
        if not docs_dir.exists():
            continue
        for path in sorted(docs_dir.glob("*.md")):
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
        match = re.search(r"^edge_id:\s*(.+)$", text, re.M)
        return match.group(1).strip() if match else f"edge_doc:{path.stem}"
    if doc_type == "method_atom":
        match = re.search(r"^atom_id:\s*(.+)$", text, re.M)
        return match.group(1).strip() if match else f"method_atom:{path.stem}"
    if doc_type == "raw_section":
        match = re.search(r"^section_id:\s*(.+)$", text, re.M)
        return match.group(1).strip() if match else f"raw_section:{path.stem}"
    return f"generated:{path.stem}"
