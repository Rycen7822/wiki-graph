#!/usr/bin/env python3
"""Extract figure/table evidence from LaTeX sources for llm-wiki structured paper notes.

Use when direct visual inspection of paper figures is unavailable, slow, or flaky. The script scans one or more TeX files and/or every .tex file under one or more source directories, finds figure/table environments, and emits JSON with environment type, file, line, labels, included graphics, and a cleaned caption/block preview. It is deliberately conservative: it does not infer chart values beyond captions/source text.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

ENV_RE = re.compile(r"\\begin\{(figure\*?|table\*?)\}(.+?)\\end\{\1\}", re.S)
CAPTION_RE = re.compile(r"\\caption(?:\[[^\]]*\])?\{", re.S)
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
COMMAND_RE = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?")


def iter_tex(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from sorted(path.rglob("*.tex"))


def iter_tex_many(paths: Sequence[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for path in paths:
        for tex in iter_tex(path):
            key = tex.resolve()
            if key in seen:
                continue
            seen.add(key)
            yield tex


def balanced_brace_text(text: str, start: int) -> str:
    """Return text inside the brace opened immediately before/at start."""
    open_idx = text.find("{", start)
    if open_idx < 0:
        return ""
    depth = 0
    out = []
    for ch in text[open_idx + 1 :]:
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            if depth == 0:
                break
            depth -= 1
            out.append(ch)
        else:
            out.append(ch)
    return "".join(out)


def clean_latex(s: str) -> str:
    s = re.sub(r"%.*", "", s)
    s = COMMAND_RE.sub("", s)
    s = s.replace("~", " ")
    s = re.sub(r"[{}]", "", s)
    return " ".join(s.split())


def extract_file(path: Path) -> list[dict]:
    text = path.read_text(errors="replace")
    items = []
    for m in ENV_RE.finditer(text):
        env = m.group(1)
        block = m.group(0)
        rel_start = m.start()
        line = text[:rel_start].count("\n") + 1
        cap = ""
        cap_m = CAPTION_RE.search(block)
        if cap_m:
            cap = balanced_brace_text(block, cap_m.start())
        label_m = LABEL_RE.search(block)
        items.append(
            {
                "file": str(path),
                "line": line,
                "env": env,
                "label": label_m.group(1) if label_m else None,
                "includegraphics": GRAPHICS_RE.findall(block),
                "caption": clean_latex(cap),
                "block_preview": clean_latex(block[:800]),
            }
        )
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="one or more main.tex files or extracted arXiv source directories")
    ap.add_argument("--limit", type=int, default=200, help="maximum items to print")
    args = ap.parse_args()
    roots = [Path(path).expanduser().resolve() for path in args.paths]
    missing = [str(path) for path in roots if not path.exists()]
    if missing:
        ap.error("path(s) not found: " + ", ".join(missing))
    items: list[dict] = []
    for tex in iter_tex_many(roots):
        items.extend(extract_file(tex))
    items.sort(key=lambda x: (x["file"], x["line"]))
    print(json.dumps(items[: args.limit], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
