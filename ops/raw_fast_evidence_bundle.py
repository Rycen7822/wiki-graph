#!/usr/bin/env python3
"""Build a temp evidence bundle for llm-wiki raw-fast paper clipping.

This script is intentionally read-only with respect to the human wiki root. It
fetches/extracts/probes source evidence into a caller-provided workdir, emits a
raw-note skeleton, and leaves synthesis of the canonical raw note to the agent.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from ops.wiki_native_cli import DEFAULT_WIKI_ROOT

USER_AGENT = "Hermes llm-wiki raw-fast evidence bundle"
TEXT_EXTENSIONS = {".txt", ".json", ".md", ".html", ".htm", ".js", ".toml", ".yaml", ".yml", ".tex", ".xml"}
DEFAULT_MAX_DOWNLOAD_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_TEXT_BYTES = 8 * 1024 * 1024
STRICT_SECRET_PATTERNS = {
    "openai_key": re.compile(r"(?<![A-Za-z0-9_-])sk-(?:proj-)?[A-Za-z0-9_]{20,}(?![A-Za-z0-9_-])"),
    "anthropic_key": re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{30,}"),
    "hf_token": re.compile(r"hf_[A-Za-z0-9]{30,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
}


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


class TimingRecorder:
    """Small JSON-serializable wall-clock timing collector for wrapper reports."""

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self.started_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        self.steps: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _round_seconds(value: float) -> float:
        return round(max(value, 0.0), 6)

    def _entry_from_result(self, elapsed: float, result: Any) -> dict[str, Any]:
        entry: dict[str, Any] = {"elapsed_seconds": self._round_seconds(elapsed)}
        if isinstance(result, dict):
            for key in ["ok", "status", "returncode", "error", "message", "skipped", "reason"]:
                if key in result and result.get(key) is not None:
                    entry[key] = result.get(key)
        elif isinstance(result, list):
            entry["result_type"] = "list"
            entry["count"] = len(result)
            if all(isinstance(item, dict) and "ok" in item for item in result):
                entry["ok"] = all(bool(item.get("ok")) for item in result)
        else:
            entry["result_type"] = type(result).__name__
        return entry

    def record(self, name: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            self.steps[name] = {
                "elapsed_seconds": self._round_seconds(time.perf_counter() - started),
                "ok": False,
                "error": type(exc).__name__,
                "message": str(exc),
            }
            raise
        self.steps[name] = self._entry_from_result(time.perf_counter() - started, result)
        return result

    def snapshot(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "total_seconds": self._round_seconds(time.perf_counter() - self._start),
            "steps": self.steps,
        }


def attach_timings(payload: dict[str, Any], timings: TimingRecorder) -> dict[str, Any]:
    payload["timings"] = timings.snapshot()
    return payload


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path, limit: int | None = None) -> str:
    data = path.read_text(encoding="utf-8", errors="replace")
    return data[:limit] if limit is not None else data


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_command(command: list[str], timeout: int = 60) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "command": command,
        }
    except FileNotFoundError as exc:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": str(exc), "command": command, "error": type(exc).__name__}
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "command": command,
            "error": "TimeoutExpired",
            "timeout": timeout,
        }


def fetch_url_to_file(url: str, dest: Path, timeout: int, max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "file":
        source = Path(urllib.request.url2pathname(parsed.path))
        size = source.stat().st_size
        if size > max_bytes:
            return {"ok": False, "url": url, "dest": str(dest), "error": "FileTooLarge", "bytes": size, "max_bytes": max_bytes}
        shutil.copyfile(source, dest)
        return {
            "ok": True,
            "url": url,
            "status": 200,
            "content_type": "application/pdf" if dest.suffix.lower() == ".pdf" else "application/octet-stream",
            "bytes": dest.stat().st_size,
            "sha256": sha256_file(dest),
            "dest": str(dest),
        }
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        h = hashlib.sha256()
        total = 0
        with urllib.request.urlopen(req, timeout=timeout) as response, dest.open("wb") as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    out.close()
                    dest.unlink(missing_ok=True)
                    return {"ok": False, "url": url, "dest": str(dest), "error": "DownloadTooLarge", "bytes": total, "max_bytes": max_bytes}
                h.update(chunk)
                out.write(chunk)
            return {
                "ok": True,
                "url": url,
                "status": getattr(response, "status", None),
                "content_type": response.headers.get("content-type"),
                "bytes": total,
                "sha256": h.hexdigest(),
                "dest": str(dest),
            }
    except Exception as exc:
        return {"ok": False, "url": url, "dest": str(dest), "error": type(exc).__name__, "message": str(exc)}


def fetch_text(url: str, timeout: int, max_bytes: int = DEFAULT_MAX_TEXT_BYTES) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(url)
    try:
        if parsed.scheme == "file":
            path = Path(urllib.request.url2pathname(parsed.path))
            size = path.stat().st_size
            if size > max_bytes:
                return {"ok": False, "url": url, "status": 200, "error": "FileTooLarge", "bytes": size, "max_bytes": max_bytes}
            data = path.read_bytes()
            return {"ok": True, "url": url, "status": 200, "content_type": "text/plain", "text": data.decode("utf-8", "replace"), "bytes": len(data)}
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = response.read(512 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    return {"ok": False, "url": url, "status": getattr(response, "status", None), "content_type": response.headers.get("content-type"), "error": "ResponseTooLarge", "bytes": total, "max_bytes": max_bytes}
                chunks.append(chunk)
            data = b"".join(chunks)
            return {
                "ok": True,
                "url": url,
                "status": getattr(response, "status", None),
                "content_type": response.headers.get("content-type"),
                "text": data.decode("utf-8", "replace"),
                "bytes": len(data),
            }
    except Exception as exc:
        return {"ok": False, "url": url, "error": type(exc).__name__, "message": str(exc)}


def detect_kind(url: str, requested: str) -> str:
    if requested != "auto":
        return requested
    if re.search(r"arxiv\.org/(abs|pdf|html)/\d{4}\.\d{4,5}", url):
        return "arxiv"
    return "direct-pdf"


def arxiv_id_from_url(url: str) -> str | None:
    match = re.search(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})(?:v\d+)?", url)
    return match.group(1) if match else None


def safe_extract_tar(tar_path: Path, dest: Path) -> dict[str, Any]:
    extracted: list[str] = []
    errors: list[str] = []
    skipped: list[str] = []
    root = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(tar_path) as tf:
            for member in tf.getmembers():
                target = (dest / member.name).resolve()
                if not (target == root or target.is_relative_to(root)):
                    errors.append(f"blocked path traversal: {member.name}")
                    continue
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    extracted.append(member.name)
                    continue
                if not member.isfile():
                    skipped.append(member.name)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(member)
                if src is None:
                    skipped.append(member.name)
                    continue
                with src, target.open("wb") as out:
                    shutil.copyfileobj(src, out, length=1024 * 1024)
                extracted.append(member.name)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    return {"ok": not errors, "extracted_count": len(extracted), "extracted": extracted[:200], "skipped": skipped[:200], "errors": errors}


def run_pdfinfo(pdf: Path, out: Path) -> dict[str, Any]:
    result = run_command(["pdfinfo", str(pdf)], timeout=60)
    write_text(out, result.get("stdout") or "")
    return {k: v for k, v in result.items() if k not in {"stdout", "stderr"}} | {"stderr_tail": (result.get("stderr") or "")[-800:]}


def run_pdftotext(pdf: Path, out: Path, layout: bool) -> dict[str, Any]:
    command = ["pdftotext"]
    if layout:
        command.append("-layout")
    command += [str(pdf), str(out)]
    result = run_command(command, timeout=120)
    if not out.exists():
        write_text(out, "")
    return {k: v for k, v in result.items() if k not in {"stdout", "stderr"}} | {"stderr_tail": (result.get("stderr") or "")[-800:]}


def extract_pdf_links(pdf: Path, out: Path) -> dict[str, Any]:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        links = {"ok": False, "error": type(exc).__name__, "message": str(exc), "links": []}
        write_json(out, links)
        return links
    links: list[dict[str, Any]] = []
    try:
        doc = fitz.open(str(pdf))
        for page_i, page in enumerate(doc, 1):
            for link in page.get_links():
                uri = link.get("uri")
                if uri:
                    links.append({"page": page_i, "uri": uri})
        doc.close()
        payload = {"ok": True, "links": links, "link_count": len(links)}
    except Exception as exc:
        payload = {"ok": False, "error": type(exc).__name__, "message": str(exc), "links": links, "link_count": len(links)}
    write_json(out, payload)
    return payload


def run_docling(pdf: Path, workdir: Path, strict: bool = False, timeout: int = 120) -> dict[str, Any]:
    out_md = workdir / "docling.md"
    out_json = workdir / "docling.json"
    code = r'''
import json, sys
from pathlib import Path
pdf = Path(sys.argv[1])
out_md = Path(sys.argv[2])
out_json = Path(sys.argv[3])
try:
    from docling.document_converter import DocumentConverter
    converter = DocumentConverter()
    result = converter.convert(str(pdf))
    doc = result.document
    markdown = doc.export_to_markdown()
    out_md.write_text(markdown, encoding="utf-8")
    try:
        as_dict = doc.export_to_dict()
    except Exception:
        as_dict = {"markdown_chars": len(markdown)}
    out_json.write_text(json.dumps(as_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "markdown_chars": len(markdown)}))
except Exception as exc:
    payload = {"ok": False, "error": type(exc).__name__, "message": str(exc)}
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload))
    raise SystemExit(1)
'''
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code, str(pdf), str(out_md), str(out_json)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        payload = {"ok": False, "error": "TimeoutExpired", "message": f"Docling exceeded {timeout}s", "outputs": {"markdown": str(out_md), "json": str(out_json)}, "stderr_tail": (exc.stderr or "")[-800:]}
        if strict:
            payload["strict_failure"] = True
        write_json(out_json, payload)
        return payload
    payload = parse_docling_stdout(completed.stdout)
    if completed.returncode == 0 and isinstance(payload, dict) and payload.get("ok"):
        return {"ok": True, "markdown": str(out_md), "json": str(out_json), "markdown_chars": payload.get("markdown_chars"), "stderr_tail": (completed.stderr or "")[-800:]}
    if not isinstance(payload, dict):
        payload = {"ok": False, "error": "DoclingFailed", "message": "Docling did not return JSON"}
    payload = {**payload, "ok": False, "outputs": {"markdown": str(out_md), "json": str(out_json)}, "stderr_tail": (completed.stderr or "")[-800:]}
    if strict:
        payload["strict_failure"] = True
    if not out_json.exists():
        write_json(out_json, payload)
    return payload


def parse_docling_stdout(stdout: str) -> dict[str, Any] | None:
    text = (stdout or "").strip()
    if not text:
        return None
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def section_inventory(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if re.match(r"^(\d+(?:\.\d+)*)\s+[A-Z][A-Za-z0-9 ,:;()/-]{2,}$", stripped) or re.match(r"^(Abstract|Introduction|Related Work|Method|Methods|Experiments?|Results?|Discussion|Conclusion|Limitations|Appendix)\b", stripped, re.I):
            rows.append({"line": line_no, "heading": stripped[:180]})
    return rows[:200]


def figure_table_inventory(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(r"(?is)\b((?:Figure|Fig\.|Table)\s*\d+[^\n]{0,120}(?:\n.{0,240})?)")
    for match in pattern.finditer(text):
        excerpt = re.sub(r"\s+", " ", match.group(1)).strip()
        rows.append({"kind": "table" if excerpt.lower().startswith("table") else "figure", "caption_excerpt": excerpt[:500]})
    return rows[:200]


def extract_urls(text: str) -> list[str]:
    raw = re.findall(r"https?://[^\s)\]}>\"']+", text)
    return sorted({u.rstrip(".,;:") for u in raw})


GITHUB_EXCLUDED_OWNER_SEGMENTS = {"features", "topics", "collections", "marketplace", "pricing", "about", "login", "search"}
RESOURCE_HEALTH_MAX_CANDIDATES = 8


def _normalized_url_host(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.lower().removeprefix("www.")


def _resource_url_candidate(url: str) -> dict[str, Any] | None:
    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    parts = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
    base = {"url": url, "source": "source_exposed_exact_url", "status": "not_checked"}
    if host == "github.com" and len(parts) >= 2 and parts[0].lower() not in GITHUB_EXCLUDED_OWNER_SEGMENTS:
        owner, repo = parts[0], re.sub(r"\.git$", "", parts[1])
        if owner and repo:
            return {**base, "type": "github", "repo": f"{owner}/{repo}"}
    if host == "huggingface.co" and parts:
        if len(parts) >= 3 and parts[0] in {"datasets", "spaces"}:
            return {**base, "type": "hf", "hf_kind": parts[0], "repo": f"{parts[1]}/{parts[2]}"}
        if len(parts) >= 2:
            return {**base, "type": "hf", "hf_kind": "models", "repo": f"{parts[0]}/{parts[1]}"}
    if host and host not in {"arxiv.org", "export.arxiv.org", "doi.org", "dx.doi.org"} and re.search(r"project|code|artifact|software|release|demo|model|dataset", url, re.I):
        return {**base, "type": "project_page"}
    return None


def probe_exact_link_health(url: str, *, timeout: int = 8, max_bytes: int = 1024) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "file":
        path = Path(urllib.request.url2pathname(parsed.path))
        return {"ok": path.exists(), "chosen": "file", "url": url, "status": "verified_present" if path.exists() else "probe_failed", "error": None if path.exists() else "file_missing"}
    attempts: list[dict[str, Any]] = []
    for method in ["HEAD", "GET"]:
        try:
            req = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})
            if method == "GET":
                req.add_header("Range", f"bytes=0-{max_bytes - 1}")
            with urllib.request.urlopen(req, timeout=timeout) as response:
                code = getattr(response, "status", None)
                item = {"method": method, "ok": code is not None and 200 <= int(code) < 400, "http_code": code, "effective_url": response.geturl(), "content_type": response.headers.get("content-type")}
                attempts.append(item)
                if item["ok"]:
                    return {"ok": True, "chosen": method, "url": url, "status": "verified_present", "attempts": attempts}
        except urllib.error.HTTPError as exc:
            item = {"method": method, "ok": False, "http_code": exc.code, "error": type(exc).__name__, "message": str(exc)[:300]}
            attempts.append(item)
            if method == "HEAD" and exc.code in {403, 405}:
                continue
        except Exception as exc:
            attempts.append({"method": method, "ok": False, "error": type(exc).__name__, "message": str(exc)[:300]})
    return {"ok": False, "chosen": attempts[-1].get("method") if attempts else None, "url": url, "status": "probe_failed", "attempts": attempts, "error": attempts[-1].get("error") if attempts else "probe_failed"}


def classify_source_exposed_resources(urls: list[str], *, health_mode: str = "direct", timeout: int = 8) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None]] = set()
    for url in urls:
        candidate = _resource_url_candidate(url)
        if not candidate:
            continue
        key = (candidate["type"], candidate.get("repo") or candidate["url"])
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    health_enabled = health_mode == "direct"
    for candidate in candidates[:RESOURCE_HEALTH_MAX_CANDIDATES]:
        if not health_enabled:
            candidate["health"] = {"skipped": True, "reason": "resource_health_disabled"}
            continue
        health = probe_exact_link_health(candidate["url"], timeout=timeout)
        candidate["health"] = health
        candidate["status"] = "verified_present" if health.get("ok") else "probe_failed"
    for candidate in candidates[RESOURCE_HEALTH_MAX_CANDIDATES:]:
        candidate["health"] = {"skipped": True, "reason": "candidate_limit"}
    review_required = any(item.get("status") == "probe_failed" for item in candidates)
    return {
        "ok": True,
        "health_mode": health_mode,
        "review_required": review_required,
        "candidates": candidates,
        "github": [item for item in candidates if item.get("type") == "github"],
        "hf": [item for item in candidates if item.get("type") == "hf"],
        "project_pages": [item for item in candidates if item.get("type") == "project_page"],
    }


def build_resource_probe(text: str, links_payload: dict[str, Any], probes: list[str], *, health_mode: str = "direct", timeout: int = 8) -> dict[str, Any]:
    urls = set(extract_urls(text))
    for item in links_payload.get("links") or []:
        if isinstance(item, dict) and item.get("uri"):
            urls.add(str(item["uri"]))
    probes_set = set(probes)
    effective_health_mode = "none" if "none" in probes_set else health_mode
    sorted_urls = sorted(urls)
    source_exposed = classify_source_exposed_resources(sorted_urls, health_mode=effective_health_mode, timeout=timeout)
    if "none" in probes_set:
        return {"ok": True, "skipped": True, "urls": sorted_urls, "source_exposed_resources": source_exposed, "probes": []}
    doi_strings = sorted(set(re.findall(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, flags=re.I)))[:50]
    arxiv_ids = sorted(set(re.findall(r"arXiv[: ]?(\d{4}\.\d{4,5})", text, flags=re.I)))[:50]
    results: list[dict[str, Any]] = []
    if "doi" in probes_set:
        for doi in doi_strings:
            results.append({"ok": True, "type": "doi", "doi": doi.rstrip('.'), "status": "detected", "url": f"https://doi.org/{doi.rstrip('.')}", "error": None, "evidence": {"matched_text": doi}})
    if "arxiv" in probes_set:
        for arxiv_id in arxiv_ids:
            results.append({"ok": True, "type": "arxiv", "id": arxiv_id, "status": "detected", "url": f"https://arxiv.org/abs/{arxiv_id}", "error": None, "evidence": {"matched_text": arxiv_id}})
    return {"ok": True, "urls": sorted_urls, "doi_strings": doi_strings, "arxiv_ids": arxiv_ids, "source_exposed_resources": source_exposed, "probes": results}


def title_from_text(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        stripped = re.sub(r"^#{1,6}\s+", "", stripped).strip()
        if 5 <= len(stripped) <= 180 and not stripped.lower().startswith(("abstract", "arxiv", "http")):
            return stripped
    return "Untitled Paper"


def slugify(text: str, max_len: int = 96) -> str:
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", text.strip()).strip("-")
    return (slug or "Untitled-Paper")[:max_len].strip("-") or "Untitled-Paper"


SAFE_RASTER_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SAFE_IMAGE_EXTENSIONS = SAFE_RASTER_IMAGE_EXTENSIONS | {".pdf"}
SAFE_IMAGE_EXTENSION_ORDER = [".png", ".jpg", ".jpeg", ".webp", ".pdf"]


def redact_secret_like_text(text: str) -> str:
    redacted = text or ""
    for pattern in STRICT_SECRET_PATTERNS.values():
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def compact_ws(text: str, max_len: int = 500) -> str:
    return re.sub(r"\s+", " ", redact_secret_like_text(text or "")).strip()[:max_len]


def strip_tex_markup(text: str, max_len: int = 500) -> str:
    out = text or ""
    for _ in range(3):
        out = re.sub(r"\\(?:textbf|textit|emph|section|subsection|caption|title|url)\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", out)
    out = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?", "", out)
    out = out.replace("{", "").replace("}", "")
    out = out.replace("~", " ").replace("\\&", "&")
    return compact_ws(out, max_len=max_len)


def tex_command_arg(text: str, command: str) -> str | None:
    match = re.search(rf"\\{re.escape(command)}\*?(?:\[[^\]]*\])?\{{(.*?)\}}", text, re.S)
    return strip_tex_markup(match.group(1), max_len=500) if match else None


def collect_tex_files(workdir: Path) -> list[Path]:
    source = workdir / "source"
    roots = [source] if source.exists() else [workdir]
    files: list[Path] = []
    for root in roots:
        files.extend(sorted(p for p in root.rglob("*.tex") if p.is_file()))
    return files[:50]


def read_tex_bundle(workdir: Path, limit_per_file: int = 200_000) -> tuple[list[Path], str]:
    tex_files = collect_tex_files(workdir)
    chunks: list[str] = []
    for path in tex_files:
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="replace")[:limit_per_file])
        except OSError:
            continue
    return tex_files, "\n".join(chunks)


def extract_tex_abstract(tex: str) -> str:
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S | re.I)
    return strip_tex_markup(match.group(1), max_len=1200) if match else ""


def extract_tex_section_cards(tex: str) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    pattern = re.compile(r"\\(?P<level>section|subsection|subsubsection)\*?\{(?P<title>.*?)\}", re.S)
    matches = list(pattern.finditer(tex))
    for idx, match in enumerate(matches[:80]):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else min(len(tex), start + 2000)
        heading = strip_tex_markup(match.group("title"), max_len=180)
        excerpt = strip_tex_markup(tex[start:end], max_len=600)
        cards.append({"level": match.group("level"), "heading": heading, "excerpt": excerpt})
    return cards


def extract_tex_equation_cards(tex: str) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    env_pattern = re.compile(r"\\begin\{(?P<env>equation|align|gather|multline)\*?\}(?P<body>.*?)\\end\{(?P=env)\*?\}", re.S)
    for match in env_pattern.finditer(tex):
        body = match.group("body")
        label = tex_command_arg(body, "label")
        formula = re.sub(r"\\label\{[^{}]*\}", "", body).strip()
        cards.append({"env": match.group("env"), "label": label, "formula": compact_ws(formula, max_len=900)})
        if len(cards) >= 60:
            break
    return cards


def extract_tex_table_cards(tex: str) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for match in re.finditer(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", tex, re.S | re.I):
        env = match.group(1)
        caption = tex_command_arg(env, "caption")
        label = tex_command_arg(env, "label")
        tabular = re.search(r"\\begin\{tabular\}\{[^{}]*\}(.*?)\\end\{tabular\}", env, re.S | re.I)
        body_excerpt = strip_tex_markup(tabular.group(1) if tabular else env, max_len=900)
        cards.append({"label": label, "caption": caption, "body_excerpt": body_excerpt})
        if len(cards) >= 40:
            break
    return cards


def extract_figure_contexts(workdir: Path) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    source_root = (workdir / "source").resolve()
    tex_files = collect_tex_files(workdir)
    figure_env = re.compile(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", re.S | re.I)
    include_re = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}", re.S)
    for tex_path in tex_files:
        try:
            text = tex_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        envs = list(figure_env.finditer(text))
        if not envs:
            envs = [m for m in include_re.finditer(text)]
        for env_match in envs:
            env_text = env_match.group(1) if env_match.re is figure_env else env_match.group(0)
            caption = tex_command_arg(env_text, "caption")
            label = tex_command_arg(env_text, "label")
            for include in include_re.findall(env_text):
                contexts.append({"tex_file": str(tex_path), "tex_rel": str(tex_path.relative_to(source_root)) if source_root.exists() and tex_path.resolve().is_relative_to(source_root) else tex_path.name, "include": include.strip(), "caption": caption, "label": label})
                if len(contexts) >= 80:
                    return contexts
    return contexts


def _resolved_include_candidate(source_root: Path, tex_file: Path, include: str) -> tuple[Path | None, str | None]:
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", include):
        return None, "remote_url"
    raw = Path(include)
    if raw.is_absolute() or any(part == ".." for part in raw.parts):
        return None, "path_traversal"
    if any(part.startswith(".") for part in raw.parts):
        return None, "hidden_path"
    if raw.suffix and raw.suffix.lower() not in SAFE_IMAGE_EXTENSIONS:
        return None, "unsupported_extension"
    bases = [tex_file.parent / raw, source_root / raw]
    candidates: list[Path] = []
    seen: set[str] = set()
    for base in bases:
        expanded = [base] if raw.suffix else [base.with_suffix(ext) for ext in SAFE_IMAGE_EXTENSION_ORDER]
        for candidate in expanded:
            key = str(candidate)
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
    root = source_root.resolve()
    for candidate in candidates:
        if not candidate.exists():
            continue
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root):
            return None, "path_traversal"
        if resolved.suffix.lower() not in SAFE_IMAGE_EXTENSIONS:
            return None, "unsupported_extension"
        return resolved, None
    return None, "missing_source"


def extract_tex_figure_cards(workdir: Path) -> list[dict[str, Any]]:
    source_root = (workdir / "source").resolve()
    cards: list[dict[str, Any]] = []
    for context in extract_figure_contexts(workdir):
        tex_file = Path(context["tex_file"])
        resolved, reason = _resolved_include_candidate(source_root, tex_file, context["include"])
        cards.append({
            "label": context.get("label"),
            "caption": context.get("caption"),
            "include": context.get("include"),
            "tex_rel": context.get("tex_rel"),
            "localizable": bool(resolved),
            "source_path": str(resolved.relative_to(source_root)) if resolved and resolved.is_relative_to(source_root) else None,
            "reason": reason,
        })
    return cards


def extract_result_cards(section_cards: list[dict[str, Any]], table_cards: list[dict[str, Any]], figure_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for card in table_cards[:8]:
        cards.append({"source": "table", "label": card.get("label"), "caption": card.get("caption"), "evidence_excerpt": card.get("body_excerpt")})
    for card in figure_cards[:8]:
        caption = card.get("caption") or ""
        if re.search(r"result|benchmark|performance|accuracy|score|ablation|实验|结果", caption, re.I):
            cards.append({"source": "figure", "label": card.get("label"), "caption": caption})
    for card in section_cards:
        if re.search(r"result|experiment|evaluation|ablation|benchmark", card.get("heading", ""), re.I):
            cards.append({"source": "section", "heading": card.get("heading"), "evidence_excerpt": card.get("excerpt")})
    return cards[:20]


def placeholder_title(text: str | None) -> bool:
    value = compact_ws(text or "", max_len=220).lower()
    if not value or value in {"untitled paper", "paper"}:
        return True
    return bool(re.fullmatch(r"(?:<!--\s*)?(?:image|figure|img)(?:\s*-->)?", value)) or value.startswith("<!-- image")


def arxiv_api_title(workdir: Path) -> str | None:
    api_path = workdir / "api.xml"
    if not api_path.exists():
        return None
    text = api_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"<entry\b.*?<title[^>]*>(.*?)</title>", text, re.S | re.I)
    if not match:
        return None
    title = compact_ws(re.sub(r"<[^>]+>", " ", match.group(1)), max_len=300)
    return title or None


def summarize_resource_boundary(resource_probe: dict[str, Any], metadata: dict[str, Any] | None = None, source_inventory: dict[str, Any] | None = None, pdf_info: dict[str, Any] | None = None, secret_scan: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = metadata or {}
    source_exposed = resource_probe.get("source_exposed_resources") if isinstance(resource_probe.get("source_exposed_resources"), dict) else {}
    hf_buckets = {
        "models": {"status": "not_checked", "items": []},
        "datasets": {"status": "not_checked", "items": []},
        "spaces": {"status": "not_checked", "items": []},
    }
    for item in source_exposed.get("hf") or []:
        kind = item.get("hf_kind") or "models"
        bucket = hf_buckets.setdefault(kind, {"status": "not_checked", "items": []})
        bucket.setdefault("items", []).append(item)
    for bucket in hf_buckets.values():
        statuses = {entry.get("status") for entry in bucket.get("items") or []}
        if "probe_failed" in statuses:
            bucket["status"] = "probe_failed"
        elif "verified_present" in statuses:
            bucket["status"] = "verified_present"
        elif bucket.get("items"):
            bucket["status"] = "not_checked"

    draft: dict[str, Any] = {
        "ok": True,
        "advisory": True,
        "metadata": metadata,
        "health_mode": source_exposed.get("health_mode", "none"),
        "review_required": bool(source_exposed.get("review_required")),
        "manual_reason": None,
        "github": source_exposed.get("github") or [],
        "project_pages": source_exposed.get("project_pages") or [],
        "hf": hf_buckets,
        "doi": [],
        "arxiv": [],
        "warnings": [],
    }
    if draft["review_required"]:
        failed = [item for item in (source_exposed.get("candidates") or []) if item.get("status") == "probe_failed"]
        draft["manual_reason"] = {"stage": "resource_health", "error": "source_exposed_link_probe_failed", "failed_urls": [item.get("url") for item in failed]}
    for probe in resource_probe.get("probes") or []:
        ptype = str(probe.get("type") or "")
        if ptype in {"doi", "arxiv"}:
            draft[ptype].append({"status": probe.get("status") or ("verified" if probe.get("ok") else "probe_failed"), "id": probe.get("id"), "doi": probe.get("doi"), "url": probe.get("url"), "error": probe.get("error")})
    if secret_scan and secret_scan.get("strict_secret_hits"):
        draft["warnings"].append("strict_secret_hits_present_in_evidence; do not copy raw secret text")
    if source_inventory:
        draft["source_inventory"] = {k: source_inventory.get(k) for k in ["tex_files", "figure_table_items"] if k in source_inventory}
    if pdf_info:
        draft["pdf_info"] = pdf_info
    return draft


def resource_status_summary(resource_boundary: dict[str, Any] | None) -> str:
    if not resource_boundary:
        return "resource_boundary=missing"
    github = resource_boundary.get("github") or []
    projects = resource_boundary.get("project_pages") or []
    hf = resource_boundary.get("hf") or {}
    github_text = ";".join(f"{item.get('status')}:{item.get('url')}" for item in github) or "none"
    project_text = ";".join(f"{item.get('status')}:{item.get('url')}" for item in projects) or "none"
    hf_text = ",".join(f"{key}:{value.get('status')}" for key, value in sorted(hf.items())) or "none"
    review = "yes" if resource_boundary.get("review_required") else "no"
    return f"source_exposed_resources github={github_text}; hf={hf_text}; project_pages={project_text}; review_required={review}"


def render_resource_boundary_markdown(draft: dict[str, Any]) -> str:
    lines = ["# Resource boundary draft", "", "Advisory sidecar: source-exposed exact-link resource triage. Keep this out of raw-note body/frontmatter.", "", f"- health_mode: {draft.get('health_mode')}", f"- review_required: {draft.get('review_required')}", "", "## GitHub"]
    if draft.get("github"):
        for item in draft.get("github") or []:
            lines.append(f"- {item.get('url')}: status={item.get('status')}, repo={item.get('repo')}")
    else:
        lines.append("- none")
    lines.append("\n## Hugging Face")
    for kind, bucket in (draft.get("hf") or {}).items():
        lines.append(f"- {kind}: status={bucket.get('status')}, count={len(bucket.get('items') or [])}")
    lines.append("\n## Project pages")
    if draft.get("project_pages"):
        for item in draft.get("project_pages") or []:
            lines.append(f"- {item.get('url')}: status={item.get('status')}")
    else:
        lines.append("- none")
    if draft.get("doi"):
        lines.append("\n## DOI")
        for item in draft["doi"]:
            lines.append(f"- {item.get('doi')}: status={item.get('status')}")
    if draft.get("arxiv"):
        lines.append("\n## arXiv")
        for item in draft["arxiv"]:
            lines.append(f"- {item.get('id')}: status={item.get('status')}")
    if draft.get("manual_reason"):
        lines.append("\n## Manual follow-up reason")
        lines.append(f"- {json.dumps(draft.get('manual_reason'), ensure_ascii=False, sort_keys=True)}")
    if draft.get("warnings"):
        lines.append("\n## Warnings")
        for warning in draft["warnings"]:
            lines.append(f"- {warning}")
    return "\n".join(lines).rstrip() + "\n"


def build_paper_digest(workdir: Path, title: str, source_url: str, resource_probe: dict[str, Any] | None = None, files: dict[str, str] | None = None) -> dict[str, Any]:
    tex_files, tex_text = read_tex_bundle(workdir)
    text_candidates: list[str] = []
    for rel in ["docling.md", "paper.layout.txt", "paper.raw.txt"]:
        path = workdir / rel
        if path.exists():
            text_candidates.append(read_text(path, limit=100_000))
    combined_text = "\n".join(text_candidates)
    api_title = arxiv_api_title(workdir)
    tex_title = tex_command_arg(tex_text, "title") if tex_text else None
    metadata_title = next((candidate for candidate in [api_title, tex_title, title] if candidate and not placeholder_title(candidate)), title or "Untitled Paper")
    section_cards = extract_tex_section_cards(tex_text) if tex_text else section_inventory(combined_text)
    equation_cards = extract_tex_equation_cards(tex_text) if tex_text else []
    table_cards = extract_tex_table_cards(tex_text) if tex_text else []
    figure_cards = extract_tex_figure_cards(workdir) if tex_text else figure_table_inventory(combined_text)
    abstract = extract_tex_abstract(tex_text) if tex_text else ""
    if not abstract:
        abstract_match = re.search(r"(?is)abstract\s+(.{80,1200}?)(?:\n\s*\n|introduction|method)", combined_text)
        abstract = compact_ws(abstract_match.group(1), max_len=1200) if abstract_match else ""
    resource_boundary = summarize_resource_boundary(resource_probe or {}, metadata={"title": metadata_title, "source_url": source_url})
    implementation_cards = []
    for url in sorted(set(extract_urls(tex_text + "\n" + combined_text)))[:30]:
        if "github.com" in url or "huggingface.co" in url or "code" in url.lower():
            implementation_cards.append({"url": url, "evidence": "detected in source/text"})
    limitation_cards = [card for card in section_cards if re.search(r"limitation|discussion|caveat|局限", card.get("heading", ""), re.I)][:10]
    warnings: list[str] = []
    if not tex_text:
        warnings.append("tex_source_missing; digest fell back to PDF text inventory")
    if not equation_cards:
        warnings.append("no_tex_equation_cards_detected")
    if not figure_cards:
        warnings.append("no_figure_cards_detected")
    return {
        "ok": True,
        "source_priority": [name for name, present in [("arxiv_api", (workdir / "api.xml").exists()), ("tex_source", bool(tex_text)), ("docling", (workdir / "docling.md").exists()), ("pdftotext", (workdir / "paper.layout.txt").exists())] if present],
        "metadata_card": {"title": metadata_title, "source_url": source_url, "tex_files": [str(p.relative_to(workdir)) for p in tex_files[:20]]},
        "abstract_card": {"text": abstract, "source": "tex_source" if extract_tex_abstract(tex_text) else "pdf_text" if abstract else "missing"},
        "section_cards": section_cards[:40],
        "equation_cards": equation_cards,
        "table_cards": table_cards,
        "figure_cards": figure_cards,
        "result_cards": extract_result_cards(section_cards if isinstance(section_cards, list) else [], table_cards, figure_cards if isinstance(figure_cards, list) else []),
        "implementation_cards": implementation_cards,
        "limitation_cards": limitation_cards,
        "resource_cards": resource_boundary,
        "quality_warnings": warnings,
        "files_used": sorted(set((files or {}).values())),
    }


def render_paper_digest_markdown(digest: dict[str, Any]) -> str:
    lines = ["# Paper digest", "", "Advisory evidence cards: use for synthesis, not as a final note.", "", f"- title: {digest.get('metadata_card', {}).get('title')}", f"- source: {digest.get('metadata_card', {}).get('source_url')}", ""]
    abstract = digest.get("abstract_card", {}).get("text")
    if abstract:
        lines += ["## Abstract card", abstract, ""]
    for heading, key in [("Section cards", "section_cards"), ("Equation cards", "equation_cards"), ("Table cards", "table_cards"), ("Figure cards", "figure_cards"), ("Result cards", "result_cards"), ("Implementation cards", "implementation_cards"), ("Limitation cards", "limitation_cards")]:
        lines.append(f"## {heading}")
        items = digest.get(key) or []
        if not items:
            lines.append("- none detected")
        for item in items[:12]:
            compact = json.dumps(item, ensure_ascii=False, sort_keys=True)
            lines.append(f"- {compact[:900]}")
        lines.append("")
    if digest.get("quality_warnings"):
        lines.append("## Quality warnings")
        for warning in digest["quality_warnings"]:
            lines.append(f"- {warning}")
    return "\n".join(lines).rstrip() + "\n"


def image_dimensions(path: Path) -> dict[str, Any]:
    try:
        data = path.read_bytes()[:32]
        if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
            return {"width": int.from_bytes(data[16:20], "big"), "height": int.from_bytes(data[20:24], "big")}
        try:
            from PIL import Image  # type: ignore
            with Image.open(path) as im:
                return {"width": im.width, "height": im.height}
        except Exception as exc:
            return {"dimensions_error": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:
        return {"dimensions_error": f"{type(exc).__name__}: {exc}"}


def render_pdf_first_page_to_png(source_pdf: Path, dest_png: Path) -> dict[str, Any]:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "message": str(exc)}
    try:
        dest_png.parent.mkdir(parents=True, exist_ok=True)
        doc = fitz.open(str(source_pdf))
        try:
            if doc.page_count < 1:
                return {"ok": False, "error": "empty_pdf"}
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            pix.save(str(dest_png))
            return {"ok": True, "page_count": doc.page_count, "rendered_page": 1, "render_width": pix.width, "render_height": pix.height}
        finally:
            doc.close()
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "message": str(exc)}


def localize_source_figures(root: Path, workdir: Path, image_slug: str) -> dict[str, Any]:
    slug = slugify(image_slug, max_len=96).lower()
    if not slug:
        return {"ok": False, "error": "missing_image_slug", "entries": [], "refused": []}
    source_root = (workdir / "source").resolve()
    # Structured paper raw notes do not retain copied figures/charts. Keep any
    # rendered assets inside the evidence workdir so closeout can clean them.
    evidence_images_root = (workdir / "localized_figures_assets").resolve()
    dest_dir = (evidence_images_root / slug).resolve()
    if not dest_dir.is_relative_to(evidence_images_root):
        return {"ok": False, "error": "image_dest_outside_evidence_dir", "entries": [], "refused": []}
    entries: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    used_names: set[str] = set()
    render_dir = workdir / "localized_figure_renders"
    for context in extract_figure_contexts(workdir):
        tex_file = Path(context["tex_file"])
        resolved, reason = _resolved_include_candidate(source_root, tex_file, context["include"])
        if not resolved:
            refused.append({"include": context.get("include"), "label": context.get("label"), "reason": reason})
            continue
        caption_slug = slugify(context.get("caption") or context.get("label") or resolved.stem, max_len=64).lower()
        source_sha = sha256_file(resolved)
        source_ext = resolved.suffix.lower()
        payload_path = resolved
        output_ext = source_ext
        localization_method = "copy"
        render_info: dict[str, Any] | None = None
        if source_ext == ".pdf":
            output_ext = ".png"
            localization_method = "pdf_render_first_page"
            render_tmp = render_dir / f"figure-{len(entries) + len(refused) + 1:02d}-{caption_slug}.png"
            render_info = render_pdf_first_page_to_png(resolved, render_tmp)
            if not render_info.get("ok"):
                refused.append({"include": context.get("include"), "label": context.get("label"), "reason": "pdf_render_failed", "render": render_info})
                continue
            payload_path = render_tmp
        base_name = f"figure-{len(entries) + 1:02d}-{caption_slug}{output_ext}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        payload_sha = sha256_file(payload_path)
        name = base_name
        suffix = 2
        reused_existing = False
        while True:
            dest = dest_dir / name
            if name not in used_names:
                if not dest.exists():
                    break
                try:
                    if sha256_file(dest) == payload_sha:
                        reused_existing = True
                        break
                except OSError:
                    pass
            stem = Path(base_name).stem
            name = f"{stem}-{suffix:02d}{output_ext}"
            suffix += 1
        used_names.add(name)
        if not dest.exists():
            shutil.copyfile(payload_path, dest)
        dest_rel = dest.relative_to(workdir).as_posix()
        dim = image_dimensions(dest)
        entry = {
            "label": context.get("label"),
            "caption": context.get("caption"),
            "include": context.get("include"),
            "source_rel": str(resolved.relative_to(source_root)) if resolved.is_relative_to(source_root) else str(resolved),
            "dest_rel": dest_rel,
            "bytes": dest.stat().st_size,
            "sha256": sha256_file(dest),
            "source_sha256": source_sha,
            "localization_method": localization_method,
            "reused_existing": reused_existing,
            "raw_note_policy": "temporary_inspection_only_do_not_embed_markdown_image",
        }
        if render_info:
            entry["render"] = render_info
        entry.update(dim)
        entries.append(entry)
    return {"ok": True, "image_slug": slug, "entries": entries, "refused": refused, "entry_count": len(entries), "refused_count": len(refused)}


def render_localized_figures_markdown(localized: dict[str, Any]) -> str:
    lines = ["# Localized figures", "", "Advisory sidecar: temporary inspection assets only. Do not paste Markdown image snippets into structured raw notes; write the figure/table-derived conclusion in prose.", ""]
    for entry in localized.get("entries") or []:
        method = entry.get("localization_method") or "copy"
        source_hash = f" source_sha256={entry.get('source_sha256')}" if entry.get("source_sha256") else ""
        lines.append(f"- {entry.get('label')}: `{entry.get('dest_rel')}` method={method} sha256={entry.get('sha256')}{source_hash} bytes={entry.get('bytes')} policy={entry.get('raw_note_policy')}")
    if localized.get("refused"):
        lines.append("\n## Refused candidates")
        for item in localized.get("refused") or []:
            lines.append(f"- {item.get('include')}: reason={item.get('reason')}, label={item.get('label')}")
    return "\n".join(lines).rstrip() + "\n"


def build_note_block_drafts(frontmatter: dict[str, Any], resource_draft: dict[str, Any] | None, localized_figures: dict[str, Any] | None, digest: dict[str, Any] | None) -> str:
    lines = ["# Note block drafts", "", "Advisory deterministic blocks only. These are not a final canonical note and do not replace LLM synthesis or closeout.", "", "## Frontmatter draft", "", yamlish(frontmatter), "", "## Resource status draft", ""]
    if resource_draft:
        for repo in resource_draft.get("github") or []:
            lines.append(f"- GitHub `{repo.get('repo')}`: status={repo.get('status')}, license={repo.get('license')}, fork={repo.get('fork')}, archived={repo.get('archived')}.")
        for kind, bucket in (resource_draft.get("hf") or {}).items():
            lines.append(f"- HF {kind}: status={bucket.get('status')}, query={bucket.get('query')}, count={bucket.get('count')}.")
    else:
        lines.append("- resource draft missing; inspect raw probe JSON manually.")
    lines.append("\n## Temporary figure/table inspection provenance draft")
    if localized_figures and localized_figures.get("entries"):
        for entry in localized_figures["entries"]:
            lines.append(f"- {entry.get('label')}: `{entry.get('dest_rel')}` sha256={entry.get('sha256')}; inspect if needed, then write only the prose conclusion in the raw note.")
    else:
        lines.append("- no temporary source-figure assets generated by sidecar.")
    lines.append("\n## Evidence-card reminder")
    if digest:
        lines.append(f"- equations={len(digest.get('equation_cards') or [])}, tables={len(digest.get('table_cards') or [])}, figures={len(digest.get('figure_cards') or [])}, warnings={len(digest.get('quality_warnings') or [])}.")
    lines.append("- Final summary, abstract, methodology, results, limitations, and questions must still be synthesized from evidence and verified by closeout.")
    return "\n".join(lines).rstrip() + "\n"


def next_raw_path(root: Path, title: str, now: dt.datetime | None = None) -> str:
    now = now or dt.datetime.now()
    yymm = now.strftime("%y%m")
    yymmdd = now.strftime("%y%m%d")
    month_dir = root / "raw" / "clip" / yymm
    used: set[int] = set()
    if month_dir.exists():
        for p in month_dir.glob(f"{yymmdd}*.md"):
            match = re.match(rf"{re.escape(yymmdd)}(\d{{2}})_", p.name)
            if match:
                used.add(int(match.group(1)))
    seq = (max(used) + 1) if used else 1
    if seq > 99:
        raise ValueError(f"raw clip sequence exhausted for {yymmdd}: max existing sequence is {max(used)}")
    return f"raw/clip/{yymm}/{yymmdd}{seq:02d}_{slugify(title)}.md"


def _source_identity(url: str, kind: str) -> dict[str, Any]:
    identity: dict[str, Any] = {"source_url": url, "source_id": url, "kind": kind}
    arxiv_id = arxiv_id_from_url(url)
    if arxiv_id:
        base_id = re.sub(r"v\d+$", "", arxiv_id)
        identity.update(
            {
                "arxiv_id": arxiv_id,
                "arxiv_id_base": base_id,
                "source_id": f"arxiv:{arxiv_id}",
                "canonical_abs_url": f"https://arxiv.org/abs/{arxiv_id}",
                "canonical_pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            }
        )
    return identity


def _duplicate_terms(title: str, url: str, kind: str) -> list[str]:
    terms = [title, url]
    identity = _source_identity(url, kind)
    for key in ["arxiv_id", "arxiv_id_base", "canonical_abs_url", "canonical_pdf_url"]:
        value = identity.get(key)
        if isinstance(value, str) and value not in terms:
            terms.append(value)
    return [term for term in terms if term]


def _duplicate_bucket(rel: str) -> str:
    if rel.startswith("raw/"):
        return "raw"
    if rel.startswith("_meta/"):
        return "meta"
    if rel == "log.md" or rel.endswith("/log.md"):
        return "log"
    return "compiled"


def scan_duplicate_hits(root: Path, title: str, url: str, kind: str, *, limit: int = 20) -> dict[str, list[dict[str, Any]]]:
    terms = _duplicate_terms(title, url, kind)
    buckets: dict[str, list[dict[str, Any]]] = {"raw": [], "compiled": [], "meta": [], "log": []}
    if not root.exists():
        return buckets
    lowered_terms = [(term, term.lower()) for term in terms]
    for path in sorted(root.rglob("*.md")):
        if sum(len(items) for items in buckets.values()) >= limit:
            break
        try:
            text = path.read_text(encoding="utf-8", errors="replace")[:1_000_000]
        except OSError:
            continue
        lower = text.lower()
        matched = [term for term, lowered in lowered_terms if lowered and lowered in lower]
        if not matched:
            continue
        rel = path.relative_to(root).as_posix()
        buckets[_duplicate_bucket(rel)].append({"path": rel, "matched_terms": matched[:5]})
    return buckets


def _load_pending_count(path: Path) -> int | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, dict):
        pending = data.get("pending")
        if isinstance(pending, list):
            return len(pending)
        pending_count = data.get("pending_count")
        if isinstance(pending_count, int):
            return pending_count
    return None


def read_queue_status(state_dir: Path | None) -> dict[str, Any]:
    if state_dir is None:
        return {"checked": False, "reason": "state_dir_not_provided"}
    wiki_path = state_dir / "pending_wiki_integration.json"
    native_path = state_dir / "pending_native_refresh.json"
    wiki_count = _load_pending_count(wiki_path)
    native_count = _load_pending_count(native_path)
    return {
        "checked": True,
        "state_dir": str(state_dir),
        "wiki_ledger_path": str(wiki_path),
        "native_ledger_path": str(native_path),
        "wiki_pending_count": wiki_count or 0,
        "native_pending_count": native_count or 0,
        "wiki_ledger_exists": wiki_path.exists(),
        "native_ledger_exists": native_path.exists(),
    }


def build_raw_fast_preflight(
    root: Path,
    title: str,
    url: str,
    kind: str,
    *,
    workdir: Path,
    state_dir: Path | None = None,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    now = now or dt.datetime.now().astimezone()
    duplicate_hits = scan_duplicate_hits(root, title, url, kind)
    duplicate_summary = {bucket: len(items) for bucket, items in duplicate_hits.items()}
    return {
        "ok": True,
        "timestamp": now.isoformat(timespec="minutes"),
        "root": str(root),
        "workdir": str(workdir),
        "kind": kind,
        "title": title,
        "source_identity": _source_identity(url, kind),
        "next_raw_path": next_raw_path(root, title, now=now),
        "duplicate_terms": _duplicate_terms(title, url, kind),
        "duplicate_hits": duplicate_hits,
        "duplicate_summary": duplicate_summary,
        "queue_status": read_queue_status(state_dir),
    }


def render_preflight_markdown(preflight: dict[str, Any]) -> str:
    lines = [
        "# Raw-fast preflight",
        "",
        f"- title: {preflight.get('title')}",
        f"- kind: {preflight.get('kind')}",
        f"- next_raw_path: `{preflight.get('next_raw_path')}`",
    ]
    duplicate_summary = preflight.get("duplicate_summary") or {}
    lines.append(f"- duplicate_hits: raw={duplicate_summary.get('raw', 0)}, compiled={duplicate_summary.get('compiled', 0)}, meta={duplicate_summary.get('meta', 0)}, log={duplicate_summary.get('log', 0)}")
    queue = preflight.get("queue_status") or {}
    if queue.get("checked"):
        lines.append(f"- queues: wiki_pending={queue.get('wiki_pending_count', 0)}, native_pending={queue.get('native_pending_count', 0)}")
    else:
        lines.append(f"- queues: not checked ({queue.get('reason')})")
    lines.append("\n## Duplicate hits")
    for bucket, hits in (preflight.get("duplicate_hits") or {}).items():
        lines.append(f"\n### {bucket}")
        if not hits:
            lines.append("- none")
        for hit in hits[:10]:
            lines.append(f"- `{hit.get('path')}` — {', '.join(hit.get('matched_terms') or [])}")
    return "\n".join(lines).rstrip() + "\n"


def _brief_cards_from_digest(digest: dict[str, Any] | None, *, max_cards: int = 12) -> list[dict[str, Any]]:
    if not digest:
        return []
    cards: list[dict[str, Any]] = []
    for bucket, label in [
        ("abstract_cards", "abstract"),
        ("section_cards", "section"),
        ("equation_cards", "equation"),
        ("result_cards", "result"),
        ("table_cards", "table"),
        ("figure_cards", "figure"),
    ]:
        for card in digest.get(bucket) or []:
            if len(cards) >= max_cards:
                return cards
            text = compact_ws(str(card.get("text") or card.get("caption") or card.get("section") or card))[:500]
            cards.append({"kind": label, "text": text, "source_bucket": bucket})
    return cards


def build_agent_brief(bundle: dict[str, Any], *, preflight: dict[str, Any] | None, digest: dict[str, Any] | None, resource_boundary: dict[str, Any] | None) -> dict[str, Any]:
    files = bundle.get("files") or {}
    duplicate_summary = (preflight or {}).get("duplicate_summary") or {}
    resource_boundary = resource_boundary or {}
    return {
        "ok": True,
        "protected_anchors": {
            "title": bundle.get("title_guess") or (preflight or {}).get("title"),
            "kind": bundle.get("kind") or (preflight or {}).get("kind"),
            "source_url": bundle.get("source_url"),
            "next_raw_path": bundle.get("next_raw_path") or (preflight or {}).get("next_raw_path"),
            "warnings": bundle.get("warnings") or [],
        },
        "duplicate_summary": duplicate_summary,
        "queue_status": (preflight or {}).get("queue_status") or {"checked": False},
        "evidence_cards": _brief_cards_from_digest(digest),
        "resource_caveats": {
            "status": "standardized" if resource_boundary else "missing",
            "github_count": len(resource_boundary.get("github") or []),
            "hf_statuses": {key: value.get("status") for key, value in (resource_boundary.get("hf") or {}).items()},
            "review_required": bool(resource_boundary.get("review_required")),
            "manual_reason": resource_boundary.get("manual_reason"),
            "summary": resource_status_summary(resource_boundary),
            "policy": "Keep resource/probe/link-health detail out of the raw note body and frontmatter except canonical source provenance.",
        },
        "source_refs": {
            "evidence_bundle": "evidence_bundle.json",
            "paper_digest": files.get("paper_digest") or "paper_digest.json",
            "resource_boundary": files.get("resource_boundary_draft") or "resource_boundary_draft.json",
            "raw_text": files.get("raw_text"),
            "layout_text": files.get("layout_text"),
        },
        "agent_actions": [
            "Read this brief first; reopen cited sidecars only for uncertain claims.",
            "Write or patch the canonical raw note under next_raw_path only after duplicate review.",
            "Run raw-fast closeout; do not run native refresh while wiki integration is pending.",
        ],
    }


def render_agent_brief_markdown(brief: dict[str, Any]) -> str:
    anchors = brief.get("protected_anchors") or {}
    lines = ["# Agent brief", "", f"- title: {anchors.get('title')}", f"- source_url: {anchors.get('source_url')}", f"- next_raw_path: `{anchors.get('next_raw_path')}`"]
    dup = brief.get("duplicate_summary") or {}
    lines.append(f"- duplicate summary: raw={dup.get('raw', 0)}, compiled={dup.get('compiled', 0)}, meta={dup.get('meta', 0)}, log={dup.get('log', 0)}")
    resource = brief.get("resource_caveats") or {}
    lines.append(f"- resource status: {resource.get('status')} — {resource.get('policy')}")
    lines.append("\n## Evidence cards")
    for card in brief.get("evidence_cards") or []:
        lines.append(f"- **{card.get('kind')}**: {card.get('text')}")
    if not brief.get("evidence_cards"):
        lines.append("- no compact evidence cards; inspect `paper.raw.txt` or `paper_digest.json`.")
    lines.append("\n## Next actions")
    for action in brief.get("agent_actions") or []:
        lines.append(f"- {action}")
    return "\n".join(lines).rstrip() + "\n"


def build_evidence_report(bundle: dict[str, Any], *, preflight: dict[str, Any] | None, digest: dict[str, Any] | None, resource_boundary: dict[str, Any] | None) -> dict[str, Any]:
    digest = digest or {}
    resource_boundary = resource_boundary or {}
    return {
        "ok": True,
        "status": "standardized",
        "bundle_summary": {
            "kind": bundle.get("kind"),
            "source_url": bundle.get("source_url"),
            "title_guess": bundle.get("title_guess"),
            "next_raw_path": bundle.get("next_raw_path") or (preflight or {}).get("next_raw_path"),
            "warnings": bundle.get("warnings") or [],
        },
        "preflight": {
            "duplicate_summary": (preflight or {}).get("duplicate_summary") or {},
            "queue_status": (preflight or {}).get("queue_status") or {"checked": False},
        },
        "paper_digest": {
            "status": "standardized" if digest else "missing",
            "abstract_cards": len(digest.get("abstract_cards") or []),
            "section_cards": len(digest.get("section_cards") or []),
            "equation_cards": len(digest.get("equation_cards") or []),
            "result_cards": len(digest.get("result_cards") or []),
            "warnings": len(digest.get("quality_warnings") or []),
        },
        "resource_boundary": {
            "status": "standardized" if resource_boundary else "missing",
            "github_count": len(resource_boundary.get("github") or []),
            "github": resource_boundary.get("github") or [],
            "project_pages": resource_boundary.get("project_pages") or [],
            "hf_statuses": {key: value.get("status") for key, value in (resource_boundary.get("hf") or {}).items()},
            "review_required": bool(resource_boundary.get("review_required")),
            "manual_reason": resource_boundary.get("manual_reason"),
            "summary": resource_status_summary(resource_boundary),
            "policy": "External resource/probe details stay in evidence/closeout reports, not raw-note body.",
        },
    }


def render_evidence_report_markdown(report: dict[str, Any]) -> str:
    summary = report.get("bundle_summary") or {}
    preflight = report.get("preflight") or {}
    digest = report.get("paper_digest") or {}
    resource = report.get("resource_boundary") or {}
    return (
        "# Standardized evidence report\n\n"
        f"- title: {summary.get('title_guess')}\n"
        f"- source_url: {summary.get('source_url')}\n"
        f"- next_raw_path: `{summary.get('next_raw_path')}`\n"
        f"- duplicate_summary: {json.dumps(preflight.get('duplicate_summary') or {}, ensure_ascii=False, sort_keys=True)}\n"
        f"- paper_digest: status={digest.get('status')}, abstract_cards={digest.get('abstract_cards')}, section_cards={digest.get('section_cards')}, equation_cards={digest.get('equation_cards')}, result_cards={digest.get('result_cards')}\n"
        f"- resource_boundary: status={resource.get('status')}, github_count={resource.get('github_count')}, hf_statuses={json.dumps(resource.get('hf_statuses') or {}, ensure_ascii=False, sort_keys=True)}\n"
        f"- policy: {resource.get('policy')}\n"
    )


def build_note_candidate(frontmatter: dict[str, Any], *, preflight: dict[str, Any] | None, digest: dict[str, Any] | None) -> dict[str, Any]:
    title = frontmatter.get("title") or (preflight or {}).get("title") or "Untitled raw-fast note"
    cards = _brief_cards_from_digest(digest, max_cards=8)
    sections = [
        ("一句话总结", "<!-- advisory: synthesize one source-grounded sentence; do not copy URLs or resource status. -->"),
        ("论文摘要（中文）", "<!-- advisory: summarize the abstract and problem setting from evidence cards. -->"),
        ("Motivation", "<!-- advisory: explain the research gap and why this paper matters. -->"),
        ("Methodology", "<!-- advisory: integrate objectives, formulas, architecture, and method evidence in prose. -->"),
        ("关键实验结果 / 作者结论", "<!-- advisory: write conclusions supported by result/table/figure cards, not copied tables. -->"),
        ("对未来研究的启发", "<!-- advisory: list durable research implications. -->"),
        ("可能的局限", "<!-- advisory: list limitations that are explicit or strongly implied by the source. -->"),
        ("可继续追问的问题", "<!-- advisory: list follow-up questions for future reading. -->"),
    ]
    return {
        "ok": True,
        "advisory": True,
        "title": title,
        "next_raw_path": (preflight or {}).get("next_raw_path"),
        "frontmatter": frontmatter,
        "evidence_card_count": len(cards),
        "sections": [{"heading": heading, "placeholder": placeholder} for heading, placeholder in sections],
        "policy": "Candidate is deterministic scaffolding only; agent must synthesize scientific prose and run raw-fast verification.",
    }


def render_note_candidate_markdown(candidate: dict[str, Any]) -> str:
    lines = [yamlish(candidate.get("frontmatter") or {}), ""]
    lines.append("<!-- advisory: guarded candidate; fill prose from source evidence, then run raw-fast verifier. -->")
    for section in candidate.get("sections") or []:
        lines.extend(["", f"## {section.get('heading')}", "", section.get("placeholder") or ""])
    return "\n".join(lines).rstrip() + "\n"


def build_agent_handoff(bundle: dict[str, Any], *, brief: dict[str, Any], evidence_report: dict[str, Any], note_candidate: dict[str, Any], closeout_args: dict[str, Any] | None = None) -> dict[str, Any]:
    raw_resource = evidence_report.get("resource_boundary")
    resource = raw_resource if isinstance(raw_resource, dict) else {}
    resource_review_required = bool(resource.get("review_required"))
    manual_paths: list[str] = []
    source_refs = brief.get("source_refs") or {}
    source_refs["note_candidate"] = "note_candidate.md"
    source_refs["evidence_report"] = "evidence_report.md"
    source_refs["agent_brief"] = "agent_brief.md"
    return {
        "ok": True,
        "status": "ready" if not resource_review_required else "manual_required",
        "protected_anchors": brief.get("protected_anchors") or {},
        "duplicate_summary": brief.get("duplicate_summary") or {},
        "queue_status": brief.get("queue_status") or {},
        "evidence_cards": brief.get("evidence_cards") or [],
        "resource_status_summary": resource.get("summary") or resource_status_summary(resource),
        "resource_review_required": resource_review_required,
        "manual_reason": resource.get("manual_reason") if resource_review_required else None,
        "manual_reference_paths": manual_paths,
        "source_refs": source_refs,
        "closeout_args": closeout_args,
        "closeout_args_path": None,
        "agent_actions": [
            "Read this handoff first; reopen referenced sidecars only for uncertain scientific claims or reported failures.",
            "Do not run broad GitHub/HF/project discovery when resource_review_required is false.",
            "Write the canonical raw note under protected_anchors.next_raw_path, then run closeout with generated closeout_args when available.",
        ],
        "note_candidate": {"path": "note_candidate.md", "advisory": bool(note_candidate.get("advisory", True))},
    }


def render_agent_handoff_markdown(handoff: dict[str, Any]) -> str:
    anchors = handoff.get("protected_anchors") or {}
    lines = [
        "# Raw-fast agent handoff",
        "",
        f"- status: {handoff.get('status')}",
        f"- title: {anchors.get('title')}",
        f"- source_url: {anchors.get('source_url')}",
        f"- next_raw_path: `{anchors.get('next_raw_path')}`",
        f"- resource_review_required: {handoff.get('resource_review_required')}",
        f"- resource_status_summary: {handoff.get('resource_status_summary')}",
    ]
    dup = handoff.get("duplicate_summary") or {}
    lines.append(f"- duplicate summary: raw={dup.get('raw', 0)}, compiled={dup.get('compiled', 0)}, meta={dup.get('meta', 0)}, log={dup.get('log', 0)}")
    if handoff.get("manual_reason"):
        lines.append("\n## Manual follow-up")
        lines.append(f"- reason: {json.dumps(handoff.get('manual_reason'), ensure_ascii=False, sort_keys=True)}")
        for path in handoff.get("manual_reference_paths") or []:
            lines.append(f"- read if needed: `{path}`")
    lines.append("\n## Source refs")
    for key, value in sorted((handoff.get("source_refs") or {}).items()):
        if value:
            lines.append(f"- {key}: `{value}`")
    lines.append("\n## Evidence cards")
    for card in handoff.get("evidence_cards") or []:
        lines.append(f"- **{card.get('kind')}**: {card.get('text')}")
    if not handoff.get("evidence_cards"):
        lines.append("- no compact evidence cards; inspect cited sidecars only for uncertain scientific claims.")
    lines.append("\n## Next actions")
    for action in handoff.get("agent_actions") or []:
        lines.append(f"- {action}")
    return "\n".join(lines).rstrip() + "\n"


def write_agent_automation_sidecars(
    workdir: Path,
    files: dict[str, str],
    bundle: dict[str, Any],
    *,
    frontmatter: dict[str, Any],
    preflight: dict[str, Any],
    digest: dict[str, Any] | None,
    resource_boundary: dict[str, Any] | None,
) -> dict[str, Any]:
    write_json(workdir / "raw_fast_preflight.json", preflight)
    write_text(workdir / "raw_fast_preflight.md", render_preflight_markdown(preflight))
    files["raw_fast_preflight"] = "raw_fast_preflight.json"
    files["raw_fast_preflight_markdown"] = "raw_fast_preflight.md"

    brief = build_agent_brief(bundle, preflight=preflight, digest=digest, resource_boundary=resource_boundary)
    write_json(workdir / "agent_brief.json", brief)
    write_text(workdir / "agent_brief.md", render_agent_brief_markdown(brief))
    files["agent_brief"] = "agent_brief.json"
    files["agent_brief_markdown"] = "agent_brief.md"

    report = build_evidence_report(bundle, preflight=preflight, digest=digest, resource_boundary=resource_boundary)
    write_json(workdir / "evidence_report.json", report)
    write_text(workdir / "evidence_report.md", render_evidence_report_markdown(report))
    files["evidence_report"] = "evidence_report.json"
    files["evidence_report_markdown"] = "evidence_report.md"

    candidate = build_note_candidate(frontmatter, preflight=preflight, digest=digest)
    write_json(workdir / "note_candidate.json", candidate)
    write_text(workdir / "note_candidate.md", render_note_candidate_markdown(candidate))
    files["note_candidate"] = "note_candidate.json"
    files["note_candidate_markdown"] = "note_candidate.md"

    handoff = build_agent_handoff(bundle, brief=brief, evidence_report=report, note_candidate=candidate)
    write_json(workdir / "agent_handoff.json", handoff)
    write_text(workdir / "agent_handoff.md", render_agent_handoff_markdown(handoff))
    files["agent_handoff"] = "agent_handoff.json"
    files["agent_handoff_markdown"] = "agent_handoff.md"

    return {
        "ok": True,
        "status": "written",
        "files": {
            key: files[key]
            for key in [
                "raw_fast_preflight",
                "agent_brief",
                "agent_brief_markdown",
                "evidence_report",
                "evidence_report_markdown",
                "note_candidate",
                "note_candidate_markdown",
                "agent_handoff",
                "agent_handoff_markdown",
            ]
        },
        "brief_cards": len(brief.get("evidence_cards") or []),
        "handoff_status": handoff.get("status"),
        "resource_review_required": bool(handoff.get("resource_review_required")),
    }


def scan_secrets(workdir: Path) -> dict[str, Any]:
    hits: list[dict[str, Any]] = []
    placeholder_hits: list[dict[str, Any]] = []
    for path in sorted(workdir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in STRICT_SECRET_PATTERNS.items():
            for match in pattern.finditer(text):
                token = match.group(0)
                redacted = token[:6] + "..." + token[-4:] if len(token) > 12 else "[REDACTED]"
                # Common AWS documentation example should not be copied, but keep it as placeholder/example.
                if token == ("AKIA" + "IOSFODNN7EXAMPLE"):
                    placeholder_hits.append({"file": str(path.relative_to(workdir)), "pattern": name, "match": redacted})
                else:
                    hits.append({"file": str(path.relative_to(workdir)), "pattern": name, "match": redacted})
        if "PASTE_AWS_ACCESS_KEY_ID_HERE" in text:
            placeholder_hits.append({"file": str(path.relative_to(workdir)), "pattern": "placeholder", "match": "PASTE_AWS_ACCESS_KEY_ID_HERE"})
    return {"strict_secret_hits": hits, "placeholder_hits": placeholder_hits}


def build_frontmatter(title: str, source: str, kind: str) -> dict[str, Any]:
    return {
        "title": title,
        "created": dt.datetime.now().strftime("%Y-%m-%d"),
        "updated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "type": "raw-note",
        "domain": "paper",
        "source": source,
        "capture_route": f"raw-fast evidence bundle ({kind})",
        "captured": dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z (%z)"),
    }


def yamlish(frontmatter: dict[str, Any]) -> str:
    lines = ["---"]
    for key, value in frontmatter.items():
        sval = str(value).replace('"', '\\"')
        lines.append(f'{key}: "{sval}"')
    lines.append("---")
    return "\n".join(lines)


def build_note_skeleton(frontmatter: dict[str, Any]) -> str:
    return yamlish(frontmatter) + """

## 一句话总结


## 论文摘要（中文）


## Motivation


## Methodology


## 关键实验结果 / 作者结论


## 对未来研究的启发


## 可能的局限


## 可继续追问的问题

"""


def _write_pdf_resource_boundary_sidecars(
    workdir: Path,
    files: dict[str, str],
    resource_probe: dict[str, Any],
    *,
    title: str,
    url: str,
    pdfinfo: dict[str, Any],
    secret_scan: dict[str, Any],
) -> dict[str, Any]:
    draft = summarize_resource_boundary(
        resource_probe,
        metadata={"title": title, "source_url": url},
        pdf_info=pdfinfo,
        secret_scan=secret_scan,
    )
    write_json(workdir / "resource_boundary_draft.json", draft)
    write_text(workdir / "resource_boundary_draft.md", render_resource_boundary_markdown(draft))
    files["resource_boundary_draft"] = "resource_boundary_draft.json"
    files["resource_boundary_draft_markdown"] = "resource_boundary_draft.md"
    return {
        "ok": True,
        "status": "written",
        "github_count": len(draft.get("github") or []),
        "hf_statuses": {key: value.get("status") for key, value in (draft.get("hf") or {}).items()},
    }


def _write_pdf_localized_figure_sidecars(
    root: Path,
    workdir: Path,
    files: dict[str, str],
    *,
    title: str,
    image_slug: str | None,
) -> dict[str, Any]:
    localized = localize_source_figures(root, workdir, image_slug or slugify(title).lower())
    write_json(workdir / "localized_figures.json", localized)
    write_text(workdir / "localized_figures.md", render_localized_figures_markdown(localized))
    files["localized_figures"] = "localized_figures.json"
    files["localized_figures_markdown"] = "localized_figures.md"
    return localized


def _write_pdf_digest_sidecars(
    workdir: Path,
    files: dict[str, str],
    *,
    title: str,
    url: str,
    resource_probe: dict[str, Any],
) -> dict[str, Any]:
    digest = build_paper_digest(workdir, title, url, resource_probe=resource_probe, files=files)
    write_json(workdir / "paper_digest.json", digest)
    write_text(workdir / "paper_digest.md", render_paper_digest_markdown(digest))
    files["paper_digest"] = "paper_digest.json"
    files["paper_digest_markdown"] = "paper_digest.md"
    return {
        "ok": True,
        "status": "written",
        "equations": len(digest.get("equation_cards") or []),
        "tables": len(digest.get("table_cards") or []),
        "figures": len(digest.get("figure_cards") or []),
        "warnings": len(digest.get("quality_warnings") or []),
    }


def _write_pdf_note_block_drafts(
    workdir: Path,
    files: dict[str, str],
    *,
    frontmatter: dict[str, Any],
    resource_draft_payload: dict[str, Any] | None,
    localized_payload: dict[str, Any] | None,
    digest_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    draft_text = build_note_block_drafts(frontmatter, resource_draft_payload, localized_payload, digest_payload)
    write_text(workdir / "note_block_drafts.md", draft_text)
    files["note_block_drafts"] = "note_block_drafts.md"
    return {"ok": True, "status": "written", "bytes": len(draft_text.encode("utf-8"))}


def _write_pdf_enrichment_sidecars(
    *,
    root: Path,
    workdir: Path,
    files: dict[str, str],
    timings: TimingRecorder,
    title: str,
    url: str,
    pdfinfo: dict[str, Any],
    resource_probe: dict[str, Any],
    secret_scan: dict[str, Any],
    frontmatter: dict[str, Any],
    resource_draft: bool,
    paper_digest: bool,
    localize_figures: bool,
    image_slug: str | None,
) -> dict[str, Any]:
    resource_draft_payload: dict[str, Any] | None = None
    localized_payload: dict[str, Any] | None = None
    digest_payload: dict[str, Any] | None = None

    if resource_draft or paper_digest:
        timings.record(
            "resource_boundary_draft",
            _write_pdf_resource_boundary_sidecars,
            workdir,
            files,
            resource_probe,
            title=title,
            url=url,
            pdfinfo=pdfinfo,
            secret_scan=secret_scan,
        )
        resource_draft_payload = json.loads((workdir / "resource_boundary_draft.json").read_text(encoding="utf-8"))

    if localize_figures:
        localized_payload = timings.record(
            "localize_figures",
            _write_pdf_localized_figure_sidecars,
            root,
            workdir,
            files,
            title=title,
            image_slug=image_slug,
        )

    if paper_digest:
        timings.record(
            "paper_digest",
            _write_pdf_digest_sidecars,
            workdir,
            files,
            title=title,
            url=url,
            resource_probe=resource_probe,
        )
        digest_payload = json.loads((workdir / "paper_digest.json").read_text(encoding="utf-8"))
        timings.record(
            "note_block_drafts",
            _write_pdf_note_block_drafts,
            workdir,
            files,
            frontmatter=frontmatter,
            resource_draft_payload=resource_draft_payload,
            localized_payload=localized_payload,
            digest_payload=digest_payload,
        )

    return {
        "resource_draft": resource_draft_payload,
        "localized_figures": localized_payload,
        "paper_digest": digest_payload,
    }


def process_pdf(
    url: str,
    kind: str,
    root: Path,
    workdir: Path,
    pdf_backend: str,
    strict_pdf_backend: bool,
    probes: list[str],
    timeout: int,
    timings: TimingRecorder | None = None,
    paper_digest: bool = False,
    resource_draft: bool = False,
    localize_figures: bool = False,
    image_slug: str | None = None,
    state_dir: Path | None = None,
    resource_health: str = "direct",
) -> dict[str, Any]:
    timings = timings or TimingRecorder()
    warnings: list[str] = []
    files: dict[str, str] = {}
    pdf_path = workdir / "paper.pdf"
    fetch = timings.record("fetch_pdf", fetch_url_to_file, url, pdf_path, timeout)
    files["pdf"] = "paper.pdf"
    if not fetch.get("ok"):
        payload = {"ok": False, "kind": kind, "workdir": str(workdir), "fetch": fetch, "warnings": warnings}
        attach_timings(payload, timings)
        write_json(workdir / "evidence_bundle.json", payload)
        return payload

    pdfinfo = timings.record("pdfinfo", run_pdfinfo, pdf_path, workdir / "pdfinfo.txt")
    files["pdfinfo"] = "pdfinfo.txt"
    layout = timings.record("pdftotext_layout", run_pdftotext, pdf_path, workdir / "paper.layout.txt", layout=True)
    raw = timings.record("pdftotext_raw", run_pdftotext, pdf_path, workdir / "paper.raw.txt", layout=False)
    files["layout_text"] = "paper.layout.txt"
    files["raw_text"] = "paper.raw.txt"
    links = timings.record("pdf_links", extract_pdf_links, pdf_path, workdir / "links.json")
    files["links"] = "links.json"
    if not links.get("ok"):
        warnings.append(f"PyMuPDF link extraction failed: {links.get('error')}: {links.get('message')}")

    docling_result: dict[str, Any] | None = None
    if pdf_backend in {"docling", "auto"}:
        docling_result = timings.record("docling", run_docling, pdf_path, workdir, strict=strict_pdf_backend, timeout=timeout)
        if (workdir / "docling.md").exists():
            files["docling_markdown"] = "docling.md"
        if (workdir / "docling.json").exists():
            files["docling_json"] = "docling.json"
        if not docling_result.get("ok"):
            message = f"Docling extraction failed: {docling_result.get('error')}: {docling_result.get('message')}"
            warnings.append(message)
            if strict_pdf_backend:
                payload = {"ok": False, "kind": kind, "workdir": str(workdir), "fetch": fetch, "pdf_backend_requested": pdf_backend, "docling": docling_result, "warnings": warnings}
                attach_timings(payload, timings)
                write_json(workdir / "evidence_bundle.json", payload)
                return payload
    if pdf_backend == "pdftotext":
        docling_result = timings.record("docling", lambda: {"ok": False, "skipped": True, "reason": "pdf_backend=pdftotext"})

    text = ""
    for candidate in [workdir / "docling.md", workdir / "paper.layout.txt", workdir / "paper.raw.txt"]:
        if candidate.exists():
            ctext = read_text(candidate)
            if ctext.strip():
                text += "\n" + ctext
    title = title_from_text(text) or Path(urllib.parse.urlparse(url).path).stem
    inventory = timings.record("inventory", lambda: {"ok": True, "sections": section_inventory(text), "figures": figure_table_inventory(text)})
    sections = inventory["sections"]
    figures = inventory["figures"]
    write_json(workdir / "section_inventory.json", {"sections": sections})
    write_json(workdir / "figure_table_inventory.json", {"items": figures})
    files["section_inventory"] = "section_inventory.json"
    files["figure_table_inventory"] = "figure_table_inventory.json"

    tex_files_for_resources, tex_text_for_resources = read_tex_bundle(workdir)
    resource_text = text + ("\n" + tex_text_for_resources if tex_text_for_resources else "")
    resource_probe = timings.record("resource_probe", build_resource_probe, resource_text, links, probes, health_mode=resource_health, timeout=min(timeout, 12))
    write_json(workdir / "resource_probe.json", resource_probe)
    files["resource_probe"] = "resource_probe.json"

    secret_scan = timings.record("secret_scan", scan_secrets, workdir)
    write_json(workdir / "secret_scan.json", secret_scan)
    files["secret_scan"] = "secret_scan.json"

    fm = timings.record("candidate_frontmatter", build_frontmatter, title, url, kind)
    write_json(workdir / "candidate_frontmatter.json", fm)
    skeleton = timings.record("note_skeleton", build_note_skeleton, fm)
    write_text(workdir / "note_skeleton.md", skeleton)
    files["candidate_frontmatter"] = "candidate_frontmatter.json"
    files["note_skeleton"] = "note_skeleton.md"

    enrichment = _write_pdf_enrichment_sidecars(
        root=root,
        workdir=workdir,
        files=files,
        timings=timings,
        title=title,
        url=url,
        pdfinfo=pdfinfo,
        resource_probe=resource_probe,
        secret_scan=secret_scan,
        frontmatter=fm,
        resource_draft=resource_draft,
        paper_digest=paper_digest,
        localize_figures=localize_figures,
        image_slug=image_slug,
    )
    preflight = timings.record("raw_fast_preflight", build_raw_fast_preflight, root, title, url, kind, workdir=workdir, state_dir=state_dir)

    payload = {
        "ok": True,
        "kind": kind,
        "source_url": url,
        "workdir": str(workdir),
        "pdf_backend_requested": pdf_backend,
        "pdf_backend_effective": "docling" if docling_result and docling_result.get("ok") and pdf_backend in {"docling", "auto"} else "pdftotext",
        "fetch": fetch,
        "pdfinfo": pdfinfo,
        "pdftotext": {"layout": layout, "raw": raw},
        "docling": docling_result,
        "title_guess": title,
        "files": files,
        "next_raw_path": preflight["next_raw_path"],
        "preflight": preflight,
        "warnings": warnings,
    }
    payload["agent_automation"] = timings.record(
        "agent_automation_sidecars",
        write_agent_automation_sidecars,
        workdir,
        files,
        payload,
        frontmatter=fm,
        preflight=preflight,
        digest=enrichment.get("paper_digest"),
        resource_boundary=enrichment.get("resource_draft"),
    )
    attach_timings(payload, timings)
    write_json(workdir / "evidence_bundle.json", payload)
    return payload


def process_arxiv(
    url: str,
    root: Path,
    workdir: Path,
    pdf_backend: str,
    strict_pdf_backend: bool,
    probes: list[str],
    timeout: int,
    timings: TimingRecorder | None = None,
    paper_digest: bool = False,
    resource_draft: bool = False,
    localize_figures: bool = False,
    image_slug: str | None = None,
    state_dir: Path | None = None,
    resource_health: str = "direct",
) -> dict[str, Any]:
    timings = timings or TimingRecorder()
    arxiv_id = arxiv_id_from_url(url)
    if not arxiv_id:
        return process_pdf(url, "direct-pdf", root, workdir, pdf_backend, strict_pdf_backend, probes, timeout, timings, paper_digest=paper_digest, resource_draft=resource_draft, localize_figures=localize_figures, image_slug=image_slug, state_dir=state_dir, resource_health=resource_health)
    api_url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"id_list": arxiv_id})
    abs_url = f"https://arxiv.org/abs/{arxiv_id}"
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    eprint_url = f"https://arxiv.org/e-print/{arxiv_id}"
    api = timings.record("arxiv_api", fetch_text, api_url, timeout)
    abs_page = timings.record("arxiv_abs_page", fetch_text, abs_url, timeout)
    write_text(workdir / "api.xml", api.get("text") or "")
    write_text(workdir / "abs.html", abs_page.get("text") or "")
    eprint = timings.record("arxiv_eprint_fetch", fetch_url_to_file, eprint_url, workdir / "eprint.tar", timeout)
    source_extract = timings.record("arxiv_source_extract", lambda: safe_extract_tar(workdir / "eprint.tar", workdir / "source") if eprint.get("ok") else {"ok": False, "errors": ["eprint fetch failed"], "extracted_count": 0})
    tex_files = [str(p.relative_to(workdir / "source")) for p in (workdir / "source").rglob("*.tex")] if (workdir / "source").exists() else []
    tex_text = "\n".join((workdir / "source" / p).read_text(encoding="utf-8", errors="replace")[:200000] for p in tex_files[:20])
    tex_figures = figure_table_inventory(tex_text)
    write_json(workdir / "source_inventory.json", {"eprint": eprint, "source_extract": source_extract, "tex_files": tex_files[:200], "figure_table_items": tex_figures})
    payload = process_pdf(pdf_url, "arxiv", root, workdir, pdf_backend, strict_pdf_backend, probes, timeout, timings, paper_digest=paper_digest, resource_draft=resource_draft, localize_figures=localize_figures, image_slug=image_slug, state_dir=state_dir, resource_health=resource_health)
    payload["arxiv"] = {"id": arxiv_id, "api_url": api_url, "abs_url": abs_url, "pdf_url": pdf_url, "eprint_url": eprint_url, "api_ok": api.get("ok"), "abs_ok": abs_page.get("ok"), "eprint_ok": eprint.get("ok"), "source_extract": source_extract, "tex_files": tex_files[:200]}
    payload.setdefault("files", {})["api"] = "api.xml"
    payload.setdefault("files", {})["abs_html"] = "abs.html"
    payload.setdefault("files", {})["source_inventory"] = "source_inventory.json"
    attach_timings(payload, timings)
    write_json(workdir / "evidence_bundle.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an llm-wiki raw-fast evidence bundle without writing the wiki")
    parser.add_argument("--url", required=True)
    parser.add_argument("--kind", choices=["auto", "direct-pdf", "arxiv"], default="auto")
    parser.add_argument("--root", type=Path, default=DEFAULT_WIKI_ROOT)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, default=None, help="Optional native state dir to read pending queue counts for preflight; never mutated by this command")
    parser.add_argument("--probe", action="append", default=None, choices=["arxiv", "doi", "none"], help="Probe class to run; repeatable. Default is arxiv+doi only. Use --probe none for offline tests.")
    parser.add_argument("--pdf-backend", choices=["docling", "pdftotext", "auto"], default="docling")
    parser.add_argument("--strict-pdf-backend", action="store_true", help="Fail if the requested structured PDF backend fails instead of falling back to pdftotext evidence")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--paper-digest", action="store_true", help="Write paper_digest.json/md and deterministic note block drafts as temp sidecars")
    parser.add_argument("--resource-draft", action="store_true", help="Write resource_boundary_draft.json/md from completed probes")
    parser.add_argument("--localize-figures", action="store_true", help="Render/copy safe source figures into the evidence workdir for temporary inspection; do not embed them in structured raw notes")
    parser.add_argument("--resource-health", choices=["direct", "none"], default="direct", help="Exact-link health mode for source-exposed GitHub/HF/project URLs. --probe none disables network health.")
    parser.add_argument("--image-slug", default=None, help="Required with --localize-figures; slug under the workdir-local localized_figures_assets/ directory")
    return parser.parse_args()


def main() -> int:
    timings = TimingRecorder()
    args = parse_args()
    root = args.root.resolve()
    workdir = args.workdir.resolve()
    state_dir = args.state_dir.resolve() if args.state_dir else None
    if workdir == root or workdir.is_relative_to(root):
        print_json({"ok": False, "stage": "preflight", "error": "workdir_inside_wiki_root", "root": str(root), "workdir": str(workdir), "timings": timings.snapshot()})
        return 1
    if args.localize_figures and not args.image_slug:
        print_json({"ok": False, "stage": "preflight", "error": "missing_image_slug", "message": "--image-slug is required with --localize-figures", "timings": timings.snapshot()})
        return 1
    workdir.mkdir(parents=True, exist_ok=True)
    probes = args.probe if args.probe is not None else ["arxiv", "doi"]
    kind = detect_kind(args.url, args.kind)
    if kind == "arxiv":
        payload = process_arxiv(args.url, root, workdir, args.pdf_backend, args.strict_pdf_backend, probes, args.timeout, timings, paper_digest=args.paper_digest, resource_draft=args.resource_draft, localize_figures=args.localize_figures, image_slug=args.image_slug, state_dir=state_dir, resource_health=args.resource_health)
    else:
        payload = process_pdf(args.url, "direct-pdf", root, workdir, args.pdf_backend, args.strict_pdf_backend, probes, args.timeout, timings, paper_digest=args.paper_digest, resource_draft=args.resource_draft, localize_figures=args.localize_figures, image_slug=args.image_slug, state_dir=state_dir, resource_health=args.resource_health)
    attach_timings(payload, timings)
    write_json(workdir / "evidence_bundle.json", payload)
    print_json(payload)
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
