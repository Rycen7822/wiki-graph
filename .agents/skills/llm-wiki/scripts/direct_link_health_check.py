#!/usr/bin/env python3
"""Fast exact-link health checks for llm-wiki raw-fast clipping.

Checks only URLs explicitly supplied by the caller. It does not search GitHub/HF,
inspect repo trees, enumerate HF siblings, or download large artifacts.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from typing import Any


def _text_tail(value: object, limit: int = 500) -> str:
    """Return a JSON-safe text tail for subprocess output.

    `subprocess.TimeoutExpired` can expose stdout/stderr as bytes even when
    `subprocess.run(..., text=True)` was requested, so normalize before
    serializing probe results.
    """
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", "replace")
    else:
        text = str(value)
    return text[-limit:]


def curl_probe(url: str, method: str, connect_timeout: int, max_time: int, retry: int, max_filesize: int) -> dict[str, Any]:
    write_out = "%{http_code}\t%{url_effective}\t%{content_type}\t%{time_total}\t%{size_download}"
    cmd = [
        "curl",
        "-L",
        "--retry",
        str(retry),
        "--retry-delay",
        "1",
        "--connect-timeout",
        str(connect_timeout),
        "--max-time",
        str(max_time),
        "-sS",
        "-o",
        "/dev/null",
        "-w",
        write_out,
    ]
    if method == "HEAD":
        cmd.append("-I")
    else:
        cmd.extend(["--range", "0-0", "--max-filesize", str(max_filesize)])
    cmd.append(url)
    started = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=max_time + connect_timeout + 5)
    except subprocess.TimeoutExpired as exc:
        return {"method": method, "ok": False, "url": url, "returncode": None, "error": "TimeoutExpired", "elapsed_seconds": round(time.time() - started, 3), "stdout": _text_tail(exc.stdout), "stderr": _text_tail(exc.stderr)}
    parts = (proc.stdout or "").split("\t")
    http_code = None
    if parts and parts[0].isdigit():
        http_code = int(parts[0])
    reachable = http_code is not None and 200 <= http_code < 400
    # curl returns 63 when --max-filesize stops a reachable response. For health checks that is still reachability evidence.
    ok = reachable and proc.returncode in (0, 18, 23, 63)
    return {
        "method": method,
        "ok": ok,
        "reachable": reachable,
        "url": url,
        "returncode": proc.returncode,
        "http_code": http_code,
        "effective_url": parts[1] if len(parts) > 1 else None,
        "content_type": parts[2] if len(parts) > 2 else None,
        "curl_time_total": float(parts[3]) if len(parts) > 3 and parts[3].replace('.', '', 1).isdigit() else None,
        "size_download": float(parts[4]) if len(parts) > 4 and parts[4].replace('.', '', 1).isdigit() else None,
        "stderr": _text_tail(proc.stderr),
    }


def check_url(url: str, args: argparse.Namespace) -> dict[str, Any]:
    head = curl_probe(url, "HEAD", args.connect_timeout, args.head_max_time, args.retry, args.max_filesize)
    code = head.get("http_code")
    if head.get("ok") and code not in {405, 403, 000}:
        return {"url": url, "ok": True, "chosen": "HEAD", "attempts": [head]}
    get = curl_probe(url, "GET", args.connect_timeout, args.get_max_time, args.retry, args.max_filesize)
    return {"url": url, "ok": bool(get.get("ok")), "chosen": "GET", "attempts": [head, get]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast curl-based exact-link health checks for llm-wiki clipping")
    parser.add_argument("urls", nargs="*", help="Exact URLs to check; if omitted, read one URL per line from stdin")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--connect-timeout", type=int, default=8)
    parser.add_argument("--head-max-time", type=int, default=15)
    parser.add_argument("--get-max-time", type=int, default=20)
    parser.add_argument("--retry", type=int, default=1)
    parser.add_argument("--max-filesize", type=int, default=1_048_576, help="GET fallback cap in bytes")
    parser.add_argument("--jsonl", action="store_true", help="Emit one JSON object per URL instead of one JSON array")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    urls = [u.strip() for u in args.urls if u.strip()]
    if not urls:
        urls = [line.strip() for line in sys.stdin if line.strip() and not line.lstrip().startswith("#")]
    if not urls:
        print(json.dumps({"ok": False, "error": "no_urls"}, ensure_ascii=False))
        return 2
    max_workers = max(1, min(args.workers, len(urls)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(lambda u: check_url(u, args), urls))
    if args.jsonl:
        for item in results:
            print(json.dumps(item, ensure_ascii=False, default=str))
    else:
        print(json.dumps({"ok": all(item.get("ok") for item in results), "count": len(results), "results": results}, ensure_ascii=False, indent=2, default=str))
    return 0 if all(item.get("ok") for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
