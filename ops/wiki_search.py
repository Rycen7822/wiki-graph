#!/usr/bin/env python3
import json
from pathlib import Path
import urllib.error
import urllib.request

from llm_wiki_native.contracts import DEFAULT_NEIGHBOR_LIMIT, DEFAULT_QUERY_MODE, DEFAULT_TOP_K, RESPONSE_PROFILES, SUPPORTED_QUERY_MODES
from llm_wiki_native.query_contract import STRUCTURED_QUERY_SUITE_KEYS
from ops.wiki_native_lib import (
    add_query_event,
    common_paths_parser,
    print_json,
    RAW_NOTE_CONTRACT_SECTION_KINDS,
    raw_section_query_for_kind,
    save_evidence_pack,
)



def http_json(method: str, url: str, payload: dict | None = None, *, timeout: int = 60) -> dict:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code} {url}: {body}") from exc
    return json.loads(raw) if raw else {}


def run_native_api_query(args, query: str, *, section_kind: str | None = None) -> dict:
    endpoint = "/query/data" if args.data_only else "/query"
    payload = {
        "query": query,
        "mode": args.mode,
        "top_k": args.top_k,
        "neighbor_limit": args.neighbor_k,
    }
    native_workspace = getattr(args, "native_workspace", None)
    if native_workspace:
        payload["workspace_id"] = native_workspace
    query_vector = getattr(args, "query_vector", None)
    if query_vector:
        payload["query_vector"] = json.loads(query_vector)
    if section_kind:
        payload["section_kind"] = section_kind
    response_profile = getattr(args, "response_profile", None)
    if response_profile:
        payload["response_profile"] = response_profile
    response = http_json("POST", args.server.rstrip("/") + endpoint, payload, timeout=120)
    pack = None
    if args.save_evidence_pack:
        pack = save_evidence_pack(args.state_dir, query, args.mode, response)
    add_query_event(args.state_dir, query, args.mode, str(pack) if pack else None)
    return {"query": query, "mode": args.mode, "section_kind": section_kind, "evidence_pack": str(pack) if pack else None, "backend": "native", "response": response}


def run_query(args, query: str) -> dict:
    section_kind = (args.section_kind or "").strip().lower()
    effective_query = raw_section_query_for_kind(section_kind, query) if section_kind else query
    backend = getattr(args, "backend", None)
    if backend not in (None, "native"):
        raise ValueError("wiki_search is native-only; remove unsupported backend override")
    return run_native_api_query(args, effective_query, section_kind=section_kind or None)


def main() -> int:
    parser = common_paths_parser("Query the llm-wiki native service and optionally save an evidence pack")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--mode", default=DEFAULT_QUERY_MODE, choices=sorted(SUPPORTED_QUERY_MODES))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--section-kind",
        choices=RAW_NOTE_CONTRACT_SECTION_KINDS,
        help="Prefix a raw_section query and, with --data-only, return only matching raw_section_docs chunks.",
    )
    parser.add_argument("--save-evidence-pack", action="store_true")
    parser.add_argument("--data-only", action="store_true", help="Use /query/data retrieval without LLM answer generation")
    parser.add_argument("--response-profile", choices=sorted(RESPONSE_PROFILES), help="Native response profile for /query/data or /query envelopes")
    parser.add_argument("--native-workspace", help="Workspace id override for native API queries")
    parser.add_argument("--query-vector", help="JSON array query vector forwarded to native API queries")
    parser.add_argument("--neighbor-k", type=int, default=DEFAULT_NEIGHBOR_LIMIT, help="Max native semantic neighbor records per hit")
    parser.add_argument("--query-suite", type=Path, help="plain query-list JSONL with one query/q field per row")
    args = parser.parse_args()
    if args.query_suite:
        rows = []
        plain_query_keys = {"query", "q"}
        for line in args.query_suite.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("wiki_search.py --query-suite accepts plain query-list JSONL objects")
            extra_keys = set(item) - plain_query_keys
            if extra_keys & STRUCTURED_QUERY_SUITE_KEYS:
                raise ValueError(
                    "structured native query suites must run through python -m ops.collect_native_query_report; "
                    "wiki_search.py --query-suite accepts plain query-list JSONL only"
                )
            if extra_keys:
                raise ValueError("wiki_search.py --query-suite accepts plain query-list JSONL with only query or q")
            query = item.get("query") or item.get("q")
            if not str(query or "").strip():
                raise ValueError("wiki_search.py --query-suite rows require query or q")
            rows.append(run_query(args, str(query)))
        print_json({"query_suite": str(args.query_suite), "runs": rows})
        return 0
    if not args.query:
        parser.error("query is required unless --query-suite is set")
    print_json(run_query(args, args.query))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
