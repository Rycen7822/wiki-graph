#!/usr/bin/env python3
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

from wiki_native_lib import (
    add_query_event,
    common_paths_parser,
    print_json,
    RAW_NOTE_CONTRACT_SECTION_KINDS,
    raw_section_query_for_kind,
    save_evidence_pack,
)


def _load_native_backend():
    native_src = Path(__file__).resolve().parents[1] / "llm-wiki-native" / "src"
    if str(native_src) not in sys.path:
        sys.path.insert(0, str(native_src))
    from llm_wiki_native.retrieval.context import assemble_context
    from llm_wiki_native.retrieval.query_engine import NativeQueryEngine
    from llm_wiki_native.storage.sqlite_workspace import SQLiteWorkspace

    return assemble_context, NativeQueryEngine, SQLiteWorkspace


def _native_query_vector(args, query: str, embedding_provider=None) -> list[float]:
    query_vector = getattr(args, "query_vector", None)
    if query_vector:
        return json.loads(query_vector)
    if embedding_provider is not None:
        return embedding_provider.embed_query(query)
    raise ValueError("local native query requires --query-vector JSON or embedding provider")


def _expand_context_block_neighbors(response: dict) -> dict:
    expansions = []
    for block in response.get("context_blocks", []):
        if not isinstance(block, dict):
            continue
        for neighbor in block.get("neighbors", []):
            if not isinstance(neighbor, dict):
                continue
            if neighbor.get("edge_type") != "section_similarity":
                continue
            expansions.append(
                {
                    "seed_record_id": block.get("record_id"),
                    "seed_source_path": block.get("source_path"),
                    "neighbor_id": neighbor.get("neighbor_id"),
                    "edge_type": neighbor.get("edge_type"),
                    "weight": neighbor.get("weight"),
                    "payload": neighbor.get("payload", {}),
                }
            )
    response["section_neighbor_expansions"] = expansions
    return response


def run_native_local_query(args, query: str, *, embedding_provider=None) -> dict:
    if not args.data_only:
        raise ValueError("local native query requires --data-only")
    if not args.native_db:
        raise ValueError("local native query requires --native-db")
    if not args.native_workspace:
        raise ValueError("local native query requires --native-workspace")
    query_vector = _native_query_vector(args, query, embedding_provider)
    assemble_context, NativeQueryEngine, SQLiteWorkspace = _load_native_backend()
    db = SQLiteWorkspace(args.native_db)
    engine = NativeQueryEngine(db)
    record_types = ("entity", "relationship", "chunk", "section") if args.expand_section_neighbors else ("entity", "relationship", "chunk")
    result = engine.query(
        args.native_workspace,
        query,
        query_vector,
        mode=args.mode,
        top_k=args.top_k,
        record_types=record_types,
        neighbor_limit=args.neighbor_k,
    )
    response = assemble_context(result)
    if args.expand_section_neighbors:
        response = _expand_context_block_neighbors(response)
    add_query_event(args.state_dir, query, args.mode, None)
    return {"query": query, "mode": args.mode, "section_kind": None, "evidence_pack": None, "backend": "native", "response": response}


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
    if section_kind:
        payload["section_kind"] = section_kind
    response = http_json("POST", args.server.rstrip("/") + endpoint, payload, timeout=120)
    pack = None
    if args.save_evidence_pack:
        pack = save_evidence_pack(args.state_dir, query, args.mode, response)
    add_query_event(args.state_dir, query, args.mode, str(pack) if pack else None)
    return {"query": query, "mode": args.mode, "section_kind": section_kind, "evidence_pack": str(pack) if pack else None, "backend": "native", "response": response}


def run_query(args, query: str, *, native_embedding_provider=None) -> dict:
    section_kind = (args.section_kind or "").strip().lower()
    effective_query = raw_section_query_for_kind(section_kind, query) if section_kind else query
    backend = getattr(args, "backend", None)
    if backend not in (None, "native"):
        raise ValueError("wiki_search is native-only; remove unsupported backend override")
    native_db = getattr(args, "native_db", None)
    native_workspace = getattr(args, "native_workspace", None)
    query_vector = getattr(args, "query_vector", None)
    if native_db or (native_workspace and query_vector):
        if section_kind:
            raise ValueError("local native query does not support --section-kind")
        if args.save_evidence_pack:
            raise ValueError("local native query does not support evidence-pack")
        return run_native_local_query(args, effective_query, embedding_provider=native_embedding_provider)
    if args.expand_section_neighbors:
        raise ValueError("native API query does not support --expand-section-neighbors")
    return run_native_api_query(args, effective_query, section_kind=section_kind or None)


def main() -> int:
    parser = common_paths_parser("Query the llm-wiki native service and optionally save an evidence pack")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--mode", default="mix", choices=["mix", "naive", "bypass"])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--section-kind",
        choices=RAW_NOTE_CONTRACT_SECTION_KINDS,
        help="Prefix a raw_section query and, with --data-only, return only matching raw_section_docs chunks.",
    )
    parser.add_argument("--save-evidence-pack", action="store_true")
    parser.add_argument("--data-only", action="store_true", help="Use /query/data retrieval without LLM answer generation")
    parser.add_argument("--native-db", type=Path, help="SQLite workspace DB for explicit local native queries")
    parser.add_argument("--native-workspace", help="Workspace id for native API or explicit local native queries")
    parser.add_argument("--query-vector", help="JSON array query vector for explicit local native queries")
    parser.add_argument("--expand-section-neighbors", action="store_true", help="With --data-only, append reviewed semantic section-neighbor expansions from state/section_similarity_edges.jsonl")
    parser.add_argument("--neighbor-k", type=int, default=5, help="Max semantic section-neighbor expansions per direct raw-section hit")
    parser.add_argument("--benchmark", type=Path)
    args = parser.parse_args()
    if args.benchmark:
        rows = []
        for line in args.benchmark.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(run_query(args, item.get("query") or item.get("q") or line))
        print_json({"benchmark": str(args.benchmark), "runs": rows})
        return 0
    if not args.query:
        parser.error("query is required unless --benchmark is set")
    print_json(run_query(args, args.query))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
