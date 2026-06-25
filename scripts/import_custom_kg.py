#!/usr/bin/env python3
"""Import llm-wiki deterministic chunks/edges through LightRAG custom_kg.

Run this with the Python environment that has LightRAG installed, e.g.:
/home/xu/.local/share/uv/tools/lightrag-hku/bin/python scripts/import_custom_kg.py ...

The script writes only under the external LightRAG workdir/state. Stop the
lightrag-server first so the JSON/GraphML/NanoVectorDB files are not shared by
both processes during the import.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from lightrag_runtime_env import env_bool, env_float, env_int, load_env_file, port_open, redact_summary
from wiki_lightrag_lib import (
    DEFAULT_STATE_DIR,
    DEFAULT_WIKI_ROOT,
    DEFAULT_WORKDIR,
    build_custom_kg_payload,
    ensure_state_dirs,
    now_stamp,
    print_json,
)


def build_rag(workdir: Path, storage_dir: Path | None = None):
    try:
        from lightrag import LightRAG
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
        from lightrag.utils import EmbeddingFunc
    except Exception as exc:  # pragma: no cover - depends on external LightRAG env
        raise RuntimeError(
            "LightRAG package is unavailable in this Python. Run with "
            "/home/xu/.local/share/uv/tools/lightrag-hku/bin/python."
        ) from exc

    llm_model = os.environ.get("LLM_MODEL", "hermes-agent")
    llm_host = os.environ.get("LLM_BINDING_HOST") or os.environ.get("OPENAI_BASE_URL")
    llm_key = os.environ.get("LLM_BINDING_API_KEY") or os.environ.get("OPENAI_API_KEY")
    llm_timeout = env_int("LLM_TIMEOUT", 1800)
    llm_max_tokens = env_int("OPENAI_LLM_MAX_TOKENS", 2048)
    temperature = env_float("TEMPERATURE", 0.2)

    embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_host = os.environ.get("EMBEDDING_BINDING_HOST") or os.environ.get("OPENAI_BASE_URL")
    embedding_key = os.environ.get("EMBEDDING_BINDING_API_KEY") or os.environ.get("OPENAI_API_KEY")
    embedding_dim = env_int("EMBEDDING_DIM", 1536)
    embedding_token_limit = env_int("EMBEDDING_MAX_TOKEN_SIZE", env_int("MAX_EMBED_TOKENS", 8192))
    embedding_timeout = env_int("EMBEDDING_TIMEOUT", 120)

    async def llm_func(prompt: str, system_prompt: str | None = None, history_messages=None, **kwargs: Any):
        return await openai_complete_if_cache(
            llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            base_url=llm_host,
            api_key=llm_key,
            timeout=llm_timeout,
            temperature=temperature,
            max_tokens=llm_max_tokens,
            **kwargs,
        )

    async def embed_func(texts, embedding_dim: int | None = None, context: str = "document"):
        return await openai_embed.func(
            texts=texts,
            model=embedding_model,
            base_url=embedding_host,
            api_key=embedding_key,
            embedding_dim=embedding_dim,
            max_token_size=embedding_token_limit,
            context=context,
        )

    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=embedding_token_limit,
        func=embed_func,
        model_name=embedding_model,
    )

    return LightRAG(
        working_dir=str(storage_dir or (workdir / "rag_storage")),
        workspace=os.environ.get("WORKSPACE", ""),
        llm_model_func=llm_func,
        llm_model_name=llm_model,
        llm_model_max_async=env_int("MAX_ASYNC", 4),
        llm_model_kwargs={},
        summary_max_tokens=env_int("SUMMARY_MAX_TOKENS", 1200),
        summary_context_size=env_int("SUMMARY_CONTEXT_SIZE", 12000),
        chunk_token_size=env_int("CHUNK_SIZE", 1200),
        chunk_overlap_token_size=env_int("CHUNK_OVERLAP_SIZE", 100),
        embedding_func=embedding_func,
        embedding_batch_num=env_int("EMBEDDING_BATCH_NUM", 10),
        embedding_func_max_async=env_int("EMBEDDING_FUNC_MAX_ASYNC", env_int("EMBEDDING_MAX_ASYNC", 8)),
        default_llm_timeout=llm_timeout,
        default_embedding_timeout=embedding_timeout,
        kv_storage=os.environ.get("LIGHTRAG_KV_STORAGE", "JsonKVStorage"),
        graph_storage=os.environ.get("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage"),
        vector_storage=os.environ.get("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage"),
        doc_status_storage=os.environ.get("LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage"),
        vector_db_storage_cls_kwargs={"cosine_better_than_threshold": env_float("COSINE_THRESHOLD", 0.2)},
        enable_llm_cache=env_bool("ENABLE_LLM_CACHE", True),
        enable_llm_cache_for_entity_extract=env_bool("ENABLE_LLM_CACHE_FOR_EXTRACT", True),
        max_parallel_insert=env_int("MAX_PARALLEL_INSERT", 2),
        max_graph_nodes=env_int("MAX_GRAPH_NODES", 1000),
        addon_params={
            "language": os.environ.get("SUMMARY_LANGUAGE", "Chinese"),
            "entity_types": [x.strip() for x in os.environ.get("ENTITY_TYPES", "").split(",") if x.strip()],
        },
    )


async def run_import(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    workdir = args.workdir.resolve()
    state_dir = args.state_dir.resolve()
    ensure_state_dirs(state_dir)
    env_values = load_env_file(workdir / ".env")

    payload, payload_summary = build_custom_kg_payload(root, state_dir, args.limit_docs, args.limit_edges)
    from custom_kg_incremental import build_custom_kg_manifest

    desired_manifest = build_custom_kg_manifest(payload)
    summary: dict[str, Any] = {
        "started_at": now_stamp(),
        "wiki_root": str(root),
        "workdir": str(workdir),
        "state_dir": str(state_dir),
        "payload": payload_summary,
        "manifest": desired_manifest.get("summary", {}),
        "import_mode": "full_rebuild",
        "dry_run": args.dry_run,
        "env": redact_summary({k: env_values.get(k, "") for k in [
            "LLM_BINDING", "LLM_BINDING_HOST", "LLM_MODEL",
            "EMBEDDING_BINDING", "EMBEDDING_BINDING_HOST", "EMBEDDING_MODEL", "EMBEDDING_DIM",
            "MAX_GLEANING", "MAX_EXTRACT_INPUT_TOKENS",
        ]}),
    }
    if args.dry_run:
        return summary
    if port_open(args.server_host, args.server_port) and not args.allow_server_running:
        raise RuntimeError(
            f"{args.server_host}:{args.server_port} is listening. Stop lightrag-server before custom_kg import."
        )

    rag = build_rag(workdir)
    await rag.initialize_storages()
    try:
        await rag.ainsert_custom_kg(payload)
    finally:
        await rag.finalize_storages()
    summary["finished_at"] = now_stamp()
    from custom_kg_incremental import load_manifest, write_successful_manifest

    manifest_written = write_successful_manifest(
        state_dir,
        desired_manifest,
        import_mode="full_rebuild",
        previous_manifest=load_manifest(state_dir),
    )
    summary["manifest_path"] = str(manifest_written)
    report = state_dir / "custom_kg_import_report.json"
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["report_path"] = str(report)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Import llm-wiki deterministic custom_kg into LightRAG storage")
    parser.add_argument("--root", type=Path, default=DEFAULT_WIKI_ROOT)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--limit-edges", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-server-running", action="store_true")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=9621)
    args = parser.parse_args()
    try:
        print_json(asyncio.run(run_import(args)))
        return 0
    except Exception as exc:
        print_json({"error": type(exc).__name__, "message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
