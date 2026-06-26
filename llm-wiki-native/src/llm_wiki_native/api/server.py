"""Starlette shadow API for native retrieval.

The app is default-off and intended for port 9622. It accepts precomputed query
vectors; embedding and answer generation are deliberately outside this layer.
"""

from __future__ import annotations

from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from llm_wiki_native.contracts import DEFAULT_NATIVE_PORT
from llm_wiki_native.retrieval.context import assemble_context
from llm_wiki_native.retrieval.query_engine import NativeQueryEngine


def _query_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "workspace_id": str(payload["workspace_id"]),
        "query": str(payload.get("query", "")),
        "query_vector": list(payload["query_vector"]),
        "mode": str(payload.get("mode", "mix")),
        "top_k": int(payload.get("top_k", 20)),
        "record_types": tuple(payload.get("record_types", ("entity", "relationship", "chunk"))),
        "neighbor_limit": int(payload.get("neighbor_limit", 5)),
    }


def create_app(engine: NativeQueryEngine) -> Starlette:
    async def health(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "service": "llm-wiki-native", "default_port": DEFAULT_NATIVE_PORT})

    async def query_data(request: Request) -> JSONResponse:
        payload = await request.json()
        result = engine.query(**_query_kwargs(payload))
        return JSONResponse(assemble_context(result, max_chars_per_block=int(payload.get("max_chars_per_block", 1200))))

    async def query_trace(request: Request) -> JSONResponse:
        payload = await request.json()
        result = engine.query(**_query_kwargs(payload))
        return JSONResponse({"trace": result["trace"]})

    async def value_error(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=400)

    async def key_error(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=404)

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/query/data", query_data, methods=["POST"]),
            Route("/native/query/trace", query_trace, methods=["POST"]),
        ],
        exception_handlers={ValueError: value_error, KeyError: key_error},
    )
