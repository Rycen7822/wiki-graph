"""Starlette shadow API for native retrieval.

The app is default-off and intended for port 9622. It accepts precomputed query
vectors; embedding and answer generation are deliberately outside this layer.
"""

from __future__ import annotations

import hmac
import json
import math
import os
from typing import Any

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from llm_wiki_native.contracts import DEFAULT_NATIVE_PORT
from llm_wiki_native.retrieval.context import assemble_context

MAX_REQUEST_BODY_BYTES = 1_000_000
MAX_QUERY_VECTOR_DIM = 8_192
MAX_TOP_K = 100
MAX_NEIGHBOR_LIMIT = 50


class RequestEntityTooLarge(ValueError):
    pass


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def _query_vector(value: Any) -> list[float]:
    if not isinstance(value, list):
        raise ValueError("query_vector must be a list of finite numbers")
    if not value:
        raise ValueError("query_vector must not be empty")
    if len(value) > MAX_QUERY_VECTOR_DIM:
        raise ValueError(f"query_vector exceeds max dimension {MAX_QUERY_VECTOR_DIM}")
    vector: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("query_vector must contain only finite numbers")
        numeric = float(item)
        if not math.isfinite(numeric):
            raise ValueError("query_vector must contain only finite numbers")
        vector.append(numeric)
    return vector


def _query_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    if "workspace_id" not in payload:
        raise ValueError("workspace_id is required")
    if "query_vector" not in payload:
        raise ValueError("query_vector is required")
    return {
        "workspace_id": str(payload["workspace_id"]),
        "query": str(payload.get("query", "")),
        "query_vector": _query_vector(payload["query_vector"]),
        "mode": str(payload.get("mode", "mix")),
        "top_k": _bounded_int(payload.get("top_k", 20), default=20, minimum=1, maximum=MAX_TOP_K, field="top_k"),
        "record_types": tuple(payload.get("record_types", ("entity", "relationship", "chunk"))),
        "neighbor_limit": _bounded_int(payload.get("neighbor_limit", 5), default=5, minimum=0, maximum=MAX_NEIGHBOR_LIMIT, field="neighbor_limit"),
    }


async def _json_payload(request: Request) -> dict[str, Any]:
    body = await request.body()
    if len(body) > MAX_REQUEST_BODY_BYTES:
        raise RequestEntityTooLarge(f"request body exceeds {MAX_REQUEST_BODY_BYTES} bytes")
    try:
        payload = json.loads(body.decode("utf-8") if body else "{}")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid JSON request body") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON request body must be an object")
    return payload


def _authorized(request: Request, api_key: str | None) -> bool:
    if not api_key:
        return True
    auth_header = request.headers.get("authorization", "")
    prefix = "Bearer "
    if not auth_header.startswith(prefix):
        return False
    return hmac.compare_digest(auth_header[len(prefix) :], api_key)


def create_app(engine: Any) -> Starlette:
    api_key = os.getenv("LLM_WIKI_NATIVE_API_KEY")

    def unauthorized() -> JSONResponse:
        return JSONResponse({"error": "unauthorized"}, status_code=401, headers={"WWW-Authenticate": "Bearer"})

    async def health(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "service": "llm-wiki-native", "default_port": DEFAULT_NATIVE_PORT})

    async def query_data(request: Request) -> JSONResponse:
        if not _authorized(request, api_key):
            return unauthorized()
        payload = await _json_payload(request)
        result = await run_in_threadpool(engine.query, **_query_kwargs(payload))
        max_chars = _bounded_int(payload.get("max_chars_per_block", 1200), default=1200, minimum=1, maximum=20_000, field="max_chars_per_block")
        return JSONResponse(assemble_context(result, max_chars_per_block=max_chars))

    async def query_trace(request: Request) -> JSONResponse:
        if not _authorized(request, api_key):
            return unauthorized()
        payload = await _json_payload(request)
        result = await run_in_threadpool(engine.query, **_query_kwargs(payload))
        return JSONResponse({"trace": result["trace"]})

    async def value_error(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=400)

    async def request_too_large(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=413)

    async def not_implemented(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=501)

    async def key_error(_request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=404)

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/query/data", query_data, methods=["POST"]),
            Route("/native/query/trace", query_trace, methods=["POST"]),
        ],
        exception_handlers={RequestEntityTooLarge: request_too_large, NotImplementedError: not_implemented, ValueError: value_error, KeyError: key_error},
    )
