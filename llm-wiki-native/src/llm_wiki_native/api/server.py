"""Starlette API for native llm-wiki retrieval.

The app serves immutable native workspaces. Query embeddings and answer
generation are supplied by configured providers or injected test doubles.
"""

from __future__ import annotations

import argparse
import hmac
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from llm_wiki_native.answer import AnswerGenerator, NativeAnswerConfig, NativeAnswerGenerator
from llm_wiki_native.contracts import DEFAULT_NATIVE_PORT
from llm_wiki_native.embedding import EmbeddingProvider, NativeEmbedding, NativeEmbeddingConfig
from llm_wiki_native.retrieval.context import assemble_context

MAX_REQUEST_BODY_BYTES = 1_000_000
MAX_QUERY_VECTOR_DIM = 8_192
MAX_TOP_K = 100
MAX_NEIGHBOR_LIMIT = 50
UNSUPPORTED_DOCUMENT_ENDPOINT = {
    "error": "unsupported",
    "detail": "native workspaces are built from immutable artifacts; live document endpoints are disabled",
}
EMBEDDING_ENV_KEYS = (
    "LLM_WIKI_NATIVE_EMBEDDING_BASE_URL",
    "LLM_WIKI_NATIVE_EMBEDDING_MODEL",
    "LLM_WIKI_NATIVE_EMBEDDING_API_KEY",
    "EMBEDDING_BINDING_HOST",
    "EMBEDDING_MODEL",
    "EMBEDDING_BINDING_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
)
ANSWER_ENV_KEYS = (
    "LLM_WIKI_NATIVE_ANSWER_BASE_URL",
    "LLM_WIKI_NATIVE_ANSWER_MODEL",
    "LLM_WIKI_NATIVE_ANSWER_API_KEY",
)


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


def _query_kwargs(
    payload: dict[str, Any],
    *,
    embedding_provider: EmbeddingProvider | None = None,
    default_workspace_id: str | None = None,
) -> dict[str, Any]:
    workspace_id = payload.get("workspace_id") or default_workspace_id
    if not workspace_id:
        raise ValueError("workspace_id is required")
    query = str(payload.get("query", ""))
    mode = str(payload.get("mode", "mix"))
    if "query_vector" in payload:
        query_vector = _query_vector(payload["query_vector"])
    elif mode == "bypass":
        query_vector = []
    elif embedding_provider is not None:
        query_vector = _query_vector(embedding_provider.embed_query(query))
    else:
        raise ValueError("query_vector is required when embedding provider is not configured")
    return {
        "workspace_id": str(workspace_id),
        "query": query,
        "query_vector": query_vector,
        "mode": mode,
        "top_k": _bounded_int(payload.get("top_k", 20), default=20, minimum=1, maximum=MAX_TOP_K, field="top_k"),
        "record_types": tuple(payload.get("record_types", ("entity", "relationship", "chunk"))),
        "section_kind": str(payload["section_kind"]) if payload.get("section_kind") else None,
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


def _any_configured(env: Mapping[str, str], keys: tuple[str, ...]) -> bool:
    return any(env.get(key, "").strip() for key in keys)


def embedding_provider_from_env(env: Mapping[str, str] | None = None) -> EmbeddingProvider | None:
    values = os.environ if env is None else env
    if not _any_configured(values, EMBEDDING_ENV_KEYS):
        return None
    return NativeEmbedding(NativeEmbeddingConfig.from_env(values))


def answer_generator_from_env(env: Mapping[str, str] | None = None) -> AnswerGenerator | None:
    values = os.environ if env is None else env
    if not _any_configured(values, ANSWER_ENV_KEYS):
        return None
    return NativeAnswerGenerator(NativeAnswerConfig.from_env(values))


def _answer_response_payload(
    *,
    query: str,
    mode: str,
    context: dict[str, Any],
    answer_generator: AnswerGenerator | None,
) -> dict[str, Any]:
    if answer_generator is None:
        raise NotImplementedError("native answer generator is not configured")
    generated = answer_generator.generate(query, context, mode=mode)
    if isinstance(generated, str):
        answer_payload: dict[str, Any] = {"response": generated}
    elif isinstance(generated, dict):
        answer_payload = dict(generated)
    else:
        raise ValueError("native answer generator must return a string or JSON object")
    return {
        "query": query,
        "mode": mode,
        **answer_payload,
        "data": context,
        "trace": context.get("trace", {}),
    }


def _bypass_response_payload(*, query: str, mode: str, context: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": query,
        "mode": mode,
        "response": "",
        "references": [],
        "data": context,
        "trace": context.get("trace", {}),
    }


def create_app(
    engine: Any,
    embedding_provider: EmbeddingProvider | None = None,
    answer_generator: AnswerGenerator | None = None,
    default_workspace_id: str | None = None,
) -> Starlette:
    api_key = os.getenv("LLM_WIKI_NATIVE_API_KEY")

    def unauthorized() -> JSONResponse:
        return JSONResponse({"error": "unauthorized"}, status_code=401, headers={"WWW-Authenticate": "Bearer"})

    async def health(_request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "service": "llm-wiki-native",
                "backend": "native-zvec",
                "default_port": DEFAULT_NATIVE_PORT,
                "active_workspace_id": default_workspace_id,
            }
        )

    async def query_data(request: Request) -> JSONResponse:
        if not _authorized(request, api_key):
            return unauthorized()
        payload = await _json_payload(request)
        result = await run_in_threadpool(engine.query, **_query_kwargs(payload, embedding_provider=embedding_provider, default_workspace_id=default_workspace_id))
        max_chars = _bounded_int(payload.get("max_chars_per_block", 1200), default=1200, minimum=1, maximum=20_000, field="max_chars_per_block")
        return JSONResponse(assemble_context(result, max_chars_per_block=max_chars))

    async def query(request: Request) -> JSONResponse:
        if not _authorized(request, api_key):
            return unauthorized()
        payload = await _json_payload(request)
        result = await run_in_threadpool(engine.query, **_query_kwargs(payload, embedding_provider=embedding_provider, default_workspace_id=default_workspace_id))
        max_chars = _bounded_int(payload.get("max_chars_per_block", 1200), default=1200, minimum=1, maximum=20_000, field="max_chars_per_block")
        context = assemble_context(result, max_chars_per_block=max_chars)
        mode = str(result.get("trace", {}).get("mode", payload.get("mode", "mix")))
        query_text = str(payload.get("query", ""))
        if mode == "bypass":
            return JSONResponse(_bypass_response_payload(query=query_text, mode=mode, context=context))
        answer_payload = await run_in_threadpool(
            _answer_response_payload,
            query=query_text,
            mode=mode,
            context=context,
            answer_generator=answer_generator,
        )
        return JSONResponse(answer_payload)

    async def query_trace(request: Request) -> JSONResponse:
        if not _authorized(request, api_key):
            return unauthorized()
        payload = await _json_payload(request)
        result = await run_in_threadpool(engine.query, **_query_kwargs(payload, embedding_provider=embedding_provider, default_workspace_id=default_workspace_id))
        return JSONResponse({"trace": result["trace"]})

    async def unsupported_documents(_request: Request) -> JSONResponse:
        return JSONResponse(UNSUPPORTED_DOCUMENT_ENDPOINT, status_code=501)

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
            Route("/query", query, methods=["POST"]),
            Route("/query/data", query_data, methods=["POST"]),
            Route("/native/query/trace", query_trace, methods=["POST"]),
            Route("/documents/text", unsupported_documents, methods=["POST"]),
            Route("/documents/texts", unsupported_documents, methods=["POST"]),
            Route("/documents/track_status/{track_id:path}", unsupported_documents, methods=["GET"]),
        ],
        exception_handlers={RequestEntityTooLarge: request_too_large, NotImplementedError: not_implemented, ValueError: value_error, KeyError: key_error},
    )


def run_server(
    argv: list[str] | None = None,
    *,
    engine_loader: Any | None = None,
    runner: Any | None = None,
    env: Mapping[str, str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Run the llm-wiki native API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_NATIVE_PORT)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--workspace-file", type=Path, required=True)
    parser.add_argument("--allow-workspace-status", action="append", default=[])
    args = parser.parse_args(argv)
    allowed_statuses = ("prepared", *tuple(args.allow_workspace_status))
    if engine_loader is None:
        from llm_wiki_native.runtime import load_engine_from_prepared_workspace

        def engine_loader(path: Path) -> Any:
            return load_engine_from_prepared_workspace(path, allowed_statuses=allowed_statuses)

    engine = engine_loader(args.workspace_file)
    default_workspace_id = getattr(engine, "default_workspace_id", None)
    app = create_app(
        engine,
        embedding_provider=embedding_provider_from_env(env),
        answer_generator=answer_generator_from_env(env),
        default_workspace_id=default_workspace_id,
    )
    app.state.state_dir = args.state_dir
    app.state.allow_workspace_status = tuple(args.allow_workspace_status)
    app.state.default_workspace_id = default_workspace_id
    if runner is None:
        import uvicorn

        runner = uvicorn.run
    runner(app, host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_server(argv)


if __name__ == "__main__":
    raise SystemExit(main())
