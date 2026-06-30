from __future__ import annotations

import asyncio
from typing import Any

import httpx

from llm_wiki_native.api.server import create_app


class FakeEngine:
    def query(self, **_kwargs: Any) -> dict[str, Any]:
        return {"hits": [], "trace": {"mode": "mix"}}


async def _request_async(app, method: str, path: str, **kwargs: Any) -> httpx.Response:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, **kwargs)


def _request(app, method: str, path: str, **kwargs: Any) -> httpx.Response:
    return asyncio.run(_request_async(app, method, path, **kwargs))


def test_retired_document_endpoints_are_not_registered_after_native_cutover() -> None:
    app = create_app(FakeEngine())
    routes = {route.path for route in app.routes}

    assert "/documents/text" not in routes
    assert "/documents/texts" not in routes
    assert "/documents/track_status/{track_id:path}" not in routes
    assert _request(app, "POST", "/documents/text", json={}).status_code == 404
    assert _request(app, "POST", "/documents/texts", json={}).status_code == 404
    assert _request(app, "GET", "/documents/track_status/example").status_code == 404
