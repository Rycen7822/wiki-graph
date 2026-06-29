from __future__ import annotations

import asyncio
import json

from llm_wiki_native.api.server import create_app


class FakeEngine:
    pass


def test_document_endpoints_return_structured_unsupported_responses() -> None:
    app = create_app(FakeEngine())
    routes = {route.path: route for route in app.routes}

    for path in (
        "/documents/text",
        "/documents/texts",
        "/documents/track_status/{track_id:path}",
    ):
        assert path in routes
        response = asyncio.run(routes[path].endpoint(object()))

        assert response.status_code == 501
        body = json.loads(response.body.decode("utf-8"))
        assert body == {
            "error": "unsupported",
            "detail": "native workspaces are built from immutable artifacts; live document endpoints are disabled",
        }
