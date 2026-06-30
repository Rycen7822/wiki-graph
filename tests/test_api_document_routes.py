from __future__ import annotations

from typing import Any

from llm_wiki_native.api.server import create_app
from support import request_asgi as _request


class FakeEngine:
    def query(self, **_kwargs: Any) -> dict[str, Any]:
        return {"hits": [], "trace": {"mode": "mix"}}


def test_retired_document_endpoints_are_not_registered_after_native_cutover() -> None:
    app = create_app(FakeEngine())
    routes = {route.path for route in app.routes}

    assert "/documents/text" not in routes
    assert "/documents/texts" not in routes
    assert "/documents/track_status/{track_id:path}" not in routes
    assert _request(app, "POST", "/documents/text", json={}, raise_app_exceptions=False).status_code == 404
    assert _request(app, "POST", "/documents/texts", json={}, raise_app_exceptions=False).status_code == 404
    assert _request(app, "GET", "/documents/track_status/example", raise_app_exceptions=False).status_code == 404
