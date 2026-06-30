#!/usr/bin/env python3
"""Probe whether a baseline query endpoint declares explicit vector input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fetch_json(url: str, *, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code} {url}: {body}") from exc
    return json.loads(body) if body else {}


def _resolve_ref(openapi: dict[str, Any], ref: str) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"unsupported non-local schema ref: {ref}")
    current: Any = openapi
    for part in ref[2:].split("/"):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"schema ref not found: {ref}")
        current = current[part]
    if not isinstance(current, dict):
        raise ValueError(f"schema ref does not resolve to an object: {ref}")
    return current


def _collect_properties(
    openapi: dict[str, Any],
    schema: dict[str, Any],
    *,
    seen_refs: tuple[str, ...] = (),
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    ref = schema.get("$ref")
    if isinstance(ref, str):
        if ref in seen_refs:
            raise ValueError(f"recursive schema ref: {ref}")
        properties.update(_collect_properties(openapi, _resolve_ref(openapi, ref), seen_refs=(*seen_refs, ref)))
    raw_properties = schema.get("properties")
    if isinstance(raw_properties, dict):
        properties.update(raw_properties)
    for combiner in ("allOf", "anyOf", "oneOf"):
        members = schema.get(combiner)
        if isinstance(members, list):
            for member in members:
                if isinstance(member, dict):
                    properties.update(_collect_properties(openapi, member, seen_refs=seen_refs))
    return properties


def _request_schema(openapi: dict[str, Any], *, endpoint: str) -> tuple[dict[str, Any] | None, str | None]:
    path_item = (openapi.get("paths") or {}).get(endpoint)
    if not isinstance(path_item, dict):
        return None, None
    operation = path_item.get("post")
    if not isinstance(operation, dict):
        return None, None
    request_body = operation.get("requestBody")
    if not isinstance(request_body, dict):
        return None, None
    content = request_body.get("content")
    if not isinstance(content, dict):
        return None, None
    media = content.get("application/json")
    if not isinstance(media, dict):
        return None, None
    schema = media.get("schema")
    if not isinstance(schema, dict):
        return None, None
    ref = schema.get("$ref")
    return schema, ref if isinstance(ref, str) else None


def probe_openapi_contract(openapi: dict[str, Any], *, endpoint: str = "/query/data") -> dict[str, Any]:
    blockers: list[str] = []
    schema, schema_ref = _request_schema(openapi, endpoint=endpoint)
    properties: dict[str, Any] = {}
    if schema is None:
        blockers.append(f"baseline {endpoint} POST request schema not found")
    else:
        properties = _collect_properties(openapi, schema)
        if "query_vector" not in properties:
            blockers.append(f"baseline {endpoint} request schema does not declare query_vector")

    query_vector_declared = "query_vector" in properties
    return {
        "ok": not blockers,
        "endpoint": endpoint,
        "method": "POST",
        "request_schema_ref": schema_ref,
        "request_properties": sorted(str(key) for key in properties),
        "query_vector_declared": query_vector_declared,
        "explicit_vector_comparable": query_vector_declared and not blockers,
        "blockers": blockers,
    }


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe baseline query explicit-vector request contract")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--server", help="Baseline server base URL; /openapi.json is fetched from this URL")
    source.add_argument("--openapi-json", type=Path, help="Saved OpenAPI JSON file")
    parser.add_argument("--endpoint", default="/query/data")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        if args.openapi_json:
            openapi = _read_json(args.openapi_json)
            source_label = str(args.openapi_json)
        else:
            openapi_url = args.server.rstrip("/") + "/openapi.json"
            openapi = _fetch_json(openapi_url, timeout=args.timeout)
            source_label = openapi_url
        report = probe_openapi_contract(openapi, endpoint=args.endpoint)
        report["schema_source"] = source_label
    except Exception as exc:
        report = {
            "ok": False,
            "error": "probe_failed",
            "message": str(exc),
            "exception_type": type(exc).__name__,
        }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print_json(report)
    return 0 if report.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
