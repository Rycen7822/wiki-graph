from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import probe_baseline_query_vector_contract  # noqa: E402


def _openapi_schema(*, query_vector: bool) -> dict:
    properties = {
        "query": {"type": "string"},
        "mode": {"type": "string"},
        "top_k": {"type": "integer"},
    }
    if query_vector:
        properties["query_vector"] = {"items": {"type": "number"}, "type": "array"}
    return {
        "openapi": "3.1.0",
        "paths": {
            "/query/data": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/QueryRequest"}
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "QueryRequest": {
                    "properties": properties,
                    "type": "object",
                }
            }
        },
    }


def test_probe_fails_closed_when_query_data_schema_lacks_query_vector(tmp_path: Path, capsys) -> None:
    schema_path = tmp_path / "openapi.json"
    schema_path.write_text(json.dumps(_openapi_schema(query_vector=False)), encoding="utf-8")

    result = probe_baseline_query_vector_contract.main(
        ["--openapi-json", str(schema_path), "--endpoint", "/query/data"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 1
    assert payload["ok"] is False
    assert payload["explicit_vector_comparable"] is False
    assert payload["query_vector_declared"] is False
    assert payload["request_schema_ref"] == "#/components/schemas/QueryRequest"
    assert "baseline /query/data request schema does not declare query_vector" in payload["blockers"]


def test_probe_accepts_query_data_schema_with_query_vector(tmp_path: Path, capsys) -> None:
    schema_path = tmp_path / "openapi.json"
    schema_path.write_text(json.dumps(_openapi_schema(query_vector=True)), encoding="utf-8")

    result = probe_baseline_query_vector_contract.main(
        ["--openapi-json", str(schema_path), "--endpoint", "/query/data"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["ok"] is True
    assert payload["explicit_vector_comparable"] is True
    assert payload["query_vector_declared"] is True
    assert payload["blockers"] == []
