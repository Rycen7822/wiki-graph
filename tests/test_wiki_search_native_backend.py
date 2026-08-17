import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

from ops import wiki_search  # noqa: E402

LOCAL_SQLITE_CLI_FLAGS = ("--native" + "-db", "--expand-section" + "-neighbors")


def _args(tmp_path, **over) -> SimpleNamespace:
    values = dict(
        backend="native",
        native_workspace=None,
        query_vector=None,
        mode="mix",
        top_k=3,
        section_kind=None,
        data_only=True,
        save_evidence_pack=False,
        state_dir=tmp_path,
        workdir=tmp_path,
        server="http://127.0.0.1:9621",
        neighbor_k=5,
    )
    values.update(over)
    return SimpleNamespace(**values)


def _fake_http_json(monkeypatch, body, *, calls=None):
    def fake(method, url, payload, *, timeout=60):
        if calls is not None:
            calls.append({"method": method, "url": url, "payload": payload, "timeout": timeout})
        return body

    monkeypatch.setattr(wiki_search, "http_json", fake)


def test_wiki_search_default_native_api_uses_server_without_query_vector(tmp_path, monkeypatch) -> None:
    calls = []
    _fake_http_json(
        monkeypatch,
        {"context_blocks": [{"source_path": "alpha.md"}], "hits": []},
        calls=calls,
    )
    args = _args(tmp_path, section_kind="methodology")

    result = wiki_search.run_query(args, "GraphRAG bottleneck")

    assert result["backend"] == "native"
    assert calls[0]["method"] == "POST"
    assert calls[0]["url"] == "http://127.0.0.1:9621/query/data"
    assert "query_vector" not in calls[0]["payload"]
    assert calls[0]["payload"]["retrieval_goal"] == "focused"
    assert calls[0]["payload"]["section_kind"] == "methodology"
    assert result["retrieval_goal"] == "focused"
    assert result["response"]["context_blocks"][0]["source_path"] == "alpha.md"


def test_wiki_search_can_disable_query_event_recording(tmp_path, monkeypatch) -> None:
    def fail_add_query_event(*_args, **_kwargs):
        raise AssertionError("query event should not be recorded")

    _fake_http_json(monkeypatch, {"context_blocks": [{"source_path": "alpha.md"}], "hits": []})
    monkeypatch.setattr(wiki_search, "add_query_event", fail_add_query_event)
    args = _args(tmp_path, record_query_event=False)

    result = wiki_search.run_query(args, "GraphRAG bottleneck")

    assert result["backend"] == "native"
    assert result["response"]["context_blocks"][0]["source_path"] == "alpha.md"


@pytest.mark.subprocess
def test_wiki_search_cli_help_is_native_only() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ops.wiki_search", "--help"],
        check=True,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )

    assert "--backend" not in result.stdout
    assert "--chunk-top-k" not in result.stdout
    for marker in LOCAL_SQLITE_CLI_FLAGS:
        assert marker not in result.stdout
    assert "--query-vector" in result.stdout
    assert "--retrieval-goal" in result.stdout
    assert "--no-record-query-event" in result.stdout
    assert "records query events by default" in result.stdout
    assert "plain query-list JSONL" in result.stdout
    assert "--query-suite" in result.stdout
    assert "--benchmark" not in result.stdout
    assert ("light" + "rag") not in result.stdout.lower()


def test_wiki_search_query_suite_cli_batches_queries_without_benchmark_label(tmp_path, monkeypatch, capsys) -> None:
    query_suite = tmp_path / "query-suite.jsonl"
    query_suite.write_text('{"query": "alpha"}\n{"q": "beta"}\n', encoding="utf-8")
    calls = []

    def fake_run_query(args, query):
        calls.append(query)
        return {"query": query, "backend": "native", "response": {"hits": []}}

    monkeypatch.setattr(wiki_search, "run_query", fake_run_query)
    monkeypatch.setattr(sys, "argv", ["wiki_search.py", "--query-suite", str(query_suite), "--data-only"])

    assert wiki_search.main() == 0

    assert calls == ["alpha", "beta"]
    payload = json.loads(capsys.readouterr().out)
    assert payload["query_suite"] == str(query_suite)
    assert "benchmark" not in payload


def test_wiki_search_query_suite_rejects_structured_native_rows_before_http(tmp_path, monkeypatch) -> None:
    query_suite = tmp_path / "structured-suite.jsonl"
    query_suite.write_text(
        json.dumps(
            {
                "query": "alpha",
                "retrieval_goal": "coverage",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fail_run_query(_args, _query):
        raise AssertionError("structured native suite row reached query runner")

    monkeypatch.setattr(wiki_search, "run_query", fail_run_query)
    monkeypatch.setattr(sys, "argv", ["wiki_search.py", "--query-suite", str(query_suite), "--data-only"])

    with pytest.raises(ValueError, match="ops.collect_native_query_report"):
        wiki_search.main()


def test_wiki_search_rejects_stale_non_native_backend_override(tmp_path) -> None:
    args = _args(tmp_path, backend="legacy")

    with pytest.raises(ValueError, match="native-only"):
        wiki_search.run_query(args, "alpha")


def test_wiki_search_api_forwards_explicit_query_vector_to_server(tmp_path, monkeypatch) -> None:
    calls = []
    _fake_http_json(
        monkeypatch,
        {"context_blocks": [{"source_path": "vector.md"}], "hits": []},
        calls=calls,
    )
    args = _args(
        tmp_path,
        native_workspace="native-test",
        query_vector=json.dumps([1.0, 0.0]),
        top_k=1,
        response_profile="debug",
        retrieval_goal="coverage",
    )

    result = wiki_search.run_query(args, "alpha")

    assert result["backend"] == "native"
    assert calls == [
        {
            "method": "POST",
            "url": "http://127.0.0.1:9621/query/data",
            "payload": {
                "query": "alpha",
                "mode": "mix",
                "top_k": 1,
                "neighbor_limit": 5,
                "retrieval_goal": "coverage",
                "response_profile": "debug",
                "workspace_id": "native-test",
                "query_vector": [1.0, 0.0],
            },
            "timeout": 120,
        }
    ]
    assert result["retrieval_goal"] == "coverage"
    assert result["response"]["context_blocks"][0]["source_path"] == "vector.md"


def test_wiki_search_evidence_pack_receives_bounded_goal_metadata_without_vector_or_secret(
    tmp_path,
    monkeypatch,
) -> None:
    captured: dict = {}

    monkeypatch.setattr(
        wiki_search,
        "http_json",
        lambda *_args, **_kwargs: {"context_blocks": [], "source_paths": [], "trace": {}},
    )

    def fake_save(state_dir, query, mode, response, *, request_metadata):
        captured.update(request_metadata)
        path = state_dir / "evidence.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok", encoding="utf-8")
        return path

    monkeypatch.setattr(wiki_search, "save_evidence_pack", fake_save)
    args = _args(
        tmp_path,
        native_workspace="native-test",
        query_vector=json.dumps([1.0, 0.0]),
        retrieval_goal="coverage",
        save_evidence_pack=True,
        state_dir=tmp_path / "state",
        neighbor_k=2,
        response_profile="debug",
        record_query_event=False,
        api_key="SHOULD-NOT-APPEAR",
    )

    result = wiki_search.run_query(args, "alpha")

    assert result["retrieval_goal"] == "coverage"
    assert captured["retrieval_goal"] == "coverage"
    assert "query_vector" not in captured
    assert "api_key" not in captured


def test_saved_evidence_pack_bounds_metadata_and_preserves_legacy_schema(tmp_path) -> None:
    from ops import wiki_native_query_events

    pack = wiki_native_query_events.save_evidence_pack(
        tmp_path / "state",
        "answer route query",
        "mix",
        {
            "response": "answer",
            "references": ["raw/example.md"],
            "data": {"context_blocks": [{"source_path": "raw/example.md", "text": "context"}]},
            "trace": {"retrieval_backend": "zvec", "context_block_count": 1},
        },
        request_metadata={
            "retrieval_goal": "coverage",
            "top_k": 10**1000,
            "workspace_id": "workspace-" + ("x" * 5000),
            "record_types": ["section-" + ("y" * 1000) for _ in range(100)],
            "query_vector": [1.0, 0.0],
            "api_key": "SHOULD-NOT-APPEAR",
        },
    )

    text = pack.read_text(encoding="utf-8")
    assert "- file_path: `raw/example.md`" in text
    assert "Retrieval goal: coverage" in text
    assert "query_vector" not in text
    assert "SHOULD-NOT-APPEAR" not in text
    metadata_json = text.split("## Request Metadata\n\n```json\n", 1)[1].split("\n```", 1)[0]
    assert len(metadata_json.encode("utf-8")) <= 1600
    assert json.loads(metadata_json)["retrieval_goal"] == "coverage"

    legacy_pack = wiki_native_query_events.save_evidence_pack(
        tmp_path / "state",
        "legacy query",
        "mix",
        {"response": "answer", "references": []},
    )
    assert '"retrieval_goal": "focused"' in legacy_pack.read_text(encoding="utf-8")

    events_db = wiki_native_query_events.init_query_events_db(tmp_path / "state")
    with sqlite3.connect(events_db) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(query_events)")]
    assert columns == ["id", "query", "mode", "rewritten_queries", "evidence_pack_path", "created_at"]
