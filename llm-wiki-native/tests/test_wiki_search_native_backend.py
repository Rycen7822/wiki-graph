import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import wiki_search  # noqa: E402

LOCAL_SQLITE_CLI_FLAGS = ("--native" + "-db", "--expand-section" + "-neighbors")
LOCAL_SQLITE_SOURCE_MARKERS = (
    '"--native' + '-db"',
    '"--expand-section' + '-neighbors"',
    "run_native" + "_local_query",
    "_native" + "_query_vector",
    "_load_native" + "_backend",
    "_expand_context" + "_block_neighbors",
    "SQLite" + "Workspace",
    "Native" + "QueryEngine",
    '"retrieval_backend": ' + '"sqlite"',
)


def test_wiki_search_default_native_api_uses_server_without_query_vector(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_http_json(method, url, payload, *, timeout=60):
        calls.append({"method": method, "url": url, "payload": payload, "timeout": timeout})
        return {"context_blocks": [{"source_path": "alpha.md"}], "hits": []}

    monkeypatch.setattr(wiki_search, "http_json", fake_http_json)
    args = SimpleNamespace(
        backend="native",
        native_workspace=None,
        query_vector=None,
        mode="mix",
        top_k=3,
        section_kind="methodology",
        data_only=True,
        save_evidence_pack=False,
        state_dir=tmp_path,
        workdir=tmp_path,
        server="http://127.0.0.1:9621",
        neighbor_k=5,
    )

    result = wiki_search.run_query(args, "GraphRAG bottleneck")

    assert result["backend"] == "native"
    assert calls[0]["method"] == "POST"
    assert calls[0]["url"] == "http://127.0.0.1:9621/query/data"
    assert "query_vector" not in calls[0]["payload"]
    assert calls[0]["payload"]["section_kind"] == "methodology"
    assert result["response"]["context_blocks"][0]["source_path"] == "alpha.md"


def test_wiki_search_does_not_export_retired_query_helpers() -> None:
    retired = "light" + "rag"
    assert not hasattr(wiki_search, f"query_{retired}")
    assert not hasattr(wiki_search, f"query_{retired}_data")


def test_wiki_search_cli_help_is_native_only() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "wiki_search.py"), "--help"],
        check=True,
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
    )

    assert "--backend" not in result.stdout
    assert "--chunk-top-k" not in result.stdout
    assert "--intent" not in result.stdout
    assert "--driver" not in result.stdout
    for marker in LOCAL_SQLITE_CLI_FLAGS:
        assert marker not in result.stdout
    assert "--query-vector" in result.stdout
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
                "mode": "mix",
                "top_k": 3,
                "query_vector": [1.0, 0.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fail_run_query(_args, _query):
        raise AssertionError("structured native suite row reached query runner")

    monkeypatch.setattr(wiki_search, "run_query", fail_run_query)
    monkeypatch.setattr(sys, "argv", ["wiki_search.py", "--query-suite", str(query_suite), "--data-only"])

    with pytest.raises(ValueError, match="collect_native_query_report.py"):
        wiki_search.main()


def test_wiki_search_source_has_no_retired_backend_branch() -> None:
    retired = "light" + "rag"
    text = (SCRIPTS / "wiki_search.py").read_text(encoding="utf-8")

    assert retired not in text.lower()
    assert f'choices=["native", "{retired}"]' not in text
    assert ("Light" + "RAG backend") not in text
    assert '"--chunk-top-k"' not in text
    assert '"--intent"' not in text
    assert '"--driver"' not in text
    for marker in LOCAL_SQLITE_SOURCE_MARKERS:
        assert marker not in text
    assert 'choices=["local", "global", "hybrid", "naive", "mix", "bypass"]' not in text
    assert 'choices=["mix", "naive", "bypass"]' in text


def test_wiki_search_rejects_stale_non_native_backend_override(tmp_path) -> None:
    args = SimpleNamespace(
        backend="legacy",
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

    with pytest.raises(ValueError, match="native-only"):
        wiki_search.run_query(args, "alpha")


def test_wiki_search_api_forwards_explicit_query_vector_to_server(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_http_json(method, url, payload, *, timeout=60):
        calls.append({"method": method, "url": url, "payload": payload, "timeout": timeout})
        return {"context_blocks": [{"source_path": "vector.md"}], "hits": []}

    monkeypatch.setattr(wiki_search, "http_json", fake_http_json)
    args = SimpleNamespace(
        backend="native",
        native_workspace="native-test",
        query_vector=json.dumps([1.0, 0.0]),
        mode="mix",
        top_k=1,
        section_kind=None,
        data_only=True,
        save_evidence_pack=False,
        state_dir=tmp_path,
        workdir=tmp_path,
        server="http://127.0.0.1:9621",
        neighbor_k=5,
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
                "workspace_id": "native-test",
                "query_vector": [1.0, 0.0],
            },
            "timeout": 120,
        }
    ]
    assert result["response"]["context_blocks"][0]["source_path"] == "vector.md"


def test_wiki_search_source_has_no_local_sqlite_query_branch() -> None:
    text = (SCRIPTS / "wiki_search.py").read_text(encoding="utf-8")

    for marker in LOCAL_SQLITE_SOURCE_MARKERS:
        assert marker not in text
