from pathlib import Path

from llm_wiki_native.reports import compare_shadow_response


def test_shadow_diff_passes_when_native_preserves_required_paths_and_entities() -> None:
    baseline = {
        "context_blocks": [{"source_path": "a.md", "source_id": "doc:a"}],
        "hits": [{"record_id": "doc:a"}],
    }
    native = {
        "context_blocks": [{"source_path": "a.md", "source_id": "doc:a"}],
        "hits": [{"record_id": "doc:a"}],
    }

    report = compare_shadow_response(
        "q001",
        baseline,
        native,
        must_include_paths=["a.md"],
        must_include_entities=["doc:a"],
    )

    assert report["ok"] is True
    assert report["blockers"] == []
    assert report["baseline_paths"] == ["a.md"]
    assert ("light" + "rag_paths") not in report
    assert report["path_overlap"] == ["a.md"]


def test_shadow_diff_blocks_when_native_misses_required_path_or_entity() -> None:
    baseline = {"context_blocks": [{"source_path": "a.md", "source_id": "doc:a"}], "hits": [{"record_id": "doc:a"}]}
    native = {"context_blocks": [{"source_path": "b.md", "source_id": "doc:b"}], "hits": [{"record_id": "doc:b"}]}

    report = compare_shadow_response(
        "q001",
        baseline,
        native,
        must_include_paths=["a.md"],
        must_include_entities=["doc:a"],
    )

    assert report["ok"] is False
    assert "missing required path: a.md" in report["blockers"]
    assert "missing required entity: doc:a" in report["blockers"]


def test_shadow_report_source_uses_baseline_not_retired_backend() -> None:
    source = compare_shadow_response.__globals__["__file__"]
    text = Path(source).read_text(encoding="utf-8")

    assert ("light" + "rag") not in text.lower()
