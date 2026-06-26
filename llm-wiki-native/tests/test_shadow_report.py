from llm_wiki_native.reports import compare_shadow_response


def test_shadow_diff_passes_when_native_preserves_required_paths_and_entities() -> None:
    lightrag = {
        "context_blocks": [{"source_path": "a.md", "source_id": "doc:a"}],
        "hits": [{"record_id": "doc:a"}],
    }
    native = {
        "context_blocks": [{"source_path": "a.md", "source_id": "doc:a"}],
        "hits": [{"record_id": "doc:a"}],
    }

    report = compare_shadow_response(
        "q001",
        lightrag,
        native,
        must_include_paths=["a.md"],
        must_include_entities=["doc:a"],
    )

    assert report["ok"] is True
    assert report["blockers"] == []
    assert report["path_overlap"] == ["a.md"]


def test_shadow_diff_blocks_when_native_misses_required_path_or_entity() -> None:
    lightrag = {"context_blocks": [{"source_path": "a.md", "source_id": "doc:a"}], "hits": [{"record_id": "doc:a"}]}
    native = {"context_blocks": [{"source_path": "b.md", "source_id": "doc:b"}], "hits": [{"record_id": "doc:b"}]}

    report = compare_shadow_response(
        "q001",
        lightrag,
        native,
        must_include_paths=["a.md"],
        must_include_entities=["doc:a"],
    )

    assert report["ok"] is False
    assert "missing required path: a.md" in report["blockers"]
    assert "missing required entity: doc:a" in report["blockers"]
