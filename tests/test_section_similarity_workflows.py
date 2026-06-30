import sys
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from support import write  # noqa: E402
from ops.wiki_native_jsonl import jsonl_read  # noqa: E402
from ops.wiki_native_section_similarity import _section_rank_lists  # noqa: E402
from ops.wiki_native_section_similarity import _section_rank_lists_scalar  # noqa: E402
from ops.wiki_native_section_similarity import build_section_similarity_edges  # noqa: E402
from ops.wiki_native_section_similarity import build_section_similarity_edges_from_index  # noqa: E402
from ops.wiki_native_section_similarity import section_similarity_embedding_text  # noqa: E402
from ops.wiki_native_section_similarity import section_similarity_index_summary  # noqa: E402
from ops.wiki_native_section_similarity import section_similarity_report_summary  # noqa: E402
from ops.wiki_native_section_similarity import select_section_similarity_edges  # noqa: E402


def test_section_similarity_embedding_text_uses_clean_section_content_without_sidecar_boilerplate() -> None:
    section = {
        "section_id": "raw_section:26010101_Foo-Paper:future",
        "source_id": "raw_clip:26010101_Foo-Paper",
        "source_path": "raw/clip/2601/26010101_Foo-Paper.md",
        "paper_title": "Foo Paper",
        "section_kind": "future",
        "section_title": "对未来研究的启发",
        "content": "- Future work should connect memory repair with section-level evidence retrieval.",
    }
    text = section_similarity_embedding_text(section)
    assert "Title: Foo Paper" in text
    assert "Section kind: future" in text
    assert "Future work should connect memory repair" in text
    assert "LLM_WIKI_RAW_SECTION" not in text
    assert "RAW_SECTION_OF" not in text


def test_build_section_similarity_edges_keeps_sparse_mutual_edges_and_excludes_same_raw_note() -> None:
    sections = [
        {"section_id": "raw_section:a:future", "source_id": "raw_clip:a", "source_path": "raw/clip/a.md", "paper_title": "A", "section_kind": "future", "section_title": "Future", "content": "alpha"},
        {"section_id": "raw_section:b:future", "source_id": "raw_clip:b", "source_path": "raw/clip/b.md", "paper_title": "B", "section_kind": "future", "section_title": "Future", "content": "alpha neighbor"},
        {"section_id": "raw_section:c:future", "source_id": "raw_clip:c", "source_path": "raw/clip/c.md", "paper_title": "C", "section_kind": "future", "section_title": "Future", "content": "orthogonal"},
        {"section_id": "raw_section:a:limitations", "source_id": "raw_clip:a", "source_path": "raw/clip/a.md", "paper_title": "A", "section_kind": "limitations", "section_title": "Limitations", "content": "same raw note should be excluded"},
        {"section_id": "raw_section:d:questions", "source_id": "raw_clip:d", "source_path": "raw/clip/d.md", "paper_title": "D", "section_kind": "questions", "section_title": "Questions", "content": "problem neighbor"},
    ]
    embeddings = {
        "raw_section:a:future": [1.0, 0.0, 0.0],
        "raw_section:b:future": [0.96, 0.28, 0.0],
        "raw_section:c:future": [0.0, 1.0, 0.0],
        "raw_section:a:limitations": [1.0, 0.0, 0.0],
        "raw_section:d:questions": [1.0, 0.0, 0.0],
    }
    edges = build_section_similarity_edges(
        sections,
        embeddings,
        same_kind_k=1,
        cross_kind_k=1,
        same_kind_min_cosine=0.9,
        cross_kind_min_cosine=0.9,
        cross_kind_pairs=[("future", "questions"), ("future", "limitations")],
        mutual=True,
    )
    edge_pairs = {(edge["src_id"], edge["tgt_id"], edge["source_section_kind"], edge["target_section_kind"]) for edge in edges}
    assert ("raw_section:a:future", "raw_section:b:future", "future", "future") in edge_pairs
    assert ("raw_section:a:future", "raw_section:d:questions", "future", "questions") in edge_pairs
    assert all(not {edge["src_id"], edge["tgt_id"]} == {"raw_section:a:future", "raw_section:a:limitations"} for edge in edges)
    assert all(edge["type"] == "SEMANTIC_SECTION_NEIGHBOR" for edge in edges)
    assert all(edge["mutual_knn"] for edge in edges)


def test_section_similarity_index_round_trips_full_builder_edges(tmp_path: Path) -> None:

    sections = [
        {"section_id": "raw_section:a:future", "source_id": "raw_clip:a", "source_path": "raw/clip/a.md", "paper_title": "A", "section_kind": "future", "section_title": "Future", "content": "alpha"},
        {"section_id": "raw_section:b:future", "source_id": "raw_clip:b", "source_path": "raw/clip/b.md", "paper_title": "B", "section_kind": "future", "section_title": "Future", "content": "alpha peer"},
        {"section_id": "raw_section:c:future", "source_id": "raw_clip:c", "source_path": "raw/clip/c.md", "paper_title": "C", "section_kind": "future", "section_title": "Future", "content": "other peer"},
        {"section_id": "raw_section:d:questions", "source_id": "raw_clip:d", "source_path": "raw/clip/d.md", "paper_title": "D", "section_kind": "questions", "section_title": "Questions", "content": "question peer"},
        {"section_id": "raw_section:e:questions", "source_id": "raw_clip:e", "source_path": "raw/clip/e.md", "paper_title": "E", "section_kind": "questions", "section_title": "Questions", "content": "question peer 2"},
    ]
    embeddings = {
        "raw_section:a:future": [1.0, 0.0, 0.0],
        "raw_section:b:future": [0.96, 0.28, 0.0],
        "raw_section:c:future": [0.7, 0.714142842854285, 0.0],
        "raw_section:d:questions": [1.0, 0.0, 0.0],
        "raw_section:e:questions": [0.96, 0.28, 0.0],
    }
    index_path = tmp_path / "section_similarity_index.sqlite"

    full_edges = build_section_similarity_edges(
        sections,
        embeddings,
        same_kind_k=2,
        cross_kind_k=1,
        same_kind_min_cosine=0.7,
        cross_kind_min_cosine=0.9,
        cross_kind_pairs=[("future", "questions")],
        mutual=True,
        embedding_model="test-embedding",
        embedding_dim=3,
        index_path=index_path,
    )
    indexed_edges = build_section_similarity_edges_from_index(
        index_path,
        sections,
        embeddings,
        cross_kind_pairs=[("future", "questions")],
        mutual=True,
        embedding_model="test-embedding",
        embedding_dim=3,
    )

    def comparable(edge: dict[str, object]) -> tuple[object, ...]:
        return (
            edge["edge_id"],
            edge["src_id"],
            edge["tgt_id"],
            edge["pair_kind"],
            edge["cosine"],
            edge["source_rank"],
            edge["target_rank"],
        )

    assert [comparable(edge) for edge in indexed_edges] == [comparable(edge) for edge in full_edges]
    summary = section_similarity_index_summary(index_path)
    assert summary["directed_rows"] >= len(full_edges)
    assert summary["family_count"] >= 2


def test_build_section_similarity_graph_writes_section_similarity_index_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from ops import build_section_similarity_graph

    root = tmp_path / "wiki"
    state = tmp_path / "state"
    workdir = tmp_path / "workdir"
    state.mkdir(parents=True)
    workdir.mkdir(parents=True)
    rows = [
        {"section_id": "raw_section:a:future", "source_id": "raw_clip:a", "source_path": "raw/clip/a.md", "paper_title": "A", "section_kind": "future", "section_title": "Future", "content": "alpha"},
        {"section_id": "raw_section:b:future", "source_id": "raw_clip:b", "source_path": "raw/clip/b.md", "paper_title": "B", "section_kind": "future", "section_title": "Future", "content": "alpha peer"},
    ]
    write(state / "raw_sections.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

    monkeypatch.setattr(
        build_section_similarity_graph,
        "embedding_config",
        lambda _workdir: {"model": "test-embedding", "embedding_dim": 3, "env": {}, "batch_size": 10},
    )
    monkeypatch.setattr(
        build_section_similarity_graph,
        "build_embedding_rows",
        lambda rows_arg, config, cache_path, reuse_cache=True: (
            [
                {**rows_arg[0], "text_hash": "h-a", "embedding_model": config["model"], "embedding_dim": 3, "embedding": [1.0, 0.0, 0.0]},
                {**rows_arg[1], "text_hash": "h-b", "embedding_model": config["model"], "embedding_dim": 3, "embedding": [1.0, 0.0, 0.0]},
            ],
            {"cache_hits": 0, "embedded": 2, "total": 2, "cache_path": cache_path.as_posix()},
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_section_similarity_graph.py",
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--workdir",
            str(workdir),
            "--same-kind-k",
            "1",
            "--same-kind-min-cosine",
            "0.9",
            "--cross-kind-pairs",
            "",
            "--min-content-chars",
            "1",
            "--sample-edges",
            "1",
        ],
    )

    assert build_section_similarity_graph.main() == 0
    index_path = state / "section_similarity_index.sqlite"
    summary = section_similarity_index_summary(index_path)
    assert summary["directed_rows"] == 2
    reports = sorted((state / "section_similarity_reports").glob("*_section_similarity_report.json"))
    assert reports
    report = json.loads(reports[-1].read_text(encoding="utf-8"))
    assert report["section_similarity_index"]["directed_rows"] == 2
    assert len(jsonl_read(state / "section_similarity_edges.candidates.jsonl")) == 1


def test_build_section_similarity_graph_reports_provider_failure_without_partial_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from ops import build_section_similarity_graph

    root = tmp_path / "wiki"
    state = tmp_path / "state"
    workdir = tmp_path / "workdir"
    state.mkdir(parents=True)
    workdir.mkdir(parents=True)
    row = {
        "section_id": "raw_section:a:future",
        "source_id": "raw_clip:a",
        "source_path": "raw/clip/a.md",
        "paper_title": "A",
        "section_kind": "future",
        "section_title": "Future",
        "content": "alpha content long enough for embedding",
    }
    write(state / "raw_sections.jsonl", json.dumps(row, ensure_ascii=False) + "\n")
    for name in [
        "EMBEDDING_BINDING_HOST",
        "OPENAI_BASE_URL",
        "EMBEDDING_BINDING_API_KEY",
        "OPENAI_API_KEY",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_section_similarity_graph.py",
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--workdir",
            str(workdir),
            "--section-kinds",
            "future",
            "--min-content-chars",
            "1",
        ],
    )

    assert build_section_similarity_graph.main() == 1

    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is False
    assert printed["error"] == "section_embedding_failed"
    assert printed["section_count"] == 1
    assert "EMBEDDING_BINDING_HOST or OPENAI_BASE_URL is required" in printed["message"]
    assert not (state / "section_embeddings.jsonl").exists()
    assert not (state / "section_similarity_edges.candidates.jsonl").exists()
    assert list((state / "section_similarity_reports").glob("*_section_similarity_report.json")) == []


def test_section_rank_lists_fast_matches_scalar_reference_on_boundary_fixture() -> None:
    sections = [
        {"section_id": "raw_section:src:future", "source_id": "raw_clip:src", "source_path": "raw/clip/src.md", "paper_title": "Src", "section_kind": "future", "section_title": "Future", "content": "source"},
        {"section_id": "raw_section:a_target:future", "source_id": "raw_clip:a", "source_path": "raw/clip/a.md", "paper_title": "A", "section_kind": "future", "section_title": "Future", "content": "tie a"},
        {"section_id": "raw_section:b_target:future", "source_id": "raw_clip:b", "source_path": "raw/clip/b.md", "paper_title": "B", "section_kind": "future", "section_title": "Future", "content": "tie b"},
        {"section_id": "raw_section:c_below:future", "source_id": "raw_clip:c", "source_path": "raw/clip/c.md", "paper_title": "C", "section_kind": "future", "section_title": "Future", "content": "below threshold"},
        {"section_id": "raw_section:same_note:future", "source_id": "raw_clip:src", "source_path": "raw/clip/src.md", "paper_title": "Src", "section_kind": "future", "section_title": "Future", "content": "same raw note"},
        {"section_id": "raw_section:q1:questions", "source_id": "raw_clip:q1", "source_path": "raw/clip/q1.md", "paper_title": "Q1", "section_kind": "questions", "section_title": "Questions", "content": "cross"},
        {"section_id": "raw_section:q2:questions", "source_id": "raw_clip:q2", "source_path": "raw/clip/q2.md", "paper_title": "Q2", "section_kind": "questions", "section_title": "Questions", "content": "cross lower"},
    ]
    embeddings = {
        "raw_section:src:future": [1.0, 0.0, 0.0],
        "raw_section:a_target:future": [0.8, 0.6, 0.0],
        "raw_section:b_target:future": [0.8, 0.6, 0.0],
        "raw_section:c_below:future": [0.719999, 0.6949759247363014, 0.0],
        "raw_section:same_note:future": [1.0, 0.0, 0.0],
        "raw_section:q1:questions": [0.76, 0.6499230723708769, 0.0],
        "raw_section:q2:questions": [0.759999, 0.6499242414779487, 0.0],
    }

    fast_same = _section_rank_lists(sections, embeddings, "future", "future", 2, 0.72)
    scalar_same = _section_rank_lists_scalar(sections, embeddings, "future", "future", 2, 0.72)
    fast_cross = _section_rank_lists(sections, embeddings, "future", "questions", 1, 0.76)
    scalar_cross = _section_rank_lists_scalar(sections, embeddings, "future", "questions", 1, 0.76)

    assert fast_same == scalar_same
    assert fast_cross == scalar_cross
    assert ("raw_section:src:future", "raw_section:a_target:future") in fast_same
    assert ("raw_section:src:future", "raw_section:b_target:future") in fast_same
    assert ("raw_section:src:future", "raw_section:c_below:future") not in fast_same
    assert ("raw_section:src:future", "raw_section:same_note:future") not in fast_same
    assert fast_same[("raw_section:src:future", "raw_section:a_target:future")]["rank"] == 1
    assert fast_same[("raw_section:src:future", "raw_section:b_target:future")]["rank"] == 2


def test_section_rank_lists_fast_falls_back_to_scalar_for_zero_and_mismatched_vectors() -> None:
    sections = [
        {"section_id": "raw_section:zero:future", "source_id": "raw_clip:zero", "source_path": "raw/clip/zero.md", "paper_title": "Zero", "section_kind": "future", "section_title": "Future", "content": "zero"},
        {"section_id": "raw_section:short:future", "source_id": "raw_clip:short", "source_path": "raw/clip/short.md", "paper_title": "Short", "section_kind": "future", "section_title": "Future", "content": "short"},
        {"section_id": "raw_section:long:future", "source_id": "raw_clip:long", "source_path": "raw/clip/long.md", "paper_title": "Long", "section_kind": "future", "section_title": "Future", "content": "long"},
    ]
    embeddings = {
        "raw_section:zero:future": [0.0, 0.0],
        "raw_section:short:future": [1.0, 0.0],
        "raw_section:long:future": [1.0, 0.0, 0.0],
    }

    assert _section_rank_lists(sections, embeddings, "future", "future", 2, 0.0) == _section_rank_lists_scalar(
        sections, embeddings, "future", "future", 2, 0.0
    )


def test_section_similarity_report_summary_counts_edges_and_hubs() -> None:
    sections = [
        {"section_id": "raw_section:a:future", "section_kind": "future"},
        {"section_id": "raw_section:b:future", "section_kind": "future"},
        {"section_id": "raw_section:c:questions", "section_kind": "questions"},
    ]
    edges = [
        {"src_id": "raw_section:a:future", "tgt_id": "raw_section:b:future", "pair_kind": "future:future", "cosine": 0.91},
        {"src_id": "raw_section:a:future", "tgt_id": "raw_section:c:questions", "pair_kind": "future:questions", "cosine": 0.84},
    ]
    report = section_similarity_report_summary(sections, edges)
    assert report["section_count"] == 3
    assert report["section_count_by_kind"] == {"future": 2, "questions": 1}
    assert report["edge_count"] == 2
    assert report["edge_count_by_pair_kind"] == {"future:future": 1, "future:questions": 1}
    assert report["top_hubs"][0]["section_id"] == "raw_section:a:future"
    assert report["cosine_by_pair_kind"]["future:future"]["max"] == 0.91


def test_select_section_similarity_edges_marks_reviewed_high_value_pairs() -> None:
    candidates = [
        {"src_id": "a", "tgt_id": "b", "pair_kind": "summary:summary", "cosine": 0.9},
        {"src_id": "c", "tgt_id": "d", "pair_kind": "future:future", "cosine": 0.8},
        {"src_id": "e", "tgt_id": "f", "pair_kind": "limitations:questions", "cosine": 0.81},
    ]
    selected = select_section_similarity_edges(candidates, allowed_pair_kinds={"future:future", "limitations:questions"})
    assert [edge["pair_kind"] for edge in selected] == ["future:future", "limitations:questions"]
    assert all(edge["review_status"] == "phase2_selected" for edge in selected)
    assert all("Semantic proximity only" in edge["review_note"] for edge in selected)
