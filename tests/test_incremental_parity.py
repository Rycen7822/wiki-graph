"""Incremental-vs-full build parity harness (speedup plan M5/P-10).

Builds a scripted v1 corpus, mutates it to v2 (add/update/delete per record class),
then asserts a copy-then-delta incremental build equals a from-scratch full build.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from types import SimpleNamespace
from pathlib import Path

import pytest

from llm_wiki_native.workspace_build import apply_incremental_workspace_from_state
from ops import native_zvec_materialize
from support import dump_workspace_tables, materialize_argv, sample_kg_manifest, write_kg_state

pytestmark = [pytest.mark.integration, pytest.mark.requires_zvec]


def _manifest_v1() -> dict:
    return sample_kg_manifest(chunk_hash="chunk-a-v1")


def _manifest_v2() -> dict:
    manifest = _manifest_v1()
    manifest["chunks"]["chunk-a"] = {
        **manifest["chunks"]["chunk-a"],
        "content": "Alpha revised",
        "content_hash": "chunk-a-v2",
        "vector_hash": "chunk-a-v2",
    }
    manifest["chunks"]["chunk-b"] = {
        "record_type": "chunk",
        "record_id": "chunk-b",
        "content": "Beta",
        "content_hash": "chunk-b-v1",
        "vector_hash": "chunk-b-v1",
        "embedding_model": "test-embedding",
        "embedding_dim": 2,
        "embedding_params_version": "v1",
        "source_id": "doc:b",
        "file_path": "b.md",
    }
    del manifest["relationships"]["doc:a<SEP>tag:x"]
    return manifest


def _write_state(state: Path, version: int) -> Path:
    if version == 1:
        return write_kg_state(
            state,
            manifest=_manifest_v1(),
            section_similarity_edges=[{"src_id": "doc:a", "tgt_id": "doc:b", "cosine": 0.9}],
            raw_sections=[
                {
                    "section_id": "raw_section:doc-a:method",
                    "source_id": "doc:a",
                    "source_path": "a.md",
                    "section_kind": "methodology",
                    "content": "Method body",
                    "text_hash": "section-s1-v1",
                }
            ],
            section_embeddings=[
                {"section_id": "raw_section:doc-a:method", "text_hash": "section-s1-v1", "embedding": [0.5, 0.5]}
            ],
            vectors={"chunk-a-v1": [1.0, 0.0], "entity-vector": [0.9, 0.1], "rel-vector": [0.0, 1.0]},
        )
    return write_kg_state(
        state,
        manifest=_manifest_v2(),
        section_similarity_edges=[
            {"src_id": "doc:a", "tgt_id": "doc:b", "cosine": 0.7},
            {"src_id": "doc:b", "tgt_id": "doc:c", "cosine": 0.4},
        ],
        raw_sections=[
            {
                "section_id": "raw_section:doc-a:method",
                "source_id": "doc:a",
                "source_path": "a.md",
                "section_kind": "methodology",
                "content": "Method body revised",
                "text_hash": "section-s1-v2",
            },
            {
                "section_id": "raw_section:doc-b:intro",
                "source_id": "doc:b",
                "source_path": "b.md",
                "section_kind": "summary",
                "content": "Intro body",
                "text_hash": "section-s2-v1",
            },
        ],
        section_embeddings=[
            {"section_id": "raw_section:doc-a:method", "text_hash": "section-s1-v2", "embedding": [0.4, 0.6]},
            {"section_id": "raw_section:doc-b:intro", "text_hash": "section-s2-v1", "embedding": [0.2, 0.8]},
        ],
        vectors={
            "chunk-a-v1": [1.0, 0.0],
            "chunk-a-v2": [0.8, 0.2],
            "chunk-b-v1": [0.1, 0.9],
            "entity-vector": [0.9, 0.1],
            "rel-vector": [0.0, 1.0],
        },
    )


def _write_wiki(wiki_root: Path, version: int) -> Path:
    raw_clip = wiki_root / "raw" / "clip"
    raw_clip.mkdir(parents=True, exist_ok=True)
    if version == 1:
        (raw_clip / "a.md").write_text("# Doc A\n\nalpha body\n", encoding="utf-8")
    else:
        (raw_clip / "a.md").write_text("# Doc A v2\n\nalpha body revised\n", encoding="utf-8")
        (raw_clip / "b.md").write_text("# Doc B\n\nbeta body\n", encoding="utf-8")
    return wiki_root


def _full_build(state: Path, wiki_root: Path, workspace_root: Path, workspace_id: str) -> dict:
    return native_zvec_materialize.build(
        SimpleNamespace(
            root=wiki_root,
            state_dir=state,
            workspace_root=workspace_root,
            workspace_id=workspace_id,
            embedding_profile="conservative",
            prepare_only=False,
            fill_missing_vectors=False,
            reuse_unchanged_workspace=False,
        )
    )


def _incremental_build(state: Path, wiki_root: Path, workspace_root: Path, source_id: str, workspace_id: str) -> dict:
    shutil.copytree(workspace_root / source_id, workspace_root / workspace_id)
    report = apply_incremental_workspace_from_state(
        state,
        workspace_root / workspace_id / "native.sqlite",
        workspace_id,
        zvec_path=workspace_root / workspace_id / "zvec_records",
        source_root=wiki_root,
    )
    return report


def _dump(db_path: Path) -> dict[str, list[tuple]]:
    return dump_workspace_tables(db_path, mask_workspace_id=True)


def _assert_parity(incremental: dict, full: dict, workspace_root: Path, inc_id: str, full_id: str) -> None:
    assert incremental["counts"] == full["counts"]
    assert incremental["source_manifest_hash"] == full["source_manifest_hash"]
    assert incremental["edge_count"] == full["edge_count"]
    assert incremental["lexical_span_count"] == full["lexical_span_count"]
    assert incremental["audit"]["ok"] is True
    assert incremental["vector_audit"]["ok"] is True
    assert full["audit"]["ok"] is True
    assert _dump(workspace_root / inc_id / "native.sqlite") == _dump(workspace_root / full_id / "native.sqlite")
    assert incremental["zvec"]["record_count"] == full["zvec"]["record_count"]
    assert incremental["zvec"]["self_nearest_top1_ok"] is True


def test_incremental_matches_full_build_after_mutation(tmp_path) -> None:
    workspace_root = tmp_path / "workspaces"
    state_v1 = _write_state(tmp_path / "state-v1", 1)
    wiki_v1 = _write_wiki(tmp_path / "wiki-v1", 1)
    state_v2 = _write_state(tmp_path / "state-v2", 2)
    wiki_v2 = _write_wiki(tmp_path / "wiki-v2", 2)

    full_v1 = _full_build(state_v1, wiki_v1, workspace_root, "ws-src")
    assert full_v1["ok"] is True

    incremental = _incremental_build(state_v2, wiki_v2, workspace_root, "ws-src", "ws-inc")
    full = _full_build(state_v2, wiki_v2, workspace_root, "ws-full")

    _assert_parity(incremental, full["native_report"], workspace_root, "ws-inc", "ws-full")

    delta = incremental["delta"]
    # Scripted v1→v2 corpus: chunk-b + section s2 added; chunk-a + section s1 updated;
    # relationship deleted. Similarity edge doc:a→doc:b weight changes (update),
    # doc:b→doc:c is added, and the relationship-backed edge is deleted.
    # Span ids embed a content hash, so a text edit is delete+add, never update.
    assert delta["records"] == {"added": 2, "updated": 2, "deleted": 1, "changed": True}
    assert delta["edges"] == {"added": 1, "updated": 1, "deleted": 1, "changed": True}
    assert delta["spans"] == {"added": 2, "updated": 0, "deleted": 1, "changed": True}


def test_incremental_noop_delta_keeps_workspace_identical(tmp_path) -> None:
    workspace_root = tmp_path / "workspaces"
    state_v1 = _write_state(tmp_path / "state-v1", 1)
    wiki_v1 = _write_wiki(tmp_path / "wiki-v1", 1)

    full_v1 = _full_build(state_v1, wiki_v1, workspace_root, "ws-src")
    assert full_v1["ok"] is True
    before = _dump(workspace_root / "ws-src" / "native.sqlite")

    incremental = _incremental_build(state_v1, wiki_v1, workspace_root, "ws-src", "ws-noop")

    assert incremental["delta"] == {
        "records": {"added": 0, "updated": 0, "deleted": 0, "changed": False},
        "edges": {"added": 0, "updated": 0, "deleted": 0, "changed": False},
        "spans": {"added": 0, "updated": 0, "deleted": 0, "changed": False},
    }
    assert incremental["audit"]["ok"] is True
    assert incremental["vector_audit"]["ok"] is True
    assert _dump(workspace_root / "ws-noop" / "native.sqlite") == before
    assert incremental["zvec"]["record_count"] == full_v1["zvec_doc_count"]


def _cli_build(wiki_root: Path, state: Path, workspace_root: Path, workspace_id: str, *extra: str) -> list[str]:
    return materialize_argv(
        wiki_root,
        state,
        workspace_root,
        "--embedding-profile",
        "conservative",
        "--prepare-only",
        *extra,
        workspace_id=workspace_id,
    )


def test_cli_incremental_from_builds_matches_full_and_finalizes(tmp_path, capsys) -> None:
    workspace_root = tmp_path / "workspaces"
    state_v1 = _write_state(tmp_path / "state-v1", 1)
    wiki_v1 = _write_wiki(tmp_path / "wiki-v1", 1)
    state_v2 = _write_state(tmp_path / "state-v2", 2)
    wiki_v2 = _write_wiki(tmp_path / "wiki-v2", 2)

    assert native_zvec_materialize.main(_cli_build(wiki_v1, state_v1, workspace_root, "ws-src")) == 0
    capsys.readouterr()
    assert native_zvec_materialize.main(["finalize", "--workspace-root", str(workspace_root), "--reason", "initial"]) == 0
    active = json.loads(capsys.readouterr().out)
    assert active["workspace_id"] == "ws-src"

    assert (
        native_zvec_materialize.main(_cli_build(wiki_v2, state_v2, workspace_root, "ws-inc", "--incremental-from", "ws-src"))
        == 0
    )
    incremental = json.loads(capsys.readouterr().out)
    assert incremental["ok"] is True
    assert incremental["incremental_from"] == "ws-src"
    assert incremental["source_integrity"]["vector_coverage_ok"] is True
    assert incremental["delta"]["records"]["added"] >= 2

    # cutover rehearsal: finalize promotes the incremental workspace before the reference build runs
    assert native_zvec_materialize.main(["finalize", "--workspace-root", str(workspace_root), "--reason", "incremental"]) == 0
    active = json.loads(capsys.readouterr().out)
    assert active["workspace_id"] == "ws-inc"

    assert native_zvec_materialize.main(_cli_build(wiki_v2, state_v2, workspace_root, "ws-full")) == 0
    full = json.loads(capsys.readouterr().out)
    assert incremental["counts"] == full["counts"]
    assert incremental["native_report"]["source_manifest_hash"] == full["native_report"]["source_manifest_hash"]
    assert incremental["zvec_doc_count"] == full["zvec_doc_count"]
    assert _dump(workspace_root / "ws-inc" / "native.sqlite") == _dump(workspace_root / "ws-full" / "native.sqlite")


def test_cli_incremental_from_rejects_corrupt_source(tmp_path, capsys) -> None:
    workspace_root = tmp_path / "workspaces"
    state_v1 = _write_state(tmp_path / "state-v1", 1)
    wiki_v1 = _write_wiki(tmp_path / "wiki-v1", 1)

    assert native_zvec_materialize.main(_cli_build(wiki_v1, state_v1, workspace_root, "ws-src")) == 0
    capsys.readouterr()

    conn = sqlite3.connect(workspace_root / "ws-src" / "native.sqlite")
    try:
        conn.execute("DELETE FROM record WHERE record_type = 'chunk' AND record_id = 'chunk-a'")
        conn.commit()
    finally:
        conn.close()

    code = native_zvec_materialize.main(_cli_build(wiki_v1, state_v1, workspace_root, "ws-inc", "--incremental-from", "ws-src"))
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"] == "incremental_source_invalid"
    assert payload["incremental_from"] == "ws-src"
    assert not (workspace_root / "ws-inc").exists()
