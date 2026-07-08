from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops import native_zvec_materialize  # noqa: E402
from support import write_json, write_jsonl, write_vector_cache  # noqa: E402


def _sample_state(
    tmp_path: Path,
    *,
    with_vectors: bool = True,
    with_manifest_vectors: bool | None = None,
    with_section_embeddings: bool | None = None,
) -> Path:
    if with_manifest_vectors is None:
        with_manifest_vectors = with_vectors
    if with_section_embeddings is None:
        with_section_embeddings = with_vectors
    state = tmp_path / "state"
    write_json(
        state / "custom_kg_manifest.json",
        {
            "metadata": {
                "embedding_model": "test-embedding",
                "embedding_dim": 2,
                "embedding_params_version": "v1",
            },
            "chunks": {
                "chunk-a": {
                    "record_type": "chunk",
                    "record_id": "chunk-a",
                    "content": "Alpha",
                    "content_hash": "chunk-hash",
                    "vector_hash": "chunk-hash",
                    "embedding_model": "test-embedding",
                    "embedding_dim": 2,
                    "embedding_params_version": "v1",
                    "source_id": "doc:a",
                    "file_path": "a.md",
                }
            },
            "entities": {
                "doc:a": {
                    "record_type": "entity",
                    "record_id": "doc:a",
                    "content": "doc:a\nAlpha",
                    "vector_hash": "entity-vector",
                    "embedding_model": "test-embedding",
                    "embedding_dim": 2,
                    "embedding_params_version": "v1",
                    "metadata_hash": "entity-meta",
                    "source_logical_id": "doc:a",
                    "file_path": "a.md",
                }
            },
            "relationships": {
                "doc:a<SEP>tag:x": {
                    "record_type": "relationship",
                    "record_id": "doc:a<SEP>tag:x",
                    "src_id": "doc:a",
                    "tgt_id": "tag:x",
                    "content": "RELATED\tdoc:a\ttag:x\nAlpha tag",
                    "vector_hash": "rel-vector",
                    "embedding_model": "test-embedding",
                    "embedding_dim": 2,
                    "embedding_params_version": "v1",
                    "metadata_hash": "rel-meta",
                    "weight": 0.6,
                    "source_logical_id": "doc:a",
                    "file_path": "a.md",
                }
            },
        },
    )
    write_jsonl(state / "section_similarity_edges.jsonl", [{"src_id": "doc:a", "tgt_id": "doc:b", "cosine": 0.9}])
    write_jsonl(
        state / "raw_sections.jsonl",
        [
            {
                "section_id": "raw_section:doc-a:method",
                "source_id": "doc:a",
                "source_path": "a.md",
                "paper_title": "Alpha",
                "section_kind": "method",
                "section_title": "Method",
                "canonical_section_title": "method",
                "content": "Method body",
            }
        ],
    )
    if with_section_embeddings:
        write_jsonl(
            state / "section_embeddings.jsonl",
            [{"section_id": "raw_section:doc-a:method", "text_hash": "section-vector", "embedding": [0.5, 0.5]}],
        )
    if with_manifest_vectors:
        write_vector_cache(
            state / "vector_cache.sqlite",
            {
                "chunk-hash": [1.0, 0.0],
                "entity-vector": [0.9, 0.1],
                "rel-vector": [0.0, 1.0],
            },
        )
    return state


def test_build_prepare_only_writes_report_and_prepared_pointer_without_activation(tmp_path, capsys) -> None:
    state = _sample_state(tmp_path)
    workspace_root = tmp_path / "native_zvec" / "workspaces"
    wiki_root = tmp_path / "wiki"
    wiki_root.mkdir()

    assert (
        native_zvec_materialize.main(
            [
                "build",
                "--root",
                str(wiki_root),
                "--state-dir",
                str(state),
                "--workspace-root",
                str(workspace_root),
                "--workspace-id",
                "native-test",
                "--embedding-profile",
                "conservative",
                "--prepare-only",
            ]
        )
        == 0
    )

    printed = json.loads(capsys.readouterr().out)
    workspace_dir = workspace_root / "native-test"
    build_report_path = workspace_dir / "build_report.json"
    prepared_path = workspace_root.parent / "prepared_workspace.json"

    assert printed["ok"] is True
    assert printed["prepare_only"] is True
    assert printed["build_report"] == str(build_report_path)
    assert printed["sqlite_record_count"] == printed["zvec_doc_count"] == 4
    assert printed["self_nearest_top1_ok"] is True
    assert printed["vector_coverage"]["missing"] == 0
    assert build_report_path.exists()
    assert prepared_path.exists()
    assert not (workspace_root.parent / "active_workspace.json").exists()
    assert json.loads(build_report_path.read_text(encoding="utf-8")) == printed


def test_audit_reads_existing_build_report(tmp_path, capsys) -> None:
    state = _sample_state(tmp_path)
    workspace_root = tmp_path / "native_zvec" / "workspaces"
    native_zvec_materialize.main(
        [
            "build",
            "--root",
            str(tmp_path / "wiki"),
            "--state-dir",
            str(state),
            "--workspace-root",
            str(workspace_root),
            "--workspace-id",
            "native-test",
            "--prepare-only",
        ]
    )
    capsys.readouterr()

    assert (
        native_zvec_materialize.main(
            [
                "audit",
                "--workspace-root",
                str(workspace_root),
                "--workspace-id",
                "native-test",
            ]
        )
        == 0
    )

    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is True
    assert printed["workspace_id"] == "native-test"
    assert printed["build_report_found"] is True


def test_build_reuses_existing_candidate_when_state_fingerprints_match(tmp_path, monkeypatch) -> None:
    state = _sample_state(tmp_path)
    workspace_root = tmp_path / "native_zvec" / "workspaces"
    wiki_root = tmp_path / "wiki"
    wiki_root.mkdir()
    args = SimpleNamespace(
        root=wiki_root,
        state_dir=state,
        workspace_root=workspace_root,
        workspace_id="native-test",
        embedding_profile="conservative",
        prepare_only=True,
        fill_missing_vectors=False,
        reuse_unchanged_workspace=False,
    )
    first = native_zvec_materialize.build(args)

    def fail_build_workspace_from_state(*_args, **_kwargs):
        raise AssertionError("matching input fingerprints should reuse the existing candidate report")

    monkeypatch.setattr(native_zvec_materialize, "build_workspace_from_state", fail_build_workspace_from_state)
    reused = native_zvec_materialize.build(
        SimpleNamespace(
            root=wiki_root,
            state_dir=state,
            workspace_root=workspace_root,
            workspace_id="native-test",
            embedding_profile="conservative",
            prepare_only=True,
            fill_missing_vectors=False,
            reuse_unchanged_workspace=True,
        )
    )

    assert first["reused_existing_workspace"] is False
    assert reused["ok"] is True
    assert reused["workspace_id"] == "native-test"
    assert reused["reused_existing_workspace"] is True
    assert reused["reuse_reason"] == "state_input_fingerprints_match"
    assert reused["input_fingerprints"] == first["input_fingerprints"]


def test_preflight_reports_missing_state_inputs_without_workspace_writes(tmp_path, capsys) -> None:
    state = tmp_path / "state"
    workspace_root = tmp_path / "native_zvec" / "workspaces"

    assert (
        native_zvec_materialize.main(
            [
                "preflight",
                "--root",
                str(tmp_path / "wiki"),
                "--state-dir",
                str(state),
                "--workspace-root",
                str(workspace_root),
                "--workspace-id",
                "native-test",
            ]
        )
        == 1
    )

    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is False
    assert printed["command"] == "preflight"
    assert printed["missing"] == [
        str(state / "custom_kg_manifest.json"),
        str(state / "raw_sections.jsonl"),
        str(state / "section_similarity_edges.jsonl"),
    ]
    assert not workspace_root.exists()


def test_preflight_reports_missing_vector_prerequisites_without_workspace_writes(tmp_path, capsys) -> None:
    state = _sample_state(tmp_path, with_vectors=False)
    workspace_root = tmp_path / "native_zvec" / "workspaces"

    assert (
        native_zvec_materialize.main(
            [
                "preflight",
                "--root",
                str(tmp_path / "wiki"),
                "--state-dir",
                str(state),
                "--workspace-root",
                str(workspace_root),
                "--workspace-id",
                "native-test",
            ]
        )
        == 1
    )

    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is False
    assert printed["missing"] == [
        str(state / "vector_cache.sqlite"),
        str(state / "section_embeddings.jsonl"),
    ]
    assert not workspace_root.exists()


def test_build_can_fill_missing_manifest_vectors_with_explicit_injected_embedder(tmp_path) -> None:
    state = _sample_state(tmp_path, with_manifest_vectors=False, with_section_embeddings=True)
    workspace_root = tmp_path / "native_zvec" / "workspaces"
    wiki_root = tmp_path / "wiki"
    wiki_root.mkdir()
    calls = []

    def fake_embedder(texts, *, workdir, embedding_model, embedding_dim):
        calls.append(
            {
                "texts": list(texts),
                "workdir": workdir,
                "embedding_model": embedding_model,
                "embedding_dim": embedding_dim,
            }
        )
        return [[1.0, 0.0], [0.0, 1.0], [0.25, 0.75]]

    report = native_zvec_materialize.build(
        SimpleNamespace(
            root=wiki_root,
            state_dir=state,
            workspace_root=workspace_root,
            workspace_id="native-test",
            embedding_profile="conservative",
            prepare_only=True,
            fill_missing_vectors=True,
        ),
        embed_texts_func=fake_embedder,
    )

    assert report["ok"] is True
    assert report["vector_fill"]["before_missing"] == 3
    assert report["vector_fill"]["after_missing"] == 0
    assert report["vector_fill"]["fill"]["summary"] == {"embedded": 3, "total": 3}
    assert calls == [
        {
            "texts": ["Alpha", "doc:a\nAlpha", "RELATED\tdoc:a\ttag:x\nAlpha tag"],
            "workdir": state.parent,
            "embedding_model": "test-embedding",
            "embedding_dim": 2,
        }
    ]
    assert report["vector_coverage"]["missing"] == 0
    assert json.loads(Path(report["build_report"]).read_text(encoding="utf-8"))["vector_fill"]["after_missing"] == 0


def test_build_reports_missing_vectors_as_json_without_prepared_pointer(tmp_path, capsys) -> None:
    state = _sample_state(tmp_path, with_vectors=False)
    workspace_root = tmp_path / "native_zvec" / "workspaces"
    wiki_root = tmp_path / "wiki"
    wiki_root.mkdir()

    assert (
        native_zvec_materialize.main(
            [
                "build",
                "--root",
                str(wiki_root),
                "--state-dir",
                str(state),
                "--workspace-root",
                str(workspace_root),
                "--workspace-id",
                "native-test",
                "--prepare-only",
            ]
        )
        == 1
    )

    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is False
    assert printed["error"] == "missing_native_vectors"
    assert printed["missing_vectors"]["total_missing"] == 4
    assert printed["missing_vectors"]["by_type"] == {
        "chunk": 1,
        "entity": 1,
        "relationship": 1,
        "section": 1,
    }
    assert not (workspace_root.parent / "prepared_workspace.json").exists()
    assert not (workspace_root / "native-test" / "build_report.json").exists()


def test_build_reports_vector_fill_failure_as_json_without_prepared_pointer(tmp_path, capsys, monkeypatch) -> None:
    state = _sample_state(tmp_path, with_manifest_vectors=False, with_section_embeddings=True)
    workspace_root = tmp_path / "native_zvec" / "workspaces"
    wiki_root = tmp_path / "wiki"
    wiki_root.mkdir()
    for name in [
        "EMBEDDING_BINDING_HOST",
        "OPENAI_BASE_URL",
        "EMBEDDING_BINDING_API_KEY",
        "OPENAI_API_KEY",
    ]:
        monkeypatch.delenv(name, raising=False)

    assert (
        native_zvec_materialize.main(
            [
                "build",
                "--root",
                str(wiki_root),
                "--state-dir",
                str(state),
                "--workspace-root",
                str(workspace_root),
                "--workspace-id",
                "native-test",
                "--prepare-only",
                "--fill-missing-vectors",
            ]
        )
        == 1
    )

    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is False
    assert printed["error"] == "vector_fill_failed"
    assert printed["before_missing"] == 3
    assert "EMBEDDING_BINDING_HOST or OPENAI_BASE_URL is required" in printed["message"]
    assert not (workspace_root.parent / "prepared_workspace.json").exists()
    assert not (workspace_root / "native-test" / "build_report.json").exists()


def test_finalize_and_rollback_use_default_native_pointer_paths(tmp_path, capsys, monkeypatch) -> None:
    workspace_root = tmp_path / "native_zvec" / "workspaces"
    pointer_dir = workspace_root.parent
    calls = []

    def fake_finalize(prepared_workspace_path, active_workspace_path, history_path, *, reason):
        calls.append(("finalize", prepared_workspace_path, active_workspace_path, history_path, reason))
        return {"schema_version": 1, "workspace_id": "new", "status": "active"}

    def fake_rollback(active_workspace_path, history_path):
        calls.append(("rollback", active_workspace_path, history_path))
        return {"schema_version": 1, "workspace_id": "old", "status": "active"}

    monkeypatch.setattr(native_zvec_materialize, "finalize_prepared_workspace", fake_finalize)
    monkeypatch.setattr(native_zvec_materialize, "rollback_active_workspace", fake_rollback)

    assert (
        native_zvec_materialize.main(
            ["finalize", "--workspace-root", str(workspace_root), "--reason", "test finalize"]
        )
        == 0
    )
    finalized = json.loads(capsys.readouterr().out)
    assert finalized["status"] == "active"
    assert finalized["workspace_id"] == "new"

    assert native_zvec_materialize.main(["rollback", "--workspace-root", str(workspace_root)]) == 0
    rolled_back = json.loads(capsys.readouterr().out)
    assert rolled_back["workspace_id"] == "old"
    assert calls == [
        (
            "finalize",
            pointer_dir / "prepared_workspace.json",
            pointer_dir / "active_workspace.json",
            pointer_dir / "active_workspace.history.jsonl",
            "test finalize",
        ),
        ("rollback", pointer_dir / "active_workspace.json", pointer_dir / "active_workspace.history.jsonl"),
    ]
