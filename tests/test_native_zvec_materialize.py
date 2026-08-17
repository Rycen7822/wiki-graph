from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from ops import native_zvec_materialize  # noqa: E402
from support import clear_embedding_env, dump_workspace_tables, materialize_argv, sample_kg_manifest, write_kg_state  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.requires_zvec]


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
    return write_kg_state(
        state,
        manifest=sample_kg_manifest(),
        section_similarity_edges=[{"src_id": "doc:a", "tgt_id": "doc:b", "cosine": 0.9}],
        raw_sections=[
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
        section_embeddings=(
            [{"section_id": "raw_section:doc-a:method", "text_hash": "section-vector", "embedding": [0.5, 0.5]}]
            if with_section_embeddings
            else None
        ),
        vectors=(
            {
                "chunk-hash": [1.0, 0.0],
                "entity-vector": [0.9, 0.1],
                "rel-vector": [0.0, 1.0],
            }
            if with_manifest_vectors
            else None
        ),
    )


def test_build_prepare_only_writes_report_and_prepared_pointer_without_activation(tmp_path, capsys) -> None:
    state = _sample_state(tmp_path)
    workspace_root = tmp_path / "native_zvec" / "workspaces"
    wiki_root = tmp_path / "wiki"
    wiki_root.mkdir()

    assert (
        native_zvec_materialize.main(
            materialize_argv(wiki_root, state, workspace_root, "--embedding-profile", "conservative", "--prepare-only")
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


def test_build_report_records_phase_timings(tmp_path, capsys) -> None:
    state = _sample_state(tmp_path)
    workspace_root = tmp_path / "native_zvec" / "workspaces"
    wiki_root = tmp_path / "wiki"
    wiki_root.mkdir()

    assert (
        native_zvec_materialize.main(
            materialize_argv(wiki_root, state, workspace_root, "--embedding-profile", "conservative", "--prepare-only")
        )
        == 0
    )

    printed = json.loads(capsys.readouterr().out)
    timings = printed["native_report"]["phase_timings"]
    expected = {
        "load_artifacts",
        "materialize_manifest",
        "materialize_sections",
        "materialize_edges",
        "spans_walk",
        "materialize_spans",
        "zvec_assembly",
        "zvec_insert",
        "zvec_optimize",
        "zvec_smoke",
        "audits",
        "total",
    }
    assert expected <= set(timings)
    for name in expected:
        assert isinstance(timings[name], (int, float)), name
        assert timings[name] >= 0, name
    phases = expected - {"total"}
    phase_sum = sum(timings[name] for name in phases)
    # phases overlap across threads: total covers each phase and is bounded by the phase sum
    assert timings["total"] >= max(timings[name] for name in phases) - 0.01
    assert timings["total"] <= phase_sum + 0.05
    assert printed["build_seconds"] >= timings["total"] - 0.01


def test_threaded_build_is_deterministic_across_runs(tmp_path, capsys) -> None:
    state = _sample_state(tmp_path)
    workspace_root = tmp_path / "native_zvec" / "workspaces"
    wiki_root = tmp_path / "wiki"
    wiki_root.mkdir()
    (wiki_root / "note.md").write_text("# Title\n\nalpha beta gamma\n", encoding="utf-8")

    reports = []
    for iteration in range(3):
        for suffix in ("a", "b"):
            workspace_id = f"native-{iteration}{suffix}"
            assert (
                native_zvec_materialize.main(
                    materialize_argv(
                        wiki_root,
                        state,
                        workspace_root,
                        "--embedding-profile",
                        "conservative",
                        "--prepare-only",
                        workspace_id=workspace_id,
                    )
                )
                == 0
            )
            reports.append((workspace_id, json.loads(capsys.readouterr().out)))

    reference = reports[0][1]
    for _, report in reports[1:]:
        assert report["counts"] == reference["counts"]
        assert report["native_report"]["source_manifest_hash"] == reference["native_report"]["source_manifest_hash"]
        assert report["sqlite_edge_count"] == reference["sqlite_edge_count"]
        assert report["native_report"]["lexical_span_count"] == reference["native_report"]["lexical_span_count"]
        assert report["zvec_doc_count"] == reference["zvec_doc_count"]

    def _dump(workspace_id: str) -> dict[str, list[tuple]]:
        return dump_workspace_tables(workspace_root / workspace_id / "native.sqlite", mask_workspace_id=True)

    reference_dump = _dump(reports[0][0])
    for workspace_id, _ in reports[1:]:
        assert _dump(workspace_id) == reference_dump


def test_audit_reads_existing_build_report(tmp_path, capsys) -> None:
    state = _sample_state(tmp_path)
    workspace_root = tmp_path / "native_zvec" / "workspaces"
    native_zvec_materialize.main(materialize_argv(tmp_path / "wiki", state, workspace_root, "--prepare-only"))
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


@pytest.mark.parametrize(
    "with_state_files,expected_names",
    [
        (False, ["custom_kg_manifest.json", "raw_sections.jsonl", "section_similarity_edges.jsonl"]),
        (True, ["vector_cache.sqlite", "section_embeddings.jsonl"]),
    ],
    ids=["missing_state_inputs", "missing_vector_prerequisites"],
)
def test_preflight_reports_missing_inputs_without_workspace_writes(
    tmp_path, capsys, with_state_files: bool, expected_names: list[str]
) -> None:
    state = _sample_state(tmp_path, with_vectors=False) if with_state_files else tmp_path / "state"
    workspace_root = tmp_path / "native_zvec" / "workspaces"

    assert (
        native_zvec_materialize.main(
            materialize_argv(tmp_path / "wiki", state, workspace_root, command="preflight")
        )
        == 1
    )

    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is False
    if not with_state_files:
        assert printed["command"] == "preflight"
    assert printed["missing"] == [str(state / name) for name in expected_names]
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
        native_zvec_materialize.main(materialize_argv(wiki_root, state, workspace_root, "--prepare-only"))
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
    clear_embedding_env(
        monkeypatch,
        "EMBEDDING_BINDING_HOST",
        "OPENAI_BASE_URL",
        "EMBEDDING_BINDING_API_KEY",
        "OPENAI_API_KEY",
    )

    assert (
        native_zvec_materialize.main(
            materialize_argv(wiki_root, state, workspace_root, "--prepare-only", "--fill-missing-vectors")
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
