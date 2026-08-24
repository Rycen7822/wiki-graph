from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from llm_wiki_native.source_docs import collect_source_docs
from ops import native_semantic_artifact_refresh
from ops.wiki_native_raw_section_extract import extract_raw_note_sections
from support import sample_wiki, write_json, write_jsonl


TARGET_PATH = "raw/clip/2601/26010101_Foo-Paper.md"
CONTRACT = {
    "embedding_model": "BAAI/bge-m3",
    "embedding_dim": 1024,
    "embedding_params_version": "v1",
}


def _write_valid_semantic_artifacts(root: Path, state: Path) -> list[dict]:
    doc = next(doc for doc in collect_source_docs(root) if doc.rel_path == TARGET_PATH)
    sections = extract_raw_note_sections(doc)
    write_jsonl(state / "raw_sections.jsonl", sections)
    write_jsonl(state / "method_atoms.jsonl", [])
    write_jsonl(state / "seed_edges.jsonl", [])
    write_jsonl(state / "section_embeddings.jsonl", [])
    write_jsonl(state / "section_similarity_edges.candidates.jsonl", [])
    source_ids = [doc.canonical_id, *[section["section_id"] for section in sections]]
    chunks = {
        f"chunk-{index}": {"source_id": source_id, "content": f"content {index}"}
        for index, source_id in enumerate(source_ids)
    }
    write_json(
        state / "custom_kg_manifest.json",
        {
            "schema_version": 1,
            "metadata": {"schema_version": 1, **CONTRACT},
            "chunks": chunks,
            "entities": {},
            "relationships": {},
        },
    )
    return sections


def _failure_codes(result: dict) -> set[str]:
    return {str(row["code"]) for row in result["failures"]}


def test_semantic_artifact_gate_verifies_current_section_content_and_manifest_coverage(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "state"
    sections = _write_valid_semantic_artifacts(root, state)

    result = native_semantic_artifact_refresh.validate_semantic_artifacts(
        root,
        state,
        integrated_paths=[TARGET_PATH],
        runtime_contract=CONTRACT,
    )

    assert result["ok"] is True
    assert result["covered_path_count"] == 1
    assert result["expected_section_count"] == len(sections)
    assert result["expected_manifest_source_count"] == len(sections) + 1
    assert result["missing_manifest_source_count"] == 0
    assert result["failures"] == []


def test_semantic_artifact_gate_rejects_stale_section_content_and_missing_manifest_source(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "state"
    _write_valid_semantic_artifacts(root, state)
    raw_path = root / TARGET_PATH
    raw_path.write_text(
        raw_path.read_text(encoding="utf-8").replace("A direct method", "A changed direct method"),
        encoding="utf-8",
    )
    manifest_path = state / "custom_kg_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["chunks"].pop(next(iter(manifest["chunks"])))
    write_json(manifest_path, manifest)

    result = native_semantic_artifact_refresh.validate_semantic_artifacts(
        root,
        state,
        integrated_paths=[TARGET_PATH],
        runtime_contract=CONTRACT,
    )

    assert result["ok"] is False
    assert "raw-section-stale" in _failure_codes(result)
    assert "custom-kg-source-coverage-missing" in _failure_codes(result)
    assert result["covered_path_count"] == 0


def test_semantic_artifact_gate_requires_opt_in_for_embedding_contract_change(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "state"
    _write_valid_semantic_artifacts(root, state)
    cache_path = state / "vector_cache.sqlite"
    with sqlite3.connect(cache_path) as conn:
        conn.execute(
            "CREATE TABLE vector_cache("
            "vector_hash TEXT PRIMARY KEY, embedding_model TEXT NOT NULL, "
            "embedding_dim INTEGER NOT NULL, embedding_params_version TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO vector_cache VALUES (?, ?, ?, ?)",
            ("hash-a", CONTRACT["embedding_model"], CONTRACT["embedding_dim"], CONTRACT["embedding_params_version"]),
        )
    changed_contract = {
        "embedding_model": "text-embedding-3-small",
        "embedding_dim": 1536,
        "embedding_params_version": "v1",
    }

    result = native_semantic_artifact_refresh.validate_semantic_artifacts(
        root,
        state,
        integrated_paths=[TARGET_PATH],
        runtime_contract=changed_contract,
    )

    assert result["ok"] is False
    assert "manifest-runtime-embedding-contract-mismatch" in _failure_codes(result)
    assert "vector-cache-embedding-contract-change-requires-opt-in" in _failure_codes(result)


@pytest.mark.subprocess
def test_custom_kg_export_manifest_loads_embedding_contract_from_explicit_workdir(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "state"
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / ".env").write_text(
        "EMBEDDING_MODEL=local-contract-model\nEMBEDDING_DIM=7\nEMBEDDING_PARAMS_VERSION=test-v2\n",
        encoding="utf-8",
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"EMBEDDING_MODEL", "EMBEDDING_DIM", "EMBEDDING_PARAMS_VERSION"}
    }

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ops.custom_kg_incremental",
            "export-manifest",
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--workdir",
            str(workdir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    metadata = json.loads((state / "custom_kg_manifest.json").read_text(encoding="utf-8"))["metadata"]
    assert metadata["embedding_model"] == "local-contract-model"
    assert metadata["embedding_dim"] == 7
    assert metadata["embedding_params_version"] == "test-v2"


def test_active_workspace_gate_verifies_exact_source_sections_and_lexical_spans(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "state"
    doc = next(doc for doc in collect_source_docs(root) if doc.rel_path == TARGET_PATH)
    expected_sections = len(extract_raw_note_sections(doc))
    workspace = state / "native_zvec" / "workspaces" / "active-test"
    workspace.mkdir(parents=True)
    sqlite_path = workspace / "native.sqlite"
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute("CREATE TABLE record(record_type TEXT NOT NULL, source_path TEXT NOT NULL)")
        conn.execute("CREATE TABLE lexical_span(source_path TEXT NOT NULL)")
        conn.executemany(
            "INSERT INTO record VALUES ('section', ?)",
            [(TARGET_PATH,)] * expected_sections,
        )
        conn.execute("INSERT INTO lexical_span VALUES (?)", (TARGET_PATH,))
    write_json(
        state / "native_zvec" / "active_workspace.json",
        {
            "workspace_id": "active-test",
            "source_root": str(root.resolve()),
            "sqlite_path": str(sqlite_path),
        },
    )

    passed = native_semantic_artifact_refresh.validate_active_workspace_coverage(
        root,
        state,
        integrated_paths=[TARGET_PATH],
    )
    assert passed["ok"] is True
    assert passed["covered_path_count"] == 1
    assert passed["paths"][0]["section_record_count"] == expected_sections
    assert passed["paths"][0]["lexical_span_count"] == 1

    with sqlite3.connect(sqlite_path) as conn:
        conn.execute("DELETE FROM lexical_span WHERE source_path = ?", (TARGET_PATH,))
    failed = native_semantic_artifact_refresh.validate_active_workspace_coverage(
        root,
        state,
        integrated_paths=[TARGET_PATH],
    )
    assert failed["ok"] is False
    assert "active-workspace-source-coverage-missing" in _failure_codes(failed)


@pytest.mark.subprocess
def test_stage_failure_redacts_secret_values(tmp_path: Path) -> None:
    secret = "do-not-write-this-api-key"
    env = os.environ.copy()
    env["EMBEDDING_BINDING_API_KEY"] = secret

    with pytest.raises(RuntimeError) as exc_info:
        native_semantic_artifact_refresh.run_stage_subprocess(
            "secret-failure",
            [
                sys.executable,
                "-c",
                "import os,sys; sys.stderr.write(os.environ['EMBEDDING_BINDING_API_KEY']); sys.exit(2)",
            ],
            tmp_path,
            env,
            30,
        )

    assert secret not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)


def test_parallel_extract_stages_share_single_walk(tmp_path: Path, monkeypatch) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "state"
    docs = collect_source_docs(root)

    from ops import wiki_native_artifacts, wiki_native_raw_section_extract

    def _forbidden_walk(_root):
        raise AssertionError("extract stage re-walked the corpus despite shared docs")

    monkeypatch.setattr(wiki_native_artifacts, "collect_source_docs", _forbidden_walk)
    monkeypatch.setattr(wiki_native_raw_section_extract, "collect_source_docs", _forbidden_walk)

    results = native_semantic_artifact_refresh._run_parallel_extract_stages(root.resolve(), state, docs)

    assert set(results) == {"method_atoms", "raw_sections", "seed_edges"}
    for stage in ("method_atoms", "raw_sections", "seed_edges"):
        assert results[stage]["exit_code"] == 0
        assert results[stage]["runner"] == "in_process_shared_walk"
    assert (state / "raw_sections.jsonl").exists()
    assert (state / "method_atoms.jsonl").exists()
    assert (state / "seed_edges.jsonl").exists()


def test_extract_stages_with_shared_docs_match_self_walk_parity(tmp_path: Path) -> None:
    from ops.wiki_native_artifacts import build_seed_edges, extract_method_atoms
    from ops.wiki_native_raw_section_extract import extract_raw_sections

    root = sample_wiki(tmp_path)
    docs = collect_source_docs(root)
    state_shared = tmp_path / "shared"
    state_walked = tmp_path / "walked"

    extract_method_atoms(root, state_shared, docs=docs)
    extract_raw_sections(root, state_shared, docs=docs)
    build_seed_edges(root, state_shared, docs=docs)
    extract_method_atoms(root, state_walked)
    extract_raw_sections(root, state_walked)
    build_seed_edges(root, state_walked)

    for name in ("method_atoms.jsonl", "raw_sections.jsonl", "seed_edges.jsonl"):
        assert (state_shared / name).read_bytes() == (state_walked / name).read_bytes()


def test_refresh_converts_validator_crash_to_saved_machine_failure(tmp_path: Path, monkeypatch) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "state"
    workdir = tmp_path / "repo"
    workdir.mkdir()
    def fake_stage(stage, command, cwd, env, timeout):
        return {"stage": stage, "exit_code": 0, "elapsed_s": 0.0, "output": {"ok": True}}
    monkeypatch.setattr(
        native_semantic_artifact_refresh,
        "validate_semantic_artifacts",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("validator exploded")),
    )
    with pytest.raises(native_semantic_artifact_refresh.SemanticArtifactRefreshError) as exc_info:
        native_semantic_artifact_refresh.refresh_semantic_artifacts(
            root,
            state,
            workdir=workdir,
            integrated_paths=[TARGET_PATH],
            stage_runner=fake_stage,
        )
    report = exc_info.value.report
    assert report["validation"]["failures"][0]["code"] == "semantic-artifact-validator-error"
    saved = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    assert saved["report_path"] == report["report_path"]
    assert saved["ok"] is False
