import json
from pathlib import Path
import subprocess
import sys

import pytest

import llm_wiki_native.workspace_build as workspace_build
from llm_wiki_native.build import MissingNativeVectorsError
from llm_wiki_native.cli import main
from llm_wiki_native.storage.sqlite_workspace import SQLiteWorkspace
from support import sample_kg_manifest, write_kg_state


@pytest.mark.subprocess
def test_cli_module_help_works_with_python_m() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "llm_wiki_native.cli", "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Build and audit llm-wiki native workspaces" in result.stdout
    assert "build-workspace" in result.stdout


def _sample_state(tmp_path: Path, *, with_vectors: bool = True) -> Path:
    return write_kg_state(
        tmp_path / "state",
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
            if with_vectors
            else None
        ),
        vectors=(
            {"chunk-hash": [1.0, 0.0], "entity-vector": [0.9, 0.1], "rel-vector": [0.0, 1.0]} if with_vectors else None
        ),
    )


class _FakeZvecStats:
    attempted = 4
    inserted = 4
    failed = 0


class _FakeZvecSmoke:
    checked = 4
    passed = 4
    failures = []


class _FakeZvecWorkspace:
    def __init__(self) -> None:
        self.records = []
        self.flushed = False
        self.sample_doc_ids = []

    def bulk_insert(self, records):
        self.records = list(records)
        return _FakeZvecStats()

    def flush_optimize_close(self) -> None:
        self.flushed = True

    def self_nearest_smoke(self, sample_doc_ids):
        self.sample_doc_ids = list(sample_doc_ids)
        return _FakeZvecSmoke()


def _fake_zvec_factory(created: dict):
    def factory(path: Path, embedding_dim: int) -> _FakeZvecWorkspace:
        created["path"] = path
        created["embedding_dim"] = embedding_dim
        created["workspace"] = _FakeZvecWorkspace()
        return created["workspace"]

    return factory


def test_workspace_build_materializes_manifest_and_edges_without_activation(tmp_path) -> None:
    state = _sample_state(tmp_path)
    db_path = tmp_path / "native.sqlite"

    report = workspace_build.build_workspace_from_state(state, db_path, "native-test")

    db = SQLiteWorkspace(db_path)
    assert report["audit"]["ok"] is True
    assert report["status"] == "audited"
    assert report["counts"] == {"chunks": 1, "entities": 1, "relationships": 1, "sections": 1}
    assert report["edge_count"] == 2
    assert report["vector_audit"]["ok"] is True
    assert db.get_workspace_status("native-test") == "audited"
    vector_audit = db.audit_vector_coverage("native-test")
    assert vector_audit["counts"]["chunk"] == {"records": 1, "vectors": 1, "missing": 0}
    assert vector_audit["counts"]["section"] == {"records": 1, "vectors": 1, "missing": 0}
    section = db.get_record("native-test", "section", "raw_section:doc-a:method")
    assert section["source_path"] == "a.md"
    assert section["source_id"] == "doc:a"
    assert section["payload"]["section_kind"] == "method"
    assert [item["neighbor_id"] for item in db.neighbors("native-test", "doc:a")] == ["doc:b", "tag:x"]


def test_build_workspace_materializes_source_root_lexical_spans(tmp_path) -> None:
    state = _sample_state(tmp_path)
    wiki_root = tmp_path / "wiki"
    (wiki_root / "_meta").mkdir(parents=True)
    (wiki_root / "concepts").mkdir(parents=True)
    (wiki_root / "_meta" / "raw-clip-map.md").write_text(
        "# Raw Clip Map\n\n- raw/clip/2601/26010101_Foo-Paper.md :: MapOnlyNeedle\n",
        encoding="utf-8",
    )
    (wiki_root / "concepts" / "table-demo.md").write_text(
        "---\ntitle: Table Demo\n---\n# Table Demo\n\n## Results\n\n| Method | Result |\n|---|---|\n| Alpha | TableOnlyNeedle |\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "native.sqlite"

    report = workspace_build.build_workspace_from_state(state, db_path, "native-test", source_root=wiki_root)

    db = SQLiteWorkspace(db_path)
    assert report["lexical_span_count"] >= 2
    map_hits = db.query_lexical_spans("native-test", "MapOnlyNeedle", limit=5)
    table_hits = db.query_lexical_spans("native-test", "TableOnlyNeedle", limit=5)
    assert map_hits[0]["span_kind"] == "map.row"
    assert map_hits[0]["source_path"] == "_meta/raw-clip-map.md"
    assert table_hits[0]["span_kind"] == "table.row"
    assert table_hits[0]["source_path"] == "concepts/table-demo.md"


def test_workspace_build_can_write_zvec_staging_workspace_and_prepared_pointer(tmp_path) -> None:
    state = _sample_state(tmp_path)
    db_path = tmp_path / "native.sqlite"
    zvec_path = tmp_path / "native_zvec" / "workspaces" / "native-test" / "zvec_records"
    prepared_path = tmp_path / "native_zvec" / "prepared_workspace.json"

    created = {}

    report = workspace_build.build_workspace_from_state(
        state,
        db_path,
        "native-test",
        zvec_path=zvec_path,
        prepared_workspace_path=prepared_path,
        zvec_workspace_factory=_fake_zvec_factory(created),
    )

    assert created["path"] == zvec_path
    assert created["embedding_dim"] == 2
    assert [(record.record_type, record.record_id) for record in created["workspace"].records] == [
        ("chunk", "chunk-a"),
        ("entity", "doc:a"),
        ("relationship", "doc:a<SEP>tag:x"),
        ("section", "raw_section:doc-a:method"),
    ]
    assert report["zvec"] == {
        "path": str(zvec_path),
        "embedding_dim": 2,
        "record_count": 4,
        "insert_stats": {"attempted": 4, "inserted": 4, "failed": 0},
        "self_nearest": {"checked": 4, "failures": [], "ok": True, "passed": 4},
        "self_nearest_top1_ok": True,
    }
    assert created["workspace"].flushed is True
    assert len(created["workspace"].sample_doc_ids) == 4
    pointer = json.loads(prepared_path.read_text(encoding="utf-8"))
    assert report["prepared_workspace"] == str(prepared_path)
    assert pointer["schema_version"] == 1
    assert pointer["workspace_id"] == "native-test"
    assert pointer["status"] == "prepared"
    assert pointer["sqlite_path"] == str(db_path)
    assert pointer["zvec_path"] == str(zvec_path)
    assert pointer["source_manifest_hash"] == report["source_manifest_hash"]
    assert pointer["counts"] == report["counts"]
    assert pointer["zvec"] == report["zvec"]


def test_workspace_build_fails_closed_when_vectors_are_missing(tmp_path) -> None:
    state = _sample_state(tmp_path, with_vectors=False)
    db_path = tmp_path / "native.sqlite"

    with pytest.raises(MissingNativeVectorsError) as exc:
        workspace_build.build_workspace_from_state(state, db_path, "native-test")

    assert db_path.exists() is False
    assert exc.value.report["total_missing"] == 4
    assert exc.value.report["by_type"] == {"chunk": 1, "entity": 1, "relationship": 1, "section": 1}


def test_cli_main_builds_zvec_workspace_and_prints_json_report(tmp_path, capsys, monkeypatch) -> None:
    state = _sample_state(tmp_path)
    db_path = tmp_path / "native.sqlite"
    zvec_path = tmp_path / "native.zvec"

    created = {}

    monkeypatch.setattr(workspace_build, "_default_zvec_workspace_factory", _fake_zvec_factory(created))

    assert main(
        [
            "build-workspace",
            "--state-dir",
            str(state),
            "--db",
            str(db_path),
            "--workspace-id",
            "native-test",
            "--zvec-workspace",
            str(zvec_path),
        ]
    ) == 0

    report = json.loads(capsys.readouterr().out)
    assert created["path"] == zvec_path
    assert created["embedding_dim"] == 2
    assert report["workspace_id"] == "native-test"
    assert report["audit"]["ok"] is True
    assert report["zvec"]["path"] == str(zvec_path)
