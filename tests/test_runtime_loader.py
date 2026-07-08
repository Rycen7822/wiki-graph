from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_wiki_native import runtime
from llm_wiki_native.storage.sqlite_workspace import SQLiteWorkspace
from support import native_record


def _write_pointer(tmp_path: Path, *, status: str = "active", workspace_id: str | None = None) -> tuple[Path, Path, Path]:
    workspace_id = workspace_id or f"native-{status}"
    pointer_path = tmp_path / f"{status}_workspace.json"
    sqlite_path = tmp_path / f"{status}_records.sqlite"
    zvec_path = tmp_path / f"{status}_zvec_records"
    pointer_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workspace_id": workspace_id,
                "status": status,
                "sqlite_path": str(sqlite_path),
                "zvec_path": str(zvec_path),
            }
        ),
        encoding="utf-8",
    )
    return pointer_path, sqlite_path, zvec_path


def test_load_engine_from_workspace_pointer_opens_active_workspace_read_only(tmp_path, monkeypatch) -> None:
    pointer_path, sqlite_path, zvec_path = _write_pointer(tmp_path, workspace_id="native-active")
    calls = {}

    class DB:
        pass

    class Zvec:
        pass

    def open_existing(cls, path: Path, *, read_only: bool = True) -> DB:
        calls["sqlite_path"] = path
        calls["sqlite_read_only"] = read_only
        return DB()

    def zvec_factory(path: Path, *, read_only: bool) -> Zvec:
        calls["zvec_path"] = path
        calls["zvec_read_only"] = read_only
        return Zvec()

    monkeypatch.setattr(runtime.SQLiteWorkspace, "open_existing", classmethod(open_existing))

    engine = runtime.load_engine_from_workspace_pointer(
        pointer_path,
        zvec_workspace_factory=zvec_factory,
    )

    assert calls == {
        "sqlite_path": sqlite_path,
        "sqlite_read_only": True,
        "zvec_path": zvec_path,
        "zvec_read_only": True,
    }
    assert isinstance(engine.db, DB)
    assert isinstance(engine.zvec_workspace, Zvec)
    assert getattr(engine, "default_workspace_id") == "native-active"


def test_active_pointer_loads_audited_sqlite_workspace_as_production_shape(tmp_path) -> None:
    pointer_path = tmp_path / "active_workspace.json"
    sqlite_path = tmp_path / "records.sqlite"
    zvec_path = tmp_path / "zvec_records"
    db = SQLiteWorkspace(sqlite_path)
    db.create_workspace("native-active", "manifest-hash")
    db.put_record(native_record("native-active", "entity", "doc:a", "Alpha", source_path="alpha.md"))
    db.put_vector("native-active", "entity", "doc:a", "doc:a:vector", [1.0, 0.0])
    db.mark_audited("native-active", {"chunks": 0, "entities": 1, "relationships": 0, "sections": 0}, require_vectors=True)
    pointer_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workspace_id": "native-active",
                "status": "active",
                "sqlite_path": str(sqlite_path),
                "zvec_path": str(zvec_path),
            }
        ),
        encoding="utf-8",
    )

    class Hit:
        doc_id = "entity:doc:a"
        score = 1.0
        fields = {"record_type": "entity", "record_id": "doc:a"}

    class Zvec:
        def query_mix(self, query: str, query_vector: list[float], top_k: int, filter_expr: str | None):
            return [Hit()]

    engine = runtime.load_engine_from_workspace_pointer(
        pointer_path,
        zvec_workspace_factory=lambda path, *, read_only: Zvec(),
    )

    result = engine.query("native-active", "alpha", [1.0, 0.0], mode="mix", top_k=1, record_types=("entity",))

    assert db.get_workspace_status("native-active") == "audited"
    assert engine.default_workspace_id == "native-active"
    assert result["trace"]["retrieval_backend"] == "zvec"
    assert result["hits"][0]["record"]["vector_text"] == "Alpha"


def test_load_engine_from_workspace_pointer_status_policy_for_prepared(tmp_path) -> None:
    pointer_path, _, _ = _write_pointer(tmp_path, status="prepared", workspace_id="native-prepared")

    with pytest.raises(ValueError, match="workspace pointer status must be one of: active"):
        runtime.load_engine_from_workspace_pointer(pointer_path)

    class DB:
        pass

    class Zvec:
        pass

    engine = runtime.load_engine_from_workspace_pointer(
        pointer_path,
        allowed_statuses=("active", "prepared"),
        sqlite_workspace_factory=lambda path: DB(),
        zvec_workspace_factory=lambda path, *, read_only: Zvec(),
    )

    assert getattr(engine, "default_workspace_id") == "native-prepared"
