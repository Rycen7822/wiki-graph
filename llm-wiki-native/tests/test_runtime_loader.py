from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_wiki_native import runtime


def test_load_engine_from_workspace_pointer_uses_read_only_zvec_factory_for_active_default(tmp_path) -> None:
    pointer_path = tmp_path / "active_workspace.json"
    sqlite_path = tmp_path / "records.sqlite"
    zvec_path = tmp_path / "zvec_records"
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
    calls = {}

    class DB:
        pass

    class Zvec:
        pass

    def sqlite_factory(path: Path) -> DB:
        calls["sqlite_path"] = path
        return DB()

    def zvec_factory(path: Path, *, read_only: bool) -> Zvec:
        calls["zvec_path"] = path
        calls["read_only"] = read_only
        return Zvec()

    engine = runtime.load_engine_from_workspace_pointer(
        pointer_path,
        sqlite_workspace_factory=sqlite_factory,
        zvec_workspace_factory=zvec_factory,
    )

    assert calls == {
        "sqlite_path": sqlite_path,
        "zvec_path": zvec_path,
        "read_only": True,
    }
    assert isinstance(engine.db, DB)
    assert isinstance(engine.zvec_workspace, Zvec)
    assert engine.default_workspace_id == "native-active"


def test_load_engine_from_workspace_pointer_rejects_prepared_by_default(tmp_path) -> None:
    pointer_path = tmp_path / "prepared_workspace.json"
    pointer_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workspace_id": "native-prepared",
                "status": "prepared",
                "sqlite_path": str(tmp_path / "records.sqlite"),
                "zvec_path": str(tmp_path / "zvec_records"),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="workspace pointer status must be one of: active"):
        runtime.load_engine_from_workspace_pointer(pointer_path)


def test_load_engine_from_workspace_pointer_accepts_prepared_when_explicitly_allowed(tmp_path) -> None:
    pointer_path = tmp_path / "prepared_workspace.json"
    sqlite_path = tmp_path / "records.sqlite"
    zvec_path = tmp_path / "zvec_records"
    pointer_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "workspace_id": "native-prepared",
                "status": "prepared",
                "sqlite_path": str(sqlite_path),
                "zvec_path": str(zvec_path),
            }
        ),
        encoding="utf-8",
    )

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

    assert engine.default_workspace_id == "native-prepared"
