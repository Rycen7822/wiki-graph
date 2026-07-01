from __future__ import annotations

import json
from pathlib import Path

import pytest

from ops import batch_native_refresh


pytestmark = pytest.mark.integration


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_native_refresh_cutover_rehearsal_promotes_prepared_pointer_and_clears_pending(tmp_path: Path) -> None:
    root = tmp_path / "wiki"
    state_dir = tmp_path / "wikigraph" / "state"
    workspace_root = batch_native_refresh.default_workspace_root(state_dir)
    watched_file = tmp_path / "existing-storage" / "sentinel.json"
    watched_file.parent.mkdir(parents=True)
    watched_file.write_text('{"stable": true}', encoding="utf-8")
    batch_native_refresh.mark_pending(state_dir, root, reason="cutover rehearsal")
    calls: list[tuple] = []

    def build_workspace(**kwargs):
        assert kwargs["root"] == root
        assert kwargs["state_dir"] == state_dir
        assert kwargs["workspace_root"] == workspace_root
        assert kwargs["embedding_profile"] == "conservative"
        calls.append(("build", kwargs["workspace_id"], kwargs["fill_missing_vectors"]))
        workspace_dir = workspace_root / kwargs["workspace_id"]
        workspace_dir.mkdir(parents=True)
        prepared = {
            "schema_version": 1,
            "workspace_id": kwargs["workspace_id"],
            "status": "prepared",
            "sqlite_path": str(workspace_dir / "native.sqlite"),
            "zvec_path": str(workspace_dir / "zvec_records"),
        }
        batch_native_refresh.prepared_workspace_path(state_dir).write_text(
            json.dumps(prepared, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return {"ok": True, "workspace_id": kwargs["workspace_id"], "prepared_workspace": str(batch_native_refresh.prepared_workspace_path(state_dir))}

    def restart_service(*, state_dir: Path) -> dict:
        calls.append(("restart", state_dir))
        return {"service": "llm-wiki-native", "status": "ok"}

    def query_smoke(*, state_dir: Path, active: dict) -> dict:
        calls.append(("smoke", state_dir, active["workspace_id"], active["status"]))
        return {"ok": True, "trace": {"retrieval_backend": "zvec", "vector_hit_count": 1}}

    result = batch_native_refresh.refresh_cutover(
        root=root,
        state_dir=state_dir,
        workspace_root=workspace_root,
        workspace_id="candidate-rehearsal",
        embedding_profile="conservative",
        build_workspace=build_workspace,
        restart_service=restart_service,
        query_smoke=query_smoke,
        required_unchanged_paths=[watched_file],
    )

    active_path = batch_native_refresh.active_workspace_path(state_dir)
    history_path = batch_native_refresh.active_workspace_history_path(state_dir)
    active = _read_json(active_path)
    history = _read_jsonl(history_path)

    assert result["cutover"] is True
    assert result["skipped"] is False
    assert result["build_executed"] is True
    assert result["restart_executed"] is True
    assert result["query_smoke_executed"] is True
    assert result["pending_clear_executed"] is True
    assert result["pending_cleared"] is True
    assert result["unchanged_path_audit"]["ok"] is True
    assert not batch_native_refresh.pending_ledger_path(state_dir).exists()
    assert active["workspace_id"] == "candidate-rehearsal"
    assert active["status"] == "active"
    assert history[-1]["previous"] is None
    assert history[-1]["current"] == active
    assert history[-1]["reason"] == "native graph incremental refresh: cutover"
    assert calls == [
        ("build", "candidate-rehearsal", True),
        ("restart", state_dir),
        ("smoke", state_dir, "candidate-rehearsal", "active"),
    ]
