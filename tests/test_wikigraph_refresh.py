import importlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def test_batch_wikigraph_refresh_is_importable_and_replaces_old_script_name() -> None:
    module = importlib.import_module("batch_wikigraph_refresh")

    script_names = {path.name for path in SCRIPTS.glob("batch_*refresh.py")}
    old_token = "light" + "rag"

    assert module.__name__ == "batch_wikigraph_refresh"
    assert "batch_wikigraph_refresh.py" in script_names
    assert all(old_token not in name for name in script_names)


def test_batch_wikigraph_refresh_imports_wikigraph_pending_owner_directly() -> None:
    old_backend = "light" + "rag"
    text = (SCRIPTS / "batch_wikigraph_refresh.py").read_text(encoding="utf-8")

    assert "from wiki_wikigraph_refresh_pending import" in text
    assert "LEGACY" + "_WIKIGRAPH" not in text
    assert "WIKIGRAPH" + "_LEGACY" not in text
    assert "legacy" + "-wikigraph-" not in text
    assert f"from wiki_{old_backend}_lib import" not in text
    assert f"mark_{old_backend}_refresh_pending" not in text
    assert f"pending_{old_backend}_refresh_status" not in text


def test_batch_wikigraph_refresh_retired_wrapper_has_no_old_backend_command_planner(tmp_path: Path) -> None:
    module = importlib.import_module("batch_wikigraph_refresh")
    root = tmp_path / "wiki"
    state = tmp_path / "state"
    workdir = tmp_path / "work"

    assert module.build_refresh_command_groups(root, state, workdir) == {"artifact": [], "full_import": []}
    assert module.build_refresh_commands(root, state, workdir) == []
    for planner_name in ("build_incremental_import_commands", "build_full_materialization_import_commands"):
        planner = getattr(module, planner_name)
        try:
            planner(root, state, workdir)
        except module.RetiredWikigraphActivationError as exc:
            assert "batch_native_refresh.py" in str(exc)
        else:  # pragma: no cover - this branch is the regression under test
            raise AssertionError(f"{planner_name} did not fail closed")

    source = (SCRIPTS / "batch_wikigraph_refresh.py").read_text(encoding="utf-8")
    assert "custom_kg_incremental.py" not in source
    assert "rag_storage" not in source
    assert "systemctl" not in source


def test_batch_wikigraph_refresh_cli_is_readonly_retired_and_does_not_write_old_ledger(tmp_path: Path) -> None:
    root = tmp_path / "wiki"
    state = tmp_path / "state"
    workdir = tmp_path / "work"
    script = SCRIPTS / "batch_wikigraph_refresh.py"

    dry = subprocess.run(
        [sys.executable, str(script), "refresh", "--root", str(root), "--state-dir", str(state), "--workdir", str(workdir), "--force", "--dry-run"],
        check=True,
        text=True,
        capture_output=True,
    )
    dry_payload = json.loads(dry.stdout)
    assert dry_payload["retired"] is True
    assert dry_payload["would_run"] is False
    assert dry_payload["commands"] == []

    mark = subprocess.run(
        [sys.executable, str(script), "mark-pending", "--root", str(root), "--state-dir", str(state), "--raw-path", "raw/clip/2601/26010101_Test.md"],
        text=True,
        capture_output=True,
    )
    assert mark.returncode == 1
    mark_payload = json.loads(mark.stdout)
    assert mark_payload["retired"] is True
    assert not (state / "pending_wikigraph_refresh.json").exists()
