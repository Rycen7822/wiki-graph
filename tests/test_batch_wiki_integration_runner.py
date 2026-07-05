import sys
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from support import sample_wiki, write  # noqa: E402
from ops import batch_native_refresh  # noqa: E402
from ops import batch_wiki_integration  # noqa: E402
from ops.wiki_native_wiki_integration_pending import load_pending_wiki_integration_ledger  # noqa: E402
from ops.wiki_native_wiki_integration_pending import mark_pending_wiki_integration  # noqa: E402


def test_batch_wiki_integration_auto_integrate_defaults_to_local_runner_at_threshold(tmp_path: Path, monkeypatch) -> None:
    root = sample_wiki(tmp_path)
    write(
        root / "index.md",
        "# LLM Wiki Index\n\n> Last updated: 2026-05-18 16:00 | Total pages: 4\n\n"
        "## Concepts\n\n- [[foo]] - Foo page.\n\n"
        "## Queries\n\n- [[bar]] - Bar page.\n\n"
        "## Meta\n\n- [[raw-clip-map]] - Raw clip map.\n- [[topic-map]] - Topic map.\n",
    )
    state = tmp_path / "work" / "wikigraph" / "state"
    for idx in range(10):
        mark_pending_wiki_integration(
            state,
            root,
            raw_path=f"raw/clip/2601/260108{idx:02d}_Paper.md",
            title=f"Paper {idx}",
            topic_hints=["local runner topic"],
            required_sections=["summary"],
        )

    native_calls = []

    def fake_native_refresh_after_wiki_integration(root_arg, state_arg, *, workdir, reason):
        native_calls.append((root_arg, state_arg, workdir, reason, len(batch_native_refresh.pending_entries(state_arg))))
        batch_native_refresh.clear_pending(state_arg)
        return 0, {
            "native_refresh": True,
            "runs": [
                {
                    "refresh_kind": "incremental",
                    "fill_missing_vectors": True,
                    "vector_cache_required": True,
                }
            ],
            "status_after": batch_native_refresh.status(root_arg, state_arg),
        }

    monkeypatch.setattr(batch_wiki_integration, "run_native_refresh_after_wiki_integration", fake_native_refresh_after_wiki_integration)

    code, payload = batch_wiki_integration.run_auto_integration(root, state, reason="threshold")

    assert code == 0
    assert payload["ran"] is True
    assert payload["runner"] == "local"
    assert payload["command"][0] == "integrate-local"
    assert payload["local_result"]["apply"]["operations_applied"] >= 11
    assert payload["local_result"]["validation"]["errors_count"] == 0
    assert "input_fingerprints" not in payload["local_result"]["validation"]
    assert payload["local_result"]["native_refresh"]["runs"][0]["fill_missing_vectors"] is True
    assert payload["post_status"]["pending_count"] == 0
    assert load_pending_wiki_integration_ledger(state)["pending"] == []
    assert len(batch_native_refresh.pending_entries(state)) == 0
    assert native_calls == [(root, state, ROOT, "threshold", 1)]


def test_batch_wiki_integration_auto_integrate_runs_configured_runner_at_threshold_and_requires_cleared_ledger(tmp_path: Path, monkeypatch) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    for idx in range(10):
        mark_pending_wiki_integration(state, root, raw_path=f"raw/clip/2601/260106{idx:02d}_Paper.md", title=f"Paper {idx}", required_sections=["summary"])
    fake_runner = tmp_path / "fake_wiki_integrator.py"
    write(
        fake_runner,
        "import os, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from ops.wiki_native_wiki_integration_bridge import clear_pending_wiki_integration_after_success\n"
        "root = Path(os.environ['LLM_WIKI_ROOT'])\n"
        "state = Path(os.environ['LLM_WIKI_STATE_DIR'])\n"
        "prompt_path = Path(os.environ['LLM_WIKI_INTEGRATION_PROMPT'])\n"
        "seen = state / 'fake_runner_seen.txt'\n"
        "seen.write_text(prompt_path.read_text(encoding='utf-8')[:1600], encoding='utf-8')\n"
        "clear_pending_wiki_integration_after_success(root, state, reason=os.environ.get('LLM_WIKI_INTEGRATION_REASON', 'threshold'))\n",
    )

    native_calls = []

    def fake_native_refresh_after_wiki_integration(root_arg, state_arg, *, workdir, reason):
        native_calls.append((root_arg, state_arg, workdir, reason, len(batch_native_refresh.pending_entries(state_arg))))
        batch_native_refresh.clear_pending(state_arg)
        return 0, {
            "native_refresh": True,
            "runs": [
                {
                    "refresh_kind": "incremental",
                    "fill_missing_vectors": True,
                    "vector_cache_required": True,
                }
            ],
            "status_after": batch_native_refresh.status(root_arg, state_arg),
        }

    monkeypatch.setattr(batch_wiki_integration, "run_native_refresh_after_wiki_integration", fake_native_refresh_after_wiki_integration)

    code, payload = batch_wiki_integration.run_auto_integration(
        root,
        state,
        reason="threshold",
        integration_command=f"{sys.executable} {fake_runner}",
    )

    assert code == 0
    assert payload["ran"] is True
    assert payload["pre_status"]["should_integrate"] is True
    assert payload["post_status"]["should_integrate"] is False
    assert payload["post_status"]["pending_count"] == 0
    assert payload["native_refresh"]["runs"][0]["fill_missing_vectors"] is True
    assert payload["prompt_path"].endswith(".md")
    assert payload["plan_path"].endswith("_wiki_integration_plan.json")
    assert payload["plan_hash"]
    assert payload["plan_operations"] >= 1
    assert payload["prompt_chars"] < 5000
    assert Path(payload["plan_path"]).exists()
    assert "Plan artifact" in (state / "fake_runner_seen.txt").read_text(encoding="utf-8")
    assert "batch wiki integration" in (state / "fake_runner_seen.txt").read_text(encoding="utf-8")
    assert load_pending_wiki_integration_ledger(state)["pending"] == []
    assert len(batch_native_refresh.pending_entries(state)) == 0
    assert native_calls == [(root, state, ROOT, "threshold", 1)]


def _set_native_cutover_env(monkeypatch, watched_dir: Path) -> None:
    monkeypatch.setenv("LLM_WIKI_NATIVE_RESTART_COMMAND", f"{sys.executable} -c pass")
    monkeypatch.setenv("LLM_WIKI_NATIVE_SMOKE_URL", "http://127.0.0.1:9621/query/data")
    monkeypatch.setenv("LLM_WIKI_NATIVE_SMOKE_QUERY", "native refresh smoke")
    monkeypatch.setenv("LLM_WIKI_NATIVE_UNCHANGED_PATH", str(watched_dir))


def _append_native_history(state_dir: Path, reasons: list[str]) -> None:
    history = batch_native_refresh.active_workspace_history_path(state_dir)
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("a", encoding="utf-8") as f:
        for idx, reason in enumerate(reasons):
            f.write(json.dumps({"reason": reason, "current": {"workspace_id": f"ws-{idx}"}}) + "\n")


def test_native_refresh_followthrough_runs_incremental_then_due_full_rebuild_with_vector_cache(tmp_path: Path, monkeypatch) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    watched_dir = tmp_path / "watched"
    watched_dir.mkdir()
    _append_native_history(state, [f"native graph incremental refresh: cutover {idx}" for idx in range(4)])
    batch_native_refresh.mark_pending(state, root, reason="wiki-integration:threshold")
    _set_native_cutover_env(monkeypatch, watched_dir)
    runs = []

    def fake_refresh_cutover(**kwargs):
        status_before = batch_native_refresh.status(kwargs["root"], kwargs["state_dir"])
        refresh_kind = status_before["next_refresh_kind"]
        runs.append((refresh_kind, kwargs["workspace_id"], kwargs["fill_missing_vectors"], kwargs["force"]))
        assert kwargs["fill_missing_vectors"] is True
        assert kwargs["force"] is True
        batch_native_refresh.clear_pending(kwargs["state_dir"])
        history = batch_native_refresh.active_workspace_history_path(kwargs["state_dir"])
        history.parent.mkdir(parents=True, exist_ok=True)
        with history.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"reason": batch_native_refresh.native_refresh_reason(refresh_kind, "cutover"), "current": {"workspace_id": kwargs["workspace_id"]}}) + "\n")
        status_after_policy = batch_native_refresh.native_refresh_policy_status(kwargs["state_dir"])
        marked = batch_native_refresh.mark_full_rebuild_pending_if_due(
            kwargs["state_dir"],
            kwargs["root"],
            refresh_kind=refresh_kind,
            status_after_policy=status_after_policy,
        )
        status_after = batch_native_refresh.status(kwargs["root"], kwargs["state_dir"])
        return {
            "cutover": True,
            "skipped": False,
            "refresh_kind": refresh_kind,
            "fill_missing_vectors": kwargs["fill_missing_vectors"],
            "vector_cache_required": True,
            "policy_native_pending": marked,
            "status_after": status_after,
        }

    monkeypatch.setattr(batch_wiki_integration.batch_native_refresh, "refresh_cutover", fake_refresh_cutover)

    code, payload = batch_wiki_integration.run_native_refresh_after_wiki_integration(root, state, workdir=ROOT, reason="threshold")

    assert code == 0
    assert [row[0] for row in runs] == ["incremental", "full-rebuild"]
    assert [run["refresh_kind"] for run in payload["runs"]] == ["incremental", "full-rebuild"]
    assert all(row[2] is True for row in runs)
    assert payload["status_after"]["should_refresh"] is False
    assert batch_native_refresh.pending_entries(state) == []


def test_native_refresh_followthrough_surfaces_cutover_failure_without_clearing_pending(tmp_path: Path, monkeypatch) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    watched_dir = tmp_path / "watched"
    watched_dir.mkdir()
    batch_native_refresh.mark_pending(state, root, reason="wiki-integration:threshold")
    _set_native_cutover_env(monkeypatch, watched_dir)

    def fake_refresh_cutover(**kwargs):
        raise RuntimeError("native service smoke failed")

    monkeypatch.setattr(batch_wiki_integration.batch_native_refresh, "refresh_cutover", fake_refresh_cutover)

    code, payload = batch_wiki_integration.run_native_refresh_after_wiki_integration(root, state, workdir=ROOT, reason="threshold")

    assert code == 16
    assert payload["failure"]["reason"] == "native-refresh-failed"
    assert "native service smoke failed" in payload["failure"]["message"]
    assert payload["status_after"]["should_refresh"] is True
    assert len(batch_native_refresh.pending_entries(state)) == 1


def test_batch_wiki_integration_auto_integrate_records_failure_if_runner_leaves_ledger_pending(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    for idx in range(10):
        mark_pending_wiki_integration(state, root, raw_path=f"raw/clip/2601/260107{idx:02d}_Paper.md", title=f"Paper {idx}")
    noop_runner = tmp_path / "noop_integrator.py"
    write(noop_runner, "print('noop integration runner returned without clearing ledger')\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ops.batch_wiki_integration",
            "auto-integrate",
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--reason",
            "threshold",
            "--integration-command",
            f"{sys.executable} {noop_runner}",
        ],
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert result.returncode == 12
    assert payload["ran"] is True
    assert payload["post_status"]["should_integrate"] is True
    assert payload["failure"]["reason"] == "auto-integrate-incomplete"
    assert load_pending_wiki_integration_ledger(state)["last_failed_integration"]["reason"] == "auto-integrate-incomplete"
