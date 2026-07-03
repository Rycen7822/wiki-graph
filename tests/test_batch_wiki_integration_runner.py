import sys
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from support import sample_wiki, write  # noqa: E402
from ops import batch_native_refresh  # noqa: E402
from ops import batch_wiki_integration  # noqa: E402
from ops.wiki_native_wiki_integration_bridge import clear_pending_wiki_integration_after_success  # noqa: E402
from ops.wiki_native_wiki_checks import wiki_root_machine_pollution  # noqa: E402
from ops.wiki_native_wiki_integration_pending import DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD  # noqa: E402
from ops.wiki_native_wiki_integration_pending import load_pending_wiki_integration_ledger  # noqa: E402
from ops.wiki_native_wiki_integration_pending import mark_pending_wiki_integration  # noqa: E402
from ops.wiki_native_wiki_integration_pending import pending_wiki_integration_status  # noqa: E402
def test_batch_wiki_integration_auto_integrate_defaults_to_local_runner_at_threshold(tmp_path: Path) -> None:
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
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["ran"] is True
    assert payload["runner"] == "local"
    assert payload["command"][0] == "integrate-local"
    assert payload["local_result"]["apply"]["operations_applied"] >= 11
    assert payload["local_result"]["validation"]["errors_count"] == 0
    assert "input_fingerprints" not in payload["local_result"]["validation"]
    assert payload["post_status"]["pending_count"] == 0
    assert load_pending_wiki_integration_ledger(state)["pending"] == []
    assert len(batch_native_refresh.pending_entries(state)) == 1


def test_batch_wiki_integration_auto_integrate_runs_configured_runner_at_threshold_and_requires_cleared_ledger(tmp_path: Path) -> None:
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
            f"{sys.executable} {fake_runner}",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["ran"] is True
    assert payload["pre_status"]["should_integrate"] is True
    assert payload["post_status"]["should_integrate"] is False
    assert payload["post_status"]["pending_count"] == 0
    assert payload["prompt_path"].endswith(".md")
    assert payload["plan_path"].endswith("_wiki_integration_plan.json")
    assert payload["plan_hash"]
    assert payload["plan_operations"] >= 1
    assert payload["prompt_chars"] < 5000
    assert Path(payload["plan_path"]).exists()
    assert "Plan artifact" in (state / "fake_runner_seen.txt").read_text(encoding="utf-8")
    assert "batch wiki integration" in (state / "fake_runner_seen.txt").read_text(encoding="utf-8")
    assert load_pending_wiki_integration_ledger(state)["pending"] == []
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
