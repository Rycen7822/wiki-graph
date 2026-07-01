import sys
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from support import sample_wiki, write  # noqa: E402
from ops import batch_native_refresh  # noqa: E402
from ops import batch_wiki_integration  # noqa: E402
from ops.wiki_native_lib import clear_pending_wiki_integration_after_success  # noqa: E402
from ops.wiki_native_wiki_checks import wiki_root_machine_pollution  # noqa: E402
from ops.wiki_native_wiki_integration_pending import DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD  # noqa: E402
from ops.wiki_native_wiki_integration_pending import load_pending_wiki_integration_ledger  # noqa: E402
from ops.wiki_native_wiki_integration_pending import mark_pending_wiki_integration  # noqa: E402
from ops.wiki_native_wiki_integration_pending import pending_wiki_integration_status  # noqa: E402


def test_mark_pending_wiki_integration_tracks_raw_fast_queue_without_wiki_pollution(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    entry = mark_pending_wiki_integration(
        state,
        root,
        raw_path="raw/clip/2601/26010101_Foo-Paper.md",
        title="Foo Paper",
        source_id="https://arxiv.org/abs/2601.0101",
        topic_hints=["agents", "rag"],
        required_sections=["summary", "methodology"],
        resource_status_summary="official abs/pdf verified",
    )
    ledger = load_pending_wiki_integration_ledger(state)
    assert entry["status"] == "raw_saved"
    assert ledger["threshold"] == DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD == 10
    assert ledger["dirty"] is True
    assert len(ledger["pending"]) == 1
    assert ledger["pending"][0]["source_id"] == "https://arxiv.org/abs/2601.0101"
    assert ledger["pending"][0]["topic_hints"] == ["agents", "rag"]
    assert (state / "pending_wiki_integration.json").exists()
    assert not (root / "pending_wiki_integration.json").exists()
    assert wiki_root_machine_pollution(root) == []
    status = pending_wiki_integration_status(root, state)
    assert status["pending_count"] == 1
    assert status["should_integrate"] is False


def test_pending_wiki_integration_status_triggers_at_threshold_and_clears_after_success(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    for idx in range(9):
        mark_pending_wiki_integration(state, root, raw_path=f"raw/clip/2601/260101{idx:02d}_Paper.md", title=f"Paper {idx}")
    below = pending_wiki_integration_status(root, state)
    assert below["pending_count"] == 9
    assert below["actionable_pending_count"] == 9
    assert below["threshold"] == DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD == 10
    assert below["should_integrate"] is False

    mark_pending_wiki_integration(state, root, raw_path="raw/clip/2601/26010109_Paper.md", title="Paper 9")
    status = pending_wiki_integration_status(root, state)
    assert status["pending_count"] == 10
    assert status["actionable_pending_count"] == 10
    assert status["threshold"] == DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD == 10
    assert status["should_integrate"] is True
    assert "pending_threshold_reached" in status["reasons"]

    cleared = clear_pending_wiki_integration_after_success(root, state, reason="threshold")
    assert cleared["cleared_count"] == 10
    assert cleared["remaining_pending_count"] == 0
    ledger = load_pending_wiki_integration_ledger(state)
    assert ledger["pending"] == []
    assert ledger["dirty"] is False
    assert ledger["last_successful_integration_raw_count"] == 1


def test_pending_wiki_integration_status_uses_persisted_threshold(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    for idx in range(5):
        mark_pending_wiki_integration(state, root, raw_path=f"raw/clip/2601/260102{idx:02d}_Paper.md", title=f"Paper {idx}", threshold=5)

    status = pending_wiki_integration_status(root, state)

    assert status["pending_count"] == 5
    assert status["actionable_pending_count"] == 5
    assert status["threshold"] == 5
    assert status["should_integrate"] is True
    assert "pending_threshold_reached" in status["reasons"]


def test_review_wiki_integration_status_routes_manual_review_items(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    mark_pending_wiki_integration(state, root, raw_path="raw/clip/2601/26010400_Needs-Review.md", title="Needs Review", status="needs_review")

    wiki_status = pending_wiki_integration_status(root, state, reason="pre-query")

    assert wiki_status["actionable_pending_count"] == 0
    assert wiki_status["review_pending_count"] == 1
    assert "pending_items_need_review" in wiki_status["reasons"]
    assert wiki_status["next_required_action"] == "manual_review"


def test_wiki_integration_plan_is_order_independent_and_keeps_ambiguous_items_in_review_queue(tmp_path: Path) -> None:
    from ops.wiki_integration_plan import build_wiki_integration_plan

    root_a = sample_wiki(tmp_path / "a")
    root_b = sample_wiki(tmp_path / "b")
    state_a = tmp_path / "a" / "state"
    state_b = tmp_path / "b" / "state"
    items = [
        {
            "raw_path": "raw/clip/2601/26010102_Routed.md",
            "title": "Routed",
            "source_id": "source:routed",
            "topic_hints": ["retrieval", "agents"],
            "required_sections": ["summary"],
        },
        {
            "raw_path": "raw/clip/2601/26010101_Ambiguous.md",
            "title": "Ambiguous",
            "source_id": "source:ambiguous",
            "topic_hints": [],
            "required_sections": ["summary"],
        },
    ]
    for item in items:
        mark_pending_wiki_integration(state_a, root_a, **item)
    for item in reversed(items):
        mark_pending_wiki_integration(state_b, root_b, **item)

    plan_a = build_wiki_integration_plan(root_a, state_a, reason="manual")
    plan_b = build_wiki_integration_plan(root_b, state_b, reason="manual")

    assert plan_a["plan_hash"] == plan_b["plan_hash"]
    assert plan_a["dry_run"] is True
    assert plan_a["writes_wiki"] is False
    assert plan_a["compiled_page_writes"] == []
    operations = plan_a["operations"]
    assert [op["raw_path"] for op in operations if op["op"] == "raw_map_upsert"] == [
        "raw/clip/2601/26010102_Routed.md"
    ]
    review_ops = [op for op in operations if op["op"] == "review_queue_add"]
    assert review_ops == [
        {
            "op": "review_queue_add",
            "raw_path": "raw/clip/2601/26010101_Ambiguous.md",
            "title": "Ambiguous",
            "reason": "missing_topic_hints",
        }
    ]
    assert wiki_root_machine_pollution(root_a) == []


def test_clear_pending_wiki_integration_marks_integrated_items_for_native_refresh(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    first = "raw/clip/2601/26010102_Raw-Fast-A.md"
    second = "raw/clip/2601/26010103_Raw-Fast-B.md"
    mark_pending_wiki_integration(state, root, raw_path=first, title="Raw Fast A", required_sections=["summary", "methodology"])
    mark_pending_wiki_integration(state, root, raw_path=second, title="Raw Fast B", required_sections=["summary"])

    cleared = clear_pending_wiki_integration_after_success(root, state, integrated_paths=[first], reason="threshold")

    assert cleared["cleared_count"] == 1
    assert cleared["remaining_pending_count"] == 1
    assert cleared["marked_native_pending_count"] == 1
    wiki_ledger = load_pending_wiki_integration_ledger(state)
    assert [item["raw_path"] for item in wiki_ledger["pending"]] == [second]
    assert wiki_ledger["dirty"] is True
    assert not (state / "pending_wikigraph_refresh.json").exists()
    native_entries = batch_native_refresh.pending_entries(state)
    assert len(native_entries) == 1
    assert native_entries[0]["reason"] == "wiki-integration:threshold"


def test_clear_pending_wiki_integration_without_integrated_paths_marks_native_refresh_once(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    first = "raw/clip/2601/26010102_Raw-Fast-A.md"
    second = "raw/clip/2601/26010103_Raw-Fast-B.md"
    mark_pending_wiki_integration(state, root, raw_path=first, title="Raw Fast A")
    mark_pending_wiki_integration(state, root, raw_path=second, title="Raw Fast B")

    cleared = clear_pending_wiki_integration_after_success(root, state, reason="threshold")

    assert cleared["cleared_count"] == 2
    assert cleared["remaining_pending_count"] == 0
    assert cleared["marked_native_pending_count"] == 1
    assert load_pending_wiki_integration_ledger(state)["pending"] == []
    assert len(batch_native_refresh.pending_entries(state)) == 1


def test_batch_wiki_integration_cli_status_mark_and_clear_are_external(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wikigraph" / "state"
    module_cmd = [sys.executable, "-m", "ops.batch_wiki_integration"]

    mark = subprocess.run(
        [
            *module_cmd,
            "mark-pending",
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--raw-path",
            "raw/clip/2601/26010101_Foo-Paper.md",
            "--title",
            "Foo Paper",
            "--source-id",
            "https://arxiv.org/abs/2601.0101",
            "--topic-hint",
            "agents",
            "--required-section",
            "methodology",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    mark_payload = json.loads(mark.stdout)
    assert mark_payload["pending_count"] == 1
    assert mark_payload["threshold"] == DEFAULT_PENDING_WIKI_INTEGRATION_THRESHOLD == 10
    assert mark_payload["pending"][0]["topic_hints"] == ["agents"]
    assert mark_payload["pending"][0]["required_sections"] == ["methodology"]

    status = subprocess.run([*module_cmd, "status", "--root", str(root), "--state-dir", str(state)], check=True, text=True, capture_output=True)
    assert json.loads(status.stdout)["should_integrate"] is False
    assert not (root / "pending_wiki_integration.json").exists()

    clear = subprocess.run(
        [*module_cmd, "clear-success", "--root", str(root), "--state-dir", str(state), "--integrated-path", "raw/clip/2601/26010101_Foo-Paper.md"],
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(clear.stdout)["cleared_count"] == 1
    assert load_pending_wiki_integration_ledger(state)["pending"] == []


def test_batch_wiki_integration_prompt_uses_repo_local_workdir_paths(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wiki-graph" / "state"
    status = {
        "pending_count": 1,
        "actionable_pending_count": 1,
        "threshold": 10,
        "actionable_pending": [
            {
                "raw_path": "raw/clip/2606/26062906_skill-neologisms-towards-skill-based-continual-learning.md",
                "title": "Skill Neologisms",
                "source_id": "arXiv:2605.04970v2",
                "topic_hints": ["skill neologisms", "continual learning"],
            }
        ],
    }

    plan = {
        "plan_hash": "plan123",
        "operations": [
            {"op": "raw_map_upsert", "raw_path": "raw/clip/2606/26062906_skill-neologisms-towards-skill-based-continual-learning.md"},
            {"op": "review_queue_add", "raw_path": "raw/clip/2606/26062907_needs-review.md"},
        ],
    }
    plan_path = state / "wiki_integration_plans" / "plan.json"

    prompt = batch_wiki_integration.build_auto_integration_prompt(root, state, status, "manual", plan=plan, plan_path=plan_path)

    assert ("/home/" + "xu/project/wiki/wikigraph") not in prompt
    assert f"Native refresh workdir: `{ROOT}`" in prompt
    assert f"Plan artifact: `{plan_path}`" in prompt
    assert "plan_hash: `plan123`; operations=2; routed_raw_notes=1; review_queue_additions=1" in prompt
    assert "Skill Neologisms" not in prompt
    assert "python -m ops.validate_wiki" in prompt
    assert "--sync-raw-map-snapshot" in prompt
    assert "python -m ops.batch_wiki_integration clear-success" in prompt
    assert "python -m ops.batch_native_refresh status" in prompt
    assert "python -m ops.batch_native_refresh preflight-cutover" in prompt
    assert "python -m ops.batch_native_refresh refresh" in prompt
    assert "--fill-missing-vectors" in prompt
    assert "next_refresh_kind" in prompt
    assert "after 5 completed incremental graph updates" in prompt
    assert "prepare-only output is a prepared artifact, not live graph freshness" in prompt
    assert "Use bounded reads/searches for `_meta/raw-clip-map.md` and `_meta/topic-map.md`; keep large map files out of context." in prompt
    assert "references/raw-fast-batch-wiki-integration.md" in prompt
    assert "references/wiki-core-operations.md" in prompt
    assert "references/wiki-operational-pitfalls.md" not in prompt


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
        "from ops.wiki_native_lib import clear_pending_wiki_integration_after_success\n"
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
