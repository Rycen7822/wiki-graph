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
