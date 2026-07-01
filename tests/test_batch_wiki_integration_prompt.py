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
