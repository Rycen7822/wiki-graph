import sys
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from support import sample_wiki, write  # noqa: E402
from ops import batch_native_refresh  # noqa: E402
from ops import batch_wiki_integration  # noqa: E402
from ops.wiki_integration_plan import _canonical_plan_hash  # noqa: E402
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


def _write_plan(path: Path, root: Path, state: Path, operations: list[dict], *, compiled_page_writes: list[dict] | None = None, plan_hash: str | None = None) -> dict:
    compiled_page_writes = compiled_page_writes or []
    plan = {
        "schema_version": 1,
        "dry_run": True,
        "writes_wiki": False,
        "root": str(root),
        "state_dir": str(state),
        "reason": "threshold",
        "operations": operations,
        "compiled_page_writes": compiled_page_writes,
    }
    plan["plan_hash"] = plan_hash or _canonical_plan_hash(operations, compiled_page_writes)
    write(path, json.dumps(plan, ensure_ascii=False, indent=2))
    return plan


def test_apply_plan_cli_updates_machine_owned_maps_and_log(tmp_path: Path, capsys) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wiki-graph" / "state"
    write(root / "log.md", "# Wiki Log\n")
    raw_path = "raw/clip/2601/26010102_New-Paper.md"
    plan_path = state / "wiki_integration_plans" / "plan.json"
    plan = _write_plan(
        plan_path,
        root,
        state,
        [
            {
                "op": "raw_map_upsert",
                "raw_path": raw_path,
                "title": "New Paper",
                "source_id": "https://arxiv.org/abs/2601.0102",
                "required_sections": ["summary", "abstract"],
            },
            {"op": "topic_map_route", "topic": "local execution", "raw_path": raw_path, "title": "New Paper"},
            {"op": "log_batch_entry", "routed_paths": [raw_path], "review_paths": [], "reason": "threshold"},
        ],
    )

    code = batch_wiki_integration.main(
        [
            "apply-plan",
            "--root",
            str(root),
            "--state-dir",
            str(state),
            "--plan",
            str(plan_path),
            "--reason",
            "threshold",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["applied"] is True
    assert payload["writes_wiki"] is True
    assert payload["plan_hash"] == plan["plan_hash"]
    assert payload["operations_applied"] == 3
    assert raw_path in (root / "_meta" / "raw-clip-map.md").read_text(encoding="utf-8")
    assert "New Paper" in (root / "_meta" / "raw-clip-map.md").read_text(encoding="utf-8")
    assert "local execution" in (root / "_meta" / "topic-map.md").read_text(encoding="utf-8")
    assert (root / "log.md").read_text(encoding="utf-8").count(plan["plan_hash"]) == 1

    assert batch_wiki_integration.main(["apply-plan", "--root", str(root), "--state-dir", str(state), "--plan", str(plan_path)]) == 0
    capsys.readouterr()
    assert (root / "_meta" / "raw-clip-map.md").read_text(encoding="utf-8").count(raw_path) == 1
    assert (root / "_meta" / "topic-map.md").read_text(encoding="utf-8").count("local execution") == 1
    assert (root / "log.md").read_text(encoding="utf-8").count(plan["plan_hash"]) == 1


def test_apply_plan_cli_fails_closed_for_manual_or_unsupported_ops(tmp_path: Path, capsys) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wiki-graph" / "state"
    write(root / "log.md", "# Wiki Log\n")
    before = {
        path: path.read_text(encoding="utf-8")
        for path in [root / "_meta" / "raw-clip-map.md", root / "_meta" / "topic-map.md", root / "log.md"]
    }
    cases = [
        ([{"op": "review_queue_add", "raw_path": "raw/clip/2601/review.md", "title": "Review"}], [], "manual_review_required"),
        ([{"op": "unknown_op", "raw_path": "raw/clip/2601/bad.md"}], [], "unsupported_operation"),
        ([], [{"path": "concepts/new.md", "content": "# New"}], "compiled_page_writes_not_supported"),
    ]
    for idx, (operations, compiled, expected_error) in enumerate(cases):
        plan_path = state / "wiki_integration_plans" / f"bad-{idx}.json"
        _write_plan(plan_path, root, state, operations, compiled_page_writes=compiled)

        code = batch_wiki_integration.main(["apply-plan", "--root", str(root), "--state-dir", str(state), "--plan", str(plan_path)])
        payload = json.loads(capsys.readouterr().out)

        assert code == 13
        assert payload["applied"] is False
        assert expected_error in payload["errors"]
        for path, text in before.items():
            assert path.read_text(encoding="utf-8") == text


def test_apply_plan_cli_rejects_plan_hash_mismatch_without_writes(tmp_path: Path, capsys) -> None:
    root = sample_wiki(tmp_path)
    state = tmp_path / "work" / "wiki-graph" / "state"
    raw_map = root / "_meta" / "raw-clip-map.md"
    before = raw_map.read_text(encoding="utf-8")
    plan_path = state / "wiki_integration_plans" / "bad-hash.json"
    _write_plan(
        plan_path,
        root,
        state,
        [{"op": "raw_map_upsert", "raw_path": "raw/clip/2601/new.md", "title": "New"}],
        plan_hash="not-the-canonical-hash",
    )

    code = batch_wiki_integration.main(["apply-plan", "--root", str(root), "--state-dir", str(state), "--plan", str(plan_path)])
    payload = json.loads(capsys.readouterr().out)

    assert code == 13
    assert payload["applied"] is False
    assert "plan_hash_mismatch" in payload["errors"]
    assert raw_map.read_text(encoding="utf-8") == before
