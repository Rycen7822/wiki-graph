import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ops import raw_fast_batch
from ops.raw_fast_publish import publish_raw_note
from ops.wiki_native_wiki_integration_pending import (
    load_pending_wiki_integration_ledger,
    mark_pending_wiki_integration,
    mark_pending_wiki_integration_batch,
)
from support import sample_wiki, write


BODY = """## 一句话总结

A compact staged note is published only by the batch parent.

## 论文摘要（中文）

该测试说明 worker 草稿可以在统一 closeout 中安全发布。

## Motivation

Parallel reading should not create parallel canonical writers.

## Methodology

The parent renders each body with script-owned metadata and allocates the final sequence under one lock.

## 关键实验结果 / 作者结论

The accepted items retain input order while a failed worker remains isolated.

## 对未来研究的启发

The same owner can support single-note and batch closeout paths.

## 可能的局限

This fixture checks deterministic mechanics rather than semantic depth.

## 可继续追问的问题

How should a later version expose optional reviewer feedback?
"""


def _init(tmp_path: Path, count: int, *, batch_id: str = "batch-test", threshold: int | None = None, runner: str = "local"):
    root = sample_wiki(tmp_path)
    state_dir = tmp_path / "state"
    result = raw_fast_batch.init_batch(
        root=root,
        state_dir=state_dir,
        urls=[f"https://arxiv.org/abs/2608.{index:05d}" for index in range(count)],
        tmp_root=tmp_path / "batches",
        batch_id=batch_id,
        threshold=threshold,
        runner=runner,
    )
    return root, state_dir, Path(result["manifest_path"]), result


def _ready(item: dict, index: int, *, source: str | None = None) -> None:
    workdir = Path(item["workdir"])
    source = source or f"https://arxiv.org/abs/2608.{index:05d}"
    write(
        workdir / "candidate_frontmatter.json",
        json.dumps(
            {
                "title": f"Batch Paper {index}",
                "source": source,
                "type": "raw-note",
                "domain": "machine-learning",
                "tags": ["batch-test"],
                "topic_hints": ["batch-topic"],
            }
        ),
    )
    write(workdir / "raw_body_draft.md", BODY)
    write(
        workdir / "worker_result.json",
        json.dumps(
            {
                "ok": True,
                "status": "ready",
                "source_id": source,
                "topic_hints": ["batch-topic"],
                "required_sections": ["index.md"],
                "resource_status_summary": "batch test",
            }
        ),
    )


def _prepared_for_contract(item: dict, index: int, *, source: str | None = None) -> Path:
    workdir = Path(item["workdir"])
    source = source or f"https://arxiv.org/abs/2608.{index:05d}"
    writing_contract = workdir / "writing-contract.md"
    write(writing_contract, "# Worker writing contract\n")
    write(workdir / "agent_handoff.md", "# Raw-fast agent handoff\n")
    write(
        workdir / "agent_handoff.json",
        json.dumps(
            {
                "status": "ready",
                "resource_review_required": False,
                "writing_contract_refs": [{"path": str(writing_contract)}],
                "closeout_args": {"resource_status_summary": "script-owned resource summary"},
            }
        ),
    )
    write(
        workdir / "candidate_frontmatter.json",
        json.dumps(
            {
                "title": f"Batch Paper {index}",
                "source": source,
                "type": "raw-note",
                "domain": "machine-learning",
                "tags": ["batch-test"],
                "topic_hints": ["batch-topic"],
            }
        ),
    )
    return workdir


def _manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_init_creates_isolated_workdirs_without_canonical_writes(tmp_path: Path) -> None:
    root, state_dir, manifest_path, result = _init(tmp_path, 2)
    raw_files = list((root / "raw" / "clip").glob("*/*.md"))

    assert manifest_path.is_file()
    assert result["phase"] == "staging"
    workdirs = [item["workdir"] for item in result["items"]]
    assert len(workdirs) == len(set(workdirs)) == 2
    assert all(Path(path).is_dir() for path in workdirs)
    assert result["items"][0]["worker_contract"]["canonical_writes_allowed"] is False
    assert [path.name for path in raw_files] == ["26010101_Foo-Paper.md"]
    assert not (state_dir / "pending_wiki_integration.json").exists()


def test_init_cli_returns_zero_with_ok_payload(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = sample_wiki(tmp_path)
    code = raw_fast_batch.main(
        [
            "init",
            "--root",
            str(root),
            "--state-dir",
            str(tmp_path / "state"),
            "--tmp-root",
            str(tmp_path / "batches"),
            "--batch-id",
            "cli-init",
            "--url",
            "https://arxiv.org/abs/2608.20195",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["ok"] is True
    assert Path(payload["manifest_path"]).is_file()


def test_worker_contracts_are_concise_and_delegate_only_generated_paths(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    result = raw_fast_batch.init_batch(
        root=root,
        state_dir=tmp_path / "state",
        urls=["https://paperswithcode.co/paper/2608.23392"],
        tmp_root=tmp_path / "batches",
        batch_id="worker-contract",
    )
    canonical_source = "https://arxiv.org/abs/2608.23392"
    workdir = _prepared_for_contract(result["items"][0], 0, source=canonical_source)

    payload = raw_fast_batch.materialize_worker_contracts(Path(result["manifest_path"]))
    contract = json.loads((workdir / "worker_contract.json").read_text(encoding="utf-8"))
    prompt = (workdir / "worker_prompt.md").read_text(encoding="utf-8")

    assert payload["ok"] is True
    assert payload["ready_count"] == 1
    assert set(contract) == {
        "version",
        "workdir",
        "title",
        "source_id",
        "handoff",
        "writing_contracts",
        "candidate_frontmatter",
        "body_draft",
        "worker_result",
        "finish",
    }
    assert contract["source_id"] == canonical_source
    assert contract["finish"]["argv"][-2:] == ["--contract", str(workdir / "worker_contract.json")]
    assert set(payload["worker_tasks"][0]) == {"goal", "context"}
    assert canonical_source not in payload["worker_tasks"][0]["context"]
    assert "arxiv:2608.23392" not in payload["worker_tasks"][0]["context"]
    assert len(prompt) < 1600
    assert "Do not write `worker_result.json`" in prompt
    assert _manifest(Path(result["manifest_path"]))["items"][0]["status"] == "worker_ready"


def test_worker_contracts_keep_ready_items_when_one_prepare_is_missing(tmp_path: Path) -> None:
    _root, _state_dir, manifest_path, result = _init(tmp_path, 2, batch_id="worker-contract-partial")
    _prepared_for_contract(result["items"][0], 0)

    payload = raw_fast_batch.materialize_worker_contracts(manifest_path)
    items = _manifest(manifest_path)["items"]

    assert payload["ok"] is True
    assert payload["ready_count"] == 1
    assert payload["failed_count"] == 1
    assert len(payload["worker_tasks"]) == 1
    assert items[0]["status"] == "worker_ready"
    assert items[1]["status"] == "worker_contract_failed"
    assert items[1]["error"] == "agent_handoff_missing"


def test_finish_worker_writes_exact_identity_and_closeout_consumes_it(tmp_path: Path) -> None:
    root, _state_dir, manifest_path, result = _init(tmp_path, 1, batch_id="finish-worker")
    canonical_source = "https://arxiv.org/abs/2608.00000"
    workdir = _prepared_for_contract(result["items"][0], 0, source=canonical_source)
    raw_fast_batch.materialize_worker_contracts(manifest_path)
    write(workdir / "raw_body_draft.md", BODY)

    finished = raw_fast_batch.finish_worker(workdir / "worker_contract.json")
    worker_result = json.loads((workdir / "worker_result.json").read_text(encoding="utf-8"))
    report = raw_fast_batch.closeout_batch(manifest_path, auto_integrate=False)

    assert finished["ok"] is True
    assert worker_result == {
        "ok": True,
        "required_sections": ["index.md"],
        "resource_status_summary": "script-owned resource summary",
        "source_id": canonical_source,
        "status": "ready",
        "topic_hints": ["batch-topic"],
    }
    assert report["ok"] is True
    assert report["published_count"] == 1
    assert (root / report["raw_files"][0]).is_file()


def test_finish_worker_rejects_protected_identity_change_and_removes_stale_result(tmp_path: Path) -> None:
    _root, _state_dir, manifest_path, result = _init(tmp_path, 1, batch_id="finish-conflict")
    workdir = _prepared_for_contract(result["items"][0], 0)
    raw_fast_batch.materialize_worker_contracts(manifest_path)
    write(workdir / "raw_body_draft.md", BODY)
    write(workdir / "worker_result.json", json.dumps({"ok": True, "status": "ready"}))
    frontmatter = json.loads((workdir / "candidate_frontmatter.json").read_text(encoding="utf-8"))
    frontmatter["source"] = "https://arxiv.org/abs/9999.99999"
    write(workdir / "candidate_frontmatter.json", json.dumps(frontmatter))

    finished = raw_fast_batch.finish_worker(workdir / "worker_contract.json")

    assert finished["ok"] is False
    assert finished["error"] == "protected_identity_conflict"
    assert not (workdir / "worker_result.json").exists()


def test_worker_contract_and_finish_worker_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _root, _state_dir, manifest_path, result = _init(tmp_path, 1, batch_id="worker-cli")
    workdir = _prepared_for_contract(result["items"][0], 0)

    contract_code = raw_fast_batch.main(["worker-contracts", "--manifest", str(manifest_path)])
    contract_payload = json.loads(capsys.readouterr().out)
    write(workdir / "raw_body_draft.md", BODY)
    finish_code = raw_fast_batch.main(["finish-worker", "--contract", str(workdir / "worker_contract.json")])
    finish_payload = json.loads(capsys.readouterr().out)

    assert contract_code == 0
    assert contract_payload["ready_count"] == 1
    assert finish_code == 0
    assert finish_payload["source_id"] == "https://arxiv.org/abs/2608.00000"


def test_partial_batch_publishes_ready_items_in_input_order_and_recovers_idempotently(tmp_path: Path) -> None:
    root, state_dir, manifest_path, result = _init(tmp_path, 3)
    initial_raw_count = len(list((root / "raw" / "clip").glob("*/*.md")))
    _ready(result["items"][0], 0)
    _ready(result["items"][2], 2)
    write(Path(result["items"][1]["workdir"]) / "worker_result.json", json.dumps({"ok": False, "error": "draft_failed"}))

    first = raw_fast_batch.closeout_batch(manifest_path, auto_integrate=False)

    assert first["ok"] is True
    assert first["raw_status"] == "partial"
    assert first["input_count"] == 3
    assert first["ready_count"] == 2
    assert first["published_count"] == 2
    assert first["reused_count"] == 0
    assert first["failed_count"] == 1
    assert [item["input_index"] for item in first["items"]] == [0, 1, 2]
    published = [item for item in first["items"] if item["status"] == "published"]
    matches = [re.search(r"(\d{6})(\d{2})_", item["raw_file"]) for item in published]
    assert all(match is not None for match in matches)
    assert [int(match.group(2)) for match in matches if match] == [1, 2]
    ledger = load_pending_wiki_integration_ledger(state_dir)
    assert len(ledger["pending"]) == 2
    log_text = (root / "log.md").read_text(encoding="utf-8")
    assert all(log_text.count(item["raw_file"]) == 1 for item in published)

    second = raw_fast_batch.closeout_batch(manifest_path, auto_integrate=False)

    assert second["ok"] is True
    assert second["published_count"] == 0
    assert second["reused_count"] == 2
    assert second["failed_count"] == 1
    assert second["log_entries_appended"] == 0
    assert len(load_pending_wiki_integration_ledger(state_dir)["pending"]) == 2
    assert len(list((root / "raw" / "clip").glob("*/*.md"))) == initial_raw_count + 2


def test_batch_threshold_makes_exactly_one_local_decision_after_publish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, state_dir, manifest_path, result = _init(tmp_path, 5, threshold=10)
    for index, item in enumerate(result["items"]):
        _ready(item, index)
    mark_pending_wiki_integration_batch(
        state_dir,
        root,
        [
            {
                "raw_path": f"raw/clip/2608/260830{index + 1:02d}_Old-{index}.md",
                "title": f"Old {index}",
                "source_id": f"old:{index}",
                "topic_hints": ["batch-topic"],
                "status": "raw_saved",
            }
            for index in range(7)
        ],
        threshold=10,
    )
    calls: list[dict] = []

    def _run_once(*args, **kwargs):
        calls.append(kwargs)
        return 0, {"plan_path": str(tmp_path / "plan.json"), "post_status": {"blocking_pending_count": 12}}

    monkeypatch.setattr(raw_fast_batch, "run_auto_integration", _run_once)

    report = raw_fast_batch.closeout_batch(manifest_path, auto_integrate=True)

    assert report["ok"] is True
    assert report["published_count"] == 5
    assert report["wiki"]["pending_before"] == 7
    assert report["wiki"]["pending_after_publish"] == 12
    assert report["wiki"]["decision_count"] == 1
    assert report["wiki"]["integration_attempted"] is True
    assert len(calls) == 1
    assert calls[0]["runner"] == "local"
    assert calls[0]["defer_native_refresh"] is True


def test_preexisting_review_does_not_block_publish_and_external_runner_defers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, state_dir, manifest_path, result = _init(tmp_path, 1, threshold=1, runner="external")
    _ready(result["items"][0], 0)
    mark_pending_wiki_integration(
        state_dir,
        root,
        raw_path="raw/clip/2608/26083001_Review.md",
        title="Review",
        source_id="review:1",
        topic_hints=[],
        status="needs_review",
        threshold=1,
    )

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("external batch runner must never execute")

    monkeypatch.setattr(raw_fast_batch, "run_auto_integration", _unexpected)

    report = raw_fast_batch.closeout_batch(manifest_path, auto_integrate=True)

    assert report["published_count"] == 1
    assert report["raw_status"] == "complete"
    assert report["wiki_status"] == "needs_review"
    assert report["wiki"]["integration_attempted"] is False
    assert report["wiki"]["integration_deferred_reason"] == "external_runner_deferred"
    assert (root / report["raw_files"][0]).is_file()
    assert len(load_pending_wiki_integration_ledger(state_dir)["pending"]) == 2


def test_recovery_hash_conflict_stops_without_overwrite(tmp_path: Path) -> None:
    root, _state_dir, manifest_path, result = _init(tmp_path, 1)
    _ready(result["items"][0], 0)
    first = raw_fast_batch.closeout_batch(manifest_path, auto_integrate=False)
    raw_path = root / first["raw_files"][0]
    raw_path.write_text("externally changed\n", encoding="utf-8")

    second = raw_fast_batch.closeout_batch(manifest_path, auto_integrate=False)

    assert second["ok"] is False
    assert second["phase"] == "publish_failed"
    assert second["failure"]["error"] == "recorded_raw_hash_conflict"
    assert raw_path.read_text(encoding="utf-8") == "externally changed\n"


def test_local_integration_failure_keeps_raw_and_pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, state_dir, manifest_path, result = _init(tmp_path, 1, threshold=1)
    _ready(result["items"][0], 0)

    def _fail_integration(*_args, **_kwargs):
        return 13, {"failure": {"reason": "validation_failed"}}

    monkeypatch.setattr(raw_fast_batch, "run_auto_integration", _fail_integration)

    report = raw_fast_batch.closeout_batch(manifest_path, auto_integrate=True)

    assert report["ok"] is False
    assert report["phase"] == "integration_failed"
    assert report["published_count"] == 1
    assert report["wiki"]["integration_attempted"] is True
    assert report["graph_status"] == "pending"
    assert (root / report["raw_files"][0]).is_file()
    assert len(load_pending_wiki_integration_ledger(state_dir)["pending"]) == 1


def test_single_publisher_and_batch_closeout_share_sequence_owner(tmp_path: Path) -> None:
    root, state_dir, manifest_path, result = _init(tmp_path, 1)
    _ready(result["items"][0], 0)

    with ThreadPoolExecutor(max_workers=2) as executor:
        batch_future = executor.submit(raw_fast_batch.closeout_batch, manifest_path, auto_integrate=False)
        single_future = executor.submit(
            publish_raw_note,
            root,
            state_dir,
            title="Concurrent Single",
            raw_text="single publisher\n",
        )
        batch_report = batch_future.result()
        single_result = single_future.result()

    assert batch_report["ok"] is True
    batch_raw = batch_report["raw_files"][0]
    single_raw = single_result["raw_file"]
    assert batch_raw != single_raw
    batch_match = re.search(r"(\d{6})(\d{2})_", batch_raw)
    single_match = re.search(r"(\d{6})(\d{2})_", single_raw)
    assert batch_match is not None and single_match is not None
    assert batch_match.group(1) == single_match.group(1)
    assert {int(batch_match.group(2)), int(single_match.group(2))} == {1, 2}
