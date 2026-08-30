import multiprocessing as mp
from pathlib import Path

import pytest

from ops.wiki_mutation_lock import WikiMutationLockTimeout, atomic_write_json, wiki_mutation_lock
from ops.wiki_native_wiki_integration_pending import load_pending_wiki_integration_ledger


def _mark_pending_worker(state_dir: str, root: str, index: int) -> str:
    from ops.wiki_native_wiki_integration_pending import mark_pending_wiki_integration

    raw_path = f"raw/clip/2601/260101{index:02d}_Paper-{index}.md"
    mark_pending_wiki_integration(Path(state_dir), Path(root), raw_path=raw_path, title=f"Paper {index}")
    return raw_path


def _lock_timeout_worker(state_dir: str) -> str:
    try:
        with wiki_mutation_lock(Path(state_dir), timeout=0.1, poll_interval=0.01):
            return "acquired"
    except WikiMutationLockTimeout as exc:
        return exc.code


def test_wiki_mutation_lock_is_reentrant_and_atomic_json_uses_unique_temp(tmp_path: Path) -> None:
    state = tmp_path / "state"
    target = state / "payload.json"

    with wiki_mutation_lock(state):
        with wiki_mutation_lock(state):
            atomic_write_json(target, {"value": 1})

    assert target.read_text(encoding="utf-8").endswith("\n")
    assert list(state.glob(".payload.json.*.tmp")) == []


def test_wiki_mutation_lock_times_out_in_another_process(tmp_path: Path) -> None:
    state = tmp_path / "state"
    context = mp.get_context("spawn")

    with wiki_mutation_lock(state):
        with context.Pool(1) as pool:
            result = pool.apply(_lock_timeout_worker, (str(state),))

    assert result == "mutation_lock_timeout"


@pytest.mark.parametrize("count", [8])
def test_concurrent_pending_marks_do_not_lose_entries(tmp_path: Path, count: int) -> None:
    root = tmp_path / "wiki"
    state = tmp_path / "state"
    (root / "raw" / "clip" / "2601").mkdir(parents=True)
    context = mp.get_context("spawn")

    with context.Pool(count) as pool:
        paths = pool.starmap(
            _mark_pending_worker,
            [(str(state), str(root), index) for index in range(1, count + 1)],
        )

    ledger = load_pending_wiki_integration_ledger(state)
    assert sorted(item["raw_path"] for item in ledger["pending"]) == sorted(paths)
    assert len(ledger["pending"]) == count
