import datetime as dt
import multiprocessing as mp
import re
from pathlib import Path

import pytest

from ops.raw_fast_publish import (
    RawPublishConflict,
    allocate_raw_paths_locked,
    publish_raw_note,
    publish_raw_text_exclusive_locked,
)
from ops.wiki_mutation_lock import wiki_mutation_lock
from support import sample_wiki


def _publish_worker(root: str, state_dir: str, index: int) -> str:
    result = publish_raw_note(
        Path(root),
        Path(state_dir),
        title=f"Concurrent Paper {index}",
        raw_text=f"note {index}\n",
        requested_raw_file="raw/clip/2608/26083101_Provisional.md",
    )
    return str(result["raw_file"])


def test_allocate_raw_paths_locked_preserves_input_order(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state_dir = tmp_path / "state"
    with wiki_mutation_lock(state_dir):
        paths = allocate_raw_paths_locked(
            root,
            ["First", "Second", "Third"],
            now=dt.datetime(2026, 8, 31, 12, 0),
        )

    assert paths == [
        "raw/clip/2608/26083101_First.md",
        "raw/clip/2608/26083102_Second.md",
        "raw/clip/2608/26083103_Third.md",
    ]


def test_concurrent_publishers_get_unique_sequences_without_overwrite(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state_dir = tmp_path / "state"
    context = mp.get_context("spawn")
    with context.Pool(8) as pool:
        paths = pool.starmap(_publish_worker, [(str(root), str(state_dir), index) for index in range(8)])

    assert len(paths) == len(set(paths)) == 8
    matches = [re.search(r"260831(\d{2})_", path) for path in paths]
    assert all(match is not None for match in matches)
    sequences = sorted(int(match.group(1)) for match in matches if match is not None)
    assert sequences == list(range(1, 9))
    contents = {(root / path).read_text(encoding="utf-8") for path in paths}
    assert contents == {f"note {index}\n" for index in range(8)}


def test_exclusive_create_rejects_unexpected_existing_target(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state_dir = tmp_path / "state"
    raw_file = "raw/clip/2608/26083101_Conflict.md"
    path = root / raw_file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("original\n", encoding="utf-8")

    with wiki_mutation_lock(state_dir):
        with pytest.raises(RawPublishConflict) as exc_info:
            publish_raw_text_exclusive_locked(root, raw_file, "replacement\n")

    assert exc_info.value.code == "raw_file_conflict"
    assert path.read_text(encoding="utf-8") == "original\n"


def test_existing_refresh_is_explicit_and_preserves_path(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state_dir = tmp_path / "state"
    raw_file = "raw/clip/2608/26083101_Existing.md"
    path = root / raw_file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("old\n", encoding="utf-8")

    result = publish_raw_note(
        root,
        state_dir,
        title="Existing",
        raw_text="new\n",
        requested_raw_file=raw_file,
        overwrite_existing=True,
    )

    assert result["raw_file"] == raw_file
    assert result["refreshed"] is True
    assert result["reallocated"] is False
    assert path.read_text(encoding="utf-8") == "new\n"
