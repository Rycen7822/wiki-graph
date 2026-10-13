import datetime as dt
import multiprocessing as mp
import re
from pathlib import Path

import pytest

from ops.raw_fast_publish import (
    RawPublishConflict,
    RawPublishError,
    allocate_raw_paths_locked,
    publish_raw_note,
    publish_raw_text_exclusive_locked,
    validate_raw_relative_path,
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


@pytest.mark.parametrize("existing_sequence", [0, 98, 999])
def test_concurrent_publishers_get_unique_sequences_without_overwrite(tmp_path: Path, existing_sequence: int) -> None:
    root = sample_wiki(tmp_path)
    state_dir = tmp_path / "state"
    existing_path = root / f"raw/clip/2608/260831{existing_sequence:02d}_Existing.md"
    requested_path = root / "raw/clip/2608/26083101_Provisional.md"
    if existing_sequence:
        existing_path.parent.mkdir(parents=True, exist_ok=True)
        existing_path.write_text("existing note\n", encoding="utf-8")
        # Occupy the worker's explicit request so every publisher must reallocate.
        requested_path.write_text("existing provisional note\n", encoding="utf-8")
    context = mp.get_context("spawn")
    with context.Pool(8) as pool:
        paths = pool.starmap(_publish_worker, [(str(root), str(state_dir), index) for index in range(8)])

    assert len(paths) == len(set(paths)) == 8
    matches = [re.search(r"260831(\d{2,})_", path) for path in paths]
    assert all(match is not None for match in matches)
    sequences = sorted(int(match.group(1)) for match in matches if match is not None)
    assert sequences == list(range(existing_sequence + 1, existing_sequence + 9))
    contents = {(root / path).read_text(encoding="utf-8") for path in paths}
    assert contents == {f"note {index}\n" for index in range(8)}
    if existing_sequence:
        assert existing_path.read_text(encoding="utf-8") == "existing note\n"
        assert requested_path.read_text(encoding="utf-8") == "existing provisional note\n"


@pytest.mark.parametrize("sequence", ["01", "100", "1000"])
def test_exclusive_create_rejects_unexpected_existing_target(tmp_path: Path, sequence: str) -> None:
    root = sample_wiki(tmp_path)
    state_dir = tmp_path / "state"
    raw_file = f"raw/clip/2608/260831{sequence}_Conflict.md"
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


@pytest.mark.parametrize("sequence", ["01", "99", "100", "999", "1000"])
def test_validate_raw_relative_path_accepts_legacy_and_expanded_sequences(sequence: str) -> None:
    match = validate_raw_relative_path(f"raw/clip/2610/261011{sequence}_Paper.md")

    assert match.groupdict() == {"yymm": "2610", "yymmdd": "261011", "seq": sequence, "slug": "Paper"}


@pytest.mark.parametrize(
    "raw_file",
    [
        "raw/clip/2609/26101101_Paper.md",
        "raw/clip/2609/261011100_Paper.md",
        "raw/clip/2610/../26101101_Paper.md",
        "raw/clip/2610/../261011100_Paper.md",
        "raw/clip/2610/261011100_../Paper.md",
        "/raw/clip/2610/261011100_Paper.md",
        "raw/clip/2610/2610111_Paper.md",
        "raw/clip/2610/261011100_Paper.txt",
    ],
)
def test_validate_raw_relative_path_rejects_invalid_paths(raw_file: str) -> None:
    with pytest.raises(RawPublishError) as exc_info:
        validate_raw_relative_path(raw_file)

    assert exc_info.value.code == "invalid_raw_file"


@pytest.mark.parametrize("existing_sequence", [98, 100, 999])
def test_allocate_raw_paths_locked_reloads_expanded_sequences_from_disk(tmp_path: Path, existing_sequence: int) -> None:
    root = sample_wiki(tmp_path)
    state_dir = tmp_path / "state"
    existing_path = root / f"raw/clip/2610/261011{existing_sequence:02d}_Existing.md"
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    existing_path.write_text("existing note\n", encoding="utf-8")

    with wiki_mutation_lock(state_dir):
        paths = allocate_raw_paths_locked(root, ["First", "Second", "Third"], now=dt.datetime(2026, 10, 11, 12, 0))

    assert paths == [
        f"raw/clip/2610/261011{existing_sequence + offset:02d}_{title}.md"
        for offset, title in enumerate(["First", "Second", "Third"], start=1)
    ]
    assert existing_path.read_text(encoding="utf-8") == "existing note\n"


def test_publish_requested_expanded_path_is_preserved_and_reallocated_on_collision(tmp_path: Path) -> None:
    root = sample_wiki(tmp_path)
    state_dir = tmp_path / "state"
    requested = "raw/clip/2610/261011100_Requested.md"
    first = publish_raw_note(root, state_dir, title="First", raw_text="first\n", requested_raw_file=requested)
    second = publish_raw_note(root, state_dir, title="Second", raw_text="second\n", requested_raw_file=requested)

    assert first["raw_file"] == requested
    assert first["reallocated"] is False
    assert second["raw_file"] == "raw/clip/2610/261011101_Second.md"
    assert second["reallocated"] is True
    assert (root / requested).read_text(encoding="utf-8") == "first\n"
    assert (root / second["raw_file"]).read_text(encoding="utf-8") == "second\n"
