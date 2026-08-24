from __future__ import annotations

from pathlib import Path

import numpy as np

from llm_wiki_native import artifacts
from llm_wiki_native.section_embedding_store import load_rows, sidecar_path, upsert_rows
from ops.build_section_similarity_graph import load_cached_embeddings
from support import write_jsonl

MODEL = "BAAI/bge-m3"


def _row(section_id: str, dim: int = 8, *, text_hash: str = "h", seed: float = 1.0) -> dict:
    return {
        "section_id": section_id,
        "source_id": f"src-{section_id}",
        "source_path": f"raw/clip/2601/{section_id}.md",
        "section_kind": "methodology",
        "section_title": "Method",
        "text_hash": text_hash,
        "text_chars": 42,
        "embedding_model": MODEL,
        "embedding_dim": dim,
        "embedding": [seed + i * 0.25 for i in range(dim)],
    }


def test_sidecar_roundtrip_is_bit_exact_with_jsonl_floats(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    rows = [_row("b", seed=2.0), _row("a", seed=1.0)]
    # jsonl roundtrip gives float64 lists; sidecar must reproduce identical values
    write_jsonl(state / "section_embeddings.jsonl", rows)
    jsonl_rows = artifacts._read_jsonl(state / "section_embeddings.jsonl")

    assert upsert_rows(state, rows) == 2
    loaded = load_rows(state)

    assert loaded is not None
    assert [row["section_id"] for row in loaded] == ["a", "b"]  # sorted like the jsonl writer
    for sidecar_row, jsonl_row in zip(loaded, sorted(jsonl_rows, key=lambda r: r["section_id"])):
        assert np.array_equal(sidecar_row["embedding"], np.asarray(jsonl_row["embedding"], dtype=np.float64))
        for key in ("source_id", "source_path", "section_kind", "text_hash", "embedding_model", "embedding_dim"):
            assert sidecar_row[key] == jsonl_row[key]


def test_artifacts_loader_prefers_sidecar_and_matches_jsonl_values(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    rows = [_row("s1"), _row("s2", seed=3.0)]
    write_jsonl(state / "section_embeddings.jsonl", rows)

    from_jsonl = artifacts.load_section_embeddings(state)
    assert not sidecar_path(state).exists()

    upsert_rows(state, rows)
    from_sidecar = artifacts.load_section_embeddings(state)

    assert [r["section_id"] for r in from_jsonl] == [r["section_id"] for r in from_sidecar]
    for a, b in zip(from_jsonl, from_sidecar):
        assert a["embedding"].dtype == np.float32
        assert b["embedding"].dtype == np.float32
        assert np.array_equal(a["embedding"], b["embedding"])


def test_load_cached_embeddings_reads_sidecar(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    rows = [_row("s1"), _row("s2", seed=3.0, text_hash="h2")]
    write_jsonl(state / "section_embeddings.jsonl", rows)
    upsert_rows(state, rows)
    # corrupt the jsonl after sidecar write: cache reads must come from the sidecar
    write_jsonl(state / "section_embeddings.jsonl", [])

    cached = load_cached_embeddings(state / "section_embeddings.jsonl", MODEL)

    assert set(cached) == {"s1", "s2"}
    assert np.allclose(cached["s2"]["embedding"], rows[1]["embedding"])
    other_model = load_cached_embeddings(state / "section_embeddings.jsonl", "other-model")
    assert other_model == {}
