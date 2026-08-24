import threading
import time
from pathlib import Path

import pytest

from ops import build_section_similarity_graph  # noqa: E402
from ops import custom_kg_vector_fill  # noqa: E402
from ops.vector_cache import VectorCache, resolve_manifest_vectors  # noqa: E402
from support import clear_embedding_env  # noqa: E402


def _write_embed_env(workdir: Path) -> None:
    (workdir / ".env").write_text(
        "\n".join(
            [
                "EMBEDDING_BINDING=openai",
                "EMBEDDING_BINDING_HOST=https://embedding.local/v1",
                "EMBEDDING_BINDING_API_KEY=secret",
                "EMBEDDING_MODEL=BAAI/bge-m3",
                "EMBEDDING_DIM=2",
            ]
        ),
        encoding="utf-8",
    )


def _manifest(record_count: int) -> dict:
    return {
        "metadata": {"embedding_model": "embed-a", "embedding_dim": 2, "embedding_params_version": "v1"},
        "chunks": {
            f"chunk:{index}": {
                "record_type": "chunk",
                "record_id": f"chunk:{index}",
                "vector_hash": f"hash-{index}",
                "content": f"Doc {index} content",
                "embedding_model": "embed-a",
                "embedding_dim": 2,
                "embedding_params_version": "v1",
            }
            for index in range(record_count)
        },
        "entities": {},
        "relationships": {},
    }


def _vector_report(record_count: int) -> dict:
    return {"missing": {"chunks": [f"chunk:{index}" for index in range(record_count)], "entities": [], "relationships": []}}


class _ConcurrencyTracker:
    def __init__(self, delay_s: float = 0.05) -> None:
        self.delay_s = delay_s
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls: list[list[str]] = []
        self._lock = threading.Lock()

    def __call__(self, texts, **_kwargs):
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            self.calls.append(list(texts))
        try:
            time.sleep(self.delay_s)
            return [[float(len(text)), float(index)] for index, text in enumerate(texts)]
        finally:
            with self._lock:
                self.in_flight -= 1


def _resolved_count(manifest: dict, cache: VectorCache) -> int:
    report = resolve_manifest_vectors(manifest, cache)
    return report["summary"]["total"]["hits"]


def _resolved_vectors(manifest: dict, cache: VectorCache) -> dict[str, list[float]]:
    report = resolve_manifest_vectors(manifest, cache)
    return {
        key: record["vector"]
        for key, record in report["resolved"]["chunks"].items()
    }


def test_fill_missing_manifest_vectors_serial_by_default_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clear_embedding_env(monkeypatch, "EMBEDDING_BINDING_HOST", "EMBEDDING_BINDING_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_KEY")
    _write_embed_env(tmp_path)
    tracker = _ConcurrencyTracker()
    manifest = _manifest(30)
    cache = VectorCache(tmp_path / "cache.sqlite")

    report = custom_kg_vector_fill.fill_missing_manifest_vectors(
        manifest, _vector_report(30), cache, workdir=tmp_path, embed_texts_func=tracker
    )

    assert report["total_batches"] == 3
    assert tracker.max_in_flight == 1
    assert _resolved_count(manifest, cache) == 30


def test_fill_missing_manifest_vectors_parallelizes_batches_within_profile_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clear_embedding_env(monkeypatch, "EMBEDDING_BINDING_HOST", "EMBEDDING_BINDING_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_KEY")
    _write_embed_env(tmp_path)
    tracker = _ConcurrencyTracker()
    cache = VectorCache(tmp_path / "cache.sqlite")

    manifest = _manifest(64)
    report = custom_kg_vector_fill.fill_missing_manifest_vectors(
        manifest,
        _vector_report(64),
        cache,
        workdir=tmp_path,
        embed_texts_func=tracker,
        embedding_profile="operator-fast",
    )

    assert report["batch_size"] == 32
    assert report["total_batches"] == 2
    assert tracker.max_in_flight == 2
    assert _resolved_count(manifest, cache) == 64
    assert report["summary"] == {"total": 64, "embedded": 64}
    assert report["by_collection"] == {"chunks": 64}


def test_fill_missing_manifest_vectors_concurrent_result_matches_serial_cache(tmp_path: Path) -> None:
    manifest = _manifest(46)
    vector_report = _vector_report(46)

    def embedder(texts, **_kwargs):
        return [[float(len(text)), 1.0] for text in texts]

    caches: dict[str, VectorCache] = {}
    for profile in ("conservative", "operator-fast"):
        caches[profile] = VectorCache(tmp_path / f"cache-{profile}.sqlite")
        custom_kg_vector_fill.fill_missing_manifest_vectors(
            manifest, vector_report, caches[profile], workdir=tmp_path, embed_texts_func=embedder, embedding_profile=profile
        )
    assert _resolved_vectors(manifest, caches["conservative"]) == _resolved_vectors(manifest, caches["operator-fast"])


def test_fill_missing_manifest_vectors_concurrent_failure_writes_no_cache(tmp_path: Path) -> None:
    tracker_calls = []

    def flaky_embedder(texts, **_kwargs):
        tracker_calls.append(list(texts))
        if any("Doc 5 " in text for text in texts):
            raise RuntimeError("provider boom")
        return [[0.1, 0.2] for _text in texts]

    cache = VectorCache(tmp_path / "cache.sqlite")
    with pytest.raises(RuntimeError, match="provider boom"):
        custom_kg_vector_fill.fill_missing_manifest_vectors(
            _manifest(20),
            _vector_report(20),
            cache,
            workdir=tmp_path,
            embed_texts_func=flaky_embedder,
            embedding_profile="balanced-medium",
        )
    assert _resolved_count(_manifest(20), cache) == 0


def test_fill_missing_manifest_vectors_empty_content_raises_before_any_embed(tmp_path: Path) -> None:
    manifest = _manifest(12)
    manifest["chunks"]["chunk:7"]["content"] = ""
    calls = []

    def embedder(texts, **_kwargs):
        calls.append(list(texts))
        return [[0.1, 0.2] for _text in texts]

    with pytest.raises(RuntimeError, match="cannot fill missing vector for empty content record: chunk:7"):
        custom_kg_vector_fill.fill_missing_manifest_vectors(
            manifest,
            _vector_report(12),
            VectorCache(tmp_path / "cache.sqlite"),
            workdir=tmp_path,
            embed_texts_func=embedder,
            embedding_profile="operator-fast",
        )
    assert calls == []


def test_build_embedding_rows_parallelizes_with_max_async(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = _ConcurrencyTracker(delay_s=0.02)

    def fake_embed(texts, _config, max_attempts=3):
        return tracker(texts)

    monkeypatch.setattr(build_section_similarity_graph, "openai_compatible_embed", fake_embed)
    sections = [
        {
            "section_id": f"sec:{index:03d}",
            "source_id": "src:a",
            "source_path": "raw/a.md",
            "paper_title": "Paper A",
            "section_kind": "methodology",
            "section_title": "Methodology",
            "content": f"content {index} " + "x" * 40,
        }
        for index in range(12)
    ]
    config = {"model": "embed-a", "embedding_dim": 2, "batch_size": 3, "max_async": 4}
    rows, stats = build_section_similarity_graph.build_embedding_rows(
        sections, config, tmp_path / "section_embeddings.jsonl", reuse_cache=False
    )

    assert stats["embedded"] == 12
    assert len(rows) == 12
    assert tracker.max_in_flight > 1
    assert tracker.max_in_flight <= 4
    assert [row["section_id"] for row in rows] == sorted(f"sec:{index:03d}" for index in range(12))
    assert len(tracker.calls) == 4


def test_build_embedding_rows_default_serial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tracker = _ConcurrencyTracker(delay_s=0.01)

    def fake_embed(texts, _config, max_attempts=3):
        return tracker(texts)

    monkeypatch.setattr(build_section_similarity_graph, "openai_compatible_embed", fake_embed)
    sections = [
        {
            "section_id": f"sec:{index:03d}",
            "source_path": "raw/a.md",
            "section_kind": "methodology",
            "content": f"content {index} " + "x" * 40,
        }
        for index in range(6)
    ]
    rows, stats = build_section_similarity_graph.build_embedding_rows(
        sections, {"model": "embed-a", "batch_size": 2}, tmp_path / "section_embeddings.jsonl", reuse_cache=False
    )

    assert stats["embedded"] == 6
    assert len(rows) == 6
    assert tracker.max_in_flight == 1
