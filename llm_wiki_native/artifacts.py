"""Read-only loaders for native llm-wiki state artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from llm_wiki_native.section_embedding_store import load_rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(item)
    return rows


def load_custom_kg_manifest(state_dir: Path) -> dict[str, Any]:
    return _read_json(Path(state_dir) / "custom_kg_manifest.json")


def load_section_similarity_edges(state_dir: Path) -> list[dict[str, Any]]:
    return _read_jsonl(Path(state_dir) / "section_similarity_edges.jsonl")


def load_raw_sections(state_dir: Path) -> list[dict[str, Any]]:
    return _read_jsonl(Path(state_dir) / "raw_sections.jsonl")


def load_section_embeddings(state_dir: Path) -> list[dict[str, Any]]:
    # Prefer the sqlite sidecar (dual-read window); fall back to the jsonl.
    sidecar_rows = load_rows(state_dir)
    if sidecar_rows is not None:
        rows = sidecar_rows
    else:
        rows = _read_jsonl(Path(state_dir) / "section_embeddings.jsonl")
    # Box floats once into float32 ndarrays; consumers (zvec insert, sqlite
    # vector blobs) convert at their own boundaries instead of re-listing per hop.
    for row in rows:
        embedding = row.get("embedding")
        if isinstance(embedding, list):
            row["embedding"] = np.asarray(embedding, dtype=np.float32)
        elif isinstance(embedding, np.ndarray) and embedding.dtype != np.float32:
            row["embedding"] = embedding.astype(np.float32)
    return rows
