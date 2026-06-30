#!/usr/bin/env python3
"""Native state directory helpers."""

from __future__ import annotations

from pathlib import Path

STATE_SUBDIRS = [
    "edge_docs",
    "method_atom_docs",
    "raw_section_docs",
    "raw_section_audits",
    "section_similarity_reports",
    "evidence_packs",
    "validation_reports",
]


def ensure_state_dirs(state_dir: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    for name in STATE_SUBDIRS:
        (state_dir / name).mkdir(parents=True, exist_ok=True)
