"""Pure record types for zvec workspace materialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ZvecRecord:
    record_type: str
    record_id: str
    canonical_id: str
    source_id: str
    source_kind_code: int
    source_path_hash: str
    source_path: str
    title: str
    vector_hash: str
    content_hash: str
    metadata_hash: str
    content: str
    tokens: int
    # list[float] or float32 np.ndarray; compare=False avoids ndarray-equality
    # blowups in generated dataclass __eq__.
    embedding: Any = field(compare=False)
    section_kind: str | None = None
