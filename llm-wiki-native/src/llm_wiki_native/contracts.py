"""Public constants for the llm-wiki native shadow kernel."""

from __future__ import annotations

NATIVE_SCHEMA_VERSION = 1
WORKSPACE_SCHEMA_VERSION = 1
DEFAULT_NATIVE_PORT = 9622
SUPPORTED_QUERY_MODES = {"local", "global", "hybrid", "naive", "mix", "bypass"}
IMPLEMENTED_QUERY_MODES = {"mix", "naive", "bypass"}
RECORD_TYPES = {"chunk", "entity", "relationship", "section"}
RECORD_TYPE_CODES = {"chunk": 1, "entity": 2, "relationship": 3, "section": 4}
SOURCE_KIND_CODES = {"compiled": 1, "raw": 2, "generated": 3, "debug": 4}
SECTION_KIND_CODES = {
    "none": 0,
    "summary": 1,
    "abstract": 2,
    "motivation": 3,
    "methodology": 4,
    "results": 5,
    "future": 6,
    "limitations": 7,
    "questions": 8,
    "other": 99,
}
