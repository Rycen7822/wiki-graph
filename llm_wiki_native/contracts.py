"""Public constants for the llm-wiki native retrieval kernel."""

from __future__ import annotations

NATIVE_SCHEMA_VERSION = 1
WORKSPACE_SCHEMA_VERSION = 1
DEFAULT_NATIVE_PORT = 9621
DEFAULT_QUERY_MODE = "mix"
SUPPORTED_RETRIEVAL_GOALS = {"focused", "coverage"}
DEFAULT_RETRIEVAL_GOAL = "focused"
DEFAULT_TOP_K = 20
MAX_TOP_K = 100
DEFAULT_NEIGHBOR_LIMIT = 5
MAX_NEIGHBOR_LIMIT = 50
DEFAULT_MAX_CHARS_PER_BLOCK = 1200
MAX_CHARS_PER_BLOCK = 20_000
MAX_QUERY_VECTOR_DIM = 8_192
DEFAULT_RESPONSE_PROFILE = "standard"
RESPONSE_PROFILES = {"compact", "standard", "debug"}
SUPPORTED_QUERY_MODES = {"mix", "naive", "bypass"}
RECORD_TYPES = {"chunk", "entity", "relationship", "section"}
DEFAULT_QUERY_RECORD_TYPES = ("entity", "relationship", "chunk", "section")
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
