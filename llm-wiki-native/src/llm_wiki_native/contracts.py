"""Public constants for the llm-wiki native shadow kernel."""

from __future__ import annotations

NATIVE_SCHEMA_VERSION = 1
DEFAULT_NATIVE_PORT = 9622
SUPPORTED_QUERY_MODES = {"local", "global", "hybrid", "naive", "mix", "bypass"}
RECORD_TYPES = {"chunk", "entity", "relationship", "section"}
