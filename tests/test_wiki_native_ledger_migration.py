from __future__ import annotations

import ast
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"
SCRIPTS = OPS
sys.path.insert(0, str(ROOT))

from ops import batch_native_refresh  # noqa: E402
from ops import wiki_native_lib  # noqa: E402
def test_wiki_integration_ledger_normalizes_to_current_schema(tmp_path) -> None:
    old_backend = "light" + "rag"
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    ledger_path = state_dir / "pending_wiki_integration.json"
    ledger_path.write_text(
        json.dumps(
            {
                "version": 1,
                "threshold": 10,
                "pending": [],
                "dirty": False,
                f"last_marked_{old_backend}_pending": [{"raw_path": "raw/clip/old.md"}],
                f"last_marked_{old_backend}_pending_count": 1,
                "unexpected_future_or_retired_field": "drop me",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    loaded = wiki_native_lib.load_pending_wiki_integration_ledger(state_dir)
    wiki_native_lib.save_pending_wiki_integration_ledger(state_dir, loaded)
    persisted = json.loads(ledger_path.read_text(encoding="utf-8"))

    assert f"last_marked_{old_backend}_pending" not in loaded
    assert f"last_marked_{old_backend}_pending_count" not in loaded
    assert "unexpected_future_or_retired_field" not in loaded
    assert f"last_marked_{old_backend}_pending" not in persisted
    assert f"last_marked_{old_backend}_pending_count" not in persisted
    assert "unexpected_future_or_retired_field" not in persisted
