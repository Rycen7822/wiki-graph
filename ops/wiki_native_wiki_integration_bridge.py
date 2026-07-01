"""Bridge successful wiki integration into native graph refresh pending state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ops.batch_native_refresh import mark_pending as mark_native_refresh_pending
from ops.batch_native_refresh import pending_ledger_path as pending_native_refresh_ledger_path
from ops.wiki_native_wiki_integration_pending import (
    clear_pending_wiki_integration_after_success as clear_wiki_integration_pending_after_success,
)


def clear_pending_wiki_integration_after_success(
    root: Path,
    state_dir: Path,
    integrated_paths: list[str] | None = None,
    reason: str = "integration",
) -> dict[str, Any]:
    """Clear wiki-integration pending and carry cleared work into native refresh pending."""

    result = clear_wiki_integration_pending_after_success(
        root,
        state_dir,
        integrated_paths=integrated_paths,
        reason=reason,
    )
    marked_native_pending: list[dict[str, Any]] = []
    if int(result.get("cleared_count") or 0) > 0:
        marked_native_pending.append(
            mark_native_refresh_pending(
                state_dir,
                root,
                reason=f"wiki-integration:{reason}",
            )
        )
    return {
        **result,
        "marked_native_pending": marked_native_pending,
        "marked_native_pending_count": len(marked_native_pending),
        "native_ledger_path": str(pending_native_refresh_ledger_path(state_dir)),
    }
