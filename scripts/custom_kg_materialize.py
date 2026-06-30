"""Retired custom_kg file-storage materialization entrypoint.

Native production materializes zvec workspaces from state artifacts. The old
file-backend writer is intentionally fail-closed so tests and operators cannot
accidentally recreate retired ``rag_storage``/GraphML/VDB storage as a production
path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

CUSTOM_KG_FILE_STORAGE_MATERIALIZER_RETIRED_MESSAGE = (
    "custom KG file-storage materializer is retired after native zvec production cutover; "
    "use export-manifest plus native_zvec_materialize.py preflight/build for native staging"
)


def materialize_file_storage_from_manifest(
    manifest: dict[str, Any],
    resolved_vectors: dict[str, dict[str, dict[str, Any]]],
    storage_dir: Path,
) -> dict[str, Any]:
    """Fail closed before writing retired file-backend storage."""

    raise RuntimeError(CUSTOM_KG_FILE_STORAGE_MATERIALIZER_RETIRED_MESSAGE)
