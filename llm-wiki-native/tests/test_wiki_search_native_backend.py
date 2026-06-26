import json
import sys
from pathlib import Path
from types import SimpleNamespace

from llm_wiki_native.storage.sqlite_workspace import NativeRecord, SQLiteWorkspace

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import wiki_search  # noqa: E402


def test_wiki_search_native_backend_is_explicit_and_default_off(tmp_path) -> None:
    db_path = tmp_path / "native.sqlite"
    db = SQLiteWorkspace(db_path)
    db.create_workspace("native-test", "manifest-hash")
    db.put_record(
        NativeRecord(
            workspace_id="native-test",
            record_type="entity",
            record_id="doc:a",
            vector_text="Alpha",
            content_hash="doc:a:content",
            metadata_hash="doc:a:metadata",
            vector_hash="doc:a:vector",
            source_path="alpha.md",
            source_id="doc:a",
            payload={"title": "Alpha"},
        )
    )
    db.put_vector("native-test", "entity", "doc:a", "doc:a:vector", [1.0, 0.0])
    args = SimpleNamespace(
        backend="native",
        native_db=db_path,
        native_workspace="native-test",
        query_vector=json.dumps([1.0, 0.0]),
        mode="mix",
        top_k=1,
        chunk_top_k=10,
        section_kind=None,
        data_only=True,
        expand_section_neighbors=False,
        save_evidence_pack=False,
        state_dir=tmp_path,
        workdir=tmp_path,
        server="http://127.0.0.1:9621",
        neighbor_k=5,
    )

    result = wiki_search.run_query(args, "alpha")

    assert result["backend"] == "native"
    assert result["response"]["context_blocks"][0]["source_path"] == "alpha.md"
