import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def test_batch_wikigraph_refresh_is_importable_and_replaces_old_script_name() -> None:
    module = importlib.import_module("batch_wikigraph_refresh")

    script_names = {path.name for path in SCRIPTS.glob("batch_*refresh.py")}
    old_token = "light" + "rag"

    assert module.__name__ == "batch_wikigraph_refresh"
    assert "batch_wikigraph_refresh.py" in script_names
    assert all(old_token not in name for name in script_names)


def test_batch_wikigraph_refresh_imports_wikigraph_pending_owner_directly() -> None:
    old_backend = "light" + "rag"
    text = (SCRIPTS / "batch_wikigraph_refresh.py").read_text(encoding="utf-8")

    assert "from wiki_wikigraph_refresh_pending import" in text
    assert f"from wiki_{old_backend}_lib import" not in text
    assert f"mark_{old_backend}_refresh_pending" not in text
    assert f"pending_{old_backend}_refresh_status" not in text
