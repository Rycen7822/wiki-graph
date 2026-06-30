import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ops.wiki_native_jsonl import jsonl_read  # noqa: E402
from ops.wiki_native_jsonl import jsonl_write  # noqa: E402


def test_jsonl_read_streams_rows_in_order_and_skips_blank_lines(tmp_path: Path) -> None:

    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n  {"b": 2}  \n', encoding="utf-8")

    assert jsonl_read(path) == [{"a": 1}, {"b": 2}]

    roundtrip = tmp_path / "roundtrip.jsonl"
    assert jsonl_write(roundtrip, [{"b": 2}, {"a": 1}]) == 2
    assert jsonl_read(roundtrip) == [{"b": 2}, {"a": 1}]
