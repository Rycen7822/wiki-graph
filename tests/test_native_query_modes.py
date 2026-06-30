import pytest

from llm_wiki_native.retrieval.query_engine import NativeQueryEngine


class _Hit:
    doc_id = "chunk:chunk-a"
    score = 0.75
    fields = {"record_type": "chunk", "record_id": "chunk-a"}


class _DB:
    def get_workspace_status(self, workspace_id: str) -> str:
        return "audited"

    def get_record(self, workspace_id: str, record_type: str, record_id: str):
        return {"record_type": record_type, "record_id": record_id, "vector_text": "Alpha"}

    def neighbors(self, workspace_id: str, record_id: str, *, limit: int):
        return [{"neighbor_id": "doc:b", "limit": limit}]


class _ZvecWorkspace:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object, int, str | None]] = []

    def query_mix(self, query: str, query_vector: list[float], top_k: int, filter_expr: str | None):
        self.calls.append(("mix", query, top_k, filter_expr))
        return [_Hit()]

    def query_vector(self, query_vector: list[float], top_k: int, filter_expr: str | None):
        self.calls.append(("naive", query_vector, top_k, filter_expr))
        return [_Hit()]


def test_mix_mode_uses_zvec_workspace() -> None:
    zvec = _ZvecWorkspace()
    engine = NativeQueryEngine(_DB(), zvec_workspace=zvec)

    result = engine.query(
        "native-test",
        "alpha query",
        [1.0, 0.0],
        mode="mix",
        top_k=3,
        record_types=("chunk",),
    )

    assert zvec.calls == [("mix", "alpha query", 3, "record_type_code in (1)")]
    assert result["trace"]["mode"] == "mix"
    assert result["trace"]["retrieval_backend"] == "zvec"
    assert result["hits"][0]["record_id"] == "chunk-a"


def test_naive_mode_uses_zvec_vector_search() -> None:
    zvec = _ZvecWorkspace()
    engine = NativeQueryEngine(_DB(), zvec_workspace=zvec)

    result = engine.query(
        "native-test",
        "alpha query",
        [1.0, 0.0],
        mode="naive",
        top_k=2,
        record_types=("chunk",),
    )

    assert zvec.calls == [("naive", [1.0, 0.0], 2, "record_type_code in (1)")]
    assert result["trace"]["mode"] == "naive"
    assert result["trace"]["retrieval_backend"] == "zvec"
    assert result["hits"][0]["doc_id"] == "chunk:chunk-a"


def test_naive_mode_with_section_kind_uses_section_filter() -> None:
    zvec = _ZvecWorkspace()
    engine = NativeQueryEngine(_DB(), zvec_workspace=zvec)  # type: ignore[arg-type]

    result = engine.query(
        "native-test",
        "method query",
        [1.0, 0.0],
        mode="naive",
        top_k=2,
        record_types=("chunk",),
        section_kind="methodology",
    )

    assert zvec.calls == [("naive", [1.0, 0.0], 2, "record_type_code in (4) and section_kind_code in (4)")]
    assert result["trace"]["mode"] == "naive"
    assert result["trace"]["section_kind"] == "methodology"


def test_bypass_mode_skips_retrieval() -> None:
    zvec = _ZvecWorkspace()
    engine = NativeQueryEngine(_DB(), zvec_workspace=zvec)

    result = engine.query("native-test", "alpha query", [1.0, 0.0], mode="bypass", top_k=2)

    assert zvec.calls == []
    assert result["hits"] == []
    assert result["trace"]["mode"] == "bypass"
    assert result["trace"]["retrieval_backend"] == "bypass"
    assert result["trace"]["vector_hit_count"] == 0


@pytest.mark.parametrize("mode", ["local", "global", "hybrid"])
def test_old_graph_modes_are_not_supported_native_modes(mode: str) -> None:
    engine = NativeQueryEngine(_DB(), zvec_workspace=_ZvecWorkspace())

    with pytest.raises(ValueError, match="unsupported mode"):
        engine.query("native-test", "alpha query", [1.0, 0.0], mode=mode, top_k=2)
