import pytest

from llm_wiki_native.storage.sqlite_workspace import SQLiteWorkspace


@pytest.fixture
def db(tmp_path):
    workspace = SQLiteWorkspace(tmp_path / "native.sqlite")
    workspace.create_workspace("native-test", "manifest-hash")
    return workspace


def test_graph_index_returns_deterministic_undirected_neighbors(db) -> None:
    db.put_edge("native-test", "relationship", "doc:a", "tag:x", 0.7, {"source": "manifest"})
    db.put_edge("native-test", "section_similarity", "doc:a", "doc:b", 0.9, {"source": "section-sim"})
    db.put_edge("native-test", "relationship", "doc:a", "tag:y", 0.7, {"source": "manifest"})

    assert [(item["neighbor_id"], item["edge_type"], item["weight"]) for item in db.neighbors("native-test", "doc:a")] == [
        ("doc:b", "section_similarity", 0.9),
        ("tag:x", "relationship", 0.7),
        ("tag:y", "relationship", 0.7),
    ]
    assert [(item["neighbor_id"], item["edge_type"]) for item in db.neighbors("native-test", "tag:x")] == [
        ("doc:a", "relationship")
    ]


def test_graph_index_filters_edge_types_and_limits_results(db) -> None:
    db.put_edge("native-test", "relationship", "doc:a", "tag:x", 0.7, {})
    db.put_edge("native-test", "section_similarity", "doc:a", "doc:b", 0.9, {})

    results = db.neighbors("native-test", "doc:a", edge_types={"relationship"}, limit=1)

    assert [(item["neighbor_id"], item["edge_type"]) for item in results] == [("tag:x", "relationship")]


def test_graph_index_preserves_reverse_directed_edges(db) -> None:
    db.put_edge("native-test", "relationship", "doc:a", "tag:x", 0.7, {"source": "forward"})
    db.put_edge("native-test", "relationship", "tag:x", "doc:a", 0.8, {"source": "reverse"})

    assert db.count_edges("native-test") == 2
    results = db.neighbors("native-test", "doc:a")

    assert [
        (item["src_id"], item["tgt_id"], item["neighbor_id"], item["weight"], item["payload"]["source"])
        for item in results
    ] == [
        ("tag:x", "doc:a", "tag:x", 0.8, "reverse"),
        ("doc:a", "tag:x", "tag:x", 0.7, "forward"),
    ]


def test_graph_index_does_not_duplicate_self_loop_neighbors(db) -> None:
    db.put_edge("native-test", "self", "doc:a", "doc:a", 1.0, {"source": "loop"})

    results = db.neighbors("native-test", "doc:a")

    assert [(item["src_id"], item["tgt_id"], item["neighbor_id"], item["payload"]["source"]) for item in results] == [
        ("doc:a", "doc:a", "doc:a", "loop")
    ]


def test_sqlite_workspace_enables_pragmas_and_neighbor_indexes(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")

    with db._connect() as conn:
        assert int(conn.execute("PRAGMA foreign_keys").fetchone()[0]) == 1
        assert int(conn.execute("PRAGMA busy_timeout").fetchone()[0]) >= 5000
        assert str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"
        indexes = {str(row["name"]) for row in conn.execute("PRAGMA index_list('edge')").fetchall()}

    assert {"idx_edge_workspace_src", "idx_edge_workspace_tgt"}.issubset(indexes)
