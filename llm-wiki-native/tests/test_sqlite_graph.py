from llm_wiki_native.storage.sqlite_workspace import SQLiteWorkspace


def test_graph_index_returns_deterministic_undirected_neighbors(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
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


def test_graph_index_filters_edge_types_and_limits_results(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    db.put_edge("native-test", "relationship", "doc:a", "tag:x", 0.7, {})
    db.put_edge("native-test", "section_similarity", "doc:a", "doc:b", 0.9, {})

    results = db.neighbors("native-test", "doc:a", edge_types={"relationship"}, limit=1)

    assert [(item["neighbor_id"], item["edge_type"]) for item in results] == [("tag:x", "relationship")]


def test_graph_index_deduplicates_reverse_undirected_edges(tmp_path) -> None:
    db = SQLiteWorkspace(tmp_path / "native.sqlite")
    db.create_workspace("native-test", "manifest-hash")
    db.put_edge("native-test", "relationship", "doc:a", "tag:x", 0.7, {"source": "forward"})
    db.put_edge("native-test", "relationship", "tag:x", "doc:a", 0.8, {"source": "reverse"})

    results = db.neighbors("native-test", "doc:a")

    assert [(item["neighbor_id"], item["weight"], item["payload"]["source"]) for item in results] == [("tag:x", 0.8, "reverse")]
