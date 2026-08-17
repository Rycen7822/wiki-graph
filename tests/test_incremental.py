from __future__ import annotations

from llm_wiki_native.incremental import delta_summary, diff_by_key, native_record_fingerprint
from support import native_record


def _fingerprinted(records):
    return {(record.record_type, record.record_id): native_record_fingerprint(record) for record in records}


def test_diff_by_key_add_update_delete() -> None:
    previous = _fingerprinted(
        [
            native_record("ws", "chunk", "keep", text="same"),
            native_record("ws", "chunk", "mutate", text="before"),
            native_record("ws", "chunk", "drop", text="gone"),
        ]
    )
    current = _fingerprinted(
        [
            native_record("ws", "chunk", "keep", text="same"),
            native_record("ws", "chunk", "mutate", text="after"),
            native_record("ws", "chunk", "fresh", text="new"),
        ]
    )

    delta = diff_by_key(previous, current)

    assert delta.added == frozenset({("chunk", "fresh")})
    assert delta.updated == frozenset({("chunk", "mutate")})
    assert delta.deleted == frozenset({("chunk", "drop")})
    assert delta.changed is True


def test_diff_by_key_noop_yields_empty_sets() -> None:
    same = _fingerprinted([native_record("ws", "entity", "e1")])

    delta = diff_by_key(same, dict(same))

    assert delta.changed is False
    assert delta.added == frozenset()
    assert delta.updated == frozenset()
    assert delta.deleted == frozenset()
    assert delta_summary(delta) == {"added": 0, "updated": 0, "deleted": 0, "changed": False}


def test_native_record_fingerprint_covers_payload_only_mutation() -> None:
    base = native_record("ws", "chunk", "c1", text="same text")
    mutated = native_record("ws", "chunk", "c1", text="same text", payload={"title": "retitled"})

    assert base.content_hash == mutated.content_hash
    assert base.vector_hash == mutated.vector_hash
    assert native_record_fingerprint(base) != native_record_fingerprint(mutated)


def test_diff_by_key_for_edge_and_span_classes() -> None:
    previous = {
        ("relationship", "a->b"): "fp1",
        ("section_similarity", "a~c"): "fp2",
        ("lexical_span", "s1"): "fp3",
    }
    current = {
        ("relationship", "a->b"): "fp1",
        ("section_similarity", "a~c"): "fp2-changed",
        ("lexical_span", "s2"): "fp4",
    }

    delta = diff_by_key(previous, current)

    assert delta.added == frozenset({("lexical_span", "s2")})
    assert delta.updated == frozenset({("section_similarity", "a~c")})
    assert delta.deleted == frozenset({("lexical_span", "s1")})
    assert delta_summary(delta) == {"added": 1, "updated": 1, "deleted": 1, "changed": True}
