from scripts.make_proposals import changed_slugs_first


def test_changed_slugs_sort_first():
    changes = [{"slug": "b"}, {"slug": "b"}, {"slug": "d"}]
    order = changed_slugs_first(["a", "b", "c", "d"], changes)
    assert order[0] in {"b", "d"}
    assert set(order) == {"a", "b", "c", "d"}
    assert order.index("b") < order.index("a")
