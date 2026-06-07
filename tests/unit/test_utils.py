import pytest
from neurokinematics.data.utils import IndexedList


class _Item:
    def __init__(self, item_id, value):
        self.item_id = item_id
        self.value = value


@pytest.fixture
def items():
    return [_Item("a", 1), _Item("b", 2), _Item("c", 3)]


@pytest.fixture
def indexed_list(items):
    return IndexedList(items, id_attr="item_id")


def test_integer_indexing(indexed_list, items):
    assert indexed_list[0] is items[0]
    assert indexed_list[2] is items[2]


def test_string_indexing(indexed_list, items):
    assert indexed_list["a"] is items[0]
    assert indexed_list["c"] is items[2]


def test_string_key_not_found(indexed_list):
    with pytest.raises(KeyError):
        _ = indexed_list["z"]


def test_len(indexed_list):
    assert len(indexed_list) == 3


def test_iteration(indexed_list, items):
    assert list(indexed_list) == items


def test_append(indexed_list):
    new_item = _Item("d", 4)
    indexed_list.append(new_item)
    assert len(indexed_list) == 4
    assert indexed_list["d"] is new_item


def test_empty_list():
    il = IndexedList(id_attr="item_id")
    assert len(il) == 0
    assert list(il) == []


def test_default_items_none():
    il = IndexedList()
    assert len(il) == 0
