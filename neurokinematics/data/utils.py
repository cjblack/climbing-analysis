class IndexedList:
    def __init__(self, items=None, id_attr='id'):
        self._items = items or []
        self._id_attr = id_attr

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._items[key]
        elif isinstance(key, str):
            for item in self._items:
                if getattr(item, self._id_attr) == key:
                    return item
            raise KeyError(f"No item with id '{key}'")

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def append(self, item):
        self._items.append(item)