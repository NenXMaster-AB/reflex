"""Result caching keyed by (state, question, model).

Jev is deterministic enough in practice that repeating an identical question
against identical state is pure waste. Caching is on by default and keyed by
content, so it stays correct when state changes.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Cache(Protocol):
    """Minimal cache interface; supply your own to back this with Redis etc."""

    def get(self, key: str) -> Any | None: ...

    def set(self, key: str, value: Any) -> None: ...


class LRUCache:
    """Bounded in-process cache. Not shared across processes."""

    def __init__(self, maxsize: int = 4096) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self.maxsize = maxsize
        self._data: OrderedDict[str, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        if key in self._data:
            self._data.move_to_end(key)
            self.hits += 1
            return self._data[key]
        self.misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()
        self.hits = self.misses = 0

    def __len__(self) -> int:
        return len(self._data)


class NullCache:
    """Disables caching."""

    def get(self, key: str) -> Any | None:
        return None

    def set(self, key: str, value: Any) -> None:
        return None


def cache_key(state_key: str, question_key: str, model: str) -> str:
    return f"{model}:{state_key}:{question_key}"
