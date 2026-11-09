"""In-process stores: plan cache and conversation memory.

Both are per-process LRU + TTL maps. For several replicas behind a load balancer, back them with Redis
behind the same small interface.
"""

import copy
import time
from collections import OrderedDict
from typing import Any


class TTLCache:
    def __init__(self, max_entries: int, ttl_seconds: float):
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._max = max_entries
        self._ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        item = self._data.get(key)
        if item is None or item[0] < time.monotonic():
            if item is not None:
                del self._data[key]
            self.misses += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        return copy.deepcopy(item[1])

    def set(self, key: str, value: Any) -> None:
        if self._max <= 0:
            return
        self._data[key] = (time.monotonic() + self._ttl, copy.deepcopy(value))
        self._data.move_to_end(key)
        while len(self._data) > self._max:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)
