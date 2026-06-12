"""Per-process grammar compile cache for the worker.

The :class:`StreamingProcessor` holds a single :class:`GrammarLRU`
instance and consults it before invoking
:func:`sie_server.processors.grammar_compile.compile_outlines`. The
cache key is ``(tokenizer_hash, schema_hash, backend)`` — see
:mod:`sie_server.types.grammar` for the schema-hash derivation.

Thread safety
-------------
The compile path runs inside :func:`asyncio.to_thread` so cache
mutations can happen off the asyncio loop thread. A
:class:`threading.Lock` guards :meth:`get` / :meth:`put` operations
on the underlying :class:`OrderedDict`. The lock is held only for
the dict mutation itself — value materialisation (the compile) runs
outside the lock so a slow compile cannot block readers.

Eviction
--------
LRU. :meth:`get` calls ``move_to_end`` to refresh recency;
:meth:`put` evicts the head when the cap is reached. Default cap is
64 entries, which is many for the realistic schema-cardinality of a
single worker (most production traffic settles on a handful of
schemas).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

# Cache key: (tokenizer_hash, schema_hash, backend). Stable tuple so
# dictionary iteration is deterministic for unit tests.
GrammarCacheKey = tuple[str, str, str]

# Distinguishes "key absent" from "key present with a falsy/``None`` value".
# A future backend may legitimately cache a real object (or even ``None``);
# using ``dict.get(key)`` alone would treat that as a miss.
_MISSING = object()


class GrammarLRU:
    """Bounded thread-safe LRU mapping :class:`GrammarCacheKey` → ``Any``.

    The stored value is opaque to the cache — it is whatever
    :func:`compile_outlines` returns. For the SGLang grammar path the
    value is a sentinel marker (``True``) confirming the compile would
    succeed; the actual processor lives inside SGLang's grammar
    backend. The cache contract does not depend on the value shape,
    so future backends can store real processor objects without
    changing this module.
    """

    def __init__(self, maxsize: int = 64) -> None:
        if maxsize <= 0:
            msg = "GrammarLRU maxsize must be positive"
            raise ValueError(msg)
        self._maxsize = maxsize
        self._data: OrderedDict[GrammarCacheKey, Any] = OrderedDict()
        # ``threading.Lock`` because ``asyncio.to_thread`` runs the
        # compile in a worker thread; concurrent ``put`` from there +
        # ``get`` from the asyncio loop must not corrupt the OrderedDict.
        self._lock = threading.Lock()

    def get(self, key: GrammarCacheKey) -> Any | None:
        """Return the cached value or ``None`` and bump recency."""
        with self._lock:
            value = self._data.get(key, _MISSING)
            if value is _MISSING:
                return None
            self._data.move_to_end(key)
            return value

    def put(self, key: GrammarCacheKey, value: Any) -> None:
        """Insert / refresh the entry, evicting the LRU head if full."""
        with self._lock:
            if key in self._data:
                self._data[key] = value
                self._data.move_to_end(key)
                return
            self._data[key] = value
            if len(self._data) > self._maxsize:
                # ``popitem(last=False)`` removes the least-recently-used
                # entry. ``OrderedDict`` guarantees insertion-order
                # traversal, and ``move_to_end`` on every ``get`` keeps
                # the head pointing at the LRU entry.
                self._data.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
