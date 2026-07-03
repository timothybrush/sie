"""Model-residency policy types.

This module is the first seam of the planned ResidencyPolicy (issue #1569),
which will eventually own the eviction decision, the LRU order, and the
OOM-recovery cooldown that are today spread across ``core/registry.py``,
``core/memory.py`` and ``core/worker/oom_recovery.py``.

It deliberately holds only leaf types so both the registry and the worker's
OOM-recovery executor can import it without a cycle (the worker must not
import the registry).
"""

from __future__ import annotations

from enum import Enum


class EvictionResult(Enum):
    """Typed outcome of a single LRU eviction attempt.

    Replaces the bool that ``evict_lru_excluding`` used to return, which
    collapsed three distinct non-eviction outcomes into ``False`` — so the
    caller (the worker's OOM recovery) could not tell "nothing eligible to
    evict" from "the registry was too busy to acquire the lock in time". The
    eviction *decision* now reports why, which the recovery path logs and a
    later ResidencyPolicy slice can act on.
    """

    EVICTED = "evicted"
    """A sibling model was unloaded and its GPU memory freed."""

    NO_CANDIDATE = "no_candidate"
    """No eligible sibling to evict (all pinned, already unloading, not
    loaded, or the only resident model is the caller itself)."""

    LOCK_TIMEOUT = "lock_timeout"
    """The registry load-lock could not be acquired within the soft timeout;
    another residency operation (load/unload/drain) held it. Transient —
    the caller may retry."""

    UNLOAD_FAILED = "unload_failed"
    """A candidate was selected but its unload raised."""
