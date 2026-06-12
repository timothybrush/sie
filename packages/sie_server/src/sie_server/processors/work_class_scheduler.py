"""Weighted-fair-queue scheduler for mixed-class worker pools.

Lifts the hard "a pool serves generation **or** embedding, never both"
invariant (see :mod:`sie_server.core.pool_isolation`) by fairly sharing a
worker's slots between work classes when the operator opts in via
``pool.fairness``. Each class gets a reserved ``min_slots`` floor (never
consumable by another class, so neither starves) and competes for the
remaining shared slots by weight via a virtual-time deficit counter.

This module is the algorithmic core: pure :mod:`asyncio`, no NATS. The
pull-loop integration (acquire-before-dispatch, release-at-chunk-boundary),
the ``SaturationGate`` snapshot wiring, and load tests are separate layers;
this component is unit-tested in isolation first.

Slots are released at **completion** (chunk boundary for generation, batch
completion for embedding) — never mid-decode — so KV-cache locality is
preserved. Fairness governs *class share*; the worker's KV-budget admission
gate still governs absolute generation concurrency *inside* a granted slot.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

# Work-class names. ``embedding`` covers encode/score/extract (all the
# non-generation, GPU-batch shapes); ``generation`` is the streaming decode
# path.
GENERATION_CLASS = "generation"
EMBEDDING_CLASS = "embedding"


def classify_work_class(op: str) -> str:
    """Map a work-item op to its scheduler class.

    ``generate`` → :data:`GENERATION_CLASS`; ``encode`` / ``score`` /
    ``extract`` (and anything else) → :data:`EMBEDDING_CLASS`.
    """
    return GENERATION_CLASS if op == "generate" else EMBEDDING_CLASS


@dataclass(frozen=True)
class WorkClassConfig:
    """Per-class fairness parameters.

    Attributes:
        weight: Relative share of the *shared* (non-floor) slot pool. Must
            be > 0. A class with weight 3 receives ~3x the shared slots of a
            weight-1 class over a long contended run.
        min_slots: Slots reserved for this class, never consumable by any
            other class. Guarantees a starvation floor. Must be >= 0.
    """

    weight: float
    min_slots: int


@dataclass(frozen=True)
class FairnessConfig:
    """Validated ``pool.fairness`` configuration."""

    total_slots: int
    classes: Mapping[str, WorkClassConfig]

    def validate(self) -> None:
        """Raise :class:`ValueError` on a misconfiguration that would
        deadlock or mis-share the pool.

        - ``total_slots`` >= 1
        - every weight > 0, every ``min_slots`` >= 0
        - ``sum(min_slots) <= total_slots`` (an over-subscribed floor would
          deadlock the weighted pool — see plan §7).
        """
        if self.total_slots < 1:
            raise ValueError(f"total_slots must be >= 1, got {self.total_slots}")
        if not self.classes:
            raise ValueError("fairness config must define at least one class")
        floor_sum = 0
        for name, cfg in self.classes.items():
            if cfg.weight <= 0:
                raise ValueError(f"class {name!r} weight must be > 0, got {cfg.weight}")
            if cfg.min_slots < 0:
                raise ValueError(f"class {name!r} min_slots must be >= 0, got {cfg.min_slots}")
            floor_sum += cfg.min_slots
        if floor_sum > self.total_slots:
            raise ValueError(
                f"sum(min_slots)={floor_sum} exceeds total_slots={self.total_slots}; "
                "an over-subscribed floor would deadlock the weighted pool"
            )


@dataclass
class _ClassState:
    cfg: WorkClassConfig
    leased: int = 0
    # Virtual time for weighted fair queueing: each *shared* grant adds
    # 1/weight, so the class with the smallest virtual time (most "owed"
    # service relative to its weight) wins the next contended slot.
    virtual_time: float = 0.0
    waiters: deque[asyncio.Future[None]] = field(default_factory=deque)


class WorkClassScheduler:
    """Fair-queueing slot scheduler across work classes.

    Usage::

        sched = WorkClassScheduler(config)
        async with sched.lease(classify_work_class(op)):
            ...  # dispatch the work item; slot released on exit

    or the explicit form ``await sched.acquire(cls)`` / ``sched.release(cls)``.
    """

    def __init__(self, config: FairnessConfig) -> None:
        config.validate()
        self._config = config
        self._classes: dict[str, _ClassState] = {name: _ClassState(cfg=cfg) for name, cfg in config.classes.items()}
        # Set when nothing is leased and nothing is waiting — used by drain().
        self._idle = asyncio.Event()
        self._idle.set()

    # -- public API ---------------------------------------------------------

    async def acquire(self, work_class: str) -> None:
        """Wait until a slot is granted to ``work_class``."""
        state = self._state(work_class)
        # Fast path: grantable right now and no one ahead in this class.
        if not state.waiters and self._grant_kind(work_class) is not None:
            self._grant(work_class)
            self._refresh_idle()
            return
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        state.waiters.append(fut)
        self._idle.clear()
        self._pump()
        await fut

    def release(self, work_class: str) -> None:
        """Release a slot held by ``work_class`` (call once per acquire)."""
        state = self._state(work_class)
        if state.leased <= 0:
            raise RuntimeError(f"release without a held slot for class {work_class!r}")
        state.leased -= 1
        self._pump()
        self._refresh_idle()

    def lease(self, work_class: str) -> _Lease:
        """Async context manager: acquire on enter, release on exit."""
        return _Lease(self, work_class)

    async def drain(self) -> None:
        """Wait until all in-flight slots are released and no waiters remain.

        The caller is expected to have stopped admitting new work first
        (the pull loop stops pulling before draining), so this converges.
        """
        self._refresh_idle()
        await self._idle.wait()

    def saturation_snapshot(self) -> dict[str, object]:
        """Per-class + aggregate occupancy for the ``SaturationGate`` / metrics."""
        per_class = {
            name: {
                "leased": st.leased,
                "queued": len(st.waiters),
                "min_slots": st.cfg.min_slots,
                "weight": st.cfg.weight,
            }
            for name, st in self._classes.items()
        }
        total_leased = sum(st.leased for st in self._classes.values())
        total_queued = sum(len(st.waiters) for st in self._classes.values())
        return {
            "total_slots": self._config.total_slots,
            "total_leased": total_leased,
            "total_queued": total_queued,
            "classes": per_class,
        }

    # -- internals ----------------------------------------------------------

    def _state(self, work_class: str) -> _ClassState:
        try:
            return self._classes[work_class]
        except KeyError:
            raise ValueError(f"unknown work class {work_class!r}; configured: {sorted(self._classes)}") from None

    def _reserved_for_others(self, work_class: str) -> int:
        """Floor slots currently reserved for *other* classes (unfilled
        portion of their ``min_slots``) — these cannot be lent out.
        """
        return sum(max(0, st.cfg.min_slots - st.leased) for name, st in self._classes.items() if name != work_class)

    def _total_leased(self) -> int:
        return sum(st.leased for st in self._classes.values())

    def _grant_kind(self, work_class: str) -> str | None:
        """Return ``"floor"`` / ``"shared"`` if a slot is available to
        ``work_class`` right now, else ``None``.

        A class may take a floor slot whenever it is below its own
        ``min_slots`` (always available, reserved for it). Otherwise it may
        take a shared slot only if free capacity remains after honouring
        every *other* class's floor reservation.
        """
        state = self._classes[work_class]
        if state.leased < state.cfg.min_slots:
            return "floor"
        free = self._config.total_slots - self._total_leased()
        if free - self._reserved_for_others(work_class) > 0:
            return "shared"
        return None

    def _grant(self, work_class: str) -> None:
        """Account a grant (caller has verified availability)."""
        state = self._classes[work_class]
        kind = self._grant_kind(work_class)
        state.leased += 1
        # Only contended (shared) grants advance virtual time; floor grants
        # are reserved capacity and must not skew the weighted share.
        if kind == "shared":
            state.virtual_time += 1.0 / state.cfg.weight

    def _pump(self) -> None:
        """Grant as many waiting requests as current capacity allows,
        choosing contended slots by smallest weight-normalized virtual time.
        """
        while True:
            # Floor grants first (cheap, reserved, never skew fairness).
            floor_cls = next(
                (name for name, st in self._classes.items() if st.waiters and st.leased < st.cfg.min_slots),
                None,
            )
            if floor_cls is not None:
                self._resolve_one(floor_cls)
                continue
            # Then a contended slot to the most-owed waiting class.
            eligible = [
                (st.virtual_time, name)
                for name, st in self._classes.items()
                if st.waiters and self._grant_kind(name) == "shared"
            ]
            if not eligible:
                break
            eligible.sort()
            self._resolve_one(eligible[0][1])
        self._refresh_idle()

    def _resolve_one(self, work_class: str) -> None:
        state = self._classes[work_class]
        fut = state.waiters.popleft()
        self._grant(work_class)
        if not fut.done():
            fut.set_result(None)

    def _refresh_idle(self) -> None:
        if self._total_leased() == 0 and all(not st.waiters for st in self._classes.values()):
            self._idle.set()
        else:
            self._idle.clear()


class _Lease:
    """Async context manager returned by :meth:`WorkClassScheduler.lease`."""

    def __init__(self, scheduler: WorkClassScheduler, work_class: str) -> None:
        self._scheduler = scheduler
        self._work_class = work_class

    async def __aenter__(self) -> None:
        await self._scheduler.acquire(self._work_class)

    async def __aexit__(self, *exc: object) -> None:
        self._scheduler.release(self._work_class)
