"""Unit tests for the mixed-pool fair-queue scheduler.

Pure asyncio — no NATS. Covers config validation, the min_slots starvation
floor, weighted-fair shares, slot release waking the most-owed waiter,
drain, and the saturation snapshot.
"""

from __future__ import annotations

import asyncio

import pytest
from sie_server.processors.work_class_scheduler import (
    EMBEDDING_CLASS,
    GENERATION_CLASS,
    FairnessConfig,
    WorkClassConfig,
    WorkClassScheduler,
    classify_work_class,
)


def _cfg(total: int, **classes: tuple[float, int]) -> FairnessConfig:
    return FairnessConfig(
        total_slots=total,
        classes={name: WorkClassConfig(weight=w, min_slots=m) for name, (w, m) in classes.items()},
    )


class TestConfigValidation:
    def test_valid_config_passes(self) -> None:
        _cfg(6, generation=(3, 1), embedding=(1, 2)).validate()

    def test_oversubscribed_floor_rejected(self) -> None:
        with pytest.raises(ValueError, match="floor"):
            _cfg(2, generation=(1, 2), embedding=(1, 2)).validate()

    def test_nonpositive_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="weight"):
            _cfg(4, generation=(0, 0), embedding=(1, 0)).validate()

    def test_zero_total_slots_rejected(self) -> None:
        with pytest.raises(ValueError, match="total_slots"):
            _cfg(0, generation=(1, 0)).validate()

    def test_scheduler_rejects_invalid_config(self) -> None:
        with pytest.raises(ValueError, match="floor"):
            WorkClassScheduler(_cfg(2, generation=(1, 2), embedding=(1, 2)))


def test_classify_work_class() -> None:
    assert classify_work_class("generate") == GENERATION_CLASS
    for op in ("encode", "score", "extract", "anything"):
        assert classify_work_class(op) == EMBEDDING_CLASS


@pytest.mark.asyncio
async def test_min_slots_floor_survives_generation_saturation() -> None:
    # 4 slots; embedding reserves 2; generation has no floor.
    s = WorkClassScheduler(_cfg(4, generation=(10, 0), embedding=(1, 2)))
    # Generation can take only the 2 *shared* slots (embedding's floor of 2
    # is reserved and not lendable).
    await s.acquire(GENERATION_CLASS)
    await s.acquire(GENERATION_CLASS)
    blocked = asyncio.create_task(s.acquire(GENERATION_CLASS))
    await asyncio.sleep(0)
    assert not blocked.done(), "generation must not consume embedding's floor"
    # Embedding still gets its reserved floor despite generation saturation.
    await asyncio.wait_for(s.acquire(EMBEDDING_CLASS), timeout=0.5)
    await asyncio.wait_for(s.acquire(EMBEDDING_CLASS), timeout=0.5)
    # Releasing a generation slot frees the blocked generation waiter.
    s.release(GENERATION_CLASS)
    await asyncio.wait_for(blocked, timeout=0.5)


@pytest.mark.asyncio
async def test_weighted_shares_converge_to_ratio() -> None:
    # Single contended slot; weight 3:1 should grant generation ~3x.
    s = WorkClassScheduler(_cfg(1, generation=(3, 0), embedding=(1, 0)))
    await s.acquire(GENERATION_CLASS)  # seed: holds the only slot
    waiters = [asyncio.create_task(s.acquire(GENERATION_CLASS)) for _ in range(100)]
    waiters += [asyncio.create_task(s.acquire(EMBEDDING_CLASS)) for _ in range(100)]
    await asyncio.sleep(0)  # let them enqueue

    counts = {GENERATION_CLASS: 0, EMBEDDING_CLASS: 0}
    held = GENERATION_CLASS
    for _ in range(80):
        s.release(held)  # frees the slot; _pump grants the most-owed waiter
        snap = s.saturation_snapshot()
        held = GENERATION_CLASS if snap["classes"][GENERATION_CLASS]["leased"] == 1 else EMBEDDING_CLASS
        counts[held] += 1

    ratio = counts[GENERATION_CLASS] / max(1, counts[EMBEDDING_CLASS])
    assert 2.0 <= ratio <= 4.5, f"expected ~3:1, got {counts} (ratio {ratio:.2f})"

    for w in waiters:
        w.cancel()


@pytest.mark.asyncio
async def test_release_without_held_slot_raises() -> None:
    s = WorkClassScheduler(_cfg(2, generation=(1, 0)))
    with pytest.raises(RuntimeError, match="without a held slot"):
        s.release(GENERATION_CLASS)


@pytest.mark.asyncio
async def test_unknown_class_raises() -> None:
    s = WorkClassScheduler(_cfg(2, generation=(1, 0)))
    with pytest.raises(ValueError, match="unknown work class"):
        await s.acquire("nope")


@pytest.mark.asyncio
async def test_lease_context_manager_releases() -> None:
    s = WorkClassScheduler(_cfg(1, generation=(1, 0)))
    async with s.lease(GENERATION_CLASS):
        assert s.saturation_snapshot()["total_leased"] == 1
    assert s.saturation_snapshot()["total_leased"] == 0


@pytest.mark.asyncio
async def test_drain_completes_when_all_released() -> None:
    s = WorkClassScheduler(_cfg(2, generation=(1, 0), embedding=(1, 0)))
    await s.acquire(GENERATION_CLASS)
    await s.acquire(EMBEDDING_CLASS)

    async def _drainer() -> bool:
        await s.drain()
        return True

    drain_task = asyncio.create_task(_drainer())
    await asyncio.sleep(0)
    assert not drain_task.done(), "drain must wait for in-flight slots"
    s.release(GENERATION_CLASS)
    s.release(EMBEDDING_CLASS)
    assert await asyncio.wait_for(drain_task, timeout=0.5)


@pytest.mark.asyncio
async def test_saturation_snapshot_shape() -> None:
    s = WorkClassScheduler(_cfg(4, generation=(3, 1), embedding=(1, 2)))
    await s.acquire(GENERATION_CLASS)
    snap = s.saturation_snapshot()
    assert snap["total_slots"] == 4
    assert snap["total_leased"] == 1
    assert snap["classes"][GENERATION_CLASS]["leased"] == 1
    assert snap["classes"][EMBEDDING_CLASS]["min_slots"] == 2
