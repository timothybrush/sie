"""Tests for ModelWorker passthrough mode.

In passthrough mode the ModelWorker bypasses its internal BatchFormer /
per-LoRA queues / adaptive controller / FCFS process loop, and treats
every ``submit*`` call as a fully-formed GPU forward pass — exactly the
frame that arrived over IPC from the worker-sidecar.

These tests assert the contract:

1. No background ``_process_loop`` task is spawned.
2. No per-LoRA ``BatchFormer`` is constructed.
3. No adaptive controller is built (even if ``adaptive_batching.enabled=True``).
4. Submit dispatches synchronously: by the time ``submit*`` returns its
   future, the future is already done.
5. Concurrent submits with different LoRAs serialise around
   ``set_active_lora`` so the GPU never sees the wrong LoRA active.
6. Backward compat: with passthrough off, every existing test behaviour
   stays unchanged (covered by the existing 42 worker tests).
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest
from sie_server.core.inference_output import EncodeOutput
from sie_server.core.prepared import TextPreparedItem, make_text_item
from sie_server.core.worker import ModelWorker, WorkerConfig
from sie_server.core.worker.types import AdaptiveBatchingParams
from sie_server.types.inputs import Item


@pytest.fixture
def mock_adapter() -> MagicMock:
    """Adapter that echoes a [batch_size, 3] dense tensor."""
    mock = MagicMock()
    mock.encode.side_effect = lambda items, *args, **kwargs: EncodeOutput(
        dense=np.array([[0.1, 0.2, 0.3]] * len(items), dtype=np.float32),
        batch_size=len(items),
    )
    return mock


@pytest.fixture
def prepared_item() -> TextPreparedItem:
    return make_text_item([1, 2, 3, 4, 5], 0)


def test_passthrough_skips_internal_state(mock_adapter: MagicMock) -> None:
    """Passthrough mode skips per-LoRA batchers + adaptive controller.

    The internal worker state still exists (so existing call-sites that
    iterate ``self._batchers`` don't crash), but nothing is allocated
    upfront — the dict is empty and the controller is None.
    """
    config = WorkerConfig(
        passthrough_mode=True,
        adaptive_batching=AdaptiveBatchingParams(enabled=True),  # MUST be ignored.
    )
    worker = ModelWorker(mock_adapter, config)

    assert worker._passthrough_mode is True
    assert worker._batchers == {}, "no BatchFormer should be allocated upfront"
    assert worker._adaptive_controller is None, "adaptive controller forced off in passthrough"
    assert worker._latency_tracker is None
    assert worker._efficiency_tracker is None


@pytest.mark.asyncio
async def test_passthrough_start_does_not_spawn_process_loop(mock_adapter: MagicMock) -> None:
    """``start()`` in passthrough returns without creating a background task."""
    worker = ModelWorker(mock_adapter, WorkerConfig(passthrough_mode=True))
    await worker.start()
    try:
        assert worker.is_running is True
        assert worker._process_task is None, "no _process_loop should be spawned"
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_passthrough_submit_dispatches_immediately(
    mock_adapter: MagicMock, prepared_item: TextPreparedItem
) -> None:
    """The future is already done by the time submit() returns.

    No batcher means no coalesce wait, no FCFS poll loop. The single GPU
    forward pass runs synchronously inside the inference executor and
    completes before submit() returns its (already-resolved) future.
    """
    worker = ModelWorker(mock_adapter, WorkerConfig(passthrough_mode=True))
    await worker.start()
    try:
        future = await worker.submit(
            [prepared_item],
            [Item(text="hello")],
            ["dense"],
        )
        # Already resolved — no extra await needed beyond the executor
        # round-trip already consumed by submit_passthrough().
        assert future.done(), "passthrough submit must resolve the future synchronously"

        worker_result = future.result()
        assert worker_result.output.batch_size == 1
        np.testing.assert_array_equal(
            worker_result.output.dense[0],
            np.array([0.1, 0.2, 0.3], dtype=np.float32),
        )
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_passthrough_each_submit_is_one_forward_pass(
    mock_adapter: MagicMock, prepared_item: TextPreparedItem
) -> None:
    """N concurrent submits → N adapter.encode() calls (NOT one big batch).

    Default mode coalesces concurrent submits into a single GPU batch.
    Passthrough deliberately does NOT — the worker-sidecar already owns
    batch formation, and re-batching here is the dual-controller path
    we're carving out. The adapter sees exactly the frames the sidecar
    sent.
    """
    worker = ModelWorker(mock_adapter, WorkerConfig(passthrough_mode=True))
    await worker.start()
    try:
        # Three independent submits, each with one prepared item.
        results = await asyncio.gather(
            worker.submit([prepared_item], [Item(text="a")], ["dense"]),
            worker.submit([prepared_item], [Item(text="b")], ["dense"]),
            worker.submit([prepared_item], [Item(text="c")], ["dense"]),
        )
        assert len(results) == 3

        # In passthrough every submit dispatches its own GPU call. Default-
        # mode coalescing would collapse this into 1 call; we want 3.
        assert mock_adapter.encode.call_count == 3, (
            f"passthrough: expected 3 forward passes (one per submit), got {mock_adapter.encode.call_count}"
        )
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_passthrough_serialises_lora_switch(mock_adapter: MagicMock, prepared_item: TextPreparedItem) -> None:
    """Concurrent submits across LoRAs serialise around set_active_lora.

    The hazard: coro A sets LoRA "x", schedules its forward pass on the
    inference executor (returns immediately), then coro B sets LoRA "y"
    on the asyncio thread *before* A's forward pass actually runs on the
    executor — so A executes with LoRA "y" active. The passthrough lock
    prevents this by holding the (set_lora → process_batch) pair atomic.

    We assert the invariant by tracking, for each call, the LoRA that
    was active when ``encode`` actually fired.
    """
    set_lora_calls: list[str | None] = []
    encode_lora_at_call: list[str | None] = []

    def on_set_lora(lora: str | None) -> None:
        mock_adapter._current_lora = lora
        set_lora_calls.append(lora)

    def on_encode(items, *args, **kwargs):
        # Record which LoRA was active at the moment encode was invoked
        # by the inference thread. Under the passthrough lock this MUST
        # equal the LoRA the corresponding submit() requested.
        encode_lora_at_call.append(getattr(mock_adapter, "_current_lora", None))
        return EncodeOutput(
            dense=np.array([[0.1, 0.2, 0.3]] * len(items), dtype=np.float32),
            batch_size=len(items),
        )

    mock_adapter.set_active_lora.side_effect = on_set_lora
    mock_adapter.encode.side_effect = on_encode

    worker = ModelWorker(mock_adapter, WorkerConfig(passthrough_mode=True))
    await worker.start()
    try:
        await asyncio.gather(
            worker.submit([prepared_item], [Item(text="a")], ["dense"], options={"lora": "lora-x"}),
            worker.submit([prepared_item], [Item(text="b")], ["dense"], options={"lora": "lora-y"}),
            worker.submit([prepared_item], [Item(text="c")], ["dense"], options={"lora": "lora-x"}),
        )

        # Each submit must have set LoRA right before its own encode.
        # The pairing isn't guaranteed in order across concurrent submits,
        # but the count must match and every encode must see SOME LoRA
        # that was set by the immediately-preceding set_active_lora call.
        assert mock_adapter.encode.call_count == 3
        assert mock_adapter.set_active_lora.call_count == 3
        # The atomic pairing means each encode_lora MUST equal one of the
        # LoRAs we asked for, with the right multiset.
        assert sorted(s or "" for s in encode_lora_at_call) == ["lora-x", "lora-x", "lora-y"]
    finally:
        await worker.stop()


def test_passthrough_default_off() -> None:
    """Default config has passthrough OFF — backwards compat with default deploys."""
    config = WorkerConfig()
    assert config.passthrough_mode is False


def test_passthrough_env_var_override(monkeypatch: pytest.MonkeyPatch, mock_adapter: MagicMock) -> None:
    """``SIE_WORKER_PASSTHROUGH=1`` forces passthrough even on default config."""
    monkeypatch.setenv("SIE_WORKER_PASSTHROUGH", "1")
    worker = ModelWorker(mock_adapter, WorkerConfig())  # passthrough_mode=False
    assert worker._passthrough_mode is True


def test_passthrough_env_var_off_by_default(mock_adapter: MagicMock) -> None:
    """Without the env var, default config keeps passthrough OFF."""
    worker = ModelWorker(mock_adapter, WorkerConfig())
    assert worker._passthrough_mode is False
    # And the existing worker state is up.
    assert None in worker._batchers
