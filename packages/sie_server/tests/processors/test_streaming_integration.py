"""Integration-style tests for the streaming-generation path.

These tests are *integration-style* — they exercise the full worker-side
streaming pipeline (StreamingProcessor + chunk batcher + cancel event)
end-to-end with a fake adapter and a mocked NATS client. The full
worker-↔-gateway-↔-real-NATS path is gated behind ``mise run test -- -i``;
this file runs as a normal unit test so it provides protective coverage
against regressions in the streaming contract without the cost of a real
NATS server.

Two scenarios:

1. **TTFT vs E2E latency.** A fake adapter yields chunks every ~200 ms.
   We measure wall-clock time to the first published chunk versus the
   wall-clock time to the terminal chunk. ``ttft_ms < 0.4 * elapsed_ms``
   proves the worker is streaming (not blocking until terminal).

2. **Client-disconnect cancel.** Mid-stream, we call
   ``StreamingProcessor.signal_cancel`` (simulating the cancel-subject
   subscription firing on a gateway cancel publish). The processor
   publishes a ``finish_reason: "cancelled"`` terminal and ACKs.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import msgpack
import pytest
from sie_server.adapters._generation_base import GenerationAdapter, GenerationChunk
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters.base import ModelCapabilities, ModelDims
from sie_server.processors.streaming import StreamingProcessor


def _make_work_item(**overrides: Any) -> dict[str, Any]:
    wi: dict[str, Any] = {
        "work_item_id": "req-1.0",
        "request_id": "req-1",
        "item_index": 0,
        "total_items": 1,
        "operation": "generate",
        "model_id": "test/model",
        "profile_id": "default",
        "pool_name": "default",
        "machine_profile": "default",
        "router_id": "router-1",
        "reply_subject": "_INBOX.router-1.req-1",
        "timestamp": time.time(),
        "generate": {"prompt": "Hello", "max_new_tokens": 64},
    }
    wi.update(overrides)
    return wi


def _make_msg(wi: dict[str, Any]) -> AsyncMock:
    msg = AsyncMock()
    msg.data = msgpack.packb(wi, use_bin_type=True)
    return msg


class _PacedAdapter(GenerationAdapter):
    """Yields a fixed text-delta cadence with a configurable inter-chunk gap."""

    spec = AdapterSpec(inputs=("text",), outputs=("tokens",), unload_fields=())

    def __init__(self, deltas: list[str], gap_s: float, prompt_tokens: int = 5) -> None:
        self._device = None
        self._deltas = deltas
        self._gap = gap_s
        self._prompt_tokens = prompt_tokens

    def load(self, device: str) -> None:  # pragma: no cover
        self._device = device

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(inputs=["text"], outputs=["tokens"])

    @property
    def dims(self) -> ModelDims:
        return ModelDims()

    async def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
    ) -> AsyncIterator[GenerationChunk]:
        _ = (prompt, max_new_tokens, temperature, top_p, stop)
        for i, delta in enumerate(self._deltas):
            if i > 0:
                await asyncio.sleep(self._gap)
            yield GenerationChunk(text_delta=delta, is_first=(i == 0))
        yield GenerationChunk(
            text_delta="",
            done=True,
            finish_reason="stop",
            prompt_tokens=self._prompt_tokens,
            completion_tokens=len(self._deltas),
        )


def _make_registry(adapter: GenerationAdapter) -> MagicMock:
    registry = MagicMock()
    registry.is_loaded.return_value = True
    registry.get.return_value = adapter
    registry.device = "cpu"
    return registry


def _decode_chunks(nc_mock: AsyncMock) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for call in nc_mock.publish.await_args_list:
        _subject, data = call.args
        chunks.append(msgpack.unpackb(data, raw=False))
    return chunks


@pytest.mark.asyncio
async def test_streaming_ttft_well_under_e2e_latency() -> None:
    """First published chunk arrives well before the terminal chunk.

    This is the key invariant proving the worker is *streaming*: if
    ``StreamingProcessor`` were blocking on the entire iterator before
    publishing anything, TTFT would equal E2E latency. We require TTFT
    to be at most 40 % of E2E latency.
    """
    nc = AsyncMock()
    # 5 deltas with 60 ms gap each → ~240 ms total iterator runtime.
    adapter = _PacedAdapter(deltas=["he", "llo", " ", "world", "!"], gap_s=0.06)
    proc = StreamingProcessor(nc=nc, registry=_make_registry(adapter), worker_id="w1")
    msg = _make_msg(_make_work_item())

    # Record the wall-clock time of each ``nc.publish`` call so we can
    # measure TTFT (= first publish) vs E2E (= last publish).
    publish_times: list[float] = []
    started_at = time.monotonic()

    async def _capture(*args: Any, **_kwargs: Any) -> None:
        publish_times.append(time.monotonic() - started_at)

    nc.publish.side_effect = _capture

    await proc.process(msg, "test/model")

    assert len(publish_times) >= 2, f"expected ≥2 publishes, got {publish_times}"
    ttft = publish_times[0]
    e2e = publish_times[-1]
    # TTFT should be a fraction of the total — if it equals E2E we are
    # blocking, which is the walking-skeleton regression we're guarding against.
    assert ttft < 0.4 * e2e, f"TTFT {ttft:.3f}s vs E2E {e2e:.3f}s — looks blocking"

    msg.ack.assert_awaited()


class _ToolCallAdapter(GenerationAdapter):
    """Yields one text chunk + a Qwen-style ``<tool_call>`` block."""

    spec = AdapterSpec(inputs=("text",), outputs=("tokens",), unload_fields=())

    def __init__(self) -> None:
        self._device = None

    def load(self, device: str) -> None:  # pragma: no cover
        self._device = device

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(inputs=["text"], outputs=["tokens"])

    @property
    def dims(self) -> ModelDims:
        return ModelDims()

    async def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        **_: Any,
    ) -> AsyncIterator[GenerationChunk]:
        _ = (prompt, max_new_tokens, temperature, top_p, stop)
        yield GenerationChunk(text_delta="I will call a tool: ", is_first=True)
        yield GenerationChunk(text_delta='<tool_call>{"name":"get_time","arguments":{"tz":"UTC"}}</tool_call>')
        yield GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=4, completion_tokens=9)


@pytest.mark.asyncio
async def test_streaming_with_tools_emits_tool_calls_on_envelope() -> None:
    """A work item carrying ``tools`` plumbs through the parser and
    surfaces ``tool_calls`` on at least one published chunk envelope.
    The terminal envelope reports ``finish_reason: "tool_calls"``.
    """
    nc = AsyncMock()
    adapter = _ToolCallAdapter()
    proc = StreamingProcessor(nc=nc, registry=_make_registry(adapter), worker_id="w1")
    wi = _make_work_item(
        generate={
            "prompt": "What time is it?",
            "max_new_tokens": 64,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "description": "Return the current time",
                        "parameters": {
                            "type": "object",
                            "properties": {"tz": {"type": "string"}},
                        },
                    },
                }
            ],
            "tool_choice": "auto",
        }
    )
    msg = _make_msg(wi)
    await proc.process(msg, "test/model")

    decoded = _decode_chunks(nc)
    # At least one non-terminal chunk must carry ``tool_calls``.
    tc_chunks = [c for c in decoded if c.get("tool_calls")]
    assert tc_chunks, f"expected tool_calls on at least one chunk: {decoded}"
    first = tc_chunks[0]["tool_calls"][0]
    assert first["index"] == 0
    assert first["type"] == "function"
    assert first["function"]["name"] == "get_time"
    # Terminal chunk uses ``tool_calls`` finish reason after parser.
    terminal = decoded[-1]
    assert terminal["done"] is True
    assert terminal["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_streaming_cancel_mid_stream_produces_cancelled_terminal() -> None:
    """Cancellation mid-stream yields a ``finish_reason: cancelled`` terminal."""
    nc = AsyncMock()
    adapter = _PacedAdapter(deltas=["a", "b", "c", "d", "e", "f"], gap_s=0.05)
    proc = StreamingProcessor(nc=nc, registry=_make_registry(adapter), worker_id="w1")
    msg = _make_msg(_make_work_item())

    async def _cancel_after_two_publishes() -> None:
        for _ in range(200):
            if nc.publish.await_count >= 2:
                break
            await asyncio.sleep(0.01)
        proc.signal_cancel("req-1")

    cancel_task = asyncio.create_task(_cancel_after_two_publishes())
    await proc.process(msg, "test/model")
    await cancel_task

    decoded = _decode_chunks(nc)
    terminal = decoded[-1]
    assert terminal["done"] is True
    assert terminal["finish_reason"] == "cancelled"
    # The text accumulated so far must be present in the deltas (we did
    # NOT receive every scripted chunk).
    body = "".join(c.get("text_delta", "") for c in decoded)
    assert body  # something was produced
    assert body != "abcdef"  # but not the full scripted output
    msg.ack.assert_awaited()
