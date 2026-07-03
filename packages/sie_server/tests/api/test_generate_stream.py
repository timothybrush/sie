"""Tests for the SIE-native ``/v1/generate`` streaming SSE shaper.

``_stream_generate_events`` emits the ``GenerateChunk`` wire shape that
``sie_sdk.SIEClient.stream_generate`` consumes (mirroring the gateway's
``build_generate_chunk_event``). These run on any platform with a fake adapter —
no MLX/torch — so a regression in the SSE contract is caught in normal CI, not
only the Mac nightly.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from sie_server.adapters._generation_base import GenerationChunk
from sie_server.api.generate import _stream_generate_events


class _FakeAdapter:
    """Duck-typed GenerationAdapter that yields preset chunks (or raises)."""

    def __init__(self, chunks: list[GenerationChunk], raise_after: int | None = None) -> None:
        self._chunks = chunks
        self._raise_after = raise_after

    async def generate(self, **_kwargs: Any) -> AsyncIterator[GenerationChunk]:
        for i, c in enumerate(self._chunks):
            if self._raise_after is not None and i == self._raise_after:
                raise RuntimeError("boom")
            yield c


def _parse_sse(raw: list[str]) -> tuple[list[dict[str, Any]], bool]:
    """Parse emitted SSE strings into events; returns (events, saw_DONE)."""
    events: list[dict[str, Any]] = []
    saw_done = False
    for block in raw:
        for line in block.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                saw_done = True
                continue
            events.append(json.loads(payload))
    return events, saw_done


async def _drain(adapter: Any) -> list[str]:
    return [
        s
        async for s in _stream_generate_events(
            adapter, prompt="hi", max_new_tokens=8, temperature=0.0, top_p=1.0, stop=None, seed=None, logit_bias=None
        )
    ]


async def test_stream_shapes_generatechunk() -> None:
    chunks = [
        GenerationChunk(text_delta="Hello", done=False, is_first=True),
        GenerationChunk(text_delta=" world", done=False),
        GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=4, completion_tokens=2),
    ]
    raw = await _drain(_FakeAdapter(chunks))
    events, saw_done = _parse_sse(raw)
    assert saw_done is True

    deltas = [e for e in events if not e.get("done")]
    terminals = [e for e in events if e.get("done")]
    assert [e["text_delta"] for e in deltas] == ["Hello", " world"]
    # monotonic seq + stable request_id across the stream
    assert [e["seq"] for e in deltas] == [0, 1]
    assert len({e["request_id"] for e in events}) == 1

    assert len(terminals) == 1
    term = terminals[0]
    assert term["done"] is True
    assert term["finish_reason"] == "stop"
    assert term["usage"] == {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
    assert "ttft_ms" in term  # first non-empty delta was observed


async def test_stream_error_emits_terminal_error_chunk() -> None:
    chunks = [
        GenerationChunk(text_delta="partial", done=False, is_first=True),
        GenerationChunk(text_delta="never", done=False),
    ]
    raw = await _drain(_FakeAdapter(chunks, raise_after=1))
    events, saw_done = _parse_sse(raw)
    assert saw_done is True
    terminal = next(e for e in events if e.get("done"))
    assert terminal["finish_reason"] == "error"
    assert terminal["error"]["code"] == "inference_error"
    # The raw exception text must NOT leak to the client (CodeQL info-exposure);
    # it is logged server-side only. The client gets a generic message.
    assert "boom" not in terminal["error"]["message"]
    assert terminal["error"]["message"] == "internal error during generation"


async def test_stream_empty_generation_still_terminates() -> None:
    chunks = [GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=1, completion_tokens=0)]
    raw = await _drain(_FakeAdapter(chunks))
    events, saw_done = _parse_sse(raw)
    assert saw_done is True
    terminal = next(e for e in events if e.get("done"))
    assert terminal["finish_reason"] == "stop"
    assert terminal["usage"]["completion_tokens"] == 0
    # No text delta was produced → no ttft_ms on the terminal chunk.
    assert "ttft_ms" not in terminal


async def test_done_is_final_sse_line() -> None:
    chunks = [
        GenerationChunk(text_delta="hi", done=False),
        GenerationChunk(text_delta="", done=True, finish_reason="stop"),
    ]
    raw = await _drain(_FakeAdapter(chunks))
    assert raw[-1].strip() == "data: [DONE]"
