"""Tests for the SSE payload iterators (``sie_sdk.client._sse``)."""

from __future__ import annotations

import pytest
from sie_sdk.client._sse import aiter_sse_payloads, iter_sse_payloads


async def _alines(lines: list[str]):
    for line in lines:
        yield line


def _lines() -> list[str]:
    return [
        'data: {"a":1}',
        "",  # event separator (blank line)
        ": keep-alive comment",  # comment → skipped
        'data: {"b":2}',
        "data: [DONE]",
        'data: {"never":true}',  # after [DONE] → must not be yielded
    ]


def test_iter_sse_payloads_extracts_data_and_stops_on_done() -> None:
    out = list(iter_sse_payloads(_lines()))
    assert out == ['{"a":1}', '{"b":2}']


def test_iter_sse_payloads_strips_single_leading_space() -> None:
    # "data:" prefix removed, then exactly one leading space stripped.
    assert list(iter_sse_payloads(["data:  x", "data:y"])) == [" x", "y"]


def test_iter_sse_payloads_skips_blank_and_comment_lines() -> None:
    assert list(iter_sse_payloads(["", ": ka", "data: only"])) == ["only"]


@pytest.mark.asyncio
async def test_aiter_sse_payloads_extracts_data_and_stops_on_done() -> None:
    out = [p async for p in aiter_sse_payloads(_alines(_lines()))]
    assert out == ['{"a":1}', '{"b":2}']
