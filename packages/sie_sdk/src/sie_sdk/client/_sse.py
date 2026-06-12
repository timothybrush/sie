"""Server-Sent Events parsing for the SIE SDK streaming surfaces.

The parsers consume an iterator of already-decoded text *lines* (newline
stripped) so they work for both transports the SDK uses: the sync client
feeds httpx's ``Response.iter_lines``; the async client feeds an adapter over
aiohttp's byte ``StreamReader``. The only job here is to pull the ``data:``
payload out of each line and honour the ``[DONE]`` terminator.

Scope mirrors the gateway (``packages/sie_gateway/src/handlers/sse.rs``):
the SIE gateway emits one single-line ``data: <json>`` event per chunk plus
the literal ``data: [DONE]`` terminator. Blank separator lines and ``:``
keep-alive comments are skipped; ``event:`` / ``id:`` / ``retry:`` fields and
multi-line ``data:`` continuations are not produced and are not handled here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator

_SSE_DONE = "[DONE]"


def _extract_data_payload(line: str) -> str | None:
    """Return the ``data:`` payload of one SSE line, or ``None`` to skip.

    Blank lines (event separators) and ``:`` keep-alive comments yield
    ``None``. Per the SSE spec a single leading space after the colon is
    stripped.
    """
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None
    value = line[len("data:") :]
    value = value.removeprefix(" ")
    return value


def iter_sse_payloads(lines: Iterable[str]) -> Iterator[str]:
    """Yield ``data:`` payloads from an iterable of SSE lines.

    Stops (without yielding) on the ``[DONE]`` sentinel or a clean EOF.
    The caller owns the underlying stream; closing it cancels the upstream
    request (the gateway forwards the disconnect so the worker stops
    generating).
    """
    for line in lines:
        payload = _extract_data_payload(line)
        if payload is None:
            continue
        if payload == _SSE_DONE:
            return
        yield payload


async def aiter_sse_payloads(lines: AsyncIterable[str]) -> AsyncIterator[str]:
    """Async counterpart to :func:`iter_sse_payloads`."""
    async for line in lines:
        payload = _extract_data_payload(line)
        if payload is None:
            continue
        if payload == _SSE_DONE:
            return
        yield payload
