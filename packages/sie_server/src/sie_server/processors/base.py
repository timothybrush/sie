"""``MessageProcessor`` Protocol — the seam introduced by the walking skeleton.

The existing batch path (encode/score/extract) still lives in
``NatsPullLoop._process_messages`` and is not yet flipped through this
protocol — the seam is sufficient for the walking skeleton. Generation
work items are dispatched here.

A processor owns the full lifecycle of a single message: deserialization,
inference, reply publish, and ACK/NAK. Returning from ``process()``
indicates the message has been handled (either ACKed or NAKed); raising
indicates a bug — the loop will log and continue.
"""

from __future__ import annotations

from typing import Any, Protocol


class MessageProcessor(Protocol):
    """Strategy interface for handling a single NATS work message."""

    async def process(self, msg: Any, model_id: str) -> None: ...
