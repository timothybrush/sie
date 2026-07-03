"""``MessageProcessor`` Protocol for sidecar-driven generation work.

The sidecar owns queue fetch and settlement. Python generation work items are
dispatched here through the sidecar IPC path.

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
