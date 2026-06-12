"""Periodic worker-health publisher over NATS.

The gateway already subscribes to ``sie.health.>`` in
``packages/sie_gateway/src/discovery/nats_health.rs`` and accepts both
msgpack and JSON payloads. The canonical health transport in the
codebase was originally the WebSocket ``/ws/status`` endpoint; the
routing rollout adds NATS as a parallel, opt-in path so that pure-NATS deployments
(no HTTP fan-out from the gateway to every worker) can carry the same
information — including the new ``saturated`` flag.

**Opt-in.** The publisher only starts when ``SIE_HEALTH_NATS=1``.
The WS path remains the default; we keep the NATS path off until it
has been validated against real workloads (see plan "Risks" #4 for
the budgeting rationale).

The payload is the same ``WorkerStatusMessage`` produced by
``api.ws.build_status_message`` so there is a single source of truth for
the schema. Encoding is msgpack — matching what the gateway parses
first; JSON would also work but msgpack is what the rest of the
worker→gateway plane already speaks.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import msgpack
import nats
from nats.aio.client import Client as NATSClient

if TYPE_CHECKING:
    from sie_sdk.types import WorkerStatusMessage


logger = logging.getLogger(__name__)

# Matches the WS publish cadence in `api/ws.py`. The gateway's existing
# subscriber tolerates either rate; staying in lockstep keeps cluster
# views consistent across transports.
DEFAULT_PUBLISH_INTERVAL_S: float = 0.2

# Subject template. The gateway subscribes to `sie.health.>` per worker
# (see `discovery/nats_health.rs::handle_nats_message`).
HEALTH_SUBJECT_TEMPLATE: str = "sie.health.{worker_id}"

# Env flag that turns the publisher on. Off by default — see module docstring.
ENABLE_ENV: str = "SIE_HEALTH_NATS"


def is_enabled() -> bool:
    """True iff the operator opted in to NATS health publishing."""
    return os.environ.get(ENABLE_ENV, "") == "1"


class NatsHealthPublisher:
    """Background task that publishes ``WorkerStatusMessage`` to NATS.

    The publisher is a thin loop on top of :func:`nats.connect`; it
    intentionally does not share the JetStream connection used by
    ``nats_pull_loop`` so a publish failure cannot impact work
    consumption. The NATS Python client multiplexes over one TCP
    connection per process anyway, so the cost is one extra logical
    subscription, not a second socket.
    """

    def __init__(
        self,
        nats_url: str,
        worker_id: str,
        build_status: Callable[[], Awaitable[WorkerStatusMessage]],
        interval_s: float = DEFAULT_PUBLISH_INTERVAL_S,
    ) -> None:
        self._nats_url = nats_url
        self._worker_id = worker_id
        self._build_status = build_status
        self._interval_s = interval_s
        self._subject = HEALTH_SUBJECT_TEMPLATE.format(worker_id=worker_id)
        self._nc: NATSClient | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        """Connect and spawn the periodic publish loop.

        Graceful degradation: a connection failure logs a warning and
        leaves the publisher inert. The WS path keeps working either
        way.
        """
        try:
            # Bounded connect attempt: we want graceful degradation on
            # startup if the NATS endpoint is wrong, not an indefinite
            # block. Once connected, the client's own reconnect loop
            # handles transient drops.
            self._nc = await nats.connect(
                self._nats_url,
                max_reconnect_attempts=-1,
                reconnect_time_wait=2,
                connect_timeout=2,
                allow_reconnect=True,
            )
        except Exception:  # noqa: BLE001 — degrade gracefully when NATS down
            logger.warning(
                "NATS health publisher failed to connect (url=%s); WS path remains canonical",
                self._nats_url,
                exc_info=True,
            )
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="sie-health-nats-publisher")
        logger.info(
            "NATS health publisher started: subject=%s, interval_s=%.3f",
            self._subject,
            self._interval_s,
        )

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                msg = await self._build_status()
                payload = msgpack.packb(dict(msg), use_bin_type=True)
                assert self._nc is not None
                await self._nc.publish(self._subject, payload)
            except Exception:  # noqa: BLE001 — never crash the worker on a publish hiccup
                logger.debug("NATS health publish failed", exc_info=True)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval_s)
            except TimeoutError:
                pass

    async def stop(self) -> None:
        """Signal the loop to exit and drain the NATS connection."""
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except TimeoutError:
                self._task.cancel()
            self._task = None
        if self._nc is not None:
            try:
                await self._nc.drain()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                logger.debug("NATS health publisher drain failed", exc_info=True)
            self._nc = None
