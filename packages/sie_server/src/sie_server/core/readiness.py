"""Shared readiness state for worker.

This module provides a single source of truth for worker readiness, used by:
- /readyz endpoint (K8s readiness probe)
- /ws/status WebSocket (gateway health check via `ready` field)

The gateway only routes requests to workers that report ready=True.
This prevents routing to workers that are still starting up.

The worker's queue path is driven by the worker-sidecar over UDS IPC.
Startup readiness and normal unloading flow through
``mark_ready``/``mark_not_ready``. Sidecar deployments can additionally
gate readiness on the sidecar's heartbeat being fresh via
``register_liveness_probe``; direct HTTP/local deployments can leave that
probe unset.

Usage:
    from sie_server.core.readiness import is_ready, mark_ready, mark_not_ready

    mark_ready()   # After startup completes
    yield
    mark_not_ready()  # During shutdown
"""

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

_ready = False
_liveness_probe: Callable[[], bool] | None = None


def is_ready() -> bool:
    """Check if worker is ready to accept traffic.

    Returns:
        True if the worker has completed startup, has not shut down, and any
        registered external liveness probe (e.g., worker-sidecar heartbeat)
        still reports healthy.
    """
    if not _ready:
        return False
    return not (_liveness_probe is not None and not _liveness_probe())


def mark_ready() -> None:
    """Mark worker as ready for traffic.

    Called by lifespan after all startup tasks complete (before yield).
    """
    global _ready
    logger.info("Worker is ready for traffic")
    _ready = True


def mark_not_ready() -> None:
    """Mark worker as not ready for traffic.

    Called by lifespan during shutdown (after yield).
    """
    global _ready
    logger.info("Worker shutting down, no longer accepting traffic")
    _ready = False


def register_liveness_probe(probe: Callable[[], bool] | None) -> None:
    """Register (or clear) an external liveness check.

    When set, ``is_ready()`` returns ``False`` whenever ``probe()`` returns
    ``False``, even if the worker is otherwise marked ready. Passing ``None``
    clears the probe — used during shutdown to avoid dangling references.
    """
    global _liveness_probe
    _liveness_probe = probe
