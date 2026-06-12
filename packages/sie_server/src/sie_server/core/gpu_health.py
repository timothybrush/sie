"""GPU health probe.

A wedged CUDA context (e.g. after a ``device-side assert``) returns the same
sticky error on every subsequent CUDA call, but a process-alive liveness check
never touches CUDA, so it stays green forever. The result: ``/healthz`` and
``/readyz`` both report 200 while every inference returns 500, the kubelet never
restarts the pod, and the gateway keeps routing to a dead worker.

This module forces a tiny CUDA sync so the readiness surfaces — ``/readyz`` and
the ``ready`` field the gateway reads from ``/ws/status`` — can detect the wedge
and stop advertising the worker as able to serve inference.

The probe's device sync blocks (``.item()`` waits for queued kernels to drain),
so async callers should use :func:`gpu_is_healthy_async`, which runs the probe on
a worker thread and keeps the event loop — inference handlers and the 200ms
``/ws/status`` push — responsive.

See issue #1025.
"""

from __future__ import annotations

import asyncio
import logging
import time

import torch

logger = logging.getLogger(__name__)

# Re-probing forces a CUDA device sync, so cache the result briefly. The
# /ws/status loop calls this every 200ms while the K8s readiness probe polls every
# ~5s. A short TTL keeps the signal fresh for the probe while bounding the GPU
# contention the status loop would otherwise add. A wedged context is sticky, so a
# couple of seconds of staleness is harmless.
_PROBE_TTL_SECONDS = 2.0

_cached_healthy: bool = True
_cached_at: float = 0.0


def _run_gpu_probe(device: str) -> None:
    """Force a CUDA sync on ``device`` via a 1-element matmul.

    On a healthy GPU this costs a few microseconds. On a wedged context the
    queued sticky error surfaces here — ``.item()`` forces the device sync, so a
    ``cudaErrorAssert`` state raises instead of being masked.
    """
    x = torch.zeros((1, 1), device=device)
    (x @ x).item()


def _probe_all_devices() -> bool:
    """Probe every visible CUDA device and return whether all can run kernels.

    A worker process can be assigned more than one GPU (``gpu.count > 1``), so a
    wedge on any device must surface — probing only the default device would miss
    a wedged ``cuda:1`` while the worker kept reporting ready (issue #1025). K8s
    scopes ``device_count()`` to the pod's allocated GPUs.

    Returns:
        True if every device runs the probe kernel; False if any device's context
        is wedged. A transient ``OutOfMemoryError`` is treated as healthy: the
        device is full, not wedged, and inference back-pressure must not escalate
        to a not-ready/restart.
    """
    for index in range(torch.cuda.device_count()):
        try:
            _run_gpu_probe(f"cuda:{index}")
        except torch.cuda.OutOfMemoryError:
            # Out of memory is not a wedge — the device can still run kernels once
            # memory frees, and the memory manager handles eviction. Reporting
            # not-ready here would needlessly depool/restart a recoverable worker.
            logger.warning(
                "GPU health probe hit OutOfMemoryError on cuda:%d; treating as "
                "healthy (memory pressure, not a wedged context).",
                index,
            )
        except Exception:
            # Catch broadly: a sticky device-side assert, a driver/context fault,
            # or any non-RuntimeError CUDA error all mean this device cannot run
            # kernels. Catching everything also guarantees the probe never escapes
            # into the caller — an uncaught error would 500 the health endpoints
            # and crash the /ws/status loop instead of cleanly reporting not-ready.
            logger.exception(
                "GPU health probe failed on cuda:%d: CUDA context appears wedged "
                "(sticky device error). Reporting not-ready so the worker is "
                "pulled from routing and restarted.",
                index,
            )
            return False
    return True


def _cache_is_fresh(now: float) -> bool:
    return (now - _cached_at) < _PROBE_TTL_SECONDS


def _store_result(healthy: bool, now: float) -> None:
    global _cached_healthy, _cached_at
    _cached_healthy = healthy
    _cached_at = now


def gpu_is_healthy(*, use_cache: bool = True) -> bool:
    """Return whether the GPU can actually run a CUDA kernel right now.

    This is the synchronous entry point; it blocks on the CUDA device sync. Async
    code should call :func:`gpu_is_healthy_async` so the probe runs off the event
    loop. This variant is kept for non-async callers and tests.

    On CPU-only workers there is no GPU context to wedge, so this returns True
    (readiness must not hinge on a device that does not exist).

    Args:
        use_cache: When True (default) reuse a recent probe result within
            ``_PROBE_TTL_SECONDS`` instead of forcing another CUDA sync. Pass
            False to always run a fresh probe.

    Returns:
        True if CUDA is unavailable or every device's probe kernel completes;
        False if any device's CUDA context is wedged.
    """
    if not torch.cuda.is_available():
        return True

    now = time.monotonic()
    if use_cache and _cache_is_fresh(now):
        return _cached_healthy

    healthy = _probe_all_devices()
    _store_result(healthy, time.monotonic())
    return healthy


async def gpu_is_healthy_async(*, use_cache: bool = True) -> bool:
    """Async GPU health check that runs the blocking probe off the event loop.

    The probe forces a CUDA device sync (``.item()``); on a busy GPU that can
    block until queued kernels drain. Running it via :func:`asyncio.to_thread`
    keeps the worker's event loop — inference handlers and the 200ms /ws/status
    push — responsive while the probe waits (issue #1025). A fresh cache hit is
    returned without touching a thread.

    Args and return value match :func:`gpu_is_healthy`.
    """
    if not torch.cuda.is_available():
        return True

    now = time.monotonic()
    if use_cache and _cache_is_fresh(now):
        return _cached_healthy

    healthy = await asyncio.to_thread(_probe_all_devices)
    _store_result(healthy, time.monotonic())
    return healthy


def reset_gpu_health_cache() -> None:
    """Clear the cached probe result so the next call re-probes.

    Intended for tests that toggle the simulated GPU state between calls.
    """
    _store_result(True, 0.0)
