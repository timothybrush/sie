"""Regression test: WS-status ``name`` must match pull-loop ``worker_id``.

C1 from the routing-rollout review: when ``SIE_WORKER_ID`` is set (or the
pull loop falls through to ``uuid4``), the gateway used to register
the worker under a different name than the worker subscribed on,
silently breaking HRW direct dispatch.

The fix threads the pull loop's resolved ``worker_id`` into
``build_status_message`` and uses it for the WS payload's ``name``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


class _FakePullLoop:
    """Minimal stand-in exposing only the ``worker_id`` attribute the
    status builder reads.
    """

    def __init__(self, worker_id: str) -> None:
        self.worker_id = worker_id

    def update_saturation(self) -> bool:
        return False


@pytest.mark.asyncio
async def test_build_status_message_uses_pull_loop_worker_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the pull loop is wired in, its ``worker_id`` wins over env vars.

    The scenario this prevents: ``SIE_WORKER_ID="w-prod-3"`` is set and
    ``HOSTNAME`` is the K8s pod name. Pre-fix, the pull loop subscribed
    on ``sie.work.{pool}.{machine_profile}.{bundle}.*.w-prod-3`` but the
    WS payload reported
    ``name = {pod-name}`` — the gateway's HRW pick used ``{pod-name}``
    for the dispatch subject and no worker was listening on it.
    """
    monkeypatch.setenv("SIE_WORKER_ID", "w-prod-3")
    monkeypatch.setenv("HOSTNAME", "pod-abc-123")

    from sie_server.api.ws import build_status_message

    # `build_status_message` reaches into the registry for memory
    # thresholds, model status, etc.; we only care about the `name`
    # field here, so stub out the registry with the minimum surface.
    registry = MagicMock()
    registry.memory_manager.pressure_threshold_pct = 0.0
    registry._loaded = {}

    # Stub the helpers the builder calls so we don't drag in GPU,
    # prometheus, or model registry plumbing.
    monkeypatch.setattr("sie_server.api.ws.get_gpu_metrics", list)
    monkeypatch.setattr("sie_server.api.ws.get_model_status", lambda r: [])
    monkeypatch.setattr(
        "sie_server.api.ws.collect_prometheus_metrics",
        lambda: {"counters": {}, "histograms": {}},
    )
    monkeypatch.setattr("sie_server.api.ws.compute_bundle_config_hash_cached", lambda r, b: "")
    monkeypatch.setattr("sie_server.api.ws.is_ready", lambda: True)

    pull_loop = _FakePullLoop(worker_id="w-prod-3")
    status: dict[str, Any] = await build_status_message(registry, pull_loop=pull_loop)

    # The direct-dispatch routing contract: the registered name must equal the
    # subscription's worker_id.
    assert status["name"] == "w-prod-3"


@pytest.mark.asyncio
async def test_build_status_message_falls_back_to_env_without_pull_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-queue deployments (no pull loop) keep the legacy behaviour
    but now honour ``SIE_WORKER_ID`` too.
    """
    monkeypatch.setenv("SIE_WORKER_ID", "w-env-1")
    monkeypatch.setenv("HOSTNAME", "pod-other")

    from sie_server.api.ws import build_status_message

    registry = MagicMock()
    registry.memory_manager.pressure_threshold_pct = 0.0
    registry._loaded = {}
    monkeypatch.setattr("sie_server.api.ws.get_gpu_metrics", list)
    monkeypatch.setattr("sie_server.api.ws.get_model_status", lambda r: [])
    monkeypatch.setattr(
        "sie_server.api.ws.collect_prometheus_metrics",
        lambda: {"counters": {}, "histograms": {}},
    )
    monkeypatch.setattr("sie_server.api.ws.compute_bundle_config_hash_cached", lambda r, b: "")
    monkeypatch.setattr("sie_server.api.ws.is_ready", lambda: True)

    status: dict[str, Any] = await build_status_message(registry, pull_loop=None)
    assert status["name"] == "w-env-1"
