"""Tests for the GPU health probe (issue #1025).

A wedged CUDA context (e.g. after a device-side assert) keeps returning the same
sticky error on every CUDA call. A process-alive check never touches CUDA and so
reports healthy forever. The GPU health probe forces a tiny CUDA sync so callers
can detect the wedge.
"""

import pytest
from sie_server.core import gpu_health
from sie_server.core.gpu_health import (
    gpu_is_healthy,
    gpu_is_healthy_async,
    reset_gpu_health_cache,
)


@pytest.fixture(autouse=True)
def _gpu_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test starts with a clean probe cache and a single simulated GPU.

    Tests that exercise the probe patch ``is_available`` to True; pinning
    ``device_count`` to 1 keeps the probe loop deterministic (one device).
    """
    reset_gpu_health_cache()
    monkeypatch.setattr(gpu_health.torch.cuda, "device_count", lambda: 1)


class TestGpuIsHealthy:
    """Tests for gpu_is_healthy()."""

    def test_cpu_only_worker_is_healthy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no CUDA device there is nothing to wedge, so report healthy."""
        monkeypatch.setattr(gpu_health.torch.cuda, "is_available", lambda: False)
        assert gpu_is_healthy() is True

    def test_healthy_gpu_returns_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A probe that completes means the CUDA context can run kernels."""
        monkeypatch.setattr(gpu_health.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(gpu_health, "_run_gpu_probe", lambda device: None)
        assert gpu_is_healthy() is True

    def test_wedged_gpu_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A sticky CUDA error surfaces from the probe and marks the GPU unhealthy."""
        monkeypatch.setattr(gpu_health.torch.cuda, "is_available", lambda: True)

        def boom(device: str) -> None:
            raise RuntimeError("CUDA error: device-side assert triggered")

        monkeypatch.setattr(gpu_health, "_run_gpu_probe", boom)
        assert gpu_is_healthy() is False

    def test_oom_is_treated_as_healthy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A transient OutOfMemoryError is memory pressure, not a wedge → healthy.

        OutOfMemoryError subclasses RuntimeError; the device can still run kernels
        once memory frees, so it must not depool/restart a recoverable worker.
        """
        monkeypatch.setattr(gpu_health.torch.cuda, "is_available", lambda: True)

        def oom(device: str) -> None:
            raise gpu_health.torch.cuda.OutOfMemoryError("CUDA out of memory")

        monkeypatch.setattr(gpu_health, "_run_gpu_probe", oom)
        assert gpu_is_healthy() is True

    def test_non_runtime_error_is_unhealthy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A non-RuntimeError CUDA fault is caught (not escaped) and reported unhealthy."""
        monkeypatch.setattr(gpu_health.torch.cuda, "is_available", lambda: True)

        def boom(device: str) -> None:
            raise ValueError("unexpected accelerator error")

        monkeypatch.setattr(gpu_health, "_run_gpu_probe", boom)
        assert gpu_is_healthy() is False

    def test_probes_every_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All visible devices are probed; a wedge on a non-default device is caught."""
        monkeypatch.setattr(gpu_health.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(gpu_health.torch.cuda, "device_count", lambda: 2)
        probed: list[str] = []

        def probe(device: str) -> None:
            probed.append(device)
            if device == "cuda:1":
                raise RuntimeError("CUDA error: device-side assert triggered")

        monkeypatch.setattr(gpu_health, "_run_gpu_probe", probe)
        assert gpu_is_healthy() is False
        assert probed == ["cuda:0", "cuda:1"]

    def test_result_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Repeated calls within the TTL reuse the result instead of re-syncing CUDA."""
        monkeypatch.setattr(gpu_health.torch.cuda, "is_available", lambda: True)
        calls = {"n": 0}

        def counting_probe(device: str) -> None:
            calls["n"] += 1

        monkeypatch.setattr(gpu_health, "_run_gpu_probe", counting_probe)

        assert gpu_is_healthy() is True
        assert gpu_is_healthy() is True
        assert calls["n"] == 1

    def test_use_cache_false_forces_reprobe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """use_cache=False bypasses the TTL cache and re-runs the probe."""
        monkeypatch.setattr(gpu_health.torch.cuda, "is_available", lambda: True)
        calls = {"n": 0}

        def counting_probe(device: str) -> None:
            calls["n"] += 1

        monkeypatch.setattr(gpu_health, "_run_gpu_probe", counting_probe)

        assert gpu_is_healthy(use_cache=False) is True
        assert gpu_is_healthy(use_cache=False) is True
        assert calls["n"] == 2

    def test_cpu_only_does_not_call_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The CUDA probe is never run when no GPU is present."""
        monkeypatch.setattr(gpu_health.torch.cuda, "is_available", lambda: False)

        def fail(device: str) -> None:
            raise AssertionError("probe must not run on a CPU-only worker")

        monkeypatch.setattr(gpu_health, "_run_gpu_probe", fail)
        assert gpu_is_healthy() is True


class TestGpuIsHealthyAsync:
    """The async entry point runs the blocking probe off the event loop."""

    @pytest.mark.asyncio
    async def test_async_healthy_returns_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gpu_health.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(gpu_health, "_run_gpu_probe", lambda device: None)
        assert await gpu_is_healthy_async() is True

    @pytest.mark.asyncio
    async def test_async_wedged_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gpu_health.torch.cuda, "is_available", lambda: True)

        def boom(device: str) -> None:
            raise RuntimeError("CUDA error: device-side assert triggered")

        monkeypatch.setattr(gpu_health, "_run_gpu_probe", boom)
        assert await gpu_is_healthy_async() is False

    @pytest.mark.asyncio
    async def test_async_cpu_only_is_healthy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gpu_health.torch.cuda, "is_available", lambda: False)
        assert await gpu_is_healthy_async() is True
