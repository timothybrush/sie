"""Tests for the FastAPI app factory."""

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig
from sie_server.config.model import EmbeddingDim, EncodeTask, ModelConfig, ProfileConfig, Tasks
from sie_server.core import readiness
from sie_server.core.postprocessor_registry import PostprocessorRegistry


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the SIE server."""
    app = AppFactory.create_app(AppStateConfig())
    return TestClient(app)


class TestAppFactory:
    """Tests for the FastAPI app factory."""

    def test_create_app_returns_fastapi_instance(self) -> None:
        """App factory returns a FastAPI application."""
        app = AppFactory.create_app(AppStateConfig())
        assert isinstance(app, FastAPI)

    def test_app_has_correct_metadata(self) -> None:
        """App has correct title and version."""
        app = AppFactory.create_app(AppStateConfig())
        assert app.title == "SIE Server"
        assert app.version == "0.1.0"

    def test_health_routes_registered(self, client: TestClient) -> None:
        """Health routes are registered in the app."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        openapi = response.json()
        paths = openapi["paths"]

        assert "/healthz" in paths
        assert "/livez" in paths
        assert "/readyz" in paths


class TestAllRoutersAlwaysMounted:
    """Queue routing is the only supported mode, so routers are mounted
    unconditionally.

    /ws/status MUST be mounted because it is the gateway's
    worker-registration channel: the gateway opens
    `ws://<worker>/ws/status` to learn pool_name, bundle,
    machine_profile, and readiness. Without it, the gateway's
    WorkerRegistry stays empty and every request falls through to the
    202 "no queue worker available" branch. This test suite locks that
    contract in.
    """

    @staticmethod
    def _route_paths(app: FastAPI) -> set[str]:
        return {getattr(r, "path", "") for r in app.router.routes}

    def test_ws_status_is_always_mounted(self) -> None:
        app = AppFactory.create_app(AppStateConfig())
        assert "/ws/status" in self._route_paths(app)

    def test_probe_and_metrics_routes_are_always_mounted(self) -> None:
        app = AppFactory.create_app(AppStateConfig())
        paths = self._route_paths(app)
        for path in ("/healthz", "/readyz", "/metrics"):
            assert path in paths, f"{path} missing — breaks K8s probes/Prometheus"

    def test_inference_routers_are_mounted(self) -> None:
        """Inference routers stay mounted unconditionally.

        In production these endpoints are not a real ingress (the Rust
        gateway is queue-only and publishes to JetStream, not HTTP),
        but the routes exist on every pod for local debugging and to
        keep the existing test/OpenAPI surface intact. If a future
        refactor deletes them, update this test along with the router
        imports in ``app_factory.create_app``.
        """
        app = AppFactory.create_app(AppStateConfig())
        paths = self._route_paths(app)
        for path in (
            "/v1/encode/{model:path}",
            "/v1/score/{model:path}",
            "/v1/extract/{model:path}",
            "/v1/embeddings",
            "/v1/models",
        ):
            assert path in paths, f"{path} router missing"


class TestIpcHeartbeatReadiness:
    """IPC heartbeat readiness is opt-in so direct HTTP stays usable."""

    @staticmethod
    def _short_socket_path() -> Path:
        # macOS caps AF_UNIX paths at 104 bytes; pytest temp roots can exceed it.
        return Path("/tmp") / f"sie-ipc-{uuid.uuid4().hex[:12]}.sock"  # noqa: S108

    @pytest.mark.asyncio
    async def test_direct_server_readiness_does_not_require_sidecar_ping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SIE_IPC_REQUIRE_HEARTBEAT", raising=False)
        socket_path = self._short_socket_path()
        monkeypatch.setenv("SIE_IPC_SOCKET_PATH", str(socket_path))
        readiness.register_liveness_probe(None)
        readiness.mark_ready()

        try:
            async with AppFactory._ipc_server(MagicMock()):
                assert readiness.is_ready() is True
        finally:
            readiness.mark_not_ready()
            readiness.register_liveness_probe(None)
            socket_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_sidecar_deployments_gate_readiness_on_ipc_ping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIE_IPC_REQUIRE_HEARTBEAT", "true")
        socket_path = self._short_socket_path()
        monkeypatch.setenv("SIE_IPC_SOCKET_PATH", str(socket_path))
        readiness.register_liveness_probe(None)
        readiness.mark_ready()

        try:
            async with AppFactory._ipc_server(MagicMock()):
                assert readiness.is_ready() is False
        finally:
            readiness.mark_not_ready()
            readiness.register_liveness_probe(None)
            socket_path.unlink(missing_ok=True)


def _direct_encode_adapter_output(items: list[Any], output_types: list[str], **_kwargs: Any) -> Any:
    from sie_server.core.inference_output import EncodeOutput

    dense = None
    if "dense" in output_types:
        dense = np.array([[0.1, 0.2, 0.3]] * len(items), dtype=np.float32)

    return EncodeOutput(
        dense=dense,
        sparse=None,
        multivector=None,
        batch_size=len(items),
        dense_dim=3 if dense is not None else None,
        multivector_token_dim=None,
    )


def _direct_http_registry() -> tuple[MagicMock, ThreadPoolExecutor]:
    adapter = MagicMock()
    adapter.encode = MagicMock(side_effect=_direct_encode_adapter_output)

    registry = MagicMock()
    registry.has_model.return_value = True
    registry.is_loaded.return_value = True
    registry.is_loading.return_value = False
    registry.is_unloading.return_value = False
    registry.is_failed.return_value = False
    registry.get_failure.return_value = None
    registry.get.return_value = adapter
    registry.get_config.return_value = ModelConfig(
        sie_id="test-model",
        hf_id="org/test-model",
        tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=3))),
        profiles={"default": ProfileConfig(adapter_path="test:Adapter", max_batch_tokens=8192)},
    )
    registry.model_names = ["test-model"]
    registry.device = "cpu"

    preprocessor_registry = MagicMock()
    preprocessor_registry.has_tokenizer.return_value = False
    preprocessor_registry.has_preprocessor.return_value = False
    registry.preprocessor_registry = preprocessor_registry

    executor = ThreadPoolExecutor(max_workers=1)
    registry.postprocessor_registry = PostprocessorRegistry(executor)
    return registry, executor


class TestDirectServerHttpPath:
    """Direct Python HTTP inference remains usable without the worker-sidecar."""

    def test_app_factory_mounts_direct_generate_route(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SIE_IPC_REQUIRE_HEARTBEAT", raising=False)
        app = AppFactory.create_app(AppStateConfig())
        paths = {getattr(route, "path", "") for route in app.routes}
        assert "/v1/generate/{model:path}" in paths

    def test_app_factory_direct_encode_path_without_sidecar(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SIE_IPC_REQUIRE_HEARTBEAT", raising=False)
        app = AppFactory.create_app(AppStateConfig())
        registry, executor = _direct_http_registry()
        app.state.registry = registry

        try:
            client = TestClient(app)
            response = client.post(
                "/v1/encode/test-model",
                json={"items": [{"text": "direct python path"}]},
                headers={"Accept": "application/json"},
            )
            models_response = client.get("/v1/models")
        finally:
            executor.shutdown(wait=True)

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "test-model"
        assert data["items"][0]["dense"]["dims"] == 3
        assert data["items"][0]["dense"]["values"] == pytest.approx([0.1, 0.2, 0.3])

        assert models_response.status_code == 200
        assert models_response.json()["models"][0]["name"] == "test-model"


class TestPreloadModels:
    """Tests for the _preload_models startup behavior."""

    @pytest.mark.asyncio
    async def test_preload_loads_models(self) -> None:
        """_preload_models calls load_async for each model."""
        registry = AsyncMock()
        config = AppStateConfig(device="cpu", preload_models=["model-a", "model-b"])

        await AppFactory._preload_models(registry, config)

        assert registry.load_async.call_count == 2
        registry.load_async.assert_any_call("model-a", "cpu")
        registry.load_async.assert_any_call("model-b", "cpu")

    @pytest.mark.asyncio
    async def test_preload_skips_when_none(self) -> None:
        """_preload_models is a no-op when preload_models is None."""
        registry = AsyncMock()
        config = AppStateConfig(device="cpu", preload_models=None)

        await AppFactory._preload_models(registry, config)

        registry.load_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_preload_continues_on_failure(self) -> None:
        """Failed preload doesn't prevent other models from loading."""
        registry = AsyncMock()
        registry.load_async.side_effect = [RuntimeError("OOM"), AsyncMock()]
        config = AppStateConfig(device="cpu", preload_models=["model-a", "model-b"])

        await AppFactory._preload_models(registry, config)  # Should not raise

        assert registry.load_async.call_count == 2


class TestPreloadModelsEnvRoundTrip:
    """Tests for preload_models env var serialization."""

    def test_preload_models_env_round_trip(self, monkeypatch) -> None:
        """preload_models survives save_to_env_vars / from_env_vars cycle."""
        # Clean env first
        monkeypatch.delenv("SIE_PRELOAD_MODELS", raising=False)
        monkeypatch.delenv("SIE_MODELS_DIR", raising=False)
        monkeypatch.delenv("SIE_MODEL_FILTER", raising=False)
        monkeypatch.delenv("SIE_DEVICE", raising=False)
        monkeypatch.delenv("SIE_POOL", raising=False)

        config = AppStateConfig(preload_models=["model-a", "model-b"])
        config.save_to_env_vars()
        restored = AppStateConfig.from_env_vars()
        assert restored.preload_models == ["model-a", "model-b"]

    def test_preload_models_none_round_trip(self, monkeypatch) -> None:
        """preload_models=None survives env round-trip."""
        monkeypatch.delenv("SIE_PRELOAD_MODELS", raising=False)
        monkeypatch.delenv("SIE_MODELS_DIR", raising=False)
        monkeypatch.delenv("SIE_MODEL_FILTER", raising=False)
        monkeypatch.delenv("SIE_DEVICE", raising=False)
        monkeypatch.delenv("SIE_POOL", raising=False)

        config = AppStateConfig(preload_models=None)
        config.save_to_env_vars()
        restored = AppStateConfig.from_env_vars()
        assert restored.preload_models is None
        assert restored.pool_name is None

    def test_pool_name_env_round_trip(self, monkeypatch) -> None:
        """SIE_POOL survives save_to_env_vars / from_env_vars cycle."""
        monkeypatch.delenv("SIE_PRELOAD_MODELS", raising=False)
        monkeypatch.delenv("SIE_MODELS_DIR", raising=False)
        monkeypatch.delenv("SIE_MODEL_FILTER", raising=False)
        monkeypatch.delenv("SIE_DEVICE", raising=False)
        monkeypatch.delenv("SIE_POOL", raising=False)

        config = AppStateConfig(pool_name="customer-a")
        config.save_to_env_vars()
        restored = AppStateConfig.from_env_vars()
        assert restored.pool_name == "customer-a"


class TestModelRegistryConfig:
    @pytest.mark.asyncio
    async def test_model_registry_receives_pool_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        created_kwargs: dict[str, Any] = {}

        class FakeRegistry:
            async def start_memory_monitor(self) -> None:
                pass

            async def start_idle_evictor(self) -> None:
                pass

            async def start_hot_reload(self) -> None:
                pass

            async def stop_memory_monitor(self) -> None:
                pass

            async def stop_idle_evictor(self) -> None:
                pass

            async def stop_hot_reload(self) -> None:
                pass

            async def unload_all_async(self) -> None:
                pass

        def fake_model_registry(**kwargs: Any) -> FakeRegistry:
            created_kwargs.update(kwargs)
            return FakeRegistry()

        monkeypatch.setattr("sie_server.app.app_factory.ModelRegistry", fake_model_registry)

        async with AppFactory._model_registry(AppStateConfig(pool_name="pool-a")):
            pass

        assert created_kwargs["pool_name"] == "pool-a"


class TestConfigureTorchThreads:
    def test_default_uses_half_cpu_count(self, monkeypatch) -> None:
        monkeypatch.delenv("SIE_TORCH_NUM_THREADS", raising=False)
        monkeypatch.setattr("sie_server.app.app_factory.os.cpu_count", lambda: 8)
        set_n = MagicMock()
        set_interop = MagicMock()
        monkeypatch.setattr("sie_server.app.app_factory.torch.set_num_threads", set_n)
        monkeypatch.setattr("sie_server.app.app_factory.torch.set_num_interop_threads", set_interop)
        AppFactory._configure_torch_threads()
        set_n.assert_called_once_with(4)
        set_interop.assert_called_once_with(1)

    def test_default_floor_when_cpu_count_is_none(self, monkeypatch) -> None:
        monkeypatch.delenv("SIE_TORCH_NUM_THREADS", raising=False)
        monkeypatch.setattr("sie_server.app.app_factory.os.cpu_count", lambda: None)
        set_n = MagicMock()
        monkeypatch.setattr("sie_server.app.app_factory.torch.set_num_threads", set_n)
        monkeypatch.setattr("sie_server.app.app_factory.torch.set_num_interop_threads", MagicMock())
        AppFactory._configure_torch_threads()
        set_n.assert_called_once_with(2)

    def test_env_override_honored(self, monkeypatch) -> None:
        monkeypatch.setenv("SIE_TORCH_NUM_THREADS", "3")
        set_n = MagicMock()
        monkeypatch.setattr("sie_server.app.app_factory.torch.set_num_threads", set_n)
        monkeypatch.setattr("sie_server.app.app_factory.torch.set_num_interop_threads", MagicMock())
        AppFactory._configure_torch_threads()
        set_n.assert_called_once_with(3)

    def test_invalid_env_override_falls_back(self, monkeypatch, caplog) -> None:
        monkeypatch.setenv("SIE_TORCH_NUM_THREADS", "abc")
        monkeypatch.setattr("sie_server.app.app_factory.os.cpu_count", lambda: 8)
        set_n = MagicMock()
        monkeypatch.setattr("sie_server.app.app_factory.torch.set_num_threads", set_n)
        monkeypatch.setattr("sie_server.app.app_factory.torch.set_num_interop_threads", MagicMock())
        with caplog.at_level(logging.WARNING, logger="sie_server.app.app_factory"):
            AppFactory._configure_torch_threads()
        set_n.assert_called_once_with(4)
        assert any("not a positive integer" in r.message for r in caplog.records)

    def test_zero_env_override_falls_back(self, monkeypatch, caplog) -> None:
        monkeypatch.setenv("SIE_TORCH_NUM_THREADS", "0")
        monkeypatch.setattr("sie_server.app.app_factory.os.cpu_count", lambda: 8)
        set_n = MagicMock()
        monkeypatch.setattr("sie_server.app.app_factory.torch.set_num_threads", set_n)
        monkeypatch.setattr("sie_server.app.app_factory.torch.set_num_interop_threads", MagicMock())
        with caplog.at_level(logging.WARNING, logger="sie_server.app.app_factory"):
            AppFactory._configure_torch_threads()
        set_n.assert_called_once_with(4)
        assert any("not a positive integer" in r.message for r in caplog.records)

    def test_interop_runtime_error_is_swallowed(self, monkeypatch, caplog) -> None:
        monkeypatch.delenv("SIE_TORCH_NUM_THREADS", raising=False)
        monkeypatch.setattr("sie_server.app.app_factory.torch.set_num_threads", MagicMock())

        def _raise(_n: int) -> None:
            raise RuntimeError("parallel runtime already started")

        monkeypatch.setattr("sie_server.app.app_factory.torch.set_num_interop_threads", _raise)
        with caplog.at_level(logging.WARNING, logger="sie_server.app.app_factory"):
            AppFactory._configure_torch_threads()  # must not raise
        assert any("set_num_interop_threads" in r.message for r in caplog.records)
