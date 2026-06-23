import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sie_server.config.model import EmbeddingDim, EncodeTask, ModelConfig, ProfileConfig, Tasks
from sie_server.core.memory import MemoryConfig
from sie_server.core.registry import ModelRegistry


def _make_config(
    name: str = "test",
    hf_id: str | None = "org/test",
    dense_dim: int = 768,
    max_sequence_length: int | None = None,
) -> ModelConfig:
    return ModelConfig(
        sie_id=name,
        hf_id=hf_id,
        tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=dense_dim))),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
                max_batch_tokens=8192,
            )
        },
        max_sequence_length=max_sequence_length,
    )


@pytest.fixture(autouse=True)
def patch_ensure_model_cached():
    """Patch ensure_model_cached to avoid actual HF downloads in tests."""
    with patch("sie_sdk.cache.ensure_model_cached") as mock:
        mock.return_value = Path("/fake/cache/models--org--test")
        yield mock


async def _drain_background_tasks(registry: ModelRegistry) -> None:
    """Await the registry's background tasks (pinned eager-load reconciles and the
    model-load tasks they spawn), so a test can observe the eventual effect.

    Fails loudly rather than hiding a regression: surfaces any task exception and
    asserts the set settles within the bounded retries (the spawned tasks are all
    fast/mocked in these tests, so a non-settling set means a real bug).
    """
    for _ in range(10):
        pending = [task for task in list(registry._background_tasks) if not task.done()]
        if not pending:
            return
        results = await asyncio.gather(*pending, return_exceptions=True)
        errors = [r for r in results if isinstance(r, BaseException)]
        assert not errors, f"background task(s) failed: {errors!r}"
    still_pending = [task for task in list(registry._background_tasks) if not task.done()]
    assert not still_pending, f"background tasks did not settle: {len(still_pending)} pending"


class TestAsyncLoading:
    """Tests for async model loading."""

    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        """Create a mock adapter."""
        mock = MagicMock()
        mock.capabilities.outputs = ["dense"]
        mock.memory_footprint.return_value = 1_000_000
        return mock

    @pytest.fixture
    def registry_with_model(self, mock_adapter: MagicMock) -> ModelRegistry:
        """Create registry with a model config ready to load."""
        registry = ModelRegistry()
        config = _make_config(name="test-model", hf_id="org/test")
        registry.add_config(config)
        return registry

    async def test_replace_configs_invalidates_when_model_dir_changes(self, tmp_path: Path) -> None:
        """Same config with a new model_dir must update adapter resolution state."""
        registry = ModelRegistry()
        config = _make_config(name="test-model", hf_id="org/test")
        old_dir = tmp_path / "old"
        new_dir = tmp_path / "new"

        await registry.replace_configs_async([config], model_dir=old_dir)
        registry._loaded["test-model"] = MagicMock()
        registry._do_unload = AsyncMock()

        invalidated = await registry.replace_configs_async(
            [_make_config(name="test-model", hf_id="org/test")],
            model_dir=new_dir,
        )

        assert invalidated == {"test-model"}
        assert registry._model_dirs["test-model"] == new_dir
        registry._do_unload.assert_awaited_once_with("test-model")

    async def test_load_async_basic(self, registry_with_model: ModelRegistry) -> None:
        """Test basic async loading."""
        with patch("sie_server.core.model_loader.load_adapter") as mock_load:
            mock_adapter = MagicMock()
            mock_adapter.memory_footprint.return_value = 1000
            mock_load.return_value = mock_adapter

            adapter = await registry_with_model.load_async("test-model", "cpu")

            assert adapter is mock_adapter
            assert registry_with_model.is_loaded("test-model")
            mock_adapter.load.assert_called_once_with("cpu")

    async def test_load_async_returns_existing_if_loaded(self, registry_with_model: ModelRegistry) -> None:
        """Second call to load_async returns existing model without reloading."""
        with patch("sie_server.core.model_loader.load_adapter") as mock_load:
            mock_adapter = MagicMock()
            mock_adapter.memory_footprint.return_value = 1000
            mock_load.return_value = mock_adapter

            # First load
            adapter1 = await registry_with_model.load_async("test-model", "cpu")
            # Second load should return same adapter without reloading
            adapter2 = await registry_with_model.load_async("test-model", "cpu")

            assert adapter1 is adapter2
            # load_adapter should only be called once
            mock_load.assert_called_once()

    async def test_load_async_concurrent_same_model(self, registry_with_model: ModelRegistry) -> None:
        """Two concurrent loads for same model only load once."""
        import asyncio

        load_count = 0
        load_event = asyncio.Event()

        def slow_load(device: str) -> None:
            nonlocal load_count
            load_count += 1
            # Signal that load started
            load_event.set()

        with patch("sie_server.core.model_loader.load_adapter") as mock_load:
            mock_adapter = MagicMock()
            mock_adapter.memory_footprint.return_value = 1000
            mock_adapter.load = slow_load
            mock_load.return_value = mock_adapter

            # Start two concurrent loads
            task1 = asyncio.create_task(registry_with_model.load_async("test-model", "cpu"))
            task2 = asyncio.create_task(registry_with_model.load_async("test-model", "cpu"))

            adapter1, adapter2 = await asyncio.gather(task1, task2)

            # Both should return the same adapter
            assert adapter1 is adapter2
            # Load should only happen once
            assert load_count == 1

    async def test_is_unloading_flag(self, registry_with_model: ModelRegistry) -> None:
        """is_unloading returns correct state."""
        assert not registry_with_model.is_unloading("test-model")

    async def test_is_loading_flag_initial_state(self, registry_with_model: ModelRegistry) -> None:
        """is_loading returns False before load starts."""
        # Model is configured but not loading yet
        assert not registry_with_model.is_loading("test-model")

    async def test_is_loading_flag_cleared_after_load(self, registry_with_model: ModelRegistry) -> None:
        """is_loading returns False after load completes."""
        with patch("sie_server.core.model_loader.load_adapter") as mock_load:
            mock_adapter = MagicMock()
            mock_adapter.memory_footprint.return_value = 1000
            mock_load.return_value = mock_adapter

            await registry_with_model.load_async("test-model", "cpu")

            # After load completes, is_loading should be False
            assert not registry_with_model.is_loading("test-model")
            assert registry_with_model.is_loaded("test-model")

    async def test_is_loading_flag_cleared_on_failure(self, registry_with_model: ModelRegistry) -> None:
        """is_loading is cleared even when load fails."""
        with patch("sie_server.core.model_loader.load_adapter") as mock_load:
            mock_load.side_effect = RuntimeError("Load failed")

            with pytest.raises(RuntimeError, match="Load failed"):
                await registry_with_model.load_async("test-model", "cpu")

            # After failure, is_loading should still be False
            assert not registry_with_model.is_loading("test-model")
            assert not registry_with_model.is_loaded("test-model")

    async def test_unload_async_drains_worker(self, registry_with_model: ModelRegistry) -> None:
        """unload_async stops worker before unloading adapter."""
        with patch("sie_server.core.model_loader.load_adapter") as mock_load:
            mock_adapter = MagicMock()
            mock_adapter.memory_footprint.return_value = 1000
            mock_load.return_value = mock_adapter

            await registry_with_model.load_async("test-model", "cpu")

            # Get the worker
            worker = registry_with_model.get_worker("test-model")
            assert worker is not None

            # Start the worker
            await registry_with_model.start_worker("test-model")

            # Now unload
            await registry_with_model.unload_async("test-model")

            assert not registry_with_model.is_loaded("test-model")
            mock_adapter.unload.assert_called_once()

    async def test_unload_async_closes_client_before_adapter_unload(self, registry_with_model: ModelRegistry) -> None:
        """H5: ``aclose_client`` (HTTP client close) is awaited BEFORE
        ``unload()`` (which terminates the SGLang subprocess). Closing the
        client against a still-live subprocess avoids leaked fds / a wedged
        half-open socket.
        """
        from unittest.mock import AsyncMock

        order: list[str] = []

        with patch("sie_server.core.model_loader.load_adapter") as mock_load:
            mock_adapter = MagicMock()
            mock_adapter.memory_footprint.return_value = 1000

            async def _aclose_client() -> None:
                order.append("aclose_client")

            def _unload() -> None:
                order.append("unload")

            mock_adapter.aclose_client = AsyncMock(side_effect=_aclose_client)
            mock_adapter.unload = MagicMock(side_effect=_unload)
            mock_load.return_value = mock_adapter

            await registry_with_model.load_async("test-model", "cpu")
            await registry_with_model.unload_async("test-model")

        mock_adapter.aclose_client.assert_awaited_once()
        mock_adapter.unload.assert_called_once()
        # Client close strictly precedes subprocess teardown.
        assert order == ["aclose_client", "unload"]

    async def test_unload_async_without_aclose_client_still_unloads(self, registry_with_model: ModelRegistry) -> None:
        """Adapters without ``aclose_client`` (e.g. embedding adapters) still
        unload cleanly — the new close path is opt-in via getattr.
        """
        with patch("sie_server.core.model_loader.load_adapter") as mock_load:
            mock_adapter = MagicMock()
            mock_adapter.capabilities.outputs = ["dense"]
            mock_adapter.memory_footprint.return_value = 1000
            # Simulate an adapter that does NOT expose ``aclose_client``
            # (e.g. an embedding adapter). ``getattr(..., None)`` must
            # short-circuit and unload still runs.
            del mock_adapter.aclose_client
            mock_load.return_value = mock_adapter

            await registry_with_model.load_async("test-model", "cpu")
            await registry_with_model.unload_async("test-model")

            mock_adapter.unload.assert_called_once()
            assert not registry_with_model.is_loaded("test-model")

    async def test_unload_all_async(self, registry_with_model: ModelRegistry) -> None:
        """unload_all_async unloads all models."""
        # Add another model
        config2 = _make_config(name="model-2", hf_id="org/test2")
        registry_with_model.add_config(config2)

        with patch("sie_server.core.model_loader.load_adapter") as mock_load:
            mock_adapter = MagicMock()
            mock_adapter.memory_footprint.return_value = 1000
            mock_load.return_value = mock_adapter

            await registry_with_model.load_async("test-model", "cpu")
            await registry_with_model.load_async("model-2", "cpu")

            assert len(registry_with_model.loaded_model_names) == 2

            await registry_with_model.unload_all_async()

            assert len(registry_with_model.loaded_model_names) == 0

    async def test_load_async_model_not_found_raises(self) -> None:
        """load_async raises KeyError for unknown model."""
        registry = ModelRegistry()

        with pytest.raises(KeyError, match="not found"):
            await registry.load_async("unknown-model", "cpu")

    async def test_load_async_while_unloading_raises(self, registry_with_model: ModelRegistry) -> None:
        """load_async raises RuntimeError if model is being unloaded."""
        # Manually set unloading flag
        registry_with_model._unloading.add("test-model")

        with pytest.raises(RuntimeError, match="currently being unloaded"):
            await registry_with_model.load_async("test-model", "cpu")


class TestProactiveEviction:
    """Tests for proactive memory eviction (pre-load and background monitor)."""

    @pytest.fixture
    def mock_adapter_factory(self) -> MagicMock:
        """Create a factory that returns fresh mock adapters."""

        def make_mock():
            mock = MagicMock()
            mock.capabilities.outputs = ["dense"]
            mock.memory_footprint.return_value = 1000
            return mock

        return make_mock

    @pytest.fixture
    def registry_with_models(self, mock_adapter_factory: MagicMock) -> ModelRegistry:
        """Create registry with 3 model configs."""
        from sie_server.core.memory import MemoryConfig

        registry = ModelRegistry(
            memory_config=MemoryConfig(pressure_threshold=0.85),
        )

        for name in ["model-a", "model-b", "model-c"]:
            config = _make_config(name=name, hf_id=f"org/{name}")
            registry.add_config(config)

        return registry

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_pre_load_eviction_triggers_when_above_threshold(
        self, mock_load_adapter: MagicMock, registry_with_models: ModelRegistry, mock_adapter_factory: MagicMock
    ) -> None:
        """Pre-load eviction evicts LRU when memory is above threshold."""
        adapters = [mock_adapter_factory() for _ in range(3)]
        mock_load_adapter.side_effect = adapters

        # Load first two models
        await registry_with_models.load_async("model-a", "cpu")
        await registry_with_models.load_async("model-b", "cpu")

        assert registry_with_models.is_loaded("model-a")
        assert registry_with_models.is_loaded("model-b")

        # Mock memory pressure (90% usage, threshold is 85%)
        with patch.object(registry_with_models._memory_manager, "check_pressure", side_effect=[True, False]):
            # Load third model - should trigger eviction of model-a (LRU)
            await registry_with_models.load_async("model-c", "cpu")

        # model-a should be evicted, model-b and model-c loaded
        assert not registry_with_models.is_loaded("model-a")
        assert registry_with_models.is_loaded("model-b")
        assert registry_with_models.is_loaded("model-c")

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_pre_load_eviction_loop_evicts_multiple(
        self, mock_load_adapter: MagicMock, registry_with_models: ModelRegistry, mock_adapter_factory: MagicMock
    ) -> None:
        """Pre-load eviction can evict multiple models until below threshold."""
        adapters = [mock_adapter_factory() for _ in range(3)]
        mock_load_adapter.side_effect = adapters

        # Load first two models
        await registry_with_models.load_async("model-a", "cpu")
        await registry_with_models.load_async("model-b", "cpu")

        # Mock memory pressure that requires evicting both models
        # check_pressure: True, True, False (evict a, evict b, then ok)
        with patch.object(registry_with_models._memory_manager, "check_pressure", side_effect=[True, True, False]):
            await registry_with_models.load_async("model-c", "cpu")

        # Both model-a and model-b should be evicted
        assert not registry_with_models.is_loaded("model-a")
        assert not registry_with_models.is_loaded("model-b")
        assert registry_with_models.is_loaded("model-c")

    async def test_background_monitor_loop_runs(self) -> None:
        """Background monitor loop runs and checks pressure periodically."""
        import asyncio

        from sie_server.core.memory import MemoryConfig

        # Create registry with short check interval for testing
        registry = ModelRegistry(
            memory_config=MemoryConfig(
                pressure_threshold=0.85,
                memory_check_interval_s=0.005,  # 5ms for fast testing
            ),
        )

        # Track how many times check_pressure is called
        check_count = 0

        def counting_check() -> bool:
            nonlocal check_count
            check_count += 1
            return False  # No pressure, so no eviction needed

        registry._memory_manager.check_pressure = counting_check

        await registry.start_memory_monitor()
        try:
            # Wait for a few check cycles
            await asyncio.sleep(0.02)
            # Should have been called multiple times
            assert check_count >= 2
        finally:
            await registry.stop_memory_monitor()

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_memory_monitor_does_not_evict_only_loaded_model(
        self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock
    ) -> None:
        """Pressure monitor keeps a sole resident model to avoid reload loops."""
        registry = ModelRegistry(
            memory_config=MemoryConfig(
                pressure_threshold=0.95,
                memory_check_interval_s=0.005,
            ),
        )
        registry.add_config(_make_config(name="model-a", hf_id="org/model-a"))
        mock_adapter = mock_adapter_factory()
        mock_load_adapter.return_value = mock_adapter

        await registry.load_async("model-a", "cpu")
        registry._memory_manager.check_pressure = MagicMock(return_value=True)

        await registry.start_memory_monitor()
        try:
            await asyncio.sleep(0.02)
        finally:
            await registry.stop_memory_monitor()

        assert registry.is_loaded("model-a")
        mock_adapter.unload.assert_not_called()

    async def test_memory_monitor_starts_and_stops(self) -> None:
        """Memory monitor can be started and stopped cleanly."""
        from sie_server.core.memory import MemoryConfig

        registry = ModelRegistry(
            memory_config=MemoryConfig(memory_check_interval_s=0.1),
        )

        assert registry._monitor_task is None
        assert not registry._monitor_running

        await registry.start_memory_monitor()

        assert registry._monitor_task is not None
        assert registry._monitor_running

        await registry.stop_memory_monitor()

        assert registry._monitor_task is None
        assert not registry._monitor_running

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_adapter_unload_called_on_unload(
        self, mock_load_adapter: MagicMock, registry_with_models: ModelRegistry, mock_adapter_factory: MagicMock
    ) -> None:
        """Adapter.unload() is called when model is unloaded.

        Memory cleanup (gc.collect + empty_cache) is the adapter's responsibility.
        See the memory management contract in ModelAdapter docstring (base.py).
        """
        mock_adapter = mock_adapter_factory()
        mock_load_adapter.return_value = mock_adapter

        await registry_with_models.load_async("model-a", "cuda:0")
        await registry_with_models.unload_async("model-a")

        mock_adapter.unload.assert_called_once()


class TestSetPinnedModels:
    """Tests for the runtime pinned-set mutator (gateway -> worker bridge)."""

    def _registry(self, pinned: list[str] | None = None) -> ModelRegistry:
        registry = ModelRegistry(pinned_models=pinned)
        registry.add_config(_make_config(name="model-a", hf_id="org/model-a"))
        registry.add_config(_make_config(name="model-b", hf_id="org/model-b"))
        # Spy on eager-load so tests assert intent without driving real loads.
        registry.start_load_async = AsyncMock(return_value=True)  # type: ignore[method-assign]
        return registry

    async def test_sets_pinned_and_eager_loads_new(self) -> None:
        registry = self._registry()
        result = await registry.set_pinned_models(["model-a"])

        assert result == frozenset({"model-a"})
        assert registry._pinned_models == frozenset({"model-a"})
        registry.start_load_async.assert_awaited_once_with("model-a", "cpu")

    async def test_replaces_set_and_demotes_removed(self) -> None:
        registry = self._registry(pinned=["model-a"])
        registry.start_load_async.reset_mock()

        await registry.set_pinned_models(["model-b"])

        # model-a is no longer pinned (demoted to evictable); model-b is now pinned.
        assert registry._pinned_models == frozenset({"model-b"})
        assert not registry._is_pinned("model-a")
        registry.start_load_async.assert_awaited_once_with("model-b", "cpu")

    async def test_empty_set_unpins_all(self) -> None:
        registry = self._registry(pinned=["model-a"])
        registry.start_load_async.reset_mock()

        await registry.set_pinned_models([])

        assert registry._pinned_models == frozenset()
        registry.start_load_async.assert_not_awaited()

    async def test_idempotent_noop_when_unchanged(self) -> None:
        registry = self._registry(pinned=["model-a"])
        registry.start_load_async.reset_mock()

        await registry.set_pinned_models(["model-a"])

        # Already loaded-or-pinned: unchanged set must not re-trigger eager-load.
        registry.start_load_async.assert_not_awaited()

    async def test_profile_variant_pin_loads_and_protects_the_variant(self) -> None:
        # A non-default profile is a first-class config entry keyed by the full
        # "base:profile" id; pinning it must load/protect the VARIANT, not the base.
        registry = ModelRegistry()
        registry.add_config(_make_config(name="Org/Model-A", hf_id="org/model-a"))
        registry.add_config(_make_config(name="Org/Model-A:fp8", hf_id="org/model-a"))
        registry.start_load_async = AsyncMock(return_value=True)  # type: ignore[method-assign]

        await registry.set_pinned_models(["Org/Model-A:FP8"])

        # Stored as the lowercased profile-qualified id; eager-load targets the variant.
        assert registry._pinned_models == frozenset({"org/model-a:fp8"})
        registry.start_load_async.assert_awaited_once_with("Org/Model-A:fp8", "cpu")
        # The variant is protected from eviction; the base (never pinned) is not.
        assert registry._is_pinned("Org/Model-A:fp8")
        assert not registry._is_pinned("Org/Model-A")

    async def test_default_profile_folds_to_base(self) -> None:
        registry = ModelRegistry()
        registry.add_config(_make_config(name="org/model-a", hf_id="org/model-a"))
        registry.start_load_async = AsyncMock(return_value=True)  # type: ignore[method-assign]

        await registry.set_pinned_models(["org/model-a:default"])

        assert registry._pinned_models == frozenset({"org/model-a"})
        registry.start_load_async.assert_awaited_once_with("org/model-a", "cpu")

    async def test_unknown_config_is_pinned_but_not_loaded(self) -> None:
        registry = self._registry()

        await registry.set_pinned_models(["missing/model"])

        # Pinned (protected if it ever loads) but not eager-loaded (no local config).
        assert registry._pinned_models == frozenset({"missing/model"})
        registry.start_load_async.assert_not_awaited()

    async def test_already_loaded_pinned_not_reloaded(self) -> None:
        registry = self._registry()
        registry._loaded["model-a"] = MagicMock()  # simulate already resident

        await registry.set_pinned_models(["model-a"])

        assert registry._pinned_models == frozenset({"model-a"})
        registry.start_load_async.assert_not_awaited()

    async def test_add_config_eager_loads_already_pinned_model(self) -> None:
        # P1b: a pin recorded before its config arrives must load once the config
        # is added at runtime (default Helm workers receive configs via the sidecar).
        registry = ModelRegistry(pinned_models=["late/model"])
        registry.start_load_async = AsyncMock(return_value=True)  # type: ignore[method-assign]

        # No config yet: nothing to load.
        await registry._eager_load_pinned()
        registry.start_load_async.assert_not_awaited()

        # Config arrives -> add_config schedules a background eager-load reconcile.
        registry.add_config(_make_config(name="late/model", hf_id="late/hf"))
        await _drain_background_tasks(registry)

        registry.start_load_async.assert_any_await("late/model", "cpu")

    async def test_add_config_ignores_non_pinned_model(self) -> None:
        registry = ModelRegistry(pinned_models=["pinned/model"])
        registry.start_load_async = AsyncMock(return_value=True)  # type: ignore[method-assign]

        registry.add_config(_make_config(name="other/model", hf_id="other/hf"))
        await _drain_background_tasks(registry)

        registry.start_load_async.assert_not_awaited()

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_replace_configs_reloads_changed_pinned_model(
        self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock
    ) -> None:
        # P1b: replace_configs_async unloads a changed pinned model; it must reload it.
        mock_load_adapter.side_effect = [mock_adapter_factory(), mock_adapter_factory()]
        registry = ModelRegistry(pinned_models=["org/model"])
        registry.add_config(_make_config(name="org/model", hf_id="org/model-hf"))
        await registry.load_async("org/model", "cpu")
        assert registry.is_loaded("org/model")

        # Replace with a semantically-changed config (different dim) -> unload + reload.
        await registry.replace_configs_async([_make_config(name="org/model", hf_id="org/model-hf", dense_dim=512)])
        await _drain_background_tasks(registry)

        assert registry.is_loaded("org/model")

    @patch("sie_server.core.model_loader.load_adapter")
    async def test_replace_configs_loads_newly_known_pinned_model(
        self, mock_load_adapter: MagicMock, mock_adapter_factory: MagicMock
    ) -> None:
        # P1b: a pin whose config first appears via replace_configs must load.
        mock_load_adapter.return_value = mock_adapter_factory()
        registry = ModelRegistry(pinned_models=["org/model"])

        await registry.replace_configs_async([_make_config(name="org/model", hf_id="org/model-hf")])
        await _drain_background_tasks(registry)

        assert registry.is_loaded("org/model")

    @pytest.fixture
    def mock_adapter_factory(self) -> MagicMock:
        def make_mock():
            mock = MagicMock()
            mock.capabilities.outputs = ["dense"]
            mock.memory_footprint.return_value = 1_000_000
            return mock

        return make_mock
