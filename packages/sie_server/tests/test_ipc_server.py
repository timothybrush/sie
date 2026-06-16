from __future__ import annotations

import asyncio
import contextlib
import os
import struct
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import numpy as np
import pytest
from sie_server.api.ws import compute_bundle_config_hash_cached
from sie_server.core.inference_output import ExtractOutput, ScoreOutput
from sie_server.core.registry import ModelRegistry
from sie_server.core.timing import RequestTiming
from sie_server.core.worker.types import WorkerResult
from sie_server.ipc_server import IpcServer, IpcServerError
from sie_server.ipc_types import (
    IPC_VERSION,
    ApplyModelConfigRequest,
    ReplaceModelConfigEntry,
    ReplaceModelConfigsRequest,
)
from sie_server.queue_executor import QueueExecutor

_LEN_STRUCT = struct.Struct("!I")


def _short_sock_path() -> Path:
    """Return a UDS path short enough for AF_UNIX on macOS (~104 char limit).

    pytest's ``tmp_path`` ends up under ``/private/var/folders/.../`` which
    overshoots the limit — put sockets in ``/tmp`` with a short random name.
    """
    base = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
    # Fall back to /tmp if TMPDIR is itself too long (macOS default TMPDIR is).
    if len(str(base)) > 20:
        base = Path("/tmp")  # noqa: S108
    return base / f"sie-{uuid.uuid4().hex[:12]}.sock"


class _Client:
    """Minimal UDS IPC client used only by these tests."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = writer

    @classmethod
    async def connect(cls, path: Path) -> _Client:
        r, w = await asyncio.open_unix_connection(str(path))
        return cls(r, w)

    async def close(self) -> None:
        self.writer.close()
        with contextlib.suppress(Exception):
            await self.writer.wait_closed()

    async def rpc(self, method: str, body: dict, *, request_id: str = "r1") -> dict:
        envelope = {
            "version": IPC_VERSION,
            "method": method,
            "request_id": request_id,
            "body": body,
        }
        payload = msgpack.packb(envelope, use_bin_type=True)
        self.writer.write(_LEN_STRUCT.pack(len(payload)) + payload)
        await self.writer.drain()

        header = await self.reader.readexactly(_LEN_STRUCT.size)
        (length,) = _LEN_STRUCT.unpack(header)
        resp_bytes = await self.reader.readexactly(length)
        return msgpack.unpackb(resp_bytes, raw=False)


def _make_executor() -> tuple[QueueExecutor, MagicMock]:
    reg = MagicMock()
    reg.model_names = ["test/model"]
    reg.device = "cpu"
    reg.is_loaded.return_value = True
    reg.is_loading.return_value = False
    reg.get_config.return_value = MagicMock()
    reg.get_configs_snapshot.return_value = {}
    return QueueExecutor(reg), reg


@pytest.fixture
async def server_and_path():
    executor, _reg = _make_executor()
    sock = _short_sock_path()
    srv = IpcServer(sock, executor, worker_id="worker-test", stale_after_ms=10_000)
    await srv.start()
    try:
        yield srv, sock
    finally:
        await srv.stop(drain_timeout_s=1.0)


# -----------------------------------------------------------------------------
# Transport / framing
# -----------------------------------------------------------------------------


class TestFraming:
    @pytest.mark.asyncio
    async def test_ping_roundtrip(self, server_and_path) -> None:
        srv, sock = server_and_path
        client = await _Client.connect(sock)
        try:
            resp = await client.rpc("Ping", {"timestamp_ms": 123.0})
            assert resp["ok"] is True
            assert resp["version"] == IPC_VERSION
            assert resp["request_id"] == "r1"
            assert resp["body"]["timestamp_ms"] == 123.0
            assert resp["body"]["worker_id"] == "worker-test"
            assert resp["body"]["bundle_config_hash"] == ""
            assert srv.is_heartbeat_fresh()
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_multiple_requests_on_one_connection(self, server_and_path) -> None:
        _srv, sock = server_and_path
        client = await _Client.connect(sock)
        try:
            r1 = await client.rpc("Ping", {"timestamp_ms": 1.0}, request_id="a")
            r2 = await client.rpc("Ping", {"timestamp_ms": 2.0}, request_id="b")
            assert r1["request_id"] == "a"
            assert r2["request_id"] == "b"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_unknown_method_returns_error(self, server_and_path) -> None:
        _srv, sock = server_and_path
        client = await _Client.connect(sock)
        try:
            resp = await client.rpc("NoSuchMethod", {})
            assert resp["ok"] is False
            assert "unknown method" in (resp["error"] or "")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_bad_envelope_version_returns_error(self) -> None:
        executor, _reg = _make_executor()
        sock = _short_sock_path()
        async with IpcServer(sock, executor, worker_id="w"):
            client = await _Client.connect(sock)
            try:
                envelope = {"version": 999, "method": "Ping", "request_id": "r1", "body": {}}
                payload = msgpack.packb(envelope, use_bin_type=True)
                client.writer.write(_LEN_STRUCT.pack(len(payload)) + payload)
                await client.writer.drain()
                header = await client.reader.readexactly(_LEN_STRUCT.size)
                (length,) = _LEN_STRUCT.unpack(header)
                resp_bytes = await client.reader.readexactly(length)
                resp = msgpack.unpackb(resp_bytes, raw=False)
                assert resp["ok"] is False
                assert "unsupported version" in resp["error"]
            finally:
                await client.close()

    @pytest.mark.asyncio
    async def test_malformed_msgpack_closes_connection(self, server_and_path) -> None:
        _srv, sock = server_and_path
        client = await _Client.connect(sock)
        try:
            client.writer.write(_LEN_STRUCT.pack(3) + b"\xff\xff\xff")
            await client.writer.drain()
            data = await client.reader.read(1)
            assert data == b""  # connection closed on protocol error
        finally:
            await client.close()


# -----------------------------------------------------------------------------
# EnsureModelReady
# -----------------------------------------------------------------------------


class TestEnsureModelReady:
    @pytest.mark.asyncio
    async def test_ready_model(self, server_and_path) -> None:
        _srv, sock = server_and_path
        client = await _Client.connect(sock)
        try:
            resp = await client.rpc("EnsureModelReady", {"model_id": "test/model"})
            assert resp["ok"] is True
            assert resp["body"]["state"] == "ready"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_batch_budget_included_on_ready(self) -> None:
        """Parity #4: EnsureModelReady carries per-model batch budget for the
        Rust consumer's fair-dispatch cap.
        """
        reg = MagicMock()
        reg.model_names = ["test/model"]
        reg.device = "cpu"
        reg.is_loaded.return_value = True
        reg.is_loading.return_value = False
        worker = MagicMock()
        worker._batch_config = MagicMock(max_batch_requests=42)
        reg.get_worker.return_value = worker
        executor = QueueExecutor(reg)
        sock = _short_sock_path()
        srv = IpcServer(sock, executor, worker_id="worker-test", stale_after_ms=10_000)
        await srv.start()
        try:
            client = await _Client.connect(sock)
            try:
                resp = await client.rpc("EnsureModelReady", {"model_id": "test/model"})
                assert resp["ok"] is True
                assert resp["body"]["state"] == "ready"
                assert resp["body"]["batch_budget"] == 42
            finally:
                await client.close()
        finally:
            await srv.stop(drain_timeout_s=1.0)


# -----------------------------------------------------------------------------
# ApplyModelConfig
# -----------------------------------------------------------------------------


class TestApplyModelConfig:
    def test_apply_model_config_hash_cache_is_per_registry(self) -> None:
        stale_registry = ModelRegistry(models_dir=None)
        stale_registry._config_version = 1
        assert compute_bundle_config_hash_cached(stale_registry, "default") == ""

        registry = ModelRegistry(models_dir=None, model_filter=["existing/model"])
        executor = QueueExecutor(registry)

        resp = executor.apply_model_config(
            ApplyModelConfigRequest(
                bundle_id="default",
                model_id="test/model",
                epoch=7,
                bundle_config_hash="from-sie-config",
                profiles_added=["default"],
                model_config="""
sie_id: test/model
hf_id: sentence-transformers/all-MiniLM-L6-v2
tasks:
  encode:
    dense:
      dim: 384
profiles:
  default:
    adapter_path: sie_server.adapters.sentence_transformer:Adapter
    max_batch_tokens: 4096
""",
            )
        )

        assert resp.applied is True
        assert resp.bundle_config_hash
        assert resp.config_version == 1

    @pytest.mark.asyncio
    async def test_apply_model_config_adds_registry_config_and_returns_hash(self) -> None:
        registry = ModelRegistry(models_dir=None, model_filter=["existing/model"])
        executor = QueueExecutor(registry)
        sock = _short_sock_path()
        srv = IpcServer(sock, executor, worker_id="worker-test", stale_after_ms=10_000)
        await srv.start()
        try:
            client = await _Client.connect(sock)
            try:
                resp = await client.rpc(
                    "ApplyModelConfig",
                    {
                        "bundle_id": "default",
                        "model_id": "test/model",
                        "epoch": 7,
                        "bundle_config_hash": "from-sie-config",
                        "profiles_added": ["default"],
                        "model_config": """
sie_id: test/model
hf_id: sentence-transformers/all-MiniLM-L6-v2
tasks:
  encode:
    dense:
      dim: 384
profiles:
  default:
    adapter_path: sie_server.adapters.sentence_transformer:Adapter
    max_batch_tokens: 4096
""",
                    },
                )
                assert resp["ok"] is True
                assert resp["body"]["applied"] is True
                assert resp["body"]["bundle_config_hash"]
                assert resp["body"]["config_version"] == 1
                assert registry.get_config("test/model").sie_id == "test/model"
            finally:
                await client.close()
        finally:
            await srv.stop(drain_timeout_s=1.0)

    @pytest.mark.asyncio
    async def test_apply_model_config_rejects_model_id_mismatch(self) -> None:
        registry = ModelRegistry(models_dir=None)
        executor = QueueExecutor(registry)
        sock = _short_sock_path()
        srv = IpcServer(sock, executor, worker_id="worker-test", stale_after_ms=10_000)
        await srv.start()
        try:
            client = await _Client.connect(sock)
            try:
                resp = await client.rpc(
                    "ApplyModelConfig",
                    {
                        "bundle_id": "default",
                        "model_id": "notification/model",
                        "epoch": 7,
                        "bundle_config_hash": "from-sie-config",
                        "profiles_added": ["default"],
                        "model_config": """
sie_id: yaml/model
hf_id: sentence-transformers/all-MiniLM-L6-v2
tasks:
  encode:
    dense:
      dim: 384
profiles:
  default:
    adapter_path: sie_server.adapters.sentence_transformer:Adapter
    max_batch_tokens: 4096
""",
                    },
                )
                assert resp["ok"] is False
                assert "model_id mismatch" in (resp["error"] or "")
            finally:
                await client.close()
        finally:
            await srv.stop(drain_timeout_s=1.0)

    @pytest.mark.asyncio
    async def test_replace_model_configs_requires_models_field(self) -> None:
        registry = ModelRegistry(models_dir=None)
        executor = QueueExecutor(registry)

        model_config = """
sie_id: kept/model
hf_id: sentence-transformers/all-MiniLM-L6-v2
tasks:
  encode:
    dense:
      dim: 384
profiles:
  default:
    adapter_path: sie_server.adapters.sentence_transformer:Adapter
    max_batch_tokens: 4096
"""
        await executor.replace_model_configs(
            ReplaceModelConfigsRequest(
                bundle_id="default",
                epoch=7,
                bundle_config_hash="",
                models=[
                    ReplaceModelConfigEntry(
                        model_id="kept/model",
                        model_config=model_config,
                    )
                ],
            )
        )
        assert registry.has_model("kept/model")

        sock = _short_sock_path()
        srv = IpcServer(sock, executor, worker_id="worker-test", stale_after_ms=10_000)
        await srv.start()
        try:
            client = await _Client.connect(sock)
            try:
                resp = await client.rpc(
                    "ReplaceModelConfigs",
                    {
                        "bundle_id": "default",
                        "epoch": 8,
                        "bundle_config_hash": "",
                    },
                )
                assert resp["ok"] is False
                assert "Object missing required field `models`" in (resp["error"] or "")
                assert registry.has_model("kept/model")
            finally:
                await client.close()
        finally:
            await srv.stop(drain_timeout_s=1.0)

    @pytest.mark.asyncio
    async def test_replace_model_configs_drops_removed_registry_config(self) -> None:
        registry = ModelRegistry(models_dir=None, model_filter=["existing/model"])
        executor = QueueExecutor(registry)

        def model_yaml(model_id: str) -> str:
            return f"""
sie_id: {model_id}
hf_id: sentence-transformers/all-MiniLM-L6-v2
tasks:
  encode:
    dense:
      dim: 384
profiles:
  default:
    adapter_path: sie_server.adapters.sentence_transformer:Adapter
    max_batch_tokens: 4096
"""

        await executor.replace_model_configs(
            ReplaceModelConfigsRequest(
                bundle_id="default",
                epoch=7,
                bundle_config_hash="",
                models=[
                    ReplaceModelConfigEntry(
                        model_id="stale/model",
                        model_config=model_yaml("stale/model"),
                    ),
                    ReplaceModelConfigEntry(
                        model_id="kept/model",
                        model_config=model_yaml("kept/model"),
                    ),
                ],
            )
        )
        assert registry.has_model("stale/model")
        assert registry.has_model("kept/model")
        registry._loaded["stale/model"] = MagicMock()
        registry._do_unload = AsyncMock()

        resp = await executor.replace_model_configs(
            ReplaceModelConfigsRequest(
                bundle_id="default",
                epoch=8,
                bundle_config_hash="",
                models=[
                    ReplaceModelConfigEntry(
                        model_id="kept/model",
                        model_config=model_yaml("kept/model"),
                    )
                ],
            )
        )

        assert resp.applied is True
        assert resp.applied_models == ["kept/model"]
        registry._do_unload.assert_awaited_once_with("stale/model")
        assert not registry.has_model("stale/model")
        assert registry.has_model("kept/model")

    @pytest.mark.asyncio
    async def test_replace_model_configs_keeps_loaded_model_for_identical_config(self) -> None:
        registry = ModelRegistry(models_dir=None)
        executor = QueueExecutor(registry)

        def model_yaml(model_id: str) -> str:
            return f"""
sie_id: {model_id}
hf_id: sentence-transformers/all-MiniLM-L6-v2
tasks:
  encode:
    dense:
      dim: 384
profiles:
  default:
    adapter_path: sie_server.adapters.sentence_transformer:Adapter
    max_batch_tokens: 4096
"""

        await executor.replace_model_configs(
            ReplaceModelConfigsRequest(
                bundle_id="default",
                epoch=7,
                bundle_config_hash="",
                models=[
                    ReplaceModelConfigEntry(
                        model_id="kept/model",
                        model_config=model_yaml("kept/model"),
                    )
                ],
            )
        )
        config_version = registry._config_version
        registry._loaded["kept/model"] = MagicMock()
        registry._do_unload = AsyncMock()

        resp = await executor.replace_model_configs(
            ReplaceModelConfigsRequest(
                bundle_id="default",
                epoch=8,
                bundle_config_hash="",
                models=[
                    ReplaceModelConfigEntry(
                        model_id="kept/model",
                        model_config=model_yaml("kept/model"),
                    )
                ],
            )
        )

        assert resp.applied is True
        assert resp.applied_models == ["kept/model"]
        registry._do_unload.assert_not_awaited()
        assert registry._config_version == config_version
        assert registry.has_model("kept/model")

    @pytest.mark.asyncio
    async def test_replace_model_configs_unloads_loaded_model_for_changed_config(self) -> None:
        registry = ModelRegistry(models_dir=None)
        executor = QueueExecutor(registry)

        def model_yaml(model_id: str, *, max_batch_tokens: int) -> str:
            return f"""
sie_id: {model_id}
hf_id: sentence-transformers/all-MiniLM-L6-v2
tasks:
  encode:
    dense:
      dim: 384
profiles:
  default:
    adapter_path: sie_server.adapters.sentence_transformer:Adapter
    max_batch_tokens: {max_batch_tokens}
"""

        await executor.replace_model_configs(
            ReplaceModelConfigsRequest(
                bundle_id="default",
                epoch=7,
                bundle_config_hash="",
                models=[
                    ReplaceModelConfigEntry(
                        model_id="kept/model",
                        model_config=model_yaml("kept/model", max_batch_tokens=4096),
                    )
                ],
            )
        )
        registry._loaded["kept/model"] = MagicMock()
        registry._do_unload = AsyncMock()

        resp = await executor.replace_model_configs(
            ReplaceModelConfigsRequest(
                bundle_id="default",
                epoch=8,
                bundle_config_hash="",
                models=[
                    ReplaceModelConfigEntry(
                        model_id="kept/model",
                        model_config=model_yaml("kept/model", max_batch_tokens=8192),
                    )
                ],
            )
        )

        assert resp.applied is True
        assert resp.applied_models == ["kept/model"]
        registry._do_unload.assert_awaited_once_with("kept/model")
        assert registry.get_config("kept/model").profiles["default"].max_batch_tokens == 8192

    @pytest.mark.asyncio
    async def test_replace_model_configs_reports_pool_filtered_models(self) -> None:
        registry = ModelRegistry(models_dir=None, pool_name="customer-a")
        executor = QueueExecutor(registry)

        def model_yaml(model_id: str, *, pool: str | None = None) -> str:
            pool_line = f"pool: {pool}\n" if pool is not None else ""
            return f"""
sie_id: {model_id}
hf_id: sentence-transformers/all-MiniLM-L6-v2
{pool_line}tasks:
  encode:
    dense:
      dim: 384
profiles:
  default:
    adapter_path: sie_server.adapters.sentence_transformer:Adapter
    max_batch_tokens: 4096
"""

        resp = await executor.replace_model_configs(
            ReplaceModelConfigsRequest(
                bundle_id="default",
                epoch=8,
                bundle_config_hash="",
                models=[
                    ReplaceModelConfigEntry(
                        model_id="default/model",
                        model_config=model_yaml("default/model"),
                    ),
                    ReplaceModelConfigEntry(
                        model_id="tenant/model",
                        model_config=model_yaml("tenant/model", pool="customer-a"),
                    ),
                ],
            )
        )

        assert resp.applied is True
        assert resp.applied_models == ["tenant/model"]
        assert not registry.has_model("default/model")
        assert registry.has_model("tenant/model")


# -----------------------------------------------------------------------------
# ProcessEncodeBatch
# -----------------------------------------------------------------------------


class TestProcessEncodeBatch:
    @pytest.mark.asyncio
    async def test_encode_batch_roundtrip(self, server_and_path) -> None:
        _srv, sock = server_and_path
        client = await _Client.connect(sock)
        try:
            fake_outputs = [{"dense": [0.25, 0.75]}]
            with patch(
                "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
                new_callable=AsyncMock,
                return_value=(fake_outputs, RequestTiming()),
            ):
                resp = await client.rpc(
                    "ProcessEncodeBatch",
                    {
                        "model_id": "test/model",
                        "items": [
                            {
                                "work_item_id": "req-1.0",
                                "request_id": "req-1",
                                "item_index": 0,
                                "total_items": 1,
                                "timestamp": time.time(),
                                "item": {"text": "hello"},
                                "output_types": ["dense"],
                                "is_query": False,
                                "payload_fetch_ms": 0.0,
                            }
                        ],
                    },
                )

            assert resp["ok"] is True
            outcomes = resp["body"]["outcomes"]
            assert len(outcomes) == 1
            outcome = outcomes[0]
            assert outcome["work_item_id"] == "req-1.0"
            assert outcome["disposition"] == "publish_and_ack"
            inner = msgpack.unpackb(outcome["result_msgpack"], raw=False)
            assert inner == {"dense": [0.25, 0.75]}
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_encode_batch_during_drain_rejected(self, server_and_path) -> None:
        srv, sock = server_and_path
        client = await _Client.connect(sock)
        try:
            drain_resp = await client.rpc("Drain", {"deadline_ms": 1000})
            assert drain_resp["ok"] is True
            assert srv._drain_event.is_set()

            resp = await client.rpc(
                "ProcessEncodeBatch",
                {
                    "model_id": "test/model",
                    "items": [
                        {
                            "work_item_id": "req-2.0",
                            "request_id": "req-2",
                            "item_index": 0,
                            "total_items": 1,
                            "timestamp": time.time(),
                            "item": {"text": "x"},
                            "output_types": ["dense"],
                            "is_query": False,
                            "payload_fetch_ms": 0.0,
                        }
                    ],
                },
            )
            assert resp["ok"] is False
            assert resp["error"] == "draining"
        finally:
            await client.close()


# -----------------------------------------------------------------------------
# ProcessScoreBatch / ProcessExtractBatch smoke
# -----------------------------------------------------------------------------


class TestProcessScoreAndExtract:
    @pytest.mark.asyncio
    async def test_score_batch_roundtrip(self, server_and_path) -> None:
        srv, sock = server_and_path
        worker = AsyncMock()
        score_output = ScoreOutput(scores=np.array([0.9, 0.1], dtype=np.float32))
        wr = WorkerResult(output=score_output, timing=RequestTiming())
        fut: asyncio.Future[WorkerResult] = asyncio.Future()
        fut.set_result(wr)
        worker.submit_score = AsyncMock(return_value=fut)
        srv._executor._registry.start_worker = AsyncMock(return_value=worker)

        client = await _Client.connect(sock)
        try:
            resp = await client.rpc(
                "ProcessScoreBatch",
                {
                    "model_id": "test/model",
                    "items": [
                        {
                            "work_item_id": "req-s.0",
                            "request_id": "req-s",
                            "item_index": 0,
                            "total_items": 1,
                            "timestamp": time.time(),
                            "query_item": {"text": "q"},
                            "score_items": [
                                {"text": "a", "id": "doc-a"},
                                {"text": "b", "id": "doc-b"},
                            ],
                            "payload_fetch_ms": 0.0,
                        }
                    ],
                },
            )
            assert resp["ok"] is True
            outcomes = resp["body"]["outcomes"]
            assert outcomes[0]["disposition"] == "publish_and_ack"
            # Score output is Rust-frameable. The legacy ``result_msgpack``
            # is None; the parallel arrays land on ``raw_output.score`` and
            # the Rust publisher does the descending stable sort + rank
            # assignment.
            assert outcomes[0]["result_msgpack"] is None
            score = outcomes[0]["raw_output"]["score"]
            assert score["item_ids"] == ["doc-a", "doc-b"]
            assert score["scores"] == pytest.approx([0.9, 0.1])
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_extract_batch_roundtrip(self, server_and_path) -> None:
        srv, sock = server_and_path
        worker = AsyncMock()
        extract_output = ExtractOutput(
            entities=[[{"text": "Alice", "label": "person", "score": 0.99, "start": 0, "end": 5}]]
        )
        wr = WorkerResult(output=extract_output, timing=RequestTiming())
        fut: asyncio.Future[WorkerResult] = asyncio.Future()
        fut.set_result(wr)
        worker.submit_extract = AsyncMock(return_value=fut)
        srv._executor._registry.start_worker = AsyncMock(return_value=worker)

        client = await _Client.connect(sock)
        try:
            resp = await client.rpc(
                "ProcessExtractBatch",
                {
                    "model_id": "test/model",
                    "items": [
                        {
                            "work_item_id": "req-e.0",
                            "request_id": "req-e",
                            "item_index": 0,
                            "total_items": 1,
                            "timestamp": time.time(),
                            "item": {"text": "Alice works at Acme."},
                            "labels": ["person"],
                            "payload_fetch_ms": 0.0,
                        }
                    ],
                },
            )
            assert resp["ok"] is True
            outcomes = resp["body"]["outcomes"]
            assert outcomes[0]["disposition"] == "publish_and_ack"
            inner = msgpack.unpackb(outcomes[0]["result_msgpack"], raw=False)
            assert "entities" in inner
        finally:
            await client.close()


# -----------------------------------------------------------------------------
# Heartbeat / readiness
# -----------------------------------------------------------------------------


class TestHeartbeat:
    @pytest.mark.asyncio
    async def test_heartbeat_stale_when_never_pinged(self) -> None:
        executor, _reg = _make_executor()
        srv = IpcServer(_short_sock_path(), executor, worker_id="w", stale_after_ms=100)
        assert srv.is_heartbeat_fresh() is False

    @pytest.mark.asyncio
    async def test_heartbeat_expires(self) -> None:
        executor, _reg = _make_executor()
        sock = _short_sock_path()
        async with IpcServer(sock, executor, worker_id="w", stale_after_ms=50) as srv:
            client = await _Client.connect(sock)
            try:
                await client.rpc("Ping", {"timestamp_ms": 0.0})
                assert srv.is_heartbeat_fresh() is True
                # Simulate stale heartbeat by asking about a future monotonic time.
                future = (srv.last_ping_monotonic or 0.0) + 10.0
                assert srv.is_heartbeat_fresh(now_monotonic=future) is False
            finally:
                await client.close()


# -----------------------------------------------------------------------------
# Generation sidecar IPC
# -----------------------------------------------------------------------------


class TestGenerationSidecarIpc:
    @pytest.mark.asyncio
    async def test_worker_capabilities_reports_generation_models(self) -> None:
        executor, reg = _make_executor()
        gen_cfg = MagicMock()
        gen_cfg.tasks.generate = object()
        encode_cfg = MagicMock()
        encode_cfg.tasks.generate = None
        reg.get_configs_snapshot.return_value = {
            "encode/model": encode_cfg,
            "z-generate/model": gen_cfg,
        }
        sock = _short_sock_path()
        async with IpcServer(sock, executor, worker_id="w"):
            client = await _Client.connect(sock)
            try:
                resp = await client.rpc("WorkerCapabilities", {})
            finally:
                await client.close()

        assert resp["ok"] is True
        assert resp["body"] == {
            "has_generation_models": True,
            "generation_models": ["z-generate/model"],
        }

    @pytest.mark.asyncio
    async def test_signal_generate_cancel_forwards_to_streaming_processor(self) -> None:
        executor, _reg = _make_executor()
        processor = MagicMock()
        processor.signal_cancel.return_value = True
        sock = _short_sock_path()
        async with IpcServer(sock, executor, worker_id="w") as srv:
            srv._get_streaming_processor = AsyncMock(return_value=processor)  # type: ignore[method-assign]
            client = await _Client.connect(sock)
            try:
                resp = await client.rpc("SignalGenerateCancel", {"request_id": "req-123"})
            finally:
                await client.close()

        assert resp["ok"] is True
        assert resp["body"] == {"matched": True}
        processor.signal_cancel.assert_called_once_with("req-123")


# -----------------------------------------------------------------------------
# Socket cleanup
# -----------------------------------------------------------------------------


class TestSocketLifecycle:
    @pytest.mark.asyncio
    async def test_stale_socket_is_unlinked_on_start(self) -> None:
        sock = _short_sock_path()
        sock.touch()
        assert sock.exists()
        executor, _reg = _make_executor()
        async with IpcServer(sock, executor, worker_id="w"):
            # Socket should exist but be of type socket now.
            client = await _Client.connect(sock)
            try:
                resp = await client.rpc("Ping", {"timestamp_ms": 0.0})
                assert resp["ok"] is True
            finally:
                await client.close()

    @pytest.mark.asyncio
    async def test_socket_unlinked_on_stop(self) -> None:
        sock = _short_sock_path()
        executor, _reg = _make_executor()
        srv = IpcServer(sock, executor, worker_id="w")
        await srv.start()
        assert sock.exists()
        await srv.stop(drain_timeout_s=0.5)
        assert not sock.exists()


# -----------------------------------------------------------------------------
# Drain deadline propagation
# -----------------------------------------------------------------------------


class TestDrainDeadline:
    @pytest.mark.asyncio
    async def test_deadline_ms_sets_drain_timeout(self, server_and_path) -> None:
        srv, sock = server_and_path
        client = await _Client.connect(sock)
        try:
            resp = await client.rpc("Drain", {"deadline_ms": 2500})
            assert resp["ok"] is True
            assert resp["body"]["acknowledged"] is True
            assert srv._drain_deadline_s == pytest.approx(2.5)
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_missing_deadline_leaves_drain_unset(self, server_and_path) -> None:
        srv, sock = server_and_path
        client = await _Client.connect(sock)
        try:
            resp = await client.rpc("Drain", {})
            assert resp["ok"] is True
            assert srv._drain_deadline_s is None
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_non_positive_deadline_is_ignored(self, server_and_path) -> None:
        srv, sock = server_and_path
        client = await _Client.connect(sock)
        try:
            for bad in (0, -1, -1000.5):
                resp = await client.rpc("Drain", {"deadline_ms": bad})
                assert resp["ok"] is True
            assert srv._drain_deadline_s is None
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_bool_deadline_is_ignored(self, server_and_path) -> None:
        """`bool` is a subclass of `int` in Python, so `deadline_ms=True`
        would otherwise set a 1ms timeout and break shutdowns.
        """
        srv, sock = server_and_path
        client = await _Client.connect(sock)
        try:
            resp = await client.rpc("Drain", {"deadline_ms": True})
            assert resp["ok"] is True
            assert srv._drain_deadline_s is None
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_repeated_drain_tightens_deadline(self, server_and_path) -> None:
        """Second Drain with a smaller deadline must win; a later drain
        with a LARGER deadline must not loosen the stored budget — Rust
        may retry a Drain RPC and we can't let a retransmit extend the
        SIGTERM countdown.
        """
        srv, sock = server_and_path
        client = await _Client.connect(sock)
        try:
            await client.rpc("Drain", {"deadline_ms": 5000}, request_id="a")
            assert srv._drain_deadline_s == pytest.approx(5.0)
            await client.rpc("Drain", {"deadline_ms": 1000}, request_id="b")
            assert srv._drain_deadline_s == pytest.approx(1.0)
            # Loose retry must not widen the window.
            await client.rpc("Drain", {"deadline_ms": 10_000}, request_id="c")
            assert srv._drain_deadline_s == pytest.approx(1.0)
        finally:
            await client.close()


# -----------------------------------------------------------------------------
# Silence unused-import warning for IpcServerError; kept exported for future tests.
# -----------------------------------------------------------------------------

_ = IpcServerError
