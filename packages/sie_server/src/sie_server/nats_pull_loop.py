from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import os
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Mapping
from pathlib import Path
from typing import Any

import msgpack
import msgpack_numpy
import nats
import nats.errors
import nats.js.errors
from nats.aio.client import Client as NATSClient
from nats.js import JetStreamContext
from nats.js.api import (
    AckPolicy,
    ConsumerConfig,
    DiscardPolicy,
    RetentionPolicy,
    StorageType,
    StreamConfig,
)
from sie_sdk import SIEAsyncClient, SIEError
from sie_sdk.queue_types import (
    WORK_SUBJECT_PREFIX,
    WorkItem,
    WorkResult,
    denormalize_model_id,
    normalize_worker_id,
    work_consumer_name,
    work_pool_stream_name,
    work_pool_stream_subjects,
    work_stream_name,
    work_worker_consumer_name,
    work_worker_stream_name,
    work_worker_stream_subjects,
)

from sie_server.core.adaptive_batching import LatencyTracker
from sie_server.core.extract_cost import build_extract_prepared_items
from sie_server.core.inference_output import ScoreOutput
from sie_server.core.oom import is_oom_error
from sie_server.core.prepared import ScorePreparedItem
from sie_server.core.registry import ModelRegistry
from sie_server.core.timing import RequestTiming
from sie_server.core.worker.handlers.extract import ExtractHandler
from sie_server.processors.work_class_scheduler import (
    EMBEDDING_CLASS,
    GENERATION_CLASS,
    FairnessConfig,
    WorkClassConfig,
    WorkClassScheduler,
    classify_work_class,
)
from sie_server.types.inputs import InvalidMediaError, Item
from sie_server.types.responses import ErrorCode

msgpack_numpy.patch()

logger = logging.getLogger(__name__)

# Default ACK wait in seconds — nats.py converts to nanoseconds internally.
_ACK_WAIT_S = 30

# Streaming-generation ack_wait: long generations would exceed the 30s
# default. Use 5 minutes for pools that host any generation model; the
# worker emits ``msg.in_progress()`` between chunks to refresh the timer
# for genuinely slow decodes.
_ACK_WAIT_S_GENERATION = 300

# Default batch budget — controls how many NATS messages a single fetch()
# returns.  This is the primary driver of GPU batch size in queue mode:
# the pull loop groups fetched messages by model and dispatches them as a
# single GPU forward pass.  Configurable via SIE_NATS_FETCH_BUDGET.
_DEFAULT_BATCH_BUDGET = int(os.environ.get("SIE_NATS_FETCH_BUDGET", "64"))

# Maximum concurrent batch-processing tasks to avoid ACK timeout storms.
_MAX_CONCURRENT_BATCHES = 4

# TTL (seconds) for caching the bundle config hash.
_CONFIG_HASH_CACHE_TTL_S = 5.0

# Interval (seconds) between retries for models whose initial subscription failed.
_FAILED_MODEL_RETRY_INTERVAL = 30.0

# Pool admission defaults. When SIE_GATEWAY_URL is unset the gate is disabled
# for local/single-worker compatibility.
_POOL_ADMISSION_CHECK_INTERVAL_S = float(os.environ.get("SIE_POOL_ADMISSION_CHECK_INTERVAL_S", "5.0"))
_POOL_ADMISSION_PAUSE_S = float(os.environ.get("SIE_POOL_ADMISSION_PAUSE_S", "1.0"))
_POOL_ADMISSION_STALE_AFTER_S = float(os.environ.get("SIE_POOL_ADMISSION_STALE_AFTER_S", "30.0"))

# Graceful drain timeout (seconds) — matches _ACK_WAIT_S so in-flight batches
# have time to finish and ACK before the pull loop shuts down.
_DRAIN_TIMEOUT_S = 30.0

# NAK delay (seconds) for items targeting unloaded models.
# JetStream redelivers after this delay, giving time for loading to complete.
# Configurable via SIE_NAK_DELAY_S; with max_deliver=20, the default gives
# ~100s total retry budget — enough for cold model downloads.
_NAK_DELAY_S = float(os.environ.get("SIE_NAK_DELAY_S", "5.0"))

# NAK delay for items that hit OOM (RESOURCE_EXHAUSTED) on this worker.
# Slightly longer than the default to give the OOMed worker time to drain
# its in-flight requests (and for any sibling worker to pick up the NAKed
# item via JetStream redelivery). The delay is intentionally NOT zero —
# instant redelivery would just reroute the same item back to the still-
# pressured worker. Tunable via SIE_OOM_NAK_DELAY_S.
_OOM_NAK_DELAY_S = float(os.environ.get("SIE_OOM_NAK_DELAY_S", "10.0"))

# Maximum delivery attempts before a message is sent to the DLQ.
# Configurable via SIE_MAX_DELIVER; must be high enough to cover model load
# times (10-60s+ for cold HuggingFace downloads).
_MAX_DELIVER = int(os.environ.get("SIE_MAX_DELIVER", "20"))

_MIN_SUBJECT_PARTS = 4  # sie.work.{model}.{pool}

# Dynamic fetch timeout bounds.  With the pool-level stream design (single
# subscription per pool), idle polling cost is just one NATS fetch RPC per
# cycle — much lower than the old O(N-models) design.  The max can be kept
# very tight to minimise head-of-line delay at low concurrency.
#
# At 1ms minimum, the worker checks for new messages every 1ms when busy.
# At 20ms maximum, the worst-case idle delay is 20ms (avg ~10ms).
# This translates to ~50 idle polls/sec to NATS — negligible load.
_MIN_FETCH_TIMEOUT_S = 0.001  # 1ms  — near-instant when messages flow
_MAX_FETCH_TIMEOUT_S = 0.02  # 20ms — tight idle delay
_BACKOFF_GROWTH = 2.0  # 1ms → 2ms → 4ms → 8ms → 16ms → 20ms (5 steps)

# Thrashing detection: if a model is background-loaded this many times
# within this window, log a warning suggesting separate bundles.
_THRASH_WINDOW_S = 300.0  # 5 minutes
_THRASH_THRESHOLD = 4  # 4 loads in 5 minutes = thrashing

# Max age (seconds) for messages in the pool-level stream.
# Configurable via SIE_STREAM_MAX_AGE_S to give headroom for slow model loads.
_DEFAULT_STREAM_MAX_AGE_S = int(os.environ.get("SIE_STREAM_MAX_AGE_S", "120"))

# Thread pool for S3 payload fetches — avoids exhausting the default
# asyncio thread pool when fetching many large payloads concurrently.
_PAYLOAD_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=16, thread_name_prefix="payload-fetch")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _normalize_pool_name(name: str | None) -> str:
    value = (name or "").strip().lower()
    if value in {"", "_default"}:
        return "default"
    return value


def _first_csv_token(value: str | None) -> str | None:
    if not value:
        return None
    for token in value.split(","):
        token = token.strip()
        if token:
            return token
    return None


def _lookup_case_insensitive(mapping: Mapping[str, Any], key: str) -> Any:
    key_lower = key.lower()
    for candidate, value in mapping.items():
        if candidate.lower() == key_lower:
            return value
    return None


class _PoolAdmissionGate:
    """Decides whether this worker may pull from its configured queue pool."""

    def __init__(
        self,
        *,
        pool_name: str,
        worker_id: str,
        machine_profile: str,
        gateway_url: str | None,
        api_key: str | None,
        check_interval_s: float = _POOL_ADMISSION_CHECK_INTERVAL_S,
        pause_s: float = _POOL_ADMISSION_PAUSE_S,
        stale_after_s: float = _POOL_ADMISSION_STALE_AFTER_S,
        client: Any | None = None,
    ) -> None:
        self._pool_name = pool_name
        self._worker_id = worker_id
        self._machine_profile = machine_profile
        self._gateway_url = (gateway_url or "").strip()
        self._check_interval_s = check_interval_s
        self.pause_s = pause_s
        self._stale_after_s = stale_after_s
        self._client = client
        self._owns_client = client is None
        self._enabled = bool(self._gateway_url)
        self._last_check_at = 0.0
        self._last_success_at = 0.0
        self._admitted = _normalize_pool_name(pool_name) == "default"
        self._last_reason = "initial"

        if self._enabled and self._client is None:
            self._client = SIEAsyncClient(self._gateway_url, timeout_s=5.0, api_key=api_key or None)

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.close()

    async def admitted(self) -> bool:
        if not self._enabled or self._client is None:
            return True

        now = time.monotonic()
        if now - self._last_check_at < self._check_interval_s:
            return self._admitted

        self._last_check_at = now
        try:
            pool = await self._client.get_pool(self._pool_name)
        except SIEError as exc:
            return self._handle_error(now, str(exc))
        except Exception as exc:  # noqa: BLE001
            return self._handle_error(now, str(exc))

        self._last_success_at = now
        admitted, reason = self._decide(pool)
        self._set_admitted(admitted, reason)
        return self._admitted

    def _handle_error(self, now: float, error: str) -> bool:
        if self._last_success_at and now - self._last_success_at <= self._stale_after_s:
            return self._admitted

        # Keep the default pool available during gateway/status glitches. Named
        # pools fail closed because their caps are isolation contracts.
        fail_open = _normalize_pool_name(self._pool_name) == "default"
        self._set_admitted(fail_open, f"status_error:{error[:120]}")
        return self._admitted

    def _decide(self, pool: Mapping[str, Any] | None) -> tuple[bool, str]:
        if pool is None:
            is_default = _normalize_pool_name(self._pool_name) == "default"
            return is_default, "pool_missing"

        spec_raw = pool.get("spec", {})
        spec = spec_raw if isinstance(spec_raw, Mapping) else {}
        caps = spec.get("gpu_caps") or {}
        if not isinstance(caps, dict) or not caps:
            return True, "uncapped"

        cap = _lookup_case_insensitive(caps, self._machine_profile) if self._machine_profile else None
        if cap is None:
            return True, "profile_uncapped"
        try:
            if int(cap) <= 0:
                return False, "cap_exhausted"
        except (TypeError, ValueError):
            return False, "malformed_gpu_cap"

        status_raw = pool.get("status", {})
        status = status_raw if isinstance(status_raw, Mapping) else {}
        assigned = status.get("assigned_workers", [])
        if not isinstance(assigned, list):
            return False, "malformed_assigned_workers"

        for worker in assigned:
            if isinstance(worker, dict) and worker.get("name") == self._worker_id:
                return True, "assigned"
        return False, "not_assigned"

    def _set_admitted(self, admitted: bool, reason: str) -> None:
        if admitted == self._admitted and reason == self._last_reason:
            return
        self._admitted = admitted
        self._last_reason = reason
        logger.info(
            "Pool admission %s (pool=%s, worker=%s, machine_profile=%s, reason=%s)",
            "granted" if admitted else "paused",
            self._pool_name,
            self._worker_id,
            self._machine_profile,
            reason,
        )


# Map numpy dtype to the wire-format dtype string expected by the SDK.
_NP_DTYPE_MAP = {"float32": "float32", "float16": "float16", "int8": "int8", "uint8": "binary"}


def _wrap_encode_output(output: dict, config: Any) -> dict:
    """Wrap raw numpy arrays from ``EncodeHandler.format_output`` into the
    ``DenseVector``/``SparseVector`` wire format that the SDK expects.

    The HTTP path does this via ``encode.py:_build_response_items`` + Pydantic
    ``EncodeResult``.  In the queue path the worker must produce the same shape
    *before* msgpack-serializing, because the gateway embeds the blob as-is.
    """
    import numpy as np  # noqa: PLC0415

    wrapped = dict(output)

    if "dense" in wrapped and isinstance(wrapped["dense"], np.ndarray):
        arr = wrapped["dense"]
        encode_task = getattr(config, "tasks", None)
        encode_task = getattr(encode_task, "encode", None)
        dense_cfg = getattr(encode_task, "dense", None) if encode_task else None
        dense_dim = dense_cfg.dim if dense_cfg else None

        is_binary = arr.dtype == np.uint8 and dense_dim and arr.shape[0] < dense_dim
        dims = dense_dim if dense_dim is not None else arr.shape[0]
        dtype = "binary" if is_binary else _NP_DTYPE_MAP.get(str(arr.dtype), "float32")

        wrapped["dense"] = {"dims": int(dims), "dtype": dtype, "values": arr}

    if "sparse" in wrapped and isinstance(wrapped["sparse"], dict):
        # sparse already comes as {"indices": ndarray, "values": ndarray}
        pass

    if "multivector" in wrapped and isinstance(wrapped["multivector"], np.ndarray):
        arr = wrapped["multivector"]
        wrapped["multivector"] = {"values": arr}

    return wrapped


# ---------------------------------------------------------------------------
# Metrics (optional — gracefully degrade if prometheus_client not installed)
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Histogram

    PULL_ITEMS_FETCHED = Histogram(
        "sie_pull_loop_items_fetched",
        "Number of items fetched per pull cycle",
        ["model"],
        buckets=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    )
    PULL_BATCH_PROCESS_SECONDS = Histogram(
        "sie_pull_loop_batch_process_seconds",
        "Time to process a pulled batch (seconds)",
        ["model", "operation"],
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )
    PULL_QUEUE_WAIT_SECONDS = Histogram(
        "sie_pull_loop_queue_wait_seconds",
        "Time items waited in NATS queue before being pulled (seconds)",
        ["model"],
        buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    )
    PULL_CONFIG_HASH_MISMATCHES = Counter(
        "sie_pull_loop_config_hash_mismatches_total",
        "Config hash mismatches detected (log-only, items still processed)",
        ["model"],
    )
    PULL_NAK_UNLOADED = Counter(
        "sie_pull_loop_nak_unloaded_total",
        "Work items NAKed because the target model is not loaded",
        ["model"],
    )
    PULL_MODEL_LOADS = Counter(
        "sie_pull_loop_model_loads_total",
        "Background model loads triggered by demand",
        ["model"],
    )
    PULL_GENERATE_DISPATCH_ERRORS = Counter(
        "sie_pull_loop_generate_dispatch_errors_total",
        "Unhandled exceptions escaping StreamingProcessor.process (message NAK'd)",
        ["model"],
    )
    _HAS_METRICS = True
except ImportError:
    _HAS_METRICS = False


# ---------------------------------------------------------------------------
# Minimal payload store (read-only) — self-contained to avoid cross-package
# imports from the config service or gateway.
# ---------------------------------------------------------------------------


class _PayloadStore:
    """Read-only payload fetcher (local filesystem or S3)."""

    async def get(self, key: str) -> bytes:
        raise NotImplementedError


class _LocalPayloadStore(_PayloadStore):
    def __init__(self, base_dir: str) -> None:
        self._base_dir = Path(base_dir)

    def _safe_path(self, key: str) -> Path:
        """Resolve key to a path inside base_dir, rejecting traversal."""
        target = (self._base_dir / key).resolve()
        base = self._base_dir.resolve()
        try:
            target.relative_to(base)
        except ValueError:
            raise ValueError(f"Path traversal detected: {key}") from None
        return target

    async def get(self, key: str) -> bytes:
        path = self._safe_path(key)
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(_PAYLOAD_THREAD_POOL, path.read_bytes)
        except FileNotFoundError:
            raise KeyError(f"Payload not found: {key}") from None


class _S3PayloadStore(_PayloadStore):
    def __init__(self, bucket: str, prefix: str = "payloads") -> None:
        self._bucket = bucket
        self._prefix = prefix
        self._client: object | None = None
        self._client_lock = threading.Lock()

    def _get_client(self) -> object:
        """Get or create a cached boto3 S3 client (thread-safe)."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    import boto3  # noqa: PLC0415

                    self._client = boto3.client("s3")
        return self._client

    async def get(self, key: str) -> bytes:
        full_key = f"{self._prefix}/{key}" if self._prefix else key

        def _fetch() -> bytes:
            client = self._get_client()
            response = client.get_object(Bucket=self._bucket, Key=full_key)  # type: ignore
            body = response["Body"]
            try:
                return body.read()
            finally:
                body.close()

        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(_PAYLOAD_THREAD_POOL, _fetch)
        except KeyError:
            raise
        except Exception as e:
            import botocore.exceptions  # noqa: PLC0415 — optional dep, lazy import

            if isinstance(e, botocore.exceptions.ClientError):
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code in ("NoSuchKey", "404"):
                    raise KeyError(f"Payload not found: {key}") from e
            raise


class _GCSPayloadStore(_PayloadStore):
    """Read-only GCS payload store for the worker pull loop."""

    def __init__(self, bucket: str, prefix: str) -> None:
        self._bucket_name = bucket
        self._prefix = prefix
        self._client: Any = None

    def _get_bucket(self) -> Any:
        if self._client is None:
            try:
                from google.cloud import storage  # noqa: PLC0415
            except ImportError:
                raise ImportError(
                    "google-cloud-storage is required for GCS payload stores. Install the google-cloud-storage package"
                ) from None
            client = storage.Client()
            self._client = client.bucket(self._bucket_name)
        return self._client

    async def get(self, key: str) -> bytes:
        bucket = self._get_bucket()
        full_key = f"{self._prefix}/{key}" if self._prefix else key
        blob = bucket.blob(full_key)
        try:
            data: bytes = await asyncio.to_thread(blob.download_as_bytes)
            return data
        except Exception as e:
            _not_found = None
            with contextlib.suppress(ImportError):
                from google.api_core.exceptions import NotFound  # noqa: PLC0415

                _not_found = NotFound
            if _not_found is not None and isinstance(e, _not_found):
                raise KeyError(f"Payload not found: {key}") from e
            raise


def _create_payload_store(url: str | None) -> _PayloadStore | None:
    """Create a read-only payload store from a URL."""
    if not url:
        return None
    if url.startswith("s3://"):
        parts = url[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else "payloads"
        return _S3PayloadStore(bucket=bucket, prefix=prefix)
    if url.startswith("gs://"):
        parts = url[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else "payloads"
        return _GCSPayloadStore(bucket=bucket, prefix=prefix)
    if "://" in url:
        raise ValueError(
            f"Unsupported payload store URL scheme: {url!r}. "
            "Supported: 's3://bucket/prefix', 'gs://bucket/prefix', or a local filesystem path."
        )
    return _LocalPayloadStore(base_dir=url)


def _resolve_generation_admission(registry: ModelRegistry, model_id: str) -> tuple[int | None, bool | None]:
    """Resolve generation admission settings for the model being processed."""
    try:
        config = registry.get_config(model_id)
    except KeyError:
        return None, None
    if config.tasks.generate is None:
        return None, None
    try:
        resolved = config.resolve_profile("default")
    except Exception:  # noqa: BLE001
        logger.debug(
            "Failed to resolve 'default' profile for admission lookup on %s",
            config.sie_id,
            exc_info=True,
        )
        return None, None
    return resolved.kv_budget_tokens, resolved.admission_enabled


def _fairness_config_from_env() -> FairnessConfig | None:
    """Build the opt-in pool-fairness config from env, or ``None`` (default).

    Off unless ``SIE_POOL_FAIRNESS_ENABLED`` is truthy. When on, the generation
    and embedding work classes share ``SIE_POOL_FAIRNESS_TOTAL_SLOTS`` by weight,
    with a reserved ``min_slots`` floor per class so embedding/score are not
    starved under generation saturation. Returns ``None`` (scheduler off →
    dispatch byte-identical to today) unless explicitly enabled, so this is a
    zero-impact default. A misconfiguration is logged and disables the scheduler
    rather than crashing the worker.
    """
    if os.environ.get("SIE_POOL_FAIRNESS_ENABLED", "").strip().lower() not in ("1", "true", "yes", "on"):
        return None

    def _int(name: str, default: int) -> int:
        try:
            return int(os.environ[name])
        except (KeyError, ValueError):
            return default

    def _float(name: str, default: float) -> float:
        try:
            return float(os.environ[name])
        except (KeyError, ValueError):
            return default

    cfg = FairnessConfig(
        total_slots=_int("SIE_POOL_FAIRNESS_TOTAL_SLOTS", 8),
        classes={
            GENERATION_CLASS: WorkClassConfig(
                weight=_float("SIE_POOL_FAIRNESS_GEN_WEIGHT", 1.0),
                min_slots=_int("SIE_POOL_FAIRNESS_GEN_MIN_SLOTS", 0),
            ),
            EMBEDDING_CLASS: WorkClassConfig(
                weight=_float("SIE_POOL_FAIRNESS_EMB_WEIGHT", 1.0),
                min_slots=_int("SIE_POOL_FAIRNESS_EMB_MIN_SLOTS", 1),
            ),
        },
    )
    try:
        cfg.validate()
    except ValueError as exc:
        logger.error("Invalid SIE_POOL_FAIRNESS_* config (%s); disabling fairness scheduler", exc)
        return None
    return cfg


class NatsPullLoop:
    """Pull work items from NATS JetStream and feed to ModelWorker.

    One pull loop per worker process. The loop round-robins across models
    in the worker's bundle, pulling items from each model's JetStream
    subject and submitting them to the appropriate ModelWorker.
    """

    def __init__(
        self,
        nc: NATSClient,
        js: JetStreamContext,
        registry: ModelRegistry,
        bundle_id: str,
        pool_name: str,
        payload_store_url: str | None = None,
    ) -> None:
        self._nc = nc
        self._js = js
        self._registry = registry
        self._bundle_id = bundle_id
        self._pool_name = pool_name
        self._payload_store_url = payload_store_url
        self._subscriptions: dict[str, Any] = {}  # model_id → PullSubscription
        self._running = False
        self._pull_task: asyncio.Task[None] | None = None
        # Parallel pull task for the per-worker direct-dispatch
        # subscription. ``None`` until ``start()`` creates it; left
        # ``None`` on direct-dispatch boot failures so the pool path keeps
        # working.
        self._worker_pull_task: asyncio.Task[None] | None = None
        self._worker_sub: Any | None = None
        self._hosts_generation_model: bool | None = None
        self._in_flight_tasks: set[asyncio.Task[None]] = set()
        # Outstanding generation streams, tracked separately from the
        # GPU-batch ``_in_flight_tasks``. Generation is decoupled from
        # ``_batch_sem`` (see :meth:`_process_generate_items`): each stream
        # runs as its own task so it does NOT hold a GPU-batch slot for its
        # whole lifetime — that capped generation at ``_MAX_CONCURRENT_BATCHES``
        # and starved direct-dispatch. Concurrency is instead bounded by the
        # KV-budget admission gate inside :class:`StreamingProcessor`. We
        # keep strong refs here so the loop doesn't GC a stream mid-flight
        # and so :meth:`stop` can drain them gracefully.
        self._in_flight_generate_tasks: set[asyncio.Task[None]] = set()
        self._payload_store: _PayloadStore | None = None
        # Stable worker identity. Priority order:
        # 1. SIE_WORKER_ID — explicit operator override; survives pod restarts.
        # 2. HOSTNAME — set by K8s (per-pod), reasonable for cluster deployments.
        # 3. POD_NAME — secondary K8s fallback if HOSTNAME is masked.
        # 4. uuid4 hex — last-resort for local dev so multi-process runs don't
        #    collide on the durable consumer name ``gen-{worker_id}``.
        # Previous default was a bare ``"worker"`` literal which broke local
        # parallelism by sharing the same JetStream durable across processes.
        #
        # The resolved value is then normalized via ``normalize_worker_id``
        # (mirrors the gateway's ``normalize_model_id`` in
        # ``packages/sie_gateway/src/queue/publisher.rs``) before any subject
        # is composed. Kubernetes pod hostnames embed ``.`` which is the NATS
        # subject separator; without normalization here the gateway publishes
        # to ``sie.work.{model}.{pool}.{normalized}`` while the worker would
        # bind its stream to ``sie.work.*.{pool}.{raw-with-dots}`` and
        # direct-dispatch would silently miss until the pool-fallback fires
        # (workstream G-M5).
        _worker_id_raw = (
            os.environ.get("SIE_WORKER_ID")
            or os.environ.get("HOSTNAME")
            or os.environ.get("POD_NAME")
            or uuid.uuid4().hex
        )
        try:
            self._worker_id = normalize_worker_id(_worker_id_raw)
        except ValueError:
            # All four sources empty/whitespace is extremely unlikely
            # (uuid4().hex is non-empty by construction) but if a future
            # change broke that invariant we want a deterministic fallback
            # rather than silently substituting "worker" — that constant
            # collides durable consumers across processes.
            logger.error(
                "nats_pull_loop: resolved worker_id %r is empty after "
                "normalization; falling back to fresh uuid4 to avoid "
                "durable-consumer collisions",
                _worker_id_raw,
            )
            self._worker_id = normalize_worker_id(uuid.uuid4().hex)
        # Keep the pre-normalization form for log messages where the raw
        # hostname is more recognizable to operators. Subject construction
        # MUST use ``self._worker_id`` (the normalized form) so the gateway
        # and worker agree on the wire.
        self._worker_id_raw = _worker_id_raw
        if self._worker_id == _worker_id_raw:
            logger.info("nats_pull_loop resolved worker_id=%s", self._worker_id)
        else:
            logger.info(
                "nats_pull_loop resolved worker_id=%s (normalized from %r)",
                self._worker_id,
                _worker_id_raw,
            )
        # Saturation gate. Driven from in-flight task count vs
        # the aggregate ``max_batch_requests``. Read by ``api/ws.py``
        # (via ``app.state.nats_pull_loop``) and the optional NATS
        # health publisher.
        from sie_server.health.saturation import SaturationGate  # noqa: PLC0415

        self._saturation_gate = SaturationGate()
        self._machine_profile = os.environ.get("SIE_MACHINE_PROFILE") or os.environ.get("SIE_GPU_TYPE", "")
        gateway_url = os.environ.get("SIE_GATEWAY_URL", "") if _env_bool("SIE_POOL_ADMISSION_ENABLED", True) else ""
        self._admission_gate = _PoolAdmissionGate(
            pool_name=self._pool_name,
            worker_id=self._worker_id,
            machine_profile=self._machine_profile,
            gateway_url=gateway_url,
            api_key=_first_csv_token(os.environ.get("SIE_GATEWAY_API_KEY") or os.environ.get("SIE_AUTH_TOKEN")),
        )
        self._failed_models: set[str] = set()
        self._config_hash_cache: str | None = None
        self._config_hash_cache_time: float = 0.0
        self._batch_sem = asyncio.Semaphore(_MAX_CONCURRENT_BATCHES)

        # Opt-in mixed-pool fairness: when ``SIE_POOL_FAIRNESS_ENABLED`` is set,
        # dispatch is gated by a weighted-fair-queue scheduler with per-class
        # ``min_slots`` floors (generation vs embedding/score/extract), so a
        # saturated generation stream cannot starve embeddings on a shared pool.
        # ``None`` (default) → unchanged dispatch.
        _fairness_cfg = _fairness_config_from_env()
        self._scheduler: WorkClassScheduler | None = (
            WorkClassScheduler(_fairness_cfg) if _fairness_cfg is not None else None
        )
        if self._scheduler is not None:
            logger.info("nats_pull_loop: work-class fairness scheduler enabled (%s)", _fairness_cfg)

        # Reactive model loading state
        self._loading_models: set[str] = set()  # Models currently being loaded
        self._load_tasks: dict[str, asyncio.Task[None]] = {}  # model → background load task
        self._load_history: list[tuple[str, float]] = []  # (model_id, timestamp) for thrash detection

        # Adaptive fetch timeout: tracks queue-path latency (time from
        # gateway publish to result publish) and adjusts fetch_timeout to
        # balance item accumulation (throughput) vs head-of-line delay.
        # When latency is under the target, the timeout can increase to
        # allow more items to accumulate → larger GPU batches.  When over
        # the target, the timeout shrinks for faster pickup.
        self._queue_latency_tracker = LatencyTracker(window_size=200, min_samples=10)

        # ``MessageProcessor`` seam. The existing batch path
        # (encode/score/extract) is not yet flipped through this protocol;
        # only generation work items are dispatched here.
        from sie_server.processors.admission import resolve_admission_enabled  # noqa: PLC0415
        from sie_server.processors.streaming import StreamingProcessor  # noqa: PLC0415

        def resolve_effective_admission(model_id: str) -> tuple[int | None, bool | None]:
            kv_budget_tokens, profile_admission = _resolve_generation_admission(self._registry, model_id)
            return kv_budget_tokens, resolve_admission_enabled(profile_admission=profile_admission)

        self._streaming_processor = StreamingProcessor(
            nc=self._nc,
            registry=self._registry,
            worker_id=self._worker_id,
            admission_resolver=resolve_effective_admission,
        )
        # Cancel subscription (cluster-scope ``cancel.>``). Created
        # lazily in :meth:`start`; ``None`` until then so :meth:`stop`
        # tolerates an early shutdown.
        self._cancel_sub: Any | None = None
        self._pool_sub: Any | None = None

    async def start(self) -> None:
        """Start the pull loop.

        Creates a single multiplexed JetStream consumer for the worker's
        pool and begins pulling work items for all models.
        """
        self._running = True
        self._payload_store = _create_payload_store(self._payload_store_url)

        # Migrate: remove legacy per-model streams that overlap with the
        # new pool-level stream subjects. ``add_stream`` will reject the
        # pool stream if any per-model stream has an overlapping subject.
        await self._migrate_per_model_streams()

        self._refresh_generation_pool_flag()

        # Create the single pool-level stream + consumer
        await self._ensure_pool_subscription()

        if self._pool_hosts_generation_model():
            # Per-worker direct-dispatch stream + consumer. The
            # pool stream's wildcard ``sie.work.*.{pool}`` captures exactly
            # 3 subject tokens after the prefix; the per-worker subject has
            # 4 tokens (``…{pool}.{worker_id}``) so it cannot overlap. Both
            # subscriptions feed the same ``_process_messages`` path.
            # Failure here is non-fatal — workers without a per-worker
            # subscription degrade to the pool-fallback path (the gateway
            # already handles "no eligible worker" by publishing to the
            # pool subject).
            try:
                await self._ensure_per_worker_subscription()
            except Exception:
                logger.exception("Failed to create per-worker subscription (continuing with pool-only delivery)")

            # Subscribe (cluster-scope) to cancel.> on core NATS so
            # the StreamingProcessor can react to client disconnect cancels.
            # request_id uniqueness is the filter; router_id is informational.
            await self._ensure_cancel_subscription()
        else:
            logger.info("Skipping generation direct-dispatch subscriptions for non-generation pool %s", self._pool_name)

        # Grammar prewarm. Compile each model's
        # ``tasks.generate.prewarm_grammars`` ahead of request traffic so
        # the cold-start TTFT for those schemas excludes Outlines
        # compile cost. Runs before ``_pull_task`` starts so the cache
        # is hot by the time the first work item arrives. Non-generation
        # models are skipped via :func:`is_generation_model`. Per-entry
        # failures are absorbed inside the processor and surfaced via
        # ``sie_worker_grammar_prewarm_total{outcome="failed"}``.
        await self._prewarm_grammars()

        # Start the main pull loop
        self._pull_task = asyncio.create_task(self._run())
        # Parallel pull task draining the per-worker stream.
        # Spawned only if the per-worker subscription was created.
        if self._worker_sub is not None:
            self._worker_pull_task = asyncio.create_task(self._run_worker_pull())
        logger.info(
            "NatsPullLoop started (bundle=%s, pool=%s, models=%d, stream=%s)",
            self._bundle_id,
            self._pool_name,
            len(self._registry.model_names),
            work_pool_stream_name(self._pool_name),
        )

    async def stop(self) -> None:
        """Stop the pull loop gracefully.

        Cancels the pull task and waits for in-flight processing to complete.
        """
        self._running = False

        # Cancel the main pull loop
        if self._pull_task is not None:
            self._pull_task.cancel()
            try:
                await self._pull_task
            except asyncio.CancelledError:
                pass
            self._pull_task = None

        if self._worker_pull_task is not None:
            self._worker_pull_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_pull_task
            self._worker_pull_task = None

        # Wait for in-flight tasks to complete (graceful drain).
        # Do NOT cancel — let current batches finish, ACK, and drain cleanly.
        # Items processed during drain are ACKed normally; no GPU work wasted.
        #
        # ``_in_flight_generate_tasks`` is drained alongside the GPU-batch
        # ``_in_flight_tasks``: generation streams run as independent tasks
        # (decoupled from ``_batch_sem`` — see ``_process_generate_items``)
        # so they would otherwise not be awaited here, dropping in-flight
        # generations on shutdown. Draining both with the same timeout +
        # cancel-on-overrun policy keeps the graceful-shutdown contract.
        drain_tasks = self._in_flight_tasks | self._in_flight_generate_tasks
        if drain_tasks:
            logger.info(
                "Draining %d in-flight task(s) (%d batch, %d generation)",
                len(drain_tasks),
                len(self._in_flight_tasks),
                len(self._in_flight_generate_tasks),
            )
            _done, pending = await asyncio.wait(drain_tasks, timeout=_DRAIN_TIMEOUT_S)
            if pending:
                logger.warning("Drain timeout: %d tasks still pending, cancelling", len(pending))
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
        self._in_flight_tasks.clear()
        self._in_flight_generate_tasks.clear()

        # Drain the fairness scheduler (no-op when disabled): pulling has
        # stopped and in-flight work is settled above, so all slots are
        # released and this returns promptly.
        if self._scheduler is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._scheduler.drain(), timeout=5.0)

        # Cancel background model load tasks
        for load_task in self._load_tasks.values():
            load_task.cancel()
        if self._load_tasks:
            await asyncio.gather(*self._load_tasks.values(), return_exceptions=True)
        self._load_tasks.clear()
        self._loading_models.clear()

        for sub in self._subscriptions.values():
            try:
                await sub.unsubscribe()
            except Exception:  # noqa: BLE001
                logger.debug("Failed to unsubscribe during shutdown")
        self._subscriptions.clear()

        if self._pool_sub is not None:
            try:
                await self._pool_sub.unsubscribe()
            except Exception:  # noqa: BLE001
                logger.debug("Failed to unsubscribe pool sub during shutdown")
            self._pool_sub = None
        await self._admission_gate.close()

        # Tear down the per-worker subscription.
        if self._worker_sub is not None:
            try:
                await self._worker_sub.unsubscribe()
            except Exception:  # noqa: BLE001
                logger.debug("Failed to unsubscribe per-worker sub during shutdown")
            self._worker_sub = None

        if hasattr(self, "_cancel_sub") and self._cancel_sub is not None:
            try:
                await self._cancel_sub.unsubscribe()
            except Exception:  # noqa: BLE001
                logger.debug("Failed to unsubscribe cancel sub during shutdown")
            self._cancel_sub = None

        logger.info("NatsPullLoop stopped")

    async def handle_reconnect(self) -> None:
        """Re-create pull subscriptions after NATS reconnect.

        After a NATS server restart, existing subscriptions may be invalid.
        This clears them and re-creates consumers for all known models.
        """
        # Unsubscribe stale subscriptions (best-effort)
        for sub in self._subscriptions.values():
            try:
                await sub.unsubscribe()
            except Exception:  # noqa: BLE001
                logger.debug("Failed to unsubscribe during reconnect")
        self._subscriptions.clear()

        # Unsubscribe stale pool subscription (best-effort)
        if self._pool_sub is not None:
            try:
                await self._pool_sub.unsubscribe()
            except Exception:  # noqa: BLE001
                logger.debug("Failed to unsubscribe old pool sub during reconnect")
            self._pool_sub = None

        if self._worker_pull_task is not None:
            self._worker_pull_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_pull_task
            self._worker_pull_task = None

        # Also drop the per-worker subscription so the next
        # `_ensure_per_worker_subscription` call re-binds against the
        # new connection.
        if self._worker_sub is not None:
            try:
                await self._worker_sub.unsubscribe()
            except Exception:  # noqa: BLE001
                logger.debug("Failed to unsubscribe old per-worker sub during reconnect")
            self._worker_sub = None

        self._refresh_generation_pool_flag()

        # Re-create the pool subscription
        await self._ensure_pool_subscription()
        if self._pool_hosts_generation_model():
            # Re-create the per-worker subscription (best-effort).
            try:
                await self._ensure_per_worker_subscription()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to re-create per-worker subscription on reconnect", exc_info=True)
        else:
            logger.info("Reconnect kept non-generation pool %s on pool-only delivery", self._pool_name)

        # Re-spawn the parallel pull task if needed.
        if (
            self._running
            and self._worker_sub is not None
            and (self._worker_pull_task is None or self._worker_pull_task.done())
        ):
            self._worker_pull_task = asyncio.create_task(self._run_worker_pull())
        logger.info("NatsPullLoop reconnected — re-created pool subscription")

    async def add_model(self, model_id: str) -> None:
        """Refresh transport wiring after a model is hot-added.

        The pool subscription already covers all encode / score /
        extract models. If the added model is the first generation
        model in this worker pool, create the per-worker
        direct-dispatch subscription now so hot reload preserves the
        same generation path as cold start.
        """
        if not self._running:
            return

        was_generation_pool = self._pool_hosts_generation_model()
        if not self._refresh_generation_pool_flag():
            return

        if self._worker_sub is None:
            try:
                await self._ensure_per_worker_subscription()
            except Exception:
                logger.exception(
                    "Failed to create per-worker subscription after adding model %s (continuing with pool-only delivery)",
                    model_id,
                )

        if not was_generation_pool:
            await self._rebind_pool_subscription(reason=f"first generation model added: {model_id}")

        if self._cancel_sub is None:
            await self._ensure_cancel_subscription()

        if self._worker_sub is not None and (self._worker_pull_task is None or self._worker_pull_task.done()):
            self._worker_pull_task = asyncio.create_task(self._run_worker_pull())

    # -- Pool-level multiplexed stream/consumer --------------------------------

    async def _migrate_per_model_streams(self) -> None:
        """Delete legacy per-model streams that overlap with the pool stream.

        The old design created ``WORK_{model_id}`` streams per model.
        The new pool-level stream ``WORK_POOL_{pool}`` uses ``sie.work.*.{pool}``
        which overlaps. NATS rejects streams with overlapping subjects, so we
        must clean up old streams before creating the pool stream.
        """
        model_ids = list(self._registry.model_names)
        deleted = 0
        for model_id in model_ids:
            stream_name = work_stream_name(model_id)
            try:
                await self._js.delete_stream(stream_name)
                deleted += 1
            except Exception:  # noqa: BLE001, S110
                pass  # Stream may not exist — that's fine
        if deleted:
            logger.info("Migrated: deleted %d legacy per-model streams", deleted)

    async def _ensure_pool_subscription(self) -> None:
        """Create the single pool-level stream and pull consumer.

        One stream captures ``sie.work.*.{pool}`` (all models for this pool).
        One durable consumer pulls from it. This replaces the old O(N-models)
        per-model subscription loop with a single fetch point.
        """
        stream_name = work_pool_stream_name(self._pool_name)
        subjects = work_pool_stream_subjects(self._pool_name)
        consumer_name = work_consumer_name(self._bundle_id, self._pool_name)

        config = StreamConfig(
            name=stream_name,
            subjects=subjects,
            retention=RetentionPolicy.WORK_QUEUE,
            max_age=_DEFAULT_STREAM_MAX_AGE_S,
            max_msgs=100_000,
            storage=StorageType.MEMORY,
            num_replicas=1,
            discard=DiscardPolicy.NEW,
        )

        try:
            await self._js.add_stream(config)
            logger.info(
                "Pool stream created/verified: %s (subjects=%s, max_age=%ds)",
                stream_name,
                subjects,
                _DEFAULT_STREAM_MAX_AGE_S,
            )
        except Exception:
            logger.exception("Failed to create pool stream: %s", stream_name)
            raise

        # Filter subject: all models for this pool
        filter_subject = f"{WORK_SUBJECT_PREFIX}.*.{self._pool_name}"

        # §4.4: extend ack_wait to 5 min if any model in this
        # pool declares a generate task. Per §4.3 generation pools are
        # isolated from embed/extract pools, so this is per-pool.
        ack_wait_s = _ACK_WAIT_S_GENERATION if self._pool_hosts_generation_model() else _ACK_WAIT_S

        consumer_config = ConsumerConfig(
            durable_name=consumer_name,
            filter_subject=filter_subject,
            ack_policy=AckPolicy.EXPLICIT,
            ack_wait=ack_wait_s,
            max_deliver=_MAX_DELIVER,
            max_ack_pending=1000,
        )

        try:
            sub = await self._js.pull_subscribe(
                subject=filter_subject,
                durable=consumer_name,
                stream=stream_name,
                config=consumer_config,
            )
        except nats.js.errors.BadRequestError:
            logger.warning(
                "Consumer %s config mismatch (likely max_deliver upgrade), recreating",
                consumer_name,
            )
            await self._js.delete_consumer(stream_name, consumer_name)
            sub = await self._js.pull_subscribe(
                subject=filter_subject,
                durable=consumer_name,
                stream=stream_name,
                config=consumer_config,
            )
        except Exception:
            logger.exception("Failed to create pool consumer: %s", consumer_name)
            raise

        self._pool_sub = sub
        logger.info(
            "Pool consumer created: %s on stream %s (filter=%s)",
            consumer_name,
            stream_name,
            filter_subject,
        )

    async def _rebind_pool_subscription(self, *, reason: str) -> None:
        """Recreate the pool consumer when generation settings change."""
        if self._pool_sub is not None:
            try:
                await self._pool_sub.unsubscribe()
            except Exception:  # noqa: BLE001
                logger.debug("Failed to unsubscribe old pool sub during rebind")
            self._pool_sub = None

        await self._ensure_pool_subscription()
        logger.info("Pool consumer re-bound (%s)", reason)

    async def _ensure_per_worker_subscription(self) -> None:
        """Create the per-worker direct-dispatch stream and pull consumer.

        Subject pattern: ``sie.work.*.{pool}.{worker_id}`` — one
        additional token beyond the pool stream's three-token
        wildcard, so the two streams cannot double-deliver. Durable
        consumer name ``gen-{worker_id}`` per the routing spec; the
        worker_id must be stable across boots to avoid orphan
        durables (see ``NatsPullLoop.__init__`` for the resolution
        priority).
        """
        stream_name = work_worker_stream_name(self._worker_id)
        subjects = work_worker_stream_subjects(self._pool_name, self._worker_id)
        consumer_name = work_worker_consumer_name(self._worker_id)
        filter_subject = subjects[0]

        # Same retention/storage settings as the pool stream — the
        # per-worker stream sits on the same JetStream cluster and
        # carries the same item shape; only the addressing differs.
        config = StreamConfig(
            name=stream_name,
            subjects=subjects,
            retention=RetentionPolicy.WORK_QUEUE,
            max_age=_DEFAULT_STREAM_MAX_AGE_S,
            max_msgs=100_000,
            storage=StorageType.MEMORY,
            num_replicas=1,
            discard=DiscardPolicy.NEW,
        )

        try:
            await self._js.add_stream(config)
            logger.info(
                "Per-worker stream created/verified: %s (subjects=%s)",
                stream_name,
                subjects,
            )
        except Exception:
            logger.exception("Failed to create per-worker stream: %s", stream_name)
            raise

        ack_wait_s = _ACK_WAIT_S_GENERATION if self._pool_hosts_generation_model() else _ACK_WAIT_S
        consumer_config = ConsumerConfig(
            durable_name=consumer_name,
            filter_subject=filter_subject,
            ack_policy=AckPolicy.EXPLICIT,
            ack_wait=ack_wait_s,
            max_deliver=_MAX_DELIVER,
            max_ack_pending=1000,
        )

        try:
            sub = await self._js.pull_subscribe(
                subject=filter_subject,
                durable=consumer_name,
                stream=stream_name,
                config=consumer_config,
            )
        except nats.js.errors.BadRequestError:
            logger.warning(
                "Per-worker consumer %s config mismatch, recreating",
                consumer_name,
            )
            await self._js.delete_consumer(stream_name, consumer_name)
            sub = await self._js.pull_subscribe(
                subject=filter_subject,
                durable=consumer_name,
                stream=stream_name,
                config=consumer_config,
            )
        except Exception:
            logger.exception("Failed to create per-worker consumer: %s", consumer_name)
            raise

        self._worker_sub = sub
        logger.info(
            "Per-worker consumer created: %s on stream %s (filter=%s)",
            consumer_name,
            stream_name,
            filter_subject,
        )

    def _refresh_generation_pool_flag(self) -> bool:
        self._hosts_generation_model = self._detect_pool_hosts_generation_model()
        return self._hosts_generation_model

    def _pool_hosts_generation_model(self) -> bool:
        """Return whether this pool hosts generation models.

        The answer is cached because status heartbeats call the saturation
        path frequently. Startup, reconnect, and hot-add refresh the flag
        so normal status updates stay O(1) with respect to model count.
        """
        if self._hosts_generation_model is None:
            return self._refresh_generation_pool_flag()
        return self._hosts_generation_model

    def _detect_pool_hosts_generation_model(self) -> bool:
        """Detect whether any model in this pool declares a generate task.

        Used to pick the worker transport and the appropriate JetStream
        ``ack_wait`` (§4.4: long generations would exceed the default 30s
        window).
        """
        for model_id in self._registry.model_names:
            try:
                config = self._registry.get_config(model_id)
            except KeyError:
                continue
            tasks = getattr(config, "tasks", None)
            if tasks is not None and getattr(tasks, "generate", None) is not None:
                return True
        return False

    async def _prewarm_grammars(self) -> None:
        """Prewarm grammar caches for every generation model in the registry.

        Iterates the worker's loaded configs and invokes
        :meth:`StreamingProcessor.prewarm_grammars_for_model` on each
        generation config (the
        :func:`~sie_server.core.pool_isolation.is_generation_model` gate
        excludes encode/score/extract). Failures inside the processor
        are absorbed and surfaced via metrics; this loop never raises.

        Sequential rather than parallel because (a) prewarm is a one-shot
        boot-time operation, (b) the dedicated grammar executor only has
        4 threads anyway, and (c) sequential keeps log lines clean for
        operators watching boot.
        """
        from sie_server.core.pool_isolation import is_generation_model  # noqa: PLC0415

        for model_id in self._registry.model_names:
            try:
                config = self._registry.get_config(model_id)
            except KeyError:
                # Model removed between iteration start and lookup
                # (hot-reload race). Skip silently — the next reload
                # will re-attempt.
                continue
            if not is_generation_model(config):
                continue
            await self._streaming_processor.prewarm_grammars_for_model(model_id)

    async def _ensure_cancel_subscription(self) -> None:
        """Subscribe (core NATS) to ``cancel.>`` for streaming-generation cancels.

        The cancel path uses cluster-scope subscription with in-process
        request_id filtering: every worker hears every cancel signal but
        only acts on those whose request_id is currently in flight on
        this worker (via :meth:`StreamingProcessor.signal_cancel`).

        request_ids are globally unique UUIDs so the cluster-scope
        approach is correct without router_id discrimination. Spurious
        signals for someone else's request_id are dropped cheaply
        (DashMap lookup).
        """
        # Expected ``cancel.{router_id}.{request_id}`` → 3 tokens after split.
        _CANCEL_SUBJECT_TOKENS = 3

        async def _on_cancel(msg: Any) -> None:
            try:
                # We take the last token (everything after the second
                # dot) as the request_id and let the streaming processor
                # decide whether to act on it.
                subject = msg.subject
                parts = subject.split(".", 2)
                if len(parts) < _CANCEL_SUBJECT_TOKENS:
                    return
                request_id = parts[2]
                matched = self._streaming_processor.signal_cancel(request_id)
                if matched:
                    logger.info("Generation cancel signalled for %s", request_id)
            except Exception:  # noqa: BLE001
                logger.debug("Failed to process cancel message", exc_info=True)

        try:
            self._cancel_sub = await self._nc.subscribe("cancel.>", cb=_on_cancel)
            logger.info("Subscribed to cancel.> for generation cancellation signals")
        except Exception:  # noqa: BLE001
            logger.warning("Failed to subscribe to cancel.> — cancellation disabled", exc_info=True)
            self._cancel_sub = None

    def _get_batch_budget(self, model_id: str) -> int:
        """Get the batch budget for a model from its BatchConfig.

        Accesses ModelWorker._batch_config (private) — falls back to
        _DEFAULT_BATCH_BUDGET if unavailable.
        """
        try:
            worker = self._registry.get_worker(model_id)
            if worker is not None and hasattr(worker, "_batch_config"):
                return worker._batch_config.max_batch_requests
        except (KeyError, AttributeError):
            pass
        return _DEFAULT_BATCH_BUDGET

    @property
    def worker_id(self) -> str:
        """Stable worker identity used for the per-worker NATS subject
        and the JetStream durable consumer name. See ``__init__`` for
        the resolution order.
        """
        return self._worker_id

    def in_flight_count(self) -> int:
        """Current in-flight count for the pool's transport class.

        Generation pools count streaming requests; non-generation pools
        count fetched queue batches. Keeping those signals separate is
        important: the generation rollout added direct-dispatch state,
        but encode / score / extract still rely on the shared pool queue
        and must not report zero in-flight work just because no
        generation stream is active.
        """
        if self._pool_hosts_generation_model():
            return self._streaming_processor.in_flight_count()
        return len(self._in_flight_tasks)

    def aggregate_max_batch_requests(self) -> int:
        """Aggregate batch capacity across loaded models.

        The routing rollout uses ``in_flight / aggregate_max_batch_requests`` as
        the saturation fraction. We take the minimum across loaded
        models because the GPU batch is model-specific and the
        smallest capacity is the binding constraint — mirroring
        ``api.ws.build_status_message``.
        """
        # Snapshot `_loaded` to dodge `RuntimeError` from concurrent
        # mutation; same approach as `build_status_message`.
        loaded_snapshot = list(getattr(self._registry, "_loaded", {}).values())
        budgets = [
            w._batch_config.max_batch_requests
            for w in (lm.worker for lm in loaded_snapshot)
            if w is not None and hasattr(w, "_batch_config")
        ]
        if budgets:
            return min(budgets)
        return _DEFAULT_BATCH_BUDGET

    def update_saturation(self) -> bool:
        """Recompute the saturation flag from the current in-flight
        count and aggregate capacity, returning the new latched value.

        Idempotent. Callers (the WS status builder, the optional NATS
        health publisher, and the per-worker subscription gate) drive
        this on every status emission so the gate sees a steady
        stream of observations under load.

        Admission-control rollout: generation pools prefer the KV-budget
        signal (``kv_reserved / kv_budget``) regardless of whether
        admission is actually enabled — the budget is the right
        denominator for a generation worker's saturation, and the
        reserve counter is live whenever the streaming processor is
        wired in. Non-gen pools (no budget configured) fall back to the
        original ``in_flight / max_batch_requests`` fraction.
        """
        budget = self._effective_kv_budget()
        if budget is not None and budget > 0:
            return self._saturation_gate.update(
                in_flight=self._streaming_processor.kv_reserved_tokens(),
                capacity=budget,
            )
        return self._saturation_gate.update(
            in_flight=self.in_flight_count(),
            capacity=self.aggregate_max_batch_requests(),
        )

    def _effective_kv_budget(self) -> int | None:
        """Worker-wide KV budget for the saturation gate.

        The streaming processor resolves admission budgets *per request* and
        is never seeded with a worker-wide value, so its
        ``kv_budget_tokens`` property is ``None`` in production — which left
        the KV-budget saturation signal dead and silently falling back to
        the request-count fraction. Derive the budget here instead: the
        explicit processor override if one was injected, else the minimum
        ``kv_budget_tokens`` across loaded generation models (the binding
        constraint, mirroring :meth:`aggregate_max_batch_requests`). Returns
        ``None`` when no loaded generation model declares a budget, so
        non-generation pools keep using the request-count fraction.
        """
        explicit = self._streaming_processor.kv_budget_tokens
        if explicit is not None and explicit > 0:
            return explicit
        budgets: list[int] = []
        for model_id in self._registry.loaded_model_names:
            budget, _ = _resolve_generation_admission(self._registry, model_id)
            if budget is not None and budget > 0:
                budgets.append(budget)
        return min(budgets) if budgets else None

    @property
    def saturated(self) -> bool:
        """Latched saturation flag, without recomputing. Use
        :meth:`update_saturation` to refresh.
        """
        return self._saturation_gate.saturated

    def _check_config_hash(self, model_id: str, wi: WorkItem) -> bool:
        """Log-only soft check of bundle_config_hash — always returns True.

        Compares the hash from the work item against the local config.
        Mismatches are logged but do not reject the item, because hash
        computation differs between gateway and worker registries.
        """
        expected_hash = wi.get("bundle_config_hash")
        if not expected_hash:
            return True  # No hash set — backward compatible

        # Use cached hash (recompute every _CONFIG_HASH_CACHE_TTL_S seconds)
        now = time.monotonic()
        if self._config_hash_cache is None or (now - self._config_hash_cache_time) > _CONFIG_HASH_CACHE_TTL_S:
            try:
                from sie_server.api.ws import _compute_bundle_config_hash  # noqa: PLC0415

                self._config_hash_cache = _compute_bundle_config_hash(self._registry, self._bundle_id)
                self._config_hash_cache_time = now
            except Exception:  # noqa: BLE001
                logger.debug("Could not compute config hash for validation", exc_info=True)
                return True

        if self._config_hash_cache and self._config_hash_cache != expected_hash:
            # Log-only: hash computation differs between gateway and worker
            # registries (different model filtering). Allow processing to
            # proceed — the hash is a soft check, not a hard gate. Still
            # record the mismatch on the dedicated counter so operators can
            # see drift even though items are not rejected.
            if _HAS_METRICS:
                PULL_CONFIG_HASH_MISMATCHES.labels(model=model_id).inc()
            logger.debug(
                "Config hash mismatch for %s (gateway=%s, worker=%s) — processing anyway",
                wi.get("work_item_id"),
                expected_hash[:8],
                self._config_hash_cache[:8],
            )

        return True

    def _extract_model_id(self, msg: Any) -> str | None:
        """Extract model_id from the NATS message subject.

        Subject format: ``sie.work.{normalized_model_id}.{pool_name}``
        We reverse the normalization to recover the original model_id.
        """
        subject = msg.subject  # e.g., "sie.work.BAAI__bge-m3.l4"
        parts = subject.split(".")
        if len(parts) < _MIN_SUBJECT_PARTS:
            return None
        normalized = parts[2]  # e.g., "BAAI__bge-m3"
        return denormalize_model_id(normalized)

    def _adaptive_fetch_timeout(self, current: float) -> float:
        """Adjust fetch timeout based on observed queue-path latency.

        Under load, when latency is below the target SLO, increase the
        timeout to allow more items to accumulate in the queue → bigger
        GPU batches → better throughput.  When over SLO, decrease for
        faster pickup → lower latency at the cost of smaller batches.

        At idle (not enough samples), fall back to the static adaptive
        backoff (caller handles that case).
        """
        observed = self._queue_latency_tracker.p50()
        if observed is None:
            return _MIN_FETCH_TIMEOUT_S  # Not enough samples — reset to minimum

        # Target: 50ms by default (matches EngineConfig default).
        # In production, this should be configurable per pool.
        target = float(os.environ.get("SIE_ADAPTIVE_TARGET_P50_MS", "50"))

        headroom_ms = target - observed
        gain = 0.2  # Conservative gain for fetch timeout

        # Convert headroom to a timeout adjustment:
        # +headroom → increase timeout (we can afford to wait → bigger batches)
        # -headroom → decrease timeout (over SLO → pick up faster)
        adjustment_s = headroom_ms * gain / 1000.0  # ms → seconds
        new_timeout = current + adjustment_s
        return max(_MIN_FETCH_TIMEOUT_S, min(_MAX_FETCH_TIMEOUT_S, new_timeout))

    async def _run(self) -> None:
        """Main pull loop — single-consumer multiplexed design.

        Pulls from the single pool-level consumer, groups messages by
        model_id (extracted from the NATS subject), and dispatches
        per-model batches for processing with fair scheduling.

        This eliminates the O(N-models) sequential scan of the old design.
        One ``fetch()`` call replaces 83+ per-model fetches.

        **Fair dispatch:** When a batch contains multiple models, each
        model's messages are capped at the model's batch budget.  Excess
        messages are NAK'd for redelivery, preventing a hot model from
        monopolising the GPU.

        **Adaptive fetch timeout:** When the latency tracker has enough
        samples, the fetch timeout is adjusted based on observed p50 vs
        the target SLO. Under load with headroom, the timeout grows to
        allow item accumulation (throughput). Over SLO, it shrinks.
        At idle, static backoff applies (1ms → 20ms).
        """
        fetch_timeout = _MIN_FETCH_TIMEOUT_S

        while self._running:
            # Check completed background load tasks
            self._reap_load_tasks()

            if not await self._admission_gate.admitted():
                fetch_timeout = _MIN_FETCH_TIMEOUT_S
                await asyncio.sleep(self._admission_gate.pause_s)
                continue

            pool_sub = self._pool_sub
            if pool_sub is None:
                fetch_timeout = _MIN_FETCH_TIMEOUT_S
                await asyncio.sleep(fetch_timeout)
                continue

            try:
                messages = await pool_sub.fetch(batch=_DEFAULT_BATCH_BUDGET, timeout=fetch_timeout)
            except nats.errors.TimeoutError:
                # No items — back off
                fetch_timeout = min(fetch_timeout * _BACKOFF_GROWTH, _MAX_FETCH_TIMEOUT_S)
                continue
            except Exception:  # noqa: BLE001
                logger.warning("Pool pull error", exc_info=True)
                fetch_timeout = min(fetch_timeout * _BACKOFF_GROWTH, _MAX_FETCH_TIMEOUT_S)
                await asyncio.sleep(fetch_timeout)
                continue

            if not messages:
                fetch_timeout = min(fetch_timeout * _BACKOFF_GROWTH, _MAX_FETCH_TIMEOUT_S)
                continue

            # Adjust fetch timeout — use adaptive controller if we have
            # enough latency samples, otherwise reset to minimum.
            fetch_timeout = self._adaptive_fetch_timeout(fetch_timeout)

            await self._dispatch_batch(messages)

    async def _dispatch_batch(self, messages: list[Any]) -> None:
        """Group a fetched batch by model_id and dispatch via
        :meth:`_process_messages`.

        Shared by the pool pull loop (:meth:`_run`) and the
        per-worker pull loop (:meth:`_run_worker_pull`). Behaviour
        identical to the original inline body: model-not-loaded
        triggers a NAK + background load; over-budget batches NAK the
        overflow for re-delivery; the per-batch GPU dispatch is
        serialised through ``_batch_sem``.
        """
        # Group messages by model_id (extracted from subject)
        model_groups: dict[str, list[Any]] = {}
        for msg in messages:
            model_id = self._extract_model_id(msg)
            if model_id is None:
                logger.warning("Could not extract model_id from subject: %s", msg.subject)
                try:
                    await msg.nak()
                except Exception:  # noqa: BLE001, S110
                    pass
                continue
            model_groups.setdefault(model_id, []).append(msg)

        # Fair dispatch: round-robin across models, capping per-model batch
        for model_id, model_msgs in model_groups.items():
            # Skip models currently loading — NAK for redelivery
            if model_id in self._loading_models:
                for m in model_msgs:
                    try:
                        await m.nak(delay=_NAK_DELAY_S)
                    except Exception:  # noqa: BLE001, S110
                        pass
                continue

            # Check if model is loaded
            if not self._registry.is_loaded(model_id):
                await self._handle_unloaded_model(model_id, model_msgs)
                continue

            # Apply per-model batch cap for fairness.  If more messages
            # arrived than the model's batch budget, NAK the excess so
            # they are redelivered (possibly to another worker).
            budget = self._get_batch_budget(model_id)
            dispatch_msgs = model_msgs[:budget]
            overflow = model_msgs[budget:]
            for m in overflow:
                try:
                    await m.nak(delay=0.1)  # fast redeliver
                except Exception:  # noqa: BLE001, S110
                    pass

            if _HAS_METRICS:
                PULL_ITEMS_FETCHED.labels(model=model_id).observe(len(dispatch_msgs))

            # Limit concurrent batch processing
            await self._batch_sem.acquire()

            async def _guarded_process(msgs: list[Any], mdl: str) -> None:
                try:
                    await self._process_messages(mdl, msgs)
                finally:
                    self._batch_sem.release()

            task = asyncio.create_task(_guarded_process(dispatch_msgs, model_id))
            self._in_flight_tasks.add(task)
            task.add_done_callback(self._in_flight_tasks.discard)

    async def _run_worker_pull(self) -> None:
        """Parallel pull loop for the per-worker direct-dispatch stream.

        Identical batching/dispatch semantics to :meth:`_run`, but
        sources messages from :attr:`_worker_sub` (filter
        ``sie.work.*.{pool}.{worker_id}``). Both loops share
        ``_batch_sem`` so the worker's GPU is the global concurrency
        limit, not "GPU × 2".
        """
        worker_sub = self._worker_sub
        if worker_sub is None:
            logger.debug("Per-worker pull loop requested without a subscription; exiting")
            return

        fetch_timeout = _MIN_FETCH_TIMEOUT_S
        while self._running:
            try:
                messages = await worker_sub.fetch(batch=_DEFAULT_BATCH_BUDGET, timeout=fetch_timeout)
            except nats.errors.TimeoutError:
                fetch_timeout = min(fetch_timeout * _BACKOFF_GROWTH, _MAX_FETCH_TIMEOUT_S)
                continue
            except Exception:  # noqa: BLE001
                logger.warning("Per-worker pull error", exc_info=True)
                fetch_timeout = min(fetch_timeout * _BACKOFF_GROWTH, _MAX_FETCH_TIMEOUT_S)
                await asyncio.sleep(fetch_timeout)
                continue

            if not messages:
                fetch_timeout = min(fetch_timeout * _BACKOFF_GROWTH, _MAX_FETCH_TIMEOUT_S)
                continue

            fetch_timeout = self._adaptive_fetch_timeout(fetch_timeout)
            await self._dispatch_batch(messages)

    def _reap_load_tasks(self) -> None:
        """Check completed background load tasks and clean up state."""
        for model_id in list(self._load_tasks):
            task = self._load_tasks[model_id]
            if not task.done():
                continue

            self._loading_models.discard(model_id)
            del self._load_tasks[model_id]

            try:
                exc = task.exception()
            except asyncio.CancelledError:
                logger.debug("Background model load cancelled for %s", model_id)
                continue
            if exc is not None:
                logger.warning("Background model load failed for %s: %s", model_id, exc)
            else:
                logger.info("Model %s loaded via background demand, now pulling", model_id)

    async def _handle_unloaded_model(self, model_id: str, messages: list[Any]) -> None:
        """Handle items pulled for a model that isn't loaded yet.

        NAKs all items with a delay so JetStream redelivers them after
        the model has had time to load. Triggers a background load if
        one isn't already in progress.

        Uses a longer NAK delay when a load is already in progress to
        conserve the delivery budget (max_deliver) for slow model loads.
        """
        already_loading = model_id in self._loading_models
        nak_delay = _NAK_DELAY_S * 2 if already_loading else _NAK_DELAY_S

        for msg in messages:
            try:
                await msg.nak(delay=nak_delay)
            except Exception:  # noqa: BLE001
                logger.debug("Failed to NAK message for unloaded model %s", model_id)

        if _HAS_METRICS:
            PULL_NAK_UNLOADED.labels(model=model_id).inc(len(messages))

        logger.info(
            "NAKed %d items for unloaded model %s (redeliver in %.0fs, loading=%s)",
            len(messages),
            model_id,
            nak_delay,
            already_loading,
        )

        # Trigger background load if not already in progress
        if model_id not in self._loading_models:
            self._check_thrashing(model_id)
            self._loading_models.add(model_id)
            load_task = asyncio.create_task(self._background_load_model(model_id))
            self._load_tasks[model_id] = load_task

            if _HAS_METRICS:
                PULL_MODEL_LOADS.labels(model=model_id).inc()

            logger.info("Triggered background load for model %s", model_id)

    async def _background_load_model(self, model_id: str) -> None:
        """Load a model in the background without blocking the pull loop.

        ``registry.load_async()`` serializes loads via an internal lock,
        runs the actual GPU loading in a thread pool, and may evict LRU
        models to free memory. If the evicted model was being served,
        the pull loop will discover ``is_loaded() == False`` on the next
        iteration and NAK its items — no race condition.
        """
        try:
            device = self._registry.device
            await self._registry.load_async(model_id, device)
            self._load_history.append((model_id, time.monotonic()))
        except Exception:
            logger.warning("Background load failed for model %s", model_id, exc_info=True)
            self._loading_models.discard(model_id)
            raise

    def _check_thrashing(self, model_id: str) -> None:
        """Detect and warn about model load/evict thrashing.

        If any model has been background-loaded ``_THRASH_THRESHOLD`` or
        more times within ``_THRASH_WINDOW_S``, log a warning.  This
        indicates the GPU cannot hold all in-demand models simultaneously
        and the operator should use separate bundles or add GPU capacity.
        """
        now = time.monotonic()
        cutoff = now - _THRASH_WINDOW_S

        # Prune old entries
        self._load_history = [(m, t) for m, t in self._load_history if t > cutoff]

        # Count recent loads for this model
        recent = sum(1 for m, _ in self._load_history if m == model_id)
        if recent >= _THRASH_THRESHOLD:
            logger.warning(
                "Model thrashing detected: %s loaded %d times in the last %.0fs. "
                "GPU memory cannot hold all in-demand models simultaneously. "
                "Consider using separate bundles per model or adding GPU capacity.",
                model_id,
                recent,
                _THRASH_WINDOW_S,
            )

    async def _process_messages(self, model_id: str, messages: list[Any]) -> None:
        """Process a batch of pulled NATS messages for a model.

        Groups encode/extract items into batches for optimal GPU utilization.
        Score items are submitted concurrently for cross-request batching.
        """
        work_items: list[tuple[WorkItem, Any]] = []  # (work_item, nats_msg)

        for msg in messages:
            try:
                wi: WorkItem = msgpack.unpackb(msg.data, raw=False)
                # Validate reply_subject to prevent injection attacks.
                # Only accept subjects under the _INBOX prefix.
                reply_subj = wi.get("reply_subject", "")
                if reply_subj and not reply_subj.startswith("_INBOX."):
                    logger.warning(
                        "Rejecting work item with suspicious reply_subject=%r "
                        "model_id=%r bundle_id=%r request_id=%r subject=%r",
                        reply_subj[:60],
                        wi.get("model_id"),
                        wi.get("bundle_id"),
                        wi.get("request_id"),
                        getattr(msg, "subject", None),
                    )
                    await msg.ack()  # Consume to prevent redelivery
                    continue
                work_items.append((wi, msg))
            except Exception:  # noqa: BLE001
                logger.warning("Failed to deserialize work item", exc_info=True)
                await msg.nak()

        if not work_items:
            return

        # Record queue wait times
        if _HAS_METRICS:
            for wi, _ in work_items:
                queue_wait_s = time.time() - wi.get("timestamp", time.time())
                if queue_wait_s > 0:
                    PULL_QUEUE_WAIT_SECONDS.labels(model=model_id).observe(queue_wait_s)

        # Validate bundle_config_hash — log-only soft check (never NAKs)
        for wi, _ in work_items:
            self._check_config_hash(model_id, wi)
        valid_items = work_items

        if not valid_items:
            return

        # Group by operation for batch processing. Generate items go through
        # the ``MessageProcessor`` seam (one task per message) — they don't
        # batch on the GPU side (SGLang does continuous batching internally)
        # and they hold a worker slot for longer than embedding/extract.
        encode_items: list[tuple[WorkItem, Any]] = []
        score_items: list[tuple[WorkItem, Any]] = []
        extract_items: list[tuple[WorkItem, Any]] = []
        generate_items: list[tuple[WorkItem, Any]] = []

        for wi, msg in valid_items:
            op = wi.get("operation", "encode")
            if op == "encode":
                encode_items.append((wi, msg))
            elif op == "score":
                score_items.append((wi, msg))
            elif op == "extract":
                extract_items.append((wi, msg))
            elif op == "generate":
                generate_items.append((wi, msg))
            else:
                logger.warning("Unknown operation %s for %s", op, wi.get("work_item_id"))
                await self._publish_error(wi, "unknown_operation", f"Unknown operation: {op}")
                await msg.ack()

        # Process batches concurrently
        tasks: list[Any] = []
        if encode_items:
            tasks.append(self._run_in_class_slot("encode", self._process_encode_batch(model_id, encode_items)))
        if score_items:
            # Score items submitted concurrently for BatchFormer cross-request batching
            tasks.append(self._run_in_class_slot("score", self._process_score_batch(model_id, score_items)))
        if extract_items:
            tasks.append(self._run_in_class_slot("extract", self._process_extract_batch(model_id, extract_items)))
        if generate_items:
            # Generation slots are acquired per-stream inside
            # ``_guarded_generate_process`` (each stream is its own task), not
            # here — the batch only *dispatches* generate streams.
            tasks.append(self._process_generate_items(model_id, generate_items))

        await asyncio.gather(*tasks)

    async def _process_generate_items(self, model_id: str, items_msgs: list[tuple[WorkItem, Any]]) -> None:
        """Dispatch generate work items through the StreamingProcessor seam.

        One concurrent task per message: generation is intentionally
        unbatched at this layer (SGLang batches internally).

        H2 fix — decouple generation from ``_batch_sem``. Previously this
        method ``await``ed the streams to completion, which (because the
        caller holds ``_batch_sem`` for the whole ``_process_messages``
        body) pinned a GPU-batch slot for each stream's entire lifetime.
        With ``_MAX_CONCURRENT_BATCHES=4`` that capped the worker at 4
        concurrent generations regardless of how much KV budget was free,
        and starved the per-worker direct-dispatch path of slots.

        Instead each stream is launched as an independent tracked task and
        this method returns immediately. ``_batch_sem`` is therefore
        released once the batch is *dispatched* (when ``_process_messages``
        returns), NOT when the streams finish. Generation concurrency is
        bounded by the KV-budget admission gate inside
        :class:`StreamingProcessor` (``_try_reserve`` /
        ``reason="kv_budget"`` NAK), which is the intended backpressure.

        Safety: ``StreamingProcessor.process`` owns its whole lifecycle —
        per-message JetStream ``in_progress()`` heartbeats, terminal
        publish, and ACK/NAK all happen inside it. So a fire-and-forget
        task does not drop acks or heartbeats. The embedding / score /
        extract paths are untouched (they still run synchronously under
        ``_batch_sem`` because they DO contend for the GPU as a batch).
        Tasks are tracked in ``_in_flight_generate_tasks`` for graceful
        drain on :meth:`stop`.

        BUG 3 — guarded dispatch. ``process`` only catches the
        ``msgpack.unpackb`` decode; ANY other exception (model load race,
        publish failure, a bug) escapes the fire-and-forget task. Without a
        guard that left the JetStream message unsettled (→ redelivery storm
        + KV-budget leak via the Bug 2/7 path) AND surfaced as an
        unobserved-task-exception. Wrap the coroutine so an unexpected
        exception is logged with context, counted, and NAKs the message for
        redelivery — mirroring the batch path's ``_guarded_process`` spirit.
        """
        for _, msg in items_msgs:
            task = asyncio.create_task(self._guarded_generate_process(msg, model_id))
            self._in_flight_generate_tasks.add(task)
            task.add_done_callback(self._in_flight_generate_tasks.discard)

    @contextlib.asynccontextmanager
    async def _class_slot(self, op: str) -> AsyncIterator[None]:
        """Hold a work-class scheduler slot for the wrapped work, when the
        opt-in fairness scheduler is enabled; a no-op pass-through otherwise.

        One slot == one in-flight unit of GPU work (a generation stream or a
        batch). The slot is released on exit (including on exception) by the
        scheduler's lease, so a crash never leaks a slot.
        """
        if self._scheduler is None:
            yield
            return
        async with self._scheduler.lease(classify_work_class(op)):
            yield

    async def _run_in_class_slot(self, op: str, coro: Awaitable[None]) -> None:
        """Await ``coro`` while holding its work-class slot (no-op when the
        fairness scheduler is disabled).
        """
        async with self._class_slot(op):
            await coro

    async def _guarded_generate_process(self, msg: Any, model_id: str) -> None:
        """Run ``StreamingProcessor.process`` with a top-level exception guard.

        ``process`` settles the message itself on every *expected* path. This
        guard is the safety net for the *unexpected*: it ensures an escaped
        exception is observed (logged + metric) and the message is NAK'd for
        redelivery rather than left in-flight forever (BUG 3).
        """
        try:
            async with self._class_slot("generate"):
                await self._streaming_processor.process(msg, model_id)
        except Exception:
            logger.exception(
                "Unhandled exception in StreamingProcessor.process for model %s; NAKing for redelivery",
                model_id,
            )
            if _HAS_METRICS:
                PULL_GENERATE_DISPATCH_ERRORS.labels(model=model_id).inc()
            try:
                await msg.nak(delay=_NAK_DELAY_S)
            except Exception:  # noqa: BLE001
                logger.debug("Failed to NAK generate msg after unhandled process exception")

    async def _process_encode_batch(self, model_id: str, items_msgs: list[tuple[WorkItem, Any]]) -> None:
        """Process a batch of encode work items through EncodePipeline."""
        from sie_server.core.encode_pipeline import EncodePipeline  # noqa: PLC0415

        batch_start = time.monotonic()

        # Resolve all item payloads (with fetch timing)
        resolved: list[tuple[WorkItem, Any, Item, float]] = []  # (..., payload_fetch_ms)
        for wi, msg in items_msgs:
            fetch_start = time.monotonic()
            item = await self._resolve_item(wi)
            fetch_ms = (time.monotonic() - fetch_start) * 1000 if wi.get("payload_ref") else 0.0
            if item is None:
                await self._publish_error(wi, "payload_error", "Failed to resolve item payload")
                await msg.ack()
                continue
            resolved.append((wi, msg, item, fetch_ms))

        if not resolved:
            return

        try:
            config = self._registry.get_config(model_id)
        except KeyError:
            # Model config not found or model evicted mid-batch — NAK for redelivery.
            # Another worker (or this one after re-loading) will process them.
            for wi, msg, _, _fm in resolved:
                try:
                    await msg.nak(delay=_NAK_DELAY_S)
                except Exception:  # noqa: BLE001
                    logger.debug("Failed to NAK message for evicted model %s", model_id)
            return

        # Group items by encode params to handle heterogeneous batches.
        # Items from different API requests may have different output_types,
        # instruction, is_query, or options — these cannot be mixed in a
        # single EncodePipeline.run_encode() call.
        groups: dict[tuple, list[tuple[WorkItem, Any, Item, float]]] = {}
        for wi, msg, item, fetch_ms in resolved:
            output_types = tuple(wi.get("output_types") or ["dense"])
            instruction = wi.get("instruction")
            is_query = wi.get("is_query", False)
            # options dict is not hashable — use sorted tuple of items
            options = wi.get("options") or {}
            options_key = msgpack.packb(options, use_bin_type=True) if options else b""
            key = (output_types, instruction, is_query, options_key)
            groups.setdefault(key, []).append((wi, msg, item, fetch_ms))

        # Process each sub-batch
        for (output_types_t, instruction, is_query, options_key), group in groups.items():
            output_types = list(output_types_t)
            options = msgpack.unpackb(options_key, raw=False) if options_key else {}
            all_items = [item for _, _, item, _ in group]
            queue_times = [(time.time() - wi.get("timestamp", time.time())) * 1000 for wi, _, _, _ in group]
            fetch_times = [fm for _, _, _, fm in group]

            try:
                formatted_outputs, timing = await EncodePipeline.run_encode(
                    registry=self._registry,
                    model=model_id,
                    items=all_items,
                    output_types=output_types,
                    instruction=instruction,
                    config=config,
                    is_query=is_query,
                    options=options,
                )

                for idx, (wi, msg, _, _fm) in enumerate(group):
                    output = formatted_outputs[idx] if idx < len(formatted_outputs) else {}
                    # Wrap raw numpy arrays in the DenseVector/SparseVector wire
                    # format that the SDK client expects:
                    #   dense  → {"dims": N, "dtype": str, "values": ndarray}
                    #   sparse → {"indices": ndarray, "values": ndarray}
                    # This matches ``encode.py:_build_response_items`` on the HTTP path.
                    output = _wrap_encode_output(output, config)
                    result_data = msgpack.packb(output, use_bin_type=True)
                    reply_subject = wi.get("reply_subject", "")
                    if reply_subject:
                        result: WorkResult = {
                            "work_item_id": wi.get("work_item_id", ""),
                            "request_id": wi.get("request_id", ""),
                            "item_index": wi.get("item_index", 0),
                            "success": True,
                            "result_msgpack": result_data,
                            "queue_ms": queue_times[idx],
                            "processing_ms": 0.0,
                            "worker_id": self._worker_id,
                        }
                        if timing.inference_ms is not None:
                            result["inference_ms"] = timing.inference_ms
                        if timing.tokenization_ms > 0:
                            result["tokenization_ms"] = timing.tokenization_ms
                        if timing.postprocessing_ms > 0:
                            result["postprocessing_ms"] = timing.postprocessing_ms
                        result["payload_fetch_ms"] = fetch_times[idx]
                        result_bytes = msgpack.packb(result, use_bin_type=True)
                        try:
                            await self._nc.publish(reply_subject, result_bytes)
                        except Exception:  # noqa: BLE001
                            logger.warning("Failed to publish result for %s", wi.get("work_item_id"), exc_info=True)
                            continue  # Don't ACK — let JetStream redeliver

                        # Feed adaptive latency tracker: total queue-path
                        # latency = queue wait + inference + postprocessing.
                        total_ms = queue_times[idx] + (timing.inference_ms or 0) + timing.postprocessing_ms
                        self._queue_latency_tracker.record(total_ms)

                    await msg.ack()

            except asyncio.CancelledError:
                # Worker was evicted (model unloaded) mid-batch — NAK for redelivery
                logger.info("Encode batch cancelled (model %s likely evicted) — NAKing items", model_id)
                for wi, msg, _, _fm in group:
                    try:
                        await msg.nak(delay=_NAK_DELAY_S)
                    except Exception:  # noqa: BLE001
                        logger.debug("Failed to NAK message for evicted model %s", model_id)
            except Exception as e:  # noqa: BLE001
                code, msg_text = self._classify_inference_exception(e)
                logger.warning(
                    "Encode sub-batch failed for model %s (code=%s): %s",
                    model_id,
                    code,
                    e,
                )
                # For non-retryable codes: publish the error result to the
                # gateway's reply subject and ACK the JetStream message —
                # redelivering would just reproduce the same failure.
                #
                # For RESOURCE_EXHAUSTED: skip ``_publish_error`` and only
                # NAK so JetStream can redeliver the work item to a sibling
                # worker (or this worker after memory clears). Publishing
                # the error here would fill the gateway's per-item slot and
                # complete the request immediately, dropping the redelivered
                # work on arrival (the gateway's inbox handler discards
                # replies whose ``request_id`` is no longer pending). The
                # caller still sees a 503 RESOURCE_EXHAUSTED — the gateway
                # synthesises one from result-await timeouts — and the SDK
                # auto-retries; the NAK simply gives non-retrying clients
                # (``max_oom_retries=0``) a real second attempt instead of
                # the wasted compute the previous "publish + NAK"
                # combination produced.
                is_resource_exhausted = code == ErrorCode.RESOURCE_EXHAUSTED.value
                for wi, msg, _, _fm in group:
                    if not is_resource_exhausted:
                        await self._publish_error(wi, code, msg_text)
                    await self._ack_or_nak_after_error(msg, code)

        if _HAS_METRICS:
            elapsed = time.monotonic() - batch_start
            PULL_BATCH_PROCESS_SECONDS.labels(model=model_id, operation="encode").observe(elapsed)

    async def _process_score_batch(self, model_id: str, items_msgs: list[tuple[WorkItem, Any]]) -> None:
        """Process score items concurrently for BatchFormer cross-request batching."""
        batch_start = time.monotonic()

        async def _process_one(wi: WorkItem, msg: Any) -> None:
            try:
                await self._process_single_score(model_id, wi, msg)
            except Exception:  # noqa: BLE001
                logger.warning("Score failed for %s", wi.get("work_item_id"), exc_info=True)
                try:
                    await self._publish_error(wi, "internal_error", "Unexpected processing failure")
                except Exception:  # noqa: BLE001
                    logger.debug("Failed to publish error result for %s", wi.get("work_item_id"))
                try:
                    await msg.ack()
                except Exception:  # noqa: BLE001
                    logger.debug("Failed to ACK message for %s", wi.get("work_item_id"))

        await asyncio.gather(*(_process_one(wi, msg) for wi, msg in items_msgs))

        if _HAS_METRICS:
            elapsed = time.monotonic() - batch_start
            PULL_BATCH_PROCESS_SECONDS.labels(model=model_id, operation="score").observe(elapsed)

    async def _process_extract_batch(self, model_id: str, items_msgs: list[tuple[WorkItem, Any]]) -> None:
        """Process extract items individually but concurrently."""
        batch_start = time.monotonic()

        # Extract items go through worker.submit_extract individually
        # but concurrent submission lets BatchFormer batch them
        async def _process_one(wi: WorkItem, msg: Any) -> None:
            try:
                fetch_start = time.monotonic()
                item = await self._resolve_item(wi)
                fetch_ms = (time.monotonic() - fetch_start) * 1000 if wi.get("payload_ref") else 0.0
                if item is None:
                    await self._publish_error(wi, "payload_error", "Failed to resolve item payload")
                    await msg.ack()
                    return
                await self._process_single_extract(model_id, wi, msg, item, fetch_ms)
            except Exception:  # noqa: BLE001
                logger.warning("Extract failed for %s", wi.get("work_item_id"), exc_info=True)
                try:
                    await self._publish_error(wi, "internal_error", "Unexpected processing failure")
                except Exception:  # noqa: BLE001
                    logger.debug("Failed to publish error result for %s", wi.get("work_item_id"))
                try:
                    await msg.ack()
                except Exception:  # noqa: BLE001
                    logger.debug("Failed to ACK message for %s", wi.get("work_item_id"))

        await asyncio.gather(*(_process_one(wi, msg) for wi, msg in items_msgs))

        if _HAS_METRICS:
            elapsed = time.monotonic() - batch_start
            PULL_BATCH_PROCESS_SECONDS.labels(model=model_id, operation="extract").observe(elapsed)

    async def _process_single_score(
        self,
        model_id: str,
        wi: WorkItem,
        msg: Any,
    ) -> None:
        """Run the score pipeline for a single item, ACK, and publish result."""
        queue_ms = (time.time() - wi.get("timestamp", time.time())) * 1000

        # Start the model worker (lazy loading)
        try:
            worker = await self._registry.start_worker(model_id)
        except (KeyError, RuntimeError) as e:
            logger.info("Model %s not available for score: %s — NAKing", model_id, e)
            await msg.nak(delay=_NAK_DELAY_S)
            return

        query = wi.get("query_item")
        items = wi.get("score_items")

        # Resolve offloaded score payload
        payload_fetch_ms = 0.0
        if query is None and wi.get("query_payload_ref"):
            ref_key: str = wi["query_payload_ref"]  # type: ignore
            fetch_start = time.monotonic()
            payload_data = await self._fetch_payload(ref_key)
            payload_fetch_ms = (time.monotonic() - fetch_start) * 1000
            if payload_data:
                decoded = msgpack.unpackb(payload_data, raw=False)
                query = decoded.get("query")
                items = decoded.get("items")

        if query is None or items is None:
            await self._publish_error(wi, "payload_error", "Missing query or items for score")
            await msg.ack()
            return

        query_item = self._dict_to_item(query) if isinstance(query, dict) else query
        score_items = [self._dict_to_item(it) if isinstance(it, dict) else it for it in items]

        instruction = wi.get("instruction")
        options = wi.get("options") or {}

        try:
            timing = RequestTiming()

            # Build prepared items with cost (query + doc char count)
            query_text = query_item.text
            query_len = len(query_text) if query_text else 0
            timing.start_tokenization()
            prepared_items = []
            for i, it in enumerate(score_items):
                item_text = it.text
                doc_len = len(item_text) if item_text else 0
                prepared_items.append(ScorePreparedItem(cost=query_len + doc_len, original_index=i))
            timing.end_tokenization()

            # Submit to worker and await result
            future = await worker.submit_score(
                prepared_items=prepared_items,
                query=query_item,
                items=score_items,
                instruction=instruction,
                options=options,
                timing=timing,
            )
            worker_result = await future

            # Extract scores and build ranked ScoreEntry list.
            # The gateway wraps this in {"model": ..., "items": <blob>}, so
            # result_data must be the scores list, not a full ScoreResponse.
            score_output: ScoreOutput = worker_result.output  # type: ignore
            raw_scores = [float(score_output.scores[i]) for i in range(score_output.batch_size)]

            scored_items = []
            for i, sc in enumerate(raw_scores):
                item_id = score_items[i].id if score_items[i].id is not None else f"item-{i}"
                scored_items.append((item_id, sc))
            scored_items.sort(key=lambda x: x[1], reverse=True)

            score_entries = [
                {"item_id": item_id, "score": sc, "rank": rank} for rank, (item_id, sc) in enumerate(scored_items)
            ]
            result_data = msgpack.packb(score_entries, use_bin_type=True)
            inference_ms = timing.inference_ms
        except Exception as e:  # noqa: BLE001
            code, msg_text = self._classify_inference_exception(e)
            logger.warning(
                "Score failed for %s (code=%s): %s",
                wi.get("work_item_id"),
                code,
                e,
            )
            # See ``_process_encode_group`` for the rationale: for
            # ``RESOURCE_EXHAUSTED`` we skip ``_publish_error`` and only NAK so
            # JetStream can redeliver to a sibling worker, leaving the
            # gateway's per-item slot free for the retry.
            if code != ErrorCode.RESOURCE_EXHAUSTED.value:
                await self._publish_error(wi, code, msg_text)
            await self._ack_or_nak_after_error(msg, code)
            return

        # Publish result
        reply_subject = wi.get("reply_subject", "")
        if reply_subject and result_data is not None:
            result: WorkResult = {
                "work_item_id": wi.get("work_item_id", ""),
                "request_id": wi.get("request_id", ""),
                "item_index": wi.get("item_index", 0),
                "success": True,
                "result_msgpack": result_data,
                "queue_ms": queue_ms,
                "processing_ms": 0.0,
                "worker_id": self._worker_id,
            }
            if inference_ms is not None:
                result["inference_ms"] = inference_ms
            if timing.tokenization_ms > 0:
                result["tokenization_ms"] = timing.tokenization_ms
            if timing.postprocessing_ms > 0:
                result["postprocessing_ms"] = timing.postprocessing_ms
            if payload_fetch_ms > 0:
                result["payload_fetch_ms"] = payload_fetch_ms
            result_bytes = msgpack.packb(result, use_bin_type=True)
            try:
                await self._nc.publish(reply_subject, result_bytes)
            except Exception:  # noqa: BLE001
                logger.warning("Failed to publish result for %s", wi.get("work_item_id"), exc_info=True)
                return  # Don't ACK — let JetStream redeliver

        await msg.ack()

    async def _process_single_extract(
        self,
        model_id: str,
        wi: WorkItem,
        msg: Any,
        item: Item,
        payload_fetch_ms: float = 0.0,
    ) -> None:
        """Run the extract pipeline for a single item, ACK, and publish result."""
        queue_ms = (time.time() - wi.get("timestamp", time.time())) * 1000

        # Start the model worker (lazy loading)
        try:
            worker = await self._registry.start_worker(model_id)
        except (KeyError, RuntimeError) as e:
            logger.info("Model %s not available for extract: %s — NAKing", model_id, e)
            await msg.nak(delay=_NAK_DELAY_S)
            return

        labels = wi.get("labels")
        output_schema = wi.get("output_schema")
        instruction = wi.get("instruction")
        options = wi.get("options") or {}

        try:
            timing = RequestTiming()

            # Build prepared items with cost (character count for text, byte size for documents)
            timing.start_tokenization()
            prepared_items = build_extract_prepared_items([item])
            timing.end_tokenization()

            # Submit to worker and await result
            future = await worker.submit_extract(
                prepared_items=prepared_items,
                items=[item],
                labels=labels,
                output_schema=output_schema,
                instruction=instruction,
                options=options,
                timing=timing,
            )
            worker_result = await future

            # Format output using ExtractHandler (matches api/extract.py)
            extraction_results = ExtractHandler.format_output(worker_result.output)  # type: ignore
            result_data = msgpack.packb(extraction_results[0] if extraction_results else {}, use_bin_type=True)
            inference_ms = timing.inference_ms
        except Exception as e:  # noqa: BLE001
            code, msg_text = self._classify_inference_exception(e)
            logger.warning(
                "Extract failed for %s (code=%s): %s",
                wi.get("work_item_id"),
                code,
                e,
            )
            # See ``_process_encode_group`` for the rationale: for
            # ``RESOURCE_EXHAUSTED`` we skip ``_publish_error`` and only NAK so
            # JetStream can redeliver to a sibling worker, leaving the
            # gateway's per-item slot free for the retry.
            if code != ErrorCode.RESOURCE_EXHAUSTED.value:
                await self._publish_error(wi, code, msg_text)
            await self._ack_or_nak_after_error(msg, code)
            return

        # Publish result
        reply_subject = wi.get("reply_subject", "")
        if reply_subject and result_data is not None:
            result: WorkResult = {
                "work_item_id": wi.get("work_item_id", ""),
                "request_id": wi.get("request_id", ""),
                "item_index": wi.get("item_index", 0),
                "success": True,
                "result_msgpack": result_data,
                "queue_ms": queue_ms,
                "processing_ms": 0.0,
                "worker_id": self._worker_id,
            }
            if inference_ms is not None:
                result["inference_ms"] = inference_ms
            if timing.tokenization_ms > 0:
                result["tokenization_ms"] = timing.tokenization_ms
            if timing.postprocessing_ms > 0:
                result["postprocessing_ms"] = timing.postprocessing_ms
            if payload_fetch_ms > 0:
                result["payload_fetch_ms"] = payload_fetch_ms
            result_bytes = msgpack.packb(result, use_bin_type=True)
            try:
                await self._nc.publish(reply_subject, result_bytes)
            except Exception:  # noqa: BLE001
                logger.warning("Failed to publish result for %s", wi.get("work_item_id"), exc_info=True)
                return  # Don't ACK — let JetStream redeliver

        await msg.ack()

    async def _resolve_item(self, wi: WorkItem) -> Item | None:
        """Resolve a work item's payload (inline or from payload store)."""
        item_data = wi.get("item")
        if item_data is not None:
            if isinstance(item_data, dict):
                return self._dict_to_item(item_data)
            return item_data

        # Fetch from payload store
        payload_ref = wi.get("payload_ref")
        if payload_ref:
            payload_bytes = await self._fetch_payload(payload_ref)
            if payload_bytes:
                item_dict = msgpack.unpackb(payload_bytes, raw=False)
                return self._dict_to_item(item_dict)

        return None

    @staticmethod
    def _dict_to_item(d: dict) -> Item:
        """Convert a dict (from SDK wire format) to an Item.

        The SDK sends ``content`` while the server ``Item`` struct expects
        ``text``.  Map the field and drop any other unknown keys so that the
        ``Item`` constructor doesn't raise.
        """
        if "content" in d and "text" not in d:
            d = {**d, "text": d.pop("content")}
        known = set(Item.__struct_fields__)
        return Item(**{k: v for k, v in d.items() if k in known})

    async def _fetch_payload(self, ref_key: str) -> bytes | None:
        """Fetch payload from the cached payload store."""
        if self._payload_store is None:
            logger.warning("Payload ref %s but no payload store configured", ref_key)
            return None

        try:
            return await self._payload_store.get(ref_key)
        except KeyError:
            logger.warning("Payload not found: %s", ref_key)
            return None

    async def _ack_or_nak_after_error(self, msg: Any, code: str) -> None:
        """Decide ACK vs NAK after a failed inference handler.

        Retryable codes (currently just ``RESOURCE_EXHAUSTED``) are NAKed
        with a delay so JetStream redelivers the work item — potentially
        to a sibling worker, or to this worker after its memory pressure
        clears. This means a client without SDK auto-retry (or with
        ``max_oom_retries=0``) still gets the work executed eventually,
        instead of silently losing it.

        Non-retryable codes (the legacy ``"inference_error"`` literal,
        ``"unknown_operation"``, ``"payload_error"``, ``"internal_error"``)
        are ACKed: redelivering them would just reproduce the same failure.

        Behaviour on delivery-status failures:
        - ACK exceptions are swallowed at debug level — the error response
          has already been published, and JetStream's ack-wait will
          eventually redeliver, which is harmless for non-retryable
          failures (the redelivered request will fail the same way and
          ACK again).
        - NAK exceptions on the ``RESOURCE_EXHAUSTED`` path are
          *deliberately not* followed by an ACK fallback. Falling through
          to ACK would silently drop the work item — JetStream would
          consider it delivered and never retry — which is exactly the
          opposite of the retryable contract. Instead, we log the NAK
          failure and return without ACKing, leaving the message unacked
          so JetStream redelivers it after ``ack_wait`` expires.
        """
        if code == ErrorCode.RESOURCE_EXHAUSTED.value:
            try:
                await msg.nak(delay=_OOM_NAK_DELAY_S)
            except Exception:  # noqa: BLE001
                # Do NOT fall through to ACK — that would silently drop
                # the work item. Leave it unacked; JetStream will
                # redeliver after ack_wait expires.
                logger.warning(
                    "Failed to NAK RESOURCE_EXHAUSTED message; leaving unacked for JetStream redelivery",
                    exc_info=True,
                )
            return
        try:
            await msg.ack()
        except Exception:  # noqa: BLE001
            logger.debug("Failed to ACK message after error", exc_info=True)

    @staticmethod
    def _classify_inference_exception(exc: BaseException) -> tuple[str, str]:
        """Map an inference-path exception to ``(error_code, error_msg)``.

        Used by the queue (NATS) path's catch-all error handler so the
        gateway — which translates ``WorkResult.error_code`` into HTTP
        responses — can emit the same retryable contract the HTTP path
        emits via ``InferenceErrorHandler``.

        Currently distinguishes:
        - OOM (any flavour, including ``ResourceExhaustedError``) →
          ``RESOURCE_EXHAUSTED`` (gateway maps to 503 + Retry-After,
          SDK auto-retries).
        - Malformed media input (``InvalidMediaError``, e.g. an un-decoded
          base64 ``str`` reaching a preprocessor/adapter) → ``INVALID_INPUT``
          (gateway maps to 400; the caller's payload is wrong, not retryable).
        - Everything else → the legacy ``"inference_error"`` literal so
          existing dashboards / alerts keep working.

        Note the case mismatch: ``"inference_error"`` (lowercase) is the
        wire value the queue path has used historically; ``ErrorCode``
        enum values are uppercase. Aligning the legacy literal is out of
        scope here — see follow-up work.
        """
        if is_oom_error(exc):
            return ErrorCode.RESOURCE_EXHAUSTED.value, str(exc)
        if isinstance(exc, InvalidMediaError):
            return ErrorCode.INVALID_INPUT.value, str(exc)
        return "inference_error", str(exc)

    async def _publish_error(self, wi: WorkItem, error_code: str, error_msg: str) -> None:
        """Publish an error result to the reply subject."""
        reply_subject = wi.get("reply_subject", "")
        if not reply_subject:
            return

        result: WorkResult = {
            "work_item_id": wi.get("work_item_id", ""),
            "request_id": wi.get("request_id", ""),
            "item_index": wi.get("item_index", 0),
            "success": False,
            "error": error_msg,
            "error_code": error_code,
            "worker_id": self._worker_id,
        }
        result_bytes = msgpack.packb(result, use_bin_type=True)
        try:
            await self._nc.publish(reply_subject, result_bytes)
        except Exception:  # noqa: BLE001
            logger.warning("Failed to publish error result for %s", wi.get("work_item_id"))
