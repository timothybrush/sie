"""Prometheus metrics for SIE Server.

Exposes metrics for monitoring request throughput, latency, batching efficiency,
and resource utilization.

Metrics follow Prometheus naming conventions:
- sie_ prefix for all metrics
- _total suffix for counters
- _seconds suffix for duration histograms
- _bytes suffix for memory metrics
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from prometheus_client import Counter, Gauge, Histogram

if TYPE_CHECKING:
    from sie_server.core.timing import RequestTiming

logger = logging.getLogger(__name__)

# Histogram buckets for request duration (in seconds)
# Covers 1ms to 30s range, optimized for inference workloads
DURATION_BUCKETS = (
    0.001,  # 1ms
    0.005,  # 5ms
    0.01,  # 10ms
    0.025,  # 25ms
    0.05,  # 50ms
    0.1,  # 100ms
    0.25,  # 250ms
    0.5,  # 500ms
    1.0,  # 1s
    2.5,  # 2.5s
    5.0,  # 5s
    10.0,  # 10s
    30.0,  # 30s
)

# Shared histogram buckets for generation TTFT/TPOT, identical on the
# worker and gateway sides so the dashboard's "gateway minus worker"
# overhead-attribution panel subtracts buckets with the same edges.
# MUST stay in sync with ``packages/sie_gateway/src/metrics.rs``
# ``TTFT_TPOT_BUCKETS`` (per the metrics rollout's acceptance criterion).
TTFT_TPOT_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
)


# Histogram buckets for batch sizes
BATCH_SIZE_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)

# Histogram buckets for token counts
TOKEN_BUCKETS = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)


# -----------------------------------------------------------------------------
# Request Metrics
# -----------------------------------------------------------------------------

REQUESTS_TOTAL = Counter(
    "sie_requests_total",
    "Total number of requests processed",
    ["model", "endpoint", "status"],
)

REQUEST_DURATION = Histogram(
    "sie_request_duration_seconds",
    "Request duration breakdown by phase",
    ["model", "endpoint", "phase"],
    buckets=DURATION_BUCKETS,
)


# -----------------------------------------------------------------------------
# Batching Metrics
# -----------------------------------------------------------------------------

BATCH_SIZE = Histogram(
    "sie_batch_size",
    "Number of items per batch",
    ["model"],
    buckets=BATCH_SIZE_BUCKETS,
)

TOKENS_PROCESSED = Counter(
    "sie_tokens_processed_total",
    "Total number of tokens processed",
    ["model"],
)


# -----------------------------------------------------------------------------
# Generation Streaming Metrics
# -----------------------------------------------------------------------------
#
# Adapter-level time-to-first-token and mean time-per-output-token,
# observed inside the streaming adapter itself (before any NATS or
# StreamingProcessor overhead). The dashboard subtracts these from
# the gateway's ``sie_gateway_generation_ttft_seconds`` /
# ``sie_gateway_generation_tpot_seconds`` to attribute overhead.
#
# Bucket choices use TTFT_TPOT_BUCKETS — the same edges the gateway
# uses, so subtraction is honest.

GENERATION_TTFT = Histogram(
    "sie_worker_generation_ttft_seconds",
    "Adapter-observed time-to-first-token (start of generate() to first non-empty yield)",
    ["model", "grammar"],
    buckets=TTFT_TPOT_BUCKETS,
)

GENERATION_TPOT = Histogram(
    "sie_worker_generation_tpot_seconds",
    "Adapter-observed mean time-per-output-token (first yield to terminal yield divided by completion tokens)",
    ["model", "grammar"],
    buckets=TTFT_TPOT_BUCKETS,
)


# -----------------------------------------------------------------------------
# Grammar / Structured Outputs
# -----------------------------------------------------------------------------
#
# Compile latency is the per-(tokenizer, schema) cost of validating a
# grammar via Outlines. Cache hits skip the compile entirely; misses
# observe the histogram below before populating the cache. The
# ``kind`` label distinguishes ``json_schema`` (typically slow,
# proportional to schema complexity) from ``regex`` (typically fast).
#
# A 5s budget at the worker (``asyncio.wait_for`` in
# :meth:`StreamingProcessor.process`) is the contractual upper bound;
# the bucket edges below sit comfortably below that so a near-timeout
# compile is visible in the histogram before it actually times out.

GRAMMAR_COMPILE_SECONDS = Histogram(
    "sie_worker_grammar_compile_seconds",
    "Outlines grammar compile wall-clock time (cache miss path only)",
    ["model", "kind"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)

GRAMMAR_CACHE_HITS = Counter(
    "sie_worker_grammar_cache_hits_total",
    "Grammar LRU lookups that returned a cached compile result",
    ["model"],
)

GRAMMAR_CACHE_MISSES = Counter(
    "sie_worker_grammar_cache_misses_total",
    "Grammar LRU lookups that triggered a fresh compile",
    ["model"],
)

# Pre-warmed grammar compiles performed at worker boot from
# ``tasks.generate.prewarm_grammars`` in the model config. Distinct from
# the request-path compile counters because:
#
# * a prewarm miss is *intended* (no traffic has populated the cache yet)
# * a prewarm failure must NOT block model load — operators see it via
#   ``outcome="failed"`` rather than as a startup crash
# * the wall-clock histogram lets operators size the prewarm-grammar list
#   without exceeding the worker's start-up budget
#
# Same bucket edges as ``GRAMMAR_COMPILE_SECONDS`` so dashboards can
# overlay request-path vs. prewarm compile latency.
GRAMMAR_PREWARM_SECONDS = Histogram(
    "sie_worker_grammar_prewarm_seconds",
    "Wall-clock duration of pre-warmed grammar compiles (model-load phase)",
    ["model", "kind"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)

GRAMMAR_PREWARM_TOTAL = Counter(
    "sie_worker_grammar_prewarm_total",
    "Pre-warmed grammar compiles attempted at model load",
    ["model", "kind", "outcome"],
)

# -----------------------------------------------------------------------------
# Structured-output grammar metrics
# -----------------------------------------------------------------------------
#
# The worker-side Outlines preflight is no longer on the request hot
# path; SGLang's server-side grammar backend (Outlines,
# xgrammar, or llguidance) is the single authority. These metrics give
# operators visibility into structured-output behaviour from the
# worker's side without re-introducing the duplicate preflight.
#
# Labels:
#   * ``backend`` — the SGLang grammar backend identifier
#     (``outlines`` / ``xgrammar`` / ``llguidance`` / ``unknown``).
#     Resolved from the active adapter's ``_grammar_backend`` so the
#     metrics partition by what SGLang is actually doing.
#   * ``mode`` — the grammar kind on the wire
#     (``json_schema`` / ``regex`` / ``ebnf``).
#
# ``sie_grammar_compile_seconds`` only fires when the legacy worker
# preflight is enabled via ``SIE_GRAMMAR_PREFLIGHT_DEBUG=1``. In
# production traffic it stays at zero observations. The structured-
# output TTFT histogram is the proxy for SGLang's own internal compile
# cost — when SGLang has to construct an FSM for a unique schema, the
# first-token latency for that request inflates accordingly.
GRAMMAR_COMPILE_SECONDS_STRUCTURED_OUTPUT = Histogram(
    "sie_grammar_compile_seconds",
    "Worker-side preflight grammar-compile duration (only emitted when SIE_GRAMMAR_PREFLIGHT_DEBUG=1; off by default)",
    ["backend", "mode"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)

STRUCTURED_OUTPUT_TTFT_SECONDS = Histogram(
    "sie_structured_output_ttft_seconds",
    "Time-to-first-token for structured-output requests, proxy for SGLang's server-side grammar-construction cost",
    ["backend", "mode"],
    buckets=TTFT_TPOT_BUCKETS,
)

GRAMMAR_CACHE_HITS_STRUCTURED_OUTPUT = Counter(
    "sie_grammar_cache_hits_total",
    "Worker-side grammar-cache hits (preflight cache; only fires when SIE_GRAMMAR_PREFLIGHT_DEBUG=1)",
    ["backend"],
)

GRAMMAR_CACHE_MISSES_STRUCTURED_OUTPUT = Counter(
    "sie_grammar_cache_misses_total",
    "Worker-side grammar-cache misses (preflight cache; only fires when SIE_GRAMMAR_PREFLIGHT_DEBUG=1)",
    ["backend"],
)

GRAMMAR_UNIQUE_SCHEMA_TOTAL = Counter(
    "sie_grammar_unique_schema_total",
    "Lifetime count of unique (backend, mode, schema-hash) tuples observed "
    "on the worker — diagnostic signal for unique-schema burst workloads",
    ["backend", "mode"],
)


# -----------------------------------------------------------------------------
# Generation Admission Control
# -----------------------------------------------------------------------------
#
# Admission is OOM-protection, not performance: the rejection counter
# is alerting signal (a sustained non-zero rate means workers are
# undersized for the offered load). The two gauges describe the
# instantaneous shape of the reserved KV-cache fraction and are
# emitted on every reserve/release **regardless** of the admission
# feature flag — the routing saturation gate reads
# ``kv_reserved / kv_budget`` even when admission itself is off.

GENERATION_ADMISSION_REJECTED = Counter(
    "sie_worker_generation_admission_rejected_total",
    "Generation work items rejected by the per-worker admission controller",
    ["model", "reason"],
)

GENERATION_KV_RESERVED_TOKENS = Gauge(
    "sie_worker_generation_kv_reserved_tokens",
    "Currently-reserved KV-cache tokens across all in-flight generations on this worker",
    ["model"],
)

GENERATION_IN_FLIGHT = Gauge(
    "sie_worker_generation_in_flight",
    "Currently-streaming generation requests on this worker",
    ["model"],
)

# Bumped when ``StreamingProcessor._resolve_admission_for_model``
# catches an exception from the operator-supplied admission resolver.
# Resolver failures silently disable admission for the affected model,
# so this counter gives operators an alert path on a buggy resolver
# rolling out.
GENERATION_ADMISSION_RESOLVER_ERRORS = Counter(
    "sie_worker_generation_admission_resolver_errors_total",
    "Exceptions raised by the per-model generation admission resolver",
    ["model"],
)

# H9 — cancel-tombstone observability.
#
# Bumped when the StreamingProcessor's decode-start tombstone check
# fires: a cancel for ``request_id`` arrived BEFORE any decode
# attempt had registered its in-flight handle (the
# direct-dispatch-then-first-chunk-timeout race), and the work
# item finally reached this worker. Without the tombstone both the
# original direct-dispatch and the pool-republished attempt would
# decode in parallel, doubling GPU work, KV reserve, and billing.
# Each increment = one prevented double-execution. Operators alert
# on a sustained non-zero rate per (model, pool) — that's a sign
# either the first-chunk window is too tight, the direct-dispatch
# target is genuinely cold, or both. See
# ``docs/adr/0003-generation-timeouts-bypass-global-ceiling.md``.
GENERATION_FALLBACK_DUPLICATE_TOTAL = Counter(
    "sie_generation_fallback_duplicate_total",
    "First-chunk fallback double-execution prevented by the worker's cancel tombstone (H9)",
    ["model", "pool"],
)


# -----------------------------------------------------------------------------
# Saturation Signal
# -----------------------------------------------------------------------------
#
# The gateway routes around saturated workers. The worker exposes the
# current saturation flag as a 0/1 gauge so operators can see, at a
# glance, which models are pushing back on the gateway. Toggling is
# owned by the worker-side admission controller; this module owns only
# the metric definition. Default value 0 is set at import time so the
# family is non-empty on /metrics even before the first admission
# decision.

GENERATION_SATURATED = Gauge(
    "sie_worker_generation_saturated",
    "Current per-model saturation flag (1 = at/above high-water mark, 0 = below)",
    ["model"],
)


# -----------------------------------------------------------------------------
# Speculative Decoding
# -----------------------------------------------------------------------------
#
# The speculative-decoding investigation probes a speculative side-cell.
# When that cell is active it emits its token-acceptance ratio here.
# Defined here in the metrics module so the dashboard can ship a panel
# that simply renders empty when the probe is not running.

SPECULATIVE_ACCEPTANCE_RATE = Gauge(
    "sie_worker_speculative_acceptance_rate",
    "Speculative-decoding token-acceptance ratio (only emitted when a speculative side-cell is active)",
    ["model"],
)


# -----------------------------------------------------------------------------
# Queue Metrics
# -----------------------------------------------------------------------------

QUEUE_DEPTH = Gauge(
    "sie_queue_depth",
    "Current number of pending items in queue",
    ["model"],
)

# IPC batch shape metrics expose how one sidecar batch fragments into
# Python/GPU forward passes. Bucket sizes match `sie_gpu_batch_items`
# so dashboards can compare IPC and GPU batch shapes directly.
_IPC_BATCH_SIZE_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

IPC_BATCH_ITEMS = Histogram(
    "sie_ipc_batch_items",
    "Number of items in a single IPC batch received from the worker-sidecar",
    ["model", "endpoint"],
    buckets=_IPC_BATCH_SIZE_BUCKETS,
)

IPC_BATCH_SUB_GROUPS = Histogram(
    "sie_ipc_batch_sub_groups",
    "Number of sub-groups (separate GPU forward passes) from one IPC batch",
    ["model", "endpoint"],
    buckets=_IPC_BATCH_SIZE_BUCKETS,
)

IPC_BATCH_SUB_GROUP_ITEMS = Histogram(
    "sie_ipc_batch_sub_group_items",
    "Items per sub-group after IPC-batch splitting",
    ["model", "endpoint"],
    buckets=_IPC_BATCH_SIZE_BUCKETS,
)


# -----------------------------------------------------------------------------
# Model Metrics
# -----------------------------------------------------------------------------

MODEL_LOADED = Gauge(
    "sie_model_loaded",
    "Whether a model is currently loaded (1=loaded, 0=not loaded)",
    ["model", "device"],
)

MODEL_MEMORY_BYTES = Gauge(
    "sie_model_memory_bytes",
    "Estimated GPU memory usage for a loaded model in bytes",
    ["model", "device"],
)

MODEL_LOAD_TIMEOUTS = Counter(
    "sie_model_load_timeouts_total",
    "Number of post-download model-load timeouts, broken down by stage. "
    "The stage label is one of: ``instantiate`` (adapter object construction) or "
    "``load`` (adapter.load + warmup). Download is bounded separately by "
    "HF_HUB_DOWNLOAD_TIMEOUT and does NOT increment this counter.",
    ["model", "stage"],
)


# -----------------------------------------------------------------------------
# OOM Recovery Metrics
# -----------------------------------------------------------------------------
#
# These mirror ``sie_server.core.oom.OomRecoveryStats`` and are bumped from
# inside ``BatchExecutor`` whenever the corresponding strategy fires. Operators
# use them to:
#   * Detect a sustained recovery rate (= a real memory leak or undersized
#     pool, not just transient pressure).
#   * Tune ``SIE_OOM_RECOVERY__MAX_SPLIT_DEPTH`` based on how often
#     ``batch_splits`` fires relative to ``terminal_failures``.
#   * Dashboard "recovery saved this many requests" via
#     ``recoveries_succeeded`` (analytics value separate from
#     ``terminal_failures``, which is the alert signal).

OOM_RECOVERIES_ATTEMPTED = Counter(
    "sie_oom_recoveries_attempted_total",
    "Number of OOM events caught at the worker dispatch boundary",
    ["model"],
)

OOM_RECOVERIES_SUCCEEDED = Counter(
    "sie_oom_recoveries_succeeded_total",
    "OOM recovery attempts that fully succeeded (every metadata got a result)",
    ["model"],
)

OOM_TERMINAL_FAILURES = Counter(
    "sie_oom_terminal_failures_total",
    "OOM events where every recovery strategy was exhausted (clients see RESOURCE_EXHAUSTED)",
    ["model"],
)

OOM_CACHE_CLEARS = Counter(
    "sie_oom_cache_clears_total",
    "Cache-clear recovery actions executed",
    ["model"],
)

OOM_EVICTIONS_TRIGGERED = Counter(
    "sie_oom_evictions_triggered_total",
    "Sibling-model evictions performed during OOM recovery",
    ["model"],
)

OOM_BATCH_SPLITS = Counter(
    "sie_oom_batch_splits_total",
    "Top-level batch-split recovery invocations (recursive halves not counted separately)",
    ["model"],
)


# -----------------------------------------------------------------------------
# Idle Eviction Metrics
# -----------------------------------------------------------------------------
#
# Bumped from ``ModelRegistry._idle_evict_loop`` each time a cold model is
# unloaded by the proactive idle-TTL evictor. Different from
# ``sie_oom_evictions_triggered_total`` (which is the reactive sibling-eviction
# step inside ``BatchExecutor``) — this metric tracks the proactive cleanup
# loop and is what operators key on when validating ``SIE_IDLE_EVICT_S``.

IDLE_EVICTIONS_TOTAL = Counter(
    "sie_idle_evictions_total",
    "Models unloaded by the proactive idle-TTL evictor",
    ["model"],
)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


_MIN_MASKABLE_SECRET_LEN = 8


def _mask_secret(secret: str) -> str:
    """Mask a credential for logging: keep the last 4 chars, redact the rest.

    Short values are fully redacted so we never reveal a meaningful prefix.
    """
    if len(secret) <= _MIN_MASKABLE_SECRET_LEN:
        return "***"
    return f"***{secret[-4:]}"


def record_request(
    model: str,
    endpoint: str,
    status: str,
    timing: RequestTiming | None = None,
    *,
    request_id: str | None = None,
    api_key: str | None = None,
    queue_depth: int | None = None,
) -> None:
    """Record metrics and structured log for a completed request.

    Args:
        model: Model name.
        endpoint: Endpoint name (encode, score, extract).
        status: Request status (success, error, queue_full).
        timing: Optional timing information for latency breakdown.
        request_id: Optional request ID for tracing.
        api_key: Optional API key (masked) for audit.
        queue_depth: Optional current queue depth at request time.
    """
    # Increment request counter
    REQUESTS_TOTAL.labels(model=model, endpoint=endpoint, status=status).inc()

    # Record latency breakdown if timing available
    if timing is not None:
        # Total duration
        REQUEST_DURATION.labels(model=model, endpoint=endpoint, phase="total").observe(
            timing.total_ms / 1000
        )  # Convert ms to seconds

        # Queue time (if tracked)
        if timing.queue_ms > 0:
            REQUEST_DURATION.labels(model=model, endpoint=endpoint, phase="queue").observe(timing.queue_ms / 1000)

        # Tokenization time (if tracked)
        if timing.tokenization_ms > 0:
            REQUEST_DURATION.labels(model=model, endpoint=endpoint, phase="tokenize").observe(
                timing.tokenization_ms / 1000
            )

        # Inference time (if tracked)
        if timing.inference_ms > 0:
            REQUEST_DURATION.labels(model=model, endpoint=endpoint, phase="inference").observe(
                timing.inference_ms / 1000
            )

    # Emit structured log for observability (Loki, etc.)
    log_extra: dict[str, object] = {"model": model, "endpoint": endpoint, "status": status}
    if request_id is not None:
        log_extra["request_id"] = request_id
    if api_key is not None:
        # Mask defensively: never emit a raw bearer token to centralized
        # logs (Loki), regardless of whether the caller pre-masked it.
        log_extra["api_key"] = _mask_secret(api_key)
    if queue_depth is not None:
        log_extra["queue_depth"] = queue_depth
    if timing is not None:
        log_extra["latency_ms"] = timing.total_ms
        log_extra["tokenization_ms"] = timing.tokenization_ms
        log_extra["queue_ms"] = timing.queue_ms
        log_extra["inference_ms"] = timing.inference_ms
    logger.debug("Request completed", extra=log_extra)


def record_batch(model: str, batch_size: int, tokens: int) -> None:
    """Record metrics for a processed batch.

    Args:
        model: Model name.
        batch_size: Number of items in the batch.
        tokens: Total tokens in the batch.
    """
    BATCH_SIZE.labels(model=model).observe(batch_size)
    TOKENS_PROCESSED.labels(model=model).inc(tokens)


def set_queue_depth(model: str, depth: int) -> None:
    """Update the queue depth gauge for a model.

    Args:
        model: Model name.
        depth: Current queue depth.
    """
    QUEUE_DEPTH.labels(model=model).set(depth)


def record_ipc_batch_shape(
    model: str,
    endpoint: str,
    total_items: int,
    sub_group_sizes: list[int],
) -> None:
    """Record the IPC batch shape before Python dispatches GPU work."""
    if total_items <= 0:
        return
    IPC_BATCH_ITEMS.labels(model=model, endpoint=endpoint).observe(total_items)
    IPC_BATCH_SUB_GROUPS.labels(model=model, endpoint=endpoint).observe(len(sub_group_sizes))
    for size in sub_group_sizes:
        # Zero-size entries are invalid; skip them rather than recording
        # misleading 0-bucket samples.
        if size > 0:
            IPC_BATCH_SUB_GROUP_ITEMS.labels(model=model, endpoint=endpoint).observe(size)


def set_model_loaded(model: str, device: str, loaded: bool) -> None:
    """Update the model loaded gauge.

    Args:
        model: Model name.
        device: Device the model is loaded on.
        loaded: Whether the model is loaded.
    """
    MODEL_LOADED.labels(model=model, device=device).set(1 if loaded else 0)


def set_model_memory(model: str, device: str, memory_bytes: int) -> None:
    """Update the model memory gauge.

    Args:
        model: Model name.
        device: Device the model is loaded on.
        memory_bytes: Estimated GPU memory usage in bytes.
    """
    MODEL_MEMORY_BYTES.labels(model=model, device=device).set(memory_bytes)


def increment_model_load_timeout(model: str, stage: str) -> None:
    """Increment the ``sie_model_load_timeouts_total`` counter.

    Args:
        model: Model name.
        stage: One of ``instantiate`` or ``load``. Download stalls are
            handled by ``huggingface_hub`` and are not counted here.
    """
    MODEL_LOAD_TIMEOUTS.labels(model=model, stage=stage).inc()


def record_oom_recovery_event(
    model: str,
    *,
    action: str | None = None,
    attempted: bool = False,
    succeeded: bool = False,
    terminal: bool = False,
) -> None:
    """Bump the appropriate OOM recovery counters for ``model``.

    Centralises the prom-metric writes so ``BatchExecutor`` doesn't have
    to know about the prometheus client. Multiple flags can be set in a
    single call (e.g., ``attempted=True`` plus ``succeeded=True``) to
    record both events for the same recovery cycle.

    Args:
        model: Model name (becomes the ``model`` label).
        action: Strategy action name when bumping a per-strategy counter.
            One of ``"cache_clear"``, ``"evict_lru"``, or ``"split_batch"``;
            ``None`` skips the per-strategy bump.
        attempted: Bump ``sie_oom_recoveries_attempted_total``.
        succeeded: Bump ``sie_oom_recoveries_succeeded_total``.
        terminal: Bump ``sie_oom_terminal_failures_total``.
    """
    if attempted:
        OOM_RECOVERIES_ATTEMPTED.labels(model=model).inc()
    if succeeded:
        OOM_RECOVERIES_SUCCEEDED.labels(model=model).inc()
    if terminal:
        OOM_TERMINAL_FAILURES.labels(model=model).inc()
    if action is None:
        return
    if action == "cache_clear":
        OOM_CACHE_CLEARS.labels(model=model).inc()
    elif action == "evict_lru":
        OOM_EVICTIONS_TRIGGERED.labels(model=model).inc()
    elif action == "split_batch":
        OOM_BATCH_SPLITS.labels(model=model).inc()
    else:
        # Fail fast: silently dropping unknown actions causes per-strategy
        # metrics to drift out of sync with the executor's real behaviour
        # (e.g., a typo at the call site in ``BatchExecutor`` would never
        # surface as a missing-counter alert).
        msg = f"record_oom_recovery_event: unknown action {action!r} for model {model!r}"
        raise ValueError(msg)


class GenerationStreamTimer:
    """Stateful helper that turns a stream of adapter yields into one
    TTFT observation and one TPOT observation per request.

    The streaming adapter calls :meth:`mark_yield` once per
    :class:`GenerationChunk` it produces (including the terminal one),
    then :meth:`finalize` after the loop exits. The timer:

    * Observes ``sie_worker_generation_ttft_seconds`` on the first
      non-empty yield (i.e. when ``has_text`` is True for the first
      time).
    * Records the wall-clock window from first non-empty yield to the
      *last* observed yield, then on :meth:`finalize` observes that
      window divided by ``completion_tokens`` (or by the number of
      non-empty yields seen if the terminal chunk did not supply a
      token count) into ``sie_worker_generation_tpot_seconds``.

    Centralising the bookkeeping here keeps the adapter free of
    per-yield ``time.perf_counter`` plumbing and gives us a single
    unit-testable surface.

    Note: instances use ``__slots__`` for per-request allocation
    economy. The five private attributes listed there are the entire
    state of a timer — adding a new field requires updating
    ``__slots__`` too, or the assignment will raise at runtime. Tests
    write directly to these slots to pin deterministic timing.
    """

    __slots__ = (
        "_completion_yields",
        "_first_yield_at",
        "_grammar",
        "_last_yield_at",
        "_model",
        "_started_at",
    )

    def __init__(self, model: str, *, grammar: str = "none") -> None:
        """Create a timer for one streaming generation.

        Args:
            model: Model name for the histogram label.
            grammar: Bounded-cardinality label value — one of ``none``,
                ``json_schema``, ``regex``. The streaming processor
                derives this from the work envelope's grammar spec
                before instantiating the timer.
        """
        self._model = model
        self._grammar = grammar
        self._started_at = time.perf_counter()
        self._first_yield_at: float | None = None
        self._last_yield_at: float | None = None
        self._completion_yields = 0

    def mark_yield(self, *, has_text: bool) -> None:
        """Record a yield from the adapter. ``has_text`` is True iff the
        yielded chunk carries a non-empty ``text_delta``.
        """
        now = time.perf_counter()
        if has_text:
            if self._first_yield_at is None:
                self._first_yield_at = now
                # First non-empty yield → TTFT observation. We do this
                # eagerly (not in finalize()) so the metric is correct
                # even if the iterator is cancelled before terminal.
                GENERATION_TTFT.labels(model=self._model, grammar=self._grammar).observe(now - self._started_at)
            self._last_yield_at = now
            self._completion_yields += 1

    def finalize(self, *, completion_tokens: int | None = None) -> None:
        """Compute TPOT and observe it. Safe to call zero or one times.

        ``completion_tokens`` (from the terminal chunk's usage) is the
        preferred denominator. Falls back to the number of non-empty
        yields when the terminal chunk did not report token counts.
        """
        if self._first_yield_at is None or self._last_yield_at is None:
            # No non-empty yields observed (timeout / error path); no
            # TPOT observation. TTFT is also absent in this case.
            return
        window = max(self._last_yield_at - self._first_yield_at, 0.0)
        denominator = completion_tokens if completion_tokens and completion_tokens > 0 else self._completion_yields
        if denominator <= 0:
            return
        GENERATION_TPOT.labels(model=self._model, grammar=self._grammar).observe(window / denominator)


def record_idle_eviction(model: str) -> None:
    """Bump ``sie_idle_evictions_total`` for ``model``.

    Called from the registry's idle-evict loop after a successful unload.
    """
    IDLE_EVICTIONS_TOTAL.labels(model=model).inc()


def set_generation_saturated(model: str, *, saturated: bool) -> None:
    """Toggle ``sie_worker_generation_saturated`` for ``model``.

    Owned conceptually by the worker-side admission controller; the metric
    itself lives here so the observability surface is one file. Use
    `0`/`1` semantics (a gauge rather than a counter) so the dashboard
    can render a per-model timeline of saturation pressure.
    """
    GENERATION_SATURATED.labels(model=model).set(1 if saturated else 0)


def set_speculative_acceptance_rate(model: str, rate: float) -> None:
    """Update ``sie_worker_speculative_acceptance_rate`` for ``model``.

    Only the speculative-decoding side-cell calls this. The
    metric remains defined alongside the rest of the §4.11 family so
    the dashboard can ship a panel that renders empty until the probe is active.

    The acceptance rate is mathematically constrained to ``[0, 1]``;
    an out-of-range value indicates a caller bug (e.g. an inverted
    accept/total ratio in the probe). Log a warning so the bug
    surfaces, then clamp so the dashboard doesn't render garbage.
    """
    if not 0.0 <= rate <= 1.0:
        logger.warning(
            "speculative_acceptance_rate=%.6f out of [0, 1] for model=%s; clamping",
            rate,
            model,
        )
    normalized = max(0.0, min(1.0, rate))
    SPECULATIVE_ACCEPTANCE_RATE.labels(model=model).set(normalized)
