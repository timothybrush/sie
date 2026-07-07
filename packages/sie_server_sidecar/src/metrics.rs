//! Prometheus metrics + HTTP /metrics + /healthz server.
//!
//! Port defaults to 9095. The dispatcher and scheduler register the
//! counters/histograms used by the worker hot path.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;

use http_body_util::Full;
use hyper::body::Bytes;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use prometheus::{
    Encoder, Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge, Opts,
    Registry, TextEncoder,
};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tracing::{error, info, warn};

use crate::readiness::Readiness;
use crate::shutdown::Shutdown;

/// Histogram buckets for items-fetched-per-pull-cycle.
const ITEMS_FETCHED_BUCKETS: &[f64] = &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0];

/// Histogram buckets for batch-process-seconds.
const BATCH_PROCESS_BUCKETS: &[f64] = &[
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
];

/// Finer buckets for sub-millisecond IPC RPCs (UDS to the local backend).
const IPC_REQUEST_BUCKETS: &[f64] = &[
    0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
    5.0, 10.0, 30.0,
];

/// Buckets for payload bytes (request/response frames).
const BYTES_BUCKETS: &[f64] = &[
    256.0,
    1024.0,
    4096.0,
    16_384.0,
    65_536.0,
    262_144.0,
    1_048_576.0,
    4_194_304.0,
    16_777_216.0,
    67_108_864.0,
];

/// Buckets for drain-duration-seconds (graceful shutdown timing).
const DRAIN_DURATION_BUCKETS: &[f64] = &[
    0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 60.0,
];

/// Buckets for NATS JetStream redelivery counts (capped at 10).
const DELIVER_COUNT_BUCKETS: &[f64] = &[1.0, 2.0, 3.0, 5.0, 10.0];

/// Prometheus metrics shared by the worker.
///
/// The `sie_worker_*` surface is internal/Rust-only; the `sie_pull_loop_*`
/// surface preserves the metric names used by existing Grafana dashboards
/// that predate the sidecar-owned scheduler.
///
/// ## Label cardinality
///
/// Metric labels that carry a `model` dimension use the full model_id
/// by default. For deployments that worry about cardinality (very
/// wide model fleets, untrusted model_ids, multi-tenant clusters),
/// set `SIE_METRIC_MODEL_ALLOWLIST` to the active/materialized
/// model ids; anything outside that set will be re-labeled `"other"`.
/// This is applied via [`MetricsRegistry::model_label`] — use it
/// everywhere a `model` label value is constructed.
pub struct MetricsRegistry {
    pub registry: Registry,
    pub messages_received_total: IntCounter,
    pub messages_acked_total: IntCounter,
    pub messages_naked_total: IntCounter,
    pub pool_admission_naks_total: IntCounterVec,
    pub ipc_requests_total: IntCounter,
    pub ipc_failures_total: IntCounter,
    pub inflight_batches: IntGauge,

    pub pull_items_fetched: HistogramVec,
    pub pull_batch_process_seconds: HistogramVec,
    pub pull_nak_unloaded_total: IntCounterVec,
    pub pull_model_loads_total: IntCounterVec,
    pub generate_model_loading_responses_total: IntCounterVec,

    // Per-backend visibility. These measure what actually landed in
    // each backend after routing, so operators can compare native and
    // Python batch rates, outcome counts, and backend latency.
    pub backend_process_seconds: HistogramVec,
    pub backend_batch_items: HistogramVec,
    pub backend_process_errors_total: IntCounterVec,
    pub backend_ensure_ready_seconds: HistogramVec,
    pub backend_item_outcomes_total: IntCounterVec,

    /// Backend timing breakdown for successful backend round-trips.
    /// Observations are populated from per-item `inference_ms` /
    /// `tokenization_ms` / `postprocessing_ms` fields on each
    /// `ItemOutcome`.
    ///
    /// IPC backends report the backend sub-group wall time on every
    /// item in that sub-group, so percentiles are request-volume
    /// weighted backend times, not amortised per-item costs. Backends
    /// may use different timing semantics; compare distributions
    /// within the same backend/model and use rate(sum)/rate(count)
    /// when an aggregate mean is required.
    ///
    /// Labels are bounded: `phase` is a 3-value enum, `operation`
    /// is restricted to encode/score/extract at the observe site,
    /// and `model` goes through `model_label()` which honours
    /// `SIE_METRIC_MODEL_ALLOWLIST`.
    pub backend_phase_seconds: HistogramVec,

    // IPC client visibility (per-method).
    pub ipc_request_seconds: HistogramVec,
    pub ipc_request_bytes: HistogramVec,
    pub ipc_response_bytes: HistogramVec,
    pub ipc_connect_total: IntCounterVec,
    pub ipc_reconnect_total: IntCounter,

    // IPC connection pool visibility. `ipc_pool_size` is a constant
    // reflecting the configured pool capacity; `ipc_pool_inflight` is
    // the live count of checked-out slots (≈ concurrent in-flight RPCs);
    // `ipc_pool_acquire_wait_seconds` measures how long a caller waited
    // for a free slot — a non-zero p50 is the signal that the pool is
    // saturated and should be grown.
    pub ipc_pool_size: IntGauge,
    pub ipc_pool_inflight: IntGauge,
    pub ipc_pool_acquire_wait_seconds: Histogram,

    // IPC multiplexer (`SIE_IPC_MUX=1`) visibility. Both metrics are
    // registered unconditionally so dashboards can compare the slot-pool
    // path to the mux path side-by-side; they stay at zero on the
    // slot-pool path. `ipc_mux_inflight` is a gauge of in-flight RPCs
    // through the multiplexer; without an explicit cap (the eventual
    // SIE_IPC_MUX_MAX_INFLIGHT_PER_POD knob) it tracks the raw concurrency
    // dispatched onto the single UDS connection.
    // `ipc_mux_acquire_wait_seconds` is the wall time spent waiting for
    // a per-pod inflight permit — the histogram is wired through
    // `MuxClient::call_raw` so it lights up the moment the cap is added,
    // no rebuild required.
    pub ipc_mux_inflight: IntGauge,
    pub ipc_mux_acquire_wait_seconds: Histogram,

    // Payload store visibility.
    pub payload_fetch_total: IntCounter,
    pub payload_fetch_errors_total: IntCounterVec,
    pub payload_fetch_seconds: HistogramVec,
    pub payload_bytes: HistogramVec,

    // JetStream ACK/NAK outcome accounting.
    pub jetstream_ack_failures_total: IntCounter,
    pub jetstream_nak_failures_total: IntCounter,
    pub nats_fetch_errors_total: IntCounter,
    pub nats_stream_errors_total: IntCounter,
    pub nats_redelivery_total: IntCounter,
    pub nats_deliver_count: HistogramVec,

    // Live worker config apply.
    pub config_deltas_total: IntCounterVec,
    pub config_epoch: IntGauge,

    // Shutdown / drain timing.
    pub shutdown_drain_seconds: HistogramVec,
    pub shutdown_drain_deadline_exceeded_total: IntCounter,

    // Router fall-through events (A said supports, returned UnsupportedModel,
    // router moved on to B).
    pub router_fallthrough_total: IntCounterVec,

    /// Scheduler-family metrics. Registered at startup but populated
    /// only once the Rust scheduler is actually
    /// draining traffic — keeps `/metrics` shape stable across
    /// partially-wired / fully-wired deployments.
    pub scheduler: crate::scheduler::SchedulerMetrics,

    /// Optional active/materialized model ids for bounding cardinality.
    /// None => unrestricted (matches Python's behaviour).
    materialized_model_ids: Option<std::collections::HashSet<String>>,
}

impl MetricsRegistry {
    pub fn new() -> anyhow::Result<Self> {
        let registry = Registry::new();

        let messages_received_total =
            IntCounter::new("sie_worker_messages_received_total", "NATS msgs pulled")?;
        let messages_acked_total =
            IntCounter::new("sie_worker_messages_acked_total", "NATS msgs ACKed")?;
        let messages_naked_total =
            IntCounter::new("sie_worker_messages_naked_total", "NATS msgs NAKed")?;
        let pool_admission_naks_total = IntCounterVec::new(
            Opts::new(
                "sie_worker_pool_admission_naks_total",
                "Work items NAKed before IPC because this worker is not assigned to the logical admission pool",
            ),
            &["reason"],
        )?;
        let ipc_requests_total = IntCounter::new("sie_worker_ipc_requests_total", "IPC RPCs sent")?;
        let ipc_failures_total = IntCounter::new(
            "sie_worker_ipc_failures_total",
            "IPC RPCs that returned ok=false or failed",
        )?;
        let inflight_batches = IntGauge::new(
            "sie_worker_inflight_batches",
            "Number of batches currently in flight through IPC",
        )?;

        let pull_items_fetched = HistogramVec::new(
            HistogramOpts::new(
                "sie_pull_loop_items_fetched",
                "Number of items fetched per pull cycle",
            )
            .buckets(ITEMS_FETCHED_BUCKETS.to_vec()),
            &["model"],
        )?;
        let pull_batch_process_seconds = HistogramVec::new(
            HistogramOpts::new(
                "sie_pull_loop_batch_process_seconds",
                "Time to process a pulled batch (seconds)",
            )
            .buckets(BATCH_PROCESS_BUCKETS.to_vec()),
            &["model", "operation"],
        )?;
        let pull_nak_unloaded_total = IntCounterVec::new(
            Opts::new(
                "sie_pull_loop_nak_unloaded_total",
                "Work items NAKed because the target model is not loaded",
            ),
            &["model"],
        )?;
        let pull_model_loads_total = IntCounterVec::new(
            Opts::new(
                "sie_pull_loop_model_loads_total",
                "Background model loads triggered by demand",
            ),
            &["model"],
        )?;
        let generate_model_loading_responses_total = IntCounterVec::new(
            Opts::new(
                "sie_worker_generate_model_loading_responses_total",
                "Terminal MODEL_LOADING generation responses emitted by the sidecar before execution",
            ),
            &["model", "state", "result"],
        )?;

        let backend_process_seconds = HistogramVec::new(
            HistogramOpts::new(
                "sie_worker_backend_process_seconds",
                "End-to-end wall time for a single batch in a backend (seconds)",
            )
            .buckets(BATCH_PROCESS_BUCKETS.to_vec()),
            &["backend", "operation", "model", "result"],
        )?;
        let backend_batch_items = HistogramVec::new(
            HistogramOpts::new(
                "sie_worker_backend_batch_items",
                "Number of items in a batch handed to a backend",
            )
            .buckets(ITEMS_FETCHED_BUCKETS.to_vec()),
            &["backend", "operation"],
        )?;
        let backend_process_errors_total = IntCounterVec::new(
            Opts::new(
                "sie_worker_backend_process_errors_total",
                "Batch-level errors returned by a backend (broken out by kind)",
            ),
            &["backend", "operation", "error_kind"],
        )?;
        let backend_ensure_ready_seconds = HistogramVec::new(
            HistogramOpts::new(
                "sie_worker_backend_ensure_ready_seconds",
                "Wall time for ensure_model_ready calls (seconds)",
            )
            .buckets(BATCH_PROCESS_BUCKETS.to_vec()),
            &["backend", "model", "result"],
        )?;
        let backend_item_outcomes_total = IntCounterVec::new(
            Opts::new(
                "sie_worker_backend_item_outcomes_total",
                "Per-item dispositions returned by a backend (publish, error, ack-to-drop, ...)",
            ),
            &["backend", "operation", "disposition"],
        )?;

        let backend_phase_seconds = HistogramVec::new(
            HistogramOpts::new(
                "sie_worker_backend_phase_seconds",
                "Per-item wall time for each backend phase (tokenize, inference, postprocess), in seconds. Populated from ItemOutcome timings on the success path.",
            )
            .buckets(BATCH_PROCESS_BUCKETS.to_vec()),
            &["operation", "model", "phase"],
        )?;

        let ipc_request_seconds = HistogramVec::new(
            HistogramOpts::new(
                "sie_worker_ipc_request_seconds",
                "Wall time of a single IPC RPC, by method and outcome",
            )
            .buckets(IPC_REQUEST_BUCKETS.to_vec()),
            &["method", "result"],
        )?;
        let ipc_request_bytes = HistogramVec::new(
            HistogramOpts::new(
                "sie_worker_ipc_request_bytes",
                "Bytes in outgoing IPC request frames",
            )
            .buckets(BYTES_BUCKETS.to_vec()),
            &["method"],
        )?;
        let ipc_response_bytes = HistogramVec::new(
            HistogramOpts::new(
                "sie_worker_ipc_response_bytes",
                "Bytes in incoming IPC response frames",
            )
            .buckets(BYTES_BUCKETS.to_vec()),
            &["method"],
        )?;
        let ipc_connect_total = IntCounterVec::new(
            Opts::new(
                "sie_worker_ipc_connect_total",
                "IPC socket connect attempts, by outcome",
            ),
            &["result"],
        )?;
        let ipc_reconnect_total = IntCounter::new(
            "sie_worker_ipc_reconnect_total",
            "IPC connection resets that triggered a reconnect",
        )?;

        let ipc_pool_size = IntGauge::new(
            "sie_worker_ipc_pool_size",
            "Configured IPC connection pool capacity (constant after startup)",
        )?;
        let ipc_pool_inflight = IntGauge::new(
            "sie_worker_ipc_pool_inflight",
            "IPC connections currently checked out of the pool (concurrent in-flight RPCs)",
        )?;
        let ipc_pool_acquire_wait_seconds = Histogram::with_opts(
            HistogramOpts::new(
                "sie_worker_ipc_pool_acquire_wait_seconds",
                "Wall time a caller waited for a free IPC pool slot; non-zero p50 indicates \
                 the pool is saturated and should be enlarged via SIE_IPC_POOL_SIZE",
            )
            .buckets(IPC_REQUEST_BUCKETS.to_vec()),
        )?;

        let ipc_mux_inflight = IntGauge::new(
            "sie_worker_ipc_mux_inflight",
            "RPCs currently in flight on the IPC multiplexer (SIE_IPC_MUX=1). Stays at 0 \
             when the slot-pool transport is in use.",
        )?;
        let ipc_mux_acquire_wait_seconds = Histogram::with_opts(
            HistogramOpts::new(
                "sie_worker_ipc_mux_acquire_wait_seconds",
                "Wall time a caller waited for a multiplexer inflight permit \
                 (SIE_IPC_MUX_MAX_INFLIGHT_PER_POD). Non-zero p50 means the cap is shaping \
                 dispatch to Python; flat zero means no cap is in effect.",
            )
            .buckets(IPC_REQUEST_BUCKETS.to_vec()),
        )?;

        let payload_fetch_total = IntCounter::new(
            "sie_worker_payload_fetch_total",
            "Payload store fetches (payload_ref inline resolution)",
        )?;
        let payload_fetch_errors_total = IntCounterVec::new(
            Opts::new(
                "sie_worker_payload_fetch_errors_total",
                "Payload store fetch failures, by reason",
            ),
            &["reason"],
        )?;
        let payload_fetch_seconds = HistogramVec::new(
            HistogramOpts::new(
                "sie_worker_payload_fetch_seconds",
                "Payload store fetch wall time (seconds)",
            )
            .buckets(IPC_REQUEST_BUCKETS.to_vec()),
            &["result"],
        )?;
        let payload_bytes = HistogramVec::new(
            HistogramOpts::new(
                "sie_worker_payload_bytes",
                "Payload store fetch payload size (bytes)",
            )
            .buckets(BYTES_BUCKETS.to_vec()),
            &["result"],
        )?;

        let jetstream_ack_failures_total = IntCounter::new(
            "sie_worker_jetstream_ack_failures_total",
            "JetStream ACK RPCs that failed at the broker",
        )?;
        let jetstream_nak_failures_total = IntCounter::new(
            "sie_worker_jetstream_nak_failures_total",
            "JetStream NAK RPCs that failed at the broker",
        )?;
        let nats_fetch_errors_total = IntCounter::new(
            "sie_worker_nats_fetch_errors_total",
            "Failures to build a JetStream pull/fetch stream",
        )?;
        let nats_stream_errors_total = IntCounter::new(
            "sie_worker_nats_stream_errors_total",
            "Errors reading messages from an active JetStream pull stream",
        )?;
        let nats_redelivery_total = IntCounter::new(
            "sie_worker_nats_redelivery_total",
            "Messages observed with deliver_count > 1 (JetStream redeliveries)",
        )?;
        let nats_deliver_count = HistogramVec::new(
            HistogramOpts::new(
                "sie_worker_nats_deliver_count",
                "JetStream delivery count observed per message (1 = first delivery)",
            )
            .buckets(DELIVER_COUNT_BUCKETS.to_vec()),
            &["operation"],
        )?;
        let config_deltas_total = IntCounterVec::new(
            Opts::new(
                "sie_worker_config_deltas_total",
                "Bundle-scoped worker config notifications by kind and result",
            ),
            &["kind", "result"],
        )?;
        let config_epoch = IntGauge::new(
            "sie_worker_config_epoch",
            "Highest bundle-scoped config epoch successfully applied by this worker",
        )?;

        let shutdown_drain_seconds = HistogramVec::new(
            HistogramOpts::new(
                "sie_worker_shutdown_drain_seconds",
                "Graceful shutdown drain wall time (seconds)",
            )
            .buckets(DRAIN_DURATION_BUCKETS.to_vec()),
            &["result"],
        )?;
        let shutdown_drain_deadline_exceeded_total = IntCounter::new(
            "sie_worker_shutdown_drain_deadline_exceeded_total",
            "Shutdown drains that hit the configured deadline",
        )?;

        let router_fallthrough_total = IntCounterVec::new(
            Opts::new(
                "sie_worker_router_fallthrough_total",
                "Router fell through from one backend to the next because the first returned UnsupportedModel",
            ),
            &["from_backend", "to_backend", "operation"],
        )?;

        registry.register(Box::new(messages_received_total.clone()))?;
        registry.register(Box::new(messages_acked_total.clone()))?;
        registry.register(Box::new(messages_naked_total.clone()))?;
        registry.register(Box::new(pool_admission_naks_total.clone()))?;
        registry.register(Box::new(ipc_requests_total.clone()))?;
        registry.register(Box::new(ipc_failures_total.clone()))?;
        registry.register(Box::new(inflight_batches.clone()))?;
        registry.register(Box::new(pull_items_fetched.clone()))?;
        registry.register(Box::new(pull_batch_process_seconds.clone()))?;
        registry.register(Box::new(pull_nak_unloaded_total.clone()))?;
        registry.register(Box::new(pull_model_loads_total.clone()))?;
        registry.register(Box::new(generate_model_loading_responses_total.clone()))?;
        registry.register(Box::new(backend_process_seconds.clone()))?;
        registry.register(Box::new(backend_batch_items.clone()))?;
        registry.register(Box::new(backend_process_errors_total.clone()))?;
        registry.register(Box::new(backend_ensure_ready_seconds.clone()))?;
        registry.register(Box::new(backend_item_outcomes_total.clone()))?;
        registry.register(Box::new(backend_phase_seconds.clone()))?;
        registry.register(Box::new(ipc_request_seconds.clone()))?;
        registry.register(Box::new(ipc_request_bytes.clone()))?;
        registry.register(Box::new(ipc_response_bytes.clone()))?;
        registry.register(Box::new(ipc_connect_total.clone()))?;
        registry.register(Box::new(ipc_reconnect_total.clone()))?;
        registry.register(Box::new(ipc_pool_size.clone()))?;
        registry.register(Box::new(ipc_pool_inflight.clone()))?;
        registry.register(Box::new(ipc_pool_acquire_wait_seconds.clone()))?;
        registry.register(Box::new(ipc_mux_inflight.clone()))?;
        registry.register(Box::new(ipc_mux_acquire_wait_seconds.clone()))?;
        registry.register(Box::new(payload_fetch_total.clone()))?;
        registry.register(Box::new(payload_fetch_errors_total.clone()))?;
        registry.register(Box::new(payload_fetch_seconds.clone()))?;
        registry.register(Box::new(payload_bytes.clone()))?;
        registry.register(Box::new(jetstream_ack_failures_total.clone()))?;
        registry.register(Box::new(jetstream_nak_failures_total.clone()))?;
        registry.register(Box::new(nats_fetch_errors_total.clone()))?;
        registry.register(Box::new(nats_stream_errors_total.clone()))?;
        registry.register(Box::new(nats_redelivery_total.clone()))?;
        registry.register(Box::new(nats_deliver_count.clone()))?;
        registry.register(Box::new(config_deltas_total.clone()))?;
        registry.register(Box::new(config_epoch.clone()))?;
        registry.register(Box::new(shutdown_drain_seconds.clone()))?;
        registry.register(Box::new(shutdown_drain_deadline_exceeded_total.clone()))?;
        registry.register(Box::new(router_fallthrough_total.clone()))?;

        // Register the scheduler metric family. Keeps the Scheduler
        // module self-contained — this file doesn't need to know
        // about the internal layout or buckets, it just offers the
        // registry handle.
        let scheduler = crate::scheduler::SchedulerMetrics::register(&registry)?;

        let materialized_model_ids = std::env::var("SIE_METRIC_MODEL_ALLOWLIST")
            .ok()
            .map(|raw| {
                raw.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect::<std::collections::HashSet<_>>()
            })
            .filter(|set| !set.is_empty());

        Ok(Self {
            registry,
            messages_received_total,
            messages_acked_total,
            messages_naked_total,
            pool_admission_naks_total,
            ipc_requests_total,
            ipc_failures_total,
            inflight_batches,
            pull_items_fetched,
            pull_batch_process_seconds,
            pull_nak_unloaded_total,
            pull_model_loads_total,
            generate_model_loading_responses_total,
            backend_process_seconds,
            backend_batch_items,
            backend_process_errors_total,
            backend_ensure_ready_seconds,
            backend_item_outcomes_total,
            backend_phase_seconds,
            ipc_request_seconds,
            ipc_request_bytes,
            ipc_response_bytes,
            ipc_connect_total,
            ipc_reconnect_total,
            ipc_pool_size,
            ipc_pool_inflight,
            ipc_pool_acquire_wait_seconds,
            ipc_mux_inflight,
            ipc_mux_acquire_wait_seconds,
            payload_fetch_total,
            payload_fetch_errors_total,
            payload_fetch_seconds,
            payload_bytes,
            jetstream_ack_failures_total,
            jetstream_nak_failures_total,
            nats_fetch_errors_total,
            nats_stream_errors_total,
            nats_redelivery_total,
            nats_deliver_count,
            config_deltas_total,
            config_epoch,
            shutdown_drain_seconds,
            shutdown_drain_deadline_exceeded_total,
            router_fallthrough_total,
            scheduler,
            materialized_model_ids,
        })
    }

    /// Resolve a raw model id into the value that should be used as
    /// a `model` Prometheus label. Honors `SIE_METRIC_MODEL_ALLOWLIST`;
    /// models outside the active/materialized set get bucketed as
    /// `"other"` to bound cardinality. If no set is configured we pass
    /// the id through unchanged.
    pub fn model_label<'a>(&self, model_id: &'a str) -> std::borrow::Cow<'a, str> {
        match &self.materialized_model_ids {
            Some(set) if !set.contains(model_id) => std::borrow::Cow::Borrowed("other"),
            _ => std::borrow::Cow::Borrowed(model_id),
        }
    }

    /// Test-only constructor that bypasses env parsing so unit tests
    /// can exercise [`model_label`] without racing on
    /// `SIE_METRIC_MODEL_ALLOWLIST`. Pass `None` for "no configured set"
    /// (passthrough behaviour) or `Some(set)` to enable bucketing.
    #[cfg(test)]
    fn with_materialized_models_for_tests(
        materialized_model_ids: Option<std::collections::HashSet<String>>,
    ) -> anyhow::Result<Self> {
        let mut reg = Self::new()?;
        // Even an explicit empty set means "no configured set" — matches
        // the `filter(|set| !set.is_empty())` rule in `new()`.
        reg.materialized_model_ids = materialized_model_ids.filter(|s| !s.is_empty());
        Ok(reg)
    }
}

/// Serve the sidecar's HTTP probe + observability surface.
///
/// The caller provides the `MetricsRegistry` so counters updated by the
/// dispatcher and friends are exposed here.
/// * `GET /metrics` — Prometheus text (counters/histograms wired by
///   the dispatcher and friends).
/// * `GET /healthz` — **liveness**. Always 200 unless the process is
///   wedged hard enough that the tokio runtime stops servicing this
///   handler. Operators (and K8s `livenessProbe`) point here.
/// * `GET /readyz` — **readiness**. 200 iff the IPC handshake has
///   succeeded AND the most recent heartbeat-echo arrived within
///   the configured freshness window AND we are not draining. See
///   [`crate::readiness`] for the precise contract.
///
/// The Helm chart can point the K8s `readinessProbe` here during the
/// dual-probe transition instead of at Python `/readyz` on port 8080.
/// See `docs/architecture-guide.md` for the readiness contract.
pub fn spawn_metrics_server(
    port: u16,
    metrics: Arc<MetricsRegistry>,
    readiness: Arc<Readiness>,
    shutdown: Arc<Shutdown>,
) -> anyhow::Result<JoinHandle<()>> {
    let registry = Arc::clone(&metrics);

    let handle = tokio::spawn(async move {
        let addr: SocketAddr = ([0, 0, 0, 0], port).into();
        let listener = match TcpListener::bind(addr).await {
            Ok(l) => l,
            Err(e) => {
                error!(error = %e, addr = %addr, "metrics server: bind failed");
                return;
            }
        };
        info!(addr = %addr, "metrics server listening");

        loop {
            let accept = listener.accept();
            let shutdown_wait = shutdown.wait();
            tokio::select! {
                biased;
                _ = shutdown_wait => {
                    info!("metrics server stopping (shutdown)");
                    return;
                }
                res = accept => {
                    match res {
                        Ok((stream, _peer)) => {
                            let reg = Arc::clone(&registry);
                            let ready = Arc::clone(&readiness);
                            let io = TokioIo::new(stream);
                            tokio::spawn(async move {
                                let svc = service_fn(move |req| {
                                    handle_request(req, Arc::clone(&reg), Arc::clone(&ready))
                                });
                                if let Err(e) = hyper::server::conn::http1::Builder::new()
                                    .serve_connection(io, svc)
                                    .await
                                {
                                    warn!(error = %e, "metrics connection failed");
                                }
                            });
                        }
                        Err(e) => {
                            warn!(error = %e, "metrics accept failed");
                        }
                    }
                }
            }
        }
    });

    Ok(handle)
}

async fn handle_request(
    req: Request<hyper::body::Incoming>,
    metrics: Arc<MetricsRegistry>,
    readiness: Arc<Readiness>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    match (req.method(), req.uri().path()) {
        (&Method::GET, "/metrics") => {
            let encoder = TextEncoder::new();
            let mut buf = Vec::new();
            if let Err(e) = encoder.encode(&metrics.registry.gather(), &mut buf) {
                warn!(error = %e, "metrics encode failed");
            }
            let resp = Response::builder()
                .status(StatusCode::OK)
                .header("Content-Type", encoder.format_type())
                .body(Full::new(Bytes::from(buf)))
                .unwrap();
            Ok(resp)
        }
        (&Method::GET, "/healthz") => Ok(Response::builder()
            .status(StatusCode::OK)
            .body(Full::new(Bytes::from_static(b"ok")))
            .unwrap()),
        (&Method::GET, "/readyz") => Ok(readyz_response(&readiness)),
        _ => Ok(Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Full::new(Bytes::from_static(b"not found")))
            .unwrap()),
    }
}

fn readyz_status_and_body(readiness: &Readiness) -> (StatusCode, String) {
    let snap = readiness.snapshot();
    if snap.is_ready() {
        (StatusCode::OK, "ok".to_owned())
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            format!("not ready: {}", snap.reason()),
        )
    }
}

fn readyz_response(readiness: &Readiness) -> Response<Full<Bytes>> {
    let (status, body) = readyz_status_and_body(readiness);
    Response::builder()
        .status(status)
        .header("Content-Type", "text/plain; charset=utf-8")
        .body(Full::new(Bytes::from(body)))
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_constructs_cleanly() {
        let m = MetricsRegistry::new().unwrap();
        m.messages_received_total.inc();
        m.messages_acked_total.inc_by(5);
        m.inflight_batches.set(3);
        assert_eq!(m.messages_received_total.get(), 1);
        assert_eq!(m.messages_acked_total.get(), 5);
        assert_eq!(m.inflight_batches.get(), 3);

        m.pull_items_fetched.with_label_values(&["m1"]).observe(4.0);
        m.pull_batch_process_seconds
            .with_label_values(&["m1", "encode"])
            .observe(0.123);
        m.pull_nak_unloaded_total
            .with_label_values(&["m1"])
            .inc_by(3);
        m.pull_model_loads_total.with_label_values(&["m1"]).inc();
        m.generate_model_loading_responses_total
            .with_label_values(&["m1", "loading_started", "published_acked"])
            .inc();

        let families = m.registry.gather();
        let names: Vec<&str> = families.iter().map(|f| f.name()).collect();
        for expected in [
            "sie_worker_messages_received_total",
            "sie_pull_loop_items_fetched",
            "sie_pull_loop_batch_process_seconds",
            "sie_pull_loop_nak_unloaded_total",
            "sie_pull_loop_model_loads_total",
            "sie_worker_generate_model_loading_responses_total",
        ] {
            assert!(names.contains(&expected), "missing metric {expected}");
        }
    }

    #[test]
    fn model_label_is_passthrough_when_materialized_set_is_absent() {
        // Default install (no SIE_METRIC_MODEL_ALLOWLIST set) keeps
        // existing dashboards unchanged: every unique model_id shows
        // up as its own Prometheus label. No bucketing.
        let m = MetricsRegistry::with_materialized_models_for_tests(None).unwrap();
        assert_eq!(m.model_label("any/model"), "any/model");
        assert_eq!(m.model_label(""), "");
    }

    #[test]
    fn model_label_passes_through_materialized_ids() {
        let set: std::collections::HashSet<String> =
            ["bge-m3", "sentence-transformers/all-MiniLM-L6-v2"]
                .iter()
                .map(|s| s.to_string())
                .collect();
        let m = MetricsRegistry::with_materialized_models_for_tests(Some(set)).unwrap();
        assert_eq!(m.model_label("bge-m3"), "bge-m3");
        assert_eq!(
            m.model_label("sentence-transformers/all-MiniLM-L6-v2"),
            "sentence-transformers/all-MiniLM-L6-v2"
        );
    }

    #[test]
    fn model_label_buckets_unknown_ids_as_other() {
        // Misbehaving or unexpected model_ids must not explode
        // Prometheus cardinality. Anything not in the active set
        // collapses to the sentinel `"other"` bucket, regardless of
        // content (unicode, path separators, length, empty string).
        let set: std::collections::HashSet<String> = ["bge-m3".to_string()].into_iter().collect();
        let m = MetricsRegistry::with_materialized_models_for_tests(Some(set)).unwrap();
        for rogue in [
            "random/model",
            "",
            "🦀🦀🦀",
            &"x".repeat(4096),
            "weird chars & '\"  with spaces",
        ] {
            assert_eq!(
                m.model_label(rogue),
                "other",
                "expected bucket for {rogue:?}"
            );
        }
    }

    #[test]
    fn empty_materialized_set_is_equivalent_to_no_configured_set() {
        // Matches the env-parse rule in `new()` that treats an empty
        // `SIE_METRIC_MODEL_ALLOWLIST=""` as absent — otherwise
        // operators could accidentally collapse every label to
        // `"other"` by clearing the env var.
        let empty: std::collections::HashSet<String> = std::collections::HashSet::new();
        let m = MetricsRegistry::with_materialized_models_for_tests(Some(empty)).unwrap();
        assert_eq!(m.model_label("whatever"), "whatever");
    }

    /// Guardrail that `backend_phase_seconds` is wired up: name +
    /// labels match what dispatcher.rs observes into, the histogram
    /// is registered in the shared Prometheus registry, and sample
    /// observations make it through `gather()`. Without this test a
    /// typo in either the declaration or the `with_label_values()`
    /// call would silently produce a Prometheus error counter
    /// ("inconsistent label cardinality") at runtime — the kind of
    /// thing you only notice when the dashboard stays flat under
    /// load.
    #[test]
    fn backend_phase_seconds_is_registered_and_observable() {
        let m = MetricsRegistry::new().unwrap();

        // Exercise the same three phases the dispatcher emits, with a
        // couple of distinct (operation, model) combinations so we
        // prove the label set survives round-tripping.
        for (op, model, phase, ms) in [
            ("encode", "BAAI/bge-m3", "tokenize", 1.2_f64),
            ("encode", "BAAI/bge-m3", "inference", 42.7),
            ("encode", "BAAI/bge-m3", "postprocess", 0.3),
            ("score", "cross-encoder/ms-marco", "inference", 88.1),
        ] {
            m.backend_phase_seconds
                .with_label_values(&[op, model, phase])
                .observe(ms / 1_000.0);
        }

        let families = m.registry.gather();
        let fam = families
            .iter()
            .find(|f| f.name() == "sie_worker_backend_phase_seconds")
            .expect("backend_phase_seconds not registered in shared registry");

        // Expect exactly four time-series (one per distinct label
        // tuple), each with a single observation.
        assert_eq!(
            fam.get_metric().len(),
            4,
            "expected 4 label-tuples, got {}",
            fam.get_metric().len()
        );
        for metric in fam.get_metric() {
            assert_eq!(
                metric.get_histogram().get_sample_count(),
                1,
                "expected 1 sample per tuple, got {}",
                metric.get_histogram().get_sample_count()
            );
            let labels: std::collections::HashMap<&str, &str> = metric
                .get_label()
                .iter()
                .map(|lp| (lp.name(), lp.value()))
                .collect();
            assert!(labels.contains_key("operation"));
            assert!(labels.contains_key("model"));
            assert!(labels.contains_key("phase"));
            assert!(
                matches!(labels["phase"], "tokenize" | "inference" | "postprocess"),
                "unexpected phase label: {:?}",
                labels["phase"]
            );
        }
    }
}
