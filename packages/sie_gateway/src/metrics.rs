use prometheus::{
    linear_buckets, Counter, CounterVec, Gauge, GaugeVec, HistogramOpts, HistogramVec, IntGauge,
    Opts, Registry,
};
use std::sync::{Arc, LazyLock, OnceLock};

// Routing decision that the HTTP metrics middleware reads as label
// values for `sie_gateway_requests_total` and
// `sie_gateway_request_latency_seconds`. The proxy handler writes this
// once after it has split `pool/profile`, resolved `l4 -> l4-spot`,
// and validated against `configured_gpus`, so the middleware records
// the *canonical* `machine_profile` the router actually used instead
// of whatever the client shoved into `x-sie-machine-profile`. Keeping
// both halves of the request flow in sync is what lets operators join
// these histograms against every other `{machine_profile}` metric on
// the dashboard (rejections, pending demand, active leases, …).
#[derive(Clone, Debug, Default)]
pub struct MetricLabels {
    pub machine_profile: String,
}

/// Bound a caller-influenced label value before it becomes a Prometheus
/// label. Pool names are caller-defined (bench / isolation pools are a
/// feature), so we can't reject unknown values — but we *can* stop a
/// single request from injecting an absurdly long or junk-charset label,
/// which is the trivial cardinality / memory DoS. Empty → `unknown`;
/// over-length or out-of-charset → `invalid`; otherwise passes through.
pub fn sanitize_label(value: &str) -> String {
    const MAX_LABEL_LEN: usize = 48;
    if value.is_empty() {
        return "unknown".to_string();
    }
    if value.len() > MAX_LABEL_LEN
        || !value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | ':' | '-'))
    {
        return "invalid".to_string();
    }
    value.to_string()
}

/// Bound a caller-influenced **model id** before it becomes a Prometheus
/// label. Identical intent to [`sanitize_label`] but the charset also
/// permits `/` because model ids are `org/name`-shaped (e.g.
/// `BAAI/bge-m3`) — passing those through `sanitize_label` would collapse
/// every real model to `"invalid"`. We still reject the subject-/SSE-/
/// log-dangerous chars (`*`, `>`, whitespace, control, etc.) and bound
/// length so a hostile `POST /v1/generate/<10KB-of-junk>` can't mint an
/// unbounded number of series or an enormous one. Empty → `unknown`;
/// over-length / out-of-charset → `invalid`.
///
/// Note: this does NOT case-fold. Case-variant collapsing of *known*
/// models happens upstream at the request boundary
/// (`ModelRegistry::resolve_canonical_model_name`); this is only the
/// cardinality / charset backstop for unknown ids.
pub fn sanitize_model_label(model: &str) -> String {
    // Model ids are longer than pool names (`org/long-model-name-v0.2`),
    // so allow more headroom than `sanitize_label`'s 48 while still
    // bounding the worst case.
    const MAX_MODEL_LABEL_LEN: usize = 128;
    if model.is_empty() {
        return "unknown".to_string();
    }
    if model.len() > MAX_MODEL_LABEL_LEN
        || !model
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | ':' | '-' | '/'))
    {
        return "invalid".to_string();
    }
    model.to_string()
}

// Request-extension carrier for `MetricLabels`. Installed empty by the
// metrics middleware, filled by the proxy handler post-normalization,
// and read by the middleware after the inner service responds. The
// `OnceLock` is the cheap cell: one uncontended write + one read per
// inference request, no lock and no async-unsafe primitives.
//
// If a handler returns before calling `set` (e.g. the `model is
// required` early exit, which has no GPU yet), the middleware falls
// back to the `"unknown"` label — a real, operator-friendly value
// rather than an empty string or a noisy panic.
#[derive(Clone, Default)]
pub struct MetricLabelsSlot(Arc<OnceLock<MetricLabels>>);

impl MetricLabelsSlot {
    pub fn set(&self, labels: MetricLabels) {
        // First write wins. A handler that accidentally sets twice
        // (cleanup path + main path, for example) should not panic
        // here — observability code must never take down the request.
        let _ = self.0.set(labels);
    }

    pub fn get(&self) -> Option<&MetricLabels> {
        self.0.get()
    }
}

pub static REGISTRY: LazyLock<Registry> = LazyLock::new(|| {
    let r = Registry::new();
    r.register(Box::new(REQUEST_COUNT.clone())).unwrap();
    r.register(Box::new(REQUEST_LATENCY.clone())).unwrap();
    r.register(Box::new(PROVISIONING_RESPONSES.clone()))
        .unwrap();
    r.register(Box::new(PENDING_DEMAND.clone())).unwrap();
    r.register(Box::new(POOL_WARM_FLOOR.clone())).unwrap();
    r.register(Box::new(REJECTED_REQUESTS.clone())).unwrap();
    r.register(Box::new(ACTIVE_LEASE_GPUS.clone())).unwrap();
    r.register(Box::new(WORKER_COUNT.clone())).unwrap();
    r.register(Box::new(WORKER_QUEUE_DEPTH.clone())).unwrap();
    r.register(Box::new(WORKER_MEMORY_USED.clone())).unwrap();
    r.register(Box::new(MODEL_WORKERS.clone())).unwrap();
    r.register(Box::new(QUEUE_PUBLISH_SECONDS.clone())).unwrap();
    r.register(Box::new(QUEUE_ITEMS_PUBLISHED.clone())).unwrap();
    r.register(Box::new(QUEUE_RESULT_WAIT.clone())).unwrap();
    r.register(Box::new(QUEUE_PAYLOAD_OFFLOADS.clone()))
        .unwrap();
    r.register(Box::new(QUEUE_INBOX_SKIPS.clone())).unwrap();
    r.register(Box::new(QUEUE_ACK_FAILURES.clone())).unwrap();
    r.register(Box::new(GENERATION_STALE_ATTEMPT_CHUNKS.clone()))
        .unwrap();
    r.register(Box::new(GENERATION_INVALID_CHUNKS.clone()))
        .unwrap();
    r.register(Box::new(GENERATION_SEQ_GAP_CHUNKS.clone()))
        .unwrap();
    r.register(Box::new(GENERATION_TIMEOUTS.clone())).unwrap();
    r.register(Box::new(GENERATION_CANCELLED.clone())).unwrap();
    r.register(Box::new(GENERATION_TTFT.clone())).unwrap();
    r.register(Box::new(GENERATION_TPOT.clone())).unwrap();
    r.register(Box::new(GENERATION_TOTAL_TOKENS.clone()))
        .unwrap();
    // Direct-dispatch routing metrics
    r.register(Box::new(ROUTING_FALLBACK_TOTAL.clone()))
        .unwrap();
    r.register(Box::new(GENERATION_FALLBACK_REFUSED_TOTAL.clone()))
        .unwrap();
    r.register(Box::new(RATE_LIMIT_TOTAL.clone())).unwrap();
    r.register(Box::new(ROUTING_KEY_SOURCE.clone())).unwrap();
    r.register(Box::new(ROUTING_HRW_RING_SIZE.clone())).unwrap();
    r.register(Box::new(ROUTING_CACHE_HIT_ESTIMATE.clone()))
        .unwrap();
    r.register(Box::new(KV_RESERVATION_KNOWN.clone())).unwrap();
    r.register(Box::new(DLQ_EVENTS.clone())).unwrap();
    r.register(Box::new(GRAMMAR_REJECTS.clone())).unwrap();
    r.register(Box::new(CONFIG_BOOTSTRAP_FAILURES.clone()))
        .unwrap();
    r.register(Box::new(CONFIG_BOOTSTRAP_DEGRADED.clone()))
        .unwrap();
    r.register(Box::new(CONFIG_EPOCH.clone())).unwrap();
    r.register(Box::new(CONFIG_DELTAS.clone())).unwrap();
    r.register(Box::new(NATS_CONNECTED.clone())).unwrap();
    r.register(Box::new(POOL_EVENTS.clone())).unwrap();
    r.register(Box::new(DLQ_REPUBLISH_FAILURES.clone()))
        .unwrap();
    r
});

// 1. sie_gateway_requests_total
pub static REQUEST_COUNT: LazyLock<CounterVec> = LazyLock::new(|| {
    CounterVec::new(
        Opts::new("sie_gateway_requests_total", "Total proxied requests"),
        &["endpoint", "status", "machine_profile"],
    )
    .unwrap()
});

// 2. sie_gateway_request_latency_seconds
pub static REQUEST_LATENCY: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new("sie_gateway_request_latency_seconds", "Request latency").buckets(vec![
            // The sub-5 ms range is where the gateway's own
            // overhead for small JSON requests now lives — without
            // these extra low buckets p50 gets squashed into the
            // `0.005` bucket and small wins disappear on dashboards.
            0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0,
            10.0, 30.0, 60.0,
        ]),
        &["endpoint", "machine_profile"],
    )
    .unwrap()
});

// 3. sie_gateway_provisioning_responses_total
pub static PROVISIONING_RESPONSES: LazyLock<CounterVec> = LazyLock::new(|| {
    CounterVec::new(
        Opts::new(
            "sie_gateway_provisioning_responses_total",
            "Provisioning responses returned when no ready worker exists",
        ),
        &["machine_profile"],
    )
    .unwrap()
});

// 4. sie_gateway_pending_demand
pub static PENDING_DEMAND: LazyLock<GaugeVec> = LazyLock::new(|| {
    GaugeVec::new(
        Opts::new(
            "sie_gateway_pending_demand",
            "Pools with unmet demand (KEDA trigger)",
        ),
        &["pool", "machine_profile", "bundle"],
    )
    .unwrap()
});

// sie_gateway_pool_warm_floor
//
// Per-pool warm floor (minimum warm machines) the operator set via
// `PoolSpec.minimum_worker_count`. The gateway re-emits this every 30s for
// every (pool, machine_profile, bundle) lane the pool can name; KEDA reads it
// as a separate trigger and, because KEDA takes the max desiredReplicas across
// triggers, keeps ceil(N/1)=N machines warm. When `minimum_worker_count` is 0
// the series is absent (or cleared), so `or vector(0)` in the trigger leaves
// scale-from-zero unchanged. Distinct from `PENDING_DEMAND` (request-driven,
// 120s expiry): this one is config-driven and persists while
// minimum_worker_count>0.
pub static POOL_WARM_FLOOR: LazyLock<GaugeVec> = LazyLock::new(|| {
    GaugeVec::new(
        Opts::new(
            "sie_gateway_pool_warm_floor",
            "Per-pool warm floor (minimum warm machines) — KEDA trigger",
        ),
        &["pool", "machine_profile", "bundle"],
    )
    .unwrap()
});

/// Set the warm-floor gauge for one `(pool, machine_profile, bundle)`
/// lane to `value` (the pool's `minimum_worker_count`). Called by the gateway's
/// warm-floor emitter loop on every tick for each lane a pool can name.
pub fn set_pool_warm_floor(pool: &str, machine_profile: &str, bundle: &str, value: f64) {
    POOL_WARM_FLOOR
        .with_label_values(&[pool, machine_profile, bundle])
        .set(value);
}

/// Zero the warm-floor gauge for a lane the emitter previously set but
/// no longer wants (pool deleted, `minimum_worker_count` dropped to 0, or the
/// profile/bundle changed). Driving it to 0 — rather than dropping the
/// series — keeps `or vector(0)` in the KEDA trigger satisfied so the
/// lane can scale back to zero.
pub fn clear_pool_warm_floor(pool: &str, machine_profile: &str, bundle: &str) {
    POOL_WARM_FLOOR
        .with_label_values(&[pool, machine_profile, bundle])
        .set(0.0);
}

// 5. sie_gateway_active_lease_gpus
pub static ACTIVE_LEASE_GPUS: LazyLock<GaugeVec> = LazyLock::new(|| {
    GaugeVec::new(
        Opts::new(
            "sie_gateway_active_lease_gpus",
            "GPUs from active leases (KEDA trigger)",
        ),
        &["pool", "machine_profile", "bundle"],
    )
    .unwrap()
});

pub static REJECTED_REQUESTS: LazyLock<CounterVec> = LazyLock::new(|| {
    CounterVec::new(
        Opts::new(
            "sie_gateway_rejected_requests_total",
            "Requests rejected before reaching a worker",
        ),
        &["pool", "machine_profile", "bundle", "reason"],
    )
    .unwrap()
});

// 6. sie_gateway_workers
pub static WORKER_COUNT: LazyLock<GaugeVec> = LazyLock::new(|| {
    GaugeVec::new(
        Opts::new("sie_gateway_workers", "Worker count by health status"),
        &["status"],
    )
    .unwrap()
});

// 7. sie_gateway_worker_queue_depth
pub static WORKER_QUEUE_DEPTH: LazyLock<GaugeVec> = LazyLock::new(|| {
    GaugeVec::new(
        Opts::new("sie_gateway_worker_queue_depth", "Per-worker queue depth"),
        &["pool", "worker", "machine_profile", "bundle"],
    )
    .unwrap()
});

// 8. sie_gateway_worker_memory_used_bytes
pub static WORKER_MEMORY_USED: LazyLock<GaugeVec> = LazyLock::new(|| {
    GaugeVec::new(
        Opts::new(
            "sie_gateway_worker_memory_used_bytes",
            "Per-worker GPU memory used",
        ),
        &["worker", "machine_profile", "bundle"],
    )
    .unwrap()
});

// 9. sie_gateway_model_workers
pub static MODEL_WORKERS: LazyLock<GaugeVec> = LazyLock::new(|| {
    GaugeVec::new(
        Opts::new("sie_gateway_model_workers", "Workers per model"),
        &["model"],
    )
    .unwrap()
});

// 10. sie_gateway_queue_publish_seconds
pub static QUEUE_PUBLISH_SECONDS: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "sie_gateway_queue_publish_seconds",
            "Work item publish time",
        )
        .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]),
        &["operation"],
    )
    .unwrap()
});

// 11. sie_gateway_queue_items_published
pub static QUEUE_ITEMS_PUBLISHED: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "sie_gateway_queue_items_published",
            "Items per publish batch",
        )
        .buckets(linear_buckets(1.0, 4.0, 16).unwrap()),
        &["operation"],
    )
    .unwrap()
});

// 12. sie_gateway_queue_result_wait_seconds
pub static QUEUE_RESULT_WAIT: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new("sie_gateway_queue_result_wait_seconds", "Result wait time").buckets(
            vec![
                0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 30.0,
            ],
        ),
        &["operation"],
    )
    .unwrap()
});

// 13. sie_gateway_queue_payload_offloads_total
pub static QUEUE_PAYLOAD_OFFLOADS: LazyLock<Counter> = LazyLock::new(|| {
    Counter::new(
        "sie_gateway_queue_payload_offloads_total",
        "Large payload offloads",
    )
    .unwrap()
});

// 14. sie_gateway_queue_inbox_skips_total
pub static QUEUE_INBOX_SKIPS: LazyLock<Counter> = LazyLock::new(|| {
    Counter::new(
        "sie_gateway_queue_inbox_skips_total",
        "Inbox messages skipped via fast-path request_id check",
    )
    .unwrap()
});

pub static QUEUE_ACK_FAILURES: LazyLock<Counter> = LazyLock::new(|| {
    Counter::new(
        "sie_gateway_queue_ack_failures_total",
        "JetStream publish acks that failed (fire-and-forget monitoring)",
    )
    .unwrap()
});

// Shared histogram buckets for generation TTFT/TPOT, identical on the
// gateway and worker sides so the "gateway minus worker" overhead
// attribution panel on the generation-poc dashboard subtracts buckets
// that have the same edges. MUST stay in sync with
// ``packages/sie_server/src/sie_server/observability/metrics.py``
// ``TTFT_TPOT_BUCKETS`` (per the metrics rollout's acceptance criterion).
pub const TTFT_TPOT_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
];

// sie_gateway_generation_ttft_seconds
//
// Gateway-observed time-to-first-token: publish → first non-empty
// chunk received on the inbox. Includes worker-side TTFT plus
// NATS + queue overhead, so the overhead attribution panel subtracts
// the worker's own ``sie_worker_generation_ttft_seconds``.
//
// The ``grammar`` label takes one of ``none|json_schema|regex|ebnf``
// (the four values ``grammar_label()`` produces). Cardinality is
// bounded by spec — see the metrics rollout's acceptance criteria.
pub static GENERATION_TTFT: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "sie_gateway_generation_ttft_seconds",
            "Gateway-observed time-to-first-token (publish to first chunk)",
        )
        .buckets(TTFT_TPOT_BUCKETS.to_vec()),
        &["model", "pool", "grammar"],
    )
    .unwrap()
});

// sie_gateway_generation_tpot_seconds
//
// Mean inter-chunk gap over a single request (first chunk → last
// chunk divided by completion-token count). One observation per
// successful generation.
pub static GENERATION_TPOT: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "sie_gateway_generation_tpot_seconds",
            "Gateway-observed mean time-per-output-token",
        )
        .buckets(TTFT_TPOT_BUCKETS.to_vec()),
        &["model", "pool", "grammar"],
    )
    .unwrap()
});

// sie_gateway_generation_total_tokens
//
// Cumulative prompt and completion token counts read from the
// terminal chunk's ``usage`` block. ``kind`` is ``prompt`` or
// ``completion`` — keeping them as one CounterVec rather than two
// separate counters means the dashboard can read either side from a
// single PromQL query with a label selector.
pub static GENERATION_TOTAL_TOKENS: LazyLock<CounterVec> = LazyLock::new(|| {
    CounterVec::new(
        Opts::new(
            "sie_gateway_generation_total_tokens",
            "Generation tokens accounted from terminal-chunk usage",
        ),
        &["model", "pool", "kind"],
    )
    .unwrap()
});

/// Canonical, bounded-cardinality grammar-kind label used by the
/// gateway TTFT/TPOT histograms. The three values mirror the grammar feature's
/// `GrammarSpec` variants plus the no-grammar case. Centralising the
/// mapping keeps the label set finite even if `GrammarSpec` grows
/// new variants later — the caller would have to amend this function
/// to introduce a new label value.
pub fn grammar_label(grammar: Option<&crate::queue::publisher::GrammarSpec>) -> &'static str {
    use crate::queue::publisher::GrammarSpec;
    match grammar {
        None => "none",
        Some(GrammarSpec::JsonSchema { .. }) => "json_schema",
        Some(GrammarSpec::Regex { .. }) => "regex",
        Some(GrammarSpec::Ebnf { .. }) => "ebnf",
    }
}

/// Record the gateway-side TTFT/TPOT histograms and the prompt /
/// completion token counters for one successful generation. Called
/// from the proxy handler on the success path, after the terminal
/// chunk's outcome is built. All three observations land under
/// `(model, pool[, grammar])` labels that come from the request
/// envelope — no caller-controlled cardinality is introduced.
pub fn record_generation_success(
    model: &str,
    pool: &str,
    grammar: &str,
    ttft_ms: Option<f64>,
    tpot_ms: Option<f64>,
    usage: Option<&crate::queue::streaming::UsageBlock>,
) {
    // Both `model` and `pool` are caller-influenced (the model id comes
    // off the request path/body; the pool off `X-SIE-Pool`). The DoS
    // hardening that bounded `pool` originally missed `model`, leaving
    // an unbounded-cardinality / memory vector — a hostile client could
    // walk `POST /v1/generate/<random>` and mint a new series per
    // request. Bound both. Canonicalisation upstream
    // (`resolve_canonical_model_name`) folds case variants of *known*
    // models; `sanitize_label` is the backstop for unknown / oversized /
    // junk-charset ids.
    let model = &sanitize_model_label(model);
    let pool = &sanitize_label(pool);
    if let Some(t) = ttft_ms {
        GENERATION_TTFT
            .with_label_values(&[model, pool, grammar])
            .observe(t / 1000.0);
    }
    if let Some(t) = tpot_ms {
        GENERATION_TPOT
            .with_label_values(&[model, pool, grammar])
            .observe(t / 1000.0);
    }
    if let Some(u) = usage {
        GENERATION_TOTAL_TOKENS
            .with_label_values(&[model, pool, "prompt"])
            .inc_by(u.prompt_tokens as f64);
        GENERATION_TOTAL_TOKENS
            .with_label_values(&[model, pool, "completion"])
            .inc_by(u.completion_tokens as f64);
    }
    // Ensure the cache-hit-estimate family has a series
    // for every (model, pool) that has ever served a request.
    // Touching `with_label_values` materialises the series (default
    // 0.0); the rolling-window writer in `set_routing_cache_hit_estimate`
    // will overwrite it once the routing-cache work lands.
    let _ = ROUTING_CACHE_HIT_ESTIMATE.with_label_values(&[model, pool]);
}

// sie_gateway_generation_stale_attempt_chunks_total
//
// Counts streaming chunks dropped because their ``attempt_id`` does not
// match the latched attempt for the request. JetStream-redelivered work
// after a worker crash is the canonical producer (the redelivered run
// generates a fresh attempt_id; if the original gateway already latched
// a different one, the late chunks land here).
pub static GENERATION_STALE_ATTEMPT_CHUNKS: LazyLock<CounterVec> = LazyLock::new(|| {
    CounterVec::new(
        Opts::new(
            "sie_gateway_generation_stale_attempt_chunks_total",
            "Streaming chunks dropped due to stale attempt_id",
        ),
        &["model", "pool"],
    )
    .unwrap()
});

// sie_gateway_generation_invalid_chunks_total
//
// Counts chunks rejected by ``StreamCollector::apply`` after deserialization
// because a wire-level invariant was violated (unknown ``kind`` discriminator,
// non-finite timing field, unknown ``finish_reason``). A non-zero counter is
// a hard signal that a worker is emitting malformed envelopes.
pub static GENERATION_INVALID_CHUNKS: LazyLock<CounterVec> = LazyLock::new(|| {
    CounterVec::new(
        Opts::new(
            "sie_gateway_generation_invalid_chunks_total",
            "Streaming chunks dropped due to wire-level invariant violation",
        ),
        &["model", "pool", "reason"],
    )
    .unwrap()
});

// sie_gateway_generation_seq_gap_chunks_total
//
// Counts chunks rejected by ``StreamCollector::apply`` because a gap was
// detected in the per-attempt ``seq`` sequence (chunk.seq > last + 1).
// Per H6: the worker's no-silent-drop guarantee means it only advances
// ``seq`` after a successful enqueue, so a gap on the wire is a genuine
// transport failure between worker and gateway. The pending stream is
// failed with a ``transport_failure`` error. A non-zero counter is a
// hard signal of a NATS-level message loss or a worker bug that bypasses
// the H6 invariant.
pub static GENERATION_SEQ_GAP_CHUNKS: LazyLock<CounterVec> = LazyLock::new(|| {
    CounterVec::new(
        Opts::new(
            "sie_gateway_generation_seq_gap_chunks_total",
            "Streaming chunks rejected because a per-attempt seq gap was detected",
        ),
        &["model", "pool"],
    )
    .unwrap()
});

// sie_gateway_generation_timeout_total
//
// Counts the three independent generation-stream timeouts:
// ``first_chunk`` (worker silent before any chunk arrived),
// ``inter_chunk`` (gap between chunks once streaming started),
// ``overall`` (hard cap derived from max_new_tokens + slack).
pub static GENERATION_TIMEOUTS: LazyLock<CounterVec> = LazyLock::new(|| {
    CounterVec::new(
        Opts::new(
            "sie_gateway_generation_timeout_total",
            "Generation timeouts by kind",
        ),
        &["model", "pool", "kind"],
    )
    .unwrap()
});

// sie_gateway_generation_cancelled_total
//
// Counts streaming generations cancelled by the client. ``stage`` is
// ``before_first_chunk`` (cancel hit before any reply arrived) or
// ``mid_stream`` (cancel hit after at least one chunk).
pub static GENERATION_CANCELLED: LazyLock<CounterVec> = LazyLock::new(|| {
    CounterVec::new(
        Opts::new(
            "sie_gateway_generation_cancelled_total",
            "Streaming generations cancelled by client disconnect",
        ),
        &["model", "pool", "stage"],
    )
    .unwrap()
});

// HRW direct-dispatch fallback counter.
//
// Each fallback reason is recorded against its `(model, pool)` so
// dashboards can see whether one model is uniformly degrading vs
// gateway-wide health. Reason values are an enumerated set:
// `unhealthy_skipped` — HRW pick was filtered out because the worker
//   flipped to Unhealthy between snapshot build and dispatch (also
//   covers "no worker has this model loaded" since the ring is empty
//   in both cases — disambiguate via `sie_gateway_workers_total`).
// `saturated_skipped` — HRW pick was excluded because the worker's
//   saturation flag is true.
// `no_key` — the request carried no `routing_key`, no
//   `prompt_cache_key`, and no prompt to fall back on, so HRW
//   couldn't pick. Gateway fell back to pool round-robin.
// `nak_kv_budget` / `nak_model_not_loaded` / `nak_worker_shutting_down`
//   — worker emitted a `kind:"nak"` inbox envelope with the
//   corresponding `reason`; gateway republished to the pool.
// `nak` — same as above but for an unrecognised `reason` value
//   (forward-compat catch-all so we never lose signal).
// `first_chunk_timeout` — direct-dispatched worker never sent a
//   chunk within the first-chunk window; gateway republished to
//   the pool.
pub static ROUTING_FALLBACK_TOTAL: LazyLock<CounterVec> = LazyLock::new(|| {
    CounterVec::new(
        Opts::new(
            "sie_gateway_routing_fallback_total",
            "HRW direct-dispatch fallbacks by reason",
        ),
        &["model", "pool", "reason"],
    )
    .unwrap()
});

// sie_gateway_generation_fallback_refused_total
//
// H9 — first-chunk-fallback republishes refused by the gateway's per-
// (model, pool) token bucket. Today the only emitter is the
// ``rate_limited`` reason, set when [`WorkPublisher::republish_to_pool_outcome`]
// returns `RateLimited` instead of `Republished`. Future refusal
// reasons (admin disable, circuit breaker open) would land as
// additional values on the same counter without a schema bump.
pub static GENERATION_FALLBACK_REFUSED_TOTAL: LazyLock<CounterVec> = LazyLock::new(|| {
    CounterVec::new(
        Opts::new(
            "sie_gateway_generation_fallback_refused_total",
            "First-chunk-fallback republishes refused by the gateway (H9)",
        ),
        &["model", "pool", "reason"],
    )
    .unwrap()
});

// sie_gateway_rate_limit_total
//
// Counts 429 responses surfaced by the gateway, labelled by reason.
// Today's only emitter is the ``kv_pool_saturated`` path: worker
// NAKed with ``kv_budget`` *and* the gateway's pool-fallback republish
// also failed. Future per-tenant rate limiters would land additional
// ``reason`` values here without a metric-schema bump.
pub static RATE_LIMIT_TOTAL: LazyLock<CounterVec> = LazyLock::new(|| {
    CounterVec::new(
        Opts::new(
            "sie_gateway_rate_limit_total",
            "429 rate-limit responses surfaced by the gateway",
        ),
        &["model", "pool", "reason"],
    )
    .unwrap()
});

// Routing key source histogram-like counter.
//
// Records which input the routing key was sourced from for each
// request. Source values: `routing_key`, `prompt_cache_key`,
// `system_message`, `prompt_prefix`, `none`.
pub static ROUTING_KEY_SOURCE: LazyLock<CounterVec> = LazyLock::new(|| {
    CounterVec::new(
        Opts::new(
            "sie_gateway_routing_key_source_total",
            "Routing key resolutions by source field",
        ),
        &["model", "pool", "source"],
    )
    .unwrap()
});

// HRW ring size gauge.
//
// Exposes the current number of eligible workers per `(model, pool)`.
// The router builds the ring per-request so this gauge is set from the
// proxy on each pick; future optimisation may move ring construction
// to a background snapshot rebuild driven by the registry callbacks.
pub static ROUTING_HRW_RING_SIZE: LazyLock<prometheus::GaugeVec> = LazyLock::new(|| {
    prometheus::GaugeVec::new(
        Opts::new(
            "sie_gateway_routing_hrw_ring_size",
            "Current number of eligible workers in the HRW ring per (model, pool)",
        ),
        &["model", "pool"],
    )
    .unwrap()
});

// sie_gateway_routing_cache_hit_estimate
//
// §4.11 — hash-key collision rate proxy. Read by operators
// as an *estimate* of how often a request's routing key hashes to a
// worker that already holds the relevant prefix in KV cache. The
// gateway cannot observe cache hits directly (the worker doesn't
// report them), so this gauge surfaces the proxy signal the routing
// layer can measure: of the last N (model, pool) requests, the
// fraction whose HRW pick was the same as the previous request for
// the same routing key (i.e. the ring is stable and the key keeps
// landing on the same worker).
//
// The metric is a gauge rather than a counter because the routing
// layer maintains a small rolling window per (model, pool); the
// gauge value is the latest computed ratio in `[0.0, 1.0]`. Writers
// must clamp into that range before calling `.set`.
pub static ROUTING_CACHE_HIT_ESTIMATE: LazyLock<GaugeVec> = LazyLock::new(|| {
    GaugeVec::new(
        Opts::new(
            "sie_gateway_routing_cache_hit_estimate",
            "Rolling-window estimate of routing-key → worker affinity stability",
        ),
        &["model", "pool"],
    )
    .unwrap()
});

/// Update the routing cache-hit estimate for one (model, pool). The
/// caller is responsible for windowing logic and clamping `ratio` to
/// `[0.0, 1.0]`; this helper exists so the metric module owns the
/// label vocabulary.
///
/// Writers: the proxy routing path calls this on every successful
/// dispatch with a rolling-window observation. Until a richer windowing
/// implementation lands (tracked alongside the routing cache work),
/// the simplest legitimate signal is "did this request's HRW pick
/// match the previous one for the same key" — recorded as 0.0/1.0.
#[allow(dead_code)]
pub fn set_routing_cache_hit_estimate(model: &str, pool: &str, ratio: f64) {
    ROUTING_CACHE_HIT_ESTIMATE
        .with_label_values(&[&sanitize_model_label(model), &sanitize_label(pool)])
        .set(ratio.clamp(0.0, 1.0));
}

// sie_gateway_kv_reservation_known
//
// §4.11 — gateway-side mirror of the worker's most recent
// KV-reservation report. The worker emits
// `sie_worker_generation_kv_reserved_tokens` on its own /metrics
// surface, but operators ask "what does the gateway *think* this
// worker has reserved right now" when debugging saturation routing.
// The gateway updates this gauge whenever it receives a worker
// status / saturation envelope from the inbox.
//
// Labels are `(pool, worker)` per spec — the gateway doesn't break
// down worker-side reservations by model in the routing layer, so
// neither does this mirror.
pub static KV_RESERVATION_KNOWN: LazyLock<GaugeVec> = LazyLock::new(|| {
    GaugeVec::new(
        Opts::new(
            "sie_gateway_kv_reservation_known",
            "Gateway's last-known KV-reservation token count per worker (mirrors worker reports)",
        ),
        &["pool", "worker"],
    )
    .unwrap()
});

/// Update the gateway's view of a worker's reserved KV-cache tokens.
/// Call sites: status-envelope handlers and the routing snapshot
/// rebuild. A worker that disappears (leaves the registry) should
/// have its series reset to zero by the registry sweep path.
///
/// The `WorkerStatusMessage` envelope does not currently carry a
/// `kv_reserved_tokens` field; once it does, the registry update path
/// will forward the value here. Until then, `update_worker_metrics`
/// seeds zero so the family is non-empty on /metrics scrapes.
#[allow(dead_code)]
pub fn set_kv_reservation_known(pool: &str, worker: &str, tokens: f64) {
    KV_RESERVATION_KNOWN
        .with_label_values(&[&sanitize_label(pool), worker])
        .set(tokens);
}

pub static DLQ_EVENTS: LazyLock<CounterVec> = LazyLock::new(|| {
    CounterVec::new(
        Opts::new(
            "sie_gateway_dlq_events_total",
            "Messages successfully forwarded to dead letter queue after max delivery attempts",
        ),
        &["stream", "consumer"],
    )
    .unwrap()
});

// sie_gateway_grammar_reject_total
//
// Counts gateway-side rejections of structured-output
// requests, labelled by the precise reason so dashboards can
// distinguish payload-size hits from depth violations from
// capability-gate rejections. The label vocabulary is fixed by
// :func:`handlers::grammar::record_reject` — adding a new reason
// requires updating the constant table there to keep cardinality
// bounded.
pub static GRAMMAR_REJECTS: LazyLock<CounterVec> = LazyLock::new(|| {
    CounterVec::new(
        Opts::new(
            "sie_gateway_grammar_reject_total",
            "Grammar requests rejected by the gateway, broken down by reason",
        ),
        &["reason"],
    )
    .unwrap()
});

/// Bump :static:`GRAMMAR_REJECTS` for ``reason``. Helper exists so
/// call sites don't need to import the metric directly — keeps the
/// label vocabulary in one place.
pub fn record_grammar_reject(reason: &str) {
    GRAMMAR_REJECTS.with_label_values(&[reason]).inc();
}

// sie_gateway_config_bootstrap_failures_total
//
// Incremented every time the background bootstrap retry task
// (`state::config_bootstrap::spawn_bootstrap_retry`) fails to fetch a
// clean export from `sie-config`. A sustained nonzero rate on this
// counter combined with a stale `config_epoch` is the operator signal
// that the gateway has been serving the filesystem seed for a long time
// and is at risk of diverging from the control plane.
pub static CONFIG_BOOTSTRAP_FAILURES: LazyLock<Counter> = LazyLock::new(|| {
    Counter::new(
        "sie_gateway_config_bootstrap_failures_total",
        "Failed attempts to fetch the initial config snapshot from sie-config",
    )
    .unwrap()
});

// sie_gateway_config_bootstrap_degraded
//
// Binary gauge: 1 while the bootstrap has never succeeded AND at least
// `DEGRADED_THRESHOLD` has elapsed since the gateway started; 0 once
// the bootstrap catches up (or from the start if `sie-config` is not
// configured). Alerts on value=1 are the SRE-visible "gateway stuck on
// the filesystem seed" signal.
pub static CONFIG_BOOTSTRAP_DEGRADED: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::new(
        "sie_gateway_config_bootstrap_degraded",
        "1 iff the background bootstrap from sie-config has not yet succeeded after the degraded threshold",
    )
    .unwrap()
});

// sie_gateway_config_epoch
//
// The furthest-known control-plane epoch observed by this gateway.
// Updated by: bootstrap fetch, NATS Core delta handler, and the epoch
// poller. The single most useful alert in the whole system joins this
// against `sie_config_epoch` (published by sie-config):
//
//     (sie_config_epoch - on() sie_gateway_config_epoch) > 0
//
// If the gateway lags the control plane by any amount for more than a
// minute, either NATS deltas are not flowing or the poller is stuck.
pub static CONFIG_EPOCH: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::new(
        "sie_gateway_config_epoch",
        "Control-plane epoch currently applied on this gateway",
    )
    .unwrap()
});

// sie_gateway_config_deltas_total
//
// Counts config-delta events processed from NATS Core, split by the
// kind of delta and whether the apply succeeded. A nonzero
// `result="error"` rate is the signal to investigate payload
// compatibility or the poller; a sustained zero `result="applied"`
// rate during known config writes is the signal that NATS Core
// subscriptions are broken (the fallback epoch poller will catch the
// drift eventually, but slowly).
pub static CONFIG_DELTAS: LazyLock<CounterVec> = LazyLock::new(|| {
    CounterVec::new(
        Opts::new(
            "sie_gateway_config_deltas_total",
            "Config deltas processed from NATS Core, by kind and apply result",
        ),
        &["kind", "result"],
    )
    .unwrap()
});

// sie_gateway_nats_connected
//
// 1 iff the gateway is currently connected to NATS. Unlabeled: the
// gateway uses a single `async-nats` client for JetStream (inference
// work + results) and NATS Core (config deltas), so all three
// logical streams share one connection state. Introducing a `stream`
// label would suggest independent connectivity the code does not
// track and cannot claim. If we ever split clients, reintroduce the
// label then.
pub static NATS_CONNECTED: LazyLock<Gauge> = LazyLock::new(|| {
    Gauge::new(
        "sie_gateway_nats_connected",
        "1 iff the gateway is currently connected to NATS",
    )
    .unwrap()
});

// sie_gateway_pool_events_total
//
// Pool lifecycle counter. `ACTIVE_LEASE_GPUS` shows current state but
// not the rate of churn; this counter gives it. Use it with a 5m rate
// window to catch runaway pool creation or unexpected expirations.
pub static POOL_EVENTS: LazyLock<CounterVec> = LazyLock::new(|| {
    CounterVec::new(
        Opts::new("sie_gateway_pool_events_total", "Pool lifecycle events"),
        &["event"],
    )
    .unwrap()
});

// sie_gateway_dlq_republish_failures_total
//
// Companion to `DLQ_EVENTS`: the existing counter only counts
// *successful* DLQ forwards, so an outage in the DLQ publish path is
// completely invisible on the success counter alone. This one counts
// the other half.
pub static DLQ_REPUBLISH_FAILURES: LazyLock<CounterVec> = LazyLock::new(|| {
    CounterVec::new(
        Opts::new(
            "sie_gateway_dlq_republish_failures_total",
            "Messages that failed to be forwarded to the dead letter queue",
        ),
        &["stream", "consumer"],
    )
    .unwrap()
});

/// Helper: flip `NATS_CONNECTED` to 1.0 (connected) or 0.0
/// (disconnected). Centralized so call sites don't have to remember
/// the 0/1 convention.
pub fn set_nats_connected(connected: bool) {
    NATS_CONNECTED.set(if connected { 1.0 } else { 0.0 });
}

/// Pre-instantiate request/demand `*Vec` metrics with empty-string
/// label sentinels so the metric family always appears in `/metrics`,
/// even on a freshly booted gateway pod that has served zero proxied
/// traffic yet.
///
/// Without this, Prometheus' text encoder silently elides empty `*Vec`
/// families, so dashboards and alerts on a quiet gateway (right after
/// rolling restart, or in a dev cluster with no callers) see "no data"
/// for `sie_gateway_requests_total`, `sie_gateway_pending_demand`, etc.
/// until the first matching observation lands. The empty-label sample
/// is a flat zero that real queries naturally filter out
/// (`{machine_profile=~".+"}` etc.) and `rate()` over a zero series is
/// also zero, so dashboards that don't filter still show 0 instead of
/// "no data" — strictly an improvement.
pub fn init_metric_families() {
    REQUEST_COUNT.with_label_values(&["", "", ""]).inc_by(0.0);
    PROVISIONING_RESPONSES.with_label_values(&[""]).inc_by(0.0);
    PENDING_DEMAND.with_label_values(&["", "", ""]).set(0.0);
    POOL_WARM_FLOOR.with_label_values(&["", "", ""]).set(0.0);
    ACTIVE_LEASE_GPUS.with_label_values(&["", "", ""]).set(0.0);
    REJECTED_REQUESTS
        .with_label_values(&["", "", "", ""])
        .inc_by(0.0);
    // Seed the gauge families that other call sites write to,
    // so they appear in /metrics even before the first real
    // observation. Empty `*Vec` families would otherwise be elided
    // by the text encoder.
    ROUTING_CACHE_HIT_ESTIMATE
        .with_label_values(&["", ""])
        .set(0.0);
    KV_RESERVATION_KNOWN.with_label_values(&["", ""]).set(0.0);
}

pub struct WorkerSnapshot {
    pub name: String,
    pub machine_profile: String,
    pub bundle: String,
    /// The pool this worker has registered against (used by routing + metrics).
    /// Distinct from `machine_profile`: one profile (e.g. `l4-spot`)
    /// can serve multiple pools. Sourced from
    /// `WorkerStatusMessage.pool_name`. Empty string for legacy
    /// workers that pre-date the pool registration message.
    pub pool_name: String,
    pub queue_depth: i32,
    pub memory_used_bytes: i64,
    pub healthy: bool,
}

pub fn update_worker_metrics(workers: &[WorkerSnapshot]) {
    // Reset per-worker gauges before repopulating. Without this, a
    // worker that disappears (pod termination, GPU drain, bundle
    // reassignment) leaves a ghost series with its last-known queue
    // depth / memory stuck forever — dashboards would treat a
    // long-gone worker as still active. `ACTIVE_LEASE_GPUS` uses the
    // same reset-then-repopulate pattern in `update_pool_metrics`.
    WORKER_QUEUE_DEPTH.reset();
    WORKER_MEMORY_USED.reset();
    // Keep `kv_reservation_known` series in sync with the
    // current worker set. The actual value lands once the worker
    // status envelope carries `kv_reserved_tokens`; for now we seed
    // 0.0 so dashboards know which workers exist. Re-seed the empty
    // sentinel after `.reset()` so the family stays visible on
    // /metrics even when the gateway has zero registered workers
    // (idle boot, drained cluster).
    KV_RESERVATION_KNOWN.reset();
    KV_RESERVATION_KNOWN.with_label_values(&["", ""]).set(0.0);

    let mut healthy_count = 0;
    let mut unhealthy_count = 0;

    for w in workers {
        if w.healthy {
            healthy_count += 1;
        } else {
            unhealthy_count += 1;
        }
        let pool_name = w.pool_name.trim().to_ascii_lowercase();
        WORKER_QUEUE_DEPTH
            .with_label_values(&[&pool_name, &w.name, &w.machine_profile, &w.bundle])
            .set(w.queue_depth as f64);
        WORKER_MEMORY_USED
            .with_label_values(&[&w.name, &w.machine_profile, &w.bundle])
            .set(w.memory_used_bytes as f64);
        // §4.11 spec table: labels are `(pool, worker)`. The actual
        // KV-reservation value lands once `WorkerStatusMessage` carries
        // `kv_reserved_tokens`; seeding 0.0 keeps the series visible
        // (and ghost-free, thanks to the reset above).
        KV_RESERVATION_KNOWN
            .with_label_values(&[&pool_name, &w.name])
            .set(0.0);
    }

    // `WORKER_COUNT` has only two fixed label values (`healthy` /
    // `unhealthy`), so an always-set strategy is enough — nothing to
    // reset.
    WORKER_COUNT
        .with_label_values(&["healthy"])
        .set(healthy_count as f64);
    WORKER_COUNT
        .with_label_values(&["unhealthy"])
        .set(unhealthy_count as f64);
}

pub fn update_model_metrics(models: &std::collections::HashMap<String, usize>) {
    // Reset before repopulating: otherwise, models removed from the
    // registry (explicit deletion or config reload that drops them)
    // keep exporting their last-known worker count until the gateway
    // restarts. Scrape frequency is low enough that the reset window
    // is invisible to Prometheus rate() queries.
    MODEL_WORKERS.reset();
    for (name, count) in models {
        MODEL_WORKERS.with_label_values(&[name]).set(*count as f64);
    }
}

pub fn record_rejected_request_for_pool(
    pool: &str,
    machine_profile: &str,
    bundle: &str,
    reason: &str,
) {
    let effective_pool = if pool.is_empty() { "unknown" } else { pool };
    let effective_profile = if machine_profile.is_empty() {
        "unknown"
    } else {
        machine_profile
    };
    REJECTED_REQUESTS
        .with_label_values(&[effective_pool, effective_profile, bundle, reason])
        .inc();
}

/// Clear PENDING_DEMAND for WorkerGroups that now have healthy workers.
/// Called when worker metrics are updated to cancel stale demand signals.
/// Uses the DemandTracker to properly cancel expiry timers when clearing.
pub fn clear_fulfilled_demand(
    healthy_worker_groups: &std::collections::HashSet<(String, String, String)>,
    demand_tracker: &crate::state::demand_tracker::DemandTracker,
) {
    // Gather current demand label pairs from the metric
    let metric_families = REGISTRY.gather();
    for mf in &metric_families {
        if mf.name() != "sie_gateway_pending_demand" {
            continue;
        }
        for m in mf.get_metric() {
            let labels: std::collections::HashMap<&str, &str> = m
                .get_label()
                .iter()
                .map(|l| (l.name(), l.value()))
                .collect();
            if let Some(&gpu) = labels.get("machine_profile") {
                let pool = labels.get("pool").copied().unwrap_or("default");
                let bundle = labels.get("bundle").copied().unwrap_or("default");
                if healthy_worker_groups.contains(&(
                    pool.to_lowercase(),
                    gpu.to_lowercase(),
                    bundle.to_lowercase(),
                )) {
                    let current = PENDING_DEMAND.with_label_values(&[pool, gpu, bundle]).get();
                    if current > 0.0 {
                        demand_tracker.clear(pool, gpu, bundle);
                        tracing::info!(
                            pool = pool,
                            gpu = gpu,
                            bundle = bundle,
                            "cleared pending demand — healthy workers available"
                        );
                    }
                }
            }
        }
    }
}

pub fn update_pool_metrics(pools: &[crate::types::Pool]) {
    use crate::state::pool_manager::DEFAULT_POOL_NAME;
    use crate::types::PoolState;

    // Reset gauge to avoid stale entries from deleted pools, then
    // re-seed the empty-label sentinel so the family stays visible
    // in `/metrics` on an idle gateway whose only pool is the
    // default pool (which is skipped below by design). Without this
    // re-seed the `init_metric_families()` sentinel is wiped on the
    // very first scrape.
    ACTIVE_LEASE_GPUS.reset();
    ACTIVE_LEASE_GPUS.with_label_values(&["", "", ""]).set(0.0);

    for pool in pools {
        if pool.status.state == PoolState::Active && pool.spec.name != DEFAULT_POOL_NAME {
            if let Some(bundle) = pool.spec.bundle.as_deref() {
                for (gpu_type, count) in &pool.spec.gpus {
                    ACTIVE_LEASE_GPUS
                        .with_label_values(&[&pool.spec.name, gpu_type, bundle])
                        .add(*count as f64);
                }
            } else {
                let mut assigned_by_lane: std::collections::HashMap<(&str, &str), usize> =
                    std::collections::HashMap::new();
                for worker in &pool.status.assigned_workers {
                    *assigned_by_lane
                        .entry((&worker.gpu, &worker.bundle))
                        .or_default() += 1;
                }
                for ((gpu_type, bundle), count) in assigned_by_lane {
                    ACTIVE_LEASE_GPUS
                        .with_label_values(&[&pool.spec.name, gpu_type, bundle])
                        .add(count as f64);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::pool::{AssignedWorker, Pool, PoolSpec, PoolState, PoolStatus};
    use std::collections::HashMap;
    use std::sync::Mutex;

    static METRICS_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn reset_test_metrics() {
        ACTIVE_LEASE_GPUS.reset();
        WORKER_COUNT.reset();
        WORKER_QUEUE_DEPTH.reset();
        WORKER_MEMORY_USED.reset();
        MODEL_WORKERS.reset();
        PENDING_DEMAND.reset();
    }

    fn make_pool(
        name: &str,
        state: PoolState,
        gpus: HashMap<String, u32>,
        bundle: Option<String>,
    ) -> Pool {
        Pool {
            spec: PoolSpec {
                name: name.to_string(),
                bundle,
                gpus,
                gpu_caps: HashMap::new(),
                ttl_seconds: None,
                minimum_worker_count: 0,
            },
            status: PoolStatus {
                state,
                assigned_workers: vec![],
                created_at: 0.0,
                last_renewed: 0.0,
            },
        }
    }

    #[test]
    fn test_update_pool_metrics_active_pool() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        // Force registry init
        let _ = &*REGISTRY;
        reset_test_metrics();

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 2);
        let pools = vec![make_pool(
            "eval-pool",
            PoolState::Active,
            gpus,
            Some("premium".to_string()),
        )];

        update_pool_metrics(&pools);

        let val = ACTIVE_LEASE_GPUS
            .with_label_values(&["eval-pool", "l4-spot", "premium"])
            .get();
        assert!((val - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_update_pool_metrics_unfiltered_pool_uses_assigned_worker_bundles() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;
        reset_test_metrics();

        let mut gpus = HashMap::new();
        gpus.insert("rtx6000".to_string(), 2);
        let mut pool = make_pool("customer-acme", PoolState::Active, gpus, None);
        pool.status.assigned_workers = vec![
            AssignedWorker {
                name: "worker-default".to_string(),
                url: "http://worker-default:8080".to_string(),
                gpu: "rtx6000".to_string(),
                bundle: "default".to_string(),
            },
            AssignedWorker {
                name: "worker-sglang".to_string(),
                url: "http://worker-sglang:8080".to_string(),
                gpu: "rtx6000".to_string(),
                bundle: "sglang".to_string(),
            },
        ];

        update_pool_metrics(&[pool]);

        let default_val = ACTIVE_LEASE_GPUS
            .with_label_values(&["customer-acme", "rtx6000", "default"])
            .get();
        let sglang_val = ACTIVE_LEASE_GPUS
            .with_label_values(&["customer-acme", "rtx6000", "sglang"])
            .get();
        assert!((default_val - 1.0).abs() < f64::EPSILON);
        assert!((sglang_val - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_update_pool_metrics_skips_default_pool() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;
        reset_test_metrics();

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        let pools = vec![make_pool("default", PoolState::Active, gpus, None)];

        update_pool_metrics(&pools);

        // Default pool should not contribute to active lease GPUs
        let val = ACTIVE_LEASE_GPUS
            .with_label_values(&["default", "l4-spot", "default"])
            .get();
        assert!((val - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_update_pool_metrics_skips_pending() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;
        reset_test_metrics();

        let mut gpus = HashMap::new();
        gpus.insert("a100".to_string(), 4);
        let pools = vec![make_pool(
            "pending-pool",
            PoolState::Pending,
            gpus,
            Some("default".to_string()),
        )];

        update_pool_metrics(&pools);

        let val = ACTIVE_LEASE_GPUS
            .with_label_values(&["pending-pool", "a100", "default"])
            .get();
        assert!((val - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_update_pool_metrics_reset_clears_stale() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;
        reset_test_metrics();

        ACTIVE_LEASE_GPUS
            .with_label_values(&["stale-pool", "stale-gpu", "stale-bundle"])
            .set(10.0);

        update_pool_metrics(&[]);

        // Stale series must be gone, but the empty-label sentinel
        // re-seeded by `update_pool_metrics` itself must remain so the
        // family stays visible in `/metrics`.
        let families = REGISTRY.gather();
        let lease = families
            .iter()
            .find(|mf| mf.name() == "sie_gateway_active_lease_gpus")
            .expect("sie_gateway_active_lease_gpus should still be present after reset");
        let stale_present = lease.get_metric().iter().any(|m| {
            m.get_label().iter().any(|l| {
                (l.name() == "machine_profile" && l.value() == "stale-gpu")
                    || (l.name() == "bundle" && l.value() == "stale-bundle")
            })
        });
        assert!(!stale_present, "stale label combination survived reset");
        let sentinel_present = lease
            .get_metric()
            .iter()
            .any(|m| m.get_label().iter().all(|l| l.value().is_empty()));
        assert!(
            sentinel_present,
            "empty-label sentinel missing after update_pool_metrics(&[])"
        );
    }

    #[test]
    fn test_update_worker_metrics() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;
        reset_test_metrics();

        let workers = vec![
            WorkerSnapshot {
                name: "w1".to_string(),
                machine_profile: "l4".to_string(),
                bundle: "default".to_string(),
                pool_name: "pool-a".to_string(),
                queue_depth: 5,
                memory_used_bytes: 1000,
                healthy: true,
            },
            WorkerSnapshot {
                name: "w2".to_string(),
                machine_profile: "a100".to_string(),
                bundle: "premium".to_string(),
                pool_name: "pool-b".to_string(),
                queue_depth: 2,
                memory_used_bytes: 2000,
                healthy: false,
            },
        ];

        update_worker_metrics(&workers);

        let healthy = WORKER_COUNT.with_label_values(&["healthy"]).get();
        let unhealthy = WORKER_COUNT.with_label_values(&["unhealthy"]).get();
        assert!((healthy - 1.0).abs() < f64::EPSILON);
        assert!((unhealthy - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_update_worker_metrics_canonicalizes_pool_label_case() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;
        reset_test_metrics();

        update_worker_metrics(&[WorkerSnapshot {
            name: "w1".to_string(),
            machine_profile: "l4".to_string(),
            bundle: "default".to_string(),
            pool_name: "Customer-Acme".to_string(),
            queue_depth: 5,
            memory_used_bytes: 1000,
            healthy: true,
        }]);

        let canonical = WORKER_QUEUE_DEPTH
            .with_label_values(&["customer-acme", "w1", "l4", "default"])
            .get();
        assert!((canonical - 5.0).abs() < f64::EPSILON);

        let families = REGISTRY.gather();
        let mixed_case_present = families
            .iter()
            .filter(|mf| mf.name() == "sie_gateway_worker_queue_depth")
            .flat_map(|mf| mf.get_metric().iter())
            .any(|m| {
                m.get_label()
                    .iter()
                    .any(|l| l.name() == "pool" && l.value() == "Customer-Acme")
            });
        assert!(
            !mixed_case_present,
            "worker queue-depth labels must use canonical lowercase pool names"
        );
    }

    #[test]
    fn test_update_model_metrics() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;
        reset_test_metrics();

        let mut models = HashMap::new();
        models.insert("BAAI/bge-m3".to_string(), 3usize);
        models.insert("openai/clip".to_string(), 1usize);

        update_model_metrics(&models);

        let val = MODEL_WORKERS.with_label_values(&["BAAI/bge-m3"]).get();
        assert!((val - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_update_worker_metrics_evicts_stale_series() {
        // Regression: previously, a worker that disappeared between
        // scrapes (pod termination, GPU drain) left a ghost series
        // with its last-known queue depth / memory. We now reset
        // before repopulating, so only the current snapshot is
        // visible.
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;
        reset_test_metrics();

        let first = vec![WorkerSnapshot {
            name: "ghost-worker".to_string(),
            machine_profile: "l4-spot".to_string(),
            bundle: "default".to_string(),
            pool_name: "default-pool".to_string(),
            queue_depth: 42,
            memory_used_bytes: 123_456,
            healthy: true,
        }];
        update_worker_metrics(&first);
        let d = WORKER_QUEUE_DEPTH
            .with_label_values(&["default-pool", "ghost-worker", "l4-spot", "default"])
            .get();
        assert!((d - 42.0).abs() < f64::EPSILON);

        // Simulate the worker disappearing: next snapshot does not
        // include it. A different worker is reported instead.
        let second = vec![WorkerSnapshot {
            name: "other-worker".to_string(),
            machine_profile: "a100".to_string(),
            bundle: "premium".to_string(),
            pool_name: "premium-pool".to_string(),
            queue_depth: 7,
            memory_used_bytes: 999,
            healthy: true,
        }];
        update_worker_metrics(&second);

        // The new worker is visible.
        let d2 = WORKER_QUEUE_DEPTH
            .with_label_values(&["premium-pool", "other-worker", "a100", "premium"])
            .get();
        assert!((d2 - 7.0).abs() < f64::EPSILON);

        // The stale series must be gone. `get()` on a
        // non-existent label set would create it at 0.0; instead we
        // inspect the registry directly to confirm the series was
        // dropped.
        let families = REGISTRY.gather();
        let stale_present = families
            .iter()
            .filter(|mf| mf.name() == "sie_gateway_worker_queue_depth")
            .flat_map(|mf| mf.get_metric().iter())
            .any(|m| {
                m.get_label()
                    .iter()
                    .any(|l| l.name() == "worker" && l.value() == "ghost-worker")
            });
        assert!(
            !stale_present,
            "ghost-worker series must be evicted after it disappears from the snapshot"
        );
    }

    #[test]
    fn test_update_model_metrics_evicts_stale_series() {
        // Same regression for MODEL_WORKERS: removing a model from
        // the registry must clear its series.
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;
        reset_test_metrics();

        let mut first = HashMap::new();
        first.insert("org/ghost-model".to_string(), 5usize);
        update_model_metrics(&first);
        let d = MODEL_WORKERS.with_label_values(&["org/ghost-model"]).get();
        assert!((d - 5.0).abs() < f64::EPSILON);

        let mut second = HashMap::new();
        second.insert("org/other-model".to_string(), 2usize);
        update_model_metrics(&second);

        let families = REGISTRY.gather();
        let stale_present = families
            .iter()
            .filter(|mf| mf.name() == "sie_gateway_model_workers")
            .flat_map(|mf| mf.get_metric().iter())
            .any(|m| {
                m.get_label()
                    .iter()
                    .any(|l| l.name() == "model" && l.value() == "org/ghost-model")
            });
        assert!(
            !stale_present,
            "ghost-model series must be evicted after it disappears from the registry"
        );
    }

    #[test]
    fn test_clear_fulfilled_demand() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;
        reset_test_metrics();

        let tracker = crate::state::demand_tracker::DemandTracker::new();

        // Set demand for l4-spot
        PENDING_DEMAND
            .with_label_values(&["default", "l4-spot", "default"])
            .set(1.0);

        // Healthy workers include default/l4-spot/default.
        let mut worker_groups = std::collections::HashSet::new();
        worker_groups.insert((
            "default".to_string(),
            "l4-spot".to_string(),
            "default".to_string(),
        ));

        clear_fulfilled_demand(&worker_groups, &tracker);

        let val = PENDING_DEMAND
            .with_label_values(&["default", "l4-spot", "default"])
            .get();
        assert!((val - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clear_fulfilled_demand_preserves_unmatched() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;
        reset_test_metrics();

        let tracker = crate::state::demand_tracker::DemandTracker::new();

        // Set demand for h100 (no healthy workers)
        PENDING_DEMAND
            .with_label_values(&["default", "h100", "default"])
            .set(1.0);

        // Only default/l4-spot/default is healthy.
        let mut worker_groups = std::collections::HashSet::new();
        worker_groups.insert((
            "default".to_string(),
            "l4-spot".to_string(),
            "default".to_string(),
        ));

        clear_fulfilled_demand(&worker_groups, &tracker);

        // h100 demand should remain
        let val = PENDING_DEMAND
            .with_label_values(&["default", "h100", "default"])
            .get();
        assert!((val - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clear_fulfilled_demand_preserves_different_bundle() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;
        reset_test_metrics();

        let tracker = crate::state::demand_tracker::DemandTracker::new();

        // Same machine profile has demand in two bundles.
        PENDING_DEMAND
            .with_label_values(&["default", "l4-spot", "default"])
            .set(1.0);
        PENDING_DEMAND
            .with_label_values(&["default", "l4-spot", "sglang"])
            .set(1.0);
        PENDING_DEMAND
            .with_label_values(&["tenant-a", "l4-spot", "default"])
            .set(1.0);

        // A healthy default-pool default-bundle worker must not clear
        // sglang demand or another pool's default-bundle demand.
        let mut worker_groups = std::collections::HashSet::new();
        worker_groups.insert((
            "default".to_string(),
            "l4-spot".to_string(),
            "default".to_string(),
        ));

        clear_fulfilled_demand(&worker_groups, &tracker);

        let default_val = PENDING_DEMAND
            .with_label_values(&["default", "l4-spot", "default"])
            .get();
        let sglang_val = PENDING_DEMAND
            .with_label_values(&["default", "l4-spot", "sglang"])
            .get();
        let tenant_val = PENDING_DEMAND
            .with_label_values(&["tenant-a", "l4-spot", "default"])
            .get();
        assert!((default_val - 0.0).abs() < f64::EPSILON);
        assert!((sglang_val - 1.0).abs() < f64::EPSILON);
        assert!((tenant_val - 1.0).abs() < f64::EPSILON);
    }

    // ------------------------------------------------------------------
    // Direct emission tests for the 5 metrics added in the audit pass.
    //
    // These tests read the global Prometheus registry, so they share
    // `METRICS_TEST_LOCK` with every other test in this module to avoid
    // racing with concurrent increments.
    // ------------------------------------------------------------------

    #[test]
    fn test_config_epoch_mirrors_on_set_max() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;

        let epoch = crate::state::config_epoch::ConfigEpoch::new();
        // Pick a baseline well above any value other tests would set so
        // our assertions aren't clobbered by an unrelated test that ran
        // before we took the lock (Prometheus counters/gauges are global).
        let base = CONFIG_EPOCH.get().max(0) as u64 + 1_000_000;

        assert!(epoch.set_max(base + 1));
        assert_eq!(CONFIG_EPOCH.get(), (base + 1) as i64);

        // Non-advancing set_max is a no-op on the gauge too.
        CONFIG_EPOCH.set((base + 1) as i64);
        assert!(!epoch.set_max(base));
        assert_eq!(CONFIG_EPOCH.get(), (base + 1) as i64);

        // force_set mirrors even on a backward move (recovery path).
        epoch.force_set(base);
        assert_eq!(CONFIG_EPOCH.get(), base as i64);
    }

    #[test]
    fn test_config_deltas_labels_and_increment() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;

        // All four `(kind, result)` pairs we emit in manager.rs must
        // register on the CounterVec. We just verify that `.inc()` moves
        // the observed value; the label validation is implicit (Prometheus
        // would panic on a cardinality mismatch).
        for labels in &[
            ("model_added", "applied"),
            ("model_added", "parse_error"),
            ("model_added", "apply_error"),
            ("model_added", "rejected_untrusted"),
            ("epoch_bump", "applied"),
            ("epoch_bump", "rejected_untrusted"),
        ] {
            let before = CONFIG_DELTAS.with_label_values(&[labels.0, labels.1]).get();
            CONFIG_DELTAS.with_label_values(&[labels.0, labels.1]).inc();
            let after = CONFIG_DELTAS.with_label_values(&[labels.0, labels.1]).get();
            assert!(
                (after - before - 1.0).abs() < f64::EPSILON,
                "({}, {}) did not increment",
                labels.0,
                labels.1
            );
        }
    }

    #[test]
    fn test_nats_connected_gauge_flips() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;

        set_nats_connected(true);
        assert_eq!(NATS_CONNECTED.get(), 1.0);

        set_nats_connected(false);
        assert_eq!(NATS_CONNECTED.get(), 0.0);

        // Idempotent: flipping to the same value does not panic or
        // double-count (gauges are set, not incremented).
        set_nats_connected(false);
        assert_eq!(NATS_CONNECTED.get(), 0.0);
    }

    #[test]
    fn test_pool_events_labels_and_increment() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;

        for event in &["created", "updated", "renewed", "deleted", "expired"] {
            let before = POOL_EVENTS.with_label_values(&[event]).get();
            POOL_EVENTS.with_label_values(&[event]).inc();
            let after = POOL_EVENTS.with_label_values(&[event]).get();
            assert!(
                (after - before - 1.0).abs() < f64::EPSILON,
                "pool_events{{event={}}} did not increment",
                event
            );
        }
    }

    #[test]
    fn test_record_generation_success_observes_histograms_and_counters() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;

        // Capture baselines because counters/histograms are global and
        // other tests in this module may have observed values too.
        let prompt_before = GENERATION_TOTAL_TOKENS
            .with_label_values(&["m", "p", "prompt"])
            .get();
        let completion_before = GENERATION_TOTAL_TOKENS
            .with_label_values(&["m", "p", "completion"])
            .get();
        let ttft_before = GENERATION_TTFT
            .with_label_values(&["m", "p", "none"])
            .get_sample_count();
        let tpot_before = GENERATION_TPOT
            .with_label_values(&["m", "p", "none"])
            .get_sample_count();

        let usage = crate::queue::streaming::UsageBlock {
            prompt_tokens: 7,
            completion_tokens: 11,
            total_tokens: 18,
        };
        record_generation_success("m", "p", "none", Some(123.0), Some(45.0), Some(&usage));

        assert!(
            (GENERATION_TOTAL_TOKENS
                .with_label_values(&["m", "p", "prompt"])
                .get()
                - prompt_before
                - 7.0)
                .abs()
                < f64::EPSILON
        );
        assert!(
            (GENERATION_TOTAL_TOKENS
                .with_label_values(&["m", "p", "completion"])
                .get()
                - completion_before
                - 11.0)
                .abs()
                < f64::EPSILON
        );
        assert_eq!(
            GENERATION_TTFT
                .with_label_values(&["m", "p", "none"])
                .get_sample_count(),
            ttft_before + 1
        );
        assert_eq!(
            GENERATION_TPOT
                .with_label_values(&["m", "p", "none"])
                .get_sample_count(),
            tpot_before + 1
        );
    }

    #[test]
    fn test_record_generation_success_skips_missing_observations() {
        // Stale-attempt / early-error outcomes can land here with
        // ttft/tpot/usage all None. The helper must not panic and must
        // not observe spurious zero buckets.
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;

        let ttft_before = GENERATION_TTFT
            .with_label_values(&["m2", "p2", "none"])
            .get_sample_count();
        let tpot_before = GENERATION_TPOT
            .with_label_values(&["m2", "p2", "none"])
            .get_sample_count();

        record_generation_success("m2", "p2", "none", None, None, None);

        assert_eq!(
            GENERATION_TTFT
                .with_label_values(&["m2", "p2", "none"])
                .get_sample_count(),
            ttft_before
        );
        assert_eq!(
            GENERATION_TPOT
                .with_label_values(&["m2", "p2", "none"])
                .get_sample_count(),
            tpot_before
        );
    }

    #[test]
    fn test_record_generation_success_label_cardinality_bounded() {
        // Metrics-rollout acceptance: no caller-controlled label feeds these
        // metrics. Two requests against the same (model, pool) must
        // *not* create new label series, regardless of any user-side
        // identifier (which is never passed in).
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;

        let usage = crate::queue::streaming::UsageBlock {
            prompt_tokens: 1,
            completion_tokens: 2,
            total_tokens: 3,
        };
        record_generation_success(
            "card/model",
            "default",
            "none",
            Some(10.0),
            Some(1.0),
            Some(&usage),
        );
        record_generation_success(
            "card/model",
            "default",
            "none",
            Some(11.0),
            Some(1.1),
            Some(&usage),
        );

        // After two records, exactly one (model, pool, grammar) series
        // should exist on each histogram and exactly two on the token
        // counter (one each for prompt/completion).
        let families = REGISTRY.gather();
        let ttft_series = families
            .iter()
            .find(|mf| mf.name() == "sie_gateway_generation_ttft_seconds")
            .expect("ttft family present")
            .get_metric()
            .iter()
            .filter(|m| {
                m.get_label()
                    .iter()
                    .any(|l| l.name() == "model" && l.value() == "card/model")
            })
            .count();
        assert_eq!(
            ttft_series, 1,
            "TTFT must collapse to one series per (model, pool, grammar)"
        );

        let token_series = families
            .iter()
            .find(|mf| mf.name() == "sie_gateway_generation_total_tokens")
            .expect("total_tokens family present")
            .get_metric()
            .iter()
            .filter(|m| {
                m.get_label()
                    .iter()
                    .any(|l| l.name() == "model" && l.value() == "card/model")
            })
            .count();
        assert_eq!(
            token_series, 2,
            "total_tokens must have exactly prompt+completion series per (model, pool)"
        );
    }

    #[test]
    fn test_sanitize_model_label_bounds_attacker_input() {
        // Real org/name model ids must pass through unchanged (the whole
        // point of a model-specific sanitizer over `sanitize_label`,
        // which rejects `/`).
        assert_eq!(sanitize_model_label("BAAI/bge-m3"), "BAAI/bge-m3");
        assert_eq!(
            sanitize_model_label("Qwen/Qwen3-4B-Instruct-2507"),
            "Qwen/Qwen3-4B-Instruct-2507"
        );
        // Profile-variant form (`base:profile`) is also legal.
        assert_eq!(sanitize_model_label("org/m:a100"), "org/m:a100");

        // Empty → unknown.
        assert_eq!(sanitize_model_label(""), "unknown");

        // Subject-/SSE-/log-dangerous chars and oversized inputs collapse
        // to a single bounded sentinel.
        assert_eq!(sanitize_model_label(&"x".repeat(10_000)), "invalid");
        assert_eq!(sanitize_model_label("evil\nmodel"), "invalid");
        assert_eq!(sanitize_model_label("a*b"), "invalid");
        assert_eq!(sanitize_model_label("a>b"), "invalid");
        assert_eq!(sanitize_model_label("a b"), "invalid");
        assert_eq!(sanitize_model_label("emoji😀"), "invalid");
    }

    #[test]
    fn test_record_generation_success_model_label_cardinality_attacker_varied() {
        // H1 acceptance: attacker-varied model strings must NOT mint an
        // unbounded number of label series. Junk inputs all collapse to
        // the single `"invalid"` series; an empty id to `"unknown"`.
        // Case-variants of an *unknown* model do NOT collapse here (that
        // is the upstream canonicalisation's job), but they remain
        // bounded valid-charset labels rather than unbounded junk.
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;

        let attacker_models: [String; 8] = [
            "Org/Model".to_string(),
            "org/model".to_string(),
            "ORG/MODEL".to_string(),
            "a".repeat(10_000),
            "x\ninjected".to_string(),
            "wild*card".to_string(),
            "sub>ject".to_string(),
            "white space".to_string(),
        ];
        for m in &attacker_models {
            record_generation_success(m, "atk-pool", "none", Some(1.0), Some(1.0), None);
        }

        // Collect the distinct model-label values that landed on the
        // TTFT family for our attack pool.
        let families = REGISTRY.gather();
        let mut model_values: std::collections::HashSet<String> = std::collections::HashSet::new();
        for mf in &families {
            if mf.name() != "sie_gateway_generation_ttft_seconds" {
                continue;
            }
            for m in mf.get_metric() {
                let labels = m.get_label();
                let is_atk_pool = labels
                    .iter()
                    .any(|l| l.name() == "pool" && l.value() == "atk-pool");
                if is_atk_pool {
                    if let Some(model_label) = labels.iter().find(|l| l.name() == "model") {
                        model_values.insert(model_label.value().to_string());
                    }
                }
            }
        }

        // The 5 junk inputs (10KB, `\n`, `*`, `>`, whitespace) must all
        // be the single `"invalid"` sentinel, and the 3 case-variants of
        // `org/model` pass through verbatim (valid charset). So at most 4
        // distinct series: invalid + the 3 case variants. Crucially,
        // NONE of the raw junk strings appear as a label.
        assert!(
            model_values.contains("invalid"),
            "junk model ids must collapse to `invalid`, got: {model_values:?}"
        );
        assert!(
            model_values.len() <= 4,
            "attacker-varied models must stay bounded, got {} series: {model_values:?}",
            model_values.len()
        );
        let raw_junk: [String; 5] = [
            "a".repeat(10_000),
            "x\ninjected".to_string(),
            "wild*card".to_string(),
            "sub>ject".to_string(),
            "white space".to_string(),
        ];
        for junk in &raw_junk {
            assert!(
                !model_values.contains(junk),
                "raw junk must never appear as a label"
            );
        }
    }

    #[test]
    fn test_dlq_republish_failures_increment() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;

        let before = DLQ_REPUBLISH_FAILURES
            .with_label_values(&["SIE_WORK", "gateway-consumer"])
            .get();
        DLQ_REPUBLISH_FAILURES
            .with_label_values(&["SIE_WORK", "gateway-consumer"])
            .inc();
        let after = DLQ_REPUBLISH_FAILURES
            .with_label_values(&["SIE_WORK", "gateway-consumer"])
            .get();
        assert!((after - before - 1.0).abs() < f64::EPSILON);
    }

    /// Without `init_metric_families`, `*Vec` metrics with no
    /// observations are silently elided by Prometheus' text encoder,
    /// so a freshly booted gateway shows "no data" for the
    /// request/demand panels. This test pins the contract: after
    /// resetting the relevant vecs and calling the init helper, every
    /// targeted family must show up in `REGISTRY.gather()`.
    #[test]
    fn test_init_metric_families_exposes_empty_vecs() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;

        REQUEST_COUNT.reset();
        PROVISIONING_RESPONSES.reset();
        PENDING_DEMAND.reset();
        ACTIVE_LEASE_GPUS.reset();
        REJECTED_REQUESTS.reset();

        // Sanity: post-reset, none of the targeted families should
        // appear in `gather()`. If this precondition ever stops
        // holding, the test below stops proving anything useful.
        let pre = REGISTRY.gather();
        for needle in [
            "sie_gateway_requests_total",
            "sie_gateway_pending_demand",
            "sie_gateway_active_lease_gpus",
            "sie_gateway_rejected_requests_total",
            "sie_gateway_provisioning_responses_total",
        ] {
            assert!(
                !pre.iter().any(|mf| mf.name() == needle),
                "precondition violated: {needle} still in gather() after reset"
            );
        }

        init_metric_families();
        // Simulate a scrape on an idle gateway: no pools, no workers,
        // no models. `update_pool_metrics(&[])` is the riskiest path
        // because it calls `ACTIVE_LEASE_GPUS.reset()` before
        // repopulating, and on an idle gateway there's nothing to
        // repopulate from. If the family disappears here, the bug
        // ships unnoticed.
        update_pool_metrics(&[]);

        let post = REGISTRY.gather();
        for needle in [
            "sie_gateway_requests_total",
            "sie_gateway_pending_demand",
            "sie_gateway_active_lease_gpus",
            "sie_gateway_rejected_requests_total",
            "sie_gateway_provisioning_responses_total",
        ] {
            let found = post
                .iter()
                .find(|mf| mf.name() == needle)
                .unwrap_or_else(|| panic!("{needle} missing after init + scrape path"));
            assert!(
                !found.get_metric().is_empty(),
                "{needle} family present but has zero samples"
            );
        }
    }

    /// §67 + §75: every gateway-side §4.11 metric family
    /// is registered with the REGISTRY and appears in
    /// `REGISTRY.gather()` after `init_metric_families()` runs and a
    /// single generation success is recorded. Mirrors the
    /// `test_full_section_4_11_worker_metric_surface_is_emitted` Python
    /// test on the worker side.
    #[test]
    fn test_full_section_4_11_gateway_metric_surface_is_registered() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;
        init_metric_families();

        // Drive one observation through each family we own. Other
        // families (timeouts, cancellations, stale-attempt drops) are
        // exercised by their owning slice's tests; presence in
        // gather() is enough.
        let usage = crate::queue::streaming::UsageBlock {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
        };
        record_generation_success(
            "surface/model",
            "surface/pool",
            "json_schema",
            Some(1.0),
            Some(0.1),
            Some(&usage),
        );
        ROUTING_FALLBACK_TOTAL
            .with_label_values(&["surface/model", "surface/pool", "no_key"])
            .inc();
        GENERATION_STALE_ATTEMPT_CHUNKS
            .with_label_values(&["surface/model", "surface/pool"])
            .inc();
        GENERATION_TIMEOUTS
            .with_label_values(&["surface/model", "surface/pool", "first_chunk"])
            .inc();
        GENERATION_CANCELLED
            .with_label_values(&["surface/model", "surface/pool", "mid_stream"])
            .inc();
        set_kv_reservation_known("surface/pool", "surface-worker", 256.0);
        set_routing_cache_hit_estimate("surface/model", "surface/pool", 0.75);

        let families: std::collections::HashSet<String> = REGISTRY
            .gather()
            .iter()
            .map(|mf| mf.name().to_string())
            .collect();
        for expected in [
            "sie_gateway_generation_ttft_seconds",
            "sie_gateway_generation_tpot_seconds",
            "sie_gateway_generation_total_tokens",
            "sie_gateway_routing_fallback_total",
            "sie_gateway_generation_stale_attempt_chunks_total",
            "sie_gateway_routing_cache_hit_estimate",
            "sie_gateway_kv_reservation_known",
            "sie_gateway_generation_timeout_total",
            "sie_gateway_generation_cancelled_total",
        ] {
            assert!(
                families.contains(expected),
                "§4.11 gateway metric {expected:?} missing from REGISTRY.gather()"
            );
        }
    }

    /// The warm-floor gauge must be registered so KEDA can scrape it from
    /// `/metrics`; an unregistered series never reaches the encoder.
    #[test]
    fn test_pool_warm_floor_is_registered() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;
        // Drive one observation so the family has at least one sample.
        set_pool_warm_floor("warm-pool", "l4", "default", 1.0);
        let families: std::collections::HashSet<String> = REGISTRY
            .gather()
            .iter()
            .map(|mf| mf.name().to_string())
            .collect();
        assert!(
            families.contains("sie_gateway_pool_warm_floor"),
            "sie_gateway_pool_warm_floor missing from REGISTRY.gather()"
        );
        clear_pool_warm_floor("warm-pool", "l4", "default");
    }

    /// `set_pool_warm_floor` writes `value` and `clear_pool_warm_floor`
    /// zeroes it, both keyed in `(pool, machine_profile, bundle)` label
    /// order, across multiple independent lanes.
    #[test]
    fn test_pool_warm_floor_set_and_clear_multiple_lanes() {
        let _guard = METRICS_TEST_LOCK.lock().unwrap();
        let _ = &*REGISTRY;
        POOL_WARM_FLOOR.reset();

        set_pool_warm_floor("tenant", "l4", "default", 2.0);
        set_pool_warm_floor("tenant", "a100", "sglang", 3.0);

        assert!(
            (POOL_WARM_FLOOR
                .with_label_values(&["tenant", "l4", "default"])
                .get()
                - 2.0)
                .abs()
                < f64::EPSILON
        );
        assert!(
            (POOL_WARM_FLOOR
                .with_label_values(&["tenant", "a100", "sglang"])
                .get()
                - 3.0)
                .abs()
                < f64::EPSILON
        );

        clear_pool_warm_floor("tenant", "l4", "default");
        assert!(
            POOL_WARM_FLOOR
                .with_label_values(&["tenant", "l4", "default"])
                .get()
                .abs()
                < f64::EPSILON
        );
        // The other lane is untouched by clearing the first.
        assert!(
            (POOL_WARM_FLOOR
                .with_label_values(&["tenant", "a100", "sglang"])
                .get()
                - 3.0)
                .abs()
                < f64::EPSILON
        );
        POOL_WARM_FLOOR.reset();
    }
}
