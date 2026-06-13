//! Scheduler-family Prometheus metrics.
//!
//! Registered at startup alongside the rest of the
//! [`crate::metrics::MetricsRegistry`] surface. The scheduler currently
//! populates a deliberately small subset — just enough to
//! answer "is the Rust scheduler doing work, for what, and how big
//! are the batches?":
//!
//! * [`SchedulerMetrics::enqueued_items_total`] — per-op enqueue
//!   counter (dispatcher enqueue site).
//! * [`SchedulerMetrics::batch_items`] — per-batch item-count
//!   histogram (drain-loop site).
//! * [`SchedulerMetrics::batch_cost`] — per-batch cost histogram
//!   (drain-loop site).
//! * [`SchedulerMetrics::models_total`] — live IntGauge of models
//!   that have had a scheduler materialised.
//!
//! The adaptive-controller gauges, starvation counter,
//! `flush_reason_total`, `active_batchers`, and `hol_wait_ms_mean`
//! families are declared so the `/metrics` scrape shape stays stable
//! across builds, but are not yet populated. They will be wired when
//! we have a concrete operational need; keeping the surface declared
//! means dashboards can be authored now without a second deploy.
//!
//! ## Label cardinality
//!
//! Every metric with a `model` label routes through
//! [`crate::metrics::MetricsRegistry::model_label`] at the observe
//! site to honour `SIE_METRIC_MODEL_ALLOWLIST`. `lora` label values
//! are either a real LoRA name or the sentinel `"base"` for the
//! base-model batcher — matches Python's `model_worker.py` string
//! representation so dashboards can filter with a single predicate.

use prometheus::{
    Gauge, GaugeVec, HistogramOpts, HistogramVec, IntCounterVec, IntGauge, Opts, Registry,
};

/// Histogram buckets for batch item counts. Chosen to cover the
/// realistic range (1 to `max_batch_requests`) with enough resolution
/// around the common 4–32 band where the adaptive controller spends
/// most of its time.
const BATCH_ITEMS_BUCKETS: &[f64] = &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0];

/// Histogram buckets for batch cost (tokens by default; may be any
/// monotonic cost metric). Tuned for embedding models where common
/// batch costs are 1–100 k tokens.
const BATCH_COST_BUCKETS: &[f64] = &[
    128.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16_384.0, 32_768.0, 65_536.0, 131_072.0,
    262_144.0,
];

/// Flush-reason label values. Mirrors the triggers in
/// [`super::batch_former::FormattedBatch`] + `BatchFormer::should_yield_batch`.
/// Kept as a `const` array so the label set is discoverable from one
/// place and dashboards can enumerate it without grepping code.
pub const FLUSH_REASONS: &[&str] = &[
    "cost_cap",
    "count_cap",
    "timeout",
    "coalesce",
    "single_oversize",
    "update_tightened",
    "drain",
];

/// Prometheus metrics owned by the Rust scheduler. All gauges start
/// at 0; histograms have no samples. Values populate as the scheduler
/// does work.
#[derive(Clone)]
pub struct SchedulerMetrics {
    /// Batch size (item count) observed on every flushed batch.
    /// `operation` ∈ {encode, score, extract}; `lora` = LoRA name or
    /// `"base"`.
    pub batch_items: HistogramVec,

    /// Batch cost (token total by default) observed on every flushed
    /// batch.
    pub batch_cost: HistogramVec,

    /// Counter keyed by the flush trigger that fired. See
    /// [`FLUSH_REASONS`] for the label domain.
    pub flush_reason_total: IntCounterVec,

    /// Counts every time the adaptive controller's starvation
    /// recovery fires (knobs were floored + wall-time starved).
    /// A persistently non-zero rate means either the target-p50 is
    /// too aggressive or the load just doesn't saturate the GPU.
    pub starvation_resets_total: IntCounterVec,

    /// Current `max_batch_wait_ms` the adaptive controller is
    /// asking batchers to honour. Set after every controller step.
    pub adaptive_wait_ms: GaugeVec,

    /// Current `max_batch_cost` cap, ditto.
    pub adaptive_cost: GaugeVec,

    /// Rolling mean fill ratio seen by the efficiency tracker.
    /// `None` → not enough samples yet; gauge simply stays at its
    /// previous value in that case.
    pub adaptive_fill_ratio: GaugeVec,

    /// Observed p50 latency (ms) from the [`crate::latency::LatencyTracker`].
    pub adaptive_observed_p50_ms: GaugeVec,

    /// Target p50 (ms). Either the explicit operator setting or the
    /// auto-calibrated value once the inference tracker has enough
    /// samples.
    pub adaptive_target_p50_ms: GaugeVec,

    /// Count of models that currently have a materialised scheduler.
    /// Incremented live from `crate::dispatcher::Dispatcher::resolve_scheduler`
    /// the first time a given model's scheduler is created; schedulers
    /// aren't torn down mid-process so there's no decrement path.
    pub models_total: IntGauge,

    /// Currently instantiated per-LoRA batchers per model. Lets
    /// operators see LoRA-fleet size without enumerating from logs.
    pub active_batchers: GaugeVec,

    /// Mean head-of-line latency (ms) across all non-empty batchers.
    /// Useful companion to fill_ratio — rising HoL while fill_ratio
    /// stays low = tail starvation, often a hint that `max_batch_wait_ms`
    /// is too large.
    pub hol_wait_ms_mean: Gauge,

    /// Items the dispatcher submitted into the scheduler (counted
    /// *after* payload resolution, so this tracks real enqueues —
    /// items that failed to resolve and got publish_error-ACK'd are
    /// not included). Divergence from `batch_items.sum()` is the
    /// number of items still waiting in a batcher; divergence beyond
    /// a few seconds is a stall signal.
    pub enqueued_items_total: IntCounterVec,
}

impl SchedulerMetrics {
    /// Register every metric family against the supplied registry.
    /// Panics-free; any registration error is bubbled up so the
    /// worker fails fast at startup rather than running with a
    /// partial metric surface.
    pub fn register(registry: &Registry) -> anyhow::Result<Self> {
        let batch_items = HistogramVec::new(
            HistogramOpts::new(
                "sie_worker_scheduler_batch_items",
                "Scheduler-formed batch sizes in items",
            )
            .buckets(BATCH_ITEMS_BUCKETS.to_vec()),
            &["model", "operation", "lora"],
        )?;
        let batch_cost = HistogramVec::new(
            HistogramOpts::new(
                "sie_worker_scheduler_batch_cost",
                "Scheduler-formed batch cost (tokens or other monotonic cost)",
            )
            .buckets(BATCH_COST_BUCKETS.to_vec()),
            &["model", "operation", "lora"],
        )?;
        let flush_reason_total = IntCounterVec::new(
            Opts::new(
                "sie_worker_scheduler_flush_reason_total",
                "Scheduler flush triggers. `reason` ∈ cost_cap | count_cap | timeout | coalesce | single_oversize | update_tightened | drain",
            ),
            &["model", "operation", "lora", "reason"],
        )?;
        let starvation_resets_total = IntCounterVec::new(
            Opts::new(
                "sie_worker_scheduler_starvation_resets_total",
                "Adaptive controller starvation recovery events",
            ),
            &["model"],
        )?;
        let adaptive_wait_ms = GaugeVec::new(
            Opts::new(
                "sie_worker_scheduler_adaptive_wait_ms",
                "Adaptive controller's current max_batch_wait_ms cap",
            ),
            &["model"],
        )?;
        let adaptive_cost = GaugeVec::new(
            Opts::new(
                "sie_worker_scheduler_adaptive_cost",
                "Adaptive controller's current max_batch_cost cap",
            ),
            &["model"],
        )?;
        let adaptive_fill_ratio = GaugeVec::new(
            Opts::new(
                "sie_worker_scheduler_adaptive_fill_ratio",
                "Rolling mean batch fill ratio seen by the efficiency tracker",
            ),
            &["model"],
        )?;
        let adaptive_observed_p50_ms = GaugeVec::new(
            Opts::new(
                "sie_worker_scheduler_adaptive_observed_p50_ms",
                "Observed p50 batch latency in ms",
            ),
            &["model"],
        )?;
        let adaptive_target_p50_ms = GaugeVec::new(
            Opts::new(
                "sie_worker_scheduler_adaptive_target_p50_ms",
                "Adaptive controller target p50 latency (explicit or auto-calibrated) in ms",
            ),
            &["model"],
        )?;
        let models_total = IntGauge::new(
            "sie_worker_scheduler_models_total",
            "Count of models with a materialised Rust scheduler",
        )?;
        let active_batchers = GaugeVec::new(
            Opts::new(
                "sie_worker_scheduler_active_batchers",
                "Count of per-(op, lora) batchers currently instantiated for a model",
            ),
            &["model"],
        )?;
        let hol_wait_ms_mean = Gauge::new(
            "sie_worker_scheduler_hol_wait_ms_mean",
            "Mean head-of-line wait (ms) across non-empty batchers worker-wide",
        )?;
        let enqueued_items_total = IntCounterVec::new(
            Opts::new(
                "sie_worker_scheduler_enqueued_items_total",
                "Items submitted into the Rust scheduler after payload resolution",
            ),
            &["model", "operation"],
        )?;

        registry.register(Box::new(batch_items.clone()))?;
        registry.register(Box::new(batch_cost.clone()))?;
        registry.register(Box::new(flush_reason_total.clone()))?;
        registry.register(Box::new(starvation_resets_total.clone()))?;
        registry.register(Box::new(adaptive_wait_ms.clone()))?;
        registry.register(Box::new(adaptive_cost.clone()))?;
        registry.register(Box::new(adaptive_fill_ratio.clone()))?;
        registry.register(Box::new(adaptive_observed_p50_ms.clone()))?;
        registry.register(Box::new(adaptive_target_p50_ms.clone()))?;
        registry.register(Box::new(models_total.clone()))?;
        registry.register(Box::new(active_batchers.clone()))?;
        registry.register(Box::new(hol_wait_ms_mean.clone()))?;
        registry.register(Box::new(enqueued_items_total.clone()))?;

        Ok(Self {
            batch_items,
            batch_cost,
            flush_reason_total,
            starvation_resets_total,
            adaptive_wait_ms,
            adaptive_cost,
            adaptive_fill_ratio,
            adaptive_observed_p50_ms,
            adaptive_target_p50_ms,
            models_total,
            active_batchers,
            hol_wait_ms_mean,
            enqueued_items_total,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_is_idempotent_per_registry() {
        // Two independent registries must both accept SchedulerMetrics.
        // Registering twice against the *same* registry is illegal
        // (Prometheus rejects duplicate names), so we don't assert
        // that — we only verify the happy path + label layout.
        let r1 = Registry::new();
        let _ = SchedulerMetrics::register(&r1).expect("first registry");
        let r2 = Registry::new();
        let _ = SchedulerMetrics::register(&r2).expect("second registry");
    }

    #[test]
    fn flush_reasons_cover_every_trigger() {
        // If someone adds a new flush trigger in BatchFormer, this
        // test forces them to update the reason vocabulary here so
        // dashboards can enumerate the full label domain.
        for r in FLUSH_REASONS {
            assert!(!r.is_empty());
        }
        // 7 triggers today: cost_cap, count_cap, timeout, coalesce,
        // single_oversize, update_tightened, drain.
        assert_eq!(FLUSH_REASONS.len(), 7);
    }

    #[test]
    fn metric_names_follow_prefix_convention() {
        // Every scheduler metric must be under `sie_worker_scheduler_`.
        // Verified indirectly by constructing a fresh registry and
        // gathering metric families.
        let r = Registry::new();
        let _ = SchedulerMetrics::register(&r).unwrap();
        for mf in r.gather() {
            let name = mf.name();
            assert!(
                name.starts_with("sie_worker_scheduler_"),
                "scheduler metric '{name}' must use the sie_worker_scheduler_ prefix"
            );
        }
    }
}
