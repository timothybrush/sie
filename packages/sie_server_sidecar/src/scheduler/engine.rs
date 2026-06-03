//! [`Scheduler`] — the per-model scheduler glue that ties
//! [`BatchFormer`], [`AdaptiveBatchController`],
//! [`BatchEfficiencyTracker`], and [`LatencyTracker`] together.
//!
//! Ported from `sie_server/core/worker/model_worker.py::ModelWorker`
//! but **only the scheduling half** — admission, per-LoRA batch
//! formation, FCFS fairness, and the adaptive-controller step after
//! each batch. The forward pass + adapter I/O still happen
//! Python-side; this module decides *which batch* to dispatch and
//! *when*, then hands it off.
//!
//! ## Routing
//!
//! Matches Python's asymmetric policy exactly:
//!
//! * **Encode / Extract** route by `options["lora"]` → per-LoRA batcher.
//!   `None` (or the empty-string normalisation, see [`LoraKey`]) means
//!   the base model.
//! * **Score** always goes to the base batcher regardless of
//!   `options["lora"]`. Keeping score on the base queue avoids a
//!   head-of-line-blocking class of incidents that fired the last
//!   time this was unified — comment is in
//!   `model_worker.py::_submit_score`.
//!
//! ## Fairness
//!
//! A batcher's head age is the arrival time of its oldest still-pending
//! item. Each `BatcherEntry` tracks this as a lock-free
//! [`AtomicU64`] (nanos since the scheduler's epoch; `0` means the
//! batcher is empty). `pick_oldest` scans every entry with a single
//! `Relaxed` load per entry; under the scheduler's expected O(tens) of
//! entries this is cheaper than a mutex in every hot path.
//!
//! The per-batcher head is set on the *first* submit and cleared only
//! when an extract drains the queue to empty — mid-flush, it still
//! reflects the original first-submit time. Python does the same;
//! see `batcher.py::_extract_batch` "Reset timers if we've emptied
//! the queue". This is the correct fairness primitive: "how long has
//! the oldest item in this LoRA been waiting".

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{Mutex, Notify, RwLock};

use crate::latency::LatencyTracker;

use super::adaptive::AdaptiveBatchController;
use super::batch_config::BatchConfig;
use super::batch_former::{BatchFormer, FormattedBatch, HasCost};
use super::trackers::BatchEfficiencyTracker;

/// Snapshot returned by [`Scheduler::record_completion`] for the
/// caller to feed into Prometheus / structured logs after a batch
/// completes. Mirrors the post-step block in Python's
/// `model_worker.py::_step_adaptive_controller` (~lines 752-765)
/// where `ADAPTIVE_BATCH_WAIT`, `ADAPTIVE_BATCH_COST`,
/// `ADAPTIVE_P50`, `ADAPTIVE_HEADROOM`, `ADAPTIVE_FILL_RATIO`,
/// `ADAPTIVE_STARVATION_STREAK`, and `ADAPTIVE_STARVATION_RESETS`
/// are pushed.
///
/// The dispatcher consumes this and emits the corresponding
/// `sie_worker_scheduler_adaptive_*` Prometheus families. Returning
/// the snapshot (rather than emitting metrics from inside the
/// scheduler) keeps the scheduler module free of label-vocabulary
/// concerns and keeps the Prometheus dependency boundary at the
/// dispatcher seam.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecordCompletionOutcome {
    /// New `max_batch_wait_ms` cap after the controller step.
    pub new_wait_ms: f64,
    /// New `max_batch_cost` cap after the controller step.
    pub new_batch_cost: u64,
    /// p50 of the latency tracker at step time, if enough samples
    /// have accumulated. `None` during warm-up.
    pub observed_p50_ms: Option<f64>,
    /// Controller's current target p50, either explicit or
    /// auto-calibrated. `None` until calibration completes.
    pub target_p50_ms: Option<f64>,
    /// Mean fill ratio reported by the efficiency tracker.
    pub fill_ratio: Option<f64>,
    /// Consecutive sub-`starvation_batch_size` batches as of this
    /// step. Reset to 0 when recovery fires.
    pub starvation_streak: u32,
    /// Increase in `AdaptiveBatchController::starvation_resets()`
    /// since the previous `record_completion` call. Callers
    /// `inc_by` their starvation-reset counter by this delta — the
    /// underlying counter is monotonic so a delta is the right
    /// shape for a Prometheus `IntCounter`.
    pub starvation_resets_delta: u32,
    /// Size of the batch whose completion drove this step. Echoed
    /// back so callers don't have to re-thread it through.
    pub batch_size: usize,
}

/// Operation class of a submitted item.
///
/// Kept separate from `work_types::WorkItem::op` so the scheduler
/// module has no dependency on the NATS-facing wire types. The
/// dispatcher translates at the call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    Encode,
    Score,
    Extract,
}

/// LoRA routing key. Normalises both `None` and `Some("")` to the
/// single canonical base key so the map doesn't fragment on empty
/// strings coming off the wire.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LoraKey(Option<String>);

impl LoraKey {
    /// The base model key — used explicitly by score submits and by
    /// encode/extract submits with no `options["lora"]`.
    #[must_use]
    pub fn base() -> Self {
        Self(None)
    }

    /// Build from a raw string; empty strings normalise to `base()`.
    ///
    /// Named `from_name` rather than `from_str` so it doesn't collide
    /// with [`std::str::FromStr`]'s convention (which expects a
    /// `Result` return + an associated error type). This cannot fail,
    /// and adding a bogus infallible error type just to satisfy the
    /// trait would be churn for no caller benefit.
    #[must_use]
    pub fn from_name(s: &str) -> Self {
        if s.is_empty() {
            Self::base()
        } else {
            Self(Some(s.to_owned()))
        }
    }

    /// Build from `Option<String>`, normalising `Some("")` to `None`.
    #[must_use]
    pub fn from_option(o: Option<String>) -> Self {
        match o {
            Some(s) if s.is_empty() => Self::base(),
            other => Self(other),
        }
    }

    /// `true` when this key refers to the base model (no LoRA).
    #[must_use]
    pub fn is_base(&self) -> bool {
        self.0.is_none()
    }

    /// Borrow the LoRA name, or `None` if this is the base key.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        self.0.as_deref()
    }
}

/// Map key for the scheduler's batcher registry. Named alias keeps
/// signatures readable and silences `clippy::type_complexity` on the
/// `RwLock<HashMap<...>>` field of [`Scheduler`].
type BatcherKey = (Op, LoraKey);

/// Map of live batchers keyed by `(op, lora)`. See
/// [`BatcherEntry`] for what each value holds.
type BatcherMap<I, T> = HashMap<BatcherKey, Arc<BatcherEntry<I, T>>>;

/// One entry in the scheduler's (op, lora) → batcher map. Holds the
/// [`BatchFormer`] plus a lock-free head-age atomic the FCFS scan
/// reads.
#[derive(Debug)]
struct BatcherEntry<I: HasCost, T> {
    op: Op,
    lora: LoraKey,
    former: Arc<BatchFormer<I, T>>,
    /// Nanos since [`Scheduler::epoch`]. `0` ⇒ batcher has no pending
    /// items. Any nonzero value is the timestamp at which the first
    /// *currently-pending* item was submitted — i.e. the arrival time
    /// of this batcher's oldest waiter.
    head_ns: AtomicU64,
}

/// The per-model scheduler.
///
/// Generic over the item type `I` (must implement [`HasCost`]) and
/// the caller-supplied per-item metadata `T`. Production wires this as
/// `Scheduler<SchedulerItem, WorkMeta>` at the dispatcher boundary;
/// tests use simple stubs.
pub struct Scheduler<I: HasCost, T> {
    batchers: RwLock<BatcherMap<I, T>>,
    controller: Mutex<AdaptiveBatchController>,
    latency: Mutex<LatencyTracker>,
    efficiency: Mutex<BatchEfficiencyTracker>,
    /// Current caps used for *newly created* batchers. Existing
    /// batchers receive updates via [`Scheduler::record_completion`]
    /// after each controller step. Kept behind an async `RwLock`
    /// because readers are rare (new-batcher creation only).
    config: RwLock<BatchConfig>,
    /// Fired on every submit that inserts into a previously empty
    /// batcher. [`Scheduler::consume_next`] uses this to wake its
    /// FCFS scan without polling.
    new_item: Notify,
    /// Wall-clock origin for per-batcher `head_ns` atomics. Stored as
    /// `Instant` (monotonic) so head-age comparisons are unaffected by
    /// wall-clock jumps. `Arc`'d only because the struct is sometimes
    /// cloned in tests; the field itself is `Copy`.
    epoch: Instant,
}

impl<I, T> Scheduler<I, T>
where
    I: HasCost + Send + Sync + 'static,
    T: Send + Sync + 'static,
{
    /// Construct a scheduler with the given initial caps and a
    /// default [`AdaptiveBatchController`] (auto-calibrating target,
    /// Python-parity defaults elsewhere).
    ///
    /// Most callers want [`Scheduler::builder`] to customise the
    /// controller / tracker windows.
    #[must_use]
    pub fn new(config: BatchConfig) -> Self {
        Self::builder().config(config).build()
    }

    /// Fluent builder. See [`SchedulerBuilder`].
    #[must_use]
    pub fn builder() -> SchedulerBuilder<I, T> {
        SchedulerBuilder::default()
    }

    // ---- Introspection helpers (used by `consume_next` and tests) ----

    /// Current snapshot of the batch caps. Held by value — callers
    /// always get a consistent view even if another task is in the
    /// middle of [`Scheduler::record_completion`].
    pub async fn config(&self) -> BatchConfig {
        *self.config.read().await
    }

    /// Sum of pending items across every (op, lora) batcher.
    pub async fn total_pending_count(&self) -> usize {
        let map = self.batchers.read().await;
        let mut total = 0usize;
        for e in map.values() {
            total += e.former.pending_count().await;
        }
        total
    }

    /// Sum of pending cost across every (op, lora) batcher.
    pub async fn total_pending_cost(&self) -> u64 {
        let map = self.batchers.read().await;
        let mut total = 0u64;
        for e in map.values() {
            total += e.former.pending_cost().await;
        }
        total
    }

    /// Snapshot of the adaptive controller's observable state. Useful
    /// for dashboards / readiness probes.
    pub async fn controller_snapshot(&self) -> super::adaptive::AdaptiveBatchState {
        let ctrl = self.controller.lock().await;
        let observed = self.latency.lock().await.p50();
        let fill = self.efficiency.lock().await.mean_fill_ratio();
        ctrl.snapshot(observed, fill)
    }

    // ---- Submit path ----

    /// Enqueue a single item. Routing rules: encode/extract go to
    /// `(op, lora)`; score always forces `lora = base` regardless of
    /// what the caller passed.
    pub async fn submit(&self, op: Op, lora: LoraKey, item: I, metadata: T) {
        let key = Self::route_key(op, lora);
        let entry = self.get_or_create(&key).await;
        self.mark_head_on_first_submit(&entry);
        entry.former.submit(item, metadata).await;
    }

    /// Bulk submit under a single BatchFormer-lock. Use this for
    /// multi-item requests (SDK batches of ≤ 64 are common).
    pub async fn submit_many(&self, op: Op, lora: LoraKey, items: Vec<(I, T)>) {
        if items.is_empty() {
            return;
        }
        let key = Self::route_key(op, lora);
        let entry = self.get_or_create(&key).await;
        self.mark_head_on_first_submit(&entry);
        entry.former.submit_many(items).await;
    }

    // ---- Consume path ----

    /// Block until some batcher has pending items, FCFS-pick the one
    /// whose oldest-head is oldest, and return its flushed batch.
    ///
    /// `immediate = true` forwards directly to
    /// [`BatchFormer::get_batch`] so a single pending item flushes at
    /// once without the `max_batch_wait_ms` wait — used when the
    /// caller was idle and knows no further arrivals are likely.
    pub async fn consume_next(&self, immediate: bool) -> (Op, LoraKey, FormattedBatch<I, T>) {
        loop {
            // Arm the notification *before* scanning: a submit that
            // fires between scan and await is captured in the
            // Notified's permit slot (see `Notified::enable` docs).
            let notified = self.new_item.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();

            if let Some(entry) = self.pick_oldest().await {
                let batch = entry.former.get_batch(immediate).await;
                self.maybe_clear_head(&entry).await;
                return (entry.op, entry.lora.clone(), batch);
            }
            notified.await;
        }
    }

    /// Non-blocking variant of [`Self::consume_next`]. Returns `None`
    /// if no batcher has pending items *or* if the FCFS-picked batcher
    /// isn't yet ready to flush (flush triggers haven't fired).
    pub async fn try_consume_next(&self) -> Option<(Op, LoraKey, FormattedBatch<I, T>)> {
        let entry = self.pick_oldest().await?;
        let batch = entry.former.try_get_batch().await?;
        self.maybe_clear_head(&entry).await;
        Some((entry.op, entry.lora.clone(), batch))
    }

    /// Continuous-batching tail: drain whatever's left in `(op, lora)`'s
    /// batcher immediately, bypassing flush triggers. Returns `None`
    /// when the batcher has no pending items (or doesn't exist).
    ///
    /// Matches the Python adapter process's `try_drain` call after each forward
    /// pass. Expected usage: after consume_next returns a batch for
    /// `(op, lora)`, loop on `try_drain_same(op, lora)` until it
    /// returns `None`.
    pub async fn try_drain_same(&self, op: Op, lora: LoraKey) -> Option<FormattedBatch<I, T>> {
        let key = Self::route_key(op, lora);
        let entry = self.batchers.read().await.get(&key).cloned()?;
        let batch = entry.former.try_drain().await?;
        self.maybe_clear_head(&entry).await;
        Some(batch)
    }

    // ---- Controller step + efficiency / latency plumbing ----

    /// Feed the auto-calibration tracker with an inference-only
    /// sample (GPU forward pass, no queue / no postprocess).
    ///
    /// Python's ModelWorker calls this with `RequestTiming.inference_ms`;
    /// once the controller is calibrated this is a cheap no-op.
    pub async fn record_inference_sample(&self, inference_ms: f64) {
        let mut ctrl = self.controller.lock().await;
        ctrl.record_inference_sample(inference_ms);
    }

    /// Record one item's total latency (queue + inference +
    /// postprocess) — call **once per item** in a completed batch.
    ///
    /// Mirrors Python's `_check_partial_results` block at
    /// `model_worker.py:1004` which calls
    /// `_latency_tracker.record(metadata.timing.total_ms)` for every
    /// completed item. Per-item recording is mathematically critical
    /// for the controller: feeding per-batch MAX values would bias
    /// `observed_p50_ms` toward the per-item p99 of the underlying
    /// distribution and drive `max_batch_wait_ms` to its floor under
    /// concurrent load.
    pub async fn record_latency_sample(&self, total_ms: f64) {
        self.latency.lock().await.record(total_ms);
    }

    /// Bulk variant of [`Self::record_latency_sample`] — pushes
    /// every value in `samples` under a single mutex acquire. Use
    /// when the caller already has the per-item totals collected
    /// (e.g. the dispatcher's per-item zip over `outcomes` x
    /// `metadata`); avoids N lock acquires for typical batches of
    /// 16–32 items.
    pub async fn record_latency_samples(&self, samples: &[f64]) {
        if samples.is_empty() {
            return;
        }
        let mut t = self.latency.lock().await;
        for s in samples {
            t.record(*s);
        }
    }

    /// Record a batch's completion metrics (efficiency + controller
    /// step) and propagate the new caps to every live batcher.
    ///
    /// **Latency samples must be fed beforehand** via
    /// [`Self::record_latency_sample`] / [`Self::record_latency_samples`].
    /// This call only consumes the *current* tracker p50 — it does
    /// not record any latency itself. Splitting the surface that
    /// way keeps the per-item-vs-per-batch sample-rate question
    /// purely a caller concern.
    ///
    /// `batch_cost` = total cost of the batch that just completed
    /// (used by the efficiency tracker → fill_ratio knob). `batch_size`
    /// = item count (used by the starvation detector).
    ///
    /// Returns a [`RecordCompletionOutcome`] for the caller to push
    /// to Prometheus. Mirrors Python's `model_worker.py:752-765`.
    pub async fn record_completion(
        &self,
        batch_cost: u64,
        batch_size: usize,
    ) -> RecordCompletionOutcome {
        let current_cap = self.config.read().await.max_batch_cost;
        self.efficiency.lock().await.record(batch_cost, current_cap);

        // Snapshot the trackers *before* the controller step so the
        // returned outcome reflects the same observation the
        // controller acted on (Python's `_step_adaptive_controller`
        // does the same — it reads `observed_p50` once and reuses
        // the value for both the step and the metric push).
        let observed_p50 = self.latency.lock().await.p50();
        let fill = self.efficiency.lock().await.mean_fill_ratio();

        let (new_wait, new_cost, target_p50, starvation_streak, starvation_resets_delta) = {
            let mut ctrl = self.controller.lock().await;
            let prev_resets = ctrl.starvation_resets();
            let (w, c) = ctrl.step(observed_p50, fill, Some(batch_size));
            let new_resets = ctrl.starvation_resets();
            (
                w,
                c,
                ctrl.target_p50_ms(),
                ctrl.starvation_streak(),
                new_resets.saturating_sub(prev_resets),
            )
        };

        // Update the shared snapshot. Short critical section under
        // the write lock — we only mutate two fields.
        let new_cfg = {
            let mut cfg = self.config.write().await;
            cfg.max_batch_wait_ms = new_wait;
            cfg.max_batch_cost = new_cost;
            *cfg
        };

        // Propagate to every live batcher. We snapshot the Vec of
        // Arcs under the read lock and drop it before the async
        // `update_config` calls so a concurrent submit that creates a
        // new batcher doesn't deadlock. A batcher created *after*
        // this snapshot sees `new_cfg` via `config.read()` in
        // `get_or_create`, so we don't lose the update.
        let entries: Vec<Arc<BatcherEntry<I, T>>> = {
            let map = self.batchers.read().await;
            map.values().cloned().collect()
        };
        for entry in entries {
            entry.former.update_config(new_cfg).await;
        }

        RecordCompletionOutcome {
            new_wait_ms: new_wait,
            new_batch_cost: new_cost,
            observed_p50_ms: observed_p50,
            target_p50_ms: target_p50,
            fill_ratio: fill,
            starvation_streak,
            starvation_resets_delta,
            batch_size,
        }
    }

    // ---- Internal helpers ----

    fn route_key(op: Op, lora: LoraKey) -> BatcherKey {
        match op {
            // Score always lives on the base queue, regardless of
            // what the caller asked for. Matches Python's
            // `_submit_score` behaviour; see the module-level
            // doc-comment for why this is load-bearing.
            Op::Score => (Op::Score, LoraKey::base()),
            _ => (op, lora),
        }
    }

    async fn get_or_create(&self, key: &BatcherKey) -> Arc<BatcherEntry<I, T>> {
        // Fast path: read lock, return existing.
        if let Some(e) = self.batchers.read().await.get(key) {
            return Arc::clone(e);
        }
        // Slow path: take the write lock and double-check — another
        // task may have inserted concurrently.
        let mut map = self.batchers.write().await;
        if let Some(e) = map.get(key) {
            return Arc::clone(e);
        }
        let cfg = *self.config.read().await;
        let entry = Arc::new(BatcherEntry {
            op: key.0,
            lora: key.1.clone(),
            former: Arc::new(BatchFormer::new(cfg)),
            head_ns: AtomicU64::new(0),
        });
        map.insert(key.clone(), Arc::clone(&entry));
        entry
    }

    /// Set `head_ns` to "now" if the entry was empty. If it was
    /// already non-zero (i.e. this batcher already has pending items
    /// from an earlier submit), leave the existing value alone —
    /// Python keeps the original first-submit time across mid-flush
    /// churn for fairness accounting.
    fn mark_head_on_first_submit(&self, entry: &BatcherEntry<I, T>) {
        let now_ns = self.epoch.elapsed().as_nanos() as u64;
        // Ensure we never store 0 on a real submit — 0 is reserved
        // for "empty". Instants that round to 0 ns are
        // astronomically unlikely (would require submit in the
        // first tick of the runtime's clock), but saturating_add
        // here keeps the invariant airtight.
        let ts = now_ns.saturating_add(1);
        let _ = entry
            .head_ns
            .compare_exchange(0, ts, Ordering::AcqRel, Ordering::Relaxed);
        // If the CAS fails, head_ns was already non-zero — keep
        // the older timestamp. Also fire the ready-notify so the
        // consumer's FCFS loop picks this up even on the first
        // submit into a brand-new LoRA.
        self.new_item.notify_one();
    }

    /// Zero `head_ns` iff the batcher has no pending items after an
    /// extract. Leaves non-zero values alone otherwise.
    async fn maybe_clear_head(&self, entry: &BatcherEntry<I, T>) {
        if entry.former.pending_count().await == 0 {
            entry.head_ns.store(0, Ordering::Release);
        }
    }

    async fn pick_oldest(&self) -> Option<Arc<BatcherEntry<I, T>>> {
        let map = self.batchers.read().await;
        let mut best: Option<(u64, Arc<BatcherEntry<I, T>>)> = None;
        for entry in map.values() {
            let head = entry.head_ns.load(Ordering::Acquire);
            if head == 0 {
                continue;
            }
            match &best {
                None => best = Some((head, Arc::clone(entry))),
                Some((ts, _)) if head < *ts => {
                    best = Some((head, Arc::clone(entry)));
                }
                _ => {}
            }
        }
        best.map(|(_, e)| e)
    }
}

// `Scheduler` intentionally doesn't derive `Clone`; the expected
// pattern is `Arc<Scheduler<...>>`. Making it clonable would hide
// the aliasing and let a test accidentally fork the adaptive
// controller state.
impl<I: HasCost, T> std::fmt::Debug for Scheduler<I, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scheduler")
            .field("epoch", &self.epoch)
            .finish_non_exhaustive()
    }
}

// ---- Builder ----

/// Builder for [`Scheduler`]. Separated from the plain `new` path so
/// operators can tune the controller / tracker windows per model
/// without the `new` signature growing indefinitely.
pub struct SchedulerBuilder<I: HasCost, T> {
    config: BatchConfig,
    controller: Option<AdaptiveBatchController>,
    latency_window: usize,
    latency_min_samples: usize,
    efficiency_window: usize,
    _phantom: std::marker::PhantomData<(I, T)>,
}

impl<I: HasCost, T> Default for SchedulerBuilder<I, T> {
    fn default() -> Self {
        Self {
            config: BatchConfig::default(),
            controller: None,
            // Python's `LatencyTracker(window_size=200, min_samples=10)`.
            latency_window: 200,
            latency_min_samples: 10,
            efficiency_window: 50,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<I, T> SchedulerBuilder<I, T>
where
    I: HasCost + Send + Sync + 'static,
    T: Send + Sync + 'static,
{
    /// Override the initial batch caps.
    #[must_use]
    pub fn config(mut self, config: BatchConfig) -> Self {
        self.config = config;
        self
    }

    /// Supply a pre-configured adaptive controller. If omitted the
    /// scheduler uses [`AdaptiveBatchController::default`].
    #[must_use]
    pub fn controller(mut self, ctrl: AdaptiveBatchController) -> Self {
        self.controller = Some(ctrl);
        self
    }

    /// Override the latency tracker window. Default 200 samples,
    /// 10-sample warm-up.
    #[must_use]
    pub fn latency_window(mut self, window: usize, min_samples: usize) -> Self {
        self.latency_window = window;
        self.latency_min_samples = min_samples;
        self
    }

    /// Override the efficiency tracker window. Default 50.
    #[must_use]
    pub fn efficiency_window(mut self, window: usize) -> Self {
        self.efficiency_window = window;
        self
    }

    #[must_use]
    pub fn build(self) -> Scheduler<I, T> {
        // Default controller mirrors Python's
        // `ModelWorker.__init__` *production* wiring against the
        // current `BatchConfig` (model `max_batch_tokens` =
        // `cfg.max_batch_cost`, initial wait = `cfg.max_batch_wait_ms`,
        // cost floor / ceiling = `max_batch_tokens // 4` ..
        // `max_batch_tokens * 4`, `cost_gain = gain * 0.5`,
        // `min_wait_ms = 15.0`). The bare `default()` / module-level
        // dataclass values are test-only floors — `SchedulerBuilder`
        // is the production entry point and matches what the Python
        // worker has been deploying.
        //
        // `SIE_ADAPTIVE_BATCH_*` env vars override on top, so
        // operators can pin individual knobs without a recompile.
        // Tests using `.controller(...)` continue to bypass both env
        // and BatchConfig derivation entirely (deterministic).
        let cfg = self.config;
        Scheduler {
            batchers: RwLock::new(HashMap::new()),
            controller: Mutex::new(
                self.controller
                    .unwrap_or_else(|| AdaptiveBatchController::from_batch_config_and_env(&cfg)),
            ),
            latency: Mutex::new(LatencyTracker::new(
                self.latency_window,
                self.latency_min_samples,
            )),
            efficiency: Mutex::new(BatchEfficiencyTracker::new(self.efficiency_window)),
            config: RwLock::new(self.config),
            new_item: Notify::new(),
            epoch: Instant::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep as tsleep, timeout, Duration as TDuration};

    // ---- Stub item type ----

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct StubItem {
        cost: u64,
        idx: usize,
    }
    impl HasCost for StubItem {
        fn cost(&self) -> u64 {
            self.cost
        }
        fn original_index(&self) -> usize {
            self.idx
        }
    }

    fn make() -> Scheduler<StubItem, u32> {
        Scheduler::builder()
            .config(BatchConfig {
                max_batch_cost: 1_000_000,
                max_batch_requests: 1000,
                max_batch_wait_ms: 10_000.0,
                coalesce_ms: 10_000.0,
                coalesce_ratio: 1.0,
            })
            .build()
    }

    // ---- Routing ----

    #[tokio::test]
    async fn encode_routes_per_lora() {
        let s = make();
        s.submit(
            Op::Encode,
            LoraKey::from_name("a"),
            StubItem { cost: 1, idx: 0 },
            0,
        )
        .await;
        s.submit(
            Op::Encode,
            LoraKey::from_name("b"),
            StubItem { cost: 2, idx: 0 },
            1,
        )
        .await;
        assert_eq!(s.total_pending_count().await, 2);
        let map = s.batchers.read().await;
        assert!(map.contains_key(&(Op::Encode, LoraKey::from_name("a"))));
        assert!(map.contains_key(&(Op::Encode, LoraKey::from_name("b"))));
    }

    #[tokio::test]
    async fn extract_routes_per_lora() {
        let s = make();
        s.submit(
            Op::Extract,
            LoraKey::from_name("a"),
            StubItem { cost: 1, idx: 0 },
            0,
        )
        .await;
        let map = s.batchers.read().await;
        assert!(map.contains_key(&(Op::Extract, LoraKey::from_name("a"))));
    }

    #[tokio::test]
    async fn score_always_routes_to_base_even_with_lora_arg() {
        let s = make();
        s.submit(
            Op::Score,
            LoraKey::from_name("should-be-ignored"),
            StubItem { cost: 1, idx: 0 },
            0,
        )
        .await;
        let map = s.batchers.read().await;
        // Must not have created a LoRA-keyed score batcher.
        assert!(
            !map.contains_key(&(Op::Score, LoraKey::from_name("should-be-ignored"))),
            "score submit with lora arg must not fragment the score queue"
        );
        assert!(map.contains_key(&(Op::Score, LoraKey::base())));
    }

    #[tokio::test]
    async fn empty_string_lora_normalises_to_base() {
        // Empty string and `base()` are indistinguishable as keys —
        // which is the whole point of the normalisation. We verify
        // that both submits hit the *same* batcher (map size stays 1)
        // and that the surviving key is `base()`.
        let s = make();
        s.submit(
            Op::Encode,
            LoraKey::from_name(""),
            StubItem { cost: 1, idx: 0 },
            0,
        )
        .await;
        s.submit(Op::Encode, LoraKey::base(), StubItem { cost: 1, idx: 1 }, 1)
            .await;
        let map = s.batchers.read().await;
        assert_eq!(map.len(), 1, "empty-string lora must not fragment the map");
        assert!(map.contains_key(&(Op::Encode, LoraKey::base())));
        // `LoraKey::from_name("") == LoraKey::base()` by construction —
        // pin that invariant here so a future refactor can't silently
        // break the fragment-prevention.
        assert_eq!(LoraKey::from_name(""), LoraKey::base());
    }

    #[tokio::test]
    async fn submit_many_funnels_into_one_batcher() {
        let s = make();
        let items: Vec<(StubItem, u32)> = (0..5)
            .map(|i| (StubItem { cost: 1, idx: i }, i as u32))
            .collect();
        s.submit_many(Op::Encode, LoraKey::base(), items).await;
        assert_eq!(s.total_pending_count().await, 5);
    }

    // ---- FCFS fairness ----

    #[tokio::test]
    async fn fcfs_picks_oldest_head_across_loras() {
        let s = make();
        // LoRA A arrives first.
        s.submit(
            Op::Encode,
            LoraKey::from_name("a"),
            StubItem { cost: 1, idx: 0 },
            1,
        )
        .await;
        // Tiny real delay so the head_ns values are distinguishable.
        tsleep(TDuration::from_millis(2)).await;
        s.submit(
            Op::Encode,
            LoraKey::from_name("b"),
            StubItem { cost: 1, idx: 0 },
            2,
        )
        .await;

        let (op, lora, batch) = s.consume_next(true).await;
        assert_eq!(op, Op::Encode);
        assert_eq!(lora, LoraKey::from_name("a"));
        assert_eq!(batch.size(), 1);
        assert_eq!(batch.metadata[0], 1);
    }

    #[tokio::test]
    async fn fcfs_ignores_empty_batchers() {
        let s = make();
        // Populate LoRA A first, drain it.
        s.submit(
            Op::Encode,
            LoraKey::from_name("a"),
            StubItem { cost: 1, idx: 0 },
            1,
        )
        .await;
        let (_, _, _) = s.consume_next(true).await;
        assert_eq!(s.total_pending_count().await, 0);

        // Now submit to LoRA B; FCFS must pick B (A is empty).
        s.submit(
            Op::Encode,
            LoraKey::from_name("b"),
            StubItem { cost: 1, idx: 0 },
            2,
        )
        .await;
        let (_, lora, _) = s.consume_next(true).await;
        assert_eq!(lora, LoraKey::from_name("b"));
    }

    #[tokio::test]
    async fn head_ns_clears_only_when_batcher_drains_to_empty() {
        let s = Scheduler::<StubItem, u32>::builder()
            .config(BatchConfig {
                max_batch_cost: 5, // small cap so extract leaves remainder
                max_batch_requests: 1000,
                max_batch_wait_ms: 10_000.0,
                coalesce_ms: 10_000.0,
                coalesce_ratio: 1.0,
            })
            .build();
        // Two items, cost 3 each. First extract takes one (3 ≤ 5),
        // stops because 3+3 > 5.
        s.submit(Op::Encode, LoraKey::base(), StubItem { cost: 3, idx: 0 }, 1)
            .await;
        s.submit(Op::Encode, LoraKey::base(), StubItem { cost: 3, idx: 0 }, 2)
            .await;

        let (_, _, batch) = s.consume_next(false).await;
        assert_eq!(batch.size(), 1);
        assert_eq!(s.total_pending_count().await, 1, "one item must remain");

        // The entry's head_ns must still be non-zero so FCFS keeps
        // honouring it on the next scan.
        let entry = {
            let map = s.batchers.read().await;
            Arc::clone(map.get(&(Op::Encode, LoraKey::base())).unwrap())
        };
        let head = entry.head_ns.load(Ordering::Acquire);
        assert_ne!(
            head, 0,
            "head_ns must stay non-zero while pending items remain"
        );

        // Drain the rest; head should clear.
        let (_, _, _) = s.consume_next(true).await;
        assert_eq!(s.total_pending_count().await, 0);
        let head = entry.head_ns.load(Ordering::Acquire);
        assert_eq!(head, 0, "head_ns must clear when queue empties");
    }

    #[tokio::test]
    async fn consume_next_blocks_when_empty_and_wakes_on_submit() {
        let s = Arc::new(make());
        let consumer = {
            let s = Arc::clone(&s);
            tokio::spawn(async move { s.consume_next(true).await })
        };

        // Let the consumer register.
        tsleep(TDuration::from_millis(5)).await;

        s.submit(
            Op::Encode,
            LoraKey::base(),
            StubItem { cost: 1, idx: 0 },
            42,
        )
        .await;

        let (op, lora, batch) = timeout(TDuration::from_secs(2), consumer)
            .await
            .expect("consumer must wake within 2 s")
            .unwrap();
        assert_eq!(op, Op::Encode);
        assert!(lora.is_base());
        assert_eq!(batch.metadata[0], 42);
    }

    #[tokio::test]
    async fn try_consume_next_returns_none_when_empty() {
        let s = make();
        assert!(s.try_consume_next().await.is_none());
    }

    #[tokio::test]
    async fn try_drain_same_returns_continuous_batching_tail() {
        let s = make();
        // Two separate submits so we can drain one now and the other
        // via try_drain_same.
        s.submit(Op::Encode, LoraKey::base(), StubItem { cost: 1, idx: 0 }, 1)
            .await;
        let (_, _, _) = s.consume_next(true).await;

        s.submit(Op::Encode, LoraKey::base(), StubItem { cost: 2, idx: 0 }, 2)
            .await;
        let drain = s
            .try_drain_same(Op::Encode, LoraKey::base())
            .await
            .expect("must drain pending tail");
        assert_eq!(drain.size(), 1);
        assert_eq!(drain.metadata[0], 2);
    }

    // ---- Adaptive controller wiring ----

    #[tokio::test]
    async fn record_completion_propagates_config_to_all_batchers() {
        let ctrl = AdaptiveBatchController::builder()
            .target_p50_ms(Some(50.0))
            .update_interval(1) // step on every record_completion
            .min_wait_ms(1.0)
            .max_wait_ms(100.0)
            .min_batch_cost(100)
            .max_batch_cost(100_000)
            .initial_wait_ms(20.0)
            .initial_batch_cost(5_000)
            .build();
        let s = Scheduler::<StubItem, u32>::builder()
            .config(BatchConfig {
                max_batch_cost: 5_000,
                max_batch_requests: 1000,
                max_batch_wait_ms: 20.0,
                coalesce_ms: 20.0,
                coalesce_ratio: 0.5,
            })
            .controller(ctrl)
            // Short warm-up so a handful of completions is enough to
            // drive p50 — the default is 10 samples which would force
            // a noisier test.
            .latency_window(20, 3)
            .build();

        // Submit to two distinct LoRAs so we can observe propagation
        // to both.
        s.submit(
            Op::Encode,
            LoraKey::from_name("a"),
            StubItem { cost: 1, idx: 0 },
            0,
        )
        .await;
        s.submit(
            Op::Encode,
            LoraKey::from_name("b"),
            StubItem { cost: 1, idx: 0 },
            0,
        )
        .await;

        // Simulate several completions with observed < target so the
        // latency tracker crosses its warm-up threshold and the PI
        // loop bumps `wait_ms` upward.
        for _ in 0..5 {
            s.record_latency_sample(10.0).await;
            let _ = s.record_completion(32, 32).await;
        }

        let cfg_a = {
            let map = s.batchers.read().await;
            map[&(Op::Encode, LoraKey::from_name("a"))]
                .former
                .config()
                .await
        };
        let cfg_b = {
            let map = s.batchers.read().await;
            map[&(Op::Encode, LoraKey::from_name("b"))]
                .former
                .config()
                .await
        };
        assert!(
            cfg_a.max_batch_wait_ms > 20.0,
            "wait must have grown from 20 (positive headroom); got {}",
            cfg_a.max_batch_wait_ms
        );
        assert_eq!(
            cfg_a, cfg_b,
            "all batchers must see the same caps after record_completion"
        );
        // Scheduler's own snapshot tracks the same value.
        assert_eq!(s.config().await, cfg_a);
    }

    #[tokio::test]
    async fn record_completion_feeds_efficiency_tracker() {
        // Efficiency tracker must see a fill ratio; confirm by
        // running enough completions to tighten cost knob.
        let ctrl = AdaptiveBatchController::builder()
            .target_p50_ms(Some(50.0))
            .update_interval(1)
            .fill_ratio_threshold(0.1) // always saturated in test terms
            .cost_gain(1.0)
            .initial_batch_cost(1_000)
            .min_batch_cost(100)
            .max_batch_cost(10_000)
            .build();
        let s = Scheduler::<StubItem, u32>::builder()
            .config(BatchConfig {
                max_batch_cost: 1_000,
                max_batch_requests: 1000,
                max_batch_wait_ms: 15.0,
                coalesce_ms: 15.0,
                coalesce_ratio: 0.5,
            })
            .controller(ctrl)
            .build();
        s.submit(Op::Encode, LoraKey::base(), StubItem { cost: 1, idx: 0 }, 0)
            .await;

        // Many completions with observed < target and a high fill
        // ratio → cost must grow.
        let before = s.config().await.max_batch_cost;
        for _ in 0..20 {
            s.record_latency_sample(10.0).await;
            let _ = s.record_completion(800, 32).await;
        }
        let after = s.config().await.max_batch_cost;
        assert!(after > before, "cost cap must grow: {before} → {after}");
    }

    #[tokio::test]
    async fn controller_snapshot_reflects_latency_and_fill() {
        let s = Scheduler::<StubItem, u32>::builder()
            .config(BatchConfig::default())
            .controller(
                AdaptiveBatchController::builder()
                    .target_p50_ms(Some(50.0))
                    .update_interval(1)
                    .build(),
            )
            .build();
        // Seed enough samples to cross the latency tracker's min.
        for _ in 0..12 {
            s.record_latency_sample(10.0).await;
            let _ = s.record_completion(32, 32).await;
        }
        let snap = s.controller_snapshot().await;
        assert_eq!(snap.target_p50_ms, Some(50.0));
        assert_eq!(snap.observed_p50_ms, Some(10.0));
        assert_eq!(snap.headroom_ms, Some(40.0));
        assert!(snap.fill_ratio.is_some());
    }

    #[tokio::test]
    async fn drain_feeds_do_not_step_controller() {
        // Regression for #4: Python's `_process_loop` steps the
        // adaptive controller exactly once per wave, with the
        // primary batch's size (`model_worker.py:828, 855-870`).
        // Drains feed inference + latency samples but must not
        // advance the controller's internal state.
        //
        // Encoded here at the scheduler level: the dispatcher
        // routes Primary → `record_completion`, Drain →
        // `record_inference_sample` + `record_latency_samples`
        // (no `record_completion`). This test exercises the
        // scheduler-side surface contract; the
        // [`crate::dispatcher::WaveRole`] enum + the
        // `scheduler_drain_loop` plumbing make sure drains take
        // the second path.
        let ctrl = AdaptiveBatchController::builder()
            .target_p50_ms(Some(50.0))
            .update_interval(1) // step on every record_completion
            .starvation_batch_size(1) // size>1 resets streak; we use
            // batch_size=1 so each `step()` strictly increments.
            .build();
        let s = Scheduler::<StubItem, u32>::builder()
            .config(BatchConfig::default())
            .controller(ctrl)
            .latency_window(20, 1)
            .build();

        // Wave 1 — primary at batch_size=1 increments the streak.
        s.record_latency_sample(10.0).await;
        let _ = s.record_completion(1, 1).await;
        assert_eq!(
            s.controller_snapshot().await.starvation_streak,
            1,
            "primary must advance streak"
        );

        // Three drains — feed inference + latency only. Streak
        // must NOT advance.
        for _ in 0..3 {
            s.record_inference_sample(8.0).await;
            s.record_latency_samples(&[10.0]).await;
        }
        assert_eq!(
            s.controller_snapshot().await.starvation_streak,
            1,
            "drain feeds must not step the controller (streak should stay at 1, got {})",
            s.controller_snapshot().await.starvation_streak,
        );

        // Wave 2 — next primary advances exactly once again.
        s.record_latency_sample(10.0).await;
        let _ = s.record_completion(1, 1).await;
        assert_eq!(
            s.controller_snapshot().await.starvation_streak,
            2,
            "second primary must advance streak by exactly one wave",
        );
    }

    #[tokio::test]
    async fn inference_sample_feeds_autocalibration() {
        let ctrl = AdaptiveBatchController::builder()
            .target_p50_ms(None) // auto-calibrate
            .calibration_multiplier(2.0)
            .update_interval(1)
            .build();
        let s = Scheduler::<StubItem, u32>::builder()
            .controller(ctrl)
            .build();

        // Feed 20 inference samples at 10 ms so inference_p50 = 10.
        for _ in 0..20 {
            s.record_inference_sample(10.0).await;
        }
        // One completion triggers the calibration step.
        let _ = s.record_completion(0, 1).await;
        let snap = s.controller_snapshot().await;
        assert!(snap.calibrated);
        assert_eq!(snap.target_p50_ms, Some(20.0));
    }

    #[tokio::test]
    async fn record_completion_returns_populated_outcome_shape() {
        // Verify the new outcome plumbing: every field is populated
        // with the post-step controller state. The starvation
        // recovery itself is already exhaustively tested in
        // `adaptive.rs::tests::starvation_recovery_*`; here we just
        // confirm the snapshot wiring carries the values through.
        let ctrl = AdaptiveBatchController::builder()
            .target_p50_ms(Some(50.0))
            .update_interval(1)
            .initial_wait_ms(20.0)
            .initial_batch_cost(4_000)
            .build();
        let s = Scheduler::<StubItem, u32>::builder()
            .config(BatchConfig::default())
            .controller(ctrl)
            .latency_window(20, 1)
            .build();

        s.record_latency_sample(10.0).await;
        let snap = s.record_completion(800, 32).await;

        assert!(snap.new_wait_ms > 0.0);
        assert!(snap.new_batch_cost > 0);
        assert_eq!(snap.batch_size, 32);
        assert_eq!(snap.target_p50_ms, Some(50.0));
        assert_eq!(snap.observed_p50_ms, Some(10.0));
        assert!(snap.fill_ratio.is_some());
        // No starvation triggered yet → delta is zero.
        assert_eq!(snap.starvation_resets_delta, 0);

        // After the step the scheduler's shared config snapshot
        // carries the same caps as the returned outcome — operators
        // reading `s.config()` after a completion see the same
        // numbers Prometheus does.
        let cfg = s.config().await;
        assert_eq!(cfg.max_batch_wait_ms, snap.new_wait_ms);
        assert_eq!(cfg.max_batch_cost, snap.new_batch_cost);
    }

    // ---- BatchConfig-derived production-parity defaults ----

    #[tokio::test]
    async fn builder_default_controller_uses_production_parity_floors() {
        // No `.controller(...)` supplied → builder must derive the
        // controller from the BatchConfig with Python production
        // wiring (min_wait_ms=15, min_batch_cost=cfg.max_batch_cost/4,
        // cost_gain=gain*0.5). This is the production code path —
        // every deployed Rust scheduler hits this branch.
        let s: Scheduler<StubItem, u32> = Scheduler::builder()
            .config(BatchConfig::default()) // max_batch_cost = 16_384
            .build();
        let snap = s.controller_snapshot().await;
        // Snapshot doesn't expose floors directly; reach into the
        // controller via the lock for these invariants.
        let ctrl = s.controller.lock().await;
        assert!(
            (ctrl.min_wait_ms - 15.0).abs() < f64::EPSILON,
            "builder must produce production-parity min_wait_ms=15.0, got {}",
            ctrl.min_wait_ms
        );
        assert_eq!(
            ctrl.min_batch_cost, 4_096,
            "builder must produce production-parity min_batch_cost=max_batch_tokens/4"
        );
        assert!(
            (ctrl.cost_gain - (ctrl.gain * 0.5)).abs() < f64::EPSILON,
            "builder must produce coupled cost_gain = gain*0.5"
        );
        // Sanity: snapshot still reflects an uncalibrated auto-mode
        // scheduler — production deploys auto-calibrate target_p50.
        assert!(!snap.calibrated);
        assert!(snap.target_p50_ms.is_none());
    }

    // ---- Aggregates ----

    #[tokio::test]
    async fn pending_count_and_cost_sum_across_batchers() {
        let s = make();
        s.submit(
            Op::Encode,
            LoraKey::from_name("a"),
            StubItem { cost: 5, idx: 0 },
            0,
        )
        .await;
        s.submit(
            Op::Encode,
            LoraKey::from_name("b"),
            StubItem { cost: 7, idx: 0 },
            0,
        )
        .await;
        s.submit(Op::Score, LoraKey::base(), StubItem { cost: 3, idx: 0 }, 0)
            .await;
        assert_eq!(s.total_pending_count().await, 3);
        assert_eq!(s.total_pending_cost().await, 15);
    }
}
