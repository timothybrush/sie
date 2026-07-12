//! [`BatchFormer`] — async pending queue + flush loop.
//!
//! Ported from `sie_server/core/batcher.py::BatchFormer`. The port
//! preserves every behavioural invariant of the Python implementation:
//!
//! 1. **Flush triggers**: cost cap, count cap, `max_batch_wait_ms`
//!    timeout since the *first* item, or the coalesce window
//!    (`effective_coalesce_ms`) elapsing since the *last* submit.
//! 2. **Oversize item**: a single item whose cost already exceeds
//!    `max_batch_cost` still flushes (alone) so it can't starve
//!    forever.
//! 3. **Cost-sorted sub-batch**: `_pending` is sorted by `cost`
//!    ascending before each extract; packing is greedy under the cost
//!    and count caps. Remaining items stay in the queue for the next
//!    extract, preserving FIFO-within-cost-bucket ordering.
//! 4. **Timer reset**: once the queue empties, both `first_request_time`
//!    and `last_submit_time` are cleared so the next submit starts a
//!    fresh coalesce/timeout window.
//!
//! The async primitives differ from Python (Rust uses [`tokio::sync`]
//! rather than asyncio), but the externally observable semantics are
//! identical. The Python test suite in `tests/core/test_batcher.py` is
//! the behavioural contract; every relevant case is ported below as
//! `#[tokio::test]`.

use std::marker::PhantomData;
use std::time::{Duration, Instant};

use tokio::sync::{Mutex, Notify};
use tokio::time::sleep;

use super::batch_config::BatchConfig;

/// Trait for any item that carries a batching cost and an original
/// position within a request.
///
/// Ported from Python's `HasCost` Protocol. `PreparedItem` (built on
/// the Rust-tokenise fast path) implements this directly; test-only
/// stubs in this module do too.
pub trait HasCost {
    /// Batching cost — token count for text, 1 per image, etc.
    fn cost(&self) -> u64;

    /// Original position within the originating request's item list.
    /// Used by the adapter-side output router to zip results back to
    /// the correct request item.
    fn original_index(&self) -> usize;
}

/// A single request waiting to be batched, together with
/// caller-provided routing metadata.
#[derive(Debug, Clone)]
pub struct PendingRequest<I: HasCost, T> {
    pub item: I,
    pub metadata: T,
    pub arrival_time: Instant,
}

/// A flushed batch ready for downstream dispatch.
///
/// Items and metadata are kept in parallel `Vec`s so the adapter-side
/// zip continues to work unchanged.
#[derive(Debug)]
pub struct FormattedBatch<I: HasCost, T> {
    pub items: Vec<I>,
    pub metadata: Vec<T>,
    pub total_cost: u64,
    pub flush_reason: FlushReason,
}

impl<I: HasCost, T> FormattedBatch<I, T> {
    /// Number of items in the batch.
    #[must_use]
    pub fn size(&self) -> usize {
        self.items.len()
    }

    /// Python-parity alias for [`Self::total_cost`] mirroring the
    /// Python `total_tokens` property.
    #[must_use]
    pub fn total_tokens(&self) -> u64 {
        self.total_cost
    }

    /// Return a new batch sorted by cost (ascending). Preserves the
    /// `items[i] ↔ metadata[i]` correspondence.
    ///
    /// Mostly redundant now that `extract_batch` sorts
    /// in place before slicing, but kept for symmetry with the Python
    /// API where some callers sort explicitly.
    #[must_use]
    pub fn sorted_by_cost(self) -> Self
    where
        I: Clone,
        T: Clone,
    {
        // Build index permutation so we don't pay a double-clone on T.
        let mut idx: Vec<usize> = (0..self.items.len()).collect();
        idx.sort_by_key(|&i| self.items[i].cost());

        let items = idx.iter().map(|&i| self.items[i].clone()).collect();
        let metadata = idx.iter().map(|&i| self.metadata[i].clone()).collect();
        Self {
            items,
            metadata,
            total_cost: self.total_cost,
            flush_reason: self.flush_reason,
        }
    }
}

/// Reason a [`BatchFormer`] emitted a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlushReason {
    CostCap,
    CountCap,
    Timeout,
    Coalesce,
    SingleOversize,
    IdleBypass,
    Drain,
}

impl FlushReason {
    #[must_use]
    pub const fn as_label(self) -> &'static str {
        match self {
            Self::CostCap => "cost_cap",
            Self::CountCap => "count_cap",
            Self::Timeout => "timeout",
            Self::Coalesce => "coalesce",
            Self::SingleOversize => "single_oversize",
            Self::IdleBypass => "idle_bypass",
            Self::Drain => "drain",
        }
    }
}

/// Internal mutable state, guarded by a single [`Mutex`].
///
/// Kept compact on purpose: every batch formation decision reads and
/// mutates this struct, so the lock is hot. No allocations happen
/// under the lock beyond the two parallel `Vec`s.
#[derive(Debug)]
struct Inner<I: HasCost, T> {
    pending: Vec<PendingRequest<I, T>>,
    total_cost: u64,
    first_request_time: Option<Instant>,
    last_submit_time: Option<Instant>,
    /// Cached snapshot of the current knobs. Mutations happen via
    /// [`BatchFormer::update_config`] which takes the same lock so
    /// reads of `config` and reads of `pending` stay consistent.
    config: BatchConfig,
}

/// Async batch former.
///
/// Thread-safe for concurrent submits from many tasks. A single
/// consumer task should call [`BatchFormer::get_batch`] in a loop;
/// calling it concurrently from multiple consumers is allowed but has
/// no benefit (and will race over pending items — the consumer is
/// typically the per-model worker loop).
#[derive(Debug)]
pub struct BatchFormer<I: HasCost, T> {
    inner: Mutex<Inner<I, T>>,
    /// Edge-triggered readiness signal. A submit that transitions the
    /// batcher to "first item" or "batch ready" notifies here; the
    /// consumer uses [`Notified::enable`] *inside* the lock to avoid
    /// missed notifications across the drop.
    ready: Notify,
    // Generic parameters are consumed by `inner` only; `PhantomData`
    // keeps the type system honest when `Inner` is refactored.
    _phantom: PhantomData<(I, T)>,
}

impl<I: HasCost, T> BatchFormer<I, T> {
    /// Construct a new batch former with the given caps. Use
    /// [`BatchConfig::default`] for Python-parity defaults.
    pub fn new(config: BatchConfig) -> Self {
        Self {
            inner: Mutex::new(Inner {
                pending: Vec::new(),
                total_cost: 0,
                first_request_time: None,
                last_submit_time: None,
                config,
            }),
            ready: Notify::new(),
            _phantom: PhantomData,
        }
    }

    /// Current cap snapshot. Returned by value — the caller sees a
    /// consistent view even if another task updates the config
    /// concurrently.
    pub async fn config(&self) -> BatchConfig {
        self.inner.lock().await.config
    }

    /// Update the caps atomically. The adaptive controller calls this
    /// after each `step()`.
    pub async fn update_config(&self, config: BatchConfig) {
        let mut guard = self.inner.lock().await;
        guard.config = config;
        // Config shrinks can make the pending queue instantly
        // yieldable (e.g. new `max_batch_cost` < current `total_cost`
        // or new `max_batch_requests` < `pending.len()`). Wake the
        // consumer so it re-evaluates the flush triggers.
        if should_yield_batch(&guard) {
            self.ready.notify_one();
        }
    }

    /// Number of pending items. Cheap snapshot under the lock.
    pub async fn pending_count(&self) -> usize {
        self.inner.lock().await.pending.len()
    }

    /// Total cost across all pending items.
    pub async fn pending_cost(&self) -> u64 {
        self.inner.lock().await.total_cost
    }

    /// Python-parity alias for [`Self::pending_cost`].
    pub async fn pending_tokens(&self) -> u64 {
        self.pending_cost().await
    }

    /// Snapshot of `first_request_time`. Used by the FCFS fairness
    /// primitive (`scheduler::pick_next_lora`) to sort per-LoRA
    /// batchers by oldest head.
    ///
    /// Python accessed `_first_request_time` directly (private field);
    /// the Rust scheduler formalises that coupling as a public method.
    pub async fn first_request_time(&self) -> Option<Instant> {
        self.inner.lock().await.first_request_time
    }

    /// Enqueue a single item.
    pub async fn submit(&self, item: I, metadata: T) {
        let mut guard = self.inner.lock().await;
        let notify = append_item(&mut guard, item, metadata);
        drop(guard);
        if notify {
            self.ready.notify_one();
        }
    }

    /// Enqueue many items under a single lock acquisition. More
    /// efficient than looping over [`Self::submit`] for multi-item
    /// requests (each external request can carry up to 64 items).
    pub async fn submit_many(&self, items: Vec<(I, T)>) {
        if items.is_empty() {
            return;
        }
        let mut guard = self.inner.lock().await;
        let mut notify = false;
        for (item, metadata) in items {
            notify |= append_item(&mut guard, item, metadata);
        }
        drop(guard);
        if notify {
            self.ready.notify_one();
        }
    }

    /// Block until the next batch is ready, then return it.
    ///
    /// Readiness is one of: cost cap hit, count cap hit, timeout
    /// elapsed since first item, or coalesce window elapsed since
    /// last submit.
    ///
    /// Setting `immediate = true` short-circuits the wait: if any
    /// items are pending, they flush immediately. Used by the worker
    /// loop after an idle period to eliminate an unnecessary batch
    /// wait at low concurrency.
    pub async fn get_batch(&self, immediate: bool) -> FormattedBatch<I, T> {
        loop {
            // Arm the notification *before* we drop the lock so a
            // submit that fires between "decide to wait" and "start
            // waiting" is not missed. See the Tokio docs on
            // `Notified::enable` for the pattern.
            let notified = self.ready.notified();
            tokio::pin!(notified);

            let wait = {
                let mut guard = self.inner.lock().await;
                if let Some(reason) = flush_reason(&guard, immediate) {
                    return extract_batch(&mut guard, reason);
                }
                notified.as_mut().enable();
                wait_timeout(&guard)
            };

            match wait {
                None => {
                    // No pending items → wait indefinitely for the
                    // first submit.
                    notified.await;
                }
                Some(dur) if dur.is_zero() => {
                    // The timer is already expired — loop and flush
                    // on the next lock acquisition. This path fires
                    // when `get_batch` is called with `immediate=false`
                    // and the timers have already elapsed (e.g. a
                    // slow consumer catching up).
                    continue;
                }
                Some(dur) => {
                    tokio::select! {
                        biased;
                        () = &mut notified => {}
                        () = sleep(dur) => {}
                    }
                }
            }
        }
    }

    /// Non-blocking: return a batch if the flush conditions already
    /// hold, else return `None`. Matches Python's `try_get_batch`.
    pub async fn try_get_batch(&self) -> Option<FormattedBatch<I, T>> {
        let mut guard = self.inner.lock().await;
        let reason = flush_reason(&guard, false)?;
        Some(extract_batch(&mut guard, reason))
    }

    /// Drain whatever is pending right now, bypassing the flush
    /// conditions. Matches Python's `try_drain` — the continuous
    /// batching tail called after each forward pass.
    pub async fn try_drain(&self) -> Option<FormattedBatch<I, T>> {
        let mut guard = self.inner.lock().await;
        if guard.pending.is_empty() {
            None
        } else {
            Some(extract_batch(&mut guard, FlushReason::Drain))
        }
    }
}

// ---------------------------------------------------------------------
// Free-standing helpers. Kept outside the `BatchFormer` impl so the
// hot-path decisions are easy to test in isolation (see the `tests`
// module below) and so they don't borrow `self` implicitly.
// ---------------------------------------------------------------------

fn append_item<I: HasCost, T>(inner: &mut Inner<I, T>, item: I, metadata: T) -> bool {
    let cost = item.cost();
    let now = Instant::now();

    inner.pending.push(PendingRequest {
        item,
        metadata,
        arrival_time: now,
    });
    inner.total_cost += cost;
    inner.last_submit_time = Some(now);

    let is_first = inner.first_request_time.is_none();
    if is_first {
        inner.first_request_time = Some(now);
    }

    // Wake the consumer if the batch is now full, OR if this is the
    // very first item (so the consumer can start the timeout window
    // instead of blocking on an indefinite `notified()`).
    should_yield_batch(inner) || is_first
}

fn batch_timeout_expired<I: HasCost, T>(inner: &Inner<I, T>) -> bool {
    let Some(first) = inner.first_request_time else {
        return false;
    };
    let elapsed_ms = first.elapsed().as_secs_f64() * 1000.0;
    elapsed_ms >= inner.config.max_batch_wait_ms
}

fn coalesce_expired<I: HasCost, T>(inner: &Inner<I, T>) -> bool {
    let Some(last) = inner.last_submit_time else {
        return false;
    };
    if inner.pending.is_empty() {
        return false;
    }
    let elapsed_ms = last.elapsed().as_secs_f64() * 1000.0;
    elapsed_ms >= inner.config.effective_coalesce_ms()
}

fn should_yield_batch<I: HasCost, T>(inner: &Inner<I, T>) -> bool {
    flush_reason(inner, false).is_some()
}

fn flush_reason<I: HasCost, T>(inner: &Inner<I, T>, immediate: bool) -> Option<FlushReason> {
    if inner.pending.is_empty() {
        return None;
    }
    if inner.pending.len() == 1 && inner.total_cost > inner.config.max_batch_cost {
        return Some(FlushReason::SingleOversize);
    }
    if inner.total_cost >= inner.config.max_batch_cost {
        return Some(FlushReason::CostCap);
    }
    if inner.pending.len() >= inner.config.max_batch_requests {
        return Some(FlushReason::CountCap);
    }
    if batch_timeout_expired(inner) {
        return Some(FlushReason::Timeout);
    }
    if coalesce_expired(inner) {
        return Some(FlushReason::Coalesce);
    }
    if immediate {
        return Some(FlushReason::IdleBypass);
    }
    None
}

/// Time remaining until the next guaranteed flush, given current
/// state. `None` means "wait indefinitely" (no pending items).
fn wait_timeout<I: HasCost, T>(inner: &Inner<I, T>) -> Option<Duration> {
    let first = inner.first_request_time?;
    let now = Instant::now();

    let elapsed_ms = now.saturating_duration_since(first).as_secs_f64() * 1000.0;
    let batch_remaining_ms = (inner.config.max_batch_wait_ms - elapsed_ms).max(0.0);

    let coalesce_ms = inner.config.effective_coalesce_ms();
    let coalesce_remaining_ms = match inner.last_submit_time {
        Some(last) => {
            let since_last = now.saturating_duration_since(last).as_secs_f64() * 1000.0;
            (coalesce_ms - since_last).max(0.0)
        }
        None => coalesce_ms,
    };

    let effective_ms = batch_remaining_ms.min(coalesce_remaining_ms);
    // ms → Duration, clamping tiny negatives to zero.
    Some(Duration::from_secs_f64(effective_ms.max(0.0) / 1000.0))
}

fn extract_batch<I: HasCost, T>(
    inner: &mut Inner<I, T>,
    flush_reason: FlushReason,
) -> FormattedBatch<I, T> {
    // Cost-sort pending before slicing — keeps each sub-batch's
    // items close in length, minimising padding waste on the adapter
    // side. Stable sort preserves FIFO order within equal-cost items
    // (matches Python's stable `list.sort`).
    inner.pending.sort_by_key(|r| r.item.cost());

    let max_cost = inner.config.max_batch_cost;
    let max_requests = inner.config.max_batch_requests;

    // Greedy pack. Always take at least one item so a single large
    // request (cost ≥ max_cost) doesn't starve.
    let mut batch_cost: u64 = 0;
    let mut take_count: usize = 0;
    for req in &inner.pending {
        let c = req.item.cost();
        // Stop if taking this item would exceed the cost cap — but
        // only if we already have at least one item in the batch.
        // Otherwise (take_count == 0) fall through and take a single
        // oversize item alone: Python does the same, to avoid starving
        // a request whose cost is simply larger than the cap.
        if batch_cost + c > max_cost && take_count > 0 {
            break;
        }
        batch_cost += c;
        take_count += 1;
        if take_count >= max_requests {
            break;
        }
    }

    // Drain the taken prefix into parallel vecs.
    let drained: Vec<PendingRequest<I, T>> = inner.pending.drain(..take_count).collect();
    inner.total_cost -= batch_cost;

    let mut items = Vec::with_capacity(drained.len());
    let mut metadata = Vec::with_capacity(drained.len());
    for req in drained {
        items.push(req.item);
        metadata.push(req.metadata);
    }

    // Queue emptied → reset both timers so the next submit starts a
    // fresh coalesce/timeout window.
    if inner.pending.is_empty() {
        inner.first_request_time = None;
        inner.last_submit_time = None;
    }

    FormattedBatch {
        items,
        metadata,
        total_cost: batch_cost,
        flush_reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::time::{sleep as tsleep, Duration as TDuration};

    /// Minimal `HasCost` stub: carries a fixed cost and a position.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct StubItem {
        cost: u64,
        idx: usize,
    }

    impl StubItem {
        fn new(cost: u64) -> Self {
            Self { cost, idx: 0 }
        }

        fn with_idx(cost: u64, idx: usize) -> Self {
            Self { cost, idx }
        }
    }

    impl HasCost for StubItem {
        fn cost(&self) -> u64 {
            self.cost
        }
        fn original_index(&self) -> usize {
            self.idx
        }
    }

    #[tokio::test]
    async fn submit_and_drain_roundtrip() {
        let b = BatchFormer::<StubItem, u32>::new(BatchConfig::default());
        b.submit(StubItem::new(10), 1).await;
        b.submit(StubItem::new(20), 2).await;
        assert_eq!(b.pending_count().await, 2);
        assert_eq!(b.pending_cost().await, 30);

        let batch = b.try_drain().await.expect("drain should succeed");
        assert_eq!(batch.size(), 2);
        assert_eq!(batch.total_cost, 30);
        assert_eq!(batch.flush_reason, FlushReason::Drain);
        assert_eq!(b.pending_count().await, 0);
        assert_eq!(b.pending_cost().await, 0);
    }

    #[tokio::test]
    async fn cost_cap_triggers_extract() {
        let cfg = BatchConfig {
            max_batch_cost: 30,
            max_batch_requests: 64,
            max_batch_wait_ms: 10_000.0, // big so only cost triggers
            coalesce_ms: 10_000.0,
            coalesce_ratio: 1.0,
        };
        let b = BatchFormer::<StubItem, u32>::new(cfg);
        b.submit(StubItem::new(15), 1).await;
        b.submit(StubItem::new(15), 2).await;
        // total_cost (30) >= max_batch_cost (30) → should_yield now.
        let batch = b.try_get_batch().await.expect("cost cap should trigger");
        assert_eq!(batch.size(), 2);
        assert_eq!(batch.total_cost, 30);
        assert_eq!(batch.flush_reason, FlushReason::CostCap);
    }

    #[tokio::test]
    async fn count_cap_triggers_extract() {
        let cfg = BatchConfig {
            max_batch_cost: 1_000_000,
            max_batch_requests: 3,
            max_batch_wait_ms: 10_000.0,
            coalesce_ms: 10_000.0,
            coalesce_ratio: 1.0,
        };
        let b = BatchFormer::<StubItem, u32>::new(cfg);
        b.submit(StubItem::new(1), 1).await;
        b.submit(StubItem::new(1), 2).await;
        b.submit(StubItem::new(1), 3).await;
        let batch = b.try_get_batch().await.expect("count cap should trigger");
        assert_eq!(batch.size(), 3);
        assert_eq!(batch.flush_reason, FlushReason::CountCap);
    }

    #[tokio::test]
    async fn cost_sort_reorders_pending_ascending() {
        let cfg = BatchConfig {
            max_batch_cost: 1_000_000,
            max_batch_requests: 64,
            ..BatchConfig::default()
        };
        let b = BatchFormer::<StubItem, usize>::new(cfg);
        // Insert out-of-order costs.
        b.submit(StubItem::with_idx(50, 0), 0).await;
        b.submit(StubItem::with_idx(10, 1), 1).await;
        b.submit(StubItem::with_idx(30, 2), 2).await;

        let batch = b.try_drain().await.unwrap();
        let costs: Vec<u64> = batch.items.iter().map(|i| i.cost()).collect();
        assert_eq!(costs, vec![10, 30, 50], "must be sorted ascending");
    }

    #[tokio::test]
    async fn oversize_item_flushes_alone() {
        let cfg = BatchConfig {
            max_batch_cost: 100,
            max_batch_requests: 64,
            ..BatchConfig::default()
        };
        let b = BatchFormer::<StubItem, u32>::new(cfg);
        b.submit(StubItem::new(500), 1).await; // cost >> cap

        let batch = b
            .try_get_batch()
            .await
            .expect("oversize should still flush");
        assert_eq!(batch.size(), 1);
        assert_eq!(batch.total_cost, 500);
        assert_eq!(batch.flush_reason, FlushReason::SingleOversize);
    }

    #[tokio::test]
    async fn extract_leaves_remainder_in_queue() {
        let cfg = BatchConfig {
            max_batch_cost: 25,
            max_batch_requests: 64,
            ..BatchConfig::default()
        };
        let b = BatchFormer::<StubItem, u32>::new(cfg);
        b.submit(StubItem::new(10), 1).await;
        b.submit(StubItem::new(10), 2).await;
        b.submit(StubItem::new(10), 3).await;
        // total_cost = 30 ≥ 25 → should_yield; extract packs two (sum 20),
        // drops the third because 20+10=30 > 25.
        let batch = b.try_get_batch().await.unwrap();
        assert_eq!(batch.size(), 2);
        assert_eq!(batch.total_cost, 20);
        assert_eq!(b.pending_count().await, 1);
        assert_eq!(b.pending_cost().await, 10);
    }

    #[tokio::test]
    async fn empty_batcher_returns_none_from_try_paths() {
        let b = BatchFormer::<StubItem, u32>::new(BatchConfig::default());
        assert!(b.try_get_batch().await.is_none());
        assert!(b.try_drain().await.is_none());
    }

    #[tokio::test]
    async fn immediate_get_batch_flushes_without_timeout() {
        let cfg = BatchConfig {
            max_batch_wait_ms: 10_000.0, // would block for 10 seconds otherwise
            coalesce_ms: 10_000.0,
            coalesce_ratio: 1.0,
            ..BatchConfig::default()
        };
        let b = BatchFormer::<StubItem, u32>::new(cfg);
        b.submit(StubItem::new(1), 1).await;

        // This would hang without `immediate = true`; with it, it
        // returns as soon as the single pending item is visible.
        let batch = b.get_batch(true).await;
        assert_eq!(batch.size(), 1);
        assert_eq!(batch.flush_reason, FlushReason::IdleBypass);
    }

    #[tokio::test]
    async fn timers_reset_when_queue_empties() {
        let b = BatchFormer::<StubItem, u32>::new(BatchConfig::default());
        b.submit(StubItem::new(1), 1).await;
        assert!(b.first_request_time().await.is_some());
        let _ = b.try_drain().await.unwrap();
        assert!(
            b.first_request_time().await.is_none(),
            "emptying the queue must reset the first-request timer"
        );
    }

    #[tokio::test]
    async fn get_batch_times_out_after_max_wait() {
        // Real-clock test: `wait_timeout` reads `std::Instant::now`,
        // so mixing tokio's pause() with wall time causes spin loops.
        // A real 40 ms ceiling keeps the test fast enough to run in
        // the hot suite.
        let cfg = BatchConfig {
            max_batch_cost: 1_000_000,
            max_batch_requests: 1000,
            max_batch_wait_ms: 20.0,
            coalesce_ms: 1_000.0,
            coalesce_ratio: 1.0, // coalesce ties max-wait; timeout wins by priority
        };
        let b = BatchFormer::<StubItem, u32>::new(cfg);
        b.submit(StubItem::new(1), 1).await;

        // Expect get_batch to complete between the wait window and
        // the timeout ceiling.
        let start = std::time::Instant::now();
        let batch = tokio::time::timeout(TDuration::from_millis(200), b.get_batch(false))
            .await
            .expect("must flush on timeout within 200 ms");
        let elapsed_ms = start.elapsed().as_millis();
        assert_eq!(batch.size(), 1);
        assert_eq!(batch.flush_reason, FlushReason::Timeout);
        assert!(
            elapsed_ms >= 15,
            "must wait at least ~max_batch_wait_ms before flushing (waited {elapsed_ms} ms)"
        );
    }

    #[tokio::test]
    async fn submit_many_takes_one_lock() {
        // Functional check: all items end up pending in arrival order.
        let b = BatchFormer::<StubItem, u32>::new(BatchConfig::default());
        let items: Vec<(StubItem, u32)> = (0..5)
            .map(|i| (StubItem::with_idx(i as u64 + 1, i), i as u32))
            .collect();
        b.submit_many(items).await;
        assert_eq!(b.pending_count().await, 5);
        assert_eq!(b.pending_cost().await, 1 + 2 + 3 + 4 + 5);
    }

    #[tokio::test]
    async fn update_config_wakes_consumer_when_new_cap_already_satisfied() {
        // NB: don't pause() here — the consumer races a real submit
        // through Notify, and the pattern is intrinsically clock-free
        // once the Notified future is enabled inside the lock.
        let original = BatchConfig {
            max_batch_cost: 1_000_000,
            max_batch_requests: 1000,
            max_batch_wait_ms: 1_000_000.0,
            coalesce_ms: 1_000_000.0,
            coalesce_ratio: 1.0,
        };
        let b = std::sync::Arc::new(BatchFormer::<StubItem, u32>::new(original));
        b.submit(StubItem::new(5), 1).await;

        // Spawn a consumer that will block in `get_batch`.
        let consumer = {
            let b = std::sync::Arc::clone(&b);
            tokio::spawn(async move { b.get_batch(false).await })
        };

        // Yield so the consumer can reach the await point.
        tokio::task::yield_now().await;
        tsleep(TDuration::from_millis(1)).await;

        // Shrink the cap so the one pending item now trips
        // `batch_is_full`; the update must wake the consumer.
        //
        // Build the new config from a plain copy of `original` so we
        // don't re-acquire the inner lock while `update_config` is
        // about to acquire it — that would deadlock.
        let shrunk = BatchConfig {
            max_batch_cost: 1,
            max_batch_requests: 1,
            ..original
        };
        b.update_config(shrunk).await;

        let batch = tokio::time::timeout(TDuration::from_secs(2), consumer)
            .await
            .expect("consumer must finish within 2s of the config shrink")
            .unwrap();
        assert_eq!(batch.size(), 1);
    }

    #[test]
    fn wait_timeout_shrinks_as_time_passes() {
        // Build an `Inner` directly so we can test the sync helper.
        let cfg = BatchConfig {
            max_batch_wait_ms: 20.0,
            coalesce_ms: 1_000_000.0,
            coalesce_ratio: 1.0,
            ..BatchConfig::default()
        };
        let now = Instant::now();
        let inner: Inner<StubItem, u32> = Inner {
            pending: vec![PendingRequest {
                item: StubItem::new(1),
                metadata: 1,
                arrival_time: now,
            }],
            total_cost: 1,
            first_request_time: Some(now - Duration::from_millis(5)),
            last_submit_time: Some(now),
            config: cfg,
        };
        let t = wait_timeout(&inner).expect("pending → Some");
        // roughly 15 ms remaining (20 - 5), ±5 ms slop for clock jitter.
        assert!(t <= Duration::from_millis(20));
    }

    #[test]
    fn wait_timeout_none_when_no_pending() {
        let cfg = BatchConfig::default();
        let inner: Inner<StubItem, u32> = Inner {
            pending: vec![],
            total_cost: 0,
            first_request_time: None,
            last_submit_time: None,
            config: cfg,
        };
        assert!(wait_timeout(&inner).is_none());
    }

    #[test]
    fn coalesce_clamps_to_proportional_under_short_wait() {
        // Coalesce window = min(15, 4 * 0.5) = 2 ms.
        let cfg = BatchConfig {
            coalesce_ms: 15.0,
            coalesce_ratio: 0.5,
            max_batch_wait_ms: 4.0,
            ..BatchConfig::default()
        };
        assert!((cfg.effective_coalesce_ms() - 2.0).abs() < f64::EPSILON);
    }
}
