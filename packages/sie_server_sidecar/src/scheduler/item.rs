//! Concrete item + metadata types used by the dispatcher-level
//! [`crate::scheduler::Scheduler`].
//!
//! The [`Scheduler`](super::engine::Scheduler) is generic over the
//! item type `I: HasCost` and the caller-supplied metadata `T`. The
//! generic surface keeps the scheduler engine testable against simple
//! stubs (see `engine.rs::tests::StubItem`), but production needs one
//! concrete choice of `(I, T)` so the dispatcher can pass a single
//! [`super::SchedulerRegistry`] Arc around.
//!
//! That's what this module is: the single production choice.
//!
//! * [`SchedulerItem`] is a tagged union over the three op-specific
//!   batch items the Python adapter already consumes. Same on-wire
//!   types — no re-encoding when we pack them into a
//!   [`crate::ipc_types::RunBatchRequest`].
//! * [`SchedulerMeta`] carries everything the dispatcher's
//!   `apply_outcomes` path needs — the [`WorkItem`] for publishing
//!   the result, the JetStream [`Message`] for the ACK, and the
//!   payload-fetch timing we already record per item. Submit time is
//!   stashed as an [`Instant`] for queue-age telemetry (opt-in; the
//!   scheduler engine has its own per-batcher head-of-line atomic
//!   for FCFS, which is cheaper for the hot path).
//!
//! ## Cost semantics
//!
//! Scheduler flush triggers fire on **cost-cap** (sum of item costs),
//! **count-cap**, or **wait-cap**. The cost of a single item matches
//! Python's `PreparedTokens.total_tokens` when the dispatcher
//! tokenised Rust-side, and falls back to `1` otherwise:
//!
//! * **Encode / Extract**: sum of `prepared_tokens.input_ids[i].len()`
//!   — there is usually exactly one inner sequence per item, but we
//!   sum defensively. `1` when `prepared_tokens` is `None` (Python
//!   will retokenise).
//! * **Score**: same sum (the score items emit the
//!   `[query, doc_0]` layout), `1` when absent.
//!
//! The adaptive controller auto-calibrates against observed latency,
//! so being "approximately right" on cost is enough; what matters is
//! that the same cost function runs in both the cap check and the
//! `record_completion` → controller-step path so the controller
//! never over/under-counts its own flushes. That's what this module
//! pins down in one place.

use std::time::Instant;

use crate::delivery::Delivery;
use crate::ipc_types::{
    EncodeBatchItem, ExtractBatchItem, PreparedTokens, RunBatchItem, ScoreBatchItem,
};
use crate::work_types::WorkItem;

use super::batch_former::HasCost;
use super::engine::{LoraKey, Op};

/// Sum of per-inner-sequence token counts on a [`PreparedTokens`]
/// blob, clamped to `>= 1` so an empty-tokens payload doesn't yield
/// a zero-cost batcher (which would break the min-cost cap check in
/// the adaptive controller). `None` → `1` (same as Python's
/// "no prepared tokens, assume unit cost" fallback).
fn cost_from_prepared(pt: Option<&PreparedTokens>) -> u64 {
    let Some(pt) = pt else {
        return 1;
    };
    let total: u64 = pt.input_ids.iter().map(|seq| seq.len() as u64).sum::<u64>();
    total.max(1)
}

/// Concrete per-item payload the production scheduler carries. One of
/// the three variants, all of which are already the on-wire types the
/// Python adapter loop consumes — so packing a batch into a
/// [`crate::ipc_types::RunBatchRequest`] is O(items) moves and zero re-serialisation.
///
/// Named `SchedulerItem` rather than `Item` to avoid collision with
/// the msgpack-native `item` fields that live inside the inner variants.
#[derive(Debug, Clone)]
pub enum SchedulerItem {
    Encode(EncodeBatchItem),
    Score(ScoreBatchItem),
    Extract(ExtractBatchItem),
}

impl SchedulerItem {
    /// Operation class — used by the dispatcher to pick the right
    /// batcher key when submitting into the scheduler, and by the
    /// drain loop to tag metrics + logs.
    #[must_use]
    pub fn op(&self) -> Op {
        match self {
            Self::Encode(_) => Op::Encode,
            Self::Score(_) => Op::Score,
            Self::Extract(_) => Op::Extract,
        }
    }

    /// Convert into the IPC-level [`RunBatchItem`] tagged-struct.
    /// Matches the helper constructors in [`crate::ipc_types`] so
    /// the `op` discriminator is always synced with the populated
    /// optional field — never construct the struct by hand.
    #[must_use]
    pub fn into_run_batch_item(self) -> RunBatchItem {
        match self {
            Self::Encode(e) => RunBatchItem::encode(e),
            Self::Score(s) => RunBatchItem::score(s),
            Self::Extract(x) => RunBatchItem::extract(x),
        }
    }

    /// Same as [`Self::into_run_batch_item`] but additionally copies the
    /// W3C trace context (`traceparent` / `tracestate`) off the
    /// originating [`WorkItem`] onto the produced [`RunBatchItem`]. The
    /// per-op constructors have no `WorkItem` access, so the trace
    /// context is plumbed here — at the dispatch site where the batch's
    /// items are zipped back together with their metadata — so the
    /// Python non-streaming worker loop can re-extract the gateway span
    /// and attach `worker.run_batch` as its child.
    #[must_use]
    pub fn into_run_batch_item_with_trace(self, wi: &WorkItem) -> RunBatchItem {
        let mut rbi = self.into_run_batch_item();
        rbi.traceparent = wi.traceparent.clone();
        rbi.tracestate = wi.tracestate.clone();
        rbi
    }
}

impl HasCost for SchedulerItem {
    fn cost(&self) -> u64 {
        match self {
            Self::Encode(e) => cost_from_prepared(e.prepared_tokens.as_ref()),
            Self::Score(s) => cost_from_prepared(s.prepared_tokens.as_ref()),
            // Extract doesn't emit prepared tokens on the Rust side in
            // the current path (Python owns extract tokenisation — see
            // `docs/architecture-guide.md`), so
            // there is no per-item seq_len to pay with. Unit cost
            // matches Python's fallback.
            Self::Extract(_) => 1,
        }
    }

    fn original_index(&self) -> usize {
        // Original request-local index is what the cost-sorted
        // packer uses to restore order inside a flushed batch so
        // downstream outcomes line back up with `item_index`. Same
        // wiid → same original_index → stable ordering across retries.
        let idx = match self {
            Self::Encode(e) => e.item_index,
            Self::Score(s) => s.item_index,
            Self::Extract(x) => x.item_index,
        };
        idx as usize
    }
}

/// Caller-supplied per-item metadata the dispatcher needs to carry
/// through the scheduler to finish the request lifecycle on the
/// other side.
///
/// Not `Clone`: on the NATS arm the JetStream `Message` inside
/// [`Delivery`] owns the ACK/NAK token, which must not be duplicated.
/// The scheduler moves items by value (vec of `(item, metadata)` into
/// the batcher, vec back out of the flushed
/// [`crate::scheduler::batch_former::FormattedBatch`]), so
/// single-ownership is preserved end-to-end.
pub struct SchedulerMeta {
    /// The parsed work item — needed for `publish_result` / `publish_error`
    /// (reply subject, request id, timings, output types, etc.).
    pub wi: WorkItem,
    /// Settlement token (NATS message or local-ingest slot, P2.10 §4.6)
    /// — owns the ACK / NAK. Dropping a NATS one without either is safe
    /// but leaks: JetStream redelivers after `ack_wait` expires; a local
    /// one surfaces at the ingest layer as a missing-result timeout. The
    /// drain loop is responsible for ACKing on success and NAKing on
    /// backend failure.
    pub delivery: Delivery,
    /// Payload-store fetch latency (ms). Surfaced on the published
    /// [`crate::publisher::Timings`] and on the `payload_fetch_seconds`
    /// histogram.
    pub fetch_ms: f64,
    /// True when this item was pulled from a worker-specific direct-dispatch
    /// subject rather than the pool subject.
    pub worker_direct: bool,
    /// When the dispatcher enqueued this item into the scheduler.
    /// Used only for queue-age telemetry; the engine's FCFS pick is
    /// driven by its own `head_ns` atomic (see
    /// [`super::engine::Scheduler`]).
    pub submitted_at: Instant,
    /// Adapter worker child selected by the sidecar when this item entered
    /// the scheduler. Used to keep per-child queue-pressure metrics
    /// balanced even if model placement changes before the batch flushes.
    pub worker_child_index: Option<usize>,
}

impl SchedulerMeta {
    /// Construct the standard triple stamped with `Instant::now`.
    #[must_use]
    pub fn new(wi: WorkItem, delivery: Delivery, fetch_ms: f64) -> Self {
        Self::new_with_worker_direct(wi, delivery, fetch_ms, false)
    }

    /// Construct metadata with an explicit delivery-origin marker.
    #[must_use]
    pub fn new_with_worker_direct(
        wi: WorkItem,
        delivery: Delivery,
        fetch_ms: f64,
        worker_direct: bool,
    ) -> Self {
        Self {
            wi,
            delivery,
            fetch_ms,
            worker_direct,
            submitted_at: Instant::now(),
            worker_child_index: None,
        }
    }

    #[must_use]
    pub fn with_worker_child_index(mut self, child_index: usize) -> Self {
        self.worker_child_index = Some(child_index);
        self
    }
}

impl std::fmt::Debug for SchedulerMeta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The NATS `Message` inside `Delivery` intentionally doesn't
        // implement Debug in a useful way for our purposes — skip it.
        // The rest is cheap to print.
        f.debug_struct("SchedulerMeta")
            .field("wi_work_item_id", &self.wi.work_item_id)
            .field("wi_request_id", &self.wi.request_id)
            .field("fetch_ms", &self.fetch_ms)
            .field("worker_direct", &self.worker_direct)
            .field("submitted_at", &self.submitted_at)
            .field("worker_child_index", &self.worker_child_index)
            .finish_non_exhaustive()
    }
}

/// Read the LoRA key off a `WorkItem.options["lora"]`. Empty string
/// and absent both normalise to [`LoraKey::base`] via
/// [`LoraKey::from_option`]. Unknown-shape options (e.g. `lora` present
/// as a number) fall through to base so a misbehaving client can't
/// fragment the batcher map.
///
/// Defined here (not in `work_types.rs`) because routing policy is a
/// scheduler concern — keeping it adjacent to the scheduler's
/// [`LoraKey`] definition means one grep surfaces both the parser and
/// the consumer.
#[must_use]
pub fn lora_from_options(options: &Option<serde_json::Value>) -> LoraKey {
    let Some(opts) = options.as_ref() else {
        return LoraKey::base();
    };
    match opts.get("lora").and_then(|v| v.as_str()) {
        Some(s) => LoraKey::from_name(s),
        None => LoraKey::base(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ipc_types::WireValue;

    fn text_item(text: &str) -> WireValue {
        WireValue::Map(vec![(WireValue::from("text"), WireValue::from(text))])
    }

    fn pt_with_lens(lens: &[usize]) -> PreparedTokens {
        let input_ids: Vec<Vec<u32>> = lens
            .iter()
            .map(|&n| (0..n as u32).collect::<Vec<u32>>())
            .collect();
        PreparedTokens {
            input_ids,
            attention_mask: vec![],
            token_type_ids: vec![],
            tokenizer_id: "test".into(),
            max_seq_len: 512,
        }
    }

    fn encode_item(idx: u32, prepared: Option<PreparedTokens>) -> EncodeBatchItem {
        EncodeBatchItem {
            work_item_id: format!("r.{idx}"),
            request_id: "r".into(),
            item_index: idx,
            total_items: 1,
            timestamp: 0.0,
            item: text_item("x"),
            output_types: None,
            instruction: None,
            is_query: false,
            options: None,
            profile_id: None,
            bundle_config_hash: None,
            payload_fetch_ms: 0.0,
            prepared_tokens: prepared,
        }
    }

    #[test]
    fn cost_prefers_prepared_tokens_sum() {
        let it = SchedulerItem::Encode(encode_item(0, Some(pt_with_lens(&[17]))));
        assert_eq!(it.cost(), 17);
    }

    #[test]
    fn cost_sums_multiple_inner_sequences() {
        // Score emits `[query, doc_0]` as two inner sequences on the
        // same PreparedTokens — the flushed batch cost must count
        // both so the controller sees the real forward-pass cost.
        let it = SchedulerItem::Score(ScoreBatchItem {
            work_item_id: "r.0".into(),
            request_id: "r".into(),
            item_index: 0,
            total_items: 1,
            timestamp: 0.0,
            query_item: text_item("q"),
            score_items: vec![text_item("d")],
            instruction: None,
            options: None,
            profile_id: None,
            payload_fetch_ms: 0.0,
            prepared_tokens: Some(pt_with_lens(&[5, 13])),
        });
        assert_eq!(it.cost(), 18);
    }

    #[test]
    fn cost_clamps_to_one_on_empty_prepared() {
        // An empty `PreparedTokens` (no inner sequences) returns 0
        // without the clamp — that would let a submit land on a
        // scheduler whose min_batch_cost cap never trips. Clamping
        // to 1 matches Python's "unknown cost → assume one" fallback.
        let it = SchedulerItem::Encode(encode_item(0, Some(pt_with_lens(&[]))));
        assert_eq!(it.cost(), 1);
    }

    #[test]
    fn cost_falls_back_to_one_without_prepared() {
        let it = SchedulerItem::Encode(encode_item(0, None));
        assert_eq!(it.cost(), 1);
    }

    #[test]
    fn extract_cost_is_unit_by_policy() {
        let it = SchedulerItem::Extract(ExtractBatchItem {
            work_item_id: "r.0".into(),
            request_id: "r".into(),
            item_index: 0,
            total_items: 1,
            timestamp: 0.0,
            item: text_item("x"),
            labels: None,
            output_schema: None,
            instruction: None,
            options: None,
            profile_id: None,
            bundle_config_hash: None,
            payload_fetch_ms: 0.0,
        });
        assert_eq!(it.cost(), 1);
    }

    #[test]
    fn original_index_matches_item_index() {
        let it = SchedulerItem::Encode(encode_item(7, None));
        assert_eq!(it.original_index(), 7);
    }

    #[test]
    fn op_matches_variant() {
        let e = SchedulerItem::Encode(encode_item(0, None));
        assert_eq!(e.op(), Op::Encode);
        let s = SchedulerItem::Score(ScoreBatchItem {
            work_item_id: "".into(),
            request_id: "".into(),
            item_index: 0,
            total_items: 1,
            timestamp: 0.0,
            query_item: WireValue::Nil,
            score_items: vec![],
            instruction: None,
            options: None,
            profile_id: None,
            payload_fetch_ms: 0.0,
            prepared_tokens: None,
        });
        assert_eq!(s.op(), Op::Score);
        let x = SchedulerItem::Extract(ExtractBatchItem {
            work_item_id: "".into(),
            request_id: "".into(),
            item_index: 0,
            total_items: 1,
            timestamp: 0.0,
            item: WireValue::Nil,
            labels: None,
            output_schema: None,
            instruction: None,
            options: None,
            profile_id: None,
            bundle_config_hash: None,
            payload_fetch_ms: 0.0,
        });
        assert_eq!(x.op(), Op::Extract);
    }

    #[test]
    fn into_run_batch_item_syncs_discriminator() {
        let rbi = SchedulerItem::Encode(encode_item(3, None)).into_run_batch_item();
        assert_eq!(rbi.op, "encode");
        assert_eq!(rbi.work_item_id, "r.3");
        assert_eq!(rbi.request_id, "r");
        assert_eq!(rbi.item_index, 3);
        assert!(rbi.encode.is_some());
        assert!(rbi.score.is_none());
        assert!(rbi.extract.is_none());
    }

    #[test]
    fn into_run_batch_item_with_trace_copies_trace_context() {
        let mut wi = sample_work_item();
        wi.traceparent = Some("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01".into());
        wi.tracestate = Some("vendor=opaque".into());

        let rbi = SchedulerItem::Encode(encode_item(0, None)).into_run_batch_item_with_trace(&wi);

        assert_eq!(
            rbi.traceparent.as_deref(),
            Some("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01")
        );
        assert_eq!(rbi.tracestate.as_deref(), Some("vendor=opaque"));
        // The op/payload plumbing must still be intact.
        assert_eq!(rbi.op, "encode");
        assert!(rbi.encode.is_some());
    }

    #[test]
    fn into_run_batch_item_with_trace_defaults_to_none() {
        let wi = sample_work_item();
        let rbi = SchedulerItem::Encode(encode_item(0, None)).into_run_batch_item_with_trace(&wi);
        assert!(rbi.traceparent.is_none());
        assert!(rbi.tracestate.is_none());
    }

    fn sample_work_item() -> WorkItem {
        WorkItem {
            work_item_id: "r.0".into(),
            request_id: "r".into(),
            item_index: 0,
            total_items: 1,
            operation: "encode".into(),
            model_id: "BAAI/bge-m3".into(),
            profile_id: String::new(),
            engine: String::new(),
            pool_name: "l4".into(),
            admission_pool: "l4".into(),
            machine_profile: "l4-spot".into(),
            item: None,
            payload_ref: None,
            output_types: None,
            instruction: None,
            is_query: false,
            options: None,
            query_item: None,
            query_payload_ref: None,
            score_items: None,
            labels: None,
            output_schema: None,
            generate: None,
            routing_key: None,
            prompt_cache_key: None,
            bundle_config_hash: String::new(),
            router_id: "r1".into(),
            reply_subject: "_INBOX.r1.r".into(),
            traceparent: None,
            tracestate: None,
            timestamp: 0.0,
        }
    }

    // ---- LoraKey from options ----

    #[test]
    fn lora_from_options_none_is_base() {
        assert!(lora_from_options(&None).is_base());
    }

    #[test]
    fn lora_from_options_missing_key_is_base() {
        let opts = Some(serde_json::json!({"other": "x"}));
        assert!(lora_from_options(&opts).is_base());
    }

    #[test]
    fn lora_from_options_empty_string_is_base() {
        // Empty string must normalise the same way `LoraKey::from_name`
        // does, else the batcher map fragments on `""` vs `None`.
        let opts = Some(serde_json::json!({"lora": ""}));
        assert!(lora_from_options(&opts).is_base());
    }

    #[test]
    fn lora_from_options_string_preserved() {
        let opts = Some(serde_json::json!({"lora": "tenant-a"}));
        let key = lora_from_options(&opts);
        assert_eq!(key.as_str(), Some("tenant-a"));
    }

    #[test]
    fn lora_from_options_non_string_falls_back_to_base() {
        // A client sending `{"lora": 42}` by accident must not
        // fragment the map on a non-str key. The parser returns
        // base; Python would do the same via its `options.get("lora")`
        // call chain (numbers coerce to str there, but the Rust
        // policy is stricter — prefer base to a silent weird key).
        let opts = Some(serde_json::json!({"lora": 42}));
        assert!(lora_from_options(&opts).is_base());
    }
}
