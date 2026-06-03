//! Encode an `ItemOutcome` into a `WorkResult` and publish it to the
//! gateway's reply subject.

use async_nats::Client;
use thiserror::Error;
use tracing::debug;

use crate::ipc_types::{Disposition, ItemOutcome, RawOutput};
use crate::output::{
    build_dense_payload, build_multivector_payload, build_score_payload, build_sparse_payload,
    l2_normalize_in_place, ShapeError,
};
use crate::work_types::WorkResult;

#[derive(Debug, Error)]
pub enum PublishError {
    #[error("nats publish: {0}")]
    Nats(#[from] async_nats::PublishError),
    #[error("encode WorkResult: {0}")]
    Encode(#[from] rmp_serde::encode::Error),
    #[error("empty reply_subject — cannot publish")]
    EmptyReplySubject,
}

/// Per-item timings stamped onto `WorkResult` by the worker.
///
/// * `queue_ms` — wall-clock from gateway publish (`WorkItem.timestamp`)
///   to worker result publish.
/// * `payload_fetch_ms` — time spent resolving `payload_ref` /
///   `query_payload_ref` against the payload store; `0.0` for inline items.
#[derive(Debug, Clone, Copy, Default)]
pub struct Timings {
    pub queue_ms: f64,
    pub payload_fetch_ms: f64,
}

pub struct WorkPublisher {
    client: Client,
    worker_id: String,
}

impl WorkPublisher {
    pub fn new(client: Client, worker_id: impl Into<String>) -> Self {
        Self {
            client,
            worker_id: worker_id.into(),
        }
    }

    pub fn worker_id(&self) -> &str {
        &self.worker_id
    }

    /// Publish a `WorkResult` built from `outcome` + worker-side `timings`.
    /// Returns `PublishError::EmptyReplySubject` on empty subject so the
    /// caller can treat the publish (and ACK) as fire-and-forget.
    pub async fn publish_result(
        &self,
        reply_subject: &str,
        outcome: &ItemOutcome,
        timings: Option<Timings>,
    ) -> Result<(), PublishError> {
        check_reply_subject(reply_subject)?;

        // If Python emitted `raw_output` (deferring wire framing to Rust),
        // shape it into `result_msgpack` here. Any shape error becomes an
        // error `WorkResult` so a misconfigured model cannot silently drop
        // a request.
        let owned;
        let effective = match shape_raw_output_for_wire(outcome) {
            ShapeOutcome::Unchanged => outcome,
            ShapeOutcome::Shaped(o) => {
                owned = o;
                &owned
            }
        };

        let result = build_work_result(effective, &self.worker_id, timings);
        let bytes = rmp_serde::to_vec_named(&result)?;
        debug!(
            reply = %reply_subject,
            request_id = %result.request_id,
            bytes = bytes.len(),
            "publishing WorkResult"
        );
        self.client
            .publish(reply_subject.to_string(), bytes.into())
            .await?;
        Ok(())
    }

    /// Publish a pre-encoded generation chunk produced by Python's
    /// `StreamingProcessor`. The sidecar owns NATS I/O; Python owns the
    /// chunk envelope bytes.
    pub async fn publish_raw(
        &self,
        reply_subject: &str,
        payload: Vec<u8>,
    ) -> Result<(), PublishError> {
        check_reply_subject(reply_subject)?;
        self.client
            .publish(reply_subject.to_string(), payload.into())
            .await?;
        Ok(())
    }
}

/// Result of inspecting an `ItemOutcome.raw_output` for Rust-side
/// shaping: either pass through unchanged, or emit a replacement
/// outcome with `result_msgpack` filled in (success or error).
///
/// `Shaped` carries a full `ItemOutcome` (~360 B) so the enum is
/// variant-heavy. Boxing the inner outcome to shrink the enum is a
/// mechanical refactor tracked for its own cleanup cycle — every
/// consumer site would have to thread a `Box` through, which is noise
/// for this hot path. Scoped `#[allow]` here rather than a
/// crate-wide one so the warning re-fires if anyone else builds a
/// heavyweight enum elsewhere.
#[allow(clippy::large_enum_variant)]
enum ShapeOutcome {
    Unchanged,
    Shaped(ItemOutcome),
}

/// Attempt to shape `outcome.raw_output` into `outcome.result_msgpack`
/// bytes using the Rust-side output shapers.
///
/// Rules:
///
/// * `raw_output == None` → `Unchanged`. Legacy Python path.
/// * `raw_output == Some(_)` but `result_msgpack` is non-empty →
///   `Unchanged`. Python gave us pre-framed bytes; trust them. The
///   Python side gates `raw_output` emission behind an env var, so
///   a mixed outcome is a wire artefact of a rolling deploy and
///   we take the safer, already-framed path.
/// * `raw_output == Some(_)` with empty `result_msgpack` and a
///   recognised variant → run the shaper; on success emit
///   `PublishAndAck` with the produced bytes, on failure emit
///   `PublishErrorAndAck` with the error message. Never silently
///   sinks the work item.
/// * `raw_output == Some(_)` with no recognised variant populated
///   (e.g. a future Rust build sees a variant it doesn't know) →
///   emit `PublishErrorAndAck` so we don't hand back an empty
///   `result_msgpack` that the SDK would misinterpret as "success
///   but no embeddings".
fn shape_raw_output_for_wire(outcome: &ItemOutcome) -> ShapeOutcome {
    let Some(raw) = outcome.raw_output.as_ref() else {
        return ShapeOutcome::Unchanged;
    };
    if !outcome.result_msgpack.is_empty() {
        return ShapeOutcome::Unchanged;
    }
    // Only shape on success outcomes. Error dispositions carry no
    // payload anyway, so fall through to legacy handling.
    if !matches!(outcome.disposition, Disposition::PublishAndAck) {
        return ShapeOutcome::Unchanged;
    }

    match shape(raw) {
        Ok(bytes) => ShapeOutcome::Shaped(ItemOutcome {
            result_msgpack: bytes,
            // Drop the raw_output so we never pay the serialisation
            // cost for the inner arrays again in `rmp_serde::to_vec`.
            raw_output: None,
            ..outcome.clone()
        }),
        Err(e) => {
            tracing::warn!(
                work_item_id = %outcome.work_item_id,
                request_id = %outcome.request_id,
                error = %e,
                "raw_output shape failed; downgrading to error WorkResult"
            );
            ShapeOutcome::Shaped(ItemOutcome {
                disposition: Disposition::PublishErrorAndAck,
                result_msgpack: Vec::new(),
                error: Some(format!("raw_output shape: {e}")),
                error_code: Some("raw_output_shape_error".into()),
                raw_output: None,
                ..outcome.clone()
            })
        }
    }
}

/// Dispatch a [`RawOutput`] to the appropriate shaper. Returns a
/// `RawOutputUnknown` error if no recognised variant is populated;
/// callers turn that into a `PublishErrorAndAck` so unknown future
/// variants surface as a typed error rather than an empty success.
fn shape(raw: &RawOutput) -> Result<Vec<u8>, ShapeError> {
    // Each outcome carries exactly one variant. Dispatch order is
    // fixed so a test that accidentally populates two variants gets
    // a deterministic result: dense → score → sparse → multivector.
    // A legitimate multi-variant encode item (e.g. dense + sparse
    // for the same text) is framed via the Python-framed fallback path —
    // the Python gate refuses to mint a multi-variant `RawOutput`.
    if let Some(dense) = raw.dense.as_ref() {
        let mut values = dense.values.clone();
        if dense.normalize {
            l2_normalize_in_place(&mut values);
        }
        return build_dense_payload(&values, dense.dim as usize);
    }
    if let Some(score) = raw.score.as_ref() {
        return build_score_payload(&score.scores, &score.item_ids);
    }
    if let Some(sparse) = raw.sparse.as_ref() {
        return build_sparse_payload(&sparse.indices, &sparse.values, sparse.dims);
    }
    if let Some(mv) = raw.multivector.as_ref() {
        return build_multivector_payload(&mv.values, mv.num_tokens, mv.token_dims);
    }
    Err(ShapeError::MsgpackWrite(
        "raw_output had no recognised variant populated (forward-compat placeholder?)".into(),
    ))
}

/// Convert an `ItemOutcome` into a `WorkResult` (gateway-facing wire
/// type). `NakRetry` should never reach this path — the caller filters.
/// `timings.is_some()` → queue / processing / payload_fetch are stamped;
/// error-only paths pass `None` and produce a bare error result.
pub fn build_work_result(
    outcome: &ItemOutcome,
    worker_id: &str,
    timings: Option<Timings>,
) -> WorkResult {
    let (success, error, error_code) = match outcome.disposition {
        Disposition::PublishAndAck => (true, None, None),
        Disposition::PublishErrorAndAck => {
            (false, outcome.error.clone(), outcome.error_code.clone())
        }
        Disposition::NakRetry => {
            // Shouldn't be published — if the caller passed one anyway,
            // default to an error WorkResult so the gateway surfaces it.
            (
                false,
                Some(
                    outcome
                        .error
                        .clone()
                        .unwrap_or_else(|| "nak_retry reached publisher".to_string()),
                ),
                outcome.error_code.clone(),
            )
        }
    };

    // Timing fields are only stamped on the success/publish path.
    // Error results omit them entirely. `processing_ms = 0.0` is a
    // placeholder the gateway ignores when inference_ms is set.
    let (queue_ms, processing_ms, payload_fetch_ms) = match timings {
        Some(t) => (
            Some(t.queue_ms),
            Some(0.0),
            (t.payload_fetch_ms > 0.0).then_some(t.payload_fetch_ms),
        ),
        None => (None, None, None),
    };

    WorkResult {
        work_item_id: outcome.work_item_id.clone(),
        request_id: outcome.request_id.clone(),
        item_index: outcome.item_index,
        success,
        result_msgpack: outcome.result_msgpack.clone(),
        error,
        error_code,
        inference_ms: outcome.inference_ms,
        queue_ms,
        processing_ms,
        worker_id: Some(worker_id.to_string()),
        tokenization_ms: outcome.tokenization_ms,
        postprocessing_ms: outcome.postprocessing_ms,
        payload_fetch_ms,
    }
}

/// Decide whether a disposition should be published at all.
pub fn should_publish(disposition: &Disposition) -> bool {
    matches!(
        disposition,
        Disposition::PublishAndAck | Disposition::PublishErrorAndAck
    )
}

/// Reject empty / whitespace-only reply subjects so we never publish to
/// an empty NATS subject (which the client would silently drop).
pub(crate) fn check_reply_subject(subject: &str) -> Result<(), PublishError> {
    if subject.trim().is_empty() {
        return Err(PublishError::EmptyReplySubject);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc_types::{DenseOutput, MultivectorOutput, ScoreOutputRaw, SparseOutput};
    use crate::output::{build_multivector_payload, build_sparse_payload};

    fn outcome() -> ItemOutcome {
        ItemOutcome {
            work_item_id: "req-1.0".into(),
            request_id: "req-1".into(),
            item_index: 0,
            disposition: Disposition::PublishAndAck,
            nak_delay_ms: None,
            result_msgpack: vec![0x81, 0xa5, b'h', b'e', b'l', b'l', b'o', 0x05],
            error: None,
            error_code: None,
            inference_ms: Some(17.5),
            tokenization_ms: Some(1.1),
            postprocessing_ms: Some(0.3),
            raw_output: None,
        }
    }

    #[test]
    fn success_outcome_maps_to_success_result() {
        let r = build_work_result(&outcome(), "worker-1", None);
        assert!(r.success);
        assert!(r.error.is_none());
        assert_eq!(r.request_id, "req-1");
        assert_eq!(r.item_index, 0);
        assert_eq!(r.worker_id.as_deref(), Some("worker-1"));
        assert_eq!(r.inference_ms, Some(17.5));
        assert_eq!(r.tokenization_ms, Some(1.1));
        assert_eq!(r.postprocessing_ms, Some(0.3));
        assert_eq!(r.result_msgpack.len(), 8);
    }

    #[test]
    fn publish_error_outcome_maps_to_failure_result() {
        let mut o = outcome();
        o.disposition = Disposition::PublishErrorAndAck;
        o.error = Some("kapow".into());
        o.error_code = Some("INTERNAL".into());
        let r = build_work_result(&o, "w", None);
        assert!(!r.success);
        assert_eq!(r.error.as_deref(), Some("kapow"));
        assert_eq!(r.error_code.as_deref(), Some("INTERNAL"));
    }

    #[test]
    fn should_publish_rejects_nak_retry() {
        assert!(should_publish(&Disposition::PublishAndAck));
        assert!(should_publish(&Disposition::PublishErrorAndAck));
        assert!(!should_publish(&Disposition::NakRetry));
    }

    #[test]
    fn work_result_is_msgpack_round_trippable() {
        let r = build_work_result(&outcome(), "w", None);
        let bytes = rmp_serde::to_vec_named(&r).unwrap();
        let back: WorkResult = rmp_serde::from_slice(&bytes).unwrap();
        assert!(back.success);
        assert_eq!(back.request_id, "req-1");
        assert_eq!(back.result_msgpack.len(), 8);
    }

    #[test]
    fn timings_populate_queue_and_processing_fields() {
        let timings = Timings {
            queue_ms: 12.5,
            payload_fetch_ms: 3.2,
        };
        let r = build_work_result(&outcome(), "w", Some(timings));
        assert_eq!(r.queue_ms, Some(12.5));
        // Python sets processing_ms = 0.0 as a placeholder; we match exactly.
        assert_eq!(r.processing_ms, Some(0.0));
        assert_eq!(r.payload_fetch_ms, Some(3.2));
    }

    #[test]
    fn zero_payload_fetch_ms_is_omitted() {
        // Matches Python: `if payload_fetch_ms > 0:` before inclusion.
        let timings = Timings {
            queue_ms: 5.0,
            payload_fetch_ms: 0.0,
        };
        let r = build_work_result(&outcome(), "w", Some(timings));
        assert_eq!(r.queue_ms, Some(5.0));
        assert!(r.payload_fetch_ms.is_none());
    }

    #[test]
    fn check_reply_subject_rejects_empty_and_whitespace() {
        assert!(matches!(
            check_reply_subject(""),
            Err(PublishError::EmptyReplySubject)
        ));
        assert!(matches!(
            check_reply_subject("   "),
            Err(PublishError::EmptyReplySubject)
        ));
        assert!(check_reply_subject("\t\n").is_err());
    }

    #[test]
    fn check_reply_subject_accepts_normal_inbox() {
        assert!(check_reply_subject("_INBOX.r1.abc").is_ok());
        assert!(check_reply_subject("sie.results.q1").is_ok());
    }

    #[test]
    fn error_path_omits_timings_fields() {
        // Error-only publishes (payload resolution failure, unknown op) go
        // through the `None` timings path and should not carry queue_ms etc.
        let mut o = outcome();
        o.disposition = Disposition::PublishErrorAndAck;
        o.error = Some("oops".into());
        o.error_code = Some("payload_error".into());
        let r = build_work_result(&o, "w", None);
        assert!(!r.success);
        assert!(r.queue_ms.is_none());
        assert!(r.processing_ms.is_none());
        assert!(r.payload_fetch_ms.is_none());
    }

    // ---- raw_output shaping ---------------------------------------

    #[test]
    fn raw_output_none_leaves_outcome_unchanged() {
        let o = outcome();
        // `outcome()` pre-fills legacy `result_msgpack` and no
        // `raw_output` — the shaper is a no-op.
        match shape_raw_output_for_wire(&o) {
            ShapeOutcome::Unchanged => {}
            ShapeOutcome::Shaped(_) => {
                panic!("raw_output shaper must not touch legacy outcomes that already have result_msgpack");
            }
        }
    }

    #[test]
    fn raw_output_dense_is_shaped_into_msgpack() {
        let mut o = outcome();
        o.result_msgpack = Vec::new();
        o.raw_output = Some(RawOutput {
            dense: Some(DenseOutput {
                values: vec![1.0, 2.0, 3.0],
                dim: 3,
                normalize: false,
            }),
            ..Default::default()
        });
        let shaped = match shape_raw_output_for_wire(&o) {
            ShapeOutcome::Shaped(s) => s,
            ShapeOutcome::Unchanged => panic!("expected raw_output shaping"),
        };
        // Byte-for-byte identical to the Python
        // `_wrap_encode_output({"dense": np.float32([1,2,3])})` golden.
        let expected = build_dense_payload(&[1.0, 2.0, 3.0], 3).unwrap();
        assert_eq!(shaped.result_msgpack, expected);
        assert!(matches!(shaped.disposition, Disposition::PublishAndAck));
        assert!(
            shaped.raw_output.is_none(),
            "shaper must strip raw_output to avoid double-encoding"
        );
        assert_eq!(shaped.work_item_id, "req-1.0");
        assert_eq!(
            shaped.inference_ms,
            Some(17.5),
            "timing metadata must round-trip"
        );
    }

    #[test]
    fn raw_output_dense_with_normalize_l2s_before_packing() {
        let mut o = outcome();
        o.result_msgpack = Vec::new();
        o.raw_output = Some(RawOutput {
            dense: Some(DenseOutput {
                values: vec![3.0, 4.0],
                dim: 2,
                normalize: true,
            }),
            ..Default::default()
        });
        let shaped = match shape_raw_output_for_wire(&o) {
            ShapeOutcome::Shaped(s) => s,
            _ => unreachable!(),
        };
        // [3,4] L2-normalised is [0.6, 0.8].
        let expected = build_dense_payload(&[0.6, 0.8], 2).unwrap();
        assert_eq!(shaped.result_msgpack, expected);
    }

    #[test]
    fn raw_output_score_is_shaped_and_sorted() {
        let mut o = outcome();
        o.result_msgpack = Vec::new();
        o.raw_output = Some(RawOutput {
            score: Some(ScoreOutputRaw {
                scores: vec![0.5, 0.9],
                item_ids: vec!["b".into(), "a".into()],
            }),
            ..Default::default()
        });
        let shaped = match shape_raw_output_for_wire(&o) {
            ShapeOutcome::Shaped(s) => s,
            _ => unreachable!(),
        };
        // Rust shaper sorts desc: a(0.9,rank=0), b(0.5,rank=1).
        let expected =
            build_score_payload(&[0.5, 0.9], &["b".to_string(), "a".to_string()]).unwrap();
        assert_eq!(shaped.result_msgpack, expected);
    }

    #[test]
    fn raw_output_shape_error_becomes_error_outcome() {
        // Mismatched dim → shaper returns ShapeError, publisher
        // must NOT panic; it downgrades to PublishErrorAndAck with
        // a typed error code so the gateway surfaces the incident.
        let mut o = outcome();
        o.result_msgpack = Vec::new();
        o.raw_output = Some(RawOutput {
            dense: Some(DenseOutput {
                values: vec![1.0, 2.0],
                dim: 3,
                normalize: false,
            }),
            ..Default::default()
        });
        let shaped = match shape_raw_output_for_wire(&o) {
            ShapeOutcome::Shaped(s) => s,
            _ => unreachable!(),
        };
        assert!(matches!(
            shaped.disposition,
            Disposition::PublishErrorAndAck
        ));
        assert!(shaped.result_msgpack.is_empty());
        assert_eq!(shaped.error_code.as_deref(), Some("raw_output_shape_error"));
        assert!(shaped
            .error
            .as_deref()
            .unwrap_or("")
            .contains("raw_output shape"));
    }

    #[test]
    fn raw_output_with_preexisting_msgpack_is_not_reshaped() {
        // Mixed wire: Python sent both legacy and raw_output.
        // Trust the already-framed bytes — shaper treats this as
        // a rolling-deploy wire artefact.
        let mut o = outcome(); // legacy result_msgpack populated
        o.raw_output = Some(RawOutput {
            dense: Some(DenseOutput {
                values: vec![9.0, 9.0, 9.0],
                dim: 3,
                normalize: false,
            }),
            ..Default::default()
        });
        match shape_raw_output_for_wire(&o) {
            ShapeOutcome::Unchanged => {}
            ShapeOutcome::Shaped(_) => panic!("must prefer pre-framed result_msgpack"),
        }
    }

    #[test]
    fn raw_output_on_error_disposition_is_ignored() {
        // Defensive: Python shouldn't attach raw_output to an
        // error disposition, but if it does we don't shape.
        let mut o = outcome();
        o.disposition = Disposition::PublishErrorAndAck;
        o.result_msgpack = Vec::new();
        o.error = Some("upstream".into());
        o.raw_output = Some(RawOutput {
            dense: Some(DenseOutput {
                values: vec![1.0],
                dim: 1,
                normalize: false,
            }),
            ..Default::default()
        });
        match shape_raw_output_for_wire(&o) {
            ShapeOutcome::Unchanged => {}
            ShapeOutcome::Shaped(_) => {
                panic!("must not run shaper on error outcomes; raw_output should be ignored")
            }
        }
    }

    #[test]
    fn raw_output_sparse_is_shaped_into_msgpack() {
        // End-to-end publisher hook for sparse: the Python side
        // sends only the typed `SparseOutput`; the publisher must
        // pack exactly the bytes the legacy `_wrap_encode_output`
        // path produces so SDK decoders see zero change.
        let mut o = outcome();
        o.result_msgpack = Vec::new();
        o.raw_output = Some(RawOutput {
            sparse: Some(SparseOutput {
                indices: vec![3, 7, 42],
                values: vec![0.5, 1.5, 2.5],
                dims: Some(30522),
            }),
            ..Default::default()
        });
        let shaped = match shape_raw_output_for_wire(&o) {
            ShapeOutcome::Shaped(s) => s,
            ShapeOutcome::Unchanged => panic!("expected raw_output sparse shaping"),
        };
        let expected = build_sparse_payload(&[3, 7, 42], &[0.5, 1.5, 2.5], Some(30522)).unwrap();
        assert_eq!(shaped.result_msgpack, expected);
        assert!(shaped.raw_output.is_none());
        assert!(matches!(shaped.disposition, Disposition::PublishAndAck));
    }

    #[test]
    fn raw_output_sparse_length_mismatch_becomes_error() {
        // indices/values length parity is a hard invariant; if the
        // Python gate somehow lets a mismatched payload through
        // the publisher must surface a typed error rather than
        // crash or publish garbage.
        let mut o = outcome();
        o.result_msgpack = Vec::new();
        o.raw_output = Some(RawOutput {
            sparse: Some(SparseOutput {
                indices: vec![1, 2, 3],
                values: vec![0.5], // mismatched
                dims: None,
            }),
            ..Default::default()
        });
        let shaped = match shape_raw_output_for_wire(&o) {
            ShapeOutcome::Shaped(s) => s,
            _ => unreachable!(),
        };
        assert!(matches!(
            shaped.disposition,
            Disposition::PublishErrorAndAck
        ));
        assert_eq!(shaped.error_code.as_deref(), Some("raw_output_shape_error"));
    }

    #[test]
    fn raw_output_multivector_is_shaped_into_msgpack() {
        let mut o = outcome();
        o.result_msgpack = Vec::new();
        let values: Vec<f32> = (0..12).map(|i| i as f32).collect();
        o.raw_output = Some(RawOutput {
            multivector: Some(MultivectorOutput {
                values: values.clone(),
                num_tokens: 3,
                token_dims: 4,
            }),
            ..Default::default()
        });
        let shaped = match shape_raw_output_for_wire(&o) {
            ShapeOutcome::Shaped(s) => s,
            ShapeOutcome::Unchanged => panic!("expected raw_output multivector shaping"),
        };
        let expected = build_multivector_payload(&values, 3, 4).unwrap();
        assert_eq!(shaped.result_msgpack, expected);
    }

    #[test]
    fn raw_output_empty_variant_becomes_error_outcome() {
        // Forward-compat: a future Rust build sees RawOutput with
        // no recognised variant populated (all None). Must surface
        // as a typed error, not a silent empty success.
        let mut o = outcome();
        o.result_msgpack = Vec::new();
        o.raw_output = Some(RawOutput::default());
        let shaped = match shape_raw_output_for_wire(&o) {
            ShapeOutcome::Shaped(s) => s,
            _ => unreachable!(),
        };
        assert!(matches!(
            shaped.disposition,
            Disposition::PublishErrorAndAck
        ));
        assert_eq!(shaped.error_code.as_deref(), Some("raw_output_shape_error"));
    }
}
