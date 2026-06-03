//! Adapter-local error taxonomy.
//!
//! Mirrors the variant set of `sie_server_sidecar::backend::BackendError`
//! at the wire level so we can map straight onto the same per-method
//! response semantics in the IPC server (P2). The two enums are
//! deliberately separate types: the sidecar's `BackendError` carries
//! backend-selection fall-through semantics (see `is_retryable`) that don't
//! apply on the adapter side, and we don't want to drag the
//! `async_trait` `InferenceBackend` plumbing into this crate just to
//! reuse one enum.
//!
//! Variant semantics:
//! * [`BackendError::Transient`] — transport-level failure (file
//!   I/O, GPU OOM that may resolve with a retry). The IPC server
//!   surfaces this as a non-`ok` `ResponseEnvelope`; the sidecar
//!   NAKs the JetStream group with the base delay.
//! * [`BackendError::Inference`] — batch-level forward-pass failure.
//!   Reserved; new code paths should fan errors out per-item inside
//!   `Ok(BatchOutcome)` instead.
//! * [`BackendError::UnsupportedModel`] — model_id not in the engine
//!   registry, or wrong op for an encoder-only engine. The sidecar
//!   uses this as the backend-selection fall-through trigger.
//! * [`BackendError::Draining`] — adapter is shutting down. NAK
//!   fast.

#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("transient: {0}")]
    Transient(String),

    #[error("inference: {0}")]
    Inference(String),

    #[error("unsupported model: {0}")]
    UnsupportedModel(String),

    #[error("draining")]
    Draining,
}
