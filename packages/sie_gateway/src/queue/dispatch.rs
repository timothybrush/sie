//! Transport-neutral dispatch seam between the HTTP handlers and the
//! queue publisher (roadmap P2.1, design §4.2/§13).
//!
//! [`WorkDispatcher`] captures the exact handler-facing surface of
//! [`WorkPublisher`] — publish, cancel, republish, and pending-state
//! snapshots — so handlers depend on `Arc<dyn WorkDispatcher>` rather
//! than the concrete NATS-backed publisher. NATS stays the only
//! implementation; lifecycle methods (inbox subscription, backpressure
//! monitor, cleanup, drain) deliberately stay on the concrete type
//! owned by `main.rs`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tokio::sync::{broadcast, oneshot};
use tracing::warn;

use super::publisher::WorkPublisher;
// Re-exports for future consumers of the dispatch seam (also serve as
// this module's own imports for the trait signatures below).
pub use super::publisher::{PendingGenerationSnapshot, PublishTarget, WorkParams, WorkResult};
pub use super::streaming::{ChunkEnvelope, StreamOutcome};

/// Handler→publisher dispatch boundary. Signatures mirror the inherent
/// [`WorkPublisher`] methods exactly; see the inherent methods for the
/// full behavioural documentation.
#[async_trait]
pub trait WorkDispatcher: Send + Sync {
    #[allow(clippy::too_many_arguments)]
    async fn publish_work(
        &self,
        target: PublishTarget,
        admission_pool: &str,
        endpoint: &str,
        model: &str,
        engine: &str,
        bundle_config_hash: &str,
        items: Vec<rmpv::Value>,
        params: &WorkParams,
    ) -> Result<(String, oneshot::Receiver<Vec<WorkResult>>), String>;

    #[allow(clippy::too_many_arguments)]
    async fn publish_generate_streaming(
        &self,
        target: PublishTarget,
        display_model: &str,
        engine: &str,
        bundle_config_hash: &str,
        params: &WorkParams,
        admission_pool: &str,
    ) -> Result<
        (
            String,
            oneshot::Receiver<StreamOutcome>,
            Arc<tokio::sync::Notify>,
        ),
        String,
    >;

    #[allow(clippy::too_many_arguments)]
    async fn publish_generate_streaming_sse(
        &self,
        target: PublishTarget,
        display_model: &str,
        engine: &str,
        bundle_config_hash: &str,
        params: &WorkParams,
        admission_pool: &str,
    ) -> Result<
        (
            String,
            oneshot::Receiver<StreamOutcome>,
            broadcast::Receiver<ChunkEnvelope>,
        ),
        String,
    >;

    async fn publish_cancel(&self, request_id: &str);

    async fn republish_to_pool(
        &self,
        request_id: &str,
        reason: &'static str,
    ) -> Result<bool, String>;

    async fn republish_pending_result_to_pool(
        &self,
        request_id: &str,
        reason: &'static str,
    ) -> Result<bool, String>;

    fn drop_pending_stream(&self, request_id: &str);

    fn pending_generation_snapshot(&self) -> PendingGenerationSnapshot;

    fn pending_generation_for_model(&self, model_id: &str) -> PendingGenerationSnapshot;

    fn stream_observed_first_chunk(&self, request_id: &str) -> bool;

    fn stream_chunk_timing(&self, request_id: &str) -> Option<(Option<Instant>, Option<Instant>)>;
}

#[async_trait]
impl WorkDispatcher for WorkPublisher {
    async fn publish_work(
        &self,
        target: PublishTarget,
        admission_pool: &str,
        endpoint: &str,
        model: &str,
        engine: &str,
        bundle_config_hash: &str,
        items: Vec<rmpv::Value>,
        params: &WorkParams,
    ) -> Result<(String, oneshot::Receiver<Vec<WorkResult>>), String> {
        WorkPublisher::publish_work(
            self,
            target,
            admission_pool,
            endpoint,
            model,
            engine,
            bundle_config_hash,
            items,
            params,
        )
        .await
    }

    async fn publish_generate_streaming(
        &self,
        target: PublishTarget,
        display_model: &str,
        engine: &str,
        bundle_config_hash: &str,
        params: &WorkParams,
        admission_pool: &str,
    ) -> Result<
        (
            String,
            oneshot::Receiver<StreamOutcome>,
            Arc<tokio::sync::Notify>,
        ),
        String,
    > {
        WorkPublisher::publish_generate_streaming(
            self,
            target,
            display_model,
            engine,
            bundle_config_hash,
            params,
            admission_pool,
        )
        .await
    }

    async fn publish_generate_streaming_sse(
        &self,
        target: PublishTarget,
        display_model: &str,
        engine: &str,
        bundle_config_hash: &str,
        params: &WorkParams,
        admission_pool: &str,
    ) -> Result<
        (
            String,
            oneshot::Receiver<StreamOutcome>,
            broadcast::Receiver<ChunkEnvelope>,
        ),
        String,
    > {
        WorkPublisher::publish_generate_streaming_sse(
            self,
            target,
            display_model,
            engine,
            bundle_config_hash,
            params,
            admission_pool,
        )
        .await
    }

    async fn publish_cancel(&self, request_id: &str) {
        WorkPublisher::publish_cancel(self, request_id).await
    }

    async fn republish_to_pool(
        &self,
        request_id: &str,
        reason: &'static str,
    ) -> Result<bool, String> {
        WorkPublisher::republish_to_pool(self, request_id, reason).await
    }

    async fn republish_pending_result_to_pool(
        &self,
        request_id: &str,
        reason: &'static str,
    ) -> Result<bool, String> {
        WorkPublisher::republish_pending_result_to_pool(self, request_id, reason).await
    }

    fn drop_pending_stream(&self, request_id: &str) {
        WorkPublisher::drop_pending_stream(self, request_id)
    }

    fn pending_generation_snapshot(&self) -> PendingGenerationSnapshot {
        WorkPublisher::pending_generation_snapshot(self)
    }

    fn pending_generation_for_model(&self, model_id: &str) -> PendingGenerationSnapshot {
        WorkPublisher::pending_generation_for_model(self, model_id)
    }

    fn stream_observed_first_chunk(&self, request_id: &str) -> bool {
        WorkPublisher::stream_observed_first_chunk(self, request_id)
    }

    fn stream_chunk_timing(&self, request_id: &str) -> Option<(Option<Instant>, Option<Instant>)> {
        WorkPublisher::stream_chunk_timing(self, request_id)
    }
}

/// Extension methods that need an owned `Arc<dyn WorkDispatcher>` handle
/// (they spawn detached tasks holding a clone), so they cannot live on
/// the object-safe [`WorkDispatcher`] trait itself.
pub trait WorkDispatcherExt {
    /// Arm a one-shot recovery for non-streaming worker-direct batch work.
    ///
    /// Capped logical pools direct-dispatch encode/score/extract to one
    /// admitted worker so unassigned peers do not burn the JetStream
    /// delivery budget. If that worker or its private consumer disappears
    /// after the gateway publishes, the worker-specific stream cannot fail
    /// over by itself. This timer republishes any still-missing items to the
    /// lane's pool subject once; late duplicate results are harmless because
    /// the result collector removes itself on first complete response.
    fn spawn_batch_direct_fallback(&self, request_id: String, delay: Duration);
}

impl WorkDispatcherExt for Arc<dyn WorkDispatcher> {
    fn spawn_batch_direct_fallback(&self, request_id: String, delay: Duration) {
        let publisher = Arc::clone(self);
        tokio::spawn(async move {
            tokio::time::sleep(delay).await;
            if let Err(e) = publisher
                .republish_pending_result_to_pool(&request_id, "batch_direct_timeout")
                .await
            {
                warn!(
                    request_id = %request_id,
                    error = %e,
                    "batch direct-dispatch fallback failed"
                );
            }
        });
    }
}
