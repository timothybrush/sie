//! NATS JetStream pull consumer.
//!
//! This module owns the pool-level stream and durable pull consumer
//! matching the Python naming contract (see
//! `config::WorkerConfig::{stream_name,consumer_name,subject_filter}`),
//! then exposes a long-lived pull stream for the dispatcher. ACK/NAK,
//! batching, and adaptive fetch timing live above this layer.

use std::time::Duration;

use async_nats::jetstream::consumer::pull::{Config as ConsumerConfig, Stream as PullStream};
use async_nats::jetstream::consumer::{AckPolicy, PullConsumer};
use async_nats::jetstream::stream::{
    Config as StreamConfig, DiscardPolicy, RetentionPolicy, StorageType, Stream as JsStream,
};
use async_nats::jetstream::Context as JsContext;
use futures_util::TryStreamExt;
use thiserror::Error;
use tracing::{debug, info, warn};

use crate::config::WorkerConfig;
use crate::subject::subjects_overlap;

/// Default consumer parameters. Must match the gateway's stream creator
/// and any other worker in the pool — when multiple creators share a
/// durable consumer by name, the first to create it wins, and subsequent
/// callers silently inherit whatever was there.
const ACK_WAIT_SECS: u64 = 30;
const GENERATION_ACK_WAIT_SECS: u64 = 300;
const DEFAULT_MAX_DELIVER: i64 = 20;
const DEFAULT_MAX_ACK_PENDING: i64 = 1000;
const DEFAULT_STREAM_MAX_AGE_SECS: u64 = ACK_WAIT_SECS * (DEFAULT_MAX_DELIVER as u64);
const STREAM_MAX_MSGS: i64 = 100_000;

/// Env-overridable `max_deliver`. Mirrors Python's `SIE_MAX_DELIVER`
/// (default 20). With the default 30s ACK wait, this gives a 600s retry
/// envelope before a message hits the DLQ.
fn max_deliver() -> i64 {
    std::env::var("SIE_MAX_DELIVER")
        .ok()
        .and_then(|s| s.parse::<i64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_MAX_DELIVER)
}

/// Env-overridable stream `max_age`. Mirrors Python's
/// `SIE_STREAM_MAX_AGE_S` (default 600). Should be >=
/// `max_deliver * ack_wait` so messages don't expire mid-retry.
fn stream_max_age_secs() -> u64 {
    std::env::var("SIE_STREAM_MAX_AGE_S")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_STREAM_MAX_AGE_SECS)
}

/// Env-overridable `max_ack_pending`. Python hardcodes 1000 for now but
/// exposing the knob keeps the option open without a code change.
fn max_ack_pending() -> i64 {
    std::env::var("SIE_MAX_ACK_PENDING")
        .ok()
        .and_then(|s| s.parse::<i64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_MAX_ACK_PENDING)
}

fn generation_stream_max_age_secs() -> u64 {
    stream_max_age_secs().max(GENERATION_ACK_WAIT_SECS * max_deliver() as u64)
}

/// Opt-in: when `SIE_NATS_CONSERVATIVE_CLEANUP=1`, the stale-durable
/// sweep only deletes consumers that have **zero** pull waiters
/// (`num_waiting == 0`) and **zero** in-flight acks
/// (`num_ack_pending == 0`). Default behaviour (env unset / `0`)
/// keeps the historical aggressive delete because today's topology is
/// one bundle per pool — flipping bundles on a pool must clobber the
/// old durable for the new worker to bind.
///
/// Flip this on if you ever introduce a topology where two bundles
/// legitimately share a pool stream with overlapping filters; we
/// then refuse to delete a healthy peer consumer and let the
/// authoritative `get_or_create_consumer` error surface so the
/// operator can resolve the conflict explicitly.
fn conservative_cleanup() -> bool {
    std::env::var("SIE_NATS_CONSERVATIVE_CLEANUP")
        .ok()
        .filter(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .is_some()
}

#[derive(Debug, Error)]
pub enum NatsSetupError {
    #[error("ensure stream {name}: {source}")]
    EnsureStream {
        name: String,
        #[source]
        source: async_nats::Error,
    },
    #[error("ensure consumer {name}: {source}")]
    EnsureConsumer {
        name: String,
        #[source]
        source: async_nats::Error,
    },
    #[error("connect NATS: {source}")]
    Connect {
        #[source]
        source: async_nats::ConnectError,
    },
}

/// Connect to NATS and return a JetStream context.
pub async fn connect(nats_url: &str) -> Result<(async_nats::Client, JsContext), NatsSetupError> {
    let client = async_nats::connect(nats_url)
        .await
        .map_err(|source| NatsSetupError::Connect { source })?;
    let js = async_nats::jetstream::new(client.clone());
    Ok((client, js))
}

/// Predicate used by [`cleanup_overlapping_consumers`]. Returns true
/// iff any of the consumer's configured filter(s) overlaps the
/// worker's intended `desired_subject` per NATS wildcard semantics.
///
/// Split out from `Info` so it can be unit-tested without constructing
/// the (large, time-stamped) `Info` value.
fn any_filter_overlaps(
    primary_filter: &str,
    multi_filters: &[String],
    desired_subject: &str,
) -> bool {
    if !primary_filter.is_empty() && subjects_overlap(primary_filter, desired_subject) {
        return true;
    }
    multi_filters
        .iter()
        .any(|s| subjects_overlap(s, desired_subject))
}

fn is_legacy_per_model_work_stream(stream_name: &str) -> bool {
    stream_name.starts_with("WORK_") && !stream_name.starts_with("WORK_POOL_")
}

fn legacy_stream_overlaps_pool(
    stream_name: &str,
    subjects: &[String],
    desired_stream_name: &str,
    desired_subject: &str,
) -> bool {
    stream_name != desired_stream_name
        && is_legacy_per_model_work_stream(stream_name)
        && subjects
            .iter()
            .any(|subject| subjects_overlap(subject, desired_subject))
}

fn legacy_stream_is_safe_to_delete(
    stream_name: &str,
    subjects: &[String],
    desired_stream_name: &str,
    desired_subject: &str,
    message_count: u64,
) -> bool {
    message_count == 0
        && legacy_stream_overlaps_pool(stream_name, subjects, desired_stream_name, desired_subject)
}

/// Best-effort: delete empty old per-model `WORK_<model>` streams whose
/// subjects overlap the pool stream we are about to create.
///
/// Older Python adapter processes created one WorkQueue stream per model. The Rust
/// sidecar creates one pool stream (`WORK_POOL_<pool>`) for
/// `sie.work.*.<pool>`. NATS forbids overlapping WorkQueue streams, so a
/// leftover per-model stream can prevent the new pool stream from being
/// created. We only delete empty streams that match the old naming convention
/// and overlap this pool's subject filter; non-empty streams must be drained
/// or migrated explicitly so startup never drops queued work.
async fn cleanup_legacy_per_model_streams(
    js: &JsContext,
    desired_stream_name: &str,
    desired_subject: &str,
) {
    let mut listing = js.streams();
    let mut stale = Vec::new();

    loop {
        match listing.try_next().await {
            Ok(Some(info)) => {
                if legacy_stream_overlaps_pool(
                    &info.config.name,
                    &info.config.subjects,
                    desired_stream_name,
                    desired_subject,
                ) {
                    if legacy_stream_is_safe_to_delete(
                        &info.config.name,
                        &info.config.subjects,
                        desired_stream_name,
                        desired_subject,
                        info.state.messages,
                    ) {
                        stale.push((info.config.name, info.config.subjects));
                    } else {
                        warn!(
                            stale_stream = %info.config.name,
                            desired_stream = %desired_stream_name,
                            desired_subject,
                            messages = info.state.messages,
                            subjects = ?info.config.subjects,
                            "legacy per-model stream overlaps the pool stream but still has \
                             queued messages; drain or migrate it before starting this sidecar"
                        );
                    }
                }
            }
            Ok(None) => break,
            Err(e) => {
                warn!(
                    error = %e,
                    desired_stream = %desired_stream_name,
                    desired_subject,
                    "listing streams failed during legacy per-model stream cleanup; \
                     proceeding to pool stream creation"
                );
                return;
            }
        }
    }

    let mut deleted = 0usize;
    for (stream_name, subjects) in stale {
        warn!(
            stale_stream = %stream_name,
            desired_stream = %desired_stream_name,
            desired_subject,
            subjects = ?subjects,
            "deleting legacy per-model stream that overlaps the pool stream"
        );
        match js.delete_stream(&stream_name).await {
            Ok(_) => deleted += 1,
            Err(e) => warn!(
                stale_stream = %stream_name,
                error = %e,
                "failed to delete legacy per-model stream; pool stream creation may fail"
            ),
        }
    }

    if deleted > 0 {
        info!(
            desired_stream = %desired_stream_name,
            desired_subject,
            deleted,
            "deleted legacy per-model streams before pool stream creation"
        );
    }
}

/// Best-effort: remove any durable consumer on `stream` whose name is
/// not `desired_name` but whose filter subject(s) overlap
/// `desired_subject`. NATS WorkQueue forbids overlapping filters across
/// consumers on the same stream — without this, the worker can't bind
/// after a bundle/engine flip on a shared pool.
///
/// Failures (list errors, individual deletes) are logged but never
/// returned: we'd rather let `get_or_create_consumer` produce the
/// authoritative error a moment later than block startup on a transient
/// listing hiccup. In practice the only scenarios where listing fails
/// are also scenarios where the consumer create will fail loudly.
async fn cleanup_overlapping_consumers(
    stream: &JsStream,
    desired_name: &str,
    desired_subject: &str,
) {
    let conservative = conservative_cleanup();
    let mut listing = stream.consumers();
    let mut stale: Vec<(String, i64, i64)> = Vec::new();
    loop {
        match listing.try_next().await {
            Ok(Some(info)) => {
                if info.name == desired_name {
                    continue;
                }
                if any_filter_overlaps(
                    &info.config.filter_subject,
                    &info.config.filter_subjects,
                    desired_subject,
                ) {
                    stale.push((
                        info.name,
                        info.num_waiting as i64,
                        info.num_ack_pending as i64,
                    ));
                }
            }
            Ok(None) => break,
            Err(e) => {
                warn!(
                    error = %e,
                    desired_subject,
                    "listing consumers failed during stale-durable sweep; \
                     proceeding to consumer-create which will surface the \
                     authoritative error if a true overlap remains"
                );
                return;
            }
        }
    }

    for (name, num_waiting, num_ack_pending) in stale {
        if conservative && (num_waiting > 0 || num_ack_pending > 0) {
            warn!(
                stream = %stream.cached_info().config.name,
                stale_consumer = %name,
                desired_consumer = %desired_name,
                desired_subject,
                num_waiting,
                num_ack_pending,
                "skipping stale-durable delete: peer consumer looks active and \
                 SIE_NATS_CONSERVATIVE_CLEANUP=1; consumer-create will surface \
                 the overlap error and the operator must resolve it manually"
            );
            continue;
        }
        warn!(
            stream = %stream.cached_info().config.name,
            stale_consumer = %name,
            desired_consumer = %desired_name,
            desired_subject,
            num_waiting,
            num_ack_pending,
            conservative,
            "deleting stale durable that overlaps this worker's filter \
             (left over from a prior bundle/engine deploy on this pool)"
        );
        if let Err(e) = stream.delete_consumer(&name).await {
            warn!(
                stale_consumer = %name,
                error = %e,
                "failed to delete stale durable; consumer-create will likely fail next"
            );
        }
    }
}

/// Ensure the pool stream and the durable pull consumer exist. Called once
/// per worker startup; safe to call again for slow stream/durable reconcile.
pub async fn ensure_stream_and_consumer(
    js: &JsContext,
    config: &WorkerConfig,
) -> Result<PullConsumer, NatsSetupError> {
    ensure_stream_and_consumer_inner(js, config, true).await
}

/// Re-ensure the pool stream and durable consumer without rebuilding the
/// active pull stream. This is intentionally a slow control-plane repair path:
/// the hot fetch loop still owns pull-stream recovery on terminal stream errors.
pub async fn reconcile_stream_and_consumer(
    js: &JsContext,
    config: &WorkerConfig,
) -> Result<(), NatsSetupError> {
    ensure_stream_and_consumer_inner(js, config, false)
        .await
        .map(|_| ())
}

/// Ensure the worker-specific stream used by generation direct-dispatch.
///
/// The gateway publishes direct generation work to
/// `sie.work.{model}.{pool}.{worker_id}`. That subject intentionally does
/// not match the pool stream (`sie.work.*.{pool}`), so the sidecar must
/// bind this second stream for generation to avoid relying solely on
/// first-chunk fallback republish to the pool.
pub async fn ensure_worker_stream_and_consumer(
    js: &JsContext,
    config: &WorkerConfig,
) -> Result<PullConsumer, NatsSetupError> {
    let stream_name = config.worker_stream_name();
    let subject = config.worker_subject_filter();
    let consumer_name = config.worker_consumer_name();

    let stream_cfg = StreamConfig {
        name: stream_name.clone(),
        subjects: vec![subject.clone()],
        retention: RetentionPolicy::WorkQueue,
        storage: StorageType::Memory,
        max_age: Duration::from_secs(generation_stream_max_age_secs()),
        max_messages: STREAM_MAX_MSGS,
        num_replicas: 1,
        discard: DiscardPolicy::New,
        ..Default::default()
    };

    let desired_ack_wait = Duration::from_secs(GENERATION_ACK_WAIT_SECS);
    let desired_max_deliver = max_deliver();
    let desired_max_ack_pending = max_ack_pending();
    let stream =
        js.get_or_create_stream(stream_cfg)
            .await
            .map_err(|e| NatsSetupError::EnsureStream {
                name: stream_name.clone(),
                source: e.into(),
            })?;

    cleanup_overlapping_consumers(&stream, &consumer_name, &subject).await;

    let consumer_cfg = ConsumerConfig {
        durable_name: Some(consumer_name.clone()),
        filter_subject: subject.clone(),
        ack_policy: AckPolicy::Explicit,
        ack_wait: desired_ack_wait,
        max_deliver: desired_max_deliver,
        max_ack_pending: desired_max_ack_pending,
        ..Default::default()
    };
    let consumer: PullConsumer = stream
        .get_or_create_consumer(&consumer_name, consumer_cfg)
        .await
        .map_err(|e| NatsSetupError::EnsureConsumer {
            name: consumer_name.clone(),
            source: e.into(),
        })?;
    info!(
        stream = %stream_name,
        consumer = %consumer_name,
        subject = %subject,
        ack_wait_s = GENERATION_ACK_WAIT_SECS,
        "ensured generation direct-dispatch stream and pull consumer"
    );
    Ok(consumer)
}

pub async fn reconcile_worker_stream_and_consumer(
    js: &JsContext,
    config: &WorkerConfig,
) -> Result<(), NatsSetupError> {
    ensure_worker_stream_and_consumer(js, config)
        .await
        .map(|_| ())
}

async fn ensure_stream_and_consumer_inner(
    js: &JsContext,
    config: &WorkerConfig,
    log_success_at_info: bool,
) -> Result<PullConsumer, NatsSetupError> {
    let stream_name = config.stream_name();
    let subject = config.subject_filter();
    let consumer_name = config.consumer_name();

    let stream_cfg = StreamConfig {
        name: stream_name.clone(),
        subjects: vec![subject.clone()],
        retention: RetentionPolicy::WorkQueue,
        storage: StorageType::Memory,
        max_age: Duration::from_secs(stream_max_age_secs()),
        max_messages: STREAM_MAX_MSGS,
        num_replicas: 1,
        discard: DiscardPolicy::New,
        ..Default::default()
    };

    let desired_max_age = stream_cfg.max_age;
    let desired_max_deliver = max_deliver();
    let desired_max_ack_pending = max_ack_pending();
    let desired_ack_wait = Duration::from_secs(ACK_WAIT_SECS);

    cleanup_legacy_per_model_streams(js, &stream_name, &subject).await;

    let stream =
        js.get_or_create_stream(stream_cfg)
            .await
            .map_err(|e| NatsSetupError::EnsureStream {
                name: stream_name.clone(),
                source: e.into(),
            })?;

    // `get_or_create_stream` is a no-op against an EXISTING stream with a
    // different config (the first writer's config wins). If the gateway
    // and the worker disagree on `max_age`, retries may be purged before
    // `max_deliver` is reached — silent data loss. Warn so ops can tell.
    let observed_max_age = stream.cached_info().config.max_age;
    if observed_max_age != desired_max_age {
        warn!(
            stream = %stream_name,
            observed_max_age_s = observed_max_age.as_secs(),
            desired_max_age_s = desired_max_age.as_secs(),
            "stream max_age drift: existing stream config disagrees with this worker's intent; \
             whoever created the stream first wins. This can cause retries to expire before \
             max_deliver is reached. Align SIE_STREAM_MAX_AGE_S across gateway + workers."
        );
    }

    // Self-heal stale durables left behind by a prior bundle deploy on
    // this pool. Without this, NATS rejects our consumer create with
    // `consumer filter subject overlaps with X` and the worker
    // CrashLoops until an operator runs `nats consumer rm` by hand.
    // Concretely: a previous deploy may leave `<old-bundle>_l4`
    // filtering on `sie.work.*.l4` while the new `<new-bundle>_l4`
    // wants the same filter.
    cleanup_overlapping_consumers(&stream, &consumer_name, &subject).await;

    let consumer_cfg = ConsumerConfig {
        durable_name: Some(consumer_name.clone()),
        filter_subject: subject.clone(),
        ack_policy: AckPolicy::Explicit,
        ack_wait: desired_ack_wait,
        max_deliver: desired_max_deliver,
        max_ack_pending: desired_max_ack_pending,
        ..Default::default()
    };

    let mut consumer: PullConsumer = stream
        .get_or_create_consumer(&consumer_name, consumer_cfg)
        .await
        .map_err(|e| NatsSetupError::EnsureConsumer {
            name: consumer_name.clone(),
            source: e.into(),
        })?;

    // Same drift warning for the durable consumer. `get_or_create_consumer`
    // inherits the existing durable's config; new tuning here is silently
    // dropped until someone deletes the consumer.
    if let Ok(info) = consumer.info().await {
        if info.config.max_deliver != desired_max_deliver
            || info.config.ack_wait != desired_ack_wait
            || info.config.max_ack_pending != desired_max_ack_pending
        {
            warn!(
                consumer = %consumer_name,
                observed_max_deliver = info.config.max_deliver,
                desired_max_deliver,
                observed_ack_wait_s = info.config.ack_wait.as_secs(),
                desired_ack_wait_s = desired_ack_wait.as_secs(),
                observed_max_ack_pending = info.config.max_ack_pending,
                desired_max_ack_pending,
                "durable consumer config drift: existing durable does not match this worker's \
                 intended tuning. Delete the consumer or update the owning provisioner."
            );
        }
    }

    if log_success_at_info {
        info!(
            stream = %stream_name,
            consumer = %consumer_name,
            subject = %subject,
            "ensured NATS stream and pull consumer"
        );
    } else {
        debug!(
            stream = %stream_name,
            consumer = %consumer_name,
            subject = %subject,
            "reconciled NATS stream and pull consumer"
        );
    }
    Ok(consumer)
}

/// Lightweight wrapper around a pull consumer stream.
pub struct NatsConsumer {
    consumer: PullConsumer,
}

impl NatsConsumer {
    pub fn new(consumer: PullConsumer) -> Self {
        Self { consumer }
    }

    /// Begin a message stream with the given per-fetch batch size and
    /// expiry. Callers drive it via `futures_util::StreamExt::next` in the
    /// dispatcher loop.
    ///
    /// **Important:** the returned [`PullStream`] is designed to be held
    /// open for the process lifetime. Dropping & recreating it per tick
    /// is pathological — any messages the server has already routed to a
    /// not-yet-drained stream become `ack_pending` until `ack_wait`
    /// elapses (30 s by default), producing a catastrophic throughput
    /// collapse. Callers therefore construct one stream at startup and poll it
    /// with `.next().await` for as long as the worker runs.
    pub async fn messages(
        &self,
        batch: usize,
        expires: Duration,
    ) -> Result<PullStream, async_nats::Error> {
        let stream = self
            .consumer
            .stream()
            .max_messages_per_batch(batch)
            .expires(expires)
            .messages()
            .await?;
        Ok(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::WorkerConfig;
    use std::path::PathBuf;

    #[test]
    fn consumer_config_matches_shared_defaults() {
        // Shared defaults (env-overridable where documented) across all
        // consumer creators in the pool:
        //   ack_wait = 30s
        //   max_deliver = 20            (env SIE_MAX_DELIVER)
        //   max_ack_pending = 1000      (env SIE_MAX_ACK_PENDING)
        //   stream max_age = 600s       (env SIE_STREAM_MAX_AGE_S)
        //   stream max_msgs = 100_000
        assert_eq!(ACK_WAIT_SECS, 30);
        assert_eq!(DEFAULT_MAX_DELIVER, 20);
        assert_eq!(DEFAULT_MAX_ACK_PENDING, 1000);
        assert_eq!(DEFAULT_STREAM_MAX_AGE_SECS, 600);
        assert_eq!(STREAM_MAX_MSGS, 100_000);
    }

    #[test]
    fn env_free_accessors_match_defaults() {
        // Env-free calls must return the hardcoded defaults. We do not
        // touch the environment (would be racy with other tests) — this
        // just guards against accidental typos in the env-var key.
        // If a developer happens to set these env vars locally, the
        // assertion may legitimately differ; skip-gate on that case.
        if std::env::var("SIE_MAX_DELIVER").is_err() {
            assert_eq!(max_deliver(), DEFAULT_MAX_DELIVER);
        }
        if std::env::var("SIE_STREAM_MAX_AGE_S").is_err() {
            assert_eq!(stream_max_age_secs(), DEFAULT_STREAM_MAX_AGE_SECS);
        }
        if std::env::var("SIE_MAX_ACK_PENDING").is_err() {
            assert_eq!(max_ack_pending(), DEFAULT_MAX_ACK_PENDING);
        }
    }

    #[test]
    fn config_produces_expected_names() {
        let cfg = WorkerConfig {
            nats_url: String::new(),
            pool: "default".into(),
            bundle: "b".into(),
            ipc_socket_path: PathBuf::from("/tmp/s.sock"),
            ipc_pool_size: 1,
            payload_store_url: None,
            gateway_url: None,
            gateway_api_key: None,
            pool_admission_enabled: true,
            pool_admission_check_interval_ms: 5_000,
            pool_admission_pause_ms: 1_000,
            pool_admission_stale_after_ms: 30_000,
            metrics_port: 9095,
            worker_id: "w".into(),
            ping_interval_ms: 2000,
            ready_stale_mult: 3,
            machine_profile: "default".into(),
            gpu_count: 1,
            bundle_config_hash: String::new(),
            config_service_url: None,
            config_service_token: None,
            config_poll_interval_ms: 30_000,
            config_full_export_interval_ms: 300_000,
            nats_config_trusted_producers: vec!["sie-config".into()],
            health_publish_interval_ms: 5_000,
        };
        assert_eq!(cfg.stream_name(), "WORK_POOL_default");
        assert_eq!(cfg.consumer_name(), "b_default");
        assert_eq!(cfg.subject_filter(), "sie.work.*.default");
    }

    // Connect/ensure tests need a live NATS and are covered by integration
    // smoke tests. Here we just ensure the code compiles and the surface is usable.
    #[allow(dead_code)]
    fn _compile_check(_js: &JsContext, _cfg: &WorkerConfig) {}

    // ----- legacy per-model stream cleanup predicate --------------------------

    #[test]
    fn legacy_stream_cleanup_targets_overlapping_per_model_streams() {
        let subjects = vec!["sie.work.BAAI__bge-m3.l4".to_string()];
        assert!(legacy_stream_overlaps_pool(
            "WORK_BAAI__bge-m3",
            &subjects,
            "WORK_POOL_l4",
            "sie.work.*.l4"
        ));
        assert!(legacy_stream_is_safe_to_delete(
            "WORK_BAAI__bge-m3",
            &subjects,
            "WORK_POOL_l4",
            "sie.work.*.l4",
            0
        ));
    }

    #[test]
    fn legacy_stream_cleanup_does_not_auto_delete_non_empty_streams() {
        let subjects = vec!["sie.work.BAAI__bge-m3.l4".to_string()];
        assert!(!legacy_stream_is_safe_to_delete(
            "WORK_BAAI__bge-m3",
            &subjects,
            "WORK_POOL_l4",
            "sie.work.*.l4",
            1
        ));
    }

    #[test]
    fn legacy_stream_cleanup_ignores_pool_streams() {
        let subjects = vec!["sie.work.*.l4".to_string()];
        assert!(!legacy_stream_overlaps_pool(
            "WORK_POOL_l4",
            &subjects,
            "WORK_POOL_l4",
            "sie.work.*.l4"
        ));
        assert!(!legacy_stream_overlaps_pool(
            "WORK_POOL_other",
            &subjects,
            "WORK_POOL_l4",
            "sie.work.*.l4"
        ));
    }

    #[test]
    fn legacy_stream_cleanup_ignores_non_overlapping_work_streams() {
        let subjects = vec!["sie.work.BAAI__bge-m3.h100".to_string()];
        assert!(!legacy_stream_overlaps_pool(
            "WORK_BAAI__bge-m3",
            &subjects,
            "WORK_POOL_l4",
            "sie.work.*.l4"
        ));
    }

    #[test]
    fn legacy_stream_cleanup_ignores_unrelated_stream_names() {
        let subjects = vec!["sie.work.BAAI__bge-m3.l4".to_string()];
        assert!(!legacy_stream_overlaps_pool(
            "ORDERS",
            &subjects,
            "WORK_POOL_l4",
            "sie.work.*.l4"
        ));
    }

    // ----- stale-durable predicate -------------------------------------------

    #[test]
    fn any_filter_overlaps_detects_bundle_flip_collision() {
        // Real-world case: previous deploy left a `<stale-bundle>_l4`
        // consumer with filter `sie.work.*.l4`; new deploy wants
        // `default_l4` with the same filter. The cleanup pass must
        // mark the old one as overlapping.
        let stale_filter = "sie.work.*.l4";
        let desired = "sie.work.*.l4";
        assert!(any_filter_overlaps(stale_filter, &[], desired));
    }

    #[test]
    fn any_filter_overlaps_ignores_other_pools() {
        // A `default_h100` consumer on a shared NATS account must not
        // be deleted just because we're deploying `default_l4`.
        assert!(!any_filter_overlaps(
            "sie.work.*.h100",
            &[],
            "sie.work.*.l4"
        ));
    }

    #[test]
    fn any_filter_overlaps_handles_filter_subjects_list() {
        // Newer NATS consumers can use the multi-filter
        // `filter_subjects` list (with `filter_subject` blank).
        let multi = vec!["sie.work.foo.h100".to_string(), "sie.work.*.l4".to_string()];
        assert!(any_filter_overlaps("", &multi, "sie.work.*.l4"));

        let multi_disjoint = vec![
            "sie.work.foo.h100".to_string(),
            "sie.work.*.eval-l4".to_string(),
        ];
        assert!(!any_filter_overlaps("", &multi_disjoint, "sie.work.*.l4"));
    }

    #[test]
    fn any_filter_overlaps_treats_empty_primary_as_unset() {
        // Empty `filter_subject` with empty `filter_subjects` means
        // a consumer with no filter — JetStream actually rejects
        // creating that on a WorkQueue stream, so it shouldn't
        // appear, but if it ever does we don't want to spuriously
        // delete it.
        assert!(!any_filter_overlaps("", &[], "sie.work.*.l4"));
    }
}
