//! Local-ingest server: a UDS listener speaking the sidecar dispatch
//! protocol, feeding the same dispatcher/prep/backend pipeline the NATS
//! pull loop feeds.
//!
//! Added for the broker-less sidecar lane. In this mode there is no NATS:
//! a thin handler shim forwards each `run_batch` call over this socket as
//! one `publish_work` op, and the sidecar answers with the msgpack
//! `WorkResult` array. Concurrent `publish_work` calls coalesce in the
//! sidecar's per-model scheduler exactly like concurrent NATS fetches do.
//!
//! ## Wire contract (v0.1)
//!
//! * Frame: `u32` little-endian length + msgpack map. 64 MiB cap.
//! * Request: `{"id": u64, "op": str, "body": map}`; response
//!   `{"id": u64, "ok": bool, "error": str|nil, "body": map}`.
//! * Ops: `ping` → `{}`; `publish_work` → `{results: bin}`; `cancel` →
//!   `{}` (accepted but a no-op here: the lane's one-shot batch calls
//!   are dropped by the caller side on cancel, and the gateway already
//!   discards late results — real in-flight cancellation lands with the
//!   streaming ops later).
//! * Errors are `"ExceptionType: message"` strings; a decodable request
//!   with an unknown op or bad body answers `ok=false` and keeps the
//!   connection open; an undecodable frame closes the connection.
//!
//! ## Divergences from the NATS ingest, by design
//!
//! * **No `reply_subject` enforcement** — results never touch a NATS
//!   subject; every result rides this socket back to the caller, so the
//!   `_INBOX.` anti-injection check is meaningless here.
//! * **No broker redelivery** — a dispatcher NAK surfaces as a
//!   [`LocalDeliveryEvent::Retry`] and is re-dispatched in-process with a
//!   bounded attempt budget, after which it becomes a typed error result.
//!   Same proportional stand-in the reference Python lane uses
//!   (`_NAK_MAX_ATTEMPTS`).
//! * **Worker-side admission re-check** happens here (the lane has
//!   one pool identity), mirroring the reference lane engine's
//!   `_admission_error`: rejections are typed errors, not NAKs — there is
//!   no differently-assigned worker to redeliver to.

use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context;
use serde::Deserialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, info, warn};

use crate::delivery::{Delivery, LocalDelivery, LocalDeliveryEvent};
use crate::dispatcher::Dispatcher;
use crate::shutdown::Shutdown;
use crate::work_types::{WorkItem, WorkResult};

/// Defensive ceiling on a single frame — matches the reference dispatcher
/// server's `DEFAULT_MAX_FRAME_BYTES` (over-length = protocol error →
/// connection closed).
const MAX_FRAME_BYTES: u32 = 64 * 1024 * 1024;

/// Default liveness deadline for a `publish_work` op when the request omits
/// `timeout_ms` (or sends `0`). A missing/zero value must NOT disable the
/// deadline: an unbounded wait pins the connection and leaks the detached
/// dispatch forever if a slot never settles. It therefore falls back to this
/// bounded ceiling; callers that need longer set an explicit positive
/// `timeout_ms`.
const DEFAULT_PUBLISH_WORK_TIMEOUT_MS: u64 = 300_000;

/// Redelivery budget per item: how many NAK-triggered re-dispatches one
/// item gets before its NAK becomes a typed error result. Matches the
/// Python lane's `_NAK_MAX_ATTEMPTS`.
const LOCAL_REDELIVERY_MAX_ATTEMPTS: u32 = 3;

/// Ceiling on one local retry backoff. NATS NAK delays reach 5s
/// (`SIE_NAK_DELAY_S` default); with a caller synchronously awaiting the
/// batch there is no reason to wait longer per hop. Matches the Python
/// lane's `_NAK_MAX_DELAY_S`.
const LOCAL_RETRY_MAX_DELAY_MS: u64 = 5_000;

/// Error code for the worker-side admission re-check — same string
/// the reference Python lane emits.
const POOL_ADMISSION_ERROR_CODE: &str = "pool_admission_rejected";

const OP_PING: &str = "ping";
const OP_PUBLISH_WORK: &str = "publish_work";
const OP_CANCEL: &str = "cancel";

// ---------------------------------------------------------------------------
// Wire envelopes
// ---------------------------------------------------------------------------

/// One-shot body decode covering every v0.1 op: unknown fields are
/// ignored, absent fields default, so `ping`'s `{}` and `publish_work`'s
/// full body both land here without a second msgpack pass over the
/// (potentially large) `items` bytes.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RequestBody {
    lane: String,
    #[allow(dead_code)] // wire field; op routing uses WorkItem.operation
    endpoint: String,
    #[allow(dead_code)] // wire field; the engine resolves per-item model_id
    model: String,
    #[allow(dead_code)] // wire field; engine routing is per-item
    engine: String,
    admission_pool: String,
    #[allow(dead_code)] // per-item hashes are checked by the dispatcher
    bundle_config_hash: String,
    #[allow(dead_code)] // cancel correlation — no-op in v0.1 (see module doc)
    request_id: String,
    /// Opaque `WorkParams` bytes. Every executor-relevant field is also
    /// serialized per-item on the WorkItem maps (gateway `WorkItemRef`
    /// contract), so the lane never decodes it — same as the Python lane.
    #[allow(dead_code)]
    params: serde_bytes::ByteBuf,
    items: serde_bytes::ByteBuf,
    timeout_ms: i64,
}

#[derive(Debug, Deserialize)]
struct RequestEnvelope {
    id: u64,
    op: String,
    #[serde(default)]
    body: RequestBody,
}

fn encode_response(id: u64, ok: bool, error: Option<&str>, body: rmpv::Value) -> Vec<u8> {
    let map = rmpv::Value::Map(vec![
        (rmpv::Value::from("id"), rmpv::Value::from(id)),
        (rmpv::Value::from("ok"), rmpv::Value::from(ok)),
        (
            rmpv::Value::from("error"),
            match error {
                Some(e) => rmpv::Value::from(e),
                None => rmpv::Value::Nil,
            },
        ),
        (rmpv::Value::from("body"), body),
    ]);
    let mut payload = Vec::new();
    rmpv::encode::write_value(&mut payload, &map).expect("msgpack encode to Vec cannot fail");
    let mut frame = Vec::with_capacity(4 + payload.len());
    frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    frame.extend_from_slice(&payload);
    frame
}

fn empty_body() -> rmpv::Value {
    rmpv::Value::Map(Vec::new())
}

fn results_body(results_bytes: Vec<u8>) -> rmpv::Value {
    rmpv::Value::Map(vec![(
        rmpv::Value::from("results"),
        rmpv::Value::Binary(results_bytes),
    )])
}

/// Read one length-prefixed frame; `Ok(None)` on clean EOF between frames.
async fn read_frame<R: AsyncReadExt + Unpin>(reader: &mut R) -> std::io::Result<Option<Vec<u8>>> {
    let mut header = [0u8; 4];
    match reader.read_exact(&mut header).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let len = u32::from_le_bytes(header);
    if len > MAX_FRAME_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("frame length {len} exceeds max {MAX_FRAME_BYTES}"),
        ));
    }
    // Grow the buffer as bytes actually arrive rather than pre-allocating the
    // full declared length: a client can cheaply declare up to the 64 MiB cap,
    // so an eager `vec![0u8; len]` would let a 4-byte header pin 64 MiB before
    // any body arrives (local memory amplification). `take` bounds the read to
    // the declared length; a short/truncated frame is a protocol error.
    let mut payload = Vec::new();
    let read = reader
        .take(u64::from(len))
        .read_to_end(&mut payload)
        .await?;
    if read != len as usize {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!("frame truncated: read {read} of {len} declared bytes"),
        ));
    }
    Ok(Some(payload))
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

/// Lane identity + shared handles one connection task needs.
struct IngestShared {
    dispatcher: Arc<Dispatcher>,
    shutdown: Arc<Shutdown>,
    /// Physical pool this lane serves (`SIE_POOL`), pre-normalized.
    lane_pool: String,
    worker_id: String,
}

/// Bind the local-ingest UDS and serve until shutdown. Consumes the
/// listener task; callers spawn it.
pub async fn run_local_ingest(
    socket_path: &Path,
    dispatcher: Arc<Dispatcher>,
    pool: &str,
    worker_id: &str,
    shutdown: Arc<Shutdown>,
) -> anyhow::Result<()> {
    // Unlink a stale socket file (crash leftovers) before binding — same
    // bind-time cleanup as the Python IPC/dispatcher servers.
    if socket_path.exists() {
        std::fs::remove_file(socket_path)
            .with_context(|| format!("unlink stale local socket {}", socket_path.display()))?;
    }
    if let Some(parent) = socket_path.parent() {
        let parent_existed = parent.exists();
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create socket dir {}", parent.display()))?;
        // Only tighten a directory we created — never clobber the mode of a
        // pre-existing (possibly shared, e.g. `/tmp`) parent. A dedicated
        // socket dir at 0700 denies traversal to non-owners regardless of the
        // socket node's own mode.
        if !parent_existed {
            std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700))
                .with_context(|| format!("restrict socket dir {}", parent.display()))?;
        }
    }
    let listener = UnixListener::bind(socket_path)
        .with_context(|| format!("bind local ingest socket {}", socket_path.display()))?;
    // Restrict the socket node to the owner. It is bound with the process
    // umask default (`0o777 & !umask`), so under a permissive container umask
    // (0000/0002) it would be group/world-connectable and any local process
    // could inject WorkItems onto this worker's GPU (compute theft / DoS).
    // connect(2) checks write permission on the node, so 0600 gates it.
    std::fs::set_permissions(socket_path, std::fs::Permissions::from_mode(0o600))
        .with_context(|| format!("restrict local ingest socket {}", socket_path.display()))?;
    info!(socket = %socket_path.display(), pool, "local-ingest: listening");

    let shared = Arc::new(IngestShared {
        dispatcher,
        shutdown: Arc::clone(&shutdown),
        lane_pool: normalize_pool(pool),
        worker_id: worker_id.to_string(),
    });

    loop {
        tokio::select! {
            biased;
            _ = shutdown.wait() => break,
            accepted = listener.accept() => {
                match accepted {
                    Ok((stream, _)) => {
                        let shared = Arc::clone(&shared);
                        tokio::spawn(async move {
                            handle_connection(stream, shared).await;
                        });
                    }
                    Err(e) => {
                        warn!(error = %e, "local-ingest: accept failed");
                    }
                }
            }
        }
    }
    let _ = std::fs::remove_file(socket_path);
    info!("local-ingest: listener stopped");
    Ok(())
}

async fn handle_connection(stream: UnixStream, shared: Arc<IngestShared>) {
    let (mut reader, writer) = stream.into_split();
    let writer = Arc::new(Mutex::new(writer));
    loop {
        let payload = tokio::select! {
            biased;
            _ = shared.shutdown.wait() => break,
            frame = read_frame(&mut reader) => match frame {
                Ok(Some(p)) => p,
                Ok(None) => break, // clean EOF
                Err(e) => {
                    warn!(error = %e, "local-ingest: closing connection on malformed frame");
                    break;
                }
            },
        };
        let request: RequestEnvelope = match rmp_serde::from_slice(&payload) {
            Ok(r) => r,
            Err(e) => {
                // No usable `id` to correlate an error response — close,
                // per the protocol contract.
                warn!(error = %e, "local-ingest: closing connection on malformed request envelope");
                break;
            }
        };
        // Each op runs as its own task; responses may interleave and are
        // correlated by `id` (same shape as the Python dispatcher server).
        let shared_op = Arc::clone(&shared);
        let writer_op = Arc::clone(&writer);
        tokio::spawn(async move {
            let frame = run_op(request, &shared_op).await;
            let mut w = writer_op.lock().await;
            if let Err(e) = w.write_all(&frame).await {
                debug!(error = %e, "local-ingest: response write failed (caller gone)");
            }
        });
    }
}

async fn run_op(request: RequestEnvelope, shared: &IngestShared) -> Vec<u8> {
    match request.op.as_str() {
        OP_PING => encode_response(request.id, true, None, empty_body()),
        OP_CANCEL => encode_response(request.id, true, None, empty_body()),
        OP_PUBLISH_WORK => match publish_work(request.body, shared).await {
            Ok(results_bytes) => {
                encode_response(request.id, true, None, results_body(results_bytes))
            }
            Err(e) => encode_response(request.id, false, Some(&e), empty_body()),
        },
        other => encode_response(
            request.id,
            false,
            Some(&format!("ValueError: unknown op: {other:?}")),
            empty_body(),
        ),
    }
}

// ---------------------------------------------------------------------------
// publish_work
// ---------------------------------------------------------------------------

fn normalize_pool(raw: &str) -> String {
    raw.trim().to_ascii_lowercase()
}

/// Worker-side admission re-check — the local-mode mirror of the
/// NATS decode-path `PoolAdmissionGate` check, reduced exactly like the
/// reference lane's `_admission_error`: one pool identity, no
/// assigned-logical-pool list, so "requested pool must be empty or equal
/// the lane pool" with batch meta and per-item fields required to agree.
fn admission_error(lane_pool: &str, meta_pool: &str, item_pool: &str) -> Option<String> {
    if !meta_pool.is_empty() && !item_pool.is_empty() && meta_pool != item_pool {
        return Some(format!(
            "admission_pool mismatch: batch meta={meta_pool:?} item={item_pool:?}"
        ));
    }
    for requested in [meta_pool, item_pool] {
        if !requested.is_empty() && requested != lane_pool {
            return Some(format!(
                "lane pool {lane_pool:?} does not serve admission_pool {requested:?}"
            ));
        }
    }
    None
}

fn error_result(wi: &WorkItem, worker_id: &str, code: &str, message: &str) -> WorkResult {
    WorkResult {
        work_item_id: wi.work_item_id.clone(),
        request_id: wi.request_id.clone(),
        item_index: wi.item_index,
        success: false,
        result_msgpack: Vec::new(),
        error: Some(message.to_string()),
        error_code: Some(code.to_string()),
        inference_ms: None,
        queue_ms: None,
        processing_ms: None,
        worker_id: Some(worker_id.to_string()),
        tokenization_ms: None,
        postprocessing_ms: None,
        payload_fetch_ms: None,
        units: None,
        // Local-ingest deliveries are worker-direct by construction: the
        // caller addressed this worker's socket, not a pool subject.
        worker_direct: true,
    }
}

async fn publish_work(body: RequestBody, shared: &IngestShared) -> Result<Vec<u8>, String> {
    let items: Vec<WorkItem> = rmp_serde::from_slice(&body.items)
        .map_err(|e| format!("DecodeError: items is not a msgpack WorkItem array: {e}"))?;
    let n = items.len();
    shared
        .dispatcher
        .metrics
        .messages_received_total
        .inc_by(n as u64);
    let mut results: Vec<Option<WorkResult>> = Vec::with_capacity(n);
    results.resize_with(n, || None);

    let (tx, mut rx) = mpsc::unbounded_channel::<LocalDeliveryEvent>();
    let meta_pool = normalize_pool(&body.admission_pool);
    let mut dispatchable: Vec<(WorkItem, Delivery)> = Vec::with_capacity(n);
    for (slot, wi) in items.iter().enumerate() {
        let item_pool = normalize_pool(&wi.admission_pool);
        if let Some(reason) = admission_error(&shared.lane_pool, &meta_pool, &item_pool) {
            results[slot] = Some(error_result(
                wi,
                &shared.worker_id,
                POOL_ADMISSION_ERROR_CODE,
                &reason,
            ));
            continue;
        }
        dispatchable.push((
            wi.clone(),
            Delivery::Local(LocalDelivery::new(slot, 0, tx.clone())),
        ));
    }
    // `tx` stays alive for the whole collect loop so NAK-triggered
    // re-dispatches can mint fresh senders. Loop termination is driven by
    // the `pending` count (every delivery settles as Result or Retry — the
    // dispatcher's settlement invariant, the same one the NATS path
    // ultimately backstops with ack_wait), plus timeout/shutdown.

    let mut pending = results.iter().filter(|r| r.is_none()).count();
    if pending > 0 {
        let dispatcher = Arc::clone(&shared.dispatcher);
        let batch_size = dispatchable.len();
        tokio::spawn(async move {
            dispatcher
                .dispatch_decoded(dispatchable, batch_size, Instant::now())
                .await;
        });
    }

    // A missing or non-positive `timeout_ms` must NOT disable the deadline:
    // that turns the timeout arm into `pending()` forever, pinning the
    // connection and leaking the detached dispatch if a slot never settles.
    // Fall back to a bounded default so every op is guaranteed to settle.
    let timeout_ms = if body.timeout_ms > 0 {
        body.timeout_ms as u64
    } else {
        DEFAULT_PUBLISH_WORK_TIMEOUT_MS
    };
    let deadline = tokio::time::Instant::now() + Duration::from_millis(timeout_ms);
    while pending > 0 {
        let event = tokio::select! {
            biased;
            _ = shared.shutdown.wait() => {
                // Wire lifecycle: in-flight ops answer the exact
                // string "cancelled" on shutdown.
                return Err("cancelled".to_string());
            }
            _ = tokio::time::sleep_until(deadline) => {
                return Err(format!(
                    "TimeoutError: publish_work timed out after {timeout_ms}ms (lane '{}')",
                    body.lane
                ));
            }
            event = rx.recv() => match event {
                Some(e) => e,
                // Unreachable while our own `tx` is alive; defensive break.
                None => break,
            },
        };
        match event {
            LocalDeliveryEvent::Result { slot, result } => {
                if slot < results.len() && results[slot].is_none() {
                    results[slot] = Some(*result);
                    pending -= 1;
                } else {
                    debug!(slot, "local-ingest: duplicate/out-of-range result ignored");
                }
            }
            LocalDeliveryEvent::Retry {
                slot,
                attempt,
                delay_ms,
            } => {
                if slot >= results.len() || results[slot].is_some() {
                    debug!(slot, "local-ingest: retry for settled slot ignored");
                    continue;
                }
                if attempt >= LOCAL_REDELIVERY_MAX_ATTEMPTS {
                    let wi = &items[slot];
                    warn!(
                        work_item_id = %wi.work_item_id,
                        attempts = attempt + 1,
                        "local-ingest: redelivery budget exhausted — typed error"
                    );
                    results[slot] = Some(error_result(
                        wi,
                        &shared.worker_id,
                        "inference_error",
                        &format!(
                            "worker requested redelivery (nak) for {:?} but the local-ingest \
                             lane has no broker redelivery; exhausted {LOCAL_REDELIVERY_MAX_ATTEMPTS} \
                             in-lane retries",
                            wi.work_item_id
                        ),
                    ));
                    pending -= 1;
                    continue;
                }
                // Bounded in-process redelivery: re-dispatch this single
                // item after the NAK delay (capped — the caller is
                // synchronously waiting).
                let delay = Duration::from_millis(delay_ms.min(LOCAL_RETRY_MAX_DELAY_MS));
                let wi = items[slot].clone();
                let delivery = Delivery::Local(LocalDelivery::new(slot, attempt + 1, tx.clone()));
                let dispatcher = Arc::clone(&shared.dispatcher);
                debug!(
                    work_item_id = %wi.work_item_id,
                    attempt = attempt + 1,
                    delay_ms = delay.as_millis() as u64,
                    "local-ingest: re-dispatching NAKed item"
                );
                tokio::spawn(async move {
                    tokio::time::sleep(delay).await;
                    dispatcher
                        .dispatch_decoded(vec![(wi, delivery)], 1, Instant::now())
                        .await;
                });
            }
        }
    }

    drop(tx);
    let final_results: Vec<WorkResult> = results
        .into_iter()
        .enumerate()
        .map(|(slot, r)| {
            r.unwrap_or_else(|| {
                // A delivery was dropped without settling (should not
                // happen; the dispatcher always answers). Fail typed
                // rather than hanging or omitting the slot.
                error_result(
                    &items[slot],
                    &shared.worker_id,
                    "inference_error",
                    "work item was dropped without an outcome",
                )
            })
        })
        .collect();
    rmp_serde::to_vec_named(&final_results)
        .map_err(|e| format!("EncodeError: results encode failed: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_accepts_empty_and_matching_pools() {
        assert_eq!(admission_error("default", "", ""), None);
        assert_eq!(admission_error("default", "default", ""), None);
        assert_eq!(admission_error("default", "", "default"), None);
        assert_eq!(admission_error("default", "default", "default"), None);
    }

    #[test]
    fn admission_rejects_mismatched_meta_and_item() {
        let err = admission_error("default", "a", "b").expect("mismatch rejected");
        assert!(err.contains("admission_pool mismatch"));
    }

    #[test]
    fn admission_rejects_foreign_pool() {
        let err = admission_error("default", "other", "").expect("foreign pool rejected");
        assert!(err.contains("does not serve"));
    }
}
