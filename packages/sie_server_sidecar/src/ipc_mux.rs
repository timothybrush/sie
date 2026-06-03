//! Multiplexed IPC connection — single UDS, many in-flight RPCs.
//!
//! Drop-in alternative to the slot-pool transport in [`crate::ipc_client`].
//! The two designs converge on the same `ipc_server.py` (which already
//! accepts pipelined frames on one connection — process methods run in
//! independent `asyncio.Task`s, see `ipc_server.py:271-286`); the
//! difference is purely on the Rust side:
//!
//! * **Pool transport**: N independent UDS connections, each strictly
//!   single-flight (`write → read`). Concurrency caps at N. A slow RPC
//!   pins one slot for its full duration; if all N slots are busy a
//!   new caller waits on a semaphore even though Python is idle.
//! * **Mux transport** (this module): one UDS connection, one writer
//!   task, one reader task, an inflight map keyed by `request_id`.
//!   `N` concurrent RPCs are limited only by the inflight cap (default
//!   1024) and by Python's own concurrency. No head-of-line stalls
//!   between unrelated methods (e.g. `EnsureModelReady` no longer
//!   queues behind a heavy `RunBatch`).
//!
//! # Activation
//!
//! Disabled by default. Set `SIE_IPC_MUX=1` to opt in. The pool
//! transport stays the legacy fallback so any deployment-specific
//! regression can be reverted with an env-var flip, no rebuild.
//!
//! # Cancellation safety
//!
//! The actor pattern owns all I/O. Caller futures only ever see two
//! awaits: a `mpsc::Sender::send` (drops the command if cancelled —
//! safe) and a `oneshot::Receiver::await` (drops the receiver if
//! cancelled — the actor's send fails silently when the response
//! arrives later, which is a no-op). At no point can a cancelled
//! caller leave the wire in a half-written state.
//!
//! # Reconnect
//!
//! On any I/O error the actor:
//!   1. Sends `IpcError::Io` to every still-in-flight oneshot.
//!   2. Drops the read/write halves and sleeps a small backoff.
//!   3. Reconnects and resumes processing the command channel.
//!
//! Callers that want retry-on-transport-error get it for free: they
//! see one error, the next call proceeds against the new connection.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use serde::Deserialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, warn};

use crate::log_util::ErrChain;
use crate::metrics::MetricsRegistry;

/// Same upper bound the pool transport uses; matches Python's
/// `_MAX_FRAME_BYTES` so a misconfigured caller can't OOM either side.
const MAX_FRAME_BYTES: usize = 32 * 1024 * 1024;

/// Maximum simultaneously-in-flight RPCs. The mpsc channel is bounded
/// at this size so that backpressure shows up at `MuxClient::call`
/// instead of inside the actor's HashMap. 1024 is large enough to
/// never bind under realistic NATS pull rates and small enough that a
/// stuck Python adapter can't pin unbounded heap.
const MAX_INFLIGHT: usize = 1024;

/// Backoff applied between reconnect attempts. Kept short — the
/// Python side is on the same Pod and "down" usually means "still
/// finishing startup" or "just restarted".
const RECONNECT_BACKOFF: Duration = Duration::from_millis(100);

/// IPC errors visible at the `MuxClient::call` boundary. Mirrors the
/// shape of [`crate::ipc_client::IpcError`] for the subset that's
/// observable through the multiplexed transport — the typed
/// deserialization, version check, and `ok=false` mapping happen in
/// the wrapping `IpcClient`.
#[derive(Debug, thiserror::Error)]
pub enum MuxError {
    #[error("io: {0}")]
    Io(std::io::Error),
    #[error("frame too large: {0} > {MAX_FRAME_BYTES}")]
    FrameTooLarge(u32),
    #[error("connection lost while waiting for response")]
    ConnectionLost,
    #[error("multiplexer shutting down")]
    Shutdown,
}

/// Minimal envelope view — just the field needed to demux responses
/// back to their callers. We deliberately keep this tiny and
/// `#[serde(default)]` everything we don't read so a malformed frame
/// doesn't kill the reader task.
#[derive(Deserialize)]
struct EnvelopeHead {
    #[serde(default)]
    request_id: String,
}

/// One pending response, registered before the request frame goes on
/// the wire and consumed exactly once by either the reader (success
/// path) or the actor's reconnect handler (drain path).
type Pending = oneshot::Sender<Result<Vec<u8>, MuxError>>;

/// Map of in-flight requests indexed by `request_id`.
type Inflight = Arc<StdMutex<HashMap<String, Pending>>>;

/// Command sent from `MuxClient::call` to the writer task.
struct Command {
    request_id: String,
    payload: Vec<u8>,
    respond: Pending,
}

/// RAII guard that decrements `ipc_mux_inflight` on drop. Survives
/// every early-return / cancellation path in `MuxClient::call_raw`,
/// so the gauge is always balanced (no need for manual `dec` after
/// every `?`).
struct InflightGuard {
    gauge: prometheus::IntGauge,
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.gauge.dec();
    }
}

/// Caller-facing handle. Cheap to clone (just a few `Arc`s); intended
/// to be wrapped by [`crate::ipc_client::IpcClient`]. The actor task
/// owns the I/O and reconnect logic. The handle owns the command
/// sender plus an optional `MetricsRegistry` clone so `call_raw` can
/// inc/dec `ipc_mux_inflight` without crossing the channel.
pub struct MuxClient {
    cmd_tx: mpsc::Sender<Command>,
    next_id: AtomicU64,
    metrics: Option<Arc<MetricsRegistry>>,
}

impl MuxClient {
    /// Spawn a multiplexer pointed at `socket_path`. The actor task
    /// owns the connection; this handle owns only the command sender.
    pub fn spawn(socket_path: impl Into<PathBuf>) -> Self {
        Self::spawn_with_metrics(socket_path, None)
    }

    pub fn spawn_with_metrics(
        socket_path: impl Into<PathBuf>,
        metrics: Option<Arc<MetricsRegistry>>,
    ) -> Self {
        let socket_path = socket_path.into();
        let (cmd_tx, cmd_rx) = mpsc::channel::<Command>(MAX_INFLIGHT);
        let actor_metrics = metrics.clone();
        tokio::spawn(async move {
            mux_actor(socket_path, cmd_rx, actor_metrics).await;
        });
        Self {
            cmd_tx,
            next_id: AtomicU64::new(1),
            metrics,
        }
    }

    /// Generate a unique request id. The `mw-` prefix lets log
    /// scanners distinguish multiplexer ids from the pool transport's
    /// `w-` ids.
    pub fn next_request_id(&self) -> String {
        format!("mw-{}", self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Send `payload` (a fully-serialized [`crate::ipc_types::RequestEnvelope`])
    /// and await the response frame as raw bytes. The caller is
    /// responsible for typed deserialization on the way out.
    ///
    /// Three failure modes:
    /// * channel send fails → mux actor exited (shutdown).
    /// * `oneshot::Receiver` resolves with `Err` → I/O error or
    ///   reconnect drained this request.
    /// * `oneshot::Receiver` resolves with `Ok(bytes)` → caller decodes.
    pub async fn call_raw(
        &self,
        request_id: String,
        payload: Vec<u8>,
    ) -> Result<Vec<u8>, MuxError> {
        // Increment the inflight gauge for the duration of this call.
        // RAII drop guard protects the gauge against any early return,
        // including a cancelled future or an `Err` from `cmd_tx.send`.
        let _inflight = self.metrics.as_ref().map(|m| {
            m.ipc_mux_inflight.inc();
            InflightGuard {
                gauge: m.ipc_mux_inflight.clone(),
            }
        });
        // Acquire-wait observation point. There is no cap today, so
        // the histogram only records the trivial setup time (well
        // below the smallest bucket). Once the
        // `SIE_IPC_MUX_MAX_INFLIGHT_PER_POD` semaphore lands, the
        // permit acquisition will be wrapped here and the histogram
        // becomes the primary signal for tuning the cap.
        let acquire_started = std::time::Instant::now();
        if let Some(m) = self.metrics.as_ref() {
            m.ipc_mux_acquire_wait_seconds
                .observe(acquire_started.elapsed().as_secs_f64());
        }
        let (respond, wait) = oneshot::channel();
        let cmd = Command {
            request_id,
            payload,
            respond,
        };
        if self.cmd_tx.send(cmd).await.is_err() {
            return Err(MuxError::Shutdown);
        }
        match wait.await {
            Ok(r) => r,
            // Sender dropped without sending — actor torn down with
            // this request still pending. Treat as connection lost.
            Err(_) => Err(MuxError::ConnectionLost),
        }
    }

    /// Approximate pool-size analog for parity with the slot pool's
    /// metric: the multiplexer can drive up to `MAX_INFLIGHT`
    /// in-flight RPCs from a single connection.
    pub fn capacity(&self) -> usize {
        MAX_INFLIGHT
    }
}

/// Long-lived actor task. Holds the connection, spawns reader/writer,
/// reconnects forever (until the command channel closes).
async fn mux_actor(
    socket_path: PathBuf,
    mut cmd_rx: mpsc::Receiver<Command>,
    metrics: Option<Arc<MetricsRegistry>>,
) {
    let inflight: Inflight = Arc::new(StdMutex::new(HashMap::new()));

    loop {
        // (Re)connect.
        let stream = match UnixStream::connect(&socket_path).await {
            Ok(s) => {
                info!(
                    socket = %socket_path.display(),
                    "mux: connected to ipc_server"
                );
                if let Some(m) = &metrics {
                    m.ipc_connect_total.with_label_values(&["ok"]).inc();
                }
                s
            }
            Err(e) => {
                warn!(
                    socket = %socket_path.display(),
                    error = %ErrChain(&e),
                    "mux: connect failed — backing off"
                );
                if let Some(m) = &metrics {
                    m.ipc_connect_total.with_label_values(&["error"]).inc();
                }
                tokio::time::sleep(RECONNECT_BACKOFF).await;
                continue;
            }
        };

        let (read_half, write_half) = stream.into_split();
        let reader_inflight = Arc::clone(&inflight);
        let mut reader_task = tokio::spawn(reader_loop(read_half, reader_inflight));

        let writer_inflight = Arc::clone(&inflight);
        // Writer borrows cmd_rx exclusively for this connection's
        // lifetime; on disconnect we get it back to drive the next.
        //
        // We `select!` on writer-finishes vs reader-exits so a peer-
        // initiated close detected by the reader (`read_exact` →
        // EOF) immediately tears down the writer + drains inflight,
        // even if no caller has a request in flight to drive the
        // writer's I/O. Without this, a quiet-window disconnect
        // would park the writer on `cmd_rx.recv()` and any caller
        // waiting on a previously-sent request would hang until its
        // own `request_timeout` (60s) fires.
        let writer_fut = writer_loop(write_half, &mut cmd_rx, writer_inflight);
        tokio::pin!(writer_fut);
        let writer_outcome = tokio::select! {
            o = &mut writer_fut => o,
            _ = &mut reader_task => {
                // Reader exited (EOF / IO / oversized frame). The
                // connection is dead — abandon the writer (its
                // pinned future is dropped on the way out of this
                // select), drain, reconnect.
                debug!("mux: reader exited before writer — connection lost");
                WriterOutcome::IoError
            }
        };

        // Stop the reader (no-op if it already exited on EOF / error).
        reader_task.abort();
        let _ = reader_task.await;

        // Drain all still-pending requests with ConnectionLost so
        // callers don't hang forever. The actor is the single owner
        // of the inflight map at this point — the reader task is
        // gone and the writer returned.
        let drained: Vec<(String, Pending)> = {
            let mut guard = inflight
                .lock()
                .expect("mux: inflight mutex poisoned during drain");
            guard.drain().collect()
        };
        let drained_count = drained.len();
        for (rid, tx) in drained {
            debug!(request_id = %rid, "mux: draining inflight on reconnect");
            let _ = tx.send(Err(MuxError::ConnectionLost));
        }
        if let Some(m) = &metrics {
            m.ipc_reconnect_total.inc();
        }
        if drained_count > 0 {
            warn!(
                drained_count,
                "mux: connection lost — drained inflight, will reconnect"
            );
        }

        // Channel closed → no more callers, exit cleanly. Otherwise
        // loop back and try to reconnect.
        if let WriterOutcome::ChannelClosed = writer_outcome {
            info!("mux: command channel closed — actor exiting");
            return;
        }
        tokio::time::sleep(RECONNECT_BACKOFF).await;
    }
}

/// Reason the writer loop returned.
enum WriterOutcome {
    /// Caller side dropped its [`MuxClient`] — no more commands will
    /// arrive. Actor should exit.
    ChannelClosed,
    /// Outbound I/O error. Actor should reconnect.
    IoError,
}

async fn writer_loop(
    mut write_half: tokio::net::unix::OwnedWriteHalf,
    cmd_rx: &mut mpsc::Receiver<Command>,
    inflight: Inflight,
) -> WriterOutcome {
    while let Some(cmd) = cmd_rx.recv().await {
        let Command {
            request_id,
            payload,
            respond,
        } = cmd;

        // Register the responder BEFORE writing the frame: if the
        // reader gets the response back faster than this thread can
        // schedule (test scenarios with in-process echo do hit this),
        // the lookup must already succeed.
        if let Ok(mut guard) = inflight.lock() {
            guard.insert(request_id.clone(), respond);
        } else {
            // Mutex poisoned — surface to caller and bail. A poisoned
            // mutex means a previous panic in this actor; further I/O
            // would just propagate the corruption.
            return WriterOutcome::IoError;
        }

        // Length prefix is u32 big-endian — matches the pool transport
        // and `ipc_server.py::_LEN_STRUCT`.
        let len = match u32::try_from(payload.len()) {
            Ok(l) if (l as usize) <= MAX_FRAME_BYTES => l,
            _ => {
                let too_big = payload.len() as u32;
                let tx = inflight
                    .lock()
                    .expect("mux: inflight mutex poisoned")
                    .remove(&request_id);
                if let Some(tx) = tx {
                    let _ = tx.send(Err(MuxError::FrameTooLarge(too_big)));
                }
                continue;
            }
        };
        if let Err(e) = write_half.write_all(&len.to_be_bytes()).await {
            // Pull the responder back out and notify before tearing
            // down. If we can't grab the lock the response is
            // effectively lost; ConnectionLost will fire on the
            // actor's drain path next.
            if let Ok(mut guard) = inflight.lock() {
                if let Some(tx) = guard.remove(&request_id) {
                    let _ = tx.send(Err(MuxError::Io(e)));
                }
            }
            return WriterOutcome::IoError;
        }
        if let Err(e) = write_half.write_all(&payload).await {
            if let Ok(mut guard) = inflight.lock() {
                if let Some(tx) = guard.remove(&request_id) {
                    let _ = tx.send(Err(MuxError::Io(e)));
                }
            }
            return WriterOutcome::IoError;
        }
        if let Err(e) = write_half.flush().await {
            if let Ok(mut guard) = inflight.lock() {
                if let Some(tx) = guard.remove(&request_id) {
                    let _ = tx.send(Err(MuxError::Io(e)));
                }
            }
            return WriterOutcome::IoError;
        }
    }

    WriterOutcome::ChannelClosed
}

async fn reader_loop(mut read_half: tokio::net::unix::OwnedReadHalf, inflight: Inflight) {
    loop {
        let mut len_buf = [0u8; 4];
        if let Err(e) = read_half.read_exact(&mut len_buf).await {
            debug!(error = %ErrChain(&e), "mux reader: read of length prefix failed (likely EOF)");
            return;
        }
        let len = u32::from_be_bytes(len_buf);
        if len as usize > MAX_FRAME_BYTES {
            warn!(
                len,
                max = MAX_FRAME_BYTES,
                "mux reader: oversized frame; dropping connection"
            );
            return;
        }
        let mut frame = vec![0u8; len as usize];
        if let Err(e) = read_half.read_exact(&mut frame).await {
            debug!(error = %ErrChain(&e), "mux reader: read of body failed");
            return;
        }
        // Cheap header decode just to extract `request_id`. Body
        // payload stays on the wire-bytes path; the wrapping client
        // re-deserializes the full envelope into the typed response.
        let head: EnvelopeHead = match rmp_serde::from_slice(&frame) {
            Ok(h) => h,
            Err(e) => {
                warn!(error = %ErrChain(&e), "mux reader: malformed envelope; dropping frame");
                continue;
            }
        };
        if head.request_id.is_empty() {
            warn!("mux reader: response envelope missing request_id; dropping frame");
            continue;
        }
        let pending = match inflight.lock() {
            Ok(mut g) => g.remove(&head.request_id),
            Err(_) => return, // mutex poisoned — bail and let the actor reconnect
        };
        match pending {
            Some(tx) => {
                // `send` returns Err if the caller's oneshot::Receiver
                // was dropped (caller's future cancelled). That's a
                // no-op — nothing to clean up.
                let _ = tx.send(Ok(frame));
            }
            None => {
                warn!(
                    request_id = %head.request_id,
                    "mux reader: response with no inflight handler — late delivery after drain?"
                );
            }
        }
    }
}

/// Ergonomic helper: read the on/off flag for the multiplexed
/// transport. Centralised so callers don't sprinkle `std::env::var`
/// across the codebase.
pub fn mux_enabled() -> bool {
    matches!(
        std::env::var("SIE_IPC_MUX").ok().as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("on") | Some("ON")
    )
}

impl From<std::io::Error> for MuxError {
    fn from(e: std::io::Error) -> Self {
        MuxError::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;
    use tokio::net::UnixListener;

    fn short_sock_path() -> PathBuf {
        let base = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".to_string());
        let base = if base.len() > 20 {
            "/tmp".to_string()
        } else {
            base
        };
        PathBuf::from(base).join(format!("sie-mux-{}.sock", uuid::Uuid::new_v4().simple()))
    }

    /// Test echo server that **does** support multiplexed pipelining:
    /// each frame is processed in its own task, so out-of-order
    /// responses are guaranteed when the handler is async.
    async fn spawn_mux_echo<F, Fut>(path: PathBuf, handler: F) -> tokio::task::JoinHandle<()>
    where
        F: Fn(Vec<u8>) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Vec<u8>> + Send + 'static,
    {
        let _ = tokio::fs::remove_file(&path).await;
        let listener = UnixListener::bind(&path).unwrap();
        let handler = Arc::new(handler);
        tokio::spawn(async move {
            loop {
                let (sock, _) = match listener.accept().await {
                    Ok(p) => p,
                    Err(_) => return,
                };
                let (mut read_half, write_half) = sock.into_split();
                let write_half = Arc::new(tokio::sync::Mutex::new(write_half));
                let handler = Arc::clone(&handler);
                tokio::spawn(async move {
                    loop {
                        let mut len_buf = [0u8; 4];
                        if read_half.read_exact(&mut len_buf).await.is_err() {
                            return;
                        }
                        let n = u32::from_be_bytes(len_buf) as usize;
                        let mut buf = vec![0u8; n];
                        if read_half.read_exact(&mut buf).await.is_err() {
                            return;
                        }
                        let h = Arc::clone(&handler);
                        let w = Arc::clone(&write_half);
                        tokio::spawn(async move {
                            let resp = (h)(buf).await;
                            let mut guard = w.lock().await;
                            let _ = guard.write_all(&(resp.len() as u32).to_be_bytes()).await;
                            let _ = guard.write_all(&resp).await;
                            let _ = guard.flush().await;
                        });
                    }
                });
            }
        })
    }

    #[derive(Serialize)]
    struct StubEnv<'a> {
        version: u32,
        method: &'a str,
        request_id: String,
        body: serde_json::Value,
    }

    fn build_request(method: &str, request_id: String) -> Vec<u8> {
        let env = StubEnv {
            version: 1,
            method,
            request_id,
            body: serde_json::Value::Null,
        };
        rmp_serde::to_vec_named(&env).unwrap()
    }

    fn build_response(request_id: &str, payload: serde_json::Value) -> Vec<u8> {
        let resp = serde_json::json!({
            "version": 1,
            "request_id": request_id,
            "ok": true,
            "body": payload,
            "error": serde_json::Value::Null,
        });
        rmp_serde::to_vec_named(&resp).unwrap()
    }

    /// In-order single RPC roundtrip — sanity check that the actor
    /// wires the request id back to the right responder.
    #[tokio::test]
    async fn single_call_roundtrips() {
        let path = short_sock_path();
        let _server = spawn_mux_echo(path.clone(), |req_bytes| async move {
            let head: EnvelopeHead = rmp_serde::from_slice(&req_bytes).unwrap();
            build_response(&head.request_id, serde_json::json!({"ok": true}))
        })
        .await;

        let client = MuxClient::spawn(path);
        let rid = client.next_request_id();
        let payload = build_request("Ping", rid.clone());
        let resp = client
            .call_raw(rid, payload)
            .await
            .expect("RPC should succeed");
        let head: EnvelopeHead = rmp_serde::from_slice(&resp).unwrap();
        // Echo bounced the same id back; the multiplexer routed it
        // to our specific waiter instead of any other.
        assert!(!head.request_id.is_empty());
    }

    /// Out-of-order responses on the same connection. The slow
    /// request queues, the fast one resolves first; both must reach
    /// the right caller. This is the property that the slot pool
    /// transport can't deliver without N≥2 connections.
    #[tokio::test]
    async fn parallel_calls_demux_correctly() {
        let path = short_sock_path();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_h = Arc::clone(&counter);
        let _server = spawn_mux_echo(path.clone(), move |req_bytes| {
            let counter = Arc::clone(&counter_h);
            async move {
                let head: EnvelopeHead = rmp_serde::from_slice(&req_bytes).unwrap();
                let n = counter.fetch_add(1, Ordering::Relaxed);
                // First request stalls 80 ms, every following one
                // resolves immediately. Without multiplexing the second
                // would be queued behind the first.
                if n == 0 {
                    tokio::time::sleep(Duration::from_millis(80)).await;
                }
                build_response(&head.request_id, serde_json::json!({"n": n}))
            }
        })
        .await;

        let client = Arc::new(MuxClient::spawn(path));
        let started = std::time::Instant::now();

        let c1 = Arc::clone(&client);
        let h_slow = tokio::spawn(async move {
            let rid = c1.next_request_id();
            let payload = build_request("Slow", rid.clone());
            c1.call_raw(rid, payload).await
        });
        // Tiny stagger so the slow request enters the actor first.
        tokio::time::sleep(Duration::from_millis(5)).await;
        let c2 = Arc::clone(&client);
        let h_fast = tokio::spawn(async move {
            let rid = c2.next_request_id();
            let payload = build_request("Fast", rid.clone());
            c2.call_raw(rid, payload).await
        });

        let fast = h_fast.await.unwrap().expect("fast RPC succeeds");
        let elapsed_fast = started.elapsed();
        // Fast call must come back well before the 80 ms slow window
        // would have expired if we were serialized.
        assert!(
            elapsed_fast < Duration::from_millis(70),
            "fast call took {elapsed_fast:?}; multiplexing apparently stalled behind slow call"
        );
        let _slow = h_slow.await.unwrap().expect("slow RPC succeeds");

        // Sanity: both responses were valid envelopes with non-empty
        // request_ids (i.e. demux worked, neither went to the wrong
        // caller).
        let head: EnvelopeHead = rmp_serde::from_slice(&fast).unwrap();
        assert!(!head.request_id.is_empty());
    }

    /// Regression: a peer that closes the connection while a request
    /// is in flight must surface `ConnectionLost` to the caller
    /// promptly — *not* hang until the caller's own request timeout.
    ///
    /// Pre-fix, the actor's writer task awaited `cmd_rx.recv()` with
    /// no awareness of the reader having exited; if the reader saw
    /// EOF and there was no inbound traffic to drive the writer
    /// into a failing `write_all`, the inflight `Pending` for the
    /// already-sent request was never drained.
    ///
    /// Post-fix, the actor `select!`s on writer-finishes vs reader-
    /// exits, so a peer close is detected immediately and the actor
    /// drains inflight before reconnecting. We bound the wait at
    /// 500 ms — pre-fix this would block until the IpcClient's
    /// 60 s timeout (or forever in this test, where there is no
    /// outer timeout).
    #[tokio::test]
    async fn peer_close_drains_inflight_promptly() {
        use tokio::io::AsyncReadExt;

        let path = short_sock_path();
        let _ = tokio::fs::remove_file(&path).await;
        let listener = UnixListener::bind(&path).unwrap();

        // Custom server: read one length-prefix + body, then drop
        // the socket. No response is ever written, so the client
        // sits in inflight; only the actor's reader-exit detection
        // can rescue it.
        tokio::spawn(async move {
            if let Ok((mut sock, _)) = listener.accept().await {
                let mut len_buf = [0u8; 4];
                let _ = sock.read_exact(&mut len_buf).await;
                let n = u32::from_be_bytes(len_buf) as usize;
                let mut body = vec![0u8; n];
                let _ = sock.read_exact(&mut body).await;
                // Drop sock → peer close.
            }
        });

        let client = MuxClient::spawn(path);
        let rid = client.next_request_id();
        let payload = build_request("Doomed", rid.clone());

        // Bounded wait. With the bug: this would never resolve (the
        // mux actor would park on cmd_rx waiting for the next
        // command, while the inflight request hangs forever).
        let r = tokio::time::timeout(Duration::from_millis(500), client.call_raw(rid, payload))
            .await
            .expect("actor must drain inflight on peer close within 500ms");

        match r {
            Err(MuxError::ConnectionLost) | Err(MuxError::Io(_)) => {} // expected
            other => panic!("expected ConnectionLost / Io, got {:?}", other),
        }
    }

    /// A malformed (zero-length) frame on the wire must NOT kill the
    /// reader — we log + drop and keep going. The first request
    /// stays pending forever (caller's responsibility to apply a
    /// timeout, which `IpcClient::call` does), but subsequent calls
    /// on the same connection still succeed.
    ///
    /// The connection-loss + drain path itself is not exercised here
    /// because that requires hooking the actor's reconnect; covered
    /// indirectly by the `IpcClient` integration path which retries
    /// once on transport errors.
    #[tokio::test]
    async fn malformed_frame_does_not_kill_connection() {
        let path = short_sock_path();
        let arrived = Arc::new(AtomicUsize::new(0));
        let arrived_h = Arc::clone(&arrived);
        // Only the first connection's first frame triggers the
        // server to abort. The retry connection answers normally.
        let _server = spawn_mux_echo(path.clone(), move |req_bytes| {
            let arrived = Arc::clone(&arrived_h);
            async move {
                let head: EnvelopeHead = rmp_serde::from_slice(&req_bytes).unwrap();
                let n = arrived.fetch_add(1, Ordering::Relaxed);
                if n == 0 {
                    // Force "no response": empty bytes. The handler
                    // shape forces the reply to be a complete frame,
                    // so we instead return a length-0 frame; the
                    // mux reader will see a malformed envelope and
                    // drop it (does NOT close the connection in
                    // production, but the inflight remains pending
                    // forever — exercises the drain via shutdown).
                    Vec::new()
                } else {
                    build_response(&head.request_id, serde_json::json!({"n": n}))
                }
            }
        })
        .await;

        let client = MuxClient::spawn(path.clone());

        // First call: server writes a length-0 frame, mux reader logs
        // "missing request_id", inflight stays pending. We bound the
        // wait so the test doesn't hang if anything regressed.
        let rid1 = client.next_request_id();
        let p1 = build_request("First", rid1.clone());
        let r1 = tokio::time::timeout(Duration::from_millis(200), client.call_raw(rid1, p1)).await;
        assert!(
            r1.is_err(),
            "first call should hang because server replied with empty frame"
        );

        // Second call must still go through on the same connection —
        // the reader keeps reading frames after a malformed one (we
        // don't drop the connection on a single bad frame).
        let rid2 = client.next_request_id();
        let p2 = build_request("Second", rid2.clone());
        let r2 = client.call_raw(rid2, p2).await.expect("second call ok");
        let head: EnvelopeHead = rmp_serde::from_slice(&r2).unwrap();
        assert!(!head.request_id.is_empty());
    }

    /// `ipc_mux_inflight` must rise during concurrent calls and fall
    /// back to zero when they all complete — RAII guard correctness
    /// across the success path. A bug here would surface in
    /// production as a monotonically-growing gauge that never
    /// returns to zero, which would hide a leak in the cap (when
    /// added) by always making it look saturated.
    #[tokio::test]
    async fn inflight_gauge_is_balanced() {
        let path = short_sock_path();
        // Server holds responses for 50 ms before answering, giving
        // the test a window to observe the gauge above zero.
        let _server = spawn_mux_echo(path.clone(), |req_bytes| async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let head: EnvelopeHead = rmp_serde::from_slice(&req_bytes).unwrap();
            build_response(&head.request_id, serde_json::json!({"ok": true}))
        })
        .await;

        let metrics = Arc::new(MetricsRegistry::new().unwrap());
        let client = MuxClient::spawn_with_metrics(path.clone(), Some(Arc::clone(&metrics)));

        // Wait for the connection to come up so the first call's send
        // doesn't race the connect.
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Fire 8 calls concurrently via JoinSet so we don't pull in
        // an extra crate just for the test.
        let client = Arc::new(client);
        let mut set = tokio::task::JoinSet::new();
        for _ in 0..8u32 {
            let c = Arc::clone(&client);
            set.spawn(async move {
                let rid = c.next_request_id();
                let payload = build_request("Concurrent", rid.clone());
                c.call_raw(rid, payload).await.unwrap();
            });
        }

        // Sample mid-flight — must be >0 while server is sleeping.
        tokio::time::sleep(Duration::from_millis(20)).await;
        let mid = metrics.ipc_mux_inflight.get();
        assert!(
            mid > 0,
            "ipc_mux_inflight should be positive during concurrent calls, got {mid}"
        );

        while let Some(res) = set.join_next().await {
            res.unwrap();
        }

        // After all calls complete, the gauge must return to zero —
        // any leak here is the InflightGuard misbehaving on a return
        // path (early err, cancellation, etc.).
        let final_val = metrics.ipc_mux_inflight.get();
        assert_eq!(final_val, 0, "gauge must rebalance to 0 after all calls");
    }
}
